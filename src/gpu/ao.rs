//! Bake ambient occlusion and a bent normal into each splat.
//!
//! Our ambient term is a constant fraction of albedo, which is why the inside
//! of a dense splat cloud (a groom especially) reads as a solid mass: every
//! splat receives as much ambient light as the ones on the outside. This walks
//! the cloud once and records, per splat, how buried it is and which way is
//! open — cheap at render time, since both ride in spare channels the splats
//! already carry.

use bytemuck::{Pod, Zeroable};

use super::{compute_pipeline, dispatch_dims, storage_entry, uniform_entry, GaussianBuffer, GpuContext};
use super::renderer::RenderSettings;
use crate::types::BBox;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    bbox_min: [f32; 4],
    cell: [f32; 4],
    dims: [u32; 4],
    tune: [f32; 4],
}

/// How the occlusion is measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AoSettings {
    /// Density grid side. 128 is ~8 MB and plenty for a groom.
    pub grid: u32,
    /// Directions sampled per splat.
    pub directions: u32,
    /// Steps marched along each direction.
    pub steps: u32,
    /// Step length, in grid cells.
    pub step_cells: f32,
    /// Scales the density before it becomes optical depth. The grid already
    /// holds splat area over cell volume, which is the extinction coefficient,
    /// so 1 is the physical value; lower it for a softer result.
    pub density: f32,
    /// Where rays leave a splat, in its own standard deviations along each
    /// ray. A strand is far thinner than a grid cell, so its rays start a
    /// step out whatever this is; a clump is wider, and rays starting inside
    /// it would count its own mass as occlusion.
    pub self_extent: f32,
}

impl Default for AoSettings {
    fn default() -> Self {
        Self {
            grid: 128,
            directions: 16,
            steps: 12,
            step_cells: 1.5,
            density: 1.0,
            self_extent: 3.0,
        }
    }
}

pub struct AoBaker {
    voxelize: wgpu::ComputePipeline,
    bake: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    grid: Option<(u32, wgpu::Buffer)>,
}

impl AoBaker {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;
        use wgpu::ShaderStages as S;
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ao bgl"),
            entries: &[
                uniform_entry(0, S::COMPUTE),
                storage_entry(1, S::COMPUTE, false),
                storage_entry(2, S::COMPUTE, false),
            ],
        });
        let module = super::shader_with_common(device, "ao.wgsl", include_str!("shaders/ao.wgsl"));
        Self {
            voxelize: compute_pipeline(device, "ao voxelize", &[&bgl], &module, "voxelize"),
            bake: compute_pipeline(device, "ao bake", &[&bgl], &module, "bake"),
            bgl,
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ao params"),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            grid: None,
        }
    }

    /// Write occlusion and a bent normal into every splat's spare `pbr`
    /// channels. `bounds` should cover the splats.
    pub fn bake(
        &mut self,
        ctx: &GpuContext,
        gaussians: &GaussianBuffer,
        bounds: &BBox,
        cfg: &AoSettings,
    ) {
        if gaussians.count == 0 || !bounds.is_valid() {
            return;
        }
        let device = &ctx.device;
        let grid = cfg.grid.clamp(16, 256);
        let cells = (grid as u64).pow(3);
        if self.grid.as_ref().map(|g| g.0) != Some(grid) {
            self.grid = Some((
                grid,
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("ao density grid"),
                    size: cells * 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            ));
        }
        let grid_buf = &self.grid.as_ref().unwrap().1;

        // A margin keeps splats on the boundary from sampling outside the grid.
        let size = bounds.size() * 1.1;
        let min = bounds.center() - size * 0.5;
        let cell = size / grid as f32;
        let params = Params {
            bbox_min: min.extend(0.0).to_array(),
            // Converted splats store their scale in grid units.
            cell: cell
                .extend(gaussians.scale_multiplier(RenderSettings::default().gaussian_std))
                .to_array(),
            dims: [grid, gaussians.count, cfg.steps, cfg.directions.max(1)],
            tune: [1.0, cfg.density, cfg.step_cells, cfg.self_extent],
        };
        ctx.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ao"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gaussians.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ao bake"),
        });
        enc.clear_buffer(grid_buf, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ao"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind, &[]);
            let (x, y) = dispatch_dims(gaussians.count.div_ceil(256));
            pass.set_pipeline(&self.voxelize);
            pass.dispatch_workgroups(x, y, 1);
            pass.set_pipeline(&self.bake);
            pass.dispatch_workgroups(x, y, 1);
        }
        ctx.queue.submit([enc.finish()]);
    }
}
