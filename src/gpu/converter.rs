//! GPU mesh -> gaussian converter (port of `ConversionPass`).

use std::time::{Duration, Instant};

use super::scene::{mesh_bind_group_layout, GpuScene};
use super::{pipeline_layout, shader_with_common, storage_entry, GaussianBuffer, GpuContext};
use crate::types::{SourceFormat, MAX_GAUSSIANS};

/// Which bounding box drives the planar re-projection of each mesh.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BBoxMode {
    /// One box for the whole scene: uniform sampling density across meshes.
    #[default]
    Scene,
    /// Each mesh uses its own box: every mesh gets `resolution^2` texels,
    /// so small parts get denser sampling.
    PerMesh,
}

#[derive(Clone, Copy, Debug)]
pub struct ConvertSettings {
    /// Side of the square conversion render target ("sampling density").
    pub resolution: u32,
    pub bbox_mode: BBoxMode,
}

impl Default for ConvertSettings {
    fn default() -> Self {
        // Original UI default: quality 0.5 between 16 and 1024 => 520.
        Self {
            resolution: resolution_from_quality(0.5, 1024),
            bbox_mode: BBoxMode::Scene,
        }
    }
}

/// UI "Sampling density" slider -> resolution, as in the original (`minRes + q * (maxRes - minRes)`).
pub fn resolution_from_quality(quality: f32, max_res: u32) -> u32 {
    const MIN_RES: u32 = 16;
    (MIN_RES as f32 + quality.clamp(0.0, 1.0) * (max_res.saturating_sub(MIN_RES)) as f32) as u32
}

pub struct ConversionStats {
    pub gaussians: u32,
    /// Fragments generated (can exceed capacity, in which case splats were dropped).
    pub fragments: u32,
    pub capacity: u32,
    pub duration: Duration,
}

pub struct Converter {
    pipeline: wgpu::RenderPipeline,
    output_bgl: wgpu::BindGroupLayout,
    counter: wgpu::Buffer,
    target: Option<(u32, wgpu::TextureView)>,
}

const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

impl Converter {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;
        let module =
            shader_with_common(device, "convert.wgsl", include_str!("shaders/convert.wgsl"));
        let output_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("convert output bgl"),
            entries: &[
                storage_entry(0, wgpu::ShaderStages::FRAGMENT, false),
                storage_entry(1, wgpu::ShaderStages::FRAGMENT, false),
            ],
        });
        let mesh_bgl = mesh_bind_group_layout(device);
        let layout = pipeline_layout(device, "convert", &[&output_bgl, &mesh_bgl]);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("convert"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TARGET_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convert counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            output_bgl,
            counter,
            target: None,
        }
    }

    /// Capacity reserved for a conversion (same heuristic as the original).
    pub fn capacity_for(resolution: u32, mesh_count: usize, device_max: u32) -> u32 {
        let meshes = mesh_count.max(1) as u64;
        let r = resolution as u64;
        (r * r * 6 * meshes)
            .min(MAX_GAUSSIANS as u64)
            .min(device_max as u64) as u32
    }

    /// Convert `scene` into `out`. Blocks until the GPU is done so the gaussian
    /// count is known (the original reads the atomic counter back the same way).
    pub fn convert(
        &mut self,
        ctx: &GpuContext,
        scene: &GpuScene,
        settings: ConvertSettings,
        out: &mut GaussianBuffer,
    ) -> ConversionStats {
        let start = Instant::now();
        let device = &ctx.device;
        let res = settings
            .resolution
            .clamp(1, device.limits().max_texture_dimension_2d);
        let capacity = Self::capacity_for(res, scene.meshes.len(), ctx.max_gaussians());
        out.ensure_capacity(ctx, capacity);

        for mesh in &scene.meshes {
            let mut p = mesh.params;
            let bbox = match settings.bbox_mode {
                BBoxMode::Scene => scene.bbox,
                BBoxMode::PerMesh => mesh.bbox,
            };
            p.bbox_min = bbox.min.extend(0.0).to_array();
            p.bbox_max = bbox.max.extend(0.0).to_array();
            p.flags[3] = capacity;
            ctx.queue
                .write_buffer(&mesh.params_buffer, 0, bytemuck::bytes_of(&p));
        }

        if self.target.as_ref().map(|t| t.0) != Some(res) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("convert target"),
                size: wgpu::Extent3d {
                    width: res,
                    height: res,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TARGET_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.target = Some((res, tex.create_view(&Default::default())));
        }
        let target = &self.target.as_ref().unwrap().1;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("convert output"),
            layout: &self.output_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.counter.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("convert"),
        });
        enc.clear_buffer(&self.counter, 0, None);
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("convert"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            for mesh in &scene.meshes {
                pass.set_bind_group(1, &mesh.bind_group, &[]);
                pass.draw(0..mesh.vertex_count, 0..1);
            }
        }
        ctx.queue.submit([enc.finish()]);

        let fragments =
            u32::from_le_bytes(ctx.read_buffer(&self.counter, 0, 4).try_into().unwrap());
        let count = fragments.min(capacity);

        // Shrink the buffer to the actual count to release the (possibly large) reserve.
        if count > 0 && count < capacity {
            let size = count as u64 * 96;
            let compact = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gaussians"),
                size,
                usage: GaussianBuffer::USAGE,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&out.buffer, 0, &compact, 0, size);
            ctx.queue.submit([enc.finish()]);
            out.buffer = compact;
            out.capacity = count;
        }

        out.count = count;
        out.format = SourceFormat::Converted;
        out.ply_has_pbr = false;
        out.resolution = res;
        out.generation += 1;
        out.bounds = scene.bbox;
        if fragments > capacity {
            log::warn!("conversion produced {fragments} fragments but capacity is {capacity}; extra splats were dropped");
        }
        ConversionStats {
            gaussians: count,
            fragments,
            capacity,
            duration: start.elapsed(),
        }
    }
}
