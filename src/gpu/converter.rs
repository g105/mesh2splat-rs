//! GPU mesh -> gaussian converter (port of `ConversionPass`).

use std::time::{Duration, Instant};

use super::merge::GpuMerger;
use super::scene::{detail_bind_group_layout, mesh_bind_group_layout, GpuScene};
use super::{
    compute_pipeline, dispatch_dims, pipeline_layout, shader_with_common, storage_entry,
    GaussianBuffer, GpuContext,
};
use crate::merge::{self, GridInfo, MergeSettings, MergeStats};
use crate::types::{GaussianVertex, SourceFormat, MAX_GAUSSIANS};
use wgpu::util::DeviceExt;

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
    /// Merge alike neighbouring splats into larger ones after conversion.
    pub merge: Option<MergeSettings>,
    /// Give triangles with little texture detail a coarser sampling grid.
    pub detail: Option<DetailSettings>,
}

/// Detail-aware sampling density.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetailSettings {
    /// Largest material difference (0..1) a coarser level may introduce.
    pub tolerance: f32,
    /// Coarsest level, as a power of two: 1 = half density, 3 = an eighth.
    pub max_level: u32,
}

impl Default for DetailSettings {
    fn default() -> Self {
        Self::from_strength(Self::DEFAULT_STRENGTH)
    }
}

impl DetailSettings {
    pub const DEFAULT_STRENGTH: f32 = 0.25;

    /// One-knob preset: 0 = only perfectly flat material, 1 = aggressive.
    pub fn from_strength(strength: f32) -> Self {
        Self {
            tolerance: 0.1 * strength.clamp(0.0, 1.0),
            max_level: 3,
        }
    }
}

impl Default for ConvertSettings {
    fn default() -> Self {
        // Original UI default: quality 0.5 between 16 and 1024 => 520.
        Self {
            resolution: resolution_from_quality(0.5, 1024),
            bbox_mode: BBoxMode::Scene,
            merge: None,
            detail: None,
        }
    }
}

/// UI "Sampling density" slider -> resolution, as in the original (`minRes + q * (maxRes - minRes)`).
pub fn resolution_from_quality(quality: f32, max_res: u32) -> u32 {
    const MIN_RES: u32 = 16;
    (MIN_RES as f32 + quality.clamp(0.0, 1.0) * (max_res.saturating_sub(MIN_RES)) as f32) as u32
}

pub struct ConversionStats {
    /// Final splat count (after merging, if enabled).
    pub gaussians: u32,
    /// Merge results, when merging was enabled.
    pub merge: Option<MergeStats>,
    /// Fragments generated (can exceed capacity, in which case splats were dropped).
    pub fragments: u32,
    pub capacity: u32,
    pub duration: Duration,
}

pub struct Converter {
    /// Merge on the GPU when enabled and supported (else on the CPU).
    pub gpu_merge: bool,
    merger: Option<GpuMerger>,
    pipeline: wgpu::RenderPipeline,
    detail_pipeline: wgpu::ComputePipeline,
    pass_levels: Vec<wgpu::Buffer>,
    output_bgl: wgpu::BindGroupLayout,
    counter: wgpu::Buffer,
    /// Render targets by level: `targets[l]` has side `resolution >> l`.
    targets: Option<(u32, Vec<wgpu::TextureView>)>,
}

const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Coarsest sampling level the converter can draw (1/8 density).
pub const MAX_DETAIL_LEVEL: u32 = 3;

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
                super::uniform_entry(2, wgpu::ShaderStages::VERTEX_FRAGMENT),
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
        let detail_module =
            shader_with_common(device, "detail.wgsl", include_str!("shaders/detail.wgsl"));
        let detail_bgl = detail_bind_group_layout(device);
        let detail_pipeline =
            compute_pipeline(device, "detail", &[&detail_bgl], &detail_module, "main");
        // One tiny uniform per sampling level; a single buffer rewritten
        // between passes would not work, as queue writes land before the
        // whole command buffer.
        let pass_levels = (0..=MAX_DETAIL_LEVEL)
            .map(|l| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("convert pass level"),
                    contents: bytemuck::cast_slice(&[l, 0u32, 0, 0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            })
            .collect();
        let counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convert counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            gpu_merge: true,
            merger: None,
            pipeline,
            detail_pipeline,
            pass_levels,
            output_bgl,
            counter,
            targets: None,
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

        for (mesh_index, mesh) in scene.meshes.iter().enumerate() {
            let mut p = mesh.params;
            let bbox = match settings.bbox_mode {
                BBoxMode::Scene => scene.bbox,
                BBoxMode::PerMesh => mesh.bbox,
            };
            // .w carries the mesh index into the gaussians (see convert.wgsl).
            p.bbox_min = bbox.min.extend(mesh_index as f32).to_array();
            p.detail = [
                0,
                mesh.triangle_count,
                settings.detail.map_or(0, |d| d.max_level.min(MAX_DETAIL_LEVEL)),
                settings.detail.is_some() as u32,
            ];
            p.detail_tol = [
                settings.detail.map_or(0.0, |d| d.tolerance),
                res as f32,
                0.0,
                0.0,
            ];
            p.bbox_max = bbox.max.extend(0.0).to_array();
            p.flags[3] = capacity;
            ctx.queue
                .write_buffer(&mesh.params_buffer, 0, bytemuck::bytes_of(&p));
        }

        let levels = settings
            .detail
            .map_or(1, |d| d.max_level.min(MAX_DETAIL_LEVEL) + 1);
        if self.targets.as_ref().map(|t| t.0) != Some(res)
            || self.targets.as_ref().is_some_and(|t| t.1.len() < levels as usize)
        {
            // One target per level; level l lands on every 2^l-th grid cell.
            let views = (0..levels)
                .map(|l| {
                    let side = (res >> l).max(1);
                    let tex = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("convert target"),
                        size: wgpu::Extent3d {
                            width: side,
                            height: side,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: TARGET_FORMAT,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[],
                    });
                    tex.create_view(&Default::default())
                })
                .collect();
            self.targets = Some((res, views));
        }
        let targets = &self.targets.as_ref().unwrap().1;

        let bind_groups: Vec<wgpu::BindGroup> = (0..levels)
            .map(|level| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.pass_levels[level as usize].as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("convert"),
        });
        enc.clear_buffer(&self.counter, 0, None);
        // Pick each triangle's sampling level from its material detail.
        if settings.detail.is_some() {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("detail"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.detail_pipeline);
            for mesh in &scene.meshes {
                if mesh.triangle_count == 0 {
                    continue;
                }
                let (x, y) = dispatch_dims(mesh.triangle_count.div_ceil(64));
                pass.set_bind_group(0, &mesh.detail_bind_group, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
        }
        for level in 0..levels {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("convert"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets[level as usize],
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
            pass.set_bind_group(0, &bind_groups[level as usize], &[]);
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
        // (Merging replaces the buffer anyway.)
        if count > 0 && count < capacity && settings.merge.is_none() {
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

        let mut merge_stats = None;
        let count = match settings.merge {
            Some(cfg) if count > 0 => {
                let merge_start = Instant::now();
                let grid = GridInfo {
                    resolution: res,
                    boxes: match settings.bbox_mode {
                        BBoxMode::Scene => vec![scene.bbox],
                        BBoxMode::PerMesh => scene.meshes.iter().map(|m| m.bbox).collect(),
                    },
                };
                if self.gpu_merge && GpuMerger::supports(&grid) && self.merger.is_none() {
                    self.merger = GpuMerger::new(ctx);
                }
                if let (true, true, Some(merger)) = (
                    self.gpu_merge,
                    GpuMerger::supports(&grid),
                    self.merger.as_mut(),
                ) {
                    let (buffer, merged, mut stats) =
                        merger.run(ctx, &out.buffer, count, &grid, &cfg);
                    ctx.wait_idle();
                    stats.duration = merge_start.elapsed();
                    stats.gpu = true;
                    out.buffer = buffer;
                    out.capacity = merged.max(1);
                    merge_stats = Some(stats);
                    merged
                } else {
                    let splats: Vec<GaussianVertex> = bytemuck::cast_slice(&ctx.read_buffer(
                        &out.buffer,
                        0,
                        count as u64 * std::mem::size_of::<GaussianVertex>() as u64,
                    ))
                    .to_vec();
                    let (merged, mut stats) = merge::merge(&splats, &grid, &cfg);
                    out.buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("gaussians"),
                        contents: bytemuck::cast_slice(&merged),
                        usage: GaussianBuffer::USAGE,
                    });
                    out.capacity = merged.len() as u32;
                    stats.duration = merge_start.elapsed();
                    merge_stats = Some(stats);
                    merged.len() as u32
                }
            }
            _ => count,
        };

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
            merge: merge_stats,
            fragments,
            capacity,
            duration: start.elapsed(),
        }
    }
}
