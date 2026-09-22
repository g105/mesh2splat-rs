//! wgpu implementation of the converter and the renderer.

pub mod converter;
pub mod merge;
pub mod renderer;
pub mod scene;
pub mod sort;
pub mod texture;

use std::sync::Arc;

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

pub use converter::{BBoxMode, ConvertSettings, Converter};
pub use renderer::{FrameStats, RenderSettings, Renderer, StageTimes};
pub use scene::GpuScene;

use crate::types::{BBox, GaussianVertex, SourceFormat, MAX_GAUSSIANS};

/// Device + queue pair used by all GPU code.
#[derive(Clone)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub timestamps: bool,
}

/// Limits requested from the adapter. The gaussian buffers get big
/// (96 bytes per splat, up to 7M splats), so ask for the adapter's maximum
/// buffer sizes instead of the conservative WebGPU defaults.
pub fn required_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let a = adapter.limits();
    wgpu::Limits {
        max_buffer_size: a.max_buffer_size,
        max_storage_buffer_binding_size: a.max_storage_buffer_binding_size,
        max_storage_buffers_per_shader_stage: a.max_storage_buffers_per_shader_stage.min(16),
        max_compute_workgroups_per_dimension: a.max_compute_workgroups_per_dimension,
        max_texture_dimension_2d: a.max_texture_dimension_2d,
        ..wgpu::Limits::default()
    }
}

/// Optional features we take advantage of when present.
pub fn optional_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    adapter.features() & wgpu::Features::TIMESTAMP_QUERY
}

pub fn device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
    wgpu::DeviceDescriptor {
        label: Some("mesh2splat device"),
        required_features: optional_features(adapter),
        required_limits: required_limits(adapter),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }
}

impl GpuContext {
    /// Create a context without a window (CLI conversion / offscreen rendering).
    pub fn new_headless() -> Result<Self> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .context("no suitable GPU adapter found")?;
            let (device, queue) = adapter
                .request_device(&device_descriptor(&adapter))
                .await
                .context("failed to create GPU device")?;
            Ok(Self::from_parts(&adapter, device, queue))
        })
    }

    pub fn from_parts(adapter: &wgpu::Adapter, device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let timestamps = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        Self {
            device,
            queue,
            adapter_info: adapter.get_info(),
            timestamps,
        }
    }

    /// Largest number of gaussians a single storage binding can hold on this
    /// device (96 bytes each), capped at [`MAX_GAUSSIANS`].
    pub fn max_gaussians(&self) -> u32 {
        let l = self.device.limits();
        let bytes = l.max_storage_buffer_binding_size.min(l.max_buffer_size);
        (bytes / std::mem::size_of::<GaussianVertex>() as u64).min(MAX_GAUSSIANS as u64) as u32
    }

    /// Block until all submitted work is done.
    pub fn wait_idle(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Synchronously read back `size` bytes of `buffer` starting at `offset`.
    /// `buffer` needs `COPY_SRC` usage.
    pub fn read_buffer(&self, buffer: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
        if size == 0 {
            return Vec::new();
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let (tx, rx) = flume::bounded(1);
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.wait_idle();
        rx.recv()
            .expect("map callback dropped")
            .expect("buffer map failed");
        let data = staging.slice(..).get_mapped_range().to_vec();
        staging.unmap();
        data
    }
}

/// GPU-resident gaussians plus the metadata needed to interpret them.
pub struct GaussianBuffer {
    pub buffer: wgpu::Buffer,
    /// Capacity in gaussians.
    pub capacity: u32,
    /// Number of valid gaussians.
    pub count: u32,
    pub format: SourceFormat,
    /// Only meaningful for [`SourceFormat::Ply`].
    pub ply_has_pbr: bool,
    /// Sampling resolution that produced these gaussians (converted meshes only).
    pub resolution: u32,
    /// Bumped whenever the contents change, so dependent resources can react.
    pub generation: u64,
    /// Model-space bounds of the gaussian centers (lets the renderer sort on
    /// 16-bit depth keys; [`BBox::EMPTY`] falls back to full 32-bit keys).
    pub bounds: BBox,
}

impl GaussianBuffer {
    pub const USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
        .union(wgpu::BufferUsages::COPY_SRC)
        .union(wgpu::BufferUsages::COPY_DST);

    pub fn new_empty(ctx: &GpuContext) -> Self {
        Self {
            buffer: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gaussians"),
                size: std::mem::size_of::<GaussianVertex>() as u64,
                usage: Self::USAGE,
                mapped_at_creation: false,
            }),
            capacity: 1,
            count: 0,
            format: SourceFormat::Converted,
            ply_has_pbr: false,
            resolution: 1,
            generation: 0,
            bounds: BBox::EMPTY,
        }
    }

    /// Make sure the buffer can hold `capacity` gaussians (contents are discarded on growth).
    pub fn ensure_capacity(&mut self, ctx: &GpuContext, capacity: u32) {
        let capacity = capacity.max(1);
        if capacity != self.capacity {
            self.buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gaussians"),
                size: capacity as u64 * std::mem::size_of::<GaussianVertex>() as u64,
                usage: Self::USAGE,
                mapped_at_creation: false,
            });
            self.capacity = capacity;
        }
    }

    /// Upload gaussians loaded from a PLY file.
    pub fn upload_ply(&mut self, ctx: &GpuContext, gaussians: &[GaussianVertex], has_pbr: bool) {
        let max = ctx.max_gaussians() as usize;
        let gaussians = if gaussians.len() > max {
            log::warn!(
                "PLY has {} gaussians but this GPU can hold {max}; truncating",
                gaussians.len()
            );
            &gaussians[..max]
        } else {
            gaussians
        };
        let data: &[u8] = if gaussians.is_empty() {
            &[0u8; 96]
        } else {
            bytemuck::cast_slice(gaussians)
        };
        self.buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gaussians"),
                contents: data,
                usage: Self::USAGE,
            });
        self.capacity = gaussians.len().max(1) as u32;
        self.count = gaussians.len() as u32;
        self.format = SourceFormat::Ply;
        self.ply_has_pbr = has_pbr;
        self.resolution = 1;
        self.generation += 1;
        self.bounds = BBox::EMPTY;
        for g in gaussians {
            self.bounds.grow(glam::Vec3::from_slice(&g.position[..3]));
        }
    }

    /// Download the valid gaussians.
    pub fn download(&self, ctx: &GpuContext) -> Vec<GaussianVertex> {
        let bytes = ctx.read_buffer(
            &self.buffer,
            0,
            self.count as u64 * std::mem::size_of::<GaussianVertex>() as u64,
        );
        bytemuck::cast_slice(&bytes).to_vec()
    }

    /// Multiplier that turns stored scales into world-space standard deviations.
    pub fn scale_multiplier(&self, gaussian_std: f32) -> f32 {
        match self.format {
            SourceFormat::Converted => gaussian_std / self.resolution.max(1) as f32,
            SourceFormat::Ply => 1.0,
        }
    }
}

/// Build a shader module from `common.wgsl` + `src`.
pub(crate) fn shader_with_common(
    device: &wgpu::Device,
    label: &str,
    src: &str,
) -> wgpu::ShaderModule {
    let code = format!("{}\n{}", include_str!("shaders/common.wgsl"), src);
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(code.into()),
    })
}

pub(crate) fn storage_entry(
    binding: u32,
    vis: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn uniform_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn texture_entry(
    binding: u32,
    vis: wgpu::ShaderStages,
    sample_type: wgpu::TextureSampleType,
    dim: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: dim,
            multisampled: false,
        },
        count: None,
    }
}

pub(crate) fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    layouts: &[&wgpu::BindGroupLayout],
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    let bgls: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(|l| Some(*l)).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &bgls,
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

pub(crate) fn pipeline_layout(
    device: &wgpu::Device,
    label: &str,
    layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::PipelineLayout {
    let bgls: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(|l| Some(*l)).collect();
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &bgls,
        immediate_size: 0,
    })
}

/// Split a 1D dispatch into (x, y) so it never exceeds the per-dimension limit.
pub(crate) fn dispatch_dims(groups: u32) -> (u32, u32) {
    const MAX: u32 = 65535;
    if groups <= MAX {
        (groups, 1)
    } else {
        let y = groups.div_ceil(MAX);
        (groups.div_ceil(y), y)
    }
}

pub type SharedTexture = Arc<texture::GpuTexture>;
