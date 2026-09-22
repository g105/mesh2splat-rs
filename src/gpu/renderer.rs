//! Gaussian splat renderer (port of the original render passes):
//! depth prepass -> mesh G-buffer -> gaussian prepass -> radix sort ->
//! splat G-buffer -> point-light shadow cube -> deferred shading.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};

use super::scene::{mesh_bind_group_layout, GpuScene};
use super::sort::{RadixSorter, DRAW_ARGS_OFFSET};
use super::{
    compute_pipeline, dispatch_dims, pipeline_layout, storage_entry, texture_entry, uniform_entry,
    GaussianBuffer, GpuContext,
};
use crate::camera::{Camera, FAR_PLANE, NEAR_PLANE};
use crate::types::{RenderMode, SourceFormat};

pub const SHADOW_RESOLUTION: u32 = 1024;

const POS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const MR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Final image. Holds gamma-encoded values (like the original's default framebuffer).
pub const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Everything the UI can tweak that affects a frame.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderSettings {
    pub render_mode: RenderMode,
    /// "Gaussian Scale" slider (standard deviation multiplier for converted meshes).
    pub gaussian_std: f32,
    /// Cull opaque splats hidden behind the source mesh (converted meshes only).
    pub depth_test: bool,
    pub lighting: bool,
    pub light_intensity: f32,
    pub light_color: Vec3,
    /// Point light transform (only the translation is used).
    pub light_transform: Mat4,
    /// Transform applied to the splats (and the mesh).
    pub model_transform: Mat4,
    pub background: [f32; 4],
    /// Mesh on the left of the split, splats on the right.
    pub split_screen: bool,
    /// 0 = all mesh, 1 = all splats.
    pub split_position: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            render_mode: RenderMode::Final,
            gaussian_std: 0.65,
            depth_test: false,
            lighting: false,
            light_intensity: 10.0,
            light_color: Vec3::ONE,
            light_transform: Mat4::from_translation(Vec3::new(1.5, 1.5, 1.5)),
            model_transform: Mat4::IDENTITY,
            background: [0.0, 0.0, 0.0, 1.0],
            split_screen: false,
            split_position: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub visible_gaussians: u32,
    /// GPU time of the last completed frame (needs `TIMESTAMP_QUERY`).
    pub gpu_ms: Option<f64>,
    /// Per-stage GPU times of the splat path (needs `TIMESTAMP_QUERY` and a non-empty frame).
    pub stages: Option<StageTimes>,
}

/// GPU milliseconds spent in the main splat stages of one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTimes {
    /// Projection, culling and 2D covariance.
    pub prepass: f64,
    /// Radix sort (including the indirect-args setup).
    pub sort: f64,
    /// Splat rasterization into the G-buffer.
    pub splat: f64,
}

// Timestamp query slots.
const TS_FRAME_BEGIN: u32 = 0;
const TS_FRAME_END: u32 = 1;
const TS_PREPASS_BEGIN: u32 = 2;
const TS_PREPASS_END: u32 = 3;
const TS_SPLAT_BEGIN: u32 = 4;
const TS_SPLAT_END: u32 = 5;
const TS_COUNT: u32 = 6;

// --- uniforms --------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    world_to_view: [[f32; 4]; 4],
    view_to_clip: [[f32; 4]; 4],
    model_to_world: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    inv_model_rot: [[f32; 4]; 4],
    model_scale: [f32; 4],
    resolution: [f32; 2],
    near_far: [f32; 2],
    std_dev: f32,
    gaussian_count: u32,
    render_mode: u32,
    format: u32,
    ply_has_pbr: u32,
    depth_test: u32,
    sort_min: f32,
    sort_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowFrameUniform {
    views: [[[f32; 4]; 4]; 6],
    proj: [[f32; 4]; 4],
    model_to_world: [[f32; 4]; 4],
    inv_model_rot: [[f32; 4]; 4],
    model_scale: [f32; 4],
    light_pos: [f32; 4],
    resolution: [f32; 2],
    near_far: [f32; 2],
    std_dev: f32,
    gaussian_count: u32,
    format: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightingUniform {
    light_pos: [f32; 4],
    cam_pos: [f32; 4],
    light_color: [f32; 4],
    background: [f32; 4],
    face_view_proj: [[[f32; 4]; 4]; 6],
    far_plane: f32,
    split_x: f32,
    render_mode: u32,
    lighting: u32,
    split_enabled: u32,
    shadow_res: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawParams {
    light_pos: [f32; 4],
    far_plane: f32,
    face: u32,
    _pad: [u32; 2],
}

// --- resources -------------------------------------------------------------

struct Tex {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

fn make_tex(
    device: &wgpu::Device,
    label: &str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> Tex {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    Tex { texture, view }
}

struct GBuffer {
    pos: Tex,
    normal: Tex,
    albedo: Tex,
    mr: Tex,
}

impl GBuffer {
    fn new(device: &wgpu::Device, label: &str, size: (u32, u32)) -> Self {
        let u = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        Self {
            pos: make_tex(device, &format!("{label} position"), size, POS_FORMAT, u),
            normal: make_tex(device, &format!("{label} normal"), size, NORMAL_FORMAT, u),
            albedo: make_tex(device, &format!("{label} albedo"), size, ALBEDO_FORMAT, u),
            mr: make_tex(
                device,
                &format!("{label} metallic-roughness"),
                size,
                MR_FORMAT,
                u,
            ),
        }
    }

    fn attachments(&self) -> [Option<wgpu::RenderPassColorAttachment<'_>>; 4] {
        let att = |v| {
            Some(wgpu::RenderPassColorAttachment {
                view: v,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        [
            att(&self.pos.view),
            att(&self.normal.view),
            att(&self.albedo.view),
            att(&self.mr.view),
        ]
    }
}

struct Targets {
    size: (u32, u32),
    splat: GBuffer,
    mesh: GBuffer,
    mesh_gbuffer_depth: Tex,
    mesh_depth: Tex,
    output: Tex,
}

struct GaussianResources {
    capacity: u32,
    quads: wgpu::Buffer,
}

struct ShadowResources {
    capacity: u32,
    squads: wgpu::Buffer,
    face_slot: wgpu::Buffer,
    face_list: wgpu::Buffer,
}

struct BindGroups {
    key: (u32, u32, u64, u32, u32),
    prepass: wgpu::BindGroup,
    splat: wgpu::BindGroup,
    frame: wgpu::BindGroup,
    deferred: wgpu::BindGroup,
    shadow: Option<(wgpu::BindGroup, wgpu::BindGroup)>,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum ReadbackState {
    Idle,
    Copied,
    Pending,
}

struct StatsReadback {
    query_set: Option<wgpu::QuerySet>,
    resolve: Option<wgpu::Buffer>,
    buffer: wgpu::Buffer,
    state: ReadbackState,
    ready: Arc<AtomicBool>,
    period: f32,
    last: FrameStats,
    /// Slot holding the prepass start of the copied frame (the frame-begin slot
    /// when the prepass was the first pass), or `None` if no prepass ran.
    prepass_begin_slot: Option<u32>,
}

pub struct Renderer {
    frame_buf: wgpu::Buffer,
    shadow_frame_buf: wgpu::Buffer,
    lighting_buf: wgpu::Buffer,
    draw_params: Vec<wgpu::Buffer>,
    counters: wgpu::Buffer,
    face_counts: wgpu::Buffer,
    shadow_args: wgpu::Buffer,

    sorter: RadixSorter,

    prepass: wgpu::ComputePipeline,
    splat: wgpu::RenderPipeline,
    splat_overdraw: wgpu::RenderPipeline,
    mesh_gbuffer: wgpu::RenderPipeline,
    mesh_depth: wgpu::RenderPipeline,
    shadow_project: wgpu::ComputePipeline,
    shadow_compact: wgpu::ComputePipeline,
    shadow_write_args: wgpu::ComputePipeline,
    shadow_draw: wgpu::RenderPipeline,
    deferred: wgpu::RenderPipeline,

    prepass_bgl: wgpu::BindGroupLayout,
    splat_bgl: wgpu::BindGroupLayout,
    frame_bgl: wgpu::BindGroupLayout,
    shadow_bgl: wgpu::BindGroupLayout,
    shadow_draw_bgl: wgpu::BindGroupLayout,
    deferred_bgl: wgpu::BindGroupLayout,
    face_bgs: Vec<wgpu::BindGroup>,

    _shadow_tex: wgpu::Texture,
    shadow_layers: Vec<wgpu::TextureView>,
    shadow_array: wgpu::TextureView,

    targets: Option<Targets>,
    gres: Option<GaussianResources>,
    sres: Option<ShadowResources>,
    bind_groups: Option<BindGroups>,
    stats: StatsReadback,
}

fn blend(src: wgpu::BlendFactor, dst: wgpu::BlendFactor) -> wgpu::BlendState {
    let c = wgpu::BlendComponent {
        src_factor: src,
        dst_factor: dst,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState { color: c, alpha: c }
}

fn gbuffer_targets(blend: Option<wgpu::BlendState>) -> [Option<wgpu::ColorTargetState>; 4] {
    let t = |format| {
        Some(wgpu::ColorTargetState {
            format,
            blend,
            write_mask: wgpu::ColorWrites::ALL,
        })
    };
    [
        t(POS_FORMAT),
        t(NORMAL_FORMAT),
        t(ALBEDO_FORMAT),
        t(MR_FORMAT),
    ]
}

fn shader(device: &wgpu::Device, label: &str, src: &str, with_frame: bool) -> wgpu::ShaderModule {
    let code = if with_frame {
        format!("{}\n{}", include_str!("shaders/frame.wgsl"), src)
    } else {
        src.to_string()
    };
    super::shader_with_common(device, label, &code)
}

fn uniform_buffer(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn m4(m: Mat4) -> [[f32; 4]; 4] {
    m.to_cols_array_2d()
}

/// View matrices of the six cube faces (same orientation as the original).
/// View-depth range of the gaussians' bounds, as `(min, 65535 / (max - min))`
/// for 16-bit sort keys. Returns a zero scale (sort on 32-bit float depth)
/// when the bounds are unknown.
fn sort_depth_range(gaussians: &GaussianBuffer, model_view: Mat4) -> (f32, f32) {
    let b = gaussians.bounds;
    if !b.is_valid() {
        return (0.0, 0.0);
    }
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for i in 0..8 {
        let corner = Vec3::new(
            if i & 1 == 0 { b.min.x } else { b.max.x },
            if i & 2 == 0 { b.min.y } else { b.max.y },
            if i & 4 == 0 { b.min.z } else { b.max.z },
        );
        let depth = -model_view.transform_point3(corner).z;
        lo = lo.min(depth);
        hi = hi.max(depth);
    }
    let lo = lo.max(NEAR_PLANE);
    let hi = hi.max(lo + 1e-6);
    (lo, 65535.0 / (hi - lo))
}

pub fn cube_face_views(light: Vec3) -> [Mat4; 6] {
    [
        Mat4::look_at_rh(light, light + Vec3::X, -Vec3::Y),
        Mat4::look_at_rh(light, light - Vec3::X, -Vec3::Y),
        Mat4::look_at_rh(light, light + Vec3::Y, Vec3::Z),
        Mat4::look_at_rh(light, light - Vec3::Y, -Vec3::Z),
        Mat4::look_at_rh(light, light + Vec3::Z, -Vec3::Y),
        Mat4::look_at_rh(light, light - Vec3::Z, -Vec3::Y),
    ]
}

impl Renderer {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;
        use wgpu::ShaderStages as S;

        // --- layouts
        let prepass_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prepass bgl"),
            entries: &[
                uniform_entry(0, S::COMPUTE),
                storage_entry(1, S::COMPUTE, true),
                storage_entry(2, S::COMPUTE, false),
                storage_entry(3, S::COMPUTE, false),
                storage_entry(4, S::COMPUTE, false),
                storage_entry(5, S::COMPUTE, false),
                texture_entry(
                    6,
                    S::COMPUTE,
                    wgpu::TextureSampleType::Depth,
                    wgpu::TextureViewDimension::D2,
                ),
            ],
        });
        let splat_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat bgl"),
            entries: &[
                uniform_entry(0, S::VERTEX_FRAGMENT),
                storage_entry(1, S::VERTEX, true),
                storage_entry(2, S::VERTEX, true),
            ],
        });
        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame bgl"),
            entries: &[uniform_entry(0, S::VERTEX_FRAGMENT)],
        });
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow bgl"),
            entries: &[
                uniform_entry(0, S::COMPUTE),
                storage_entry(1, S::COMPUTE, true),
                storage_entry(2, S::COMPUTE, false),
                storage_entry(3, S::COMPUTE, false),
                storage_entry(4, S::COMPUTE, false),
                storage_entry(5, S::COMPUTE, false),
                storage_entry(6, S::COMPUTE, false),
            ],
        });
        let shadow_draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow draw bgl"),
            entries: &[
                storage_entry(0, S::VERTEX, true),
                storage_entry(1, S::VERTEX, true),
                storage_entry(2, S::VERTEX, true),
            ],
        });
        let shadow_face_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow face bgl"),
            entries: &[uniform_entry(0, S::VERTEX_FRAGMENT)],
        });
        let float_tex = |b| {
            texture_entry(
                b,
                S::FRAGMENT,
                wgpu::TextureSampleType::Float { filterable: false },
                wgpu::TextureViewDimension::D2,
            )
        };
        let deferred_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deferred bgl"),
            entries: &[
                uniform_entry(0, S::FRAGMENT),
                float_tex(1),
                float_tex(2),
                float_tex(3),
                float_tex(4),
                float_tex(5),
                float_tex(6),
                float_tex(7),
                float_tex(8),
                texture_entry(
                    9,
                    S::FRAGMENT,
                    wgpu::TextureSampleType::Depth,
                    wgpu::TextureViewDimension::D2Array,
                ),
            ],
        });
        let mesh_bgl = mesh_bind_group_layout(device);

        // --- pipelines
        let prepass_mod = shader(
            device,
            "prepass.wgsl",
            include_str!("shaders/prepass.wgsl"),
            true,
        );
        let prepass = compute_pipeline(device, "prepass", &[&prepass_bgl], &prepass_mod, "main");

        let splat_mod = shader(
            device,
            "splat.wgsl",
            include_str!("shaders/splat.wgsl"),
            true,
        );
        let splat_layout = pipeline_layout(device, "splat", &[&splat_bgl]);
        let make_splat = |b: wgpu::BlendState, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&splat_layout),
                vertex: wgpu::VertexState {
                    module: &splat_mod,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                // One 4-vertex strip per splat: a third fewer vertex invocations than a triangle list.
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &splat_mod,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &gbuffer_targets(Some(b)),
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        // Front-to-back "under" blending (Bernhard Kerbl's 3DGS tutorial, slide 25).
        let splat = make_splat(
            blend(wgpu::BlendFactor::OneMinusDstAlpha, wgpu::BlendFactor::One),
            "splat",
        );
        let splat_overdraw = make_splat(
            blend(wgpu::BlendFactor::One, wgpu::BlendFactor::One),
            "splat overdraw",
        );

        let mesh_mod = shader(device, "mesh.wgsl", include_str!("shaders/mesh.wgsl"), true);
        let mesh_layout = pipeline_layout(device, "mesh", &[&frame_bgl, &mesh_bgl]);
        let depth_state = |write| wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(write),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        };
        let mesh_gbuffer = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh gbuffer"),
            layout: Some(&mesh_layout),
            vertex: wgpu::VertexState {
                module: &mesh_mod,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_state(true)),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &mesh_mod,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &gbuffer_targets(None),
            }),
            multiview_mask: None,
            cache: None,
        });
        let mesh_depth = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh depth prepass"),
            layout: Some(&mesh_layout),
            vertex: wgpu::VertexState {
                module: &mesh_mod,
                entry_point: Some("vs_depth"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_state(true)),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });

        let shadow_mod = shader(
            device,
            "shadow.wgsl",
            include_str!("shaders/shadow.wgsl"),
            false,
        );
        let shadow_project = compute_pipeline(
            device,
            "shadow project",
            &[&shadow_bgl],
            &shadow_mod,
            "project",
        );
        let shadow_compact = compute_pipeline(
            device,
            "shadow compact",
            &[&shadow_bgl],
            &shadow_mod,
            "compact",
        );
        let shadow_write_args = compute_pipeline(
            device,
            "shadow args",
            &[&shadow_bgl],
            &shadow_mod,
            "write_args",
        );
        let shadow_draw_mod = shader(
            device,
            "shadow_draw.wgsl",
            include_str!("shaders/shadow_draw.wgsl"),
            false,
        );
        let shadow_draw_layout =
            pipeline_layout(device, "shadow draw", &[&shadow_draw_bgl, &shadow_face_bgl]);
        let shadow_draw = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow draw"),
            layout: Some(&shadow_draw_layout),
            vertex: wgpu::VertexState {
                module: &shadow_draw_mod,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_state(true)),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shadow_draw_mod,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            multiview_mask: None,
            cache: None,
        });

        let deferred_mod = shader(
            device,
            "deferred.wgsl",
            include_str!("shaders/deferred.wgsl"),
            false,
        );
        let deferred_layout = pipeline_layout(device, "deferred", &[&deferred_bgl]);
        let deferred = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("deferred"),
            layout: Some(&deferred_layout),
            vertex: wgpu::VertexState {
                module: &deferred_mod,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &deferred_mod,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: OUTPUT_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // --- buffers
        let frame_buf = uniform_buffer(device, "frame", std::mem::size_of::<FrameUniform>());
        let shadow_frame_buf = uniform_buffer(
            device,
            "shadow frame",
            std::mem::size_of::<ShadowFrameUniform>(),
        );
        let lighting_buf =
            uniform_buffer(device, "lighting", std::mem::size_of::<LightingUniform>());
        let draw_params: Vec<wgpu::Buffer> = (0..6)
            .map(|_| {
                uniform_buffer(
                    device,
                    "shadow draw params",
                    std::mem::size_of::<DrawParams>(),
                )
            })
            .collect();
        let face_bgs = draw_params
            .iter()
            .map(|b| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shadow face"),
                    layout: &shadow_face_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b.as_entire_binding(),
                    }],
                })
            })
            .collect();
        let counters = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("counters"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let face_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow face counts"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow draw args"),
            size: 6 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow cube"),
            size: wgpu::Extent3d {
                width: SHADOW_RESOLUTION,
                height: SHADOW_RESOLUTION,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_layers = (0..6)
            .map(|i| {
                shadow_tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow face"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let shadow_array = shadow_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let (query_set, resolve) = if ctx.timestamps {
            (
                Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: TS_COUNT,
                })),
                Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("timestamp resolve"),
                    size: TS_COUNT as u64 * 8,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })),
            )
        } else {
            (None, None)
        };
        let stats = StatsReadback {
            query_set,
            resolve,
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stats readback"),
                size: TS_COUNT as u64 * 8 + 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            state: ReadbackState::Idle,
            ready: Arc::new(AtomicBool::new(false)),
            period: ctx.queue.get_timestamp_period(),
            last: FrameStats::default(),
            prepass_begin_slot: None,
        };

        Self {
            frame_buf,
            shadow_frame_buf,
            lighting_buf,
            draw_params,
            counters,
            face_counts,
            shadow_args,
            sorter: RadixSorter::new(ctx),
            prepass,
            splat,
            splat_overdraw,
            mesh_gbuffer,
            mesh_depth,
            shadow_project,
            shadow_compact,
            shadow_write_args,
            shadow_draw,
            deferred,
            prepass_bgl,
            splat_bgl,
            frame_bgl,
            shadow_bgl,
            shadow_draw_bgl,
            deferred_bgl,
            face_bgs,
            _shadow_tex: shadow_tex,
            shadow_layers,
            shadow_array,
            targets: None,
            gres: None,
            sres: None,
            bind_groups: None,
            stats,
        }
    }

    /// The rendered image (gamma-encoded RGBA8).
    pub fn output(&self) -> Option<(&wgpu::Texture, &wgpu::TextureView)> {
        self.targets
            .as_ref()
            .map(|t| (&t.output.texture, &t.output.view))
    }

    pub fn stats(&self) -> FrameStats {
        self.stats.last
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        if self.targets.as_ref().map(|t| t.size) == Some(size) {
            return;
        }
        let u = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        self.targets = Some(Targets {
            size,
            splat: GBuffer::new(device, "splat", size),
            mesh: GBuffer::new(device, "mesh", size),
            mesh_gbuffer_depth: make_tex(
                device,
                "mesh gbuffer depth",
                size,
                DEPTH_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            ),
            mesh_depth: make_tex(device, "mesh depth", size, DEPTH_FORMAT, u),
            output: make_tex(
                device,
                "output",
                size,
                OUTPUT_FORMAT,
                u | wgpu::TextureUsages::COPY_SRC,
            ),
        });
        self.bind_groups = None;
    }

    fn ensure_gaussian_resources(&mut self, ctx: &GpuContext, capacity: u32) {
        let capacity = capacity.max(1);
        if self.gres.as_ref().map(|g| g.capacity) != Some(capacity) {
            self.gres = Some(GaussianResources {
                capacity,
                quads: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("quads"),
                    size: capacity as u64 * 48,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
            });
            self.sorter.ensure_capacity(ctx, capacity, &self.counters);
            self.bind_groups = None;
        }
    }

    fn ensure_shadow_resources(&mut self, device: &wgpu::Device, capacity: u32) {
        let capacity = capacity.max(1);
        if self.sres.as_ref().map(|s| s.capacity) != Some(capacity) {
            let mk = |label, size: u64| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            };
            self.sres = Some(ShadowResources {
                capacity,
                squads: mk("shadow quads", capacity as u64 * 48),
                face_slot: mk("shadow face slot", capacity as u64 * 4),
                face_list: mk("shadow face list", capacity as u64 * 4),
            });
            if let Some(bg) = &mut self.bind_groups {
                bg.shadow = None;
            }
        }
    }

    fn ensure_bind_groups(&mut self, device: &wgpu::Device, gaussians: &GaussianBuffer) {
        let t = self.targets.as_ref().unwrap();
        let g = self.gres.as_ref().unwrap();
        let key = (
            t.size.0,
            t.size.1,
            gaussians.generation,
            gaussians.capacity,
            g.capacity,
        );
        if self.bind_groups.as_ref().map(|b| b.key) != Some(key) {
            let e = |binding, resource| wgpu::BindGroupEntry { binding, resource };
            let tv = |v| wgpu::BindingResource::TextureView(v);
            let prepass = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prepass"),
                layout: &self.prepass_bgl,
                entries: &[
                    e(0, self.frame_buf.as_entire_binding()),
                    e(1, gaussians.buffer.as_entire_binding()),
                    e(2, g.quads.as_entire_binding()),
                    e(3, self.sorter.keys[0].as_entire_binding()),
                    e(4, self.sorter.vals[0].as_entire_binding()),
                    e(5, self.counters.as_entire_binding()),
                    e(6, tv(&t.mesh_depth.view)),
                ],
            });
            let splat = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("splat"),
                layout: &self.splat_bgl,
                entries: &[
                    e(0, self.frame_buf.as_entire_binding()),
                    e(1, g.quads.as_entire_binding()),
                    e(2, self.sorter.vals[0].as_entire_binding()),
                ],
            });
            let frame = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("frame"),
                layout: &self.frame_bgl,
                entries: &[e(0, self.frame_buf.as_entire_binding())],
            });
            let deferred = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("deferred"),
                layout: &self.deferred_bgl,
                entries: &[
                    e(0, self.lighting_buf.as_entire_binding()),
                    e(1, tv(&t.splat.pos.view)),
                    e(2, tv(&t.splat.normal.view)),
                    e(3, tv(&t.splat.albedo.view)),
                    e(4, tv(&t.splat.mr.view)),
                    e(5, tv(&t.mesh.pos.view)),
                    e(6, tv(&t.mesh.normal.view)),
                    e(7, tv(&t.mesh.albedo.view)),
                    e(8, tv(&t.mesh.mr.view)),
                    e(9, tv(&self.shadow_array)),
                ],
            });
            self.bind_groups = Some(BindGroups {
                key,
                prepass,
                splat,
                frame,
                deferred,
                shadow: None,
            });
        }
        if let (Some(bg), Some(s)) = (&mut self.bind_groups, &self.sres) {
            if bg.shadow.is_none() {
                let e = |binding, resource| wgpu::BindGroupEntry { binding, resource };
                let compute = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shadow"),
                    layout: &self.shadow_bgl,
                    entries: &[
                        e(0, self.shadow_frame_buf.as_entire_binding()),
                        e(1, gaussians.buffer.as_entire_binding()),
                        e(2, s.squads.as_entire_binding()),
                        e(3, s.face_slot.as_entire_binding()),
                        e(4, self.face_counts.as_entire_binding()),
                        e(5, s.face_list.as_entire_binding()),
                        e(6, self.shadow_args.as_entire_binding()),
                    ],
                });
                let draw = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shadow draw"),
                    layout: &self.shadow_draw_bgl,
                    entries: &[
                        e(0, s.squads.as_entire_binding()),
                        e(1, s.face_list.as_entire_binding()),
                        e(2, self.face_counts.as_entire_binding()),
                    ],
                });
                bg.shadow = Some((compute, draw));
            }
        }
    }

    /// Record one frame into `enc`. After submitting, call [`Renderer::after_submit`].
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        camera: &Camera,
        settings: &RenderSettings,
        gaussians: &GaussianBuffer,
        scene: Option<&GpuScene>,
        size: (u32, u32),
    ) {
        let device = &ctx.device;
        let size = (size.0.max(1), size.1.max(1));
        self.ensure_targets(device, size);
        self.ensure_gaussian_resources(ctx, gaussians.capacity);
        let lighting = settings.lighting && settings.render_mode == RenderMode::Final;
        if lighting {
            self.ensure_shadow_resources(device, gaussians.capacity);
        }
        self.ensure_bind_groups(device, gaussians);

        let count = gaussians.count;
        let converted = gaussians.format == SourceFormat::Converted;
        let scene = scene.filter(|s| !s.meshes.is_empty());
        let depth_test = settings.depth_test && converted && scene.is_some();
        let split = settings.split_screen && scene.is_some();

        // --- uniforms
        let (w, h) = (size.0 as f32, size.1 as f32);
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(w / h);
        let model = settings.model_transform;
        let inv_rot = Mat4::from_mat3(Mat3::from_mat4(model).inverse());
        let normal_matrix = model.inverse().transpose();
        let (c0, c1) = (model.x_axis.length(), model.y_axis.length());
        let model_scale = [c0, c0, c1, 0.0]; // same (quirky) choice of columns as the original
        let std_dev = gaussians.scale_multiplier(settings.gaussian_std);
        let format = gaussians.format as u32;
        let (sort_min, sort_scale) = sort_depth_range(gaussians, view * model);
        let key_bits = if sort_scale > 0.0 { 16 } else { 32 };

        let frame = FrameUniform {
            world_to_view: m4(view),
            view_to_clip: m4(proj),
            model_to_world: m4(model),
            normal_matrix: m4(normal_matrix),
            inv_model_rot: m4(inv_rot),
            model_scale,
            resolution: [w, h],
            near_far: [NEAR_PLANE, FAR_PLANE],
            std_dev,
            gaussian_count: count,
            render_mode: settings.render_mode as u32,
            format,
            ply_has_pbr: gaussians.ply_has_pbr as u32,
            depth_test: depth_test as u32,
            sort_min,
            sort_scale,
        };
        ctx.queue
            .write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&frame));

        let light_pos = settings.light_transform.w_axis.truncate();
        let face_views = cube_face_views(light_pos);
        let shadow_proj = Mat4::perspective_rh(90f32.to_radians(), 1.0, NEAR_PLANE, FAR_PLANE);
        let lighting_u = LightingUniform {
            light_pos: light_pos.extend(1.0).to_array(),
            cam_pos: camera.position.extend(1.0).to_array(),
            light_color: settings
                .light_color
                .extend(settings.light_intensity)
                .to_array(),
            background: settings.background,
            face_view_proj: face_views.map(|v| m4(shadow_proj * v)),
            far_plane: FAR_PLANE,
            split_x: settings.split_position.clamp(0.0, 1.0) * w,
            render_mode: settings.render_mode as u32,
            lighting: lighting as u32,
            split_enabled: split as u32,
            shadow_res: SHADOW_RESOLUTION,
            _pad: [0; 2],
        };
        ctx.queue
            .write_buffer(&self.lighting_buf, 0, bytemuck::bytes_of(&lighting_u));
        if lighting {
            let sf = ShadowFrameUniform {
                views: face_views.map(m4),
                proj: m4(shadow_proj),
                model_to_world: m4(model),
                inv_model_rot: m4(inv_rot),
                model_scale,
                light_pos: light_pos.extend(1.0).to_array(),
                resolution: [SHADOW_RESOLUTION as f32; 2],
                near_far: [NEAR_PLANE, FAR_PLANE],
                std_dev,
                gaussian_count: count,
                format,
                _pad: 0,
            };
            ctx.queue
                .write_buffer(&self.shadow_frame_buf, 0, bytemuck::bytes_of(&sf));
            for (f, buf) in self.draw_params.iter().enumerate() {
                let dp = DrawParams {
                    light_pos: light_pos.extend(1.0).to_array(),
                    far_plane: FAR_PLANE,
                    face: f as u32,
                    _pad: [0; 2],
                };
                ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&dp));
            }
        }

        let t = self.targets.as_ref().unwrap();
        let bg = self.bind_groups.as_ref().unwrap();
        let g = self.gres.as_ref().unwrap();
        let _ = g;

        // Timestamp at the start of the first pass and the end of the last one.
        let stats_copy = self.stats.state == ReadbackState::Idle;
        let qs = self.stats.query_set.as_ref().filter(|_| stats_copy);
        let mut first_pass = true;
        let mut begin_ts = || {
            let f = first_pass;
            first_pass = false;
            f
        };

        // --- 1. mesh depth prepass (for mesh/gaussian occlusion)
        if depth_test {
            let ts = qs
                .filter(|_| begin_ts())
                .map(|q| wgpu::RenderPassTimestampWrites {
                    query_set: q,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: None,
                });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("depth prepass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.mesh_depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.mesh_depth);
            pass.set_bind_group(0, &bg.frame, &[]);
            for mesh in scene.unwrap().meshes.iter().filter(|m| m.opaque) {
                pass.set_bind_group(1, &mesh.bind_group, &[]);
                pass.draw(0..mesh.vertex_count, 0..1);
            }
        }

        // --- 2. mesh G-buffer (split-screen comparison)
        if split {
            let ts = qs
                .filter(|_| begin_ts())
                .map(|q| wgpu::RenderPassTimestampWrites {
                    query_set: q,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: None,
                });
            let atts = t.mesh.attachments();
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mesh gbuffer"),
                color_attachments: &atts,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.mesh_gbuffer_depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.mesh_gbuffer);
            pass.set_bind_group(0, &bg.frame, &[]);
            for mesh in &scene.unwrap().meshes {
                pass.set_bind_group(1, &mesh.bind_group, &[]);
                pass.draw(0..mesh.vertex_count, 0..1);
            }
        }

        // --- 3/4. gaussian prepass + sort
        enc.clear_buffer(&self.counters, 0, None);
        let mut prepass_begin_slot = None;
        if count > 0 {
            let ts = qs.map(|q| {
                let slot = if begin_ts() { TS_FRAME_BEGIN } else { TS_PREPASS_BEGIN };
                prepass_begin_slot = Some(slot);
                wgpu::ComputePassTimestampWrites {
                    query_set: q,
                    beginning_of_pass_write_index: Some(slot),
                    end_of_pass_write_index: Some(TS_PREPASS_END),
                }
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gaussian prepass"),
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.prepass);
                pass.set_bind_group(0, &bg.prepass, &[]);
                let (x, y) = dispatch_dims(count.div_ceil(256));
                pass.dispatch_workgroups(x, y, 1);
            }
            self.sorter.encode(enc, key_bits);
        }

        // --- 5. splats -> G-buffer
        {
            // The splat pass is never first when the prepass ran, so its begin
            // slot only doubles as the frame begin for empty frames.
            let ts = qs.map(|q| wgpu::RenderPassTimestampWrites {
                query_set: q,
                beginning_of_pass_write_index: Some(if begin_ts() {
                    TS_FRAME_BEGIN
                } else {
                    TS_SPLAT_BEGIN
                }),
                end_of_pass_write_index: Some(TS_SPLAT_END),
            });
            let atts = t.splat.attachments();
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splat gbuffer"),
                color_attachments: &atts,
                depth_stencil_attachment: None,
                timestamp_writes: ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if count > 0 {
                pass.set_pipeline(if settings.render_mode == RenderMode::Overdraw {
                    &self.splat_overdraw
                } else {
                    &self.splat
                });
                pass.set_bind_group(0, &bg.splat, &[]);
                pass.draw_indirect(&self.sorter.args, DRAW_ARGS_OFFSET);
            }
        }

        // --- 6. point light shadow cube map
        if lighting {
            let (shadow_compute, shadow_draw) = bg.shadow.as_ref().expect("shadow bind groups");
            enc.clear_buffer(&self.face_counts, 0, None);
            if count > 0 {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("shadow prepass"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, shadow_compute, &[]);
                let (x, y) = dispatch_dims(count.div_ceil(256));
                pass.set_pipeline(&self.shadow_project);
                pass.dispatch_workgroups(x, y, 1);
                pass.set_pipeline(&self.shadow_compact);
                pass.dispatch_workgroups(x, y, 1);
                pass.set_pipeline(&self.shadow_write_args);
                pass.dispatch_workgroups(1, 1, 1);
            }
            for face in 0..6 {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow face"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.shadow_layers[face],
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if count > 0 {
                    pass.set_pipeline(&self.shadow_draw);
                    pass.set_bind_group(0, shadow_draw, &[]);
                    pass.set_bind_group(1, &self.face_bgs[face], &[]);
                    pass.draw_indirect(&self.shadow_args, face as u64 * 16);
                }
            }
        }

        // --- 7. deferred shading / composite
        {
            let ts = qs.map(|q| wgpu::RenderPassTimestampWrites {
                query_set: q,
                beginning_of_pass_write_index: if begin_ts() { Some(TS_FRAME_BEGIN) } else { None },
                end_of_pass_write_index: Some(TS_FRAME_END),
            });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("deferred"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.output.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.deferred);
            pass.set_bind_group(0, &bg.deferred, &[]);
            pass.draw(0..3, 0..1);
        }

        // --- stats readback (non-blocking)
        if stats_copy {
            if let (Some(q), Some(resolve)) = (&self.stats.query_set, &self.stats.resolve) {
                enc.resolve_query_set(q, 0..TS_COUNT, resolve, 0);
                enc.copy_buffer_to_buffer(resolve, 0, &self.stats.buffer, 0, TS_COUNT as u64 * 8);
            }
            enc.copy_buffer_to_buffer(&self.counters, 0, &self.stats.buffer, TS_COUNT as u64 * 8, 4);
            self.stats.state = ReadbackState::Copied;
            self.stats.prepass_begin_slot = prepass_begin_slot;
        }
    }

    /// Kick off the asynchronous stats readback for the frame just submitted and
    /// collect results of earlier frames. Never blocks.
    pub fn after_submit(&mut self, ctx: &GpuContext) {
        if self.stats.state == ReadbackState::Copied {
            let ready = self.stats.ready.clone();
            self.stats
                .buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    if r.is_ok() {
                        ready.store(true, Ordering::Release);
                    }
                });
            self.stats.state = ReadbackState::Pending;
        }
        let _ = ctx.device.poll(wgpu::PollType::Poll);
        if self.stats.state == ReadbackState::Pending
            && self.stats.ready.swap(false, Ordering::Acquire)
        {
            {
                let data = self.stats.buffer.slice(..).get_mapped_range();
                let n = TS_COUNT as usize;
                let words: &[u64] = bytemuck::cast_slice(&data[..n * 8]);
                let visible = u32::from_le_bytes(data[n * 8..n * 8 + 4].try_into().unwrap());
                let period = self.stats.period as f64;
                let span = |a: u32, b: u32| {
                    let (a, b) = (words[a as usize], words[b as usize]);
                    (b >= a).then(|| (b - a) as f64 * period / 1e6)
                };
                let timed = self.stats.query_set.is_some();
                let gpu_ms = span(TS_FRAME_BEGIN, TS_FRAME_END)
                    .filter(|&ms| timed && ms > 0.0);
                let stages = self.stats.prepass_begin_slot.filter(|_| timed).and_then(|b| {
                    Some(StageTimes {
                        prepass: span(b, TS_PREPASS_END)?,
                        sort: span(TS_PREPASS_END, TS_SPLAT_BEGIN)?,
                        splat: span(TS_SPLAT_BEGIN, TS_SPLAT_END)?,
                    })
                });
                self.stats.last = FrameStats {
                    visible_gaussians: visible,
                    gpu_ms,
                    stages,
                };
            }
            self.stats.buffer.unmap();
            self.stats.state = ReadbackState::Idle;
        }
    }

    /// Blocking readback of the output image as tightly packed RGBA8 rows.
    pub fn read_output(&self, ctx: &GpuContext) -> Option<(u32, u32, Vec<u8>)> {
        let t = self.targets.as_ref()?;
        let (w, h) = t.size;
        let row = (w * 4).div_ceil(256) * 256;
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output readback"),
            size: row as u64 * h as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &t.output.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit([enc.finish()]);
        let bytes = ctx.read_buffer(&buf, 0, row as u64 * h as u64);
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h as usize {
            out.extend_from_slice(&bytes[y * row as usize..][..(w * 4) as usize]);
        }
        Some((w, h, out))
    }

    /// Visible splat count of the last frame, read synchronously (tests / CLI).
    pub fn read_visible_count(&self, ctx: &GpuContext) -> u32 {
        u32::from_le_bytes(ctx.read_buffer(&self.counters, 0, 4).try_into().unwrap())
    }
}
