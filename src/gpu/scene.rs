//! GPU copy of a loaded glTF scene (vertex buffers, textures, per-mesh bind groups).

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::texture::{material_sampler, GpuTexture};
use super::{storage_entry, texture_entry, uniform_entry, GpuContext};
use crate::scene::Scene;
use crate::types::BBox;

/// Per-mesh uniform shared by the converter and the mesh passes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct MeshParams {
    pub bbox_min: [f32; 4],
    pub bbox_max: [f32; 4],
    pub base_color_factor: [f32; 4],
    /// x = has albedo map, y = has normal map, z = has metallic-roughness map,
    /// w = max gaussians (converter capacity)
    pub flags: [u32; 4],
    /// x = pass level, y = triangle count, z = max level, w = detail enabled
    pub detail: [u32; 4],
    /// x = detail tolerance, y = conversion resolution
    pub detail_tol: [f32; 4],
}

pub struct GpuMesh {
    pub name: String,
    pub vertex_count: u32,
    pub vertex_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,
    pub params: MeshParams,
    pub bbox: BBox,
    pub opaque: bool,
    pub bind_group: wgpu::BindGroup,
    /// Per-triangle sampling level (written by the detail pass).
    pub levels_buffer: wgpu::Buffer,
    pub detail_bind_group: wgpu::BindGroup,
    pub triangle_count: u32,
}

pub struct GpuScene {
    pub meshes: Vec<GpuMesh>,
    pub bbox: BBox,
    pub triangle_count: usize,
}

/// Layout of the per-mesh bind group (group 1 in `convert.wgsl` and `mesh.wgsl`).
pub fn mesh_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    use wgpu::ShaderStages as S;
    let tex = |b| {
        texture_entry(
            b,
            S::FRAGMENT,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::D2,
        )
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh bgl"),
        entries: &[
            uniform_entry(0, S::VERTEX_FRAGMENT),
            storage_entry(1, S::VERTEX, true),
            tex(2),
            tex(3),
            tex(4),
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: S::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            storage_entry(6, S::VERTEX, true),
        ],
    })
}

/// Layout of the per-mesh bind group of `detail.wgsl` (same resources, but the
/// levels are written and everything is visible to the compute stage).
pub fn detail_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    use wgpu::ShaderStages as S;
    let tex = |b| {
        texture_entry(
            b,
            S::COMPUTE,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::D2,
        )
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("detail bgl"),
        entries: &[
            uniform_entry(0, S::COMPUTE),
            storage_entry(1, S::COMPUTE, true),
            tex(2),
            tex(3),
            tex(4),
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: S::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            storage_entry(6, S::COMPUTE, false),
        ],
    })
}

impl GpuScene {
    pub fn upload(ctx: &GpuContext, scene: &Scene) -> Self {
        let device = &ctx.device;
        let layout = mesh_bind_group_layout(device);
        let detail_layout = detail_bind_group_layout(device);
        let sampler = material_sampler(device);
        let white = Arc::new(GpuTexture::white(ctx));

        // Upload each distinct image once.
        let mut cache: HashMap<usize, Arc<GpuTexture>> = HashMap::new();
        let mut upload = |t: &Option<Arc<crate::scene::TextureData>>| -> Option<Arc<GpuTexture>> {
            let t = t.as_ref()?;
            Some(
                cache
                    .entry(t.id)
                    .or_insert_with(|| Arc::new(GpuTexture::from_data(ctx, t, "material texture")))
                    .clone(),
            )
        };

        let mut meshes = Vec::with_capacity(scene.meshes.len());
        for mesh in &scene.meshes {
            let albedo = upload(&mesh.material.base_color_texture);
            let normal = upload(&mesh.material.normal_texture);
            let mr = upload(&mesh.material.metallic_roughness_texture);

            let params = MeshParams {
                bbox_min: mesh.bbox.min.extend(0.0).to_array(),
                bbox_max: mesh.bbox.max.extend(0.0).to_array(),
                base_color_factor: mesh.material.base_color_factor.to_array(),
                flags: [
                    albedo.is_some() as u32,
                    normal.is_some() as u32,
                    mr.is_some() as u32,
                    0,
                ],
                detail: [0, (mesh.vertices.len() / 3) as u32, 0, 0],
                detail_tol: [0.0; 4],
            };
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&mesh.name),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let view = |t: &Option<Arc<GpuTexture>>| t.clone().unwrap_or_else(|| white.clone());
            let (a, n, m) = (view(&albedo), view(&normal), view(&mr));
            let triangle_count = (mesh.vertices.len() / 3) as u32;
            let levels_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("triangle levels"),
                size: (triangle_count.max(1) as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mesh bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&a.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&n.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&m.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: levels_buffer.as_entire_binding(),
                    },
                ],
            });
            let detail_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("detail bind group"),
                layout: &detail_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&a.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&n.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&m.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: levels_buffer.as_entire_binding(),
                    },
                ],
            });
            meshes.push(GpuMesh {
                name: mesh.name.clone(),
                vertex_count: mesh.vertices.len() as u32,
                vertex_buffer,
                params_buffer,
                params,
                bbox: mesh.bbox,
                opaque: mesh.material.base_color_factor.w == 1.0,
                bind_group,
                levels_buffer,
                detail_bind_group,
                triangle_count,
            });
        }
        Self {
            meshes,
            bbox: scene.bbox,
            triangle_count: scene.triangle_count(),
        }
    }
}
