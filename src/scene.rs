//! glTF / GLB loading. Port of `SceneManager::parseGltfFile` & friends.
//!
//! Meshes are flattened into non-indexed triangle lists in *world space*
//! (node transforms baked in), exactly like the original.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

use crate::types::BBox;

/// Vertex layout shared by the converter, the mesh G-buffer pass and the depth
/// prepass (read through a storage buffer, "vertex pulling").
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vertex {
    /// xyz = world position, w = 1
    pub position: [f32; 4],
    /// xyz = world normal, w = 0
    pub normal: [f32; 4],
    /// xyz = tangent, w = handedness
    pub tangent: [f32; 4],
    /// xy = TEXCOORD_0, zw unused
    pub uv: [f32; 4],
}

/// Decoded RGBA8 texture (top row first, as in glTF).
#[derive(Debug)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// Stable identifier (glTF image index) used to share GPU textures.
    pub id: usize,
}

#[derive(Clone, Debug)]
pub struct Material {
    pub name: String,
    pub base_color_factor: Vec4,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub base_color_texture: Option<Arc<TextureData>>,
    pub normal_texture: Option<Arc<TextureData>>,
    pub metallic_roughness_texture: Option<Arc<TextureData>>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color_factor: Vec4::ONE,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            base_color_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Mesh {
    pub name: String,
    /// Non-indexed triangle list: `vertices.len() % 3 == 0`.
    pub vertices: Vec<Vertex>,
    pub material: Material,
    pub bbox: BBox,
    pub surface_area: f32,
}

impl Mesh {
    pub fn triangle_count(&self) -> usize {
        self.vertices.len() / 3
    }
}

#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub bbox: BBox,
}

impl Scene {
    pub fn triangle_count(&self) -> usize {
        self.meshes.iter().map(Mesh::triangle_count).sum()
    }
}

/// Load a `.glb` or `.gltf` file.
pub fn load_gltf(path: impl AsRef<Path>) -> Result<Scene> {
    let path = path.as_ref();
    let (doc, buffers, images) = gltf::import(path)
        .with_context(|| format!("failed to load glTF file {}", path.display()))?;

    // Decode every image once and share between materials.
    let mut texture_cache: HashMap<usize, Arc<TextureData>> = HashMap::new();
    let mut get_texture = |image_index: usize| -> Option<Arc<TextureData>> {
        if let Some(t) = texture_cache.get(&image_index) {
            return Some(t.clone());
        }
        let img = images.get(image_index)?;
        match to_rgba8(img) {
            Some(rgba) => {
                let t = Arc::new(TextureData {
                    width: img.width,
                    height: img.height,
                    rgba,
                    id: image_index,
                });
                texture_cache.insert(image_index, t.clone());
                Some(t)
            }
            None => {
                log::warn!(
                    "unsupported image format {:?} for image {image_index}",
                    img.format
                );
                None
            }
        }
    };

    // Collect (mesh index, world transform) by walking the default scene.
    let mut instances: Vec<(usize, Mat4)> = Vec::new();
    fn traverse(node: gltf::Node, parent: Mat4, out: &mut Vec<(usize, Mat4)>) {
        let local = Mat4::from_cols_array_2d(&node.transform().matrix());
        let world = parent * local;
        if let Some(mesh) = node.mesh() {
            out.push((mesh.index(), world));
        }
        for child in node.children() {
            traverse(child, world, out);
        }
    }
    if let Some(scene) = doc.default_scene().or_else(|| doc.scenes().next()) {
        for node in scene.nodes() {
            traverse(node, Mat4::IDENTITY, &mut instances);
        }
    }
    // Fallback: no scene graph -> every mesh with identity transform.
    if instances.is_empty() {
        instances = doc.meshes().map(|m| (m.index(), Mat4::IDENTITY)).collect();
    }

    let gltf_meshes: Vec<gltf::Mesh> = doc.meshes().collect();
    let mut meshes = Vec::new();
    let mut counter = 0usize;

    for (mesh_index, world) in instances {
        let gmesh = &gltf_meshes[mesh_index];
        let normal_matrix = Mat3::from_mat4(world).inverse().transpose();
        let linear = Mat3::from_mat4(world);

        for primitive in gmesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                log::info!(
                    "skipping non-triangle primitive ({:?}) in mesh {:?}",
                    primitive.mode(),
                    gmesh.name()
                );
                continue;
            }
            let reader = primitive.reader(|b| Some(&buffers[b.index()]));
            let Some(positions) = reader.read_positions() else {
                log::warn!(
                    "primitive in mesh {:?} has no POSITION attribute, skipping",
                    gmesh.name()
                );
                continue;
            };
            let positions: Vec<Vec3> = positions.map(Vec3::from).collect();
            let normals: Option<Vec<Vec3>> =
                reader.read_normals().map(|n| n.map(Vec3::from).collect());
            let uvs: Option<Vec<Vec2>> = reader
                .read_tex_coords(0)
                .map(|t| t.into_f32().map(Vec2::from).collect());
            let tangents: Option<Vec<Vec4>> =
                reader.read_tangents().map(|t| t.map(Vec4::from).collect());
            let indices: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            if indices.len() < 3 || !indices.len().is_multiple_of(3) {
                log::warn!(
                    "invalid index count {} in mesh {:?}, skipping",
                    indices.len(),
                    gmesh.name()
                );
                continue;
            }
            if indices.iter().any(|&i| i as usize >= positions.len()) {
                log::warn!("out of range index in mesh {:?}, skipping", gmesh.name());
                continue;
            }

            let base_name = gmesh.name().filter(|n| !n.is_empty()).unwrap_or("mesh");
            let name = format!("{base_name}_{counter}");
            counter += 1;

            let material = parse_material(&primitive.material(), &mut get_texture);

            let mut vertices = Vec::with_capacity(indices.len());
            let mut bbox = BBox::EMPTY;
            let mut area = 0.0f32;
            for tri in indices.chunks_exact(3) {
                let idx = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
                let mut pos = [Vec3::ZERO; 3];
                let mut nrm = [Vec3::ZERO; 3];
                let mut uv = [Vec2::ZERO; 3];
                let mut tan = [Vec4::ZERO; 3];
                for e in 0..3 {
                    pos[e] = world.transform_point3(positions[idx[e]]);
                    if let Some(u) = &uvs {
                        uv[e] = u[idx[e]];
                    }
                    if let Some(n) = &normals {
                        nrm[e] = (normal_matrix * n[idx[e]]).normalize_or_zero();
                    }
                }
                if normals.is_none() {
                    let face_n = (pos[1] - pos[0]).cross(pos[2] - pos[0]).normalize_or_zero();
                    nrm = [face_n; 3];
                }
                if let Some(t) = &tangents {
                    for e in 0..3 {
                        let t4 = t[idx[e]];
                        let tv = (linear * t4.truncate()).normalize_or_zero();
                        tan[e] = tv.extend(t4.w);
                    }
                } else {
                    // Per-face tangent from UV derivatives (same naive scheme as the original).
                    let dp1 = pos[1] - pos[0];
                    let dp2 = pos[2] - pos[0];
                    let duv1 = uv[1] - uv[0];
                    let duv2 = uv[2] - uv[0];
                    let mut det = duv1.x * duv2.y - duv1.y * duv2.x;
                    if det.abs() < 1e-8 {
                        det = 1.0;
                    }
                    let inv = 1.0 / det;
                    let mut t = ((dp1 * duv2.y - dp2 * duv1.y) * inv).normalize_or_zero();
                    let b = ((dp2 * duv1.x - dp1 * duv2.x) * inv).normalize_or_zero();
                    let n = dp1.cross(dp2).normalize_or_zero();
                    if t == Vec3::ZERO {
                        t = n.any_orthonormal_vector();
                    }
                    let handed = if n.cross(t).dot(b) < 0.0 { -1.0 } else { 1.0 };
                    tan = [t.extend(handed); 3];
                }
                for e in 0..3 {
                    bbox.grow(pos[e]);
                    vertices.push(Vertex {
                        position: pos[e].extend(1.0).to_array(),
                        normal: nrm[e].extend(0.0).to_array(),
                        tangent: tan[e].to_array(),
                        uv: [uv[e].x, uv[e].y, 0.0, 0.0],
                    });
                }
                area += 0.5 * (pos[1] - pos[0]).cross(pos[2] - pos[0]).length();
            }

            meshes.push(Mesh {
                name,
                vertices,
                material,
                bbox,
                surface_area: area,
            });
        }
    }

    if meshes.is_empty() {
        bail!("no triangle meshes found in {}", path.display());
    }
    let bbox = meshes.iter().fold(BBox::EMPTY, |acc, m| acc.union(&m.bbox));
    Ok(Scene { meshes, bbox })
}

fn parse_material(
    mat: &gltf::Material,
    get_texture: &mut impl FnMut(usize) -> Option<Arc<TextureData>>,
) -> Material {
    let pbr = mat.pbr_metallic_roughness();
    let tex = |info: Option<usize>, get: &mut dyn FnMut(usize) -> Option<Arc<TextureData>>| {
        info.and_then(get)
    };
    let base = pbr
        .base_color_texture()
        .map(|t| t.texture().source().index());
    let mr = pbr
        .metallic_roughness_texture()
        .map(|t| t.texture().source().index());
    let nrm = mat.normal_texture().map(|t| t.texture().source().index());
    Material {
        name: mat.name().unwrap_or_default().to_string(),
        base_color_factor: Vec4::from(pbr.base_color_factor()),
        metallic_factor: pbr.metallic_factor(),
        roughness_factor: pbr.roughness_factor(),
        base_color_texture: tex(base, get_texture),
        normal_texture: tex(nrm, get_texture),
        metallic_roughness_texture: tex(mr, get_texture),
    }
}

/// Convert any glTF image format to tightly packed RGBA8.
fn to_rgba8(img: &gltf::image::Data) -> Option<Vec<u8>> {
    use gltf::image::Format as F;
    let px = img.width as usize * img.height as usize;
    let p = &img.pixels;
    let u16_at = |i: usize| -> u8 { (u16::from_le_bytes([p[2 * i], p[2 * i + 1]]) >> 8) as u8 };
    let f32_at = |i: usize| -> u8 {
        let v = f32::from_le_bytes([p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]]);
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    };
    let mut out = vec![255u8; px * 4];
    match img.format {
        F::R8G8B8A8 => out.copy_from_slice(&p[..px * 4]),
        F::R8G8B8 => {
            for i in 0..px {
                out[4 * i..4 * i + 3].copy_from_slice(&p[3 * i..3 * i + 3]);
            }
        }
        F::R8G8 => {
            for i in 0..px {
                out[4 * i] = p[2 * i];
                out[4 * i + 1] = p[2 * i + 1];
                out[4 * i + 2] = 0;
            }
        }
        F::R8 => {
            for i in 0..px {
                out[4 * i..4 * i + 3].fill(p[i]);
            }
        }
        F::R16G16B16A16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = u16_at(i);
            }
        }
        F::R16G16B16 => {
            for i in 0..px {
                for c in 0..3 {
                    out[4 * i + c] = u16_at(3 * i + c);
                }
            }
        }
        F::R16G16 => {
            for i in 0..px {
                out[4 * i] = u16_at(2 * i);
                out[4 * i + 1] = u16_at(2 * i + 1);
                out[4 * i + 2] = 0;
            }
        }
        F::R16 => {
            for i in 0..px {
                let v = u16_at(i);
                out[4 * i..4 * i + 3].fill(v);
            }
        }
        F::R32G32B32FLOAT => {
            for i in 0..px {
                for c in 0..3 {
                    out[4 * i + c] = f32_at(3 * i + c);
                }
            }
        }
        F::R32G32B32A32FLOAT => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f32_at(i);
            }
        }
    }
    Some(out)
}
