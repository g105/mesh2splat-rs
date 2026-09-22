//! Shared data types and constants.

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};

/// Upper bound on the number of gaussians the renderer will sort/draw.
/// Mirrors `MAX_GAUSSIANS_TO_SORT` in the original.
pub const MAX_GAUSSIANS: u32 = 7_000_000;

/// Zeroth-order spherical harmonics coefficient.
pub const SH_C0: f32 = 0.282_094_8;

/// Default metallic / roughness used when a mesh has no metallic-roughness map
/// (same defaults as the original converter).
pub const DEFAULT_METALLIC: f32 = 0.1;
pub const DEFAULT_ROUGHNESS: f32 = 0.5;

/// One gaussian as stored on the GPU. Identical layout to the original
/// `GaussianVertex` SSBO struct (6 x vec4 = 96 bytes).
///
/// * `position.xyz` – model-space mean
/// * `color`        – RGBA (gamma space), alpha = opacity
/// * `scale.xyz`    – per-axis standard deviation. For freshly converted meshes
///   this is *unscaled* (multiply by `gaussian_std / resolution`), for loaded
///   PLY files it is the real (exp'ed) scale.
/// * `normal.xyz`   – model-space normal
/// * `rotation`     – quaternion stored as (w, x, y, z)
/// * `pbr.xy`       – metallic, roughness
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GaussianVertex {
    pub position: [f32; 4],
    pub color: [f32; 4],
    pub scale: [f32; 4],
    pub normal: [f32; 4],
    pub rotation: [f32; 4],
    pub pbr: [f32; 4],
}

/// Where the gaussians currently held by the renderer came from.
/// Matches the `u_format` uniform in the original shaders.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SourceFormat {
    /// Converted from a mesh: scales must be multiplied by `std / resolution`.
    #[default]
    Converted = 0,
    /// Loaded from a 3DGS .ply file: scales are final.
    Ply = 1,
}

/// PLY export layouts (same three options as the original).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PlyFormat {
    /// Standard 3DGS layout (positions, normals, f_dc, 45 x f_rest, opacity, scale, rot).
    #[default]
    Standard,
    /// Standard layout without the 45 `f_rest` coefficients (always zero for
    /// converted meshes): ~3.6x smaller and still read by common 3DGS viewers.
    StandardSh0,
    /// Standard layout minus SH rest, plus metallic / roughness.
    Pbr,
    /// Quantised layout with 8-bit color, octahedral normals and 8-bit PBR.
    CompressedPbr,
}

impl PlyFormat {
    pub const ALL: [PlyFormat; 4] = [
        PlyFormat::Standard,
        PlyFormat::StandardSh0,
        PlyFormat::Pbr,
        PlyFormat::CompressedPbr,
    ];

    pub fn label(self) -> &'static str {
        match self {
            PlyFormat::Standard => "PLY Standard Format",
            PlyFormat::StandardSh0 => "PLY Standard, SH0 only (smaller)",
            PlyFormat::Pbr => "PLY PBR",
            PlyFormat::CompressedPbr => "PLY Compressed PBR",
        }
    }
}

/// Visualisation modes. Discriminants match `u_renderMode` in the original.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RenderMode {
    Albedo = 0,
    Depth = 1,
    Normal = 2,
    Geometry = 3,
    Overdraw = 4,
    Pbr = 5,
    #[default]
    Final = 6,
}

impl RenderMode {
    /// Order used in the UI combo box (same as the original).
    pub const UI_ORDER: [RenderMode; 7] = [
        RenderMode::Final,
        RenderMode::Albedo,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::Geometry,
        RenderMode::Overdraw,
        RenderMode::Pbr,
    ];

    pub fn label(self) -> &'static str {
        match self {
            RenderMode::Final => "Final (Shaded)",
            RenderMode::Albedo => "Albedo",
            RenderMode::Depth => "Depth",
            RenderMode::Normal => "Normals",
            RenderMode::Geometry => "Geometry",
            RenderMode::Overdraw => "Overdraw",
            RenderMode::Pbr => "PBR (metallic-roughness)",
        }
    }
}

pub fn sh_from_color(c: Vec3) -> Vec3 {
    (c - Vec3::splat(0.5)) / SH_C0
}

pub fn color_from_sh(sh: Vec3) -> Vec3 {
    sh * SH_C0 + Vec3::splat(0.5)
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Logit, with alpha clamped to (1e-7, 1 - 1e-7) so fully opaque splats get a
/// large finite value. (The original's `-log(1/(a + 1e-8) - 1)` yields `+inf`
/// for `a == 1` in f32.)
pub fn inv_sigmoid(alpha: f32) -> f32 {
    let a = (alpha as f64).clamp(1e-7, 1.0 - 1e-7);
    (a / (1.0 - a)).ln() as f32
}

/// Axis aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for BBox {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl BBox {
    pub const EMPTY: BBox = BBox {
        min: Vec3::splat(f32::MAX),
        max: Vec3::splat(f32::MIN),
    };

    pub fn grow(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub fn union(&self, o: &BBox) -> BBox {
        BBox {
            min: self.min.min(o.min),
            max: self.max.max(o.max),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }
}

/// Convenience: convert a `Vec4` to the raw array stored in [`GaussianVertex`].
pub fn v4(v: Vec4) -> [f32; 4] {
    v.to_array()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_layout_is_96_bytes() {
        assert_eq!(std::mem::size_of::<GaussianVertex>(), 96);
    }

    #[test]
    fn sh_roundtrip() {
        let c = Vec3::new(0.2, 0.5, 0.9);
        let back = color_from_sh(sh_from_color(c));
        assert!((back - c).length() < 1e-6);
    }

    #[test]
    fn sigmoid_roundtrip() {
        for a in [0.01f32, 0.3, 0.5, 0.9, 0.99] {
            assert!((sigmoid(inv_sigmoid(a)) - a).abs() < 1e-4);
        }
        assert!(inv_sigmoid(1.0).is_finite());
        assert!(sigmoid(inv_sigmoid(1.0)) > 0.999_99);
    }
}
