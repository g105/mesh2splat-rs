//! Hair grooms in Cem Yuksel's `.hair` strand format, and the strand-aligned
//! splats built from them.
//!
//! The converter samples triangles on a planar grid, which makes every splat
//! about one cell square: run a groom through it and the strand direction is
//! lost. Splats built straight from the strands instead are oriented and shaped
//! by each segment — long along the strand, thin across, flat — which is what
//! the anisotropic (hair) shading in the renderer expects.
//!
//! Models: <https://www.cemyuksel.com/research/hairmodels> (free for personal
//! and research use).

use std::path::Path;

use anyhow::{bail, Context, Result};
use glam::{Mat3, Quat, Vec3};

use crate::types::{BBox, GaussianVertex, DEFAULT_METALLIC, DEFAULT_ROUGHNESS};

/// One strand: a polyline with a colour per point.
#[derive(Clone, Debug)]
pub struct Strand {
    pub points: Vec<Vec3>,
    pub colors: Vec<Vec3>,
}

/// A loaded groom, with the file's own default strand thickness.
#[derive(Clone, Debug)]
pub struct Groom {
    pub strands: Vec<Strand>,
    /// Strand width in the groom's own units.
    pub thickness: f32,
}

/// How to turn strands into splats.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrandSplats {
    /// Splat width; `None` uses the groom's own thickness.
    pub width: Option<f32>,
    /// Splats per strand segment. More gives smoother curls.
    pub per_segment: usize,
    /// Opacity of each splat. Below 1 lets strands blend.
    pub alpha: f32,
}

impl Default for StrandSplats {
    fn default() -> Self {
        Self {
            width: None,
            per_segment: 2,
            alpha: 0.85,
        }
    }
}

fn f32_at(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

impl Groom {
    /// Read the binary `.hair` format: a 128-byte header, then the arrays the
    /// flags mark as present (segments, points, thickness, transparency, colour).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let data =
            std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
        if data.len() < 128 || &data[..4] != b"HAIR" {
            bail!("{} is not a .hair file", path.display());
        }
        let strand_count = u32_at(&data, 4) as usize;
        let point_count = u32_at(&data, 8) as usize;
        let flags = u32_at(&data, 12);
        let default_segments = u32_at(&data, 16) as usize;
        let thickness = f32_at(&data, 20);
        let default_color = Vec3::new(f32_at(&data, 28), f32_at(&data, 32), f32_at(&data, 36));
        let (has_segments, has_points) = (flags & 1 != 0, flags & 2 != 0);
        let (has_thickness, has_transparency) = (flags & 4 != 0, flags & 8 != 0);
        let has_color = flags & 16 != 0;
        if !has_points {
            bail!("{} has no point array", path.display());
        }

        let mut off = 128;
        let segments: Vec<usize> = if has_segments {
            let s = (0..strand_count)
                .map(|i| {
                    u16::from_le_bytes(data[off + i * 2..off + i * 2 + 2].try_into().unwrap())
                        as usize
                })
                .collect();
            off += strand_count * 2;
            s
        } else {
            vec![default_segments; strand_count]
        };
        let points_at = off;
        off += point_count * 12;
        if has_thickness {
            off += point_count * 4;
        }
        if has_transparency {
            off += point_count * 4;
        }
        let colors_at = has_color.then_some(off);
        let need = points_at + point_count * 12;
        if data.len() < need {
            bail!("{} is truncated", path.display());
        }

        let mut strands = Vec::with_capacity(strand_count);
        let mut p = 0usize;
        for seg in segments {
            let n = seg + 1;
            let mut points = Vec::with_capacity(n);
            let mut colors = Vec::with_capacity(n);
            for k in 0..n {
                let o = points_at + (p + k) * 12;
                points.push(Vec3::new(
                    f32_at(&data, o),
                    f32_at(&data, o + 4),
                    f32_at(&data, o + 8),
                ));
                colors.push(match colors_at {
                    Some(c) => {
                        let o = c + (p + k) * 12;
                        Vec3::new(f32_at(&data, o), f32_at(&data, o + 4), f32_at(&data, o + 8))
                    }
                    None => default_color,
                });
            }
            p += n;
            strands.push(Strand { points, colors });
        }
        Ok(Self { strands, thickness })
    }

    pub fn bbox(&self) -> BBox {
        let mut b = BBox::EMPTY;
        for s in &self.strands {
            for p in &s.points {
                b.grow(*p);
            }
        }
        b
    }

    pub fn segment_count(&self) -> usize {
        self.strands
            .iter()
            .map(|s| s.points.len().saturating_sub(1))
            .sum()
    }

    /// Keep about `max_strands` strands, spread evenly through the groom.
    pub fn subsampled(&self, max_strands: usize) -> Self {
        let step = (self.strands.len() / max_strands.max(1)).max(1);
        Self {
            strands: self.strands.iter().step_by(step).cloned().collect(),
            thickness: self.thickness,
        }
    }

    /// Centre the groom on the origin and scale it into a `size`-unit box.
    /// Grooms are modelled at their own scale (tens of units), while the
    /// renderer's near/far planes, shadow bias and gaussian scale all assume a
    /// unit-ish model — at the groom's own scale everything self-shadows.
    pub fn normalized(&self, size: f32) -> Self {
        let b = self.bbox();
        let fit = size / b.size().max_element().max(1e-6);
        let offset = b.center();
        Self {
            strands: self
                .strands
                .iter()
                .filter(|s| s.points.len() > 1)
                .map(|s| Strand {
                    points: s.points.iter().map(|p| (*p - offset) * fit).collect(),
                    colors: s.colors.clone(),
                })
                .collect(),
            thickness: self.thickness * fit,
        }
    }

    /// One splat per piece of segment, oriented by the segment's tangent and
    /// shaped like it. Returned splats use [`crate::SourceFormat::Ply`]
    /// semantics: the scales are final world standard deviations.
    pub fn splats(&self, cfg: &StrandSplats) -> Vec<GaussianVertex> {
        let width = cfg.width.unwrap_or(self.thickness);
        let center = self.bbox().center();
        let k = cfg.per_segment.max(1);
        let mut out = Vec::with_capacity(self.segment_count() * k);
        for s in &self.strands {
            for i in 0..s.points.len().saturating_sub(1) {
                let (p0, p1) = (s.points[i], s.points[i + 1]);
                let (c0, c1) = (s.colors[i], s.colors[i + 1]);
                for j in 0..k {
                    let (t0, t1) = (j as f32 / k as f32, (j + 1) as f32 / k as f32);
                    let (a, b) = (p0.lerp(p1, t0), p0.lerp(p1, t1));
                    let mid = (a + b) * 0.5;
                    let tangent = (b - a).normalize_or(Vec3::Y);
                    // Columns of the rotation are the splat's axes in scale
                    // order (length, width, thin), and must stay right-handed
                    // or the quaternion comes out wrong. Flipping two of them
                    // keeps that while turning the flat side away from the
                    // groom's centre.
                    let radial = (mid - center).normalize_or(Vec3::Z);
                    let mut side = tangent.cross(radial).normalize_or(Vec3::X);
                    let mut normal = tangent.cross(side);
                    if normal.dot(radial) < 0.0 {
                        side = -side;
                        normal = -normal;
                    }
                    let q = Quat::from_mat3(&Mat3::from_cols(tangent, side, normal)).normalize();
                    let color = c0.lerp(c1, (t0 + t1) * 0.5).clamp(Vec3::ZERO, Vec3::ONE);
                    out.push(GaussianVertex {
                        position: mid.extend(1.0).to_array(),
                        color: color.extend(cfg.alpha).to_array(),
                        // Half-length: neighbouring splats overlap.
                        scale: [
                            (b - a).length() * 0.5,
                            width * 0.5,
                            (width * 0.05).max(1e-6),
                            0.0,
                        ],
                        normal: normal.extend(0.0).to_array(),
                        rotation: [q.w, q.x, q.y, q.z],
                        pbr: [DEFAULT_METALLIC, DEFAULT_ROUGHNESS, 0.0, 1.0],
                    });
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight_groom(strands: usize, segments: usize) -> Groom {
        Groom {
            thickness: 0.1,
            strands: (0..strands)
                .map(|i| Strand {
                    points: (0..=segments)
                        .map(|k| Vec3::new(i as f32, k as f32 * 0.5, 0.0))
                        .collect(),
                    colors: vec![Vec3::splat(0.5); segments + 1],
                })
                .collect(),
        }
    }

    #[test]
    fn splats_follow_the_strand() {
        let g = straight_groom(4, 6);
        let splats = g.splats(&StrandSplats {
            per_segment: 2,
            ..Default::default()
        });
        assert_eq!(splats.len(), g.segment_count() * 2);
        for s in &splats {
            // Longest axis first, and it points along the strand (+Y here).
            let axes = Mat3::from_quat(Quat::from_xyzw(
                s.rotation[1],
                s.rotation[2],
                s.rotation[3],
                s.rotation[0],
            ));
            assert!(s.scale[0] > s.scale[1] && s.scale[1] > s.scale[2]);
            assert!(axes.col(0).dot(Vec3::Y).abs() > 0.999, "{:?}", axes.col(0));
            // Right-handed, so the shaders' row/column convention holds.
            assert!(axes.determinant() > 0.99, "{}", axes.determinant());
        }
    }

    #[test]
    fn normalizing_fits_the_box_and_scales_thickness() {
        let g = straight_groom(3, 8).normalized(2.0);
        let b = g.bbox();
        assert!((b.size().max_element() - 2.0).abs() < 1e-4);
        assert!(b.center().length() < 1e-4);
        assert!(g.thickness < 0.1, "thickness scales with the groom");
    }

    #[test]
    fn subsampling_spreads_over_the_groom() {
        let g = straight_groom(100, 4);
        let s = g.subsampled(10);
        assert!(s.strands.len() <= 11 && s.strands.len() >= 10);
        // Evenly spread, not the first ten.
        assert!(s.strands.last().unwrap().points[0].x > 50.0);
    }
}
