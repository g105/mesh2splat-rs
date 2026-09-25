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

/// One strand: a polyline with a colour, width and opacity per point.
/// Root-to-tip thinning and see-through tips are most of what makes a groom
/// look soft at the edges, and the file carries both.
#[derive(Clone, Debug)]
pub struct Strand {
    pub points: Vec<Vec3>,
    pub colors: Vec<Vec3>,
    /// Strand width at each point.
    pub thickness: Vec<f32>,
    /// Opacity at each point (1 = opaque).
    pub alpha: Vec<f32>,
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
    /// Splat width; `None` uses each point's own thickness from the file.
    pub width: Option<f32>,
    /// Multiplies the width, whichever it comes from, so a widened strand
    /// keeps its taper. There is no upper limit: a strand wider than its
    /// segments are long is drawn with fewer, longer splats (see
    /// [`MIN_ELONGATION`]).
    pub width_scale: f32,
    /// Thickness of the splats' cross-section as a fraction of their width:
    /// near 0 a flat ribbon, which vanishes edge on; towards 1 a round tube,
    /// which keeps a wide strand's volume from every side. Capped below 1 so
    /// the flat side, and with it the normal, stays defined.
    pub roundness: f32,
    /// Splats per strand segment. More gives smoother curls.
    pub per_segment: usize,
    /// Multiplies each point's own opacity from the file.
    pub alpha: f32,
}

impl Default for StrandSplats {
    fn default() -> Self {
        Self {
            width: None,
            width_scale: 1.0,
            roundness: 0.05,
            per_segment: 2,
            alpha: 0.85,
        }
    }
}

/// How much longer than wide a strand splat must stay. The renderer and the
/// pooler take a splat's longest axis for the strand's direction: a splat
/// wider than long would turn its highlight, and its pooling, across the
/// strand.
pub const MIN_ELONGATION: f32 = 1.25;

/// Most roundness a strand splat may have: at 1 its two short axes tie and
/// its normal is arbitrary.
pub const MAX_ROUNDNESS: f32 = 0.9;

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
        let default_thickness = f32_at(&data, 20);
        let default_alpha = 1.0 - f32_at(&data, 24);
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
        let thickness_at = has_thickness.then(|| {
            let at = off;
            off += point_count * 4;
            at
        });
        let transparency_at = has_transparency.then(|| {
            let at = off;
            off += point_count * 4;
            at
        });
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
            let mut thickness = Vec::with_capacity(n);
            let mut alpha = Vec::with_capacity(n);
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
                thickness.push(match thickness_at {
                    Some(t) => f32_at(&data, t + (p + k) * 4),
                    None => default_thickness,
                });
                alpha.push(match transparency_at {
                    Some(t) => 1.0 - f32_at(&data, t + (p + k) * 4),
                    None => default_alpha,
                });
            }
            p += n;
            strands.push(Strand {
                points,
                colors,
                thickness,
                alpha,
            });
        }
        Ok(Self {
            strands,
            thickness: default_thickness,
        })
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
                    thickness: s.thickness.iter().map(|t| t * fit).collect(),
                    alpha: s.alpha.clone(),
                })
                .collect(),
            thickness: self.thickness * fit,
        }
    }

    /// One splat per piece of segment, oriented by the segment's tangent and
    /// shaped like it. Returned splats use [`crate::SourceFormat::Ply`]
    /// semantics: the scales are final world standard deviations.
    pub fn splats(&self, cfg: &StrandSplats) -> Vec<GaussianVertex> {
        let center = self.bbox().center();
        let per_segment = cfg.per_segment.max(1);
        let roundness = cfg.roundness.clamp(0.0, MAX_ROUNDNESS);
        let mut out = Vec::with_capacity(self.segment_count() * per_segment);
        for s in &self.strands {
            let width_at = |i: usize| {
                cfg.width.unwrap_or_else(|| s.thickness.get(i).copied().unwrap_or(self.thickness))
                    * cfg.width_scale
            };
            let kept = keep_points(&s.points, width_at);
            for w in kept.windows(2) {
                let (i, i1) = (w[0], w[1]);
                let (p0, p1) = (s.points[i], s.points[i1]);
                let (c0, c1) = (s.colors[i], s.colors[i1]);
                // As many splats as fit while each stays longer than wide.
                let most = (p1 - p0).length()
                    / (MIN_ELONGATION * width_at(i).max(width_at(i1)).max(1e-12));
                let k = per_segment.min(most as usize).max(1);
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
                    let mid_t = (t0 + t1) * 0.5;
                    let color = c0.lerp(c1, mid_t).clamp(Vec3::ZERO, Vec3::ONE);
                    // Width and opacity taper along the strand unless overridden.
                    let lerp = |a: f32, b: f32| a + (b - a) * mid_t;
                    let width = lerp(width_at(i), width_at(i1));
                    let alpha = cfg.alpha
                        * lerp(
                            s.alpha.get(i).copied().unwrap_or(1.0),
                            s.alpha.get(i1).copied().unwrap_or(1.0),
                        );
                    out.push(GaussianVertex {
                        position: mid.extend(1.0).to_array(),
                        // Not capped at 1: a clump's opacity carries how many
                        // strands' worth of light it stops.
                        color: color.extend(alpha.max(0.0)).to_array(),
                        // Half-length: neighbouring splats overlap.
                        scale: [
                            (b - a).length() * 0.5,
                            width * 0.5,
                            (width * 0.5 * roundness).max(1e-6),
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

/// The points of a strand to draw it through: all of them, unless it is so
/// wide that splats spanning its segments would come out wider than long.
/// Then points are skipped until the chord is long enough. Detail finer than
/// the strand is wide cannot show anyway.
fn keep_points(points: &[Vec3], width_at: impl Fn(usize) -> f32) -> Vec<usize> {
    let n = points.len();
    if n < 2 {
        return (0..n).collect();
    }
    let long_enough = |a: usize, b: usize| {
        (points[b] - points[a]).length() >= MIN_ELONGATION * width_at(a).max(width_at(b))
    };
    let mut kept = vec![0];
    for i in 1..n - 1 {
        if long_enough(*kept.last().unwrap(), i) {
            kept.push(i);
        }
    }
    // The tip always stays; if the last stretch to it is too short, it
    // replaces the point before rather than leaving a stub.
    if kept.len() > 1 && !long_enough(*kept.last().unwrap(), n - 1) {
        kept.pop();
    }
    kept.push(n - 1);
    kept
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
                    // Tapers from root to tip, as a real groom does.
                    thickness: (0..=segments)
                        .map(|k| 0.1 * (1.0 - 0.5 * k as f32 / segments as f32))
                        .collect(),
                    alpha: vec![1.0; segments + 1],
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
    fn splat_width_tapers_with_the_strand() {
        let g = straight_groom(1, 8);
        let splats = g.splats(&StrandSplats::default());
        let width = |s: &GaussianVertex| s.scale[1];
        // Width comes from the file's per-point thickness unless overridden.
        assert!(width(splats.first().unwrap()) > width(splats.last().unwrap()));
        let fixed = g.splats(&StrandSplats {
            width: Some(0.05),
            ..Default::default()
        });
        assert!(fixed.iter().all(|s| (width(s) - 0.025).abs() < 1e-6));
    }

    #[test]
    fn widening_keeps_the_taper_and_the_strand_direction() {
        let g = straight_groom(1, 8);
        let width = |s: &GaussianVertex| s.scale[1];
        let base = g.splats(&StrandSplats::default());
        // Wide enough that the original segments are far shorter than wide.
        let wide = g.splats(&StrandSplats {
            width_scale: 10.0,
            ..Default::default()
        });
        assert!(wide.len() > 1 && wide.len() < base.len(), "fewer, longer splats");
        // 10 times the file's width wherever the splat sits: this strand
        // tapers linearly from 0.1 at the root to 0.05 at the tip, y = 4.
        for s in &wide {
            let file = 0.1 * (1.0 - 0.5 * s.position[1] / 4.0);
            assert!((width(s) - 10.0 * file * 0.5).abs() < 1e-4, "{} vs {}", width(s), file);
        }
        assert!(width(wide.first().unwrap()) > width(wide.last().unwrap()));
        for s in &wide {
            assert!(s.scale[0] > s.scale[1] && s.scale[1] > s.scale[2], "{:?}", s.scale);
        }
        // The strand is still covered root to tip.
        let ys: Vec<f32> = wide.iter().map(|s| s.position[1]).collect();
        let span = ys.last().unwrap() - ys.first().unwrap() + wide.last().unwrap().scale[0] * 2.0;
        assert!(span > 3.0, "{span}");
    }

    #[test]
    fn roundness_thickens_the_cross_section() {
        let g = straight_groom(1, 4);
        let round = g.splats(&StrandSplats {
            roundness: 0.8,
            ..Default::default()
        });
        assert!((round[0].scale[2] / round[0].scale[1] - 0.8).abs() < 1e-5);
        // Never a tie between the short axes, which would lose the normal.
        let over = g.splats(&StrandSplats {
            roundness: 5.0,
            ..Default::default()
        });
        assert!(over[0].scale[2] < over[0].scale[1]);
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
