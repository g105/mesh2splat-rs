//! Clump detection: group strands that travel together, and replace each group
//! with one wide strand that fills the volume its members occupied.
//!
//! A stylised groom (feature animation, hair cards) reads as a few hundred
//! clumps rather than tens of thousands of hairs. Detecting those clumps in a
//! dense groom gives a low-resolution version with the same silhouette and
//! volume, at a fraction of the splats.
//!
//! Strands are compared as whole curves: each is resampled by arc length to a
//! handful of points and clustered with k-means on those, so two strands clump
//! when they start near each other *and* go the same way. Each clump then
//! becomes its members' mean curve, as wide at each point as its members are
//! spread around it there.

use glam::Vec3;

use crate::hair::{Groom, Strand};

/// Most opacity a clump may carry, in opaque layers. Far above what any real
/// clump reaches; there so a degenerate one cannot swamp the shadow map.
pub const MAX_DENSITY: f32 = 64.0;
use crate::types::BBox;

/// How to find clumps and what to build from them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClumpSettings {
    /// Clumps to find. Fewer clumps, wider strands.
    pub clumps: usize,
    /// Points each strand is resampled to for comparing curves.
    pub samples: usize,
    /// k-means refinement passes.
    pub iterations: usize,
}

impl Default for ClumpSettings {
    fn default() -> Self {
        Self {
            clumps: 500,
            samples: 12,
            iterations: 8,
        }
    }
}

/// The clumps found, and which clump each strand went to.
pub struct Clumping {
    /// One strand per clump.
    pub groom: Groom,
    /// For every input strand, the clump it belongs to.
    pub assignment: Vec<u32>,
    /// Strands in each clump, in the order of `groom.strands`.
    pub members: Vec<u32>,
}

/// `points` resampled to `n` points evenly spaced along the curve.
fn resample<T: Copy>(
    points: &[Vec3],
    values: &[T],
    n: usize,
    lerp: impl Fn(T, T, f32) -> T,
) -> (Vec<Vec3>, Vec<T>) {
    let mut arc = Vec::with_capacity(points.len());
    let mut total = 0.0;
    arc.push(0.0);
    for w in points.windows(2) {
        total += (w[1] - w[0]).length();
        arc.push(total);
    }
    let mut out_p = Vec::with_capacity(n);
    let mut out_v = Vec::with_capacity(n);
    let mut seg = 0;
    for k in 0..n {
        let s = total * k as f32 / (n - 1).max(1) as f32;
        while seg + 2 < points.len() && arc[seg + 1] < s {
            seg += 1;
        }
        let len = (arc[seg + 1] - arc[seg]).max(1e-12);
        let t = ((s - arc[seg]) / len).clamp(0.0, 1.0);
        out_p.push(points[seg].lerp(points[seg + 1], t));
        out_v.push(lerp(values[seg], values[seg + 1], t));
    }
    (out_p, out_v)
}

fn resample_points(points: &[Vec3], n: usize) -> Vec<Vec3> {
    resample(points, points, n, |a, b, t| a.lerp(b, t)).0
}

/// Mean squared distance between two curves sampled alike.
fn curve_distance(a: &[Vec3], b: &[Vec3]) -> f32 {
    a.iter().zip(b).map(|(p, q)| p.distance_squared(*q)).sum::<f32>() / a.len() as f32
}

/// Uniform grid over the clump centres' roots, so each strand only compares
/// itself against the clumps rooted nearby rather than all of them.
struct RootGrid {
    origin: Vec3,
    cell: f32,
    dims: [i32; 3],
    cells: Vec<Vec<u32>>,
}

impl RootGrid {
    fn new(roots: &[Vec3], cell: f32) -> Self {
        let mut bbox = BBox::EMPTY;
        for r in roots {
            bbox.grow(*r);
        }
        let cell = cell.max(1e-6);
        let size = (bbox.size() / cell).ceil().as_ivec3().max(glam::IVec3::ONE);
        let dims = [size.x, size.y, size.z];
        let mut grid = Self {
            origin: bbox.min,
            cell,
            dims,
            cells: vec![Vec::new(); (dims[0] * dims[1] * dims[2]) as usize],
        };
        for (i, r) in roots.iter().enumerate() {
            let c = grid.coord(*r);
            let idx = grid.index(c);
            grid.cells[idx].push(i as u32);
        }
        grid
    }

    fn coord(&self, p: Vec3) -> [i32; 3] {
        let c = ((p - self.origin) / self.cell).floor().as_ivec3();
        [
            c.x.clamp(0, self.dims[0] - 1),
            c.y.clamp(0, self.dims[1] - 1),
            c.z.clamp(0, self.dims[2] - 1),
        ]
    }

    fn index(&self, c: [i32; 3]) -> usize {
        ((c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]) as usize
    }

    /// Clumps rooted in the cells around `p`, widening the search until some
    /// turn up.
    fn near(&self, p: Vec3, out: &mut Vec<u32>) {
        out.clear();
        let c = self.coord(p);
        let most = self.dims.iter().copied().max().unwrap_or(1);
        let mut r = 1;
        loop {
            for z in (c[2] - r).max(0)..=(c[2] + r).min(self.dims[2] - 1) {
                for y in (c[1] - r).max(0)..=(c[1] + r).min(self.dims[1] - 1) {
                    for x in (c[0] - r).max(0)..=(c[0] + r).min(self.dims[0] - 1) {
                        out.extend_from_slice(&self.cells[self.index([x, y, z])]);
                    }
                }
            }
            if !out.is_empty() || r >= most {
                return;
            }
            r *= 2;
        }
    }
}

/// Runs `f(first, chunk)` over `items` split across the available cores.
fn par_chunks<T: Send>(items: &mut [T], f: impl Fn(usize, &mut [T]) + Sync) {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = items.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (i, part) in items.chunks_mut(chunk).enumerate() {
            let f = &f;
            scope.spawn(move || f(i * chunk, part));
        }
    });
}

impl Groom {
    /// Group the strands into about `cfg.clumps` clumps and build one strand
    /// per clump. Deterministic: the same groom and settings give the same
    /// clumps.
    pub fn clumped(&self, cfg: &ClumpSettings) -> Clumping {
        let strands: Vec<&Strand> = self.strands.iter().filter(|s| s.points.len() > 1).collect();
        let n = strands.len();
        let samples = cfg.samples.max(2);
        let k = cfg.clumps.clamp(1, n.max(1));
        if n == 0 {
            return Clumping {
                groom: Groom {
                    strands: Vec::new(),
                    thickness: self.thickness,
                },
                assignment: Vec::new(),
                members: Vec::new(),
            };
        }
        let curves: Vec<Vec<Vec3>> = strands.iter().map(|s| resample_points(&s.points, samples)).collect();

        // Start from strands spread evenly through the groom; files list
        // strands in scan order, so that spreads the seeds over the scalp.
        let mut centres: Vec<Vec<Vec3>> =
            (0..k).map(|c| curves[c * n / k].clone()).collect();
        // Clumps are about as far apart as their roots are, so search that far.
        let mut root_box = BBox::EMPTY;
        for c in &curves {
            root_box.grow(c[0]);
        }
        let extent = root_box.size();
        let area = (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x).max(1e-12);
        let spacing = (area / k as f32).sqrt();

        let mut assignment = vec![0u32; n];
        for _ in 0..cfg.iterations.max(1) {
            let roots: Vec<Vec3> = centres.iter().map(|c| c[0]).collect();
            let grid = RootGrid::new(&roots, spacing * 2.0);
            par_chunks(&mut assignment, |first, part| {
                let mut near = Vec::new();
                for (j, a) in part.iter_mut().enumerate() {
                    let curve = &curves[first + j];
                    grid.near(curve[0], &mut near);
                    let best = near
                        .iter()
                        .map(|&c| (c, curve_distance(curve, &centres[c as usize])))
                        .min_by(|a, b| a.1.total_cmp(&b.1));
                    *a = best.map_or(0, |b| b.0);
                }
            });
            // Move each centre to its members' mean curve. A centre nobody
            // chose keeps its place and may pick up strands next pass.
            let mut sums = vec![vec![Vec3::ZERO; samples]; k];
            let mut counts = vec![0u32; k];
            for (curve, &a) in curves.iter().zip(&assignment) {
                counts[a as usize] += 1;
                for (s, p) in sums[a as usize].iter_mut().zip(curve) {
                    *s += *p;
                }
            }
            for ((centre, sum), &count) in centres.iter_mut().zip(&sums).zip(&counts) {
                if count > 0 {
                    for (c, s) in centre.iter_mut().zip(sum) {
                        *c = *s / count as f32;
                    }
                }
            }
        }

        // Build one strand per non-empty clump.
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, &a) in assignment.iter().enumerate() {
            groups[a as usize].push(i);
        }
        let mut remap = vec![u32::MAX; k];
        let mut out = Vec::new();
        let mut members = Vec::new();
        for (c, group) in groups.iter().enumerate() {
            if group.is_empty() {
                continue;
            }
            remap[c] = out.len() as u32;
            members.push(group.len() as u32);
            out.push(clump_strand(group.iter().map(|&i| strands[i]), self.thickness));
        }
        for a in &mut assignment {
            *a = remap[*a as usize];
        }
        Clumping {
            groom: Groom {
                strands: out,
                thickness: self.thickness,
            },
            assignment,
            members,
        }
    }
}

/// One strand standing in for `members`: their mean curve, at the resolution
/// of a typical member, as wide at each point as they are spread around it.
fn clump_strand<'a>(members: impl Iterator<Item = &'a Strand> + Clone, default_width: f32) -> Strand {
    let mut lens: Vec<usize> = members.clone().map(|s| s.points.len()).collect();
    lens.sort_unstable();
    let points = lens[lens.len() / 2].max(2);
    let count = lens.len() as f32;

    // Every member resampled to the same parameterisation.
    struct Sampled {
        points: Vec<Vec3>,
        colors: Vec<Vec3>,
        width: Vec<f32>,
        alpha: Vec<f32>,
    }
    let sampled: Vec<Sampled> = members
        .map(|s| {
            let pad = |v: &[f32], d: f32| -> Vec<f32> {
                (0..s.points.len()).map(|i| v.get(i).copied().unwrap_or(d)).collect()
            };
            let (p, colors) = resample(&s.points, &s.colors, points, |a, b, t| a.lerp(b, t));
            let width = resample(&s.points, &pad(&s.thickness, default_width), points, lerp).1;
            let alpha = resample(&s.points, &pad(&s.alpha, 1.0), points, lerp).1;
            Sampled {
                points: p,
                colors,
                width,
                alpha,
            }
        })
        .collect();

    let mean = |f: &dyn Fn(&Sampled, usize) -> Vec3, i: usize| -> Vec3 {
        sampled.iter().map(|s| f(s, i)).sum::<Vec3>() / count
    };
    let centre: Vec<Vec3> = (0..points).map(|i| mean(&|s, i| s.points[i], i)).collect();
    let mut out = Strand {
        points: centre.clone(),
        colors: Vec::with_capacity(points),
        thickness: Vec::with_capacity(points),
        alpha: Vec::with_capacity(points),
    };
    for i in 0..points {
        let prev = centre[i.saturating_sub(1)];
        let next = centre[(i + 1).min(points - 1)];
        let tangent = (next - prev).normalize_or_zero();
        // Spread across the clump only: members sitting further along the
        // curve than the centre are not what makes it wide.
        let mut across = 0.0;
        let mut own = 0.0;
        let mut stops = 0.0;
        for s in &sampled {
            let d = s.points[i] - centre[i];
            let d = d - tangent * d.dot(tangent);
            across += d.length_squared();
            own += (s.width[i] * 0.5).powi(2);
            stops += s.alpha[i] * s.width[i] * 0.5;
        }
        // A strand splat's width is two standard deviations of its gaussian;
        // a round spread's variance along any one axis is half its total. The
        // members' own widths add to that.
        let sigma = (across / count * 0.5 + own / count).sqrt();
        out.thickness.push(sigma * 2.0);
        // As opaque as the members were together, per unit length: the same
        // light stopped over a wider face needs less opacity. Often more than
        // 1 — many strands stop more light than one opaque layer — which only
        // shadows and occlusion can show (see `GaussianVertex::color`).
        out.alpha.push((stops / sigma.max(1e-12)).clamp(0.0, MAX_DENSITY));
        out.colors.push(mean(&|s, i| s.colors[i], i));
    }
    out
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `bundles` tight bundles of `per` parallel strands, far apart.
    fn bundles(bundles: usize, per: usize) -> Groom {
        let mut strands = Vec::new();
        for b in 0..bundles {
            for s in 0..per {
                let base = Vec3::new(b as f32 * 10.0, 0.0, 0.0);
                let offset = Vec3::new((s % 3) as f32 * 0.1, 0.0, (s / 3) as f32 * 0.1);
                let points: Vec<Vec3> = (0..8).map(|i| base + offset + Vec3::Y * i as f32).collect();
                strands.push(Strand {
                    colors: vec![Vec3::splat(b as f32 / bundles as f32); 8],
                    thickness: vec![0.01; 8],
                    alpha: vec![0.5; 8],
                    points,
                });
            }
        }
        Groom {
            strands,
            thickness: 0.01,
        }
    }

    #[test]
    fn separate_bundles_become_separate_clumps() {
        let g = bundles(4, 9);
        let c = g.clumped(&ClumpSettings {
            clumps: 4,
            ..Default::default()
        });
        assert_eq!(c.groom.strands.len(), 4);
        assert_eq!(c.members, vec![9; 4]);
        // Every strand of a bundle went to the same clump.
        for b in 0..4 {
            let first = c.assignment[b * 9];
            assert!(c.assignment[b * 9..(b + 1) * 9].iter().all(|&a| a == first));
        }
    }

    #[test]
    fn a_clump_is_as_wide_as_its_members_are_spread() {
        let g = bundles(1, 9);
        let c = g.clumped(&ClumpSettings {
            clumps: 1,
            ..Default::default()
        });
        let s = &c.groom.strands[0];
        // Members sit on a 3 x 3 grid 0.1 apart: 0.0816 standard deviation
        // along each axis across the strand.
        let sigma = s.thickness[3] / 2.0;
        let expected = ((2.0f32 / 3.0 * 0.01) + 0.005f32.powi(2)).sqrt();
        assert!((sigma - expected).abs() < 1e-3, "{sigma} vs {expected}");
        // Centred on the bundle, running along it.
        assert!((s.points[0] - Vec3::new(0.1, 0.0, 0.1)).length() < 1e-4);
        assert!((s.points.last().unwrap().y - 7.0).abs() < 1e-4);
    }

    #[test]
    fn a_clump_stops_as_much_light_as_its_members() {
        let g = bundles(1, 9);
        let c = g.clumped(&ClumpSettings {
            clumps: 1,
            ..Default::default()
        });
        let s = &c.groom.strands[0];
        let sigma = s.thickness[3] / 2.0;
        let members = 9.0 * 0.5 * 0.005;
        assert!((s.alpha[3] * sigma - members).abs() < 1e-5);
    }

    #[test]
    fn a_dense_clump_keeps_all_its_opacity() {
        // Many opaque strands packed tight stop more light than one opaque
        // layer; capping that at 1 would lighten its shadow.
        let mut g = bundles(1, 9);
        for s in &mut g.strands {
            s.alpha.fill(1.0);
            s.thickness.fill(0.2);
        }
        let c = g.clumped(&ClumpSettings {
            clumps: 1,
            ..Default::default()
        });
        let s = &c.groom.strands[0];
        let sigma = s.thickness[3] / 2.0;
        assert!(s.alpha[3] > 1.0, "{}", s.alpha[3]);
        assert!((s.alpha[3] * sigma - 9.0 * 0.1).abs() < 1e-4);
    }

    #[test]
    fn clumping_is_deterministic() {
        let g = bundles(5, 6);
        let cfg = ClumpSettings {
            clumps: 3,
            ..Default::default()
        };
        assert_eq!(g.clumped(&cfg).assignment, g.clumped(&cfg).assignment);
    }
}
