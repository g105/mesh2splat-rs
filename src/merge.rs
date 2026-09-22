//! Merge neighbouring converted splats that look alike into larger ones.
//!
//! The converter emits one splat per texel of a planar projection grid (see
//! `convert.wgsl`), which is uniform density: a flat grey panel gets as many
//! splats as a detailed decal. Every triangle projected on the same axis shares
//! that grid, so neighbouring splats of a surface are neighbouring grid cells,
//! across triangle edges and UV seams.
//!
//! This builds a quadtree over the grid: a full 2x2 block whose colour, normal,
//! metallic/roughness and flatness stay within tolerance becomes one splat
//! twice the size, and merged blocks merge again up to `max_level`. Surfaces
//! that overlap in the projection (the front and back of a closed mesh) share
//! grid cells, so the candidates of a block are split into depth layers first;
//! a layer only merges when it has exactly one member in each of the four
//! cells, so surface edges and ambiguous overlaps stay as they are. The merged splat's shape is the members' average shape
//! scaled by the block size, which keeps the "Gaussian Scale" slider meaning
//! "standard deviation relative to splat spacing".

use glam::{DMat3, DVec3, Mat3, Quat, Vec2, Vec3, Vec4};

use crate::types::{BBox, GaussianVertex};

/// How alike a block must be to become one splat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MergeSettings {
    /// Largest per-channel spread (0..1, gamma-space RGBA) inside a merged splat.
    pub color_tolerance: f32,
    /// Largest angle between any member normal and the merged normal, degrees.
    pub normal_tolerance_deg: f32,
    /// Largest RMS distance of member centers from their best-fit plane, as a
    /// fraction of the merged splat's footprint (curvature allowance).
    pub flatness: f32,
    /// Largest block, as a power of two: 1 = 2x2, 4 = 16x16.
    pub max_level: u32,
}

impl Default for MergeSettings {
    fn default() -> Self {
        Self::from_strength(Self::DEFAULT_STRENGTH)
    }
}

impl MergeSettings {
    /// On DamagedHelmet this halves the splat count at ~45 dB PSNR, about 4.5 dB
    /// better than reaching the same count by lowering the sampling resolution.
    pub const DEFAULT_STRENGTH: f32 = 0.25;

    /// One-knob preset: 0 = merge only identical splats, 1 = aggressive.
    pub fn from_strength(strength: f32) -> Self {
        let t = strength.clamp(0.0, 1.0);
        Self {
            color_tolerance: 0.1 * t,
            normal_tolerance_deg: 20.0 * t,
            flatness: 0.1 * t,
            max_level: 4,
        }
    }
}

/// How the grid coordinates stored in the splats map back to world space.
pub struct GridInfo {
    /// Side of the conversion render target.
    pub resolution: u32,
    /// Projection box of each mesh; a single entry means every mesh shared it.
    pub boxes: Vec<BBox>,
}

#[derive(Clone, Debug, Default)]
pub struct MergeStats {
    pub input: usize,
    pub output: usize,
    /// Merged splats created at each level (index 0 = 2x2 blocks).
    pub merged_per_level: Vec<usize>,
    /// Wall time of the merge (set by the converter).
    pub duration: std::time::Duration,
    /// Whether it ran on the GPU.
    pub gpu: bool,
}

/// Grid cell marker for splats that did not come from the converter.
pub const NO_GRID: u32 = 3 << 30;

/// Decode the `scale.w` grid tag written by `convert.wgsl`: (axis, x, y).
fn grid_cell(g: &GaussianVertex) -> Option<(u32, u32, u32)> {
    let bits = g.scale[3].to_bits();
    let axis = bits >> 30;
    (axis < 3).then_some((axis, bits & 0x7fff, (bits >> 15) & 0x7fff))
}

fn mesh_index(g: &GaussianVertex) -> u32 {
    g.normal[3].to_bits()
}

// Cell keys: group (mesh or 0) | axis | y | x.
fn key(group: u32, axis: u32, x: u32, y: u32) -> u64 {
    ((group as u64) << 34) | ((axis as u64) << 32) | ((y as u64) << 16) | x as u64
}

fn unkey(k: u64) -> (u32, u32, u32, u32) {
    (
        (k >> 34) as u32,
        ((k >> 32) & 3) as u32,
        (k & 0xffff) as u32,
        ((k >> 16) & 0xffff) as u32,
    )
}

// Entries are leaf splat indices, or merged node indices tagged with NODE.
const NODE: u32 = 1 << 31;

/// Symmetric 3x3 stored as xx, xy, xz, yy, yz, zz.
type Sym = [f64; 6];

fn sym(m: DMat3) -> Sym {
    [m.x_axis.x, m.y_axis.x, m.z_axis.x, m.y_axis.y, m.z_axis.y, m.z_axis.z]
}

fn unsym(s: &Sym) -> DMat3 {
    DMat3::from_cols(
        DVec3::new(s[0], s[1], s[2]),
        DVec3::new(s[1], s[3], s[4]),
        DVec3::new(s[2], s[4], s[5]),
    )
}

/// Rotation matrix whose columns are the splat's local axes, matching
/// `cast_quat_to_mat3` / `compute_cov3d` in `common.wgsl`
/// (covariance = R * diag(scale^2) * R^T). Rotation is stored (w, x, y, z).
fn splat_axes(rotation: [f32; 4]) -> DMat3 {
    let [w, x, y, z] = rotation;
    DMat3::from_quat(glam::DQuat::from_xyzw(x as f64, y as f64, z as f64, w as f64).normalize())
}

/// Unscaled shape covariance of a splat.
fn shape_covariance(g: &GaussianVertex) -> DMat3 {
    let r = splat_axes(g.rotation);
    let s = DVec3::new(g.scale[0] as f64, g.scale[1] as f64, g.scale[2] as f64);
    r * DMat3::from_diagonal(s * s) * r.transpose()
}

/// Eigen-decomposition of a symmetric matrix (cyclic Jacobi).
/// Returns (eigenvalues, eigenvectors as columns).
fn eigen_sym(m: DMat3) -> (DVec3, DMat3) {
    let mut a = m.to_cols_array_2d(); // a[col][row]; symmetric so layout is moot
    let mut v = DMat3::IDENTITY.to_cols_array_2d();
    for _ in 0..32 {
        let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        if off < 1e-30 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for col in &mut a {
                let (akp, akq) = (col[p], col[q]);
                col[p] = c * akp - s * akq;
                col[q] = s * akp + c * akq;
            }
            let (col_p, col_q) = (a[p], a[q]);
            for k in 0..3 {
                a[p][k] = c * col_p[k] - s * col_q[k];
                a[q][k] = s * col_p[k] + c * col_q[k];
            }
            for row in &mut v {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    // `v` was updated as rows of V^T, i.e. v[k] is row k of V: transpose back.
    let vecs = DMat3::from_cols_array_2d(&v).transpose();
    (DVec3::new(a[0][0], a[1][1], a[2][2]), vecs)
}

/// Smallest eigenvalue of a symmetric 3x3 matrix (closed form, Smith 1961).
fn min_eigenvalue(m: DMat3) -> f64 {
    let (a, b, c) = (m.x_axis.x, m.y_axis.y, m.z_axis.z);
    let (d, e, f) = (m.y_axis.x, m.z_axis.y, m.z_axis.x); // xy, yz, xz
    let p1 = d * d + e * e + f * f;
    if p1 <= 1e-30 * (a * a + b * b + c * c).max(1e-300) {
        return a.min(b).min(c);
    }
    let q = (a + b + c) / 3.0;
    let p2 = (a - q).powi(2) + (b - q).powi(2) + (c - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    let bm = (m - DMat3::from_diagonal(DVec3::splat(q))) * (1.0 / p);
    let r = (bm.determinant() / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    q + 2.0 * p * (phi + 2.0 * std::f64::consts::FRAC_PI_3).cos()
}

/// Rotation (w, x, y, z) and scales of a splat with the given covariance.
fn splat_from_covariance(cov: DMat3) -> ([f32; 4], [f32; 3]) {
    let (vals, mut vecs) = eigen_sym(cov);
    if vecs.determinant() < 0.0 {
        vecs.z_axis = -vecs.z_axis;
    }
    let q = Quat::from_mat3(&Mat3::from_cols(
        vecs.x_axis.as_vec3(),
        vecs.y_axis.as_vec3(),
        vecs.z_axis.as_vec3(),
    ))
    .normalize();
    let s = |v: f64| (v.max(0.0).sqrt() as f32).max(1e-7);
    ([q.w, q.x, q.y, q.z], [s(vals.x), s(vals.y), s(vals.z)])
}

/// Running statistics of the original splats inside a block.
#[derive(Clone, Copy)]
struct Stats {
    count: f64,
    sum_p: DVec3,
    sum_pp: Sym,
    sum_cov: Sym,
    sum_c: Vec4,
    cmin: Vec4,
    cmax: Vec4,
    sum_n: Vec3,
    /// Largest angle (radians) between any member normal and `sum_n`.
    cone: f32,
    sum_pbr: Vec2,
    pmin: Vec2,
    pmax: Vec2,
}

impl Stats {
    fn leaf(g: &GaussianVertex) -> Self {
        let p = DVec3::new(g.position[0] as f64, g.position[1] as f64, g.position[2] as f64);
        let c = Vec4::from_array(g.color);
        let pbr = Vec2::new(g.pbr[0], g.pbr[1]);
        let n = Vec3::from_slice(&g.normal[..3]).normalize_or(Vec3::Z);
        Self {
            count: 1.0,
            sum_p: p,
            sum_pp: sym(DMat3::from_cols(p * p.x, p * p.y, p * p.z)),
            sum_cov: sym(shape_covariance(g)),
            sum_c: c,
            cmin: c,
            cmax: c,
            sum_n: n,
            cone: 0.0,
            sum_pbr: pbr,
            pmin: pbr,
            pmax: pbr,
        }
    }

    fn combine(ch: &[Stats; 4]) -> Self {
        let mut s = ch[0];
        for c in &ch[1..] {
            s.count += c.count;
            s.sum_p += c.sum_p;
            for i in 0..6 {
                s.sum_pp[i] += c.sum_pp[i];
                s.sum_cov[i] += c.sum_cov[i];
            }
            s.sum_c += c.sum_c;
            s.cmin = s.cmin.min(c.cmin);
            s.cmax = s.cmax.max(c.cmax);
            s.sum_n += c.sum_n;
            s.sum_pbr += c.sum_pbr;
            s.pmin = s.pmin.min(c.pmin);
            s.pmax = s.pmax.max(c.pmax);
        }
        let n = s.sum_n.normalize_or(Vec3::Z);
        s.cone = ch
            .iter()
            .map(|c| c.cone + c.sum_n.normalize_or(n).dot(n).clamp(-1.0, 1.0).acos())
            .fold(0.0, f32::max);
        s
    }

    /// Whether the block may become one splat. `footprint` is its side in world units.
    fn uniform(&self, cfg: &MergeSettings, footprint: f64) -> bool {
        let c_spread = (self.cmax - self.cmin).max_element();
        let p_spread = (self.pmax - self.pmin).max_element();
        if c_spread > cfg.color_tolerance
            || p_spread > 2.0 * cfg.color_tolerance
            || self.cone > cfg.normal_tolerance_deg.to_radians()
        {
            return false;
        }
        // RMS distance of the member centers from their best-fit plane.
        let mean = self.sum_p / self.count;
        let scatter = unsym(&self.sum_pp) * (1.0 / self.count)
            - DMat3::from_cols(mean * mean.x, mean * mean.y, mean * mean.z);
        let off_plane = min_eigenvalue(scatter).max(0.0).sqrt();
        off_plane <= cfg.flatness as f64 * footprint
    }

    fn to_splat(self, level: u32) -> GaussianVertex {
        let inv = 1.0 / self.count;
        let p = self.sum_p * inv;
        let side = (1u64 << level) as f64;
        let cov = unsym(&self.sum_cov) * (inv * side * side);
        let (rotation, scale) = splat_from_covariance(cov);
        let c = self.sum_c / self.count as f32;
        let n = self.sum_n.normalize_or(Vec3::Z);
        let pbr = self.sum_pbr / self.count as f32;
        GaussianVertex {
            position: [p.x as f32, p.y as f32, p.z as f32, 1.0],
            color: c.to_array(),
            scale: [scale[0], scale[1], scale[2], f32::from_bits(NO_GRID)],
            normal: [n.x, n.y, n.z, 0.0],
            rotation,
            pbr: [pbr.x, pbr.y, 0.0, 1.0],
        }
    }
}

/// World size of one grid cell for splats projected along `axis` in `bbox`
/// (mirrors `ortho_uv` in `convert.wgsl`).
pub(crate) fn cell_size(bbox: &BBox, axis: u32, resolution: u32) -> f64 {
    let s = bbox.size();
    let r = match axis {
        0 => s.y.max(s.z),
        1 => s.x.max(s.z),
        _ => s.x.max(s.y),
    };
    r as f64 / resolution.max(1) as f64
}

/// A leaf splat or merged node sitting in a grid cell at the current level.
#[derive(Clone, Copy)]
struct Item {
    /// Cell key at the current level.
    key: u64,
    /// Coordinate along the projection axis, to tell overlapping layers apart.
    depth: f32,
    entry: u32,
    /// Which quarter of its parent cell the item sits in.
    quadrant: u8,
}

/// A block that passed the merge test.
struct Merged {
    key: u64,
    children: [u32; 4],
    stats: Stats,
    depth: f32,
}

/// Merge alike neighbours of freshly converted splats. Splats without a grid
/// tag (e.g. loaded from a PLY) pass through unchanged.
pub fn merge(
    splats: &[GaussianVertex],
    grid: &GridInfo,
    cfg: &MergeSettings,
) -> (Vec<GaussianVertex>, MergeStats) {
    let shared = grid.boxes.len() <= 1;
    let mut items: Vec<Item> = splats
        .iter()
        .enumerate()
        .filter_map(|(i, g)| {
            let (axis, x, y) = grid_cell(g)?;
            let group = if shared { 0 } else { mesh_index(g) };
            Some(Item {
                key: key(group, axis, x, y),
                depth: g.position[axis as usize],
                entry: i as u32,
                quadrant: 0,
            })
        })
        .collect();

    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut leaf_used = vec![false; splats.len()];
    let mut nodes: Vec<(Stats, u32)> = Vec::new();
    let mut node_used: Vec<bool> = Vec::new();
    let mut stats = MergeStats {
        input: splats.len(),
        ..Default::default()
    };

    for level in 1..=cfg.max_level {
        // Move every item to its parent cell, remembering which quarter it came from.
        for it in &mut items {
            let (group, axis, x, y) = unkey(it.key);
            it.quadrant = ((y & 1) * 2 + (x & 1)) as u8;
            it.key = key(group, axis, x >> 1, y >> 1);
        }
        items.sort_unstable_by(|a, b| a.key.cmp(&b.key).then(a.depth.total_cmp(&b.depth)));

        // Cells are independent: split the sorted items at cell boundaries
        // across threads, then apply the results in order.
        let mut cuts = vec![0];
        for t in 1..threads {
            let mut c = (items.len() * t / threads).max(*cuts.last().unwrap());
            while c > 0 && c < items.len() && items[c].key == items[c - 1].key {
                c += 1;
            }
            cuts.push(c.min(items.len()));
        }
        cuts.push(items.len());
        let ctx = LevelCtx {
            splats,
            nodes: &nodes,
            grid,
            shared,
            cfg,
            level,
        };
        let results: Vec<Vec<Merged>> = std::thread::scope(|s| {
            let handles: Vec<_> = cuts
                .windows(2)
                .map(|w| {
                    let chunk = &items[w[0]..w[1]];
                    let ctx = &ctx;
                    s.spawn(move || ctx.merge_cells(chunk))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut next: Vec<Item> = Vec::new();
        let before = nodes.len();
        for m in results.into_iter().flatten() {
            for e in m.children {
                if e & NODE != 0 {
                    node_used[(e & !NODE) as usize] = true;
                } else {
                    leaf_used[e as usize] = true;
                }
            }
            next.push(Item {
                key: m.key,
                depth: m.depth,
                entry: NODE | nodes.len() as u32,
                quadrant: 0,
            });
            nodes.push((m.stats, level));
            node_used.push(false);
        }
        stats.merged_per_level.push(nodes.len() - before);
        if next.is_empty() {
            break;
        }
        items = next;
    }

    let mut out: Vec<GaussianVertex> = splats
        .iter()
        .zip(&leaf_used)
        .filter(|(_, used)| !**used)
        .map(|(g, _)| *g)
        .collect();
    let live: Vec<&(Stats, u32)> = nodes
        .iter()
        .zip(&node_used)
        .filter(|(_, used)| !**used)
        .map(|(n, _)| n)
        .collect();
    let per = live.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = live
            .chunks(per)
            .map(|c| s.spawn(move || c.iter().map(|(st, l)| st.to_splat(*l)).collect::<Vec<_>>()))
            .collect();
        for h in handles {
            out.extend(h.join().unwrap());
        }
    });
    stats.output = out.len();
    (out, stats)
}

/// Read-only state shared by the threads of one merge level.
struct LevelCtx<'a> {
    splats: &'a [GaussianVertex],
    nodes: &'a [(Stats, u32)],
    grid: &'a GridInfo,
    shared: bool,
    cfg: &'a MergeSettings,
    level: u32,
}

impl LevelCtx<'_> {
    /// Colour and metallic/roughness bounds of a leaf or node.
    fn bounds(&self, e: u32) -> (Vec4, Vec4, Vec2, Vec2) {
        if e & NODE != 0 {
            let s = &self.nodes[(e & !NODE) as usize].0;
            (s.cmin, s.cmax, s.pmin, s.pmax)
        } else {
            let g = &self.splats[e as usize];
            let c = Vec4::from_array(g.color);
            let p = Vec2::new(g.pbr[0], g.pbr[1]);
            (c, c, p, p)
        }
    }

    fn stats(&self, e: u32) -> Stats {
        if e & NODE != 0 {
            self.nodes[(e & !NODE) as usize].0
        } else {
            Stats::leaf(&self.splats[e as usize])
        }
    }

    /// Try every depth layer of every cell in `items` (sorted by key, then depth).
    fn merge_cells(&self, items: &[Item]) -> Vec<Merged> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < items.len() {
            let k = items[i].key;
            let mut end = i + 1;
            while end < items.len() && items[end].key == k {
                end += 1;
            }
            let (group, axis, _, _) = unkey(k);
            let bbox = &self.grid.boxes[if self.shared { 0 } else { group as usize }];
            let footprint =
                cell_size(bbox, axis, self.grid.resolution) * (1u64 << self.level) as f64;
            // Split the cell's candidates into depth layers.
            let mut start = i;
            for j in i..end {
                if j + 1 < end && ((items[j + 1].depth - items[j].depth) as f64) <= footprint {
                    continue;
                }
                let layer = &items[start..=j];
                start = j + 1;
                if let Some(m) = self.try_merge(layer, k, axis, footprint) {
                    out.push(m);
                }
            }
            i = end;
        }
        out
    }

    fn try_merge(&self, layer: &[Item], key: u64, axis: u32, footprint: f64) -> Option<Merged> {
        if layer.len() != 4 {
            return None;
        }
        let mut ch = [u32::MAX; 4];
        for it in layer {
            ch[it.quadrant as usize] = it.entry;
        }
        if ch.contains(&u32::MAX) {
            return None; // two members in one cell
        }
        // Cheap colour / material test before the full statistics.
        let (mut cmin, mut cmax, mut pmin, mut pmax) = self.bounds(ch[0]);
        for &e in &ch[1..] {
            let (a, b, c, d) = self.bounds(e);
            cmin = cmin.min(a);
            cmax = cmax.max(b);
            pmin = pmin.min(c);
            pmax = pmax.max(d);
        }
        if (cmax - cmin).max_element() > self.cfg.color_tolerance
            || (pmax - pmin).max_element() > 2.0 * self.cfg.color_tolerance
        {
            return None;
        }
        let stats = Stats::combine(&ch.map(|e| self.stats(e)));
        if !stats.uniform(self.cfg, footprint) {
            return None;
        }
        Some(Merged {
            key,
            children: ch,
            depth: (stats.sum_p[axis as usize] / stats.count) as f32,
            stats,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_eigenvalue_matches_jacobi() {
        let a = DMat3::from_cols(
            DVec3::new(4.0, 1.0, -2.0),
            DVec3::new(1.0, 3.0, 0.5),
            DVec3::new(-2.0, 0.5, 5.0),
        );
        let (vals, _) = eigen_sym(a);
        assert!((min_eigenvalue(a) - vals.min_element()).abs() < 1e-9);
        let d = DMat3::from_diagonal(DVec3::new(3.0, 1e-12, 2.0));
        assert!(min_eigenvalue(d).abs() < 1e-9);
    }

    fn close(a: DMat3, b: DMat3, tol: f64) -> bool {
        (a - b).to_cols_array().iter().all(|d| d.abs() <= tol)
    }

    #[test]
    fn covariance_round_trip() {
        let q = Quat::from_euler(glam::EulerRot::XYZ, 0.3, -1.1, 2.0);
        let g = GaussianVertex {
            rotation: [q.w, q.x, q.y, q.z],
            scale: [0.5, 2.0, 1e-3, 0.0],
            ..Default::default()
        };
        let cov = shape_covariance(&g);
        let (rotation, scale) = splat_from_covariance(cov);
        let back = GaussianVertex {
            rotation,
            scale: [scale[0], scale[1], scale[2], 0.0],
            ..Default::default()
        };
        assert!(close(cov, shape_covariance(&back), 1e-5));
    }

    /// Axis-2 grid of `n x n` splats on z = 0 with spacing 1, color from `color(x, y)`.
    fn plane(n: u32, color: impl Fn(u32, u32) -> [f32; 4]) -> (Vec<GaussianVertex>, GridInfo) {
        let mut v = Vec::new();
        for y in 0..n {
            for x in 0..n {
                v.push(GaussianVertex {
                    position: [x as f32 + 0.5, y as f32 + 0.5, 0.0, 1.0],
                    color: color(x, y),
                    scale: [n as f32, n as f32, 1e-7, f32::from_bits((2 << 30) | (y << 15) | x)],
                    normal: [0.0, 0.0, 1.0, 0.0],
                    rotation: [1.0, 0.0, 0.0, 0.0],
                    pbr: [0.1, 0.5, 0.0, 1.0],
                });
            }
        }
        let grid = GridInfo {
            resolution: n,
            boxes: vec![BBox {
                min: Vec3::ZERO,
                max: Vec3::new(n as f32, n as f32, 0.0),
            }],
        };
        (v, grid)
    }

    #[test]
    fn uniform_plane_collapses_to_one_splat() {
        let (v, grid) = plane(16, |_, _| [0.5, 0.5, 0.5, 1.0]);
        let (out, stats) = merge(&v, &grid, &MergeSettings::default());
        assert_eq!(out.len(), 1, "{stats:?}");
        let g = out[0];
        assert!((Vec3::from_slice(&g.position[..3]) - Vec3::new(8.0, 8.0, 0.0)).length() < 1e-4);
        // 16x the per-splat scale in-plane, still flat.
        let mut s = g.scale[..3].to_vec();
        s.sort_by(f32::total_cmp);
        assert!(s[0] < 1e-5 && (s[1] - 256.0).abs() < 1e-2 && (s[2] - 256.0).abs() < 1e-2, "{s:?}");
        // The thin axis is the plane normal.
        let thin = g.scale[..3].iter().position(|&v| v < 1e-5).unwrap();
        assert!(splat_axes(g.rotation).col(thin).dot(DVec3::Z).abs() > 0.999);
    }

    #[test]
    fn checkerboard_does_not_merge() {
        let (v, grid) = plane(8, |x, y| {
            let c = ((x + y) % 2) as f32;
            [c, c, c, 1.0]
        });
        let (out, _) = merge(&v, &grid, &MergeSettings::default());
        assert_eq!(out.len(), v.len());
    }

    #[test]
    fn edges_layers_and_duplicates() {
        // Two differently coloured halves: each 4x8 half merges to two 4x4 blocks.
        let (mut v, grid) = plane(8, |x, _| if x < 4 { [1.0, 0.0, 0.0, 1.0] } else { [0.0, 0.0, 1.0, 1.0] });
        let (out, _) = merge(&v, &grid, &MergeSettings::default());
        assert_eq!(out.len(), 4);
        // A second surface far behind cell (0, 0) is its own layer and does not
        // block the front one.
        let mut far = v[0];
        far.position[2] = 5.0;
        v.push(far);
        let (out, _) = merge(&v, &grid, &MergeSettings::default());
        assert_eq!(out.len(), 4 + 1);
        // A duplicate at (nearly) the same depth is ambiguous and blocks every
        // block containing it: its 2x2 keeps 2 + 3 splats, the left half's
        // other seven 2x2 blocks give three 2x2 + one 4x4, the right half two 4x4.
        let mut near = v[0];
        near.position[2] = 0.1;
        v.pop();
        v.push(near);
        let (out, _) = merge(&v, &grid, &MergeSettings::default());
        assert_eq!(out.len(), 5 + 3 + 1 + 2);
    }
}
