// GPU version of `src/merge.rs`: merge alike neighbouring splats of the
// conversion grid into larger ones, one quadtree level per round.
//
// Per level: `prepare` moves every item (a splat or a merged node) to its
// parent cell and writes its depth as the sort key; after the depth sort,
// `rekey` writes the parent cell as the key and a stable sort groups items by
// cell, depth-ordered inside it. `evaluate` then runs one thread per sorted
// position: the thread that starts a depth layer holding exactly one item per
// quadrant tests the block and, if it is uniform, appends a merged node and a
// next-level item. `emit_leaves` / `emit_nodes` finally collect everything
// that was not merged into a bigger block.

struct Item {
    key: u32,       // cell (group, axis, y, x) at the current level
    depth: f32,     // coordinate along the projection axis
    entry: u32,     // splat index, or NODE | node index
    quadrant: u32,  // quarter of the parent cell (set by `prepare`)
};

// Statistics of the original splats inside a merged block. Moments are kept
// centred (mean + scatter) so f32 stays accurate.
struct Node {
    mean: vec4<f32>,      // xyz mean position, w = number of original splats
    scatter0: vec4<f32>,  // centre scatter xx, xy, xz, yy
    scatter1: vec4<f32>,  // yz, zz, level, _
    shape0: vec4<f32>,    // mean unscaled shape covariance xx, xy, xz, yy
    shape1: vec4<f32>,    // yz, zz, normal cone (radians), _
    color: vec4<f32>,
    cmin: vec4<f32>,
    cmax: vec4<f32>,
    normal: vec4<f32>,    // xyz = sum of member normals
    pbr: vec4<f32>,       // mean metallic, roughness | min metallic, roughness
    pbr_max: vec4<f32>,   // max metallic, roughness
};

struct Params {
    n: u32,             // items at this level (nodes for emit_nodes)
    level: u32,
    child_res: u32,     // grid side of the items' cells
    parent_res: u32,
    groups: u32,        // 1 = all meshes share the grid, else one per mesh
    node_capacity: u32,
    leaf_count: u32,
    _p0: u32,
    color_tol: f32,
    normal_tol: f32,    // radians
    flatness: f32,
    _p1: f32,
    // Depth sort key: (depth - depth_min[axis]) * depth_scale[axis] as 16 bits,
    // or the full float order when depth_scale.w == 0.
    depth_min: vec4<f32>,
    depth_scale: vec4<f32>,
};

const NODE: u32 = 0x80000000u;
const NO_KEY: u32 = 0xffffffffu;
const NO_GRID: u32 = 0xc0000000u; // axis 3: not on a conversion grid

@group(0) @binding(0) var<uniform> P: Params;
// World size of one level-0 cell, per group (xyz = projection axis).
@group(0) @binding(1) var<uniform> cell_sizes: array<vec4<f32>, 256>;
@group(0) @binding(2) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(3) var<storage, read_write> items: array<Item>;
@group(0) @binding(4) var<storage, read_write> next_items: array<Item>;
@group(0) @binding(5) var<storage, read_write> keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> vals: array<u32>;
@group(0) @binding(7) var<storage, read_write> nodes: array<Node>;
// [0, leaf_count): splat merged; then one flag per node.
@group(0) @binding(8) var<storage, read_write> used: array<u32>;
// 0 = next-level items, 1 = nodes, 2 = output splats
@group(0) @binding(9) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> out_gaussians: array<Gaussian>;

fn thread_index(gid3: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid3.x + gid3.y * nwg.x * 256u;
}

// Float -> u32 with the same ordering.
fn order_bits(f: f32) -> u32 {
    let b = bitcast<u32>(f);
    return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

@compute @workgroup_size(256)
fn init(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = thread_index(gid3, nwg);
    if (i >= P.leaf_count) {
        return;
    }
    let g = gaussians[i];
    let bits = bitcast<u32>(g.scale.w);
    let axis = bits >> 30u;
    var it: Item;
    it.entry = i;
    it.quadrant = 0u;
    it.depth = 0.0;
    it.key = NO_KEY;
    if (axis < 3u) {
        let x = bits & 0x7fffu;
        let y = (bits >> 15u) & 0x7fffu;
        var group = 0u;
        if (P.groups > 1u) {
            group = bitcast<u32>(g.normal.w);
        }
        it.key = ((group * 3u + axis) * P.child_res + y) * P.child_res + x;
        it.depth = g.position[axis];
    }
    items[i] = it;
}

@compute @workgroup_size(256)
fn prepare(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = thread_index(gid3, nwg);
    if (i >= P.n) {
        return;
    }
    var it = items[i];
    if (it.key != NO_KEY) {
        let r = P.child_res;
        let x = it.key % r;
        let y = (it.key / r) % r;
        let ga = it.key / (r * r);
        it.quadrant = (y & 1u) * 2u + (x & 1u);
        let rp = P.parent_res;
        it.key = (ga * rp + (y >> 1u)) * rp + (x >> 1u);
        items[i] = it;
    }
    if (P.depth_scale.w > 0.0 && it.key != NO_KEY) {
        let axis = (it.key / (P.parent_res * P.parent_res)) % 3u;
        keys[i] = u32(clamp((it.depth - P.depth_min[axis]) * P.depth_scale[axis], 0.0, 65535.0));
    } else if (P.depth_scale.w > 0.0) {
        keys[i] = 0xffffu;
    } else {
        keys[i] = order_bits(it.depth);
    }
    vals[i] = i;
}

@compute @workgroup_size(256)
fn rekey(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let j = thread_index(gid3, nwg);
    if (j >= P.n) {
        return;
    }
    keys[j] = items[vals[j]].key;
}

// --- statistics ------------------------------------------------------------

fn leaf(g: Gaussian) -> Node {
    var n: Node;
    n.mean = vec4<f32>(g.position.xyz, 1.0);
    n.scatter0 = vec4<f32>(0.0);
    n.scatter1 = vec4<f32>(0.0);
    let q = g.rotation / max(length(g.rotation), 1e-20);
    let s = compute_cov3d(cast_quat_to_mat3(q), g.scale.xyz);
    n.shape0 = vec4<f32>(s[0][0], s[1][0], s[2][0], s[1][1]);
    n.shape1 = vec4<f32>(s[2][1], s[2][2], 0.0, 0.0);
    n.color = g.color;
    n.cmin = g.color;
    n.cmax = g.color;
    let len = length(g.normal.xyz);
    n.normal = vec4<f32>(select(vec3<f32>(0.0, 0.0, 1.0), g.normal.xyz / len, len > 0.0), 0.0);
    n.pbr = vec4<f32>(g.pbr.xy, g.pbr.xy);
    n.pbr_max = vec4<f32>(g.pbr.xy, 0.0, 0.0);
    return n;
}

fn child(entry: u32) -> Node {
    if ((entry & NODE) != 0u) {
        return nodes[entry & ~NODE];
    }
    return leaf(gaussians[entry]);
}

fn sym(a: vec4<f32>, b: vec4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        vec3<f32>(a.x, a.y, a.z),
        vec3<f32>(a.y, a.w, b.x),
        vec3<f32>(a.z, b.x, b.y));
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    return select(fallback, v / l, l > 0.0);
}

fn combine(children: array<Node, 4>) -> Node {
    var c = children; // runtime indexing needs a variable
    var w = 0.0;
    var mean = vec3<f32>(0.0);
    for (var k = 0u; k < 4u; k++) {
        w += c[k].mean.w;
        mean += c[k].mean.xyz * c[k].mean.w;
    }
    mean /= w;
    var n: Node;
    n.mean = vec4<f32>(mean, w);
    var scatter = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    var shape = scatter;
    n.color = vec4<f32>(0.0);
    n.cmin = c[0].cmin;
    n.cmax = c[0].cmax;
    var nsum = vec3<f32>(0.0);
    var pbr = vec2<f32>(0.0);
    var pmin = c[0].pbr.zw;
    var pmax = c[0].pbr_max.xy;
    for (var k = 0u; k < 4u; k++) {
        let wk = c[k].mean.w;
        let d = c[k].mean.xyz - mean;
        scatter += (sym(c[k].scatter0, c[k].scatter1) + mat3x3<f32>(d * d.x, d * d.y, d * d.z)) * wk;
        shape += sym(c[k].shape0, c[k].shape1) * wk;
        n.color += c[k].color * wk;
        n.cmin = min(n.cmin, c[k].cmin);
        n.cmax = max(n.cmax, c[k].cmax);
        nsum += c[k].normal.xyz;
        pbr += c[k].pbr.xy * wk;
        pmin = min(pmin, c[k].pbr.zw);
        pmax = max(pmax, c[k].pbr_max.xy);
    }
    scatter *= 1.0 / w;
    shape *= 1.0 / w;
    n.color /= w;
    n.normal = vec4<f32>(nsum, 0.0);
    n.pbr = vec4<f32>(pbr / w, pmin);
    n.pbr_max = vec4<f32>(pmax, 0.0, 0.0);
    let nm = safe_normalize(nsum, vec3<f32>(0.0, 0.0, 1.0));
    var cone = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let ck = safe_normalize(c[k].normal.xyz, nm);
        cone = max(cone, c[k].shape1.z + acos(clamp(dot(ck, nm), -1.0, 1.0)));
    }
    n.scatter0 = vec4<f32>(scatter[0][0], scatter[1][0], scatter[2][0], scatter[1][1]);
    n.scatter1 = vec4<f32>(scatter[2][1], scatter[2][2], f32(P.level), 0.0);
    n.shape0 = vec4<f32>(shape[0][0], shape[1][0], shape[2][0], shape[1][1]);
    n.shape1 = vec4<f32>(shape[2][1], shape[2][2], cone, 0.0);
    return n;
}

struct Eigen {
    values: vec3<f32>,
    vectors: mat3x3<f32>, // columns
};

// Cyclic Jacobi for a symmetric 3x3 (Numerical Recipes' rotation formulas).
fn eigen_sym(m: mat3x3<f32>) -> Eigen {
    var a = array<vec3<f32>, 3>(m[0], m[1], m[2]);
    var v = array<vec3<f32>, 3>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    let scale = max(max(abs(m[0][0]), abs(m[1][1])), max(abs(m[2][2]), 1e-30));
    for (var sweep = 0u; sweep < 8u; sweep++) {
        let off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2]);
        if (off <= 1e-7 * scale) {
            break;
        }
        for (var pair = 0u; pair < 3u; pair++) {
            // pairs (0,1) (0,2) (1,2)
            let p = select(0u, 1u, pair == 2u);
            let q = select(pair + 1u, 2u, pair == 2u);
            let apq = a[p][q];
            if (abs(apq) <= 1e-30) {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            var t = 1.0;
            if (theta != 0.0) {
                t = sign(theta) / (abs(theta) + sqrt(theta * theta + 1.0));
            }
            let c = 1.0 / sqrt(t * t + 1.0);
            let s = t * c;
            let tau = s / (1.0 + c);
            let r = 3u - p - q;
            let arp = a[r][p];
            let arq = a[r][q];
            a[p][p] -= t * apq;
            a[q][q] += t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            a[r][p] = arp - s * (arq + tau * arp);
            a[p][r] = a[r][p];
            a[r][q] = arq + s * (arp - tau * arq);
            a[q][r] = a[r][q];
            // v[col][row]: eigenvector k is v[k].
            let vp = v[p];
            let vq = v[q];
            v[p] = vp - s * (vq + tau * vp);
            v[q] = vq + s * (vp - tau * vq);
        }
    }
    var e: Eigen;
    e.values = vec3<f32>(a[0][0], a[1][1], a[2][2]);
    e.vectors = mat3x3<f32>(v[0], v[1], v[2]);
    return e;
}

fn uniform_block(n: Node, footprint: f32) -> bool {
    let c_spread = n.cmax - n.cmin;
    let p_spread = n.pbr_max.xy - n.pbr.zw;
    if (max(max(c_spread.x, c_spread.y), max(c_spread.z, c_spread.w)) > P.color_tol
        || max(p_spread.x, p_spread.y) > 2.0 * P.color_tol
        || n.shape1.z > P.normal_tol) {
        return false;
    }
    let e = eigen_sym(sym(n.scatter0, n.scatter1));
    let off_plane = sqrt(max(min(min(e.values.x, e.values.y), e.values.z), 0.0));
    return off_plane <= P.flatness * footprint;
}

// --- evaluate ----------------------------------------------------------------

fn footprint_of(key: u32) -> f32 {
    let rp = P.parent_res;
    let ga = key / (rp * rp);
    let cs = cell_sizes[ga / 3u];
    return cs[ga % 3u] * f32(1u << P.level);
}

fn same_layer(prev: Item, next: Item, fp: f32) -> bool {
    return next.key == prev.key && next.depth - prev.depth <= fp;
}

@compute @workgroup_size(256)
fn evaluate(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let j = thread_index(gid3, nwg);
    if (j + 3u >= P.n) {
        return;
    }
    var ch: array<Item, 4>;
    ch[0] = items[vals[j]];
    if (ch[0].key == NO_KEY) {
        return;
    }
    let fp = footprint_of(ch[0].key);
    // Only the first item of a depth layer does the work...
    if (j > 0u && same_layer(items[vals[j - 1u]], ch[0], fp)) {
        return;
    }
    // ...and only for layers of exactly four.
    for (var k = 1u; k < 4u; k++) {
        ch[k] = items[vals[j + k]];
        if (!same_layer(ch[k - 1u], ch[k], fp)) {
            return;
        }
    }
    if (j + 4u < P.n && same_layer(ch[3], items[vals[j + 4u]], fp)) {
        return;
    }
    var quadrants = 0u;
    for (var k = 0u; k < 4u; k++) {
        quadrants |= 1u << ch[k].quadrant;
    }
    if (quadrants != 15u) {
        return; // two members in one cell
    }

    // Cheap colour / material test before the full statistics.
    var cmin = vec4<f32>(3.0e38);
    var cmax = vec4<f32>(-3.0e38);
    var pmin = vec2<f32>(3.0e38);
    var pmax = vec2<f32>(-3.0e38);
    for (var k = 0u; k < 4u; k++) {
        let e = ch[k].entry;
        if ((e & NODE) != 0u) {
            let nd = &nodes[e & ~NODE];
            cmin = min(cmin, (*nd).cmin);
            cmax = max(cmax, (*nd).cmax);
            pmin = min(pmin, (*nd).pbr.zw);
            pmax = max(pmax, (*nd).pbr_max.xy);
        } else {
            let g = &gaussians[e];
            cmin = min(cmin, (*g).color);
            cmax = max(cmax, (*g).color);
            pmin = min(pmin, (*g).pbr.xy);
            pmax = max(pmax, (*g).pbr.xy);
        }
    }
    let cs = cmax - cmin;
    let ps = pmax - pmin;
    if (max(max(cs.x, cs.y), max(cs.z, cs.w)) > P.color_tol || max(ps.x, ps.y) > 2.0 * P.color_tol) {
        return;
    }

    let merged = combine(array<Node, 4>(child(ch[0].entry), child(ch[1].entry), child(ch[2].entry), child(ch[3].entry)));
    if (!uniform_block(merged, fp)) {
        return;
    }
    let idx = atomicAdd(&counters[1], 1u);
    if (idx >= P.node_capacity) {
        return;
    }
    nodes[idx] = merged;
    for (var k = 0u; k < 4u; k++) {
        let e = ch[k].entry;
        if ((e & NODE) != 0u) {
            used[P.leaf_count + (e & ~NODE)] = 1u;
        } else {
            used[e] = 1u;
        }
    }
    let axis = (ch[0].key / (P.parent_res * P.parent_res)) % 3u;
    var next: Item;
    next.key = ch[0].key;
    next.depth = merged.mean[axis];
    next.entry = NODE | idx;
    next.quadrant = 0u;
    next_items[atomicAdd(&counters[0], 1u)] = next;
}

// --- output ------------------------------------------------------------------

// GLM quat_cast (same as convert.wgsl). Returns (x, y, z, w).
fn quat_cast(m: mat3x3<f32>) -> vec4<f32> {
    let fx = m[0][0] - m[1][1] - m[2][2];
    let fy = m[1][1] - m[0][0] - m[2][2];
    let fz = m[2][2] - m[0][0] - m[1][1];
    let fw = m[0][0] + m[1][1] + m[2][2];
    var idx = 0;
    var big = fw;
    if (fx > big) { big = fx; idx = 1; }
    if (fy > big) { big = fy; idx = 2; }
    if (fz > big) { big = fz; idx = 3; }
    let v = sqrt(big + 1.0) * 0.5;
    let mult = 0.25 / v;
    if (idx == 0) {
        return vec4<f32>((m[1][2] - m[2][1]) * mult, (m[2][0] - m[0][2]) * mult, (m[0][1] - m[1][0]) * mult, v);
    } else if (idx == 1) {
        return vec4<f32>(v, (m[0][1] + m[1][0]) * mult, (m[2][0] + m[0][2]) * mult, (m[1][2] - m[2][1]) * mult);
    } else if (idx == 2) {
        return vec4<f32>((m[0][1] + m[1][0]) * mult, v, (m[1][2] + m[2][1]) * mult, (m[2][0] - m[0][2]) * mult);
    }
    return vec4<f32>((m[2][0] + m[0][2]) * mult, (m[1][2] + m[2][1]) * mult, v, (m[0][1] - m[1][0]) * mult);
}

fn to_splat(n: Node) -> Gaussian {
    let side = f32(1u << u32(n.scatter1.z));
    let e = eigen_sym(sym(n.shape0, n.shape1) * (side * side));
    var vecs = e.vectors;
    if (determinant(vecs) < 0.0) {
        vecs[2] = -vecs[2];
    }
    let q = normalize(quat_cast(vecs));
    var g: Gaussian;
    g.position = vec4<f32>(n.mean.xyz, 1.0);
    g.color = n.color;
    g.scale = vec4<f32>(max(sqrt(max(e.values, vec3<f32>(0.0))), vec3<f32>(1e-7)), bitcast<f32>(NO_GRID));
    g.normal = vec4<f32>(safe_normalize(n.normal.xyz, vec3<f32>(0.0, 0.0, 1.0)), 0.0);
    g.rotation = vec4<f32>(q.w, q.x, q.y, q.z);
    g.pbr = vec4<f32>(n.pbr.xy, 0.0, 1.0);
    return g;
}

@compute @workgroup_size(256)
fn emit_leaves(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = thread_index(gid3, nwg);
    if (i >= P.leaf_count || used[i] != 0u) {
        return;
    }
    out_gaussians[atomicAdd(&counters[2], 1u)] = gaussians[i];
}

@compute @workgroup_size(256)
fn emit_nodes(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let k = thread_index(gid3, nwg);
    if (k >= P.n || used[P.leaf_count + k] != 0u) {
        return;
    }
    out_gaussians[atomicAdd(&counters[2], 1u)] = to_splat(nodes[k]);
}
