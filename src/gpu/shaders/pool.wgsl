// GPU version of `merge_occluded`: pool the splats the occlusion bake found
// buried into coarse ones that fill the same volume.
//
// Rather than accumulate per cell with atomics — WGSL has no float atomics, and
// fixed point across quantities this different in magnitude is a trap — this
// keys each buried splat by its cell, sorts, and gives one thread each run of
// equal keys. A thread then accumulates its whole cluster in registers, which
// is both exact and free of contention.

struct PoolParams {
    /// Grid origin and cell size.
    bbox_min: vec4<f32>,
    cell: vec4<f32>,
    /// x = grid side, y = splat count, z = smallest cluster worth pooling,
    /// w = clusters found
    dims: vec4<u32>,
    /// x = occlusion at or below which a splat counts as buried,
    /// y = fixed-point scale for the size reduction, z = model size
    tune: vec4<f32>,
};

struct Run {
    start: u32,
    len: u32,
};

const NO_KEY: u32 = 0xffffffffu;
const NO_GRID: u32 = 0xc0000000u;
/// Bounds the walk when a cell holds an unreasonable number of splats.
const MAX_RUN: u32 = 4096u;

@group(0) @binding(0) var<uniform> P: PoolParams;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> vals: array<u32>;
@group(0) @binding(4) var<storage, read_write> runs: array<Run>;
/// 1 once a splat has been pooled into a cluster.
@group(0) @binding(5) var<storage, read_write> used: array<u32>;
/// 0 = clusters, 1 = output splats, 2 = splats pooled away
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> out_gaussians: array<Gaussian>;

fn thread(gid3: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid3.x + gid3.y * nwg.x * 256u;
}

/// The two in-plane standard deviations, largest first.
fn splat_face(scale: vec3<f32>) -> vec2<f32> {
    var s = scale;
    if (s.x < s.y) { let t = s.x; s.x = s.y; s.y = t; }
    if (s.y < s.z) { let t = s.y; s.y = s.z; s.z = t; }
    if (s.x < s.y) { let t = s.x; s.x = s.y; s.y = t; }
    return s.xy;
}

/// How much light a splat stops: opacity over its face.
fn extinction(g: Gaussian) -> f32 {
    let f = splat_face(g.scale.xyz);
    return g.color.a * 3.14159265 * f.x * f.y;
}

/// Mean splat size, so the cell can follow the splats rather than the model.
/// WGSL has no float atomics, so this accumulates fixed point — as a fraction
/// of the model, scaled so that the sum of every splat still fits in a u32.
@compute @workgroup_size(256)
fn measure(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = thread(gid3, nwg);
    if (i >= P.dims.y) {
        return;
    }
    let f = splat_face(gaussians[i].scale.xyz);
    let fraction = clamp(f.x / max(P.tune.z, 1e-12), 0.0, 1.0);
    atomicAdd(&counters[3], u32(fraction * P.tune.y));
}

@compute @workgroup_size(256)
fn key(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = thread(gid3, nwg);
    if (i >= P.dims.y) {
        return;
    }
    let g = gaussians[i];
    vals[i] = i;
    used[i] = 0u;
    // `pbr.z` is 0 for splats that were never baked: unknown, not buried.
    if (g.pbr.z <= 0.0 || g.pbr.z > P.tune.x) {
        keys[i] = NO_KEY;
        return;
    }
    let n = i32(P.dims.x);
    let c = clamp(
        vec3<i32>(floor((g.position.xyz - P.bbox_min.xyz) / P.cell.xyz)),
        vec3<i32>(0),
        vec3<i32>(n - 1));
    keys[i] = u32((c.z * n + c.y) * n + c.x);
}

@compute @workgroup_size(256)
fn mark_runs(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let j = thread(gid3, nwg);
    if (j >= P.dims.y) {
        return;
    }
    let k = keys[j];
    if (k == NO_KEY) {
        return;
    }
    // Only the thread at the start of a run does the work.
    if (j > 0u && keys[j - 1u] == k) {
        return;
    }
    var len = 1u;
    while (j + len < P.dims.y && len < MAX_RUN && keys[j + len] == k) {
        len += 1u;
    }
    if (len < P.dims.z) {
        return; // too few to be worth pooling
    }
    let id = atomicAdd(&counters[0], 1u);
    runs[id] = Run(j, len);
    atomicAdd(&counters[2], len);
    for (var m = 0u; m < len; m++) {
        used[vals[j + m]] = 1u;
    }
}

@compute @workgroup_size(64)
fn emit_clusters(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let c = gid3.x + gid3.y * nwg.x * 64u;
    if (c >= P.dims.w) {
        return;
    }
    let run = runs[c];

    // One thread owns the whole cluster, so this accumulates in registers.
    var weight = 0.0;
    var mean = vec3<f32>(0.0);
    var color = vec3<f32>(0.0);
    var normal = vec3<f32>(0.0);
    var pbr = vec2<f32>(0.0);
    var ao = 0.0;
    var stops = 0.0;
    for (var m = 0u; m < run.len; m++) {
        let g = gaussians[vals[run.start + m]];
        let w = max(extinction(g), 1e-12);
        weight += w;
        mean += g.position.xyz * w;
        color += g.color.rgb * w;
        normal += g.normal.xyz * w;
        pbr += g.pbr.xy * w;
        ao += g.pbr.z * w;
        stops += extinction(g);
    }
    let inv = 1.0 / weight;
    mean *= inv;

    // The cluster's own spread plus each member's shape: a blob filling the
    // volume the strands occupied.
    var cov = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    for (var m = 0u; m < run.len; m++) {
        let g = gaussians[vals[run.start + m]];
        let w = max(extinction(g), 1e-12);
        let q = g.rotation / max(length(g.rotation), 1e-20);
        let rot = cast_quat_to_mat3(q);
        let d = g.position.xyz - mean;
        cov += (compute_cov3d(rot, g.scale.xyz) + mat3x3<f32>(d * d.x, d * d.y, d * d.z)) * w;
    }
    cov *= inv;

    let e = eigen_sym(cov);
    var vecs = e.vectors;
    if (determinant(vecs) < 0.0) {
        vecs[2] = -vecs[2];
    }
    let scale = max(sqrt(max(e.values, vec3<f32>(0.0))), vec3<f32>(1e-7));
    // Keep the volume as opaque as the strands were: a much bigger face needs
    // proportionally less opacity to stop the same light.
    let face = splat_face(scale);
    let area = 3.14159265 * face.x * face.y;
    let alpha = clamp(stops / max(area, 1e-12), 0.0, 1.0);
    let q = normalize(quat_cast(vecs));

    var g: Gaussian;
    g.position = vec4<f32>(mean, 1.0);
    g.color = vec4<f32>(color * inv, alpha);
    g.scale = vec4<f32>(scale, bitcast<f32>(NO_GRID));
    let n = normal * inv;
    g.normal = vec4<f32>(select(vec3<f32>(0.0, 0.0, 1.0), normalize(n), length(n) > 1e-12), 0.0);
    g.rotation = vec4<f32>(q.w, q.x, q.y, q.z);
    g.pbr = vec4<f32>(pbr * inv, ao * inv, 0.0);
    out_gaussians[atomicAdd(&counters[1], 1u)] = g;
}

@compute @workgroup_size(256)
fn emit_kept(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = thread(gid3, nwg);
    if (i >= P.dims.y || used[i] != 0u) {
        return;
    }
    out_gaussians[atomicAdd(&counters[1], 1u)] = gaussians[i];
}
