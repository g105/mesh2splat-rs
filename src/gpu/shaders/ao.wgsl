// Bake ambient occlusion and a bent normal into each splat.
//
// Splat clouds have no surface to trace against, so this voxelizes their
// opacity into a density grid and then, per splat, marches that grid along a
// few directions. The average transmittance is the occlusion; the
// transmittance-weighted mean direction is the bent normal (where light can
// actually reach from). Both are stored in the splat's spare `pbr` channels,
// so shading gets them for free afterwards.

struct AoParams {
    /// Grid origin; cell size, w = what turns stored scales into world ones.
    bbox_min: vec4<f32>,
    cell: vec4<f32>,
    /// x = grid side, y = splat count, z = ray steps, w = directions
    dims: vec4<u32>,
    /// x = density scale, y = occlusion strength, z = step length in cells,
    /// w = where rays leave a splat, in its standard deviations
    tune: vec4<f32>,
};

@group(0) @binding(0) var<uniform> P: AoParams;
@group(0) @binding(1) var<storage, read_write> gaussians: array<Gaussian>;
/// Fixed-point density, so it can be accumulated atomically.
@group(0) @binding(2) var<storage, read_write> grid: array<atomic<u32>>;

const FIXED: f32 = 1024.0;

fn grid_index(c: vec3<i32>) -> u32 {
    let n = i32(P.dims.x);
    let cl = clamp(c, vec3<i32>(0), vec3<i32>(n - 1));
    return u32((cl.z * n + cl.y) * n + cl.x);
}

fn cell_of(p: vec3<f32>) -> vec3<f32> {
    return (p - P.bbox_min.xyz) / P.cell.xyz;
}

fn thread(gid3: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid3.x + gid3.y * nwg.x * 256u;
}

// --- voxelize ----------------------------------------------------------------

@compute @workgroup_size(256)
fn voxelize(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = thread(gid3, nwg);
    if (i >= P.dims.y) {
        return;
    }
    let g = gaussians[i];
    // Sort the axes so the two largest describe the splat's face.
    var s = g.scale.xyz;
    if (s.x < s.y) { let t = s.x; s.x = s.y; s.y = t; }
    if (s.y < s.z) { let t = s.y; s.y = s.z; s.z = t; }
    if (s.x < s.y) { let t = s.x; s.x = s.y; s.y = t; }
    // Opacity times area: how much light this splat stops.
    let mass = g.color.a * 3.14159265 * s.x * s.y * P.tune.x;

    // Spread it over the cells the splat actually covers. Dropping it all in
    // one cell leaves the grid a set of spikes that rays slip between.
    let radius = max(s.x, P.cell.x * 0.5);
    let c = cell_of(g.position.xyz);
    let span = min(vec3<i32>(ceil(vec3<f32>(radius) / P.cell.xyz)), vec3<i32>(3));
    let base = vec3<i32>(floor(c));
    var total = 0.0;
    var weights: array<f32, 343>; // (2 * 3 + 1)^3
    var n = 0u;
    for (var z = -span.z; z <= span.z; z++) {
        for (var y = -span.y; y <= span.y; y++) {
            for (var x = -span.x; x <= span.x; x++) {
                let d = (vec3<f32>(vec3<i32>(x, y, z)) + 0.5 - fract(c)) * P.cell.xyz;
                let w = exp(-0.5 * dot(d, d) / max(radius * radius, 1e-12));
                weights[n] = w;
                total += w;
                n += 1u;
            }
        }
    }
    if (total <= 0.0) {
        return;
    }
    // Density is mass per unit volume, so divide by what a cell holds.
    let cell_volume = P.cell.x * P.cell.y * P.cell.z;
    n = 0u;
    for (var z = -span.z; z <= span.z; z++) {
        for (var y = -span.y; y <= span.y; y++) {
            for (var x = -span.x; x <= span.x; x++) {
                let amount = mass * weights[n] / total / cell_volume;
                n += 1u;
                if (amount > 1e-4) {
                    atomicAdd(&grid[grid_index(base + vec3<i32>(x, y, z))], u32(amount * FIXED));
                }
            }
        }
    }
}

// --- bake --------------------------------------------------------------------

fn density_at(p: vec3<f32>) -> f32 {
    let c = vec3<i32>(floor(cell_of(p)));
    let n = i32(P.dims.x);
    if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= n || c.y >= n || c.z >= n) {
        return 0.0;
    }
    return f32(atomicLoad(&grid[grid_index(c)])) / FIXED;
}

/// Evenly spread directions on the sphere (Fibonacci).
fn ray_direction(i: u32, count: u32) -> vec3<f32> {
    let k = f32(i) + 0.5;
    let phi = acos(1.0 - 2.0 * k / f32(count));
    let theta = 3.883222 * k; // pi * (1 + sqrt(5))
    return vec3<f32>(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
}

@compute @workgroup_size(256)
fn bake(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = thread(gid3, nwg);
    if (i >= P.dims.y) {
        return;
    }
    var g = gaussians[i];
    let origin = g.position.xyz;
    let steps = P.dims.z;
    let dirs = P.dims.w;
    let step = P.cell.x * P.tune.z;

    // The splat's own axes, to find how far it reaches along each ray.
    let rot = cast_quat_to_mat3(g.rotation / max(length(g.rotation), 1e-20));
    let axes = mat3x3<f32>(splat_axis(rot, 0u), splat_axis(rot, 1u), splat_axis(rot, 2u));
    let inv_scale = 1.0 / max(g.scale.xyz * P.cell.w, vec3<f32>(1e-12));

    var open = 0.0;
    var bent = vec3<f32>(0.0);
    for (var d = 0u; d < dirs; d++) {
        let dir = ray_direction(d, dirs);
        var optical = 0.0;
        // Start clear of the splat, or it occludes itself: a step out, or past
        // its own extent along this ray if it reaches further, as a clump does.
        let reach = P.tune.w / max(length((dir * axes) * inv_scale), 1e-12);
        let start = max(step, reach);
        for (var s = 0u; s < steps; s++) {
            optical += density_at(origin + dir * (start + step * f32(s))) * step;
        }
        let t = exp(-P.tune.y * optical);
        open += t;
        bent += dir * t;
    }
    let ao = open / f32(dirs);
    let bent_n = normalize(select(bent, g.normal.xyz, length(bent) < 1e-6));

    // Spare channels: .z = occlusion, .w = bent normal (octahedral, 2 x 16 bit).
    g.pbr.z = ao;
    g.pbr.w = bitcast<f32>(pack2x16unorm(oct_encode(bent_n)));
    gaussians[i] = g;
}
