// Per-triangle sampling level for detail-aware conversion.
//
// The converter gives every triangle the same splat density, so a flat panel
// costs as many splats as a detailed decal. This pass picks, per triangle, the
// coarsest grid level whose splats still reproduce the material: it samples the
// textures at the texel footprint of level 0 and of the candidate level (that
// is, blurred to the coarser splat spacing) and keeps the level while the two
// stay within tolerance. `convert.wgsl` then rasterizes each level into its own
// (smaller) target, which lands on every 2^level-th cell of the same grid.

@group(0) @binding(0) var<uniform> params: MeshParams;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var albedo_tex: texture_2d<f32>;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var mr_tex: texture_2d<f32>;
@group(0) @binding(5) var mat_sampler: sampler;
@group(0) @binding(6) var<storage, read_write> levels: array<u32>;

fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

// Same planar projection as convert.wgsl.
fn ortho_uv(p: vec3<f32>, an: vec3<f32>) -> vec2<f32> {
    let bmin = params.bbox_min.xyz;
    let bmax = params.bbox_max.xyz;
    let rel = p - bmin;
    let size = bmax - bmin;
    if (an.x > an.y && an.x > an.z) {
        return vec2<f32>(rel.y, rel.z) / max(size.y, size.z);
    } else if (an.y > an.z) {
        return vec2<f32>(rel.x, rel.z) / max(size.x, size.z);
    }
    return vec2<f32>(rel.x, rel.y) / max(size.x, size.y);
}

/// Largest material difference between the two mip levels at `uv`.
fn deviation(uv: vec2<f32>, lod_fine: f32, lod_coarse: f32) -> f32 {
    var d = 0.0;
    if (params.flags.x == 1u) {
        let a = textureSampleLevel(albedo_tex, mat_sampler, uv, lod_fine);
        let b = textureSampleLevel(albedo_tex, mat_sampler, uv, lod_coarse);
        let e = abs(a - b);
        d = max(d, max(max(e.x, e.y), max(e.z, e.w)));
    }
    if (params.flags.y == 1u) {
        let a = textureSampleLevel(normal_tex, mat_sampler, uv, lod_fine).xyz;
        let b = textureSampleLevel(normal_tex, mat_sampler, uv, lod_coarse).xyz;
        let e = abs(a - b);
        d = max(d, max(e.x, max(e.y, e.z)));
    }
    if (params.flags.z == 1u) {
        let a = textureSampleLevel(mr_tex, mat_sampler, uv, lod_fine).bg;
        let b = textureSampleLevel(mr_tex, mat_sampler, uv, lod_coarse).bg;
        let e = abs(a - b);
        d = max(d, max(e.x, e.y));
    }
    return d;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let tri = gid3.x + gid3.y * nwg.x * 64u;
    if (tri >= params.detail.y) {
        return;
    }
    let v0 = vertices[tri * 3u];
    let v1 = vertices[tri * 3u + 1u];
    let v2 = vertices[tri * 3u + 2u];
    let p = array<vec3<f32>, 3>(v0.position.xyz, v1.position.xyz, v2.position.xyz);
    let an = abs(normalize(cross(p[1] - p[0], p[2] - p[0])));

    // Grid cells and texels the triangle covers.
    let g0 = ortho_uv(p[0], an);
    let g1 = ortho_uv(p[1], an);
    let g2 = ortho_uv(p[2], an);
    let res = params.detail_tol.y;
    let cells = 0.5 * abs(cross2(g1 - g0, g2 - g0)) * res * res;
    let dims = vec2<f32>(textureDimensions(albedo_tex, 0));
    let texels = 0.5 * abs(cross2(v1.uv.xy - v0.uv.xy, v2.uv.xy - v0.uv.xy)) * dims.x * dims.y;
    if (cells < 4.0) {
        levels[tri] = 0u; // too small to coarsen
        return;
    }
    // Mip level whose texels match one grid cell.
    let lod_base = max(0.5 * log2(max(texels, 1.0) / cells), 0.0);
    // A splat may not outgrow its own triangle.
    let size_cap = u32(max(floor(0.5 * log2(cells)) - 2.0, 0.0));

    // Sample points: centroid, edge midpoints and points near the corners.
    var bary = array<vec3<f32>, 7>(
        vec3<f32>(0.334, 0.333, 0.333),
        vec3<f32>(0.5, 0.5, 0.0),
        vec3<f32>(0.0, 0.5, 0.5),
        vec3<f32>(0.5, 0.0, 0.5),
        vec3<f32>(0.8, 0.1, 0.1),
        vec3<f32>(0.1, 0.8, 0.1),
        vec3<f32>(0.1, 0.1, 0.8));

    var level = 0u;
    let max_level = min(params.detail.z, size_cap);
    for (var l = 1u; l <= max_level; l++) {
        var worst = 0.0;
        for (var k = 0u; k < 7u; k++) {
            let b = bary[k];
            let uv = v0.uv.xy * b.x + v1.uv.xy * b.y + v2.uv.xy * b.z;
            worst = max(worst, deviation(uv, lod_base, lod_base + f32(l)));
        }
        if (worst > params.detail_tol.x) {
            break;
        }
        level = l;
    }
    levels[tri] = level;
}
