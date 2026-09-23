// Mesh -> gaussian conversion (port of converterVS/GS/FS.glsl).
//
// The original uses a geometry shader to compute per-triangle data and to
// re-project each triangle into a normalized 2D "UV" space (dominant-axis
// planar projection inside the mesh bounding box). WebGPU has no geometry
// shaders, so the vertex shader pulls all three vertices of its triangle from
// a storage buffer and redundantly computes the same per-triangle values.
// Every rasterized fragment then becomes one gaussian.

@group(0) @binding(0) var<storage, read_write> gaussians: array<Gaussian>;
@group(0) @binding(1) var<storage, read_write> counter: atomic<u32>;
// Sampling level drawn by this pass (detail-aware density).
@group(0) @binding(2) var<uniform> pass_level: vec4<u32>;

@group(1) @binding(0) var<uniform> params: MeshParams;
@group(1) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(1) @binding(2) var albedo_tex: texture_2d<f32>;
@group(1) @binding(3) var normal_tex: texture_2d<f32>;
@group(1) @binding(4) var mr_tex: texture_2d<f32>;
@group(1) @binding(5) var mat_sampler: sampler;
@group(1) @binding(6) var<storage, read> tri_levels: array<u32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) @interpolate(flat) scale: vec3<f32>,
    @location(5) @interpolate(flat) quat: vec4<f32>,
    @location(6) @interpolate(flat) axis: u32,
};

// Planar projection onto the plane of the dominant normal axis, normalized by
// the largest bbox extent of the two remaining axes.
fn ortho_uv(p: vec3<f32>, an: vec3<f32>) -> vec2<f32> {
    let bmin = params.bbox_min.xyz;
    let bmax = params.bbox_max.xyz;
    let rel = p - bmin;
    let size = bmax - bmin;
    if (an.x > an.y && an.x > an.z) {
        let r = max(size.y, size.z);
        return vec2<f32>(rel.y, rel.z) / r;
    } else if (an.y > an.z) {
        let r = max(size.x, size.z);
        return vec2<f32>(rel.x, rel.z) / r;
    }
    let r = max(size.x, size.y);
    return vec2<f32>(rel.x, rel.y) / r;
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let tri = vid / 3u;
    let corner = vid % 3u;
    // Detail-aware density: each triangle is rasterized in the pass of its own
    // level, into a target of resolution >> level.
    var level = 0u;
    if (params.detail.w == 1u) {
        level = tri_levels[tri];
    }
    if (level != pass_level.x) {
        var skip: VsOut;
        skip.clip = vec4<f32>(10.0, 10.0, 0.0, 1.0); // outside the clip volume
        return skip;
    }
    let v0 = vertices[tri * 3u];
    let v1 = vertices[tri * 3u + 1u];
    let v2 = vertices[tri * 3u + 2u];
    let p = array<vec3<f32>, 3>(v0.position.xyz, v1.position.xyz, v2.position.xyz);

    var e1 = p[1] - p[0];
    var e2 = p[2] - p[0];
    var e3 = p[2] - p[1];
    if (length(e2) > length(e1) && length(e2) > length(e3)) {
        let t = e1; e1 = e2; e2 = t;
    } else if (length(e3) > length(e1) && length(e3) > length(e2)) {
        let t = e1; e1 = e3; e3 = t;
    }
    e1 = normalize(e1);
    let n = normalize(cross(e1, e2));
    let an = abs(n);

    let uvs = array<vec2<f32>, 3>(ortho_uv(p[0], an), ortho_uv(p[1], an), ortho_uv(p[2], an));

    // Tangent frame: longest edge, bitangent, face normal.
    let x_axis = e1;
    let y_axis = normalize(cross(n, x_axis));
    let q = quat_cast(mat3x3<f32>(x_axis, y_axis, n));

    // Jacobian of the (normalized UV -> 3D) map. Its columns give the 3D
    // footprint of one unit of UV along u and v.
    let uv_m = mat2x2<f32>(uvs[1] - uvs[0], uvs[2] - uvs[0]);
    let v_m = mat2x3<f32>(p[1] - p[0], p[2] - p[0]);
    let j = v_m * inverse_mat2(uv_m);

    var out: VsOut;
    let this_v = array<Vertex, 3>(v0, v1, v2)[corner];
    out.clip = vec4<f32>(uvs[corner] * 2.0 - 1.0, 0.0, 1.0);
    out.position = this_v.position.xyz;
    out.uv = this_v.uv.xy;
    out.tangent = this_v.tangent;
    out.normal = this_v.normal.xyz;
    // Coarser levels carry proportionally larger splats: the renderer scales
    // every splat by std / resolution, which is the level-0 spacing.
    let spread = f32(1u << level);
    out.scale = vec3<f32>(length(j[0]) * spread, length(j[1]) * spread, 1e-7);
    out.quat = vec4<f32>(q.w, q.x, q.y, q.z);
    // Same branch order as ortho_uv.
    if (an.x > an.y && an.x > an.z) {
        out.axis = 0u;
    } else if (an.y > an.z) {
        out.axis = 1u;
    } else {
        out.axis = 2u;
    }
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Sample first: implicit-derivative sampling must be in uniform control flow.
    let albedo_s = textureSample(albedo_tex, mat_sampler, in.uv);
    let normal_s = textureSample(normal_tex, mat_sampler, in.uv).xyz;
    let mr_s = textureSample(mr_tex, mat_sampler, in.uv).bg; // b = metallic, g = roughness

    var color = vec4<f32>(1.0);
    if (params.flags.x == 1u) {
        color = albedo_s;
    }

    var n = in.normal;
    if (params.flags.y == 1u) {
        let mapped = normalize(normal_s * 2.0 - 1.0);
        let bitangent = normalize(cross(in.normal, in.tangent.xyz)) * in.tangent.w;
        let tbn = mat3x3<f32>(in.tangent.xyz, bitangent, normalize(in.normal));
        n = normalize(tbn * mapped);
    }

    var metal_rough = vec2<f32>(0.1, 0.5);
    if (params.flags.z == 1u) {
        metal_rough = mr_s;
    }

    let index = atomicAdd(&counter, 1u);
    if (index < params.flags.w) {
        var g: Gaussian;
        g.position = vec4<f32>(in.position, 1.0);
        g.color = color * params.base_color_factor;
        // Spare .w slots record where the splat sits on the conversion grid
        // (read back by `merge`): axis << 30 | y << 15 | x, and the mesh index.
        // Only level-0 splats sit on the full-resolution grid the merge uses.
        var grid = 0xc0000000u; // axis 3: not on the grid
        if (pass_level.x == 0u) {
            let cell = vec2<u32>(in.clip.xy);
            grid = (in.axis << 30u) | (min(cell.y, 0x7fffu) << 15u) | min(cell.x, 0x7fffu);
        }
        g.scale = vec4<f32>(in.scale, bitcast<f32>(grid));
        g.normal = vec4<f32>(n, bitcast<f32>(u32(params.bbox_min.w)));
        g.rotation = in.quat;
        g.pbr = vec4<f32>(metal_rough, 0.0, 1.0);
        gaussians[index] = g;
    }
    return vec4<f32>(0.0);
}
