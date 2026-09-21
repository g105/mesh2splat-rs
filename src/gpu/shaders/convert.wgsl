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

@group(1) @binding(0) var<uniform> params: MeshParams;
@group(1) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(1) @binding(2) var albedo_tex: texture_2d<f32>;
@group(1) @binding(3) var normal_tex: texture_2d<f32>;
@group(1) @binding(4) var mr_tex: texture_2d<f32>;
@group(1) @binding(5) var mat_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) @interpolate(flat) scale: vec3<f32>,
    @location(5) @interpolate(flat) quat: vec4<f32>,
};

// Copied and translated from GLM (quat_cast). Returns (x, y, z, w).
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
    var q: vec4<f32>;
    if (idx == 0) {
        q = vec4<f32>((m[1][2] - m[2][1]) * mult, (m[2][0] - m[0][2]) * mult, (m[0][1] - m[1][0]) * mult, v);
    } else if (idx == 1) {
        q = vec4<f32>(v, (m[0][1] + m[1][0]) * mult, (m[2][0] + m[0][2]) * mult, (m[1][2] - m[2][1]) * mult);
    } else if (idx == 2) {
        q = vec4<f32>((m[0][1] + m[1][0]) * mult, v, (m[1][2] + m[2][1]) * mult, (m[2][0] - m[0][2]) * mult);
    } else {
        q = vec4<f32>((m[2][0] + m[0][2]) * mult, (m[1][2] + m[2][1]) * mult, v, (m[0][1] - m[1][0]) * mult);
    }
    return q;
}

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
    out.scale = vec3<f32>(length(j[0]), length(j[1]), 1e-7);
    out.quat = vec4<f32>(q.w, q.x, q.y, q.z);
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
        g.scale = vec4<f32>(in.scale, 0.0);
        g.normal = vec4<f32>(n, 0.0);
        g.rotation = in.quat;
        g.pbr = vec4<f32>(metal_rough, 0.0, 1.0);
        gaussians[index] = g;
    }
    return vec4<f32>(0.0);
}
