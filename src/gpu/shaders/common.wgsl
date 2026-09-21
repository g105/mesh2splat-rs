// Shared structs and helpers (port of common.glsl). Prepended to every shader.

struct Gaussian {
    position: vec4<f32>,
    color: vec4<f32>,
    scale: vec4<f32>,
    normal: vec4<f32>,
    rotation: vec4<f32>, // (w, x, y, z)
    pbr: vec4<f32>,      // metallic, roughness
};

struct Vertex {
    position: vec4<f32>,
    normal: vec4<f32>,
    tangent: vec4<f32>,
    uv: vec4<f32>,
};

struct MeshParams {
    bbox_min: vec4<f32>,
    bbox_max: vec4<f32>,
    base_color_factor: vec4<f32>,
    flags: vec4<u32>, // has albedo, has normal, has metallic-roughness, max gaussians
};

// Per-visible-splat data produced by the prepass (QuadNdcTransformation).
struct Quad {
    mean_ndc: vec4<f32>,   // xy = NDC center
    axes_ndc: vec4<f32>,   // xy = major axis, zw = minor axis (NDC)
    color: vec4<f32>,
    conic: vec4<f32>,      // xyz = inverse 2D covariance (a, b, c), w = view depth
    normal: vec4<f32>,     // xyz = encoded normal, w = metallic
    ws_pos: vec4<f32>,     // xyz = world position, w = roughness
};

fn random2d(co: vec2<f32>) -> f32 {
    let dt = dot(co, vec2<f32>(12.9898, 78.233));
    let sn = dt - 3.14 * floor(dt / 3.14);
    return fract(sin(sn) * 43758.5453);
}

// NB: quat is (w, x, y, z), packed in .x .y .z .w
fn cast_quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let first = vec3<f32>(
        1.0 - 2.0 * (q.z * q.z + q.w * q.w),
        2.0 * (q.y * q.z - q.x * q.w),
        2.0 * (q.y * q.w + q.x * q.z));
    let second = vec3<f32>(
        2.0 * (q.y * q.z + q.x * q.w),
        1.0 - 2.0 * (q.y * q.y + q.w * q.w),
        2.0 * (q.z * q.w - q.x * q.y));
    let third = vec3<f32>(
        2.0 * (q.y * q.w - q.x * q.z),
        2.0 * (q.z * q.w + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    return mat3x3<f32>(first, second, third);
}

fn compute_cov3d(rot: mat3x3<f32>, s: vec3<f32>) -> mat3x3<f32> {
    let sm = mat3x3<f32>(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
    let m = sm * rot;
    return transpose(m) * m;
}

fn inverse_mat2(m: mat2x2<f32>) -> mat2x2<f32> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if (det == 0.0) {
        return mat2x2<f32>(0.0, 0.0, 0.0, 0.0);
    }
    return mat2x2<f32>(m[1][1] / det, -m[0][1] / det, -m[1][0] / det, m[0][0] / det);
}

fn exponential_depth(view_depth: f32, near_far: vec2<f32>) -> f32 {
    let nd = clamp((view_depth - near_far.x) / (near_far.y - near_far.x), 0.0, 1.0);
    return clamp(exp(-20.0 * nd), 0.0, 1.0);
}

fn encode_normal(n: vec3<f32>) -> vec3<f32> {
    return n * 0.5 + 0.5;
}

fn decode_normal(n: vec3<f32>) -> vec3<f32> {
    return n * 2.0 - 1.0;
}

// Screen-space EWA projection shared by the main and shadow prepasses.
// Returns false when the splat should be culled. On success writes
// (major axis, minor axis) in NDC to `axes` and the conic to `conic`.
struct Projected {
    ok: bool,
    axes: vec4<f32>,
    conic: vec3<f32>,
};

fn project_cov(cov3d: mat3x3<f32>, view: mat4x4<f32>, proj: mat4x4<f32>, p_view: vec3<f32>, resolution: vec2<f32>) -> Projected {
    var out: Projected;
    out.ok = false;
    // https://github.com/graphdeco-inria/diff-gaussian-rasterization forward.cu#L74
    let tz = p_view.z;
    let tz2 = tz * tz;
    let jsx = -(proj[0][0] * resolution.x) / (2.0 * tz);
    let jsy = -(proj[1][1] * resolution.y) / (2.0 * tz);
    let jtx = (proj[0][0] * p_view.x * resolution.x) / (2.0 * tz2);
    let jty = (proj[1][1] * p_view.y * resolution.y) / (2.0 * tz2);
    let j = mat3x3<f32>(vec3<f32>(jsx, 0.0, 0.0), vec3<f32>(0.0, jsy, 0.0), vec3<f32>(jtx, jty, 0.0));
    let w = mat3x3<f32>(view[0].xyz, view[1].xyz, view[2].xyz);
    let jw = j * w;
    let v = jw * cov3d * transpose(jw);

    // low-pass filter on the 2x2 block
    let c00 = v[0][0] + 0.3;
    let c01 = v[0][1];
    let c11 = v[1][1] + 0.3;

    let mid = c00 + c11;
    let delta = length(vec2<f32>(c00 - c11, 2.0 * c01));
    let l1 = 0.5 * (mid + delta);
    let l2 = 0.5 * (mid - delta);
    if (l2 < 0.0) {
        return out;
    }
    // Eigenvector of l1. The ratio below is the mediant of the two textbook
    // forms ((l1-a)/b and b/(l1-c)); guard the degenerate cases that make the
    // original produce NaNs (e.g. a perfectly isotropic footprint).
    let den = c01 - c11 + l1;
    var diag: vec2<f32>;
    if (abs(den) > 1e-12) {
        diag = normalize(vec2<f32>(1.0, (-c00 + c01 + l1) / den));
    } else if (abs(c01) > 1e-12) {
        diag = normalize(vec2<f32>(1.0, -1.0));
    } else if (c00 >= c11) {
        diag = vec2<f32>(1.0, 0.0);
    } else {
        diag = vec2<f32>(0.0, 1.0);
    }
    let major = min(3.0 * sqrt(l1), 1024.0) * diag;
    let minor = min(3.0 * sqrt(l2), 1024.0) * vec2<f32>(diag.y, -diag.x);
    let half_res = resolution * 0.5;
    out.axes = vec4<f32>(major / half_res, minor / half_res);
    let inv = inverse_mat2(mat2x2<f32>(c00, c01, c01, c11));
    out.conic = vec3<f32>(inv[0][0], inv[0][1], inv[1][1]);
    out.ok = true;
    return out;
}
