// Deferred shading / composite (port of gaussianSplattingDeferredPS.glsl).
//
// Differences from the original, all deliberate:
//  * G-buffer values accumulated with front-to-back blending are normalized
//    by their accumulated alpha before use (the original used them raw, which
//    pulls positions/normals toward zero on semi-transparent edges).
//  * Metallic is read from the channel the splat pass writes it to (the
//    original read `.b`, which is always 0).
//  * The background color is composited using the accumulated alpha.
//  * "Final" without lighting enabled shows unlit albedo.
//  * PI is 3.14159... The original's `#define PI 22.0f/7.0f` (no parentheses)
//    turns `albedo / PI` into `albedo / 22 / 7`, making diffuse ~49x too dark.

struct Lighting {
    light_pos: vec4<f32>,
    cam_pos: vec4<f32>,
    light_color: vec4<f32>,  // rgb, w = intensity
    background: vec4<f32>,
    face_view_proj: array<mat4x4<f32>, 6>,
    far_plane: f32,
    split_x: f32,            // split position in pixels
    render_mode: u32,
    lighting: u32,
    split_enabled: u32,
    shadow_res: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> lt: Lighting;
@group(0) @binding(1) var s_pos: texture_2d<f32>;
@group(0) @binding(2) var s_normal: texture_2d<f32>;
@group(0) @binding(3) var s_albedo: texture_2d<f32>;
@group(0) @binding(4) var s_mr: texture_2d<f32>;
@group(0) @binding(5) var m_pos: texture_2d<f32>;
@group(0) @binding(6) var m_normal: texture_2d<f32>;
@group(0) @binding(7) var m_albedo: texture_2d<f32>;
@group(0) @binding(8) var m_mr: texture_2d<f32>;
@group(0) @binding(9) var shadow_map: texture_depth_2d_array;

const PI: f32 = 3.14159265;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let ndh = max(dot(n, h), 0.0);
    let denom = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometry_schlick_ggx(ndv: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return ndv / (ndv * (1.0 - k) + k);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    return geometry_schlick_ggx(max(dot(n, v), 0.0), roughness) * geometry_schlick_ggx(max(dot(n, l), 0.0), roughness);
}

fn face_index(dir: vec3<f32>) -> u32 {
    let a = abs(dir);
    if (a.x >= a.y && a.x >= a.z) {
        return select(1u, 0u, dir.x > 0.0);
    } else if (a.y >= a.x && a.y >= a.z) {
        return select(3u, 2u, dir.y > 0.0);
    }
    return select(5u, 4u, dir.z > 0.0);
}

// Manual cube lookup into the 6-layer array so the face orientation is exactly
// the one used when rendering the shadow map.
fn closest_depth(dir: vec3<f32>) -> f32 {
    let face = face_index(dir);
    let clip = lt.face_view_proj[face] * vec4<f32>(lt.light_pos.xyz + dir, 1.0);
    let ndc = clip.xy / clip.w;
    let res = i32(lt.shadow_res);
    let px = vec2<i32>(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - 0.5 * ndc.y) * f32(res));
    return textureLoad(shadow_map, clamp(px, vec2<i32>(0), vec2<i32>(res - 1)), i32(face), 0) * lt.far_plane;
}

fn shadow_factor(pos: vec3<f32>) -> f32 {
    var offsets = array<vec3<f32>, 20>(
        vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, -1.0, 1.0), vec3<f32>(-1.0, -1.0, 1.0), vec3<f32>(-1.0, 1.0, 1.0),
        vec3<f32>(1.0, 1.0, -1.0), vec3<f32>(1.0, -1.0, -1.0), vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(-1.0, 1.0, -1.0),
        vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, -1.0, 0.0), vec3<f32>(-1.0, -1.0, 0.0), vec3<f32>(-1.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(-1.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, -1.0), vec3<f32>(-1.0, 0.0, -1.0),
        vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, -1.0, 1.0), vec3<f32>(0.0, -1.0, -1.0), vec3<f32>(0.0, 1.0, -1.0));
    let to_frag = pos - lt.light_pos.xyz;
    let current = length(to_frag);
    let dir = normalize(to_frag);
    let bias = 0.05;
    let disk = 0.025;
    var s = 0.0;
    for (var i = 0; i < 20; i++) {
        if (current - bias > closest_depth(dir + offsets[i] * disk)) {
            s += 1.0;
        }
    }
    return s / 20.0;
}

fn over_background(premul_rgb: vec3<f32>, alpha: f32) -> vec4<f32> {
    let a = clamp(alpha, 0.0, 1.0);
    return vec4<f32>(premul_rgb + lt.background.rgb * (1.0 - a), 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(frag.xy);
    let use_mesh = lt.split_enabled == 1u && frag.x < lt.split_x;
    if (lt.split_enabled == 1u && abs(frag.x - lt.split_x) < 1.0) {
        return vec4<f32>(1.0);
    }

    var pos_s: vec4<f32>;
    var nrm_s: vec4<f32>;
    var alb_s: vec4<f32>;
    var mr_s: vec4<f32>;
    if (use_mesh) {
        pos_s = textureLoad(m_pos, px, 0);
        nrm_s = textureLoad(m_normal, px, 0);
        alb_s = textureLoad(m_albedo, px, 0);
        mr_s = textureLoad(m_mr, px, 0);
    } else {
        pos_s = textureLoad(s_pos, px, 0);
        nrm_s = textureLoad(s_normal, px, 0);
        alb_s = textureLoad(s_albedo, px, 0);
        mr_s = textureLoad(s_mr, px, 0);
    }
    let alpha = alb_s.a;
    let eps = 1e-6;

    if (lt.render_mode == 5u) {
        let mr = mr_s.rg / max(mr_s.a, eps);
        return over_background(vec3<f32>(mr, 0.0) * alpha, alpha);
    }
    if (lt.render_mode != 6u || lt.lighting == 0u || alpha <= eps) {
        return over_background(alb_s.rgb, alpha);
    }

    let albedo_gamma = alb_s.rgb / max(alpha, eps);
    let albedo = pow(albedo_gamma, vec3<f32>(2.2));
    let pos = pos_s.xyz / max(pos_s.a, eps);
    let n = normalize(decode_normal(nrm_s.xyz / max(nrm_s.a, eps)));
    let mr = mr_s.rg / max(mr_s.a, eps);
    let metallic = mr.x;
    let roughness = mr.y;

    let shadow = shadow_factor(pos);
    let l = normalize(lt.light_pos.xyz - pos);
    let v = normalize(lt.cam_pos.xyz - pos);
    let h = normalize(v + l);
    let d = length(lt.light_pos.xyz - pos);
    let radiance = lt.light_color.rgb * lt.light_color.w / (d * d);

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let ndf = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, v, l, roughness);
    let specular = ndf * g * f / (4.0 * max(dot(n, v), 0.0) * max(dot(n, l), 0.0) + 0.0001);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let ndl = max(dot(n, l), 0.0);
    let lo = (kd * albedo / PI + specular) * radiance * ndl * (1.0 - shadow);
    var color = vec3<f32>(0.3) * albedo + lo;
    color = color / (color + vec3<f32>(1.0));
    color = pow(color, vec3<f32>(1.0 / 2.2));
    return over_background(color * alpha, alpha);
}
