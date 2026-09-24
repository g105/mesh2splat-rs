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
@group(0) @binding(10) var opacity_map: texture_2d_array<f32>;
@group(0) @binding(11) var s_tangent: texture_2d<f32>;
@group(0) @binding(12) var m_tangent: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
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

/// Light left after the splats between `pos` and the light absorbed their
/// share (the opacity slices in front of the fragment).
fn optical_depth(pos: vec3<f32>) -> f32 {
    let to_frag = pos - lt.light_pos.xyz;
    let dist = length(to_frag);
    let dir = to_frag / max(dist, 1e-6);
    let face = face_index(dir);
    let clip = lt.face_view_proj[face] * vec4<f32>(lt.light_pos.xyz + dir, 1.0);
    let ndc = clip.xy / clip.w;
    let res = i32(lt.shadow_res);
    let px = vec2<i32>(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - 0.5 * ndc.y) * f32(res));
    let slices = textureLoad(opacity_map, clamp(px, vec2<i32>(0), vec2<i32>(res - 1)), i32(face), 0);
    return slices_to_optical_depth(slices, dist, lt.far_plane);
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
    var tan_s: vec4<f32>;
    if (use_mesh) {
        pos_s = textureLoad(m_pos, px, 0);
        nrm_s = textureLoad(m_normal, px, 0);
        alb_s = textureLoad(m_albedo, px, 0);
        mr_s = textureLoad(m_mr, px, 0);
        tan_s = textureLoad(m_tangent, px, 0);
    } else {
        pos_s = textureLoad(s_pos, px, 0);
        nrm_s = textureLoad(s_normal, px, 0);
        alb_s = textureLoad(s_albedo, px, 0);
        mr_s = textureLoad(s_mr, px, 0);
        tan_s = textureLoad(s_tangent, px, 0);
    }
    let alpha = alb_s.a;
    let eps = 1e-6;

    if (lt.render_mode == 5u) {
        let mr = mr_s.rg / max(mr_s.a, eps);
        return over_background(vec3<f32>(mr, 0.0) * alpha, alpha);
    }
    // In forward mode the splat pass shaded each splat already.
    if (lt.render_mode != 6u || lt.lighting == 0u || lt.forward == 1u || alpha <= eps) {
        return over_background(alb_s.rgb, alpha);
    }

    let albedo_gamma = alb_s.rgb / max(alpha, eps);
    let albedo = pow(albedo_gamma, vec3<f32>(2.2));
    let pos = pos_s.xyz / max(pos_s.a, eps);
    let n = normalize(decode_normal(nrm_s.xyz / max(nrm_s.a, eps)));
    let mr = mr_s.rgb / max(mr_s.a, eps);
    let metallic = mr.x;
    let roughness = mr.y;
    let ao = mr.z;

    // Graded (opacity) or binary (depth) self-shadowing.
    var shadow = 0.0;
    var through = 0.0;
    var tint = vec3<f32>(1.0);
    if (lt.opacity_shadows == 1u) {
        let optical = optical_depth(pos);
        let t = exp(-lt.shadow_density * optical);
        shadow = 1.0 - t;
        through = t * lt.transmission;
        tint = attenuation_tint(lt.attenuation.rgb, lt.attenuation.w, optical);
    } else {
        shadow = shadow_factor(pos);
    }
    let l = normalize(lt.light_pos.xyz - pos);
    let v = normalize(lt.cam_pos.xyz - pos);
    let h = normalize(v + l);
    let d = length(lt.light_pos.xyz - pos);
    let radiance = lt.light_color.rgb * lt.light_color.w / (d * d);

    // Strands crossing at a pixel can average to nothing: any direction in
    // the plane of the normal is then as good as another. So can a G-buffer
    // without the tangent, which only shading that reads it pays for.
    let has_tangent = lt.tangents == 1u && tan_s.a > eps;
    let t_avg = select(vec3<f32>(0.0), tan_s.xyz / max(tan_s.a, eps) * 2.0 - 1.0, has_tangent);
    let t_flat = t_avg - n * dot(n, t_avg);
    let t = select(plane_basis(n)[0], normalize(t_flat), length(t_flat) > 1e-3);
    var lo = vec3<f32>(0.0);
    if (lt.hair == 1u) {
        lo = hair_lighting(t, n, v, l, albedo, lt.fibre.yz, f0_from_ior(lt.fibre.x), lt.fibre.w);
    } else {
        lo = pbr_lighting(n, t, v, l, albedo, metallic, roughness, lt.fibre.x, lt.aniso.x);
    }
    lo += sheen_lighting(n, v, l, lt.sheen.rgb, lt.sheen.w);
    lo *= radiance * (1.0 - shadow);
    // Light that came through the splats in front lights this one from behind.
    if (through > 0.0) {
        lo += transmission_term(v, l, albedo, through, tint) * radiance;
    }
    // Ambient only reaches a pixel as far as the bake says it is open. (The
    // forward path also weighs it by the bent normal, which there is no room
    // for here.)
    let color = tonemap(vec3<f32>(0.3 * ao) * albedo + lo);
    return over_background(color * alpha, alpha);
}
