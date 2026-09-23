// Shading shared by the deferred pass and the forward (per-splat) path.
// Only pure functions live here: the two paths bind their textures and
// uniforms differently, so each does its own lookups and calls in.

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
    hair: u32,               // anisotropic (Kajiya-Kay) shading
    opacity_shadows: u32,    // graded shadow from accumulated opacity
    forward: u32,            // splats shaded themselves
    transmission: f32,       // how much light comes through the splats
    shadow_density: f32,     // accumulated opacity -> optical depth
    _p0: u32,
    // rgb = colour left after `w` worth of optical depth (Beer-Lambert).
    attenuation: vec4<f32>,
};

const PI: f32 = 3.14159265;

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

/// Kajiya-Kay: hair scatters around the fibre, so both terms use the angle to
/// the tangent instead of the normal. Two specular lobes, shifted along the
/// normal, give the primary white highlight and the dimmer tinted secondary.
fn hair_lighting(
    t: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    albedo: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    let sin_tl = sqrt(max(1.0 - dot(t, l) * dot(t, l), 0.0));
    let h = normalize(v + l);
    let t1 = normalize(t - n * 0.08);
    let t2 = normalize(t + n * 0.12);
    let e1 = mix(160.0, 24.0, clamp(roughness, 0.0, 1.0));
    let sin_t1 = sqrt(max(1.0 - dot(t1, h) * dot(t1, h), 0.0));
    let sin_t2 = sqrt(max(1.0 - dot(t2, h) * dot(t2, h), 0.0));
    let primary = pow(sin_t1, e1);
    let secondary = pow(sin_t2, e1 * 0.35);
    return albedo * sin_tl + vec3<f32>(primary * 0.35) + albedo * secondary * 0.2;
}

/// Standard metal/rough PBR.
fn pbr_lighting(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let h = normalize(v + l);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let ndf = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, v, l, roughness);
    let specular = ndf * g * f / (4.0 * max(dot(n, v), 0.0) * max(dot(n, l), 0.0) + 0.0001);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    return (kd * albedo / PI + specular) * max(dot(n, l), 0.0);
}

/// Opacity shadow slices in front of a fragment -> how much light survives.
/// Opacity accumulated in the slices in front of a fragment.
fn slices_to_optical_depth(slices: vec4<f32>, dist: f32, far_plane: f32) -> f32 {
    let s = clamp(dist / far_plane, 0.0, 1.0) * 3.0;
    let k = floor(s);
    let frac = s - k;
    var optical = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let w = clamp(k - f32(i), 0.0, 1.0) + select(0.0, frac, f32(i) == k);
        optical += slices[i] * w;
    }
    return optical;
}

fn slices_to_transmittance(slices: vec4<f32>, dist: f32, far_plane: f32, density: f32) -> f32 {
    return exp(-density * slices_to_optical_depth(slices, dist, far_plane));
}

/// Beer-Lambert through the splats in front: `attenuation` is the colour left
/// after light travelled `distance` worth of them, so the tint deepens with
/// optical depth. Blonde hair reddens towards the tips this way; dark hair
/// simply stays dark.
fn attenuation_tint(attenuation: vec3<f32>, distance: f32, optical: f32) -> vec3<f32> {
    let depth = optical / max(distance, 1e-4);
    return pow(max(attenuation, vec3<f32>(1e-4)), vec3<f32>(depth));
}

/// Light scattered forward through the splats in front, for backlit hair.
/// `tint` is what the splats in between left of it.
fn transmission_term(
    v: vec3<f32>,
    l: vec3<f32>,
    albedo: vec3<f32>,
    through: f32,
    tint: vec3<f32>,
) -> vec3<f32> {
    let back = pow(clamp(dot(-v, l) * 0.5 + 0.5, 0.0, 1.0), 3.0);
    return albedo * tint * back * through;
}

/// Tone map and gamma, shared so both paths match.
fn tonemap(color: vec3<f32>) -> vec3<f32> {
    return pow(color / (color + vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}
