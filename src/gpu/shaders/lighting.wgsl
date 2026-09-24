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
    tangents: u32,           // 1 = the G-buffer has the strand tangent
    // rgb = colour left after `w` worth of optical depth (Beer-Lambert).
    attenuation: vec4<f32>,
    // x = IOR, y = lengthwise roughness, z = crosswise roughness, w = lobe shift
    fibre: vec4<f32>,
    // rgb = sheen colour, w = sheen roughness
    sheen: vec4<f32>,
    // x = anisotropy (0 = isotropic GGX)
    aniso: vec4<f32>,
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

/// Reflectance at normal incidence from the index of refraction. Hair is
/// about 1.55, most dielectrics 1.5, which is where the usual 0.04 comes from.
fn f0_from_ior(ior: f32) -> f32 {
    let r = (ior - 1.0) / (ior + 1.0);
    return r * r;
}

/// Roughness -> specular exponent.
fn lobe_exponent(roughness: f32) -> f32 {
    return mix(180.0, 8.0, clamp(roughness, 0.0, 1.0));
}

/// Hair scatters around the fibre, so the terms use the angle to the tangent
/// rather than the normal (Kajiya-Kay). Two lobes shifted along the normal:
/// the primary is surface reflection and stays the colour of the light, the
/// secondary has passed through the fibre and carries its colour. `rough` is
/// (lengthwise, crosswise) — a highlight that is sharp along the strand and
/// broad across it is what makes hair look like hair.
fn hair_lighting(
    t: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    albedo: vec3<f32>,
    rough: vec2<f32>,
    f0: f32,
    shift: f32,
) -> vec3<f32> {
    let sin_tl = sqrt(max(1.0 - dot(t, l) * dot(t, l), 0.0));
    let h = normalize(v + l);
    let t1 = normalize(t - n * shift);
    let t2 = normalize(t + n * shift * 1.5);
    let sin_t1 = sqrt(max(1.0 - dot(t1, h) * dot(t1, h), 0.0));
    let sin_t2 = sqrt(max(1.0 - dot(t2, h) * dot(t2, h), 0.0));
    // Fresnel on the primary, so the highlight fires at grazing angles.
    let fresnel = f0 + (1.0 - f0) * pow(clamp(1.0 - max(dot(h, v), 0.0), 0.0, 1.0), 5.0);
    let primary = pow(sin_t1, lobe_exponent(rough.x)) * fresnel * 8.0;
    let secondary = pow(sin_t2, lobe_exponent(rough.y)) * 0.2;
    return albedo * sin_tl + vec3<f32>(primary) + albedo * secondary;
}

/// Charlie sheen: the soft rim fibrous surfaces show at grazing angles, and
/// what makes fur read as fur rather than plastic.
fn sheen_lighting(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    color: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    if (max(color.r, max(color.g, color.b)) <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = normalize(v + l);
    let inv_a = 1.0 / max(roughness * roughness, 1e-3);
    let sin_h = sqrt(max(1.0 - dot(n, h) * dot(n, h), 0.0));
    let d = (2.0 + inv_a) * pow(sin_h, inv_a) / (2.0 * PI);
    // Ashikhmin's cheap visibility term, which keeps the rim from blowing out.
    let vis = 1.0 / (4.0 * (max(dot(n, l), 0.0) + max(dot(n, v), 0.0)
        - max(dot(n, l), 0.0) * max(dot(n, v), 0.0) + 1e-4));
    return color * d * vis * max(dot(n, l), 0.0);
}

/// GGX stretched along the tangent: `anisotropy` 0 is the isotropic form.
fn distribution_ggx_aniso(
    n: vec3<f32>,
    t: vec3<f32>,
    h: vec3<f32>,
    roughness: f32,
    anisotropy: f32,
) -> f32 {
    if (anisotropy <= 0.0) {
        return distribution_ggx(n, h, roughness);
    }
    let a = roughness * roughness;
    let aspect = sqrt(1.0 - 0.9 * clamp(anisotropy, 0.0, 1.0));
    let ax = max(a / aspect, 1e-4);
    let ay = max(a * aspect, 1e-4);
    let b = normalize(cross(n, t));
    let th = dot(t, h) / ax;
    let bh = dot(b, h) / ay;
    let ndh = dot(n, h);
    let d = th * th + bh * bh + ndh * ndh;
    return 1.0 / (PI * ax * ay * d * d);
}

/// Standard metal/rough PBR, with an optional anisotropic highlight.
fn pbr_lighting(
    n: vec3<f32>,
    t: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ior: f32,
    anisotropy: f32,
) -> vec3<f32> {
    let h = normalize(v + l);
    let f0 = mix(vec3<f32>(f0_from_ior(ior)), albedo, metallic);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let ndf = distribution_ggx_aniso(n, t, h, roughness, anisotropy);
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
