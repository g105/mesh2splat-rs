// Splat rasterization into the G-buffer (port of gaussianSplattingVS/PS.glsl).
// Quads are drawn front-to-back with (ONE_MINUS_DST_ALPHA, ONE) blending.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> quads: array<Quad>;
@group(0) @binding(2) var<storage, read> order: array<u32>;

// Forward shading (`fs_forward`) only: each splat shades itself, so a pixel
// blends shaded strands instead of one averaged surface. Uses its own normal
// and tangent rather than the G-buffer's weighted average of everything.
@group(1) @binding(0) var<uniform> lt: Lighting;
@group(1) @binding(1) var opacity_map: texture_2d_array<f32>;

// Kept small on purpose: tile-based GPUs write every vertex output to memory.
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    // Offset from the splat center in standard deviations along its axes, so
    // the gaussian is exp(-0.5 * |local|^2) with no per-fragment conic math.
    @location(0) local: vec2<f32>,
    @location(1) @interpolate(flat) ws_pos: vec3<f32>,
    @location(2) @interpolate(flat) packed: vec4<u32>, // color, normal, metal_rough, ao/bent
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
    // Triangle strip (-1,-1) (1,-1) (-1,1) (1,1).
    let c = vec2<f32>(f32(vid & 1u), f32(vid >> 1u)) * 2.0 - 1.0;
    let q = quads[order[iid]];
    let major = unpack2x16float(q.axes_ndc.x);
    let minor = unpack2x16float(q.axes_ndc.y);
    var out: VsOut;
    out.pos = vec4<f32>(q.mean_ndc + c.x * major + c.y * minor, 0.0, 1.0);
    out.local = c * unpack2x16float(q.extent);
    out.ws_pos = q.ws_pos;
    out.packed = vec4<u32>(q.color, q.normal, q.metal_rough, q.ao_bent);
    return out;
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

/// Opacity between `pos` and the light, from the opacity shadow map.
fn optical_depth(pos: vec3<f32>) -> f32 {
    if (lt.opacity_shadows != 1u) {
        return 0.0;
    }
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

struct GBufferOut {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) albedo: vec4<f32>,
    @location(3) metal_rough: vec4<f32>,
};

@fragment
fn fs_main(in: VsOut) -> GBufferOut {
    let color = unpack4x8unorm(in.packed.x);
    let opacity = color.a;
    let g = exp(-0.5 * dot(in.local, in.local));
    // Skip fragments that would add less than 1/255 (mostly the quad corners).
    if (g * opacity < 1.0 / 255.0) {
        discard;
    }

    var out: GBufferOut;
    if (frame.render_mode == 4u) {
        out.albedo = vec4<f32>(0.01, 0.005, 0.0, 0.01);
    } else {
        let ao = mix(1.0, unpack_ao(in.packed.w), frame.ao_deferred);
        out.albedo = vec4<f32>(color.rgb * ao * opacity, opacity) * g;
    }
    out.position = vec4<f32>(in.ws_pos, 1.0) * g;
    // Premultiply by opacity so rgb / a in the deferred pass is an opacity-weighted
    // average (the original stored the encoded normal un-premultiplied).
    let n = encode_normal(oct_decode(unpack2x16unorm(in.packed.y)));
    out.normal = vec4<f32>(n * opacity, opacity) * g;
    // .z is the tangent angle (see the prepass), averaged like everything else.
    out.metal_rough = vec4<f32>(unpack4x8unorm(in.packed.z).xyz, 1.0) * g;
    return out;
}

// Forward shading: same coverage as `fs_main`, but the colour written is the
// splat's own shaded colour, so the front-to-back blend composites lit strands.
@fragment
fn fs_forward(in: VsOut) -> GBufferOut {
    let color = unpack4x8unorm(in.packed.x);
    let opacity = color.a;
    let g = exp(-0.5 * dot(in.local, in.local));
    if (g * opacity < 1.0 / 255.0) {
        discard;
    }
    let mr = unpack4x8unorm(in.packed.z);
    let n = oct_decode(unpack2x16unorm(in.packed.y));
    let t = decode_tangent(n, mr.z);
    let pos = in.ws_pos;
    let v = normalize(lt.cam_pos.xyz - pos);
    let l = normalize(lt.light_pos.xyz - pos);
    let d = length(lt.light_pos.xyz - pos);
    let radiance = lt.light_color.rgb * lt.light_color.w / (d * d);
    let albedo = pow(color.rgb, vec3<f32>(2.2));

    let optical = optical_depth(pos);
    let through = exp(-lt.shadow_density * optical);
    var lo = vec3<f32>(0.0);
    if (lt.hair == 1u) {
        lo = hair_lighting(t, n, v, l, albedo, lt.fibre.yz, f0_from_ior(lt.fibre.x), lt.fibre.w);
    } else {
        lo = pbr_lighting(n, t, v, l, albedo, mr.x, mr.y, lt.fibre.x, lt.aniso.x);
    }
    lo += sheen_lighting(n, v, l, lt.sheen.rgb, lt.sheen.w);
    lo *= radiance * through;
    if (lt.transmission > 0.0) {
        let tint = attenuation_tint(lt.attenuation.rgb, lt.attenuation.w, optical);
        lo += transmission_term(v, l, albedo, through * lt.transmission, tint) * radiance;
    }
    // Ambient only reaches a splat as far as the bake says it is open, and
    // from the direction that was open.
    let ao = unpack_ao(in.packed.w);
    let bent = unpack_bent(in.packed.w);
    let ambient = 0.3 * ao * mix(0.6, 1.0, clamp(dot(bent, l) * 0.5 + 0.5, 0.0, 1.0));
    let shaded = tonemap(ambient * albedo + lo);

    var out: GBufferOut;
    out.albedo = vec4<f32>(shaded * opacity, opacity) * g;
    out.position = vec4<f32>(pos, 1.0) * g;
    out.normal = vec4<f32>(encode_normal(n) * opacity, opacity) * g;
    out.metal_rough = vec4<f32>(mr.xyz, 1.0) * g;
    return out;
}
