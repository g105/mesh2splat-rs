// Point-light cube shadow map from gaussians (port of gaussianPointShadowMappingCS.glsl
// and gaussianPointLightCubeMapShadow*.glsl). Splats are rendered as opaque ellipses
// into six depth layers; each splat goes to the face its center projects to.

struct ShadowQuad {
    mean_ndc: vec4<f32>,
    axes_ndc: vec4<f32>,
    ws_pos: vec4<f32>,
};

struct ShadowFrame {
    views: array<mat4x4<f32>, 6>,
    proj: mat4x4<f32>,
    model_to_world: mat4x4<f32>,
    inv_model_rot: mat4x4<f32>,
    model_scale: vec4<f32>,
    light_pos: vec4<f32>,
    resolution: vec2<f32>,
    near_far: vec2<f32>,
    std_dev: f32,
    gaussian_count: u32,
    format: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> sf: ShadowFrame;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read_write> squads: array<ShadowQuad>;
@group(0) @binding(3) var<storage, read_write> face_slot: array<u32>;
@group(0) @binding(4) var<storage, read_write> face_counts: array<atomic<u32>, 8>;
@group(0) @binding(5) var<storage, read_write> face_list: array<u32>;
@group(0) @binding(6) var<storage, read_write> draw_args: array<u32>;

const INVALID: u32 = 0xffffffffu;

fn face_index(dir: vec3<f32>) -> u32 {
    let a = abs(dir);
    if (a.x >= a.y && a.x >= a.z) {
        return select(1u, 0u, dir.x > 0.0);
    } else if (a.y >= a.x && a.y >= a.z) {
        return select(3u, 2u, dir.y > 0.0);
    }
    return select(5u, 4u, dir.z > 0.0);
}

fn gid_of(gid3: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid3.x + gid3.y * nwg.x * 256u;
}

@compute @workgroup_size(256)
fn project(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let gid = gid_of(gid3, nwg);
    if (gid >= sf.gaussian_count) {
        return;
    }
    face_slot[gid] = INVALID;
    let g = gaussians[gid];
    let ws = sf.model_to_world * vec4<f32>(g.position.xyz, 1.0);
    let face = face_index(normalize(ws.xyz - sf.light_pos.xyz));
    let view = sf.views[face];
    let vs = view * vec4<f32>(ws.xyz, 1.0);
    let clip_pos = sf.proj * vs;
    let clip = 1.05 * clip_pos.w;
    if (clip_pos.w <= 0.0 || clip_pos.z < 0.0 || clip_pos.x < -clip || clip_pos.x > clip || clip_pos.y < -clip || clip_pos.y > clip) {
        return;
    }
    var multiplier = 1.0;
    if (sf.format == 0u) {
        multiplier = sf.std_dev;
    }
    let ms = sf.model_scale.xyz;
    let scale = g.scale.xyz * multiplier * (ms * ms);
    let inv_rot = mat3x3<f32>(sf.inv_model_rot[0].xyz, sf.inv_model_rot[1].xyz, sf.inv_model_rot[2].xyz);
    let cov3d = compute_cov3d(cast_quat_to_mat3(g.rotation) * inv_rot, scale);
    let p = project_cov(cov3d, view, sf.proj, vs.xyz, sf.resolution);
    if (!p.ok) {
        return;
    }
    var q: ShadowQuad;
    q.mean_ndc = vec4<f32>(clip_pos.xyz / clip_pos.w, 1.0);
    q.axes_ndc = p.axes;
    q.ws_pos = ws;
    squads[gid] = q;
    let slot = atomicAdd(&face_counts[face], 1u);
    face_slot[gid] = face | (slot << 3u);
}

@compute @workgroup_size(256)
fn compact(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let gid = gid_of(gid3, nwg);
    if (gid >= sf.gaussian_count) {
        return;
    }
    let fs = face_slot[gid];
    if (fs == INVALID) {
        return;
    }
    let face = fs & 7u;
    var offset = 0u;
    for (var f = 0u; f < face; f++) {
        offset += atomicLoad(&face_counts[f]);
    }
    face_list[offset + (fs >> 3u)] = gid;
}

@compute @workgroup_size(1)
fn write_args() {
    for (var f = 0u; f < 6u; f++) {
        draw_args[f * 4u + 0u] = 4u; // one triangle strip per splat
        draw_args[f * 4u + 1u] = atomicLoad(&face_counts[f]);
        draw_args[f * 4u + 2u] = 0u;
        draw_args[f * 4u + 3u] = 0u;
    }
}
