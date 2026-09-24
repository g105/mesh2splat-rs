// Per-gaussian projection, culling and 2D covariance (port of gaussianSplattingPrepassCS.glsl).
// Visible splats are appended to `quads`, with their view depth as sort key.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read_write> quads: array<Quad>;
@group(0) @binding(3) var<storage, read_write> keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> vals: array<u32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var mesh_depth: texture_depth_2d;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let gid = gid3.x + gid3.y * nwg.x * 256u;
    if (gid >= frame.gaussian_count) {
        return;
    }
    let g = gaussians[gid];

    let ws = frame.model_to_world * vec4<f32>(g.position.xyz, 1.0);
    let vs = frame.world_to_view * vec4<f32>(ws.xyz, 1.0);
    let clip_pos = frame.view_to_clip * vs;

    let clip = 1.05 * clip_pos.w;
    if (clip_pos.w <= 0.0 || clip_pos.z < 0.0 || clip_pos.x < -clip || clip_pos.x > clip || clip_pos.y < -clip || clip_pos.y > clip) {
        return;
    }

    // Optional occlusion test against the mesh depth buffer (opaque splats of converted meshes only).
    if (frame.depth_test == 1u && g.color.a > 0.95 && frame.format == 0u) {
        let ndc = clip_pos.xy / clip_pos.w;
        let dims = vec2<i32>(textureDimensions(mesh_depth));
        let px = vec2<i32>(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - 0.5 * ndc.y) * vec2<f32>(dims));
        let depth = textureLoad(mesh_depth, clamp(px, vec2<i32>(0), dims - 1), 0);
        let my_depth = clip_pos.z / clip_pos.w;
        if (my_depth > depth + 0.00002) {
            return;
        }
    }

    var multiplier = 1.0;
    if (frame.format == 0u) {
        multiplier = frame.std_dev;
    }
    let ms = frame.model_scale.xyz;
    let scale = g.scale.xyz * multiplier * (ms * ms);

    let inv_rot = mat3x3<f32>(frame.inv_model_rot[0].xyz, frame.inv_model_rot[1].xyz, frame.inv_model_rot[2].xyz);
    let rot = cast_quat_to_mat3(g.rotation) * inv_rot;
    let cov3d = compute_cov3d(rot, scale);

    // Longest axis: the strand / fibre direction, for anisotropic shading.
    var max_idx = 0u;
    if (g.scale.y > g.scale.x && g.scale.y >= g.scale.z) {
        max_idx = 1u;
    } else if (g.scale.z > g.scale.x && g.scale.z > g.scale.y) {
        max_idx = 2u;
    }
    let tangent = splat_axis(rot, max_idx);

    var normal_ws = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    if (frame.format == 0u || frame.ply_has_pbr != 0u) {
        let n = (frame.normal_matrix * vec4<f32>(g.normal.xyz, 1.0)).xyz;
        normal_ws = vec4<f32>(encode_normal(n), g.color.a);
    } else {
        // Shortest axis as normal (https://arxiv.org/pdf/2311.17977, p. 4)
        var min_idx = 0u;
        if (g.scale.y < g.scale.z && g.scale.y < g.scale.x) {
            min_idx = 1u;
        } else if (g.scale.z < g.scale.y && g.scale.z < g.scale.x) {
            min_idx = 2u;
        }
        normal_ws = vec4<f32>(encode_normal(splat_axis(rot, min_idx)), g.color.a);
    }

    var color = g.color;
    let mode = frame.render_mode;
    if (mode == 1u) {
        color = vec4<f32>(vec3<f32>(exponential_depth(-vs.z, frame.near_far)), g.color.a);
    } else if (mode == 2u) {
        color = normal_ws;
    } else if (mode == 3u) {
        let c = vec2<f32>(f32(gid % 4096u), f32(gid / 4096u));
        color = vec4<f32>(random2d(c), random2d(c.yx), random2d(c.yx * 1.234), 1.0);
    }

    // Beyond sqrt(2 ln(255 a)) sigma a splat contributes less than 1/255, so
    // faint splats get smaller quads (capped at the original 3 sigma).
    let alpha = color.a;
    if (alpha < 1.0 / 255.0) {
        return;
    }
    let k = min(1.0, sqrt(2.0 * log(255.0 * alpha)) / 3.0);
    if (k <= 0.0) {
        return;
    }

    let p = project_cov(cov3d, frame.world_to_view, frame.view_to_clip, vs.xyz, frame.resolution);
    if (!p.ok) {
        return;
    }

    let idx = atomicAdd(&counters[0], 1u);
    if (idx >= arrayLength(&quads)) {
        return;
    }
    var q: Quad;
    q.mean_ndc = clip_pos.xy / clip_pos.w;
    q.axes_ndc = vec2<u32>(pack2x16float(p.axes.xy * k), pack2x16float(p.axes.zw * k));
    q.ws_pos = ws.xyz;
    q.extent = pack2x16float(p.extent * k);
    q.color = pack4x8unorm(color);
    q.normal = pack2x16unorm(oct_encode(normalize(decode_normal(normal_ws.xyz))));
    let n_unit = normalize(decode_normal(normal_ws.xyz));
    let t_flat = normalize(tangent - n_unit * dot(tangent, n_unit));
    q.metal_rough = pack_material(g.pbr.xy, n_unit, t_flat);
    // Baked occlusion (pbr.z) and bent normal (pbr.w); unbaked splats, and
    // every splat when occlusion is off, are fully open and bend along their
    // own normal.
    if (frame.use_ao > 0.0 && g.pbr.z > 0.0) {
        let bent = oct_decode(unpack2x16unorm(bitcast<u32>(g.pbr.w)));
        q.ao_bent = pack_ao_bent(bent, g.pbr.z);
    } else {
        q.ao_bent = pack_ao_bent(n_unit, 1.0);
    }
    quads[idx] = q;
    if (frame.sort_scale > 0.0) {
        keys[idx] = u32(clamp((-vs.z - frame.sort_min) * frame.sort_scale, 0.0, 65535.0));
    } else {
        // Negative view z: larger magnitude => larger bit pattern, so ascending
        // u32 order is front-to-back.
        keys[idx] = bitcast<u32>(vs.z);
    }
    vals[idx] = idx;
}
