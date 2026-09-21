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
        normal_ws = vec4<f32>(encode_normal(rot[min_idx]), g.color.a);
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

    let p = project_cov(cov3d, frame.world_to_view, frame.view_to_clip, vs.xyz, frame.resolution);
    if (!p.ok) {
        return;
    }

    let idx = atomicAdd(&counters[0], 1u);
    if (idx >= arrayLength(&quads)) {
        return;
    }
    var q: Quad;
    q.mean_ndc = vec4<f32>(clip_pos.xyz / clip_pos.w, 1.0);
    q.axes_ndc = p.axes;
    q.color = color;
    q.conic = vec4<f32>(p.conic, -vs.z);
    q.normal = vec4<f32>(normal_ws.xyz, g.pbr.x);
    q.ws_pos = vec4<f32>(ws.xyz, g.pbr.y);
    quads[idx] = q;
    // Negative view z: larger magnitude => larger bit pattern, so ascending
    // u32 order is front-to-back.
    keys[idx] = bitcast<u32>(vs.z);
    vals[idx] = idx;
}
