// Draw the per-face splat lists into the cube shadow map layers.

struct ShadowQuad {
    mean_ndc: vec4<f32>,
    axes_ndc: vec4<f32>,
    ws_pos: vec4<f32>,
};

struct DrawParams {
    light_pos: vec4<f32>,
    far_plane: f32,
    face: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read> squads: array<ShadowQuad>;
@group(0) @binding(1) var<storage, read> face_list: array<u32>;
@group(0) @binding(2) var<storage, read> face_counts: array<u32, 8>;
@group(1) @binding(0) var<uniform> dp: DrawParams;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) ws_pos: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, -1.0));
    var offset = 0u;
    for (var f = 0u; f < dp.face; f++) {
        offset += face_counts[f];
    }
    let q = squads[face_list[offset + iid]];
    let c = corners[vid];
    var out: VsOut;
    out.pos = vec4<f32>(q.mean_ndc.xy + c.x * q.axes_ndc.xy + c.y * q.axes_ndc.zw, 0.0, 1.0);
    out.ws_pos = q.ws_pos.xyz;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @builtin(frag_depth) f32 {
    return length(in.ws_pos - dp.light_pos.xyz) / dp.far_plane;
}
