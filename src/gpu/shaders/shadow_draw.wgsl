// Draw the per-face splat lists into the cube shadow map layers.
//
// Two fragment entries share the vertex shader:
//  * `fs_main` writes the nearest distance to the light (a binary shadow test);
//  * `fs_opacity` accumulates how much light each splat absorbs into four
//    distance slices, which gives graded self-shadowing and lets the deferred
//    pass work out how much light survived the strands in between.

struct ShadowQuad {
    mean_ndc: vec4<f32>,
    axes_ndc: vec4<f32>,
    ws_pos: vec4<f32>,
    shape: vec4<f32>, // xy = half-axes in standard deviations, z = opacity
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
    /// Offset from the splat center in standard deviations.
    @location(1) local: vec2<f32>,
    @location(2) @interpolate(flat) opacity: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
    var offset = 0u;
    for (var f = 0u; f < dp.face; f++) {
        offset += face_counts[f];
    }
    let q = squads[face_list[offset + iid]];
    // Triangle strip (-1,-1) (1,-1) (-1,1) (1,1).
    let c = vec2<f32>(f32(vid & 1u), f32(vid >> 1u)) * 2.0 - 1.0;
    var out: VsOut;
    out.pos = vec4<f32>(q.mean_ndc.xy + c.x * q.axes_ndc.xy + c.y * q.axes_ndc.zw, 0.0, 1.0);
    out.ws_pos = q.ws_pos.xyz;
    out.local = c * q.shape.xy;
    out.opacity = q.shape.z;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @builtin(frag_depth) f32 {
    return length(in.ws_pos - dp.light_pos.xyz) / dp.far_plane;
}

// Opacity shadow map: each splat adds what it absorbs to the slice its centre
// falls in, split across the two nearest slices so the transition is smooth.
// The deferred pass sums the slices in front of a fragment to get optical depth.
@fragment
fn fs_opacity(in: VsOut) -> @location(0) vec4<f32> {
    let g = exp(-0.5 * dot(in.local, in.local)) * in.opacity;
    if (g < 1.0 / 255.0) {
        discard;
    }
    let slice = clamp(length(in.ws_pos - dp.light_pos.xyz) / dp.far_plane, 0.0, 1.0) * 3.0;
    let k = floor(slice);
    let frac = slice - k;
    let near = g * (1.0 - frac);
    let far = g * frac;
    // Channel k gets the near part, k + 1 the rest.
    var out = vec4<f32>(0.0);
    if (k < 0.5) {
        out = vec4<f32>(near, far, 0.0, 0.0);
    } else if (k < 1.5) {
        out = vec4<f32>(0.0, near, far, 0.0);
    } else if (k < 2.5) {
        out = vec4<f32>(0.0, 0.0, near, far);
    } else {
        out = vec4<f32>(0.0, 0.0, 0.0, near + far);
    }
    return out;
}
