// Splat rasterization into the G-buffer (port of gaussianSplattingVS/PS.glsl).
// Quads are drawn front-to-back with (ONE_MINUS_DST_ALPHA, ONE) blending.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> quads: array<Quad>;
@group(0) @binding(2) var<storage, read> order: array<u32>;

// Kept small on purpose: tile-based GPUs write every vertex output to memory.
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    // Offset from the splat center in standard deviations along its axes, so
    // the gaussian is exp(-0.5 * |local|^2) with no per-fragment conic math.
    @location(0) local: vec2<f32>,
    @location(1) @interpolate(flat) ws_pos: vec3<f32>,
    @location(2) @interpolate(flat) packed: vec3<u32>, // color, normal, metal_rough
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
    out.packed = vec3<u32>(q.color, q.normal, q.metal_rough);
    return out;
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
        out.albedo = vec4<f32>(color.rgb * opacity, opacity) * g;
    }
    out.position = vec4<f32>(in.ws_pos, 1.0) * g;
    // Premultiply by opacity so rgb / a in the deferred pass is an opacity-weighted
    // average (the original stored the encoded normal un-premultiplied).
    let n = encode_normal(oct_decode(unpack2x16unorm(in.packed.y)));
    out.normal = vec4<f32>(n * opacity, opacity) * g;
    out.metal_rough = vec4<f32>(unpack4x8unorm(in.packed.z).xy, 0.0, 1.0) * g;
    return out;
}
