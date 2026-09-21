// Splat rasterization into the G-buffer (port of gaussianSplattingVS/PS.glsl).
// Quads are drawn front-to-back with (ONE_MINUS_DST_ALPHA, ONE) blending.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> quads: array<Quad>;
@group(0) @binding(2) var<storage, read> order: array<u32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) color: vec3<f32>,
    @location(1) @interpolate(flat) opacity: f32,
    @location(2) @interpolate(flat) screen: vec2<f32>,
    @location(3) @interpolate(flat) conic: vec3<f32>,
    @location(4) @interpolate(flat) normal: vec3<f32>,
    @location(5) @interpolate(flat) ws_pos: vec3<f32>,
    @location(6) @interpolate(flat) metal_rough: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
    // Two triangles: V0 V1 V2, V0 V2 V3 with V0(-1,-1) V1(-1,1) V2(1,1) V3(1,-1)
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, -1.0));
    let c = corners[vid];
    let q = quads[order[iid]];
    var out: VsOut;
    out.pos = vec4<f32>(q.mean_ndc.xy + c.x * q.axes_ndc.xy + c.y * q.axes_ndc.zw, 0.0, 1.0);
    out.conic = vec3<f32>(-0.5 * q.conic.x, -q.conic.y, -0.5 * q.conic.z);
    out.color = q.color.rgb * q.color.a;
    out.opacity = q.color.a;
    // Bottom-left origin, like gl_FragCoord in the original.
    out.screen = (q.mean_ndc.xy + 1.0) * 0.5 * frame.resolution;
    out.normal = q.normal.xyz;
    out.ws_pos = q.ws_pos.xyz;
    out.metal_rough = vec2<f32>(q.normal.w, q.ws_pos.w);
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
    // WebGPU's framebuffer origin is top-left; flip to the GL convention used for `screen`.
    let frag = vec2<f32>(in.pos.x, frame.resolution.y - in.pos.y);
    let d = in.screen - frag;
    let power = dot(in.conic.xzy, vec3<f32>(d * d, d.x * d.y));
    let g = exp(power);

    var out: GBufferOut;
    if (frame.render_mode == 4u) {
        out.albedo = vec4<f32>(0.01, 0.005, 0.0, 0.01);
    } else {
        out.albedo = vec4<f32>(in.color, in.opacity) * g;
    }
    out.position = vec4<f32>(in.ws_pos, 1.0) * g;
    // Premultiply by opacity so rgb / a in the deferred pass is an opacity-weighted
    // average (the original stored the encoded normal un-premultiplied).
    out.normal = vec4<f32>(in.normal * in.opacity, in.opacity) * g;
    out.metal_rough = vec4<f32>(in.metal_rough, 0.0, 1.0) * g;
    return out;
}
