// Mesh rendering into its own G-buffer (split-screen comparison) and the
// depth-only prepass used for mesh/gaussian occlusion (port of meshRender*.glsl, depthPrepass*.glsl).

@group(0) @binding(0) var<uniform> frame: Frame;

@group(1) @binding(0) var<uniform> params: MeshParams;
@group(1) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(1) @binding(2) var albedo_tex: texture_2d<f32>;
@group(1) @binding(3) var normal_tex: texture_2d<f32>;
@group(1) @binding(4) var mr_tex: texture_2d<f32>;
@group(1) @binding(5) var mat_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) view_depth: f32,
    @location(5) @interpolate(flat) tri: u32,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let v = vertices[vid];
    let world = frame.model_to_world * vec4<f32>(v.position.xyz, 1.0);
    let nm = mat3x3<f32>(frame.normal_matrix[0].xyz, frame.normal_matrix[1].xyz, frame.normal_matrix[2].xyz);
    let view = frame.world_to_view * world;
    var out: VsOut;
    out.pos = frame.view_to_clip * view;
    out.world_pos = world.xyz;
    out.normal = normalize(nm * v.normal.xyz);
    out.tangent = vec4<f32>(normalize(nm * v.tangent.xyz), v.tangent.w);
    out.uv = v.uv.xy;
    out.view_depth = -view.z;
    out.tri = vid / 3u;
    return out;
}

@vertex
fn vs_depth(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let v = vertices[vid];
    return frame.view_to_clip * frame.world_to_view * frame.model_to_world * vec4<f32>(v.position.xyz, 1.0);
}

struct GBufferOut {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) albedo: vec4<f32>,
    @location(3) metal_rough: vec4<f32>,
};

fn hash(x: f32) -> f32 {
    return fract(sin(x) * 43758.5453);
}

@fragment
fn fs_main(in: VsOut) -> GBufferOut {
    let albedo_s = textureSample(albedo_tex, mat_sampler, in.uv);
    let normal_s = textureSample(normal_tex, mat_sampler, in.uv).xyz;
    let mr_s = textureSample(mr_tex, mat_sampler, in.uv).bg;

    var albedo = params.base_color_factor;
    if (params.flags.x == 1u) {
        albedo *= albedo_s;
    }
    var n = normalize(in.normal);
    if (params.flags.y == 1u) {
        let mapped = normalize(normal_s * 2.0 - 1.0);
        let t = normalize(in.tangent.xyz);
        let b = normalize(cross(n, t)) * in.tangent.w;
        n = normalize(mat3x3<f32>(t, b, n) * mapped);
    }
    let encoded = encode_normal(n);
    var metal_rough = vec2<f32>(0.1, 0.5);
    if (params.flags.z == 1u) {
        metal_rough = mr_s;
    }
    let depth = exponential_depth(in.view_depth, frame.near_far);
    let id = f32(in.tri);

    var color: vec4<f32>;
    switch frame.render_mode {
        case 1u: { color = vec4<f32>(vec3<f32>(depth), 1.0); }
        case 2u: { color = vec4<f32>(encoded, 1.0); }
        case 3u: { color = vec4<f32>(hash(id * 311.7), hash(id * 269.5 + 1.3), hash(id * 183.3 + 2.7), 1.0); }
        case 4u: { color = vec4<f32>(0.01, 0.005, 0.0, 0.01); }
        default: { color = albedo; }
    }

    var out: GBufferOut;
    out.position = vec4<f32>(in.world_pos, 1.0);
    out.normal = vec4<f32>(encoded, 1.0);
    out.albedo = vec4<f32>(color.rgb, 1.0);
    out.metal_rough = vec4<f32>(metal_rough, 0.0, 1.0);
    return out;
}
