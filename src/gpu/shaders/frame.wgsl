// Per-frame uniforms shared by the prepass, splat and mesh passes.
struct Frame {
    world_to_view: mat4x4<f32>,
    view_to_clip: mat4x4<f32>,
    model_to_world: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,  // transpose(inverse(model_to_world))
    inv_model_rot: mat4x4<f32>,  // inverse(mat3(model_to_world)), padded
    model_scale: vec4<f32>,
    resolution: vec2<f32>,
    near_far: vec2<f32>,
    std_dev: f32,
    gaussian_count: u32,
    render_mode: u32,
    format: u32,       // 0 = converted mesh, 1 = loaded ply
    ply_has_pbr: u32,
    depth_test: u32,
    // View depth -> 16-bit sort key: (depth - sort_min) * sort_scale.
    // sort_scale == 0 means sort on the raw 32-bit float depth.
    sort_min: f32,
    sort_scale: f32,
    // Deferred shading has no spare G-buffer channel for baked occlusion, so
    // the splat pass folds it into the colour it writes. That dims direct light
    // as well as ambient; the forward path applies it to ambient only.
    ao_deferred: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
};
