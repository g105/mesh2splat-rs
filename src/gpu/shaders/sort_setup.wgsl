// ---------------------------------------------------------------------------
// Setup: turn the prepass' visible counter into sort dispatch + draw arguments.

struct SetupParams {
    capacity: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read> setup_counter: array<u32>;
@group(0) @binding(1) var<storage, read_write> setup_info: array<u32>;
@group(0) @binding(2) var<storage, read_write> setup_args: array<u32>;
@group(0) @binding(3) var<uniform> setup_params: SetupParams;

@compute @workgroup_size(1)
fn setup() {
    let n = min(setup_counter[0], setup_params.capacity);
    setup_info[0] = n;
    // dispatch_workgroups_indirect args for histogram / scatter
    setup_args[0] = (n + 2047u) / 2048u; // must match BLOCK in sort.wgsl
    setup_args[1] = 1u;
    setup_args[2] = 1u;
    // draw_indirect args for the splat pass: one 4-vertex triangle strip per splat
    setup_args[3] = 4u;
    setup_args[4] = n;
    setup_args[5] = 0u;
    setup_args[6] = 0u;
}
