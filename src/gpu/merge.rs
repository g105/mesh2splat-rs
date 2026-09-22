//! GPU version of [`crate::merge`]: the same quadtree merge, run with compute
//! shaders (`shaders/merge.wgsl`) so the splats never leave the GPU.

use bytemuck::{Pod, Zeroable};

use super::sort::RadixSorter;
use super::{
    dispatch_dims, shader_with_common, storage_entry, uniform_entry, GaussianBuffer, GpuContext,
};
use crate::merge::{GridInfo, MergeSettings, MergeStats};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    level: u32,
    child_res: u32,
    parent_res: u32,
    groups: u32,
    node_capacity: u32,
    leaf_count: u32,
    _p0: u32,
    color_tol: f32,
    normal_tol: f32,
    flatness: f32,
    _p1: f32,
    depth_min: [f32; 4],
    depth_scale: [f32; 4],
}

/// Per-run working buffers, kept between runs and grown on demand (wgpu
/// zero-fills every new buffer, which is not free at these sizes).
struct Scratch {
    capacity: u32,
    items: [wgpu::Buffer; 2],
    nodes: wgpu::Buffer,
    used: wgpu::Buffer,
}

impl Scratch {
    fn new(device: &wgpu::Device, capacity: u32) -> Self {
        use wgpu::BufferUsages as U;
        let n = capacity as u64;
        Self {
            capacity,
            items: [
                storage(device, "merge items a", n * ITEM_BYTES, U::empty()),
                storage(device, "merge items b", n * ITEM_BYTES, U::empty()),
            ],
            nodes: storage(device, "merge nodes", node_capacity(capacity) as u64 * NODE_BYTES, U::empty()),
            used: storage(device, "merge used", (n + node_capacity(capacity) as u64) * 4, U::COPY_DST),
        }
    }
}

/// Every merge replaces 4 items by 1, so the nodes of all levels stay below n / 3.
fn node_capacity(count: u32) -> u32 {
    count / 3 + 1
}

const MAX_GROUPS: usize = 256;
const ITEM_BYTES: u64 = 16;
const NODE_BYTES: u64 = 176;
const GAUSSIAN_BYTES: u64 = 96;
const STORAGE_BUFFERS: u32 = 9;

pub struct GpuMerger {
    init: wgpu::ComputePipeline,
    prepare: wgpu::ComputePipeline,
    rekey: wgpu::ComputePipeline,
    evaluate: wgpu::ComputePipeline,
    emit_leaves: wgpu::ComputePipeline,
    emit_nodes: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    sorter: RadixSorter,
    params: wgpu::Buffer,
    cell_sizes: wgpu::Buffer,
    /// `[0]` = items to sort (read by the sorter's setup pass).
    count: wgpu::Buffer,
    /// next-level items, nodes, output splats
    counters: wgpu::Buffer,
    scratch: Option<Scratch>,
    /// Bound as the output while the levels run (nothing is emitted then).
    no_output: wgpu::Buffer,
}

fn e(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn storage(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE | extra,
        mapped_at_creation: false,
    })
}

impl GpuMerger {
    /// `None` when the device cannot bind enough storage buffers per stage.
    pub fn new(ctx: &GpuContext) -> Option<Self> {
        let device = &ctx.device;
        if device.limits().max_storage_buffers_per_shader_stage < STORAGE_BUFFERS {
            return None;
        }
        use wgpu::ShaderStages as S;
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("merge bgl"),
            entries: &[
                uniform_entry(0, S::COMPUTE),
                uniform_entry(1, S::COMPUTE),
                storage_entry(2, S::COMPUTE, true),
                storage_entry(3, S::COMPUTE, false),
                storage_entry(4, S::COMPUTE, false),
                storage_entry(5, S::COMPUTE, false),
                storage_entry(6, S::COMPUTE, false),
                storage_entry(7, S::COMPUTE, false),
                storage_entry(8, S::COMPUTE, false),
                storage_entry(9, S::COMPUTE, false),
                storage_entry(10, S::COMPUTE, false),
            ],
        });
        let module = shader_with_common(device, "merge.wgsl", include_str!("shaders/merge.wgsl"));
        let layout = super::pipeline_layout(device, "merge", &[&bgl]);
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let uniform = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as U;
        Some(Self {
            init: pipeline("init"),
            prepare: pipeline("prepare"),
            rekey: pipeline("rekey"),
            evaluate: pipeline("evaluate"),
            emit_leaves: pipeline("emit_leaves"),
            emit_nodes: pipeline("emit_nodes"),
            bgl,
            sorter: RadixSorter::new(ctx),
            params: uniform("merge params", std::mem::size_of::<Params>() as u64),
            cell_sizes: uniform("merge cell sizes", MAX_GROUPS as u64 * 16),
            count: storage(device, "merge count", 16, U::COPY_DST),
            counters: storage(device, "merge counters", 16, U::COPY_DST | U::COPY_SRC),
            scratch: None,
            no_output: storage(device, "merge no output", GAUSSIAN_BYTES, U::empty()),
        })
    }

    /// Whether `grid` fits the shader's 32-bit cell keys and group table.
    pub fn supports(grid: &GridInfo) -> bool {
        let groups = grid.boxes.len().max(1) as u64;
        let res = grid.resolution as u64;
        groups <= MAX_GROUPS as u64 && groups * 3 * res * res < u32::MAX as u64
    }

    /// Merge the first `count` splats of `input`. Returns the merged splats in
    /// a new buffer, their count and the merge statistics.
    pub fn run(
        &mut self,
        ctx: &GpuContext,
        input: &wgpu::Buffer,
        count: u32,
        grid: &GridInfo,
        cfg: &MergeSettings,
    ) -> (wgpu::Buffer, u32, MergeStats) {
        assert!(Self::supports(grid));
        let device = &ctx.device;
        let node_capacity = node_capacity(count);
        let groups = grid.boxes.len().max(1);
        if self.scratch.as_ref().is_none_or(|s| s.capacity < count) {
            self.scratch = Some(Scratch::new(device, count));
        }
        let Scratch {
            items, nodes, used, ..
        } = self.scratch.as_ref().unwrap();
        self.sorter.ensure_capacity(ctx, count, &self.count);

        // 16-bit depth keys are enough when their step is well below a grid
        // cell (items closer than a cell always share a layer anyway).
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        let mut min_cell = [f64::MAX; 3];
        for b in &grid.boxes {
            for a in 0..3 {
                lo[a] = lo[a].min(b.min[a]);
                hi[a] = hi[a].max(b.max[a]);
                min_cell[a] = min_cell[a].min(crate::merge::cell_size(b, a as u32, grid.resolution));
            }
        }
        let quantize = (0..3).all(|a| ((hi[a] - lo[a]) as f64 / 65535.0) <= 0.25 * min_cell[a]);
        let (depth_min, depth_scale, depth_bits) = if quantize {
            let s = |a: usize| 65535.0 / (hi[a] - lo[a]).max(1e-30);
            ([lo[0], lo[1], lo[2], 0.0], [s(0), s(1), s(2), 1.0], 16)
        } else {
            ([0.0; 4], [0.0; 4], 32)
        };

        let mut sizes = [[0f32; 4]; MAX_GROUPS];
        for (i, b) in grid.boxes.iter().enumerate() {
            for (axis, size) in sizes[i].iter_mut().take(3).enumerate() {
                *size = crate::merge::cell_size(b, axis as u32, grid.resolution) as f32;
            }
        }
        ctx.queue
            .write_buffer(&self.cell_sizes, 0, bytemuck::cast_slice(&sizes));

        // Bind groups for both directions of the items ping-pong.
        let bind_group = |cur: &wgpu::Buffer, next: &wgpu::Buffer, out: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("merge"),
                layout: &self.bgl,
                entries: &[
                    e(0, &self.params),
                    e(1, &self.cell_sizes),
                    e(2, input),
                    e(3, cur),
                    e(4, next),
                    e(5, &self.sorter.keys[0]),
                    e(6, &self.sorter.vals[0]),
                    e(7, nodes),
                    e(8, used),
                    e(9, &self.counters),
                    e(10, out),
                ],
            })
        };
        let bgs = [
            bind_group(&items[0], &items[1], &self.no_output),
            bind_group(&items[1], &items[0], &self.no_output),
        ];

        let base = Params {
            n: count,
            level: 0,
            child_res: grid.resolution,
            parent_res: grid.resolution,
            groups: groups as u32,
            node_capacity,
            leaf_count: count,
            _p0: 0,
            color_tol: cfg.color_tolerance,
            normal_tol: cfg.normal_tolerance_deg.to_radians(),
            flatness: cfg.flatness,
            _p1: 0.0,
            depth_min,
            depth_scale,
        };
        let dispatch = |pass: &mut wgpu::ComputePass, p: &wgpu::ComputePipeline, n: u32| {
            if n > 0 {
                let (x, y) = dispatch_dims(n.div_ceil(256));
                pass.set_pipeline(p);
                pass.dispatch_workgroups(x, y, 1);
            }
        };
        let read_counters = |ctx: &GpuContext| -> [u32; 3] {
            let b = ctx.read_buffer(&self.counters, 0, 12);
            let w: &[u32] = bytemuck::cast_slice(&b);
            [w[0], w[1], w[2]]
        };

        let mut stats = MergeStats {
            input: count as usize,
            ..Default::default()
        };
        let mut enc = device.create_command_encoder(&Default::default());
        enc.clear_buffer(used, 0, Some((count as u64 + node_capacity as u64) * 4));
        enc.clear_buffer(&self.counters, 0, None);
        ctx.queue.submit([enc.finish()]);

        let mut n = count;
        let mut res = grid.resolution;
        let mut nodes_before = 0;
        for level in 1..=cfg.max_level {
            let parent_res = res.div_ceil(2);
            let p = Params {
                n,
                level,
                child_res: res,
                parent_res,
                ..base
            };
            ctx.queue
                .write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
            ctx.queue
                .write_buffer(&self.count, 0, bytemuck::bytes_of(&n));
            let bg = &bgs[(level as usize - 1) % 2];
            let key_space = groups as u64 * 3 * parent_res as u64 * parent_res as u64;
            let key_bits = 64 - key_space.leading_zeros();

            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("merge level"),
            });
            enc.clear_buffer(&self.counters, 0, Some(4)); // next-level item count
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, bg, &[]);
                if level == 1 {
                    dispatch(&mut pass, &self.init, n);
                }
                dispatch(&mut pass, &self.prepare, n);
            }
            self.sorter.encode(&mut enc, depth_bits); // by depth...
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, bg, &[]);
                dispatch(&mut pass, &self.rekey, n);
            }
            self.sorter.encode(&mut enc, key_bits); // ...then stably by cell
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, bg, &[]);
                dispatch(&mut pass, &self.evaluate, n);
            }
            ctx.queue.submit([enc.finish()]);

            let [next, total_nodes, _] = read_counters(ctx);
            let total_nodes = total_nodes.min(node_capacity);
            stats
                .merged_per_level
                .push((total_nodes - nodes_before) as usize);
            nodes_before = total_nodes;
            n = next;
            res = parent_res;
            if n == 0 {
                break;
            }
        }

        // Each merge turned 4 items into 1, so the output size is known: emit
        // straight into an exactly sized buffer.
        let out_count = count - 3 * nodes_before;
        let result = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gaussians"),
            size: (out_count as u64 * GAUSSIAN_BYTES).max(GAUSSIAN_BYTES),
            usage: GaussianBuffer::USAGE,
            mapped_at_creation: false,
        });
        let emit_bg = bind_group(&items[0], &items[1], &result);
        let p = Params {
            n: nodes_before,
            ..base
        };
        ctx.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("merge emit"),
        });
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &emit_bg, &[]);
            dispatch(&mut pass, &self.emit_leaves, count);
            dispatch(&mut pass, &self.emit_nodes, nodes_before);
        }
        ctx.queue.submit([enc.finish()]);
        stats.output = out_count as usize;
        debug_assert_eq!(read_counters(ctx)[2], out_count);
        (result, out_count, stats)
    }
}
