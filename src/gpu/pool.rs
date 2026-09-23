//! GPU version of [`crate::merge::merge_occluded`]: pool buried splats into
//! coarse volume-filling ones without a round trip through host memory.

use bytemuck::{Pod, Zeroable};

use super::sort::RadixSorter;
use super::{
    compute_pipeline, dispatch_dims, shader_with_common, storage_entry, uniform_entry,
    GaussianBuffer, GpuContext,
};
use crate::merge::{
    quantile_from_histogram, MergeStats, VolumeMergeSettings, MAX_DIRECTION_BINS, OCCLUSION_BINS,
    OPEN_QUANTILE,
};
use crate::types::BBox;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    bbox_min: [f32; 4],
    cell: [f32; 4],
    dims: [u32; 4],
    bins: [u32; 4],
    tune: [f32; 4],
}

const RUN_BYTES: u64 = 8;
const GAUSSIAN_BYTES: u64 = 96;
/// Cell indices have to fit a sort key: 1024^3 is comfortably inside u32.
const MAX_GRID: u32 = 1024;
/// Counters, then the occlusion histogram the buried fraction resolves against.
const COUNTERS: u64 = 4;
const COUNTER_BYTES: u64 = (COUNTERS + OCCLUSION_BINS as u64) * 4;
/// Headroom for the fixed-point size reduction: every splat contributes at
/// most `SIZE_BUDGET / count`, so the sum cannot overflow a u32.
const SIZE_BUDGET: f64 = 4.0e9;

pub struct GpuPooler {
    measure: wgpu::ComputePipeline,
    key: wgpu::ComputePipeline,
    mark_runs: wgpu::ComputePipeline,
    emit_clusters: wgpu::ComputePipeline,
    emit_kept: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    sorter: RadixSorter,
    params: wgpu::Buffer,
    /// `[0]` = items to sort, for the sorter's setup pass.
    count: wgpu::Buffer,
    /// clusters, output splats, splats pooled away
    counters: wgpu::Buffer,
    scratch: Option<Scratch>,
}

struct Scratch {
    capacity: u32,
    runs: wgpu::Buffer,
    used: wgpu::Buffer,
}

fn storage(device: &wgpu::Device, label: &str, size: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE | extra,
        mapped_at_creation: false,
    })
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl GpuPooler {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;
        use wgpu::ShaderStages as S;
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool bgl"),
            entries: &[
                uniform_entry(0, S::COMPUTE),
                storage_entry(1, S::COMPUTE, true),
                storage_entry(2, S::COMPUTE, false),
                storage_entry(3, S::COMPUTE, false),
                storage_entry(4, S::COMPUTE, false),
                storage_entry(5, S::COMPUTE, false),
                storage_entry(6, S::COMPUTE, false),
                storage_entry(7, S::COMPUTE, false),
            ],
        });
        let module = shader_with_common(device, "pool.wgsl", include_str!("shaders/pool.wgsl"));
        use wgpu::BufferUsages as U;
        Self {
            measure: compute_pipeline(device, "pool measure", &[&bgl], &module, "measure"),
            key: compute_pipeline(device, "pool key", &[&bgl], &module, "key"),
            mark_runs: compute_pipeline(device, "pool runs", &[&bgl], &module, "mark_runs"),
            emit_clusters: compute_pipeline(device, "pool clusters", &[&bgl], &module, "emit_clusters"),
            emit_kept: compute_pipeline(device, "pool kept", &[&bgl], &module, "emit_kept"),
            bgl,
            sorter: RadixSorter::new(ctx),
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pool params"),
                size: std::mem::size_of::<Params>() as u64,
                usage: U::UNIFORM | U::COPY_DST,
                mapped_at_creation: false,
            }),
            count: storage(device, "pool count", 16, U::COPY_DST),
            counters: storage(
                device,
                "pool counters",
                COUNTER_BYTES,
                U::COPY_DST | U::COPY_SRC,
            ),
            scratch: None,
        }
    }

    /// Pool the buried splats of `gaussians` into a new buffer. Needs the
    /// occlusion bake to have run; unbaked splats read as unknown and are kept.
    pub fn run(
        &mut self,
        ctx: &GpuContext,
        gaussians: &GaussianBuffer,
        bounds: &BBox,
        cfg: &VolumeMergeSettings,
    ) -> (wgpu::Buffer, u32, MergeStats) {
        let device = &ctx.device;
        let count = gaussians.count;
        let mut stats = MergeStats {
            input: count as usize,
            output: count as usize,
            ..Default::default()
        };
        if count == 0 {
            return (
                storage(device, "gaussians", GAUSSIAN_BYTES, wgpu::BufferUsages::COPY_SRC),
                0,
                stats,
            );
        }
        if self.scratch.as_ref().is_none_or(|s| s.capacity < count) {
            self.scratch = Some(Scratch {
                capacity: count,
                runs: storage(device, "pool runs", count as u64 * RUN_BYTES, wgpu::BufferUsages::empty()),
                used: storage(device, "pool used", count as u64 * 4, wgpu::BufferUsages::empty()),
            });
        }
        let scratch = self.scratch.as_ref().unwrap();
        self.sorter.ensure_capacity(ctx, count, &self.count);

        let size = bounds.size() * 1.1;
        let min = bounds.center() - size * 0.5;
        let mut params = Params {
            bbox_min: min.extend(0.0).to_array(),
            cell: [1.0; 4],
            dims: [1, count, cfg.min_cluster.max(2) as u32, 0],
            bins: [
                cfg.direction_bins.clamp(1, MAX_DIRECTION_BINS),
                cfg.along_cells.max(1),
                0,
                0,
            ],
            tune: [
                // Resolved from the histogram the measure pass builds.
                0.0,
                (SIZE_BUDGET / count as f64) as f32,
                size.max_element(),
                0.0,
            ],
        };
        ctx.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        ctx.queue
            .write_buffer(&self.count, 0, bytemuck::bytes_of(&count));

        let out_scratch = storage(
            device,
            "pool out",
            count as u64 * GAUSSIAN_BYTES,
            wgpu::BufferUsages::COPY_SRC,
        );
        let bind = |out: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pool"),
                layout: &self.bgl,
                entries: &[
                    entry(0, &self.params),
                    entry(1, &gaussians.buffer),
                    entry(2, &self.sorter.keys[0]),
                    entry(3, &self.sorter.vals[0]),
                    entry(4, &scratch.runs),
                    entry(5, &scratch.used),
                    entry(6, &self.counters),
                    entry(7, out),
                ],
            })
        };
        let bg = bind(&out_scratch);
        let dispatch = |pass: &mut wgpu::ComputePass, p: &wgpu::ComputePipeline, n: u32, per: u32| {
            if n > 0 {
                let (x, y) = dispatch_dims(n.div_ceil(per));
                pass.set_pipeline(p);
                pass.dispatch_workgroups(x, y, 1);
            }
        };

        // The cell has to follow the splats, not the model: measure them first.
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pool measure"),
        });
        enc.clear_buffer(&self.counters, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &bg, &[]);
            dispatch(&mut pass, &self.measure, count, 256);
        }
        ctx.queue.submit([enc.finish()]);
        let measured = ctx.read_buffer(&self.counters, 0, COUNTER_BYTES);
        let measured: &[u32] = bytemuck::cast_slice(&measured);
        let total = measured[3];
        // Same histogram and the same bin edges as the CPU pooler, so both
        // pick the same threshold for a given fraction.
        params.tune[0] = cfg.relative_openness
            * quantile_from_histogram(&measured[COUNTERS as usize..], count as u64, OPEN_QUANTILE);
        // Back out of the fixed point: mean fraction of the model, then units.
        let fixed = SIZE_BUDGET / count as f64;
        let mean_face = total as f64 / fixed / count as f64 * size.max_element() as f64;
        // Same rule as the CPU pooler: a cell a few splats wide.
        let cell = cfg.cell.unwrap_or((mean_face * 6.0) as f32).max(1e-6);
        let grid = ((size.max_element() / cell).ceil() as u32).clamp(1, MAX_GRID);
        params.cell = glam::Vec3::splat(cell).extend(0.0).to_array();
        params.dims[0] = grid;
        ctx.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        // Key by cell, sort, then one thread per run of equal keys.
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pool"),
        });
        enc.clear_buffer(&self.counters, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &bg, &[]);
            dispatch(&mut pass, &self.key, count, 256);
        }
        // Keys are 4 bits of direction bin over 3 x 9 bits of cell coordinate
        // (see `key` in the shader), so the sort has 31 bits to look at.
        let bits = 31;
        self.sorter.encode(&mut enc, bits);
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &bg, &[]);
            dispatch(&mut pass, &self.mark_runs, count, 256);
        }
        ctx.queue.submit([enc.finish()]);

        let c = ctx.read_buffer(&self.counters, 0, 12);
        let c: &[u32] = bytemuck::cast_slice(&c);
        let (clusters, pooled) = (c[0], c[2]);
        stats.merged_per_level = vec![clusters as usize];

        // Every pooled splat is replaced by its cluster, so the size is known.
        let out_count = count - pooled + clusters;
        params.dims[3] = clusters;
        ctx.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let result = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gaussians"),
            size: (out_count as u64 * GAUSSIAN_BYTES).max(GAUSSIAN_BYTES),
            usage: GaussianBuffer::USAGE,
            mapped_at_creation: false,
        });
        let out_bg = bind(&result);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pool emit"),
        });
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &out_bg, &[]);
            dispatch(&mut pass, &self.emit_clusters, clusters, 64);
            dispatch(&mut pass, &self.emit_kept, count, 256);
        }
        ctx.queue.submit([enc.finish()]);
        stats.output = out_count as usize;
        stats.gpu = true;
        debug_assert_eq!(ctx.read_buffer(&self.counters, 4, 4)[..4], out_count.to_le_bytes());
        (result, out_count, stats)
    }
}
