//! GPU radix sort used to order splats front-to-back.
//!
//! Replaces the `gl-radix-sort` dependency of the original. The number of
//! elements is read on the GPU from a counter buffer, and the histogram /
//! scatter kernels are dispatched indirectly, so sorting never stalls the CPU.

use super::{compute_pipeline, storage_entry, uniform_entry, GpuContext};

pub const BLOCK: u32 = 2048;
const PASSES: u32 = 8; // 32 bits / 4 bits per pass
const UNIFORM_STRIDE: u64 = 256;

/// Byte offset of the `draw_indirect` arguments inside [`RadixSorter::args`].
pub const DRAW_ARGS_OFFSET: u64 = 12;

pub struct RadixSorter {
    histogram: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    setup: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    setup_bgl: wgpu::BindGroupLayout,
    pass_params: wgpu::Buffer,
    setup_params: wgpu::Buffer,
    pub capacity: u32,
    /// Ping-pong key / value buffers. The caller writes unsorted pairs into
    /// index 0; after [`RadixSorter::encode`] the sorted result is in index 0 again.
    pub keys: [wgpu::Buffer; 2],
    pub vals: [wgpu::Buffer; 2],
    block_hist: wgpu::Buffer,
    /// `[0]` = number of sorted elements.
    pub info: wgpu::Buffer,
    /// `[0..3]` dispatch args, `[3..7]` draw args (6 vertices x count instances).
    pub args: wgpu::Buffer,
    bind_groups: Vec<wgpu::BindGroup>,
    setup_bg: Option<wgpu::BindGroup>,
}

fn buf(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage,
        mapped_at_creation: false,
    })
}

impl RadixSorter {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sort.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sort.wgsl").into()),
        });
        let setup_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sort_setup.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sort_setup.wgsl").into()),
        });
        use wgpu::ShaderStages as S;
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort bgl"),
            entries: &[
                storage_entry(0, S::COMPUTE, true),
                storage_entry(1, S::COMPUTE, true),
                storage_entry(2, S::COMPUTE, true),
                storage_entry(3, S::COMPUTE, false),
                storage_entry(4, S::COMPUTE, false),
                storage_entry(5, S::COMPUTE, false),
                uniform_entry(6, S::COMPUTE),
            ],
        });
        let setup_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort setup bgl"),
            entries: &[
                storage_entry(0, S::COMPUTE, true),
                storage_entry(1, S::COMPUTE, false),
                storage_entry(2, S::COMPUTE, false),
                uniform_entry(3, S::COMPUTE),
            ],
        });
        let histogram = compute_pipeline(device, "sort histogram", &[&bgl], &module, "histogram");
        let scan = compute_pipeline(device, "sort scan", &[&bgl], &module, "scan");
        let scatter = compute_pipeline(device, "sort scatter", &[&bgl], &module, "scatter");
        let setup = compute_pipeline(device, "sort setup", &[&setup_bgl], &setup_module, "setup");

        let pass_params = buf(
            device,
            "sort pass params",
            UNIFORM_STRIDE * PASSES as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let mut data = vec![0u8; (UNIFORM_STRIDE * PASSES as u64) as usize];
        for p in 0..PASSES as usize {
            data[p * UNIFORM_STRIDE as usize..][..4].copy_from_slice(&(p as u32 * 4).to_le_bytes());
        }
        ctx.queue.write_buffer(&pass_params, 0, &data);
        let setup_params = buf(
            device,
            "sort setup params",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let kv_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let mk = |l| buf(device, l, 16, kv_usage);
        Self {
            histogram,
            scan,
            scatter,
            setup,
            bgl,
            setup_bgl,
            pass_params,
            setup_params,
            capacity: 0,
            keys: [mk("keys a"), mk("keys b")],
            vals: [mk("vals a"), mk("vals b")],
            block_hist: mk("block hist"),
            info: buf(device, "sort info", 16, kv_usage),
            args: buf(
                device,
                "sort args",
                32,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_SRC,
            ),
            bind_groups: Vec::new(),
            setup_bg: None,
        }
    }

    /// Resize for up to `capacity` elements. `counter` holds the element count
    /// in its first u32 (e.g. the prepass' atomic visible counter).
    pub fn ensure_capacity(&mut self, ctx: &GpuContext, capacity: u32, counter: &wgpu::Buffer) {
        let capacity = capacity.max(1);
        let device = &ctx.device;
        if capacity > self.capacity || self.bind_groups.is_empty() {
            let kv_usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC;
            let size = capacity as u64 * 4;
            self.keys = [
                buf(device, "keys a", size, kv_usage),
                buf(device, "keys b", size, kv_usage),
            ];
            self.vals = [
                buf(device, "vals a", size, kv_usage),
                buf(device, "vals b", size, kv_usage),
            ];
            self.block_hist = buf(
                device,
                "block hist",
                capacity.div_ceil(BLOCK) as u64 * 16 * 4,
                kv_usage,
            );
            self.capacity = capacity;
            self.bind_groups = (0..PASSES as usize)
                .map(|p| {
                    let (src, dst) = (p % 2, (p + 1) % 2);
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("sort pass"),
                        layout: &self.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.info.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.keys[src].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.vals[src].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.keys[dst].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: self.vals[dst].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: self.block_hist.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &self.pass_params,
                                    offset: p as u64 * UNIFORM_STRIDE,
                                    size: wgpu::BufferSize::new(16),
                                }),
                            },
                        ],
                    })
                })
                .collect();
        }
        ctx.queue.write_buffer(
            &self.setup_params,
            0,
            bytemuck::cast_slice(&[capacity, 0, 0, 0]),
        );
        self.setup_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sort setup"),
            layout: &self.setup_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.args.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.setup_params.as_entire_binding(),
                },
            ],
        }));
    }

    /// Record the sort. Expects unsorted pairs in `keys[0]` / `vals[0]`.
    pub fn encode(&self, enc: &mut wgpu::CommandEncoder) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("radix sort"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.setup);
        pass.set_bind_group(
            0,
            self.setup_bg.as_ref().expect("ensure_capacity not called"),
            &[],
        );
        pass.dispatch_workgroups(1, 1, 1);
        for bg in &self.bind_groups {
            pass.set_bind_group(0, bg, &[]);
            pass.set_pipeline(&self.histogram);
            pass.dispatch_workgroups_indirect(&self.args, 0);
            pass.set_pipeline(&self.scan);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.scatter);
            pass.dispatch_workgroups_indirect(&self.args, 0);
        }
    }
}
