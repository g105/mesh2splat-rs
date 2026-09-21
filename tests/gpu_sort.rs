//! GPU radix sort vs. CPU stable sort. Needs a GPU (or lavapipe).

use mesh2splat::gpu::sort::RadixSorter;
use mesh2splat::gpu::GpuContext;

fn xorshift(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 16) as u32
}

fn check(ctx: &GpuContext, sorter: &mut RadixSorter, n: usize, key_mask: u32, seed: u64) {
    let mut s = seed;
    let keys: Vec<u32> = (0..n).map(|_| xorshift(&mut s) & key_mask).collect();
    let vals: Vec<u32> = (0..n as u32).collect();

    let counter = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue
        .write_buffer(&counter, 0, bytemuck::cast_slice(&[n as u32, 0, 0, 0]));
    sorter.ensure_capacity(ctx, (n as u32).max(1), &counter);
    if n > 0 {
        ctx.queue
            .write_buffer(&sorter.keys[0], 0, bytemuck::cast_slice(&keys));
        ctx.queue
            .write_buffer(&sorter.vals[0], 0, bytemuck::cast_slice(&vals));
    }
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    sorter.encode(&mut enc);
    ctx.queue.submit([enc.finish()]);

    let info = ctx.read_buffer(&sorter.info, 0, 4);
    assert_eq!(u32::from_le_bytes(info.try_into().unwrap()) as usize, n);
    if n == 0 {
        return;
    }
    let out_k: Vec<u32> =
        bytemuck::cast_slice(&ctx.read_buffer(&sorter.keys[0], 0, n as u64 * 4)).to_vec();
    let out_v: Vec<u32> =
        bytemuck::cast_slice(&ctx.read_buffer(&sorter.vals[0], 0, n as u64 * 4)).to_vec();

    let mut expected: Vec<(u32, u32)> = keys.iter().copied().zip(vals.iter().copied()).collect();
    expected.sort_by_key(|p| p.0); // stable
    for i in 0..n {
        assert_eq!(
            (out_k[i], out_v[i]),
            expected[i],
            "mismatch at {i} (n = {n}, mask = {key_mask:#x})"
        );
    }
    // args: dispatch x = blocks, draw instance count = n
    let args: Vec<u32> = bytemuck::cast_slice(&ctx.read_buffer(&sorter.args, 0, 28)).to_vec();
    assert_eq!(args[0] as usize, n.div_ceil(2048));
    assert_eq!(args[3], 6);
    assert_eq!(args[4] as usize, n);
}

#[test]
fn radix_sort_matches_cpu() {
    let ctx = match GpuContext::new_headless() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let mut sorter = RadixSorter::new(&ctx);
    for (i, &n) in [
        0usize, 1, 2, 4, 5, 127, 128, 2047, 2048, 2049, 10_000, 100_003, 1_000_000,
    ]
    .iter()
    .enumerate()
    {
        check(
            &ctx,
            &mut sorter,
            n,
            u32::MAX,
            0x9E3779B97F4A7C15 ^ i as u64,
        );
    }
    // Many duplicates -> exercises stability.
    check(&ctx, &mut sorter, 50_000, 0xF, 7);
    // Float-bit keys of negative depths (what the renderer sorts).
    let n = 30_000;
    let mut s = 11u64;
    let depths: Vec<f32> = (0..n)
        .map(|_| -((xorshift(&mut s) % 100_000) as f32) / 1000.0 - 0.01)
        .collect();
    let mut sorted = depths.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap()); // nearest (least negative) first
    let bits: Vec<u32> = depths.iter().map(|d| d.to_bits()).collect();
    let mut by_bits = bits.clone();
    by_bits.sort();
    let back: Vec<f32> = by_bits.iter().map(|b| f32::from_bits(*b)).collect();
    assert_eq!(
        back, sorted,
        "u32 order of negative floats must be front-to-back"
    );
}
