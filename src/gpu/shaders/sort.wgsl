// GPU-driven stable LSD radix sort of (u32 key, u32 value) pairs.
// 8 passes x 4 bits. Each pass: per-block digit histogram, one exclusive scan
// over all (digit, block) counts, then a stable scatter.
// The element count lives on the GPU (written by `setup`), so no CPU readback
// is needed between the prepass and the draw.

const WG: u32 = 128u;       // threads per workgroup (histogram / scatter)
const ITEMS: u32 = 16u;     // keys per thread
const BLOCK: u32 = 2048u;   // WG * ITEMS
const RADIX: u32 = 16u;

struct PassParams {
    shift: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read> info: array<u32>; // [0] = element count
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read> vals_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> vals_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> block_hist: array<u32>;
@group(0) @binding(6) var<uniform> pass_params: PassParams;

var<workgroup> hist: array<atomic<u32>, 16>;
var<workgroup> table: array<u32, 2048>; // [digit * WG + thread]
var<workgroup> tsum: array<u32, 256>;

fn num_blocks(n: u32) -> u32 {
    return (n + BLOCK - 1u) / BLOCK;
}

@compute @workgroup_size(128)
fn histogram(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    if (lid < RADIX) {
        atomicStore(&hist[lid], 0u);
    }
    workgroupBarrier();
    let n = info[0];
    let base = wid.x * BLOCK;
    let shift = pass_params.shift;
    for (var i = 0u; i < ITEMS; i++) {
        let idx = base + i * WG + lid;
        if (idx < n) {
            atomicAdd(&hist[(keys_in[idx] >> shift) & 15u], 1u);
        }
    }
    workgroupBarrier();
    if (lid < RADIX) {
        block_hist[lid * num_blocks(n) + wid.x] = atomicLoad(&hist[lid]);
    }
}

// Single workgroup exclusive scan over the digit-major (digit, block) table.
@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_index) lid: u32) {
    let n = info[0];
    let total = num_blocks(n) * RADIX;
    let per = (total + 255u) / 256u;
    let start = min(lid * per, total);
    let end = min(start + per, total);
    var s = 0u;
    for (var i = start; i < end; i++) {
        s += block_hist[i];
    }
    tsum[lid] = s;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off = off << 1u) {
        var t = 0u;
        if (lid >= off) {
            t = tsum[lid - off];
        }
        workgroupBarrier();
        tsum[lid] += t;
        workgroupBarrier();
    }
    var run = tsum[lid] - s;
    for (var i = start; i < end; i++) {
        let v = block_hist[i];
        block_hist[i] = run;
        run += v;
    }
}

@compute @workgroup_size(128)
fn scatter(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = info[0];
    let nb = num_blocks(n);
    let shift = pass_params.shift;
    let base = wid.x * BLOCK + lid * ITEMS; // contiguous chunk per thread => stable

    for (var d = 0u; d < RADIX; d++) {
        table[d * WG + lid] = 0u;
    }
    var local_keys: array<u32, 16>;
    for (var i = 0u; i < ITEMS; i++) {
        let idx = base + i;
        if (idx < n) {
            let k = keys_in[idx];
            local_keys[i] = k;
            let d = (k >> shift) & 15u;
            table[d * WG + lid] += 1u;
        }
    }
    workgroupBarrier();

    // Exclusive scan of the 2048-entry table; thread t owns entries [16t, 16t+16).
    var s = 0u;
    for (var i = 0u; i < 16u; i++) {
        s += table[lid * 16u + i];
    }
    tsum[lid] = s;
    workgroupBarrier();
    for (var off = 1u; off < WG; off = off << 1u) {
        var t = 0u;
        if (lid >= off) {
            t = tsum[lid - off];
        }
        workgroupBarrier();
        tsum[lid] += t;
        workgroupBarrier();
    }
    var run = tsum[lid] - s;
    for (var i = 0u; i < 16u; i++) {
        let v = table[lid * 16u + i];
        table[lid * 16u + i] = run;
        run += v;
    }
    workgroupBarrier();

    var cnt: array<u32, 16>;
    for (var d = 0u; d < RADIX; d++) {
        cnt[d] = 0u;
    }
    for (var i = 0u; i < ITEMS; i++) {
        let idx = base + i;
        if (idx < n) {
            let k = local_keys[i];
            let d = (k >> shift) & 15u;
            let pos = block_hist[d * nb + wid.x] + (table[d * WG + lid] - table[d * WG]) + cnt[d];
            cnt[d] += 1u;
            keys_out[pos] = k;
            vals_out[pos] = vals_in[idx];
        }
    }
}

