
struct GcParams {
    rows_per_chunk: u32,
    row_words: u32,
    total_rows: u32,
    n_chunks: u32,
};

@group(0) @binding(0) var<storage, read> gc0: array<u32>;
@group(0) @binding(1) var<storage, read> gc1: array<u32>;
@group(0) @binding(2) var<storage, read> gc2: array<u32>;
@group(0) @binding(3) var<storage, read> gc3: array<u32>;
@group(0) @binding(4) var<storage, read> gc4: array<u32>;
@group(0) @binding(5) var<storage, read> gc5: array<u32>;
@group(0) @binding(6) var<storage, read> gc6: array<u32>;
@group(0) @binding(7) var<storage, read> gc7: array<u32>;
@group(0) @binding(8) var<storage, read> gc_idx: array<i32>;
@group(0) @binding(9) var<storage, read_write> gc_out: array<u32>;
@group(0) @binding(10) var<uniform> gc_params: GcParams;

fn gc_at(c: u32, i: u32) -> u32 {
    if (c == 0u) { return gc0[i]; }
    if (c == 1u) { return gc1[i]; }
    if (c == 2u) { return gc2[i]; }
    if (c == 3u) { return gc3[i]; }
    if (c == 4u) { return gc4[i]; }
    if (c == 5u) { return gc5[i]; }
    if (c == 6u) { return gc6[i]; }
    return gc7[i];
}

@compute @workgroup_size(256)
fn e4b_gather_chunks(@builtin(local_invocation_id) tid: vec3<u32>) {
    var s = u32(max(gc_idx[0], 0));
    if (s >= gc_params.total_rows) {
        s = 0u;
    }
    var c = s / gc_params.rows_per_chunk;
    if (c >= gc_params.n_chunks) {
        c = gc_params.n_chunks - 1u;
    }
    let r = s - c * gc_params.rows_per_chunk;
    let rw = gc_params.row_words;
    let base = r * rw;
    for (var w = tid.x; w < rw; w = w + 256u) {
        gc_out[w] = gc_at(c, base + w);
    }
}

@compute @workgroup_size(256)
fn e4b_gather_chunks_mk(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let t = wg.x;
    var s = u32(max(gc_idx[t], 0));
    if (s >= gc_params.total_rows) {
        s = 0u;
    }
    var c = s / gc_params.rows_per_chunk;
    if (c >= gc_params.n_chunks) {
        c = gc_params.n_chunks - 1u;
    }
    let r = s - c * gc_params.rows_per_chunk;
    let rw = gc_params.row_words;
    let base = r * rw;
    let ob = t * rw;
    for (var w = tid.x; w < rw; w = w + 256u) {
        gc_out[ob + w] = gc_at(c, base + w);
    }
}
