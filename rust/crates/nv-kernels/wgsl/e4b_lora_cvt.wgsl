
struct E4bLoraCvt {
    m: u32,
    width: u32,
    pk_row_words: u32,
    wide_row_elems: u32,
    wide_col_off: u32,
    total: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> e4bl_pk: array<u32>;
@group(0) @binding(1) var<storage, read_write> e4bl_wide: array<u32>;
@group(0) @binding(2) var<uniform> e4bl_p: E4bLoraCvt;

@compute @workgroup_size(256)
fn e4b_lora_widen(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= e4bl_p.total) {
        return;
    }
    let r = i / e4bl_p.width;
    let c = i - r * e4bl_p.width;
    let word = e4bl_pk[r * e4bl_p.pk_row_words + (c >> 1u)];
    let piece = select(word & 0xffffu, word >> 16u, (c & 1u) == 1u);
    e4bl_wide[r * e4bl_p.wide_row_elems + e4bl_p.wide_col_off + c] = piece;
}

@compute @workgroup_size(256)
fn e4b_lora_repack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= e4bl_p.total) {
        return;
    }
    let pairs = e4bl_p.width >> 1u;
    let r = i / pairs;
    let c = (i - r * pairs) << 1u;
    let base = r * e4bl_p.wide_row_elems + e4bl_p.wide_col_off + c;
    let lo = e4bl_wide[base] & 0xffffu;
    let hi = e4bl_wide[base + 1u] & 0xffffu;
    e4bl_pk[r * e4bl_p.pk_row_words + (c >> 1u)] = lo | (hi << 16u);
}
