
struct Q3smParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(10) var<storage, read> sm_g: array<u32>;
@group(0) @binding(11) var<storage, read> sm_u: array<u32>;
@group(0) @binding(12) var<storage, read_write> sm_y: array<u32>;
@group(0) @binding(13) var<uniform> sm_p: Q3smParams;

@compute @workgroup_size(64)
fn q3w_silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= sm_p.n_words) {
        return;
    }
    let gw = sm_g[w];
    let uw = sm_u[w];
    let g0 = bf16_lo(gw);
    let g1 = bf16_hi(gw);
    let a0 = bf16_decode(bf16_encode(g0 / (1.0 + exp(-g0)))) * bf16_lo(uw);
    let a1 = bf16_decode(bf16_encode(g1 / (1.0 + exp(-g1)))) * bf16_hi(uw);
    sm_y[w] = bf16_pack(a0, a1);
}

struct Q3geParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
};

@group(0) @binding(30) var<storage, read> ge_emb: array<u32>;
@group(0) @binding(31) var<storage, read> ge_tok: array<i32>;
@group(0) @binding(32) var<storage, read_write> ge_out: array<u32>;
@group(0) @binding(33) var<uniform> ge_p: Q3geParams;

@compute @workgroup_size(256)
fn q3w_gather_embed(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    var s = 0u;
    if (ge_tok[0] > 0) {
        s = u32(ge_tok[0]);
    }
    if (s >= ge_p.vocab) {
        s = 0u;
    }
    if (s < ge_p.row_off) {
        return;
    }
    if (s >= ge_p.row_off + ge_p.n_rows) {
        return;
    }
    let base = (s - ge_p.row_off) * ge_p.hidden_words;
    let w = wid.x * 256u + lid.x;
    if (w >= ge_p.hidden_words) {
        return;
    }
    ge_out[w] = ge_emb[base + w];
}

