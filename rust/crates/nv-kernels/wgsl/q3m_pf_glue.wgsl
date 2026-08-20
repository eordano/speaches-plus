
struct PfCopyParams {
    rows: u32,
    row_words: u32,
    src_stride_words: u32,
    slots: u32,
};

@group(0) @binding(0) var<storage, read> pfc_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> pfc_dst: array<u32>;
@group(0) @binding(2) var<uniform> pfc_p: PfCopyParams;

@compute @workgroup_size(256)
fn q3w_pf_replicate_token_row_to_slots(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pfc_p.rows * pfc_p.row_words) {
        return;
    }
    let r = i / pfc_p.row_words;
    let w = i - r * pfc_p.row_words;
    let t = r / pfc_p.slots;
    pfc_dst[i] = pfc_src[t * pfc_p.src_stride_words + w];
}

@compute @workgroup_size(64)
fn q3w_pf_pack_padded_ids_to_flat_sel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pfc_p.rows * pfc_p.row_words) {
        return;
    }
    let t = i / pfc_p.row_words;
    let j = i - t * pfc_p.row_words;
    pfc_dst[i] = pfc_src[t * pfc_p.src_stride_words + j];
}

struct PfGatherParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
};

@group(0) @binding(10) var<storage, read> pfg_emb: array<u32>;
@group(0) @binding(11) var<storage, read> pfg_tok: array<i32>;
@group(0) @binding(12) var<storage, read_write> pfg_out: array<u32>;
@group(0) @binding(13) var<uniform> pfg_p: PfGatherParams;

@compute @workgroup_size(256)
fn q3w_pf_gather_embed_m(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = wid.y;
    var s = 0u;
    if (pfg_tok[t] > 0) {
        s = u32(pfg_tok[t]);
    }
    if (s >= pfg_p.vocab) {
        s = 0u;
    }
    if (s < pfg_p.row_off || s >= pfg_p.row_off + pfg_p.n_rows) {
        return;
    }
    let w = wid.x * 256u + lid.x;
    if (w >= pfg_p.hidden_words) {
        return;
    }
    pfg_out[t * pfg_p.hidden_words + w] = pfg_emb[(s - pfg_p.row_off) * pfg_p.hidden_words + w];
}
