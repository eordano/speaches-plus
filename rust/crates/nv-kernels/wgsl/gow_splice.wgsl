struct GowSpParams { hidden_words: u32, m: u32, pad0: u32, pad1: u32 };
@group(0) @binding(4) var<storage, read> gsp_rows: array<u32>;
@group(0) @binding(5) var<storage, read> gsp_mask: array<u32>;
@group(0) @binding(6) var<uniform> gsp_p: GowSpParams;
@compute @workgroup_size(256)
fn gow_pf_splice_embed_rows(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = wid.y;
    if (gsp_mask[t] == 0u) { return; }
    let w = wid.x * 256u + lid.x;
    if (w >= gsp_p.hidden_words) { return; }
    pge_out[t * gsp_p.hidden_words + w] = gsp_rows[t * gsp_p.hidden_words + w];
}
