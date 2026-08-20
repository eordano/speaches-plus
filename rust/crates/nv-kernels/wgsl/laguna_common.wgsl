
struct LgGatherParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
};

@group(0) @binding(30) var<storage, read> lgg_embed: array<u32>;
@group(0) @binding(31) var<storage, read> lgg_tok: array<u32>;
@group(0) @binding(32) var<storage, read_write> lgg_out: array<u32>;
@group(0) @binding(33) var<uniform> lgg_p: LgGatherParams;

@compute @workgroup_size(256)
fn lgw_gather_embed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= lgg_p.hidden_words) {
        return;
    }
    let tok = lgg_tok[0];
    if (tok < lgg_p.row_off || tok >= lgg_p.row_off + lgg_p.n_rows) {
        return;
    }
    let row = tok - lgg_p.row_off;
    lgg_out[i] = lgg_embed[row * lgg_p.hidden_words + i];
}

struct LgSiluParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(10) var<storage, read> lgs_gate: array<u32>;
@group(0) @binding(11) var<storage, read> lgs_up: array<u32>;
@group(0) @binding(12) var<storage, read_write> lgs_out: array<u32>;
@group(0) @binding(13) var<uniform> lgs_p: LgSiluParams;

@compute @workgroup_size(64)
fn lgw_silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= lgs_p.n_words) {
        return;
    }
    let g = lgs_gate[i];
    let u = lgs_up[i];
    let lo = bf16_lo(g);
    let hi = bf16_hi(g);
    let slo = lo / (1.0 + exp(-lo));
    let shi = hi / (1.0 + exp(-hi));
    lgs_out[i] = bf16_pack(slo * bf16_lo(u), shi * bf16_hi(u));
}

struct LgArgmaxParams {
    n: u32,
    groups: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(40) var<storage, read> lga_logits: array<f32>;
@group(0) @binding(41) var<storage, read_write> lga_pv: array<f32>;
@group(0) @binding(42) var<storage, read_write> lga_pi: array<u32>;
@group(0) @binding(43) var<storage, read_write> lga_out: array<u32>;
@group(0) @binding(44) var<uniform> lga_p: LgArgmaxParams;

var<workgroup> lga_v: array<f32, 256>;
var<workgroup> lga_i: array<u32, 256>;

fn lga_reduce(tid: u32) {
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let a = lga_v[tid];
            let bv = lga_v[tid + stride];
            if (bv > a || (bv == a && lga_i[tid + stride] < lga_i[tid])) {
                lga_v[tid] = bv;
                lga_i[tid] = lga_i[tid + stride];
            }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn lgw_argmax_stage1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let per = (lga_p.n + lga_p.groups - 1u) / lga_p.groups;
    let start = wid.x * per;
    let end = min(start + per, lga_p.n);
    var best = -3.4e38;
    var bi = 0u;
    var i = start + tid;
    loop {
        if (i >= end) { break; }
        let v = lga_logits[i];
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
        i = i + 256u;
    }
    lga_v[tid] = best;
    lga_i[tid] = bi;
    workgroupBarrier();
    lga_reduce(tid);
    if (tid == 0u) {
        lga_pv[wid.x] = lga_v[0];
        lga_pi[wid.x] = lga_i[0];
    }
}

@compute @workgroup_size(256)
fn lgw_argmax_stage2(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var best = -3.4e38;
    var bi = 0u;
    var i = tid;
    loop {
        if (i >= lga_p.groups) { break; }
        let v = lga_pv[i];
        let idx = lga_pi[i];
        if (v > best || (v == best && idx < bi)) {
            best = v;
            bi = idx;
        }
        i = i + 256u;
    }
    lga_v[tid] = best;
    lga_i[tid] = bi;
    workgroupBarrier();
    lga_reduce(tid);
    if (tid == 0u) {
        lga_out[0] = lga_i[0];
    }
}
