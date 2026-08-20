
struct QgPkParams {
    dst_word_off: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct QgSplitParams {
    q_rows: u32,
    kv_rows: u32,
    v_off: u32,
    pad0: u32,
};

@group(0) @binding(30) var<uniform> qg_pk_params: QgPkParams;
@group(0) @binding(31) var<storage, read_write> qg_y_q: array<u32>;
@group(0) @binding(32) var<storage, read_write> qg_y_k: array<u32>;
@group(0) @binding(33) var<storage, read_write> qg_y_v: array<u32>;
@group(0) @binding(34) var<uniform> qg_split_params: QgSplitParams;

fn qg_scatter(row: u32, word: u32) {
    if (row < qg_split_params.q_rows) {
        qg_y_q[row >> 1u] = word;
        return;
    }
    let kr = row - qg_split_params.q_rows;
    if (kr < qg_split_params.kv_rows) {
        qg_y_k[kr >> 1u] = word;
    }
    if (row >= qg_split_params.v_off) {
        let vr = row - qg_split_params.v_off;
        if (vr < qg_split_params.kv_rows) {
            qg_y_v[vr >> 1u] = word;
        }
    }
}
