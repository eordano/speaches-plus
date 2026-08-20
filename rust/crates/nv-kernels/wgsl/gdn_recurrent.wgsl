struct GdnRecParams {
    batch: u32,
    seq: u32,
    heads: u32,
    pairs: u32,
};

@group(0) @binding(0) var<storage, read> gr_q: array<f32>;
@group(0) @binding(1) var<storage, read> gr_k: array<f32>;
@group(0) @binding(2) var<storage, read> gr_v: array<f32>;
@group(0) @binding(3) var<storage, read> gr_g: array<f32>;
@group(0) @binding(4) var<storage, read> gr_beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> gr_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> gr_state: array<f32>;
@group(0) @binding(7) var<uniform> gr_params: GdnRecParams;

const GR_DIM: u32 = 128u;
const GR_STATE: u32 = 16384u;

var<workgroup> gr_kbuf: array<f32, 128>;
var<workgroup> gr_qbuf: array<f32, 128>;

@compute @workgroup_size(128)
fn gdn_recurrent_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let bh = wg.y * nwg.x + wg.x;
    let lid = tid.x;

    if (bh >= gr_params.pairs) {
        return;
    }

    let b = bh / gr_params.heads;
    let h = bh % gr_params.heads;

    var st: array<vec4<f32>, 32>;
    for (var i = 0u; i < 32u; i = i + 1u) {
        st[i] = vec4<f32>(0.0);
    }

    for (var t = 0u; t < gr_params.seq; t = t + 1u) {
        let kv_base = (b * gr_params.seq + t) * gr_params.heads + h;
        let vec_base = kv_base * GR_DIM;

        let ge = gr_g[kv_base];
        let bt = gr_beta[kv_base];

        gr_kbuf[lid] = gr_k[vec_base + lid];
        gr_qbuf[lid] = gr_q[vec_base + lid];
        workgroupBarrier();

        let v_t = gr_v[vec_base + lid];

        var kv_mem = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let kk = i * 4u;
            let kv = vec4<f32>(
                gr_kbuf[kk], gr_kbuf[kk + 1u], gr_kbuf[kk + 2u], gr_kbuf[kk + 3u]);
            let s = st[i] * ge;
            st[i] = s;
            kv_mem = fma(s.x, kv.x, kv_mem);
            kv_mem = fma(s.y, kv.y, kv_mem);
            kv_mem = fma(s.z, kv.z, kv_mem);
            kv_mem = fma(s.w, kv.w, kv_mem);
        }

        let delta = (v_t - kv_mem) * bt;

        var out_v = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let kk = i * 4u;
            let kv = vec4<f32>(
                gr_kbuf[kk], gr_kbuf[kk + 1u], gr_kbuf[kk + 2u], gr_kbuf[kk + 3u]);
            let qv = vec4<f32>(
                gr_qbuf[kk], gr_qbuf[kk + 1u], gr_qbuf[kk + 2u], gr_qbuf[kk + 3u]);
            let s = fma(kv, vec4<f32>(delta), st[i]);
            st[i] = s;
            out_v = fma(s.x, qv.x, out_v);
            out_v = fma(s.y, qv.y, out_v);
            out_v = fma(s.z, qv.z, out_v);
            out_v = fma(s.w, qv.w, out_v);
        }
        gr_out[vec_base + lid] = out_v;
        workgroupBarrier();
    }

    let sbase = bh * GR_STATE;
    for (var i = 0u; i < 32u; i = i + 1u) {
        let kk = i * 4u;
        let s = st[i];
        gr_state[sbase + kk * GR_DIM + lid] = s.x;
        gr_state[sbase + (kk + 1u) * GR_DIM + lid] = s.y;
        gr_state[sbase + (kk + 2u) * GR_DIM + lid] = s.z;
        gr_state[sbase + (kk + 3u) * GR_DIM + lid] = s.w;
    }
}
