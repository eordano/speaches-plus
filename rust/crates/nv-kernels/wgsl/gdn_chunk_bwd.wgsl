
struct GdnChunkParams {
    batch: u32,
    seq: u32,
    heads: u32,
    pairs: u32,
};

@group(0) @binding(0) var<storage, read> gc_q: array<f32>;
@group(0) @binding(1) var<storage, read> gc_k: array<f32>;
@group(0) @binding(2) var<storage, read> gc_v: array<f32>;
@group(0) @binding(3) var<storage, read> gc_gb: array<f32>;
@group(0) @binding(4) var<storage, read_write> gc_states: array<f32>;
@group(0) @binding(5) var<storage, read_write> gc_out: array<f32>;
@group(0) @binding(6) var<storage, read> gc_dout: array<f32>;
@group(0) @binding(7) var<storage, read_write> gc_dstate: array<f32>;
@group(0) @binding(8) var<storage, read_write> gc_dqkv: array<f32>;
@group(0) @binding(9) var<storage, read_write> gc_dgb: array<f32>;
@group(0) @binding(10) var<uniform> gc_p: GdnChunkParams;

const GC_DIM: u32 = 128u;
const GC_STATE: u32 = 16384u;

var<workgroup> gc_kbuf: array<f32, 128>;
var<workgroup> gc_qbuf: array<f32, 128>;
var<workgroup> gc_dobuf: array<f32, 128>;
var<workgroup> gc_dqred: array<vec4<f32>, 128>;
var<workgroup> gc_dkred: array<vec4<f32>, 128>;
var<workgroup> gc_scred: array<vec2<f32>, 128>;

@compute @workgroup_size(128)
fn gdn_chunk_fwd_states_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let bh = wg.y * nwg.x + wg.x;
    let lid = tid.x;
    if (bh >= gc_p.pairs) {
        return;
    }
    let b = bh / gc_p.heads;
    let h = bh % gc_p.heads;
    let srow = bh * (gc_p.seq + 1u);

    var st: array<vec4<f32>, 32>;
    let s0 = srow * GC_STATE;
    for (var i = 0u; i < 32u; i = i + 1u) {
        let kk = i * 4u;
        st[i] = vec4<f32>(
            gc_states[s0 + kk * GC_DIM + lid],
            gc_states[s0 + (kk + 1u) * GC_DIM + lid],
            gc_states[s0 + (kk + 2u) * GC_DIM + lid],
            gc_states[s0 + (kk + 3u) * GC_DIM + lid]);
    }

    for (var t = 0u; t < gc_p.seq; t = t + 1u) {
        let kv_base = (b * gc_p.seq + t) * gc_p.heads + h;
        let vec_base = kv_base * GC_DIM;
        let ge = gc_gb[kv_base * 2u];
        let bt = gc_gb[kv_base * 2u + 1u];
        gc_kbuf[lid] = gc_k[vec_base + lid];
        gc_qbuf[lid] = gc_q[vec_base + lid];
        workgroupBarrier();
        let v_t = gc_v[vec_base + lid];

        var kv_mem = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let kk = i * 4u;
            let kv = vec4<f32>(
                gc_kbuf[kk], gc_kbuf[kk + 1u], gc_kbuf[kk + 2u], gc_kbuf[kk + 3u]);
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
                gc_kbuf[kk], gc_kbuf[kk + 1u], gc_kbuf[kk + 2u], gc_kbuf[kk + 3u]);
            let qv = vec4<f32>(
                gc_qbuf[kk], gc_qbuf[kk + 1u], gc_qbuf[kk + 2u], gc_qbuf[kk + 3u]);
            let s = fma(kv, vec4<f32>(delta), st[i]);
            st[i] = s;
            out_v = fma(s.x, qv.x, out_v);
            out_v = fma(s.y, qv.y, out_v);
            out_v = fma(s.z, qv.z, out_v);
            out_v = fma(s.w, qv.w, out_v);
        }
        gc_out[vec_base + lid] = out_v;

        let sb = (srow + t + 1u) * GC_STATE;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let kk = i * 4u;
            let s = st[i];
            gc_states[sb + kk * GC_DIM + lid] = s.x;
            gc_states[sb + (kk + 1u) * GC_DIM + lid] = s.y;
            gc_states[sb + (kk + 2u) * GC_DIM + lid] = s.z;
            gc_states[sb + (kk + 3u) * GC_DIM + lid] = s.w;
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(128)
fn gdn_chunk_bwd_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let bh = wg.y * nwg.x + wg.x;
    let lid = tid.x;
    if (bh >= gc_p.pairs) {
        return;
    }
    let b = bh / gc_p.heads;
    let h = bh % gc_p.heads;
    let srow = bh * (gc_p.seq + 1u);
    let plane = gc_p.batch * gc_p.seq * gc_p.heads * GC_DIM;

    var gcar: array<vec4<f32>, 32>;
    let d0 = bh * GC_STATE;
    for (var i = 0u; i < 32u; i = i + 1u) {
        let kk = i * 4u;
        gcar[i] = vec4<f32>(
            gc_dstate[d0 + kk * GC_DIM + lid],
            gc_dstate[d0 + (kk + 1u) * GC_DIM + lid],
            gc_dstate[d0 + (kk + 2u) * GC_DIM + lid],
            gc_dstate[d0 + (kk + 3u) * GC_DIM + lid]);
    }

    for (var tt = 0u; tt < gc_p.seq; tt = tt + 1u) {
        let t = gc_p.seq - 1u - tt;
        let kv_base = (b * gc_p.seq + t) * gc_p.heads + h;
        let vec_base = kv_base * GC_DIM;
        let ge = gc_gb[kv_base * 2u];
        let bt = gc_gb[kv_base * 2u + 1u];
        workgroupBarrier();
        gc_kbuf[lid] = gc_k[vec_base + lid];
        gc_qbuf[lid] = gc_q[vec_base + lid];
        gc_dobuf[lid] = gc_dout[vec_base + lid];
        workgroupBarrier();
        let v_c = gc_v[vec_base + lid];
        let do_c = gc_dobuf[lid];

        var sp: array<vec4<f32>, 32>;
        let sb = (srow + t) * GC_STATE;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let kk = i * 4u;
            sp[i] = vec4<f32>(
                gc_states[sb + kk * GC_DIM + lid],
                gc_states[sb + (kk + 1u) * GC_DIM + lid],
                gc_states[sb + (kk + 2u) * GC_DIM + lid],
                gc_states[sb + (kk + 3u) * GC_DIM + lid]);
        }

        var kv_mem = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let kk = i * 4u;
            let kv = vec4<f32>(
                gc_kbuf[kk], gc_kbuf[kk + 1u], gc_kbuf[kk + 2u], gc_kbuf[kk + 3u]);
            let s = sp[i] * ge;
            kv_mem = fma(s.x, kv.x, kv_mem);
            kv_mem = fma(s.y, kv.y, kv_mem);
            kv_mem = fma(s.z, kv.z, kv_mem);
            kv_mem = fma(s.w, kv.w, kv_mem);
        }
        let delta_c = (v_c - kv_mem) * bt;

        var ddelta = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let kk = i * 4u;
            let kv = vec4<f32>(
                gc_kbuf[kk], gc_kbuf[kk + 1u], gc_kbuf[kk + 2u], gc_kbuf[kk + 3u]);
            let qv = vec4<f32>(
                gc_qbuf[kk], gc_qbuf[kk + 1u], gc_qbuf[kk + 2u], gc_qbuf[kk + 3u]);
            let g1 = fma(qv, vec4<f32>(do_c), gcar[i]);
            ddelta = ddelta + dot(g1, kv);
        }
        let dv_c = bt * ddelta;
        let dm_c = -bt * ddelta;
        gc_dqkv[2u * plane + vec_base + lid] = dv_c;
        let db_part = ddelta * (v_c - kv_mem);
        var dge_part = 0.0;

        for (var i = 0u; i < 32u; i = i + 1u) {
            let kk = i * 4u;
            let kv = vec4<f32>(
                gc_kbuf[kk], gc_kbuf[kk + 1u], gc_kbuf[kk + 2u], gc_kbuf[kk + 3u]);
            let qv = vec4<f32>(
                gc_qbuf[kk], gc_qbuf[kk + 1u], gc_qbuf[kk + 2u], gc_qbuf[kk + 3u]);
            let s_hat = sp[i] * ge;
            let st4 = fma(kv, vec4<f32>(delta_c), s_hat);
            let g1 = fma(qv, vec4<f32>(do_c), gcar[i]);
            let g2 = fma(kv, vec4<f32>(dm_c), g1);
            dge_part = dge_part + dot(sp[i], g2);
            gcar[i] = g2 * ge;
            gc_dqred[lid] = st4 * do_c;
            gc_dkred[lid] = g1 * delta_c + s_hat * dm_c;
            workgroupBarrier();
            for (var s = 64u; s > 0u; s = s >> 1u) {
                if (lid < s) {
                    gc_dqred[lid] = gc_dqred[lid] + gc_dqred[lid + s];
                    gc_dkred[lid] = gc_dkred[lid] + gc_dkred[lid + s];
                }
                workgroupBarrier();
            }
            if (lid < 4u) {
                var dq4 = gc_dqred[0u];
                var dk4 = gc_dkred[0u];
                gc_dqkv[vec_base + kk + lid] = dq4[lid];
                gc_dqkv[plane + vec_base + kk + lid] = dk4[lid];
            }
            workgroupBarrier();
        }

        gc_scred[lid] = vec2<f32>(dge_part, db_part);
        workgroupBarrier();
        for (var s = 64u; s > 0u; s = s >> 1u) {
            if (lid < s) {
                gc_scred[lid] = gc_scred[lid] + gc_scred[lid + s];
            }
            workgroupBarrier();
        }
        if (lid == 0u) {
            gc_dgb[kv_base * 2u] = gc_scred[0u].x;
            gc_dgb[kv_base * 2u + 1u] = gc_scred[0u].y;
        }
    }

    for (var i = 0u; i < 32u; i = i + 1u) {
        let kk = i * 4u;
        let s = gcar[i];
        gc_dstate[d0 + kk * GC_DIM + lid] = s.x;
        gc_dstate[d0 + (kk + 1u) * GC_DIM + lid] = s.y;
        gc_dstate[d0 + (kk + 2u) * GC_DIM + lid] = s.z;
        gc_dstate[d0 + (kk + 3u) * GC_DIM + lid] = s.w;
    }
}
