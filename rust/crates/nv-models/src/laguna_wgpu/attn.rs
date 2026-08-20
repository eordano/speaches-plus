use anyhow::Result;

use super::config::{window_start, GateKind, LagunaShapes, LayerShape, MAX_HEAD_DIM};
use super::gpu::{
    alloc_lin_scratch, push_lin_gemv, upload_lin, Builder, Sources, StepBuffers, W8Scope,
};
use super::weights::{bf16_val, pack_pairs, HostAttention};
use super::{rbf, ref_gemv_lin, softplus, RefState};

pub const ATTN_WGSL: &str = include_str!("../../../nv-kernels/wgsl/laguna_attn.wgsl");

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct NormRopeParams {
    pub n_rows: u32,
    pub head_dim: u32,
    pub src_stride: u32,
    pub rot_half: u32,
    pub eps: f32,
    pub out_scale: f32,
    pub rope_table_half: u32,
    pub pad0: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct KvWriteParams {
    pub kv_words: u32,
    pub kv_capacity: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct AttnDecodeParams {
    pub n_q_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub gqa_group: u32,
    pub kv_capacity: u32,
    pub is_sliding: u32,
    pub scale: f32,
    pub pad0: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct AttnGateParams {
    pub n_words: u32,
    pub head_dim: u32,
    pub gate_kind: u32,
    pub n_q_heads: u32,
}

pub struct RopeTablesGpu {
    pub cos: wgpu::Buffer,
    pub sin: wgpu::Buffer,
    pub half: usize,
}

pub fn gate_code(kind: GateKind) -> u32 {
    match kind {
        GateKind::None => 0,
        GateKind::PerHead => 1,
        GateKind::PerElement => 2,
    }
}

fn wbytes(elems: usize) -> u64 {
    (elems.div_ceil(2).max(1) * 4) as u64
}

#[allow(clippy::too_many_arguments)]
pub fn build_attn_layer(
    b: &mut Builder,
    s: &Sources,
    shapes: &LagunaShapes,
    layer: &LayerShape,
    w: &HostAttention,
    x_normed_packed: &wgpu::Buffer,
    out_packed: &wgpu::Buffer,
    step: &StepBuffers,
    rope: &RopeTablesGpu,
) -> Result<()> {
    let li = layer.idx;
    let hidden = shapes.hidden_size;
    let hd = layer.head_dim;
    let n_q = layer.num_q_heads;
    let n_kv = layer.num_kv_heads;
    let q_rows = layer.q_rows;
    let kv_rows = layer.kv_rows;
    let words = hd / 2;
    let rot_half = layer.rotary_dim / 2;

    anyhow::ensure!(
        hd.is_multiple_of(2) && hd <= MAX_HEAD_DIM,
        "layer {li}: head_dim {hd} must be even and <= {MAX_HEAD_DIM}"
    );
    anyhow::ensure!(
        n_kv > 0 && layer.gqa_group * n_kv == n_q,
        "layer {li}: {n_q} q heads not a multiple of {n_kv} kv heads"
    );
    anyhow::ensure!(
        rope.half == rot_half,
        "layer {li}: rope table half {} != rotary_dim/2 {rot_half}",
        rope.half
    );
    anyhow::ensure!(
        w.q.n() == q_rows && w.q.k() == hidden,
        "layer {li}: q_proj is [{},{}], want [{q_rows},{hidden}]",
        w.q.n(),
        w.q.k()
    );
    anyhow::ensure!(
        w.k.n() == kv_rows && w.k.k() == hidden,
        "layer {li}: k_proj is [{},{}], want [{kv_rows},{hidden}]",
        w.k.n(),
        w.k.k()
    );
    anyhow::ensure!(
        w.v.n() == kv_rows && w.v.k() == hidden,
        "layer {li}: v_proj is [{},{}], want [{kv_rows},{hidden}]",
        w.v.n(),
        w.v.k()
    );
    anyhow::ensure!(
        w.o.n() == hidden && w.o.k() == q_rows,
        "layer {li}: o_proj is [{},{}], want [{hidden},{q_rows}]",
        w.o.n(),
        w.o.k()
    );
    anyhow::ensure!(
        w.q_norm.len() == hd && w.k_norm.len() == hd,
        "layer {li}: q_norm/k_norm must be [{hd}], got {}/{}",
        w.q_norm.len(),
        w.k_norm.len()
    );
    anyhow::ensure!(
        (shapes.gate_kind == GateKind::None) == w.g.is_none(),
        "layer {li}: gating {:?} disagrees with g_proj presence",
        shapes.gate_kind
    );
    if let Some(g) = &w.g {
        anyhow::ensure!(
            g.n() == layer.gate_rows && g.k() == hidden,
            "layer {li}: g_proj is [{},{}], want [{},{hidden}]",
            g.n(),
            g.k(),
            layer.gate_rows
        );
    }

    let qw = upload_lin(b, &format!("lgw-l{li}-qw"), &w.q, W8Scope::Attn);
    let kw = upload_lin(b, &format!("lgw-l{li}-kw"), &w.k, W8Scope::Attn);
    let vw = upload_lin(b, &format!("lgw-l{li}-vw"), &w.v, W8Scope::Attn);
    let ow = upload_lin(b, &format!("lgw-l{li}-ow"), &w.o, W8Scope::Attn);
    let qs = alloc_lin_scratch(b, &format!("lgw-l{li}-qs"), &qw);
    let ks = alloc_lin_scratch(b, &format!("lgw-l{li}-ks"), &kw);
    let vs = alloc_lin_scratch(b, &format!("lgw-l{li}-vs"), &vw);
    let os = alloc_lin_scratch(b, &format!("lgw-l{li}-os"), &ow);

    let q_buf = b.zeros(&format!("lgw-l{li}-q"), wbytes(q_rows));
    let k_buf = b.zeros(&format!("lgw-l{li}-k"), wbytes(kv_rows));
    let v_buf = b.zeros(&format!("lgw-l{li}-v"), wbytes(kv_rows));
    let q_rot = b.zeros(&format!("lgw-l{li}-qrot"), wbytes(q_rows));
    let k_rot = b.zeros(&format!("lgw-l{li}-krot"), wbytes(kv_rows));
    let heads = b.zeros(&format!("lgw-l{li}-heads"), wbytes(q_rows));

    push_lin_gemv(
        b,
        s,
        &format!("lgw-l{li}-qproj"),
        &qw,
        &qs,
        x_normed_packed,
        &q_buf,
    )?;
    push_lin_gemv(
        b,
        s,
        &format!("lgw-l{li}-kproj"),
        &kw,
        &ks,
        x_normed_packed,
        &k_buf,
    )?;
    push_lin_gemv(
        b,
        s,
        &format!("lgw-l{li}-vproj"),
        &vw,
        &vs,
        x_normed_packed,
        &v_buf,
    )?;

    let gate = match (&w.g, shapes.gate_kind) {
        (Some(g), GateKind::PerHead) | (Some(g), GateKind::PerElement) => {
            let gw = upload_lin(b, &format!("lgw-l{li}-gw"), g, W8Scope::Attn);
            let gs = alloc_lin_scratch(b, &format!("lgw-l{li}-gs"), &gw);
            let g_buf = b.zeros(&format!("lgw-l{li}-g"), wbytes(layer.gate_rows));
            push_lin_gemv(
                b,
                s,
                &format!("lgw-l{li}-gproj"),
                &gw,
                &gs,
                x_normed_packed,
                &g_buf,
            )?;
            Some(g_buf)
        }
        _ => None,
    };

    let qn = b.upload_u32(&format!("lgw-l{li}-qn"), &pack_pairs(&w.q_norm));
    let kn = b.upload_u32(&format!("lgw-l{li}-kn"), &pack_pairs(&w.k_norm));

    for (label, n_rows, src, nw, dst) in [
        (format!("lgw-l{li}-qnr"), n_q, &q_buf, &qn, &q_rot),
        (format!("lgw-l{li}-knr"), n_kv, &k_buf, &kn, &k_rot),
    ] {
        let p = b.uni(
            &format!("{label}-p"),
            NormRopeParams {
                n_rows: n_rows as u32,
                head_dim: hd as u32,
                src_stride: words as u32,
                rot_half: rot_half as u32,
                eps: shapes.rms_norm_eps,
                out_scale: layer.rope_out_scale,
                rope_table_half: rope.half as u32,
                pad0: 0,
            },
        );
        let grid = b.grid1(n_rows as u64, 1);
        b.push(
            &label,
            &s.attn,
            "lgw_norm_rope",
            &[
                (0, src),
                (1, nw),
                (2, &rope.cos),
                (3, &rope.sin),
                (4, &step.step),
                (5, &p),
                (6, dst),
            ],
            grid,
        )?;
    }

    let cap = layer.kv_capacity_tokens;
    let kv_words = kv_rows / 2;
    let kc = b.state_zeros(&format!("lgw-l{li}-kc"), (cap * kv_words * 4) as u64);
    let vc = b.state_zeros(&format!("lgw-l{li}-vc"), (cap * kv_words * 4) as u64);
    for (label, src, dst) in [
        (format!("lgw-l{li}-kwrite"), &k_rot, &kc),
        (format!("lgw-l{li}-vwrite"), &v_buf, &vc),
    ] {
        let p = b.uni(
            &format!("{label}-p"),
            KvWriteParams {
                kv_words: kv_words as u32,
                kv_capacity: cap as u32,
                pad0: 0,
                pad1: 0,
            },
        );
        let grid = b.grid1(kv_words as u64, 64);
        b.push(
            &label,
            &s.attn,
            "lgw_kv_write",
            &[(10, src), (11, &step.step), (12, &p), (13, dst)],
            grid,
        )?;
    }

    let dp = b.uni(
        &format!("lgw-l{li}-adp"),
        AttnDecodeParams {
            n_q_heads: n_q as u32,
            n_kv_heads: n_kv as u32,
            head_dim: hd as u32,
            gqa_group: layer.gqa_group as u32,
            kv_capacity: cap as u32,
            is_sliding: u32::from(layer.is_sliding()),
            scale: layer.attn_softmax_scale,
            pad0: 0,
        },
    );
    let grid = b.grid1(n_q as u64, 1);
    b.push(
        &format!("lgw-l{li}-attn"),
        &s.attn,
        "lgw_attn_decode",
        &[
            (20, &q_rot),
            (21, &kc),
            (22, &vc),
            (23, &step.uni),
            (24, &dp),
            (25, &heads),
        ],
        grid,
    )?;

    let mixed = match &gate {
        Some(g_buf) => {
            let gated = b.zeros(&format!("lgw-l{li}-gated"), wbytes(q_rows));
            let gp = b.uni(
                &format!("lgw-l{li}-agp"),
                AttnGateParams {
                    n_words: (q_rows / 2) as u32,
                    head_dim: hd as u32,
                    gate_kind: gate_code(shapes.gate_kind),
                    n_q_heads: n_q as u32,
                },
            );
            let grid = b.grid1((q_rows / 2) as u64, 64);
            b.push(
                &format!("lgw-l{li}-gate"),
                &s.attn,
                "lgw_attn_gate",
                &[(30, &heads), (31, g_buf), (32, &gp), (33, &gated)],
                grid,
            )?;
            gated
        }
        None => heads,
    };

    push_lin_gemv(
        b,
        s,
        &format!("lgw-l{li}-oproj"),
        &ow,
        &os,
        &mixed,
        out_packed,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn ref_attn(
    shapes: &LagunaShapes,
    layer: &LayerShape,
    w: &HostAttention,
    x_normed: &[f32],
    rope_cos_row: &[f32],
    rope_sin_row: &[f32],
    st: &mut RefState,
    pos_tokens: usize,
) -> Result<Vec<f32>> {
    let li = layer.idx;
    let hd = layer.head_dim;
    let n_q = layer.num_q_heads;
    let n_kv = layer.num_kv_heads;
    let rot_half = layer.rotary_dim / 2;
    anyhow::ensure!(
        rope_cos_row.len() >= rot_half && rope_sin_row.len() >= rot_half,
        "layer {li}: rope row has {} entries, need {rot_half}",
        rope_cos_row.len()
    );

    let mut q: Vec<f32> = ref_gemv_lin(&w.q, x_normed).into_iter().map(rbf).collect();
    let mut k: Vec<f32> = ref_gemv_lin(&w.k, x_normed).into_iter().map(rbf).collect();
    let v: Vec<f32> = ref_gemv_lin(&w.v, x_normed).into_iter().map(rbf).collect();

    ref_head_rmsnorm(&mut q, n_q, hd, &w.q_norm, shapes.rms_norm_eps);
    ref_head_rmsnorm(&mut k, n_kv, hd, &w.k_norm, shapes.rms_norm_eps);
    ref_partial_rope_inplace(
        &mut q,
        n_q,
        hd,
        layer.rotary_dim,
        rope_cos_row,
        rope_sin_row,
        layer.rope_out_scale,
    );
    ref_partial_rope_inplace(
        &mut k,
        n_kv,
        hd,
        layer.rotary_dim,
        rope_cos_row,
        rope_sin_row,
        layer.rope_out_scale,
    );

    st.kc[li].extend_from_slice(&k);
    st.vc[li].extend_from_slice(&v);
    let total = st.kc[li].len() / layer.kv_rows;
    anyhow::ensure!(
        total == pos_tokens + 1,
        "layer {li}: kv cache holds {total} tokens at position {pos_tokens}"
    );
    let start = window_start(total, layer.window_tokens);

    let mut out = vec![0f32; layer.attn_out_elems()];
    let mut probs = vec![0f32; total];
    for h in 0..n_q {
        let kvh = h / layer.gqa_group;
        let qb = h * hd;
        let mut m = f32::NEG_INFINITY;
        for j in start..total {
            let kb = (j * n_kv + kvh) * hd;
            let mut acc = 0f32;
            for d in 0..hd {
                acc = q[qb + d].mul_add(st.kc[li][kb + d], acc);
            }
            let sc = acc * layer.attn_softmax_scale;
            probs[j] = sc;
            if sc > m {
                m = sc;
            }
        }
        let mut l = 0f32;
        for j in start..total {
            let p = (probs[j] - m).exp();
            probs[j] = p;
            l += p;
        }
        for j in start..total {
            let vb = (j * n_kv + kvh) * hd;
            let p = probs[j];
            for d in 0..hd {
                out[qb + d] = p.mul_add(st.vc[li][vb + d], out[qb + d]);
            }
        }
        let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
        for d in 0..hd {
            out[qb + d] = rbf(out[qb + d] * inv);
        }
    }

    if let Some(gl) = &w.g {
        let graw: Vec<f32> = ref_gemv_lin(gl, x_normed).into_iter().map(rbf).collect();
        for h in 0..n_q {
            for d in 0..hd {
                let gi = match shapes.gate_kind {
                    GateKind::None => continue,
                    GateKind::PerHead => h,
                    GateKind::PerElement => h * hd + d,
                };
                let gv = rbf(softplus(graw[gi]));
                out[h * hd + d] = rbf(out[h * hd + d] * gv);
            }
        }
    }

    Ok(ref_gemv_lin(&w.o, &out).into_iter().map(rbf).collect())
}

pub fn ref_partial_rope_inplace(
    vec_heads: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    cos_row: &[f32],
    sin_row: &[f32],
    out_scale: f32,
) {
    let half = rotary_dim / 2;
    for r in 0..n_heads {
        let base = r * head_dim;
        for i in 0..half {
            let a = vec_heads[base + i];
            let b = vec_heads[base + i + half];
            let c = cos_row[i];
            let s = sin_row[i];
            vec_heads[base + i] = rbf((a * c - b * s) * out_scale);
            vec_heads[base + i + half] = rbf((a * s + b * c) * out_scale);
        }
    }
}

pub fn ref_head_rmsnorm(
    rows: &mut [f32],
    n_rows: usize,
    head_dim: usize,
    norm_w: &[u16],
    eps: f32,
) {
    for r in 0..n_rows {
        let base = r * head_dim;
        let mut ss = 0f32;
        for d in 0..head_dim {
            ss = rows[base + d].mul_add(rows[base + d], ss);
        }
        let inv = 1.0 / (ss / head_dim as f32 + eps).sqrt();
        for d in 0..head_dim {
            rows[base + d] = rbf(rows[base + d] * inv * bf16_val(norm_w[d]));
        }
    }
}
