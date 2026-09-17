use anyhow::{ensure, Context, Result};
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch::{drive_blocking, ReadbackArm};
use nv_kernels::wgpu_backend::kernels::qlora_bwd_linear::{
    dx_w4_f32_arm, gemm_nn_f32_arm, lora_grads_f32_arm,
};
use nv_kernels::wgpu_backend::kernels::qlora_bwd_norm::rmsnorm_bwd_f32_arm;
use nv_kernels::wgpu_backend::kernels::qlora_bwd_rope::{rope_bwd_f32_arm, rope_fwd_f32_arm};
use nv_kernels::wgpu_qlora::qlora_attn::{
    grouped_attn_bwd_arm, grouped_attn_fwd_arm, grouped_to_kv_bskd, grouped_to_q_bshd,
    kv_bskd_to_grouped, q_bshd_to_grouped, GroupedAttnDims, GroupedAttnFwd,
};
use nv_kernels::wgpu_qlora::qlora_head_ce::{head_ce_bwd_arm, head_ce_fwd_arm, W4Head};
use nv_weights::WeightLoader;

use super::lora::{AdamW, LoraVar};
use super::peft;
use super::weights::{
    embed_rows_f32, gamma_f32, TrainDims, TrainLayer, W4Mod, FINAL_NORM, LM_HEAD_MODULE,
};

pub const BUCKET_PADDED_TOKEN_BUDGET_MATCHES_NV_TRAIN_DOTS: usize = 512;

pub const EXAMPLE_TOKEN_CEILING_WHERE_THE_NATIVE_LIVE_GRAPH_OOMS_INTO_THE_CHECKPOINT_FALLBACK:
    usize = 4070;

pub const LORA_LEAVES_IN_NV_TRAIN_ATTACH_ORDER: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

pub const ROW_WEIGHTS_PRESERVE_MEAN_OF_PER_EXAMPLE_MEANS: &str =
    "each example's target mask is divided by its own unmasked count before padding, so the \
     bucket loss is a plain sum of weighted CE rows and the step loss divides once by the \
     example count, exactly nv-train's mean-of-per-example-means; a flat mean over bucket rows \
     would overweight long examples and change the trajectory";

pub const PREFIX_LAYERS_STREAM_THROUGH_ONCE_AND_DROP_SO_ONLY_THE_WINDOW_STAYS_RESIDENT: &str =
    "layers below the trainable window are frozen and position-independent across steps, so \
     each is loaded, applied to every example's hidden states, and dropped before the next; \
     only the grad_layers window plus the lm_head hold GPU residency for the whole run";

pub struct TrainSample {
    pub ids: Vec<u32>,
    pub completion_start: usize,
    pub vision_start: usize,
    pub n_vis: usize,
    pub rows: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct TrainOptions {
    pub grad_layers: usize,
    pub rank: usize,
    pub alpha: f64,
    pub lr: f64,
    pub b_lr_ratio: f64,
    pub steps: usize,
    pub seed: u64,
    pub targets: Vec<String>,
    pub base_model_name: String,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            grad_layers: 8,
            rank: 16,
            alpha: 32.0,
            lr: 1e-4,
            b_lr_ratio: 1.0,
            steps: 20,
            seed: 7,
            targets: LORA_LEAVES_IN_NV_TRAIN_ATTACH_ORDER
                .iter()
                .map(|s| s.to_string())
                .collect(),
            base_model_name: "mlx-community/dots.ocr-4bit".to_string(),
        }
    }
}

pub struct TrainReport {
    pub losses: Vec<f32>,
    pub all_grads_finite: bool,
    pub window_start: usize,
    pub modules: Vec<String>,
    pub adapter_safetensors: Vec<u8>,
    pub adapter_config_json: Vec<u8>,
}

pub(crate) struct RopeTables {
    pub(crate) cos: Vec<f32>,
    pub(crate) sin: Vec<f32>,
    pub(crate) half: usize,
    pub(crate) max_seq: usize,
}

pub(crate) fn rope_tables(head_dim: usize, theta: f32, max_seq: usize) -> RopeTables {
    rope_tables_partial(head_dim, theta, max_seq, head_dim / 2)
}

pub(crate) fn rope_tables_partial(
    head_dim: usize,
    theta: f32,
    max_seq: usize,
    angles: usize,
) -> RopeTables {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| {
            if i < angles {
                1.0 / theta.powf((i as f32 * 2.0) / head_dim as f32)
            } else {
                0.0
            }
        })
        .collect();
    let mut cos = vec![0f32; max_seq * half];
    let mut sin = vec![0f32; max_seq * half];
    for p in 0..max_seq {
        for i in 0..half {
            let t = p as f32 * inv_freq[i];
            cos[p * half + i] = t.cos();
            sin[p * half + i] = t.sin();
        }
    }
    RopeTables {
        cos,
        sin,
        half,
        max_seq,
    }
}

pub fn refuse_rows_beyond_the_native_window_bound(samples: &[TrainSample]) -> Result<()> {
    for (i, s) in samples.iter().enumerate() {
        let t = s.ids.len().saturating_sub(1);
        ensure!(
            t <= EXAMPLE_TOKEN_CEILING_WHERE_THE_NATIVE_LIVE_GRAPH_OOMS_INTO_THE_CHECKPOINT_FALLBACK,
            "sample {i}: {t} shifted tokens is past the {} bound where the native live-graph \
             backward OOMs and scripts/ocr_corpus_bless.py retries under \
             NV_DOTS_QLORA_CHECKPOINT=1; this WGSL arm has no checkpointed walk and a single \
             bucket at that length overflows the browser attention buffers, so refuse here \
             instead -- re-dump the corpus with shorter rows or train that row on the native arm",
            EXAMPLE_TOKEN_CEILING_WHERE_THE_NATIVE_LIVE_GRAPH_OOMS_INTO_THE_CHECKPOINT_FALLBACK
        );
    }
    Ok(())
}

pub(crate) struct Ex {
    pub(crate) h: Vec<f32>,
    pub(crate) t: usize,
    pub(crate) tgt: Vec<u32>,
    pub(crate) mask: Vec<f32>,
    pub(crate) aux: Vec<f32>,
}

fn embed_spliced_shifted(loader: &WeightLoader, s: &TrainSample, hidden: usize) -> Result<Ex> {
    ensure!(
        s.ids.len() >= 2
            && s.vision_start + s.n_vis < s.completion_start
            && s.completion_start < s.ids.len(),
        "inconsistent sample: len {} vision_start {} n_vis {} completion_start {}",
        s.ids.len(),
        s.vision_start,
        s.n_vis,
        s.completion_start
    );
    ensure!(
        s.rows.len() == s.n_vis * hidden,
        "vision rows length {} != {} x {hidden}",
        s.rows.len(),
        s.n_vis
    );
    let t = s.ids.len() - 1;
    ensure!(
        s.vision_start + s.n_vis <= t,
        "vision span [{}, {}) exceeds shifted length {t}",
        s.vision_start,
        s.vision_start + s.n_vis
    );
    let mut h = embed_rows_f32(loader, &s.ids[..t], hidden)?;
    h[s.vision_start * hidden..(s.vision_start + s.n_vis) * hidden].copy_from_slice(&s.rows);
    let tgt = s.ids[1..].to_vec();
    let mask: Vec<f32> = (1..s.ids.len())
        .map(|p| if p >= s.completion_start { 1.0 } else { 0.0 })
        .collect();
    Ok(Ex {
        h,
        t,
        tgt,
        mask,
        aux: Vec::new(),
    })
}

pub(crate) struct Bucket {
    pub(crate) h0: Vec<f32>,
    pub(crate) members: Vec<usize>,
    pub(crate) rows: usize,
    pub(crate) tp: usize,
    pub(crate) tgt: Vec<u32>,
    pub(crate) wrow: Vec<f32>,
    pub(crate) positions: Vec<u32>,
    pub(crate) aux: Vec<f32>,
}

pub(crate) fn pack_buckets(exs: &[Ex], hidden: usize) -> Result<Vec<Bucket>> {
    pack_buckets_aux(exs, hidden, 0)
}

pub(crate) fn pack_buckets_aux(exs: &[Ex], hidden: usize, aux_width: usize) -> Result<Vec<Bucket>> {
    let mut order: Vec<usize> = (0..exs.len()).collect();
    order.sort_by_key(|&i| (usize::MAX - exs[i].t, i));
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for &i in &order {
        match groups.last_mut() {
            Some(g)
                if (g.len() + 1) * exs[g[0]].t
                    <= BUCKET_PADDED_TOKEN_BUDGET_MATCHES_NV_TRAIN_DOTS =>
            {
                g.push(i)
            }
            _ => groups.push(vec![i]),
        }
    }
    let mut out = Vec::with_capacity(groups.len());
    for g in &groups {
        let tp = exs[g[0]].t;
        let mut h0 = Vec::with_capacity(g.len() * tp * hidden);
        let mut tgt = Vec::with_capacity(g.len() * tp);
        let mut wrow = Vec::with_capacity(g.len() * tp);
        let mut positions = Vec::with_capacity(g.len() * tp);
        let mut aux = Vec::with_capacity(g.len() * tp * aux_width);
        for &i in g {
            let ex = &exs[i];
            ensure!(
                ex.t <= tp,
                "bucket rows must be length-sorted: {} > {tp}",
                ex.t
            );
            ensure!(
                ex.aux.len() == ex.t * aux_width,
                "example {i}: aux length {} != {} x {aux_width}",
                ex.aux.len(),
                ex.t
            );
            h0.extend_from_slice(&ex.h);
            h0.extend(std::iter::repeat_n(0f32, (tp - ex.t) * hidden));
            aux.extend_from_slice(&ex.aux);
            aux.extend(std::iter::repeat_n(0f32, (tp - ex.t) * aux_width));
            let mut ids = ex.tgt.clone();
            ids.resize(tp, 0);
            tgt.extend(ids);
            let unmasked: f32 = ex.mask.iter().sum();
            ensure!(
                unmasked > 0.0,
                "example {i}: every target position is masked"
            );
            let _ = ROW_WEIGHTS_PRESERVE_MEAN_OF_PER_EXAMPLE_MEANS;
            wrow.extend(ex.mask.iter().map(|w| w / unmasked));
            wrow.extend(std::iter::repeat_n(0f32, tp - ex.t));
            positions.extend(0..tp as u32);
        }
        out.push(Bucket {
            h0,
            members: g.clone(),
            rows: g.len(),
            tp,
            tgt,
            wrow,
            positions,
            aux,
        });
    }
    Ok(out)
}

pub(crate) fn rmsnorm_f(x: &[f32], gamma: &[f32], rows: usize, h: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * h..(r + 1) * h];
        let mut sq = 0f32;
        for v in row {
            sq = v.mul_add(*v, sq);
        }
        let rinv = 1.0 / (sq / h as f32 + eps).sqrt();
        for i in 0..h {
            y[r * h + i] = row[i] * rinv * gamma[i];
        }
    }
    y
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn silu_prime(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 + x * (1.0 - s))
}

pub(crate) fn add_assign(a: &mut [f32], b: &[f32]) {
    for (av, bv) in a.iter_mut().zip(b.iter()) {
        *av += *bv;
    }
}

pub(crate) struct StepFactors {
    pub(crate) at: Vec<Vec<f32>>,
    pub(crate) bt: Vec<Vec<f32>>,
}

impl StepFactors {
    pub(crate) fn from(loras: &[LoraVar]) -> Self {
        Self {
            at: loras.iter().map(|l| l.a_t()).collect(),
            bt: loras.iter().map(|l| l.b_t()).collect(),
        }
    }
}

pub(crate) async fn proj_fwd(
    ctx: &WgpuContext,
    w: &W4Mod,
    x: &[f32],
    m: usize,
    lora: Option<(&LoraVar, &[f32], &[f32])>,
    arm: ReadbackArm,
) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    let mut y = w.forward_arm(ctx, x, m, arm).await?;
    let Some((lv, at, bt)) = lora else {
        return Ok((y, None));
    };
    let mut xr = vec![0f32; m * lv.r];
    gemm_nn_f32_arm(ctx, x, at, &mut xr, m, lv.k, lv.r, arm).await?;
    let mut delta = vec![0f32; m * lv.n];
    gemm_nn_f32_arm(ctx, &xr, bt, &mut delta, m, lv.r, lv.n, arm).await?;
    for (yv, dv) in y.iter_mut().zip(delta.iter()) {
        *yv += lv.scaling * *dv;
    }
    Ok((y, Some(xr)))
}

pub(crate) async fn proj_bwd(
    ctx: &WgpuContext,
    w: &W4Mod,
    dy: &[f32],
    x_input: &[f32],
    m: usize,
    lora: Option<(&LoraVar, &[f32], usize)>,
    acc: &mut [Vec<f32>],
    arm: ReadbackArm,
) -> Result<Vec<f32>> {
    let mut dx = vec![0f32; m * w.k];
    dx_w4_f32_arm(
        ctx, dy, &w.words, &w.scales, &w.biases, &mut dx, m, w.n, w.k, arm,
    )
    .await?;
    if let Some((lv, xr, var_base)) = lora {
        let g = lora_grads_f32_arm(
            ctx, dy, x_input, xr, &lv.a, &lv.b, lv.scaling, m, w.n, w.k, lv.r, arm,
        )
        .await?;
        add_assign(&mut dx, &g.dx);
        add_assign(&mut acc[var_base], &g.da);
        add_assign(&mut acc[var_base + 1], &g.db);
    }
    Ok(dx)
}

pub(crate) async fn rope_run(
    ctx: &WgpuContext,
    x: &[f32],
    tokens: usize,
    heads: usize,
    rt: &RopeTables,
    positions: &[u32],
    backward: bool,
    arm: ReadbackArm,
) -> Result<Vec<f32>> {
    let mut out = vec![0f32; x.len()];
    if backward {
        rope_bwd_f32_arm(
            ctx, x, positions, &rt.cos, &rt.sin, &mut out, tokens, heads, rt.half, rt.max_seq, arm,
        )
        .await?;
    } else {
        rope_fwd_f32_arm(
            ctx, x, positions, &rt.cos, &rt.sin, &mut out, tokens, heads, rt.half, rt.max_seq, arm,
        )
        .await?;
    }
    Ok(out)
}

struct LayerStash {
    x_in: Vec<f32>,
    normed: Vec<f32>,
    attn: GroupedAttnFwd,
    attn_bshd: Vec<f32>,
    after_mix: Vec<f32>,
    normed_post: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    silu_gate: Vec<f32>,
    hmlp: Vec<f32>,
    xr: [Option<Vec<f32>>; 7],
}

fn attn_dims(d: &TrainDims, b: usize, t: usize) -> GroupedAttnDims {
    GroupedAttnDims {
        groups: b * d.kv_heads,
        factor: d.heads / d.kv_heads,
        sq: t,
        sk: t,
        head_dim: d.head_dim,
        scale: 1.0 / (d.head_dim as f32).sqrt(),
        causal: true,
        window: 0,
    }
}

#[allow(clippy::too_many_arguments)]
async fn layer_forward(
    ctx: &WgpuContext,
    layer: &TrainLayer,
    d: &TrainDims,
    loras: &[LoraVar],
    slots: Option<&[Option<usize>; 7]>,
    facs: &StepFactors,
    x: &[f32],
    b: usize,
    t: usize,
    rt: &RopeTables,
    positions: &[u32],
    keep: bool,
    arm: ReadbackArm,
) -> Result<(Vec<f32>, Option<LayerStash>)> {
    let m = b * t;
    let lv = |leaf: usize| {
        slots
            .and_then(|s| s[leaf])
            .map(|i| (&loras[i], facs.at[i].as_slice(), facs.bt[i].as_slice()))
    };
    let normed = rmsnorm_f(x, &layer.pre_gamma, m, d.hidden, d.eps);
    let (q, xr_q) = proj_fwd(ctx, &layer.q, &normed, m, lv(0), arm).await?;
    let (k, xr_k) = proj_fwd(ctx, &layer.k, &normed, m, lv(1), arm).await?;
    let (v, xr_v) = proj_fwd(ctx, &layer.v, &normed, m, lv(2), arm).await?;
    let q_ro = rope_run(ctx, &q, m, d.heads, rt, positions, false, arm).await?;
    let k_ro = rope_run(ctx, &k, m, d.kv_heads, rt, positions, false, arm).await?;
    let qg = q_bshd_to_grouped(&q_ro, b, t, d.heads, d.kv_heads, d.head_dim);
    let kg = kv_bskd_to_grouped(&k_ro, b, t, d.kv_heads, d.head_dim);
    let vg = kv_bskd_to_grouped(&v, b, t, d.kv_heads, d.head_dim);
    let attn = grouped_attn_fwd_arm(ctx, &qg, &kg, &vg, &attn_dims(d, b, t), arm).await?;
    let attn_bshd = grouped_to_q_bshd(&attn.out, b, t, d.heads, d.kv_heads, d.head_dim);
    let (o, xr_o) = proj_fwd(ctx, &layer.o, &attn_bshd, m, lv(3), arm).await?;
    let after_mix: Vec<f32> = x.iter().zip(o.iter()).map(|(a, b)| a + b).collect();
    let normed_post = rmsnorm_f(&after_mix, &layer.post_gamma, m, d.hidden, d.eps);
    let (gate, xr_g) = proj_fwd(ctx, &layer.gate, &normed_post, m, lv(4), arm).await?;
    let (up, xr_u) = proj_fwd(ctx, &layer.up, &normed_post, m, lv(5), arm).await?;
    let silu_gate: Vec<f32> = gate.iter().map(|g| silu(*g)).collect();
    let hmlp: Vec<f32> = silu_gate
        .iter()
        .zip(up.iter())
        .map(|(s, u)| s * u)
        .collect();
    let (dn, xr_d) = proj_fwd(ctx, &layer.down, &hmlp, m, lv(6), arm).await?;
    let out: Vec<f32> = after_mix
        .iter()
        .zip(dn.iter())
        .map(|(a, b)| a + b)
        .collect();
    if !keep {
        return Ok((out, None));
    }
    Ok((
        out,
        Some(LayerStash {
            x_in: x.to_vec(),
            normed,
            attn,
            attn_bshd,
            after_mix,
            normed_post,
            gate,
            up,
            silu_gate,
            hmlp,
            xr: [xr_q, xr_k, xr_v, xr_o, xr_g, xr_u, xr_d],
        }),
    ))
}

#[allow(clippy::too_many_arguments)]
async fn layer_backward(
    ctx: &WgpuContext,
    layer: &TrainLayer,
    d: &TrainDims,
    loras: &[LoraVar],
    slots: &[Option<usize>; 7],
    st: &LayerStash,
    dout: &[f32],
    b: usize,
    t: usize,
    rt: &RopeTables,
    positions: &[u32],
    acc: &mut [Vec<f32>],
    arm: ReadbackArm,
) -> Result<Vec<f32>> {
    let m = b * t;
    let h = d.hidden;
    let lv = |leaf: usize| {
        slots[leaf].map(|i| {
            (
                &loras[i],
                st.xr[leaf]
                    .as_deref()
                    .expect("forward stashed xr for every attached lora"),
                2 * i,
            )
        })
    };
    let d_hmlp = proj_bwd(ctx, &layer.down, dout, &st.hmlp, m, lv(6), acc, arm).await?;
    let d_up: Vec<f32> = d_hmlp
        .iter()
        .zip(st.silu_gate.iter())
        .map(|(g, s)| g * s)
        .collect();
    let d_gate: Vec<f32> = d_hmlp
        .iter()
        .zip(st.up.iter())
        .zip(st.gate.iter())
        .map(|((g, u), gx)| g * u * silu_prime(*gx))
        .collect();
    let mut d_np = proj_bwd(
        ctx,
        &layer.gate,
        &d_gate,
        &st.normed_post,
        m,
        lv(4),
        acc,
        arm,
    )
    .await?;
    let d_np_up = proj_bwd(ctx, &layer.up, &d_up, &st.normed_post, m, lv(5), acc, arm).await?;
    add_assign(&mut d_np, &d_np_up);
    let mut d_after = dout.to_vec();
    let mut d_norm_in = vec![0f32; m * h];
    rmsnorm_bwd_f32_arm(
        ctx,
        &st.after_mix,
        &layer.post_gamma,
        &d_np,
        &mut d_norm_in,
        m,
        h,
        d.eps,
        arm,
    )
    .await?;
    add_assign(&mut d_after, &d_norm_in);
    let d_attn_bshd = proj_bwd(ctx, &layer.o, &d_after, &st.attn_bshd, m, lv(3), acc, arm).await?;
    let d_attn_g = q_bshd_to_grouped(&d_attn_bshd, b, t, d.heads, d.kv_heads, d.head_dim);
    let g = grouped_attn_bwd_arm(ctx, &st.attn, &d_attn_g, arm).await?;
    let dq_ro = grouped_to_q_bshd(&g.dq, b, t, d.heads, d.kv_heads, d.head_dim);
    let dk_ro = grouped_to_kv_bskd(&g.dk, b, t, d.kv_heads, d.head_dim);
    let dv = grouped_to_kv_bskd(&g.dv, b, t, d.kv_heads, d.head_dim);
    let dq = rope_run(ctx, &dq_ro, m, d.heads, rt, positions, true, arm).await?;
    let dk = rope_run(ctx, &dk_ro, m, d.kv_heads, rt, positions, true, arm).await?;
    let mut d_normed = proj_bwd(ctx, &layer.q, &dq, &st.normed, m, lv(0), acc, arm).await?;
    let d_normed_k = proj_bwd(ctx, &layer.k, &dk, &st.normed, m, lv(1), acc, arm).await?;
    add_assign(&mut d_normed, &d_normed_k);
    let d_normed_v = proj_bwd(ctx, &layer.v, &dv, &st.normed, m, lv(2), acc, arm).await?;
    add_assign(&mut d_normed, &d_normed_v);
    let mut d_x = vec![0f32; m * h];
    rmsnorm_bwd_f32_arm(
        ctx,
        &st.x_in,
        &layer.pre_gamma,
        &d_normed,
        &mut d_x,
        m,
        h,
        d.eps,
        arm,
    )
    .await?;
    add_assign(&mut d_x, &d_after);
    Ok(d_x)
}

pub const HEAD_ROWS_PER_CHUNK_64_KEEPS_THE_VOCAB_WIDE_DISPATCH_UNDER_65535_WORKGROUPS: usize = 64;

#[allow(clippy::too_many_arguments)]
async fn head_ce_rows(
    ctx: &WgpuContext,
    head: &W4Head,
    x: &[f32],
    tgt: &[u32],
    wrow: &[f32],
    gup: &[f32],
    k: usize,
    arm: ReadbackArm,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let m = tgt.len();
    let chunk = HEAD_ROWS_PER_CHUNK_64_KEEPS_THE_VOCAB_WIDE_DISPATCH_UNDER_65535_WORKGROUPS;
    let mut loss_rows = Vec::with_capacity(m);
    let mut dx = Vec::with_capacity(m * k);
    for c0 in (0..m).step_by(chunk) {
        let c1 = (c0 + chunk).min(m);
        let fwd = head_ce_fwd_arm(
            ctx,
            head,
            &x[c0 * k..c1 * k],
            &tgt[c0..c1],
            &wrow[c0..c1],
            c1 - c0,
            arm,
        )
        .await?;
        loss_rows.extend_from_slice(&fwd.loss_rows);
        dx.extend(
            head_ce_bwd_arm(
                ctx,
                head,
                &fwd,
                &tgt[c0..c1],
                &wrow[c0..c1],
                &gup[c0..c1],
                arm,
            )
            .await?,
        );
    }
    Ok((loss_rows, dx))
}

fn step_seconds_start() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        0.0
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
}

pub fn run_qlora(
    ctx: &WgpuContext,
    loader: &WeightLoader,
    config_json: &str,
    samples: &[TrainSample],
    opts: &TrainOptions,
) -> Result<TrainReport> {
    drive_blocking(run_qlora_arm(
        ctx,
        loader,
        config_json,
        samples,
        opts,
        ReadbackArm::Blocking,
        &mut |_, _| {},
    ))
}

pub async fn run_qlora_pumped(
    ctx: &WgpuContext,
    loader: &WeightLoader,
    config_json: &str,
    samples: &[TrainSample],
    opts: &TrainOptions,
    on_step: &mut dyn FnMut(usize, f32),
) -> Result<TrainReport> {
    run_qlora_arm(
        ctx,
        loader,
        config_json,
        samples,
        opts,
        ReadbackArm::Pumped,
        on_step,
    )
    .await
}

async fn run_qlora_arm(
    ctx: &WgpuContext,
    loader: &WeightLoader,
    config_json: &str,
    samples: &[TrainSample],
    opts: &TrainOptions,
    arm: ReadbackArm,
    on_step: &mut dyn FnMut(usize, f32),
) -> Result<TrainReport> {
    let d = TrainDims::from_config_json(config_json)?;
    ensure!(
        opts.grad_layers >= 1 && opts.grad_layers <= d.layers,
        "grad_layers {} outside 1..={}",
        opts.grad_layers,
        d.layers
    );
    ensure!(!samples.is_empty(), "no training samples");
    refuse_rows_beyond_the_native_window_bound(samples)?;
    let mut exs: Vec<Ex> = samples
        .iter()
        .map(|s| embed_spliced_shifted(loader, s, d.hidden))
        .collect::<Result<_>>()?;
    let max_t = exs.iter().map(|e| e.t).max().unwrap_or(2).max(2);
    let rt = rope_tables(d.head_dim, d.rope_theta, max_t);
    let window_start = d.layers - opts.grad_layers;
    let empty_facs = StepFactors {
        at: Vec::new(),
        bt: Vec::new(),
    };
    let _ = PREFIX_LAYERS_STREAM_THROUGH_ONCE_AND_DROP_SO_ONLY_THE_WINDOW_STAYS_RESIDENT;
    for li in 0..window_start {
        let layer = TrainLayer::load(ctx, loader, &d, li)
            .with_context(|| format!("load frozen prefix layer {li}"))?;
        for ex in exs.iter_mut() {
            let positions: Vec<u32> = (0..ex.t as u32).collect();
            let (out, _) = layer_forward(
                ctx,
                &layer,
                &d,
                &[],
                None,
                &empty_facs,
                &ex.h,
                1,
                ex.t,
                &rt,
                &positions,
                false,
                arm,
            )
            .await?;
            ex.h = out;
        }
    }
    let window: Vec<TrainLayer> = (window_start..d.layers)
        .map(|li| {
            TrainLayer::load(ctx, loader, &d, li)
                .with_context(|| format!("load trainable window layer {li}"))
        })
        .collect::<Result<_>>()?;
    let final_gamma = gamma_f32(loader, FINAL_NORM, d.hidden)?;
    let head = W4Mod::load(ctx, loader, LM_HEAD_MODULE, d.vocab, d.hidden, false)?;

    let mut loras: Vec<LoraVar> = Vec::new();
    let mut slots: Vec<[Option<usize>; 7]> = Vec::new();
    let mut seed = opts.seed;
    for layer in &window {
        let mut s = [None; 7];
        for (leaf_i, (leaf, w)) in layer.projections().iter().enumerate() {
            if !opts.targets.iter().any(|t| t == leaf) {
                continue;
            }
            s[leaf_i] = Some(loras.len());
            loras.push(LoraVar::init(
                layer.module_name(leaf),
                w.n,
                w.k,
                opts.rank,
                opts.alpha,
                seed,
            ));
            seed = seed.wrapping_add(1);
        }
        slots.push(s);
    }
    ensure!(
        !loras.is_empty(),
        "no LoRA target matched: targets {:?}",
        opts.targets
    );

    let buckets = pack_buckets(&exs, d.hidden)?;
    drop(exs);
    let sizes: Vec<usize> = loras.iter().flat_map(|l| [l.r * l.k, l.n * l.r]).collect();
    let mut adam = AdamW::new(&sizes, opts.lr, opts.b_lr_ratio);
    let inv = 1.0f64 / samples.len() as f64;
    let mut losses: Vec<f32> = Vec::with_capacity(opts.steps);
    let mut all_grads_finite = true;
    for step in 0..opts.steps {
        let t0 = step_seconds_start();
        let facs = StepFactors::from(&loras);
        let mut acc: Vec<Vec<f32>> = sizes.iter().map(|&n| vec![0f32; n]).collect();
        let mut loss_sum = 0f32;
        for bkt in &buckets {
            let m = bkt.rows * bkt.tp;
            let mut x = bkt.h0.clone();
            let mut stashes: Vec<LayerStash> = Vec::with_capacity(window.len());
            for (wi, layer) in window.iter().enumerate() {
                let (out, st) = layer_forward(
                    ctx,
                    layer,
                    &d,
                    &loras,
                    Some(&slots[wi]),
                    &facs,
                    &x,
                    bkt.rows,
                    bkt.tp,
                    &rt,
                    &bkt.positions,
                    true,
                    arm,
                )
                .await?;
                stashes.push(st.expect("keep=true stashes every window layer"));
                x = out;
            }
            let normed_final = rmsnorm_f(&x, &final_gamma, m, d.hidden, d.eps);
            let gup = vec![1f32; m];
            let (loss_rows, ce_dx) = head_ce_rows(
                ctx,
                &head.gpu,
                &normed_final,
                &bkt.tgt,
                &bkt.wrow,
                &gup,
                d.hidden,
                arm,
            )
            .await?;
            let bucket_loss: f32 = loss_rows.iter().sum();
            if !bucket_loss.is_finite() {
                all_grads_finite = false;
                eprintln!("[ocr-web-train] step {step}: FORWARD loss non-finite");
            }
            loss_sum += bucket_loss;
            let mut dout = vec![0f32; m * d.hidden];
            rmsnorm_bwd_f32_arm(
                ctx,
                &x,
                &final_gamma,
                &ce_dx,
                &mut dout,
                m,
                d.hidden,
                d.eps,
                arm,
            )
            .await?;
            for wi in (0..window.len()).rev() {
                let st = stashes.pop().expect("one stash per window layer");
                dout = layer_backward(
                    ctx,
                    &window[wi],
                    &d,
                    &loras,
                    &slots[wi],
                    &st,
                    &dout,
                    bkt.rows,
                    bkt.tp,
                    &rt,
                    &bkt.positions,
                    acc.as_mut_slice(),
                    arm,
                )
                .await?;
            }
        }
        for g in acc.iter_mut() {
            for v in g.iter_mut() {
                *v *= inv as f32;
                if !v.is_finite() {
                    all_grads_finite = false;
                }
            }
        }
        adam.step(&mut loras, &acc);
        let lv = loss_sum * inv as f32;
        losses.push(lv);
        on_step(step, lv);
        eprintln!(
            "[ocr-web-train] step {step:3}  loss {lv:.6}  {:.2}s",
            step_seconds_start() - t0
        );
    }

    let modules: Vec<String> = loras.iter().map(|l| l.module.clone()).collect();
    let adapter_safetensors = peft::adapter_safetensors_bytes(&loras)?;
    let adapter_config_json =
        peft::adapter_config_json_bytes(&loras, opts.rank, opts.alpha, &opts.base_model_name);
    Ok(TrainReport {
        losses,
        all_grads_finite,
        window_start,
        modules,
        adapter_safetensors,
        adapter_config_json,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_of_len(ids_len: usize) -> TrainSample {
        TrainSample {
            ids: vec![1; ids_len],
            completion_start: 2,
            vision_start: 0,
            n_vis: 1,
            rows: Vec::new(),
        }
    }

    #[test]
    fn rows_at_the_native_oom_bound_pass_and_one_past_it_refuses() {
        let bound =
            EXAMPLE_TOKEN_CEILING_WHERE_THE_NATIVE_LIVE_GRAPH_OOMS_INTO_THE_CHECKPOINT_FALLBACK;
        refuse_rows_beyond_the_native_window_bound(&[sample_of_len(bound + 1)])
            .expect("a row exactly at the bound trains");
        let err = refuse_rows_beyond_the_native_window_bound(&[
            sample_of_len(3),
            sample_of_len(bound + 2),
        ])
        .expect_err("a row past the bound must refuse before touching the GPU");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("sample 1"),
            "names the offending sample: {msg}"
        );
        assert!(
            msg.contains("NV_DOTS_QLORA_CHECKPOINT=1"),
            "names the native escape: {msg}"
        );
    }
}
