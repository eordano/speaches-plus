use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch::{drive_blocking, ReadbackArm};
use nv_kernels::wgpu_backend::kernels::qlora_bwd_norm::rmsnorm_bwd_f32_arm;
use nv_kernels::wgpu_qlora::qlora_attn::{
    grouped_attn_bwd_arm, grouped_attn_fwd_arm, grouped_to_kv_bskd, grouped_to_q_bshd,
    kv_bskd_to_grouped, q_bshd_to_grouped, GroupedAttnDims, GroupedAttnFwd,
};
use nv_kernels::wgpu_qlora::qlora_head_ce::{
    head_ce_bwd_arm, head_ce_fwd_arm, W4Head, HEAD_CE_CHUNK_ROWS, WORKGROUP_SIZE,
};
use nv_models::gemma4::LayerType;
use nv_weights::WeightLoader;

use super::weights::{per_layer_raw_rows, scaled_embed_rows, E4bDims, E4bLayer, E4bTower};
use crate::train::lora::{AdamW, LoraVar};
use crate::train::peft;
use crate::train::schedule::{
    add_assign, pack_buckets_aux, proj_bwd, proj_fwd, rmsnorm_f, rope_run, rope_tables_partial,
    Bucket, Ex, RopeTables, StepFactors, TrainOptions, TrainReport,
};

pub const E4B_BASE_MODEL_NAME: &str = "google/gemma-4-E4B-it";

pub const E4B_SCHEDULE_REPLAYS_THE_SERVING_LAYER_ALGEBRA_WITH_THREE_RESIDUAL_BRANCHES: &str =
    "t = rms(x)*ln_in; q,k,v = W t; per-head rms with q_norm/k_norm/ones; neox rope with the \
     layer type's theta and partial angle count; grouped attention at scale 1.0, causal, \
     windowed on sliding layers; h1 = x + rms(W_o a)*ln_pa; h2 = h1 + rms(W_down \
     (gelu_tanh(W_gate rms(h1)*ln_pf) * W_up ...))*ln_po; g = gelu_tanh(W_plgate h2) * pli; \
     x' = (h2 + rms(W_plproj g)*ln_pli) * layer_scalar -- exactly the dispatch order \
     gemma4_e4b_wgpu::emit_layer issues at m>1, in f32 instead of bf16";

pub const E4B_WINDOW_LAYERS_MUST_NOT_OWN_A_KV_SOURCE_BECAUSE_DK_DV_OF_A_SHARED_LAYER_ARE_DROPPED:
    &str =
    "layers past first_kv_shared reuse the rope'd k and normed v of the last earlier layer of \
     their type; the backward of such a layer emits dK/dV that belong to that source layer, and \
     the schedule drops them, which is exact only while the source is frozen -- so every kv \
     source of a window layer must lie below the window";

pub const E4B_ATTENTION_IS_UNSCALED_LIKE_THE_SERVING_GRAPH: f32 = 1.0;

pub const GELU_SQRT2_OVER_PI: f32 = 0.797_884_6;
pub const GELU_CUBIC: f32 = 0.044_715;
pub const GELU_TANH_ARG_CLAMP_MIRRORS_EVERY_SERVING_GELU_FOLD: f32 = 10.0;

pub struct E4bTrainSample {
    pub ids: Vec<u32>,
    pub completion_start: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AuxLayout {
    pub(super) base: usize,
    pub(super) width: usize,
}

impl AuxLayout {
    pub(super) fn full(d: &E4bDims) -> Self {
        Self {
            base: 0,
            width: d.ple_row,
        }
    }

    pub(super) fn window(d: &E4bDims, window_start: usize) -> Self {
        Self {
            base: window_start,
            width: (d.layers - window_start) * d.hpl,
        }
    }
}

pub(super) type KvPair = (Vec<f32>, Vec<f32>);

pub(super) struct WindowRun<'a> {
    pub(super) d: &'a E4bDims,
    pub(super) window: &'a [E4bLayer],
    pub(super) final_norm: &'a [f32],
    pub(super) head: &'a W4Head,
    pub(super) buckets: &'a [Bucket],
    pub(super) hs: &'a [Vec<f32>],
    pub(super) kv_stash: &'a BTreeMap<usize, Vec<KvPair>>,
    pub(super) lay: AuxLayout,
    pub(super) window_start: usize,
    pub(super) n_samples: usize,
}

pub fn e4b_train_options() -> TrainOptions {
    TrainOptions {
        base_model_name: E4B_BASE_MODEL_NAME.to_string(),
        ..TrainOptions::default()
    }
}

fn gelu_tanh(x: f32) -> f32 {
    let u = GELU_SQRT2_OVER_PI * (x + GELU_CUBIC * x * x * x);
    let c = GELU_TANH_ARG_CLAMP_MIRRORS_EVERY_SERVING_GELU_FOLD;
    let t = u.clamp(-c, c).tanh();
    0.5 * x * (1.0 + t)
}

fn gelu_tanh_prime(x: f32) -> f32 {
    let u = GELU_SQRT2_OVER_PI * (x + GELU_CUBIC * x * x * x);
    let c = GELU_TANH_ARG_CLAMP_MIRRORS_EVERY_SERVING_GELU_FOLD;
    let inside = u.abs() < c;
    let t = u.clamp(-c, c).tanh();
    let du = if inside {
        GELU_SQRT2_OVER_PI * (1.0 + 3.0 * GELU_CUBIC * x * x)
    } else {
        0.0
    };
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du
}

fn scaled(a: &[f32], s: f32) -> Vec<f32> {
    a.iter().map(|v| v * s).collect()
}

fn summed(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

async fn norm_bwd(
    ctx: &WgpuContext,
    x: &[f32],
    gamma: &[f32],
    dy: &[f32],
    rows: usize,
    h: usize,
    eps: f32,
    arm: ReadbackArm,
) -> Result<Vec<f32>> {
    let mut dx = vec![0f32; rows * h];
    rmsnorm_bwd_f32_arm(ctx, x, gamma, dy, &mut dx, rows, h, eps, arm).await?;
    Ok(dx)
}

pub(super) async fn build_examples(
    ctx: &WgpuContext,
    loader: &WeightLoader,
    d: &E4bDims,
    tower: &E4bTower,
    samples: &[E4bTrainSample],
    arm: ReadbackArm,
) -> Result<Vec<Ex>> {
    let _ = super::weights::E4B_PER_LAYER_INPUT_ROWS_ARE_TOKEN_CONSTANTS_BECAUSE_EVERY_TENSOR_FEEDING_THEM_IS_FROZEN;
    let mut out = Vec::with_capacity(samples.len());
    let sa = std::f32::consts::FRAC_1_SQRT_2;
    let sb = (d.hpl as f32).sqrt() * std::f32::consts::FRAC_1_SQRT_2;
    for (i, s) in samples.iter().enumerate() {
        ensure!(
            s.ids.len() >= 2 && s.completion_start >= 1 && s.completion_start < s.ids.len(),
            "sample {i}: len {} completion_start {} is not a prompt followed by a completion",
            s.ids.len(),
            s.completion_start
        );
        let t = s.ids.len() - 1;
        let h = scaled_embed_rows(loader, d, tower, &s.ids[..t])?;
        let raw = per_layer_raw_rows(loader, d, &s.ids[..t])?;
        let proj = tower.plmp.forward_arm(ctx, &h, t, arm).await?;
        let normed = rmsnorm_f(&proj, &tower.plp_norm, t * d.layers, d.hpl, d.eps);
        let aux: Vec<f32> = normed
            .iter()
            .zip(raw.iter())
            .map(|(n, r)| n * sa + r * sb)
            .collect();
        let tgt = s.ids[1..].to_vec();
        let mask: Vec<f32> = (1..s.ids.len())
            .map(|p| if p >= s.completion_start { 1.0 } else { 0.0 })
            .collect();
        out.push(Ex {
            h,
            t,
            tgt,
            mask,
            aux,
        });
    }
    Ok(out)
}

fn attn_dims(d: &E4bDims, li: usize, b: usize, t: usize) -> GroupedAttnDims {
    let kvh = d.kv_heads(li);
    GroupedAttnDims {
        groups: b * kvh,
        factor: d.heads / kvh,
        sq: t,
        sk: t,
        head_dim: d.head_dim(li),
        scale: E4B_ATTENTION_IS_UNSCALED_LIKE_THE_SERVING_GRAPH,
        causal: true,
        window: d.window(li),
    }
}

pub(super) struct Stash {
    x_in: Vec<f32>,
    t_norm: Vec<f32>,
    q: Vec<f32>,
    k: Option<Vec<f32>>,
    v: Option<Vec<f32>>,
    attn: GroupedAttnFwd,
    attn_bshd: Vec<f32>,
    o: Vec<f32>,
    h1: Vec<f32>,
    mlp_in: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    dn: Vec<f32>,
    h2: Vec<f32>,
    plg: Vec<f32>,
    g: Vec<f32>,
    t1: Vec<f32>,
    xr: [Option<Vec<f32>>; 7],
}

pub(super) struct Fwd {
    pub(super) out: Vec<f32>,
    pub(super) stash: Option<Stash>,
    pub(super) kv: Option<KvPair>,
}

fn pli_slice_mul(
    d: &E4bDims,
    li: usize,
    lay: AuxLayout,
    aux: &[f32],
    m: usize,
    x: &[f32],
) -> Vec<f32> {
    let mut out = vec![0f32; m * d.hpl];
    for r in 0..m {
        let base = r * lay.width + (li - lay.base) * d.hpl;
        for j in 0..d.hpl {
            out[r * d.hpl + j] = x[r * d.hpl + j] * aux[base + j];
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn layer_forward(
    ctx: &WgpuContext,
    layer: &E4bLayer,
    d: &E4bDims,
    rt: &RopeTables,
    loras: &[LoraVar],
    slots: Option<&[Option<usize>; 7]>,
    facs: &StepFactors,
    x: &[f32],
    b: usize,
    t: usize,
    positions: &[u32],
    aux: &[f32],
    lay: AuxLayout,
    shared_kv: Option<&KvPair>,
    keep: bool,
    arm: ReadbackArm,
) -> Result<Fwd> {
    let _ = E4B_SCHEDULE_REPLAYS_THE_SERVING_LAYER_ALGEBRA_WITH_THREE_RESIDUAL_BRANCHES;
    let li = layer.index;
    let m = b * t;
    let h = d.hidden;
    let hd = d.head_dim(li);
    let kvh = d.kv_heads(li);
    let lv = |leaf: usize| {
        slots
            .and_then(|s| s[leaf])
            .map(|i| (&loras[i], facs.at[i].as_slice(), facs.bt[i].as_slice()))
    };
    let t_norm = rmsnorm_f(x, &layer.ln_in, m, h, d.eps);
    let (q, xr_q) = proj_fwd(ctx, &layer.q, &t_norm, m, lv(0), arm).await?;
    let qn = rmsnorm_f(&q, &layer.q_norm, m * d.heads, hd, d.eps);
    let q_ro = rope_run(ctx, &qn, m, d.heads, rt, positions, false, arm).await?;
    let qg = q_bshd_to_grouped(&q_ro, b, t, d.heads, kvh, hd);
    let (kg, vg, k, v, xr_k, xr_v) = match (shared_kv, &layer.k, &layer.v) {
        (Some((kg, vg)), _, _) => (kg.clone(), vg.clone(), None, None, None, None),
        (None, Some(wk), Some(wv)) => {
            let (k, xr_k) = proj_fwd(ctx, wk, &t_norm, m, lv(1), arm).await?;
            let kn = rmsnorm_f(&k, &layer.k_norm, m * kvh, hd, d.eps);
            let k_ro = rope_run(ctx, &kn, m, kvh, rt, positions, false, arm).await?;
            let (v, xr_v) = proj_fwd(ctx, wv, &t_norm, m, lv(2), arm).await?;
            let vn = rmsnorm_f(&v, &layer.v_norm, m * kvh, hd, d.eps);
            (
                kv_bskd_to_grouped(&k_ro, b, t, kvh, hd),
                kv_bskd_to_grouped(&vn, b, t, kvh, hd),
                Some(k),
                Some(v),
                xr_k,
                xr_v,
            )
        }
        _ => anyhow::bail!(
            "layer {li}: kv-shared layer reached without its source stash (source {:?})",
            layer.kv_source
        ),
    };
    let attn = grouped_attn_fwd_arm(ctx, &qg, &kg, &vg, &attn_dims(d, li, b, t), arm).await?;
    let attn_bshd = grouped_to_q_bshd(&attn.out, b, t, d.heads, kvh, hd);
    let (o, xr_o) = proj_fwd(ctx, &layer.o, &attn_bshd, m, lv(3), arm).await?;
    let h1 = summed(x, &rmsnorm_f(&o, &layer.ln_pa, m, h, d.eps));
    let mlp_in = rmsnorm_f(&h1, &layer.ln_pf, m, h, d.eps);
    let (gate, xr_g) = proj_fwd(ctx, &layer.gate, &mlp_in, m, lv(4), arm).await?;
    let (up, xr_u) = proj_fwd(ctx, &layer.up, &mlp_in, m, lv(5), arm).await?;
    let act: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| gelu_tanh(*g) * u)
        .collect();
    let (dn, xr_d) = proj_fwd(ctx, &layer.down, &act, m, lv(6), arm).await?;
    let h2 = summed(&h1, &rmsnorm_f(&dn, &layer.ln_po, m, h, d.eps));
    let plg = layer.plgate.forward_arm(ctx, &h2, m, arm).await?;
    let plg_act: Vec<f32> = plg.iter().map(|g| gelu_tanh(*g)).collect();
    let g = pli_slice_mul(d, li, lay, aux, m, &plg_act);
    let t1 = layer.plproj.forward_arm(ctx, &g, m, arm).await?;
    let out = scaled(
        &summed(&h2, &rmsnorm_f(&t1, &layer.ln_pli, m, h, d.eps)),
        layer.layer_scalar,
    );
    let kv = if layer.kv_source.is_none() {
        Some((kg, vg))
    } else {
        None
    };
    if !keep {
        return Ok(Fwd {
            out,
            stash: None,
            kv,
        });
    }
    Ok(Fwd {
        out,
        stash: Some(Stash {
            x_in: x.to_vec(),
            t_norm,
            q,
            k,
            v,
            attn,
            attn_bshd,
            o,
            h1,
            mlp_in,
            gate,
            up,
            act,
            dn,
            h2,
            plg,
            g,
            t1,
            xr: [xr_q, xr_k, xr_v, xr_o, xr_g, xr_u, xr_d],
        }),
        kv,
    })
}

#[allow(clippy::too_many_arguments)]
async fn layer_backward(
    ctx: &WgpuContext,
    layer: &E4bLayer,
    d: &E4bDims,
    rt: &RopeTables,
    loras: &[LoraVar],
    slots: &[Option<usize>; 7],
    st: &Stash,
    dout: &[f32],
    b: usize,
    t: usize,
    positions: &[u32],
    aux: &[f32],
    lay: AuxLayout,
    acc: &mut [Vec<f32>],
    arm: ReadbackArm,
) -> Result<Vec<f32>> {
    let li = layer.index;
    let m = b * t;
    let h = d.hidden;
    let hd = d.head_dim(li);
    let kvh = d.kv_heads(li);
    let eps = d.eps;
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
    let d_h2_in = scaled(dout, layer.layer_scalar);
    let d_t1 = norm_bwd(ctx, &st.t1, &layer.ln_pli, &d_h2_in, m, h, eps, arm).await?;
    let d_g = proj_bwd(ctx, &layer.plproj, &d_t1, &st.g, m, None, acc, arm).await?;
    let d_plg_act = pli_slice_mul(d, li, lay, aux, m, &d_g);
    let d_plg: Vec<f32> = d_plg_act
        .iter()
        .zip(st.plg.iter())
        .map(|(dy, x)| dy * gelu_tanh_prime(*x))
        .collect();
    let mut d_h2 = d_h2_in;
    add_assign(
        &mut d_h2,
        &proj_bwd(ctx, &layer.plgate, &d_plg, &st.h2, m, None, acc, arm).await?,
    );
    let d_dn = norm_bwd(ctx, &st.dn, &layer.ln_po, &d_h2, m, h, eps, arm).await?;
    let d_act = proj_bwd(ctx, &layer.down, &d_dn, &st.act, m, lv(6), acc, arm).await?;
    let d_up: Vec<f32> = d_act
        .iter()
        .zip(st.gate.iter())
        .map(|(da, g)| da * gelu_tanh(*g))
        .collect();
    let d_gate: Vec<f32> = d_act
        .iter()
        .zip(st.up.iter())
        .zip(st.gate.iter())
        .map(|((da, u), g)| da * u * gelu_tanh_prime(*g))
        .collect();
    let mut d_mlp_in = proj_bwd(ctx, &layer.gate, &d_gate, &st.mlp_in, m, lv(4), acc, arm).await?;
    add_assign(
        &mut d_mlp_in,
        &proj_bwd(ctx, &layer.up, &d_up, &st.mlp_in, m, lv(5), acc, arm).await?,
    );
    let mut d_h1 = d_h2;
    add_assign(
        &mut d_h1,
        &norm_bwd(ctx, &st.h1, &layer.ln_pf, &d_mlp_in, m, h, eps, arm).await?,
    );
    let d_o = norm_bwd(ctx, &st.o, &layer.ln_pa, &d_h1, m, h, eps, arm).await?;
    let d_attn_bshd = proj_bwd(ctx, &layer.o, &d_o, &st.attn_bshd, m, lv(3), acc, arm).await?;
    let d_attn_g = q_bshd_to_grouped(&d_attn_bshd, b, t, d.heads, kvh, hd);
    let g = grouped_attn_bwd_arm(ctx, &st.attn, &d_attn_g, arm).await?;
    let dq_ro = grouped_to_q_bshd(&g.dq, b, t, d.heads, kvh, hd);
    let dqn = rope_run(ctx, &dq_ro, m, d.heads, rt, positions, true, arm).await?;
    let dq = norm_bwd(ctx, &st.q, &layer.q_norm, &dqn, m * d.heads, hd, eps, arm).await?;
    let mut d_t = proj_bwd(ctx, &layer.q, &dq, &st.t_norm, m, lv(0), acc, arm).await?;
    if let (Some(k), Some(v), Some(wk), Some(wv)) = (&st.k, &st.v, &layer.k, &layer.v) {
        let dk_ro = grouped_to_kv_bskd(&g.dk, b, t, kvh, hd);
        let dkn = rope_run(ctx, &dk_ro, m, kvh, rt, positions, true, arm).await?;
        let dk = norm_bwd(ctx, k, &layer.k_norm, &dkn, m * kvh, hd, eps, arm).await?;
        add_assign(
            &mut d_t,
            &proj_bwd(ctx, wk, &dk, &st.t_norm, m, lv(1), acc, arm).await?,
        );
        let dvn = grouped_to_kv_bskd(&g.dv, b, t, kvh, hd);
        let dv = norm_bwd(ctx, v, &layer.v_norm, &dvn, m * kvh, hd, eps, arm).await?;
        add_assign(
            &mut d_t,
            &proj_bwd(ctx, wv, &dv, &st.t_norm, m, lv(2), acc, arm).await?,
        );
    } else {
        let _ =
            E4B_WINDOW_LAYERS_MUST_NOT_OWN_A_KV_SOURCE_BECAUSE_DK_DV_OF_A_SHARED_LAYER_ARE_DROPPED;
    }
    let mut d_x = d_h1;
    add_assign(
        &mut d_x,
        &norm_bwd(ctx, &st.x_in, &layer.ln_in, &d_t, m, h, eps, arm).await?,
    );
    Ok(d_x)
}

fn head_chunk_rows(ctx: &WgpuContext, n: usize) -> usize {
    let per_dim = ctx.caps.max_compute_workgroups_per_dimension as usize;
    (per_dim * WORKGROUP_SIZE as usize / n).clamp(1, HEAD_CE_CHUNK_ROWS)
}

#[allow(clippy::too_many_arguments)]
async fn head_ce_rows(
    ctx: &WgpuContext,
    head: &W4Head,
    x: &[f32],
    tgt: &[u32],
    wrow: &[f32],
    k: usize,
    arm: ReadbackArm,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let m = tgt.len();
    let chunk = head_chunk_rows(ctx, head.n);
    let gup = vec![1f32; m];
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

fn now_seconds() -> f64 {
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

pub fn run_e4b_qlora(
    ctx: &WgpuContext,
    loader: &WeightLoader,
    config_json: &str,
    samples: &[E4bTrainSample],
    opts: &TrainOptions,
) -> Result<TrainReport> {
    drive_blocking(run_e4b_qlora_arm(
        ctx,
        loader,
        config_json,
        samples,
        opts,
        ReadbackArm::Blocking,
        &mut |_, _| {},
    ))
}

pub async fn run_e4b_qlora_pumped(
    ctx: &WgpuContext,
    loader: &WeightLoader,
    config_json: &str,
    samples: &[E4bTrainSample],
    opts: &TrainOptions,
    on_step: &mut dyn FnMut(usize, f32),
) -> Result<TrainReport> {
    run_e4b_qlora_arm(
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

async fn run_e4b_qlora_arm(
    ctx: &WgpuContext,
    loader: &WeightLoader,
    config_json: &str,
    samples: &[E4bTrainSample],
    opts: &TrainOptions,
    arm: ReadbackArm,
    on_step: &mut dyn FnMut(usize, f32),
) -> Result<TrainReport> {
    let d = E4bDims::from_config_json(config_json)?;
    let window_start = window_start_checked(&d, opts)?;
    ensure!(!samples.is_empty(), "no training samples");
    let sources = kv_sources(&d, 0..d.layers);
    let tower = E4bTower::load(ctx, loader, &d).context("load E4B tower")?;
    let exs = build_examples(ctx, loader, &d, &tower, samples, arm).await?;
    let buckets: Vec<Bucket> = pack_buckets_aux(&exs, d.hidden, d.ple_row)?;
    drop(exs);
    let mut hs: Vec<Vec<f32>> = buckets.iter().map(|b| b.h0.clone()).collect();
    let rts = rope_tables_for(&d, max_tp(&buckets));
    let empty_facs = StepFactors {
        at: Vec::new(),
        bt: Vec::new(),
    };
    let lay = AuxLayout::full(&d);
    let mut kv_stash: BTreeMap<usize, Vec<KvPair>> = BTreeMap::new();
    for li in 0..window_start {
        let layer = E4bLayer::load(ctx, loader, &d, li)
            .with_context(|| format!("load frozen prefix layer {li}"))?;
        let mut kvs = Vec::with_capacity(buckets.len());
        for (bi, bkt) in buckets.iter().enumerate() {
            let shared = layer.kv_source.map(|s| &kv_stash[&s][bi]);
            let f = layer_forward(
                ctx,
                &layer,
                &d,
                rt_of(&rts, layer.kind),
                &[],
                None,
                &empty_facs,
                &hs[bi],
                bkt.rows,
                bkt.tp,
                &bkt.positions,
                &bkt.aux,
                lay,
                shared,
                false,
                arm,
            )
            .await?;
            hs[bi] = f.out;
            if sources.contains(&li) {
                kvs.push(f.kv.expect("a kv source layer computes its own k and v"));
            }
        }
        if sources.contains(&li) {
            kv_stash.insert(li, kvs);
        }
    }
    let window: Vec<E4bLayer> = (window_start..d.layers)
        .map(|li| {
            E4bLayer::load(ctx, loader, &d, li)
                .with_context(|| format!("load trainable window layer {li}"))
        })
        .collect::<Result<_>>()?;
    train_window(
        ctx,
        WindowRun {
            d: &d,
            window: &window,
            final_norm: &tower.final_norm,
            head: &tower.head,
            buckets: &buckets,
            hs: &hs,
            kv_stash: &kv_stash,
            lay,
            window_start,
            n_samples: samples.len(),
        },
        opts,
        arm,
        on_step,
    )
    .await
}

pub(super) fn window_start_checked(d: &E4bDims, opts: &TrainOptions) -> Result<usize> {
    ensure!(
        opts.grad_layers >= 1 && opts.grad_layers <= d.layers,
        "grad_layers {} outside 1..={}",
        opts.grad_layers,
        d.layers
    );
    let window_start = d.layers - opts.grad_layers;
    for li in window_start..d.layers {
        if let Some(src) = d.kv_source(li) {
            ensure!(
                src < window_start,
                "layer {li} in the trainable window shares kv with layer {src} inside it: {}",
                E4B_WINDOW_LAYERS_MUST_NOT_OWN_A_KV_SOURCE_BECAUSE_DK_DV_OF_A_SHARED_LAYER_ARE_DROPPED
            );
        }
    }
    Ok(window_start)
}

pub(super) fn kv_sources(d: &E4bDims, layers: std::ops::Range<usize>) -> Vec<usize> {
    let mut s: Vec<usize> = layers.filter_map(|li| d.kv_source(li)).collect();
    s.sort_unstable();
    s.dedup();
    s
}

pub(super) fn max_tp(buckets: &[Bucket]) -> usize {
    buckets.iter().map(|b| b.tp).max().unwrap_or(2).max(2)
}

pub(super) fn rope_tables_for(d: &E4bDims, max_t: usize) -> [RopeTables; 2] {
    [LayerType::SlidingAttention, LayerType::FullAttention].map(|kind| {
        rope_tables_partial(
            d.head_dim_of(kind),
            d.theta(kind),
            max_t,
            d.rope_angles(kind),
        )
    })
}

pub(super) fn rt_of(rts: &[RopeTables; 2], kind: LayerType) -> &RopeTables {
    match kind {
        LayerType::SlidingAttention => &rts[0],
        LayerType::FullAttention => &rts[1],
    }
}

pub(super) async fn train_window(
    ctx: &WgpuContext,
    run: WindowRun<'_>,
    opts: &TrainOptions,
    arm: ReadbackArm,
    on_step: &mut dyn FnMut(usize, f32),
) -> Result<TrainReport> {
    let WindowRun {
        d,
        window,
        final_norm,
        head,
        buckets,
        hs,
        kv_stash,
        lay,
        window_start,
        n_samples,
    } = run;
    ensure!(n_samples > 0, "no training samples");
    let rts = rope_tables_for(d, max_tp(buckets));
    let mut loras: Vec<LoraVar> = Vec::new();
    let mut slots: Vec<[Option<usize>; 7]> = Vec::new();
    let mut seed = opts.seed;
    for layer in window {
        let mut s = [None; 7];
        for (leaf_i, proj) in layer.projections().iter().enumerate() {
            let Some((leaf, w)) = proj else { continue };
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
    let sizes: Vec<usize> = loras.iter().flat_map(|l| [l.r * l.k, l.n * l.r]).collect();
    let mut adam = AdamW::new(&sizes, opts.lr, opts.b_lr_ratio);
    let inv = 1.0f64 / n_samples as f64;
    let mut losses: Vec<f32> = Vec::with_capacity(opts.steps);
    let mut all_grads_finite = true;
    for step in 0..opts.steps {
        let t0 = now_seconds();
        let facs = StepFactors::from(&loras);
        let mut acc: Vec<Vec<f32>> = sizes.iter().map(|&n| vec![0f32; n]).collect();
        let mut loss_sum = 0f32;
        for (bi, bkt) in buckets.iter().enumerate() {
            let m = bkt.rows * bkt.tp;
            let mut x = hs[bi].clone();
            let mut stashes: Vec<Stash> = Vec::with_capacity(window.len());
            for (wi, layer) in window.iter().enumerate() {
                let shared = layer.kv_source.map(|s| &kv_stash[&s][bi]);
                let f = layer_forward(
                    ctx,
                    layer,
                    d,
                    rt_of(&rts, layer.kind),
                    &loras,
                    Some(&slots[wi]),
                    &facs,
                    &x,
                    bkt.rows,
                    bkt.tp,
                    &bkt.positions,
                    &bkt.aux,
                    lay,
                    shared,
                    true,
                    arm,
                )
                .await?;
                stashes.push(f.stash.expect("keep=true stashes every window layer"));
                x = f.out;
            }
            let normed_final = rmsnorm_f(&x, final_norm, m, d.hidden, d.eps);
            let (loss_rows, ce_dx) =
                head_ce_rows(ctx, head, &normed_final, &bkt.tgt, &bkt.wrow, d.hidden, arm).await?;
            let bucket_loss: f32 = loss_rows.iter().sum();
            if !bucket_loss.is_finite() {
                all_grads_finite = false;
                eprintln!("[e4b-web-train] step {step}: FORWARD loss non-finite");
            }
            loss_sum += bucket_loss;
            let mut dout = norm_bwd(ctx, &x, final_norm, &ce_dx, m, d.hidden, d.eps, arm).await?;
            for wi in (0..window.len()).rev() {
                let st = stashes.pop().expect("one stash per window layer");
                dout = layer_backward(
                    ctx,
                    &window[wi],
                    d,
                    rt_of(&rts, window[wi].kind),
                    &loras,
                    &slots[wi],
                    &st,
                    &dout,
                    bkt.rows,
                    bkt.tp,
                    &bkt.positions,
                    &bkt.aux,
                    lay,
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
            "[e4b-web-train] step {step:3}  loss {lv:.6}  {:.2}s",
            now_seconds() - t0
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

    #[test]
    fn gelu_tanh_prime_matches_a_central_difference_inside_and_outside_the_clamp() {
        for &x in &[-4.0f32, -1.0, -0.3, 0.0, 0.7, 2.5, 4.0] {
            let h = 1e-3f32;
            let fd = (gelu_tanh(x + h) - gelu_tanh(x - h)) / (2.0 * h);
            let an = gelu_tanh_prime(x);
            assert!(
                (fd - an).abs() < 2e-3,
                "x {x}: finite difference {fd} vs analytic {an}"
            );
        }
        assert_eq!(
            gelu_tanh_prime(30.0),
            1.0,
            "past the clamp the tanh is flat and the slope is the linear term"
        );
    }
}
