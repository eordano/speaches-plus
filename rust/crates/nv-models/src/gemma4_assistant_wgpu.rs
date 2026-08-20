use anyhow::{Context, Result};

use nv_kernels::wgpu_backend::dispatch::{self, GpuBind, GpuTensor, GpuUniform, Recorded};
use nv_kernels::wgpu_backend::kernels::assistant_drafter as ad;
use nv_kernels::wgpu_backend::{compose, WgpuContext};

use crate::gemma4_e4b_wgpu::pack_pairs;

const LABEL: &str = "nv_models_gemma4_assistant_wgpu";

#[derive(Clone, Debug)]
pub struct AssistantWgpuSpec {
    pub backbone_hidden: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub n_heads: usize,
    pub vocab: usize,
    pub n_centroids: usize,
    pub top_k: usize,
    pub eps: f32,
    pub sliding_window: usize,
    pub sliding_theta: f32,
    pub full_theta: f32,
    pub full_partial: f64,
    pub sliding_hd: usize,
    pub full_hd: usize,
    pub sliding_nkv: usize,
    pub full_nkv: usize,
    pub layers_sliding: Vec<bool>,
    pub eos: Vec<u32>,
    pub embed_normalizer: f32,
}

impl AssistantWgpuSpec {
    pub fn vocab_per_centroid(&self) -> usize {
        self.vocab / self.n_centroids
    }
}

pub struct AssistantLayerWeights {
    pub q: Vec<u16>,
    pub q_norm: Vec<u16>,
    pub o: Vec<u16>,
    pub gate: Vec<u16>,
    pub up: Vec<u16>,
    pub down: Vec<u16>,
    pub ln_in: Vec<u16>,
    pub ln_post_attn: Vec<u16>,
    pub ln_pre_ff: Vec<u16>,
    pub ln_post_ff: Vec<u16>,
    pub scalar: f32,
}

pub struct AssistantWgpuWeights {
    pub pre: Vec<u16>,
    pub post: Vec<u16>,
    pub norm: Vec<u16>,
    pub lm_head: Vec<u16>,
    pub centroids: Vec<u16>,
    pub ordering: Vec<u32>,
    pub layers: Vec<AssistantLayerWeights>,
}

pub struct BackboneKvBinding<'a> {
    pub k_fp8: &'a wgpu::Buffer,
    pub v_fp8: &'a wgpu::Buffer,
    pub k_scales: &'a wgpu::Buffer,
    pub v_scales: &'a wgpu::Buffer,
}

fn default_inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect()
}

fn proportional_inv_freq(head_dim: usize, theta: f32, partial_rotary_factor: f64) -> Vec<f32> {
    let rope_angles = ((partial_rotary_factor * head_dim as f64) as usize) / 2;
    let mut out = Vec::with_capacity(head_dim / 2);
    for i in 0..rope_angles {
        out.push(1.0 / theta.powf(2.0 * i as f32 / head_dim as f32));
    }
    out.resize(head_dim / 2, 0.0);
    out
}

fn cos_sin(inv_freq: &[f32], position: usize) -> (Vec<f32>, Vec<f32>) {
    let hd = inv_freq.len() * 2;
    let mut cos = Vec::with_capacity(hd);
    let mut sin = Vec::with_capacity(hd);
    for _ in 0..2 {
        for &f in inv_freq {
            let ang = position as f32 * f;
            cos.push(ang.cos());
            sin.push(ang.sin());
        }
    }
    (cos, sin)
}

fn grid1(threads: usize, wg: usize) -> (u32, u32, u32) {
    ((threads.div_ceil(wg).max(1)) as u32, 1, 1)
}

pub struct Gemma4AssistantWgpu {
    ctx: &'static WgpuContext,
    spec: AssistantWgpuSpec,
    max_len: usize,
    k_max: usize,
    step: Recorded,
    sliding_inv: Vec<f32>,
    full_inv: Vec<f32>,
    tok: GpuTensor<u32>,
    steps: GpuTensor<u32>,
    count: GpuTensor<u32>,
    hidden: GpuTensor<f32>,
    cos_s: GpuTensor<f32>,
    sin_s: GpuTensor<f32>,
    cos_f: GpuTensor<f32>,
    sin_f: GpuTensor<f32>,
    attn_s: GpuUniform<ad::AdAttnParams>,
    attn_f: GpuUniform<ad::AdAttnParams>,
    attn_s_base: ad::AdAttnParams,
    attn_f_base: ad::AdAttnParams,
    last_position: Option<usize>,
    unpack_decode: Option<Recorded>,
    unpack_verify: Option<Recorded>,
    unpack_p: Option<GpuUniform<UpkParams>>,
    _keep: Vec<Box<dyn std::any::Any>>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct UpkParams {
    src_off: u32,
    n_words: u32,
    pad0: u32,
    pad1: u32,
}

const UNPACK_HIDDEN_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4a_unpack_hidden.wgsl");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedHiddenSrc {
    Decode,
    Verify { word_off: usize },
}

impl Gemma4AssistantWgpu {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spec: AssistantWgpuSpec,
        weights: &AssistantWgpuWeights,
        embed_chunks: &[wgpu::Buffer],
        embed_rows_per_chunk: usize,
        sliding_kv: BackboneKvBinding<'_>,
        full_kv: BackboneKvBinding<'_>,
        max_len: usize,
        k_max: usize,
    ) -> Result<Self> {
        let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("no wgpu adapter: {e}"))?;
        let (bh, h, nh) = (spec.backbone_hidden, spec.hidden, spec.n_heads);
        let imm = spec.intermediate;
        let hd_max = spec.sliding_hd.max(spec.full_hd);
        let cand = spec.top_k * spec.vocab_per_centroid();
        anyhow::ensure!(k_max > 0 && max_len > 0, "empty drafter geometry");
        anyhow::ensure!(
            !spec.layers_sliding.is_empty() && spec.layers_sliding.len() == weights.layers.len(),
            "layer count mismatch: {} types vs {} weight sets",
            spec.layers_sliding.len(),
            weights.layers.len()
        );
        anyhow::ensure!(
            spec.n_heads.is_multiple_of(spec.sliding_nkv)
                && spec.n_heads.is_multiple_of(spec.full_nkv),
            "n_heads {} not divisible by kv heads {}/{}",
            spec.n_heads,
            spec.sliding_nkv,
            spec.full_nkv
        );
        anyhow::ensure!(
            spec.vocab.is_multiple_of(spec.n_centroids),
            "vocab {} not divisible by {} centroids",
            spec.vocab,
            spec.n_centroids
        );
        anyhow::ensure!(
            weights.ordering.len() == spec.vocab,
            "token ordering holds {}, want {}",
            weights.ordering.len(),
            spec.vocab
        );
        for (name, kv, nkv, hd) in [
            ("sliding", &sliding_kv, spec.sliding_nkv, spec.sliding_hd),
            ("full", &full_kv, spec.full_nkv, spec.full_hd),
        ] {
            let fp8 = (max_len * nkv * hd) as u64;
            let scales = (max_len * nkv * 4) as u64;
            anyhow::ensure!(
                kv.k_fp8.size() >= fp8
                    && kv.v_fp8.size() >= fp8
                    && kv.k_scales.size() >= scales
                    && kv.v_scales.size() >= scales,
                "{name} kv buffers too small for max_len {max_len} x {nkv} x {hd}"
            );
        }
        anyhow::ensure!(
            !embed_chunks.is_empty()
                && embed_chunks.len() <= ad::EMBED_MAX_CHUNKS
                && embed_rows_per_chunk > 0,
            "embed table split into {} chunks; drafter supports 1..={}",
            embed_chunks.len(),
            ad::EMBED_MAX_CHUNKS
        );
        anyhow::ensure!(
            embed_chunks.len() * embed_rows_per_chunk >= spec.vocab,
            "embed table chunks cover {} rows, want {}",
            embed_chunks.len() * embed_rows_per_chunk,
            spec.vocab
        );

        let check = |name: &str, data: &[u16], want: usize| -> Result<()> {
            anyhow::ensure!(
                data.len() == want,
                "{name}: {} elements, want {want}",
                data.len()
            );
            Ok(())
        };
        check("pre_projection", &weights.pre, h * 2 * bh)?;
        check("post_projection", &weights.post, bh * h)?;
        check("norm", &weights.norm, h)?;
        check("lm_head", &weights.lm_head, spec.vocab * h)?;
        check("centroids", &weights.centroids, spec.n_centroids * h)?;

        let src = compose(ad::WGSL);
        let mut keep: Vec<Box<dyn std::any::Any>> = Vec::new();
        let up16 = |ctx: &'static WgpuContext, label: &str, bits: &[u16]| -> GpuTensor<u32> {
            GpuTensor::upload(ctx, label, &pack_pairs(bits))
        };

        let xcat = GpuTensor::<f32>::zeroed(ctx, "adw-xcat", 2 * bh);
        let xa = GpuTensor::<f32>::zeroed(ctx, "adw-xa", h);
        let xb = GpuTensor::<f32>::zeroed(ctx, "adw-xb", h);
        let t0 = GpuTensor::<f32>::zeroed(ctx, "adw-t0", h);
        let t1 = GpuTensor::<f32>::zeroed(ctx, "adw-t1", h);
        let q = GpuTensor::<f32>::zeroed(ctx, "adw-q", nh * hd_max);
        let q2 = GpuTensor::<f32>::zeroed(ctx, "adw-q2", nh * hd_max);
        let ctxb = GpuTensor::<f32>::zeroed(ctx, "adw-ctx", nh * hd_max);
        let gbuf = GpuTensor::<f32>::zeroed(ctx, "adw-g", imm);
        let hn = GpuTensor::<f32>::zeroed(ctx, "adw-hn", h);
        let cent = GpuTensor::<f32>::zeroed(ctx, "adw-cent", spec.n_centroids);
        let scores = GpuTensor::<f32>::zeroed(ctx, "adw-scores", nh * max_len);
        let top_idx = GpuTensor::<u32>::zeroed(ctx, "adw-top", spec.top_k);
        let cand_ids = GpuTensor::<u32>::zeroed(ctx, "adw-cand-ids", cand);
        let cand_logits = GpuTensor::<f32>::zeroed(ctx, "adw-cand-logits", cand);
        let hidden = GpuTensor::<f32>::zeroed(ctx, "adw-hidden", bh);
        let tok = GpuTensor::<u32>::zeroed(ctx, "adw-tok", 1);
        let steps = GpuTensor::<u32>::zeroed(ctx, "adw-steps", k_max);
        let count = GpuTensor::<u32>::zeroed(ctx, "adw-count", 1);
        let cos_s = GpuTensor::<f32>::zeroed(ctx, "adw-cos-s", spec.sliding_hd);
        let sin_s = GpuTensor::<f32>::zeroed(ctx, "adw-sin-s", spec.sliding_hd);
        let cos_f = GpuTensor::<f32>::zeroed(ctx, "adw-cos-f", spec.full_hd);
        let sin_f = GpuTensor::<f32>::zeroed(ctx, "adw-sin-f", spec.full_hd);

        let w_pre = up16(ctx, "adw-w-pre", &weights.pre);
        let w_post = up16(ctx, "adw-w-post", &weights.post);
        let w_norm = up16(ctx, "adw-w-norm", &weights.norm);
        let w_head = up16(ctx, "adw-w-head", &weights.lm_head);
        let w_cent = up16(ctx, "adw-w-cent", &weights.centroids);
        let w_order = GpuTensor::upload(ctx, "adw-w-order", &weights.ordering);

        let u_embed = GpuUniform::new(
            ctx,
            "adw-u-embed",
            &ad::AdEmbedParams {
                bh: bh as u32,
                rows_per_chunk: embed_rows_per_chunk as u32,
                norm: spec.embed_normalizer,
                ..Default::default()
            },
        );
        let gemv_u = |label: &str, n: usize, k: usize, act: u32, mode: u32| {
            GpuUniform::new(
                ctx,
                label,
                &ad::AdGemvParams {
                    n: n as u32,
                    k: k as u32,
                    act,
                    mode,
                },
            )
        };
        let rms_u = |label: &str, rows: usize, dim: usize| {
            GpuUniform::new(
                ctx,
                label,
                &ad::AdRmsParams {
                    rows: rows as u32,
                    dim: dim as u32,
                    eps: spec.eps,
                    ..Default::default()
                },
            )
        };
        let u_pre = gemv_u("adw-u-pre", h, 2 * bh, 0, 0);
        let u_post = gemv_u("adw-u-post", bh, h, 0, 0);
        let u_cent = gemv_u("adw-u-cent", spec.n_centroids, h, 0, 0);
        let u_rms_h = rms_u("adw-u-rms-h", 1, h);
        let u_qn_s = rms_u("adw-u-qn-s", nh, spec.sliding_hd);
        let u_qn_f = rms_u("adw-u-qn-f", nh, spec.full_hd);
        let u_rope_s = GpuUniform::new(
            ctx,
            "adw-u-rope-s",
            &ad::AdRopeParams {
                nh: nh as u32,
                hd: spec.sliding_hd as u32,
                ..Default::default()
            },
        );
        let u_rope_f = GpuUniform::new(
            ctx,
            "adw-u-rope-f",
            &ad::AdRopeParams {
                nh: nh as u32,
                hd: spec.full_hd as u32,
                ..Default::default()
            },
        );
        let attn_s_base = ad::AdAttnParams {
            n_kv: spec.sliding_nkv as u32,
            nh: nh as u32,
            hd: spec.sliding_hd as u32,
            len: 0,
            start: 0,
            stride: max_len as u32,
            ..Default::default()
        };
        let attn_f_base = ad::AdAttnParams {
            n_kv: spec.full_nkv as u32,
            nh: nh as u32,
            hd: spec.full_hd as u32,
            len: 0,
            start: 0,
            stride: max_len as u32,
            ..Default::default()
        };
        let attn_s = GpuUniform::new(ctx, "adw-u-attn-s", &attn_s_base);
        let attn_f = GpuUniform::new(ctx, "adw-u-attn-f", &attn_f_base);
        let u_add_one = GpuUniform::new(
            ctx,
            "adw-u-add-one",
            &ad::AdAddParams {
                n: h as u32,
                scale: 1.0,
                ..Default::default()
            },
        );
        let u_topk = GpuUniform::new(
            ctx,
            "adw-u-topk",
            &ad::AdTopkParams {
                n: spec.n_centroids as u32,
                k: spec.top_k as u32,
                ..Default::default()
            },
        );
        let u_cand = GpuUniform::new(
            ctx,
            "adw-u-cand",
            &ad::AdCandParams {
                top: spec.top_k as u32,
                vpc: spec.vocab_per_centroid() as u32,
                h: h as u32,
                ..Default::default()
            },
        );
        let u_pick = GpuUniform::new(
            ctx,
            "adw-u-pick",
            &ad::AdPickParams {
                n: cand as u32,
                ..Default::default()
            },
        );

        let mut step = Recorded::new();
        let mut push =
            |entry: &str, binds: &[(u32, &dyn GpuBind)], grid: (u32, u32, u32)| -> Result<()> {
                step.push(ctx, LABEL, &src, entry, binds, grid)
                    .map_err(|e| anyhow::anyhow!("record {entry}: {e}"))
            };

        let chunk = |i: usize| embed_chunks.get(i).unwrap_or(&embed_chunks[0]);
        push(
            ad::ENTRY_EMBED_CONCAT,
            &[
                (0, chunk(0)),
                (1, chunk(1)),
                (2, chunk(2)),
                (3, chunk(3)),
                (4, &tok),
                (5, &hidden),
                (6, &xcat),
                (7, &u_embed),
            ],
            grid1(bh, 256),
        )?;
        push(
            ad::ENTRY_GEMV,
            &[(10, &w_pre), (11, &xcat), (12, &xa), (13, &u_pre)],
            (h as u32, 1, 1),
        )?;

        let mut layer_keep: Vec<Box<dyn std::any::Any>> = Vec::new();
        for (li, lw) in weights.layers.iter().enumerate() {
            let sliding = spec.layers_sliding[li];
            let hd = if sliding {
                spec.sliding_hd
            } else {
                spec.full_hd
            };
            let q_dim = nh * hd;
            check(&format!("layer {li} q_proj"), &lw.q, q_dim * h)?;
            check(&format!("layer {li} q_norm"), &lw.q_norm, hd)?;
            check(&format!("layer {li} o_proj"), &lw.o, h * q_dim)?;
            check(&format!("layer {li} gate"), &lw.gate, imm * h)?;
            check(&format!("layer {li} up"), &lw.up, imm * h)?;
            check(&format!("layer {li} down"), &lw.down, h * imm)?;
            let wq = up16(ctx, "adw-w-q", &lw.q);
            let wqn = up16(ctx, "adw-w-qn", &lw.q_norm);
            let wo = up16(ctx, "adw-w-o", &lw.o);
            let wg = up16(ctx, "adw-w-gate", &lw.gate);
            let wu = up16(ctx, "adw-w-up", &lw.up);
            let wd = up16(ctx, "adw-w-down", &lw.down);
            let ln_in = up16(ctx, "adw-w-ln-in", &lw.ln_in);
            let ln_pa = up16(ctx, "adw-w-ln-pa", &lw.ln_post_attn);
            let ln_pf = up16(ctx, "adw-w-ln-pf", &lw.ln_pre_ff);
            let ln_po = up16(ctx, "adw-w-ln-po", &lw.ln_post_ff);
            let u_q = gemv_u("adw-u-q", q_dim, h, 0, 0);
            let u_o = gemv_u("adw-u-o", h, q_dim, 0, 0);
            let u_gate = gemv_u("adw-u-gate", imm, h, 1, 0);
            let u_up = gemv_u("adw-u-up", imm, h, 0, 1);
            let u_down = gemv_u("adw-u-down", h, imm, 0, 0);
            let u_add_layer = GpuUniform::new(
                ctx,
                "adw-u-add-layer",
                &ad::AdAddParams {
                    n: h as u32,
                    scale: lw.scalar,
                    ..Default::default()
                },
            );
            let (u_qn, u_rope, u_attn, kv) = if sliding {
                (&u_qn_s, &u_rope_s, &attn_s, &sliding_kv)
            } else {
                (&u_qn_f, &u_rope_f, &attn_f, &full_kv)
            };
            let (cos, sin) = if sliding {
                (&cos_s, &sin_s)
            } else {
                (&cos_f, &sin_f)
            };

            push(
                ad::ENTRY_RMSNORM,
                &[(20, &xa), (21, &ln_in), (22, &t0), (23, &u_rms_h)],
                (1, 1, 1),
            )?;
            push(
                ad::ENTRY_GEMV,
                &[(10, &wq), (11, &t0), (12, &q), (13, &u_q)],
                (q_dim as u32, 1, 1),
            )?;
            push(
                ad::ENTRY_RMSNORM,
                &[(20, &q), (21, &wqn), (22, &q2), (23, u_qn)],
                (nh as u32, 1, 1),
            )?;
            push(
                ad::ENTRY_ROPE,
                &[(30, &q2), (31, cos), (32, sin), (33, u_rope)],
                grid1(q_dim, 256),
            )?;
            push(
                ad::ENTRY_ATTN_SCORES,
                &[
                    (40, &q2),
                    (41, kv.k_fp8),
                    (43, kv.k_scales),
                    (45, &scores),
                    (47, u_attn),
                ],
                (max_len.div_ceil(256) as u32, nh as u32, 1),
            )?;
            push(
                ad::ENTRY_ATTN_SOFTMAX,
                &[(45, &scores), (47, u_attn)],
                (nh as u32, 1, 1),
            )?;
            push(
                ad::ENTRY_ATTN_CTX,
                &[
                    (42, kv.v_fp8),
                    (44, kv.v_scales),
                    (45, &scores),
                    (46, &ctxb),
                    (47, u_attn),
                ],
                grid1(q_dim, 256),
            )?;
            push(
                ad::ENTRY_GEMV,
                &[(10, &wo), (11, &ctxb), (12, &t1), (13, &u_o)],
                (h as u32, 1, 1),
            )?;
            push(
                ad::ENTRY_RMSNORM,
                &[(20, &t1), (21, &ln_pa), (22, &t0), (23, &u_rms_h)],
                (1, 1, 1),
            )?;
            push(
                ad::ENTRY_ADD_SCALE,
                &[(50, &xa), (51, &t0), (52, &xb), (53, &u_add_one)],
                grid1(h, 256),
            )?;
            push(
                ad::ENTRY_RMSNORM,
                &[(20, &xb), (21, &ln_pf), (22, &t0), (23, &u_rms_h)],
                (1, 1, 1),
            )?;
            push(
                ad::ENTRY_GEMV,
                &[(10, &wg), (11, &t0), (12, &gbuf), (13, &u_gate)],
                (imm as u32, 1, 1),
            )?;
            push(
                ad::ENTRY_GEMV,
                &[(10, &wu), (11, &t0), (12, &gbuf), (13, &u_up)],
                (imm as u32, 1, 1),
            )?;
            push(
                ad::ENTRY_GEMV,
                &[(10, &wd), (11, &gbuf), (12, &t1), (13, &u_down)],
                (h as u32, 1, 1),
            )?;
            push(
                ad::ENTRY_RMSNORM,
                &[(20, &t1), (21, &ln_po), (22, &t0), (23, &u_rms_h)],
                (1, 1, 1),
            )?;
            push(
                ad::ENTRY_ADD_SCALE,
                &[(50, &xb), (51, &t0), (52, &xa), (53, &u_add_layer)],
                grid1(h, 256),
            )?;

            layer_keep.push(Box::new((wq, wqn, wo, wg, wu, wd)));
            layer_keep.push(Box::new((ln_in, ln_pa, ln_pf, ln_po)));
            layer_keep.push(Box::new((u_q, u_o, u_gate, u_up, u_down, u_add_layer)));
        }

        push(
            ad::ENTRY_RMSNORM,
            &[(20, &xa), (21, &w_norm), (22, &hn), (23, &u_rms_h)],
            (1, 1, 1),
        )?;
        push(
            ad::ENTRY_GEMV,
            &[(10, &w_post), (11, &hn), (12, &hidden), (13, &u_post)],
            (bh as u32, 1, 1),
        )?;
        push(
            ad::ENTRY_GEMV,
            &[(10, &w_cent), (11, &hn), (12, &cent), (13, &u_cent)],
            (spec.n_centroids as u32, 1, 1),
        )?;
        push(
            ad::ENTRY_TOPK,
            &[(60, &cent), (61, &top_idx), (62, &u_topk)],
            (1, 1, 1),
        )?;
        push(
            ad::ENTRY_CAND_LOGITS,
            &[
                (70, &top_idx),
                (71, &w_order),
                (72, &w_head),
                (73, &hn),
                (74, &cand_ids),
                (75, &cand_logits),
                (76, &u_cand),
            ],
            grid1(cand, 64),
        )?;
        push(
            ad::ENTRY_PICK,
            &[
                (80, &cand_ids),
                (81, &cand_logits),
                (82, &tok),
                (83, &steps),
                (84, &count),
                (85, &u_pick),
            ],
            (1, 1, 1),
        )?;

        keep.push(Box::new((
            xcat, xa, xb, t0, t1, q, q2, ctxb, gbuf, hn, cent,
        )));
        keep.push(Box::new((scores, top_idx, cand_ids, cand_logits)));
        keep.push(Box::new((w_pre, w_post, w_norm, w_head, w_cent, w_order)));
        keep.push(Box::new((
            u_embed, u_pre, u_post, u_cent, u_rms_h, u_qn_s, u_qn_f,
        )));
        keep.push(Box::new((
            u_rope_s, u_rope_f, u_add_one, u_topk, u_cand, u_pick,
        )));
        keep.extend(layer_keep);

        let sliding_inv = default_inv_freq(spec.sliding_hd, spec.sliding_theta);
        let full_inv = proportional_inv_freq(spec.full_hd, spec.full_theta, spec.full_partial);

        Ok(Self {
            ctx,
            spec,
            max_len,
            k_max,
            step,
            sliding_inv,
            full_inv,
            tok,
            steps,
            count,
            hidden,
            cos_s,
            sin_s,
            cos_f,
            sin_f,
            attn_s,
            attn_f,
            attn_s_base,
            attn_f_base,
            last_position: None,
            unpack_decode: None,
            unpack_verify: None,
            unpack_p: None,
            _keep: keep,
        })
    }

    pub fn spec(&self) -> &AssistantWgpuSpec {
        &self.spec
    }

    pub fn bind_hidden_sources(
        &mut self,
        decode_hid: &wgpu::Buffer,
        verify_hid: Option<&wgpu::Buffer>,
    ) -> Result<()> {
        let n_words = self.spec.backbone_hidden / 2;
        let upk = GpuUniform::new(
            self.ctx,
            "adw-upk-p",
            &UpkParams {
                src_off: 0,
                n_words: n_words as u32,
                pad0: 0,
                pad1: 0,
            },
        );
        let pl = dispatch::cached_compute_pipeline(
            self.ctx,
            "adw-unpack-hidden",
            UNPACK_HIDDEN_WGSL,
            "adw_unpack_hidden",
        )
        .map_err(err)?;
        let grid = dispatch::workgroup_count_1d(self.ctx, n_words as u64, 256);
        let mk_rec = |src: &wgpu::Buffer| {
            let mut r = Recorded::new();
            r.push_raw(
                self.ctx,
                pl.clone(),
                &[(0, src), (1, self.hidden.raw()), (2, upk.raw())],
                grid,
            );
            r
        };
        self.unpack_decode = Some(mk_rec(decode_hid));
        self.unpack_verify = verify_hid.map(mk_rec);
        self.unpack_p = Some(upk);
        Ok(())
    }

    fn prepare_round(&mut self, last_token: u32, committed: usize, k: usize) -> Result<()> {
        anyhow::ensure!(committed > 0, "no committed context to draft from");
        anyhow::ensure!(
            committed <= self.max_len,
            "committed {committed} beyond drafter max_len {}",
            self.max_len
        );
        anyhow::ensure!(
            k >= 1 && k <= self.k_max,
            "draft k {k} out of 1..={}",
            self.k_max
        );
        let s_len = committed.min(self.spec.sliding_window.max(1));
        let s_start = committed - s_len;
        let mut ps = self.attn_s_base;
        ps.len = s_len as u32;
        ps.start = s_start as u32;
        self.attn_s.write(self.ctx, &ps);
        let mut pf = self.attn_f_base;
        pf.len = committed as u32;
        pf.start = 0;
        self.attn_f.write(self.ctx, &pf);
        if self.last_position != Some(committed) {
            let (cs, ss) = cos_sin(&self.sliding_inv, committed);
            self.cos_s.write(self.ctx, &cs).map_err(err)?;
            self.sin_s.write(self.ctx, &ss).map_err(err)?;
            let (cf, sf) = cos_sin(&self.full_inv, committed);
            self.cos_f.write(self.ctx, &cf).map_err(err)?;
            self.sin_f.write(self.ctx, &sf).map_err(err)?;
            self.last_position = Some(committed);
        }
        self.tok.write(self.ctx, &[last_token]).map_err(err)?;
        self.count.write(self.ctx, &[0]).map_err(err)?;
        Ok(())
    }

    fn drain_tokens(&mut self, k: usize) -> Result<Vec<u32>> {
        let mut toks = self
            .steps
            .download_range(self.ctx, 0, k)
            .map_err(err)
            .context("assistant wgpu drafter token readback")?;
        if let Some(p) = toks.iter().position(|t| self.spec.eos.contains(t)) {
            toks.truncate(p + 1);
        }
        Ok(toks)
    }

    pub fn propose(
        &mut self,
        last_token: u32,
        last_hidden: &[f32],
        committed: usize,
        k: usize,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(
            last_hidden.len() == self.spec.backbone_hidden,
            "last_hidden holds {} values, want {}",
            last_hidden.len(),
            self.spec.backbone_hidden
        );
        self.prepare_round(last_token, committed, k)?;
        self.hidden.write(self.ctx, last_hidden).map_err(err)?;
        self.step.replay_n(self.ctx, k).map_err(err)?;
        self.drain_tokens(k)
    }

    pub fn propose_packed(
        &mut self,
        last_token: u32,
        committed: usize,
        k: usize,
        src: PackedHiddenSrc,
    ) -> Result<Vec<u32>> {
        let n_words = (self.spec.backbone_hidden / 2) as u32;
        let word_off = match src {
            PackedHiddenSrc::Decode => 0usize,
            PackedHiddenSrc::Verify { word_off } => word_off,
        };
        {
            let upk = self
                .unpack_p
                .as_ref()
                .context("propose_packed before bind_hidden_sources")?;
            upk.write(
                self.ctx,
                &UpkParams {
                    src_off: word_off as u32,
                    n_words,
                    pad0: 0,
                    pad1: 0,
                },
            );
        }
        self.prepare_round(last_token, committed, k)?;
        let rec = match src {
            PackedHiddenSrc::Decode => self.unpack_decode.as_mut(),
            PackedHiddenSrc::Verify { .. } => self.unpack_verify.as_mut(),
        };
        rec.context("propose_packed hidden source not bound")?
            .replay(self.ctx)
            .map_err(err)?;
        self.step.replay_n(self.ctx, k).map_err(err)?;
        self.drain_tokens(k)
    }
}

fn err(e: nv_kernels::wgpu_backend::WgpuError) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}
