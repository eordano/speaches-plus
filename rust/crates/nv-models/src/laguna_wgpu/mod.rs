pub mod attn;
pub mod config;
pub mod dense;
pub mod gpu;
pub mod moe;
pub mod weights;

use anyhow::Result;

use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::WgpuContext;

pub use config::{
    rope_inv_freq, rope_tables_from_inv_freq, window_start, FfnKind, GateKind, LagunaConfig,
    LagunaGating, LagunaShapes, LayerShape, LayerType, MlpLayerType, ARGMAX_GROUPS, MAX_HEAD_DIM,
    MAX_TOPK, NVFP4_BLOCK,
};
pub use gpu::{VramReport, STEP_POS, STEP_SLIDING_START, STEP_TOKEN, STEP_TOTAL};
pub use weights::{
    bf16_bits, bf16_val, dequantize_nvfp4_host, quantize_nvfp4_host, HostAttention, HostBf16Lin,
    HostDenseMlp, HostExperts, HostFfn, HostLayer, HostLin, HostMoe, HostNvfp4Lin, HostWeights,
    WeightSource,
};

use attn::RopeTablesGpu;
use config::rope_tables_from_inv_freq as rope_tables;
use gpu::{
    push_residual_add, push_rmsnorm, push_rmsnorm_residual, ArgmaxParams, Builder, GatherParams,
    GemvBf16Params, Pass, Sources, StepBuffers, StepUniform,
};
use weights::{pack_pairs, HostBf16Lin as Bf16Lin};

pub const PREFILL_M_MAX: usize = 64;

pub fn prefill_m() -> usize {
    match std::env::var("NV_LAGUNA_WGPU_PREFILL_M")
        .or_else(|_| std::env::var("NV_WGPU_PREFILL_M"))
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        Some(0) => 0,
        Some(m) => m.clamp(2, PREFILL_M_MAX),
        None => PREFILL_M_MAX,
    }
}

pub fn prefill_list_bytes_per_token_charged_by_mem_fit() -> usize {
    std::mem::size_of::<StepUniform>()
}

pub struct LagunaWgpu {
    ctx: &'static WgpuContext,
    config: LagunaConfig,
    shapes: LagunaShapes,
    max_seq_tokens: usize,
    pos: usize,
    validated: bool,
    prefix_validated: bool,
    passes: Vec<Pass>,
    head_start: usize,
    _buffers: Vec<wgpu::Buffer>,
    step: StepBuffers,
    token_out: wgpu::Buffer,
    logits: wgpu::Buffer,
    state_buffers: Vec<(wgpu::Buffer, u64)>,
    vocab: usize,
    vram: VramReport,
    pf_list: Option<wgpu::Buffer>,
    pf_m: usize,
    v_tokens: Option<wgpu::Buffer>,
    pf_validated: bool,
}

impl LagunaWgpu {
    pub fn new(config: LagunaConfig, weights: &HostWeights, max_seq_tokens: usize) -> Result<Self> {
        Self::build(config, WeightSource::Host(weights), max_seq_tokens)
    }

    pub fn from_loader(
        config: LagunaConfig,
        weights: &nv_weights::WeightLoader,
        max_seq_tokens: usize,
    ) -> Result<Self> {
        Self::build(config, WeightSource::Loader(weights), max_seq_tokens)
    }

    pub fn config(&self) -> &LagunaConfig {
        &self.config
    }

    pub fn shapes(&self) -> &LagunaShapes {
        &self.shapes
    }

    pub fn vram_report(&self) -> &VramReport {
        &self.vram
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    pub fn current_pos(&self) -> usize {
        self.pos
    }

    pub fn max_seq_tokens(&self) -> usize {
        self.max_seq_tokens
    }

    pub fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        if self.state_buffers.is_empty() {
            return Ok(());
        }
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        for (buf, _) in &self.state_buffers {
            enc.clear_buffer(buf, 0, None);
        }
        self.ctx.queue.submit([enc.finish()]);
        Ok(())
    }

    fn build(config: LagunaConfig, src: WeightSource<'_>, max_seq_tokens: usize) -> Result<Self> {
        let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
        let shapes = LagunaShapes::derive(&config, max_seq_tokens)?;
        let s = Sources::new();

        let hidden = shapes.hidden_size;
        let hidden_words = shapes.hidden_words();
        let eps = shapes.rms_norm_eps;
        let vocab = shapes.vocab_size;
        let bytes = (hidden_words * 4) as u64;

        let mut b = Builder::new(ctx);
        let step = StepBuffers::alloc(&mut b);

        let res = b.zeros("lgw-res", bytes);
        let res2 = b.zeros("lgw-res2", bytes);
        let normed = b.zeros("lgw-normed", bytes);
        let normed_post = b.zeros("lgw-normed-post", bytes);
        let attn_out = b.zeros("lgw-attn-out", bytes);
        let ffn_out = b.zeros("lgw-ffn-out", bytes);

        let (fcos, fsin) = rope_tables(&shapes.rope_inv_freq_full, max_seq_tokens);
        let rope_full = RopeTablesGpu {
            cos: b.upload_f32("lgw-rope-full-cos", &fcos),
            sin: b.upload_f32("lgw-rope-full-sin", &fsin),
            half: shapes.rope_inv_freq_full.len(),
        };
        let (scos, ssin) = rope_tables(&shapes.rope_inv_freq_sliding, max_seq_tokens);
        let rope_sliding = RopeTablesGpu {
            cos: b.upload_f32("lgw-rope-slide-cos", &scos),
            sin: b.upload_f32("lgw-rope-slide-sin", &ssin),
            half: shapes.rope_inv_freq_sliding.len(),
        };

        let embed = src.embed(&shapes)?;
        anyhow::ensure!(
            embed.len() == vocab * hidden,
            "embed has {} values, want {}",
            embed.len(),
            vocab * hidden
        );
        let chunk_rows = b.row_chunk(hidden);
        let mut off = 0usize;
        while off < vocab {
            let rows = chunk_rows.min(vocab - off);
            let buf = b.upload_u32(
                "lgw-embed",
                &pack_pairs(&embed[off * hidden..(off + rows) * hidden]),
            );
            let p = b.uni(
                "lgw-embed-p",
                GatherParams {
                    row_off: off as u32,
                    n_rows: rows as u32,
                    hidden_words: hidden_words as u32,
                    vocab: vocab as u32,
                },
            );
            let grid = b.grid1(hidden_words as u64, 256);
            b.push(
                "lgw-gather",
                &s.common,
                "lgw_gather_embed",
                &[(30, &buf), (31, &step.tok), (32, &res), (33, &p)],
                grid,
            )?;
            off += rows;
        }
        drop(embed);

        for li in 0..shapes.num_layers {
            let layer_shape = *shapes.layer(li);
            let hl = src.layer(&shapes, li)?;
            if li == 0 {
                let ln_w = b.upload_u32("lgw-ln", &pack_pairs(&hl.input_ln));
                push_rmsnorm(&mut b, &s, "lgw-rms0", &res, &ln_w, &normed, hidden, eps)?;
            }

            let rope = if layer_shape.is_sliding() {
                &rope_sliding
            } else {
                &rope_full
            };
            attn::build_attn_layer(
                &mut b,
                &s,
                &shapes,
                &layer_shape,
                &hl.attn,
                &normed,
                &attn_out,
                &step,
                rope,
            )?;

            let post_w = b.upload_u32("lgw-post-ln", &pack_pairs(&hl.post_attn_ln));
            push_rmsnorm_residual(
                &mut b,
                &s,
                "lgw-rmsres-post",
                &attn_out,
                &res,
                &post_w,
                &normed_post,
                hidden,
                eps,
            )?;

            match &hl.ffn {
                HostFfn::Dense(d) => dense::build_dense_mlp(
                    &mut b,
                    &s,
                    &shapes,
                    &layer_shape,
                    d,
                    &normed_post,
                    &ffn_out,
                )?,
                HostFfn::Moe(m) => moe::build_moe_layer(
                    &mut b,
                    &s,
                    &shapes,
                    &layer_shape,
                    m,
                    &normed_post,
                    &ffn_out,
                )?,
            }

            if li + 1 < shapes.num_layers {
                let next = src.layer_input_ln(&shapes, li + 1)?;
                let nw = b.upload_u32("lgw-next-ln", &pack_pairs(&next));
                push_rmsnorm_residual(
                    &mut b,
                    &s,
                    "lgw-rmsres-next",
                    &ffn_out,
                    &res,
                    &nw,
                    &normed,
                    hidden,
                    eps,
                )?;
            } else {
                push_residual_add(&mut b, &s, "lgw-resadd", &ffn_out, &res, &res2, hidden)?;
            }
        }

        let final_w = b.upload_u32("lgw-final-ln", &pack_pairs(&src.final_norm(&shapes)?));
        let final_x = b.zeros("lgw-final-x", bytes);
        push_rmsnorm(
            &mut b,
            &s,
            "lgw-final-rms",
            &res2,
            &final_w,
            &final_x,
            hidden,
            eps,
        )?;

        let head_start = b.passes.len();
        let logits = b.zeros("lgw-logits", (vocab * 4) as u64);
        let lm = src.lm_head(&shapes)?;
        anyhow::ensure!(
            lm.len() == vocab * hidden,
            "lm_head has {} values, want {}",
            lm.len(),
            vocab * hidden
        );
        let mut off = 0usize;
        while off < vocab {
            let rows = (chunk_rows.min(vocab - off)) & !1usize;
            let rows = if rows == 0 { vocab - off } else { rows };
            let wbuf = b.upload_u32(
                "lgw-lmhead",
                &pack_pairs(&lm[off * hidden..(off + rows) * hidden]),
            );
            let pairs = rows.div_ceil(2);
            let grid = b.grid1(pairs as u64, 1);
            let p = b.uni(
                "lgw-lmhead-p",
                GemvBf16Params {
                    n_rows: rows as u32,
                    k_words: hidden_words as u32,
                    groups_x: grid.0,
                    out_f32: 1,
                    w_row_words: hidden_words as u32,
                    x_off_words: 0,
                    y_off_words: off as u32,
                    alpha: 1.0,
                    ..Default::default()
                },
            );
            b.push(
                "lgw-lmhead",
                &s.gemv_bf16,
                "lgw_gemv_bf16",
                &[(0, &wbuf), (1, &final_x), (2, &p), (3, &logits)],
                grid,
            )?;
            off += rows;
        }
        drop(lm);

        let pv = b.zeros("lgw-am-pv", (ARGMAX_GROUPS * 4) as u64);
        let pi = b.zeros("lgw-am-pi", (ARGMAX_GROUPS * 4) as u64);
        let token_out = b.zeros("lgw-token", 4);
        let ap = b.uni(
            "lgw-am-p",
            ArgmaxParams {
                n: vocab as u32,
                groups: ARGMAX_GROUPS as u32,
                ..Default::default()
            },
        );
        b.push(
            "lgw-am1",
            &s.common,
            "lgw_argmax_stage1",
            &[(40, &logits), (41, &pv), (42, &pi), (44, &ap)],
            (ARGMAX_GROUPS as u32, 1, 1),
        )?;
        b.push(
            "lgw-am2",
            &s.common,
            "lgw_argmax_stage2",
            &[(41, &pv), (42, &pi), (43, &token_out), (44, &ap)],
            (1, 1, 1),
        )?;

        b.flush_staging();
        let vram = b.report();
        if gpu::vram_report_enabled() {
            eprint!("[laguna-wgpu] {}", vram.render());
        }

        let pf_m = prefill_m();
        let pf_list = (pf_m >= 2).then(|| {
            dispatch::storage_zeroed(
                ctx,
                "lgw-pf-list",
                (pf_m * prefill_list_bytes_per_token_charged_by_mem_fit()) as u64,
            )
        });
        let v_tokens = (pf_m >= 2)
            .then(|| dispatch::storage_zeroed(ctx, "lgw-verify-tokens", (pf_m * 4) as u64));

        let Builder {
            core,
            passes,
            state_buffers,
            ..
        } = b;
        let buffers = core.buffers;

        Ok(Self {
            ctx,
            config,
            shapes,
            max_seq_tokens,
            pos: 0,
            validated: false,
            prefix_validated: false,
            passes,
            head_start,
            _buffers: buffers,
            step,
            token_out,
            logits,
            state_buffers,
            vocab,
            vram,
            pf_list,
            pf_m,
            v_tokens,
            pf_validated: false,
        })
    }

    fn step_inner(&mut self, token: u32, full: bool) -> Result<()> {
        anyhow::ensure!((token as usize) < self.vocab, "token {token} out of vocab");
        anyhow::ensure!(
            self.pos < self.max_seq_tokens,
            "kv cache full at {} (max_seq_tokens {})",
            self.pos,
            self.max_seq_tokens
        );
        let total = self.pos + 1;
        let sliding_start = window_start(total, Some(self.shapes.sliding_window));
        self.step.write(
            self.ctx,
            token,
            self.pos as u32,
            total as u32,
            sliding_start as u32,
        );

        let need_scope = if full {
            !self.validated
        } else {
            !self.validated && !self.prefix_validated
        };
        let scope = if need_scope {
            Some(
                self.ctx
                    .device
                    .push_error_scope(wgpu::ErrorFilter::Validation),
            )
        } else {
            None
        };
        let passes = if full {
            &self.passes[..]
        } else {
            &self.passes[..self.head_start]
        };
        let labels: Vec<&str> = if dispatch::profile::enabled() {
            passes.iter().map(|p| p.entry.as_str()).collect()
        } else {
            Vec::new()
        };
        let cb = dispatch::encode_pass_list_labeled(
            self.ctx,
            passes.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
            &labels,
        );
        self.ctx.queue.submit([cb]);
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("laguna_wgpu decode step validation: {e}");
            }
            if full {
                self.validated = true;
            }
            self.prefix_validated = true;
        }
        self.pos += 1;
        Ok(())
    }

    crate::wgpu_step_readback_api!();

    pub fn prefill_chunk_len(&self) -> usize {
        if self.pf_list.is_none() {
            0
        } else {
            self.pf_m
        }
    }

    fn prefill_chunk_one_submission_of_per_token_passes(&mut self, chunk: &[u32]) -> Result<()> {
        let n = chunk.len();
        anyhow::ensure!(
            (2..=self.pf_m).contains(&n),
            "prefill chunk is {n} tokens, want 2..={}; a 1-token chunk is the per-token path \
             wearing the chunked path's name",
            self.pf_m
        );
        self.one_submission_of_per_token_passes(chunk, false)?;
        self.pos += n;
        Ok(())
    }

    fn one_submission_of_per_token_passes(&mut self, chunk: &[u32], with_head: bool) -> Result<()> {
        let n = chunk.len();
        anyhow::ensure!(
            (1..=self.pf_m).contains(&n),
            "batched submission is {n} tokens, want 1..={}",
            self.pf_m
        );
        anyhow::ensure!(
            self.pos + n <= self.max_seq_tokens,
            "kv cache full at {} + {n} (max_seq_tokens {})",
            self.pos,
            self.max_seq_tokens
        );
        for &t in chunk {
            anyhow::ensure!((t as usize) < self.vocab, "token {t} out of vocab");
        }
        let rec = std::mem::size_of::<StepUniform>();
        let mut host: Vec<u8> = Vec::with_capacity(n * rec);
        for (i, &t) in chunk.iter().enumerate() {
            let total = self.pos + i + 1;
            host.extend_from_slice(bytemuck::bytes_of(&StepUniform {
                tok: t,
                pos: (self.pos + i) as u32,
                total: total as u32,
                sliding_start: window_start(total, Some(self.shapes.sliding_window)) as u32,
            }));
        }
        let list = self
            .pf_list
            .clone()
            .expect("prefill chunk without pf list");
        self.ctx.queue.write_buffer(&list, 0, &host);
        let scope = if self.pf_validated {
            None
        } else {
            Some(
                self.ctx
                    .device
                    .push_error_scope(wgpu::ErrorFilter::Validation),
            )
        };
        let end = if with_head {
            self.passes.len()
        } else {
            self.head_start
        };
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        for i in 0..n {
            let base = (i * rec) as u64;
            enc.copy_buffer_to_buffer(&list, base, &self.step.tok, 0, 4);
            enc.copy_buffer_to_buffer(&list, base, &self.step.step, 0, rec as u64);
            enc.copy_buffer_to_buffer(&list, base, &self.step.uni, 0, rec as u64);
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in &self.passes[..end] {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
            drop(pass);
            if let Some(out) = self.v_tokens.as_ref().filter(|_| with_head) {
                enc.copy_buffer_to_buffer(&self.token_out, 0, out, (i * 4) as u64, 4);
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("laguna_wgpu batched submission validation: {e}");
            }
            self.pf_validated = true;
            self.prefix_validated = true;
            if with_head {
                self.validated = true;
            }
        }
        Ok(())
    }

    pub fn verify_max_rows(&self) -> usize {
        if self.v_tokens.is_none() || self.pf_list.is_none() {
            0
        } else {
            self.pf_m
        }
    }

    pub fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
        let rows = self.verify_max_rows();
        anyhow::ensure!(
            rows > 0,
            "verify_chain needs the batched prefill submission: NV_LAGUNA_WGPU_PREFILL_M >= 2"
        );
        let n = batch.len();
        anyhow::ensure!(
            (1..=rows).contains(&n),
            "verify_chain batch of {n} out of 1..={rows}"
        );
        self.one_submission_of_per_token_passes(batch, true)?;
        let out = self
            .v_tokens
            .as_ref()
            .expect("verify rows imply the buffer");
        dispatch::read_back::<u32>(self.ctx, out, n)
            .map_err(|e| anyhow::anyhow!("verify token readback: {e}"))
    }

    pub fn advance(&mut self, n: usize) -> Result<()> {
        anyhow::ensure!(
            self.pos + n <= self.max_seq_tokens,
            "advance {n} from {} past max_seq_tokens {}",
            self.pos,
            self.max_seq_tokens
        );
        self.pos += n;
        Ok(())
    }

    pub fn prefill_tokens(&mut self, tokens: &[u32]) -> Result<usize> {
        let mut done = 0usize;
        let m = self.prefill_chunk_len();
        if m == 0 {
            return Ok(done);
        }
        loop {
            let left = tokens.len() - done;
            if left < 2 {
                return Ok(done);
            }
            let take = m
                .min(left)
                .min(self.max_seq_tokens.saturating_sub(self.pos));
            if take < 2 {
                return Ok(done);
            }
            self.prefill_chunk_one_submission_of_per_token_passes(&tokens[done..done + take])?;
            done += take;
        }
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        anyhow::ensure!(!tokens.is_empty(), "prefill needs at least one token");
        let (last, rest) = tokens.split_last().expect("non-empty");
        let done = self.prefill_tokens(rest)?;
        for t in &rest[done..] {
            self.prefill_step(*t)?;
        }
        self.decode_step(*last)
    }
}

pub struct RefState {
    pub kc: Vec<Vec<f32>>,
    pub vc: Vec<Vec<f32>>,
    pub pos: usize,
}

impl RefState {
    pub fn new(shapes: &LagunaShapes) -> Self {
        Self {
            kc: vec![Vec::new(); shapes.num_layers],
            vc: vec![Vec::new(); shapes.num_layers],
            pos: 0,
        }
    }
}

pub fn rbf(x: f32) -> f32 {
    bf16_val(bf16_bits(x))
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn softplus(x: f32) -> f32 {
    x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
}

fn warp_tree_sum(parts: &[f32; 256]) -> f32 {
    let mut s = *parts;
    let mut stride = 16usize;
    while stride > 0 {
        for l in 0..256 {
            if (l & 31) < stride {
                s[l] += s[l + stride];
            }
        }
        stride >>= 1;
    }
    let a = (s[0] + s[128]) + (s[64] + s[192]);
    let b = (s[32] + s[160]) + (s[96] + s[224]);
    a + b
}

fn block_tree_sum(parts: &[f32; 256]) -> f32 {
    let mut s = *parts;
    let mut stride = 128usize;
    while stride > 0 {
        for l in 0..stride {
            s[l] += s[l + stride];
        }
        stride >>= 1;
    }
    s[0]
}

pub fn ref_rmsnorm(x: &[f32], w: &[u16], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut parts = [0f32; 256];
    for (i, v) in x.iter().enumerate() {
        let l = i & 255;
        parts[l] = v.mul_add(*v, parts[l]);
    }
    let ss = warp_tree_sum(&parts);
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    (0..n).map(|i| rbf(x[i] * inv * bf16_val(w[i]))).collect()
}

pub fn ref_rmsnorm_residual(delta: &[f32], res: &mut [f32], w: &[u16], eps: f32) -> Vec<f32> {
    let n = res.len();
    let words = n / 2;
    let mut parts = [0f32; 256];
    let mut sums = vec![0f32; n];
    for i in 0..words {
        let lo = delta[2 * i] + res[2 * i];
        let hi = delta[2 * i + 1] + res[2 * i + 1];
        sums[2 * i] = lo;
        sums[2 * i + 1] = hi;
        let l = i & 255;
        parts[l] = parts[l] + lo * lo + hi * hi;
    }
    for i in 0..n {
        res[i] = rbf(sums[i]);
    }
    let ss = block_tree_sum(&parts);
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    (0..n).map(|i| rbf(res[i] * inv * bf16_val(w[i]))).collect()
}

pub fn ref_gemv_bf16(w: &Bf16Lin, x: &[f32]) -> Vec<f32> {
    let words = w.k.div_ceil(2);
    let mut y = vec![0f32; w.n];
    let mut red = [0f32; 128];
    for r in 0..w.n {
        red.fill(0.0);
        let base = r * w.k;
        for i in 0..words {
            let l = i & 127;
            let mut acc = red[l];
            acc = bf16_val(w.w[base + 2 * i]).mul_add(x[2 * i], acc);
            if 2 * i + 1 < w.k {
                acc = bf16_val(w.w[base + 2 * i + 1]).mul_add(x[2 * i + 1], acc);
            }
            red[l] = acc;
        }
        let mut stride = 64usize;
        while stride > 0 {
            for l in 0..stride {
                red[l] += red[l + stride];
            }
            stride >>= 1;
        }
        y[r] = red[0];
    }
    y
}

pub fn ref_quant_x(x: &[f32], global: f32) -> Vec<f32> {
    let row: Vec<f32> = x.to_vec();
    let t = nv_quant::nvfp4::Nvfp4Tensor::quantize_rows_with_global(&[row], global);
    t.dequantize().remove(0)
}

pub fn ref_gemv_nvfp4(lin: &HostNvfp4Lin, x: &[f32]) -> Vec<f32> {
    let w = dequantize_nvfp4_host(lin);
    let x_eff = ref_quant_x(x, lin.input_global);
    let mut y = vec![0f32; lin.n];
    for r in 0..lin.n {
        let mut acc = 0f32;
        for c in 0..lin.k {
            acc += w[r * lin.k + c] * x_eff[c];
        }
        y[r] = acc;
    }
    y
}

pub fn ref_gemv_lin(lin: &HostLin, x: &[f32]) -> Vec<f32> {
    match lin {
        HostLin::Bf16(l) => ref_gemv_bf16(l, x),
        HostLin::Nvfp4(l) => ref_gemv_nvfp4(l, x),
    }
}

pub fn reference_step(
    shapes: &LagunaShapes,
    hw: &HostWeights,
    st: &mut RefState,
    token: u32,
) -> Result<Vec<f32>> {
    let hidden = shapes.hidden_size;
    let eps = shapes.rms_norm_eps;
    let pos = st.pos;

    let (fcos, fsin) = rope_tables(&shapes.rope_inv_freq_full, pos + 1);
    let (scos, ssin) = rope_tables(&shapes.rope_inv_freq_sliding, pos + 1);
    let fhalf = shapes.rope_inv_freq_full.len();
    let shalf = shapes.rope_inv_freq_sliding.len();

    let mut res: Vec<f32> = (0..hidden)
        .map(|i| bf16_val(hw.embed[token as usize * hidden + i]))
        .collect();

    let mut normed = ref_rmsnorm(&res, &hw.layers[0].input_ln, eps);
    for li in 0..shapes.num_layers {
        let layer_shape = *shapes.layer(li);
        let layer = &hw.layers[li];
        let (cos_row, sin_row) = if layer_shape.is_sliding() {
            (
                &scos[pos * shalf..(pos + 1) * shalf],
                &ssin[pos * shalf..(pos + 1) * shalf],
            )
        } else {
            (
                &fcos[pos * fhalf..(pos + 1) * fhalf],
                &fsin[pos * fhalf..(pos + 1) * fhalf],
            )
        };
        let mixed = attn::ref_attn(
            shapes,
            &layer_shape,
            &layer.attn,
            &normed,
            cos_row,
            sin_row,
            st,
            pos,
        )?;
        let normed_post = ref_rmsnorm_residual(&mixed, &mut res, &layer.post_attn_ln, eps);
        let ffn = match &layer.ffn {
            HostFfn::Dense(d) => dense::ref_dense_mlp(shapes, &layer_shape, d, &normed_post)?,
            HostFfn::Moe(m) => moe::ref_moe(shapes, m, &normed_post)?,
        };
        if li + 1 < shapes.num_layers {
            normed = ref_rmsnorm_residual(&ffn, &mut res, &hw.layers[li + 1].input_ln, eps);
        } else {
            for i in 0..hidden {
                res[i] = rbf(res[i] + ffn[i]);
            }
        }
    }

    let fx = ref_rmsnorm(&res, &hw.final_norm, eps);
    let lm = Bf16Lin {
        w: if hw.lm_head.is_empty() {
            hw.embed.clone()
        } else {
            hw.lm_head.clone()
        },
        n: shapes.vocab_size,
        k: hidden,
    };
    st.pos += 1;
    Ok(ref_gemv_bf16(&lm, &fx))
}

pub fn ref_argmax(logits: &[f32]) -> u32 {
    let mut best = f32::NEG_INFINITY;
    let mut bi = 0u32;
    for (i, v) in logits.iter().enumerate() {
        if *v > best {
            best = *v;
            bi = i as u32;
        }
    }
    bi
}

crate::wgpu_state_snapshot::impl_wgpu_state_snapshot!(LagunaWgpu, max_seq_tokens);
