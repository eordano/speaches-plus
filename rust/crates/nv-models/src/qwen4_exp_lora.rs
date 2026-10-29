use super::*;
use candle_core::Var;
use nv_layers::linear::LoraDeltaHook;
use nv_layers::moe_train::{MarlinMoeTrainStacks, MoeExpertsMarlinActRowsOp, MoeExpertsMarlinWeightedTrainOp, PackedNvfp4Rows};
use nv_train::qwen38::{gdn_scan, TrainGdn};
use nv_train::{LoraConfig, LoraTrainable};
pub use nv_train::{RawLora, TokenRow};

#[path = "qwen4_exp_tiny.rs"]
pub mod tiny;
#[path = "qwen4_exp_expert_lora.rs"]
pub mod expert;
use expert::{ExpertLoraForm, ExpertLoraLayer};
use std::collections::HashMap;

pub const LINEAR_TARGETS_ARE_THE_BF16_CHECKPOINT_PROJECTIONS_AND_THE_ROUTED_EXPERTS_ADAPT_THROUGH_THE_MOE_LAYER_EXPERT_HOOK: &str =
    "linear targets are the bf16-checkpoint projections every serving path runs through \
     Linear::forward: full-attention q/k/v/o and the shared expert gate/up/down. The routed experts \
     run as one grouped NVFP4 GEMM with no per-expert Linear, so they adapt only through the \
     MoeLayer expert-LoRA hook (expert::ExpertLoraLayer, --expert-lora), which adds a low-rank \
     delta on each routed expert's down projection beside the untouched grouped GEMM";

pub const EXPERT_LORA_DECODE_RUNS_EAGER_BECAUSE_ITS_ROUTED_ROWS_ARE_PLANNED_ON_THE_HOST: &str =
    "the expert-LoRA delta sorts the routed slots by expert on the host and recomputes the served \
     gate/up rows through the prefill dispatch, which reads its tile plan back from the device, so \
     a model with an expert hook attached decodes eagerly instead of replaying the captured graph";

pub const GDN_PROJECTIONS_ARE_NOT_TARGETS_BECAUSE_THE_FUSED_DECODE_KERNELS_READ_THEIR_WEIGHTS_DIRECTLY: &str =
    "the gated-delta-net decode path is a fused kernel over the projection weights, so a hook on \
     its in_proj or out_proj Linear would apply in prefill and silently vanish in decode";

pub const THE_LOSS_IS_THE_ENGINE_FORWARD_AND_THE_GRADIENT_IS_A_PER_LAYER_TWIN_EVALUATED_AT_THE_ENGINE_ACTIVATIONS: &str =
    "every step runs the serving forward (the same Qwen4Exp layer code, grouped NVFP4 W4A4 \
     experts, fp8 side GEMMs, fused hyper-connection and attention kernels, with the adapter \
     applied by the same Linear hook seam serving uses) and records the hyper-connection streams \
     entering each trainable layer. The backward then re-runs one layer at a time as an autograd \
     twin started from those exact streams: candle ops for the hyper-connection glue, the \
     serving fp8 GEMM with a dX kernel for the side projections, Marlin W4A16 over the same \
     NVFP4 bytes for the experts. A twin's rounding differs from the engine's kernels, but each \
     layer's Jacobian is taken at the activations the engine actually produced, so the error \
     does not compound through depth";

pub const FROZEN_EXPERT_STACKS_HAND_OVER_THROUGH_PINNED_HOST_RAM_ONE_LAYER_AHEAD_OF_THE_TWIN_BACKWARD: &str =
    "with --host-stacks-gib the window layers' Marlin train stacks are built once, parked in pinned \
     host RAM and dropped from the device; each backward uploads the next lower layer's stacks on a \
     copy stream while the current layer's twin runs, so the device holds two layers' stacks instead \
     of the whole window";

pub const THE_TWIN_EXPERT_ACTIVATION_TAKES_THE_SERVED_W4A4_VALUES_AND_THE_MARLIN_W4A16_GRADIENT: &str =
    "the expert-LoRA delta reads act_e(x); the twin recomputes it through the served W4A4 grouped \
     gate/up for its value and through Marlin W4A16 for its gradient (straight-through), so the \
     adapter's own gradient is taken at the activations serving feeds it";

pub const DEFAULT_TARGETS: [&str; 7] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];

fn layer_prefix(li: usize) -> String {
    format!("model.language_model.layers.{li}")
}

pub fn module_name(li: usize, leaf: &str) -> String {
    match leaf {
        "q_proj" | "k_proj" | "v_proj" | "o_proj" => format!("{}.self_attn.{leaf}", layer_prefix(li)),
        _ => format!("{}.mlp.shared_expert.{leaf}", layer_prefix(li)),
    }
}

impl Qwen4Exp {
    pub fn lora_target_linears(&self) -> Vec<(String, usize, &'static str, &Linear)> {
        let mut out = Vec::new();
        for (li, layer) in self.layers.iter().enumerate() {
            if let Mixer::Attn(a) = &layer.mixer {
                for (leaf, lin) in [("q_proj", &a.q_proj), ("k_proj", &a.k_proj), ("v_proj", &a.v_proj), ("o_proj", &a.o_proj)] {
                    out.push((module_name(li, leaf), li, leaf, lin));
                }
            }
            let m = &layer.moe;
            for (leaf, lin) in [("gate_proj", &m.shared_gate_proj), ("up_proj", &m.shared_up_proj), ("down_proj", &m.shared_down_proj)] {
                out.push((module_name(li, leaf), li, leaf, lin));
            }
        }
        out
    }

    pub fn detach_all_lora(&self) {
        for (_, _, _, lin) in self.lora_target_linears() {
            lin.detach_lora();
        }
        for layer in &self.layers {
            layer.moe.detach_expert_lora();
        }
    }

    pub fn has_expert_lora(&self) -> bool {
        self.layers.iter().any(|l| l.moe.expert_lora().is_some())
    }

    pub fn attach_expert_lora(&self, li: usize, l: Arc<ExpertLoraLayer>) -> Result<()> {
        let layer = self.layers.get(li).ok_or_else(|| anyhow::anyhow!("expert lora layer {li} outside the model"))?;
        layer.moe.attach_expert_lora(l, &self.cfg)
    }

    pub fn embed_hc_rows(&self, tokens: &[u32]) -> Result<Tensor> {
        let stream = stream_of(&self.device)?;
        let tokens_dev: CudaSlice<u32> = stream.clone_htod(tokens)?;
        self.embed_hc(&tokens_dev, tokens.len())
    }

    pub fn hc_after_prefix(&self, tokens: &[u32], first_trainable: usize, state: &mut Qwen4ExpState) -> Result<Tensor> {
        ensure!(
            !tokens.is_empty() && tokens.len() <= PREFILL_CHUNK_2048_IS_THE_PREFIX_CHECKPOINT_GRID_SO_A_RESUMED_PREFILL_CHUNKS_EXACTLY_AS_A_COLD_ONE,
            "qwen4_exp lora: a training row of {} tokens is outside one serving prefill chunk",
            tokens.len()
        );
        state.reset(&self.cfg, &self.device)?;
        let hc = self.embed_hc_rows(tokens)?;
        if first_trainable == 0 {
            return Ok(hc);
        }
        let stream = stream_of(&self.device)?;
        let tokens_dev: CudaSlice<u32> = stream.clone_htod(tokens)?;
        let t = tokens.len();
        let positions = Tensor::from_vec((0..t as i32).collect::<Vec<_>>(), (t,), &self.device)?;
        let staged = self.staged_moe_ctx_for_rows(t)?;
        let mode = FwdMode {
            pos_buf: None,
            moe_ctx: if t == 1 { Some(&self.decode_ctx) } else { staged.as_deref() },
            verify_rows: false,
            last_layer: Some(first_trainable - 1),
        };
        let mut prof = PhaseProf::disabled();
        let (hc, pending) = self.layers_forward(hc, tokens, &tokens_dev, state, 0, &positions, mode, &mut prof, false)?;
        super::hc_settle(hc, pending, &self.cfg)
    }

    pub fn exact_layers_from(&self, hc: Tensor, tokens: &[u32], first: usize, state: &mut Qwen4ExpState) -> Result<(Vec<Tensor>, Tensor)> {
        let cfg = &self.cfg;
        let t = tokens.len();
        ensure!(
            self.layers[first..].iter().all(|l| l.ple.is_none()),
            "qwen4_exp lora: the trainable window starting at layer {first} holds a PLE layer, whose \
             n-gram history this exact window pass does not replay"
        );
        let positions = Tensor::from_vec((0..t as i32).collect::<Vec<_>>(), (t,), &self.device)?;
        let staged = self.staged_moe_ctx_for_rows(t)?;
        let moe_ctx = if t == 1 { Some(&self.decode_ctx) } else { staged.as_deref() };
        let mut hc = hc;
        let mut inputs = Vec::with_capacity(self.layers.len() - first);
        for layer in &self.layers[first..] {
            inputs.push(hc.clone());
            let (mixed, inject) = layer.attn_hc.mix(&hc, cfg)?;
            let y = match &layer.mixer {
                Mixer::Gdn(la) => self.gdn_forward(la, &mixed, &state.gdn[layer.gdn_slot.unwrap()], None)?,
                Mixer::Attn(a) => self.attention_forward(a, &mixed, &mut state.attn[layer.attn_slot.unwrap()], 0, &positions)?,
            };
            hc = hc_combine(&hc, &y, inject.as_ref().unwrap(), cfg)?;
            let (mixed, inject) = layer.mlp_hc.mix(&hc, cfg)?;
            let y = layer.moe.forward(&mixed, cfg, moe_ctx)?;
            hc = hc_combine(&hc, &y, inject.as_ref().unwrap(), cfg)?;
        }
        Ok((inputs, hc))
    }

    pub fn logits_from_hc(&self, hc: &Tensor) -> Result<Tensor> {
        let (mixed, _) = self.final_hc.mix(hc, &self.cfg)?;
        Ok(self.lm_head.forward(&mixed)?.to_dtype(DType::F32)?)
    }

    pub fn logits_all_rows(&self, tokens: &[u32], state: &mut Qwen4ExpState) -> Result<Tensor> {
        state.reset(&self.cfg, &self.device)?;
        let (_, hc) = self.forward_rows(tokens, state)?;
        self.logits_from_hc(&hc)
    }
}

pub struct ExactOrTwinLoraHook {
    lora: Arc<LoraTrainable>,
}

impl LoraDeltaHook for ExactOrTwinLoraHook {
    fn in_features(&self) -> usize {
        self.lora.in_features()
    }

    fn out_features(&self) -> usize {
        self.lora.out_features()
    }

    fn apply(&self, x2: &Tensor, y2: &Tensor, win: Option<(usize, usize)>) -> Result<Option<Tensor>> {
        let (a, b) = if x2.track_op() {
            (self.lora.a_tensor().clone(), self.lora.b_tensor().clone())
        } else {
            (self.lora.a_tensor().detach(), self.lora.b_tensor().detach())
        };
        let delta = nv_train::lora_delta(&a, &b, self.lora.scaling(), x2, win)?.to_dtype(y2.dtype())?;
        Ok(Some((y2 + delta)?))
    }
}

pub struct Site {
    pub module: String,
    pub layer: usize,
    pub lora: Arc<LoraTrainable>,
}

pub struct ExpertSite {
    pub layer: usize,
    pub a: Var,
    pub b: Var,
    pub scale: f64,
    hook: Arc<ExpertLoraLayer>,
}

#[derive(Clone, Debug)]
pub struct ExpertLoraArgs {
    pub form: ExpertLoraForm,
    pub rank: usize,
    pub alpha: f64,
    pub layers: usize,
}

pub const EXPERT_LORA_DEFAULT_FORM_EXPERT_A_RANK_16_OVER_THE_WINDOW_COSTS_0_47_GIB_OF_FP32_ADAM_STATE_ON_REAL_FLASH_NEXT: &str = "expert-a";

impl Default for ExpertLoraArgs {
    fn default() -> Self {
        Self { form: ExpertLoraForm::None, rank: 16, alpha: 32.0, layers: 6 }
    }
}

#[derive(Clone, Debug)]
pub struct LoraTrainArgs {
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<String>,
    pub window: usize,
    pub steps: usize,
    pub lr: f64,
    pub seed: u64,
    pub clip: f64,
    pub vram_reserve_gib: f64,
    pub prefix_cache_gib: f64,
    pub expert: ExpertLoraArgs,
    pub host_stacks_gib: f64,
    pub host_optimizer: bool,
}

impl Default for LoraTrainArgs {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            targets: DEFAULT_TARGETS.iter().map(|s| s.to_string()).collect(),
            window: 12,
            steps: 400,
            lr: 1e-4,
            seed: 20261011,
            clip: 1.0,
            vram_reserve_gib: 6.0,
            prefix_cache_gib: 4.0,
            expert: ExpertLoraArgs::default(),
            host_stacks_gib: 0.0,
            host_optimizer: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StepRecord {
    pub row: usize,
    pub tokens: usize,
    pub exact_loss: f32,
    pub twin_loss: f32,
    pub grad_norm: f32,
    pub skipped_nonfinite: bool,
    pub exact_ms: f64,
    pub backward_ms: f64,
    pub optimizer_ms: f64,
}

pub const HOST_OPTIMIZER_STREAMS_EACH_VARIABLES_ADAM_MOMENTS_THROUGH_PINNED_HOST_RAM_ONE_VARIABLE_AHEAD_OF_THE_UPDATE: &str =
    "with --host-optimizer the fp32 Adam moments live in pinned host buffers; the step uploads \
     variable i+1's moments on a copy stream while variable i updates, and downloads the new \
     moments on the same stream, so the device holds two variables' moments at a time";

enum Moments {
    Device(Vec<(Tensor, Tensor)>),
    Host { bufs: Vec<(nv_layers::pinned_host::PinnedHostBuf, nv_layers::pinned_host::PinnedHostBuf)>, shapes: Vec<Vec<usize>>, copy: Arc<CudaStream> },
}

struct Adam {
    moments: Moments,
    t: i32,
}

fn wrap_f32(slice: CudaSlice<f32>, shape: &[usize], device: &Device) -> Result<Tensor> {
    let Device::Cuda(dev) = device else { bail!("adam: not cuda") };
    Ok(Tensor::from_storage(
        Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(slice, dev.clone())),
        shape,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

impl Adam {
    fn new(vars: &[Var], host: bool) -> Result<Self> {
        let moments = if !host {
            Moments::Device(vars.iter().map(|v| Ok((v.as_tensor().zeros_like()?, v.as_tensor().zeros_like()?))).collect::<Result<_>>()?)
        } else {
            let device = vars.first().map(|v| v.as_tensor().device().clone()).context("adam: no variables")?;
            let stream = stream_of(&device)?;
            let mut bufs = Vec::with_capacity(vars.len());
            for v in vars {
                let bytes = v.as_tensor().elem_count() * 4;
                let zero = |what| -> Result<nv_layers::pinned_host::PinnedHostBuf> {
                    let mut b = nv_layers::pinned_host::PinnedHostBuf::alloc(&stream, bytes, what)?;
                    b.as_mut_slice().fill(0);
                    Ok(b)
                };
                bufs.push((zero("adam first moment")?, zero("adam second moment")?));
            }
            Moments::Host { bufs, shapes: vars.iter().map(|v| v.as_tensor().dims().to_vec()).collect(), copy: stream.fork()? }
        };
        Ok(Self { moments, t: 0 })
    }

    fn update(var: &Var, g: &Tensor, m: &Tensor, v: &Tensor, scale: f64, lr: f64, t: i32) -> Result<(Tensor, Tensor)> {
        let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
        let (bc1, bc2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
        let g = g.affine(scale, 0.0)?;
        let m = ((m * b1)? + (&g * (1.0 - b1))?)?;
        let s = ((v * b2)? + (g.sqr()? * (1.0 - b2))?)?;
        let upd = (m.affine(1.0 / bc1, 0.0)? / (s.affine(1.0 / bc2, 0.0)?.sqrt()? + eps)?)?;
        var.set(&(var.as_tensor() - (upd * lr)?)?)?;
        Ok((m, s))
    }

    fn step(&mut self, vars: &[Var], grads: &[Tensor], scale: f64, lr: f64) -> Result<()> {
        self.t += 1;
        let t = self.t;
        match &mut self.moments {
            Moments::Device(ms) => {
                for ((var, g), slot) in vars.iter().zip(grads).zip(ms.iter_mut()) {
                    *slot = Self::update(var, g, &slot.0, &slot.1, scale, lr, t)?;
                }
            }
            Moments::Host { bufs, shapes, copy } => {
                let _ = HOST_OPTIMIZER_STREAMS_EACH_VARIABLES_ADAM_MOMENTS_THROUGH_PINNED_HOST_RAM_ONE_VARIABLE_AHEAD_OF_THE_UPDATE;
                let Some(device) = vars.first().map(|v| v.as_tensor().device().clone()) else { return Ok(()) };
                let compute = stream_of(&device)?;
                let upload = |i: usize, bufs: &Vec<(nv_layers::pinned_host::PinnedHostBuf, nv_layers::pinned_host::PinnedHostBuf)>| -> Result<(CudaSlice<f32>, CudaSlice<f32>, cudarc::driver::CudaEvent)> {
                    let (m, _) = bufs[i].0.upload_on::<f32>(&compute, copy)?;
                    let (v, ev) = bufs[i].1.upload_on::<f32>(&compute, copy)?;
                    Ok((m, v, ev))
                };
                let mut next = if vars.is_empty() { None } else { Some(upload(0, bufs)?) };
                let mut keep = Vec::with_capacity(vars.len());
                for i in 0..vars.len() {
                    let (m, v, ev) = next.take().context("adam: missing prefetched moments")?;
                    if i + 1 < vars.len() {
                        next = Some(upload(i + 1, bufs)?);
                    }
                    compute.wait(&ev)?;
                    let (m, v) = (wrap_f32(m, &shapes[i], &device)?, wrap_f32(v, &shapes[i], &device)?);
                    let (m, v) = Self::update(&vars[i], &grads[i], &m, &v, scale, lr, t)?;
                    let (m, v) = (m.contiguous()?, v.contiguous()?);
                    let (bm, bv) = &mut bufs[i];
                    for (src, dst) in [(&m, bm), (&v, bv)] {
                        let (st, _) = src.storage_and_layout();
                        let Storage::Cuda(c) = &*st else { bail!("adam: moment not cuda") };
                        dst.download_into_on(c.as_cuda_slice::<f32>()?, &compute, copy)?;
                    }
                    keep.push((m, v));
                }
                copy.synchronize()?;
                drop(keep);
            }
        }
        Ok(())
    }

    fn device_bytes(&self) -> usize {
        match &self.moments {
            Moments::Device(ms) => ms.iter().map(|(m, v)| 4 * (m.elem_count() + v.elem_count())).sum(),
            Moments::Host { .. } => 0,
        }
    }

    fn host_bytes(&self) -> usize {
        match &self.moments {
            Moments::Device(_) => 0,
            Moments::Host { bufs, .. } => bufs.iter().map(|(m, v)| m.len() + v.len()).sum(),
        }
    }
}

pub struct TwinGdn {
    train: TrainGdn,
    norm_w: Tensor,
}

pub struct Qwen4ExpLoraTrainer<'m> {
    model: &'m Qwen4Exp,
    pub sites: Vec<Site>,
    pub experts: Vec<ExpertSite>,
    pub first_trainable: usize,
    args: LoraTrainArgs,
    gdn: HashMap<usize, TwinGdn>,
    stacks: HashMap<usize, Arc<MarlinMoeTrainStacks>>,
    host_stacks: HashMap<usize, nv_layers::moe_train::HostMarlinMoeTrainStacks>,
    host_budget: nv_layers::pinned_host::PinnedBudget,
    copy_stream: Option<Arc<CudaStream>>,
    pending: Option<(usize, Arc<MarlinMoeTrainStacks>, cudarc::driver::CudaEvent)>,
    pub stack_uploads: usize,
    stack_order: Vec<usize>,
    stack_cap: usize,
    pub stack_build_ms: f64,
    pub stack_bytes_per_layer: usize,
    rope: Rope,
    state: Qwen4ExpState,
    max_seq: usize,
    prefix: HashMap<usize, Tensor>,
    prefix_bytes: usize,
    adam: Adam,
}

fn dense_attention_limit_ok(cfg: &Qwen4ExpConfig, max_seq: usize) -> Result<()> {
    ensure!(
        max_seq <= cfg.dense_attention_context_limit()
            && max_seq <= PREFILL_CHUNK_2048_IS_THE_PREFIX_CHECKPOINT_GRID_SO_A_RESUMED_PREFILL_CHUNKS_EXACTLY_AS_A_COLD_ONE,
        "qwen4_exp lora: max_seq {max_seq} exceeds the dense-attention limit {} or one prefill chunk; \
         the twin attention is dense causal SDPA, which equals the engine only below the QSA budget",
        cfg.dense_attention_context_limit()
    );
    Ok(())
}

impl<'m> Qwen4ExpLoraTrainer<'m> {
    pub fn new(model: &'m Qwen4Exp, args: LoraTrainArgs, max_seq: usize) -> Result<Self> {
        let _ = LINEAR_TARGETS_ARE_THE_BF16_CHECKPOINT_PROJECTIONS_AND_THE_ROUTED_EXPERTS_ADAPT_THROUGH_THE_MOE_LAYER_EXPERT_HOOK;
        let _ = GDN_PROJECTIONS_ARE_NOT_TARGETS_BECAUSE_THE_FUSED_DECODE_KERNELS_READ_THEIR_WEIGHTS_DIRECTLY;
        let _ = THE_LOSS_IS_THE_ENGINE_FORWARD_AND_THE_GRADIENT_IS_A_PER_LAYER_TWIN_EVALUATED_AT_THE_ENGINE_ACTIVATIONS;
        let cfg = model.config();
        dense_attention_limit_ok(cfg, max_seq)?;
        let n = cfg.num_hidden_layers;
        let ple_max = cfg.ple_layer_ids_zero_based.iter().copied().max();
        let deepest = n - ple_max.map_or(0, |p| p + 1);
        ensure!(args.window >= 1 && deepest >= 1, "window {} leaves no trainable layer", args.window);
        let mut args = args;
        args.window = args.window.min(deepest);
        let first = n - args.window;
        for t in &args.targets {
            ensure!(
                DEFAULT_TARGETS.contains(&t.as_str()),
                "target {t:?} is not one of {DEFAULT_TARGETS:?}; {}",
                LINEAR_TARGETS_ARE_THE_BF16_CHECKPOINT_PROJECTIONS_AND_THE_ROUTED_EXPERTS_ADAPT_THROUGH_THE_MOE_LAYER_EXPERT_HOOK
            );
        }
        model.detach_all_lora();
        let lcfg = LoraConfig { r: args.rank, alpha: args.alpha, dropout: 0.0 };
        let mut sites = Vec::new();
        for (i, (module, li, leaf, lin)) in model.lora_target_linears().into_iter().enumerate() {
            if li < first || !args.targets.iter().any(|t| t == leaf) {
                continue;
            }
            let lora = Arc::new(LoraTrainable::from_features(
                lin.out_features(),
                lin.in_features(),
                lcfg,
                args.seed ^ (i as u64).wrapping_mul(0x9E37_79B9),
                &model.device,
            )?);
            lin.attach_lora(Arc::new(ExactOrTwinLoraHook { lora: lora.clone() }))?;
            sites.push(Site { module, layer: li, lora });
        }
        let mut experts = Vec::new();
        if args.expert.form != ExpertLoraForm::None {
            let e = &args.expert;
            let layers = e.layers.clamp(1, args.window);
            for li in n - layers..n {
                let (a, b) = expert::init_expert_lora(e.form, cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size, e.rank, args.seed ^ (0xE7 + li as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15), &model.device)?;
                let (a, b) = (Var::from_tensor(&a)?, Var::from_tensor(&b)?);
                let scale = e.alpha / e.rank as f64;
                let hook = Arc::new(ExpertLoraLayer::new(a.as_tensor().clone(), b.as_tensor().clone(), scale)?);
                model.attach_expert_lora(li, hook.clone())?;
                experts.push(ExpertSite { layer: li, a, b, scale, hook });
            }
        }
        ensure!(
            !sites.is_empty() || !experts.is_empty(),
            "qwen4_exp lora: targets {:?} match no projection in layers {first}..{n} and no expert lora was asked for",
            args.targets
        );
        let mut gdn = HashMap::new();
        for (li, layer) in model.layers.iter().enumerate().skip(first) {
            if let Mixer::Gdn(la) = &layer.mixer {
                let (lcfg, [qkv, z, a, b, out], [conv, a_log, dt_bias, norm_w]) = la.train_twin_parts_sharing_the_resident_projections();
                gdn.insert(li, TwinGdn { train: TrainGdn::new(lcfg, qkv, z, a, b, conv, a_log, dt_bias, norm_w.clone(), out), norm_w });
            }
        }
        let rope = Rope::new(
            RopeConfig { head_dim: cfg.rotary_dim, max_seq_len: max_seq.max(2), base: cfg.rope_theta, kind: RopeKind::Standard },
            &model.device,
        )?;
        let state = model.new_state(max_seq.max(2))?;
        let all_vars: Vec<Var> = sites.iter().flat_map(|s| s.lora.trainable_vars()).chain(experts.iter().flat_map(|e| [e.a.clone(), e.b.clone()])).collect();
        let adam = Adam::new(&all_vars, args.host_optimizer)?;
        let host_budget = nv_layers::pinned_host::PinnedBudget::new((args.host_stacks_gib.max(0.0) * (1u64 << 30) as f64) as usize);
        let copy_stream = if args.host_stacks_gib > 0.0 { Some(stream_of(&model.device)?.fork()?) } else { None };
        Ok(Self {
            model,
            sites,
            experts,
            first_trainable: first,
            args,
            gdn,
            stacks: HashMap::new(),
            host_stacks: HashMap::new(),
            host_budget,
            copy_stream,
            pending: None,
            stack_uploads: 0,
            stack_order: Vec::new(),
            stack_cap: usize::MAX,
            stack_build_ms: 0.0,
            stack_bytes_per_layer: 0,
            rope,
            state,
            max_seq,
            prefix: HashMap::new(),
            prefix_bytes: 0,
            adam,
        })
    }

    pub fn window(&self) -> usize {
        self.args.window
    }

    pub fn trainable_params(&self) -> usize {
        self.sites.iter().map(|s| s.lora.rank() * (s.lora.in_features() + s.lora.out_features())).sum::<usize>()
            + self.expert_params()
    }

    pub fn expert_params(&self) -> usize {
        self.experts.iter().map(|e| e.a.as_tensor().elem_count() + e.b.as_tensor().elem_count()).sum()
    }

    pub fn stacks_resident(&self) -> usize {
        self.stacks.len()
    }

    pub fn optimizer_device_bytes(&self) -> usize {
        self.adam.device_bytes()
    }

    pub fn optimizer_host_bytes(&self) -> usize {
        self.adam.host_bytes()
    }

    pub fn stacks_parked_on_host(&self) -> usize {
        self.host_stacks.len()
    }

    pub fn host_stack_bytes(&self) -> usize {
        self.host_stacks.values().map(|h| h.bytes()).sum()
    }

    fn prefetch_stacks(&mut self, li: usize) -> Result<()> {
        let _ = FROZEN_EXPERT_STACKS_HAND_OVER_THROUGH_PINNED_HOST_RAM_ONE_LAYER_AHEAD_OF_THE_TWIN_BACKWARD;
        if self.pending.is_some() || self.stacks.contains_key(&li) {
            return Ok(());
        }
        let (Some(h), Some(copy)) = (self.host_stacks.get(&li), self.copy_stream.as_ref()) else {
            return Ok(());
        };
        let compute = stream_of(&self.model.device)?;
        let (st, ev) = h.upload(&compute, copy)?;
        self.stack_uploads += 1;
        self.pending = Some((li, Arc::new(st), ev));
        Ok(())
    }

    fn stacks_for(&mut self, li: usize) -> Result<Arc<MarlinMoeTrainStacks>> {
        let compute = stream_of(&self.model.device)?;
        if let Some((pl, st, ev)) = self.pending.take() {
            if pl == li {
                compute.wait(&ev)?;
                return Ok(st);
            }
            self.pending = Some((pl, st, ev));
        }
        if let Some(s) = self.stacks.get(&li) {
            return Ok(s.clone());
        }
        if let (Some(h), Some(copy)) = (self.host_stacks.get(&li), self.copy_stream.as_ref()) {
            let (st, ev) = h.upload(&compute, copy)?;
            self.stack_uploads += 1;
            compute.wait(&ev)?;
            return Ok(Arc::new(st));
        }
        while self.stacks.len() >= self.stack_cap.max(1) {
            let old = self.stack_order.remove(0);
            self.stacks.remove(&old);
        }
        let model = self.model;
        let t0 = std::time::Instant::now();
        let st = Arc::new(stacks_from_checkpoint(&model.weights, model.config(), li, model.config().num_experts, &compute)?);
        compute.synchronize()?;
        self.stack_build_ms += 1000.0 * t0.elapsed().as_secs_f64();
        if self.stack_bytes_per_layer == 0 {
            self.stack_bytes_per_layer = st.device_bytes();
            let (free, _) = cudarc::driver::result::mem_get_info().unwrap_or((0, 0));
            let spare = free as f64 - self.args.vram_reserve_gib * (1u64 << 30) as f64;
            self.stack_cap = 1 + (spare.max(0.0) / st.device_bytes().max(1) as f64) as usize;
        }
        if self.copy_stream.is_some() && self.host_budget.try_take(st.device_bytes()) {
            self.host_stacks.insert(li, st.park_on_host(&compute)?);
            return Ok(st);
        }
        self.stacks.insert(li, st.clone());
        self.stack_order.push(li);
        Ok(st)
    }

    fn twin_attention(&self, a: &Attention, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
        twin_attention_at(self.model.config(), &self.rope, a, x, positions)
    }

    fn twin_gdn(&self, li: usize, x: &Tensor) -> Result<Tensor> {
        twin_gdn_at(self.model.config(), self.gdn.get(&li).ok_or_else(|| anyhow::anyhow!("no gdn twin for layer {li}"))?, x)
    }

    fn twin_moe(&self, li: usize, moe: &MoeLayer, x: &Tensor, stacks: Arc<MarlinMoeTrainStacks>) -> Result<Tensor> {
        let expert = self.experts.iter().find(|e| e.layer == li).map(|e| e.hook.as_ref());
        twin_moe_at(self.model.config(), moe, x, stacks, expert)
    }

    pub fn twin_layer(&mut self, li: usize, hc: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let stacks = self.stacks_for(li)?;
        self.twin_layer_with(li, hc, positions, stacks)
    }

    fn twin_layer_with(&self, li: usize, hc: &Tensor, positions: &Tensor, stacks: Arc<MarlinMoeTrainStacks>) -> Result<Tensor> {
        let cfg = self.model.config();
        let layer = &self.model.layers[li];
        let hc = hc.to_dtype(DType::BF16)?;
        let (mixed, inject) = layer.attn_hc.mix_eager(&hc, cfg)?;
        let y = match &layer.mixer {
            Mixer::Gdn(_) => self.twin_gdn(li, &mixed)?,
            Mixer::Attn(a) => self.twin_attention(a, &mixed, positions)?,
        };
        let hc = hc_combine_eager(&hc, &y, inject.as_ref().unwrap(), cfg)?;
        let (mixed, inject) = layer.mlp_hc.mix_eager(&hc, cfg)?;
        let y = self.twin_moe(li, &layer.moe, &mixed, stacks)?;
        hc_combine_eager(&hc, &y, inject.as_ref().unwrap(), cfg)
    }

    pub fn twin_component_errors(&mut self, li: usize, hc: &Tensor) -> Result<Vec<(&'static str, f32)>> {
        let rel = |a: &Tensor, b: &Tensor| -> Result<f32> {
            let a = a.to_dtype(DType::F32)?;
            let b = b.to_dtype(DType::F32)?;
            Ok((&a - &b)?.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? / b.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?.max(1e-30))
        };
        let model = self.model;
        let cfg = model.config();
        let layer = &model.layers[li];
        let (t, _) = hc.dims2()?;
        let hc = hc.to_dtype(DType::BF16)?;
        let var = Var::from_tensor(&hc.to_dtype(DType::F32)?)?;
        let tracked = var.as_tensor().to_dtype(DType::BF16)?;
        let positions = Tensor::from_vec((0..t as i32).collect::<Vec<_>>(), (t,), &model.device)?;
        let positions_u = Tensor::from_vec((0..t as u32).collect::<Vec<_>>(), (t,), &model.device)?;
        let (mixed, _) = layer.attn_hc.mix(&hc, cfg)?;
        let (mixed_twin, _) = layer.attn_hc.mix_eager(&tracked, cfg)?;
        let mut out = vec![("attn_hc_mix", rel(&mixed_twin, &mixed)?)];
        self.state.reset(cfg, &model.device)?;
        let tracked_mixed = (mixed.to_dtype(DType::F32)? + var.as_tensor().narrow(1, 0, cfg.hidden_size)?.affine(0.0, 0.0)?)?.to_dtype(DType::BF16)?;
        let (name, engine, twin) = match &layer.mixer {
            Mixer::Gdn(la) => ("gdn", model.gdn_forward(la, &mixed, &self.state.gdn[layer.gdn_slot.unwrap()], None)?, self.twin_gdn(li, &tracked_mixed)?),
            Mixer::Attn(a) => (
                "attention",
                model.attention_forward(a, &mixed, &mut self.state.attn[layer.attn_slot.unwrap()], 0, &positions)?,
                self.twin_attention(a, &tracked_mixed, &positions_u)?,
            ),
        };
        out.push((name, rel(&twin, &engine)?));
        let stacks = self.stacks_for(li)?;
        let staged = model.staged_moe_ctx_for_rows(t)?;
        let ctx = if t == 1 { Some(&model.decode_ctx) } else { staged.as_deref() };
        let moe_engine = layer.moe.forward(&mixed, cfg, ctx)?;
        let moe_twin = self.twin_moe(li, &layer.moe, &tracked_mixed, stacks)?;
        out.push(("moe", rel(&moe_twin, &moe_engine)?));
        Ok(out)
    }

    pub fn twin_logits(&self, hc: &Tensor) -> Result<Tensor> {
        let (mixed, _) = self.model.final_hc.mix_eager(&hc.to_dtype(DType::BF16)?, self.model.config())?;
        Ok(self.model.lm_head.forward(&mixed)?.to_dtype(DType::F32)?)
    }

    fn prefix_for(&mut self, row_id: usize, ids: &[u32]) -> Result<Tensor> {
        if let Some(h) = self.prefix.get(&row_id) {
            return Ok(h.to_device(&self.model.device)?);
        }
        let h = self.model.hc_after_prefix(ids, self.first_trainable, &mut self.state)?;
        let bytes = h.elem_count() * h.dtype().size_in_bytes();
        if ((self.prefix_bytes + bytes) as f64) < self.args.prefix_cache_gib * (1u64 << 30) as f64 {
            self.prefix_bytes += bytes;
            self.prefix.insert(row_id, h.to_device(&Device::Cpu)?);
        }
        Ok(h)
    }

    fn exact_streams(&mut self, row_id: usize, ids: &[u32]) -> Result<(Vec<Tensor>, Tensor)> {
        ensure!(ids.len() >= 2 && ids.len() <= self.max_seq, "row of {} tokens outside 2..={}", ids.len(), self.max_seq);
        let h = self.prefix_for(row_id, ids)?;
        self.state.reset(self.model.config(), &self.model.device)?;
        self.model.exact_layers_from(h, ids, self.first_trainable, &mut self.state)
    }

    pub fn exact_pass(&mut self, row_id: usize, ids: &[u32]) -> Result<(Vec<Tensor>, Tensor, Tensor)> {
        let (inputs, hc) = self.exact_streams(row_id, ids)?;
        let logits = self.model.logits_from_hc(&hc)?;
        Ok((inputs, hc, logits))
    }

    pub fn exact_loss(&mut self, row_id: usize, row: &TokenRow) -> Result<f32> {
        let (_, hc) = self.exact_streams(row_id, &row.ids)?;
        let (start, len) = scored_rows(row);
        Ok(scored_ce(&self.model.logits_from_hc(&hc.narrow(0, start, len)?)?, row)?.to_scalar::<f32>()?)
    }

    pub fn grads(&mut self, row_id: usize, row: &TokenRow) -> Result<(f32, f32, Vec<Tensor>)> {
        self.prefetch_stacks(self.model.layers.len() - 1)?;
        let (inputs, hc) = self.exact_streams(row_id, &row.ids)?;
        let (start, len) = scored_rows(row);
        let exact = scored_ce(&self.model.logits_from_hc(&hc.narrow(0, start, len)?)?, row)?.to_scalar::<f32>()?;
        let t = row.ids.len();
        let positions = Tensor::from_vec((0..t as u32).collect::<Vec<_>>(), (t,), &self.model.device)?;
        let top = Var::from_tensor(&hc.to_dtype(DType::F32)?)?;
        let twin_loss_t = scored_ce(&self.twin_logits(&top.as_tensor().narrow(0, start, len)?)?, row)?;
        let twin = twin_loss_t.to_scalar::<f32>()?;
        let gs = twin_loss_t.backward()?;
        let mut g = gs.get(top.as_tensor()).ok_or_else(|| anyhow::anyhow!("no gradient reached the final streams"))?.clone();
        drop(gs);
        let vars: Vec<(usize, Var)> = self
            .sites
            .iter()
            .flat_map(|s| s.lora.trainable_vars().into_iter().map(move |v| (s.layer, v)))
            .chain(self.experts.iter().flat_map(|e| [(e.layer, e.a.clone()), (e.layer, e.b.clone())]))
            .collect();
        let mut grads: Vec<Option<Tensor>> = vec![None; vars.len()];
        for (wi, li) in (self.first_trainable..self.model.layers.len()).enumerate().rev() {
            let stacks = self.stacks_for(li)?;
            if li > self.first_trainable {
                self.prefetch_stacks(li - 1)?;
            }
            let x = Var::from_tensor(&inputs[wi].to_dtype(DType::F32)?)?;
            let y = self.twin_layer_with(li, x.as_tensor(), &positions, stacks)?;
            let s = y.to_dtype(DType::F32)?.mul(&g)?.sum_all()?;
            let gs = s.backward()?;
            for (slot, (vl, v)) in vars.iter().enumerate() {
                if *vl == li {
                    if let Some(gv) = gs.get(v.as_tensor()) {
                        grads[slot] = Some(gv.clone());
                    }
                }
            }
            g = gs.get(x.as_tensor()).ok_or_else(|| anyhow::anyhow!("no gradient reached layer {li}'s input"))?.clone();
        }
        let grads = grads
            .into_iter()
            .zip(&vars)
            .map(|(g, (_, v))| g.map_or_else(|| v.as_tensor().zeros_like(), Ok))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok((exact, twin, grads))
    }

    pub fn vars(&self) -> Vec<Var> {
        self.sites.iter().flat_map(|s| s.lora.trainable_vars()).chain(self.experts.iter().flat_map(|e| [e.a.clone(), e.b.clone()])).collect()
    }

    pub fn step(&mut self, row_id: usize, row: &TokenRow) -> Result<StepRecord> {
        let t0 = std::time::Instant::now();
        let (exact, twin, grads) = self.grads(row_id, row)?;
        let backward_ms = 1000.0 * t0.elapsed().as_secs_f64();
        let mut sq = 0f64;
        for g in &grads {
            sq += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        }
        let norm = sq.sqrt() as f32;
        let mut rec = StepRecord { row: row_id, tokens: row.ids.len(), exact_loss: exact, twin_loss: twin, grad_norm: norm, backward_ms, ..Default::default() };
        if !norm.is_finite() {
            rec.skipped_nonfinite = true;
            return Ok(rec);
        }
        let scale = if self.args.clip > 0.0 && norm as f64 > self.args.clip { self.args.clip / norm as f64 } else { 1.0 };
        let t0 = std::time::Instant::now();
        let vars = self.vars();
        self.adam.step(&vars, &grads, scale, self.args.lr)?;
        rec.optimizer_ms = 1000.0 * t0.elapsed().as_secs_f64();
        Ok(rec)
    }

    pub fn train(&mut self, rows: &[TokenRow], mut on_step: impl FnMut(usize, &StepRecord)) -> Result<Vec<StepRecord>> {
        ensure!(!rows.is_empty(), "no training rows");
        let mut order: Vec<usize> = (0..rows.len()).collect();
        let mut rng = self.args.seed | 1;
        let mut out = Vec::with_capacity(self.args.steps);
        for step in 0..self.args.steps {
            if step % rows.len() == 0 {
                for i in (1..order.len()).rev() {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    order.swap(i, (rng >> 33) as usize % (i + 1));
                }
            }
            let r = order[step % rows.len()];
            let rec = self.step(r, &rows[r])?;
            on_step(step, &rec);
            out.push(rec);
        }
        Ok(out)
    }

    pub fn export_entries(&self) -> Result<Vec<RawLora>> {
        self.sites
            .iter()
            .map(|s| {
                Ok(RawLora {
                    module: s.module.clone(),
                    a: s.lora.a_tensor().to_dtype(DType::BF16)?,
                    b: s.lora.b_tensor().affine(s.lora.scaling(), 0.0)?.to_dtype(DType::BF16)?,
                })
            })
            .collect()
    }

    pub fn export(&self, dir: &Path, base_model_name: &str) -> Result<Vec<RawLora>> {
        self.export_for_base(dir, base_model_name, None)
    }

    pub fn expert_meta(&self, base_identity: Option<&str>) -> Option<expert::ExpertLoraMeta> {
        let cfg = self.model.config();
        (!self.experts.is_empty()).then(|| expert::ExpertLoraMeta {
            format: expert::EXPERT_LORA_FORMAT.to_string(),
            form: self.args.expert.form.as_str().to_string(),
            rank: self.args.expert.rank,
            layers: self.experts.iter().map(|e| e.layer).collect(),
            num_experts: cfg.num_experts,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
            base_identity: base_identity.map(str::to_string),
        })
    }

    pub fn export_expert_layers(&self) -> Result<Vec<(usize, Tensor, Tensor)>> {
        self.experts
            .iter()
            .map(|e| Ok((e.layer, e.a.as_tensor().to_dtype(DType::BF16)?, e.b.as_tensor().affine(e.scale, 0.0)?.to_dtype(DType::BF16)?)))
            .collect()
    }

    pub fn export_for_base(&self, dir: &Path, base_model_name: &str, base_identity: Option<&str>) -> Result<Vec<RawLora>> {
        let entries = self.export_entries()?;
        nv_train::save_peft_raw(dir, &entries, self.args.rank, self.args.rank as f64, 0.0, base_model_name)?;
        let cfg_path = dir.join("adapter_config.json");
        let mut cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)?;
        cfg["expert_lora"] = serde_json::json!(self.args.expert.form.as_str());
        if let Some(id) = base_identity {
            cfg["base_identity"] = serde_json::json!(id);
        }
        std::fs::write(&cfg_path, serde_json::to_string_pretty(&cfg)? + "\n")?;
        for stale in [expert::EXPERT_LORA_FILE, expert::EXPERT_LORA_META] {
            let _ = std::fs::remove_file(dir.join(stale));
        }
        if let Some(meta) = self.expert_meta(base_identity) {
            expert::write_expert_lora(dir, &meta, &self.export_expert_layers()?)?;
        }
        Ok(entries)
    }
}

fn stacks_from_checkpoint(w: &WeightLoader, cfg: &Qwen4ExpConfig, li: usize, experts: usize, stream: &Arc<CudaStream>) -> Result<MarlinMoeTrainStacks> {
    let mut raw: Vec<[(&[u8], &[u8], f32); 3]> = Vec::with_capacity(experts);
    for e in 0..experts {
        let get = |proj: &str| -> Result<(&[u8], &[u8], f32)> {
            let b = format!("{}.mlp.experts.{e}.{proj}", layer_prefix(li));
            Ok((w.raw_bytes(&format!("{b}.weight"))?, w.raw_bytes(&format!("{b}.weight_scale"))?, f32_scalar(w, &format!("{b}.weight_scale_2"))?))
        };
        raw.push([get("gate_proj")?, get("up_proj")?, get("down_proj")?]);
    }
    let rows: Vec<[PackedNvfp4Rows<'_>; 3]> = raw
        .iter()
        .map(|p| p.map(|(packed, scales_linear, global)| PackedNvfp4Rows { packed, scales_linear, global }))
        .collect();
    MarlinMoeTrainStacks::build_from_checkpoint_rows(&rows, cfg.hidden_size, cfg.moe_intermediate_size, stream)
}

fn twin_attention_at(cfg: &Qwen4ExpConfig, rope: &Rope, a: &Attention, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
    let (t, _) = x.dims2()?;
    let (nq, nkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
    let q_raw = a.q_proj.forward(x)?.reshape((t, nq, 2 * hd))?;
    let q = q_raw.narrow(2, 0, hd)?.contiguous()?;
    let gate = q_raw.narrow(2, hd, hd)?.contiguous()?.reshape((t, nq * hd))?;
    let k = a.k_proj.forward(x)?.reshape((t, nkv, hd))?.contiguous()?;
    let v = a.v_proj.forward(x)?.reshape((1, t, nkv, hd))?.contiguous()?;
    let q = head_rmsnorm_bf16(&q, &a.q_norm_w, cfg.rms_norm_eps)?.unsqueeze(0)?;
    let k = head_rmsnorm_bf16(&k, &a.k_norm_w, cfg.rms_norm_eps)?.unsqueeze(0)?;
    let q = rope.apply_train_partial_scaled(&q, positions, cfg.rotary_dim, hd, 1.0)?.to_dtype(DType::BF16)?;
    let k = rope.apply_train_partial_scaled(&k, positions, cfg.rotary_dim, hd, 1.0)?.to_dtype(DType::BF16)?;
    let out = nv_layers::attn::sdpa(
        &q,
        &k,
        &v,
        &AttnConfig { num_heads: nq, num_kv_heads: nkv, head_dim: hd, softmax_scale: 1.0 / (hd as f32).sqrt(), causal: true },
    )?;
    let gated = out
        .reshape((t, nq * hd))?
        .to_dtype(DType::F32)?
        .mul(&candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?)?
        .to_dtype(DType::BF16)?;
    a.o_proj.forward(&gated)
}

fn twin_gdn_at(cfg: &Qwen4ExpConfig, g: &TwinGdn, x: &Tensor) -> Result<Tensor> {
    let (t, _) = x.dims2()?;
    let p = g.train.head(&x.unsqueeze(0)?)?;
    let (n_v, d_k, d_v) = g.train.scan_dims();
    let core = gdn_scan(&p.q_scaled, &p.k_grp, &p.v_f, &p.g_exp, &p.beta, p.b, p.t, n_v, d_k, d_v)?;
    let core = core.to_dtype(DType::BF16)?;
    let xf = core.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf
        .broadcast_div(&(var + cfg.rms_norm_eps)?.sqrt()?)?
        .broadcast_mul(&g.norm_w.to_dtype(DType::F32)?)?
        .to_dtype(DType::BF16)?;
    let gate = candle_nn::ops::sigmoid(&p.z.reshape((1, t, n_v, d_v))?.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)?;
    let flat = normed.mul(&gate)?.reshape((t, n_v * d_v))?;
    g.train.out_proj.forward(&flat)
}

fn twin_moe_at(cfg: &Qwen4ExpConfig, moe: &MoeLayer, x: &Tensor, stacks: Arc<MarlinMoeTrainStacks>, expert: Option<&ExpertLoraLayer>) -> Result<Tensor> {
    let (t, _) = x.dims2()?;
    let (e, k) = (cfg.num_experts, cfg.num_experts_per_tok);
    let logits = moe.router.forward(x)?.to_dtype(DType::F32)?;
    let ids = nv_layers::moe_grouped::route_topk_ids_on_the_device_as_u32_tokens_by_k(&logits.detach(), None, t, e, k, 0, 0.0, x.device())?;
    let w = candle_nn::ops::softmax(&logits.gather(&ids, 1)?.contiguous()?, D::Minus1)?.contiguous()?;
    let xb = x.to_dtype(DType::BF16)?.contiguous()?;
    let op = MoeExpertsMarlinWeightedTrainOp::new_with_device_ids(stacks.clone(), ids.clone(), t, k);
    let mut routed = xb.apply_op2(&w, op)?.to_dtype(DType::F32)?;
    if let Some(l) = expert {
        let _ = THE_TWIN_EXPERT_ACTIVATION_TAKES_THE_SERVED_W4A4_VALUES_AND_THE_MARLIN_W4A16_GRADIENT;
        let marlin = xb.apply_op1(MoeExpertsMarlinActRowsOp::new_with_device_ids(stacks, ids.clone(), t, k))?.to_dtype(DType::F32)?;
        let slot_experts: Vec<u32> = ids.flatten_all()?.to_vec1()?;
        let served = nv_layers::moe_grouped::routed_expert_act_rows_through_the_served_w4a4_gate_up(&moe.grouped, &xb.detach(), &moe.router.forward(&xb.detach())?, t, k, x.device())?;
        let act = if served.slot_experts == slot_experts { (&marlin + (served.act_slots - marlin.detach())?)? } else { marlin };
        routed = (routed + l.delta(&act, &slot_experts, &w, t, k, true)?)?;
    }
    let g = moe.shared_gate_proj.forward(x)?.to_dtype(DType::F32)?;
    let u = moe.shared_up_proj.forward(x)?.to_dtype(DType::F32)?;
    let act = candle_nn::ops::silu(&g)?.mul(&u)?.to_dtype(DType::BF16)?;
    let y = moe.shared_down_proj.forward(&act)?.to_dtype(DType::F32)?;
    let gl = moe.shared_expert_gate.forward(x)?.to_dtype(DType::F32)?;
    let shared = y.broadcast_mul(&candle_nn::ops::sigmoid(&gl)?)?;
    Ok((routed + shared)?.to_dtype(DType::BF16)?)
}

pub const ONLY_COMPLETION_ROWS_REACH_THE_LM_HEAD_BECAUSE_HARNESS_PROMPTS_ARE_TEN_THOUSAND_TOKENS_AND_TARGETS_A_FEW_HUNDRED: &str =
    "the nightly SFT rows carry ~37k-character prompts against ~430-character targets, so the \
     248k-vocab logits, their softmax and their gradient are taken for the completion rows only";

pub fn scored_rows(row: &TokenRow) -> (usize, usize) {
    let start = row.completion_start.max(1) - 1;
    (start, row.ids.len() - 1 - start)
}

pub fn scored_ce(logits_scored: &Tensor, row: &TokenRow) -> Result<Tensor> {
    let _ = ONLY_COMPLETION_ROWS_REACH_THE_LM_HEAD_BECAUSE_HARNESS_PROMPTS_ARE_TEN_THOUSAND_TOKENS_AND_TARGETS_A_FEW_HUNDRED;
    let (start, len) = scored_rows(row);
    let targets = Tensor::from_vec(row.ids[start + 1..start + 1 + len].to_vec(), (len,), logits_scored.device())?;
    Ok(candle_nn::loss::cross_entropy(logits_scored, &targets)?)
}

pub fn served_row_ce(model: &Qwen4Exp, row: &TokenRow, state: &mut Qwen4ExpState) -> Result<f32> {
    state.reset(model.config(), model.device())?;
    let (_, hc) = model.forward_rows(&row.ids, state)?;
    let (start, len) = scored_rows(row);
    Ok(scored_ce(&model.logits_from_hc(&hc.narrow(0, start, len)?)?, row)?.to_scalar::<f32>()?)
}

pub fn row_ce(logits: &Tensor, row: &TokenRow) -> Result<Tensor> {
    let t = row.ids.len();
    let targets = Tensor::from_vec(row.ids[1..].to_vec(), (t - 1,), logits.device())?;
    let mask = row.target_mask_where_predicting_a_prompt_token_weighs_zero();
    nv_train::ce_mean_over_unmasked_positions(&logits.narrow(0, 0, t - 1)?, &targets, &mask)
}

pub fn attach_static_entries(model: &Qwen4Exp, entries: &[RawLora]) -> Result<usize> {
    model.detach_all_lora();
    let by_name: HashMap<String, &Linear> = model.lora_target_linears().into_iter().map(|(n, _, _, l)| (n, l)).collect();
    for e in entries {
        let lin = by_name
            .get(&e.module)
            .ok_or_else(|| anyhow::anyhow!("adapter module {} is not a qwen4_exp lora target", e.module))?;
        lin.attach_lora(Arc::new(nv_train::StaticLoraHook::new(e.a.clone(), e.b.clone(), 1.0)?))?;
    }
    Ok(entries.len())
}

pub const BENCH_EXPERT_SUBSET_KEEPS_THE_REAL_SHAPES_AND_TOP_K_WHILE_FITTING_BESIDE_A_LIVE_ENGINE: usize = 128;

fn free_gib() -> f64 {
    cudarc::driver::result::mem_get_info().map(|(f, _)| f as f64 / (1u64 << 30) as f64).unwrap_or(0.0)
}

fn timed(device: &Device, reps: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    let stream = stream_of(device)?;
    f()?;
    stream.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        f()?;
    }
    stream.synchronize()?;
    Ok(1000.0 * t0.elapsed().as_secs_f64() / reps as f64)
}

fn randn_bf16(shape: (usize, usize), scale: f64, device: &Device) -> Result<Tensor> {
    Ok(Tensor::randn(0f32, scale as f32, shape, device)?.to_dtype(DType::BF16)?)
}

static GRAPH_GIB_HIGH_WATER: std::sync::Mutex<f64> = std::sync::Mutex::new(0.0);

fn take_graph_gib() -> f64 {
    std::mem::take(&mut *GRAPH_GIB_HIGH_WATER.lock().unwrap())
}

fn fwd_bwd(x: &Tensor, f: impl Fn(&Tensor) -> Result<Tensor>) -> Result<()> {
    let before = free_gib();
    let v = Var::from_tensor(&x.to_dtype(DType::F32)?)?;
    let y = f(&v.as_tensor().to_dtype(DType::BF16)?)?;
    stream_of(x.device())?.synchronize()?;
    let mut hw = GRAPH_GIB_HIGH_WATER.lock().unwrap();
    *hw = hw.max(before - free_gib());
    drop(hw);
    let g = y.ones_like()?.to_dtype(DType::F32)?;
    let _ = y.to_dtype(DType::F32)?.mul(&g)?.sum_all()?.backward()?;
    Ok(())
}

pub fn bench_real_layer(base: &Path, gdn_layer: usize, tokens: usize, experts: usize, device: &Device) -> Result<serde_json::Value> {
    let cfg = Qwen4ExpConfig::from_hf_json_file(&base.join("config.json"))?;
    ensure!(!cfg.full_attention[gdn_layer], "layer {gdn_layer} is not a GDN layer");
    let attn_layer = (gdn_layer..cfg.num_hidden_layers).find(|&l| cfg.full_attention[l]).context("no attention layer at or after the GDN layer")?;
    let need = 2.0 + experts as f64 / cfg.num_experts as f64 * 4.5;
    ensure!(free_gib() > need, "bench needs {need:.1} GiB free beside whatever else holds the card, {:.1} free", free_gib());
    let weights = WeightLoader::open_dir(base, device)?;
    let stream = stream_of(device)?;
    let fp8: Fp8Runner = Arc::new(Mutex::new(nv_quant::fp8::Fp8GemmRunner::new(stream.clone())?));
    let (t, h) = (tokens, cfg.hidden_size);
    let mut out = serde_json::json!({"tokens": t, "gdn_layer": gdn_layer, "attn_layer": attn_layer, "experts_benched": experts,
                                     "num_experts": cfg.num_experts, "free_gib_at_start": free_gib()});
    let mut cfg_sub = cfg.clone();
    cfg_sub.num_experts = experts;
    let p = |l: usize| layer_prefix(l);
    {
        let t0 = std::time::Instant::now();
        let stacks = Arc::new(stacks_from_checkpoint(&weights, &cfg_sub, gdn_layer, experts, &stream)?);
        stream.synchronize()?;
        out["stack_build_ms"] = serde_json::json!(1000.0 * t0.elapsed().as_secs_f64());
        out["stack_bytes"] = serde_json::json!(stacks.device_bytes());
        let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream.clone())?));
        let grouped = load_grouped_experts_from_disk(&weights, &format!("{}.mlp", p(gdn_layer)), &cfg_sub, runner, &stream)?;
        let router_w = load_bf16(&weights, &format!("{}.mlp.gate.weight", p(gdn_layer)))?.narrow(0, 0, experts)?.contiguous()?;
        let si = cfg.shared_expert_intermediate_size;
        let m = format!("{}.mlp", p(gdn_layer));
        let moe = MoeLayer {
            router: Linear::new_no_pretranspose(router_w, None)?,
            grouped: Arc::new(grouped),
            shared_gate_proj: load_linear_side(&weights, &format!("{m}.shared_expert.gate_proj.weight"), si, h, Some(&fp8), device)?,
            shared_up_proj: load_linear_side(&weights, &format!("{m}.shared_expert.up_proj.weight"), si, h, Some(&fp8), device)?,
            shared_down_proj: load_linear_side(&weights, &format!("{m}.shared_expert.down_proj.weight"), h, si, Some(&fp8), device)?,
            shared_expert_gate: load_linear_bf16(&weights, &format!("{m}.shared_expert_gate.weight"), 1, h)?,
            expert_lora: std::sync::RwLock::new(None),
        };
        let x = randn_bf16((t, h), 1.0, device)?;
        out["moe_engine_fwd_ms"] = serde_json::json!(timed(device, 3, || {
            moe.forward(&x, &cfg_sub, None)?;
            Ok(())
        })?);
        let before = free_gib();
        out["moe_twin_fwd_bwd_ms"] = serde_json::json!(timed(device, 3, || fwd_bwd(&x, |xb| twin_moe_at(&cfg_sub, &moe, xb, stacks.clone(), None)))?);
        out["moe_twin_graph_gib"] = serde_json::json!(take_graph_gib());
        out["moe_free_gib_after"] = serde_json::json!(free_gib());
        out["moe_free_gib_before_twin"] = serde_json::json!(before);
    }
    {
        let mut la = LinearAttention::from_loader_bf16_checkpoint_projections_quantized_to_fp8_resident_halving_gdn_decode_weight_traffic(
            cfg.linear_attn_config(),
            &format!("{}.linear_attn", p(gdn_layer)),
            &weights,
            DType::BF16,
            device,
            &fp8,
        )?;
        la.set_output_gate_sigmoid_process_wide_because_the_gdn_kernels_read_one_constant_flag(true)?;
        let (lcfg, [qkv, z, a, b, o], [conv, a_log, dt, norm_w]) = la.train_twin_parts_sharing_the_resident_projections();
        let twin = TwinGdn { train: TrainGdn::new(lcfg, qkv, z, a, b, conv, a_log, dt, norm_w.clone(), o), norm_w };
        let state = la.new_fused_state(device)?;
        let x = randn_bf16((t, h), 1.0, device)?;
        out["gdn_engine_fwd_ms"] = serde_json::json!(timed(device, 3, || {
            let mut tmp = Some(state.clone());
            la.forward_with_state(&x.unsqueeze(0)?.contiguous()?, &mut tmp)?;
            Ok(())
        })?);
        let before = free_gib();
        out["gdn_twin_fwd_bwd_ms"] = serde_json::json!(timed(device, 2, || fwd_bwd(&x, |xb| twin_gdn_at(&cfg, &twin, xb)))?);
        out["gdn_twin_graph_gib"] = serde_json::json!(take_graph_gib());
        out["gdn_twin_free_gib_drop"] = serde_json::json!(before - free_gib());
    }
    {
        let attn = Attention::load(&weights, &format!("{}.self_attn", p(attn_layer)), &cfg, Some(&fp8), device)?;
        let rope = Rope::new(RopeConfig { head_dim: cfg.rotary_dim, max_seq_len: t.max(2), base: cfg.rope_theta, kind: RopeKind::Standard }, device)?;
        let x = randn_bf16((t, h), 1.0, device)?;
        let pos_i = Tensor::from_vec((0..t as i32).collect::<Vec<_>>(), (t,), device)?;
        let pos_u = Tensor::from_vec((0..t as u32).collect::<Vec<_>>(), (t,), device)?;
        let (nq, nkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        out["attn_engine_fwd_ms"] = serde_json::json!(timed(device, 3, || {
            let q_raw = attn.q_proj.forward(&x)?.reshape((t, nq, 2 * hd))?;
            let q = head_rmsnorm_bf16(&q_raw.narrow(2, 0, hd)?.contiguous()?, &attn.q_norm_w, cfg.rms_norm_eps)?;
            let k = head_rmsnorm_bf16(&attn.k_proj.forward(&x)?.reshape((t, nkv, hd))?, &attn.k_norm_w, cfg.rms_norm_eps)?;
            let v = attn.v_proj.forward(&x)?.reshape((1, t, nkv, hd))?.contiguous()?;
            let _ = attn.idx_qk_proj.forward(&x)?;
            let (q, k) = rope.apply_partial_oop_bf16_cuda_one_launch_replacing_slice_cast_rope_cast_concat(&q, &k, &pos_i, hd)?;
            let o = flash_attn(&q.unsqueeze(0)?.contiguous()?, &k.unsqueeze(0)?.contiguous()?, &v, &AttnConfig { num_heads: nq, num_kv_heads: nkv, head_dim: hd, softmax_scale: 1.0 / (hd as f32).sqrt(), causal: true })?;
            attn.o_proj.forward(&o.reshape((t, nq * hd))?)?;
            Ok(())
        })?);
        out["attn_twin_fwd_bwd_ms"] = serde_json::json!(timed(device, 3, || fwd_bwd(&x, |xb| twin_attention_at(&cfg, &rope, &attn, xb, &pos_u)))?);
        out["attn_twin_graph_gib"] = serde_json::json!(take_graph_gib());
    }
    {
        let gate = HcGate::load(&weights, &format!("{}.attn_hyper_connection", p(gdn_layer)), &cfg, true)?;
        let hc = randn_bf16((t, cfg.hc_hidden()), 1.0, device)?;
        let y = randn_bf16((t, h), 1.0, device)?;
        out["hc_engine_mix_combine_ms"] = serde_json::json!(timed(device, 3, || {
            let (_, inj) = gate.mix(&hc, &cfg)?;
            hc_combine(&hc, &y, inj.as_ref().unwrap(), &cfg)?;
            Ok(())
        })?);
        out["hc_twin_mix_combine_fwd_bwd_ms"] = serde_json::json!(timed(device, 3, || fwd_bwd(&hc, |hb| {
            let (m, inj) = gate.mix_eager(hb, &cfg)?;
            hc_combine_eager(hb, &(&y + &m)?, inj.as_ref().unwrap(), &cfg)
        }))?);
        out["hc_twin_graph_gib"] = serde_json::json!(take_graph_gib());
    }
    if free_gib() > 6.0 {
        let lm = load_linear_side(&weights, "lm_head.weight", cfg.vocab_size, h, Some(&fp8), device)?;
        let x = randn_bf16((t, h), 1.0, device)?;
        let targets = Tensor::from_vec((0..t as u32).map(|i| i * 7919 % cfg.vocab_size as u32).collect::<Vec<_>>(), (t,), device)?;
        out["lm_head_engine_fwd_ms"] = serde_json::json!(timed(device, 2, || {
            lm.forward(&x)?;
            Ok(())
        })?);
        let before = free_gib();
        out["lm_head_twin_ce_fwd_bwd_ms"] = serde_json::json!(timed(device, 2, || {
            let v = Var::from_tensor(&x.to_dtype(DType::F32)?)?;
            let logits = lm.forward(&v.as_tensor().to_dtype(DType::BF16)?)?.to_dtype(DType::F32)?;
            let _ = candle_nn::loss::cross_entropy(&logits, &targets)?.backward()?;
            Ok(())
        })?);
        out["lm_head_free_gib_drop"] = serde_json::json!(before - free_gib());
    }
    out["free_gib_at_end"] = serde_json::json!(free_gib());
    Ok(out)
}
