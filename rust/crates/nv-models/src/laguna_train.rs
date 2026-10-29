use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use nv_layers::linear::{Linear, LoraDeltaHook};
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_layers::weight_load::load_rmsnorm;
use nv_train::{
    ce_mean_over_unmasked_positions,
    eval_loss_scalar_is_the_sum_of_the_per_row_ce_means_in_dataset_row_order,
    grad_is_finite_by_one_scalar_readback_because_nan_and_inf_survive_a_sum,
    load_jsonl_token_dataset_strict as load_jsonl_token_dataset, save_peft_raw, LoraConfig,
    LoraTrainable, RawLora, TokenRow, TrainingLoraHook,
    JSONL_MAX_SEQ_IS_512_SHARED_BY_BOTH_ENDPOINT_ARMS_BECAUSE_THE_DEFAULT_QWEN_WINDOWS_GDN_GRAPH_AT_512_IS_ABOUT_14_GB as JSONL_MAX_SEQ,
};
use nv_weights::WeightLoader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::gemma4::LayerType;
use crate::laguna::{softplus_f32, yarn_inv_freq, LagunaConfig, LagunaGating, LagunaMoe};
use crate::train_runner::select_device;
use nv_layers::mlp::Mlp;
use nv_layers::train_attn::gqa_sdpa_recompute_with_group_serialization;

#[path = "laguna_train_routing.rs"]
mod routing;

#[path = "laguna_train_checkpoint.rs"]
mod checkpoint;

pub const LAGUNA_TRAINS_FULL_DEPTH_WITHOUT_WINDOWING_BECAUSE_THE_TRUNK_IS_2048_HIDDEN_BY_40_LAYERS:
    &str =
    "Laguna-XS-2.1 is hidden 2048 x 40 layers, against the 5120 x 64 qwen3.8 27B whose GDN \
     activation memory forced that family's grad_layers window; laguna's full-graph activations \
     at JSONL_MAX_SEQ 512 are an order of magnitude smaller per layer and the MoE expert bulk is \
     frozen, so this trainer takes the whole depth in one graph and exposes no depth-window knob";

pub const LAGUNA_LORA_SITES_ARE_THE_ATTENTION_PROJECTIONS_BECAUSE_MLP_MEANS_PACKED_EXPERTS: &str =
    "laguna LoRA targets are q, k, v, o and g, the per-layer attention projections that stay \
     dense bf16 even in the NVFP4 checkpoint; mlp targets are refused because 39 of 40 real \
     layers route through 256 nvfp4-packed experts, so an mlp adapter would train only layer 0's \
     lone dense mlp while the job record claims mlp coverage";

pub const DEFAULT_TRAIN_SEQUENCE_LENGTH: usize = JSONL_MAX_SEQ;
pub const MAX_TRAIN_SEQUENCE_LENGTH: usize = 32768;

pub fn validate_sequence_length(sequence_length: usize) -> Result<()> {
    ensure!(
        (2..=MAX_TRAIN_SEQUENCE_LENGTH).contains(&sequence_length),
        "Laguna max_sequence_length must be between 2 and {}",
        MAX_TRAIN_SEQUENCE_LENGTH
    );
    Ok(())
}

#[derive(Clone, Debug)]
pub struct LagunaTrainArgs {
    pub input_adapter: Option<PathBuf>,
    pub base: PathBuf,
    pub data: PathBuf,
    pub eval_data: Option<PathBuf>,
    pub out: PathBuf,
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<String>,
    pub steps: usize,
    pub lr: f64,
    pub seed: u64,
}

impl LagunaTrainArgs {
    pub fn default_targets() -> Vec<String> {
        ["q", "k", "v", "o", "g"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct LagunaTrainSummary {
    pub max_sequence_length: usize,
    pub losses: Vec<f32>,
    pub eval_loss_adapter: Option<f32>,
    pub eval_loss_base: Option<f32>,
    pub eval_ce_adapter: Vec<f32>,
    pub eval_ce_base: Vec<f32>,
    pub trainable_vars: usize,
    pub num_examples: usize,
    pub num_eval_examples: usize,
    pub base_dtype: String,
    pub packed_linears: usize,
    pub grouped_moe_copies: usize,
    pub device: String,
    pub config_layers: usize,
    pub moe_layers: usize,
    pub dense_mlp_layers: usize,
    pub all_grads_finite: bool,
    pub modules: Vec<String>,
    pub adapter_path: PathBuf,
    pub config_path: PathBuf,
}

enum TrainFfn {
    Dense {
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
    },
    Moe(LagunaMoe),
}

struct TrainLayer {
    kind: LayerType,
    num_heads: usize,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    g_proj: Option<Linear>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    ffn: TrainFfn,
}

pub struct LagunaTrainModel {
    config: LagunaConfig,
    embed_weight: Tensor,
    layers: Vec<TrainLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    sliding_rope: Rope,
    full_rope: Rope,
    full_attn_factor: f32,
    dtype: DType,
    device: Device,
    packed_linears: usize,
}

fn open_base_weights(path: &Path, device: &Device) -> Result<WeightLoader> {
    let st = path.join("model.safetensors");
    if st.is_file() {
        WeightLoader::open_file(&st, device)
            .with_context(|| format!("open weights {}", st.display()))
    } else {
        WeightLoader::open_dir(path, device)
            .with_context(|| format!("open sharded weights {}", path.display()))
    }
}

fn load_dense_linear(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Linear> {
    let module = name.strip_suffix(".weight").unwrap_or(name);
    ensure!(
        !["weight_scale", "weight_scale_2", "weight_global_scale"]
            .iter().any(|suffix| weights.has(&format!("{module}.{suffix}"))),
        "Laguna training cannot load scaled {module} as dense; supported quantized weights use native NVFP4 weight_packed"
    );
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != out_features || d[1] != in_features {
        bail!("linear {name}: expected [{out_features}, {in_features}], got {d:?}");
    }
    Linear::new(w, None)
}

impl LagunaTrainModel {
    pub fn load(base: &Path, device: &Device) -> Result<Self> {
        Self::load_with_sequence_length(base, device, DEFAULT_TRAIN_SEQUENCE_LENGTH)
    }

    pub fn load_with_sequence_length(
        base: &Path,
        device: &Device,
        sequence_length: usize,
    ) -> Result<Self> {
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let mut model =
            Self::load_with_dtype_and_sequence_length(base, device, dtype, sequence_length)?;
        model.build_grouped_expert_stacks_and_free_the_per_expert_copies();
        Ok(model)
    }

    pub const GROUPED_STACKS_REPLACE_THE_PER_EXPERT_LINEARS_AT_LOAD_SO_THE_EXPERT_WEIGHTS_ARE_RESIDENT_ONCE_NOT_TWICE: usize = 1;

    fn build_grouped_expert_stacks_and_free_the_per_expert_copies(&mut self) {
        #[cfg(feature = "cuda")]
        {
            let hidden = self.config.hidden_size;
            let device = self.device.clone();
            let mut built = 0usize;
            for layer in self.layers.iter_mut() {
                if let TrainFfn::Moe(moe) = &mut layer.ffn {
                    if routing::grouped_train_stacks(moe, hidden, &device).is_some() {
                        moe.experts = Vec::new();
                        built += 1;
                    }
                }
            }
            if built > 0 {
                let _ = Self::GROUPED_STACKS_REPLACE_THE_PER_EXPERT_LINEARS_AT_LOAD_SO_THE_EXPERT_WEIGHTS_ARE_RESIDENT_ONCE_NOT_TWICE;
                eprintln!(
                    "[moe_train] {built} MoE layers hold grouped stacks; per-expert Linears freed"
                );
            }
        }
    }

    pub fn load_with_dtype(base: &Path, device: &Device, dtype: DType) -> Result<Self> {
        Self::load_with_dtype_and_sequence_length(
            base,
            device,
            dtype,
            DEFAULT_TRAIN_SEQUENCE_LENGTH,
        )
    }

    fn load_with_dtype_and_sequence_length(
        base: &Path,
        device: &Device,
        dtype: DType,
        sequence_length: usize,
    ) -> Result<Self> {
        validate_sequence_length(sequence_length)?;
        let cfg_path = base.join("config.json");
        let config = LagunaConfig::from_hf_json_file(&cfg_path)
            .with_context(|| format!("read laguna config {}", cfg_path.display()))?;
        let weights = open_base_weights(base, device)?;
        let has_packed = weights
            .names()
            .iter()
            .any(|n| n.ends_with(".weight_packed"));
        if has_packed {
            ensure!(
                device.is_cuda() && cfg!(feature = "cuda") && dtype == DType::BF16,
                "Laguna nvfp4 weight_packed training requires the CUDA backend with BF16 activations; the frozen base stays packed"
            );
            let quant = nv_weights::QuantizationConfig::from_hf_json_file(&cfg_path)?;
            ensure!(
                quant.scheme == nv_weights::QuantScheme::Nvfp4,
                "Laguna weight_packed training requires an NVFP4 quantization_config"
            );
        }
        #[cfg(feature = "cuda")]
        let nvfp4_runner = if has_packed {
            let Device::Cuda(dev) = device else {
                unreachable!()
            };
            Some(Arc::new(std::sync::Mutex::new(
                nv_quant::nvfp4::Nvfp4GemmRunner::new(dev.cuda_stream())?,
            )))
        } else {
            None
        };
        let packed_linears = std::cell::Cell::new(0usize);
        let load_linear = |weights: &WeightLoader,
                           name: &str,
                           out_features: usize,
                           in_features: usize,
                           dtype| {
            #[cfg(feature = "cuda")]
            if let Some(runner) = &nvfp4_runner {
                let module = name.strip_suffix(".weight").unwrap_or(name);
                if weights.has(&format!("{module}.weight_packed")) {
                    ensure!(
                        in_features > 0 && out_features > 0 && in_features.is_multiple_of(16)
                            && weights.shape_of(&format!("{module}.weight_scale"))
                                == Some(vec![out_features, in_features / 16]),
                        "Laguna NVFP4 {module}: expected one block scale per 16 input values and output row"
                    );
                    let linear = nv_layers::moe::nvfp4_linear_from_disk_pub(
                        weights,
                        module,
                        out_features,
                        in_features,
                        runner.clone(),
                        device,
                    )
                    .with_context(|| format!("load resident Laguna NVFP4 {module}"))?;
                    packed_linears.set(packed_linears.get() + 1);
                    return Ok(linear);
                }
            }
            load_dense_linear(weights, name, out_features, in_features, dtype)
        };

        let embed_weight = weights
            .get("model.embed_tokens.weight", dtype)
            .context("load model.embed_tokens.weight")?;
        let ed = embed_weight.dims();
        if ed.len() != 2 || ed[0] != config.vocab_size || ed[1] != config.hidden_size {
            bail!(
                "laguna embed: expected [{}, {}], got {:?}",
                config.vocab_size,
                config.hidden_size,
                ed
            );
        }

        let hidden = config.hidden_size;
        let head_dim = config.head_dim;
        let n_kv = config.num_key_value_heads;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{idx}");
            let n_q = config.num_heads_for_layer(idx);
            let g_proj = match config.gating {
                LagunaGating::None => None,
                LagunaGating::PerHead => Some(load_linear(
                    &weights,
                    &format!("{prefix}.self_attn.g_proj.weight"),
                    n_q,
                    hidden,
                    dtype,
                )?),
                LagunaGating::PerElement => Some(load_linear(
                    &weights,
                    &format!("{prefix}.self_attn.g_proj.weight"),
                    n_q * head_dim,
                    hidden,
                    dtype,
                )?),
            };
            let ffn = if config.is_moe_layer(idx) {
                let gate = load_linear(
                    &weights,
                    &format!("{prefix}.mlp.gate.weight"),
                    config.num_experts,
                    hidden,
                    dtype,
                )?;
                let bias_name = format!("{prefix}.mlp.experts.e_score_correction_bias");
                let selection_bias = if weights.has(&bias_name) {
                    weights
                        .get(&bias_name, DType::F32)?
                        .reshape((1usize, config.num_experts))?
                } else {
                    Tensor::zeros((1usize, config.num_experts), DType::F32, device)?
                };
                let expert_mlp = |eprefix: &str| -> Result<Mlp> {
                    Mlp::new(
                        load_linear(
                            &weights,
                            &format!("{eprefix}.gate_proj.weight"),
                            config.moe_intermediate_size,
                            hidden,
                            dtype,
                        )?,
                        load_linear(
                            &weights,
                            &format!("{eprefix}.up_proj.weight"),
                            config.moe_intermediate_size,
                            hidden,
                            dtype,
                        )?,
                        load_linear(
                            &weights,
                            &format!("{eprefix}.down_proj.weight"),
                            hidden,
                            config.moe_intermediate_size,
                            dtype,
                        )?,
                    )
                };
                let mut experts = Vec::with_capacity(config.num_experts);
                for e in 0..config.num_experts {
                    experts.push(expert_mlp(&format!("{prefix}.mlp.experts.{e}"))?);
                }
                let sprefix = format!("{prefix}.mlp.shared_expert");
                let shared_expert = Mlp::new(
                    load_linear(
                        &weights,
                        &format!("{sprefix}.gate_proj.weight"),
                        config.shared_expert_intermediate_size,
                        hidden,
                        dtype,
                    )?,
                    load_linear(
                        &weights,
                        &format!("{sprefix}.up_proj.weight"),
                        config.shared_expert_intermediate_size,
                        hidden,
                        dtype,
                    )?,
                    load_linear(
                        &weights,
                        &format!("{sprefix}.down_proj.weight"),
                        hidden,
                        config.shared_expert_intermediate_size,
                        dtype,
                    )?,
                )?;
                TrainFfn::Moe(LagunaMoe {
                    num_experts: config.num_experts,
                    top_k: config.num_experts_per_tok,
                    norm_topk: config.norm_topk_prob,
                    routed_scaling: config.moe_routed_scaling_factor,
                    softcap: config.moe_router_logit_softcapping,
                    gate,
                    selection_bias,
                    experts,
                    shared_expert,
                    #[cfg(feature = "cuda")]
                    grouped: std::sync::Mutex::new(None),
                    #[cfg(feature = "cuda")]
                    train_stacks: std::sync::Mutex::new(None),
                })
            } else {
                TrainFfn::Dense {
                    gate_proj: load_linear(
                        &weights,
                        &format!("{prefix}.mlp.gate_proj.weight"),
                        config.intermediate_size,
                        hidden,
                        dtype,
                    )?,
                    up_proj: load_linear(
                        &weights,
                        &format!("{prefix}.mlp.up_proj.weight"),
                        config.intermediate_size,
                        hidden,
                        dtype,
                    )?,
                    down_proj: load_linear(
                        &weights,
                        &format!("{prefix}.mlp.down_proj.weight"),
                        hidden,
                        config.intermediate_size,
                        dtype,
                    )?,
                }
            };
            layers.push(TrainLayer {
                kind: config.layer_kind(idx),
                num_heads: n_q,
                input_layernorm: load_rmsnorm(
                    &weights,
                    &format!("{prefix}.input_layernorm.weight"),
                    hidden,
                    config.rms_norm_eps,
                    dtype,
                )?,
                post_attention_layernorm: load_rmsnorm(
                    &weights,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    hidden,
                    config.rms_norm_eps,
                    dtype,
                )?,
                q_proj: load_linear(
                    &weights,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    n_q * head_dim,
                    hidden,
                    dtype,
                )?,
                k_proj: load_linear(
                    &weights,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    n_kv * head_dim,
                    hidden,
                    dtype,
                )?,
                v_proj: load_linear(
                    &weights,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    n_kv * head_dim,
                    hidden,
                    dtype,
                )?,
                o_proj: load_linear(
                    &weights,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    hidden,
                    n_q * head_dim,
                    dtype,
                )?,
                g_proj,
                q_norm: load_rmsnorm(
                    &weights,
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    head_dim,
                    config.rms_norm_eps,
                    dtype,
                )?,
                k_norm: load_rmsnorm(
                    &weights,
                    &format!("{prefix}.self_attn.k_norm.weight"),
                    head_dim,
                    config.rms_norm_eps,
                    dtype,
                )?,
                ffn,
            });
        }

        let final_norm = load_rmsnorm(
            &weights,
            "model.norm.weight",
            hidden,
            config.rms_norm_eps,
            dtype,
        )?;
        let lm_head_weight = if config.tie_word_embeddings {
            embed_weight.clone()
        } else {
            weights
                .get("lm_head.weight", dtype)
                .context("load lm_head.weight")?
        };
        let lm_head = Linear::new_no_pretranspose(lm_head_weight, None)?;

        let rope_len = config.max_position_embeddings.min(sequence_length * 2);
        let full_dim = config.rotary_dim_full();
        let full_rope = Rope::from_inv_freq(
            RopeConfig {
                head_dim: full_dim,
                max_seq_len: rope_len,
                base: config.full_rope_params().rope_theta,
                kind: RopeKind::Yarn,
            },
            &yarn_inv_freq(full_dim, config.full_rope_params()),
            device,
        )?;
        let sliding_dim = config.rotary_dim_sliding();
        let sliding_rope = Rope::from_inv_freq(
            RopeConfig {
                head_dim: sliding_dim,
                max_seq_len: rope_len,
                base: config.sliding_rope_params().rope_theta,
                kind: RopeKind::Standard,
            },
            &yarn_inv_freq(sliding_dim, config.sliding_rope_params()),
            device,
        )?;
        let full_attn_factor = config.full_attention_factor();

        Ok(Self {
            config,
            embed_weight,
            layers,
            final_norm,
            lm_head,
            sliding_rope,
            full_rope,
            full_attn_factor,
            dtype,
            device: device.clone(),
            packed_linears: packed_linears.get(),
        })
    }

    pub fn config(&self) -> &LagunaConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn packed_linears(&self) -> usize {
        self.packed_linears
    }

    pub fn grouped_moe_copies(&self) -> Result<usize> {
        #[cfg(feature = "cuda")]
        {
            let mut copies = 0;
            for layer in &self.layers {
                if let TrainFfn::Moe(moe) = &layer.ffn {
                    copies += usize::from(
                        moe.grouped
                            .lock()
                            .map_err(|_| anyhow::anyhow!("Laguna grouped weights lock poisoned"))?
                            .as_ref()
                            .is_some_and(Option::is_some),
                    );
                }
            }
            Ok(copies)
        }
        #[cfg(not(feature = "cuda"))]
        Ok(0)
    }

    pub fn moe_and_dense_mlp_layer_counts(&self) -> (usize, usize) {
        let moe = self
            .layers
            .iter()
            .filter(|l| matches!(l.ffn, TrainFfn::Moe(_)))
            .count();
        (moe, self.layers.len() - moe)
    }

    pub fn moe_of_the_first_sparse_layer_so_tests_can_gate_its_backward(
        &self,
    ) -> Option<&LagunaMoe> {
        self.layers.iter().find_map(|l| match &l.ffn {
            TrainFfn::Moe(moe) => Some(moe),
            TrainFfn::Dense { .. } => None,
        })
    }

    fn embed_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            bail!("LagunaTrainModel.forward_logits: tokens must be [1, seq], got {dims:?}");
        }
        let seq = dims[1];
        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        let hidden = self
            .embed_weight
            .detach()
            .index_select(&tokens_flat, 0)?
            .reshape((1usize, seq, self.config.hidden_size))?
            .to_dtype(self.dtype)?;
        Ok(hidden)
    }

    fn forward_layer(
        &self,
        layer: &TrainLayer,
        hidden: &Tensor,
        positions: &Tensor,
        seq: usize,
    ) -> Result<Tensor> {
        let dev = hidden.device().clone();
        let after_attn = nv_train::train_prof::phase(&dev, "fwd_attention_layer", || {
            let normed = layer.input_layernorm.forward_candle(&hidden)?;
            let attn_out = self.attention_forward(layer, &normed, positions, seq)?;
            Ok(hidden.add(&attn_out.to_dtype(hidden.dtype())?)?)
        })?;
        let ffn_phase = match &layer.ffn {
            TrainFfn::Dense { .. } => "fwd_dense_mlp_layer",
            TrainFfn::Moe(_) => "fwd_moe_layer",
        };
        let out = nv_train::train_prof::phase(&dev, ffn_phase, || {
            let normed_mlp = layer.post_attention_layernorm.forward_candle(&after_attn)?;
            let ffn_out = match &layer.ffn {
                TrainFfn::Dense {
                    gate_proj,
                    up_proj,
                    down_proj,
                } => {
                    let gate = gate_proj.forward(&normed_mlp)?;
                    let up = up_proj.forward(&normed_mlp)?;
                    down_proj.forward(&candle_nn::ops::silu(&gate)?.mul(&up)?)?
                }
                TrainFfn::Moe(moe) => {
                    let flat = normed_mlp
                        .reshape((seq, self.config.hidden_size))?
                        .contiguous()?;
                    routing::forward(moe, &flat)?.reshape((1usize, seq, self.config.hidden_size))?
                }
            };
            Ok(after_attn.add(&ffn_out.to_dtype(after_attn.dtype())?)?)
        })?;
        Ok(out)
    }

    fn final_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let dev = hidden.device().clone();
        nv_train::train_prof::phase(&dev, "fwd_final_norm_and_lm_head", || {
            let normed = self.final_norm.forward_candle(&hidden)?;
            Ok(self.lm_head.forward(&normed)?.to_dtype(DType::F32)?)
        })
    }

    pub fn forward_logits(&self, tokens: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let mut hidden = self.embed_tokens(tokens)?;
        let seq = tokens.dim(1)?;
        for layer in &self.layers {
            hidden = self.forward_layer(layer, &hidden, positions, seq)?;
        }
        self.final_logits(&hidden)
    }

    fn attention_forward(
        &self,
        layer: &TrainLayer,
        x: &Tensor,
        positions: &Tensor,
        seq: usize,
    ) -> Result<Tensor> {
        let head_dim = self.config.head_dim;
        let n_q = layer.num_heads;
        let n_kv = self.config.num_key_value_heads;
        #[cfg(feature = "cuda")]
        let proj_span = fwd_span(x, "fwd_attn_qkv_projections_with_lora");

        let q = layer
            .q_proj
            .forward(x)?
            .reshape((1usize, seq, n_q, head_dim))?;
        let q_normed = layer.q_norm.forward_candle(&q)?;
        let k = layer
            .k_proj
            .forward(x)?
            .reshape((1usize, seq, n_kv, head_dim))?;
        let k_normed = layer.k_norm.forward_candle(&k)?;
        let v = layer
            .v_proj
            .forward(x)?
            .reshape((1usize, seq, n_kv, head_dim))?;
        #[cfg(feature = "cuda")]
        drop(proj_span);
        #[cfg(feature = "cuda")]
        let rope_span = fwd_span(x, "fwd_attn_norms_rope_casts");

        let (rope, rotary_dim, rot_scale) = match layer.kind {
            LayerType::SlidingAttention => {
                (&self.sliding_rope, self.config.rotary_dim_sliding(), 1.0f32)
            }
            LayerType::FullAttention => (
                &self.full_rope,
                self.config.rotary_dim_full(),
                self.full_attn_factor,
            ),
        };
        let (q_rot, k_rot) = partial_rope_scaled_matching_serving_apply_partial_rope_scaled(
            &q_normed, &k_normed, rope, positions, rotary_dim, head_dim, rot_scale,
        )?;
        let q_rot = q_rot.to_dtype(self.dtype)?.contiguous()?;
        let k_rot = k_rot.to_dtype(self.dtype)?.contiguous()?;
        let v = v.contiguous()?;

        #[cfg(feature = "cuda")]
        drop(rope_span);
        let window = match layer.kind {
            LayerType::SlidingAttention => Some(self.config.sliding_window),
            LayerType::FullAttention => None,
        };
        let _ = nv_layers::train_attn::GQA_SDPA_RECOMPUTE_KEEPS_ONLY_Q_K_V_AND_O_ON_THE_GRAPH_AND_REBUILDS_THE_PROBABILITIES_IN_THE_BACKWARD;
        let q_rot = if attention_output_stays_f32_like_the_expanded_head_path_did() {
            q_rot.to_dtype(DType::F32)?
        } else {
            q_rot
        };
        let attn_out = gqa_sdpa_recompute_with_group_serialization(
            &q_rot,
            &k_rot,
            &v,
            1,
            n_q,
            n_kv,
            head_dim,
            seq,
            window,
            checkpoint::enabled(),
        )?;

        #[cfg(feature = "cuda")]
        let gate_span = fwd_span(x, "fwd_attn_gate_softplus_and_o_proj");
        let gated = match (&layer.g_proj, self.config.gating) {
            (Some(g_proj), LagunaGating::PerHead) => {
                let g = softplus_f32(&g_proj.forward(x)?.to_dtype(DType::F32)?)?;
                let g = g
                    .reshape((1usize, seq, n_q, 1usize))?
                    .to_dtype(attn_out.dtype())?;
                attn_out.broadcast_mul(&g)?
            }
            (Some(g_proj), LagunaGating::PerElement) => {
                let g = softplus_f32(&g_proj.forward(x)?.to_dtype(DType::F32)?)?;
                let g = g
                    .reshape((1usize, seq, n_q, head_dim))?
                    .to_dtype(attn_out.dtype())?;
                attn_out.broadcast_mul(&g)?
            }
            _ => attn_out,
        };

        let attn_flat = gated
            .reshape((1usize, seq, n_q * head_dim))?
            .to_dtype(self.dtype)?;
        let out = layer.o_proj.forward(&attn_flat);
        #[cfg(feature = "cuda")]
        drop(gate_span);
        out
    }
}

pub fn partial_rope_scaled_matching_serving_apply_partial_rope_scaled(
    q: &Tensor,
    k: &Tensor,
    rope: &Rope,
    positions: &Tensor,
    rotary_dim: usize,
    head_dim: usize,
    rot_scale: f32,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if nv_layers::rope::train_rope_op_serves(q) && q.dtype() == k.dtype() {
        let q_out = rope.apply_train_partial_scaled(
            q,
            positions,
            rotary_dim.min(head_dim),
            head_dim,
            rot_scale,
        )?;
        let k_out = rope.apply_train_partial_scaled(
            k,
            positions,
            rotary_dim.min(head_dim),
            head_dim,
            rot_scale,
        )?;
        return Ok((q_out, k_out));
    }
    let scale_rot = |t: Tensor| -> Result<Tensor> {
        if (rot_scale - 1.0).abs() < f32::EPSILON {
            Ok(t)
        } else {
            Ok(t.to_dtype(DType::F32)?.affine(rot_scale as f64, 0.0)?)
        }
    };
    if rotary_dim >= head_dim {
        let (q_r, k_r) = rope.apply_candle(
            &q.to_dtype(DType::F32)?,
            &k.to_dtype(DType::F32)?,
            positions,
        )?;
        return Ok((scale_rot(q_r)?, scale_rot(k_r)?));
    }
    let last = q.dims().len() - 1;
    let q_rot = q.narrow(last, 0, rotary_dim)?.contiguous()?;
    let q_pass = q
        .narrow(last, rotary_dim, head_dim - rotary_dim)?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let k_last = k.dims().len() - 1;
    let k_rot = k.narrow(k_last, 0, rotary_dim)?.contiguous()?;
    let k_pass = k
        .narrow(k_last, rotary_dim, head_dim - rotary_dim)?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let (q_r, k_r) = rope.apply_candle(
        &q_rot.to_dtype(DType::F32)?,
        &k_rot.to_dtype(DType::F32)?,
        positions,
    )?;
    let q_out = Tensor::cat(&[&scale_rot(q_r)?, &q_pass], last)?;
    let k_out = Tensor::cat(&[&scale_rot(k_r)?, &k_pass], k_last)?;
    Ok((q_out, k_out))
}

struct LoraSite<'a> {
    module: String,
    lin: &'a Linear,
    seed: u64,
}

fn want(targets: &[String], leaf: &str) -> bool {
    targets.iter().any(|t| t == leaf)
}

fn build_sites<'a>(
    model: &'a LagunaTrainModel,
    targets: &[String],
    seed0: u64,
) -> Result<Vec<LoraSite<'a>>> {
    const SUPPORTED: [&str; 5] = ["q", "k", "v", "o", "g"];
    for t in targets {
        if !SUPPORTED.contains(&t.as_str()) {
            bail!(
                "unsupported laguna target {t:?}: {}",
                LAGUNA_LORA_SITES_ARE_THE_ATTENTION_PROJECTIONS_BECAUSE_MLP_MEANS_PACKED_EXPERTS
            );
        }
    }
    if want(targets, "g") && model.config.gating == LagunaGating::None {
        bail!(
            "target \"g\" requested but this laguna config has gating disabled and no g_proj \
             weights; request target [q, k, v, o] instead"
        );
    }
    let mut out = Vec::new();
    let mut si = seed0;
    for (li, layer) in model.layers.iter().enumerate() {
        let p = format!("model.layers.{li}.self_attn");
        let mut push = |leaf: &str, lin: &'a Linear| {
            out.push(LoraSite {
                module: format!("{p}.{leaf}_proj"),
                lin,
                seed: si,
            });
            si += 1;
        };
        if want(targets, "q") {
            push("q", &layer.q_proj);
        }
        if want(targets, "k") {
            push("k", &layer.k_proj);
        }
        if want(targets, "v") {
            push("v", &layer.v_proj);
        }
        if want(targets, "o") {
            push("o", &layer.o_proj);
        }
        if want(targets, "g") {
            if let Some(g) = &layer.g_proj {
                push("g", g);
            }
        }
    }
    if out.is_empty() {
        bail!("no trainable laguna targets selected (--target was {targets:?})");
    }
    Ok(out)
}

fn solo_row_ce(model: &LagunaTrainModel, row: &TokenRow) -> Result<Tensor> {
    let dev = model.device();
    let seq = row.ids.len();
    let tokens = Tensor::from_vec(row.ids.clone(), (1usize, seq), dev)?;
    let positions = Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), seq, dev)?;
    row_loss_from_logits(&model.forward_logits(&tokens, &positions)?, row)
}

fn row_loss_from_logits(logits: &Tensor, row: &TokenRow) -> Result<Tensor> {
    let dev = logits.device();
    let seq = row.ids.len();
    let logits = logits.squeeze(0)?;
    let inp = logits.narrow(0, 0, seq - 1)?.contiguous()?;
    let tgt = Tensor::from_vec(row.ids[1..].to_vec(), seq - 1, dev)?;
    ce_mean_over_unmasked_positions(
        &inp,
        &tgt,
        &row.target_mask_where_predicting_a_prompt_token_weighs_zero(),
    )
}

fn per_row_eval_ce(model: &LagunaTrainModel, rows: &[TokenRow]) -> Result<Vec<f32>> {
    rows.iter()
        .map(|row| {
            if checkpoint::enabled() {
                checkpoint::eval_row_ce(model, row)
            } else {
                Ok(solo_row_ce(model, row)?.to_scalar::<f32>()?)
            }
        })
        .collect()
}

fn attention_output_stays_f32_like_the_expanded_head_path_did() -> bool {
    nv_config::env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(
        "NV_TRAIN_ATTN_OUT_F32",
    )
}

pub const ONE_ROW_ON_THE_GRAPH_AT_A_TIME_BECAUSE_SUMMING_EVERY_ROW_LOSS_BEFORE_ONE_BACKWARD_HELD_EVERY_ROW_GRAPH_AT_ONCE: usize = 1;

pub fn run(args: &LagunaTrainArgs) -> Result<LagunaTrainSummary> {
    run_with_sequence_length(args, DEFAULT_TRAIN_SEQUENCE_LENGTH)
}

pub fn run_with_sequence_length(
    args: &LagunaTrainArgs,
    sequence_length: usize,
) -> Result<LagunaTrainSummary> {
    validate_sequence_length(sequence_length)?;
    let device = if !nv_config::env_arm_present_at_all_reread_every_call("NV_TRAIN_DEVICE")
        && open_base_weights(&args.base, &Device::Cpu)?
            .names()
            .iter()
            .any(|n| n.ends_with(".weight_packed"))
    {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0).context("open CUDA device for resident Laguna NVFP4 training")?
        }
        #[cfg(not(feature = "cuda"))]
        bail!("Laguna nvfp4 weight_packed training requires a build with the CUDA feature")
    } else {
        select_device()?
    };
    let device_label = if device.is_cuda() { "cuda" } else { "cpu" };
    let input = match &args.input_adapter {
        Some(path) => {
            validate_input_adapter(args, path)?;
            Some(WeightLoader::open_file(
                path.join("adapter_model.safetensors"),
                &device,
            )?)
        }
        None => None,
    };
    let model = LagunaTrainModel::load_with_sequence_length(&args.base, &device, sequence_length)?;
    let sequence_length = sequence_length.min(model.config.max_position_embeddings);
    let (moe_layers, dense_mlp_layers) = model.moe_and_dense_mlp_layer_counts();
    eprintln!(
        "LAGUNA_BASE 1  layers {}  hidden {}  vocab {}  moe_layers {moe_layers}  device \
         {device_label}  base_dtype {:?}  packed_linears {}",
        model.config.num_hidden_layers,
        model.config.hidden_size,
        model.config.vocab_size,
        model.dtype(),
        model.packed_linears()
    );

    let sites = build_sites(&model, &args.targets, args.seed)?;
    if let Some(weights) = &input {
        let expected: std::collections::BTreeSet<String> = sites
            .iter()
            .flat_map(|s| {
                [
                    format!("base_model.model.{}.lora_A.weight", s.module),
                    format!("base_model.model.{}.lora_B.weight", s.module),
                ]
            })
            .collect();
        ensure!(
            weights
                .names()
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
                == expected,
            "input adapter target modules differ from training targets"
        );
    }
    let cfg = LoraConfig {
        r: args.rank,
        alpha: args.alpha,
        dropout: 0.0,
    };
    let mut loras: Vec<Arc<LoraTrainable>> = Vec::new();
    let mut vars: Vec<Var> = Vec::new();
    for s in &sites {
        let lora = Arc::new(LoraTrainable::from_features(
            s.lin.out_features(),
            s.lin.in_features(),
            cfg,
            s.seed,
            &device,
        )?);
        if let Some(weights) = &input {
            for (var, suffix) in lora.trainable_vars().iter().zip(["A", "B"]) {
                let value = weights.get(
                    &format!("base_model.model.{}.lora_{suffix}.weight", s.module),
                    DType::F32,
                )?;
                ensure!(
                    value.dims() == var.dims(),
                    "input adapter tensor shape mismatch"
                );
                ensure!(
                    grad_is_finite_by_one_scalar_readback_because_nan_and_inf_survive_a_sum(
                        &value
                    )?,
                    "input adapter has non-finite weights"
                );
                var.set(&value)?;
            }
        }
        vars.extend(lora.trainable_vars());
        let hook: Arc<dyn LoraDeltaHook> = Arc::new(TrainingLoraHook::new(lora.clone(), true));
        s.lin.attach_lora(hook)?;
        loras.push(lora);
    }

    let dataset = load_jsonl_token_dataset(
        &args.data,
        &args.base,
        model.config.vocab_size,
        sequence_length,
    )?;

    let mut losses = Vec::with_capacity(args.steps);
    let all_grads_finite = true;
    if args.steps > 0 {
        let mut opt = AdamW::new(
            vars.clone(),
            ParamsAdamW {
                lr: args.lr,
                weight_decay: 0.0,
                ..Default::default()
            },
        )?;
        for step in 0..args.steps {
            let dev = model.device().clone();
            let _ = ONE_ROW_ON_THE_GRAPH_AT_A_TIME_BECAUSE_SUMMING_EVERY_ROW_LOSS_BEFORE_ONE_BACKWARD_HELD_EVERY_ROW_GRAPH_AT_ONCE;
            let mut lv = 0f32;
            let mut grads: Option<candle_core::backprop::GradStore> = None;
            ensure!(!dataset.is_empty(), "empty laguna training batch");
            for row in &dataset {
                let (row_loss, row_grads) = if checkpoint::enabled() {
                    checkpoint::row_gradients(&model, row, &vars)?
                } else {
                    let loss = nv_train::train_prof::phase(&dev, "forward_row", || {
                        solo_row_ce(&model, row)
                    })?;
                    let value = loss.to_scalar::<f32>()?;
                    let gradients = nv_train::train_prof::phase(&dev, "backward_row", || {
                        loss.backward().map_err(Into::into)
                    })?;
                    (value, gradients)
                };
                lv += row_loss;
                grads = Some(match grads {
                    None => row_grads,
                    Some(mut acc) => {
                        for v in vars.iter() {
                            if let Some(g) = row_grads.get(v) {
                                let summed = match acc.get(v) {
                                    Some(a) => (a + g)?,
                                    None => g.clone(),
                                };
                                acc.insert(v, summed);
                            }
                        }
                        acc
                    }
                });
            }
            ensure!(
                lv.is_finite(),
                "Laguna training step {step}: non-finite loss"
            );
            let grads = grads.context("empty laguna training batch")?;
            nv_train::train_prof::phase(&dev, "grad_finite_readbacks", || {
                for (index, v) in vars.iter().enumerate() {
                    let g = grads.get(v).with_context(|| {
                        format!("Laguna training step {step}: missing gradient for adapter variable {index}")
                    })?;
                    ensure!(
                        grad_is_finite_by_one_scalar_readback_because_nan_and_inf_survive_a_sum(g)?,
                        "Laguna training step {step}: non-finite gradient for adapter variable {index}"
                    );
                }
                Ok(())
            })?;
            nv_train::train_prof::phase(&dev, "optimizer_step", || {
                opt.step(&grads).map_err(Into::into)
            })?;
            losses.push(lv);
            if step < 3 || step % 10 == 0 || step == args.steps - 1 {
                eprintln!("step {step:4}  loss = {lv:.6e}");
            }
        }
    }

    let (eval_ce_adapter, eval_ce_base) = match &args.eval_data {
        Some(path) => {
            let eval_set = load_jsonl_token_dataset(
                path,
                &args.base,
                model.config.vocab_size,
                sequence_length,
            )?;
            let with_adapter = per_row_eval_ce(&model, &eval_set)?;
            for s in &sites {
                s.lin.detach_lora();
            }
            let base_only = per_row_eval_ce(&model, &eval_set)?;
            for (s, lora) in sites.iter().zip(loras.iter()) {
                let hook: Arc<dyn LoraDeltaHook> =
                    Arc::new(TrainingLoraHook::new(lora.clone(), true));
                s.lin.attach_lora(hook)?;
            }
            eprintln!(
                "eval  loss_adapter = {:.6e}  loss_base = {:.6e}  examples {}",
                with_adapter.iter().sum::<f32>(),
                base_only.iter().sum::<f32>(),
                eval_set.len()
            );
            (with_adapter, base_only)
        }
        None => (Vec::new(), Vec::new()),
    };
    let eval_loss_adapter =
        eval_loss_scalar_is_the_sum_of_the_per_row_ce_means_in_dataset_row_order(&eval_ce_adapter);
    let eval_loss_base =
        eval_loss_scalar_is_the_sum_of_the_per_row_ce_means_in_dataset_row_order(&eval_ce_base);

    let entries: Vec<RawLora> = sites
        .iter()
        .zip(loras.iter())
        .map(|(s, lora)| RawLora {
            module: s.module.clone(),
            a: lora.a_tensor().clone(),
            b: lora.b_tensor().clone(),
        })
        .collect();
    save_peft_raw(
        &args.out,
        &entries,
        args.rank,
        args.alpha,
        0.0,
        &args.base.to_string_lossy(),
    )?;

    let modules: Vec<String> = entries.iter().map(|e| e.module.clone()).collect();
    let num_eval_examples = eval_ce_base.len();
    Ok(LagunaTrainSummary {
        max_sequence_length: sequence_length,
        losses,
        eval_loss_adapter,
        eval_loss_base,
        eval_ce_adapter,
        eval_ce_base,
        trainable_vars: vars.len(),
        num_examples: dataset.len(),
        num_eval_examples,
        base_dtype: if model.packed_linears() > 0 {
            format!("NVFP4+{:?}", model.dtype())
        } else {
            format!("{:?}", model.dtype())
        },
        packed_linears: model.packed_linears(),
        grouped_moe_copies: model.grouped_moe_copies()?,
        device: device_label.to_string(),
        config_layers: model.config.num_hidden_layers,
        moe_layers,
        dense_mlp_layers,
        all_grads_finite,
        modules,
        adapter_path: args.out.join("adapter_model.safetensors"),
        config_path: args.out.join("adapter_config.json"),
    })
}

pub fn tiny_laguna_config_json() -> String {
    serde_json::json!({
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "vocab_size": 96,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "norm_topk_prob": true,
        "mlp_only_layers": [],
        "decoder_sparse_step": 1,
        "tie_word_embeddings": false,
        "gating": "per-head",
        "sliding_window": 8,
        "moe_routed_scaling_factor": 2.5,
        "moe_router_logit_softcapping": 5.0,
        "rope_parameters": {
            "full_attention": {
                "rope_theta": 500000.0,
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 64,
                "beta_slow": 1.0,
                "beta_fast": 64.0,
                "attention_factor": 1.3465735902799727,
                "partial_rotary_factor": 0.5
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0
            }
        },
        "layer_types": [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention"
        ],
        "mlp_layer_types": ["dense", "sparse", "sparse", "dense"],
        "num_attention_heads_per_layer": [4, 8, 8, 4]
    })
    .to_string()
}

pub struct LagunaFixtureLcg(pub u64);

impl LagunaFixtureLcg {
    pub fn next_f32(&mut self, std: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        u * std * 1.732
    }

    pub fn tensor(&mut self, shape: &[usize], std: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| self.next_f32(std)).collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }
}

pub fn write_tiny_laguna_checkpoint(dir: &Path) -> LagunaConfig {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let cfg_json = tiny_laguna_config_json();
    std::fs::write(dir.join("config.json"), &cfg_json).unwrap();
    let cfg = LagunaConfig::from_hf_json_str(&cfg_json).unwrap();

    let mut rng = LagunaFixtureLcg(0x001A_600A_5EED);
    let mut t: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let n_kv = cfg.num_key_value_heads;
    t.insert(
        "model.embed_tokens.weight".into(),
        rng.tensor(&[cfg.vocab_size, h], 0.05),
    );
    t.insert(
        "model.norm.weight".into(),
        rng.tensor(&[h], 0.05).affine(1.0, 1.0).unwrap(),
    );
    t.insert(
        "lm_head.weight".into(),
        rng.tensor(&[cfg.vocab_size, h], 0.05),
    );
    for li in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{li}");
        let n_q = cfg.num_heads_for_layer(li);
        let mut norm = |name: String, dim: usize| {
            t.insert(name, rng.tensor(&[dim], 0.05).affine(1.0, 1.0).unwrap());
        };
        norm(format!("{p}.input_layernorm.weight"), h);
        norm(format!("{p}.post_attention_layernorm.weight"), h);
        norm(format!("{p}.self_attn.q_norm.weight"), hd);
        norm(format!("{p}.self_attn.k_norm.weight"), hd);
        t.insert(
            format!("{p}.self_attn.q_proj.weight"),
            rng.tensor(&[n_q * hd, h], 0.08),
        );
        t.insert(
            format!("{p}.self_attn.k_proj.weight"),
            rng.tensor(&[n_kv * hd, h], 0.08),
        );
        t.insert(
            format!("{p}.self_attn.v_proj.weight"),
            rng.tensor(&[n_kv * hd, h], 0.08),
        );
        t.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rng.tensor(&[h, n_q * hd], 0.08),
        );
        t.insert(
            format!("{p}.self_attn.g_proj.weight"),
            rng.tensor(&[n_q, h], 0.08),
        );
        if cfg.is_moe_layer(li) {
            t.insert(
                format!("{p}.mlp.gate.weight"),
                rng.tensor(&[cfg.num_experts, h], 0.08),
            );
            t.insert(
                format!("{p}.mlp.experts.e_score_correction_bias"),
                rng.tensor(&[cfg.num_experts], 0.02),
            );
            for e in 0..cfg.num_experts {
                let ep = format!("{p}.mlp.experts.{e}");
                t.insert(
                    format!("{ep}.gate_proj.weight"),
                    rng.tensor(&[cfg.moe_intermediate_size, h], 0.08),
                );
                t.insert(
                    format!("{ep}.up_proj.weight"),
                    rng.tensor(&[cfg.moe_intermediate_size, h], 0.08),
                );
                t.insert(
                    format!("{ep}.down_proj.weight"),
                    rng.tensor(&[h, cfg.moe_intermediate_size], 0.08),
                );
            }
            let sp = format!("{p}.mlp.shared_expert");
            t.insert(
                format!("{sp}.gate_proj.weight"),
                rng.tensor(&[cfg.shared_expert_intermediate_size, h], 0.08),
            );
            t.insert(
                format!("{sp}.up_proj.weight"),
                rng.tensor(&[cfg.shared_expert_intermediate_size, h], 0.08),
            );
            t.insert(
                format!("{sp}.down_proj.weight"),
                rng.tensor(&[h, cfg.shared_expert_intermediate_size], 0.08),
            );
        } else {
            t.insert(
                format!("{p}.mlp.gate_proj.weight"),
                rng.tensor(&[cfg.intermediate_size, h], 0.08),
            );
            t.insert(
                format!("{p}.mlp.up_proj.weight"),
                rng.tensor(&[cfg.intermediate_size, h], 0.08),
            );
            t.insert(
                format!("{p}.mlp.down_proj.weight"),
                rng.tensor(&[h, cfg.intermediate_size], 0.08),
            );
        }
    }
    candle_core::safetensors::save(&t, dir.join("model.safetensors")).unwrap();
    cfg
}

#[cfg(feature = "cuda")]
fn fwd_span(x: &Tensor, name: &'static str) -> Option<nv_layers::prof_events::Span> {
    match x.device() {
        Device::Cuda(d) => {
            nv_layers::prof_events::span(name, &nv_layers::cuda_stream::current_stream(d))
        }
        _ => None,
    }
}

pub fn validate_input_adapter(args: &LagunaTrainArgs, path: &Path) -> Result<()> {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("adapter_config.json"))?)?;
    ensure!(
        config["r"].as_u64() == Some(args.rank as u64),
        "input adapter rank mismatch"
    );
    ensure!(
        config["lora_alpha"].as_f64() == Some(args.alpha),
        "input adapter alpha mismatch"
    );
    ensure!(
        config["peft_type"].as_str() == Some("LORA"),
        "input adapter must be LoRA"
    );
    for key in ["use_rslora", "use_dora", "fan_in_fan_out"] {
        ensure!(
            !config[key].as_bool().unwrap_or(false),
            "unsupported input adapter variant: {key}"
        );
    }
    ensure!(
        config["bias"].as_str().unwrap_or("none") == "none",
        "input adapter bias training is unsupported"
    );
    for key in ["modules_to_save", "rank_pattern", "alpha_pattern"] {
        ensure!(
            config[key].is_null() || config[key].as_object().is_some_and(|v| v.is_empty()),
            "unsupported input adapter setting: {key}"
        );
    }
    let base = config["base_model_name_or_path"]
        .as_str()
        .context("input adapter has no base identity")?;
    ensure!(
        std::fs::canonicalize(base)? == std::fs::canonicalize(&args.base)?,
        "input adapter base mismatch"
    );
    if args.out.exists() {
        ensure!(
            std::fs::canonicalize(path)? != std::fs::canonicalize(&args.out)?,
            "output must not overwrite input adapter"
        );
    }
    Ok(())
}
