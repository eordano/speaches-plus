use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_weights::WeightLoader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::qwen38::{
    average_grads_bailing_on_an_untouched_factor, collect_raw_entries, flat_positions,
    fold_example_grads, WindowOptimizer,
};
use crate::{LoraConfig, LoraTrainable, TokenRow, TrainingLoraHook};

pub const DOTS_QLORA_TRAINS_ONLY_THE_MLX_AFFINE_U4_GS64_EXPORT: &str =
    "this trainer exists for exactly one checkpoint format, the mlx-community dots.ocr-4bit \
     export (affine u4, group size 64, quantized trunk + embed + lm_head, fp q/k/v bias rows, \
     norms with the +1 already folded in): open_dir refuses a checkpoint whose config \
     quantization block or layer-0 tensors do not match, so a bf16, nvfp4 or fp8 dots \
     checkpoint fails at open instead of silently training against the wrong base";

pub const DOTS_BASE_IS_FROZEN_DEQUANT_SO_ONLY_LORA_VARS_CARRY_GRADIENT: &str =
    "the 4-bit base dequantizes once at load into plain candle Tensors (never Vars), the \
     LoRA delta rides nv_layers::Linear's hook on the pre-bias GEMM output, and the \
     optimiser only ever sees the A/B Vars, so a training step cannot move base weights \
     and the saved adapter composes with the untouched 4-bit snapshot at serving time";

pub const DOTS_VISION_ROWS_ARE_PRECOMPUTED_SERVING_OUTPUTS_SPLICED_AS_CONSTANTS: &str =
    "training consumes the same [n_vis, hidden] f32 rows the serving tower produces \
     (nvk-dots-vision-dump replays prepare + tower encode + build_prompt_tokens), spliced \
     over the imgpad embedding rows exactly where DotsQ3dDecoder::generate splices them; \
     the tower is never instantiated here and never trains";

pub const DOTS_NORMS_LOAD_RAW_BECAUSE_THE_MLX_EXPORT_FOLDS_THE_PLUS_ONE: &str =
    "language_model.* norm gammas in the mlx export already carry the +1 that qwen \
     checkpoints store zero-centred, matching the serving loader's \
     MLX_EXPORTS_STORE_ZERO_CENTERED_NORMS_WITH_THE_PLUS_ONE_FOLDED_IN rule";

pub const WINDOW_LINEARS_PRETRANSPOSE_SO_THE_FALLBACK_STOPS_PAYING_A_TRANSPOSE_PER_FORWARD: &str =
    "the step profile priced the window forwards' Metal fallback at ~9 GB of weight transposes \
     rematerialized per training step; Linear::new stores weight_t once at build (transient for \
     the streamed frozen prefix, resident for the trained window) and the fallback reuses it \
     bitwise-by-construction. Measured trade on the n25 basis: -0.2 s/step for +1.7 GB peak.";

pub const DOTS_AFFINE_GROUP_SIZE: usize = 64;
pub const DOTS_LM_PREFIX: &str = "language_model.model";

pub fn dots_layer_prefix(index: usize) -> String {
    format!("{DOTS_LM_PREFIX}.layers.{index}")
}

pub const DOTS_EMBED_MODULE: &str = "language_model.model.embed_tokens";
pub const DOTS_FINAL_NORM: &str = "language_model.model.norm.weight";
pub const DOTS_LM_HEAD_MODULE: &str = "language_model.model.lm_head";

#[derive(Clone, Debug)]
pub struct DotsConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
}

impl DotsConfig {
    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s).context("parse dots config json")?;
        ensure!(
            v.get("model_type").and_then(|x| x.as_str()) == Some("dots_ocr"),
            "config model_type {:?} != dots_ocr",
            v.get("model_type")
        );
        let q = v
            .get("quantization")
            .or_else(|| v.get("quantization_config"))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no quantization block in config.json; {}",
                    DOTS_QLORA_TRAINS_ONLY_THE_MLX_AFFINE_U4_GS64_EXPORT
                )
            })?;
        let qget = |k: &str| q.get(k).cloned().unwrap_or(serde_json::Value::Null);
        ensure!(
            qget("mode").as_str() == Some("affine")
                && qget("bits").as_u64() == Some(4)
                && qget("group_size").as_u64() == Some(DOTS_AFFINE_GROUP_SIZE as u64),
            "quantization block {q} is not {{mode: affine, bits: 4, group_size: {}}}; {}",
            DOTS_AFFINE_GROUP_SIZE,
            DOTS_QLORA_TRAINS_ONLY_THE_MLX_AFFINE_U4_GS64_EXPORT
        );
        let get_u = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| anyhow::anyhow!("missing/invalid {k}"))
        };
        let hidden_size = get_u("hidden_size")?;
        let num_attention_heads = get_u("num_attention_heads")?;
        let head_dim = match v.get("head_dim").and_then(|x| x.as_u64()) {
            Some(hd) => hd as usize,
            None => hidden_size / num_attention_heads,
        };
        Ok(Self {
            hidden_size,
            num_attention_heads,
            head_dim,
            num_hidden_layers: get_u("num_hidden_layers")?,
            num_key_value_heads: get_u("num_key_value_heads")?,
            intermediate_size: get_u("intermediate_size")?,
            vocab_size: get_u("vocab_size")?,
            rms_norm_eps: v
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-6),
            rope_theta: v
                .get("rope_theta")
                .and_then(|x| x.as_f64())
                .unwrap_or(1_000_000.0) as f32,
        })
    }

    pub fn from_hf_json_file(p: &Path) -> Result<Self> {
        Self::from_hf_json_str(
            &std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?,
        )
    }
}

pub struct DotsW4Weights {
    loader: WeightLoader,
    device: Device,
}

impl DotsW4Weights {
    pub fn open_dir(dir: &Path, device: &Device) -> Result<Self> {
        let loader = WeightLoader::open_dir(dir, &Device::Cpu)?;
        let probe = format!("{}.mlp.gate_proj.scales", dots_layer_prefix(0));
        ensure!(
            loader.has(&probe),
            "{probe} missing from {}; {}",
            dir.display(),
            DOTS_QLORA_TRAINS_ONLY_THE_MLX_AFFINE_U4_GS64_EXPORT
        );
        Ok(Self {
            loader,
            device: device.clone(),
        })
    }

    fn raw_u32(&self, name: &str) -> Result<Vec<u32>> {
        let bytes = self.loader.raw_bytes(name)?;
        ensure!(
            bytes.len() % 4 == 0,
            "{name}: byte length not a u32 multiple"
        );
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn raw_bf16_f32(&self, name: &str) -> Result<Vec<f32>> {
        ensure!(
            self.loader.st_dtype_of(name) == Some(nv_weights::StDtype::BF16),
            "{name}: expected BF16 storage"
        );
        let bytes = self.loader.raw_bytes(name)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect())
    }

    pub fn dense_f32(&self, module: &str, n: usize, k: usize) -> Result<Vec<f32>> {
        let wname = format!("{module}.weight");
        let shape = self
            .loader
            .shape_of(&wname)
            .ok_or_else(|| anyhow::anyhow!("{wname}: missing"))?;
        ensure!(
            shape == [n, k / 8],
            "{wname}: shape {shape:?} != [{n}, {}]",
            k / 8
        );
        let words = self.raw_u32(&wname)?;
        let scales = self.raw_bf16_f32(&format!("{module}.scales"))?;
        let biases = self.raw_bf16_f32(&format!("{module}.biases"))?;
        ensure!(
            scales.len() == biases.len() && scales.len() == n * (k / DOTS_AFFINE_GROUP_SIZE),
            "{module}: scales/biases length {} != {} rows x {} groups of {}",
            scales.len(),
            n,
            k / DOTS_AFFINE_GROUP_SIZE,
            DOTS_AFFINE_GROUP_SIZE
        );
        Ok(nv_quant::mlx_affine::dequantize_mlx_affine_linear(
            &words,
            &scales,
            &biases,
            n,
            k,
            DOTS_AFFINE_GROUP_SIZE,
        ))
    }

    pub fn dense_linear(
        &self,
        module: &str,
        n: usize,
        k: usize,
        with_bias: bool,
        dtype: DType,
    ) -> Result<Linear> {
        let w = Tensor::from_vec(self.dense_f32(module, n, k)?, (n, k), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(&self.device)?;
        let bias = if with_bias {
            let b = self.raw_bf16_f32(&format!("{module}.bias"))?;
            ensure!(b.len() == n, "{module}.bias: length {} != {n}", b.len());
            Some(
                Tensor::from_vec(b, n, &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_device(&self.device)?,
            )
        } else {
            None
        };
        let _ = WINDOW_LINEARS_PRETRANSPOSE_SO_THE_FALLBACK_STOPS_PAYING_A_TRANSPOSE_PER_FORWARD;
        Linear::new(w, bias)
    }

    pub fn embed_rows_f32(&self, ids: &[u32], hidden: usize) -> Result<Vec<f32>> {
        let wname = format!("{DOTS_EMBED_MODULE}.weight");
        let shape = self
            .loader
            .shape_of(&wname)
            .ok_or_else(|| anyhow::anyhow!("{wname}: missing"))?;
        let vocab = shape[0];
        ensure!(
            shape == [vocab, hidden / 8],
            "{wname}: shape {shape:?} != [{vocab}, {}]",
            hidden / 8
        );
        let wbytes = self.loader.raw_bytes(&wname)?;
        let sbytes = self
            .loader
            .raw_bytes(&format!("{DOTS_EMBED_MODULE}.scales"))?;
        let bbytes = self
            .loader
            .raw_bytes(&format!("{DOTS_EMBED_MODULE}.biases"))?;
        let words_per_row = hidden / 8;
        let groups = hidden / DOTS_AFFINE_GROUP_SIZE;
        let bf = |bytes: &[u8], i: usize| -> f32 {
            f32::from_bits((u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]) as u32) << 16)
        };
        let mut out = Vec::with_capacity(ids.len() * hidden);
        for &id in ids {
            let row = id as usize;
            ensure!(row < vocab, "token id {id} out of vocab {vocab}");
            let wrow: Vec<u32> = (0..words_per_row)
                .map(|i| {
                    let o = (row * words_per_row + i) * 4;
                    u32::from_le_bytes([wbytes[o], wbytes[o + 1], wbytes[o + 2], wbytes[o + 3]])
                })
                .collect();
            let srow: Vec<f32> = (0..groups).map(|g| bf(sbytes, row * groups + g)).collect();
            let brow: Vec<f32> = (0..groups).map(|g| bf(bbytes, row * groups + g)).collect();
            out.extend(nv_quant::mlx_affine::dequantize_mlx_affine_linear(
                &wrow,
                &srow,
                &brow,
                1,
                hidden,
                DOTS_AFFINE_GROUP_SIZE,
            ));
        }
        Ok(out)
    }

    pub fn norm_gamma(&self, name: &str, dtype: DType) -> Result<Tensor> {
        Ok(self
            .loader
            .get(name, DType::F32)?
            .to_dtype(dtype)?
            .to_device(&self.device)?)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn w4_raw(
        &self,
        module: &str,
        n: usize,
        k: usize,
    ) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>)> {
        let wname = format!("{module}.weight");
        let shape = self
            .loader
            .shape_of(&wname)
            .ok_or_else(|| anyhow::anyhow!("{wname}: missing"))?;
        ensure!(
            shape == [n, k / 8],
            "{wname}: shape {shape:?} != [{n}, {}]",
            k / 8
        );
        let words = self.raw_u32(&wname)?;
        let scales = self.raw_bf16_f32(&format!("{module}.scales"))?;
        let biases = self.raw_bf16_f32(&format!("{module}.biases"))?;
        ensure!(
            scales.len() == biases.len() && scales.len() == n * (k / DOTS_AFFINE_GROUP_SIZE),
            "{module}: scales/biases length {} != {} rows x {} groups",
            scales.len(),
            n,
            k / DOTS_AFFINE_GROUP_SIZE
        );
        Ok((words, scales, biases))
    }

    pub fn dense_pretransposed(
        &self,
        module: &str,
        n: usize,
        k: usize,
        dtype: DType,
    ) -> Result<Tensor> {
        Ok(
            Tensor::from_vec(self.dense_f32(module, n, k)?, (n, k), &Device::Cpu)?
                .t()?
                .contiguous()?
                .to_dtype(dtype)?
                .to_device(&self.device)?,
        )
    }
}

pub const DOTS_HEAD_W_DELETES_A_933MB_STRIDED_TRANSPOSE_COPY_PER_HEAD_CHUNK: &str =
    "nv_layers::Linear's Metal fallback materializes weight.t().contiguous() on every forward, \
     so the chunked head CE re-transposed the [151936,1536] fp32 lm_head once per chunk (12x \
     per step on the n25 basis). NV_DOTS_HEAD_W picks the resident layout: unset/'view' holds \
     the untransposed [151936,1536] tensor and runs the forward GEMM through a transposed view \
     (no copy ever, and the backward dx GEMM then sees a contiguous rhs, the faster mlx layout \
     at the 151936-k shape); 1 holds one resident [1536,151936] pretransposed tensor, feeding \
     identical bytes to the identical GEMM layout the per-call copy fed, so it is bitwise by \
     construction; 0 restores the per-call Linear path. All three arms reproduce the canonical \
     20-step trajectory, the 28 eval CEs and the adapter delta-amax bit-for-bit on the n25 \
     basis, so the no-copy contiguous-dx arm ships as the default";

pub enum DotsHeadWeight {
    TransposeEveryCall(Linear),
    Pretransposed(Tensor),
    TransposedView(Tensor),
    #[cfg(feature = "metal")]
    W4Fused(std::sync::Arc<crate::dots_w4head::DotsW4Head>),
}

impl DotsHeadWeight {
    pub fn from_weights(src: &DotsW4Weights, cfg: &DotsConfig, dtype: DType) -> Result<Self> {
        let _ = DOTS_HEAD_W_DELETES_A_933MB_STRIDED_TRANSPOSE_COPY_PER_HEAD_CHUNK;
        let (n, k) = (cfg.vocab_size, cfg.hidden_size);
        if crate::w4head_requested_via_env() {
            #[cfg(feature = "metal")]
            {
                let (words, scales, biases) = src.w4_raw(DOTS_LM_HEAD_MODULE, n, k)?;
                return Ok(Self::W4Fused(crate::dots_w4head::DotsW4Head::from_raw(
                    words,
                    scales,
                    biases,
                    n,
                    k,
                    src.device(),
                )?));
            }
            #[cfg(not(feature = "metal"))]
            anyhow::bail!(
                "NV_DOTS_QLORA_W4HEAD=1 needs the metal feature: the fused W4 head is an MSL path"
            );
        }
        match std::env::var("NV_DOTS_HEAD_W").as_deref() {
            Ok("0") => Ok(Self::TransposeEveryCall(src.dense_linear(
                DOTS_LM_HEAD_MODULE,
                n,
                k,
                false,
                dtype,
            )?)),
            Ok("1") => Ok(Self::Pretransposed(src.dense_pretransposed(
                DOTS_LM_HEAD_MODULE,
                n,
                k,
                dtype,
            )?)),
            _ => Ok(Self::TransposedView(
                Tensor::from_vec(
                    src.dense_f32(DOTS_LM_HEAD_MODULE, n, k)?,
                    (n, k),
                    &Device::Cpu,
                )?
                .to_dtype(dtype)?
                .to_device(src.device())?,
            )),
        }
    }

    fn matmul_2d(&self, x: &Tensor, dims: &[usize], k: usize, n: usize) -> Result<Tensor> {
        ensure!(
            *dims.last().unwrap() == k,
            "head input last dim {} != hidden {k}",
            dims.last().unwrap()
        );
        let leading: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((leading, k))?;
        let out2 = match self {
            Self::Pretransposed(w_t) => x2.matmul(w_t)?,
            Self::TransposedView(w) => x2.matmul(&w.t()?)?,
            Self::TransposeEveryCall(_) => unreachable!("Linear owns its own forward"),
            #[cfg(feature = "metal")]
            Self::W4Fused(h) => crate::dots_w4head::logits_f32(h, &x2.contiguous()?)?,
        };
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(n);
        Ok(out2.reshape(out_dims)?)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::TransposeEveryCall(lin) => lin.forward(x),
            Self::Pretransposed(w_t) => {
                let (k, n) = (w_t.dims()[0], w_t.dims()[1]);
                self.matmul_2d(x, x.dims(), k, n)
            }
            Self::TransposedView(w) => {
                let (n, k) = (w.dims()[0], w.dims()[1]);
                self.matmul_2d(x, x.dims(), k, n)
            }
            #[cfg(feature = "metal")]
            Self::W4Fused(h) => self.matmul_2d(x, x.dims(), h.k, h.n),
        }
    }

    #[cfg(feature = "metal")]
    pub fn w4_fused(&self) -> Option<&std::sync::Arc<crate::dots_w4head::DotsW4Head>> {
        match self {
            Self::W4Fused(h) => Some(h),
            _ => None,
        }
    }
}

#[cfg(feature = "metal")]
fn rope_fused_qk(
    rope: &Rope,
    q: &Tensor,
    k: &Tensor,
    positions: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    crate::dots_rope::fused_qk(rope, q, k, positions)
}

#[cfg(not(feature = "metal"))]
fn rope_fused_qk(
    _rope: &Rope,
    _q: &Tensor,
    _k: &Tensor,
    _positions: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    Ok(None)
}

pub struct DotsTrainAttn {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl DotsTrainAttn {
    pub fn from_weights(
        src: &DotsW4Weights,
        prefix: &str,
        cfg: &DotsConfig,
        dtype: DType,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        Ok(Self {
            q_proj: src.dense_linear(&format!("{prefix}.q_proj"), q_out, h, true, dtype)?,
            k_proj: src.dense_linear(&format!("{prefix}.k_proj"), kv_out, h, true, dtype)?,
            v_proj: src.dense_linear(&format!("{prefix}.v_proj"), kv_out, h, true, dtype)?,
            o_proj: src.dense_linear(&format!("{prefix}.o_proj"), h, q_out, false, dtype)?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, rope: &Rope, positions: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        ensure!(
            dims.len() == 3,
            "DotsTrainAttn::forward expects [B, T, H], got {dims:?}"
        );
        let (b, t) = (dims[0], dims[1]);
        let in_dtype = x.dtype();
        let hd = self.head_dim;

        let q = self.q_proj.forward(x)?.reshape((b, t, self.n_heads, hd))?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, t, self.n_kv_heads, hd))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, t, self.n_kv_heads, hd))?;

        let (qr, kr) = match rope_fused_qk(rope, &q, &k, positions)? {
            Some(pair) => pair,
            None => rope.apply_candle(
                &q.to_dtype(DType::F32)?,
                &k.to_dtype(DType::F32)?,
                positions,
            )?,
        };
        let attn_out = nv_layers::attn::sdpa(
            &qr.to_dtype(in_dtype)?,
            &kr.to_dtype(in_dtype)?,
            &v,
            &nv_layers::attn::AttnConfig {
                num_heads: self.n_heads,
                num_kv_heads: self.n_kv_heads,
                head_dim: hd,
                softmax_scale: 1.0 / (hd as f32).sqrt(),
                causal: true,
            },
        )?;
        self.o_proj
            .forward(&attn_out.reshape((b, t, self.n_heads * hd))?)
    }
}

pub struct DotsTrainMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl DotsTrainMlp {
    pub fn from_weights(
        src: &DotsW4Weights,
        prefix: &str,
        cfg: &DotsConfig,
        dtype: DType,
    ) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate_proj: src.dense_linear(&format!("{prefix}.gate_proj"), i, h, false, dtype)?,
            up_proj: src.dense_linear(&format!("{prefix}.up_proj"), i, h, false, dtype)?,
            down_proj: src.dense_linear(&format!("{prefix}.down_proj"), h, i, false, dtype)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj
            .forward(&candle_nn::ops::silu(&gate)?.mul(&up)?)
    }
}

pub struct DotsTrainLayer {
    pre_norm: RmsNorm,
    post_norm: RmsNorm,
    pub attn: DotsTrainAttn,
    pub mlp: DotsTrainMlp,
    pub index: usize,
}

impl DotsTrainLayer {
    pub fn from_weights(
        src: &DotsW4Weights,
        cfg: &DotsConfig,
        index: usize,
        dtype: DType,
    ) -> Result<Self> {
        let prefix = dots_layer_prefix(index);
        Ok(Self {
            pre_norm: RmsNorm::new(
                src.norm_gamma(&format!("{prefix}.input_layernorm.weight"), dtype)?,
                cfg.rms_norm_eps,
            ),
            post_norm: RmsNorm::new(
                src.norm_gamma(&format!("{prefix}.post_attention_layernorm.weight"), dtype)?,
                cfg.rms_norm_eps,
            ),
            attn: DotsTrainAttn::from_weights(src, &format!("{prefix}.self_attn"), cfg, dtype)?,
            mlp: DotsTrainMlp::from_weights(src, &format!("{prefix}.mlp"), cfg, dtype)?,
            index,
        })
    }

    pub fn forward(&self, x: &Tensor, rope: &Rope, positions: &Tensor) -> Result<Tensor> {
        let normed = self.pre_norm.forward_candle(x)?;
        let after_mix = x.add(&self.attn.forward(&normed, rope, positions)?)?;
        let normed_post = self.post_norm.forward_candle(&after_mix)?;
        Ok(after_mix.add(&self.mlp.forward(&normed_post)?)?)
    }

    pub fn projections(&self) -> Vec<(&'static str, &Linear)> {
        vec![
            ("q_proj", &self.attn.q_proj),
            ("k_proj", &self.attn.k_proj),
            ("v_proj", &self.attn.v_proj),
            ("o_proj", &self.attn.o_proj),
            ("gate_proj", &self.mlp.gate_proj),
            ("up_proj", &self.mlp.up_proj),
            ("down_proj", &self.mlp.down_proj),
        ]
    }

    pub fn module_name(&self, leaf: &str) -> String {
        let prefix = dots_layer_prefix(self.index);
        match leaf {
            "q_proj" | "k_proj" | "v_proj" | "o_proj" => format!("{prefix}.self_attn.{leaf}"),
            _ => format!("{prefix}.mlp.{leaf}"),
        }
    }
}

pub fn dots_default_targets() -> Vec<String> {
    [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

pub fn attach_dots_window_loras(
    window: &[DotsTrainLayer],
    targets: &[String],
    lcfg: LoraConfig,
    seed0: u64,
    device: &Device,
) -> Result<(Vec<(String, Arc<LoraTrainable>)>, Vec<Var>)> {
    let mut loras: Vec<(String, Arc<LoraTrainable>)> = Vec::new();
    let mut vars: Vec<Var> = Vec::new();
    let mut seed = seed0;
    for layer in window {
        for (leaf, lin) in layer.projections() {
            if !targets.iter().any(|t| t == leaf) {
                continue;
            }
            let lora = Arc::new(LoraTrainable::from_features(
                lin.out_features(),
                lin.in_features(),
                lcfg,
                seed,
                device,
            )?);
            seed = seed.wrapping_add(1);
            vars.extend(lora.trainable_vars());
            lin.attach_lora(Arc::new(TrainingLoraHook::new(lora.clone(), true)))?;
            loras.push((layer.module_name(leaf), lora));
        }
    }
    ensure!(
        !loras.is_empty(),
        "no LoRA target matched: targets {targets:?}"
    );
    Ok((loras, vars))
}

pub struct DotsSample {
    pub ids: Vec<u32>,
    pub completion_start: usize,
    pub vision_start: usize,
    pub n_vis: usize,
    pub rows: Vec<f32>,
}

pub const DUMP_IDS: &str = "ids";
pub const DUMP_ROWS: &str = "rows";
pub const DUMP_VISION_START: &str = "vision_start";
pub const DUMP_COMPLETION_START: &str = "completion_start";

pub fn load_dump_dir(dir: &Path, hidden: usize) -> Result<Vec<DotsSample>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read dump dir {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    paths.sort();
    ensure!(
        !paths.is_empty(),
        "no .safetensors samples in {}",
        dir.display()
    );
    let mut out = Vec::with_capacity(paths.len());
    for p in &paths {
        let t = candle_core::safetensors::load(p, &Device::Cpu)
            .with_context(|| format!("load {}", p.display()))?;
        let get = |k: &str| -> Result<&Tensor> {
            t.get(k)
                .ok_or_else(|| anyhow::anyhow!("{}: missing tensor {k}", p.display()))
        };
        let ids = get(DUMP_IDS)?.to_vec1::<u32>()?;
        let vision_start = get(DUMP_VISION_START)?.to_vec1::<u32>()?[0] as usize;
        let completion_start = get(DUMP_COMPLETION_START)?.to_vec1::<u32>()?[0] as usize;
        let rows_t = get(DUMP_ROWS)?;
        let rdims = rows_t.dims().to_vec();
        ensure!(
            rdims.len() == 2 && rdims[1] == hidden,
            "{}: rows shape {rdims:?} != [n, {hidden}]",
            p.display()
        );
        let n_vis = rdims[0];
        let rows = rows_t.flatten_all()?.to_vec1::<f32>()?;
        ensure!(
            ids.len() >= 2
                && vision_start + n_vis < completion_start
                && completion_start < ids.len(),
            "{}: inconsistent sample: len {} vision_start {vision_start} n_vis {n_vis} \
             completion_start {completion_start}",
            p.display(),
            ids.len()
        );
        out.push(DotsSample {
            ids,
            completion_start,
            vision_start,
            n_vis,
            rows,
        });
    }
    Ok(out)
}

type Ex = (Tensor, Tensor, Tensor, Vec<f32>);

fn embed_spliced_shifted(
    src: &DotsW4Weights,
    s: &DotsSample,
    hidden: usize,
    dtype: DType,
    device: &Device,
) -> Result<Ex> {
    let t = s.ids.len() - 1;
    ensure!(
        s.vision_start + s.n_vis <= t,
        "vision span [{}, {}) exceeds shifted length {t}",
        s.vision_start,
        s.vision_start + s.n_vis
    );
    let mut v = src.embed_rows_f32(&s.ids[..t], hidden)?;
    v[s.vision_start * hidden..(s.vision_start + s.n_vis) * hidden].copy_from_slice(&s.rows);
    let h = Tensor::from_vec(v, (1, t, hidden), &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)?;
    let positions = flat_positions(1, t, device)?;
    let tgt = Tensor::from_vec(s.ids[1..].to_vec(), t, device)?;
    let mask = TokenRow {
        ids: s.ids.clone(),
        completion_start: s.completion_start,
    }
    .target_mask_where_predicting_a_prompt_token_weighs_zero();
    Ok((h, positions, tgt, mask))
}

pub const DOTS_BUCKET_PADDED_TOKEN_BUDGET_CAPS_ACTIVATION_PEAK_WHILE_CUTTING_PER_LAUNCH_OVERHEAD:
    usize = 512;

struct DotsBucket {
    h: Tensor,
    positions: Tensor,
    tgt: Tensor,
    row_mean_weights: Tensor,
    rows: usize,
    padded_t: usize,
}

fn pack_length_sorted_buckets(exs: &[Ex], device: &Device) -> Result<Vec<DotsBucket>> {
    let mut order: Vec<usize> = (0..exs.len()).collect();
    order.sort_by_key(|&i| (usize::MAX - exs[i].0.dims()[1], i));
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for &i in &order {
        match groups.last_mut() {
            Some(g)
                if (g.len() + 1) * exs[g[0]].0.dims()[1]
                    <= DOTS_BUCKET_PADDED_TOKEN_BUDGET_CAPS_ACTIVATION_PEAK_WHILE_CUTTING_PER_LAUNCH_OVERHEAD =>
            {
                g.push(i)
            }
            _ => groups.push(vec![i]),
        }
    }
    groups
        .iter()
        .map(|g| build_bucket(exs, g, device))
        .collect()
}

fn build_bucket(exs: &[Ex], idxs: &[usize], device: &Device) -> Result<DotsBucket> {
    let tp = exs[idxs[0]].0.dims()[1];
    let mut hs: Vec<Tensor> = Vec::with_capacity(idxs.len());
    let mut tgt_ids: Vec<u32> = Vec::with_capacity(idxs.len() * tp);
    let mut weights: Vec<f32> = Vec::with_capacity(idxs.len() * tp);
    for &i in idxs {
        let (h, _, tgt, mask) = &exs[i];
        let t = h.dims()[1];
        ensure!(
            t <= tp,
            "bucket rows must be length-sorted: {t} > padded {tp}"
        );
        hs.push(if t < tp {
            let pad = Tensor::zeros((1, tp - t, h.dims()[2]), h.dtype(), h.device())?;
            Tensor::cat(&[h, &pad], 1)?
        } else {
            h.clone()
        });
        let mut ids = tgt.to_vec1::<u32>()?;
        ids.resize(tp, 0);
        tgt_ids.extend(ids);
        let unmasked: f32 = mask.iter().sum();
        ensure!(
            unmasked > 0.0,
            "example {i}: every target position is masked"
        );
        weights.extend(mask.iter().map(|w| w / unmasked));
        weights.extend(std::iter::repeat_n(0.0f32, tp - t));
    }
    Ok(DotsBucket {
        h: Tensor::cat(&hs, 0)?,
        positions: flat_positions(idxs.len(), tp, device)?,
        tgt: Tensor::from_vec(tgt_ids, idxs.len() * tp, device)?,
        row_mean_weights: Tensor::from_vec(weights, idxs.len() * tp, device)?,
        rows: idxs.len(),
        padded_t: tp,
    })
}

pub const DOTS_HEAD_CE_ROW_CHUNK_KEEPS_ONE_VOCAB_LOGITS_SLICE_LIVE_AT_A_TIME: usize = 256;

pub fn head_ce_chunk_rows_overridable_for_reladdering_because_chunking_reassociates_the_loss_sum(
) -> usize {
    std::env::var("NV_DOTS_HEAD_CE_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&c: &usize| c > 0)
        .unwrap_or(DOTS_HEAD_CE_ROW_CHUNK_KEEPS_ONE_VOCAB_LOGITS_SLICE_LIVE_AT_A_TIME)
}

#[cfg(feature = "metal")]
fn w4_fused_chunk_partial(
    lm_head: &DotsHeadWeight,
    xc: &Tensor,
    tgt_c: &Tensor,
    w_c: &Tensor,
) -> Result<Option<Tensor>> {
    match lm_head.w4_fused() {
        Some(h) => Ok(Some(
            crate::dots_w4head::fused_ce_rows(h, &xc.contiguous()?, tgt_c, w_c)?.sum_all()?,
        )),
        None => Ok(None),
    }
}

#[cfg(not(feature = "metal"))]
fn w4_fused_chunk_partial(
    _lm_head: &DotsHeadWeight,
    _xc: &Tensor,
    _tgt_c: &Tensor,
    _w_c: &Tensor,
) -> Result<Option<Tensor>> {
    Ok(None)
}

fn bucket_head_loss_and_boundary_grad(
    final_norm: &RmsNorm,
    lm_head: &DotsHeadWeight,
    vocab: usize,
    b: &DotsBucket,
    hv: &Var,
) -> Result<(f32, Tensor)> {
    let n = b.rows * b.padded_t;
    let hidden = hv.as_tensor().dims()[2];
    let flat = hv.as_tensor().reshape((n, hidden))?;
    let mut loss_sum: Option<Tensor> = None;
    let mut grad_head: Option<Tensor> = None;
    let mut start = 0usize;
    let chunk =
        head_ce_chunk_rows_overridable_for_reladdering_because_chunking_reassociates_the_loss_sum();
    while start < n {
        let len = chunk.min(n - start);
        let xc = final_norm.forward_candle(&flat.narrow(0, start, len)?)?;
        let tgt_c = b.tgt.narrow(0, start, len)?;
        let w_c = b.row_mean_weights.narrow(0, start, len)?;
        let partial = match w4_fused_chunk_partial(lm_head, &xc, &tgt_c, &w_c)? {
            Some(p) => p,
            None => {
                let logits = lm_head
                    .forward(&xc)?
                    .reshape((len, vocab))?
                    .to_dtype(DType::F32)?;
                let logp = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
                let picked = logp.gather(&tgt_c.unsqueeze(1)?, 1)?.squeeze(1)?;
                picked.mul(&w_c)?.sum_all()?.neg()?
            }
        };
        let gs = partial.backward()?;
        let g = gs
            .get(hv.as_tensor())
            .ok_or_else(|| anyhow::anyhow!("no grad for the chunked head boundary"))?;
        grad_head = Some(match grad_head.take() {
            Some(a) => a.add(g)?,
            None => g.clone(),
        });
        loss_sum = Some(match loss_sum.take() {
            Some(a) => a.add(&partial.detach())?,
            None => partial.detach(),
        });
        start += len;
    }
    let lv = loss_sum
        .ok_or_else(|| anyhow::anyhow!("empty bucket has no head loss"))?
        .to_scalar::<f32>()?;
    let grad = grad_head.expect("chunk loop ran at least once so the boundary grad exists");
    Ok((lv, grad))
}

pub const DOTS_LIVE_BUCKET_GRAPH_DELETES_THE_CHECKPOINT_REFORWARD_AND_REASSOCIATES_GRADS: &str =
    "the qlora step profile priced the per-layer checkpoint re-forward at 1.08 s/step (26 \
     percent), so the default bucket backward keeps the whole grad-layer graph live through one \
     forward and runs one surrogate backward from the head boundary; the chunked head CE still \
     sees a detached boundary Var, so the head phase is unchanged, but layer grads flow through \
     one autograd pass instead of eight isolated surrogate backwards and may reassociate \
     (non-bitwise vs the checkpointed path). NV_DOTS_QLORA_CHECKPOINT=1 restores the \
     re-forwarding per-layer backward and its lower activation footprint";

fn bucket_loss_and_grads_live_graph(
    window: &[DotsTrainLayer],
    final_norm: &RmsNorm,
    lm_head: &DotsHeadWeight,
    rope: &Rope,
    vocab: usize,
    b: &DotsBucket,
    vars: &[Var],
) -> Result<(f32, Vec<Option<Tensor>>)> {
    let mut x = b.h.detach();
    for layer in window {
        x = layer.forward(&x, rope, &b.positions)?;
    }
    let hv = Var::from_tensor(&x.detach())?;
    let (lv, grad_boundary) =
        bucket_head_loss_and_boundary_grad(final_norm, lm_head, vocab, b, &hv)?;
    let surrogate = x
        .to_dtype(DType::F32)?
        .mul(&grad_boundary.detach().to_dtype(DType::F32)?)?
        .sum_all()?;
    let gs = surrogate.backward()?;
    Ok((lv, vars.iter().map(|v| gs.get(v).cloned()).collect()))
}

fn bucket_loss_and_grads_checkpointed(
    window: &[DotsTrainLayer],
    final_norm: &RmsNorm,
    lm_head: &DotsHeadWeight,
    rope: &Rope,
    vocab: usize,
    b: &DotsBucket,
    vars: &[Var],
) -> Result<(f32, Vec<Option<Tensor>>)> {
    let mut boundaries: Vec<Tensor> = Vec::with_capacity(window.len() + 1);
    boundaries.push(b.h.detach());
    for layer in window {
        let out = layer
            .forward(boundaries.last().unwrap(), rope, &b.positions)?
            .detach();
        boundaries.push(out);
    }
    let hv = Var::from_tensor(boundaries.last().unwrap())?;
    let (lv, mut grad_next) =
        bucket_head_loss_and_boundary_grad(final_norm, lm_head, vocab, b, &hv)?;
    let mut acc: Vec<Option<Tensor>> = vec![None; vars.len()];
    for li in (0..window.len()).rev() {
        let inp = Var::from_tensor(&boundaries[li])?;
        let out = window[li].forward(inp.as_tensor(), rope, &b.positions)?;
        let surrogate = out
            .to_dtype(DType::F32)?
            .mul(&grad_next.detach().to_dtype(DType::F32)?)?
            .sum_all()?;
        let layer_gs = surrogate.backward()?;
        for (vi, v) in vars.iter().enumerate() {
            if let Some(g) = layer_gs.get(v) {
                let g = g.clone();
                acc[vi] = Some(match acc[vi].take() {
                    Some(a) => a.add(&g.to_dtype(a.dtype())?)?,
                    None => g,
                });
            }
        }
        grad_next = layer_gs
            .get(inp.as_tensor())
            .cloned()
            .with_context(|| format!("no grad for checkpointed window layer {li} input"))?;
    }
    Ok((lv, acc))
}

fn dots_logits_f32_after_head(
    final_norm: &RmsNorm,
    lm_head: &DotsHeadWeight,
    x: &Tensor,
    vocab: usize,
) -> Result<Tensor> {
    let t = x.dims()[1];
    let x = final_norm.forward_candle(x)?;
    Ok(lm_head
        .forward(&x)?
        .reshape((t, vocab))?
        .to_dtype(DType::F32)?)
}

fn detached_window_output(
    window: &[DotsTrainLayer],
    rope: &Rope,
    h: &Tensor,
    positions: &Tensor,
) -> Result<Tensor> {
    let mut x = h.detach();
    for layer in window {
        x = layer.forward(&x, rope, positions)?.detach();
    }
    Ok(x)
}

fn per_row_ce(
    window: &[DotsTrainLayer],
    final_norm: &RmsNorm,
    lm_head: &DotsHeadWeight,
    rope: &Rope,
    vocab: usize,
    exs: &[Ex],
) -> Result<Vec<f32>> {
    exs.iter()
        .map(|(h, positions, tgt, mask)| {
            let x = detached_window_output(window, rope, h, positions)?;
            let logits = dots_logits_f32_after_head(final_norm, lm_head, &x, vocab)?;
            Ok(crate::ce_mean_over_unmasked_positions(&logits, tgt, mask)?.to_scalar::<f32>()?)
        })
        .collect()
}

fn dots_train_steps(
    vars: &[Var],
    loras: &[(String, Arc<LoraTrainable>)],
    steps: usize,
    lr: f64,
    b_lr_ratio: f64,
    n_examples: usize,
    n_units: usize,
    mut loss_sum_and_grads_for_unit: impl FnMut(usize) -> Result<(f32, Vec<Option<Tensor>>)>,
) -> Result<(Vec<f32>, Vec<f64>, bool)> {
    let mut losses: Vec<f32> = Vec::with_capacity(steps);
    let mut step_seconds: Vec<f64> = Vec::with_capacity(steps);
    let mut all_grads_finite = true;
    if steps == 0 {
        return Ok((losses, step_seconds, all_grads_finite));
    }
    let mut opt = WindowOptimizer::new(vars, lr, b_lr_ratio)?;
    let fence_examples_on_metal = vars.first().is_some_and(|v| v.device().is_metal())
        && std::env::var("NV_DOTS_METAL_EXAMPLE_READBACK").as_deref() == Ok("1");
    let _ = METAL_PER_EXAMPLE_GRAD_READBACK_IS_AN_OPT_IN_ATTRIBUTION_PROBE;
    for step in 0..steps {
        let t0 = std::time::Instant::now();
        let mut acc: Vec<Option<Tensor>> = vec![None; vars.len()];
        let mut loss_sum = 0f32;
        for i in 0..n_units {
            let (lv, grads) = loss_sum_and_grads_for_unit(i)?;
            if !lv.is_finite() {
                eprintln!("[dots-lora-train] step {step} unit {i}: FORWARD loss non-finite");
            }
            if fence_examples_on_metal {
                for (vi, g) in grads.iter().enumerate() {
                    let Some(g) = g else { continue };
                    let bad = g
                        .flatten_all()
                        .and_then(|f| f.abs())
                        .and_then(|f| f.max(0))
                        .and_then(|m| m.to_scalar::<f32>())
                        .map(|m| !m.is_finite())
                        .unwrap_or(true);
                    if bad && lv.is_finite() {
                        let (name, half) = match loras.get(vi / 2) {
                            Some((n, _)) => (n.as_str(), if vi % 2 == 0 { "A" } else { "B" }),
                            None => ("?", "?"),
                        };
                        eprintln!(
                            "[dots-lora-train] step {step} unit {i}: finite loss {lv:.6} \
                             but grad var {vi} ({name} lora_{half}) is non-finite"
                        );
                    }
                }
            }
            loss_sum += lv;
            fold_example_grads(&mut acc, grads)?;
        }
        let inv = 1.0 / n_examples as f64;
        if !average_grads_bailing_on_an_untouched_factor(&mut acc, loras, inv, step)? {
            all_grads_finite = false;
        }
        opt.step(vars, &acc)?;
        if let Some(v) = vars.first() {
            let d = v.device();
            if d.is_metal() {
                d.synchronize()?;
            }
        }
        let lv = loss_sum * inv as f32;
        losses.push(lv);
        step_seconds.push(t0.elapsed().as_secs_f64());
        eprintln!(
            "[dots-lora-train] step {step:3}  loss {lv:.6}  {:.2}s",
            step_seconds[step]
        );
    }
    Ok((losses, step_seconds, all_grads_finite))
}

#[derive(Clone, Debug)]
pub struct DotsTrainArgs {
    pub base: PathBuf,
    pub data: PathBuf,
    pub eval_data: Option<PathBuf>,
    pub out: PathBuf,
    pub grad_layers: usize,
    pub rank: usize,
    pub alpha: f64,
    pub lr: f64,
    pub b_lr_ratio: f64,
    pub steps: usize,
    pub targets: Vec<String>,
    pub seed: u64,
    pub dtype: DType,
}

#[derive(Debug)]
pub struct DotsTrainSummary {
    pub losses: Vec<f32>,
    pub step_seconds: Vec<f64>,
    pub prefix_seconds: f64,
    pub window_build_seconds: f64,
    pub window_start: usize,
    pub modules: Vec<String>,
    pub trainable_vars: usize,
    pub num_examples: usize,
    pub num_eval_examples: usize,
    pub eval_ce_base: Vec<f32>,
    pub eval_ce_adapter: Vec<f32>,
    pub min_module_delta_amax: f32,
    pub max_module_delta_amax: f32,
    pub all_grads_finite: bool,
    pub device: String,
    pub adapter_dir: PathBuf,
}

impl DotsTrainSummary {
    pub fn eval_loss_base(&self) -> Option<f32> {
        crate::eval_loss_scalar_is_the_sum_of_the_per_row_ce_means_in_dataset_row_order(
            &self.eval_ce_base,
        )
    }

    pub fn eval_loss_adapter(&self) -> Option<f32> {
        crate::eval_loss_scalar_is_the_sum_of_the_per_row_ce_means_in_dataset_row_order(
            &self.eval_ce_adapter,
        )
    }
}

pub const METAL_NON_FINITE_GRADS_WERE_A_RANK5_STRIDED_REDUCE_OOB_NOT_A_RACE_SO_THE_WALK_RUNS_UNSYNCED: &str =
    "candle-metal-kernels 0.11.0's get_strided_idx_fallback decomposed rank>=5 strided reduces \
     outer-dim-first, reading out of bounds (k/v grads hit it through the gqa expand backward's \
     rank-5 sum); the pinned upstream main fixes the indexer, Metal shader validation reports \
     zero invalid loads, and the unsynced walk is finite, so training no longer sets \
     CANDLE_METAL_BACKWARD_SYNC_EVERY. The env stays honored for reproducing the old fenced \
     numbers (~24.9 s/step on the M2 Ultra at sync=1).";

pub const METAL_PER_EXAMPLE_GRAD_READBACK_IS_AN_OPT_IN_ATTRIBUTION_PROBE: &str =
    "NV_DOTS_METAL_EXAMPLE_READBACK=1 reads every grad back to the host per example and prints \
     an attributed FORWARD/BACKWARD line for any non-finite value (module name plus lora half). \
     It was the fence and the instrument that localized the strided-reduce bug; with the kernel \
     fixed it is a ~2 percent diagnostic, off by default.";

pub const METAL_PER_OP_ENCODER_OVERHEAD_DOMINATES_THE_STEP_SO_DOTS_BATCHES_FOUR_OPS_PER_COMMAND_BUFFER:
    &str =
    "the qlora 4-bit step profile brackets ~80 percent of the step as candle-Metal per-op \
     encoder plus fence overhead, not arithmetic, and the command-batching ladder \
     (CANDLE_METAL_COMPUTE_PER_BUFFER 500/50/10/4/2/1) is loss-bit-identical at every rung \
     with its knee at 4: 4.16 -> 3.6 s/step on the n25 basis. The dots entrypoints default \
     the env to 4 before metal device creation (the backend reads it once, at Commands::new); \
     a caller-set value always wins.";

pub const FUSION_LEVERS_ARE_BITWISE_SO_THE_DOTS_ENTRYPOINTS_DEFAULT_THEM_ON: &str =
    "the R3 census found the step's largest hidden cost was a host position round-trip \
     draining the whole Metal queue 96x/step; the four levers (resident rope positions, \
     fused rope kernel with exact rounding and GradStore order, transposed lora factor \
     views, resident attn mask/scale caches) each reproduce the canonical trajectory \
     bit-identically alone and composed, so they default on at the dots entrypoints with \
     caller-set values winning; the composed arm's command-batching knee moved from 4 to 8. \
     Composed default on the n25 basis: 1.47 s/step at 16.5 GB peak (was 1.98 at 14.2); \
     NV_DOTS_QLORA_CHECKPOINT=1 stays the low-memory escape and each lever env unwinds \
     individually.";

pub const THE_CER_BLESSING_FLIPPED_THE_REASSOCIATING_ARMS_ON_BYTE_IDENTICAL_OCR_OUTPUT: &str =
    "the named corpus instrument ran on adapters trained under each arm (identical serving \
     path, only the adapter differing): OCR hypotheses byte-identical on every held-out and \
     every train row, markup-neutral CER 0.0 both arms, so the w4 fused head and grouped-GQA \
     attention default on and the canonical trajectory re-pins to the composed arm's (still \
     deterministic to every digit across reps). The pre-blessing bit-exact path remains one \
     env away (NV_DOTS_QLORA_W4HEAD=0 NV_ATTN_GQA_GROUPED=0 reproduces 1.47 s/step and the \
     prior canonical exactly).";

pub fn metal_command_batching_default() {
    let _ = METAL_PER_OP_ENCODER_OVERHEAD_DOMINATES_THE_STEP_SO_DOTS_BATCHES_FOUR_OPS_PER_COMMAND_BUFFER;
    let _ = FUSION_LEVERS_ARE_BITWISE_SO_THE_DOTS_ENTRYPOINTS_DEFAULT_THEM_ON;
    let _ = THE_CER_BLESSING_FLIPPED_THE_REASSOCIATING_ARMS_ON_BYTE_IDENTICAL_OCR_OUTPUT;
    for (key, val) in [
        ("CANDLE_METAL_COMPUTE_PER_BUFFER", "8"),
        ("NV_ROPE_POS_RESIDENT", "1"),
        ("NV_DOTS_ROPE_FUSED", "1"),
        ("NV_DOTS_LORA_TVIEW", "1"),
        ("NV_ATTN_RESIDENT", "1"),
        ("NV_ATTN_GQA_GROUPED", "1"),
    ] {
        if std::env::var_os(key).is_none() {
            std::env::set_var(key, val);
        }
    }
}

pub fn run(args: &DotsTrainArgs, device: &Device) -> Result<DotsTrainSummary> {
    if device.is_metal() && std::env::var_os("NV_DOTS_QLORA_W4HEAD").is_none() {
        std::env::set_var("NV_DOTS_QLORA_W4HEAD", "1");
    }
    let _ =
        METAL_NON_FINITE_GRADS_WERE_A_RANK5_STRIDED_REDUCE_OOB_NOT_A_RACE_SO_THE_WALK_RUNS_UNSYNCED;
    let cfg = DotsConfig::from_hf_json_file(&args.base.join("config.json"))?;
    ensure!(
        args.grad_layers >= 1 && args.grad_layers <= cfg.num_hidden_layers,
        "grad_layers {} outside 1..={}",
        args.grad_layers,
        cfg.num_hidden_layers
    );
    let src = DotsW4Weights::open_dir(&args.base, device)?;
    let train_samples = load_dump_dir(&args.data, cfg.hidden_size)?;
    let eval_samples = match &args.eval_data {
        Some(p) => load_dump_dir(p, cfg.hidden_size)?,
        None => Vec::new(),
    };
    let max_t = train_samples
        .iter()
        .chain(eval_samples.iter())
        .map(|s| s.ids.len() - 1)
        .max()
        .unwrap_or(2);
    let rope = Rope::new(
        RopeConfig {
            head_dim: cfg.head_dim,
            max_seq_len: max_t.max(2),
            base: cfg.rope_theta,
            kind: RopeKind::Standard,
        },
        device,
    )?;

    let t_prefix = std::time::Instant::now();
    let mut exs: Vec<Ex> = train_samples
        .iter()
        .map(|s| embed_spliced_shifted(&src, s, cfg.hidden_size, args.dtype, device))
        .collect::<Result<_>>()?;
    let mut eval_exs: Vec<Ex> = eval_samples
        .iter()
        .map(|s| embed_spliced_shifted(&src, s, cfg.hidden_size, args.dtype, device))
        .collect::<Result<_>>()?;
    let window_start = cfg.num_hidden_layers - args.grad_layers;
    for li in 0..window_start {
        let layer = DotsTrainLayer::from_weights(&src, &cfg, li, args.dtype)?;
        for (h, positions, _, _) in exs.iter_mut().chain(eval_exs.iter_mut()) {
            *h = layer.forward(h, &rope, positions)?.detach();
        }
        drop(layer);
    }
    let prefix_seconds = t_prefix.elapsed().as_secs_f64();

    let t_window = std::time::Instant::now();
    let mut window: Vec<DotsTrainLayer> = Vec::with_capacity(args.grad_layers);
    for li in window_start..cfg.num_hidden_layers {
        window.push(
            DotsTrainLayer::from_weights(&src, &cfg, li, args.dtype)
                .with_context(|| format!("build trainable window layer {li}"))?,
        );
    }
    let final_norm = RmsNorm::new(
        src.norm_gamma(DOTS_FINAL_NORM, args.dtype)?,
        cfg.rms_norm_eps,
    );
    let lm_head = DotsHeadWeight::from_weights(&src, &cfg, args.dtype)?;
    let window_build_seconds = t_window.elapsed().as_secs_f64();

    let eval_ce_base = per_row_ce(
        &window,
        &final_norm,
        &lm_head,
        &rope,
        cfg.vocab_size,
        &eval_exs,
    )?;

    let lcfg = LoraConfig {
        r: args.rank,
        alpha: args.alpha,
        dropout: 0.0,
    };
    let (loras, vars) = attach_dots_window_loras(&window, &args.targets, lcfg, args.seed, device)?;

    let buckets = pack_length_sorted_buckets(&exs, device)?;
    let _ = DOTS_LIVE_BUCKET_GRAPH_DELETES_THE_CHECKPOINT_REFORWARD_AND_REASSOCIATES_GRADS;
    let checkpointed = std::env::var("NV_DOTS_QLORA_CHECKPOINT").as_deref() == Ok("1");
    let (losses, step_seconds, all_grads_finite) = dots_train_steps(
        &vars,
        &loras,
        args.steps,
        args.lr,
        args.b_lr_ratio,
        exs.len(),
        buckets.len(),
        |i| {
            if checkpointed {
                bucket_loss_and_grads_checkpointed(
                    &window,
                    &final_norm,
                    &lm_head,
                    &rope,
                    cfg.vocab_size,
                    &buckets[i],
                    &vars,
                )
            } else {
                bucket_loss_and_grads_live_graph(
                    &window,
                    &final_norm,
                    &lm_head,
                    &rope,
                    cfg.vocab_size,
                    &buckets[i],
                    &vars,
                )
            }
        },
    )?;

    let eval_ce_adapter = per_row_ce(
        &window,
        &final_norm,
        &lm_head,
        &rope,
        cfg.vocab_size,
        &eval_exs,
    )?;

    let (entries, min_amax, max_amax) = collect_raw_entries(&loras)?;
    crate::save_peft_raw(
        &args.out,
        &entries,
        args.rank,
        args.alpha,
        0.0,
        &args.base.display().to_string(),
    )?;

    Ok(DotsTrainSummary {
        losses,
        step_seconds,
        prefix_seconds,
        window_build_seconds,
        window_start,
        modules: loras.iter().map(|(m, _)| m.clone()).collect(),
        trainable_vars: vars.len(),
        num_examples: exs.len(),
        num_eval_examples: eval_exs.len(),
        eval_ce_base,
        eval_ce_adapter,
        min_module_delta_amax: min_amax,
        max_module_delta_amax: max_amax,
        all_grads_finite,
        device: format!("{device:?}"),
        adapter_dir: args.out.clone(),
    })
}

pub fn write_dump_sample(
    path: &Path,
    ids: &[u32],
    rows: &[f32],
    hidden: usize,
    vision_start: usize,
    completion_start: usize,
) -> Result<()> {
    ensure!(
        rows.len().is_multiple_of(hidden),
        "rows length not a hidden multiple"
    );
    let n_vis = rows.len() / hidden;
    ensure!(
        ids.len() >= 2 && vision_start + n_vis < completion_start && completion_start < ids.len(),
        "inconsistent dump sample: len {} vision_start {vision_start} n_vis {n_vis} \
         completion_start {completion_start}",
        ids.len()
    );
    let dev = Device::Cpu;
    let mut tensors = std::collections::HashMap::new();
    tensors.insert(
        DUMP_IDS.to_string(),
        Tensor::from_vec(ids.to_vec(), ids.len(), &dev)?,
    );
    tensors.insert(
        DUMP_ROWS.to_string(),
        Tensor::from_vec(rows.to_vec(), (n_vis, hidden), &dev)?,
    );
    tensors.insert(
        DUMP_VISION_START.to_string(),
        Tensor::from_vec(vec![vision_start as u32], 1, &dev)?,
    );
    tensors.insert(
        DUMP_COMPLETION_START.to_string(),
        Tensor::from_vec(vec![completion_start as u32], 1, &dev)?,
    );
    candle_core::safetensors::save(&tensors, path)?;
    Ok(())
}
