use anyhow::{anyhow, ensure, Context, Result};
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch::{drive_blocking, ReadbackArm};
use nv_kernels::wgpu_qlora::qlora_head_ce::{
    head_logits_arm, upload_head, W4Head, HEAD_CE_CHUNK_ROWS,
};
use nv_weights::{StDtype, WeightLoader};

pub const TRAIN_ACCEPTS_ONLY_THE_MLX_AFFINE_U4_GS64_EXPORT_MIRRORING_NV_TRAIN_DOTS: &str =
    "the browser trainer exists for exactly one checkpoint format, the mlx-community \
     dots.ocr-4bit export (affine u4, group size 64, quantized trunk + embed + lm_head, fp \
     q/k/v bias rows, norms with the +1 already folded in); the config quantization block and \
     layer tensor shapes are checked at load so any other dots checkpoint fails at open \
     instead of silently training against the wrong base";

pub const AFFINE_GROUP: usize = 64;
pub const LM_PREFIX: &str = "language_model.model";
pub const FINAL_NORM: &str = "language_model.model.norm.weight";
pub const EMBED_MODULE: &str = "language_model.model.embed_tokens";
pub const LM_HEAD_MODULE: &str = "language_model.model.lm_head";

pub fn layer_prefix(index: usize) -> String {
    format!("{LM_PREFIX}.layers.{index}")
}

#[derive(Clone, Copy, Debug)]
pub struct TrainDims {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f32,
}

impl TrainDims {
    pub fn from_config_json(s: &str) -> Result<Self> {
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
                anyhow!(
                    "no quantization block in config.json; {}",
                    TRAIN_ACCEPTS_ONLY_THE_MLX_AFFINE_U4_GS64_EXPORT_MIRRORING_NV_TRAIN_DOTS
                )
            })?;
        ensure!(
            q.get("mode").and_then(|x| x.as_str()) == Some("affine")
                && q.get("bits").and_then(|x| x.as_u64()) == Some(4)
                && q.get("group_size").and_then(|x| x.as_u64()) == Some(AFFINE_GROUP as u64),
            "quantization block {q} is not {{mode: affine, bits: 4, group_size: {AFFINE_GROUP}}}; {}",
            TRAIN_ACCEPTS_ONLY_THE_MLX_AFFINE_U4_GS64_EXPORT_MIRRORING_NV_TRAIN_DOTS
        );
        let get = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| anyhow!("missing/invalid {k}"))
        };
        let hidden = get("hidden_size")?;
        let heads = get("num_attention_heads")?;
        let head_dim = match v.get("head_dim").and_then(|x| x.as_u64()) {
            Some(hd) => hd as usize,
            None => hidden / heads,
        };
        Ok(Self {
            hidden,
            heads,
            head_dim,
            layers: get("num_hidden_layers")?,
            kv_heads: get("num_key_value_heads")?,
            inter: get("intermediate_size")?,
            vocab: get("vocab_size")?,
            eps: v
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-6) as f32,
            rope_theta: v
                .get("rope_theta")
                .and_then(|x| x.as_f64())
                .unwrap_or(1_000_000.0) as f32,
        })
    }
}

pub fn bf16_bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn raw_bf16(loader: &WeightLoader, name: &str) -> Result<Vec<f32>> {
    ensure!(
        loader.st_dtype_of(name) == Some(StDtype::BF16),
        "{name}: expected BF16 storage"
    );
    Ok(bf16_bytes_f32(loader.raw_bytes(name)?))
}

fn raw_u32(loader: &WeightLoader, name: &str) -> Result<Vec<u32>> {
    let bytes = loader.raw_bytes(name)?;
    ensure!(
        bytes.len() % 4 == 0,
        "{name}: byte length not a u32 multiple"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn gamma_f32(loader: &WeightLoader, name: &str, expect: usize) -> Result<Vec<f32>> {
    let g = raw_bf16(loader, name)?;
    ensure!(g.len() == expect, "{name}: length {} != {expect}", g.len());
    Ok(g)
}

pub struct W4Mod {
    pub module: String,
    pub n: usize,
    pub k: usize,
    pub words: Vec<u32>,
    pub scales: Vec<f32>,
    pub biases: Vec<f32>,
    pub bias_rows: Option<Vec<f32>>,
    pub gpu: W4Head,
}

impl W4Mod {
    pub fn load(
        ctx: &WgpuContext,
        loader: &WeightLoader,
        module: &str,
        n: usize,
        k: usize,
        with_bias: bool,
    ) -> Result<Self> {
        let wname = format!("{module}.weight");
        let shape = loader
            .shape_of(&wname)
            .ok_or_else(|| anyhow!("{wname}: missing"))?;
        ensure!(
            shape == [n, k / 8],
            "{wname}: shape {shape:?} != [{n}, {}]",
            k / 8
        );
        let words = raw_u32(loader, &wname)?;
        let scales = raw_bf16(loader, &format!("{module}.scales"))?;
        let biases = raw_bf16(loader, &format!("{module}.biases"))?;
        ensure!(
            scales.len() == biases.len() && scales.len() == n * (k / AFFINE_GROUP),
            "{module}: scales/biases length {} != {} rows x {} groups",
            scales.len(),
            n,
            k / AFFINE_GROUP
        );
        let bias_rows = if with_bias {
            let b = raw_bf16(loader, &format!("{module}.bias"))?;
            ensure!(b.len() == n, "{module}.bias: length {} != {n}", b.len());
            Some(b)
        } else {
            None
        };
        let gpu = upload_head(ctx, &words, &scales, &biases, n, k)?;
        Ok(Self {
            module: module.to_string(),
            n,
            k,
            words,
            scales,
            biases,
            bias_rows,
            gpu,
        })
    }

    pub fn forward(&self, ctx: &WgpuContext, x: &[f32], m: usize) -> Result<Vec<f32>> {
        drive_blocking(self.forward_arm(ctx, x, m, ReadbackArm::Blocking))
    }

    pub async fn forward_arm(
        &self,
        ctx: &WgpuContext,
        x: &[f32],
        m: usize,
        arm: ReadbackArm,
    ) -> Result<Vec<f32>> {
        ensure!(
            x.len() == m * self.k,
            "{}: forward x length {} != {m} x {}",
            self.module,
            x.len(),
            self.k
        );
        let mut y = Vec::with_capacity(m * self.n);
        for c0 in (0..m).step_by(HEAD_CE_CHUNK_ROWS) {
            let c1 = (c0 + HEAD_CE_CHUNK_ROWS).min(m);
            y.extend(
                head_logits_arm(ctx, &self.gpu, &x[c0 * self.k..c1 * self.k], c1 - c0, arm).await?,
            );
        }
        if let Some(b) = &self.bias_rows {
            for r in 0..m {
                for (yv, bv) in y[r * self.n..(r + 1) * self.n].iter_mut().zip(b.iter()) {
                    *yv += *bv;
                }
            }
        }
        Ok(y)
    }
}

pub struct TrainLayer {
    pub index: usize,
    pub pre_gamma: Vec<f32>,
    pub post_gamma: Vec<f32>,
    pub q: W4Mod,
    pub k: W4Mod,
    pub v: W4Mod,
    pub o: W4Mod,
    pub gate: W4Mod,
    pub up: W4Mod,
    pub down: W4Mod,
}

impl TrainLayer {
    pub fn load(
        ctx: &WgpuContext,
        loader: &WeightLoader,
        d: &TrainDims,
        index: usize,
    ) -> Result<Self> {
        let p = layer_prefix(index);
        let h = d.hidden;
        let q_out = d.heads * d.head_dim;
        let kv_out = d.kv_heads * d.head_dim;
        Ok(Self {
            index,
            pre_gamma: gamma_f32(loader, &format!("{p}.input_layernorm.weight"), h)?,
            post_gamma: gamma_f32(loader, &format!("{p}.post_attention_layernorm.weight"), h)?,
            q: W4Mod::load(
                ctx,
                loader,
                &format!("{p}.self_attn.q_proj"),
                q_out,
                h,
                true,
            )?,
            k: W4Mod::load(
                ctx,
                loader,
                &format!("{p}.self_attn.k_proj"),
                kv_out,
                h,
                true,
            )?,
            v: W4Mod::load(
                ctx,
                loader,
                &format!("{p}.self_attn.v_proj"),
                kv_out,
                h,
                true,
            )?,
            o: W4Mod::load(
                ctx,
                loader,
                &format!("{p}.self_attn.o_proj"),
                h,
                q_out,
                false,
            )?,
            gate: W4Mod::load(
                ctx,
                loader,
                &format!("{p}.mlp.gate_proj"),
                d.inter,
                h,
                false,
            )?,
            up: W4Mod::load(ctx, loader, &format!("{p}.mlp.up_proj"), d.inter, h, false)?,
            down: W4Mod::load(
                ctx,
                loader,
                &format!("{p}.mlp.down_proj"),
                h,
                d.inter,
                false,
            )?,
        })
    }

    pub fn projections(&self) -> [(&'static str, &W4Mod); 7] {
        [
            ("q_proj", &self.q),
            ("k_proj", &self.k),
            ("v_proj", &self.v),
            ("o_proj", &self.o),
            ("gate_proj", &self.gate),
            ("up_proj", &self.up),
            ("down_proj", &self.down),
        ]
    }

    pub fn module_name(&self, leaf: &str) -> String {
        let p = layer_prefix(self.index);
        match leaf {
            "q_proj" | "k_proj" | "v_proj" | "o_proj" => format!("{p}.self_attn.{leaf}"),
            _ => format!("{p}.mlp.{leaf}"),
        }
    }
}

pub fn embed_rows_f32(loader: &WeightLoader, ids: &[u32], hidden: usize) -> Result<Vec<f32>> {
    let wname = format!("{EMBED_MODULE}.weight");
    let shape = loader
        .shape_of(&wname)
        .ok_or_else(|| anyhow!("{wname}: missing"))?;
    let vocab = shape[0];
    ensure!(
        shape == [vocab, hidden / 8],
        "{wname}: shape {shape:?} != [{vocab}, {}]",
        hidden / 8
    );
    let wbytes = loader.raw_bytes(&wname)?;
    let sbytes = loader.raw_bytes(&format!("{EMBED_MODULE}.scales"))?;
    let bbytes = loader.raw_bytes(&format!("{EMBED_MODULE}.biases"))?;
    let words_per_row = hidden / 8;
    let groups = hidden / AFFINE_GROUP;
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
            AFFINE_GROUP,
        ));
    }
    Ok(out)
}
