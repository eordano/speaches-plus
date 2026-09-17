use anyhow::{anyhow, ensure, Context, Result};
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_qlora::qlora_head_ce::{upload_head, W4Head};
use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_quant::mlx_affine::quantize_mlx_affine_linear;
use nv_weights::{StDtype, WeightLoader};
use safetensors::tensor::{Dtype, TensorView};

use crate::train::weights::{bf16_bytes_f32, W4Mod, AFFINE_GROUP};

pub const PREFIX: &str = "model.language_model";

pub const E4B_TRAINER_QUANTIZES_THE_BF16_TEXT_TOWER_TO_MLX_AFFINE_U4_GS64_AT_LOAD_SO_THE_DOTS_W4_KERNELS_CARRY_IT: &str =
    "google/gemma-4-E4B-it ships bf16 projections and a tied bf16 embed; every projection and \
     the head are quantized on the host through nv_quant's bit-exact mlx affine port into the \
     words/scales/biases triple upload_head takes, so the E4B schedule reuses the dots W4 \
     forward, dX and CE kernels unchanged; norms, layer scalars and the per-layer embedding \
     rows stay f32, and the qat-w4a16 export (symmetric nibbles, its own group size) is \
     refused rather than re-derived";

pub const E4B_HEAD_IS_THE_QUANTIZED_EMBED_TABLE_WITH_THE_FINAL_LOGIT_SOFTCAP_ON: &str =
    "tie_word_embeddings is asserted, so the CE head is the embed table quantized in row \
     chunks and uploaded once with W4Head.softcap = final_logit_softcapping; the embedding \
     lookup reads the bf16 rows directly and scales them by sqrt(hidden) like the serving \
     prologue";

pub const E4B_PER_LAYER_INPUT_ROWS_ARE_TOKEN_CONSTANTS_BECAUSE_EVERY_TENSOR_FEEDING_THEM_IS_FROZEN: &str =
    "pli = rms(per_layer_model_projection(scaled_embed)) / sqrt2 + ple_row * sqrt(hpl) / sqrt2 \
     depends on the token id alone (embed, projection, projection norm and the per-layer \
     table are all frozen), so the rows are computed once per sample, packed into the bucket \
     as an aux channel, and carry no gradient";

pub const EMBED_QUANT_ROW_CHUNK_KEEPS_THE_TRANSIENT_F32_TABLE_UNDER_A_HUNDRED_MEGABYTES: usize =
    8192;

pub const WINDOW_SHARDS_CARRY_THE_HOST_QUANTIZER_OUTPUT_VERBATIM_SO_THE_TAB_TRAINS_THE_SAME_W4:
    &str =
    "the window shards store each trainable-window projection and the tied head as the exact \
     u32 words and f32 scales/biases quantize_mlx_affine_linear produced natively, plus the \
     f32 norms and layer scalars; a tab loads them without a quantizer, so its W4 graph is \
     bitwise the native trainer's, and nothing bf16 (12 GB tower, 5.6 GB per-layer table) has \
     to cross the wasm32 heap";

pub const W4_WORDS: &str = "w4.words";
pub const W4_SCALES: &str = "w4.scales";
pub const W4_BIASES: &str = "w4.biases";
pub const EMBED_MODULE: &str = "model.language_model.embed_tokens";
pub const FINAL_NORM: &str = "model.language_model.norm.weight";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerSource {
    Bf16Checkpoint,
    WindowShard,
}

pub type W4Host = (Vec<u32>, Vec<f32>, Vec<f32>);

#[derive(Clone, Debug)]
pub struct E4bDims {
    pub hidden: usize,
    pub inter: usize,
    pub layers: usize,
    pub heads: usize,
    pub vocab: usize,
    pub hpl: usize,
    pub ple_row: usize,
    pub eps: f32,
    pub sliding_window: usize,
    pub softcap: f32,
    pub max_seq: usize,
    pub first_kv_shared: usize,
    kinds: Vec<LayerType>,
    kv_sources: Vec<Option<usize>>,
    head_dims: [usize; 2],
    kv_heads: [usize; 2],
    thetas: [f32; 2],
    angles: [usize; 2],
}

fn slot(kind: LayerType) -> usize {
    match kind {
        LayerType::SlidingAttention => 0,
        LayerType::FullAttention => 1,
    }
}

impl E4bDims {
    pub fn from_config_json(s: &str) -> Result<Self> {
        let cfg = Gemma4Config::from_hf_json_str(s).context("parse gemma4 config json")?;
        ensure!(
            cfg.has_per_layer_embeddings() && cfg.tie_word_embeddings && !cfg.enable_moe_block,
            "the E4B trainer needs per-layer embeddings, a tied head and no MoE block; got \
             hpl {} tied {} moe {}",
            cfg.hidden_size_per_layer_input,
            cfg.tie_word_embeddings,
            cfg.enable_moe_block
        );
        ensure!(
            cfg.hidden_activation == "gelu_pytorch_tanh",
            "hidden_activation {} is not gelu_pytorch_tanh",
            cfg.hidden_activation
        );
        let kinds = [LayerType::SlidingAttention, LayerType::FullAttention];
        let head_dims = kinds.map(|k| cfg.head_dim_for(k));
        let kv_heads = kinds.map(|k| cfg.num_kv_heads_for(k));
        let thetas = kinds.map(|k| cfg.rope_theta_for(k));
        let angles = kinds.map(|k| {
            ((cfg.rope_partial_factor_for(k) * cfg.head_dim_for(k) as f32 / 2.0) as usize)
                .min(cfg.head_dim_for(k) / 2)
        });
        for hd in head_dims {
            ensure!(hd.is_multiple_of(2), "head_dim {hd} must be even");
        }
        Ok(Self {
            hidden: cfg.hidden_size,
            inter: cfg.intermediate_size,
            layers: cfg.num_hidden_layers,
            heads: cfg.num_attention_heads,
            vocab: cfg.vocab_size,
            hpl: cfg.hidden_size_per_layer_input,
            ple_row: cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
            eps: cfg.rms_norm_eps as f32,
            sliding_window: cfg.sliding_window,
            softcap: cfg.final_logit_softcapping,
            max_seq: cfg.max_position_embeddings,
            first_kv_shared: cfg.first_kv_shared_layer_idx(),
            kinds: cfg.layer_types.clone(),
            kv_sources: (0..cfg.num_hidden_layers)
                .map(|i| cfg.kv_source_layer(i))
                .collect(),
            head_dims,
            kv_heads,
            thetas,
            angles,
        })
    }

    pub fn kind(&self, li: usize) -> LayerType {
        self.kinds[li]
    }

    pub fn head_dim(&self, li: usize) -> usize {
        self.head_dims[slot(self.kinds[li])]
    }

    pub fn kv_heads(&self, li: usize) -> usize {
        self.kv_heads[slot(self.kinds[li])]
    }

    pub fn window(&self, li: usize) -> usize {
        match self.kinds[li] {
            LayerType::SlidingAttention => self.sliding_window,
            LayerType::FullAttention => 0,
        }
    }

    pub fn theta(&self, kind: LayerType) -> f32 {
        self.thetas[slot(kind)]
    }

    pub fn rope_angles(&self, kind: LayerType) -> usize {
        self.angles[slot(kind)]
    }

    pub fn head_dim_of(&self, kind: LayerType) -> usize {
        self.head_dims[slot(kind)]
    }

    pub fn kv_source(&self, li: usize) -> Option<usize> {
        self.kv_sources[li]
    }

    pub fn layer_prefix(&self, li: usize) -> String {
        format!("{PREFIX}.layers.{li}")
    }
}

fn bf16_f32(loader: &WeightLoader, name: &str, expect: usize) -> Result<Vec<f32>> {
    ensure!(
        loader.st_dtype_of(name) == Some(StDtype::BF16),
        "{name}: expected BF16 storage (the qat-w4a16 export is not accepted; {})",
        E4B_TRAINER_QUANTIZES_THE_BF16_TEXT_TOWER_TO_MLX_AFFINE_U4_GS64_AT_LOAD_SO_THE_DOTS_W4_KERNELS_CARRY_IT
    );
    let v = bf16_bytes_f32(loader.raw_bytes(name)?);
    ensure!(v.len() == expect, "{name}: length {} != {expect}", v.len());
    Ok(v)
}

pub fn w4_host_from_bf16(
    loader: &WeightLoader,
    module: &str,
    n: usize,
    k: usize,
) -> Result<W4Host> {
    let _ = E4B_TRAINER_QUANTIZES_THE_BF16_TEXT_TOWER_TO_MLX_AFFINE_U4_GS64_AT_LOAD_SO_THE_DOTS_W4_KERNELS_CARRY_IT;
    let wname = format!("{module}.weight");
    let shape = loader
        .shape_of(&wname)
        .ok_or_else(|| anyhow!("{wname}: missing"))?;
    ensure!(shape == [n, k], "{wname}: shape {shape:?} != [{n}, {k}]");
    ensure!(
        k.is_multiple_of(AFFINE_GROUP),
        "{wname}: k {k} is not a multiple of the affine group {AFFINE_GROUP}"
    );
    let w = bf16_f32(loader, &wname, n * k)?;
    Ok(quantize_mlx_affine_linear(&w, n, k, AFFINE_GROUP))
}

fn le_f32(loader: &WeightLoader, name: &str, expect: usize) -> Result<Vec<f32>> {
    let v: Vec<f32> = match loader.st_dtype_of(name) {
        Some(StDtype::F32) => loader
            .raw_bytes(name)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Some(StDtype::BF16) => bf16_bytes_f32(loader.raw_bytes(name)?),
        other => anyhow::bail!("{name}: expected F32 or BF16 storage, got {other:?}"),
    };
    ensure!(v.len() == expect, "{name}: length {} != {expect}", v.len());
    Ok(v)
}

fn le_u32(loader: &WeightLoader, name: &str, expect: usize) -> Result<Vec<u32>> {
    ensure!(
        loader.st_dtype_of(name) == Some(StDtype::U32),
        "{name}: expected U32 storage"
    );
    let v: Vec<u32> = loader
        .raw_bytes(name)?
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    ensure!(v.len() == expect, "{name}: length {} != {expect}", v.len());
    Ok(v)
}

pub fn w4_host_from_shard(
    loader: &WeightLoader,
    module: &str,
    n: usize,
    k: usize,
) -> Result<W4Host> {
    let _ = WINDOW_SHARDS_CARRY_THE_HOST_QUANTIZER_OUTPUT_VERBATIM_SO_THE_TAB_TRAINS_THE_SAME_W4;
    let groups = n * (k / AFFINE_GROUP);
    Ok((
        le_u32(loader, &format!("{module}.{W4_WORDS}"), n * k / 8)?,
        le_f32(loader, &format!("{module}.{W4_SCALES}"), groups)?,
        le_f32(loader, &format!("{module}.{W4_BIASES}"), groups)?,
    ))
}

pub fn w4_host(
    loader: &WeightLoader,
    module: &str,
    n: usize,
    k: usize,
    source: LayerSource,
) -> Result<W4Host> {
    match source {
        LayerSource::Bf16Checkpoint => w4_host_from_bf16(loader, module, n, k),
        LayerSource::WindowShard => w4_host_from_shard(loader, module, n, k),
    }
}

fn gamma(
    loader: &WeightLoader,
    name: &str,
    expect: usize,
    source: LayerSource,
) -> Result<Vec<f32>> {
    match source {
        LayerSource::Bf16Checkpoint => bf16_f32(loader, name, expect),
        LayerSource::WindowShard => le_f32(loader, name, expect),
    }
}

pub fn w4_load(
    ctx: &WgpuContext,
    loader: &WeightLoader,
    module: &str,
    n: usize,
    k: usize,
    source: LayerSource,
) -> Result<W4Mod> {
    let (words, scales, biases) = w4_host(loader, module, n, k, source)?;
    let gpu = upload_head(ctx, &words, &scales, &biases, n, k)?;
    Ok(W4Mod {
        module: module.to_string(),
        n,
        k,
        words,
        scales,
        biases,
        bias_rows: None,
        gpu,
    })
}

pub struct E4bLayer {
    pub index: usize,
    pub kind: LayerType,
    pub ln_in: Vec<f32>,
    pub ln_pa: Vec<f32>,
    pub ln_pf: Vec<f32>,
    pub ln_po: Vec<f32>,
    pub ln_pli: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub v_norm: Vec<f32>,
    pub layer_scalar: f32,
    pub q: W4Mod,
    pub k: Option<W4Mod>,
    pub v: Option<W4Mod>,
    pub o: W4Mod,
    pub gate: W4Mod,
    pub up: W4Mod,
    pub down: W4Mod,
    pub plgate: W4Mod,
    pub plproj: W4Mod,
    pub kv_source: Option<usize>,
}

pub struct LayerShapes {
    pub modules: Vec<(String, usize, usize)>,
    pub norms: Vec<(String, usize)>,
}

pub fn layer_shapes(d: &E4bDims, li: usize) -> LayerShapes {
    let p = d.layer_prefix(li);
    let h = d.hidden;
    let hd = d.head_dim(li);
    let q_out = d.heads * hd;
    let kv_out = d.kv_heads(li) * hd;
    let mut modules = vec![(format!("{p}.self_attn.q_proj"), q_out, h)];
    let mut norms = vec![
        (format!("{p}.input_layernorm.weight"), h),
        (format!("{p}.post_attention_layernorm.weight"), h),
        (format!("{p}.pre_feedforward_layernorm.weight"), h),
        (format!("{p}.post_feedforward_layernorm.weight"), h),
        (format!("{p}.post_per_layer_input_norm.weight"), h),
        (format!("{p}.self_attn.q_norm.weight"), hd),
        (format!("{p}.layer_scalar"), 1),
    ];
    if d.kv_source(li).is_none() {
        modules.push((format!("{p}.self_attn.k_proj"), kv_out, h));
        modules.push((format!("{p}.self_attn.v_proj"), kv_out, h));
        norms.push((format!("{p}.self_attn.k_norm.weight"), hd));
    }
    modules.extend([
        (format!("{p}.self_attn.o_proj"), h, q_out),
        (format!("{p}.mlp.gate_proj"), d.inter, h),
        (format!("{p}.mlp.up_proj"), d.inter, h),
        (format!("{p}.mlp.down_proj"), h, d.inter),
        (format!("{p}.per_layer_input_gate"), d.hpl, h),
        (format!("{p}.per_layer_projection"), h, d.hpl),
    ]);
    LayerShapes { modules, norms }
}

impl E4bLayer {
    pub fn load(ctx: &WgpuContext, loader: &WeightLoader, d: &E4bDims, li: usize) -> Result<Self> {
        Self::load_from(ctx, loader, d, li, LayerSource::Bf16Checkpoint)
    }

    pub fn load_from(
        ctx: &WgpuContext,
        loader: &WeightLoader,
        d: &E4bDims,
        li: usize,
        src: LayerSource,
    ) -> Result<Self> {
        let p = d.layer_prefix(li);
        let h = d.hidden;
        let hd = d.head_dim(li);
        let q_out = d.heads * hd;
        let kv_out = d.kv_heads(li) * hd;
        let kv_source = d.kv_source(li);
        let w4 = |m: &str, n: usize, k: usize| w4_load(ctx, loader, m, n, k, src);
        let (k, v, k_norm) = if kv_source.is_some() {
            (None, None, vec![1f32; hd])
        } else {
            (
                Some(w4(&format!("{p}.self_attn.k_proj"), kv_out, h)?),
                Some(w4(&format!("{p}.self_attn.v_proj"), kv_out, h)?),
                gamma(loader, &format!("{p}.self_attn.k_norm.weight"), hd, src)?,
            )
        };
        Ok(Self {
            index: li,
            kind: d.kind(li),
            ln_in: gamma(loader, &format!("{p}.input_layernorm.weight"), h, src)?,
            ln_pa: gamma(
                loader,
                &format!("{p}.post_attention_layernorm.weight"),
                h,
                src,
            )?,
            ln_pf: gamma(
                loader,
                &format!("{p}.pre_feedforward_layernorm.weight"),
                h,
                src,
            )?,
            ln_po: gamma(
                loader,
                &format!("{p}.post_feedforward_layernorm.weight"),
                h,
                src,
            )?,
            ln_pli: gamma(
                loader,
                &format!("{p}.post_per_layer_input_norm.weight"),
                h,
                src,
            )?,
            q_norm: gamma(loader, &format!("{p}.self_attn.q_norm.weight"), hd, src)?,
            k_norm,
            v_norm: vec![1f32; hd],
            layer_scalar: gamma(loader, &format!("{p}.layer_scalar"), 1, src)?[0],
            q: w4(&format!("{p}.self_attn.q_proj"), q_out, h)?,
            k,
            v,
            o: w4(&format!("{p}.self_attn.o_proj"), h, q_out)?,
            gate: w4(&format!("{p}.mlp.gate_proj"), d.inter, h)?,
            up: w4(&format!("{p}.mlp.up_proj"), d.inter, h)?,
            down: w4(&format!("{p}.mlp.down_proj"), h, d.inter)?,
            plgate: w4(&format!("{p}.per_layer_input_gate"), d.hpl, h)?,
            plproj: w4(&format!("{p}.per_layer_projection"), h, d.hpl)?,
            kv_source,
        })
    }

    pub fn projections(&self) -> [Option<(&'static str, &W4Mod)>; 7] {
        [
            Some(("q_proj", &self.q)),
            self.k.as_ref().map(|m| ("k_proj", m)),
            self.v.as_ref().map(|m| ("v_proj", m)),
            Some(("o_proj", &self.o)),
            Some(("gate_proj", &self.gate)),
            Some(("up_proj", &self.up)),
            Some(("down_proj", &self.down)),
        ]
    }

    pub fn module_name(&self, leaf: &str) -> String {
        let p = format!("{PREFIX}.layers.{}", self.index);
        match leaf {
            "q_proj" | "k_proj" | "v_proj" | "o_proj" => format!("{p}.self_attn.{leaf}"),
            _ => format!("{p}.mlp.{leaf}"),
        }
    }
}

pub struct E4bTower {
    pub embed_scale: f32,
    pub plmp: W4Mod,
    pub plp_norm: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub head: W4Head,
}

pub fn quantize_embed(loader: &WeightLoader, d: &E4bDims) -> Result<W4Host> {
    let ename = format!("{EMBED_MODULE}.weight");
    let shape = loader
        .shape_of(&ename)
        .ok_or_else(|| anyhow!("{ename}: missing"))?;
    ensure!(
        shape == [d.vocab, d.hidden],
        "{ename}: shape {shape:?} != [{}, {}]",
        d.vocab,
        d.hidden
    );
    ensure!(
        loader.st_dtype_of(&ename) == Some(StDtype::BF16),
        "{ename}: expected BF16 storage"
    );
    let bytes = loader.raw_bytes(&ename)?;
    let groups = d.hidden / AFFINE_GROUP;
    let mut words = Vec::with_capacity(d.vocab * d.hidden / 8);
    let mut scales = Vec::with_capacity(d.vocab * groups);
    let mut biases = Vec::with_capacity(d.vocab * groups);
    let chunk = EMBED_QUANT_ROW_CHUNK_KEEPS_THE_TRANSIENT_F32_TABLE_UNDER_A_HUNDRED_MEGABYTES;
    for r0 in (0..d.vocab).step_by(chunk) {
        let r1 = (r0 + chunk).min(d.vocab);
        let w = bf16_bytes_f32(&bytes[r0 * d.hidden * 2..r1 * d.hidden * 2]);
        let (cw, cs, cb) = quantize_mlx_affine_linear(&w, r1 - r0, d.hidden, AFFINE_GROUP);
        words.extend(cw);
        scales.extend(cs);
        biases.extend(cb);
    }
    Ok((words, scales, biases))
}

pub fn upload_e4b_head(ctx: &WgpuContext, d: &E4bDims, host: &W4Host) -> Result<W4Head> {
    let _ = E4B_HEAD_IS_THE_QUANTIZED_EMBED_TABLE_WITH_THE_FINAL_LOGIT_SOFTCAP_ON;
    let mut head = upload_head(ctx, &host.0, &host.1, &host.2, d.vocab, d.hidden)?;
    head.softcap = d.softcap;
    Ok(head)
}

pub struct E4bWindow {
    pub layers: Vec<E4bLayer>,
    pub final_norm: Vec<f32>,
    pub head: W4Head,
}

impl E4bWindow {
    pub fn load(
        ctx: &WgpuContext,
        loader: &WeightLoader,
        d: &E4bDims,
        window_start: usize,
    ) -> Result<Self> {
        let layers = (window_start..d.layers)
            .map(|li| {
                E4bLayer::load_from(ctx, loader, d, li, LayerSource::WindowShard)
                    .with_context(|| format!("load window layer {li} from the window shard"))
            })
            .collect::<Result<_>>()?;
        let host = w4_host_from_shard(loader, EMBED_MODULE, d.vocab, d.hidden)?;
        Ok(Self {
            layers,
            final_norm: le_f32(loader, FINAL_NORM, d.hidden)?,
            head: upload_e4b_head(ctx, d, &host)?,
        })
    }
}

fn f32_le_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u32_le_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn serialize_blobs(blobs: &[(String, Dtype, Vec<usize>, Vec<u8>)]) -> Result<Vec<u8>> {
    let views = blobs
        .iter()
        .map(|(name, dt, shape, bytes)| {
            TensorView::new(*dt, shape.clone(), bytes).map(|v| (name.as_str(), v))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(safetensors::serialize(views, None)?)
}

fn push_w4(
    blobs: &mut Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
    module: &str,
    n: usize,
    k: usize,
    host: W4Host,
) {
    let groups = k / AFFINE_GROUP;
    blobs.push((
        format!("{module}.{W4_WORDS}"),
        Dtype::U32,
        vec![n, k / 8],
        u32_le_bytes(&host.0),
    ));
    blobs.push((
        format!("{module}.{W4_SCALES}"),
        Dtype::F32,
        vec![n, groups],
        f32_le_bytes(&host.1),
    ));
    blobs.push((
        format!("{module}.{W4_BIASES}"),
        Dtype::F32,
        vec![n, groups],
        f32_le_bytes(&host.2),
    ));
}

pub fn window_layers_shard_bytes(
    loader: &WeightLoader,
    d: &E4bDims,
    window_start: usize,
) -> Result<Vec<u8>> {
    let _ = WINDOW_SHARDS_CARRY_THE_HOST_QUANTIZER_OUTPUT_VERBATIM_SO_THE_TAB_TRAINS_THE_SAME_W4;
    let mut blobs = Vec::new();
    for li in window_start..d.layers {
        let shapes = layer_shapes(d, li);
        for (module, n, k) in &shapes.modules {
            push_w4(
                &mut blobs,
                module,
                *n,
                *k,
                w4_host_from_bf16(loader, module, *n, *k)?,
            );
        }
        for (name, len) in &shapes.norms {
            blobs.push((
                name.clone(),
                Dtype::F32,
                vec![*len],
                f32_le_bytes(&bf16_f32(loader, name, *len)?),
            ));
        }
    }
    blobs.push((
        FINAL_NORM.to_string(),
        Dtype::F32,
        vec![d.hidden],
        f32_le_bytes(&bf16_f32(loader, FINAL_NORM, d.hidden)?),
    ));
    serialize_blobs(&blobs)
}

pub fn window_head_shard_bytes(loader: &WeightLoader, d: &E4bDims) -> Result<Vec<u8>> {
    let mut blobs = Vec::new();
    push_w4(
        &mut blobs,
        EMBED_MODULE,
        d.vocab,
        d.hidden,
        quantize_embed(loader, d)?,
    );
    serialize_blobs(&blobs)
}

impl E4bTower {
    pub fn load(ctx: &WgpuContext, loader: &WeightLoader, d: &E4bDims) -> Result<Self> {
        let head = upload_e4b_head(ctx, d, &quantize_embed(loader, d)?)?;
        Ok(Self {
            embed_scale: (d.hidden as f32).sqrt(),
            plmp: w4_load(
                ctx,
                loader,
                &format!("{PREFIX}.per_layer_model_projection"),
                d.ple_row,
                d.hidden,
                LayerSource::Bf16Checkpoint,
            )?,
            plp_norm: bf16_f32(
                loader,
                &format!("{PREFIX}.per_layer_projection_norm.weight"),
                d.hpl,
            )?,
            final_norm: bf16_f32(loader, FINAL_NORM, d.hidden)?,
            head,
        })
    }
}

fn bf16_rows(
    loader: &WeightLoader,
    name: &str,
    ids: &[u32],
    rows: usize,
    width: usize,
) -> Result<Vec<f32>> {
    let shape = loader
        .shape_of(name)
        .ok_or_else(|| anyhow!("{name}: missing"))?;
    ensure!(
        shape == [rows, width],
        "{name}: shape {shape:?} != [{rows}, {width}]"
    );
    ensure!(
        loader.st_dtype_of(name) == Some(StDtype::BF16),
        "{name}: expected BF16 storage"
    );
    let bytes = loader.raw_bytes(name)?;
    let mut out = Vec::with_capacity(ids.len() * width);
    for &id in ids {
        let r = id as usize;
        ensure!(r < rows, "token id {id} out of range {rows}");
        out.extend(bf16_bytes_f32(&bytes[r * width * 2..(r + 1) * width * 2]));
    }
    Ok(out)
}

pub fn scaled_embed_rows(
    loader: &WeightLoader,
    d: &E4bDims,
    tower: &E4bTower,
    ids: &[u32],
) -> Result<Vec<f32>> {
    let mut rows = bf16_rows(
        loader,
        &format!("{EMBED_MODULE}.weight"),
        ids,
        d.vocab,
        d.hidden,
    )?;
    for v in rows.iter_mut() {
        *v *= tower.embed_scale;
    }
    Ok(rows)
}

pub fn per_layer_raw_rows(loader: &WeightLoader, d: &E4bDims, ids: &[u32]) -> Result<Vec<f32>> {
    let name = format!("{PREFIX}.embed_tokens_per_layer.weight");
    let rows = loader
        .shape_of(&name)
        .map(|s| s[0])
        .ok_or_else(|| anyhow!("{name}: missing"))?;
    bf16_rows(loader, &name, ids, rows, d.ple_row)
}
