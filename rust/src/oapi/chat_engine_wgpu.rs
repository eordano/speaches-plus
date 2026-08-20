pub mod batch;
pub mod learned;
pub mod mem_fit;
pub mod mm;
pub mod persist;
pub mod spec;

pub use crate::oapi::lora;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context as _;
use rand_core::Rng;
use rand_core::SeedableRng;
use rand_pcg::Pcg64;
use tokio::sync::mpsc;

use crate::oapi::chat::{
    ChatEngine, ChatEvent, ChatGenerateRequest, ChatMessageIn, LogprobEntry, TemplateKwargs, Tool,
    ToolChoice, TopLogprob,
};
use crate::oapi::chat_engine::ChatRegistry;
use crate::oapi::chat_template::ChatTemplate;

use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu;
use nv_models::gemma4_wgpu::{host_weights_from_loader, Gemma4Wgpu};
use nv_models::gpt_oss_wgpu::{GptOssConfig, GptOssWgpu};
use nv_models::laguna::LagunaConfig;
use nv_models::laguna_wgpu::LagunaWgpu;
use nv_models::prefix_reuse::RewindLimits;
use nv_models::qwen3_5_dense_wgpu::{Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use nv_models::qwen3_5_moe::Qwen3MoeConfig;
use nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu;

pub const DEFAULT_MAX_SEQ_CAP_COVERS_A_1792PX_IMAGE_PROMPT_PLUS_REPLY: usize = 8192;
pub const DEFAULT_MAX_SEQ: usize = DEFAULT_MAX_SEQ_CAP_COVERS_A_1792PX_IMAGE_PROMPT_PLUS_REPLY;
pub const DEFAULT_MAX_NEW_TOKENS: usize = 512;
pub const MAX_SEQ_ENV: &str = "NV_WGPU_CHAT_MAX_SEQ";
pub const WARMUP_ENV: &str = "NV_WGPU_WARMUP";

pub const PREFIX_REUSE_ENV: &str = "NV_WGPU_PREFIX_REUSE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WgpuModelKind {
    Gemma4Dense,
    Gemma4E4b,
    Gemma4Moe,
    Qwen3_5Moe,
    Qwen3_5Dense,
    GptOss,
    Laguna,
}

impl WgpuModelKind {
    pub const ALL: [Self; 7] = [
        Self::Gemma4Dense,
        Self::Gemma4E4b,
        Self::Gemma4Moe,
        Self::Qwen3_5Moe,
        Self::Qwen3_5Dense,
        Self::GptOss,
        Self::Laguna,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Gemma4Dense => "gemma4-dense (nv-models::gemma4_wgpu)",
            Self::Gemma4E4b => "gemma4-e4b (nv-models::gemma4_e4b_wgpu)",
            Self::Gemma4Moe => "gemma4-moe (nv-models::gemma4_moe_wgpu)",
            Self::Qwen3_5Moe => "qwen3.5-moe (nv-models::qwen3_5_moe_wgpu)",
            Self::Qwen3_5Dense => "qwen3.5-dense (nv-models::qwen3_5_dense_wgpu)",
            Self::GptOss => "gpt-oss (nv-models::gpt_oss_wgpu)",
            Self::Laguna => "laguna-xs (nv-models::laguna_wgpu)",
        }
    }

    pub fn spec_chain_slug(self) -> &'static str {
        match self {
            Self::Gemma4E4b => "gemma4-e4b",
            Self::Qwen3_5Dense => "qwen3.5-dense",
            Self::GptOss => "gpt-oss",
            Self::Gemma4Dense | Self::Gemma4Moe | Self::Qwen3_5Moe | Self::Laguna => "",
        }
    }
}

pub fn classify_wgpu_model(raw_cfg: &str) -> anyhow::Result<WgpuModelKind> {
    let v: serde_json::Value =
        serde_json::from_str(raw_cfg).map_err(|e| anyhow::anyhow!("parse config.json: {e}"))?;
    let mut tags: Vec<String> = Vec::new();
    if let Some(arr) = v.get("architectures").and_then(|x| x.as_array()) {
        tags.extend(
            arr.iter()
                .filter_map(|a| a.as_str())
                .map(|s| s.to_ascii_lowercase()),
        );
    }
    if let Some(mt) = v.get("model_type").and_then(|x| x.as_str()) {
        tags.push(mt.to_ascii_lowercase());
    }
    let is = |pat: &str| tags.iter().any(|t| t.starts_with(pat));
    if is("qwen3_5moe") || is("qwen3_5_moe") || is("qwen3.5moe") || is("qwen3.5_moe") {
        return Ok(WgpuModelKind::Qwen3_5Moe);
    }
    if is("qwen3_5") || is("qwen3.5") {
        return Ok(WgpuModelKind::Qwen3_5Dense);
    }
    if is("gpt_oss") || is("gptoss") || is("gpt-oss") {
        return Ok(WgpuModelKind::GptOss);
    }
    if is("laguna") {
        return Ok(WgpuModelKind::Laguna);
    }
    if is("gemma4") {
        let cfg = Gemma4Config::from_hf_json_str(raw_cfg).context("parse gemma4 config")?;
        if cfg.enable_moe_block {
            return Ok(WgpuModelKind::Gemma4Moe);
        }
        return Ok(if cfg.has_per_layer_embeddings() {
            WgpuModelKind::Gemma4E4b
        } else {
            WgpuModelKind::Gemma4Dense
        });
    }

    if is("diffusion_gemma") || is("diffusiongemma") {
        anyhow::bail!(
            "DiffusionGemma is not autoregressive and cannot be served by this engine yet. \
             Its text_config matches google/gemma-4-26B-A4B-it exactly, so the weights would \
             load and the forward pass would run -- and produce garbage: the model was trained \
             to iteratively denoise a masked {canvas}-token canvas \
             (architectures DiffusionGemmaForBlockDiffusion, model_type diffusion_gemma), not \
             to predict the next token. Serving it needs a block-diffusion decode loop.",
            canvas = v
                .get("canvas_length")
                .and_then(|x| x.as_u64())
                .unwrap_or(256)
        )
    }
    anyhow::bail!(
        "wgpu chat serving supports gemma4 (dense + E4B), qwen3_5_moe, qwen3_5 dense, \
         gpt_oss and laguna only; config.json advertises {tags:?}"
    )
}

pub use crate::oapi::model_ids::model_id_for_dir;

pub fn gguf_config_json(gguf: &Path) -> anyhow::Result<String> {
    let loader = nv_weights::GgufLoader::open(gguf, &candle_core::Device::Cpu)
        .with_context(|| format!("open gguf {}", gguf.display()))?;
    nv_models::gemma4_gguf::gemma4_moe_config_json_from_gguf(&loader)
        .with_context(|| format!("synthesize config from {}", gguf.display()))
}

pub fn ensure_serving_sidecars(model_dir: &Path) -> anyhow::Result<()> {
    let missing = nv_weights::gguf::missing_gguf_sidecars(model_dir);
    if missing.is_empty() {
        return Ok(());
    }
    let Some(gguf) = nv_weights::gguf::lone_gguf_file(model_dir) else {
        return Ok(());
    };
    let written = nv_weights::gguf::ensure_gguf_sidecars(model_dir).with_context(|| {
        format!(
            "{} holds a bare GGUF checkpoint and none of {missing:?}; serving needs them and \
             they were synthesized from {}'s own metadata, not copied from another size -- the \
             chat template differs between gemma-4 sizes",
            model_dir.display(),
            gguf.display()
        )
    })?;
    if !written.is_empty() {
        tracing::info!(
            dir = %model_dir.display(),
            gguf = %gguf.display(),
            files = ?written,
            "synthesized serving sidecars from gguf metadata"
        );
    }
    Ok(())
}

fn gguf_eos_ids(gguf: &Path) -> Option<Vec<u32>> {
    let loader = nv_weights::GgufLoader::open(gguf, &candle_core::Device::Cpu).ok()?;
    let mut out: Vec<u32> = ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"]
        .iter()
        .filter_map(|k| loader.md_u64(k).ok())
        .map(|x| x as u32)
        .collect();
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

pub fn eos_ids_for_serving(dir: &Path) -> anyhow::Result<Vec<u32>> {
    match eos_ids_from_dir(dir) {
        Ok(ids) => Ok(ids),
        Err(sidecar_err) => {
            match nv_weights::gguf::lone_gguf_file(dir).and_then(|gguf| gguf_eos_ids(&gguf)) {
                Some(ids) => Ok(ids),
                None => Err(sidecar_err),
            }
        }
    }
}

pub fn eos_ids_from_dir(dir: &Path) -> anyhow::Result<Vec<u32>> {
    let mut out: Vec<u32> = Vec::new();
    for (file, key) in [
        ("generation_config.json", "eos_token_id"),
        ("config.json", "eos_token_id"),
    ] {
        let Ok(raw) = std::fs::read_to_string(dir.join(file)) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
            continue;
        };
        for scope in [Some(&v), v.get("text_config")].into_iter().flatten() {
            match scope.get(key) {
                Some(serde_json::Value::Number(n)) => {
                    if let Some(x) = n.as_u64() {
                        out.push(x as u32);
                    }
                }
                Some(serde_json::Value::Array(a)) => {
                    out.extend(a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32));
                }
                _ => {}
            }
        }
        if !out.is_empty() {
            out.sort_unstable();
            out.dedup();
            return Ok(out);
        }
    }
    anyhow::bail!(
        "no eos_token_id in {}/generation_config.json or config.json: refusing to serve a \
         model with no stop condition",
        dir.display()
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BosDeclaration {
    Absent,

    Suppressed,
    Token(String),
}

impl BosDeclaration {
    pub fn label(&self) -> String {
        match self {
            Self::Absent => "absent".to_string(),
            Self::Suppressed => "suppressed (add_bos_token: false)".to_string(),
            Self::Token(t) => format!("bos_token {t:?}"),
        }
    }
}

pub fn bos_declaration_from_json(raw: &str) -> BosDeclaration {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) else {
        return BosDeclaration::Absent;
    };
    if v.get("add_bos_token").and_then(|x| x.as_bool()) == Some(false) {
        return BosDeclaration::Suppressed;
    }
    match v.get("bos_token") {
        Some(serde_json::Value::String(s)) if !s.is_empty() => BosDeclaration::Token(s.clone()),
        Some(serde_json::Value::Object(o)) => o
            .get("content")
            .and_then(|c| c.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| BosDeclaration::Token(s.to_string()))
            .unwrap_or(BosDeclaration::Absent),
        _ => BosDeclaration::Absent,
    }
}

pub fn prompt_bos_id<F>(
    declaration: &BosDeclaration,
    token_to_id: &F,
    eos_ids: &[u32],
    model_id: &str,
) -> Option<u32>
where
    F: Fn(&str) -> Option<u32> + ?Sized,
{
    let token = match declaration {
        BosDeclaration::Absent | BosDeclaration::Suppressed => return None,
        BosDeclaration::Token(t) => t,
    };
    let Some(id) = token_to_id(token) else {
        tracing::warn!(
            model = model_id,
            token = token.as_str(),
            "tokenizer_config declares a bos_token that is not in the vocabulary; not prepending"
        );
        return None;
    };
    if eos_ids.contains(&id) {
        tracing::warn!(
            model = model_id,
            token = token.as_str(),
            id,
            "declared bos_token is also an EOS member; not prepending it"
        );
        return None;
    }
    Some(id)
}

#[derive(Debug, Clone)]
pub struct PromptHeadProbe {
    pub model_id: String,
    pub bos_declaration: BosDeclaration,
    pub eos_ids: Vec<u32>,
    pub generation_config_bos: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub prompt_ids: Vec<u32>,
    pub legacy_prompt_ids: Vec<u32>,
}

pub fn probe_prompt_head(
    model_dir: &Path,
    messages: &[ChatMessageIn],
) -> anyhow::Result<PromptHeadProbe> {
    ensure_serving_sidecars(model_dir)?;
    let template = ChatTemplate::load_reason(model_dir)
        .map_err(|reason| anyhow::anyhow!("no official chat template: {reason}"))?;
    let mut tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    nv_tokenizer::sanitize_for_serving(&mut tokenizer);

    let eos_ids = eos_ids_for_serving(model_dir)?;
    let generation_config_bos = bos_id_from_dir(model_dir);
    let bos_declaration = std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
        .map(|raw| bos_declaration_from_json(&raw))
        .unwrap_or(BosDeclaration::Absent);
    let model_id = model_id_for_dir(model_dir);

    let lookup = |t: &str| tokenizer.token_to_id(t);
    let bos_token_id = prompt_bos_id(&bos_declaration, &lookup, &eos_ids, &model_id);

    let rendered = render_official_with_tools(&template, messages, &[])?;
    let encoded = tokenizer
        .encode(rendered.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let base: Vec<u32> = encoded.get_ids().to_vec();
    anyhow::ensure!(
        !base.is_empty(),
        "official template rendered an empty prompt"
    );

    let prepend = |ids: &mut Vec<u32>, bos: Option<u32>| {
        if let Some(b) = bos {
            if ids.first().copied() != Some(b) {
                ids.insert(0, b);
            }
        }
    };
    let mut prompt_ids = base.clone();
    prepend(&mut prompt_ids, bos_token_id);
    let mut legacy_prompt_ids = base;
    prepend(&mut legacy_prompt_ids, generation_config_bos);

    Ok(PromptHeadProbe {
        model_id,
        bos_declaration,
        eos_ids,
        generation_config_bos,
        bos_token_id,
        prompt_ids,
        legacy_prompt_ids,
    })
}

pub fn bos_id_from_dir(dir: &Path) -> Option<u32> {
    for file in ["generation_config.json", "config.json"] {
        let Ok(raw) = std::fs::read_to_string(dir.join(file)) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
            continue;
        };
        for scope in [Some(&v), v.get("text_config")].into_iter().flatten() {
            if let Some(n) = scope.get("bos_token_id").and_then(|x| x.as_u64()) {
                return Some(n as u32);
            }
        }
    }
    None
}

fn kv_disk(d: &Decoder) -> &dyn persist::KvDisk {
    match d {
        Decoder::Gemma4Dense(m) => m.as_ref(),
        Decoder::Gemma4E4b(m) => m.as_ref(),
        Decoder::Gemma4Moe(m) => m.as_ref(),
        Decoder::Qwen3_5Moe(m) => m.as_ref(),
        Decoder::Qwen3_5Dense(m) => m.as_ref(),
        Decoder::GptOss(m) => m.as_ref(),
        Decoder::Laguna(m) => m.as_ref(),
    }
}

fn kv_disk_mut(d: &mut Decoder) -> &mut dyn persist::KvDisk {
    match d {
        Decoder::Gemma4Dense(m) => m.as_mut(),
        Decoder::Gemma4E4b(m) => m.as_mut(),
        Decoder::Gemma4Moe(m) => m.as_mut(),
        Decoder::Qwen3_5Moe(m) => m.as_mut(),
        Decoder::Qwen3_5Dense(m) => m.as_mut(),
        Decoder::GptOss(m) => m.as_mut(),
        Decoder::Laguna(m) => m.as_mut(),
    }
}

fn resume_response_snapshot(
    decoder: &mut Decoder,
    prefix: &mut PrefixCache,
    meta: &persist::Meta,
    id: &str,
) {
    let Some(dir) = persist::cache_dir() else {
        return;
    };
    let path = persist::response_snapshot_path(&dir, id);
    let Some((tokens, frontier)) = persist::peek_stream(&path) else {
        tracing::warn!(id, "response resume: no readable snapshot; serving cold");
        return;
    };
    if prefix.frontier == frontier
        && nv_models::prefix_reuse::common_prefix_len(&prefix.tokens, &tokens) >= frontier
    {
        return;
    }
    *prefix = PrefixCache::default();
    if let Err(e) = decoder.reset() {
        tracing::warn!(id, error = format!("{e:#}"), "response resume: reset failed; serving cold");
        return;
    }
    match persist::restore_file(&path, meta, kv_disk_mut(decoder)) {
        Ok(Some((tokens, frontier))) => {
            tracing::info!(id, frontier, "response snapshot resumed");
            *prefix = PrefixCache { tokens, frontier };
        }
        Ok(None) => tracing::warn!(id, "response resume: snapshot vanished; serving cold"),
        Err(e) => {
            tracing::warn!(id, error = format!("{e:#}"), "response resume rejected; serving cold");
        }
    }
}

fn store_response_snapshot(
    decoder: &Decoder,
    prefix: &PrefixCache,
    meta: &persist::Meta,
    id: &str,
) {
    let Some(dir) = persist::cache_dir() else {
        return;
    };
    let path = persist::response_snapshot_path(&dir, id);
    persist::save_file(&path, meta, kv_disk(decoder), &prefix.tokens, prefix.frontier);
    persist::gc_response_store(&dir);
}

enum Decoder {
    Gemma4Dense(Box<Gemma4Wgpu>),
    Gemma4E4b(Box<Gemma4E4bWgpu>),
    Gemma4Moe(Box<Gemma4MoeWgpu>),
    Qwen3_5Moe(Box<Qwen3MoeWgpu>),
    Qwen3_5Dense(Box<Qwen3_5DenseWgpu>),
    GptOss(Box<GptOssWgpu>),
    Laguna(Box<LagunaWgpu>),
}

impl Decoder {
    fn reset(&mut self) -> anyhow::Result<()> {
        match self {
            Self::Gemma4Dense(m) => m.reset(),
            Self::Gemma4E4b(m) => m.reset(),
            Self::Gemma4Moe(m) => m.reset()?,
            Self::Qwen3_5Moe(m) => m.reset()?,
            Self::Qwen3_5Dense(m) => m.reset()?,
            Self::GptOss(m) => m.reset()?,
            Self::Laguna(m) => m.reset()?,
        }
        Ok(())
    }

    fn prefill_chunk_len(&self) -> usize {
        match self {
            Self::Gemma4E4b(m) => m.prefill_chunk_len(),
            Self::Qwen3_5Dense(m) => m.prefill_chunk_len(),
            Self::Gemma4Dense(m) => m.prefill_chunk_len(),
            Self::Gemma4Moe(m) => m.prefill_chunk_len(),
            Self::GptOss(m) => m.prefill_chunk_len(),
            Self::Qwen3_5Moe(m) => m.prefill_chunk_len(),
            Self::Laguna(m) => m.prefill_chunk_len(),
        }
    }

    fn prefill_tokens(&mut self, tokens: &[u32]) -> anyhow::Result<usize> {
        match self {
            Self::Gemma4E4b(m) => m.prefill_tokens(tokens),
            Self::Qwen3_5Dense(m) => m.prefill_tokens(tokens),
            Self::Gemma4Dense(m) => m.prefill_tokens(tokens),
            Self::Gemma4Moe(m) => m.prefill_tokens(tokens),
            Self::GptOss(m) => m.prefill_tokens(tokens),
            Self::Qwen3_5Moe(m) => m.prefill_tokens(tokens),
            Self::Laguna(m) => m.prefill_tokens(tokens),
        }
    }

    fn kind(&self) -> WgpuModelKind {
        match self {
            Self::Gemma4Dense(_) => WgpuModelKind::Gemma4Dense,
            Self::Gemma4E4b(_) => WgpuModelKind::Gemma4E4b,
            Self::Gemma4Moe(_) => WgpuModelKind::Gemma4Moe,
            Self::Qwen3_5Moe(_) => WgpuModelKind::Qwen3_5Moe,
            Self::Qwen3_5Dense(_) => WgpuModelKind::Qwen3_5Dense,
            Self::GptOss(_) => WgpuModelKind::GptOss,
            Self::Laguna(_) => WgpuModelKind::Laguna,
        }
    }

    fn install_qwen3_mrope_rows(
        &mut self,
        pos: &nv_models::qwen3_mm_splice::Qwen3MropePositions,
        section: [usize; 3],
    ) -> anyhow::Result<()> {
        match self {
            Self::Qwen3_5Dense(m) => {
                m.install_mrope_rows_for_prompt_and_shift_continuation_rows_by_delta(pos, section)
            }
            other => anyhow::bail!(
                "{}: mrope rope rows are wired only for the qwen3.5-dense wgpu graph",
                other.kind().label()
            ),
        }
    }

    fn prefill_tokens_with_embed_rows(
        &mut self,
        tokens: &[u32],
        splices: &[nv_models::embed_row_splice::EmbedRowSplice],
    ) -> anyhow::Result<usize> {
        match self {
            Self::Qwen3_5Dense(m) => m.prefill_tokens_with_image_rows(tokens, splices),
            Self::GptOss(m) => m.prefill_tokens_with_image_rows(tokens, splices),
            Self::Gemma4Dense(m) => m.prefill_tokens_with_embed_rows(tokens, splices),
            Self::Gemma4Moe(m) => m.prefill_tokens_with_embed_rows(tokens, splices),
            other => {
                let kind = other.kind();
                anyhow::bail!(
                    "{}: {}",
                    kind.label(),
                    mm::embed_row_route(kind)
                        .err()
                        .unwrap_or("embed-row prefill is not dispatched for this kind")
                )
            }
        }
    }

    fn step(&mut self, token: u32) -> anyhow::Result<u32> {
        match self {
            Self::Gemma4Dense(m) => m.decode_step(token),
            Self::Gemma4E4b(m) => m.decode_step(token),
            Self::Gemma4Moe(m) => m.decode_step(token),
            Self::Qwen3_5Moe(m) => m.decode_step(token),
            Self::Qwen3_5Dense(m) => m.decode_step(token),
            Self::GptOss(m) => m.decode_step(token),
            Self::Laguna(m) => m.decode_step(token),
        }
    }

    fn prefill_step(&mut self, token: u32) -> anyhow::Result<()> {
        match self {
            Self::Gemma4Dense(m) => m.prefill_step(token),
            Self::Gemma4E4b(m) => m.decode_step(token).map(|_| ()),
            Self::Gemma4Moe(m) => m.prefill_step(token),
            Self::Qwen3_5Moe(m) => m.prefill_step(token),
            Self::Qwen3_5Dense(m) => m.prefill_step(token),
            Self::GptOss(m) => m.prefill_step(token),
            Self::Laguna(m) => m.prefill_step(token),
        }
    }

    fn step_logits(&mut self, token: u32) -> anyhow::Result<(u32, Vec<f32>)> {
        match self {
            Self::Gemma4Dense(m) => m.decode_step_logits(token),
            Self::Gemma4E4b(m) => m.decode_step_logits(token),
            Self::Gemma4Moe(m) => m.decode_step_logits(token),
            Self::Qwen3_5Moe(m) => m.decode_step_logits(token),
            Self::Qwen3_5Dense(m) => m.decode_step_logits(token),
            Self::GptOss(m) => m.decode_step_logits(token),
            Self::Laguna(m) => m.decode_step_logits(token),
        }
    }

    fn pass_count(&self) -> usize {
        match self {
            Self::Gemma4Dense(m) => m.pass_count(),
            Self::Gemma4E4b(m) => m.pass_count(),
            Self::Gemma4Moe(m) => m.pass_count(),
            Self::Qwen3_5Moe(m) => m.pass_count(),
            Self::Qwen3_5Dense(m) => m.pass_count(),
            Self::GptOss(m) => m.pass_count(),
            Self::Laguna(m) => m.pass_count(),
        }
    }

    fn current_pos(&self) -> usize {
        match self {
            Self::Gemma4Dense(m) => m.current_pos(),
            Self::Gemma4E4b(m) => m.current_pos(),
            Self::Gemma4Moe(m) => m.current_pos(),
            Self::Qwen3_5Moe(m) => m.current_pos(),
            Self::Qwen3_5Dense(m) => m.current_pos(),
            Self::GptOss(m) => m.current_pos(),
            Self::Laguna(m) => m.current_pos(),
        }
    }

    fn kind_label(&self) -> &'static str {
        match self {
            Self::Gemma4Dense(_) => WgpuModelKind::Gemma4Dense.label(),
            Self::Gemma4E4b(_) => WgpuModelKind::Gemma4E4b.label(),
            Self::Gemma4Moe(_) => WgpuModelKind::Gemma4Moe.label(),
            Self::Qwen3_5Moe(_) => WgpuModelKind::Qwen3_5Moe.label(),
            Self::Qwen3_5Dense(_) => WgpuModelKind::Qwen3_5Dense.label(),
            Self::GptOss(_) => WgpuModelKind::GptOss.label(),
            Self::Laguna(_) => WgpuModelKind::Laguna.label(),
        }
    }

    fn chain_verify_geometry(&self) -> (usize, usize) {
        match self {
            Self::Gemma4E4b(m) => (m.verify_max_rows(), m.prefill_chunk_len()),
            Self::Qwen3_5Dense(m) => (m.verify_max_rows(), m.prefill_chunk_len()),
            Self::GptOss(m) => (m.verify_max_rows(), m.prefill_chunk_len()),
            _ => (0, 0),
        }
    }

    fn has_chain_verify(&self) -> bool {
        matches!(
            self,
            Self::Gemma4E4b(_) | Self::Qwen3_5Dense(_) | Self::GptOss(_)
        )
    }

    fn rewind_limits(&self) -> RewindLimits {
        match self {
            Self::Gemma4Dense(m) => m.rewind_limits(),
            Self::Gemma4E4b(m) => m.rewind_limits(),
            Self::Gemma4Moe(_)
            | Self::Qwen3_5Moe(_)
            | Self::Qwen3_5Dense(_)
            | Self::GptOss(_)
            | Self::Laguna(_) => RewindLimits::NONE,
        }
    }

    fn rewind_to(&mut self, pos: usize) -> anyhow::Result<bool> {
        match self {
            Self::Gemma4Dense(m) => m.rewind_to(pos),
            Self::Gemma4E4b(m) => m.rewind_to(pos),
            Self::Gemma4Moe(_)
            | Self::Qwen3_5Moe(_)
            | Self::Qwen3_5Dense(_)
            | Self::GptOss(_)
            | Self::Laguna(_) => Ok(false),
        }
    }
}

#[derive(Default)]
struct PrefixCache {
    tokens: Vec<u32>,
    frontier: usize,
}

fn prefix_reuse_enabled() -> bool {
    std::env::var(PREFIX_REUSE_ENV).ok().as_deref() == Some("1")
}

fn plan_prefix_reuse(
    decoder: &mut Decoder,
    prev: &PrefixCache,
    prompt_ids: &[u32],
) -> anyhow::Result<usize> {
    if prev.tokens.is_empty() || !prefix_reuse_enabled() {
        return Ok(0);
    }
    let lcp = nv_models::prefix_reuse::common_prefix_len(&prev.tokens, prompt_ids)
        .min(prompt_ids.len() - 1);
    if let Some(target) =
        nv_models::prefix_reuse::exact_extend_target(prev.frontier, lcp, decoder.current_pos())
    {
        return Ok(target);
    }
    let Some(target) = decoder.rewind_limits().target(prev.frontier, lcp) else {
        return Ok(0);
    };
    Ok(if decoder.rewind_to(target)? { target } else { 0 })
}

impl batch::BatchStepper for Decoder {
    fn batch_capacity(&self) -> usize {
        match self {
            Self::Gemma4Dense(m) => m.batch_slots().max(1),
            _ => 1,
        }
    }

    fn reset_batch(&mut self, slots: usize) -> anyhow::Result<()> {
        match self {
            Self::Gemma4Dense(m) => {
                anyhow::ensure!(
                    slots <= m.batch_slots(),
                    "batch of {slots} on a graph built for {}",
                    m.batch_slots()
                );
                for s in 0..m.batch_slots() {
                    m.reset_slot(s)?;
                }
                Ok(())
            }
            other => anyhow::bail!("{}", batch_route_refusal(other.kind())),
        }
    }

    fn prefill_slot(&mut self, slot: usize, tokens: &[u32]) -> anyhow::Result<u32> {
        match self {
            Self::Gemma4Dense(m) => m.prefill_slot(slot, tokens),
            other => anyhow::bail!("{}", batch_route_refusal(other.kind())),
        }
    }

    fn decode_step_batch(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<u32>> {
        match self {
            Self::Gemma4Dense(m) => m.decode_step_batch(tokens),
            other => anyhow::bail!("{}", batch_route_refusal(other.kind())),
        }
    }

    fn end_batch(&mut self) {
        if let Self::Gemma4Dense(m) = self {
            let _ = m.select_slot(0);
        }
    }
}

pub const BATCH_GRAPH_SEAM: &str = "this wgpu graph holds one KV region: batched serving needs \
     reset_slot(slot) + prefill_slot(slot, tokens) alongside decode_step_batch(tokens), because a \
     batched step can advance sequences it has no way to create";

pub fn batch_route_gap(kind: WgpuModelKind) -> Option<batch::BatchGap> {
    match kind {
        WgpuModelKind::Gemma4Dense => None,
        WgpuModelKind::Gemma4E4b | WgpuModelKind::Gemma4Moe | WgpuModelKind::GptOss => {
            Some(batch::BatchGap::ATTENTION_KV_IN_ONE_REGION)
        }
        WgpuModelKind::Qwen3_5Dense | WgpuModelKind::Qwen3_5Moe | WgpuModelKind::Laguna => {
            Some(batch::BatchGap::RECURRENT_STATE_IN_ONE_REGION)
        }
    }
}

pub fn batch_route_refusal(kind: WgpuModelKind) -> String {
    match batch_route_gap(kind) {
        Some(gap) => format!("{}: {gap}; {BATCH_GRAPH_SEAM}", kind.label()),
        None => format!(
            "{}: this kind has a batched decode graph; a refusal here is a slot or capacity \
             fault, not a missing route",
            kind.label()
        ),
    }
}

fn batch_graph_slots(model_dir: &Path, raw_cfg: &str, max_seq: usize) -> usize {
    let knobs = batch::BatchKnobs::from_env();
    if !knobs.enabled() {
        return 0;
    }
    let (Some(kv), Some(est)) = (
        batch::kv_geometry_from_config(raw_cfg, batch::KV_ELEM_BYTES),
        mem_fit::estimate_model_with_max_seq(model_dir, max_seq),
    ) else {
        tracing::warn!(
            model = %model_dir.display(),
            "batched serving refused: the KV geometry or the weight size could not be read, and \
             admission must not guess a slot on a platform that SIGKILLs instead of failing"
        );
        return 0;
    };
    let admission = batch::Admission {
        budget: batch::StepBudget {
            weight_bytes: (est.weight_gib * (1u64 << 30) as f64)
                as u64,
            kv,
        },
        max_seq: max_seq as u64,
        knobs,
    };

    let free = mem_fit::available_memory_gib().map(|a| a - est.load_peak_gib);
    let slots = admission
        .slots(nv_models::gemma4_wgpu::MK_MAX, free)
        .min(nv_models::gemma4_wgpu::MK_MAX);
    if slots >= 2 {
        slots
    } else {
        0
    }
}

pub fn batch_admits(kind: WgpuModelKind, req: &ChatGenerateRequest) -> Result<(), batch::Refusal> {
    if req.mm.as_ref().is_some_and(mm::media_present) {
        return Err(batch::Refusal::MmSplice);
    }
    if HostSampler::new(req, 0).needs_logits() {
        return Err(batch::Refusal::NeedsHostLogits);
    }
    if spec_route_eligible(kind, false, spec::SpecKnobs::from_env())
        || (kind == WgpuModelKind::Gemma4E4b && nv_models::gemma4_e4b_wgpu::chain_k_from_env() > 1)
    {
        return Err(batch::Refusal::MultiRowRoute);
    }
    if batch_route_gap(kind).is_some() {
        return Err(batch::Refusal::KindHasNoBatchGraph);
    }
    Ok(())
}

pub const MTP_ENV_OPTS_IN_THE_CHAIN_ROUTE_BECAUSE_THE_MEASURED_CHAIN_LOSS_WAS_DRAFTERLESS: &str =
    "NV_Q3D_MTP";

pub fn mtp_opted_in() -> bool {
    matches!(
        std::env::var(MTP_ENV_OPTS_IN_THE_CHAIN_ROUTE_BECAUSE_THE_MEASURED_CHAIN_LOSS_WAS_DRAFTERLESS)
            .ok()
            .as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}

pub fn spec_route_eligible(
    kind: WgpuModelKind,
    needs_logits: bool,
    knobs: spec::SpecKnobs,
) -> bool {
    knobs.enabled
        && !needs_logits
        && (knobs.kinds.admits(kind.spec_chain_slug())
            || (mtp_opted_in() && kind == WgpuModelKind::Qwen3_5Dense))
}

pub fn wgpu_spec_decode_status(
    kind: WgpuModelKind,
    knobs: spec::SpecKnobs,
) -> Option<&'static str> {
    if knobs.enabled && knobs.kinds.admits(kind.spec_chain_slug()) {
        Some("on")
    } else {
        None
    }
}

pub const CHAIN_VERIFY_SEAM: &str =
    "the spec chain route needs a decoder with a multi-row verify forward whose commit is exact: \
     verify_chain(batch) must argmax one row per batch token and advance(n) must leave the decoder \
     in the state n successive decode_step calls would have left it in";

struct DecoderChainTarget<'a> {
    decoder: &'a mut Decoder,
    max_seq: usize,
    stepped: bool,

    want_hidden: bool,

    want_hidden_host: bool,
    hidden_row: Option<Vec<f32>>,
    hidden_loc: Option<learned::HiddenLoc>,
}

impl spec::ChainVerifyTarget for DecoderChainTarget<'_> {
    fn verify_chain(&mut self, batch: &[u32]) -> anyhow::Result<Vec<u32>> {
        if batch.len() == 1 {
            let t = self.decoder.step(batch[0])?;
            self.stepped = true;
            return Ok(vec![t]);
        }
        match &mut *self.decoder {
            Decoder::Gemma4E4b(m) => m.verify_chain(batch),
            Decoder::Qwen3_5Dense(m) => m.verify_chain(batch),
            Decoder::GptOss(m) => m.verify_chain(batch),
            other => anyhow::bail!("{CHAIN_VERIFY_SEAM}; {} has none", other.kind_label()),
        }
    }

    fn advance(&mut self, n: usize) -> anyhow::Result<()> {
        if self.stepped {
            anyhow::ensure!(n == 1, "plain spec round tried to commit {n} rows");
            self.stepped = false;
            if self.want_hidden {
                self.hidden_loc = Some(learned::HiddenLoc::Decode);
                if self.want_hidden_host {
                    let Decoder::Gemma4E4b(m) = &mut *self.decoder else {
                        anyhow::bail!("{ASSISTANT_DRAFTER_HIDDEN_ROWS_ARE_GEMMA4_E4B_ONLY}");
                    };
                    self.hidden_row = Some(m.decode_hidden_row()?);
                }
            }
            return Ok(());
        }
        match &mut *self.decoder {
            Decoder::Gemma4E4b(m) => {
                m.advance(n)?;
                if self.want_hidden {
                    self.hidden_loc = Some(learned::HiddenLoc::Verify(n - 1));
                    if self.want_hidden_host {
                        self.hidden_row = Some(m.verify_hidden_row(n - 1)?);
                    }
                }
            }
            Decoder::Qwen3_5Dense(m) => m.advance(n)?,
            Decoder::GptOss(m) => m.advance(n)?,
            other => anyhow::bail!("{CHAIN_VERIFY_SEAM}; {} has none", other.kind_label()),
        }
        Ok(())
    }

    fn capacity(&self) -> usize {
        let (rows, span) = self.decoder.chain_verify_geometry();
        spec::chain_capacity(rows, self.decoder.current_pos(), span, self.max_seq)
    }
}

pub const ASSISTANT_DRAFTER_HIDDEN_ROWS_ARE_GEMMA4_E4B_ONLY: &str =
    "the learned assistant drafter reads the verifier's hidden row, which only gemma4_e4b_wgpu \
     exposes (decode_hidden_row / verify_hidden_row); every other chain-route kind drafts from the \
     model-free suffix automaton, so want_hidden must stay false for it";

struct Shared {
    model_id: String,
    kind: WgpuModelKind,
    tokenizer: Arc<tokenizers::Tokenizer>,
    template: Arc<ChatTemplate>,
    eos_ids: Vec<u32>,
    bos_token_id: Option<u32>,
    strip_specials: Vec<String>,
    max_seq: usize,
    vocab: usize,
    default_max_new: usize,
    mm_input: bool,
    mm_caps: std::sync::OnceLock<Option<mm::MmCaps>>,

    budget: Option<batch::StepBudget>,

    batch_capacity: std::sync::atomic::AtomicUsize,
    batches_run: std::sync::atomic::AtomicUsize,
    prefix_tokens_reused: std::sync::atomic::AtomicUsize,
    passes: std::sync::atomic::AtomicUsize,
    spec_rounds: std::sync::atomic::AtomicUsize,
    spec_rounds_with_draft: std::sync::atomic::AtomicUsize,
    spec_drafted: std::sync::atomic::AtomicUsize,
    spec_accepted: std::sync::atomic::AtomicUsize,
    spec_emitted: std::sync::atomic::AtomicUsize,

    spec_gate: spec::AssistantGate,
}

enum Job {
    Generate {
        req: ChatGenerateRequest,
        tx: mpsc::Sender<ChatEvent>,
    },
}

pub struct WgpuChatEngine {
    inner: Arc<Shared>,
    jobs: std::sync::Mutex<std::sync::mpsc::Sender<Job>>,
    worker: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Drop for WgpuChatEngine {
    fn drop(&mut self) {
        let (dead_tx, _) = std::sync::mpsc::channel();
        *self
            .jobs
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = dead_tx;
        if let Some(handle) = self
            .worker
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            let _ = handle.join();
        }
    }
}

impl WgpuChatEngine {
    pub fn max_seq_env_override() -> Option<usize> {
        std::env::var(MAX_SEQ_ENV)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
    }

    pub fn default_max_seq_for(model_dir: &Path) -> usize {
        if let Some(v) = Self::max_seq_env_override() {
            return v;
        }
        let model_max = std::fs::read_to_string(model_dir.join("config.json"))
            .ok()
            .and_then(|raw| config_max_position(&raw));
        match model_max {
            Some(mp) => mp.min(DEFAULT_MAX_SEQ),
            None => DEFAULT_MAX_SEQ,
        }
    }

    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        Self::load_with(model_dir, Self::default_max_seq_for(model_dir), None)
    }

    pub fn load_with(
        model_dir: &Path,
        max_seq: usize,
        model_id: Option<String>,
    ) -> anyhow::Result<Self> {
        Self::load_with_lora(model_dir, max_seq, model_id, None)
    }

    pub fn load_with_lora(
        model_dir: &Path,
        max_seq: usize,
        model_id: Option<String>,
        adapter_dir: Option<&Path>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(max_seq > 0, "max_seq must be positive");
        nv_kernels::wgpu_backend::WgpuContext::shared()
            .map_err(|e| anyhow::anyhow!("no wgpu adapter: {e}"))?;
        ensure_serving_sidecars(model_dir)?;

        let cfg_path = model_dir.join("config.json");
        let raw_cfg = match std::fs::read_to_string(&cfg_path) {
            Ok(raw) => raw,
            Err(read_err) => match nv_weights::gguf::lone_gguf_file(model_dir) {
                Some(gguf) => gguf_config_json(&gguf)?,
                None => {
                    return Err(anyhow::Error::new(read_err)
                        .context(format!("read {}", cfg_path.display())));
                }
            },
        };
        let kind = classify_wgpu_model(&raw_cfg)?;

        let template = ChatTemplate::load_reason(model_dir).map_err(|reason| {
            anyhow::anyhow!(
                "no official chat template in {}: {reason}. The wgpu chat engine refuses to \
                 serve a hand-rolled or ChatML-fallback prompt.",
                model_dir.display()
            )
        })?;

        let mut tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

        nv_tokenizer::sanitize_for_serving(&mut tokenizer);
        let tokenizer = tokenizer;
        let eos_ids = eos_ids_for_serving(model_dir)?;
        let model_id = model_id.unwrap_or_else(|| model_id_for_dir(model_dir));

        let bos_declaration = std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
            .map(|raw| bos_declaration_from_json(&raw))
            .unwrap_or(BosDeclaration::Absent);
        let bos_token_id = prompt_bos_id(
            &bos_declaration,
            &|t: &str| tokenizer.token_to_id(t),
            &eos_ids,
            &model_id,
        );

        let vocab = match kind {
            WgpuModelKind::Gemma4Dense | WgpuModelKind::Gemma4E4b => {
                Gemma4Config::from_hf_json_str(&raw_cfg)?.vocab_size
            }
            WgpuModelKind::Gemma4Moe => {
                Gemma4MoeConfig::from_hf_json_str(&raw_cfg)?.base.vocab_size
            }
            WgpuModelKind::Qwen3_5Moe => Qwen3MoeConfig::from_hf_json_str(&raw_cfg)?.vocab_size,
            WgpuModelKind::Qwen3_5Dense => {
                Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg)?.vocab_size
            }
            WgpuModelKind::GptOss => GptOssConfig::from_hf_json_str(&raw_cfg)?.vocab_size,
            WgpuModelKind::Laguna => LagunaConfig::from_hf_json_str(&raw_cfg)?.vocab_size,
        };

        let strip_specials = special_strip_list(&tokenizer);
        let mm_spec = mm::detect(kind, model_dir, &raw_cfg, &tokenizer)?;
        let budget =
            batch::kv_geometry_from_config(&raw_cfg, batch::KV_ELEM_BYTES).and_then(|kv| {
                mem_fit::estimate_model_with_max_seq(model_dir, max_seq).map(|e| {
                    batch::StepBudget {
                        weight_bytes: (e.weight_gib * (1u64 << 30) as f64) as u64,
                        kv,
                    }
                })
            });
        let inner = Arc::new(Shared {
            model_id,
            kind,
            tokenizer: Arc::new(tokenizer),
            template,
            eos_ids,
            bos_token_id,
            strip_specials,
            max_seq,
            vocab,
            default_max_new: DEFAULT_MAX_NEW_TOKENS,
            mm_input: mm_spec.is_some(),
            mm_caps: std::sync::OnceLock::new(),
            budget,
            batch_capacity: std::sync::atomic::AtomicUsize::new(0),
            batches_run: std::sync::atomic::AtomicUsize::new(0),
            prefix_tokens_reused: std::sync::atomic::AtomicUsize::new(0),
            passes: std::sync::atomic::AtomicUsize::new(0),
            spec_rounds: std::sync::atomic::AtomicUsize::new(0),
            spec_rounds_with_draft: std::sync::atomic::AtomicUsize::new(0),
            spec_drafted: std::sync::atomic::AtomicUsize::new(0),
            spec_accepted: std::sync::atomic::AtomicUsize::new(0),
            spec_emitted: std::sync::atomic::AtomicUsize::new(0),
            spec_gate: spec::AssistantGate::new(),
        });

        let (jobs_tx, jobs_rx) = std::sync::mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<usize, String>>();
        let worker_meta = inner.clone();
        let dir = model_dir.to_path_buf();
        let adapter = adapter_dir.map(Path::to_path_buf);
        let worker = std::thread::Builder::new()
            .name("wgpu-chat".into())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                let built =
                    build_decoder_with_lora(&dir, &raw_cfg, kind, max_seq, adapter.as_deref());
                let (mut decoder, mm_rt) = match built {
                    Ok(mut d) => {
                        if warmup_enabled() {
                            warmup_decoder(&mut d, worker_meta.bos_token_id);
                        }
                        let mm_rt = match mm_spec {
                            None => None,
                            Some(spec) => {
                                let fatal = spec.tower_load_failure_is_fatal();
                                match mm::MmRuntime::load(spec, &dir) {
                                    Ok(rt) => Some(rt),
                                    Err(e) if fatal => {
                                        let _ = ready_tx.send(Err(format!(
                                            "multimodal tower load failed: {e:#}"
                                        )));
                                        return;
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            kind = kind.label(),
                                            error = format!("{e:#}"),
                                            "multimodal tower load failed; text serving continues \
                                             and media requests will be refused by name"
                                        );
                                        None
                                    }
                                }
                            }
                        };
                        let _ = worker_meta
                            .mm_caps
                            .set(mm_rt.as_ref().map(mm::MmRuntime::caps));
                        worker_meta.batch_capacity.store(
                            batch::BatchStepper::batch_capacity(&d),
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        let _ = ready_tx.send(Ok(d.pass_count()));
                        (d, mm_rt)
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("{e:#}")));
                        return;
                    }
                };
                let mut assistant = match &decoder {
                    Decoder::Gemma4E4b(m) => learned::AssistantSpecDrafter::from_env(&dir, m),
                    _ => None,
                };
                let mut prefix = PrefixCache::default();
                let persist_meta = persist::Meta {
                    model_id: worker_meta.model_id.clone(),
                    kind: kind.label().to_string(),
                    adapter: adapter.as_deref().map(|p| p.display().to_string()),
                    max_seq,
                };
                let mut persist = persist::Session::from_env(persist_meta.clone());
                if let Some(p) = persist.as_ref() {
                    if let Some((tokens, frontier)) = p.restore(kv_disk_mut(&mut decoder)) {
                        prefix = PrefixCache { tokens, frontier };
                    }
                }
                while let Ok(job) = jobs_rx.recv() {
                    let Job::Generate { req, tx } = job;
                    let (batched, mut queue) =
                        collect_batch(&worker_meta, &decoder, (req, tx), &jobs_rx);
                    if batch::batch_pays(batched.len()) {
                        prefix = PrefixCache::default();
                        let back = run_batch_blocking(&worker_meta, &mut decoder, batched);
                        queue.splice(0..0, back);
                    } else {
                        queue.splice(0..0, batched);
                    }
                    for (req, tx) in queue {
                        if let Some(id) = req.kv_resume.as_deref() {
                            resume_response_snapshot(&mut decoder, &mut prefix, &persist_meta, id);
                        }
                        let store_id = req.kv_store.clone();
                        if let Err(err) = run_blocking(
                            &worker_meta,
                            &mut decoder,
                            &mut prefix,
                            assistant.as_mut(),
                            mm_rt.as_ref(),
                            req,
                            &tx,
                        ) {
                            let _ = push_blocking(
                                &tx,
                                ChatEvent::Error(format!("{err:#}")),
                                send_timeout(),
                            );
                        }
                        if let Some(id) = store_id.as_deref() {
                            store_response_snapshot(&decoder, &prefix, &persist_meta, id);
                        }
                    }
                    if let Some(p) = persist.as_mut() {
                        p.maybe_save(kv_disk(&decoder), &prefix.tokens, prefix.frontier);
                    }
                }
                if let Some(p) = persist.as_mut() {
                    p.save_now(kv_disk(&decoder), &prefix.tokens, prefix.frontier);
                }
            })
            .context("spawn wgpu chat worker thread")?;

        let passes = ready_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("wgpu chat worker died during model load"))?
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        inner
            .passes
            .store(passes, std::sync::atomic::Ordering::Relaxed);

        tracing::info!(
            model = %inner.model_id,
            kind = kind.label(),
            passes_per_token = passes,
            max_seq,
            vocab,
            eos = ?inner.eos_ids,
            lora = %adapter_dir.map(|p| p.display().to_string()).unwrap_or_else(|| "none".into()),
            "wgpu chat engine ready"
        );

        Ok(Self {
            inner,
            jobs: std::sync::Mutex::new(jobs_tx),
            worker: std::sync::Mutex::new(Some(worker)),
        })
    }

    pub fn kind(&self) -> WgpuModelKind {
        self.inner.kind
    }

    pub fn eos_ids(&self) -> &[u32] {
        &self.inner.eos_ids
    }

    pub fn max_seq(&self) -> usize {
        self.inner.max_seq
    }

    pub fn passes_per_token(&self) -> usize {
        self.inner.passes.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn batch_capacity(&self) -> usize {
        self.inner
            .batch_capacity
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn batches_run(&self) -> usize {
        self.inner
            .batches_run
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn prefix_tokens_reused(&self) -> usize {
        self.inner
            .prefix_tokens_reused
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn last_spec_stats(&self) -> spec::SpecStats {
        use std::sync::atomic::Ordering::Relaxed;
        spec::SpecStats {
            rounds: self.inner.spec_rounds.load(Relaxed),
            rounds_with_draft: self.inner.spec_rounds_with_draft.load(Relaxed),
            drafted: self.inner.spec_drafted.load(Relaxed),
            accepted: self.inner.spec_accepted.load(Relaxed),
            emitted: self.inner.spec_emitted.load(Relaxed),
        }
    }

    pub fn render_with_tools(&self, messages: &[ChatMessageIn], tools: &[Tool]) -> String {
        self.render_with_tools_kwargs(messages, tools, &TemplateKwargs::new())
    }

    pub fn render_with_tools_kwargs(
        &self,
        messages: &[ChatMessageIn],
        tools: &[Tool],
        extra: &TemplateKwargs,
    ) -> String {
        render_official_with_tools_kwargs(&self.inner.template, messages, tools, extra)
            .unwrap_or_else(|err| {
                tracing::error!(error = %err, "official chat template render failed");
                String::new()
            })
    }
}

pub fn render_official_with_tools(
    template: &ChatTemplate,
    messages: &[ChatMessageIn],
    tools: &[Tool],
) -> anyhow::Result<String> {
    render_official_with_tools_kwargs(template, messages, tools, &TemplateKwargs::new())
}

pub fn render_official_with_tools_kwargs(
    template: &ChatTemplate,
    messages: &[ChatMessageIn],
    tools: &[Tool],
    extra: &TemplateKwargs,
) -> anyhow::Result<String> {
    let msgs = serde_json::to_value(messages)?;
    let kw = crate::oapi::chat::merged_template_kwargs(template, extra);
    if tools.is_empty() {
        return template.render_with_kwargs(&msgs, None, true, &kw);
    }
    let tools_json = serde_json::to_value(tools)?;
    template.render_with_kwargs(&msgs, Some(&tools_json), true, &kw)
}

pub fn template_supports_tools(template: &ChatTemplate, probe: &[ChatMessageIn]) -> bool {
    let Ok(msgs) = serde_json::to_value(probe) else {
        return false;
    };
    let tool = serde_json::json!([{
        "type": "function",
        "function": {"name": "nv_probe_tool_support", "description": "probe", "parameters": {}}
    }]);
    let Ok(with) = template.render(&msgs, Some(&tool), true) else {
        return false;
    };
    let Ok(without) = template.render(&msgs, None, true) else {
        return false;
    };
    with != without
}

pub struct StopScanner {
    sent: String,
    stops: Vec<String>,
    max_stop: usize,
    pub stopped: bool,
    pub matched: Option<String>,
}

impl StopScanner {
    pub fn new(stops: &[String]) -> Self {
        let stops: Vec<String> = stops.iter().filter(|s| !s.is_empty()).cloned().collect();
        let max_stop = stops.iter().map(|s| s.len()).max().unwrap_or(0);
        Self {
            sent: String::new(),
            stops,
            max_stop,
            stopped: false,
            matched: None,
        }
    }

    pub fn step(&mut self, full: &str) -> (String, bool) {
        if self.stopped {
            return (String::new(), true);
        }
        let mut end = full.len();
        while let Some(c) = full[..end].chars().next_back() {
            if c == '\u{FFFD}' {
                end -= c.len_utf8();
            } else {
                break;
            }
        }
        let visible = &full[..end];
        if !visible.starts_with(self.sent.as_str()) {
            let piece = common_suffix_delta(&self.sent, visible).to_string();
            self.sent = visible.to_string();
            return (piece, false);
        }
        if self.max_stop > 0 {
            let mut hit: Option<usize> = None;
            for s in &self.stops {
                if let Some(at) = visible.find(s.as_str()) {
                    hit = Some(hit.map_or(at, |h: usize| h.min(at)));
                }
            }
            if let Some(at) = hit {
                self.matched = self
                    .stops
                    .iter()
                    .find(|s| visible[at..].starts_with(s.as_str()))
                    .cloned();
                let cut = at.max(self.sent.len());
                let piece = visible[self.sent.len()..cut].to_string();
                self.sent.push_str(&piece);
                self.stopped = true;
                return (piece, true);
            }
        }
        let mut emit_to = visible.len();
        if self.max_stop > 0 {
            let mut idx = visible
                .len()
                .saturating_sub(self.max_stop.saturating_sub(1));
            while idx < visible.len() && !visible.is_char_boundary(idx) {
                idx += 1;
            }
            while idx < visible.len() {
                let suffix = &visible[idx..];
                if self.stops.iter().any(|s| s.starts_with(suffix)) {
                    emit_to = idx;
                    break;
                }
                idx += suffix.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
            }
        }
        let emit_to = emit_to.max(self.sent.len());
        let piece = visible[self.sent.len()..emit_to].to_string();
        self.sent.push_str(&piece);
        (piece, false)
    }

    pub fn finish(&mut self, full: &str) -> String {
        if self.stopped {
            return String::new();
        }
        let (mut piece, hit) = self.step(full);
        if !hit && full.len() > self.sent.len() && full.starts_with(self.sent.as_str()) {
            piece.push_str(&full[self.sent.len()..]);
            self.sent = full.to_string();
        }
        piece
    }
}

fn common_suffix_delta<'a>(emitted: &str, fresh: &'a str) -> &'a str {
    let mut split = 0usize;
    for (ec, (ni, nc)) in emitted.chars().zip(fresh.char_indices()) {
        if ec != nc {
            break;
        }
        split = ni + nc.len_utf8();
    }
    &fresh[split..]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Push {
    Sent,
    Closed,
    TimedOut,
}

fn send_timeout() -> std::time::Duration {
    let ms = std::env::var("NV_SSE_SEND_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(10_000)
        .max(1);
    std::time::Duration::from_millis(ms)
}

fn push_blocking(
    tx: &mpsc::Sender<ChatEvent>,
    ev: ChatEvent,
    timeout: std::time::Duration,
) -> Push {
    use tokio::sync::mpsc::error::TrySendError;
    let deadline = std::time::Instant::now() + timeout;
    let mut ev = ev;
    loop {
        match tx.try_send(ev) {
            Ok(()) => return Push::Sent,
            Err(TrySendError::Closed(_)) => return Push::Closed,
            Err(TrySendError::Full(back)) => {
                if std::time::Instant::now() >= deadline {
                    return Push::TimedOut;
                }
                ev = back;
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
    }
}

pub const FAST_SAMPLE_CAP: usize = 4096;

pub const FAST_TOP_P_CAP: usize = 2048;

#[derive(Clone, Copy, Debug)]
struct Cand {
    val: f32,
    idx: u32,
}

fn cand_better(a: &Cand, b: &Cand) -> bool {
    a.val > b.val || (a.val == b.val && a.idx < b.idx)
}

struct Worst(Cand);

impl PartialEq for Worst {
    fn eq(&self, other: &Self) -> bool {
        self.0.val == other.0.val && self.0.idx == other.0.idx
    }
}

impl Eq for Worst {}

impl PartialOrd for Worst {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Worst {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if cand_better(&self.0, &other.0) {
            std::cmp::Ordering::Less
        } else if cand_better(&other.0, &self.0) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

pub const DECODE_PIPE_ENV_STAYS_OPT_IN_UNTIL_A_SERVING_AB_LANDS_A_WIN: &str =
    "NV_WGPU_DECODE_PIPE=1 routes the plain greedy E4B loop through decode_step_pipelined \
     (device-resident next token, one step in flight); the stepped path stays the default \
     because the pipe's serving-level win is unmeasured, and the pipe is bit-identical to \
     stepped decode per e4b_host_sync::e4b_pipe_matches_stepped_bitwise";

fn decode_pipe_opted_in() -> bool {
    std::env::var("NV_WGPU_DECODE_PIPE").ok().as_deref() == Some("1")
}

fn fast_sampler_enabled() -> bool {
    std::env::var("NV_WGPU_FAST_SAMPLER").ok().as_deref() != Some("0")
}

fn top_m_candidates(logits: &[f32], inv_t: f32, m: usize) -> Vec<Cand> {
    let mut heap: std::collections::BinaryHeap<Worst> =
        std::collections::BinaryHeap::with_capacity(m + 1);
    for (i, &v) in logits.iter().enumerate() {
        let c = Cand {
            val: v * inv_t,
            idx: i as u32,
        };
        if heap.len() < m {
            heap.push(Worst(c));
            continue;
        }
        if cand_better(&c, &heap.peek().map(|w| w.0).unwrap()) {
            heap.pop();
            heap.push(Worst(c));
        }
    }
    let mut out: Vec<Cand> = heap.into_iter().map(|w| w.0).collect();
    out.sort_unstable_by(|a, b| {
        b.val
            .partial_cmp(&a.val)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.idx.cmp(&b.idx))
    });
    out
}

pub fn fast_sample_checked(
    logits: &[f32],
    p: &nv_layers::sampler::SamplingParams,
    u01: f32,
) -> Option<Option<u32>> {
    let n = logits.len();
    if n == 0 || p.is_greedy() {
        return None;
    }
    let k_req = p.top_k;
    if let Some(k) = k_req {
        if k == 0 || (k < n && k > FAST_SAMPLE_CAP) {
            return None;
        }
    }
    let k_eff = k_req.filter(|&k| k < n);
    let tp = p.top_p.filter(|&t| t > 0.0 && t < 1.0);
    if k_eff.is_none() && tp.is_none() {
        return None;
    }

    let mut amax = 0usize;
    let mut amax_v = f32::NEG_INFINITY;
    let mut seen = false;
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            return None;
        }
        if !seen || v > amax_v {
            amax_v = v;
            amax = i;
            seen = true;
        }
    }
    let point_mass = Some(Some(amax as u32));

    let inv_t = 1.0f32 / p.temperature.max(1e-6);

    let (surv, mut probs) = match k_eff {
        Some(k) => {
            let mut surv = top_m_candidates(logits, inv_t, k);
            surv.sort_unstable_by_key(|c| c.idx);
            let max = surv.iter().map(|c| c.val).fold(f32::NEG_INFINITY, f32::max);
            if !max.is_finite() {
                return point_mass;
            }
            let probs: Vec<f32> = surv.iter().map(|c| (c.val - max).exp()).collect();
            (surv, probs)
        }
        None => {
            let max = logits
                .iter()
                .map(|&v| v * inv_t)
                .fold(f32::NEG_INFINITY, f32::max);
            if !max.is_finite() {
                return point_mass;
            }
            let mut sum = 0.0f32;
            for &v in logits {
                sum += (v * inv_t - max).exp();
            }
            if sum <= 0.0 || !sum.is_finite() {
                return point_mass;
            }
            let want = tp?;
            let floor = (want * sum).ceil();
            if !floor.is_finite() || floor > FAST_TOP_P_CAP as f32 {
                return None;
            }
            let mut m = (floor as usize)
                .max(64)
                .next_power_of_two()
                .min(n)
                .min(FAST_TOP_P_CAP);
            loop {
                let ranked = top_m_candidates(logits, inv_t, m);
                let mut cum = 0.0f32;
                let mut keep = ranked.len();
                for (j, c) in ranked.iter().enumerate() {
                    cum += (c.val - max).exp() / sum;
                    if cum >= want {
                        keep = j + 1;
                        break;
                    }
                }
                if keep < ranked.len() || ranked.len() == n {
                    let mut surv: Vec<Cand> = ranked[..keep].to_vec();
                    surv.sort_unstable_by_key(|c| c.idx);
                    let probs: Vec<f32> = surv.iter().map(|c| (c.val - max).exp() / sum).collect();
                    return finish_from_survivors(&surv, probs, p, u01, amax);
                }
                if m >= FAST_TOP_P_CAP {
                    return None;
                }
                m = (m * 8).min(n).min(FAST_TOP_P_CAP);
            }
        }
    };

    let mut sum = 0.0f32;
    for &v in &probs {
        sum += v;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return point_mass;
    }
    for v in probs.iter_mut() {
        *v /= sum;
    }

    if let Some(want) = tp {
        let mut order: Vec<usize> = (0..surv.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(surv[a].idx.cmp(&surv[b].idx))
        });
        let mut cum = 0.0f32;
        let mut keep = vec![false; surv.len()];
        for &j in &order {
            keep[j] = true;
            cum += probs[j];
            if cum >= want {
                break;
            }
        }
        for (j, v) in probs.iter_mut().enumerate() {
            if !keep[j] {
                *v = 0.0;
            }
        }
    }

    finish_from_survivors(&surv, probs, p, u01, amax)
}

fn finish_from_survivors(
    surv: &[Cand],
    mut probs: Vec<f32>,
    p: &nv_layers::sampler::SamplingParams,
    u01: f32,
    amax: usize,
) -> Option<Option<u32>> {
    if let Some(mp) = p.min_p {
        if mp > 0.0 {
            let pmax = probs.iter().cloned().fold(0.0f32, f32::max);
            let thresh = mp * pmax;
            for v in probs.iter_mut() {
                if *v < thresh {
                    *v = 0.0;
                }
            }
        }
    }
    let mut renorm = 0.0f32;
    for &v in &probs {
        renorm += v;
    }
    if renorm <= 0.0 || !renorm.is_finite() {
        return Some(Some(amax as u32));
    }
    for v in probs.iter_mut() {
        *v /= renorm;
    }
    let u = u01.clamp(0.0, 1.0 - f32::EPSILON);
    let mut acc = 0.0f32;
    for (j, c) in surv.iter().enumerate() {
        acc += probs[j];
        if u < acc {
            return Some(Some(c.idx));
        }
    }
    for j in (0..surv.len()).rev() {
        if probs[j] > 0.0 {
            return Some(Some(surv[j].idx));
        }
    }
    Some(None)
}

pub const WGPU_SAMPLER_SEAM_TODO: &str = "in-shader sampling is implemented and gated \
    (nv_kernels::wgpu_backend::kernels::sampler::sampler_exact_token_buffers, 16416/16416 \
    token-for-token agreement with nv_layers::sampler) but it is NOT reachable from this engine: \
    every nv-models wgpu decoder keeps its logits GpuTensor private and only exposes \
    decode_step_logits -> Vec<f32>, so the vocab-sized download happens before the sampler could \
    run. Wiring it needs one accessor per decoder module (fn logits_buffer(&self) -> &wgpu::Buffer \
    plus fn token_buffer(&self) -> &wgpu::Buffer) and a Decoder::step_sampled arm here; those \
    files are owned by other lanes. Until then the non-greedy path pays the download and uses the \
    exact host fast path in fast_sample_checked instead.";

pub fn in_shader_sampling_plan(
    req: &ChatGenerateRequest,
    vocab: usize,
) -> Option<nv_kernels::wgpu_backend::kernels::sampler::ExactSampling> {
    use nv_kernels::wgpu_backend::kernels::sampler::ExactSampling;
    if req.logprobs
        || req.guided.is_some()
        || !req.logit_bias.is_empty()
        || req.presence_penalty.unwrap_or(0.0) != 0.0
        || req.frequency_penalty.unwrap_or(0.0) != 0.0
        || (req.repetition_penalty.unwrap_or(1.0) - 1.0).abs() > f32::EPSILON
    {
        return None;
    }
    let plan = ExactSampling {
        temperature: req.temperature.unwrap_or(0.0).max(0.0),
        top_k: req.top_k.unwrap_or(0),
        top_p: req.top_p.unwrap_or(1.0),
        min_p: req.min_p.unwrap_or(0.0),
        u01: None,
        seed: req.seed.unwrap_or(0),
    };
    plan.supported(vocab).then_some(plan)
}

pub fn sample_token_exact(
    logits: &[f32],
    p: &nv_layers::sampler::SamplingParams,
    u01: f32,
) -> Option<u32> {
    if fast_sampler_enabled() {
        if let Some(hit) = fast_sample_checked(logits, p, u01) {
            return hit;
        }
    }
    nv_layers::sampler::sample_token_checked(logits, p, u01)
}

pub struct HostSampler {
    params: nv_layers::sampler::SamplingParams,
    rng: Pcg64,
    counts: std::collections::HashMap<u32, u32>,
    prompt_tokens: Vec<u32>,
    logit_bias: Vec<(u32, f32)>,
    guided: Option<crate::oapi::chat_engine::GuidedRun>,
    grammar_requested: bool,
    logprobs: bool,
    top_logprobs: usize,
}

pub struct Picked {
    pub token: u32,
    pub logprob: Option<f32>,
    pub top: Vec<(u32, f32)>,
}

impl HostSampler {
    pub fn new(req: &ChatGenerateRequest, seed: u64) -> Self {
        Self {
            params: nv_layers::sampler::SamplingParams {
                temperature: req.temperature.unwrap_or(0.0).max(0.0),
                top_k: req.top_k.map(|k| k as usize),
                top_p: req.top_p,
                min_p: req.min_p,
                presence_penalty: req.presence_penalty.unwrap_or(0.0),
                frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
                repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
            },
            rng: Pcg64::seed_from_u64(seed),
            counts: std::collections::HashMap::new(),
            prompt_tokens: Vec::new(),
            logit_bias: req.logit_bias.clone(),
            guided: None,
            grammar_requested: req.guided.is_some(),
            logprobs: req.logprobs,
            top_logprobs: req.top_logprobs,
        }
    }

    pub fn set_guided(&mut self, guided: nv_grammar::GuidedDecoder, max_new: usize) {
        self.guided = Some(crate::oapi::chat_engine::GuidedRun::new(guided, max_new));
    }

    pub fn guided(&self) -> Option<&nv_grammar::GuidedDecoder> {
        self.guided.as_ref().map(|g| g.decoder())
    }

    pub fn guided_mut(&mut self) -> Option<&mut nv_grammar::GuidedDecoder> {
        self.guided.as_mut().map(|g| g.decoder_mut())
    }

    pub fn needs_logits(&self) -> bool {
        !self.params.is_greedy()
            || self.params.has_penalties()
            || !self.logit_bias.is_empty()
            || self.guided.is_some()
            || self.grammar_requested
            || self.logprobs
    }

    pub fn seed_prompt(&mut self, ids: &[u32]) {
        if (self.params.repetition_penalty - 1.0).abs() <= f32::EPSILON {
            return;
        }
        let mut v = ids.to_vec();
        v.sort_unstable();
        v.dedup();
        self.prompt_tokens = v;
    }

    fn uniform(&mut self) -> f32 {
        let raw = self.rng.next_u64() >> 11;
        ((raw as f64) / ((1u64 << 53) as f64)) as f32
    }

    fn record(&mut self, tok: u32) {
        *self.counts.entry(tok).or_insert(0) += 1;
    }

    pub fn pick(&mut self, logits: &[f32]) -> anyhow::Result<Picked> {
        use nv_layers::sampler;
        let need_copy =
            self.params.has_penalties() || !self.logit_bias.is_empty() || self.guided.is_some();
        let chosen = if need_copy {
            let mut lg = logits.to_vec();
            for &(id, bias) in &self.logit_bias {
                if let Some(v) = lg.get_mut(id as usize) {
                    *v += bias;
                }
            }
            if self.params.has_penalties() {
                let seen: Vec<(u32, u32)> = self.counts.iter().map(|(&t, &c)| (t, c)).collect();
                sampler::apply_penalties_with_prompt(
                    &mut lg,
                    &seen,
                    &self.prompt_tokens,
                    &self.params,
                );
            }
            if let Some(g) = self.guided.as_mut() {
                g.apply_mask(&mut lg);
            }
            let u = self.uniform();
            sample_token_exact(&lg, &self.params, u)
        } else if self.params.is_greedy() {
            sampler::argmax_checked(logits)
        } else {
            let u = self.uniform();
            sample_token_exact(logits, &self.params, u)
        };
        let token = chosen.ok_or_else(|| {
            anyhow::anyhow!("no legal token: the sampling mask left every candidate at -inf")
        })?;
        if let Some(g) = self.guided.as_mut() {
            anyhow::ensure!(
                g.advance(token),
                "guided decoding died on token {token}: the grammar can accept no continuation, \
                 so the rest of this completion would be unconstrained"
            );
        }
        let mut top = Vec::new();
        let mut logprob = None;
        if self.logprobs {
            let lps = sampler::logprobs_full(logits, 1.0);
            logprob = Some(
                lps.get(token as usize)
                    .copied()
                    .unwrap_or(f32::NEG_INFINITY),
            );
            if self.top_logprobs > 0 {
                for i in sampler::top_n_indices(&lps, self.top_logprobs) {
                    top.push((i as u32, lps[i]));
                }
            }
        }
        self.record(token);
        Ok(Picked {
            token,
            logprob,
            top,
        })
    }
}

fn logprob_entry(
    tokenizer: &tokenizers::Tokenizer,
    token: u32,
    logprob: Option<f32>,
    top: &[(u32, f32)],
) -> LogprobEntry {
    let piece = |t: u32| tokenizer.decode(&[t], false).unwrap_or_default();
    let text = piece(token);
    LogprobEntry {
        bytes: text.as_bytes().to_vec(),
        token: text,
        logprob: logprob.unwrap_or(0.0),
        top_logprobs: top
            .iter()
            .map(|&(t, lp)| {
                let s = piece(t);
                TopLogprob {
                    bytes: s.as_bytes().to_vec(),
                    token: s,
                    logprob: lp,
                }
            })
            .collect(),
    }
}

fn os_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    n ^ (std::process::id() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[async_trait::async_trait]
impl ChatEngine for WgpuChatEngine {
    fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    fn spec_decode_status(&self) -> Option<&'static str> {
        wgpu_spec_decode_status(self.inner.kind, spec::SpecKnobs::from_env())
    }

    fn supports_mm_input(&self) -> bool {
        self.inner.mm_input
    }

    fn official_template(&self) -> Option<&ChatTemplate> {
        Some(self.inner.template.as_ref())
    }

    fn render_prompt(&self, messages: &[ChatMessageIn]) -> String {
        self.render_with_tools(messages, &[])
    }

    fn render_chat(
        &self,
        messages: &[ChatMessageIn],
        tools: &[Tool],
        choice: &ToolChoice,
    ) -> String {
        self.render_chat_kwargs(messages, tools, choice, &TemplateKwargs::new())
    }

    fn render_chat_kwargs(
        &self,
        messages: &[ChatMessageIn],
        tools: &[Tool],
        choice: &ToolChoice,
        extra: &TemplateKwargs,
    ) -> String {
        if !tools.is_empty() {
            if !template_supports_tools(&self.inner.template, messages) {
                tracing::error!(
                    model = %self.inner.model_id,
                    "tools were requested but this model's official chat template ignores the \
                     `tools` variable: the wgpu chat engine does NOT synthesise a fake tool \
                     protocol in a system message; the request is rendered without tools"
                );
                return self.render_with_tools_kwargs(messages, &[], extra);
            }
            if !matches!(choice, ToolChoice::Auto | ToolChoice::None) {
                tracing::warn!(
                    model = %self.inner.model_id,
                    "tool_choice other than auto/none is advisory only on the wgpu engine: the \
                     official template has no forced-call encoding"
                );
            }
        }
        self.render_with_tools_kwargs(messages, tools, extra)
    }

    async fn generate(
        &self,
        req: ChatGenerateRequest,
        tx: mpsc::Sender<ChatEvent>,
    ) -> anyhow::Result<()> {
        if let Some(media) = req.mm.as_ref().filter(|m| mm::media_present(m)) {
            if let Some(reason) = mm::refuse_media(
                self.inner.kind,
                self.inner.mm_caps.get().copied().flatten(),
                !media.images.is_empty(),
                !media.audios.is_empty(),
            ) {
                return Err(anyhow::Error::new(crate::oapi::chat::UnsupportedMedia(
                    reason,
                )));
            }
        }
        let job = Job::Generate {
            req,
            tx: tx.clone(),
        };
        let sent = match self.jobs.lock() {
            Ok(s) => s.send(job).is_ok(),
            Err(_) => false,
        };
        if !sent {
            let _ = tx
                .send(ChatEvent::Error(
                    "wgpu chat worker thread is gone: the resident model was torn down".into(),
                ))
                .await;
        }
        Ok(())
    }
}

fn build_decoder(
    model_dir: &Path,
    raw_cfg: &str,
    kind: WgpuModelKind,
    max_seq: usize,
) -> anyhow::Result<Decoder> {
    build_decoder_with_lora(model_dir, raw_cfg, kind, max_seq, None)
}

pub fn check_lora_target_kind(kind: WgpuModelKind, adapter_dir: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        matches!(kind, WgpuModelKind::Gemma4E4b),
        "lora adapters are wired only for the gemma4-e4b wgpu graph; {} has no lora sites \
         (adapter {})",
        kind.label(),
        adapter_dir.display()
    );
    Ok(())
}

fn build_decoder_with_lora(
    model_dir: &Path,
    raw_cfg: &str,
    kind: WgpuModelKind,
    max_seq: usize,
    adapter_dir: Option<&Path>,
) -> anyhow::Result<Decoder> {
    if let Some(adir) = adapter_dir {
        check_lora_target_kind(kind, adir)?;
    }
    let t0 = std::time::Instant::now();
    if let Some(gguf) = nv_weights::gguf::lone_gguf_file(model_dir) {
        anyhow::ensure!(
            adapter_dir.is_none(),
            "lora adapters are not wired for gguf checkpoints ({})",
            gguf.display()
        );
        anyhow::ensure!(
            matches!(kind, WgpuModelKind::Gemma4Moe),
            "gguf serving is wired for the gemma4 MoE only; {} classified as {}",
            gguf.display(),
            kind.label()
        );
        let decoder = Decoder::Gemma4Moe(Box::new(
            Gemma4MoeWgpu::from_gguf(&gguf, max_seq)
                .with_context(|| format!("build Gemma4MoeWgpu from {}", gguf.display()))?,
        ));
        tracing::info!(
            kind = kind.label(),
            gguf = %gguf.display(),
            build_s = format!("{:.1}", t0.elapsed().as_secs_f64()),
            "wgpu decoder built from gguf"
        );
        return Ok(decoder);
    }
    let loader = nv_weights::WeightLoader::open_dir(model_dir, &candle_core::Device::Cpu)
        .with_context(|| format!("open weights in {}", model_dir.display()))?;
    let decoder = match kind {
        WgpuModelKind::Gemma4E4b if adapter_dir.is_some() => {
            let adir = adapter_dir.expect("checked");
            let cfg = Gemma4Config::from_hf_json_str(raw_cfg).context("parse gemma4 e4b")?;
            let lora = nv_models::gemma4_e4b_wgpu::E4bLora::from_peft_dir(adir, &cfg)
                .with_context(|| format!("build e4b lora sites from {}", adir.display()))?;
            tracing::info!(
                adapter = %adir.display(),
                rank = lora.rank(),
                matched_modules = lora.matched_modules(),
                skipped_modules = lora.skipped_modules().len(),
                lora_passes = lora.total_pass_count(),
                "lora adapter resident"
            );
            Decoder::Gemma4E4b(Box::new(
                Gemma4E4bWgpu::from_loader_with_lora(cfg, &loader, max_seq, Some(&lora))
                    .context("build Gemma4E4bWgpu with lora")?,
            ))
        }
        WgpuModelKind::Gemma4Dense => {
            let cfg = Gemma4Config::from_hf_json_str(raw_cfg).context("parse gemma4")?;
            let host =
                host_weights_from_loader(&cfg, &loader).context("gemma4 wgpu host weights")?;

            let slots = batch_graph_slots(model_dir, raw_cfg, max_seq);
            Decoder::Gemma4Dense(Box::new(if slots >= 2 {
                Gemma4Wgpu::new_batched(cfg, &host, max_seq, slots)
                    .context("build batched Gemma4Wgpu")?
            } else {
                Gemma4Wgpu::new(cfg, &host, max_seq).context("build Gemma4Wgpu")?
            }))
        }
        WgpuModelKind::Gemma4E4b => {
            let cfg = Gemma4Config::from_hf_json_str(raw_cfg).context("parse gemma4 e4b")?;
            Decoder::Gemma4E4b(Box::new(
                Gemma4E4bWgpu::from_loader(cfg, &loader, max_seq)
                    .context("build Gemma4E4bWgpu from loader")?,
            ))
        }
        WgpuModelKind::Gemma4Moe => {
            let cfg = Gemma4MoeConfig::from_hf_json_str(raw_cfg).context("parse gemma4 moe")?;
            Decoder::Gemma4Moe(Box::new(
                Gemma4MoeWgpu::from_loader(cfg, &loader, max_seq)
                    .context("build Gemma4MoeWgpu from loader")?,
            ))
        }
        WgpuModelKind::Qwen3_5Moe => {
            let cfg = Qwen3MoeConfig::from_hf_json_str(raw_cfg).context("parse qwen3.5-moe")?;
            Decoder::Qwen3_5Moe(Box::new(
                Qwen3MoeWgpu::from_loader(cfg, &loader, max_seq).context("build Qwen3MoeWgpu")?,
            ))
        }
        WgpuModelKind::Qwen3_5Dense => {
            let cfg =
                Qwen3_5DenseConfig::from_hf_json_str(raw_cfg).context("parse qwen3.5 dense")?;
            let mut m = Qwen3_5DenseWgpu::from_loader(cfg, &loader, max_seq)
                .context("build Qwen3_5DenseWgpu")?;
            if mtp_opted_in() {
                m.mtp_attach(&loader).context(
                    "attach the qwen3.8 MTP drafter head (NV_Q3D_MTP=1 on a checkpoint without \
                     mtp.* tensors cannot serve)",
                )?;
                tracing::info!("qwen3.8 mtp drafter head resident");
            }
            Decoder::Qwen3_5Dense(Box::new(m))
        }
        WgpuModelKind::GptOss => {
            let cfg = GptOssConfig::from_hf_json_str(raw_cfg).context("parse gpt_oss")?;
            Decoder::GptOss(Box::new(
                GptOssWgpu::from_loader(cfg, &loader, max_seq).context("build GptOssWgpu")?,
            ))
        }
        WgpuModelKind::Laguna => {
            let cfg = LagunaConfig::from_hf_json_str(raw_cfg).context("parse laguna")?;
            Decoder::Laguna(Box::new(
                LagunaWgpu::from_loader(cfg, &loader, max_seq).context("build LagunaWgpu")?,
            ))
        }
    };
    drop(loader);
    tracing::info!(
        kind = kind.label(),
        load_s = t0.elapsed().as_secs_f64(),
        passes_per_token = decoder.pass_count(),
        "wgpu decoder resident"
    );
    Ok(decoder)
}

fn config_max_position(raw_cfg: &str) -> Option<usize> {
    let v: serde_json::Value = serde_json::from_str(raw_cfg).ok()?;
    v.get("text_config")
        .and_then(|t| t.get("max_position_embeddings"))
        .or_else(|| v.get("max_position_embeddings"))
        .and_then(serde_json::Value::as_u64)
        .map(|m| m as usize)
}

fn chunked_prefill_enabled() -> bool {
    std::env::var("NV_WGPU_CHAT_CHUNKED_PREFILL")
        .ok()
        .as_deref()
        != Some("0")
}

fn warmup_enabled() -> bool {
    std::env::var(WARMUP_ENV).ok().as_deref() != Some("0")
}

fn warmup_passes(decoder: &mut Decoder, tok: u32) -> anyhow::Result<()> {
    let chunk = decoder.prefill_chunk_len();
    if chunked_prefill_enabled() && chunk > 0 {
        decoder.prefill_tokens(&vec![tok; chunk])?;
    }
    decoder.prefill_step(tok)?;
    decoder.step(tok)?;
    decoder.step_logits(tok)?;
    if let Decoder::Gemma4E4b(m) = decoder {
        if m.verify_max_rows() >= 2 && m.current_pos() + m.prefill_chunk_len() <= m.max_seq() {
            m.verify_chain(&[tok, tok])?;
        }
    }
    Ok(())
}

fn warmup_decoder(decoder: &mut Decoder, bos: Option<u32>) {
    let t0 = std::time::Instant::now();
    let tok = bos.unwrap_or(0);
    if let Err(e) = decoder.reset() {
        tracing::warn!("wgpu warmup pre-reset failed: {e:#}");
        return;
    }
    let warmed = warmup_passes(decoder, tok);
    if let Err(e) = decoder.reset() {
        tracing::warn!("wgpu warmup post-reset failed: {e:#}");
    }
    match warmed {
        Ok(()) => tracing::info!(
            warmup_s = t0.elapsed().as_secs_f64(),
            "wgpu pipelines warmed"
        ),
        Err(e) => tracing::warn!("wgpu warmup failed (serving continues cold): {e:#}"),
    }
}

fn clip_logits(l: &[f32], vocab: usize) -> anyhow::Result<&[f32]> {
    anyhow::ensure!(
        l.len() >= vocab,
        "wgpu logits readback returned {} values, expected at least {vocab}",
        l.len()
    );
    Ok(&l[..vocab])
}

fn strip_list_from_specials(specials: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut out: Vec<String> = specials
        .into_iter()
        .filter(|s| !s.is_empty() && !crate::oapi::chat::TOOL_WIRE_TOKENS.contains(&s.as_str()))
        .collect();
    out.sort();
    out.dedup();
    out.sort_by_key(|s| std::cmp::Reverse(s.len()));
    out
}

fn special_strip_list(tok: &tokenizers::Tokenizer) -> Vec<String> {
    strip_list_from_specials(
        tok.get_added_tokens_decoder()
            .into_values()
            .filter(|t| t.special)
            .map(|t| t.content),
    )
}

fn strip_specials(mut text: String, drop: &[String]) -> String {
    for d in drop {
        if text.contains(d.as_str()) {
            text = text.replace(d.as_str(), "");
        }
    }
    text
}

fn push_reasoning_if_no_answer(
    shared: &Shared,
    generated: &[u32],
    visible: &str,
    tx: &mpsc::Sender<ChatEvent>,
    timeout: std::time::Duration,
) {
    if shared.kind != WgpuModelKind::GptOss || !visible.trim().is_empty() || generated.is_empty() {
        return;
    }
    let Ok(raw) = shared.tokenizer.decode(generated, false) else {
        return;
    };
    let reasoning = crate::oapi::chat_template::harmony_reasoning_text(&raw);
    if !reasoning.trim().is_empty() {
        let _ = push_blocking(tx, ChatEvent::ReasoningDelta(reasoning), timeout);
    }
}

fn visible_text(shared: &Shared, generated: &[u32]) -> anyhow::Result<String> {
    let raw = shared
        .tokenizer
        .decode(generated, false)
        .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
    if shared.kind == WgpuModelKind::GptOss {
        return Ok(crate::oapi::chat_template::harmony_final_text(&raw));
    }
    Ok(strip_specials(raw, &shared.strip_specials))
}

const INCR_DETOK_ENV_DECODES_A_TRAILING_TOKEN_WINDOW_PER_STEP_INSTEAD_OF_THE_WHOLE_SEQUENCE: &str =
    "NV_WGPU_INCR_DETOK";

fn incr_detok_enabled(kind: WgpuModelKind) -> bool {
    kind != WgpuModelKind::GptOss
        && std::env::var(INCR_DETOK_ENV_DECODES_A_TRAILING_TOKEN_WINDOW_PER_STEP_INSTEAD_OF_THE_WHOLE_SEQUENCE)
            .ok()
            .as_deref()
            != Some("0")
}

struct VisibleStream {
    incremental: bool,
    full: String,
    prefix_tokens: usize,
    read_tokens: usize,
}

impl VisibleStream {
    fn new(kind: WgpuModelKind) -> Self {
        Self {
            incremental: incr_detok_enabled(kind),
            full: String::new(),
            prefix_tokens: 0,
            read_tokens: 0,
        }
    }

    fn step<'a>(&'a mut self, shared: &Shared, generated: &[u32]) -> anyhow::Result<&'a str> {
        if !self.incremental {
            self.full = visible_text(shared, generated)?;
            return Ok(&self.full);
        }
        self.step_incremental(&shared.tokenizer, &shared.strip_specials, generated)
    }

    fn step_incremental<'a>(
        &'a mut self,
        tokenizer: &tokenizers::Tokenizer,
        strip: &[String],
        generated: &[u32],
    ) -> anyhow::Result<&'a str> {
        let window = tokenizer
            .decode(&generated[self.prefix_tokens..], false)
            .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
        if window.ends_with('\u{FFFD}') {
            return Ok(&self.full);
        }
        let held = tokenizer
            .decode(&generated[self.prefix_tokens..self.read_tokens], false)
            .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
        if let Some(piece) = window.strip_prefix(held.as_str()) {
            if !piece.is_empty() {
                self.full
                    .push_str(&strip_specials(piece.to_string(), strip));
                self.prefix_tokens = self.read_tokens;
            }
        } else {
            let raw = tokenizer
                .decode(generated, false)
                .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
            self.full = strip_specials(raw, strip);
            self.prefix_tokens = generated.len().saturating_sub(1);
        }
        self.read_tokens = generated.len();
        Ok(&self.full)
    }
}

type Pending = (ChatGenerateRequest, mpsc::Sender<ChatEvent>);

fn batch_slots(shared: &Shared, capacity: usize, knobs: batch::BatchKnobs) -> usize {
    if !knobs.enabled() {
        return 1;
    }
    let Some(budget) = shared.budget else {
        return 1;
    };
    batch::Admission {
        budget,
        max_seq: shared.max_seq as u64,
        knobs,
    }
    .slots(capacity, None)
}

static BATCH_NOTICE: std::sync::Once = std::sync::Once::new();

fn collect_batch(
    shared: &Arc<Shared>,
    decoder: &Decoder,
    first: Pending,
    rx: &std::sync::mpsc::Receiver<Job>,
) -> (Vec<Pending>, Vec<Pending>) {
    let knobs = batch::BatchKnobs::from_env();
    if !knobs.enabled() {
        return (Vec::new(), vec![first]);
    }
    let capacity = batch::BatchStepper::batch_capacity(decoder);
    let slots = batch_slots(shared, capacity, knobs);
    if slots < 2 {
        BATCH_NOTICE.call_once(|| {
            tracing::info!(
                model = %shared.model_id,
                kind = shared.kind.label(),
                requested = knobs.max_batch,
                graph_capacity = capacity,
                budgeted_slots = slots,
                seam = %batch_route_refusal(shared.kind),
                "batched serving requested but unavailable; serving single-stream"
            );
        });
        return (Vec::new(), vec![first]);
    }
    if let Err(why) = batch_admits(shared.kind, &first.0) {
        tracing::debug!(model = %shared.model_id, reason = %why, "single-stream by admission");
        return (Vec::new(), vec![first]);
    }

    let mut batched = vec![first];
    let mut single: Vec<Pending> = Vec::new();
    let deadline = std::time::Instant::now() + knobs.window;
    while batched.len() < slots {
        match rx.try_recv() {
            Ok(Job::Generate { req, tx }) => {
                if batch_admits(shared.kind, &req).is_ok() {
                    batched.push((req, tx));
                } else {
                    single.push((req, tx));
                }
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                if std::time::Instant::now() >= deadline {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
        }
    }
    (batched, single)
}

fn prepare_prompt(shared: &Shared, req: &ChatGenerateRequest) -> Result<(Vec<u32>, usize), String> {
    let encoded = shared
        .tokenizer
        .encode(req.prompt.as_str(), false)
        .map_err(|e| format!("tokenize: {e}"))?;
    let mut ids: Vec<u32> = encoded.get_ids().to_vec();
    if let Some(bos) = shared.bos_token_id {
        if ids.first().copied() != Some(bos) {
            ids.insert(0, bos);
        }
    }
    if ids.is_empty() {
        return Err("empty prompt after tokenization".to_string());
    }
    if ids.len() >= shared.max_seq {
        return Err(format!(
            "prompt of {} tokens does not fit the {}-token wgpu KV window (set {MAX_SEQ_ENV})",
            ids.len(),
            shared.max_seq
        ));
    }
    let requested = if req.max_new_tokens == 0 {
        shared.default_max_new
    } else {
        req.max_new_tokens
    };
    let max_new = requested.min(shared.max_seq - ids.len());
    if max_new == 0 {
        return Err(format!(
            "no room to generate: prompt {} tokens fills the {}-token wgpu KV window",
            ids.len(),
            shared.max_seq
        ));
    }
    Ok((ids, max_new))
}

struct StreamSlot {
    shared: Arc<Shared>,
    tx: mpsc::Sender<ChatEvent>,
    timeout: std::time::Duration,
    scanner: StopScanner,
    visible: VisibleStream,
    generated: Vec<u32>,
    max_new: usize,
    finish_reason: &'static str,
    aborted: bool,
}

impl batch::SlotSink for StreamSlot {
    fn accept(&mut self, sampled: u32) -> anyhow::Result<batch::SlotStep> {
        if self.shared.eos_ids.contains(&sampled) {
            self.finish_reason = "stop";
            return Ok(batch::SlotStep::Done);
        }
        if self.tx.is_closed() {
            self.aborted = true;
            return Ok(batch::SlotStep::Done);
        }
        self.generated.push(sampled);
        let full = self.visible.step(&self.shared, &self.generated)?;
        let (piece, stop_hit) = self.scanner.step(full);
        if !piece.is_empty()
            && push_blocking(&self.tx, ChatEvent::TextDelta(piece), self.timeout) != Push::Sent
        {
            self.aborted = true;
            return Ok(batch::SlotStep::Done);
        }
        if stop_hit {
            self.finish_reason = "stop";
            return Ok(batch::SlotStep::Done);
        }
        if self.generated.len() >= self.max_new {
            self.finish_reason = "length";
            return Ok(batch::SlotStep::Done);
        }
        Ok(batch::SlotStep::Feed(sampled))
    }

    fn finish(&mut self) {
        if self.aborted {
            return;
        }
        if let Ok(full) = visible_text(&self.shared, &self.generated) {
            let tail = self.scanner.finish(&full);
            if !tail.is_empty() {
                let _ = push_blocking(&self.tx, ChatEvent::TextDelta(tail), self.timeout);
            }
            push_reasoning_if_no_answer(
                &self.shared,
                &self.generated,
                &full,
                &self.tx,
                self.timeout,
            );
        }
        if let Some(stop_sequence) = self.scanner.matched.take() {
            let _ = push_blocking(&self.tx, ChatEvent::StoppedBy { stop_sequence }, self.timeout);
        }
        let _ = push_blocking(
            &self.tx,
            ChatEvent::Done {
                finish_reason: self.finish_reason.to_string(),
                completion_tokens: self.generated.len() as u32,
            },
            self.timeout,
        );
    }
}

fn run_batch_blocking(
    shared: &Arc<Shared>,
    model: &mut dyn batch::BatchStepper,
    jobs: Vec<Pending>,
) -> Vec<Pending> {
    let timeout = send_timeout();
    let mut prepared: Vec<(
        ChatGenerateRequest,
        mpsc::Sender<ChatEvent>,
        Vec<u32>,
        usize,
    )> = Vec::with_capacity(jobs.len());
    for (req, tx) in jobs {
        match prepare_prompt(shared, &req) {
            Ok((ids, max_new)) => prepared.push((req, tx, ids, max_new)),
            Err(msg) => {
                let _ = push_blocking(&tx, ChatEvent::Error(msg), timeout);
            }
        }
    }
    let hand_back = |v: Vec<(
        ChatGenerateRequest,
        mpsc::Sender<ChatEvent>,
        Vec<u32>,
        usize,
    )>| { v.into_iter().map(|(r, t, _, _)| (r, t)).collect::<Vec<_>>() };
    if prepared.len() < 2 {
        return hand_back(prepared);
    }
    if let Err(e) = model.reset_batch(prepared.len()) {
        tracing::info!(
            model = %shared.model_id,
            error = %format!("{e:#}"),
            "batched decode refused by the graph; serving single-stream"
        );
        return hand_back(prepared);
    }

    let txs: Vec<mpsc::Sender<ChatEvent>> = prepared.iter().map(|p| p.1.clone()).collect();
    let broadcast = |msg: String| {
        for tx in &txs {
            let _ = push_blocking(tx, ChatEvent::Error(msg.clone()), timeout);
        }
    };
    let mut seeded = Vec::with_capacity(prepared.len());
    let mut max_steps = 0u64;
    let mut prompt_total = 0u64;
    for (slot, (req, tx, ids, max_new)) in prepared.into_iter().enumerate() {
        prompt_total += ids.len() as u64;
        max_steps = max_steps.max(max_new as u64 + 2);
        let _ = push_blocking(
            &tx,
            ChatEvent::Started {
                prompt_tokens: ids.len() as u32,
            },
            timeout,
        );
        let first = match model.prefill_slot(slot, &ids) {
            Ok(t) => t,
            Err(e) => {
                broadcast(format!("batched prefill failed at slot {slot}: {e:#}"));
                return Vec::new();
            }
        };
        seeded.push((
            StreamSlot {
                shared: shared.clone(),
                tx,
                timeout,
                scanner: StopScanner::new(&req.stop),
                visible: VisibleStream::new(shared.kind),
                generated: Vec::with_capacity(max_new),
                max_new,
                finish_reason: "length",
                aborted: false,
            },
            first,
        ));
    }

    let mut b = match batch::Batch::new(seeded) {
        Ok(b) => b,
        Err(e) => {
            broadcast(format!("{e:#}"));
            return Vec::new();
        }
    };
    shared
        .batches_run
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let drained = b.drain(model, max_steps);
    model.end_batch();
    let stats = match drained {
        Ok(s) => s,
        Err(e) => {
            broadcast(format!("batched decode failed: {e:#}"));
            return Vec::new();
        }
    };

    tracing::info!(
        model = %shared.model_id,
        kind = shared.kind.label(),
        batch = stats.batch,
        steps = stats.steps,
        prompt_tokens_total = prompt_total,
        completion_tokens_total = stats.emitted(),
        aggregate_tok_s = stats.aggregate_tok_s(),
        best_stream_ms_per_token = stats.best_stream_ms_per_token(),
        worst_stream_ms_per_token = stats.worst_stream_ms_per_token(),
        "wgpu batched chat request complete"
    );
    Vec::new()
}

fn run_blocking(
    shared: &Arc<Shared>,
    decoder: &mut Decoder,
    prefix: &mut PrefixCache,
    mut assistant: Option<&mut learned::AssistantSpecDrafter>,
    mm_rt: Option<&mm::MmRuntime>,
    req: ChatGenerateRequest,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    let prev = std::mem::take(prefix);
    let timeout = send_timeout();
    let fail = |tx: &mpsc::Sender<ChatEvent>, msg: String| {
        let _ = push_blocking(tx, ChatEvent::Error(msg), timeout);
    };

    let media = req.mm.as_ref().filter(|m| mm::media_present(m));
    let mm_active = media.is_some();
    if let Some(media) = media {
        if let Some(reason) = mm::refuse_media(
            shared.kind,
            mm_rt.map(mm::MmRuntime::caps),
            !media.images.is_empty(),
            !media.audios.is_empty(),
        ) {
            fail(tx, reason);
            return Ok(());
        }
    }
    let qwen3_mm = mm_rt.and_then(mm::MmRuntime::qwen3).filter(|_| mm_active);
    let gemma4_mm = mm_rt.and_then(mm::MmRuntime::gemma4).filter(|_| mm_active);
    let mut prepped: Vec<crate::oapi::chat_multimodal_qwen3::PreppedImage> = Vec::new();
    let prompt_string: String = match (media, qwen3_mm) {
        (Some(media), Some(q)) => {
            let spec = q.spec();
            prepped = match crate::oapi::chat_multimodal_qwen3::prep_images(spec, &media.images) {
                Ok(p) => p,
                Err(e) => {
                    fail(tx, format!("image preprocess failed: {e:#}"));
                    return Ok(());
                }
            };
            match crate::oapi::chat_multimodal_qwen3::expand_marker_prompt(
                spec,
                &req.prompt,
                &prepped,
            ) {
                Ok(s) => s,
                Err(e) => {
                    fail(tx, format!("image marker expansion failed: {e:#}"));
                    return Ok(());
                }
            }
        }
        _ => req.prompt.clone(),
    };

    let guided = match &req.guided {
        Some(spec) => {
            match crate::oapi::chat_engine::build_guided_for_request(
                &shared.tokenizer,
                &shared.eos_ids,
                spec,
                req.guided_think_close.as_deref(),
            ) {
                Ok(g) => Some(g),
                Err(e) => {
                    fail(tx, format!("guided grammar rejected: {e:#}"));
                    return Ok(());
                }
            }
        }
        None => None,
    };

    let encoded = match shared.tokenizer.encode(prompt_string.as_str(), false) {
        Ok(e) => e,
        Err(e) => {
            fail(tx, format!("tokenize: {e}"));
            return Ok(());
        }
    };
    let mut prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    if let Some(bos) = shared.bos_token_id {
        if prompt_ids.first().copied() != Some(bos) {
            prompt_ids.insert(0, bos);
        }
    }
    if prompt_ids.is_empty() {
        fail(tx, "empty prompt after tokenization".to_string());
        return Ok(());
    }
    let mut gemma4_plan: Option<crate::oapi::chat_multimodal::MmPlan> = None;
    if let (Some(g), Some(media)) = (gemma4_mm, media) {
        match g.plan(&prompt_ids, media) {
            Ok(plan) => {
                prompt_ids = plan.tokens.clone();
                gemma4_plan = Some(plan);
            }
            Err(e) => {
                fail(
                    tx,
                    format!(
                        "{}: gemma4 media marker expansion failed: {e:#}",
                        shared.kind.label()
                    ),
                );
                return Ok(());
            }
        }
    }
    if prompt_ids.len() >= shared.max_seq {
        fail(
            tx,
            format!(
                "prompt of {} tokens does not fit the {}-token wgpu KV window (set {MAX_SEQ_ENV})",
                prompt_ids.len(),
                shared.max_seq
            ),
        );
        return Ok(());
    }
    let prompt_tokens = prompt_ids.len() as u32;

    let requested = if req.max_new_tokens == 0 {
        shared.default_max_new
    } else {
        req.max_new_tokens
    };
    let max_new = requested.min(shared.max_seq - prompt_ids.len());
    if max_new == 0 {
        fail(
            tx,
            format!(
                "no room to generate: prompt {} tokens fills the {}-token wgpu KV window",
                prompt_ids.len(),
                shared.max_seq
            ),
        );
        return Ok(());
    }

    let reused = if mm_active {
        0
    } else {
        plan_prefix_reuse(decoder, &prev, &prompt_ids)?
    };
    if reused == 0 {
        decoder.reset()?;
    } else if push_blocking(
        tx,
        ChatEvent::PromptCached {
            cached_tokens: reused as u32,
        },
        timeout,
    ) != Push::Sent
    {
        return Ok(());
    }
    if push_blocking(tx, ChatEvent::Started { prompt_tokens }, timeout) != Push::Sent {
        return Ok(());
    }
    shared
        .prefix_tokens_reused
        .fetch_add(reused, std::sync::atomic::Ordering::Relaxed);

    let mut sampler = HostSampler::new(&req, req.seed.unwrap_or_else(os_seed));
    if let Some(g) = guided {
        sampler.set_guided(g, max_new);
    }
    sampler.seed_prompt(&prompt_ids);
    let needs_logits = sampler.needs_logits();

    let t_prefill = std::time::Instant::now();
    let last_idx = prompt_ids.len() - 1;
    let mut greedy = 0u32;
    let mut logits: Option<Vec<f32>> = None;
    let embed_splices: Vec<nv_models::embed_row_splice::EmbedRowSplice> = match (
        qwen3_mm,
        gemma4_mm.zip(gemma4_plan.as_ref()),
    ) {
        (Some(q), _) => {
            let mut embeds = Vec::with_capacity(prepped.len());
            for p in &prepped {
                embeds.push(q.encode(p)?);
            }
            let sp = q.splice_rows(&prompt_ids, &embeds)?;
            anyhow::ensure!(
                prompt_ids[last_idx] != q.spec().image_token_id,
                "the final prompt token must be text; the template's assistant header guarantees \
                 this"
            );
            if let Some(section) = q.spec().mrope_section {
                let grids: Vec<(usize, usize, usize)> = prepped
                    .iter()
                    .map(|p| {
                        let (gh, gw) = p.grid();
                        (1, gh / p.merge, gw / p.merge)
                    })
                    .collect();
                let mp = nv_models::qwen3_mm_splice::build_mrope_positions_matching_hf_get_rope_index(
                    &prompt_ids,
                    q.spec().image_token_id,
                    &grids,
                )?;
                if !mp.is_text_degenerate() {
                    decoder.install_qwen3_mrope_rows(&mp, section)?;
                }
            }
            sp
        }
        (None, Some((g, plan))) => {
            let sp = g.embed_rows(plan)?;
            anyhow::ensure!(
                g.modality_of_token(prompt_ids[last_idx]).is_none(),
                "the final prompt token must be text, not a media placeholder; the template's \
                 assistant header guarantees this"
            );
            sp
        }
        (None, None) => Vec::new(),
    };
    let chunked = if mm_active {
        anyhow::ensure!(
            chunked_prefill_enabled() && decoder.prefill_chunk_len() > 0,
            "media prefill requires the chunked prefill graph; unset \
             NV_WGPU_CHAT_CHUNKED_PREFILL=0 / raise NV_WGPU_PREFILL_M"
        );
        let done =
            decoder.prefill_tokens_with_embed_rows(&prompt_ids[..last_idx], &embed_splices)?;
        anyhow::ensure!(
            done == last_idx,
            "media prefill consumed {done} of {last_idx} prompt tokens"
        );
        done
    } else if chunked_prefill_enabled() && decoder.prefill_chunk_len() > 0 && last_idx >= 2 {
        reused + decoder.prefill_tokens(&prompt_ids[reused..last_idx])?
    } else {
        reused
    };
    for (i, t) in prompt_ids.iter().enumerate().skip(chunked) {
        if i == last_idx && needs_logits {
            let (g, l) = decoder.step_logits(*t)?;
            greedy = g;
            logits = Some(l);
        } else if i == last_idx {
            greedy = decoder.step(*t)?;
        } else {
            decoder.prefill_step(*t)?;
        }
    }
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    let mut picked = match &logits {
        Some(l) => sampler.pick(clip_logits(l, shared.vocab)?)?,
        None => Picked {
            token: greedy,
            logprob: None,
            top: Vec::new(),
        },
    };
    let mut last = picked.token;

    let knobs = spec::SpecKnobs::from_env();
    let mut spec_state: Option<(spec::SpecLoop, VecDeque<u32>)> =
        if spec_route_eligible(shared.kind, needs_logits, knobs) && decoder.has_chain_verify() {
            let mut sl = spec::SpecLoop::new(knobs);
            sl.prime(&prompt_ids);
            sl.prime(&[last]);
            Some((sl, VecDeque::new()))
        } else {
            None
        };

    let ema_floor_override = std::env::var(spec::SPEC_ASSISTANT_EMA_FLOOR_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok());
    let gate = &shared.spec_gate;
    let probe_rounds = spec::assistant_probe_rounds();
    let probe_min_match = std::env::var(spec::SPEC_PROBE_MIN_MATCH_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(spec::SPEC_PROBE_MIN_MATCH_DEFAULT)
        .max(1);
    let mut drafter_hidden: Option<Vec<f32>> = None;
    let mut drafter_hidden_loc: Option<learned::HiddenLoc> = None;
    let mtp_on = matches!(&*decoder, Decoder::Qwen3_5Dense(m) if m.mtp_active());
    let mtp_k = nv_specdecode::qwen38_mtp::mtp_chain_depth_from_env();
    let chain_k = if matches!(decoder, Decoder::Gemma4E4b(_)) && spec_state.is_none() {
        nv_models::gemma4_e4b_wgpu::chain_k_from_env()
    } else {
        1
    };
    let mut chain_pending: VecDeque<u32> = VecDeque::new();
    let mut generated: Vec<u32> = Vec::with_capacity(max_new);
    let mut scanner = StopScanner::new(&req.stop);
    let mut visible = VisibleStream::new(shared.kind);
    let mut completion_tokens = 0u32;
    let mut finish_reason = "length".to_string();
    let mut aborted = false;
    let gpu_profile = nv_kernels::wgpu_backend::dispatch::profile::enabled();
    let profile_pos0 = decoder.current_pos();
    if gpu_profile {
        let rows = nv_kernels::wgpu_backend::dispatch::profile::report();
        let mut table = format!(
            "== {} prefill ({} tokens, {prefill_s:.3}s wall): per-dispatch GPU profile ==\n",
            shared.model_id,
            decoder.current_pos()
        );
        let mut gpu_ms = 0.0;
        for (label, count, ns) in &rows {
            gpu_ms += ns / 1e6;
            table.push_str(&format!("{label:<48} {count:>8}  {:>10.3} ms\n", ns / 1e6));
        }
        eprintln!("{table}  prefill GPU-attributed {gpu_ms:.3} ms");
        nv_kernels::wgpu_backend::dispatch::profile::reset();
    }
    let t_decode = std::time::Instant::now();

    for step in 0..max_new {
        if tx.is_closed() {
            aborted = true;
            break;
        }
        if shared.eos_ids.contains(&last) {
            finish_reason = "stop".into();
            break;
        }
        generated.push(last);
        completion_tokens = generated.len() as u32;

        let full = visible.step(shared, &generated)?;
        let (piece, stop_hit) = scanner.step(full);
        if !piece.is_empty()
            && push_blocking(tx, ChatEvent::TextDelta(piece), timeout) != Push::Sent
        {
            aborted = true;
            break;
        }
        if req.logprobs {
            let entry = logprob_entry(&shared.tokenizer, last, picked.logprob, &picked.top);
            if push_blocking(tx, ChatEvent::Logprob(entry), timeout) != Push::Sent {
                aborted = true;
                break;
            }
        }
        if stop_hit {
            finish_reason = "stop".into();
            break;
        }
        if step + 1 >= max_new {
            break;
        }

        if needs_logits {
            let (_, l) = decoder.step_logits(last)?;
            picked = sampler.pick(clip_logits(&l, shared.vocab)?)?;
        } else if let Some((sl, pending)) = spec_state.as_mut() {
            if pending.is_empty() {
                anyhow::ensure!(
                    decoder.has_chain_verify(),
                    "{CHAIN_VERIFY_SEAM}; the spec decode loop reached {}",
                    decoder.kind_label()
                );
                let t_round = std::time::Instant::now();
                let drafted_before = sl.stats().rounds_with_draft;
                let assistant_on = assistant.is_some()
                    && match ema_floor_override {
                        Some(f) => sl.ema_value() >= f,
                        None => gate.should_draft(probe_rounds),
                    };
                let mut learned_draft: Option<Vec<u32>> = None;
                if assistant_on {
                    let (rows, span) = decoder.chain_verify_geometry();
                    let cap =
                        spec::chain_capacity(rows, decoder.current_pos(), span, shared.max_seq);
                    if let (Some(a), Decoder::Gemma4E4b(m)) = (assistant.as_mut(), &mut *decoder) {
                        let want = sl.knobs().k.min(cap.saturating_sub(1));
                        let proposed = if want == 0 {
                            None
                        } else if a.gpu_active() {
                            drafter_hidden_loc.map(|loc| a.propose_loc(m, last, loc, want))
                        } else {
                            drafter_hidden
                                .as_deref()
                                .map(|h| a.propose(m, last, h, want))
                        };
                        match proposed {
                            Some(Ok(d)) if !d.is_empty() => learned_draft = Some(d),
                            Some(Ok(_)) | None => {}
                            Some(Err(e)) => tracing::warn!(
                                error = %format!("{e:#}"),
                                "assistant drafter propose failed; suffix fallback this round"
                            ),
                        }
                    }
                }
                let mtp_draft: Option<Vec<u32>> = if let Decoder::Qwen3_5Dense(m) = &mut *decoder {
                    if m.mtp_active() {
                        let cap = spec::chain_capacity(
                            m.verify_max_rows(),
                            m.current_pos(),
                            m.prefill_chunk_len(),
                            shared.max_seq,
                        );
                        let want = mtp_k.min(cap.saturating_sub(1));
                        let draft_now = match ema_floor_override {
                            Some(f) => sl.ema_value() >= f,
                            None => gate.should_draft(probe_rounds),
                        };
                        if want >= 1 && draft_now {
                            Some(m.mtp_draft_round(last, want)?)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                let mtp_drafted = mtp_draft.is_some();
                let gpu_drafter = assistant.as_ref().is_some_and(|a| a.gpu_active());
                let mut tgt = DecoderChainTarget {
                    decoder: &mut *decoder,
                    max_seq: shared.max_seq,
                    stepped: false,
                    want_hidden: assistant_on,
                    want_hidden_host: assistant_on && !gpu_drafter,
                    hidden_row: None,
                    hidden_loc: None,
                };
                let emitted = match mtp_draft {
                    Some(d) => sl.round_with_draft(&mut tgt, last, d)?,
                    None if mtp_on => sl.round_with_draft(&mut tgt, last, Vec::new())?,
                    None => match learned_draft {
                        Some(d) => sl.round_with_draft(&mut tgt, last, d)?,
                        None if assistant.is_some() => {
                            let probe = sl.propose_draft_min(
                                spec::ChainVerifyTarget::capacity(&tgt),
                                probe_min_match,
                            );
                            sl.round_with_draft(&mut tgt, last, probe)?
                        }
                        None => sl.round(&mut tgt, last)?,
                    },
                };
                drafter_hidden = tgt.hidden_row.take();
                drafter_hidden_loc = tgt.hidden_loc.take();
                if mtp_drafted {
                    if let Decoder::Qwen3_5Dense(m) = &mut *decoder {
                        m.mtp_post_verify(&emitted[..emitted.len() - 1])?;
                    }
                }
                let round_ns = t_round.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
                if sl.stats().rounds_with_draft > drafted_before {
                    gate.observe_spec(round_ns, emitted.len());
                } else {
                    gate.observe_decode(round_ns);
                }
                pending.extend(emitted);
            }
            let token = pending
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("spec round emitted no tokens"))?;
            picked = Picked {
                token,
                logprob: None,
                top: Vec::new(),
            };
        } else if chain_k > 1 {
            if chain_pending.is_empty() {
                let Decoder::Gemma4E4b(m) = decoder else {
                    anyhow::bail!("chained decode loop reached a non-E4B decoder");
                };
                let room = shared.max_seq.saturating_sub(m.current_pos()).max(1);
                let left = max_new.saturating_sub(step + 1).max(1);
                let want = chain_k.min(room).min(left);
                let pos0 = m.current_pos();
                let mut batch = m.decode_chain(last, want)?;
                if let Some(j) = batch.iter().position(|t| shared.eos_ids.contains(t)) {
                    if j + 1 < batch.len() {
                        batch.truncate(j + 1);
                        m.truncate_to(pos0 + j + 1)?;
                    }
                }
                chain_pending.extend(batch);
            }
            let token = chain_pending
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("chained decode emitted no tokens"))?;
            picked = Picked {
                token,
                logprob: None,
                top: Vec::new(),
            };
        } else if decode_pipe_opted_in() && matches!(decoder, Decoder::Gemma4E4b(_)) {
            let Decoder::Gemma4E4b(m) = decoder else {
                unreachable!()
            };
            let token = if m.decode_pipe_inflight() == 0 {
                m.decode_step_pipelined(Some(last))?
            } else {
                m.decode_step_pipelined(None)?
            };
            picked = Picked {
                token,
                logprob: None,
                top: Vec::new(),
            };
        } else {
            picked = Picked {
                token: decoder.step(last)?,
                logprob: None,
                top: Vec::new(),
            };
        }
        last = picked.token;
    }
    if let Decoder::Gemma4E4b(m) = decoder {
        if m.decode_pipe_inflight() > 0 {
            m.decode_pipe_abort()?;
        }
    }
    let decode_s = t_decode.elapsed().as_secs_f64();

    if gpu_profile {
        let per_step = decoder.pass_count();

        let steps = decoder.current_pos().saturating_sub(profile_pos0);
        let rows = nv_kernels::wgpu_backend::dispatch::profile::report();
        let seen: u64 = rows.iter().map(|(_, c, _)| *c).sum();
        let mut table = format!(
            "== {} decode: per-dispatch GPU profile ==\n",
            shared.model_id
        );
        for (label, count, ns) in &rows {
            table.push_str(&format!("{label:<48} {count:>8}  {:>10.3} ms\n", ns / 1e6));
        }
        let gpu_ms = nv_kernels::wgpu_backend::dispatch::profile::total_ns() / 1e6;

        let ts = nv_kernels::wgpu_backend::WgpuContext::shared()
            .map(|c| c.caps.timestamp_query)
            .unwrap_or(false);
        eprintln!(
            "{table}  timestamp_query={ts}; \
             wall {:.3} ms / {steps} graph steps ({completion_tokens} tokens emitted); \
             GPU-attributed {gpu_ms:.3} ms = {:.1}% of wall; \
             dispatches seen {seen} of {} expected ({} missed)",
            decode_s * 1000.0,
            if decode_s > 0.0 {
                gpu_ms / (decode_s * 1000.0) * 100.0
            } else {
                0.0
            },
            per_step * steps,
            (per_step * steps) as i64 - seen as i64,
        );
    }

    if !aborted && !generated.is_empty() {
        if let Ok(full) = visible_text(shared, &generated) {
            let tail = scanner.finish(&full);
            if !tail.is_empty() {
                let _ = push_blocking(tx, ChatEvent::TextDelta(tail), timeout);
            }
            push_reasoning_if_no_answer(shared, &generated, &full, tx, timeout);
        }
    }

    if let Some((sl, _)) = &spec_state {
        let stats = sl.stats();
        use std::sync::atomic::Ordering::Relaxed;
        shared.spec_rounds.store(stats.rounds, Relaxed);
        shared
            .spec_rounds_with_draft
            .store(stats.rounds_with_draft, Relaxed);
        shared.spec_drafted.store(stats.drafted, Relaxed);
        shared.spec_accepted.store(stats.accepted, Relaxed);
        shared.spec_emitted.store(stats.emitted, Relaxed);
        tracing::info!(
            model = %shared.model_id,
            spec = %stats.summary(),
            accept_ema = sl.ema_value(),
            drafter = if assistant.is_some() {
                "assistant"
            } else if mtp_on {
                "mtp"
            } else {
                "suffix"
            },
            gate = %gate.summary(),
            gate_drafting = gate.should_draft(probe_rounds),
            "wgpu spec decode request stats"
        );
    }

    tracing::info!(
        model = %shared.model_id,
        kind = shared.kind.label(),
        prompt_tokens,
        completion_tokens,
        sampling = if needs_logits { "host-logits" } else { "in-shader-argmax" },
        prefill_ms = prefill_s * 1000.0,
        ms_per_token = if completion_tokens > 0 {
            decode_s * 1000.0 / completion_tokens as f64
        } else {
            0.0
        },
        kv_pos = decoder.current_pos(),
        prefix_reused = reused,
        "wgpu chat request complete"
    );

    if !mm_active {
        let mut folded = prompt_ids;
        folded.extend_from_slice(&generated);
        *prefix = PrefixCache {
            tokens: folded,
            frontier: decoder.current_pos(),
        };
    }

    if !aborted {
        if let Some(stop_sequence) = scanner.matched.take() {
            let _ = push_blocking(tx, ChatEvent::StoppedBy { stop_sequence }, timeout);
        }
        let _ = push_blocking(
            tx,
            ChatEvent::Done {
                finish_reason,
                completion_tokens,
            },
            timeout,
        );
    }
    Ok(())
}

pub fn engine_from_env_dirs() -> anyhow::Result<Vec<Arc<dyn ChatEngine>>> {
    let raw = std::env::var("NV_WGPU_CHAT_MODEL_DIRS")
        .or_else(|_| std::env::var("NV_CHAT_MODEL_DIR"))
        .map_err(|_| anyhow::anyhow!("set NV_WGPU_CHAT_MODEL_DIRS or NV_CHAT_MODEL_DIR"))?;
    let dirs: Vec<PathBuf> = raw
        .split([',', ':'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    anyhow::ensure!(!dirs.is_empty(), "no model directories configured");
    if let Some((verdicts, available_gib)) = mem_fit::plan(&dirs) {
        let oversized: Vec<String> = dirs
            .iter()
            .zip(verdicts.iter())
            .filter_map(|(d, v)| match v {
                mem_fit::Fit::WontFit { estimated_gib, .. } => {
                    Some(format!("{} (~{estimated_gib:.1} GiB)", model_id_for_dir(d)))
                }
                mem_fit::Fit::Fits => None,
            })
            .collect();
        if !oversized.is_empty() {
            anyhow::bail!(
                "refusing to load {} of {} requested chat models -- estimated wired footprint \
                 exceeds available memory (~{available_gib:.1} GiB free): {}. Free memory, \
                 request fewer models, or serve one chat model per co-tenant box.",
                oversized.len(),
                dirs.len(),
                oversized.join(", ")
            );
        }
    }
    let mut out: Vec<Arc<dyn ChatEngine>> = Vec::with_capacity(dirs.len());
    let mut bases: Vec<(String, PathBuf)> = Vec::with_capacity(dirs.len());
    for d in dirs {
        let eng = WgpuChatEngine::load(&d)?;
        bases.push((eng.model_id().to_string(), d));
        out.push(Arc::new(eng));
    }
    attach_env_adapters(&mut out, &bases);
    record_wgpu_ids(&out);
    Ok(out)
}

pub const WGPU_CHAT_MODEL_DIRS_ENV: &str = "NV_WGPU_CHAT_MODEL_DIRS";
pub const SERVE_BACKEND_ENV: &str = "NV_SERVE_BACKEND";
pub const CHAT_MODEL_DIRS_ENV: &str = "NV_CHAT_MODEL_DIRS";
pub const CHAT_MODEL_DIR_ENV: &str = "NV_CHAT_MODEL_DIR";

pub const WGPU_ALIAS_SUFFIX: &str = "#wgpu";

pub fn split_model_dirs(raw: &str) -> Vec<PathBuf> {
    raw.split([',', ':'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgpuRegistryPlan {
    Delegate,

    Extend(Vec<PathBuf>),

    Replace(Vec<PathBuf>),
}

impl WgpuRegistryPlan {
    pub fn decide(
        backend: Option<&str>,
        wgpu_dirs: Option<&str>,
        chat_model_dirs: Option<&str>,
        chat_model_dir: Option<&str>,
    ) -> Self {
        let wants_wgpu = backend
            .map(|b| b.trim().eq_ignore_ascii_case("wgpu"))
            .unwrap_or(false);
        let explicit = wgpu_dirs.map(split_model_dirs).unwrap_or_default();

        if wants_wgpu {
            let dirs = if !explicit.is_empty() {
                explicit
            } else {
                let listed = chat_model_dirs.map(split_model_dirs).unwrap_or_default();
                if listed.is_empty() {
                    chat_model_dir.map(split_model_dirs).unwrap_or_default()
                } else {
                    listed
                }
            };
            return if dirs.is_empty() {
                Self::Delegate
            } else {
                Self::Replace(dirs)
            };
        }

        if explicit.is_empty() {
            Self::Delegate
        } else {
            Self::Extend(explicit)
        }
    }

    pub fn from_env() -> Self {
        let backend = std::env::var(SERVE_BACKEND_ENV).ok();
        let wgpu_dirs = std::env::var(WGPU_CHAT_MODEL_DIRS_ENV).ok();
        let listed = std::env::var(CHAT_MODEL_DIRS_ENV).ok();
        let single = std::env::var(CHAT_MODEL_DIR_ENV).ok();
        Self::decide(
            backend.as_deref(),
            wgpu_dirs.as_deref(),
            listed.as_deref(),
            single.as_deref(),
        )
    }

    pub fn dirs(&self) -> &[PathBuf] {
        match self {
            Self::Delegate => &[],
            Self::Extend(d) | Self::Replace(d) => d,
        }
    }
}

pub fn wgpu_id_after_collisions(taken: &[String], id: &str) -> String {
    if taken.iter().any(|t| t == id) {
        format!("{id}{WGPU_ALIAS_SUFFIX}")
    } else {
        id.to_string()
    }
}

struct AliasedEngine {
    id: String,
    inner: Arc<dyn ChatEngine>,
}

#[async_trait::async_trait]
impl ChatEngine for AliasedEngine {
    fn model_id(&self) -> &str {
        &self.id
    }

    fn spec_decode_status(&self) -> Option<&'static str> {
        self.inner.spec_decode_status()
    }

    async fn generate(
        &self,
        req: ChatGenerateRequest,
        tx: mpsc::Sender<ChatEvent>,
    ) -> anyhow::Result<()> {
        self.inner.generate(req, tx).await
    }

    fn render_prompt(&self, messages: &[ChatMessageIn]) -> String {
        self.inner.render_prompt(messages)
    }

    fn official_template(&self) -> Option<&ChatTemplate> {
        self.inner.official_template()
    }

    fn render_chat(
        &self,
        messages: &[ChatMessageIn],
        tools: &[Tool],
        choice: &ToolChoice,
    ) -> String {
        self.inner.render_chat(messages, tools, choice)
    }

    fn render_chat_kwargs(
        &self,
        messages: &[ChatMessageIn],
        tools: &[Tool],
        choice: &ToolChoice,
        extra: &TemplateKwargs,
    ) -> String {
        self.inner
            .render_chat_kwargs(messages, tools, choice, extra)
    }

    fn thinking_split_supported(&self) -> bool {
        self.inner.thinking_split_supported()
    }
}

pub fn alias_engine(id: String, inner: Arc<dyn ChatEngine>) -> Arc<dyn ChatEngine> {
    Arc::new(AliasedEngine { id, inner })
}

static WGPU_SERVED_IDS: std::sync::OnceLock<std::sync::Mutex<Vec<String>>> =
    std::sync::OnceLock::new();

fn served_ids() -> &'static std::sync::Mutex<Vec<String>> {
    WGPU_SERVED_IDS.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

pub fn registered_wgpu_model_ids() -> Vec<String> {
    served_ids().lock().map(|v| v.clone()).unwrap_or_default()
}

fn record_wgpu_ids(engines: &[Arc<dyn ChatEngine>]) {
    if let Ok(mut v) = served_ids().lock() {
        for e in engines {
            let id = e.model_id().to_string();
            if !v.contains(&id) {
                v.push(id);
            }
        }
    }
}

fn load_wgpu_engines(dirs: &[PathBuf]) -> Vec<Arc<dyn ChatEngine>> {
    let mut out: Vec<Arc<dyn ChatEngine>> = Vec::with_capacity(dirs.len());
    let mut bases: Vec<(String, PathBuf)> = Vec::with_capacity(dirs.len());
    let fit_plan = mem_fit::plan(dirs);
    for (i, d) in dirs.iter().enumerate() {
        if let Some((verdicts, available_gib)) = &fit_plan {
            if let mem_fit::Fit::WontFit { estimated_gib, .. } = verdicts[i] {
                tracing::error!(
                    model = %model_id_for_dir(d),
                    dir = %d.display(),
                    estimated_gib = format!("{estimated_gib:.1}"),
                    available_gib = format!("{available_gib:.1}"),
                    "refusing to load chat model: estimated wired footprint exceeds available \
                     memory; loading the remaining requested models that fit instead of risking \
                     a silent OOM kill. Free memory, request fewer models, or serve one chat \
                     model per co-tenant box."
                );
                continue;
            }
        }
        match WgpuChatEngine::load(d) {
            Ok(e) => {
                bases.push((e.model_id().to_string(), d.clone()));
                out.push(Arc::new(e));
            }
            Err(err) => {
                tracing::error!(
                    dir = %d.display(),
                    "WgpuChatEngine::load failed for {}: {err:#}",
                    d.display()
                )
            }
        }
    }
    attach_env_adapters(&mut out, &bases);
    out
}

pub fn registry_from_env_with_wgpu() -> Option<ChatRegistry> {
    let plan = match WgpuRegistryPlan::from_env() {
        WgpuRegistryPlan::Extend(dirs) if !cfg!(feature = "cuda") => {
            tracing::debug!(
                "{WGPU_CHAT_MODEL_DIRS_ENV} on a build without the cuda backend: no separate \
                 base registry exists to extend, so the listed dirs are served once instead of \
                 being loaded twice under a {WGPU_ALIAS_SUFFIX} alias"
            );
            WgpuRegistryPlan::Replace(dirs)
        }
        other => other,
    };
    match plan {
        WgpuRegistryPlan::Delegate => crate::oapi::chat_engine::registry_from_env(),
        WgpuRegistryPlan::Replace(dirs) => {
            let engines = load_wgpu_engines(&dirs);
            if engines.is_empty() {
                panic!(
                    "WgpuChatEngine::load failed for every directory in {}: {}\n\
                     Refusing to start with no chat engine. Fix the model load \
                     error above, point {} elsewhere, or unset it (and {}=wgpu) \
                     to let the cuda selection point serve.",
                    WGPU_CHAT_MODEL_DIRS_ENV,
                    dirs.iter()
                        .map(|d| d.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    WGPU_CHAT_MODEL_DIRS_ENV,
                    SERVE_BACKEND_ENV,
                );
            }
            record_wgpu_ids(&engines);
            ChatRegistry::from_engines(engines)
        }
        WgpuRegistryPlan::Extend(dirs) => {
            let base = crate::oapi::chat_engine::registry_from_env();
            let mut all: Vec<Arc<dyn ChatEngine>> = Vec::new();
            let mut taken: Vec<String> = Vec::new();
            if let Some(base) = &base {
                for id in base.model_ids() {
                    if let Some(e) = base.engines.get(id) {
                        all.push(e.clone());
                        taken.push(id.clone());
                    }
                }
            }
            let mut added: Vec<Arc<dyn ChatEngine>> = Vec::new();
            for e in load_wgpu_engines(&dirs) {
                let id = wgpu_id_after_collisions(&taken, e.model_id());
                taken.push(id.clone());
                added.push(if id == e.model_id() {
                    e
                } else {
                    alias_engine(id, e)
                });
            }
            record_wgpu_ids(&added);
            all.extend(added);
            ChatRegistry::from_engines(all)
        }
    }
}

fn attach_env_adapters(out: &mut Vec<Arc<dyn ChatEngine>>, bases: &[(String, PathBuf)]) {
    let configured = std::env::var_os(lora::ADAPTER_DIRS_ENV).is_some();
    let mut cat = lora::catalog_from_env();
    let base_ids: Vec<String> = bases.iter().map(|(id, _)| id.clone()).collect();
    cat.drop_ids(&base_ids, "adapter id collides with a loaded base model id");

    let mut registered: Vec<lora::AdapterEntry> = Vec::new();
    for entry in cat.entries() {
        let base = match select_base(entry.base_model.as_deref(), bases) {
            Ok(b) => b,
            Err(reason) => {
                tracing::warn!(
                    adapter = %entry.id,
                    reason = %reason,
                    "lora adapter skipped; the server continues without it"
                );
                continue;
            }
        };
        match WgpuChatEngine::load_with_lora(
            &base.1,
            WgpuChatEngine::default_max_seq_for(&base.1),
            Some(entry.id.clone()),
            Some(&entry.dir),
        ) {
            Ok(eng) => {
                tracing::info!(
                    adapter = %entry.id,
                    base = %base.0,
                    dir = %entry.dir.display(),
                    "lora adapter registered as a chat model id"
                );
                out.push(Arc::new(eng));
                registered.push(entry.clone());
            }
            Err(err) => tracing::warn!(
                adapter = %entry.id,
                dir = %entry.dir.display(),
                error = %format!("{err:#}"),
                "lora adapter failed to load; the server continues without it"
            ),
        }
    }
    lora::publish_served(lora::AdapterCatalog::from_entries(registered));
    lora::warn_if_typo_can_be_swallowed(out.len(), configured);
}

pub fn select_base<'a>(
    declared: Option<&str>,
    bases: &'a [(String, PathBuf)],
) -> Result<&'a (String, PathBuf), String> {
    if bases.is_empty() {
        return Err("no base chat model is loaded".into());
    }
    if let Some(want) = declared {
        let leaf = want.rsplit('/').next().unwrap_or(want);
        if let Some(hit) = bases
            .iter()
            .find(|(id, _)| id == want || id == leaf || id.rsplit('/').next() == Some(leaf))
        {
            return Ok(hit);
        }
        if bases.len() > 1 {
            return Err(format!(
                "base_model_name_or_path {want:?} matches none of the loaded models {:?}",
                bases.iter().map(|(id, _)| id).collect::<Vec<_>>()
            ));
        }
        tracing::warn!(
            declared = %want,
            attached = %bases[0].0,
            "lora adapter declares a base model that is not the one loaded; attaching anyway \
             because exactly one base model is resident"
        );
    }
    Ok(&bases[0])
}

#[cfg(test)]
mod fast_sampler_tests {
    use super::*;
    use nv_layers::sampler::{sample_token_checked, SamplingParams};

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) as f32
    }

    fn synth(n: usize, seed: u64, spread: f32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n).map(|_| (lcg(&mut s) - 0.5) * spread).collect()
    }

    fn param_grid() -> Vec<SamplingParams> {
        let mut v = Vec::new();
        for &t in &[0.1f32, 0.7, 1.0, 1.7] {
            for &k in &[None, Some(1usize), Some(2), Some(40), Some(257)] {
                for &tp in &[None, Some(0.1f32), Some(0.5), Some(0.9), Some(0.999)] {
                    for &mp in &[None, Some(0.05f32)] {
                        v.push(SamplingParams {
                            temperature: t,
                            top_k: k,
                            top_p: tp,
                            min_p: mp,
                            ..Default::default()
                        });
                    }
                }
            }
        }
        v
    }

    #[test]
    fn fast_path_token_matches_nv_layers_sampler_exactly() {
        let mut checked = 0usize;
        let mut applied = 0usize;
        for (vi, &vocab) in [37usize, 512, 4096].iter().enumerate() {
            for spread in [0.5f32, 6.0, 40.0] {
                let logits = synth(vocab, 0xC0FFEE + vi as u64 * 7919, spread);
                for p in param_grid() {
                    for step in 0..37u32 {
                        let u = step as f32 / 37.0;
                        let want = sample_token_checked(&logits, &p, u);
                        let fast = fast_sample_checked(&logits, &p, u);
                        checked += 1;
                        if let Some(got) = fast {
                            applied += 1;
                            assert_eq!(
                                got, want,
                                "vocab={vocab} spread={spread} params={p:?} u={u}"
                            );
                        }
                    }
                }
            }
        }
        println!("FASTSAMPLE draws={checked} fast_path_taken={applied}");
        assert!(
            applied * 3 > checked,
            "fast path covered too little: {applied}/{checked}"
        );
    }

    #[test]
    fn fast_path_matches_under_heavy_ties() {
        let mut logits = vec![2.0f32; 1024];
        for (i, v) in logits.iter_mut().enumerate() {
            if i % 7 == 0 {
                *v = 5.0;
            }
            if i % 101 == 0 {
                *v = -3.0;
            }
        }
        for p in param_grid() {
            for step in 0..53u32 {
                let u = step as f32 / 53.0;
                if let Some(got) = fast_sample_checked(&logits, &p, u) {
                    assert_eq!(
                        got,
                        sample_token_checked(&logits, &p, u),
                        "tie case params={p:?} u={u}"
                    );
                }
            }
        }
    }

    #[test]
    fn top_k_one_is_greedy_and_temperature_zero_is_greedy() {
        let logits = synth(2048, 42, 9.0);
        let am = nv_layers::sampler::argmax_checked(&logits).unwrap();
        let k1 = SamplingParams {
            temperature: 0.8,
            top_k: Some(1),
            ..Default::default()
        };
        for step in 0..29u32 {
            let u = step as f32 / 29.0;
            assert_eq!(sample_token_exact(&logits, &k1, u), Some(am));
        }
        let t0 = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(fast_sample_checked(&logits, &t0, 0.9), None);
        assert_eq!(sample_token_exact(&logits, &t0, 0.9), Some(am));
    }

    #[test]
    fn top_p_one_equals_pure_temperature_sampling() {
        let logits = synth(777, 9, 7.0);
        let pure = SamplingParams {
            temperature: 0.9,
            ..Default::default()
        };
        let p1 = SamplingParams {
            temperature: 0.9,
            top_p: Some(1.0),
            ..Default::default()
        };
        for step in 0..101u32 {
            let u = step as f32 / 101.0;
            assert_eq!(
                sample_token_exact(&logits, &p1, u),
                sample_token_exact(&logits, &pure, u),
                "u={u}"
            );
        }
    }

    #[test]
    fn degenerate_inputs_do_not_divide_by_zero() {
        let p = SamplingParams {
            temperature: 0.7,
            top_k: Some(8),
            top_p: Some(0.9),
            ..Default::default()
        };
        assert_eq!(fast_sample_checked(&[], &p, 0.5), None);
        let flat = vec![0.0f32; 64];
        assert_eq!(
            fast_sample_checked(&flat, &p, 0.5),
            Some(sample_token_checked(&flat, &p, 0.5))
        );
        let huge = vec![1.0e30f32; 64];
        assert_eq!(
            fast_sample_checked(&huge, &p, 0.5),
            Some(sample_token_checked(&huge, &p, 0.5))
        );
        let nan = vec![f32::NAN, 1.0, 2.0, 3.0];
        assert_eq!(
            fast_sample_checked(&nan, &p, 0.5),
            None,
            "non-finite must defer to the host sampler"
        );
        let inf = vec![f32::INFINITY, 1.0, 2.0, 3.0];
        assert_eq!(fast_sample_checked(&inf, &p, 0.5), None);
    }

    #[test]
    fn empirical_frequencies_track_the_host_distribution_tail() {
        let logits = synth(4096, 77, 8.0);
        let p = SamplingParams {
            temperature: 0.9,
            top_k: Some(32),
            top_p: Some(0.95),
            ..Default::default()
        };
        let want = nv_layers::sampler::distribution(&logits, &p);
        let n = 120_000u32;
        let mut counts = std::collections::HashMap::<u32, u32>::new();
        for step in 0..n {
            let u = (step as f64 + 0.5) as f32 / n as f32;
            let t = sample_token_exact(&logits, &p, u).unwrap();
            *counts.entry(t).or_insert(0) += 1;
        }
        let support: Vec<usize> = (0..want.len()).filter(|&i| want[i] > 0.0).collect();
        assert!(
            support.len() > 4,
            "test needs a real tail, got {}",
            support.len()
        );
        for &i in &support {
            let emp = counts.get(&(i as u32)).copied().unwrap_or(0) as f32 / n as f32;
            assert!(
                (emp - want[i]).abs() < 1e-3,
                "token {i}: empirical {emp} vs host {}",
                want[i]
            );
        }
        for (&t, &c) in &counts {
            assert!(
                want[t as usize] > 0.0,
                "token {t} sampled {c} times but has zero host mass"
            );
        }
    }

    fn peaked(n: usize, seed: u64) -> Vec<f32> {
        let mut v = synth(n, seed, 4.0);
        let mut s = seed ^ 0x5DEECE66D;
        for j in 0..96usize {
            let i = {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 20) as usize % n
            };
            v[i] += 15.0 - 0.14 * j as f32;
        }
        v
    }

    #[test]
    fn fast_path_cost_at_serving_vocab_sizes() {
        let iters = 20usize;
        let cases: [(&str, SamplingParams); 3] = [
            (
                "temp+top_p",
                SamplingParams {
                    temperature: 0.8,
                    top_p: Some(0.95),
                    ..Default::default()
                },
            ),
            (
                "temp+top_k",
                SamplingParams {
                    temperature: 0.8,
                    top_k: Some(40),
                    ..Default::default()
                },
            ),
            (
                "temp+top_k+top_p",
                SamplingParams {
                    temperature: 0.8,
                    top_k: Some(40),
                    top_p: Some(0.95),
                    ..Default::default()
                },
            ),
        ];
        for &vocab in &[151936usize, 262144] {
            for (shape, logits) in [
                ("flat-random", synth(vocab, 0xA5A5_1234, 24.0)),
                ("peaked", peaked(vocab, 0xA5A5_1234)),
            ] {
                for (label, p) in &cases {
                    let mut sink = 0u64;
                    let t = std::time::Instant::now();
                    for _ in 0..iters {
                        sink = sink.wrapping_add(match fast_sample_checked(&logits, p, 0.4242) {
                            Some(Some(tok)) => tok as u64,
                            _ => u64::MAX,
                        });
                    }
                    let fast = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
                    let taken = fast_sample_checked(&logits, p, 0.4242).is_some();
                    let t = std::time::Instant::now();
                    for _ in 0..iters {
                        sink = sink
                            .wrapping_add(sample_token_checked(&logits, p, 0.4242).unwrap() as u64);
                    }
                    let host = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
                    let t = std::time::Instant::now();
                    for _ in 0..iters {
                        sink = sink
                            .wrapping_add(sample_token_exact(&logits, p, 0.4242).unwrap() as u64);
                    }
                    let served = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
                    println!(
                        "FASTCOST vocab={vocab} shape={shape:<11} case={label:<17} \
host_ms={host:.3} served_ms={served:.3} speedup={:.1}x fast_path={taken} sink={sink}",
                        host / served
                    );
                    let _ = fast;
                }
            }
        }
    }
}

#[cfg(test)]
mod detokenize_tests {
    use super::*;

    use crate::oapi::chat::NATIVE_WIRE_TOKENS;
    use crate::oapi::tool_parse::HERMES_WIRE_TOKENS;

    const FRAMING_SPECIALS: [&str; 5] = ["<bos>", "<eos>", "<pad>", "<|turn>", "<turn|>"];

    fn grammar_delimiters() -> Vec<&'static str> {
        NATIVE_WIRE_TOKENS
            .iter()
            .chain(HERMES_WIRE_TOKENS.iter())
            .copied()
            .collect()
    }

    fn specials() -> Vec<String> {
        strip_list_from_specials(
            FRAMING_SPECIALS
                .iter()
                .copied()
                .chain(grammar_delimiters())
                .map(|s| s.to_string()),
        )
    }

    #[test]
    fn tool_call_delimiters_are_never_stripped() {
        let keep = specials();
        for d in grammar_delimiters() {
            assert!(
                !keep.iter().any(|s| s == d),
                "{d} is a delimiter the tool parser scans for, but the wgpu path would \
                 strip it and hand the parser a body it cannot recognise"
            );
        }
        let raw = "<|turn>model\n<|tool_call>call:get_weather{<|\"|>city<|\"|>: <|\"|>Oslo<|\"|>}<tool_call|><turn|>";
        assert_eq!(
            strip_specials(raw.to_string(), &keep),
            "model\n<|tool_call>call:get_weather{<|\"|>city<|\"|>: <|\"|>Oslo<|\"|>}<tool_call|>"
        );
    }

    #[test]
    fn turn_and_sentinel_markers_still_go_away() {
        let keep = specials();
        for f in FRAMING_SPECIALS {
            assert!(
                keep.iter().any(|s| s == f),
                "{f} is framing, not protocol, and must be stripped from user-visible text"
            );
        }
        assert_eq!(
            keep.len(),
            FRAMING_SPECIALS.len(),
            "the wgpu strip list must be exactly the checkpoint's specials minus every \
             delimiter the tool parser scans for: {keep:?}"
        );
        assert_eq!(strip_specials("<bos>hi<eos>".to_string(), &keep), "hi");
        assert_eq!(
            strip_specials("plain answer".to_string(), &keep),
            "plain answer"
        );
    }

    #[test]
    fn json_tool_call_block_survives_detokenization() {
        let keep = specials();
        let raw = r#"<tool_call>{"name":"get","arguments":{"x":1}}</tool_call><turn|>"#;
        let visible = strip_specials(raw.to_string(), &keep);
        let parsed = crate::oapi::chat::parse_model_tool_calls(&visible, None);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "get");
    }

    #[test]
    fn longest_special_is_stripped_first() {
        let keep = strip_list_from_specials(["<a>", "<a><b>"].iter().map(|s| s.to_string()));
        assert_eq!(keep.first().map(String::as_str), Some("<a><b>"));
    }
}

#[cfg(test)]
mod batch_route_tests {
    use super::*;

    fn greedy() -> ChatGenerateRequest {
        ChatGenerateRequest {
            prompt: "hi".into(),
            max_new_tokens: 8,
            stop: Vec::new(),
            seed: None,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            guided: None,
            guided_think_close: None,
            logit_bias: Vec::new(),
            logprobs: false,
            top_logprobs: 0,
            kv_resume: None,
            kv_store: None,
            mm: None,
        }
    }

    #[test]
    fn batching_is_off_unless_the_knob_is_set() {
        assert!(
            std::env::var(batch::BATCH_ENV).is_err(),
            "{} is set in this test process, so the default-off claim cannot be checked here",
            batch::BATCH_ENV
        );
        assert!(!batch::BatchKnobs::from_env().enabled());
        assert_eq!(batch::BatchKnobs::from_env().max_batch, 1);
    }

    #[test]
    fn host_sampling_requests_are_kept_out_of_the_batch() {
        let kind = WgpuModelKind::Gemma4Dense;
        assert_eq!(
            batch_admits(kind, &greedy()),
            Ok(()),
            "gemma4-dense is the only kind whose graph has slots, so it is the only kind whose \
             admission can reach the sampling checks at all"
        );
        for mutate in [
            (|r: &mut ChatGenerateRequest| r.temperature = Some(0.7)) as fn(&mut _),
            |r: &mut ChatGenerateRequest| {
                r.temperature = Some(1.0);
                r.top_p = Some(0.9);
            },
            |r: &mut ChatGenerateRequest| r.repetition_penalty = Some(1.1),
            |r: &mut ChatGenerateRequest| r.logprobs = true,
            |r: &mut ChatGenerateRequest| r.logit_bias = vec![(3, 1.0)],
            |r: &mut ChatGenerateRequest| {
                r.guided = Some(nv_grammar::GrammarSpec::JsonSchema(
                    serde_json::json!({"type": "boolean"}),
                ))
            },
        ] {
            let mut r = greedy();
            mutate(&mut r);
            assert_eq!(
                batch_admits(kind, &r),
                Err(batch::Refusal::NeedsHostLogits),
                "a host-sampled request was admitted to the batch: the batch route decodes with \
                 in-shader argmax and never installs a grammar, so admitting one returns free \
                 prose under HTTP 200 to a caller who asked for a schema"
            );
        }
    }

    #[test]
    #[ignore = "loads a resident checkpoint and drives the GPU"]
    fn a_batched_stream_matches_the_same_prompt_served_alone() {
        let dir = PathBuf::from(
            std::env::var("NV_CHAT_MODEL_DIR")
                .expect("NV_CHAT_MODEL_DIR must point at a gemma4 dense checkpoint"),
        );
        let width = batch::batch_break_even(nv_models::gemma4_wgpu::MK_MAX).expect(
            "no batch width in 2..=MK_MAX pays under the measured step model, so the worker \
             would never take the batch route and this gate would pass by serving every prompt \
             single-stream",
        );
        std::env::set_var(batch::BATCH_ENV, width.to_string());
        let engine = WgpuChatEngine::load_with(&dir, 512, None).expect("load wgpu chat engine");
        assert_eq!(
            engine.kind(),
            WgpuModelKind::Gemma4Dense,
            "only the gemma4 dense graph has a batched decode path"
        );
        assert!(
            engine.batch_capacity() >= width,
            "the graph reported {} slots against a break-even width of {width}: a build-time \
             disabler fired and this test would pass by serving every prompt single-stream",
            engine.batch_capacity()
        );
        assert!(
            batch_route_gap(engine.kind()).is_none(),
            "the loaded engine batches, so the per-kind gap table must not be refusing it: {}",
            batch_route_refusal(engine.kind())
        );

        let all_prompts = [
            "Explain why the sky is blue.",
            "Name three rivers in Europe.",
            "What is the capital of Japan?",
            "List two uses for baking soda.",
            "Describe a cumulus cloud.",
            "Why do leaves change colour?",
        ];
        let prompts: Vec<&str> = all_prompts.into_iter().take(width).collect();
        assert!(
            prompts.len() == width && batch::batch_pays(width),
            "the fixture must hand the worker exactly the {width} streams the step model calls \
             break-even; {} prompts would be split back to single-stream",
            prompts.len()
        );
        let mk = |p: &str| {
            let mut r = greedy();
            r.prompt = p.to_string();
            r.max_new_tokens = 24;
            r
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        async fn collect(mut rx: mpsc::Receiver<ChatEvent>) -> String {
            let mut text = String::new();
            while let Some(ev) = rx.recv().await {
                match ev {
                    ChatEvent::TextDelta(s) => text.push_str(&s),
                    ChatEvent::Error(e) => panic!("engine error: {e}"),
                    ChatEvent::Done { .. } => break,
                    _ => {}
                }
            }
            text
        }

        let (batched, batched_s) = rt.block_on(async {
            let mut rxs = Vec::new();
            for p in &prompts {
                let (tx, rx) = mpsc::channel(256);
                engine.generate(mk(p), tx).await.unwrap();
                rxs.push(rx);
            }
            let t0 = std::time::Instant::now();
            let mut out = Vec::new();
            for rx in rxs {
                out.push(collect(rx).await);
            }
            (out, t0.elapsed().as_secs_f64())
        });

        let mut alone = Vec::new();
        let mut alone_s = Vec::new();
        for p in &prompts {
            let (text, secs) = rt.block_on(async {
                let (tx, rx) = mpsc::channel(256);
                engine.generate(mk(p), tx).await.unwrap();
                let t0 = std::time::Instant::now();
                let text = collect(rx).await;
                (text, t0.elapsed().as_secs_f64())
            });
            alone.push(text);
            alone_s.push(secs);
        }

        let tokens = |s: &str| {
            engine
                .inner
                .tokenizer
                .encode(s, false)
                .unwrap()
                .get_ids()
                .len()
        };
        eprintln!(
            "B={width} batched wall {batched_s:.3}s for {:?} visible tokens; alone {:?} for \
             {:?} -- aggregate and per-stream are separate questions",
            batched.iter().map(|t| tokens(t)).collect::<Vec<_>>(),
            alone_s.iter().map(|s| format!("{s:.3}s")).collect::<Vec<_>>(),
            alone.iter().map(|t| tokens(t)).collect::<Vec<_>>(),
        );
        assert_eq!(
            engine.batches_run(),
            1,
            "exactly one batch must have run: the {width} streams batched and the {width} \
             references did not"
        );
        for (slot, (b, a)) in batched.iter().zip(&alone).enumerate() {
            assert!(!b.is_empty(), "slot {slot} emitted nothing");
            assert_eq!(b, a, "slot {slot} diverged from its single-stream self");
        }

        async fn timed(mut rx: mpsc::Receiver<ChatEvent>, t0: std::time::Instant) -> (u32, f64) {
            let mut n = 0u32;
            while let Some(ev) = rx.recv().await {
                match ev {
                    ChatEvent::Done {
                        completion_tokens, ..
                    } => {
                        n = completion_tokens;
                        break;
                    }
                    ChatEvent::Error(e) => panic!("engine error: {e}"),
                    _ => {}
                }
            }
            (n, t0.elapsed().as_secs_f64())
        }
        let batched_arm = || {
            rt.block_on(async {
                let mut rxs = Vec::new();
                for p in &prompts {
                    let (tx, rx) = mpsc::channel(256);
                    engine.generate(mk(p), tx).await.unwrap();
                    rxs.push(rx);
                }
                let t0 = std::time::Instant::now();
                let mut n = 0u32;
                let mut worst = 0f64;
                for rx in rxs {
                    let (tok, secs) = timed(rx, t0).await;
                    n += tok;
                    worst = worst.max(secs);
                }
                (n, t0.elapsed().as_secs_f64(), worst)
            })
        };
        let serial_arm = || {
            rt.block_on(async {
                let t0 = std::time::Instant::now();
                let mut n = 0u32;
                let mut worst = 0f64;
                for p in &prompts {
                    let (tx, rx) = mpsc::channel(256);
                    engine.generate(mk(p), tx).await.unwrap();
                    let s = std::time::Instant::now();
                    let (tok, secs) = timed(rx, s).await;
                    n += tok;
                    worst = worst.max(secs / tok.max(1) as f64);
                }
                (n, t0.elapsed().as_secs_f64(), worst)
            })
        };
        batched_arm();
        serial_arm();
        eprintln!("round  arm         tokens  wall_s  aggregate_tok_s  worst_stream_ms_per_token");
        for r in 0..3 {
            for (name, (n, wall, worst)) in [
                ("batched", batched_arm()),
                ("serial ", serial_arm()),
                ("null-b ", batched_arm()),
            ] {
                let per_tok_ms = if name == "serial " {
                    worst * 1000.0
                } else {
                    worst * 1000.0 / (n as f64 / width as f64).max(1.0)
                };
                eprintln!(
                    "{r}      {name}     {n:>4}  {wall:>6.3}  {:>15.2}  {per_tok_ms:>25.1}",
                    n as f64 / wall
                );
            }
        }
    }

    #[test]
    fn a_graph_that_does_not_declare_a_capacity_gets_one_slot() {
        struct Undeclared;
        impl batch::BatchStepper for Undeclared {
            fn reset_batch(&mut self, _: usize) -> anyhow::Result<()> {
                anyhow::bail!("unreachable")
            }
            fn prefill_slot(&mut self, _: usize, _: &[u32]) -> anyhow::Result<u32> {
                anyhow::bail!("unreachable")
            }
            fn decode_step_batch(&mut self, _: &[u32]) -> anyhow::Result<Vec<u32>> {
                anyhow::bail!("unreachable")
            }
        }
        assert_eq!(batch::BatchStepper::batch_capacity(&Undeclared), 1);
    }

    #[test]
    fn qwen3_8_config_serves_through_the_qwen3_5_dense_wgpu_path() {
        let dense = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5"}"#;
        assert_eq!(
            classify_wgpu_model(dense).unwrap(),
            WgpuModelKind::Qwen3_5Dense,
            "qwen3.8-27B shares model_type qwen3_5 with qwen3.5-dense, so it must route to the \
             dense wgpu decoder (which dequantizes its F8_E4M3 DeltaNet/lm_head weights at load); \
             a qwen3.8-specific arm that breaks this silently unserves the checkpoint"
        );
        let flagship = r#"{"architectures":["Qwen3_5MoeForCausalLM"],"model_type":"qwen3_5_moe_text"}"#;
        assert_eq!(
            classify_wgpu_model(flagship).unwrap(),
            WgpuModelKind::Qwen3_5Moe,
            "the qwen3.8-2.4T-A95B flagship is model_type qwen3_5_moe_text and classifies as MoE"
        );
    }
}

#[cfg(test)]
mod incr_detok {
    use super::*;

    fn fresh_stream() -> VisibleStream {
        VisibleStream {
            incremental: true,
            full: String::new(),
            prefix_tokens: 0,
            read_tokens: 0,
        }
    }

    fn metaspace_tokenizer() -> tokenizers::Tokenizer {
        let json = serde_json::json!({
            "version": "1.0",
            "added_tokens": [
                {"id": 6, "content": "<|end|>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true}
            ],
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "▁Hello": 1, "▁world": 2, "▁split": 3, "ting": 4, "▁x": 5},
                "unk_token": "<unk>"
            },
            "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}
        });
        tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
    }

    fn bytelevel_tokenizer() -> tokenizers::Tokenizer {
        let json = serde_json::json!({
            "version": "1.0",
            "added_tokens": [],
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "a": 1, "ðŁ": 2, "ĻĤ": 3, "Ġok": 4},
                "unk_token": "<unk>"
            },
            "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}
        });
        tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
    }

    #[test]
    fn suffix_decode_alone_drops_the_metaspace_leading_space_so_the_stream_keeps_a_one_token_anchor()
    {
        let tok = metaspace_tokenizer();
        assert_eq!(tok.decode(&[2], false).unwrap(), "world");
        assert_eq!(tok.decode(&[1, 2], false).unwrap(), "Hello world");
    }

    #[test]
    fn incremental_visible_matches_full_decode_under_metaspace_and_special_strip() {
        let tok = metaspace_tokenizer();
        let strip = vec!["<|end|>".to_string()];
        let ids = [1u32, 2, 3, 4, 6, 5];
        let mut vs = fresh_stream();
        for n in 1..=ids.len() {
            let inc = vs
                .step_incremental(&tok, &strip, &ids[..n])
                .unwrap()
                .to_string();
            let full = strip_specials(tok.decode(&ids[..n], false).unwrap(), &strip);
            assert_eq!(inc, full, "prefix of {n} tokens");
        }
        assert_eq!(vs.full, "Hello world splitting x");
    }

    #[test]
    fn a_byte_token_ending_mid_utf8_char_is_held_until_the_codepoint_completes() {
        let tok = bytelevel_tokenizer();
        let strip: Vec<String> = Vec::new();
        let ids = [1u32, 2, 3, 4];
        let mut vs = fresh_stream();
        assert_eq!(vs.step_incremental(&tok, &strip, &ids[..1]).unwrap(), "a");
        assert_eq!(
            vs.step_incremental(&tok, &strip, &ids[..2]).unwrap(),
            "a",
            "the half-emoji byte token must be held, not streamed as U+FFFD"
        );
        assert_eq!(
            vs.step_incremental(&tok, &strip, &ids[..3]).unwrap(),
            "a\u{1f642}"
        );
        assert_eq!(
            vs.step_incremental(&tok, &strip, &ids[..4]).unwrap(),
            "a\u{1f642} ok"
        );
        assert_eq!(
            vs.full,
            tok.decode(&ids, false).unwrap(),
            "the accumulated stream must equal one full decode of the same ids"
        );
    }

    #[test]
    fn per_step_full_redecode_costs_more_than_the_incremental_window_on_a_real_tokenizer() {
        let Ok(dir) = std::env::var("NV_DETOK_BENCH_TOKENIZER_DIR") else {
            eprintln!(
                "skip: NV_DETOK_BENCH_TOKENIZER_DIR not set (point it at a model dir with \
                 tokenizer.json); this timing probe is meaningless in debug builds"
            );
            return;
        };
        let tok =
            tokenizers::Tokenizer::from_file(std::path::Path::new(&dir).join("tokenizer.json"))
                .expect("tokenizer.json");
        let text = "The sky appears blue because shorter wavelengths scatter far more strongly \
                    off air molecules, a dependence Rayleigh derived as inverse fourth power. "
            .repeat(64);
        let ids = tok
            .encode(text.as_str(), false)
            .expect("encode bench text")
            .get_ids()
            .to_vec();
        let n = ids.len();
        assert!(n >= 1024, "bench corpus produced only {n} tokens; want >= 1024");
        let strip = special_strip_list(&tok);

        let t0 = std::time::Instant::now();
        let mut vs = fresh_stream();
        let mut incr_final = String::new();
        for k in 1..=n {
            incr_final = vs
                .step_incremental(&tok, &strip, &ids[..k])
                .unwrap()
                .to_string();
        }
        let incr_s = t0.elapsed().as_secs_f64();

        let t1 = std::time::Instant::now();
        let mut full_final = String::new();
        let mut last_step_s = 0.0;
        for k in 1..=n {
            let t = std::time::Instant::now();
            full_final = strip_specials(tok.decode(&ids[..k], false).unwrap(), &strip);
            last_step_s = t.elapsed().as_secs_f64();
        }
        let full_s = t1.elapsed().as_secs_f64();

        assert_eq!(incr_final, full_final, "the two paths must emit the same text");
        eprintln!(
            "DETOK-BENCH tokens={n} strip_list={} full_redecode_ms_per_token_mean={:.4} \
             full_redecode_ms_at_final_token={:.4} incremental_ms_per_token={:.4} \
             basis=release_lib_test_single_thread tokenizer_dir={dir}",
            strip.len(),
            full_s * 1e3 / n as f64,
            last_step_s * 1e3,
            incr_s * 1e3 / n as f64,
        );
        assert!(
            incr_s < full_s,
            "at {n} tokens the O(n^2) full-redecode loop ({full_s:.3}s) must cost more than the \
             incremental window loop ({incr_s:.3}s)"
        );
    }
}
