use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context as _;

use super::build::{LoadedModel, NvEngineChat};
use crate::oapi::chat::{ChatEngine, ChatEvent, ChatGenerateRequest, ChatMessageIn};
use crate::oapi::lora;

pub fn cuda_adapter_tag_that_changes_when_the_weights_change(dir: &Path) -> String {
    let weights = dir.join(lora::ADAPTER_WEIGHTS_FILE);
    match std::fs::read(&weights) {
        Ok(bytes) => {
            let mut h = xxhash_rust::xxh3::Xxh3::new();
            h.update(&bytes);
            format!("{}@{}:{:016x}", dir.display(), bytes.len(), h.digest())
        }
        Err(_) => dir.display().to_string(),
    }
}

static CUDA_BASE_DIRS: std::sync::OnceLock<Vec<(String, PathBuf)>> = std::sync::OnceLock::new();

pub fn cuda_base_dirs() -> Vec<(String, PathBuf)> {
    CUDA_BASE_DIRS.get().cloned().unwrap_or_default()
}

pub fn select_cuda_base<'a>(
    declared: Option<&str>,
    bases: &'a [(String, PathBuf)],
) -> Result<&'a (String, PathBuf), String> {
    if bases.is_empty() {
        return Err("no base chat model is loaded on the cuda engine".into());
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
                "base_model_name_or_path {want:?} matches none of the loaded cuda models {:?}",
                bases.iter().map(|(id, _)| id).collect::<Vec<_>>()
            ));
        }
        tracing::warn!(
            declared = %want,
            attached = %bases[0].0,
            "lora adapter declares a base model that is not the one loaded; attaching anyway \
             because exactly one cuda base model is resident"
        );
    }
    Ok(&bases[0])
}

pub struct CudaLoraAdapterEngine {
    adapter_id: String,
    adapter_kv_tag: String,
    engine: NvEngineChat,
}

impl CudaLoraAdapterEngine {
    pub fn try_load(base_dir: &Path, adapter_id: &str, adapter_dir: &Path) -> anyhow::Result<Self> {
        let eng = NvEngineChat::try_load_for_adapter_serving(base_dir)
            .with_context(|| format!("load cuda base engine from {}", base_dir.display()))?;
        let adapter =
            nv_weights::lora_adapter::LoraAdapter::load(adapter_dir, &eng.inner.device)
                .with_context(|| format!("load lora adapter from {}", adapter_dir.display()))?;
        let install = match &eng.inner.model {
            LoadedModel::Gemma4(m) => nv_specdecode::lora_spec::install_gemma4_serving_adapter(
                m, &adapter,
            )
            .with_context(|| format!("install lora adapter {adapter_id:?} on cuda gemma4"))?,
            LoadedModel::Laguna(m) => nv_specdecode::lora_spec::install_laguna_serving_adapter(
                m, &adapter,
            )
            .with_context(|| format!("install lora adapter {adapter_id:?} on cuda laguna"))?,
            _ => anyhow::bail!(
                "try_load_for_adapter_serving admitted family {:?} for {}; its family gate \
                 must refuse everything outside the gemma4/laguna allowlist of hookable \
                 nv-layers Linear projections",
                eng.inner.family,
                base_dir.display()
            ),
        };
        for note in &install.skipped {
            tracing::warn!(adapter = %adapter_id, note = %note, "lora module skipped");
        }
        tracing::info!(
            adapter = %adapter_id,
            base = %eng.inner.model_id,
            matched_modules = install.matched_modules,
            hooked_linears = install.hooked_linears,
            "cuda lora adapter resident; lean build never constructed the batch-engine, \
             drafter, or laguna spec-serve arms because their forwards bypass nv-layers \
             Linear hooks"
        );
        Ok(Self {
            adapter_id: adapter_id.to_string(),
            adapter_kv_tag: cuda_adapter_tag_that_changes_when_the_weights_change(adapter_dir),
            engine: eng,
        })
    }

    pub fn adapter_kv_tag_that_any_future_cuda_kv_persist_must_key_on(&self) -> &str {
        &self.adapter_kv_tag
    }
}

#[async_trait::async_trait]
impl ChatEngine for CudaLoraAdapterEngine {
    fn model_id(&self) -> &str {
        &self.adapter_id
    }

    fn spec_decode_status(&self) -> Option<&'static str> {
        self.engine.spec_decode_status()
    }

    fn supports_mm_input(&self) -> bool {
        self.engine.supports_mm_input()
    }

    fn mm_markers(&self) -> (&'static str, &'static str) {
        self.engine.mm_markers()
    }

    fn render_prompt(&self, messages: &[ChatMessageIn]) -> String {
        self.engine.render_prompt(messages)
    }

    fn official_template(&self) -> Option<&crate::oapi::chat_template::ChatTemplate> {
        self.engine.official_template()
    }

    async fn generate(
        &self,
        req: ChatGenerateRequest,
        tx: tokio::sync::mpsc::Sender<ChatEvent>,
    ) -> anyhow::Result<()> {
        self.engine.generate(req, tx).await
    }
}

pub const QWEN4_EXP_LORA_ADAPTER_DIRS_ENV: &str = "NV_Q4EXP_LORA_ADAPTER_DIRS";

pub struct Qwen4ExpAdapterEngine {
    adapter_id: String,
    registry: Arc<nv_specdecode::qwen4_exp_adapters::Qwen4ExpAdapterRegistry>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    chat_template: Option<Arc<crate::oapi::chat_template::ChatTemplate>>,
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: Vec<u32>,
}

impl Qwen4ExpAdapterEngine {
    pub fn adapter_kv_tag_that_any_future_cuda_kv_persist_must_key_on(&self) -> Option<String> {
        self.registry.current(&self.adapter_id).map(|v| v.tag())
    }
}

struct Qwen4ExpEngineParts {
    tokenizer: Arc<tokenizers::Tokenizer>,
    chat_template: Option<Arc<crate::oapi::chat_template::ChatTemplate>>,
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: Vec<u32>,
}

static QWEN4_EXP_ENGINE_PARTS: std::sync::OnceLock<Qwen4ExpEngineParts> = std::sync::OnceLock::new();

pub fn qwen4_exp_adapter_engine(adapter_id: &str) -> Option<Arc<dyn ChatEngine>> {
    let registry = qwen4_exp_adapter_registry()?;
    registry.current(adapter_id)?;
    let p = QWEN4_EXP_ENGINE_PARTS.get()?;
    Some(Arc::new(Qwen4ExpAdapterEngine {
        adapter_id: adapter_id.to_string(),
        registry,
        tokenizer: p.tokenizer.clone(),
        chat_template: p.chat_template.clone(),
        kv_max_seq_len: p.kv_max_seq_len,
        default_max_new: p.default_max_new,
        eos_ids: p.eos_ids.clone(),
    }))
}

static QWEN4_EXP_ADAPTERS: std::sync::OnceLock<Arc<nv_specdecode::qwen4_exp_adapters::Qwen4ExpAdapterRegistry>> =
    std::sync::OnceLock::new();

pub fn qwen4_exp_adapter_registry() -> Option<Arc<nv_specdecode::qwen4_exp_adapters::Qwen4ExpAdapterRegistry>> {
    QWEN4_EXP_ADAPTERS.get().cloned()
}

fn registry_for(
    model: &Arc<nv_models::qwen4_exp::Qwen4Exp>,
) -> anyhow::Result<Arc<nv_specdecode::qwen4_exp_adapters::Qwen4ExpAdapterRegistry>> {
    if let Some(r) = QWEN4_EXP_ADAPTERS.get() {
        anyhow::ensure!(Arc::ptr_eq(r.model(), model), "a second resident qwen4_exp engine; runtime adapters serve the first");
        return Ok(r.clone());
    }
    let r = Arc::new(nv_specdecode::qwen4_exp_adapters::Qwen4ExpAdapterRegistry::new(
        model.clone(),
        nv_specdecode::qwen4_exp_adapters::Qwen4ExpAdapterRegistry::park_budget_from_env(),
        crate::oapi::cold_tier::chat_min_free_bytes(),
    )?);
    Ok(QWEN4_EXP_ADAPTERS.get_or_init(|| r).clone())
}

pub(crate) fn qwen4_exp_runtime_lora_engine(
    inner: &super::build::NvEngineInner,
    adapter_id: &str,
    adapter_dir: &Path,
) -> Option<anyhow::Result<Arc<dyn ChatEngine>>> {
    let LoadedModel::Qwen4Exp(model) = &inner.model else {
        return None;
    };
    Some((|| {
        let registry = registry_for(model)?;
        let _ = QWEN4_EXP_ENGINE_PARTS.set(Qwen4ExpEngineParts {
            tokenizer: inner.tokenizer.clone(),
            chat_template: inner.chat_template.clone(),
            kv_max_seq_len: inner.kv_max_seq_len,
            default_max_new: inner.default_max_new_tokens as usize,
            eos_ids: inner.eos_token_ids.clone(),
        });
        let v = registry
            .load(adapter_id, adapter_dir)
            .with_context(|| format!("load qwen4_exp runtime lora {adapter_id:?} from {}", adapter_dir.display()))?;
        tracing::info!(
            adapter = %adapter_id,
            version = %v.version,
            base = %inner.model_id,
            modules = v.serving.modules(),
            expert_lora = v.form.as_str(),
            expert_layers = v.serving.expert_layers(),
            device_bytes = v.device_bytes,
            "qwen4_exp runtime lora resident on the base engine's weights; requests to this id \
             attach its hooks under the engine permit and run non-speculative"
        );
        Ok(Arc::new(Qwen4ExpAdapterEngine {
            adapter_id: adapter_id.to_string(),
            registry,
            tokenizer: inner.tokenizer.clone(),
            chat_template: inner.chat_template.clone(),
            kv_max_seq_len: inner.kv_max_seq_len,
            default_max_new: inner.default_max_new_tokens as usize,
            eos_ids: inner.eos_token_ids.clone(),
        }) as Arc<dyn ChatEngine>)
    })())
}

pub fn hot_swap_qwen4_exp_adapter(id: &str, dir: &Path) -> Option<anyhow::Result<(String, bool)>> {
    let registry = qwen4_exp_adapter_registry()?;
    let existed = registry.current(id).is_some();
    Some(registry.load(id, dir).map(|v| (v.version.clone(), existed)))
}

pub fn unload_qwen4_exp_adapter(id: &str) -> bool {
    qwen4_exp_adapter_registry().and_then(|r| r.unload(id)).is_some()
}

pub fn qwen4_exp_adapter_metrics() -> String {
    let Some(r) = qwen4_exp_adapter_registry() else {
        return String::new();
    };
    let mut out = String::from(
        "# HELP speaches_q4exp_lora_adapter_device_bytes Device bytes of each live qwen4_exp runtime LoRA version.\n\
         # TYPE speaches_q4exp_lora_adapter_device_bytes gauge\n",
    );
    for row in r.rows() {
        out.push_str(&format!(
            "speaches_q4exp_lora_adapter_device_bytes{{adapter=\"{}\",version=\"{}\",expert_lora=\"{}\",rank=\"{}\"}} {}\n",
            row.id, row.version, row.form, row.rank, row.device_bytes
        ));
        out.push_str(&format!(
            "speaches_q4exp_lora_adapter_in_flight_requests{{adapter=\"{}\",version=\"{}\"}} {}\n",
            row.id, row.version, row.in_flight
        ));
    }
    out.push_str(&format!(
        "# HELP speaches_q4exp_lora_host_parked_bytes Pinned host bytes holding replaced or unloaded adapter versions for rollback.\n\
         # TYPE speaches_q4exp_lora_host_parked_bytes gauge\n\
         speaches_q4exp_lora_host_parked_bytes {}\n\
         speaches_pinned_host_bytes_live {}\n",
        r.parked_bytes(),
        nv_layers::pinned_host::pinned_host_bytes_live()
    ));
    out
}

#[async_trait::async_trait]
impl ChatEngine for Qwen4ExpAdapterEngine {
    fn model_id(&self) -> &str {
        &self.adapter_id
    }

    fn supports_mm_input(&self) -> bool {
        super::qwen4_exp_mm::vision_for(self.registry.model()).is_some()
    }

    fn supports_audio_input(&self) -> bool {
        false
    }

    fn render_prompt(&self, messages: &[ChatMessageIn]) -> String {
        super::render_qwen3_5_moe_prompt(messages)
    }

    fn official_template(&self) -> Option<&crate::oapi::chat_template::ChatTemplate> {
        self.chat_template.as_deref()
    }

    async fn generate(
        &self,
        req: ChatGenerateRequest,
        tx: tokio::sync::mpsc::Sender<ChatEvent>,
    ) -> anyhow::Result<()> {
        let version = self
            .registry
            .current(&self.adapter_id)
            .ok_or_else(|| anyhow::anyhow!("adapter {:?} was unloaded", self.adapter_id))?;
        let (model, tokenizer) = (self.registry.model().clone(), self.tokenizer.clone());
        let (kv, max_new, eos) = (self.kv_max_seq_len, self.default_max_new, self.eos_ids.clone());
        tokio::spawn(async move {
            if let Err(err) =
                super::qwen4_exp_loop::run_sampling_qwen4_exp(model, tokenizer, req, kv, max_new, &eos, Some(version), &tx).await
            {
                let _ = tx.send(ChatEvent::Error(format!("{err:#}"))).await;
            }
        });
        Ok(())
    }
}

fn attach_qwen4_exp_runtime_adapters(engines: &mut Vec<Arc<dyn ChatEngine>>) {
    let Some(raw) = std::env::var_os(QWEN4_EXP_LORA_ADAPTER_DIRS_ENV) else {
        return;
    };
    for (name, dir) in lora::split_specs(&raw.to_string_lossy()) {
        let entry = match lora::probe_adapter(&dir, name.as_deref()) {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(dir = %dir.display(), error = %format!("{err:#}"), "qwen4_exp runtime lora skipped");
                continue;
            }
        };
        if engines.iter().any(|e| e.model_id() == entry.id) {
            tracing::warn!(adapter = %entry.id, "qwen4_exp runtime lora id collides with a loaded model id; skipped");
            continue;
        }
        let declared = entry.base_model.as_deref().map(|b| b.rsplit('/').next().unwrap_or(b).to_string());
        let mut built = None;
        for eng in engines.iter() {
            let leaf = eng.model_id().rsplit('/').next().unwrap_or(eng.model_id()).to_string();
            if declared.as_ref().is_some_and(|d| *d != leaf) {
                continue;
            }
            if let Some(res) = eng.runtime_lora_twin_sharing_these_resident_weights(&entry.id, &entry.dir) {
                built = Some(res);
                break;
            }
        }
        match built {
            Some(Ok(eng)) => {
                engines.push(eng);
                lora::register_adapter(entry);
            }
            Some(Err(err)) if format!("{err:#}").contains("refuses to fall back to pageable memory") => {
                tracing::error!(adapter = %entry.id, error = %format!("{err:#}"), "pinned host memory for the qwen4_exp adapter tier is unavailable; refusing to start rather than serve the boot adapter set from pageable memory");
                std::process::exit(1);
            }
            Some(Err(err)) => tracing::warn!(adapter = %entry.id, error = %format!("{err:#}"), "qwen4_exp runtime lora failed to load; the server continues without it"),
            None => tracing::warn!(adapter = %entry.id, declared = ?entry.base_model, "no resident qwen4_exp engine matches this adapter's base; skipped"),
        }
    }
}

pub(crate) fn attach_boot_adapters(
    engines: &mut Vec<Arc<dyn ChatEngine>>,
    bases: &[(String, PathBuf)],
) {
    let _ = CUDA_BASE_DIRS.set(bases.to_vec());
    attach_qwen4_exp_runtime_adapters(engines);
    let configured = std::env::var_os(lora::ADAPTER_DIRS_ENV).is_some();
    if !configured {
        return;
    }
    let mut cat = lora::catalog_from_env();
    let base_ids: Vec<String> = bases.iter().map(|(id, _)| id.clone()).collect();
    cat.drop_ids(&base_ids, "adapter id collides with a loaded base model id");
    for entry in cat.entries() {
        let base = match select_cuda_base(entry.base_model.as_deref(), bases) {
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
        match CudaLoraAdapterEngine::try_load(&base.1, &entry.id, &entry.dir) {
            Ok(eng) => {
                tracing::info!(
                    adapter = %entry.id,
                    base = %base.0,
                    dir = %entry.dir.display(),
                    "lora adapter registered as a cuda chat model id"
                );
                engines.push(Arc::new(eng));
                lora::register_adapter(entry.clone());
            }
            Err(err) => tracing::warn!(
                adapter = %entry.id,
                dir = %entry.dir.display(),
                error = %format!("{err:#}"),
                "lora adapter failed to load on cuda; the server continues without it"
            ),
        }
    }
    lora::warn_if_typo_can_be_swallowed(engines.len(), configured);
}
