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
    pub fn try_load(
        base_dir: &Path,
        adapter_id: &str,
        adapter_dir: &Path,
    ) -> anyhow::Result<Self> {
        let eng = NvEngineChat::try_load_for_adapter_serving(base_dir)
            .with_context(|| format!("load cuda base engine from {}", base_dir.display()))?;
        let adapter = nv_weights::lora_adapter::LoraAdapter::load(adapter_dir, &eng.inner.device)
            .with_context(|| format!("load lora adapter from {}", adapter_dir.display()))?;
        let install = match &eng.inner.model {
            LoadedModel::Gemma4(m) => {
                nv_specdecode::lora_spec::install_gemma4_serving_adapter(m, &adapter)
                    .with_context(|| format!("install lora adapter {adapter_id:?} on cuda gemma4"))?
            }
            LoadedModel::Laguna(m) => {
                nv_specdecode::lora_spec::install_laguna_serving_adapter(m, &adapter)
                    .with_context(|| format!("install lora adapter {adapter_id:?} on cuda laguna"))?
            }
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

pub(crate) fn attach_boot_adapters(
    engines: &mut Vec<Arc<dyn ChatEngine>>,
    bases: &[(String, PathBuf)],
) {
    let _ = CUDA_BASE_DIRS.set(bases.to_vec());
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
