use std::path::{Path, PathBuf};

use nv_weights::lora_adapter::{PeftConfig, TargetModules};

pub const ADAPTER_DIRS_ENV: &str = "NV_LORA_ADAPTER_DIRS";
pub const ADAPTER_CONFIG_FILE: &str = "adapter_config.json";
pub const ADAPTER_WEIGHTS_FILE: &str = "adapter_model.safetensors";

#[derive(Clone, Debug)]
pub struct AdapterEntry {
    pub id: String,
    pub dir: PathBuf,
    pub base_model: Option<String>,
    pub rank: usize,
    pub alpha: f64,
    pub scaling: f64,
    pub targets: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct SkippedAdapter {
    pub dir: PathBuf,
    pub reason: String,
}

#[derive(Clone, Debug, Default)]
pub struct AdapterCatalog {
    entries: Vec<AdapterEntry>,
    skipped: Vec<SkippedAdapter>,
}

impl AdapterCatalog {
    pub fn from_entries(entries: Vec<AdapterEntry>) -> Self {
        Self {
            entries,
            skipped: Vec::new(),
        }
    }

    pub fn entries(&self) -> &[AdapterEntry] {
        &self.entries
    }

    pub fn skipped(&self) -> &[SkippedAdapter] {
        &self.skipped
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, id: &str) -> Option<&AdapterEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn ids(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.id.clone()).collect()
    }

    fn skip(&mut self, dir: &Path, reason: impl Into<String>) {
        let reason = reason.into();
        tracing::warn!(
            dir = %dir.display(),
            reason = %reason,
            "lora adapter skipped; the server continues without it"
        );
        self.skipped.push(SkippedAdapter {
            dir: dir.to_path_buf(),
            reason,
        });
    }

    pub fn drop_ids(&mut self, ids: &[String], reason: &str) {
        let mut dropped: Vec<AdapterEntry> = Vec::new();
        self.entries.retain(|e| {
            if ids.iter().any(|b| b == &e.id) {
                dropped.push(e.clone());
                false
            } else {
                true
            }
        });
        for e in dropped {
            self.skip(&e.dir, reason);
        }
    }
}

pub fn validate_id(id: &str) -> anyhow::Result<()> {
    anyhow::ensure!(!id.is_empty(), "adapter id is empty");
    anyhow::ensure!(
        !id.chars().any(char::is_whitespace),
        "adapter id {id:?} contains whitespace"
    );
    anyhow::ensure!(
        id != "." && id != "..",
        "adapter id {id:?} is not a usable name"
    );
    Ok(())
}

pub fn split_specs(raw: &str) -> Vec<(Option<String>, PathBuf)> {
    raw.split([',', ':'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| match s.split_once('=') {
            Some((name, path)) => (Some(name.trim().to_string()), PathBuf::from(path.trim())),
            None => (None, PathBuf::from(s)),
        })
        .collect()
}

pub fn looks_like_adapter_dir(dir: &Path) -> bool {
    dir.join(ADAPTER_CONFIG_FILE).is_file()
}

pub fn probe_adapter(dir: &Path, id: Option<&str>) -> anyhow::Result<AdapterEntry> {
    anyhow::ensure!(dir.is_dir(), "{} is not a directory", dir.display());
    let cfg_path = dir.join(ADAPTER_CONFIG_FILE);
    anyhow::ensure!(cfg_path.is_file(), "no {ADAPTER_CONFIG_FILE}");
    let raw = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", cfg_path.display()))?;
    let cfg = PeftConfig::from_json_str(&raw)?;
    let weights = dir.join(ADAPTER_WEIGHTS_FILE);
    anyhow::ensure!(
        weights.is_file(),
        "no {ADAPTER_WEIGHTS_FILE} (the loader reads exactly that name)"
    );
    let id = match id {
        Some(s) => s.to_string(),
        None => dir
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("directory has no usable name; use name=path"))?
            .to_string(),
    };
    validate_id(&id)?;
    let base_model = serde_json::from_str::<serde_json::Value>(&raw)
        .ok()
        .and_then(|v| {
            v.get("base_model_name_or_path")
                .and_then(|b| b.as_str())
                .map(str::to_string)
        })
        .filter(|s| !s.is_empty());
    let targets = match &cfg.target_modules {
        TargetModules::List(v) => v.clone(),
        TargetModules::Pattern(p) => vec![p.clone()],
    };
    Ok(AdapterEntry {
        id,
        dir: dir.to_path_buf(),
        base_model,
        rank: cfg.r,
        alpha: cfg.lora_alpha,
        scaling: cfg.scaling,
        targets,
    })
}

pub fn catalog_from_specs(specs: &[(Option<String>, PathBuf)]) -> AdapterCatalog {
    let mut cat = AdapterCatalog::default();
    for (name, root) in specs {
        if !root.is_dir() {
            cat.skip(root, "not a directory");
            continue;
        }
        if looks_like_adapter_dir(root) {
            push_probe(&mut cat, root, name.as_deref());
            continue;
        }
        if name.is_some() {
            cat.skip(root, format!("named entry has no {ADAPTER_CONFIG_FILE}"));
            continue;
        }
        let mut children: Vec<PathBuf> = match std::fs::read_dir(root) {
            Ok(rd) => rd
                .flatten()
                .map(|e| e.path())
                .filter(|p| looks_like_adapter_dir(p))
                .collect(),
            Err(e) => {
                cat.skip(root, format!("read_dir failed: {e}"));
                continue;
            }
        };
        children.sort();
        if children.is_empty() {
            cat.skip(
                root,
                format!("no {ADAPTER_CONFIG_FILE} here or in any child"),
            );
            continue;
        }
        for child in children {
            push_probe(&mut cat, &child, None);
        }
    }
    cat
}

fn push_probe(cat: &mut AdapterCatalog, dir: &Path, name: Option<&str>) {
    match probe_adapter(dir, name) {
        Ok(entry) => {
            if cat.entries.iter().any(|e| e.id == entry.id) {
                cat.skip(dir, format!("duplicate adapter id {:?}", entry.id));
                return;
            }
            tracing::info!(
                adapter = %entry.id,
                dir = %entry.dir.display(),
                rank = entry.rank,
                alpha = entry.alpha,
                scaling = entry.scaling,
                base_model = entry.base_model.as_deref().unwrap_or("<unspecified>"),
                targets = entry.targets.len(),
                "lora adapter discovered"
            );
            cat.entries.push(entry);
        }
        Err(err) => cat.skip(dir, format!("{err:#}")),
    }
}

pub fn catalog_from_env() -> AdapterCatalog {
    let Some(raw) = std::env::var_os(ADAPTER_DIRS_ENV) else {
        return AdapterCatalog::default();
    };
    let raw = raw.to_string_lossy().into_owned();
    let specs = split_specs(&raw);
    if specs.is_empty() {
        tracing::warn!("{ADAPTER_DIRS_ENV} is set but contains no paths");
        return AdapterCatalog::default();
    }
    let cat = catalog_from_specs(&specs);
    tracing::info!(
        loaded = cat.entries().len(),
        skipped = cat.skipped().len(),
        "lora adapter discovery complete"
    );
    cat
}

static SERVED: std::sync::OnceLock<AdapterCatalog> = std::sync::OnceLock::new();

pub fn publish_served(cat: AdapterCatalog) {
    let _ = SERVED.set(cat);
}

pub fn served() -> &'static AdapterCatalog {
    SERVED.get_or_init(AdapterCatalog::default)
}

pub use crate::oapi::chat_engine::{allow_unknown_model, ALLOW_UNKNOWN_MODEL_ENV};

pub fn typo_swallow_risk(
    engine_count: usize,
    adapters_configured: bool,
    allow_unknown: bool,
) -> bool {
    allow_unknown && adapters_configured && engine_count <= 1
}

pub fn warn_if_typo_can_be_swallowed(engine_count: usize, adapters_configured: bool) {
    if typo_swallow_risk(engine_count, adapters_configured, allow_unknown_model()) {
        tracing::warn!(
            "{ADAPTER_DIRS_ENV} is configured but no adapter engine was registered, and \
             {ALLOW_UNKNOWN_MODEL_ENV} is set. With a single engine loaded, \
             ChatRegistry::resolve serves an unknown model id from the base model instead of \
             returning 404 model_not_found, so a typo'd adapter name silently produces \
             base-model output. Unset {ALLOW_UNKNOWN_MODEL_ENV} while serving LoRA adapters."
        );
    }
}

static REGISTERED: std::sync::OnceLock<std::sync::Mutex<AdapterCatalog>> =
    std::sync::OnceLock::new();

fn registered_lock() -> &'static std::sync::Mutex<AdapterCatalog> {
    REGISTERED.get_or_init(|| std::sync::Mutex::new(AdapterCatalog::default()))
}

pub fn register_adapter(entry: AdapterEntry) {
    let mut cat = registered_lock().lock().unwrap();
    cat.entries.retain(|e| e.id != entry.id);
    tracing::info!(
        adapter = %entry.id,
        dir = %entry.dir.display(),
        rank = entry.rank,
        "fine-tuned lora adapter registered as a servable model id"
    );
    cat.entries.push(entry);
}

pub fn registered() -> AdapterCatalog {
    registered_lock().lock().unwrap().clone()
}

pub fn all_entries() -> Vec<AdapterEntry> {
    let mut out: Vec<AdapterEntry> = served().entries().to_vec();
    for e in registered().entries() {
        out.retain(|x| x.id != e.id);
        out.push(e.clone());
    }
    out
}

pub fn model_rows() -> Vec<crate::oapi::Model> {
    all_entries()
        .iter()
        .map(|entry| {
            let owner = entry
                .base_model
                .as_deref()
                .and_then(|b| b.split('/').next())
                .filter(|s| !s.is_empty())
                .unwrap_or("speaches-plus")
                .to_string();
            let mut extras = serde_json::Map::new();
            if let Some(base) = &entry.base_model {
                extras.insert("parent".into(), serde_json::Value::String(base.clone()));
            }
            extras.insert(
                "lora".into(),
                serde_json::json!({
                    "rank": entry.rank,
                    "alpha": entry.alpha,
                    "scaling": entry.scaling,
                    "target_modules": entry.targets,
                    "path": entry.dir.display().to_string(),
                }),
            );
            crate::oapi::Model {
                id: entry.id.clone(),
                created: 1,
                owned_by: owner,
                languages: None,
                task: "chat".into(),
                max_model_len: None,
                extras,
            }
        })
        .collect()
}

pub fn annotate_models(models: &mut [crate::oapi::Model], cat: &AdapterCatalog) {
    for m in models.iter_mut() {
        let Some(entry) = cat.get(&m.id) else {
            continue;
        };
        if let Some(base) = &entry.base_model {
            m.extras
                .insert("parent".into(), serde_json::Value::String(base.clone()));
        }
        m.extras.insert(
            "lora".into(),
            serde_json::json!({
                "rank": entry.rank,
                "alpha": entry.alpha,
                "scaling": entry.scaling,
                "target_modules": entry.targets,
                "path": entry.dir.display().to_string(),
            }),
        );
    }
}

#[cfg(test)]
mod unknown_model_env_tests {
    use super::*;
    use crate::oapi::chat::ChatEngine;
    use crate::oapi::chat_engine::{ChatRegistry, EchoEngine};
    use std::sync::Arc;

    const RAW_VALUES: [&str; 8] = ["1", "true", " TRUE ", "on", "yes", "0", "", "maybe"];

    const BASE_ID: &str = "base";
    const TYPO_ID: &str = "base-lora-adaptor";

    struct EnvGuard(Option<String>);

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.0.take() {
                Some(v) => std::env::set_var(ALLOW_UNKNOWN_MODEL_ENV, v),
                None => std::env::remove_var(ALLOW_UNKNOWN_MODEL_ENV),
            }
        }
    }

    #[test]
    fn the_lora_warning_predicate_tracks_the_registry_it_warns_about() {
        let _guard = EnvGuard(std::env::var(ALLOW_UNKNOWN_MODEL_ENV).ok());
        let one = ChatRegistry::single(Arc::new(EchoEngine::new(BASE_ID, "x")));
        let two = ChatRegistry::from_engines(vec![
            Arc::new(EchoEngine::new(BASE_ID, "x")) as Arc<dyn ChatEngine>,
            Arc::new(EchoEngine::new("other", "y")),
        ])
        .expect("two engines");
        for raw in RAW_VALUES {
            std::env::set_var(ALLOW_UNKNOWN_MODEL_ENV, raw);
            for (reg, engine_count) in [(&one, 1usize), (&two, 2usize)] {
                let swallowed = reg
                    .resolve(Some(TYPO_ID))
                    .map(|e| e.model_id() == BASE_ID)
                    .unwrap_or(false);
                assert_eq!(
                    typo_swallow_risk(engine_count, true, allow_unknown_model()),
                    swallowed,
                    "with {ALLOW_UNKNOWN_MODEL_ENV}={raw:?} and {engine_count} engine(s) \
                     loaded, ChatRegistry::resolve {} {TYPO_ID} from {BASE_ID} while \
                     warn_if_typo_can_be_swallowed concluded the opposite. The warning \
                     exists only to predict that fallback, so the two must read one env \
                     var name and one truthiness rule; split them and renaming either \
                     half leaves the operator silently unwarned",
                    if swallowed { "serves" } else { "refuses" }
                );
            }
        }
    }
}
