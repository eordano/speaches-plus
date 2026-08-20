use crate::WeightLoader;
use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub enum TargetModules {
    List(Vec<String>),
    Pattern(String),
}

#[derive(Clone, Debug)]
pub struct PeftConfig {
    pub r: usize,
    pub lora_alpha: f64,
    pub target_modules: TargetModules,
    pub use_rslora: bool,
    pub scaling: f64,
}

impl PeftConfig {
    pub fn from_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s).context("parse adapter_config.json")?;
        Self::from_json_value(&v)
    }

    pub fn from_json_value(v: &serde_json::Value) -> Result<Self> {
        let obj = v
            .as_object()
            .ok_or_else(|| anyhow!("adapter_config.json must be a JSON object"))?;

        let missing: Vec<&str> = ["r", "lora_alpha", "target_modules"]
            .into_iter()
            .filter(|k| !obj.contains_key(*k))
            .collect();
        if !missing.is_empty() {
            bail!("Missing required configuration fields: {missing:?}");
        }

        let r_raw = obj["r"]
            .as_i64()
            .ok_or_else(|| anyhow!("`r` must be an integer"))?;
        if r_raw <= 0 {
            bail!("LoRA rank `r` must be a positive integer, got {r_raw}.");
        }
        let r = r_raw as usize;

        let lora_alpha = obj["lora_alpha"]
            .as_f64()
            .ok_or_else(|| anyhow!("`lora_alpha` must be a number"))?;

        let target_modules = match &obj["target_modules"] {
            serde_json::Value::String(s) => TargetModules::Pattern(s.clone()),
            serde_json::Value::Array(a) => TargetModules::List(
                a.iter()
                    .map(|x| {
                        x.as_str()
                            .map(String::from)
                            .ok_or_else(|| anyhow!("`target_modules` entries must be strings"))
                    })
                    .collect::<Result<Vec<String>>>()?,
            ),
            _ => bail!("`target_modules` must be a list of strings or a string"),
        };

        let use_rslora = obj
            .get("use_rslora")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let use_dora = obj
            .get("use_dora")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let bias = obj
            .get("bias")
            .and_then(|x| x.as_str())
            .unwrap_or("none")
            .to_string();
        let modules_to_save_nonempty = match obj.get("modules_to_save") {
            None | Some(serde_json::Value::Null) => false,
            Some(serde_json::Value::Array(a)) => !a.is_empty(),
            Some(_) => true,
        };

        let mut errors: Vec<&str> = Vec::new();
        if modules_to_save_nonempty {
            errors.push("vLLM only supports modules_to_save being None.");
        }
        if use_dora {
            errors.push("vLLM does not yet support DoRA.");
        }
        if bias != "none" {
            errors.push("Adapter bias is not supported.");
        }
        if !errors.is_empty() {
            bail!("{}", errors.join(" "));
        }

        let scaling = if use_rslora {
            lora_alpha / (r as f64).sqrt()
        } else {
            lora_alpha / r as f64
        };

        Ok(Self {
            r,
            lora_alpha,
            target_modules,
            use_rslora,
            scaling,
        })
    }

    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join("adapter_config.json");
        let raw =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        Self::from_json_str(&raw)
    }
}

pub fn parse_fine_tuned_lora_name(name: &str) -> Result<(String, bool, bool)> {
    let stripped = name.strip_prefix("base_model.model.").unwrap_or(name);
    let parts: Vec<&str> = stripped.split('.').collect();
    let n = parts.len();
    if n >= 2 && parts[n - 1] == "weight" && (parts[n - 2] == "lora_A" || parts[n - 2] == "lora_B")
    {
        return Ok((parts[..n - 2].join("."), parts[n - 2] == "lora_A", false));
    }
    if parts[n - 1] == "lora_embedding_A" || parts[n - 1] == "lora_embedding_B" {
        return Ok((
            parts[..n - 1].join("."),
            parts[n - 1] == "lora_embedding_A",
            true,
        ));
    }
    bail!("{name} is unsupported LoRA weight");
}

pub fn normalize_module_name(module: &str) -> String {
    if let Some(rest) = module.strip_prefix("model.layers.") {
        return format!("model.language_model.layers.{rest}");
    }
    if module == "model.embed_tokens" {
        return "model.language_model.embed_tokens".to_string();
    }
    if let Some(rest) = module.strip_prefix("model.embed_tokens.") {
        return format!("model.language_model.embed_tokens.{rest}");
    }
    module.to_string()
}

#[derive(Debug)]
pub struct LoraLayerWeights {
    pub module_name: String,
    pub rank: usize,
    pub lora_a: Tensor,
    pub lora_b: Tensor,
    pub scaling: f64,
    pub is_embedding: bool,
}

impl LoraLayerWeights {
    pub fn optimize(&mut self) -> Result<()> {
        if self.scaling == 1.0 {
            return Ok(());
        }
        self.lora_b = self
            .lora_b
            .affine(self.scaling, 0.0)
            .map_err(|e| anyhow!("fold scaling into lora_b for {}: {e}", self.module_name))?;
        self.scaling = 1.0;
        Ok(())
    }
}

#[derive(Debug)]
pub struct LoraAdapter {
    pub config: PeftConfig,
    pub loras: BTreeMap<String, LoraLayerWeights>,
}

struct PendingModule {
    a: Option<Tensor>,
    b: Option<Tensor>,
    is_embedding: bool,
}

impl LoraAdapter {
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let config = PeftConfig::from_dir(dir)?;
        let st_path = dir.join("adapter_model.safetensors");
        let loader = WeightLoader::open_file(&st_path, device)?;

        let mut pending: BTreeMap<String, PendingModule> = BTreeMap::new();
        for name in loader.names() {
            if name.contains("lora_magnitude_vector") {
                bail!(
                    "DoRA adapter detected (tensor `{name}`): DoRA (use_dora) is not supported \
                     by this loader. Retrain a plain LoRA adapter (use_dora=false)."
                );
            }
            let (module, is_a, is_embedding) = parse_fine_tuned_lora_name(&name)?;
            let module = normalize_module_name(&module);
            let tensor = loader.get(&name, DType::BF16)?;
            let entry = pending.entry(module.clone()).or_insert(PendingModule {
                a: None,
                b: None,
                is_embedding,
            });
            if entry.is_embedding != is_embedding {
                bail!("module {module} mixes linear and embedding LoRA tensors");
            }
            let slot = if is_a { &mut entry.a } else { &mut entry.b };
            if slot.is_some() {
                bail!("duplicate LoRA tensor {name}");
            }
            *slot = Some(tensor);
        }

        let mut loras = BTreeMap::new();
        for (module, p) in pending {
            let lora_a =
                p.a.ok_or_else(|| anyhow!("module {module} missing lora_A tensor"))?;
            let lora_b =
                p.b.ok_or_else(|| anyhow!("module {module} missing lora_B tensor"))?;
            let mut weights = LoraLayerWeights {
                module_name: module.clone(),
                rank: config.r,
                lora_a,
                lora_b,
                scaling: config.scaling,
                is_embedding: p.is_embedding,
            };
            weights.optimize()?;
            loras.insert(module, weights);
        }

        Ok(Self { config, loras })
    }
}
