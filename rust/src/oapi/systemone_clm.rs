use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, ensure, Context, Result};
use candle_core::{Device, Tensor};
use nv_laya::json::Json;
use nv_laya::service::{Failure, UNAVAILABLE, UNPROCESSABLE};
use serde::Deserialize;

use super::cold_tier::ColdSlot;
use super::text_embeddings::backend::QwenEmbedder;

pub const ENV_MODEL_DIRS: &str = "NV_CLM_MODEL_DIRS";
const MAX_CACHED_ACTIONS: usize = 8192;

#[derive(Clone, Deserialize)]
pub struct Config {
    width: usize,
    depth: usize,
    projection_dim: usize,
    #[serde(default = "default_hidden")]
    hidden_size: usize,
    #[serde(default = "default_activation")]
    activation: String,
    #[serde(default)]
    layernorm: bool,
    #[serde(default)]
    residual: bool,
}

fn default_hidden() -> usize {
    4096
}
fn default_activation() -> String {
    "gelu".into()
}

#[derive(Deserialize)]
pub struct Meta {
    pub name: String,
    encoder_dir: PathBuf,
    pub max_tokens: usize,
    temperature: f64,
    logit_scale: f64,
    cfg: Config,
    format: usize,
}

struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    fn load(
        weights: &HashMap<String, Tensor>,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        let weight = weights
            .get(&format!("{name}.weight"))
            .context("missing CLM linear weight")?
            .clone();
        let bias = weights
            .get(&format!("{name}.bias"))
            .context("missing CLM linear bias")?
            .clone();
        ensure!(
            weight.dims() == [output, input] && bias.dims() == [output],
            "CLM linear shape mismatch at {name}"
        );
        Ok(Self { weight, bias })
    }
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        input.matmul(&self.weight.t()?)?.broadcast_add(&self.bias)
    }
}

pub struct Head {
    cfg: Config,
    input: Linear,
    hidden: Vec<Linear>,
    norms: Vec<(Tensor, Tensor)>,
    output: Linear,
}

impl Head {
    pub fn load(weights: &HashMap<String, Tensor>, prefix: &str, cfg: &Config) -> Result<Self> {
        ensure!(
            cfg.depth >= 2 && cfg.width > 0 && cfg.projection_dim > 0,
            "invalid CLM head dimensions"
        );
        ensure!(
            ["gelu", "relu", "silu"].contains(&cfg.activation.as_str()),
            "unsupported CLM activation"
        );
        let input = Linear::load(
            weights,
            &format!("{prefix}.inp"),
            cfg.hidden_size,
            cfg.width,
        )?;
        let output = Linear::load(
            weights,
            &format!("{prefix}.out"),
            cfg.width,
            cfg.projection_dim,
        )?;
        let mut hidden = Vec::new();
        let mut norms = Vec::new();
        for i in 0..cfg.depth - 2 {
            hidden.push(Linear::load(
                weights,
                &format!("{prefix}.hidden.{i}"),
                cfg.width,
                cfg.width,
            )?);
            if cfg.layernorm {
                let get = |suffix| {
                    weights
                        .get(&format!("{prefix}.norms.{i}.{suffix}"))
                        .cloned()
                        .ok_or_else(|| anyhow!("missing CLM layer norm {i}.{suffix}"))
                };
                let weight = get("weight")?;
                let bias = get("bias")?;
                ensure!(
                    weight.dims() == [cfg.width] && bias.dims() == [cfg.width],
                    "CLM norm shape mismatch"
                );
                norms.push((weight, bias));
            }
        }
        Ok(Self {
            cfg: cfg.clone(),
            input,
            hidden,
            norms,
            output,
        })
    }
    fn activate(&self, input: Tensor) -> candle_core::Result<Tensor> {
        match self.cfg.activation.as_str() {
            "gelu" => input.gelu_erf(),
            "relu" => input.relu(),
            _ => candle_nn::ops::silu(&input),
        }
    }
    pub fn project(&self, vector: &[f32]) -> Result<Vec<f32>> {
        let mut x = self.activate(self.input.forward(&Tensor::from_slice(
            vector,
            (1, vector.len()),
            &Device::Cpu,
        )?)?)?;
        for (i, linear) in self.hidden.iter().enumerate() {
            let mut h = linear.forward(&x)?;
            if self.cfg.layernorm {
                h = candle_nn::ops::layer_norm(&h, &self.norms[i].0, &self.norms[i].1, 1e-5)?;
            }
            h = self.activate(h)?;
            x = if self.cfg.residual { (&x + h)? } else { h };
        }
        let mut result = self.output.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        let norm = result.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for v in &mut result {
            *v /= norm;
        }
        ensure!(
            result.iter().all(|v| v.is_finite()),
            "non-finite CLM projection"
        );
        Ok(result)
    }
}

fn render(value: &Json, indent: usize) -> String {
    let pad = " ".repeat(indent);
    match value {
        Json::Null => String::new(),
        Json::Str(s) => s.clone(),
        Json::Bool(b) => b.to_string(),
        Json::Int(i) => i.to_string(),
        Json::UInt(i) => i.to_string(),
        Json::Float(f) => super::systemone_kev::py_float(*f),
        Json::Obj(entries) => entries
            .iter()
            .map(|(k, v)| {
                if matches!(v, Json::Obj(xs) if !xs.is_empty())
                    || matches!(v, Json::Arr(xs) if !xs.is_empty())
                {
                    format!("{pad}{k}:\n{}", render(v, indent + 2))
                } else {
                    format!("{pad}{k}: {}", render(v, 0))
                }
            })
            .collect::<Vec<_>>()
            .join(if indent == 0 { "\n\n" } else { "\n" }),
        Json::Arr(entries) => entries
            .iter()
            .map(|v| {
                if matches!(v, Json::Obj(xs) if !xs.is_empty())
                    || matches!(v, Json::Arr(xs) if !xs.is_empty())
                {
                    format!("{pad}-\n{}", render(v, indent + 2))
                } else {
                    format!("{pad}- {}", render(v, 0))
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn text_or_key(key: &str, desc: Option<&Json>) -> String {
    match desc {
        None | Some(Json::Null) => key.to_string(),
        Some(Json::Str(s)) if s.is_empty() => key.to_string(),
        Some(value) => render(value, 0),
    }
}

struct Question {
    id: String,
    kind: String,
    state: String,
    keys: Vec<String>,
    actions: Vec<String>,
}

fn questions(request: &Json) -> Result<Vec<Question>> {
    let state = render(request.get("state").unwrap_or(&Json::Null), 0);
    let entries = request
        .get("questions")
        .and_then(Json::as_obj)
        .context("questions must be a nonempty object")?;
    ensure!(!entries.is_empty(), "questions must be nonempty");
    let mut result = Vec::new();
    for (id, q) in entries {
        let instructions = render(q.get("instructions").unwrap_or(&Json::Null), 0);
        let state = [state.trim(), instructions.trim()]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        let kind = q
            .get("type")
            .and_then(Json::as_str)
            .context("question type missing")?;
        let criteria = q.get("criteria");
        let (keys, actions): (Vec<String>, Vec<String>) = match kind {
            "choice" => {
                let options = criteria
                    .and_then(Json::as_obj)
                    .context("choice criteria must be an object")?;
                (
                    options.iter().map(|(k, _)| k.clone()).collect(),
                    options
                        .iter()
                        .map(|(k, v)| text_or_key(k, Some(v)))
                        .collect(),
                )
            }
            "score" => {
                let options = criteria
                    .and_then(Json::as_arr)
                    .context("score criteria must be an array")?;
                ensure!(options.len() >= 2, "score needs at least two levels");
                (
                    (0..options.len()).map(|i| i.to_string()).collect(),
                    options.iter().map(|v| render(v, 0)).collect(),
                )
            }
            "noul" => {
                ensure!(
                    criteria.is_none() || matches!(criteria, Some(Json::Null | Json::Obj(_))),
                    "noul criteria must be an object"
                );
                let keys = vec!["false".to_string(), "true".to_string()];
                let actions = keys
                    .iter()
                    .map(|k| {
                        let fallback = if instructions.trim().is_empty() {
                            k.clone()
                        } else if k == "true" {
                            format!("Yes. This is true: {}", instructions.trim())
                        } else {
                            format!("No. This is false: {}", instructions.trim())
                        };
                        format!(
                            "{k}: {}",
                            text_or_key(&fallback, criteria.and_then(|v| v.get(k)))
                        )
                    })
                    .collect();
                (keys, actions)
            }
            _ => anyhow::bail!("unsupported question type {kind}"),
        };
        ensure!(
            (1..=255).contains(&keys.len()),
            "question needs 1..255 choices"
        );
        result.push(Question {
            id: id.clone(),
            kind: kind.into(),
            state,
            keys,
            actions,
        });
    }
    Ok(result)
}

#[derive(Default)]
struct Cache {
    values: HashMap<String, Vec<f32>>,
    order: VecDeque<String>,
}

pub struct Clm {
    meta: Meta,
    encoder: QwenEmbedder,
    state: Head,
    action: Head,
    cache: Mutex<Cache>,
}

fn meta(dir: &Path) -> Result<Meta> {
    let m: Meta = serde_json::from_str(&std::fs::read_to_string(dir.join("clm.json"))?)?;
    ensure!(
        m.format == 1
            && m.temperature.is_finite()
            && m.temperature > 0.0
            && m.logit_scale.is_finite(),
        "invalid CLM metadata"
    );
    ensure!(
        !m.name.is_empty() && m.max_tokens > 1,
        "invalid CLM name or context limit"
    );
    Ok(m)
}

impl Clm {
    pub fn load(dir: &Path) -> Result<Self> {
        let mut meta = meta(dir)?;
        if meta.encoder_dir.is_relative() {
            meta.encoder_dir = dir.join(&meta.encoder_dir);
        }
        let encoder = QwenEmbedder::load_clm(&meta.encoder_dir, meta.max_tokens)?;
        ensure!(
            encoder.hidden_size() == meta.cfg.hidden_size,
            "CLM head and encoder dimensions differ"
        );
        let weights =
            candle_core::safetensors::load(dir.join("clm_head.safetensors"), &Device::Cpu)?;
        let state = Head::load(&weights, "state_head", &meta.cfg)?;
        let action = Head::load(&weights, "action_head", &meta.cfg)?;
        Ok(Self {
            meta,
            encoder,
            state,
            action,
            cache: Mutex::new(Cache::default()),
        })
    }
    fn action(&self, text: &str) -> Result<(Vec<f32>, usize)> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| anyhow!("CLM cache poisoned"))?;
        if let Some(v) = cache.values.get(text) {
            return Ok((v.clone(), 0));
        }
        let (raw, tokens) = self.encoder.embed_clm(text)?;
        let projected = self.action.project(&raw)?;
        while cache.values.len() >= MAX_CACHED_ACTIONS {
            if let Some(key) = cache.order.pop_front() {
                cache.values.remove(&key);
            }
        }
        cache.order.push_back(text.into());
        cache.values.insert(text.into(), projected.clone());
        Ok((projected, tokens))
    }
    pub fn answer(&self, request: &Json) -> Result<Json, Failure> {
        let invalid = |e: anyhow::Error| Failure {
            status: UNPROCESSABLE,
            message: e.to_string(),
        };
        let unavailable = |e: anyhow::Error| Failure {
            status: UNAVAILABLE,
            message: e.to_string(),
        };
        let started = Instant::now();
        let questions = questions(request).map_err(invalid)?;
        let temperature = match request.get("temperature") {
            None => 1.0,
            Some(v) => serde_json::from_str::<f64>(&v.to_string())
                .map_err(|_| invalid(anyhow!("temperature must be numeric")))?,
        };
        if !temperature.is_finite() || temperature <= 0.0 || temperature > 100.0 {
            return Err(invalid(anyhow!("temperature must be in (0, 100]")));
        }
        let scale = self.meta.logit_scale.exp().min(100.0) / self.meta.temperature / temperature;
        let mut answers = Vec::new();
        let mut tokens = 0;
        for q in questions {
            let (raw, count) = self.encoder.embed_clm(&q.state).map_err(unavailable)?;
            tokens += count;
            let state = self.state.project(&raw).map_err(unavailable)?;
            let mut logits = Vec::new();
            for action in &q.actions {
                let (v, count) = self.action(action).map_err(unavailable)?;
                tokens += count;
                logits.push(
                    scale
                        * state
                            .iter()
                            .zip(v)
                            .map(|(a, b)| (*a as f64) * (b as f64))
                            .sum::<f64>(),
                );
            }
            answers.push((q.id.clone(), answer_shape(&q, &softmax(&logits))));
        }
        Ok(Json::obj([
            ("model", Json::Str(self.meta.name.clone())),
            ("answers", Json::Obj(answers)),
            (
                "usage",
                Json::obj([
                    ("input_tokens", Json::UInt(tokens as u64)),
                    ("output_tokens", Json::UInt(0)),
                ]),
            ),
            (
                "latency_ms",
                Json::Float(started.elapsed().as_secs_f64() * 1000.0),
            ),
        ]))
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|v| v / sum).collect()
}

fn answer_shape(q: &Question, p: &[f64]) -> Json {
    if q.kind == "noul" {
        return Json::obj([("type", Json::from("noul")), ("noul", Json::Float(p[1]))]);
    }
    let pick = (1..p.len()).fold(0, |best, i| if p[i] > p[best] { i } else { best });
    let confidence = if p.len() == 1 {
        1.0
    } else {
        (p[pick] - 1.0 / p.len() as f64) / (1.0 - 1.0 / p.len() as f64)
    };
    let mut fields = vec![
        ("type".into(), Json::Str(q.kind.clone())),
        ("confidence".into(), Json::Float(confidence)),
        (
            "probabilities".into(),
            Json::Obj(
                q.keys
                    .iter()
                    .cloned()
                    .zip(p.iter().copied().map(Json::Float))
                    .collect(),
            ),
        ),
    ];
    if q.kind == "choice" {
        fields.push(("choice".into(), Json::Str(q.keys[pick].clone())));
    } else {
        fields.push((
            "score".into(),
            Json::Float(p.iter().enumerate().map(|(i, v)| i as f64 * v).sum()),
        ));
        fields.push((
            "legend".into(),
            Json::Obj(
                q.keys
                    .iter()
                    .cloned()
                    .zip(q.actions.iter().cloned().map(Json::Str))
                    .collect(),
            ),
        ));
    }
    Json::Obj(fields)
}

pub fn configured() -> Vec<(PathBuf, Meta)> {
    std::env::var(ENV_MODEL_DIRS)
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .filter_map(|d| {
            let path = PathBuf::from(d);
            match meta(&path) {
                Ok(m) if path.join("clm_head.safetensors").is_file() => Some((path, m)),
                Ok(_) => None,
                Err(e) => {
                    tracing::warn!(path = d, error = %e, "invalid CLM checkpoint");
                    None
                }
            }
        })
        .collect()
}

pub struct Models(Vec<(String, Arc<ColdSlot<Clm>>)>);
impl Models {
    pub fn slot(&self, name: &str) -> Option<Arc<ColdSlot<Clm>>> {
        self.0
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, slot)| slot.clone())
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(n, _)| n.as_str())
    }
}

pub fn from_env() -> Option<Arc<Models>> {
    let slots: Vec<_> = configured()
        .into_iter()
        .map(|(dir, meta)| {
            let slot_name: &'static str =
                Box::leak(format!("systemone-clm-{}", meta.name).into_boxed_str());
            let slot = ColdSlot::demand(
                slot_name,
                32 * 1024 * 1024 * 1024,
                Arc::new(move || match Clm::load(&dir) {
                    Ok(model) => Some(Arc::new(model)),
                    Err(e) => {
                        tracing::warn!(error = %e, "CLM load failed");
                        None
                    }
                }),
            );
            (meta.name, slot)
        })
        .collect();
    (!slots.is_empty()).then(|| Arc::new(Models(slots)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn question_schema_matches_upstream_action_text_and_nested_prose() {
        let request: Json = serde_json::from_str(r#"{"state":{"user":"hi","context":["one","two"]},"questions":{"tool":{"type":"choice","instructions":"Pick","criteria":{"none":"No tool","run":null}}}}"#).unwrap();
        let q = questions(&request).unwrap().remove(0);
        assert_eq!(q.state, "user: hi\n\ncontext:\n  - one\n  - two\n\nPick");
        assert_eq!(q.actions, ["No tool", "run"]);
        let answer = answer_shape(&q, &[0.25, 0.75]);
        assert_eq!(answer.get("choice").and_then(Json::as_str), Some("run"));
        assert_eq!(answer.get("confidence").unwrap().to_string(), "0.5");
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires NV_CLM_HEAD_TEST=1 and NV_CLM_HEAD_TEST_DIR containing a PyTorch oracle"]
    fn checkpoint_heads_match_the_pytorch_oracle() {
        assert_eq!(std::env::var("NV_CLM_HEAD_TEST").as_deref(), Ok("1"));
        let dir = std::env::var("NV_CLM_HEAD_TEST_DIR").expect("set NV_CLM_HEAD_TEST_DIR");
        let dir = Path::new(&dir);
        let m = meta(dir).unwrap();
        let weights =
            candle_core::safetensors::load(dir.join("clm_head.safetensors"), &Device::Cpu).unwrap();
        let oracle: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("head_oracle.json")).unwrap())
                .unwrap();
        for prefix in ["state", "action"] {
            let head = Head::load(&weights, &format!("{prefix}_head"), &m.cfg).unwrap();
            let inputs: Vec<Vec<f32>> = serde_json::from_value(oracle["input"].clone()).unwrap();
            let expected: Vec<Vec<f32>> = serde_json::from_value(oracle[prefix].clone()).unwrap();
            for (input, want) in inputs.iter().zip(expected) {
                let got = head.project(input).unwrap();
                let error = got
                    .iter()
                    .zip(want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(error < 2e-5, "projection differs from PyTorch: {error}");
            }
        }
    }
}
