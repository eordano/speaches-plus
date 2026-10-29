use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_laya::json::Json;
use nv_laya::service::{Failure, UNAVAILABLE, UNPROCESSABLE};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use serde::Deserialize;
use tracing::{info, warn};

use super::cold_tier::ColdSlot;

pub const ENV_MODEL_DIRS: &str = "NV_KEV_MODEL_DIRS";
pub const META_LEAF_WRITTEN_BY_SCRIPTS_KEV_EXPORT_PY: &str = "kev.json";
const HEAD_LEAF: &str = "kev_head.safetensors";
const MAX_OPTIONS_TYPESAFE_ALLOWS_PER_QUESTION: usize = 255;
const WEIGHTS_TIMES_THIS_COVERS_ROW_ACTIVATIONS_AND_THE_LOAD_TRANSIENT: u64 = 2;
const PROBABILITIES_ARE_ROUNDED_TO_FOUR_PLACES_SO_A_255_OPTION_SUM_STAYS_WITHIN_TYPESAFE_TOLERANCE: f64 =
    1e4;

#[derive(Deserialize)]
struct Special {
    state: u32,
    question: u32,
    option: u32,
    option_end: u32,
    decide: u32,
}

#[derive(Deserialize)]
struct Meta {
    name: String,
    temperature: f64,
    head_dim: usize,
    special: Special,
    max_state: usize,
    max_branch: usize,
}

pub struct Kev {
    meta: Meta,
    tokenizer: tokenizers::Tokenizer,
    backbone: Mutex<Qwen3Moe>,
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    device: Device,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Noul,
    Choice,
    Score,
}

struct Question {
    id: String,
    kind: Kind,
    instructions: String,
    options: Vec<String>,
    keys: Vec<String>,
}

fn invalid(message: String) -> Failure {
    Failure {
        status: UNPROCESSABLE,
        message,
    }
}

fn unavailable(e: anyhow::Error) -> Failure {
    Failure {
        status: UNAVAILABLE,
        message: format!("{e:#}"),
    }
}

pub(super) fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf" } else { "-inf" }.into();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    let sign = if mantissa.starts_with('-') { "-" } else { "" };
    let digits = mantissa.trim_start_matches('-').replace('.', "");
    if (-4..16).contains(&exp) {
        let body = if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                format!("{digits}{}.0", "0".repeat(int_len - digits.len()))
            } else {
                format!("{}.{}", &digits[..int_len], &digits[int_len..])
            }
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        };
        format!("{sign}{body}")
    } else {
        let m = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        format!(
            "{sign}{m}e{}{:02}",
            if exp < 0 { "-" } else { "+" },
            exp.abs()
        )
    }
}

fn render(v: &Json, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        Json::Null => String::new(),
        Json::Bool(b) => if *b { "True" } else { "False" }.into(),
        Json::Int(i) => i.to_string(),
        Json::UInt(u) => u.to_string(),
        Json::Float(f) => py_float(*f),
        Json::Str(s) => s.clone(),
        Json::Arr(items) => items
            .iter()
            .map(|x| format!("{pad}- {}", render(x, indent + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        Json::Obj(entries) => entries
            .iter()
            .map(|(k, x)| match x {
                Json::Obj(_) | Json::Arr(_) => format!("{pad}{k}:\n{}", render(x, indent + 1)),
                _ => format!("{pad}{k}: {}", render(x, 0)),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn option_text(name: &str, desc: Option<&Json>) -> String {
    match desc {
        None | Some(Json::Null) => name.to_string(),
        Some(Json::Str(s)) if s.is_empty() => name.to_string(),
        Some(d) => format!("{name}: {}", render(d, 0)),
    }
}

fn parse(request: &Json) -> Result<(String, Vec<Question>), Failure> {
    let bad = invalid;
    let state = render(request.get("state").unwrap_or(&Json::Null), 0);
    let questions = request
        .get("questions")
        .and_then(Json::as_obj)
        .filter(|q| !q.is_empty())
        .ok_or_else(|| bad("questions must be a non-empty object".into()))?;
    let mut out = Vec::with_capacity(questions.len());
    for (id, q) in questions {
        let kind = match q.get("type").and_then(Json::as_str) {
            Some("noul") => Kind::Noul,
            Some("choice") => Kind::Choice,
            Some("score") => Kind::Score,
            other => {
                return Err(bad(format!(
                    "question {id:?}: type must be noul, choice or score, got {other:?}"
                )))
            }
        };
        let instructions = render(q.get("instructions").unwrap_or(&Json::Null), 0);
        let criteria = q.get("criteria");
        let (options, keys) = match kind {
            Kind::Noul => {
                let c = match criteria {
                    None | Some(Json::Null) => &[][..],
                    Some(c) => c.as_obj().ok_or_else(|| {
                        bad(format!("question {id:?}: noul criteria must be an object"))
                    })?,
                };
                let get = |k: &str| c.iter().rev().find(|(n, _)| n == k).map(|(_, v)| v);
                (
                    vec![
                        option_text("no", get("false")),
                        option_text("yes", get("true")),
                    ],
                    vec!["false".into(), "true".into()],
                )
            }
            Kind::Choice => {
                let c = criteria
                    .and_then(Json::as_obj)
                    .filter(|c| (1..=MAX_OPTIONS_TYPESAFE_ALLOWS_PER_QUESTION).contains(&c.len()))
                    .ok_or_else(|| bad(format!("question {id:?}: choice criteria must be an object with 1..=255 options")))?;
                (
                    c.iter().map(|(k, v)| option_text(k, Some(v))).collect(),
                    c.iter().map(|(k, _)| k.clone()).collect(),
                )
            }
            Kind::Score => {
                let c = criteria
                    .and_then(Json::as_arr)
                    .filter(|c| (1..=MAX_OPTIONS_TYPESAFE_ALLOWS_PER_QUESTION).contains(&c.len()))
                    .ok_or_else(|| {
                        bad(format!(
                            "question {id:?}: score criteria must be an array of 1..=255 levels"
                        ))
                    })?;
                (
                    c.iter().map(|x| render(x, 0)).collect(),
                    (0..c.len()).map(|i| i.to_string()).collect(),
                )
            }
        };
        out.push(Question {
            id: id.clone(),
            kind,
            instructions,
            options,
            keys,
        });
    }
    Ok((state, out))
}

fn neutralise_delimiters(text: &str) -> std::borrow::Cow<'_, str> {
    static SPECIAL: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    SPECIAL
        .get_or_init(|| regex::Regex::new(r"<\|([A-Za-z0-9_]+)\|>").expect("static pattern"))
        .replace_all(text, "<¦${1}¦>")
}

impl Kev {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let meta: Meta = serde_json::from_str(
            &std::fs::read_to_string(dir.join(META_LEAF_WRITTEN_BY_SCRIPTS_KEV_EXPORT_PY))
                .with_context(|| {
                    format!(
                        "{} has no kev.json; export it with scripts/kev_export.py",
                        dir.display()
                    )
                })?,
        )?;
        let raw = std::fs::read_to_string(dir.join("config.json"))?;
        let config = Qwen3_5DenseConfig::from_hf_json_str(&raw)?;
        let weights = WeightLoader::open_dir(dir, device)?;
        let backbone = Qwen3Moe::from_loader_dense_quantized(
            config,
            &weights,
            &QuantizationConfig::from_hf_json_str(&raw)?,
            device,
        )?;
        drop(weights);
        let head = candle_core::safetensors::load(dir.join(HEAD_LEAF), device)?;
        let take = |k: &str| -> Result<Tensor> {
            Ok(head
                .get(k)
                .ok_or_else(|| anyhow!("{HEAD_LEAF} lacks {k}"))?
                .to_dtype(DType::F32)?)
        };
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        Ok(Self {
            q_w: take("q.weight")?.t()?.contiguous()?,
            q_b: take("q.bias")?,
            k_w: take("k.weight")?.t()?.contiguous()?,
            k_b: take("k.bias")?,
            meta,
            tokenizer,
            backbone: Mutex::new(backbone),
            device: device.clone(),
        })
    }

    pub fn name(&self) -> &str {
        &self.meta.name
    }

    fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(neutralise_delimiters(text).as_ref(), false)
            .map_err(|e| anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }

    fn row_probabilities(&self, row: &[u32], decide: usize, opts: &[usize]) -> Result<Vec<f64>> {
        let n = row.len();
        let tokens = Tensor::from_vec(row.to_vec(), (1usize, n), &self.device)?;
        let positions = Tensor::from_vec((0..n as i32).collect::<Vec<_>>(), n, &self.device)?;
        let hidden = {
            let backbone = self.backbone.lock().unwrap_or_else(|e| e.into_inner());
            backbone.forward_hidden(&tokens, &positions)?
        };
        let hidden = hidden.squeeze(0)?.to_dtype(DType::F32)?;
        let at = |idx: Vec<u32>| -> Result<Tensor> {
            let len = idx.len();
            Ok(hidden.index_select(&Tensor::from_vec(idx, len, &self.device)?, 0)?)
        };
        let q = at(vec![decide as u32])?
            .matmul(&self.q_w)?
            .broadcast_add(&self.q_b)?;
        let k = at(opts.iter().map(|&o| o as u32).collect())?
            .matmul(&self.k_w)?
            .broadcast_add(&self.k_b)?;
        let scale = 1.0 / (self.meta.head_dim as f64).sqrt() / self.meta.temperature;
        let z: Vec<f32> = k.matmul(&q.t()?)?.flatten_all()?.to_vec1()?;
        let z: Vec<f64> = z.into_iter().map(|v| v as f64 * scale).collect();
        let top = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = z.iter().map(|v| (v - top).exp()).collect();
        let total: f64 = e.iter().sum();
        Ok(e.into_iter().map(|v| v / total).collect())
    }

    pub fn answer(&self, request: &Json) -> Result<Json, Failure> {
        let started = Instant::now();
        let (state, questions) = parse(request)?;
        let sp = &self.meta.special;
        let mut prefix = vec![sp.state];
        prefix.extend(
            self.tokens(&state)
                .map_err(unavailable)?
                .into_iter()
                .take(self.meta.max_state - 1),
        );
        let mut input_tokens = prefix.len();
        let mut answers = Vec::with_capacity(questions.len());
        for q in &questions {
            let mut row = prefix.clone();
            row.push(sp.question);
            row.extend(self.tokens(&q.instructions).map_err(unavailable)?);
            let mut ends = Vec::with_capacity(q.options.len());
            for o in &q.options {
                row.push(sp.option);
                row.extend(self.tokens(o).map_err(unavailable)?);
                row.push(sp.option_end);
                ends.push(row.len() - 1);
            }
            row.push(sp.decide);
            let branch = row.len() - prefix.len();
            if branch > self.meta.max_branch - prefix.len() {
                return Err(invalid(format!(
                    "branch too long: {branch} tokens with a {}-token state (row limit {})",
                    prefix.len(),
                    self.meta.max_branch
                )));
            }
            input_tokens += branch;
            let p = self
                .row_probabilities(&row, row.len() - 1, &ends)
                .map_err(unavailable)?;
            answers.push((q.id.clone(), shape_answer(q, &p)));
        }
        let answers = Json::Obj(answers);
        let output_tokens = self.tokens(&answers.python()).map(|t| t.len()).unwrap_or(0);
        Ok(Json::obj([
            ("model", Json::Str(self.meta.name.clone())),
            ("answers", answers),
            (
                "usage",
                Json::obj([
                    ("input_tokens", Json::UInt(input_tokens as u64)),
                    ("output_tokens", Json::UInt(output_tokens as u64)),
                ]),
            ),
            (
                "latency_ms",
                Json::Float(round4(started.elapsed().as_secs_f64() * 1000.0)),
            ),
        ]))
    }
}

fn round4(x: f64) -> f64 {
    let s = PROBABILITIES_ARE_ROUNDED_TO_FOUR_PLACES_SO_A_255_OPTION_SUM_STAYS_WITHIN_TYPESAFE_TOLERANCE;
    (x * s).round() / s
}

fn shape_answer(q: &Question, p: &[f64]) -> Json {
    let argmax = (0..p.len()).fold(0, |b, i| if p[i] > p[b] { i } else { b });
    let dist = |keys: &[String]| {
        Json::Obj(
            keys.iter()
                .cloned()
                .zip(p.iter().map(|v| Json::Float(round4(*v))))
                .collect(),
        )
    };
    match q.kind {
        Kind::Noul => Json::obj([
            ("type", Json::from("noul")),
            ("noul", Json::Float(round4(p[1]))),
        ]),
        Kind::Choice => {
            let k = p.len() as f64;
            let confidence = if p.len() == 1 {
                1.0
            } else {
                (p[argmax] - 1.0 / k) / (1.0 - 1.0 / k)
            };
            Json::obj([
                ("type", Json::from("choice")),
                ("choice", Json::Str(q.keys[argmax].clone())),
                ("confidence", Json::Float(round4(confidence))),
                ("probabilities", dist(&q.keys)),
            ])
        }
        Kind::Score => {
            let levels = p.len();
            let score: f64 = p.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
            let spread: f64 = p
                .iter()
                .enumerate()
                .map(|(i, v)| v * (i as f64 - argmax as f64).abs())
                .sum();
            let confidence = if levels == 1 {
                1.0
            } else {
                1.0 - spread / (levels - 1) as f64
            };
            Json::obj([
                ("type", Json::from("score")),
                ("score", Json::Float(round4(score))),
                (
                    "legend",
                    Json::Obj(
                        q.keys
                            .iter()
                            .cloned()
                            .zip(q.options.iter().map(|o| Json::Str(o.clone())))
                            .collect(),
                    ),
                ),
                ("probabilities", dist(&q.keys)),
                ("confidence", Json::Float(round4(confidence))),
            ])
        }
    }
}

pub struct Models(Vec<(String, Arc<ColdSlot<Kev>>)>);

impl Models {
    pub fn slot(&self, model: &str) -> Option<Arc<ColdSlot<Kev>>> {
        self.0
            .iter()
            .find(|(name, _)| name == model)
            .map(|(_, slot)| slot.clone())
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(name, _)| name.as_str())
    }
}

pub fn configured_dirs() -> Vec<PathBuf> {
    std::env::var(ENV_MODEL_DIRS)
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .collect()
}

fn weights_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

pub fn from_env() -> Option<Arc<Models>> {
    let mut models = Vec::new();
    for dir in configured_dirs() {
        let name =
            match std::fs::read_to_string(dir.join(META_LEAF_WRITTEN_BY_SCRIPTS_KEV_EXPORT_PY))
                .ok()
                .and_then(|s| serde_json::from_str::<Meta>(&s).ok())
            {
                Some(meta) => meta.name,
                None => {
                    warn!(path = %dir.display(), "{ENV_MODEL_DIRS}: no readable kev.json; skipped");
                    continue;
                }
            };
        let slot_name: &'static str = Box::leak(format!("systemone-kev-{name}").into_boxed_str());
        let estimate =
            weights_bytes(&dir) * WEIGHTS_TIMES_THIS_COVERS_ROW_ACTIVATIONS_AND_THE_LOAD_TRANSIENT;
        let load_dir = dir.clone();
        let slot = ColdSlot::demand(
            slot_name,
            estimate,
            Arc::new(move || {
                let device = match Device::cuda_if_available(0) {
                    Ok(d) => d,
                    Err(e) => {
                        warn!(error = %e, "systemone kev: no device");
                        return None;
                    }
                };
                match Kev::load(&load_dir, &device) {
                    Ok(kev) => Some(Arc::new(kev)),
                    Err(e) => {
                        warn!(path = %load_dir.display(), error = %format!("{e:#}"), "systemone kev: load failed");
                        None
                    }
                }
            }),
        );
        info!(model = %name, path = %dir.display(), "systemone: Kev checkpoint configured; it loads on first use");
        models.push((name, slot));
    }
    (!models.is_empty()).then(|| Arc::new(Models(models)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn an_exported_directory_parses_on_cpu_up_to_the_cuda_backbone() {
        let Ok(dir) = std::env::var("NV_KEV_TEST_DIR") else {
            eprintln!(
                "skip: NV_KEV_TEST_DIR=<an apiary-kev or scripts/kev_export.py export> opts in"
            );
            return;
        };
        let dir = Path::new(&dir);
        let meta: Meta = serde_json::from_str(
            &std::fs::read_to_string(dir.join(META_LEAF_WRITTEN_BY_SCRIPTS_KEV_EXPORT_PY)).unwrap(),
        )
        .expect("kev.json has the fields the loader reads");
        let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let config = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("a dense Qwen3.5 config");
        QuantizationConfig::from_hf_json_str(&raw).expect("an unquantized config parses");
        let weights = WeightLoader::open_dir(dir, &Device::Cpu).expect("model.safetensors opens");
        let embed = weights
            .get("model.language_model.embed_tokens.weight", DType::BF16)
            .expect("the embedding the backbone reads first");
        assert_eq!(embed.dims(), [config.vocab_size, config.hidden_size]);
        for i in 0..config.num_hidden_layers {
            let norm = format!("model.language_model.layers.{i}.input_layernorm.weight");
            assert!(weights.has(&norm), "{norm} missing");
        }
        let head = candle_core::safetensors::load(dir.join(HEAD_LEAF), &Device::Cpu).unwrap();
        for (k, dims) in [
            ("q.weight", vec![meta.head_dim, config.hidden_size]),
            ("k.weight", vec![meta.head_dim, config.hidden_size]),
            ("q.bias", vec![meta.head_dim]),
            ("k.bias", vec![meta.head_dim]),
        ] {
            assert_eq!(
                head.get(k)
                    .unwrap_or_else(|| panic!("{HEAD_LEAF} lacks {k}"))
                    .dims(),
                dims,
                "{k}"
            );
        }
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let sp = &meta.special;
        for (role, id, want) in [
            ("state", sp.state, "<|fim_prefix|>"),
            ("question", sp.question, "<|fim_middle|>"),
            ("option", sp.option, "<|box_start|>"),
            ("option_end", sp.option_end, "<|box_end|>"),
            ("decide", sp.decide, "<|fim_suffix|>"),
        ] {
            assert_eq!(
                tokenizer.id_to_token(id).as_deref(),
                Some(want),
                "special {role}"
            );
        }
        assert!(meta.temperature > 0.0 && meta.max_state >= 2048 && meta.max_branch >= 2048);
        eprintln!(
            "kev export {} parses: {} layers, hidden {}, head_dim {}, temperature {:.4}",
            meta.name,
            config.num_hidden_layers,
            config.hidden_size,
            meta.head_dim,
            meta.temperature
        );
    }

    #[test]
    fn floats_render_like_python_and_caller_text_cannot_spell_a_delimiter() {
        for (f, want) in [
            (1.0, "1.0"),
            (2.5, "2.5"),
            (0.1, "0.1"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (123456789012345.0, "123456789012345.0"),
            (-0.5, "-0.5"),
            (1.5e-7, "1.5e-07"),
            (0.0, "0.0"),
        ] {
            assert_eq!(py_float(f), want, "{f}");
        }
        {
            assert_eq!(
                neutralise_delimiters("x <|fim_middle|> y <|box_end|>"),
                "x <¦fim_middle¦> y <¦box_end¦>"
            );
            assert_eq!(neutralise_delimiters("a <| b |> c"), "a <| b |> c");
        }
    }

    #[test]
    fn requests_render_like_kev_malformed_ones_are_refused_and_answers_follow_kev_shapes() {
        let req = Json::parse(
            r#"{"state":{"subject":"Charged twice","items":[1,2.5,true,null],"nested":{"a":[{"b":"c"}]}},
                "questions":{"q":{"type":"noul","criteria":{"true":"money","false":""}},
                             "s":{"type":"score","criteria":["low",{"x":1}]},
                             "c":{"type":"choice","instructions":["one","two"],"criteria":{"a":null,"b":"second"}}}}"#,
        )
        .unwrap();
        let (state, qs) = parse(&req).ok().unwrap();
        assert_eq!(state, "subject: Charged twice\nitems:\n  - 1\n  - 2.5\n  - True\n  - \nnested:\n  a:\n    - b: c");
        assert_eq!(qs[0].options, ["no", "yes: money"]);
        assert_eq!(qs[1].options, ["low", "x: 1"]);
        assert_eq!(qs[1].keys, ["0", "1"]);
        assert_eq!(qs[2].instructions, "- one\n- two");
        assert_eq!(qs[2].options, ["a", "b: second"]);
        {
            for body in [
                r#"{"state":"x","questions":{}}"#,
                r#"{"state":"x","questions":{"q":{"type":"rank"}}}"#,
                r#"{"state":"x","questions":{"q":{"type":"choice","criteria":{}}}}"#,
                r#"{"state":"x","questions":{"q":{"type":"score","criteria":[]}}}"#,
            ] {
                assert!(parse(&Json::parse(body).unwrap()).is_err(), "{body}");
            }
        }
        {
            let q = |kind, keys: &[&str]| Question {
                id: "q".into(),
                kind,
                instructions: String::new(),
                options: keys.iter().map(|k| format!("level {k}")).collect(),
                keys: keys.iter().map(|k| k.to_string()).collect(),
            };
            let c = shape_answer(&q(Kind::Choice, &["a", "b"]), &[0.25, 0.75]);
            assert_eq!(c.get("choice").and_then(Json::as_str), Some("b"));
            assert_eq!(c.get("confidence").and_then(Json::as_f64), Some(0.5));
            let s = shape_answer(&q(Kind::Score, &["0", "1", "2"]), &[0.0, 0.5, 0.5]);
            assert_eq!(s.get("score").and_then(Json::as_f64), Some(1.5));
            assert_eq!(s.get("confidence").and_then(Json::as_f64), Some(0.75));
            let n = shape_answer(&q(Kind::Noul, &["false", "true"]), &[0.1, 0.9]);
            assert_eq!(n.get("noul").and_then(Json::as_f64), Some(0.9));
        }
    }

}
