use crate::json::Json;
use crate::model::{softmax_in_place, Batch, DecisionModel};
use crate::seq::{self, Prefix, QType, Question, Special};
use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

const PREFIX_CACHE_LIMIT: usize = 8192;

pub struct Settings {
    pub devices: Vec<Device>,
    pub dtype: DType,
    pub flash: bool,
}

pub fn open_cuda_device(ordinal: usize) -> candle_core::Result<Device> {
    #[cfg(feature = "cuda")]
    let _ = nv_quant::capture_exclusion::A_DEVICE_WIDE_SYNCHRONIZE_ON_ANY_THREAD_INVALIDATES_EVERY_CAPTURE_IN_FLIGHT_IN_THE_PROCESS;
    #[cfg(feature = "cuda")]
    let _fresh_context_synchronizes = nv_quant::capture_exclusion::no_capture_in_flight();
    Device::new_cuda_with_stream(ordinal)
}

pub fn pick_devices(spec: &str, workers_per_device: usize) -> Result<Settings> {
    let workers = workers_per_device.max(1);
    let cpu = || Settings {
        devices: vec![Device::Cpu; workers],
        dtype: DType::F32,
        flash: false,
    };
    let cuda = |ordinals: &[usize]| -> Result<Settings> {
        let mut devices = Vec::new();
        for ordinal in ordinals {
            for _ in 0..workers {
                devices.push(
                    open_cuda_device(*ordinal)
                        .with_context(|| format!("opening cuda:{ordinal}"))?,
                );
            }
        }
        Ok(Settings {
            devices,
            dtype: DType::F16,
            flash: cfg!(feature = "flash-attn"),
        })
    };
    let metal = || -> Result<Settings> {
        let device = Device::new_metal(0).context("opening the Metal device")?;
        Ok(Settings {
            devices: vec![device; workers],
            dtype: DType::F16,
            flash: false,
        })
    };
    match spec {
        "cpu" => Ok(cpu()),
        "metal" => metal(),
        "auto" => match cfg!(feature = "cuda")
            .then(|| cuda(&[0]))
            .or_else(|| cfg!(feature = "metal").then(metal))
        {
            Some(Ok(settings)) => Ok(settings),
            _ => Ok(cpu()),
        },
        list => {
            if !cfg!(feature = "cuda") {
                bail!("this build has no CUDA support; rebuild with --features cuda");
            }
            let ordinals = list
                .split(',')
                .map(|d| {
                    let index = d.trim().strip_prefix("cuda").ok_or_else(|| {
                        anyhow!("unknown device {d:?}; use auto, cpu, metal, cuda or cuda:0,cuda:1")
                    })?;
                    let index = index.trim_start_matches(':');
                    Ok(if index.is_empty() { 0 } else { index.parse()? })
                })
                .collect::<Result<Vec<usize>>>()?;
            cuda(&ordinals)
        }
    }
}

pub struct Prepared {
    pub questions: Vec<Question>,
    pub sequences: Vec<Vec<u32>>,
    pub markers: Vec<Vec<usize>>,
}

impl Prepared {
    pub fn tokens(&self) -> usize {
        self.sequences.iter().map(Vec::len).sum()
    }
}

pub struct Checkpoint {
    tokenizer: Tokenizer,
    special: Special,
    max_len: usize,
    head_max_len: usize,
    temperature: [f64; 3],
    temperature_by_options: HashMap<String, f64>,
    replicas: Vec<DecisionModel>,
    prefixes: Mutex<HashMap<String, Arc<Prefix>>>,
}

fn token_id(tok: &Tokenizer, cfg: &Json, key: &str, fallback: &str) -> Result<(u32, String)> {
    let text = match cfg.get(key) {
        Some(Json::Str(s)) => s.clone(),
        Some(entry) => entry
            .get("content")
            .and_then(Json::as_str)
            .unwrap_or(fallback)
            .to_string(),
        None => fallback.to_string(),
    };
    let id = tok
        .token_to_id(&text)
        .ok_or_else(|| anyhow!("tokenizer has no {key} {text:?}"))?;
    Ok((id, text))
}

impl Checkpoint {
    pub fn load(dir: &Path, settings: &Settings) -> Result<Self> {
        let read = |rel: &str| -> Result<Json> {
            let text = std::fs::read_to_string(dir.join(rel))
                .with_context(|| format!("reading {}", dir.join(rel).display()))?;
            Ok(Json::parse(&text)?)
        };
        let cfg = read("rl_agent_config.json")?;
        let tok_cfg = read("tokenizer/tokenizer_config.json").unwrap_or(Json::Null);
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer/tokenizer.json"))
            .map_err(|e| anyhow!("loading tokenizer.json: {e}"))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow!("{e}"))?;
        let (cls, _) = token_id(&tokenizer, &tok_cfg, "cls_token", "[CLS]")?;
        let (sep, _) = token_id(&tokenizer, &tok_cfg, "sep_token", "[SEP]")?;
        let (pad, _) = token_id(&tokenizer, &tok_cfg, "pad_token", "[PAD]")?;
        let (mask, mask_token) = token_id(&tokenizer, &tok_cfg, "mask_token", "[MASK]")?;
        let number = |key: &str, default: usize| {
            cfg.get(key)
                .and_then(Json::as_f64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };
        let max_len = number("max_len", 512);
        let mut temperature = [1.0; 3];
        for (slot, v) in temperature
            .iter_mut()
            .zip(cfg.get("temperature").and_then(Json::as_arr).unwrap_or(&[]))
        {
            *slot = v.as_f64().unwrap_or(1.0);
        }
        let temperature_by_options = cfg
            .get("temperature_by_options")
            .and_then(Json::as_obj)
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| Some((k.clone(), v.as_f64()?)))
                    .collect()
            })
            .unwrap_or_default();
        let replicas = settings
            .devices
            .iter()
            .map(|device| {
                DecisionModel::load(
                    dir,
                    device,
                    settings.dtype,
                    max_len,
                    pad,
                    number("head_layers", 2),
                    settings.flash,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            tokenizer,
            special: Special {
                cls,
                sep,
                mask,
                mask_token,
            },
            max_len,
            head_max_len: number("head_max_len", 192),
            temperature,
            temperature_by_options,
            replicas,
            prefixes: Mutex::new(HashMap::new()),
        })
    }

    pub fn replicas(&self) -> usize {
        self.replicas.len()
    }

    pub fn set_full_final_head(&mut self, full: bool) {
        for replica in &mut self.replicas {
            replica.full_final_head = full;
        }
    }

    pub fn set_cuda_graph(&mut self, on: bool) {
        for replica in &mut self.replicas {
            replica.cuda_graph = on;
        }
    }

    #[cfg(feature = "flash-attn")]
    pub fn graph_stats(&self, replica: usize) -> crate::model::GraphStats {
        self.replicas[replica % self.replicas.len()].graph_stats()
    }

    fn prefix(&self, q: &Question, def: &Json) -> Result<Arc<Prefix>> {
        let key = def.to_string();
        if let Some(hit) = self.prefixes.lock().unwrap().get(&key) {
            return Ok(hit.clone());
        }
        let built = Arc::new(seq::build_prefix(
            &self.tokenizer,
            &self.special,
            q,
            self.head_max_len,
        )?);
        if built.markers.len() != q.options.len() {
            bail!(
                "question {:?} options exceed head_max_len={}",
                q.id,
                self.head_max_len
            );
        }
        let mut cache = self.prefixes.lock().unwrap();
        if cache.len() >= PREFIX_CACHE_LIMIT {
            cache.clear();
        }
        cache.insert(key, built.clone());
        Ok(built)
    }

    pub fn prepare(&self, state: &Json, questions: &[(String, Json)]) -> Result<Prepared> {
        if questions.is_empty() {
            bail!("no questions given");
        }
        let text = state.state_text().replace(&self.special.mask_token, " ");
        let state_ids = seq::encode(&self.tokenizer, &text)?;
        let mut out = Prepared {
            questions: Vec::new(),
            sequences: Vec::new(),
            markers: Vec::new(),
        };
        for (id, def) in questions {
            let q = seq::parse_question(id, def)?;
            let prefix = self.prefix(&q, def)?;
            let (ids, markers) =
                seq::finish_sequence(&prefix, &state_ids, &self.special, self.max_len);
            if markers.len() != q.options.len() {
                bail!("question {id:?} options exceed max_len={}", self.max_len);
            }
            out.questions.push(q);
            out.sequences.push(ids);
            out.markers.push(markers);
        }
        Ok(out)
    }

    pub fn run(&self, replica: usize, jobs: &[&Prepared]) -> Result<Vec<Json>> {
        let sequences: Vec<Vec<u32>> = jobs
            .iter()
            .flat_map(|j| j.sequences.iter().cloned())
            .collect();
        let markers: Vec<Vec<usize>> = jobs
            .iter()
            .flat_map(|j| j.markers.iter().cloned())
            .collect();
        let qtypes: Vec<u32> = jobs
            .iter()
            .flat_map(|j| j.questions.iter().map(|q| q.qtype as u32))
            .collect();
        let model = &self.replicas[replica % self.replicas.len()];
        let out = model.forward(&Batch {
            sequences: &sequences,
            markers: &markers,
            qtypes: &qtypes,
        })?;
        let mut results = Vec::with_capacity(jobs.len());
        let mut row = 0;
        for job in jobs {
            let mut answers = Vec::with_capacity(job.questions.len());
            for q in &job.questions {
                answers.push((
                    q.id.clone(),
                    self.answer(model, q, &out.logits[row], &out.pooled[row]),
                ));
                row += 1;
            }
            results.push(Json::obj([
                ("model", Json::Null),
                ("answers", Json::Obj(answers)),
                (
                    "usage",
                    Json::obj([
                        ("input_tokens", job.tokens().into()),
                        ("output_tokens", 0usize.into()),
                    ]),
                ),
            ]));
        }
        Ok(results)
    }

    fn answer(&self, model: &DecisionModel, q: &Question, logits: &[f32], pooled: &[f32]) -> Json {
        let k = logits.len();
        let mut raw = logits.to_vec();
        softmax_in_place(&mut raw);
        let mut sorted = raw.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let kk = k.max(2) as f32;
        let entropy = -raw.iter().map(|p| p * p.max(1e-9).ln()).sum::<f32>() / kk.ln();
        let second = sorted.get(1).copied().unwrap_or(0.0);
        let act = model
            .act_head
            .probabilities(pooled, [sorted[0], sorted[0] - second, entropy, kk / 255.0]);
        let action = Json::obj([("act_probability", round4(act[0] as f64))]);

        let bucket = format!(
            "{}:{}",
            q.qtype.name(),
            if k <= 2 {
                "2"
            } else if k <= 5 {
                "3-5"
            } else if k <= 10 {
                "6-10"
            } else {
                "11+"
            }
        );
        let t = self
            .temperature_by_options
            .get(&bucket)
            .copied()
            .unwrap_or(self.temperature[q.qtype as usize])
            .max(1e-3);
        let z: Vec<f64> = logits.iter().map(|l| *l as f64 / t).collect();
        let zmax = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = z.iter().map(|v| (v - zmax).exp()).collect();
        let total: f64 = exp.iter().sum();
        let p: Vec<f64> = exp.iter().map(|e| e / total).collect();
        let ent: f64 = -p.iter().map(|v| v * v.clamp(1e-12, 1.0).ln()).sum::<f64>();
        let confidence = round4((1.0 - ent / (k.max(2) as f64).ln()).clamp(0.0, 1.0));
        let kind = ("type", Json::from(q.qtype.name()));
        match q.qtype {
            QType::Choice => {
                let best = p
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let probs = Json::obj(
                    q.keys
                        .iter()
                        .zip(&p)
                        .map(|(key, v)| (key.clone(), round4(*v))),
                );
                Json::obj([
                    kind,
                    ("choice", q.keys[best].clone().into()),
                    ("probabilities", probs),
                    ("confidence", confidence),
                    ("action", action),
                ])
            }
            QType::Score => {
                let expected: f64 = p.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
                let legend = Json::obj(
                    q.keys
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (i.to_string(), Json::from(c.clone()))),
                );
                let probs = Json::obj(
                    p.iter()
                        .enumerate()
                        .map(|(i, v)| (i.to_string(), round4(*v))),
                );
                Json::obj([
                    kind,
                    ("score", round4(expected)),
                    ("legend", legend),
                    ("probabilities", probs),
                    ("confidence", confidence),
                    ("action", action),
                ])
            }
            QType::Noul => Json::obj([
                kind,
                ("noul", round4(p[1])),
                ("confidence", round4(p[1].max(1.0 - p[1]))),
                ("action", action),
            ]),
        }
    }
}

fn round4(v: f64) -> Json {
    Json::Float((v * 1e4).round() / 1e4)
}
