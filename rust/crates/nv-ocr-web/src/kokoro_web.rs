use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

use anyhow::{anyhow, Context, Result};
use candle_core::{Device, Tensor};
use nv_tts::kokoro::serve::KokoroCandle;
use nv_tts::kokoro::style_fit::{self, StyleFitConfig};
use nv_tts::kokoro::train::{Encoders, Targets};
use nv_tts::kokoro::wespeaker::WeSpeaker;
use nv_tts::kokoro::{KokoroFrontEnd, TrainConfig};
use nv_tts::SpeakerEncoder;
use safetensors::tensor::{Dtype, TensorView};

pub const KOKORO_IN_A_TAB_IS_THE_CANDLE_CPU_GRAPH_COMPILED_TO_WASM32_NOT_A_WGPU_PORT: &str =
    "nv-tts has no wgpu kernels; the browser surface runs the same candle graphs the engine's \
     /v1/voices/fit runs (Kokoro front end, WeSpeaker, the Qwen speaker encoder) on the wasm32 \
     CPU backend, single-threaded, so the style tier is the engine's evolution loop over \
     staged stock rows and the adapter tier is the engine's candle-autograd AdaIN fit, both \
     against reference clips staged as raw 24 kHz f32";

pub const KOKORO_STAGE_NAMES: [&str; 5] = [
    "kokoro.safetensors",
    "wespeaker/graph.json",
    "wespeaker/weights.safetensors",
    "qwen-spk/config.json",
    "qwen-spk/model.safetensors",
];

thread_local! {
    static KOKORO_STAGE: RefCell<BTreeMap<String, Vec<u8>>> = const { RefCell::new(BTreeMap::new()) };
    static KOKORO: RefCell<Option<KokoroCandle>> = const { RefCell::new(None) };
    static KOKORO_REFS: RefCell<BTreeMap<String, Vec<f32>>> = const { RefCell::new(BTreeMap::new()) };
    static KOKORO_ADAPTER: RefCell<Option<HashMap<String, Tensor>>> = const { RefCell::new(None) };
}

pub fn kokoro_stage_clear() {
    KOKORO_STAGE.with(|s| s.borrow_mut().clear());
    KOKORO_REFS.with(|s| s.borrow_mut().clear());
}

pub fn kokoro_stage_reserve(name: &str, size: u32) {
    KOKORO_STAGE.with(|s| {
        s.borrow_mut()
            .entry(name.to_string())
            .or_default()
            .reserve_exact(size as usize)
    });
}

pub fn kokoro_stage_push(name: &str, chunk: &[u8]) {
    KOKORO_STAGE.with(|s| {
        s.borrow_mut()
            .entry(name.to_string())
            .or_default()
            .extend_from_slice(chunk)
    });
}

pub fn kokoro_stage_ref(name: &str, pcm24k: &[f32]) {
    KOKORO_REFS.with(|s| s.borrow_mut().insert(name.to_string(), pcm24k.to_vec()));
}

pub fn kokoro_staged() -> String {
    let files: Vec<serde_json::Value> = KOKORO_STAGE.with(|s| {
        s.borrow()
            .iter()
            .map(|(k, v)| serde_json::json!({"name": k, "bytes": v.len()}))
            .collect()
    });
    let refs: Vec<serde_json::Value> = KOKORO_REFS.with(|s| {
        s.borrow()
            .iter()
            .map(|(k, v)| serde_json::json!({"name": k, "samples": v.len()}))
            .collect()
    });
    serde_json::json!({"files": files, "refs": refs}).to_string()
}

fn take_stage(stage: &mut BTreeMap<String, Vec<u8>>, name: &str) -> Result<Vec<u8>> {
    stage
        .remove(name)
        .ok_or_else(|| anyhow!("kokoro stage carries no {name}"))
}

fn build_kokoro() -> Result<(usize, bool)> {
    let _ = KOKORO_IN_A_TAB_IS_THE_CANDLE_CPU_GRAPH_COMPILED_TO_WASM32_NOT_A_WGPU_PORT;
    let mut stage = KOKORO_STAGE.with(|s| std::mem::take(&mut *s.borrow_mut()));
    let dev = Device::Cpu;
    let model = KokoroFrontEnd::from_safetensors_bytes(
        take_stage(&mut stage, KOKORO_STAGE_NAMES[0])?,
        &dev,
    )
    .context("build the Kokoro front end from the staged safetensors")?;
    let encoders = if KOKORO_STAGE_NAMES[1..]
        .iter()
        .all(|n| stage.contains_key(*n))
    {
        let graph = String::from_utf8(take_stage(&mut stage, KOKORO_STAGE_NAMES[1])?)?;
        let wes = WeSpeaker::from_bytes(
            &graph,
            &take_stage(&mut stage, KOKORO_STAGE_NAMES[2])?,
            &dev,
        )
        .context("build WeSpeaker")?;
        let cfg = String::from_utf8(take_stage(&mut stage, KOKORO_STAGE_NAMES[3])?)?;
        let qwen = SpeakerEncoder::from_standalone_bytes(
            &cfg,
            &take_stage(&mut stage, KOKORO_STAGE_NAMES[4])?,
            &dev,
        )
        .context("build the Qwen speaker encoder")?;
        Some(Encoders::from_parts(qwen, wes, &dev)?)
    } else {
        None
    };
    let has_encoders = encoders.is_some();
    let candle = KokoroCandle::from_parts(model, encoders);
    let n = candle.n_adain();
    KOKORO.with(|k| *k.borrow_mut() = Some(candle));
    Ok((n, has_encoders))
}

pub fn init_core() -> Result<String> {
    let (n_adain, encoders) = build_kokoro()?;
    Ok(serde_json::json!({"n_adain": n_adain, "encoders": encoders}).to_string())
}

fn with_kokoro<T>(f: impl FnOnce(&KokoroCandle) -> Result<T>) -> Result<T> {
    KOKORO.with(|k| {
        let g = k.borrow();
        let c = g
            .as_ref()
            .ok_or_else(|| anyhow!("kokoro_init has not run"))?;
        f(c)
    })
}

fn padded(ids: &[u32]) -> Vec<u32> {
    let mut v = Vec::with_capacity(ids.len() + 2);
    v.push(0);
    v.extend_from_slice(ids);
    v.push(0);
    v
}

pub fn synth_core(
    ids: &[u32],
    style: &[f32],
    speed: f64,
    seed: u64,
    adapter: Option<&HashMap<String, Tensor>>,
) -> Result<Vec<f32>> {
    with_kokoro(|c| c.synthesize(&padded(ids), style, speed, adapter, seed))
}

pub fn adapter_map() -> Option<HashMap<String, Tensor>> {
    KOKORO_ADAPTER.with(|a| a.borrow().clone())
}

struct FitOpts {
    refs: Vec<String>,
    train_tokens: Vec<Vec<u32>>,
    heldout_tokens: Vec<Vec<u32>>,
    stock: BTreeMap<String, Vec<f32>>,
    budget: usize,
    population: usize,
    lambda: f32,
    patience: usize,
    min_delta: f32,
    seed: u64,
    epochs: usize,
    wmix: f64,
    train_seconds: Option<f64>,
}

pub const DECODER_FRAMES_PER_SECOND_IS_24000_OVER_THE_GENERATOR_UPSAMPLE_SCALE: usize =
    24_000 / 300;

fn tokens_of(v: &serde_json::Value, key: &str) -> Result<Vec<Vec<u32>>> {
    v.get(key)
        .and_then(|a| a.as_array())
        .ok_or_else(|| anyhow!("fit opts: {key} must be a list of token lists"))?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| anyhow!("fit opts: {key} rows must be lists"))?
                .iter()
                .map(|t| {
                    t.as_u64()
                        .map(|x| x as u32)
                        .ok_or_else(|| anyhow!("fit opts: {key} token is not a u32"))
                })
                .collect()
        })
        .collect()
}

impl FitOpts {
    fn parse(opts_json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(opts_json).context("fit opts json")?;
        let sd = StyleFitConfig::default();
        let num = |k: &str, d: f64| v.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
        let mut stock = BTreeMap::new();
        if let Some(m) = v.get("stock").and_then(|x| x.as_object()) {
            for (k, row) in m {
                let r: Vec<f32> = row
                    .as_array()
                    .ok_or_else(|| anyhow!("stock row {k} is not a list"))?
                    .iter()
                    .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                    .collect();
                stock.insert(k.clone(), r);
            }
        }
        Ok(Self {
            refs: v
                .get("refs")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            train_tokens: tokens_of(&v, "train_tokens")?,
            heldout_tokens: tokens_of(&v, "heldout_tokens")?,
            stock,
            budget: num("budget", sd.budget as f64) as usize,
            population: num("population", sd.population as f64) as usize,
            lambda: num("lambda", sd.lambda as f64) as f32,
            patience: num("patience", sd.patience as f64) as usize,
            min_delta: num("min_delta", sd.min_delta as f64) as f32,
            seed: num("seed", sd.seed as f64) as u64,
            epochs: num("epochs", TrainConfig::default().epochs as f64) as usize,
            wmix: num("wmix", TrainConfig::default().wmix),
            train_seconds: v.get("train_seconds").and_then(|x| x.as_f64()),
        })
    }
}

fn staged_refs(names: &[String]) -> Result<Vec<Vec<f32>>> {
    KOKORO_REFS.with(|r| {
        let r = r.borrow();
        let picked: Vec<&String> = if names.is_empty() {
            r.keys().collect()
        } else {
            names.iter().collect()
        };
        picked
            .into_iter()
            .map(|n| {
                r.get(n)
                    .cloned()
                    .ok_or_else(|| anyhow!("no staged reference clip {n}"))
            })
            .collect()
    })
}

fn targets_of(c: &KokoroCandle, refs: &[Vec<f32>]) -> Result<Targets> {
    let refs: Vec<Vec<f32>> = refs.iter().filter(|r| r.len() >= 24_000).cloned().collect();
    if refs.is_empty() {
        return Err(anyhow!(
            "no usable reference clip (need >= 1 s of 24 kHz audio)"
        ));
    }
    c.targets(&refs)
}

pub fn fit_style_core(opts_json: &str, on_event: &mut dyn FnMut(String)) -> Result<String> {
    let on_event = RefCell::new(on_event);
    let on_event = |j: String| (on_event.borrow_mut())(j);
    let opts = FitOpts::parse(opts_json)?;
    let refs = staged_refs(&opts.refs)?;
    let out = with_kokoro(|c| {
        let targets = targets_of(c, &refs)?;
        let stock: Vec<(String, Vec<f32>)> = opts
            .stock
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let train: Vec<Vec<u32>> = opts.train_tokens.iter().map(|t| padded(t)).collect();
        let heldout: Vec<Vec<u32>> = opts.heldout_tokens.iter().map(|t| padded(t)).collect();
        let score_on = |style: &[f32], set: &[Vec<u32>]| -> Result<f32> {
            let mut acc = 0.0f32;
            for (i, t) in set.iter().enumerate() {
                let audio = c.synthesize(t, style, 1.0, None, 11 + i as u64)?;
                acc += c.score(&audio, &targets)?;
            }
            Ok(acc / set.len().max(1) as f32)
        };
        let cfg = StyleFitConfig {
            budget: opts.budget,
            population: opts.population,
            lambda: opts.lambda,
            patience: opts.patience,
            min_delta: opts.min_delta,
            seed: opts.seed,
        };
        let fit = style_fit::evolve(
            &stock,
            &cfg,
            |s| score_on(s, &train),
            |s| score_on(s, &heldout),
            |best_stock| {
                on_event(
                    serde_json::json!({"type": "style_start", "best_stock": best_stock.0, "wespeaker": best_stock.1}).to_string(),
                )
            },
            |ev| {
                on_event(
                    serde_json::json!({"type": "style_gen", "generation": ev.generation, "validation": ev.validation, "best": ev.best}).to_string(),
                )
            },
        )?;
        Ok(serde_json::json!({
            "style": fit.style,
            "weights": fit.weights,
            "validation": fit.validation,
            "best_stock": {"name": fit.best_stock.0, "wespeaker": fit.best_stock.1},
            "history": fit.history,
            "generations": fit.generations,
            "best_step": fit.best_step,
            "stop": fit.stop,
        }))
    })?;
    Ok(out.to_string())
}

pub fn fit_adapter_core(
    opts_json: &str,
    style: &[f32],
    on_event: &mut dyn FnMut(String),
) -> Result<String> {
    let opts = FitOpts::parse(opts_json)?;
    let refs = staged_refs(&opts.refs)?;
    let out = with_kokoro(|c| {
        let targets = targets_of(c, &refs)?;
        let train: Vec<Vec<u32>> = opts.train_tokens.iter().map(|t| padded(t)).collect();
        let heldout: Vec<Vec<u32>> = opts.heldout_tokens.iter().map(|t| padded(t)).collect();
        let cfg = TrainConfig {
            epochs: opts.epochs.max(1),
            patience: opts.patience.max(1),
            min_delta: opts.min_delta,
            wmix: opts.wmix,
            window_frames: opts.train_seconds.map(|sec| {
                ((sec * DECODER_FRAMES_PER_SECOND_IS_24000_OVER_THE_GENERATOR_UPSAMPLE_SCALE as f64)
                    .ceil() as usize)
                    .max(8)
            }),
            ..TrainConfig::default()
        };
        let fit = c.fit_adapter(&targets, style, &train, &heldout, &cfg, |rep| {
            on_event(
                serde_json::json!({"type": "adapter_epoch", "epoch": rep.epoch, "train_loss": rep.train_loss, "heldout_wespeaker": rep.heldout_wespeaker, "heldout_qwen": rep.heldout_qwen}).to_string(),
            )
        })?;
        let epochs: Vec<serde_json::Value> = fit
            .reports
            .iter()
            .map(|r| serde_json::json!({"epoch": r.epoch, "train_loss": r.train_loss, "heldout_wespeaker": r.heldout_wespeaker, "heldout_qwen": r.heldout_qwen}))
            .collect();
        let n = fit.adain.len();
        KOKORO_ADAPTER.with(|a| *a.borrow_mut() = Some(fit.adain));
        Ok(serde_json::json!({
            "best_heldout": fit.best_heldout,
            "best_epoch": fit.best_epoch,
            "stop": fit.stop,
            "epochs": epochs,
            "tensors": n,
        }))
    })?;
    Ok(out.to_string())
}

pub fn adapter_safetensors() -> Result<Option<Vec<u8>>> {
    let Some(map) = KOKORO_ADAPTER.with(|a| a.borrow().clone()) else {
        return Ok(None);
    };
    let mut blobs: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(map.len());
    for (k, v) in &map {
        let vals = v.flatten_all()?.to_vec1::<f32>()?;
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for x in vals {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        blobs.push((k.clone(), v.dims().to_vec(), bytes));
    }
    blobs.sort_by(|a, b| a.0.cmp(&b.0));
    let views = blobs
        .iter()
        .map(|(name, shape, bytes)| {
            TensorView::new(Dtype::F32, shape.clone(), bytes).map(|v| (name.as_str(), v))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(Some(safetensors::serialize(views, None)?))
}
