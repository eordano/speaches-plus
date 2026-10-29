use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use nv_tts::kokoro::style_fit::{self, StyleFitConfig};
use nv_tts::kokoro::TrainConfig;
use nv_tts::{VoiceKind, VoiceMeta, VoiceProfileStore, ADAPTER_FILE, PACK_FILE};
use serde::{Deserialize, Serialize};

use super::KokoroHandle;
use crate::diarization::embedding::EmbeddingModel;

pub const TRAIN_SENTENCES: [&str; 8] = [
    "The data center build out this year is bigger than anything the industry has done before.",
    "Memory prices went up because every hyperscaler signed long term contracts at the same time.",
    "Most of the cost of a leading edge wafer is lithography, and that share keeps growing.",
    "You cannot build a gigawatt of capacity without the transformers, and those are sold out.",
    "The interesting question is who captures the margin, the chip vendor or the cloud provider.",
    "Every roadmap assumes yields improve, and the last two nodes did not cooperate on schedule.",
    "Inference demand is now large enough that it changes how the chips themselves get designed.",
    "Packaging turned out to be the constraint nobody planned for, and it takes years to add.",
];

pub const TRAIN_SENTENCES_ES: [&str; 8] = [
    "El vapor sale con la primera luz y la carga tiene que estar en el muelle antes de la medianoche.",
    "Nadie en la plaza sabe quién firmó la orden, pero todos saben quién la va a pagar.",
    "Si el gobernador pide caballos, los pide para su gente y no para los de afuera.",
    "La pulpería cierra cuando se acaba la vela, y esta noche la vela se acaba temprano.",
    "Le dije al capataz que el agua del arroyo no alcanza para dos estancias en enero.",
    "Los papeles del consulado llegaron mojados y la mitad de las firmas se borraron.",
    "Mi padre repetía que un pacto entre enemigos dura lo que dura el miedo que lo hizo.",
    "Cuente las carretas antes de prometer nada, porque después nadie se acuerda del número.",
];

pub const DEFAULT_SENTENCES_ES: [&str; 4] = [
    "Buenas noches, forastero, la plaza está tranquila pero el muelle todavía no.",
    "No le puedo vender fiado, y menos a alguien que llegó ayer con el vapor.",
    "El enviado del imperio habla de navegación libre, pero nunca dice quién pone los barcos.",
    "Cuando la máquina contesta, lo hace con palabras prestadas de los que ya no están.",
];

pub const DEFAULT_SENTENCES: [&str; 4] = [
    "The biggest bottleneck right now is not the chips, it is power and the memory supply chain.",
    "Google didn't see revenue skyrocket, and in a sense that tells you where the demand actually is.",
    "Everyone's roadmap goes from the current node to the next one, but the cost per wafer keeps climbing.",
    "If you look at the supply chain for high bandwidth memory, every single vendor is sold out through next year.",
];

#[derive(Clone, Debug, Deserialize)]
pub struct FitRequest {
    pub name: String,
    #[serde(default)]
    pub speaker_id: Option<String>,

    #[serde(default)]
    pub refs: Vec<String>,

    #[serde(default)]
    pub sentences: Vec<String>,
    #[serde(default = "d_budget")]
    pub budget: usize,
    #[serde(default = "d_patience")]
    pub patience: usize,
    #[serde(default = "d_min_delta")]
    pub min_delta: f32,
    #[serde(default = "d_population")]
    pub population: usize,
    #[serde(default = "d_lambda")]
    pub lambda: f32,
    #[serde(default = "d_seed")]
    pub seed: u64,

    #[serde(default = "d_tier")]
    pub tier: String,

    #[serde(default = "d_epochs")]
    pub epochs: usize,
    #[serde(default)]
    pub speaker: Option<String>,
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default = "d_language")]
    pub language: String,
}
fn d_language() -> String {
    "en-us".to_string()
}
pub fn corpus_for(language: &str) -> (&'static [&'static str], &'static [&'static str]) {
    if language.starts_with("es") {
        (&TRAIN_SENTENCES_ES, &DEFAULT_SENTENCES_ES)
    } else {
        (&TRAIN_SENTENCES, &DEFAULT_SENTENCES)
    }
}
fn d_tier() -> String {
    "style".to_string()
}
fn d_epochs() -> usize {
    12
}
fn d_budget() -> usize {
    60
}
fn d_patience() -> usize {
    8
}
fn d_min_delta() -> f32 {
    0.005
}
fn d_population() -> usize {
    12
}
fn d_lambda() -> f32 {
    0.02
}
fn d_seed() -> u64 {
    1
}

#[derive(Clone, Debug, Serialize)]
pub struct FitEvent {
    pub stage: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wsim: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    pub secs: f32,
}

pub fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    let ch = spec.channels.max(1) as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono: Vec<f32> = if ch == 1 {
        samples
    } else {
        samples
            .chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

pub fn write_wav_24k(path: &Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

pub fn resample_linear(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = ((x.len() as f64) * (to as f64) / (from as f64)) as usize;
    let step = (x.len() as f64 - 1.0) / (n.max(2) as f64 - 1.0);
    (0..n)
        .map(|i| {
            let pos = i as f64 * step;
            let j = pos.floor() as usize;
            let t = (pos - j as f64) as f32;
            let a = x[j.min(x.len() - 1)];
            let b = x[(j + 1).min(x.len() - 1)];
            a + (b - a) * t
        })
        .collect()
}

pub fn npy_f32(shape: &[usize], data: &[f32]) -> Vec<u8> {
    let dims = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let shape_s = if shape.len() == 1 {
        format!("({dims},)")
    } else {
        format!("({dims})")
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_s}, }}");
    let pad = 64 - ((10 + header.len() + 1) % 64);
    header.push_str(&" ".repeat(pad % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / n).collect()
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub struct Fitter {
    pub kokoro: KokoroHandle,
    pub embedding: Arc<EmbeddingModel>,
    pub store: Arc<VoiceProfileStore>,

    pub root: PathBuf,
}

impl Fitter {
    fn resolve_ref(&self, p: &str) -> Result<PathBuf> {
        let path = if Path::new(p).is_absolute() {
            PathBuf::from(p)
        } else {
            self.root.join(p)
        };
        let canon = path
            .canonicalize()
            .with_context(|| format!("reference clip {p:?}"))?;
        let root = self
            .root
            .canonicalize()
            .unwrap_or_else(|_| self.root.clone());
        if !canon.starts_with(&root) {
            return Err(anyhow!("reference clip {p:?} is outside the workspace"));
        }
        Ok(canon)
    }

    fn embed_24k(&self, audio24k: &[f32]) -> Result<Vec<f32>> {
        let x16 = resample_linear(audio24k, 24_000, 16_000);
        Ok(unit(&self.embedding.embed(&x16)?))
    }

    pub fn run(&self, req: &FitRequest, emit: impl FnMut(FitEvent)) -> Result<VoiceMeta> {
        let emit_cell = std::cell::RefCell::new(emit);
        let emit = |e: FitEvent| (emit_cell.borrow_mut())(e);
        if req.tier != "style" && req.tier != "adapter" {
            return Err(anyhow!("tier must be style or adapter"));
        }
        if req.tier == "adapter" && self.kokoro.candle().is_none() {
            return Err(anyhow!("adapter fitting needs KOKORO_CANDLE_WEIGHTS"));
        }
        let staging = crate::voices::stages::TempDir(self.store.root().join(format!(
            ".fit-{}-{}", std::process::id(), std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?.as_nanos())));
        let output = VoiceProfileStore::open(&staging.0)?;
        let t0 = Instant::now();
        let secs = |t0: Instant| t0.elapsed().as_secs_f32();
        if req.refs.is_empty() {
            return Err(anyhow!("refs: at least one reference clip is required"));
        }

        let mut target = vec![0.0f32; 0];
        let mut n_refs = 0usize;
        for r in req.refs.iter().take(12) {
            let (x, sr) = read_wav_mono(&self.resolve_ref(r)?)?;
            let x16 = resample_linear(&x, sr, 16_000);
            if x16.len() < 16_000 {
                continue;
            }
            let e = unit(&self.embedding.embed(&x16)?);
            if target.is_empty() {
                target = vec![0.0; e.len()];
            }
            for (t, v) in target.iter_mut().zip(&e) {
                *t += v;
            }
            n_refs += 1;
        }
        if n_refs == 0 {
            return Err(anyhow!("no usable reference clip (need >= 1 s of audio)"));
        }
        let target = unit(&target);
        let language =
            super::http::espeak_voice_for(&req.language).map_err(|e| anyhow!("language: {e}"))?;
        let (train_corpus, default_sentences) = corpus_for(language);
        let sentences: Vec<String> = if req.sentences.is_empty() {
            default_sentences.iter().map(|s| s.to_string()).collect()
        } else {
            req.sentences.clone()
        };
        let mut tokens = Vec::with_capacity(sentences.len());
        for s in &sentences {
            tokens.push(self.kokoro.tokens_for(s, language)?.1);
        }
        let train_sentences = training_sentences(&sentences, train_corpus)?;
        let train_tokens = train_sentences
            .iter()
            .map(|s| self.kokoro.tokens_for(s, language).map(|v| v.1))
            .collect::<Result<Vec<_>>>()?;
        let score_on = |style: &[f32], set: &[Vec<i64>]| -> Result<f32> {
            let mut acc = 0.0f32;
            for t in set {
                let audio = self.kokoro.synth_tokens_with_style(t, style, 1.0)?;
                acc += dot(&self.embed_24k(&audio)?, &target);
            }
            Ok(acc / set.len() as f32)
        };

        let score = |style: &[f32]| score_on(style, &train_tokens);
        let heldout_score = |style: &[f32]| score_on(style, &tokens);
        let stock = self.kokoro.stock_style_rows();
        let cfg = StyleFitConfig {
            budget: req.budget,
            population: req.population,
            lambda: req.lambda,
            patience: req.patience,
            min_delta: req.min_delta,
            seed: req.seed,
        };
        let fit = style_fit::evolve(
            &stock,
            &cfg,
            score,
            heldout_score,
            |best_stock| {
                emit(FitEvent {
                    stage: "start",
                    step: None,
                    wsim: Some(best_stock.1),
                    best: None,
                    message: Some(format!(
                        "best stock voice {} ({:.3}); {} refs, {} sentences, {} stock voices",
                        best_stock.0,
                        best_stock.1,
                        n_refs,
                        sentences.len(),
                        stock.len()
                    )),
                    secs: secs(t0),
                });
            },
            |ev| {
                emit(FitEvent {
                    stage: "train",
                    step: Some(ev.generation),
                    wsim: Some(ev.validation),
                    best: Some(ev.best),
                    message: None,
                    secs: secs(t0),
                });
            },
        )?;
        let best_stock = fit.best_stock.clone();
        let reason = fit.stop;
        let history = fit.history.clone();
        emit(FitEvent {
            stage: "stop",
            step: Some(fit.generations),
            wsim: None,
            best: Some(history.iter().cloned().fold(f32::NEG_INFINITY, f32::max)),
            message: Some(reason.to_string()),
            secs: secs(t0),
        });

        let style = fit.style.clone();
        let dim = style.len();
        let final_score = fit.validation;
        let name = req.name.clone();
        output.write_file(&name, PACK_FILE, &npy_f32(&[1, 1, dim], &style))?;
        let mut kind = VoiceKind::KokoroStyle;
        let mut eval = serde_json::json!({
            "wespeaker_heldout": final_score, "best_stock": {"name": best_stock.0, "wespeaker": best_stock.1},
            "generations": fit.generations, "best_step": fit.best_step, "stop": reason, "history": history,
        });
        let mut adapter_map: Option<std::collections::HashMap<String, candle_core::Tensor>> = None;
        if req.tier == "adapter" {
            let candle = self
                .kokoro
                .candle()
                .ok_or_else(|| anyhow!("tier adapter needs the candle model: set KOKORO_CANDLE_WEIGHTS, NV_QWEN_VOICE_ENCODER_DIR, NV_WESPEAKER_CANDLE_DIR"))?;

            let mut refs24: Vec<Vec<f32>> = Vec::new();
            for r in req.refs.iter().take(12) {
                let (x, sr) = read_wav_mono(&self.resolve_ref(r)?)?;
                let x24 = resample_linear(&x, sr, 24_000);
                if x24.len() >= 24_000 {
                    refs24.push(x24);
                }
            }
            let targets = candle.targets(&refs24)?;
            let pad = |t: &[i64]| -> Vec<u32> {
                let mut v = Vec::with_capacity(t.len() + 2);
                v.push(0);
                v.extend(t.iter().map(|x| *x as u32));
                v.push(0);
                v
            };
            let heldout: Vec<Vec<u32>> = tokens.iter().map(|t| pad(t)).collect();
            let train: Vec<Vec<u32>> = train_tokens.iter().map(|t| pad(t)).collect();
            let cfg = TrainConfig {
                epochs: req.epochs.max(1),
                patience: req.patience.max(1),
                min_delta: req.min_delta,
                ..TrainConfig::default()
            };
            let t_ad = Instant::now();
            let out = candle.fit_adapter(&targets, &style, &train, &heldout, &cfg, |rep| {
                emit(FitEvent {
                    stage: "train",
                    step: Some(rep.epoch),
                    wsim: Some(rep.heldout_wespeaker),
                    best: None,
                    message: Some(format!(
                        "adapter epoch {} train loss {:.4} qwen {:.3}",
                        rep.epoch, rep.train_loss, rep.heldout_qwen
                    )),
                    secs: secs(t0),
                });
            })?;
            emit(FitEvent {
                stage: "stop",
                step: Some(out.reports.len()),
                wsim: None,
                best: Some(out.best_heldout),
                message: Some(format!(
                    "adapter: {} after {} epochs ({:.0}s)",
                    out.stop,
                    out.reports.len(),
                    t_ad.elapsed().as_secs_f32()
                )),
                secs: secs(t0),
            });

            let st_path = output.file_path(&name, "adapter.safetensors");
            if let Some(parent) = st_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            candle_core::safetensors::save(&out.adain, &st_path)?;
            let mut entries: Vec<(String, Vec<usize>, Vec<f32>)> =
                Vec::with_capacity(out.adain.len());
            for (k, v) in &out.adain {
                entries.push((
                    k.clone(),
                    v.dims().to_vec(),
                    v.flatten_all()?.to_vec1::<f32>()?,
                ));
            }
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            super::npz::write_npz(&output.file_path(&name, ADAPTER_FILE), &entries)?;
            kind = VoiceKind::KokoroAdapter;
            eval = serde_json::json!({
                "wespeaker_heldout": out.best_heldout, "best_epoch": out.best_epoch, "stop": out.stop,
                "epochs": out.reports.iter().map(|r| serde_json::json!({"epoch": r.epoch, "train_loss": r.train_loss, "heldout_wespeaker": r.heldout_wespeaker, "heldout_qwen": r.heldout_qwen})).collect::<Vec<_>>(),
                "style_tier": {"wespeaker_heldout": final_score, "generations": fit.generations},
                "best_stock": {"name": best_stock.0, "wespeaker": best_stock.1},
            });
            adapter_map = Some(out.adain);
        }
        for (i, t) in tokens.iter().enumerate().take(4) {
            let audio = match (&adapter_map, self.kokoro.candle()) {
                (Some(map), Some(candle)) => {
                    let mut ids: Vec<u32> = vec![0];
                    ids.extend(t.iter().map(|x| *x as u32));
                    ids.push(0);
                    candle.synthesize(&ids, &style, 1.0, Some(map), 11 + i as u64)?
                }
                _ => self.kokoro.synth_tokens_with_style(t, &style, 1.0)?,
            };
            let p = output.file_path(&name, &format!("sample_{i}.wav"));
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent)?;
            }
            write_wav_24k(&p, &audio)?;
        }
        let meta = VoiceMeta {
            schema_version: nv_tts::voice_profile::SCHEMA_VERSION,
            name: name.clone(),
            kind,
            speaker_id: req.speaker_id.clone(),
            embedding: Vec::new(),
            design_params: None,
            producer: Some(serde_json::json!({
                "tool": "speaches-plus/tts::fit", "tier": req.tier, "algo": "es-blend", "population": req.population,
                "lambda": req.lambda, "patience": req.patience, "min_delta": req.min_delta, "seed": req.seed,
                "epochs": req.epochs, "refs": n_refs, "sentences": sentences, "train_sentences": train_sentences,
            })),
            eval: Some(eval),
            created: Some(chrono_like_now()),
        };
        output.put_meta(&meta)?;
        publish_voice(&output.dir_for(&name), &self.store.root().join(&name))?;
        emit(FitEvent {
            stage: "result",
            step: None,
            wsim: Some(final_score),
            best: Some(fit.validation),
            message: Some(name),
            secs: secs(t0),
        });
        Ok(meta)
    }
}

fn publish_voice(staged: &Path, destination: &Path) -> Result<()> {
    if !staged.join(nv_tts::META_FILE).is_file() {
        return Err(anyhow!("staged voice has no metadata"));
    }
    let revision = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    let versions = destination.join(".versions");
    std::fs::create_dir_all(&versions)?;
    let generation = versions.join(&revision);
    std::fs::rename(staged, &generation)?;
    let pointer = destination.join(format!(".current-{revision}"));
    std::os::unix::fs::symlink(Path::new(".versions").join(revision), &pointer)?;
    std::fs::rename(&pointer, destination.join("current"))?;
    Ok(())
}

fn training_sentences(heldout: &[String], corpus: &[&str]) -> Result<Vec<String>> {
    let normalize = |s: &str| {
        s.to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    let excluded: Vec<String> = heldout.iter().map(|s| normalize(s)).collect();
    let train: Vec<String> = corpus
        .iter()
        .filter(|s| !excluded.contains(&normalize(s)))
        .map(|s| s.to_string())
        .collect();
    if train.is_empty() {
        return Err(anyhow!("held-out sentences exhaust the training corpus"));
    }
    Ok(train)
}

fn chrono_like_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    let (h, m, s) = ((secs % 86_400) / 3600, (secs % 3600) / 60, secs % 60);

    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use nv_tts::kokoro::style_fit::EarlyStopping;
    use nv_tts::kokoro::Rng;

    #[test]
    fn voices_fit_form_needs_no_refs_defaults_to_english_and_es_selects_the_spanish_corpus() {
        let req: FitRequest = serde_json::from_value(serde_json::json!({
            "speaker": "spk-0001", "name": "example", "tier": "adapter", "steps": 2
        }))
        .unwrap();
        assert!(req.refs.is_empty());
        assert_eq!(req.speaker.as_deref(), Some("spk-0001"));
        {
            let req: FitRequest =
                serde_json::from_value(serde_json::json!({"name": "v", "refs": ["a.wav"]})).unwrap();
            assert_eq!(req.language, "en-us");
            let (train, heldout) = corpus_for("es");
            assert_eq!(train.len(), TRAIN_SENTENCES_ES.len());
            assert_eq!(heldout, &DEFAULT_SENTENCES_ES);
            assert!(train.iter().all(|s| !TRAIN_SENTENCES.contains(s)));
            assert_eq!(corpus_for("en-gb").0, &TRAIN_SENTENCES);
            assert_eq!(super::super::http::espeak_voice_for("es-AR"), Ok("es-419"));
            assert!(super::super::http::espeak_voice_for("pt-BR").is_err());
        }
    }

    #[test]
    fn voices_training_excludes_requested_sentences_even_with_four_or_more() {
        let heldout: Vec<String> = TRAIN_SENTENCES[..4].iter().map(|s| s.to_string()).collect();
        let training = training_sentences(&heldout, &TRAIN_SENTENCES).unwrap();
        assert_eq!(training.len(), 4);
        assert!(training.iter().all(|s| !heldout.contains(s)));
        assert!(training_sentences(
            &TRAIN_SENTENCES
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            &TRAIN_SENTENCES
        )
        .is_err());
    }

    #[test]
    fn voices_failed_and_successful_publication_preserve_complete_generations() {
        let td = crate::voices::stages::TempDir::new("publish").unwrap();
        let live = td.0.join("live");
        std::fs::create_dir(&live).unwrap();
        std::fs::write(live.join("pack.npy"), b"old pack").unwrap();
        std::fs::write(live.join("voice.json"), b"old meta").unwrap();
        assert!(publish_voice(&td.0.join("missing"), &live).is_err());
        assert_eq!(std::fs::read(live.join("pack.npy")).unwrap(), b"old pack");
        let staged = td.0.join("staged");
        std::fs::create_dir(&staged).unwrap();
        std::fs::write(staged.join("pack.npy"), b"new pack").unwrap();
        std::fs::write(staged.join("voice.json"), b"new meta").unwrap();
        publish_voice(&staged, &live).unwrap();
        let store = VoiceProfileStore::open(&td.0).unwrap();
        assert_eq!(
            std::fs::read(store.file_path("live", "pack.npy")).unwrap(),
            b"new pack"
        );
        assert_eq!(std::fs::read(store.path_for("live")).unwrap(), b"new meta");
        assert_eq!(std::fs::read(live.join("pack.npy")).unwrap(), b"old pack");
        let first_generation = store.dir_for("live");
        let staged = td.0.join("second");
        std::fs::create_dir(&staged).unwrap();
        std::fs::write(staged.join("voice.json"), b"second meta").unwrap();
        std::fs::write(staged.join("pack.npy"), b"second pack").unwrap();
        publish_voice(&staged, &live).unwrap();
        assert_eq!(
            std::fs::read(store.file_path("live", "pack.npy")).unwrap(),
            b"second pack"
        );
        assert_eq!(
            std::fs::read(first_generation.join("pack.npy")).unwrap(),
            b"new pack"
        );
        assert_eq!(store.list_all().unwrap(), vec!["live"]);
        store.delete("live").unwrap();
        assert!(!live.exists());
        assert!(store.list_all().unwrap().is_empty());
    }

    #[test]
    fn early_stopping_matches_the_python_policy() {
        let mut es = EarlyStopping::new(3, 0.005, 100);
        let stops: Vec<bool> = [0.30, 0.40, 0.50, 0.503, 0.504, 0.501]
            .iter()
            .map(|s| es.update(*s))
            .collect();
        assert_eq!(stops, vec![false, false, false, false, false, true]);
        assert!((es.best - 0.5).abs() < 1e-6);
        assert_eq!(es.best_step, 2);
        let mut es = EarlyStopping::new(50, 0.0, 4);
        assert_eq!(
            [0.1, 0.2, 0.3, 0.4]
                .iter()
                .map(|s| es.update(*s))
                .collect::<Vec<_>>(),
            vec![false, false, false, true]
        );
    }

    #[test]
    fn npy_round_trips_resample_keeps_duration_rng_is_deterministic_and_timestamps_are_iso() {
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
        let bytes = npy_f32(&[1, 1, 256], &data);
        let voice = super::super::npz::parse_npy(&bytes).unwrap();
        assert_eq!(voice.shape, vec![1, 1, 256]);
        assert_eq!(voice.data, data);
        assert_eq!(
            (bytes.len() - data.len() * 4) % 64,
            0,
            "header padded to 64 bytes"
        );
        {
            let x: Vec<f32> = (0..24_000).map(|i| (i as f32 * 0.01).sin()).collect();
            let y = resample_linear(&x, 24_000, 16_000);
            assert_eq!(y.len(), 16_000);
            let mut a = Rng(7);
            let mut b = Rng(7);
            assert_eq!(a.normal(), b.normal());
            let m: f32 = (0..20_000).map(|_| a.normal()).sum::<f32>() / 20_000.0;
            assert!(m.abs() < 0.05, "mean {m}");
        }
        {
            let t = chrono_like_now();
            assert_eq!(t.len(), 20);
            assert!(t.starts_with("20") && t.ends_with('Z'), "{t}");
        }
    }

}
