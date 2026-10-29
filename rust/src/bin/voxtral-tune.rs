use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::Tensor;
use clap::{Parser, Subcommand};
use nv_models::voxtral::{DraftHead, VoxtralRealtime};
use nv_models::voxtral_train::{
    target_stream_for, train_draft_head, train_lora, word_error_rate, StreamRow, VoxtralLora,
};
use nv_train::LoraConfig;
use speaches_plus::audio::decode_any::decode_any_to_16k_mono;
use speaches_plus::stt::voxtral::{model_dir, serving_device};

#[derive(Parser)]
#[command(
    name = "voxtral-tune",
    about = "Own-voice LoRA adaptation for Voxtral Realtime"
)]
struct Cli {
    #[arg(long, env = "VOXTRAL_MODEL_DIR")]
    model_dir: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Prepare {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Train {
        #[arg(long)]
        rows: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 16)]
        rank: usize,
        #[arg(long)]
        alpha: Option<f64>,
        #[arg(long, default_value_t = 1e-4)]
        lr: f64,
        #[arg(long, default_value_t = 3)]
        epochs: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    Eval {
        #[arg(long)]
        rows: PathBuf,
        #[arg(long)]
        adapter: Option<PathBuf>,
        #[arg(long)]
        draft_head: Option<PathBuf>,
    },
    DraftHead {
        #[arg(long)]
        rows: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        #[arg(long, default_value_t = 300)]
        steps: usize,
        #[arg(long, default_value_t = 1e-2)]
        lr: f64,
        #[arg(long)]
        hold_out_odd: bool,
    },
    Transcribe {
        file: PathBuf,
        #[arg(long)]
        adapter: Option<PathBuf>,
    },
}

#[derive(serde::Deserialize)]
struct ClipRow {
    audio: String,
    text: String,
}

fn load_model(dir: Option<PathBuf>, adapter: Option<&PathBuf>) -> Result<VoxtralRealtime> {
    let dir = model_dir(&PathBuf::from("models"), dir)?;
    VoxtralRealtime::load_dir_with_adapter(&dir, &serving_device(), adapter.map(|p| p.as_path()))
}

fn audio_of(path: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {path}"))?;
    decode_any_to_16k_mono(&bytes, None).with_context(|| format!("decode {path}"))
}

fn read_jsonl<T: serde::de::DeserializeOwned>(path: &PathBuf) -> Result<Vec<T>> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<T>(l).with_context(|| format!("parse row: {l}")))
        .collect()
}

fn embeds_of(model: &VoxtralRealtime, audio: &[f32]) -> Result<Tensor> {
    model.encode(&model.features(audio)?)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Prepare { data, out } => {
            let model = load_model(cli.model_dir, None)?;
            let clips: Vec<ClipRow> = read_jsonl(&data)?;
            let mut lines = Vec::with_capacity(clips.len());
            let mut skipped = 0;
            for clip in &clips {
                let audio = audio_of(&clip.audio)?;
                let (hypothesis, generated) = model.transcribe(&audio)?;
                match target_stream_for(&model, &generated, &clip.text) {
                    Ok(target) => {
                        let mut stream = model.prefix_tokens();
                        stream.extend(target);
                        let row = StreamRow {
                            audio: clip.audio.clone(),
                            text: clip.text.clone(),
                            hypothesis: hypothesis.trim().to_string(),
                            stream,
                        };
                        lines.push(serde_json::to_string(&row)?);
                    }
                    Err(e) => {
                        skipped += 1;
                        eprintln!("skip {}: {e:#}", clip.audio);
                    }
                }
            }
            std::fs::write(&out, lines.join("\n") + "\n")?;
            println!(
                "rows {} skipped {skipped} -> {}",
                lines.len(),
                out.display()
            );
        }
        Cmd::Train {
            rows,
            out,
            rank,
            alpha,
            lr,
            epochs,
            seed,
        } => {
            let model = load_model(cli.model_dir.clone(), None)?;
            let rows: Vec<StreamRow> = read_jsonl(&rows)?;
            let mut prepared = Vec::with_capacity(rows.len());
            for r in &rows {
                prepared.push((embeds_of(&model, &audio_of(&r.audio)?)?, r.stream.clone()));
            }
            let cfg = LoraConfig {
                r: rank,
                alpha: alpha.unwrap_or(rank as f64),
                dropout: 0.0,
            };
            let lora = VoxtralLora::attach(&model, cfg, seed)?;
            let report = train_lora(&model, &lora, &prepared, epochs, lr)?;
            lora.save_peft(&out, speaches_plus::stt::voxtral::VOXTRAL_MODEL_ID)?;
            VoxtralLora::detach(&model);
            if let (Some(f), Some(l)) = (report.losses.first(), report.losses.last()) {
                println!("loss first {f:.4} last {l:.4} -> {}", out.display());
            }
        }
        Cmd::Eval {
            rows,
            adapter,
            draft_head,
        } => {
            let model = load_model(cli.model_dir, adapter.as_ref())?;
            if let Some(h) = &draft_head {
                model.set_draft_head(Some(DraftHead::load(h, model.device())?));
            }
            let rows: Vec<StreamRow> = read_jsonl(&rows)?;
            let (mut errors, mut words) = (0usize, 0usize);
            let audio: Vec<Vec<f32>> = rows.iter().map(|r| audio_of(&r.audio)).collect::<Result<_>>()?;
            let _ = model.transcribe(&audio[0])?;
            let before = model.graph_rows_steps_committed();
            let t0 = std::time::Instant::now();
            for (r, a) in rows.iter().zip(&audio) {
                let (text, _) = model.transcribe(a)?;
                let (e, w) = word_error_rate(&text, &r.text);
                errors += e;
                words += w;
                println!("{e}/{w}\t{}\t{}", r.audio, text.trim());
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!(
                "WER {:.2}% ({errors}/{words}) adapter={} draft_head={}",
                100.0 * errors as f64 / words.max(1) as f64,
                adapter
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "none".into()),
                draft_head
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "none".into())
            );
            if let (Some((rows_m, s0, c0)), Some((_, s1, c1))) = (before, model.graph_rows_steps_committed()) {
                println!(
                    "graph rows={rows_m}: {} steps, {:.2} committed/step, {:.0} ms for {} clips ({:.1} ms/clip)",
                    s1 - s0,
                    (c1 - c0) as f64 / (s1 - s0).max(1) as f64,
                    ms,
                    rows.len(),
                    ms / rows.len().max(1) as f64
                );
            }
        }
        Cmd::DraftHead {
            rows,
            out,
            top_k,
            steps,
            lr,
            hold_out_odd,
        } => {
            let model = load_model(cli.model_dir, None)?;
            let rows: Vec<StreamRow> = read_jsonl(&rows)?;
            let mut train = Vec::new();
            let mut held = Vec::new();
            for (i, r) in rows.iter().enumerate() {
                let audio = audio_of(&r.audio)?;
                let embeds = embeds_of(&model, &audio)?;
                let own = model.decode_stream(&embeds)?;
                if hold_out_odd && i % 2 == 1 {
                    held.push((embeds, own));
                } else {
                    train.push((embeds, own));
                }
            }
            let (head, report) = train_draft_head(&model, &train, top_k, steps, lr)?;
            head.save(&out)?;
            println!(
                "classes {} rows {} loss first {:.4} last {:.4} train acc {:.3} -> {}",
                report.classes.len(),
                train.len(),
                report.losses.first().copied().unwrap_or(f32::NAN),
                report.losses.last().copied().unwrap_or(f32::NAN),
                report.train_accuracy,
                out.display()
            );
            if !held.is_empty() {
                let (mut hits, mut total) = (0usize, 0usize);
                for (embeds, own) in &held {
                    let x = nv_models::voxtral_train::draft_features(&model, embeds, own.len())?;
                    let pred = head.predict_classes(&x.unsqueeze(0)?)?;
                    let targets = nv_models::voxtral_train::draft_targets(&report.classes, &own[..x.dims()[0]]);
                    hits += pred.iter().zip(&targets).filter(|(p, t)| p == t).count();
                    total += targets.len();
                }
                println!("held-out odd rows {}: acc {:.3}", held.len(), hits as f32 / total.max(1) as f32);
            }
        }
        Cmd::Transcribe { file, adapter } => {
            let model = load_model(cli.model_dir, adapter.as_ref())?;
            let (text, tokens) = model.transcribe(&audio_of(&file.display().to_string())?)?;
            println!("{}", text.trim());
            eprintln!("{} steps", tokens.len());
        }
    }
    Ok(())
}
