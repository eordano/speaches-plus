mod capture;
mod export;
mod name;
mod records;
mod score;
mod stripes;
mod train;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use records::{Cutter, Record};

pub const HOLDOUT_SESSION_SHARE: u64 = 15;

#[derive(Parser)]
#[command(
    name = "nv-sae",
    about = "Sparse autoencoders over middle-layer residual states of agent logs, one dictionary per capture model and layer"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Cut {
        #[arg(long)]
        root: Vec<PathBuf>,
        #[arg(long, default_value = "claude")]
        agent: String,
        #[arg(long)]
        host: String,
        #[arg(long, default_value_t = 6)]
        depth: usize,
        #[arg(long)]
        no_tools: bool,
        #[arg(long)]
        out: PathBuf,
    },
    Capture {
        #[arg(long)]
        records: PathBuf,
        #[arg(long, default_value = "http://127.0.0.1:1327")]
        url: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value_t = 19)]
        layer: usize,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long)]
        tokens: bool,
        #[arg(long)]
        exclude: Option<PathBuf>,
        #[arg(long, default_value_t = capture::CAPTURE_BATCH_RECORDS)]
        batch: usize,
        #[arg(long, default_value_t = 0)]
        pause_ms: u64,
        #[arg(long, default_value_t = 0)]
        max_seconds: u64,
        #[arg(long)]
        out: PathBuf,
    },
    Merge {
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        since: Option<String>,
        dirs: Vec<PathBuf>,
    },
    Train {
        #[arg(long)]
        states: PathBuf,
        #[arg(long, default_value = "means")]
        key: String,
        #[arg(long)]
        records: PathBuf,
        #[arg(long)]
        spans: Option<PathBuf>,
        #[arg(long, default_value_t = 256)]
        width: usize,
        #[arg(long, default_value_t = 4)]
        top_k: usize,
        #[arg(long, default_value_t = 6000)]
        steps: usize,
        #[arg(long, default_value_t = 256)]
        batch: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value_t = 0.0)]
        l1: f64,
        #[arg(long, default_value_t = 17)]
        seed: u64,
        #[arg(long)]
        dense_baseline: bool,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        layer: Option<usize>,
        #[arg(long)]
        out: PathBuf,
    },
    Score {
        #[arg(long)]
        sae: PathBuf,
        #[arg(long)]
        states: PathBuf,
        #[arg(long, default_value = "means")]
        key: String,
        #[arg(long)]
        out: PathBuf,
    },
    Name {
        #[arg(long)]
        sae: PathBuf,
        #[arg(long)]
        scored: PathBuf,
        #[arg(long)]
        records: PathBuf,
        #[arg(long, default_value = "http://127.0.0.1:1327")]
        url: String,
        #[arg(long, default_value = "poolside/Laguna-XS-2.1-NVFP4")]
        model: String,
        #[arg(long)]
        only: Option<String>,
        #[arg(long)]
        spans: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    Dictionary {
        #[arg(long)]
        records: PathBuf,
        #[arg(long, default_value = "http://127.0.0.1:1327")]
        url: String,
        #[arg(long)]
        name_url: Option<String>,
        #[arg(long, default_value = "poolside/Laguna-XS-2.1-NVFP4")]
        model: String,
        #[arg(long, default_value_t = 19)]
        layer: usize,
        #[arg(long)]
        tokens: bool,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value_t = 256)]
        width: usize,
        #[arg(long, default_value_t = 4)]
        top_k: usize,
        #[arg(long, default_value_t = 6000)]
        steps: usize,
        #[arg(long, default_value_t = 256)]
        batch: usize,
        #[arg(long, default_value_t = 17)]
        seed: u64,
        #[arg(long)]
        skip_names: bool,
        #[arg(long)]
        stream: bool,
        #[arg(long)]
        title: Option<String>,
        #[arg(long)]
        out: PathBuf,
    },
    Correlate {
        #[arg(long)]
        sae: PathBuf,
        #[arg(long)]
        scored: PathBuf,
        #[arg(long)]
        records: PathBuf,
        #[arg(long)]
        feedback: PathBuf,
        #[arg(long)]
        names: Option<PathBuf>,
        #[arg(long, default_value_t = 3)]
        min_votes: usize,
    },
    Export {
        #[arg(long)]
        sae: PathBuf,
        #[arg(long)]
        scored: PathBuf,
        #[arg(long)]
        records: PathBuf,
        #[arg(long)]
        names: Option<PathBuf>,
        #[arg(long)]
        spans: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value = "Session reader")]
        title: String,
        #[arg(long)]
        out: PathBuf,
    },
}

fn read_records(path: &PathBuf) -> Result<Vec<Record>> {
    let raw = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).context("record line"))
        .collect()
}

fn read_ids(path: &PathBuf) -> Result<std::collections::HashSet<String>> {
    #[derive(serde::Deserialize)]
    struct IdOnly {
        id: String,
    }
    let f = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    let mut ids = std::collections::HashSet::new();
    for line in std::io::BufRead::lines(std::io::BufReader::new(f)) {
        let line = line?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t.starts_with('{') {
            let r: IdOnly = serde_json::from_str(t).context("exclude line")?;
            ids.insert(r.id);
        } else {
            ids.insert(t.to_string());
        }
    }
    Ok(ids)
}

fn write_records(path: &PathBuf, records: &[Record]) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    for r in records {
        serde_json::to_writer(&mut f, r)?;
        std::io::Write::write_all(&mut f, b"\n")?;
    }
    Ok(())
}

fn identity(
    states: &std::path::Path,
    model: Option<String>,
    layer: Option<usize>,
) -> Result<(String, usize)> {
    let manifest = states.parent().and_then(|d| capture::read_manifest(d).ok());
    let model = model
        .or_else(|| manifest.as_ref().map(|m| m.model.clone()))
        .filter(|m| !m.is_empty());
    let layer = layer.or_else(|| manifest.as_ref().map(|m| m.layer));
    match (model, layer) {
        (Some(m), Some(l)) => Ok((m, l)),
        _ => anyhow::bail!(
            "{}: no capture.json beside the states and no --model/--layer; a dictionary must name the capture model and layer it belongs to",
            states.display()
        ),
    }
}

fn holdout_mask(records: &[Record]) -> Vec<bool> {
    use sha2::Digest;
    records
        .iter()
        .map(|r| {
            let h = sha2::Sha256::digest(format!("{}:{}", r.agent, r.session).as_bytes());
            u64::from_le_bytes(h[..8].try_into().unwrap()) % 100 >= HOLDOUT_SESSION_SHARE
        })
        .collect()
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Cut {
            root,
            agent,
            host,
            depth,
            no_tools,
            out,
        } => {
            let cutter = Cutter {
                host,
                include_tools: !no_tools,
            };
            let mut all = Vec::new();
            for r in &root {
                for f in records::walk_jsonl(r, depth) {
                    match cutter.cut_file(&f, &agent) {
                        Ok(mut rs) => all.append(&mut rs),
                        Err(e) => eprintln!("skip {}: {e:#}", f.display()),
                    }
                }
            }
            let all = records::dedupe(all);
            let counts = all.iter().fold([0usize; 3], |mut c, r| {
                c[r.kind as usize] += 1;
                c
            });
            write_records(&out, &all)?;
            println!(
                "{}",
                serde_json::json!({"records": all.len(), "user": counts[0], "assistant": counts[1], "tool": counts[2], "out": out})
            );
        }
        Cmd::Capture {
            records,
            url,
            model,
            layer,
            tokenizer,
            tokens,
            exclude,
            batch,
            pause_ms,
            max_seconds,
            out,
        } => {
            let mut recs = read_records(&records)?;
            if let Some(p) = exclude.filter(|p| p.exists()) {
                let done = read_ids(&p)?;
                recs.retain(|r| !done.contains(&r.id));
            }
            if recs.is_empty() {
                std::fs::create_dir_all(&out)?;
                std::fs::write(out.join("records.jsonl"), "")?;
                println!(
                    "{}",
                    serde_json::json!({"records": 0, "skipped": "nothing new"})
                );
                return Ok(());
            }
            let tok = match tokenizer {
                Some(p) => Some(
                    tokenizers::Tokenizer::from_file(&p)
                        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?,
                ),
                None => None,
            };
            let client = capture::Client::new(&url, model.as_deref())?;
            let pace = capture::Pace {
                batch: if tokens {
                    batch.min(capture::CAPTURE_BATCH_RECORDS_WITH_TOKENS)
                } else {
                    batch
                },
                pause: std::time::Duration::from_millis(pause_ms),
                deadline: (max_seconds > 0).then(|| {
                    std::time::Instant::now() + std::time::Duration::from_secs(max_seconds)
                }),
            };
            let m =
                capture::capture_records(&client, &recs, layer, &out, tok.as_ref(), tokens, &pace)?;
            println!("{}", serde_json::to_string(&m)?);
        }
        Cmd::Merge { out, since, dirs } => {
            let n = capture::merge(&dirs, &out, since.as_deref())?;
            println!("{}", serde_json::json!({"records": n, "out": out}));
        }
        Cmd::Train {
            states,
            key,
            records,
            spans,
            width,
            top_k,
            steps,
            batch,
            lr,
            l1,
            seed,
            dense_baseline,
            model,
            layer,
            out,
        } => {
            std::fs::create_dir_all(&out)?;
            let dev = train::device();
            let x = score::load_matrix(&states, &key)?;
            let identity = identity(&states, model, layer)?;
            let recs = read_records(&records)?;
            let record_mask = holdout_mask(&recs);
            let mask = match spans {
                Some(p) => {
                    let sp = capture::load_spans(&p)?;
                    let mut m = vec![true; x.dims()[0]];
                    for s in &sp {
                        for t in s.start..s.end {
                            m[t] = record_mask[s.record];
                        }
                    }
                    m
                }
                None => {
                    anyhow::ensure!(
                        recs.len() == x.dims()[0],
                        "records ({}) and states ({}) disagree",
                        recs.len(),
                        x.dims()[0]
                    );
                    record_mask
                }
            };
            let cfg = train::TrainConfig {
                width,
                top_k,
                steps,
                batch,
                lr,
                l1,
                seed,
            };
            let (sae, mut report) = train::train(&x, &mask, &cfg, &dev)?;
            sae.save(&out.join("sae.safetensors"))?;
            let mut result = serde_json::to_value(&report)?;
            if dense_baseline {
                let dense_cfg = train::TrainConfig {
                    top_k: width,
                    ..cfg.clone()
                };
                let (_, dense) = train::train(&x, &mask, &dense_cfg, &dev)?;
                result["validation_dense_baseline_mse"] = serde_json::json!(dense.mse_validation);
            }
            report.progress.clear();
            result["encoder_sha256"] =
                serde_json::json!(name::sha256_file(&out.join("sae.safetensors"))?);
            result["model"] = serde_json::json!(identity.0);
            result["layer"] = serde_json::json!(identity.1);
            result["hidden_size"] = serde_json::json!(x.dims()[1]);
            std::fs::write(
                out.join("result.json"),
                serde_json::to_string_pretty(&result)?,
            )?;
            let scored = score::score(&sae, &x, 1024)?;
            score::save_scored(&out.join("scored.safetensors"), &scored)?;
            println!(
                "{}",
                serde_json::json!({"mse_train": report.mse_train, "mse_validation": report.mse_validation, "mean_baseline": report.validation_mean_baseline_mse, "dense_baseline": result.get("validation_dense_baseline_mse"), "dead": report.dead_features, "seconds": report.seconds})
            );
        }
        Cmd::Score {
            sae,
            states,
            key,
            out,
        } => {
            let dev = train::device();
            let sae = train::Sae::load(&sae, &dev)?;
            let x = score::load_matrix(&states, &key)?;
            anyhow::ensure!(
                x.dims()[1] == sae.mu.dims()[0],
                "dictionary expects hidden size {} but {} holds {}; score only with the dictionary of the capture model and layer",
                sae.mu.dims()[0],
                states.display(),
                x.dims()[1]
            );
            let scored = score::score(&sae, &x, 1024)?;
            score::save_scored(&out, &scored)?;
            println!(
                "{}",
                serde_json::json!({"rows": scored.indices.len(), "width": scored.width, "top_k": scored.top_k})
            );
        }
        Cmd::Name {
            sae,
            scored,
            records,
            url,
            model,
            only,
            spans,
            tokenizer,
            out,
        } => {
            let dev = candle_core::Device::Cpu;
            let s = train::Sae::load(&sae, &dev)?;
            let sc = score::load_scored(&scored, s.width())?;
            let recs = read_records(&records)?;
            let feats: Vec<usize> = match only {
                Some(list) => list
                    .split(',')
                    .map(|x| x.trim().parse())
                    .collect::<Result<_, _>>()?,
                None => (0..s.width()).collect(),
            };
            let namer = name::Namer::new(&url, &model)?;
            let named = match (spans, tokenizer) {
                (Some(sp), Some(tk)) => {
                    let tok = tokenizers::Tokenizer::from_file(&tk)
                        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
                    namer.name_tokens(&sc, &capture::load_spans(&sp)?, &tok, &feats)
                }
                _ => {
                    anyhow::ensure!(
                        recs.len() == sc.indices.len(),
                        "records and scored rows disagree"
                    );
                    namer.name_records(&sc, &recs, &feats)
                }
            };
            let names = name::Names {
                model,
                encoder_sha256: name::sha256_file(&sae)?,
                names: named,
            };
            std::fs::write(&out, serde_json::to_string_pretty(&names)?)?;
            println!(
                "{}",
                serde_json::json!({"named": names.names.iter().filter(|n| n.name.is_some()).count(), "of": names.names.len()})
            );
        }
        Cmd::Dictionary {
            records,
            url,
            name_url,
            model,
            layer,
            tokens,
            tokenizer,
            width,
            top_k,
            steps,
            batch,
            seed,
            skip_names,
            stream,
            title,
            out,
        } => {
            anyhow::ensure!(
                !tokens || tokenizer.is_some(),
                "--tokens needs --tokenizer for naming and export"
            );
            std::fs::create_dir_all(&out)?;
            let recs = read_records(&records)?;
            let tok = match &tokenizer {
                Some(p) => Some(
                    tokenizers::Tokenizer::from_file(p)
                        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?,
                ),
                None => None,
            };
            let cap = out.join("capture");
            if stream && tokens {
                let client = capture::Client::new(&url, Some(&model))?;
                let dev = train::device();
                let cfg = train::TrainConfig {
                    width,
                    top_k,
                    steps,
                    batch,
                    lr: 1e-3,
                    l1: 0.0,
                    seed,
                };
                let pass_batch = capture::CAPTURE_BATCH_RECORDS_WITH_TOKENS;
                let train_recs: Vec<Record> = recs
                    .iter()
                    .zip(holdout_mask(&recs))
                    .filter(|(_, t)| *t)
                    .map(|(r, _)| r.clone())
                    .collect();
                let mut mean: Vec<f64> = Vec::new();
                let mut m2: Vec<f64> = Vec::new();
                let mut n = 0f64;
                let moments_records: Vec<Record> = train_recs.iter().take(400).cloned().collect();
                capture::stream_batches(
                    &client,
                    &moments_records,
                    layer,
                    tok.as_ref(),
                    pass_batch,
                    |_, resp| {
                        mean.resize(resp.hidden_size, 0.0);
                        m2.resize(resp.hidden_size, 0.0);
                        for r in resp.records {
                            train::running_moments(
                                r.token_states.as_deref().unwrap_or(&[]),
                                &mut mean,
                                &mut m2,
                                &mut n,
                            );
                        }
                        Ok(())
                    },
                )?;
                let mu: Vec<f32> = mean.iter().map(|x| *x as f32).collect();
                let sd: Vec<f32> = m2
                    .iter()
                    .map(|x| ((x / n.max(1.0)).sqrt() as f32) + train::NORMALIZATION_EPS)
                    .collect();
                let mut trainer = train::StreamTrainer::new(&cfg, &mu, &sd, &dev)?;
                eprintln!("stream: normalization from {} records ({} tokens); training {steps} steps while capturing", moments_records.len(), n as usize);
                let mut passes = 0;
                while trainer.steps_done < steps {
                    passes += 1;
                    let target = steps;
                    capture::stream_batches(
                        &client,
                        &train_recs,
                        layer,
                        tok.as_ref(),
                        pass_batch,
                        |_, resp| {
                            for r in resp.records {
                                trainer.push(r.token_states.as_deref().unwrap_or(&[]));
                            }
                            if trainer.steps_done < target {
                                trainer.train_steps(
                                    train::STREAM_STEPS_PER_REFILL.min(target - trainer.steps_done),
                                )?;
                            }
                            Ok(())
                        },
                    )?;
                    eprintln!(
                        "stream: pass {passes} done, {} steps, last loss {:?}",
                        trainer.steps_done,
                        trainer.losses.last()
                    );
                    if passes >= 8 {
                        break;
                    }
                }
                trainer.sae.save(&out.join("sae.safetensors"))?;
                let (steps_done, losses) = (trainer.steps_done, trainer.losses.clone());
                let sae = trainer.sae;
                std::fs::create_dir_all(&cap)?;
                let mut indices: Vec<Vec<u32>> = Vec::new();
                let mut values: Vec<Vec<f32>> = Vec::new();
                let mut spans: Vec<capture::Span> = Vec::new();
                let mut kept: Vec<Record> = Vec::new();
                let mut total = 0usize;
                capture::stream_batches(
                    &client,
                    &recs,
                    layer,
                    tok.as_ref(),
                    pass_batch,
                    |chunk, resp| {
                        for (r, c) in chunk.iter().zip(resp.records) {
                            let states = c.token_states.unwrap_or_default();
                            let x = candle_core::Tensor::new(states.concat().as_slice(), &dev)?
                                .reshape((states.len(), mu.len()))?;
                            let s = score::score(&sae, &x, 1024)?;
                            spans.push(capture::Span {
                                record: kept.len(),
                                start: total,
                                end: total + states.len(),
                                ids: c.input_token_ids,
                            });
                            total += states.len();
                            indices.extend(s.indices);
                            values.extend(s.values);
                            kept.push((*r).clone());
                        }
                        Ok(())
                    },
                )?;
                let dense_sample: Vec<Vec<f32>> = (0..indices.len())
                    .step_by((indices.len() / 60_000).max(1))
                    .map(|i| {
                        let mut v = vec![0f32; width];
                        for (f, x) in indices[i].iter().zip(&values[i]) {
                            v[*f as usize] = *x;
                        }
                        v
                    })
                    .collect();
                let scored = score::Scored {
                    width,
                    top_k,
                    scale: score::percentile_scale(&dense_sample, width),
                    indices,
                    values,
                };
                score::save_scored(&out.join("scored.safetensors"), &scored)?;
                std::fs::write(cap.join("spans.json"), serde_json::to_string(&spans)?)?;
                write_records(&out.join("records.jsonl"), &kept)?;
                std::fs::write(
                    out.join("result.json"),
                    serde_json::to_string_pretty(
                        &serde_json::json!({"config": cfg, "model": model, "layer": layer, "hidden_size": mu.len(), "mode": "stream-tokens", "tokens": total, "records": kept.len(), "steps": steps_done, "losses": losses, "encoder_sha256": name::sha256_file(&out.join("sae.safetensors"))?}),
                    )?,
                )?;
                let mut names = None;
                if !skip_names {
                    let namer = name::Namer::new(name_url.as_deref().unwrap_or(&url), &model)?;
                    let feats: Vec<usize> = (0..width).collect();
                    let n = name::Names {
                        model: model.clone(),
                        encoder_sha256: name::sha256_file(&out.join("sae.safetensors"))?,
                        names: namer.name_tokens(&scored, &spans, tok.as_ref().unwrap(), &feats),
                    };
                    std::fs::write(out.join("names.json"), serde_json::to_string_pretty(&n)?)?;
                    names = Some(n);
                }
                let title = title.unwrap_or_else(|| {
                    format!("layer-{layer} token SAE · {width} features top-{top_k} (streamed)")
                });
                stripes::stripes(
                    &kept,
                    &spans,
                    &scored,
                    names.as_ref(),
                    tok.as_ref().unwrap(),
                    &title,
                    &out.join("stripes.html"),
                )?;
                println!(
                    "{}",
                    serde_json::json!({"out": out, "records": kept.len(), "tokens": total, "named": names.as_ref().map(|n| n.names.iter().filter(|x| x.name.is_some()).count()), "page": out.join("stripes.html")})
                );
                return Ok(());
            }
            if !cap.join("states.safetensors").exists() {
                let m = capture::capture_records(
                    &capture::Client::new(&url, Some(&model))?,
                    &recs,
                    layer,
                    &cap,
                    tok.as_ref(),
                    tokens,
                    &capture::Pace::steady(),
                )?;
                eprintln!(
                    "captured {} records, {} tokens in {:.0}s",
                    m.records, m.tokens, m.seconds
                );
            }
            let recs = read_records(&cap.join("records.jsonl"))?;
            let dev = train::device();
            let (states, key, spans) = if tokens {
                (
                    cap.join("tokens.safetensors"),
                    "tokens",
                    Some(capture::load_spans(&cap.join("spans.json"))?),
                )
            } else {
                (cap.join("states.safetensors"), "means", None)
            };
            let x = score::load_matrix(&states, key)?;
            let record_mask = holdout_mask(&recs);
            let mask = match &spans {
                Some(sp) => {
                    let mut mm = vec![true; x.dims()[0]];
                    for s in sp {
                        for t in s.start..s.end {
                            mm[t] = record_mask[s.record];
                        }
                    }
                    mm
                }
                None => record_mask,
            };
            let cfg = train::TrainConfig {
                width,
                top_k,
                steps,
                batch,
                lr: 1e-3,
                l1: 0.0,
                seed,
            };
            let (sae, report) = train::train(&x, &mask, &cfg, &dev)?;
            sae.save(&out.join("sae.safetensors"))?;
            let scored = score::score(&sae, &x, 1024)?;
            score::save_scored(&out.join("scored.safetensors"), &scored)?;
            std::fs::copy(cap.join("records.jsonl"), out.join("records.jsonl"))?;
            let mut result = serde_json::to_value(&report)?;
            result["encoder_sha256"] =
                serde_json::json!(name::sha256_file(&out.join("sae.safetensors"))?);
            result["layer"] = serde_json::json!(layer);
            result["model"] = serde_json::json!(model);
            result["hidden_size"] = serde_json::json!(x.dims()[1]);
            std::fs::write(
                out.join("result.json"),
                serde_json::to_string_pretty(&result)?,
            )?;
            eprintln!(
                "trained {width}/{top_k}: validation mse {:.3} (mean baseline {:.3}), {} dead",
                report.mse_validation, report.validation_mean_baseline_mse, report.dead_features
            );
            let mut names = None;
            if !skip_names {
                let namer = name::Namer::new(name_url.as_deref().unwrap_or(&url), &model)?;
                let feats: Vec<usize> = (0..width).collect();
                let named = match (&spans, &tok) {
                    (Some(sp), Some(tk)) => namer.name_tokens(&scored, sp, tk, &feats),
                    _ => namer.name_records(&scored, &recs, &feats),
                };
                let n = name::Names {
                    model: model.clone(),
                    encoder_sha256: name::sha256_file(&out.join("sae.safetensors"))?,
                    names: named,
                };
                std::fs::write(out.join("names.json"), serde_json::to_string_pretty(&n)?)?;
                names = Some(n);
            }
            let title = title.unwrap_or_else(|| {
                format!(
                    "layer-{layer} {} SAE · {width} features top-{top_k}",
                    if tokens { "token" } else { "record" }
                )
            });
            let page = out.join(if tokens {
                "stripes.html"
            } else {
                "sessions.html"
            });
            match (&spans, &tok) {
                (Some(sp), Some(tk)) => {
                    stripes::stripes(&recs, sp, &scored, names.as_ref(), tk, &title, &page)?
                }
                _ => export::session_reader(&recs, &scored, names.as_ref(), &title, &page)?,
            }
            println!(
                "{}",
                serde_json::json!({"out": out, "records": recs.len(), "mse_validation": report.mse_validation, "mean_baseline": report.validation_mean_baseline_mse, "named": names.as_ref().map(|n| n.names.iter().filter(|x| x.name.is_some()).count()), "page": page})
            );
        }
        Cmd::Correlate {
            sae,
            scored,
            records,
            feedback,
            names,
            min_votes,
        } => {
            let dev = candle_core::Device::Cpu;
            let s = train::Sae::load(&sae, &dev)?;
            let sc = score::load_scored(&scored, s.width())?;
            let recs = read_records(&records)?;
            anyhow::ensure!(
                recs.len() == sc.indices.len(),
                "records and scored rows disagree"
            );
            let names = match names {
                Some(p) => Some(serde_json::from_str::<name::Names>(
                    &std::fs::read_to_string(p)?,
                )?),
                None => None,
            };
            let mut votes: std::collections::HashMap<String, i32> =
                std::collections::HashMap::new();
            for line in std::fs::read_to_string(&feedback)?
                .lines()
                .filter(|l| !l.trim().is_empty())
            {
                let v: serde_json::Value = serde_json::from_str(line)?;
                if let (Some(id), Some(vote)) = (
                    v.get("record_id").and_then(|x| x.as_str()),
                    v.get("vote").and_then(|x| x.as_i64()),
                ) {
                    votes.insert(id.to_string(), vote as i32);
                }
            }
            let (mut up, mut down) = (0usize, 0usize);
            let mut per: Vec<(usize, usize)> = vec![(0, 0); s.width()];
            for (i, r) in recs.iter().enumerate() {
                let Some(v) = votes.get(&r.id) else { continue };
                if *v > 0 {
                    up += 1
                } else if *v < 0 {
                    down += 1
                } else {
                    continue;
                }
                for f in &sc.indices[i] {
                    let e = &mut per[*f as usize];
                    if *v > 0 {
                        e.0 += 1
                    } else {
                        e.1 += 1
                    }
                }
            }
            anyhow::ensure!(up + down > 0, "no voted records among these records");
            let base = up as f64 / (up + down) as f64;
            let mut rows: Vec<(usize, usize, usize, f64)> = per
                .iter()
                .enumerate()
                .filter(|(_, (u, d))| u + d >= min_votes)
                .map(|(f, (u, d))| (f, *u, *d, (*u as f64 + 0.5) / ((u + d) as f64 + 1.0)))
                .collect();
            rows.sort_by(|a, b| (b.3 - base).abs().partial_cmp(&(a.3 - base).abs()).unwrap());
            println!("voted records: {up} up, {down} down (base rate {:.2}); features with at least {min_votes} votes: {}", base, rows.len());
            println!(
                "{:>4} {:>4} {:>4} {:>6} {:>6}  name",
                "id", "up", "down", "p(up)", "lift"
            );
            for (f, u, d, p) in &rows {
                let nm = names
                    .as_ref()
                    .and_then(|n| n.names.iter().find(|x| x.feature == *f))
                    .and_then(|x| x.name.clone())
                    .unwrap_or_default();
                println!("{f:>4} {u:>4} {d:>4} {p:>6.2} {:>+6.2}  {nm}", p - base);
            }
        }
        Cmd::Export {
            sae,
            scored,
            records,
            names,
            spans,
            tokenizer,
            title,
            out,
        } => {
            let dev = candle_core::Device::Cpu;
            let s = train::Sae::load(&sae, &dev)?;
            let sc = score::load_scored(&scored, s.width())?;
            let recs = read_records(&records)?;
            let names = match names {
                Some(p) => Some(serde_json::from_str::<name::Names>(
                    &std::fs::read_to_string(p)?,
                )?),
                None => None,
            };
            match (spans, tokenizer) {
                (Some(sp), Some(tk)) => {
                    let tok = tokenizers::Tokenizer::from_file(&tk)
                        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
                    stripes::stripes(
                        &recs,
                        &capture::load_spans(&sp)?,
                        &sc,
                        names.as_ref(),
                        &tok,
                        &title,
                        &out,
                    )?;
                }
                _ => export::session_reader(&recs, &sc, names.as_ref(), &title, &out)?,
            }
            println!(
                "{}",
                serde_json::json!({"out": out, "bytes": std::fs::metadata(&out)?.len()})
            );
        }
    }
    Ok(())
}
