use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::Device;
use nv_models::deepseek_ocr::{
    DecoderPrecision, DeepSeekOcr2Pipeline, GenerateOptions, ResolutionMode, PROMPT_FREE_OCR,
    PROMPT_GROUNDING_MARKDOWN,
};
use speaches_plus::oapi::ocr_batch::{decode_rgb, BatchOptions, DsocrScheduler, JobInput, OcrJob};

fn default_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_DSOCR_DIR") {
        let p = PathBuf::from(d);
        return p.is_dir().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR-2/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

struct Args {
    pages: Vec<PathBuf>,
    concurrency: Vec<usize>,
    repeat: usize,
    max_new: usize,
    markdown: bool,
    stages: bool,
    vision_scan: bool,
    micro: bool,
    micro_decode: Option<usize>,
    #[allow(dead_code)]
    overlap: bool,
    gate: bool,
    baseline: bool,
    json: Option<PathBuf>,
    dump: Option<PathBuf>,
    prep_threads: Option<usize>,
}

fn parse_args() -> Result<Args> {
    let mut pages = Vec::new();
    let mut concurrency = vec![1usize];
    let mut repeat = 1usize;
    let mut max_new = 2048usize;
    let mut markdown = false;
    let mut stages = false;
    let mut vision_scan = false;
    let mut micro = false;
    let mut micro_decode = None;
    let mut overlap = false;
    let mut gate = false;
    let mut baseline = false;
    let mut json = None;
    let mut dump = None;
    let mut prep_threads = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--concurrency" => {
                concurrency = it
                    .next()
                    .context("--concurrency needs a value")?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()
                    .context("--concurrency list")?;
            }
            "--repeat" => repeat = it.next().context("--repeat")?.parse()?,
            "--max-new" => max_new = it.next().context("--max-new")?.parse()?,
            "--markdown" => markdown = true,
            "--stages" => stages = true,
            "--vision-scan" => vision_scan = true,
            "--micro" => micro = true,
            "--micro-decode" => micro_decode = Some(it.next().context("--micro-decode")?.parse()?),
            "--overlap" => overlap = true,
            "--gate" => gate = true,
            "--baseline" => baseline = true,
            "--json" => json = Some(PathBuf::from(it.next().context("--json")?)),
            "--dump" => dump = Some(PathBuf::from(it.next().context("--dump")?)),
            "--prep-threads" => prep_threads = Some(it.next().context("--prep-threads")?.parse()?),
            other => {
                let p = PathBuf::from(other);
                if p.is_dir() {
                    let mut fs: Vec<PathBuf> = std::fs::read_dir(&p)?
                        .flatten()
                        .map(|e| e.path())
                        .filter(|p| {
                            matches!(
                                p.extension().and_then(|e| e.to_str()),
                                Some("png") | Some("jpg") | Some("jpeg")
                            )
                        })
                        .collect();
                    fs.sort();
                    pages.extend(fs);
                } else {
                    pages.push(p);
                }
            }
        }
    }
    anyhow::ensure!(!pages.is_empty(), "no pages given");
    Ok(Args {
        pages,
        concurrency,
        repeat,
        max_new,
        markdown,
        stages,
        vision_scan,
        micro,
        micro_decode,
        overlap,
        gate,
        baseline,
        json,
        dump,
        prep_threads,
    })
}

fn main() -> Result<()> {
    if std::env::var_os("NV_OCR_BSN").is_none() {
        std::env::set_var("NV_OCR_BSN", "0");
        eprintln!("[ocr-batch-bench] pinning NV_OCR_BSN=0: this tool sweeps the legacy scheduler's slots/prep_threads, which the bs=N engine ignores; set NV_OCR_BSN=1 to override");
    }
    let args = parse_args()?;
    let prompt = if args.markdown {
        PROMPT_GROUNDING_MARKDOWN
    } else {
        PROMPT_FREE_OCR
    };
    let mode = ResolutionMode::Gundam;

    let dir = default_dir().context("DeepSeek-OCR-2 snapshot not found; set NV_DSOCR_DIR")?;
    let device = {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0)?
        }
        #[cfg(not(feature = "cuda"))]
        {
            Device::Cpu
        }
    };
    eprintln!("loading DeepSeek-OCR-2 from {}", dir.display());
    let t0 = Instant::now();
    let pipeline = Arc::new(DeepSeekOcr2Pipeline::load(
        &dir,
        &device,
        DecoderPrecision::Bf16,
    )?);
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let mut images = Vec::new();
    for p in &args.pages {
        let bytes = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
        images.push((p.clone(), bytes));
    }

    let mut records: Vec<String> = Vec::new();

    if args.stages {
        eprintln!("== stage breakdown (serial, device-synced) ==");
        for (path, bytes) in &images {
            let t = Instant::now();
            let img = decode_rgb(bytes)?;
            let decode_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let prep = nv_models::deepseek_ocr::preprocess::prepare(&img, mode)?;
            let prep_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let feats = pipeline.vision().encode_prepared(&prep)?;
            device.synchronize()?;
            let vision_ms = t.elapsed().as_secs_f64() * 1e3;
            let n_vis = prep.vision_tokens();
            let tokens = nv_models::deepseek_ocr::build_prompt_tokens(
                |s| pipeline.encode_text(s),
                prompt,
                n_vis,
            )?;
            let t = Instant::now();
            let one = pipeline.decoder().generate_detected(
                &tokens,
                Some(&feats),
                &GenerateOptions {
                    max_new_tokens: 1,
                    ..Default::default()
                },
            )?;
            device.synchronize()?;
            let prefill_ms = t.elapsed().as_secs_f64() * 1e3;
            let _ = one;
            let t = Instant::now();
            let full = pipeline.generate_tokens(
                &img,
                prompt,
                mode,
                &GenerateOptions {
                    max_new_tokens: args.max_new,
                    ..Default::default()
                },
            )?;
            device.synchronize()?;
            let total_ms = t.elapsed().as_secs_f64() * 1e3;
            let out_tokens = full.0.len();
            let decode_only = total_ms - prep_ms - vision_ms - prefill_ms;
            println!(
                "stage {} imgdecode={:.1} prep={:.1} vision={:.1} prefill={:.1} decode={:.1} \
                 total={:.1} out_tokens={} vis_tokens={} ms/step={:.2}",
                path.file_name().unwrap().to_string_lossy(),
                decode_ms,
                prep_ms,
                vision_ms,
                prefill_ms,
                decode_only,
                total_ms,
                out_tokens,
                n_vis,
                decode_only / out_tokens.max(1) as f64
            );
            records.push(format!(
                "{{\"kind\":\"stage\",\"page\":\"{}\",\"imgdecode_ms\":{:.2},\"prep_ms\":{:.2},\
                 \"vision_ms\":{:.2},\"prefill_ms\":{:.2},\"decode_ms\":{:.2},\"total_ms\":{:.2},\
                 \"out_tokens\":{},\"vision_tokens\":{}}}",
                path.file_name().unwrap().to_string_lossy(),
                decode_ms,
                prep_ms,
                vision_ms,
                prefill_ms,
                decode_only,
                total_ms,
                out_tokens,
                n_vis
            ));
        }
    }

    if args.vision_scan {
        eprintln!("== vision-tower batch scaling (device-synced) ==");
        let img = decode_rgb(&images[0].1)?;
        let prep = nv_models::deepseek_ocr::preprocess::prepare(&img, mode)?;
        let vis = pipeline.vision();
        let ts = prep.tile_size;
        let gs = prep.global_size;
        let per_tile: Vec<f32> = prep.tiles[0].clone();
        let global: Vec<f32> = prep.global.clone();
        for &b in &[1usize, 2, 4, 7, 14, 28] {
            let mut flat = Vec::with_capacity(b * 3 * ts * ts);
            for _ in 0..b {
                flat.extend_from_slice(&per_tile);
            }
            let t = candle_core::Tensor::from_slice(&flat, (b, 3, ts, ts), &Device::Cpu)?
                .to_device(&device)?
                .to_dtype(vis.dtype())?;
            let _ = vis.encode_batch(&t)?;
            device.synchronize()?;
            let t0 = Instant::now();
            let n = 3;
            for _ in 0..n {
                let _ = vis.encode_batch(&t)?;
            }
            device.synchronize()?;
            let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
            println!(
                "vision tile{ts} batch={b} ms={ms:.1} ms_per_view={:.2}",
                ms / b as f64
            );
            records.push(format!(
                "{{\"kind\":\"vision_scan\",\"view\":\"tile\",\"size\":{ts},\"batch\":{b},\
                 \"ms\":{ms:.3},\"ms_per_view\":{:.3}}}",
                ms / b as f64
            ));
        }
        for &b in &[1usize, 2, 4, 8] {
            let mut flat = Vec::with_capacity(b * 3 * gs * gs);
            for _ in 0..b {
                flat.extend_from_slice(&global);
            }
            let t = candle_core::Tensor::from_slice(&flat, (b, 3, gs, gs), &Device::Cpu)?
                .to_device(&device)?
                .to_dtype(vis.dtype())?;
            let _ = vis.encode_batch(&t)?;
            device.synchronize()?;
            let t0 = Instant::now();
            let n = 3;
            for _ in 0..n {
                let _ = vis.encode_batch(&t)?;
            }
            device.synchronize()?;
            let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
            println!(
                "vision global{gs} batch={b} ms={ms:.1} ms_per_view={:.2}",
                ms / b as f64
            );
            records.push(format!(
                "{{\"kind\":\"vision_scan\",\"view\":\"global\",\"size\":{gs},\"batch\":{b},\
                 \"ms\":{ms:.3},\"ms_per_view\":{:.3}}}",
                ms / b as f64
            ));
        }
    }

    if args.micro || args.micro_decode.is_some() {
        eprintln!("== microbench: per-stage concurrency scaling ==");
        let img = decode_rgb(&images[0].1)?;
        let prep = Arc::new(nv_models::deepseek_ocr::preprocess::prepare(&img, mode)?);
        let feats = Arc::new(pipeline.vision().encode_prepared(&prep)?);
        device.synchronize()?;
        let views = prep.tiles.len() + 1;

        for &c in if args.micro {
            &[1usize, 2, 4][..]
        } else {
            &[][..]
        } {
            let iters = 4usize;
            let t0 = Instant::now();
            let hs: Vec<_> = (0..c)
                .map(|_| {
                    let pipe = pipeline.clone();
                    let prep = prep.clone();
                    std::thread::spawn(move || {
                        for _ in 0..iters {
                            let _ = pipe.vision().encode_prepared(&prep).unwrap();
                        }
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
            device.synchronize()?;
            let dt = t0.elapsed().as_secs_f64();
            let n = (c * iters) as f64;
            println!(
                "micro vision conc={c} calls={} wall={dt:.2}s pages_vision_per_s={:.2} \
                 views_per_s={:.1} ms_per_call={:.1}",
                n as usize,
                n / dt,
                n * views as f64 / dt,
                dt * 1e3 / n
            );
            records.push(format!(
                "{{\"kind\":\"micro_vision\",\"concurrency\":{c},\"calls\":{},\
                 \"wall_s\":{dt:.4},\"calls_per_s\":{:.4}}}",
                n as usize,
                n / dt
            ));
        }

        #[cfg(feature = "cuda")]
        for c in args.micro_decode.into_iter() {
            let steps = 256usize;
            let tokens = nv_models::deepseek_ocr::build_prompt_tokens(
                |s| pipeline.encode_text(s),
                prompt,
                prep.vision_tokens(),
            )?;
            let cap = pipeline.decoder().config().max_position_embeddings;
            let gopts = GenerateOptions {
                max_new_tokens: steps,
                ..Default::default()
            };
            let tokens = Arc::new(tokens);
            let init = Arc::new(std::sync::Mutex::new(()));
            let barrier = Arc::new(std::sync::Barrier::new(c + 1));
            let hs: Vec<_> = (0..c)
                .map(|_| {
                    let pipe = pipeline.clone();
                    let tokens = tokens.clone();
                    let feats = feats.clone();
                    let gopts = gopts.clone();
                    let init = init.clone();
                    let barrier = barrier.clone();
                    std::thread::spawn(move || {
                        let mut g = {
                            let _l = init.lock().unwrap();
                            let mut g = nv_models::deepseek_ocr::DsocrDecodeGraph::new(
                                pipe.decoder_arc(),
                                cap,
                            )
                            .unwrap();
                            let _ = g.generate(&tokens, Some(&feats), &gopts).unwrap();
                            g
                        };
                        barrier.wait();
                        let out = g.generate(&tokens, Some(&feats), &gopts).unwrap();
                        out.tokens.len()
                    })
                })
                .collect();
            barrier.wait();
            let t0 = Instant::now();
            let mut total = 0usize;
            for h in hs {
                total += h.join().unwrap();
            }
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "micro decode conc={c} tokens={total} wall={dt:.2}s tok/s={:.1} \
                 ms_per_step_per_stream={:.2}",
                total as f64 / dt,
                dt * 1e3 / (total as f64 / c as f64)
            );
            records.push(format!(
                "{{\"kind\":\"micro_decode\",\"concurrency\":{c},\"tokens\":{total},\
                 \"wall_s\":{dt:.4},\"tokens_per_s\":{:.3}}}",
                total as f64 / dt
            ));
        }
    }

    #[cfg(feature = "cuda")]
    if args.overlap {
        eprintln!("== overlap probe: vision thread vs decode thread on one device ==");
        let img = decode_rgb(&images[0].1)?;
        let prep = Arc::new(nv_models::deepseek_ocr::preprocess::prepare(&img, mode)?);
        let feats = Arc::new(pipeline.vision().encode_prepared(&prep)?);
        device.synchronize()?;
        let tokens = Arc::new(nv_models::deepseek_ocr::build_prompt_tokens(
            |s| pipeline.encode_text(s),
            prompt,
            prep.vision_tokens(),
        )?);
        let steps = 256usize;
        let gopts = GenerateOptions {
            max_new_tokens: steps,
            ..Default::default()
        };
        let cap = pipeline.decoder().config().max_position_embeddings;
        let n_vis_calls = 6usize;
        let n_gen = 2usize;

        let mut graph =
            nv_models::deepseek_ocr::DsocrDecodeGraph::new(pipeline.decoder_arc(), cap)?;
        let _ = graph.generate(&tokens, Some(&feats), &gopts)?;
        for _ in 0..2 {
            let _ = pipeline.vision().encode_prepared(&prep)?;
        }
        device.synchronize()?;

        let t = Instant::now();
        for _ in 0..n_vis_calls {
            let _ = pipeline.vision().encode_prepared(&prep)?;
        }
        device.synchronize()?;
        let solo_vis = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut solo_tok = 0usize;
        for _ in 0..n_gen {
            solo_tok += graph.generate(&tokens, Some(&feats), &gopts)?.tokens.len();
        }
        device.synchronize()?;
        let solo_dec = t.elapsed().as_secs_f64();

        let barrier = Arc::new(std::sync::Barrier::new(2));
        let b2 = barrier.clone();
        let tk = tokens.clone();
        let ft = feats.clone();
        let go = gopts.clone();
        let dec = std::thread::spawn(move || {
            b2.wait();
            let t = Instant::now();
            let mut n = 0usize;
            for _ in 0..n_gen {
                n += graph.generate(&tk, Some(&ft), &go).unwrap().tokens.len();
            }
            (n, t.elapsed().as_secs_f64())
        });
        barrier.wait();
        let t = Instant::now();
        for _ in 0..n_vis_calls {
            let _ = pipeline.vision().encode_prepared(&prep)?;
        }
        let ov_vis = t.elapsed().as_secs_f64();
        let (ov_tok, ov_dec) = dec.join().unwrap();
        println!(
            "overlap solo_vision={:.2}calls/s solo_decode={:.1}tok/s | \
             overlapped_vision={:.2}calls/s overlapped_decode={:.1}tok/s | \
             vision_ratio={:.2} decode_ratio={:.2}",
            n_vis_calls as f64 / solo_vis,
            solo_tok as f64 / solo_dec,
            n_vis_calls as f64 / ov_vis,
            ov_tok as f64 / ov_dec,
            (n_vis_calls as f64 / ov_vis) / (n_vis_calls as f64 / solo_vis),
            (ov_tok as f64 / ov_dec) / (solo_tok as f64 / solo_dec)
        );
        records.push(format!(
            "{{\"kind\":\"overlap\",\"solo_vision_calls_per_s\":{:.4},\
             \"solo_decode_tok_per_s\":{:.2},\"ov_vision_calls_per_s\":{:.4},\
             \"ov_decode_tok_per_s\":{:.2}}}",
            n_vis_calls as f64 / solo_vis,
            solo_tok as f64 / solo_dec,
            n_vis_calls as f64 / ov_vis,
            ov_tok as f64 / ov_dec
        ));
    }

    if args.gate {
        eprintln!("== bs=1 interactive latency gate (serial, one job in flight) ==");
        let gopts = GenerateOptions {
            max_new_tokens: args.max_new,
            ..Default::default()
        };
        let sched = DsocrScheduler::new(
            pipeline.clone(),
            BatchOptions {
                slots: 1,
                prep_threads: 1,
                queue_depth: 4,
                loop_retry: true,
            },
        );
        for (_, bytes) in &images {
            let img = decode_rgb(bytes)?;
            let _ = pipeline.recognize(&img, prompt, mode, &gopts)?;
        }
        for round in 1..=args.repeat.max(1) {
            for (path, bytes) in &images {
                let img = decode_rgb(bytes)?;
                let t = Instant::now();
                let direct = pipeline.recognize(&img, prompt, mode, &gopts)?;
                let direct_ms = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let out = sched
                    .run_blocking(OcrJob {
                        input: JobInput::Bytes(bytes.clone()),
                        prompt: prompt.to_string(),
                        mode,
                        max_new_tokens: args.max_new,
                    })
                    .map_err(|e| anyhow::anyhow!(e))?;
                let sched_ms = t.elapsed().as_secs_f64() * 1e3;
                let same = out.text == direct;
                println!(
                    "gate round={round} page={} direct_ms={direct_ms:.0} sched_ms={sched_ms:.0} \
                     delta={:+.1}% identical={same}",
                    path.file_name().unwrap().to_string_lossy(),
                    (sched_ms - direct_ms) / direct_ms * 100.0
                );
                records.push(format!(
                    "{{\"kind\":\"gate\",\"round\":{round},\"page\":\"{}\",\
                     \"direct_ms\":{direct_ms:.2},\"sched_ms\":{sched_ms:.2},\"identical\":{same}}}",
                    path.file_name().unwrap().to_string_lossy()
                ));
            }
        }
    }

    if args.baseline {
        eprintln!("== baseline: pipeline.recognize, serial, no scheduler ==");
        let opts = GenerateOptions {
            max_new_tokens: args.max_new,
            ..Default::default()
        };
        for round in 0..(args.repeat + 1) {
            let mut lat = Vec::new();
            let t_all = Instant::now();
            for (_, bytes) in &images {
                let img = decode_rgb(bytes)?;
                let t = Instant::now();
                let _ = pipeline.recognize(&img, prompt, mode, &opts)?;
                lat.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let wall = t_all.elapsed().as_secs_f64();
            let tag = if round == 0 { "warmup" } else { "round" };
            println!(
                "baseline {tag} pages={} wall={:.2}s pages/s={:.3} mean_latency_ms={:.1}",
                images.len(),
                wall,
                images.len() as f64 / wall,
                mean(&lat)
            );
            if round > 0 {
                records.push(format!(
                    "{{\"kind\":\"baseline\",\"pages\":{},\"wall_s\":{:.4},\
                     \"pages_per_s\":{:.4},\"mean_latency_ms\":{:.2}}}",
                    images.len(),
                    wall,
                    images.len() as f64 / wall,
                    mean(&lat)
                ));
            }
        }
    }

    for &c in args.concurrency.iter().filter(|&&c| c >= 1) {
        let opts = BatchOptions {
            slots: c,
            prep_threads: args.prep_threads.unwrap_or(c.max(2)),
            queue_depth: 256,
            ..BatchOptions::from_env()
        };
        let sched = DsocrScheduler::new(pipeline.clone(), opts);
        let build_jobs = |n: usize| -> Vec<OcrJob> {
            let mut jobs = Vec::new();
            for i in 0..n {
                let (_, bytes) = &images[i % images.len()];
                jobs.push(OcrJob {
                    input: JobInput::Bytes(bytes.clone()),
                    prompt: prompt.to_string(),
                    mode,
                    max_new_tokens: args.max_new,
                });
            }
            jobs
        };

        let warm = sched.run_all(build_jobs(c.max(images.len())));
        for r in &warm {
            if let Err(e) = r {
                anyhow::bail!("warmup job failed at concurrency {c}: {e}");
            }
        }

        for round in 1..=args.repeat {
            let n = images.len() * 4;
            let t_all = Instant::now();
            let out = sched.run_all(build_jobs(n));
            let wall = t_all.elapsed().as_secs_f64();
            let mut lat: Vec<f64> = Vec::new();
            let mut toks = 0usize;
            let mut looped = 0usize;
            for r in &out {
                match r {
                    Ok(o) => {
                        lat.push(o.timings.total_ms);
                        toks += o.timings.out_tokens;
                        if o.looped {
                            looped += 1;
                        }
                    }
                    Err(e) => anyhow::bail!("job failed at concurrency {c}: {e}"),
                }
            }
            lat.sort_by(|a, b| a.total_cmp(b));
            println!(
                "conc={} prep={} round={} jobs={} wall={:.2}s pages/s={:.3} tok/s={:.1} \
                 lat_mean={:.0} lat_p50={:.0} lat_p95={:.0} looped={}",
                c,
                opts.prep_threads,
                round,
                n,
                wall,
                n as f64 / wall,
                toks as f64 / wall,
                mean(&lat),
                pct(&lat, 0.5),
                pct(&lat, 0.95),
                looped
            );
            records.push(format!(
                "{{\"kind\":\"throughput\",\"concurrency\":{},\"prep_threads\":{},\"round\":{},\
                 \"jobs\":{},\"wall_s\":{:.4},\"pages_per_s\":{:.4},\"tokens_per_s\":{:.2},\
                 \"lat_mean_ms\":{:.2},\"lat_p50_ms\":{:.2},\"lat_p95_ms\":{:.2},\"looped\":{}}}",
                c,
                opts.prep_threads,
                round,
                n,
                wall,
                n as f64 / wall,
                toks as f64 / wall,
                mean(&lat),
                pct(&lat, 0.5),
                pct(&lat, 0.95),
                looped
            ));
            if let Some(d) = args.dump.as_ref() {
                std::fs::create_dir_all(d)?;
                for (i, r) in out.iter().enumerate() {
                    if let Ok(o) = r {
                        let name = images[i % images.len()]
                            .0
                            .file_stem()
                            .unwrap()
                            .to_string_lossy()
                            .to_string();
                        std::fs::write(
                            d.join(format!("c{c}-r{round}-{i:03}-{name}.txt")),
                            &o.text,
                        )?;
                    }
                }
            }
        }
    }

    if let Some(p) = args.json {
        let body = format!("[{}]\n", records.join(",\n"));
        std::fs::write(&p, body)?;
        eprintln!("wrote {}", p.display());
    }
    Ok(())
}
