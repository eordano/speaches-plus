use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::Device;
#[cfg(feature = "cuda")]
use nv_models::deepseek_ocr::decoder_graph_batch::{BatchSampler, DsocrBatchDecodeGraph};
#[cfg(feature = "cuda")]
use nv_models::deepseek_ocr::{build_prompt_tokens, DsocrDecodeGraph, GenerateOptions};
use nv_models::deepseek_ocr::{
    DecoderPrecision, DeepSeekOcr2Pipeline, ResolutionMode, PROMPT_FREE_OCR,
};
use speaches_plus::oapi::ocr_batch::{
    decode_rgb, BatchOptions, DsocrScheduler, JobInput, OcrJob, OcrOutput,
};

#[cfg(feature = "cuda")]
use speaches_plus::oapi::ocr_batch_n::{BsnOptions, DsocrBsnEngine};

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

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(((v.len() - 1) as f64) * p).round() as usize]
}

struct Args {
    pages: Vec<PathBuf>,
    arm: String,
    concurrency: Vec<usize>,
    latency_reps: usize,
    max_new: usize,
    slots: usize,
    buckets: Option<String>,
    front: Option<usize>,
    json: Option<PathBuf>,
    dump: Option<PathBuf>,
    warmup: bool,
    #[allow(dead_code)]
    micro_step: Vec<usize>,
    #[allow(dead_code)]
    micro_iters: usize,
    #[allow(dead_code)]
    cap: Option<usize>,
    #[allow(dead_code)]
    preload: bool,
}

fn parse_args() -> Result<Args> {
    let mut pages = Vec::new();
    let mut arm = "bsn".to_string();
    let mut concurrency = vec![1usize];
    let mut latency_reps = 0usize;
    let mut max_new = 2048usize;
    let mut slots = 1usize;
    let mut buckets = None;
    let mut front = None;
    let mut json = None;
    let mut dump = None;
    let mut warmup = true;
    let mut micro_step: Vec<usize> = Vec::new();
    let mut micro_iters = 128usize;
    let mut cap = None;
    let mut preload = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--arm" => arm = it.next().context("--arm needs a value")?,
            "--concurrency" => {
                concurrency = it
                    .next()
                    .context("--concurrency needs a value")?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<std::result::Result<Vec<_>, _>>()?;
            }
            "--latency" => latency_reps = it.next().context("--latency needs N")?.parse()?,
            "--max-new" => max_new = it.next().context("--max-new needs a value")?.parse()?,
            "--slots" => slots = it.next().context("--slots needs a value")?.parse()?,
            "--buckets" => buckets = Some(it.next().context("--buckets needs a value")?),
            "--front" => front = Some(it.next().context("--front needs a value")?.parse()?),
            "--json" => json = Some(PathBuf::from(it.next().context("--json needs a path")?)),
            "--dump" => dump = Some(PathBuf::from(it.next().context("--dump needs a path")?)),
            "--no-warmup" => warmup = false,
            "--micro-step" => {
                micro_step = it
                    .next()
                    .context("--micro-step needs a value")?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<std::result::Result<Vec<_>, _>>()?;
            }
            "--micro-iters" => {
                micro_iters = it.next().context("--micro-iters needs a value")?.parse()?
            }
            "--cap" => cap = Some(it.next().context("--cap needs a value")?.parse()?),
            "--preload" => preload = true,
            "--pages-from" => {
                let p = it.next().context("--pages-from needs a path")?;
                for line in std::fs::read_to_string(&p)?.lines() {
                    let l = line.trim();
                    if !l.is_empty() {
                        pages.push(PathBuf::from(l));
                    }
                }
            }
            other => pages.push(PathBuf::from(other)),
        }
    }
    anyhow::ensure!(!pages.is_empty(), "no pages given");
    Ok(Args {
        pages,
        arm,
        concurrency,
        latency_reps,
        max_new,
        slots,
        buckets,
        front,
        json,
        dump,
        warmup,
        micro_step,
        micro_iters,
        cap,
        preload,
    })
}

enum Engine {
    Legacy(DsocrScheduler),
    #[cfg(feature = "cuda")]
    Bsn(DsocrBsnEngine),
}

impl Engine {
    fn run_all(&self, jobs: Vec<OcrJob>) -> Vec<std::result::Result<OcrOutput, String>> {
        match self {
            Engine::Legacy(s) => s.run_all(jobs),
            #[cfg(feature = "cuda")]
            Engine::Bsn(e) => e.run_all(jobs),
        }
    }
}

fn make_job(bytes: Vec<u8>, max_new: usize) -> OcrJob {
    OcrJob {
        input: JobInput::Bytes(bytes),
        prompt: PROMPT_FREE_OCR.to_string(),
        mode: ResolutionMode::Gundam,
        max_new_tokens: max_new,
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let dir = default_dir().context("DeepSeek-OCR-2 snapshot not found")?;
    let device = Device::new_cuda(0).context("cuda device")?;
    eprintln!("[bsn-bench] loading {} on {:?}", dir.display(), device);
    let pipeline = Arc::new(DeepSeekOcr2Pipeline::load(
        &dir,
        &device,
        DecoderPrecision::Bf16,
    )?);

    #[cfg(feature = "cuda")]
    if !args.micro_step.is_empty() {
        return micro_step(&pipeline, &args);
    }
    #[cfg(feature = "cuda")]
    if args.preload {
        return preload_decode(&pipeline, &args);
    }

    if let Some(b) = args.buckets.as_ref() {
        std::env::set_var("NV_DSOCR_BSN_BUCKETS", b);
    }
    if let Some(f) = args.front {
        std::env::set_var("NV_OCR_BSN_FRONT", f.to_string());
    }

    let engine = match args.arm.as_str() {
        "bs1" | "legacy" => Engine::Legacy(DsocrScheduler::new(
            pipeline.clone(),
            BatchOptions {
                slots: args.slots,
                prep_threads: args.slots.max(2),
                queue_depth: 256,
                loop_retry: true,
            },
        )),
        #[cfg(feature = "cuda")]
        "bsn" => Engine::Bsn(DsocrBsnEngine::new(
            pipeline.clone(),
            BsnOptions::from_env(),
        )),
        other => anyhow::bail!("unknown --arm {other}"),
    };

    let bytes: Vec<Vec<u8>> = args
        .pages
        .iter()
        .map(|p| std::fs::read(p).with_context(|| format!("read {}", p.display())))
        .collect::<Result<Vec<_>>>()?;
    eprintln!("[bsn-bench] {} distinct pages loaded", bytes.len());

    let mut rows: Vec<serde_json::Value> = Vec::new();

    if args.warmup {
        let t = Instant::now();
        let out = engine.run_all(vec![make_job(bytes[0].clone(), args.max_new.min(64))]);
        eprintln!(
            "[bsn-bench] warmup {:?} in {:.0} ms",
            out[0].as_ref().map(|o| o.tokens.len()),
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    if args.latency_reps > 0 {
        let mut lat = Vec::new();
        let mut toks = Vec::new();
        for r in 0..args.latency_reps {
            let img = &bytes[r % bytes.len()];
            let t = Instant::now();
            let out = engine.run_all(vec![make_job(img.clone(), args.max_new)]);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            match &out[0] {
                Ok(o) => {
                    lat.push(ms);
                    toks.push(o.tokens.len() as f64);
                }
                Err(e) => eprintln!("[bsn-bench] latency rep {r} failed: {e}"),
            }
        }
        let mut l2 = lat.clone();
        println!(
            "arm={} LATENCY reps={} mean={:.1} ms p50={:.1} p95={:.1} tokens_mean={:.0} tok/s={:.1}",
            args.arm,
            lat.len(),
            mean(&lat),
            pct(&mut l2, 0.5),
            pct(&mut l2.clone(), 0.95),
            mean(&toks),
            mean(&toks) / (mean(&lat) / 1e3)
        );
        rows.push(serde_json::json!({
            "kind": "latency",
            "arm": args.arm,
            "reps": lat.len(),
            "mean_ms": mean(&lat),
            "p50_ms": pct(&mut lat.clone(), 0.5),
            "p95_ms": pct(&mut lat.clone(), 0.95),
            "tokens_mean": mean(&toks),
        }));
    }

    let mut dumped: Vec<serde_json::Value> = Vec::new();
    let mut cursor = 0usize;
    for &c in &args.concurrency {
        anyhow::ensure!(
            c <= bytes.len(),
            "concurrency {c} exceeds the {} distinct pages given",
            bytes.len()
        );
        let mut jobs = Vec::with_capacity(c);
        let mut used = Vec::with_capacity(c);
        for _ in 0..c {
            let idx = cursor % bytes.len();
            cursor += 1;
            used.push(args.pages[idx].clone());
            jobs.push(make_job(bytes[idx].clone(), args.max_new));
        }
        let t = Instant::now();
        let outs = engine.run_all(jobs);
        let wall = t.elapsed().as_secs_f64();
        let mut lat = Vec::new();
        let mut ntok = 0usize;
        let mut errs = 0usize;
        let mut looped = 0usize;
        for (i, o) in outs.iter().enumerate() {
            match o {
                Ok(o) => {
                    lat.push(o.timings.total_ms);
                    ntok += o.tokens.len();
                    if o.looped {
                        looped += 1;
                    }
                    if args.dump.is_some() {
                        dumped.push(serde_json::json!({
                            "concurrency": c,
                            "page": used[i].to_string_lossy(),
                            "tokens": o.tokens,
                            "text": o.text,
                        }));
                    }
                }
                Err(e) => {
                    errs += 1;
                    eprintln!("[bsn-bench] c={c} job {i} failed: {e}");
                }
            }
        }
        let ok = lat.len();
        println!(
            "arm={} c={} pages={} ok={} err={} looped={} wall={:.3}s pages/s={:.3} tok/s={:.1} \
             lat_mean={:.0} lat_p50={:.0} lat_p95={:.0}",
            args.arm,
            c,
            c,
            ok,
            errs,
            looped,
            wall,
            ok as f64 / wall,
            ntok as f64 / wall,
            mean(&lat),
            pct(&mut lat.clone(), 0.5),
            pct(&mut lat.clone(), 0.95),
        );
        rows.push(serde_json::json!({
            "kind": "throughput",
            "arm": args.arm,
            "concurrency": c,
            "ok": ok,
            "err": errs,
            "looped": looped,
            "wall_s": wall,
            "pages_per_s": ok as f64 / wall,
            "tok_per_s": ntok as f64 / wall,
            "lat_mean_ms": mean(&lat),
            "lat_p50_ms": pct(&mut lat.clone(), 0.5),
            "lat_p95_ms": pct(&mut lat.clone(), 0.95),
            "out_tokens": ntok,
        }));
    }

    if let Some(p) = args.json.as_ref() {
        std::fs::write(p, serde_json::to_string_pretty(&rows)?)?;
    }
    if let Some(p) = args.dump.as_ref() {
        std::fs::write(p, serde_json::to_string_pretty(&dumped)?)?;
    }
    drop(engine);
    Ok(())
}

#[cfg(feature = "cuda")]
fn micro_step(pipeline: &DeepSeekOcr2Pipeline, args: &Args) -> Result<()> {
    let decoder = pipeline.decoder_arc();
    let cfg_max = decoder.config().max_position_embeddings;
    let cap = args.cap.unwrap_or(cfg_max).min(cfg_max);
    let vocab = decoder.config().vocab_size;
    let max_b = args.micro_step.iter().copied().max().unwrap_or(1);
    anyhow::ensure!(
        args.pages.len() >= max_b,
        "--micro-step {max_b} needs at least {max_b} DISTINCT pages, got {}",
        args.pages.len()
    );

    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut feats: Vec<candle_core::Tensor> = Vec::new();
    for p in args.pages.iter().take(max_b) {
        let bytes = std::fs::read(p)?;
        let img = decode_rgb(&bytes)?;
        let prep = nv_models::deepseek_ocr::preprocess::prepare(&img, ResolutionMode::Gundam)?;
        let f = pipeline.vision().encode_prepared(&prep)?;
        let t = build_prompt_tokens(
            |s| pipeline.encode_text(s),
            PROMPT_FREE_OCR,
            prep.vision_tokens(),
        )?;
        eprintln!(
            "[micro] {} prompt={} vis={}",
            p.display(),
            t.len(),
            prep.vision_tokens()
        );
        prompts.push(t);
        feats.push(f);
    }

    let iters = args.micro_iters;
    let mut rows: Vec<serde_json::Value> = Vec::new();

    {
        let mut g = DsocrDecodeGraph::new(decoder.clone(), cap)?;
        let mut best = f64::INFINITY;
        for _ in 0..2 {
            let max_len = (prompts[0].len() + iters + 8).min(cap);
            let mut cache = decoder.new_kv_cache(max_len)?;
            let x = decoder.embed_tokens_with_vision(&prompts[0], Some(&feats[0]))?;
            let _ = decoder.forward_embeds_hidden(&x, &mut cache)?;
            g.reset();
            g.load_kv_from_cache(&cache)?;
            drop(cache);
            let mut tok = 1u32;
            g.step(tok)?;
            let _ = g.logits_host()?;
            let t = Instant::now();
            for i in 0..iters {
                tok = ((i % 1000) + 100) as u32;
                g.step(tok)?;
                let l = g.logits_host()?;
                std::hint::black_box(l[0]);
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
            best = best.min(ms);
        }
        println!(
            "micro b=1(single-graph) ms/step={best:.3} ms/token={best:.3} tok/s={:.1}",
            1e3 / best
        );
        rows.push(serde_json::json!({"kind":"micro","engine":"bs1-graph","b":1,"ms_per_step":best,"ms_per_token":best}));
    }

    let mut graph = DsocrBatchDecodeGraph::new(decoder.clone(), cap, args.micro_step.clone())?;
    for &b in &args.micro_step {
        let mut best = f64::INFINITY;
        let mut best_step = f64::INFINITY;
        let mut best_dtoh = f64::INFINITY;
        for _round in 0..2 {
            for j in 0..b {
                let max_len = (prompts[j].len() + iters + 8).min(cap);
                graph.prefill_slot(j, &prompts[j], Some(&feats[j]), max_len)?;
            }
            let toks: Vec<Option<u32>> = (0..b).map(|_| Some(1u32)).collect();
            graph.step_batch(&toks)?;
            let _ = graph.logits_batch(b)?;
            let mut t_step = 0f64;
            let mut t_dtoh = 0f64;
            let t = Instant::now();
            for i in 0..iters {
                let toks: Vec<Option<u32>> = (0..b)
                    .map(|j| Some((((i * 7 + j * 13) % 1000) + 100) as u32))
                    .collect();
                let a = Instant::now();
                graph.step_batch(&toks)?;
                let m = Instant::now();
                let l = graph.logits_batch(b)?;
                std::hint::black_box(l[0]);
                t_step += m.duration_since(a).as_secs_f64();
                t_dtoh += m.elapsed().as_secs_f64();
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
            if ms < best {
                best = ms;
                best_step = t_step * 1e3 / iters as f64;
                best_dtoh = t_dtoh * 1e3 / iters as f64;
            }
            for j in 0..b {
                graph.release_slot(j);
            }
        }
        let per_tok = best / b as f64;
        println!(
            "micro b={b} ms/step={best:.3} (step={best_step:.3} dtoh={best_dtoh:.3})              ms/token={per_tok:.3} tok/s={:.1} vocab={vocab}",
            1e3 / per_tok
        );
        rows.push(serde_json::json!({
            "kind":"micro","engine":"bsn","b":b,
            "ms_per_step":best,"ms_step_only":best_step,"ms_dtoh":best_dtoh,
            "ms_per_token":per_tok,"tok_per_s":1e3/per_tok
        }));
    }
    if let Some(p) = args.json.as_ref() {
        std::fs::write(p, serde_json::to_string_pretty(&rows)?)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
struct Pre {
    tokens: Vec<u32>,
    cache: nv_models::deepseek_ocr::DeepseekOcrKvCache,
    logits: Vec<f32>,
    feats: candle_core::Tensor,
}

#[cfg(feature = "cuda")]
fn preload_decode(pipeline: &DeepSeekOcr2Pipeline, args: &Args) -> Result<()> {
    let decoder = pipeline.decoder_arc();
    let cfg_max = decoder.config().max_position_embeddings;
    let cap = args.cap.unwrap_or(cfg_max).min(cfg_max);
    let vocab = decoder.config().vocab_size;
    let eos = decoder.config().eos_token_id;
    let opts = GenerateOptions {
        max_new_tokens: args.max_new,
        ..GenerateOptions::recipe()
    };
    let mut rows: Vec<serde_json::Value> = Vec::new();

    for &c in &args.concurrency {
        anyhow::ensure!(
            c <= args.pages.len(),
            "--preload c={c} needs {c} DISTINCT pages, got {}",
            args.pages.len()
        );
        let mut pre: Vec<Pre> = Vec::with_capacity(c);
        let t_front = Instant::now();
        for p in args.pages.iter().take(c) {
            let bytes = std::fs::read(p)?;
            let img = decode_rgb(&bytes)?;
            let prep = nv_models::deepseek_ocr::preprocess::prepare(&img, ResolutionMode::Gundam)?;
            let feats = pipeline.vision().encode_prepared(&prep)?;
            let tokens = build_prompt_tokens(
                |s| pipeline.encode_text(s),
                PROMPT_FREE_OCR,
                prep.vision_tokens(),
            )?;
            let max_len = (tokens.len() + args.max_new).min(cap);
            let (cache, logits) =
                DsocrBatchDecodeGraph::prefill_detached(&decoder, &tokens, Some(&feats), max_len)?;
            pre.push(Pre {
                tokens,
                cache,
                logits,
                feats,
            });
        }
        let front_s = t_front.elapsed().as_secs_f64();

        let mut bs1_tokens = 0usize;
        let bs1_s;
        {
            let mut g = DsocrDecodeGraph::new(decoder.clone(), cap)?;
            let t = Instant::now();
            for it in &pre {
                let mut rng = BatchSampler::new(opts.seed);
                let mut all = it.tokens.clone();
                let mut logits = it.logits.clone();
                g.reset();
                g.load_kv_from_cache(&it.cache)?;
                let mut n = 0usize;
                loop {
                    let nt = rng.next_token(&mut logits, &all, &opts)?;
                    all.push(nt);
                    n += 1;
                    if nt == eos || n >= args.max_new || g.current_len() + 1 >= cap {
                        break;
                    }
                    g.step(nt)?;
                    logits = g.logits_host()?;
                }
                bs1_tokens += n;
            }
            bs1_s = t.elapsed().as_secs_f64();
        }

        let mut bsn_tokens = 0usize;
        let mut steps = 0usize;
        let mut hist: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        let bsn_s;
        {
            let buckets = args
                .buckets
                .as_deref()
                .map(|b| b.split(',').filter_map(|t| t.trim().parse().ok()).collect())
                .unwrap_or_else(|| vec![1usize, 2, 4, 8]);
            let mut graph = DsocrBatchDecodeGraph::new(decoder.clone(), cap, buckets)?;
            let max_b = graph.max_batch();
            let mut slots: Vec<Option<(BatchSampler, Vec<u32>, u32, usize)>> =
                (0..max_b).map(|_| None).collect();
            let mut queue = 0usize;
            let t = Instant::now();
            loop {
                while let Some(slot) = graph.free_slot() {
                    if queue >= pre.len() {
                        break;
                    }
                    let it = &pre[queue];
                    queue += 1;
                    graph.install_prefilled(slot, &it.cache)?;
                    let mut rng = BatchSampler::new(opts.seed);
                    let mut all = it.tokens.clone();
                    let mut logits = it.logits.clone();
                    let nt = rng.next_token(&mut logits, &all, &opts)?;
                    all.push(nt);
                    bsn_tokens += 1;
                    if nt == eos || args.max_new <= 1 {
                        graph.release_slot(slot);
                    } else {
                        slots[slot] = Some((rng, all, nt, 1));
                    }
                }
                let extent = graph.active_extent();
                if extent == 0 {
                    if queue >= pre.len() {
                        break;
                    }
                    continue;
                }
                let bucket = graph.bucket_for(extent).context("bucket")?;
                *hist.entry(bucket).or_insert(0) += 1;
                steps += 1;
                let toks: Vec<Option<u32>> = (0..bucket)
                    .map(|j| slots[j].as_ref().map(|s| s.2))
                    .collect();
                graph.step_batch(&toks)?;
                let logits = graph.logits_batch(bucket)?;
                for j in 0..bucket {
                    let Some(st) = slots[j].as_mut() else {
                        continue;
                    };
                    let mut row = logits[j * vocab..(j + 1) * vocab].to_vec();
                    let nt = st.0.next_token(&mut row, &st.1, &opts)?;
                    st.1.push(nt);
                    st.2 = nt;
                    st.3 += 1;
                    bsn_tokens += 1;
                    if nt == eos || st.3 >= args.max_new || graph.slot_len(j) + 1 >= cap {
                        slots[j] = None;
                        graph.release_slot(j);
                    }
                }
            }
            bsn_s = t.elapsed().as_secs_f64();
        }

        let mean_b: f64 =
            hist.iter().map(|(k, v)| (k * v) as f64).sum::<f64>() / steps.max(1) as f64;
        println!(
            "preload c={c} front={front_s:.2}s ({:.3} pg/s) | bs1 decode {bs1_s:.2}s              {bs1_tokens} tok {:.1} tok/s {:.3} pg/s | bsN decode {bsn_s:.2}s {bsn_tokens} tok              {:.1} tok/s {:.3} pg/s | speedup {:.2}x | steps={steps} mean_bucket={mean_b:.2}              hist={hist:?}",
            c as f64 / front_s,
            bs1_tokens as f64 / bs1_s,
            c as f64 / bs1_s,
            bsn_tokens as f64 / bsn_s,
            c as f64 / bsn_s,
            bs1_s / bsn_s,
        );
        println!(
            "preload c={c} END-TO-END (front serial + decode): bs1 {:.3} pg/s | bsN {:.3} pg/s",
            c as f64 / (front_s + bs1_s),
            c as f64 / (front_s + bsn_s),
        );
        rows.push(serde_json::json!({
            "kind":"preload","concurrency":c,
            "front_s":front_s,"front_pages_per_s":c as f64/front_s,
            "bs1_decode_s":bs1_s,"bs1_tokens":bs1_tokens,"bs1_tok_per_s":bs1_tokens as f64/bs1_s,
            "bs1_decode_pages_per_s":c as f64/bs1_s,
            "bsn_decode_s":bsn_s,"bsn_tokens":bsn_tokens,"bsn_tok_per_s":bsn_tokens as f64/bsn_s,
            "bsn_decode_pages_per_s":c as f64/bsn_s,
            "decode_speedup":bs1_s/bsn_s,
            "steps":steps,"mean_bucket":mean_b,
            "e2e_bs1_pages_per_s": c as f64/(front_s+bs1_s),
            "e2e_bsn_pages_per_s": c as f64/(front_s+bsn_s),
        }));
    }
    if let Some(p) = args.json.as_ref() {
        std::fs::write(p, serde_json::to_string_pretty(&rows)?)?;
    }
    Ok(())
}
