#![cfg(feature = "cuda")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

use anyhow::Result;
use candle_core::Tensor;
use nv_models::deepseek_ocr::decoder::{detect_loop, strip_grounding_tokens, LOOP_CHECK_STRIDE};
use nv_models::deepseek_ocr::decoder_graph_batch::{
    buckets_from_env, BatchSampler, DsocrBatchDecodeGraph,
};
use nv_models::deepseek_ocr::preprocess::{prepare, PreparedViews};
use nv_models::deepseek_ocr::{
    build_prompt_tokens, DeepSeekOcr2Pipeline, DeepseekOcrKvCache, GenerateOptions,
    PROMPT_FREE_OCR, PROMPT_GROUNDING_MARKDOWN,
};

use super::ocr_batch::{decode_rgb, JobInput, JobTimings, OcrJob, OcrOutput};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Clone, Debug)]
pub struct BsnOptions {
    pub buckets: Vec<usize>,
    pub cap: Option<usize>,
    pub prep_threads: usize,
    pub front_threads: usize,
    pub queue_depth: usize,
    pub loop_retry: bool,
    pub fill_target: usize,
    pub fill_wait_ms: u64,
}

impl BsnOptions {
    pub fn from_env() -> Self {
        let buckets = buckets_from_env();
        Self {
            prep_threads: env_usize("NV_OCR_BSN_PREP_THREADS", 4).max(1),
            front_threads: env_usize("NV_OCR_BSN_FRONT", 2).max(1),
            queue_depth: env_usize("NV_OCR_BSN_QUEUE", 256).max(1),
            loop_retry: std::env::var("NV_DSOCR_LOOP_RETRY")
                .map(|v| v != "0")
                .unwrap_or(true),
            cap: std::env::var("NV_DSOCR_BSN_CAP")
                .ok()
                .and_then(|v| v.parse().ok()),
            fill_target: env_usize("NV_OCR_BSN_FILL", usize::MAX),
            fill_wait_ms: env_usize("NV_OCR_BSN_FILL_MS", 250) as u64,
            buckets,
        }
    }

    pub fn max_batch(&self) -> usize {
        self.buckets.last().copied().unwrap_or(1)
    }
}

impl Default for BsnOptions {
    fn default() -> Self {
        Self {
            buckets: vec![1, 2, 4, 8],
            cap: None,
            prep_threads: 4,
            front_threads: 2,
            queue_depth: 256,
            loop_retry: true,
            fill_target: usize::MAX,
            fill_wait_ms: 250,
        }
    }
}

type Reply = mpsc::SyncSender<Result<OcrOutput, String>>;

struct Envelope {
    prompt: String,
    mode: nv_models::deepseek_ocr::ResolutionMode,
    max_new_tokens: usize,
    input: JobInput,
    submitted: Instant,
    reply: Reply,
}

struct Prepared {
    prompt: String,
    max_new_tokens: usize,
    prep: PreparedViews,
    submitted: Instant,
    queue_ms: f64,
    prep_ms: f64,
    reply: Reply,
}

struct Admitted {
    prompt: String,
    max_new_tokens: usize,
    tokens: Vec<u32>,
    cache: DeepseekOcrKvCache,
    logits: Vec<f32>,
    feats: Tensor,
    vision_tokens: usize,
    max_len: usize,
    submitted: Instant,
    queue_ms: f64,
    prep_ms: f64,
    vision_ms: f64,
    prefill_ms: f64,
    retry: bool,
    prev_best: Option<Vec<u32>>,
    reply: Reply,
}

unsafe impl Send for Admitted {}

struct Seq {
    prompt: String,
    max_new_tokens: usize,
    all_tokens: Vec<u32>,
    generated: Vec<u32>,
    next: u32,
    rng: BatchSampler,
    max_len: usize,
    feats: Tensor,
    vision_tokens: usize,
    submitted: Instant,
    queue_ms: f64,
    prep_ms: f64,
    vision_ms: f64,
    prefill_ms: f64,
    decode_started: Instant,
    retry: bool,
    prev_best: Option<Vec<u32>>,
    reply: Reply,
}

pub struct DsocrBsnEngine {
    job_tx: Option<mpsc::SyncSender<Envelope>>,
    threads: Vec<thread::JoinHandle<()>>,
    opts: BsnOptions,
    inflight: Arc<AtomicUsize>,
}

impl DsocrBsnEngine {
    pub fn new(pipeline: Arc<DeepSeekOcr2Pipeline>, opts: BsnOptions) -> Self {
        let (job_tx, job_rx) = mpsc::sync_channel::<Envelope>(opts.queue_depth);
        let (prep_tx, prep_rx) = mpsc::sync_channel::<Prepared>(opts.queue_depth);
        let (adm_tx, adm_rx) = mpsc::sync_channel::<Admitted>(opts.max_batch().max(1) * 2);
        let job_rx = Arc::new(Mutex::new(job_rx));
        let prep_rx = Arc::new(Mutex::new(prep_rx));
        let inflight = Arc::new(AtomicUsize::new(0));

        let mut threads = Vec::new();
        for i in 0..opts.prep_threads {
            let rx = job_rx.clone();
            let tx = prep_tx.clone();
            threads.push(
                thread::Builder::new()
                    .name(format!("dsocr-bsn-prep-{i}"))
                    .spawn(move || prep_loop(rx, tx))
                    .expect("spawn bsn prep thread"),
            );
        }
        drop(prep_tx);
        let cfg_max = pipeline.decoder().config().max_position_embeddings;
        let cap = opts.cap.unwrap_or(cfg_max).min(cfg_max).max(1);
        for i in 0..opts.front_threads {
            let rx = prep_rx.clone();
            let tx = adm_tx.clone();
            let pipe = pipeline.clone();
            threads.push(
                thread::Builder::new()
                    .name(format!("dsocr-bsn-front-{i}"))
                    .spawn(move || front_loop(pipe, rx, tx, cap))
                    .expect("spawn bsn front thread"),
            );
        }
        drop(adm_tx);
        {
            let pipe = pipeline.clone();
            let o = opts.clone();
            let inflight_dec = inflight.clone();
            threads.push(
                thread::Builder::new()
                    .name("dsocr-bsn-decode".to_string())
                    .spawn(move || {
                        if let Err(e) = decode_loop(pipe, adm_rx, o, inflight_dec) {
                            eprintln!("[dsocr-bsn] decode loop died: {e:#}");
                        }
                    })
                    .expect("spawn bsn decode thread"),
            );
        }
        Self {
            job_tx: Some(job_tx),
            threads,
            opts,
            inflight,
        }
    }

    pub fn options(&self) -> &BsnOptions {
        &self.opts
    }

    pub fn submit(&self, job: OcrJob) -> mpsc::Receiver<Result<OcrOutput, String>> {
        let (tx, rx) = mpsc::sync_channel(1);
        self.inflight.fetch_add(1, Ordering::Relaxed);
        let env = Envelope {
            prompt: job.prompt,
            mode: job.mode,
            max_new_tokens: job.max_new_tokens,
            input: job.input,
            submitted: Instant::now(),
            reply: tx,
        };
        if let Some(job_tx) = self.job_tx.as_ref() {
            if job_tx.send(env).is_err() {
                self.inflight.fetch_sub(1, Ordering::Relaxed);
                let (etx, erx) = mpsc::sync_channel(1);
                let _ = etx.send(Err("ocr bsn engine is shut down".to_string()));
                return erx;
            }
        }
        rx
    }

    pub fn run_blocking(&self, job: OcrJob) -> Result<OcrOutput, String> {
        self.submit(job)
            .recv()
            .map_err(|_| "ocr bsn engine dropped the job".to_string())?
    }

    pub fn run_all(&self, jobs: Vec<OcrJob>) -> Vec<Result<OcrOutput, String>> {
        let pending: Vec<_> = jobs.into_iter().map(|j| self.submit(j)).collect();
        pending
            .into_iter()
            .map(|rx| {
                rx.recv()
                    .unwrap_or_else(|_| Err("ocr bsn engine dropped the job".to_string()))
            })
            .collect()
    }
}

impl Drop for DsocrBsnEngine {
    fn drop(&mut self) {
        self.job_tx.take();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

fn prep_loop(rx: Arc<Mutex<mpsc::Receiver<Envelope>>>, tx: mpsc::SyncSender<Prepared>) {
    loop {
        let next = {
            let guard = match rx.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            guard.recv()
        };
        let Ok(env) = next else { return };
        let queue_ms = env.submitted.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let img = match env.input {
            JobInput::Image(img) => Ok(img),
            JobInput::Bytes(b) => decode_rgb(&b),
        };
        match img.and_then(|img| prepare(&img, env.mode)) {
            Ok(prep) => {
                let item = Prepared {
                    prompt: env.prompt,
                    max_new_tokens: env.max_new_tokens,
                    prep,
                    submitted: env.submitted,
                    queue_ms,
                    prep_ms: t.elapsed().as_secs_f64() * 1e3,
                    reply: env.reply,
                };
                if tx.send(item).is_err() {
                    return;
                }
            }
            Err(e) => {
                let _ = env.reply.send(Err(format!("{e:#}")));
            }
        }
    }
}

fn front_loop(
    pipeline: Arc<DeepSeekOcr2Pipeline>,
    rx: Arc<Mutex<mpsc::Receiver<Prepared>>>,
    tx: mpsc::SyncSender<Admitted>,
    cap: usize,
) {
    loop {
        let next = {
            let guard = match rx.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            guard.recv()
        };
        let Ok(item) = next else { return };
        let t0 = Instant::now();
        let vm = item.prep.vision_tokens();
        match front_one(&pipeline, item, cap) {
            Ok(adm) => {
                if std::env::var("NV_OCR_BSN_STATS")
                    .map(|v| v != "0")
                    .unwrap_or(false)
                {
                    eprintln!(
                        "[bsn-front] page done in {:.0} ms (vision {:.0} prefill {:.0} vis_tok {vm})",
                        t0.elapsed().as_secs_f64() * 1e3,
                        adm.vision_ms,
                        adm.prefill_ms
                    );
                }
                if tx.send(adm).is_err() {
                    return;
                }
            }
            Err((reply, e)) => {
                let _ = reply.send(Err(format!("{e:#}")));
            }
        }
    }
}

fn front_one(
    pipeline: &DeepSeekOcr2Pipeline,
    item: Prepared,
    cap: usize,
) -> std::result::Result<Admitted, (Reply, anyhow::Error)> {
    let t_vis = Instant::now();
    let feats = match pipeline.vision().encode_prepared(&item.prep) {
        Ok(f) => f,
        Err(e) => return Err((item.reply, e)),
    };
    let vision_ms = t_vis.elapsed().as_secs_f64() * 1e3;
    let n_vis = item.prep.vision_tokens();
    let tokens = match build_prompt_tokens(|s| pipeline.encode_text(s), &item.prompt, n_vis) {
        Ok(t) => t,
        Err(e) => return Err((item.reply, e)),
    };
    let max_len = (tokens.len() + item.max_new_tokens).min(cap);
    let t_pf = Instant::now();
    let (cache, logits) = match DsocrBatchDecodeGraph::prefill_detached(
        pipeline.decoder(),
        &tokens,
        Some(&feats),
        max_len,
    ) {
        Ok(v) => v,
        Err(e) => return Err((item.reply, e)),
    };
    Ok(Admitted {
        prompt: item.prompt,
        max_new_tokens: item.max_new_tokens,
        tokens,
        cache,
        logits,
        feats,
        vision_tokens: n_vis,
        max_len,
        submitted: item.submitted,
        queue_ms: item.queue_ms,
        prep_ms: item.prep_ms,
        vision_ms,
        prefill_ms: t_pf.elapsed().as_secs_f64() * 1e3,
        retry: false,
        prev_best: None,
        reply: item.reply,
    })
}

fn refill_retry(pipeline: &DeepSeekOcr2Pipeline, seq: &Seq, cap: usize) -> Result<Admitted> {
    let tokens = build_prompt_tokens(
        |s| pipeline.encode_text(s),
        PROMPT_GROUNDING_MARKDOWN,
        seq.vision_tokens,
    )?;
    let max_len = (tokens.len() + seq.max_new_tokens).min(cap);
    let t_pf = Instant::now();
    let (cache, logits) = DsocrBatchDecodeGraph::prefill_detached(
        pipeline.decoder(),
        &tokens,
        Some(&seq.feats),
        max_len,
    )?;
    Ok(Admitted {
        prompt: PROMPT_GROUNDING_MARKDOWN.to_string(),
        max_new_tokens: seq.max_new_tokens,
        tokens,
        cache,
        logits,
        feats: seq.feats.clone(),
        vision_tokens: seq.vision_tokens,
        max_len,
        submitted: seq.submitted,
        queue_ms: seq.queue_ms,
        prep_ms: seq.prep_ms,
        vision_ms: seq.vision_ms,
        prefill_ms: t_pf.elapsed().as_secs_f64() * 1e3,
        retry: true,
        prev_best: seq.prev_best.clone(),
        reply: seq.reply.clone(),
    })
}

struct Finished {
    seq: Seq,
    looped: bool,
    hit_eos: bool,
}

fn decode_loop(
    pipeline: Arc<DeepSeekOcr2Pipeline>,
    adm_rx: mpsc::Receiver<Admitted>,
    opts: BsnOptions,
    inflight: Arc<AtomicUsize>,
) -> Result<()> {
    let decoder = pipeline.decoder_arc();
    let cfg_max = decoder.config().max_position_embeddings;
    let cap = opts.cap.unwrap_or(cfg_max).min(cfg_max).max(1);
    let vocab = decoder.config().vocab_size;
    let eos = decoder.config().eos_token_id;
    let mut graph = DsocrBatchDecodeGraph::new(decoder.clone(), cap, opts.buckets.clone())?;
    let max_b = graph.max_batch();
    let mut slots: Vec<Option<Seq>> = (0..max_b).map(|_| None).collect();
    let mut pending_retry: Vec<Seq> = Vec::new();
    let mut upstream_open = true;
    let base_opts = GenerateOptions::default();
    let stats = std::env::var("NV_OCR_BSN_STATS")
        .map(|v| v != "0")
        .unwrap_or(false);
    let mut hist: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    let mut live_hist: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    let mut admit_ms = 0f64;
    let mut step_ms = 0f64;
    let t_loop = Instant::now();

    loop {
        let t_admit = Instant::now();
        while let Some(slot) = graph.free_slot() {
            let mut adm: Option<Admitted> = None;
            if let Some(seq) = pending_retry.pop() {
                match refill_retry(&pipeline, &seq, cap) {
                    Ok(a) => adm = Some(a),
                    Err(e) => {
                        let _ = seq.reply.send(Err(format!("{e:#}")));
                    }
                }
            }
            if adm.is_none() && upstream_open {
                let busy = slots.iter().any(|s| s.is_some());
                let active = slots.iter().filter(|s| s.is_some()).count();
                let queued = inflight.load(Ordering::Relaxed);
                let want_fill = opts.fill_wait_ms > 0
                    && queued > 0
                    && active < opts.fill_target.min(max_b).min(active + queued);
                if !busy {
                    match adm_rx.recv() {
                        Ok(a) => adm = Some(a),
                        Err(_) => upstream_open = false,
                    }
                } else if want_fill {
                    match adm_rx.recv_timeout(std::time::Duration::from_millis(opts.fill_wait_ms)) {
                        Ok(a) => adm = Some(a),
                        Err(mpsc::RecvTimeoutError::Disconnected) => upstream_open = false,
                        Err(mpsc::RecvTimeoutError::Timeout) => break,
                    }
                } else {
                    match adm_rx.try_recv() {
                        Ok(a) => adm = Some(a),
                        Err(mpsc::TryRecvError::Disconnected) => upstream_open = false,
                        Err(mpsc::TryRecvError::Empty) => {}
                    }
                }
                if adm.is_some() {
                    inflight.fetch_sub(1, Ordering::Relaxed);
                }
            }
            let Some(a) = adm else { break };
            match install(&mut graph, slot, a, &base_opts, eos, &pipeline) {
                Ok(Some(seq)) => slots[slot] = Some(seq),
                Ok(None) => {}
                Err(e) => eprintln!("[dsocr-bsn] admit failed: {e:#}"),
            }
        }

        admit_ms += t_admit.elapsed().as_secs_f64() * 1e3;
        let extent = graph.active_extent();
        if extent == 0 {
            if !upstream_open && pending_retry.is_empty() {
                break;
            }
            continue;
        }

        let bucket = graph
            .bucket_for(extent)
            .ok_or_else(|| anyhow::anyhow!("no bucket covers {extent} slots"))?;
        let toks: Vec<Option<u32>> = (0..bucket)
            .map(|j| slots.get(j).and_then(|s| s.as_ref()).map(|s| s.next))
            .collect();
        let live = toks.iter().filter(|t| t.is_some()).count();
        if stats {
            *hist.entry(bucket).or_insert(0) += 1;
            *live_hist.entry(live).or_insert(0) += 1;
        }
        let t_step = Instant::now();
        graph.step_batch(&toks)?;
        let logits = graph.logits_batch(bucket)?;
        step_ms += t_step.elapsed().as_secs_f64() * 1e3;

        let mut finished: Vec<(usize, Finished)> = Vec::new();
        for j in 0..bucket {
            let Some(seq) = slots[j].as_mut() else {
                continue;
            };
            let mut row = logits[j * vocab..(j + 1) * vocab].to_vec();
            let step_opts = base_opts_for(seq, &base_opts);
            let next = seq.rng.next_token(&mut row, &seq.all_tokens, &step_opts)?;
            seq.generated.push(next);
            seq.all_tokens.push(next);
            seq.next = next;
            let hit_eos = next == eos;
            let done = hit_eos
                || seq.generated.len() >= seq.max_new_tokens
                || graph.slot_len(j) + 1 >= seq.max_len;
            let looped_mid = !done
                && seq.generated.len().is_multiple_of(LOOP_CHECK_STRIDE)
                && detect_loop(&seq.generated).is_some();
            if done || looped_mid {
                let taken = slots[j].take().unwrap();
                let looped = looped_mid || detect_loop(&taken.generated).is_some();
                finished.push((
                    j,
                    Finished {
                        seq: taken,
                        looped,
                        hit_eos,
                    },
                ));
            }
        }
        for (j, f) in finished {
            graph.release_slot(j);
            complete(f, &pipeline, opts.loop_retry, &mut pending_retry);
        }
    }
    if stats {
        let total: usize = hist.values().sum();
        let tokens: usize = live_hist.iter().map(|(k, v)| k * v).sum();
        eprintln!(
            "[bsn-stats] steps={total} tokens={tokens} loop_wall={:.0}ms step={:.0}ms \
             admit={:.0}ms mean_live={:.2}",
            t_loop.elapsed().as_secs_f64() * 1e3,
            step_ms,
            admit_ms,
            tokens as f64 / total.max(1) as f64
        );
        eprintln!("[bsn-stats] bucket histogram: {hist:?}");
        eprintln!("[bsn-stats] live-slot histogram: {live_hist:?}");
    }
    Ok(())
}

fn base_opts_for(seq: &Seq, base: &GenerateOptions) -> GenerateOptions {
    GenerateOptions {
        max_new_tokens: seq.max_new_tokens,
        ..base.clone()
    }
}

fn install(
    graph: &mut DsocrBatchDecodeGraph,
    slot: usize,
    a: Admitted,
    base: &GenerateOptions,
    eos: u32,
    pipeline: &DeepSeekOcr2Pipeline,
) -> Result<Option<Seq>> {
    graph.install_prefilled(slot, &a.cache)?;
    let Admitted {
        prompt,
        max_new_tokens,
        tokens,
        cache,
        mut logits,
        feats,
        vision_tokens,
        max_len,
        submitted,
        queue_ms,
        prep_ms,
        vision_ms,
        prefill_ms,
        retry,
        prev_best,
        reply,
    } = a;
    drop(cache);
    let mut rng = BatchSampler::new(base.seed);
    let opts = GenerateOptions {
        max_new_tokens,
        ..base.clone()
    };
    let mut all_tokens = tokens;
    let first = rng.next_token(&mut logits, &all_tokens, &opts)?;
    all_tokens.push(first);
    let seq = Seq {
        prompt,
        max_new_tokens,
        all_tokens,
        generated: vec![first],
        next: first,
        rng,
        max_len,
        feats,
        vision_tokens,
        submitted,
        queue_ms,
        prep_ms,
        vision_ms,
        prefill_ms,
        decode_started: Instant::now(),
        retry,
        prev_best,
        reply,
    };
    if first == eos || max_new_tokens <= 1 {
        graph.release_slot(slot);
        let looped = detect_loop(&seq.generated).is_some();
        emit(seq, looped, first == eos, Some(pipeline));
        return Ok(None);
    }
    Ok(Some(seq))
}

fn complete(
    f: Finished,
    pipeline: &DeepSeekOcr2Pipeline,
    loop_retry: bool,
    pending: &mut Vec<Seq>,
) {
    let Finished {
        mut seq,
        looped,
        hit_eos,
    } = f;
    if looped {
        if let Some(d) = detect_loop(&seq.generated) {
            seq.generated.truncate(d.onset);
        }
        if loop_retry && !seq.retry && seq.prompt.trim() == PROMPT_FREE_OCR.trim() {
            seq.prev_best = Some(seq.generated.clone());
            pending.push(seq);
            return;
        }
    }
    emit(seq, looped, hit_eos, Some(pipeline));
}

fn emit(seq: Seq, looped: bool, hit_eos: bool, pipeline: Option<&DeepSeekOcr2Pipeline>) {
    let mut tokens = if seq.retry {
        strip_grounding_tokens(&seq.generated)
    } else {
        seq.generated.clone()
    };
    if let Some(prev) = seq.prev_best.as_ref() {
        if prev.len() > tokens.len() {
            tokens = prev.clone();
        }
    }
    let text = match pipeline {
        Some(p) => match p.tokenizer().decode(&tokens, true) {
            Ok(t) => t,
            Err(e) => {
                let _ = seq.reply.send(Err(format!("tokenizer decode: {e}")));
                return;
            }
        },
        None => String::new(),
    };
    let timings = JobTimings {
        queue_ms: seq.queue_ms,
        prep_ms: seq.prep_ms,
        vision_ms: seq.vision_ms,
        decode_ms: seq.decode_started.elapsed().as_secs_f64() * 1e3 + seq.prefill_ms,
        total_ms: seq.submitted.elapsed().as_secs_f64() * 1e3,
        out_tokens: tokens.len(),
        vision_tokens: seq.vision_tokens,
    };
    let _ = seq.reply.send(Ok(OcrOutput {
        text,
        tokens,
        looped,
        hit_eos,
        timings,
    }));
}
