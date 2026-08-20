use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

use anyhow::Result;
use nv_models::deepseek_ocr::decoder::{strip_grounding_tokens, GenerateOutcome};
use nv_models::deepseek_ocr::preprocess::{prepare, PreparedViews};
use nv_models::deepseek_ocr::{
    build_prompt_tokens, DeepSeekOcr2Pipeline, GenerateOptions, ResolutionMode, RgbImage,
    PROMPT_FREE_OCR, PROMPT_GROUNDING_MARKDOWN,
};

#[cfg(feature = "cuda")]
use nv_models::deepseek_ocr::DsocrDecodeGraph;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Clone, Copy, Debug)]
pub struct BatchOptions {
    pub slots: usize,
    pub prep_threads: usize,
    pub queue_depth: usize,
    pub loop_retry: bool,
}

impl BatchOptions {
    pub fn from_env() -> Self {
        let slots = env_usize("NV_OCR_BATCH_SLOTS", 1).max(1);
        let prep_threads = env_usize("NV_OCR_BATCH_PREP_THREADS", slots).max(1);
        Self {
            slots,
            prep_threads,
            queue_depth: env_usize("NV_OCR_BATCH_QUEUE", 64).max(1),
            loop_retry: std::env::var("NV_DSOCR_LOOP_RETRY")
                .map(|v| v != "0")
                .unwrap_or(true),
        }
    }
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            slots: 1,
            prep_threads: 1,
            queue_depth: 64,
            loop_retry: true,
        }
    }
}

pub enum JobInput {
    Bytes(Vec<u8>),
    Image(RgbImage),
}

pub struct OcrJob {
    pub input: JobInput,
    pub prompt: String,
    pub mode: ResolutionMode,
    pub max_new_tokens: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct JobTimings {
    pub queue_ms: f64,
    pub prep_ms: f64,
    pub vision_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub out_tokens: usize,
    pub vision_tokens: usize,
}

pub struct OcrOutput {
    pub text: String,
    pub tokens: Vec<u32>,
    pub looped: bool,
    pub hit_eos: bool,
    pub timings: JobTimings,
}

type Reply = mpsc::SyncSender<Result<OcrOutput, String>>;

struct Envelope {
    prompt: String,
    mode: ResolutionMode,
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

pub fn bsn_enabled() -> bool {
    match std::env::var("NV_OCR_BSN") {
        Ok(v) if !v.is_empty() => v != "0",
        _ => true,
    }
}

pub struct DsocrScheduler {
    job_tx: Option<mpsc::SyncSender<Envelope>>,
    threads: Vec<thread::JoinHandle<()>>,
    opts: BatchOptions,
    #[cfg(feature = "cuda")]
    bsn: Option<super::ocr_batch_n::DsocrBsnEngine>,
}

impl DsocrScheduler {
    pub fn new(pipeline: Arc<DeepSeekOcr2Pipeline>, opts: BatchOptions) -> Self {
        #[cfg(feature = "cuda")]
        if bsn_enabled() {
            let bopts = super::ocr_batch_n::BsnOptions::from_env();
            eprintln!(
                "[dsocr-batch] routing to the bs=N engine (buckets {:?}, front {}); NV_OCR_BSN=0 to opt out",
                bopts.buckets, bopts.front_threads
            );
            return Self {
                job_tx: None,
                threads: Vec::new(),
                opts,
                bsn: Some(super::ocr_batch_n::DsocrBsnEngine::new(pipeline, bopts)),
            };
        }
        let (job_tx, job_rx) = mpsc::sync_channel::<Envelope>(opts.queue_depth);
        let (prep_tx, prep_rx) = mpsc::sync_channel::<Prepared>(opts.queue_depth);
        let job_rx = Arc::new(Mutex::new(job_rx));
        let prep_rx = Arc::new(Mutex::new(prep_rx));
        let mut threads = Vec::with_capacity(opts.prep_threads + opts.slots);
        for i in 0..opts.prep_threads {
            let rx = job_rx.clone();
            let tx = prep_tx.clone();
            threads.push(
                thread::Builder::new()
                    .name(format!("dsocr-prep-{i}"))
                    .spawn(move || prep_loop(rx, tx))
                    .expect("spawn dsocr prep thread"),
            );
        }
        drop(prep_tx);
        for i in 0..opts.slots {
            let rx = prep_rx.clone();
            let pipe = pipeline.clone();
            threads.push(
                thread::Builder::new()
                    .name(format!("dsocr-gpu-{i}"))
                    .spawn(move || gpu_loop(pipe, rx, opts))
                    .expect("spawn dsocr gpu thread"),
            );
        }
        Self {
            job_tx: Some(job_tx),
            threads,
            opts,
            #[cfg(feature = "cuda")]
            bsn: None,
        }
    }

    pub fn options(&self) -> BatchOptions {
        self.opts
    }

    pub fn submit(&self, job: OcrJob) -> mpsc::Receiver<Result<OcrOutput, String>> {
        #[cfg(feature = "cuda")]
        if let Some(e) = self.bsn.as_ref() {
            return e.submit(job);
        }
        let (tx, rx) = mpsc::sync_channel(1);
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
                let (etx, erx) = mpsc::sync_channel(1);
                let _ = etx.send(Err("ocr scheduler is shut down".to_string()));
                return erx;
            }
        }
        rx
    }

    pub fn run_blocking(&self, job: OcrJob) -> Result<OcrOutput, String> {
        self.submit(job)
            .recv()
            .map_err(|_| "ocr scheduler dropped the job".to_string())?
    }

    pub fn run_all(&self, jobs: Vec<OcrJob>) -> Vec<Result<OcrOutput, String>> {
        let pending: Vec<_> = jobs.into_iter().map(|j| self.submit(j)).collect();
        pending
            .into_iter()
            .map(|rx| {
                rx.recv()
                    .unwrap_or_else(|_| Err("ocr scheduler dropped the job".to_string()))
            })
            .collect()
    }
}

impl Drop for DsocrScheduler {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            self.bsn.take();
        }
        self.job_tx.take();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

pub fn decode_rgb(bytes: &[u8]) -> Result<RgbImage> {
    RgbImage::decode(bytes)
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
        let prepared = img.and_then(|img| prepare(&img, env.mode));
        match prepared {
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

#[cfg(feature = "cuda")]
struct GraphSlot {
    graph: Option<DsocrDecodeGraph>,
    failed: bool,
}

#[cfg(feature = "cuda")]
impl GraphSlot {
    fn new() -> Self {
        Self {
            graph: None,
            failed: false,
        }
    }
}

#[cfg(not(feature = "cuda"))]
struct GraphSlot;

#[cfg(not(feature = "cuda"))]
impl GraphSlot {
    fn new() -> Self {
        Self
    }
}

fn gpu_loop(
    pipeline: Arc<DeepSeekOcr2Pipeline>,
    rx: Arc<Mutex<mpsc::Receiver<Prepared>>>,
    opts: BatchOptions,
) {
    let mut slot = GraphSlot::new();
    let mut first = true;
    loop {
        let next = {
            let guard = match rx.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            guard.recv()
        };
        let Ok(item) = next else { return };
        let out = if first {
            let _g = FIRST_JOB.lock();
            first = false;
            run_one(&pipeline, &mut slot, &item, &opts)
        } else {
            run_one(&pipeline, &mut slot, &item, &opts)
        };
        let _ = item.reply.send(out.map_err(|e| format!("{e:#}")));
    }
}

static FIRST_JOB: Mutex<()> = Mutex::new(());

fn run_one(
    pipeline: &DeepSeekOcr2Pipeline,
    slot: &mut GraphSlot,
    item: &Prepared,
    opts: &BatchOptions,
) -> Result<OcrOutput> {
    let t_vis = Instant::now();
    let feats = pipeline.vision().encode_prepared(&item.prep)?;
    let vision_ms = t_vis.elapsed().as_secs_f64() * 1e3;
    let n_vis = item.prep.vision_tokens();
    let gopts = GenerateOptions {
        max_new_tokens: item.max_new_tokens,
        ..Default::default()
    };

    let t_dec = Instant::now();
    let outcome = decode_once(pipeline, slot, &feats, n_vis, &item.prompt, &gopts)?;
    let mut out = outcome.tokens;
    let mut hit_eos = outcome.hit_eos;
    let mut looped = false;
    if let Some(d) = outcome.loop_detection {
        looped = true;
        out.truncate(d.onset);
        if item.prompt.trim() == PROMPT_FREE_OCR && opts.loop_retry {
            let retry_outcome = decode_once(
                pipeline,
                slot,
                &feats,
                n_vis,
                PROMPT_GROUNDING_MARKDOWN,
                &gopts,
            )?;
            let mut retry = retry_outcome.tokens;
            if let Some(rd) = retry_outcome.loop_detection {
                retry.truncate(rd.onset);
            }
            let retry = strip_grounding_tokens(&retry);
            if retry.len() > out.len() {
                out = retry;
                hit_eos = retry_outcome.hit_eos;
            }
        }
    }
    let decode_ms = t_dec.elapsed().as_secs_f64() * 1e3;
    if item.prompt.trim() == PROMPT_GROUNDING_MARKDOWN.trim() {
        out = strip_grounding_tokens(&out);
    }
    let text = pipeline
        .tokenizer()
        .decode(&out, true)
        .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;
    let timings = JobTimings {
        queue_ms: item.queue_ms,
        prep_ms: item.prep_ms,
        vision_ms,
        decode_ms,
        total_ms: item.submitted.elapsed().as_secs_f64() * 1e3,
        out_tokens: out.len(),
        vision_tokens: n_vis,
    };
    Ok(OcrOutput {
        text,
        tokens: out,
        looped,
        hit_eos,
        timings,
    })
}

fn decode_once(
    pipeline: &DeepSeekOcr2Pipeline,
    slot: &mut GraphSlot,
    feats: &candle_core::Tensor,
    n_vis: usize,
    prompt: &str,
    opts: &GenerateOptions,
) -> Result<GenerateOutcome> {
    let tokens = build_prompt_tokens(|s| pipeline.encode_text(s), prompt, n_vis)?;
    #[cfg(feature = "cuda")]
    {
        if nv_models::deepseek_ocr::decoder_graph::graph_enabled()
            && pipeline.device().is_cuda()
            && !slot.failed
        {
            if slot.graph.is_none() {
                let cap = pipeline.decoder().config().max_position_embeddings;
                match DsocrDecodeGraph::new(pipeline.decoder_arc(), cap) {
                    Ok(g) => slot.graph = Some(g),
                    Err(e) => {
                        slot.failed = true;
                        eprintln!("[dsocr-batch] graph init failed, falling back to eager: {e:#}");
                    }
                }
            }
            if let Some(g) = slot.graph.as_mut() {
                return g.generate(&tokens, Some(feats), opts);
            }
        }
    }
    let _ = slot;
    pipeline
        .decoder()
        .generate_detected(&tokens, Some(feats), opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_single_slot() {
        let o = BatchOptions::default();
        assert_eq!(o.slots, 1);
        assert_eq!(o.prep_threads, 1);
    }

    #[test]
    fn bsn_defaults_on_and_zero_opts_out() {
        let prev = std::env::var("NV_OCR_BSN").ok();
        std::env::remove_var("NV_OCR_BSN");
        assert!(bsn_enabled());
        std::env::set_var("NV_OCR_BSN", "");
        assert!(bsn_enabled());
        std::env::set_var("NV_OCR_BSN", "0");
        assert!(!bsn_enabled());
        std::env::set_var("NV_OCR_BSN", "1");
        assert!(bsn_enabled());
        match prev {
            Some(v) => std::env::set_var("NV_OCR_BSN", v),
            None => std::env::remove_var("NV_OCR_BSN"),
        }
    }

    #[test]
    fn env_options_clamp_to_at_least_one() {
        std::env::set_var("NV_OCR_BATCH_SLOTS", "0");
        std::env::set_var("NV_OCR_BATCH_PREP_THREADS", "0");
        let o = BatchOptions::from_env();
        std::env::remove_var("NV_OCR_BATCH_SLOTS");
        std::env::remove_var("NV_OCR_BATCH_PREP_THREADS");
        assert_eq!(o.slots, 1);
        assert_eq!(o.prep_threads, 1);
    }
}
