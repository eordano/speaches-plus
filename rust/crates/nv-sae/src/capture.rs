use anyhow::{ensure, Context, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::records::Record;

pub const CAPTURE_BATCH_RECORDS: usize = 32;
pub const CAPTURE_BATCH_RECORDS_WITH_TOKENS: usize = 8;
pub const CAPTURE_MAX_TOKENS: usize = 2048;

#[derive(Serialize)]
struct CaptureRequest<'a> {
    layer: usize,
    texts: &'a [String],
    tokens: bool,
    max_context: usize,
}

#[derive(Deserialize)]
pub struct CaptureRecord {
    pub tokens: usize,
    pub input_token_ids: Vec<u32>,
    pub mean: Vec<f32>,
    pub last: Vec<f32>,
    #[serde(default)]
    pub token_states: Option<Vec<Vec<f32>>>,
}

#[derive(Deserialize)]
pub struct CaptureResponse {
    pub layer: usize,
    pub hidden_size: usize,
    pub records: Vec<CaptureRecord>,
    pub seconds: f64,
}

pub struct Client {
    pub url: String,
    http: reqwest::blocking::Client,
}

impl Client {
    pub fn new(url: &str) -> Result<Self> {
        Ok(Client {
            url: url.trim_end_matches('/').to_string(),
            http: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(3600))
                .build()?,
        })
    }

    pub fn capture_wait(
        &self,
        texts: &[String],
        layer: usize,
        tokens: bool,
    ) -> Result<CaptureResponse> {
        let mut waited = 0u64;
        loop {
            match self.capture(texts, layer, tokens) {
                Err(e) if is_transient(&e) && waited < CAPTURE_WAIT_MAX_SECS => {
                    eprintln!(
                        "capture route unavailable ({e:#}); retrying in {CAPTURE_WAIT_STEP_SECS}s ({waited}s waited)"
                    );
                    std::thread::sleep(std::time::Duration::from_secs(CAPTURE_WAIT_STEP_SECS));
                    waited += CAPTURE_WAIT_STEP_SECS;
                }
                r => return r,
            }
        }
    }

    pub fn capture(&self, texts: &[String], layer: usize, tokens: bool) -> Result<CaptureResponse> {
        let req = CaptureRequest {
            layer,
            texts,
            tokens,
            max_context: CAPTURE_MAX_TOKENS,
        };
        let resp = self
            .http
            .post(format!("{}/v1/laguna/capture", self.url))
            .json(&req)
            .send()?;
        let status = resp.status();
        let body = resp.text()?;
        ensure!(
            status.is_success(),
            "capture {status}: {}",
            body.chars().take(400).collect::<String>()
        );
        Ok(serde_json::from_str(&body).context("capture response")?)
    }
}

pub const CAPTURE_WAIT_STEP_SECS: u64 = 30;
pub const CAPTURE_WAIT_MAX_SECS: u64 = 3600;

fn is_transient(e: &anyhow::Error) -> bool {
    let m = format!("{e:#}");
    m.contains("error sending request")
        || m.contains("capture 502 ")
        || m.contains("capture 503 ")
        || m.contains("capture 504 ")
}

#[derive(Serialize, Deserialize)]
pub struct CaptureManifest {
    pub layer: usize,
    pub hidden_size: usize,
    pub records: usize,
    pub tokens: usize,
    pub seconds: f64,
    pub url: String,
    pub skipped: usize,
    pub token_states: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Span {
    pub record: usize,
    pub start: usize,
    pub end: usize,
    pub ids: Vec<u32>,
}

struct Acc<'a> {
    layer: usize,
    with_tokens: bool,
    means: Vec<f32>,
    lasts: Vec<f32>,
    token_states: Vec<f32>,
    spans: Vec<Span>,
    kept: Vec<&'a Record>,
    token_counts: Vec<u32>,
    hidden: usize,
    seconds: f64,
    total_tokens: usize,
    skipped: usize,
}

impl<'a> Acc<'a> {
    fn ingest(&mut self, chunk: &[&'a Record], resp: CaptureResponse) -> Result<()> {
        ensure!(
            resp.layer == self.layer,
            "server captured layer {} instead of {}",
            resp.layer,
            self.layer
        );
        ensure!(
            resp.records.len() == chunk.len(),
            "server returned {} records for {}",
            resp.records.len(),
            chunk.len()
        );
        self.hidden = resp.hidden_size;
        self.seconds += resp.seconds;
        for (r, c) in chunk.iter().zip(resp.records.into_iter()) {
            self.means.extend_from_slice(&c.mean);
            self.lasts.extend_from_slice(&c.last);
            self.token_counts.push(c.tokens as u32);
            if self.with_tokens {
                let states = c.token_states.context("server omitted token states")?;
                ensure!(states.len() == c.tokens, "token state count");
                let start = self.total_tokens;
                for row in &states {
                    self.token_states.extend_from_slice(row);
                }
                self.spans.push(Span {
                    record: self.kept.len(),
                    start,
                    end: start + c.tokens,
                    ids: c.input_token_ids,
                });
            }
            self.total_tokens += c.tokens;
            self.kept.push(r);
        }
        Ok(())
    }
}

pub struct Pace {
    pub batch: usize,
    pub pause: std::time::Duration,
    pub deadline: Option<std::time::Instant>,
}

impl Pace {
    pub fn steady() -> Self {
        Pace {
            batch: CAPTURE_BATCH_RECORDS,
            pause: std::time::Duration::ZERO,
            deadline: None,
        }
    }
}

pub fn capture_records(
    client: &Client,
    records: &[Record],
    layer: usize,
    out: &Path,
    tokenizer: Option<&tokenizers::Tokenizer>,
    with_tokens: bool,
    pace: &Pace,
) -> Result<CaptureManifest> {
    std::fs::create_dir_all(out)?;
    let candidates: Vec<&Record> = records
        .iter()
        .filter(|r| match tokenizer {
            Some(t) => t
                .encode(r.text.as_str(), false)
                .map(|e| e.len() <= CAPTURE_MAX_TOKENS && !e.is_empty())
                .unwrap_or(false),
            None => true,
        })
        .collect();
    let mut acc = Acc {
        layer,
        with_tokens,
        means: Vec::new(),
        lasts: Vec::new(),
        token_states: Vec::new(),
        spans: Vec::new(),
        kept: Vec::new(),
        token_counts: Vec::new(),
        hidden: 0,
        seconds: 0.0,
        total_tokens: 0,
        skipped: records.len() - candidates.len(),
    };
    let batch = pace.batch.max(1);
    let mut left_for_next_run = 0usize;
    for chunk in candidates.chunks(batch) {
        if let Some(d) = pace.deadline {
            if std::time::Instant::now() >= d {
                left_for_next_run += chunk.len();
                continue;
            }
        }
        if !pace.pause.is_zero() && !acc.kept.is_empty() {
            std::thread::sleep(pace.pause);
        }
        let texts: Vec<String> = chunk.iter().map(|r| r.text.clone()).collect();
        match client.capture_wait(&texts, layer, with_tokens) {
            Ok(resp) => acc.ingest(chunk, resp)?,
            Err(e) => {
                eprintln!("batch failed ({e:#}); retrying one by one");
                for (r, t) in chunk.iter().zip(texts.iter()) {
                    match client.capture_wait(std::slice::from_ref(t), layer, with_tokens) {
                        Ok(one) => acc.ingest(std::slice::from_ref(r), one)?,
                        Err(e) => {
                            eprintln!("skip {} ({e:#})", r.id);
                            acc.skipped += 1;
                        }
                    }
                }
            }
        }
        eprintln!(
            "captured {} / {} records, {} tokens ({:.1}s)",
            acc.kept.len(),
            candidates.len(),
            acc.total_tokens,
            acc.seconds
        );
    }
    if left_for_next_run > 0 {
        eprintln!(
            "deadline reached: {} of {} records left for the next run",
            left_for_next_run,
            candidates.len()
        );
    }
    ensure!(!acc.kept.is_empty(), "nothing captured");
    let n = acc.kept.len();
    let hidden = acc.hidden;
    let dev = Device::Cpu;
    let mut m = HashMap::new();
    m.insert(
        "means".to_string(),
        Tensor::from_vec(acc.means, (n, hidden), &dev)?,
    );
    m.insert(
        "last".to_string(),
        Tensor::from_vec(acc.lasts, (n, hidden), &dev)?,
    );
    m.insert(
        "tokens".to_string(),
        Tensor::new(acc.token_counts.as_slice(), &dev)?,
    );
    candle_core::safetensors::save(&m, out.join("states.safetensors"))?;
    if with_tokens {
        let mut t = HashMap::new();
        t.insert(
            "tokens".to_string(),
            Tensor::from_vec(acc.token_states, (acc.total_tokens, hidden), &dev)?,
        );
        candle_core::safetensors::save(&t, out.join("tokens.safetensors"))?;
        std::fs::write(out.join("spans.json"), serde_json::to_string(&acc.spans)?)?;
    }
    let mut f = std::fs::File::create(out.join("records.jsonl"))?;
    for r in &acc.kept {
        serde_json::to_writer(&mut f, r)?;
        std::io::Write::write_all(&mut f, b"\n")?;
    }
    let manifest = CaptureManifest {
        layer,
        hidden_size: hidden,
        records: n,
        tokens: acc.total_tokens,
        seconds: acc.seconds,
        url: client.url.clone(),
        skipped: acc.skipped,
        token_states: with_tokens,
    };
    std::fs::write(
        out.join("capture.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

pub fn load_spans(path: &Path) -> Result<Vec<Span>> {
    Ok(serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| path.display().to_string())?,
    )?)
}

pub fn merge(dirs: &[std::path::PathBuf], out: &Path, since: Option<&str>) -> Result<usize> {
    std::fs::create_dir_all(out)?;
    let dev = Device::Cpu;
    let mut means = Vec::new();
    let mut lasts = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    let mut records = String::new();
    let mut hidden = 0;
    let mut seen = std::collections::HashSet::new();
    for d in dirs {
        let recs = match std::fs::read_to_string(d.join("records.jsonl")) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e).with_context(|| d.display().to_string()),
        };
        let lines: Vec<&str> = recs.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            continue;
        }
        let st = candle_core::safetensors::load(d.join("states.safetensors"), &dev)?;
        let m = st["means"].to_vec2::<f32>()?;
        let l = st["last"].to_vec2::<f32>()?;
        let t = st["tokens"].to_vec1::<u32>()?;
        ensure!(
            m.len() == lines.len() && l.len() == lines.len() && t.len() == lines.len(),
            "{}: records and states disagree",
            d.display()
        );
        for (i, line) in lines.iter().enumerate() {
            let r: Record = serde_json::from_str(line)?;
            if since.is_some_and(|s| r.time.as_str() < s) || !seen.insert(r.id.clone()) {
                continue;
            }
            hidden = m[i].len();
            means.extend_from_slice(&m[i]);
            lasts.extend_from_slice(&l[i]);
            counts.push(t[i]);
            records.push_str(line);
            records.push('\n');
        }
    }
    let n = counts.len();
    ensure!(n > 0, "nothing to merge");
    let mut st = HashMap::new();
    st.insert(
        "means".to_string(),
        Tensor::from_vec(means, (n, hidden), &dev)?,
    );
    st.insert(
        "last".to_string(),
        Tensor::from_vec(lasts, (n, hidden), &dev)?,
    );
    st.insert("tokens".to_string(), Tensor::new(counts.as_slice(), &dev)?);
    candle_core::safetensors::save(&st, out.join("states.safetensors"))?;
    std::fs::write(out.join("records.jsonl"), records)?;
    Ok(n)
}

pub fn stream_batches(
    client: &Client,
    records: &[Record],
    layer: usize,
    tokenizer: Option<&tokenizers::Tokenizer>,
    batch: usize,
    mut f: impl FnMut(&[&Record], CaptureResponse) -> Result<()>,
) -> Result<usize> {
    let candidates: Vec<&Record> = records
        .iter()
        .filter(|r| match tokenizer {
            Some(t) => t
                .encode(r.text.as_str(), false)
                .map(|e| e.len() <= CAPTURE_MAX_TOKENS && !e.is_empty())
                .unwrap_or(false),
            None => true,
        })
        .collect();
    let mut done = 0;
    for chunk in candidates.chunks(batch) {
        let texts: Vec<String> = chunk.iter().map(|r| r.text.clone()).collect();
        match client.capture_wait(&texts, layer, true) {
            Ok(resp) => {
                ensure!(
                    resp.records.len() == chunk.len(),
                    "server returned {} records for {}",
                    resp.records.len(),
                    chunk.len()
                );
                done += chunk.len();
                f(chunk, resp)?;
            }
            Err(e) => eprintln!("batch skipped ({e:#})"),
        }
    }
    Ok(done)
}
