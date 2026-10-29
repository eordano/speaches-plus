use anyhow::{ensure, Context, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::records::Record;

pub const CAPTURE_BATCH_RECORDS: usize = 32;
pub const CAPTURE_BATCH_RECORDS_WITH_TOKENS: usize = 8;
pub const CAPTURE_MAX_TOKENS: usize = 2048;
pub const CAPTURE_GIVES_UP_AFTER_8_CONSECUTIVE_RECORD_FAILURES: usize = 8;

#[derive(Serialize)]
struct CaptureRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
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
    #[serde(default)]
    pub model: String,
    pub layer: usize,
    #[serde(default)]
    pub num_layers: usize,
    pub hidden_size: usize,
    pub records: Vec<CaptureRecord>,
    pub seconds: f64,
}

pub struct Client {
    pub url: String,
    pub model: Option<String>,
    http: reqwest::blocking::Client,
}

impl Client {
    pub fn new(url: &str, model: Option<&str>) -> Result<Self> {
        Ok(Client {
            url: url.trim_end_matches('/').to_string(),
            model: model.filter(|m| !m.is_empty()).map(str::to_string),
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
            model: self.model.as_deref(),
            layer,
            texts,
            tokens,
            max_context: CAPTURE_MAX_TOKENS,
        };
        let resp = self
            .http
            .post(format!("{}/v1/capture", self.url))
            .json(&req)
            .send()?;
        let status = resp.status();
        let body = resp.text()?;
        ensure!(
            status.is_success(),
            "capture {status}: {}",
            body.chars().take(400).collect::<String>()
        );
        let out: CaptureResponse = serde_json::from_str(&body).context("capture response")?;
        if let Some(m) = &self.model {
            ensure!(
                &out.model == m,
                "capture refused: asked {m} but the engine answered as {}",
                out.model
            );
        }
        Ok(out)
    }
}

pub fn is_refusal(e: &anyhow::Error) -> bool {
    let m = format!("{e:#}");
    m.contains("capture refused")
        || m.contains("capture 404 ")
        || m.contains("capture 501 ")
        || m.contains("out of range for")
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
    #[serde(default)]
    pub model: String,
    pub layer: usize,
    #[serde(default)]
    pub num_layers: usize,
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
    model: String,
    num_layers: usize,
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
        ensure!(
            self.kept.is_empty() || (resp.model == self.model && resp.hidden_size == self.hidden),
            "capture refused: the engine switched from {} (hidden {}) to {} (hidden {}) mid-run",
            self.model,
            self.hidden,
            resp.model,
            resp.hidden_size
        );
        self.model = resp.model;
        self.num_layers = resp.num_layers;
        self.hidden = resp.hidden_size;
        self.seconds += resp.seconds;
        for (r, c) in chunk.iter().zip(resp.records) {
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
        model: String::new(),
        num_layers: 0,
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
    let mut failing = 0usize;
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
            Ok(resp) => {
                acc.ingest(chunk, resp)?;
                failing = 0;
            }
            Err(e) if is_refusal(&e) => {
                anyhow::bail!("capture refused by {} at layer {layer}: {e:#}", client.url)
            }
            Err(e) => {
                eprintln!("batch failed ({e:#}); retrying one by one");
                for (r, t) in chunk.iter().zip(texts.iter()) {
                    match client.capture_wait(std::slice::from_ref(t), layer, with_tokens) {
                        Ok(one) => {
                            acc.ingest(std::slice::from_ref(r), one)?;
                            failing = 0;
                        }
                        Err(e) if is_refusal(&e) => anyhow::bail!(
                            "capture refused by {} at layer {layer}: {e:#}",
                            client.url
                        ),
                        Err(e) => {
                            eprintln!("skip {} ({e:#})", r.id);
                            acc.skipped += 1;
                            failing += 1;
                            ensure!(
                                failing < CAPTURE_GIVES_UP_AFTER_8_CONSECUTIVE_RECORD_FAILURES,
                                "capture failing: {failing} consecutive records failed at {}, last: {e:#}",
                                client.url
                            );
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
        model: acc.model.clone(),
        num_layers: acc.num_layers,
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
    let mut identity: Option<(String, usize)> = None;
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
        let here = read_manifest(d)
            .map(|m| (m.model, m.layer))
            .with_context(|| format!("{}: capture.json", d.display()))?;
        match &identity {
            None => identity = Some(here),
            Some(first) => ensure!(
                *first == here,
                "{}: captured {} layer {} but the merge started on {} layer {}; dictionaries are per model and layer",
                d.display(),
                here.0,
                here.1,
                first.0,
                first.1
            ),
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
            ensure!(
                hidden == 0 || hidden == m[i].len(),
                "{}: hidden size {} after {hidden}",
                d.display(),
                m[i].len()
            );
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
    let (model, layer) = identity.unwrap_or_default();
    std::fs::write(
        out.join("capture.json"),
        serde_json::to_string_pretty(&CaptureManifest {
            model,
            layer,
            num_layers: 0,
            hidden_size: hidden,
            records: n,
            tokens: 0,
            seconds: 0.0,
            url: String::new(),
            skipped: 0,
            token_states: false,
        })?,
    )?;
    Ok(n)
}

pub const LEGACY_CAPTURE_MODEL_IS_LAGUNA_BECAUSE_CAPTURE_JSON_PREDATES_THE_MODEL_FIELD: &str =
    "poolside/Laguna-XS-2.1-NVFP4";

pub fn read_manifest(dir: &Path) -> Result<CaptureManifest> {
    let mut m: CaptureManifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("capture.json"))?)?;
    if m.model.is_empty() {
        m.model =
            LEGACY_CAPTURE_MODEL_IS_LAGUNA_BECAUSE_CAPTURE_JSON_PREDATES_THE_MODEL_FIELD.into();
    }
    Ok(m)
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
            Err(e) if is_refusal(&e) => {
                anyhow::bail!("capture refused by {} at layer {layer}: {e:#}", client.url)
            }
            Err(e) => eprintln!("batch skipped ({e:#})"),
        }
    }
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn night(
        root: &Path,
        name: &str,
        model: Option<&str>,
        hidden: usize,
        id: &str,
    ) -> std::path::PathBuf {
        let d = root.join(name);
        std::fs::create_dir_all(&d).unwrap();
        let rec = serde_json::json!({"id": id, "kind": "user", "agent": "claude", "host": "h", "session": "s", "time": "2026-10-10T00:00:00Z", "source": "p", "line": 1, "text": "t"});
        std::fs::write(d.join("records.jsonl"), format!("{rec}\n")).unwrap();
        let mut st = HashMap::new();
        for key in ["means", "last"] {
            st.insert(
                key.to_string(),
                Tensor::zeros((1, hidden), candle_core::DType::F32, &Device::Cpu).unwrap(),
            );
        }
        st.insert(
            "tokens".to_string(),
            Tensor::new(&[3u32], &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&st, d.join("states.safetensors")).unwrap();
        let mut m = serde_json::json!({"layer": 19, "hidden_size": hidden, "records": 1, "tokens": 3, "seconds": 0.0, "url": "", "skipped": 0, "token_states": false});
        if let Some(model) = model {
            m["model"] = serde_json::json!(model);
        }
        std::fs::write(d.join("capture.json"), m.to_string()).unwrap();
        d
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("nv-sae-merge-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn merge_keeps_one_model_and_reads_legacy_manifests_as_laguna() {
        let root = scratch("legacy");
        let a = night(&root, "a", None, 4, "r1");
        let b = night(
            &root,
            "b",
            Some(LEGACY_CAPTURE_MODEL_IS_LAGUNA_BECAUSE_CAPTURE_JSON_PREDATES_THE_MODEL_FIELD),
            4,
            "r2",
        );
        let out = root.join("out");
        assert_eq!(merge(&[a, b], &out, None).unwrap(), 2);
        let m = read_manifest(&out).unwrap();
        assert_eq!(
            (m.model.as_str(), m.layer, m.hidden_size),
            ("poolside/Laguna-XS-2.1-NVFP4", 19, 4)
        );
    }

    #[test]
    fn merge_refuses_captures_of_another_model() {
        let root = scratch("mixed");
        let a = night(&root, "a", None, 4, "r1");
        let b = night(&root, "b", Some("nvidia/Qwen3.8-Flash-Next-NVFP4"), 4, "r2");
        let err = merge(&[a, b], &root.join("out"), None).unwrap_err();
        assert!(
            format!("{err:#}").contains("dictionaries are per model and layer"),
            "{err:#}"
        );
    }

    #[test]
    fn refusals_stop_the_run_while_transient_failures_retry() {
        for m in [
            "capture 501 Not Implemented: {}",
            "capture 404 Not Found: {}",
            "capture refused: asked a but the engine answered as b",
        ] {
            assert!(is_refusal(&anyhow::anyhow!("{m}")), "{m}");
        }
        for m in [
            "capture 400 Bad Request: each input must hold 1..=2048 tokens",
            "capture 503 Service Unavailable: busy",
        ] {
            assert!(!is_refusal(&anyhow::anyhow!("{m}")), "{m}");
        }
    }
}
