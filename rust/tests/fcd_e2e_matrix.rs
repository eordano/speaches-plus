#![cfg(feature = "cuda")]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const SERVER_BIN: &str = env!("CARGO_BIN_EXE_speaches-plus");

const FULL_RUN_GATE_ENV: &str = "NV_FCD_E2E_TEST";

const DRY_RUN_PRINTS_THE_CELL_PLAN_WITHOUT_GPU_OR_WEIGHTS_ENV: &str = "NV_FCD_E2E_DRYRUN";

const MAX_ADMITTED_SEQ_ENV_DEFAULT_196608_IS_THE_PROVEN_C196_RECORD_DEPTH: &str =
    "NV_FCD_E2E_MAX_SEQ";

const DEFAULT_MAX_ADMITTED_SEQ: usize = 196_608;

const DEPTH_HEADROOM_KEEPS_PROMPT_PLUS_GENERATION_UNDER_THE_KV_CEILING: usize = 4096;

const ARTIFACT_DIR_ENV_DEFAULT_HOME_TMP_WF_FCD_E2E_BECAUSE_ROOT_TMP_IS_TMPFS: &str =
    "NV_FCD_E2E_ARTIFACT_DIR";

const QWEN38_DIR_ENV: &str = "NV_QWEN38_DIR";

const BOOT_DEADLINE_COVERS_A_22GIB_WEIGHT_LOAD_PLUS_196K_KV_ALLOC: Duration =
    Duration::from_secs(1800);

const DEEP_PREFILL_READ_TIMEOUT: Duration = Duration::from_secs(1800);

const SHORT_CELL_READ_TIMEOUT: Duration = Duration::from_secs(600);

const DEPTH_LADDER_TOKENS_ACHIEVED_MUST_BE_WITHIN_15_PCT_OF_TARGET: f64 = 0.15;

const NEEDLE_RECALL_IS_ASSERTED_ONLY_UP_TO_8192_DEEPER_CELLS_RECORD_IT: u32 = 8192;

const MAX_RUNG_DESCENT_FRACTIONS_FIND_THE_SERVED_CEILING_WHEN_PREFILL_OOMS_LOUDLY_INSIDE_THE_KV_WINDOW:
    [f64; 6] = [1.0, 0.85, 0.68, 0.51, 0.44, 0.38];

const A_LOUD_DEPTH_REFUSAL_NAMES_ONE_OF_THESE_OR_THE_DESCENT_IS_HIDING_A_SILENT_FAILURE: [&str; 4] = [
    "context_length_exceeded",
    "out_of_memory",
    "OUT_OF_MEMORY",
    "engine_error",
];

const THE_SERVED_CEILING_MUST_CLEAR_THE_PREVIOUS_RUNG_OR_THE_MAX_CELL_IS_RED: u32 = 65536;

const CHAT_ENV_FLIPPERS_REMOVED_SO_THE_CHILD_SERVES_THE_SHIPPED_DEFAULT_PATHS: [&str; 12] = [
    "NV_DRAFTER",
    "NV_DFLASH_DRAFT_DIR",
    "NV_Q38_BATCH",
    "NV_Q38_BATCH_SIZES",
    "NV_Q38_BATCH_GEMM",
    "NV_Q38_BATCH_ROWWISE",
    "NV_SERVE_BACKEND",
    "NV_WGPU_CHAT_MODEL_DIRS",
    "NV_CHAT_MODEL_DIRS",
    "NV_TTS_TALKER_DIR",
    "SPEACHES_API_KEY",
    "NV_KV_MAX_SEQ_LEN",
];

const VAD_AND_OTHER_NON_HTTP_TASKS_ADVERTISED_WITHOUT_A_MATRIX_ROUTE: [&str; 3] = [
    "voice-activity-detection",
    "speaker-embedding",
    "speaker-diarization",
];

const CELL_PLAN: [(&str, &str, u32); 18] = [
    ("*", "models", 0),
    ("embed", "embeddings", 0),
    ("kokoro", "tts", 0),
    ("whisper", "stt", 0),
    ("tesseract", "ocr", 0),
    ("*", "models.missing_chat", 0),
    ("*", "models.missing_tts", 0),
    ("qwen38", "chat.greedy", 256),
    ("qwen38", "chat.stream", 256),
    ("qwen38", "chat.think_on", 256),
    ("qwen38", "chat.think_off", 256),
    ("qwen38", "chat.multiturn", 512),
    ("qwen38", "completions", 16),
    ("qwen38", "chat.depth", 256),
    ("qwen38", "chat.depth", 8192),
    ("qwen38", "chat.depth", 65536),
    ("qwen38", "chat.depth", u32::MAX),
    ("*", "models.coverage", 0),
];

fn max_admitted_seq() -> usize {
    std::env::var(MAX_ADMITTED_SEQ_ENV_DEFAULT_196608_IS_THE_PROVEN_C196_RECORD_DEPTH)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_ADMITTED_SEQ)
}

fn resolve_depth(d: u32) -> u32 {
    if d == u32::MAX {
        (max_admitted_seq() - DEPTH_HEADROOM_KEEPS_PROMPT_PLUS_GENERATION_UNDER_THE_KV_CEILING)
            as u32
    } else {
        d
    }
}

fn artifact_dir() -> PathBuf {
    if let Ok(d) = std::env::var(ARTIFACT_DIR_ENV_DEFAULT_HOME_TMP_WF_FCD_E2E_BECAUSE_ROOT_TMP_IS_TMPFS)
    {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    PathBuf::from(home).join("tmp/wf/fcd/e2e")
}

fn emit(model: &str, route: &str, depth: u32, ok: &str, ms: u128, extra: &str) {
    let sep = if extra.is_empty() { "" } else { " " };
    println!("FCD-E2E model={model} route={route} depth={depth} ok={ok} ms={ms}{sep}{extra}");
}

fn kv_safe(s: &str) -> String {
    let mut t: String = s
        .chars()
        .map(|c| if c.is_whitespace() { '_' } else { c })
        .collect();
    t.truncate(160);
    t
}

fn hub_snapshot(repo_dir_name: &str, must_contain: &[&str]) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo_dir_name)
        .join("snapshots");
    std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| must_contain.iter().all(|f| p.join(f).exists()))
}

fn qwen38_dir() -> PathBuf {
    if let Ok(d) = std::env::var(QWEN38_DIR_ENV) {
        return PathBuf::from(d);
    }
    hub_snapshot(
        "models--unsloth--Qwen3.8-27B-NVFP4",
        &["config.json", "tokenizer.json", "chat_template.jinja"],
    )
    .expect("no Qwen3.8-27B-NVFP4 hub snapshot; set NV_QWEN38_DIR")
}

fn repo_relative(p: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(p)
}

fn symlink_replacing(target: &Path, link: &Path) {
    let _ = std::fs::remove_file(link);
    std::os::unix::fs::symlink(target, link)
        .unwrap_or_else(|e| panic!("symlink {} -> {}: {e}", link.display(), target.display()));
}

fn stage_models_dir(root: &Path) -> PathBuf {
    let dir = root.join("models");
    std::fs::create_dir_all(&dir).expect("create staged models dir");
    let kokoro = hub_snapshot("models--speaches-ai--Kokoro-82M-v1.0-ONNX", &["voices.bin"])
        .expect("Kokoro-82M-v1.0-ONNX hub snapshot missing");
    let kokoro_onnx = ["model.onnx", "onnx/model.onnx"]
        .iter()
        .map(|f| kokoro.join(f))
        .find(|p| p.is_file())
        .expect("kokoro snapshot has no model.onnx");
    symlink_replacing(&kokoro_onnx, &dir.join("kokoro-v1.0.onnx"));
    symlink_replacing(&kokoro.join("voices.bin"), &dir.join("voices.bin"));
    let whisper = hub_snapshot(
        "models--deepdml--faster-whisper-large-v3-turbo-ct2",
        &["model.bin"],
    )
    .expect("faster-whisper-large-v3-turbo-ct2 hub snapshot missing");
    symlink_replacing(&whisper, &dir.join("whisper-ct2"));
    let vad = repo_relative("silero_vad.onnx");
    assert!(vad.is_file(), "rust/silero_vad.onnx missing from the repo");
    symlink_replacing(&vad, &dir.join("silero_vad.onnx"));
    dir
}

fn embedding_dir() -> PathBuf {
    hub_snapshot(
        "models--Qwen--Qwen3-Embedding-0.6B",
        &["config.json", "tokenizer.json", "model.safetensors"],
    )
    .expect("Qwen3-Embedding-0.6B hub snapshot missing")
}

struct BootedServer {
    child: Child,
    port: u16,
    log_path: PathBuf,
}

impl Drop for BootedServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl BootedServer {
    fn log(&self) -> String {
        std::fs::read_to_string(&self.log_path).unwrap_or_default()
    }

    fn wait_until_serving(&mut self) -> serde_json::Value {
        let t0 = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().expect("try_wait") {
                panic!("server exited during boot with {status}; log:\n{}", self.log());
            }
            if let Ok(out) = http(self.port, &get_request("/v1/models"), SHORT_CELL_READ_TIMEOUT)
            {
                assert_eq!(
                    out.status, 200,
                    "/v1/models must answer 200 once bound; body: {}",
                    String::from_utf8_lossy(&out.body)
                );
                return serde_json::from_slice(&out.body).expect("/v1/models is JSON");
            }
            assert!(
                t0.elapsed() < BOOT_DEADLINE_COVERS_A_22GIB_WEIGHT_LOAD_PLUS_196K_KV_ALLOC,
                "server did not serve /v1/models within {:?}; log tail:\n{}",
                BOOT_DEADLINE_COVERS_A_22GIB_WEIGHT_LOAD_PLUS_196K_KV_ALLOC,
                tail(&self.log(), 60)
            );
            std::thread::sleep(Duration::from_secs(2));
        }
    }
}

fn tail(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral")
        .local_addr()
        .expect("local addr")
        .port()
}

fn boot_full_profile(art: &Path) -> BootedServer {
    let models_dir = stage_models_dir(art);
    let port = free_port();
    let log_path = art.join(format!("server-{}.log", std::process::id()));
    let log = std::fs::File::create(&log_path).expect("create server log");
    let mut cmd = Command::new(SERVER_BIN);
    for k in CHAT_ENV_FLIPPERS_REMOVED_SO_THE_CHILD_SERVES_THE_SHIPPED_DEFAULT_PATHS {
        cmd.env_remove(k);
    }
    cmd.env("UVICORN_HOST", "127.0.0.1");
    cmd.env("UVICORN_PORT", port.to_string());
    cmd.env("SPEACHES_PLUS_MODELS", &models_dir);
    cmd.env("STT_BACKEND", "ct2");
    cmd.env("NV_CHAT_MODEL_DIR", qwen38_dir());
    cmd.env("NV_QWEN35_DENSE_CUDA_SERVE", "1");
    cmd.env("NV_GDN_FUSED_DECODE", "1");
    cmd.env("NV_KV_MAX_SEQ_LEN", max_admitted_seq().to_string());
    cmd.env("NV_EMBEDDING_MODEL_DIR", embedding_dir());
    cmd.stdout(Stdio::from(log.try_clone().expect("clone log handle")));
    cmd.stderr(Stdio::from(log));
    let child = cmd.spawn().expect("spawn the real server binary");
    BootedServer {
        child,
        port,
        log_path,
    }
}

struct HttpOut {
    status: u16,
    body: Vec<u8>,
}

fn get_request(path: &str) -> Vec<u8> {
    format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n").into_bytes()
}

fn post_json_request(path: &str, body: &str) -> Vec<u8> {
    format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

fn post_multipart_request(path: &str, boundary: &str, body: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: multipart/form-data; boundary={boundary}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    out.extend_from_slice(body);
    out
}

fn multipart_body(boundary: &str, fields: &[(&str, Option<(&str, &str)>, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, file_meta, value) in fields {
        out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        match file_meta {
            Some((filename, mime)) => out.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n"
                )
                .as_bytes(),
            ),
            None => out.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            ),
        }
        out.extend_from_slice(value);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    out
}

fn http(port: u16, request: &[u8], read_timeout: Duration) -> Result<HttpOut, String> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(5))
        .map_err(|e| format!("connect: {e}"))?;
    s.set_read_timeout(Some(read_timeout)).map_err(|e| e.to_string())?;
    s.set_write_timeout(Some(Duration::from_secs(60)))
        .map_err(|e| e.to_string())?;
    s.write_all(request).map_err(|e| format!("write: {e}"))?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).map_err(|e| format!("read: {e}"))?;
    let header_end = find_subslice(&raw, b"\r\n\r\n").ok_or("no header terminator")?;
    let head = String::from_utf8_lossy(&raw[..header_end]).into_owned();
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("unparseable status line: {head}"))?;
    let payload = &raw[header_end + 4..];
    let chunked = head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked");
    let body = if chunked {
        decode_chunked(payload)?
    } else {
        payload.to_vec()
    };
    Ok(HttpOut { status, body })
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn decode_chunked(mut rest: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let line_end = find_subslice(rest, b"\r\n").ok_or("chunk size line unterminated")?;
        let size_str = String::from_utf8_lossy(&rest[..line_end]);
        let size = usize::from_str_radix(size_str.trim().split(';').next().unwrap_or(""), 16)
            .map_err(|e| format!("chunk size {size_str:?}: {e}"))?;
        rest = &rest[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        if rest.len() < size {
            return Err(format!(
                "chunk claims {size} bytes but only {} remain (connection cut mid-chunk)",
                rest.len()
            ));
        }
        out.extend_from_slice(&rest[..size]);
        rest = &rest[size..];
        if rest.starts_with(b"\r\n") {
            rest = &rest[2..];
        }
    }
}

fn finite_text(s: &str) -> bool {
    !s.trim().is_empty() && !s.contains('\u{FFFD}')
}

struct ChatOut {
    content: String,
    reasoning: String,
    finish_reason: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
}

fn chat_body(model: &str, messages: &serde_json::Value, max_tokens: u32, thinking: bool) -> String {
    serde_json::json!({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": thinking,
        "messages": messages,
    })
    .to_string()
}

fn chat_once(
    port: u16,
    model: &str,
    messages: &serde_json::Value,
    max_tokens: u32,
    thinking: bool,
    timeout: Duration,
) -> Result<ChatOut, String> {
    let out = http(
        port,
        &post_json_request(
            "/v1/chat/completions",
            &chat_body(model, messages, max_tokens, thinking),
        ),
        timeout,
    )?;
    let v: serde_json::Value = serde_json::from_slice(&out.body)
        .map_err(|e| format!("chat response not JSON ({e}): {}", String::from_utf8_lossy(&out.body)))?;
    if out.status != 200 {
        return Err(format!("status={} body={v}", out.status));
    }
    let msg = &v["choices"][0]["message"];
    Ok(ChatOut {
        content: msg["content"].as_str().unwrap_or_default().to_string(),
        reasoning: msg["reasoning_content"].as_str().unwrap_or_default().to_string(),
        finish_reason: v["choices"][0]["finish_reason"].as_str().unwrap_or_default().to_string(),
        prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
        completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0),
        cached_tokens: v["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0),
    })
}

fn user_messages(prompt: &str) -> serde_json::Value {
    serde_json::json!([{"role": "user", "content": prompt}])
}

const NEEDLE_SENTENCE: &str = "The secret code word is BLUEBERRY.";

const NEEDLE_ANSWER: &str = "BLUEBERRY";

const NEEDLE_QUESTION: &str =
    "What is the secret code word stated at the very beginning? Answer with only the code word.";

fn filler_facts(n: usize) -> String {
    let mut s = String::with_capacity(n * 64);
    for i in 0..n {
        s.push_str(&format!(
            "Fact {i:06}: the river gauge read {:03} centimeters on day {:03}. ",
            (i * 7) % 997,
            i % 365
        ));
    }
    s
}

fn depth_prompt(facts: usize) -> String {
    format!("{NEEDLE_SENTENCE} {}\n{NEEDLE_QUESTION}", filler_facts(facts))
}

struct DepthCalibration {
    base_tokens: f64,
    tokens_per_fact: f64,
}

fn calibrate_depth(port: u16, model: &str) -> Result<DepthCalibration, String> {
    let base = chat_once(
        port,
        model,
        &user_messages(&depth_prompt(0)),
        1,
        false,
        SHORT_CELL_READ_TIMEOUT,
    )?;
    let hundred = chat_once(
        port,
        model,
        &user_messages(&depth_prompt(100)),
        1,
        false,
        SHORT_CELL_READ_TIMEOUT,
    )?;
    if hundred.prompt_tokens <= base.prompt_tokens {
        return Err(format!(
            "calibration got prompt_tokens {} then {}; usage is not reporting prompt size",
            base.prompt_tokens, hundred.prompt_tokens
        ));
    }
    Ok(DepthCalibration {
        base_tokens: base.prompt_tokens as f64,
        tokens_per_fact: (hundred.prompt_tokens - base.prompt_tokens) as f64 / 100.0,
    })
}

impl DepthCalibration {
    fn facts_for(&self, target_tokens: u32) -> usize {
        let ceiling = (max_admitted_seq() as f64 - 1024.0 - self.base_tokens).max(0.0);
        let extra = (target_tokens as f64 - self.base_tokens).max(0.0).min(ceiling);
        (extra / self.tokens_per_fact).ceil() as usize
    }
}

type CellResult = Result<String, String>;

fn run_cell(name: &str, f: impl FnOnce() -> CellResult) -> (String, bool, u128, String) {
    let t0 = Instant::now();
    let r = f();
    let ms = t0.elapsed().as_millis();
    match r {
        Ok(extra) => (name.to_string(), true, ms, extra),
        Err(e) => (name.to_string(), false, ms, format!("err={}", kv_safe(&e))),
    }
}

struct Advertised {
    chat: Vec<String>,
    tts: Vec<String>,
    asr: Vec<String>,
    embeddings: Vec<String>,
    other_tasks: Vec<(String, String)>,
    kokoro_voice: String,
}

fn parse_models(v: &serde_json::Value) -> Result<Advertised, String> {
    let data = v["data"].as_array().ok_or("models .data is not an array")?;
    let mut a = Advertised {
        chat: vec![],
        tts: vec![],
        asr: vec![],
        embeddings: vec![],
        other_tasks: vec![],
        kokoro_voice: String::new(),
    };
    for m in data {
        let id = m["id"].as_str().unwrap_or_default().to_string();
        let task = m["task"].as_str().unwrap_or_default().to_string();
        match task.as_str() {
            "chat" => a.chat.push(id),
            "text-to-speech" => {
                if a.kokoro_voice.is_empty() {
                    if let Some(vs) = m["voices"].as_array() {
                        let names: Vec<&str> =
                            vs.iter().filter_map(|x| x.as_str()).collect();
                        a.kokoro_voice = names
                            .iter()
                            .find(|n| **n == "af_heart")
                            .copied()
                            .or_else(|| names.first().copied())
                            .unwrap_or("")
                            .to_string();
                    }
                }
                a.tts.push(id);
            }
            "automatic-speech-recognition" => a.asr.push(id),
            "embeddings" => a.embeddings.push(id),
            _ => a.other_tasks.push((id, task)),
        }
    }
    Ok(a)
}

fn sse_frames(body: &[u8]) -> Vec<serde_json::Value> {
    let text = String::from_utf8_lossy(body);
    text.split("\n\n")
        .filter_map(|f| f.trim().strip_prefix("data: "))
        .filter(|p| p.trim() != "[DONE]")
        .filter_map(|p| serde_json::from_str(p).ok())
        .collect()
}

fn wav_sanity(bytes: &[u8]) -> Result<(usize, usize), String> {
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!(
            "not a RIFF/WAVE file ({} bytes, head {:02x?})",
            bytes.len(),
            &bytes[..bytes.len().min(8)]
        ));
    }
    let data_at = find_subslice(bytes, b"data").ok_or("no data chunk")?;
    let samples = &bytes[data_at + 8..];
    let mut seen = [false; 256];
    for &b in samples.iter().take(65536) {
        seen[b as usize] = true;
    }
    let distinct = seen.iter().filter(|x| **x).count();
    Ok((samples.len(), distinct))
}

#[test]
#[ignore = "boots the real server binary with the full profile (qwen3.8 chat + kokoro + \
            whisper-ct2 + qwen3 embeddings + tesseract OCR) and drives every served modality \
            over HTTP with content assertions; set NV_FCD_E2E_TEST=1 on a quiet GPU, or \
            NV_FCD_E2E_DRYRUN=1 to print the cell plan without GPU or weights"]
fn fcd_e2e_full_profile_matrix_every_modality_answers_with_sane_content() {
    if std::env::var(DRY_RUN_PRINTS_THE_CELL_PLAN_WITHOUT_GPU_OR_WEIGHTS_ENV).as_deref() == Ok("1")
    {
        for (model, route, depth) in CELL_PLAN {
            emit(model, route, resolve_depth(depth), "dry", 0, "");
        }
        assert_eq!(CELL_PLAN.len(), 18, "cell plan drifted from its declared size");
        return;
    }
    if std::env::var(FULL_RUN_GATE_ENV).as_deref() != Ok("1") {
        panic!(
            "this test was asked for BY NAME but {FULL_RUN_GATE_ENV}=1 is not set (and \
             {DRY_RUN_PRINTS_THE_CELL_PLAN_WITHOUT_GPU_OR_WEIGHTS_ENV}=1 was not requested). \
             This is a SKIP, not a pass."
        );
    }

    let art = artifact_dir();
    std::fs::create_dir_all(&art).expect("artifact dir");
    let mut server = boot_full_profile(&art);
    let models_json = server.wait_until_serving();
    let port = server.port;
    eprintln!("[fcd-e2e] serving on port {port}; log at {}", server.log_path.display());

    let adv = parse_models(&models_json).expect("parse /v1/models");
    let mut failures: Vec<String> = Vec::new();
    let mut route_ok: Vec<(String, bool)> = Vec::new();
    let record = |model: &str, route: &str, depth: u32, res: (String, bool, u128, String),
                      failures: &mut Vec<String>,
                      route_ok: &mut Vec<(String, bool)>| {
        let (_, ok, ms, extra) = res;
        emit(model, route, depth, if ok { "1" } else { "0" }, ms, &extra);
        route_ok.push((route.to_string(), ok));
        if !ok {
            failures.push(format!("{route} depth={depth}: {extra}"));
        }
    };

    let chat_id = adv.chat.first().cloned().unwrap_or_default();
    let tts_id = adv.tts.first().cloned().unwrap_or_default();
    let asr_id = adv.asr.first().cloned().unwrap_or_default();
    let embed_id = adv.embeddings.first().cloned().unwrap_or_default();

    {
        let res = run_cell("models", || {
            if chat_id.is_empty() {
                return Err("no chat model advertised".into());
            }
            if tts_id.is_empty() || adv.kokoro_voice.is_empty() {
                return Err(format!(
                    "tts advertised={:?} voice={:?}: kokoro must be served with voices",
                    adv.tts, adv.kokoro_voice
                ));
            }
            if asr_id.is_empty() {
                return Err("no ASR model advertised".into());
            }
            if embed_id.is_empty() {
                return Err("no embeddings model advertised".into());
            }
            for (id, task) in &adv.other_tasks {
                if !VAD_AND_OTHER_NON_HTTP_TASKS_ADVERTISED_WITHOUT_A_MATRIX_ROUTE
                    .contains(&task.as_str())
                {
                    return Err(format!(
                        "model {id} advertises task {task} which this matrix does not cover; \
                         grow the net before reading it as green"
                    ));
                }
            }
            Ok(format!(
                "advertised={} chat={} tts={} asr={} embed={}",
                adv.chat.len() + adv.tts.len() + adv.asr.len() + adv.embeddings.len()
                    + adv.other_tasks.len(),
                kv_safe(&chat_id),
                kv_safe(&tts_id),
                kv_safe(&asr_id),
                kv_safe(&embed_id)
            ))
        });
        record("*", "models", 0, res, &mut failures, &mut route_ok);
    }

    {
        let res = run_cell("embeddings", || {
            let body = serde_json::json!({
                "model": embed_id,
                "input": ["a lighthouse keeper's morning routine", "the tides follow the moon"],
            })
            .to_string();
            let out = http(port, &post_json_request("/v1/embeddings", &body), SHORT_CELL_READ_TIMEOUT)?;
            let v: serde_json::Value = serde_json::from_slice(&out.body)
                .map_err(|e| format!("not JSON: {e}"))?;
            if out.status != 200 {
                return Err(format!("status={} body={v}", out.status));
            }
            let data = v["data"].as_array().ok_or("no .data array")?;
            if data.len() != 2 {
                return Err(format!("2 inputs produced {} embeddings", data.len()));
            }
            let e0: Vec<f64> = data[0]["embedding"]
                .as_array()
                .ok_or("embedding not an array")?
                .iter()
                .filter_map(|x| x.as_f64())
                .collect();
            let e1: Vec<f64> = data[1]["embedding"]
                .as_array()
                .ok_or("embedding not an array")?
                .iter()
                .filter_map(|x| x.as_f64())
                .collect();
            if e0.is_empty() || e0.len() != e1.len() {
                return Err(format!("dims {} vs {}", e0.len(), e1.len()));
            }
            if e0.iter().chain(e1.iter()).any(|x| !x.is_finite()) {
                return Err("non-finite embedding value".into());
            }
            let dot: f64 = e0.iter().zip(&e1).map(|(a, b)| a * b).sum();
            let n0: f64 = e0.iter().map(|x| x * x).sum::<f64>().sqrt();
            let n1: f64 = e1.iter().map(|x| x * x).sum::<f64>().sqrt();
            if n0 < 1e-6 || n1 < 1e-6 {
                return Err(format!("near-zero norms {n0} {n1}"));
            }
            let cos = dot / (n0 * n1);
            if cos > 0.9999 {
                return Err(format!(
                    "two different inputs embedded to cos={cos}; the embedder is constant"
                ));
            }
            Ok(format!("dim={} cos={cos:.4}", e0.len()))
        });
        record(&kv_safe(&embed_id), "embeddings", 0, res, &mut failures, &mut route_ok);
    }

    {
        let res = run_cell("tts", || {
            let body = serde_json::json!({
                "model": tts_id,
                "input": "The quick brown fox jumps over the lazy dog near the harbour.",
                "voice": adv.kokoro_voice,
                "response_format": "wav",
            })
            .to_string();
            let out = http(port, &post_json_request("/v1/audio/speech", &body), SHORT_CELL_READ_TIMEOUT)?;
            if out.status != 200 {
                return Err(format!(
                    "status={} body={}",
                    out.status,
                    kv_safe(&String::from_utf8_lossy(&out.body))
                ));
            }
            let (sample_bytes, distinct) = wav_sanity(&out.body)?;
            if sample_bytes < 32_000 {
                return Err(format!(
                    "{sample_bytes} sample bytes is under 32000 (below ~0.7s at 24kHz s16)"
                ));
            }
            if distinct < 30 {
                return Err(format!(
                    "{distinct} distinct byte values in the sample data reads as silence"
                ));
            }
            Ok(format!("wav_bytes={} distinct={distinct} voice={}", out.body.len(), adv.kokoro_voice))
        });
        record(&kv_safe(&tts_id), "tts", 0, res, &mut failures, &mut route_ok);
    }

    {
        let res = run_cell("stt", || {
            let wav = std::fs::read(repo_relative("tests/data/parakeet_fox_16k.wav"))
                .map_err(|e| format!("read wav fixture: {e}"))?;
            let boundary = "fcd-e2e-stt-boundary";
            let body = multipart_body(
                boundary,
                &[
                    ("file", Some(("fox.wav", "audio/wav")), &wav),
                    ("model", None, asr_id.as_bytes()),
                    ("response_format", None, b"json"),
                ],
            );
            let out = http(
                port,
                &post_multipart_request("/v1/audio/transcriptions", boundary, &body),
                SHORT_CELL_READ_TIMEOUT,
            )?;
            let v: serde_json::Value = serde_json::from_slice(&out.body)
                .map_err(|e| format!("not JSON: {e}"))?;
            if out.status != 200 {
                return Err(format!("status={} body={v}", out.status));
            }
            let text = v["text"].as_str().unwrap_or_default().to_lowercase();
            if !finite_text(&text) {
                return Err("empty transcription".into());
            }
            if !text.contains("fox") {
                return Err(format!("transcript lost the fox: {}", kv_safe(&text)));
            }
            Ok(format!("chars={}", text.len()))
        });
        record(&kv_safe(&asr_id), "stt", 0, res, &mut failures, &mut route_ok);
    }

    {
        let res = run_cell("ocr", || {
            let png = std::fs::read(repo_relative(
                "../conformance/fixtures/070-ocr-paragraph/input.png",
            ))
            .map_err(|e| format!("read ocr fixture: {e}"))?;
            let boundary = "fcd-e2e-ocr-boundary";
            let body = multipart_body(
                boundary,
                &[
                    ("file", Some(("page.png", "image/png")), &png),
                    ("backend", None, b"tesseract"),
                    ("format", None, b"text"),
                ],
            );
            let out = http(
                port,
                &post_multipart_request("/v1/ocr", boundary, &body),
                SHORT_CELL_READ_TIMEOUT,
            )?;
            let text = String::from_utf8_lossy(&out.body).to_string();
            if out.status != 200 {
                return Err(format!("status={} body={}", out.status, kv_safe(&text)));
            }
            let lower = text.to_lowercase();
            if !(finite_text(&text) && lower.contains("quick") && lower.contains("fox")) {
                return Err(format!("paragraph fixture text unrecognized: {}", kv_safe(&text)));
            }
            Ok(format!("chars={}", text.len()))
        });
        record("tesseract", "ocr", 0, res, &mut failures, &mut route_ok);
    }

    {
        let res = run_cell("models.missing_chat", || {
            let out = http(
                port,
                &post_json_request(
                    "/v1/chat/completions",
                    &chat_body("fcd-no-such-model", &user_messages("hi"), 8, false),
                ),
                SHORT_CELL_READ_TIMEOUT,
            )?;
            let v: serde_json::Value = serde_json::from_slice(&out.body)
                .map_err(|e| format!("unknown-model reply not JSON: {e}"))?;
            let code = v["error"]["code"].as_str().unwrap_or_default();
            if out.status == 200 {
                return Err("an unknown chat model was silently served".into());
            }
            if code != "model_not_found" {
                return Err(format!(
                    "status={} code={code:?}: the missing-model contract names model_not_found",
                    out.status
                ));
            }
            Ok(format!("status={}", out.status))
        });
        record("*", "models.missing_chat", 0, res, &mut failures, &mut route_ok);
    }

    {
        let res = run_cell("models.missing_tts", || {
            let body = serde_json::json!({
                "model": "fcd-no-such-voice-model",
                "input": "hello",
                "voice": adv.kokoro_voice,
            })
            .to_string();
            let out = http(port, &post_json_request("/v1/audio/speech", &body), SHORT_CELL_READ_TIMEOUT)?;
            if out.status == 200 {
                return Err("an unknown tts model was silently rendered".into());
            }
            let text = String::from_utf8_lossy(&out.body);
            if !text.contains("model_not_found") {
                return Err(format!(
                    "status={} body={}: refusal does not carry model_not_found",
                    out.status,
                    kv_safe(&text)
                ));
            }
            Ok(format!("status={}", out.status))
        });
        record("*", "models.missing_tts", 0, res, &mut failures, &mut route_ok);
    }

    {
        let res = run_cell("chat.greedy", || {
            let o = chat_once(
                port,
                &chat_id,
                &user_messages(&depth_prompt(10)),
                32,
                false,
                SHORT_CELL_READ_TIMEOUT,
            )?;
            if !finite_text(&o.content) {
                return Err(format!("content empty; finish={}", o.finish_reason));
            }
            if !o.content.to_uppercase().contains(NEEDLE_ANSWER) {
                return Err(format!("needle lost at depth 256: {}", kv_safe(&o.content)));
            }
            if o.completion_tokens == 0 || o.prompt_tokens == 0 {
                return Err("usage reports zero tokens".into());
            }
            Ok(format!("pt={} ct={} finish={}", o.prompt_tokens, o.completion_tokens, o.finish_reason))
        });
        record(&kv_safe(&chat_id), "chat.greedy", 256, res, &mut failures, &mut route_ok);
    }

    {
        let res = run_cell("chat.stream", || {
            let body = serde_json::json!({
                "model": chat_id,
                "max_tokens": 48,
                "temperature": 0,
                "enable_thinking": false,
                "stream": true,
                "stream_options": {"include_usage": true},
                "messages": [{"role": "user", "content":
                    "Write one sentence about a lighthouse."}],
            })
            .to_string();
            let out = http(port, &post_json_request("/v1/chat/completions", &body), SHORT_CELL_READ_TIMEOUT)?;
            if out.status != 200 {
                return Err(format!(
                    "status={} body={}",
                    out.status,
                    kv_safe(&String::from_utf8_lossy(&out.body))
                ));
            }
            let frames = sse_frames(&out.body);
            let mut content = String::new();
            let mut delta_frames = 0usize;
            let mut completion_tokens = 0u64;
            for f in &frames {
                if let Some(c) = f["choices"][0]["delta"]["content"].as_str() {
                    if !c.is_empty() {
                        delta_frames += 1;
                        content.push_str(c);
                    }
                }
                if let Some(u) = f.get("usage").filter(|u| !u.is_null()) {
                    completion_tokens = u["completion_tokens"].as_u64().unwrap_or(0);
                }
            }
            if delta_frames < 2 {
                return Err(format!(
                    "{delta_frames} content delta frames: the stream arrived as one blob or not at all"
                ));
            }
            if !finite_text(&content) {
                return Err("streamed content empty".into());
            }
            if completion_tokens == 0 {
                return Err("stream_options.include_usage returned no usage frame".into());
            }
            Ok(format!("frames={delta_frames} ct={completion_tokens} chars={}", content.len()))
        });
        record(&kv_safe(&chat_id), "chat.stream", 256, res, &mut failures, &mut route_ok);
    }

    {
        let res = run_cell("chat.think_on", || {
            let o = chat_once(
                port,
                &chat_id,
                &user_messages("How many days are in a leap year February? Think briefly."),
                512,
                true,
                SHORT_CELL_READ_TIMEOUT,
            )?;
            if !finite_text(&o.reasoning) {
                return Err(format!(
                    "enable_thinking=true produced no reasoning_content; content={} finish={}",
                    kv_safe(&o.content),
                    o.finish_reason
                ));
            }
            Ok(format!(
                "reasoning_chars={} content_chars={} finish={}",
                o.reasoning.len(),
                o.content.len(),
                o.finish_reason
            ))
        });
        record(&kv_safe(&chat_id), "chat.think_on", 256, res, &mut failures, &mut route_ok);
    }

    {
        let res = run_cell("chat.think_off", || {
            let o = chat_once(
                port,
                &chat_id,
                &user_messages("How many days are in a leap year February? Answer with a number."),
                32,
                false,
                SHORT_CELL_READ_TIMEOUT,
            )?;
            if !o.reasoning.is_empty() {
                return Err(format!(
                    "enable_thinking=false leaked {} chars of reasoning",
                    o.reasoning.len()
                ));
            }
            if !finite_text(&o.content) {
                return Err("content empty with thinking off".into());
            }
            if !o.content.contains("29") {
                return Err(format!("leap February answer wrong: {}", kv_safe(&o.content)));
            }
            Ok(format!("ct={} finish={}", o.completion_tokens, o.finish_reason))
        });
        record(&kv_safe(&chat_id), "chat.think_off", 256, res, &mut failures, &mut route_ok);
    }

    {
        let res = run_cell("chat.multiturn", || {
            let turn1 = chat_once(
                port,
                &chat_id,
                &user_messages("My favorite city is Ravenna. Please reply with exactly: noted"),
                16,
                false,
                SHORT_CELL_READ_TIMEOUT,
            )?;
            if !finite_text(&turn1.content) {
                return Err("turn 1 content empty".into());
            }
            let messages = serde_json::json!([
                {"role": "user", "content": "My favorite city is Ravenna. Please reply with exactly: noted"},
                {"role": "assistant", "content": turn1.content},
                {"role": "user", "content": "Which city did I say was my favorite? Answer with just the city name."},
            ]);
            let turn2 = chat_once(port, &chat_id, &messages, 16, false, SHORT_CELL_READ_TIMEOUT)?;
            if !turn2.content.to_lowercase().contains("ravenna") {
                return Err(format!(
                    "turn 2 lost the conversation prefix: {}",
                    kv_safe(&turn2.content)
                ));
            }
            if turn2.prompt_tokens <= turn1.prompt_tokens {
                return Err(format!(
                    "turn 2 prompt_tokens {} not greater than turn 1 {}",
                    turn2.prompt_tokens, turn1.prompt_tokens
                ));
            }
            Ok(format!(
                "t1_pt={} t2_pt={} t2_cached={}",
                turn1.prompt_tokens, turn2.prompt_tokens, turn2.cached_tokens
            ))
        });
        record(&kv_safe(&chat_id), "chat.multiturn", 512, res, &mut failures, &mut route_ok);
    }

    {
        let res = run_cell("completions", || {
            let body = serde_json::json!({
                "model": chat_id,
                "prompt": "The planets of the solar system in order from the sun are: Mercury, Venus,",
                "max_tokens": 16,
                "temperature": 0,
            })
            .to_string();
            let out = http(port, &post_json_request("/v1/completions", &body), SHORT_CELL_READ_TIMEOUT)?;
            let v: serde_json::Value = serde_json::from_slice(&out.body)
                .map_err(|e| format!("not JSON: {e}"))?;
            if out.status != 200 {
                return Err(format!("status={} body={v}", out.status));
            }
            let text = v["choices"][0]["text"].as_str().unwrap_or_default().to_string();
            if !finite_text(&text) {
                return Err("completion text empty".into());
            }
            if !text.contains("Earth") {
                return Err(format!("greedy continuation lost Earth: {}", kv_safe(&text)));
            }
            let ct = v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
            if ct == 0 {
                return Err("usage reports zero completion tokens".into());
            }
            Ok(format!("ct={ct} text={}", kv_safe(&text)))
        });
        record(&kv_safe(&chat_id), "completions", 16, res, &mut failures, &mut route_ok);
    }

    let calibration = match calibrate_depth(port, &chat_id) {
        Ok(c) => Some(c),
        Err(e) => {
            failures.push(format!("depth calibration: {e}"));
            None
        }
    };
    let depth_cell = |target: u32, cal: &DepthCalibration| -> CellResult {
        let facts = cal.facts_for(target);
        let o = chat_once(
            port,
            &chat_id,
            &user_messages(&depth_prompt(facts)),
            32,
            false,
            DEEP_PREFILL_READ_TIMEOUT,
        )?;
        let achieved = o.prompt_tokens as f64;
        let rel = (achieved - target as f64).abs() / target as f64;
        if rel > DEPTH_LADDER_TOKENS_ACHIEVED_MUST_BE_WITHIN_15_PCT_OF_TARGET {
            return Err(format!(
                "target {target} tokens but prompt_tokens={} ({} facts): the synthetic \
                 prompt generator and the served tokenizer disagree",
                o.prompt_tokens, facts
            ));
        }
        if !finite_text(&o.content) {
            return Err(format!(
                "content empty at depth {}; finish={}",
                o.prompt_tokens, o.finish_reason
            ));
        }
        if o.finish_reason.is_empty() {
            return Err("no finish_reason".into());
        }
        let needle = o.content.to_uppercase().contains(NEEDLE_ANSWER);
        if target <= NEEDLE_RECALL_IS_ASSERTED_ONLY_UP_TO_8192_DEEPER_CELLS_RECORD_IT && !needle {
            return Err(format!(
                "needle lost at asserted depth {}: {}",
                o.prompt_tokens,
                kv_safe(&o.content)
            ));
        }
        Ok(format!(
            "pt={} ct={} finish={} needle={} facts={facts}",
            o.prompt_tokens,
            o.completion_tokens,
            o.finish_reason,
            needle as u8
        ))
    };
    for depth in [256u32, 8192, 65536] {
        let Some(cal) = calibration.as_ref() else {
            emit(&kv_safe(&chat_id), "chat.depth", depth, "0", 0, "err=calibration_failed");
            route_ok.push(("chat.depth".to_string(), false));
            failures.push(format!("chat.depth depth={depth}: calibration failed"));
            continue;
        };
        let res = run_cell("chat.depth", || depth_cell(depth, cal));
        record(&kv_safe(&chat_id), "chat.depth", depth, res, &mut failures, &mut route_ok);
    }
    {
        let max_rung = resolve_depth(u32::MAX);
        match calibration.as_ref() {
            None => {
                emit(&kv_safe(&chat_id), "chat.depth", max_rung, "0", 0, "err=calibration_failed");
                route_ok.push(("chat.depth".to_string(), false));
                failures.push(format!("chat.depth depth={max_rung}: calibration failed"));
            }
            Some(cal) => {
                let res = run_cell("chat.depth", || {
                    let mut refused: Vec<String> = Vec::new();
                    for frac in
                        MAX_RUNG_DESCENT_FRACTIONS_FIND_THE_SERVED_CEILING_WHEN_PREFILL_OOMS_LOUDLY_INSIDE_THE_KV_WINDOW
                    {
                        let target = (max_rung as f64 * frac) as u32;
                        if target <= THE_SERVED_CEILING_MUST_CLEAR_THE_PREVIOUS_RUNG_OR_THE_MAX_CELL_IS_RED
                        {
                            break;
                        }
                        match depth_cell(target, cal) {
                            Ok(extra) => {
                                let oom_note = if refused.is_empty() {
                                    String::new()
                                } else {
                                    format!(" refused_above={}", refused.join(","))
                                };
                                return Ok(format!("{extra} attempted={max_rung}{oom_note}"));
                            }
                            Err(e) => {
                                let loud =
                                    A_LOUD_DEPTH_REFUSAL_NAMES_ONE_OF_THESE_OR_THE_DESCENT_IS_HIDING_A_SILENT_FAILURE
                                        .iter()
                                        .any(|k| e.contains(k));
                                if !loud {
                                    return Err(format!(
                                        "depth {target} failed without naming a capacity \
                                         constraint, so descending would hide it: {e}"
                                    ));
                                }
                                refused.push(target.to_string());
                            }
                        }
                    }
                    Err(format!(
                        "no depth above {} was served; refusals at {}",
                        THE_SERVED_CEILING_MUST_CLEAR_THE_PREVIOUS_RUNG_OR_THE_MAX_CELL_IS_RED,
                        refused.join(",")
                    ))
                });
                record(&kv_safe(&chat_id), "chat.depth", max_rung, res, &mut failures, &mut route_ok);
            }
        }
    }

    {
        let res = run_cell("models.coverage", || {
            let must_pass = ["models", "embeddings", "tts", "stt", "chat.greedy", "completions"];
            for r in must_pass {
                let ok = route_ok.iter().any(|(name, ok)| name == r && *ok);
                if !ok {
                    return Err(format!(
                        "advertised-modality route {r} has no passing cell, so an advertised \
                         model neither answered nor degraded loudly"
                    ));
                }
            }
            Ok(format!("cells={}", route_ok.len()))
        });
        let (_, ok, ms, extra) = res;
        emit("*", "models.coverage", 0, if ok { "1" } else { "0" }, ms, &extra);
        if !ok {
            failures.push(format!("models.coverage: {extra}"));
        }
    }

    let log = server.log();
    assert!(
        !log.contains("panicked"),
        "the server log recorded a panic during the matrix; log tail:\n{}",
        tail(&log, 40)
    );
    assert!(
        failures.is_empty(),
        "{} of {} matrix cells failed:\n{}",
        failures.len(),
        CELL_PLAN.len(),
        failures.join("\n")
    );
}
