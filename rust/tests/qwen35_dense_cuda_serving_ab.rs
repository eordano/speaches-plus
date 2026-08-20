#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::Router;
use tower::ServiceExt;

use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
use speaches_plus::oapi::chat_engine::{ChatRegistry, NvEngineChat};

const CUDA_SERVE_OPT_IN_ENV: &str = "NV_QWEN35_DENSE_CUDA_SERVE";
const REAL_WEIGHTS_TEST_ENV: &str = "NV_QWEN35_DENSE_CUDA_SERVE_TEST";
const MODEL_DIR_OVERRIDE_ENV: &str = "NV_QWEN35_DENSE_SERVE_DIR";

const FACTUAL_PROMPT_WHOSE_ANSWER_IS_ONE_WORD: &str =
    "What is the capital of France? Answer in one short sentence.";

const REASONING_PROMPT_LONG_ENOUGH_TO_SEPARATE_TWO_BACKENDS: &str =
    "Explain in two sentences why the sky is blue.";

const NVFP4_ACCUMULATION_ORDER_MAY_SEPARATE_TWO_BACKENDS_MID_TRACE: &str =
    "cuda and wgpu run the same greedy argmax over the same NVFP4 weights but accumulate in \
     different orders, so a long thinking trace may separate. The gate on a long generation is \
     therefore the answer both backends arrive at, with the first-divergence index reported \
     rather than asserted; the gate on a short answer, where no such separation has ever been \
     observed, is byte equality";

fn real_weights_test_enabled() -> bool {
    std::env::var(REAL_WEIGHTS_TEST_ENV).ok().as_deref() == Some("1")
}

fn model_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var(MODEL_DIR_OVERRIDE_ENV) {
        let p = PathBuf::from(d);
        return p.join("config.json").exists().then_some(p);
    }
    let root = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub/models--ig1--Qwen3.5-9B-NVFP4/snapshots");
    std::fs::read_dir(&root)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists() && p.join("model.safetensors").exists())
}

fn app(engine: Arc<dyn ChatEngine>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .with_state(ChatAppState {
            registry: ChatRegistry::single(engine),
        })
}

struct RunOut {
    content: String,
    reasoning: String,
    finish_reason: String,
    completion_tokens: u64,
    prompt_tokens: u64,
    wall_s: f64,
}

impl RunOut {
    fn stream(&self) -> String {
        format!("{}{}", self.reasoning, self.content)
    }
}

fn first_divergence(a: &str, b: &str) -> Option<usize> {
    a.chars()
        .zip(b.chars())
        .position(|(x, y)| x != y)
        .or(if a.len() == b.len() {
            None
        } else {
            Some(a.chars().count().min(b.chars().count()))
        })
}

fn run_prompt(engine: &Arc<dyn ChatEngine>, label: &str, prompt: &str, max_tokens: u32) -> RunOut {
    run_prompt_thinking(engine, label, prompt, max_tokens, false)
}

fn run_prompt_thinking(
    engine: &Arc<dyn ChatEngine>,
    label: &str,
    prompt: &str,
    max_tokens: u32,
    enable_thinking: bool,
) -> RunOut {
    let router = app(engine.clone());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async move {
        let body = format!(
            r#"{{"model":"{}","max_tokens":{max_tokens},"temperature":0,
                 "enable_thinking":{enable_thinking},
                 "messages":[{{"role":"user","content":{}}}]}}"#,
            engine.model_id(),
            serde_json::to_string(prompt).unwrap()
        );
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let t = std::time::Instant::now();
        let resp = router.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 1 << 22).await.unwrap();
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let wall_s = t.elapsed().as_secs_f64();
        assert_eq!(status, StatusCode::OK, "[{label}] {text}");
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let reasoning = v["choices"][0]["message"]["reasoning_content"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let finish_reason = v["choices"][0]["finish_reason"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let completion_tokens = v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        let prompt_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
        eprintln!(
            "[{label}] prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} \
             finish_reason={finish_reason} wall={wall_s:.2}s ({:.1} ms/tok incl. prefill)",
            wall_s * 1000.0 / completion_tokens.max(1) as f64
        );
        eprintln!("[{label}] reasoning={reasoning:?}");
        eprintln!("[{label}] content={content:?}");
        assert!(completion_tokens > 0, "[{label}] no tokens generated");
        RunOut {
            content,
            reasoning,
            finish_reason,
            completion_tokens,
            prompt_tokens,
            wall_s,
        }
    })
}

fn require_the_checkpoint_and_the_opt_in() -> PathBuf {
    if !real_weights_test_enabled() {
        panic!(
            "this test is #[ignore]d, so it was asked for BY NAME, but {REAL_WEIGHTS_TEST_ENV}=1 \
             is not set. This is a SKIP, not a pass."
        );
    }
    let Some(dir) = model_dir() else {
        panic!(
            "no qwen3.5-dense checkpoint found: set {MODEL_DIR_OVERRIDE_ENV}, or hydrate \
             models--ig1--Qwen3.5-9B-NVFP4 (config.json + model.safetensors) under \
             $HOME/.cache/huggingface/hub. This is a SKIP, not a pass."
        )
    };
    std::env::set_var(CUDA_SERVE_OPT_IN_ENV, "1");
    eprintln!(
        "[qwen35-dense-cuda] {CUDA_SERVE_OPT_IN_ENV}=1, checkpoint {}",
        dir.display()
    );
    dir
}

fn load_cuda(dir: &std::path::Path, label: &str) -> Arc<dyn ChatEngine> {
    let t = std::time::Instant::now();
    let engine = NvEngineChat::try_load(dir).unwrap_or_else(|e| {
        panic!(
            "[{label}] the qwen3.5-dense CUDA serving arm of NvEngineChat::try_load failed with \
             {CUDA_SERVE_OPT_IN_ENV}=1 set. This is the path task #40 exists to execute; do not \
             re-gate it, fix it or report what is missing: {e:#}"
        )
    });
    eprintln!(
        "[{label}] cuda engine ready in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    Arc::new(engine)
}

#[test]
#[ignore = "loads a 9.6 GiB NVFP4 checkpoint onto the GPU; set NV_QWEN35_DENSE_CUDA_SERVE_TEST=1"]
fn qwen35_dense_cuda_serving_loads_and_decodes_and_repeats_itself() {
    let dir = require_the_checkpoint_and_the_opt_in();
    let _ = tracing_subscriber::fmt()
        .with_env_filter("speaches_plus=info")
        .with_writer(std::io::stderr)
        .try_init();

    let a = load_cuda(&dir, "cuda-a");
    let r1 = run_prompt(
        &a,
        "cuda-a-run1",
        FACTUAL_PROMPT_WHOSE_ANSWER_IS_ONE_WORD,
        64,
    );
    let r2 = run_prompt(
        &a,
        "cuda-a-run2",
        FACTUAL_PROMPT_WHOSE_ANSWER_IS_ONE_WORD,
        64,
    );
    assert_eq!(
        r1.stream(),
        r2.stream(),
        "greedy decode changed between two back-to-back requests on one engine"
    );
    assert_eq!(r1.completion_tokens, r2.completion_tokens);
    assert_eq!(r1.prompt_tokens, r2.prompt_tokens);
    assert!(
        !r1.content.trim().is_empty(),
        "enable_thinking=false must yield a direct non-empty answer, got reasoning: {:?}",
        r1.reasoning
    );
    assert!(
        r1.content.to_lowercase().contains("paris"),
        "the dense cuda arm decoded text that is not the answer, which is what serving garbage \
         looks like: {:?}",
        r1.content
    );
    assert_eq!(
        r1.finish_reason, "stop",
        "the turn must end on the checkpoint's stop set (<|im_end|> 248046 alongside 248044), not \
         run out of budget: {:?}",
        r1.content
    );
    drop(a);

    let b = load_cuda(&dir, "cuda-b");
    let r3 = run_prompt(
        &b,
        "cuda-b-run1",
        FACTUAL_PROMPT_WHOSE_ANSWER_IS_ONE_WORD,
        64,
    );
    assert_eq!(
        r1.stream(),
        r3.stream(),
        "greedy decode changed across an engine reload"
    );
    eprintln!(
        "[done] cuda dense serving deterministic across 2 requests + reload; last wall {:.2}s",
        r3.wall_s
    );
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "loads the same checkpoint twice, once per backend; set NV_QWEN35_DENSE_CUDA_SERVE_TEST=1"]
fn qwen35_dense_cuda_answers_what_the_wgpu_oracle_answers() {
    use speaches_plus::oapi::chat_engine_wgpu::{WgpuChatEngine, WgpuModelKind};

    let dir = require_the_checkpoint_and_the_opt_in();
    let _ = tracing_subscriber::fmt()
        .with_env_filter("speaches_plus=info")
        .with_writer(std::io::stderr)
        .try_init();

    let cuda = load_cuda(&dir, "cuda");
    let cu_short = run_prompt(
        &cuda,
        "cuda-short",
        FACTUAL_PROMPT_WHOSE_ANSWER_IS_ONE_WORD,
        64,
    );
    let cu_long = run_prompt_thinking(
        &cuda,
        "cuda-long",
        REASONING_PROMPT_LONG_ENOUGH_TO_SEPARATE_TWO_BACKENDS,
        512,
        true,
    );
    drop(cuda);

    let t = std::time::Instant::now();
    let engine = WgpuChatEngine::load_with(&dir, 1024, None).expect("wgpu oracle did not load");
    assert_eq!(engine.kind(), WgpuModelKind::Qwen3_5Dense);
    eprintln!("[wgpu] oracle ready in {:.1}s", t.elapsed().as_secs_f64());
    let wg: Arc<dyn ChatEngine> = Arc::new(engine);
    let or_short = run_prompt(
        &wg,
        "wgpu-short",
        FACTUAL_PROMPT_WHOSE_ANSWER_IS_ONE_WORD,
        64,
    );
    let or_long = run_prompt_thinking(
        &wg,
        "wgpu-long",
        REASONING_PROMPT_LONG_ENOUGH_TO_SEPARATE_TWO_BACKENDS,
        512,
        true,
    );
    drop(wg);

    for (label, cu, ora) in [
        ("short", &cu_short, &or_short),
        ("long", &cu_long, &or_long),
    ] {
        eprintln!(
            "[ab-{label}] cuda {} chars / {} tok / {} vs wgpu {} chars / {} tok / {}; first \
             divergence at char {:?}",
            cu.stream().chars().count(),
            cu.completion_tokens,
            cu.finish_reason,
            ora.stream().chars().count(),
            ora.completion_tokens,
            ora.finish_reason,
            first_divergence(&cu.stream(), &ora.stream()),
        );
        assert_eq!(
            cu.prompt_tokens, ora.prompt_tokens,
            "[{label}] the two backends tokenized the same rendered prompt differently, so every \
             text comparison here is between two different questions"
        );
    }
    eprintln!("[ab] {NVFP4_ACCUMULATION_ORDER_MAY_SEPARATE_TWO_BACKENDS_MID_TRACE}");

    assert!(
        or_short.content.to_lowercase().contains("paris"),
        "the wgpu oracle itself did not answer the question, so it cannot judge cuda: {:?}",
        or_short.content
    );
    assert_eq!(
        cu_short.stream(),
        or_short.stream(),
        "on a short greedy answer the dense cuda arm must reproduce the real-weights-verified \
         nv_models::qwen3_5_dense_wgpu decoder byte for byte. It did when this gate was written, \
         so a separation here is a numerics change in the cuda trunk, not the accumulation-order \
         drift that only shows up over a long trace"
    );
    let mut oracle_content_words: Vec<String> = or_long
        .stream()
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 7)
        .map(|w| w.to_string())
        .collect();
    oracle_content_words.sort();
    oracle_content_words.dedup();
    let cuda_lower = cu_long.stream().to_lowercase();
    let shared = oracle_content_words
        .iter()
        .filter(|w| cuda_lower.contains(w.as_str()))
        .count();
    eprintln!(
        "[ab-long] cuda reuses {shared}/{} of the oracle's distinct content words",
        oracle_content_words.len()
    );
    assert!(
        oracle_content_words.len() >= 20,
        "the wgpu oracle produced too little long-form text to judge cuda against: {:?}",
        or_long.stream()
    );
    assert!(
        shared * 2 >= oracle_content_words.len(),
        "once the generation is long enough for the two backends to separate, the dense cuda arm \
         stops writing about the same subject as the oracle -- it reuses only {shared} of {} \
         distinct content words. That is what serving garbage looks like on a checkpoint whose \
         short answers still match. cuda {:?} vs wgpu {:?}",
        oracle_content_words.len(),
        cu_long.stream(),
        or_long.stream()
    );
}
