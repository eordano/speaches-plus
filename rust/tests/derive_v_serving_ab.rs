#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::Router;
use serde_json::{json, Value};
use tower::ServiceExt;

use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
use speaches_plus::oapi::chat_engine::{ChatRegistry, NvEngineChat};

fn resolve_model_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_CHAT_MODEL_DIR") {
        let p = PathBuf::from(d);
        if p.is_dir() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let root = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    std::fs::read_dir(&root)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").is_file())
}

async fn arm(derive: bool) -> String {
    let Some(dir) = resolve_model_dir() else {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_CHAT_MODEL_DIR")
    };

    unsafe {
        if derive {
            std::env::set_var("NV_KV_DERIVE_V", "1");
        } else {
            std::env::remove_var("NV_KV_DERIVE_V");
        }
    }

    let eng: Arc<dyn ChatEngine> = match NvEngineChat::try_load(Path::new(&dir)) {
        Ok(e) => Arc::new(e),
        Err(err) => panic!(
            "NvEngineChat::try_load({}) failed: {err:#}. The checkpoint is present, so this \
             is a FAILURE, not a skip.",
            dir.display()
        ),
    };
    let model = eng.model_id().to_string();
    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .with_state(ChatAppState {
            registry: ChatRegistry::single(eng),
        });

    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "List the first six prime numbers."}],
        "temperature": 0.0,
        "max_tokens": 48
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.expect("handler");
    assert_eq!(resp.status(), StatusCode::OK, "derive={derive}");
    let bytes = axum::body::to_bytes(resp.into_body(), 8 << 20)
        .await
        .expect("body");
    let v: Value = serde_json::from_slice(&bytes).expect("json");
    unsafe { std::env::remove_var("NV_KV_DERIVE_V") };
    v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

fn vram_mib() -> u64 {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .expect("nvidia-smi");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .lines()
        .next()
        .and_then(|l| l.trim().parse().ok())
        .unwrap_or(u64::MAX)
}

fn gate() {
    if std::env::var("NV_DERIVE_V_SERVE").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_DERIVE_V_SERVE=1");
    }
    assert!(
        std::env::var("NV_BATCH_ENGINE").is_ok(),
        "PRECONDITION NOT MET: the derive-V pool lives in the batch engine, so this test is \
         meaningless without NV_BATCH_ENGINE set -- it would compare two identical non-batch \
         runs and pass"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn the_server_generates_the_same_text_with_v_derived() {
    gate();
    let start = vram_mib();
    assert!(
        start < 2048,
        "GPU is not idle ({start} MiB used); this test loads the 31B twice and would OOM \
         or measure someone else's allocation"
    );

    let off = arm(false).await;
    assert!(!off.trim().is_empty(), "OFF arm produced nothing");

    let between = vram_mib();
    assert!(
        between < start + 2048,
        "the OFF engine did not release the GPU ({between} MiB still used, started at \
         {start}); the second arm would OOM and the failure would be blamed on derive-V"
    );

    let on = arm(true).await;
    assert!(!on.trim().is_empty(), "ON arm produced nothing");

    eprintln!("[derive-serve] OFF={off:?}");
    eprintln!("[derive-serve] ON ={on:?}");

    for needle in ["2", "3", "5", "7", "11", "13"] {
        assert!(
            on.contains(needle),
            "the derived arm lost {needle:?} from the answer: {on:?}"
        );
    }
    assert!(
        on.len() > off.len() / 2 && on.len() < off.len() * 2,
        "the derived arm's answer is a wildly different length, which is what a \
         broken derive looks like.\n OFF={off:?}\n ON ={on:?}"
    );
}
