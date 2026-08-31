#![cfg(feature = "wgpu")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use candle_core::{Device, Tensor};
use http_body_util::BodyExt;
use tower::util::ServiceExt;

use speaches_plus::oapi::chat::{ChatAppState, ChatEngine, ChatEvent, ChatGenerateRequest};
use speaches_plus::oapi::chat_engine::{ChatRegistry, EchoEngine};
use speaches_plus::oapi::chat_engine_wgpu::persist::adapter_tag_that_changes_when_the_weights_change;
use speaches_plus::oapi::lora::{AdapterEntry, ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE};
use speaches_plus::oapi::{lora, lora_runtime};

const REAL_WEIGHTS_TEST_ENV: &str = "NV_LORA_LIFECYCLE_TEST";

fn scratch(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lora-life-{label}-{}-{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn greedy_request(prompt: &str, max_new: usize) -> ChatGenerateRequest {
    ChatGenerateRequest {
        prompt: prompt.to_string(),
        max_new_tokens: max_new,
        stop: Vec::new(),
        seed: Some(1),
        temperature: None,
        top_p: None,
        top_k: None,
        min_p: None,
        presence_penalty: None,
        frequency_penalty: None,
        repetition_penalty: None,
        guided: None,
        guided_think_close: None,
        logit_bias: Vec::new(),
        logprobs: false,
        top_logprobs: 0,
        kv_resume: None,
        kv_store: None,
        mm: None,
    }
}

async fn generated_text(eng: &Arc<dyn ChatEngine>, prompt: &str, max_new: usize) -> String {
    let (tx, mut rx) = tokio::sync::mpsc::channel(256);
    eng.generate(greedy_request(prompt, max_new), tx)
        .await
        .expect("generate");
    let mut out = String::new();
    while let Some(ev) = rx.recv().await {
        if let ChatEvent::TextDelta(s) = ev {
            out.push_str(&s);
        }
    }
    out
}

fn lcg_vals(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn write_peft_adapter(
    dir: &Path,
    rank: usize,
    base_model: Option<&str>,
    modules: &[(String, usize, usize)],
    seed: u64,
    b_scale: f32,
) {
    std::fs::create_dir_all(dir).unwrap();
    let cfg = serde_json::json!({
        "r": rank,
        "lora_alpha": (2 * rank) as f64,
        "target_modules": ["q_proj"],
        "base_model_name_or_path": base_model,
    });
    std::fs::write(dir.join(ADAPTER_CONFIG_FILE), cfg.to_string()).unwrap();
    let mut tensors = std::collections::HashMap::new();
    for (i, (name, k, out)) in modules.iter().enumerate() {
        let a = Tensor::from_vec(
            lcg_vals(rank * k, seed.wrapping_add(i as u64 * 2 + 1), 0.05),
            (rank, *k),
            &Device::Cpu,
        )
        .unwrap();
        let b = Tensor::from_vec(
            lcg_vals(out * rank, seed.wrapping_add(i as u64 * 2 + 2), b_scale),
            (*out, rank),
            &Device::Cpu,
        )
        .unwrap();
        tensors.insert(format!("{name}.lora_A.weight"), a);
        tensors.insert(format!("{name}.lora_B.weight"), b);
    }
    candle_core::safetensors::save(&tensors, dir.join(ADAPTER_WEIGHTS_FILE)).unwrap();
}

#[test]
fn unloading_an_adapter_leaves_the_base_engine_arc_bit_identical_to_boot() {
    let base: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new("life-r-base", "base reply"));
    let reg = ChatRegistry::single(base);
    let before = reg.resolve_with(Some("life-r-base"), false).expect("base");
    reg.insert(Arc::new(EchoEngine::new("life-r-ada", "adapter reply")));
    assert!(reg.contains("life-r-ada"));
    reg.remove("life-r-ada").expect("remove adapter");
    let after = reg.resolve_with(Some("life-r-base"), false).expect("base");
    assert!(
        Arc::ptr_eq(&before, &after),
        "the registry returned a different base engine object after a load/unload cycle; the \
         engine-per-adapter design promises the base engine is never touched, which is the \
         whole bitwise-restoration argument"
    );
    assert!(
        reg.resolve_with(Some("life-r-ada"), false).is_none(),
        "an unloaded adapter id must resolve to nothing, not fall back silently"
    );
}

#[tokio::test]
async fn a_request_in_flight_across_an_inplace_reload_finishes_on_its_engine_and_new_requests_see_the_new_one(
) {
    let reg = ChatRegistry::single(Arc::new(EchoEngine::new("life-s-base", "x")));
    reg.insert(Arc::new(EchoEngine::new("life-s-ada", "old adapter reply")));
    let in_flight = reg.resolve_with(Some("life-s-ada"), false).expect("v1");
    reg.insert(Arc::new(EchoEngine::new("life-s-ada", "new adapter reply")));
    let fresh = reg.resolve_with(Some("life-s-ada"), false).expect("v2");
    assert!(
        !Arc::ptr_eq(&in_flight, &fresh),
        "an inplace reload must replace the engine object"
    );
    assert_eq!(
        generated_text(&in_flight, "hi", 32).await,
        "old adapter reply",
        "a request that resolved before the swap must finish on the adapter it started with"
    );
    assert_eq!(
        generated_text(&fresh, "hi", 32).await,
        "new adapter reply",
        "a request that resolves after the swap must see only the new adapter"
    );
}

#[test]
fn the_kv_adapter_tag_rotates_on_a_same_length_same_mtime_overwrite_and_is_deterministic_for_identical_bytes(
) {
    let dir = scratch("tag");
    let weights = dir.join(ADAPTER_WEIGHTS_FILE);
    let pin = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
    let pin_mtime = |p: &Path| {
        let f = std::fs::OpenOptions::new().write(true).open(p).unwrap();
        f.set_times(std::fs::FileTimes::new().set_modified(pin))
            .unwrap();
    };
    std::fs::write(&weights, b"aaaaaaaaaaaaaaaa").unwrap();
    pin_mtime(&weights);
    let t1 = adapter_tag_that_changes_when_the_weights_change(&dir);
    std::fs::write(&weights, b"bbbbbbbbbbbbbbbb").unwrap();
    pin_mtime(&weights);
    let t2 = adapter_tag_that_changes_when_the_weights_change(&dir);
    assert_ne!(
        t1, t2,
        "a fine-tune round that rewrites adapter weights with the same byte length inside one \
         mtime second must still rotate the KV tag, or a reload resumes stale KV onto new weights"
    );
    std::fs::write(&weights, b"aaaaaaaaaaaaaaaa").unwrap();
    let t3 = adapter_tag_that_changes_when_the_weights_change(&dir);
    assert_eq!(
        t1, t3,
        "identical bytes must produce the identical tag regardless of mtime, so a restart alone \
         does not throw away a warm snapshot"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_wrong_shape_or_unmatched_adapter_is_refused_loudly_with_no_partial_application() {
    use nv_models::qwen3_5_dense_wgpu::{Q3dLora, Qwen3_5DenseConfig};
    let refusal = |r: anyhow::Result<Q3dLora>, why: &str| -> String {
        match r {
            Ok(_) => panic!("{why}"),
            Err(e) => format!("{e:#}"),
        }
    };
    let mut tiny: serde_json::Value =
        serde_json::from_str(&nv_train::qwen38_fixture::tiny_config_json()).unwrap();
    tiny["text_config"]["eos_token_id"] = serde_json::json!(0);
    tiny["text_config"]["max_position_embeddings"] = serde_json::json!(4096);
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&tiny.to_string())
        .expect("tiny fixture config parses for the serving graph once eos_token_id is present");
    let full_li = 3usize;
    let hidden = 32usize;
    let q_out_gated = 32usize;

    let linear_li_where_self_attn_has_no_projection = 0usize;
    let unmatched = scratch("unmatched");
    write_peft_adapter(
        &unmatched,
        4,
        None,
        &[(
            format!(
                "model.language_model.layers.{linear_li_where_self_attn_has_no_projection}.self_attn.q_proj"
            ),
            hidden,
            q_out_gated,
        )],
        7,
        0.05,
    );
    let err = refusal(
        Q3dLora::from_peft_dir(&unmatched, &cfg),
        "an adapter whose modules match no serving projection must refuse, not load as a silent no-op",
    );
    assert!(
        err.contains("matched no"),
        "unmatched-adapter refusal must say so: {err}"
    );

    let wrong_k = scratch("wrongk");
    write_peft_adapter(
        &wrong_k,
        4,
        None,
        &[(
            format!("model.language_model.layers.{full_li}.self_attn.q_proj"),
            hidden + 2,
            q_out_gated,
        )],
        7,
        0.05,
    );
    let err = refusal(
        Q3dLora::from_peft_dir(&wrong_k, &cfg),
        "an adapter trained against a different hidden size must refuse loudly",
    );
    assert!(!err.is_empty(), "shape refusal must carry a message");

    let ungated_b = scratch("gatelanes");
    write_peft_adapter(
        &ungated_b,
        4,
        None,
        &[(
            format!("model.language_model.layers.{full_li}.self_attn.q_proj"),
            hidden,
            q_out_gated / 2,
        )],
        7,
        0.05,
    );
    let err = refusal(
        Q3dLora::from_peft_dir(&ungated_b, &cfg),
        "a q_proj adapter that covers only the value lanes of a gated projection must refuse",
    );
    assert!(
        err.contains("gate") || err.contains("lora"),
        "gate-lane refusal must name the problem: {err}"
    );
    for d in [unmatched, wrong_k, ungated_b] {
        let _ = std::fs::remove_dir_all(&d);
    }
}

#[test]
fn the_adapter_loader_rereads_bytes_from_disk_so_an_inplace_reload_serves_the_new_weights() {
    let dir = scratch("fresh");
    let module = "model.language_model.layers.3.self_attn.q_proj".to_string();
    write_peft_adapter(&dir, 4, None, &[(module.clone(), 32, 32)], 11, 0.05);
    let v1 = nv_weights::lora_adapter::LoraAdapter::load(&dir, &Device::Cpu).expect("v1 loads");
    write_peft_adapter(&dir, 4, None, &[(module.clone(), 32, 32)], 22, 0.05);
    let v2 = nv_weights::lora_adapter::LoraAdapter::load(&dir, &Device::Cpu).expect("v2 loads");
    let sum = |ad: &nv_weights::lora_adapter::LoraAdapter| -> f32 {
        let w = &ad.loras[&module];
        w.lora_a
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };
    assert_ne!(
        sum(&v1),
        sum(&v2),
        "reloading after an in-place overwrite returned the old tensors; every runtime load \
         must reread the adapter dir from disk"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

async fn body_json(resp: axum::response::Response) -> (StatusCode, serde_json::Value) {
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes).into_owned();
    (
        status,
        serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text)),
    )
}

fn post(uri: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn the_adapters_list_route_reports_catalog_status_and_the_update_routes_mount_only_under_the_flag(
) {
    std::env::remove_var(lora_runtime::RUNTIME_LORA_ENV);
    let base: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new("life-h-base", "b"));
    let reg = ChatRegistry::single(base);
    let state = ChatAppState {
        registry: reg.clone(),
    };
    let app_off = lora_runtime::routes().with_state(state.clone());
    let (status, v) = body_json(
        app_off
            .clone()
            .oneshot(Request::builder().uri("/v1/adapters").body(Body::empty()).unwrap())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["object"], "list");
    let (status, _) = body_json(
        app_off
            .oneshot(post(
                "/v1/load_lora_adapter",
                serde_json::json!({"lora_name": "life-h-x", "lora_path": "/nonexistent"}),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "without {} the update routes must not exist while the read-only list still serves",
        lora_runtime::RUNTIME_LORA_ENV
    );

    std::env::set_var(lora_runtime::RUNTIME_LORA_ENV, "1");
    let app_on = lora_runtime::routes().with_state(state.clone());
    std::env::remove_var(lora_runtime::RUNTIME_LORA_ENV);

    let (status, v) = body_json(
        app_on
            .clone()
            .oneshot(post(
                "/v1/load_lora_adapter",
                serde_json::json!({"lora_name": "life-h-x", "lora_path": "/nonexistent-lora-dir"}),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "bad lora_path must 400: {v}");

    let (status, _) = body_json(
        app_on
            .clone()
            .oneshot(post(
                "/v1/unload_lora_adapter",
                serde_json::json!({"lora_name": "life-h-neverloaded"}),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, v) = body_json(
        app_on
            .clone()
            .oneshot(post(
                "/v1/unload_lora_adapter",
                serde_json::json!({"lora_name": "life-h-base"}),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "unloading a base model must refuse: {v}"
    );

    let dir = scratch("list");
    lora::register_adapter(AdapterEntry {
        id: "life-h-ada".into(),
        dir: dir.clone(),
        base_model: Some("life-h-base".into()),
        rank: 4,
        alpha: 8.0,
        scaling: 2.0,
        targets: vec!["q_proj".into()],
    });
    reg.insert(Arc::new(EchoEngine::new("life-h-ada", "a")));
    lora_runtime::mark_loaded_now("life-h-ada");
    let (status, v) = body_json(
        app_on
            .clone()
            .oneshot(Request::builder().uri("/v1/adapters").body(Body::empty()).unwrap())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let row = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == "life-h-ada")
        .expect("a registered adapter must appear in /v1/adapters")
        .clone();
    assert_eq!(row["status"], "loaded");
    assert_eq!(row["base_model"], "life-h-base");
    assert_eq!(row["rank"], 4);
    assert!(
        row["loaded_at"].as_u64().is_some(),
        "a runtime-loaded adapter must report loaded_at"
    );

    let before = reg.resolve_with(Some("life-h-base"), false).unwrap();
    let (status, _) = body_json(
        app_on
            .clone()
            .oneshot(post(
                "/v1/unload_lora_adapter",
                serde_json::json!({"lora_name": "life-h-ada"}),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (_, v) = body_json(
        app_on
            .oneshot(Request::builder().uri("/v1/adapters").body(Body::empty()).unwrap())
            .await
            .unwrap(),
    )
    .await;
    assert!(
        !v["data"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["id"] == "life-h-ada"),
        "an unloaded adapter must leave the list, not linger as a ghost row"
    );
    let after = reg.resolve_with(Some("life-h-base"), false).unwrap();
    assert!(Arc::ptr_eq(&before, &after));
    let _ = std::fs::remove_dir_all(&dir);
}

fn cfg_scope(v: &serde_json::Value) -> &serde_json::Value {
    if v.get("text_config").is_some() {
        &v["text_config"]
    } else {
        v
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "boots the real wgpu base named by NV_WGPU_CHAT_MODEL_DIRS, then drives the full \
            adapter lifecycle over the real routes: load a crafted q_proj adapter, in-place \
            reload with rewritten bytes, fresh-load the same bytes under a new id, unload \
            everything, and require the base's greedy output bitwise equal to the never-loaded \
            run and the reload equal to the fresh load; set NV_LORA_LIFECYCLE_TEST=1"]
async fn full_lifecycle_on_a_real_wgpu_base_restores_bitwise_base_output_and_reload_serves_new_bytes(
) {
    if std::env::var(REAL_WEIGHTS_TEST_ENV).ok().as_deref() != Some("1") {
        panic!(
            "{REAL_WEIGHTS_TEST_ENV}=1 not set. Returning green here would report a lifecycle \
             proof that never ran; this is a SKIP surfaced as a failure by design."
        );
    }
    std::env::set_var(lora_runtime::RUNTIME_LORA_ENV, "1");
    let reg = speaches_plus::oapi::chat_engine_wgpu::registry_from_env_with_wgpu()
        .expect("NV_WGPU_CHAT_MODEL_DIRS must name a loadable wgpu base");
    let bases = speaches_plus::oapi::chat_engine_wgpu::wgpu_base_dirs();
    assert!(!bases.is_empty(), "boot must publish wgpu base dirs");
    let (base_id, base_dir) = bases[0].clone();
    let raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(base_dir.join("config.json")).unwrap())
            .unwrap();
    let scope = cfg_scope(&raw);
    let hidden = scope["hidden_size"].as_u64().unwrap() as usize;
    let heads = scope["num_attention_heads"].as_u64().unwrap() as usize;
    let head_dim = scope["head_dim"].as_u64().unwrap() as usize;
    let gated = scope["attn_output_gate"].as_bool().unwrap_or(false);
    let q_out = heads * head_dim * if gated { 2 } else { 1 };
    let full_li = scope["layer_types"]
        .as_array()
        .map(|l| {
            l.iter()
                .position(|t| t == "full_attention")
                .expect("a full_attention layer exists")
        })
        .unwrap_or(0);
    let module = format!("model.language_model.layers.{full_li}.self_attn.q_proj");

    let state = ChatAppState {
        registry: reg.clone(),
    };
    let app = lora_runtime::routes().with_state(state);

    let prompt = "The capital of France is";
    let base_eng = reg.resolve_with(Some(&base_id), false).expect("base");
    let text_base0 = generated_text(&base_eng, prompt, 8).await;

    let v1 = scratch("e2e-v1");
    write_peft_adapter(
        &v1,
        4,
        Some(&base_id),
        &[(module.clone(), hidden, q_out)],
        1001,
        0.2,
    );
    let (status, body) = body_json(
        app.clone()
            .oneshot(post(
                "/v1/load_lora_adapter",
                serde_json::json!({"lora_name": "life-e2e", "lora_path": v1.display().to_string()}),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "v1 load failed: {body}");
    let ada1 = reg.resolve_with(Some("life-e2e"), false).expect("adapter engine");
    let text_v1 = generated_text(&ada1, prompt, 8).await;

    write_peft_adapter(
        &v1,
        4,
        Some(&base_id),
        &[(module.clone(), hidden, q_out)],
        2002,
        0.2,
    );
    let (status, body) = body_json(
        app.clone()
            .oneshot(post(
                "/v1/load_lora_adapter",
                serde_json::json!({"lora_name": "life-e2e", "lora_path": v1.display().to_string(), "load_inplace": true}),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "in-place reload failed: {body}");
    let ada2 = reg.resolve_with(Some("life-e2e"), false).expect("reloaded engine");
    assert!(
        !Arc::ptr_eq(&ada1, &ada2),
        "reload must build a fresh engine from the rewritten bytes"
    );
    let text_v2 = generated_text(&ada2, prompt, 8).await;

    let (status, _) = body_json(
        app.clone()
            .oneshot(post(
                "/v1/unload_lora_adapter",
                serde_json::json!({"lora_name": "life-e2e"}),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, body) = body_json(
        app.clone()
            .oneshot(post(
                "/v1/load_lora_adapter",
                serde_json::json!({"lora_name": "life-e2e-fresh", "lora_path": v1.display().to_string()}),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "fresh v2 load failed: {body}");
    let fresh = reg
        .resolve_with(Some("life-e2e-fresh"), false)
        .expect("fresh engine");
    let text_fresh = generated_text(&fresh, prompt, 8).await;
    assert_eq!(
        text_v2, text_fresh,
        "an in-place reload and a fresh load of the same bytes must serve identical greedy output"
    );
    if text_v1 == text_v2 {
        eprintln!(
            "note: v1 and v2 adapters produced identical greedy text over 8 tokens; the reload \
             gate above still holds via engine replacement + fresh-load equality"
        );
    }

    let (status, _) = body_json(
        app.clone()
            .oneshot(post(
                "/v1/unload_lora_adapter",
                serde_json::json!({"lora_name": "life-e2e-fresh"}),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let text_base1 = generated_text(&base_eng, prompt, 8).await;
    assert_eq!(
        text_base0, text_base1,
        "after load, reload and unload the base engine's greedy output must be bitwise the \
         never-loaded output"
    );

    let bad = scratch("e2e-bad");
    write_peft_adapter(
        &bad,
        4,
        Some(&base_id),
        &[(module.clone(), hidden + 2, q_out)],
        3003,
        0.2,
    );
    let (status, body) = body_json(
        app.clone()
            .oneshot(post(
                "/v1/load_lora_adapter",
                serde_json::json!({"lora_name": "life-e2e-bad", "lora_path": bad.display().to_string()}),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert!(
        status == StatusCode::INTERNAL_SERVER_ERROR || status == StatusCode::BAD_REQUEST,
        "a wrong-shape adapter must be refused, got {status}: {body}"
    );
    assert!(
        reg.resolve_with(Some("life-e2e-bad"), false).is_none(),
        "a refused adapter must leave no engine behind"
    );
    for d in [v1, bad] {
        let _ = std::fs::remove_dir_all(&d);
    }
}
