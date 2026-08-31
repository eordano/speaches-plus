#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{Device, Tensor};

use speaches_plus::oapi::chat::{ChatEngine, ChatEvent, ChatGenerateRequest};
use speaches_plus::oapi::chat_engine::{ChatRegistry, CudaLoraAdapterEngine, NvEngineChat};
use speaches_plus::oapi::lora::{ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE};

const REAL_WEIGHTS_TEST_ENV: &str = "NV_LORA_LAGUNA_SERVING_TEST";
const LAGUNA_XS_SNAPSHOT_ON_THIS_BOX: &str =
    ".cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4/snapshots/main";

fn scratch(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lora-laguna-serving-{label}-{}-{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
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

fn fake_checkpoint_dir(label: &str, config_json: &str) -> PathBuf {
    let dir = scratch(label);
    std::fs::write(dir.join("config.json"), config_json).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    std::fs::write(dir.join("model.safetensors"), b"not-a-real-checkpoint").unwrap();
    dir
}

#[test]
fn lean_build_family_gate_refuses_qwen3_before_touching_weights_and_names_the_allowlist() {
    let dir = fake_checkpoint_dir(
        "qwen3-gate",
        r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}"#,
    );
    let ada = scratch("qwen3-gate-ada");
    let err = CudaLoraAdapterEngine::try_load(&dir, "gate-probe", &ada)
        .err()
        .expect("a qwen3 base must refuse adapter serving at the family gate");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("gemma4") && msg.contains("laguna") && msg.contains("hookable Linears"),
        "the refusal must name the gemma4/laguna allowlist and the hookable-Linear reason so \
         an operator knows which bases can serve adapters on cuda, got: {msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&ada);
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
    let mut engine_error = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            ChatEvent::TextDelta(s) => out.push_str(&s),
            ChatEvent::Error(e) => engine_error = Some(e),
            _ => {}
        }
    }
    if let Some(e) = engine_error {
        panic!(
            "engine emitted an error instead of text (asserted only after the stream closed, \
             so the generation thread is quiescent): {e}"
        );
    }
    assert!(
        !out.is_empty(),
        "an empty greedy generation makes every comparison in this suite vacuous"
    );
    out
}

async fn warm_first_request_jit(eng: &Arc<dyn ChatEngine>, prompt: &str) {
    let (tx, mut rx) = tokio::sync::mpsc::channel(256);
    eng.generate(greedy_request(prompt, 2), tx)
        .await
        .expect("warmup generate");
    while rx.recv().await.is_some() {}
}

mod real_weight_gates {
    use super::*;
    use nv_specdecode::lora_spec::serving_hook_engagement_counted_host_side_so_replays_of_a_captured_graph_leave_it_flat as engagement;

    static ONE_GATE_AT_A_TIME_THE_VRAM_PLAN_SUM_IS_PROCESS_WIDE: std::sync::Mutex<()> =
        std::sync::Mutex::new(());

    fn serialize_gates() -> std::sync::MutexGuard<'static, ()> {
        ONE_GATE_AT_A_TIME_THE_VRAM_PLAN_SUM_IS_PROCESS_WIDE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn require_gate() {
        if std::env::var(REAL_WEIGHTS_TEST_ENV).ok().as_deref() != Some("1") {
            panic!(
                "{REAL_WEIGHTS_TEST_ENV}=1 not set. Returning green here would report a laguna \
                 cuda lora serving proof that never ran; this is a SKIP surfaced as a failure \
                 by design."
            );
        }
        if std::env::var_os("NV_KV_MAX_SEQ_LEN_LAGUNA").is_none() {
            std::env::set_var("NV_KV_MAX_SEQ_LEN_LAGUNA", "8192");
            eprintln!(
                "NV_KV_MAX_SEQ_LEN_LAGUNA=8192 pinned by the gate: laguna's per-request eager \
                 KV cache is sized from this ceiling, and the checkpoint's 262144 \
                 max_position_embeddings would allocate a cache this box cannot hold twice"
            );
        }
        let inherited_spec = std::env::var("NV_LAGUNA_SERVE_SPEC").ok();
        std::env::set_var("NV_LAGUNA_SERVE_SPEC", "0");
        match inherited_spec.as_deref() {
            Some(v) if v != "0" => eprintln!(
                "NV_LAGUNA_SERVE_SPEC={v} inherited from the environment was FORCED to 0 by \
                 the gate: with it set the BASE engine would build the spec-serve step graph \
                 and decode eligible greedy requests through it while the adapter engine \
                 stays eager, so base-vs-adapter would compare two different decode paths \
                 and the assert_ne could pass for the wrong reason"
            ),
            _ => eprintln!(
                "NV_LAGUNA_SERVE_SPEC=0 pinned by the gate: the BASE engine's spec-serve step \
                 graph costs VRAM and threads this A/B never exercises, and the adapter \
                 engine never builds it; both engines therefore decode on the same eager \
                 per-request path, which is what the bitwise restore compares"
            ),
        }
    }

    fn required_base_dir() -> PathBuf {
        let raw = std::env::var_os("NV_CHAT_MODEL_DIR").unwrap_or_else(|| {
            panic!(
                "NV_CHAT_MODEL_DIR must point at a laguna cuda checkpoint; on this box: \
                 $HOME/{LAGUNA_XS_SNAPSHOT_ON_THIS_BOX}"
            )
        });
        let p = PathBuf::from(raw);
        assert!(p.is_dir(), "NV_CHAT_MODEL_DIR={} is not a directory", p.display());
        p
    }

    fn laguna_dims(base_dir: &Path) -> (usize, usize, usize, usize) {
        let raw = std::fs::read_to_string(base_dir.join("config.json")).expect("base config");
        let cfg =
            nv_models::laguna::LagunaConfig::from_hf_json_str(&raw).expect("laguna config parses");
        assert!(
            !cfg.is_moe_layer(0),
            "this gate hooks layer 0's dense mlp; the checkpoint's layer 0 is MoE, so pick a \
             dense layer from mlp_only_layers/mlp_layer_types and update the adapter writer"
        );
        (
            cfg.hidden_size,
            cfg.num_heads_for_layer(0) * cfg.head_dim,
            cfg.intermediate_size,
            cfg.num_hidden_layers,
        )
    }

    fn write_laguna_adapter(dir: &Path, base_dir: &Path, seed: u64) {
        std::fs::create_dir_all(dir).unwrap();
        let (hidden, q_out, inter, _layers) = laguna_dims(base_dir);
        let rank = 8usize;
        let cfg = serde_json::json!({
            "r": rank,
            "lora_alpha": (2 * rank) as f64,
            "target_modules": ["q_proj", "down_proj"],
            "base_model_name_or_path": null,
        });
        std::fs::write(dir.join(ADAPTER_CONFIG_FILE), cfg.to_string()).unwrap();
        let mut tensors = std::collections::HashMap::new();
        let dev = Device::Cpu;
        let mut push = |module: &str, in_f: usize, out_f: usize, s: u64| {
            let a = Tensor::from_vec(
                lcg_vals(rank * in_f, s.wrapping_add(1), 0.05),
                (rank, in_f),
                &dev,
            )
            .unwrap();
            let b = Tensor::from_vec(
                lcg_vals(out_f * rank, s.wrapping_add(2), 0.5),
                (out_f, rank),
                &dev,
            )
            .unwrap();
            tensors.insert(format!("{module}.lora_A.weight"), a);
            tensors.insert(format!("{module}.lora_B.weight"), b);
        };
        push("model.layers.0.self_attn.q_proj", hidden, q_out, seed);
        push(
            "model.layers.0.mlp.down_proj",
            inter,
            hidden,
            seed.wrapping_add(100),
        );
        candle_core::safetensors::save(&tensors, dir.join(ADAPTER_WEIGHTS_FILE)).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "boots the real laguna cuda base named by NV_CHAT_MODEL_DIR twice (base + \
                adapter engine), then requires: adapter greedy output differs from base, the \
                m=1 hook kernel engaged during adapter decode with zero applies inside an \
                open capture (laguna's only eager-path captures are the MoE-layer graphs, \
                which contain no hooked Linear), registry unload restores base greedy output \
                bitwise, and the unloaded id resolves to nothing; set \
                NV_LORA_LAGUNA_SERVING_TEST=1. Run: \
                NVK_LANE=gapc NVK_PKG=speaches-plus NV_LORA_LAGUNA_SERVING_TEST=1 \
                NV_CHAT_MODEL_DIR=$HOME/.cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4/snapshots/main \
                rust/scripts/nvk.sh test --test lora_laguna_cuda_serving -- --ignored \
                --test-threads=1 --nocapture"]
    async fn laguna_adapter_serves_differently_hooks_engage_outside_captures_and_unload_restores_base_bitwise(
    ) {
        let _one_gate = serialize_gates();
        require_gate();
        let base_dir = required_base_dir();
        let base: Arc<dyn ChatEngine> =
            Arc::new(NvEngineChat::try_load(&base_dir).expect("base engine loads"));
        let reg = ChatRegistry::single(base.clone());
        let prompt = "The capital of France is";
        warm_first_request_jit(&base, prompt).await;
        let text_base0 = generated_text(&base, prompt, 8).await;

        let dir = scratch("laguna-e2e");
        write_laguna_adapter(&dir, &base_dir, 1001);
        let ada: Arc<dyn ChatEngine> = Arc::new(
            CudaLoraAdapterEngine::try_load(&base_dir, "lora-laguna-e2e", &dir)
                .expect("adapter engine loads"),
        );
        reg.insert(ada.clone());
        let eng_before = engagement();
        let text_ada = generated_text(&ada, prompt, 8).await;
        let eng_after = engagement();
        assert_ne!(
            text_base0, text_ada,
            "a rank-8 q_proj+down_proj delta at b_scale 0.5 must change greedy output; \
             equality means the hooks never fired in the laguna eager decode path (check the \
             w8 scope was zeroed at install)"
        );
        assert!(
            eng_after.m1_kernel_applies > eng_before.m1_kernel_applies,
            "adapter decode must engage the m=1 two-phase kernel path; a flat counter with a \
             changed output means the candle fallback served decode, which is the slow path \
             and must be investigated"
        );
        assert_eq!(
            eng_after.m1_applies_inside_an_open_capture,
            eng_before.m1_applies_inside_an_open_capture,
            "no hook apply may land inside an open capture on laguna: the eager path's only \
             captures are the MoE-layer graphs, whose bodies contain no hooked Linear, and \
             the adapter engine never builds the spec-serve step graph"
        );

        reg.remove("lora-laguna-e2e").expect("unload adapter");
        drop(ada);
        let text_base1 = generated_text(&base, prompt, 8).await;
        assert_eq!(
            text_base0, text_base1,
            "after adapter load and unload the base engine's greedy output must be bitwise \
             the never-loaded output; the base engine object was never mutated"
        );
        assert!(reg.resolve_with(Some("lora-laguna-e2e"), false).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[ignore = "builds the laguna cuda adapter engine, drops it, rebuilds from the same dir \
                with rewritten same-length weights: the rebuild must be ADMITTED (the dropped \
                engine's RecordedBootPlan must have left the enforced sum) and the carried KV \
                tag must rotate; set NV_LORA_LAGUNA_SERVING_TEST=1. Run: \
                NVK_LANE=gapc NVK_PKG=speaches-plus NV_LORA_LAGUNA_SERVING_TEST=1 \
                NV_CHAT_MODEL_DIR=$HOME/.cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4/snapshots/main \
                rust/scripts/nvk.sh test --test lora_laguna_cuda_serving -- --ignored \
                --test-threads=1 --nocapture"]
    fn laguna_same_id_reload_with_changed_bytes_rotates_the_kv_tag_and_is_readmitted() {
        let _one_gate = serialize_gates();
        require_gate();
        let base_dir = required_base_dir();
        let dir = scratch("laguna-reload");
        write_laguna_adapter(&dir, &base_dir, 1001);
        let e1 = CudaLoraAdapterEngine::try_load(&base_dir, "lora-laguna-reload", &dir)
            .expect("v1 engine");
        let tag1 = e1
            .adapter_kv_tag_that_any_future_cuda_kv_persist_must_key_on()
            .to_string();
        drop(e1);
        write_laguna_adapter(&dir, &base_dir, 2002);
        let e2 = CudaLoraAdapterEngine::try_load(&base_dir, "lora-laguna-reload", &dir).expect(
            "v2 engine must be admitted after v1 dropped: a vram-plan refusal here means the \
             dropped laguna engine's RecordedBootPlan did not leave the process-wide enforced \
             sum",
        );
        let tag2 = e2
            .adapter_kv_tag_that_any_future_cuda_kv_persist_must_key_on()
            .to_string();
        assert_ne!(
            tag1, tag2,
            "a same-id reload with changed adapter bytes must rotate the KV tag; a stable tag \
             would let any future cuda KV persistence resume stale KV onto new weights"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
