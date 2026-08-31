use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};

use nv_layers::linear::Linear;
use nv_specdecode::lora_spec::{ServingDeltaSlice, ServingLoraHook};
use speaches_plus::oapi::chat::{ChatEngine, ChatEvent, ChatGenerateRequest};
use speaches_plus::oapi::chat_engine::{ChatRegistry, EchoEngine};
use speaches_plus::oapi::lora::{ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE};

const REAL_WEIGHTS_TEST_ENV: &str = "NV_LORA_CUDA_SERVING_TEST";

fn scratch(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lora-cuda-{label}-{}-{:x}",
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

fn bf16_tensor(seed: u64, rows: usize, cols: usize, scale: f32) -> Tensor {
    Tensor::from_vec(lcg_vals(rows * cols, seed, scale), (rows, cols), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

fn bits_of(t: &Tensor) -> Vec<u32> {
    t.contiguous()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
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
            "engine emitted an error instead of text (asserted only after the stream \
             closed, so the generation thread is quiescent and the engine can drop \
             without deadlocking the harness): {e}"
        );
    }
    assert!(
        !out.is_empty(),
        "an empty greedy generation makes every comparison in this suite vacuous; \
         if this is first-request warmup, warm the engine before capturing"
    );
    out
}

async fn warm_first_request_jit_so_captures_compare_steady_state(
    eng: &Arc<dyn ChatEngine>,
    prompt: &str,
) {
    let (tx, mut rx) = tokio::sync::mpsc::channel(256);
    eng.generate(greedy_request(prompt, 2), tx)
        .await
        .expect("warmup generate");
    while rx.recv().await.is_some() {}
}

#[test]
fn cuda_arm_unload_leaves_the_base_engine_arc_bit_identical_to_boot() {
    let base: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new("cuda-r-base", "base reply"));
    let reg = ChatRegistry::single(base);
    let before = reg.resolve_with(Some("cuda-r-base"), false).expect("base");
    reg.insert(Arc::new(EchoEngine::new("cuda-r-ada", "adapter reply")));
    assert!(reg.contains("cuda-r-ada"));
    reg.remove("cuda-r-ada").expect("remove adapter");
    let after = reg.resolve_with(Some("cuda-r-base"), false).expect("base");
    assert!(
        Arc::ptr_eq(&before, &after),
        "the registry returned a different base engine object after a load/unload cycle; the \
         cuda arm inserts whole engines under the same engine-per-adapter contract, so the base \
         engine must never be touched"
    );
    assert!(
        reg.resolve_with(Some("cuda-r-ada"), false).is_none(),
        "an unloaded adapter id must resolve to nothing, not fall back silently"
    );
}

#[tokio::test]
async fn cuda_arm_inplace_reload_finishes_in_flight_on_old_engine_and_new_requests_see_new() {
    let reg = ChatRegistry::single(Arc::new(EchoEngine::new("cuda-s-base", "x")));
    reg.insert(Arc::new(EchoEngine::new("cuda-s-ada", "old adapter reply")));
    let in_flight = reg.resolve_with(Some("cuda-s-ada"), false).expect("v1");
    reg.insert(Arc::new(EchoEngine::new("cuda-s-ada", "new adapter reply")));
    let fresh = reg.resolve_with(Some("cuda-s-ada"), false).expect("v2");
    assert!(
        !Arc::ptr_eq(&in_flight, &fresh),
        "an inplace reload must replace the engine object atomically under the registry lock"
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

fn hooked_linear(
    n: usize,
    k: usize,
    m: usize,
    slices: Vec<ServingDeltaSlice>,
) -> (Linear, Tensor, Vec<u32>) {
    let w = bf16_tensor(11, n, k, 0.5);
    let linear = Linear::new(w, None).unwrap();
    let x = bf16_tensor(12, m, k, 1.0);
    let base_bits = bits_of(&linear.forward(&x).unwrap());
    linear
        .attach_lora(ServingLoraHook::new(k, n, slices).unwrap())
        .unwrap();
    assert!(linear.has_lora());
    (linear, x, base_bits)
}

fn peft_slice(
    seed: u64,
    offset: usize,
    width: usize,
    k: usize,
    rank: usize,
    scaling: f64,
) -> ServingDeltaSlice {
    let a = bf16_tensor(seed, rank, k, 0.1);
    let b = bf16_tensor(seed ^ 0xB, width, rank, 0.3);
    ServingDeltaSlice::from_peft_pair("test.slice", offset, width, k, &a, &b, scaling).unwrap()
}

#[test]
fn serving_hook_delta_changes_the_forward_and_detach_restores_base_bitwise() {
    let (n, k, m, rank) = (48usize, 40usize, 5usize, 4usize);
    let s0 = peft_slice(21, 0, 16, k, rank, 1.0);
    let s1 = peft_slice(22, 16, 32, k, rank, 1.5);
    let (linear, x, base_bits) = hooked_linear(n, k, m, vec![s0, s1]);

    let hooked = linear.forward(&x).unwrap();
    let hooked_bits = bits_of(&hooked);
    assert_ne!(
        base_bits, hooked_bits,
        "an attached serving hook must change the forward output"
    );

    let manual_delta = |seed: u64, width: usize, scaling: f64| -> Tensor {
        let a = bf16_tensor(seed, rank, k, 0.1);
        let b = bf16_tensor(seed ^ 0xB, width, rank, 0.3);
        let x_f32 = x.to_dtype(DType::F32).unwrap();
        let a_t = a
            .t()
            .unwrap()
            .contiguous()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let b_t = b
            .t()
            .unwrap()
            .contiguous()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let mut d = x_f32.matmul(&a_t).unwrap().matmul(&b_t).unwrap();
        if scaling != 1.0 {
            d = d.affine(scaling, 0.0).unwrap();
        }
        d.to_dtype(DType::BF16).unwrap()
    };
    let d0 = manual_delta(21, 16, 1.0);
    let d1 = manual_delta(22, 32, 1.5);
    let base_t = {
        let w = bf16_tensor(11, n, k, 0.5);
        let clean = Linear::new(w, None).unwrap();
        clean.forward(&x).unwrap()
    };
    let expected = {
        let left = base_t.narrow(1, 0, 16).unwrap().add(&d0).unwrap();
        let right = base_t.narrow(1, 16, 32).unwrap().add(&d1).unwrap();
        Tensor::cat(&[left, right], 1).unwrap()
    };
    assert_eq!(
        bits_of(&expected),
        hooked_bits,
        "the hook's delta must be exactly base + (x A^T) B^T * scaling per slice, computed in \
         the same candle ops, so the two sides are bitwise equal on cpu"
    );

    linear.detach_lora();
    assert!(!linear.has_lora());
    assert_eq!(
        base_bits,
        bits_of(&linear.forward(&x).unwrap()),
        "detach must restore the base forward bitwise; anything else means unload leaks \
         adapter state into the base"
    );
}

#[test]
fn serving_hook_windowed_forward_rows_matches_the_full_forward_columns() {
    let (n, k, m, rank) = (48usize, 40usize, 5usize, 4usize);
    let s0 = peft_slice(31, 0, 16, k, rank, 1.0);
    let s1 = peft_slice(32, 16, 32, k, rank, 1.0);
    let (linear, x, _) = hooked_linear(n, k, m, vec![s0, s1]);
    let full = linear.forward(&x).unwrap();
    for (off, len) in [(0usize, 16usize), (16, 32), (8, 24), (0, 48)] {
        let win = linear.forward_rows(&x, off, len).unwrap();
        assert_eq!(
            bits_of(&full.narrow(1, off, len).unwrap()),
            bits_of(&win),
            "forward_rows({off},{len}) must serve the same lora-adjusted columns the full \
             forward serves; a window that misses the delta would give qkv slices base output \
             while the fused path serves adapter output"
        );
    }
}

#[test]
fn serving_hook_leaves_uncovered_columns_bitwise_base() {
    let (n, k, m, rank) = (48usize, 40usize, 5usize, 4usize);
    let s1 = peft_slice(41, 16, 32, k, rank, 1.0);
    let (linear, x, base_bits) = hooked_linear(n, k, m, vec![s1]);
    let hooked = linear.forward(&x).unwrap();
    let base_first = {
        let all = base_bits;
        let mut cols = Vec::new();
        for row in 0..m {
            cols.extend_from_slice(&all[row * n..row * n + 16]);
        }
        cols
    };
    let hooked_first = {
        let all = bits_of(&hooked);
        let mut cols = Vec::new();
        for row in 0..m {
            cols.extend_from_slice(&all[row * n..row * n + 16]);
        }
        cols
    };
    assert_eq!(
        base_first, hooked_first,
        "columns no slice covers must stay bitwise base output"
    );
}

#[test]
fn serving_hook_refuses_wrong_dims_loudly() {
    let k = 40usize;
    let a = bf16_tensor(51, 4, k + 2, 0.1);
    let b = bf16_tensor(52, 16, 4, 0.3);
    let err = ServingDeltaSlice::from_peft_pair("m.q_proj", 0, 16, k, &a, &b, 1.0)
        .err()
        .expect("wrong lora_A in dim must refuse");
    assert!(
        format!("{err:#}").contains("in_features"),
        "refusal must name the dim mismatch: {err:#}"
    );
    let a = bf16_tensor(53, 4, k, 0.1);
    let b = bf16_tensor(54, 20, 4, 0.3);
    let err = ServingDeltaSlice::from_peft_pair("m.q_proj", 0, 16, k, &a, &b, 1.0)
        .err()
        .expect("wrong lora_B out dim must refuse");
    assert!(
        format!("{err:#}").contains("slice width"),
        "refusal must name the slice width mismatch: {err:#}"
    );
    assert!(
        ServingLoraHook::new(k, 16, Vec::new()).is_err(),
        "a hook with zero slices must refuse instead of disabling has_lora fast paths for \
         nothing"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_kv_tag_is_a_bytes_hash_that_rotates_on_overwrite_and_is_deterministic() {
    use speaches_plus::oapi::chat_engine::cuda_adapter_tag_that_changes_when_the_weights_change;
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
    let t1 = cuda_adapter_tag_that_changes_when_the_weights_change(&dir);
    std::fs::write(&weights, b"bbbbbbbbbbbbbbbb").unwrap();
    pin_mtime(&weights);
    let t2 = cuda_adapter_tag_that_changes_when_the_weights_change(&dir);
    assert_ne!(
        t1, t2,
        "a same-length same-mtime overwrite must still rotate the tag, or a same-id reload \
         could resume stale KV onto new weights"
    );
    std::fs::write(&weights, b"aaaaaaaaaaaaaaaa").unwrap();
    let t3 = cuda_adapter_tag_that_changes_when_the_weights_change(&dir);
    assert_eq!(
        t1, t3,
        "identical bytes must produce the identical tag regardless of mtime"
    );
    #[cfg(feature = "wgpu")]
    {
        use speaches_plus::oapi::chat_engine_wgpu::persist::adapter_tag_that_changes_when_the_weights_change;
        assert_eq!(
            t3,
            adapter_tag_that_changes_when_the_weights_change(&dir),
            "the cuda and wgpu adapter tags must agree byte for byte so a base migrating \
             between backends never mixes tag namespaces"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(feature = "cuda")]
mod real_weight_gates {
    use super::*;
    use speaches_plus::oapi::chat_engine::{CudaLoraAdapterEngine, NvEngineChat};

    static ONE_GATE_AT_A_TIME_THE_VRAM_PLAN_SUM_IS_PROCESS_WIDE_SO_PARALLEL_GATES_SEE_EACH_OTHERS_ENGINES:
        std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn serialize_gates() -> std::sync::MutexGuard<'static, ()> {
        ONE_GATE_AT_A_TIME_THE_VRAM_PLAN_SUM_IS_PROCESS_WIDE_SO_PARALLEL_GATES_SEE_EACH_OTHERS_ENGINES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn require_gate() {
        if std::env::var(REAL_WEIGHTS_TEST_ENV).ok().as_deref() != Some("1") {
            panic!(
                "{REAL_WEIGHTS_TEST_ENV}=1 not set. Returning green here would report a \
                 cuda lora serving proof that never ran; this is a SKIP surfaced as a \
                 failure by design."
            );
        }
        if std::env::var_os("NV_KV_MAX_SEQ_LEN_GEMMA4").is_none() {
            std::env::set_var("NV_KV_MAX_SEQ_LEN_GEMMA4", "8192");
            eprintln!(
                "NV_KV_MAX_SEQ_LEN_GEMMA4=8192 pinned by the gate: an unpinned first engine \
                 SOLVES its kv-reserve to fill the remaining vram budget, leaving no budget \
                 for the adapter engine this gate must boot beside it"
            );
        }
    }

    fn required_base_dir() -> PathBuf {
        let raw = std::env::var_os("NV_CHAT_MODEL_DIR").unwrap_or_else(|| {
            panic!("NV_CHAT_MODEL_DIR must point at a gemma4 dense cuda checkpoint")
        });
        let p = PathBuf::from(raw);
        assert!(p.is_dir(), "NV_CHAT_MODEL_DIR={} is not a directory", p.display());
        p
    }

    fn gemma4_dims(base_dir: &Path) -> (usize, usize) {
        let raw = std::fs::read_to_string(base_dir.join("config.json")).expect("base config");
        let cfg = nv_models::gemma4::Gemma4Config::from_hf_json_str(&raw)
            .expect("gemma4 config parses");
        (cfg.hidden_size, cfg.intermediate_size)
    }

    fn write_down_proj_adapter(dir: &Path, base_dir: &Path, seed: u64) {
        std::fs::create_dir_all(dir).unwrap();
        let (hidden, inter) = gemma4_dims(base_dir);
        let rank = 8usize;
        let cfg = serde_json::json!({
            "r": rank,
            "lora_alpha": (2 * rank) as f64,
            "target_modules": ["down_proj"],
            "base_model_name_or_path": null,
        });
        std::fs::write(dir.join(ADAPTER_CONFIG_FILE), cfg.to_string()).unwrap();
        let module = "model.language_model.layers.0.mlp.down_proj";
        let a = Tensor::from_vec(
            lcg_vals(rank * inter, seed.wrapping_add(1), 0.05),
            (rank, inter),
            &Device::Cpu,
        )
        .unwrap();
        let b = Tensor::from_vec(
            lcg_vals(hidden * rank, seed.wrapping_add(2), 0.5),
            (hidden, rank),
            &Device::Cpu,
        )
        .unwrap();
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(format!("{module}.lora_A.weight"), a);
        tensors.insert(format!("{module}.lora_B.weight"), b);
        candle_core::safetensors::save(&tensors, dir.join(ADAPTER_WEIGHTS_FILE)).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "boots the real gemma4 dense cuda base named by NV_CHAT_MODEL_DIR twice (base + \
                adapter engine), then requires: adapter greedy output differs from base, \
                registry unload restores base greedy output bitwise, and the unloaded id \
                resolves to nothing; set NV_LORA_CUDA_SERVING_TEST=1. Run: \
                NVK_LANE=cusites NVK_PKG=speaches-plus NV_LORA_CUDA_SERVING_TEST=1 NV_CHAT_MODEL_DIR=<gemma4-dir> \
                rust/scripts/nvk.sh test --test lora_cuda_serving -- --ignored --test-threads=1 --nocapture"]
    async fn adapter_serves_differently_and_unload_restores_base_bitwise() {
        let _one_gate = serialize_gates();
        require_gate();
        let base_dir = required_base_dir();
        let base: Arc<dyn ChatEngine> =
            Arc::new(NvEngineChat::try_load(&base_dir).expect("base engine loads"));
        let reg = ChatRegistry::single(base.clone());
        let prompt = "The capital of France is";
        warm_first_request_jit_so_captures_compare_steady_state(&base, prompt).await;
        let text_base0 = generated_text(&base, prompt, 8).await;

        let dir = scratch("e2e");
        write_down_proj_adapter(&dir, &base_dir, 1001);
        let ada: Arc<dyn ChatEngine> = Arc::new(
            CudaLoraAdapterEngine::try_load(&base_dir, "lora-cuda-e2e", &dir)
                .expect("adapter engine loads"),
        );
        reg.insert(ada.clone());
        let text_ada = generated_text(&ada, prompt, 8).await;
        assert_ne!(
            text_base0, text_ada,
            "a rank-8 down_proj delta at b_scale 0.5 must change greedy output; equality \
             means the hooks never fired in the cuda decode path"
        );

        reg.remove("lora-cuda-e2e").expect("unload adapter");
        drop(ada);
        let text_base1 = generated_text(&base, prompt, 8).await;
        assert_eq!(
            text_base0, text_base1,
            "after adapter load and unload the base engine's greedy output must be bitwise \
             the never-loaded output; the base engine object was never mutated"
        );
        assert!(reg.resolve_with(Some("lora-cuda-e2e"), false).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[ignore = "builds the cuda adapter engine, drops it, rebuilds from the same dir with \
                rewritten same-length weights: the rebuild must be ADMITTED by the vram plan \
                (the dropped engine's RecordedBootPlan must have left the enforced sum) and \
                the carried KV tag must rotate; set NV_LORA_CUDA_SERVING_TEST=1. Run: \
                NVK_LANE=cusites NVK_PKG=speaches-plus NV_LORA_CUDA_SERVING_TEST=1 NV_CHAT_MODEL_DIR=<gemma4-dir> \
                rust/scripts/nvk.sh test --test lora_cuda_serving -- --ignored --test-threads=1 --nocapture"]
    fn same_id_reload_with_changed_bytes_rotates_the_kv_tag_and_replaces_the_engine() {
        let _one_gate = serialize_gates();
        require_gate();
        let base_dir = required_base_dir();
        let dir = scratch("reload");
        write_down_proj_adapter(&dir, &base_dir, 1001);
        let e1 = CudaLoraAdapterEngine::try_load(&base_dir, "lora-cuda-reload", &dir)
            .expect("v1 engine");
        let tag1 = e1
            .adapter_kv_tag_that_any_future_cuda_kv_persist_must_key_on()
            .to_string();
        let reg = ChatRegistry::single(Arc::new(EchoEngine::new("reload-base", "x")));
        reg.insert(Arc::new(e1));
        let old = reg.resolve_with(Some("lora-cuda-reload"), false).expect("v1");
        reg.remove("lora-cuda-reload").expect("unload v1");
        drop(old);
        write_down_proj_adapter(&dir, &base_dir, 2002);
        let e2 = CudaLoraAdapterEngine::try_load(&base_dir, "lora-cuda-reload", &dir).expect(
            "v2 engine must be admitted after v1 dropped: a vram-plan refusal here means the \
             dropped engine's RecordedBootPlan did not decrement the process-wide enforced sum",
        );
        let tag2 = e2
            .adapter_kv_tag_that_any_future_cuda_kv_persist_must_key_on()
            .to_string();
        assert_ne!(
            tag1, tag2,
            "a same-id reload with changed adapter bytes must rotate the KV tag; a stable \
             tag would let any future cuda KV persistence resume stale KV onto new weights"
        );
        reg.insert(Arc::new(e2));
        let new = reg.resolve_with(Some("lora-cuda-reload"), false).expect("v2");
        assert_eq!(new.model_id(), "lora-cuda-reload");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
