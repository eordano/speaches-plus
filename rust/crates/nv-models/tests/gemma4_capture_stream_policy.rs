#![cfg(feature = "cuda")]

mod common;
use common::config_json;
use common::HEAD_DIM_128 as HEAD_DIM;
use common::HIDDEN_128 as HIDDEN;
use common::INTER_256 as INTER;
use common::LcgTop24TwoSided as Lcg;
use common::N_KV_1 as N_KV;
use common::N_LAYERS_2 as N_LAYERS;
use common::N_Q_2 as N_Q;
use common::ones_tensor;
use common::rand_tensor;
use common::VOCAB_512 as VOCAB;
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::{Gemma4, Gemma4Config};
use nv_models::gemma4_graph::GraphedGemma4Decoder;
use nv_weights::WeightLoader;
use std::collections::HashMap;

const KV_MAX: usize = 64;

const AB_ENV: &str = "NV_GEMMA4_CAPTURE_STREAM_AB";
const AB_DEVICE_ENV: &str = "NV_GEMMA4_CAPTURE_STREAM_AB_DEVICE";

fn write_tiny_model(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();

    let mut rng = Lcg(0x5eed_cafe_f00d_0060);
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert(
        "model.language_model.embed_tokens.weight".into(),
        rand_tensor(&mut rng, (VOCAB, HIDDEN), 1.0),
    );
    tensors.insert(
        "model.language_model.norm.weight".into(),
        ones_tensor(HIDDEN),
    );
    tensors.insert(
        "lm_head.weight".into(),
        rand_tensor(&mut rng, (VOCAB, HIDDEN), 1.0),
    );
    for i in 0..N_LAYERS {
        let p = format!("model.language_model.layers.{i}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            tensors.insert(format!("{p}.{norm}.weight"), ones_tensor(HIDDEN));
        }
        tensors.insert(format!("{p}.layer_scalar"), ones_tensor(1));
        tensors.insert(
            format!("{p}.self_attn.q_proj.weight"),
            rand_tensor(&mut rng, (N_Q * HEAD_DIM, HIDDEN), 0.3),
        );
        tensors.insert(
            format!("{p}.self_attn.k_proj.weight"),
            rand_tensor(&mut rng, (N_KV * HEAD_DIM, HIDDEN), 0.3),
        );
        tensors.insert(
            format!("{p}.self_attn.v_proj.weight"),
            rand_tensor(&mut rng, (N_KV * HEAD_DIM, HIDDEN), 0.3),
        );
        tensors.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rand_tensor(&mut rng, (HIDDEN, N_Q * HEAD_DIM), 0.3),
        );
        tensors.insert(
            format!("{p}.self_attn.q_norm.weight"),
            ones_tensor(HEAD_DIM),
        );
        tensors.insert(
            format!("{p}.self_attn.k_norm.weight"),
            ones_tensor(HEAD_DIM),
        );
        tensors.insert(
            format!("{p}.mlp.gate_proj.weight"),
            rand_tensor(&mut rng, (INTER, HIDDEN), 0.3),
        );
        tensors.insert(
            format!("{p}.mlp.up_proj.weight"),
            rand_tensor(&mut rng, (INTER, HIDDEN), 0.3),
        );
        tensors.insert(
            format!("{p}.mlp.down_proj.weight"),
            rand_tensor(&mut rng, (HIDDEN, INTER), 0.3),
        );
    }
    candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
}

fn stream_device() -> Device {
    let device = Device::new_cuda_with_stream(0).expect(
        "no CUDA device 0: this suite is the only gate on the gemma4 capture-stream policy and \
         must not report success having executed nothing",
    );
    disable_event_tracking(&device);
    device
}

fn disable_event_tracking(device: &Device) {
    let Device::Cuda(d) = device else {
        panic!("expected a cuda device");
    };
    let ctx = d.cuda_stream().context().clone();
    if ctx.is_event_tracking() {
        let _ = ctx.default_stream().synchronize();
        unsafe { ctx.disable_event_tracking() };
    }
}

fn load_tiny(device: &Device) -> Gemma4 {
    static CAP_FIXTURE: std::sync::OnceLock<std::path::PathBuf> =
        std::sync::OnceLock::new();
    let dir = CAP_FIXTURE
        .get_or_init(|| {
            let d = std::env::temp_dir()
                .join(format!("{}-{}", "nv-gemma4-capture-stream-tiny", std::process::id()));
            write_tiny_model(&d);
            d
        })
        .clone();
    let cfg =
        Gemma4Config::from_hf_json_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
            .unwrap();
    let weights = WeightLoader::open_dir(&dir, device).unwrap();
    Gemma4::from_loader(cfg, &weights, device).unwrap()
}

fn ungraphed_rows(model: &Gemma4, device: &Device, tokens: &[u32]) -> Vec<Vec<f32>> {
    let Device::Cuda(d) = device else {
        panic!("expected a cuda device");
    };
    let stream = d.cuda_stream();
    let vocab = model.config().vocab_size;
    let mut cache = model.new_kv_cache_fp8(KV_MAX).expect("kv cache");
    let mut out = stream.alloc_zeros::<f32>(vocab).expect("logits buf");
    let mut rows = Vec::with_capacity(tokens.len());
    for (pos, &t) in tokens.iter().enumerate() {
        cache
            .set_pending_pos_host_only(pos, pos + 1)
            .expect("pending pos");
        let tok = Tensor::from_vec(vec![t], (1usize, 1usize), device).expect("tok tensor");
        let p = Tensor::from_vec(vec![pos as i32], 1usize, device).expect("pos tensor");
        model
            .forward_with_cache_into(&tok, &p, &mut cache, &mut out)
            .expect("eager forward");
        stream.synchronize().expect("sync");
        rows.push(stream.clone_dtoh(&out).expect("dtoh"));
    }
    rows
}

fn graphed_rows(model: &Gemma4, device: &Device, tokens: &[u32]) -> (Vec<Vec<f32>>, u64, bool) {
    let cache = model.new_kv_cache_fp8(KV_MAX).expect("kv cache");
    let mut dec = GraphedGemma4Decoder::new(model, cache, device).expect("graphed decoder");
    let rows: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| dec.forward_decode_logits_vec(t).expect("graphed decode"))
        .collect();
    (rows, dec.call_count(), dec.capture_active())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn legacy_device() -> Device {
    let device = Device::new_cuda(0).expect(
        "no CUDA device 0: this suite is the only gate on the gemma4 capture-stream coincidence \
         and must not report success having executed nothing",
    );
    disable_event_tracking(&device);
    device
}

#[test]
fn the_capture_stream_is_the_device_stream_when_there_is_one_and_a_fork_otherwise() {
    let legacy = legacy_device();
    let Device::Cuda(l) = &legacy else {
        unreachable!()
    };
    assert!(
        l.cuda_stream().cu_stream().is_null(),
        "Device::new_cuda must leave candle on the legacy NULL stream, or the rest of this suite \
         is testing the wrong configuration: that is the device chat_engine/build.rs hands \
         GraphedGemma4Decoder for ModelFamily::Gemma4"
    );
    let stream = stream_device();
    let Device::Cuda(s) = &stream else {
        unreachable!()
    };
    assert!(
        !s.cuda_stream().cu_stream().is_null(),
        "Device::new_cuda_with_stream must hand candle a real non-NULL stream"
    );
}

#[test]
fn the_forked_capture_reproduces_the_ungraphed_decode_bit_for_bit() {
    let device = legacy_device();
    let model = load_tiny(&device);
    let tokens: [u32; 12] = [3, 17, 5, 41, 9, 23, 7, 88, 2, 61, 13, 44];

    let want = ungraphed_rows(&model, &device, &tokens);
    let (got, calls, captured) = graphed_rows(&model, &device, &tokens);

    assert_eq!(calls, tokens.len() as u64);
    assert!(
        captured,
        "the decoder reported no active capture, so the graph never engaged and this comparison \
         proves nothing about capture"
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g.len(), VOCAB);
        assert!(
            g.iter().all(|v| v.is_finite()),
            "step {i}: graphed logits are not finite"
        );
        let d = max_abs_diff(g, w);
        assert_eq!(
            d, 0.0,
            "step {i}: the graph captured on a FORKED stream and the ungraphed decode disagree by \
             {d}. This is THE gate on task #60's coincidence: GraphedGemma4Decoder captures \
             Gemma4::forward_with_cache_into on a stream candle does not launch on, and that is \
             safe only while every launch in the body is an nv-* kernel honouring \
             nv_layers::cuda_stream::with_stream. A single candle-native launch entering the \
             decode path -- flash_attention at gemma4.rs, a candle matmul in a refactor, a \
             non-fp8 cache -- runs eagerly outside the capture, gets baked in by address and \
             freed with the closure's temporaries. That is the qwen #162 failure, and this \
             assertion is what makes it loud instead of silent"
        );
    }
}

#[test]
fn the_forked_capture_still_agrees_after_the_first_call_is_a_replay() {
    let device = legacy_device();
    let model = load_tiny(&device);
    let tokens: [u32; 24] = [
        3, 17, 5, 41, 9, 23, 7, 88, 2, 61, 13, 44, 31, 6, 90, 12, 400, 55, 8, 19, 77, 4, 120, 66,
    ];

    let want = ungraphed_rows(&model, &device, &tokens);
    let (got, _, captured) = graphed_rows(&model, &device, &tokens);
    assert!(captured);

    let mut worst = 0.0f32;
    let mut worst_step = 0usize;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = max_abs_diff(g, w);
        if d > worst {
            worst = d;
            worst_step = i;
        }
    }
    assert_eq!(
        worst, 0.0,
        "replay {worst_step} of {} drifted by {worst} from the ungraphed decode. Call 0 captures \
         and the rest replay; a drift that appears only after call 0 is the signature of a \
         captured kernel pointing at a buffer that only existed during capture",
        got.len()
    );
}

#[test]
fn the_device_stream_capture_reproduces_the_ungraphed_decode_too() {
    let device = stream_device();
    let model = load_tiny(&device);
    let tokens: [u32; 16] = [3, 17, 5, 41, 9, 23, 7, 88, 2, 61, 13, 44, 31, 6, 90, 12];

    let want = ungraphed_rows(&model, &device, &tokens);
    let (got, _, captured) = graphed_rows(&model, &device, &tokens);
    assert!(
        captured,
        "CaptureStream::for_device must capture on the device stream when there is one; no \
         capture means the engine fell back and this proves nothing"
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = max_abs_diff(g, w);
        assert_eq!(
            d, 0.0,
            "step {i}: capturing on the device stream disagrees with the eager path by {d}. The \
             day chat_engine/build.rs is cleared to flip ModelFamily::Gemma4 onto \
             Device::new_cuda_with_stream, this is the arm that has to already be green"
        );
    }
}

fn ab_gated() -> bool {
    std::env::var(AB_ENV).ok().as_deref() == Some("1")
}

fn ab_device() -> Device {
    match std::env::var(AB_DEVICE_ENV).ok().as_deref() {
        Some("stream") => stream_device(),
        _ => legacy_device(),
    }
}

fn ungraphed_chain(
    model: &Gemma4,
    device: &Device,
    first: u32,
    steps: usize,
    kv_max: usize,
) -> (Vec<Vec<f32>>, Vec<u32>) {
    let Device::Cuda(d) = device else {
        panic!("expected a cuda device");
    };
    let stream = d.cuda_stream();
    let vocab = model.config().vocab_size;
    let mut cache = model.new_kv_cache_fp8(kv_max).expect("kv cache");
    let mut out = stream.alloc_zeros::<f32>(vocab).expect("logits buf");
    let mut rows = Vec::with_capacity(steps);
    let mut ids = Vec::with_capacity(steps);
    let mut tok = first;
    for pos in 0..steps {
        cache
            .set_pending_pos_host_only(pos, pos + 1)
            .expect("pending pos");
        let t = Tensor::from_vec(vec![tok], (1usize, 1usize), device).expect("tok tensor");
        let p = Tensor::from_vec(vec![pos as i32], 1usize, device).expect("pos tensor");
        model
            .forward_with_cache_into(&t, &p, &mut cache, &mut out)
            .expect("eager forward");
        stream.synchronize().expect("sync");
        let row = stream.clone_dtoh(&out).expect("dtoh");
        let (top, _) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("argmax");
        tok = top as u32;
        rows.push(row);
        ids.push(tok);
    }
    (rows, ids)
}

#[test]
#[ignore]
fn real_checkpoint_decode_rate() {
    if !ab_gated() {
        panic!("set {AB_ENV}=1 to run the real-checkpoint decode-rate measurement");
    }
    let dir = std::env::var("NV_CHAT_MODEL_DIR")
        .expect("NV_CHAT_MODEL_DIR must point at a gemma4 checkpoint");
    let dir = std::path::PathBuf::from(dir);
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");

    let device = ab_device();
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse gemma4 config");
    let qcfg = nv_weights::QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse quant");
    let weights = WeightLoader::open_dir(&dir, &device).expect("open weights");
    let t_load = std::time::Instant::now();
    let model =
        Gemma4::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("load gemma4 weights");
    let load_s = t_load.elapsed().as_secs_f64();

    let kv_max: usize = std::env::var("NV_GEMMA4_CAPTURE_STREAM_AB_KV")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let steps: usize = std::env::var("NV_GEMMA4_CAPTURE_STREAM_AB_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let oracle_steps: usize = std::env::var("NV_GEMMA4_CAPTURE_STREAM_AB_ORACLE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    assert!(oracle_steps >= 2 && oracle_steps < steps);

    let first = 2u32;
    let (oracle_rows, oracle_ids) = ungraphed_chain(&model, &device, first, oracle_steps, kv_max);

    let cache = model.new_kv_cache_fp8(kv_max).expect("kv cache");
    let mut dec = GraphedGemma4Decoder::new(&model, cache, &device).expect("graphed decoder");

    let mut tok = first;
    let mut ids = Vec::with_capacity(steps);
    let mut warm_ms = 0.0f64;
    let mut per_step_ms: Vec<f64> = Vec::with_capacity(steps);
    let mut worst_row_diff = 0.0f32;
    let mut worst_row = 0usize;
    for i in 0..steps {
        let t = std::time::Instant::now();
        let row = dec.forward_decode_logits_vec(tok).expect("decode");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i == 0 {
            warm_ms = ms;
        } else {
            per_step_ms.push(ms);
        }
        if i < oracle_steps {
            let d = max_abs_diff(&row, &oracle_rows[i]);
            if d > worst_row_diff {
                worst_row_diff = d;
                worst_row = i;
            }
        }
        let (top, _) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("argmax");
        ids.push(top as u32);
        tok = if i + 1 < oracle_steps {
            oracle_ids[i]
        } else {
            top as u32
        };
    }
    assert!(
        dec.capture_active(),
        "capture never engaged, so this number is an eager decode rate wearing a graph's name"
    );
    assert_eq!(
        &ids[..oracle_steps],
        &oracle_ids[..],
        "the graphed decode picked different tokens than the ungraphed decode fed the same inputs"
    );
    assert_eq!(
        worst_row_diff, 0.0,
        "graphed row {worst_row} differs from the ungraphed row by {worst_row_diff}: the captured \
         body is not the body the eager path runs"
    );
    per_step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = per_step_ms[per_step_ms.len() / 2];
    let distinct = ids.iter().collect::<std::collections::HashSet<_>>().len();
    assert!(
        distinct > 1,
        "every decoded id was {}: a degenerate decode has no meaningful rate",
        ids[0]
    );
    eprintln!(
        "[t60-ab] device={} load_s={load_s:.1} kv_max={kv_max} steps={steps} \
         first_call_ms={warm_ms:.3} median_ms={median:.4} tok_s={:.2} oracle_steps={oracle_steps} \
         oracle_max_abs_diff={worst_row_diff} first_ids={:?}",
        std::env::var(AB_DEVICE_ENV).unwrap_or_else(|_| "legacy".into()),
        1000.0 / median,
        &ids[..ids.len().min(12)]
    );
}
