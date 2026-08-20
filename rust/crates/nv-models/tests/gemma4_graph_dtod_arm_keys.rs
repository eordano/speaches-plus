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
use nv_models::gemma4_graph::{
    GraphedGemma4Decoder, BOTH_DTOD_ARMS_ARE_SEQ_LEN_1_SO_THE_GRAPH_KEY_MUST_CARRY_THE_ARM,
    DECODE_GRAPH_KEY_LOGITS_DTOD, DECODE_GRAPH_KEY_NO_LOGITS_DTOD,
};
use nv_weights::WeightLoader;
use std::collections::HashMap;

const KV_MAX: usize = 64;

const TINY_MODEL_DIRNAME: &str = "nv-gemma4-dtod-arm-keys-tiny";

fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    static SERIALIZE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    SERIALIZE.lock().unwrap_or_else(|e| e.into_inner())
}

fn write_tiny_model(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();

    let mut rng = Lcg(0x5eed_cafe_f00d_0068);
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

fn forked_capture_device() -> Device {
    let device = Device::new_cuda(0).expect(
        "no CUDA device 0: this suite is the only gate on the two dtod_logits arms owning \
         distinct graph keys and must not report success having executed nothing",
    );
    disable_event_tracking(&device);
    device
}

fn device_stream_capture_device() -> Device {
    let device = Device::new_cuda_with_stream(0).expect(
        "no CUDA device 0: the device-stream configuration is where the warm pass runs, and it is \
         the one that must not skip it for the second arm",
    );
    disable_event_tracking(&device);
    device
}

fn load_tiny(device: &Device) -> Gemma4 {
    static ARM_FIXTURE: std::sync::OnceLock<std::path::PathBuf> =
        std::sync::OnceLock::new();
    let dir = ARM_FIXTURE
        .get_or_init(|| {
            let d = std::env::temp_dir()
                .join(format!("{}-{}", TINY_MODEL_DIRNAME, std::process::id()));
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

fn eager_rows(model: &Gemma4, device: &Device, tokens: &[u32]) -> Vec<Vec<f32>> {
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

fn new_decoder<'m>(model: &'m Gemma4, device: &Device) -> GraphedGemma4Decoder<'m> {
    let cache = model.new_kv_cache_fp8(KV_MAX).expect("kv cache");
    GraphedGemma4Decoder::new(model, cache, device).expect("graphed decoder")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

const TOKENS: [u32; 6] = [3, 17, 5, 41, 9, 23];

const TOKENS_EXTENDED: [u32; 12] = [3, 17, 5, 41, 9, 23, 7, 29, 11, 3, 51, 2];

const MIXING_THE_ARMS_ON_ONE_DECODER_IS_MEASURED_BROKEN_ON_THE_DEVICE_STREAM_CAPTURE: &str =
    "on Device::new_cuda_with_stream, a decoder that has run BOTH arms diverges from the eager \
     decode from the first replay of the logits graph onward -- max_abs_diff 0.265625, 2.7148438, \
     3.375, 2.6484375 over steps 2..5 of this suite's tiny model, identical in both capture \
     orders. It is not the shared key: it survives giving each arm its own key, it survives \
     destroying the other arm's graph on the switch, two decoders holding one graph each replay \
     bit-exactly on the same device, and a single-arm decoder replays bit-exactly. Gemma4 serves \
     on Device::new_cuda (chat_engine/build.rs keeps the device-stream flip blocked), where the \
     capture is forked and both orders are bit-exact, so this is a defect of the unused \
     configuration and is tracked separately from the graph-key collision";

fn assert_row_is_the_eager_row(got: &[f32], want: &[f32], step: usize, order: &str) {
    assert_eq!(
        got.len(),
        VOCAB,
        "step {step} ({order}): row is not one vocab wide"
    );
    assert!(
        got.iter().all(|v| v.is_finite()),
        "step {step} ({order}): graphed logits are not finite"
    );
    let d = max_abs_diff(got, want);
    assert_eq!(
        d,
        0.0,
        "step {step} ({order}): the replayed logits differ from the eager decode by {d}. A row \
         that is stale -- zero, or equal to an earlier step -- is the signature of the defect \
         this suite gates: {}. A row that merely drifts is the other finding this suite carries: \
         {}",
        BOTH_DTOD_ARMS_ARE_SEQ_LEN_1_SO_THE_GRAPH_KEY_MUST_CARRY_THE_ARM,
        MIXING_THE_ARMS_ON_ONE_DECODER_IS_MEASURED_BROKEN_ON_THE_DEVICE_STREAM_CAPTURE
    );
}

fn single_arm_control_rows<'m>(
    model: &'m Gemma4,
    device: &Device,
    tokens: &[u32],
) -> (Vec<Vec<f32>>, GraphedGemma4Decoder<'m>) {
    let mut dec = new_decoder(model, device);
    let rows = tokens
        .iter()
        .map(|&t| dec.forward_decode_logits_vec(t).expect("control decode"))
        .collect();
    (rows, dec)
}

fn probe_double_advance_kv_timeline_bit_matches_the_replays(
    model: &Gemma4,
    device: &Device,
    rows: &[(usize, Vec<f32>)],
) {
    let Device::Cuda(d) = device else {
        panic!("expected a cuda device");
    };
    let stream = d.cuda_stream();
    let vocab = model.config().vocab_size;
    let mut cache = model.new_kv_cache_fp8(KV_MAX).expect("kv cache");
    let mut out = stream.alloc_zeros::<f32>(vocab).expect("logits buf");
    let mut fwd = |cache: &mut nv_models::gemma4::Gemma4KvCacheFp8,
                   t: u32,
                   pos: i32,
                   out: &mut cudarc::driver::CudaSlice<f32>|
     -> Vec<f32> {
        let tok = Tensor::from_vec(vec![t], (1usize, 1usize), device).expect("tok tensor");
        let p = Tensor::from_vec(vec![pos], 1usize, device).expect("pos tensor");
        model
            .forward_with_cache_into(&tok, &p, cache, out)
            .expect("sim forward");
        stream.synchronize().expect("sync");
        stream.clone_dtoh(out).expect("dtoh")
    };
    let _ = fwd(&mut cache, TOKENS[0], 0, &mut out);
    let _ = fwd(&mut cache, TOKENS[0], 0, &mut out);
    let _ = fwd(&mut cache, TOKENS[1], 1, &mut out);
    let _ = fwd(&mut cache, TOKENS[1], 1, &mut out);
    let mut sim: Vec<(usize, Vec<f32>)> = Vec::new();
    for step in 2..TOKENS.len() {
        cache.reset();
        cache.advance(step);
        sim.push((step, fwd(&mut cache, TOKENS[step], step as i32, &mut out)));
    }
    let line: Vec<String> = rows
        .iter()
        .filter(|(s, _)| *s >= 2)
        .map(|(s, r)| {
            let simrow = &sim.iter().find(|(ss, _)| ss == s).expect("sim step").1;
            format!("{s}:{}", max_abs_diff(r, simrow))
        })
        .collect();
    eprintln!(
        "[PROBE double-advance sim] this simulates the broken-warm-pass KV timeline (warm and \
         capture each advance current_len, so the second captured arm writes kv(pos1) at slots \
         2 and 3 and slot 1 keeps a stale duplicate of kv(pos0) forever); replay-vs-sim \
         max_abs_diff 0 everywhere means the double-advance defect is BACK, nonzero means the \
         decoder is healthy: {}",
        line.join(" ")
    );
}

fn probe_shape_of_the_first_divergent_row(got: &[f32], want: &[f32], step: usize) {
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    if bad.is_empty() {
        eprintln!("[PROBE row shape] step {step}: no divergent entries");
        return;
    }
    let head: Vec<String> = bad
        .iter()
        .take(8)
        .map(|&i| format!("{i}:{}->{}", want[i], got[i]))
        .collect();
    eprintln!(
        "[PROBE row shape] step {step}: {} of {} entries differ, first={} last={} sample {}",
        bad.len(),
        got.len(),
        bad[0],
        bad[bad.len() - 1],
        head.join(" ")
    );
}

fn probe_scope_of_the_poison_before_the_asserts_run(
    model: &Gemma4,
    device: &Device,
    ctrl: &mut GraphedGemma4Decoder<'_>,
    want12: &[Vec<f32>],
) {
    let mut fresh = new_decoder(model, device);
    let fresh_rows: Vec<(usize, Vec<f32>)> = TOKENS_EXTENDED
        .iter()
        .enumerate()
        .map(|(step, &t)| {
            (
                step,
                fresh
                    .forward_decode_logits_vec(t)
                    .expect("fresh probe decode"),
            )
        })
        .collect();
    report_diffs("PROBE fresh decoder built after both arms", &fresh_rows, want12);
    let ctrl_more: Vec<(usize, Vec<f32>)> = TOKENS_EXTENDED[TOKENS.len()..]
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            (
                i + TOKENS.len(),
                ctrl.forward_decode_logits_vec(t)
                    .expect("control continuation decode"),
            )
        })
        .collect();
    report_diffs(
        "PROBE single-arm control replays continuing after both arms",
        &ctrl_more,
        want12,
    );
}

fn report_diffs(label: &str, got: &[(usize, Vec<f32>)], want: &[Vec<f32>]) {
    let line: Vec<String> = got
        .iter()
        .map(|(step, row)| format!("{step}:{}", max_abs_diff(row, &want[*step])))
        .collect();
    eprintln!(
        "[{label}] per-step max_abs_diff vs eager: {}",
        line.join(" ")
    );
}

fn no_logits_first_then_logits(device: &Device) {
    let model = load_tiny(device);
    let want12 = eager_rows(&model, device, &TOKENS_EXTENDED);
    let want = &want12[..TOKENS.len()];
    let (control, mut ctrl_dec) = single_arm_control_rows(&model, device, &TOKENS);
    for (step, row) in control.iter().enumerate() {
        assert_row_is_the_eager_row(row, &want[step], step, "single-arm control");
    }
    let mut dec = new_decoder(&model, device);

    dec.forward_decode(TOKENS[0]).expect("no-logits decode");
    let nodes_no_logits = dec.captured_node_count();

    let rows: Vec<(usize, Vec<f32>)> = TOKENS
        .iter()
        .enumerate()
        .skip(1)
        .map(|(step, &t)| {
            (
                step,
                dec.forward_decode_logits_vec(t).expect("logits decode"),
            )
        })
        .collect();
    report_diffs("no-logits captured first", &rows, want);
    for (step, row) in rows.iter() {
        probe_shape_of_the_first_divergent_row(row, &want[*step], *step);
    }
    probe_double_advance_kv_timeline_bit_matches_the_replays(&model, device, &rows);
    probe_scope_of_the_poison_before_the_asserts_run(&model, device, &mut ctrl_dec, &want12);
    for (step, row) in rows.iter() {
        assert_row_is_the_eager_row(row, &want[*step], *step, "no-logits captured first");
    }

    assert!(
        nodes_no_logits > 0,
        "the no-logits capture produced an empty graph, so nothing was captured at all"
    );
    assert!(
        dec.has_captured_arm(true) && dec.has_captured_arm(false),
        "after both arms ran, keys {DECODE_GRAPH_KEY_NO_LOGITS_DTOD} and \
         {DECODE_GRAPH_KEY_LOGITS_DTOD} must both hold a graph, one per arm"
    );
    let nodes_both = dec.captured_node_count();
    assert!(
        nodes_both > nodes_no_logits,
        "the cached node total did not grow when the logits arm ran ({nodes_no_logits} -> \
         {nodes_both}): the logits arm reused the no-logits graph instead of capturing its own"
    );
    eprintln!(
        "no_logits_first_then_logits: nodes no-logits={nodes_no_logits} both={nodes_both} \
         (logits arm = {}), {} replayed rows equal the eager decode exactly",
        nodes_both - nodes_no_logits,
        TOKENS.len() - 1
    );
}

fn logits_first_then_no_logits(device: &Device) {
    let model = load_tiny(device);
    let want12 = eager_rows(&model, device, &TOKENS_EXTENDED);
    let want = &want12[..TOKENS.len()];
    let (control, mut ctrl_dec) = single_arm_control_rows(&model, device, &TOKENS);
    for (step, row) in control.iter().enumerate() {
        assert_row_is_the_eager_row(row, &want[step], step, "single-arm control");
    }
    let mut dec = new_decoder(&model, device);

    let row0 = dec
        .forward_decode_logits_vec(TOKENS[0])
        .expect("logits decode");
    assert_row_is_the_eager_row(&row0, &want[0], 0, "logits captured first");
    let nodes_logits = dec.captured_node_count();

    dec.forward_decode(TOKENS[1]).expect("no-logits decode");
    let after = dec.logits_buf_snapshot().expect("logits buf snapshot");
    let d = max_abs_diff(&after, &row0);
    assert_eq!(
        d, 0.0,
        "the no-logits arm moved logits_buf by {d}: forward_decode must run \
         forward_with_cache, which never writes logits_buf, so a change here means it replayed \
         the logits arm's graph. {}",
        BOTH_DTOD_ARMS_ARE_SEQ_LEN_1_SO_THE_GRAPH_KEY_MUST_CARRY_THE_ARM
    );
    let nodes_both = dec.captured_node_count();
    assert!(
        dec.has_captured_arm(true) && dec.has_captured_arm(false),
        "after both arms ran, keys {DECODE_GRAPH_KEY_LOGITS_DTOD} and \
         {DECODE_GRAPH_KEY_NO_LOGITS_DTOD} must both hold a graph, one per arm"
    );
    assert!(
        nodes_both > nodes_logits,
        "the cached node total did not grow when the no-logits arm ran ({nodes_logits} -> \
         {nodes_both}): the no-logits arm reused the logits graph"
    );

    let rows: Vec<(usize, Vec<f32>)> = TOKENS
        .iter()
        .enumerate()
        .skip(2)
        .map(|(step, &t)| {
            (
                step,
                dec.forward_decode_logits_vec(t).expect("logits decode"),
            )
        })
        .collect();
    report_diffs("logits captured first", &rows, want);
    for (step, row) in rows.iter() {
        probe_shape_of_the_first_divergent_row(row, &want[*step], *step);
    }
    probe_double_advance_kv_timeline_bit_matches_the_replays(&model, device, &rows);
    probe_scope_of_the_poison_before_the_asserts_run(&model, device, &mut ctrl_dec, &want12);
    for (step, row) in rows.iter() {
        assert_row_is_the_eager_row(row, &want[*step], *step, "logits captured first");
    }
    eprintln!(
        "logits_first_then_no_logits: nodes logits={nodes_logits} both={nodes_both} \
         (no-logits arm = {}), logits_buf held across the no-logits step, {} later rows equal \
         the eager decode exactly",
        nodes_both - nodes_logits,
        TOKENS.len() - 2
    );
}

fn two_decoders_one_arm_each_interleaved(device: &Device) {
    let model = load_tiny(device);
    let want = eager_rows(&model, device, &TOKENS);
    let mut a = new_decoder(&model, device);
    let mut b = new_decoder(&model, device);
    let mut rows_a: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut rows_b: Vec<(usize, Vec<f32>)> = Vec::new();
    for (step, &t) in TOKENS.iter().enumerate() {
        rows_a.push((step, a.forward_decode_logits_vec(t).expect("decoder a")));
        rows_b.push((step, b.forward_decode_logits_vec(t).expect("decoder b")));
    }
    report_diffs("two decoders, decoder a", &rows_a, &want);
    report_diffs("two decoders, decoder b", &rows_b, &want);
    for (step, row) in rows_a.iter().chain(rows_b.iter()) {
        assert_row_is_the_eager_row(row, &want[*step], *step, "two decoders, one arm each");
    }
}

#[test]
fn two_graphs_of_the_same_arm_coexist_on_the_forked_capture() {
    let _serialized = one_at_a_time();
    two_decoders_one_arm_each_interleaved(&forked_capture_device());
}

#[test]
fn two_graphs_of_the_same_arm_coexist_on_the_device_stream_capture() {
    let _serialized = one_at_a_time();
    two_decoders_one_arm_each_interleaved(&device_stream_capture_device());
}

#[test]
fn the_arms_keep_their_own_graph_when_the_no_logits_arm_is_captured_first() {
    let _serialized = one_at_a_time();
    no_logits_first_then_logits(&forked_capture_device());
}

#[test]
fn the_arms_keep_their_own_graph_when_the_logits_arm_is_captured_first() {
    let _serialized = one_at_a_time();
    logits_first_then_no_logits(&forked_capture_device());
}

#[test]
fn the_device_stream_capture_keeps_the_arms_apart_when_the_no_logits_arm_is_captured_first() {
    let _serialized = one_at_a_time();
    no_logits_first_then_logits(&device_stream_capture_device());
}

#[test]
fn the_device_stream_capture_keeps_the_arms_apart_when_the_logits_arm_is_captured_first() {
    let _serialized = one_at_a_time();
    logits_first_then_no_logits(&device_stream_capture_device());
}
