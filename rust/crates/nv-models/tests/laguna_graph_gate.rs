#![cfg(feature = "cuda")]

mod common;
use common::htod_f32;
use common::htod_u8;
use common::LcgInc1HalfCentered as Lcg;
use common::sf_swizzled;
use std::sync::{Arc, Mutex, MutexGuard};

use candle_core::{DType, Device, Tensor};
use cudarc::driver::{CudaSlice, CudaStream};
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::moe_grouped::{GroupedDecodeContext, MoeGroupedWeights};
use nv_layers::norm::RmsNorm;
use nv_models::laguna::LagunaMoe;
use nv_models::laguna_graph::{graph_enabled, whole_step_graph_enabled, LagunaMoeGraphs};
use nv_quant::nvfp4::{swizzle_scales, Nvfp4GemmRunner, Nvfp4Tensor, BLOCK_SIZE};

const HIDDEN: usize = 2048;
const INTER: usize = 512;
const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const N_LAYERS: usize = 2;

const GRAPH_ENV: &str = "NV_LAGUNA_GRAPH";
const TAIL_FUSE_ENV: &str = "NV_MOE_TAIL_FUSE";

fn one_at_a_time() -> MutexGuard<'static, ()> {
    static SERIALIZE: Mutex<()> = Mutex::new(());
    SERIALIZE.lock().unwrap_or_else(|e| e.into_inner())
}

struct EnvRestore {
    key: &'static str,
    orig: Option<String>,
}

impl EnvRestore {
    fn capture(key: &'static str) -> Self {
        Self {
            key,
            orig: std::env::var(key).ok(),
        }
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        match &self.orig {
            Some(v) => std::env::set_var(self.key, v),
            None => std::env::remove_var(self.key),
        }
    }
}

fn rand_expert(seed: u64) -> (Nvfp4Tensor, Nvfp4Tensor, Nvfp4Tensor) {
    let mut rng = Lcg(seed);
    let mk = |n: usize, k: usize, rng: &mut Lcg| {
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..k).map(|_| rng.next_f32() * 0.1).collect())
            .collect();
        Nvfp4Tensor::quantize_rows(&rows)
    };
    (
        mk(INTER, HIDDEN, &mut rng),
        mk(INTER, HIDDEN, &mut rng),
        mk(HIDDEN, INTER, &mut rng),
    )
}

fn grouped_weights_from(
    stream: &Arc<CudaStream>,
    runner: &Arc<Mutex<Nvfp4GemmRunner>>,
    experts: &[(Nvfp4Tensor, Nvfp4Tensor, Nvfp4Tensor)],
    hidden: usize,
    inter: usize,
) -> MoeGroupedWeights {
    let e = experts.len();
    let (mut gate_p, mut gate_s) = (Vec::new(), Vec::new());
    let (mut up_p, mut up_s) = (Vec::new(), Vec::new());
    let (mut down_p, mut down_s) = (Vec::new(), Vec::new());
    for (g, u, d) in experts {
        gate_p.extend_from_slice(&g.data);
        gate_s.extend_from_slice(&sf_swizzled(g));
        up_p.extend_from_slice(&u.data);
        up_s.extend_from_slice(&sf_swizzled(u));
        down_p.extend_from_slice(&d.data);
        down_s.extend_from_slice(&sf_swizzled(d));
    }
    let ones = vec![1.0f32; e];
    MoeGroupedWeights {
        num_experts: e,
        hidden_size: hidden,
        intermediate_size: inter,
        folded_shared: false,
        gate_w: htod_u8(stream, &gate_p),
        gate_w_scales: htod_u8(stream, &gate_s),
        gate_alphas: htod_f32(stream, &ones),
        gate_a_stride_elems: hidden as i64,
        gate_b_stride_elems: hidden as i64,
        gate_c_stride_elems: inter as i64,
        up_w: htod_u8(stream, &up_p),
        up_w_scales: htod_u8(stream, &up_s),
        up_alphas: htod_f32(stream, &ones),
        down_w: htod_u8(stream, &down_p),
        down_w_scales: htod_u8(stream, &down_s),
        down_alphas: htod_f32(stream, &ones),
        down_a_stride_elems: inter as i64,
        down_b_stride_elems: inter as i64,
        down_c_stride_elems: hidden as i64,
        runner: runner.clone(),
        input_globals_gate_up: htod_f32(stream, &ones),
        input_globals_down: htod_f32(stream, &ones),
        input_globals_gate_up_host: ones.clone(),
        input_globals_down_host: ones,
    }
}

fn bf16_dev(rng: &mut Lcg, shape: (usize, usize), scale: f32, device: &Device) -> Tensor {
    let n = shape.0 * shape.1;
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32() * scale).collect();
    Tensor::from_vec(data, shape, &Device::Cpu)
        .expect("cpu tensor")
        .to_dtype(DType::BF16)
        .expect("to bf16")
        .to_device(device)
        .expect("to device")
}

struct World {
    device: Device,
    dev: candle_core::CudaDevice,
    stream: Arc<CudaStream>,
    w: MoeGroupedWeights,
    moe: LagunaMoe,
    norm: RmsNorm,
}

fn build_world() -> World {
    let device = Device::new_cuda(0).expect("cuda device 0");
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = dev.cuda_stream();
    let runner = Arc::new(Mutex::new(
        Nvfp4GemmRunner::new(stream.clone()).expect("nvfp4 runner"),
    ));
    let experts: Vec<_> = (0..NUM_EXPERTS)
        .map(|e| rand_expert(0xA100 + e as u64))
        .collect();
    let w = grouped_weights_from(&stream, &runner, &experts, HIDDEN, INTER);

    let mut rng = Lcg(0x5EED);
    let gate = Linear::new(bf16_dev(&mut rng, (NUM_EXPERTS, HIDDEN), 0.5, &device), None)
        .expect("gate linear");
    let bias_vals: Vec<f32> = (0..NUM_EXPERTS).map(|_| rng.next_f32() * 0.1).collect();
    let selection_bias =
        Tensor::from_vec(bias_vals, (NUM_EXPERTS,), &device).expect("selection bias");
    let shared_expert = Mlp::new(
        Linear::new(bf16_dev(&mut rng, (INTER, HIDDEN), 0.05, &device), None).expect("sh gate"),
        Linear::new(bf16_dev(&mut rng, (INTER, HIDDEN), 0.05, &device), None).expect("sh up"),
        Linear::new(bf16_dev(&mut rng, (HIDDEN, INTER), 0.05, &device), None).expect("sh down"),
    )
    .expect("shared mlp");
    let moe = LagunaMoe {
        num_experts: NUM_EXPERTS,
        top_k: TOP_K,
        norm_topk: true,
        routed_scaling: 1.0,
        softcap: 0.0,
        gate,
        selection_bias,
        experts: Vec::new(),
        shared_expert,
        grouped: Mutex::new(None),
    };
    let norm_w = Tensor::ones(HIDDEN, DType::BF16, &device).expect("norm weight");
    let norm = RmsNorm::new(norm_w, 1e-6);
    World {
        device,
        dev,
        stream,
        w,
        moe,
        norm,
    }
}

fn resid_input(seed: u64, device: &Device) -> Tensor {
    let mut rng = Lcg(seed);
    let data: Vec<f32> = (0..HIDDEN).map(|_| rng.next_f32() * 2.0).collect();
    Tensor::from_vec(data, (1usize, HIDDEN), &Device::Cpu)
        .expect("cpu resid")
        .to_dtype(DType::BF16)
        .expect("resid bf16")
        .to_device(device)
        .expect("resid to cuda")
}

fn to_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .expect("to f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("to vec")
}

fn assert_all_finite(tag: &str, v: &[f32]) {
    let bad = v.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(bad, 0, "{tag}: {bad}/{} nonfinite values", v.len());
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in max_abs_diff");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn env_key_derivation_full_and_2_enable_both_gates_1_and_true_enable_only_per_layer_and_truthy_synonyms_stay_off(
) {
    let _guard = one_at_a_time();
    let _restore = EnvRestore::capture(GRAPH_ENV);

    std::env::remove_var(GRAPH_ENV);
    assert!(
        !graph_enabled(),
        "unset {GRAPH_ENV} must leave the graph path OFF: capture is opt-in, a default-on refactor changes serving behavior for every laguna deploy"
    );
    assert!(!whole_step_graph_enabled());

    let table: &[(&str, bool, bool)] = &[
        ("1", true, false),
        ("true", true, false),
        ("TRUE", true, false),
        ("True", true, false),
        ("2", true, true),
        ("full", true, true),
        ("FULL", true, true),
        ("Full", true, true),
        ("0", false, false),
        ("false", false, false),
        ("", false, false),
        ("yes", false, false),
        ("on", false, false),
        ("3", false, false),
        ("enable", false, false),
    ];
    for (val, want_graph, want_whole) in table {
        std::env::set_var(GRAPH_ENV, val);
        assert_eq!(
            graph_enabled(),
            *want_graph,
            "{GRAPH_ENV}={val:?}: graph_enabled must be {want_graph} -- yes/on/3 rows are the negative control against a refactor to bare truthiness"
        );
        assert_eq!(
            whole_step_graph_enabled(),
            *want_whole,
            "{GRAPH_ENV}={val:?}: whole_step_graph_enabled must be {want_whole} -- only 2/full opt into the whole-step graph laguna_dflash gates on"
        );
        assert!(
            !whole_step_graph_enabled() || graph_enabled(),
            "{GRAPH_ENV}={val:?}: whole-step implies per-layer -- splitting the vocabularies would run the whole-step graph without the per-layer plumbing it layers on"
        );
    }
}

#[test]
fn fresh_graphs_have_zero_counters_no_cached_layers_are_not_failed_and_mark_failed_is_observable_before_any_capture(
) {
    let _guard = one_at_a_time();
    let device = Device::new_cuda(0).expect("cuda device 0");
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let mut graphs = LagunaMoeGraphs::new(&dev, 3).expect("LagunaMoeGraphs::new");
    assert_eq!(graphs.layers_cached(), 0, "no layer may be cached before the first forward");
    assert_eq!(graphs.captures(), 0, "captures counter must start at 0");
    assert_eq!(graphs.replays(), 0, "replays counter must start at 0");
    assert!(
        !graphs.failed(),
        "a fresh LagunaMoeGraphs must not report failed: laguna.rs gates every graph forward on !failed(), so a true here would permanently disable the captured path"
    );
    graphs.synchronize().expect("synchronize on idle forked stream");
    graphs.mark_failed();
    assert!(
        graphs.failed(),
        "mark_failed must stick: laguna.rs relies on it as the permanent uncaptured-fallback latch"
    );
    assert_eq!(graphs.layers_cached(), 0);
}

#[test]
fn forward_layer_returns_err_not_panic_on_out_of_range_layer_bad_resid_shape_dtype_device_and_mismatched_ctx_and_burns_no_capture(
) {
    let _guard = one_at_a_time();
    let world = build_world();
    let mut ctx = GroupedDecodeContext::new(HIDDEN, INTER, TOP_K, NUM_EXPERTS, &world.stream)
        .expect("decode ctx");
    let mut graphs = LagunaMoeGraphs::new(&world.dev, N_LAYERS).expect("graphs");
    let good = resid_input(0xAAAA, &world.device);

    let e = graphs
        .forward_layer(
            N_LAYERS,
            &world.moe,
            &world.norm,
            &world.w,
            &mut ctx,
            &good,
            &world.dev,
        )
        .expect_err("layer index == num_layers must be rejected");
    assert!(
        format!("{e:#}").contains("out of range"),
        "out-of-range layer error must say so, got: {e:#}"
    );

    let wide = resid_input(0xAAAB, &world.device)
        .broadcast_as((2usize, HIDDEN))
        .and_then(|t| t.contiguous())
        .expect("2-row resid");
    graphs
        .forward_layer(0, &world.moe, &world.norm, &world.w, &mut ctx, &wide, &world.dev)
        .expect_err("a [2, hidden] residual must be rejected: the graph is captured for exactly one decode token");

    let f32_resid = good.to_dtype(DType::F32).expect("f32 resid");
    graphs
        .forward_layer(0, &world.moe, &world.norm, &world.w, &mut ctx, &f32_resid, &world.dev)
        .expect_err("a F32 residual must be rejected: the captured buffers are BF16");

    let cpu_resid = good.to_device(&Device::Cpu).expect("cpu resid");
    let e = graphs
        .forward_layer(0, &world.moe, &world.norm, &world.w, &mut ctx, &cpu_resid, &world.dev)
        .expect_err("a CPU residual must be rejected before any dtod staging");
    assert!(
        format!("{e:#}").contains("must be CUDA"),
        "cpu residual error must name the device contract, got: {e:#}"
    );

    let mut ctx_bad =
        GroupedDecodeContext::new(HIDDEN, INTER, TOP_K, NUM_EXPERTS * 2, &world.stream)
            .expect("mismatched ctx");
    let e = graphs
        .forward_layer(0, &world.moe, &world.norm, &world.w, &mut ctx_bad, &good, &world.dev)
        .expect_err("a ctx whose expert count disagrees with the weights must be rejected");
    assert!(
        format!("{e:#}").contains("ctx shape mismatch"),
        "ctx mismatch error must say so, got: {e:#}"
    );

    assert_eq!(graphs.captures(), 0, "rejected inputs must not capture a graph");
    assert_eq!(graphs.replays(), 0, "rejected inputs must not replay a graph");
    assert_eq!(graphs.layers_cached(), 0, "rejected inputs must not populate a runner cache");
    assert!(
        !graphs.failed(),
        "input validation errors must not latch failed(): the CALLER decides that (laguna.rs mark_failed on forward error), not the gate region"
    );
}

#[test]
fn warm_capture_replay_lifecycle_counters_bitexact_replay_fresh_input_restaging_per_layer_slots_and_invalidate_on_mark_failed(
) {
    let _guard = one_at_a_time();
    let _restore = EnvRestore::capture(TAIL_FUSE_ENV);
    std::env::remove_var(TAIL_FUSE_ENV);

    let world = build_world();
    let mut ctx = GroupedDecodeContext::new(HIDDEN, INTER, TOP_K, NUM_EXPERTS, &world.stream)
        .expect("decode ctx");
    let mut graphs = LagunaMoeGraphs::new(&world.dev, N_LAYERS).expect("graphs");

    let a = resid_input(0x0A0A, &world.device);
    let b = resid_input(0x0B0B, &world.device);
    let a_vals = to_f32(&a);

    let out1 = to_f32(
        &graphs
            .forward_layer(0, &world.moe, &world.norm, &world.w, &mut ctx, &a, &world.dev)
            .expect("first forward: warm then capture"),
    );
    assert_eq!(graphs.captures(), 1, "first forward of a layer must capture exactly once");
    assert_eq!(graphs.replays(), 0, "first forward must not count as a replay");
    assert_eq!(graphs.layers_cached(), 1);
    assert_all_finite("capture output", &out1);
    assert!(
        max_abs_diff(&out1, &a_vals) > 1e-3,
        "graph output equals the residual input: the captured moe body is a no-op, every downstream check would be vacuous"
    );

    let out2 = to_f32(
        &graphs
            .forward_layer(0, &world.moe, &world.norm, &world.w, &mut ctx, &a, &world.dev)
            .expect("second forward: replay"),
    );
    assert_eq!(graphs.captures(), 1, "a cached layer must not re-warm or re-capture");
    assert_eq!(graphs.replays(), 1, "second forward of a cached layer must count one replay");
    assert_eq!(
        out2, out1,
        "replay diverged from the capture launch on identical input: the gemma4 ba6f05350 hazard shape -- a warm pass with host side effects makes the captured pass compute different state than the warm pass -- laguna's moe body must stay free of host-visible state so this stays bit-exact"
    );

    let out_b = to_f32(
        &graphs
            .forward_layer(0, &world.moe, &world.norm, &world.w, &mut ctx, &b, &world.dev)
            .expect("replay with fresh input"),
    );
    assert_eq!(graphs.replays(), 2);
    assert!(
        max_abs_diff(&out_b, &out1) > 1e-3,
        "replay ignored a different residual: the pre-graph dtod staging into ctx.resid_in is the ONLY input path a replay has -- if this row goes green while staging is broken the graph is baked to its capture input"
    );

    let out3 = to_f32(
        &graphs
            .forward_layer(0, &world.moe, &world.norm, &world.w, &mut ctx, &a, &world.dev)
            .expect("replay input a again"),
    );
    assert_eq!(graphs.replays(), 3);
    assert_eq!(
        out3, out1,
        "input a after input b must reproduce the input-a output bit-exactly: ctx buffers are shared scratch and every replay must fully rewrite what it reads"
    );

    let out_l1 = to_f32(
        &graphs
            .forward_layer(1, &world.moe, &world.norm, &world.w, &mut ctx, &a, &world.dev)
            .expect("first forward of layer 1"),
    );
    assert_eq!(graphs.captures(), 2, "layer 1 must capture its own graph in its own runner slot");
    assert_eq!(graphs.layers_cached(), 2);
    assert_eq!(
        out_l1, out1,
        "layer 1 with identical weights must reproduce layer 0 bit-exactly: same body, same shapes, distinct runner slot"
    );

    let out4 = to_f32(
        &graphs
            .forward_layer(0, &world.moe, &world.norm, &world.w, &mut ctx, &a, &world.dev)
            .expect("layer 0 replay after layer 1 capture"),
    );
    assert_eq!(
        graphs.captures(),
        2,
        "capturing layer 1 must not evict layer 0's graph: runner slots are per-layer, keyed by index not by a shared token"
    );
    assert_eq!(graphs.replays(), 4);
    assert_eq!(out4, out1);

    std::env::set_var(TAIL_FUSE_ENV, "0");
    let mut graphs_unfused = LagunaMoeGraphs::new(&world.dev, 1).expect("unfused graphs");
    let out_unfused = to_f32(
        &graphs_unfused
            .forward_layer(0, &world.moe, &world.norm, &world.w, &mut ctx, &a, &world.dev)
            .expect("unfused-tail capture"),
    );
    std::env::remove_var(TAIL_FUSE_ENV);
    assert_eq!(graphs_unfused.captures(), 1);
    assert_all_finite("unfused tail output", &out_unfused);
    let d = max_abs_diff(&out_unfused, &out1);
    assert!(
        d < 0.05,
        "NV_MOE_TAIL_FUSE=0 (3-kernel tail) and the fused scatter tail must agree within bf16 rounding, max_abs_diff={d}"
    );

    graphs.mark_failed();
    assert!(graphs.failed());
    assert_eq!(
        graphs.layers_cached(),
        0,
        "mark_failed must invalidate every cached runner: laguna.rs falls back to the uncaptured path and a later accidental replay of a stale graph would race it"
    );
}
