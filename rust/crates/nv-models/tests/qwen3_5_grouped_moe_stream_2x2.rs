#![cfg(feature = "cuda")]

mod common;
use common::htod_f32;
use common::htod_u8;
use common::LcgInc1HalfCentered as Lcg;
use common::sf_swizzled;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor};
use cudarc::driver::{CudaSlice, CudaStream};
use nv_layers::moe_grouped::{
    forward_grouped, forward_grouped_decode, GroupedDecodeContext, MoeGroupedWeights,
};
use nv_quant::nvfp4::{swizzle_scales, Nvfp4GemmRunner, Nvfp4Tensor, BLOCK_SIZE};

const HIDDEN: usize = 2048;
const INTER: usize = 512;
const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const N_TOKENS: usize = 16;

fn gated() -> bool {
    if std::env::var("NV_MOE_STREAM_2X2").as_deref() != Ok("1") {
        panic!("set NV_MOE_STREAM_2X2=1 to run (it must never silently skip)");
    }
    true
}

fn rand_expert(seed: u64) -> (Nvfp4Tensor, Nvfp4Tensor, Nvfp4Tensor) {
    let mut rng = Lcg(seed);
    let mk = |n: usize, k: usize, rng: &mut Lcg| {
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..k).map(|_| rng.next_f32()).collect())
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
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        gate_w: htod_u8(stream, &gate_p),
        gate_w_scales: htod_u8(stream, &gate_s),
        gate_alphas: htod_f32(stream, &ones),
        gate_a_stride_elems: HIDDEN as i64,
        gate_b_stride_elems: HIDDEN as i64,
        gate_c_stride_elems: INTER as i64,
        up_w: htod_u8(stream, &up_p),
        up_w_scales: htod_u8(stream, &up_s),
        up_alphas: htod_f32(stream, &ones),
        down_w: htod_u8(stream, &down_p),
        down_w_scales: htod_u8(stream, &down_s),
        down_alphas: htod_f32(stream, &ones),
        down_a_stride_elems: INTER as i64,
        down_b_stride_elems: INTER as i64,
        down_c_stride_elems: HIDDEN as i64,
        runner: runner.clone(),
        input_globals_gate_up: htod_f32(stream, &ones),
        input_globals_down: htod_f32(stream, &ones),
        input_globals_gate_up_host: ones.clone(),
        input_globals_down_host: ones,
    }
}

fn health(tag: &str, v: &[f32]) -> usize {
    let nan = v.iter().filter(|x| !x.is_finite()).count();
    let finite: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let (mn, mx) = finite
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), x| (a.min(*x), b.max(*x)));
    eprintln!(
        "[2x2] {tag}: nonfinite={nan}/{} min={:.4} max={:.4}",
        v.len(),
        if finite.is_empty() { f32::NAN } else { mn },
        if finite.is_empty() { f32::NAN } else { mx }
    );
    nan
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

struct ArmOut {
    prefill: Vec<f32>,
    decode: Vec<f32>,
}

fn run_arm(label: &str, device: Device) -> ArmOut {
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = dev.cuda_stream();
    eprintln!(
        "[2x2] arm={label} cu_stream={:?} multi_stream={} event_tracking={}",
        stream.cu_stream(),
        stream.context().is_in_multi_stream_mode(),
        stream.context().is_event_tracking()
    );
    let runner = Arc::new(Mutex::new(
        Nvfp4GemmRunner::new(stream.clone()).expect("nvfp4 runner"),
    ));

    let experts: Vec<_> = (0..NUM_EXPERTS).map(|e| rand_expert(0x100 + e as u64)).collect();
    let w = grouped_weights_from(&stream, &runner, &experts);

    let mut rng = Lcg(0xF00D);
    let x_f: Vec<f32> = (0..N_TOKENS * HIDDEN).map(|_| rng.next_f32()).collect();
    let x = Tensor::from_vec(x_f, (N_TOKENS, HIDDEN), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let mut ids: Vec<u32> = Vec::new();
    let mut wts: Vec<f32> = Vec::new();
    for _ in 0..N_TOKENS {
        let a = (rng.next_u32() % NUM_EXPERTS as u32) as u32;
        let b = (a + 1 + rng.next_u32() % (NUM_EXPERTS as u32 - 1)) % NUM_EXPERTS as u32;
        ids.extend_from_slice(&[a, b]);
        wts.extend_from_slice(&[0.5, 0.5]);
    }

    let prefill = forward_grouped(&w, &w, &x, &ids, &wts, N_TOKENS, TOP_K, &device)
        .expect("forward_grouped");
    let prefill = to_vec(&prefill);
    health(&format!("{label}/forward_grouped"), &prefill);

    let mut ctx =
        GroupedDecodeContext::new_multi(HIDDEN, INTER, TOP_K, NUM_EXPERTS, N_TOKENS, &stream)
            .expect("decode ctx");
    let mut rl = Lcg(0xBEEF);
    let logits_f: Vec<f32> = (0..N_TOKENS * NUM_EXPERTS).map(|_| rl.next_f32()).collect();
    let router_logits =
        Tensor::from_vec(logits_f, (N_TOKENS, NUM_EXPERTS), &device).unwrap();
    let decode = forward_grouped_decode(
        &w,
        &mut ctx,
        &x,
        &router_logits,
        None,
        0,
        0.0,
        false,
        1.0,
        &device,
    )
    .expect("forward_grouped_decode");
    let decode = to_vec(&decode);
    health(&format!("{label}/forward_grouped_decode"), &decode);

    ArmOut { prefill, decode }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0f64, |acc, (x, y)| acc.max((*x as f64 - *y as f64).abs()))
}

pub const THE_35B_GROUPED_MOE_STREAM_DEFECT_DOES_NOT_REPRODUCE_AT_ONE_LAYER: &str =
    "This suite is the negative control for the grouped-MoE stream defect, not coverage of it: \
     one layer of synthetic NVFP4 experts through forward_grouped and forward_grouped_decode is \
     bit-identical on Device::new_cuda and Device::new_cuda_with_stream, so the sm120 grouped FP4 \
     kernels are not stream-dependent by themselves. The real defect needs the 48-layer \
     RedHatAI/Qwen3.6-35B-A3B-NVFP4 checkpoint, where all layers share one GroupedDecodeContext \
     and its single CUTLASS workspace; reproduce it with qwen36_chat_validate, arm grouped+unrouted";

#[test]
#[ignore = "needs a CUDA device and the sm120 grouped FP4 kernels; set NV_MOE_STREAM_2X2=1"]
fn grouped_moe_kernels_agree_on_the_legacy_and_a_non_default_stream() {
    if !gated() {
        return;
    }
    let arm = std::env::var("NV_MOE_STREAM_ARM").unwrap_or_else(|_| "both".into());

    let plain = (arm == "both" || arm == "plain").then(|| {
        run_arm(
            "new_cuda",
            Device::new_cuda(0).expect("Device::new_cuda(0)"),
        )
    });
    let with_stream = (arm == "both" || arm == "stream").then(|| {
        run_arm(
            "new_cuda_with_stream",
            Device::new_cuda_with_stream(0).expect("Device::new_cuda_with_stream(0)"),
        )
    });

    for (label, out) in [("new_cuda", &plain), ("new_cuda_with_stream", &with_stream)] {
        let Some(out) = out else { continue };
        assert_eq!(
            out.prefill.iter().filter(|v| !v.is_finite()).count(),
            0,
            "{label}: forward_grouped produced non-finite values"
        );
        assert_eq!(
            out.decode.iter().filter(|v| !v.is_finite()).count(),
            0,
            "{label}: forward_grouped_decode produced non-finite values"
        );
    }

    if let (Some(p), Some(s)) = (&plain, &with_stream) {
        let dp = max_abs_diff(&p.prefill, &s.prefill);
        let dd = max_abs_diff(&p.decode, &s.decode);
        eprintln!("[2x2] cross-device max_abs_diff prefill={dp:.3e} decode={dd:.3e}");
        assert!(
            dp == 0.0 && dd == 0.0,
            "grouped MoE is stream-dependent at one layer: the same inputs give different outputs \
             on Device::new_cuda vs Device::new_cuda_with_stream (prefill {dp:.3e}, decode \
             {dd:.3e}) -- {THE_35B_GROUPED_MOE_STREAM_DEFECT_DOES_NOT_REPRODUCE_AT_ONE_LAYER}"
        );
    }
}
