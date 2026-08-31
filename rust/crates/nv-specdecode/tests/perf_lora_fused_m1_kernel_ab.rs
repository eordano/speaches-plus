#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_kernels::graph::CudaGraphRunner;
use nv_models::gemma4_batch_graph::capture_stream::CaptureStream;
use nv_specdecode::lora_spec::{ServingDeltaSlice, ServingLoraHook};
use std::sync::Arc;
use std::time::Instant;

const PERF_GATE_ENV: &str = "NV_LORA_KERNEL_PERF_TEST";

const GEMMA4_31B_HIDDEN: usize = 5376;
const GEMMA4_31B_INTER: usize = 21504;
const GEMMA4_31B_LAYERS: usize = 60;
const RANK: usize = 8;

fn frand(seed: u64, i: usize) -> f32 {
    let mut z = seed
        .wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1))
        .wrapping_mul(0xBF58476D1CE4E5B9);
    z ^= z >> 29;
    z = z.wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 32;
    ((z & 0xFFFF) as f32 / 65535.0) - 0.5
}

fn f32_t(seed: u64, rows: usize, cols: usize, scale: f32, dev: &Device) -> Tensor {
    let v: Vec<f32> = (0..rows * cols).map(|i| frand(seed, i) * scale).collect();
    Tensor::from_vec(v, (rows, cols), dev).unwrap()
}

fn bf16_row(seed: u64, cols: usize, dev: &Device) -> Tensor {
    f32_t(seed, 1, cols, 0.05, dev)
        .to_dtype(DType::BF16)
        .unwrap()
}

fn mlp_hooks_at_gemma4_31b_shapes(dev: &Device) -> (Arc<ServingLoraHook>, Arc<ServingLoraHook>) {
    let (h, i) = (GEMMA4_31B_HIDDEN, GEMMA4_31B_INTER);
    let gu = ServingLoraHook::new(
        h,
        2 * i,
        vec![
            ServingDeltaSlice::from_peft_pair(
                "bench.gate",
                0,
                i,
                h,
                &f32_t(11, RANK, h, 0.05, dev),
                &f32_t(12, i, RANK, 1e-4, dev),
                2.0,
            )
            .unwrap(),
            ServingDeltaSlice::from_peft_pair(
                "bench.up",
                i,
                i,
                h,
                &f32_t(13, RANK, h, 0.05, dev),
                &f32_t(14, i, RANK, 1e-4, dev),
                2.0,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let dn = ServingLoraHook::new(
        i,
        h,
        vec![ServingDeltaSlice::from_peft_pair(
            "bench.down",
            0,
            h,
            i,
            &f32_t(15, RANK, i, 0.05, dev),
            &f32_t(16, h, RANK, 1e-4, dev),
            2.0,
        )
        .unwrap()],
    )
    .unwrap();
    (gu, dn)
}

fn sync(dev: &Device) {
    if let Device::Cuda(d) = dev {
        d.cuda_stream().synchronize().unwrap();
    }
}

#[test]
#[ignore = "kernel-level perf instrument, not a correctness gate: times the ServingLoraHook m=1 \
            cuda kernel path (nv_kernels lora_fused) at the exact gemma4-31B mlp shapes the \
            serving adapter hooks (gate_up in=5376 out=43008 two slices, down in=21504 out=5376, \
            rank 8), first eager per-apply, then as 120 captured nodes in one CUDA graph \
            replayed like a decode step, so the per-hook serving delta can be attributed to \
            kernel time vs graph/node overhead vs host effects without nsys. Run: \
            NVK_LANE=fusec NVK_PKG=nv-specdecode NV_LORA_KERNEL_PERF_TEST=1 \
            rust/scripts/nvk.sh probe --test perf_lora_fused_m1_kernel_ab -- --ignored \
            --test-threads=1 --nocapture"]
fn lora_fused_m1_gemma4_mlp_shapes_eager_vs_captured_graph_replay() {
    if std::env::var(PERF_GATE_ENV).ok().as_deref() != Some("1") {
        panic!(
            "{PERF_GATE_ENV}=1 not set. Green here without a run would report kernel timings \
             that never happened; this is a SKIP surfaced as a failure by design."
        );
    }
    let device = Device::new_cuda(0).expect("cuda device 0");
    let (gu, dn) = mlp_hooks_at_gemma4_31b_shapes(&device);

    let x_gu = bf16_row(31, GEMMA4_31B_HIDDEN, &device);
    let y_gu = bf16_row(32, 2 * GEMMA4_31B_INTER, &device);
    let x_dn = bf16_row(33, GEMMA4_31B_INTER, &device);
    let y_dn = bf16_row(34, GEMMA4_31B_HIDDEN, &device);

    let apply_gu = || {
        use nv_layers::linear::LoraDeltaHook as _;
        let r = gu.apply(&x_gu, &y_gu, None).expect("gate_up apply");
        assert!(
            r.is_none(),
            "m=1 cuda path applies in place; Some() means the candle fallback ran and this \
             bench is not timing the kernel"
        );
    };
    let apply_dn = || {
        use nv_layers::linear::LoraDeltaHook as _;
        let r = dn.apply(&x_dn, &y_dn, None).expect("down apply");
        assert!(r.is_none(), "down apply took the candle fallback");
    };

    for _ in 0..50 {
        apply_gu();
        apply_dn();
    }
    sync(&device);

    let iters = 2000usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        apply_gu();
    }
    sync(&device);
    let gu_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let t1 = Instant::now();
    for _ in 0..iters {
        apply_dn();
    }
    sync(&device);
    let dn_us = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!(
        "[lora_kernel_ab] EAGER gate_up_apply_us={gu_us:.2} down_apply_us={dn_us:.2} \
         per_layer_us={:.2} x60_layers_ms={:.3}",
        gu_us + dn_us,
        (gu_us + dn_us) * GEMMA4_31B_LAYERS as f64 / 1000.0
    );

    let cap = CaptureStream::for_device(&device).expect("capture stream");
    let stream = cap.stream().clone();
    let mut runner = CudaGraphRunner::new(stream.clone());
    let key = 0x10afa5edu64;
    nv_layers::cuda_stream::with_stream(stream.clone(), || {
        for _ in 0..GEMMA4_31B_LAYERS {
            apply_gu();
            apply_dn();
        }
    });
    stream.synchronize().expect("warm sync");
    runner
        .run(key, |s| {
            nv_layers::cuda_stream::with_stream(s.clone(), || {
                for _ in 0..GEMMA4_31B_LAYERS {
                    apply_gu();
                    apply_dn();
                }
            });
            Ok(())
        })
        .expect("capture 120 lora_fused nodes");
    let nodes = runner.cached_node_count();
    let reps = 200usize;
    let tr0 = Instant::now();
    for _ in 0..reps {
        runner.run(key, |_| unreachable!("cached graph must replay")).expect("replay");
    }
    stream.synchronize().expect("replay sync");
    let replay_ms = tr0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!(
        "[lora_kernel_ab] GRAPH captured_nodes={nodes} replay_ms_per_120_hook_graph={replay_ms:.3} \
         us_per_node={:.2}",
        replay_ms * 1000.0 / nodes.max(1) as f64
    );
    println!(
        "[lora_kernel_ab] FINAL eager_per_layer_us={:.2} graph_us_per_node={:.2} \
         projected_120_hook_decode_delta_ms_eager={:.3} projected_graph_ms={:.3}",
        gu_us + dn_us,
        replay_ms * 1000.0 / nodes.max(1) as f64,
        (gu_us + dn_us) * GEMMA4_31B_LAYERS as f64 / 1000.0,
        replay_ms
    );
}
