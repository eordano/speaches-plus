#![cfg(feature = "cuda")]

mod common;
use common::detect_major;
use candle_core::{DType, Device, Tensor};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::CudaContext;
use half::bf16;
use nv_layers::linear::Linear;
use nv_quant::nvfp4::{supports_nvfp4, Nvfp4GemmRunner};
use std::sync::{Arc, Mutex};

const GEMMA4_HIDDEN_IS_THE_K_OF_EVERY_PROJECTION_5376: usize = 5376;

const GEMMA4_K_PROJ_OUT_IS_N_KV_4_TIMES_HEAD_DIM_256: usize = 1024;

const ROWS: usize = 2048;

const REFERENCE_CHUNK: usize = 1024;

const PREFILL_CHUNK_ROWS_BELOW_WHICH_THE_MODEL_SWITCHES_PRECISION_256: usize = 256;

fn smallest_chunk(len: usize, chunk: usize) -> usize {
    let tail = len % chunk;
    if tail == 0 {
        chunk.min(len)
    } else {
        chunk.min(tail)
    }
}

fn chunked_forward(lin: &Linear, x: &Tensor, chunk: usize) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::with_capacity(ROWS * GEMMA4_K_PROJ_OUT_IS_N_KV_4_TIMES_HEAD_DIM_256);
    let mut at = 0usize;
    while at < ROWS {
        let n = chunk.min(ROWS - at);
        let slice = x.narrow(0, at, n).unwrap();
        let y = lin.forward(&slice).unwrap();
        out.extend(
            y.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
        at += n;
    }
    out
}

#[test]
fn the_projection_gemm_is_chunk_invariant_so_the_prefill_floor_is_not_a_gemm_property() {
    let Ok(ctx) = CudaContext::new(0) else {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: no CUDA context");
    };
    if !supports_nvfp4(detect_major(&ctx)) {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: no NVFP4 support");
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream).unwrap()));

    let (n, k) = (
        GEMMA4_K_PROJ_OUT_IS_N_KV_4_TIMES_HEAD_DIM_256,
        GEMMA4_HIDDEN_IS_THE_K_OF_EVERY_PROJECTION_5376,
    );
    let w: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32((i as f32 * 0.0009).cos() * 0.05))
        .collect();
    let x: Vec<bf16> = (0..ROWS * k)
        .map(|i| bf16::from_f32((i as f32 * 0.0007).sin()))
        .collect();
    let w_t = Tensor::from_vec(w, (n, k), &device).unwrap();
    let x_t = Tensor::from_vec(x.clone(), (ROWS, k), &device).unwrap();
    let lin = Linear::from_bf16_quantized_nvfp4(&w_t, None, &device, runner).unwrap();

    let floor = PREFILL_CHUNK_ROWS_BELOW_WHICH_THE_MODEL_SWITCHES_PRECISION_256;
    let reference = chunked_forward(&lin, &x_t, REFERENCE_CHUNK);
    let spread = reference.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        spread > 0.0,
        "the reference output is all zeros, so every comparison below would agree for a reason \
         that has nothing to do with chunking"
    );
    eprintln!(
        "[gemm-chunk] {ROWS}x{k} @ {k}x{n}, reference chunk {REFERENCE_CHUNK}, largest |y| \
         {spread:e}, floor {floor}"
    );

    let mut disagree: Vec<(usize, usize, f32)> = Vec::new();
    for chunk in [1024usize, 700, 512, 448, 256, 255, 192, 128, 64] {
        let got = chunked_forward(&lin, &x_t, chunk);
        let worst = reference
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let min = smallest_chunk(ROWS, chunk);
        eprintln!("[gemm-chunk] chunk {chunk:>4} (smallest {min:>4})  worst |dY| = {worst:e}");
        if worst != 0.0 {
            disagree.push((chunk, min, worst));
        }
    }

    let mut bumped = x.clone();
    bumped[0] = bf16::from_f32(x[0].to_f32() + 1.0);
    let bumped_t = Tensor::from_vec(bumped, (ROWS, k), &device).unwrap();
    let perturbed = chunked_forward(&lin, &bumped_t, REFERENCE_CHUNK);
    let moved = reference
        .iter()
        .zip(&perturbed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        moved > 0.0,
        "changing an input element did not change any output, so the comparison above cannot \
         detect a difference and every 0e0 it printed means nothing"
    );
    assert!(
        disagree.is_empty(),
        "these chunkings disagree with chunk {REFERENCE_CHUNK}: {disagree:?} as (chunk, \
         smallest, worst). A projection is a per-row function and this GEMM is chunk-invariant \
         at every row count from {ROWS} down to 64, which is what makes it NOT the mechanism \
         behind the prefill floor. The floor is a precision switch in the model \
         (gemma4 prefill_w4a4_selects, m >= {floor}), not a property of the matmul. If this \
         gate ever fires, the floor has a second cause and NV_PREFILL_CHUNK_MIN needs a fresh \
         derivation"
    );
}
