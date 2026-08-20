#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use cudarc::driver::CudaContext;
use half::bf16;
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_models::gemma4::{Gemma4Attention, LayerType, PREFILL_W4A4_MIN_M};
use nv_quant::nvfp4::{supports_nvfp4, Nvfp4GemmRunner};
use std::sync::{Arc, Mutex};

fn detect_major(ctx: &CudaContext) -> i32 {
    ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    )
    .unwrap_or(0)
}

fn rel_rms(got: &[f32], expect: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, e) in got.iter().zip(expect.iter()) {
        num += ((g - e) as f64).powi(2);
        den += (*e as f64).powi(2);
    }
    (num / den.max(1e-12)).sqrt() as f32
}

fn to_host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn bf16_tensor(seed: f32, dims: (usize, usize), device: &Device) -> Tensor {
    let (r, c) = dims;
    let flat: Vec<bf16> = (0..r * c)
        .map(|i| bf16::from_f32(0.5 * ((i as f32) * seed).sin()))
        .collect();
    Tensor::from_vec(flat, (r, c), device).unwrap()
}

fn build_attn(
    device: &Device,
    runner: &Arc<Mutex<Nvfp4GemmRunner>>,
    with_fp4: bool,
) -> Gemma4Attention {
    let hidden = 256usize;
    let head_dim = 64usize;
    let n_q = 4usize;
    let n_kv = 2usize;
    let q_dim = n_q * head_dim;
    let kv_dim = n_kv * head_dim;
    let qkv_w = bf16_tensor(0.037, (q_dim + 2 * kv_dim, hidden), device);
    let o_w = bf16_tensor(0.051, (hidden, q_dim), device);
    let qkv_proj = Linear::new(qkv_w.clone(), None).unwrap();
    let o_proj = Linear::new(o_w.clone(), None).unwrap();
    let (qkv_prefill_fp4, o_prefill_fp4) = if with_fp4 {
        (
            Some(
                Linear::from_bf16_quantized_nvfp4_dev(&qkv_w, None, device, runner.clone())
                    .unwrap(),
            ),
            Some(
                Linear::from_bf16_quantized_nvfp4_dev(&o_w, None, device, runner.clone()).unwrap(),
            ),
        )
    } else {
        (None, None)
    };
    let ones = |d: usize| RmsNorm::new(Tensor::ones(d, DType::BF16, device).unwrap(), 1e-6);
    Gemma4Attention {
        kind: LayerType::SlidingAttention,
        qkv_proj,
        q_dim,
        kv_dim,
        has_v: true,
        o_proj,
        q_norm: ones(head_dim),
        k_norm: ones(head_dim),
        v_norm: ones(head_dim),
        qkv_prefill_fp4,
        o_prefill_fp4,
    }
}

#[test]
fn prefill_w4a4_routes_large_m_and_leaves_decode_bitwise() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let major = detect_major(&ctx);
    if !supports_nvfp4(major) {
        eprintln!("skip: SM {major} lacks NVFP4");
        return;
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream).unwrap()));

    let attn_fp4 = build_attn(&device, &runner, true);
    let attn_ref = build_attn(&device, &runner, false);

    let hidden = 256usize;
    let m_prefill = 1024usize;
    assert!(m_prefill >= PREFILL_W4A4_MIN_M);
    let x_big = bf16_tensor(0.013, (m_prefill, hidden), &device);

    let n_qkv = attn_fp4.q_dim + 2 * attn_fp4.kv_dim;
    let cpu_ref = |w_seed: f32, n: usize, k: usize, x_seed: f32, m: usize| -> (Vec<f32>, f32) {
        use nv_quant::nvfp4::{cpu_nvfp4_matmul_weight_row, Nvfp4Tensor};
        let rows = |seed: f32, r: usize, c: usize| -> Vec<Vec<f32>> {
            (0..r)
                .map(|i| {
                    (0..c)
                        .map(|j| bf16::from_f32(0.5 * (((i * c + j) as f32) * seed).sin()).to_f32())
                        .collect()
                })
                .collect()
        };
        let w_rows = rows(w_seed, n, k);
        let x_rows = rows(x_seed, m, k);
        let a_q = Nvfp4Tensor::quantize_rows(&x_rows);
        let b_q = Nvfp4Tensor::quantize_rows(&w_rows);
        let expect = cpu_nvfp4_matmul_weight_row(&a_q, &b_q, m, n, k);
        let expect_f: Vec<f32> = expect.iter().map(|v| v.to_f32()).collect();
        let mut bf = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f64;
                for t in 0..k {
                    acc += (x_rows[i][t] as f64) * (w_rows[j][t] as f64);
                }
                bf[i * n + j] = acc as f32;
            }
        }
        let loss = rel_rms(&expect_f, &bf);
        (expect_f, loss)
    };

    let (expect_qkv, ref_loss_qkv) = cpu_ref(0.037, n_qkv, hidden, 0.013, m_prefill);
    let (q4, k4, v4) = attn_fp4.qkv_forward(&x_big).unwrap();
    let (qr, kr, vr) = attn_ref.qkv_forward(&x_big).unwrap();
    assert_eq!(q4.dims(), qr.dims());
    assert_eq!(k4.dims(), kr.dims());
    assert_eq!(v4.dims(), vr.dims());
    let slice_cols = |lo: usize, hi: usize| -> Vec<f32> {
        let mut out = Vec::with_capacity(m_prefill * (hi - lo));
        for i in 0..m_prefill {
            out.extend_from_slice(&expect_qkv[i * n_qkv + lo..i * n_qkv + hi]);
        }
        out
    };
    let bounds = [
        ("q", &q4, &qr, 0usize, attn_fp4.q_dim),
        (
            "k",
            &k4,
            &kr,
            attn_fp4.q_dim,
            attn_fp4.q_dim + attn_fp4.kv_dim,
        ),
        ("v", &v4, &vr, attn_fp4.q_dim + attn_fp4.kv_dim, n_qkv),
    ];
    for (name, got, bf, lo, hi) in bounds {
        let g = to_host(got);
        let r_ref = rel_rms(&g, &slice_cols(lo, hi));
        let r_bf = rel_rms(&g, &to_host(bf));
        eprintln!(
            "prefill W4A4 qkv[{name}]: vs cpu-ref {r_ref:.4}, vs bf16 {r_bf:.4} (ref loss {ref_loss_qkv:.4})"
        );
        assert!(
            r_ref < 0.25,
            "qkv[{name}] diverges from the CPU W4A4 reference: {r_ref} >= 0.25 at m={m_prefill}"
        );
        assert!(
            r_bf < ref_loss_qkv * 1.25 + 0.02,
            "qkv[{name}] bf16 loss {r_bf} exceeds the reference's own loss {ref_loss_qkv} by >25%"
        );
        assert!(
            r_bf > 0.0,
            "qkv[{name}] identical to bf16 -- fp4 path not taken"
        );
    }

    let attn_in = bf16_tensor(0.021, (m_prefill, attn_fp4.q_dim), &device);
    let (expect_o, ref_loss_o) = cpu_ref(0.051, hidden, attn_fp4.q_dim, 0.021, m_prefill);
    let o4 = attn_fp4.o_forward(&attn_in).unwrap();
    let or = attn_ref.o_forward(&attn_in).unwrap();
    assert_eq!(o4.dims(), or.dims());
    let g = to_host(&o4);
    let r_ref = rel_rms(&g, &expect_o);
    let r_bf = rel_rms(&g, &to_host(&or));
    eprintln!(
        "prefill W4A4 o_proj: vs cpu-ref {r_ref:.4}, vs bf16 {r_bf:.4} (ref loss {ref_loss_o:.4})"
    );
    assert!(
        r_ref < 0.25,
        "o_proj diverges from the CPU W4A4 reference: {r_ref} >= 0.25"
    );
    assert!(
        r_bf < ref_loss_o * 1.25 + 0.02,
        "o_proj bf16 loss {r_bf} exceeds the reference's own loss {ref_loss_o} by >25%"
    );
    assert!(r_bf > 0.0, "o_proj identical to bf16 -- fp4 path not taken");

    for m_small in [1usize, 4, 32, PREFILL_W4A4_MIN_M - 1] {
        let x_small = bf16_tensor(0.017, (m_small, hidden), &device);
        let (qs4, ks4, vs4) = attn_fp4.qkv_forward(&x_small).unwrap();
        let (qsr, ksr, vsr) = attn_ref.qkv_forward(&x_small).unwrap();
        for (name, a, b) in [("q", &qs4, &qsr), ("k", &ks4, &ksr), ("v", &vs4, &vsr)] {
            assert_eq!(
                to_host(a),
                to_host(b),
                "decode qkv[{name}] must be bitwise-identical at m={m_small}"
            );
        }
        let o_in = bf16_tensor(0.019, (m_small, attn_fp4.q_dim), &device);
        assert_eq!(
            to_host(&attn_fp4.o_forward(&o_in).unwrap()),
            to_host(&attn_ref.o_forward(&o_in).unwrap()),
            "decode o_proj must be bitwise-identical at m={m_small}"
        );
    }
}
