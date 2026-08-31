#![cfg(feature = "cuda")]

mod common;
use candle_core::{DType, Device, Tensor};
use common::host;
use cudarc::driver::CudaSlice;
use half::bf16;
use nv_layers::linear::Linear;
use nv_layers::linear_attn::{LinAttnState, LinearAttention, LinearAttentionConfig};
use std::sync::{Arc, Mutex};

const DECODE_STEPS_12_ENOUGH_FOR_STATE_DRIFT_TO_COMPOUND: usize = 12;
const MROW_ROWS_SWEEP_COVERS_ROWS2_AND_XF32_DISPATCH: [usize; 3] = [2, 3, 7];

static ENV_LOCK_BECAUSE_NV_Q38_GDN_QKVZ_ONE_IS_PROCESS_GLOBAL: Mutex<()> = Mutex::new(());

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 33) & 0x7f_ffff) as f32 / 8388608.0) * 2.0 - 1.0
    }
    fn bf16s(&mut self, n: usize, gain: f32) -> Vec<bf16> {
        (0..n)
            .map(|_| bf16::from_f32(self.next_f32() * gain))
            .collect()
    }
}

fn fp8_resident_linear(
    bytes: &[u8],
    scales: &[f32],
    out_f: usize,
    in_f: usize,
    device: &Device,
) -> Linear {
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = dev.cuda_stream();
    #[allow(deprecated)]
    let wq_dev: CudaSlice<u8> = stream.clone_htod(bytes).unwrap();
    let runner = Arc::new(Mutex::new(
        nv_quant::fp8::Fp8GemmRunner::new(stream.clone()).unwrap(),
    ));
    Linear::new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
        wq_dev,
        scales.to_vec(),
        in_f,
        out_f,
        None,
        device,
        runner,
        nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
    )
    .unwrap()
}

fn build_layer(rng: &mut Lcg, device: &Device) -> (LinearAttentionConfig, LinearAttention) {
    let cfg = LinearAttentionConfig {
        hidden_size: 512,
        linear_num_key_heads: 4,
        linear_num_value_heads: 12,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        mamba_ssm_dtype: DType::F32,
        rms_eps: 1e-6,
    };
    let h = cfg.hidden_size;
    let conv_dim = cfg.conv_dim();
    let value_dim = cfg.value_dim();
    let n_v = cfg.linear_num_value_heads;
    let qkv_host = rng.bf16s(conv_dim * h, 0.05);
    let z_host = rng.bf16s(value_dim * h, 0.05);
    let (qkv_bytes, qkv_scales) =
        nv_quant::fp8::quantize_e4m3_per_row(&qkv_host, conv_dim, h).unwrap();
    let (z_bytes, z_scales) = nv_quant::fp8::quantize_e4m3_per_row(&z_host, value_dim, h).unwrap();
    let t = |v: Vec<bf16>, shape: &[usize]| {
        Tensor::from_vec(v, shape, device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    };
    let la = LinearAttention::new(
        cfg,
        fp8_resident_linear(&qkv_bytes, &qkv_scales, conv_dim, h, device),
        fp8_resident_linear(&z_bytes, &z_scales, value_dim, h, device),
        Linear::new(t(rng.bf16s(n_v * h, 0.05), &[n_v, h]), None).unwrap(),
        Linear::new(t(rng.bf16s(n_v * h, 0.05), &[n_v, h]), None).unwrap(),
        t(
            rng.bf16s(conv_dim * cfg.linear_conv_kernel_dim, 0.3),
            &[conv_dim, 1, cfg.linear_conv_kernel_dim],
        ),
        t(rng.bf16s(n_v, 0.5), &[n_v]),
        t(rng.bf16s(n_v, 0.5), &[n_v]),
        t(rng.bf16s(cfg.linear_value_head_dim, 1.0), &[cfg.linear_value_head_dim]),
        Linear::new(t(rng.bf16s(h * value_dim, 0.05), &[h, value_dim]), None).unwrap(),
    )
    .unwrap();
    (cfg, la)
}

fn assert_states_identical(name: &str, a: &LinAttnState, b: &LinAttnState) {
    assert_eq!(
        host(a.conv_state()),
        host(b.conv_state()),
        "{name}: conv state must be bitwise identical, the arm only changes launch count"
    );
    assert_eq!(
        host(a.recurrent_state()),
        host(b.recurrent_state()),
        "{name}: recurrent state must be bitwise identical, the arm only changes launch count"
    );
}

#[test]
fn qkvz_one_decode_chain_is_bitwise_identical_to_the_two_launch_baseline() {
    let _env_guard = ENV_LOCK_BECAUSE_NV_Q38_GDN_QKVZ_ONE_IS_PROCESS_GLOBAL
        .lock()
        .unwrap();
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("[qkvz-one decode] skip: no CUDA device");
        return;
    };
    let mut rng = Lcg(0x9e3779b97f4a7c15);
    let (cfg, la) = build_layer(&mut rng, &device);
    let st_off = la.new_fused_state(&device).unwrap();
    let st_on = la.new_fused_state(&device).unwrap();
    let mut arm_took_effect = false;
    for step in 0..DECODE_STEPS_12_ENOUGH_FOR_STATE_DRIFT_TO_COMPOUND {
        let x = Tensor::from_vec(
            rng.bf16s(cfg.hidden_size, 0.5),
            (1usize, 1usize, cfg.hidden_size),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        std::env::set_var("NV_Q38_GDN_QKVZ_ONE", "0");
        let out_off = la
            .forward_decode_fused(&x, &st_off)
            .unwrap()
            .expect("baseline fused decode refused");
        let out_off_host = host(&out_off);
        std::env::set_var("NV_Q38_GDN_QKVZ_ONE", "1");
        arm_took_effect = true;
        let out_on = la
            .forward_decode_fused(&x, &st_on)
            .unwrap()
            .expect("qkvz-one fused decode refused");
        if let Device::Cuda(d) = &device {
            d.cuda_stream().synchronize().unwrap();
        }
        assert_eq!(
            host(&out_on),
            out_off_host,
            "step {step}: decode output must be bitwise identical, the dual kernel replays the \
             exact per-row gemv math"
        );
        assert_states_identical(&format!("decode step {step}"), &st_on, &st_off);
    }
    assert!(arm_took_effect);
    std::env::set_var("NV_Q38_GDN_QKVZ_ONE", "0");
}

const AB_ONE_REL_L2_BOUND_2E2_MATCHES_GDN_FUSED_DECODE_HOUSE_TOLERANCE_BECAUSE_CUBLAS_MAY_TILE_N2V_DIFFERENTLY:
    f32 = 2e-2;

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

#[test]
fn ab_one_mrow_verify_stays_inside_the_house_tolerance_of_the_two_gemm_baseline() {
    let _env_guard = ENV_LOCK_BECAUSE_NV_Q38_GDN_QKVZ_ONE_IS_PROCESS_GLOBAL
        .lock()
        .unwrap();
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("[ab-one mrow] skip: no CUDA device");
        return;
    };
    let mut rng = Lcg(0xc0ffee11);
    let (cfg, la) = build_layer(&mut rng, &device);
    for &m in &MROW_ROWS_SWEEP_COVERS_ROWS2_AND_XF32_DISPATCH {
        let st_off = la.new_fused_state(&device).unwrap();
        let st_on = la.new_fused_state(&device).unwrap();
        let ckpts_off = la
            .new_fused_verify_ckpt_rows_off_one_slab_so_chunk_kernels_write_ckpts_in_place(
                &device, m,
            )
            .unwrap();
        let ckpts_on = la
            .new_fused_verify_ckpt_rows_off_one_slab_so_chunk_kernels_write_ckpts_in_place(
                &device, m,
            )
            .unwrap();
        let x = Tensor::from_vec(
            rng.bf16s(m * cfg.hidden_size, 0.5),
            (1usize, m, cfg.hidden_size),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        std::env::set_var("NV_Q38_GDN_AB_ONE", "0");
        let out_off = la
            .forward_verify_mrow_projections_once_because_per_row_fused_steps_reread_every_gdn_weight(
                &x, &st_off, &ckpts_off,
                None,
            )
            .unwrap()
            .expect("baseline mrow verify refused");
        let out_off_host = host(&out_off);
        std::env::set_var("NV_Q38_GDN_AB_ONE", "1");
        let out_on = la
            .forward_verify_mrow_projections_once_because_per_row_fused_steps_reread_every_gdn_weight(
                &x, &st_on, &ckpts_on,
                None,
            )
            .unwrap()
            .expect("ab-one mrow verify refused");
        if let Device::Cuda(d) = &device {
            d.cuda_stream().synchronize().unwrap();
        }
        let err_out = rel_l2(&host(&out_on), &out_off_host);
        let err_rec = rel_l2(
            &host(st_on.recurrent_state()),
            &host(st_off.recurrent_state()),
        );
        eprintln!("[ab-one mrow] m={m} rel_l2 out={err_out:.3e} rec_state={err_rec:.3e}");
        assert_eq!(
            host(st_on.conv_state()),
            host(st_off.conv_state()),
            "m={m}: conv state never reads a/b so it must stay bitwise identical"
        );
        for e in [err_out, err_rec] {
            assert!(
                e < AB_ONE_REL_L2_BOUND_2E2_MATCHES_GDN_FUSED_DECODE_HOUSE_TOLERANCE_BECAUSE_CUBLAS_MAY_TILE_N2V_DIFFERENTLY,
                "m={m}: ab-one diverged past the house tolerance: {e}"
            );
        }
    }
    std::env::set_var("NV_Q38_GDN_AB_ONE", "0");
}

#[test]
fn qkvz_one_mrow_verify_is_bitwise_identical_to_the_two_launch_baseline() {
    let _env_guard = ENV_LOCK_BECAUSE_NV_Q38_GDN_QKVZ_ONE_IS_PROCESS_GLOBAL
        .lock()
        .unwrap();
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("[qkvz-one mrow] skip: no CUDA device");
        return;
    };
    let mut rng = Lcg(0x51ed270b);
    let (cfg, la) = build_layer(&mut rng, &device);
    for &m in &MROW_ROWS_SWEEP_COVERS_ROWS2_AND_XF32_DISPATCH {
        let st_off = la.new_fused_state(&device).unwrap();
        let st_on = la.new_fused_state(&device).unwrap();
        let warm = Tensor::from_vec(
            rng.bf16s(cfg.hidden_size, 0.5),
            (1usize, 1usize, cfg.hidden_size),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        std::env::set_var("NV_Q38_GDN_QKVZ_ONE", "0");
        la.forward_decode_fused(&warm, &st_off).unwrap().unwrap();
        la.forward_decode_fused(&warm, &st_on).unwrap().unwrap();
        let ckpts_off = la
            .new_fused_verify_ckpt_rows_off_one_slab_so_chunk_kernels_write_ckpts_in_place(
                &device, m,
            )
            .unwrap();
        let ckpts_on = la
            .new_fused_verify_ckpt_rows_off_one_slab_so_chunk_kernels_write_ckpts_in_place(
                &device, m,
            )
            .unwrap();
        let x = Tensor::from_vec(
            rng.bf16s(m * cfg.hidden_size, 0.5),
            (1usize, m, cfg.hidden_size),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        std::env::set_var("NV_Q38_GDN_QKVZ_ONE", "0");
        let out_off = la
            .forward_verify_mrow_projections_once_because_per_row_fused_steps_reread_every_gdn_weight(
                &x, &st_off, &ckpts_off,
                None,
            )
            .unwrap()
            .expect("baseline mrow verify refused");
        let out_off_host = host(&out_off);
        std::env::set_var("NV_Q38_GDN_QKVZ_ONE", "1");
        let out_on = la
            .forward_verify_mrow_projections_once_because_per_row_fused_steps_reread_every_gdn_weight(
                &x, &st_on, &ckpts_on,
                None,
            )
            .unwrap()
            .expect("qkvz-one mrow verify refused");
        if let Device::Cuda(d) = &device {
            d.cuda_stream().synchronize().unwrap();
        }
        assert_eq!(
            host(&out_on),
            out_off_host,
            "m={m}: mrow verify output must be bitwise identical"
        );
        assert_states_identical(&format!("m={m} live"), &st_on, &st_off);
        for (j, (co, cn)) in ckpts_off.iter().zip(ckpts_on.iter()).enumerate() {
            assert_states_identical(&format!("m={m} ckpt {j}"), cn, co);
        }
    }
    std::env::set_var("NV_Q38_GDN_QKVZ_ONE", "0");
}
