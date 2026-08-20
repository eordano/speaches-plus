#![cfg(feature = "cuda")]

mod common;
use common::host;
use common::rel_l2;
use candle_core::{DType, Device, Tensor};
use cudarc::driver::CudaSlice;
use half::bf16;
use nv_layers::linear::Linear;
use nv_layers::linear_attn::{LinearAttention, LinearAttentionConfig};
use std::sync::{Arc, Mutex};

const DECODE_STEPS_16_ENOUGH_FOR_STATE_DRIFT_TO_COMPOUND: usize = 16;
const REL_L2_BOUND_2E2_MATCHES_GDN_FUSED_DECODE_HOUSE_TOLERANCE: f32 = 2e-2;

static ENV_LOCK_BECAUSE_NV_Q38_GDN_FUSE_IS_PROCESS_GLOBAL: Mutex<()> = Mutex::new(());

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

struct SharedGdnWeights {
    qkv_bytes: Vec<u8>,
    qkv_scales: Vec<f32>,
    z_bytes: Vec<u8>,
    z_scales: Vec<f32>,
    a_w: Vec<bf16>,
    b_w: Vec<bf16>,
    conv_w: Vec<bf16>,
    a_log: Vec<bf16>,
    dt_bias: Vec<bf16>,
    norm_w: Vec<bf16>,
    out_w: Vec<bf16>,
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

fn build_layer(
    cfg: LinearAttentionConfig,
    w: &SharedGdnWeights,
    device: &Device,
) -> LinearAttention {
    let h = cfg.hidden_size;
    let conv_dim = cfg.conv_dim();
    let value_dim = cfg.value_dim();
    let n_v = cfg.linear_num_value_heads;
    let t = |v: &[bf16], shape: &[usize]| {
        Tensor::from_vec(v.to_vec(), shape, device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    };
    LinearAttention::new(
        cfg,
        fp8_resident_linear(&w.qkv_bytes, &w.qkv_scales, conv_dim, h, device),
        fp8_resident_linear(&w.z_bytes, &w.z_scales, value_dim, h, device),
        Linear::new(t(&w.a_w, &[n_v, h]), None).unwrap(),
        Linear::new(t(&w.b_w, &[n_v, h]), None).unwrap(),
        t(&w.conv_w, &[conv_dim, 1, cfg.linear_conv_kernel_dim]),
        t(&w.a_log, &[n_v]),
        t(&w.dt_bias, &[n_v]),
        t(&w.norm_w, &[cfg.linear_value_head_dim]),
        Linear::new(t(&w.out_w, &[h, value_dim]), None).unwrap(),
    )
    .unwrap()
}

fn run_shape(name: &str, hidden: usize, n_k: usize, n_v: usize, d_k: usize, d_v: usize) {
    let _env_guard = ENV_LOCK_BECAUSE_NV_Q38_GDN_FUSE_IS_PROCESS_GLOBAL
        .lock()
        .unwrap();
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("[{name}] skip: no CUDA device");
        return;
    };
    let cfg = LinearAttentionConfig {
        hidden_size: hidden,
        linear_num_key_heads: n_k,
        linear_num_value_heads: n_v,
        linear_key_head_dim: d_k,
        linear_value_head_dim: d_v,
        linear_conv_kernel_dim: 4,
        mamba_ssm_dtype: DType::F32,
        rms_eps: 1e-6,
    };
    let conv_dim = cfg.conv_dim();
    let value_dim = cfg.value_dim();

    let mut rng = Lcg(0x2545f4914f6cdd1d);
    let qkv_host = rng.bf16s(conv_dim * hidden, 0.05);
    let z_host = rng.bf16s(value_dim * hidden, 0.05);
    let (qkv_bytes, qkv_scales) =
        nv_quant::fp8::quantize_e4m3_per_row(&qkv_host, conv_dim, hidden).unwrap();
    let (z_bytes, z_scales) =
        nv_quant::fp8::quantize_e4m3_per_row(&z_host, value_dim, hidden).unwrap();
    let w = SharedGdnWeights {
        qkv_bytes,
        qkv_scales,
        z_bytes,
        z_scales,
        a_w: rng.bf16s(n_v * hidden, 0.05),
        b_w: rng.bf16s(n_v * hidden, 0.05),
        conv_w: rng.bf16s(conv_dim * cfg.linear_conv_kernel_dim, 0.3),
        a_log: rng.bf16s(n_v, 0.5),
        dt_bias: rng.bf16s(n_v, 0.5),
        norm_w: rng.bf16s(d_v, 1.0),
        out_w: rng.bf16s(hidden * value_dim, 0.05),
    };

    let la_ref = build_layer(cfg, &w, &device);
    let mut la_new = build_layer(cfg, &w, &device);
    let mut concat_bytes = w.qkv_bytes.clone();
    concat_bytes.extend_from_slice(&w.z_bytes);
    let mut concat_scales = w.qkv_scales.clone();
    concat_scales.extend_from_slice(&w.z_scales);
    la_new
        .install_qkvz_concat_fp8_decode_arm(&concat_bytes, &concat_scales, &device)
        .unwrap();

    std::env::set_var("NV_Q38_GDN_FUSE", "1");
    assert!(
        nv_layers::linear_attn::gdn_fuse_env_read_per_call_so_the_kill_switch_works_mid_process()
    );

    let st_ref = la_ref.new_fused_state(&device).unwrap();
    let st_new = la_new.new_fused_state(&device).unwrap();

    for step in 0..DECODE_STEPS_16_ENOUGH_FOR_STATE_DRIFT_TO_COMPOUND {
        let x_host = rng.bf16s(hidden, 0.5);
        let x = Tensor::from_vec(x_host, (1usize, 1usize, hidden), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let out_ref = la_ref
            .forward_decode_fused(&x, &st_ref)
            .unwrap()
            .expect("reference fused decode refused");
        let out_new = la_new
            .forward_decode_fused(&x, &st_new)
            .unwrap()
            .expect("qkvz-concat fused decode refused");
        if let Device::Cuda(d) = &device {
            d.cuda_stream().synchronize().unwrap();
        }
        let conv_ref = host(st_ref.conv_state());
        let conv_new = host(st_new.conv_state());
        assert_eq!(
            conv_ref, conv_new,
            "[{name}] step {step}: conv state must be bitwise identical because the conv \
             epilogue replays the exact reference conv math"
        );
        let err = rel_l2(&host(&out_new), &host(&out_ref));
        eprintln!("[{name}] step {step} rel_l2(concat-fused vs 7-kernel fused) = {err:.3e}");
        assert!(
            err < REL_L2_BOUND_2E2_MATCHES_GDN_FUSED_DECODE_HOUSE_TOLERANCE,
            "[{name}] diverged at step {step}: {err}"
        );
    }
    let state_err = rel_l2(&host(st_new.recurrent_state()), &host(st_ref.recurrent_state()));
    eprintln!("[{name}] final recurrent state rel_l2 = {state_err:.3e}");
    assert!(
        state_err < REL_L2_BOUND_2E2_MATCHES_GDN_FUSED_DECODE_HOUSE_TOLERANCE,
        "[{name}] recurrent state drifted: {state_err}"
    );

    std::env::set_var("NV_Q38_GDN_FUSE", "0");
    let out_killed = la_new
        .forward_decode_fused(
            &Tensor::from_vec(rng.bf16s(hidden, 0.5), (1usize, 1usize, hidden), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
            &st_new,
        )
        .unwrap()
        .expect("kill switch must fall back to the 7-kernel fused path, not refuse");
    let _ = host(&out_killed);
    std::env::set_var("NV_Q38_GDN_FUSE", "1");
}

#[test]
fn gdn_fuse_chain_parity_qwen38_group_ratio_shape() {
    run_shape("vpk3", 512, 4, 12, 128, 128);
}

#[test]
fn gdn_fuse_chain_parity_tiny_shape() {
    run_shape("tiny", 256, 2, 4, 32, 32);
}
