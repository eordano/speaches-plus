#![cfg(feature = "cuda")]

mod common;
use common::build_layer;
use common::host;
use common::randn;
use common::rel_l2;
use candle_core::{DType, Device, Tensor};
use nv_layers::linear::Linear;
use nv_layers::linear_attn::{LinearAttention, LinearAttentionConfig};

fn run_shape_with_split_env_on(name: &str, n_k: usize, n_v: usize, d_k: usize, d_v: usize) {
    std::env::set_var("NV_Q38_GDN_SPLIT", "1");
    let Ok(dev) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let cfg = LinearAttentionConfig {
        hidden_size: 256,
        linear_num_key_heads: n_k,
        linear_num_value_heads: n_v,
        linear_key_head_dim: d_k,
        linear_value_head_dim: d_v,
        linear_conv_kernel_dim: 4,
        mamba_ssm_dtype: DType::F32,
        rms_eps: 1e-6,
    };
    let la = build_layer(cfg, &dev);
    assert!(la.fused_decode_supported(), "[{name}] fused unsupported");

    let prefill = randn(&[1, 5, cfg.hidden_size], &dev);
    let steps: Vec<Tensor> = (0..3)
        .map(|_| randn(&[1, 1, cfg.hidden_size], &dev))
        .collect();

    let mut ref_state = None;
    let _ = la.forward_with_state(&prefill, &mut ref_state).unwrap();
    let mut ref_outs = Vec::new();
    for s in &steps {
        ref_outs.push(host(&la.forward_with_state(s, &mut ref_state).unwrap()));
    }

    let mut seed_state = None;
    let _ = la.forward_with_state(&prefill, &mut seed_state).unwrap();
    let fused = la.new_fused_state(&dev).unwrap();
    fused.copy_data_from(seed_state.as_ref().unwrap()).unwrap();
    let mut fused_outs = Vec::new();
    for s in &steps {
        let out = la
            .forward_decode_fused(s, &fused)
            .unwrap()
            .expect("fused path not taken");
        fused_outs.push(host(&out));
    }
    if let Device::Cuda(d) = &dev {
        d.cuda_stream().synchronize().unwrap();
    }

    for (i, (r, f)) in ref_outs.iter().zip(fused_outs.iter()).enumerate() {
        let err = rel_l2(f, r);
        eprintln!("[{name}] step {i} rel_l2(split-fused vs candle) = {err:.3e}");
        assert!(
            err < 2e-2,
            "[{name}] split fused decode diverged at step {i}: {err}"
        );
    }
}

#[test]
fn gdn_split_fused_decode_tiny_shape() {
    run_shape_with_split_env_on("tiny", 2, 4, 32, 32);
}

#[test]
fn gdn_split_fused_decode_qwen38_shape_v_per_k_3_48v_16k_128() {
    run_shape_with_split_env_on("qwen38", 16, 48, 128, 128);
}

#[test]
fn gdn_split_fused_decode_tiny_v_per_k_3() {
    run_shape_with_split_env_on("tiny_vpk3", 2, 6, 32, 32);
}
