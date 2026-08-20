#![cfg(feature = "wgpu")]

mod common;
use common::gow_tiny_weights as tiny_weights;
use common::rel_err;
use nv_models::gpt_oss_wgpu as gow;
use nv_models::gpt_oss_wgpu::{GptOssConfig, GptOssLayerType};

fn reorder_config() -> GptOssConfig {
    GptOssConfig {
        hidden_size: 2880,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        intermediate_size: 2880,
        num_local_experts: 4,
        num_experts_per_tok: 2,
        vocab_size: 64,
        max_position_embeddings: 64,
        sliding_window: 4,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        swiglu_limit: 7.0,
        layer_types: vec![GptOssLayerType::Sliding, GptOssLayerType::Full],
        yarn_factor: 4.0,
        yarn_beta_fast: 32.0,
        yarn_beta_slow: 1.0,
        yarn_original_max: 16,
        tie_word_embeddings: false,
    }
}

#[test]
fn mx_sg_arm_matches_the_cpu_reference_and_the_scalar_arm() {
    let ctx = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("[wgpu] adapter: {}", ctx.info.name);
            ctx
        }
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            return;
        }
    };
    if !nv_kernels::wgpu_backend::kernels::gemv_nvfp4::sg32_ok(ctx) {
        eprintln!("[skip] subgroup width is not uniformly 32; the mx sg arm never engages here");
        return;
    }

    let cfg = reorder_config();
    let two_sg_passes = 64;
    assert!(
        cfg.hidden_size / 32 > two_sg_passes && cfg.intermediate_size / 32 > two_sg_passes,
        "both mx GEMVs must carry more k-blocks than two subgroup passes: a lane chaining \
         only two blocks forms the same pair the scalar tree forms, so nothing under 65 \
         blocks reorders the sum and the property under test never engages (gpt-oss-20b \
         itself carries 90)"
    );
    let hw = tiny_weights(&cfg, 0x6055_0007);

    std::env::remove_var(gow::MX_SG_GATE_ENV);
    let mut scalar = gow::GptOssWgpu::new(cfg.clone(), &hw, 32).expect("build scalar-arm model");
    assert_eq!(
        scalar.mx_gemv_entry(),
        gow::MX_SCALAR_ENTRY,
        "with {} unset the audited scalar arm must stay the default",
        gow::MX_SG_GATE_ENV
    );

    std::env::set_var(gow::MX_SG_GATE_ENV, "1");
    let sg_build = gow::GptOssWgpu::new(cfg.clone(), &hw, 32);
    std::env::remove_var(gow::MX_SG_GATE_ENV);
    let mut sg = sg_build.expect("build sg-arm model");
    assert_eq!(
        sg.mx_gemv_entry(),
        gow::MX_SG_ENTRY,
        "{}=1 on a 32-wide-subgroup adapter must select the sg arm, or this test proves nothing",
        gow::MX_SG_GATE_ENV
    );

    let mut st = gow::RefState::new(&cfg);
    let tokens: [u32; 7] = [3, 11, 5, 40, 2, 19, 33];
    for (i, t) in tokens.iter().enumerate() {
        let (a_scalar, l_scalar) = scalar.decode_step_logits(*t).expect("scalar-arm step");
        let (a_sg, l_sg) = sg.decode_step_logits(*t).expect("sg-arm step");
        let want = gow::reference_step(&cfg, &hw, &mut st, *t).expect("reference step");
        let (_, rel_ref) = rel_err(&l_sg, &want);
        let (abs_arm, rel_arm) = rel_err(&l_sg, &l_scalar);
        eprintln!(
            "step {i}: tok={t} sg_argmax={a_sg} scalar_argmax={a_scalar} \
             rel_vs_ref={rel_ref:.6} rel_vs_scalar={rel_arm:e} abs_vs_scalar={abs_arm:e}"
        );
        assert!(
            rel_ref < 0.05,
            "step {i}: sg-arm logits diverged from the CPU reference (rel {rel_ref})"
        );
        assert!(
            rel_arm < 2e-2,
            "step {i}: sg arm differs from the scalar arm beyond what bf16 requantization \
             of ULP-level summation-reorder noise can amplify to (rel {rel_arm}, abs {abs_arm})"
        );
        assert_eq!(
            a_sg, a_scalar,
            "step {i}: the two mx arms disagree on the argmax token"
        );
    }

    let yd_scalar = scalar.debug_probe("ydown0").expect("scalar ydown0 probe");
    let yd_sg = sg.debug_probe("ydown0").expect("sg ydown0 probe");
    let (abs_yd, rel_yd) = rel_err(&yd_sg, &yd_scalar);
    eprintln!("ydown0 raw f32 cross-arm: abs={abs_yd:e} rel={rel_yd:e}");
    assert!(
        rel_yd < 2e-2,
        "raw f32 down-GEMV outputs diverged across arms beyond what upstream bf16 \
         requantization boundaries plus summation reorder can produce (rel {rel_yd})"
    );
    assert!(
        abs_yd > 0.0,
        "the raw f32 down-GEMV outputs are bit-identical across arms; a reordered subgroup \
         sum over >32 blocks cannot round identically on every element, so the sg arm never \
         actually ran and every comparison above was scalar-vs-scalar"
    );

    let steps_past_window = tokens.len() > cfg.sliding_window;
    assert!(
        steps_past_window,
        "test must decode past the sliding window ({} steps <= window {})",
        tokens.len(),
        cfg.sliding_window
    );
}
