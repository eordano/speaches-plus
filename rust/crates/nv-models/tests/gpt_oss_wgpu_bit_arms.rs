#![cfg(feature = "wgpu")]

mod common;
use common::gow_tiny_weights as tiny_weights;
use nv_models::gpt_oss_wgpu as gow;
use nv_models::gpt_oss_wgpu::{GptOssConfig, GptOssLayerType};

static ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE: std::sync::Mutex<()> =
    std::sync::Mutex::new(());

const SHIPPED_ON_FUSE_STACK_ENVS_PINNED_TO_ZERO_SO_EACH_COMBO_STATES_ITS_WHOLE_ARM: [&str; 7] = [
    gow::MX_SG_XV4_ENV,
    gow::MX_SG_BYTE_LUT_ENV,
    gow::MX_SG_XSTAGE_ENV,
    gow::MX_SG_GUPAIR_ENV,
    gow::MX_SG_GUSW_ENV,
    gow::MOE_DNCOMB_FUSE_ENV,
    gow::MOE_DNCOMB_SGSLOT_ENV,
];

fn pin_the_shipped_on_fuse_stack_off_so_the_baseline_is_the_serial_chain() {
    for env in SHIPPED_ON_FUSE_STACK_ENVS_PINNED_TO_ZERO_SO_EACH_COMBO_STATES_ITS_WHOLE_ARM {
        std::env::set_var(env, "0");
    }
}

fn bit_arm_config() -> GptOssConfig {
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
        yarn_truncate_correction_dims: true,
        tie_word_embeddings: false,
    }
}

fn decode_logit_bits(
    cfg: &GptOssConfig,
    hw: &gow::HostWeights,
    steps: usize,
    mx_entry: &str,
    fuse: bool,
    topk_sg32: bool,
) -> Vec<Vec<u32>> {
    let mut gpu = gow::GptOssWgpu::new(cfg.clone(), hw, 64).expect("build model");
    assert_eq!(
        gpu.mx_gemv_entry(),
        mx_entry,
        "the mx arm under test never engaged, so equality would prove nothing"
    );
    let labels = gpu.decode_pass_labels();
    let count_of = |needle: &str| labels.iter().filter(|l| l.contains(needle)).count();
    assert_eq!(
        count_of("gow_fu_router_topk"),
        if fuse { cfg.num_hidden_layers } else { 0 },
        "the router+topk fuse arm engagement must match its env gate"
    );
    assert_eq!(
        count_of(gow::MOE_TOPK_SG32_ENTRY),
        if topk_sg32 { cfg.num_hidden_layers } else { 0 },
        "the sg32 topk arm engagement must match its env gate"
    );
    assert_eq!(
        count_of(gow::GEMV_SG_XS_ENTRY) > 0,
        std::env::var(gow::GEMV_SG_XSTAGE_ENV).is_ok_and(|v| v == "1"),
        "the bf16 gemv xstage arm engagement must match its env gate"
    );
    let mut tok = gpu.prefill(&[1, 2, 3, 4]).expect("prefill");
    let mut rows = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (next, logits) = gpu.decode_step_logits(tok).expect("decode");
        rows.push(logits.iter().map(|v| v.to_bits()).collect());
        tok = next;
    }
    rows
}

fn decode_logit_bits_plain(
    cfg: &GptOssConfig,
    hw: &gow::HostWeights,
    steps: usize,
    scc_engaged: bool,
) -> Vec<Vec<u32>> {
    let mut gpu = gow::GptOssWgpu::new(cfg.clone(), hw, 64).expect("build model");
    let labels = gpu.decode_pass_labels();
    let r2 = labels
        .iter()
        .filter(|l| l.ends_with(gow::GEMV_SG_R2_ENTRY))
        .count();
    assert_eq!(
        r2 > 0,
        std::env::var(gow::GEMV_SG_R2_ENV).is_ok_and(|v| v == "1"),
        "the dual-row bf16 gemv arm engagement must match its env gate (r2 passes: {r2})"
    );
    let scc = labels
        .iter()
        .filter(|l| l.ends_with(gow::FLASH_STAGE2_SCC_ENTRY))
        .count();
    let plain = labels
        .iter()
        .filter(|l| l.ends_with(":gow_flash_stage2_sink_pk"))
        .count();
    if scc_engaged {
        assert!(
            scc > 0 && plain == 0,
            "the scc stage2 arm under test never engaged (scc={scc} plain={plain}), so equality \
             would prove nothing"
        );
    } else {
        assert!(
            plain > 0 && scc == 0,
            "the baseline must run the recompute stage2 (scc={scc} plain={plain})"
        );
    }
    let mut tok = gpu.prefill(&[1, 2, 3, 4]).expect("prefill");
    let mut rows = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (next, logits) = gpu.decode_step_logits(tok).expect("decode");
        rows.push(logits.iter().map(|v| v.to_bits()).collect());
        tok = next;
    }
    rows
}

#[test]
fn flash_stage2_scc_and_chunked_submit_arms_are_bitwise_identical_to_the_default_stack() {
    let _serial = ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap();
    if nv_kernels::wgpu_backend::WgpuContext::shared().is_err() {
        eprintln!("[skip] no wgpu adapter");
        return;
    }
    let cfg = bit_arm_config();
    let hw = tiny_weights(&cfg, 0x6055_0012);
    let clear = || {
        std::env::remove_var(gow::FLASH_STAGE2_SCC_ENV);
        std::env::remove_var(gow::GEMV_SG_R2_ENV);
        std::env::remove_var(nv_kernels::wgpu_backend::dispatch::SUBMIT_CHUNK_ENV);
    };
    clear();
    let base = decode_logit_bits_plain(&cfg, &hw, 24, false);
    assert!(
        base.iter().flatten().any(|b| *b != base[0][0]),
        "degenerate logits cannot prove arm parity"
    );

    std::env::set_var(gow::FLASH_STAGE2_SCC_ENV, "1");
    let scc = decode_logit_bits_plain(&cfg, &hw, 24, true);
    clear();
    assert_eq!(
        base, scc,
        "caching the deterministic per-split scales must reproduce the recompute stage2 \
         bit-for-bit (identical fma chain order over identical scale values is the whole claim)"
    );

    std::env::set_var(nv_kernels::wgpu_backend::dispatch::SUBMIT_CHUNK_ENV, "13");
    let chunked = decode_logit_bits_plain(&cfg, &hw, 24, false);
    clear();
    assert_eq!(
        base, chunked,
        "splitting the per-step pass list into chunked submits reorders nothing on the queue, \
         so the logits must be bit-for-bit identical"
    );

    std::env::set_var(gow::GEMV_SG_R2_ENV, "1");
    let r2 = decode_logit_bits_plain(&cfg, &hw, 24, false);
    clear();
    assert_eq!(
        base, r2,
        "the dual-row bf16 gemv keeps each row's lane-strided fma chain and butterfly, so its \
         logits must be bit-for-bit identical"
    );

    std::env::set_var(gow::FLASH_STAGE2_SCC_ENV, "1");
    std::env::set_var(gow::GEMV_SG_R2_ENV, "1");
    std::env::set_var(nv_kernels::wgpu_backend::dispatch::SUBMIT_CHUNK_ENV, "13");
    let both = decode_logit_bits_plain(&cfg, &hw, 24, true);
    clear();
    assert_eq!(
        base, both,
        "scc + r2 + chunked submits must compose bit-for-bit"
    );
}

#[test]
fn byte_lut_and_router_topk_fuse_arms_are_bitwise_identical_to_the_default_stack() {
    let _serial = ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap();
    let ctx = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            return;
        }
    };
    if !nv_kernels::wgpu_backend::kernels::gemv_nvfp4::sg32_ok(ctx) {
        eprintln!("[skip] subgroup width is not uniformly 32; neither arm engages here");
        return;
    }

    let cfg = bit_arm_config();
    let hw = tiny_weights(&cfg, 0x6055_0011);
    let clear_arm_envs = || {
        pin_the_shipped_on_fuse_stack_off_so_the_baseline_is_the_serial_chain();
        std::env::remove_var(gow::MX_SG_XV4_R2_ENV);
        std::env::remove_var(gow::GEMV_SG_XSTAGE_ENV);
        std::env::remove_var(gow::MOE_ROUTER_TOPK_FUSE_ENV);
        std::env::remove_var(gow::MOE_TOPK_SG32_ENV);
    };
    clear_arm_envs();
    let base = decode_logit_bits(&cfg, &hw, 24, gow::MX_SG_ENTRY, false, false);
    assert!(
        base.iter().flatten().any(|b| *b != base[0][0]),
        "degenerate logits cannot prove arm parity"
    );

    let lut = gow::MX_SG_BYTE_LUT_ENV;
    let xv4 = gow::MX_SG_XV4_ENV;
    let r2 = gow::MX_SG_XV4_R2_ENV;
    let xs = gow::MX_SG_XSTAGE_ENV;
    let gxs = gow::GEMV_SG_XSTAGE_ENV;
    let gup = gow::MX_SG_GUPAIR_ENV;
    let combos: [(&[&str], &str, bool, bool); 12] = [
        (
            &[xv4, lut, xs],
            gow::MX_SG_XV4_LUT_XS_ENTRY,
            false,
            true,
        ),
        (
            &[xv4, lut, xs, gup],
            gow::MX_SG_XV4_R2_LUT_XS_ENTRY,
            false,
            true,
        ),
        (
            &[xv4, lut, xs, gup, gxs],
            gow::MX_SG_XV4_R2_LUT_XS_ENTRY,
            false,
            true,
        ),
        (
            &[xv4, lut, xs, gxs],
            gow::MX_SG_XV4_LUT_XS_ENTRY,
            false,
            true,
        ),
        (&[lut], gow::MX_SG_LUT_ENTRY, false, false),
        (&[xv4], gow::MX_SG_XV4_ENTRY, false, false),
        (&[r2], gow::MX_SG_XV4_R2_ENTRY, false, false),
        (&[], gow::MX_SG_ENTRY, true, false),
        (&[], gow::MX_SG_ENTRY, false, true),
        (&[r2], gow::MX_SG_XV4_R2_ENTRY, false, true),
        (&[xv4, lut], gow::MX_SG_XV4_LUT_ENTRY, false, false),
        (&[r2, lut], gow::MX_SG_XV4_R2_LUT_ENTRY, false, true),
    ];
    for (mx_envs, mx_entry, fuse, topk_sg32) in combos {
        for env in mx_envs {
            std::env::set_var(env, "1");
        }
        if fuse {
            std::env::set_var(gow::MOE_ROUTER_TOPK_FUSE_ENV, "1");
        }
        if topk_sg32 {
            std::env::set_var(gow::MOE_TOPK_SG32_ENV, "1");
        }
        let armed = decode_logit_bits(&cfg, &hw, 24, mx_entry, fuse, topk_sg32);
        clear_arm_envs();
        assert_eq!(
            base, armed,
            "mx_entry={mx_entry} fuse={fuse} topk_sg32={topk_sg32} must reproduce the default \
             stack's logits bit-for-bit (identical fma chain order and identical finite-logit \
             topk selection are the whole claim of every arm here)"
        );
    }
}

#[test]
fn int8_lmhead_arm_stays_in_the_bf16_head_tolerance_band_and_caps_the_verify_epilogue() {
    let _serial = ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap();
    let ctx = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            return;
        }
    };
    if !nv_kernels::wgpu_backend::kernels::gemv_nvfp4::sg32_ok(ctx) {
        eprintln!("[skip] subgroup width is not uniformly 32; the sg head kernels do not engage");
        return;
    }
    let cfg = bit_arm_config();
    let hw = tiny_weights(&cfg, 0x6055_0013);
    std::env::remove_var(gow::LMHEAD_I8_ENV);

    let mut base = gow::GptOssWgpu::new(cfg.clone(), &hw, 64).expect("build base");
    assert!(
        !base
            .decode_pass_labels()
            .iter()
            .any(|l| l.contains(gow::LMHEAD_I8_ENTRY)),
        "zero-env decode must keep the bf16 head"
    );
    assert!(
        base.verify_max_rows() > 0,
        "the bf16 head keeps the verify epilogue"
    );

    std::env::set_var(gow::LMHEAD_I8_ENV, "1");
    let mut armed = gow::GptOssWgpu::new(cfg.clone(), &hw, 64).expect("build i8");
    std::env::remove_var(gow::LMHEAD_I8_ENV);
    assert!(
        armed
            .decode_pass_labels()
            .iter()
            .any(|l| l.contains(gow::LMHEAD_I8_ENTRY)),
        "the int8 head arm under test never engaged, so closeness would prove nothing"
    );
    assert_eq!(
        armed.verify_max_rows(),
        0,
        "an int8 decode head against a bf16 verify head cannot keep the bit-identity contract, \
         so the arm must cap the verify epilogue to zero rows"
    );

    let mut tok_b = base.prefill(&[1, 2, 3, 4]).expect("prefill base");
    let mut tok_a = armed.prefill(&[1, 2, 3, 4]).expect("prefill i8");
    let mut spread = 0.0f32;
    for step in 0..24 {
        let (nb, lb) = base.decode_step_logits(tok_b).expect("base step");
        let (na, la) = armed.decode_step_logits(tok_a).expect("i8 step");
        let mut max_abs = 0.0f32;
        let mut max_mag = 0.0f32;
        for (x, y) in lb.iter().zip(la.iter()) {
            max_abs = max_abs.max((x - y).abs());
            max_mag = max_mag.max(x.abs());
            spread = spread.max((x - lb[0]).abs());
        }
        assert!(
            max_abs <= 0.05 * max_mag.max(1e-3),
            "step {step}: int8 head drifted beyond the 5% band the CPU-reference suite grants \
             the bf16 head (max_abs {max_abs}, max_mag {max_mag})"
        );
        tok_b = nb;
        tok_a = na;
    }
    assert!(
        spread > 1e-3,
        "degenerate logits cannot prove head closeness (spread {spread})"
    );
}

#[test]
fn kpre_and_rmsres_sglat_arms_are_bitwise_identical_to_the_record_shapes_and_must_engage() {
    let _serial = ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap();
    let ctx = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            return;
        }
    };
    if !nv_kernels::wgpu_backend::kernels::gemv_nvfp4::sg32_ok(ctx) {
        eprintln!("[skip] subgroup width is not uniformly 32; the sg arms do not engage");
        return;
    }
    let cfg = bit_arm_config();
    let hw = tiny_weights(&cfg, 0x6055_0015);
    let clear = || {
        pin_the_shipped_on_fuse_stack_off_so_the_baseline_is_the_serial_chain();
        for env in [
            gow::GEMV_SG_KPRE_ENV,
            gow::MX_KPRE_ENV,
            gow::RMSRES_SGLAT_ENV,
        ] {
            std::env::remove_var(env);
        }
    };
    clear();

    let run = |gemv_kpre: bool, mx_kpre: bool, rmsres_sglat: bool| {
        std::env::set_var(gow::MX_SG_XV4_ENV, "1");
        std::env::set_var(gow::MX_SG_BYTE_LUT_ENV, "1");
        std::env::set_var(gow::MX_SG_XSTAGE_ENV, "1");
        std::env::set_var(gow::MX_SG_GUPAIR_ENV, "1");
        std::env::set_var(gow::MX_SG_GUSW_ENV, "1");
        if gemv_kpre {
            std::env::set_var(gow::GEMV_SG_KPRE_ENV, "1");
        }
        if mx_kpre {
            std::env::set_var(gow::MX_KPRE_ENV, "1");
        }
        if rmsres_sglat {
            std::env::set_var(gow::RMSRES_SGLAT_ENV, "1");
        }
        let mut gpu = gow::GptOssWgpu::new(cfg.clone(), &hw, 64).expect("build model");
        clear();
        let labels = gpu.decode_pass_labels();
        let bf16_kpre = labels
            .iter()
            .filter(|l| l.contains(":gow_fu_gemv_bf16_sg_kpre"))
            .count();
        assert_eq!(
            bf16_kpre > 0,
            gemv_kpre,
            "the baked-k bf16 gemv arm engagement must match its env gate (kpre passes: \
             {bf16_kpre})"
        );
        let dn_kpre = labels
            .iter()
            .filter(|l| l.contains("_lut_xs_kpre"))
            .count();
        let gu_kpre = labels
            .iter()
            .filter(|l| l.contains("_swiglu_kpre"))
            .count();
        assert_eq!(
            dn_kpre > 0 && gu_kpre == cfg.num_hidden_layers,
            mx_kpre,
            "the baked-block mx gemv arms' engagement must match their env gate (dn kpre: \
             {dn_kpre}, gusw kpre: {gu_kpre})"
        );
        let sglat = labels
            .iter()
            .filter(|l| l.ends_with(gow::RMSRES_SGLAT_ENTRY))
            .count();
        assert_eq!(
            sglat,
            if rmsres_sglat {
                2 * cfg.num_hidden_layers - 1
            } else {
                0
            },
            "the rmsres sglat arm must cover exactly the per-layer post and next norms when \
             gated on and none when off"
        );
        let mut tok = gpu.prefill(&[1, 2, 3, 4]).expect("prefill");
        let mut rows: Vec<Vec<u32>> = Vec::new();
        for _ in 0..24 {
            let (next, logits) = gpu.decode_step_logits(tok).expect("decode");
            rows.push(logits.iter().map(|v| v.to_bits()).collect());
            tok = next;
        }
        rows
    };

    let base = run(false, false, false);
    assert!(
        base.iter().flatten().any(|b| *b != base[0][0]),
        "degenerate logits cannot prove arm parity"
    );
    let gemv = run(true, false, false);
    assert_eq!(
        base, gemv,
        "issuing every weight load before the fma chain keeps each lane's add order, so the \
         logits must be bit-for-bit identical"
    );
    let mx = run(false, true, false);
    assert_eq!(
        base, mx,
        "issuing the expert weight and scale streams before the fma chains keeps each \
         accumulator's kb order, so the logits must be bit-for-bit identical"
    );
    let sglat = run(false, false, true);
    assert_eq!(
        base, sglat,
        "the shuffle collapse reproduces the shared scratch tree's association exactly, so the \
         rms bits and the logits must be bit-for-bit identical"
    );
    let composed = run(true, true, true);
    assert_eq!(
        base, composed,
        "kpre on both gemv families and the rmsres shuffle collapse must compose bit-for-bit"
    );
}

#[test]
fn fused_gu_swiglu_arm_is_bitwise_identical_to_the_gemv_then_swiglu_chain() {
    let _serial = ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap();
    let ctx = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            return;
        }
    };
    if !nv_kernels::wgpu_backend::kernels::gemv_nvfp4::sg32_ok(ctx) {
        eprintln!("[skip] subgroup width is not uniformly 32; the xs mx arms do not engage");
        return;
    }
    let cfg = bit_arm_config();
    let hw = tiny_weights(&cfg, 0x6055_0014);
    let clear = pin_the_shipped_on_fuse_stack_off_so_the_baseline_is_the_serial_chain;
    clear();

    let run = |gusw: bool, gupair: bool| {
        std::env::set_var(gow::MX_SG_XV4_ENV, "1");
        std::env::set_var(gow::MX_SG_BYTE_LUT_ENV, "1");
        std::env::set_var(gow::MX_SG_XSTAGE_ENV, "1");
        if gupair {
            std::env::set_var(gow::MX_SG_GUPAIR_ENV, "1");
        }
        if gusw {
            std::env::set_var(gow::MX_SG_GUSW_ENV, "1");
        }
        let mut gpu = gow::GptOssWgpu::new(cfg.clone(), &hw, 64).expect("build model");
        clear();
        let labels = gpu.decode_pass_labels();
        let fused = labels
            .iter()
            .filter(|l| l.ends_with(gow::MX_SG_XV4_R4_LUT_XS_SWIGLU_ENTRY))
            .count();
        let sw = labels
            .iter()
            .filter(|l| l.ends_with(":gow_swiglu"))
            .count();
        if gusw {
            assert!(
                fused == cfg.num_hidden_layers && sw == 0,
                "the fused gu+swiglu arm under test never engaged (fused={fused} swiglu={sw}), \
                 so equality would prove nothing"
            );
        } else {
            assert!(
                fused == 0 && sw == cfg.num_hidden_layers,
                "the baseline must run the two-pass gemv-then-swiglu chain (fused={fused} \
                 swiglu={sw})"
            );
        }
        let mut tok = gpu.prefill(&[1, 2, 3, 4]).expect("prefill");
        let mut rows: Vec<Vec<u32>> = Vec::new();
        for _ in 0..24 {
            let (next, logits) = gpu.decode_step_logits(tok).expect("decode");
            rows.push(logits.iter().map(|v| v.to_bits()).collect());
            tok = next;
        }
        rows
    };

    let base = run(false, false);
    assert!(
        base.iter().flatten().any(|b| *b != base[0][0]),
        "degenerate logits cannot prove arm parity"
    );
    let fused = run(true, false);
    assert_eq!(
        base, fused,
        "writing the swiglu act word from the gu subgroup keeps each row's fma chain, bias add, \
         and the exact swiglu formula, so the logits must be bit-for-bit identical"
    );
    let composed = run(true, true);
    assert_eq!(
        base, composed,
        "gusw on the gu site and the r2 xs pair on the dn site must compose bit-for-bit"
    );
}

#[test]
fn dncomb_and_sgslot_arms_are_bitwise_identical_to_the_split_down_then_combine_chain() {
    let _serial = ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap();
    let ctx = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            return;
        }
    };
    if !nv_kernels::wgpu_backend::kernels::gemv_nvfp4::sg32_ok(ctx) {
        eprintln!("[skip] subgroup width is not uniformly 32; the xs mx arms do not engage");
        return;
    }
    let mut cfg = bit_arm_config();
    cfg.num_experts_per_tok = 4;
    let hw = tiny_weights(&cfg, 0x6055_0015);
    let clear = pin_the_shipped_on_fuse_stack_off_so_the_baseline_is_the_serial_chain;
    clear();

    let run = |dncomb: bool, sgslot: bool| {
        std::env::set_var(gow::MX_SG_XV4_ENV, "1");
        std::env::set_var(gow::MX_SG_BYTE_LUT_ENV, "1");
        std::env::set_var(gow::MX_SG_XSTAGE_ENV, "1");
        std::env::set_var(gow::MX_SG_GUPAIR_ENV, "1");
        std::env::set_var(gow::MX_SG_GUSW_ENV, "1");
        if dncomb {
            std::env::set_var(gow::MOE_DNCOMB_FUSE_ENV, "1");
        }
        if sgslot {
            std::env::set_var(gow::MOE_DNCOMB_SGSLOT_ENV, "1");
        }
        let mut gpu = gow::GptOssWgpu::new(cfg.clone(), &hw, 64).expect("build model");
        clear();
        let labels = gpu.decode_pass_labels();
        let fused = labels
            .iter()
            .filter(|l| l.contains("_dncomb"))
            .count();
        let combine = labels
            .iter()
            .filter(|l| l.ends_with(":gow_moe_combine"))
            .count();
        if dncomb {
            assert!(
                fused == cfg.num_hidden_layers && combine == 0,
                "the dncomb arm under test never engaged (dncomb={fused} combine={combine}); \
                 equality against a baseline that ran the same split chain proves nothing"
            );
        } else {
            assert!(
                fused == 0 && combine == cfg.num_hidden_layers,
                "the baseline must run the split down-gemv-then-combine chain \
                 (dncomb={fused} combine={combine})"
            );
        }
        let mut tok = gpu.prefill(&[1, 2, 3, 4]).expect("prefill");
        let mut rows: Vec<Vec<u32>> = Vec::new();
        for _ in 0..24 {
            let (next, logits) = gpu.decode_step_logits(tok).expect("decode");
            rows.push(logits.iter().map(|v| v.to_bits()).collect());
            tok = next;
        }
        rows
    };

    let base = run(false, false);
    assert!(
        base.iter().flatten().any(|b| *b != base[0][0]),
        "degenerate logits cannot prove arm parity"
    );
    let dncomb = run(true, false);
    assert_eq!(
        base, dncomb,
        "dncomb keeps each expert row in a register and folds the same ascending fma chain \
         gow_moe_combine would have run, so the logits must be bit-for-bit identical"
    );
    let sgslot = run(true, true);
    assert_eq!(
        base, sgslot,
        "the sgslot variant remaps rows onto subgroup slots without reordering any \
         accumulation, so it must stay bit-for-bit identical too"
    );
}

#[test]
fn fused_qkv_rope_rows_arm_is_bitwise_identical_to_the_kv8_split_chain_and_drops_four_dispatches_per_full_layer(
) {
    let _serial = ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap();
    if nv_kernels::wgpu_backend::WgpuContext::shared().is_err() {
        eprintln!("[skip] no wgpu adapter");
        return;
    }
    let mut cfg = bit_arm_config();
    cfg.head_dim = 32;
    let hw = tiny_weights(&cfg, 0x6055_0017);
    let clear = || {
        std::env::remove_var(gow::QKV_FUSE_KV8_ENV);
        std::env::remove_var(gow::KV_FP8_ENV);
    };
    clear();
    std::env::set_var(gow::KV_FP8_ENV, "1");

    let mut base = gow::GptOssWgpu::new(cfg.clone(), &hw, 64).expect("build split-chain base");
    let base_labels = base.decode_pass_labels();
    assert!(
        base_labels.iter().any(|l| l.contains("gow-at-kproj")),
        "the kv8 layer must run the split chain when the fuse arm is off, or this test \
         compares fused against fused and proves nothing"
    );
    let base_passes = base_labels.len();
    let mut tok = base.prefill(&[1, 2, 3, 4]).expect("base prefill");
    let mut base_rows: Vec<Vec<u32>> = Vec::new();
    for _ in 0..24 {
        let (next, logits) = base.decode_step_logits(tok).expect("base decode");
        base_rows.push(logits.iter().map(|v| v.to_bits()).collect());
        tok = next;
    }
    assert!(
        base_rows.iter().flatten().any(|b| *b != base_rows[0][0]),
        "degenerate logits cannot prove arm parity"
    );

    std::env::set_var(gow::QKV_FUSE_KV8_ENV, "1");
    let mut armed = gow::GptOssWgpu::new(cfg.clone(), &hw, 64).expect("build fused-rows arm");
    clear();
    let armed_labels = armed.decode_pass_labels();
    assert!(
        armed_labels
            .iter()
            .any(|l| l.contains("gow-at-furopekvrows")),
        "the fused-rows arm never engaged, so equality would prove nothing"
    );
    assert!(
        !armed_labels.iter().any(|l| l.contains("gow-at-kproj")),
        "an engaged fused-rows arm must not also run the split k projection"
    );
    let full_layers = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, GptOssLayerType::Full))
        .count();
    assert_eq!(
        base_passes - armed_labels.len(),
        4 * full_layers,
        "the arm replaces 3 gemvs + 2 ropes + 1 kv_write with 1 gemv + 1 rope on every \
         kv8 layer, so the decode pass count must drop by exactly four per full layer"
    );
    let mut tok = armed.prefill(&[1, 2, 3, 4]).expect("armed prefill");
    let mut armed_rows: Vec<Vec<u32>> = Vec::new();
    for _ in 0..24 {
        let (next, logits) = armed.decode_step_logits(tok).expect("armed decode");
        armed_rows.push(logits.iter().map(|v| v.to_bits()).collect());
        tok = next;
    }
    assert_eq!(
        base_rows, armed_rows,
        "the fused qkv gemv computes the same per-row dot in the same order as the three \
         split gemvs and the rope row writer applies the same gfu_rot math as gow_rope, \
         so the arm must be bit-for-bit identical to the split chain"
    );
}
