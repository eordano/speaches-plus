#![cfg(feature = "wgpu")]

mod common;
use common::bf16_lin_gow_bias as bf16_lin;
use common::have_gpu;
use common::mx_stack;
use common::tiny_config_gpt_oss;
use common::LcgSplitMix64TwoSided as Lcg;

use nv_models::gpt_oss_wgpu as gow;
use nv_models::gpt_oss_wgpu::{GptOssConfig, GptOssLayerType};

const MAX_SEQ: usize = 48;

const FP8_BAND_5PCT_MATCHES_THE_TINY_REFERENCE_GATE: f32 = 0.05;

struct Kv8EnvGuard {
    _serialized: std::sync::MutexGuard<'static, ()>,
}

fn kv8_env_on_serialized_because_build_reads_the_env() -> Kv8EnvGuard {
    static ONE_ENV_AT_A_TIME: std::sync::OnceLock<std::sync::Mutex<()>> =
        std::sync::OnceLock::new();
    let g = ONE_ENV_AT_A_TIME
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    std::env::set_var(gow::KV_FP8_ENV, "1");
    Kv8EnvGuard { _serialized: g }
}

impl Drop for Kv8EnvGuard {
    fn drop(&mut self) {
        std::env::remove_var(gow::KV_FP8_ENV);
    }
}

fn hd32_config_because_the_shared_fold_strides_head_dim_in_32_lane_strips() -> GptOssConfig {
    let cfg = GptOssConfig {
        head_dim: 32,
        ..tiny_config_gpt_oss()
    };
    assert!(
        cfg.layer_types.contains(&GptOssLayerType::Full),
        "the kv8 arm rides full-attention layers only; a config without one tests nothing"
    );
    cfg
}

fn norm_vec(r: &mut Lcg, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| half::bf16::from_f32(1.0 + r.next_f32() * 0.1).to_bits())
        .collect()
}

fn tiny_weights(cfg: &GptOssConfig, seed: u64) -> gow::HostWeights {
    let mut r = Lcg::new(seed);
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let layers = (0..cfg.num_hidden_layers)
        .map(|_| gow::HostLayer {
            input_ln: norm_vec(&mut r, h),
            post_attn_ln: norm_vec(&mut r, h),
            attn: gow::HostAttn {
                q: bf16_lin(&mut r, cfg.num_attention_heads * hd, h, 0.2, true),
                k: bf16_lin(&mut r, cfg.num_key_value_heads * hd, h, 0.2, true),
                v: bf16_lin(&mut r, cfg.num_key_value_heads * hd, h, 0.2, true),
                o: bf16_lin(&mut r, h, cfg.num_attention_heads * hd, 0.2, true),
                sinks: (0..cfg.num_attention_heads)
                    .map(|_| 1.0 + r.next_f32() * 0.5)
                    .collect(),
            },
            moe: gow::HostMoe {
                router: bf16_lin(&mut r, cfg.num_local_experts, h, 0.2, true),
                gate_up: mx_stack(
                    &mut r,
                    cfg.num_local_experts,
                    2 * cfg.intermediate_size,
                    h,
                    0.2,
                ),
                down: mx_stack(&mut r, cfg.num_local_experts, h, cfg.intermediate_size, 0.2),
            },
        })
        .collect();
    gow::HostWeights {
        embed: r.bf16_vec(cfg.vocab_size * h, 0.6),
        final_norm: norm_vec(&mut r, h),
        lm_head: r.bf16_vec(cfg.vocab_size * h, 0.2),
        layers,
    }
}

fn rel_err(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut peak = 1e-6f32;
    for (x, y) in a.iter().zip(b.iter()) {
        max_abs = max_abs.max((x - y).abs());
        peak = peak.max(y.abs());
    }
    (max_abs, max_abs / peak)
}

#[test]
fn kv8_fold_source_for_the_real_gptoss_geometry_validates_under_naga_without_a_gpu() {
    use nv_kernels::wgpu_backend::kernels::flash_decode as fd;
    for sg in [true, false] {
        let src = nv_kernels::wgpu_backend::compose(&format!(
            "{}\n{}",
            fd::WGSL,
            fd::fold_stage1_source_sd(64, sg, 8)
        ));
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("kv8 fold hd64 fold8 sg={sg}: wgsl parse failed: {e}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("kv8 fold hd64 fold8 sg={sg}: validation failed: {e:?}"));
        assert!(
            src.contains(&fd::fold_stage1_entry_sd(64, sg, 8)),
            "generated source must define the entry the model dispatches"
        );
    }
}

#[test]
fn kv8_decode_tracks_the_cpu_reference_within_the_fp8_band_and_holds_argmax() {
    if !have_gpu() {
        return;
    }
    let _env = kv8_env_on_serialized_because_build_reads_the_env();
    let cfg = hd32_config_because_the_shared_fold_strides_head_dim_in_32_lane_strips();
    let hw = tiny_weights(&cfg, 0x6055_8801);
    let mut gpu = gow::GptOssWgpu::new(cfg.clone(), &hw, 32).expect("build kv8 wgpu model");
    let mut st = gow::RefState::new(&cfg);
    let tokens: [u32; 7] = [3, 11, 5, 40, 2, 19, 33];
    assert!(
        tokens.len() > cfg.sliding_window,
        "must decode past the sliding window so full layers read a longer range than window ones"
    );
    let mut argmax_hits = 0usize;
    for (i, t) in tokens.iter().enumerate() {
        let (arg, logits) = gpu.decode_step_logits(*t).expect("kv8 decode step");
        let want = gow::reference_step(&cfg, &hw, &mut st, *t).expect("reference step");
        let (abs, rel) = rel_err(&logits, &want);
        let ref_arg = want
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        if arg == ref_arg {
            argmax_hits += 1;
        }
        eprintln!("kv8 step {i}: tok={t} abs={abs:.6} rel={rel:.6} arg={arg} ref={ref_arg}");
        assert!(
            rel < FP8_BAND_5PCT_MATCHES_THE_TINY_REFERENCE_GATE,
            "step {i}: kv8 logits left the fp8 band of the CPU reference (rel {rel})"
        );
    }
    assert_eq!(
        argmax_hits,
        tokens.len(),
        "kv8 argmax disagreed with the CPU reference on {} of {} steps",
        tokens.len() - argmax_hits,
        tokens.len()
    );
}

const KV8_VS_SERIAL_ATTN_BAND_3PCT_ONLY_FP8_ROUNDING_SEPARATES_THE_ARMS: f32 = 0.03;

#[test]
fn kv8_full_layer_attention_stays_in_band_with_the_serial_arm_on_identical_state() {
    if !have_gpu() {
        return;
    }
    let _env = kv8_env_on_serialized_because_build_reads_the_env();
    let cfg = hd32_config_because_the_shared_fold_strides_head_dim_in_32_lane_strips();
    let full_li = cfg
        .layer_types
        .iter()
        .position(|t| matches!(t, GptOssLayerType::Full))
        .expect("config carries a full layer");
    let hw = tiny_weights(&cfg, 0x6055_8802);
    let tokens: [u32; 6] = [7, 3, 29, 15, 8, 41];
    let mut probes: Vec<Vec<f32>> = Vec::new();
    for arm in ["0", "1"] {
        std::env::set_var(gow::KV_FP8_ENV, arm);
        let mut gpu = gow::GptOssWgpu::new(cfg.clone(), &hw, 32).expect("build arm");
        for t in tokens {
            gpu.decode_step(t).expect("arm step");
        }
        probes.push(
            gpu.debug_probe(&format!("attnpk{full_li}"))
                .expect("full-layer attn probe"),
        );
    }
    std::env::set_var(gow::KV_FP8_ENV, "1");
    let (abs, rel) = rel_err(&probes[1], &probes[0]);
    eprintln!("kv8-vs-serial attnpk{full_li} abs={abs:.6} rel={rel:.6}");
    assert!(
        rel > 0.0,
        "identical outputs mean the env flip never reached the attention arm; the A/B is vacuous"
    );
    assert!(
        rel < KV8_VS_SERIAL_ATTN_BAND_3PCT_ONLY_FP8_ROUNDING_SEPARATES_THE_ARMS,
        "kv8 full-layer attention drifted {rel} from the serial arm on identical inputs"
    );
}

#[test]
fn kv8_chunked_prefill_rows_land_in_the_fp8_cache_bit_identically_to_stepped_rows() {
    if !have_gpu() {
        return;
    }
    let _env = kv8_env_on_serialized_because_build_reads_the_env();
    let cfg = hd32_config_because_the_shared_fold_strides_head_dim_in_32_lane_strips();
    let hw = tiny_weights(&cfg, 0x6055_8803);
    let prompt: Vec<u32> = (0..21u32).map(|i| (i * 5 + 2) % 60 + 1).collect();

    let mut chunked = gow::GptOssWgpu::new(cfg.clone(), &hw, MAX_SEQ).expect("build chunked");
    assert!(
        chunked.prefill_chunk_len() > 1,
        "the M-row prefill graph must exist or this test never exercises the pf quantize twin"
    );
    let consumed = chunked.prefill_tokens(&prompt).expect("chunked prefill");
    assert_eq!(consumed, prompt.len());

    let mut stepped = gow::GptOssWgpu::new(cfg.clone(), &hw, MAX_SEQ).expect("build stepped");
    for t in &prompt {
        stepped.prefill_step(*t).expect("stepped prefill");
    }

    let (_, la) = chunked.decode_step_logits(9).expect("decode after chunks");
    let (_, lb) = stepped.decode_step_logits(9).expect("decode after steps");
    let (abs, rel) = rel_err(&la, &lb);
    eprintln!("kv8 chunked-vs-stepped decode abs={abs:.7} rel={rel:.7}");
    assert!(
        abs <= 1e-5,
        "chunk rows reached the fp8 cache differently than stepped rows (abs {abs}); the \
         M-row prefill writes bit-identical bf16 rows, the quantize twin is deterministic on \
         them, so any drift here means a chunk arm skipped or mis-addressed the twin"
    );
}

#[test]
fn kv8_sinks_stay_load_bearing_through_the_fp8_stage2() {
    if !have_gpu() {
        return;
    }
    let _env = kv8_env_on_serialized_because_build_reads_the_env();
    let cfg = hd32_config_because_the_shared_fold_strides_head_dim_in_32_lane_strips();
    let mut hw = tiny_weights(&cfg, 0x6055_8804);
    let tokens: [u32; 6] = [5, 17, 4, 33, 21, 9];
    let mut gpu = gow::GptOssWgpu::new(cfg.clone(), &hw, 32).expect("build");
    let mut base = Vec::new();
    for t in tokens {
        base = gpu.decode_step_logits(t).expect("step").1;
    }

    for layer in hw.layers.iter_mut() {
        for s in layer.attn.sinks.iter_mut() {
            *s += 8.0;
        }
    }
    let mut gpu2 = gow::GptOssWgpu::new(cfg.clone(), &hw, 32).expect("build shifted");
    let mut shifted = Vec::new();
    for t in tokens {
        shifted = gpu2.decode_step_logits(t).expect("step").1;
    }
    let (abs, _) = rel_err(&base, &shifted);
    assert!(
        abs > 1e-4,
        "raising every sink by +8 must move the logits through the fp8 flash stage2 (abs {abs})"
    );

    let mut st = gow::RefState::new(&cfg);
    let mut want = Vec::new();
    for t in tokens {
        want = gow::reference_step(&cfg, &hw, &mut st, t).expect("ref");
    }
    let (_, rel) = rel_err(&shifted, &want);
    assert!(
        rel < FP8_BAND_5PCT_MATCHES_THE_TINY_REFERENCE_GATE,
        "shifted-sink kv8 logits diverged from reference (rel {rel})"
    );
}
