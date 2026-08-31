#![cfg(feature = "wgpu")]

mod common;
use common::bf16_lin_gow_bias as bf16_lin;
use common::mx_stack;
use common::tiny_config_gpt_oss;
use common::LcgSplitMix64TwoSided as Lcg;

use nv_models::gpt_oss_wgpu as gow;
use nv_models::gpt_oss_wgpu::{GptOssConfig, GptOssLayerType};

const MAX_SEQ_1400_CLEARS_THE_1280_KNEE_WITH_ROOM_FOR_TIMED_STEPS: usize = 1400;

const SHALLOW_SPLITS_8_IS_THE_RECORD_ARM_PIN_TUNED_AT_256_AND_8K: u32 = 8;

static ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE: std::sync::Mutex<()> =
    std::sync::Mutex::new(());

fn hd32_kv8_config_with_room_past_the_knee() -> GptOssConfig {
    let cfg = GptOssConfig {
        head_dim: 32,
        max_position_embeddings: 4096,
        ..tiny_config_gpt_oss()
    };
    assert!(
        cfg.layer_types.contains(&GptOssLayerType::Full),
        "the depth gate rides full-attention layers only; a config without one tests nothing"
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

struct EnvGuard {
    _serialized: std::sync::MutexGuard<'static, ()>,
    keys: Vec<&'static str>,
}

fn env_on(pairs: &[(&'static str, &str)]) -> EnvGuard {
    let g = ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let mut keys = Vec::new();
    std::env::set_var(gow::KV_FP8_ENV, "1");
    keys.push(gow::KV_FP8_ENV);
    for (k, v) in pairs {
        std::env::set_var(k, v);
        keys.push(k);
    }
    EnvGuard {
        _serialized: g,
        keys,
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for k in &self.keys {
            std::env::remove_var(k);
        }
    }
}

#[test]
fn the_measured_knee_and_every_band_boundary_are_constants_the_gate_reads_not_numbers_a_caller_passes(
) {
    let shallow = SHALLOW_SPLITS_8_IS_THE_RECORD_ARM_PIN_TUNED_AT_256_AND_8K;
    let knee = gow::FLASH_DEPTHGATE_KNEE_TOKENS_1280_IS_WHERE_SPLITS_8_AND_64_TIE_AND_BELOW_IT_8_WINS;
    let knee2 = gow::FLASH_DEPTHGATE_KNEE2_TOKENS_3584_IS_WHERE_47_OVERTAKES_24_ON_THE_SAME_LADDER;
    let knee3 =
        gow::FLASH_DEPTHGATE_KNEE3_TOKENS_32768_IS_WHERE_64_TIES_47_AND_WINS_EVERY_DEPTH_BEYOND;
    assert!(
        knee < knee2 && knee2 < knee3,
        "the bands must be ordered: the measured ladder rises 8 -> 24 -> 47 -> 64 as depth grows"
    );
    for tokens in [1u32, 256, 1024, knee - 1] {
        assert_eq!(
            gow::flash_depthgate_splits_for_context_tokens(tokens, shallow),
            shallow,
            "below the knee the gate must return the caller's tuned split count unchanged at \
             {tokens} tokens, or the record arm's shallow pin stops being honoured"
        );
    }
    for (tokens, want) in [
        (
            knee,
            gow::FLASH_DEPTHGATE_BAND1_SPLITS_24_IS_ONE_WORKGROUP_PER_SM_AT_8_KV_HEADS_ON_188_SMS,
        ),
        (
            knee2 - 1,
            gow::FLASH_DEPTHGATE_BAND1_SPLITS_24_IS_ONE_WORKGROUP_PER_SM_AT_8_KV_HEADS_ON_188_SMS,
        ),
        (
            knee2,
            gow::FLASH_DEPTHGATE_BAND2_SPLITS_47_IS_TWO_WORKGROUPS_PER_SM_AT_8_KV_HEADS_ON_188_SMS,
        ),
        (
            knee3 - 1,
            gow::FLASH_DEPTHGATE_BAND2_SPLITS_47_IS_TWO_WORKGROUPS_PER_SM_AT_8_KV_HEADS_ON_188_SMS,
        ),
        (
            knee3,
            gow::FLASH_SPLITS_64_KVSHARE_GRID_IS_N_KV_X_SPLITS_AND_43TOKS_AT_120K_BEAT_S16_BY_1P6X,
        ),
        (
            131072,
            gow::FLASH_SPLITS_64_KVSHARE_GRID_IS_N_KV_X_SPLITS_AND_43TOKS_AT_120K_BEAT_S16_BY_1P6X,
        ),
    ] {
        assert_eq!(
            gow::flash_depthgate_splits_for_context_tokens(tokens, shallow),
            want,
            "the band at {tokens} tokens moved off the measured ladder"
        );
    }
    assert!(
        gow::flash_depthgate_splits_for_context_tokens(knee3, 128) == 64
            && gow::flash_depthgate_splits_for_context_tokens(1, 128) == 128,
        "an explicit split count is the SHALLOW band only; above the knee the measured bands own \
         the grid, which is the whole point of the gate"
    );
}

#[test]
fn the_scratch_provision_covers_every_band_the_gate_can_pick_or_a_deep_step_writes_out_of_bounds() {
    let _g = env_on(&[(gow::FLASH_DEPTHGATE_ENV, "1")]);
    for shallow in [1u32, 8, 24, 47, 64, 128] {
        let provisioned = gow::flash_depthgate_scratch_splits_must_cover_every_band_the_gate_can_pick(
            shallow,
        );
        for tokens in [1u32, 1280, 3584, 32768, 131072] {
            assert!(
                gow::flash_depthgate_splits_for_context_tokens(tokens, shallow) <= provisioned,
                "shallow {shallow} at {tokens} tokens picks more splits than the scratch rows \
                 provisioned; stage1 indexes scratch by split and would write past the buffer"
            );
        }
    }
    assert_eq!(
        gow::flash_depthgate_scratch_splits_must_cover_every_band_the_gate_can_pick(8),
        gow::FLASH_SPLITS_64_KVSHARE_GRID_IS_N_KV_X_SPLITS_AND_43TOKS_AT_120K_BEAT_S16_BY_1P6X,
        "the record arm's shallow pin of 8 must still provision the deep band's rows"
    );
}

fn grid_y_after_step_at(pos: usize, gate: bool) -> Option<u32> {
    let pairs: Vec<(&'static str, &str)> = if gate {
        vec![
            (gow::FLASH_DEPTHGATE_ENV, "1"),
            ("NV_GPTOSS_WGPU_FLASH_SPLITS", "8"),
        ]
    } else {
        vec![("NV_GPTOSS_WGPU_FLASH_SPLITS", "8")]
    };
    let _g = env_on(&pairs);
    let ctx = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            return None;
        }
    };
    eprintln!("[wgpu] adapter: {}", ctx.info.name);
    let cfg = hd32_kv8_config_with_room_past_the_knee();
    let hw = tiny_weights(&cfg, 0x9e37_79b9);
    let mut gpu = gow::GptOssWgpu::new(
        cfg,
        &hw,
        MAX_SEQ_1400_CLEARS_THE_1280_KNEE_WITH_ROOM_FOR_TIMED_STEPS,
    )
    .expect("build model");
    assert!(
        gpu.full_attention_flash1_entry().is_some(),
        "no full-attention flash1 pass was recorded, so this suite would assert on nothing"
    );
    gpu.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(pos)
        .expect("synthetic fill");
    gpu.decode_step(1).expect("decode step");
    gpu.depthgate_full_attention_flash1_grid_y_as_last_encoded()
}

#[test]
fn the_gate_off_grid_is_the_pinned_split_count_at_every_depth_and_the_gate_on_grid_follows_the_bands(
) {
    let shallow = SHALLOW_SPLITS_8_IS_THE_RECORD_ARM_PIN_TUNED_AT_256_AND_8K;
    let Some(off_shallow) = grid_y_after_step_at(256, false) else {
        return;
    };
    let off_deep = grid_y_after_step_at(1300, false).expect("adapter was present a moment ago");
    assert_eq!(
        (off_shallow, off_deep),
        (shallow, shallow),
        "with the gate off the split count is read once at build and must never move; a default \
         that moved would be an unannounced performance-default flip"
    );
    let on_shallow = grid_y_after_step_at(256, true).expect("adapter was present a moment ago");
    let on_deep = grid_y_after_step_at(1300, true).expect("adapter was present a moment ago");
    assert_eq!(
        on_shallow, shallow,
        "below the knee the armed gate must dispatch the pinned shallow grid, byte-for-byte the \
         gate-off shape"
    );
    assert_eq!(
        on_deep,
        gow::FLASH_DEPTHGATE_BAND1_SPLITS_24_IS_ONE_WORKGROUP_PER_SM_AT_8_KV_HEADS_ON_188_SMS,
        "past the knee the armed gate must widen the stage1 grid to the measured band; a grid \
         that stayed at 8 is the 3x-at-120k defect the gate exists to fix"
    );
}

#[test]
fn arming_the_gate_below_the_knee_is_bit_for_bit_the_unarmed_decode_because_the_bands_start_above_it(
) {
    let steps = 8usize;
    let run = |gate: bool| -> Option<Vec<Vec<u32>>> {
        let pairs: Vec<(&'static str, &str)> = if gate {
            vec![
                (gow::FLASH_DEPTHGATE_ENV, "1"),
                ("NV_GPTOSS_WGPU_FLASH_SPLITS", "8"),
            ]
        } else {
            vec![("NV_GPTOSS_WGPU_FLASH_SPLITS", "8")]
        };
        let _g = env_on(&pairs);
        let ctx = match nv_kernels::wgpu_backend::WgpuContext::shared() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("[skip] no wgpu adapter: {e}");
                return None;
            }
        };
        eprintln!("[wgpu] adapter: {}", ctx.info.name);
        let cfg = hd32_kv8_config_with_room_past_the_knee();
        let hw = tiny_weights(&cfg, 0x51ed_2701);
        let mut gpu = gow::GptOssWgpu::new(
            cfg,
            &hw,
            MAX_SEQ_1400_CLEARS_THE_1280_KNEE_WITH_ROOM_FOR_TIMED_STEPS,
        )
        .expect("build model");
        let mut tok = 3u32;
        let mut out = Vec::new();
        for _ in 0..steps {
            let (t, logits) = gpu.decode_step_logits(tok).expect("decode step");
            out.push(logits.iter().map(|v| v.to_bits()).collect::<Vec<u32>>());
            tok = t;
        }
        Some(out)
    };
    let Some(off) = run(false) else { return };
    let on = run(true).expect("adapter was present a moment ago");
    assert_eq!(
        off.len(),
        steps,
        "the reference run produced no logits to compare"
    );
    assert_eq!(
        on, off,
        "the armed gate changed logit bits at depths below the knee, where it is defined to \
         select the pinned shallow split count and therefore encode the identical grid"
    );
}

#[test]
fn every_generated_stage1_rewrite_validates_under_naga_at_the_real_gptoss_geometry_without_a_gpu() {
    use nv_kernels::wgpu_backend::kernels::flash_decode as fd;
    let base = format!("{}\n{}", fd::WGSL, fd::fold_stage1_source_sd(64, true, 8));
    let mut cases: Vec<(String, String)> = vec![(
        gow::flash_rowpair_stage1_entry_sd(64, 8),
        gow::flash_rowpair_stage1_source_sd(64, 8),
    )];
    for unroll in [1u32, 2, 4, 8] {
        for block in [false, true] {
            cases.push((
                gow::flash_kpre_stage1_entry_sd(64, 8, unroll, block),
                gow::flash_kpre_stage1_source_sd(64, 8, unroll, block),
            ));
        }
    }
    for (entry, body) in cases {
        let src = nv_kernels::wgpu_backend::compose(&format!("{base}\n{body}"));
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("{entry}: wgsl parse failed: {e}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{entry}: validation failed: {e:?}"));
        assert!(
            src.contains(&format!("fn {entry}(")),
            "{entry}: the generated source must define the entry the model dispatches, or the \
             arm would fall back to the stock kernel and measure nothing"
        );
    }
}

#[test]
fn the_gate_and_the_three_stage1_rewrites_are_default_off_and_the_three_rewrites_stay_banked() {
    for env in [
        gow::FLASH_DEPTHGATE_ENV,
        gow::FLASH_DEPTHGATE_SPLITS_ENV,
        gow::FLASH_KPRE_ENV,
        gow::FLASH_KPRE_UNROLL_ENV,
        gow::FLASH_ROWPAIR_ENV,
        gow::FLASH_BLOCKSPLIT_ENV,
    ] {
        assert!(
            std::env::var(env).is_err(),
            "{env} leaked into the default-contract probe; this test must see the shipped \
             defaults, not a caller's pins"
        );
    }
    assert!(
        !gow::flash_depthgate_enabled_shallow_tuned_splits_starve_the_grid_at_depth(),
        "the depth gate is an opt-in arm: it re-partitions the split-k reduction at depth, which \
         is the fp-reorder class, and no performance default may flip without its own round"
    );
    assert!(
        !gow::flash_kpre_banked_negative_preloading_four_rounds_costs_more_warp_residency_than_the_loads_in_flight_buy(),
        "NV_GOW_FLASH_KPRE stays banked OFF: at 122880 with the gate armed it measured 13.74 \
         (unroll 4) / 14.05 (unroll 2) / 17.50 (unroll 8) against a 12.79-12.85 base, and its \
         stage1 span rose 7.87 -> 9.34 ms/step; the e4b W4_KPRE win does not port to a kernel \
         whose per-thread softmax state already bounds warp residency"
    );
    assert!(
        !gow::flash_rowpair_banked_neutral_two_rows_per_warp_round_fills_the_idle_lanes_and_halves_the_rounds_for_no_wall(),
        "NV_GOW_FLASH_ROWPAIR stays banked OFF: giving the idle upper 16 lanes a second KV row \
         halves the round count and moves the stage1 span only 7.87 -> 7.80 ms/step (203.8 -> \
         205.8 GB/s), and the wall is unchanged at 12.90-12.97 vs 12.77-12.82 -- the R2 lesson, \
         spans move and the wall does not"
    );
    assert!(
        !gow::flash_blocksplit_banked_negative_cyclic_splits_sweep_the_kv_stream_in_lockstep_and_contiguous_blocks_do_not(),
        "NV_GOW_FLASH_BLOCKSPLIT stays banked OFF: giving each split a contiguous position block \
         instead of the cyclic stride measured 13.21-13.24 at 122880 against a 12.78-12.81 base \
         and its stage1 span rose 7.87 -> 8.39 ms/step; the cyclic partition's global lockstep \
         over the KV stream beats per-split page locality"
    );
}
