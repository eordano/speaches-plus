#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::TINY_E4B_CONFIG;

use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b_wgpu as e4b;
use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;

static ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE:
    std::sync::Mutex<()> = std::sync::Mutex::new(());

struct EnvGuard {
    _serialized: std::sync::MutexGuard<'static, ()>,
    keys: Vec<&'static str>,
}

fn env_on(pairs: &[(&'static str, &str)]) -> EnvGuard {
    let g = ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let mut keys = Vec::new();
    for (k, v) in pairs {
        std::env::set_var(k, v);
        keys.push(*k);
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

const SHALLOW_SPLITS_ENV: &str = "NV_WGPU_FLASH_SPLITS";

const SHALLOW_LADDER_RUNGS_SPAN_BOTH_SIDES_OF_THE_HAND_UNROLLED_16: [u32; 7] =
    [1, 2, 4, 8, 16, 32, 64];

const NAGA_LADDER_RUNGS_COVER_EVERY_COUNT_THE_SCRATCH_PROVISIONS: [u32; 10] =
    [1, 2, 4, 8, 16, 32, 64, 94, 128, 256];

#[test]
fn only_the_shipped_sixteen_keeps_the_hand_unrolled_stage2_and_every_other_count_takes_the_generated_merge(
) {
    for splits in NAGA_LADDER_RUNGS_COVER_EVERY_COUNT_THE_SCRATCH_PROVISIONS {
        let (src, entry) = e4b::e4b_flash2_stage2_module_source_and_entry(splits);
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
            panic!("shallow stage2 at {splits} splits does not parse: {e:?}")
        });
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("shallow stage2 at {splits} splits fails validation: {e:?}"));
        let want = if splits == e4b::FLASH_SPLITS {
            "g4w_flash_splitk_stage2_pk"
        } else {
            e4b::FLASH2_DEEP_ENTRY
        };
        assert_eq!(
            entry, want,
            "at {splits} splits the shallow decode must dispatch `{want}`; {}",
            e4b::SHALLOW_STAGE2_RULE
        );
        assert!(
            src.contains(&format!("let splits = {splits}u;")),
            "the dispatched shallow stage2 must carry its own split count at {splits}; {}",
            e4b::SHALLOW_STAGE2_RULE
        );
    }
}

#[test]
fn the_hand_unrolled_body_still_merges_sixteen_partials_whatever_let_splits_is_folded_to() {
    let (prelude, entry16) = e4b::e4b_flash2_stage2_module_source_and_entry(e4b::FLASH_SPLITS);
    assert_eq!(
        entry16, "g4w_flash_splitk_stage2_pk",
        "the shipped split count must keep the hand-unrolled body, or this change moved a \
         performance default instead of fixing a defect"
    );
    assert_eq!(
        e4b::e4b_flash2_pk_folded_truncating_to_splits(&prelude, e4b::FLASH_SPLITS),
        prelude,
        "at the shipped split count the fold is the identity, so the serving stage2 is the \
         unmodified shared body byte for byte and no measured e4b number can move with this \
         routing. {}",
        e4b::SHALLOW_STAGE2_RULE
    );
    assert!(
        prelude.contains("ssc15") && !prelude.contains("ssc16"),
        "g4shared_flash2_pk is expected to hand-unroll exactly 16 partial accumulators; if that \
         changed, {} is stale and the fold may have become safe or become wrong in a new way",
        e4b::SHALLOW_STAGE2_RULE
    );
    for splits in [1u32, 2, 4, 8, 32, 64, 128] {
        let folded = e4b::e4b_flash2_pk_folded_truncating_to_splits(&prelude, splits);
        assert!(
            folded.contains(&format!("let splits = {splits}u;")),
            "the truncating fold must rewrite the anchor at {splits}"
        );
        assert!(
            folded.contains("ssc15") && !folded.contains("ssc16"),
            "folding `{}` to {splits} rewrites only the per-head base and the running-max scan: \
             the merge stays unrolled to 16 partials, which is why the shallow decode may not use \
             it at {splits}. {}",
            e4b::FLASH2_PK_SPLIT_ANCHOR,
            e4b::SHALLOW_STAGE2_RULE
        );
    }
}

const TINY_MAX_SEQ_192_EXCEEDS_16_TIMES_FD_WARPS_SO_A_TRUNCATED_MERGE_DROPS_REAL_POSITIONS: usize =
    192;

const TINY_STEPS_190_KEEPS_EVERY_STEP_INSIDE_THE_CACHE: usize = 190;

const TINY_WINDOW_8_SCANS_ONE_SPLIT_AT_EVERY_RUNG_SO_ONLY_A_WIDE_WINDOW_PUTS_EVERY_LAYER_PAST_16_TIMES_FD_WARPS:
    (&str, &str) = ("\"sliding_window\": 8,", "\"sliding_window\": 512,");

fn wide_window_tiny_config() -> Gemma4Config {
    let (narrow, wide) = TINY_WINDOW_8_SCANS_ONE_SPLIT_AT_EVERY_RUNG_SO_ONLY_A_WIDE_WINDOW_PUTS_EVERY_LAYER_PAST_16_TIMES_FD_WARPS;
    assert_eq!(
        TINY_E4B_CONFIG.matches(narrow).count(),
        1,
        "the shared tiny e4b config no longer carries `{narrow}`, so this ladder would silently \
         fall back to a window narrower than one split's worth of positions and stop exercising \
         the merge width at all"
    );
    Gemma4Config::from_hf_json_str(&TINY_E4B_CONFIG.replace(narrow, wide)).unwrap()
}

fn ladder_logits(splits: u32) -> Option<(EnvGuard, Vec<Vec<f32>>)> {
    ctx_or_skip()?;
    let g = env_on(&[
        (SHALLOW_SPLITS_ENV, &splits.to_string()),
        (e4b::E4B_SPLIT_DEPTH_ENV, "0"),
    ]);
    assert_eq!(
        e4b::decode_splits(),
        splits,
        "{SHALLOW_SPLITS_ENV} must reach the decoder, or this ladder measures one arm seven times"
    );
    assert!(
        e4b::deep_split_arm().is_none(),
        "{}=0 must retire the deep arm so this ladder is the shallow merge alone",
        e4b::E4B_SPLIT_DEPTH_ENV
    );
    let config = wide_window_tiny_config();
    let weights = common::tiny_e4b_host_weights(&config, 0x5eed);
    let mut m = Gemma4E4bWgpu::new(
        wide_window_tiny_config(),
        &weights,
        TINY_MAX_SEQ_192_EXCEEDS_16_TIMES_FD_WARPS_SO_A_TRUNCATED_MERGE_DROPS_REAL_POSITIONS,
    )
    .expect("tiny e4b wgpu build");
    assert_eq!(
        m.split_bands_shallow_deep_deep2().0,
        splits,
        "the built model must report the shallow band this rung asked for"
    );
    let rows: Vec<Vec<f32>> = (0..TINY_STEPS_190_KEEPS_EVERY_STEP_INSIDE_THE_CACHE)
        .map(|i| m.decode_step_logits((i * 7 % 512) as u32).unwrap().1)
        .collect();
    let fingerprint = rows
        .iter()
        .flat_map(|r| r.iter())
        .fold(0u64, |a, v| a.rotate_left(1) ^ v.to_bits() as u64);
    eprintln!(
        "E4B-SHALLOW-SPLIT-RUNG splits={splits} window={} rows={} last_row_sum={:.6e} \
         fingerprint={fingerprint:#018x}",
        config.sliding_window,
        rows.len(),
        rows.last().map(|r| r.iter().sum::<f32>()).unwrap_or(0.0)
    );
    Some((g, rows))
}

const REPAIRED_MERGE_READS_ZERO_AND_THE_TRUNCATING_FOLD_READS_4E_3_ON_THIS_HARNESS_SO_A_PAIR_OF_RUNGS_MAY_MOVE_A_LOGIT_BY_AT_MOST_THIS_FRACTION_OF_THE_LADDERS_SCALE:
    f32 = 1.0e-3;

const NO_PAIR_OF_RUNGS_MAY_FLIP_A_GREEDY_ARGMAX_BECAUSE_A_SPLIT_COUNT_IS_NOT_A_DIFFERENT_FUNCTION:
    usize = 0;

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |acc, (i, v)| {
            if *v > acc.1 {
                (i, *v)
            } else {
                acc
            }
        })
        .0
}

#[test]
fn every_pair_of_shallow_split_counts_decodes_the_same_rows_because_a_split_count_is_only_a_merge_width(
) {
    let mut ladder: Vec<(u32, Vec<Vec<f32>>)> = Vec::new();
    for splits in SHALLOW_LADDER_RUNGS_SPAN_BOTH_SIDES_OF_THE_HAND_UNROLLED_16 {
        let Some((g, rows)) = ladder_logits(splits) else {
            return;
        };
        drop(g);
        ladder.push((splits, rows));
    }
    let scale: f32 = ladder
        .iter()
        .flat_map(|(_, r)| r.iter().flat_map(|row| row.iter()))
        .fold(0.0f32, |a, v| a.max(v.abs()));
    assert!(
        scale.is_finite() && scale > 0.0,
        "the ladder produced no finite logit scale, so it has nothing to normalise against"
    );
    let mut worst_pair = (0u32, 0u32, 0.0f32, 0usize);
    for (i, (sa, ra)) in ladder.iter().enumerate() {
        for (sb, rb) in ladder.iter().skip(i + 1) {
            let mut worst = 0.0f32;
            let mut flips = 0usize;
            for (a, b) in ra.iter().zip(rb.iter()) {
                assert_eq!(a.len(), b.len(), "vocab width moved between rungs");
                for (x, y) in a.iter().zip(b.iter()) {
                    worst = worst.max((x - y).abs());
                }
                if argmax(a) != argmax(b) {
                    flips += 1;
                }
            }
            eprintln!(
                "E4B-SHALLOW-SPLIT-NULL pair={sa}v{sb} rows={} worst_abs_logit_delta={worst:.6e} \
                 logit_scale={scale:.6e} relative={:.6e} argmax_flips={flips}",
                ra.len(),
                worst / scale
            );
            if worst > worst_pair.2 {
                worst_pair = (*sa, *sb, worst, flips);
            }
        }
    }
    let (sa, sb, worst, flips) = worst_pair;
    eprintln!(
        "E4B-SHALLOW-SPLIT-NULL-BAND rungs={:?} worst_pair={sa}v{sb} \
         worst_abs_logit_delta={worst:.6e} relative={:.6e} argmax_flips={flips}",
        SHALLOW_LADDER_RUNGS_SPAN_BOTH_SIDES_OF_THE_HAND_UNROLLED_16,
        worst / scale
    );
    assert!(
        worst / scale
            <= REPAIRED_MERGE_READS_ZERO_AND_THE_TRUNCATING_FOLD_READS_4E_3_ON_THIS_HARNESS_SO_A_PAIR_OF_RUNGS_MAY_MOVE_A_LOGIT_BY_AT_MOST_THIS_FRACTION_OF_THE_LADDERS_SCALE,
        "splits {sa} and {sb} disagree by {worst:.6e} against a ladder scale of {scale:.6e}: \
         stage1 writes exactly as many partials as the grid has splits and stage2 must merge \
         exactly that many, so a move beyond bf16 reassociation noise means the merge width and \
         the write width disagree. Every rung of this ladder is bit-identical on the repaired \
         merge and the truncating fold (NV_E4B_DEEP_STAGE2_LEGACY16=1) reads 4.36e-3 here. {}",
        e4b::SHALLOW_STAGE2_RULE
    );
    assert_eq!(
        flips, NO_PAIR_OF_RUNGS_MAY_FLIP_A_GREEDY_ARGMAX_BECAUSE_A_SPLIT_COUNT_IS_NOT_A_DIFFERENT_FUNCTION,
        "splits {sa} and {sb} flipped {flips} greedy argmaxes: a reduction order may perturb a \
         near-tie, it may not rewrite the decode. {}",
        e4b::SHALLOW_STAGE2_RULE
    );
}

#[test]
fn the_legacy_env_still_reaches_the_truncating_fold_so_the_defect_stays_measurable() {
    let _g = env_on(&[(e4b::E4B_DEEP_STAGE2_LEGACY16_ENV, "1")]);
    for splits in [8u32, 32, 64] {
        let (_, entry) = e4b::e4b_flash2_stage2_module_source_and_entry(splits);
        assert_eq!(
            entry, "g4w_flash_splitk_stage2_pk",
            "{}=1 must restore the truncating fold on the shallow arm too, or the numbers \
             published on it can never be reproduced",
            e4b::E4B_DEEP_STAGE2_LEGACY16_ENV
        );
    }
}

#[test]
fn a_zero_env_build_provisions_the_scratch_the_shipped_shallow_count_needs_and_no_more() {
    let _g = env_on(&[]);
    assert_eq!(
        e4b::decode_splits(),
        e4b::FLASH_SPLITS,
        "the shipped shallow split count is the hand-unrolled 16 and this change does not move it"
    );
    assert!(
        e4b::scratch_splits() >= e4b::decode_splits(),
        "stage1 indexes the split-k scratch by split, so the provision must cover the shallow band"
    );
}
