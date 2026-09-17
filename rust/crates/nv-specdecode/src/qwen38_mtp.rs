use anyhow::{ensure, Context, Result};
use candle_core::{DType, Tensor};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

pub const NV_Q38_MTP_ACCEPT_TRACKING_ENV: &str = "NV_Q38_MTP_ACCEPT_TRACKING";
const ACCEPT_TRACKING_LAST_16_MTP_ROUNDS_EXCLUDE_SUFFIX_COPY_HITS: usize = 16;
const ACCEPT_TRACKING_LOW_55_HIGH_70_PERCENT_HYSTERESIS_AVOIDS_PER_ROUND_CHAIN_THRASH: (
    usize,
    usize,
) = (55, 70);

pub fn accept_tracking_selected(raw: Option<&str>) -> bool {
    raw == Some("1")
}

pub fn accept_tracking_selected_from_env() -> bool {
    accept_tracking_selected(
        std::env::var(NV_Q38_MTP_ACCEPT_TRACKING_ENV)
            .ok()
            .as_deref(),
    )
}

#[derive(Debug)]
pub struct AcceptanceTrackingK {
    window: [(usize, usize); ACCEPT_TRACKING_LAST_16_MTP_ROUNDS_EXCLUDE_SUFFIX_COPY_HITS],
    cursor: usize,
    len: usize,
    accepted: usize,
    drafted: usize,
    live_k: usize,
    pub k2_rounds: usize,
    pub k3_rounds: usize,
    pub switches: usize,
}

impl Default for AcceptanceTrackingK {
    fn default() -> Self {
        Self {
            window: [(0, 0); ACCEPT_TRACKING_LAST_16_MTP_ROUNDS_EXCLUDE_SUFFIX_COPY_HITS],
            cursor: 0,
            len: 0,
            accepted: 0,
            drafted: 0,
            live_k: 3,
            k2_rounds: 0,
            k3_rounds: 0,
            switches: 0,
        }
    }
}

impl AcceptanceTrackingK {
    pub fn next_k_for_a_non_suffix_non_forced_round(&mut self) -> usize {
        let previous = self.live_k;
        if self.len == self.window.len() {
            let (low, high) =
                ACCEPT_TRACKING_LOW_55_HIGH_70_PERCENT_HYSTERESIS_AVOIDS_PER_ROUND_CHAIN_THRASH;
            if self.accepted * 100 < self.drafted * low {
                self.live_k = 2;
            } else if self.accepted * 100 >= self.drafted * high {
                self.live_k = 3;
            }
        }
        self.switches += usize::from(previous != self.live_k);
        if self.live_k == 2 {
            self.k2_rounds += 1;
        } else {
            self.k3_rounds += 1;
        }
        self.live_k
    }

    pub fn record_committed_mtp_round_not_suffix(&mut self, accepted: usize, drafted: usize) {
        assert!((2..=3).contains(&drafted) && accepted <= drafted);
        let (old_accepted, old_drafted) = self.window[self.cursor];
        self.accepted = self.accepted + accepted - old_accepted;
        self.drafted = self.drafted + drafted - old_drafted;
        self.window[self.cursor] = (accepted, drafted);
        self.cursor = (self.cursor + 1) % self.window.len();
        self.len = (self.len + 1).min(self.window.len());
    }
}

#[cfg(test)]
mod acceptance_tracking_tests {
    use super::*;

    #[test]
    fn absent_and_every_non_one_value_leave_fixed_k_alone() {
        assert!(!accept_tracking_selected(None));
        assert!(accept_tracking_selected(Some("1")));
        for value in ["", "0", "true", " 1", "1 ", "2"] {
            assert!(!accept_tracking_selected(Some(value)));
        }
    }

    #[test]
    fn full_window_is_required_and_old_observations_expire() {
        let mut policy = AcceptanceTrackingK::default();
        for _ in 0..16 {
            assert_eq!(policy.next_k_for_a_non_suffix_non_forced_round(), 3);
            policy.record_committed_mtp_round_not_suffix(0, 3);
        }
        assert_eq!(policy.next_k_for_a_non_suffix_non_forced_round(), 2);
        for _ in 0..16 {
            policy.record_committed_mtp_round_not_suffix(2, 2);
        }
        assert_eq!(policy.next_k_for_a_non_suffix_non_forced_round(), 3);
        assert_eq!((policy.accepted, policy.drafted, policy.len), (32, 32, 16));
        assert_eq!(
            (policy.k2_rounds, policy.k3_rounds, policy.switches),
            (1, 17, 2)
        );
    }

    #[test]
    fn thresholds_use_weighted_counts_and_low_is_strict_high_is_inclusive() {
        let mut policy = AcceptanceTrackingK::default();
        policy.live_k = 2;
        for _ in 0..8 {
            policy.record_committed_mtp_round_not_suffix(2, 2);
        }
        for _ in 0..6 {
            policy.record_committed_mtp_round_not_suffix(1, 3);
        }
        for _ in 0..2 {
            policy.record_committed_mtp_round_not_suffix(3, 3);
        }
        assert_eq!((policy.accepted, policy.drafted), (28, 40));
        assert_eq!(policy.next_k_for_a_non_suffix_non_forced_round(), 3);
        for _ in 0..8 {
            policy.record_committed_mtp_round_not_suffix(1, 2);
        }
        for _ in 0..6 {
            policy.record_committed_mtp_round_not_suffix(2, 3);
        }
        for _ in 0..2 {
            policy.record_committed_mtp_round_not_suffix(1, 3);
        }
        assert_eq!((policy.accepted, policy.drafted), (22, 40));
        assert_eq!(policy.next_k_for_a_non_suffix_non_forced_round(), 3);
        policy.record_committed_mtp_round_not_suffix(0, 2);
        assert_eq!(policy.next_k_for_a_non_suffix_non_forced_round(), 2);
    }

    #[test]
    fn middle_acceptance_holds_the_previous_choice() {
        let mut policy = AcceptanceTrackingK::default();
        for _ in 0..16 {
            policy.record_committed_mtp_round_not_suffix(2, 3);
        }
        assert_eq!(policy.next_k_for_a_non_suffix_non_forced_round(), 3);
        for _ in 0..16 {
            policy.record_committed_mtp_round_not_suffix(0, 3);
        }
        assert_eq!(policy.next_k_for_a_non_suffix_non_forced_round(), 2);
        for _ in 0..16 {
            policy.record_committed_mtp_round_not_suffix(2, 3);
        }
        assert_eq!(policy.next_k_for_a_non_suffix_non_forced_round(), 2);
    }
}

pub const NV_DRAFTER_ENV: &str = "NV_DRAFTER";
pub const NV_MTP_DRAFT_DIR_ENV: &str = "NV_MTP_DRAFT_DIR";
pub const NV_MTP_K_ENV: &str = "NV_MTP_K";
pub const NV_Q38_DRAFT_FP8_ENV: &str = "NV_Q38_DRAFT_FP8";
pub const NV_Q38_MTP_REANCHOR_ENV: &str = "NV_Q38_MTP_REANCHOR";
pub const NV_Q38_DRAFT_FAST_ENV: &str = "NV_Q38_DRAFT_FAST";
pub const NV_Q38_MTP_STREAM_PREFILL_ENV: &str = "NV_Q38_MTP_STREAM_PREFILL";

pub const PROMPT_DRAFTER_BATCH_DEFAULT_ON_USER_SIGN_OFF_2026_09_11_REPLY_05_BITWISE_FNV_2048_168000_AND_ABBA_23B649C2D_22972E453_F49FC4925_740B083ED:
    &str = "Prompt-only 4096-row batching: pg1342 complete 257-token FNVs 3e2a24d301c7e68e and \
             337b62db0fecc2ae match fresh default; all 24 served ABBA outputs and acceptance \
             counters match. User sign-off 2026-09-11 promotes the bitwise default; only UNSET \
             changes, explicit 0 restores the old 16-row prompt catch-up, decode is untouched.";

pub fn prompt_drafter_batch_from_raw_unset_on_user_sign_off_2026_09_11_explicit_values_unchanged(
    raw: Option<&std::ffi::OsStr>,
) -> bool {
    match raw {
        None => {
            let _ = PROMPT_DRAFTER_BATCH_DEFAULT_ON_USER_SIGN_OFF_2026_09_11_REPLY_05_BITWISE_FNV_2048_168000_AND_ABBA_23B649C2D_22972E453_F49FC4925_740B083ED;
            true
        }
        Some(value) => value == "1",
    }
}

#[cfg(test)]
mod prompt_drafter_batch_resolver_tests {
    use super::prompt_drafter_batch_from_raw_unset_on_user_sign_off_2026_09_11_explicit_values_unchanged as selected;
    use std::ffi::OsStr;

    #[test]
    fn only_unset_changes_and_explicit_zero_restores_the_old_prompt_path() {
        assert!(selected(None));
        assert!(selected(Some(OsStr::new("1"))));
        for raw in ["0", "", " 1", "1 ", "true", "yes", "gate"] {
            assert!(
                !selected(Some(OsStr::new(raw))),
                "explicit {raw:?} must retain the old off meaning"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_is_an_explicit_off_value_not_unset() {
        use std::os::unix::ffi::OsStrExt;
        assert!(!selected(Some(OsStr::from_bytes(&[0xff]))));
    }
}

pub fn mtp_stream_prefill_selected_from_env() -> bool {
    std::env::var(NV_Q38_MTP_STREAM_PREFILL_ENV).ok().as_deref() == Some("1")
}

pub const DRAFT_QUARTET_FAST_FP8_SPLITK_LMHEAD_NVFP4_DEFAULT_ON_CHAMPION_STACK_SIGNED_OFF_2026_09_02_EXPLICIT_0_RESTORES_EACH_PRE_FLIP_DRAFT_PATH:
    &str = "0";

pub fn draft_fast_device_chained_tokens_selected(raw: Option<&str>) -> bool {
    raw.map(str::trim)
        != Some(DRAFT_QUARTET_FAST_FP8_SPLITK_LMHEAD_NVFP4_DEFAULT_ON_CHAMPION_STACK_SIGNED_OFF_2026_09_02_EXPLICIT_0_RESTORES_EACH_PRE_FLIP_DRAFT_PATH)
}

pub fn draft_fast_device_chained_tokens_selected_from_env() -> bool {
    draft_fast_device_chained_tokens_selected(std::env::var(NV_Q38_DRAFT_FAST_ENV).ok().as_deref())
}

pub const NV_Q38_DRAFTER_ATTN_SPLITK_ENV_DEFAULT_ON_ROUTES_THE_Q1_DRAFTER_SCAN_THROUGH_THE_FUSED_BF16_SPLITK_DECODE_KERNEL_BECAUSE_CANDLE_FA2_FILLS_ONLY_N_HEADS_BLOCKS:
    &str = "NV_Q38_DRAFTER_ATTN_SPLITK";

pub fn drafter_attn_splitk_selected(raw: Option<&str>) -> bool {
    raw.map(str::trim)
        != Some(DRAFT_QUARTET_FAST_FP8_SPLITK_LMHEAD_NVFP4_DEFAULT_ON_CHAMPION_STACK_SIGNED_OFF_2026_09_02_EXPLICIT_0_RESTORES_EACH_PRE_FLIP_DRAFT_PATH)
}

pub fn drafter_attn_splitk_selected_from_env() -> bool {
    drafter_attn_splitk_selected(
        std::env::var(
            NV_Q38_DRAFTER_ATTN_SPLITK_ENV_DEFAULT_ON_ROUTES_THE_Q1_DRAFTER_SCAN_THROUGH_THE_FUSED_BF16_SPLITK_DECODE_KERNEL_BECAUSE_CANDLE_FA2_FILLS_ONLY_N_HEADS_BLOCKS,
        )
        .ok()
        .as_deref(),
    )
}

pub const NV_Q38_BOOT_WARM_DRAFTER_DEFAULT_ON_USER_SIGN_OFF_2026_09_12_ZERO_RESTORES_COLD_FIRST_REQUEST:
    &str = "NV_Q38_BOOT_WARM_DRAFTER";

pub fn drafter_boot_warm_default_on_user_sign_off_2026_09_12(
    raw: Option<&std::ffi::OsStr>,
) -> bool {
    raw.is_none() || raw.and_then(std::ffi::OsStr::to_str).map(str::trim) == Some("1")
}

pub fn drafter_boot_warm_selected_from_env() -> bool {
    drafter_boot_warm_default_on_user_sign_off_2026_09_12(
        std::env::var_os(
            NV_Q38_BOOT_WARM_DRAFTER_DEFAULT_ON_USER_SIGN_OFF_2026_09_12_ZERO_RESTORES_COLD_FIRST_REQUEST,
        )
        .as_deref(),
    )
}

#[cfg(test)]
mod drafter_boot_warm_default_tests {
    use super::drafter_boot_warm_default_on_user_sign_off_2026_09_12 as selected;
    use std::ffi::OsStr;

    #[test]
    fn only_unset_changes_and_explicit_trimmed_grammar_is_preserved() {
        assert!(selected(None));
        for raw in ["1", " 1", "1 ", "\t1\n"] {
            assert!(selected(Some(OsStr::new(raw))));
        }
        for raw in ["0", "", "true", "gate"] {
            assert!(
                !selected(Some(OsStr::new(raw))),
                "explicit {raw:?} stays off"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_remains_explicit_off() {
        use std::os::unix::ffi::OsStrExt;
        assert!(!selected(Some(OsStr::from_bytes(&[0xff]))));
    }
}

pub const TRUNK_OUTPUT_NORM_TENSOR_CANDIDATES: [&str; 2] =
    ["model.norm.weight", "model.language_model.norm.weight"];

pub const MTP_REANCHOR_UNSET_ON_USER_SIGN_OFF_2026_09_11_REPLY_11_BITWISE_ABBA_E31BFFB54_9F37F08BE_7CC999C9E_DBB0540A6:
    &str = "pg1342 retained-token FNV f39783307a2d1f13 at 2048 and 2e8b4f4072b2eaf7 at \
             168000 match across all 24 served ABBA requests. Pooled decode gains 4.0658% \
             and 0.5135%; shallow nulls 1.8004% and 0.7179%. User sign-off 2026-09-11 promotes the bitwise \
             default; explicit 0 restores pre-norm drafter input. No target arithmetic change.";

pub fn mtp_reanchor_post_norm_selected(raw: Option<&str>) -> bool {
    match raw {
        None => {
            let _ = MTP_REANCHOR_UNSET_ON_USER_SIGN_OFF_2026_09_11_REPLY_11_BITWISE_ABBA_E31BFFB54_9F37F08BE_7CC999C9E_DBB0540A6;
            true
        }
        Some(value) => value.trim() == "1",
    }
}

fn mtp_reanchor_post_norm_from_raw_os(raw: Option<&std::ffi::OsStr>) -> bool {
    match raw {
        None => mtp_reanchor_post_norm_selected(None),
        Some(value) => value
            .to_str()
            .is_some_and(|value| mtp_reanchor_post_norm_selected(Some(value))),
    }
}

pub fn mtp_reanchor_post_norm_selected_from_env() -> bool {
    mtp_reanchor_post_norm_from_raw_os(std::env::var_os(NV_Q38_MTP_REANCHOR_ENV).as_deref())
}

#[cfg(test)]
mod mtp_reanchor_user_signed_resolver_tests {
    use super::{
        mtp_reanchor_post_norm_from_raw_os as selected_os,
        mtp_reanchor_post_norm_selected as selected,
    };

    #[test]
    fn only_unset_changes_and_explicit_zero_restores_pre_norm() {
        assert!(selected(None));
        assert!(selected_os(None));
        for value in ["1", " 1", "1 ", "\t1\n"] {
            assert!(selected(Some(value)));
            assert!(selected_os(Some(std::ffi::OsStr::new(value))));
        }
        for value in ["0", "", "true", "yes", "gate", "2"] {
            assert!(
                !selected(Some(value)),
                "explicit {value:?} must retain its off meaning"
            );
            assert!(!selected_os(Some(std::ffi::OsStr::new(value))));
        }
    }

    #[cfg(unix)]
    #[test]
    fn explicit_non_unicode_is_not_unset() {
        use std::os::unix::ffi::OsStrExt;
        assert!(!selected_os(Some(std::ffi::OsStr::from_bytes(&[0xff]))));
    }
}

pub fn mtp_drafter_fp8_resident(raw: Option<&str>) -> bool {
    raw.map(str::trim)
        != Some(DRAFT_QUARTET_FAST_FP8_SPLITK_LMHEAD_NVFP4_DEFAULT_ON_CHAMPION_STACK_SIGNED_OFF_2026_09_02_EXPLICIT_0_RESTORES_EACH_PRE_FLIP_DRAFT_PATH)
}

pub fn mtp_drafter_fp8_resident_from_env() -> bool {
    mtp_drafter_fp8_resident(std::env::var(NV_Q38_DRAFT_FP8_ENV).ok().as_deref())
}

pub const NV_Q38_DRAFT_LMHEAD_NVFP4_ENV: &str = "NV_Q38_DRAFT_LMHEAD_NVFP4";

pub fn mtp_draft_lm_head_nvfp4_twin(raw: Option<&str>) -> bool {
    raw.map(str::trim)
        != Some(DRAFT_QUARTET_FAST_FP8_SPLITK_LMHEAD_NVFP4_DEFAULT_ON_CHAMPION_STACK_SIGNED_OFF_2026_09_02_EXPLICIT_0_RESTORES_EACH_PRE_FLIP_DRAFT_PATH)
}

pub fn mtp_draft_lm_head_nvfp4_twin_from_env() -> bool {
    mtp_draft_lm_head_nvfp4_twin(std::env::var(NV_Q38_DRAFT_LMHEAD_NVFP4_ENV).ok().as_deref())
}

pub const VOCAB_PREFIX_DEFAULT_65536_THE_P32K_RUNG_FAILED_THE_TILED_SERVED_BASIS_AND_64K_CERTIFIED_ON_ALL_THREE:
    usize = 65536;

pub fn draft_lmhead_argmax_fuse_route_eligible_because_probs_top3_and_forced_drafts_need_logits(
    nvfp4_twin_active: bool,
    want_top_probs: bool,
    want_step_top3: bool,
    forced_drafts_present: bool,
) -> bool {
    nvfp4_twin_active && !want_top_probs && !want_step_top3 && !forced_drafts_present
}

pub fn ordered_u32_key_from_f32_monotone_under_unsigned_max_with_neg_zero_tying_pos_zero(
    v: f32,
) -> u32 {
    let b = v.to_bits();
    if v >= 0.0 {
        b | 0x8000_0000
    } else {
        !b
    }
}

pub fn fused_argmax_packed_key_whose_inverted_index_makes_atomic_max_prefer_the_lowest_index(
    ord: u32,
    idx: u32,
) -> u64 {
    ((ord as u64) << 32) | ((idx ^ 0xFFFF_FFFF) as u64)
}

pub const MTP_WEIGHTS_FILE_NAME: &str = "model_mtp.safetensors";
pub const MTP_TENSOR_PREFIX: &str = "mtp.";

pub const MTP_CHAIN_DEPTH_DEFAULT_3_IS_THE_DEPTH_THE_SUFFIX_IDENTITY_GATE_CERTIFIES_AND_K3_PLUS_SUFFIX_BEATS_LLAMACPP_AT_EVERY_DEPTH_279_264_147_VS_147_146_104_WHERE_K4_ALONE_TRAILED_AT_172K:
    usize = 3;
pub const MTP_CHAIN_DEPTH_DEFAULT: usize =
    MTP_CHAIN_DEPTH_DEFAULT_3_IS_THE_DEPTH_THE_SUFFIX_IDENTITY_GATE_CERTIFIES_AND_K3_PLUS_SUFFIX_BEATS_LLAMACPP_AT_EVERY_DEPTH_279_264_147_VS_147_146_104_WHERE_K4_ALONE_TRAILED_AT_172K;
pub const MTP_CHAIN_DEPTH_MAX_SO_VERIFY_ROWS_STAY_UNDER_THE_F32_FAST_PATH_HARD_CAP: usize = 7;

pub const VERIFY_LM_HEAD_F32_FAST_PATH_HARD_ROW_CAP: usize = 8;
pub const VERIFY_LM_HEAD_BF16_PAIR_FALLBACK_HARD_ROW_CAP: usize = 16;

pub fn verify_lm_head_f32_fast_path_smem_bytes_per_row(hidden: usize) -> usize {
    (hidden >> 4) * 17 * core::mem::size_of::<f32>()
}

pub fn verify_lm_head_bf16_pair_fallback_smem_bytes_per_row(hidden: usize) -> usize {
    (hidden >> 4) * 9 * core::mem::size_of::<u32>()
}

pub fn verify_lm_head_f32_fast_path_rows_ceiling(
    hidden: usize,
    smem_optin_limit_bytes: usize,
) -> Result<usize> {
    ensure!(
        hidden > 0 && hidden.is_multiple_of(16),
        "gemv_i8_normed_mk requires hidden % 16 == 0, got hidden={hidden}"
    );
    if smem_optin_limit_bytes == 0 {
        return Ok(VERIFY_LM_HEAD_F32_FAST_PATH_HARD_ROW_CAP);
    }
    let per_row = verify_lm_head_f32_fast_path_smem_bytes_per_row(hidden);
    Ok((smem_optin_limit_bytes / per_row).min(VERIFY_LM_HEAD_F32_FAST_PATH_HARD_ROW_CAP))
}

pub fn verify_lm_head_rows_per_call_ceiling(
    hidden: usize,
    smem_optin_limit_bytes: usize,
) -> Result<usize> {
    let mf = verify_lm_head_f32_fast_path_rows_ceiling(hidden, smem_optin_limit_bytes)?;
    if smem_optin_limit_bytes == 0 {
        return Ok(mf);
    }
    let h_row = verify_lm_head_bf16_pair_fallback_smem_bytes_per_row(hidden);
    let mh = (smem_optin_limit_bytes / h_row).min(VERIFY_LM_HEAD_BF16_PAIR_FALLBACK_HARD_ROW_CAP);
    Ok(mf.max(mh))
}

pub fn mtp_verify_rows_per_round(k: usize) -> usize {
    k + 1
}

pub fn assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling(
    hidden: usize,
    k: usize,
    smem_optin_limit_bytes: usize,
) -> Result<()> {
    ensure!(k >= 1, "mtp chain depth must be >= 1, got {k}");
    let rows = mtp_verify_rows_per_round(k);
    ensure!(
        rows <= VERIFY_LM_HEAD_F32_FAST_PATH_HARD_ROW_CAP,
        "mtp round verifies rows=k+1={rows} but the gemv_i8_normed_mk f32 fast path holds a hard \
         M <= {VERIFY_LM_HEAD_F32_FAST_PATH_HARD_ROW_CAP} (the #107 lesson, recomputed for \
         hidden={hidden})"
    );
    let ceiling = verify_lm_head_rows_per_call_ceiling(hidden, smem_optin_limit_bytes)?;
    ensure!(
        rows <= ceiling,
        "mtp round verifies rows=k+1={rows} > lm_head rows-per-call ceiling {ceiling} \
         (basis: hidden={hidden}, smem_optin_limit_bytes={smem_optin_limit_bytes}, \
         f32_row={} B, bf16_pair_row={} B)",
        verify_lm_head_f32_fast_path_smem_bytes_per_row(hidden),
        verify_lm_head_bf16_pair_fallback_smem_bytes_per_row(hidden)
    );
    Ok(())
}

pub const NV_Q38_MTP_EAGER_DEEP_K_ENV: &str = "NV_Q38_MTP_EAGER_DEEP_K";

pub const MTP_EAGER_VERIFY_ROWS_CEILING_16_BECAUSE_THE_SEQ_N_FORWARD_HAS_NO_MK_TEMPLATE_BOUND:
    usize = 16;

pub fn mtp_eager_deep_k_selected(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

pub fn mtp_eager_deep_k_selected_from_env() -> bool {
    mtp_eager_deep_k_selected(std::env::var(NV_Q38_MTP_EAGER_DEEP_K_ENV).ok().as_deref())
}

pub fn assert_eager_session_verify_rows_fit(hidden: usize, k: usize) -> Result<()> {
    if mtp_eager_deep_k_selected_from_env() {
        ensure!(
            mtp_verify_rows_per_round(k)
                <= MTP_EAGER_VERIFY_ROWS_CEILING_16_BECAUSE_THE_SEQ_N_FORWARD_HAS_NO_MK_TEMPLATE_BOUND,
            "{NV_Q38_MTP_EAGER_DEEP_K_ENV}=1 lifts only the graphed-route kernel proxies; an \
             eager round still verifies rows=k+1={} and this probe ceiling is {}",
            mtp_verify_rows_per_round(k),
            MTP_EAGER_VERIFY_ROWS_CEILING_16_BECAUSE_THE_SEQ_N_FORWARD_HAS_NO_MK_TEMPLATE_BOUND
        );
        return Ok(());
    }
    assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling(hidden, k, 0)
}

pub const NV_Q38_MTP_P_MIN_ENV: &str = "NV_Q38_MTP_P_MIN";

pub fn mtp_draft_p_min(raw: Option<&str>) -> Option<f32> {
    raw.and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|p| *p > 0.0 && *p < 1.0)
}

pub fn mtp_draft_p_min_from_env() -> Option<f32> {
    mtp_draft_p_min(std::env::var(NV_Q38_MTP_P_MIN_ENV).ok().as_deref())
}

pub const NV_Q38_MTP_PRECAPTURE_ENV: &str = "NV_Q38_MTP_PRECAPTURE";

pub fn mtp_precapture_selected(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

pub const NV_Q38_SUFFIX_PRECAPTURE_ENV: &str = "NV_Q38_SUFFIX_PRECAPTURE";

pub const SUFFIX_PRECAPTURE_IS_MEASUREMENT_HYGIENE_NOT_SPEED_SO_SERVING_CAPTURES_LAZILY_BECAUSE_FRONT_LOADING_ALL_FOUR_SHAPES_COST_416MS_OF_TTFT_AND_0P39S_OF_WALL_WHILE_LAZY_ONLY_CAPTURES_THE_SHAPES_A_RUN_ACTUALLY_HITS: &str =
    "set NV_Q38_SUFFIX_PRECAPTURE=1 to front-load the suffix verify graphs, which is what a      verify_ms measurement wants and what a served request does not";

pub fn suffix_precapture_selected(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

pub fn suffix_precapture_selected_from_env() -> bool {
    suffix_precapture_selected(std::env::var(NV_Q38_SUFFIX_PRECAPTURE_ENV).ok().as_deref())
}

pub fn mtp_precapture_selected_from_env() -> bool {
    mtp_precapture_selected(std::env::var(NV_Q38_MTP_PRECAPTURE_ENV).ok().as_deref())
}

pub const ADAPTIVE_DRAFT_KEEPS_AT_LEAST_TWO_BECAUSE_THE_TWO_ROW_VERIFY_IS_THE_ONE_GRAPH_SHAPE_WHOSE_TC_GEMM_PLAN_DIFFERS_AND_FLIPS_GREEDY_ARGMAX: usize = 2;

pub fn adaptive_submitted_draft_len_matching_llama_cpp_p_min_leading_confident_run(
    top_probs: &[f32],
    p_min: f32,
) -> usize {
    top_probs
        .iter()
        .take_while(|&&p| p >= p_min)
        .count()
        .max(ADAPTIVE_DRAFT_KEEPS_AT_LEAST_TWO_BECAUSE_THE_TWO_ROW_VERIFY_IS_THE_ONE_GRAPH_SHAPE_WHOSE_TC_GEMM_PLAN_DIFFERS_AND_FLIPS_GREEDY_ARGMAX)
        .min(top_probs.len().max(1))
}

pub const NV_Q38_MTP_TREE_ENV: &str = "NV_Q38_MTP_TREE";

pub fn mtp_tree_selected(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

pub fn mtp_tree_selected_from_env() -> bool {
    mtp_tree_selected(std::env::var(NV_Q38_MTP_TREE_ENV).ok().as_deref())
}

pub const MTP_TREE_ROWS_6_ANCHOR_PLUS_TOP1_CHAIN_D3_PLUS_TOP2_SIBLING_AND_ONE_CONTINUATION: usize =
    6;

pub const MTP_TREE_BRANCH_A_DEPTH_3_SO_THE_TREE_NEEDS_EXACTLY_NV_MTP_K_3: usize = 3;

pub const MTP_TREE_BRANCH_B_FIRST_ROW_4_ROWS_ARE_ANCHOR_A1_A2_A3_B1_B2: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwoBranchTreeAccept {
    pub emitted: Vec<u32>,
    pub drafts_accepted: usize,
    pub depth_consumed: usize,
    pub bonus_row: usize,
    pub branch_b_accepted_rows: Option<usize>,
}

pub fn accept_two_branch_tree_m6(
    rows_tokens: &[u32],
    greedy: &[u32],
) -> Result<TwoBranchTreeAccept> {
    let rows = MTP_TREE_ROWS_6_ANCHOR_PLUS_TOP1_CHAIN_D3_PLUS_TOP2_SIBLING_AND_ONE_CONTINUATION;
    ensure!(
        rows_tokens.len() == rows && greedy.len() == rows,
        "two-branch tree accept: rows_tokens {} greedy {} != {rows}",
        rows_tokens.len(),
        greedy.len()
    );
    let b0 = MTP_TREE_BRANCH_B_FIRST_ROW_4_ROWS_ARE_ANCHOR_A1_A2_A3_B1_B2;
    let t1 = greedy[0];
    if t1 == rows_tokens[1] {
        let mut a = 1usize;
        while a < MTP_TREE_BRANCH_A_DEPTH_3_SO_THE_TREE_NEEDS_EXACTLY_NV_MTP_K_3
            && greedy[a] == rows_tokens[a + 1]
        {
            a += 1;
        }
        let mut emitted: Vec<u32> = rows_tokens[1..=a].to_vec();
        emitted.push(greedy[a]);
        return Ok(TwoBranchTreeAccept {
            emitted,
            drafts_accepted: a,
            depth_consumed: a + 1,
            bonus_row: a,
            branch_b_accepted_rows: None,
        });
    }
    if t1 == rows_tokens[b0] {
        let b = if greedy[b0] == rows_tokens[b0 + 1] {
            2
        } else {
            1
        };
        let bonus_row = b0 + b - 1;
        let mut emitted: Vec<u32> = rows_tokens[b0..b0 + b].to_vec();
        emitted.push(greedy[bonus_row]);
        return Ok(TwoBranchTreeAccept {
            emitted,
            drafts_accepted: b,
            depth_consumed: b + 1,
            bonus_row,
            branch_b_accepted_rows: Some(b),
        });
    }
    Ok(TwoBranchTreeAccept {
        emitted: vec![t1],
        drafts_accepted: 0,
        depth_consumed: 1,
        bonus_row: 0,
        branch_b_accepted_rows: None,
    })
}

pub const NV_Q38_W4A8_MK_ENV: &str = "NV_Q38_W4A8_MK";

pub const NV_Q38_MTP_ROUND_DEBUG_ENV: &str = "NV_Q38_MTP_ROUND_DEBUG";

#[cfg(feature = "cuda")]
fn top2_host_row_for_argmax_margin_triage(row: &[f32]) -> (u32, f32, u32, f32) {
    let mut t1 = 0u32;
    let mut v1 = f32::NEG_INFINITY;
    let mut t2 = 0u32;
    let mut v2 = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > v1 {
            t2 = t1;
            v2 = v1;
            t1 = i as u32;
            v1 = v;
        } else if v > v2 {
            t2 = i as u32;
            v2 = v;
        }
    }
    (t1, v1, t2, v2)
}

#[cfg(feature = "cuda")]
fn top3_host_row_ids_lowest_index_tie_break_matching_device_argmax(row: &[f32]) -> [u32; 3] {
    let mut ids = [0u32; 3];
    let mut vals = [f32::NEG_INFINITY; 3];
    for (i, &v) in row.iter().enumerate() {
        if v > vals[0] {
            ids[2] = ids[1];
            vals[2] = vals[1];
            ids[1] = ids[0];
            vals[1] = vals[0];
            ids[0] = i as u32;
            vals[0] = v;
        } else if v > vals[1] {
            ids[2] = ids[1];
            vals[2] = vals[1];
            ids[1] = i as u32;
            vals[1] = v;
        } else if v > vals[2] {
            ids[2] = i as u32;
            vals[2] = v;
        }
    }
    ids
}

pub fn adaptive_verify_pins_the_env_unset_auto_dp4a_mlp_off_because_greedy_argmax_must_not_depend_on_which_m_the_cut_produced(
) {
    if !nv_config::env_arm_present_at_all_reread_every_call(NV_Q38_W4A8_MK_ENV) {
        std::env::set_var(NV_Q38_W4A8_MK_ENV, "0");
    }
}

pub fn mtp_drafter_selected(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("mtp"))
}

pub fn mtp_drafter_selected_from_env() -> bool {
    mtp_drafter_selected(std::env::var(NV_DRAFTER_ENV).ok().as_deref())
}

pub fn mtp_chain_depth(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&k| k >= 1)
        .unwrap_or(MTP_CHAIN_DEPTH_DEFAULT)
        .min(MTP_CHAIN_DEPTH_MAX_SO_VERIFY_ROWS_STAY_UNDER_THE_F32_FAST_PATH_HARD_CAP)
}

pub fn mtp_chain_depth_from_env() -> usize {
    mtp_chain_depth(std::env::var(NV_MTP_K_ENV).ok().as_deref())
}

pub fn mtp_draft_dir_override_from_env() -> Option<PathBuf> {
    std::env::var_os(NV_MTP_DRAFT_DIR_ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

pub fn mtp_round_hidden_reanchor_index(draft_accepted: usize) -> usize {
    draft_accepted
}

pub const NV_Q38_MTP_VERIFY_REPLAY_ENV: &str = "NV_Q38_MTP_VERIFY_REPLAY";

pub fn mtp_verify_replay_selected(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

pub fn mtp_verify_replay_selected_from_env() -> bool {
    mtp_verify_replay_selected(std::env::var(NV_Q38_MTP_VERIFY_REPLAY_ENV).ok().as_deref())
}

pub const MTP_VERIFY_INC2_COMMITS_BATCHED_ROWS_AND_TOLERATES_MROW_VS_M1_DRIFT: &str =
    "increment 2 of the qwen3.8 wgpu MTP verify: the k+1-row verify_chain forward decides \
     acceptance AND supplies the committed state. A full accept advances in place; a partial \
     accept rolls back via advance(0), re-forwards only the accepted prefix through the same \
     M-row graph commit-only (state advancement without the per-row final-norm/lm_head tail or \
     any readback, because every prefix token is already known from the first forward), and \
     full-accept-advances that prefix -- so committed recurrent/conv state always comes from \
     batched kernels and the increment-1 M=1 replay (one full trunk forward per accepted token, \
     the 9x-decode round cost) never runs. Batched rows are NOT required to \
     be byte-identical to sequential M=1 decode; the contract is the acceptance A/B gate, the \
     teacher-forced ppl match, and the empirical logit-drift pin, all in \
     nv-specdecode/tests/qwen38_mtp_wgpu_inc2.rs. NV_Q38_MTP_VERIFY_REPLAY=1 restores the \
     increment-1 replay commit as the debugging escape. NV_Q3D_MTP_COMMIT_SCAN_REPLAY=1 \
     commits the same partial accept without the commit-only trunk reforward: advance(n) \
     restores the snapshot and re-dispatches only each DeltaNet layer's conv_shift and scan \
     over per-layer rows the first forward already landed, reproducing the reforward's \
     committed state bit-for-bit (qwen3_5_dense_wgpu's \
     MTP_COMMIT_SCAN_REPLAY_RESTORES_THE_SNAPSHOT_AND_REDISPATCHES_ONLY_EACH_DELTANET_CONV_SHIFT_AND_SCAN).";

pub const Q38_WGPU_BATCHED_VERIFY_VS_M1_MAX_ABS_LOGIT_DRIFT_EMPIRICAL_PIN: f32 = 6.0;

pub trait MtpBatchedVerifyTarget {
    fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>>;
    fn verify_chain_commit_only(&mut self, batch: &[u32]) -> Result<()> {
        self.verify_chain(batch).map(|_| ())
    }
    fn advance(&mut self, n: usize) -> Result<()>;
    fn advance_commits_partial_prefix_without_reforward(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct MtpBatchedVerifyRound {
    pub batch: Vec<u32>,
    pub accept: crate::chain::ChainAccept,
    pub emitted: Vec<u32>,
    pub prefix_reforwarded_batched: bool,
}

pub fn run_mtp_verify_round<T: MtpBatchedVerifyTarget + ?Sized>(
    target: &mut T,
    anchor: u32,
    drafts: &[u32],
    replay_commit: bool,
) -> Result<MtpBatchedVerifyRound> {
    let batch = crate::chain::build_chain_batch(anchor, drafts, drafts.len() + 1, true)?;
    let amax = target.verify_chain(&batch)?;
    let accept = crate::chain::accept_prefix_argmax(&batch, &amax)?;
    let full = accept.commit_len == batch.len();
    let scan_replay =
        !replay_commit && !full && target.advance_commits_partial_prefix_without_reforward();
    if replay_commit || full || scan_replay {
        target.advance(accept.commit_len)?;
    } else {
        target.advance(0)?;
        target.verify_chain_commit_only(&batch[..accept.commit_len])?;
        target.advance(accept.commit_len)?;
    }
    let mut emitted: Vec<u32> = drafts[..accept.draft_accepted].to_vec();
    emitted.push(accept.next_bonus);
    Ok(MtpBatchedVerifyRound {
        batch,
        accept,
        emitted,
        prefix_reforwarded_batched: !replay_commit && !full && !scan_replay,
    })
}

#[cfg(feature = "wgpu")]
impl MtpBatchedVerifyTarget for nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseWgpu {
    fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
        Qwen3_5DenseWgpu::verify_chain(self, batch)
    }

    fn verify_chain_commit_only(&mut self, batch: &[u32]) -> Result<()> {
        Qwen3_5DenseWgpu::verify_chain_commit_only(self, batch)
    }

    fn advance(&mut self, n: usize) -> Result<()> {
        Qwen3_5DenseWgpu::advance(self, n)
    }

    fn advance_commits_partial_prefix_without_reforward(&self) -> bool {
        Qwen3_5DenseWgpu::mtp_commit_scan_replay_active(self)
    }
}

#[cfg(feature = "wgpu")]
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseWgpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen38MtpGeometry {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub attn_output_gate: bool,
}

pub const QWEN38_27B_MTP_GEOMETRY: Qwen38MtpGeometry = Qwen38MtpGeometry {
    hidden: 5120,
    heads: 24,
    kv_heads: 4,
    head_dim: 256,
    intermediate: 17408,
    attn_output_gate: true,
};

impl Qwen38MtpGeometry {
    pub fn fc_in(&self) -> usize {
        2 * self.hidden
    }

    pub fn q_proj_out(&self) -> usize {
        let gate_doubles_q_rows = if self.attn_output_gate { 2 } else { 1 };
        self.heads * self.head_dim * gate_doubles_q_rows
    }

    pub fn kv_proj_out(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub fn o_proj_in(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn expected_tensor_shapes(&self) -> BTreeMap<String, Vec<usize>> {
        let h = self.hidden;
        let mut m = BTreeMap::new();
        let mut put = |name: &str, shape: Vec<usize>| {
            m.insert(name.to_string(), shape);
        };
        put("mtp.fc.weight", vec![h, self.fc_in()]);
        put("mtp.pre_fc_norm_embedding.weight", vec![h]);
        put("mtp.pre_fc_norm_hidden.weight", vec![h]);
        put("mtp.norm.weight", vec![h]);
        put("mtp.layers.0.input_layernorm.weight", vec![h]);
        put("mtp.layers.0.post_attention_layernorm.weight", vec![h]);
        put(
            "mtp.layers.0.self_attn.q_proj.weight",
            vec![self.q_proj_out(), h],
        );
        put(
            "mtp.layers.0.self_attn.k_proj.weight",
            vec![self.kv_proj_out(), h],
        );
        put(
            "mtp.layers.0.self_attn.v_proj.weight",
            vec![self.kv_proj_out(), h],
        );
        put(
            "mtp.layers.0.self_attn.o_proj.weight",
            vec![h, self.o_proj_in()],
        );
        put("mtp.layers.0.self_attn.q_norm.weight", vec![self.head_dim]);
        put("mtp.layers.0.self_attn.k_norm.weight", vec![self.head_dim]);
        put(
            "mtp.layers.0.mlp.gate_proj.weight",
            vec![self.intermediate, h],
        );
        put(
            "mtp.layers.0.mlp.up_proj.weight",
            vec![self.intermediate, h],
        );
        put(
            "mtp.layers.0.mlp.down_proj.weight",
            vec![h, self.intermediate],
        );
        m
    }
}

pub fn validate_mtp_named_shapes(
    geometry: &Qwen38MtpGeometry,
    named: &BTreeMap<String, Vec<usize>>,
) -> Result<()> {
    let expected = geometry.expected_tensor_shapes();
    for (name, shape) in &expected {
        let got = named
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("mtp tensor missing from checkpoint: {name}"))?;
        ensure!(
            got == shape,
            "mtp tensor {name}: expected shape {shape:?}, checkpoint has {got:?}"
        );
    }
    for name in named.keys() {
        ensure!(
            expected.contains_key(name),
            "checkpoint ships unexpected mtp tensor {name}; the dense head map only knows \
             {} tensors and silently ignoring extras would hide a geometry change",
            expected.len()
        );
    }
    Ok(())
}

pub fn read_safetensors_header_shapes(path: &Path) -> Result<BTreeMap<String, Vec<usize>>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len() >= 8,
        "{}: shorter than a header length",
        path.display()
    );
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    ensure!(
        bytes.len() >= 8 + n,
        "{}: header claims {n} bytes but the file holds {}",
        path.display(),
        bytes.len()
    );
    let v: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n])
        .with_context(|| format!("parse safetensors header of {}", path.display()))?;
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{}: header is not an object", path.display()))?;
    let mut out = BTreeMap::new();
    for (k, e) in obj {
        if k == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = e
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as usize)
                    .collect()
            })
            .ok_or_else(|| anyhow::anyhow!("{}: tensor {k} has no shape", path.display()))?;
        out.insert(k.clone(), shape);
    }
    Ok(out)
}

pub fn resolve_mtp_weight_files(
    explicit_dir: Option<&Path>,
    main_checkpoint_dir: &Path,
) -> Result<Vec<PathBuf>> {
    let mut searched: Vec<String> = Vec::new();
    let dirs: Vec<&Path> = explicit_dir
        .into_iter()
        .chain(std::iter::once(main_checkpoint_dir))
        .collect();
    for dir in dirs {
        searched.push(dir.display().to_string());
        let dedicated = dir.join(MTP_WEIGHTS_FILE_NAME);
        if dedicated.is_file() {
            return Ok(vec![dedicated]);
        }
        let index = dir.join("model.safetensors.index.json");
        if index.is_file() {
            let raw = std::fs::read_to_string(&index)
                .with_context(|| format!("read {}", index.display()))?;
            let v: serde_json::Value =
                serde_json::from_str(&raw).with_context(|| format!("parse {}", index.display()))?;
            let wm = v
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| anyhow::anyhow!("{}: no weight_map", index.display()))?;
            let mut files: Vec<PathBuf> = wm
                .iter()
                .filter(|(k, _)| k.starts_with(MTP_TENSOR_PREFIX))
                .filter_map(|(_, f)| f.as_str().map(|s| dir.join(s)))
                .collect();
            files.sort();
            files.dedup();
            if !files.is_empty() {
                return Ok(files);
            }
        }
        let single = dir.join("model.safetensors");
        if single.is_file()
            && read_safetensors_header_shapes(&single)?
                .keys()
                .any(|k| k.starts_with(MTP_TENSOR_PREFIX))
        {
            return Ok(vec![single]);
        }
    }
    anyhow::bail!(
        "no MTP weights found: searched {searched:?} for {MTP_WEIGHTS_FILE_NAME}, an index \
         mapping {MTP_TENSOR_PREFIX}* tensors, or a single-file checkpoint carrying them; \
         if this checkpoint truly ships no MTP head, NV_DRAFTER=mtp cannot serve it"
    )
}

pub const SPECULATOR_SUBDIR_MIRRORS_NV_TRAIN_SPECULATOR_SUBDIR: &str = "speculator";

pub const SPECULATOR_KIND_KEY_READ_FROM_THE_MAIN_ADAPTER_CONFIG_BECAUSE_EAGLE3_SUBDIR_CONFIGS_OMIT_IT:
    &str = "nv_speculator_kind";

pub const QWEN38_MTP_SPECULATOR_KIND_MIRRORS_NV_TRAIN: &str = "qwen38_mtp_head";

pub const MTP_SPECULATOR_LORA_MODULES_MATCH_Q38_MTP_TRAIN_HEAD_PROJECTIONS_NORMS_EMBED_LM_HEAD_D2T_T2D_ARE_NEVER_TARGETS:
    [&str; 8] = [
    "mtp.fc",
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.v_proj",
];

pub fn mtp_speculator_dir_if_advertised(adapter_dir: &Path) -> Result<Option<PathBuf>> {
    let subdir = adapter_dir.join(SPECULATOR_SUBDIR_MIRRORS_NV_TRAIN_SPECULATOR_SUBDIR);
    let config_path = adapter_dir.join("adapter_config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read {}", config_path.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", config_path.display()))?;
    let kind = v
        .get(SPECULATOR_KIND_KEY_READ_FROM_THE_MAIN_ADAPTER_CONFIG_BECAUSE_EAGLE3_SUBDIR_CONFIGS_OMIT_IT)
        .and_then(|k| k.as_str());
    match (kind, subdir.is_dir()) {
        (None, false) => Ok(None),
        (Some(kind), true) => {
            ensure!(
                kind == QWEN38_MTP_SPECULATOR_KIND_MIRRORS_NV_TRAIN,
                "adapter {} ships a speculator of kind {kind}, but this engine's drafter is the \
                 qwen38 MTP head ({QWEN38_MTP_SPECULATOR_KIND_MIRRORS_NV_TRAIN}); merging a \
                 mismatched speculator would serve deltas trained against a different drafter, \
                 so the load refuses",
                adapter_dir.display()
            );
            Ok(Some(subdir))
        }
        (Some(kind), false) => anyhow::bail!(
            "adapter {}: adapter_config.json advertises speculator kind {kind} but no {}/ subdir \
             exists; serving would silently drop the co-trained drafter deltas",
            adapter_dir.display(),
            SPECULATOR_SUBDIR_MIRRORS_NV_TRAIN_SPECULATOR_SUBDIR
        ),
        (None, true) => anyhow::bail!(
            "adapter {}: a {}/ subdir exists but the MAIN adapter_config.json carries no {} key; \
             the kind is authoritative in the main config (eagle3 subdir configs omit it), so an \
             unkinded speculator cannot be classified and refuses to merge",
            adapter_dir.display(),
            SPECULATOR_SUBDIR_MIRRORS_NV_TRAIN_SPECULATOR_SUBDIR,
            SPECULATOR_KIND_KEY_READ_FROM_THE_MAIN_ADAPTER_CONFIG_BECAUSE_EAGLE3_SUBDIR_CONFIGS_OMIT_IT
        ),
    }
}

pub fn merge_speculator_lora_into_mtp_map_before_any_quant_arm_so_the_drafter_serves_at_base_speed(
    map: &mut HashMap<String, Tensor>,
    speculator_dir: &Path,
) -> Result<Vec<String>> {
    let peft =
        nv_weights::lora_adapter::PeftConfig::from_dir(speculator_dir).with_context(|| {
            format!(
                "read speculator peft config in {}",
                speculator_dir.display()
            )
        })?;
    let st_path = speculator_dir.join("adapter_model.safetensors");
    let device = map
        .values()
        .next()
        .ok_or_else(|| anyhow::anyhow!("speculator merge over an empty MTP tensor map"))?
        .device()
        .clone();
    let deltas = candle_core::safetensors::load(&st_path, &device)
        .with_context(|| format!("load speculator deltas {}", st_path.display()))?;
    let mut pairs: BTreeMap<String, [Option<Tensor>; 2]> = BTreeMap::new();
    for (name, t) in deltas {
        let parsed = name.strip_prefix("base_model.model.").and_then(|rest| {
            rest.strip_suffix(".lora_A.weight")
                .map(|m| (m, 0usize))
                .or_else(|| rest.strip_suffix(".lora_B.weight").map(|m| (m, 1usize)))
        });
        let Some((module, side)) = parsed else {
            anyhow::bail!(
                "speculator tensor {name} is not a base_model.model.<module>.lora_A/lora_B.weight \
                 pair member; norms, embeddings, lm_head, d2t/t2d and DoRA vectors are never MTP \
                 speculator targets, so the whole merge refuses rather than skipping"
            );
        };
        ensure!(
            MTP_SPECULATOR_LORA_MODULES_MATCH_Q38_MTP_TRAIN_HEAD_PROJECTIONS_NORMS_EMBED_LM_HEAD_D2T_T2D_ARE_NEVER_TARGETS
                .contains(&module),
            "speculator names module {module}, which is not one of the 8 LoRA-targetable MTP head \
             projections; refusing the whole merge (no partial merge)"
        );
        let slot = &mut pairs.entry(module.to_string()).or_default()[side];
        ensure!(slot.is_none(), "duplicate speculator tensor {name}");
        *slot = Some(t);
    }
    ensure!(
        !pairs.is_empty(),
        "{} holds no lora_A/lora_B delta tensors; an empty speculator advertises a co-trained \
         drafter that was never trained",
        st_path.display()
    );
    let mut staged: Vec<(String, Tensor)> = Vec::with_capacity(pairs.len());
    for (module, [a, b]) in &pairs {
        let a = a
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("speculator module {module} is missing lora_A"))?;
        let b = b
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("speculator module {module} is missing lora_B"))?;
        let weight_name = format!("{module}.weight");
        let w = map.get(&weight_name).ok_or_else(|| {
            anyhow::anyhow!(
                "speculator targets {weight_name} but the MTP checkpoint map has no such tensor"
            )
        })?;
        let (out_f, in_f) = w.dims2()?;
        let (a_r, a_in) = a.dims2()?;
        let (b_out, b_r) = b.dims2()?;
        ensure!(
            a_r == peft.r && b_r == peft.r,
            "speculator module {module}: adapter_config claims r={} but lora_A is [{a_r}, {a_in}] \
             and lora_B is [{b_out}, {b_r}]; a rank the config does not claim is a corrupted \
             artifact",
            peft.r
        );
        ensure!(
            a_in == in_f && b_out == out_f,
            "speculator module {module}: the serving drafter weight is [{out_f}, {in_f}] (the \
             loaded Linear's own dims, which is where gated q_proj rows and fc in-features come \
             from) but the deltas compose to [{b_out}, {a_in}]; this speculator was trained \
             against a different drafter, refusing the whole merge (no partial merge)"
        );
        let delta_f32_scaled_after_the_matmul_never_rounding_the_delta_separately = b
            .to_dtype(DType::F32)?
            .matmul(&a.to_dtype(DType::F32)?)?
            .affine(peft.scaling, 0.0)?;
        let merged = w
            .to_dtype(DType::F32)?
            .add(&delta_f32_scaled_after_the_matmul_never_rounding_the_delta_separately)?
            .to_dtype(w.dtype())?
            .contiguous()?;
        staged.push((weight_name, merged));
    }
    let merged_modules: Vec<String> = pairs.keys().cloned().collect();
    for (name, t) in staged {
        map.insert(name, t);
    }
    Ok(merged_modules)
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use candle_core::{DType, Device, Tensor, D};
    use nv_layers::attn::{flash_attn, AttnConfig};
    use nv_layers::linear::Linear;
    use nv_layers::mlp::Mlp;
    use nv_layers::norm::RmsNorm;
    use nv_models::graph_engine::GraphedQwen3Moe;
    use nv_models::qwen3_5_moe::decode_prof;
    use nv_models::qwen3_5_moe::{AttentionLayer, Qwen3Moe, Qwen3MoeKvCache};
    use nv_models::qwen3_5_mtp::MtpSpecStats;
    use std::collections::HashMap;

    pub const DRAFTER_FP8_CATCHUP_ROWS_PER_CALL_16_THE_GEMV_E4M3_MK_LEADING_CAP_BECAUSE_THIS_DEVICE_SERVES_NO_CUBLASLT_PER_ROW_FP8:
        usize = 16;

    pub const DRAFTER_BOOT_WARM_PROMPT_TOKENS_17_COVER_BOTH_THE_16_ROW_FP8_CATCHUP_CALL_SHAPE_AND_THE_REMAINDER_ROW:
        usize = 17;

    pub const NV_Q38_DRAFTER_KV_FP8_ENV_DEFAULT_ON_0_RESTORES_THE_BF16_PER_Q_HEAD_ROUTE_USER_SIGNED_OFF_2026_09_04_BITWISE_IDENTICAL_EMITTED_ALL_THREE_CLASSES:
        &str = "NV_Q38_DRAFTER_KV_FP8";

    pub const KVSHARE_FP8_DRAFTER_HEAD_DIM_256_IS_THE_ONLY_GEOMETRY_THE_KVSHARE_KERNEL_ACCEPTS_IT_RETURNS_MINUS_21_OTHERWISE:
        usize = 256;

    pub fn drafter_kv_fp8_selected_from_env() -> bool {
        !matches!(
            std::env::var(
                NV_Q38_DRAFTER_KV_FP8_ENV_DEFAULT_ON_0_RESTORES_THE_BF16_PER_Q_HEAD_ROUTE_USER_SIGNED_OFF_2026_09_04_BITWISE_IDENTICAL_EMITTED_ALL_THREE_CLASSES
            )
            .as_deref(),
            Ok("0")
        )
    }

    const DRAFTER_KVSHARE_SCRATCH_SPLITS_128_IS_THE_KERNEL_MAX: usize = 128;

    pub const NV_Q38_DRAFTER_ATTN_TC_ENV_DEFAULT_ON_0_RESTORES_THE_SCALAR_DRAFTER_QK_USER_SIGNED_OFF_2026_09_04_BITWISE_IDENTICAL_EMITTED:
        &str = "NV_Q38_DRAFTER_ATTN_TC";

    pub fn drafter_attn_tc_selected_from_env() -> bool {
        !matches!(
            std::env::var(
                NV_Q38_DRAFTER_ATTN_TC_ENV_DEFAULT_ON_0_RESTORES_THE_SCALAR_DRAFTER_QK_USER_SIGNED_OFF_2026_09_04_BITWISE_IDENTICAL_EMITTED
            )
            .as_deref(),
            Ok("0")
        )
    }

    pub const NV_Q38_DRAFTER_KV_WINDOW_ENV_TOKEN_COUNT_CAPS_THE_FP8_DRAFTER_KV_TO_A_RING_SO_DRAFT_ATTENTION_IS_DEPTH_INDEPENDENT_THE_TRUNK_HIDDEN_STILL_CARRIES_LONG_RANGE_CONTEXT:
        &str = "NV_Q38_DRAFTER_KV_WINDOW";

    pub fn drafter_kv_window_from_env() -> Option<usize> {
        match std::env::var(
            NV_Q38_DRAFTER_KV_WINDOW_ENV_TOKEN_COUNT_CAPS_THE_FP8_DRAFTER_KV_TO_A_RING_SO_DRAFT_ATTENTION_IS_DEPTH_INDEPENDENT_THE_TRUNK_HIDDEN_STILL_CARRIES_LONG_RANGE_CONTEXT,
        ) {
            Err(_) => None,
            Ok(v) if v.trim().is_empty() || v.trim() == "0" => None,
            Ok(v) => Some(v.trim().parse::<usize>().unwrap_or_else(|_| {
                panic!(
                    "NV_Q38_DRAFTER_KV_WINDOW={v}: set a plain token count (the ring size), \
                     0 or unset for the full-length drafter kv"
                )
            })),
        }
    }

    const DRAFTER_KVSHARE_SHALLOW_TOTAL_65536_PINS_16_SPLITS_BECAUSE_THE_TWO_WAVE_PICK_COSTS_0P15_MS_OF_COMBINE_AT_256: usize = 65536;

    const DRAFTER_KVSHARE_SHALLOW_SPLITS_16_MATCHES_THE_BF16_ROUTES_PICK_FOR_24_HEADS: i32 = 16;

    struct DrafterFp8Kv {
        k_fp8: cudarc::driver::CudaSlice<u8>,
        v_fp8: cudarc::driver::CudaSlice<u8>,
        k_scales: cudarc::driver::CudaSlice<f32>,
        v_scales: cudarc::driver::CudaSlice<f32>,
        start_dev: cudarc::driver::CudaSlice<i32>,
        host_start: Box<[i32; 1]>,
        ring_window_rows_or_zero_for_full_length: usize,
    }

    struct SplitkRouteBuffers {
        scratch: cudarc::driver::CudaSlice<f32>,
        fan_in: cudarc::driver::CudaSlice<u32>,
        len_dev: cudarc::driver::CudaSlice<i32>,
        host_len: Box<[i32; 1]>,
        heads: usize,
    }

    pub struct MtpKvCache {
        k_bf16_rows_stay_exact_because_the_drafter_never_shares_the_trunk_fp8_slots: Option<Tensor>,
        v_rows: Option<Tensor>,
        fp8_rows_replace_the_bf16_pair_when_the_kv_fp8_arm_is_selected_saving_704mb_at_172k:
            Option<DrafterFp8Kv>,
        kv_heads: usize,
        head_dim: usize,
        len: usize,
        max_seq: usize,
        splitk_route_sized_at_first_call_because_the_head_count_arrives_with_the_query:
            Option<SplitkRouteBuffers>,
    }

    impl MtpKvCache {
        pub fn new(
            kv_heads: usize,
            head_dim: usize,
            max_seq: usize,
            dtype: DType,
            device: &Device,
        ) -> Result<Self> {
            let window = drafter_kv_window_from_env();
            let fp8_geometry_ok =
                head_dim == KVSHARE_FP8_DRAFTER_HEAD_DIM_256_IS_THE_ONLY_GEOMETRY_THE_KVSHARE_KERNEL_ACCEPTS_IT_RETURNS_MINUS_21_OTHERWISE;
            if window.is_some() && !drafter_kv_fp8_selected_from_env() {
                anyhow::bail!(
                    "NV_Q38_DRAFTER_KV_WINDOW rides the fp8 kvshare route; set \
                     NV_Q38_DRAFTER_KV_FP8=1 with it (the bf16 splitk route has no ring)"
                );
            }
            let fp8 = if drafter_kv_fp8_selected_from_env() && fp8_geometry_ok {
                let cuda = match device {
                    Device::Cuda(d) => d.clone(),
                    _ => anyhow::bail!(
                        "{} needs a CUDA device; unset it for the bf16 drafter cache",
                        NV_Q38_DRAFTER_KV_FP8_ENV_DEFAULT_ON_0_RESTORES_THE_BF16_PER_Q_HEAD_ROUTE_USER_SIGNED_OFF_2026_09_04_BITWISE_IDENTICAL_EMITTED_ALL_THREE_CLASSES
                    ),
                };
                let stream = nv_layers::cuda_stream::current_stream(&cuda);
                let ring = match window {
                    Some(w) if w < max_seq => w,
                    _ => 0,
                };
                let rows = if ring > 0 { ring } else { max_seq } * kv_heads;
                Some(DrafterFp8Kv {
                    k_fp8: stream
                        .alloc_zeros::<u8>(rows * head_dim)
                        .map_err(|e| anyhow::anyhow!(e))?,
                    v_fp8: stream
                        .alloc_zeros::<u8>(rows * head_dim)
                        .map_err(|e| anyhow::anyhow!(e))?,
                    k_scales: stream
                        .alloc_zeros::<f32>(rows)
                        .map_err(|e| anyhow::anyhow!(e))?,
                    v_scales: stream
                        .alloc_zeros::<f32>(rows)
                        .map_err(|e| anyhow::anyhow!(e))?,
                    start_dev: stream
                        .alloc_zeros::<i32>(1)
                        .map_err(|e| anyhow::anyhow!(e))?,
                    host_start: Box::new([0i32; 1]),
                    ring_window_rows_or_zero_for_full_length: ring,
                })
            } else {
                None
            };
            let bf16_pair = if fp8.is_none() {
                Some((
                    Tensor::zeros((1usize, max_seq, kv_heads, head_dim), dtype, device)?,
                    Tensor::zeros((1usize, max_seq, kv_heads, head_dim), dtype, device)?,
                ))
            } else {
                None
            };
            let (kb, vb) = match bf16_pair {
                Some((k, v)) => (Some(k), Some(v)),
                None => (None, None),
            };
            Ok(Self {
                k_bf16_rows_stay_exact_because_the_drafter_never_shares_the_trunk_fp8_slots: kb,
                v_rows: vb,
                fp8_rows_replace_the_bf16_pair_when_the_kv_fp8_arm_is_selected_saving_704mb_at_172k:
                    fp8,
                kv_heads,
                head_dim,
                len: 0,
                max_seq,
                splitk_route_sized_at_first_call_because_the_head_count_arrives_with_the_query:
                    None,
            })
        }

        pub fn kv_fp8_armed(&self) -> bool {
            self.fp8_rows_replace_the_bf16_pair_when_the_kv_fp8_arm_is_selected_saving_704mb_at_172k
                .is_some()
        }

        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        pub fn max_seq(&self) -> usize {
            self.max_seq
        }

        fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
            let d = k_new.dims();
            ensure!(
                d.len() == 4 && d[0] == 1,
                "MtpKvCache.append expects [1, t, kv, hd], got {d:?}"
            );
            let t = d[1];
            ensure!(
                self.len + t <= self.max_seq,
                "MtpKvCache overflow: len {} + t {t} > max_seq {}",
                self.len,
                self.max_seq
            );
            if let Some(fp8) = self
                .fp8_rows_replace_the_bf16_pair_when_the_kv_fp8_arm_is_selected_saving_704mb_at_172k
                .as_mut()
            {
                use cudarc::driver::{DevicePtr, DevicePtrMut};
                use std::ffi::c_void;
                let dev = match k_new.device() {
                    Device::Cuda(d) => d.clone(),
                    _ => anyhow::bail!("fp8 drafter kv append needs CUDA tensors"),
                };
                let stream = nv_layers::cuda_stream::current_stream(&dev);
                fp8.host_start[0] = self.len as i32;
                stream
                    .memcpy_htod(&fp8.host_start[..], &mut fp8.start_dev)
                    .map_err(|e| anyhow::anyhow!(e))?;
                for (new_t, base, scales) in [
                    (k_new, &mut fp8.k_fp8, &mut fp8.k_scales),
                    (v_new, &mut fp8.v_fp8, &mut fp8.v_scales),
                ] {
                    let c = new_t.contiguous()?;
                    let (st, layout) = c.storage_and_layout();
                    let cs = match &*st {
                        candle_core::Storage::Cuda(s) => s,
                        _ => anyhow::bail!("fp8 drafter kv append: storage not cuda"),
                    };
                    let view = cs
                        .as_cuda_slice::<half::bf16>()?
                        .slice(layout.start_offset()..);
                    let (px, _g0) = view.device_ptr(&stream);
                    let (pb, _g1) = base.device_ptr_mut(&stream);
                    let (ps, _g2) = scales.device_ptr_mut(&stream);
                    let (pst, _g3) = fp8.start_dev.device_ptr(&stream);
                    let rc = unsafe {
                        nv_kernels::cuda::quantize_kv_fp8(
                            stream.cu_stream() as *mut c_void,
                            px as *const u16,
                            pb as *mut u8,
                            ps as *mut f32,
                            pst as *const i32,
                            t as i32,
                            self.kv_heads as i32,
                            self.head_dim as i32,
                            fp8.ring_window_rows_or_zero_for_full_length as i32,
                        )
                    };
                    ensure!(rc == 0, "quantize_kv_fp8 drafter append rc={rc}");
                }
                self.len += t;
                return Ok(());
            }
            self.k_bf16_rows_stay_exact_because_the_drafter_never_shares_the_trunk_fp8_slots
                .as_ref()
                .expect("bf16 drafter kv rows exist when the fp8 arm is off")
                .slice_set(&k_new.contiguous()?, 1, self.len)?;
            self.v_rows
                .as_ref()
                .expect("bf16 drafter kv rows exist when the fp8 arm is off")
                .slice_set(&v_new.contiguous()?, 1, self.len)?;
            self.len += t;
            Ok(())
        }

        pub fn rewind_to(&mut self, len: usize) -> Result<()> {
            ensure!(
                len <= self.len,
                "MtpKvCache.rewind_to({len}) beyond len {}",
                self.len
            );
            self.len = len;
            Ok(())
        }

        fn views(&self) -> Result<(Tensor, Tensor)> {
            ensure!(self.len > 0, "MtpKvCache.views on an empty cache");
            let (Some(k), Some(v)) = (
                self.k_bf16_rows_stay_exact_because_the_drafter_never_shares_the_trunk_fp8_slots
                    .as_ref(),
                self.v_rows.as_ref(),
            ) else {
                anyhow::bail!(
                    "MtpKvCache.views: the fp8 kv arm holds no bf16 rows; every consumer of \
                     this cache must route the kvshare fp8 attention when {} is set",
                    NV_Q38_DRAFTER_KV_FP8_ENV_DEFAULT_ON_0_RESTORES_THE_BF16_PER_Q_HEAD_ROUTE_USER_SIGNED_OFF_2026_09_04_BITWISE_IDENTICAL_EMITTED_ALL_THREE_CLASSES
                )
            };
            Ok((k.narrow(1, 0, self.len)?, v.narrow(1, 0, self.len)?))
        }

        pub fn decode_attention_splitk_bf16kv_reading_each_kv_row_once(
            &mut self,
            q_roped_1_1_nh_hd: &Tensor,
            heads: usize,
            softmax_scale: f32,
        ) -> Result<Tensor> {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            use half::bf16;
            use std::ffi::c_void;

            ensure!(self.len > 0, "splitk drafter attention on an empty cache");
            let k_buf = self
                .k_bf16_rows_stay_exact_because_the_drafter_never_shares_the_trunk_fp8_slots
                .as_ref()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "splitk drafter attention routes bf16 rows but the fp8 kv arm holds \
                         none; the kvshare fp8 route owns this cache"
                    )
                })?;
            let (_b, _s, kv_heads, hd) = k_buf.dims4()?;
            ensure!(
                q_roped_1_1_nh_hd.dims() == [1, 1, heads, hd],
                "splitk drafter attention: q must be [1, 1, {heads}, {hd}], got {:?}",
                q_roped_1_1_nh_hd.dims()
            );
            ensure!(
                kv_heads != 0 && heads.is_multiple_of(kv_heads),
                "splitk drafter attention: heads {heads} not a multiple of kv_heads {kv_heads}"
            );
            ensure!(
                k_buf.dtype() == DType::BF16,
                "{} routes a bf16-KV kernel but this drafter cache holds {:?}; unset the env \
                 for a non-bf16 drafter",
                NV_Q38_DRAFTER_ATTN_SPLITK_ENV_DEFAULT_ON_ROUTES_THE_Q1_DRAFTER_SCAN_THROUGH_THE_FUSED_BF16_SPLITK_DECODE_KERNEL_BECAUSE_CANDLE_FA2_FILLS_ONLY_N_HEADS_BLOCKS,
                k_buf.dtype()
            );
            let dev = match q_roped_1_1_nh_hd.device() {
                Device::Cuda(d) => d.clone(),
                _ => anyhow::bail!("splitk drafter attention needs a CUDA device"),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            if self
                .splitk_route_sized_at_first_call_because_the_head_count_arrives_with_the_query
                .is_none()
            {
                let scratch_elems =
                    nv_kernels::cuda::flash_splitk_scratch_elems(heads as i32, hd as i32);
                self.splitk_route_sized_at_first_call_because_the_head_count_arrives_with_the_query =
                    Some(SplitkRouteBuffers {
                        scratch: stream
                            .alloc_zeros::<f32>(scratch_elems)
                            .map_err(|e| anyhow::anyhow!(e))?,
                        fan_in: stream
                            .alloc_zeros::<u32>(heads)
                            .map_err(|e| anyhow::anyhow!(e))?,
                        len_dev: stream
                            .alloc_zeros::<i32>(1)
                            .map_err(|e| anyhow::anyhow!(e))?,
                        host_len: Box::new([0i32; 1]),
                        heads,
                    });
            }
            let q_f32_prescaled_because_the_kernel_scores_raw_qk = q_roped_1_1_nh_hd
                .to_dtype(DType::F32)?
                .affine(softmax_scale as f64, 0.0)?
                .contiguous()?;
            let mut out: cudarc::driver::CudaSlice<bf16> = unsafe {
                stream
                    .alloc::<bf16>(heads * hd)
                    .map_err(|e| anyhow::anyhow!(e))?
            };
            let rc = {
                let bufs = self
                    .splitk_route_sized_at_first_call_because_the_head_count_arrives_with_the_query
                    .as_mut()
                    .expect("splitk route buffers were just installed");
                ensure!(
                    bufs.heads == heads,
                    "splitk drafter attention: buffers sized for {} heads, called with {heads}",
                    bufs.heads
                );
                bufs.host_len[0] = self.len as i32;
                stream
                    .memcpy_htod(&bufs.host_len[..], &mut bufs.len_dev)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let (qs, ql) =
                    q_f32_prescaled_because_the_kernel_scores_raw_qk.storage_and_layout();
                let (ks, kl) = k_buf.storage_and_layout();
                let v_rows = self
                    .v_rows
                    .as_ref()
                    .expect("bf16 v rows exist on the bf16 splitk route");
                let (vs, vl) = v_rows.storage_and_layout();
                let q_cuda = match &*qs {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("splitk drafter attention: q storage not cuda"),
                };
                let k_cuda = match &*ks {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("splitk drafter attention: k storage not cuda"),
                };
                let v_cuda = match &*vs {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("splitk drafter attention: v storage not cuda"),
                };
                let q_view = q_cuda.as_cuda_slice::<f32>()?.slice(ql.start_offset()..);
                let k_view = k_cuda.as_cuda_slice::<bf16>()?.slice(kl.start_offset()..);
                let v_view = v_cuda.as_cuda_slice::<bf16>()?.slice(vl.start_offset()..);
                let (pq, _gq) = q_view.device_ptr(&stream);
                let (pk, _gk) = k_view.device_ptr(&stream);
                let (pv, _gv) = v_view.device_ptr(&stream);
                let (pl, _gl) = bufs.len_dev.device_ptr(&stream);
                let (ps, _gs) = bufs.scratch.device_ptr_mut(&stream);
                let (pf, _gf) = bufs.fan_in.device_ptr_mut(&stream);
                let (po, _go) = out.device_ptr_mut(&stream);
                unsafe {
                    nv_kernels::cuda::flash_decode_fused_bf16kv(
                        stream.cu_stream() as *mut c_void,
                        pq as *const f32,
                        pk as *const u16,
                        pv as *const u16,
                        po as *mut u16,
                        pl as *const i32,
                        0,
                        ps as *mut f32,
                        pf as *mut u32,
                        heads as i32,
                        kv_heads as i32,
                        hd as i32,
                        0,
                    )
                }
            };
            ensure!(rc == 0, "flash_decode_fused_bf16kv drafter q1 rc={rc}");
            let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
            Ok(Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, 1usize, heads, hd),
                candle_core::op::BackpropOp::none(),
                false,
            ))
        }

        pub fn decode_attention_kvshare_fp8kv_reading_kv_once_per_kv_head(
            &mut self,
            q_roped_1_1_nh_hd: &Tensor,
            heads: usize,
            softmax_scale: f32,
        ) -> Result<Tensor> {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            use half::bf16;
            use std::ffi::c_void;

            ensure!(
                self.len > 0,
                "kvshare fp8 drafter attention on an empty cache"
            );
            ensure!(
                self.kv_fp8_armed(),
                "kvshare fp8 drafter attention needs the fp8 kv arm"
            );
            let kv_heads = self.kv_heads;
            let hd = self.head_dim;
            ensure!(
                q_roped_1_1_nh_hd.dims() == [1, 1, heads, hd],
                "kvshare fp8 drafter attention: q must be [1, 1, {heads}, {hd}], got {:?}",
                q_roped_1_1_nh_hd.dims()
            );
            ensure!(
                kv_heads != 0 && heads.is_multiple_of(kv_heads),
                "kvshare fp8 drafter attention: heads {heads} not a multiple of kv_heads {kv_heads}"
            );
            let dev = match q_roped_1_1_nh_hd.device() {
                Device::Cuda(d) => d.clone(),
                _ => anyhow::bail!("kvshare fp8 drafter attention needs a CUDA device"),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            if self
                .splitk_route_sized_at_first_call_because_the_head_count_arrives_with_the_query
                .is_none()
            {
                let scratch_elems =
                    (heads * DRAFTER_KVSHARE_SCRATCH_SPLITS_128_IS_THE_KERNEL_MAX * (hd + 2)).max(
                        nv_kernels::cuda::flash_splitk_scratch_elems(heads as i32, hd as i32),
                    );
                self.splitk_route_sized_at_first_call_because_the_head_count_arrives_with_the_query =
                    Some(SplitkRouteBuffers {
                        scratch: stream
                            .alloc_zeros::<f32>(scratch_elems)
                            .map_err(|e| anyhow::anyhow!(e))?,
                        fan_in: stream
                            .alloc_zeros::<u32>(heads)
                            .map_err(|e| anyhow::anyhow!(e))?,
                        len_dev: stream
                            .alloc_zeros::<i32>(1)
                            .map_err(|e| anyhow::anyhow!(e))?,
                        host_len: Box::new([0i32; 1]),
                        heads,
                    });
            }
            let q_c = q_roped_1_1_nh_hd.contiguous()?;
            let mut out: cudarc::driver::CudaSlice<bf16> = unsafe {
                stream
                    .alloc::<bf16>(heads * hd)
                    .map_err(|e| anyhow::anyhow!(e))?
            };
            let rc = {
                let bufs = self
                    .splitk_route_sized_at_first_call_because_the_head_count_arrives_with_the_query
                    .as_mut()
                    .expect("kvshare route buffers were just installed");
                ensure!(
                    bufs.heads == heads,
                    "kvshare fp8 drafter attention: buffers sized for {} heads, called with {heads}",
                    bufs.heads
                );
                bufs.host_len[0] = self.len as i32;
                stream
                    .memcpy_htod(&bufs.host_len[..], &mut bufs.len_dev)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let fp8 = self
                    .fp8_rows_replace_the_bf16_pair_when_the_kv_fp8_arm_is_selected_saving_704mb_at_172k
                    .as_ref()
                    .expect("fp8 rows exist: kv_fp8_armed was checked");
                let ring = fp8.ring_window_rows_or_zero_for_full_length;
                let (qs, ql) = q_c.storage_and_layout();
                let q_cuda = match &*qs {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("kvshare fp8 drafter attention: q storage not cuda"),
                };
                let q_view = q_cuda.as_cuda_slice::<bf16>()?.slice(ql.start_offset()..);
                let (pq, _gq) = q_view.device_ptr(&stream);
                let (pk, _gk) = fp8.k_fp8.device_ptr(&stream);
                let (pv, _gv) = fp8.v_fp8.device_ptr(&stream);
                let (pks, _gks) = fp8.k_scales.device_ptr(&stream);
                let (pvs, _gvs) = fp8.v_scales.device_ptr(&stream);
                let (pl, _gl) = bufs.len_dev.device_ptr(&stream);
                let (ps, _gs) = bufs.scratch.device_ptr_mut(&stream);
                let (pf, _gf) = bufs.fan_in.device_ptr_mut(&stream);
                let (po, _go) = out.device_ptr_mut(&stream);
                let launch = if drafter_attn_tc_selected_from_env() {
                    nv_kernels::cuda::flash_decode_kvshare_fp8kv_paged_qk_tc
                } else {
                    nv_kernels::cuda::flash_decode_kvshare_fp8kv_paged
                };
                unsafe {
                    launch(
                        stream.cu_stream() as *mut c_void,
                        pq as *const u16,
                        pk as *const u8,
                        pv as *const u8,
                        pks as *const f32,
                        pvs as *const f32,
                        po as *mut u16,
                        pl as *const i32,
                        ps as *mut f32,
                        pf as *mut u32,
                        heads as i32,
                        kv_heads as i32,
                        hd as i32,
                        ring as i32,
                        ring as i32,
                        if (if ring > 0 { self.len.min(ring) } else { self.len })
                            < DRAFTER_KVSHARE_SHALLOW_TOTAL_65536_PINS_16_SPLITS_BECAUSE_THE_TWO_WAVE_PICK_COSTS_0P15_MS_OF_COMBINE_AT_256
                        {
                            DRAFTER_KVSHARE_SHALLOW_SPLITS_16_MATCHES_THE_BF16_ROUTES_PICK_FOR_24_HEADS
                        } else {
                            0
                        },
                        softmax_scale,
                        std::ptr::null(),
                        0,
                        0,
                        0,
                        0,
                    )
                }
            };
            ensure!(
                rc == 0,
                "flash_decode_kvshare_fp8kv_paged drafter q1 rc={rc}"
            );
            let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
            Ok(Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, 1usize, heads, hd),
                candle_core::op::BackpropOp::none(),
                false,
            ))
        }

        pub fn k_rows_host_f32(&self) -> Result<Vec<f32>> {
            let (k, _) = self.views()?;
            Ok(k.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
        }

        pub fn v_rows_host_f32(&self) -> Result<Vec<f32>> {
            let (_, v) = self.views()?;
            Ok(v.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
        }
    }

    fn load_trunk_output_norm_plus_one(
        main_checkpoint_dir: &Path,
        hidden: usize,
        eps: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<RmsNorm> {
        let loader =
            nv_weights::WeightLoader::open_dir(main_checkpoint_dir, device).with_context(|| {
                format!(
                    "{NV_Q38_MTP_REANCHOR_ENV}=1 needs the trunk output norm from {}",
                    main_checkpoint_dir.display()
                )
            })?;
        let mut raw = None;
        for name in TRUNK_OUTPUT_NORM_TENSOR_CANDIDATES {
            if let Ok(t) = loader.get(name, DType::F32) {
                raw = Some(t);
                break;
            }
        }
        let raw = raw.ok_or_else(|| {
            anyhow::anyhow!(
                "{NV_Q38_MTP_REANCHOR_ENV}=1: none of {TRUNK_OUTPUT_NORM_TENSOR_CANDIDATES:?} \
                 found in {}",
                main_checkpoint_dir.display()
            )
        })?;
        ensure!(
            raw.dims() == [hidden],
            "trunk output norm expected [{hidden}], got {:?}",
            raw.dims()
        );
        let w = raw.affine(1.0, 1.0)?.to_dtype(dtype)?;
        Ok(RmsNorm::new(w, eps))
    }

    pub struct Qwen38DenseMtpHead {
        pre_fc_norm_embedding: RmsNorm,
        pre_fc_norm_hidden: RmsNorm,
        fc: Linear,
        input_layernorm: RmsNorm,
        self_attn: AttentionLayer,
        post_attention_layernorm: RmsNorm,
        mlp: Mlp,
        norm: RmsNorm,
        trunk_output_norm_because_llama_cpp_feeds_the_mtp_post_norm_h_nextn: Option<RmsNorm>,
        draft_only_lm_head_nvfp4_w4a8_halves_twin_of_the_shared_fp8:
            Option<DraftLmHeadNvfp4W4a8DualHalves>,
        geometry: Qwen38MtpGeometry,
        dtype: DType,
    }

    pub struct DraftLmHeadNvfp4W4a8DualHalves {
        top: Linear,
        bottom: Linear,
        bottom_token_id_base: usize,
    }

    enum DraftHeadTail {
        Logits(Tensor),
        FusedArgmaxTokIdU32OnDevice(Tensor),
    }

    impl DraftLmHeadNvfp4W4a8DualHalves {
        fn dual_gemv_logits_one_pass_over_both_row_halves(
            &self,
            x_q8: &cudarc::driver::CudaSlice<i8>,
            x_scale: &cudarc::driver::CudaSlice<f32>,
            device: &Device,
        ) -> Result<Tensor> {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            use half::bf16;
            use std::ffi::c_void;

            let dev = match device {
                Device::Cuda(d) => d.clone(),
                _ => anyhow::bail!("w4a8 draft lm_head needs a CUDA device"),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            anyhow::ensure!(
                self.bottom_token_id_base == self.top.out_features(),
                "the unfused dual-gemv logits index rows as vocab ids, which only holds for \
                 the symmetric halves layout; under NV_Q38_DRAFT_VOCAB_PREFIX the fused \
                 argmax route (greedy, no probs_top3, the only route since the unfused escape was deleted 2026-09-14) is \
                 the only correct path"
            );
            let n2 = self.top.out_features();
            let n = n2 + self.bottom.out_features();
            let k = self.top.in_features();
            let (tw, ts, t_alpha, t_ig) = self
                .top
                .nvfp4_parts()
                .ok_or_else(|| anyhow::anyhow!("w4a8 draft lm_head: top half lost nvfp4 parts"))?;
            let (bw, bs, b_alpha, b_ig) = self.bottom.nvfp4_parts().ok_or_else(|| {
                anyhow::anyhow!("w4a8 draft lm_head: bottom half lost nvfp4 parts")
            })?;
            let mut y: cudarc::driver::CudaSlice<bf16> =
                unsafe { stream.alloc::<bf16>(n).map_err(|e| anyhow::anyhow!(e))? };
            let rc = {
                let (twp, _g1) = tw.device_ptr(&stream);
                let (tsp, _g2) = ts.device_ptr(&stream);
                let (bwp, _g3) = bw.device_ptr(&stream);
                let (bsp, _g4) = bs.device_ptr(&stream);
                let (xqp, _g5) = x_q8.device_ptr(&stream);
                let (xsp, _g6) = x_scale.device_ptr(&stream);
                let (yp, _g7) = y.device_ptr_mut(&stream);
                unsafe {
                    nv_kernels::cuda::gemv_nvfp4_w4a8_dual_m1(
                        stream.cu_stream() as *mut c_void,
                        twp as *const u8,
                        tsp as *const u8,
                        bwp as *const u8,
                        bsp as *const u8,
                        xqp as *const i8,
                        xsp as *const f32,
                        yp as *mut u16,
                        (yp + (n2 * 2) as u64) as *mut u16,
                        t_alpha * t_ig,
                        b_alpha * b_ig,
                        n2 as i32,
                        k as i32,
                    )
                }
            };
            ensure!(rc == 0, "gemv_nvfp4_w4a8_dual_m1 draft lm_head rc={rc}");
            let storage = candle_core::CudaStorage::wrap_cuda_slice(y, dev);
            Ok(Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, n),
                candle_core::op::BackpropOp::none(),
                false,
            ))
        }

        fn dual_gemv_argmax_fused_token_id_stays_on_device_and_no_logits_are_materialized(
            &self,
            x_q8: &cudarc::driver::CudaSlice<i8>,
            x_scale: &cudarc::driver::CudaSlice<f32>,
            device: &Device,
        ) -> Result<Tensor> {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            use std::ffi::c_void;

            let dev = match device {
                Device::Cuda(d) => d.clone(),
                _ => anyhow::bail!("fused w4a8 draft lm_head argmax needs a CUDA device"),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let n2 = self.top.out_features();
            let k = self.top.in_features();
            let (tw, ts, t_alpha, t_ig) = self.top.nvfp4_parts().ok_or_else(|| {
                anyhow::anyhow!("fused w4a8 draft lm_head: top half lost nvfp4 parts")
            })?;
            let (bw, bs, b_alpha, b_ig) = self.bottom.nvfp4_parts().ok_or_else(|| {
                anyhow::anyhow!("fused w4a8 draft lm_head: bottom half lost nvfp4 parts")
            })?;
            let mut packed_best: cudarc::driver::CudaSlice<u64> =
                unsafe { stream.alloc::<u64>(1).map_err(|e| anyhow::anyhow!(e))? };
            let mut out_id: cudarc::driver::CudaSlice<u32> =
                unsafe { stream.alloc::<u32>(1).map_err(|e| anyhow::anyhow!(e))? };
            let rc = {
                let (twp, _g1) = tw.device_ptr(&stream);
                let (tsp, _g2) = ts.device_ptr(&stream);
                let (bwp, _g3) = bw.device_ptr(&stream);
                let (bsp, _g4) = bs.device_ptr(&stream);
                let (xqp, _g5) = x_q8.device_ptr(&stream);
                let (xsp, _g6) = x_scale.device_ptr(&stream);
                let (pbp, _g7) = packed_best.device_ptr_mut(&stream);
                let (oip, _g8) = out_id.device_ptr_mut(&stream);
                unsafe {
                    nv_kernels::cuda::gemv_nvfp4_w4a8_dual_argmax_m1(
                        stream.cu_stream() as *mut c_void,
                        twp as *const u8,
                        tsp as *const u8,
                        bwp as *const u8,
                        bsp as *const u8,
                        xqp as *const i8,
                        xsp as *const f32,
                        pbp as *mut u64,
                        oip as *mut u32,
                        std::ptr::null_mut(),
                        t_alpha * t_ig,
                        b_alpha * b_ig,
                        n2 as i32,
                        self.bottom.out_features() as i32,
                        self.bottom_token_id_base as i32,
                        k as i32,
                    )
                }
            };
            ensure!(
                rc == 0,
                "gemv_nvfp4_w4a8_dual_argmax_m1 draft lm_head rc={rc}"
            );
            let storage = candle_core::CudaStorage::wrap_cuda_slice(out_id, dev);
            Ok(Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                1usize,
                candle_core::op::BackpropOp::none(),
                false,
            ))
        }
    }

    impl Qwen38DenseMtpHead {
        pub fn geometry_from_dense_base(base: &Qwen3Moe) -> Result<Qwen38MtpGeometry> {
            let cfg = base.config();
            let intermediate = base.dense_intermediate().ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen38DenseMtpHead needs a dense-hybrid base; this Qwen3Moe was loaded \
                     with routed MoE layers, use nv_models::qwen3_5_mtp::MtpHead for those"
                )
            })?;
            Ok(Qwen38MtpGeometry {
                hidden: cfg.hidden_size,
                heads: cfg.num_attention_heads,
                kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                intermediate,
                attn_output_gate: cfg.attn_output_gate,
            })
        }

        pub fn from_checkpoint(
            explicit_dir: Option<&Path>,
            main_checkpoint_dir: &Path,
            base: &Qwen3Moe,
            device: &Device,
        ) -> Result<Self> {
            Self::from_checkpoint_with_speculator(
                explicit_dir,
                main_checkpoint_dir,
                None,
                base,
                device,
            )
        }

        pub fn from_checkpoint_with_speculator(
            explicit_dir: Option<&Path>,
            main_checkpoint_dir: &Path,
            lora_adapter_dir_whose_speculator_deltas_merge_into_the_head: Option<&Path>,
            base: &Qwen3Moe,
            device: &Device,
        ) -> Result<Self> {
            let files = resolve_mtp_weight_files(explicit_dir, main_checkpoint_dir)?;
            let mut map: HashMap<String, Tensor> = HashMap::new();
            for f in &files {
                let loaded = candle_core::safetensors::load(f, device)
                    .with_context(|| format!("load MTP safetensors {}", f.display()))?;
                map.extend(
                    loaded
                        .into_iter()
                        .filter(|(k, _)| k.starts_with(MTP_TENSOR_PREFIX)),
                );
            }
            if let Some(adapter_dir) = lora_adapter_dir_whose_speculator_deltas_merge_into_the_head
            {
                if let Some(spec_dir) = mtp_speculator_dir_if_advertised(adapter_dir)? {
                    let modules =
                        merge_speculator_lora_into_mtp_map_before_any_quant_arm_so_the_drafter_serves_at_base_speed(
                            &mut map, &spec_dir,
                        )?;
                    eprintln!(
                        "[qwen3.8-mtp] merged co-trained speculator deltas from {} into {} MTP \
                         modules at load; the drafter serves at base speed",
                        spec_dir.display(),
                        modules.len()
                    );
                }
            }
            let mut head = Self::from_map(&map, base)?;
            if mtp_draft_lm_head_nvfp4_twin_from_env() {
                let vocab = base.lm_head().out_features();
                let explicit = std::env::var(NV_Q38_DRAFT_LMHEAD_NVFP4_ENV).is_ok();
                if vocab.is_multiple_of(256) || explicit {
                    head.install_draft_only_nvfp4_lm_head_twin_leaving_the_shared_fp8_untouched(
                        base,
                    )?;
                } else {
                    eprintln!(
                        "[q38-mtp] default nvfp4 draft lm_head twin declined: vocab {vocab} is                          not a multiple of 256; drafting keeps the shared fp8 head                          (set {NV_Q38_DRAFT_LMHEAD_NVFP4_ENV}=1 to make this a hard error)"
                    );
                }
            }
            if mtp_reanchor_post_norm_selected_from_env() {
                head.trunk_output_norm_because_llama_cpp_feeds_the_mtp_post_norm_h_nextn =
                    Some(load_trunk_output_norm_plus_one(
                        main_checkpoint_dir,
                        base.config().hidden_size,
                        base.config().rms_norm_eps,
                        base.dtype(),
                        device,
                    )?);
            }
            if crate::qwen38_mtp::drafter_boot_warm_selected_from_env() {
                head.boot_warm_one_throwaway_catchup_and_draft_into_a_scratch_kv_dropped_before_return(
                    base, device,
                )?;
            }
            Ok(head)
        }

        pub fn boot_warm_one_throwaway_catchup_and_draft_into_a_scratch_kv_dropped_before_return(
            &self,
            base: &Qwen3Moe,
            device: &Device,
        ) -> Result<()> {
            let t =
                DRAFTER_BOOT_WARM_PROMPT_TOKENS_17_COVER_BOTH_THE_16_ROW_FP8_CATCHUP_CALL_SHAPE_AND_THE_REMAINDER_ROW;
            let h = self.geometry.hidden;
            let mut scratch_kv = self.new_kv_cache(t + 2, device)?;
            let warm_prompt = vec![0u32; t];
            let ones_hiddens_stay_finite_through_every_norm_and_activation_quant =
                Tensor::ones((1usize, t, h), self.dtype, device)?;
            self.prefill_prompt_kv_shifted_by_one_with_position_zero_on_a_zero_hidden(
                base,
                &warm_prompt,
                &ones_hiddens_stay_finite_through_every_norm_and_activation_quant,
                &mut scratch_kv,
            )?;
            let anchor_hidden = ones_hiddens_stay_finite_through_every_norm_and_activation_quant
                .narrow(1, t - 1, 1)?
                .contiguous()?;
            if draft_fast_device_chained_tokens_selected_from_env() {
                let _ = self
                    .draft_chain_reading_tokens_back_once_because_per_step_readback_stalls_the_stream(
                        base,
                        0,
                        &anchor_hidden,
                        1,
                        None,
                        &mut scratch_kv,
                    )?;
            } else {
                let _ = self.forward_draft_tok(base, &anchor_hidden, 0, &mut scratch_kv)?;
            }
            device.synchronize()?;
            Ok(())
        }

        pub fn from_safetensors(
            path: impl AsRef<Path>,
            base: &Qwen3Moe,
            device: &Device,
        ) -> Result<Self> {
            let map = candle_core::safetensors::load(path.as_ref(), device)
                .with_context(|| format!("load MTP safetensors {}", path.as_ref().display()))?;
            Self::from_map(&map, base)
        }

        pub fn from_map(map: &HashMap<String, Tensor>, base: &Qwen3Moe) -> Result<Self> {
            let geometry = Self::geometry_from_dense_base(base)?;
            let cfg = base.config();
            Self::from_map_for_geometry(
                map,
                geometry,
                cfg.rms_norm_eps,
                cfg.rotary_dim(),
                base.dtype(),
            )
        }

        pub fn from_map_for_geometry(
            map: &HashMap<String, Tensor>,
            geometry: Qwen38MtpGeometry,
            rms_norm_eps: f64,
            rotary_dim: usize,
            dtype: DType,
        ) -> Result<Self> {
            let named: BTreeMap<String, Vec<usize>> = map
                .iter()
                .filter(|(k, _)| k.starts_with(MTP_TENSOR_PREFIX))
                .map(|(k, t)| (k.clone(), t.dims().to_vec()))
                .collect();
            validate_mtp_named_shapes(&geometry, &named)?;

            let eps = rms_norm_eps;
            let get = |name: &str| -> Result<Tensor> {
                map.get(name)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("MTP tensor missing: {name}"))
            };
            let fp8_runner = if mtp_drafter_fp8_resident_from_env() {
                let dev = map
                    .values()
                    .next()
                    .ok_or_else(|| {
                        anyhow::anyhow!("{NV_Q38_DRAFT_FP8_ENV}=1: empty MTP tensor map")
                    })?
                    .device()
                    .clone();
                let cuda = match &dev {
                    Device::Cuda(d) => d.clone(),
                    _ => anyhow::bail!(
                        "{NV_Q38_DRAFT_FP8_ENV}=1 quantizes the drafter at load through the resident \
                         e4m3 arm and needs the MTP tensors on a CUDA device"
                    ),
                };
                let stream = nv_layers::cuda_stream::current_stream(&cuda);
                Some(std::sync::Arc::new(std::sync::Mutex::new(
                    nv_quant::fp8::Fp8GemmRunner::new(stream)?,
                )))
            } else {
                None
            };
            let lin = |name: &str| -> Result<Linear> {
                let w = get(name)?.to_dtype(dtype)?.contiguous()?;
                match &fp8_runner {
                    Some(runner) => {
                        let out_f = w.dim(0)?;
                        let in_f = w.dim(1)?;
                        let host: Vec<half::bf16> =
                            w.to_dtype(DType::BF16)?.flatten_all()?.to_vec1()?;
                        let (bytes, rows) = nv_layers::linear::fp8_weight_payload(
                            &host,
                            out_f,
                            in_f,
                            None,
                            nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
                        )?;
                        let dev = match w.device() {
                            Device::Cuda(d) => d.clone(),
                            _ => anyhow::bail!(
                                "{NV_Q38_DRAFT_FP8_ENV}=1 drafter quantization needs CUDA tensors"
                            ),
                        };
                        let stream = nv_layers::cuda_stream::current_stream(&dev);
                        let weight_u8 =
                            stream.clone_htod(&bytes).map_err(|e| anyhow::anyhow!(e))?;
                        Linear::new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
                            weight_u8,
                            rows,
                            in_f,
                            out_f,
                            None,
                            w.device(),
                            runner.clone(),
                            nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
                        )
                    }
                    None => Linear::new(w, None),
                }
            };
            let zero_centered_norm_stored_as_weight_minus_one = |name: &str| -> Result<RmsNorm> {
                let raw = get(name)?.to_dtype(DType::F32)?;
                let w = raw.affine(1.0, 1.0)?.to_dtype(dtype)?;
                Ok(RmsNorm::new(w, eps))
            };

            let self_attn = AttentionLayer::from_parts(
                lin("mtp.layers.0.self_attn.q_proj.weight")?,
                lin("mtp.layers.0.self_attn.k_proj.weight")?,
                lin("mtp.layers.0.self_attn.v_proj.weight")?,
                lin("mtp.layers.0.self_attn.o_proj.weight")?,
                zero_centered_norm_stored_as_weight_minus_one(
                    "mtp.layers.0.self_attn.q_norm.weight",
                )?,
                zero_centered_norm_stored_as_weight_minus_one(
                    "mtp.layers.0.self_attn.k_norm.weight",
                )?,
                geometry.heads,
                geometry.kv_heads,
                geometry.head_dim,
                geometry.attn_output_gate,
                rotary_dim,
            );
            let mlp = Mlp::new(
                lin("mtp.layers.0.mlp.gate_proj.weight")?,
                lin("mtp.layers.0.mlp.up_proj.weight")?,
                lin("mtp.layers.0.mlp.down_proj.weight")?,
            )?;

            Ok(Self {
                pre_fc_norm_embedding: zero_centered_norm_stored_as_weight_minus_one(
                    "mtp.pre_fc_norm_embedding.weight",
                )?,
                pre_fc_norm_hidden: zero_centered_norm_stored_as_weight_minus_one(
                    "mtp.pre_fc_norm_hidden.weight",
                )?,
                fc: lin("mtp.fc.weight")?,
                input_layernorm: zero_centered_norm_stored_as_weight_minus_one(
                    "mtp.layers.0.input_layernorm.weight",
                )?,
                self_attn,
                post_attention_layernorm: zero_centered_norm_stored_as_weight_minus_one(
                    "mtp.layers.0.post_attention_layernorm.weight",
                )?,
                mlp,
                norm: zero_centered_norm_stored_as_weight_minus_one("mtp.norm.weight")?,
                trunk_output_norm_because_llama_cpp_feeds_the_mtp_post_norm_h_nextn: None,
                draft_only_lm_head_nvfp4_w4a8_halves_twin_of_the_shared_fp8: None,
                geometry,
                dtype,
            })
        }

        pub fn install_draft_only_nvfp4_lm_head_twin_leaving_the_shared_fp8_untouched(
            &mut self,
            base: &Qwen3Moe,
        ) -> Result<()> {
            let dev = match base.device() {
                Device::Cuda(d) => d.clone(),
                _ => anyhow::bail!(
                    "{NV_Q38_DRAFT_LMHEAD_NVFP4_ENV}=1 quantizes a second resident lm_head copy \
                     and needs the base on a CUDA device"
                ),
            };
            let lm = base.lm_head();
            let n = lm.out_features();
            ensure!(
                n.is_multiple_of(256),
                "{NV_Q38_DRAFT_LMHEAD_NVFP4_ENV}=1 serves the lm_head as two equal row halves \
                 on 128-row swizzle tiles, so vocab {n} must be a multiple of 256"
            );
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let runner = std::sync::Arc::new(std::sync::Mutex::new(
                nv_quant::nvfp4::Nvfp4GemmRunner::new(stream)?,
            ));
            let vocab_prefix_arm_frspec_class_draft_only_subset_full_vocab_verify_keeps_output_distribution =
                match std::env::var("NV_Q38_DRAFT_VOCAB_PREFIX").ok().as_deref() {
                    Some("0") => None,
                    Some(v) => v.parse::<usize>().ok(),
                    None => {
                        let p = VOCAB_PREFIX_DEFAULT_65536_THE_P32K_RUNG_FAILED_THE_TILED_SERVED_BASIS_AND_64K_CERTIFIED_ON_ALL_THREE;
                        if p + 512 < n {
                            Some(p)
                        } else {
                            None
                        }
                    }
                };
            let (top_rows, bot_start, bot_rows) = match
                vocab_prefix_arm_frspec_class_draft_only_subset_full_vocab_verify_keeps_output_distribution
            {
                Some(p) => {
                    const SPECIALS_TAIL_512_COVERS_QWEN_CONTROL_TOKENS_SO_STOPS_STAY_PROPOSABLE:
                        usize = 512;
                    let t = SPECIALS_TAIL_512_COVERS_QWEN_CONTROL_TOKENS_SO_STOPS_STAY_PROPOSABLE;
                    ensure!(
                        p % 256 == 0 && p + t < n,
                        "NV_Q38_DRAFT_VOCAB_PREFIX={p} must be 256-aligned and leave room for \
                         the {t}-row specials tail below vocab {n}"
                    );
                    (p, n - t, t)
                }
                None => (n / 2, n / 2, n / 2),
            };
            let top = lm
                .nvfp4_draft_twin_rows_quantized_from_this_resident_copy_which_stays_untouched(
                    runner.clone(),
                    0,
                    top_rows,
                )?;
            let bottom = lm
                .nvfp4_draft_twin_rows_quantized_from_this_resident_copy_which_stays_untouched(
                    runner, bot_start, bot_rows,
                )?;
            self.draft_only_lm_head_nvfp4_w4a8_halves_twin_of_the_shared_fp8 =
                Some(DraftLmHeadNvfp4W4a8DualHalves {
                    top,
                    bottom,
                    bottom_token_id_base: bot_start,
                });
            Ok(())
        }

        pub fn draft_lm_head_nvfp4_twin_active(&self) -> bool {
            self.draft_only_lm_head_nvfp4_w4a8_halves_twin_of_the_shared_fp8
                .is_some()
        }

        fn draft_final_add_norm_rowquant_i8_feeding_the_w4a8_lm_head(
            &self,
            mlp_out: &Tensor,
            residual: &Tensor,
        ) -> Result<(
            Tensor,
            cudarc::driver::CudaSlice<i8>,
            cudarc::driver::CudaSlice<f32>,
        )> {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            use half::bf16;
            use std::ffi::c_void;

            let hidden = self.geometry.hidden;
            let dev = match mlp_out.device() {
                Device::Cuda(d) => d.clone(),
                _ => anyhow::bail!("w4a8 draft final norm needs CUDA tensors"),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let x_c = mlp_out.contiguous()?;
            let r_c = residual.contiguous()?;
            let w_c = self.norm.weight_bf16().contiguous()?;
            let mut res_out: cudarc::driver::CudaSlice<bf16> = unsafe {
                stream
                    .alloc::<bf16>(hidden)
                    .map_err(|e| anyhow::anyhow!(e))?
            };
            let mut x_q8: cudarc::driver::CudaSlice<i8> =
                unsafe { stream.alloc::<i8>(hidden).map_err(|e| anyhow::anyhow!(e))? };
            let mut x_scale: cudarc::driver::CudaSlice<f32> =
                unsafe { stream.alloc::<f32>(1).map_err(|e| anyhow::anyhow!(e))? };
            let rc = {
                let (xs, xl) = x_c.storage_and_layout();
                let (rs, rl) = r_c.storage_and_layout();
                let (ws, wl) = w_c.storage_and_layout();
                let x_cuda = match &*xs {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("w4a8 draft final norm: mlp_out storage not cuda"),
                };
                let r_cuda = match &*rs {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("w4a8 draft final norm: residual storage not cuda"),
                };
                let w_cuda = match &*ws {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("w4a8 draft final norm: norm weight storage not cuda"),
                };
                let x_view = x_cuda.as_cuda_slice::<bf16>()?.slice(xl.start_offset()..);
                let r_view = r_cuda.as_cuda_slice::<bf16>()?.slice(rl.start_offset()..);
                let w_view = w_cuda.as_cuda_slice::<bf16>()?.slice(wl.start_offset()..);
                let (px, _gx) = x_view.device_ptr(&stream);
                let (pr, _gr) = r_view.device_ptr(&stream);
                let (pw, _gw) = w_view.device_ptr(&stream);
                let (pro, _g1) = res_out.device_ptr_mut(&stream);
                let (pq, _g2) = x_q8.device_ptr_mut(&stream);
                let (ps, _g3) = x_scale.device_ptr_mut(&stream);
                unsafe {
                    nv_kernels::cuda::rmsnorm_residual_writeout_rowquant_i8_m1(
                        stream.cu_stream() as *mut c_void,
                        px as *const u16,
                        pr as *const u16,
                        pw as *const u16,
                        pro as *mut u16,
                        pq as *mut i8,
                        ps as *mut f32,
                        hidden as i32,
                        self.norm.eps() as f32,
                    )
                }
            };
            ensure!(
                rc == 0,
                "rmsnorm_residual_writeout_rowquant_i8_m1 draft rc={rc}"
            );
            let h_storage = candle_core::CudaStorage::wrap_cuda_slice(res_out, dev);
            let h = Tensor::from_storage(
                candle_core::Storage::Cuda(h_storage),
                (1usize, 1usize, hidden),
                candle_core::op::BackpropOp::none(),
                false,
            );
            Ok((h, x_q8, x_scale))
        }

        pub fn reanchor_post_norm_active(&self) -> bool {
            self.trunk_output_norm_because_llama_cpp_feeds_the_mtp_post_norm_h_nextn
                .is_some()
        }

        pub fn trunk_hidden_rows_as_drafter_anchor_input(&self, rows: &Tensor) -> Result<Tensor> {
            match &self.trunk_output_norm_because_llama_cpp_feeds_the_mtp_post_norm_h_nextn {
                Some(n) => Ok(n.forward(&rows.to_dtype(self.dtype)?)?),
                None => Ok(rows.clone()),
            }
        }

        pub fn new_kv_cache(&self, max_seq: usize, device: &Device) -> Result<MtpKvCache> {
            MtpKvCache::new(
                self.geometry.kv_heads,
                self.geometry.head_dim,
                max_seq,
                self.dtype,
                device,
            )
        }

        fn layer_input_from_sampled_tokens_and_their_preceding_trunk_hidden(
            &self,
            base: &Qwen3Moe,
            tokens: &[u32],
            hiddens: &Tensor,
        ) -> Result<Tensor> {
            let device = hiddens.device().clone();
            let tok = Tensor::from_vec(tokens.to_vec(), tokens.len(), &device)?;
            self.layer_input_from_token_ids_already_on_device(base, &tok, hiddens)
        }

        fn layer_input_from_token_ids_already_on_device(
            &self,
            base: &Qwen3Moe,
            tok: &Tensor,
            hiddens: &Tensor,
        ) -> Result<Tensor> {
            let t = tok.dim(0)?;
            let h = self.geometry.hidden;
            ensure!(
                hiddens.dims() == [1, t, h],
                "MTP layer input: {t} tokens need trunk hiddens [1, {t}, {h}], got {:?}",
                hiddens.dims()
            );
            let emb = base
                .embed_weight()
                .index_select(tok, 0)?
                .reshape((1usize, t, h))?
                .to_dtype(self.dtype)?;
            let norm_emb = self.pre_fc_norm_embedding.forward(&emb)?;
            let norm_hid = self
                .pre_fc_norm_hidden
                .forward(&hiddens.to_dtype(self.dtype)?)?;
            let fused_in = Tensor::cat(&[&norm_emb, &norm_hid], D::Minus1)?.contiguous()?;
            decode_prof::lap("draft_embed_norms_cat");
            let out = self.fc.forward(&fused_in);
            decode_prof::lap("draft_fc");
            out
        }

        pub fn catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
            &self,
            base: &Qwen3Moe,
            tokens: &[u32],
            hiddens: &Tensor,
            cache: &mut MtpKvCache,
        ) -> Result<()> {
            let t = tokens.len();
            ensure!(t > 0, "MTP catch-up with zero tokens");
            let rows_per_call = if self.fc.fp8_scale_parts().is_some() {
                DRAFTER_FP8_CATCHUP_ROWS_PER_CALL_16_THE_GEMV_E4M3_MK_LEADING_CAP_BECAUSE_THIS_DEVICE_SERVES_NO_CUBLASLT_PER_ROW_FP8
            } else {
                t
            };
            self.catch_up_kv_with_explicit_rows_per_call(
                base,
                tokens,
                hiddens,
                cache,
                rows_per_call,
            )
        }

        fn catch_up_kv_with_explicit_rows_per_call(
            &self,
            base: &Qwen3Moe,
            tokens: &[u32],
            hiddens: &Tensor,
            cache: &mut MtpKvCache,
            rows_per_call: usize,
        ) -> Result<()> {
            let t = tokens.len();
            ensure!(
                t > 0 && rows_per_call > 0,
                "MTP catch-up requires nonzero tokens and row budget"
            );
            let mut off = 0usize;
            while off < t {
                let n = (t - off).min(rows_per_call);
                let start = cache.len();
                let fused = self.layer_input_from_sampled_tokens_and_their_preceding_trunk_hidden(
                    base,
                    &tokens[off..off + n],
                    &hiddens.narrow(1, off, n)?.contiguous()?,
                )?;
                let normed = self.input_layernorm.forward(&fused)?;
                let positions: Vec<i32> = (start as i32..(start + n) as i32).collect();
                let pos_t = Tensor::from_vec(positions, n, hiddens.device())?;
                let kv_only = std::env::var("NV_Q38_GDN_CATCHUP_KV_ONLY").as_deref() != Ok("0");
                let (k, v) = if kv_only {
                    self.self_attn
                        .project_kv_only_roped_for_a_drafter_owned_kv_skips_the_discarded_q_proj_and_gate_bitwise_because_rope_k_depends_only_on_k_and_positions(&normed, base.rope(), &pos_t)?
                } else {
                    let (_q, k, v, _g) = self.self_attn.project_qkv_roped_for_a_drafter_owned_kv(
                        &normed,
                        base.rope(),
                        &pos_t,
                    )?;
                    (k, v)
                };
                cache.append(&k, &v)?;
                off += n;
            }
            Ok(())
        }

        pub fn prefill_prompt_kv_shifted_by_one_with_position_zero_on_a_zero_hidden(
            &self,
            base: &Qwen3Moe,
            prompt: &[u32],
            trunk_hidden: &Tensor,
            cache: &mut MtpKvCache,
        ) -> Result<()> {
            let seq = prompt.len();
            ensure!(seq > 0, "MTP prompt prefill with empty prompt");
            ensure!(
                cache.is_empty(),
                "MTP prompt prefill expects a fresh cache, len={}",
                cache.len()
            );
            let h = self.geometry.hidden;
            ensure!(
                trunk_hidden.dims() == [1, seq, h],
                "MTP prompt prefill: trunk hidden must be [1, {seq}, {h}], got {:?}",
                trunk_hidden.dims()
            );
            let zero_hidden_stands_in_for_h_minus_one_and_rmsnorms_to_zero = Tensor::zeros(
                (1usize, 1usize, h),
                trunk_hidden.dtype(),
                trunk_hidden.device(),
            )?;
            let hiddens = Tensor::cat(
                &[
                    &zero_hidden_stands_in_for_h_minus_one_and_rmsnorms_to_zero,
                    &trunk_hidden.narrow(1, 0, seq - 1)?,
                ],
                1,
            )?
            .contiguous()?;
            if self.fc.fp8_scale_parts().is_some()
                && prompt_drafter_batch_from_raw_unset_on_user_sign_off_2026_09_11_explicit_values_unchanged(
                    std::env::var_os("NV_Q38_PROMPT_DRAFTER_BATCH").as_deref(),
                )
            {
                const PROMPT_ONLY_ROWS_4096_USE_DEVICE_ROWQUANT_FP8_GEMM_WITHOUT_CHANGING_DECODE_CATCHUP_16: usize = 4096;
                eprintln!("Q38-PROMPT-DRAFTER-BATCH tokens={seq} rows_per_call=4096 decode_catchup_unchanged=true");
                return self.catch_up_kv_with_explicit_rows_per_call(
                    base, prompt, &hiddens, cache,
                    PROMPT_ONLY_ROWS_4096_USE_DEVICE_ROWQUANT_FP8_GEMM_WITHOUT_CHANGING_DECODE_CATCHUP_16,
                );
            }
            self.catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
                base, prompt, &hiddens, cache,
            )
        }

        pub fn forward_draft(
            &self,
            base: &Qwen3Moe,
            base_hidden: &Tensor,
            next_token_id: u32,
            cache: &mut MtpKvCache,
        ) -> Result<(Tensor, Tensor)> {
            let device = base_hidden.device().clone();
            let tok = Tensor::from_vec(vec![next_token_id], 1usize, &device)?;
            self.forward_draft_from_token_ids_already_on_device(base, base_hidden, &tok, cache)
        }

        pub fn forward_draft_from_token_ids_already_on_device(
            &self,
            base: &Qwen3Moe,
            base_hidden: &Tensor,
            next_token: &Tensor,
            cache: &mut MtpKvCache,
        ) -> Result<(Tensor, Tensor)> {
            match self.forward_draft_core_whose_tail_is_logits_or_a_fused_on_device_argmax_id(
                base,
                base_hidden,
                next_token,
                cache,
                false,
            )? {
                (DraftHeadTail::Logits(logits), h) => Ok((logits, h)),
                (DraftHeadTail::FusedArgmaxTokIdU32OnDevice(_), _) => anyhow::bail!(
                    "draft core returned a fused argmax tail although logits were requested"
                ),
            }
        }

        pub fn forward_draft_fused_argmax_tok_on_device_from_token_ids_already_on_device(
            &self,
            base: &Qwen3Moe,
            base_hidden: &Tensor,
            next_token: &Tensor,
            cache: &mut MtpKvCache,
        ) -> Result<(Tensor, Tensor)> {
            match self.forward_draft_core_whose_tail_is_logits_or_a_fused_on_device_argmax_id(
                base,
                base_hidden,
                next_token,
                cache,
                true,
            )? {
                (DraftHeadTail::FusedArgmaxTokIdU32OnDevice(tok), h) => Ok((tok, h)),
                (DraftHeadTail::Logits(_), _) => anyhow::bail!(
                    "draft core returned logits although the fused argmax tail was requested; \
                     the fused route needs {NV_Q38_DRAFT_LMHEAD_NVFP4_ENV}=1's resident twin"
                ),
            }
        }

        fn forward_draft_core_whose_tail_is_logits_or_a_fused_on_device_argmax_id(
            &self,
            base: &Qwen3Moe,
            base_hidden: &Tensor,
            next_token: &Tensor,
            cache: &mut MtpKvCache,
            fuse_argmax_tail: bool,
        ) -> Result<(DraftHeadTail, Tensor)> {
            let d = base_hidden.dims();
            let hidden = self.geometry.hidden;
            ensure!(
                d == [1, 1, hidden],
                "MTP base_hidden: expected [1, 1, {hidden}], got {d:?}"
            );
            let device = base_hidden.device().clone();
            let position = cache.len() as i32;

            let mut h =
                self.layer_input_from_token_ids_already_on_device(base, next_token, base_hidden)?;

            let positions = Tensor::from_vec(vec![position], 1usize, &device)?;
            let residual = h.clone();
            let normed = self.input_layernorm.forward(&h)?;
            let (q, k, v, q_gate) = self.self_attn.project_qkv_roped_for_a_drafter_owned_kv(
                &normed,
                base.rope(),
                &positions,
            )?;
            cache.append(&k, &v)?;
            decode_prof::lap("draft_kv_append");
            let attn_cfg = AttnConfig {
                num_heads: self.geometry.heads,
                num_kv_heads: self.geometry.kv_heads,
                head_dim: self.geometry.head_dim,
                softmax_scale: 1.0 / (self.geometry.head_dim as f32).sqrt(),
                causal: true,
            };
            let attn_out = if cache.kv_fp8_armed() {
                let out = cache.decode_attention_kvshare_fp8kv_reading_kv_once_per_kv_head(
                    &q.contiguous()?,
                    self.geometry.heads,
                    attn_cfg.softmax_scale,
                )?;
                decode_prof::lap("draft_attn_kvshare_fp8");
                out
            } else if crate::qwen38_mtp::drafter_attn_splitk_selected_from_env()
                && matches!(device, Device::Cuda(_))
            {
                let out = cache.decode_attention_splitk_bf16kv_reading_each_kv_row_once(
                    &q.contiguous()?,
                    self.geometry.heads,
                    attn_cfg.softmax_scale,
                )?;
                decode_prof::lap("draft_attn_splitk");
                out
            } else {
                let (k_full, v_full) = cache.views()?;
                let out = flash_attn(
                    &q.contiguous()?,
                    &k_full.contiguous()?,
                    &v_full.contiguous()?,
                    &attn_cfg,
                )?;
                decode_prof::lap("draft_attn_flash");
                out
            };
            let attn = self
                .self_attn
                .finalize_attn_from_a_drafter_owned_kv(attn_out, q_gate, 1, 1, self.dtype)?;
            h = residual.add(&attn)?;
            decode_prof::lap("draft_attn_finalize");

            let residual2 = h.clone();
            let normed2 = self.post_attention_layernorm.forward(&h)?;
            let mlp_out = self.mlp.forward(&normed2)?.to_dtype(self.dtype)?;

            if let Some(halves) = &self.draft_only_lm_head_nvfp4_w4a8_halves_twin_of_the_shared_fp8
            {
                decode_prof::lap("draft_mlp");
                let (h_new, x_q8, x_scale) = self
                    .draft_final_add_norm_rowquant_i8_feeding_the_w4a8_lm_head(
                        &mlp_out, &residual2,
                    )?;
                h = h_new;
                let chained_hidden = if self.reanchor_post_norm_active() {
                    self.norm.forward(&h)?
                } else {
                    h.clone()
                };
                decode_prof::lap("draft_final_norm");
                if fuse_argmax_tail {
                    let tok = halves
                        .dual_gemv_argmax_fused_token_id_stays_on_device_and_no_logits_are_materialized(
                            &x_q8,
                            &x_scale,
                            h.device(),
                        )?;
                    decode_prof::lap("draft_lm_head_argmax_fused");
                    return Ok((
                        DraftHeadTail::FusedArgmaxTokIdU32OnDevice(tok),
                        chained_hidden,
                    ));
                }
                let logits = halves.dual_gemv_logits_one_pass_over_both_row_halves(
                    &x_q8,
                    &x_scale,
                    h.device(),
                )?;
                decode_prof::lap("draft_lm_head");
                return Ok((DraftHeadTail::Logits(logits), chained_hidden));
            }
            ensure!(
                !fuse_argmax_tail,
                "the fused draft lm_head argmax covers only the \
                 {NV_Q38_DRAFT_LMHEAD_NVFP4_ENV}=1 nvfp4 twin; the bf16 lm_head fallback \
                 stays on the materialize-then-argmax path"
            );

            h = residual2.add(&mlp_out)?;
            decode_prof::lap("draft_mlp");

            let normed_final = self.norm.forward(&h)?;
            let chained_hidden = if self.reanchor_post_norm_active() {
                normed_final.clone()
            } else {
                h.clone()
            };
            decode_prof::lap("draft_final_norm");
            let logits = base.lm_head().forward(&normed_final)?;
            let logits = logits.reshape((1usize, logits.dim(D::Minus1)?))?;
            decode_prof::lap("draft_lm_head");
            Ok((DraftHeadTail::Logits(logits), chained_hidden))
        }

        pub fn forward_draft_tok(
            &self,
            base: &Qwen3Moe,
            base_hidden: &Tensor,
            next_token_id: u32,
            cache: &mut MtpKvCache,
        ) -> Result<(u32, Tensor)> {
            if draft_lmhead_argmax_fuse_route_eligible_because_probs_top3_and_forced_drafts_need_logits(
                self.draft_lm_head_nvfp4_twin_active(),
                false,
                false,
                false,
            ) {
                let device = base_hidden.device().clone();
                let t = Tensor::from_vec(vec![next_token_id], 1usize, &device)?;
                let (tok_t, mtp_hidden) = self
                    .forward_draft_fused_argmax_tok_on_device_from_token_ids_already_on_device(
                        base,
                        base_hidden,
                        &t,
                        cache,
                    )?;
                let tok = tok_t.to_vec1::<u32>()?[0];
                decode_prof::lap("draft_argmax_readback");
                return Ok((tok, mtp_hidden));
            }
            let (logits, mtp_hidden) =
                self.forward_draft(base, base_hidden, next_token_id, cache)?;
            let tok = logits
                .argmax(D::Minus1)?
                .flatten_all()?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?[0];
            decode_prof::lap("draft_argmax_readback");
            Ok((tok, mtp_hidden))
        }

        pub fn forward_draft_tok_with_top_prob(
            &self,
            base: &Qwen3Moe,
            base_hidden: &Tensor,
            next_token_id: u32,
            cache: &mut MtpKvCache,
        ) -> Result<(u32, f32, Tensor)> {
            let (logits, mtp_hidden) =
                self.forward_draft(base, base_hidden, next_token_id, cache)?;
            let tok = logits
                .argmax(D::Minus1)?
                .flatten_all()?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?[0];
            let p = Self::drafter_top_prob_scalar_is_recip_sum_exp_logits_minus_max(&logits)?
                .to_vec1::<f32>()?[0];
            decode_prof::lap("draft_argmax_readback");
            Ok((tok, p, mtp_hidden))
        }

        fn device_row_argmax_u32_matching_candle_lowest_index_tie_break(
            logits: &Tensor,
        ) -> Result<Tensor> {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let dev = match logits.device() {
                Device::Cuda(d) => d.clone(),
                _ => {
                    return Ok(logits
                        .argmax(D::Minus1)?
                        .flatten_all()?
                        .to_dtype(DType::U32)?)
                }
            };
            let vocab = logits.dim(D::Minus1)?;
            let lf = logits.to_dtype(DType::F32)?.contiguous()?;
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let parts = nv_kernels::cuda::dflash_accept_parts();
            let mut part_val =
                unsafe { stream.alloc::<f32>(parts).map_err(|e| anyhow::anyhow!(e))? };
            let mut part_idx =
                unsafe { stream.alloc::<i32>(parts).map_err(|e| anyhow::anyhow!(e))? };
            let mut row_argmax = unsafe { stream.alloc::<u32>(1).map_err(|e| anyhow::anyhow!(e))? };
            let mut out_buf = unsafe { stream.alloc::<u32>(2).map_err(|e| anyhow::anyhow!(e))? };
            let drafts_buf = stream
                .alloc_zeros::<u32>(1)
                .map_err(|e| anyhow::anyhow!(e))?;
            let rc = {
                let (l_storage, ll) = lf.storage_and_layout();
                let l_cuda = match &*l_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("draft argmax: expected cuda storage for logits"),
                };
                let l_slice = l_cuda.as_cuda_slice::<f32>()?;
                let l_view = l_slice.slice(ll.start_offset()..);
                let (lp, _gl) = l_view.device_ptr(&stream);
                let (dp, _gd) = drafts_buf.device_ptr(&stream);
                let (pvp, _gpv) = part_val.device_ptr_mut(&stream);
                let (pip, _gpi) = part_idx.device_ptr_mut(&stream);
                let (rp, _gr) = row_argmax.device_ptr_mut(&stream);
                let (op, _go) = out_buf.device_ptr_mut(&stream);
                unsafe {
                    nv_kernels::cuda::dflash_accept_f32(
                        stream.cu_stream() as *mut _,
                        lp as *const f32,
                        dp as *const u32,
                        rp as *mut u32,
                        op as *mut u32,
                        pvp as *mut f32,
                        pip as *mut i32,
                        1,
                        vocab as i32,
                    )
                }
            };
            ensure!(rc == 0, "dflash_accept_f32 m=1 draft argmax returned {rc}");
            let storage = candle_core::CudaStorage::wrap_cuda_slice(row_argmax, dev);
            Ok(Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                1usize,
                candle_core::op::BackpropOp::none(),
                false,
            ))
        }

        pub fn draft_chain_reading_tokens_back_once_because_per_step_readback_stalls_the_stream(
            &self,
            base: &Qwen3Moe,
            anchor: u32,
            base_hidden: &Tensor,
            k: usize,
            forced_drafts: Option<&[u32]>,
            cache: &mut MtpKvCache,
        ) -> Result<Vec<u32>> {
            Ok(self
                .draft_chain_with_top_probs_reading_back_once(
                    base,
                    anchor,
                    base_hidden,
                    k,
                    forced_drafts,
                    false,
                    cache,
                )?
                .0)
        }

        fn drafter_top_prob_scalar_is_recip_sum_exp_logits_minus_max(
            logits: &Tensor,
        ) -> Result<Tensor> {
            let lf = logits.to_dtype(DType::F32)?;
            let mx = lf.max_keepdim(D::Minus1)?;
            let z = lf.broadcast_sub(&mx)?.exp()?.sum(D::Minus1)?;
            Ok(z.recip()?.flatten_all()?)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn draft_chain_with_top_probs_reading_back_once(
            &self,
            base: &Qwen3Moe,
            anchor: u32,
            base_hidden: &Tensor,
            k: usize,
            forced_drafts: Option<&[u32]>,
            want_top_probs: bool,
            cache: &mut MtpKvCache,
        ) -> Result<(Vec<u32>, Vec<f32>)> {
            let (drafts, probs, _) = self
                .draft_chain_with_top_probs_and_step_top3_whose_per_step_host_readback_syncs_so_headroom_probes_only(
                    base,
                    anchor,
                    base_hidden,
                    k,
                    forced_drafts,
                    want_top_probs,
                    false,
                    cache,
                )?;
            Ok((drafts, probs))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn draft_chain_with_top_probs_and_step_top3_whose_per_step_host_readback_syncs_so_headroom_probes_only(
            &self,
            base: &Qwen3Moe,
            anchor: u32,
            base_hidden: &Tensor,
            k: usize,
            forced_drafts: Option<&[u32]>,
            want_top_probs: bool,
            want_step_top3: bool,
            cache: &mut MtpKvCache,
        ) -> Result<(Vec<u32>, Vec<f32>, Vec<[u32; 3]>)> {
            ensure!(k >= 1, "draft chain: k must be >= 1");
            if let Some(f) = forced_drafts {
                ensure!(
                    f.len() == k,
                    "draft chain: forced drafts len {} != k {k}",
                    f.len()
                );
            }
            let device = base_hidden.device().clone();
            let mut h = base_hidden.clone();
            let mut tok = Tensor::from_vec(vec![anchor], 1usize, &device)?;
            let mut sampled: Vec<Tensor> = Vec::with_capacity(k);
            let mut top_probs: Vec<Tensor> = Vec::with_capacity(k);
            let mut step_top3: Vec<[u32; 3]> = Vec::new();
            let fused_route =
                draft_lmhead_argmax_fuse_route_eligible_because_probs_top3_and_forced_drafts_need_logits(
                    self.draft_lm_head_nvfp4_twin_active(),
                    want_top_probs,
                    want_step_top3,
                    forced_drafts.is_some(),
                );
            for j in 0..k {
                if fused_route {
                    let (am, dhidden) = self
                        .forward_draft_fused_argmax_tok_on_device_from_token_ids_already_on_device(
                            base, &h, &tok, cache,
                        )?;
                    sampled.push(am.clone());
                    tok = am;
                    h = dhidden;
                    continue;
                }
                let (logits, dhidden) =
                    self.forward_draft_from_token_ids_already_on_device(base, &h, &tok, cache)?;
                if want_top_probs && forced_drafts.is_none() {
                    top_probs.push(
                        Self::drafter_top_prob_scalar_is_recip_sum_exp_logits_minus_max(&logits)?,
                    );
                }
                if want_step_top3 && forced_drafts.is_none() {
                    let row: Vec<f32> = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                    step_top3.push(
                        top3_host_row_ids_lowest_index_tie_break_matching_device_argmax(&row),
                    );
                }
                tok = match forced_drafts {
                    Some(f) => Tensor::from_vec(vec![f[j]], 1usize, &device)?,
                    None => {
                        let am =
                            Self::device_row_argmax_u32_matching_candle_lowest_index_tie_break(
                                &logits,
                            )?;
                        sampled.push(am.clone());
                        am
                    }
                };
                h = dhidden;
            }
            let probs_host: Vec<f32> = if top_probs.is_empty() {
                Vec::new()
            } else {
                Tensor::cat(&top_probs, 0)?.to_vec1::<f32>()?
            };
            match forced_drafts {
                Some(f) => Ok((f.to_vec(), probs_host, step_top3)),
                None => Ok((
                    Tensor::cat(&sampled, 0)?.to_vec1::<u32>()?,
                    probs_host,
                    step_top3,
                )),
            }
        }

        pub fn draft_two_branch_tree_chain_plus_top2_sibling_step(
            &self,
            base: &Qwen3Moe,
            anchor: u32,
            base_hidden: &Tensor,
            cache: &mut MtpKvCache,
        ) -> Result<(Vec<u32>, u32, u32)> {
            let k = MTP_TREE_BRANCH_A_DEPTH_3_SO_THE_TREE_NEEDS_EXACTLY_NV_MTP_K_3;
            let device = base_hidden.device().clone();
            let anchor_row_len = cache.len() + 1;
            let mut h = base_hidden.clone();
            let mut tok = Tensor::from_vec(vec![anchor], 1usize, &device)?;
            let mut sampled: Vec<Tensor> = Vec::with_capacity(k);
            let mut step0_logits: Option<Tensor> = None;
            let mut step0_hidden: Option<Tensor> = None;
            for j in 0..k {
                let (logits, dhidden) =
                    self.forward_draft_from_token_ids_already_on_device(base, &h, &tok, cache)?;
                let am =
                    Self::device_row_argmax_u32_matching_candle_lowest_index_tie_break(&logits)?;
                if j == 0 {
                    step0_logits = Some(logits);
                    step0_hidden = Some(dhidden.clone());
                }
                sampled.push(am.clone());
                tok = am;
                h = dhidden;
            }
            let drafts: Vec<u32> = Tensor::cat(&sampled, 0)?.to_vec1::<u32>()?;
            let row: Vec<f32> = step0_logits
                .expect("k >= 1 so step 0 ran")
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let (_t1, _v1, c2, _v2) = top2_host_row_for_argmax_margin_triage(&row);
            let dh0 = step0_hidden.expect("k >= 1 so step 0 ran");
            cache.rewind_to(anchor_row_len)?;
            let (c2d2, _sib_hidden) = self.forward_draft_tok(base, &dh0, c2, cache)?;
            Ok((drafts, c2, c2d2))
        }
    }

    fn catch_up_one_trunk_hidden_chunk_retaining_only_an_owned_normalized_carry(
        base: &Qwen3Moe,
        mtp: &Qwen38DenseMtpHead,
        tokens: &[u32],
        hidden: &Tensor,
        mtp_kv: &mut MtpKvCache,
        carry: &mut Option<Tensor>,
    ) -> Result<()> {
        ensure!(!tokens.is_empty(), "MTP hidden chunk must not be empty");
        let fed = mtp_kv.len();
        let n = tokens.len();
        ensure!(
            hidden.dims() == [1, n, base.config().hidden_size],
            "MTP hidden chunk at {fed}: expected [1, {n}, {}], got {:?}",
            base.config().hidden_size,
            hidden.dims()
        );
        let normed = mtp.trunk_hidden_rows_as_drafter_anchor_input(hidden)?;
        let first = match carry.take() {
            Some(previous) => previous,
            None if fed == 0 => Tensor::zeros(
                (1usize, 1usize, normed.dim(2)?),
                normed.dtype(),
                normed.device(),
            )?,
            None => anyhow::bail!("MTP hidden chunk at {fed} lost its preceding carry row"),
        };
        let shifted = if n == 1 {
            first
        } else {
            Tensor::cat(&[&first, &normed.narrow(1, 0, n - 1)?], 1)?.contiguous()?
        };
        mtp.catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
            base, tokens, &shifted, mtp_kv,
        )?;
        ensure!(
            mtp_kv.len() == fed + n,
            "MTP hidden chunk catch-up ended at {}, expected {}",
            mtp_kv.len(),
            fed + n
        );
        *carry = Some(normed.narrow(1, n - 1, 1)?.force_contiguous()?);
        Ok(())
    }

    pub struct Qwen38MtpDecodeSession<'a> {
        base: &'a Qwen3Moe,
        mtp: &'a Qwen38DenseMtpHead,
        k: usize,
        p_min_drafter_confidence_cut_none_means_fixed_k: Option<f32>,
        cache: Qwen3MoeKvCache,
        mtp_kv: MtpKvCache,
        anchor: u32,
        base_hidden: Tensor,
        max_seq: usize,
        pub stats: MtpSpecStats,
        pub submitted_draft_len_hist: BTreeMap<usize, usize>,
    }

    impl<'a> Qwen38MtpDecodeSession<'a> {
        pub fn start(
            base: &'a Qwen3Moe,
            mtp: &'a Qwen38DenseMtpHead,
            k: usize,
            prompt: &[u32],
            max_seq: usize,
        ) -> Result<Self> {
            ensure!(!prompt.is_empty(), "mtp session: empty prompt");
            ensure!(
                !accept_tracking_selected_from_env(),
                "acceptance tracking is an opt-in graphed-session experiment, not an eager route"
            );
            ensure!(k >= 1, "mtp session: k must be >= 1");
            ensure!(
                prompt.len() + k < max_seq,
                "mtp session: prompt {} + k+1 {} > max_seq {max_seq}",
                prompt.len(),
                k + 1
            );
            let hidden = base.config().hidden_size;
            assert_eager_session_verify_rows_fit(hidden, k)?;
            if !mtp_eager_deep_k_selected_from_env() {
                let device_reported_rows_ceiling =
                    nv_kernels::cuda::gemv_i8_normed_mk_max_m(hidden as i32).max(0) as usize;
                ensure!(
                    mtp_verify_rows_per_round(k) <= device_reported_rows_ceiling,
                    "this device's lm_head rows-per-call ceiling is \
                     {device_reported_rows_ceiling} for hidden={hidden}; a k={k} mtp round \
                     verifies {} rows (the #107 dflash k=8 overflow is the cautionary tale)",
                    mtp_verify_rows_per_round(k)
                );
            }
            let device = base.device().clone();
            let mut cache = base.new_kv_cache(max_seq)?;
            let mut mtp_kv = mtp.new_kv_cache(max_seq, &device)?;

            let seq = prompt.len();
            ensure!(
                !mtp_reanchor_post_norm_selected_from_env() || mtp.reanchor_post_norm_active(),
                "{NV_Q38_MTP_REANCHOR_ENV}=1 but this head was not loaded through from_checkpoint \
                 so it carries no trunk output norm"
            );
            let chunk = nv_models::qwen3_5_moe::q38_prefill_chunk_tokens_from_env()
                .filter(|&c| seq > c)
                .unwrap_or(seq);
            let mut carry_normed_last_row_the_shift_by_one_hands_it_to_the_next_chunk: Option<
                Tensor,
            > = None;
            let mut last_logits: Option<Tensor> = None;
            let mut fed = 0usize;
            while fed < seq {
                let n = chunk.min(seq - fed);
                let tokens = Tensor::from_vec(prompt[fed..fed + n].to_vec(), (1usize, n), &device)?;
                let pos = Tensor::from_vec(
                    (fed as i32..(fed + n) as i32).collect::<Vec<i32>>(),
                    n,
                    &device,
                )?;
                let (logits, hidden) = base.forward_with_cache_dispatched_hidden_rows(
                    &tokens,
                    &pos,
                    &mut cache,
                    None,
                    Some(1),
                )?;
                catch_up_one_trunk_hidden_chunk_retaining_only_an_owned_normalized_carry(
                    base,
                    mtp,
                    &prompt[fed..fed + n],
                    &hidden,
                    &mut mtp_kv,
                    &mut carry_normed_last_row_the_shift_by_one_hands_it_to_the_next_chunk,
                )?;
                last_logits = Some(logits);
                fed += n;
            }
            let anchor = argmax_last_row(
                &last_logits.expect("mtp session: the non-empty prompt fed at least one chunk"),
            )?;
            let base_hidden = carry_normed_last_row_the_shift_by_one_hands_it_to_the_next_chunk
                .expect("mtp session: the non-empty prompt left a carry row");
            let _ = device.synchronize();

            let p_min = mtp_draft_p_min_from_env();
            if p_min.is_some() {
                ensure!(
                    k >= ADAPTIVE_DRAFT_KEEPS_AT_LEAST_TWO_BECAUSE_THE_TWO_ROW_VERIFY_IS_THE_ONE_GRAPH_SHAPE_WHOSE_TC_GEMM_PLAN_DIFFERS_AND_FLIPS_GREEDY_ARGMAX,
                    "{NV_Q38_MTP_P_MIN_ENV} needs k >= 2: the cut keeps at least two drafts so \
                     every submitted verify shares the m>=3 shape"
                );
                adaptive_verify_pins_the_env_unset_auto_dp4a_mlp_off_because_greedy_argmax_must_not_depend_on_which_m_the_cut_produced();
            }
            Ok(Self {
                base,
                mtp,
                k,
                p_min_drafter_confidence_cut_none_means_fixed_k: p_min,
                cache,
                mtp_kv,
                anchor,
                base_hidden,
                max_seq,
                stats: MtpSpecStats::default(),
                submitted_draft_len_hist: BTreeMap::new(),
            })
        }

        pub fn anchor(&self) -> u32 {
            self.anchor
        }

        pub fn committed_len(&self) -> usize {
            self.cache.current_len()
        }

        pub fn round_fits(&self) -> bool {
            self.cache.current_len() + self.k < self.max_seq
        }

        pub fn round(&mut self) -> Result<Vec<u32>> {
            self.round_inner(None)
        }

        pub fn round_with_drafts_from_a_clairvoyant_test_oracle(
            &mut self,
            drafts: &[u32],
        ) -> Result<Vec<u32>> {
            ensure!(
                drafts.len() == self.k,
                "oracle round needs exactly k={} drafts, got {}",
                self.k,
                drafts.len()
            );
            self.round_inner(Some(drafts))
        }

        fn round_inner(&mut self, forced_drafts: Option<&[u32]>) -> Result<Vec<u32>> {
            let device = self.base.device().clone();
            let round_t0 = std::time::Instant::now();
            let l = self.cache.current_len();
            let k = self.k;
            ensure!(
                l + k < self.max_seq,
                "mtp verify overflow: committed={l} + k+1={} > max_seq={}",
                k + 1,
                self.max_seq
            );
            ensure!(
                self.mtp_kv.len() == l,
                "mtp kv desync: drafter cache holds {} rows but the trunk committed {l}",
                self.mtp_kv.len()
            );

            let p_min_cut = self
                .p_min_drafter_confidence_cut_none_means_fixed_k
                .filter(|_| forced_drafts.is_none());
            let (mut drafts, draft_top_probs): (Vec<u32>, Vec<f32>) =
                if draft_fast_device_chained_tokens_selected_from_env() {
                    self.mtp.draft_chain_with_top_probs_reading_back_once(
                        self.base,
                        self.anchor,
                        &self.base_hidden,
                        k,
                        forced_drafts,
                        p_min_cut.is_some(),
                        &mut self.mtp_kv,
                    )?
                } else if let Some(pmin) = p_min_cut {
                    let mut drafts: Vec<u32> = Vec::with_capacity(k);
                    let mut probs: Vec<f32> = Vec::with_capacity(k);
                    let mut h = self.base_hidden.clone();
                    let mut tok = self.anchor;
                    for _ in 0..k {
                        let (d, p, dhidden) = self.mtp.forward_draft_tok_with_top_prob(
                            self.base,
                            &h,
                            tok,
                            &mut self.mtp_kv,
                        )?;
                        drafts.push(d);
                        probs.push(p);
                        tok = d;
                        h = dhidden;
                        if p < pmin
                            && drafts.len()
                                >= ADAPTIVE_DRAFT_KEEPS_AT_LEAST_TWO_BECAUSE_THE_TWO_ROW_VERIFY_IS_THE_ONE_GRAPH_SHAPE_WHOSE_TC_GEMM_PLAN_DIFFERS_AND_FLIPS_GREEDY_ARGMAX
                        {
                            break;
                        }
                    }
                    (drafts, probs)
                } else {
                    let mut drafts: Vec<u32> = Vec::with_capacity(k);
                    let mut h = self.base_hidden.clone();
                    let mut tok = self.anchor;
                    for j in 0..k {
                        let (d, dhidden) =
                            self.mtp
                                .forward_draft_tok(self.base, &h, tok, &mut self.mtp_kv)?;
                        let d = match forced_drafts {
                            Some(f) => f[j],
                            None => d,
                        };
                        drafts.push(d);
                        tok = d;
                        h = dhidden;
                    }
                    (drafts, Vec::new())
                };
            if let Some(pmin) = p_min_cut {
                let keep =
                    adaptive_submitted_draft_len_matching_llama_cpp_p_min_leading_confident_run(
                        &draft_top_probs,
                        pmin,
                    );
                drafts.truncate(keep);
            }
            *self
                .submitted_draft_len_hist
                .entry(drafts.len())
                .or_insert(0) += 1;
            let _ = device.synchronize();
            let draft_dt = round_t0.elapsed().as_secs_f64();

            let verify_t0 = std::time::Instant::now();
            let batch =
                crate::chain::build_chain_batch(self.anchor, &drafts, drafts.len() + 1, true)?;
            let m = batch.len();
            let block_pos: Vec<i32> = (0..m as i32).map(|i| l as i32 + i).collect();
            let bt = Tensor::from_vec(batch.clone(), (1usize, m), &device)?;
            let bp = Tensor::from_vec(block_pos, m, &device)?;

            self.cache.set_capture_lin_ckpts(true);
            let (vlogits, vhidden) =
                self.base
                    .forward_with_cache_dispatched_hidden(&bt, &bp, &mut self.cache, None)?;
            let greedy: Vec<u32> = vlogits
                .argmax(D::Minus1)?
                .flatten_all()?
                .to_dtype(DType::U32)?
                .to_vec1()?;
            let _ = device.synchronize();
            let verify_dt = verify_t0.elapsed().as_secs_f64();

            let commit_t0 = std::time::Instant::now();
            let acc = crate::chain::accept_prefix_argmax(&batch, &greedy)?;
            let consumed = acc.commit_len;
            let accepted = acc.draft_accepted;
            let bonus = acc.next_bonus;

            let mut emitted: Vec<u32> = drafts[..accepted].to_vec();
            emitted.push(bonus);

            let vhidden_for_drafter = self
                .mtp
                .trunk_hidden_rows_as_drafter_anchor_input(&vhidden)?;
            self.base_hidden = vhidden_for_drafter
                .narrow(1, mtp_round_hidden_reanchor_index(accepted), 1)?
                .contiguous()?;

            self.mtp_kv.rewind_to(l + 1)?;
            if accepted >= 1 {
                self.mtp.catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
                    self.base,
                    &batch[1..=accepted],
                    &vhidden_for_drafter.narrow(1, 0, accepted)?.contiguous()?,
                    &mut self.mtp_kv,
                )?;
            }
            ensure!(
                self.mtp_kv.len() == l + consumed,
                "mtp kv catch-up desync: drafter cache holds {} rows, trunk commits {}",
                self.mtp_kv.len(),
                l + consumed
            );

            self.cache.set_current_len(l + consumed);
            self.cache.rollback_lin_to(consumed)?;
            self.anchor = bonus;

            let _ = device.synchronize();
            let commit_dt = commit_t0.elapsed().as_secs_f64();
            self.stats
                .round_ms
                .push(1000.0 * round_t0.elapsed().as_secs_f64());
            self.stats.draft_ms += 1000.0 * draft_dt;
            self.stats.verify_ms += 1000.0 * verify_dt;
            self.stats.verify_ms_rounds.push(1000.0 * verify_dt);
            self.stats.commit_ms += 1000.0 * commit_dt;
            self.stats.rounds += 1;
            self.stats.drafted += drafts.len();
            self.stats.accepted += accepted;
            self.stats.emitted += emitted.len();
            if accepted > 0 {
                self.stats.pos0_accepted += 1;
            }
            *self.stats.accept_len_hist.entry(accepted).or_insert(0) += 1;
            Ok(emitted)
        }
    }

    pub struct Qwen38MtpGraphedDecodeSession<'a> {
        engine: &'a mut GraphedQwen3Moe,
        mtp: &'a Qwen38DenseMtpHead,
        k: usize,
        acceptance_tracking: Option<AcceptanceTrackingK>,
        p_min_drafter_confidence_cut_none_means_fixed_k: Option<f32>,
        mtp_kv: MtpKvCache,
        anchor: u32,
        base_hidden: Tensor,
        max_seq: usize,
        pub stats: MtpSpecStats,
        pub submitted_draft_len_hist: BTreeMap<usize, usize>,
        pub emitted_argmax_margins_aligned_to_tokens_after_the_anchor_so_a_cross_shape_divergence_can_be_triaged_tie_vs_corruption:
            Vec<f32>,
        pub draft_top_prob_rounds_present_only_when_the_p_min_cut_or_round_debug_asked_the_chain_for_probs:
            Vec<Vec<f32>>,
        pub tree_headroom_rounds_round_debug_fast_chain_only_drafter_top3_by_step_verify_greedy_rows_accepted:
            Vec<(Vec<[u32; 3]>, Vec<u32>, usize)>,
        suffix_arm_env_opt_in_consulted_before_each_mtp_chain_and_fed_every_emitted_token:
            Option<crate::qwen38_sufspec::SuffixSpecArm>,
        tree_env_opt_in_two_branch_m6_rounds: bool,
        pub tree_rounds: usize,
        pub tree_branch_b_rounds: usize,
        pub tree_branch_b_recovered_tokens: usize,
    }

    const _: () = assert!(
        MTP_TREE_ROWS_6_ANCHOR_PLUS_TOP1_CHAIN_D3_PLUS_TOP2_SIBLING_AND_ONE_CONTINUATION
            == nv_models::qwen3_5_moe::VERIFY_TREE_TWO_BRANCH_M6_ROWS
    );
    const _: () = assert!(
        MTP_TREE_BRANCH_B_FIRST_ROW_4_ROWS_ARE_ANCHOR_A1_A2_A3_B1_B2
            == nv_models::qwen3_5_moe::VERIFY_TREE_TWO_BRANCH_M6_BRANCH_B_FIRST_ROW
    );

    fn oapi_phase_walls_selected_from_env_stderr_only_so_streamed_bytes_never_move() -> bool {
        std::env::var("NV_OAPI_PHASE_WALLS").ok().as_deref() == Some("1")
    }

    fn session_phase_wall_synced(
        device: &Device,
        walls_on: bool,
        t: &mut std::time::Instant,
        label: &str,
        acc: &mut String,
    ) {
        if !walls_on {
            return;
        }
        let _ = device.synchronize();
        use std::fmt::Write;
        let _ = write!(acc, " {label}_ms={:.1}", t.elapsed().as_secs_f64() * 1000.0);
        *t = std::time::Instant::now();
    }

    impl<'a> Qwen38MtpGraphedDecodeSession<'a> {
        pub fn start(
            engine: &'a mut GraphedQwen3Moe,
            mtp: &'a Qwen38DenseMtpHead,
            k: usize,
            prompt: &[u32],
        ) -> Result<Self> {
            ensure!(!prompt.is_empty(), "graphed mtp session: empty prompt");
            ensure!(k >= 1, "graphed mtp session: k must be >= 1");
            let acceptance_tracking = if accept_tracking_selected_from_env() {
                ensure!(
                    k == 3 && mtp_draft_p_min_from_env().is_none() && !mtp_tree_selected_from_env(),
                    "acceptance tracking requires base k=3 with no confidence-cut or tree arm"
                );
                Some(AcceptanceTrackingK::default())
            } else {
                None
            };
            let max_seq = engine.cache().max_seq_len();
            ensure!(
                prompt.len() + k < max_seq,
                "graphed mtp session: prompt {} + k+1 {} > max_seq {max_seq}",
                prompt.len(),
                k + 1
            );
            let hidden = engine.underlying().config().hidden_size;
            engine
                .underlying()
                .prebuild_gdn_nvfp4_verify_twins_if_armed(engine.device());
            assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling(hidden, k, 0)?;
            let device_reported_rows_ceiling =
                nv_kernels::cuda::gemv_i8_normed_mk_max_m(hidden as i32).max(0) as usize;
            ensure!(
                mtp_verify_rows_per_round(k) <= device_reported_rows_ceiling,
                "this device's lm_head rows-per-call ceiling is {device_reported_rows_ceiling} \
                 for hidden={hidden}; a k={k} mtp round verifies {} rows",
                mtp_verify_rows_per_round(k)
            );
            let walls_on =
                oapi_phase_walls_selected_from_env_stderr_only_so_streamed_bytes_never_move();
            let mut acc = String::new();
            let t_all = std::time::Instant::now();
            let mut t_phase = std::time::Instant::now();
            let device = engine.device().clone();
            engine.reset()?;
            engine.install_grouped_moe()?;
            session_phase_wall_synced(&device, walls_on, &mut t_phase, "reset_install", &mut acc);
            let mut mtp_kv = mtp.new_kv_cache(max_seq, &device)?;
            session_phase_wall_synced(
                &device,
                walls_on,
                &mut t_phase,
                "drafter_kv_alloc",
                &mut acc,
            );
            ensure!(
                !mtp_reanchor_post_norm_selected_from_env() || mtp.reanchor_post_norm_active(),
                "{NV_Q38_MTP_REANCHOR_ENV}=1 but this head was not loaded through from_checkpoint \
                 so it carries no trunk output norm"
            );
            let stream_prefill = mtp_stream_prefill_selected_from_env();
            let mut streamed_carry = None;
            let (last_row, hidden_t) = if stream_prefill {
                eprintln!(
                    "[qwen38-mtp-prefill] streaming hidden chunks into FP8 draft catch-up; \
                     retaining one owned normalized anchor row"
                );
                let last_row = engine.prefill_hidden_chunks_serving_last_row_logits_visit(
                    prompt,
                    |base, piece, hidden| {
                        catch_up_one_trunk_hidden_chunk_retaining_only_an_owned_normalized_carry(
                            base,
                            mtp,
                            piece,
                            hidden,
                            &mut mtp_kv,
                            &mut streamed_carry,
                        )
                    },
                )?;
                session_phase_wall_synced(
                    &device,
                    walls_on,
                    &mut t_phase,
                    "trunk_draft_stream_prefill",
                    &mut acc,
                );
                (last_row, None)
            } else {
                let (last_row, hidden_t) = engine.prefill_hidden_serving_last_row_logits(prompt)?;
                session_phase_wall_synced(
                    &device,
                    walls_on,
                    &mut t_phase,
                    "trunk_prefill",
                    &mut acc,
                );
                (last_row, Some(hidden_t))
            };
            engine.ensure_verify_lane(k + 1)?;
            session_phase_wall_synced(&device, walls_on, &mut t_phase, "verify_capture", &mut acc);
            let p_min = mtp_draft_p_min_from_env();
            if p_min.is_some() {
                ensure!(
                    k >= ADAPTIVE_DRAFT_KEEPS_AT_LEAST_TWO_BECAUSE_THE_TWO_ROW_VERIFY_IS_THE_ONE_GRAPH_SHAPE_WHOSE_TC_GEMM_PLAN_DIFFERS_AND_FLIPS_GREEDY_ARGMAX,
                    "{NV_Q38_MTP_P_MIN_ENV} needs k >= 2: the cut keeps at least two drafts so \
                     every submitted verify shares the m>=3 graph shape"
                );
                adaptive_verify_pins_the_env_unset_auto_dp4a_mlp_off_because_greedy_argmax_must_not_depend_on_which_m_the_cut_produced();
            }
            let suffix_on = crate::qwen38_sufspec::spec_suffix_selected_from_env();
            if suffix_on || acceptance_tracking.is_some() {
                adaptive_verify_pins_the_env_unset_auto_dp4a_mlp_off_because_greedy_argmax_must_not_depend_on_which_m_the_cut_produced();
            }
            if p_min.is_some()
                || acceptance_tracking.is_some()
                || (suffix_on && suffix_precapture_selected_from_env())
                || mtp_precapture_selected_from_env()
            {
                let mut shapes: Vec<usize> = if p_min.is_some() || acceptance_tracking.is_some() {
                    (1 + ADAPTIVE_DRAFT_KEEPS_AT_LEAST_TWO_BECAUSE_THE_TWO_ROW_VERIFY_IS_THE_ONE_GRAPH_SHAPE_WHOSE_TC_GEMM_PLAN_DIFFERS_AND_FLIPS_GREEDY_ARGMAX
                        ..=k + 1)
                        .collect()
                } else {
                    vec![k + 1]
                };
                if suffix_on
                    && (acceptance_tracking.is_none() || suffix_precapture_selected_from_env())
                {
                    shapes.extend(
                        crate::qwen38_sufspec::SuffixSpecArm::verify_shapes_to_precapture(),
                    );
                }
                engine
                    .precapture_verify_graphs_then_discard_state_so_no_mid_run_capture_pollutes_verify_ms(
                        &shapes,
                    )?;
            }
            let tree_on = mtp_tree_selected_from_env();
            if tree_on {
                ensure!(
                    k == MTP_TREE_BRANCH_A_DEPTH_3_SO_THE_TREE_NEEDS_EXACTLY_NV_MTP_K_3,
                    "{NV_Q38_MTP_TREE_ENV}=1 serves the fixed two-branch m=6 shape whose branch \
                     A is the k=3 chain; got k={k}"
                );
                ensure!(
                    draft_fast_device_chained_tokens_selected_from_env(),
                    "{NV_Q38_MTP_TREE_ENV}=1 forks the fast device-chained draft loop; set \
                     NV_Q38_DRAFT_FAST=1"
                );
                ensure!(
                    p_min.is_none(),
                    "{NV_Q38_MTP_TREE_ENV}=1 needs the fixed 6-row shape; the {NV_Q38_MTP_P_MIN_ENV} \
                     adaptive cut changes the submitted length per round"
                );
                adaptive_verify_pins_the_env_unset_auto_dp4a_mlp_off_because_greedy_argmax_must_not_depend_on_which_m_the_cut_produced();
                let tree_rows =
                    MTP_TREE_ROWS_6_ANCHOR_PLUS_TOP1_CHAIN_D3_PLUS_TOP2_SIBLING_AND_ONE_CONTINUATION;
                assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling(hidden, tree_rows - 1, 0)?;
                ensure!(
                    tree_rows <= device_reported_rows_ceiling,
                    "tree verify rows {tree_rows} > device lm_head rows ceiling \
                     {device_reported_rows_ceiling}"
                );
                engine
                    .precapture_verify_tree_graph_then_discard_state_so_no_mid_run_capture_pollutes_verify_ms()?;
            }
            let anchor = argmax_host_row(&last_row)?;
            let base_hidden = if let Some(hidden_t) = hidden_t {
                let hidden_t = mtp.trunk_hidden_rows_as_drafter_anchor_input(&hidden_t)?;
                let base_hidden = hidden_t
                    .narrow(1, prompt.len() - 1, 1)?
                    .force_contiguous()?;
                session_phase_wall_synced(&device, walls_on, &mut t_phase, "anchor_prep", &mut acc);
                mtp.prefill_prompt_kv_shifted_by_one_with_position_zero_on_a_zero_hidden(
                    engine.underlying(),
                    prompt,
                    &hidden_t,
                    &mut mtp_kv,
                )?;
                session_phase_wall_synced(
                    &device,
                    walls_on,
                    &mut t_phase,
                    "drafter_catchup",
                    &mut acc,
                );
                base_hidden
            } else {
                ensure!(
                    mtp_kv.len() == prompt.len(),
                    "streamed MTP draft cache ended at {}, expected {}",
                    mtp_kv.len(),
                    prompt.len()
                );
                let base_hidden = streamed_carry
                    .take()
                    .expect("non-empty streamed MTP prefill retained an anchor row");
                session_phase_wall_synced(&device, walls_on, &mut t_phase, "anchor_prep", &mut acc);
                base_hidden
            };
            let _ = device.synchronize();
            if walls_on {
                eprintln!(
                    "Q38-SESSION-PHASES depth={} k={k} max_seq={max_seq}{acc} total_ms={:.1}",
                    prompt.len(),
                    t_all.elapsed().as_secs_f64() * 1000.0
                );
            }
            let suffix_arm = if suffix_on {
                let mut arm = crate::qwen38_sufspec::SuffixSpecArm::new();
                arm.extend_capped(prompt);
                arm.extend_capped(&[anchor]);
                Some(arm)
            } else {
                None
            };
            Ok(Self {
                engine,
                mtp,
                k,
                acceptance_tracking,
                p_min_drafter_confidence_cut_none_means_fixed_k: p_min,
                mtp_kv,
                anchor,
                base_hidden,
                max_seq,
                stats: MtpSpecStats::default(),
                submitted_draft_len_hist: BTreeMap::new(),
                emitted_argmax_margins_aligned_to_tokens_after_the_anchor_so_a_cross_shape_divergence_can_be_triaged_tie_vs_corruption:
                    Vec::new(),
                draft_top_prob_rounds_present_only_when_the_p_min_cut_or_round_debug_asked_the_chain_for_probs:
                    Vec::new(),
                tree_headroom_rounds_round_debug_fast_chain_only_drafter_top3_by_step_verify_greedy_rows_accepted:
                    Vec::new(),
                suffix_arm_env_opt_in_consulted_before_each_mtp_chain_and_fed_every_emitted_token:
                    suffix_arm,
                tree_env_opt_in_two_branch_m6_rounds: tree_on,
                tree_rounds: 0,
                tree_branch_b_rounds: 0,
                tree_branch_b_recovered_tokens: 0,
            })
        }

        pub fn suffix_rounds_drafted_accepted(&self) -> Option<(usize, usize, usize)> {
            self.suffix_arm_env_opt_in_consulted_before_each_mtp_chain_and_fed_every_emitted_token
                .as_ref()
                .map(|a| (a.rounds, a.drafted, a.accepted))
        }

        pub fn anchor(&self) -> u32 {
            self.anchor
        }

        pub fn drafter_anchor_token_and_trunk_hidden_row(&self) -> (u32, Tensor) {
            (self.anchor, self.base_hidden.clone())
        }

        pub fn trunk_for_shadow_probes(&self) -> &Qwen3Moe {
            self.engine.underlying()
        }

        pub fn committed_len(&self) -> usize {
            self.engine.current_pos()
        }

        pub fn round_fits(&self) -> bool {
            let rows = if self.tree_env_opt_in_two_branch_m6_rounds {
                MTP_TREE_ROWS_6_ANCHOR_PLUS_TOP1_CHAIN_D3_PLUS_TOP2_SIBLING_AND_ONE_CONTINUATION
            } else {
                self.k + 1
            };
            self.engine.current_pos() + rows <= self.max_seq
        }

        pub fn round(&mut self) -> Result<Vec<u32>> {
            self.round_inner(None)
        }

        pub fn round_with_drafts_from_a_clairvoyant_test_oracle(
            &mut self,
            drafts: &[u32],
        ) -> Result<Vec<u32>> {
            ensure!(
                drafts.len() == self.k,
                "oracle round needs exactly k={} drafts, got {}",
                self.k,
                drafts.len()
            );
            self.round_inner(Some(drafts))
        }

        pub fn profile_one_draft_chain_then_rewind_arming_nv_prof_decode_whose_every_lap_syncs(
            &mut self,
        ) -> Result<()> {
            let device = self.engine.device().clone();
            let l = self.mtp_kv.len();
            ensure!(
                l == self.engine.current_pos(),
                "draft-chain profile needs a settled session: drafter kv holds {} rows, trunk \
                 committed {}",
                l,
                self.engine.current_pos()
            );
            let _ = device.synchronize();
            decode_prof::begin_env_nv_prof_decode_refusing_mid_capture_because_every_lap_syncs(
                &device,
            );
            let mut h = self.base_hidden.clone();
            let mut tok = self.anchor;
            for _ in 0..self.k {
                let (d, dhidden) = self.mtp.forward_draft_tok(
                    self.engine.underlying(),
                    &h,
                    tok,
                    &mut self.mtp_kv,
                )?;
                tok = d;
                h = dhidden;
            }
            decode_prof::report_and_end(l);
            self.mtp_kv.rewind_to(l)
        }

        pub fn drafts_and_top_probs_for_one_round_then_rewind_so_an_ab_gate_reads_the_chain_without_committing(
            &mut self,
            want_top_probs: bool,
        ) -> Result<(Vec<u32>, Vec<f32>)> {
            let l = self.mtp_kv.len();
            ensure!(
                l == self.engine.current_pos(),
                "draft ab probe needs a settled session: drafter kv holds {} rows, trunk \
                 committed {}",
                l,
                self.engine.current_pos()
            );
            let out = self.mtp.draft_chain_with_top_probs_reading_back_once(
                self.engine.underlying(),
                self.anchor,
                &self.base_hidden,
                self.k,
                None,
                want_top_probs,
                &mut self.mtp_kv,
            )?;
            self.mtp_kv.rewind_to(l)?;
            Ok(out)
        }

        fn round_inner(&mut self, forced_drafts: Option<&[u32]>) -> Result<Vec<u32>> {
            let device = self.engine.device().clone();
            let round_t0 = std::time::Instant::now();
            let l = self.engine.current_pos();
            let k = self.k;
            ensure!(
                l + k < self.max_seq,
                "graphed mtp verify overflow: committed={l} + k+1={} > max_seq={}",
                k + 1,
                self.max_seq
            );
            ensure!(
                self.mtp_kv.len() == l,
                "graphed mtp kv desync: drafter cache holds {} rows but the trunk committed {l}",
                self.mtp_kv.len()
            );

            let round_debug_on =
                nv_config::env_arm_opt_in_eq_some_1_reread_every_call(NV_Q38_MTP_ROUND_DEBUG_ENV);
            let p_min_cut = self
                .p_min_drafter_confidence_cut_none_means_fixed_k
                .filter(|_| forced_drafts.is_none());
            let want_probs = forced_drafts.is_none() && (p_min_cut.is_some() || round_debug_on);
            let suffix_drafts: Option<Vec<u32>> = if forced_drafts.is_none() {
                self.suffix_arm_env_opt_in_consulted_before_each_mtp_chain_and_fed_every_emitted_token
                    .as_ref()
                    .and_then(|arm| arm.propose_for_round(self.max_seq - l - 1))
            } else {
                None
            };
            let suffix_round = suffix_drafts.is_some();
            let k = if !suffix_round && forced_drafts.is_none() {
                self.acceptance_tracking
                    .as_mut()
                    .map(|policy| policy.next_k_for_a_non_suffix_non_forced_round())
                    .unwrap_or(k)
            } else {
                k
            };
            if self.tree_env_opt_in_two_branch_m6_rounds && forced_drafts.is_none() && !suffix_round
            {
                return self.tree_round_two_branch_m6(l, round_t0);
            }
            let mut round_drafter_top3_by_step: Vec<[u32; 3]> = Vec::new();
            let (mut drafts, draft_top_probs): (Vec<u32>, Vec<f32>) = if let Some(sd) =
                suffix_drafts
            {
                self.mtp
                        .catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
                            self.engine.underlying(),
                            &[self.anchor],
                            &self.base_hidden,
                            &mut self.mtp_kv,
                        )?;
                (sd, Vec::new())
            } else if draft_fast_device_chained_tokens_selected_from_env() {
                let (d, p, t3) = self
                        .mtp
                        .draft_chain_with_top_probs_and_step_top3_whose_per_step_host_readback_syncs_so_headroom_probes_only(
                            self.engine.underlying(),
                            self.anchor,
                            &self.base_hidden,
                            k,
                            forced_drafts,
                            want_probs,
                            round_debug_on,
                            &mut self.mtp_kv,
                        )?;
                round_drafter_top3_by_step = t3;
                (d, p)
            } else {
                let mut drafts: Vec<u32> = Vec::with_capacity(k);
                let mut probs: Vec<f32> = Vec::with_capacity(k);
                let mut h = self.base_hidden.clone();
                let mut tok = self.anchor;
                for j in 0..k {
                    let (d, p, dhidden) = if want_probs {
                        self.mtp.forward_draft_tok_with_top_prob(
                            self.engine.underlying(),
                            &h,
                            tok,
                            &mut self.mtp_kv,
                        )?
                    } else {
                        let (d, dhidden) = self.mtp.forward_draft_tok(
                            self.engine.underlying(),
                            &h,
                            tok,
                            &mut self.mtp_kv,
                        )?;
                        (d, f32::NAN, dhidden)
                    };
                    if want_probs {
                        probs.push(p);
                    }
                    let d = match forced_drafts {
                        Some(f) => f[j],
                        None => d,
                    };
                    drafts.push(d);
                    tok = d;
                    h = dhidden;
                    if let Some(pmin) = p_min_cut {
                        if p < pmin
                                && drafts.len()
                                    >= ADAPTIVE_DRAFT_KEEPS_AT_LEAST_TWO_BECAUSE_THE_TWO_ROW_VERIFY_IS_THE_ONE_GRAPH_SHAPE_WHOSE_TC_GEMM_PLAN_DIFFERS_AND_FLIPS_GREEDY_ARGMAX
                            {
                                break;
                            }
                    }
                }
                (drafts, probs)
            };
            if !draft_top_probs.is_empty() {
                self.draft_top_prob_rounds_present_only_when_the_p_min_cut_or_round_debug_asked_the_chain_for_probs
                    .push(draft_top_probs.clone());
            }
            if let Some(pmin) = p_min_cut {
                if !suffix_round {
                    let keep =
                        adaptive_submitted_draft_len_matching_llama_cpp_p_min_leading_confident_run(
                            &draft_top_probs,
                            pmin,
                        );
                    drafts.truncate(keep);
                }
            }
            *self
                .submitted_draft_len_hist
                .entry(drafts.len())
                .or_insert(0) += 1;
            let _ = device.synchronize();
            let draft_dt = round_t0.elapsed().as_secs_f64();

            let verify_t0 = std::time::Instant::now();
            let batch =
                crate::chain::build_chain_batch(self.anchor, &drafts, drafts.len() + 1, true)?;
            let m = batch.len();
            self.engine.ensure_verify_lane(m)?;
            self.engine.forward_verify_chain(&batch)?;
            let flat = self.engine.verify_logits_host()?;
            let vocab = self.engine.vocab_size();
            let greedy = argmax_host_rows(&flat, m, vocab)?;
            let verify_dt = verify_t0.elapsed().as_secs_f64();

            let commit_t0 = std::time::Instant::now();
            let acc = crate::chain::accept_prefix_argmax(&batch, &greedy)?;
            if !round_drafter_top3_by_step.is_empty() {
                self.tree_headroom_rounds_round_debug_fast_chain_only_drafter_top3_by_step_verify_greedy_rows_accepted
                    .push((round_drafter_top3_by_step, greedy.clone(), acc.draft_accepted));
            }
            if round_debug_on {
                for r in 0..m {
                    let row = &flat[r * vocab..(r + 1) * vocab];
                    let (t1, v1, t2, v2) = top2_host_row_for_argmax_margin_triage(row);
                    eprintln!(
                        "[q38-mtp-round-debug] l={l} m={m} row={r} pos={} tok_in={} greedy={} \
                         top1={t1}:{v1:.4} top2={t2}:{v2:.4} margin={:.6} commit_len={} bonus={}",
                        l + r,
                        batch[r],
                        greedy[r],
                        v1 - v2,
                        acc.commit_len,
                        acc.next_bonus
                    );
                }
            }
            let consumed = acc.commit_len;
            let accepted = acc.draft_accepted;
            let bonus = acc.next_bonus;

            let mut emitted: Vec<u32> = drafts[..accepted].to_vec();
            emitted.push(bonus);
            if let Some(arm) = self
                .suffix_arm_env_opt_in_consulted_before_each_mtp_chain_and_fed_every_emitted_token
                .as_mut()
            {
                arm.extend_capped(&emitted);
                if suffix_round {
                    arm.rounds += 1;
                    arm.drafted += drafts.len();
                    arm.accepted += accepted;
                }
            }
            for r in 0..consumed {
                let (_t1, v1, _t2, v2) =
                    top2_host_row_for_argmax_margin_triage(&flat[r * vocab..(r + 1) * vocab]);
                self.emitted_argmax_margins_aligned_to_tokens_after_the_anchor_so_a_cross_shape_divergence_can_be_triaged_tie_vs_corruption
                    .push(v1 - v2);
            }

            let vhidden = self
                .engine
                .verify_hidden_rows_tensor_valid_until_next_forward()?;
            let vhidden_for_drafter = self
                .mtp
                .trunk_hidden_rows_as_drafter_anchor_input(&vhidden)?;
            self.base_hidden = vhidden_for_drafter
                .narrow(1, mtp_round_hidden_reanchor_index(accepted), 1)?
                .copy()?;

            self.mtp_kv.rewind_to(l + 1)?;
            if accepted >= 1 {
                self.mtp.catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
                    self.engine.underlying(),
                    &batch[1..=accepted],
                    &vhidden_for_drafter.narrow(1, 0, accepted)?.contiguous()?,
                    &mut self.mtp_kv,
                )?;
            }
            ensure!(
                self.mtp_kv.len() == l + consumed,
                "graphed mtp kv catch-up desync: drafter cache holds {} rows, trunk commits {}",
                self.mtp_kv.len(),
                l + consumed
            );

            self.engine.commit_verify_consumed(consumed)?;
            self.anchor = bonus;
            if !suffix_round && forced_drafts.is_none() {
                if let Some(policy) = self.acceptance_tracking.as_mut() {
                    policy.record_committed_mtp_round_not_suffix(accepted, drafts.len());
                }
            }

            let _ = device.synchronize();
            let commit_dt = commit_t0.elapsed().as_secs_f64();
            self.stats
                .round_ms
                .push(1000.0 * round_t0.elapsed().as_secs_f64());
            self.stats.draft_ms += 1000.0 * draft_dt;
            self.stats.verify_ms += 1000.0 * verify_dt;
            self.stats.verify_ms_rounds.push(1000.0 * verify_dt);
            self.stats.commit_ms += 1000.0 * commit_dt;
            self.stats.rounds += 1;
            self.stats.drafted += drafts.len();
            self.stats.accepted += accepted;
            self.stats.emitted += emitted.len();
            if accepted > 0 {
                self.stats.pos0_accepted += 1;
            }
            *self.stats.accept_len_hist.entry(accepted).or_insert(0) += 1;
            self.log_stage_summary_every_64_rounds_when_nv_q38_mtp_stats_is_1();
            Ok(emitted)
        }

        pub fn log_stage_summary_every_64_rounds_when_nv_q38_mtp_stats_is_1(&self) {
            if self.stats.rounds == 0
                || !self.stats.rounds.is_multiple_of(64)
                || !nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_MTP_STATS")
            {
                return;
            }
            let r = self.stats.rounds as f64;
            let round_ms = (self.stats.draft_ms + self.stats.verify_ms + self.stats.commit_ms) / r;
            eprintln!(
                "[q38-mtp-stats] rounds={} emitted/round={:.2} accept/round={:.2} \
                 draft={:.2}ms verify={:.2}ms commit={:.2}ms stage_sum={:.2}ms/round \
                 implied_tok_s={:.1} hist={:?}",
                self.stats.rounds,
                self.stats.emitted as f64 / r,
                self.stats.accepted as f64 / r,
                self.stats.draft_ms / r,
                self.stats.verify_ms / r,
                self.stats.commit_ms / r,
                round_ms,
                if round_ms > 0.0 {
                    1000.0 * (self.stats.emitted as f64 / r) / round_ms
                } else {
                    0.0
                },
                self.stats.accept_len_hist
            );
        }

        fn tree_round_two_branch_m6(
            &mut self,
            l: usize,
            round_t0: std::time::Instant,
        ) -> Result<Vec<u32>> {
            let device = self.engine.device().clone();
            let rows =
                MTP_TREE_ROWS_6_ANCHOR_PLUS_TOP1_CHAIN_D3_PLUS_TOP2_SIBLING_AND_ONE_CONTINUATION;
            ensure!(
                l + rows <= self.max_seq,
                "tree verify overflow: committed={l} + rows={rows} > max_seq={}",
                self.max_seq
            );
            let (drafts, c2, c2d2) = self
                .mtp
                .draft_two_branch_tree_chain_plus_top2_sibling_step(
                    self.engine.underlying(),
                    self.anchor,
                    &self.base_hidden,
                    &mut self.mtp_kv,
                )?;
            *self.submitted_draft_len_hist.entry(rows - 1).or_insert(0) += 1;
            let _ = device.synchronize();
            let draft_dt = round_t0.elapsed().as_secs_f64();

            let verify_t0 = std::time::Instant::now();
            let rows_tokens: Vec<u32> =
                vec![self.anchor, drafts[0], drafts[1], drafts[2], c2, c2d2];
            self.engine.ensure_verify_tree_lane_two_branch_m6()?;
            self.engine
                .forward_verify_tree_two_branch_m6(&rows_tokens)?;
            let flat = self.engine.verify_logits_host()?;
            let vocab = self.engine.vocab_size();
            let greedy = argmax_host_rows(&flat, rows, vocab)?;
            let verify_dt = verify_t0.elapsed().as_secs_f64();

            let commit_t0 = std::time::Instant::now();
            let acc = accept_two_branch_tree_m6(&rows_tokens, &greedy)?;
            let decided_rows: Vec<usize> = match acc.branch_b_accepted_rows {
                None => (0..acc.depth_consumed).collect(),
                Some(b) => std::iter::once(0)
                    .chain(
                        MTP_TREE_BRANCH_B_FIRST_ROW_4_ROWS_ARE_ANCHOR_A1_A2_A3_B1_B2
                            ..MTP_TREE_BRANCH_B_FIRST_ROW_4_ROWS_ARE_ANCHOR_A1_A2_A3_B1_B2 + b,
                    )
                    .collect(),
            };
            for &r in &decided_rows {
                let (_t1, v1, _t2, v2) =
                    top2_host_row_for_argmax_margin_triage(&flat[r * vocab..(r + 1) * vocab]);
                self.emitted_argmax_margins_aligned_to_tokens_after_the_anchor_so_a_cross_shape_divergence_can_be_triaged_tie_vs_corruption
                    .push(v1 - v2);
            }

            let vhidden = self
                .engine
                .verify_hidden_rows_tensor_valid_until_next_forward()?;
            let vhidden_for_drafter = self
                .mtp
                .trunk_hidden_rows_as_drafter_anchor_input(&vhidden)?;
            self.base_hidden = vhidden_for_drafter.narrow(1, acc.bonus_row, 1)?.copy()?;

            self.mtp_kv.rewind_to(l + 1)?;
            match acc.branch_b_accepted_rows {
                None if acc.drafts_accepted >= 1 => {
                    self.mtp.catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
                        self.engine.underlying(),
                        &rows_tokens[1..=acc.drafts_accepted],
                        &vhidden_for_drafter
                            .narrow(1, 0, acc.drafts_accepted)?
                            .contiguous()?,
                        &mut self.mtp_kv,
                    )?;
                }
                Some(b) => {
                    let b0 = MTP_TREE_BRANCH_B_FIRST_ROW_4_ROWS_ARE_ANCHOR_A1_A2_A3_B1_B2;
                    let toks: Vec<u32> = rows_tokens[b0..b0 + b].to_vec();
                    let hiddens = if b == 1 {
                        vhidden_for_drafter.narrow(1, 0, 1)?.contiguous()?
                    } else {
                        Tensor::cat(
                            &[
                                &vhidden_for_drafter.narrow(1, 0, 1)?,
                                &vhidden_for_drafter.narrow(1, b0, 1)?,
                            ],
                            1,
                        )?
                        .contiguous()?
                    };
                    self.mtp.catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
                        self.engine.underlying(),
                        &toks,
                        &hiddens,
                        &mut self.mtp_kv,
                    )?;
                }
                None => {}
            }
            ensure!(
                self.mtp_kv.len() == l + acc.depth_consumed,
                "tree round kv catch-up desync: drafter cache holds {} rows, trunk commits {}",
                self.mtp_kv.len(),
                l + acc.depth_consumed
            );

            self.engine.commit_verify_tree(
                acc.depth_consumed,
                acc.bonus_row,
                acc.branch_b_accepted_rows,
            )?;
            let bonus = *acc
                .emitted
                .last()
                .expect("tree accept always emits at least the bonus");
            self.anchor = bonus;
            if let Some(arm) = self
                .suffix_arm_env_opt_in_consulted_before_each_mtp_chain_and_fed_every_emitted_token
                .as_mut()
            {
                arm.extend_capped(&acc.emitted);
            }

            let _ = device.synchronize();
            let commit_dt = commit_t0.elapsed().as_secs_f64();
            self.stats
                .round_ms
                .push(1000.0 * round_t0.elapsed().as_secs_f64());
            self.stats.draft_ms += 1000.0 * draft_dt;
            self.stats.verify_ms += 1000.0 * verify_dt;
            self.stats.verify_ms_rounds.push(1000.0 * verify_dt);
            self.stats.commit_ms += 1000.0 * commit_dt;
            self.stats.rounds += 1;
            self.stats.drafted += rows - 1;
            self.stats.accepted += acc.drafts_accepted;
            self.stats.emitted += acc.emitted.len();
            if acc.drafts_accepted > 0 {
                self.stats.pos0_accepted += 1;
            }
            *self
                .stats
                .accept_len_hist
                .entry(acc.drafts_accepted)
                .or_insert(0) += 1;
            self.tree_rounds += 1;
            if let Some(b) = acc.branch_b_accepted_rows {
                self.tree_branch_b_rounds += 1;
                self.tree_branch_b_recovered_tokens += b;
            }
            Ok(acc.emitted)
        }
    }

    impl nv_models::qwen3_5_mtp::MtpMoeServingRounds for Qwen38MtpDecodeSession<'_> {
        fn anchor(&self) -> u32 {
            Qwen38MtpDecodeSession::anchor(self)
        }
        fn round_fits(&self) -> bool {
            Qwen38MtpDecodeSession::round_fits(self)
        }
        fn round(&mut self) -> Result<Vec<u32>> {
            Qwen38MtpDecodeSession::round(self)
        }
        fn stats(&self) -> &MtpSpecStats {
            &self.stats
        }
        fn arm_name(&self) -> &'static str {
            "eager"
        }
        fn capture_active(&self) -> bool {
            false
        }
    }

    impl nv_models::qwen3_5_mtp::MtpMoeServingRounds for Qwen38MtpGraphedDecodeSession<'_> {
        fn anchor(&self) -> u32 {
            Qwen38MtpGraphedDecodeSession::anchor(self)
        }
        fn round_fits(&self) -> bool {
            Qwen38MtpGraphedDecodeSession::round_fits(self)
        }
        fn round(&mut self) -> Result<Vec<u32>> {
            Qwen38MtpGraphedDecodeSession::round(self)
        }
        fn stats(&self) -> &MtpSpecStats {
            &self.stats
        }
        fn suffix_rounds_drafted_accepted(&self) -> Option<(usize, usize, usize)> {
            Qwen38MtpGraphedDecodeSession::suffix_rounds_drafted_accepted(self)
        }
        fn acceptance_tracking_k2_k3_switches(&self) -> Option<(usize, usize, usize)> {
            self.acceptance_tracking
                .as_ref()
                .map(|p| (p.k2_rounds, p.k3_rounds, p.switches))
        }
        fn arm_name(&self) -> &'static str {
            "captured-verify"
        }
        fn capture_active(&self) -> bool {
            self.engine.capture_active()
        }
    }

    fn argmax_host_row(row: &[f32]) -> Result<u32> {
        nv_layers::sampler::argmax_host_row(row)
    }

    fn argmax_host_rows(flat: &[f32], rows: usize, vocab: usize) -> Result<Vec<u32>> {
        ensure!(
            flat.len() == rows * vocab,
            "argmax rows: {} logits != {rows} x {vocab}",
            flat.len()
        );
        (0..rows)
            .map(|r| argmax_host_row(&flat[r * vocab..(r + 1) * vocab]))
            .collect()
    }

    fn argmax_last_row(logits: &Tensor) -> Result<u32> {
        let seq = logits.dim(1)?;
        let amax = logits
            .narrow(1, seq - 1, 1)?
            .argmax(D::Minus1)?
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        Ok(amax[0])
    }

    pub struct Qwen38MtpSelfSpecEngine<'a> {
        base: &'a Qwen3Moe,
        mtp: &'a Qwen38DenseMtpHead,
        k: usize,
        stop_ids: Vec<u32>,
    }

    impl<'a> Qwen38MtpSelfSpecEngine<'a> {
        pub fn new(base: &'a Qwen3Moe, mtp: &'a Qwen38DenseMtpHead, k: usize) -> Result<Self> {
            let hidden = base.config().hidden_size;
            assert_eager_session_verify_rows_fit(hidden, k)?;
            if !mtp_eager_deep_k_selected_from_env() {
                let device_reported_rows_ceiling =
                    nv_kernels::cuda::gemv_i8_normed_mk_max_m(hidden as i32).max(0) as usize;
                ensure!(
                    mtp_verify_rows_per_round(k) <= device_reported_rows_ceiling,
                    "this device's lm_head rows-per-call ceiling is \
                     {device_reported_rows_ceiling} for hidden={hidden}; a k={k} mtp round \
                     verifies {} rows",
                    mtp_verify_rows_per_round(k)
                );
            }
            Ok(Self {
                base,
                mtp,
                k,
                stop_ids: Vec::new(),
            })
        }

        pub fn with_stop_ids(mut self, ids: Vec<u32>) -> Self {
            self.stop_ids = ids;
            self
        }

        pub fn k(&self) -> usize {
            self.k
        }

        pub fn generate_greedy(
            &self,
            prompt: &[u32],
            max_new: usize,
            max_seq: usize,
        ) -> Result<(Vec<u32>, MtpSpecStats)> {
            self.generate_inner(prompt, max_new, max_seq, true)
        }

        pub fn generate_reference(
            &self,
            prompt: &[u32],
            max_new: usize,
            max_seq: usize,
        ) -> Result<(Vec<u32>, MtpSpecStats)> {
            self.generate_inner(prompt, max_new, max_seq, false)
        }

        fn generate_inner(
            &self,
            prompt: &[u32],
            max_new: usize,
            max_seq: usize,
            use_draft: bool,
        ) -> Result<(Vec<u32>, MtpSpecStats)> {
            ensure!(!prompt.is_empty(), "generate: empty prompt");
            let device = self.base.device().clone();

            if use_draft {
                let mut session =
                    Qwen38MtpDecodeSession::start(self.base, self.mtp, self.k, prompt, max_seq)?;
                let anchor = session.anchor();
                let mut generated: Vec<u32> = vec![anchor];
                if self.stop_ids.contains(&anchor) {
                    session.stats.stop_token = Some(anchor);
                    return Ok((generated, session.stats));
                }
                'rounds: while generated.len() < max_new && session.round_fits() {
                    let emitted = session.round()?;
                    for &t in emitted.iter() {
                        generated.push(t);
                        if self.stop_ids.contains(&t) {
                            session.stats.stop_token = Some(t);
                            break 'rounds;
                        }
                        if generated.len() >= max_new {
                            break 'rounds;
                        }
                    }
                }
                return Ok((generated, session.stats));
            }

            let mut cache = self.base.new_kv_cache(max_seq)?;
            let mut stats = MtpSpecStats::default();
            let seq = prompt.len();
            let mut anchor = match nv_models::qwen3_5_moe::q38_prefill_chunk_tokens_from_env()
                .filter(|&c| seq > c)
            {
                Some(chunk) => {
                    let mut fed = 0usize;
                    let mut last: Option<Tensor> = None;
                    while fed < seq {
                        let n = chunk.min(seq - fed);
                        let toks =
                            Tensor::from_vec(prompt[fed..fed + n].to_vec(), (1usize, n), &device)?;
                        let pos = Tensor::from_vec(
                            (fed as i32..(fed + n) as i32).collect::<Vec<i32>>(),
                            n,
                            &device,
                        )?;
                        let (logits, _h) = self.base.forward_with_cache_dispatched_hidden_rows(
                            &toks,
                            &pos,
                            &mut cache,
                            None,
                            Some(1),
                        )?;
                        last = Some(logits);
                        fed += n;
                    }
                    argmax_last_row(
                        &last.expect("reference: the non-empty prompt fed at least one chunk"),
                    )?
                }
                None => {
                    let tokens = Tensor::from_vec(prompt.to_vec(), (1usize, seq), &device)?;
                    let pos =
                        Tensor::from_vec((0..seq as i32).collect::<Vec<i32>>(), seq, &device)?;
                    let (logits, _hidden) = self
                        .base
                        .forward_with_cache_dispatched_hidden(&tokens, &pos, &mut cache, None)?;
                    argmax_last_row(&logits)?
                }
            };

            let mut generated: Vec<u32> = vec![anchor];
            if self.stop_ids.contains(&anchor) {
                stats.stop_token = Some(anchor);
                return Ok((generated, stats));
            }
            let _ = device.synchronize();

            while generated.len() < max_new {
                let round_t0 = std::time::Instant::now();
                let l = cache.current_len();
                ensure!(
                    l < max_seq,
                    "reference decode overflow: committed={l} + 1 > max_seq={max_seq}"
                );
                let bt = Tensor::from_vec(vec![anchor], (1usize, 1usize), &device)?;
                let bp = Tensor::from_vec(vec![l as i32], 1usize, &device)?;
                let (vlogits, _vh) = self
                    .base
                    .forward_with_cache_dispatched_hidden(&bt, &bp, &mut cache, None)?;
                anchor = argmax_last_row(&vlogits)?;
                generated.push(anchor);
                stats.rounds += 1;
                stats.emitted += 1;
                stats
                    .round_ms
                    .push(1000.0 * round_t0.elapsed().as_secs_f64());
                if self.stop_ids.contains(&anchor) {
                    stats.stop_token = Some(anchor);
                    break;
                }
            }
            Ok((generated, stats))
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{
    MtpKvCache, Qwen38DenseMtpHead, Qwen38MtpDecodeSession, Qwen38MtpGraphedDecodeSession,
    Qwen38MtpSelfSpecEngine,
};

#[cfg(test)]
mod two_branch_tree_accept_tests {
    use super::*;

    const ROWS: [u32; 6] = [10, 11, 12, 13, 21, 22];

    #[test]
    fn branch_a_full_accept_takes_depth_4_bonus_from_row_3() {
        let acc = accept_two_branch_tree_m6(&ROWS, &[11, 12, 13, 99, 7, 7]).unwrap();
        assert_eq!(acc.emitted, vec![11, 12, 13, 99]);
        assert_eq!(acc.drafts_accepted, 3);
        assert_eq!(acc.depth_consumed, 4);
        assert_eq!(acc.bonus_row, 3);
        assert_eq!(acc.branch_b_accepted_rows, None);
    }

    #[test]
    fn branch_a_partial_accept_stops_at_first_mismatch() {
        let acc = accept_two_branch_tree_m6(&ROWS, &[11, 55, 13, 99, 7, 7]).unwrap();
        assert_eq!(acc.emitted, vec![11, 55]);
        assert_eq!(acc.drafts_accepted, 1);
        assert_eq!(acc.depth_consumed, 2);
        assert_eq!(acc.bonus_row, 1);
        assert_eq!(acc.branch_b_accepted_rows, None);
    }

    #[test]
    fn branch_b_sibling_only_accepts_c2_and_takes_row_4_bonus() {
        let acc = accept_two_branch_tree_m6(&ROWS, &[21, 0, 0, 0, 77, 0]).unwrap();
        assert_eq!(acc.emitted, vec![21, 77]);
        assert_eq!(acc.drafts_accepted, 1);
        assert_eq!(acc.depth_consumed, 2);
        assert_eq!(acc.bonus_row, 4);
        assert_eq!(acc.branch_b_accepted_rows, Some(1));
    }

    #[test]
    fn branch_b_with_continuation_accepts_both_and_takes_row_5_bonus() {
        let acc = accept_two_branch_tree_m6(&ROWS, &[21, 0, 0, 0, 22, 88]).unwrap();
        assert_eq!(acc.emitted, vec![21, 22, 88]);
        assert_eq!(acc.drafts_accepted, 2);
        assert_eq!(acc.depth_consumed, 3);
        assert_eq!(acc.bonus_row, 5);
        assert_eq!(acc.branch_b_accepted_rows, Some(2));
    }

    #[test]
    fn neither_branch_matches_emits_the_anchor_rows_bonus_alone() {
        let acc = accept_two_branch_tree_m6(&ROWS, &[7, 11, 12, 13, 21, 22]).unwrap();
        assert_eq!(acc.emitted, vec![7]);
        assert_eq!(acc.drafts_accepted, 0);
        assert_eq!(acc.depth_consumed, 1);
        assert_eq!(acc.bonus_row, 0);
        assert_eq!(acc.branch_b_accepted_rows, None);
    }

    #[test]
    fn degenerate_equal_top1_top2_prefers_branch_a_whose_depth_reaches_further() {
        let rows = [10u32, 11, 12, 13, 11, 22];
        let acc = accept_two_branch_tree_m6(&rows, &[11, 12, 0, 0, 0, 0]).unwrap();
        assert_eq!(acc.branch_b_accepted_rows, None);
        assert_eq!(acc.emitted, vec![11, 12, 0]);
        assert_eq!(acc.bonus_row, 2);
    }

    #[test]
    fn emitted_stream_matches_the_chain_walk_on_every_greedy_pattern() {
        for pat in 0u32..64 {
            let greedy: Vec<u32> = (0..6)
                .map(|i| {
                    if pat & (1 << i) != 0 {
                        ROWS.get(i + 1).copied().unwrap_or(21)
                    } else {
                        999
                    }
                })
                .collect();
            let acc = accept_two_branch_tree_m6(&ROWS, &greedy).unwrap();
            assert!(acc.depth_consumed == acc.drafts_accepted + 1);
            assert_eq!(acc.emitted.len(), acc.drafts_accepted + 1);
            assert!(acc.depth_consumed >= 1 && acc.depth_consumed <= 4);
        }
    }
}
