use nv_lookup::SuffixAutomaton;

pub const NV_Q38_SPEC_SUFFIX_ENV: &str = "NV_Q38_SPEC_SUFFIX";

pub const NV_Q38_SPEC_SUFFIX_DEFAULT_ON_0_RESTORES_THE_PLAIN_MTP_CHAIN_FLIPPED_2026_09_04_BITWISE_LOSSLESS_BY_QWEN38_SUFSPEC_GATE_AND_279_264_147_TOK_S_VS_THE_PLAIN_166_156_99: &str =
    NV_Q38_SPEC_SUFFIX_ENV;

pub fn spec_suffix_selected(raw: Option<&str>) -> bool {
    !matches!(raw.map(str::trim), Some("0"))
}

pub const SUFFIX_LOSSLESSNESS_IS_CERTIFIED_ONLY_AT_MTP_CHAIN_DEPTH_3_BECAUSE_QWEN38_SUFSPEC_GATE_DIVERGES_ON_TILED_AND_FREEFORM_AT_K4: usize = 3;
pub const NV_Q38_SPEC_SUFFIX_K2_UNCERTIFIED_ENV: &str = "NV_Q38_SPEC_SUFFIX_K2_UNCERTIFIED";
pub const SUFFIX_K2_OVERRIDE_IS_OPT_IN_BECAUSE_K2_VERIFY_REASSOCIATES_AND_FAILED_THE_THREE_CLASS_WALK: usize = 2;

pub fn spec_suffix_selected_with_uncertified_k2_opt_in(
    raw: Option<&str>,
    k: usize,
    override_raw: Option<&str>,
) -> bool {
    spec_suffix_selected_at_a_certified_chain_depth(raw, k)
        || (spec_suffix_selected(raw)
            && k == SUFFIX_K2_OVERRIDE_IS_OPT_IN_BECAUSE_K2_VERIFY_REASSOCIATES_AND_FAILED_THE_THREE_CLASS_WALK
            && override_raw == Some("1"))
}

pub fn spec_suffix_selected_at_a_certified_chain_depth(raw: Option<&str>, k: usize) -> bool {
    spec_suffix_selected(raw)
        && k == SUFFIX_LOSSLESSNESS_IS_CERTIFIED_ONLY_AT_MTP_CHAIN_DEPTH_3_BECAUSE_QWEN38_SUFSPEC_GATE_DIVERGES_ON_TILED_AND_FREEFORM_AT_K4
}

pub fn spec_suffix_selected_from_env() -> bool {
    let raw = std::env::var(NV_Q38_SPEC_SUFFIX_ENV).ok();
    let k = crate::qwen38_mtp::mtp_chain_depth_from_env();
    let override_raw = std::env::var(NV_Q38_SPEC_SUFFIX_K2_UNCERTIFIED_ENV).ok();
    let on =
        spec_suffix_selected_with_uncertified_k2_opt_in(raw.as_deref(), k, override_raw.as_deref());
    if on && k == SUFFIX_K2_OVERRIDE_IS_OPT_IN_BECAUSE_K2_VERIFY_REASSOCIATES_AND_FAILED_THE_THREE_CLASS_WALK {
        static NOTED_K2: std::sync::Once = std::sync::Once::new();
        NOTED_K2.call_once(|| eprintln!("[qwen3.8-sufspec] explicit uncertified k2+suffix experiment; plain k2 failed its quality certificate and this combined route requires its own certificate"));
    }
    if spec_suffix_selected(raw.as_deref()) && !on {
        static NOTED: std::sync::Once = std::sync::Once::new();
        NOTED.call_once(|| {
            eprintln!(
                "[qwen3.8-sufspec] {NV_Q38_SPEC_SUFFIX_ENV}=1 IGNORED at NV_MTP_K={k}: the \
                 suffix hybrid's greedy-emitted-stream identity gate (qwen38_sufspec_gate) \
                 passes only at chain depth {}, and diverges on the tiled and freeform classes \
                 at k=4, so arming it here would change WHAT is emitted. Set NV_MTP_K={} to \
                 take the certified suffix arm.",
                SUFFIX_LOSSLESSNESS_IS_CERTIFIED_ONLY_AT_MTP_CHAIN_DEPTH_3_BECAUSE_QWEN38_SUFSPEC_GATE_DIVERGES_ON_TILED_AND_FREEFORM_AT_K4,
                SUFFIX_LOSSLESSNESS_IS_CERTIFIED_ONLY_AT_MTP_CHAIN_DEPTH_3_BECAUSE_QWEN38_SUFSPEC_GATE_DIVERGES_ON_TILED_AND_FREEFORM_AT_K4
            );
        });
    }
    on
}

pub const SUFFIX_DRAFT_MAX_7_LOSSLESS_AFTER_THE_M67HUNT_AND_ARMV_PLAN_FIXES_STREAMS_BITWISE_THE_CLAMP5_HASHES_ON_EVERY_CLASS_BOTH_TC_SETTINGS_PLUS_5PCT_AGENTIC: usize = 7;

pub const NV_Q38_SUFFIX_DRAFT_MAX_ENV: &str = "NV_Q38_SUFFIX_DRAFT_MAX";

pub const SUFFIX_DRAFT_OVERRIDE_CEILING_7_BECAUSE_DEEP_RUNGS_ARE_PRICED_BUT_THE_TCTILE_ARM_STOPS_AT_8_THE_GDN_MROW_CAP_AT_16_AND_ONLY_THE_TCTILE_CERT_MAY_FLIP_IT: usize = 7;

pub fn suffix_draft_max(raw: Option<&str>) -> usize {
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return SUFFIX_DRAFT_MAX_7_LOSSLESS_AFTER_THE_M67HUNT_AND_ARMV_PLAN_FIXES_STREAMS_BITWISE_THE_CLAMP5_HASHES_ON_EVERY_CLASS_BOTH_TC_SETTINGS_PLUS_5PCT_AGENTIC;
    };
    let v: usize = s
        .parse()
        .unwrap_or_else(|_| panic!("{NV_Q38_SUFFIX_DRAFT_MAX_ENV}={s:?} is not a draft count"));
    assert!(
        (1..=SUFFIX_DRAFT_OVERRIDE_CEILING_7_BECAUSE_DEEP_RUNGS_ARE_PRICED_BUT_THE_TCTILE_ARM_STOPS_AT_8_THE_GDN_MROW_CAP_AT_16_AND_ONLY_THE_TCTILE_CERT_MAY_FLIP_IT)
            .contains(&v),
        "{NV_Q38_SUFFIX_DRAFT_MAX_ENV}={v} refused: the m-invariance suite \
         (qwen38_verify_m_invariance) and the identity gate (qwen38_sufspec_gate) stop at \
         verify m=8, so 7 drafts is the deepest measurable suffix"
    );
    v
}

pub fn suffix_draft_max_from_env() -> usize {
    suffix_draft_max(std::env::var(NV_Q38_SUFFIX_DRAFT_MAX_ENV).ok().as_deref())
}

pub fn verify_shapes_for_draft_max(draft_max: usize) -> Vec<usize> {
    (SUFFIX_MIN_DRAFT_4_A_SUFFIX_ROUND_MUST_OFFER_STRICTLY_MORE_THAN_THE_K3_MTP_CHAIN_ELSE_THE_MTP_ROUND_IS_THE_BETTER_BET + 1..=draft_max + 1).collect()
}

pub const SUFFIX_MIN_MATCH_4_MEASURED_BEST_EST_TOKENS_PER_ROUND_ON_ALL_THREE_HEADROOM_CLASSES_WHILE_2_ADMITS_SUB_MTP_ACCEPTS:
    usize = 4;

pub const SUFFIX_MIN_MATCH_DEFAULT_8_BECAUSE_A_STRICTER_TRAILING_MATCH_STOPS_SUFFIX_ROUNDS_FIRING_ON_WEAK_REPEATS_RECOVERING_FREEFORM_63P0_TO_65P8_WITH_TILED_AND_THE_OAI_COMPARATOR_UNCHANGED_AT_255P8:
    usize = 8;

pub const NV_Q38_SUFFIX_MIN_MATCH_ENV: &str = "NV_Q38_SUFFIX_MIN_MATCH";

pub fn suffix_min_match(raw: Option<&str>) -> usize {
    let Some(v) = raw
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v >= 1)
    else {
        return SUFFIX_MIN_MATCH_DEFAULT_8_BECAUSE_A_STRICTER_TRAILING_MATCH_STOPS_SUFFIX_ROUNDS_FIRING_ON_WEAK_REPEATS_RECOVERING_FREEFORM_63P0_TO_65P8_WITH_TILED_AND_THE_OAI_COMPARATOR_UNCHANGED_AT_255P8;
    };
    v
}

pub fn suffix_min_match_from_env() -> usize {
    suffix_min_match(std::env::var(NV_Q38_SUFFIX_MIN_MATCH_ENV).ok().as_deref())
}

pub const SUFFIX_MIN_DRAFT_4_A_SUFFIX_ROUND_MUST_OFFER_STRICTLY_MORE_THAN_THE_K3_MTP_CHAIN_ELSE_THE_MTP_ROUND_IS_THE_BETTER_BET: usize =
    4;

pub fn suffix_min_draft_strictly_more_than_the_live_mtp_chain(k: usize) -> usize {
    (k + 1).max(
        SUFFIX_MIN_DRAFT_4_A_SUFFIX_ROUND_MUST_OFFER_STRICTLY_MORE_THAN_THE_K3_MTP_CHAIN_ELSE_THE_MTP_ROUND_IS_THE_BETTER_BET,
    )
}

pub const SUFFIX_TREE_TOKEN_CAP_262144_ARCTIC_STYLE_MEMORY_BOUND_EXTENSION_AND_PROPOSALS_BOTH_STOP_AT_THE_CAP_BECAUSE_A_FROZEN_TREE_TAIL_NO_LONGER_ENDS_AT_THE_ANCHOR:
    usize = 262144;

pub struct SuffixSpecArm {
    sam: SuffixAutomaton,
    draft_max: usize,
    min_draft: usize,
    min_match: usize,
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
}

impl Default for SuffixSpecArm {
    fn default() -> Self {
        Self::new()
    }
}

impl SuffixSpecArm {
    pub fn new() -> Self {
        Self::with_draft_max(suffix_draft_max_from_env())
    }

    pub fn with_draft_max(draft_max: usize) -> Self {
        assert!(
            (1..=SUFFIX_DRAFT_OVERRIDE_CEILING_7_BECAUSE_DEEP_RUNGS_ARE_PRICED_BUT_THE_TCTILE_ARM_STOPS_AT_8_THE_GDN_MROW_CAP_AT_16_AND_ONLY_THE_TCTILE_CERT_MAY_FLIP_IT)
                .contains(&draft_max),
            "suffix draft max {draft_max} is outside the measured 1..=7 band"
        );
        Self {
            sam: SuffixAutomaton::new(),
            draft_max,
            min_draft: SUFFIX_MIN_DRAFT_4_A_SUFFIX_ROUND_MUST_OFFER_STRICTLY_MORE_THAN_THE_K3_MTP_CHAIN_ELSE_THE_MTP_ROUND_IS_THE_BETTER_BET,
            min_match: suffix_min_match_from_env(),
            rounds: 0,
            drafted: 0,
            accepted: 0,
        }
    }

    pub fn draft_max(&self) -> usize {
        self.draft_max
    }

    pub fn indexed_len(&self) -> usize {
        self.sam.len()
    }

    pub fn extend_capped(&mut self, toks: &[u32]) {
        let room = SUFFIX_TREE_TOKEN_CAP_262144_ARCTIC_STYLE_MEMORY_BOUND_EXTENSION_AND_PROPOSALS_BOTH_STOP_AT_THE_CAP_BECAUSE_A_FROZEN_TREE_TAIL_NO_LONGER_ENDS_AT_THE_ANCHOR
            .saturating_sub(self.sam.len());
        self.sam.extend_slice(&toks[..toks.len().min(room)]);
    }

    pub fn at_cap(&self) -> bool {
        self.sam.len()
            >= SUFFIX_TREE_TOKEN_CAP_262144_ARCTIC_STYLE_MEMORY_BOUND_EXTENSION_AND_PROPOSALS_BOTH_STOP_AT_THE_CAP_BECAUSE_A_FROZEN_TREE_TAIL_NO_LONGER_ENDS_AT_THE_ANCHOR
    }

    pub fn propose_for_round(&self, kv_headroom_draft_rows: usize) -> Option<Vec<u32>> {
        if self.at_cap() {
            return None;
        }
        let cap = self.draft_max.min(kv_headroom_draft_rows);
        if cap < self.min_draft {
            return None;
        }
        self.sam
            .propose(cap, self.min_match)
            .filter(|p| p.tokens.len() >= self.min_draft)
            .map(|p| p.tokens)
    }

    pub fn verify_shapes_to_precapture() -> Vec<usize> {
        verify_shapes_for_draft_max(suffix_draft_max_from_env())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_is_opt_out_only_the_string_0_disarms() {
        assert!(spec_suffix_selected(Some("1")));
        assert!(spec_suffix_selected(None));
        assert!(spec_suffix_selected(Some("true")));
        assert!(!spec_suffix_selected(Some("0")));
        assert!(!spec_suffix_selected(Some(" 0 ")));
    }

    #[test]
    fn the_suffix_arm_refuses_any_chain_depth_its_identity_gate_did_not_certify() {
        assert!(spec_suffix_selected_at_a_certified_chain_depth(
            Some("1"),
            3
        ));
        assert!(
            spec_suffix_selected_at_a_certified_chain_depth(None, 3),
            "the arm is default-ON at the certified depth since 2026-09-04"
        );
        for k in [1usize, 2, 4, 5, 6, 7] {
            assert!(
                !spec_suffix_selected_at_a_certified_chain_depth(None, k),
                "an uncertified chain depth must disarm the default-on suffix arm at k={k}"
            );
            assert!(
                !spec_suffix_selected_at_a_certified_chain_depth(Some("1"), k),
                "qwen38_sufspec_gate proves the greedy emitted stream bitwise identical only at \
                 chain depth 3; at k={k} it diverges on tiled and freeform, so the arm must \
                 decline rather than silently change what is emitted"
            );
        }
        assert!(!spec_suffix_selected_at_a_certified_chain_depth(
            Some("0"),
            3
        ));
    }

    #[test]
    fn uncertified_k2_requires_its_own_explicit_override_and_suffix_zero_still_wins() {
        for k in 1..=7 {
            for raw in [None, Some("1"), Some("0"), Some(" 0 ")] {
                for opt in [None, Some("0"), Some("1"), Some("true"), Some(" 1")] {
                    assert_eq!(
                        spec_suffix_selected_with_uncertified_k2_opt_in(raw, k, opt),
                        spec_suffix_selected(raw) && (k == 3 || (k == 2 && opt == Some("1")))
                    );
                }
            }
        }
        assert_eq!(suffix_min_draft_strictly_more_than_the_live_mtp_chain(2), 4);
    }

    #[test]
    fn a_suffix_round_must_out_draft_the_live_mtp_chain() {
        assert_eq!(suffix_min_draft_strictly_more_than_the_live_mtp_chain(3), 4);
        assert_eq!(suffix_min_draft_strictly_more_than_the_live_mtp_chain(4), 5);
        assert_eq!(suffix_min_draft_strictly_more_than_the_live_mtp_chain(1), 4);
    }

    #[test]
    fn repeated_block_proposes_and_short_headroom_declines() {
        let mut arm = SuffixSpecArm::new();
        let block: Vec<u32> = (0..24).collect();
        arm.extend_capped(&block);
        arm.extend_capped(&[900, 901]);
        arm.extend_capped(&block[..20]);
        let p = arm
            .propose_for_round(64)
            .expect("a 20-token re-match must propose");
        assert_eq!(
            p.len(),
            SUFFIX_DRAFT_MAX_7_LOSSLESS_AFTER_THE_M67HUNT_AND_ARMV_PLAN_FIXES_STREAMS_BITWISE_THE_CLAMP5_HASHES_ON_EVERY_CLASS_BOTH_TC_SETTINGS_PLUS_5PCT_AGENTIC
        );
        assert_eq!(
            p,
            vec![20, 21, 22, 23, 900, 901, 0],
            "continuation must replay what followed the earlier occurrence"
        );
        assert!(
            arm.propose_for_round(3).is_none(),
            "a round with only 3 draft rows of kv headroom is below the suffix minimum and \
             must fall back to the mtp chain"
        );
    }

    #[test]
    fn short_match_stays_none_below_min_match() {
        let mut arm = SuffixSpecArm::new();
        arm.extend_capped(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3]);
        assert!(
            arm.propose_for_round(64).is_none(),
            "a 3-token trailing match is under SUFFIX_MIN_MATCH and must not draft"
        );
    }

    #[test]
    fn cap_freezes_extension_and_proposals_together() {
        let mut arm = SuffixSpecArm::new();
        let cap = SUFFIX_TREE_TOKEN_CAP_262144_ARCTIC_STYLE_MEMORY_BOUND_EXTENSION_AND_PROPOSALS_BOTH_STOP_AT_THE_CAP_BECAUSE_A_FROZEN_TREE_TAIL_NO_LONGER_ENDS_AT_THE_ANCHOR;
        let block: Vec<u32> = (0..64).collect();
        while arm.indexed_len() < cap {
            arm.extend_capped(&block);
        }
        assert_eq!(arm.indexed_len(), cap);
        arm.extend_capped(&block);
        assert_eq!(
            arm.indexed_len(),
            cap,
            "extension past the cap must be a no-op"
        );
        assert!(
            arm.propose_for_round(64).is_none(),
            "a frozen tree no longer tracks the anchor so proposing from it drafts stale context"
        );
    }

    #[test]
    fn precapture_shapes_cover_every_reachable_verify_m_at_the_shipped_and_override_maxima() {
        assert_eq!(
            verify_shapes_for_draft_max(
                SUFFIX_DRAFT_MAX_7_LOSSLESS_AFTER_THE_M67HUNT_AND_ARMV_PLAN_FIXES_STREAMS_BITWISE_THE_CLAMP5_HASHES_ON_EVERY_CLASS_BOTH_TC_SETTINGS_PLUS_5PCT_AGENTIC
            ),
            vec![5, 6, 7, 8]
        );
        assert_eq!(
            verify_shapes_for_draft_max(
                SUFFIX_DRAFT_OVERRIDE_CEILING_7_BECAUSE_DEEP_RUNGS_ARE_PRICED_BUT_THE_TCTILE_ARM_STOPS_AT_8_THE_GDN_MROW_CAP_AT_16_AND_ONLY_THE_TCTILE_CERT_MAY_FLIP_IT
            ),
            vec![5, 6, 7, 8]
        );
    }

    #[test]
    fn draft_max_override_defaults_to_the_shipped_clamp_and_parses_the_band() {
        assert_eq!(
            suffix_draft_max(None),
            SUFFIX_DRAFT_MAX_7_LOSSLESS_AFTER_THE_M67HUNT_AND_ARMV_PLAN_FIXES_STREAMS_BITWISE_THE_CLAMP5_HASHES_ON_EVERY_CLASS_BOTH_TC_SETTINGS_PLUS_5PCT_AGENTIC
        );
        assert_eq!(suffix_draft_max(Some(" 7 ")), 7);
        assert_eq!(suffix_draft_max(Some("1")), 1);
        assert_eq!(suffix_draft_max(Some("6")), 6);
    }

    #[test]
    #[should_panic(expected = "refused")]
    fn draft_max_8_is_refused_because_nothing_past_m8_verify_is_measured() {
        suffix_draft_max(Some("8"));
    }

    #[test]
    #[should_panic(expected = "refused")]
    fn draft_max_0_is_refused() {
        suffix_draft_max(Some("0"));
    }

    #[test]
    fn override_max_7_proposes_seven_and_the_shape_ladder_stays_reachable() {
        let mut arm = SuffixSpecArm::with_draft_max(7);
        let block: Vec<u32> = (0..24).collect();
        arm.extend_capped(&block);
        arm.extend_capped(&[900, 901, 902, 903]);
        arm.extend_capped(&block[..20]);
        let p = arm
            .propose_for_round(64)
            .expect("a 20-token re-match must propose");
        assert_eq!(p, vec![20, 21, 22, 23, 900, 901, 902]);
        assert!(
            arm.propose_for_round(3).is_none(),
            "kv headroom below the suffix minimum still falls back to the mtp chain"
        );
    }
}
