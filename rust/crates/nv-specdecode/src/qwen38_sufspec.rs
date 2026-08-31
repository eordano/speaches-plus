use nv_lookup::SuffixAutomaton;

pub const NV_Q38_SPEC_SUFFIX_ENV: &str = "NV_Q38_SPEC_SUFFIX";

pub fn spec_suffix_selected(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

pub fn spec_suffix_selected_from_env() -> bool {
    spec_suffix_selected(std::env::var(NV_Q38_SPEC_SUFFIX_ENV).ok().as_deref())
}

pub const SUFFIX_DRAFT_MAX_7_LOSSLESS_AFTER_THE_M67HUNT_AND_ARMV_PLAN_FIXES_STREAMS_BITWISE_THE_CLAMP5_HASHES_ON_EVERY_CLASS_BOTH_TC_SETTINGS_PLUS_5PCT_AGENTIC: usize = 7;

pub const NV_Q38_SUFFIX_DRAFT_MAX_ENV: &str = "NV_Q38_SUFFIX_DRAFT_MAX";

pub const SUFFIX_DRAFT_OVERRIDE_CEILING_7_BECAUSE_PAST_M8_NOTHING_IS_MEASURED_AND_THE_M_INVARIANCE_SUITE_COVERS_ONLY_4_TO_8: usize = 7;

pub fn suffix_draft_max(raw: Option<&str>) -> usize {
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return SUFFIX_DRAFT_MAX_7_LOSSLESS_AFTER_THE_M67HUNT_AND_ARMV_PLAN_FIXES_STREAMS_BITWISE_THE_CLAMP5_HASHES_ON_EVERY_CLASS_BOTH_TC_SETTINGS_PLUS_5PCT_AGENTIC;
    };
    let v: usize = s.parse().unwrap_or_else(|_| {
        panic!("{NV_Q38_SUFFIX_DRAFT_MAX_ENV}={s:?} is not a draft count")
    });
    assert!(
        (1..=SUFFIX_DRAFT_OVERRIDE_CEILING_7_BECAUSE_PAST_M8_NOTHING_IS_MEASURED_AND_THE_M_INVARIANCE_SUITE_COVERS_ONLY_4_TO_8)
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
    (SUFFIX_MIN_DRAFT_4_A_SUFFIX_ROUND_MUST_OFFER_STRICTLY_MORE_THAN_THE_K3_MTP_CHAIN_ELSE_THE_MTP_ROUND_IS_THE_BETTER_BET + 1
        ..=draft_max + 1)
        .collect()
}

pub const SUFFIX_MIN_MATCH_4_MEASURED_BEST_EST_TOKENS_PER_ROUND_ON_ALL_THREE_HEADROOM_CLASSES_WHILE_2_ADMITS_SUB_MTP_ACCEPTS:
    usize = 4;

pub const SUFFIX_MIN_DRAFT_4_A_SUFFIX_ROUND_MUST_OFFER_STRICTLY_MORE_THAN_THE_K3_MTP_CHAIN_ELSE_THE_MTP_ROUND_IS_THE_BETTER_BET: usize =
    4;

pub const SUFFIX_TREE_TOKEN_CAP_262144_ARCTIC_STYLE_MEMORY_BOUND_EXTENSION_AND_PROPOSALS_BOTH_STOP_AT_THE_CAP_BECAUSE_A_FROZEN_TREE_TAIL_NO_LONGER_ENDS_AT_THE_ANCHOR:
    usize = 262144;

pub struct SuffixSpecArm {
    sam: SuffixAutomaton,
    draft_max: usize,
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
            (1..=SUFFIX_DRAFT_OVERRIDE_CEILING_7_BECAUSE_PAST_M8_NOTHING_IS_MEASURED_AND_THE_M_INVARIANCE_SUITE_COVERS_ONLY_4_TO_8)
                .contains(&draft_max),
            "suffix draft max {draft_max} is outside the measured 1..=7 band"
        );
        Self {
            sam: SuffixAutomaton::new(),
            draft_max,
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
        if cap < SUFFIX_MIN_DRAFT_4_A_SUFFIX_ROUND_MUST_OFFER_STRICTLY_MORE_THAN_THE_K3_MTP_CHAIN_ELSE_THE_MTP_ROUND_IS_THE_BETTER_BET {
            return None;
        }
        self.sam
            .propose(
                cap,
                SUFFIX_MIN_MATCH_4_MEASURED_BEST_EST_TOKENS_PER_ROUND_ON_ALL_THREE_HEADROOM_CLASSES_WHILE_2_ADMITS_SUB_MTP_ACCEPTS,
            )
            .filter(|p| {
                p.tokens.len()
                    >= SUFFIX_MIN_DRAFT_4_A_SUFFIX_ROUND_MUST_OFFER_STRICTLY_MORE_THAN_THE_K3_MTP_CHAIN_ELSE_THE_MTP_ROUND_IS_THE_BETTER_BET
            })
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
    fn env_opt_in_is_exactly_the_string_1() {
        assert!(spec_suffix_selected(Some("1")));
        assert!(spec_suffix_selected(Some(" 1 ")));
        assert!(!spec_suffix_selected(Some("0")));
        assert!(!spec_suffix_selected(Some("true")));
        assert!(!spec_suffix_selected(None));
    }

    #[test]
    fn repeated_block_proposes_and_short_headroom_declines() {
        let mut arm = SuffixSpecArm::new();
        let block: Vec<u32> = (0..24).collect();
        arm.extend_capped(&block);
        arm.extend_capped(&[900, 901]);
        arm.extend_capped(&block[..20]);
        let p = arm.propose_for_round(64).expect("a 20-token re-match must propose");
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
        assert_eq!(arm.indexed_len(), cap, "extension past the cap must be a no-op");
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
                SUFFIX_DRAFT_OVERRIDE_CEILING_7_BECAUSE_PAST_M8_NOTHING_IS_MEASURED_AND_THE_M_INVARIANCE_SUITE_COVERS_ONLY_4_TO_8
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
        let p = arm.propose_for_round(64).expect("a 20-token re-match must propose");
        assert_eq!(p, vec![20, 21, 22, 23, 900, 901, 902]);
        assert!(
            arm.propose_for_round(3).is_none(),
            "kv headroom below the suffix minimum still falls back to the mtp chain"
        );
    }
}
