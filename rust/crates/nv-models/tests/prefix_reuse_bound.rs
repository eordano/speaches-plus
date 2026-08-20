use nv_models::gemma4::{kv_fp8_ring_slots, VERIFY_PREFILL_CHUNK};
use nv_models::prefix_reuse::{
    common_prefix_len, exact_extend_target, RewindLimits, RING_REWIND_RESERVED_SLOTS,
};

const WINDOW: usize = 512;
const CHUNK: usize = 256;

fn ring_still_holds_what_a_rewind_rereads(
    ring: usize,
    window: usize,
    frontier: usize,
    target: usize,
) -> bool {
    let oldest = (target + 1).saturating_sub(window);
    (oldest..target).all(|p| frontier - p <= ring)
}

fn advance_the_ring_survives(ring: usize, window: usize, target: usize) -> usize {
    (0..=(ring + window + 8))
        .take_while(|&a| ring_still_holds_what_a_rewind_rereads(ring, window, target + a, target))
        .last()
        .unwrap_or(0)
}

#[test]
fn a_linear_cache_readmits_a_prefix_however_far_the_last_run_ran() {
    let l = RewindLimits::positional(0, WINDOW, 0);
    assert_eq!(l.max_advance(), Some(usize::MAX));
    assert!(l.admits(1_000_000, 1));
    assert!(l.admits(96_000, 95_999));
    assert!(
        !l.admits(100, 101),
        "a rewind cannot invent unwritten positions"
    );
    assert!(!l.admits(100, 0), "position 0 is a reset, not a reuse");
}

#[test]
fn the_ring_bound_is_what_the_ring_survives_less_the_reserved_slots() {
    for window in [WINDOW, 1024, 4096] {
        let ring = kv_fp8_ring_slots(window);
        let l = RewindLimits::positional(ring, window, 0);
        let max = l.max_advance().expect("a ring is still position-indexed");
        let survives = advance_the_ring_survives(ring, window, 4096);
        assert_eq!(
            max,
            survives - RING_REWIND_RESERVED_SLOTS,
            "window {window}: the bound must sit exactly {RING_REWIND_RESERVED_SLOTS} slot(s) \
             inside the advance the ring physically survives"
        );
        assert!(
            max > VERIFY_PREFILL_CHUNK,
            "window {window}: a ring that cannot outlive one prefill chunk could never readmit \
             a chunk-aligned prefix"
        );
    }
}

#[test]
fn a_rewind_is_refused_wherever_the_ring_no_longer_holds_what_it_would_re_read() {
    let window = WINDOW;
    let ring = kv_fp8_ring_slots(window);
    let l = RewindLimits::positional(ring, window, 0);
    for target in [1usize, 999, 40_000] {
        let survives = advance_the_ring_survives(ring, window, target);
        for advance in 0..(survives + 8) {
            let admitted = l.admits(target + advance, target);
            let sound =
                ring_still_holds_what_a_rewind_rereads(ring, window, target + advance, target);
            assert!(
                !admitted || sound,
                "target {target}: admitted an advance of {advance} the ring has already written \
                 over"
            );
            assert!(
                admitted || advance + RING_REWIND_RESERVED_SLOTS > survives || target < window,
                "target {target}: refused an advance of {advance} the ring survives with room \
                 to spare"
            );
        }
    }
}

#[test]
fn a_wider_window_in_the_same_ring_buys_less_reuse() {
    let ring = kv_fp8_ring_slots(WINDOW);
    let wide = RewindLimits::positional(ring, WINDOW + 64, 0);
    let narrow = RewindLimits::positional(ring, WINDOW, 0);
    assert_eq!(
        wide.max_advance().unwrap() + 64,
        narrow.max_advance().unwrap()
    );
    let tight = RewindLimits::positional(WINDOW, WINDOW, 0);
    assert_eq!(
        tight.max_advance(),
        Some(advance_the_ring_survives(WINDOW, WINDOW, 4096) - RING_REWIND_RESERVED_SLOTS)
    );
    assert_eq!(
        tight.max_advance(),
        Some(0),
        "a ring no bigger than its window readmits a prefix only where it already sits"
    );
}

#[test]
fn a_decoder_carrying_state_that_is_not_indexed_by_position_refuses_outright() {
    let l = RewindLimits::NONE;
    assert_eq!(l.max_advance(), None);
    assert!(!l.admits(4096, 4095));
    assert_eq!(l.target(4096, 4095), None);
}

#[test]
fn a_reused_prefill_restarts_on_a_chunk_boundary() {
    let l = RewindLimits::positional(0, WINDOW, CHUNK);
    assert_eq!(l.target(4096, 1000), Some(768));
    assert_eq!(l.target(4096, 1024), Some(1024));
    assert_eq!(
        l.target(4096, CHUNK - 1),
        None,
        "a prefix shorter than one chunk rounds down to a reset"
    );
    assert!(!l.admits(4096, 1000), "off-boundary targets are refused");
    assert!(l.admits(4096, 768));
}

#[test]
fn the_chunk_boundary_a_target_lands_on_is_the_one_a_cold_run_would_have_used() {
    let l = RewindLimits::positional(0, WINDOW, CHUNK);
    for lcp in 0..(CHUNK * 5) {
        let Some(t) = l.target(CHUNK * 5, lcp) else {
            assert!(lcp < CHUNK);
            continue;
        };
        assert!(t <= lcp && t.is_multiple_of(CHUNK));
        assert!(lcp - t < CHUNK);
    }
}

#[test]
fn a_prompt_that_extends_the_folded_stream_resumes_at_the_frontier_without_rewinding() {
    assert_eq!(
        exact_extend_target(10, 12, 10),
        Some(10),
        "resuming at the live frontier moves no bytes and no cursor, so it is legal even for a \
         decoder whose state cannot rewind"
    );
    assert_eq!(
        exact_extend_target(10, 10, 10),
        Some(10),
        "an lcp equal to the frontier is an exact extension"
    );
}

#[test]
fn exact_extension_refuses_divergence_and_a_decoder_that_left_the_frontier() {
    assert_eq!(
        exact_extend_target(10, 9, 10),
        None,
        "divergence before the frontier needs a rewind, which admission must plan separately"
    );
    assert_eq!(
        exact_extend_target(10, 12, 7),
        None,
        "a live position off the frontier means the cached tokens no longer describe the state"
    );
    assert_eq!(
        exact_extend_target(10, 12, 13),
        None,
        "committed-but-unemitted tokens past the recorded stream refuse exact extension"
    );
    assert_eq!(
        exact_extend_target(0, 5, 0),
        None,
        "an empty cache has nothing to resume"
    );
}

#[test]
fn the_prefix_of_two_prompts_stops_at_the_first_divergent_token() {
    assert_eq!(common_prefix_len(&[1, 2, 3, 4], &[1, 2, 9, 4]), 2);
    assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3]), 3);
    assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3, 4]), 3);
    assert_eq!(common_prefix_len(&[], &[1]), 0);
    assert_eq!(common_prefix_len(&[9], &[1]), 0);
}
