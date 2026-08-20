#![allow(dead_code)]

use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

pub const WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM: usize = 200;
const PLATEAU_WINDOW_4_STEPS_MEDIAN_VS_THE_PREVIOUS_4: usize = 4;
const PLATEAU_FLOOR_8_STEPS_THE_OLD_FIXED_WARMUP_THAT_DID_NOT_RAMP_CLOCKS: usize = 8;
const PLATEAU_RATIO_LAST_WINDOW_MEDIAN_WITHIN_10_PCT_OF_PREVIOUS: f64 = 0.10;
const PLATEAU_CAP_15_SECS_SO_A_SLOW_DEEP_RUNG_CANNOT_WARM_FOREVER: f64 = 15.0;

pub fn serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value(
) -> MutexGuard<'static, ()> {
    static ONE_GPU_TEST_AT_A_TIME: OnceLock<Mutex<()>> = OnceLock::new();
    ONE_GPU_TEST_AT_A_TIME
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned_by_a_sibling_test_panic| poisoned_by_a_sibling_test_panic.into_inner())
}

fn median_of_window(window: &[f64]) -> f64 {
    let mut sorted = window.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
}

pub fn warmup_until_step_time_plateaus_because_post_idle_clock_ramp_reads_2x_slow(
    one_untimed_decode_step: &mut dyn FnMut(),
) -> usize {
    let started = Instant::now();
    let mut ms: Vec<f64> = Vec::new();
    loop {
        let t0 = Instant::now();
        one_untimed_decode_step();
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
        let n = ms.len();
        if n >= PLATEAU_FLOOR_8_STEPS_THE_OLD_FIXED_WARMUP_THAT_DID_NOT_RAMP_CLOCKS
            && n >= 2 * PLATEAU_WINDOW_4_STEPS_MEDIAN_VS_THE_PREVIOUS_4
        {
            let w = PLATEAU_WINDOW_4_STEPS_MEDIAN_VS_THE_PREVIOUS_4;
            let last = median_of_window(&ms[n - w..]);
            let prev = median_of_window(&ms[n - 2 * w..n - w]);
            if prev > 0.0
                && (last - prev).abs()
                    <= PLATEAU_RATIO_LAST_WINDOW_MEDIAN_WITHIN_10_PCT_OF_PREVIOUS * prev
            {
                return n;
            }
        }
        if n >= WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
            || started.elapsed().as_secs_f64()
                >= PLATEAU_CAP_15_SECS_SO_A_SLOW_DEEP_RUNG_CANNOT_WARM_FOREVER
        {
            return n;
        }
    }
}

pub fn warmup_to_plateau_then_time_steps(
    mut one_decode_step: impl FnMut(),
    timed_steps: usize,
) -> (usize, Vec<f64>) {
    let warmup_steps =
        warmup_until_step_time_plateaus_because_post_idle_clock_ramp_reads_2x_slow(
            &mut one_decode_step,
        );
    let mut step_ms = Vec::with_capacity(timed_steps);
    for _ in 0..timed_steps {
        let t0 = Instant::now();
        one_decode_step();
        step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    (warmup_steps, step_ms)
}
