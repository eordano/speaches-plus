#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;

const SM_COUNT_ON_THE_ONE_BOX_EVERY_RUNG_IN_THIS_SUITE_WAS_MEASURED_ON: i32 = 188;

const WAVES_PER_SM_THE_PICKER_TARGETS_BECAUSE_ROUNDS_14_AND_15_MEASURED_THE_WINNER_AT_ONE_OR_TWO_TIMES_THE_SM_COUNT:
    i64 = 2;

const INSTANTIATED_RUNGS_ANY_OTHER_VALUE_FALLS_INTO_THE_SWITCH_DEFAULT_AND_LAUNCHES_A_GRID_ITS_TEMPLATE_DOES_NOT_MATCH:
    &[i32] = &[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 47];

const THE_SM_EXACT_RUNG_EVERY_FAMILY_HAD_BEFORE_THE_LADDER_GAINED_ITS_LANE_GROUP_RUNGS_BELOW_SIX: &[(i32, i32)] =
    &[(32, 12), (16, 24), (24, 16), (8, 47)];

const THE_LANE_GROUP_PICK_DIVIDES_THE_ONE_LANE_RUNG_BY_THE_LANE_COUNT_SO_NH_TIMES_SPLITS_TIMES_LANES_STAYS_AT_THE_ONE_LANE_TARGET:
    &[(i32, i32)] = &[(1, 16), (2, 8), (3, 4), (4, 4), (6, 2), (8, 2), (12, 1), (16, 1)];

const SM_EXACT_RUNGS_MEASURED_SLOWER_ON_EVERY_FAMILY_SO_NV_FLASH_SM_RUNGS_STAYS_DEFAULT_OFF: &[(&str, &str)] = &[
    ("gemma4-31B rung 12 vs stock 16", "+2.7% @32768, +5.7% @131072, +8.0% @196608"),
    ("qwen3.5-9B rung 24 vs stock 32", "neutral @256 and @8192, +1.8% @32768"),
    ("qwen3.8-27B", "geometric no-op, rung 16 both ways"),
];

const THE_ONLY_MEASURED_WIN_IS_A_DEEP_BAND_AND_IT_IS_NOT_AN_SM_MULTIPLE: (&str, &str) = (
    "gemma4-31B NV_FLASH_SPLITS_RUNG=32 = 1024 workgroups = 5.45 per SM",
    "-1.0 to -2.8% at 131072 and 196608, +0.8 to +1.2% at 256 through 65536",
);

const FAMILIES_THAT_ROUTE_THROUGH_THE_FUSED_FP8_DECODE_PICKER: &[(&str, i32, i32, i32)] = &[
    ("gemma4-31B sliding-1024 layers", 32, 16, 1024),
    ("gemma4-31B full-attention layers", 32, 16, 0),
    ("qwen3.5-9B full-attention layers", 16, 4, 0),
    ("qwen3.6-35B full-attention layers", 16, 2, 0),
    ("qwen3.8-27B full-attention layers", 24, 4, 0),
];

const HD: usize = 256;

const BF16_SPLIT_REASSOCIATION_BAND_A_STRUCTURAL_FAN_IN_OR_STRIDE_BUG_MISSES_IT_BY_ORDERS_OF_MAGNITUDE:
    f32 = 2.0e-2;

fn armed() -> bool {
    std::env::var("NV_FLASH_SM_RUNGS").as_deref() == Ok("1")
}

fn stock_three_rung_ladder_of_powers_of_two(nh: i32, sm: i32) -> i32 {
    let target = WAVES_PER_SM_THE_PICKER_TARGETS_BECAUSE_ROUNDS_14_AND_15_MEASURED_THE_WINNER_AT_ONE_OR_TWO_TIMES_THE_SM_COUNT
        * sm as i64;
    if nh as i64 * 8 >= target {
        8
    } else if nh as i64 * 16 >= target {
        16
    } else {
        32
    }
}

fn sm_exact_rung(nh: i32, sm: i32) -> i32 {
    let target = WAVES_PER_SM_THE_PICKER_TARGETS_BECAUSE_ROUNDS_14_AND_15_MEASURED_THE_WINNER_AT_ONE_OR_TWO_TIMES_THE_SM_COUNT
        * sm as i64;
    let mut best = INSTANTIATED_RUNGS_ANY_OTHER_VALUE_FALLS_INTO_THE_SWITCH_DEFAULT_AND_LAUNCHES_A_GRID_ITS_TEMPLATE_DOES_NOT_MATCH[0];
    let mut best_d = i64::MAX;
    for &r in INSTANTIATED_RUNGS_ANY_OTHER_VALUE_FALLS_INTO_THE_SWITCH_DEFAULT_AND_LAUNCHES_A_GRID_ITS_TEMPLATE_DOES_NOT_MATCH {
        let d = (nh as i64 * r as i64 - target).abs();
        if d < best_d {
            best_d = d;
            best = r;
        }
    }
    best
}

#[test]
fn the_box_this_suite_pins_its_workgroup_arithmetic_to_is_the_box_it_is_running_on() {
    let sm = cuda::flash_sm_count();
    assert_eq!(
        sm, SM_COUNT_ON_THE_ONE_BOX_EVERY_RUNG_IN_THIS_SUITE_WAS_MEASURED_ON,
        "every rung constant in this suite and in the lagsplit ladder is a multiple of 188 SMs; \
         on a {sm}-SM card the rungs are arithmetic for a different machine and must be \
         re-measured before any of them is quoted"
    );
}

#[test]
fn the_default_pick_is_the_stock_power_of_two_ladder_and_the_armed_pick_is_the_sm_exact_rung() {
    let sm = cuda::flash_sm_count();
    for &(name, nh, _nkv, window) in FAMILIES_THAT_ROUTE_THROUGH_THE_FUSED_FP8_DECODE_PICKER {
        let got = cuda::flash_decode_fp8_splits_pick(nh, window);
        let want = if armed() {
            sm_exact_rung(nh, sm)
        } else {
            stock_three_rung_ladder_of_powers_of_two(nh, sm)
        };
        assert_eq!(
            got, want,
            "{name}: NV_FLASH_SM_RUNGS={} picked {got} splits for NH={nh}, expected {want}",
            if armed() { "1" } else { "unset" }
        );
        assert!(
            INSTANTIATED_RUNGS_ANY_OTHER_VALUE_FALLS_INTO_THE_SWITCH_DEFAULT_AND_LAUNCHES_A_GRID_ITS_TEMPLATE_DOES_NOT_MATCH
                .contains(&got),
            "{name}: picked rung {got} is not instantiated by fused_fp8_splits_switch_launch, so \
             the launch would run the default <16> template on a grid.y of {got}"
        );
    }
}

#[test]
fn arming_the_picker_does_not_move_the_flagship_qwen38_geometry() {
    let sm = cuda::flash_sm_count();
    let (nh, window) = (24, 0);
    assert_eq!(
        sm_exact_rung(nh, sm),
        stock_three_rung_ladder_of_powers_of_two(nh, sm),
        "qwen3.8 runs NH=24 and 24x16=384 is already within one rung of the {} target, so the \
         arm must be a geometric no-op for it; if this fires the flagship needs a full re-measure",
        WAVES_PER_SM_THE_PICKER_TARGETS_BECAUSE_ROUNDS_14_AND_15_MEASURED_THE_WINNER_AT_ONE_OR_TWO_TIMES_THE_SM_COUNT
            * sm as i64
    );
    assert_eq!(cuda::flash_decode_fp8_splits_pick(nh, window), 16);
}

#[test]
fn the_sm_exact_arm_is_default_off_because_every_family_measured_slower_with_it() {
    assert!(
        !armed() || std::env::var("NV_FLASH_SM_RUNGS").is_ok(),
        "unreachable"
    );
    assert_eq!(
        SM_EXACT_RUNGS_MEASURED_SLOWER_ON_EVERY_FAMILY_SO_NV_FLASH_SM_RUNGS_STAYS_DEFAULT_OFF.len(),
        3,
        "the ladder that decided this arm covered three families; adding a fourth means \
         re-measuring, not extending the list"
    );
    assert_eq!(
        cuda::flash_decode_fp8_splits_pick(32, 0),
        if armed() { 12 } else { 16 },
        "hitting 2x the SM count exactly ({}) is what rounds 14 and 15 found for STARVED grids at \
         0.34-0.68 workgroups per SM; this kernel already runs at 2.72 and prefers MORE queued \
         waves at depth, so the wave-quantisation argument does not transfer -- measured \
         {SM_EXACT_RUNGS_MEASURED_SLOWER_ON_EVERY_FAMILY_SO_NV_FLASH_SM_RUNGS_STAYS_DEFAULT_OFF:?}, \
         and the only win found was {THE_ONLY_MEASURED_WIN_IS_A_DEEP_BAND_AND_IT_IS_NOT_AN_SM_MULTIPLE:?}",
        2 * cuda::flash_sm_count()
    );
}

#[test]
fn the_stock_ladder_overshoots_the_wave_target_where_the_sm_exact_rung_hits_it() {
    let sm = cuda::flash_sm_count();
    let target = WAVES_PER_SM_THE_PICKER_TARGETS_BECAUSE_ROUNDS_14_AND_15_MEASURED_THE_WINNER_AT_ONE_OR_TWO_TIMES_THE_SM_COUNT
        * sm as i64;
    for &(name, nh, _nkv, _w) in FAMILIES_THAT_ROUTE_THROUGH_THE_FUSED_FP8_DECODE_PICKER {
        let stock_wg = nh as i64 * stock_three_rung_ladder_of_powers_of_two(nh, sm) as i64;
        let exact_wg = nh as i64 * sm_exact_rung(nh, sm) as i64;
        assert!(
            (exact_wg - target).abs() <= (stock_wg - target).abs(),
            "{name}: sm-exact rung lands {exact_wg} workgroups and the stock ladder lands \
             {stock_wg} against a {target} target"
        );
    }
    assert_eq!(
        32 * stock_three_rung_ladder_of_powers_of_two(32, sm),
        512,
        "gemma4-31B NH=32 on the stock ladder is two full waves plus a third wave of \
         {} workgroups against a {target} target",
        512 - target
    );
    assert_eq!(32 * sm_exact_rung(32, sm), 384);
    assert_eq!(16 * stock_three_rung_ladder_of_powers_of_two(16, sm), 512);
    assert_eq!(16 * sm_exact_rung(16, sm), 384);
}

#[test]
fn widening_the_ladder_with_rungs_below_six_moves_no_familys_sm_exact_rung() {
    let sm = cuda::flash_sm_count();
    for &(nh, want) in THE_SM_EXACT_RUNG_EVERY_FAMILY_HAD_BEFORE_THE_LADDER_GAINED_ITS_LANE_GROUP_RUNGS_BELOW_SIX
    {
        assert_eq!(
            sm_exact_rung(nh, sm),
            want,
            "NH={nh}: the lane-group rungs 1..4 are strictly smaller than every rung that was \
             already in the ladder, so no family whose two-per-SM target is above NH*4 may move; \
             a move here means NV_FLASH_SM_RUNGS=1 now picks a different grid than the ladder \
             those rungs were measured on"
        );
    }
}

#[test]
fn the_lane_group_pick_leaves_one_lane_alone_and_divides_the_rung_by_a_wider_group() {
    for &(name, nh, _nkv, window) in FAMILIES_THAT_ROUTE_THROUGH_THE_FUSED_FP8_DECODE_PICKER {
        assert_eq!(
            cuda::flash_decode_fp8_splits_pick_for_a_lane_group(nh, window, 1),
            cuda::flash_decode_fp8_splits_pick(nh, window),
            "{name}: at one lane in flight the lane-group pick must BE the shipped pick, or \
             every solo decode and every parked family silently changes reduction order"
        );
    }
    let (nh, window) = (24, 0);
    for &(lanes, want) in THE_LANE_GROUP_PICK_DIVIDES_THE_ONE_LANE_RUNG_BY_THE_LANE_COUNT_SO_NH_TIMES_SPLITS_TIMES_LANES_STAYS_AT_THE_ONE_LANE_TARGET
    {
        let got = cuda::flash_decode_fp8_splits_pick_for_a_lane_group(nh, window, lanes);
        assert_eq!(got, want, "qwen3.8 NH=24 at {lanes} lanes picked rung {got}");
        assert!(
            INSTANTIATED_RUNGS_ANY_OTHER_VALUE_FALLS_INTO_THE_SWITCH_DEFAULT_AND_LAUNCHES_A_GRID_ITS_TEMPLATE_DOES_NOT_MATCH
                .contains(&got),
            "rung {got} is not instantiated, so the launch would run the default <16> template \
             on a grid.y of {got}"
        );
        assert!(
            got <= cuda::flash_decode_fp8_splits_pick(nh, window),
            "the lane-group pick may only ever LOWER the rung: a higher one would overrun the \
             split-k scratch the cache provisioned for the one-lane widest rung"
        );
    }
}

#[test]
fn the_decode_scratch_provision_covers_every_rung_the_picker_can_pick() {
    for &(name, nh, _nkv, window) in FAMILIES_THAT_ROUTE_THROUGH_THE_FUSED_FP8_DECODE_PICKER {
        let sp = cuda::flash_decode_fp8_splits_pick(nh, window);
        let elems = cuda::flash_splitk_scratch_elems(nh, HD as i32);
        assert!(
            elems >= (nh as usize) * (sp as usize) * (HD + 2),
            "{name}: scratch provisions {elems} f32 but rung {sp} at NH={nh} HD={HD} strides \
             {} -- a short provision is an out-of-bounds write in the split store",
            (nh as usize) * (sp as usize) * (HD + 2)
        );
    }
}

fn xorshift(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    ((*state >> 9) as f32 / (1u32 << 23) as f32) - 1.0
}

fn decode_at_rung(nh: usize, nkv: usize, total: usize, window: usize, rung: i32) -> Vec<f32> {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let raw = stream.cu_stream() as *mut c_void;
    let mut state = 0x2545_f491u32;
    let q_host: Vec<u16> = (0..nh * HD)
        .map(|_| bf16::from_f32(xorshift(&mut state)).to_bits())
        .collect();
    let mut fill = 0x9e37_79b9u32;
    let mut kv_host: Vec<u8> = vec![0u8; total * nkv * HD];
    for chunk in kv_host.chunks_mut(4) {
        fill ^= fill << 13;
        fill ^= fill >> 17;
        fill ^= fill << 5;
        let b = (fill & 0x7f7f_7f7f).to_le_bytes();
        chunk.copy_from_slice(&b[..chunk.len()]);
    }
    let sc_host: Vec<f32> = (0..total * nkv)
        .map(|_| 0.01 + xorshift(&mut state).abs() * 0.01)
        .collect();
    #[allow(deprecated)]
    let d_q: CudaSlice<u16> = stream.clone_htod(&q_host).unwrap();
    #[allow(deprecated)]
    let d_k: CudaSlice<u8> = stream.clone_htod(&kv_host).unwrap();
    #[allow(deprecated)]
    let d_v: CudaSlice<u8> = stream.clone_htod(&kv_host).unwrap();
    #[allow(deprecated)]
    let d_ks: CudaSlice<f32> = stream.clone_htod(&sc_host).unwrap();
    #[allow(deprecated)]
    let d_vs: CudaSlice<f32> = stream.clone_htod(&sc_host).unwrap();
    #[allow(deprecated)]
    let d_nt: CudaSlice<i32> = stream.clone_htod(&vec![total as i32]).unwrap();
    let top = *INSTANTIATED_RUNGS_ANY_OTHER_VALUE_FALLS_INTO_THE_SWITCH_DEFAULT_AND_LAUNCHES_A_GRID_ITS_TEMPLATE_DOES_NOT_MATCH
        .last()
        .unwrap() as usize;
    let mut d_scratch: CudaSlice<f32> = stream.alloc_zeros::<f32>(nh * top * (HD + 2)).unwrap();
    let mut d_fan: CudaSlice<u32> = stream.alloc_zeros::<u32>(nh).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(nh * HD).unwrap();
    let scaling = 1.0f32 / (HD as f32).sqrt();
    std::env::set_var("NV_FLASH_SPLITS_RUNG", rung.to_string());
    let rc = {
        let (p_q, _a) = d_q.device_ptr(&stream);
        let (p_k, _b) = d_k.device_ptr(&stream);
        let (p_v, _c) = d_v.device_ptr(&stream);
        let (p_ks, _d) = d_ks.device_ptr(&stream);
        let (p_vs, _e) = d_vs.device_ptr(&stream);
        let (p_nt, _f) = d_nt.device_ptr(&stream);
        let (p_out, _g) = d_out.device_ptr_mut(&stream);
        let (p_sc, _h) = d_scratch.device_ptr_mut(&stream);
        let (p_fan, _i) = d_fan.device_ptr_mut(&stream);
        unsafe {
            cuda::flash_decode_fused_fp8kv_paged(
                raw,
                p_q as *const u16,
                p_k as *const u8,
                p_v as *const u8,
                p_ks as *const f32,
                p_vs as *const f32,
                p_out as *mut u16,
                p_nt as *const i32,
                p_sc as *mut f32,
                p_fan as *mut u32,
                nh as i32,
                nkv as i32,
                HD as i32,
                window as i32,
                0,
                scaling,
                std::ptr::null(),
                0,
            )
        }
    };
    std::env::remove_var("NV_FLASH_SPLITS_RUNG");
    assert_eq!(rc, 0, "fused fp8 decode at rung {rung} returned {rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out: Vec<u16> = stream.clone_dtoh(&d_out).unwrap();
    out.into_iter().map(|b| bf16::from_bits(b).to_f32()).collect()
}

#[test]
fn every_instantiated_rung_agrees_with_the_stock_rung_on_real_fp8_kv() {
    for &(name, nh, nkv, window, total) in &[
        ("gemma4-31B sliding", 32usize, 16usize, 1024usize, 4096usize),
        ("gemma4-31B full", 32, 16, 0, 4096),
        ("qwen3.5-9B full", 16, 4, 0, 4096),
    ] {
        let reference = decode_at_rung(nh, nkv, total, window, 16);
        for &rung in INSTANTIATED_RUNGS_ANY_OTHER_VALUE_FALLS_INTO_THE_SWITCH_DEFAULT_AND_LAUNCHES_A_GRID_ITS_TEMPLATE_DOES_NOT_MATCH
        {
            let got = decode_at_rung(nh, nkv, total, window, rung);
            let mut worst = 0.0f32;
            let mut at = 0usize;
            for (i, (a, b)) in reference.iter().zip(got.iter()).enumerate() {
                let d = (a - b).abs() / a.abs().max(b.abs()).max(1e-6);
                if d > worst {
                    worst = d;
                    at = i;
                }
            }
            assert!(
                worst <= BF16_SPLIT_REASSOCIATION_BAND_A_STRUCTURAL_FAN_IN_OR_STRIDE_BUG_MISSES_IT_BY_ORDERS_OF_MAGNITUDE,
                "{name}: rung {rung} vs rung 16 worst rel {worst:.6} at element {at}; the band is \
                 split-count reassociation of a bf16 output, not a licence for a wrong fan-in \
                 ticket or a wrong scratch stride"
            );
        }
    }
}
