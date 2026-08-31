#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

const Q38_GDN_NK: usize = 16;
const Q38_GDN_NV: usize = 48;
const Q38_GDN_DK: usize = 128;
const Q38_GDN_DV: usize = 128;
const Q38_CONV_DIM: usize = 10240;
const Q38_VALUE_DIM: usize = 6144;

const THE_VEC4_ARM_KEEPS_SUBS_AND_KPS_SO_EVERY_V_COLUMN_SEES_THE_SAME_ADDITION_ORDER_AS_THE_SHIPPED_LANECOL_MAP: &str =
    "arm 0 gives one v column to each thread and arms 1..3 give four, but SUBS stays 8 and KPS \
     stays 16 in all of them, so the per-column j loop and the per-column shared-memory tree are \
     the same additions in the same order and the only legitimate result is BIT-IDENTICAL state; \
     a tolerance here would hide exactly the reassociation this arm was designed to avoid";

const A_SINGLE_STEP_CANNOT_SEPARATE_THE_ARMS_BECAUSE_THE_RECURRENCE_IS_WHAT_ACCUMULATES_A_ONE_ULP_DIVERGENCE: &str =
    "the GDN state is fed back into itself every step, so a one-ulp difference in step 1 is \
     amplified by 160 steps of gating and delta-rule update; the gate therefore drives the same \
     forced-decode depth the flagship bit-identity proofs use rather than comparing one call";

const HEAD_DIM_128_IS_THE_REAL_ONE_AND_A_SMALLER_CONFIG_WOULD_EXERCISE_ONE_LANE_STRIP: &str =
    "qwen3.8 runs GDN at d_k = d_v = 128 with 48 value heads; the vec4 arms differ from arm 0 \
     only in how a 128-wide v row is split across threads, so a narrower head would collapse the \
     very partition under test";

const STEPS_SO_THE_RECURRENCE_ACCUMULATES_ANY_REASSOCIATION: usize = 160;
const LANES_UNDER_TEST_IS_THE_SERVED_BATCH_CEILING: usize = 16;
const INPUT_BANKS_SO_THE_STATE_DOES_NOT_SETTLE_ON_A_FIXED_POINT: usize = 8;
const COLD_POOL_BYTES_TARGET: usize = 768 * 1024 * 1024;

const A_ONE_TO_ONE_STREAMING_KERNEL_AT_THE_STATE_SIZE_AND_COLDNESS_IS_THE_ONLY_CEILING_THIS_KERNEL_OWES_A_RATIO_TO: i32 =
    -1;

const MEMCPY_DTOD_IS_AN_IMPLEMENTATION_NOT_A_BANDWIDTH_LIMIT_SO_THE_CEILING_ARM_IS_A_KERNEL: &str =
    "quoting a kernel against cudaMemcpyDeviceToDevice can put it above 100% of its own ceiling; \
     the ceiling arm here is nv_kernels_silu_f32 over the same cold pool at the same 3.146 MB per \
     lane, which reads four bytes and writes four bytes per element exactly as the state step does, \
     measured by the same graphed instrument in the same interleaved rotation";

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u32() & 0x7f_ffff) as f32 / 8388608.0) * 2.0 - 1.0
    }
    fn bf16_words(&mut self, n: usize, gain: f32) -> Vec<u16> {
        (0..n)
            .map(|_| bf16::from_f32(self.next_f32() * gain).to_bits())
            .collect()
    }
    fn f32_vals(&mut self, n: usize, gain: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * gain).collect()
    }
}

fn bench_gate(name: &str) {
    if std::env::var("NV_KERNELS_BENCH").as_deref() != Ok("1") {
        panic!(
            "NV_KERNELS_BENCH is not 1, so {name} executed nothing. It is already #[ignore]d, \
             which makes --ignored the opt-in; a second env gate behind that can only make an \
             --ignored run print `1 passed` for a bench that measured nothing."
        );
    }
}

fn least_squares_slope_and_intercept(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|v| v * v).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(a, b)| a * b).sum();
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    (slope, (sy - slope * sx) / n)
}

struct GdnStepInputs {
    mixed: CudaSlice<u16>,
    z: CudaSlice<u16>,
    ab: CudaSlice<u16>,
    a_log: CudaSlice<u16>,
    dt: CudaSlice<u16>,
    nw: CudaSlice<u16>,
    out: CudaSlice<u16>,
    qn: CudaSlice<f32>,
    kn: CudaSlice<f32>,
    ge: CudaSlice<f32>,
    be: CudaSlice<f32>,
    core: CudaSlice<u16>,
}

impl GdnStepInputs {
    fn new(stream: &Arc<CudaStream>, rng: &mut Lcg, lanes: usize, banks: usize) -> Self {
        let span = lanes * banks;
        let ab_stride = 2 * Q38_GDN_NV;
        let key_dim = Q38_GDN_NK * Q38_GDN_DK;
        let mixed_h = rng.bf16_words(span * Q38_CONV_DIM, 0.5);
        let z_h = rng.bf16_words(span * Q38_VALUE_DIM, 0.5);
        let ab_h = rng.bf16_words(span * ab_stride, 0.5);
        let a_log_h = rng.bf16_words(Q38_GDN_NV, 0.5);
        let dt_h = rng.bf16_words(Q38_GDN_NV, 0.5);
        let nw_h = rng.bf16_words(Q38_GDN_DV, 0.5);
        #[allow(deprecated)]
        let this = Self {
            mixed: stream.clone_htod(&mixed_h).unwrap(),
            z: stream.clone_htod(&z_h).unwrap(),
            ab: stream.clone_htod(&ab_h).unwrap(),
            a_log: stream.clone_htod(&a_log_h).unwrap(),
            dt: stream.clone_htod(&dt_h).unwrap(),
            nw: stream.clone_htod(&nw_h).unwrap(),
            out: stream.alloc_zeros::<u16>(lanes * Q38_VALUE_DIM).unwrap(),
            qn: stream.alloc_zeros::<f32>(lanes * key_dim).unwrap(),
            kn: stream.alloc_zeros::<f32>(lanes * key_dim).unwrap(),
            ge: stream.alloc_zeros::<f32>(lanes * Q38_GDN_NV).unwrap(),
            be: stream.alloc_zeros::<f32>(lanes * Q38_GDN_NV).unwrap(),
            core: stream
                .alloc_zeros::<u16>(lanes * Q38_GDN_NV * Q38_GDN_DV)
                .unwrap(),
        };
        stream.synchronize().unwrap();
        this
    }

}

const TAKING_A_DEVICE_POINTER_INSIDE_A_CAPTURE_JOINS_STREAMS_AND_INVALIDATES_IT_WITH_CUDA_901: &str =
    "cudarc's device_ptr/device_ptr_mut may record a cross-stream dependency, which a stream in \
     capture mode reports as cudaErrorStreamCaptureInvalidated; every pointer a captured body \
     needs must therefore be resolved to a raw address before capture begins";

#[derive(Clone, Copy)]
struct GdnStepPtrs {
    mixed: u64,
    z: u64,
    ab: u64,
    a_log: u64,
    dt: u64,
    nw: u64,
    out: u64,
    qn: u64,
    kn: u64,
    ge: u64,
    be: u64,
    core: u64,
}

macro_rules! resolve_gdn_step_ptrs_before_any_capture_begins {
    ($inputs:expr, $stream:expr) => {{
        let (mixed, _a) = $inputs.mixed.device_ptr($stream);
        let (z, _b) = $inputs.z.device_ptr($stream);
        let (ab, _c) = $inputs.ab.device_ptr($stream);
        let (a_log, _d) = $inputs.a_log.device_ptr($stream);
        let (dt, _e) = $inputs.dt.device_ptr($stream);
        let (nw, _f) = $inputs.nw.device_ptr($stream);
        let (out, _g) = $inputs.out.device_ptr_mut($stream);
        let (qn, _h) = $inputs.qn.device_ptr_mut($stream);
        let (kn, _i) = $inputs.kn.device_ptr_mut($stream);
        let (ge, _j) = $inputs.ge.device_ptr_mut($stream);
        let (be, _k) = $inputs.be.device_ptr_mut($stream);
        let (core, _l) = $inputs.core.device_ptr_mut($stream);
        GdnStepPtrs { mixed, z, ab, a_log, dt, nw, out, qn, kn, ge, be, core }
    }};
}

fn launch_group(
    stream: &Arc<CudaStream>,
    p: &GdnStepPtrs,
    state_ptrs: &[*mut f32],
    lanes: usize,
    bank: usize,
) -> i32 {
    let ab_stride = 2 * Q38_GDN_NV;
    let m_off = (bank * lanes * Q38_CONV_DIM * 2) as u64;
    let z_off = (bank * lanes * Q38_VALUE_DIM * 2) as u64;
    let ab_off = (bank * lanes * ab_stride * 2) as u64;
    unsafe {
        cuda::gdn_decode_step_split_group_bf16(
            stream.cu_stream() as *mut c_void,
            (p.mixed + m_off) as *const u16,
            (p.z + z_off) as *const u16,
            (p.ab + ab_off) as *const u16,
            (p.ab + ab_off + (Q38_GDN_NV * 2) as u64) as *const u16,
            p.a_log as *const u16,
            p.dt as *const u16,
            p.nw as *const u16,
            state_ptrs.as_ptr(),
            p.out as *mut u16,
            p.qn as *mut f32,
            p.kn as *mut f32,
            p.ge as *mut f32,
            p.be as *mut f32,
            p.core as *mut u16,
            Q38_GDN_NK as i32,
            Q38_GDN_NV as i32,
            Q38_GDN_DK as i32,
            Q38_GDN_DV as i32,
            1e-6,
            lanes as i32,
            ab_stride as i32,
        )
    }
}

fn set_arm(arm: i32) {
    let rc = unsafe { cuda::gdn_decode_step_group_vec4_arm_set(arm) };
    assert_eq!(rc, 0, "gdn_decode_step_group_vec4_arm_set({arm}) rc={rc}");
}

#[test]
fn gdn_group_step_vec4_arms_are_bit_identical_to_the_shipped_lanecol_map_over_a_forced_decode_run() {
    eprintln!("GDNCOAL-WHY {THE_VEC4_ARM_KEEPS_SUBS_AND_KPS_SO_EVERY_V_COLUMN_SEES_THE_SAME_ADDITION_ORDER_AS_THE_SHIPPED_LANECOL_MAP}");
    eprintln!("GDNCOAL-WHY {A_SINGLE_STEP_CANNOT_SEPARATE_THE_ARMS_BECAUSE_THE_RECURRENCE_IS_WHAT_ACCUMULATES_A_ONE_ULP_DIVERGENCE}");
    eprintln!("GDNCOAL-WHY {HEAD_DIM_128_IS_THE_REAL_ONE_AND_A_SMALLER_CONFIG_WOULD_EXERCISE_ONE_LANE_STRIP}");

    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.default_stream();
    let mut rng = Lcg(0x9e3779b97f4a7c15);
    let lanes = LANES_UNDER_TEST_IS_THE_SERVED_BATCH_CEILING;
    let state_elems = Q38_GDN_NV * Q38_GDN_DK * Q38_GDN_DV;

    let mut inputs = GdnStepInputs::new(
        &stream,
        &mut rng,
        lanes,
        INPUT_BANKS_SO_THE_STATE_DOES_NOT_SETTLE_ON_A_FIXED_POINT,
    );
    let seed: Vec<Vec<f32>> = (0..lanes)
        .map(|_| rng.f32_vals(state_elems, 0.02))
        .collect();

    let mut reference: Vec<Vec<f32>> = Vec::new();
    let mut reference_out: Vec<u16> = Vec::new();
    for arm in [0i32, 1, 2, 3] {
        set_arm(arm);
        let mut states: Vec<CudaSlice<f32>> = seed
            .iter()
            .map(|s| {
                #[allow(deprecated)]
                let d: CudaSlice<f32> = stream.clone_htod(s).unwrap();
                d
            })
            .collect();
        stream.synchronize().unwrap();
        for step in 0..STEPS_SO_THE_RECURRENCE_ACCUMULATES_ANY_REASSOCIATION {
            let ptrs: Vec<*mut f32> = states
                .iter()
                .map(|b| {
                    let (p, _g) = b.device_ptr(&stream);
                    p as *mut f32
                })
                .collect();
            let bank = step % INPUT_BANKS_SO_THE_STATE_DOES_NOT_SETTLE_ON_A_FIXED_POINT;
            let sp = resolve_gdn_step_ptrs_before_any_capture_begins!(inputs, &stream);
            let rc = launch_group(&stream, &sp, &ptrs, lanes, bank);
            assert_eq!(rc, 0, "arm={arm} step={step} rc={rc}");
        }
        stream.synchronize().unwrap();
        let got: Vec<Vec<f32>> = states
            .iter_mut()
            .map(|b| stream.memcpy_dtov(b).unwrap())
            .collect();
        let got_out: Vec<u16> = stream.memcpy_dtov(&inputs.out).unwrap();

        let moved = got[0]
            .iter()
            .zip(&seed[0])
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(
            moved * 2 > state_elems,
            "arm={arm} left {moved}/{state_elems} state floats at their seed value; a comparison \
             against an unevolved state proves nothing about the thread map"
        );
        let finite = got[0].iter().filter(|v| v.is_finite()).count();
        assert_eq!(
            finite, state_elems,
            "arm={arm} produced non-finite state after \
             {STEPS_SO_THE_RECURRENCE_ACCUMULATES_ANY_REASSOCIATION} steps"
        );

        if arm == 0 {
            reference = got;
            reference_out = got_out;
            continue;
        }
        let mut mismatches = 0usize;
        let mut first: Option<(usize, usize, u32, u32)> = None;
        for (l, (a, b)) in got.iter().zip(&reference).enumerate() {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                if x.to_bits() != y.to_bits() {
                    mismatches += 1;
                    if first.is_none() {
                        first = Some((l, i, x.to_bits(), y.to_bits()));
                    }
                }
            }
        }
        assert_eq!(
            mismatches,
            0,
            "arm={arm} diverged from the shipped lanecol map in {mismatches} of {} state floats \
             (first {first:?}); {THE_VEC4_ARM_KEEPS_SUBS_AND_KPS_SO_EVERY_V_COLUMN_SEES_THE_SAME_ADDITION_ORDER_AS_THE_SHIPPED_LANECOL_MAP}",
            lanes * state_elems
        );
        assert!(
            got_out == reference_out,
            "arm={arm} matched the state but not the gated output words"
        );
        eprintln!(
            "GDNCOAL-BITID arm={arm} lanes={lanes} steps={STEPS_SO_THE_RECURRENCE_ACCUMULATES_ANY_REASSOCIATION} \
             state_floats_compared={} out_words_compared={} mismatches=0",
            lanes * state_elems,
            reference_out.len()
        );
    }
    set_arm(0);
}

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_KERNELS_BENCH=1 -- prices the float4-along-v arms against the shipped lanecol map inside a CAPTURED graph over an L2-cold state pool, interleaved and rotated, because the serving decode step is captured and the state is DRAM-resident at every batch size"]
fn gdn_group_step_vec4_arms_priced_against_the_shipped_map_inside_a_captured_graph() {
    bench_gate("the GDN vec4-arm graphed price");
    eprintln!("GDNCOAL-WHY {THE_VEC4_ARM_KEEPS_SUBS_AND_KPS_SO_EVERY_V_COLUMN_SEES_THE_SAME_ADDITION_ORDER_AS_THE_SHIPPED_LANECOL_MAP}");

    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.new_stream().expect("capture stream");
    let mut rng = Lcg(0xbb67ae8584caa73b);
    let max_lanes = LANES_UNDER_TEST_IS_THE_SERVED_BATCH_CEILING;
    let state_elems = Q38_GDN_NV * Q38_GDN_DK * Q38_GDN_DV;
    let one_lane_state_bytes = state_elems * 4;
    let calls_per_graph = 64usize;

    let pool_len = (COLD_POOL_BYTES_TARGET / one_lane_state_bytes).next_multiple_of(max_lanes);
    let seed_state = rng.f32_vals(state_elems, 0.02);
    #[allow(deprecated)]
    let seed: CudaSlice<f32> = stream.clone_htod(&seed_state).unwrap();
    let mut pool: Vec<CudaSlice<f32>> = Vec::with_capacity(pool_len);
    for _ in 0..pool_len {
        let mut d: CudaSlice<f32> = stream.alloc_zeros::<f32>(state_elems).unwrap();
        stream.memcpy_dtod(&seed, &mut d).unwrap();
        pool.push(d);
    }
    stream.synchronize().unwrap();
    eprintln!("GDNCOAL-WHY {TAKING_A_DEVICE_POINTER_INSIDE_A_CAPTURE_JOINS_STREAMS_AND_INVALIDATES_IT_WITH_CUDA_901}");
    eprintln!("GDNCOAL-WHY {MEMCPY_DTOD_IS_AN_IMPLEMENTATION_NOT_A_BANDWIDTH_LIMIT_SO_THE_CEILING_ARM_IS_A_KERNEL}");
    let mut inputs = GdnStepInputs::new(&stream, &mut rng, max_lanes, 1);
    let step_ptrs = resolve_gdn_step_ptrs_before_any_capture_begins!(inputs, &stream);
    let pool_ptrs: Vec<u64> = pool
        .iter()
        .map(|b| {
            let (p, _g) = b.device_ptr(&stream);
            p
        })
        .collect();

    let lane_ladder = [1usize, 2, 4, 8, 16];
    let arms = [
        A_ONE_TO_ONE_STREAMING_KERNEL_AT_THE_STATE_SIZE_AND_COLDNESS_IS_THE_ONLY_CEILING_THIS_KERNEL_OWES_A_RATIO_TO,
        0i32,
        1,
        2,
        3,
    ];
    let mut cells: Vec<(i32, usize)> = Vec::new();
    for a in arms {
        for l in lane_ladder {
            cells.push((a, l));
        }
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); cells.len()];
    let mut runner = nv_kernels::graph::CudaGraphRunner::new(stream.clone());
    let reps_including_one_discarded_warmup = 5usize;
    for rep in 0..reps_including_one_discarded_warmup {
        for shift in 0..cells.len() {
            let idx = (shift + rep) % cells.len();
            let (arm, lanes) = cells[idx];
            if arm >= 0 {
                set_arm(arm);
            }
            let token = (idx as u64) + 1;
            let groups = pool_len / lanes;
            let pool_ptrs_ref = &pool_ptrs;
            runner
                .run(token, |s| {
                    for call in 0..calls_per_graph {
                        let base = (call % groups) * lanes;
                        if arm < 0 {
                            for l in 0..lanes {
                                let src = pool_ptrs_ref[base + l];
                                let dst = pool_ptrs_ref[(base + l + pool_len / 2) % pool_len];
                                let rc = unsafe {
                                    cuda::silu_f32(
                                        s.cu_stream() as *mut c_void,
                                        src as *const f32,
                                        dst as *mut f32,
                                        state_elems,
                                    )
                                };
                                assert_eq!(rc, 0, "graphed ceiling lanes={lanes} rc={rc}");
                            }
                            continue;
                        }
                        let ptrs: Vec<*mut f32> = (0..lanes)
                            .map(|l| pool_ptrs_ref[base + l] as *mut f32)
                            .collect();
                        let rc = launch_group(s, &step_ptrs, &ptrs, lanes, 0);
                        assert_eq!(rc, 0, "graphed arm={arm} lanes={lanes} rc={rc}");
                    }
                    Ok(())
                })
                .unwrap();
            stream.synchronize().unwrap();
            let replays = 30usize;
            let t0 = Instant::now();
            for _ in 0..replays {
                runner
                    .run(token, |_| panic!("recapture on a cached token"))
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / replays as f64 / calls_per_graph as f64;
            if rep > 0 {
                samples[idx].push(us);
            }
        }
    }
    set_arm(0);

    let mut per_arm: Vec<(i32, Vec<f64>, Vec<f64>)> =
        arms.iter().map(|a| (*a, Vec::new(), Vec::new())).collect();
    for (idx, (arm, lanes)) in cells.iter().enumerate() {
        let mut v = samples[idx].clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = v[v.len() / 2];
        let bytes = 2 * lanes * one_lane_state_bytes;
        let gbs = bytes as f64 / med / 1e3;
        eprintln!(
            "GDNCOAL-GRAPH arm={arm} lanes={lanes} threads={} us_per_call={med:.2} \
             spread_us={:.2} state_mb={:.2} state_rw_gbs={gbs:.0}",
            match arm {
                1 => 64,
                2 => 128,
                _ => 256,
            },
            v[v.len() - 1] - v[0],
            bytes as f64 / 1048576.0
        );
        let slot = per_arm.iter_mut().find(|(a, _, _)| a == arm).unwrap();
        slot.1.push(*lanes as f64);
        slot.2.push(med);
    }
    let mut ceiling_gbs = 0.0f64;
    for (arm, xs, ys) in &per_arm {
        let (slope, intercept) = least_squares_slope_and_intercept(xs, ys);
        let marginal = (2 * one_lane_state_bytes) as f64 / slope / 1e3;
        if *arm == A_ONE_TO_ONE_STREAMING_KERNEL_AT_THE_STATE_SIZE_AND_COLDNESS_IS_THE_ONLY_CEILING_THIS_KERNEL_OWES_A_RATIO_TO
        {
            ceiling_gbs = marginal;
        }
        eprintln!(
            "GDNCOAL-FIT arm={arm} us = {intercept:.2} + {slope:.2}*lanes ; marginal_rw_gbs={marginal:.0} ; pct_of_measured_ceiling={:.0}",
            100.0 * marginal / ceiling_gbs
        );
    }
    assert!(
        ceiling_gbs > 0.0,
        "the ceiling arm produced no slope, so every percentage above is undefined; \
         {MEMCPY_DTOD_IS_AN_IMPLEMENTATION_NOT_A_BANDWIDTH_LIMIT_SO_THE_CEILING_ARM_IS_A_KERNEL}"
    );
}
