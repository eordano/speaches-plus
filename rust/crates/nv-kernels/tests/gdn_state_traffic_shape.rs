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

const A_RATE_IS_NOT_COMPARABLE_ACROSS_TRANSFER_SIZES_SO_THE_CEILING_IS_A_CURVE_IN_BYTES_NOT_ONE_NUMBER: &str =
    "the 1475 GB/s calibration cell moves 256 MB each way; one GDN lane's recurrent state is \
     3.146 MB each way, which is a different size class, and a size class must be measured before \
     a ratio between the two is published";

const THE_SOLO_LAUNCHER_TIMES_THREE_DEPENDENT_KERNELS_SO_ATTRIBUTING_ALL_OF_ITS_MICROSECONDS_TO_STATE_BYTES_CHARGES_PREP_AND_GATE_LAUNCH_LATENCY_TO_BANDWIDTH: &str =
    "nv_kernels_gdn_decode_step_split_bf16 issues prep (n_k blocks), the lanecol step \
     (n_v*4 blocks) and gate (n_v blocks); only the middle one touches the recurrent state, so \
     state_rw_gbs computed over the launcher's total wall time is a lower bound on the step \
     kernel's own rate, not a measurement of it";

const LANE_SCALING_SEPARATES_A_FIXED_PER_CALL_COST_FROM_THE_MARGINAL_RATE_BECAUSE_ONLY_THE_STATE_BYTES_SCALE_WITH_LANES: &str =
    "the group launcher issues the same three kernels at every lane count and multiplies only the \
     work, so a two-term fit us = fixed + slope*lanes reads the marginal state rate off the slope \
     and the launch-and-tail cost off the intercept";

const L2_WARM_AND_L2_COLD_ARMS_EXIST_BECAUSE_A_FOUR_WAY_ROTATION_OVER_A_THREE_MEGABYTE_STATE_NEVER_LEAVES_L2_ON_THIS_CARD: &str =
    "a rate measured over a working set smaller than L2 is an L2 rate; the cold arm rolls over a \
     pool larger than any plausible L2 so the same kernel is priced against DRAM";

const COLD_POOL_BYTES_TARGET: usize = 768 * 1024 * 1024;

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

fn bench_us(stream: &Arc<CudaStream>, iters: usize, mut launch: impl FnMut(usize)) -> f64 {
    for i in 0..5 {
        launch(i);
    }
    stream.synchronize().unwrap();
    let t0 = Instant::now();
    for i in 0..iters {
        launch(i);
    }
    stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn bench_gate(name: &str) {
    if std::env::var("NV_KERNELS_BENCH").as_deref() != Ok("1") {
        panic!(
            "NV_KERNELS_BENCH is not 1, so {name} executed nothing. It is already #[ignore]d, \
             which makes --ignored the opt-in; a second env gate behind that can only make an \
             --ignored run print `1 passed` for a bench that measured nothing. A suite whose \
             precondition is absent must fail or be ignored, never return green."
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

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_KERNELS_BENCH=1 -- sweeps a device-to-device copy across three decades of transfer size so a rate quoted for a 3 MB recurrent state is compared against the ceiling at ITS size class rather than against the 256 MB calibration cell"]
fn dram_rate_by_transfer_size_class_because_the_ceiling_is_a_curve_in_bytes_not_one_number() {
    bench_gate("the transfer-size-class sweep");
    eprintln!("GDNSTATE-WHY {A_RATE_IS_NOT_COMPARABLE_ACROSS_TRANSFER_SIZES_SO_THE_CEILING_IS_A_CURVE_IN_BYTES_NOT_ONE_NUMBER}");
    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.default_stream();

    let one_lane_state_bytes = Q38_GDN_NV * Q38_GDN_DK * Q38_GDN_DV * 4;
    let sizes: Vec<usize> = vec![
        256 * 1024,
        1024 * 1024,
        one_lane_state_bytes,
        2 * one_lane_state_bytes,
        4 * one_lane_state_bytes,
        8 * one_lane_state_bytes,
        16 * one_lane_state_bytes,
        64 * 1024 * 1024,
        128 * 1024 * 1024,
        256 * 1024 * 1024,
        512 * 1024 * 1024,
    ];
    for n in sizes {
        let src: CudaSlice<u8> = stream.alloc_zeros::<u8>(n).unwrap();
        let mut dst: CudaSlice<u8> = stream.alloc_zeros::<u8>(n).unwrap();
        let iters = if n < 16 * 1024 * 1024 { 2000 } else { 60 };
        let us = bench_us(&stream, iters, |_| {
            stream.memcpy_dtod(&src, &mut dst).unwrap();
        });
        let gbs = (2 * n) as f64 / us / 1e3;
        eprintln!(
            "GDNSTATE-SIZE dtod_copy bytes_each_way={n} mb={:.3} us={us:.2} rw_gbs={gbs:.0}",
            n as f64 / 1048576.0
        );
    }
}

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_KERNELS_BENCH=1 -- prices gdn_decode_step_split's recurrent-state read/write against lane parallelism and against L2 residency, so the published 613 GB/s solo rate is split into a fixed per-call cost and a marginal state rate"]
fn gdn_decode_step_split_state_rw_rate_by_lane_parallelism_and_by_l2_residency() {
    bench_gate("the GDN state lane-parallelism sweep");
    eprintln!("GDNSTATE-WHY {THE_SOLO_LAUNCHER_TIMES_THREE_DEPENDENT_KERNELS_SO_ATTRIBUTING_ALL_OF_ITS_MICROSECONDS_TO_STATE_BYTES_CHARGES_PREP_AND_GATE_LAUNCH_LATENCY_TO_BANDWIDTH}");
    eprintln!("GDNSTATE-WHY {LANE_SCALING_SEPARATES_A_FIXED_PER_CALL_COST_FROM_THE_MARGINAL_RATE_BECAUSE_ONLY_THE_STATE_BYTES_SCALE_WITH_LANES}");
    eprintln!("GDNSTATE-WHY {L2_WARM_AND_L2_COLD_ARMS_EXIST_BECAUSE_A_FOUR_WAY_ROTATION_OVER_A_THREE_MEGABYTE_STATE_NEVER_LEAVES_L2_ON_THIS_CARD}");

    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.default_stream();
    let mut rng = Lcg(0x243f6a8885a308d3);

    let n_k = Q38_GDN_NK;
    let n_v = Q38_GDN_NV;
    let d_k = Q38_GDN_DK;
    let d_v = Q38_GDN_DV;
    let key_dim = n_k * d_k;
    let ab_stride = 2 * n_v;
    let max_lanes = 16usize;
    let state_elems = n_v * d_k * d_v;
    let one_lane_state_bytes = state_elems * 4;

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
    eprintln!(
        "GDNSTATE-POOL lane_state_mb={:.3} pool_len={pool_len} pool_mb={:.0}",
        one_lane_state_bytes as f64 / 1048576.0,
        (pool_len * one_lane_state_bytes) as f64 / 1048576.0
    );

    let mixed_h = rng.bf16_words(max_lanes * Q38_CONV_DIM, 0.5);
    let z_h = rng.bf16_words(max_lanes * Q38_VALUE_DIM, 0.5);
    let ab_h = rng.bf16_words(max_lanes * ab_stride, 0.5);
    let a_log_h = rng.bf16_words(n_v, 0.5);
    let dt_h = rng.bf16_words(n_v, 0.5);
    let nw_h = rng.bf16_words(d_v, 0.5);
    #[allow(deprecated)]
    let d_mixed: CudaSlice<u16> = stream.clone_htod(&mixed_h).unwrap();
    #[allow(deprecated)]
    let d_z: CudaSlice<u16> = stream.clone_htod(&z_h).unwrap();
    #[allow(deprecated)]
    let d_ab: CudaSlice<u16> = stream.clone_htod(&ab_h).unwrap();
    #[allow(deprecated)]
    let d_alog: CudaSlice<u16> = stream.clone_htod(&a_log_h).unwrap();
    #[allow(deprecated)]
    let d_dt: CudaSlice<u16> = stream.clone_htod(&dt_h).unwrap();
    #[allow(deprecated)]
    let d_nw: CudaSlice<u16> = stream.clone_htod(&nw_h).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(max_lanes * Q38_VALUE_DIM).unwrap();
    let mut d_qn: CudaSlice<f32> = stream.alloc_zeros::<f32>(max_lanes * key_dim).unwrap();
    let mut d_kn: CudaSlice<f32> = stream.alloc_zeros::<f32>(max_lanes * key_dim).unwrap();
    let mut d_ge: CudaSlice<f32> = stream.alloc_zeros::<f32>(max_lanes * n_v).unwrap();
    let mut d_be: CudaSlice<f32> = stream.alloc_zeros::<f32>(max_lanes * n_v).unwrap();
    let mut d_core: CudaSlice<u16> = stream.alloc_zeros::<u16>(max_lanes * n_v * d_v).unwrap();

    let mut cold_lanes: Vec<f64> = Vec::new();
    let mut cold_us: Vec<f64> = Vec::new();

    for residency in ["cold", "warm"] {
        for lanes in [1usize, 2, 4, 8, 16] {
            let iters = 400usize;
            let groups = pool_len / lanes;
            let us = bench_us(&stream, iters, |it| {
                let base = if residency == "cold" {
                    (it % groups) * lanes
                } else {
                    0
                };
                let ptrs: Vec<*mut f32> = (0..lanes)
                    .map(|l| {
                        let (p, _g) = pool[base + l].device_ptr(&stream);
                        p as *mut f32
                    })
                    .collect();
                let (pm, _a) = d_mixed.device_ptr(&stream);
                let (pz, _b) = d_z.device_ptr(&stream);
                let (pab, _c) = d_ab.device_ptr(&stream);
                let (pal, _d) = d_alog.device_ptr(&stream);
                let (pdt, _e) = d_dt.device_ptr(&stream);
                let (pnw, _f) = d_nw.device_ptr(&stream);
                let (po, _g) = d_out.device_ptr_mut(&stream);
                let (pqn, _h) = d_qn.device_ptr_mut(&stream);
                let (pkn, _i) = d_kn.device_ptr_mut(&stream);
                let (pge, _j) = d_ge.device_ptr_mut(&stream);
                let (pbe, _k) = d_be.device_ptr_mut(&stream);
                let (pco, _l) = d_core.device_ptr_mut(&stream);
                let rc = unsafe {
                    cuda::gdn_decode_step_split_group_bf16(
                        stream.cu_stream() as *mut c_void,
                        pm as *const u16,
                        pz as *const u16,
                        pab as *const u16,
                        (pab + (n_v * 2) as u64) as *const u16,
                        pal as *const u16,
                        pdt as *const u16,
                        pnw as *const u16,
                        ptrs.as_ptr(),
                        po as *mut u16,
                        pqn as *mut f32,
                        pkn as *mut f32,
                        pge as *mut f32,
                        pbe as *mut f32,
                        pco as *mut u16,
                        n_k as i32,
                        n_v as i32,
                        d_k as i32,
                        d_v as i32,
                        1e-6,
                        lanes as i32,
                        ab_stride as i32,
                    )
                };
                assert_eq!(rc, 0, "gdn_decode_step_split_group_bf16 rc={rc} lanes={lanes}");
            });
            let bytes = 2 * lanes * one_lane_state_bytes;
            let gbs = bytes as f64 / us / 1e3;
            eprintln!(
                "GDNSTATE-LANES group_split l2={residency} lanes={lanes} blocks={} us={us:.2} state_mb={:.2} state_rw_gbs={gbs:.0}",
                n_v * 4 * lanes,
                bytes as f64 / 1048576.0
            );
            if residency == "cold" {
                cold_lanes.push(lanes as f64);
                cold_us.push(us);
            }
        }
    }

    let (slope_us_per_lane, intercept_us) =
        least_squares_slope_and_intercept(&cold_lanes, &cold_us);
    let marginal_gbs = (2 * one_lane_state_bytes) as f64 / slope_us_per_lane / 1e3;
    eprintln!(
        "GDNSTATE-FIT group_split_cold us = {intercept_us:.2} + {slope_us_per_lane:.2}*lanes ; marginal_state_rw_gbs={marginal_gbs:.0}"
    );

    for residency in ["cold", "warm"] {
        let groups = pool_len;
        let iters = 400usize;
        let us = bench_us(&stream, iters, |it| {
            let idx = if residency == "cold" { it % groups } else { 0 };
            let (pst, _s) = pool[idx].device_ptr(&stream);
            let (pm, _a) = d_mixed.device_ptr(&stream);
            let (pz, _b) = d_z.device_ptr(&stream);
            let (pab, _c) = d_ab.device_ptr(&stream);
            let (pal, _d) = d_alog.device_ptr(&stream);
            let (pdt, _e) = d_dt.device_ptr(&stream);
            let (pnw, _f) = d_nw.device_ptr(&stream);
            let (po, _g) = d_out.device_ptr_mut(&stream);
            let (pqn, _h) = d_qn.device_ptr_mut(&stream);
            let (pkn, _i) = d_kn.device_ptr_mut(&stream);
            let (pge, _j) = d_ge.device_ptr_mut(&stream);
            let (pbe, _k) = d_be.device_ptr_mut(&stream);
            let (pco, _l) = d_core.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::gdn_decode_step_split_bf16(
                    stream.cu_stream() as *mut c_void,
                    pm as *const u16,
                    pz as *const u16,
                    pab as *const u16,
                    (pab + (n_v * 2) as u64) as *const u16,
                    pal as *const u16,
                    pdt as *const u16,
                    pnw as *const u16,
                    pst as *mut f32,
                    po as *mut u16,
                    pqn as *mut f32,
                    pkn as *mut f32,
                    pge as *mut f32,
                    pbe as *mut f32,
                    pco as *mut u16,
                    n_k as i32,
                    n_v as i32,
                    d_k as i32,
                    d_v as i32,
                    1e-6,
                )
            };
            assert_eq!(rc, 0, "gdn_decode_step_split_bf16 rc={rc}");
        });
        let bytes = 2 * one_lane_state_bytes;
        let gbs = bytes as f64 / us / 1e3;
        eprintln!(
            "GDNSTATE-SOLO solo_split l2={residency} lanes=1 us={us:.2} state_rw_gbs={gbs:.0}"
        );
    }
}

const A_CAPTURED_GRAPH_IS_THE_SERVING_PATH_SO_THE_FIXED_PER_CALL_TERM_MUST_BE_RE_READ_WITH_DISPATCH_REMOVED: &str =
    "the eager fit's intercept carries three dependent eager launch dispatches that a captured \
     decode graph does not pay; round 28's null on collapsing per-lane launches was measured with \
     graphs ON and this term has to be too or it will be re-attributed to bandwidth";

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_KERNELS_BENCH=1 -- repeats the GDN state lane sweep inside a CAPTURED graph, because the serving decode step is captured and an eager per-call fixed term is not the one serving pays"]
fn gdn_decode_step_split_state_rw_rate_by_lane_parallelism_inside_a_captured_graph() {
    bench_gate("the graphed GDN state lane sweep");
    eprintln!("GDNSTATE-WHY {A_CAPTURED_GRAPH_IS_THE_SERVING_PATH_SO_THE_FIXED_PER_CALL_TERM_MUST_BE_RE_READ_WITH_DISPATCH_REMOVED}");

    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.new_stream().expect("capture stream");
    let mut rng = Lcg(0xb7e151628aed2a6b);

    let n_k = Q38_GDN_NK;
    let n_v = Q38_GDN_NV;
    let d_k = Q38_GDN_DK;
    let d_v = Q38_GDN_DV;
    let key_dim = n_k * d_k;
    let ab_stride = 2 * n_v;
    let max_lanes = 16usize;
    let state_elems = n_v * d_k * d_v;
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

    let mixed_h = rng.bf16_words(max_lanes * Q38_CONV_DIM, 0.5);
    let z_h = rng.bf16_words(max_lanes * Q38_VALUE_DIM, 0.5);
    let ab_h = rng.bf16_words(max_lanes * ab_stride, 0.5);
    let a_log_h = rng.bf16_words(n_v, 0.5);
    let dt_h = rng.bf16_words(n_v, 0.5);
    let nw_h = rng.bf16_words(d_v, 0.5);
    #[allow(deprecated)]
    let d_mixed: CudaSlice<u16> = stream.clone_htod(&mixed_h).unwrap();
    #[allow(deprecated)]
    let d_z: CudaSlice<u16> = stream.clone_htod(&z_h).unwrap();
    #[allow(deprecated)]
    let d_ab: CudaSlice<u16> = stream.clone_htod(&ab_h).unwrap();
    #[allow(deprecated)]
    let d_alog: CudaSlice<u16> = stream.clone_htod(&a_log_h).unwrap();
    #[allow(deprecated)]
    let d_dt: CudaSlice<u16> = stream.clone_htod(&dt_h).unwrap();
    #[allow(deprecated)]
    let d_nw: CudaSlice<u16> = stream.clone_htod(&nw_h).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(max_lanes * Q38_VALUE_DIM).unwrap();
    let mut d_qn: CudaSlice<f32> = stream.alloc_zeros::<f32>(max_lanes * key_dim).unwrap();
    let mut d_kn: CudaSlice<f32> = stream.alloc_zeros::<f32>(max_lanes * key_dim).unwrap();
    let mut d_ge: CudaSlice<f32> = stream.alloc_zeros::<f32>(max_lanes * n_v).unwrap();
    let mut d_be: CudaSlice<f32> = stream.alloc_zeros::<f32>(max_lanes * n_v).unwrap();
    let mut d_core: CudaSlice<u16> = stream.alloc_zeros::<u16>(max_lanes * n_v * d_v).unwrap();

    let (pm, _a) = d_mixed.device_ptr(&stream);
    let (pz, _b) = d_z.device_ptr(&stream);
    let (pab, _c) = d_ab.device_ptr(&stream);
    let (pal, _d) = d_alog.device_ptr(&stream);
    let (pdt, _e) = d_dt.device_ptr(&stream);
    let (pnw, _f) = d_nw.device_ptr(&stream);
    let (po, _g) = d_out.device_ptr_mut(&stream);
    let (pqn, _h) = d_qn.device_ptr_mut(&stream);
    let (pkn, _i) = d_kn.device_ptr_mut(&stream);
    let (pge, _j) = d_ge.device_ptr_mut(&stream);
    let (pbe, _k) = d_be.device_ptr_mut(&stream);
    let (pco, _l) = d_core.device_ptr_mut(&stream);
    let pool_ptrs: Vec<u64> = pool
        .iter()
        .map(|b| {
            let (p, _g) = b.device_ptr(&stream);
            p
        })
        .collect();

    let mut graph_lanes: Vec<f64> = Vec::new();
    let mut graph_us: Vec<f64> = Vec::new();
    let mut runner = nv_kernels::graph::CudaGraphRunner::new(stream.clone());
    for (token, lanes) in [1usize, 2, 4, 8, 16].into_iter().enumerate() {
        let groups = pool_len / lanes;
        let body = |s: &Arc<CudaStream>| {
            for call in 0..calls_per_graph {
                let base = (call % groups) * lanes;
                let ptrs: Vec<*mut f32> = (0..lanes)
                    .map(|l| pool_ptrs[base + l] as *mut f32)
                    .collect();
                let rc = unsafe {
                    cuda::gdn_decode_step_split_group_bf16(
                        s.cu_stream() as *mut c_void,
                        pm as *const u16,
                        pz as *const u16,
                        pab as *const u16,
                        (pab + (n_v * 2) as u64) as *const u16,
                        pal as *const u16,
                        pdt as *const u16,
                        pnw as *const u16,
                        ptrs.as_ptr(),
                        po as *mut u16,
                        pqn as *mut f32,
                        pkn as *mut f32,
                        pge as *mut f32,
                        pbe as *mut f32,
                        pco as *mut u16,
                        n_k as i32,
                        n_v as i32,
                        d_k as i32,
                        d_v as i32,
                        1e-6,
                        lanes as i32,
                        ab_stride as i32,
                    )
                };
                assert_eq!(rc, 0, "graphed gdn_decode_step_split_group_bf16 rc={rc} lanes={lanes}");
            }
        };
        runner
            .run(token as u64 + 1, |s| {
                body(s);
                Ok(())
            })
            .unwrap();
        stream.synchronize().unwrap();
        let reps = 30usize;
        let t0 = Instant::now();
        for _ in 0..reps {
            runner
                .run(token as u64 + 1, |_| {
                    panic!("recapture on a cached token");
                })
                .unwrap();
        }
        stream.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64 / calls_per_graph as f64;
        let bytes = 2 * lanes * one_lane_state_bytes;
        let gbs = bytes as f64 / us / 1e3;
        eprintln!(
            "GDNSTATE-GRAPH group_split graphed lanes={lanes} blocks={} us_per_call={us:.2} state_mb={:.2} state_rw_gbs={gbs:.0}",
            n_v * 4 * lanes,
            bytes as f64 / 1048576.0
        );
        graph_lanes.push(lanes as f64);
        graph_us.push(us);
    }
    let (slope, intercept) = least_squares_slope_and_intercept(&graph_lanes, &graph_us);
    let marginal_gbs = (2 * one_lane_state_bytes) as f64 / slope / 1e3;
    eprintln!(
        "GDNSTATE-FIT group_split_graphed us = {intercept:.2} + {slope:.2}*lanes ; marginal_state_rw_gbs={marginal_gbs:.0}"
    );
}
