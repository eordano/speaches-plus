#![cfg(feature = "cuda")]

mod common;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

const QWEN38_FULL_ATTENTION_NH_NKV_HD_FROM_THE_CHECKPOINT_CONFIG_NOT_A_PROXY: (usize, usize, usize) =
    (24, 4, 256);

const QWEN38_HAS_THIS_MANY_FULL_ATTENTION_LAYERS_SO_ONE_GROUP_WALL_TIMES_THIS_IS_THE_WHOLE_ATTN_LANES_STAGE:
    usize = 16;

const A_KV_SET_THAT_FITS_IN_L2_MAKES_EVERY_SPLIT_RUNG_LOOK_ALIKE_BECAUSE_THE_SCRATCH_AND_THE_CACHE_BOTH_COME_BACK_RESIDENT_SO_THE_LADDER_ROTATES_OVER_A_WORKING_SET_LARGER_THAN_L2:
    usize = 768 << 20;

const THE_SPLIT_PICK_TARGETS_TWO_WORKGROUPS_PER_SM_FOR_ONE_LANE_AND_CANNOT_SEE_THAT_M_LANES_LAUNCH_AT_ONCE_SO_THE_LADDER_REPORTS_BLOCKS_IN_FLIGHT_BESIDE_EVERY_CELL:
    &str = "blocks_in_flight = NH * splits * lanes; at lanes=16 the shipped rung puts sixteen \
            times the two-per-SM target on the card and every extra block writes and re-reads a \
            (HD+2)-float partial through the split-k scratch";

const RUNGS_THE_LADDER_ASKS_FOR_INCLUDING_ONES_THAT_MAY_NOT_BE_INSTANTIATED_BECAUSE_A_REFUSAL_IS_A_READING:
    [i32; 10] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32];

const RUNG_47_IS_LEFT_OUT_BECAUSE_FLASH_SPLITK_FUSED_FP8_KERNELS_FINAL_MERGE_HOLDS_ITS_PER_SPLIT_RESCALES_IN_A_FIXED_SHARED_SSC_32_AND_A_47_SPLIT_LAUNCH_WRITES_FIFTEEN_FLOATS_PAST_IT:
    i32 = 47;

const WIDEST_INSTANTIATED_RUNG_SIZES_THE_SCRATCH_ONCE_SO_NO_CELL_REALLOCATES: i32 = 47;

fn depths_env_nv_q38_attn_lane_depths_default_96_1024_8192() -> Vec<usize> {
    std::env::var("NV_Q38_ATTN_LANE_DEPTHS")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![96, 1024, 8192])
}

fn lane_counts_env_nv_q38_attn_lane_counts_default_1_2_4_8_16() -> Vec<usize> {
    std::env::var("NV_Q38_ATTN_LANE_COUNTS")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16])
}

struct LanePool {
    k: CudaSlice<u8>,
    v: CudaSlice<u8>,
    ks: CudaSlice<f32>,
    vs: CudaSlice<f32>,
    q: CudaSlice<u16>,
    out: CudaSlice<u16>,
    scratch: Vec<CudaSlice<f32>>,
    fan_in: Vec<CudaSlice<u32>>,
    n_total: CudaSlice<i32>,
    replicas: usize,
}

fn build_pool(stream: &Arc<CudaStream>, lanes: usize, depth: usize) -> LanePool {
    let (nh, nkv, hd) = QWEN38_FULL_ATTENTION_NH_NKV_HD_FROM_THE_CHECKPOINT_CONFIG_NOT_A_PROXY;
    let per_lane_bytes = depth * nkv * hd;
    let one_set = per_lane_bytes * lanes;
    let replicas = (A_KV_SET_THAT_FITS_IN_L2_MAKES_EVERY_SPLIT_RUNG_LOOK_ALIKE_BECAUSE_THE_SCRATCH_AND_THE_CACHE_BOTH_COME_BACK_RESIDENT_SO_THE_LADDER_ROTATES_OVER_A_WORKING_SET_LARGER_THAN_L2
        / (2 * one_set.max(1)))
    .max(1);
    let pos_total = depth * lanes * replicas;

    let mut state = 0x2545_f491u32;
    let kbytes: Vec<u8> = (0..pos_total * nkv * hd)
        .map(|_| {
            let f = common::xorshift(&mut state);
            (((f * 0.5 + 0.5) * 200.0) as u8).max(1)
        })
        .collect();
    let scales: Vec<f32> = (0..pos_total * nkv).map(|_| 0.03f32).collect();
    let qv: Vec<u16> = (0..lanes * nh * hd)
        .map(|_| half::bf16::from_f32(common::xorshift(&mut state) * 0.1).to_bits())
        .collect();

    #[allow(deprecated)]
    let k: CudaSlice<u8> = stream.clone_htod(&kbytes).unwrap();
    #[allow(deprecated)]
    let v: CudaSlice<u8> = stream.clone_htod(&kbytes).unwrap();
    #[allow(deprecated)]
    let ks: CudaSlice<f32> = stream.clone_htod(&scales).unwrap();
    #[allow(deprecated)]
    let vs: CudaSlice<f32> = stream.clone_htod(&scales).unwrap();
    #[allow(deprecated)]
    let q: CudaSlice<u16> = stream.clone_htod(&qv).unwrap();
    let out: CudaSlice<u16> = stream.alloc_zeros::<u16>(lanes * nh * hd).unwrap();
    #[allow(deprecated)]
    let n_total: CudaSlice<i32> = stream.clone_htod(&vec![depth as i32; lanes]).unwrap();
    let scratch_elems = nh * WIDEST_INSTANTIATED_RUNG_SIZES_THE_SCRATCH_ONCE_SO_NO_CELL_REALLOCATES
        as usize
        * (hd + 2);
    let scratch = (0..lanes)
        .map(|_| stream.alloc_zeros::<f32>(scratch_elems).unwrap())
        .collect();
    let fan_in = (0..lanes)
        .map(|_| stream.alloc_zeros::<u32>(nh).unwrap())
        .collect();
    stream.synchronize().unwrap();
    LanePool {
        k,
        v,
        ks,
        vs,
        q,
        out,
        scratch,
        fan_in,
        n_total,
        replicas,
    }
}

#[allow(clippy::too_many_arguments)]
fn one_group_launch(
    pool: &mut LanePool,
    streams: &[Arc<CudaStream>],
    lanes: usize,
    depth: usize,
    iter: usize,
) -> i32 {
    let (nh, nkv, hd) = QWEN38_FULL_ATTENTION_NH_NKV_HD_FROM_THE_CHECKPOINT_CONFIG_NOT_A_PROXY;
    let mut worst = 0i32;
    for i in 0..lanes {
        let s = &streams[i];
        let slot = ((iter * lanes + i) % (pool.replicas * lanes)) * depth;
        let kview = pool.k.slice(slot * nkv * hd..);
        let vview = pool.v.slice(slot * nkv * hd..);
        let ksview = pool.ks.slice(slot * nkv..);
        let vsview = pool.vs.slice(slot * nkv..);
        let qview = pool.q.slice(i * nh * hd..);
        let ntview = pool.n_total.slice(i..);
        let (kp, _a) = kview.device_ptr(s);
        let (vp, _b) = vview.device_ptr(s);
        let (ksp, _c) = ksview.device_ptr(s);
        let (vsp, _d) = vsview.device_ptr(s);
        let (qp, _e) = qview.device_ptr(s);
        let (ntp, _f) = ntview.device_ptr(s);
        let (scp, _g) = pool.scratch[i].device_ptr_mut(s);
        let (fip, _h) = pool.fan_in[i].device_ptr_mut(s);
        let (op, _i) = pool.out.device_ptr_mut(s);
        let rc = unsafe {
            cuda::flash_decode_fused_fp8kv(
                s.cu_stream() as *mut c_void,
                qp as *const u16,
                kp as *const u8,
                vp as *const u8,
                ksp as *const f32,
                vsp as *const f32,
                (op as usize + i * nh * hd * 2) as *mut u16,
                ntp as *const i32,
                scp as *mut f32,
                fip as *mut u32,
                nh as i32,
                nkv as i32,
                hd as i32,
                0,
                0,
                0.0625,
            )
        };
        if rc != 0 {
            worst = rc;
        }
    }
    worst
}

#[test]
#[ignore = "prices the qwen3.8 full-attention decode kernel as a LANE GROUP: m concurrent \
            per-lane launches on m streams, over a KV working set larger than L2, at every \
            split rung including ones that may not be instantiated (a refusal is printed, not \
            asserted). Set NV_Q38_ATTN_LANE_LADDER=1"]
fn the_lane_group_attention_decode_is_priced_at_every_lane_count_and_split_rung_over_a_kv_set_larger_than_l2_with_its_refusals_visible(
) {
    if std::env::var("NV_Q38_ATTN_LANE_LADDER").as_deref() != Ok("1") {
        panic!("set NV_Q38_ATTN_LANE_LADDER=1 to run this GPU bench (it must never silently skip)");
    }
    eprintln!(
        "[q38-attn-lane] {THE_SPLIT_PICK_TARGETS_TWO_WORKGROUPS_PER_SM_FOR_ONE_LANE_AND_CANNOT_SEE_THAT_M_LANES_LAUNCH_AT_ONCE_SO_THE_LADDER_REPORTS_BLOCKS_IN_FLIGHT_BESIDE_EVERY_CELL}"
    );
    let (nh, _nkv, _hd) = QWEN38_FULL_ATTENTION_NH_NKV_HD_FROM_THE_CHECKPOINT_CONFIG_NOT_A_PROXY;
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let main = ctx.default_stream();
    let sm = cuda::flash_sm_count();
    eprintln!("[q38-attn-lane] sm_count={sm} two_per_sm_target={}", 2 * sm);
    let _ = RUNG_47_IS_LEFT_OUT_BECAUSE_FLASH_SPLITK_FUSED_FP8_KERNELS_FINAL_MERGE_HOLDS_ITS_PER_SPLIT_RESCALES_IN_A_FIXED_SHARED_SSC_32_AND_A_47_SPLIT_LAUNCH_WRITES_FIFTEEN_FLOATS_PAST_IT;
    for m in [1i32, 2, 3, 4, 6, 8, 12, 16] {
        eprintln!(
            "[q38-attn-lane] PICK lanes={m} one_lane_rung={} lane_group_rung={}",
            cuda::flash_decode_fp8_splits_pick(nh as i32, 0),
            cuda::flash_decode_fp8_splits_pick_for_a_lane_group(nh as i32, 0, m),
        );
    }

    let iters: usize = std::env::var("NV_Q38_ATTN_LANE_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let warm: usize = 5;

    for depth in depths_env_nv_q38_attn_lane_depths_default_96_1024_8192() {
        for lanes in lane_counts_env_nv_q38_attn_lane_counts_default_1_2_4_8_16() {
            let streams: Vec<Arc<CudaStream>> = (0..lanes)
                .map(|i| {
                    if i == 0 {
                        main.clone()
                    } else {
                        ctx.new_stream().expect("lane stream")
                    }
                })
                .collect();
            let mut pool = build_pool(&main, lanes, depth);
            let ws = 2 * depth * lanes * pool.replicas * 4 * 256;
            for rung in RUNGS_THE_LADDER_ASKS_FOR_INCLUDING_ONES_THAT_MAY_NOT_BE_INSTANTIATED_BECAUSE_A_REFUSAL_IS_A_READING
            {
                std::env::set_var("NV_FLASH_SPLITS_RUNG", rung.to_string());
                let effective = cuda::flash_decode_fp8_splits_pick(nh as i32, 0);
                if effective != rung {
                    eprintln!(
                        "[q38-attn-lane] depth={depth} lanes={lanes} rung={rung} \
                         REFUSED not_instantiated effective={effective}"
                    );
                    continue;
                }
                for it in 0..warm {
                    let rc = one_group_launch(&mut pool, &streams, lanes, depth, it);
                    if rc != 0 {
                        eprintln!(
                            "[q38-attn-lane] depth={depth} lanes={lanes} rung={rung} \
                             REFUSED launch rc={rc}"
                        );
                        break;
                    }
                }
                if main.synchronize().is_err() {
                    eprintln!(
                        "[q38-attn-lane] depth={depth} lanes={lanes} rung={rung} REFUSED sync"
                    );
                    continue;
                }
                for s in &streams {
                    s.synchronize().unwrap();
                }
                let t0 = Instant::now();
                let mut rc_all = 0i32;
                for it in 0..iters {
                    let rc = one_group_launch(&mut pool, &streams, lanes, depth, warm + it);
                    if rc != 0 {
                        rc_all = rc;
                    }
                }
                for s in &streams {
                    s.synchronize().unwrap();
                }
                let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
                if rc_all != 0 {
                    eprintln!(
                        "[q38-attn-lane] depth={depth} lanes={lanes} rung={rung} REFUSED rc={rc_all}"
                    );
                    continue;
                }
                let blocks = nh as i64 * rung as i64 * lanes as i64;
                let kv_bytes = 2.0 * depth as f64 * 4.0 * 256.0 * lanes as f64;
                let scratch_bytes = 2.0 * nh as f64 * rung as f64 * 258.0 * 4.0 * lanes as f64;
                let step_ms = us
                    * QWEN38_HAS_THIS_MANY_FULL_ATTENTION_LAYERS_SO_ONE_GROUP_WALL_TIMES_THIS_IS_THE_WHOLE_ATTN_LANES_STAGE
                        as f64
                    / 1000.0;
                eprintln!(
                    "[q38-attn-lane] LANEGROUP depth={depth} lanes={lanes} rung={rung} \
                     group_us={us:.2} blocks_in_flight={blocks} \
                     kv_bytes={kv_bytes:.0} scratch_bytes={scratch_bytes:.0} \
                     scratch_over_kv={:.2}x attn_only_ms_per_step={step_ms:.3} ws_bytes={ws}",
                    scratch_bytes / kv_bytes,
                );
            }
            std::env::remove_var("NV_FLASH_SPLITS_RUNG");
        }
    }
}
