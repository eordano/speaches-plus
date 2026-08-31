#![cfg(feature = "cuda")]

mod common;
use common::xorshift;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;

const HEAD_DIM: usize = 512;
const N_Q: usize = 16;
const N_KV: usize = 4;
const BLOCK_SIZE: usize = 16;
const TOTAL: usize = 4096;

const WIDEST_INSTANTIATED_RUNG_AND_EVERY_SPLIT_WRITES_ONE_SSC_SLOT: usize = 47;

const SHIPPED_RUNGS_WHOSE_OUTPUT_BITS_MAY_NEVER_MOVE_WITHOUT_A_REMEASURE: [i32; 2] = [16, 32];

const BF16_SPLIT_REASSOCIATION_BAND_AN_SSC_OVERRUN_INTO_A_NEIGHBOURING_SHARED_VALUE_IS_NOT_REASSOCIATION:
    f32 = 2.0e-2;

fn fnv1a64(bits: &[u16]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bits {
        for byte in b.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn decode_bits_at_rung(rung: i32) -> Vec<u16> {
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.default_stream();
    let raw = stream.cu_stream() as *mut c_void;

    let mut state = 0x2545_f491u32;
    let elems = TOTAL * N_KV * HEAD_DIM;
    let src: Vec<u16> = (0..elems)
        .map(|_| bf16::from_f32(xorshift(&mut state)).to_bits())
        .collect();
    let q: Vec<u16> = (0..N_Q * HEAD_DIM)
        .map(|_| bf16::from_f32(xorshift(&mut state)).to_bits())
        .collect();

    let blocks = TOTAL.div_ceil(BLOCK_SIZE);
    let table: Vec<i32> = (0..blocks as i32).collect();
    let slots = blocks * BLOCK_SIZE;

    #[allow(deprecated)]
    let d_table: CudaSlice<i32> = stream.clone_htod(&table).unwrap();
    #[allow(deprecated)]
    let d_start: CudaSlice<i32> = stream.clone_htod(&vec![0i32]).unwrap();
    #[allow(deprecated)]
    let d_ntot: CudaSlice<i32> = stream.clone_htod(&vec![TOTAL as i32]).unwrap();
    #[allow(deprecated)]
    let d_q: CudaSlice<u16> = stream.clone_htod(&q).unwrap();
    #[allow(deprecated)]
    let d_src: CudaSlice<u16> = stream.clone_htod(&src).unwrap();

    let quantize = || -> (CudaSlice<u8>, CudaSlice<f32>) {
        let mut d_fp8: CudaSlice<u8> =
            stream.alloc_zeros::<u8>(slots * N_KV * HEAD_DIM).unwrap();
        let mut d_sc: CudaSlice<f32> = stream.alloc_zeros::<f32>(slots * N_KV).unwrap();
        let rc = {
            let (p_src, _a) = d_src.device_ptr(&stream);
            let (p_st, _b) = d_start.device_ptr(&stream);
            let (p_tb, _c) = d_table.device_ptr(&stream);
            let (p_fp8, _d) = d_fp8.device_ptr_mut(&stream);
            let (p_sc, _e) = d_sc.device_ptr_mut(&stream);
            unsafe {
                cuda::quantize_kv_fp8_paged(
                    raw,
                    p_src as *const u16,
                    p_fp8 as *mut u8,
                    p_sc as *mut f32,
                    p_st as *const i32,
                    p_tb as *const i32,
                    BLOCK_SIZE as i32,
                    TOTAL as i32,
                    N_KV as i32,
                    HEAD_DIM as i32,
                )
            }
        };
        assert_eq!(rc, 0, "quantize rc={rc}");
        stream.synchronize().unwrap();
        (d_fp8, d_sc)
    };
    let (d_k, d_ks) = quantize();
    let (d_v, d_vs) = quantize();

    let mut d_scratch: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(
            N_Q * WIDEST_INSTANTIATED_RUNG_AND_EVERY_SPLIT_WRITES_ONE_SSC_SLOT * (HEAD_DIM + 2),
        )
        .unwrap();
    let mut d_fan: CudaSlice<u32> = stream.alloc_zeros::<u32>(N_Q).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(N_Q * HEAD_DIM).unwrap();
    let scaling = 1.0f32 / (HEAD_DIM as f32).sqrt();

    std::env::set_var("NV_FLASH_SPLITS_RUNG", rung.to_string());
    let picked = cuda::flash_decode_fp8_splits_pick(N_Q as i32, 0);
    assert_eq!(
        picked, rung,
        "NV_FLASH_SPLITS_RUNG={rung} did not reach the picker (picked {picked}); this launch \
         would measure a different rung and every comparison below would be vacuous"
    );
    let rc = {
        let (p_q, _a) = d_q.device_ptr(&stream);
        let (p_k, _b) = d_k.device_ptr(&stream);
        let (p_v, _c) = d_v.device_ptr(&stream);
        let (p_ks, _d) = d_ks.device_ptr(&stream);
        let (p_vs, _e) = d_vs.device_ptr(&stream);
        let (p_nt, _f) = d_ntot.device_ptr(&stream);
        let (p_tb, _g) = d_table.device_ptr(&stream);
        let (p_out, _h) = d_out.device_ptr_mut(&stream);
        let (p_sc, _i) = d_scratch.device_ptr_mut(&stream);
        let (p_fan, _j) = d_fan.device_ptr_mut(&stream);
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
                N_Q as i32,
                N_KV as i32,
                HEAD_DIM as i32,
                0,
                0,
                scaling,
                p_tb as *const i32,
                BLOCK_SIZE as i32,
            )
        }
    };
    std::env::remove_var("NV_FLASH_SPLITS_RUNG");
    assert_eq!(rc, 0, "fused fp8 decode at rung {rung} returned {rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    stream.clone_dtoh(&d_out).unwrap()
}

#[test]
fn rung_47_writes_one_ssc_slot_per_split_so_the_array_must_span_47_not_32() {
    let reference = decode_bits_at_rung(16);
    eprintln!("[ssc47] rung 16 fnv1a64 {:016x}", fnv1a64(&reference));
    let ref_f32: Vec<f32> = reference
        .iter()
        .map(|b| bf16::from_bits(*b).to_f32())
        .collect();
    let mut sweep: Vec<i32> = SHIPPED_RUNGS_WHOSE_OUTPUT_BITS_MAY_NEVER_MOVE_WITHOUT_A_REMEASURE
        .iter()
        .filter(|r| **r != 16)
        .copied()
        .collect();
    sweep.push(WIDEST_INSTANTIATED_RUNG_AND_EVERY_SPLIT_WRITES_ONE_SSC_SLOT as i32);
    for rung in sweep {
        let got = decode_bits_at_rung(rung);
        eprintln!("[ssc47] rung {rung} fnv1a64 {:016x}", fnv1a64(&got));
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for (i, (w, g)) in ref_f32.iter().zip(&got).enumerate() {
            let g = bf16::from_bits(*g).to_f32();
            let d = (w - g).abs() / w.abs().max(g.abs()).max(1e-6);
            if d > worst {
                worst = d;
                at = i;
            }
        }
        eprintln!("[ssc47] rung {rung} vs 16 worst rel {worst:.6} at {at}");
        assert!(
            worst
                <= BF16_SPLIT_REASSOCIATION_BAND_AN_SSC_OVERRUN_INTO_A_NEIGHBOURING_SHARED_VALUE_IS_NOT_REASSOCIATION,
            "rung {rung} vs rung 16 worst rel {worst:.6} at element {at}: splits past slot 31 \
             must land in ssc storage of their own, not in whatever shared value follows the array"
        );
    }
}
