#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use std::ffi::c_void;

fn cpu_reference(
    topk_ids: &[i32],
    n_tokens: usize,
    k: usize,
    num_experts: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut counts = vec![0i32; num_experts];
    for &e in topk_ids {
        counts[e as usize] += 1;
    }
    let mut offsets = vec![0i32; num_experts + 1];
    for e in 0..num_experts {
        offsets[e + 1] = offsets[e] + counts[e];
    }
    let mut cursors = vec![0i32; num_experts];
    let mut permuted = vec![0i32; n_tokens * k];
    let mut inv_perm = vec![0i32; n_tokens * k];
    for t in 0..n_tokens * k {
        let n = (t / k) as i32;
        let e = topk_ids[t] as usize;
        let pos = (offsets[e] + cursors[e]) as usize;
        cursors[e] += 1;
        permuted[pos] = n;
        inv_perm[t] = pos as i32;
    }
    (permuted, offsets, inv_perm)
}

fn run_case(topk_ids: Vec<i32>, n_tokens: usize, k: usize, num_experts: usize) {
    let ctx = CudaContext::new(0).expect("cuda ctx");
    let stream = ctx.default_stream();

    #[allow(deprecated)]
    let ids_dev = stream.memcpy_stod(&topk_ids).unwrap();
    let mut perm_dev = stream.alloc_zeros::<i32>(n_tokens * k).unwrap();
    let mut offsets_dev = stream.alloc_zeros::<i32>(num_experts + 1).unwrap();
    let mut inv_dev = stream.alloc_zeros::<i32>(n_tokens * k).unwrap();
    let mut scratch_dev = stream.alloc_zeros::<i32>(num_experts).unwrap();

    let rc = {
        let (ids_p, _g1) = ids_dev.device_ptr(&stream);
        let (perm_p, _g2) = perm_dev.device_ptr_mut(&stream);
        let (off_p, _g3) = offsets_dev.device_ptr_mut(&stream);
        let (inv_p, _g4) = inv_dev.device_ptr_mut(&stream);
        let (scr_p, _g5) = scratch_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::moe_permute(
                stream.cu_stream() as *mut c_void,
                ids_p as *const i32,
                perm_p as *mut i32,
                off_p as *mut i32,
                inv_p as *mut i32,
                scr_p as *mut i32,
                n_tokens as i32,
                k as i32,
                num_experts as i32,
            )
        }
    };
    assert_eq!(rc, 0, "moe_permute rc={rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let perm_got = stream.memcpy_dtov(&perm_dev).unwrap();
    #[allow(deprecated)]
    let offsets_got = stream.memcpy_dtov(&offsets_dev).unwrap();
    #[allow(deprecated)]
    let inv_got = stream.memcpy_dtov(&inv_dev).unwrap();

    let (perm_ref, offsets_ref, inv_ref) = cpu_reference(&topk_ids, n_tokens, k, num_experts);

    assert_eq!(offsets_got, offsets_ref, "expert_offsets mismatch");

    for t in 0..n_tokens * k {
        let pos = inv_got[t] as usize;
        let expected_n = (t / k) as i32;
        assert_eq!(
            perm_got[pos], expected_n,
            "inv_perm/permuted_token_idx inconsistent at slot {t}"
        );
    }
    for e in 0..num_experts {
        let lo = offsets_got[e] as usize;
        let hi = offsets_got[e + 1] as usize;
        let mut got_seg: Vec<i32> = perm_got[lo..hi].to_vec();
        let mut ref_seg: Vec<i32> = perm_ref[lo..hi].to_vec();
        got_seg.sort();
        ref_seg.sort();
        assert_eq!(got_seg, ref_seg, "expert {e} segment multiset differs");
    }
    let mut sorted_inv = inv_got.clone();
    sorted_inv.sort();
    let expected_inv: Vec<i32> = (0..(n_tokens * k) as i32).collect();
    assert_eq!(sorted_inv, expected_inv, "inv_perm is not a permutation");
    let mut sorted_ref = inv_ref.clone();
    sorted_ref.sort();
    assert_eq!(sorted_ref, expected_inv);
}

#[test]
fn moe_permute_tiny() {
    let ids = vec![0, 1, 1, 2, 0, 2, 1, 0];
    run_case(ids, 4, 2, 3);
}

#[test]
fn moe_permute_qwen_like() {
    let n = 32usize;
    let k = 8usize;
    let e = 256usize;
    let mut ids = Vec::with_capacity(n * k);
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    for _ in 0..(n * k) {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ids.push(((seed >> 33) as u32 % e as u32) as i32);
    }
    run_case(ids, n, k, e);
}

#[test]
fn moe_permute_empty_experts() {
    let n = 16usize;
    let k = 4usize;
    let e = 256usize;
    let ids = vec![42i32; n * k];
    run_case(ids, n, k, e);
}
