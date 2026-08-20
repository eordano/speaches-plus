#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::moe_permute::moe_permute;

pub fn cpu_reference(
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
    (offsets, permuted, inv_perm)
}

pub fn lcg_ids(total: usize, num_experts: usize, seed0: u64) -> Vec<i32> {
    let mut seed = seed0;
    (0..total)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as u32 % num_experts as u32) as i32
        })
        .collect()
}

fn run_case(test: &str, ids: &[i32], n_tokens: usize, k: usize, num_experts: usize) {
    let Some(ctx) = ctx_or_skip(test) else {
        return;
    };
    let total = n_tokens * k;
    let mut offsets = vec![-7i32; num_experts + 1];
    let mut perm = vec![-7i32; total];
    let mut inv = vec![-7i32; total];
    moe_permute(
        ctx,
        ids,
        &mut offsets,
        &mut perm,
        &mut inv,
        n_tokens,
        k,
        num_experts,
    )
    .expect("moe_permute");

    let (off_ref, perm_ref, inv_ref) = cpu_reference(ids, n_tokens, k, num_experts);
    assert_eq!(offsets, off_ref, "{test}: expert_offsets mismatch");
    assert_eq!(inv, inv_ref, "{test}: inv_perm mismatch");
    assert_eq!(perm, perm_ref, "{test}: permuted_token_idx mismatch");

    let mut sorted = inv.clone();
    sorted.sort();
    let expect: Vec<i32> = (0..total as i32).collect();
    assert_eq!(sorted, expect, "{test}: inv_perm is not a permutation");
    for t in 0..total {
        assert_eq!(
            perm[inv[t] as usize],
            (t / k) as i32,
            "{test}: inv/perm inconsistent at {t}"
        );
        let e = ids[t] as usize;
        assert!(
            inv[t] >= offsets[e] && inv[t] < offsets[e + 1],
            "{test}: slot {t} landed outside expert {e} range"
        );
    }
    eprintln!("{test}: bit-exact vs CPU oracle over {total} slots");
}

#[test]
fn wgpu_moe_permute_tiny() {
    let ids = vec![0, 1, 1, 2, 0, 2, 1, 0];
    run_case("wgpu_moe_permute_tiny", &ids, 4, 2, 3);
}

#[test]
fn wgpu_moe_permute_k1() {
    let ids = lcg_ids(64, 8, 0x9E3779B97F4A7C15);
    run_case("wgpu_moe_permute_k1", &ids, 64, 1, 8);
}

#[test]
fn wgpu_moe_permute_qwen_like() {
    let (n, k, e) = (32usize, 8usize, 256usize);
    let ids = lcg_ids(n * k, e, 0x9E3779B97F4A7C15);
    run_case("wgpu_moe_permute_qwen_like", &ids, n, k, e);
}

#[test]
fn wgpu_moe_permute_empty_experts() {
    let (n, k, e) = (16usize, 4usize, 256usize);
    let ids = vec![42i32; n * k];
    run_case("wgpu_moe_permute_empty_experts", &ids, n, k, e);
}

#[test]
fn wgpu_moe_permute_single_expert_all() {
    let (n, k, e) = (300usize, 2usize, 1usize);
    let ids = vec![0i32; n * k];
    run_case("wgpu_moe_permute_single_expert_all", &ids, n, k, e);
}

#[test]
fn wgpu_moe_permute_multi_block() {
    let (n, k, e) = (1024usize, 4usize, 128usize);
    let ids = lcg_ids(n * k, e, 0xdeadbeefcafe1234);
    run_case("wgpu_moe_permute_multi_block", &ids, n, k, e);
}

#[test]
fn wgpu_moe_permute_ragged_tail() {
    let (n, k, e) = (259usize, 3usize, 17usize);
    let ids = lcg_ids(n * k, e, 0x0123456789abcdef);
    run_case("wgpu_moe_permute_ragged_tail", &ids, n, k, e);
}

#[test]
fn wgpu_moe_permute_padded_perm_tail_untouched() {
    let Some(ctx) = ctx_or_skip("wgpu_moe_permute_padded_perm_tail_untouched") else {
        return;
    };
    let (n, k, e) = (8usize, 2usize, 4usize);
    let total = n * k;
    let ids = lcg_ids(total, e, 0x5555aaaa5555aaaa);
    let mut offsets = vec![0i32; e + 1];
    let mut perm = vec![-1i32; total + 32];
    let mut inv = vec![0i32; total];
    moe_permute(ctx, &ids, &mut offsets, &mut perm, &mut inv, n, k, e).expect("moe_permute");
    let (off_ref, perm_ref, inv_ref) = cpu_reference(&ids, n, k, e);
    assert_eq!(offsets, off_ref);
    assert_eq!(&perm[..total], &perm_ref[..]);
    assert_eq!(inv, inv_ref);
    assert!(
        perm[total..].iter().all(|&x| x == -1),
        "padding was written"
    );
}

#[test]
fn wgpu_moe_permute_zero_sized_is_noop() {
    let Some(ctx) = ctx_or_skip("wgpu_moe_permute_zero_sized_is_noop") else {
        return;
    };
    let mut offsets = vec![9i32; 5];
    let mut perm: Vec<i32> = Vec::new();
    let mut inv: Vec<i32> = Vec::new();
    moe_permute(ctx, &[], &mut offsets, &mut perm, &mut inv, 0, 2, 4).expect("n_tokens=0");
    assert_eq!(offsets, vec![9i32; 5]);
}

#[test]
fn wgpu_moe_permute_shape_errors() {
    let Some(ctx) = ctx_or_skip("wgpu_moe_permute_shape_errors") else {
        return;
    };
    let mut offsets = vec![0i32; 4];
    let mut perm = vec![0i32; 8];
    let mut inv = vec![0i32; 8];
    let err = moe_permute(ctx, &[0i32; 7], &mut offsets, &mut perm, &mut inv, 4, 2, 3).unwrap_err();
    assert!(format!("{err}").contains("topk_ids"), "{err}");
}
