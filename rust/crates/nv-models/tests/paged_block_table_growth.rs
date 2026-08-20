#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::Gemma4Cache;
use nv_models::paged_fp8::{LayerKvGeometry, PagedGemma4Cache, PagedKvFp8Pool, PagedPoolConfig};
use std::sync::{Arc, Mutex};

const BLOCK_SIZE: usize = 16;
const NUM_BLOCKS: usize = 8;
const N_KV: usize = 2;
const HEAD_DIM: usize = 128;

fn cfg() -> PagedPoolConfig {
    PagedPoolConfig {
        num_blocks: NUM_BLOCKS,
        block_size: BLOCK_SIZE,
        layers: vec![LayerKvGeometry {
            n_kv: N_KV,
            head_dim: HEAD_DIM,
        }],
        layer_blocks: vec![NUM_BLOCKS],
        layer_sliding: vec![false],
        lanes: 0,
        sliding_ring_blocks: 0,
    }
}

fn token_value(t: usize, h: usize, d: usize) -> f32 {
    let x = ((t * 31 + h * 7 + d) % 61) as f32 / 61.0;
    (t as f32 % 8.0) - 3.5 + x
}

fn token_tensor(t: usize, device: &Device) -> Tensor {
    let mut host = Vec::with_capacity(N_KV * HEAD_DIM);
    for h in 0..N_KV {
        for d in 0..HEAD_DIM {
            host.push(token_value(t, h, d));
        }
    }
    Tensor::from_vec(host, (1, 1, N_KV, HEAD_DIM), device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

fn new_cache(device: &Device) -> (Arc<Mutex<PagedKvFp8Pool>>, PagedGemma4Cache) {
    let pool = Arc::new(Mutex::new(
        PagedKvFp8Pool::new(cfg(), device).expect("pool"),
    ));
    let cache = PagedGemma4Cache::new(pool.clone(), device).expect("cache");
    (pool, cache)
}

fn check_view(cache: &mut PagedGemma4Cache, n_tokens: usize) {
    let (kg, vg) = cache.view(0, n_tokens).expect("view");
    for (name, t) in [("k", kg), ("v", vg)] {
        let got: Vec<f32> = t
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            got.len(),
            n_tokens * N_KV * HEAD_DIM,
            "{name}: view returned {} elems for {n_tokens} tokens",
            got.len()
        );
        for tk in 0..n_tokens {
            for h in 0..N_KV {
                for d in 0..HEAD_DIM {
                    let want = token_value(tk, h, d);
                    let idx = (tk * N_KV + h) * HEAD_DIM + d;
                    let tol = want.abs() / 12.0 + 5e-2;
                    assert!(
                        (got[idx] - want).abs() <= tol,
                        "{name} mismatch at token {tk} head {h} dim {d}: got {} want {want} \
                         (tol {tol}) -- a block-table entry was not uploaded",
                        got[idx]
                    );
                }
            }
        }
    }
}

#[test]
#[ignore]
fn growing_block_table_round_trips_every_token() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let (_pool, mut cache) = new_cache(&device);

    let blocks: [u32; 4] = [6, 1, 4, 3];
    let n_tokens = blocks.len() * BLOCK_SIZE;

    for t in 0..n_tokens {
        let need = t / BLOCK_SIZE + 1;
        cache.set_block_table(&blocks[..need]).expect("set table");
        cache.prepare_for_decode(t, t + 1).expect("prepare");
        let kv = token_tensor(t, &device);
        cache.write_at(0, &kv, &kv).expect("write_at");
        cache.advance(1);
    }

    assert_eq!(cache.block_table(), &blocks[..]);
    check_view(&mut cache, n_tokens);
    eprintln!("growing_block_table_round_trips_every_token: {n_tokens} tokens OK");
}

#[test]
#[ignore]
fn remapped_prefix_entry_is_reuploaded() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let (_pool, mut cache) = new_cache(&device);

    cache.set_block_table(&[6u32]).expect("set table");
    for t in 0..BLOCK_SIZE {
        cache.prepare_for_decode(t, t + 1).expect("prepare");
        let kv = token_tensor(t, &device);
        cache.write_at(0, &kv, &kv).expect("write_at");
        cache.advance(1);
    }
    check_view(&mut cache, BLOCK_SIZE);

    cache.set_block_table(&[1u32]).expect("set remapped table");
    let (kg, _) = cache.view(0, BLOCK_SIZE).expect("view");
    let got: Vec<f32> = kg
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let peak = got.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(
        peak < 1e-3,
        "entry 0 was not re-uploaded: reading through the remapped table still \
         returns block 6's KV (peak |value| {peak}, expected the zeroed block 1)"
    );

    cache.set_block_table(&[6u32]).expect("map back");
    check_view(&mut cache, BLOCK_SIZE);
    eprintln!("remapped_prefix_entry_is_reuploaded: OK (remapped peak {peak})");
}

#[test]
#[ignore]
fn shrink_then_regrow_uploads_the_new_tail() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let (pool, mut cache) = new_cache(&device);

    cache.set_block_table(&[6u32, 1]).expect("set table");
    for t in 0..BLOCK_SIZE {
        cache.prepare_for_decode(t, t + 1).expect("prepare");
        let kv = token_tensor(t, &device);
        cache.write_at(0, &kv, &kv).expect("write_at");
        cache.advance(1);
    }

    cache.set_block_table(&[6u32]).expect("shrink");
    assert_eq!(cache.block_table(), &[6u32]);
    check_view(&mut cache, BLOCK_SIZE);

    cache.set_block_table(&[6u32, 4]).expect("regrow");
    for t in BLOCK_SIZE..2 * BLOCK_SIZE {
        cache.prepare_for_decode(t, t + 1).expect("prepare");
        let kv = token_tensor(t, &device);
        cache.write_at(0, &kv, &kv).expect("write_at");
        cache.advance(1);
    }
    check_view(&mut cache, 2 * BLOCK_SIZE);

    let probe_block = |block: u32| -> Vec<f32> {
        let mut probe = PagedGemma4Cache::new(pool.clone(), &device).expect("probe cache");
        probe.set_block_table(&[block]).expect("probe table");
        let (kg, _) = probe.view(0, BLOCK_SIZE).expect("probe view");
        kg.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let want_second: Vec<f32> = (BLOCK_SIZE..2 * BLOCK_SIZE)
        .flat_map(|t| (0..N_KV).flat_map(move |h| (0..HEAD_DIM).map(move |d| token_value(t, h, d))))
        .collect();
    let matches = |got: &[f32]| -> bool {
        got.iter()
            .zip(&want_second)
            .all(|(g, w)| (g - w).abs() <= w.abs() / 12.0 + 5e-2)
    };

    assert!(
        matches(&probe_block(4)),
        "the second block of tokens did not land in the re-grown entry (block 4)"
    );
    assert!(
        !matches(&probe_block(1)),
        "the second block of tokens landed in the STALE table entry (block 1): \
         the regrow H2D was elided"
    );
    eprintln!("shrink_then_regrow_uploads_the_new_tail: OK");
}

#[test]
#[ignore]
fn capacity_guard_follows_a_shrunk_table() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let (_pool, mut cache) = new_cache(&device);

    cache.set_block_table(&[6u32, 1, 4]).expect("set table");
    cache.set_block_table(&[6u32]).expect("shrink");
    assert_eq!(
        cache.block_table().len(),
        1,
        "the host mirror must track the shrink even though no H2D was needed"
    );

    cache
        .prepare_for_decode(BLOCK_SIZE, BLOCK_SIZE + 1)
        .unwrap();
    let kv = token_tensor(0, &device);
    let err = cache
        .write_at(0, &kv, &kv)
        .expect_err("a write past the single mapped block must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("past the"),
        "expected the capacity guard to fire, got: {msg}"
    );
    eprintln!("capacity_guard_follows_a_shrunk_table: OK ({msg})");
}
