#![cfg(feature = "cuda")]

use candle_core::{DType, Device};
use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::paged_fp8::{DeriveVPlan, PagedKvFp8Pool, PagedPoolConfig, DERIVE_V_ENV};
use std::path::Path;

const BLOCK_SIZE: usize = 16;

const CTX_SLOTS: usize = 262144;

const LANES: usize = 1;

const FULL_KV_HEADS: usize = 4;
const FULL_HEAD_DIM: usize = 512;

const SETTLE_SLACK: usize = 256 << 20;

const MIN_SAVED_BYTES: usize = 4900 * (1 << 20);

fn free_mem() -> usize {
    nv_layers::cudarc::driver::result::mem_get_info()
        .expect("mem_get_info")
        .0
}

fn gib(b: usize) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn k_norm_scalars(dir: &str, cfg: &Gemma4Config) -> Vec<Option<f32>> {
    let weights =
        nv_weights::WeightLoader::open_dir(Path::new(dir), &Device::Cpu).expect("open checkpoint");
    cfg.layer_types
        .iter()
        .enumerate()
        .map(|(i, kind)| {
            if *kind != LayerType::FullAttention {
                return None;
            }
            let name = format!("model.language_model.layers.{i}.self_attn.k_norm.weight");
            let w: Vec<f32> = weights
                .get(&name, DType::BF16)
                .unwrap_or_else(|e| panic!("{name}: {e}"))
                .flatten_all()
                .and_then(|t| t.to_dtype(DType::F32))
                .and_then(|t| t.to_vec1())
                .unwrap();
            let lo = w.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            (lo == hi).then_some(lo)
        })
        .collect()
}

#[test]
#[ignore]
fn deriving_v_stops_allocating_the_v_slab_on_full_layers() {
    if std::env::var("NV_KV_DERIVE_V_TEST").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_KV_DERIVE_V_TEST=1");
    }
    let dir = std::env::var("NV_CHAT_MODEL_DIR")
        .expect("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: NV_CHAT_MODEL_DIR unset");
    let cfg = Gemma4Config::from_hf_json_file(Path::new(&format!("{dir}/config.json")))
        .expect("gemma4 config");

    let n_full = cfg
        .layer_types
        .iter()
        .filter(|k| **k == LayerType::FullAttention)
        .count();
    let expected_v_bytes = n_full * CTX_SLOTS * (FULL_KV_HEADS * FULL_HEAD_DIM + FULL_KV_HEADS * 4);

    let device = Device::new_cuda(0).expect("cuda device 0");
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);

    let full_blocks = CTX_SLOTS / BLOCK_SIZE;
    let pool_cfg = PagedPoolConfig::from_gemma4_hybrid(&cfg, full_blocks, BLOCK_SIZE, LANES);
    let w_k = k_norm_scalars(&dir, &cfg);
    let plan = DeriveVPlan::new(&cfg, &pool_cfg, &w_k).expect("derive plan");
    eprintln!(
        "[derive-vram] checkpoint {dir}\n[derive-vram] {} layers, {n_full} full, \
         block_size {BLOCK_SIZE}, full_blocks {full_blocks} ({CTX_SLOTS} slots), \
         rope_angles {}, plan covers {} layer(s)",
        cfg.layer_types.len(),
        plan.rope_angles(),
        plan.layer_count()
    );
    assert_eq!(
        plan.layer_count(),
        n_full,
        "the capability predicate covers {} of the {n_full} full-attention layers; \
         a pool that derives on none of them saves nothing and this test would \
         still see whatever the allocator felt like",
        plan.layer_count()
    );

    {
        let warm = stream.alloc_zeros::<u8>(1 << 20).expect("warm-up alloc");
        stream.synchronize().unwrap();
        drop(warm);
        stream.synchronize().unwrap();
    }

    let measure = |on: bool| -> (usize, usize, usize) {
        std::env::set_var(DERIVE_V_ENV, if on { "1" } else { "0" });
        let free_before = free_mem();
        let pool = PagedKvFp8Pool::new_derive_v(pool_cfg.clone(), &device, &plan)
            .unwrap_or_else(|e| panic!("pool (derive={on}): {e}"));
        stream.synchronize().unwrap();
        let used = free_before.saturating_sub(free_mem());
        let out = (used, pool.derive_layers(), pool.pool_bytes());
        drop(pool);
        stream.synchronize().unwrap();
        out
    };

    let base = free_mem();
    let (used_off, layers_off, bytes_off) = measure(false);
    let settled = free_mem();
    let (used_on, layers_on, bytes_on) = measure(true);
    std::env::remove_var(DERIVE_V_ENV);

    eprintln!(
        "[derive-vram] OFF: mem_get_info {:.3} GiB, pool_bytes {:.3} GiB, derive layers {layers_off}\n\
         [derive-vram] ON : mem_get_info {:.3} GiB, pool_bytes {:.3} GiB, derive layers {layers_on}\n\
         [derive-vram] saved {:.3} GiB measured, {:.3} GiB accounted",
        gib(used_off),
        gib(bytes_off),
        gib(used_on),
        gib(bytes_on),
        gib(used_off.saturating_sub(used_on)),
        gib(bytes_off.saturating_sub(bytes_on)),
    );

    assert!(
        base.saturating_sub(settled) < SETTLE_SLACK,
        "the OFF pool did not come back: {:.3} GiB still held after drop+sync, so the \
         ON measurement is reading a warm allocator, not a smaller pool",
        gib(base.saturating_sub(settled))
    );
    assert_eq!(
        layers_off, 0,
        "the pool derived on {layers_off} layer(s) with {DERIVE_V_ENV} off; the default \
         is supposed to be the stored-V path"
    );
    assert_eq!(
        layers_on, n_full,
        "expected the {n_full} full-attention layers to derive, got {layers_on}"
    );
    assert_eq!(
        bytes_off - bytes_on,
        expected_v_bytes,
        "the pool's own accounting does not match the {n_full}-layer V slab"
    );
    assert!(
        used_off.saturating_sub(used_on) >= MIN_SAVED_BYTES,
        "mem_get_info says the ON pool only gives back {:.3} GiB (off {:.3}, on {:.3}); \
         the V slab is {:.3} GiB, so it is still being allocated",
        gib(used_off.saturating_sub(used_on)),
        gib(used_off),
        gib(used_on),
        gib(expected_v_bytes)
    );
}
