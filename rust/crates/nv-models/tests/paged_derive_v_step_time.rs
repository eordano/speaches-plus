#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::{Gemma4, Gemma4Cache, Gemma4Config};
use nv_models::paged_fp8::{
    DeriveVPlan, PagedGemma4Cache, PagedKvFp8Pool, PagedPoolConfig, DERIVE_V_ENV,
};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const BLOCK_SIZE: usize = 16;
const SEED_TOKENS: usize = 16;
const WARMUP: usize = 5;
const TIMED: usize = 30;
const CONTEXTS: [usize; 2] = [32768, 131072];

fn step(model: &Gemma4, cache: &mut PagedGemma4Cache, tok: u32) -> u32 {
    let position = Gemma4Cache::current_len(cache);
    let mut caches: Vec<&mut PagedGemma4Cache> = vec![cache];
    let logits = model
        .forward_decode_batched(&[tok], &[position], &mut caches)
        .expect("decode step");
    let v: Vec<f32> = logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let mut best = 0u32;
    let mut bestv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bestv {
            bestv = x;
            best = i as u32;
        }
    }
    best
}

fn time_at(
    model: &Gemma4,
    device: &Device,
    cfg: &Gemma4Config,
    ctx: usize,
    derive: bool,
) -> (f64, usize, u64) {
    let full_blocks = (ctx + WARMUP + TIMED + BLOCK_SIZE).div_ceil(BLOCK_SIZE);
    let pool_cfg = PagedPoolConfig::from_gemma4_hybrid(cfg, full_blocks, BLOCK_SIZE, 1);
    let plan = DeriveVPlan::from_model(model, &pool_cfg).expect("plan");

    std::env::set_var(DERIVE_V_ENV, if derive { "1" } else { "0" });
    let pool = Arc::new(Mutex::new(
        PagedKvFp8Pool::new_derive_v(pool_cfg.clone(), device, &plan).expect("pool"),
    ));
    let derive_layers = pool.lock().unwrap().derive_layers();
    let table: Vec<u32> = (0..pool_cfg.num_blocks as u32).collect();
    let mut cache = PagedGemma4Cache::new(pool.clone(), device).expect("cache");
    cache.set_block_table(&table).expect("table");

    let ids: Vec<u32> = (0..SEED_TOKENS as u32).map(|i| 2 + i).collect();
    let tokens = Tensor::from_vec(ids.clone(), (1usize, SEED_TOKENS), device).unwrap();
    let pos: Vec<i32> = (0..SEED_TOKENS as i32).collect();
    let pos = Tensor::from_vec(pos, SEED_TOKENS, device).unwrap();
    model
        .forward_with_cache_last(&tokens, &pos, &mut cache)
        .expect("seed prefill");
    Gemma4Cache::advance(&mut cache, ctx - SEED_TOKENS);

    let mut tok = 2u32;
    for _ in 0..WARMUP {
        tok = step(model, &mut cache, tok);
    }
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    nv_layers::cuda_stream::current_stream(&dev)
        .synchronize()
        .unwrap();
    let t0 = Instant::now();
    for _ in 0..TIMED {
        tok = step(model, &mut cache, tok);
    }
    nv_layers::cuda_stream::current_stream(&dev)
        .synchronize()
        .unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / TIMED as f64;
    let dispatches = pool.lock().unwrap().derive_dispatches();
    let _ = tok;
    (us, derive_layers, dispatches)
}

#[test]
#[ignore]
fn what_reconstructing_v_costs_per_decode_step() {
    if std::env::var("NV_KV_DERIVE_V_TIME").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_KV_DERIVE_V_TIME=1");
    }
    let dir = std::env::var("NV_CHAT_MODEL_DIR")
        .expect("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: NV_CHAT_MODEL_DIR unset");
    let dir = Path::new(&dir);
    let device = Device::new_cuda(0).expect("cuda device 0");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("cfg");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("qcfg");
    let weights = WeightLoader::open_dir(dir, &device).expect("weights");
    let model =
        Gemma4::from_loader_quantized(cfg.clone(), &weights, &qcfg, &device).expect("model");

    eprintln!(
        "[derive-time] {} | {TIMED} timed steps after {WARMUP} warmup, block_size {BLOCK_SIZE}, \
         n_q {} n_kv(full) {} head_dim(full) {}",
        dir.display(),
        cfg.num_attention_heads,
        cfg.num_kv_heads_for(nv_models::gemma4::LayerType::FullAttention),
        cfg.head_dim_for(nv_models::gemma4::LayerType::FullAttention),
    );
    for ctx in CONTEXTS {
        let (us_off, layers_off, disp_off) = time_at(&model, &device, &cfg, ctx, false);
        let (us_on, layers_on, disp_on) = time_at(&model, &device, &cfg, ctx, true);
        eprintln!(
            "[derive-time] ctx {ctx:7}: stored-V {us_off:9.1} us/step (layers {layers_off}, \
             dispatch {disp_off}) | derived-V {us_on:9.1} us/step (layers {layers_on}, \
             dispatch {disp_on}) | {:.3}x",
            us_on / us_off
        );
        assert!(
            layers_off == 0 && disp_off == 0,
            "the stored-V run used the derive path"
        );
        assert!(
            layers_on > 0 && disp_on >= TIMED as u64,
            "the derived-V run dispatched {disp_on} times over {layers_on} layer(s); this is \
             not a measurement of the derive path"
        );
    }
    std::env::remove_var(DERIVE_V_ENV);
}
