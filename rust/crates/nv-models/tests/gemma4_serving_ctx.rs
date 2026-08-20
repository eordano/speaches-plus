#![cfg(feature = "cuda")]

mod common;
use common::argmax;
use common::ctx_tokens_from_env_default_256_8k_196k;
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::{Gemma4, Gemma4Cache, Gemma4Config, VERIFY_PREFILL_CHUNK};
use nv_models::gemma4_batch_graph::{BucketPlan, Gemma4BatchGraphFamily, SlotUpdate};
use nv_models::paged_fp8::{PagedGemma4Cache, PagedKvFp8Pool, PagedPoolConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

mod ctx_timing_common;
use common::gemma4_snapshot_dir_env_override_then_home_hub as snapshot_dir_env_override_then_home_hub;

const SERVING_BLOCK_SIZE_16_SAME_AS_BUILD_GEMMA4_BATCH_ENGINE_IN_CHAT_ENGINE_BATCH_RS: usize = 16;
const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS: usize = 16;
const GRAPHED_ARM_NON_POOL_VRAM_HEADROOM_6GIB_FOR_ACTIVATIONS_LOGITS_AND_GRAPH_MEMPOOL: usize =
    6 << 30;
const SINGLE_SEQUENCE_BUCKET_BECAUSE_THIS_GATE_TIMES_BS1_SERVING_DECODE: usize = 1;

fn free_vram_bytes_after_touching_the_current_stream(device: &Device) -> usize {
    if let Device::Cuda(d) = device {
        let _ = nv_layers::cuda_stream::current_stream(d);
        if let Ok((free, _total)) = cudarc::driver::result::mem_get_info() {
            return free;
        }
    }
    0
}

fn chunked_prefill_at_the_serving_verify_chunk_returning_secs_and_last_logits_row(
    model: &Gemma4,
    device: &Device,
    cache: &mut PagedGemma4Cache,
    depth: usize,
    vocab: usize,
) -> (f64, Vec<f32>) {
    let t0 = Instant::now();
    let mut pos = 0usize;
    let mut last_row: Vec<f32> = Vec::new();
    while pos < depth {
        let n = VERIFY_PREFILL_CHUNK.min(depth - pos);
        let ids: Vec<u32> = (0..n).map(|i| 2000 + ((pos + i) as u32 % 30000)).collect();
        let tokens = Tensor::from_vec(ids, (1usize, n), device).expect("tokens");
        let positions = Tensor::from_vec(
            (pos as i32..(pos + n) as i32).collect::<Vec<_>>(),
            n,
            device,
        )
        .expect("positions");
        let logits = model
            .forward_with_cache(&tokens, &positions, cache)
            .unwrap_or_else(|e| panic!("prefill chunk at pos {pos}: {e:#}"));
        pos += n;
        if pos >= depth {
            let v: Vec<f32> = logits
                .to_dtype(DType::F32)
                .expect("f32")
                .flatten_all()
                .expect("flatten")
                .to_vec1()
                .expect("host");
            last_row = v[(n - 1) * vocab..n * vocab].to_vec();
        }
    }
    (t0.elapsed().as_secs_f64(), last_row)
}

fn median_mean_of_sorted(mut ms: Vec<f64>) -> (f64, f64) {
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = ms[ms.len() / 2];
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    (median, mean)
}

#[test]
#[ignore = "loads the 31B; set NV_GEMMA4_SERVING_TEST=1 -- serving-path decode ms/token vs KV depth on the paged fp8 cache: graphed Gemma4BatchGraphFamily when the lanes==0 pool fits VRAM, else the serving eager forward_decode_batched on the hybrid ring pool with the reason printed; primed via chunked prefill at the serving VERIFY_PREFILL_CHUNK with W4A4 at defaults; the 196k prefill runs the default gather-attention path and takes hours -- that cost is part of the record, not a reason to skip"]
fn gemma4_cuda_serving_path_decode_ms_per_token_vs_context_depth_graphed_or_paged_batched() {
    if std::env::var("NV_GEMMA4_SERVING_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_GEMMA4_SERVING_TEST=1 to run this gate, or NV_MODELS_ALLOW_SKIP=1 to \
                 skip it on purpose; a 196k serving gate that silently reports ok would hide \
                 exactly the failure class task #106 exists to catch"
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_GEMMA4_SERVING_TEST=1 to run");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model =
        Arc::new(Gemma4::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model"));
    drop(weights);

    let depths = ctx_tokens_from_env_default_256_8k_196k();
    let max_pos = model.config().max_position_embeddings;
    let vocab = model.config().vocab_size;
    let bs = SERVING_BLOCK_SIZE_16_SAME_AS_BUILD_GEMMA4_BATCH_ENGINE_IN_CHAT_ENGINE_BATCH_RS;

    for &depth in &depths {
        let total_slots = depth
            + ctx_timing_common::WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
            + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
            + KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS;
        assert!(
            total_slots < max_pos,
            "depth {depth} + decode steps exceeds max_position_embeddings {max_pos}; a model \
             declaring {max_pos} must serve this depth, so a trip here is a config bug"
        );
        let seq_blocks = total_slots.div_ceil(bs);
        let graph_cfg = PagedPoolConfig::from_gemma4(
            model.config(),
            seq_blocks + SINGLE_SEQUENCE_BUCKET_BECAUSE_THIS_GATE_TIMES_BS1_SERVING_DECODE,
            bs,
        );
        let graph_pool_bytes = graph_cfg.pool_bytes();
        let per_token_bytes = graph_cfg.bytes_per_block() / bs;
        let free = free_vram_bytes_after_touching_the_current_stream(&device);
        let use_graphed = match std::env::var("NV_G4_SERVING_ARM").ok().as_deref() {
            Some("graphed") => true,
            Some("eager") => false,
            _ => {
                graph_pool_bytes
                    + GRAPHED_ARM_NON_POOL_VRAM_HEADROOM_6GIB_FOR_ACTIVATIONS_LOGITS_AND_GRAPH_MEMPOOL
                    <= free
            }
        };
        let table: Vec<u32> = (0..seq_blocks as u32).collect();

        if use_graphed {
            let pool = Arc::new(Mutex::new(
                PagedKvFp8Pool::new(graph_cfg, &device).expect("lanes==0 paged fp8 pool"),
            ));
            let mut cache = PagedGemma4Cache::new(pool.clone(), &device).expect("cache");
            cache.set_block_table(&table).expect("block table");
            let (prefill_s, last_row) =
                chunked_prefill_at_the_serving_verify_chunk_returning_secs_and_last_logits_row(
                    &model, &device, &mut cache, depth, vocab,
                );
            assert_eq!(
                cache.current_len(),
                depth,
                "chunked prefill must leave the paged cache at the primed depth"
            );
            let mut family = Gemma4BatchGraphFamily::new(
                model.clone(),
                pool.clone(),
                &device,
                BucketPlan::new(vec![
                    SINGLE_SEQUENCE_BUCKET_BECAUSE_THIS_GATE_TIMES_BS1_SERVING_DECODE,
                ]),
                seq_blocks as u32,
                total_slots,
            )
            .unwrap_or_else(|e| panic!("graph family at depth {depth}: {e:#}"));

            let mut tok = argmax(&last_row);
            let mut pos = depth;
            let (warmup_steps, step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
                || {
                    let rows = family
                        .step(&[SlotUpdate {
                            token: tok,
                            pos: pos as i32,
                            n_total: pos as i32 + 1,
                            block_table: table.clone(),
                            lora_slot: -1,
                        }])
                        .unwrap_or_else(|e| {
                            panic!("graphed serving decode at depth {pos}: {e:#}")
                        });
                    assert_eq!(rows.len(), 1, "bucket-1 step returns one row");
                    assert_eq!(rows[0].len(), vocab, "logit row width");
                    tok = argmax(&rows[0]);
                    pos += 1;
                },
                TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
            );
            assert!(
                family.replays() >= TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN as u64,
                "graphed basis requires replayed steps, got captures={} replays={}",
                family.captures(),
                family.replays()
            );
            let (median, mean) = median_mean_of_sorted(step_ms);
            eprintln!(
                "CTX-SCALING gemma4-cuda-serving depth={depth} basis=graphed_paged_fp8_Gemma4BatchGraphFamily_bucket1_host_rows_included captures={} replays={} median_ms_tok={median:.3} mean_ms_tok={mean:.3} tok_s={:.1} prefill_s={prefill_s:.1} prefill_chunk={} pool_gb={:.1} free_gb_at_choice={:.1} steps={} warmup_steps={warmup_steps}",
                family.captures(),
                family.replays(),
                1000.0 / median,
                VERIFY_PREFILL_CHUNK,
                graph_pool_bytes as f64 / 1e9,
                free as f64 / 1e9,
                TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
            );
        } else {
            eprintln!(
                "GRAPHED-ARM-INFEASIBLE gemma4 depth={depth}: Gemma4BatchGraphFamily accepts only \
                 a lanes==0 pool (its new() bails on hybrid KV-ring lanes), and a lanes==0 \
                 PagedPoolConfig::from_gemma4 allocates all 50 sliding layers at full depth: \
                 {per_token_bytes} B/token x {total_slots} slots = {:.1} GB KV vs {:.1} GB free. \
                 This is a pool-geometry limit, not a capture-shape limit: the family reads \
                 n_total from n_total_dev on-device and its max_ctx rounds to {} which is inside \
                 max_position_embeddings {max_pos}. Falling back to the serving eager \
                 forward_decode_batched on the hybrid ring pool, which is exactly what \
                 build_gemma4_batch_engine does when NV_KV_RING is on.",
                graph_pool_bytes as f64 / 1e9,
                free as f64 / 1e9,
                total_slots.next_power_of_two()
            );
            let hybrid_cfg =
                PagedPoolConfig::from_gemma4_hybrid(model.config(), seq_blocks, bs, 1);
            let hybrid_bytes = hybrid_cfg.pool_bytes();
            let pool = Arc::new(Mutex::new(
                PagedKvFp8Pool::new(hybrid_cfg, &device).expect("hybrid ring paged fp8 pool"),
            ));
            let mut cache = PagedGemma4Cache::new(pool.clone(), &device).expect("cache");
            cache.set_block_table(&table).expect("block table");
            let (prefill_s, last_row) =
                chunked_prefill_at_the_serving_verify_chunk_returning_secs_and_last_logits_row(
                    &model, &device, &mut cache, depth, vocab,
                );
            assert_eq!(
                cache.current_len(),
                depth,
                "chunked prefill must leave the paged cache at the primed depth"
            );
            let mut tok = argmax(&last_row);
            let mut pos = depth;
            let (warmup_steps, step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
                || {
                    let mut caches: Vec<&mut PagedGemma4Cache> = vec![&mut cache];
                    let logits = model
                        .forward_decode_batched(&[tok], &[pos], &mut caches)
                        .unwrap_or_else(|e| {
                            panic!("paged batched decode at depth {pos}: {e:#}")
                        });
                    let v: Vec<f32> = logits
                        .to_dtype(DType::F32)
                        .expect("f32")
                        .flatten_all()
                        .expect("flatten")
                        .to_vec1()
                        .expect("host");
                    tok = argmax(&v[0..vocab]);
                    pos += 1;
                },
                TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
            );
            let (median, mean) = median_mean_of_sorted(step_ms);
            eprintln!(
                "CTX-SCALING gemma4-cuda-serving depth={depth} basis=eager_paged_fp8_hybrid_ring_forward_decode_batched_host_row_included ring_decodes={} median_ms_tok={median:.3} mean_ms_tok={mean:.3} tok_s={:.1} prefill_s={prefill_s:.1} prefill_chunk={} pool_gb={:.1} graphed_pool_would_need_gb={:.1} free_gb_at_choice={:.1} steps={} warmup_steps={warmup_steps}",
                cache.ring_decodes(),
                1000.0 / median,
                VERIFY_PREFILL_CHUNK,
                hybrid_bytes as f64 / 1e9,
                graph_pool_bytes as f64 / 1e9,
                free as f64 / 1e9,
                TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
            );
        }
    }
}

const SYNTHETIC_FILL_CHUNK_256_ANY_SLACK_SIZED_CHUNK_FITS_THE_PAGED_APPEND_AND_THE_SLIDING_RING_TABLE:
    usize = 256;

fn fixed_seed_small_nonzero_values_so_the_fp8_quantize_amax_stays_finite(len: usize) -> Vec<f32> {
    let mut state = 0x9e3779b97f4a7c15u64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 0.5
        })
        .collect()
}

#[test]
#[ignore = "loads the 31B; set NV_GEMMA4_SERVING_TEST=1 -- same depth ladder as the serving gate but the paged fp8 hybrid-ring cache is filled synthetically through prepare_for_decode/write_at/advance (decode ms/token reads cache SIZE, not values), so the 196k point costs seconds of fill instead of hours of chunked prefill; times the serving eager forward_decode_batched arm, the arm that reaches 196k"]
fn gemma4_cuda_serving_path_decode_ms_per_token_vs_context_depth_synthetic_cache_fill_hybrid_ring_eager_arm(
) {
    if std::env::var("NV_GEMMA4_SERVING_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_GEMMA4_SERVING_TEST=1 to run this gate, or NV_MODELS_ALLOW_SKIP=1 to \
                 skip it on purpose; a 196k serving gate that silently reports ok would hide \
                 exactly the failure class task #106 exists to catch"
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_GEMMA4_SERVING_TEST=1 to run");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model =
        Arc::new(Gemma4::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model"));
    drop(weights);

    let depths = ctx_tokens_from_env_default_256_8k_196k();
    let max_pos = model.config().max_position_embeddings;
    let vocab = model.config().vocab_size;
    let bs = SERVING_BLOCK_SIZE_16_SAME_AS_BUILD_GEMMA4_BATCH_ENGINE_IN_CHAT_ENGINE_BATCH_RS;
    let n_layers = model.config().num_hidden_layers;
    let chunk =
        SYNTHETIC_FILL_CHUNK_256_ANY_SLACK_SIZED_CHUNK_FITS_THE_PAGED_APPEND_AND_THE_SLIDING_RING_TABLE;
    let templates: Vec<(Tensor, Tensor)> = (0..n_layers)
        .map(|li| {
            let kind = model.config().layer_kind(li);
            let hd = model.config().head_dim_for(kind);
            let n_kv = model.config().num_kv_heads_for(kind);
            let vals =
                fixed_seed_small_nonzero_values_so_the_fp8_quantize_amax_stays_finite(
                    chunk * n_kv * hd,
                );
            let k = Tensor::from_vec(vals.clone(), (1usize, chunk, n_kv, hd), &device)
                .expect("k template")
                .to_dtype(DType::BF16)
                .expect("k bf16");
            let v = Tensor::from_vec(vals, (1usize, chunk, n_kv, hd), &device)
                .expect("v template")
                .to_dtype(DType::BF16)
                .expect("v bf16");
            (k, v)
        })
        .collect();

    for &depth in &depths {
        let total_slots = depth
            + ctx_timing_common::WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
            + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
            + KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS;
        assert!(
            total_slots < max_pos,
            "depth {depth} + decode steps exceeds max_position_embeddings {max_pos}; a model \
             declaring {max_pos} must serve this depth, so a trip here is a config bug"
        );
        let seq_blocks = total_slots.div_ceil(bs);
        let hybrid_cfg = PagedPoolConfig::from_gemma4_hybrid(model.config(), seq_blocks, bs, 1);
        let hybrid_bytes = hybrid_cfg.pool_bytes();
        let pool = Arc::new(Mutex::new(
            PagedKvFp8Pool::new(hybrid_cfg, &device).expect("hybrid ring paged fp8 pool"),
        ));
        let mut cache = PagedGemma4Cache::new(pool.clone(), &device).expect("cache");
        let table: Vec<u32> = (0..seq_blocks as u32).collect();
        cache.set_block_table(&table).expect("block table");

        let fill_start = Instant::now();
        let mut pos = 0usize;
        while pos < depth {
            let n = chunk.min(depth - pos);
            cache
                .prepare_for_decode(pos, pos + n)
                .expect("prepare_for_decode");
            for (li, (k, v)) in templates.iter().enumerate() {
                if n == chunk {
                    cache.write_at(li, k, v).expect("write_at");
                } else {
                    let kn = k.narrow(1, 0, n).expect("k tail");
                    let vn = v.narrow(1, 0, n).expect("v tail");
                    cache.write_at(li, &kn, &vn).expect("write_at tail");
                }
            }
            cache.advance(n);
            pos += n;
        }
        device.synchronize().expect("sync before stopping the fill clock");
        let fill_s = fill_start.elapsed().as_secs_f64();
        assert_eq!(
            cache.current_len(),
            depth,
            "synthetic fill must leave the paged cache at the requested depth"
        );

        let mut tok = 2000u32;
        let mut pos = depth;
        let (warmup_steps, step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                let mut caches: Vec<&mut PagedGemma4Cache> = vec![&mut cache];
                let logits = model
                    .forward_decode_batched(&[tok], &[pos], &mut caches)
                    .unwrap_or_else(|e| panic!("paged batched decode at depth {pos}: {e:#}"));
                let v: Vec<f32> = logits
                    .to_dtype(DType::F32)
                    .expect("f32")
                    .flatten_all()
                    .expect("flatten")
                    .to_vec1()
                    .expect("host");
                tok = argmax(&v[0..vocab]);
                pos += 1;
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        let (median, mean) = median_mean_of_sorted(step_ms);
        eprintln!(
            "CTX-SCALING gemma4-cuda-serving-synthfill depth={depth} basis=eager_paged_fp8_hybrid_ring_forward_decode_batched_host_row_included_synthetic_kv_fill ring_decodes={} median_ms_tok={median:.3} mean_ms_tok={mean:.3} tok_s={:.1} fill_s={fill_s:.1} pool_gb={:.1} steps={} warmup_steps={warmup_steps}",
            cache.ring_decodes(),
            1000.0 / median,
            hybrid_bytes as f64 / 1e9,
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
        );
    }
}
