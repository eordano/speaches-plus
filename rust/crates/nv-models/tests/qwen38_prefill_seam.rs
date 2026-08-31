#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::CudaEvent;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::time::Instant;

mod common;
mod ctx_timing_common;
mod hub_snapshot;
use common::qwen38_snapshot_dir_env_override_then_home_hub;

const SEAM_DEPTHS_DEFAULT_8192_32768_MATCHING_THE_DEPTH_HARNESS_ROWS: &[usize] = &[8192, 32768];
const KV_HEADROOM_8_ROWS_KEEPS_THE_CACHE_STRICTLY_LARGER_THAN_THE_PROMPT: usize = 8;
const WARMUP_PREFILLS_1_BECAUSE_THE_FIRST_CHUNK_PAYS_ONE_TIME_MODULE_AND_WORKSPACE_INIT: usize = 1;

fn depths_from_env() -> Vec<usize> {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) => v
            .split(',')
            .map(|s| {
                let s = s.trim();
                let (num, mult) = match s.strip_suffix(['k', 'K']) {
                    Some(n) => (n, 1024usize),
                    None => (s, 1usize),
                };
                num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
            })
            .collect(),
        Err(_) => SEAM_DEPTHS_DEFAULT_8192_32768_MATCHING_THE_DEPTH_HARNESS_ROWS.to_vec(),
    }
}

fn gate_env_or_allow_skip() -> bool {
    if std::env::var("NV_QWEN38_SERVING_TEST").ok().as_deref() == Some("1") {
        return true;
    }
    if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
        panic!(
            "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_QWEN38_SERVING_TEST=1, or \
             NV_MODELS_ALLOW_SKIP=1 to skip on purpose"
        );
    }
    eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_QWEN38_SERVING_TEST=1 to run");
    false
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 dense arm; set NV_QWEN38_SERVING_TEST=1 -- \
            chunk-seam stall instrument for the serving chunked prefill: CUDA events bracket \
            every chunk's dispatch on the model stream, so seam_gap = GPU-timeline idle between \
            chunk N's last enqueued op and chunk N+1's loop-top marker, and the drained-boundary \
            count says whether the host ever let the pipeline run dry; depths via NV_CTX_TOKENS \
            (default 8192,32768) -- a seam total under 2 percent of wall is a clean bill for the \
            vLLM v0.28 sync-elimination disease class"]
fn qwen38_cuda_chunked_prefill_seam_gap_between_chunks_vs_wall() {
    if !gate_env_or_allow_skip() {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("build Qwen3.8-27B on the eager cuda dense arm");
    drop(weights);
    assert!(model.is_dense(), "from_loader_dense_quantized must yield the dense arm");

    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let ctx = stream.context().clone();
    let timing_event = || -> CudaEvent {
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .expect("timing event")
    };

    for depth in depths_from_env() {
        let max_seq = depth + KV_HEADROOM_8_ROWS_KEEPS_THE_CACHE_STRICTLY_LARGER_THAN_THE_PROMPT;
        let ids: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
        let chunk = nv_models::qwen3_5_moe::q38_prefill_chunk_tokens_from_env()
            .filter(|&c| depth > c)
            .unwrap_or(depth);
        for _ in 0..WARMUP_PREFILLS_1_BECAUSE_THE_FIRST_CHUNK_PAYS_ONE_TIME_MODULE_AND_WORKSPACE_INIT
        {
            let mut cache = model.new_kv_cache(max_seq).expect("kv cache");
            let mut fed = 0usize;
            while fed < depth {
                let n = chunk.min(depth - fed);
                let start = cache.current_len();
                let tokens = Tensor::from_vec(ids[fed..fed + n].to_vec(), (1usize, n), &device)
                    .expect("tokens");
                let pos = Tensor::from_vec(
                    (start as i32..(start + n) as i32).collect::<Vec<i32>>(),
                    n,
                    &device,
                )
                .expect("positions");
                let _ = model
                    .forward_with_cache_dispatched_rows(&tokens, &pos, &mut cache, None, Some(1))
                    .expect("warmup prefill chunk");
                fed += n;
            }
            device.synchronize().expect("warmup sync");
        }

        let mut cache = model.new_kv_cache(max_seq).expect("kv cache");
        device.synchronize().expect("sync before starting the clock");
        let mut heads: Vec<CudaEvent> = Vec::with_capacity(depth.div_ceil(chunk));
        let mut tails: Vec<CudaEvent> = Vec::with_capacity(depth.div_ceil(chunk));
        let mut prev_tail_still_running_at_loop_top = 0usize;
        let mut drained_boundary_host_prep_ms = 0f64;
        let t0 = Instant::now();
        let mut fed = 0usize;
        let mut last_logits = None;
        while fed < depth {
            let boundary_drained = match tails.last() {
                Some(prev_tail) => {
                    if prev_tail.is_complete() {
                        true
                    } else {
                        prev_tail_still_running_at_loop_top += 1;
                        false
                    }
                }
                None => false,
            };
            let t_prep = Instant::now();
            let head = timing_event();
            head.record(&stream).expect("record head");
            let n = chunk.min(depth - fed);
            let start = cache.current_len();
            let tokens =
                Tensor::from_vec(ids[fed..fed + n].to_vec(), (1usize, n), &device).expect("tokens");
            let pos = Tensor::from_vec(
                (start as i32..(start + n) as i32).collect::<Vec<i32>>(),
                n,
                &device,
            )
            .expect("positions");
            if boundary_drained {
                drained_boundary_host_prep_ms += t_prep.elapsed().as_secs_f64() * 1e3;
            }
            let logits = model
                .forward_with_cache_dispatched_rows(&tokens, &pos, &mut cache, None, Some(1))
                .unwrap_or_else(|e| panic!("prefill chunk at pos {fed} depth {depth}: {e:#}"));
            let tail = timing_event();
            tail.record(&stream).expect("record tail");
            heads.push(head);
            tails.push(tail);
            last_logits = Some(logits);
            fed += n;
        }
        let host_enqueue_ms = t0.elapsed().as_secs_f64() * 1e3;
        device.synchronize().expect("sync before stopping the clock");
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        let n_chunks = heads.len();
        assert_eq!(cache.current_len(), depth, "prefill must commit all {depth} tokens");
        let last_row: Vec<f32> = last_logits
            .expect("at least one chunk ran")
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to host");
        assert_eq!(
            last_row.iter().filter(|v| v.is_finite()).count(),
            last_row.len(),
            "prefill produced non-finite logits at depth {depth}"
        );

        let mut seam_gap_ms_total = 0f64;
        let mut seam_gap_ms_max = 0f64;
        for i in 0..n_chunks - 1 {
            let gap = tails[i].elapsed_ms(&heads[i + 1]).expect("seam elapsed") as f64;
            let gap = gap.max(0.0);
            seam_gap_ms_total += gap;
            seam_gap_ms_max = seam_gap_ms_max.max(gap);
        }
        let mut busy_ms_total = 0f64;
        for i in 0..n_chunks {
            busy_ms_total += heads[i].elapsed_ms(&tails[i]).expect("busy elapsed") as f64;
        }
        let span_ms = heads[0].elapsed_ms(&tails[n_chunks - 1]).expect("span") as f64;
        let seam_pct = 100.0 * seam_gap_ms_total / wall_ms;
        let gpu_lead_ms = wall_ms - host_enqueue_ms;
        println!(
            "QWEN38-SEAM depth={depth} chunks={n_chunks} chunk={chunk} wall_ms={wall_ms:.1} \
             span_ms={span_ms:.1} busy_ms_total={busy_ms_total:.1} \
             seam_gap_ms_total={seam_gap_ms_total:.3} seam_gap_ms_max={seam_gap_ms_max:.3} \
             seam_pct={seam_pct:.3} host_enqueue_ms={host_enqueue_ms:.1} \
             gpu_lead_ms={gpu_lead_ms:.1} \
             boundaries_with_gpu_still_busy={prev_tail_still_running_at_loop_top}/{} \
             drained_boundary_host_prep_ms={drained_boundary_host_prep_ms:.3}",
            n_chunks - 1
        );
    }
}
