#![cfg(feature = "cuda")]

mod common;
use common::qwen38_snapshot_dir_env_override_then_home_hub;
use common::argmax;
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4_batch_graph::BucketPlan;
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::qwen38_batch::Qwen38BatchLanes;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;

mod ctx_timing_common;

const SOAK_GATE_ENV_NV_VRAM_SOAK_1_BECAUSE_A_SILENT_SKIP_WOULD_READ_AS_A_LEAK_FREE_PASS: &str =
    "NV_VRAM_SOAK";

const ENGINE_CYCLES_6_ENOUGH_FOR_A_PER_CYCLE_SLOPE_TO_COMPOUND_PAST_TOLERANCE: usize = 6;

const PREFILL_LENGTHS_17_256_511_1024_ODD_LENGTHS_CATCH_SHAPE_KEYED_SCRATCH_GROWTH: [usize; 4] =
    [17, 256, 511, 1024];

const DECODE_STEPS_32_PER_PREFILL_ENOUGH_TO_CAPTURE_AND_REPLAY_THE_DECODE_GRAPH: usize = 32;

const POST_DROP_FREE_TOLERANCE_MIB_256_CYCLE1_ABSORBS_ALLOCATOR_AND_CUBLAS_WARMUP: f64 = 256.0;

const LADDER_TOLERANCE_MIB_192_A_LEAKED_GRAPH_OR_SCRATCH_PER_RUNG_COMPOUNDS_PAST_THIS: f64 = 192.0;

const KV_SLOT_HEADROOM_64_BEYOND_PREFILL_PLUS_DECODE: usize = 64;

const SYNTH_FILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN35_DENSE_PREFILL_BLOCK: usize = 512;

const DEPTH_LADDER_256_8K_256_RETURNING_SHALLOW_PROVES_DEEP_STATE_IS_RELEASED: [usize; 3] =
    [256, 8 * 1024, 256];

const VERIFY_ROWS_3_THE_K2_MTP_ROUND_SHAPE_ONE_LANE_PER_ENGINE_BY_DESIGN: usize = 3;

const BUCKET_LADDER_1_2_4_2_1_NARROWING_AND_WIDENING_CHURNS_THE_KEYED_BUCKET_GRAPHS: [usize; 5] =
    [1, 2, 4, 2, 1];

const BUCKET_SWEEPS_3_CAPTURE_CHURN_MUST_BE_LINEAR_PER_SWEEP_AND_FREE_MEMORY_FLAT: usize = 3;

const BATCH_STEPS_8_MATCHES_THE_BIT_IDENTITY_SUITE: usize = 8;

const BATCH_MAX_SEQ_512_SMALL_BOUNDED_LANES: usize = 512;

fn soak_gate_or_panic() {
    if std::env::var(SOAK_GATE_ENV_NV_VRAM_SOAK_1_BECAUSE_A_SILENT_SKIP_WOULD_READ_AS_A_LEAK_FREE_PASS)
        .as_deref()
        != Ok("1")
    {
        panic!(
            "set {SOAK_GATE_ENV_NV_VRAM_SOAK_1_BECAUSE_A_SILENT_SKIP_WOULD_READ_AS_A_LEAK_FREE_PASS}=1 \
             to run this soak; it must never silently skip"
        );
    }
}

fn device_free_mib() -> f64 {
    let (free, _total) = cudarc::driver::result::mem_get_info().expect("cuMemGetInfo");
    free as f64 / (1024.0 * 1024.0)
}

fn graph_mempool_reserved_mib(ordinal: usize) -> f64 {
    use cudarc::driver::sys;
    let Ok(devh) = cudarc::driver::result::device::get(ordinal as i32) else {
        return -1.0;
    };
    let mut reserved: u64 = 0;
    unsafe {
        let _ = sys::cuDeviceGetGraphMemAttribute(
            devh,
            sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,
            &mut reserved as *mut u64 as *mut std::ffi::c_void,
        );
    }
    reserved as f64 / (1024.0 * 1024.0)
}

fn load_dense_arm(device: &Device) -> Qwen3Moe {
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, device)
        .expect("build Qwen3.8-27B dense arm");
    assert!(model.is_dense(), "quantized dense loader must yield the dense arm");
    model
}

fn synthetic_ids(len: usize) -> Vec<u32> {
    (0..len).map(|i| 2000 + (i as u32 % 30000)).collect()
}

#[test]
#[ignore = "loads the Qwen3.8-27B NVFP4 dense arm 6 times in ONE process (build graphed engine -> \
            real prefill at 17/256/511/1024 -> 32 captured decode steps each -> drop engine), \
            sampling cuMemGetInfo after every drop; set NV_VRAM_SOAK=1 -- post-drop free memory \
            must return to within 256 MiB of the cycle-1 baseline, and a second build failing \
            CUDA_ERROR_INVALID_VALUE is the graph-pool-retention hazard this soak exists to \
            reproduce (NV_GRAPH_TEARDOWN_DEBUG=1 localizes which teardown step left state behind)"]
fn engine_cycle_soak_post_drop_free_memory_returns_to_the_cycle1_baseline() {
    soak_gate_or_panic();
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let max_len = *PREFILL_LENGTHS_17_256_511_1024_ODD_LENGTHS_CATCH_SHAPE_KEYED_SCRATCH_GROWTH
        .iter()
        .max()
        .unwrap();
    let cache_slots =
        max_len + DECODE_STEPS_32_PER_PREFILL_ENOUGH_TO_CAPTURE_AND_REPLAY_THE_DECODE_GRAPH
            + KV_SLOT_HEADROOM_64_BEYOND_PREFILL_PLUS_DECODE;

    let free_before_first_build = device_free_mib();
    let mut post_drop_free: Vec<f64> = Vec::new();
    for cycle in 1..=ENGINE_CYCLES_6_ENOUGH_FOR_A_PER_CYCLE_SLOPE_TO_COMPOUND_PAST_TOLERANCE {
        let model = load_dense_arm(&device);
        let mut eng = GraphedQwen3Moe::new(model, &device, cache_slots).unwrap_or_else(|e| {
            panic!(
                "cycle {cycle}: GraphedQwen3Moe build failed -- a CUDA_ERROR_INVALID_VALUE here \
                 on cycle >= 2 is the second-engine-in-one-process retention hazard (graph pool \
                 or per-stream quant caches not released by the previous drop): {e:#}"
            )
        });
        let free_after_build = device_free_mib();
        for &len in &PREFILL_LENGTHS_17_256_511_1024_ODD_LENGTHS_CATCH_SHAPE_KEYED_SCRATCH_GROWTH {
            eng.reset().unwrap_or_else(|e| panic!("cycle {cycle} len {len} reset: {e:#}"));
            let last = eng
                .prefill(&synthetic_ids(len))
                .unwrap_or_else(|e| panic!("cycle {cycle} prefill len {len}: {e:#}"));
            eng.install_grouped_moe()
                .expect("dense branch of install_grouped_moe arms capture without a dispatch");
            let mut cur = argmax(&last);
            for step in 0..DECODE_STEPS_32_PER_PREFILL_ENOUGH_TO_CAPTURE_AND_REPLAY_THE_DECODE_GRAPH
            {
                cur = eng.forward_decode_logits(cur).unwrap_or_else(|e| {
                    panic!("cycle {cycle} len {len} decode step {step}: {e:#}")
                });
            }
            assert!(
                eng.capture_active(),
                "cycle {cycle} len {len}: decode fell back to uncaptured, so this cycle stopped \
                 exercising the graph mempool the soak exists to churn; diagnose the capture \
                 blocker printed above"
            );
        }
        let free_after_work = device_free_mib();
        drop(eng);
        device.synchronize().expect("sync after engine drop");
        let free_after_drop = device_free_mib();
        let graph_reserved = graph_mempool_reserved_mib(0);
        post_drop_free.push(free_after_drop);
        eprintln!(
            "[vram-soak] engine-cycle cycle={cycle} free_before_first_build={free_before_first_build:.1} \
             free_after_build={free_after_build:.1} free_after_work={free_after_work:.1} \
             free_after_drop={free_after_drop:.1} graph_mempool_reserved_mib={graph_reserved:.1}"
        );
    }

    let baseline = post_drop_free[0];
    for (i, &free) in post_drop_free.iter().enumerate().skip(1) {
        let cycle = i + 1;
        let held = baseline - free;
        assert!(
            held <= POST_DROP_FREE_TOLERANCE_MIB_256_CYCLE1_ABSORBS_ALLOCATOR_AND_CUBLAS_WARMUP,
            "cycle {cycle}: post-drop free {free:.1} MiB sits {held:.1} MiB below the cycle-1 \
             baseline {baseline:.1} MiB, above the {} MiB tolerance -- an engine drop is \
             retaining device memory (graph mempool, per-stream quant caches, or a KV/scratch \
             allocation with an unbounded lifetime)",
            POST_DROP_FREE_TOLERANCE_MIB_256_CYCLE1_ABSORBS_ALLOCATOR_AND_CUBLAS_WARMUP
        );
    }
    let last = *post_drop_free.last().unwrap();
    eprintln!(
        "[vram-soak] engine-cycle VERDICT cycles={} baseline_mib={baseline:.1} last_mib={last:.1} \
         slope_mib_per_cycle={:.2}",
        post_drop_free.len(),
        (baseline - last) / (post_drop_free.len() - 1) as f64
    );
}

#[test]
#[ignore = "loads the Qwen3.8-27B NVFP4 dense arm ONCE and ladders depth 256 -> 8k -> 256 twice \
            with captured decode plus a 3-row captured verify chain per rung; set NV_VRAM_SOAK=1 \
            -- captured graph node count per rung must be identical between sweeps (keyed-map \
            cardinality is bounded), sweep-2 free memory must match sweep-1 within 192 MiB, and \
            after the final reset the device graph mempool must report 0 reserved bytes \
            (invalidation actually frees, not just marks unused)"]
fn graphed_depth_ladder_and_verify_lane_keep_graph_map_cardinality_and_free_memory_stable() {
    soak_gate_or_panic();
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let model = load_dense_arm(&device);
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let max_depth = *DEPTH_LADDER_256_8K_256_RETURNING_SHALLOW_PROVES_DEEP_STATE_IS_RELEASED
        .iter()
        .max()
        .unwrap();
    let cache_slots = max_depth
        + DECODE_STEPS_32_PER_PREFILL_ENOUGH_TO_CAPTURE_AND_REPLAY_THE_DECODE_GRAPH
        + VERIFY_ROWS_3_THE_K2_MTP_ROUND_SHAPE_ONE_LANE_PER_ENGINE_BY_DESIGN
        + KV_SLOT_HEADROOM_64_BEYOND_PREFILL_PLUS_DECODE;
    let mut eng = GraphedQwen3Moe::new(model, &device, cache_slots)
        .unwrap_or_else(|e| panic!("GraphedQwen3Moe with {cache_slots} kv slots: {e:#}"));

    let chunk = SYNTH_FILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN35_DENSE_PREFILL_BLOCK;
    let vals: Vec<f32> = {
        let mut state = 0x9e3779b97f4a7c15u64;
        (0..chunk * n_kv * hd)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 0.5
            })
            .collect()
    };
    let k_template = Tensor::from_vec(vals.clone(), (1usize, chunk, n_kv, hd), &device)
        .expect("k template")
        .to_dtype(DType::BF16)
        .expect("k bf16");
    let v_template = Tensor::from_vec(vals, (1usize, chunk, n_kv, hd), &device)
        .expect("v template")
        .to_dtype(DType::BF16)
        .expect("v bf16");

    let mut per_rung: Vec<Vec<(usize, usize, f64)>> = Vec::new();
    for sweep in 1..=2usize {
        let mut rungs: Vec<(usize, usize, f64)> = Vec::new();
        for &depth in &DEPTH_LADDER_256_8K_256_RETURNING_SHALLOW_PROVES_DEEP_STATE_IS_RELEASED {
            eng.reset()
                .unwrap_or_else(|e| panic!("sweep {sweep} depth {depth} reset: {e:#}"));
            let real_prefix = 256usize.min(depth);
            let last = eng
                .prefill(&synthetic_ids(real_prefix))
                .unwrap_or_else(|e| panic!("sweep {sweep} depth {depth} prefill: {e:#}"));
            while eng.current_pos() < depth {
                let n = chunk.min(depth - eng.current_pos());
                let (kn, vn) = if n == chunk {
                    (k_template.clone(), v_template.clone())
                } else {
                    (
                        k_template.narrow(1, 0, n).expect("k tail"),
                        v_template.narrow(1, 0, n).expect("v tail"),
                    )
                };
                eng.prime_kv_depth_synthetically_for_ctx_timing_decode_reads_cache_size_not_values(
                    &kn, &vn,
                )
                .unwrap_or_else(|e| {
                    panic!("sweep {sweep} depth {depth} synthetic prime: {e:#}")
                });
            }
            eng.install_grouped_moe()
                .expect("dense branch of install_grouped_moe arms capture without a dispatch");
            let mut cur = argmax(&last);
            for step in 0..4 {
                cur = eng.forward_decode_logits(cur).unwrap_or_else(|e| {
                    panic!("sweep {sweep} depth {depth} decode step {step}: {e:#}")
                });
            }
            eng.ensure_verify_lane(VERIFY_ROWS_3_THE_K2_MTP_ROUND_SHAPE_ONE_LANE_PER_ENGINE_BY_DESIGN)
                .unwrap_or_else(|e| panic!("sweep {sweep} depth {depth} verify lane: {e:#}"));
            let chain: Vec<u32> =
                (0..VERIFY_ROWS_3_THE_K2_MTP_ROUND_SHAPE_ONE_LANE_PER_ENGINE_BY_DESIGN)
                    .map(|j| 2000 + (cur + j as u32) % 30000)
                    .collect();
            eng.forward_verify_chain(&chain)
                .unwrap_or_else(|e| panic!("sweep {sweep} depth {depth} verify chain: {e:#}"));
            eng.commit_verify_consumed(1)
                .unwrap_or_else(|e| panic!("sweep {sweep} depth {depth} commit: {e:#}"));
            for step in 0..2 {
                cur = eng.forward_decode_logits(cur).unwrap_or_else(|e| {
                    panic!("sweep {sweep} depth {depth} post-verify decode {step}: {e:#}")
                });
            }
            eng.synchronize().expect("sync before sampling");
            let nodes = eng.captured_graph_node_count();
            let free = device_free_mib();
            eprintln!(
                "[vram-soak] ladder sweep={sweep} depth={depth} captured_graph_nodes={nodes} \
                 free_mib={free:.1} graph_mempool_reserved_mib={:.1}",
                graph_mempool_reserved_mib(0)
            );
            rungs.push((depth, nodes, free));
        }
        per_rung.push(rungs);
    }

    let (s1, s2) = (&per_rung[0], &per_rung[1]);
    for (a, b) in s1.iter().zip(s2.iter()) {
        assert_eq!(
            a.1, b.1,
            "captured graph node count changed between sweeps at depth {} ({} -> {}); the graph \
             map is accumulating keys or nodes on shape repetition instead of replaying",
            a.0, a.1, b.1
        );
        let held = a.2 - b.2;
        assert!(
            held <= LADDER_TOLERANCE_MIB_192_A_LEAKED_GRAPH_OR_SCRATCH_PER_RUNG_COMPOUNDS_PAST_THIS,
            "depth {}: sweep-2 free {:.1} MiB sits {held:.1} MiB below sweep-1 {:.1} MiB, above \
             the {} MiB tolerance -- repeating the same depth ladder is consuming device memory",
            a.0,
            b.2,
            a.2,
            LADDER_TOLERANCE_MIB_192_A_LEAKED_GRAPH_OR_SCRATCH_PER_RUNG_COMPOUNDS_PAST_THIS
        );
    }

    eng.reset().expect("final reset");
    let reserved = graph_mempool_reserved_mib(0);
    assert!(
        reserved == 0.0,
        "after the final reset the device graph mempool still reports {reserved:.1} MiB \
         reserved; invalidate_graphs_synced marked graphs unused but the trim did not hand the \
         physical pages back"
    );
    let rungs_per_sweep = s1.len();
    drop(eng);
    device.synchronize().expect(
        "the first synchronize after the graphed engine drop surfaced a deferred teardown \
         error; the CtxErrDrain last field must drain what the capture-recorded event waits \
         stash during field drops",
    );
    eprintln!("[vram-soak] ladder VERDICT sweeps=2 rungs_per_sweep={rungs_per_sweep} stable");
}

#[test]
#[ignore = "loads the Qwen3.8-27B NVFP4 dense arm ONCE into Qwen38BatchLanes and churns bucket \
            occupancy 1/2/4/2/1 for 3 sweeps; set NV_VRAM_SOAK=1 -- every bucket graph is \
            captured in sweep 1 and later sweeps must only replay (captures() flat, replays() \
            growing), and free memory after each sweep must match the sweep-1 baseline within \
            192 MiB"]
fn batch_lanes_bucket_churn_keeps_captures_bounded_and_free_memory_stable() {
    soak_gate_or_panic();
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let model = load_dense_arm(&device);
    let plan = BucketPlan::new(vec![1, 2, 4]);
    let mut lanes = Qwen38BatchLanes::new(model, &device, BATCH_MAX_SEQ_512_SMALL_BOUNDED_LANES, plan)
        .expect("build batch lanes");
    let slots = lanes.lanes();

    let mut sweep_end_free: Vec<f64> = Vec::new();
    let mut sweep_end_captures: Vec<u64> = Vec::new();
    let mut sweep_end_replays: Vec<u64> = Vec::new();
    let mut sweep_end_nodes: Vec<usize> = Vec::new();
    for sweep in 1..=BUCKET_SWEEPS_3_CAPTURE_CHURN_MUST_BE_LINEAR_PER_SWEEP_AND_FREE_MEMORY_FLAT {
        for &b in &BUCKET_LADDER_1_2_4_2_1_NARROWING_AND_WIDENING_CHURNS_THE_KEYED_BUCKET_GRAPHS {
            assert!(b <= slots, "bucket {b} exceeds lane count {slots}");
            let mut cur: Vec<u32> = Vec::with_capacity(b);
            for lane in 0..b {
                let prompt = synthetic_ids(33 + 7 * lane);
                let row = lanes
                    .prefill_lane(lane, &prompt)
                    .unwrap_or_else(|e| panic!("sweep {sweep} B={b} prefill lane {lane}: {e:#}"));
                cur.push(argmax(&row));
            }
            for step in 0..BATCH_STEPS_8_MATCHES_THE_BIT_IDENTITY_SUITE {
                let feed: Vec<Option<u32>> = cur.iter().map(|&t| Some(t)).collect();
                let out = lanes
                    .step_batch(&feed)
                    .unwrap_or_else(|e| panic!("sweep {sweep} B={b} step {step}: {e:#}"));
                for lane in 0..b {
                    let row = out[lane].as_ref().unwrap_or_else(|| {
                        panic!("sweep {sweep} B={b} step {step}: active lane {lane} returned no row")
                    });
                    cur[lane] = argmax(row);
                }
            }
        }
        lanes.synchronize().expect("sync before sampling");
        let free = device_free_mib();
        eprintln!(
            "[vram-soak] buckets sweep={sweep} captures={} replays={} captured_nodes={} \
             free_mib={free:.1}",
            lanes.captures(),
            lanes.replays(),
            lanes.captured_node_count()
        );
        sweep_end_free.push(free);
        sweep_end_captures.push(lanes.captures());
        sweep_end_replays.push(lanes.replays());
        sweep_end_nodes.push(lanes.captured_node_count());
    }

    for i in 1..sweep_end_captures.len() {
        let delta = sweep_end_captures[i] - sweep_end_captures[i - 1];
        assert_eq!(
            delta,
            sweep_end_captures[0],
            "sweep {} performed {delta} graph captures where sweep 1 performed {} -- prefill_lane \
             invalidates the bucket graphs by design so capture churn is linear per sweep, and a \
             superlinear count means keys are multiplying on repetition",
            i + 1,
            sweep_end_captures[0]
        );
        assert_eq!(
            sweep_end_nodes[i],
            sweep_end_nodes[0],
            "sweep {} ended holding {} captured graph nodes where sweep 1 ended with {} -- the \
             live graph map at the same ladder point must hold the same graphs",
            i + 1,
            sweep_end_nodes[i],
            sweep_end_nodes[0]
        );
        assert!(
            sweep_end_replays[i] > sweep_end_replays[i - 1],
            "sweep {} added no graph replays; the batch step stopped using the captured graphs \
             and the capture-stability assertion above became vacuous",
            i + 1
        );
        let held = sweep_end_free[0] - sweep_end_free[i];
        assert!(
            held <= LADDER_TOLERANCE_MIB_192_A_LEAKED_GRAPH_OR_SCRATCH_PER_RUNG_COMPOUNDS_PAST_THIS,
            "sweep {}: free {:.1} MiB sits {held:.1} MiB below the sweep-1 baseline {:.1} MiB, \
             above the {} MiB tolerance -- bucket churn is consuming device memory",
            i + 1,
            sweep_end_free[i],
            sweep_end_free[0],
            LADDER_TOLERANCE_MIB_192_A_LEAKED_GRAPH_OR_SCRATCH_PER_RUNG_COMPOUNDS_PAST_THIS
        );
    }
    let stable_captures = sweep_end_captures[0];
    drop(lanes);
    device.synchronize().expect(
        "the first synchronize after the batch lanes drop surfaced a deferred teardown error; \
         the CtxErrDrain last field must drain what the capture-recorded event waits stash \
         during field drops",
    );
    eprintln!(
        "[vram-soak] buckets VERDICT sweeps={} captures={stable_captures} stable",
        BUCKET_SWEEPS_3_CAPTURE_CHURN_MUST_BE_LINEAR_PER_SWEEP_AND_FREE_MEMORY_FLAT
    );
}
