#![cfg(feature = "cuda")]

mod common;
use common::argmax_partial_cmp as argmax;
use common::envn;
use common::prompt_for;
use common::qwen36_nvfp4_snapshot_dir_env_override_then_home_hub;
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4_batch_graph::BucketPlan;
use nv_models::qwen3_5_moe::qwen36_batch::Qwen36BatchLanes;
use nv_models::qwen3_5_moe::{MoeDispatch, Qwen3Moe, Qwen3MoeConfig, Qwen3MoeKvCache};
use nv_weights::{QuantizationConfig, WeightLoader};

mod ctx_timing_common;

const STEPS_8_MATCHES_THE_QWEN38_BATCH_BIT_IDENTITY_SUITE: usize = 8;
const SLOTS_4_THE_FIRST_INCREMENT_CONCURRENCY_TARGET: usize = 4;

const GROUPED_ARM_NEAR_TIE_REL_MARGIN_1E1_A_MISMATCH_IS_ONLY_TOLERATED_WHEN_SOLO_SCORES_THE_PICKED_TOKEN_INSIDE_THIS_MARGIN_OF_ITS_BEST_SIZED_ABOVE_THE_MEASURED_8E2_ROUTE_PERTURBATION:
    f32 = 1.0e-1;

fn default_the_fp8_resident_projection_arms_because_bf16_gdn_inproj_and_lm_head_are_the_only_non_row_twin_m_row_routes_the_bisect_found(
) {
    for arm in ["NV_Q36_GDN_FP8", "NV_Q36_LM_HEAD_FP8"] {
        if std::env::var(arm).is_err() {
            std::env::set_var(arm, "1");
        }
    }
    eprintln!(
        "[q36-batch] twin basis: NV_Q36_GDN_FP8={:?} NV_Q36_LM_HEAD_FP8={:?} (both=1 is the \
         minimal byte-identity basis: fp8-resident projections route through the row-twin \
         quantized mk family, while the bf16 defaults diverge at m-rows)",
        std::env::var("NV_Q36_GDN_FP8").ok(),
        std::env::var("NV_Q36_LM_HEAD_FP8").ok()
    );
}

fn load_real_qwen36(device: &Device) -> Qwen3Moe {
    let dir = qwen36_nvfp4_snapshot_dir_env_override_then_home_hub();
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw).expect("parse qwen3.6 config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, device)
        .expect("build Qwen3.6-35B-A3B MoE");
    assert!(!model.is_dense(), "the qwen3.6 checkpoint must load the routed-MoE trunk");
    model
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn solo_stream_hook_decode_host_routed_prefill_matching_prefill_lane(
    model: &Qwen3Moe,
    hook: &dyn MoeDispatch,
    cache: &mut Qwen3MoeKvCache,
    device: &Device,
    prompt: &[u32],
    steps: usize,
) -> (Vec<u32>, Vec<Vec<f32>>) {
    cache.reset();
    let (last, rest) = prompt.split_last().expect("prompt");
    let seq = rest.len();
    let tokens = Tensor::from_vec(rest.to_vec(), (1usize, seq), device).expect("tokens");
    let positions =
        Tensor::from_vec((0..seq as i32).collect::<Vec<_>>(), seq, device).expect("pos");
    model
        .forward_with_cache_dispatched_rows(&tokens, &positions, cache, None, Some(1))
        .expect("solo host-routed prefill");
    let mut toks = Vec::with_capacity(steps);
    let mut rows = Vec::with_capacity(steps);
    let mut t = *last;
    for _ in 0..steps {
        let pos = cache.current_len();
        let tokens = Tensor::from_vec(vec![t], (1usize, 1usize), device).expect("token");
        let positions = Tensor::from_vec(vec![pos as i32], 1usize, device).expect("pos");
        let logits = model
            .forward_with_cache_dispatched(&tokens, &positions, cache, Some(hook))
            .expect("solo hooked decode step");
        let row: Vec<f32> = logits
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("host");
        let n = argmax(&row);
        rows.push(row);
        toks.push(n);
        t = n;
    }
    (toks, rows)
}

struct GateTally {
    steps: usize,
    exact_argmax: usize,
    near_tie_mismatch: usize,
    bit_diff_lanes_total: usize,
    worst_rel: f32,
}

fn compare_row(
    got: &[f32],
    solo: &[f32],
    byte_identity: bool,
    tally: &mut GateTally,
    ctx: &str,
) {
    let scale = solo.iter().fold(0f32, |a, b| a.max(b.abs())).max(1e-6);
    let worst = got
        .iter()
        .zip(solo.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    tally.worst_rel = tally.worst_rel.max(worst / scale);
    let diff = bits(got)
        .iter()
        .zip(bits(solo).iter())
        .filter(|(a, b)| a != b)
        .count();
    tally.bit_diff_lanes_total += diff;
    if byte_identity {
        assert_eq!(
            diff,
            0,
            "{ctx}: {diff} of {} logit lanes differ from the solo twin (max |delta| {worst:.3e}); \
             rowwise=moe must be byte-identical because every non-MoE route is an m-row twin at \
             B<=4 and the MoE runs per-row m=1 through the same hook",
            solo.len()
        );
    }
    let got_am = argmax(got) as usize;
    let s1 = argmax(solo) as usize;
    if got_am == s1 {
        tally.exact_argmax += 1;
    } else {
        let rel_gap_of_the_picked_token_under_the_solo_scoring = (solo[s1] - solo[got_am]) / scale;
        assert!(
            rel_gap_of_the_picked_token_under_the_solo_scoring
                < GROUPED_ARM_NEAR_TIE_REL_MARGIN_1E1_A_MISMATCH_IS_ONLY_TOLERATED_WHEN_SOLO_SCORES_THE_PICKED_TOKEN_INSIDE_THIS_MARGIN_OF_ITS_BEST_SIZED_ABOVE_THE_MEASURED_8E2_ROUTE_PERTURBATION,
            "{ctx}: batch argmax {got_am} vs solo {s1}; solo scores the picked token \
             {rel_gap_of_the_picked_token_under_the_solo_scoring:.3e} rel below its best \
             (worst rel {:.3e}); outside the argmax-class contract",
            worst / scale
        );
        tally.near_tie_mismatch += 1;
    }
    tally.steps += 1;
}

fn run_gate(byte_identity_arm: bool) {
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    default_the_fp8_resident_projection_arms_because_bf16_gdn_inproj_and_lm_head_are_the_only_non_row_twin_m_row_routes_the_bisect_found();
    if byte_identity_arm {
        std::env::set_var("NV_Q36_BATCH_ROWWISE", "moe");
    } else {
        std::env::remove_var("NV_Q36_BATCH_ROWWISE");
    }
    eprintln!(
        "[q36-batch] gate arm: NV_Q36_BATCH_ROWWISE={:?} (moe = per-row m=1 hook calls, byte-class \
         at B<=4; unset = rows=B grouped MoE, argmax-class with a solo-scored near-tie allowance)",
        std::env::var("NV_Q36_BATCH_ROWWISE").ok()
    );
    let device = Device::new_cuda_with_stream(0).expect("cuda with stream");
    let model = load_real_qwen36(&device);
    let vocab = model.config().vocab_size;
    let max_seq = envn("NV_Q36_BATCH_MAXSEQ", 128);
    let slots = SLOTS_4_THE_FIRST_INCREMENT_CONCURRENCY_TARGET;
    let steps = STEPS_8_MATCHES_THE_QWEN38_BATCH_BIT_IDENTITY_SUITE;
    let plan = BucketPlan::new(vec![1, 2, 4]);
    let mut lanes =
        Qwen36BatchLanes::new(model, &device, max_seq, plan).expect("build q36 batch lanes");
    assert_eq!(lanes.lanes(), slots);

    let prompts: Vec<Vec<u32>> = (0..slots)
        .map(|j| prompt_for(j, 9 + 3 * j, vocab))
        .collect();

    let mut solo_caches: Vec<Qwen3MoeKvCache> = (0..slots)
        .map(|_| {
            let mut c = lanes.model().new_kv_cache(max_seq).expect("solo cache");
            c.set_fused_lin_attn(true);
            c
        })
        .collect();
    let solo: Vec<(Vec<u32>, Vec<Vec<f32>>)> = (0..slots)
        .map(|j| {
            solo_stream_hook_decode_host_routed_prefill_matching_prefill_lane(
                lanes.model(),
                lanes.moe_dispatch_ref(),
                &mut solo_caches[j],
                &device,
                &prompts[j],
                steps,
            )
        })
        .collect();
    drop(solo_caches);
    let d0: std::collections::HashSet<u32> = bits(&solo[0].1[0]).into_iter().collect();
    assert!(
        d0.len() > 16,
        "solo logits are degenerate ({} distinct of {vocab}); the compare would be vacuous",
        d0.len()
    );
    for j in 1..slots {
        assert!(
            (0..steps).any(|i| bits(&solo[j].1[i]) != bits(&solo[0].1[i])),
            "slot {j}'s solo logits equal slot 0's at every step; the cross-lane compare would be vacuous"
        );
    }

    let mut tally = GateTally {
        steps: 0,
        exact_argmax: 0,
        near_tie_mismatch: 0,
        bit_diff_lanes_total: 0,
        worst_rel: 0.0,
    };
    let mut cur: Vec<u32> = Vec::with_capacity(slots);
    for (j, p) in prompts.iter().enumerate() {
        lanes
            .prefill_lane(j, &p[..p.len() - 1])
            .expect("prefill lane");
        cur.push(*p.last().unwrap());
    }
    for i in 0..steps {
        let step: Vec<Option<u32>> = cur.iter().map(|&t| Some(t)).collect();
        let out = lanes.step_batch(&step).expect("batch step");
        for j in 0..slots {
            let row = out[j].as_ref().expect("active lane row");
            compare_row(
                row,
                &solo[j].1[i],
                byte_identity_arm,
                &mut tally,
                &format!("B={slots} step {i} lane {j}"),
            );
            cur[j] = solo[j].0[i];
        }
    }
    eprintln!(
        "[q36-batch] B={slots}: {} compares, exact_argmax={} near_tie={} bit_diff_lanes={} worst_rel={:.3e} \
         (captures={} replays={} nodes={})",
        tally.steps,
        tally.exact_argmax,
        tally.near_tie_mismatch,
        tally.bit_diff_lanes_total,
        tally.worst_rel,
        lanes.captures(),
        lanes.replays(),
        lanes.captured_node_count()
    );

    let narrow = slots - 1;
    let mut cur: Vec<u32> = Vec::with_capacity(narrow);
    for j in 0..narrow {
        let p = &prompts[j];
        lanes
            .prefill_lane(j, &p[..p.len() - 1])
            .expect("re-prefill lane");
        cur.push(*p.last().unwrap());
    }
    for i in 0..steps {
        let mut step: Vec<Option<u32>> = cur.iter().map(|&t| Some(t)).collect();
        step.push(None);
        let out = lanes.step_batch(&step).expect("narrow batch step");
        assert!(out[narrow].is_none(), "padded lane must return no row");
        for j in 0..narrow {
            let row = out[j].as_ref().expect("active lane row");
            compare_row(
                row,
                &solo[j].1[i],
                byte_identity_arm,
                &mut tally,
                &format!("B={narrow}-on-{slots} step {i} lane {j}"),
            );
            cur[j] = solo[j].0[i];
        }
    }
    eprintln!("[q36-batch] B={narrow} padded on a {slots}-row graph: contract held");

    lanes
        .prefill_lane(0, &prompts[0][..prompts[0].len() - 1])
        .expect("re-prefill lane 0");
    let mut t = *prompts[0].last().unwrap();
    for i in 0..steps {
        let out = lanes.step_batch(&[Some(t)]).expect("b1 batch step");
        let row = out[0].as_ref().expect("lane 0 row");
        let diff = bits(row)
            .iter()
            .zip(bits(&solo[0].1[i]).iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diff, 0,
            "B=1 step {i}: {diff} logit lanes differ from solo; the B=1 bucket shares every \
             m=1 route with the solo twin and must be bit-identical on both gate arms"
        );
        t = solo[0].0[i];
    }
    eprintln!("[q36-batch] B=1 bucket: bit-identical to solo (m=1 routes shared, both arms)");
    assert!(
        lanes.captures() >= 2,
        "expected one capture per touched bucket; captures={}",
        lanes.captures()
    );
    assert!(
        lanes.replays() > 0,
        "batch steps never replayed a captured graph; the graph arm is vacuous"
    );
    let mismatch_share = tally.near_tie_mismatch as f64 / tally.steps.max(1) as f64;
    eprintln!(
        "[q36-batch] GATE arm={} steps={} exact_argmax_share={:.3} near_tie_share={:.3} worst_rel={:.3e}",
        if byte_identity_arm { "rowwise_moe_byte_identity" } else { "grouped_rows_argmax_class" },
        tally.steps,
        tally.exact_argmax as f64 / tally.steps.max(1) as f64,
        mismatch_share,
        tally.worst_rel
    );
    std::env::remove_var("NV_Q36_BATCH_ROWWISE");
}

#[test]
#[ignore = "loads the Qwen3.6-35B NVFP4 checkpoint; set NV_Q36_BATCH_TEST=1 -- rowwise=moe byte-identity gate: every lane's logits equal its solo twin bit-for-bit"]
fn real_qwen36_moe_batch_decode_rowwise_moe_byte_identity() {
    if std::env::var("NV_Q36_BATCH_TEST").as_deref() != Ok("1") {
        panic!("set NV_Q36_BATCH_TEST=1 to run this GPU test (it must never silently skip)");
    }
    run_gate(true);
}

#[test]
#[ignore = "loads the Qwen3.6-35B NVFP4 checkpoint; set NV_Q36_BATCH_TEST=1 -- serving-default rows=B grouped MoE: argmax-class vs solo with a documented top2 near-tie allowance"]
fn real_qwen36_moe_batch_decode_grouped_rows_argmax_class() {
    if std::env::var("NV_Q36_BATCH_TEST").as_deref() != Ok("1") {
        panic!("set NV_Q36_BATCH_TEST=1 to run this GPU test (it must never silently skip)");
    }
    run_gate(false);
}

#[test]
#[ignore = "loads the Qwen3.6-35B NVFP4 checkpoint; set NV_Q36_BRATE=1 -- B-ladder: aggregate tok/s at the NV_Q36_BATCH_SIZES buckets, depth via NV_CTX_TOKENS (default 256), synthetic KV prime"]
fn real_qwen36_moe_batch_decode_rate_ladder() {
    if std::env::var("NV_Q36_BRATE").as_deref() != Ok("1") {
        panic!("set NV_Q36_BRATE=1 to run this GPU test (it must never silently skip)");
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda with stream");
    let model = load_real_qwen36(&device);
    let vocab = model.config().vocab_size;
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let depth = {
        let s = std::env::var("NV_CTX_TOKENS").unwrap_or_else(|_| "256".into());
        let s = s.trim();
        match s.strip_suffix('k') {
            Some(n) => n.parse::<usize>().expect("NV_CTX_TOKENS") * 1024,
            None => s.parse::<usize>().expect("NV_CTX_TOKENS"),
        }
    };
    let steps = envn("NV_Q36_RATE_STEPS", 32);
    let reps = envn("NV_Q36_RATE_REPS", 3);
    let plan = nv_models::qwen3_5_moe::qwen36_batch::q36_batch_bucket_plan_env_nv_q36_batch_sizes();
    let sizes: Vec<usize> = plan.sizes().to_vec();
    let max_seq = depth + sizes.len() * (reps + 1) * steps + 64;
    let free_before = cudarc::driver::result::mem_get_info().map(|(f, _)| f).unwrap_or(0);
    let mut lanes =
        Qwen36BatchLanes::new(model, &device, max_seq, plan).expect("build q36 batch lanes");
    let free_after = cudarc::driver::result::mem_get_info().map(|(f, _)| f).unwrap_or(0);
    eprintln!(
        "[q36-batch-rate] {} lanes of max_seq {} take {:.2} GiB total ({:.3} GiB per lane, KV + GDN \
         state + engine buffers)",
        lanes.lanes(),
        max_seq,
        free_before.saturating_sub(free_after) as f64 / (1u64 << 30) as f64,
        free_before.saturating_sub(free_after) as f64
            / (1u64 << 30) as f64
            / lanes.lanes() as f64,
    );

    let prefill_len = 23usize;
    let chunk = 512usize.min(depth.saturating_sub(prefill_len)).max(1);
    let mut state = 0x9e3779b97f4a7c15u64;
    let vals: Vec<f32> = (0..chunk * n_kv * hd)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 0.5
        })
        .collect();
    let k_t = Tensor::from_vec(vals.clone(), (1usize, chunk, n_kv, hd), &device)
        .expect("k template")
        .to_dtype(DType::BF16)
        .expect("bf16");
    let v_t = Tensor::from_vec(vals, (1usize, chunk, n_kv, hd), &device)
        .expect("v template")
        .to_dtype(DType::BF16)
        .expect("bf16");

    let prompt = prompt_for(0, prefill_len + 1, vocab);
    for lane in 0..lanes.lanes() {
        lanes
            .prefill_lane(lane, &prompt[..prompt.len() - 1])
            .expect("prefill lane");
        while lanes.lane_pos(lane) + chunk <= depth {
            lanes
                .prime_lane_kv_depth_synthetically_for_ctx_timing_decode_reads_cache_size_not_values(
                    lane, &k_t, &v_t,
                )
                .expect("prime lane");
        }
    }
    eprintln!(
        "[q36-batch-rate] primed {} lanes to pos {} (target depth {depth})",
        lanes.lanes(),
        lanes.lane_pos(0)
    );

    let seed_tok = *prompt.last().unwrap();
    let mut results: Vec<(usize, f64)> = Vec::new();
    for &bsz in &sizes {
        let mut cur: Vec<Option<u32>> = (0..bsz).map(|_| Some(seed_tok)).collect();
        let mut ms_acc: Vec<f64> = Vec::new();
        for r in 0..=reps {
            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                let out = lanes
                    .step_batch_greedy_tokens_on_device_byte_exact_vs_host_sampling_by_the_widening_and_tie_break_constants(&cur)
                    .expect("rate step");
                for (j, o) in out.iter().enumerate() {
                    cur[j] = Some(o.expect("active lane token"));
                }
            }
            lanes.synchronize().expect("sync");
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
            if r == 0 {
                eprintln!("[q36-batch-rate] B={bsz} warmup rep discarded: {ms:.2} ms/step");
                continue;
            }
            ms_acc.push(ms);
        }
        let mean = ms_acc.iter().sum::<f64>() / ms_acc.len() as f64;
        eprintln!("[q36-batch-rate] B={bsz} moe events ladder (last replay of this bucket):");
        lanes
            .moe_dispatch_ref()
            .dump_phase_ladder_after_replay_when_nv_q36_moe_events_1();
        results.push((bsz, mean));
        eprintln!(
            "[q36-batch-rate] B={bsz} depth={} step {mean:.2} ms = {:.2} ms/lane-tok | aggregate {:.1} tok/s (captures={} replays={})",
            lanes.lane_pos(0),
            mean / bsz as f64,
            bsz as f64 * 1000.0 / mean,
            lanes.captures(),
            lanes.replays()
        );
    }
    let b1 = results
        .iter()
        .find(|(b, _)| *b == 1)
        .map(|(_, m)| *m)
        .unwrap_or(results[0].1);
    for (bsz, ms) in &results {
        eprintln!(
            "[q36-batch-rate] SUMMARY B={bsz}: {:.2} ms/step, {:.2}x aggregate vs B=1, {:.2}x per-lane latency, \
             basis: RedHatAI/Qwen3.6-35B-A3B-NVFP4, synthetic prime depth {}, {} timed steps x {} reps",
            ms,
            b1 * *bsz as f64 / ms,
            ms / b1,
            lanes.lane_pos(0),
            steps,
            reps
        );
    }
}
