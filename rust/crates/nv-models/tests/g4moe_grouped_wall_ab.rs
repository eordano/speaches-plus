#![cfg(feature = "wgpu")]

mod common;
use common::snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights;
use nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu;

const DEPENDENT_WGPU_DISPATCH_US_MEASURED_ON_THIS_ADAPTER_NO_CUDA_GRAPH_EQUIVALENT: f64 = 2.05;

const LADDER_ARMS_DEFAULT_THE_TOP_BIT_IDENTICAL_RUNG_QKVFOLD_AND_TAILFOLD_ARE_BANKED_NEGATIVE:
    &str = "GROUPED_GEMV,FOLD_CHAIN,RPNORM,RPFUSE,TOPK_NX2";

const EVERY_LADDER_ARM_THE_BUILDER_READS_SO_ARM_A_CAN_PIN_THEM_ALL_OFF: &[&str] = &[
    "NV_G4MOE_GROUPED_GEMV",
    "NV_G4MOE_FOLD_CHAIN",
    "NV_G4MOE_RPNORM",
    "NV_G4MOE_RPFUSE",
    "NV_G4MOE_TOPK_NX2",
    "NV_G4MOE_TOPK_FOLD",
    "NV_G4MOE_TAILFOLD",
    "NV_G4MOE_QKVFOLD",
    "NV_G4MOE_ROUTER_CHAIN",
    "NV_G4MOE_GEMV_WIDE",
    "NV_G4MOE_HEAD_ONE_SHARD",
    "NV_G4MOE_LMHEAD_TIGHT_GRID",
];

const BASE_LADDER_ARM_A_ALSO_CARRIES_SO_A_NON_LADDER_ARM_CAN_BE_PRICED_ON_TOP_OF_THE_FUSED_STACK:
    &str = "";

const ARMS_THAT_RESHAPE_A_GRID_NOT_THE_PASS_LIST_SO_AN_EQUAL_PASS_COUNT_IS_THE_WHOLE_POINT:
    &[&str] = &["NV_G4MOE_LMHEAD_TIGHT_GRID"];

const RECORD_STACK_ARMS_BOTH_INSTANCES_SHARE_SO_THE_ONLY_DIFFERENCE_IS_THE_LADDER: &[&str] = &[
    "NV_G4MOE_GEMV_SGLAT",
    "NV_G4MOE_GU_FUSEDX",
    "NV_G4MOE_DOWN_EXPAR",
    "NV_G4MOE_W4_XV4",
];

const KERNEL_ARMS_INERT_OFF_THE_GROUPED_PATH_SO_PINNING_THEM_OFF_LEAVES_ARM_A_UNCHANGED: &[&str] =
    &["NV_G4MOE_GU_FUSEDX", "NV_G4MOE_DOWN_EXPAR", "NV_G4MOE_W4_XV4"];

fn record_arms_shared_by_both_instances() -> Vec<String> {
    match std::env::var("NV_G4MOE_AB_RECORD_ARMS") {
        Err(_) => RECORD_STACK_ARMS_BOTH_INSTANCES_SHARE_SO_THE_ONLY_DIFFERENCE_IS_THE_LADDER
            .iter()
            .map(|s| s.to_string())
            .collect(),
        Ok(v) => v
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| format!("NV_G4MOE_{}", s.trim()))
            .collect(),
    }
}

const DEPTH_DEFAULT_256_IS_THE_RUNG_THE_10_96_TO_11_25_MS_STEP_WAS_MEASURED_AT: usize = 256;
const ROTATIONS_DEFAULT_6_SO_EACH_ORDER_AB_AND_BA_IS_RUN_THREE_TIMES: usize = 6;
const TIMED_STEPS_PER_ROTATION_DEFAULT_24: usize = 24;
const DISCARDED_STEPS_AT_THE_HEAD_OF_EACH_ROTATION_SO_THE_HANDOFF_IS_NOT_TIMED: usize = 4;
const PLATEAU_WARMUP_STEPS_CEILING_THE_LOOP_ALSO_STOPS_ON_AN_8_SECOND_BUDGET: usize = 200;

const SUBMIT_CHUNK_IS_NOT_A_G4MOE_ARM_THIS_SUITE_CAN_PIN_YET_IS_WORTH_20_PCT_SO_EVERY_RUN_MUST_STATE_IT:
    &str = "NV_WGPU_SUBMIT_CHUNK";

const SYNTHFILL_ENV_STEP_PRIMING_196608_TOKENS_ON_TWO_INSTANCES_IS_HOURS_WHERE_A_FILL_IS_SECONDS:
    &str = "NV_G4MOE_AB_SYNTHFILL";

const KV_SLOTS_OF_HEADROOM_PAST_THE_LAST_TIMED_STEP: usize = 64;

fn kv_slots_the_whole_run_will_consume_a_fixed_1024_cannot_hold_a_long_rotation_plan(
    depth: usize,
    rotations: usize,
    timed: usize,
) -> usize {
    depth
        + PLATEAU_WARMUP_STEPS_CEILING_THE_LOOP_ALSO_STOPS_ON_AN_8_SECOND_BUDGET
        + rotations * (DISCARDED_STEPS_AT_THE_HEAD_OF_EACH_ROTATION_SO_THE_HANDOFF_IS_NOT_TIMED + timed)
        + KV_SLOTS_OF_HEADROOM_PAST_THE_LAST_TIMED_STEP
}

const BITCHECK_STEPS_256_TIMES_262144_LOGITS_IS_THE_67_108_864_COMPARISON_STANDARD_THIS_FAMILY_HOLDS:
    usize = 256;

const LOGIT_BIT_COMPARISONS_A_SPEED_NUMBER_ON_THIS_FAMILY_IS_NOT_ALLOWED_TO_BE_QUOTED_UNDER:
    usize = 67_108_864;

struct ForcedTokenWalkSharedByBothInstancesSoNeitherTrajectoryCanDrift(u64);

impl ForcedTokenWalkSharedByBothInstancesSoNeitherTrajectoryCanDrift {
    fn next(&mut self, vocab: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32) % vocab
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn median(sorted_input: &mut [f64]) -> f64 {
    sorted_input.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted_input.len();
    if n % 2 == 1 {
        sorted_input[n / 2]
    } else {
        (sorted_input[n / 2 - 1] + sorted_input[n / 2]) / 2.0
    }
}

fn pin_record_stack_and(ladder_on: &[String]) -> common::EnvPins {
    let record_on = record_arms_shared_by_both_instances();
    let mut vars: Vec<(&'static str, Option<&str>)> = Vec::new();
    for k in RECORD_STACK_ARMS_BOTH_INSTANCES_SHARE_SO_THE_ONLY_DIFFERENCE_IS_THE_LADDER {
        vars.push((k, record_on.iter().any(|a| a == k).then_some("1")));
    }
    for k in KERNEL_ARMS_INERT_OFF_THE_GROUPED_PATH_SO_PINNING_THEM_OFF_LEAVES_ARM_A_UNCHANGED {
        assert!(
            RECORD_STACK_ARMS_BOTH_INSTANCES_SHARE_SO_THE_ONLY_DIFFERENCE_IS_THE_LADDER
                .contains(k),
            "{k} must be pinnable by this suite or arm A and arm B cannot be held equal on it"
        );
    }
    for k in EVERY_LADDER_ARM_THE_BUILDER_READS_SO_ARM_A_CAN_PIN_THEM_ALL_OFF {
        let on = ladder_on.iter().any(|a| a == k);
        vars.push((k, on.then_some("1")));
    }
    common::EnvPins::pin(&vars)
}

fn build_one(
    dir: &std::path::Path,
    max_seq: usize,
    ladder_on: &[String],
) -> (Gemma4MoeWgpu, usize) {
    let config = nv_models::gemma4_moe::Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json");
    let loader = nv_weights::WeightLoader::open_dir(dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let pins = pin_record_stack_and(ladder_on);
    let built = Gemma4MoeWgpu::from_loader(config, &loader, max_seq);
    drop(pins);
    drop(loader);
    let m = built.expect("build from loader");
    let n = m.pass_count();
    (m, n)
}

#[test]
#[ignore = "loads TWO real ~26B-A4B MoE instances on one wgpu context (2 x ~16.6 GiB); set NV_G4MOE_WALL_AB=1 -- prices the in-tree default-off dispatch-fusion ladder on the wall clock. Every ladder arm is read at graph-BUILD time, so one process is one arm and the usual interleaved same-session pairing is impossible; this suite builds one instance per arm instead and interleaves their decode steps with the order rotated. NV_G4MOE_AB_LADDER picks the arms arm-B adds, NV_CTX_TOKENS the prime depth"]
fn grouped_ladder_removes_dependent_dispatches_and_the_wall_clock_delta_is_priced_against_2_05_us_each(
) {
    if std::env::var("NV_G4MOE_WALL_AB").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_G4MOE_WALL_AB != 1");
        return;
    }
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_G4MOE_SNAPSHOT",
        &["models--google--gemma-4-26B-A4B-it"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("[wall-ab] adapter: {}", ctx.summary());

    let ladder_on: Vec<String> = std::env::var("NV_G4MOE_AB_LADDER")
        .unwrap_or_else(|_| {
            LADDER_ARMS_DEFAULT_THE_TOP_BIT_IDENTICAL_RUNG_QKVFOLD_AND_TAILFOLD_ARE_BANKED_NEGATIVE
                .to_string()
        })
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| format!("NV_G4MOE_{}", s.trim()))
        .collect();
    let base_on: Vec<String> = std::env::var("NV_G4MOE_AB_BASE")
        .unwrap_or_else(|_| {
            BASE_LADDER_ARM_A_ALSO_CARRIES_SO_A_NON_LADDER_ARM_CAN_BE_PRICED_ON_TOP_OF_THE_FUSED_STACK
                .to_string()
        })
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| format!("NV_G4MOE_{}", s.trim()))
        .collect();
    for a in ladder_on.iter().chain(base_on.iter()) {
        assert!(
            EVERY_LADDER_ARM_THE_BUILDER_READS_SO_ARM_A_CAN_PIN_THEM_ALL_OFF.contains(&a.as_str()),
            "NV_G4MOE_AB_LADDER names {a}, which arm A cannot pin off, so the two instances \
             would not differ by the ladder alone"
        );
    }
    for a in &base_on {
        assert!(
            ladder_on.contains(a),
            "NV_G4MOE_AB_BASE names {a} but NV_G4MOE_AB_LADDER does not: arm B must be arm A \
             PLUS the arm under test, never arm A minus one"
        );
    }

    let depth = env_usize(
        "NV_CTX_TOKENS",
        DEPTH_DEFAULT_256_IS_THE_RUNG_THE_10_96_TO_11_25_MS_STEP_WAS_MEASURED_AT,
    );
    let rotations = env_usize(
        "NV_G4MOE_AB_ROTATIONS",
        ROTATIONS_DEFAULT_6_SO_EACH_ORDER_AB_AND_BA_IS_RUN_THREE_TIMES,
    );
    let timed = env_usize("NV_G4MOE_AB_STEPS", TIMED_STEPS_PER_ROTATION_DEFAULT_24);
    let max_seq = kv_slots_the_whole_run_will_consume_a_fixed_1024_cannot_hold_a_long_rotation_plan(
        depth, rotations, timed,
    );

    let synthfill = std::env::var(
        SYNTHFILL_ENV_STEP_PRIMING_196608_TOKENS_ON_TWO_INSTANCES_IS_HOURS_WHERE_A_FILL_IS_SECONDS,
    )
    .ok()
    .as_deref()
        == Some("1");
    let vocab = nv_models::gemma4_moe::Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json")
        .base
        .vocab_size as u32;
    let (mut a, passes_a) = build_one(&dir, max_seq, &base_on);
    let (mut b, passes_b) = build_one(&dir, max_seq, &ladder_on);
    let record_on = record_arms_shared_by_both_instances();
    for a in &record_on {
        assert!(
            RECORD_STACK_ARMS_BOTH_INSTANCES_SHARE_SO_THE_ONLY_DIFFERENCE_IS_THE_LADDER
                .contains(&a.as_str()),
            "NV_G4MOE_AB_RECORD_ARMS names {a}, which this suite does not pin, so it would leak \
             from the ambient environment into only one of the two instances"
        );
    }
    eprintln!(
        "[wall-ab] shared_record_arms={}  arm_a=shared+{} passes/token={passes_a}  arm_b=shared+{} passes/token={passes_b}  removed={}  {}={}  synthfill={}",
        record_on.join("+"),
        base_on.join("+"),
        ladder_on.join("+"),
        passes_a as i64 - passes_b as i64,
        SUBMIT_CHUNK_IS_NOT_A_G4MOE_ARM_THIS_SUITE_CAN_PIN_YET_IS_WORTH_20_PCT_SO_EVERY_RUN_MUST_STATE_IT,
        std::env::var(
            SUBMIT_CHUNK_IS_NOT_A_G4MOE_ARM_THIS_SUITE_CAN_PIN_YET_IS_WORTH_20_PCT_SO_EVERY_RUN_MUST_STATE_IT
        )
        .unwrap_or_else(|_| "unset".to_string()),
        synthfill
    );
    let grid_shape_arm = ladder_on.iter().any(|a| {
        ARMS_THAT_RESHAPE_A_GRID_NOT_THE_PASS_LIST_SO_AN_EQUAL_PASS_COUNT_IS_THE_WHOLE_POINT
            .contains(&a.as_str())
    });
    assert!(
        passes_b < passes_a || grid_shape_arm,
        "arm B removed no dependent dispatch ({passes_b} vs {passes_a}): that is not a fusion"
    );

    let mut walk = ForcedTokenWalkSharedByBothInstancesSoNeitherTrajectoryCanDrift(0x5eed_4d7a);
    a.reset().expect("reset arm a");
    b.reset().expect("reset arm b");
    let mut first_divergence: Option<(usize, usize, u32, u32)> = None;
    let mut compared_logits = 0usize;
    let bitcheck_steps = env_usize(
        "NV_G4MOE_AB_BITCHECK_STEPS",
        BITCHECK_STEPS_256_TIMES_262144_LOGITS_IS_THE_67_108_864_COMPARISON_STANDARD_THIS_FAMILY_HOLDS,
    )
    .min(depth);
    let bit_identity_is_proved_on_real_decode_steps_from_reset_never_on_synthetic_state =
        if synthfill { bitcheck_steps } else { depth };
    let prime_start = std::time::Instant::now();
    for step in 0..bit_identity_is_proved_on_real_decode_steps_from_reset_never_on_synthetic_state {
        let t = walk.next(vocab);
        if step >= bitcheck_steps {
            a.decode_step(t).expect("arm a prime step");
            b.decode_step(t).expect("arm b prime step");
            continue;
        }
        let (ta, la) = a.decode_step_logits(t).expect("arm a prime step");
        let (tb, lb) = b.decode_step_logits(t).expect("arm b prime step");
        assert_eq!(la.len(), lb.len(), "logit vectors differ in length");
        compared_logits += la.len();
        if first_divergence.is_none() {
            if ta != tb {
                first_divergence = Some((step, usize::MAX, ta, tb));
            } else if let Some(i) = la
                .iter()
                .zip(lb.iter())
                .position(|(x, y)| x.to_bits() != y.to_bits())
            {
                first_divergence = Some((step, i, la[i].to_bits(), lb[i].to_bits()));
            }
        }
    }
    assert!(
        compared_logits
            >= LOGIT_BIT_COMPARISONS_A_SPEED_NUMBER_ON_THIS_FAMILY_IS_NOT_ALLOWED_TO_BE_QUOTED_UNDER,
        "only {compared_logits} raw-f32 logit bit-comparisons were made; the standing standard \
         on this family is \
         {LOGIT_BIT_COMPARISONS_A_SPEED_NUMBER_ON_THIS_FAMILY_IS_NOT_ALLOWED_TO_BE_QUOTED_UNDER} \
         per run and a wall number measured under it is not admissible"
    );
    let prime_s = prime_start.elapsed().as_secs_f64();
    match first_divergence {
        None => eprintln!(
            "[wall-ab] BIT-IDENTICAL over {bit_identity_is_proved_on_real_decode_steps_from_reset_never_on_synthetic_state} real-weight decode steps: {compared_logits} logit bit-comparisons, zero mismatches, prime_s={prime_s:.1}"
        ),
        Some((step, i, x, y)) => eprintln!(
            "[wall-ab] NOT BIT-IDENTICAL: first divergence at step {step} logit index {i} (a=0x{x:08x} b=0x{y:08x})"
        ),
    }
    assert!(
        first_divergence.is_none(),
        "the fused ladder must reproduce the unfused pass list bit for bit on real weights; \
         a divergence makes this an fp-reorder change and pulls in docs/book/08.2-quality-verdicts.md"
    );

    if synthfill {
        let fill_start = std::time::Instant::now();
        for m in [&mut a, &mut b] {
            m.reset().expect("reset before synthetic fill");
            m.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(depth)
                .unwrap_or_else(|e| panic!("synthetic state fill at depth {depth}: {e:#}"));
        }
        ctx.poll_blocking()
            .expect("drain synthetic fill writes before stopping the fill clock");
        let fill_s = fill_start.elapsed().as_secs_f64();
        assert_eq!(
            (a.current_pos(), b.current_pos()),
            (depth, depth),
            "both arms must sit at the requested depth after the synthetic fill or the pair is \
             not comparing the same context length"
        );
        eprintln!("[wall-ab] synthfill both arms to depth={depth} fill_s={fill_s:.1}");
    }

    let mut warm = ForcedTokenWalkSharedByBothInstancesSoNeitherTrajectoryCanDrift(0x117_5eed);
    for m in [&mut a, &mut b] {
        let mut steps = 0usize;
        let t0 = std::time::Instant::now();
        while steps < 200 && t0.elapsed().as_secs_f64() < 8.0 {
            m.decode_step(warm.next(vocab)).expect("plateau warmup step");
            steps += 1;
        }
        eprintln!("[wall-ab] plateau warmup steps={steps}");
    }

    let mut ms_a: Vec<f64> = Vec::new();
    let mut ms_b: Vec<f64> = Vec::new();
    for r in 0..rotations {
        let a_first = r % 2 == 0;
        for slot in 0..2 {
            let run_a = (slot == 0) == a_first;
            let (m, sink) = if run_a {
                (&mut a, &mut ms_a)
            } else {
                (&mut b, &mut ms_b)
            };
            for _ in 0..DISCARDED_STEPS_AT_THE_HEAD_OF_EACH_ROTATION_SO_THE_HANDOFF_IS_NOT_TIMED {
                m.decode_step(warm.next(vocab)).expect("discarded step");
            }
            for _ in 0..timed {
                let t = warm.next(vocab);
                let t0 = std::time::Instant::now();
                m.decode_step(t).expect("timed step");
                sink.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        let mut ta = ms_a[ms_a.len() - timed..].to_vec();
        let mut tb = ms_b[ms_b.len() - timed..].to_vec();
        eprintln!(
            "[wall-ab] rotation {r} order={} a_median_ms={:.4} b_median_ms={:.4}",
            if a_first { "A,B" } else { "B,A" },
            median(&mut ta),
            median(&mut tb)
        );
    }

    let med_a = median(&mut ms_a.clone());
    let med_b = median(&mut ms_b.clone());
    let removed = passes_a - passes_b;
    let predicted_ms =
        removed as f64 * DEPENDENT_WGPU_DISPATCH_US_MEASURED_ON_THIS_ADAPTER_NO_CUDA_GRAPH_EQUIVALENT
            / 1000.0;
    let measured_ms = med_a - med_b;
    eprintln!(
        "WALL-AB g4moe depth={depth} synthfill={synthfill} submit_chunk={} shared={} ladder={} pos_end_a={} pos_end_b={} steps_a={} steps_b={} \
         a_median_ms={med_a:.4} b_median_ms={med_b:.4} delta_ms={measured_ms:.4} \
         removed_dispatches={removed} predicted_ms={predicted_ms:.4} \
         realised_fraction={:.3} speedup={:.4}x",
        std::env::var(
            SUBMIT_CHUNK_IS_NOT_A_G4MOE_ARM_THIS_SUITE_CAN_PIN_YET_IS_WORTH_20_PCT_SO_EVERY_RUN_MUST_STATE_IT
        )
        .unwrap_or_else(|_| "unset".to_string()),
        record_on.join("+"),
        ladder_on.join("+"),
        a.current_pos(),
        b.current_pos(),
        ms_a.len(),
        ms_b.len(),
        measured_ms / predicted_ms,
        med_a / med_b
    );
    let mut sa = ms_a.clone();
    let mut sb = ms_b.clone();
    sa.sort_by(|x, y| x.partial_cmp(y).unwrap());
    sb.sort_by(|x, y| x.partial_cmp(y).unwrap());
    eprintln!(
        "WALL-AB spread a_min={:.4} a_p25={:.4} a_p75={:.4} a_max={:.4} b_min={:.4} b_p25={:.4} b_p75={:.4} b_max={:.4}",
        sa[0],
        sa[sa.len() / 4],
        sa[3 * sa.len() / 4],
        sa[sa.len() - 1],
        sb[0],
        sb[sb.len() / 4],
        sb[3 * sb.len() / 4],
        sb[sb.len() - 1]
    );
}

const LADDER_RUNGS_DEFAULT_THREE_IS_THE_MOST_THAT_FITS_CHAIN_LONGER_LADDERS_ON_A_SHARED_RUNG:
    &str = ";GROUPED_GEMV;GROUPED_GEMV,FOLD_CHAIN";

const INSTANCE_GIB_EACH_MEASURED_BY_THE_FOURTH_RUNG_OOMING_ON_A_96_GIB_CARD: f64 = 30.0;
const CARD_GIB_THE_RUNG_SET_MUST_FIT_INSIDE: f64 = 90.0;

#[test]
#[ignore = "loads ONE real ~26B-A4B MoE instance PER LADDER RUNG on one wgpu context; THREE rungs is the measured ceiling, a fourth returns wgpu Out of Memory on the 96 GiB card, so longer ladders are chained across runs on a shared rung; set NV_G4MOE_WALL_LADDER=1. Exists because the two-instance A/B cannot be compared ACROSS processes: a whole-process thermal/clock drift moved both arms of one run 4% together, which makes a marginal us-per-dispatch computed from two different runs meaningless. Every rung is resident and interleaved here instead, so the marginal rate between adjacent rungs is a within-process difference. NV_G4MOE_WALL_RUNGS is a semicolon-separated list of comma-separated arm sets, the first being the unfused baseline"]
fn every_ladder_rung_is_resident_at_once_so_the_marginal_us_per_removed_dispatch_is_a_within_process_difference(
) {
    if std::env::var("NV_G4MOE_WALL_LADDER").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_G4MOE_WALL_LADDER != 1");
        return;
    }
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_G4MOE_SNAPSHOT",
        &["models--google--gemma-4-26B-A4B-it"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("[wall-ladder] adapter: {}", ctx.summary());

    let rungs: Vec<Vec<String>> = std::env::var("NV_G4MOE_WALL_RUNGS")
        .unwrap_or_else(|_| {
            LADDER_RUNGS_DEFAULT_THREE_IS_THE_MOST_THAT_FITS_CHAIN_LONGER_LADDERS_ON_A_SHARED_RUNG.to_string()
        })
        .split(';')
        .map(|spec| {
            spec.split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| format!("NV_G4MOE_{}", s.trim()))
                .collect()
        })
        .collect();
    assert!(
        rungs.len() >= 2,
        "a ladder needs at least a baseline rung and one fused rung"
    );
    let marginal_base_opt_in = std::env::var("NV_G4MOE_WALL_MARGINAL_BASE").ok().as_deref()
        == Some("1");
    assert!(
        rungs[0].is_empty() || marginal_base_opt_in,
        "rung 0 must be the unfused baseline so every delta is measured against the unfused \
         reference; a fused rung 0 measures MARGINAL rates only and makes the bit-identity check \
         a mutual-agreement check between fused rungs rather than a proof against the unfused \
         pass list, so it requires NV_G4MOE_WALL_MARGINAL_BASE=1 and a prior run proving rung 0 \
         itself bit-identical to the unfused baseline"
    );
    assert!(
        rungs.len() as f64 * INSTANCE_GIB_EACH_MEASURED_BY_THE_FOURTH_RUNG_OOMING_ON_A_96_GIB_CARD
            <= CARD_GIB_THE_RUNG_SET_MUST_FIT_INSIDE,
        "{} rungs x {INSTANCE_GIB_EACH_MEASURED_BY_THE_FOURTH_RUNG_OOMING_ON_A_96_GIB_CARD} GiB does not fit \
         {CARD_GIB_THE_RUNG_SET_MUST_FIT_INSIDE} GiB; an OOM mid-run publishes nothing",
        rungs.len()
    );
    for r in rungs.iter().flatten() {
        assert!(
            EVERY_LADDER_ARM_THE_BUILDER_READS_SO_ARM_A_CAN_PIN_THEM_ALL_OFF.contains(&r.as_str()),
            "NV_G4MOE_WALL_RUNGS names {r}, which the baseline rung cannot pin off, so the rungs \
             would not differ by the ladder alone"
        );
    }

    let depth = env_usize(
        "NV_CTX_TOKENS",
        DEPTH_DEFAULT_256_IS_THE_RUNG_THE_10_96_TO_11_25_MS_STEP_WAS_MEASURED_AT,
    );
    let rotations = env_usize(
        "NV_G4MOE_AB_ROTATIONS",
        ROTATIONS_DEFAULT_6_SO_EACH_ORDER_AB_AND_BA_IS_RUN_THREE_TIMES,
    );
    let timed = env_usize("NV_G4MOE_AB_STEPS", TIMED_STEPS_PER_ROTATION_DEFAULT_24);
    let max_seq = kv_slots_the_whole_run_will_consume_a_fixed_1024_cannot_hold_a_long_rotation_plan(
        depth, rotations, timed,
    );
    let vocab = nv_models::gemma4_moe::Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json")
        .base
        .vocab_size as u32;

    let record_on = record_arms_shared_by_both_instances();
    let mut models: Vec<Gemma4MoeWgpu> = Vec::new();
    let mut passes: Vec<usize> = Vec::new();
    for spec in &rungs {
        let (m, n) = build_one(&dir, max_seq, spec);
        eprintln!(
            "[wall-ladder] rung {} arms={} passes/token={n}",
            models.len(),
            if spec.is_empty() {
                "baseline-unfused".to_string()
            } else {
                spec.join("+")
            }
        );
        models.push(m);
        passes.push(n);
    }
    eprintln!(
        "[wall-ladder] shared_record_arms={} rungs={} depth={depth}",
        record_on.join("+"),
        rungs.len()
    );

    let mut walk = ForcedTokenWalkSharedByBothInstancesSoNeitherTrajectoryCanDrift(0x5eed_4d7a);
    for m in models.iter_mut() {
        m.reset().expect("reset rung");
    }
    let mut mismatching_rungs: Vec<usize> = Vec::new();
    let mut compared_logits = 0usize;
    for _ in 0..depth {
        let t = walk.next(vocab);
        let (_, base) = models[0].decode_step_logits(t).expect("baseline prime step");
        for (i, m) in models.iter_mut().enumerate().skip(1) {
            let (_, l) = m.decode_step_logits(t).expect("rung prime step");
            assert_eq!(l.len(), base.len(), "rung {i} logit vector differs in length");
            compared_logits += l.len();
            if l.iter()
                .zip(base.iter())
                .any(|(x, y)| x.to_bits() != y.to_bits())
                && !mismatching_rungs.contains(&i)
            {
                mismatching_rungs.push(i);
            }
        }
    }
    eprintln!(
        "[wall-ladder] {compared_logits} logit bit-comparisons vs rung 0 over {depth} real-weight \
         decode steps; rungs that diverged: {mismatching_rungs:?}"
    );
    assert!(
        mismatching_rungs.is_empty(),
        "every fused rung must reproduce the unfused pass list bit for bit on real weights; \
         a divergence makes that rung an fp-reorder change and pulls in docs/book/08.2-quality-verdicts.md"
    );

    let mut warm = ForcedTokenWalkSharedByBothInstancesSoNeitherTrajectoryCanDrift(0x117_5eed);
    for m in models.iter_mut() {
        let mut steps = 0usize;
        let t0 = std::time::Instant::now();
        while steps < 200 && t0.elapsed().as_secs_f64() < 8.0 {
            m.decode_step(warm.next(vocab)).expect("plateau warmup step");
            steps += 1;
        }
    }

    let n = models.len();
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); n];
    for r in 0..rotations {
        for slot in 0..n {
            let idx = (slot + r) % n;
            for _ in 0..DISCARDED_STEPS_AT_THE_HEAD_OF_EACH_ROTATION_SO_THE_HANDOFF_IS_NOT_TIMED {
                models[idx]
                    .decode_step(warm.next(vocab))
                    .expect("discarded step");
            }
            for _ in 0..timed {
                let t = warm.next(vocab);
                let t0 = std::time::Instant::now();
                models[idx].decode_step(t).expect("timed step");
                samples[idx].push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        let meds: Vec<String> = (0..n)
            .map(|i| {
                let mut tail = samples[i][samples[i].len() - timed..].to_vec();
                format!("{:.4}", median(&mut tail))
            })
            .collect();
        eprintln!(
            "[wall-ladder] rotation {r} start_rung={} medians_ms=[{}]",
            r % n,
            meds.join(", ")
        );
    }

    let meds: Vec<f64> = (0..n).map(|i| median(&mut samples[i].clone())).collect();
    let mut prev_removed = 0usize;
    let mut prev_delta = 0.0f64;
    for i in 0..n {
        let removed = passes[0] - passes[i];
        let delta = meds[0] - meds[i];
        let marginal_removed = removed - prev_removed;
        let marginal_us = if marginal_removed == 0 {
            f64::NAN
        } else {
            (delta - prev_delta) * 1000.0 / marginal_removed as f64
        };
        eprintln!(
            "WALL-LADDER rung={i} arms={} passes={} removed={removed} median_ms={:.4} \
             delta_ms={delta:.4} cumulative_us_per_dispatch={:.3} marginal_removed={marginal_removed} \
             marginal_us_per_dispatch={marginal_us:.3} predicted_ms_at_2_05us={:.4} realised={:.3}",
            if rungs[i].is_empty() {
                "baseline-unfused".to_string()
            } else {
                rungs[i].join("+")
            },
            passes[i],
            meds[i],
            if removed == 0 {
                f64::NAN
            } else {
                delta * 1000.0 / removed as f64
            },
            removed as f64
                * DEPENDENT_WGPU_DISPATCH_US_MEASURED_ON_THIS_ADAPTER_NO_CUDA_GRAPH_EQUIVALENT
                / 1000.0,
            if removed == 0 {
                f64::NAN
            } else {
                delta
                    / (removed as f64
                        * DEPENDENT_WGPU_DISPATCH_US_MEASURED_ON_THIS_ADAPTER_NO_CUDA_GRAPH_EQUIVALENT
                        / 1000.0)
            }
        );
        prev_removed = removed;
        prev_delta = delta;
    }
}
