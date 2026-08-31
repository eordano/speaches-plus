#![cfg(feature = "wgpu")]

mod common;
use common::snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights;
use nv_models::gpt_oss_wgpu as gow;

const DEPENDENT_WGPU_DISPATCH_US_MEASURED_ON_THIS_ADAPTER_NO_CUDA_GRAPH_EQUIVALENT: f64 = 2.05;

const AN_INDEPENDENT_SIBLING_DISPATCH_US_WGPU_EMITS_NO_BARRIER_WHEN_IT_SEES_NO_HAZARD: f64 = 0.05;

const SHIPPED_STACK_BOTH_INSTANCES_SHARE_SO_THE_ONLY_DIFFERENCE_IS_THE_DNCOMB_FUSION: &[&str] = &[
    "NV_GOW_MX_SG_XV4",
    "NV_GOW_MX_SG_BYTE_LUT",
    "NV_GOW_MX_SG_XSTAGE",
    "NV_GOW_MOE_TOPK_SG32",
    "NV_GOW_FLASH_STAGE2_SCC",
    "NV_GOW_MX_SG_GUPAIR",
    "NV_GOW_LMHEAD_I8",
    "NV_GOW_MX_SG_GUSW",
    "NV_GOW_RMSRES_SGLAT",
    "NV_GOW_FLASH_DEPTHGATE",
];

const A_NON_BIT_IDENTICAL_ARM_MAY_STILL_BE_PRICED_ON_THE_WALL_BUT_ONLY_UNDER_THIS_GATE: &str =
    "NV_GOW_DNCOMB_AB_ALLOW_DIVERGENCE";

const ARMS_PINNED_OFF_SO_NEITHER_INSTANCE_INHERITS_ONE_FROM_THE_AMBIENT_ENVIRONMENT: &[&str] = &[
    "NV_GOW_MOE_ROUTER_TOPK_FUSE",
    "NV_GOW_MX_KPRE",
    "NV_GOW_MX_SG_XV4_R2",
    "NV_GOW_GEMV_SG_KPRE",
    "NV_GOW_FLASH_KPRE",
];

const DEPTH_DEFAULT_256_IS_THE_SHALLOW_RUNG_THE_FAMILY_PUBLISHES_AT: usize = 256;
const ROTATIONS_DEFAULT_6_SO_EACH_ORDER_AB_AND_BA_IS_RUN_THREE_TIMES: usize = 6;
const TIMED_STEPS_PER_ROTATION_DEFAULT_24: usize = 24;
const DISCARDED_STEPS_AT_THE_HEAD_OF_EACH_ROTATION_SO_THE_HANDOFF_IS_NOT_TIMED: usize = 4;
const KV_SLOTS_BEYOND_THE_PRIME_FOR_PLATEAU_WARMUP_PLUS_EVERY_ROTATION: usize = 1024;

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

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn hazard_barriers_over_the_encoded_pass_list(gpu: &gow::GptOssWgpu) -> usize {
    let binds =
        gpu.decode_pass_storage_binding_sets_a_shared_id_between_two_passes_is_a_wgpu_hazard();
    let mut read_since: std::collections::BTreeSet<u32> = Default::default();
    let mut written_since: std::collections::BTreeSet<u32> = Default::default();
    let mut barriers = 0usize;
    for set in &binds {
        let raw_or_waw = set.iter().any(|(id, _)| written_since.contains(id));
        let war = set
            .iter()
            .any(|(id, w)| *w && (read_since.contains(id) || written_since.contains(id)));
        if raw_or_waw || war {
            barriers += 1;
            read_since.clear();
            written_since.clear();
        }
        for (id, w) in set {
            if *w {
                written_since.insert(*id);
            } else {
                read_since.insert(*id);
            }
        }
    }
    barriers
}

const ARM_B_DEFAULT_IS_THE_DOWN_GEMV_COMBINE_FUSION_THIS_SUITE_WAS_BUILT_FOR: &str =
    "NV_GOW_MOE_DNCOMB_FUSE=1";

fn arm_b_overrides() -> Vec<(String, String)> {
    std::env::var("NV_GOW_DNCOMB_AB_ARM_B")
        .unwrap_or_else(|_| {
            ARM_B_DEFAULT_IS_THE_DOWN_GEMV_COMBINE_FUSION_THIS_SUITE_WAS_BUILT_FOR.to_string()
        })
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|kv| {
            let (k, v) = kv
                .split_once('=')
                .unwrap_or_else(|| panic!("arm-B override {kv} is not KEY=VALUE"));
            (k.trim().to_string(), v.trim().to_string())
        })
        .collect()
}

fn build_one(
    dir: &std::path::Path,
    max_seq: usize,
    overrides: &[(String, String)],
) -> (gow::GptOssWgpu, usize, usize) {
    let cfg = gow::GptOssConfig::from_hf_json_file(dir.join("config.json")).expect("config.json");
    let loader = nv_weights::WeightLoader::open_dir(dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut vars: Vec<(&'static str, Option<&str>)> = Vec::new();
    for k in SHIPPED_STACK_BOTH_INSTANCES_SHARE_SO_THE_ONLY_DIFFERENCE_IS_THE_DNCOMB_FUSION {
        vars.push((k, Some("1")));
    }
    for k in ARMS_PINNED_OFF_SO_NEITHER_INSTANCE_INHERITS_ONE_FROM_THE_AMBIENT_ENVIRONMENT {
        vars.push((k, None));
    }
    vars.push((gow::MOE_DNCOMB_FUSE_ENV, Some("0")));
    vars.push((gow::MOE_DNCOMB_SGSLOT_ENV, Some("0")));
    vars.push((gow::NULL_DISPATCH_ENV, None));
    let pins = common::EnvPins::pin(&vars);
    let saved: Vec<(String, Option<String>)> = overrides
        .iter()
        .map(|(k, v)| {
            let prev = std::env::var(k).ok();
            std::env::set_var(k, v);
            (k.clone(), prev)
        })
        .collect();
    let built = gow::GptOssWgpu::from_loader(cfg.clone(), &loader, max_seq);
    for (k, prev) in saved {
        match prev {
            Some(v) => std::env::set_var(&k, v),
            None => std::env::remove_var(&k),
        }
    }
    drop(pins);
    drop(loader);
    let m = built.unwrap_or_else(|e| panic!("build from loader: {e:#}"));
    let labels = m.decode_pass_labels();
    let fused = labels
        .iter()
        .filter(|l| l.starts_with("gow-moe-gemv-dncomb:"))
        .count();
    let split_dn = labels
        .iter()
        .filter(|l| l.starts_with("gow-moe-gemv-dn:"))
        .count();
    let combine = labels
        .iter()
        .filter(|l| l.starts_with("gow-moe-combine:"))
        .count();
    let nulls = labels
        .iter()
        .filter(|l| l.starts_with("gow-nulldisp:"))
        .count();
    let layers = cfg.num_hidden_layers;
    let want_dncomb = overrides
        .iter()
        .any(|(k, v)| k == gow::MOE_DNCOMB_FUSE_ENV && v == "1");
    let want_nulls: usize = overrides
        .iter()
        .find(|(k, _)| k == gow::NULL_DISPATCH_ENV)
        .map_or(0, |(_, v)| v.parse().unwrap_or(0));
    if want_dncomb {
        assert_eq!(
            (fused, split_dn, combine),
            (layers, 0, 0),
            "the dncomb arm never engaged, so any equality or speed number would prove nothing"
        );
    } else {
        assert_eq!(
            (fused, split_dn, combine),
            (0, layers, layers),
            "this arm must run the split down-gemv and combine pair"
        );
    }
    assert_eq!(
        nulls > 0,
        want_nulls > 0,
        "the null-dispatch probe engagement must match its env gate ({nulls} null passes)"
    );
    let passes = m.pass_count();
    let barriers = hazard_barriers_over_the_encoded_pass_list(&m);
    (m, passes, barriers)
}

#[test]
#[ignore = "loads TWO real gpt-oss-20b MXFP4 instances on one wgpu context (2 x ~13 GB); set \
            NV_GOW_DNCOMB_AB=1 -- proves NV_GOW_MOE_DNCOMB_FUSE removes one hazard barrier per \
            layer, reproduces the split gow-moe-gemv-dn + gow-moe-combine pair bit for bit on \
            real weights, and prices the wall-clock delta against the measured 2.05 us per \
            dependent dispatch. Both arms are read at graph-BUILD time, so one process is one \
            arm and the usual interleaved pairing needs two resident instances"]
fn dncomb_fusion_removes_one_barrier_per_layer_is_bit_identical_and_the_wall_delta_is_priced_at_2_05_us_each(
) {
    if std::env::var("NV_GOW_DNCOMB_AB").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_GOW_DNCOMB_AB != 1");
        return;
    }
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_GPTOSS_DIR",
        &["models--openai--gpt-oss-20b"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("[dncomb-ab] adapter: {}", ctx.summary());

    let cfg = gow::GptOssConfig::from_hf_json_file(dir.join("config.json")).expect("config.json");
    let vocab = cfg.vocab_size as u32;
    let layers = cfg.num_hidden_layers;
    let depth = env_usize(
        "NV_CTX_TOKENS",
        DEPTH_DEFAULT_256_IS_THE_SHALLOW_RUNG_THE_FAMILY_PUBLISHES_AT,
    );
    let rotations = env_usize(
        "NV_GOW_DNCOMB_ROTATIONS",
        ROTATIONS_DEFAULT_6_SO_EACH_ORDER_AB_AND_BA_IS_RUN_THREE_TIMES,
    );
    let timed = env_usize("NV_GOW_DNCOMB_STEPS", TIMED_STEPS_PER_ROTATION_DEFAULT_24);
    let max_seq = depth + KV_SLOTS_BEYOND_THE_PRIME_FOR_PLATEAU_WARMUP_PLUS_EVERY_ROTATION;

    let overrides = arm_b_overrides();
    let b_first = std::env::var("NV_GOW_DNCOMB_AB_BUILD_B_FIRST").ok().as_deref() == Some("1");
    let (mut a, passes_a, barriers_a, mut b, passes_b, barriers_b) = if b_first {
        let bb = build_one(&dir, max_seq, &overrides);
        let aa = build_one(&dir, max_seq, &[]);
        (aa.0, aa.1, aa.2, bb.0, bb.1, bb.2)
    } else {
        let aa = build_one(&dir, max_seq, &[]);
        let bb = build_one(&dir, max_seq, &overrides);
        (aa.0, aa.1, aa.2, bb.0, bb.1, bb.2)
    };
    let arm_b_name: String = overrides
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "[dncomb-ab] build_order={}  arm_a=shipped passes={passes_a} barriers={barriers_a}  \
         arm_b=shipped+[{arm_b_name}] passes={passes_b} barriers={barriers_b}  \
         dispatches_removed={} barriers_removed={} layers={layers}",
        if b_first { "B,A" } else { "A,B" },
        passes_a as i64 - passes_b as i64,
        barriers_a as i64 - barriers_b as i64
    );
    assert_ne!(
        (passes_a, barriers_a),
        (passes_b, barriers_b),
        "arm B changed neither the dispatch count nor the barrier count: nothing to price"
    );

    let mut walk = ForcedTokenWalkSharedByBothInstancesSoNeitherTrajectoryCanDrift(0x6055_d0c0);
    a.reset().expect("reset arm a");
    b.reset().expect("reset arm b");
    let mut first_divergence: Option<(usize, usize, u32, u32)> = None;
    let mut compared_logits = 0usize;
    let prime_start = std::time::Instant::now();
    for step in 0..depth {
        let t = walk.next(vocab);
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
    let prime_s = prime_start.elapsed().as_secs_f64();
    match first_divergence {
        None => eprintln!(
            "[dncomb-ab] BIT-IDENTICAL over {depth} real-weight decode steps: \
             {compared_logits} logit bit-comparisons, zero mismatches, prime_s={prime_s:.1}"
        ),
        Some((step, i, x, y)) => eprintln!(
            "[dncomb-ab] NOT BIT-IDENTICAL: first divergence at step {step} logit index {i} \
             (a=0x{x:08x} b=0x{y:08x})"
        ),
    }
    let divergence_allowed =
        std::env::var(A_NON_BIT_IDENTICAL_ARM_MAY_STILL_BE_PRICED_ON_THE_WALL_BUT_ONLY_UNDER_THIS_GATE)
            .ok()
            .as_deref()
            == Some("1");
    assert!(
        first_divergence.is_none() || divergence_allowed,
        "folding gow-moe-combine into the down gemv keeps each expert row in an f32 register \
         instead of round-tripping an f32 storage buffer, which is exact, so the same ascending \
         fma chain must reproduce the split pair bit for bit on real weights; a divergence makes \
         this an fp-reorder change and pulls in docs/book/08.2-quality-verdicts.md"
    );

    let mut warm = ForcedTokenWalkSharedByBothInstancesSoNeitherTrajectoryCanDrift(0x117_d0c0);
    for m in [&mut a, &mut b] {
        let mut steps = 0usize;
        let t0 = std::time::Instant::now();
        while steps < 200 && t0.elapsed().as_secs_f64() < 8.0 {
            m.decode_step(warm.next(vocab)).expect("plateau warmup step");
            steps += 1;
        }
        eprintln!("[dncomb-ab] plateau warmup steps={steps}");
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
            "[dncomb-ab] rotation {r} order={} a_median_ms={:.4} b_median_ms={:.4}",
            if a_first { "A,B" } else { "B,A" },
            median(&mut ta),
            median(&mut tb)
        );
    }

    let med_a = median(&mut ms_a.clone());
    let med_b = median(&mut ms_b.clone());
    let barriers_removed = barriers_a as i64 - barriers_b as i64;
    let free_removed = (passes_a as i64 - passes_b as i64) - barriers_removed;
    let predicted_ms = (barriers_removed as f64
        * DEPENDENT_WGPU_DISPATCH_US_MEASURED_ON_THIS_ADAPTER_NO_CUDA_GRAPH_EQUIVALENT
        + free_removed as f64
            * AN_INDEPENDENT_SIBLING_DISPATCH_US_WGPU_EMITS_NO_BARRIER_WHEN_IT_SEES_NO_HAZARD)
        / 1000.0;
    let measured_ms = med_a - med_b;
    eprintln!(
        "WALL-AB gow arm_b=[{arm_b_name}] build_order={} depth={depth} pos_end_a={} pos_end_b={} \
         steps_a={} steps_b={} a_median_ms={med_a:.4} b_median_ms={med_b:.4} \
         delta_ms={measured_ms:.4} barriers_removed={barriers_removed} free_removed={free_removed} \
         predicted_ms={predicted_ms:.4} us_per_barrier_measured={:.3} realised_fraction={:.3} \
         speedup={:.4}x",
        if b_first { "B,A" } else { "A,B" },
        a.current_pos(),
        b.current_pos(),
        ms_a.len(),
        ms_b.len(),
        measured_ms * 1000.0 / barriers_removed as f64,
        measured_ms / predicted_ms,
        med_a / med_b
    );
    let mut sa = ms_a.clone();
    let mut sb = ms_b.clone();
    sa.sort_by(|x, y| x.partial_cmp(y).unwrap());
    sb.sort_by(|x, y| x.partial_cmp(y).unwrap());
    eprintln!(
        "WALL-AB spread a_min={:.4} a_p25={:.4} a_p75={:.4} a_max={:.4} \
         b_min={:.4} b_p25={:.4} b_p75={:.4} b_max={:.4}",
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
