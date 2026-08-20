#![cfg(feature = "wgpu")]
#![allow(dead_code)]

mod common;
use common::env_usize;
#[path = "fp8_contract_prompts.rs"]
mod prompts;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use prompts::{
    ab_prompts, compare, decide_flip, describe_selection, resolve_pack, ArmObservation,
    DistributionalSummary, FlipBar, FlipDecision, FlipVerdict, FreeRun, InstrumentReport,
    PromptKind, PromptPack, RunTable, StopReason, StopSet, SuiteReport, TemplatedPrompt,
    FLIP_BAR_DOC, KERNEL_FLIP_VS_WEIGHT_FORMAT_FLIP, MARGIN_TRAP, REPRODUCIBILITY_LIMIT,
    WHY_A_PACK,
};
use common::ctx_or_skip_bool as ctx_or_skip;

static ENV_LOCK: Mutex<()> = Mutex::new(());

pub const WHAT_THIS_DECIDES: &str = "\
Two proposed default flips, both implemented, both currently OFF, neither bit-identical end to
end. This harness decides them on evidence:

  F1  NV_WGPU_NVFP4_V2=1 (Gemma-4-31B-IT-NVFP4) / NV_Q3_WGPU_NVFP4_V2=1 (Qwen3.6-35B-A3B-NVFP4).
      Every v2 route is bit-exact against the shipping route PER KERNEL (pinned by
      wgpu_gemma4_nvfp4_v2_routing.rs and wgpu_qwen35_nvfp4_v2_routing.rs), but the k-block
      summation ORDER differs, so end to end the 31B is not bit-identical and its completion
      hash changes. Qwen3.6 is reported byte-identical both ways; if that holds here the verdict
      is ACCEPT (BIT-IDENTICAL), which is strictly stronger than passing the bar.

  F2  NV_WGPU_LMHEAD_INT8=1 (Gemma-4-31B-IT-NVFP4). An int8 logit projection perturbs the logits
      directly, which is the one place a perturbation can move argmax with nothing downstream to
      damp it. The existing coverage is a synthetic 6-layer agreement test
      (gemma4_wgpu_quant.rs::synthetic_lmhead_int8_agreement_and_determinism, which says in its
      own assertion message that argmax agreement is NOT a quality signal there) and a real-weight
      forced-replay agreement ratio in the same file. Neither is free-running, neither is
      EOS-aware, and neither states a bar. This one is.

  F1+F2 is measured as its own arm because that is the configuration that would actually ship.

NO TIMING IS COLLECTED HERE. Throughput is a separate lane's measurement and mixing the two is
how a quality claim gets contaminated by a busy GPU.";

const SCRUBBED_KNOBS: [&str; 11] = [
    "NV_WGPU_NVFP4_V2",
    "NV_Q3_WGPU_NVFP4_V2",
    "NV_Q3_WGPU_ROUTER_PAR",
    "NV_WGPU_NVFP4_TREE",
    "NV_WGPU_NVFP4_SG32_V2",
    "NV_WGPU_LMHEAD_INT8",
    "NV_E4B_WGPU_W8",
    "NV_Q3_WGPU_W8_EXPERTS",
    "NV_G4_WGPU_W8_FFN",
    "NV_G4_WGPU_W8_FFN_GROUP",
    "NV_E4B_WGPU_W8_GROUP",
];

struct Arm {
    label: &'static str,
    env: &'static [(&'static str, &'static str)],
    expects_v2_route: bool,
    expects_byte_change: bool,
}

const GEMMA_REFERENCE: Arm = Arm {
    label: "shipping-default",
    env: &[],
    expects_v2_route: false,
    expects_byte_change: false,
};

const GEMMA_CANDIDATES: [Arm; 3] = [
    Arm {
        label: "F1:nvfp4-v2",
        env: &[("NV_WGPU_NVFP4_V2", "1")],
        expects_v2_route: true,
        expects_byte_change: false,
    },
    Arm {
        label: "F2:lmhead-int8",
        env: &[("NV_WGPU_LMHEAD_INT8", "1")],
        expects_v2_route: false,
        expects_byte_change: true,
    },
    Arm {
        label: "F1+F2",
        env: &[("NV_WGPU_NVFP4_V2", "1"), ("NV_WGPU_LMHEAD_INT8", "1")],
        expects_v2_route: true,
        expects_byte_change: true,
    },
];

const QWEN_REFERENCE: Arm = Arm {
    label: "shipping-default",
    env: &[],
    expects_v2_route: false,
    expects_byte_change: false,
};

const QWEN_CANDIDATES: [Arm; 3] = [
    Arm {
        label: "R2-revert:router-serial",
        env: &[("NV_Q3_WGPU_ROUTER_PAR", "0")],
        expects_v2_route: false,
        expects_byte_change: false,
    },
    Arm {
        label: "R3-revert:nvfp4-v1",
        env: &[("NV_Q3_WGPU_NVFP4_V2", "0")],
        expects_v2_route: false,
        expects_byte_change: false,
    },
    Arm {
        label: "W8:int8-experts",
        env: &[("NV_Q3_WGPU_W8_EXPERTS", "1")],
        expects_v2_route: false,
        expects_byte_change: false,
    },
];

pub const SILENT_NO_OP_IS_THE_WORST_OUTCOME: &str = "\
A flip that does not reach the code silently reports ACCEPT (BIT-IDENTICAL), because the two arms
really are the same binary. That is the single most dangerous failure mode of an A/B harness, so
every candidate arm declares what must observably change and the harness panics if it does not:
the v2 arms must flip nvfp4_v2_enabled(ctx) true, and the int8 lm_head arm must change
weight_bytes_per_token().";

fn scrub_and_report() {
    let ambient: Vec<String> = SCRUBBED_KNOBS
        .iter()
        .filter_map(|k| std::env::var(k).ok().map(|v| format!("{k}={v}")))
        .collect();
    if ambient.is_empty() {
        eprintln!("[env] no flip knob was set in the ambient environment");
    } else {
        eprintln!(
            "[env] SCRUBBING ambient flip knobs so the reference arm is the shipping default, \
             not whatever this shell had: {}",
            ambient.join(" ")
        );
    }
    for k in SCRUBBED_KNOBS {
        std::env::remove_var(k);
    }
    for k in ["NV_DETERMINISTIC", "NV_WGPU_PROFILE"] {
        eprintln!(
            "[env] {k}={:?} (not scrubbed; reported so a run is reproducible)",
            std::env::var(k).ok()
        );
    }
    eprintln!("{SILENT_NO_OP_IS_THE_WORST_OUTCOME}");
}

fn apply(a: &Arm) {
    for k in SCRUBBED_KNOBS {
        std::env::remove_var(k);
    }
    for (k, v) in a.env {
        std::env::set_var(k, v);
    }
    let set: Vec<String> = a.env.iter().map(|(k, v)| format!("{k}={v}")).collect();
    eprintln!(
        "[arm {}] env {}",
        a.label,
        if set.is_empty() {
            "<none: shipping defaults>".to_string()
        } else {
            set.join(" ")
        }
    );
}

fn opt_in(var: &str) -> bool {
    if std::env::var(var).ok().as_deref() == Some("1") {
        return true;
    }
    eprintln!("skipping: set {var}=1 to run (real-weight test)");
    false
}

fn limit_prompts(mut chosen: Vec<&TemplatedPrompt>) -> Vec<&TemplatedPrompt> {
    let Some(n) = std::env::var("NV_FLIP_PROMPTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0 && *n < chosen.len())
    else {
        return chosen;
    };
    chosen.truncate(n);
    eprintln!(
        "[shakedown] NV_FLIP_PROMPTS={n}: truncated to {} prompt(s) ({} control). A truncated run \
         is a PLUMBING SHAKEDOWN, not evidence -- SuiteReport::validate refuses a claim below \
         {} prompts and decide_flip refuses one with no control.",
        chosen.len(),
        chosen
            .iter()
            .filter(|p| p.kind == PromptKind::Control)
            .count(),
        prompts::MIN_PROMPTS_FOR_A_CLAIM
    );
    chosen
}

fn arm_selected(a: &Arm) -> bool {
    let Ok(only) = std::env::var("NV_FLIP_ARMS") else {
        return true;
    };
    let keep = only
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .any(|s| a.label.starts_with(s) || a.label == s);
    if !keep {
        eprintln!("[{}] skipped by NV_FLIP_ARMS={only}", a.label);
    }
    keep
}

fn out_dir() -> PathBuf {
    let d = std::env::var("NV_FLIP_OUT")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("NV_CHAT_EVAL_OUT").map(PathBuf::from))
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".cache/nvk-tmp/flip-eval")
        });
    std::fs::create_dir_all(&d).ok();
    d
}

fn pack_or_refuse(dir: &Path) -> PromptPack {
    match resolve_pack(dir) {
        Ok((p, pack)) => {
            eprintln!("[pack] {}", p.display());
            eprintln!("{WHY_A_PACK}");
            pack
        }
        Err(e) => panic!("REFUSING TO RUN: {e}"),
    }
}

struct StepRow {
    gpu: u32,
    cpu_top1: u32,
    margin: f32,
    logits: Option<Vec<f32>>,
}

fn top2_of(l: &[f32]) -> (u32, f32) {
    let (i1, v1, _, v2) = prompts::top2(l);
    (i1, v1 - v2)
}

trait Decoder {
    fn reset(&mut self) -> anyhow::Result<()>;
    fn step(&mut self, token: u32) -> anyhow::Result<(u32, Vec<f32>)>;
}

struct GemmaDecoder<'a>(&'a mut nv_models::gemma4_wgpu::Gemma4Wgpu);

impl Decoder for GemmaDecoder<'_> {
    fn reset(&mut self) -> anyhow::Result<()> {
        self.0.reset();
        Ok(())
    }
    fn step(&mut self, token: u32) -> anyhow::Result<(u32, Vec<f32>)> {
        self.0.decode_step_logits(token)
    }
}

struct QwenDecoder<'a>(&'a mut nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu);

impl Decoder for QwenDecoder<'_> {
    fn reset(&mut self) -> anyhow::Result<()> {
        self.0.reset()
    }
    fn step(&mut self, token: u32) -> anyhow::Result<(u32, Vec<f32>)> {
        self.0.decode_step_logits(token)
    }
}

fn prefill(d: &mut dyn Decoder, prompt: &[u32]) -> anyhow::Result<(u32, Vec<f32>)> {
    d.reset()?;
    let mut logits = Vec::new();
    let mut gpu = 0u32;
    for t in prompt {
        let (tok, l) = d.step(*t)?;
        gpu = tok;
        logits = l;
    }
    anyhow::ensure!(!logits.is_empty(), "empty logits after prefill");
    Ok((gpu, logits))
}

fn free_run(
    d: &mut dyn Decoder,
    prompt: &[u32],
    max_steps: usize,
    stops: &StopSet,
    keep: usize,
) -> anyhow::Result<(Vec<StepRow>, Option<u32>)> {
    let (mut gpu, mut logits) = prefill(d, prompt)?;
    let mut out = Vec::with_capacity(max_steps);
    let mut stopped = None;
    for i in 0..max_steps {
        let (cpu_top1, margin) = top2_of(&logits);
        out.push(StepRow {
            gpu,
            cpu_top1,
            margin,
            logits: (i < keep).then(|| logits.clone()),
        });
        if stops.contains(gpu) {
            stopped = Some(gpu);
            break;
        }
        let (tok, l) = d.step(gpu)?;
        gpu = tok;
        logits = l;
    }
    Ok((out, stopped))
}

fn forced_run(
    d: &mut dyn Decoder,
    prompt: &[u32],
    forced: &[u32],
    keep: usize,
) -> anyhow::Result<Vec<StepRow>> {
    let (mut gpu, mut logits) = prefill(d, prompt)?;
    let mut out = Vec::with_capacity(forced.len());
    for (i, want) in forced.iter().enumerate() {
        let (cpu_top1, margin) = top2_of(&logits);
        out.push(StepRow {
            gpu,
            cpu_top1,
            margin,
            logits: (i < keep).then(|| logits.clone()),
        });
        if i + 1 < forced.len() {
            let (tok, l) = d.step(*want)?;
            gpu = tok;
            logits = l;
        }
    }
    Ok(out)
}

fn leg(
    d: &mut dyn Decoder,
    p: &TemplatedPrompt,
    max_steps: usize,
    stops: &StopSet,
    keep: usize,
    forced_ids: Option<&[u32]>,
) -> anyhow::Result<(Vec<StepRow>, Option<u32>, Vec<StepRow>)> {
    let (rows, stopped) = free_run(d, &p.ids, max_steps, stops, 0)?;
    let ids: Vec<u32> = match forced_ids {
        Some(v) => v.to_vec(),
        None => rows.iter().map(|r| r.gpu).collect(),
    };
    let forced = forced_run(d, &p.ids, &ids, keep)?;
    Ok((rows, stopped, forced))
}

fn run_of(arm: &str, p: &TemplatedPrompt, rows: &[StepRow], stopped: Option<u32>) -> FreeRun {
    FreeRun {
        arm: arm.to_string(),
        prompt_label: p.label.clone(),
        tokens: rows.iter().map(|r| r.gpu).collect(),
        margins: rows.iter().map(|r| r.margin).collect(),
        reason: if stopped.is_some() {
            StopReason::HitStopToken
        } else {
            StopReason::ReachedMaxSteps
        },
        stop_token: stopped,
        text: String::new(),
    }
}

struct ReferenceLeg {
    runs: Vec<FreeRun>,
    forced: Vec<Vec<StepRow>>,
    table: RunTable,
}

fn sampler_note(rows: &[StepRow]) -> String {
    let n = rows.iter().filter(|r| r.gpu != r.cpu_top1).count();
    let first = rows.iter().position(|r| r.gpu != r.cpu_top1);
    format!(
        "GPU-sampler vs CPU-argmax mismatches {n}/{} (first at {first:?})",
        rows.len()
    )
}

struct ArmResult {
    decision: FlipDecision,
    instrument: InstrumentReport,
}

#[allow(clippy::too_many_arguments)]
fn score_candidate(
    flip: &str,
    model: &str,
    reference_arm: &str,
    candidate_arm: &str,
    chosen: &[&TemplatedPrompt],
    stops: &StopSet,
    reference: &ReferenceLeg,
    decode: &mut dyn FnMut(
        &TemplatedPrompt,
        &[u32],
    ) -> anyhow::Result<(Vec<StepRow>, Option<u32>, Vec<StepRow>)>,
    decode_text: &dyn Fn(&[u32]) -> String,
    label_of: &dyn Fn(u32) -> String,
    max_steps: usize,
) -> anyhow::Result<(ArmResult, SuiteReport, Vec<DistributionalSummary>, RunTable)> {
    let mut suite = SuiteReport::new(&format!("{flip} on {model}"), reference_arm, candidate_arm);
    let mut dists: Vec<DistributionalSummary> = Vec::new();
    let mut table = RunTable::new(&format!("{model} :: {candidate_arm}"));

    for (i, p) in chosen.iter().enumerate() {
        let ref_run = &reference.runs[i];
        let (free_rows, stopped, forced_rows) = decode(p, &ref_run.tokens)?;
        let mut cand = run_of(candidate_arm, p, &free_rows, stopped);
        cand.text = decode_text(&cand.tokens);
        eprintln!(
            "[{candidate_arm}] {:<24} free-running {} tokens, {}",
            p.label,
            cand.tokens.len(),
            sampler_note(&free_rows)
        );
        table.push(ArmObservation::of(p, &cand, stops, max_steps, label_of));

        let mut d = DistributionalSummary::new(&p.label, p.kind);
        for (r, c) in reference.forced[i].iter().zip(forced_rows.iter()) {
            match (&r.logits, &c.logits) {
                (Some(a), Some(b)) => d.push(prompts::step_delta(a, b)),
                _ => break,
            }
        }
        eprintln!("[{candidate_arm}] {d}");
        dists.push(d);

        suite.push(compare(p, ref_run, &cand));
    }

    eprintln!("{table}");
    eprintln!("{suite}");
    let decision = decide_flip(flip, model, FlipBar::from_env(), &suite, &dists);
    let instrument = InstrumentReport::new(&format!("{model} :: {candidate_arm}"), &dists);
    Ok((
        ArmResult {
            decision,
            instrument,
        },
        suite,
        dists,
        table,
    ))
}

fn per_prompt_json(r: &InstrumentReport) -> serde_json::Value {
    let row = |s: &DistributionalSummary| {
        serde_json::json!({
            "prompt": s.prompt_label,
            "kind": format!("{}", s.prompt_kind),
            "steps": s.steps,
            "mean_kl_nats": s.mean_kl,
            "max_step_kl_nats": s.max_kl,
            "relative_perplexity_pct": prompts::relative_perplexity_pct(s.mean_kl),
            "forced_top1": s.top1_rate(),
            "forced_top1_agree": s.top1_agree,
            "mean_rho": s.mean_rho,
            "max_rho": s.max_rho,
            "steps_rho_soft": s.steps_rho_soft,
            "worst_rank": s.worst_rank,
            "bit_identical": s.bit_identical,
        })
    };
    serde_json::json!({
        "relative_perplexity_pct": r.relative_perplexity_pct(),
        "worst_control_mean_kl_nats": r.worst_control_mean_kl(),
        "worst_control_max_step_kl_nats": r.worst_control_max_kl(),
        "pooled_control_mean_kl_nats": r.pooled_control_mean_kl(),
        "median_control_mean_kl_nats": r.median_control_mean_kl(),
        "visible_signal_floor_nats": r.floor,
        "controls_above_floor": r.controls_above_floor(),
        "controls_measured": r.control_count(),
        "signal_concentration": r.concentration_line(),
        "control_rows_worst_first": r.controls_worst_first().iter().map(|s| row(s)).collect::<Vec<_>>(),
        "open_ended_rows_worst_first_descriptive_only":
            r.open_ended_worst_first().iter().map(|s| row(s)).collect::<Vec<_>>(),
    })
}

fn write_artifact(model: &str, results: &[ArmResult]) {
    let rows: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            let d = &r.decision;
            serde_json::json!({
                "flip": d.flip,
                "model": d.model,
                "verdict": format!("{}", d.verdict),
                "accepted": d.accepted(),
                "bar": format!("{}", d.bar),
                "pass": d.evidence,
                "fail": d.failures,
                "descriptive": d.descriptive,
                "instrument": per_prompt_json(&r.instrument),
            })
        })
        .collect();
    let p = out_dir().join(format!("flip-verdicts-{model}.json"));
    let body = serde_json::json!({
        "bar_doc": FLIP_BAR_DOC,
        "instrument_doc": prompts::WHY_THE_INSTRUMENT_BLOCK_PRINTS_NEXT_TO_EVERY_VERDICT,
        "max_over_prompts_caveat": prompts::MAX_OVER_PROMPTS_CAVEAT,
        "kernel_vs_weight_format": KERNEL_FLIP_VS_WEIGHT_FORMAT_FLIP,
        "verdicts": rows,
    });
    if let Ok(s) = serde_json::to_vec_pretty(&body) {
        if std::fs::write(&p, s).is_ok() {
            eprintln!("[artifact] {}", p.display());
        }
    }
}

fn conclude(results: &[ArmResult]) {
    eprintln!("\n================ FLIP VERDICTS ================");
    eprintln!("{KERNEL_FLIP_VS_WEIGHT_FORMAT_FLIP}\n");
    for r in results {
        eprintln!("{}", r.decision);
        eprintln!("{}\n", r.instrument);
    }
    for r in results {
        let d = &r.decision;
        println!(
            "FLIP-VERDICT {} {} {} | {}",
            d.model,
            d.flip,
            d.verdict,
            r.instrument.headline()
        );
    }
    eprintln!("{MARGIN_TRAP}");
    eprintln!("{REPRODUCIBILITY_LIMIT}");

    let decisions: Vec<&FlipDecision> = results.iter().map(|r| &r.decision).collect();
    let void: Vec<&FlipDecision> = decisions
        .iter()
        .copied()
        .filter(|d| d.verdict == FlipVerdict::Void)
        .collect();
    assert!(
        void.is_empty(),
        "{} arm(s) came back VOID: the reference is broken, so nothing was measured. {}",
        void.len(),
        void.iter()
            .map(|d| format!("{}/{}: {}", d.model, d.flip, d.failures.join("; ")))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    if std::env::var("NV_FLIP_ASSERT_ACCEPT").ok().as_deref() == Some("1") {
        let rejected: Vec<&FlipDecision> = decisions
            .iter()
            .copied()
            .filter(|d| !d.accepted())
            .collect();
        assert!(
            rejected.is_empty(),
            "NV_FLIP_ASSERT_ACCEPT=1 and {} arm(s) REJECTED: {}",
            rejected.len(),
            rejected
                .iter()
                .map(|d| format!("{}/{}: {}", d.model, d.flip, d.failures.join("; ")))
                .collect::<Vec<_>>()
                .join(" | ")
        );
    } else {
        eprintln!(
            "NOTE: a REJECT is a measurement result, not a harness failure, so this test is green \
             either way. Set NV_FLIP_ASSERT_ACCEPT=1 to turn the bar into a landing gate once a \
             flip is proposed for default-on."
        );
    }
}

#[test]
#[ignore]
fn flip_quality_gemma4_31b_nvfp4() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !opt_in("NV_FLIP_EVAL_TEST") {
        return;
    }
    if !ctx_or_skip() {
        return;
    }
    eprintln!("{WHAT_THIS_DECIDES}");
    eprintln!("{FLIP_BAR_DOC}");
    scrub_and_report();

    let dir = prompts::gemma4_nvfp4_dir().expect("Gemma-4-31B-IT-NVFP4 snapshot");
    let pack = pack_or_refuse(&dir);
    let stops = pack.stop_set();
    let chosen = limit_prompts(ab_prompts(&pack));
    eprintln!("{}", describe_selection(&pack, &chosen));
    assert!(!chosen.is_empty(), "the pack contributed no prompt");

    let config = nv_models::gemma4::Gemma4Config::from_hf_json_file(&dir.join("config.json"))
        .expect("config");
    let t0 = std::time::Instant::now();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader)
        .expect("stage host weights");
    drop(loader);
    eprintln!("host weight staging {:.1}s", t0.elapsed().as_secs_f64());

    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let label_of = |t: u32| tokenizer.id_to_token(t).unwrap_or_else(|| format!("<{t}>"));
    let decode_text = |ids: &[u32]| tokenizer.decode(ids, false).unwrap_or_default();

    let max_steps = env_usize("NV_FLIP_STEPS", 96);
    let keep = env_usize("NV_FLIP_KL_STEPS", 24);
    let max_seq = env_usize("NV_FLIP_MAX_SEQ", 2048);
    eprintln!(
        "free-running window {max_steps} steps, forced-context distributional window {keep} steps \
         ({} MiB of retained reference logits at vocab {}), kv max_seq {max_seq}",
        chosen.len() * keep * config.vocab_size * 4 / (1 << 20),
        config.vocab_size
    );

    apply(&GEMMA_REFERENCE);
    let mut m = nv_models::gemma4_wgpu::Gemma4Wgpu::new(config.clone(), &host, max_seq)
        .expect("build reference");
    for k in SCRUBBED_KNOBS {
        std::env::remove_var(k);
    }
    let base_bytes = m.weight_bytes_per_token();
    eprintln!(
        "[{}] {} passes, {base_bytes} weight bytes/token",
        GEMMA_REFERENCE.label,
        m.pass_count()
    );

    let mut reference = ReferenceLeg {
        runs: Vec::new(),
        forced: Vec::new(),
        table: RunTable::new("Gemma-4-31B-IT-NVFP4 :: reference"),
    };
    for p in &chosen {
        let (rows, stopped, forced) =
            leg(&mut GemmaDecoder(&mut m), p, max_steps, &stops, keep, None)
                .expect("reference leg");
        let mut r = run_of(GEMMA_REFERENCE.label, p, &rows, stopped);
        r.text = decode_text(&r.tokens);
        eprintln!(
            "[{}] {:<24} {} tokens, {}",
            GEMMA_REFERENCE.label,
            p.label,
            r.tokens.len(),
            sampler_note(&rows)
        );
        reference
            .table
            .push(ArmObservation::of(p, &r, &stops, max_steps, &label_of));
        reference.runs.push(r);
        reference.forced.push(forced);
    }
    drop(m);
    eprintln!("{}", reference.table);
    reference
        .table
        .assert_controls_terminated_for(GEMMA_REFERENCE.label)
        .expect("the reference arm must end its turn on every control prompt");

    let mut decisions: Vec<ArmResult> = Vec::new();
    for a in GEMMA_CANDIDATES.iter().filter(|a| arm_selected(a)) {
        apply(a);
        let v2_live = nv_kernels::wgpu_backend::WgpuContext::shared()
            .map(nv_models::gemma4_wgpu::nvfp4_v2_enabled)
            .unwrap_or(false);
        assert!(
            !a.expects_v2_route || v2_live,
            "[{}] declares the nvfp4 v2 route but nvfp4_v2_enabled() is false under its env. \
             {SILENT_NO_OP_IS_THE_WORST_OUTCOME}",
            a.label
        );
        let built = nv_models::gemma4_wgpu::Gemma4Wgpu::new(config.clone(), &host, max_seq);
        for k in SCRUBBED_KNOBS {
            std::env::remove_var(k);
        }
        let mut cm = match built {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[{}] SKIP build failed: {e}", a.label);
                continue;
            }
        };
        let bytes = cm.weight_bytes_per_token();
        eprintln!(
            "[{}] {} passes, {bytes} weight bytes/token ({:+.3}% vs reference), v2 route live {v2_live}",
            a.label,
            cm.pass_count(),
            100.0 * (bytes as f64 - base_bytes as f64) / base_bytes as f64
        );
        assert!(
            !a.expects_byte_change || bytes != base_bytes,
            "[{}] declares a per-token byte change but weight_bytes_per_token() is still \
             {base_bytes}. {SILENT_NO_OP_IS_THE_WORST_OUTCOME}",
            a.label
        );
        let mut decode = |p: &TemplatedPrompt, forced_ids: &[u32]| {
            leg(
                &mut GemmaDecoder(&mut cm),
                p,
                max_steps,
                &stops,
                keep,
                Some(forced_ids),
            )
        };
        let (result, _, _, _) = score_candidate(
            a.label,
            "gemma-4-31B-IT-NVFP4",
            GEMMA_REFERENCE.label,
            a.label,
            &chosen,
            &stops,
            &reference,
            &mut decode,
            &decode_text,
            &label_of,
            max_steps,
        )
        .expect("score candidate");
        drop(cm);
        decisions.push(result);
    }

    assert!(!decisions.is_empty(), "no candidate arm produced a verdict");
    write_artifact("gemma-4-31B-IT-NVFP4", &decisions);
    conclude(&decisions);
}

#[test]
#[ignore]
fn flip_quality_qwen36_moe_nvfp4() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !opt_in("NV_FLIP_EVAL_TEST") {
        return;
    }
    if !ctx_or_skip() {
        return;
    }
    eprintln!("{WHAT_THIS_DECIDES}");
    eprintln!("{FLIP_BAR_DOC}");
    scrub_and_report();

    let dir = qwen36_dir().expect("Qwen3.6-35B-A3B-NVFP4 snapshot; set NV_QWEN36_DIR");
    eprintln!("[snapshot] {}", dir.display());
    let pack = pack_or_refuse(&dir);
    let stops = pack.stop_set();
    let chosen = limit_prompts(ab_prompts(&pack));
    eprintln!("{}", describe_selection(&pack, &chosen));
    assert!(!chosen.is_empty(), "the pack contributed no prompt");

    let config =
        nv_models::qwen3_5_moe::Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json"))
            .expect("config");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let label_of = |t: u32| tokenizer.id_to_token(t).unwrap_or_else(|| format!("<{t}>"));
    let decode_text = |ids: &[u32]| tokenizer.decode(ids, false).unwrap_or_default();

    let max_steps = env_usize("NV_FLIP_STEPS", 96);
    let keep = env_usize("NV_FLIP_KL_STEPS", 24);
    let max_seq = env_usize("NV_FLIP_MAX_SEQ", 2048);

    apply(&QWEN_REFERENCE);
    let mut m =
        nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu::from_loader(config.clone(), &loader, max_seq)
            .expect("build reference");
    for k in SCRUBBED_KNOBS {
        std::env::remove_var(k);
    }
    let ref_passes = m.pass_count();
    eprintln!("[{}] {ref_passes} passes", QWEN_REFERENCE.label);
    assert_router_arm(&m, QWEN_REFERENCE.label, true);

    let mut reference = ReferenceLeg {
        runs: Vec::new(),
        forced: Vec::new(),
        table: RunTable::new("Qwen3.6-35B-A3B-NVFP4 :: reference"),
    };
    for p in &chosen {
        let (rows, stopped, forced) =
            leg(&mut QwenDecoder(&mut m), p, max_steps, &stops, keep, None).expect("reference leg");
        let mut r = run_of(QWEN_REFERENCE.label, p, &rows, stopped);
        r.text = decode_text(&r.tokens);
        eprintln!(
            "[{}] {:<24} {} tokens, {}",
            QWEN_REFERENCE.label,
            p.label,
            r.tokens.len(),
            sampler_note(&rows)
        );
        reference
            .table
            .push(ArmObservation::of(p, &r, &stops, max_steps, &label_of));
        reference.runs.push(r);
        reference.forced.push(forced);
    }
    drop(m);
    eprintln!("{}", reference.table);
    reference
        .table
        .assert_controls_terminated_for(QWEN_REFERENCE.label)
        .expect("the reference arm must end its turn on every control prompt");

    let mut decisions: Vec<ArmResult> = Vec::new();
    for a in QWEN_CANDIDATES.iter().filter(|a| arm_selected(a)) {
        apply(a);
        let par_expected = nv_models::qwen3_5_moe_wgpu::router_par_enabled();
        let v2_live = nv_kernels::wgpu_backend::WgpuContext::shared()
            .map(nv_models::qwen3_5_moe_wgpu::nvfp4_v2_enabled)
            .unwrap_or(false);
        assert!(
            !a.expects_v2_route || v2_live,
            "[{}] declares the nvfp4 v2 route but nvfp4_v2_enabled() is false under its env. \
             {SILENT_NO_OP_IS_THE_WORST_OUTCOME}",
            a.label
        );
        let built = nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu::from_loader(
            config.clone(),
            &loader,
            max_seq,
        );
        for k in SCRUBBED_KNOBS {
            std::env::remove_var(k);
        }
        let mut cm = match built {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[{}] SKIP build failed: {e}", a.label);
                continue;
            }
        };
        eprintln!(
            "[{}] {} passes ({} reference)",
            a.label,
            cm.pass_count(),
            ref_passes
        );
        assert_router_arm(&cm, a.label, par_expected);
        assert!(
            !a.label.starts_with("R2-revert") || !par_expected,
            "[{}] is the serial-router revert but router_par_enabled() is still true under its \
             env, so it would measure the default against itself. \
             {SILENT_NO_OP_IS_THE_WORST_OUTCOME}",
            a.label
        );
        assert!(
            !a.label.starts_with("R3-revert") || !v2_live,
            "[{}] is the nvfp4-v1 revert but nvfp4_v2_enabled() is still true under its env, so \
             it would measure the default against itself. {SILENT_NO_OP_IS_THE_WORST_OUTCOME}",
            a.label
        );
        assert!(
            !a.label.starts_with("W8") || cm.pass_count() != ref_passes,
            "[{}] the int8-experts path removes the activation-quant passes, so an unchanged              pass count means the flag never reached the builder.              {SILENT_NO_OP_IS_THE_WORST_OUTCOME}",
            a.label
        );
        let mut decode = |p: &TemplatedPrompt, forced_ids: &[u32]| {
            leg(
                &mut QwenDecoder(&mut cm),
                p,
                max_steps,
                &stops,
                keep,
                Some(forced_ids),
            )
        };
        let (result, _, dists, _) = score_candidate(
            a.label,
            "Qwen3.6-35B-A3B-NVFP4",
            QWEN_REFERENCE.label,
            a.label,
            &chosen,
            &stops,
            &reference,
            &mut decode,
            &decode_text,
            &label_of,
            max_steps,
        )
        .expect("score candidate");
        let bit_identical = dists
            .iter()
            .filter(|d| d.prompt_kind == PromptKind::Control && d.steps > 0)
            .all(|d| d.bit_identical);
        eprintln!(
            "[{}] byte-identical across every retained control logit word: {bit_identical}. The \
             in-tree claim is that Qwen3.6 is byte-identical both ways; this is the end-to-end \
             check of it, and the routing test wgpu_qwen35_nvfp4_v2_routing.rs is the per-kernel \
             one.",
            a.label
        );
        drop(cm);
        decisions.push(result);
    }

    assert!(!decisions.is_empty(), "no candidate arm produced a verdict");
    write_artifact("Qwen3.6-35B-A3B-NVFP4", &decisions);
    conclude(&decisions);
}

fn assert_router_arm(
    m: &nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu,
    label: &str,
    expect_parallel: bool,
) {
    let (par, total) = m.router_parallel_layers();
    eprintln!("[{label}] router top-k: {par}/{total} layers parallel");
    assert!(total > 0, "[{label}] the graph has no router pass at all");
    let want = if expect_parallel { total } else { 0 };
    assert_eq!(
        par, want,
        "[{label}] expected {want}/{total} layers on the parallel router, built {par}. \
         {SILENT_NO_OP_IS_THE_WORST_OUTCOME}"
    );
}

fn qwen36_dir() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("NV_QWEN36_DIR") {
        let p = PathBuf::from(v);
        if p.join("config.json").exists() {
            return Some(p);
        }
        eprintln!("NV_QWEN36_DIR={} has no config.json", p.display());
        return None;
    }
    prompts::snapshots_of("RedHatAI/Qwen3.6-35B-A3B-NVFP4")
        .into_iter()
        .map(|(_, p)| p)
        .find(|p| p.join("chat_template.jinja").exists())
}

#[test]
fn the_flips_are_opt_in_and_the_reference_arm_pins_the_shipping_defaults() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert!(
        GEMMA_REFERENCE.env.is_empty(),
        "the reference arm must set no knob"
    );
    assert!(
        QWEN_REFERENCE.env.is_empty(),
        "the reference arm must set no knob"
    );
    for a in GEMMA_CANDIDATES.iter().chain(QWEN_CANDIDATES.iter()) {
        assert!(!a.env.is_empty(), "candidate arm {} sets nothing", a.label);

        let label_guarded = a.label.starts_with("W8")
            || a.label.starts_with("R2-revert")
            || a.label.starts_with("R3-revert");
        assert!(
            a.expects_v2_route || a.expects_byte_change || label_guarded,
            "candidate arm {} declares no observable effect, so a silent no-op would report \
             ACCEPT (BIT-IDENTICAL). {SILENT_NO_OP_IS_THE_WORST_OUTCOME}",
            a.label
        );
        for (k, _) in a.env {
            assert!(
                SCRUBBED_KNOBS.contains(k),
                "candidate arm {} sets {k}, which the reference arm does not scrub, so an ambient \
                 {k} would leak into the reference and both arms would measure the same thing",
                a.label
            );
        }
    }
    #[allow(clippy::assertions_on_constants)]
    {
        assert!(
            !nv_models::gemma4_wgpu::NVFP4_V2_DEFAULT_ON,
            "gemma4_wgpu::NVFP4_V2_DEFAULT_ON is the compiled half of F1 and it is already true, \
             so the reference arm is running the candidate config with the knob scrubbed and F1 \
             would compare a config against itself. {SILENT_NO_OP_IS_THE_WORST_OUTCOME}"
        );
    }
    for k in SCRUBBED_KNOBS {
        std::env::remove_var(k);
    }
    let Ok(ctx) = nv_kernels::wgpu_backend::WgpuContext::shared() else {
        eprintln!("no adapter; the env-shape assertions above still ran");
        return;
    };
    assert!(
        !nv_models::gemma4_wgpu::nvfp4_v2_enabled(ctx),
        "NV_WGPU_NVFP4_V2 is already default-on; there is nothing to flip and this harness would \
         be comparing a config against itself"
    );

    assert!(
        nv_models::qwen3_5_moe_wgpu::nvfp4_v2_enabled(ctx),
        "Qwen's nvfp4 v2 route is not reachable, so R3-revert reverts to the shipping arm and \
         this harness would be comparing a config against itself"
    );
    std::env::set_var("NV_WGPU_NVFP4_V2", "1");
    let on = nv_models::gemma4_wgpu::nvfp4_v2_enabled(ctx);
    std::env::remove_var("NV_WGPU_NVFP4_V2");
    eprintln!(
        "NV_WGPU_NVFP4_V2=1 reaches the gemma4 route on this adapter: {on} (false means the \
         adapter's subgroup width is not 32 and F1 is unmeasurable here, not that it failed)"
    );
}

struct Scripted {
    vocab: usize,
    pos: usize,
    terminate_after: usize,
    stop_id: u32,
    near_tie_at: usize,
    runner_up_bump: f32,
    invert_at: Option<usize>,
}

impl Scripted {
    fn new(vocab: usize, stop_id: u32) -> Self {
        Self {
            vocab,
            pos: 0,
            terminate_after: usize::MAX,
            stop_id,
            near_tie_at: usize::MAX,
            runner_up_bump: 0.0,
            invert_at: None,
        }
    }
}

impl Decoder for Scripted {
    fn reset(&mut self) -> anyhow::Result<()> {
        self.pos = 0;
        Ok(())
    }
    fn step(&mut self, token: u32) -> anyhow::Result<(u32, Vec<f32>)> {
        self.pos += 1;
        let mut l = vec![-8.0f32; self.vocab];
        if self.pos > self.terminate_after {
            l[self.stop_id as usize] = 14.0;
            l[(self.stop_id as usize + 1) % self.vocab] = 0.0;
            return Ok((self.stop_id, l));
        }
        let peak = (token as usize).wrapping_mul(2654435761) % (self.vocab - 8) + 4;
        let runner = (peak + 1) % (self.vocab - 8) + 4;
        let margin = if self.pos == self.near_tie_at {
            0.05
        } else {
            14.0
        };
        l[peak] = margin;
        l[runner] = 0.0;
        if self.invert_at == Some(self.pos) {
            l[runner] = margin + 1.0;
        } else {
            l[runner] += self.runner_up_bump.min(margin - 0.01);
        }
        let top = prompts::top2(&l).0;
        Ok((top, l))
    }
}

fn selftest_pack() -> PromptPack {
    if let Ok(dir) = prompts::gemma4_nvfp4_dir() {
        if let Ok((p, pack)) = resolve_pack(&dir) {
            eprintln!("[selftest] driving the REAL pack {}", p.display());
            return pack;
        }
    }
    eprintln!(
        "[selftest] no cached snapshot/pack; using an in-memory fixture pack. The plumbing under \
         test is identical; only the prompt ids are fabricated."
    );
    let mk = |label: &str, kind: PromptKind, ids: Vec<u32>| {
        TemplatedPrompt::from_official_render(
            label,
            kind,
            "selftest/model",
            "selftest",
            "<synthetic-selftest>",
            0,
            "<fixture>".into(),
            ids,
        )
    };
    PromptPack {
        model_repo: "selftest/model".into(),
        snapshot: "selftest".into(),
        template_digest: "<synthetic-selftest>".into(),
        template_bytes: 0,
        stop_ids: vec![4095],
        stop_source: "selftest".into(),
        prompts: vec![
            mk(
                "control-arithmetic",
                PromptKind::Control,
                vec![2, 11, 12, 13],
            ),
            mk("control-capital", PromptKind::Control, vec![2, 21, 22, 23]),
            mk(
                "openended-explain",
                PromptKind::OpenEnded,
                vec![2, 31, 32, 33],
            ),
        ],
    }
}

#[test]
fn the_whole_scoring_pipeline_runs_without_a_gpu_and_separates_the_three_outcomes() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let pack = selftest_pack();
    let stops = StopSet {
        ids: vec![4095],
        source: "selftest".into(),
    };
    let chosen: Vec<&TemplatedPrompt> = pack
        .prompts
        .iter()
        .filter(|p| p.kind == PromptKind::Control)
        .take(3)
        .chain(
            pack.prompts
                .iter()
                .filter(|p| p.kind == PromptKind::OpenEnded)
                .take(1),
        )
        .collect();
    assert!(
        chosen.len() >= 3,
        "selftest needs controls and one open-ended row"
    );

    let vocab = 4096usize;
    let max_steps = 24usize;
    let keep = 24usize;
    let decode_text = |ids: &[u32]| {
        ids.iter()
            .map(|t| format!("w{t}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let label_of = |t: u32| format!("<{t}>");

    let mut reference = ReferenceLeg {
        runs: Vec::new(),
        forced: Vec::new(),
        table: RunTable::new("selftest reference"),
    };
    for p in &chosen {
        let mut dec = Scripted::new(vocab, 4095);
        dec.terminate_after = p.ids.len() + 8;
        dec.near_tie_at = p.ids.len() + 3;
        let (rows, stopped, forced) =
            leg(&mut dec, p, max_steps, &stops, keep, None).expect("selftest reference leg");
        let mut r = run_of("reference", p, &rows, stopped);
        r.text = decode_text(&r.tokens);
        reference
            .table
            .push(ArmObservation::of(p, &r, &stops, max_steps, &label_of));
        reference.runs.push(r);
        reference.forced.push(forced);
    }
    reference
        .table
        .assert_controls_terminated_for("reference")
        .expect("the scripted reference must end its turn");

    let arms: [(&str, f32, Option<usize>, FlipVerdict); 3] = [
        (
            "selftest:identical",
            0.0,
            None,
            FlipVerdict::AcceptBitIdentical,
        ),
        ("selftest:margin-eater", 9.0, None, FlipVerdict::Reject),
        ("selftest:answer-changer", 0.0, Some(4), FlipVerdict::Reject),
    ];
    let mut decisions = Vec::new();
    for (label, bump, invert, want) in arms {
        let mut decode = |p: &TemplatedPrompt, forced_ids: &[u32]| {
            let mut dec = Scripted::new(vocab, 4095);
            dec.terminate_after = p.ids.len() + 8;
            dec.near_tie_at = p.ids.len() + 3;
            dec.runner_up_bump = bump;
            dec.invert_at = invert.map(|i| p.ids.len() + i);
            leg(&mut dec, p, max_steps, &stops, keep, Some(forced_ids))
        };
        let (result, _, _, _) = score_candidate(
            label,
            "selftest",
            "reference",
            label,
            &chosen,
            &stops,
            &reference,
            &mut decode,
            &decode_text,
            &label_of,
            max_steps,
        )
        .expect("selftest scoring");
        let decision = &result.decision;
        eprintln!("{decision}");
        eprintln!("{}", result.instrument);
        assert_eq!(
            decision.verdict, want,
            "{label} expected {want} but got {} ({:?})",
            decision.verdict, decision.failures
        );
        let block = format!("{}", result.instrument);
        for must in [
            "RELATIVE PERPLEXITY",
            "CONTROL rows (A/B EVIDENCE)",
            "OPEN-ENDED rows (DESCRIPTIVE ONLY",
            "SIGNAL CONCENTRATION",
            "MAX-OVER-PROMPTS CAVEAT",
            "control prompts at or above each decade",
        ] {
            assert!(
                block.contains(must),
                "the instrument block printed next to the {label} verdict is missing {must:?}. \
                 Every verdict must carry the relative perplexity and the per-prompt KL table, or \
                 the reader cannot tell a 1e-4 REJECT from a 1e-1 one.\n{block}"
            );
        }
        assert!(
            result.instrument.headline().contains("rel-ppl"),
            "the machine-readable FLIP-VERDICT line must carry the perplexity: {}",
            result.instrument.headline()
        );
        decisions.push(result);
    }

    let eater = &decisions[1].decision;
    assert!(
        eater.failures.iter().any(|f| f.starts_with("G3b")),
        "the margin-eater arm must be caught by the distributional gate, not by token agreement: \
         {:?}",
        eater.failures
    );
    assert!(
        eater
            .evidence
            .iter()
            .any(|e| e.contains("greedy token-exact on") && !e.contains("token-exact on 0/")),
        "the margin-eater arm is supposed to keep its tokens; if it did not, this fixture no \
         longer proves the point: {:?}",
        eater.evidence
    );
    let changer = &decisions[2].decision;
    assert!(
        changer.failures.iter().any(|f| f.starts_with("G1b"))
            || changer.failures.iter().any(|f| f.starts_with("G2")),
        "the answer-changer arm must fail a control gate: {:?}",
        changer.failures
    );
    let identical = &decisions[0].instrument;
    assert_eq!(
        identical.controls_above_floor(),
        0,
        "a bit-identical arm must carry no prompt above the visible-signal floor, or the floor is \
         measuring float noise rather than signal: {}",
        identical.concentration_line()
    );
    assert!(
        identical.relative_perplexity_pct().abs() < 1e-9,
        "a bit-identical arm must read 0% relative perplexity: {}",
        identical.relative_perplexity_pct()
    );
    write_artifact("selftest", &decisions);
}

#[test]
fn the_bar_and_what_it_decides_are_printed_by_the_harness_itself() {
    eprintln!("{WHAT_THIS_DECIDES}");
    eprintln!("{FLIP_BAR_DOC}");
    eprintln!("{}", FlipBar::from_env());
    assert!(WHAT_THIS_DECIDES.contains("NV_WGPU_NVFP4_V2"));
    assert!(WHAT_THIS_DECIDES.contains("NV_WGPU_LMHEAD_INT8"));
    assert!(WHAT_THIS_DECIDES.contains("NO TIMING IS COLLECTED HERE"));
    assert!(FLIP_BAR_DOC.contains("Why not bit-identical"));
}

struct E4bDecoder<'a>(&'a mut nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu);

impl Decoder for E4bDecoder<'_> {
    fn reset(&mut self) -> anyhow::Result<()> {
        self.0.reset();
        Ok(())
    }
    fn step(&mut self, token: u32) -> anyhow::Result<(u32, Vec<f32>)> {
        self.0.decode_step_logits(token)
    }
}

fn e4b_bf16_dir() -> anyhow::Result<std::path::PathBuf> {
    if let Ok(d) = std::env::var("NV_E4B_DIR") {
        return Ok(std::path::PathBuf::from(d));
    }
    let home = std::env::var("HOME")?;
    let base = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots");
    std::fs::read_dir(&base)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no gemma-4-E4B-it snapshot with config.json under {}",
                base.display()
            )
        })
}

const E4B_W8_REFERENCE: Arm = Arm {
    label: "bf16-default",
    env: &[],
    expects_v2_route: false,
    expects_byte_change: false,
};

const E4B_W8_CANDIDATES: [Arm; 2] = [
    Arm {
        label: "W8:int8-rowscale",
        env: &[("NV_E4B_WGPU_W8", "1")],
        expects_v2_route: false,
        expects_byte_change: true,
    },
    Arm {
        label: "W8G128:int8-group128",
        env: &[("NV_E4B_WGPU_W8", "1"), ("NV_E4B_WGPU_W8_GROUP", "128")],
        expects_v2_route: false,
        expects_byte_change: true,
    },
];

#[test]
#[ignore]
fn flip_quality_e4b_int8() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !opt_in("NV_FLIP_EVAL_TEST") {
        return;
    }
    if !ctx_or_skip() {
        return;
    }
    eprintln!("{FLIP_BAR_DOC}");
    scrub_and_report();

    let dir = e4b_bf16_dir().expect("gemma-4-E4B-it snapshot");
    let pack = pack_or_refuse(&dir);
    let stops = pack.stop_set();
    let chosen = limit_prompts(ab_prompts(&pack));
    eprintln!("{}", describe_selection(&pack, &chosen));
    assert!(!chosen.is_empty(), "the pack contributed no prompt");

    let config = nv_models::gemma4::Gemma4Config::from_hf_json_file(&dir.join("config.json"))
        .expect("config");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");

    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let label_of = |t: u32| tokenizer.id_to_token(t).unwrap_or_else(|| format!("<{t}>"));
    let decode_text = |ids: &[u32]| tokenizer.decode(ids, false).unwrap_or_default();

    let max_steps = env_usize("NV_FLIP_STEPS", 96);
    let keep = env_usize("NV_FLIP_KL_STEPS", 24);
    let max_seq = env_usize("NV_FLIP_MAX_SEQ", 2048);

    apply(&E4B_W8_REFERENCE);
    let mut m =
        nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu::from_loader(config.clone(), &loader, max_seq)
            .expect("build reference");
    for k in SCRUBBED_KNOBS {
        std::env::remove_var(k);
    }
    let base_bytes = m.weight_bytes_per_token();
    eprintln!(
        "[{}] {} passes, {base_bytes} weight bytes/token",
        E4B_W8_REFERENCE.label,
        m.pass_count()
    );

    let mut reference = ReferenceLeg {
        runs: Vec::new(),
        forced: Vec::new(),
        table: RunTable::new("gemma-4-E4B-it :: reference"),
    };
    for p in &chosen {
        let (rows, stopped, forced) =
            leg(&mut E4bDecoder(&mut m), p, max_steps, &stops, keep, None).expect("reference leg");
        let mut r = run_of(E4B_W8_REFERENCE.label, p, &rows, stopped);
        r.text = decode_text(&r.tokens);
        eprintln!(
            "[{}] {:<24} {} tokens, {}",
            E4B_W8_REFERENCE.label,
            p.label,
            r.tokens.len(),
            sampler_note(&rows)
        );
        reference
            .table
            .push(ArmObservation::of(p, &r, &stops, max_steps, &label_of));
        reference.runs.push(r);
        reference.forced.push(forced);
    }
    drop(m);
    eprintln!("{}", reference.table);
    reference
        .table
        .assert_controls_terminated_for(E4B_W8_REFERENCE.label)
        .expect("the reference arm must end its turn on every control prompt");

    let mut decisions: Vec<ArmResult> = Vec::new();
    for a in E4B_W8_CANDIDATES.iter().filter(|a| arm_selected(a)) {
        apply(a);
        let built = nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu::from_loader(
            config.clone(),
            &loader,
            max_seq,
        );
        for k in SCRUBBED_KNOBS {
            std::env::remove_var(k);
        }
        let mut cm = match built {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[{}] SKIP build failed: {e}", a.label);
                continue;
            }
        };
        let bytes = cm.weight_bytes_per_token();
        eprintln!(
            "[{}] {} passes, {bytes} weight bytes/token ({:+.3}% vs reference)",
            a.label,
            cm.pass_count(),
            100.0 * (bytes as f64 - base_bytes as f64) / base_bytes as f64
        );
        assert!(
            !a.expects_byte_change || bytes != base_bytes,
            "[{}] declares a per-token byte change but weight_bytes_per_token() is still \
             {base_bytes}. {SILENT_NO_OP_IS_THE_WORST_OUTCOME}",
            a.label
        );
        let mut decode = |p: &TemplatedPrompt, forced_ids: &[u32]| {
            leg(
                &mut E4bDecoder(&mut cm),
                p,
                max_steps,
                &stops,
                keep,
                Some(forced_ids),
            )
        };
        let (result, _, _, _) = score_candidate(
            "W8",
            "gemma-4-E4B-it",
            E4B_W8_REFERENCE.label,
            a.label,
            &chosen,
            &stops,
            &reference,
            &mut decode,
            &decode_text,
            &label_of,
            max_steps,
        )
        .expect("score candidate");
        drop(cm);
        decisions.push(result);
    }

    assert!(!decisions.is_empty(), "no candidate arm produced a verdict");
    write_artifact("gemma-4-E4B-it", &decisions);
    conclude(&decisions);
}
