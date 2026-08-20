#![cfg(feature = "wgpu")]
#![allow(dead_code)]

#[path = "fp8_contract_prompts.rs"]
mod prompts;

use std::path::PathBuf;
use std::sync::Mutex;

use prompts::{
    ab_prompts, compare, decide_flip, describe_selection, ArmObservation, DistributionalSummary,
    FlipBar, FlipDecision, FreeRun, InstrumentReport, PromptKind, PromptPack, RunTable, StopReason,
    StopSet, SuiteReport, TemplatedPrompt, FLIP_BAR_DOC, REPRODUCIBILITY_LIMIT,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

const KNOBS: [&str; 4] = [
    "NV_Q3_WGPU_W8_DELTA",
    "NV_Q3_WGPU_W8_LMHEAD",
    "NV_Q3_WGPU_W8_PROJ_GROUP",
    "NV_Q3_WGPU_FUSE_GATEUP",
];

struct Arm {
    label: &'static str,
    env: &'static [(&'static str, &'static str)],

    expect_converted_min: usize,

    must_be_a_no_op: bool,
}

const CANDIDATES: [Arm; 7] = [
    Arm {
        label: "NULL:same-config",
        env: &[],
        expect_converted_min: 0,
        must_be_a_no_op: true,
    },
    Arm {
        label: "GUOFF:gate-up-split",
        env: &[("NV_Q3_WGPU_FUSE_GATEUP", "0")],
        expect_converted_min: 0,
        must_be_a_no_op: true,
    },
    Arm {
        label: "W8D:delta-proj-int8",
        env: &[("NV_Q3_WGPU_W8_DELTA", "1")],
        expect_converted_min: 60,
        must_be_a_no_op: false,
    },
    Arm {
        label: "W8H:lmhead-int8",
        env: &[("NV_Q3_WGPU_W8_LMHEAD", "1")],
        expect_converted_min: 1,
        must_be_a_no_op: false,
    },
    Arm {
        label: "W8D+W8H",
        env: &[("NV_Q3_WGPU_W8_DELTA", "1"), ("NV_Q3_WGPU_W8_LMHEAD", "1")],
        expect_converted_min: 61,
        must_be_a_no_op: false,
    },
    Arm {
        label: "W8DO:delta-out-only",
        env: &[("NV_Q3_WGPU_W8_DELTA", "out")],
        expect_converted_min: 30,
        must_be_a_no_op: false,
    },
    Arm {
        label: "W8DI:delta-in-only",
        env: &[("NV_Q3_WGPU_W8_DELTA", "in")],
        expect_converted_min: 30,
        must_be_a_no_op: false,
    },
];

pub const SILENT_NO_OP: &str = "\
A flip that does not reach the code silently reports ACCEPT (BIT-IDENTICAL), because the two arms
really are the same binary. Every arm here therefore asserts on the builder's own counter --
Qwen3MoeWgpu::int8_projection_bytes() -- which records how many projections were converted and how
many bytes changed while the graph was being built, not on the env var the test just set.";

fn apply(env: &[(&str, &str)]) {
    for k in KNOBS {
        std::env::remove_var(k);
    }
    for (k, v) in env {
        std::env::set_var(k, v);
    }
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(d)
}

fn opt_in(var: &str) -> bool {
    if std::env::var(var).ok().as_deref() == Some("1") {
        return true;
    }
    eprintln!("skipping: set {var}=1 to run (real-weight test)");
    false
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
        .find(|p| p.join("config.json").exists())
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

fn prefill(
    m: &mut nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu,
    prompt: &[u32],
) -> anyhow::Result<(u32, Vec<f32>)> {
    m.reset()?;
    let mut logits = Vec::new();
    let mut gpu = 0u32;
    for t in prompt {
        let (tok, l) = m.decode_step_logits(*t)?;
        gpu = tok;
        logits = l;
    }
    anyhow::ensure!(!logits.is_empty(), "empty logits after prefill");
    Ok((gpu, logits))
}

fn free_run(
    m: &mut nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu,
    prompt: &[u32],
    max_steps: usize,
    stops: &StopSet,
) -> anyhow::Result<(Vec<StepRow>, Option<u32>)> {
    let (mut gpu, mut logits) = prefill(m, prompt)?;
    let mut out = Vec::with_capacity(max_steps);
    let mut stopped = None;
    for _ in 0..max_steps {
        let (cpu_top1, margin) = top2_of(&logits);
        out.push(StepRow {
            gpu,
            cpu_top1,
            margin,
            logits: None,
        });
        if stops.contains(gpu) {
            stopped = Some(gpu);
            break;
        }
        let (tok, l) = m.decode_step_logits(gpu)?;
        gpu = tok;
        logits = l;
    }
    Ok((out, stopped))
}

fn forced_run(
    m: &mut nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu,
    prompt: &[u32],
    forced: &[u32],
    keep: usize,
) -> anyhow::Result<Vec<StepRow>> {
    let (mut gpu, mut logits) = prefill(m, prompt)?;
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
            let (tok, l) = m.decode_step_logits(*want)?;
            gpu = tok;
            logits = l;
        }
    }
    Ok(out)
}

fn leg(
    m: &mut nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu,
    p: &TemplatedPrompt,
    max_steps: usize,
    stops: &StopSet,
    keep: usize,
    forced_ids: Option<&[u32]>,
) -> anyhow::Result<(Vec<StepRow>, Option<u32>, Vec<StepRow>)> {
    let (rows, stopped) = free_run(m, &p.ids, max_steps, stops)?;
    let ids: Vec<u32> = match forced_ids {
        Some(v) => v.to_vec(),
        None => rows.iter().map(|r| r.gpu).collect(),
    };
    let forced = forced_run(m, &p.ids, &ids, keep)?;
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

const THINK_CLOSE: &str = "</think>";

fn answer_span<'a>(text: &'a str, prompt: &str) -> Option<&'a str> {
    if let Some((_, a)) = text.rsplit_once(THINK_CLOSE) {
        return Some(a);
    }
    prompt.contains(THINK_CLOSE).then_some(text)
}

fn norm(s: &str) -> String {
    let mut out = String::new();
    let mut pending = false;
    for c in s.chars() {
        if c.is_whitespace() {
            pending = !out.is_empty();
        } else if c.is_alphanumeric() {
            if pending {
                out.push(' ');
                pending = false;
            }
            out.extend(c.to_lowercase());
        } else {
            pending = pending || !out.is_empty();
        }
    }
    out
}

fn drift_windows(kls: &[f64]) -> String {
    if kls.is_empty() {
        return "no retained steps".to_string();
    }
    let q = kls.len().div_ceil(4).max(1);
    let mut parts = Vec::new();
    for (i, w) in kls.chunks(q).enumerate() {
        let mean = w.iter().sum::<f64>() / w.len() as f64;
        let max = w.iter().cloned().fold(0.0f64, f64::max);
        parts.push(format!(
            "q{}[{}..{}] mean {:.3e} max {:.3e}",
            i + 1,
            i * q,
            i * q + w.len(),
            mean,
            max
        ));
    }
    parts.join(" | ")
}

fn ctx_or_panic() {
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared().expect(
        "no wgpu adapter: this battery decides a default flip, so it panics rather than \
                 printing a green `1 passed` that measured nothing",
    );
    let st = ctx.qualify();
    assert!(st.qualified, "adapter not qualified: {:?}", st.reason);
    eprintln!("{}", ctx.summary());
}

#[test]
#[ignore = "real weights: NV_FLIP_EVAL_TEST=1, ~22 GB checkpoint, minutes per arm"]
fn flip_quality_qwen36_bf16_projections_to_int8() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !opt_in("NV_FLIP_EVAL_TEST") {
        return;
    }
    ctx_or_panic();
    eprintln!("{FLIP_BAR_DOC}");
    eprintln!("{REPRODUCIBILITY_LIMIT}");
    eprintln!("{SILENT_NO_OP}");

    let dir = qwen36_dir().expect("Qwen3.6-35B-A3B-NVFP4 snapshot; set NV_QWEN36_DIR");
    eprintln!("[snapshot] {}", dir.display());
    let (_, pack): (PathBuf, PromptPack) = prompts::resolve_pack(&dir)
        .expect("prompt pack: this harness refuses hand-written prompts");
    let stops = pack.stop_set();
    let mut chosen = ab_prompts(&pack);
    if let Some(n) = std::env::var("NV_FLIP_PROMPTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        chosen.truncate(n);
    }
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

    let max_steps = env_usize("NV_FLIP_STEPS", 192);

    let keep = env_usize("NV_FLIP_KL_STEPS", 128);
    let max_seq = env_usize("NV_FLIP_MAX_SEQ", 2048);
    assert!(
        keep >= 128,
        "NV_FLIP_KL_STEPS={keep}: the DeltaNet state is carried across tokens, so a window \
         shorter than 128 steps cannot see accumulation and must not be used to accept this flip"
    );

    apply(&[]);
    let mut m =
        nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu::from_loader(config.clone(), &loader, max_seq)
            .expect("build reference");
    let ref_passes = m.pass_count();
    let ref_conv = m.int8_projection_bytes();
    eprintln!("[reference] {ref_passes} passes, int8 projections {ref_conv:?}");
    assert_eq!(
        ref_conv.0, 0,
        "the reference arm converted {} projections to int8, so the shipping default is not \
         what it claims to be and every candidate would be measured against the wrong baseline",
        ref_conv.0
    );

    let mut ref_runs: Vec<FreeRun> = Vec::new();
    let mut ref_forced: Vec<Vec<StepRow>> = Vec::new();
    let mut ref_table = RunTable::new("Qwen3.6-35B-A3B-NVFP4 :: reference (bf16 projections)");
    for p in &chosen {
        let (rows, stopped, forced) =
            leg(&mut m, p, max_steps, &stops, keep, None).expect("reference leg");
        let mut r = run_of("reference", p, &rows, stopped);
        r.text = decode_text(&r.tokens);
        eprintln!("[reference] {:<24} {} tokens", p.label, r.tokens.len());
        ref_table.push(ArmObservation::of(p, &r, &stops, max_steps, &label_of));
        ref_runs.push(r);
        ref_forced.push(forced);
    }
    drop(m);
    eprintln!("{ref_table}");
    ref_table
        .assert_controls_terminated_for("reference")
        .expect("the reference arm must end its turn on every control prompt");

    for (p, r) in chosen.iter().zip(&ref_runs) {
        let ans = answer_span(&r.text, &p.rendered).unwrap_or_else(|| {
            panic!(
                "{}: the reference completion opened {THINK_CLOSE}'s channel and never closed it, \
                 and the prompt did not pre-close it either, so the G1B-SPLIT rows below would \
                 silently compare whole completions and report nothing",
                p.label
            )
        });
        eprintln!(
            "[G1B-SPLIT reference] {:<24} completion {} chars, of which reasoning {} ({:.1}%), \
             answer {:?}",
            p.label,
            r.text.len(),
            r.text.len() - ans.len(),
            100.0 * (r.text.len() - ans.len()) as f64 / r.text.len().max(1) as f64,
            norm(ans)
        );
    }

    let only = std::env::var("NV_FLIP_ARMS").ok();
    let mut decisions: Vec<(FlipDecision, InstrumentReport)> = Vec::new();
    for a in CANDIDATES.iter().filter(|a| {
        only.as_deref()
            .is_none_or(|o| o.split(',').any(|t| a.label.starts_with(t.trim())))
    }) {
        apply(a.env);
        let built = nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu::from_loader(
            config.clone(),
            &loader,
            max_seq,
        );
        apply(&[]);
        let mut cm = match built {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[{}] SKIP build failed: {e}", a.label);
                continue;
            }
        };
        let conv = cm.int8_projection_bytes();
        eprintln!(
            "[{}] {} passes ({ref_passes} reference); int8 projections {} converted, \
             {:.4} -> {:.4} GB",
            a.label,
            cm.pass_count(),
            conv.0,
            conv.1 as f64 / 1e9,
            conv.2 as f64 / 1e9
        );
        if a.must_be_a_no_op {
            assert_eq!(
                conv,
                (0, 0, 0),
                "[{}] the null control converted something, so it is not a null control",
                a.label
            );
        } else {
            assert!(
                conv.0 >= a.expect_converted_min && conv.2 < conv.1,
                "[{}] the builder converted {} projections ({} -> {} bytes); expected at least {} \
                 and a strict byte reduction. {SILENT_NO_OP}",
                a.label,
                conv.0,
                conv.1,
                conv.2,
                a.expect_converted_min
            );
        }

        let mut suite = SuiteReport::new(
            &format!("{} on Qwen3.6-35B-A3B-NVFP4", a.label),
            "reference",
            a.label,
        );
        let mut dists: Vec<DistributionalSummary> = Vec::new();
        let mut split: Vec<String> = Vec::new();
        let (mut whole_diff, mut answer_diff) = (0usize, 0usize);
        let mut table = RunTable::new(&format!("Qwen3.6-35B-A3B-NVFP4 :: {}", a.label));
        for (i, p) in chosen.iter().enumerate() {
            let (rows, stopped, forced) = leg(
                &mut cm,
                p,
                max_steps,
                &stops,
                keep,
                Some(&ref_runs[i].tokens),
            )
            .expect("candidate leg");
            let mut cand = run_of(a.label, p, &rows, stopped);
            cand.text = decode_text(&cand.tokens);
            table.push(ArmObservation::of(p, &cand, &stops, max_steps, &label_of));

            let mut d = DistributionalSummary::new(&p.label, p.kind);
            let mut kls: Vec<f64> = Vec::new();
            for (r, c) in ref_forced[i].iter().zip(forced.iter()) {
                match (&r.logits, &c.logits) {
                    (Some(x), Some(y)) => {
                        let sd = prompts::step_delta(x, y);
                        kls.push(sd.kl_nats);
                        d.push(sd);
                    }
                    _ => break,
                }
            }
            eprintln!("[{}] {d}", a.label);
            eprintln!(
                "[{}] {:<24} DRIFT {}",
                a.label,
                p.label,
                drift_windows(&kls)
            );

            if p.kind == PromptKind::Control {
                let (rw, cw) = (norm(&ref_runs[i].text), norm(&cand.text));
                let ra = answer_span(&ref_runs[i].text, &p.rendered).map(norm);
                let ca = answer_span(&cand.text, &p.rendered).map(norm);
                whole_diff += usize::from(rw != cw);
                answer_diff += usize::from(ra != ca);
                split.push(format!(
                    "  {:<24} whole {:<9} answer {:<9} ref {:?} -> cand {:?}",
                    p.label,
                    if rw == cw { "SAME" } else { "CHANGED" },
                    match (&ra, &ca) {
                        (Some(x), Some(y)) if x == y => "SAME",
                        (Some(_), Some(_)) => "CHANGED",
                        _ => "UNCLOSED",
                    },
                    ra.as_deref().unwrap_or("<no </think>>"),
                    ca.as_deref().unwrap_or("<no </think>>"),
                ));
            }

            dists.push(d);
            suite.push(compare(p, &ref_runs[i], &cand));
        }
        eprintln!("{table}");
        eprintln!("{suite}");
        eprintln!(
            "G1B-SPLIT {} :: whole-completion mismatches {whole_diff}/{}, ANSWER mismatches \
             {answer_diff}/{}\n{}",
            a.label,
            split.len(),
            split.len(),
            split.join("\n")
        );
        let decision = decide_flip(
            a.label,
            "Qwen3.6-35B-A3B-NVFP4",
            FlipBar::from_env(),
            &suite,
            &dists,
        );
        let instrument =
            InstrumentReport::new(&format!("Qwen3.6-35B-A3B-NVFP4 :: {}", a.label), &dists);
        eprintln!("{decision}");
        eprintln!("{instrument}");
        drop(cm);
        decisions.push((decision, instrument));
    }

    assert!(!decisions.is_empty(), "no candidate arm produced a verdict");
    for (d, i) in &decisions {
        eprintln!("FLIP-VERDICT {} | {}", d.flip, i.headline());
    }
    let worst_control_mean_kl = |i: &InstrumentReport| {
        i.controls_worst_first()
            .first()
            .map(|s| s.mean_kl)
            .unwrap_or(f64::NAN)
    };
    for (d, i) in &decisions {
        eprintln!(
            "SUMMARY {:<22} verdict {} worst-control mean KL {:.4e}",
            d.flip,
            d.verdict,
            worst_control_mean_kl(i)
        );
    }
    if std::env::var("NV_FLIP_ASSERT_ACCEPT").ok().as_deref() == Some("1") {
        for (d, _) in &decisions {
            assert!(
                matches!(
                    d.verdict,
                    prompts::FlipVerdict::Accept | prompts::FlipVerdict::AcceptBitIdentical
                ),
                "{} REJECTED: {:?}",
                d.flip,
                d.failures
            );
        }
    }
}
