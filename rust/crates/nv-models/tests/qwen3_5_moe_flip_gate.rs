#![allow(dead_code)]

#[cfg(feature = "cuda")]
#[path = "../../../tests/common/chat_eval_core.rs"]
mod harness_self_test_no_server_code;

#[cfg(feature = "wgpu")]
#[path = "fp8_contract_prompts.rs"]
mod prompts;

#[cfg(feature = "cuda")]
mod grouped_moe_dispatch {
    use candle_core::{DType, Device, Tensor};
    use nv_models::qwen3_5_moe::{GroupedMoeDispatch, Qwen3Moe, Qwen3MoeConfig};
    use nv_weights::{QuantizationConfig, WeightLoader};
    use std::path::PathBuf;

    use crate::harness_self_test_no_server_code::{self, compare, free_running, PromptPack, SuiteReport};

    fn pack_path() -> Option<PathBuf> {
        std::env::var("NV_CHAT_EVAL_PACK").ok().map(PathBuf::from)
    }

    fn weights_dir() -> Option<PathBuf> {
        std::env::var("NV_CHAT_EVAL_WEIGHTS")
            .ok()
            .map(PathBuf::from)
    }

    struct Stepper<'a> {
        model: &'a Qwen3Moe,
        dispatch: Option<&'a GroupedMoeDispatch>,
        device: Device,
        cache: nv_models::qwen3_5_moe::Qwen3MoeKvCache,
        pos: usize,
    }

    impl<'a> Stepper<'a> {
        fn new(
            model: &'a Qwen3Moe,
            dispatch: Option<&'a GroupedMoeDispatch>,
            device: &Device,
            max_seq: usize,
        ) -> Self {
            Self {
                model,
                dispatch,
                device: device.clone(),
                cache: model.new_kv_cache(max_seq).expect("cache"),
                pos: 0,
            }
        }

        fn step(&mut self, t: u32) -> anyhow::Result<Vec<f32>> {
            let disp: Option<&dyn nv_models::qwen3_5_moe::MoeDispatch> = self
                .dispatch
                .map(|d| d as &dyn nv_models::qwen3_5_moe::MoeDispatch);
            let tt = Tensor::from_vec(vec![t], (1usize, 1usize), &self.device)?;
            let pp = Tensor::from_vec(vec![self.pos as i32], 1usize, &self.device)?;
            let logits =
                self.model
                    .forward_with_cache_dispatched(&tt, &pp, &mut self.cache, disp)?;
            self.pos += 1;
            Ok(logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
        }
    }

    #[test]
    #[ignore = "loads the ~20 GB Qwen3.6 NVFP4 checkpoint; set NV_QWEN_MOE_GATE=1"]
    fn qwen36_host_vs_grouped_moe_flip_gate() {
        if std::env::var("NV_QWEN_MOE_GATE").as_deref() != Ok("1") {
            panic!("set NV_QWEN_MOE_GATE=1 to run this GPU test (it must never silently skip)");
        }
        let (Some(pack_p), Some(dir)) = (pack_path(), weights_dir()) else {
            panic!("set NV_CHAT_EVAL_PACK and NV_CHAT_EVAL_WEIGHTS (see docs/book/08.1-quality-harness.md)");
        };
        let pack = PromptPack::load_for_snapshot(&pack_p, &dir).expect("pack/snapshot mismatch");
        let stops = pack.stop_set();
        let steps = std::env::var("NV_CHAT_EVAL_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(harness_self_test_no_server_code::DEFAULT_MAX_STEPS);
        eprintln!(
            "pack {} @ {} :: {} prompts ({} controls), {}, max_steps {steps}",
            pack.model_repo,
            pack.snapshot,
            pack.prompts.len(),
            pack.controls(),
            stops
        );

        let device = Device::new_cuda(0).expect("cuda");
        let cfg = Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
        let raw = std::fs::read_to_string(dir.join("config.json")).expect("read config");
        let qconfig = QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
        let weights = WeightLoader::open_dir(&dir, &device).expect("open weights");
        let model =
            Qwen3Moe::from_loader_quantized(cfg, &weights, &qconfig, &device).expect("build model");
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
        let dispatch = GroupedMoeDispatch::from_model(&model).expect("grouped dispatch");

        let longest = pack.prompts.iter().map(|p| p.ids.len()).max().unwrap();
        let max_seq = (longest + steps + 64).next_power_of_two().max(256);

        let mut refs = Vec::new();
        for p in &pack.prompts {
            let mut s = Stepper::new(&model, None, &device, max_seq);
            let mut r = free_running("host-moe", p, &stops, steps, |t| s.step(t)).unwrap();
            r.text = tok.decode(&r.tokens, false).unwrap_or_default();
            refs.push(r);
        }
        let mut cands = Vec::new();
        for p in &pack.prompts {
            let mut s = Stepper::new(&model, Some(&dispatch), &device, max_seq);
            let mut r = free_running("grouped-moe", p, &stops, steps, |t| s.step(t)).unwrap();
            r.text = tok.decode(&r.tokens, false).unwrap_or_default();
            cands.push(r);
        }

        let mut suite = SuiteReport::new(
            "Qwen3.6 host-MoE vs grouped-MoE (CUDA, #138 flip gate)",
            "host-moe",
            "grouped-moe",
        );
        for i in 0..pack.prompts.len() {
            suite.push(compare(&pack.prompts[i], &refs[i], &cands[i]));
        }
        suite.validate().unwrap();
        eprintln!("{suite}");
        suite.assert_controls_exact().unwrap();
    }
}

#[cfg(feature = "wgpu")]
mod nvfp4_v2_default_flip {
    use crate::prompts::{
        compare, decide_flip, relative_perplexity_pct, resolve_pack, step_delta, top2,
        ArmObservation, DistributionalSummary, FlipBar, FlipVerdict, FreeRun, InstrumentReport,
        PromptKind, PromptPack, RunTable, StopReason, StopSet, SuiteReport, TemplatedPrompt,
        FLIP_BAR_DOC, MARGIN_TRAP, REPRODUCIBILITY_LIMIT, WHY_A_PACK,
    };
    use nv_models::qwen3_5_moe::Qwen3MoeConfig;
    use nv_models::qwen3_5_moe_wgpu as q3w;
    use std::path::Path;
    use std::path::PathBuf;

    pub const WHAT_THIS_DECIDES: &str = "\
NV_Q3_WGPU_NVFP4_V2 on Qwen3.6-35B-A3B-NVFP4, as a DEFAULT. Speed is settled elsewhere (A/B/A,
1.166x against 0.44% baseline drift); this suite decides quality only and collects no timing.

The v2 route sends every nvfp4 GEMV to the pair-packed fdec/warp/fmlut kernels. They compute the
same function as the shipping kernel but sum k-blocks in a different order, so end to end the
result is not guaranteed bit-identical and a distributional bar is the only honest instrument.";

    pub const TWO_INSTRUMENTS: &str = "\
TWO INSTRUMENTS, BOTH PRINTED, BECAUSE ONE OF THEM CANNOT RESOLVE A KERNEL CHANGE ALONE.

  I1 ANSWER    forced-context KL over the reference's own free-running ANSWER. This is the bar the
               repo's other flips were decided on. Its weakness is structural: a control answer is
               two or three tokens long and the reference distribution there is nearly one-hot, so
               KL is small whatever the weights do. A 4-bit FFN once read +0.0004% on this
               instrument and +15% on the same prompts' own text -- the two disagreeing by 3035x.

  I2 PROMPT-TEXT  forced-context KL at EVERY position of the prompt's own tokens: real chat text,
               real prose, real code, scored against the same reference on the same conditioning.
               No free-running, no cascade, so it is deterministic and reproducible even on the
               open-ended prompts, and it is ~4x more positions of genuinely uncertain text.

The DISAGREEMENT between them is printed next to the verdict. A flip is accepted only if BOTH sit
under the bar; a large I2/I1 ratio with I1 passing is exactly the shape that has fooled this repo
before.";

    pub const WHY_A_NULL_ARM: &str = "\
THREE NULLS, BECAUSE ONE ARM-VS-ARM NUMBER DECIDES NOTHING HERE.

  N1 REFERENCE REBUILD  a second build of the reference env, scored through the identical path.
     Whatever it reports is the harness floor: a signal the null also produces decides nothing.
  N2 CANDIDATE REBUILD  a second build of the CANDIDATE env. A between-arm effect smaller than the
     candidate's disagreement with itself is the arm's own scatter, not the kernel. Only a repeat
     of the candidate can tell those apart, and the harness floor cannot stand in for it.
  N3 SAME-BUILD REPEAT  every teacher-forced pass re-run NV_QWEN36_FLIP_REPS times inside one
     build. N1 and N2 each cost a 22 GB build and buy one observation; N3 buys one per position
     per repetition, which is the only null here with the power to catch a rare event.";

    pub const WHAT_THIS_MEASURED: &str = "\
MEASURED 2026-08-10, Apple silicon (Ultra-class) / wgpu-Metal, RedHatAI/Qwen3.6-35B-A3B-NVFP4 @965bfb0e,
thinking-off pack, 11 control + 3 open-ended prompts, 5 processes, the last at
NV_QWEN36_FLIP_REPS=20 (31.6 min, 110/110 nvfp4 GEMVs on the v2 route in both candidate legs).

  bar (I2, the instrument that can resolve this)   worst-control mean KL 4.390e-4 nats (+0.0439%)
                                                   vs the 1.0e-3 bar, max step 7.501e-3 vs 2.0e-2
  bar (I1, short answers)                          worst-control mean KL 1.455e-5 nats (+0.0015%)
  instrument disagreement I2/I1, pooled            15.9x -- I1 alone could not have decided this
  free-running greedy text                         token-exact on 11/11 controls AND 3/3
                                                   open-ended, including a 96-token completion
  N1 reference rebuild                             bit-identical, 0/771 positions, 5 processes
  N2 candidate rebuild                             bit-identical, 0.000e0 over 14 prompts x 4
                                                   statistics, 4 of 5 processes
  N3 same-build repeat                             0 mismatches in 49,344 re-run positions on the
                                                   candidate and 49,344 on the reference

THE UNEXPLAINED EVENT, AND WHAT NOW BOUNDS IT. In one v2 leg of eleven, a single teacher-forced
position of control-prose-proverb read 9.452 nats of KL -- 1,260x this arm's own worst step and
21,500x its worst prompt mean -- pushing the reference argmax to rank 3 and inverting one decision.
It has never reproduced: not in the ten v2 legs since, and not in any same-build repeat.

The event is a TRANSIENT or it is DETERMINISTIC, and those two have different instruments, so quote
the matching denominator or the bound is not a bound:

  transient (a race, a leaked workgroup word, a scheduling artefact). N3 is the instrument: it
  re-runs the identical pass inside one build and compares every logit word. The eleven legs put
  the implied per-position rate at 1.26e-4; at that rate the 49,344 candidate re-runs expect 6.2
  events. Zero were seen, so the transient class is rejected at p ~ 2e-3.

  deterministic and input-dependent. N3 cannot see it -- a deterministic corruption reproduces in
  the repeat too -- and only between-arm exposure bounds it: 0 recurrences in the ten legs since,
  i.e. under 3.9e-4 per position at 95%.

Two named mechanisms are dead, both with bit-exact proofs at the ROUTED geometry rather than by
argument, in nv-kernels' wgpu_gemv_nvfp4_v2_routed_pk: workgroup zero-init being disabled on
q3w_gemv_nvfp4_{fdec,warp} (poison parity holds at wg 64 and 128, with a tripwire that moves), and
nv2_pk_bits[sgid + 1u] running off the end of the array (no shape this checkpoint routes gives an
odd subgroup count, and pk_capable is what forbids it). What is left is a hypothesis outside these
kernels.

Exposure also moved under the suspects at 8eb9ff466: the gate/up stack, 40 of the 110 nvfp4 GEMVs
per token and the largest nvfp4 byte term, left fdec for fmlut, which is neither on the no-zero-init
list nor a user of the cross-subgroup pack. That pack now carries 50 of 110 rather than 90. Every
observation of the event predates that commit; the bound above does not.

The bar itself is retired as a blocker: v2 passes it with 2.3x margin and reproduces exactly.";

    const KNOBS: [&str; 13] = [
        "NV_NVFP4_V2_SELECT",
        "NV_Q3_WGPU_NVFP4_V2",
        "NV_Q3_WGPU_ROUTER_PAR",
        "NV_Q3_WGPU_W8_EXPERTS",
        "NV_Q3_WGPU_W8_DELTA",
        "NV_Q3_WGPU_W8_LMHEAD",
        "NV_Q3_WGPU_W8_GROUP",
        "NV_Q3_WGPU_W8_PROJ_GROUP",
        "NV_Q3_WGPU_FUSE_GATEUP",
        "NV_Q3_WGPU_FUSE_SILU_QUANT",
        "NV_Q3_WGPU_FUSE_SHARED_EXPERT",
        "NV_Q3_WGPU_DELTA_UNROLL",
        "NV_WGPU_LMHEAD_INT8",
    ];

    struct Arm {
        label: &'static str,
        env: &'static [(&'static str, &'static str)],

        expect_all_v2: bool,

        expect_no_v2: bool,
    }

    const REFERENCE: Arm = Arm {
        label: "v1:shipping-nvfp4-gemv",
        env: &[("NV_Q3_WGPU_NVFP4_V2", "0")],
        expect_all_v2: false,
        expect_no_v2: true,
    };

    const CANDIDATES: [Arm; 3] = [
        Arm {
            label: "null:v1-rebuilt",
            env: &[("NV_Q3_WGPU_NVFP4_V2", "0")],
            expect_all_v2: false,
            expect_no_v2: true,
        },
        Arm {
            label: "v2:nvfp4-v2",
            env: &[("NV_Q3_WGPU_NVFP4_V2", "1")],
            expect_all_v2: true,
            expect_no_v2: false,
        },
        Arm {
            label: "v2rep:nvfp4-v2-rebuilt",
            env: &[("NV_Q3_WGPU_NVFP4_V2", "1")],
            expect_all_v2: true,
            expect_no_v2: false,
        },
    ];

    fn env_usize(k: &str, d: usize) -> usize {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    }

    fn scrub() {
        let ambient: Vec<String> = KNOBS
            .iter()
            .filter_map(|k| std::env::var(k).ok().map(|v| format!("{k}={v}")))
            .collect();
        if ambient.is_empty() {
            eprintln!("[env] no Qwen wgpu knob was set in the ambient environment");
        } else {
            eprintln!("[env] SCRUBBING ambient knobs so the reference is the shipping default, not this shell: {}", ambient.join(" "));
        }
        for k in KNOBS {
            std::env::remove_var(k);
        }

        for k in ["NV_WGPU_NOZI", "NV_WGPU_NOZI_NVFP4_V2", "NV_DETERMINISTIC"] {
            eprintln!(
                "[env] {k}={:?} (reported so a run is reproducible)",
                std::env::var(k).ok()
            );
        }
    }

    fn apply(a: &Arm) {
        for k in KNOBS {
            std::env::remove_var(k);
        }
        for (k, v) in a.env {
            std::env::set_var(k, v);
        }
        eprintln!(
            "[arm {}] env {}",
            a.label,
            if a.env.is_empty() {
                "<none: shipping defaults>".to_string()
            } else {
                a.env
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        );
    }

    fn qwen36_dir() -> PathBuf {
        if let Ok(v) = std::env::var("NV_QWEN36_DIR") {
            let p = PathBuf::from(v);
            assert!(
                p.join("config.json").is_file(),
                "NV_QWEN36_DIR={} has no config.json",
                p.display()
            );
            return p;
        }
        let snaps = crate::prompts::snapshots_of("RedHatAI/Qwen3.6-35B-A3B-NVFP4");
        let p = snaps
            .into_iter()
            .map(|(_, p)| p)
            .find(|p| p.join("chat_template.jinja").is_file())
            .expect(
                "no RedHatAI/Qwen3.6-35B-A3B-NVFP4 snapshot in any hub root; set NV_QWEN36_DIR",
            );
        assert!(
            p.join("model.safetensors").exists() || p.join("model.safetensors.index.json").exists(),
            "snapshot {} declares the model but has no weight bytes on this disk",
            p.display()
        );
        p
    }

    const THINKING_OPEN: &str = "<think>\n";
    const THINKING_CLOSED: &str = "<think>\n\n</think>\n\n";

    pub const WHY_THE_PACK_IS_RE_RENDERED: &str = "\
THINKING MODE IS THE FIRST THING THIS GATE HAS TO GET RIGHT. Qwen3.6's chat template ends its
generation prompt with an OPEN '<think>' unless enable_thinking is false, in which case it emits a
CLOSED empty thought block. The pack on disk was rendered with it open, and measured against this
checkpoint every one of the 11 control prompts then runs 96 tokens of \"Here's a thinking
process: 1. Analyze User Input...\" and never reaches its answer. That reference is a runaway: G0
voids the battery, and the previous Qwen flip that was rejected on answer-text equality alone was
rejected on a diff between two chains of thought.

So this suite re-renders the pack with the thinking block CLOSED, by substituting the template's
own two literals -- it does not invent a prompt string. Both literals are checked against the live
chat_template.jinja, the substitution is checked to have applied to every prompt, and the result is
re-tokenized with the checkpoint's own tokenizer.json and written out so it is inspectable.
NV_QWEN36_FLIP_THINKING=1 keeps the thinking-open pack, which is how the runaway above reproduces.";

    fn thinking_off_pack(src: &Path, dir: &Path, tok: &tokenizers::Tokenizer) -> PathBuf {
        let template = std::fs::read_to_string(dir.join("chat_template.jinja")).expect("template");
        assert!(
            template.contains("enable_thinking"),
            "the live template declares no thinking switch, so this substitution is not the \
             template's own behaviour"
        );
        assert!(
            template.contains("'<think>\\n\\n</think>\\n\\n'") && template.contains("'<think>\\n'"),
            "the live template does not contain both thinking literals; re-derive the substitution \
             from {}",
            dir.join("chat_template.jinja").display()
        );
        let mut pack = PromptPack::read_json(src).expect("read source pack");
        let (mut rewritten, mut already) = (0usize, 0usize);
        for p in pack.prompts.iter_mut() {
            if p.rendered.ends_with(THINKING_CLOSED) {
                already += 1;
                continue;
            }
            let head = p.rendered.strip_suffix(THINKING_OPEN).unwrap_or_else(|| {
                panic!(
                    "prompt {:?} ends with neither thinking literal, so this substitution is not \
                     the shape it was written for: {:?}",
                    p.label, p.rendered
                )
            });
            p.rendered = format!("{head}{THINKING_CLOSED}");
            p.ids = tok
                .encode(p.rendered.as_str(), false)
                .expect("re-tokenize")
                .get_ids()
                .to_vec();
            rewritten += 1;
        }
        assert_eq!(
            rewritten + already,
            pack.prompts.len(),
            "{rewritten} rewritten + {already} already-closed does not cover all {} prompts",
            pack.prompts.len()
        );
        eprintln!("[pack] {already} prompt(s) already had the thinking block closed");

        let out = src.with_file_name(format!(
            "thinking-off-{}.json",
            src.file_stem().unwrap_or_default().to_string_lossy()
        ));
        pack.write_json(&out).expect("write derived pack");
        eprintln!(
            "[pack] re-rendered {rewritten} prompts with the thinking block CLOSED -> {}\n  \
             example: {:?}",
            out.display(),
            pack.prompts[0].rendered
        );
        out
    }

    struct StepRow {
        gpu: u32,
        margin: f32,
    }

    fn free_run(
        m: &mut q3w::Qwen3MoeWgpu,
        ids: &[u32],
        max_steps: usize,
        stops: &StopSet,
    ) -> anyhow::Result<(Vec<StepRow>, Option<u32>)> {
        m.reset()?;
        let mut logits = Vec::new();
        let mut gpu = 0u32;
        for t in ids {
            let (tok, l) = m.decode_step_logits(*t)?;
            gpu = tok;
            logits = l;
        }
        anyhow::ensure!(!logits.is_empty(), "empty logits after prefill");
        let mut out = Vec::with_capacity(max_steps);
        let mut stopped = None;
        for _ in 0..max_steps {
            let (_, v1, _, v2) = top2(&logits);
            out.push(StepRow {
                gpu,
                margin: v1 - v2,
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

    fn forced_answer(
        m: &mut q3w::Qwen3MoeWgpu,
        ids: &[u32],
        forced: &[u32],
        keep: usize,
        mut sink: impl FnMut(usize, &[f32]),
    ) -> anyhow::Result<()> {
        m.reset()?;
        let mut logits = Vec::new();
        for t in ids {
            let (_, l) = m.decode_step_logits(*t)?;
            logits = l;
        }
        anyhow::ensure!(!logits.is_empty(), "empty logits after prefill");
        for (i, want) in forced.iter().enumerate() {
            if i >= keep {
                break;
            }
            sink(i, &logits);
            if i + 1 >= keep || i + 1 >= forced.len() {
                break;
            }
            let (_, l) = m.decode_step_logits(*want)?;
            logits = l;
        }
        Ok(())
    }

    fn forced_text(
        m: &mut q3w::Qwen3MoeWgpu,
        ids: &[u32],
        mut sink: impl FnMut(usize, &[f32]),
    ) -> anyhow::Result<()> {
        m.reset()?;
        for (i, t) in ids.iter().enumerate() {
            let (_, l) = m.decode_step_logits(*t)?;
            sink(i, &l);
        }
        Ok(())
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

    struct RefLeg {
        runs: Vec<FreeRun>,
        answer: Vec<Vec<Vec<f32>>>,
        text: Vec<Vec<Vec<f32>>>,
        table: RunTable,
    }

    struct Instruments {
        answer: Vec<DistributionalSummary>,
        text: Vec<DistributionalSummary>,
        repeat: RepeatNull,
    }

    #[derive(Default, Clone)]
    struct RepeatNull {
        reps: usize,
        compared: usize,
        mismatched: usize,
        worst_abs: f32,
        worst_where: String,
    }

    impl std::fmt::Display for RepeatNull {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            if self.reps < 2 {
                return write!(f, "repeat null DISABLED (reps={})", self.reps);
            }
            write!(
                f,
                "repeat null: {}/{} re-run positions differ from the first pass in the same build",
                self.mismatched, self.compared
            )?;
            if self.mismatched > 0 {
                write!(
                    f,
                    ", worst |dlogit| {:.3e} at {}",
                    self.worst_abs, self.worst_where
                )?;
            }
            Ok(())
        }
    }

    fn pooled_control_mean(rows: &[DistributionalSummary]) -> (f64, usize) {
        let mut n = 0usize;
        let mut acc = 0f64;
        for r in rows.iter().filter(|r| r.prompt_kind == PromptKind::Control) {
            acc += r.mean_kl * r.steps as f64;
            n += r.steps;
        }
        (if n == 0 { 0.0 } else { acc / n as f64 }, n)
    }

    fn worst_control_mean(rows: &[DistributionalSummary]) -> f64 {
        rows.iter()
            .filter(|r| r.prompt_kind == PromptKind::Control && r.steps > 0)
            .map(|r| r.mean_kl)
            .fold(0.0f64, f64::max)
    }

    fn worst_control_max(rows: &[DistributionalSummary]) -> f64 {
        rows.iter()
            .filter(|r| r.prompt_kind == PromptKind::Control && r.steps > 0)
            .map(|r| r.max_kl)
            .fold(0.0f64, f64::max)
    }

    fn all_bit_identical(rows: &[DistributionalSummary]) -> bool {
        !rows.is_empty() && rows.iter().all(|r| r.steps == 0 || r.bit_identical)
    }

    fn instrument_line(name: &str, rows: &[DistributionalSummary]) -> String {
        let (pooled, steps) = pooled_control_mean(rows);
        format!(
            "{name:<12} controls: {steps} scored positions, pooled mean KL {pooled:.3e} nats \
             ({:+.4}% ppl), worst-prompt mean {:.3e}, worst step {:.3e}, bit-identical {}",
            relative_perplexity_pct(pooled),
            worst_control_mean(rows),
            worst_control_max(rows),
            all_bit_identical(rows)
        )
    }

    fn repeat_null(
        m: &mut q3w::Qwen3MoeWgpu,
        p: &TemplatedPrompt,
        first: &[Vec<f32>],
        reps: usize,
        out: &mut RepeatNull,
    ) -> anyhow::Result<()> {
        out.reps = reps;
        for _ in 1..reps {
            forced_text(m, &p.ids, |k, l| {
                let Some(f) = first.get(k) else { return };
                out.compared += 1;
                let mut worst = 0f32;
                for (a, b) in f.iter().zip(l.iter()) {
                    if a.to_bits() != b.to_bits() {
                        worst = worst.max((a - b).abs());
                    }
                }
                if worst > 0.0 {
                    out.mismatched += 1;
                    if worst > out.worst_abs {
                        out.worst_abs = worst;
                        out.worst_where = format!("{} position {k}", p.label);
                    }
                }
            })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn score_arm(
        label: &str,
        m: &mut q3w::Qwen3MoeWgpu,
        chosen: &[&TemplatedPrompt],
        reference: &RefLeg,
        stops: &StopSet,
        max_steps: usize,
        keep: usize,
        reps: usize,
        decode_text: &dyn Fn(&[u32]) -> String,
        label_of: &dyn Fn(u32) -> String,
    ) -> anyhow::Result<(SuiteReport, Instruments, RunTable)> {
        let mut suite = SuiteReport::new(
            &format!("Qwen3.6-35B-A3B-NVFP4 :: {label}"),
            &reference.runs[0].arm,
            label,
        );
        let mut table = RunTable::new(&format!("Qwen3.6-35B-A3B-NVFP4 :: {label}"));
        let mut inst = Instruments {
            answer: Vec::new(),
            text: Vec::new(),
            repeat: RepeatNull::default(),
        };

        for (i, p) in chosen.iter().enumerate() {
            let (rows, stopped) = free_run(m, &p.ids, max_steps, stops)?;
            let mut cand = run_of(label, p, &rows, stopped);
            cand.text = decode_text(&cand.tokens);
            table.push(ArmObservation::of(p, &cand, stops, max_steps, label_of));

            let mut a = DistributionalSummary::new(&p.label, p.kind);
            {
                let refs = &reference.answer[i];
                forced_answer(m, &p.ids, &reference.runs[i].tokens, keep, |k, l| {
                    if let Some(r) = refs.get(k) {
                        a.push(step_delta(r, l));
                    }
                })?;
            }
            let mut t = DistributionalSummary::new(&p.label, p.kind);
            let mut first: Vec<Vec<f32>> = Vec::with_capacity(p.ids.len());
            {
                let refs = &reference.text[i];
                forced_text(m, &p.ids, |k, l| {
                    if let Some(r) = refs.get(k) {
                        t.push(step_delta(r, l));
                    }
                    first.push(l.to_vec());
                })?;
            }
            repeat_null(m, p, &first, reps, &mut inst.repeat)?;
            eprintln!("[{label}] I1 answer      {a}");
            eprintln!("[{label}] I2 prompt-text {t}");
            inst.answer.push(a);
            inst.text.push(t);
            suite.push(compare(p, &reference.runs[i], &cand));
        }
        Ok((suite, inst, table))
    }

    #[test]
    #[ignore = "loads the 22 GB Qwen3.6-35B-A3B-NVFP4 checkpoint on wgpu; set NV_QWEN36_FLIP=1"]
    fn qwen36_nvfp4_v2_default_flip_battery() {
        assert_eq!(
            std::env::var("NV_QWEN36_FLIP").ok().as_deref(),
            Some("1"),
            "set NV_QWEN36_FLIP=1 -- this suite must never silently skip, and a skip here prints \
             as a pass"
        );
        let ctx = nv_kernels::wgpu_backend::WgpuContext::shared().expect("wgpu adapter");
        eprintln!("adapter: {}", ctx.summary());
        eprintln!(
            "{WHAT_THIS_DECIDES}\n\n{TWO_INSTRUMENTS}\n\n{WHY_A_NULL_ARM}\n\n{WHAT_THIS_MEASURED}\n\n{FLIP_BAR_DOC}"
        );
        scrub();
        let default_side = q3w::nvfp4_v2_enabled(ctx);
        eprintln!(
            "[default] NVFP4_V2_DEFAULT_ON={}, and with the env scrubbed this adapter routes \
             nvfp4 GEMVs to {}. Both arms below set the knob explicitly, so the verdict does not \
             depend on which side that is.",
            q3w::NVFP4_V2_DEFAULT_ON,
            if default_side { "v2" } else { "v1" }
        );
        assert_eq!(
            default_side,
            q3w::NVFP4_V2_DEFAULT_ON,
            "the compiled default and the runtime predicate disagree on this adapter"
        );
        std::env::set_var("NV_Q3_WGPU_NVFP4_V2", "1");
        let reachable = q3w::nvfp4_v2_enabled(ctx);
        std::env::set_var("NV_Q3_WGPU_NVFP4_V2", "0");
        let off = q3w::nvfp4_v2_enabled(ctx);
        std::env::remove_var("NV_Q3_WGPU_NVFP4_V2");
        assert!(
            reachable,
            "NV_Q3_WGPU_NVFP4_V2=1 does not enable the v2 route on this adapter (subgroup width \
             is not 32). The flip is UNMEASURABLE here, which is not the same as failing"
        );
        assert!(
            !off,
            "NV_Q3_WGPU_NVFP4_V2=0 does not disable the v2 route, so there is no escape hatch and \
             the reference arm cannot be built"
        );

        let dir = qwen36_dir();
        eprintln!("[snapshot] {}", dir.display());
        let tokenizer =
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
        let (pack_path, pack): (PathBuf, PromptPack) =
            resolve_pack(&dir).expect("prompt pack for this snapshot");
        eprintln!("[pack] {}\n{WHY_A_PACK}", pack_path.display());
        eprintln!("{WHY_THE_PACK_IS_RE_RENDERED}");
        let keep_thinking = std::env::var("NV_QWEN36_FLIP_THINKING").ok().as_deref() == Some("1");
        let (pack_path, pack) = if keep_thinking {
            eprintln!("[pack] NV_QWEN36_FLIP_THINKING=1: keeping the thinking-open pack");
            (pack_path, pack)
        } else {
            let p = thinking_off_pack(&pack_path, &dir, &tokenizer);
            let pack = PromptPack::load_for_snapshot(&p, &dir).expect("derived pack");
            (p, pack)
        };
        let _ = &pack_path;
        let stops = pack.stop_set();

        let chosen: Vec<&TemplatedPrompt> = pack.prompts.iter().collect();
        assert!(
            pack.controls() >= 8,
            "pack has only {} control prompts",
            pack.controls()
        );
        eprintln!(
            "pack {} @ {} :: {} prompts ({} control), {}",
            pack.model_repo,
            pack.snapshot,
            chosen.len(),
            pack.controls(),
            stops
        );

        let config = Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
        let vocab = config.vocab_size;
        let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
            .expect("open safetensors");
        let label_of = |t: u32| tokenizer.id_to_token(t).unwrap_or_else(|| format!("<{t}>"));
        let decode_text = |ids: &[u32]| tokenizer.decode(ids, false).unwrap_or_default();

        let max_steps = env_usize("NV_FLIP_STEPS", 96);
        let keep = env_usize("NV_FLIP_KL_STEPS", 24);
        let reps = env_usize("NV_QWEN36_FLIP_REPS", 3).max(1);
        let longest = chosen.iter().map(|p| p.ids.len()).max().unwrap();
        let max_seq = env_usize(
            "NV_FLIP_MAX_SEQ",
            (longest + max_steps + 64).next_power_of_two(),
        );
        let text_positions: usize = chosen.iter().map(|p| p.ids.len()).sum();
        eprintln!(
            "free-running window {max_steps} steps; I1 keeps {keep} answer steps/prompt, I2 keeps \
             all {text_positions} prompt positions -- {} MiB of retained reference logits at vocab \
             {vocab}, kv max_seq {max_seq}",
            (chosen.len() * keep + text_positions) * vocab * 4 / (1 << 20)
        );

        apply(&REFERENCE);
        let t0 = std::time::Instant::now();
        let mut m = q3w::Qwen3MoeWgpu::from_loader(config.clone(), &loader, max_seq)
            .expect("build reference");
        scrub();
        let (routed, total) = m.nvfp4_v2_gemvs();
        eprintln!(
            "[{}] built in {:.1}s, {} passes, nvfp4 GEMVs on v2: {routed}/{total}",
            REFERENCE.label,
            t0.elapsed().as_secs_f64(),
            m.pass_count()
        );
        assert!(total > 0, "the graph has no nvfp4 GEMV at all");
        assert_eq!(routed, 0, "the reference arm already routes to v2");

        let mut reference = RefLeg {
            runs: Vec::new(),
            answer: Vec::new(),
            text: Vec::new(),
            table: RunTable::new("Qwen3.6-35B-A3B-NVFP4 :: reference"),
        };
        let mut ref_repeat = RepeatNull::default();
        for p in &chosen {
            let (rows, stopped) = free_run(&mut m, &p.ids, max_steps, &stops).expect("free run");
            let mut r = run_of(REFERENCE.label, p, &rows, stopped);
            r.text = decode_text(&r.tokens);
            let mut ans = Vec::new();
            forced_answer(&mut m, &p.ids, &r.tokens, keep, |_, l| ans.push(l.to_vec()))
                .expect("reference answer replay");
            let mut txt = Vec::new();
            forced_text(&mut m, &p.ids, |_, l| txt.push(l.to_vec()))
                .expect("reference text replay");
            eprintln!(
                "[{}] {:<24} {} answer tokens, {} retained, {} prompt positions",
                REFERENCE.label,
                p.label,
                r.tokens.len(),
                ans.len(),
                txt.len()
            );
            repeat_null(&mut m, p, &txt, reps, &mut ref_repeat).expect("reference repeat null");
            reference
                .table
                .push(ArmObservation::of(p, &r, &stops, max_steps, &label_of));
            reference.runs.push(r);
            reference.answer.push(ans);
            reference.text.push(txt);
        }
        drop(m);
        eprintln!("[{}] SAME-BUILD {ref_repeat}", REFERENCE.label);
        eprintln!("{}", reference.table);
        reference
            .table
            .assert_controls_terminated_for(REFERENCE.label)
            .expect("the reference must end its turn on every control prompt (G0)");

        let mut verdicts: Vec<(String, FlipVerdict, String)> = Vec::new();
        let bar = FlipBar::from_env();
        let mut null_answer = 0f64;
        let mut null_text = 0f64;
        let mut cand_text_by_leg: Vec<(String, f64)> = Vec::new();
        let mut repeat_by_arm: Vec<(String, RepeatNull)> = Vec::new();
        let mut per_arm_rows: Vec<(String, Vec<[f64; 4]>)> = Vec::new();
        for a in CANDIDATES.iter() {
            apply(a);
            let built = q3w::Qwen3MoeWgpu::from_loader(config.clone(), &loader, max_seq);
            scrub();
            let mut cm = built.expect("build candidate");
            let (routed, total) = cm.nvfp4_v2_gemvs();
            eprintln!(
                "[{}] {} passes, nvfp4 GEMVs on v2: {routed}/{total}",
                a.label,
                cm.pass_count()
            );
            assert!(
                !a.expect_all_v2 || (routed == total && routed > 0),
                "[{}] declares the v2 route but the BUILDER put {routed} of {total} nvfp4 GEMVs on \
                 it. A flip that never reached the builder reports BIT-IDENTICAL and is the worst \
                 outcome an A/B can have",
                a.label
            );
            assert!(
                !a.expect_no_v2 || routed == 0,
                "[{}] is a null control but {routed} of {total} GEMVs took the v2 route",
                a.label
            );

            let (suite, inst, table) = score_arm(
                a.label,
                &mut cm,
                &chosen,
                &reference,
                &stops,
                max_steps,
                keep,
                reps,
                &decode_text,
                &label_of,
            )
            .expect("score arm");
            drop(cm);
            eprintln!("{table}");
            eprintln!("{suite}");
            per_arm_rows.push((
                a.label.to_string(),
                inst.answer
                    .iter()
                    .zip(inst.text.iter())
                    .map(|(x, y)| [x.mean_kl, x.max_kl, y.mean_kl, y.max_kl])
                    .collect(),
            ));

            let d_answer = decide_flip(a.label, "Qwen3.6-35B-A3B-NVFP4", bar, &suite, &inst.answer);
            let d_text = decide_flip(
                &format!("{} [I2 prompt-text]", a.label),
                "Qwen3.6-35B-A3B-NVFP4",
                bar,
                &suite,
                &inst.text,
            );
            eprintln!("\n---- {} ----", a.label);
            eprintln!("{}", instrument_line("I1 answer", &inst.answer));
            eprintln!("{}", instrument_line("I2 prompt-text", &inst.text));
            eprintln!("SAME-BUILD {}", inst.repeat);
            repeat_by_arm.push((a.label.to_string(), inst.repeat.clone()));
            let (pa, _) = pooled_control_mean(&inst.answer);
            let (pt, _) = pooled_control_mean(&inst.text);
            let ratio = if pa > 0.0 {
                format!("{:.1}x", pt / pa)
            } else if pt > 0.0 {
                "infinite (I1 is exactly zero)".to_string()
            } else {
                "n/a (both exactly zero)".to_string()
            };
            eprintln!(
                "INSTRUMENT DISAGREEMENT I2/I1 on pooled control mean KL: {ratio}. {}",
                if pt > pa * 10.0 && pa > 0.0 {
                    "The short-answer instrument understates this change by an order of magnitude \
                     or more -- read I2, not I1."
                } else {
                    "The two instruments agree to within an order of magnitude."
                }
            );
            eprintln!("{d_answer}");
            eprintln!("{}", InstrumentReport::new("I1 answer", &inst.answer));
            eprintln!("{d_text}");
            eprintln!("{}", InstrumentReport::new("I2 prompt-text", &inst.text));

            if a.label.starts_with("null") {
                null_answer = worst_control_mean(&inst.answer);
                null_text = worst_control_mean(&inst.text);
                eprintln!(
                    "NULL FLOOR: I1 worst-control mean KL {null_answer:.3e} nats, I2 \
                     {null_text:.3e} nats. Nothing at or below this can decide a flip."
                );
            } else {
                eprintln!(
                    "against the null floor: I1 {:.3e} vs {null_answer:.3e}, I2 {:.3e} vs \
                     {null_text:.3e}",
                    worst_control_mean(&inst.answer),
                    worst_control_mean(&inst.text)
                );
                cand_text_by_leg.push((a.label.to_string(), worst_control_mean(&inst.text)));
            }

            let both = if d_answer.accepted() && d_text.accepted() {
                if d_answer.verdict == FlipVerdict::AcceptBitIdentical
                    && d_text.verdict == FlipVerdict::AcceptBitIdentical
                {
                    FlipVerdict::AcceptBitIdentical
                } else {
                    FlipVerdict::Accept
                }
            } else if d_answer.verdict == FlipVerdict::Void || d_text.verdict == FlipVerdict::Void {
                FlipVerdict::Void
            } else {
                FlipVerdict::Reject
            };
            let why = if both == FlipVerdict::Reject {
                d_answer
                    .failures
                    .iter()
                    .map(|f| format!("I1 {f}"))
                    .chain(d_text.failures.iter().map(|f| format!("I2 {f}")))
                    .collect::<Vec<_>>()
                    .join(" | ")
            } else {
                format!(
                    "I1 pooled {:.3e} nats, I2 pooled {:.3e} nats",
                    pooled_control_mean(&inst.answer).0,
                    pooled_control_mean(&inst.text).0
                )
            };
            println!(
                "FLIP-VERDICT Qwen3.6-35B-A3B-NVFP4 {} {both} | {why}",
                a.label
            );
            verdicts.push((a.label.to_string(), both, why));
        }

        eprintln!("\n================ FLIP VERDICTS ================");
        for (label, v, why) in &verdicts {
            eprintln!("  {label:<26} {v}  {why}");
        }

        let v2_legs: Vec<&(String, Vec<[f64; 4]>)> = per_arm_rows
            .iter()
            .filter(|(l, _)| l.starts_with("v2"))
            .collect();
        assert_eq!(
            v2_legs.len(),
            2,
            "the candidate needs two builds to have a within-arm null"
        );
        let mut worst = 0f64;
        let mut worst_where = String::new();
        for (i, p) in chosen.iter().enumerate() {
            for (k, name) in ["I1 mean", "I1 max", "I2 mean", "I2 max"]
                .iter()
                .enumerate()
            {
                let (a, b) = (v2_legs[0].1[i][k], v2_legs[1].1[i][k]);
                let d = (a - b).abs();
                if d > worst {
                    worst = d;
                    worst_where = format!("{} {name} ({a:.6e} vs {b:.6e})", p.label);
                }
            }
        }
        let tolerance = 0.1 * bar.max_mean_kl;
        eprintln!(
            "WITHIN-ARM NULL on the candidate ({} vs {}): largest disagreement over {} prompts x 4 \
             statistics is {worst:.3e} nats{}",
            v2_legs[0].0,
            v2_legs[1].0,
            chosen.len(),
            if worst == 0.0 {
                " -- the two builds are BIT-IDENTICAL to each other, so the between-arm effect \
                 above is the kernel and not the weather"
                    .to_string()
            } else {
                format!(" at {worst_where}")
            }
        );
        let reproducible = worst <= tolerance;
        let legs = cand_text_by_leg
            .iter()
            .map(|(l, v)| format!("{l} {v:.3e}"))
            .collect::<Vec<_>>()
            .join(", ");
        let smallest_leg = cand_text_by_leg
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::INFINITY, f64::min);
        if !reproducible {
            eprintln!(
                "\nTHE CANDIDATE IS NOT REPRODUCIBLE, SO NOTHING ABOVE IS AN EFFECT.\n  \
                 within-arm  {worst:.3e} nats  (two builds of the SAME env, at {worst_where})\n  \
                 between-arm I2 worst-control mean KL, per leg: {legs}\n  \
                 tolerance   {tolerance:.3e} nats  (a tenth of the {:.1e} G3c bar)\n\
                 A single leg would have reported {smallest_leg:.3e} nats, {:.0}x smaller than the \
                 arm's disagreement with itself, so that number measures the arm's own scatter. \
                 The reference arm rebuilt under the identical procedure came back bit-identical \
                 on every retained logit word, so this is the candidate's kernels and not the \
                 harness, the box, or the model.",
                bar.max_mean_kl,
                worst / smallest_leg.max(f64::MIN_POSITIVE)
            );
            for v in verdicts.iter_mut().filter(|(l, _, _)| l.starts_with("v2")) {
                v.1 = FlipVerdict::Void;
                v.2 = format!(
                    "within-arm null {worst:.3e} nats > tolerance {tolerance:.3e}; the arm does \
                     not reproduce itself"
                );
                println!(
                    "FLIP-VERDICT Qwen3.6-35B-A3B-NVFP4 {} {} | {}",
                    v.0, v.1, v.2
                );
            }
        }
        eprintln!(
            "\n==== SAME-BUILD REPEAT NULL ({reps} passes of every teacher-forced prompt per arm) \
             ====\nThis is the null with real statistical power: a rebuild null buys one \
             observation per 22 GB build, this buys one per position per repetition. Any nonzero \
             row means the arm does not compute a function of its input."
        );
        eprintln!("  {:<26} {ref_repeat}", REFERENCE.label);
        for (l, r) in &repeat_by_arm {
            eprintln!("  {l:<26} {r}");
        }
        let flaky: Vec<&str> = repeat_by_arm
            .iter()
            .filter(|(_, r)| r.mismatched > 0)
            .map(|(l, _)| l.as_str())
            .chain((ref_repeat.mismatched > 0).then_some(REFERENCE.label))
            .collect();
        if flaky.is_empty() {
            eprintln!(
                "  every arm reproduced itself bit for bit within its own build, so no verdict \
                 above rests on a pass that would not repeat"
            );
        } else {
            eprintln!("  NOT A FUNCTION OF ITS INPUT: {flaky:?}");
        }
        eprintln!("{MARGIN_TRAP}\n{REPRODUCIBILITY_LIMIT}");

        eprintln!("\n================ FINAL ================");
        for (label, v, why) in &verdicts {
            eprintln!("  {label:<26} {v}  {why}");
        }

        let null = verdicts
            .iter()
            .find(|(l, _, _)| l.starts_with("null"))
            .expect("the null arm must produce a verdict");
        assert!(
            null.1 == FlipVerdict::Accept || null.1 == FlipVerdict::AcceptBitIdentical,
            "the NULL arm -- the reference config rebuilt -- failed its own bar ({}): {}. The \
             harness floor is above the bar, so nothing measured here can decide the flip",
            null.1,
            null.2
        );
        if std::env::var("NV_FLIP_ASSERT_ACCEPT").ok().as_deref() == Some("1") {
            let bad: Vec<&str> = verdicts
                .iter()
                .filter(|(_, v, _)| {
                    *v != FlipVerdict::Accept && *v != FlipVerdict::AcceptBitIdentical
                })
                .map(|(l, _, _)| l.as_str())
                .collect();
            assert!(
                bad.is_empty(),
                "NV_FLIP_ASSERT_ACCEPT=1 and {bad:?} did not ACCEPT"
            );
        } else {
            eprintln!(
                "NOTE: a REJECT, and a VOID from an arm that will not reproduce itself, are \
                 measurement RESULTS, not harness failures, so this test is green either way. \
                 NV_FLIP_ASSERT_ACCEPT=1 turns the bar into a landing gate."
            );
        }
    }

    #[test]
    fn the_battery_arms_can_actually_decide_something() {
        assert_eq!(
            REFERENCE.env,
            &[("NV_Q3_WGPU_NVFP4_V2", "0")],
            "the reference must pin the v1 kernel explicitly; an arm that means \"whatever the \
             default is\" becomes the candidate the day the default flips"
        );
        let null = &CANDIDATES[0];
        assert!(
            null.env == REFERENCE.env && null.expect_no_v2,
            "CANDIDATES[0] must be the null control: same env as the reference, no v2 route"
        );
        let cand = &CANDIDATES[1];
        assert!(
            cand.expect_all_v2 && !cand.env.is_empty(),
            "the candidate must declare an observable route change"
        );
        assert_eq!(
            CANDIDATES[1].env, CANDIDATES[2].env,
            "CANDIDATES[2] must repeat CANDIDATES[1]'s env exactly, or there is no within-arm null"
        );
        for (k, _) in cand.env {
            assert!(
                KNOBS.contains(k),
                "candidate sets {k}, which the reference does not scrub, so an ambient {k} would \
                 leak into the reference and both arms would measure the same thing"
            );
        }
        assert!(
            KNOBS.contains(&"NV_Q3_WGPU_NVFP4_V2"),
            "the knob both arms set is not in the scrub list, so an ambient value would leak"
        );
    }
}
