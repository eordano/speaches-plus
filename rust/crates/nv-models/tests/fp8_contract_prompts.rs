#![allow(dead_code)]

#[path = "../../../tests/common/chat_eval_core.rs"]
pub mod harness_self_test_no_server_code;

pub use harness_self_test_no_server_code::*;

use std::fmt;
use std::path::{Path, PathBuf};

pub const EMIT_CMD: &str = "NVK_LANE=<lane> NVK_PKG=speaches-plus NVK_FEATURES= \
     rust/scripts/nvk.sh test --test chat_eval emit_prompt_packs_for_the_measure_lane -- --nocapture";

pub const WHY_A_PACK: &str = "This harness will not accept a hand-written prompt string. On \
     2026-08-06 the raw fragment \"The capital of France is\" was measured on Gemma4-31B-IT-NVFP4: \
     96 tokens, MAX-STEPS, never ended its turn, median top-2 margin 11.354, text \" Paris. The \
     capital of France is Paris. The capital of France is Paris. ...\". That confident repetition \
     loop was the bf16 reference every earlier fp8 A/B was validated against, so the reported \
     33/33 agreement was two loops agreeing with each other. Prompts must come from a PromptPack \
     rendered through the model's own chat_template.jinja.";

pub const REPRODUCIBILITY_LIMIT: &str = "MEASURED REPRODUCIBILITY LIMIT (2026-08-06, this box): \
     three runs of a BYTE-IDENTICAL configuration on one low-margin open-ended prompt gave 66 \
     tokens ended / 78 tokens ended / 96 tokens NOT ended. A single free-running trajectory of a \
     low-margin prompt is not reproducible here and cannot carry an A/B claim. The control prompts \
     measured median top-2 margins of 15.8 to 17.8 and were stable, so only CONTROL rows are \
     A/B evidence. Open-ended rows are printed DESCRIPTIVE-ONLY.";

pub const MARGIN_TRAP: &str = "MARGIN TRAP: a high top-2 margin does NOT mean healthy output. The \
     degenerate repetition loop above sat at median margin 11.354 while looping. Never argue \
     \"every divergence is a low-margin tie-break so it is benign\" without also showing the arm \
     ENDED ITS TURN.";

pub const NVFP4_31B_REPO: &str = "nvidia/Gemma-4-31B-IT-NVFP4";
pub const SNAP_NVFP4_AUTHORITATIVE: &str = "e5ef03afa233c35cb000323ff098d4291e1dd07c";
pub const SNAP_NVFP4_LEGACY: &str = "1365cf7aa2de42546878b8d2e4a425019a0be514";

pub fn hub_roots() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if p.is_dir() && !out.contains(&p) {
            out.push(p);
        }
    };
    if let Ok(v) = std::env::var("HF_HUB_CACHE") {
        push(PathBuf::from(v));
    }
    push(PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface/hub"));
    out
}

pub fn snapshots_of(repo: &str) -> Vec<(String, PathBuf)> {
    let leaf = format!("models--{}", repo.replace('/', "--"));
    let mut out: Vec<(String, PathBuf)> = Vec::new();
    for root in hub_roots() {
        let d = root.join(&leaf).join("snapshots");
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                let id = e.file_name().to_string_lossy().to_string();
                if p.join("config.json").exists() && !out.iter().any(|(i, _)| *i == id) {
                    out.push((id, p));
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

pub fn gemma4_nvfp4_dir() -> anyhow::Result<PathBuf> {
    if let Ok(v) = std::env::var("NV_GEMMA4_DIR") {
        let p = PathBuf::from(v);
        anyhow::ensure!(
            p.join("config.json").exists(),
            "NV_GEMMA4_DIR={} has no config.json",
            p.display()
        );
        eprintln!("[snapshot] NV_GEMMA4_DIR pins {}", p.display());
        return Ok(p);
    }
    let snaps = snapshots_of(NVFP4_31B_REPO);
    anyhow::ensure!(
        !snaps.is_empty(),
        "{NVFP4_31B_REPO} is not cached in any hub root {:?}",
        hub_roots()
    );
    let want = std::env::var("NV_GEMMA4_SNAPSHOT").ok();
    let order: Vec<String> = want
        .clone()
        .into_iter()
        .chain([
            SNAP_NVFP4_AUTHORITATIVE.to_string(),
            SNAP_NVFP4_LEGACY.to_string(),
        ])
        .collect();
    eprintln!(
        "[snapshot] {NVFP4_31B_REPO} cached snapshots: {:?}",
        snaps.iter().map(|s| &s.0[..8]).collect::<Vec<_>>()
    );
    for id in &order {
        if let Some((_, p)) = snaps.iter().find(|(i, _)| i == id) {
            eprintln!(
                "[snapshot] using {} ({}). flake.nix pins {} as refs/main; the weight blobs are \
                 SHARED between the two snapshots, so no throughput number depends on this choice, \
                 but tool-calling and thinking-mode renders differ. Override with NV_GEMMA4_DIR or \
                 NV_GEMMA4_SNAPSHOT.",
                &id[..8],
                p.display(),
                &SNAP_NVFP4_AUTHORITATIVE[..8]
            );
            return Ok(p.clone());
        }
    }
    let (id, p) = snaps[0].clone();
    eprintln!(
        "[snapshot] neither the authoritative {} nor the legacy {} is cached; falling back to the \
         lexicographically first snapshot {}",
        &SNAP_NVFP4_AUTHORITATIVE[..8],
        &SNAP_NVFP4_LEGACY[..8],
        &id[..8]
    );
    Ok(p)
}

pub fn pack_search_dirs() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if p.is_dir() && !out.contains(&p) {
            out.push(p);
        }
    };
    if let Ok(v) = std::env::var("NV_CHAT_EVAL_OUT") {
        push(PathBuf::from(v));
    }
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_default());
    push(home.join(".cache/nvk-tmp/chat-eval"));
    out
}

fn candidate_packs() -> Vec<PathBuf> {
    if let Ok(v) = std::env::var("NV_CHAT_EVAL_PACK") {
        return vec![PathBuf::from(v)];
    }
    let mut out = Vec::new();
    for d in pack_search_dirs() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                let name = p
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                if name.starts_with("pack-") && name.ends_with(".json") {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out
}

pub fn resolve_pack(weights_dir: &Path) -> anyhow::Result<(PathBuf, PromptPack)> {
    let cands = candidate_packs();
    anyhow::ensure!(
        !cands.is_empty(),
        "no prompt pack found in {:?}. {WHY_A_PACK}\nEmit one with:\n  {EMIT_CMD}",
        pack_search_dirs()
    );
    let mut rejected = Vec::new();
    for p in &cands {
        match PromptPack::load_for_snapshot(p, weights_dir) {
            Ok(pack) => return Ok((p.clone(), pack)),
            Err(e) => rejected.push(format!("{}: {e}", p.display())),
        }
    }
    anyhow::bail!(
        "no prompt pack matches the chat template shipped by {}. {WHY_A_PACK}\nRe-emit with:\n  \
         {EMIT_CMD}\nrejected candidates:\n  {}",
        weights_dir.display(),
        rejected.join("\n  ")
    )
}

pub fn max_steps(default: usize) -> usize {
    std::env::var("NV_FP8_CONTRACT_WINDOW")
        .or_else(|_| std::env::var("NV_CHAT_EVAL_STEPS"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub fn include_open_ended() -> bool {
    std::env::var("NV_FP8_INCLUDE_OPEN_ENDED").ok().as_deref() == Some("1")
}

pub fn ab_prompts(pack: &PromptPack) -> Vec<&TemplatedPrompt> {
    let controls: Vec<&TemplatedPrompt> = pack
        .prompts
        .iter()
        .filter(|p| p.kind == PromptKind::Control)
        .collect();
    if !include_open_ended() {
        return controls;
    }
    pack.prompts.iter().collect()
}

pub fn is_ab_evidence(p: &TemplatedPrompt) -> bool {
    p.kind == PromptKind::Control
}

pub fn describe_selection(pack: &PromptPack, chosen: &[&TemplatedPrompt]) -> String {
    format!(
        "pack {} @ {} :: template {} ({} bytes), {}\n\
         selected {}/{} prompts for this run ({} CONTROL = A/B evidence, {} open-ended = \
         DESCRIPTIVE-ONLY). Set NV_FP8_INCLUDE_OPEN_ENDED=1 to add the open-ended prompts; they \
         are still reported as descriptive only.\n{REPRODUCIBILITY_LIMIT}\n{MARGIN_TRAP}",
        pack.model_repo,
        pack.snapshot,
        pack.template_digest,
        pack.template_bytes,
        pack.stop_set(),
        chosen.len(),
        pack.prompts.len(),
        chosen.iter().filter(|p| is_ab_evidence(p)).count(),
        chosen.iter().filter(|p| !is_ab_evidence(p)).count(),
    )
}

#[derive(Clone, Debug)]
pub struct ArmObservation {
    pub arm: String,
    pub prompt_label: String,
    pub kind: PromptKind,
    pub tokens: usize,
    pub max_steps: usize,
    pub ended_turn: bool,
    pub stop_token: Option<u32>,
    pub stop_label: String,
    pub median_margin: f32,
    pub min_margin: f32,
    pub leaked_stop_ids: Vec<u32>,
    pub repeated_tail: Option<(usize, String)>,
    pub text: String,
}

impl ArmObservation {
    pub fn of(
        prompt: &TemplatedPrompt,
        run: &FreeRun,
        stops: &StopSet,
        max_steps: usize,
        label_of: &dyn Fn(u32) -> String,
    ) -> Self {
        let ended_turn = run.reason == StopReason::HitStopToken;
        let body_end = run.tokens.len().saturating_sub(usize::from(ended_turn));
        let leaked_stop_ids: Vec<u32> = run.tokens[..body_end]
            .iter()
            .copied()
            .filter(|t| stops.contains(*t))
            .collect();
        Self {
            arm: run.arm.clone(),
            prompt_label: prompt.label.clone(),
            kind: prompt.kind,
            tokens: run.tokens.len(),
            max_steps,
            ended_turn,
            stop_token: run.stop_token,
            stop_label: run.stop_token.map(label_of).unwrap_or_default(),
            median_margin: run.median_margin(),
            min_margin: run
                .margins
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min)
                .min(f32::MAX),
            leaked_stop_ids,
            repeated_tail: repeated_tail(&run.tokens),
            text: run.text.clone(),
        }
    }

    pub fn evidence_class(&self) -> &'static str {
        match self.kind {
            PromptKind::Control => "A/B",
            PromptKind::OpenEnded => "DESCRIPTIVE-ONLY",
        }
    }

    pub fn termination(&self) -> String {
        if self.ended_turn {
            format!(
                "ENDED-TURN {} ({})",
                self.stop_token.unwrap_or_default(),
                self.stop_label
            )
        } else {
            format!("MAX-STEPS({}) NOT-TERMINATED", self.max_steps)
        }
    }

    pub fn health(&self) -> String {
        let mut flags: Vec<String> = Vec::new();
        if !self.ended_turn {
            flags.push("did-not-end-turn".into());
        }
        if let Some((period, unit)) = &self.repeated_tail {
            flags.push(format!("repetition-loop(period {period}, {unit:?})"));
        }
        if !self.leaked_stop_ids.is_empty() {
            flags.push(format!("stop-token-leak {:?}", self.leaked_stop_ids));
        }
        if flags.is_empty() {
            "ok".into()
        } else {
            flags.join(" + ")
        }
    }

    pub fn is_degenerate(&self) -> bool {
        !self.ended_turn || self.repeated_tail.is_some() || !self.leaked_stop_ids.is_empty()
    }
}

pub fn repeated_tail(tokens: &[u32]) -> Option<(usize, String)> {
    let n = tokens.len();
    if n < 12 {
        return None;
    }
    for period in 1..=12usize {
        if period * 4 > n {
            break;
        }
        let tail = &tokens[n - period * 4..];
        if tail.chunks(period).all(|c| c == &tail[..period]) {
            return Some((period, format!("{:?}", &tail[..period])));
        }
    }
    None
}

#[derive(Clone, Debug, Default)]
pub struct RunTable {
    pub title: String,
    pub rows: Vec<ArmObservation>,
}

impl RunTable {
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            rows: Vec::new(),
        }
    }

    pub fn push(&mut self, o: ArmObservation) {
        self.rows.push(o);
    }

    pub fn controls(&self) -> Vec<&ArmObservation> {
        self.rows
            .iter()
            .filter(|r| r.kind == PromptKind::Control)
            .collect()
    }

    pub fn non_terminating_controls(&self) -> Vec<&ArmObservation> {
        self.controls()
            .into_iter()
            .filter(|r| !r.ended_turn)
            .collect()
    }

    pub fn assert_controls_terminated(&self) -> anyhow::Result<()> {
        let bad = self.non_terminating_controls();
        anyhow::ensure!(
            !self.controls().is_empty(),
            "no CONTROL row in this table; a run with no control cannot support an A/B claim"
        );
        anyhow::ensure!(
            bad.is_empty(),
            "{} CONTROL row(s) hit max-steps without ending the turn: {}. A control that does not \
             terminate is a broken control, and every agreement ratio computed against it is a \
             comparison of two runaway trajectories rather than of two answers. {WHY_A_PACK}",
            bad.len(),
            bad.iter()
                .map(|r| format!("{}[{}] {} tokens", r.prompt_label, r.arm, r.tokens))
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(())
    }

    pub fn assert_controls_terminated_for(&self, arm: &str) -> anyhow::Result<()> {
        let mine: Vec<&ArmObservation> = self
            .controls()
            .into_iter()
            .filter(|r| r.arm == arm)
            .collect();
        anyhow::ensure!(
            !mine.is_empty(),
            "arm {arm:?} contributed no CONTROL row; nothing in this table is A/B evidence"
        );
        let bad: Vec<&&ArmObservation> = mine.iter().filter(|r| !r.ended_turn).collect();
        anyhow::ensure!(
            bad.is_empty(),
            "REFERENCE arm {arm:?} failed to end its turn on {} control prompt(s): {}. The \
             reference is broken, so no agreement ratio measured against it means anything. \
             {WHY_A_PACK}",
            bad.len(),
            bad.iter()
                .map(|r| format!("{} ({} tokens)", r.prompt_label, r.tokens))
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(())
    }

    pub fn degenerate_rows(&self) -> Vec<&ArmObservation> {
        self.rows.iter().filter(|r| r.is_degenerate()).collect()
    }
}

impl fmt::Display for RunTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "==== {} :: per-prompt, per-arm behaviour ====",
            self.title
        )?;
        writeln!(
            f,
            "{:<22} {:<16} {:<14} {:>6} {:<28} {:>10} {:>10}  health",
            "prompt", "evidence", "arm", "tokens", "termination", "med-marg", "min-marg"
        )?;
        for r in &self.rows {
            writeln!(
                f,
                "{:<22} {:<16} {:<14} {:>6} {:<28} {:>10.3} {:>10.3}  {}",
                r.prompt_label,
                r.evidence_class(),
                r.arm,
                r.tokens,
                r.termination(),
                r.median_margin,
                r.min_margin,
                r.health()
            )?;
        }
        let bad = self.degenerate_rows();
        if bad.is_empty() {
            writeln!(
                f,
                "every row ended its turn with no repetition loop and no stop-token leak"
            )?;
        } else {
            writeln!(f, "{} DEGENERATE row(s):", bad.len())?;
            for r in &bad {
                writeln!(f, "  {} [{}] -> {}", r.prompt_label, r.arm, r.health())?;
            }
            writeln!(
                f,
                "A degenerate row cannot support an agreement ratio: two runaway trajectories \
                 agreeing is nearly free."
            )?;
        }
        for r in &self.rows {
            writeln!(
                f,
                "--- {} [{}] {} ---",
                r.prompt_label,
                r.arm,
                r.evidence_class()
            )?;
            writeln!(f, "{:?}", r.text)?;
        }
        write!(f, "{REPRODUCIBILITY_LIMIT}")
    }
}

pub const VISIBLE_SIGNAL_FLOOR_NATS: f64 = 1e-6;

pub const WHY_THE_INSTRUMENT_BLOCK_PRINTS_NEXT_TO_EVERY_VERDICT: &str = "\
WHY THIS BLOCK PRINTS NEXT TO EVERY VERDICT. The bar is a pass/fail on ONE number -- the worst \
CONTROL prompt's mean KL -- and a pass/fail hides two things the reader needs at the moment of \
decision:\n\
\n\
  (a) HOW FAR from the bar, in units anyone can compare. ppl_cand/ppl_ref = exp(mean KL) exactly, \
so 1.269e-4 nats is +0.0127% perplexity and 2.899e-1 nats is +33.6%. Both print as one word. One \
is below this box's own run-to-run reproducibility; the other is a broken model. A REJECT at \
+0.72% and a REJECT at +33.6% are not the same finding and must not read the same.\n\
  (b) WHERE the number came from. \"worst-control mean KL\" is a MAX OVER PROMPTS of a MEAN OVER \
STEPS. If only 2 of 11 control prompts carry measurable signal, the reported number IS one \
prompt's number, and it moves when that one prompt moves. Two such runs look like a trend and are \
not one. The per-prompt table and the visible-signal count below make that readable.\n\
\n\
None of this changes the bar, the gates, or the verdict. It is instrument metadata printed where \
the decision is made, so the instrument's limits are visible at the decision point.";

pub const MAX_OVER_PROMPTS_CAVEAT: &str = "\
MAX-OVER-PROMPTS CAVEAT: G3c reports max_over_prompts(mean_over_steps(KL)). It is deliberately \
worst-case, which is right for a gate and misleading for a trend: with signal on one prompt, the \
gate number is that prompt's number. Before reading two runs as a movement, check that the count \
above the visible-signal floor is the same in both and that the concentration ratio is not ~1 \
prompt wide. Adding prompts changes a max; it barely changes a median.";

pub const KERNEL_FLIP_VS_WEIGHT_FORMAT_FLIP: &str = "\
WHAT THE BAR WAS CALIBRATED FOR. mean KL <= 1e-3 nats was set for KERNEL flips -- a different \
summation order, a fused pass, a rewritten GEMV -- where the two arms compute the same function \
and near-bit-identity is achievable, so anything above float noise is a bug worth blocking. A \
WEIGHT-FORMAT flip (bf16 -> int8, nvfp4 -> int8) changes the function on purpose: re-quantization \
always shifts the distribution, and the question is whether the shift is small relative to what \
the model's own decisions can absorb. The bar answers the kernel question. On a weight-format \
flip it still answers only the kernel question -- which is why the perplexity and the per-prompt \
table below are printed next to it rather than instead of it.";

pub fn relative_perplexity_pct(mean_kl_nats: f64) -> f64 {
    100.0 * (mean_kl_nats.exp() - 1.0)
}

pub fn visible_signal_floor() -> f64 {
    std::env::var("NV_FLIP_SIGNAL_FLOOR")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0 && v.is_finite())
        .unwrap_or(VISIBLE_SIGNAL_FLOOR_NATS)
}

const KL_DECADES: [f64; 5] = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6];

#[derive(Clone, Debug)]
pub struct InstrumentReport {
    pub title: String,
    pub rows: Vec<DistributionalSummary>,
    pub floor: f64,
}

impl InstrumentReport {
    pub fn new(title: &str, dists: &[DistributionalSummary]) -> Self {
        Self {
            title: title.to_string(),
            rows: dists.to_vec(),
            floor: visible_signal_floor(),
        }
    }

    fn worst_first(&self, kind: PromptKind) -> Vec<&DistributionalSummary> {
        let mut v: Vec<&DistributionalSummary> = self
            .rows
            .iter()
            .filter(|s| s.prompt_kind == kind && s.steps > 0)
            .collect();
        v.sort_by(|a, b| b.mean_kl.total_cmp(&a.mean_kl));
        v
    }

    pub fn controls_worst_first(&self) -> Vec<&DistributionalSummary> {
        self.worst_first(PromptKind::Control)
    }

    pub fn open_ended_worst_first(&self) -> Vec<&DistributionalSummary> {
        self.worst_first(PromptKind::OpenEnded)
    }

    pub fn unmeasured(&self) -> Vec<&DistributionalSummary> {
        self.rows.iter().filter(|s| s.steps == 0).collect()
    }

    pub fn worst_control_mean_kl(&self) -> f64 {
        self.controls_worst_first()
            .first()
            .map(|s| s.mean_kl)
            .unwrap_or(0.0)
    }

    pub fn worst_control_max_kl(&self) -> f64 {
        self.controls_worst_first()
            .iter()
            .fold(0f64, |m, s| m.max(s.max_kl))
    }

    pub fn pooled_control_mean_kl(&self) -> f64 {
        let ctl = self.controls_worst_first();
        let steps: usize = ctl.iter().map(|s| s.steps).sum();
        if steps == 0 {
            return 0.0;
        }
        ctl.iter().map(|s| s.mean_kl * s.steps as f64).sum::<f64>() / steps as f64
    }

    pub fn median_control_mean_kl(&self) -> f64 {
        let ctl = self.controls_worst_first();
        if ctl.is_empty() {
            return 0.0;
        }
        ctl[ctl.len() / 2].mean_kl
    }

    pub fn relative_perplexity_pct(&self) -> f64 {
        relative_perplexity_pct(self.worst_control_mean_kl())
    }

    pub fn controls_above_floor(&self) -> usize {
        self.controls_worst_first()
            .iter()
            .filter(|s| s.mean_kl > self.floor)
            .count()
    }

    pub fn control_count(&self) -> usize {
        self.controls_worst_first().len()
    }

    pub fn decade_counts(&self) -> Vec<(f64, usize)> {
        let ctl = self.controls_worst_first();
        KL_DECADES
            .iter()
            .map(|t| (*t, ctl.iter().filter(|s| s.mean_kl >= *t).count()))
            .collect()
    }

    pub fn concentration(&self) -> Option<(f64, f64)> {
        let ctl = self.controls_worst_first();
        (ctl.len() >= 2).then(|| (ctl[0].mean_kl, ctl[1].mean_kl))
    }

    pub fn concentration_line(&self) -> String {
        let n = self.controls_above_floor();
        let total = self.control_count();
        let floor = format!(
            "{:.1e} nats (= {:+.5}% ppl)",
            self.floor,
            relative_perplexity_pct(self.floor)
        );
        if total == 0 {
            return "no CONTROL prompt carried a forced-context step, so there is no signal to \
                    concentrate and no A/B evidence here"
                .to_string();
        }
        if n == 0 {
            return format!(
                "0/{total} control prompt(s) carry mean KL above the visible-signal floor {floor}: \
                 every control sits at or below this instrument's own resolution, so the gate \
                 number is a floor reading rather than a measurement of the flip"
            );
        }
        let spread = match self.concentration() {
            Some((w, second)) if second > 0.0 => {
                format!("worst is {:.1}x the second worst {second:.3e}", w / second)
            }
            Some((_, second)) => format!(
                "the second worst control measured exactly {second:.3e}, so the worst stands alone"
            ),
            None => "only one control prompt carried forced steps at all".to_string(),
        };
        let reads = if n == 1 {
            "the G3c number IS one prompt's number: do not read a change in it across runs as a trend"
        } else if n * 4 <= total {
            "the G3c number rests on a minority of prompts: check the same prompts carry it before \
             comparing two runs"
        } else {
            "signal is spread across most controls, so the max is a reasonable summary of it"
        };
        format!(
            "{n}/{total} control prompt(s) carry mean KL above the visible-signal floor {floor}; \
             {spread} -- {reads}"
        )
    }

    pub fn headline(&self) -> String {
        format!(
            "rel-ppl {:+.4}% worst-control-mean-KL {:.3e} max-step-KL {:.3e} signal {}/{} above {:.0e}",
            self.relative_perplexity_pct(),
            self.worst_control_mean_kl(),
            self.worst_control_max_kl(),
            self.controls_above_floor(),
            self.control_count(),
            self.floor
        )
    }
}

fn instrument_table(
    f: &mut fmt::Formatter<'_>,
    caption: &str,
    rows: &[&DistributionalSummary],
) -> fmt::Result {
    writeln!(f, "{caption}")?;
    if rows.is_empty() {
        return writeln!(f, "  <none>");
    }
    writeln!(
        f,
        "  {:<3} {:<26} {:>5} {:>10} {:>10} {:>10} {:>17} {:>9} {:>8} {:>8} {:>5}",
        "#",
        "prompt",
        "steps",
        "mean KL",
        "rel ppl",
        "max KL",
        "forced top-1",
        "rho mean",
        "rho max",
        "rho>=.5",
        "rank"
    )?;
    for (i, s) in rows.iter().enumerate() {
        writeln!(
            f,
            "  {:<3} {:<26} {:>5} {:>10.3e} {:>10} {:>10.3e} {:>17} {:>9.3} {:>8.3} {:>8} {:>5}{}",
            i + 1,
            s.prompt_label,
            s.steps,
            s.mean_kl,
            format!("{:+.4}%", relative_perplexity_pct(s.mean_kl)),
            s.max_kl,
            format!("{}/{} {:.2}%", s.top1_agree, s.steps, 100.0 * s.top1_rate()),
            s.mean_rho,
            s.max_rho,
            s.steps_rho_soft,
            s.worst_rank,
            if s.bit_identical {
                "  BIT-IDENTICAL"
            } else {
                ""
            }
        )?;
    }
    Ok(())
}

impl fmt::Display for InstrumentReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "---- INSTRUMENT :: {} :: relative perplexity + per-prompt KL ----",
            self.title
        )?;
        let ctl = self.controls_worst_first();
        if ctl.is_empty() {
            writeln!(
                f,
                "  no CONTROL prompt carried a forced-context step, so there is no A/B evidence \
                 here at all -- the verdict above rests on the token/text gates only"
            )?;
        } else {
            writeln!(
                f,
                "RELATIVE PERPLEXITY (CONTROL rows only; ppl_cand/ppl_ref = exp(mean KL) exactly):\n  \
                 worst prompt   mean KL {:>10.3e} nats -> {:+.4}%   <- this is the number G3c gates on ({})\n  \
                 pooled steps   mean KL {:>10.3e} nats -> {:+.4}%   ({} forced steps over {} control prompt(s))\n  \
                 median prompt  mean KL {:>10.3e} nats -> {:+.4}%\n  \
                 worst step     max  KL {:>10.3e} nats -> {:+.4}% if every token were that bad",
                self.worst_control_mean_kl(),
                self.relative_perplexity_pct(),
                ctl[0].prompt_label,
                self.pooled_control_mean_kl(),
                relative_perplexity_pct(self.pooled_control_mean_kl()),
                ctl.iter().map(|s| s.steps).sum::<usize>(),
                ctl.len(),
                self.median_control_mean_kl(),
                relative_perplexity_pct(self.median_control_mean_kl()),
                self.worst_control_max_kl(),
                relative_perplexity_pct(self.worst_control_max_kl()),
            )?;
        }
        instrument_table(
            f,
            "PER-PROMPT, CONTROL rows (A/B EVIDENCE), worst-first by mean KL:",
            &ctl,
        )?;
        let open = self.open_ended_worst_first();
        instrument_table(
            f,
            "PER-PROMPT, OPEN-ENDED rows (DESCRIPTIVE ONLY -- these gate NOTHING and are not A/B \
             evidence), worst-first by mean KL:",
            &open,
        )?;
        let unmeasured = self.unmeasured();
        if !unmeasured.is_empty() {
            writeln!(
                f,
                "  {} prompt(s) contributed no forced-context step and are absent from both tables: {}",
                unmeasured.len(),
                unmeasured
                    .iter()
                    .map(|s| s.prompt_label.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
        }
        writeln!(f, "SIGNAL CONCENTRATION: {}", self.concentration_line())?;
        let decades: Vec<String> = self
            .decade_counts()
            .iter()
            .map(|(t, n)| format!(">={t:.0e}: {n}"))
            .collect();
        writeln!(
            f,
            "  control prompts at or above each decade of mean KL -- {} (of {})",
            decades.join(" | "),
            self.control_count()
        )?;
        writeln!(f, "{MAX_OVER_PROMPTS_CAVEAT}")?;
        write!(f, "{WHY_THE_INSTRUMENT_BLOCK_PRINTS_NEXT_TO_EVERY_VERDICT}")
    }
}

#[cfg(test)]
mod prompt_source_tests {
    use super::*;

    fn tp(label: &str, kind: PromptKind) -> TemplatedPrompt {
        TemplatedPrompt::from_official_render(label, kind, "m", "s", "d", 1, "r".into(), vec![1, 2])
    }

    fn run(
        arm: &str,
        label: &str,
        tokens: Vec<u32>,
        margins: Vec<f32>,
        stop: Option<u32>,
    ) -> FreeRun {
        FreeRun {
            arm: arm.into(),
            prompt_label: label.into(),
            tokens,
            margins,
            reason: if stop.is_some() {
                StopReason::HitStopToken
            } else {
                StopReason::ReachedMaxSteps
            },
            stop_token: stop,
            text: "t".into(),
        }
    }

    fn stops() -> StopSet {
        StopSet {
            ids: vec![1, 106, 50],
            source: "test".into(),
        }
    }

    #[test]
    fn a_control_that_never_ends_its_turn_is_a_hard_failure() {
        let p = tp("control-capital", PromptKind::Control);
        let r = run(
            "bf16",
            "control-capital",
            (0..96).collect(),
            vec![11.354; 96],
            None,
        );
        let mut t = RunTable::new("x");
        t.push(ArmObservation::of(&p, &r, &stops(), 96, &|i| {
            format!("<{i}>")
        }));
        let e = format!("{}", t.assert_controls_terminated().unwrap_err());
        eprintln!("{e}");
        assert!(e.contains("does not terminate is a broken control"), "{e}");
        assert!(t.rows[0].is_degenerate());
        assert_eq!(t.rows[0].termination(), "MAX-STEPS(96) NOT-TERMINATED");
    }

    #[test]
    fn a_confident_repetition_loop_is_flagged_even_though_its_margin_is_high() {
        let p = tp("historical-completion", PromptKind::OpenEnded);
        let unit = [1000u32, 1001, 1002, 1003, 1004];
        let toks: Vec<u32> = unit.iter().cycle().take(96).copied().collect();
        let r = run(
            "bf16",
            "historical-completion",
            toks,
            vec![11.354; 96],
            None,
        );
        let o = ArmObservation::of(&p, &r, &stops(), 96, &|i| format!("<{i}>"));
        eprintln!(
            "median margin {:.3}, health {}",
            o.median_margin,
            o.health()
        );
        assert!(o.median_margin > 10.0, "fixture lost its high margin");
        assert!(o.repeated_tail.is_some(), "repetition loop not detected");
        assert!(o.health().contains("repetition-loop"));
        assert!(o.health().contains("did-not-end-turn"));
        assert_eq!(o.evidence_class(), "DESCRIPTIVE-ONLY");
    }

    #[test]
    fn a_healthy_control_passes_and_reads_as_ab_evidence() {
        let p = tp("control-arithmetic", PromptKind::Control);
        let mut toks: Vec<u32> = vec![19, 20, 21];
        toks.push(106);
        let r = run(
            "bf16",
            "control-arithmetic",
            toks,
            vec![16.4, 17.8, 15.9, 12.0],
            Some(106),
        );
        let mut t = RunTable::new("x");
        t.push(ArmObservation::of(&p, &r, &stops(), 96, &|i| {
            if i == 106 {
                "<turn|>".into()
            } else {
                format!("<{i}>")
            }
        }));
        t.assert_controls_terminated().unwrap();
        assert_eq!(t.rows[0].evidence_class(), "A/B");
        assert!(t.rows[0]
            .termination()
            .starts_with("ENDED-TURN 106 (<turn|>)"));
        assert!(!t.rows[0].is_degenerate());
        eprintln!("{t}");
    }

    #[test]
    fn a_stop_token_emitted_mid_stream_is_reported_as_a_leak() {
        let p = tp("openended-explain", PromptKind::OpenEnded);
        let r = run(
            "fp8",
            "openended-explain",
            vec![9, 106, 9, 1, 9, 106],
            vec![3.0; 6],
            Some(106),
        );
        let o = ArmObservation::of(&p, &r, &stops(), 96, &|i| format!("<{i}>"));
        eprintln!("{}", o.health());
        assert_eq!(o.leaked_stop_ids, vec![106, 1]);
        assert!(o.health().contains("stop-token-leak"));
    }

    #[test]
    fn the_default_ab_selection_is_controls_only() {
        let pack = PromptPack {
            model_repo: "m".into(),
            snapshot: "s".into(),
            template_digest: "d".into(),
            template_bytes: 1,
            stop_ids: vec![1],
            stop_source: "t".into(),
            prompts: vec![
                tp("control-a", PromptKind::Control),
                tp("open-a", PromptKind::OpenEnded),
                tp("control-b", PromptKind::Control),
            ],
        };
        std::env::remove_var("NV_FP8_INCLUDE_OPEN_ENDED");
        let sel = ab_prompts(&pack);
        eprintln!("{}", describe_selection(&pack, &sel));
        assert_eq!(sel.len(), 2);
        assert!(sel.iter().all(|p| is_ab_evidence(p)));
    }

    fn dist(label: &str, kind: PromptKind, kl: f64, steps: usize) -> DistributionalSummary {
        let mut d = DistributionalSummary::new(label, kind);
        for _ in 0..steps {
            d.push(StepDelta {
                reference_top1: 5,
                candidate_top1: 5,
                reference_margin: 4.0,
                kl_nats: kl,
                max_abs_logit_delta: 0.01,
                rho: 0.0,
                rank_of_reference_top1: 0,
                bit_identical: kl == 0.0,
            });
        }
        d
    }

    #[test]
    fn relative_perplexity_is_exp_of_the_mean_kl_and_separates_the_two_rejects() {
        assert!((relative_perplexity_pct(1e-3) - 0.10005).abs() < 1e-4);
        let contested = relative_perplexity_pct(7.170e-3);
        let broken = relative_perplexity_pct(2.899e-1);
        let accepted = relative_perplexity_pct(1.269e-4);
        eprintln!("accept {accepted:+.4}%  contested {contested:+.4}%  broken {broken:+.4}%");
        assert!((contested - 0.7196).abs() < 1e-3, "{contested}");
        assert!((broken - 33.64).abs() < 0.05, "{broken}");
        assert!(
            broken / contested > 40.0,
            "the two REJECTs must be two orders of magnitude apart in perplexity, or printing it \
             buys nothing: {broken} vs {contested}"
        );
    }

    #[test]
    fn a_max_driven_by_one_prompt_is_self_evident_in_the_instrument_block() {
        let mut dists = vec![dist("control-loud", PromptKind::Control, 7.170e-3, 24)];
        for i in 0..10 {
            dists.push(dist(
                &format!("control-quiet-{i}"),
                PromptKind::Control,
                0.0,
                24,
            ));
        }
        let r = InstrumentReport::new("gemma-4-E4B-it :: W8G128", &dists);
        let s = format!("{r}");
        eprintln!("{s}");
        assert_eq!(r.control_count(), 11);
        assert_eq!(r.controls_above_floor(), 1);
        assert!((r.worst_control_mean_kl() - 7.170e-3).abs() < 1e-12);
        assert!(
            r.controls_worst_first()[0].prompt_label == "control-loud",
            "the table must sort worst-first"
        );
        assert!(s.contains("1/11 control prompt(s) carry mean KL"), "{s}");
        assert!(s.contains("IS one prompt's number"), "{s}");
        assert!(
            s.contains("+0.7196%"),
            "the relative perplexity must be printed: {s}"
        );
        assert!(s.contains("MAX-OVER-PROMPTS CAVEAT"), "{s}");
        assert!(
            r.pooled_control_mean_kl() < r.worst_control_mean_kl() / 10.0,
            "pooling over steps must expose that 10 of 11 prompts contribute nothing: pooled {} \
             vs worst {}",
            r.pooled_control_mean_kl(),
            r.worst_control_mean_kl()
        );
    }

    #[test]
    fn open_ended_rows_print_in_their_own_section_and_never_enter_the_evidence_numbers() {
        let dists = vec![
            dist("control-arith", PromptKind::Control, 1.0e-5, 24),
            dist("openended-explain", PromptKind::OpenEnded, 9.9e-1, 24),
        ];
        let r = InstrumentReport::new("selftest", &dists);
        let s = format!("{r}");
        eprintln!("{s}");
        assert_eq!(r.control_count(), 1);
        assert!((r.worst_control_mean_kl() - 1.0e-5).abs() < 1e-12);
        assert_eq!(r.open_ended_worst_first().len(), 1);
        assert!(s.contains("CONTROL rows (A/B EVIDENCE)"), "{s}");
        assert!(s.contains("OPEN-ENDED rows (DESCRIPTIVE ONLY"), "{s}");
        assert!(
            r.relative_perplexity_pct() < 0.01,
            "an open-ended row must not move the reported perplexity: {}",
            r.relative_perplexity_pct()
        );
    }

    #[test]
    fn resolve_pack_refuses_a_directory_with_no_matching_template() {
        std::env::remove_var("NV_CHAT_EVAL_PACK");
        let e = format!(
            "{}",
            resolve_pack(Path::new("/nonexistent-weights-dir")).unwrap_err()
        );
        eprintln!("{e}");
        assert!(e.contains("prompt pack"), "{e}");
        assert!(e.contains("chat_eval"), "{e}");
    }
}
