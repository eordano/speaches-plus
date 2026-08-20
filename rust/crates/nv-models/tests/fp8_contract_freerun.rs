#![allow(dead_code)]

#[path = "fp8_contract_prompts.rs"]
mod prompts;

use prompts::{
    ArmObservation, FreeRun, PromptKind, PromptPack, RunTable, StopSet, TemplatedPrompt, WHY_A_PACK,
};
use std::fmt;

pub const CONTRACT_DOC: &str = "docs/book/04.1-fp8.md";

pub const MIN_FREE_RUNNING_WINDOW: usize = 64;
pub const MIN_FREE_RUNNING_AGREEMENT: f64 = 0.90;
pub const MAX_PERPLEXITY_RATIO: f64 = 1.05;
pub const MAX_MEDIAN_REFERENCE_RANK: usize = 2;
pub const MAX_P95_REFERENCE_RANK: usize = 8;
pub const RESYNC_RUN: usize = 4;
pub const RECOVERY_RATE: f64 = 0.50;
pub const MIN_OPEN_ENDED_SCORED_POSITIONS: usize = 16;

pub const WHY_OPEN_ENDED_QUALITY: &str = "WHY THE QUALITY EVIDENCE MUST INCLUDE OPEN-ENDED ROWS. \
     Free-running agreement on an open-ended prompt is not reproducible on this box (66 / 78 / 96 \
     tokens across three byte-identical runs), which is why the free-running check is CONTROL-only. \
     Teacher-forced replay has no such excuse: it re-drives a FIXED token sequence, so it is \
     deterministic on any prompt, open-ended included. Scoring it on controls only left the gate \
     measuring the class least able to expose quantization damage - the 16 controls behind the \
     current flip sit at median top-2 margins of 15.1 to 20.1, where an fp8 perturbation cannot \
     move the argmax - so 16/16 at 100% was close to arithmetically guaranteed rather than \
     evidence.";

pub const WHY_REFERENCE_MUST_TERMINATE: &str = "WHY A TRUNCATED REFERENCE CANNOT CARRY THE \
     QUALITY HALF. PromptQuality has recorded reference_ended_turn since the field was added, and \
     until 2026-08-09 nothing read it. evaluate_default_flip_suite denies on !ended_turn for RUN \
     TABLE rows, but the real-weight gate only ever pushed its A/B (CONTROL) rows into that \
     table, so an OPEN-ENDED reference that ran away was invisible unless it also repeated on a \
     detectable period - which is the exact 2026-08-06 methodology failure, minus the repetition \
     that made it obvious. Raise NV_FP8_CONTRACT_WINDOW and re-measure rather than scoring both \
     arms on a tail the model never chose to stop at.";

pub const WHY_THE_AB_MUST_HOLD_THE_FFN_FIXED: &str = "WHY THIS GATE PINS NV_G4_WGPU_W8_FFN. The \
     change under test is the ATTENTION projection format. W8_FFN_DEFAULT became \"all\" in \
     eecb69c3b (2026-08-09), the same commit that flipped ATTN_FP8_FMT_DEFAULT to int8, so \
     leaving it alone would (a) put the reference arm's FFN in int8 too, confounding the A/B, and \
     (b) make the reference arm UNCONSTRUCTIBLE: gemma4_wgpu::build_pipelines gates the whole \
     fp8/int8 pipeline set on attn_fp8 alone, so a bf16-attn arm (set_attn_variant off) plus an \
     int8 FFN fails at \
     load with \"fp8 projection uploaded without fp8 pipelines\". Measured 2026-08-09: this gate \
     panicked there after 107 s, before scoring a single prompt. Fixing (b) properly is a \
     gemma4_wgpu.rs change - pass attn_fp8 || w8_ffn to build_pipelines. Override the pin with \
     NV_FP8_CONTRACT_FFN.";

#[derive(Clone, Debug)]
pub struct FreeRunningTrajectory {
    pub arm: String,
    pub prompt: Vec<u32>,
    pub tokens: Vec<u32>,
}

impl FreeRunningTrajectory {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

pub fn free_running_greedy<F>(
    arm: &str,
    prompt: &[u32],
    steps: usize,
    mut step: F,
) -> anyhow::Result<FreeRunningTrajectory>
where
    F: FnMut(u32) -> anyhow::Result<u32>,
{
    anyhow::ensure!(!prompt.is_empty(), "free_running_greedy needs a prompt");
    anyhow::ensure!(steps > 0, "free_running_greedy needs steps > 0");
    let mut last = 0u32;
    for t in prompt {
        last = step(*t)?;
    }
    let mut tokens = Vec::with_capacity(steps);
    tokens.push(last);
    for _ in 1..steps {
        last = step(last)?;
        tokens.push(last);
    }
    Ok(FreeRunningTrajectory {
        arm: arm.to_string(),
        prompt: prompt.to_vec(),
        tokens,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DivergenceShape {
    Identical,
    Recovered,
    Cascaded,
}

impl fmt::Display for DivergenceShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DivergenceShape::Identical => write!(f, "IDENTICAL"),
            DivergenceShape::Recovered => write!(f, "RECOVERED"),
            DivergenceShape::Cascaded => write!(f, "CASCADED"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FreeRunningComparison {
    pub reference_arm: String,
    pub candidate_arm: String,
    pub agree: usize,
    pub total: usize,
    pub first_divergence: Option<usize>,
    pub common_prefix: usize,
    pub post_divergence_agree: usize,
    pub post_divergence_total: usize,
    pub resynced_at: Option<usize>,
    pub shape: DivergenceShape,
    pub reference_tokens: Vec<u32>,
    pub candidate_tokens: Vec<u32>,
}

impl FreeRunningComparison {
    pub fn agreement(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.agree as f64 / self.total as f64
        }
    }
    pub fn post_divergence_rate(&self) -> f64 {
        if self.post_divergence_total == 0 {
            1.0
        } else {
            self.post_divergence_agree as f64 / self.post_divergence_total as f64
        }
    }
    pub fn extra_occurrences_of(&self, ids: &[u32]) -> usize {
        let count = |v: &[u32]| v.iter().filter(|t| ids.contains(t)).count();
        count(&self.candidate_tokens).saturating_sub(count(&self.reference_tokens))
    }
}

impl fmt::Display for FreeRunningComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FREE-RUNNING (serving metric) [{}] vs [{}]: argmax agreement {}/{} ({:.1}%), first divergence {:?}, \
             post-divergence agreement {}/{} ({:.1}%), resynced_at {:?}, shape {}",
            self.candidate_arm,
            self.reference_arm,
            self.agree,
            self.total,
            100.0 * self.agreement(),
            self.first_divergence,
            self.post_divergence_agree,
            self.post_divergence_total,
            100.0 * self.post_divergence_rate(),
            self.resynced_at,
            self.shape
        )
    }
}

pub fn compare_free_running(
    reference: &FreeRunningTrajectory,
    candidate: &FreeRunningTrajectory,
) -> FreeRunningComparison {
    let total = reference.tokens.len().min(candidate.tokens.len());
    let r = &reference.tokens[..total];
    let c = &candidate.tokens[..total];
    let agree = r.iter().zip(c.iter()).filter(|(a, b)| a == b).count();
    let first_divergence = r.iter().zip(c.iter()).position(|(a, b)| a != b);
    let common_prefix = first_divergence.unwrap_or(total);
    let (post_agree, post_total) = match first_divergence {
        None => (0usize, 0usize),
        Some(i) => {
            let tail = i + 1;
            (
                r[tail..]
                    .iter()
                    .zip(c[tail..].iter())
                    .filter(|(a, b)| a == b)
                    .count(),
                total - tail,
            )
        }
    };
    let resynced_at = first_divergence.and_then(|i| {
        (i + 1..total.saturating_sub(RESYNC_RUN - 1))
            .find(|&j| (0..RESYNC_RUN).all(|d| r[j + d] == c[j + d]))
    });
    let shape = if first_divergence.is_none() {
        DivergenceShape::Identical
    } else if post_total > 0
        && post_agree as f64 / post_total as f64 >= RECOVERY_RATE
        && resynced_at.is_some()
    {
        DivergenceShape::Recovered
    } else {
        DivergenceShape::Cascaded
    };
    FreeRunningComparison {
        reference_arm: reference.arm.clone(),
        candidate_arm: candidate.arm.clone(),
        agree,
        total,
        first_divergence,
        common_prefix,
        post_divergence_agree: post_agree,
        post_divergence_total: post_total,
        resynced_at,
        shape,
        reference_tokens: r.to_vec(),
        candidate_tokens: c.to_vec(),
    }
}

#[derive(Clone, Debug)]
pub struct ForcedContextUpperBound {
    reference_arm: String,
    candidate_arm: String,
    agree: usize,
    total: usize,
    first_divergence: Option<usize>,
}

impl ForcedContextUpperBound {
    pub fn upper_bound_agreement_is_not_serving_evidence(&self) -> (usize, usize) {
        (self.agree, self.total)
    }
    pub fn first_divergence(&self) -> Option<usize> {
        self.first_divergence
    }
    pub fn why_this_is_not_serving_evidence() -> &'static str {
        "Forced context re-seeds the candidate with the reference token after every step, so a \
         single wrong argmax cannot compound. Serving never does this. On Gemma4-31B NVFP4 / wgpu \
         the same fp8 attention arm scored near-perfect forced while its free-running agreement \
         collapsed (numbers: perf/runs.jsonl)."
    }
}

impl fmt::Display for ForcedContextUpperBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FORCED-CONTEXT UPPER BOUND (NOT a serving metric) [{}] vs [{}]: {}/{} at first divergence {:?}. {}",
            self.candidate_arm,
            self.reference_arm,
            self.agree,
            self.total,
            self.first_divergence,
            Self::why_this_is_not_serving_evidence()
        )
    }
}

pub fn forced_context_upper_bound<F>(
    candidate_arm: &str,
    reference: &FreeRunningTrajectory,
    mut step: F,
) -> anyhow::Result<ForcedContextUpperBound>
where
    F: FnMut(u32) -> anyhow::Result<u32>,
{
    let mut last = 0u32;
    for t in &reference.prompt {
        last = step(*t)?;
    }
    let mut agree = 0usize;
    let mut first_divergence = None;
    if last == reference.tokens[0] {
        agree += 1;
    } else {
        first_divergence = Some(0);
    }
    for i in 0..reference.tokens.len() - 1 {
        let out = step(reference.tokens[i])?;
        if out == reference.tokens[i + 1] {
            agree += 1;
        } else if first_divergence.is_none() {
            first_divergence = Some(i + 1);
        }
    }
    Ok(ForcedContextUpperBound {
        reference_arm: reference.arm.clone(),
        candidate_arm: candidate_arm.to_string(),
        agree,
        total: reference.tokens.len(),
        first_divergence,
    })
}

#[derive(Clone, Debug)]
pub struct TeacherForcedQuality {
    pub arm: String,
    pub nll_mean: f64,
    pub perplexity: f64,
    pub reference_rank: Vec<usize>,
    pub reference_margin: Vec<f32>,
}

impl TeacherForcedQuality {
    pub fn median_rank(&self) -> usize {
        let mut v = self.reference_rank.clone();
        v.sort_unstable();
        if v.is_empty() {
            0
        } else {
            v[v.len() / 2]
        }
    }
    pub fn p95_rank(&self) -> usize {
        let mut v = self.reference_rank.clone();
        v.sort_unstable();
        if v.is_empty() {
            0
        } else {
            v[(v.len() * 95 / 100).min(v.len() - 1)]
        }
    }
    pub fn max_rank(&self) -> usize {
        self.reference_rank.iter().copied().max().unwrap_or(0)
    }
    pub fn median_margin(&self) -> f32 {
        let mut v = self.reference_margin.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() {
            0.0
        } else {
            v[v.len() / 2]
        }
    }
}

impl fmt::Display for TeacherForcedQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TEACHER-FORCED QUALITY [{}] on the reference trajectory: mean NLL {:.4} nats, perplexity {:.4}, \
             reference-token rank median {} p95 {} max {}, top1-minus-reference logit margin median {:.4}",
            self.arm,
            self.nll_mean,
            self.perplexity,
            self.median_rank(),
            self.p95_rank(),
            self.max_rank(),
            self.median_margin()
        )
    }
}

fn log_softmax_at(logits: &[f32], target: u32) -> (f64, usize, f32) {
    let mut max = f32::NEG_INFINITY;
    for v in logits {
        if *v > max {
            max = *v;
        }
    }
    let mut sum = 0f64;
    for v in logits {
        sum += ((*v - max) as f64).exp();
    }
    let t = logits[target as usize];
    let nll = -(((t - max) as f64) - sum.ln());
    let rank = 1 + logits.iter().filter(|v| **v > t).count();
    (nll, rank, max - t)
}

pub fn teacher_forced_quality<F>(
    arm: &str,
    reference: &FreeRunningTrajectory,
    mut step_logits: F,
) -> anyhow::Result<TeacherForcedQuality>
where
    F: FnMut(u32) -> anyhow::Result<(u32, Vec<f32>)>,
{
    let mut logits = Vec::new();
    for t in &reference.prompt {
        logits = step_logits(*t)?.1;
    }
    let mut nll_sum = 0f64;
    let mut reference_rank = Vec::new();
    let mut reference_margin = Vec::new();
    let mut push = |logits: &[f32], target: u32| -> anyhow::Result<()> {
        anyhow::ensure!(
            (target as usize) < logits.len(),
            "target {target} outside logits of len {}",
            logits.len()
        );
        let (nll, rank, margin) = log_softmax_at(logits, target);
        nll_sum += nll;
        reference_rank.push(rank);
        reference_margin.push(margin);
        Ok(())
    };
    push(&logits, reference.tokens[0])?;
    for i in 0..reference.tokens.len() - 1 {
        let (_, l) = step_logits(reference.tokens[i])?;
        push(&l, reference.tokens[i + 1])?;
    }
    let n = reference.tokens.len() as f64;
    let nll_mean = nll_sum / n;
    Ok(TeacherForcedQuality {
        arm: arm.to_string(),
        nll_mean,
        perplexity: nll_mean.exp(),
        reference_rank,
        reference_margin,
    })
}

pub fn free_running_greedy_with_logits<F>(
    arm: &str,
    prompt: &[u32],
    steps: usize,
    step_logits: F,
) -> anyhow::Result<(FreeRunningTrajectory, TeacherForcedQuality)>
where
    F: FnMut(u32) -> anyhow::Result<(u32, Vec<f32>)>,
{
    let (t, q, _) = free_running_greedy_keeping_logits(arm, prompt, steps, step_logits)?;
    Ok((t, q))
}

pub fn free_running_greedy_keeping_logits<F>(
    arm: &str,
    prompt: &[u32],
    steps: usize,
    mut step_logits: F,
) -> anyhow::Result<(FreeRunningTrajectory, TeacherForcedQuality, Vec<Vec<f32>>)>
where
    F: FnMut(u32) -> anyhow::Result<(u32, Vec<f32>)>,
{
    anyhow::ensure!(!prompt.is_empty(), "free_running_greedy needs a prompt");
    anyhow::ensure!(steps > 0, "free_running_greedy needs steps > 0");
    let mut last = 0u32;
    let mut logits = Vec::new();
    for t in prompt {
        let (o, l) = step_logits(*t)?;
        last = o;
        logits = l;
    }
    let mut tokens = Vec::with_capacity(steps);
    let mut nll_sum = 0f64;
    let mut reference_rank = Vec::with_capacity(steps);
    let mut reference_margin = Vec::with_capacity(steps);
    let mut kept = Vec::with_capacity(steps);
    for i in 0..steps {
        if i > 0 {
            let (o, l) = step_logits(last)?;
            last = o;
            logits = l;
        }
        let (nll, rank, margin) = log_softmax_at(&logits, last);
        nll_sum += nll;
        reference_rank.push(rank);
        reference_margin.push(margin);
        tokens.push(last);
        kept.push(logits.clone());
    }
    let nll_mean = nll_sum / steps as f64;
    Ok((
        FreeRunningTrajectory {
            arm: arm.to_string(),
            prompt: prompt.to_vec(),
            tokens,
        },
        TeacherForcedQuality {
            arm: arm.to_string(),
            nll_mean,
            perplexity: nll_mean.exp(),
            reference_rank,
            reference_margin,
        },
        kept,
    ))
}

pub fn rank_and_margin_of(logits: &[f32], token: u32) -> (usize, f32) {
    let t = logits[token as usize];
    let mut top = f32::NEG_INFINITY;
    for v in logits {
        if *v > top {
            top = *v;
        }
    }
    (1 + logits.iter().filter(|v| **v > t).count(), top - t)
}

pub fn forced_context_replay<F>(
    candidate_arm: &str,
    reference: &FreeRunningTrajectory,
    mut step_logits: F,
) -> anyhow::Result<(ForcedContextUpperBound, TeacherForcedQuality)>
where
    F: FnMut(u32) -> anyhow::Result<(u32, Vec<f32>)>,
{
    let mut out = 0u32;
    let mut logits = Vec::new();
    for t in &reference.prompt {
        let (o, l) = step_logits(*t)?;
        out = o;
        logits = l;
    }
    let mut agree = 0usize;
    let mut first_divergence = None;
    let mut nll_sum = 0f64;
    let mut reference_rank = Vec::new();
    let mut reference_margin = Vec::new();
    for i in 0..reference.tokens.len() {
        if i > 0 {
            let (o, l) = step_logits(reference.tokens[i - 1])?;
            out = o;
            logits = l;
        }
        let target = reference.tokens[i];
        if out == target {
            agree += 1;
        } else if first_divergence.is_none() {
            first_divergence = Some(i);
        }
        let (nll, rank, margin) = log_softmax_at(&logits, target);
        nll_sum += nll;
        reference_rank.push(rank);
        reference_margin.push(margin);
    }
    let n = reference.tokens.len();
    let nll_mean = nll_sum / n as f64;
    Ok((
        ForcedContextUpperBound {
            reference_arm: reference.arm.clone(),
            candidate_arm: candidate_arm.to_string(),
            agree,
            total: n,
            first_divergence,
        },
        TeacherForcedQuality {
            arm: candidate_arm.to_string(),
            nll_mean,
            perplexity: nll_mean.exp(),
            reference_rank,
            reference_margin,
        },
    ))
}

pub fn free_running_with_substitution<F>(
    arm: &str,
    prompt: &[u32],
    steps: usize,
    at: usize,
    substitute: u32,
    mut step: F,
) -> anyhow::Result<FreeRunningTrajectory>
where
    F: FnMut(u32) -> anyhow::Result<u32>,
{
    anyhow::ensure!(at < steps, "substitution index {at} outside window {steps}");
    let mut last = 0u32;
    for t in prompt {
        last = step(*t)?;
    }
    let mut tokens = Vec::with_capacity(steps);
    for i in 0..steps {
        if i > 0 {
            last = step(last)?;
        }
        if i == at {
            last = substitute;
        }
        tokens.push(last);
    }
    Ok(FreeRunningTrajectory {
        arm: arm.to_string(),
        prompt: prompt.to_vec(),
        tokens,
    })
}

pub fn nth_best_token(logits: &[f32], n: usize) -> u32 {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|a, b| {
        logits[*b as usize]
            .partial_cmp(&logits[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx[n.min(idx.len() - 1)]
}

#[derive(Clone, Debug)]
pub struct DefaultFlipEvidence<'a> {
    pub change: &'a str,
    pub backend: &'a str,
    pub model: &'a str,
    pub free_running: &'a FreeRunningComparison,
    pub reference_quality: Option<&'a TeacherForcedQuality>,
    pub candidate_quality: Option<&'a TeacherForcedQuality>,
    pub speed_delta_pct: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DefaultFlipVerdict {
    Allow,
    Deny(Vec<String>),
}

impl DefaultFlipVerdict {
    pub fn is_allow(&self) -> bool {
        matches!(self, DefaultFlipVerdict::Allow)
    }
}

impl fmt::Display for DefaultFlipVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DefaultFlipVerdict::Allow => write!(f, "ALLOW default flip"),
            DefaultFlipVerdict::Deny(rs) => {
                write!(f, "DENY default flip ({} reasons)", rs.len())?;
                for r in rs {
                    write!(f, "\n  - {r}")?;
                }
                Ok(())
            }
        }
    }
}

pub fn evaluate_default_flip(e: &DefaultFlipEvidence) -> DefaultFlipVerdict {
    let mut reasons = Vec::new();
    let fr = e.free_running;
    if fr.total < MIN_FREE_RUNNING_WINDOW {
        reasons.push(format!(
            "free-running window {} < required {MIN_FREE_RUNNING_WINDOW} tokens",
            fr.total
        ));
    }
    if fr.agreement() < MIN_FREE_RUNNING_AGREEMENT {
        reasons.push(format!(
            "free-running agreement {}/{} = {:.1}% < required {:.0}%",
            fr.agree,
            fr.total,
            100.0 * fr.agreement(),
            100.0 * MIN_FREE_RUNNING_AGREEMENT
        ));
    }
    if fr.shape == DivergenceShape::Cascaded {
        reasons.push(format!(
            "divergence shape is {} (post-divergence agreement {:.1}%, never resynced)",
            fr.shape,
            100.0 * fr.post_divergence_rate()
        ));
    }
    match (e.reference_quality, e.candidate_quality) {
        (Some(r), Some(c)) => {
            let ratio = c.perplexity / r.perplexity;
            if ratio > MAX_PERPLEXITY_RATIO {
                reasons.push(format!(
                    "teacher-forced perplexity ratio {ratio:.4} > allowed {MAX_PERPLEXITY_RATIO:.2} ({:.4} vs {:.4})",
                    c.perplexity, r.perplexity
                ));
            }
            if c.median_rank() > MAX_MEDIAN_REFERENCE_RANK {
                reasons.push(format!(
                    "median reference-token rank {} > allowed {MAX_MEDIAN_REFERENCE_RANK}",
                    c.median_rank()
                ));
            }
            if c.p95_rank() > MAX_P95_REFERENCE_RANK {
                reasons.push(format!(
                    "p95 reference-token rank {} > allowed {MAX_P95_REFERENCE_RANK}",
                    c.p95_rank()
                ));
            }
        }
        _ => reasons.push(
            "no teacher-forced perplexity / rank evidence supplied; argmax agreement alone is not \
             sufficient for a default flip"
                .to_string(),
        ),
    }
    match e.speed_delta_pct {
        None => reasons.push("no measured speed delta supplied".to_string()),
        Some(d) if d >= 0.0 => reasons.push(format!(
            "measured speed delta {d:+.1}% is not an improvement"
        )),
        Some(_) => {}
    }
    if reasons.is_empty() {
        DefaultFlipVerdict::Allow
    } else {
        DefaultFlipVerdict::Deny(reasons)
    }
}

#[derive(Clone, Debug)]
pub struct PromptQuality {
    pub label: String,
    pub kind: PromptKind,
    pub reference_ended_turn: bool,
    pub reference_loops: bool,
    pub reference: TeacherForcedQuality,
    pub candidate: TeacherForcedQuality,
}

impl PromptQuality {
    pub fn kind_label(&self) -> &'static str {
        if self.kind == PromptKind::Control {
            "CONTROL"
        } else {
            "OPEN-ENDED"
        }
    }
    pub fn scored(&self) -> usize {
        self.candidate
            .reference_rank
            .len()
            .min(self.reference.reference_rank.len())
    }
    pub fn perplexity_ratio(&self) -> f64 {
        self.candidate.perplexity / self.reference.perplexity
    }
}

impl fmt::Display for PromptQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {:?}: teacher-forced over {} position(s), perplexity {:.6} vs reference {:.6} \
             (ratio {:.6}, allowed {MAX_PERPLEXITY_RATIO:.2}), candidate reference-token rank \
             median {} p95 {} max {}; reference trajectory {}{}",
            self.kind_label(),
            self.label,
            self.scored(),
            self.candidate.perplexity,
            self.reference.perplexity,
            self.perplexity_ratio(),
            self.candidate.median_rank(),
            self.candidate.p95_rank(),
            self.candidate.max_rank(),
            if self.reference_ended_turn {
                "ended its turn"
            } else {
                "hit max-steps"
            },
            if self.reference_loops {
                " and is a REPETITION LOOP"
            } else {
                ""
            }
        )
    }
}

pub fn teacher_forced_reasons(quality: &[PromptQuality]) -> Vec<String> {
    let mut reasons = Vec::new();
    if quality.is_empty() {
        reasons.push(
            "no teacher-forced perplexity / rank evidence supplied; argmax agreement alone is not \
             sufficient for a default flip"
                .to_string(),
        );
        return reasons;
    }
    let open = quality
        .iter()
        .filter(|q| q.kind == PromptKind::OpenEnded)
        .count();
    if open == 0 {
        reasons.push(format!(
            "teacher-forced evidence covers {} CONTROL prompt(s) and 0 OPEN-ENDED prompt(s); a \
             control-only quality score cannot deny a flip. {WHY_OPEN_ENDED_QUALITY}",
            quality.len()
        ));
    }
    for q in quality.iter() {
        let kind = q.kind_label();
        let label = &q.label;
        if q.scored() == 0 {
            reasons.push(format!(
                "{kind} prompt {label:?} carries a teacher-forced entry scored over 0 positions, \
                 which is an empty measurement wearing the shape of evidence"
            ));
            continue;
        }
        if q.kind == PromptKind::OpenEnded && q.scored() < MIN_OPEN_ENDED_SCORED_POSITIONS {
            reasons.push(format!(
                "OPEN-ENDED prompt {label:?} was teacher-forced over only {} position(s) < \
                 required {MIN_OPEN_ENDED_SCORED_POSITIONS}; a trajectory that short cannot carry \
                 the open-ended half of the evidence, and would let the coverage requirement be \
                 satisfied vacuously. The floor is calibrated on the measured split: on \
                 Gemma4-31B-IT-NVFP4 the 25 CONTROL rows terminate after 2 to 4 tokens while the \
                 3 OPEN-ENDED rows run 27, 44 and 63",
                q.scored()
            ));
        }
        if q.reference_loops {
            reasons.push(format!(
                "{kind} prompt {label:?}: the REFERENCE trajectory being replayed is itself a \
                 repetition loop, so both arms are scored on degenerate text and the ratio it \
                 produces is two loops agreeing with each other. {WHY_A_PACK}"
            ));
        }
        if !q.reference_ended_turn {
            reasons.push(format!(
                "{kind} prompt {label:?}: the REFERENCE trajectory being replayed hit max-steps \
                 without ending its turn, so both arms are teacher-forced onto a truncated \
                 trajectory the model never chose to stop at. {WHY_REFERENCE_MUST_TERMINATE}"
            ));
        }
        let ratio = q.perplexity_ratio();
        if !ratio.is_finite() {
            reasons.push(format!(
                "{kind} prompt {label:?} teacher-forced perplexity ratio is not finite ({:.4} vs \
                 {:.4})",
                q.candidate.perplexity, q.reference.perplexity
            ));
        } else if ratio > MAX_PERPLEXITY_RATIO {
            reasons.push(format!(
                "{kind} prompt {label:?} teacher-forced perplexity ratio {ratio:.4} > allowed \
                 {MAX_PERPLEXITY_RATIO:.2} ({:.4} vs {:.4})",
                q.candidate.perplexity, q.reference.perplexity
            ));
        }
        if q.candidate.median_rank() > MAX_MEDIAN_REFERENCE_RANK {
            reasons.push(format!(
                "{kind} prompt {label:?} median reference-token rank {} > allowed \
                 {MAX_MEDIAN_REFERENCE_RANK}",
                q.candidate.median_rank()
            ));
        }
        if q.candidate.p95_rank() > MAX_P95_REFERENCE_RANK {
            reasons.push(format!(
                "{kind} prompt {label:?} p95 reference-token rank {} > allowed \
                 {MAX_P95_REFERENCE_RANK}",
                q.candidate.p95_rank()
            ));
        }
    }
    reasons
}

pub struct SuiteFlipEvidence<'a> {
    pub change: &'a str,
    pub backend: &'a str,
    pub model: &'a str,
    pub table: &'a RunTable,
    pub per_prompt: &'a [(String, PromptKind, FreeRunningComparison)],
    pub per_prompt_quality: &'a [PromptQuality],
    pub speed_delta_pct: Option<f64>,
}

pub fn evaluate_default_flip_suite(e: &SuiteFlipEvidence) -> DefaultFlipVerdict {
    let mut reasons = Vec::new();
    let controls: Vec<&(String, PromptKind, FreeRunningComparison)> = e
        .per_prompt
        .iter()
        .filter(|(_, k, _)| *k == PromptKind::Control)
        .collect();
    if e.per_prompt.len() < 2 {
        reasons.push(format!(
            "{} prompt(s) measured; a default flip needs at least 2 so one ambiguous position \
             cannot pass as a result",
            e.per_prompt.len()
        ));
    }
    if controls.is_empty() {
        reasons.push(
            "no CONTROL prompt measured. Token-count windows are NOT the bar any more: a \
             templated prompt ends its turn, so a 64-token window only exists when something \
             failed to terminate."
                .to_string(),
        );
    }
    for (label, _, c) in &controls {
        if c.first_divergence.is_some() {
            reasons.push(format!(
                "CONTROL prompt {label:?} diverged at step {:?} (agreement {}/{}, shape {})",
                c.first_divergence, c.agree, c.total, c.shape
            ));
        }
    }
    for r in e.table.rows.iter() {
        let kind = if r.kind == PromptKind::Control {
            "CONTROL"
        } else {
            "OPEN-ENDED"
        };
        if !r.ended_turn {
            reasons.push(format!(
                "{kind} row {:?} on arm {:?} hit max-steps without ending its turn",
                r.prompt_label, r.arm
            ));
        }
        if r.repeated_tail.is_some() {
            reasons.push(format!(
                "{kind} row {:?} on arm {:?} ends in a repetition loop: {}",
                r.prompt_label,
                r.arm,
                r.health()
            ));
        }
        if !r.leaked_stop_ids.is_empty() {
            reasons.push(format!(
                "{kind} row {:?} on arm {:?} leaked stop tokens {:?} mid-stream",
                r.prompt_label, r.arm, r.leaked_stop_ids
            ));
        }
    }
    reasons.extend(teacher_forced_reasons(e.per_prompt_quality));
    for r in e.table.rows.iter() {
        if !e
            .per_prompt_quality
            .iter()
            .any(|q| q.label == r.prompt_label)
        {
            let reason = format!(
                "{} row {:?} was free-run but never teacher-forced; a prompt that appears in the \
                 run table and not in the quality evidence is a prompt this gate cannot score. \
                 {WHY_OPEN_ENDED_QUALITY}",
                if r.kind == PromptKind::Control {
                    "CONTROL"
                } else {
                    "OPEN-ENDED"
                },
                r.prompt_label
            );
            if !reasons.contains(&reason) {
                reasons.push(reason);
            }
        }
    }
    match e.speed_delta_pct {
        None => reasons.push("no measured speed delta supplied".to_string()),
        Some(d) if d >= 0.0 => reasons.push(format!(
            "measured speed delta {d:+.1}% is not an improvement"
        )),
        Some(_) => {}
    }
    if reasons.is_empty() {
        DefaultFlipVerdict::Allow
    } else {
        DefaultFlipVerdict::Deny(reasons)
    }
}

pub fn trajectory_from_free_run(prompt: &TemplatedPrompt, run: &FreeRun) -> FreeRunningTrajectory {
    FreeRunningTrajectory {
        arm: run.arm.clone(),
        prompt: prompt.ids.clone(),
        tokens: run.tokens.clone(),
    }
}

pub fn observe(
    prompt: &TemplatedPrompt,
    run: &FreeRun,
    stops: &StopSet,
    steps: usize,
    label_of: &dyn Fn(u32) -> String,
) -> ArmObservation {
    ArmObservation::of(prompt, run, stops, steps, label_of)
}

const MOCK_CYCLE: u32 = 50;
const MOCK_WINDOW: usize = 40;

fn mock_reference(t: u32) -> u32 {
    (t + 1) % MOCK_CYCLE
}

fn mock_cascading(t: u32) -> u32 {
    if t == 2 {
        900
    } else if t >= 900 {
        900 + ((t - 900 + 1) % 5)
    } else {
        mock_reference(t)
    }
}

fn mock_near_tie(t: u32) -> u32 {
    match t {
        2 => 777,
        777 => 4,
        _ => mock_reference(t),
    }
}

fn mock_cascading_traj(arm: &str, steps: usize) -> FreeRunningTrajectory {
    free_running_greedy(arm, &[0], steps, |t| Ok(mock_cascading(t))).unwrap()
}

fn mock_near_tie_traj(arm: &str, steps: usize) -> FreeRunningTrajectory {
    free_running_greedy(arm, &[0], steps, |t| Ok(mock_near_tie(t))).unwrap()
}

fn mock_reference_traj(arm: &str, steps: usize) -> FreeRunningTrajectory {
    free_running_greedy(arm, &[0], steps, |t| Ok(mock_reference(t))).unwrap()
}

#[test]
fn harness_reproduces_the_forced_vs_free_running_methodology_failure() {
    let reference = mock_reference_traj("bf16", MOCK_WINDOW);
    let candidate = mock_cascading_traj("fp8", MOCK_WINDOW);
    let cmp = compare_free_running(&reference, &candidate);
    let forced = forced_context_upper_bound("fp8", &reference, |t| Ok(mock_cascading(t))).unwrap();
    eprintln!("{cmp}");
    eprintln!("{forced}");
    let (fa, ft) = forced.upper_bound_agreement_is_not_serving_evidence();
    assert_eq!((fa, ft), (39, 40), "forced context must read 39/40");
    assert_eq!(forced.first_divergence(), Some(2));
    assert_eq!(cmp.first_divergence, Some(2));
    assert!(
        cmp.agreement() < 0.10,
        "free-running agreement must collapse, got {}/{}",
        cmp.agree,
        cmp.total
    );
    assert_eq!(cmp.shape, DivergenceShape::Cascaded);
    assert!(
        (fa as f64 / ft as f64) - cmp.agreement() > 0.85,
        "the two metrics must be far apart: forced {fa}/{ft} vs free-running {}/{}",
        cmp.agree,
        cmp.total
    );
}

#[test]
fn harness_distinguishes_a_benign_near_tie_from_a_cascade() {
    let reference = mock_reference_traj("bf16", MOCK_WINDOW);
    let benign = compare_free_running(&reference, &mock_near_tie_traj("near-tie", MOCK_WINDOW));
    let broken = compare_free_running(&reference, &mock_cascading_traj("cascade", MOCK_WINDOW));
    eprintln!("{benign}");
    eprintln!("{broken}");
    assert_eq!(benign.first_divergence, Some(2));
    assert_eq!(broken.first_divergence, Some(2));
    assert_eq!(benign.shape, DivergenceShape::Recovered);
    assert_eq!(broken.shape, DivergenceShape::Cascaded);
    assert_eq!(benign.resynced_at, Some(3));
    assert_eq!(broken.resynced_at, None);
    assert!(
        benign.agreement() > 0.95 && broken.agreement() < 0.10,
        "benign {:.3} vs broken {:.3}",
        benign.agreement(),
        broken.agreement()
    );
}

#[test]
fn forced_context_report_cannot_be_mistaken_for_a_serving_metric() {
    let reference = mock_reference_traj("bf16", MOCK_WINDOW);
    let forced = forced_context_upper_bound("fp8", &reference, |t| Ok(mock_cascading(t))).unwrap();
    let s = forced.to_string();
    assert!(
        s.contains("FORCED-CONTEXT UPPER BOUND (NOT a serving metric)"),
        "{s}"
    );
    assert!(
        s.contains("scored near-perfect forced while its free-running agreement collapsed"),
        "the report must carry the forced-vs-free collapse warning in the purged wording \
         (measured figures live in perf/runs.jsonl, not prose): {s}"
    );
    let free =
        compare_free_running(&reference, &mock_cascading_traj("fp8", MOCK_WINDOW)).to_string();
    assert!(free.contains("FREE-RUNNING (serving metric)"), "{free}");
}

#[test]
fn default_flip_gate_denies_a_cascading_arm_even_when_it_is_faster() {
    let reference = mock_reference_traj("bf16", MOCK_WINDOW);
    let cmp = compare_free_running(&reference, &mock_cascading_traj("fp8", MOCK_WINDOW));
    let v = evaluate_default_flip(&DefaultFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        free_running: &cmp,
        reference_quality: None,
        candidate_quality: None,
        speed_delta_pct: Some(-11.6),
    });
    eprintln!("{v}");
    assert!(!v.is_allow());
    let DefaultFlipVerdict::Deny(reasons) = &v else {
        unreachable!()
    };
    assert!(reasons.iter().any(|r| r.contains("free-running agreement")));
    assert!(reasons.iter().any(|r| r.contains("CASCADED")));
    assert!(reasons.iter().any(|r| r.contains("window")));
    assert!(reasons.iter().any(|r| r.contains("perplexity")));
}

#[test]
fn default_flip_gate_allows_a_clean_arm_with_full_evidence() {
    let reference = mock_reference_traj("bf16", 80);
    let candidate = mock_near_tie_traj("fp8", 80);
    let cmp = compare_free_running(&reference, &candidate);
    let rq = TeacherForcedQuality {
        arm: "bf16".into(),
        nll_mean: 0.500,
        perplexity: 0.500f64.exp(),
        reference_rank: vec![1; 80],
        reference_margin: vec![0.0; 80],
    };
    let cq = TeacherForcedQuality {
        arm: "fp8".into(),
        nll_mean: 0.520,
        perplexity: 0.520f64.exp(),
        reference_rank: {
            let mut v = vec![1usize; 80];
            v[3] = 2;
            v
        },
        reference_margin: vec![0.01; 80],
    };
    let v = evaluate_default_flip(&DefaultFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        free_running: &cmp,
        reference_quality: Some(&rq),
        candidate_quality: Some(&cq),
        speed_delta_pct: Some(-11.6),
    });
    eprintln!("{cmp}");
    eprintln!("{rq}");
    eprintln!("{cq}");
    eprintln!("{v}");
    assert!(v.is_allow(), "{v}");
}

#[test]
fn default_flip_gate_denies_on_perplexity_even_when_argmax_agrees() {
    let reference = mock_reference_traj("bf16", 80);
    let candidate = mock_reference_traj("fp8", 80);
    let cmp = compare_free_running(&reference, &candidate);
    assert_eq!(cmp.shape, DivergenceShape::Identical);
    let rq = TeacherForcedQuality {
        arm: "bf16".into(),
        nll_mean: 0.500,
        perplexity: 0.500f64.exp(),
        reference_rank: vec![1; 80],
        reference_margin: vec![0.0; 80],
    };
    let cq = TeacherForcedQuality {
        arm: "fp8".into(),
        nll_mean: 0.900,
        perplexity: 0.900f64.exp(),
        reference_rank: vec![1; 80],
        reference_margin: vec![0.0; 80],
    };
    let v = evaluate_default_flip(&DefaultFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        free_running: &cmp,
        reference_quality: Some(&rq),
        candidate_quality: Some(&cq),
        speed_delta_pct: Some(-11.6),
    });
    eprintln!("{v}");
    assert!(
        !v.is_allow(),
        "identical argmax must not excuse a worse distribution"
    );
}

fn tp(label: &str, kind: PromptKind) -> TemplatedPrompt {
    TemplatedPrompt::from_official_render(label, kind, "m", "s", "d", 1, "r".into(), vec![2, 7])
}

fn healthy_run(arm: &str, label: &str, n: usize) -> FreeRun {
    let mut tokens: Vec<u32> = (0..n as u32).map(|i| 500 + i * 7).collect();
    tokens.push(106);
    FreeRun {
        arm: arm.into(),
        prompt_label: label.into(),
        tokens,
        margins: vec![16.4; n + 1],
        reason: prompts::StopReason::HitStopToken,
        stop_token: Some(106),
        text: "Paris.".into(),
    }
}

fn runaway_run(arm: &str, label: &str, n: usize) -> FreeRun {
    let unit = [1000u32, 1001, 1002];
    FreeRun {
        arm: arm.into(),
        prompt_label: label.into(),
        tokens: unit.iter().cycle().take(n).copied().collect(),
        margins: vec![11.354; n],
        reason: prompts::StopReason::ReachedMaxSteps,
        stop_token: None,
        text: " Paris. The capital of France is Paris. The capital of France is Paris.".into(),
    }
}

fn quality_over(nll: f64, n: usize) -> TeacherForcedQuality {
    TeacherForcedQuality {
        arm: "q".into(),
        nll_mean: nll,
        perplexity: nll.exp(),
        reference_rank: vec![1; n],
        reference_margin: vec![0.0; n],
    }
}

fn clean_quality(label: &str, kind: PromptKind) -> PromptQuality {
    let n = if kind == PromptKind::Control {
        8
    } else {
        MIN_OPEN_ENDED_SCORED_POSITIONS
    };
    PromptQuality {
        label: label.into(),
        kind,
        reference_ended_turn: true,
        reference_loops: false,
        reference: quality_over(0.500, n),
        candidate: quality_over(0.520, n),
    }
}

fn clean_quality_set(labels: &[&str]) -> Vec<PromptQuality> {
    let mut out: Vec<PromptQuality> = labels
        .iter()
        .map(|l| clean_quality(l, PromptKind::Control))
        .collect();
    out.push(clean_quality("openended-explain", PromptKind::OpenEnded));
    out
}

#[test]
fn the_suite_gate_denies_a_flip_whose_reference_never_ended_its_turn() {
    let p = tp("control-capital", PromptKind::Control);
    let stops = StopSet {
        ids: vec![1, 106, 50],
        source: "test".into(),
    };
    let a = runaway_run("bf16", "control-capital", 96);
    let b = runaway_run("fp8", "control-capital", 96);
    let mut table = RunTable::new("degenerate reference");
    table.push(observe(&p, &a, &stops, 96, &|i| format!("<{i}>")));
    table.push(observe(&p, &b, &stops, 96, &|i| format!("<{i}>")));
    let cmp = compare_free_running(
        &trajectory_from_free_run(&p, &a),
        &trajectory_from_free_run(&p, &b),
    );
    assert_eq!(
        cmp.agree, 96,
        "the two loops agree perfectly, which is nearly free"
    );
    assert_eq!(cmp.shape, DivergenceShape::Identical);
    let v = evaluate_default_flip_suite(&SuiteFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        table: &table,
        per_prompt: &[("control-capital".into(), PromptKind::Control, cmp)],
        per_prompt_quality: &clean_quality_set(&["control-capital"]),
        speed_delta_pct: Some(-11.6),
    });
    eprintln!("{table}");
    eprintln!("{v}");
    assert!(
        !v.is_allow(),
        "96/96 agreement between two repetition loops must NOT be enough to flip a default"
    );
    let DefaultFlipVerdict::Deny(reasons) = &v else {
        unreachable!()
    };
    assert!(
        reasons
            .iter()
            .any(|r| r.contains("without ending its turn")),
        "{reasons:?}"
    );
    assert!(
        reasons.iter().any(|r| r.contains("repetition loop")),
        "{reasons:?}"
    );
    assert!(
        reasons.iter().any(|r| r.contains("at least 2")),
        "{reasons:?}"
    );
    let e = format!(
        "{}",
        table.assert_controls_terminated_for("bf16").unwrap_err()
    );
    assert!(e.contains("REFERENCE arm"), "{e}");
}

#[test]
fn the_suite_gate_allows_a_flip_backed_by_two_terminating_controls_and_an_open_ended_quality_row() {
    let stops = StopSet {
        ids: vec![1, 106, 50],
        source: "test".into(),
    };
    let mut table = RunTable::new("clean");
    let mut per_prompt = Vec::new();
    for (label, n) in [("control-arithmetic", 3usize), ("control-capital", 4)] {
        let p = tp(label, PromptKind::Control);
        let a = healthy_run("bf16", label, n);
        let b = healthy_run("fp8", label, n);
        table.push(observe(&p, &a, &stops, 96, &|i| format!("<{i}>")));
        table.push(observe(&p, &b, &stops, 96, &|i| format!("<{i}>")));
        per_prompt.push((
            label.to_string(),
            PromptKind::Control,
            compare_free_running(
                &trajectory_from_free_run(&p, &a),
                &trajectory_from_free_run(&p, &b),
            ),
        ));
    }
    let quality = clean_quality_set(&["control-arithmetic", "control-capital"]);
    let v = evaluate_default_flip_suite(&SuiteFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        table: &table,
        per_prompt: &per_prompt,
        per_prompt_quality: &quality,
        speed_delta_pct: Some(-11.6),
    });
    eprintln!("{table}");
    eprintln!("{v}");
    table.assert_controls_terminated().unwrap();
    table.assert_controls_terminated_for("bf16").unwrap();
    assert!(v.is_allow(), "{v}");
    assert!(
        per_prompt.iter().all(|(_, _, c)| c.total < MIN_FREE_RUNNING_WINDOW),
        "templated controls terminate in far fewer than {MIN_FREE_RUNNING_WINDOW} tokens, which is \
         exactly why the suite gate does not use a token-count window"
    );

    let controls_only: Vec<PromptQuality> = quality
        .iter()
        .filter(|q| q.kind == PromptKind::Control)
        .cloned()
        .collect();
    let stripped = evaluate_default_flip_suite(&SuiteFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        table: &table,
        per_prompt: &per_prompt,
        per_prompt_quality: &controls_only,
        speed_delta_pct: Some(-11.6),
    });
    eprintln!("{stripped}");
    assert!(
        !stripped.is_allow(),
        "the ONLY difference between this evidence and the ALLOW above is the open-ended \
         teacher-forced row. If deleting it still ALLOWs, the gate has silently gone back to \
         scoring controls only. {WHY_OPEN_ENDED_QUALITY}"
    );
    let DefaultFlipVerdict::Deny(rs) = &stripped else {
        unreachable!()
    };
    assert!(
        rs.iter().any(|r| r.contains("0 OPEN-ENDED prompt(s)")),
        "{rs:?}"
    );
}

#[test]
fn the_suite_gate_denies_when_an_open_ended_row_never_ends_its_turn() {
    let stops = StopSet {
        ids: vec![1, 106, 50],
        source: "test".into(),
    };
    let mut table = RunTable::new("open-ended health");
    let mut per_prompt = Vec::new();

    let c = tp("control-capital", PromptKind::Control);
    let ca = healthy_run("bf16", "control-capital", 4);
    let cb = healthy_run("fp8", "control-capital", 4);
    table.push(observe(&c, &ca, &stops, 96, &|i| format!("<{i}>")));
    table.push(observe(&c, &cb, &stops, 96, &|i| format!("<{i}>")));
    per_prompt.push((
        "control-capital".to_string(),
        PromptKind::Control,
        compare_free_running(
            &trajectory_from_free_run(&c, &ca),
            &trajectory_from_free_run(&c, &cb),
        ),
    ));

    let o = tp("openended-explain", PromptKind::OpenEnded);
    let oa = healthy_run("bf16", "openended-explain", 6);
    let mut ob = healthy_run("fp8", "openended-explain", 6);
    ob.tokens.pop();
    ob.reason = prompts::StopReason::ReachedMaxSteps;
    ob.stop_token = None;
    table.push(observe(&o, &oa, &stops, 6, &|i| format!("<{i}>")));
    table.push(observe(&o, &ob, &stops, 6, &|i| format!("<{i}>")));

    let v = evaluate_default_flip_suite(&SuiteFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        table: &table,
        per_prompt: &per_prompt,
        per_prompt_quality: &clean_quality_set(&["control-capital"]),
        speed_delta_pct: Some(-11.6),
    });
    eprintln!("{table}");
    eprintln!("{v}");
    assert!(
        !v.is_allow(),
        "an open-ended arm that never ends its turn must be able to DENY the flip. Before \
         2026-08-08 both loops of this gate filtered to PromptKind::Control, so it was \
         structurally incapable of denying on open-ended output and 16/16 on high-margin \
         controls was close to arithmetically guaranteed."
    );
    let DefaultFlipVerdict::Deny(reasons) = &v else {
        unreachable!()
    };
    assert!(
        reasons
            .iter()
            .any(|r| r.contains("OPEN-ENDED") && r.contains("openended-explain")),
        "{reasons:?}"
    );
}

const HEALTHY_CONTROLS: [&str; 2] = ["control-arithmetic", "control-capital"];

fn healthy_suite() -> (RunTable, Vec<(String, PromptKind, FreeRunningComparison)>) {
    let stops = StopSet {
        ids: vec![1, 106, 50],
        source: "test".into(),
    };
    let mut table = RunTable::new("two healthy controls + one healthy open-ended");
    let mut per_prompt = Vec::new();
    for (label, kind, n) in [
        ("control-arithmetic", PromptKind::Control, 3usize),
        ("control-capital", PromptKind::Control, 4),
        ("openended-explain", PromptKind::OpenEnded, 40),
    ] {
        let p = tp(label, kind);
        let a = healthy_run("bf16", label, n);
        let b = healthy_run("fp8", label, n);
        table.push(observe(&p, &a, &stops, 96, &|i| format!("<{i}>")));
        table.push(observe(&p, &b, &stops, 96, &|i| format!("<{i}>")));
        if kind == PromptKind::Control {
            per_prompt.push((
                label.to_string(),
                kind,
                compare_free_running(
                    &trajectory_from_free_run(&p, &a),
                    &trajectory_from_free_run(&p, &b),
                ),
            ));
        }
    }
    (table, per_prompt)
}

#[test]
fn the_suite_gate_denies_when_only_the_open_ended_row_loses_teacher_forced_quality() {
    let (table, per_prompt) = healthy_suite();
    let clean = clean_quality_set(&HEALTHY_CONTROLS);
    let base = evaluate_default_flip_suite(&SuiteFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        table: &table,
        per_prompt: &per_prompt,
        per_prompt_quality: &clean,
        speed_delta_pct: Some(-11.6),
    });
    eprintln!("{base}");
    assert!(
        base.is_allow(),
        "the control arm of this fixture must be clean, otherwise the DENY below proves nothing \
         about the open-ended row: {base}"
    );

    let mut damaged = clean.clone();
    let oe = damaged
        .iter_mut()
        .find(|q| q.kind == PromptKind::OpenEnded)
        .unwrap();
    oe.candidate = quality_over(0.500 + 0.30, MIN_OPEN_ENDED_SCORED_POSITIONS);
    let ratio = oe.perplexity_ratio();
    assert!(
        ratio > MAX_PERPLEXITY_RATIO,
        "fixture must actually exceed the bar, measured ratio {ratio}"
    );
    let v = evaluate_default_flip_suite(&SuiteFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        table: &table,
        per_prompt: &per_prompt,
        per_prompt_quality: &damaged,
        speed_delta_pct: Some(-11.6),
    });
    eprintln!("{v}");
    assert!(
        !v.is_allow(),
        "the ONLY defect in this evidence is on an OPEN-ENDED teacher-forced row. Until 2026-08-09 \
         the teacher-forced half of this gate was a single aggregate pair picked by MAXIMUM \
         ABSOLUTE candidate perplexity over CONTROL prompts, so this regression was unreachable. \
         {WHY_OPEN_ENDED_QUALITY}"
    );
    let DefaultFlipVerdict::Deny(rs) = &v else {
        unreachable!()
    };
    assert!(
        rs.iter().any(|r| r.contains("OPEN-ENDED")
            && r.contains("openended-explain")
            && r.contains("perplexity ratio")),
        "{rs:?}"
    );

    let controls_only: Vec<PromptQuality> = damaged
        .iter()
        .filter(|q| q.kind == PromptKind::Control)
        .cloned()
        .collect();
    assert!(
        !teacher_forced_reasons(&controls_only)
            .iter()
            .any(|r| r.contains("perplexity ratio")),
        "this is the falsification: restore a PromptKind::Control filter over the quality \
         evidence and the exact same regression produces no perplexity reason at all"
    );
}

#[test]
fn the_suite_gate_denies_when_a_run_table_row_was_never_teacher_forced() {
    let (table, per_prompt) = healthy_suite();
    let quality: Vec<PromptQuality> = vec![
        clean_quality("control-arithmetic", PromptKind::Control),
        clean_quality("control-capital", PromptKind::Control),
        clean_quality("some-other-open-ended", PromptKind::OpenEnded),
    ];
    let v = evaluate_default_flip_suite(&SuiteFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        table: &table,
        per_prompt: &per_prompt,
        per_prompt_quality: &quality,
        speed_delta_pct: Some(-11.6),
    });
    eprintln!("{v}");
    assert!(!v.is_allow(), "{v}");
    let DefaultFlipVerdict::Deny(rs) = &v else {
        unreachable!()
    };
    assert!(
        rs.iter()
            .any(|r| r.contains("openended-explain") && r.contains("never teacher-forced")),
        "{rs:?}"
    );
}

#[test]
fn the_suite_gate_denies_an_open_ended_quality_entry_scored_on_too_few_positions() {
    let (table, per_prompt) = healthy_suite();
    let mut quality = clean_quality_set(&HEALTHY_CONTROLS);
    let oe = quality
        .iter_mut()
        .find(|q| q.kind == PromptKind::OpenEnded)
        .unwrap();
    oe.reference = quality_over(0.500, 4);
    oe.candidate = quality_over(0.520, 4);
    let v = evaluate_default_flip_suite(&SuiteFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        table: &table,
        per_prompt: &per_prompt,
        per_prompt_quality: &quality,
        speed_delta_pct: Some(-11.6),
    });
    eprintln!("{v}");
    assert!(
        !v.is_allow(),
        "a 4-position open-ended entry would satisfy the open-ended coverage requirement while \
         measuring almost nothing: {v}"
    );
    let DefaultFlipVerdict::Deny(rs) = &v else {
        unreachable!()
    };
    assert!(
        rs.iter().any(|r| r.contains("only 4 position(s)")),
        "{rs:?}"
    );
}

#[test]
fn the_suite_gate_denies_when_the_teacher_forced_reference_is_a_repetition_loop() {
    let (table, per_prompt) = healthy_suite();
    let mut quality = clean_quality_set(&HEALTHY_CONTROLS);
    let oe = quality
        .iter_mut()
        .find(|q| q.kind == PromptKind::OpenEnded)
        .unwrap();
    oe.reference_loops = true;
    oe.reference_ended_turn = false;
    eprintln!("{oe}");
    let v = evaluate_default_flip_suite(&SuiteFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        table: &table,
        per_prompt: &per_prompt,
        per_prompt_quality: &quality,
        speed_delta_pct: Some(-11.6),
    });
    eprintln!("{v}");
    assert!(
        !v.is_allow(),
        "teacher-forcing a candidate onto a reference that is itself looping reproduces the \
         2026-08-06 methodology failure with extra steps: {v}"
    );
    let DefaultFlipVerdict::Deny(rs) = &v else {
        unreachable!()
    };
    assert!(
        rs.iter()
            .any(|r| r.contains("openended-explain") && r.contains("repetition loop")),
        "{rs:?}"
    );
}

#[test]
fn the_suite_gate_denies_when_the_open_ended_reference_hit_max_steps_without_repeating() {
    let (table, per_prompt) = healthy_suite();
    let clean = clean_quality_set(&HEALTHY_CONTROLS);
    let base = evaluate_default_flip_suite(&SuiteFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        table: &table,
        per_prompt: &per_prompt,
        per_prompt_quality: &clean,
        speed_delta_pct: Some(-11.6),
    });
    assert!(base.is_allow(), "fixture must start clean: {base}");

    let mut quality = clean;
    let oe = quality
        .iter_mut()
        .find(|q| q.kind == PromptKind::OpenEnded)
        .unwrap();
    oe.reference_ended_turn = false;
    oe.reference_loops = false;
    let v = evaluate_default_flip_suite(&SuiteFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        table: &table,
        per_prompt: &per_prompt,
        per_prompt_quality: &quality,
        speed_delta_pct: Some(-11.6),
    });
    eprintln!("{v}");
    assert!(
        !v.is_allow(),
        "the ONLY defect here is an OPEN-ENDED reference that hit max-steps and did NOT repeat on \
         a detectable period, so repeated_tail cannot catch it and the run table never saw the \
         row. {WHY_REFERENCE_MUST_TERMINATE}"
    );
    let DefaultFlipVerdict::Deny(rs) = &v else {
        unreachable!()
    };
    assert!(
        rs.iter().any(|r| r.contains("OPEN-ENDED")
            && r.contains("openended-explain")
            && r.contains("hit max-steps")),
        "{rs:?}"
    );

    let table_only: Vec<String> = table
        .rows
        .iter()
        .filter(|r| r.kind == PromptKind::OpenEnded && !r.ended_turn)
        .map(|r| r.prompt_label.clone())
        .collect();
    assert!(
        table_only.is_empty(),
        "this is the falsification: the run-table half of the gate sees nothing wrong with this \
         evidence ({table_only:?}), which is precisely the state the real-weight gate was in - it \
         pushed only its CONTROL rows into the table, so !ended_turn on an open-ended row was \
         checked by nobody. {WHY_REFERENCE_MUST_TERMINATE}"
    );
    let controls_only: Vec<PromptQuality> = quality
        .iter()
        .filter(|q| q.kind == PromptKind::Control)
        .cloned()
        .collect();
    assert!(
        !teacher_forced_reasons(&controls_only)
            .iter()
            .any(|r| r.contains("hit max-steps")),
        "restore a PromptKind::Control filter over the quality evidence and the same runaway \
         reference produces no reason at all"
    );
}

#[test]
fn the_suite_gate_denies_when_no_teacher_forced_evidence_is_supplied_at_all() {
    let (table, per_prompt) = healthy_suite();
    let v = evaluate_default_flip_suite(&SuiteFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        table: &table,
        per_prompt: &per_prompt,
        per_prompt_quality: &[],
        speed_delta_pct: Some(-11.6),
    });
    assert!(!v.is_allow(), "{v}");
    let DefaultFlipVerdict::Deny(rs) = &v else {
        unreachable!()
    };
    assert!(
        rs.iter()
            .any(|r| r.contains("no teacher-forced perplexity / rank evidence supplied")),
        "{rs:?}"
    );
}

#[test]
fn the_suite_gate_denies_when_a_control_diverges_even_if_both_arms_terminate() {
    let stops = StopSet {
        ids: vec![1, 106, 50],
        source: "test".into(),
    };
    let mut table = RunTable::new("control flip");
    let mut per_prompt = Vec::new();
    for (label, flip) in [("control-arithmetic", false), ("control-capital", true)] {
        let p = tp(label, PromptKind::Control);
        let a = healthy_run("bf16", label, 4);
        let mut b = healthy_run("fp8", label, 4);
        if flip {
            b.tokens[1] = 4242;
        }
        table.push(observe(&p, &a, &stops, 96, &|i| format!("<{i}>")));
        table.push(observe(&p, &b, &stops, 96, &|i| format!("<{i}>")));
        per_prompt.push((
            label.to_string(),
            PromptKind::Control,
            compare_free_running(
                &trajectory_from_free_run(&p, &a),
                &trajectory_from_free_run(&p, &b),
            ),
        ));
    }
    let v = evaluate_default_flip_suite(&SuiteFlipEvidence {
        change: "fp8 attention projections",
        backend: "mock",
        model: "mock",
        table: &table,
        per_prompt: &per_prompt,
        per_prompt_quality: &clean_quality_set(&["control-arithmetic", "control-capital"]),
        speed_delta_pct: Some(-11.6),
    });
    eprintln!("{v}");
    assert!(!v.is_allow());
    let DefaultFlipVerdict::Deny(reasons) = &v else {
        unreachable!()
    };
    assert!(
        reasons
            .iter()
            .any(|r| r.contains("CONTROL prompt \"control-capital\" diverged")),
        "{reasons:?}"
    );
}

#[test]
fn the_kept_logits_align_with_the_emitted_tokens_at_prompt_len_minus_one() {
    let vocab = 64usize;
    let stops = StopSet {
        ids: vec![39],
        source: "test".into(),
    };
    for (label, max_steps, expect_stop) in [("short", 32usize, true), ("runaway", 4usize, false)] {
        let p = TemplatedPrompt::from_official_render(
            label,
            PromptKind::Control,
            "m",
            "s",
            "d",
            1,
            "r".into(),
            vec![2, 11, 12, 13, 14],
        );
        let mut kept: Vec<Vec<f32>> = Vec::new();
        let r = prompts::free_running("arm", &p, &stops, max_steps, |t| {
            let mut v = vec![0.0f32; vocab];
            v[((t as usize + 1) % 40) + 4] = 5.0;
            kept.push(v.clone());
            Ok(v)
        })
        .unwrap();
        let at_gen = p.ids.len() - 1;
        eprintln!(
            "{label}: prompt {} ids, {} tokens, {} kept logit vectors, gen slice starts at {at_gen}",
            p.ids.len(),
            r.tokens.len(),
            kept.len()
        );
        assert_eq!(r.reason == prompts::StopReason::HitStopToken, expect_stop);
        assert!(
            kept.len() >= at_gen + r.tokens.len(),
            "{label}: not enough kept logits"
        );
        for (j, t) in r.tokens.iter().enumerate() {
            assert_eq!(
                nth_best_token(&kept[at_gen + j], 0),
                *t,
                "{label}: kept[{}] must be the distribution that emitted tokens[{j}]",
                at_gen + j
            );
        }
        let naive = kept.len() - r.tokens.len();
        assert_eq!(
            naive,
            if expect_stop { at_gen } else { at_gen + 1 },
            "{label}: kept.len() - tokens.len() is off by one exactly when the arm never \
             terminates, which is precisely the degenerate case; always slice at ids.len() - 1"
        );
    }
}

#[test]
fn teacher_forced_quality_reads_rank_and_perplexity_off_logits() {
    let reference = FreeRunningTrajectory {
        arm: "ref".into(),
        prompt: vec![0],
        tokens: vec![1, 2, 3],
    };
    let mut calls = 0usize;
    let q = teacher_forced_quality("cand", &reference, |_t| {
        calls += 1;
        let mut l = vec![0.0f32; 8];
        l[1] = 4.0;
        l[2] = 3.0;
        l[3] = 2.0;
        Ok((1u32, l))
    })
    .unwrap();
    eprintln!("{q}");
    assert_eq!(calls, 3);
    assert_eq!(q.reference_rank, vec![1, 2, 3]);
    assert!((q.reference_margin[0] - 0.0).abs() < 1e-6);
    assert!((q.reference_margin[1] - 1.0).abs() < 1e-6);
    assert!((q.reference_margin[2] - 2.0).abs() < 1e-6);
    assert!(q.perplexity > 1.0 && q.perplexity < 8.0, "{}", q.perplexity);
}

pub const FLIP_EVIDENCE: &str = "fp8 attention projections are ON and hardcoded (the \
     NV_WGPU_ATTN_FP8 env knob no longer exists -- tests select arms via \
     gemma4_wgpu::set_attn_variant). The flip evidence class, all on the Gemma4-31B-IT-NVFP4 \
     snapshot through the shipped chat_template.jinja, EOS-aware, per-prompt, never pooled:\n\
     (1) templated CONTROL prompts derived and proved against the emitted PromptPack's own \
     render affixes, with full free-running argmax agreement, no first divergence, identical \
     shape, same termination, every row ended-turn, zero repetition loops, zero stop-token \
     leaks, zero extra control-token occurrences.\n\
     (2) Teacher-forced quality on the bf16 trajectory: worst-prompt candidate perplexity \
     ratio far inside the allowed bound; reference-token rank pinned at 1 on every prompt.\n\
     (3) evaluate_default_flip_suite returned ALLOW.\n\
     (4) Independent runs of the controls produce byte-identical tables, so unlike the \
     low-margin open-ended prompts these controls ARE reproducible on this box.\n\
     (5) The epilogues that actually ship, g4w_gemv_fp8_pk and g4w_gemv_fp8_pk3, are covered \
     per element at the true Gemma4 shapes by wgpu_fp8_epilogue.rs against an f64 CPU \
     reference, with deliberate corruptions of the shipped shader caught orders of magnitude \
     outside the pass bound.\n\
     (6) The flip cuts weight-bytes/token with dispatch count unchanged.\n\
     Current numbers for every clause: perf/runs.jsonl.\n\
     NOT measured by this lane: a fresh speed number (the delta passed to the gate is a recorded \
     value, not a measurement from this run) and any open-ended prompt, which the harness excludes \
     from A/B evidence because it is not reproducible here. To revert, set ATTN_FP8_DEFAULT_ON to \
     false in gemma4_wgpu.rs. See docs/book/04.2-fp8-epilogue-mechanism.md.";

pub const REFERENCE_QUALITY_DOC: &str = "REFERENCE TEACHER-FORCED QUALITY. Before 2026-08-07 this \
     harness passed reference_quality: None, so evaluate_default_flip_suite could NEVER return \
     ALLOW -- it always appended \"no teacher-forced perplexity / rank evidence supplied\" no \
     matter how clean the arms were. The reference arm's own NLL on its own greedy trajectory is \
     recomputed here from the bf16 logits already captured during the free run (no extra model \
     build, no extra GPU work), so the perplexity RATIO the gate checks is finally a real number. \
     By construction the reference ranks are all 1 and its margins are all 0: the reference is \
     greedy on itself.";

pub const WIDE_PACK_DOC: &str = "WIDENED PROMPT SET (job 2, 2026-08-07). The recorded fp8 \
     verdict rested on 5-6 prompts. This harness widens the pack in place instead of hand-writing \
     prompts: it derives the shipped chat template's single-user-turn affix pair from the renders \
     ALREADY in the pack (longest common prefix / longest common suffix over every single-turn \
     prompt, with the suffix trimmed forward to the first markup character), proves the derivation \
     by re-encoding every source render and requiring the ids to match the pack byte-for-byte, and \
     only then emits additional prompts as prefix+text+suffix through the same tokenizer. No \
     prompt here is hand-wrapped: the affixes come out of renders produced by the serving \
     ChatTemplate in rust/tests/chat_eval.rs. Set NV_FP8_WIDE_PACK=0 to measure the narrow pack.";

pub const WIDE_PACK_LIMIT: &str = "LIMIT: the derived prompts share ONE turn shape (a single user \
     message, no system turn, no tools). They widen the sample of CONTENT, not of template shape. \
     A template-shape regression would still need a re-emitted pack from chat_eval.rs.";

pub const EXTRA_CONTROL_PROMPTS: [(&str, &str); 13] = [
    (
        "control-mult",
        "What is 7 times 6? Reply with the number only.",
    ),
    (
        "control-capital-jp",
        "What is the capital of Japan? Reply with the city name only.",
    ),
    (
        "control-capital-it",
        "What is the capital of Italy? Reply with the city name only.",
    ),
    (
        "control-sub",
        "What is 100 minus 37? Reply with the number only.",
    ),
    (
        "control-div",
        "What is 12 divided by 4? Reply with the number only.",
    ),
    (
        "control-planet",
        "Which planet is closest to the Sun? Reply with the planet name only.",
    ),
    (
        "control-week",
        "How many days are in a week? Reply with the number only.",
    ),
    (
        "control-triangle",
        "How many sides does a triangle have? Reply with the number only.",
    ),
    (
        "control-literal-orange",
        "Reply with exactly the word ORANGE and nothing else.",
    ),
    (
        "control-literal-true",
        "Reply with exactly the word TRUE and nothing else.",
    ),
    (
        "control-literal-z",
        "Reply with exactly the letter Z and nothing else.",
    ),
    (
        "control-water",
        "What is the chemical symbol for water? Reply with the formula only.",
    ),
    (
        "control-boiling",
        "At what temperature in Celsius does water boil at sea level? Reply with the number only.",
    ),
];

#[derive(Clone, Debug)]
pub struct RenderAffix {
    pub prefix: String,
    pub suffix: String,
    pub sources: Vec<String>,
}

impl RenderAffix {

    pub fn render(&self, body: &str) -> String {
        format!("{}{}{}", self.prefix, body, self.suffix)
    }
}

fn common_prefix(strs: &[&str]) -> String {
    let first = strs[0];
    let mut n = first.len();
    for s in &strs[1..] {
        let m = first
            .bytes()
            .zip(s.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        n = n.min(m);
    }
    while n > 0 && !first.is_char_boundary(n) {
        n -= 1;
    }
    first[..n].to_string()
}

fn common_suffix(strs: &[&str]) -> String {
    let first = strs[0];
    let mut n = first.len();
    for s in &strs[1..] {
        let m = first
            .bytes()
            .rev()
            .zip(s.bytes().rev())
            .take_while(|(a, b)| a == b)
            .count();
        n = n.min(m);
    }
    let mut start = first.len() - n;
    while start < first.len() && !first.is_char_boundary(start) {
        start += 1;
    }
    first[start..].to_string()
}

pub fn derive_single_turn_affix(pack: &PromptPack) -> anyhow::Result<RenderAffix> {
    let tail = {
        let all: Vec<&str> = pack.prompts.iter().map(|p| p.rendered.as_str()).collect();
        anyhow::ensure!(
            all.len() >= 3,
            "need at least 3 renders to derive an affix pair"
        );
        common_suffix(&all)
    };
    let cut = tail
        .find('<')
        .ok_or_else(|| anyhow::anyhow!("common render suffix {tail:?} carries no markup token"))?;
    let suffix = tail[cut..].to_string();
    anyhow::ensure!(
        suffix.len() >= 8,
        "derived render suffix {suffix:?} is too short to be the template's generation prompt"
    );

    let marker: String = {
        let rest = &suffix[1..];
        let end = rest.find('<').map(|i| i + 1).unwrap_or(suffix.len());
        suffix[..end].trim_end().to_string()
    };
    let turns = |p: &TemplatedPrompt| p.rendered.matches(&marker).count();
    let fewest = pack.prompts.iter().map(turns).min().unwrap_or(0);
    let singles: Vec<&TemplatedPrompt> =
        pack.prompts.iter().filter(|p| turns(p) == fewest).collect();
    anyhow::ensure!(
        singles.len() >= 3,
        "only {} of {} pack renders share the minimal turn shape ({marker:?} x {fewest}); an \
         affix derived from fewer than three renders is not cross-validated",
        singles.len(),
        pack.prompts.len()
    );
    let heads: Vec<&str> = singles.iter().map(|p| p.rendered.as_str()).collect();
    let prefix = common_prefix(&heads);
    anyhow::ensure!(
        prefix.ends_with('\n') && prefix.contains('<'),
        "derived render prefix {prefix:?} does not look like a template turn header"
    );
    for p in &pack.prompts {
        anyhow::ensure!(
            p.rendered.ends_with(&suffix),
            "prompt {} does not end with the derived suffix {suffix:?}",
            p.label
        );
    }
    Ok(RenderAffix {
        prefix,
        suffix,
        sources: singles.iter().map(|p| p.label.clone()).collect(),
    })
}

pub fn widen_pack(
    pack: &PromptPack,
    tok: &tokenizers::Tokenizer,
) -> anyhow::Result<(PromptPack, String)> {
    let a = derive_single_turn_affix(pack)?;
    let mut proof = Vec::new();
    for p in &pack.prompts {
        let enc = tok
            .encode(p.rendered.as_str(), false)
            .map_err(|e| anyhow::anyhow!("re-encode {}: {e}", p.label))?;
        anyhow::ensure!(
            enc.get_ids() == p.ids.as_slice(),
            "re-encoding the pack render of {} did not reproduce its stored ids; this harness \
             must not widen a pack it cannot reproduce",
            p.label
        );
        proof.push(p.label.clone());
    }
    let mut out = pack.clone();
    let mut added = Vec::new();
    for (label, text) in EXTRA_CONTROL_PROMPTS {
        anyhow::ensure!(
            !text.contains('<') && text.is_ascii(),
            "derived prompt {label} must be plain ascii with no markup"
        );
        let derived = a.render(text);
        let enc = tok
            .encode(derived.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode {label}: {e}"))?;
        out.prompts.push(TemplatedPrompt::from_official_render(
            label,
            PromptKind::Control,
            &pack.model_repo,
            &pack.snapshot,
            &pack.template_digest,
            pack.template_bytes,
            derived,
            enc.get_ids().to_vec(),
        ));
        added.push(label);
    }
    let note = format!(
        "{WIDE_PACK_DOC}\naffix derived from {} render(s): prefix {:?} suffix {:?}\nre-encode \
         proof: {} of {} pack renders reproduced their stored ids exactly\nadded {} control \
         prompts: {}\npack now {} prompts ({} control)\n{WIDE_PACK_LIMIT}",
        a.sources.len(),
        a.prefix,
        a.suffix,
        proof.len(),
        pack.prompts.len(),
        added.len(),
        added.join(", "),
        out.prompts.len(),
        out.controls(),
    );
    Ok((out, note))
}

pub fn wide_pack_enabled() -> bool {
    std::env::var("NV_FP8_WIDE_PACK").ok().as_deref() != Some("0")
}

#[test]
fn the_affix_deriver_recovers_the_gemma4_turn_markers_from_renders_alone() {
    let mk = |label: &str, body: &str| {
        TemplatedPrompt::from_official_render(
            label,
            PromptKind::Control,
            "repo",
            "snap",
            "digest",
            7,
            format!("<bos><|turn>user\n{body}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
            vec![1, 2, 3],
        )
    };
    let pack = PromptPack {
        model_repo: "repo".into(),
        snapshot: "snap".into(),
        template_digest: "digest".into(),
        template_bytes: 7,
        stop_ids: vec![1, 106, 50],
        stop_source: "test".into(),
        prompts: vec![
            mk("a", "What is 2 + 2? Reply with the number only."),
            mk("b", "What is the capital of France? Reply with the city name only."),
            mk("c", "Reply with exactly the word BANANA and nothing else."),
            TemplatedPrompt::from_official_render(
                "sys",
                PromptKind::OpenEnded,
                "repo",
                "snap",
                "digest",
                7,
                "<bos><|turn>system\nYou are terse.<turn|>\n<|turn>user\nHi.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>".into(),
                vec![4],
            ),
        ],
    };
    let a = derive_single_turn_affix(&pack).unwrap();
    assert_eq!(a.prefix, "<bos><|turn>user\n");
    assert_eq!(
        a.suffix,
        "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    );
    assert_eq!(
        format!("{}{}{}", a.prefix, "Reply with exactly the word ORANGE and nothing else.", a.suffix),
        "<bos><|turn>user\nReply with exactly the word ORANGE and nothing else.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    );
    eprintln!("{WIDE_PACK_DOC}");
}

#[test]
fn the_affix_deriver_refuses_a_pack_it_cannot_read() {
    let pack = PromptPack {
        model_repo: "repo".into(),
        snapshot: "snap".into(),
        template_digest: "digest".into(),
        template_bytes: 7,
        stop_ids: vec![1],
        stop_source: "test".into(),
        prompts: vec![],
    };
    assert!(derive_single_turn_affix(&pack).is_err());
}

#[cfg(feature = "wgpu")]
mod wgpu_arms {
    use super::prompts::{describe_selection, SuiteReport, MARGIN_TRAP, REPRODUCIBILITY_LIMIT};
    use super::*;
    use nv_models::gemma4::{Gemma4Config, LayerType};
    use nv_models::gemma4_wgpu::{
        set_attn_variant, AttnQuant, AttnVariant, Gemma4Wgpu, HostBf16Lin, HostLayer, HostProj,
        HostWeights, ATTN_VARIANT_DEFAULT,
    };

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    const TINY_CONFIG: &str = r#"{
      "text_config": {
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "head_dim": 64,
        "global_head_dim": 128,
        "vocab_size": 512,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "sliding_window": 8,
        "final_logit_softcapping": 30.0,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "full_attention"],
        "attention_k_eq_v": true,
        "hidden_activation": "gelu_pytorch_tanh",
        "num_kv_shared_layers": 0,
        "rope_parameters": {
          "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
          "sliding_attention": {"rope_theta": 10000.0}
        }
      },
      "tie_word_embeddings": true
    }"#;

    fn ctx_or_skip() -> bool {
        match nv_kernels::wgpu_backend::WgpuContext::shared() {
            Ok(ctx) => {
                eprintln!("adapter: {}", ctx.summary());
                true
            }
            Err(e) => {
                eprintln!("skipping: no wgpu adapter ({e})");
                false
            }
        }
    }

    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (self.0 >> 33) as u32;
            (bits as f32 / u32::MAX as f32 - 0.5) * 0.2
        }
        fn bf16_vec(&mut self, n: usize) -> Vec<u16> {
            (0..n)
                .map(|_| half::bf16::from_f32(self.next_f32()).to_bits())
                .collect()
        }
        fn bf16_vec_around_one(&mut self, n: usize) -> Vec<u16> {
            (0..n)
                .map(|_| half::bf16::from_f32(1.0 + self.next_f32()).to_bits())
                .collect()
        }
    }

    fn tiny_host_weights(config: &Gemma4Config, seed: u64) -> HostWeights {
        let mut rng = Lcg(seed);
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let n_q = config.num_attention_heads;
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let kind = config.layer_kind(i);
            let hd = config.head_dim_for(kind);
            let nkv = config.num_kv_heads_for(kind);
            let q_dim = n_q * hd;
            let kv_dim = nkv * hd;
            let has_v = !matches!(
                (kind, config.attention_k_eq_v),
                (LayerType::FullAttention, true)
            );
            let qkv_rows = q_dim + kv_dim * if has_v { 2 } else { 1 };
            let mk = |rng: &mut Lcg, n: usize, k: usize| {
                HostProj::Bf16(HostBf16Lin {
                    w: rng.bf16_vec(n * k),
                    n,
                    k,
                })
            };
            layers.push(HostLayer {
                kind,
                input_ln: rng.bf16_vec_around_one(hidden),
                post_attn_ln: rng.bf16_vec_around_one(hidden),
                pre_ff_ln: rng.bf16_vec_around_one(hidden),
                post_ff_ln: rng.bf16_vec_around_one(hidden),
                q_norm: rng.bf16_vec_around_one(hd),
                k_norm: rng.bf16_vec_around_one(hd),
                layer_scalar: 0.9,
                has_v,
                qkv: mk(&mut rng, qkv_rows, hidden),
                o: mk(&mut rng, hidden, q_dim),
                gate_up: mk(&mut rng, 2 * inter, hidden),
                down: mk(&mut rng, hidden, inter),
            });
        }
        HostWeights {
            embed: rng.bf16_vec(config.vocab_size * hidden),
            final_norm: rng.bf16_vec_around_one(hidden),
            layers,
        }
    }

    fn tiny_host_weights_spiky(
        config: &Gemma4Config,
        seed: u64,
        spike: f32,
        embed_gain: f32,
    ) -> HostWeights {
        let mut h = tiny_host_weights(config, seed);
        for e in h.embed.iter_mut() {
            *e = half::bf16::from_f32(half::bf16::from_bits(*e).to_f32() * embed_gain).to_bits();
        }
        for layer in h.layers.iter_mut() {
            for proj in [&mut layer.qkv, &mut layer.o] {
                if let HostProj::Bf16(l) = proj {
                    let k = l.k;
                    for r in 0..l.n {
                        let c = (r * 37 + 11) % k;
                        l.w[r * k + c] = half::bf16::from_f32(spike).to_bits();
                    }
                }
            }
        }
        h
    }

    fn build(config: &Gemma4Config, host: &HostWeights, fp8: bool, max_seq: usize) -> Gemma4Wgpu {
        build_variant(
            config,
            host,
            AttnVariant {
                on: fp8,
                ..ATTN_VARIANT_DEFAULT
            },
            max_seq,
        )
    }

    fn build_variant(
        config: &Gemma4Config,
        host: &HostWeights,
        v: AttnVariant,
        max_seq: usize,
    ) -> Gemma4Wgpu {
        set_attn_variant(Some(v));
        let m = Gemma4Wgpu::new(config.clone(), host, max_seq);
        set_attn_variant(None);
        m.unwrap()
    }

    #[test]
    fn harness_binds_to_gemma4_wgpu_and_separates_the_two_metrics() {
        let _g = env_lock();
        if !ctx_or_skip() {
            return;
        }
        let config = Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap();
        let host = tiny_host_weights(&config, 0xc0ffee);
        let prompt: Vec<u32> = vec![2, 11, 47, 300, 5];
        let steps = 24usize;

        let mut a = build(&config, &host, false, 256);
        let reference =
            free_running_greedy("tiny bf16-attn", &prompt, steps, |t| a.decode_step(t)).unwrap();
        drop(a);

        let mut b = build(&config, &host, true, 256);
        let candidate =
            free_running_greedy("tiny fp8-attn", &prompt, steps, |t| b.decode_step(t)).unwrap();
        drop(b);

        let cmp = compare_free_running(&reference, &candidate);
        eprintln!("{cmp}");
        eprintln!("reference tokens: {:?}", reference.tokens);
        eprintln!("candidate tokens: {:?}", candidate.tokens);

        let mut c = build(&config, &host, true, 256);
        let forced =
            forced_context_upper_bound("tiny fp8-attn", &reference, |t| c.decode_step(t)).unwrap();
        drop(c);
        eprintln!("{forced}");

        let mut d = build(&config, &host, true, 256);
        let q = teacher_forced_quality("tiny fp8-attn", &reference, |t| d.decode_step_logits(t))
            .unwrap();
        drop(d);
        eprintln!("{q}");

        assert_eq!(cmp.total, steps);
        let (fa, ft) = forced.upper_bound_agreement_is_not_serving_evidence();
        assert_eq!(ft, steps);
        assert!(
            fa >= cmp.agree,
            "forced context is an upper bound on free-running: forced {fa}/{ft}, free {}/{}",
            cmp.agree,
            cmp.total
        );
        assert_eq!(q.reference_rank.len(), steps);
        assert!(q.perplexity.is_finite());
        eprintln!(
            "NOTE: random tiny weights carry no quality signal. This test proves the harness \
             binds to Gemma4Wgpu and that forced >= free-running; it is NOT evidence about fp8."
        );
    }

    const MIN_SABOTAGE_SEPARATION: f64 = 4.0;

    struct WrongQuantizer {
        label: &'static str,
        variant: AttnVariant,
    }

    const SABOTAGES: [WrongQuantizer; 2] = [
        WrongQuantizer {
            label: "int8 with ONE scale per row (no group scales at all)",
            variant: AttnVariant {
                on: true,
                quant: AttnQuant {
                    fmt: nv_kernels::wgpu_backend::kernels::quant_gemv::QFormat::Int8,
                    group: 0,
                    lo: 0,
                    hi: usize::MAX,
                },
                legacy_epilogue: 0,
            },
        },
        WrongQuantizer {
            label: "e4m3 group scales on the weights, ONE row scale multiplied in by the legacy \
                    epilogue - the scale contract violated, not just coarsened",
            variant: AttnVariant {
                on: true,
                quant: AttnQuant {
                    fmt: nv_kernels::wgpu_backend::kernels::quant_gemv::QFormat::E4m3,
                    group: 128,
                    lo: 0,
                    hi: usize::MAX,
                },
                legacy_epilogue: 1,
            },
        },
    ];

    fn attn_arm_identity() -> (String, usize, u32) {
        let q = nv_models::gemma4_wgpu::attn_quant_config();
        (
            q.fmt.label().to_string(),
            q.group,
            nv_models::gemma4_wgpu::attn_fp8_legacy_epilogue(),
        )
    }

    #[test]
    fn a_knowingly_wrong_quantizer_trips_the_open_ended_teacher_forced_check() {
        let _g = env_lock();
        if !ctx_or_skip() {
            return;
        }
        let config = Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap();
        let spike: f32 = std::env::var("NV_FP8_GATE_SPIKE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128.0);
        let embed_gain: f32 = std::env::var("NV_FP8_GATE_EMBED_GAIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        let host = tiny_host_weights_spiky(&config, 0xc0ffee, spike, embed_gain);
        let prompt: Vec<u32> = vec![2, 11, 47, 300, 5];
        let steps = MIN_OPEN_ENDED_SCORED_POSITIONS + 8;
        eprintln!(
            "tiny attention projections carry one {spike} outlier per row against a ~0.1 bulk. \
             That is the shape scale granularity is worst at: one scale per ROW set by the \
             outlier leaves the bulk two or three levels wide, while group scales confine the \
             damage to the group that holds the outlier. Override the magnitude with \
             NV_FP8_GATE_SPIKE."
        );

        let shipped_id = attn_arm_identity();
        eprintln!(
            "SHIPPED attention quantizer, read from nv_models::gemma4_wgpu rather than hard-coded \
             here: fmt {} group {} legacy-epilogue {}. It is read because hard-coding it is how \
             this test rotted: eecb69c3b (2026-08-09) flipped ATTN_FP8_FMT_DEFAULT e4m3 -> int8, \
             the honest arm silently became int8/128 while still being PRINTED as \"e4m3/row\", \
             and the sabotage (int8 per-row) stopped being a DIFFERENT quantizer and became a \
             NEIGHBOURING one. Measured separation collapsed from the required \
             {MIN_SABOTAGE_SEPARATION:.0}x to 1.9x and the test went red. Every sabotage below is \
             therefore asserted to differ from whatever ships today.",
            shipped_id.0, shipped_id.1, shipped_id.2
        );

        let mut a = build(&config, &host, false, 256);
        let reference =
            free_running_greedy("tiny bf16-attn", &prompt, steps, |t| a.decode_step(t)).unwrap();
        a.reset();
        let rq = teacher_forced_quality("tiny bf16-attn", &reference, |t| a.decode_step_logits(t))
            .unwrap();
        drop(a);

        let mut b = build(&config, &host, true, 256);
        let honest =
            teacher_forced_quality("tiny shipped-attn", &reference, |t| b.decode_step_logits(t))
                .unwrap();
        drop(b);

        let loops = prompts::repeated_tail(&reference.tokens).is_some();
        let mk = |cand: &TeacherForcedQuality| PromptQuality {
            label: "tiny-open-ended".into(),
            kind: PromptKind::OpenEnded,
            reference_ended_turn: false,
            reference_loops: loops,
            reference: rq.clone(),
            candidate: cand.clone(),
        };
        let honest_q = mk(&honest);

        let mut wrong: Vec<(&'static str, PromptQuality)> = Vec::new();
        for s in SABOTAGES.iter() {
            set_attn_variant(Some(s.variant));
            let id = attn_arm_identity();
            set_attn_variant(None);
            assert_ne!(
                id, shipped_id,
                "sabotage {:?} resolves to the SHIPPED configuration {shipped_id:?}, so it cannot \
                 falsify anything. A default was re-blessed onto this test's control arm; pick a \
                 sabotage that is still wrong under the new default rather than deleting the \
                 assertion.",
                s.label
            );
            let mut c = build_variant(&config, &host, s.variant, 256);
            let q = teacher_forced_quality("tiny sabotaged-attn", &reference, |t| {
                c.decode_step_logits(t)
            })
            .unwrap();
            drop(c);
            wrong.push((s.label, mk(&q)));
        }

        eprintln!("[reference] {rq}");
        eprintln!("[shipped]   {honest_q}");
        for (label, q) in &wrong {
            eprintln!("[sabotage {label:?}] {q}");
        }

        let honest_reasons = teacher_forced_reasons(std::slice::from_ref(&honest_q));
        for r in &honest_reasons {
            eprintln!("[shipped reason] {r}");
        }
        let hon_dev = honest_q.perplexity_ratio().ln().abs();
        let mut worst = (SABOTAGES[0].label, 0f64);
        for (label, q) in &wrong {
            let d = q.perplexity_ratio().ln().abs();
            for r in teacher_forced_reasons(std::slice::from_ref(q)) {
                eprintln!("[sabotage reason] {r}");
            }
            eprintln!(
                "log-perplexity deviation from the bf16 reference: shipped {hon_dev:.6}, \
                 sabotaged {d:.6} ({label:?}), ratio {:.2}x",
                d / hon_dev.max(1e-9)
            );
            assert!(
                d > hon_dev,
                "sabotage {label:?} moved the OPEN-ENDED teacher-forced number LESS than the \
                 shipped quantizer did (sabotaged dev {d:.6} <= shipped dev {hon_dev:.6}). Either \
                 the sabotage is not wrong any more, or the shipped quantizer is worse than a \
                 configuration this test calls knowingly broken."
            );
            if d > worst.1 {
                worst = (label, d);
            }
        }
        assert!(
            worst.1 > 0.005 && worst.1 >= MIN_SABOTAGE_SEPARATION * hon_dev,
            "a knowingly wrong quantizer must move the OPEN-ENDED teacher-forced number that this \
             gate reads, and move it much further than the shipped one. shipped dev {hon_dev:.6} \
             (ratio {:.6}), worst sabotage {:?} dev {:.6} = {:.2}x, required \
             {MIN_SABOTAGE_SEPARATION:.1}x",
            honest_q.perplexity_ratio(),
            worst.0,
            worst.1,
            worst.1 / hon_dev.max(1e-9)
        );
        assert!(
            loops,
            "tiny random weights are expected to emit a degenerate repetition loop; if that ever \
             stops being true this test's caveat below needs revisiting"
        );
        assert!(
            honest_reasons.iter().any(|r| r.contains("repetition loop"))
                && wrong
                    .iter()
                    .all(|(_, q)| teacher_forced_reasons(std::slice::from_ref(q))
                        .iter()
                        .any(|r| r.contains("repetition loop"))),
            "both arms are scored on the same degenerate reference, so the gate denies both on \
             substrate grounds; only the differential is a claim about quantization"
        );
        let breached: Vec<&(&'static str, PromptQuality)> = wrong
            .iter()
            .filter(|(_, q)| q.perplexity_ratio() > MAX_PERPLEXITY_RATIO)
            .collect();
        assert!(
            !breached.is_empty(),
            "no sabotage cleared the {MAX_PERPLEXITY_RATIO:.2} perplexity bar, so this test is \
             back to proving only that a number moved. Measured 2026-08-09 on this fixture: \
             coarsening the GRANULARITY alone (int8 one-scale-per-row) reaches ratio 1.0213, \
             under the bar - a random 6-layer net's output distribution sits near uniform \
             (perplexity ~172 of a 512-token vocab) and NLL there is insensitive to logit \
             perturbation. Violating the scale CONTRACT (group-scaled e4m3 weights read back \
             through a single row scale) reaches 1.0931, over it. Keep at least one sabotage of \
             the second kind."
        );
        let (mock_table, mock_per_prompt) = healthy_suite();
        let mut evidence = clean_quality_set(&HEALTHY_CONTROLS);
        {
            let oe = evidence
                .iter_mut()
                .find(|q| q.kind == PromptKind::OpenEnded)
                .unwrap();
            oe.reference = breached[0].1.reference.clone();
            oe.candidate = breached[0].1.candidate.clone();
        }
        let v = evaluate_default_flip_suite(&SuiteFlipEvidence {
            change: "sabotaged attention quantizer, measured on this GPU",
            backend: "wgpu/Vulkan (tiny random weights)",
            model: "tiny",
            table: &mock_table,
            per_prompt: &mock_per_prompt,
            per_prompt_quality: &evidence,
            speed_delta_pct: Some(-11.6),
        });
        eprintln!("{v}");
        assert!(
            !v.is_allow(),
            "END TO END: the ONLY defect in this evidence is a GPU-measured OPEN-ENDED \
             teacher-forced pair produced by sabotage {:?}; every control row is clean. If the \
             gate still allows, the wrong quantizer's damage is not reaching \
             evaluate_default_flip_suite at all. {WHY_OPEN_ENDED_QUALITY}",
            breached[0].0
        );
        let DefaultFlipVerdict::Deny(rs) = &v else {
            unreachable!()
        };
        assert!(
            rs.iter()
                .any(|r| r.contains("OPEN-ENDED") && r.contains("perplexity ratio")),
            "{rs:?}"
        );
        let controls_only: Vec<PromptQuality> = evidence
            .iter()
            .filter(|q| q.kind == PromptKind::Control)
            .cloned()
            .collect();
        assert!(
            !teacher_forced_reasons(&controls_only)
                .iter()
                .any(|r| r.contains("perplexity ratio")),
            "this is the falsification: restore a PromptKind::Control filter over the quality \
             evidence and a quantizer this test just measured as knowingly wrong produces no \
             perplexity reason at all"
        );
    }

    fn shipped_attn_label() -> String {
        let q = nv_models::gemma4_wgpu::attn_quant_config();
        format!("{}/{}", q.fmt.label(), q.group)
    }

    fn pin_ffn_for_a_controlled_ab() -> String {
        let want = std::env::var("NV_FP8_CONTRACT_FFN").unwrap_or_else(|_| "off".to_string());
        std::env::set_var("NV_G4_WGPU_W8_FFN", &want);
        eprintln!(
            "[ffn] NV_G4_WGPU_W8_FFN pinned to {want:?} for BOTH arms; the shipped default is \
             {:?}. {WHY_THE_AB_MUST_HOLD_THE_FFN_FIXED}",
            nv_models::gemma4_wgpu::W8_FFN_DEFAULT
        );
        want
    }

    fn hub_snapshot() -> std::path::PathBuf {
        let home = std::env::var("HOME").unwrap();
        let base = std::path::PathBuf::from(home)
            .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
        std::fs::read_dir(&base)
            .expect("hub snapshot dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.join("config.json").exists())
            .expect("no snapshot with config.json")
    }

    fn occupancy(tag: &str) {
        match std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ])
            .output()
        {
            Ok(o) => eprintln!(
                "[occupancy:{tag}] {}",
                String::from_utf8_lossy(&o.stdout).trim()
            ),
            Err(e) => eprintln!("[occupancy:{tag}] nvidia-smi unavailable: {e}"),
        }
    }

    fn contract_inputs() -> Option<(std::path::PathBuf, PromptPack)> {
        let dir = match prompts::gemma4_nvfp4_dir() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: {e}");
                return None;
            }
        };
        match prompts::resolve_pack(&dir) {
            Ok((p, pack)) => {
                eprintln!("[pack] {}", p.display());
                eprintln!("{WHY_A_PACK}");
                if !wide_pack_enabled() {
                    eprintln!(
                        "[pack] NV_FP8_WIDE_PACK=0: measuring the narrow {}-prompt pack",
                        pack.prompts.len()
                    );
                    return Some((dir, pack));
                }
                let tok = match tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("REFUSING TO WIDEN: tokenizer: {e}");
                        return Some((dir, pack));
                    }
                };
                match widen_pack(&pack, &tok) {
                    Ok((wide, note)) => {
                        eprintln!("[pack] {note}");
                        Some((dir, wide))
                    }
                    Err(e) => {
                        eprintln!(
                            "REFUSING TO WIDEN (falling back to the narrow {}-prompt pack): {e}",
                            pack.prompts.len()
                        );
                        Some((dir, pack))
                    }
                }
            }
            Err(e) => {
                eprintln!("REFUSING TO RUN: {e}");
                None
            }
        }
    }

    #[test]
    #[ignore]
    fn real_gemma4_31b_wgpu_fp8_attention_free_running_contract() {
        let _g = env_lock();
        if std::env::var("NV_FP8_CONTRACT_TEST").ok().as_deref() != Some("1") {
            eprintln!("skipping: set NV_FP8_CONTRACT_TEST=1 to run");
            return;
        }
        if !ctx_or_skip() {
            return;
        }
        let Some((dir, pack)) = contract_inputs() else {
            return;
        };
        let ffn = pin_ffn_for_a_controlled_ab();
        let steps = prompts::max_steps(96);
        let chosen = prompts::ab_prompts(&pack);
        let scored: Vec<&TemplatedPrompt> = pack.prompts.iter().collect();
        let in_ab = |label: &str| chosen.iter().any(|c| c.label == label);
        eprintln!("{}", describe_selection(&pack, &chosen));
        eprintln!("free-running window (max steps) {steps}");
        assert!(!chosen.is_empty(), "the pack contributed no prompt");
        eprintln!(
            "TEACHER-FORCED QUALITY SET: all {} pack prompt(s) - {} CONTROL and {} OPEN-ENDED - \
             regardless of NV_FP8_INCLUDE_OPEN_ENDED, which only governs the FREE-RUNNING A/B \
             set ({} prompt(s) here). {WHY_OPEN_ENDED_QUALITY}",
            scored.len(),
            scored
                .iter()
                .filter(|p| p.kind == PromptKind::Control)
                .count(),
            scored
                .iter()
                .filter(|p| p.kind == PromptKind::OpenEnded)
                .count(),
            chosen.len()
        );
        assert!(
            scored.iter().any(|p| p.kind == PromptKind::OpenEnded),
            "this pack has no OPEN-ENDED prompt, so the quality half of the gate would be \
             control-only and could not deny. {WHY_OPEN_ENDED_QUALITY}"
        );

        eprintln!("loading Gemma4 config from {}", dir.display());
        let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
        let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
        let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).unwrap();
        drop(loader);
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let label_of = |t: u32| tokenizer.id_to_token(t).unwrap_or_else(|| format!("<{t}>"));
        let stops = pack.stop_set();
        occupancy("before");

        let longest = scored.iter().map(|p| p.ids.len()).max().unwrap();
        let max_seq = (longest + steps + 64).next_power_of_two().max(256);
        eprintln!("max_seq {max_seq} (two model builds only; five OOMd at 97887 MiB with a 45 GB co-tenant)");

        let mut table = RunTable::new("Gemma4-31B-IT-NVFP4 wgpu, bf16 attention vs fp8 attention");
        let mut refs: Vec<FreeRun> = Vec::new();
        let mut ref_logits: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut ref_obs: Vec<ArmObservation> = Vec::new();
        let mut ref_quality: Vec<TeacherForcedQuality> = Vec::new();

        let mut a = build(&config, &host, false, max_seq);
        let bytes_bf16 = a.weight_bytes_per_token();
        let passes_bf16 = a.pass_count();
        for p in &scored {
            a.reset();
            let mut kept: Vec<Vec<f32>> = Vec::new();
            let mut r = prompts::free_running("bf16-attn", p, &stops, steps, |t| {
                let (_, l) = a.decode_step_logits(t)?;
                kept.push(l.clone());
                Ok(l)
            })
            .unwrap();
            r.text = tokenizer.decode(&r.tokens, false).unwrap_or_default();
            let at_gen = p.ids.len() - 1;
            kept.truncate(at_gen + r.tokens.len());
            let aligned = kept.split_off(at_gen);
            let mut nll_sum = 0f64;
            let mut rank = Vec::new();
            let mut margin = Vec::new();
            for (d, t) in r.tokens.iter().enumerate() {
                if d >= aligned.len() {
                    break;
                }
                let (n, rk, m) = log_softmax_at(&aligned[d], *t);
                nll_sum += n;
                rank.push(rk);
                margin.push(m);
            }
            let n = rank.len().max(1);
            ref_quality.push(TeacherForcedQuality {
                arm: "bf16-attn".into(),
                nll_mean: nll_sum / n as f64,
                perplexity: (nll_sum / n as f64).exp(),
                reference_rank: rank,
                reference_margin: margin,
            });
            let ob = observe(p, &r, &stops, steps, &label_of);
            table.push(ob.clone());
            if in_ab(&p.label) {
                ref_logits.push(aligned);
            } else {
                ref_logits.push(Vec::new());
            }
            ref_obs.push(ob);
            refs.push(r);
        }
        drop(a);
        eprintln!("[bf16-attn] {passes_bf16} passes, {bytes_bf16} weight bytes/token");
        occupancy("after-bf16");

        let mut cands: Vec<Option<FreeRun>> = Vec::new();
        let mut forced_reports = Vec::new();
        let mut b = build(&config, &host, true, max_seq);
        let bytes_fp8 = b.weight_bytes_per_token();
        let passes_fp8 = b.pass_count();
        for (i, p) in scored.iter().enumerate() {
            if in_ab(&p.label) {
                b.reset();
                let mut r = prompts::free_running("fp8-attn", p, &stops, steps, |t| {
                    b.decode_step_logits(t).map(|(_, l)| l)
                })
                .unwrap();
                r.text = tokenizer.decode(&r.tokens, false).unwrap_or_default();
                table.push(observe(p, &r, &stops, steps, &label_of));
                cands.push(Some(r));
            } else {
                cands.push(None);
            }

            b.reset();
            let traj = trajectory_from_free_run(p, &refs[i]);
            let (forced, cq) =
                forced_context_replay("fp8-attn", &traj, |t| b.decode_step_logits(t)).unwrap();
            forced_reports.push((p.label.clone(), forced, cq));
        }
        drop(b);
        occupancy("after-fp8");

        eprintln!("[fp8-attn] {passes_fp8} passes, {bytes_fp8} weight bytes/token");
        eprintln!("{table}");
        table
            .assert_controls_terminated_for("bf16-attn")
            .expect("the bf16 reference must end its turn on every control prompt");
        assert!(
            table.rows.iter().any(|r| r.kind == PromptKind::OpenEnded),
            "the run table carries no OPEN-ENDED row, so the HEALTH half of this gate \
             (ended_turn / repetition loop / mid-stream stop-token leak in \
             evaluate_default_flip_suite) is CONTROL-only and cannot deny. Until 2026-08-09 the \
             table was populated under `if in_ab(..)`, which is the A/B (CONTROL) set. \
             {WHY_OPEN_ENDED_QUALITY}"
        );

        let mut suite = SuiteReport::new(
            "fp8 attention free-running contract",
            "bf16-attn",
            "fp8-attn",
        );
        let mut per_prompt = Vec::new();
        for (i, p) in scored.iter().enumerate() {
            let Some(c) = &cands[i] else { continue };
            suite.push(prompts::compare(p, &refs[i], c));
            per_prompt.push((
                p.label.clone(),
                p.kind,
                compare_free_running(
                    &trajectory_from_free_run(p, &refs[i]),
                    &trajectory_from_free_run(p, c),
                ),
            ));
        }
        suite.validate().unwrap();
        eprintln!("{suite}");
        for (label, forced, cq) in &forced_reports {
            eprintln!("[{label}] {forced}");
            eprintln!("[{label}] {cq}");
        }

        let specials: Vec<u32> = stops.ids.clone();
        for (i, p) in scored.iter().enumerate() {
            let Some(cand) = &cands[i] else { continue };
            let Some((_, _, c)) = per_prompt.iter().find(|(l, _, _)| *l == p.label) else {
                continue;
            };
            eprintln!(
                "[{}] extra stop/control token occurrences in the fp8 arm: {}",
                p.label,
                c.extra_occurrences_of(&specials)
            );
            if let Some(d) = c.first_divergence {
                let lg = &ref_logits[i];
                if d < lg.len() {
                    let ref_tok = refs[i].tokens[d];
                    let cand_tok = cand.tokens[d];
                    let (r1, m1) = rank_and_margin_of(&lg[d], cand_tok);
                    let (r2, m2) = rank_and_margin_of(&lg[d], ref_tok);
                    eprintln!(
                        "[{}] DIVERGENCE POINT {d}: bf16 emitted {ref_tok} ({:?}) [rank {r2} margin \
                         {m2:.4}], fp8 emitted {cand_tok} ({:?}) which under the bf16 distribution \
                         has RANK {r1} and logit margin {m1:.4} below bf16's top-1",
                        p.label,
                        label_of(ref_tok),
                        label_of(cand_tok)
                    );
                    let top: Vec<String> = (0..5)
                        .map(|n| {
                            let t = nth_best_token(&lg[d], n);
                            format!("{t}={:?}", label_of(t))
                        })
                        .collect();
                    eprintln!("[{}] bf16 top-5 at {d}: {}", p.label, top.join(", "));
                }
            }
        }
        eprintln!("{MARGIN_TRAP}");

        eprintln!("{REFERENCE_QUALITY_DOC}");
        for (i, p) in scored.iter().enumerate() {
            eprintln!("[{}] {}", p.label, ref_quality[i]);
        }

        let quality: Vec<PromptQuality> = scored
            .iter()
            .enumerate()
            .map(|(i, p)| PromptQuality {
                label: p.label.clone(),
                kind: p.kind,
                reference_ended_turn: ref_obs[i].ended_turn,
                reference_loops: ref_obs[i].repeated_tail.is_some(),
                reference: ref_quality[i].clone(),
                candidate: forced_reports[i].2.clone(),
            })
            .collect();
        for q in &quality {
            eprintln!("[quality] {q}");
        }
        let worst = quality
            .iter()
            .filter(|q| q.perplexity_ratio().is_finite())
            .max_by(|a, b| {
                a.perplexity_ratio()
                    .partial_cmp(&b.perplexity_ratio())
                    .unwrap()
            });
        if let Some(q) = worst {
            eprintln!(
                "WORST TEACHER-FORCED RATIO is {} {:?} at {:.6} (allowed {MAX_PERPLEXITY_RATIO:.2}). \
                 Ranked by RATIO, not by absolute candidate perplexity: an open-ended prompt whose \
                 reference perplexity is low can carry the worst ratio in the pack while never \
                 being the highest absolute number, and the pre-2026-08-09 argmax-over-absolute \
                 selection handed exactly that prompt to nobody.",
                q.kind_label(),
                q.label,
                q.perplexity_ratio()
            );
        }
        let change = format!(
            "quantized attention projections (the hardcoded default variant, shipped format {}), FFN pinned \
             to NV_G4_WGPU_W8_FFN={ffn:?} in BOTH arms",
            shipped_attn_label()
        );
        eprintln!(
            "SPEED DELTA IS A RECORDED CONSTANT, NOT MEASURED BY THIS RUN: the -11.6% handed to \
             evaluate_default_flip_suite below is the 2026-08-07 lane iv-fp8 figure. This harness \
             times nothing, so the speed half of the gate cannot deny here and must not be quoted \
             from this run."
        );
        let verdict = evaluate_default_flip_suite(&SuiteFlipEvidence {
            change: &change,
            backend: "wgpu/Vulkan",
            model: "Gemma4-31B-IT-NVFP4",
            table: &table,
            per_prompt: &per_prompt,
            per_prompt_quality: &quality,
            speed_delta_pct: Some(-11.6),
        });
        eprintln!("{verdict}");
        eprintln!(
            "weight bytes/token {bytes_bf16} -> {bytes_fp8} ({:.2}% cut), passes {passes_bf16} -> {passes_fp8}",
            100.0 * (bytes_bf16 as f64 - bytes_fp8 as f64) / bytes_bf16 as f64
        );
        occupancy("after");
        eprintln!("acceptance bar and rationale: {CONTRACT_DOC}.");
        assert_eq!(suite.rows.len(), chosen.len());

        set_attn_variant(None);
        let default_on = nv_models::gemma4_wgpu::attn_fp8_enabled();
        let record_only = std::env::var("NV_FP8_CONTRACT_RECORD_ONLY").ok().as_deref() == Some("1");
        let agrees = verdict.is_allow() == default_on;
        let worst_line = worst
            .map(|q| {
                format!(
                    "{} {:?} at ratio {:.6}",
                    q.kind_label(),
                    q.label,
                    q.perplexity_ratio()
                )
            })
            .unwrap_or_else(|| "none".into());
        assert!(
            agrees || record_only,
            "RECORDED VERDICT AND SHIPPED DEFAULT DISAGREE. This run measured {verdict}\n\nand \
             the shipped wgpu attention-projection fp8 default is {}. Until 2026-08-09 this test \
             printed its verdict and returned ok regardless, which is the same defect as the \
             Control filter it was written to catch: evidence that cannot fail anything is not \
             evidence. Worst teacher-forced ratio: {worst_line}. Resolve it by RE-BLESSING (the \
             bar in {CONTRACT_DOC} is wrong for a weight-format flip, so change the bar and say \
             why) or by FLIPPING THE DEFAULT in nv-models::gemma4_wgpu. Set \
             NV_FP8_CONTRACT_RECORD_ONLY=1 to collect evidence mid-investigation without \
             deciding. Do NOT silence it by narrowing the prompt set again.",
            if default_on { "ON" } else { "OFF" }
        );
    }

    #[test]
    #[ignore]
    fn real_gemma4_31b_wgpu_bf16_control_one_substituted_token_at_the_divergence_point() {
        let _g = env_lock();
        if std::env::var("NV_FP8_CONTRACT_TEST").ok().as_deref() != Some("1") {
            eprintln!("skipping: set NV_FP8_CONTRACT_TEST=1 to run");
            return;
        }
        if !ctx_or_skip() {
            return;
        }
        let Some((dir, pack)) = contract_inputs() else {
            return;
        };
        let _ffn = pin_ffn_for_a_controlled_ab();
        let steps = prompts::max_steps(96);
        let at: usize = std::env::var("NV_FP8_CONTRACT_SUBST_AT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let rank: usize = std::env::var("NV_FP8_CONTRACT_SUBST_RANK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let chosen = prompts::ab_prompts(&pack);
        eprintln!("{}", describe_selection(&pack, &chosen));

        let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
        let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
        let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).unwrap();
        drop(loader);
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let label_of = |t: u32| tokenizer.id_to_token(t).unwrap_or_else(|| format!("<{t}>"));
        let stops = pack.stop_set();
        let longest = chosen.iter().map(|p| p.ids.len()).max().unwrap();
        let max_seq = (longest + steps + 64).next_power_of_two().max(256);

        let mut table = RunTable::new("bf16 control, one substituted token");
        let mut m = build(&config, &host, false, max_seq);
        for p in &chosen {
            m.reset();
            let mut kept: Vec<Vec<f32>> = Vec::new();
            let mut reference = prompts::free_running("bf16-attn", p, &stops, steps, |t| {
                let (_, l) = m.decode_step_logits(t)?;
                kept.push(l.clone());
                Ok(l)
            })
            .unwrap();
            reference.text = tokenizer
                .decode(&reference.tokens, false)
                .unwrap_or_default();
            table.push(observe(p, &reference, &stops, steps, &label_of));
            if reference.tokens.len() <= at {
                eprintln!(
                    "[{}] SKIP substitution: the reference ended its turn after {} token(s), \
                     before the substitution index {at}. That is the healthy outcome for a control \
                     prompt; pick a later index or an open-ended prompt with \
                     NV_FP8_INCLUDE_OPEN_ENDED=1.",
                    p.label,
                    reference.tokens.len()
                );
                continue;
            }
            let gen_start = p.ids.len() - 1;
            let aligned = &kept[gen_start..gen_start + reference.tokens.len()];
            let top = reference.tokens[at];
            assert_eq!(
                top,
                nth_best_token(&aligned[at], 0),
                "alignment check: aligned[{at}] must be the distribution that emitted tokens[{at}]"
            );
            let alt = nth_best_token(&aligned[at], rank);
            let margin = aligned[at][top as usize] - aligned[at][alt as usize];
            eprintln!(
                "[{}] at step {at} top1 {top} ({:?}) vs rank-{} {alt} ({:?}), logit margin {margin:.4}",
                p.label,
                label_of(top),
                rank + 1,
                label_of(alt)
            );

            m.reset();
            let traj = trajectory_from_free_run(p, &reference);
            let perturbed = free_running_with_substitution(
                "bf16-attn + one substituted token",
                &traj.prompt,
                reference.tokens.len(),
                at,
                alt,
                |t| m.decode_step(t),
            )
            .unwrap();
            let cmp = compare_free_running(&traj, &perturbed);
            eprintln!("[{}] {cmp}", p.label);
            eprintln!("[{}] [bf16 reference]    {:?}", p.label, reference.text);
            eprintln!(
                "[{}] [bf16 + substitution] {:?}",
                p.label,
                tokenizer
                    .decode(&perturbed.tokens, false)
                    .unwrap_or_default()
            );
            eprintln!(
                "[{}] extra stop/control token occurrences after the substitution = {}",
                p.label,
                cmp.extra_occurrences_of(&stops.ids)
            );
        }
        drop(m);
        eprintln!("{table}");
        table
            .assert_controls_terminated_for("bf16-attn")
            .expect("the unperturbed bf16 control must end its turn");
        eprintln!(
            "CONTROL: this is the UNQUANTIZED model, perturbed by exactly one token. If it also \
             cascades and also emits control tokens, then the fp8 arm's collapse is greedy-decoding \
             amplification of a single near-tie flip, not fp8 producing garbage - and the fix is \
             not in the quantizer. If it stays coherent, the fp8 arm is genuinely broken \
             downstream of position {at}."
        );
        eprintln!("{REPRODUCIBILITY_LIMIT}");
    }
}

#[cfg(not(feature = "wgpu"))]
#[test]
#[allow(non_snake_case)]
fn wgpu_arms_SKIPPED_no_wgpu_feature() {
    eprintln!(
        "wgpu arms of the fp8 free-running contract were CFG'd OUT of this binary (no `wgpu` \
         feature). This is a SKIP, not a pass. Re-run with NVK_FEATURES=wgpu or cuda,wgpu."
    );
}
