#![allow(dead_code)]

use std::fmt;
use std::path::Path;

pub const HARNESS_DOC: &str = "docs/book/08.1-quality-harness.md";
pub const MIN_PROMPTS_FOR_A_CLAIM: usize = 2;
pub const RESYNC_RUN: usize = 4;
pub const NEAR_TIE_RATIO: f32 = 0.25;
pub const DEFAULT_MAX_STEPS: usize = 96;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PromptKind {
    Control,
    OpenEnded,
}

impl fmt::Display for PromptKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PromptKind::Control => write!(f, "CONTROL"),
            PromptKind::OpenEnded => write!(f, "open-ended"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Provenance {
    OfficialTemplate,
    RawUntemplatedNotServingEvidence,
}

impl fmt::Display for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Provenance::OfficialTemplate => write!(f, "official-template"),
            Provenance::RawUntemplatedNotServingEvidence => {
                write!(f, "RAW-UNTEMPLATED(not-serving-evidence)")
            }
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TemplatedPrompt {
    pub label: String,
    pub kind: PromptKind,
    pub provenance: Provenance,
    pub model_repo: String,
    pub snapshot: String,
    pub template_digest: String,
    pub template_bytes: usize,
    pub rendered: String,
    pub ids: Vec<u32>,
}

impl TemplatedPrompt {
    pub fn from_official_render(
        label: &str,
        kind: PromptKind,
        model_repo: &str,
        snapshot: &str,
        template_digest: &str,
        template_bytes: usize,
        rendered: String,
        ids: Vec<u32>,
    ) -> Self {
        assert!(
            !ids.is_empty(),
            "templated prompt {label} tokenized to nothing"
        );
        Self {
            label: label.to_string(),
            kind,
            provenance: Provenance::OfficialTemplate,
            model_repo: model_repo.to_string(),
            snapshot: snapshot.to_string(),
            template_digest: template_digest.to_string(),
            template_bytes,
            rendered,
            ids,
        }
    }

    pub fn raw_untemplated_not_serving_evidence(
        label: &str,
        rendered: String,
        ids: Vec<u32>,
    ) -> Self {
        Self {
            label: label.to_string(),
            kind: PromptKind::OpenEnded,
            provenance: Provenance::RawUntemplatedNotServingEvidence,
            model_repo: "<none>".into(),
            snapshot: "<none>".into(),
            template_digest: "<none>".into(),
            template_bytes: 0,
            rendered,
            ids,
        }
    }

    pub fn is_serving_shaped(&self) -> bool {
        matches!(self.provenance, Provenance::OfficialTemplate)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PromptPack {
    pub model_repo: String,
    pub snapshot: String,
    pub template_digest: String,
    pub template_bytes: usize,
    pub stop_ids: Vec<u32>,
    pub stop_source: String,
    pub prompts: Vec<TemplatedPrompt>,
}

impl PromptPack {
    pub fn write_json(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }

    pub fn read_json(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read(path)?;
        Ok(serde_json::from_slice(&raw)?)
    }

    pub fn load_for_snapshot(path: &Path, weights_dir: &Path) -> anyhow::Result<Self> {
        let pack = Self::read_json(path)?;
        let live = template_digest_of_dir(weights_dir)?;
        anyhow::ensure!(
            live.0 == pack.template_digest && live.1 == pack.template_bytes,
            "prompt pack {} was rendered from template {} ({} bytes) but the weights at {} ship \
             template {} ({} bytes). Re-render the pack against the snapshot you are about to \
             measure; see {HARNESS_DOC}.",
            path.display(),
            pack.template_digest,
            pack.template_bytes,
            weights_dir.display(),
            live.0,
            live.1
        );
        anyhow::ensure!(
            pack.prompts.iter().all(|p| p.is_serving_shaped()),
            "prompt pack {} contains a non-templated prompt; refusing to load",
            path.display()
        );
        Ok(pack)
    }

    pub fn stop_set(&self) -> StopSet {
        StopSet {
            ids: self.stop_ids.clone(),
            source: self.stop_source.clone(),
        }
    }

    pub fn controls(&self) -> usize {
        self.prompts
            .iter()
            .filter(|p| p.kind == PromptKind::Control)
            .count()
    }
}

pub fn digest_hex(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    let mut g: u64 = 0x9e37_79b9_7f4a_7c15 ^ (bytes.len() as u64);
    for chunk in bytes.chunks(8) {
        let mut v: u64 = 0;
        for (i, b) in chunk.iter().enumerate() {
            v |= (*b as u64) << (8 * i);
        }
        g ^= v.wrapping_mul(0xff51_afd7_ed55_8ccd).rotate_left(31);
        g = g.rotate_left(27).wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    }
    format!("{h:016x}{g:016x}")
}

pub fn template_digest_of_dir(dir: &Path) -> anyhow::Result<(String, usize)> {
    let p = dir.join("chat_template.jinja");
    let raw = std::fs::read(&p)
        .map_err(|e| anyhow::anyhow!("no chat_template.jinja at {}: {e}", p.display()))?;
    Ok((digest_hex(&raw), raw.len()))
}

#[derive(Clone, Debug)]
pub struct StopSet {
    pub ids: Vec<u32>,
    pub source: String,
}

impl StopSet {
    pub fn from_generation_config(dir: &Path) -> anyhow::Result<Self> {
        let p = dir.join("generation_config.json");
        let raw = std::fs::read_to_string(&p)
            .map_err(|e| anyhow::anyhow!("no generation_config.json at {}: {e}", p.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw)?;
        let ids = match v.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => vec![n.as_u64().unwrap_or_default() as u32],
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .filter_map(|x| x.as_u64())
                .map(|x| x as u32)
                .collect(),
            _ => anyhow::bail!("{} has no eos_token_id", p.display()),
        };
        anyhow::ensure!(!ids.is_empty(), "{} has an empty eos_token_id", p.display());
        Ok(Self {
            ids,
            source: format!("{}::eos_token_id", p.display()),
        })
    }

    pub fn contains(&self, t: u32) -> bool {
        self.ids.contains(&t)
    }
}

impl fmt::Display for StopSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stop ids {:?} from {}", self.ids, self.source)
    }
}

pub fn top2(logits: &[f32]) -> (u32, f32, u32, f32) {
    let mut i1 = 0usize;
    let mut v1 = f32::NEG_INFINITY;
    let mut i2 = 0usize;
    let mut v2 = f32::NEG_INFINITY;
    for (i, v) in logits.iter().copied().enumerate() {
        if v > v1 {
            i2 = i1;
            v2 = v1;
            i1 = i;
            v1 = v;
        } else if v > v2 {
            i2 = i;
            v2 = v;
        }
    }
    (i1 as u32, v1, i2 as u32, v2)
}

pub fn median(mut xs: Vec<f32>) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StopReason {
    HitStopToken,
    ReachedMaxSteps,
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StopReason::HitStopToken => write!(f, "stop-token"),
            StopReason::ReachedMaxSteps => write!(f, "max-steps"),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FreeRun {
    pub arm: String,
    pub prompt_label: String,
    pub tokens: Vec<u32>,
    pub margins: Vec<f32>,
    pub reason: StopReason,
    pub stop_token: Option<u32>,
    pub text: String,
}

impl FreeRun {
    pub fn stopped_at(&self) -> Option<usize> {
        match self.reason {
            StopReason::HitStopToken => Some(self.tokens.len() - 1),
            StopReason::ReachedMaxSteps => None,
        }
    }
    pub fn median_margin(&self) -> f32 {
        median(self.margins.clone())
    }
}

pub fn free_running<F>(
    arm: &str,
    prompt: &TemplatedPrompt,
    stops: &StopSet,
    max_steps: usize,
    mut step: F,
) -> anyhow::Result<FreeRun>
where
    F: FnMut(u32) -> anyhow::Result<Vec<f32>>,
{
    anyhow::ensure!(
        prompt.is_serving_shaped(),
        "free_running refuses prompt {:?}: provenance is {}. Render it through the model's own \
         chat template first; see {HARNESS_DOC}.",
        prompt.label,
        prompt.provenance
    );
    anyhow::ensure!(max_steps > 0, "free_running needs max_steps > 0");
    let mut logits = Vec::new();
    for t in &prompt.ids {
        logits = step(*t)?;
    }
    anyhow::ensure!(!logits.is_empty(), "arm {arm} returned empty logits");
    let mut tokens = Vec::new();
    let mut margins = Vec::new();
    let mut reason = StopReason::ReachedMaxSteps;
    let mut stop_token = None;
    for _ in 0..max_steps {
        let (t1, v1, _, v2) = top2(&logits);
        tokens.push(t1);
        margins.push(v1 - v2);
        if stops.contains(t1) {
            reason = StopReason::HitStopToken;
            stop_token = Some(t1);
            break;
        }
        logits = step(t1)?;
    }
    Ok(FreeRun {
        arm: arm.to_string(),
        prompt_label: prompt.label.clone(),
        tokens,
        margins,
        reason,
        stop_token,
        text: String::new(),
    })
}

pub fn free_running_batch_prefilled<P, F>(
    arm: &str,
    prompt: &TemplatedPrompt,
    stops: &StopSet,
    max_steps: usize,
    prefill: P,
    mut step: F,
) -> anyhow::Result<FreeRun>
where
    P: FnOnce(&[u32]) -> anyhow::Result<Vec<f32>>,
    F: FnMut(u32) -> anyhow::Result<Vec<f32>>,
{
    anyhow::ensure!(
        prompt.is_serving_shaped(),
        "free_running_batch_prefilled refuses prompt {:?}: provenance is {}. Render it through \
         the model's own chat template first; see {HARNESS_DOC}.",
        prompt.label,
        prompt.provenance
    );
    anyhow::ensure!(
        max_steps > 0,
        "free_running_batch_prefilled needs max_steps > 0"
    );
    let mut logits = prefill(&prompt.ids)?;
    anyhow::ensure!(
        !logits.is_empty(),
        "arm {arm} returned empty prefill logits"
    );
    let mut tokens = Vec::new();
    let mut margins = Vec::new();
    let mut reason = StopReason::ReachedMaxSteps;
    let mut stop_token = None;
    for _ in 0..max_steps {
        let (t1, v1, _, v2) = top2(&logits);
        tokens.push(t1);
        margins.push(v1 - v2);
        if stops.contains(t1) {
            reason = StopReason::HitStopToken;
            stop_token = Some(t1);
            break;
        }
        logits = step(t1)?;
    }
    Ok(FreeRun {
        arm: arm.to_string(),
        prompt_label: prompt.label.clone(),
        tokens,
        margins,
        reason,
        stop_token,
        text: String::new(),
    })
}

#[derive(Clone, Debug)]
pub struct ForcedContextRun {
    pub arm: String,
    pub prompt_label: String,
    pub forced: Vec<u32>,
    pub argmax: Vec<u32>,
}

impl ForcedContextRun {
    pub fn upper_bound_agreement_is_not_serving_evidence(&self) -> (usize, usize) {
        let n = self.forced.len().min(self.argmax.len());
        let a = (0..n)
            .filter(|i| self.forced[*i] == self.argmax[*i])
            .count();
        (a, n)
    }
    pub fn why_this_is_not_serving_evidence() -> &'static str {
        "forced-context replay re-feeds the REFERENCE tokens at every step, so the candidate never \
         sees its own mistakes. It is an upper bound on agreement and cannot detect the decoding \
         cascade that serving actually exhibits. Report it only next to a free-running number."
    }
}

pub fn forced_context<F>(
    arm: &str,
    prompt: &TemplatedPrompt,
    forced: &[u32],
    mut step: F,
) -> anyhow::Result<ForcedContextRun>
where
    F: FnMut(u32) -> anyhow::Result<Vec<f32>>,
{
    anyhow::ensure!(
        prompt.is_serving_shaped(),
        "forced_context refuses a raw prompt"
    );
    let mut logits = Vec::new();
    for t in &prompt.ids {
        logits = step(*t)?;
    }
    let mut argmax = Vec::with_capacity(forced.len());
    for (i, f) in forced.iter().enumerate() {
        let (t1, _, _, _) = top2(&logits);
        argmax.push(t1);
        if i + 1 < forced.len() {
            logits = step(*f)?;
        }
    }
    Ok(ForcedContextRun {
        arm: arm.to_string(),
        prompt_label: prompt.label.clone(),
        forced: forced.to_vec(),
        argmax,
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
pub struct PairReport {
    pub prompt_label: String,
    pub prompt_kind: PromptKind,
    pub reference_arm: String,
    pub candidate_arm: String,
    pub agree: usize,
    pub total: usize,
    pub first_divergence: Option<usize>,
    pub resynced_at: Option<usize>,
    pub shape: DivergenceShape,
    pub post_divergence_agree: usize,
    pub post_divergence_total: usize,
    pub margin_at_divergence: Option<f32>,
    pub divergence_margins: Vec<(usize, f32)>,
    pub reference_median_margin: f32,
    pub same_termination: bool,
    pub reference_stop: (StopReason, Option<usize>),
    pub candidate_stop: (StopReason, Option<usize>),
    pub reference_text: String,
    pub candidate_text: String,
    pub reference_tokens: Vec<u32>,
    pub candidate_tokens: Vec<u32>,
}

impl PairReport {
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

    pub fn margin_ratio(&self) -> Option<f32> {
        let m = self.margin_at_divergence?;
        if self.reference_median_margin <= 0.0 {
            return None;
        }
        Some(m / self.reference_median_margin)
    }

    pub fn near_tie_divergences(&self) -> usize {
        let med = self.reference_median_margin;
        if med <= 0.0 {
            return 0;
        }
        self.divergence_margins
            .iter()
            .filter(|(_, m)| *m / med < NEAR_TIE_RATIO)
            .count()
    }

    pub fn all_divergences_are_near_ties(&self) -> bool {
        !self.divergence_margins.is_empty()
            && self.near_tie_divergences() == self.divergence_margins.len()
    }

    pub fn divergence_reads_as(&self) -> &'static str {
        match (self.first_divergence, self.margin_ratio(), self.shape) {
            (None, _, _) => "no divergence",
            (Some(_), Some(r), DivergenceShape::Recovered) if r < 0.25 => {
                "TIE-BREAK: divergence sits far below the reference's median decisiveness and the \
                 arms resynchronised. This is position ambiguity, not damage."
            }
            (Some(_), Some(r), _) if r < 0.25 => {
                "AMBIGUOUS POSITION that then cascaded: the step itself was a near-tie, so the \
                 downstream text difference is decoding amplification rather than proof of a bad \
                 kernel. Re-run with a substituted token on the reference arm to separate them."
            }
            (Some(_), _, DivergenceShape::Cascaded) => {
                "FLIP: the reference was decisive here and the candidate still disagreed, and the \
                 arms never resynchronised. Treat as a real regression."
            }
            _ => {
                "FLIP at a decisive step, but the arms resynchronised. Real but locally contained."
            }
        }
    }
}

impl fmt::Display for PairReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "  prompt {:?} [{}]  FREE-RUNNING agreement {}/{} ({:.1}%)",
            self.prompt_label,
            self.prompt_kind,
            self.agree,
            self.total,
            100.0 * self.agreement()
        )?;
        writeln!(
            f,
            "    first divergence {:?}, shape {}, resynced_at {:?}, post-divergence {}/{} ({:.1}%)",
            self.first_divergence,
            self.shape,
            self.resynced_at,
            self.post_divergence_agree,
            self.post_divergence_total,
            100.0 * self.post_divergence_rate()
        )?;
        match (self.margin_at_divergence, self.margin_ratio()) {
            (Some(m), Some(r)) => writeln!(
                f,
                "    top-2 margin at divergence {m:.4} vs reference median {:.4} (ratio {r:.3})",
                self.reference_median_margin
            )?,
            _ => writeln!(
                f,
                "    reference median top-2 margin {:.4}",
                self.reference_median_margin
            )?,
        }
        writeln!(
            f,
            "    termination: reference {} at {:?}, candidate {} at {:?} -> {}",
            self.reference_stop.0,
            self.reference_stop.1,
            self.candidate_stop.0,
            self.candidate_stop.1,
            if self.same_termination {
                "SAME"
            } else {
                "DIFFERENT (arms ended their turn in different places)"
            }
        )?;
        if !self.divergence_margins.is_empty() {
            writeln!(
                f,
                "    {} disagreeing step(s) in the common prefix, {} of them below {:.2}x the \
                 reference median margin",
                self.divergence_margins.len(),
                self.near_tie_divergences(),
                NEAR_TIE_RATIO,
            )?;
            writeln!(
                f,
                "      (only the FIRST is independent evidence: once the arms cascade they are \
                 decoding different contexts, so a decisive reference margin at a later step is \
                 expected and is NOT a second fault{})",
                if self.all_divergences_are_near_ties() {
                    ". Here every disagreement was a near-tie, which is decoding amplification \
                     rather than a kernel fault"
                } else {
                    ""
                }
            )?;
        }
        writeln!(f, "    reads as: {}", self.divergence_reads_as())?;
        writeln!(f, "    [{}] {:?}", self.reference_arm, self.reference_text)?;
        write!(f, "    [{}] {:?}", self.candidate_arm, self.candidate_text)
    }
}

pub fn compare(prompt: &TemplatedPrompt, reference: &FreeRun, candidate: &FreeRun) -> PairReport {
    let total = reference.tokens.len().max(candidate.tokens.len());
    let common = reference.tokens.len().min(candidate.tokens.len());
    let mut agree = 0usize;
    let mut first_divergence = None;
    for i in 0..common {
        if reference.tokens[i] == candidate.tokens[i] {
            agree += 1;
        } else if first_divergence.is_none() {
            first_divergence = Some(i);
        }
    }
    if first_divergence.is_none() && reference.tokens.len() != candidate.tokens.len() {
        first_divergence = Some(common);
    }

    let mut resynced_at = None;
    if let Some(d) = first_divergence {
        let mut run = 0usize;
        for i in d..common {
            if reference.tokens[i] == candidate.tokens[i] {
                run += 1;
                if run >= RESYNC_RUN {
                    resynced_at = Some(i + 1 - run);
                    break;
                }
            } else {
                run = 0;
            }
        }
    }

    let shape = match (first_divergence, resynced_at) {
        (None, _) => DivergenceShape::Identical,
        (Some(_), Some(_)) => DivergenceShape::Recovered,
        (Some(_), None) => DivergenceShape::Cascaded,
    };

    let (post_agree, post_total) = match first_divergence {
        None => (0, 0),
        Some(d) => {
            let n = common.saturating_sub(d + 1);
            let a = (d + 1..common)
                .filter(|i| reference.tokens[*i] == candidate.tokens[*i])
                .count();
            (a, n)
        }
    };

    let margin_at_divergence = first_divergence.and_then(|d| reference.margins.get(d).copied());
    let divergence_margins: Vec<(usize, f32)> = (0..common)
        .filter(|i| reference.tokens[*i] != candidate.tokens[*i])
        .filter_map(|i| reference.margins.get(i).map(|m| (i, *m)))
        .collect();

    let ref_stop = (reference.reason, reference.stopped_at());
    let cand_stop = (candidate.reason, candidate.stopped_at());

    PairReport {
        prompt_label: prompt.label.clone(),
        prompt_kind: prompt.kind,
        reference_arm: reference.arm.clone(),
        candidate_arm: candidate.arm.clone(),
        agree,
        total,
        first_divergence,
        resynced_at,
        shape,
        post_divergence_agree: post_agree,
        post_divergence_total: post_total,
        margin_at_divergence,
        divergence_margins,
        reference_median_margin: reference.median_margin(),
        same_termination: ref_stop == cand_stop,
        reference_stop: ref_stop,
        candidate_stop: cand_stop,
        reference_text: reference.text.clone(),
        candidate_text: candidate.text.clone(),
        reference_tokens: reference.tokens.clone(),
        candidate_tokens: candidate.tokens.clone(),
    }
}

#[derive(Clone, Debug)]
pub struct SuiteReport {
    pub title: String,
    pub reference_arm: String,
    pub candidate_arm: String,
    pub rows: Vec<PairReport>,
}

impl SuiteReport {
    pub fn new(title: &str, reference_arm: &str, candidate_arm: &str) -> Self {
        Self {
            title: title.to_string(),
            reference_arm: reference_arm.to_string(),
            candidate_arm: candidate_arm.to_string(),
            rows: Vec::new(),
        }
    }

    pub fn push(&mut self, row: PairReport) {
        self.rows.push(row);
    }

    pub fn controls(&self) -> Vec<&PairReport> {
        self.rows
            .iter()
            .filter(|r| r.prompt_kind == PromptKind::Control)
            .collect()
    }

    pub fn worst_agreement(&self) -> f64 {
        self.rows
            .iter()
            .map(|r| r.agreement())
            .fold(1.0f64, |a, b| a.min(b))
    }

    pub fn any_termination_mismatch(&self) -> bool {
        self.rows.iter().any(|r| !r.same_termination)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.rows.len() >= MIN_PROMPTS_FOR_A_CLAIM,
            "a quality claim needs at least {MIN_PROMPTS_FOR_A_CLAIM} prompts, got {}. A single \
             prompt cannot distinguish a regression from an ambiguous position; see {HARNESS_DOC}.",
            self.rows.len()
        );
        anyhow::ensure!(
            !self.controls().is_empty(),
            "a quality claim needs at least one CONTROL (low-entropy) prompt. Without it a \
             divergence cannot be attributed; see {HARNESS_DOC}."
        );
        Ok(())
    }

    pub fn assert_controls_exact(&self) -> anyhow::Result<()> {
        for c in self.controls() {
            anyhow::ensure!(
                c.shape == DivergenceShape::Identical && c.same_termination,
                "CONTROL prompt {:?} diverged (agreement {}/{}, shape {}, same_termination {}). A \
                 low-entropy control must come out exact; something is genuinely wrong.",
                c.prompt_label,
                c.agree,
                c.total,
                c.shape,
                c.same_termination
            );
        }
        Ok(())
    }
}

impl fmt::Display for SuiteReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "=== {} :: reference [{}] vs candidate [{}] :: {} prompts, PER-PROMPT (never pooled) ===",
            self.title,
            self.reference_arm,
            self.candidate_arm,
            self.rows.len()
        )?;
        for r in &self.rows {
            writeln!(f, "{r}")?;
        }
        write!(
            f,
            "  worst per-prompt agreement {:.1}%, termination mismatch on any prompt: {}",
            100.0 * self.worst_agreement(),
            self.any_termination_mismatch()
        )
    }
}

pub const FLIP_BAR_DOC: &str = "\
THE BAR, WRITTEN DOWN (default-flip acceptance, quality axis only).

Why not bit-identical: a bit-exact bar rejects every useful quantization ever shipped, and it is
not even the bar the SHIPPING default meets -- the k-block summation order of an nvfp4 GEMV is an
implementation detail, so 'the completion sha256 changed' and 'it diverged at generated token 7'
are statements about float addition being non-associative, not about quality. A greedy decode is a
chaotic map: one near-tie resolved the other way rewrites every later token. So the harness scores
WHERE and WHY the arms parted, not WHETHER.

A flip is ACCEPTED only if all of G0..G3 hold. Any G failing is a REJECT. A broken reference is
VOID (measure nothing, fix the reference).

G0 REFERENCE SANITY. Every CONTROL prompt on the REFERENCE arm ends its turn on a stop token from
   generation_config.json. A reference that runs to max-steps is a runaway trajectory and every
   agreement ratio scored against it compares two runaways. (This is the 2026-08-06 failure:
   3/33 free-running was scored against a 96-token repetition loop at median margin 11.354.)

G1 CONTROL ANSWER PRESERVATION. On every CONTROL prompt the candidate ends its turn AND its
   normalized answer text equals the reference's. Controls are low-entropy single-fact prompts
   with one right answer, so text equality is the quality question stated directly. Token-level
   exact match and the first-divergence INDEX are both reported, but neither is a gate: identical
   text through a different tokenization is a pass, and divergence at index 7 with identical text
   is a pass.

G2 DIVERGENCE CHARACTER. On every CONTROL prompt, any first divergence is a near-tie
   (top-2 margin at that step < 0.25x the reference's median margin) or the arms resynchronise
   within 4 tokens. A DECISIVE divergence that then cascades on a control is a REJECT: the
   reference was sure and the candidate disagreed anyway.

G3 DISTRIBUTIONAL, measured under FORCED CONTEXT on the reference token sequence so that both
   arms are scored on the same conditioning and no cascade is involved:
   G3a forced top-1 agreement >= 98% on the worst CONTROL prompt.
   G3b margin-normalised perturbation rho = (m_ref - g_cand) / m_ref, where m_ref is the
       reference's top-2 margin and g_cand is the candidate's gap between the reference argmax and
       the candidate's best competitor. rho >= 1 means the candidate inverted that decision;
       rho >= 0.5 means the perturbation ate half the decision margin. Bar: fewer than 1% of steps
       at rho >= 0.5. This is the only threshold that needs no prior calibration, because it is
       measured in units of the model's own decisiveness.
   G3c mean KL(reference || candidate) <= 1e-3 nats and max per-step KL <= 2e-2 nats. 1e-3 nats
       per token is a 0.10% relative perplexity change (exp(1e-3) = 1.0010) -- roughly two orders
       of magnitude below the 0.1-1% perplexity gaps that separate quantization recipes people
       ship, so it is a bar on being INDISTINGUISHABLE, not merely on being acceptable.
   G3d the reference argmax never falls below rank 1 (0-indexed) in the candidate distribution,
       i.e. it stays top-2 at every forced step.

BIT-IDENTICAL SHORTCUT. If every retained logit word matches bit for bit, the verdict is
ACCEPT-BIT-IDENTICAL and G1..G3 are trivially satisfied. That is a strictly stronger result than
passing the bar and it is reported as its own verdict so it is never confused with one.

OPEN-ENDED PROMPTS ARE DESCRIPTIVE ONLY and gate nothing: three runs of a byte-identical config on
one low-margin open-ended prompt measured 66 / 78 / 96 tokens on this class of box, so an
open-ended trajectory difference cannot carry a claim in either direction.";

pub const RHO_SOFT: f32 = 0.5;
pub const RHO_INVERTED: f32 = 1.0;

pub fn normalize_answer(s: &str) -> String {
    let mut out = String::new();
    let mut pending_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            pending_space = !out.is_empty();
        } else if c.is_alphanumeric() {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            for l in c.to_lowercase() {
                out.push(l);
            }
        } else {
            pending_space = pending_space || !out.is_empty();
        }
    }
    out
}

fn log_sum_exp(l: &[f32]) -> f64 {
    let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    if !m.is_finite() {
        return m;
    }
    m + l.iter().map(|v| (*v as f64 - m).exp()).sum::<f64>().ln()
}

#[derive(Clone, Copy, Debug)]
pub struct StepDelta {
    pub reference_top1: u32,
    pub candidate_top1: u32,
    pub reference_margin: f32,
    pub kl_nats: f64,
    pub max_abs_logit_delta: f32,
    pub rho: f32,
    pub rank_of_reference_top1: usize,
    pub bit_identical: bool,
}

pub fn step_delta(reference: &[f32], candidate: &[f32]) -> StepDelta {
    assert_eq!(
        reference.len(),
        candidate.len(),
        "step_delta needs same-width logit rows"
    );
    let (r1, rv1, _, rv2) = top2(reference);
    let (c1, _, _, _) = top2(candidate);
    let cv1 = candidate[r1 as usize];
    let mut best_other = f32::NEG_INFINITY;
    let mut worse = 0usize;
    let mut max_abs = 0f32;
    for (i, (a, b)) in reference.iter().zip(candidate.iter()).enumerate() {
        max_abs = max_abs.max((a - b).abs());
        if i as u32 != r1 {
            best_other = best_other.max(*b);
        }
        if *b > cv1 {
            worse += 1;
        }
    }
    let m = rv1 - rv2;
    let g = cv1 - best_other;
    let rho = if m > 0.0 { (m - g) / m } else { 0.0 };

    let lr = log_sum_exp(reference);
    let lc = log_sum_exp(candidate);
    let mut kl = 0f64;
    for (a, b) in reference.iter().zip(candidate.iter()) {
        let lpa = *a as f64 - lr;
        let p = lpa.exp();
        if p > 0.0 {
            kl += p * (lpa - (*b as f64 - lc));
        }
    }

    StepDelta {
        reference_top1: r1,
        candidate_top1: c1,
        reference_margin: m,
        kl_nats: kl.max(0.0),
        max_abs_logit_delta: max_abs,
        rho,
        rank_of_reference_top1: worse,
        bit_identical: reference
            .iter()
            .zip(candidate.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
    }
}

#[derive(Clone, Debug)]
pub struct DistributionalSummary {
    pub prompt_label: String,
    pub prompt_kind: PromptKind,
    pub steps: usize,
    pub top1_agree: usize,
    pub first_top1_divergence: Option<usize>,
    pub mean_kl: f64,
    pub max_kl: f64,
    pub mean_abs_logit_delta: f64,
    pub max_abs_logit_delta: f32,
    pub mean_rho: f64,
    pub max_rho: f32,
    pub steps_rho_soft: usize,
    pub steps_rho_inverted: usize,
    pub worst_rank: usize,
    pub bit_identical: bool,
}

impl DistributionalSummary {
    pub fn new(prompt_label: &str, prompt_kind: PromptKind) -> Self {
        Self {
            prompt_label: prompt_label.to_string(),
            prompt_kind,
            steps: 0,
            top1_agree: 0,
            first_top1_divergence: None,
            mean_kl: 0.0,
            max_kl: 0.0,
            mean_abs_logit_delta: 0.0,
            max_abs_logit_delta: 0.0,
            mean_rho: 0.0,
            max_rho: f32::NEG_INFINITY,
            steps_rho_soft: 0,
            steps_rho_inverted: 0,
            worst_rank: 0,
            bit_identical: true,
        }
    }

    pub fn push(&mut self, d: StepDelta) {
        let i = self.steps;
        self.steps += 1;
        if d.reference_top1 == d.candidate_top1 {
            self.top1_agree += 1;
        } else if self.first_top1_divergence.is_none() {
            self.first_top1_divergence = Some(i);
        }
        self.mean_kl += (d.kl_nats - self.mean_kl) / self.steps as f64;
        self.max_kl = self.max_kl.max(d.kl_nats);
        let mad = d.max_abs_logit_delta as f64;
        self.mean_abs_logit_delta += (mad - self.mean_abs_logit_delta) / self.steps as f64;
        self.max_abs_logit_delta = self.max_abs_logit_delta.max(d.max_abs_logit_delta);
        self.mean_rho += (d.rho as f64 - self.mean_rho) / self.steps as f64;
        self.max_rho = self.max_rho.max(d.rho);
        if d.rho >= RHO_SOFT {
            self.steps_rho_soft += 1;
        }
        if d.rho >= RHO_INVERTED {
            self.steps_rho_inverted += 1;
        }
        self.worst_rank = self.worst_rank.max(d.rank_of_reference_top1);
        self.bit_identical = self.bit_identical && d.bit_identical;
    }

    pub fn top1_rate(&self) -> f64 {
        if self.steps == 0 {
            1.0
        } else {
            self.top1_agree as f64 / self.steps as f64
        }
    }

    pub fn rho_soft_rate(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.steps_rho_soft as f64 / self.steps as f64
        }
    }
}

impl fmt::Display for DistributionalSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<24} [{}] forced {}/{} steps top-1 ({:.2}%) first-div {:?}  KL mean {:.3e} max \
             {:.3e} nats  |dlogit| mean {:.3e} max {:.3e}  rho mean {:.3} max {:.3} \
             (>=0.5 on {} steps = {:.2}%, inverted on {})  worst rank {}{}",
            self.prompt_label,
            self.prompt_kind,
            self.top1_agree,
            self.steps,
            100.0 * self.top1_rate(),
            self.first_top1_divergence,
            self.mean_kl,
            self.max_kl,
            self.mean_abs_logit_delta,
            self.max_abs_logit_delta,
            self.mean_rho,
            self.max_rho,
            self.steps_rho_soft,
            100.0 * self.rho_soft_rate(),
            self.steps_rho_inverted,
            self.worst_rank,
            if self.bit_identical {
                "  BIT-IDENTICAL"
            } else {
                ""
            }
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FlipBar {
    pub min_forced_top1: f64,
    pub max_rho_soft_rate: f64,
    pub max_mean_kl: f64,
    pub max_step_kl: f64,
    pub max_worst_rank: usize,
    pub require_control_answer_match: bool,
    pub require_control_termination: bool,
}

impl Default for FlipBar {
    fn default() -> Self {
        Self {
            min_forced_top1: 0.98,
            max_rho_soft_rate: 0.01,
            max_mean_kl: 1e-3,
            max_step_kl: 2e-2,
            max_worst_rank: 1,
            require_control_answer_match: true,
            require_control_termination: true,
        }
    }
}

impl FlipBar {
    pub fn from_env() -> Self {
        let f = |k: &str, d: f64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(d)
        };
        let d = Self::default();
        Self {
            min_forced_top1: f("NV_FLIP_MIN_TOP1", d.min_forced_top1),
            max_rho_soft_rate: f("NV_FLIP_MAX_RHO_RATE", d.max_rho_soft_rate),
            max_mean_kl: f("NV_FLIP_MAX_MEAN_KL", d.max_mean_kl),
            max_step_kl: f("NV_FLIP_MAX_STEP_KL", d.max_step_kl),
            max_worst_rank: f("NV_FLIP_MAX_RANK", d.max_worst_rank as f64) as usize,
            ..d
        }
    }
}

impl fmt::Display for FlipBar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BAR: G1 control answer text preserved={} and control terminates={}; G2 no decisive \
             cascading control divergence; G3a forced top-1 >= {:.1}% on the worst control; \
             G3b < {:.1}% of steps at rho >= {RHO_SOFT}; G3c mean KL <= {:.1e} and max step KL \
             <= {:.1e} nats; G3d reference argmax rank <= {} in the candidate",
            self.require_control_answer_match,
            self.require_control_termination,
            100.0 * self.min_forced_top1,
            100.0 * self.max_rho_soft_rate,
            self.max_mean_kl,
            self.max_step_kl,
            self.max_worst_rank
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlipVerdict {
    AcceptBitIdentical,
    Accept,
    Reject,
    Void,
}

impl fmt::Display for FlipVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlipVerdict::AcceptBitIdentical => write!(f, "ACCEPT (BIT-IDENTICAL)"),
            FlipVerdict::Accept => write!(f, "ACCEPT"),
            FlipVerdict::Reject => write!(f, "REJECT"),
            FlipVerdict::Void => write!(f, "VOID (reference broken; nothing was measured)"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FlipDecision {
    pub flip: String,
    pub model: String,
    pub bar: FlipBar,
    pub verdict: FlipVerdict,
    pub failures: Vec<String>,
    pub evidence: Vec<String>,
    pub descriptive: Vec<String>,
}

impl FlipDecision {
    pub fn accepted(&self) -> bool {
        matches!(
            self.verdict,
            FlipVerdict::Accept | FlipVerdict::AcceptBitIdentical
        )
    }
}

impl fmt::Display for FlipDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "FLIP-VERDICT {} :: {} :: {}",
            self.flip, self.model, self.verdict
        )?;
        writeln!(f, "  {}", self.bar)?;
        for e in &self.evidence {
            writeln!(f, "  PASS  {e}")?;
        }
        for e in &self.failures {
            writeln!(f, "  FAIL  {e}")?;
        }
        for d in &self.descriptive {
            writeln!(f, "  note  {d}")?;
        }
        write!(
            f,
            "  ACCEPT/REJECT is on CONTROL rows only; open-ended rows above are descriptive."
        )
    }
}

pub fn decide_flip(
    flip: &str,
    model: &str,
    bar: FlipBar,
    suite: &SuiteReport,
    dists: &[DistributionalSummary],
) -> FlipDecision {
    let mut d = FlipDecision {
        flip: flip.to_string(),
        model: model.to_string(),
        bar,
        verdict: FlipVerdict::Accept,
        failures: Vec::new(),
        evidence: Vec::new(),
        descriptive: Vec::new(),
    };

    let controls = suite.controls();
    if controls.is_empty() {
        d.verdict = FlipVerdict::Void;
        d.failures
            .push("G0: no CONTROL row; a flip cannot be decided from open-ended prompts".into());
        return d;
    }

    let bad_ref: Vec<&&PairReport> = controls
        .iter()
        .filter(|r| r.reference_stop.0 != StopReason::HitStopToken)
        .collect();
    if !bad_ref.is_empty() {
        d.verdict = FlipVerdict::Void;
        d.failures.push(format!(
            "G0: reference arm {:?} ran to max-steps on {} control prompt(s) ({}). Fix the \
             reference before measuring the flip; ratios against a runaway are meaningless.",
            suite.reference_arm,
            bad_ref.len(),
            bad_ref
                .iter()
                .map(|r| r.prompt_label.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        return d;
    }
    d.evidence.push(format!(
        "G0: reference arm {:?} ended its turn on all {} control prompt(s)",
        suite.reference_arm,
        controls.len()
    ));

    let control_dists: Vec<&DistributionalSummary> = dists
        .iter()
        .filter(|s| s.prompt_kind == PromptKind::Control && s.steps > 0)
        .collect();
    let all_bit_identical = !control_dists.is_empty()
        && control_dists.iter().all(|s| s.bit_identical)
        && controls
            .iter()
            .all(|r| r.shape == DivergenceShape::Identical);

    let mut token_exact = 0usize;
    let mut divergences: Vec<String> = Vec::new();
    for c in &controls {
        if c.shape == DivergenceShape::Identical {
            token_exact += 1;
        }
        if let Some(i) = c.first_divergence {
            divergences.push(format!(
                "{}@{i}({:?}, ratio {})",
                c.prompt_label,
                c.shape,
                c.margin_ratio()
                    .map(|r| format!("{r:.3}"))
                    .unwrap_or_else(|| "n/a".into())
            ));
        }
    }
    d.evidence.push(format!(
        "greedy token-exact on {token_exact}/{} controls; first-divergence index per diverging \
         control: [{}]",
        controls.len(),
        divergences.join(", ")
    ));

    if bar.require_control_termination {
        let nonterm: Vec<&&PairReport> = controls
            .iter()
            .filter(|r| r.candidate_stop.0 != StopReason::HitStopToken)
            .collect();
        if nonterm.is_empty() {
            d.evidence
                .push("G1a: candidate ended its turn on every control".into());
        } else {
            d.failures.push(format!(
                "G1a: candidate ran to max-steps on {} control(s): {}",
                nonterm.len(),
                nonterm
                    .iter()
                    .map(|r| r.prompt_label.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }

    if bar.require_control_answer_match {
        let mut mismatched: Vec<String> = Vec::new();
        for c in &controls {
            let a = normalize_answer(&c.reference_text);
            let b = normalize_answer(&c.candidate_text);
            if a != b {
                mismatched.push(format!("{}: {a:?} -> {b:?}", c.prompt_label));
            }
        }
        if mismatched.is_empty() {
            d.evidence.push(format!(
                "G1b: normalized answer text identical on all {} controls",
                controls.len()
            ));
        } else {
            d.failures.push(format!(
                "G1b: {} control(s) changed answer: {}",
                mismatched.len(),
                mismatched.join(" | ")
            ));
        }
    }

    let decisive_cascades: Vec<&&PairReport> = controls
        .iter()
        .filter(|r| {
            r.first_divergence.is_some()
                && r.shape == DivergenceShape::Cascaded
                && r.margin_ratio()
                    .map(|x| x >= NEAR_TIE_RATIO)
                    .unwrap_or(true)
        })
        .collect();
    if decisive_cascades.is_empty() {
        d.evidence
            .push("G2: every control divergence was a near-tie or the arms resynchronised".into());
    } else {
        d.failures.push(format!(
            "G2: {} control(s) diverged DECISIVELY and never resynchronised: {}",
            decisive_cascades.len(),
            decisive_cascades
                .iter()
                .map(|r| format!(
                    "{}@{:?} ratio {}",
                    r.prompt_label,
                    r.first_divergence,
                    r.margin_ratio()
                        .map(|x| format!("{x:.3}"))
                        .unwrap_or_else(|| "n/a".into())
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    if control_dists.is_empty() {
        d.failures.push(
            "G3: no forced-context distributional row on any control; the distributional bar was \
             not measured, so the flip cannot be accepted"
                .into(),
        );
    } else {
        let worst_top1 = control_dists
            .iter()
            .min_by(|a, b| a.top1_rate().partial_cmp(&b.top1_rate()).unwrap())
            .unwrap();
        if worst_top1.top1_rate() >= bar.min_forced_top1 {
            d.evidence.push(format!(
                "G3a: worst-control forced top-1 {:.2}% ({} {}/{}) >= {:.2}%",
                100.0 * worst_top1.top1_rate(),
                worst_top1.prompt_label,
                worst_top1.top1_agree,
                worst_top1.steps,
                100.0 * bar.min_forced_top1
            ));
        } else {
            d.failures.push(format!(
                "G3a: worst-control forced top-1 {:.2}% ({} {}/{}) < {:.2}%",
                100.0 * worst_top1.top1_rate(),
                worst_top1.prompt_label,
                worst_top1.top1_agree,
                worst_top1.steps,
                100.0 * bar.min_forced_top1
            ));
        }

        let soft_steps: usize = control_dists.iter().map(|s| s.steps_rho_soft).sum();
        let total_steps: usize = control_dists.iter().map(|s| s.steps).sum();
        let rate = soft_steps as f64 / total_steps.max(1) as f64;
        let inverted: usize = control_dists.iter().map(|s| s.steps_rho_inverted).sum();
        let max_rho = control_dists
            .iter()
            .fold(f32::NEG_INFINITY, |m, s| m.max(s.max_rho));
        if rate <= bar.max_rho_soft_rate {
            d.evidence.push(format!(
                "G3b: rho >= {RHO_SOFT} on {soft_steps}/{total_steps} control steps ({:.3}% <= \
                 {:.3}%), inverted on {inverted}, max rho {max_rho:.3}",
                100.0 * rate,
                100.0 * bar.max_rho_soft_rate
            ));
        } else {
            d.failures.push(format!(
                "G3b: rho >= {RHO_SOFT} on {soft_steps}/{total_steps} control steps ({:.3}% > \
                     {:.3}%), inverted on {inverted}, max rho {max_rho:.3}. The perturbation is \
                     eating the model's own decision margin.",
                100.0 * rate,
                100.0 * bar.max_rho_soft_rate
            ));
        }

        let worst_mean_kl = control_dists.iter().fold(0f64, |m, s| m.max(s.mean_kl));
        let worst_max_kl = control_dists.iter().fold(0f64, |m, s| m.max(s.max_kl));
        if worst_mean_kl <= bar.max_mean_kl && worst_max_kl <= bar.max_step_kl {
            d.evidence.push(format!(
                "G3c: worst-control mean KL {worst_mean_kl:.3e} <= {:.1e}, max step KL \
                 {worst_max_kl:.3e} <= {:.1e} nats (relative perplexity change {:.4}%)",
                bar.max_mean_kl,
                bar.max_step_kl,
                100.0 * (worst_mean_kl.exp() - 1.0)
            ));
        } else {
            d.failures.push(format!(
                "G3c: worst-control mean KL {worst_mean_kl:.3e} (bar {:.1e}), max step KL \
                     {worst_max_kl:.3e} (bar {:.1e}) nats; relative perplexity change {:.4}%",
                bar.max_mean_kl,
                bar.max_step_kl,
                100.0 * (worst_mean_kl.exp() - 1.0)
            ));
        }

        let worst_rank = control_dists
            .iter()
            .map(|s| s.worst_rank)
            .max()
            .unwrap_or(0);
        if worst_rank <= bar.max_worst_rank {
            d.evidence.push(format!(
                "G3d: reference argmax stayed at rank <= {worst_rank} in the candidate"
            ));
        } else {
            d.failures.push(format!(
                "G3d: reference argmax fell to rank {worst_rank} (bar {}) in the candidate",
                bar.max_worst_rank
            ));
        }
    }

    for r in suite
        .rows
        .iter()
        .filter(|r| r.prompt_kind == PromptKind::OpenEnded)
    {
        d.descriptive.push(format!(
            "open-ended {:?}: agreement {}/{}, first divergence {:?}, shape {} -- DESCRIPTIVE \
             ONLY, gates nothing",
            r.prompt_label, r.agree, r.total, r.first_divergence, r.shape
        ));
    }

    d.verdict = if !d.failures.is_empty() {
        FlipVerdict::Reject
    } else if all_bit_identical {
        FlipVerdict::AcceptBitIdentical
    } else {
        FlipVerdict::Accept
    };
    d
}

#[cfg(test)]
mod core_tests {
    use super::*;

    fn tp(label: &str, kind: PromptKind, ids: Vec<u32>) -> TemplatedPrompt {
        TemplatedPrompt::from_official_render(
            label,
            kind,
            "test/model",
            "snap",
            "digest",
            1,
            "<rendered>".into(),
            ids,
        )
    }

    fn logits_from(top: u32, margin: f32, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        v[top as usize] = margin;
        v
    }

    fn chain(next: fn(u32) -> (u32, f32)) -> impl FnMut(u32) -> anyhow::Result<Vec<f32>> {
        move |t| {
            let (nt, m) = next(t);
            Ok(logits_from(nt, m, 64))
        }
    }

    fn straight(t: u32) -> (u32, f32) {
        ((t + 1) % 40, 5.0)
    }

    fn tie_at_7(t: u32) -> (u32, f32) {
        if t == 6 {
            (7, 0.01)
        } else {
            ((t + 1) % 40, 5.0)
        }
    }

    #[test]
    fn free_running_stops_at_a_stop_token_and_records_where() {
        let p = tp("c", PromptKind::Control, vec![1, 2, 3]);
        let stops = StopSet {
            ids: vec![10],
            source: "test".into(),
        };
        let run = free_running("a", &p, &stops, 64, chain(straight)).unwrap();
        assert_eq!(run.reason, StopReason::HitStopToken);
        assert_eq!(run.stop_token, Some(10));
        assert_eq!(*run.tokens.last().unwrap(), 10);
        assert_eq!(run.tokens.len(), 7);
    }

    #[test]
    fn free_running_refuses_an_untemplated_prompt() {
        let p = TemplatedPrompt::raw_untemplated_not_serving_evidence("raw", "hi".into(), vec![1]);
        let stops = StopSet {
            ids: vec![10],
            source: "test".into(),
        };
        let err = free_running("a", &p, &stops, 8, chain(straight)).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("RAW-UNTEMPLATED"), "{msg}");
        assert!(msg.contains("chat template"), "{msg}");
    }

    #[test]
    fn a_near_tie_divergence_is_distinguishable_from_a_decisive_flip() {
        let p = tp("t", PromptKind::OpenEnded, vec![1]);
        let stops = StopSet {
            ids: vec![99],
            source: "test".into(),
        };
        let a = free_running("ref", &p, &stops, 12, chain(tie_at_7)).unwrap();
        let tie = a
            .margins
            .iter()
            .enumerate()
            .min_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            a.margins[tie] < 0.1,
            "fixture lost its near-tie: {:?}",
            a.margins
        );

        let mut b = a.clone();
        b.arm = "cand".into();
        b.tokens[tie] = 31;
        let near = compare(&p, &a, &b);
        assert_eq!(near.first_divergence, Some(tie));
        let ratio = near.margin_ratio().unwrap();
        assert!(ratio < 0.25, "near-tie ratio {ratio}");
        assert!(
            near.divergence_reads_as().contains("TIE-BREAK")
                || near.divergence_reads_as().contains("AMBIGUOUS"),
            "{}",
            near.divergence_reads_as()
        );

        let decisive = (0..a.tokens.len()).find(|i| a.margins[*i] > 1.0).unwrap();
        let mut c = a.clone();
        c.arm = "cand".into();
        c.tokens[decisive] = 33;
        let flip = compare(&p, &a, &c);
        assert_eq!(flip.first_divergence, Some(decisive));
        assert!(flip.margin_ratio().unwrap() >= 0.25);
        assert!(
            flip.divergence_reads_as().contains("FLIP"),
            "{}",
            flip.divergence_reads_as()
        );
    }

    #[test]
    fn suite_refuses_a_single_prompt_claim_and_a_claim_without_a_control() {
        let p = tp("t", PromptKind::OpenEnded, vec![1]);
        let stops = StopSet {
            ids: vec![99],
            source: "t".into(),
        };
        let a = free_running("ref", &p, &stops, 6, chain(straight)).unwrap();
        let mut s = SuiteReport::new("x", "ref", "cand");
        s.push(compare(&p, &a, &a));
        let e = format!("{}", s.validate().unwrap_err());
        assert!(e.contains("at least 2 prompts"), "{e}");
        s.push(compare(&p, &a, &a));
        let e = format!("{}", s.validate().unwrap_err());
        assert!(e.contains("CONTROL"), "{e}");
    }

    #[test]
    fn different_termination_is_a_first_class_signal() {
        let p = tp("c", PromptKind::Control, vec![1]);
        let stops = StopSet {
            ids: vec![10],
            source: "t".into(),
        };
        let a = free_running("ref", &p, &stops, 64, chain(straight)).unwrap();
        let mut b = a.clone();
        b.arm = "cand".into();
        b.reason = StopReason::ReachedMaxSteps;
        b.stop_token = None;
        let r = compare(&p, &a, &b);
        assert!(!r.same_termination);
        assert!(format!("{r}").contains("DIFFERENT (arms ended their turn in different places)"));
    }

    #[test]
    fn forced_context_carries_its_own_disclaimer() {
        let p = tp("t", PromptKind::OpenEnded, vec![1]);
        let f = forced_context("cand", &p, &[2, 3, 4], chain(straight)).unwrap();
        let (a, n) = f.upper_bound_agreement_is_not_serving_evidence();
        assert_eq!(n, 3);
        assert!(a <= n);
        assert!(ForcedContextRun::why_this_is_not_serving_evidence().contains("upper bound"));
    }

    #[test]
    fn every_divergence_is_scored_not_just_the_first() {
        let p = tp("t", PromptKind::OpenEnded, vec![1]);
        let stops = StopSet {
            ids: vec![99],
            source: "t".into(),
        };
        let a = free_running("ref", &p, &stops, 12, chain(tie_at_7)).unwrap();

        let mut all_ties = a.clone();
        all_ties.arm = "cand".into();
        let tie = a.margins.iter().position(|m| *m < 0.1).unwrap();
        all_ties.tokens[tie] = 31;
        all_ties.margins[tie] = a.margins[tie];
        let r = compare(&p, &a, &all_ties);
        assert_eq!(r.divergence_margins.len(), 1);
        assert!(
            r.all_divergences_are_near_ties(),
            "{:?}",
            r.divergence_margins
        );
        assert!(format!("{r}").contains("every disagreement was a near-tie"));
        assert!(format!("{r}").contains("only the FIRST is independent evidence"));

        let mut mixed = a.clone();
        mixed.arm = "cand".into();
        let decisive = (0..a.tokens.len()).find(|i| a.margins[*i] > 1.0).unwrap();
        mixed.tokens[tie] = 31;
        mixed.tokens[decisive] = 33;
        let r2 = compare(&p, &a, &mixed);
        assert_eq!(r2.divergence_margins.len(), 2);
        assert_eq!(r2.near_tie_divergences(), 1);
        assert!(!r2.all_divergences_are_near_ties());
        assert!(!format!("{r2}").contains("every disagreement was a near-tie"));
    }

    #[test]
    fn control_that_diverges_is_a_hard_error() {
        let p = tp("c", PromptKind::Control, vec![1]);
        let stops = StopSet {
            ids: vec![99],
            source: "t".into(),
        };
        let a = free_running("ref", &p, &stops, 8, chain(straight)).unwrap();
        let mut b = a.clone();
        b.tokens[3] = 63;
        let mut s = SuiteReport::new("x", "ref", "cand");
        s.push(compare(&p, &a, &b));
        let e = format!("{}", s.assert_controls_exact().unwrap_err());
        assert!(e.contains("must come out exact"), "{e}");
    }
}

#[cfg(test)]
mod flip_bar_tests {
    use super::*;

    fn tp(label: &str, kind: PromptKind) -> TemplatedPrompt {
        TemplatedPrompt::from_official_render(
            label,
            kind,
            "test/model",
            "snap",
            "digest",
            1,
            "<rendered>".into(),
            vec![1, 2, 3],
        )
    }

    fn run(arm: &str, label: &str, tokens: Vec<u32>, margins: Vec<f32>, text: &str) -> FreeRun {
        FreeRun {
            arm: arm.into(),
            prompt_label: label.into(),
            tokens,
            margins,
            reason: StopReason::HitStopToken,
            stop_token: Some(106),
            text: text.into(),
        }
    }

    fn logits(scores: &[(usize, f32)], n: usize) -> Vec<f32> {
        let mut v = vec![-8.0f32; n];
        for (i, s) in scores {
            v[*i] = *s;
        }
        v
    }

    fn clean_dist(label: &str) -> DistributionalSummary {
        let mut d = DistributionalSummary::new(label, PromptKind::Control);
        let r = logits(&[(5, 12.0), (6, 2.0)], 64);
        for _ in 0..64 {
            d.push(step_delta(&r, &logits(&[(5, 12.0001), (6, 2.0)], 64)));
        }
        d
    }

    #[test]
    fn identical_logits_give_zero_kl_zero_rho_and_read_as_bit_identical() {
        let r = logits(&[(5, 12.0), (6, 2.0), (7, 1.0)], 64);
        let d = step_delta(&r, &r);
        assert_eq!(d.kl_nats, 0.0);
        assert_eq!(d.rho, 0.0);
        assert_eq!(d.max_abs_logit_delta, 0.0);
        assert_eq!(d.rank_of_reference_top1, 0);
        assert!(d.bit_identical);
        assert_eq!(d.reference_top1, d.candidate_top1);
    }

    #[test]
    fn rho_is_measured_in_units_of_the_models_own_decisiveness() {
        let r = logits(&[(5, 12.0), (6, 2.0)], 64);
        assert_eq!(step_delta(&r, &r).reference_margin, 10.0);

        let half = step_delta(&r, &logits(&[(5, 12.0), (6, 7.0)], 64));
        assert!((half.rho - 0.5).abs() < 1e-5, "rho {}", half.rho);
        assert_eq!(half.candidate_top1, 5, "argmax has not moved yet");

        let tied = step_delta(&r, &logits(&[(5, 12.0), (6, 12.0)], 64));
        assert!((tied.rho - 1.0).abs() < 1e-5, "rho {}", tied.rho);

        let flipped = step_delta(&r, &logits(&[(5, 12.0), (6, 14.0)], 64));
        assert!(
            flipped.rho > 1.0,
            "an inverted decision must read rho > 1: {}",
            flipped.rho
        );
        assert_eq!(flipped.candidate_top1, 6);
        assert_eq!(flipped.rank_of_reference_top1, 1);

        let sharper = step_delta(&r, &logits(&[(5, 12.0), (6, 0.0)], 64));
        assert!(
            sharper.rho < 0.0,
            "a MORE decisive candidate must read rho < 0"
        );
    }

    #[test]
    fn kl_scales_with_the_perturbation_and_is_nonnegative() {
        let r = logits(&[(5, 12.0), (6, 2.0)], 64);
        let small = step_delta(&r, &logits(&[(5, 12.0), (6, 2.01)], 64)).kl_nats;
        let big = step_delta(&r, &logits(&[(5, 12.0), (6, 6.0)], 64)).kl_nats;
        assert!(small >= 0.0 && big >= 0.0);
        assert!(big > small * 10.0, "small {small:e} big {big:e}");
        assert!(
            small < 1e-3,
            "a 0.01 logit nudge should sit far under the mean-KL bar: {small:e}"
        );
    }

    #[test]
    fn normalize_answer_ignores_case_punctuation_and_spacing_but_not_content() {
        assert_eq!(normalize_answer("  Paris.\n"), normalize_answer("paris"));
        assert_eq!(normalize_answer("**4**"), normalize_answer("4"));
        assert_eq!(
            normalize_answer("A stitch in time saves nine!"),
            "a stitch in time saves nine"
        );
        assert_ne!(normalize_answer("Paris"), normalize_answer("Lyon"));
        assert_ne!(normalize_answer("4"), normalize_answer("5"));
    }

    #[test]
    fn a_near_tie_divergence_at_generated_token_7_is_accepted_when_the_answer_survives() {
        let p = tp("control-capital", PromptKind::Control);
        let p2 = tp("control-arithmetic", PromptKind::Control);
        let mut margins = vec![14.0f32; 12];
        margins[7] = 0.05;
        let a = run(
            "ref",
            &p.label,
            vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 106],
            margins.clone(),
            "Paris",
        );
        let mut cand = a.clone();
        cand.arm = "cand".into();
        cand.tokens[7] = 900;
        cand.text = "Paris.".into();
        let b = run(
            "ref",
            &p2.label,
            vec![30, 31, 106],
            vec![16.0, 15.0, 12.0],
            "4",
        );
        let mut b2 = b.clone();
        b2.arm = "cand".into();

        let mut suite = SuiteReport::new("f1", "ref", "cand");
        suite.push(compare(&p, &a, &cand));
        suite.push(compare(&p2, &b, &b2));
        assert_eq!(suite.rows[0].first_divergence, Some(7));

        let d = decide_flip(
            "F1",
            "test",
            FlipBar::default(),
            &suite,
            &[clean_dist(&p.label), clean_dist(&p2.label)],
        );
        eprintln!("{d}");
        assert_eq!(d.verdict, FlipVerdict::Accept, "{:?}", d.failures);
        assert!(d
            .evidence
            .iter()
            .any(|e| e.contains("first-divergence index")));
        assert!(
            d.evidence.iter().any(|e| e.contains("control-capital@7")),
            "the divergence INDEX must be reported, not a boolean: {:?}",
            d.evidence
        );
    }

    #[test]
    fn a_decisive_cascading_control_divergence_is_a_reject() {
        let p = tp("control-capital", PromptKind::Control);
        let p2 = tp("control-arithmetic", PromptKind::Control);
        let a = run(
            "ref",
            &p.label,
            vec![10, 11, 12, 13, 106],
            vec![14.0; 5],
            "Paris",
        );
        let mut cand = a.clone();
        cand.arm = "cand".into();
        cand.tokens = vec![10, 11, 40, 41, 106];
        cand.text = "Lyon".into();
        let b = run("ref", &p2.label, vec![30, 106], vec![16.0, 12.0], "4");
        let mut b2 = b.clone();
        b2.arm = "cand".into();

        let mut suite = SuiteReport::new("f1", "ref", "cand");
        suite.push(compare(&p, &a, &cand));
        suite.push(compare(&p2, &b, &b2));

        let d = decide_flip(
            "F1",
            "test",
            FlipBar::default(),
            &suite,
            &[clean_dist(&p.label), clean_dist(&p2.label)],
        );
        eprintln!("{d}");
        assert_eq!(d.verdict, FlipVerdict::Reject);
        assert!(
            d.failures.iter().any(|e| e.starts_with("G1b")),
            "{:?}",
            d.failures
        );
        assert!(
            d.failures.iter().any(|e| e.starts_with("G2")),
            "{:?}",
            d.failures
        );
    }

    #[test]
    fn a_reference_that_never_ends_its_turn_voids_the_verdict_instead_of_rejecting() {
        let p = tp("control-capital", PromptKind::Control);
        let mut a = run(
            "ref",
            &p.label,
            (0..96).collect(),
            vec![11.354; 96],
            "Paris Paris Paris",
        );
        a.reason = StopReason::ReachedMaxSteps;
        a.stop_token = None;
        let mut cand = a.clone();
        cand.arm = "cand".into();
        let mut suite = SuiteReport::new("f1", "ref", "cand");
        suite.push(compare(&p, &a, &cand));
        let d = decide_flip(
            "F1",
            "test",
            FlipBar::default(),
            &suite,
            &[clean_dist(&p.label)],
        );
        eprintln!("{d}");
        assert_eq!(d.verdict, FlipVerdict::Void);
        assert!(d.failures[0].starts_with("G0"), "{:?}", d.failures);
    }

    #[test]
    fn a_bit_identical_flip_gets_its_own_stronger_verdict() {
        let p = tp("control-capital", PromptKind::Control);
        let p2 = tp("control-arithmetic", PromptKind::Control);
        let a = run("ref", &p.label, vec![10, 11, 106], vec![14.0; 3], "Paris");
        let mut ca = a.clone();
        ca.arm = "cand".into();
        let b = run("ref", &p2.label, vec![30, 106], vec![16.0, 12.0], "4");
        let mut cb = b.clone();
        cb.arm = "cand".into();
        let mut suite = SuiteReport::new("f1-qwen", "ref", "cand");
        suite.push(compare(&p, &a, &ca));
        suite.push(compare(&p2, &b, &cb));

        let exact = |label: &str| {
            let mut d = DistributionalSummary::new(label, PromptKind::Control);
            let r = logits(&[(5, 12.0), (6, 2.0)], 64);
            for _ in 0..32 {
                d.push(step_delta(&r, &r));
            }
            d
        };
        let d = decide_flip(
            "F1",
            "qwen",
            FlipBar::default(),
            &suite,
            &[exact(&p.label), exact(&p2.label)],
        );
        eprintln!("{d}");
        assert_eq!(d.verdict, FlipVerdict::AcceptBitIdentical);
        assert!(d.accepted());
    }

    #[test]
    fn a_distributional_breach_rejects_even_when_every_control_token_matched() {
        let p = tp("control-capital", PromptKind::Control);
        let p2 = tp("control-arithmetic", PromptKind::Control);
        let a = run("ref", &p.label, vec![10, 11, 106], vec![14.0; 3], "Paris");
        let mut ca = a.clone();
        ca.arm = "cand".into();
        let b = run("ref", &p2.label, vec![30, 106], vec![16.0, 12.0], "4");
        let mut cb = b.clone();
        cb.arm = "cand".into();
        let mut suite = SuiteReport::new("f2", "ref", "cand");
        suite.push(compare(&p, &a, &ca));
        suite.push(compare(&p2, &b, &cb));

        let mut noisy = DistributionalSummary::new(&p.label, PromptKind::Control);
        let r = logits(&[(5, 12.0), (6, 2.0)], 64);
        for i in 0..64 {
            let c = if i % 3 == 0 {
                logits(&[(5, 12.0), (6, 9.0)], 64)
            } else {
                logits(&[(5, 12.0), (6, 2.0001)], 64)
            };
            noisy.push(step_delta(&r, &c));
        }
        assert_eq!(
            noisy.top1_agree, 64,
            "argmax never moved; only the margin was eaten"
        );

        let d = decide_flip(
            "F2",
            "test",
            FlipBar::default(),
            &suite,
            &[noisy, clean_dist(&p2.label)],
        );
        eprintln!("{d}");
        assert_eq!(d.verdict, FlipVerdict::Reject);
        assert!(
            d.failures.iter().any(|e| e.starts_with("G3b")),
            "the margin-normalised gate must fire where argmax agreement cannot: {:?}",
            d.failures
        );
    }

    #[test]
    fn missing_distributional_evidence_cannot_be_accepted_by_default() {
        let p = tp("control-capital", PromptKind::Control);
        let p2 = tp("control-arithmetic", PromptKind::Control);
        let a = run("ref", &p.label, vec![10, 11, 106], vec![14.0; 3], "Paris");
        let mut ca = a.clone();
        ca.arm = "cand".into();
        let b = run("ref", &p2.label, vec![30, 106], vec![16.0, 12.0], "4");
        let mut cb = b.clone();
        cb.arm = "cand".into();
        let mut suite = SuiteReport::new("f2", "ref", "cand");
        suite.push(compare(&p, &a, &ca));
        suite.push(compare(&p2, &b, &cb));
        let d = decide_flip("F2", "test", FlipBar::default(), &suite, &[]);
        eprintln!("{d}");
        assert_eq!(d.verdict, FlipVerdict::Reject);
        assert!(
            d.failures.iter().any(|e| e.starts_with("G3")),
            "{:?}",
            d.failures
        );
    }

    #[test]
    fn open_ended_rows_are_descriptive_and_never_flip_the_verdict() {
        let c = tp("control-capital", PromptKind::Control);
        let c2 = tp("control-arithmetic", PromptKind::Control);
        let o = tp("openended-explain", PromptKind::OpenEnded);
        let a = run("ref", &c.label, vec![10, 11, 106], vec![14.0; 3], "Paris");
        let mut ca = a.clone();
        ca.arm = "cand".into();
        let b = run("ref", &c2.label, vec![30, 106], vec![16.0, 12.0], "4");
        let mut cb = b.clone();
        cb.arm = "cand".into();
        let oa = run(
            "ref",
            &o.label,
            (0..40).collect(),
            vec![0.4; 40],
            "one story",
        );
        let mut ob = oa.clone();
        ob.arm = "cand".into();
        ob.tokens = (100..140).collect();
        ob.text = "a completely different story".into();

        let mut suite = SuiteReport::new("f1", "ref", "cand");
        suite.push(compare(&c, &a, &ca));
        suite.push(compare(&c2, &b, &cb));
        suite.push(compare(&o, &oa, &ob));

        let mut open_dist = DistributionalSummary::new(&o.label, PromptKind::OpenEnded);
        let r = logits(&[(5, 12.0), (6, 2.0)], 64);
        for _ in 0..40 {
            open_dist.push(step_delta(&r, &logits(&[(5, 12.0), (6, 14.0)], 64)));
        }
        assert_eq!(open_dist.top1_agree, 0, "fixture must be maximally bad");

        let d = decide_flip(
            "F1",
            "test",
            FlipBar::default(),
            &suite,
            &[clean_dist(&c.label), clean_dist(&c2.label), open_dist],
        );
        eprintln!("{d}");
        assert_eq!(
            d.verdict,
            FlipVerdict::Accept,
            "an open-ended row must not decide a flip: {:?}",
            d.failures
        );
        assert!(d
            .descriptive
            .iter()
            .any(|s| s.contains("openended-explain")));
    }

    #[test]
    fn the_bar_is_written_down_and_says_why_bit_identical_is_the_wrong_bar() {
        eprintln!("{}", FLIP_BAR_DOC);
        eprintln!("{}", FlipBar::default());
        assert!(FLIP_BAR_DOC.contains("Why not bit-identical"));
        assert!(FLIP_BAR_DOC.contains("token 7"));
        for g in ["G0", "G1", "G2", "G3a", "G3b", "G3c", "G3d"] {
            assert!(FLIP_BAR_DOC.contains(g), "bar text lost {g}");
        }
    }
}

#[cfg(test)]
mod harness_inventory {
    pub const WHY_THESE_ROWS_ARE_LABELLED_NOT_DELETED: &str =
        "This file imports no speaches_plus and no nv_ symbol. Its rows exercise the scoring, \
         KL/rho and verdict logic defined a few hundred lines above them, and nothing else -- \
         they are the eval harness proving itself, which is worth having and worth keeping. What \
         they are not is server coverage, and every one of them is compiled into BOTH chat_eval \
         and laguna_serve_spec, where they were the majority of the reported passes in each. A \
         reader counting green rows had no way to see that, so the outer module they arrive \
         through is named for what they cover rather than for what they are about, and this row \
         prints the split. Renaming is the fix; deleting them would throw away the only thing \
         that checks the scorer.";

    fn test_attrs_in(basename: &str) -> Option<usize> {
        let name = std::path::Path::new(basename).file_name()?;
        let mut dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        loop {
            for sub in ["tests", "tests/common"] {
                let p = dir.join(sub).join(name);
                if let Ok(src) = std::fs::read_to_string(&p) {
                    return Some(src.lines().filter(|l| l.trim() == "#[test]").count());
                }
            }
            dir = dir.parent()?;
        }
    }

    #[test]
    fn the_rows_that_cover_no_server_code_are_counted_and_named_in_the_run_output() {
        let mut path = module_path!().split("::");
        let binary = path.next().unwrap_or_default();

        let segments: Vec<&str> = module_path!().split("::").collect();
        let arrives_through = segments
            .iter()
            .find(|s| **s == "harness_self_test_no_server_code")
            .copied()
            .or_else(|| path.next());
        let self_tests = test_attrs_in(file!())
            .unwrap_or_else(|| panic!("cannot read this harness's own source at {}", file!()));
        let host = test_attrs_in(&format!("{binary}.rs")).unwrap_or_else(|| {
            panic!("cannot read the including harness's source tests/{binary}.rs")
        });
        let total = self_tests + host;
        eprintln!(
            "[{binary}] {self_tests} of {total} rows are harness self-tests covering no server \
             code; {host} row(s) exercise {binary} itself. The self-tests report under \
             harness_self_test_no_server_code::."
        );
        assert!(
            self_tests > 0 && host > 0,
            "{WHY_THESE_ROWS_ARE_LABELLED_NOT_DELETED}\n\ntests/{binary}.rs reports {host} rows \
             of its own against {self_tests} harness self-tests. A binary with zero rows of its \
             own is entirely harness self-test and must not be read as covering {binary}."
        );
        assert_eq!(
            arrives_through,
            Some("harness_self_test_no_server_code"),
            "{WHY_THESE_ROWS_ARE_LABELLED_NOT_DELETED}\n\nthese rows arrive through a module \
             named {arrives_through:?}, so the run output no longer says they cover no server \
             code. Declare this file as `mod harness_self_test_no_server_code;` in \
             tests/{binary}.rs."
        );
    }
}
