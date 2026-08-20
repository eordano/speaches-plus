use super::*;

#[cfg(feature = "cuda")]
pub(crate) fn det_hash_f32(v: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

#[cfg(feature = "cuda")]
pub(crate) struct DetHashHook;

#[cfg(feature = "cuda")]
impl nv_models::gemma4::Gemma4LayerHook for DetHashHook {
    fn after_layer(
        &mut self,
        layer_idx: usize,
        hidden: &candle_core::Tensor,
    ) -> anyhow::Result<()> {
        let v = hidden
            .flatten_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()?;
        eprintln!(
            "[NV_DEBUG_DETERMINISM] layer={layer_idx} n={} hash={:016x}",
            v.len(),
            det_hash_f32(&v)
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn last_row_logits_3d(logits: &candle_core::Tensor) -> anyhow::Result<Vec<f32>> {
    let dims = logits.dims();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1,
        "expected logits shape [1, seq, vocab], got {:?}",
        dims
    );
    let last = logits.i((0usize, dims[1] - 1, ..))?;
    Ok(last.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?)
}

pub(crate) fn sampling_params_from(
    req: &ChatGenerateRequest,
) -> nv_layers::sampler::SamplingParams {
    nv_layers::sampler::SamplingParams {
        temperature: req.temperature.unwrap_or(0.0).max(0.0),
        top_k: req.top_k.map(|k| k as usize),
        top_p: req.top_p,
        min_p: req.min_p,
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
    }
}

pub(crate) struct SampleOutput {
    pub(crate) token: u32,

    pub(crate) logprob: Option<f32>,

    pub(crate) top: Vec<(u32, f32)>,

    pub(crate) exhausted: bool,
}

pub(crate) struct ChatSampler {
    pub(crate) params: nv_layers::sampler::SamplingParams,
    pub(crate) rng: Pcg64,
    pub(crate) counts: std::collections::HashMap<u32, u32>,

    pub(crate) prompt_tokens: Vec<u32>,

    pub(crate) guided: Option<GuidedRun>,

    pub(crate) logit_bias: Vec<(u32, f32)>,
    pub(crate) logprobs: bool,
    pub(crate) top_logprobs: usize,

    scratch: Vec<f32>,
}

impl ChatSampler {
    pub(crate) fn new(
        params: nv_layers::sampler::SamplingParams,
        seed: u64,
        guided: Option<GuidedRun>,
        logit_bias: Vec<(u32, f32)>,
        logprobs: bool,
        top_logprobs: usize,
    ) -> Self {
        Self {
            params,
            rng: Pcg64::seed_from_u64(seed),
            counts: std::collections::HashMap::new(),
            prompt_tokens: Vec::new(),
            guided,
            logit_bias,
            logprobs,
            top_logprobs,
            scratch: Vec::new(),
        }
    }

    #[cfg(any(test, feature = "cuda"))]
    pub(crate) fn for_request(
        req: &ChatGenerateRequest,
        tokenizer: &tokenizers::Tokenizer,
        eos_ids: &[u32],
        max_new: usize,
    ) -> anyhow::Result<Self> {
        let guided = match &req.guided {
            Some(spec) => Some(GuidedRun::new(
                build_guided_for_request(
                    tokenizer,
                    eos_ids,
                    spec,
                    req.guided_think_close.as_deref(),
                )?,
                max_new,
            )),
            None => None,
        };
        Ok(Self::new(
            sampling_params_from(req),
            req.seed.unwrap_or_else(os_random_u64),
            guided,
            req.logit_bias.clone(),
            req.logprobs,
            req.top_logprobs,
        ))
    }

    pub(crate) fn seed_prompt(&mut self, prompt_ids: &[u32]) {
        if (self.params.repetition_penalty - 1.0).abs() <= f32::EPSILON {
            return;
        }
        let mut ids = prompt_ids.to_vec();
        ids.sort_unstable();
        ids.dedup();
        self.prompt_tokens = ids;
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn fast_greedy(&self) -> bool {
        self.params.is_greedy()
            && !self.params.has_penalties()
            && self.guided.is_none()
            && self.logit_bias.is_empty()
            && !self.logprobs
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn record_token(&mut self, tok: u32) {
        *self.counts.entry(tok).or_insert(0) += 1;
    }

    pub(crate) fn uniform_f64(&mut self) -> f64 {
        let raw = self.rng.next_u64() >> 11;
        (raw as f64) / ((1u64 << 53) as f64)
    }

    pub(crate) fn uniform(&mut self) -> f32 {
        self.uniform_f64() as f32
    }

    pub(crate) fn record_logprobs(
        &self,
        logits: &[f32],
        chosen: u32,
        top: &mut Vec<(u32, f32)>,
    ) -> Option<f32> {
        use nv_layers::sampler;
        if !self.logprobs {
            return None;
        }

        let lps = sampler::logprobs_full(logits, 1.0);
        let chosen_lp = lps
            .get(chosen as usize)
            .copied()
            .unwrap_or(f32::NEG_INFINITY);
        if self.top_logprobs > 0 {
            for i in sampler::top_n_indices(&lps, self.top_logprobs) {
                top.push((i as u32, lps[i]));
            }
        }
        Some(chosen_lp)
    }

    pub(crate) fn sample(&mut self, logits: &[f32]) -> SampleOutput {
        use nv_layers::sampler;
        let need_copy =
            self.params.has_penalties() || self.guided.is_some() || !self.logit_bias.is_empty();
        let picked;
        let logprob;
        let mut top: Vec<(u32, f32)> = Vec::new();
        if !need_copy {
            picked = if self.params.is_greedy() {
                sampler::argmax_checked(logits)
            } else {
                let u = self.uniform();
                sampler::sample_token_checked(logits, &self.params, u)
            };
            logprob = picked.and_then(|t| self.record_logprobs(logits, t, &mut top));
        } else {
            let mut lg = std::mem::take(&mut self.scratch);
            lg.clear();
            lg.extend_from_slice(logits);
            for &(id, bias) in &self.logit_bias {
                if let Some(v) = lg.get_mut(id as usize) {
                    *v += bias;
                }
            }
            if self.params.has_penalties() {
                let seen: Vec<(u32, u32)> = self.counts.iter().map(|(&t, &c)| (t, c)).collect();
                sampler::apply_penalties_with_prompt(
                    &mut lg,
                    &seen,
                    &self.prompt_tokens,
                    &self.params,
                );
            }

            if let Some(g) = &mut self.guided {
                g.apply_mask(&mut lg);
            }

            let u = self.uniform();
            picked = sampler::sample_token_checked(&lg, &self.params, u);
            self.scratch = lg;

            logprob = picked.and_then(|t| self.record_logprobs(logits, t, &mut top));
        }
        let Some(tok) = picked else {
            return SampleOutput {
                token: 0,
                logprob: None,
                top,
                exhausted: true,
            };
        };
        if let Some(g) = &mut self.guided {
            if !g.advance(tok) {
                return SampleOutput {
                    token: tok,
                    logprob,
                    top,
                    exhausted: true,
                };
            }
        }
        *self.counts.entry(tok).or_insert(0) += 1;
        SampleOutput {
            token: tok,
            logprob,
            top,
            exhausted: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn scratch_len(&self) -> usize {
        self.scratch.len()
    }

    pub(crate) fn warped_dist(&self, raw: &[f32]) -> Vec<f32> {
        use nv_layers::sampler;
        if self.params.has_penalties() || !self.logit_bias.is_empty() {
            let mut lg = raw.to_vec();
            for &(id, bias) in &self.logit_bias {
                if let Some(v) = lg.get_mut(id as usize) {
                    *v += bias;
                }
            }
            if self.params.has_penalties() {
                let seen: Vec<(u32, u32)> = self.counts.iter().map(|(&t, &c)| (t, c)).collect();
                sampler::apply_penalties_with_prompt(
                    &mut lg,
                    &seen,
                    &self.prompt_tokens,
                    &self.params,
                );
            }
            sampler::distribution(&lg, &self.params)
        } else {
            sampler::distribution(raw, &self.params)
        }
    }

    pub(crate) fn draw_from_logits(&mut self, raw: &[f32]) -> u32 {
        let probs = self.warped_dist(raw);
        let u = self.uniform();
        nv_layers::sampler::sample_from(&probs, u)
    }

    pub(crate) fn accept_draft(&mut self, raw: &[f32], drafted: u32) -> DraftOutcome {
        let probs = self.warped_dist(raw);
        let px = probs.get(drafted as usize).copied().unwrap_or(0.0) as f64;

        if self.uniform_f64() < px {
            return DraftOutcome::Accept;
        }
        let u2 = self.uniform_f64();
        let repl = nv_layers::sampler::residual_sample_checked(&probs, drafted, u2)
            .unwrap_or_else(|| nv_layers::sampler::argmax(raw));
        DraftOutcome::Reject(repl)
    }

    pub(crate) fn commit(&mut self, tok: u32) {
        *self.counts.entry(tok).or_insert(0) += 1;
    }

    pub(crate) fn pure_greedy(&self) -> bool {
        self.params.is_greedy()
            && !self.params.has_penalties()
            && self.logit_bias.is_empty()
            && self.guided.is_none()
    }
}

pub(crate) enum DraftOutcome {
    Accept,
    Reject(u32),
}

pub(crate) fn guided_vocab_bytes(
    tokenizer: &tokenizers::Tokenizer,
    eos_ids: &[u32],
) -> Arc<nv_grammar::VocabBytes> {
    type Key = (u32, Vec<u32>);
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<Key, Arc<nv_grammar::VocabBytes>>>,
    > = std::sync::OnceLock::new();

    let n = tokenizer.get_vocab_size(true) as u32;
    let key: Key = (n, eos_ids.to_vec());
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Some(hit) = cache.lock().ok().and_then(|c| c.get(&key).cloned()) {
        return hit;
    }

    let bytes: Vec<Vec<u8>> = (0..n)
        .map(|id| {
            tokenizer
                .decode(&[id], false)
                .unwrap_or_default()
                .into_bytes()
        })
        .collect();
    let built = Arc::new(nv_grammar::VocabBytes::new(bytes, eos_ids));
    if let Ok(mut c) = cache.lock() {
        c.entry(key).or_insert_with(|| built.clone());
    }
    built
}

pub(crate) fn build_guided(
    tokenizer: &tokenizers::Tokenizer,
    eos_ids: &[u32],
    spec: &nv_grammar::GrammarSpec,
) -> anyhow::Result<nv_grammar::GuidedDecoder> {
    let vocab = guided_vocab_bytes(tokenizer, eos_ids);
    nv_grammar::GuidedDecoder::from_grammar(spec, vocab)
}

pub(crate) const GUIDED_THINK_CLOSE_MAX_TOKENS: usize = 8;
pub(crate) const GUIDED_THINK_FALLBACK_MARKER: &str = "</think>";
pub(crate) const GUIDED_THINK_FALLBACK_MAX_TOKENS: usize = 1;

pub(crate) fn think_close_tokens(
    tokenizer: &tokenizers::Tokenizer,
    marker: &str,
) -> Option<Vec<u32>> {
    if let Some(id) = tokenizer.token_to_id(marker) {
        return Some(vec![id]);
    }
    let max_ids = if marker == GUIDED_THINK_FALLBACK_MARKER {
        GUIDED_THINK_FALLBACK_MAX_TOKENS
    } else {
        GUIDED_THINK_CLOSE_MAX_TOKENS
    };
    let enc = tokenizer.encode(marker, false).ok()?;
    let ids = enc.get_ids();
    (1..=max_ids).contains(&ids.len()).then(|| ids.to_vec())
}

pub(crate) fn guided_think_close_marker(
    engine: &dyn ChatEngine,
    guided: bool,
    thinking_on: bool,
) -> Option<String> {
    if !guided || !thinking_on {
        return None;
    }
    engine
        .official_template()
        .and_then(|t| t.thinking_close_marker())
        .or_else(|| {
            engine
                .thinking_split_supported()
                .then(|| GUIDED_THINK_FALLBACK_MARKER.to_string())
        })
}

pub(crate) fn template_thinking_default(engine: &dyn ChatEngine) -> Option<bool> {
    let template = engine.official_template()?;
    template
        .effective_template_kwargs()
        .get("enable_thinking")
        .and_then(|v| v.as_bool())
        .or_else(|| {
            template
                .thinking_on_when_the_switch_is_undefined_scoped_to_reasoning_effort_templates_so_qwen36_guided_defaults_are_untouched()
        })
}

pub(crate) fn build_guided_for_request(
    tokenizer: &tokenizers::Tokenizer,
    eos_ids: &[u32],
    spec: &nv_grammar::GrammarSpec,
    think_close: Option<&str>,
) -> anyhow::Result<nv_grammar::GuidedDecoder> {
    let mut g = build_guided(tokenizer, eos_ids, spec)?;
    if let Some(marker) = think_close {
        let close = think_close_tokens(tokenizer, marker)
            .or_else(|| think_close_tokens(tokenizer, GUIDED_THINK_FALLBACK_MARKER))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "this model is in thinking mode and structured output was requested, but its \
                     own close marker {marker:?} does not tokenize to 1..=\
                     {GUIDED_THINK_CLOSE_MAX_TOKENS} ids and </think> is not a token of its \
                     vocabulary either, so there is no position at which to arm the grammar. \
                     Constraining from the first token instead would force the schema over the \
                     reasoning the template just primed. Retry with enable_thinking=false."
                )
            })?;
        g.set_defer_until_sequence(&close);
    }
    Ok(g)
}

pub(crate) const GUIDED_THINK_TAIL_RESERVE: usize = 256;

fn think_budget_for(max_new: usize) -> usize {
    match max_new.checked_sub(GUIDED_THINK_TAIL_RESERVE) {
        Some(n) if n > 0 => n,
        _ => max_new.div_ceil(2),
    }
}

pub(crate) struct GuidedRun {
    decoder: nv_grammar::GuidedDecoder,
    think_budget: usize,
    sampled: usize,
    unforceable_close: Option<u32>,
    killed_by: Option<u32>,
}

impl GuidedRun {
    pub(crate) fn new(decoder: nv_grammar::GuidedDecoder, max_new: usize) -> Self {
        Self {
            decoder,
            think_budget: think_budget_for(max_new),
            sampled: 0,
            unforceable_close: None,
            killed_by: None,
        }
    }

    pub(crate) fn apply_mask(&mut self, logits: &mut [f32]) {
        if self.decoder.is_dead() {
            self.leave_no_legal_token_after_death(logits);
            return;
        }
        if !self.decoder.deferred() {
            self.decoder.apply_mask(logits);
            return;
        }
        let outside = self.decoder.close_token_outside_logits(logits.len());
        if outside.is_none() {
            if self.sampled < self.think_budget {
                return;
            }
            if self.decoder.mask_to_defer_token(logits) {
                return;
            }
        }
        let close = outside.or_else(|| self.decoder.defer_token());
        self.leave_no_legal_token(close, logits);
    }

    fn leave_no_legal_token(&mut self, close: Option<u32>, logits: &mut [f32]) {
        if self.unforceable_close.is_none() {
            self.unforceable_close = close;
            tracing::error!(
                close_token = ?close,
                logits_vocab = logits.len(),
                think_budget = self.think_budget,
                sampled = self.sampled,
                "guided decoding can never arm: this token of the model's thinking close marker \
                 is outside its logits row, so the close can neither be sampled nor forced. \
                 Masking every candidate so the request fails loudly instead of returning \
                 unconstrained prose with finish_reason=stop to a caller who asked for a schema."
            );
        }
        logits.fill(f32::NEG_INFINITY);
    }

    fn leave_no_legal_token_after_death(&self, logits: &mut [f32]) {
        tracing::error!(
            killed_by = ?self.killed_by,
            sampled = self.sampled,
            "guided decoding is dead: a token left the grammar's DFA, so no continuation exists \
             and the schema can never be satisfied. Masking every candidate so the request fails \
             loudly instead of returning the rest of the answer unconstrained with \
             finish_reason=stop to a caller who asked for a schema."
        );
        logits.fill(f32::NEG_INFINITY);
    }

    #[cfg(test)]
    pub(crate) fn unforceable_close(&self) -> Option<u32> {
        self.unforceable_close
    }

    #[cfg(test)]
    pub(crate) fn killed_by(&self) -> Option<u32> {
        self.killed_by
    }

    #[must_use = "false means the grammar died on this token: it can never accept another, so a \
                  caller that drops it emits the rest of the completion with no schema enforced \
                  at all and still answers HTTP 200"]
    pub(crate) fn advance(&mut self, tok: u32) -> bool {
        self.sampled += 1;
        if self.decoder.advance(tok) {
            return true;
        }
        if self.killed_by.is_none() {
            self.killed_by = Some(tok);
        }
        false
    }

    pub(crate) fn decoder(&self) -> &nv_grammar::GuidedDecoder {
        &self.decoder
    }

    pub(crate) fn decoder_mut(&mut self) -> &mut nv_grammar::GuidedDecoder {
        &mut self.decoder
    }
}

pub(crate) fn sample_logits(
    logits: &[f32],
    temperature: f32,
    top_k: Option<u32>,
    top_p: Option<f32>,
    rng: &mut impl Rng,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if temperature <= 1e-6 {
        return argmax_u32(logits);
    }

    let inv_t = 1.0_f32 / temperature.max(1e-6);
    let mut scaled: Vec<f32> = logits.iter().map(|&x| x * inv_t).collect();

    if let Some(k) = top_k {
        let k = (k as usize).min(scaled.len());
        if k == 0 {
            return argmax_u32(logits);
        }
        if k < scaled.len() {
            let mut idx: Vec<usize> = (0..scaled.len()).collect();
            idx.sort_by(|&a, &b| {
                scaled[b]
                    .partial_cmp(&scaled[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let keep: std::collections::HashSet<usize> = idx.into_iter().take(k).collect();
            for (i, v) in scaled.iter_mut().enumerate() {
                if !keep.contains(&i) {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }

    let mut max = f32::NEG_INFINITY;
    for &v in &scaled {
        if v > max {
            max = v;
        }
    }
    if !max.is_finite() {
        return argmax_u32(logits);
    }
    let mut probs: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return argmax_u32(logits);
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    if let Some(p_thresh) = top_p {
        if p_thresh > 0.0 && p_thresh < 1.0 {
            let mut order: Vec<usize> = (0..probs.len()).collect();
            order.sort_by(|&a, &b| {
                probs[b]
                    .partial_cmp(&probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut cum = 0.0_f32;
            let mut keep = vec![false; probs.len()];
            for &i in &order {
                keep[i] = true;
                cum += probs[i];
                if cum >= p_thresh {
                    break;
                }
            }
            let mut renorm = 0.0_f32;
            for (i, p) in probs.iter_mut().enumerate() {
                if !keep[i] {
                    *p = 0.0;
                } else {
                    renorm += *p;
                }
            }
            if renorm <= 0.0 {
                return argmax_u32(logits);
            }
            for p in probs.iter_mut() {
                *p /= renorm;
            }
        }
    }

    let raw = rng.next_u64() >> 11;
    let u = (raw as f64) / ((1u64 << 53) as f64);
    let u = u as f32;
    let mut acc = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return i as u32;
        }
    }

    for i in (0..probs.len()).rev() {
        if probs[i] > 0.0 {
            return i as u32;
        }
    }
    0
}

pub(crate) fn argmax_u32(logits: &[f32]) -> u32 {
    let (id, _) = logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
            if v > acc.1 {
                (i, v)
            } else {
                acc
            }
        });
    id as u32
}

pub(crate) fn os_random_u64() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = std::process::id() as u64;
    let c = COUNTER.fetch_add(1, Ordering::Relaxed);

    let mut z = nanos
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(pid.wrapping_mul(0xBF58476D1CE4E5B9))
        .wrapping_add(c.wrapping_mul(0x94D049BB133111EB));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod scratch_tests {
    use super::*;

    fn bias_params() -> nv_layers::sampler::SamplingParams {
        nv_layers::sampler::SamplingParams {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            min_p: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
        }
    }

    #[test]
    fn scratch_is_reset_between_consecutive_samples() {
        let mut s = ChatSampler::new(bias_params(), 1, None, vec![(3, 0.0)], false, 0);

        assert_eq!(s.sample(&[0.0f32, 0.0, 10.0, 0.0]).token, 2);
        assert_eq!(
            s.sample(&[0.0f32, 5.0, 0.0, 0.0]).token,
            1,
            "the second step sampled from a scratch still holding step 1's logits"
        );
        assert_eq!(
            s.scratch_len(),
            4,
            "the scratch grew past one vocab row, so it was appended to rather \
             than cleared"
        );
    }

    #[test]
    fn scratch_survives_an_exhausted_step() {
        let mut s = ChatSampler::new(bias_params(), 1, None, vec![(0, 0.0)], false, 0);
        assert_eq!(s.sample(&[1.0f32, 0.0, 0.0, 0.0]).token, 0);
        let n = s.scratch_len();
        assert_eq!(n, 4);

        let out = s.sample(&[f32::NEG_INFINITY; 4]);
        assert!(out.exhausted, "all -inf must report exhausted");
        assert_eq!(
            s.scratch_len(),
            4,
            "the exhausted early return dropped the scratch buffer"
        );
    }
}

#[cfg(test)]
mod think_budget_tests {
    use super::think_close_tests::{boolean_spec, harmony_tokenizer, EOS};
    use super::*;

    const ILLEGAL_ARGMAX: u32 = 2;
    const CLOSE: u32 = 4;

    fn thinking_request(max_new_tokens: usize) -> ChatGenerateRequest {
        ChatGenerateRequest {
            prompt: String::new(),
            max_new_tokens,
            stop: Vec::new(),
            seed: Some(1),
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            guided: Some(boolean_spec()),
            guided_think_close: Some("<|end|>".into()),
            logit_bias: Vec::new(),
            logprobs: false,
            top_logprobs: 0,
            kv_resume: None,
            kv_store: None,
            mm: None,
        }
    }

    fn walk_a_runaway_thought(req: &ChatGenerateRequest, max_new: usize) {
        let tk = harmony_tokenizer();
        let mut s = ChatSampler::for_request(req, &tk, &[EOS], max_new)
            .expect("<|end|> is one token of this vocabulary");
        let logits = vec![0.0f32, 1.0, 5.0, 0.5, 3.0, 0.4, 0.3, 0.2];

        for i in 0..max_new.div_ceil(2) {
            assert_eq!(
                s.sample(&logits).token,
                ILLEGAL_ARGMAX,
                "step {i}: while the model is still inside its thinking budget the grammar must \
                 mask nothing, so the schema-illegal raw argmax still wins"
            );
        }
        assert_eq!(
            s.sample(&logits).token,
            CLOSE,
            "the thinking budget is spent, so the model's own close marker is the only legal \
             token: without the forced close the model thinks to max_new_tokens, the grammar \
             never arms, and a caller who asked for a schema silently receives prose"
        );
        assert!(
            !s.guided
                .as_ref()
                .expect("guided request")
                .decoder()
                .deferred(),
            "emitting the close marker arms the grammar"
        );
        assert_eq!(
            s.sample(&logits).token,
            1,
            "grammar now live: the illegal argmax is masked and 'false' wins"
        );
    }

    #[test]
    fn a_runaway_thought_is_closed_on_the_sampler_every_cuda_engine_shares() {
        walk_a_runaway_thought(&thinking_request(8), 8);
    }

    #[test]
    fn the_budget_is_measured_against_the_cap_the_loop_will_actually_sample() {
        walk_a_runaway_thought(&thinking_request(0), 8);

        let tk = harmony_tokenizer();
        let long = ChatSampler::for_request(&thinking_request(4096), &tk, &[EOS], 4096)
            .expect("<|end|> is one token of this vocabulary");
        assert_eq!(
            long.guided.as_ref().expect("guided request").think_budget,
            4096 - GUIDED_THINK_TAIL_RESERVE,
            "past the reserve the budget is max_new minus the reserve; a request whose loop cap \
             is 0 or unset would otherwise close the thought on its very first token"
        );
    }

    #[test]
    fn a_request_without_thinking_is_constrained_from_the_first_token() {
        let tk = harmony_tokenizer();
        let mut req = thinking_request(8);
        req.guided_think_close = None;
        let mut s = ChatSampler::for_request(&req, &tk, &[EOS], 8)
            .expect("a grammar without thinking always builds");
        assert_eq!(
            s.sample(&[0.0f32, 1.0, 5.0, 0.5, 3.0, 0.4, 0.3, 0.2]).token,
            1,
            "no thought to wait for means no budget to spend: the schema binds immediately"
        );
    }

    #[test]
    fn a_dead_grammar_cannot_silently_produce_output() {
        let tk = harmony_tokenizer();
        let mut req = thinking_request(8);
        req.guided_think_close = None;
        let mut s = ChatSampler::for_request(&req, &tk, &[EOS], 8)
            .expect("a grammar without thinking always builds");
        let row = vec![0.0f32, 1.0, 5.0, 0.5, 3.0, 0.4, 0.3, 0.2];
        assert_eq!(
            s.sample(&row).token,
            1,
            "the schema binds from the first token, so the boolean literal wins"
        );

        let g = s.guided.as_mut().expect("guided request");
        assert!(
            !g.advance(ILLEGAL_ARGMAX),
            "token {ILLEGAL_ARGMAX} leaves the boolean DFA: GuidedRun::advance must hand the \
             decoder's death back to its caller instead of swallowing the bool"
        );
        assert_eq!(
            g.killed_by(),
            Some(ILLEGAL_ARGMAX),
            "the token that killed the grammar must be recorded and named, not dropped as a bool"
        );
        assert!(g.decoder().is_dead());

        let mut masked = row.clone();
        g.apply_mask(&mut masked);
        assert!(
            masked.iter().all(|x| *x == f32::NEG_INFINITY),
            "the mask itself must leave nothing legal once the grammar is dead: a decoder that \
             stops masking has stopped enforcing the schema, and every token after it is \
             unconstrained even though the request still reports success"
        );

        assert!(
            s.sample(&row).exhausted,
            "a dead decoder has no continuation, so every candidate must be masked; leaving the \
             row alone would let the request run to its token cap sampling freely and answer HTTP \
             200 with finish_reason=stop to a caller who asked for a schema"
        );
        assert!(
            s.sample(&row).exhausted,
            "death is permanent: no later step may quietly resume unconstrained sampling"
        );
    }

    #[test]
    fn a_close_marker_the_logits_row_cannot_hold_fails_the_request_on_the_first_token() {
        let tk = harmony_tokenizer();
        let mut s = ChatSampler::for_request(&thinking_request(8), &tk, &[EOS], 8)
            .expect("<|end|> is one token of this vocabulary");
        let clipped = CLOSE as usize;
        let row = vec![0.0f32; clipped];
        let out = s.sample(&row);
        assert!(
            out.exhausted,
            "the close marker id {CLOSE} is outside a logits row of width {clipped}, so it can \
             neither be sampled nor forced; every generation loop turns an exhausted step into a \
             request error, which is the only honest answer once the schema can never bind"
        );
        let g = s.guided.as_ref().expect("guided request");
        assert_eq!(
            g.unforceable_close(),
            Some(CLOSE),
            "the unforceable close must be recorded and named, not dropped as a bool"
        );
        assert!(
            g.decoder().deferred(),
            "the grammar is still inert: reporting the failure must not fake an arm"
        );
        assert!(
            s.sample(&row).exhausted,
            "the failure is a property of the model's logits width, so it cannot heal by \
             sampling further"
        );
    }

    #[test]
    fn a_logits_row_wide_enough_for_the_close_marker_is_left_alone() {
        let tk = harmony_tokenizer();
        let mut s = ChatSampler::for_request(&thinking_request(8), &tk, &[EOS], 8)
            .expect("<|end|> is one token of this vocabulary");
        let out = s.sample(&[0.0f32, 1.0, 5.0, 0.5, 3.0]);
        assert!(
            !out.exhausted,
            "a row that ends exactly at the close marker holds it: the reachability check must \
             not reject the boundary case it exists to admit"
        );
        assert_eq!(out.token, ILLEGAL_ARGMAX);
        assert_eq!(
            s.guided
                .as_ref()
                .expect("guided request")
                .unforceable_close(),
            None
        );
    }
}

#[cfg(test)]
mod think_close_tests {
    use super::*;

    const HARMONY_CLOSE: &str = "<|end|><|start|>final<|message|>";
    pub(super) const EOS: u32 = 7;

    pub(super) fn harmony_tokenizer() -> tokenizers::Tokenizer {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                ["true", "false", "9", "final"]
                    .iter()
                    .enumerate()
                    .map(|(i, s)| ((*s).to_string(), i as u32))
                    .collect(),
            )
            .build()
            .expect("word-level model");
        let mut tk = tokenizers::Tokenizer::new(model);
        tk.add_special_tokens(&[
            tokenizers::AddedToken::from("<|end|>", true),
            tokenizers::AddedToken::from("<|start|>", true),
            tokenizers::AddedToken::from("<|message|>", true),
            tokenizers::AddedToken::from("<|eos|>", true),
        ]);
        tk
    }

    fn think_split_tokenizer() -> tokenizers::Tokenizer {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                ["true", "false"]
                    .iter()
                    .enumerate()
                    .map(|(i, s)| ((*s).to_string(), i as u32))
                    .collect(),
            )
            .build()
            .expect("word-level model");
        let mut tk = tokenizers::Tokenizer::new(model);
        tk.add_special_tokens(&[
            tokenizers::AddedToken::from("</", true),
            tokenizers::AddedToken::from("think", true),
            tokenizers::AddedToken::from(">", true),
            tokenizers::AddedToken::from("<|eos|>", true),
        ]);
        tk
    }

    pub(super) fn boolean_spec() -> nv_grammar::GrammarSpec {
        nv_grammar::GrammarSpec::JsonSchema(serde_json::json!({"type": "boolean"}))
    }

    fn finite(g: &mut nv_grammar::GuidedDecoder, n: usize) -> Vec<bool> {
        let mut logits = vec![0.0f32; n];
        g.apply_mask(&mut logits);
        logits.iter().map(|x| x.is_finite()).collect()
    }

    #[test]
    fn a_multi_token_thinking_close_marker_arms_only_on_the_full_sequence() {
        let tk = harmony_tokenizer();
        let seq: Vec<u32> = tk
            .encode(HARMONY_CLOSE, false)
            .expect("harmony close encodes")
            .get_ids()
            .to_vec();
        assert_eq!(
            seq.len(),
            4,
            "the fixture stops covering the multi-token path if {HARMONY_CLOSE:?} is one token"
        );

        let mut g = build_guided_for_request(&tk, &[EOS], &boolean_spec(), Some(HARMONY_CLOSE))
            .expect(
                "a model whose thinking close is a token SEQUENCE rather than one token must \
                 still get guided decoding: refusing it turns a whole model family away from \
                 structured output",
            );
        assert!(g.deferred(), "thinking is on, so the grammar starts inert");
        assert_eq!(
            g.defer_token(),
            Some(seq[0]),
            "the deferral waits on the head of the model's own close sequence"
        );
        assert_eq!(
            finite(&mut g, 8),
            vec![true; 8],
            "nothing is masked while the model is still thinking"
        );

        assert!(g.advance(seq[0]));
        assert!(g.advance(2), "more thought after a partial close");
        assert!(
            g.deferred(),
            "a prefix of the close sequence followed by prose is coincidence, not a close"
        );
        assert_eq!(
            g.defer_token(),
            Some(seq[0]),
            "the broken match restarts at the head instead of staying stuck mid-sequence"
        );

        for (i, t) in seq.iter().enumerate() {
            assert!(g.deferred(), "step {i}: the sequence is not complete yet");
            assert!(g.advance(*t));
        }
        assert!(!g.deferred(), "the full close sequence arms the grammar");
        assert_eq!(
            finite(&mut g, 8),
            vec![true, true, false, false, false, false, false, false],
            "grammar live: only the boolean literals survive"
        );
    }

    #[test]
    fn the_forced_close_walks_the_whole_sequence_one_token_per_step() {
        let tk = harmony_tokenizer();
        let seq: Vec<u32> = tk
            .encode(HARMONY_CLOSE, false)
            .expect("harmony close encodes")
            .get_ids()
            .to_vec();
        let mut g = build_guided_for_request(&tk, &[EOS], &boolean_spec(), Some(HARMONY_CLOSE))
            .expect("multi-token close must be supported");

        for (i, want) in seq.iter().enumerate() {
            let mut logits = vec![0.0f32; 8];
            assert!(
                g.mask_to_defer_token(&mut logits),
                "step {i}: the spent thinking budget must still force a close"
            );
            let legal: Vec<u32> = (0..8u32)
                .filter(|id| logits[*id as usize].is_finite())
                .collect();
            assert_eq!(
                legal,
                vec![*want],
                "step {i}: the forced close must emit the sequence one token per step"
            );
            assert!(g.advance(*want));
        }
        assert!(
            !g.deferred(),
            "after the whole sequence is forced out the grammar must be live, or the forced \
             close never terminates and the schema is never written"
        );
        assert!(
            !g.mask_to_defer_token(&mut vec![0.0f32; 8]),
            "there is nothing left to force once the grammar is armed"
        );
    }

    #[test]
    fn a_single_token_close_marker_still_defers_on_exactly_that_token() {
        let tk = harmony_tokenizer();
        let mut g = build_guided_for_request(&tk, &[EOS], &boolean_spec(), Some("<|end|>"))
            .expect("a one-token close marker is the common case and must keep working");
        assert_eq!(g.defer_token(), Some(4));
        assert!(g.advance(4));
        assert!(!g.deferred(), "one token in, one token out");
        assert_eq!(
            finite(&mut g, 8),
            vec![true, true, false, false, false, false, false, false]
        );
    }

    #[test]
    fn the_close_marker_cap_cannot_eat_the_room_reserved_for_the_answer() {
        assert!(
            GUIDED_THINK_CLOSE_MAX_TOKENS >= 6,
            "harmony ends analysis with <|end|><|start|>assistant<|channel|>final<|message|>, \
             six tokens; a cap below that refuses the very family this path exists for"
        );
        assert!(
            GUIDED_THINK_CLOSE_MAX_TOKENS < GUIDED_THINK_TAIL_RESERVE,
            "the forced close spends one reserved token per marker token, so a cap at or above \
             the reserve lets closing the thought consume every token left to answer with; a zero \
             reserve lands the close on the last token and the schema is never emitted at all"
        );
        assert_eq!(
            GUIDED_THINK_FALLBACK_MAX_TOKENS, 1,
            "a sequence is only ever the model's OWN marker; </think> is our guess, and a guess \
             is allowed only where the model actually carries that token"
        );
    }

    #[test]
    fn the_generic_close_guess_is_never_taken_as_a_sequence() {
        let tk = think_split_tokenizer();
        assert_eq!(
            tk.encode(GUIDED_THINK_FALLBACK_MARKER, false)
                .expect("the fallback marker splits")
                .get_ids()
                .len(),
            3,
            "the fixture stops covering the guard if {GUIDED_THINK_FALLBACK_MARKER} is one token"
        );
        let msg = match build_guided_for_request(
            &tk,
            &[5],
            &boolean_spec(),
            Some(GUIDED_THINK_FALLBACK_MARKER),
        ) {
            Ok(_) => panic!(
                "a model that does not carry a {GUIDED_THINK_FALLBACK_MARKER} token does not \
                 write one: deferring on its byte pieces forces out a close the model never \
                 meant, and the caller gets prose plus an invented marker instead of the refusal"
            ),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("enable_thinking=false"),
            "the refusal must tell the caller how to proceed: {msg}"
        );
    }

    #[test]
    fn a_marker_the_tokenizer_cannot_express_is_still_refused_with_the_reason() {
        let tk = harmony_tokenizer();
        let msg = match build_guided_for_request(&tk, &[EOS], &boolean_spec(), Some("<|nope|>")) {
            Ok(_) => panic!("an unresolvable close marker has no position at which to arm"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("enable_thinking=false"),
            "the refusal must tell the caller how to proceed: {msg}"
        );
    }
}
