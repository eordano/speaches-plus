#![allow(dead_code)]

pub const PPL_CORPUS_SLICE_TOKENS_N_2048_CHOSEN_SO_31B_LOAD_PLUS_FOUR_EAGER_512_BLOCKS_FITS_2_TO_3_MIN: usize =
    2048;

pub const PPL_BLOCK_512_TOKENS_FRESH_CACHE_SINGLE_PREFILL_NO_CROSS_BLOCK_CONTEXT_BECAUSE_QWEN36_PREFILL_NLL_DEGRADES_1_9_TO_10_PAST_512_POSITIONS:
    usize = 512;

pub fn ppl_block_len_with_debug_override_which_makes_the_run_noncanonical() -> usize {
    std::env::var("NV_PPL_DEBUG_BLOCK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(PPL_BLOCK_512_TOKENS_FRESH_CACHE_SINGLE_PREFILL_NO_CROSS_BLOCK_CONTEXT_BECAUSE_QWEN36_PREFILL_NLL_DEGRADES_1_9_TO_10_PAST_512_POSITIONS)
}

pub fn any_ppl_debug_env_set_so_the_machine_line_must_not_look_canonical() -> bool {
    ["NV_PPL_DEBUG_BLOCK", "NV_PPL_DEBUG_N", "NV_PPL_DEBUG_TAIL_ONLY", "NV_PPL_DEBUG_PROBE_START"]
        .iter()
        .any(|k| std::env::var(k).is_ok())
}

pub fn require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip() {
    if std::env::var("NV_PPL_TEST").as_deref() != Ok("1") {
        panic!("set NV_PPL_TEST=1 to run (this suite must never silently skip)");
    }
}

pub fn corpus_text_from_nv_ppl_corpus_env_failing_loudly() -> String {
    let p = std::env::var("NV_PPL_CORPUS").unwrap_or_else(|_| {
        panic!(
            "NV_PPL_CORPUS is unset: point it at the shared corpus file \
             (the corpus agent writes /home/user/tmp/wf/quant/corpus/wiki.txt)"
        )
    });
    let path = std::path::PathBuf::from(&p);
    assert!(
        path.is_file(),
        "NV_PPL_CORPUS={p} does not exist or is not a file; every config must score the same corpus slice"
    );
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("NV_PPL_CORPUS={p} is unreadable: {e}"));
    assert!(
        !text.trim().is_empty(),
        "NV_PPL_CORPUS={p} is empty; an empty corpus measures nothing"
    );
    text
}

#[path = "../official_template/mod.rs"]
mod official_template;
pub use official_template::OfficialTemplate;

pub const CHAT_WRAPPED_CONTINUATION_USER_INSTRUCTION: &str =
    "Continue the following text, staying in the same style, with no commentary:";

pub fn chat_wrapped_continuation_ids_and_copy_start_rendered_through_the_snapshot_jinja(
    dir: &std::path::Path,
    tokenizer: &tokenizers::Tokenizer,
    corpus_text: &str,
    bos: u32,
    ctx: usize,
    cont: usize,
) -> (Vec<u32>, usize) {
    let all = first_n_corpus_tokens_after_tokenization(tokenizer, corpus_text, ctx + cont);
    let ctx_text = tokenizer
        .decode(&all[..ctx], false)
        .expect("decode context slice");
    let user = format!("{CHAT_WRAPPED_CONTINUATION_USER_INSTRUCTION}\n\n{ctx_text}");
    let prompt = OfficialTemplate::load(dir).render_user(&user);
    let enc = tokenizer
        .encode(prompt.as_str(), false)
        .expect("tokenize chat prompt");
    let mut ids = enc.get_ids().to_vec();
    assert_eq!(
        ids.first().copied(),
        Some(bos),
        "the official render must begin with the bos token; a template that does not embed \
         bos_token would make this harness double-count or drop it"
    );
    let score_start = ids.len();
    ids.extend_from_slice(&all[ctx..]);
    (ids, score_start)
}

pub fn first_n_corpus_tokens_after_tokenization(
    tokenizer: &tokenizers::Tokenizer,
    corpus_text: &str,
    n: usize,
) -> Vec<u32> {
    let enc = tokenizer
        .encode(corpus_text, false)
        .expect("tokenize corpus");
    let ids = enc.get_ids();
    assert!(
        ids.len() >= n,
        "corpus tokenizes to only {} tokens but the shared slice needs {n}; \
         a shorter slice here would not be comparable with other configs",
        ids.len()
    );
    ids[..n].to_vec()
}

pub fn deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
    ids: &mut [u32],
) {
    let mut s: u64 = 0x5eed_1234_9abc_def0;
    for i in (1..ids.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let j = (s % (i as u64 + 1)) as usize;
        ids.swap(i, j);
    }
}

pub struct TeacherForcedNllFp32SoftmaxF64Sum {
    pub sum_neg_ln_p_f64: f64,
    pub scored_positions: usize,
    pub argmax_hits: usize,
}

impl TeacherForcedNllFp32SoftmaxF64Sum {
    pub fn new() -> Self {
        Self {
            sum_neg_ln_p_f64: 0.0,
            scored_positions: 0,
            argmax_hits: 0,
        }
    }

    pub fn top1_accuracy(&self) -> f64 {
        assert!(
            self.scored_positions > 0,
            "no positions were scored; the harness ran nothing"
        );
        self.argmax_hits as f64 / self.scored_positions as f64
    }

    pub fn add_position_full_vocab_row(&mut self, logits_row_f32: &[f32], true_next_token: u32) {
        let t = true_next_token as usize;
        assert!(
            t < logits_row_f32.len(),
            "true next token id {t} is outside the logit row of width {}; \
             tokenizer and model vocab disagree",
            logits_row_f32.len()
        );
        let mut m = f32::NEG_INFINITY;
        let mut argmax = 0usize;
        for (j, &v) in logits_row_f32.iter().enumerate() {
            assert!(
                v.is_finite(),
                "non-finite logit at scored position {}: a NaN/Inf row poisons the mean",
                self.scored_positions
            );
            if v > m {
                m = v;
                argmax = j;
            }
        }
        if argmax == t {
            self.argmax_hits += 1;
        }
        let mut denom = 0f32;
        for &v in logits_row_f32 {
            denom += (v - m).exp();
        }
        let ln_p = (logits_row_f32[t] - m) - denom.ln();
        assert!(
            ln_p.is_finite() && ln_p <= 0.0,
            "log-prob {ln_p} out of range at scored position {}",
            self.scored_positions
        );
        self.sum_neg_ln_p_f64 += -(ln_p as f64);
        self.scored_positions += 1;
    }

    pub fn perplexity_exp_of_mean_neg_ln_p(&self) -> f64 {
        assert!(
            self.scored_positions > 0,
            "no positions were scored; the harness ran nothing"
        );
        (self.sum_neg_ln_p_f64 / self.scored_positions as f64).exp()
    }
}

pub fn print_machine_line_and_assert_real_beats_shuffled(
    family: &str,
    checkpoint: &str,
    scored_positions: usize,
    ppl_real: f64,
    ppl_shuffled: f64,
) {
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        family,
        checkpoint,
        scored_positions,
        ppl_real,
        ppl_shuffled,
        f64::NAN,
    )
}

pub fn print_machine_line_with_acc_and_assert_real_beats_shuffled(
    family: &str,
    checkpoint: &str,
    scored_positions: usize,
    ppl_real: f64,
    ppl_shuffled: f64,
    top1_acc_real_nan_when_caller_predates_accuracy: f64,
) {
    let family = if any_ppl_debug_env_set_so_the_machine_line_must_not_look_canonical() {
        format!("{family}-DEBUG-NONCANONICAL")
    } else {
        family.to_string()
    };
    if top1_acc_real_nan_when_caller_predates_accuracy.is_nan() {
        println!("PPL {family} {checkpoint} tokens={scored_positions} ppl={ppl_real:.3}");
    } else {
        println!(
            "PPL {family} {checkpoint} tokens={scored_positions} ppl={ppl_real:.3} acc={top1_acc_real_nan_when_caller_predates_accuracy:.4}"
        );
    }
    println!("PPL-SHUFFLED-CONTROL {family} {checkpoint} tokens={scored_positions} ppl={ppl_shuffled:.3}");
    assert!(
        ppl_real.is_finite() && ppl_shuffled.is_finite(),
        "non-finite perplexity: real={ppl_real} shuffled={ppl_shuffled}"
    );
    assert!(
        ppl_real < ppl_shuffled,
        "sanity gate failed: real-text ppl {ppl_real:.3} is not below shuffled-text ppl {ppl_shuffled:.3}; \
         a harness that cannot tell wikipedia from shuffled wikipedia measures nothing"
    );
}

pub fn checkpoint_label_from_snapshot_dir(dir: &std::path::Path) -> String {
    let snap = dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let repo = dir
        .ancestors()
        .find_map(|a| {
            let name = a.file_name()?.to_string_lossy().to_string();
            name.starts_with("models--").then_some(name)
        })
        .unwrap_or_else(|| "unknown-repo".into());
    let short = &snap[..snap.len().min(8)];
    format!("{repo}@{short}")
}
pub fn eprintln_ppl_skipped_when_any_true_token_exceeded_the_logit_width(
    skipped_beyond_logit_width: usize,
) {
    if skipped_beyond_logit_width > 0 {
        eprintln!(
            "PPL-SKIPPED {skipped_beyond_logit_width} positions whose true token id exceeds the \
             model's logit width (tokenizer vocab is padded past the lm head); scored positions \
             exclude them on every config identically"
        );
    }
}

pub fn score_tail_teacher_forced_decode_stepping_converting_only_scored_rows(
    step_logits: &mut dyn FnMut(usize, u32) -> candle_core::Tensor,
    ids: &[u32],
    score_start: usize,
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    let len = ids.len();
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    let mut skipped_beyond_logit_width = 0usize;
    for p in 0..len - 1 {
        let logits = step_logits(p, ids[p]);
        if p + 1 < score_start {
            continue;
        }
        let row: Vec<f32> = logits
            .to_dtype(candle_core::DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("host");
        if ids[p + 1] as usize >= row.len() {
            skipped_beyond_logit_width += 1;
            continue;
        }
        acc.add_position_full_vocab_row(&row, ids[p + 1]);
    }
    eprintln_ppl_skipped_when_any_true_token_exceeded_the_logit_width(skipped_beyond_logit_width);
    acc
}

pub fn score_tail_teacher_forced_wgpu_decode_step_logits_full_vocab_row(
    decode_step_logits_row: &mut dyn FnMut(usize, u32) -> Vec<f32>,
    vocab: usize,
    ids: &[u32],
    score_start: usize,
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    for p in 0..ids.len() - 1 {
        let row = decode_step_logits_row(p, ids[p]);
        if p + 1 < score_start {
            continue;
        }
        assert_eq!(
            row.len(),
            vocab,
            "decode_step_logits must return the full vocab row, not a top-k slice"
        );
        acc.add_position_full_vocab_row(&row, ids[p + 1]);
    }
    acc
}

pub fn score_corpus_in_blocks_first_token_input_only_one_prefill_per_fresh_cache(
    block_flat_f32_logits: &mut dyn FnMut(&[u32], usize) -> Vec<f32>,
    corpus_tokens: &[u32],
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    let block = ppl_block_len_with_debug_override_which_makes_the_run_noncanonical();
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    let mut skipped_beyond_logit_width = 0usize;
    let mut bstart = 0usize;
    while bstart < corpus_tokens.len() {
        let bend = (bstart + block).min(corpus_tokens.len());
        let ids = &corpus_tokens[bstart..bend];
        let k = ids.len();
        let flat = block_flat_f32_logits(ids, bstart);
        assert_eq!(flat.len() % k, 0, "logit rows not divisible by block len");
        let vocab = flat.len() / k;
        let nll_before = acc.sum_neg_ln_p_f64;
        let scored_before = acc.scored_positions;
        for i in 0..k - 1 {
            if ids[i + 1] as usize >= vocab {
                skipped_beyond_logit_width += 1;
                continue;
            }
            acc.add_position_full_vocab_row(&flat[i * vocab..(i + 1) * vocab], ids[i + 1]);
        }
        eprintln!(
            "PPL-BLOCK start={bstart} scored={} mean_nll={:.3}",
            acc.scored_positions - scored_before,
            (acc.sum_neg_ln_p_f64 - nll_before) / (acc.scored_positions - scored_before) as f64
        );
        bstart = bend;
    }
    eprintln_ppl_skipped_when_any_true_token_exceeded_the_logit_width(skipped_beyond_logit_width);
    acc
}

pub fn shuffle_tail_score_both_arms_assert_equal_positions_and_print_machine_line(
    family: &str,
    label: &str,
    ids: &[u32],
    score_start: usize,
    score: &mut dyn FnMut(&[u32]) -> TeacherForcedNllFp32SoftmaxF64Sum,
) {
    let real = score(ids);
    let mut shuffled_ids = ids.to_vec();
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
        &mut shuffled_ids[score_start..],
    );
    let shuffled = score(&shuffled_ids);
    assert_eq!(
        real.scored_positions, shuffled.scored_positions,
        "real and shuffled arms scored different position counts"
    );
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        family,
        label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}
