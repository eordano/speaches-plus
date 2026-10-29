use anyhow::{bail, ensure, Result};

use crate::voxtral::{EOS_TOKEN, STREAMING_PAD_TOKEN, STREAMING_WORD_TOKEN};

pub const A_WORD_GROUP_IS_ONE_STREAMING_WORD_MARK_AND_THE_TEXT_TOKENS_THAT_FOLLOW_IT: &str =
    "the realtime decoder emits [STREAMING_WORD] once per emission frame and then the subword \
     tokens of every word that completed in that frame, so a group can hold several words; a \
     word boundary inside a group is a token whose bytes start with a space";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WordGroup {
    pub mark_index: usize,
    pub words: Vec<Vec<u32>>,
}

pub fn word_groups(stream: &[u32], starts_with_space: impl Fn(u32) -> bool) -> Vec<WordGroup> {
    let _ = A_WORD_GROUP_IS_ONE_STREAMING_WORD_MARK_AND_THE_TEXT_TOKENS_THAT_FOLLOW_IT;
    let mut groups = Vec::new();
    let mut i = 0;
    while i < stream.len() {
        if stream[i] != STREAMING_WORD_TOKEN {
            i += 1;
            continue;
        }
        let mark_index = i;
        i += 1;
        let mut words: Vec<Vec<u32>> = Vec::new();
        while i < stream.len()
            && stream[i] != STREAMING_WORD_TOKEN
            && stream[i] != STREAMING_PAD_TOKEN
            && stream[i] != EOS_TOKEN
        {
            if words.is_empty() || starts_with_space(stream[i]) {
                words.push(vec![stream[i]]);
            } else {
                words.last_mut().unwrap().push(stream[i]);
            }
            i += 1;
        }
        groups.push(WordGroup { mark_index, words });
    }
    groups
}

pub fn normalize_word(w: &str) -> String {
    w.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    Keep(usize, usize),
    Substitute(usize, usize),
    DeleteHyp(usize),
    InsertRef(usize),
}

pub fn align_words(hyp: &[String], reference: &[String]) -> Vec<Align> {
    let n = hyp.len();
    let m = reference.len();
    let mut cost = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in cost.iter_mut().enumerate() {
        row[0] = i;
    }
    for j in 0..=m {
        cost[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let same = hyp[i - 1] == reference[j - 1];
            let sub = cost[i - 1][j - 1] + usize::from(!same);
            cost[i][j] = sub.min(cost[i - 1][j] + 1).min(cost[i][j - 1] + 1);
        }
    }
    let mut ops = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let same = hyp[i - 1] == reference[j - 1];
            if cost[i][j] == cost[i - 1][j - 1] + usize::from(!same) {
                ops.push(if same {
                    Align::Keep(i - 1, j - 1)
                } else {
                    Align::Substitute(i - 1, j - 1)
                });
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && cost[i][j] == cost[i - 1][j] + 1 {
            ops.push(Align::DeleteHyp(i - 1));
            i -= 1;
        } else {
            ops.push(Align::InsertRef(j - 1));
            j -= 1;
        }
    }
    ops.reverse();
    ops
}

pub const SPLICE_KEEPS_EVERY_WORD_MARK_AT_OR_AFTER_ITS_ACOUSTIC_FRAME: &str =
    "a corrected word may be longer or shorter than the model's own guess; a group that grows \
     pushes the groups after it later, never earlier, because a mark before the frame the base \
     model emitted it at would ask the model to commit before the audio justifies the word";

pub fn splice_reference_into_stream(
    hyp_stream: &[u32],
    hyp_groups: &[WordGroup],
    hyp_words: &[String],
    reference_words: &[String],
    encode_word: impl Fn(&str) -> Vec<u32>,
) -> Result<Vec<u32>> {
    let _ = SPLICE_KEEPS_EVERY_WORD_MARK_AT_OR_AFTER_ITS_ACOUSTIC_FRAME;
    let total_hyp_words: usize = hyp_groups.iter().map(|g| g.words.len()).sum();
    if total_hyp_words != hyp_words.len() {
        bail!(
            "hyp_words has {} entries but the groups hold {total_hyp_words} words",
            hyp_words.len()
        );
    }
    let hyp_norm: Vec<String> = hyp_words.iter().map(|w| normalize_word(w)).collect();
    let ref_norm: Vec<String> = reference_words.iter().map(|w| normalize_word(w)).collect();
    let ops = align_words(&hyp_norm, &ref_norm);

    let mut owner_of_hyp_word: Vec<usize> = Vec::with_capacity(total_hyp_words);
    for (gi, g) in hyp_groups.iter().enumerate() {
        owner_of_hyp_word.extend(std::iter::repeat_n(gi, g.words.len()));
    }
    let mut new_words: Vec<Vec<Vec<u32>>> = vec![Vec::new(); hyp_groups.len()];
    let mut last_group = 0usize;
    for op in &ops {
        match *op {
            Align::Keep(h, r) | Align::Substitute(h, r) => {
                last_group = owner_of_hyp_word[h];
                new_words[last_group].push(encode_word(&reference_words[r]));
            }
            Align::DeleteHyp(h) => {
                last_group = owner_of_hyp_word[h];
            }
            Align::InsertRef(r) => {
                if hyp_groups.is_empty() {
                    bail!(
                        "the hypothesis emitted no word at all; nothing to anchor the reference to"
                    );
                }
                new_words[last_group].push(encode_word(&reference_words[r]));
            }
        }
    }

    let ends_with_eos = hyp_stream.last() == Some(&EOS_TOKEN);
    let body_len = hyp_stream.len() - usize::from(ends_with_eos);
    let mut out: Vec<u32> = Vec::with_capacity(hyp_stream.len() + 8);
    for (g, words) in hyp_groups.iter().zip(&new_words) {
        if words.is_empty() {
            continue;
        }
        while out.len() < g.mark_index {
            out.push(STREAMING_PAD_TOKEN);
        }
        out.push(STREAMING_WORD_TOKEN);
        for w in words {
            out.extend_from_slice(w);
        }
    }
    if out.len() > body_len {
        bail!(
            "corrected words overflow the audio by {} steps (stream holds {body_len})",
            out.len() - body_len
        );
    }
    while out.len() < body_len {
        out.push(STREAMING_PAD_TOKEN);
    }
    if ends_with_eos {
        out.push(EOS_TOKEN);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = STREAMING_WORD_TOKEN;
    const P: u32 = STREAMING_PAD_TOKEN;

    fn space_first(id: u32) -> bool {
        id >= 2000
    }

    #[test]
    fn groups_split_words_on_space_initial_tokens() {
        let s = [P, P, W, 2001, 1002, 2003, P, W, 2004, P, P];
        let g = word_groups(&s, space_first);
        assert_eq!(
            g,
            vec![
                WordGroup {
                    mark_index: 2,
                    words: vec![vec![2001, 1002], vec![2003]]
                },
                WordGroup {
                    mark_index: 7,
                    words: vec![vec![2004]]
                },
            ]
        );
    }

    #[test]
    fn alignment_reports_keeps_substitutions_insertions_and_deletions() {
        let h = ["what", "are", "u", "doing", "extra"].map(String::from);
        let r = ["what", "are", "you", "doing", "here"].map(String::from);
        assert_eq!(
            align_words(&h, &r),
            vec![
                Align::Keep(0, 0),
                Align::Keep(1, 1),
                Align::Substitute(2, 2),
                Align::Keep(3, 3),
                Align::Substitute(4, 4),
            ]
        );
        let h = ["a", "c"].map(String::from);
        let r = ["a", "b", "c"].map(String::from);
        assert_eq!(
            align_words(&h, &r),
            vec![Align::Keep(0, 0), Align::InsertRef(1), Align::Keep(1, 2)]
        );
        let h = ["a", "x", "b"].map(String::from);
        let r = ["a", "b"].map(String::from);
        assert_eq!(
            align_words(&h, &r),
            vec![Align::Keep(0, 0), Align::DeleteHyp(1), Align::Keep(2, 1)]
        );
    }

    #[test]
    fn splice_substitutes_words_in_place_and_shifts_later_marks_only_later() {
        let hyp = [P, W, 2001, P, P, W, 2002, P, P, P];
        let groups = word_groups(&hyp, space_first);
        let hyp_words = ["one", "two"].map(String::from);
        let reference = ["won", "two"].map(String::from);
        let enc = |w: &str| match w {
            "won" => vec![2009, 1001, 1002],
            "two" => vec![2002],
            other => panic!("unexpected word {other}"),
        };
        let out = splice_reference_into_stream(&hyp, &groups, &hyp_words, &reference, enc).unwrap();
        assert_eq!(out, vec![P, W, 2009, 1001, 1002, W, 2002, P, P, P]);
        assert_eq!(out.len(), hyp.len());
    }

    #[test]
    fn splice_drops_deleted_words_attaches_inserted_ones_and_refuses_overflow() {
        let hyp = [W, 2001, W, 2002, P, P, 2];
        let groups = word_groups(&hyp, space_first);
        let hyp_words = ["one", "two"].map(String::from);
        let reference = ["one", "and", "two"].map(String::from);
        let enc = |w: &str| match w {
            "one" => vec![2001],
            "and" => vec![2005],
            "two" => vec![2002],
            other => panic!("unexpected word {other}"),
        };
        let out =
            splice_reference_into_stream(&hyp, &groups, &hyp_words, &reference, &enc).unwrap();
        assert_eq!(out, vec![W, 2001, 2005, W, 2002, P, 2]);
        let reference = ["two"].map(String::from);
        let out =
            splice_reference_into_stream(&hyp, &groups, &hyp_words, &reference, &enc).unwrap();
        assert_eq!(out, vec![P, P, W, 2002, P, P, 2]);
        let reference = ["one", "and", "and", "and", "two"].map(String::from);
        assert!(splice_reference_into_stream(&hyp, &groups, &hyp_words, &reference, &enc).is_err());
    }
}

use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Tensor, Var, D};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use nv_train::{
    ce_mean_over_unmasked_positions, save_peft_raw, FusedSplitHook, LoraConfig, LoraTrainable,
    RawLora, StaticLoraHook, TrainingLoraHook,
};
use nv_weights::lora_adapter::LoraAdapter;

use crate::voxtral::{DraftHead, VoxtralRealtime};

pub fn sdpa_train_scale_one(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let seq = q.dims()[1];
    let device = q.device().clone();
    let q3 = q
        .to_dtype(DType::F32)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((n_q, seq, head_dim))?;
    let k3 = k
        .to_dtype(DType::F32)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((n_kv, seq, head_dim))?;
    let v3 = v
        .to_dtype(DType::F32)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((n_kv, seq, head_dim))?;
    let group = n_q / n_kv;
    let map: Vec<u32> = (0..n_q).map(|h| (h / group) as u32).collect();
    let map_t = Tensor::from_vec(map, n_q, &device)?;
    let k3 = k3.index_select(&map_t, 0)?.contiguous()?;
    let v3 = v3.index_select(&map_t, 0)?.contiguous()?;
    let mut mask = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    let mask_t = Tensor::from_vec(mask, (1usize, seq, seq), &device)?;
    let scores = q3
        .matmul(&k3.transpose(1, 2)?.contiguous()?)?
        .broadcast_add(&mask_t)?;
    let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
    let out = probs.matmul(&v3)?;
    out.reshape((1, n_q, seq, head_dim))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((1, seq, n_q * head_dim))
        .map_err(Into::into)
}

pub const ONE_ADAPTER_PER_FUSED_LINEAR_EXPORTS_AS_PEFT_MODULES_SHARING_LORA_A: &str =
    "the decoder's q/k/v and gate/up live in fused Linears, so training holds one A/B pair per \
     fused weight; PEFT wants one pair per HF module, which a row split of B gives exactly: \
     q_proj, k_proj and v_proj (or gate_proj and up_proj) carry the same A and their own rows \
     of B, which is the restricted-but-valid parametrisation the serving FusedSplitHook replays";

pub struct VoxtralLora {
    pub cfg: LoraConfig,
    modules: Vec<(String, Vec<(&'static str, usize)>, Arc<LoraTrainable>)>,
}

impl VoxtralLora {
    pub fn attach(model: &VoxtralRealtime, cfg: LoraConfig, seed: u64) -> Result<Self> {
        let _ = ONE_ADAPTER_PER_FUSED_LINEAR_EXPORTS_AS_PEFT_MODULES_SHARING_LORA_A;
        let device = model.device().clone();
        let mut modules = Vec::new();
        for (idx, (prefix, linear, parts)) in model.decoder_linears().into_iter().enumerate() {
            let lora = Arc::new(LoraTrainable::from_features(
                linear.out_features(),
                linear.in_features(),
                LoraConfig { ..cfg },
                seed.wrapping_add(idx as u64 * 7919),
                &device,
            )?);
            linear.attach_lora(Arc::new(TrainingLoraHook::new(lora.clone(), true)))?;
            modules.push((prefix, parts, lora));
        }
        model.drop_graph_state();
        Ok(Self { cfg, modules })
    }

    pub fn detach(model: &VoxtralRealtime) {
        for (_, linear, _) in model.decoder_linears() {
            linear.detach_lora();
        }
        model.drop_graph_state();
    }

    pub fn vars(&self) -> Vec<Var> {
        self.modules
            .iter()
            .flat_map(|(_, _, l)| l.trainable_vars())
            .collect()
    }

    pub fn save_peft(&self, dir: &Path, base_model_name_or_path: &str) -> Result<()> {
        let mut entries = Vec::new();
        for (prefix, parts, lora) in &self.modules {
            let a = lora.a_tensor().detach();
            let b = lora.b_tensor().detach();
            let mut off = 0;
            for (leaf, rows) in parts {
                entries.push(RawLora {
                    module: format!("{prefix}.{leaf}"),
                    a: a.clone(),
                    b: b.narrow(0, off, *rows)?.contiguous()?,
                });
                off += rows;
            }
        }
        save_peft_raw(
            dir,
            &entries,
            self.cfg.r,
            self.cfg.alpha,
            self.cfg.dropout,
            base_model_name_or_path,
        )
    }
}

pub fn attach_peft_for_serving(model: &VoxtralRealtime, dir: &Path) -> Result<usize> {
    let adapter = LoraAdapter::load(dir, model.device())?;
    let mut attached = 0;
    for (prefix, linear, parts) in model.decoder_linears() {
        let mut pieces: Vec<(Tensor, Tensor, usize, usize)> = Vec::new();
        let mut off = 0;
        let mut scaling = 1.0;
        for (leaf, rows) in &parts {
            let name = format!("{prefix}.{leaf}");
            let Some(lw) = adapter
                .loras
                .get(&nv_weights::lora_adapter::normalize_module_name(&name))
            else {
                off += rows;
                continue;
            };
            ensure!(
                lw.lora_b.dims()[0] == *rows && lw.lora_a.dims()[1] == linear.in_features(),
                "adapter module {name}: B rows {} / A cols {} do not match the fused linear",
                lw.lora_b.dims()[0],
                lw.lora_a.dims()[1]
            );
            scaling = lw.scaling;
            pieces.push((lw.lora_a.clone(), lw.lora_b.clone(), off, *rows));
            off += rows;
        }
        if pieces.is_empty() {
            continue;
        }
        if parts.len() == 1 {
            let (a, b, _, _) = pieces.remove(0);
            linear.attach_lora(Arc::new(StaticLoraHook::new(a, b, scaling)?))?;
        } else {
            linear.attach_lora(Arc::new(FusedSplitHook::new(
                pieces,
                scaling,
                linear.in_features(),
                linear.out_features(),
            )?))?;
        }
        attached += 1;
    }
    ensure!(
        attached > 0,
        "adapter at {} matched no decoder linear",
        dir.display()
    );
    model.drop_graph_state();
    Ok(attached)
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StreamRow {
    pub audio: String,
    pub text: String,
    pub hypothesis: String,
    pub stream: Vec<u32>,
}

pub fn target_stream_for(
    model: &VoxtralRealtime,
    hyp_stream: &[u32],
    reference_text: &str,
) -> Result<Vec<u32>> {
    let vocab = model.vocab();
    let enc = vocab.encoder()?;
    let groups = word_groups(hyp_stream, |id| vocab.starts_with_space(id));
    let hyp_words: Vec<String> = groups
        .iter()
        .flat_map(|g| g.words.iter().map(|w| vocab.decode(w).trim().to_string()))
        .collect();
    let reference_words: Vec<String> = reference_text
        .split_whitespace()
        .map(str::to_string)
        .collect();
    splice_reference_into_stream(hyp_stream, &groups, &hyp_words, &reference_words, |w| {
        enc.encode(&format!(" {w}"))
    })
}

pub fn row_loss(
    model: &VoxtralRealtime,
    audio_embeds: &Tensor,
    full_stream: &[u32],
) -> Result<Tensor> {
    let n = full_stream.len();
    ensure!(n >= 2, "stream too short");
    let inputs = model.inputs_embeds_for_stream(&full_stream[..n - 1], audio_embeds)?;
    let logits = model.decoder_logits_train(&inputs)?;
    let targets = Tensor::new(&full_stream[1..], model.device())?;
    let prefix_len = model.prefix_len();
    let mask: Vec<f32> = (1..n)
        .map(|predicted| if predicted >= prefix_len { 1.0 } else { 0.0 })
        .collect();
    ce_mean_over_unmasked_positions(&logits, &targets, &mask)
}

pub struct TrainReport {
    pub losses: Vec<f32>,
}

pub fn train_lora(
    model: &VoxtralRealtime,
    lora: &VoxtralLora,
    rows: &[(Tensor, Vec<u32>)],
    epochs: usize,
    lr: f64,
) -> Result<TrainReport> {
    let vars = lora.vars();
    let mut opt = AdamW::new(
        vars,
        ParamsAdamW {
            lr,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;
    let mut losses = Vec::new();
    for epoch in 0..epochs {
        for (i, (embeds, stream)) in rows.iter().enumerate() {
            let loss = row_loss(model, embeds, stream)?;
            let lv = loss.to_scalar::<f32>()?;
            ensure!(lv.is_finite(), "non-finite loss at epoch {epoch} row {i}");
            opt.backward_step(&loss)?;
            losses.push(lv);
            eprintln!("epoch {epoch} row {i}: loss {lv:.4}");
        }
    }
    Ok(TrainReport { losses })
}

pub fn word_error_rate(hypothesis: &str, reference: &str) -> (usize, usize) {
    let h: Vec<String> = hypothesis
        .split_whitespace()
        .map(normalize_word)
        .filter(|w| !w.is_empty())
        .collect();
    let r: Vec<String> = reference
        .split_whitespace()
        .map(normalize_word)
        .filter(|w| !w.is_empty())
        .collect();
    let errors = align_words(&h, &r)
        .iter()
        .filter(|op| !matches!(op, Align::Keep(_, _)))
        .count();
    (errors, r.len())
}

#[cfg(test)]
mod wer_tests {
    use super::*;

    #[test]
    fn wer_counts_substitutions_insertions_and_deletions_over_reference_words() {
        assert_eq!(
            word_error_rate("what are you doing", "What are you doing?"),
            (0, 4)
        );
        assert_eq!(word_error_rate("what r you", "what are you doing"), (2, 4));
        assert_eq!(word_error_rate("", "a b"), (2, 2));
    }
}

pub struct DraftHeadReport {
    pub classes: Vec<u32>,
    pub losses: Vec<f32>,
    pub train_accuracy: f32,
}

pub const THE_DRAFT_HEAD_TRAINS_ON_THE_MODELS_OWN_GREEDY_STREAM_NOT_THE_REFERENCE: &str =
    "the verifier commits the model's own tokens, so the probe learns where the model emits \
     pads, word marks and pieces, which differs from the spliced reference stream the LoRA \
     trains on; a token outside the class set is the other class, drafted as a pad";

pub fn draft_classes(streams: &[Vec<u32>], top_k: usize) -> Vec<u32> {
    let mut counts = std::collections::HashMap::new();
    for t in streams.iter().flatten() {
        if *t != STREAMING_PAD_TOKEN && *t != STREAMING_WORD_TOKEN {
            *counts.entry(*t).or_insert(0usize) += 1;
        }
    }
    let mut by_freq: Vec<(u32, usize)> = counts.into_iter().collect();
    by_freq.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut classes = vec![STREAMING_PAD_TOKEN, STREAMING_WORD_TOKEN];
    classes.extend(by_freq.iter().take(top_k).map(|(t, _)| *t));
    classes.push(STREAMING_PAD_TOKEN);
    classes
}

pub fn draft_targets(classes: &[u32], stream: &[u32]) -> Vec<u32> {
    let other = (classes.len() - 1) as u32;
    stream
        .iter()
        .map(|t| {
            classes[..classes.len() - 1]
                .iter()
                .position(|c| c == t)
                .map_or(other, |i| i as u32)
        })
        .collect()
}

pub fn draft_features(model: &VoxtralRealtime, audio_embeds: &Tensor, stream_len: usize) -> Result<Tensor> {
    let prefix_len = model.prefix_len();
    let n = stream_len.min(audio_embeds.dims()[1] - prefix_len + 1);
    audio_embeds
        .narrow(1, prefix_len - 1, n)?
        .squeeze(0)?
        .to_dtype(DType::F32)
        .map_err(Into::into)
}

pub fn train_draft_head(
    model: &VoxtralRealtime,
    rows: &[(Tensor, Vec<u32>)],
    top_k: usize,
    steps: usize,
    lr: f64,
) -> Result<(DraftHead, DraftHeadReport)> {
    let _ = THE_DRAFT_HEAD_TRAINS_ON_THE_MODELS_OWN_GREEDY_STREAM_NOT_THE_REFERENCE;
    let device = model.device().clone();
    let streams: Vec<Vec<u32>> = rows.iter().map(|(_, s)| s.clone()).collect();
    let classes = draft_classes(&streams, top_k);
    let mut xs = Vec::with_capacity(rows.len());
    let mut ys = Vec::new();
    for (embeds, stream) in rows {
        let x = draft_features(model, embeds, stream.len())?;
        ys.extend(draft_targets(&classes, &stream[..x.dims()[0]]));
        xs.push(x);
    }
    let x = Tensor::cat(&xs, 0)?;
    let y = Tensor::new(ys.as_slice(), &device)?;
    let hidden = x.dims()[1];
    let w = Var::zeros((classes.len(), hidden), DType::F32, &device)?;
    let b = Var::zeros(classes.len(), DType::F32, &device)?;
    let mut opt = AdamW::new(
        vec![w.clone(), b.clone()],
        ParamsAdamW {
            lr,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;
    let mut losses = Vec::with_capacity(steps);
    for step in 0..steps {
        let logits = x.matmul(&w.t()?)?.broadcast_add(b.as_tensor())?;
        let loss = candle_nn::loss::cross_entropy(&logits, &y)?;
        let lv = loss.to_scalar::<f32>()?;
        ensure!(lv.is_finite(), "non-finite draft head loss at step {step}");
        opt.backward_step(&loss)?;
        losses.push(lv);
    }
    let head = DraftHead {
        weight: w.as_tensor().detach(),
        bias: b.as_tensor().detach(),
        classes: classes.clone(),
    };
    let pred = x
        .matmul(&head.weight.t()?)?
        .broadcast_add(&head.bias)?
        .argmax(D::Minus1)?
        .to_vec1::<u32>()?;
    let hits = pred.iter().zip(&ys).filter(|(p, t)| p == t).count();
    Ok((
        head,
        DraftHeadReport {
            classes,
            losses,
            train_accuracy: hits as f32 / ys.len().max(1) as f32,
        },
    ))
}

#[cfg(test)]
mod draft_tests {
    use super::*;

    #[test]
    fn draft_classes_are_pad_word_top_k_pieces_then_other() {
        let streams = vec![vec![32, 33, 500, 501, 500, 32], vec![33, 502, 500, 2]];
        assert_eq!(draft_classes(&streams, 2), vec![32, 33, 500, 2, 32]);
        assert_eq!(
            draft_targets(&[32, 33, 500, 2, 32], &[32, 33, 500, 501, 502, 2]),
            vec![0, 1, 2, 4, 4, 3]
        );
    }
}
