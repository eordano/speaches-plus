use candle_core::{DType, Device, Tensor};
use nv_train::{
    ce_mean_over_unmasked_positions, load_jsonl_token_dataset, DatasetTokenizer, TokenRow,
};
use std::path::{Path, PathBuf};

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nv-train-jsonl-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn word_level_tokenizer_json() -> String {
    serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "again": 3},
            "unk_token": "[UNK]"
        }
    })
    .to_string()
}

fn write_base(dir: &Path, with_tokenizer: bool) {
    std::fs::create_dir_all(dir).unwrap();
    if with_tokenizer {
        std::fs::write(dir.join("tokenizer.json"), word_level_tokenizer_json()).unwrap();
    }
}

#[test]
fn a_base_with_a_tokenizer_json_tokenizes_rows_through_it_and_keeps_the_prompt_boundary() {
    let root = scratch("real-tok");
    let base = root.join("base");
    write_base(&base, true);
    let tok = DatasetTokenizer::for_base(&base, 16).unwrap();
    assert!(
        tok.is_real(),
        "a tokenizer.json beside the weights must select the real tokenizer, never bytes"
    );

    let data = root.join("data.jsonl");
    std::fs::write(
        &data,
        format!(
            "{}\n{}\n",
            serde_json::json!({"text": "hello world again"}),
            serde_json::json!({"prompt": "hello world", "completion": "again hello"}),
        ),
    )
    .unwrap();
    let rows = load_jsonl_token_dataset(&data, &base, 16, 512).unwrap();
    assert_eq!(
        rows[0],
        TokenRow {
            ids: vec![1, 2, 3],
            completion_start: 0
        },
        "a text row is word-level token ids over every position"
    );
    assert_eq!(
        rows[1],
        TokenRow {
            ids: vec![1, 2, 3, 1],
            completion_start: 2
        },
        "prompt and completion tokenize separately and the boundary is the prompt token count"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_tokenizerless_fixture_base_falls_back_to_byte_ids_and_still_keeps_the_boundary() {
    let root = scratch("byte-fallback");
    let base = root.join("base");
    write_base(&base, false);
    assert!(
        !DatasetTokenizer::for_base(&base, 256).unwrap().is_real(),
        "without tokenizer.json the loader must take the named byte fallback"
    );

    let data = root.join("data.jsonl");
    std::fs::write(
        &data,
        format!(
            "{}\n",
            serde_json::json!({"prompt": "ab", "completion": "cd"})
        ),
    )
    .unwrap();
    let rows = load_jsonl_token_dataset(&data, &base, 256, 512).unwrap();
    assert_eq!(rows[0].ids, vec![97, 98, 99, 100]);
    assert_eq!(rows[0].completion_start, 2);
    assert_eq!(
        rows[0].target_mask_where_predicting_a_prompt_token_weighs_zero(),
        vec![0.0, 1.0, 1.0],
        "only positions predicting completion tokens carry loss weight"
    );
    assert_eq!(rows[0].unmasked_targets(), 2);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_base_that_looks_real_without_tokenizer_json_is_an_error_not_a_byte_fallback() {
    let root = scratch("looks-real");
    for leaf in ["tokenizer.model", "tokenizer_config.json"] {
        let base = root.join(leaf.replace('.', "-"));
        write_base(&base, false);
        std::fs::write(base.join(leaf), b"x").unwrap();
        let err = DatasetTokenizer::for_base(&base, 256)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_else(|| panic!("{leaf} beside the weights must refuse the byte fallback"));
        assert!(err.contains(leaf), "{err}");
    }

    let bare = root.join("bare-large-vocab");
    write_base(&bare, false);
    let err = DatasetTokenizer::for_base(&bare, 151936)
        .err()
        .map(|e| e.to_string())
        .expect("a real-sized vocab with no tokenizer files is an incomplete snapshot");
    assert!(err.contains("151936"), "{err}");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_tokenizer_id_at_or_past_the_config_vocab_is_an_error_not_a_silent_drop() {
    let root = scratch("oob-id");
    let base = root.join("base");
    write_base(&base, true);
    let tok = DatasetTokenizer::for_base(&base, 3).unwrap();
    assert_eq!(tok.encode("hello world", 3).unwrap(), vec![1, 2]);
    let err = tok
        .encode("hello again", 3)
        .err()
        .map(|e| e.to_string())
        .expect("id 3 against vocab 3 must be an error, never a dropped token");
    assert!(err.contains("vocab_size 3"), "{err}");

    let data = root.join("data.jsonl");
    std::fs::write(
        &data,
        format!("{}\n", serde_json::json!({"prompt": "hello", "completion": " again"})),
    )
    .unwrap();
    assert!(
        load_jsonl_token_dataset(&data, &base, 3, 512).is_err(),
        "a dataset row carrying an out-of-vocab token must fail the load, not shrink the row"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_row_whose_completion_is_empty_or_truncated_away_is_dropped_not_trained_on_prompt_bytes() {
    let root = scratch("degenerate");
    let base = root.join("base");
    write_base(&base, false);
    let data = root.join("data.jsonl");
    std::fs::write(
        &data,
        format!(
            "{}\n{}\n{}\n",
            serde_json::json!({"prompt": "prompt only", "completion": ""}),
            serde_json::json!({"prompt": "abcdefgh", "completion": "xy"}),
            serde_json::json!({"text": "ok row"}),
        ),
    )
    .unwrap();
    let rows = load_jsonl_token_dataset(&data, &base, 256, 8).unwrap();
    assert_eq!(
        rows.len(),
        1,
        "the empty-completion row and the row whose completion falls past max_seq 8 both have \
         no completion token to predict; got {rows:?}"
    );
    assert_eq!(rows[0].completion_start, 0);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn maximally_wrong_logits_on_prompt_positions_cannot_move_the_masked_loss() {
    let dev = Device::Cpu;
    let (t, vocab) = (5usize, 7usize);
    let row = TokenRow {
        ids: (0..=t as u32).collect(),
        completion_start: 3,
    };
    let mask = row.target_mask_where_predicting_a_prompt_token_weighs_zero();
    assert_eq!(mask, vec![0.0, 0.0, 1.0, 1.0, 1.0]);
    let tgt = Tensor::from_vec(row.ids[1..].to_vec(), t, &dev).unwrap();

    let base_vals: Vec<f32> = (0..t * vocab).map(|i| ((i * 37 % 11) as f32) * 0.3).collect();
    let base = Tensor::from_vec(base_vals.clone(), (t, vocab), &dev).unwrap();
    let reference = ce_mean_over_unmasked_positions(&base, &tgt, &mask)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    let mut wrong_vals = base_vals;
    for pos in 0..2 {
        for v in 0..vocab {
            wrong_vals[pos * vocab + v] = if v as u32 == row.ids[pos + 1] {
                -1e4
            } else {
                1e4
            };
        }
    }
    let wrong = Tensor::from_vec(wrong_vals, (t, vocab), &dev).unwrap();
    let corrupted = ce_mean_over_unmasked_positions(&wrong, &tgt, &mask)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert_eq!(
        reference, corrupted,
        "a prompt position scored maximally wrong changed the loss, so prompt tokens are being \
         trained on"
    );

    let all_ones = vec![1.0f32; t];
    let full = ce_mean_over_unmasked_positions(&base, &tgt, &all_ones)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let plain = candle_nn::loss::cross_entropy(&base.to_dtype(DType::F32).unwrap(), &tgt)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert_eq!(
        full, plain,
        "an all-ones mask must reproduce candle's cross_entropy exactly, so text rows train \
         unchanged"
    );
}

#[test]
fn the_masked_loss_backward_leaves_prompt_position_logit_gradients_at_zero() {
    let dev = Device::Cpu;
    let (t, vocab) = (4usize, 6usize);
    let row = TokenRow {
        ids: (0..=t as u32).collect(),
        completion_start: 2,
    };
    let mask = row.target_mask_where_predicting_a_prompt_token_weighs_zero();
    let tgt = Tensor::from_vec(row.ids[1..].to_vec(), t, &dev).unwrap();
    let vals: Vec<f32> = (0..t * vocab).map(|i| ((i * 13 % 7) as f32) * 0.2).collect();
    let logits = candle_core::Var::from_tensor(&Tensor::from_vec(vals, (t, vocab), &dev).unwrap())
        .unwrap();
    let loss = ce_mean_over_unmasked_positions(logits.as_tensor(), &tgt, &mask).unwrap();
    let grads = loss.backward().unwrap();
    let g: Vec<f32> = grads
        .get(&logits)
        .expect("the masked loss must reach the logits")
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let prompt_grad_amax = g[..vocab].iter().fold(0f32, |m, x| m.max(x.abs()));
    let completion_grad_amax = g[vocab..].iter().fold(0f32, |m, x| m.max(x.abs()));
    assert_eq!(
        prompt_grad_amax, 0.0,
        "a masked position sent gradient into its logits"
    );
    assert!(completion_grad_amax > 0.0);
}
