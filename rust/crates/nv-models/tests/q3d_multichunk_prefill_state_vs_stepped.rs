#![cfg(feature = "wgpu")]

use candle_core::Device;
use nv_models::qwen3_5_dense_wgpu::{Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use nv_weights::WeightLoader;
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn qwen38_snapshot_dir() -> PathBuf {
    let d = std::env::var("NV_QWEN38_DIR").expect(
        "set NV_QWEN38_DIR to the Qwen3.8-27B-NVFP4 snapshot; this gate must never silently skip",
    );
    let p = PathBuf::from(&d);
    assert!(
        p.join("config.json").is_file(),
        "NV_QWEN38_DIR={d} has no config.json"
    );
    p
}

#[test]
#[ignore = "loads the 27B on wgpu twice; set NV_QWEN38_WGPU_TEST=1 -- prefills 224 real tokens once through chunked prefill_tokens and once through 224 prefill_step calls on a fresh engine, dumps every state buffer via debug_state_buffer_words_for_test, and reports per-buffer word-diff counts and zero fractions; the buffers that differ localize the multi-chunk prefill state loss"]
fn q3d_chunked_prefill_state_buffers_match_the_stepped_prefill_of_the_same_tokens() {
    if std::env::var("NV_QWEN38_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_QWEN38_WGPU_TEST != 1");
        return;
    }
    let dir = qwen38_snapshot_dir();
    let cfg = Qwen3_5DenseConfig::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let corpus = std::env::var("NV_STATE_DIFF_TEXT").unwrap_or_else(|_| {
        "It is a truth universally acknowledged, that a single man in possession of a good \
         fortune, must be in want of a wife. However little known the feelings or views of such \
         a man may be on his first entering a neighbourhood, this truth is so well fixed in the \
         minds of the surrounding families, that he is considered as the rightful property of \
         some one or other of their daughters. My dear Mr. Bennet, said his lady to him one day, \
         have you heard that Netherfield Park is let at last? Mr. Bennet replied that he had not. \
         But it is, returned she; for Mrs. Long has just been here, and she told me all about it."
            .into()
    });
    let ids = tok
        .encode(corpus.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();
    let n = 224usize.min(ids.len() / 16 * 16);
    assert!(n >= 32, "corpus too short: {} tokens, need >= 32", ids.len());
    let tokens = &ids[..n];
    eprintln!("STATE-DIFF tokens={n} (chunk width comes from the engine default)");

    let run = |mode: &str, chunked: bool| -> Vec<(u64, Vec<u32>)> {
        let loader = WeightLoader::open_dir(&dir, &Device::Cpu).expect("open weights");
        let mut m =
            Qwen3_5DenseWgpu::from_loader(cfg.clone(), &loader, 512).expect("from_loader");
        drop(loader);
        if chunked {
            let done = m.prefill_tokens(tokens).expect("prefill_tokens");
            assert_eq!(
                done, n,
                "{mode}: prefill_tokens consumed {done} of {n}; pick n a multiple of the chunk"
            );
        } else {
            for &t in tokens {
                m.prefill_step(t).expect("prefill_step");
            }
        }
        m.debug_state_buffer_words_for_test()
    };

    let a = run("chunked", true);
    let b = run("stepped", false);
    assert_eq!(a.len(), b.len(), "state buffer census differs between arms");

    let mut worst: Vec<(usize, u64, usize, f64, f64)> = Vec::new();
    for (i, ((bytes_a, wa), (bytes_b, wb))) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(bytes_a, bytes_b, "buffer {i} size differs");
        let diffs = wa.iter().zip(wb.iter()).filter(|(x, y)| x != y).count();
        let zero_a = wa.iter().filter(|w| **w == 0).count() as f64 / wa.len().max(1) as f64;
        let zero_b = wb.iter().filter(|w| **w == 0).count() as f64 / wb.len().max(1) as f64;
        if diffs > 0 {
            worst.push((i, *bytes_a, diffs, zero_a, zero_b));
        }
    }
    worst.sort_by_key(|w| std::cmp::Reverse(w.2));
    eprintln!(
        "STATE-DIFF buffers={} differing={} (buffer_idx, bytes, diff_words, zero_frac_chunked, zero_frac_stepped):",
        a.len(),
        worst.len()
    );
    for (i, bytes, diffs, za, zb) in worst.iter().take(20) {
        eprintln!("  buf[{i}] {bytes}B diff_words={diffs} zero_chunked={za:.3} zero_stepped={zb:.3}");
    }
    assert!(
        worst.is_empty(),
        "{} of {} state buffers differ between chunked and stepped prefill of identical tokens; \
         the largest-diff buffers above localize the multi-chunk state loss",
        worst.len(),
        a.len()
    );
}
