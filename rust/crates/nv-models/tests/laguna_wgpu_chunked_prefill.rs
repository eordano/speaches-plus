#![cfg(feature = "laguna-wip")]

mod common;
use common::CFG;
use common::have_gpu;
use nv_models::laguna::LagunaConfig;
use nv_models::laguna_wgpu::config::LagunaShapes;
use nv_models::laguna_wgpu::weights::random_host_weights;
use nv_models::laguna_wgpu::{HostWeights, LagunaWgpu};

const MAX_SEQ: usize = 64;

const PROMPT_LENGTHS_COVER_FULL_CHUNKS_A_TAIL_AND_SHORTER_THAN_M: [usize; 3] = [40, 33, 9];

fn config() -> LagunaConfig {
    LagunaConfig::from_hf_json_str(CFG).unwrap()
}

fn weights(cfg: &LagunaConfig, seed: u64) -> HostWeights {
    let shapes = LagunaShapes::derive(cfg, MAX_SEQ).unwrap();
    random_host_weights(&shapes, seed)
}

fn prompt_ids(cfg: &LagunaConfig, n: usize, salt: u32) -> Vec<u32> {
    (0..n)
        .map(|i| ((i as u32 * 7 + salt * 13 + 1) % (cfg.vocab_size as u32 - 1)) + 1)
        .collect()
}

fn per_token_logits(cfg: &LagunaConfig, hw: &HostWeights, ids: &[u32]) -> (u32, Vec<f32>) {
    let mut m = LagunaWgpu::new(cfg.clone(), hw, MAX_SEQ).expect("build per-token model");
    let (last, rest) = ids.split_last().expect("non-empty prompt");
    for t in rest {
        m.prefill_step(*t).expect("per-token prefill step");
    }
    m.decode_step_logits(*last).expect("decode")
}

fn chunked_logits(
    cfg: &LagunaConfig,
    hw: &HostWeights,
    ids: &[u32],
) -> (u32, Vec<f32>, usize, usize) {
    let mut m = LagunaWgpu::new(cfg.clone(), hw, MAX_SEQ).expect("build chunked model");
    let chunk = m.prefill_chunk_len();
    assert!(
        chunk >= 2,
        "LagunaWgpu::prefill_chunk_len() is {chunk}: the decoder still replays one token per \
         command buffer, which is exactly what this suite exists to refuse. Do not relax this \
         bound -- a chunk of 1 is the per-token path wearing the chunked path's name."
    );
    let (last, rest) = ids.split_last().expect("non-empty prompt");
    let done = m.prefill_tokens(rest).expect("chunked prefill");
    for t in &rest[done..] {
        m.prefill_step(*t).expect("tail prefill step");
    }
    let (tok, logits) = m.decode_step_logits(*last).expect("decode");
    (tok, logits, done, chunk)
}

#[test]
fn chunked_prefill_logits_are_bit_identical_to_per_token_replay() {
    if !have_gpu() {
        panic!("needs a wgpu adapter; a skipped identity proof reads as a passed one");
    }
    let cfg = config();
    let hw = weights(&cfg, 0x1a90_a5ee_d000_0001);
    let mut seen: Vec<(usize, u32, usize)> = Vec::new();
    for (i, n) in PROMPT_LENGTHS_COVER_FULL_CHUNKS_A_TAIL_AND_SHORTER_THAN_M
        .into_iter()
        .enumerate()
    {
        let ids = prompt_ids(&cfg, n, i as u32);
        let (want_tok, want) = per_token_logits(&cfg, &hw, &ids);
        let (got_tok, got, done, chunk) = chunked_logits(&cfg, &hw, &ids);
        assert_eq!(want.len(), got.len(), "logit width changed at n={n}");
        let diff = want
            .iter()
            .zip(&got)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diff,
            0,
            "n={n} (chunk={chunk}, {done} of {} prompt tokens went through prefill_tokens): {diff} \
             of {} logits differ bit-for-bit between chunked prefill and per-token replay. The \
             chunked arm replays the SAME per-token passes inside one submission, updating \
             tok/pos/step-slots/step-uniform via buffer copies from the pf list between pass \
             blocks -- if this fires, the copy offsets into the 16-byte StepUniform records (tok \
             at i*16, pos at i*16+4, whole record into step and uni) no longer match the layout \
             the host wrote, and every token after the first computes at the wrong position, with \
             the wrong id, or with a stale sliding-window start. Do NOT relax this to argmax or a \
             tolerance: a tiny model's logits barely depend on context, so both weaker oracles \
             pass while every KV entry is wrong.",
            n - 1,
            want.len()
        );
        assert_eq!(
            want_tok, got_tok,
            "argmax token differs at n={n} (replay {want_tok} vs chunked {got_tok}); the bit \
             comparison above is the gate that bites first"
        );
        seen.push((n, want_tok, done));
    }
    assert!(
        seen.iter().all(|(_, _, done)| *done > 0),
        "prefill_tokens consumed nothing at some length, so the chunked path never ran: {seen:?}"
    );
    let base_ids = prompt_ids(&cfg, 40, 0);
    let mut poked_ids = base_ids.clone();
    let poke_at = base_ids.len() - 2;
    poked_ids[poke_at] = (poked_ids[poke_at] % (cfg.vocab_size as u32 - 1)) + 1;
    assert_ne!(
        poked_ids[poke_at], base_ids[poke_at],
        "the poke must actually change the token"
    );
    let (_, base_logits) = per_token_logits(&cfg, &hw, &base_ids);
    let (_, poked_logits) = per_token_logits(&cfg, &hw, &poked_ids);
    assert!(
        base_logits
            .iter()
            .zip(&poked_logits)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "changing one interior prompt token changed no logit bit, so the final decode never reads \
         the prefill state and the bit-identity gate above is comparing constants; this tiny \
         model cannot certify the chunked path"
    );
    eprintln!("[lgw-chunked-prefill] (len, argmax, tokens via prefill_tokens) = {seen:?}");
}

#[test]
fn chunked_prefill_is_invariant_to_slice_size() {
    if !have_gpu() {
        panic!("needs a wgpu adapter; a skipped invariance proof reads as a passed one");
    }
    let cfg = config();
    let hw = weights(&cfg, 0x1a90_a5ee_d000_0002);
    let ids = prompt_ids(&cfg, 37, 3);
    let mut rows: Vec<(usize, u32, Vec<f32>)> = Vec::new();
    for m in [4usize, 8, 16] {
        let mut model = LagunaWgpu::new(cfg.clone(), &hw, MAX_SEQ).expect("build");
        assert!(
            model.prefill_chunk_len() >= 2,
            "chunked prefill absent at m={m}"
        );
        let (last, rest) = ids.split_last().expect("non-empty");
        let mut done = 0usize;
        while done < rest.len() {
            let take = m.min(rest.len() - done);
            if take < 2 {
                break;
            }
            let got = model
                .prefill_tokens(&rest[done..done + take])
                .expect("chunked prefill");
            assert!(got > 0, "prefill_tokens made no progress at m={m}");
            done += got;
        }
        for t in &rest[done..] {
            model.prefill_step(*t).expect("tail");
        }
        let (tok, logits) = model.decode_step_logits(*last).expect("decode");
        rows.push((m, tok, logits));
    }
    let (m0, t0, l0) = &rows[0];
    for (m, t, l) in &rows[1..] {
        let diff = l0
            .iter()
            .zip(l)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diff, 0,
            "feeding the same prompt in {m0}-token and {m}-token slices produced {diff} differing \
             logits (argmax {t0} vs {t}). Chunk size is bookkeeping, not semantics."
        );
    }
    eprintln!(
        "[lgw-chunk-invariance] argmax {t0} identical across slice sizes {:?}",
        rows.iter().map(|(m, _, _)| *m).collect::<Vec<_>>()
    );
}

#[test]
fn prefill_entrypoint_routes_through_chunks_and_matches_pure_replay() {
    if !have_gpu() {
        panic!("needs a wgpu adapter; a skipped routing proof reads as a passed one");
    }
    let cfg = config();
    let hw = weights(&cfg, 0x1a90_a5ee_d000_0003);
    let ids = prompt_ids(&cfg, 23, 7);

    let mut replay = LagunaWgpu::new(cfg.clone(), &hw, MAX_SEQ).expect("build replay model");
    let (last, rest) = ids.split_last().expect("non-empty");
    for t in rest {
        replay.prefill_step(*t).expect("replay prefill step");
    }
    let want = replay.decode_step(*last).expect("replay decode");

    let mut chunked = LagunaWgpu::new(cfg.clone(), &hw, MAX_SEQ).expect("build chunked model");
    assert!(
        chunked.prefill_chunk_len() >= 2,
        "prefill() cannot route through chunks when prefill_chunk_len() is {}",
        chunked.prefill_chunk_len()
    );
    let got = chunked.prefill(&ids).expect("prefill entrypoint");
    assert_eq!(
        chunked.current_pos(),
        ids.len(),
        "prefill() must leave pos at the full prompt length"
    );
    assert_eq!(
        want, got,
        "prefill() (chunk-routed) and pure per-token replay disagree on the next token"
    );
}
