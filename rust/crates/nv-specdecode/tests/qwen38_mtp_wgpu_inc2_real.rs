#![cfg(feature = "wgpu")]

mod hub_dirs;

use nv_config::measure::Measurement;
use nv_models::qwen3_5_dense_wgpu::{Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use nv_specdecode::qwen38_mtp::{
    mtp_chain_depth_from_env, run_mtp_verify_round,
    Q38_WGPU_BATCHED_VERIFY_VS_M1_MAX_ABS_LOGIT_DRIFT_EMPIRICAL_PIN,
};
use nv_weights::WeightLoader;
use std::path::PathBuf;
use std::time::Instant;

const GATE_ENV: &str = "NV_Q38_MTP_INC2_TEST";
const REPO_DIR: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const MODEL_LABEL: &str = "unsloth/Qwen3.8-27B-NVFP4";
const CORPUS_ENV: &str = "NV_PPL_CORPUS";
const CORPUS_DEFAULT: &str = "/home/user/tmp/wf/quant/corpus/wiki.txt";
const CTX_PREFILL_64_AMORTIZES_STATE_WARMUP_BEFORE_SCORING: usize = 64;
const SCORED_POSITIONS_DEFAULT: usize = 128;
const DECODE_TOKENS_256_THE_SCOREBOARD_CONTEXT: usize = 256;
const MAX_SEQ_512_HOLDS_PROMPT_PLUS_256_DECODE_PLUS_VERIFY_ROWS: usize = 512;
const ACCEPT_AB_TOLERANCE_5_POINTS_COVERS_BINOMIAL_NOISE_AT_A_FEW_HUNDRED_DRAFTS: f64 = 0.05;
const PPL_MATCH_HALF_A_PUBLICATION_DIGIT: f64 = 0.005;

fn require_gate() {
    if std::env::var(GATE_ENV).as_deref() != Ok("1") {
        panic!("set {GATE_ENV}=1 to run the real-weights qwen3.8 wgpu MTP increment-2 gates");
    }
}

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
        if !d.is_empty() {
            let p = PathBuf::from(&d);
            assert!(p.is_dir(), "NV_QWEN38_DIR={d} is not a directory");
            return p;
        }
    }
    hub_dirs::snapshot(
        REPO_DIR,
        &["config.json", "tokenizer.json", "model_mtp.safetensors"],
    )
    .unwrap_or_else(|| {
        panic!(
            "no hydrated {REPO_DIR} snapshot carrying the MTP shard; set NV_QWEN38_DIR (this \
             gated suite refuses to vacuously pass)"
        )
    })
}

fn adapter_name() -> String {
    nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip")
        .info
        .name
        .clone()
}

fn boot(dir: &PathBuf, attach_mtp: bool) -> (Qwen3_5DenseWgpu, tokenizers::Tokenizer) {
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("qwen3.8 dense config");
    let loader = WeightLoader::open_dir(dir, &candle_core::Device::Cpu).expect("open weights");
    let mut m = Qwen3_5DenseWgpu::from_loader(
        cfg,
        &loader,
        MAX_SEQ_512_HOLDS_PROMPT_PLUS_256_DECODE_PLUS_VERIFY_ROWS,
    )
    .expect("boot Qwen3.8-27B-NVFP4 on wgpu");
    if attach_mtp {
        m.mtp_attach(&loader).expect("attach the mtp.* drafter head");
    }
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    (m, tok)
}

fn corpus_ids(tok: &tokenizers::Tokenizer, need: usize) -> Vec<u32> {
    let p = std::env::var(CORPUS_ENV).unwrap_or_else(|_| CORPUS_DEFAULT.to_string());
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!("{CORPUS_ENV}={p} unreadable ({e}); the ppl/drift gates need the pinned corpus")
    });
    let ids: Vec<u32> = tok
        .encode(text.as_str(), false)
        .expect("encode corpus")
        .get_ids()
        .to_vec();
    assert!(
        ids.len() >= need,
        "corpus {p} tokenizes to {} tokens, need {need}",
        ids.len()
    );
    ids
}

fn chat_prompt_ids(tok: &tokenizers::Tokenizer) -> Vec<u32> {
    let q = "Explain, in a few sentences, why the sky appears blue.";
    let text =
        format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    tok.encode(text.as_str(), false)
        .expect("encode chat prompt")
        .get_ids()
        .to_vec()
}

fn prefill(m: &mut Qwen3_5DenseWgpu, tokens: &[u32]) {
    let done = m.prefill_tokens(tokens).expect("chunked prefill");
    for &t in &tokens[done..] {
        m.prefill_step(t).expect("prefill step");
    }
}

fn nll_of(row: &[f32], target: usize) -> f64 {
    let mx = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
    assert!(mx.is_finite(), "non-finite logits row");
    let mut z = 0f64;
    for &v in row {
        z += ((v - mx) as f64).exp();
    }
    (mx as f64 + z.ln()) - row[target] as f64
}

fn measure(instrument: &str, tokens: usize, steps: usize, warmup: usize, value: f64, unit: &str) -> Measurement {
    Measurement {
        instrument: instrument.to_string(),
        model_at_rev: MODEL_LABEL.to_string(),
        backend: "wgpu".to_string(),
        device: adapter_name(),
        batch: 1,
        tokens,
        steps,
        warmup,
        value,
        unit: unit.to_string(),
        extras: Vec::new(),
    }
}

#[test]
#[ignore = "boots the 22.6 GB Qwen3.8-27B-NVFP4 wgpu decoder with NV_Q3D_VERIFY_M=32 so the \
            dedicated 32-row verify graph records next to the 512-row prefill graph, then \
            forwards the same k+1 chain through both from the same committed state; every \
            returned logit must match bit-for-bit, proving the pad rows were dead weight; set \
            NV_Q38_MTP_INC2_TEST=1"]
fn real_qwen38_verify32_rows_bit_match_the_padded_512_graph() {
    require_gate();
    let dir = snapshot_dir();
    std::env::set_var(
        nv_models::qwen3_5_dense_wgpu::VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM,
        "32",
    );
    let (mut m, tok) = boot(&dir, false);
    std::env::remove_var(
        nv_models::qwen3_5_dense_wgpu::VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM,
    );
    assert!(
        m.verify_chain_rides_dedicated_graph(),
        "NV_Q3D_VERIFY_M=32 must record the dedicated verify graph on the coop route; without \
         it this A/B compares the padded graph to itself and proves nothing"
    );
    assert_eq!(m.verify_max_rows(), 32, "chain capacity must be the dedicated graph's rows");
    assert_eq!(m.prefill_chunk_len(), 512, "the prefill graph must keep its serving-default M");
    let k1 = mtp_chain_depth_from_env() + 1;
    let rounds = 4usize;
    let ids = corpus_ids(&tok, CTX_PREFILL_64_AMORTIZES_STATE_WARMUP_BEFORE_SCORING + rounds * k1);
    prefill(&mut m, &ids[..CTX_PREFILL_64_AMORTIZES_STATE_WARMUP_BEFORE_SCORING]);
    let mut off = CTX_PREFILL_64_AMORTIZES_STATE_WARMUP_BEFORE_SCORING;
    for round in 0..rounds {
        let batch = ids[off..off + k1].to_vec();
        let (a_tok, a_log) = m.verify_chain_logits(&batch).expect("dedicated-graph chain");
        m.advance(0).expect("rollback the dedicated-graph forward");
        let (b_tok, b_log) = m
            .debug_verify_chain_logits_via_the_padded_prefill_graph_for_test(&batch)
            .expect("padded-graph chain");
        assert_eq!(a_log.len(), b_log.len(), "both arms must return live-rows x vocab logits");
        let mut max_abs = 0f32;
        let mut diff_bits = 0usize;
        for (x, y) in a_log.iter().zip(&b_log) {
            if x.to_bits() != y.to_bits() {
                diff_bits += 1;
                max_abs = max_abs.max((x - y).abs());
            }
        }
        assert_eq!(
            a_tok, b_tok,
            "round {round}: verify argmax rows diverged between the 32-row dedicated graph and \
             the 512-row padded graph"
        );
        assert_eq!(
            diff_bits, 0,
            "round {round}: {diff_bits} of {} logit values differ (max abs {max_abs:e}) between \
             the 32-row dedicated verify graph and the 512-row padded graph; the coop gemm \
             route table is asserted bit-equal across m_alloc bases and every other kernel is \
             per-row, so a nonzero count here names a chunk-bracket-dependent kernel and drops \
             this contract to the inc2 tolerance class \
             ({Q38_WGPU_BATCHED_VERIFY_VS_M1_MAX_ABS_LOGIT_DRIFT_EMPIRICAL_PIN} abs pin)",
            a_log.len()
        );
        eprintln!("[verify32-ab] round {round}: {} logits bit-identical across both graphs", a_log.len());
        m.advance(batch.len()).expect("commit every verified row");
        off += k1;
    }
    let timed_batch = ids[off - k1..off].to_vec();
    let mut vf_ms: Vec<f64> = Vec::new();
    let mut pad_ms: Vec<f64> = Vec::new();
    for _ in 0..12 {
        let t = Instant::now();
        let _ = m.verify_chain_logits(&timed_batch).expect("dedicated timed");
        vf_ms.push(t.elapsed().as_secs_f64() * 1e3);
        m.advance(0).expect("rollback");
        let t = Instant::now();
        let _ = m
            .debug_verify_chain_logits_via_the_padded_prefill_graph_for_test(&timed_batch)
            .expect("padded timed");
        pad_ms.push(t.elapsed().as_secs_f64() * 1e3);
        m.advance(0).expect("rollback");
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite wall"));
        v[v.len() / 2]
    };
    eprintln!(
        "[verify32-ab] verify-only wall at pos {}: dedicated 32-row {:.1} ms vs padded 512-row \
         {:.1} ms per k+1={k1} chain (logits readback included in both)",
        m.current_pos(),
        med(&mut vf_ms),
        med(&mut pad_ms),
    );
}

#[test]
#[ignore = "boots the 22.6 GB Qwen3.8-27B-NVFP4 wgpu decoder with NV_Q3D_VERIFY_M=4 so the \
            k+1-row verify chain rides the small-m legacy arms (nvfp4 m-row gemv, fp8 per-slot \
            replay, slab attention) beside the coop 512-row prefill graph; gates: verify argmax \
            rows equal the padded-512 graph's and logit drift stays under the inc2 empirical \
            pin; also prints the verify-only wall A/B; set NV_Q38_MTP_INC2_TEST=1"]
fn real_qwen38_verify4_legacy_rows_argmax_match_and_drift_pin_vs_padded_512() {
    require_gate();
    let dir = snapshot_dir();
    std::env::set_var(
        nv_models::qwen3_5_dense_wgpu::VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM,
        "4",
    );
    let (mut m, tok) = boot(&dir, false);
    std::env::remove_var(
        nv_models::qwen3_5_dense_wgpu::VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM,
    );
    assert!(
        m.verify_chain_rides_dedicated_graph(),
        "NV_Q3D_VERIFY_M=4 must record the dedicated legacy verify graph"
    );
    assert_eq!(m.verify_max_rows(), 4, "chain capacity must be the dedicated graph's rows");
    assert_eq!(m.prefill_chunk_len(), 512, "the prefill graph must keep its serving-default M");
    let k1 = 4usize.min(mtp_chain_depth_from_env() + 1);
    let rounds = 8usize;
    let ids = corpus_ids(&tok, CTX_PREFILL_64_AMORTIZES_STATE_WARMUP_BEFORE_SCORING + rounds * k1);
    prefill(&mut m, &ids[..CTX_PREFILL_64_AMORTIZES_STATE_WARMUP_BEFORE_SCORING]);
    let mut off = CTX_PREFILL_64_AMORTIZES_STATE_WARMUP_BEFORE_SCORING;
    let mut max_abs_all = 0f32;
    for round in 0..rounds {
        let batch = ids[off..off + k1].to_vec();
        let (a_tok, a_log) = m.verify_chain_logits(&batch).expect("dedicated-graph chain");
        m.advance(0).expect("rollback the dedicated-graph forward");
        let (b_tok, b_log) = m
            .debug_verify_chain_logits_via_the_padded_prefill_graph_for_test(&batch)
            .expect("padded-graph chain");
        let mut max_abs = 0f32;
        for (x, y) in a_log.iter().zip(&b_log) {
            max_abs = max_abs.max((x - y).abs());
        }
        max_abs_all = max_abs_all.max(max_abs);
        assert_eq!(
            a_tok, b_tok,
            "round {round}: verify argmax rows diverged between the 4-row legacy verify graph \
             and the 512-row coop graph (max abs logit drift {max_abs:e}); a greedy stream \
             committed off the legacy rows would diverge from the shipping stream here, so the \
             legacy vf arm is not identity-safe on this checkpoint"
        );
        assert!(
            max_abs <= Q38_WGPU_BATCHED_VERIFY_VS_M1_MAX_ABS_LOGIT_DRIFT_EMPIRICAL_PIN,
            "round {round}: legacy-vs-coop verify logit drift {max_abs} exceeds the inc2 \
             empirical pin {Q38_WGPU_BATCHED_VERIFY_VS_M1_MAX_ABS_LOGIT_DRIFT_EMPIRICAL_PIN}"
        );
        eprintln!(
            "[verify4-ab] round {round}: argmax rows match, max abs logit drift {max_abs:e}"
        );
        m.advance(batch.len()).expect("commit every verified row");
        off += k1;
    }
    let timed_batch = ids[off - k1..off].to_vec();
    let mut vf_ms: Vec<f64> = Vec::new();
    let mut pad_ms: Vec<f64> = Vec::new();
    for _ in 0..12 {
        let t = Instant::now();
        let _ = m.verify_chain_logits(&timed_batch).expect("dedicated timed");
        vf_ms.push(t.elapsed().as_secs_f64() * 1e3);
        m.advance(0).expect("rollback");
        let t = Instant::now();
        let _ = m
            .debug_verify_chain_logits_via_the_padded_prefill_graph_for_test(&timed_batch)
            .expect("padded timed");
        pad_ms.push(t.elapsed().as_secs_f64() * 1e3);
        m.advance(0).expect("rollback");
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite wall"));
        v[v.len() / 2]
    };
    eprintln!(
        "[verify4-ab] verify-only wall at pos {}: dedicated legacy 4-row {:.1} ms vs padded \
         512-row {:.1} ms per k+1={k1} chain (max abs drift over {rounds} rounds {max_abs_all:e})",
        m.current_pos(),
        med(&mut vf_ms),
        med(&mut pad_ms),
    );
}

#[test]
#[ignore = "boots the 22.6 GB Qwen3.8-27B-NVFP4 wgpu decoder with NV_Q3D_VERIFY_M=4 and \
            NV_Q3D_VF_ATTN_FLASH=loop so the k+1-row verify chain rides the per-row flash-loop \
            attention over the fp8 KV cache instead of the slab; gates: verify argmax rows \
            equal the padded-512 coop graph's and logit drift stays under the inc2 empirical \
            pin -- the drift class the slab arm itself passes against the same reference; set \
            NV_Q38_MTP_INC2_TEST=1"]
fn real_qwen38_verify4_vflash_loop_argmax_match_and_drift_pin_vs_padded_512() {
    require_gate();
    let dir = snapshot_dir();
    std::env::set_var(
        nv_models::qwen3_5_dense_wgpu::VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM,
        "4",
    );
    std::env::set_var(
        nv_models::qwen3_5_dense_wgpu::VF_ATTN_FLASH_ENV_DEFAULT_GATE_AT_THE_MEASURED_KNEE_SET_0_FOR_THE_SCORES_SLAB_ARM,
        "loop",
    );
    let (mut m, tok) = boot(&dir, false);
    std::env::remove_var(
        nv_models::qwen3_5_dense_wgpu::VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM,
    );
    std::env::remove_var(
        nv_models::qwen3_5_dense_wgpu::VF_ATTN_FLASH_ENV_DEFAULT_GATE_AT_THE_MEASURED_KNEE_SET_0_FOR_THE_SCORES_SLAB_ARM,
    );
    assert!(
        m.verify_chain_rides_dedicated_graph(),
        "NV_Q3D_VERIFY_M=4 must record the dedicated legacy verify graph; without it the flash \
         arm under test never records and this gate proves nothing"
    );
    assert_eq!(m.verify_max_rows(), 4, "chain capacity must be the dedicated graph's rows");
    let k1 = 4usize.min(mtp_chain_depth_from_env() + 1);
    let rounds = 8usize;
    let ids = corpus_ids(&tok, CTX_PREFILL_64_AMORTIZES_STATE_WARMUP_BEFORE_SCORING + rounds * k1);
    prefill(&mut m, &ids[..CTX_PREFILL_64_AMORTIZES_STATE_WARMUP_BEFORE_SCORING]);
    let mut off = CTX_PREFILL_64_AMORTIZES_STATE_WARMUP_BEFORE_SCORING;
    let mut max_abs_all = 0f32;
    for round in 0..rounds {
        let batch = ids[off..off + k1].to_vec();
        let (a_tok, a_log) = m.verify_chain_logits(&batch).expect("vflash-loop chain");
        m.advance(0).expect("rollback the vflash-loop forward");
        let (b_tok, b_log) = m
            .debug_verify_chain_logits_via_the_padded_prefill_graph_for_test(&batch)
            .expect("padded-graph chain");
        let mut max_abs = 0f32;
        for (x, y) in a_log.iter().zip(&b_log) {
            max_abs = max_abs.max((x - y).abs());
        }
        max_abs_all = max_abs_all.max(max_abs);
        assert_eq!(
            a_tok, b_tok,
            "round {round}: verify argmax rows diverged between the vflash-loop 4-row verify \
             graph and the 512-row coop graph (max abs logit drift {max_abs:e}); a greedy \
             stream committed off the flash rows would diverge from the shipping stream here, \
             so the loop arm is not drift-safe on this checkpoint"
        );
        assert!(
            max_abs <= Q38_WGPU_BATCHED_VERIFY_VS_M1_MAX_ABS_LOGIT_DRIFT_EMPIRICAL_PIN,
            "round {round}: vflash-loop-vs-coop verify logit drift {max_abs} exceeds the inc2 \
             empirical pin {Q38_WGPU_BATCHED_VERIFY_VS_M1_MAX_ABS_LOGIT_DRIFT_EMPIRICAL_PIN}"
        );
        eprintln!(
            "[verify4-vflash-ab] round {round}: argmax rows match, max abs logit drift {max_abs:e}"
        );
        m.advance(batch.len()).expect("commit every verified row");
        off += k1;
    }
    eprintln!(
        "[verify4-vflash-ab] {rounds} rounds at pos {}: argmax rows match throughout, max abs \
         logit drift {max_abs_all:e} (pin {Q38_WGPU_BATCHED_VERIFY_VS_M1_MAX_ABS_LOGIT_DRIFT_EMPIRICAL_PIN})",
        m.current_pos(),
    );
}

#[test]
#[ignore = "boots the 22.6 GB Qwen3.8-27B-NVFP4 wgpu decoder and teacher-forces the pinned \
            corpus through the k+1-row batched verify graph AND the M=1 decode path from the \
            same committed state per chunk; gates: ppl match to publication digits and the \
            empirical max-abs logit drift pin; set NV_Q38_MTP_INC2_TEST=1"]
fn real_qwen38_inc2_teacher_forced_ppl_match_and_logit_drift_pin() {
    require_gate();
    let dir = snapshot_dir();
    let (mut m, tok) = boot(&dir, false);
    let k = mtp_chain_depth_from_env();
    let chunk = std::env::var("NV_Q38_INC2_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&c| c >= 1)
        .unwrap_or(k + 1);
    let scored = std::env::var("NV_Q38_INC2_POSITIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(SCORED_POSITIONS_DEFAULT);
    let scored = scored - scored % chunk;
    let ctx = CTX_PREFILL_64_AMORTIZES_STATE_WARMUP_BEFORE_SCORING;
    let ids = corpus_ids(&tok, ctx + scored + 1);
    prefill(&mut m, &ids[..ctx]);
    assert_eq!(m.current_pos(), ctx, "prefill must commit the context");
    assert!(
        m.verify_max_rows() >= chunk,
        "verify graph rows {} cannot hold a k+1={chunk} chunk",
        m.verify_max_rows()
    );

    let vocab_probe = m.verify_chain_logits(&ids[ctx..ctx + chunk]).expect("probe chunk");
    m.advance(0).expect("probe rollback");
    let vocab = vocab_probe.1.len() / chunk;

    let mut nll_batched = 0f64;
    let mut nll_m1 = 0f64;
    let mut max_drift = 0f32;
    let mut argmax_disagreements = 0usize;
    let mut row_drift_max = vec![0f32; chunk];
    let mut row_flips = vec![0usize; chunk];
    let mut p = ctx;
    while p < ctx + scored {
        let toks = &ids[p..p + chunk];
        let (_amax, blogits) = m.verify_chain_logits(toks).expect("batched chunk");
        assert_eq!(blogits.len(), chunk * vocab, "batched chunk logits shape");
        m.advance(0)
            .expect("advance(0) must roll back the batched rows without an M=1 replay");
        assert_eq!(m.current_pos(), p, "rollback must not move pos");
        for (j, &t) in toks.iter().enumerate() {
            let (m1_top, m1_logits) = m.decode_step_logits(t).expect("m1 step");
            let brow = &blogits[j * vocab..(j + 1) * vocab];
            let target = ids[p + j + 1] as usize;
            nll_batched += nll_of(brow, target);
            nll_m1 += nll_of(&m1_logits, target);
            let mut row_drift = 0f32;
            let mut b_top = 0usize;
            for v in 1..vocab {
                if brow[v] > brow[b_top] {
                    b_top = v;
                }
            }
            for v in 0..vocab {
                row_drift = row_drift.max((brow[v] - m1_logits[v]).abs());
            }
            max_drift = max_drift.max(row_drift);
            row_drift_max[j] = row_drift_max[j].max(row_drift);
            row_flips[j] += (b_top as u32 != m1_top) as usize;
            argmax_disagreements += (b_top as u32 != m1_top) as usize;
        }
        p += chunk;
    }
    let ppl_batched = (nll_batched / scored as f64).exp();
    let ppl_m1 = (nll_m1 / scored as f64).exp();

    measure("q38_mtp_inc2_ppl", scored, scored, 0, ppl_m1, "ppl")
        .extra("arm", "m1_decode")
        .extra("ctx", ctx.to_string())
        .emit();
    measure("q38_mtp_inc2_ppl", scored, scored, 0, ppl_batched, "ppl")
        .extra("arm", "batched_verify")
        .extra("ctx", ctx.to_string())
        .emit();
    let fmt_f32s = |v: &[f32]| {
        v.iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>()
            .join("/")
    };
    let fmt_us = |v: &[usize]| {
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join("/")
    };
    measure("q38_mtp_inc2_logit_drift", scored, scored, 0, max_drift as f64, "max_abs_logit")
        .extra("argmax_disagreements", argmax_disagreements.to_string())
        .extra("chunk", chunk.to_string())
        .extra("prefill_m_rows", m.verify_max_rows().to_string())
        .extra("row_drift_max", fmt_f32s(&row_drift_max))
        .extra("row_flips", fmt_us(&row_flips))
        .emit();

    assert!(
        max_drift <= Q38_WGPU_BATCHED_VERIFY_VS_M1_MAX_ABS_LOGIT_DRIFT_EMPIRICAL_PIN,
        "max abs logit drift {max_drift} between the k+1-row batched verify and M=1 decode \
         exceeds the pin {Q38_WGPU_BATCHED_VERIFY_VS_M1_MAX_ABS_LOGIT_DRIFT_EMPIRICAL_PIN}. \
         This pin is an EMPIRICAL bound (measured 4.60 max over chunk 1/4 x rows 4/16 on \
         Qwen3.8-27B-NVFP4, RTX PRO 6000 Blackwell, 128 positions at ctx 64), not a theorem: \
         if kernels changed intentionally, re-measure here and move the pin with the basis \
         recorded in the commit"
    );
    assert!(
        (ppl_batched - ppl_m1).abs() <= PPL_MATCH_HALF_A_PUBLICATION_DIGIT,
        "teacher-forced ppl through the batched verify rows ({ppl_batched:.4}) vs M=1 decode \
         ({ppl_m1:.4}) diverged past publication digits; the M-row drift is NOT immaterial. \
         This is the known source-2 M-row attention-compute drift (a single live row through \
         the M-row graph already carries it), and it BLOCKS flipping the serving default to \
         the increment-2 batched commit until the M-row attention campaign lands"
    );
}

struct ArmStats {
    drafted: usize,
    accepted: usize,
    emitted: Vec<u32>,
    rounds: usize,
    reforwards: usize,
    secs: f64,
}

fn run_mtp_arm(
    m: &mut Qwen3_5DenseWgpu,
    prompt: &[u32],
    n: usize,
    k: usize,
    replay_commit: bool,
    clairvoyant: Option<&[u32]>,
) -> ArmStats {
    m.reset().expect("reset");
    prefill(m, &prompt[..prompt.len() - 1]);
    let mut last = m
        .decode_step(prompt[prompt.len() - 1])
        .expect("anchor decode step");
    let mut out = vec![last];
    let mut st = ArmStats {
        drafted: 0,
        accepted: 0,
        emitted: Vec::new(),
        rounds: 0,
        reforwards: 0,
        secs: 0.0,
    };
    let t0 = Instant::now();
    while out.len() < n {
        let rows = m.verify_max_rows();
        if m.current_pos() + rows.max(m.prefill_chunk_len())
            > MAX_SEQ_512_HOLDS_PROMPT_PLUS_256_DECODE_PLUS_VERIFY_ROWS
        {
            last = m.decode_step(last).expect("tail decode step");
            out.push(last);
            continue;
        }
        let want = k.min(rows - 1);
        let own = m.mtp_draft_round(last, want).expect("mtp draft round");
        let drafts: Vec<u32> = match clairvoyant {
            Some(reference) => (0..want)
                .map(|j| *reference.get(out.len() + j).unwrap_or(&own[j.min(own.len() - 1)]))
                .collect(),
            None => own,
        };
        let r = run_mtp_verify_round(m, last, &drafts, replay_commit).expect("inc2 round");
        m.mtp_post_verify(&r.batch[1..r.accept.commit_len])
            .expect("mtp post verify");
        st.rounds += 1;
        st.drafted += drafts.len();
        st.accepted += r.accept.draft_accepted;
        st.reforwards += r.prefix_reforwarded_batched as usize;
        out.extend_from_slice(&r.emitted);
        last = *out.last().expect("emitted non-empty");
    }
    st.secs = t0.elapsed().as_secs_f64();
    st.emitted = out;
    st
}

fn accept_rate(st: &ArmStats) -> f64 {
    if st.drafted == 0 {
        return 0.0;
    }
    st.accepted as f64 / st.drafted as f64
}

#[test]
#[ignore = "acceptance A/B on real weights, one boot, arms interleaved: increment-1 replay \
            commit vs increment-2 batched commit, identical clairvoyant drafts from a pinned \
            M=1 greedy reference; runs the A-vs-A null first; set NV_Q38_MTP_INC2_TEST=1"]
fn real_qwen38_inc2_acceptance_ab_replay_vs_batched() {
    require_gate();
    let dir = snapshot_dir();
    let (mut m, tok) = boot(&dir, true);
    let k = mtp_chain_depth_from_env();
    let prompt = chat_prompt_ids(&tok);
    let n = 128usize;

    m.reset().expect("reset");
    prefill(&mut m, &prompt[..prompt.len() - 1]);
    let mut last = m
        .decode_step(prompt[prompt.len() - 1])
        .expect("anchor decode step");
    let mut reference = vec![last];
    while reference.len() < n {
        last = m.decode_step(last).expect("reference decode step");
        reference.push(last);
    }

    let null_a = run_mtp_arm(&mut m, &prompt, n, k, true, Some(&reference));
    let null_b = run_mtp_arm(&mut m, &prompt, n, k, true, Some(&reference));
    assert_eq!(
        (null_a.drafted, null_a.accepted, &null_a.emitted),
        (null_b.drafted, null_b.accepted, &null_b.emitted),
        "A-vs-A null: two identical replay-commit runs disagreed; the harness is not \
         deterministic and no A/B below can be believed"
    );

    let replay = run_mtp_arm(&mut m, &prompt, n, k, true, Some(&reference));
    let batched = run_mtp_arm(&mut m, &prompt, n, k, false, Some(&reference));
    let ra = accept_rate(&replay);
    let ba = accept_rate(&batched);
    let agree = replay
        .emitted
        .iter()
        .zip(&batched.emitted)
        .take_while(|(a, b)| a == b)
        .count();

    for (arm, st, rate) in [("replay_inc1", &replay, ra), ("batched_inc2", &batched, ba)] {
        measure("q38_mtp_inc2_accept_ab", n, st.rounds, 0, rate, "accept_rate")
            .extra("arm", arm)
            .extra("k", k.to_string())
            .extra("drafted", st.drafted.to_string())
            .extra("accepted", st.accepted.to_string())
            .extra("reforward_rounds", st.reforwards.to_string())
            .extra("stream_agree_prefix", agree.to_string())
            .emit();
    }

    assert!(
        batched.reforwards > 0,
        "the batched arm never took a partial accept, so the increment-2 commit mechanism was \
         never exercised and this A/B is vacuous"
    );
    assert!(
        ba >= ra - ACCEPT_AB_TOLERANCE_5_POINTS_COVERS_BINOMIAL_NOISE_AT_A_FEW_HUNDRED_DRAFTS,
        "increment-2 batched-commit acceptance {ba:.3} regressed more than \
         {ACCEPT_AB_TOLERANCE_5_POINTS_COVERS_BINOMIAL_NOISE_AT_A_FEW_HUNDRED_DRAFTS} below the \
         increment-1 replay acceptance {ra:.3} on identical drafts"
    );
}

#[test]
#[ignore = "decode throughput A/B at 256 generated tokens on real weights, one boot, arms \
            interleaved OFF/ON/OFF/ON with an OFF-vs-OFF null; emits NV-MEASURE lines; run \
            under the GPU flock via nvk.sh probe; set NV_Q38_MTP_INC2_TEST=1"]
fn real_qwen38_inc2_decode_tokps_on_vs_off_256() {
    require_gate();
    let dir = snapshot_dir();
    let (mut m, tok) = boot(&dir, true);
    let k = mtp_chain_depth_from_env();
    let prompt = chat_prompt_ids(&tok);
    let n = DECODE_TOKENS_256_THE_SCOREBOARD_CONTEXT;
    let warmup = 16usize;

    let decode_arm = |m: &mut Qwen3_5DenseWgpu, count: usize| -> (Vec<u32>, f64) {
        m.reset().expect("reset");
        prefill(m, &prompt[..prompt.len() - 1]);
        let mut last = m
            .decode_step(prompt[prompt.len() - 1])
            .expect("anchor decode step");
        let mut out = vec![last];
        for _ in 0..warmup {
            last = m.decode_step(last).expect("warmup step");
            out.push(last);
        }
        let t0 = Instant::now();
        let timed0 = out.len();
        while out.len() < timed0 + count {
            last = m.decode_step(last).expect("decode step");
            out.push(last);
        }
        let rate = (out.len() - timed0) as f64 / t0.elapsed().as_secs_f64();
        (out, rate)
    };

    let (_o, off_null_a) = decode_arm(&mut m, n);
    let (_o, off_null_b) = decode_arm(&mut m, n);
    let (off_stream, off_tokps) = decode_arm(&mut m, n);
    let on_warm = run_mtp_arm(&mut m, &prompt, 1 + warmup, k, false, None);
    let on_a = run_mtp_arm(&mut m, &prompt, 1 + n, k, false, None);
    let (_o2, off_tokps_2) = decode_arm(&mut m, n);
    let on_b = run_mtp_arm(&mut m, &prompt, 1 + n, k, false, None);
    let replay_a = run_mtp_arm(&mut m, &prompt, 1 + n, k, true, None);
    let replay_b = run_mtp_arm(&mut m, &prompt, 1 + n, k, true, None);
    let _ = on_warm;

    let null_ratio = off_null_a / off_null_b;
    measure("q38_mtp_inc2_tokps", n, n, warmup, off_null_a, "tok_per_s")
        .extra("arm", "off_null_a")
        .extra("null_ratio", format!("{null_ratio:.4}"))
        .emit();
    measure("q38_mtp_inc2_tokps", n, n, warmup, off_null_b, "tok_per_s")
        .extra("arm", "off_null_b")
        .emit();
    measure("q38_mtp_inc2_tokps", n, n, warmup, off_tokps, "tok_per_s")
        .extra("arm", "off_a")
        .emit();
    measure("q38_mtp_inc2_tokps", n, n, warmup, off_tokps_2, "tok_per_s")
        .extra("arm", "off_b")
        .emit();
    for (name, st) in [
        ("on_a", &on_a),
        ("on_b", &on_b),
        ("on_replay_inc1_a", &replay_a),
        ("on_replay_inc1_b", &replay_b),
    ] {
        let timed_tokens = st.emitted.len().saturating_sub(1);
        let agree = off_stream
            .iter()
            .zip(&st.emitted)
            .take_while(|(a, b)| a == b)
            .count();
        measure(
            "q38_mtp_inc2_tokps",
            timed_tokens,
            st.rounds,
            0,
            timed_tokens as f64 / st.secs,
            "tok_per_s",
        )
        .extra("arm", name)
        .extra("k", k.to_string())
        .extra("accept_rate", format!("{:.3}", accept_rate(st)))
        .extra("tau", format!("{:.3}", (st.emitted.len() - 1) as f64 / st.rounds as f64))
        .extra("reforward_rounds", st.reforwards.to_string())
        .extra("off_stream_agree_prefix", agree.to_string())
        .emit();
    }
}
