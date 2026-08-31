#![cfg(feature = "wgpu")]

mod g4moe_assistant_common;
use g4moe_assistant_common as c;
use g4moe_assistant_common::official_template::OfficialTemplate;
use std::collections::BTreeMap;

const ACCEPT_DEPTHS_256_CONTROL_AND_16384_THE_BOARD_RUNG: [usize; 2] = [256, 16384];

const CORPUS_OFFSETS_0_AND_100K_MATCH_THE_EAGER_GATE_SO_THE_PARITY_READ_IS_SAME_WINDOW: [usize; 2] =
    [0, 100_000];

const TAIL_DISTINCT_4GRAM_BELOW_0_20_IS_A_GREEDY_LOOP_SO_THE_READING_IS_QUARANTINED: f64 = 0.20;

const TIMING_DECODE_STEPS_30_MEDIAN_BOUNDS_JITTER_WITHOUT_A_LONG_TAIL: usize = 30;

const TIMING_VERIFY_CHUNKS_10_RUN_LAST_BECAUSE_THEIR_KV_ROWS_ARE_GARBAGE: usize = 10;

const TIMING_DRAFT_CALLS_20_THE_DRAFTER_SYNCS_ON_ITS_OWN_ARGMAX: usize = 20;

fn distinct_ngram_fraction(ids: &[u32]) -> f64 {
    let n = 4;
    if ids.len() < n {
        return 1.0;
    }
    let total = ids.len() - n + 1;
    let distinct: std::collections::BTreeSet<&[u32]> = ids.windows(n).collect();
    distinct.len() as f64 / total as f64
}

struct ArmReading {
    mean_emitted: f64,
    breakeven: f64,
    quarantined: bool,
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    m: &mut nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu,
    host: &mut nv_specdecode::gemma4_assistant::MoeAssistantHost,
    prompt: &[u32],
    depth: usize,
    arm: &str,
    rounds_wanted: usize,
    k: usize,
    tok: &tokenizers::Tokenizer,
) -> ArmReading {
    host.clear_mirrors();
    let primed = c::prime_prompt(m, prompt).expect("prime");
    let mut anchor = primed.anchor;
    let mut hidden = primed.hidden;
    let mut recs: Vec<(usize, usize)> = Vec::with_capacity(rounds_wanted);
    let mut emitted_ids: Vec<u32> = Vec::new();
    let mut draft_ms: Vec<f64> = Vec::new();
    let t_rounds = std::time::Instant::now();
    for r in 0..rounds_wanted {
        assert!(
            m.current_pos() + k + 1 <= m.max_seq(),
            "kv headroom exhausted at round {r}"
        );
        let out = c::spec_round(m, host, &mut anchor, &mut hidden, k)
            .unwrap_or_else(|e| panic!("spec round {r}: {e:#}"));
        draft_ms.push(out.draft_s * 1e3);
        recs.push((out.submitted, out.accepted));
        emitted_ids.extend_from_slice(&out.emitted);
    }
    let rounds_s = t_rounds.elapsed().as_secs_f64();

    let mut decode_ms: Vec<f64> = Vec::new();
    let mut cur = anchor;
    for _ in 0..TIMING_DECODE_STEPS_30_MEDIAN_BOUNDS_JITTER_WITHOUT_A_LONG_TAIL {
        let t0 = std::time::Instant::now();
        cur = m.decode_step(cur).expect("timing decode step");
        decode_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let mut verify_ms: Vec<f64> = Vec::new();
    let chunk: Vec<u32> = std::iter::once(cur)
        .chain((0..k as u32).map(|i| 5 + i))
        .collect();
    for _ in 0..TIMING_VERIFY_CHUNKS_10_RUN_LAST_BECAUSE_THEIR_KV_ROWS_ARE_GARBAGE {
        let t0 = std::time::Instant::now();
        let _ = m.verify_chain(&chunk).expect("timing verify chunk");
        verify_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let mut probe_draft_ms: Vec<f64> = Vec::new();
    for _ in 0..TIMING_DRAFT_CALLS_20_THE_DRAFTER_SYNCS_ON_ITS_OWN_ARGMAX {
        let t0 = std::time::Instant::now();
        let _ = host
            .propose_from_engine(&c::EngineTap(m), cur, &hidden, k)
            .expect("timing draft");
        probe_draft_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    let n = recs.len();
    let accepted_total: usize = recs.iter().map(|r| r.1).sum();
    let submitted_total: usize = recs.iter().map(|r| r.0).sum();
    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    for r in &recs {
        *hist.entry(r.1).or_default() += 1;
    }
    let mean_emitted = (accepted_total + n) as f64 / n as f64;
    let last_q = &emitted_ids[emitted_ids.len() - emitted_ids.len() / 4..];
    let tail_distinct = distinct_ngram_fraction(last_q);
    let quarantined =
        tail_distinct < TAIL_DISTINCT_4GRAM_BELOW_0_20_IS_A_GREEDY_LOOP_SO_THE_READING_IS_QUARANTINED;
    let decode_med = c::median_ms(&mut decode_ms);
    let verify_med = c::median_ms(&mut verify_ms);
    let draft_med = c::median_ms(&mut probe_draft_ms);
    let breakeven = (draft_med + verify_med) / decode_med;
    eprintln!(
        "[G4WGPU-ACC] depth={depth} arm={arm} k={k} rounds={n} prompt_tokens={} \
         prefill_s={:.1} rounds_s={rounds_s:.1} \
         mean_accepted_drafts={:.3} mean_emitted_per_round={mean_emitted:.3} \
         accept_rate_over_submitted={:.4} mean_submitted={:.3} \
         distinct_4gram={:.3} distinct_4gram_last_quarter={tail_distinct:.3} \
         quarantined={quarantined} accept_hist={hist:?} \
         decode_ms={decode_med:.2} verify_chunk_ms={verify_med:.2} draft_ms={draft_med:.2} \
         breakeven_emitted_per_round={breakeven:.3}",
        prompt.len(),
        primed.prefill_s,
        accepted_total as f64 / n as f64,
        accepted_total as f64 / submitted_total.max(1) as f64,
        submitted_total as f64 / n as f64,
        distinct_ngram_fraction(&emitted_ids),
    );
    let text = tok.decode(&emitted_ids, true).unwrap_or_default();
    let tail: String = text
        .chars()
        .rev()
        .take(300)
        .collect::<Vec<char>>()
        .into_iter()
        .rev()
        .collect();
    eprintln!("[G4WGPU-TAIL] depth={depth} arm={arm} tail={tail:?}");
    ArmReading {
        mean_emitted,
        breakeven,
        quarantined,
    }
}

#[test]
#[ignore = "loads the 26B-A4B MoE trunk on wgpu plus the 0.8 GB google assistant drafter on \
            cuda, primes chat-wrapped and raw corpus contexts at depths 256 and 16384 through \
            the engine's chunked prefill, then counts exact-u32-argmax acceptance over \
            NV_G4DRAFT_ROUNDS (default 200) verify_chain rounds per arm through the wgpu \
            assistant taps (prenorm hidden + shared trunk KV), and times \
            decode/verify_chain/draft for the wgpu break-even; the read is acceptance parity \
            against the eager gate's numbers on the same corpus windows. NV_G4WGPU_DRAFT_TEST=1 \
            NV_G4DRAFT_ASSISTANT_DIR NV_G4DRAFT_CORPUS required; sets NV_G4MOE_FLASH_DECODE=0 \
            because verify_chain refuses the flash decode graph"]
fn g4moe_wgpu_assistant_accept_parity_at_256_and_16384_matches_the_eager_gate_windows() {
    c::gate_loud("NV_G4WGPU_DRAFT_TEST");
    c::force_the_nonflash_decode_graph_verify_chain_refuses_the_flash_default();
    let trunk_dir = c::trunk_snapshot_dir();
    let assistant_dir = c::assistant_ckpt_dir();
    let rounds_wanted = c::env_usize("NV_G4DRAFT_ROUNDS", 200);
    let k = c::env_usize("NV_G4DRAFT_K", 4);
    let tok = c::trunk_tokenizer(&trunk_dir);
    let tmpl = OfficialTemplate::load(&trunk_dir);
    let corpus_ids = c::corpus_ids(&tok);
    let deepest = *ACCEPT_DEPTHS_256_CONTROL_AND_16384_THE_BOARD_RUNG
        .iter()
        .max()
        .expect("non-empty depths");
    let need = 100_000 + deepest + c::RETOKENIZATION_DRIFT_PAD_128_KEEPS_THE_CHAT_WRAP_AT_OR_ABOVE_THE_RUNG;
    assert!(
        corpus_ids.len() >= need,
        "corpus holds {} tokens but the deepest rung at the last offset needs {need}",
        corpus_ids.len()
    );
    let timing_headroom = TIMING_DECODE_STEPS_30_MEDIAN_BOUNDS_JITTER_WITHOUT_A_LONG_TAIL
        + TIMING_VERIFY_CHUNKS_10_RUN_LAST_BECAUSE_THEIR_KV_ROWS_ARE_GARBAGE * (k + 1)
        + 64;
    let max_seq = deepest
        + c::RETOKENIZATION_DRIFT_PAD_128_KEEPS_THE_CHAT_WRAP_AT_OR_ABOVE_THE_RUNG
        + 512
        + rounds_wanted * (k + 1)
        + timing_headroom;
    let mut m = c::build_engine(&trunk_dir, max_seq);
    let mut host = c::load_host(&m, &trunk_dir, &assistant_dir);
    let (sl, fl) = m.assistant_shared_kv_layers_last_kv_writing_sliding_then_full();
    let (sl, fl) = (
        sl.expect("trunk has a kv-writing sliding layer"),
        fl.expect("trunk has a kv-writing full-attention layer"),
    );
    eprintln!(
        "[G4WGPU-SETUP] trunk={} drafter={} sliding_share_layer={sl} full_share_layer={fl} \
         k={k} rounds={rounds_wanted} drafter_device={:?}",
        trunk_dir.display(),
        assistant_dir.display(),
        host.device()
    );

    let mut deep_clean: Vec<f64> = Vec::new();
    let mut deep_breakeven: Vec<f64> = Vec::new();
    for &depth in &ACCEPT_DEPTHS_256_CONTROL_AND_16384_THE_BOARD_RUNG {
        for &off in &CORPUS_OFFSETS_0_AND_100K_MATCH_THE_EAGER_GATE_SO_THE_PARITY_READ_IS_SAME_WINDOW
        {
            let prompt = c::chat_wrapped_prompt(&tmpl, &tok, &corpus_ids, off, depth);
            let arm = format!("chat@{off}");
            let reading = run_arm(&mut m, &mut host, &prompt, depth, &arm, rounds_wanted, k, &tok);
            if depth == deepest && !reading.quarantined {
                deep_clean.push(reading.mean_emitted);
                deep_breakeven.push(reading.breakeven);
            }
        }
        let prompt = corpus_ids[..depth].to_vec();
        let reading = run_arm(&mut m, &mut host, &prompt, depth, "raw@0", rounds_wanted, k, &tok);
        if depth == deepest && !reading.quarantined {
            deep_clean.push(reading.mean_emitted);
            deep_breakeven.push(reading.breakeven);
        }
    }
    assert!(
        !deep_clean.is_empty(),
        "every 16384 arm was quarantined by the tail 4-gram diagnostic; no acceptance number \
         survives, rerun with different corpus offsets before quoting anything"
    );
    let mean_emitted = deep_clean.iter().sum::<f64>() / deep_clean.len() as f64;
    let mean_breakeven = deep_breakeven.iter().sum::<f64>() / deep_breakeven.len() as f64;
    eprintln!(
        "[G4WGPU-VERDICT] clean_mean_emitted_per_round_at_16384={mean_emitted:.3} \
         per_arm={deep_clean:?} wgpu_breakeven_emitted_per_round={mean_breakeven:.3} \
         verdict={} basis=(drafter=google/gemma-4-26B-A4B-it-assistant full-vocab-head on candle \
         cuda, target=google/gemma-4-26B-A4B-it on gemma4_moe_wgpu with NV_G4MOE_FLASH_DECODE=0, \
         verify_chain m-row graph, greedy exact-u32 accept, shared kv read back from the trunk's \
         last kv-writing sliding+full layers via the assistant taps, emitted=accepted_drafts+1, \
         break-even from measured draft+verify_chain vs decode_step medians on this wgpu path)",
        if mean_emitted >= mean_breakeven {
            "ABOVE-WGPU-BREAKEVEN"
        } else {
            "BELOW-WGPU-BREAKEVEN"
        }
    );
}
