#![cfg(feature = "cuda")]

#[path = "../../nv-models/tests/official_template/mod.rs"]
mod official_template;

use candle_core::{DType, Device, Tensor, D};
use nv_models::gemma4::{Gemma4Config, Gemma4KvCache, LayerType};
use nv_models::gemma4_moe::{Gemma4Moe, Gemma4MoeConfig};
use nv_specdecode::gemma4_assistant::{FixedSharedKv, Gemma4AssistantDrafter};
use official_template::OfficialTemplate;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

const ACCEPT_DEPTHS_256_CONTROL_AND_16384_THE_BOARD_RUNG: [usize; 2] = [256, 16384];

const ACCEPT_ROUNDS_200_DEFAULT_SO_THE_HISTOGRAM_HAS_MASS_IN_EVERY_BIN: usize = 200;

const DRAFT_K_4_MATCHES_THE_GEMMA4_31B_VERIFY_ROUND_ON_RECORD: usize = 4;

const PREFILL_CHUNK_256_IS_THE_SLIDING_CACHE_COMPACT_SLACK_SO_LARGER_CHUNKS_OVERFLOW_THE_RING: usize = 256;

const CORPUS_OFFSETS_0_AND_100K_GIVE_TWO_INDEPENDENT_GREEDY_TRAJECTORIES_SO_THE_VERDICT_IS_NOT_ONE_TEXT_SEGMENT:
    [usize; 2] = [0, 100_000];

const REPETITION_NGRAM_4_DISTINCT_FRACTION_NEAR_ZERO_MEANS_THE_GREEDY_CONTINUATION_IS_LOOPING_AND_ACCEPTANCE_IS_INFLATED: usize = 4;

const TAIL_DISTINCT_4GRAM_BELOW_0_20_IS_A_GREEDY_LOOP_SO_THE_READING_IS_QUARANTINED: f64 = 0.20;

const QUARTILE_SPLIT_4_EXPOSES_GREEDY_DEGENERATION_ACROSS_THE_COUNTED_WINDOW: usize = 4;

const DECODED_TAIL_CHARS_300_MAKE_DEGENERATION_VISIBLE_IN_THE_LOG: usize = 300;

const RETOKENIZATION_DRIFT_PAD_128_KEEPS_THE_CHAT_WRAP_AT_OR_ABOVE_THE_RUNG: usize = 128;

const TIMING_DECODE_STEPS_30_MEDIAN_BOUNDS_JITTER_WITHOUT_A_LONG_TAIL: usize = 30;

const TIMING_VERIFY_CHUNKS_10_RUN_LAST_BECAUSE_THEIR_KV_ROWS_ARE_GARBAGE: usize = 10;

const TIMING_DRAFT_CALLS_20_THE_DRAFTER_SYNCS_ON_ITS_OWN_ARGMAX: usize = 20;

fn gate_loud() {
    if std::env::var("NV_G4DRAFT_TEST").ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_G4DRAFT_TEST=1");
    }
}

fn env_dir(key: &str) -> PathBuf {
    let v = std::env::var(key)
        .unwrap_or_else(|_| panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {key}"));
    let p = PathBuf::from(v);
    assert!(p.is_dir(), "{key} is not a directory: {}", p.display());
    p
}

fn env_usize(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(v) if !v.is_empty() => v
            .parse()
            .unwrap_or_else(|_| panic!("{key} must be a plain integer, got {v:?}")),
        _ => default,
    }
}

fn trunk_snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_G4MOE_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("g4moe snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no complete gemma-4-26B-A4B-it snapshot under HOME hub; set NV_G4MOE_SNAPSHOT")
}

fn last_kv_writing_layers_of_each_kind_the_deepest_kv_the_assistant_can_share(
    cfg: &Gemma4Config,
) -> (usize, usize) {
    let mut sliding = None;
    let mut full = None;
    for (i, kind) in cfg.layer_types.iter().enumerate() {
        match kind {
            LayerType::SlidingAttention => sliding = Some(i),
            LayerType::FullAttention => full = Some(i),
        }
    }
    (
        sliding.expect("trunk has a sliding layer"),
        full.expect("trunk has a full-attention layer"),
    )
}

fn shared_kv_from_trunk_cache(
    cache: &Gemma4KvCache,
    sliding_layer: usize,
    full_layer: usize,
) -> FixedSharedKv {
    let grab = |layer: usize| -> (Tensor, Tensor) {
        let (k, v) = cache.view(layer, 0).expect("trunk kv view");
        let to_kv_major = |t: &Tensor| -> Tensor {
            t.squeeze(0)
                .and_then(|t| t.transpose(0, 1))
                .and_then(|t| t.contiguous())
                .expect("kv [1,len,n_kv,hd] -> [n_kv,len,hd]")
        };
        (to_kv_major(&k), to_kv_major(&v))
    };
    FixedSharedKv {
        sliding: grab(sliding_layer),
        full: grab(full_layer),
    }
}

fn rows_tensors(ids: &[u32], first_pos: usize, device: &Device) -> (Tensor, Tensor) {
    let m = ids.len();
    let t = Tensor::from_vec(ids.to_vec(), (1usize, m), device).expect("row ids");
    let p = Tensor::from_vec(
        (first_pos as i32..(first_pos + m) as i32).collect::<Vec<i32>>(),
        m,
        device,
    )
    .expect("row positions");
    (t, p)
}

fn argmax_rows(logits: &Tensor) -> Vec<u32> {
    logits
        .argmax(D::Minus1)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_dtype(DType::U32))
        .and_then(|t| t.to_vec1::<u32>())
        .expect("greedy argmax")
}

fn last_hidden_row(hidden: &Tensor) -> Tensor {
    let (_, seq, h) = hidden.dims3().expect("hidden [1,seq,h]");
    hidden
        .narrow(1, seq - 1, 1)
        .and_then(|t| t.reshape(h))
        .expect("last hidden row")
}

struct RoundRec {
    submitted: usize,
    accepted: usize,
}

struct ArmResult {
    recs: Vec<RoundRec>,
    emitted_ids: Vec<u32>,
    prefill_s: f64,
    rounds_s: f64,
    prompt_tokens: usize,
    decode_ms_median: f64,
    verify_chunk_ms_median: f64,
    draft_ms_median: f64,
}

fn distinct_ngram_fraction(ids: &[u32]) -> f64 {
    let n = REPETITION_NGRAM_4_DISTINCT_FRACTION_NEAR_ZERO_MEANS_THE_GREEDY_CONTINUATION_IS_LOOPING_AND_ACCEPTANCE_IS_INFLATED;
    if ids.len() < n {
        return 1.0;
    }
    let total = ids.len() - n + 1;
    let distinct: std::collections::BTreeSet<&[u32]> = ids.windows(n).collect();
    distinct.len() as f64 / total as f64
}

fn median_ms(samples: &mut Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

struct TrunkState {
    cache: Gemma4KvCache,
    anchor: u32,
    last_hidden: Tensor,
}

fn prefill_chunked(
    model: &Gemma4Moe,
    prompt: &[u32],
    max_seq: usize,
    device: &Device,
) -> TrunkState {
    let mut cache = model.new_kv_cache(max_seq).expect("kv cache");
    let depth = prompt.len();
    let mut fed = 0usize;
    let mut last: Option<(Tensor, Tensor)> = None;
    while fed < depth {
        let n =
            PREFILL_CHUNK_256_IS_THE_SLIDING_CACHE_COMPACT_SLACK_SO_LARGER_CHUNKS_OVERFLOW_THE_RING
                .min(depth - fed);
        let (t, p) = rows_tensors(&prompt[fed..fed + n], fed, device);
        let out = model
            .forward_with_cache_prenorm_hidden_the_row_the_gemma4_assistant_drafter_consumes(
                &t, &p, &mut cache,
            )
            .unwrap_or_else(|e| panic!("prefill chunk at {fed}: {e:#}"));
        last = Some(out);
        fed += n;
    }
    let _ = device.synchronize();
    assert_eq!(cache.current_len(), depth);
    let (logits, hidden) = last.expect("non-empty prompt");
    let anchor = *argmax_rows(&logits).last().expect("one anchor row");
    TrunkState {
        cache,
        anchor,
        last_hidden: last_hidden_row(&hidden),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    model: &Gemma4Moe,
    drafter: &Gemma4AssistantDrafter,
    sliding_layer: usize,
    full_layer: usize,
    prompt: &[u32],
    rounds_wanted: usize,
    k: usize,
    device: &Device,
) -> ArmResult {
    let depth = prompt.len();
    let timing_headroom = TIMING_DECODE_STEPS_30_MEDIAN_BOUNDS_JITTER_WITHOUT_A_LONG_TAIL
        + TIMING_VERIFY_CHUNKS_10_RUN_LAST_BECAUSE_THEIR_KV_ROWS_ARE_GARBAGE * (k + 1)
        + 64;
    let max_seq = depth + rounds_wanted * (k + 1) + timing_headroom;
    let pre_t0 = Instant::now();
    let mut st = prefill_chunked(model, prompt, max_seq, device);
    let prefill_s = pre_t0.elapsed().as_secs_f64();

    let embed_w = model.embed_weight().clone();
    let embed_scale = model.embed_scale() as f64;
    let embed_device = device.clone();
    let embedder = move |tok: u32| -> anyhow::Result<Tensor> {
        let idx = Tensor::from_vec(vec![tok], 1usize, &embed_device)?;
        Ok(embed_w
            .index_select(&idx, 0)?
            .squeeze(0)?
            .affine(embed_scale, 0.0)?)
    };

    let mut recs: Vec<RoundRec> = Vec::with_capacity(rounds_wanted);
    let mut emitted_ids: Vec<u32> = Vec::with_capacity(rounds_wanted * (k + 1));
    let mut draft_ms: Vec<f64> = Vec::new();
    let rounds_t0 = Instant::now();
    for r in 0..rounds_wanted {
        let l = st.cache.current_len();
        assert!(l + k + 1 <= max_seq, "kv headroom exhausted at round {r}");
        let kv = shared_kv_from_trunk_cache(&st.cache, sliding_layer, full_layer);
        let d_t0 = Instant::now();
        let drafts = drafter
            .propose(st.anchor, &st.last_hidden, l, k, &embedder, &kv)
            .unwrap_or_else(|e| panic!("draft at round {r}: {e:#}"));
        draft_ms.push(d_t0.elapsed().as_secs_f64() * 1e3);
        let submitted = drafts.len();
        assert!((1..=k).contains(&submitted), "propose returned {submitted}");
        let mut accepted = 0usize;
        let mut cur = st.anchor;
        loop {
            let (t, p) = rows_tensors(&[cur], st.cache.current_len(), device);
            let (logits, hidden) = model
                .forward_with_cache_prenorm_hidden_the_row_the_gemma4_assistant_drafter_consumes(
                    &t,
                    &p,
                    &mut st.cache,
                )
                .unwrap_or_else(|e| panic!("verify step at round {r}: {e:#}"));
            let next = argmax_rows(&logits)[0];
            if accepted < submitted && drafts[accepted] == next {
                emitted_ids.push(next);
                accepted += 1;
                cur = next;
                continue;
            }
            emitted_ids.push(next);
            st.anchor = next;
            st.last_hidden = last_hidden_row(&hidden);
            break;
        }
        recs.push(RoundRec {
            submitted,
            accepted,
        });
    }
    let _ = device.synchronize();
    let rounds_s = rounds_t0.elapsed().as_secs_f64();

    let mut decode_ms: Vec<f64> = Vec::new();
    let mut cur = st.anchor;
    for _ in 0..TIMING_DECODE_STEPS_30_MEDIAN_BOUNDS_JITTER_WITHOUT_A_LONG_TAIL {
        let (t, p) = rows_tensors(&[cur], st.cache.current_len(), device);
        let t0 = Instant::now();
        let logits = model
            .forward_with_cache(&t, &p, &mut st.cache)
            .expect("timing decode step");
        cur = argmax_rows(&logits)[0];
        let _ = device.synchronize();
        decode_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let mut verify_ms: Vec<f64> = Vec::new();
    let chunk: Vec<u32> = std::iter::once(cur).chain((0..k as u32).map(|i| 5 + i)).collect();
    for _ in 0..TIMING_VERIFY_CHUNKS_10_RUN_LAST_BECAUSE_THEIR_KV_ROWS_ARE_GARBAGE {
        let (t, p) = rows_tensors(&chunk, st.cache.current_len(), device);
        let t0 = Instant::now();
        let logits = model
            .forward_with_cache(&t, &p, &mut st.cache)
            .expect("timing verify chunk");
        let _ = argmax_rows(&logits);
        let _ = device.synchronize();
        verify_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let kv = shared_kv_from_trunk_cache(&st.cache, sliding_layer, full_layer);
    let mut probe_draft_ms: Vec<f64> = Vec::new();
    for _ in 0..TIMING_DRAFT_CALLS_20_THE_DRAFTER_SYNCS_ON_ITS_OWN_ARGMAX {
        let t0 = Instant::now();
        let _ = drafter
            .propose(
                st.anchor,
                &st.last_hidden,
                st.cache.current_len(),
                k,
                &embedder,
                &kv,
            )
            .expect("timing draft");
        probe_draft_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    ArmResult {
        recs,
        emitted_ids,
        prefill_s,
        rounds_s,
        prompt_tokens: depth,
        decode_ms_median: median_ms(&mut decode_ms),
        verify_chunk_ms_median: median_ms(&mut verify_ms),
        draft_ms_median: median_ms(&mut probe_draft_ms),
    }
}

struct ArmReading {
    mean_emitted: f64,
    quarantined: bool,
}

fn report_arm(depth: usize, arm: &str, k: usize, res: &ArmResult) -> ArmReading {
    let n = res.recs.len();
    let accepted_total: usize = res.recs.iter().map(|r| r.accepted).sum();
    let submitted_total: usize = res.recs.iter().map(|r| r.submitted).sum();
    let emitted_total = accepted_total + n;
    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    for r in &res.recs {
        *hist.entry(r.accepted).or_default() += 1;
    }
    let q = QUARTILE_SPLIT_4_EXPOSES_GREEDY_DEGENERATION_ACROSS_THE_COUNTED_WINDOW;
    let quartiles: Vec<String> = (0..q)
        .map(|i| {
            let lo = i * n / q;
            let hi = (i + 1) * n / q;
            let chunk = &res.recs[lo..hi];
            let acc: usize = chunk.iter().map(|r| r.accepted).sum();
            format!(
                "q{i}={:.2}",
                (acc + chunk.len()) as f64 / chunk.len().max(1) as f64
            )
        })
        .collect();
    let mean_emitted = emitted_total as f64 / n as f64;
    let e = &res.emitted_ids;
    let last_q = &e[e.len() - e.len() / q..];
    let tail_distinct = distinct_ngram_fraction(last_q);
    let quarantined =
        tail_distinct < TAIL_DISTINCT_4GRAM_BELOW_0_20_IS_A_GREEDY_LOOP_SO_THE_READING_IS_QUARANTINED;
    let breakeven_emitted = (res.draft_ms_median + res.verify_chunk_ms_median) / res.decode_ms_median;
    eprintln!(
        "[G4DRAFT-ACC] depth={depth} arm={arm} k={k} rounds={n} prompt_tokens={} \
         prefill_s={:.1} rounds_s={:.1} \
         mean_accepted_drafts={:.3} mean_emitted_per_round={mean_emitted:.3} \
         accept_rate_over_submitted={:.4} mean_submitted={:.3} \
         distinct_4gram={:.3} distinct_4gram_last_quarter={tail_distinct:.3} quarantined={quarantined} \
         accept_hist={hist:?} emitted_quartiles=[{}] \
         decode_ms={:.2} verify_chunk_ms={:.2} draft_ms={:.2} breakeven_emitted_per_round={breakeven_emitted:.3}",
        res.prompt_tokens,
        res.prefill_s,
        res.rounds_s,
        accepted_total as f64 / n as f64,
        accepted_total as f64 / submitted_total.max(1) as f64,
        submitted_total as f64 / n as f64,
        distinct_ngram_fraction(e),
        quartiles.join(","),
        res.decode_ms_median,
        res.verify_chunk_ms_median,
        res.draft_ms_median,
    );
    ArmReading {
        mean_emitted,
        quarantined,
    }
}

fn report_tail_text(depth: usize, arm: &str, res: &ArmResult, tok: &tokenizers::Tokenizer) {
    let text = tok.decode(&res.emitted_ids, true).unwrap_or_default();
    let tail: String = text
        .chars()
        .rev()
        .take(DECODED_TAIL_CHARS_300_MAKE_DEGENERATION_VISIBLE_IN_THE_LOG)
        .collect::<Vec<char>>()
        .into_iter()
        .rev()
        .collect();
    eprintln!("[G4DRAFT-TAIL] depth={depth} arm={arm} tail={tail:?}");
}

fn chat_wrapped_prompt(
    tmpl: &OfficialTemplate,
    tok: &tokenizers::Tokenizer,
    corpus_ids: &[u32],
    offset: usize,
    depth: usize,
) -> Vec<u32> {
    let body = tok
        .decode(
            &corpus_ids[offset
                ..offset + depth + RETOKENIZATION_DRIFT_PAD_128_KEEPS_THE_CHAT_WRAP_AT_OR_ABOVE_THE_RUNG],
            true,
        )
        .expect("decode corpus slice");
    let rendered = tmpl.render_user(&format!(
        "Continue this document, staying in its style:\n\n{body}"
    ));
    let ids: Vec<u32> = tok
        .encode(rendered.as_str(), false)
        .expect("encode rendered prompt")
        .get_ids()
        .to_vec();
    assert!(
        ids.len() >= depth,
        "chat wrap produced {} tokens for a {depth} rung; the template cannot shrink its body",
        ids.len()
    );
    ids
}

#[test]
#[ignore = "loads the ~50 GB bf16 gemma-4-26B-A4B-it MoE trunk on cuda plus the 0.8 GB \
            google external assistant drafter, primes chat-wrapped and raw corpus contexts at \
            depths 256 and 16384, then counts exact-u32-argmax acceptance over NV_G4DRAFT_ROUNDS \
            (default 200) speculative rounds per arm with seq-1 eager verify, and times \
            decode/verify/draft for the family's own break-even. NV_G4DRAFT_TEST=1 \
            NV_G4DRAFT_ASSISTANT_DIR NV_G4DRAFT_CORPUS required; acceptance instrument only, \
            wall time is not the metric here"]
fn g4moe_assistant_accept_at_256_and_16384_chat_and_raw_arms_with_family_breakeven() {
    gate_loud();
    let trunk_dir = trunk_snapshot_dir();
    let assistant_dir = env_dir("NV_G4DRAFT_ASSISTANT_DIR");
    let corpus = std::env::var("NV_G4DRAFT_CORPUS").unwrap_or_else(|_| {
        panic!(
            "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_G4DRAFT_CORPUS to a raw \
             text file holding at least offset+depth tokens; a tiled paragraph reads as an upper \
             bracket, not an acceptance number"
        )
    });
    let rounds_wanted = env_usize(
        "NV_G4DRAFT_ROUNDS",
        ACCEPT_ROUNDS_200_DEFAULT_SO_THE_HISTOGRAM_HAS_MASS_IN_EVERY_BIN,
    );
    let k = env_usize(
        "NV_G4DRAFT_K",
        DRAFT_K_4_MATCHES_THE_GEMMA4_31B_VERIFY_ROUND_ON_RECORD,
    );
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let tok = tokenizers::Tokenizer::from_file(trunk_dir.join("tokenizer.json"))
        .expect("trunk tokenizer.json");
    let tmpl = OfficialTemplate::load(&trunk_dir);
    let text = std::fs::read_to_string(&corpus).expect("corpus file");
    let corpus_ids: Vec<u32> = tok
        .encode(text.as_str(), false)
        .expect("encode corpus")
        .get_ids()
        .to_vec();
    let deepest = *ACCEPT_DEPTHS_256_CONTROL_AND_16384_THE_BOARD_RUNG
        .iter()
        .max()
        .expect("non-empty depth list");
    let max_offset = *CORPUS_OFFSETS_0_AND_100K_GIVE_TWO_INDEPENDENT_GREEDY_TRAJECTORIES_SO_THE_VERDICT_IS_NOT_ONE_TEXT_SEGMENT
        .iter()
        .max()
        .expect("non-empty offset list");
    let need =
        max_offset + deepest + RETOKENIZATION_DRIFT_PAD_128_KEEPS_THE_CHAT_WRAP_AT_OR_ABOVE_THE_RUNG;
    assert!(
        corpus_ids.len() >= need,
        "corpus {corpus} holds {} tokens but the deepest rung at the last offset needs {need}",
        corpus_ids.len()
    );

    let trunk_cfg =
        Gemma4MoeConfig::from_hf_json_file(&trunk_dir.join("config.json")).expect("trunk config");
    let (sliding_layer, full_layer) =
        last_kv_writing_layers_of_each_kind_the_deepest_kv_the_assistant_can_share(&trunk_cfg.base);
    let loader = nv_weights::WeightLoader::open_dir(&trunk_dir, &device).expect("trunk weights");
    let model = Gemma4Moe::from_loader(trunk_cfg, &loader, &device).expect("bf16 trunk");
    drop(loader);
    let drafter =
        Gemma4AssistantDrafter::try_load(&assistant_dir, &device).expect("assistant drafter");
    let acfg = drafter.config();
    assert_eq!(
        acfg.backbone_hidden_size,
        model.config().base.hidden_size,
        "assistant backbone width must match the trunk"
    );
    assert_eq!(acfg.head_dim, model.config().base.head_dim_for(LayerType::SlidingAttention));
    assert_eq!(
        acfg.global_head_dim,
        model.config().base.head_dim_for(LayerType::FullAttention)
    );
    eprintln!(
        "[G4DRAFT-SETUP] trunk={} drafter={} sliding_share_layer={sliding_layer} \
         full_share_layer={full_layer} k={k} rounds={rounds_wanted}",
        trunk_dir.display(),
        assistant_dir.display()
    );

    let mut deep_clean: Vec<f64> = Vec::new();
    let mut deep_breakeven: Vec<f64> = Vec::new();
    for &depth in &ACCEPT_DEPTHS_256_CONTROL_AND_16384_THE_BOARD_RUNG {
        for &off in
            &CORPUS_OFFSETS_0_AND_100K_GIVE_TWO_INDEPENDENT_GREEDY_TRAJECTORIES_SO_THE_VERDICT_IS_NOT_ONE_TEXT_SEGMENT
        {
            let prompt = chat_wrapped_prompt(&tmpl, &tok, &corpus_ids, off, depth);
            let res = run_arm(
                &model,
                &drafter,
                sliding_layer,
                full_layer,
                &prompt,
                rounds_wanted,
                k,
                &device,
            );
            let arm = format!("chat@{off}");
            let reading = report_arm(depth, &arm, k, &res);
            report_tail_text(depth, &arm, &res, &tok);
            if depth == deepest && !reading.quarantined {
                deep_clean.push(reading.mean_emitted);
                deep_breakeven
                    .push((res.draft_ms_median + res.verify_chunk_ms_median) / res.decode_ms_median);
            }
        }
        let prompt = &corpus_ids[..depth];
        let res = run_arm(
            &model,
            &drafter,
            sliding_layer,
            full_layer,
            prompt,
            rounds_wanted,
            k,
            &device,
        );
        let reading = report_arm(depth, "raw@0", k, &res);
        report_tail_text(depth, "raw@0", &res, &tok);
        if depth == deepest && !reading.quarantined {
            deep_clean.push(reading.mean_emitted);
            deep_breakeven
                .push((res.draft_ms_median + res.verify_chunk_ms_median) / res.decode_ms_median);
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
        "[G4DRAFT-VERDICT] clean_mean_emitted_per_round_at_16384={mean_emitted:.3} \
         per_arm={deep_clean:?} family_breakeven_emitted_per_round={mean_breakeven:.3} \
         verdict={} basis=(drafter=google/gemma-4-26B-A4B-it-assistant full-vocab-head \
         target=google/gemma-4-26B-A4B-it bf16 cuda eager forward_with_cache seq-1 verify, \
         greedy exact-u32 accept, shared kv from the trunk's last sliding+full layers, \
         emitted=accepted_drafts+1, break-even from measured draft+verify_chunk vs decode medians \
         on the same eager path)",
        if mean_emitted >= mean_breakeven {
            "ABOVE-FAMILY-BREAKEVEN"
        } else {
            "BELOW-FAMILY-BREAKEVEN"
        }
    );
}
