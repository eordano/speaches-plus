#![cfg(feature = "wgpu")]

mod hub_dirs;

use std::path::PathBuf;
use std::time::Instant;

use nv_models::qwen3_5_dense_wgpu::{Q3dLora, Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use nv_specdecode::chain::{accept_prefix_argmax, build_chain_batch, ChainAccept};
use nv_specdecode::qwen38_mtp::{
    mtp_chain_depth_from_env, mtp_verify_replay_selected_from_env, run_mtp_verify_round,
};

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const CHECKPOINT_LABEL: &str = "unsloth/Qwen3.8-27B-NVFP4";

const WINDOWS_MIN_9_BECAUSE_08_2_SAYS_FEWER_INDEPENDENT_WINDOWS_CANNOT_ADJUDICATE: usize = 9;
const PROMPT_TOKEN_CAP_504_KEEPS_EVERY_WINDOW_INSIDE_THE_CONSOLIDATION_512_EVAL_CTX: usize = 504;
const PROMPT_TOKEN_FLOOR_8_A_SHORTER_PROMPT_IS_A_DEGENERATE_WINDOW: usize = 8;
const EMITTED_PER_ARM_DEFAULT_128_BOUNDS_EACH_STREAM_WELL_UNDER_A_MINUTE: usize = 128;
const MAX_SEQ_2048_COVERS_PROMPT_CAP_PLUS_EMIT_PLUS_THE_VERIFY_GUARD: usize = 2048;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .map(|v| v.parse::<usize>().unwrap_or_else(|_| panic!("{key}={v} is not a usize")))
        .unwrap_or(default)
}

fn require_gate(gate: &str) {
    if std::env::var(gate).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {gate}=1");
    }
}

fn snapshot_dir() -> PathBuf {
    match std::env::var("NV_QWEN38_DIR") {
        Ok(d) if !d.is_empty() => {
            let p = PathBuf::from(&d);
            assert!(p.is_dir(), "NV_QWEN38_DIR={d} is not a directory");
            p
        }
        _ => hub_dirs::snapshot(
            NVFP4_REPO,
            &["config.json", "tokenizer.json", "model_mtp.safetensors"],
        )
        .unwrap_or_else(|| {
            panic!(
                "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: no hydrated {NVFP4_REPO} \
                 snapshot carrying the MTP shard; set NV_QWEN38_DIR"
            )
        }),
    }
}

fn snapshot_rev_label(dir: &std::path::Path) -> String {
    dir.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn real_adapter_dir_r8_trained_by_the_fine_tuning_endpoint() -> PathBuf {
    let raw = std::env::var("NV_LORA_CURVE_ADAPTER_DIR").unwrap_or_else(|_| {
        panic!(
            "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_LORA_CURVE_ADAPTER_DIR to \
             a peft adapter dir produced by POST /v1/fine_tuning/jobs (result_files[0]); this \
             suite measures a REAL trained adapter, never a synthetic one"
        )
    });
    let dir = PathBuf::from(&raw);
    assert!(
        dir.join("adapter_model.safetensors").is_file() && dir.join("adapter_config.json").is_file(),
        "NV_LORA_CURVE_ADAPTER_DIR={raw} lacks the peft pair adapter_config.json + \
         adapter_model.safetensors"
    );
    dir
}

fn adapter_rank_alpha_label(dir: &std::path::Path) -> String {
    let raw = std::fs::read_to_string(dir.join("adapter_config.json"))
        .expect("read adapter_config.json");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("parse adapter_config.json");
    format!(
        "r{}_alpha{}",
        v.get("r").and_then(|x| x.as_u64()).unwrap_or(0),
        v.get("lora_alpha").and_then(|x| x.as_f64()).unwrap_or(0.0)
    )
}

fn independent_prompt_windows_from_env_json() -> (Vec<String>, u64) {
    let raw = std::env::var("NV_LORA_CURVE_PROMPTS").unwrap_or_else(|_| {
        panic!(
            "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_LORA_CURVE_PROMPTS to a \
             json array of >= {WINDOWS_MIN_9_BECAUSE_08_2_SAYS_FEWER_INDEPENDENT_WINDOWS_CANNOT_ADJUDICATE} \
             prompt strings drawn one-per-independent-held-out-window \
             (scripts/extract_holdout_prompt_windows.py writes it)"
        )
    });
    let bytes = std::fs::read(&raw).unwrap_or_else(|e| panic!("read {raw}: {e}"));
    let fp = fnv64(&bytes);
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("prompt windows json");
    let prompts: Vec<String> = v
        .as_array()
        .expect("prompt windows json must be an array")
        .iter()
        .map(|e| match e {
            serde_json::Value::String(s) => s.clone(),
            other => other
                .get("prompt")
                .and_then(|p| p.as_str())
                .expect("prompt window entry must be a string or carry a prompt field")
                .to_string(),
        })
        .collect();
    assert!(
        prompts.len() >= WINDOWS_MIN_9_BECAUSE_08_2_SAYS_FEWER_INDEPENDENT_WINDOWS_CANNOT_ADJUDICATE,
        "{} prompt windows in {raw}: fewer than 9 independent windows cannot adjudicate \
         (docs/book/08.2-quality-verdicts.md)",
        prompts.len()
    );
    (prompts, fp)
}

fn prompt_ids_capped(tok: &tokenizers::Tokenizer, text: &str) -> Vec<u32> {
    let mut ids = tok
        .encode(text, false)
        .expect("encode prompt window")
        .get_ids()
        .to_vec();
    ids.truncate(PROMPT_TOKEN_CAP_504_KEEPS_EVERY_WINDOW_INSIDE_THE_CONSOLIDATION_512_EVAL_CTX);
    assert!(
        ids.len() >= PROMPT_TOKEN_FLOOR_8_A_SHORTER_PROMPT_IS_A_DEGENERATE_WINDOW,
        "prompt window tokenized to {} tokens, under the floor of \
         {PROMPT_TOKEN_FLOOR_8_A_SHORTER_PROMPT_IS_A_DEGENERATE_WINDOW}",
        ids.len()
    );
    ids
}

fn prefill_like_serving(m: &mut Qwen3_5DenseWgpu, prompt: &[u32]) -> u32 {
    let (last, rest) = prompt.split_last().expect("non-empty prompt");
    let done = m.prefill_tokens(rest).expect("chunked prefill");
    for &t in &rest[done..] {
        m.prefill_step(t).expect("prefill step");
    }
    m.decode_step(*last).expect("anchor decode step")
}

fn median_of(ms: &[f64]) -> f64 {
    let mut sorted = ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

fn log_softmax_f64(row: &[f32]) -> Vec<f64> {
    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut lse = 0f64;
    for &v in row {
        lse += ((v as f64) - mx).exp();
    }
    let lse = mx + lse.ln();
    row.iter().map(|&v| (v as f64) - lse).collect()
}

fn kl_base_to_arm_nats(base_row: &[f32], arm_row: &[f32]) -> f64 {
    let lp_b = log_softmax_f64(base_row);
    let lp_a = log_softmax_f64(arm_row);
    lp_b.iter().zip(&lp_a).map(|(pb, pa)| pb.exp() * (pb - pa)).sum()
}

struct LogitsRound {
    batch: Vec<u32>,
    amax: Vec<u32>,
    accept: ChainAccept,
    emitted: Vec<u32>,
    rows: Vec<f32>,
}

fn commit_accept_like_run_mtp_verify_round(
    gpu: &mut Qwen3_5DenseWgpu,
    batch: &[u32],
    commit_len: usize,
    replay: bool,
) {
    let full = commit_len == batch.len();
    let scan = !replay && !full && gpu.mtp_commit_scan_replay_active();
    if replay || full || scan {
        gpu.advance(commit_len).expect("advance commit");
    } else {
        gpu.advance(0).expect("advance rollback");
        gpu.verify_chain_commit_only(&batch[..commit_len])
            .expect("commit-only prefix reforward");
        gpu.advance(commit_len).expect("advance committed prefix");
    }
}

fn verify_round_with_logits(
    gpu: &mut Qwen3_5DenseWgpu,
    anchor: u32,
    drafts: &[u32],
    replay: bool,
) -> LogitsRound {
    let batch = build_chain_batch(anchor, drafts, drafts.len() + 1, true).expect("chain batch");
    let (amax, rows) = gpu.verify_chain_logits(&batch).expect("verify chain with logit rows");
    let accept = accept_prefix_argmax(&batch, &amax).expect("accept prefix");
    commit_accept_like_run_mtp_verify_round(gpu, &batch, accept.commit_len, replay);
    let mut emitted: Vec<u32> = drafts[..accept.draft_accepted].to_vec();
    emitted.push(accept.next_bonus);
    LogitsRound { batch, amax, accept, emitted, rows }
}

#[derive(Default)]
struct AcceptTally {
    rounds: usize,
    drafted: usize,
    accepted: usize,
    emitted: usize,
}

impl AcceptTally {
    fn accept_rate(&self) -> f64 {
        self.accepted as f64 / self.drafted.max(1) as f64
    }

    fn tokens_per_round(&self) -> f64 {
        self.emitted as f64 / self.rounds.max(1) as f64
    }
}

fn accept_window_no_logits(
    gpu: &mut Qwen3_5DenseWgpu,
    anchor: u32,
    k: usize,
    replay: bool,
    emit_target: usize,
    max_seq: usize,
) -> AcceptTally {
    let mut t = AcceptTally::default();
    let mut last = anchor;
    while t.emitted < emit_target {
        let rows = gpu.verify_max_rows();
        assert!(rows >= 2, "the M-row verify graph is absent");
        assert!(
            gpu.current_pos() + rows.max(gpu.prefill_chunk_len()) <= max_seq,
            "kv headroom exhausted mid-window at pos {}",
            gpu.current_pos()
        );
        let want = k.min(rows - 1);
        let drafts = gpu.mtp_draft_round(last, want).expect("mtp draft round");
        let r = run_mtp_verify_round(gpu, last, &drafts, replay).expect("mtp verify round");
        gpu.mtp_post_verify(&r.batch[1..r.accept.commit_len])
            .expect("mtp post verify");
        t.rounds += 1;
        t.drafted += drafts.len();
        t.accepted += r.accept.draft_accepted;
        t.emitted += r.emitted.len();
        last = *r.emitted.last().expect("a round always emits the bonus token");
    }
    assert!(
        t.drafted > 0 && t.accepted > 0,
        "zero acceptance over {} drafts: refuse the window",
        t.drafted
    );
    t
}

struct KlWindow {
    tally: AcceptTally,
    round_kl_means: Vec<f64>,
    all_row_kls: Vec<f64>,
    mirror_top1_agree_rows: usize,
    mirror_rows: usize,
}

fn adapter_drives_base_mirrors_teacher_forced_kl_window(
    adapted: &mut Qwen3_5DenseWgpu,
    base: &mut Qwen3_5DenseWgpu,
    anchor: u32,
    k: usize,
    replay: bool,
    emit_target: usize,
    max_seq: usize,
    window: usize,
    basis: &str,
) -> KlWindow {
    let mut w = KlWindow {
        tally: AcceptTally::default(),
        round_kl_means: Vec::new(),
        all_row_kls: Vec::new(),
        mirror_top1_agree_rows: 0,
        mirror_rows: 0,
    };
    let mut last = anchor;
    while w.tally.emitted < emit_target {
        let rows = adapted.verify_max_rows().min(base.verify_max_rows());
        assert!(rows >= 2, "the M-row verify graph is absent on one arm");
        for gpu in [&mut *adapted, &mut *base] {
            assert!(
                gpu.current_pos() + rows.max(gpu.prefill_chunk_len()) <= max_seq,
                "kv headroom exhausted mid-window at pos {}",
                gpu.current_pos()
            );
        }
        let want = k.min(rows - 1);
        let drafts = adapted.mtp_draft_round(last, want).expect("adapter draft round");
        let r = verify_round_with_logits(adapted, last, &drafts, replay);
        adapted
            .mtp_post_verify(&r.batch[1..r.accept.commit_len])
            .expect("adapter mtp post verify");
        let (base_amax, base_rows) = base
            .verify_chain_logits(&r.batch)
            .expect("base mirror verify chain with logit rows");
        commit_accept_like_run_mtp_verify_round(base, &r.batch, r.accept.commit_len, replay);
        assert!(
            !drafts.is_empty(),
            "the teacher-forced base mirror never drafts, so it has no mtp round state to \
             post-verify; only the adapter arm maintains its drafter head"
        );
        let live = r.batch.len();
        let vocab = r.rows.len() / live;
        assert_eq!(r.rows.len(), live * vocab, "adapter logit rows are not row-major");
        assert_eq!(
            base_rows.len(),
            r.rows.len(),
            "mirror logit rows must match the driver's row count"
        );
        let mut round_kls: Vec<f64> = Vec::with_capacity(live);
        for i in 0..live {
            round_kls.push(kl_base_to_arm_nats(
                &base_rows[i * vocab..(i + 1) * vocab],
                &r.rows[i * vocab..(i + 1) * vocab],
            ));
        }
        w.mirror_rows += live;
        w.mirror_top1_agree_rows +=
            base_amax.iter().zip(&r.amax).filter(|(b, a)| b == a).count();
        let kl_mean = round_kls.iter().sum::<f64>() / live as f64;
        let kl_max = round_kls.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "LORA-CURVE-KL window={window} round={} rows={live} commit_len={} \
             kl_mean_nats={kl_mean:.5} kl_max_nats={kl_max:.5} \
             basis=teacher_forced_adapter_stream_through_the_m_row_verify_graph {basis}",
            w.tally.rounds,
            r.accept.commit_len,
        );
        w.round_kl_means.push(kl_mean);
        w.all_row_kls.extend(round_kls);
        w.tally.rounds += 1;
        w.tally.drafted += drafts.len();
        w.tally.accepted += r.accept.draft_accepted;
        w.tally.emitted += r.emitted.len();
        last = *r.emitted.last().expect("a round always emits the bonus token");
    }
    assert_eq!(
        adapted.current_pos(),
        base.current_pos(),
        "teacher forcing must hold both engines at the same committed depth"
    );
    w
}

fn paired_mean_sd_t(deltas: &[f64]) -> (f64, f64, f64) {
    let n = deltas.len() as f64;
    let mean = deltas.iter().sum::<f64>() / n;
    let var = deltas.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (n - 1.0);
    let sd = var.sqrt();
    let t = if sd > 0.0 { mean / (sd / n.sqrt()) } else { f64::INFINITY * mean.signum() };
    (mean, sd, t)
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B dense hybrid TWICE on wgpu (base + REAL trained \
            rank-8 adapter from POST /v1/fine_tuning/jobs, both with the shipped MTP head); set \
            NV_QWEN38_WGPU_TEST=1 NV_LORA_CURVE_ADAPTER_DIR=<job result_files[0]> \
            NV_LORA_CURVE_PROMPTS=<json array of >=9 held-out window prompts> -- per window: \
            base-arm MTP acceptance, adapter-arm MTP acceptance, and per-round \
            KL(base||adapter) from verify-chain logit rows with the base engine teacher-forced \
            along the adapter's committed stream; ends with the 08.2 paired verdict row over \
            the >=9 independent windows"]
fn real_rank8_adapter_mtp_acceptance_curve_with_per_round_verify_chain_kl_over_nine_windows() {
    require_gate("NV_QWEN38_WGPU_TEST");
    let dir = snapshot_dir();
    let rev = snapshot_rev_label(&dir);
    let adapter_dir = real_adapter_dir_r8_trained_by_the_fine_tuning_endpoint();
    let adapter_label = adapter_rank_alpha_label(&adapter_dir);
    let (prompts, prompts_fp) = independent_prompt_windows_from_env_json();
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());

    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("qwen3.8 dense config");
    let max_seq = MAX_SEQ_2048_COVERS_PROMPT_CAP_PLUS_EMIT_PLUS_THE_VERIFY_GUARD;
    let tok =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");

    let lora = Q3dLora::from_peft_dir(&adapter_dir, &cfg).expect("build q3d lora sites");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let t0 = Instant::now();
    let mut base =
        Qwen3_5DenseWgpu::from_loader(cfg.clone(), &loader, max_seq).expect("base engine boot");
    base.mtp_attach(&loader).expect("attach mtp head to the base engine");
    let base_boot_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let mut adapted =
        Qwen3_5DenseWgpu::from_loader_with_lora(cfg.clone(), &loader, max_seq, Some(&lora))
            .expect("adapter engine boot");
    adapted.mtp_attach(&loader).expect("attach mtp head to the adapter engine");
    let adapted_boot_s = t1.elapsed().as_secs_f64();
    drop(loader);
    assert!(!base.lora_resident(), "base engine must carry no lora sites");
    assert!(adapted.lora_resident(), "adapter engine must carry resident lora sites");

    let k = mtp_chain_depth_from_env();
    let replay = mtp_verify_replay_selected_from_env();
    let emit_target = env_usize(
        "NV_LORA_CURVE_EMITTED",
        EMITTED_PER_ARM_DEFAULT_128_BOUNDS_EACH_STREAM_WELL_UNDER_A_MINUTE,
    );
    let basis = format!(
        "k={k} replay={replay} emit_target={emit_target} adapter={adapter_label} \
         adapter_dir={} prompts_fp={prompts_fp:016x} windows={} \
         checkpoint={CHECKPOINT_LABEL}@{rev}",
        adapter_dir.display(),
        prompts.len(),
    );
    eprintln!(
        "LORA-CURVE-BOOT base_boot_s={base_boot_s:.1} adapted_boot_s={adapted_boot_s:.1} \
         max_seq={max_seq} {basis}"
    );

    let mut delta_accept: Vec<f64> = Vec::new();
    let mut delta_tpr: Vec<f64> = Vec::new();
    for (wi, text) in prompts.iter().enumerate() {
        let ids = prompt_ids_capped(&tok, text);
        let prompt_fp = fnv64(text.as_bytes());
        base.reset().expect("reset base for the acceptance pass");
        let t_pre = Instant::now();
        let anchor_b = prefill_like_serving(&mut base, &ids);
        let base_prefill_s = t_pre.elapsed().as_secs_f64();
        let tb = accept_window_no_logits(&mut base, anchor_b, k, replay, emit_target, max_seq);
        eprintln!(
            "LORA-CURVE-WINDOW window={wi} arm=base prompt_tokens={} prompt_fp={prompt_fp:016x} \
             rounds={} drafted={} accepted={} accept_rate={:.3} tokens_per_round={:.2} \
             emitted={} prefill_s={base_prefill_s:.1} {basis}",
            ids.len(),
            tb.rounds,
            tb.drafted,
            tb.accepted,
            tb.accept_rate(),
            tb.tokens_per_round(),
            tb.emitted,
        );

        adapted.reset().expect("reset adapter for the kl pass");
        let t_pre = Instant::now();
        let anchor_l = prefill_like_serving(&mut adapted, &ids);
        let lora_prefill_s = t_pre.elapsed().as_secs_f64();
        base.reset().expect("reset base for the teacher-forced mirror");
        let _ = prefill_like_serving(&mut base, &ids);
        let w = adapter_drives_base_mirrors_teacher_forced_kl_window(
            &mut adapted,
            &mut base,
            anchor_l,
            k,
            replay,
            emit_target,
            max_seq,
            wi,
            &basis,
        );
        eprintln!(
            "LORA-CURVE-WINDOW window={wi} arm=lora prompt_tokens={} prompt_fp={prompt_fp:016x} \
             rounds={} drafted={} accepted={} accept_rate={:.3} tokens_per_round={:.2} \
             emitted={} prefill_s={lora_prefill_s:.1} {basis}",
            ids.len(),
            w.tally.rounds,
            w.tally.drafted,
            w.tally.accepted,
            w.tally.accept_rate(),
            w.tally.tokens_per_round(),
            w.tally.emitted,
        );
        eprintln!(
            "LORA-CURVE-KL-WINDOW window={wi} rounds={} rows={} kl_mean_nats={:.5} \
             kl_round_median_nats={:.5} kl_max_nats={:.5} mirror_top1_agree={}/{} {basis}",
            w.round_kl_means.len(),
            w.all_row_kls.len(),
            w.all_row_kls.iter().sum::<f64>() / w.all_row_kls.len().max(1) as f64,
            median_of(&w.round_kl_means),
            w.all_row_kls.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            w.mirror_top1_agree_rows,
            w.mirror_rows,
        );
        delta_accept.push(w.tally.accept_rate() - tb.accept_rate());
        delta_tpr.push(w.tally.tokens_per_round() - tb.tokens_per_round());
    }

    let n = delta_accept.len();
    let (am, asd, at) = paired_mean_sd_t(&delta_accept);
    let (tm, tsd, tt) = paired_mean_sd_t(&delta_tpr);
    let pos = delta_accept.iter().filter(|d| **d > 0.0).count();
    let neg = delta_accept.iter().filter(|d| **d < 0.0).count();
    eprintln!(
        "LORA-CURVE-PAIRED n={n} mean_delta_accept={am:+.4} sd={asd:.4} t={at:+.2} \
         sign=+{pos}/-{neg} mean_delta_tokens_per_round={tm:+.3} sd_tpr={tsd:.3} t_tpr={tt:+.2} \
         pairing=adapter_minus_base_per_independent_window {basis}"
    );
    assert!(
        n >= WINDOWS_MIN_9_BECAUSE_08_2_SAYS_FEWER_INDEPENDENT_WINDOWS_CANNOT_ADJUDICATE,
        "{n} paired windows landed: fewer than 9 cannot adjudicate"
    );
}
