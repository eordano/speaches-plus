#![cfg(feature = "cuda")]

mod hub_dirs;

use candle_core::{DType, Device, Tensor, D};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeKvCache, Qwen3_5DenseConfig};
use nv_specdecode::qwen38_mtp::{
    mtp_draft_dir_override_from_env, Qwen38DenseMtpHead, Qwen38MtpGraphedDecodeSession,
    MTP_WEIGHTS_FILE_NAME,
};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::Path;
use tokenizers::Tokenizer;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const PROBE_ROUNDS: usize = 16;
const PROBE_K: usize = 3;
const PROBE_MAX_SEQ: usize = 1024;
const PROBE_PROMPT: &str =
    "Write a short story about a lighthouse keeper who discovers something unusual.";

fn render_real_chat_template_no_thinking(dir: &Path, question: &str) -> String {
    use minijinja::{context, Environment, Value as JinjaValue};
    fn raise_exception(msg: String) -> Result<JinjaValue, minijinja::Error> {
        Err(minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            msg,
        ))
    }
    fn strftime_now(_fmt: String) -> Result<JinjaValue, minijinja::Error> {
        Ok(JinjaValue::from("1970-01-01"))
    }
    let source = std::fs::read_to_string(dir.join("chat_template.jinja"))
        .expect("chat_template.jinja must ship with the checkpoint");
    let mut env = Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", raise_exception);
    env.add_function("strftime_now", strftime_now);
    env.add_template_owned("chat", source).expect("template compiles");
    env.get_template("chat")
        .unwrap()
        .render(context! {
            messages => vec![serde_json::json!({"role": "user", "content": question})],
            add_generation_prompt => true,
            enable_thinking => false,
        })
        .expect("the official template must render a plain user turn")
}

fn row_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<f32>())
        .expect("hidden row to host")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

fn max_abs_delta(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .fold(0f64, f64::max)
}

fn shadow_decode_one(
    trunk: &Qwen3Moe,
    cache: &mut Qwen3MoeKvCache,
    tok: u32,
    device: &Device,
) -> (Vec<f32>, u32) {
    let p = cache.current_len();
    let bt = Tensor::from_vec(vec![tok], (1usize, 1usize), device).expect("shadow tok");
    let bp = Tensor::from_vec(vec![p as i32], 1usize, device).expect("shadow pos");
    let (logits, hidden) = trunk
        .forward_with_cache_dispatched_hidden_rows(&bt, &bp, cache, None, None)
        .expect("shadow decode step");
    let greedy = logits
        .argmax(D::Minus1)
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_dtype(DType::U32))
        .and_then(|x| x.to_vec1::<u32>())
        .expect("shadow argmax")[0];
    (row_f32(&hidden), greedy)
}

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 checkpoint; set NV_Q38_MTP=1 -- per-round anchor \
            mismatch probe: the drafter's anchor hidden row (produced by the graphed m=k+1 \
            verify lane for rounds >= 1, by the prefill for round 0) is compared against an \
            eager m=1 decode recomputation of the SAME position on a shadow KV cache fed the \
            identical committed stream; reports cosine/max-abs per round, an off-by-one detector \
            (cosine against the previous position, which must sit far below the same-position \
            cosine), and greedy-token agreement between the two numeric paths; run with \
            NV_Q38_MTP_REANCHOR unset so both rows are pre-norm and directly comparable"]
fn qwen38_mtp_anchor_row_vs_eager_recompute_of_the_same_position() {
    if std::env::var("NV_Q38_MTP").as_deref() != Ok("1") {
        panic!("set NV_Q38_MTP=1 to run (it must never silently skip)");
    }
    assert!(
        std::env::var("NV_Q38_MTP_REANCHOR").as_deref() != Ok("1"),
        "this probe compares raw pre-norm rows; run it with NV_Q38_MTP_REANCHOR unset"
    );
    let dir = hub_dirs::snapshot(
        NVFP4_REPO,
        &[
            "config.json",
            "tokenizer.json",
            "chat_template.jinja",
            MTP_WEIGHTS_FILE_NAME,
        ],
    )
    .expect("Qwen3.8-27B-NVFP4 snapshot with the MTP shard not found in the hub cache");

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("track-2 dense NVFP4 loader");
    drop(weights);
    let mut engine = nv_models::graph_engine::GraphedQwen3Moe::new(base, &device, PROBE_MAX_SEQ)
        .expect("graphed engine");
    let mtp = Qwen38DenseMtpHead::from_checkpoint(
        mtp_draft_dir_override_from_env().as_deref(),
        &dir,
        engine.underlying(),
        &device,
    )
    .expect("MTP head");

    let prompt_text = render_real_chat_template_no_thinking(&dir, PROBE_PROMPT);
    let prompt: Vec<u32> = tok
        .encode(prompt_text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();
    assert!(prompt.len() >= 2, "probe prompt must hold at least 2 tokens");

    let mut shadow_cache = engine
        .underlying()
        .new_kv_cache(PROBE_MAX_SEQ)
        .expect("shadow kv cache");

    let mut session = Qwen38MtpGraphedDecodeSession::start(&mut engine, &mtp, PROBE_K, &prompt)
        .expect("graphed session");

    let (mut shadow_same, mut shadow_prev, mut shadow_greedy) = {
        let trunk = session.trunk_for_shadow_probes();
        let seq = prompt.len();
        let tokens = Tensor::from_vec(prompt.clone(), (1usize, seq), &device).expect("tokens");
        let pos = Tensor::from_vec((0..seq as i32).collect::<Vec<i32>>(), seq, &device)
            .expect("positions");
        let (logits, hidden) = trunk
            .forward_with_cache_dispatched_hidden_rows(
                &tokens,
                &pos,
                &mut shadow_cache,
                None,
                Some(1),
            )
            .expect("shadow prefill");
        let greedy = logits
            .argmax(D::Minus1)
            .and_then(|x| x.flatten_all())
            .and_then(|x| x.to_dtype(DType::U32))
            .and_then(|x| x.to_vec1::<u32>())
            .expect("prefill argmax")[0];
        let same = row_f32(&hidden.narrow(1, seq - 1, 1).expect("last prefill row"));
        let prev = row_f32(&hidden.narrow(1, seq - 2, 1).expect("prev prefill row"));
        let _ = device.synchronize();
        (same, prev, greedy)
    };

    let mut sum_cos = 0f64;
    let mut sum_cos_verify_rounds = 0f64;
    let mut verify_rounds = 0usize;
    let mut tok_agree = 0usize;
    let mut off_by_one_violations: Vec<(usize, f64, f64)> = Vec::new();
    for round in 0..PROBE_ROUNDS {
        let (anchor_tok, anchor_row_t) = session.drafter_anchor_token_and_trunk_hidden_row();
        let anchor_row = row_f32(&anchor_row_t);
        assert_eq!(
            shadow_cache.current_len(),
            session.committed_len(),
            "shadow cache desynced from the session at round {round}"
        );

        let cos_same = cosine(&anchor_row, &shadow_same);
        let cos_prev = cosine(&anchor_row, &shadow_prev);
        let maxabs = max_abs_delta(&anchor_row, &shadow_same);
        let agree = anchor_tok == shadow_greedy;
        let source = if round == 0 { "prefill" } else { "verify-lane" };
        eprintln!(
            "[q38-mtp-anchor-probe] round={round} source={source} pos={} cos_same={cos_same:.6} \
             max_abs={maxabs:.4} cos_prev_pos_off_by_one_detector={cos_prev:.4} \
             greedy_tok_agree_mrow_vs_eager={} anchor_tok={anchor_tok} eager_tok={shadow_greedy}",
            session.committed_len() - 1,
            u8::from(agree),
        );
        sum_cos += cos_same;
        if round > 0 {
            sum_cos_verify_rounds += cos_same;
            verify_rounds += 1;
        }
        tok_agree += usize::from(agree);
        if cos_same <= cos_prev {
            off_by_one_violations.push((round, cos_same, cos_prev));
        }

        if !session.round_fits() {
            break;
        }
        let emitted = session.round().expect("probe round");
        let committed: Vec<u32> = std::iter::once(anchor_tok)
            .chain(emitted[..emitted.len() - 1].iter().copied())
            .collect();
        {
            let trunk = session.trunk_for_shadow_probes();
            for &t in &committed {
                shadow_prev = std::mem::take(&mut shadow_same);
                let (h, g) = shadow_decode_one(trunk, &mut shadow_cache, t, &device);
                shadow_same = h;
                shadow_greedy = g;
            }
            let _ = device.synchronize();
        }
    }

    eprintln!(
        "[q38-mtp-anchor-probe] SUMMARY rounds={PROBE_ROUNDS} k={PROBE_K} \
         mean_cos_same_all={:.6} mean_cos_same_verify_lane_rounds={:.6} \
         greedy_tok_agree={tok_agree}/{PROBE_ROUNDS} \
         basis=(model=unsloth/Qwen3.8-27B-NVFP4 anchor=m-row-verify-lane \
         shadow=eager-m1-decode same committed stream, both pre-norm hidden)",
        sum_cos / PROBE_ROUNDS as f64,
        sum_cos_verify_rounds / verify_rounds.max(1) as f64,
    );
    assert!(
        off_by_one_violations.is_empty(),
        "anchor rows matched the PREVIOUS position better than their own in rounds \
         {off_by_one_violations:?} (round, cos_same, cos_prev); the embed/hidden pairing is \
         off by one"
    );
}
