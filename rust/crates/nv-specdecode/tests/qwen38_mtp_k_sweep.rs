#![cfg(feature = "cuda")]

mod hub_dirs;

use candle_core::Device;
use minijinja::{context, Environment, Value as JinjaValue};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_specdecode::qwen38_mtp::{
    mtp_draft_dir_override_from_env, mtp_drafter_fp8_resident_from_env, Qwen38DenseMtpHead,
    Qwen38MtpGraphedDecodeSession, Qwen38MtpSelfSpecEngine, MTP_WEIGHTS_FILE_NAME,
};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::Path;
use tokenizers::Tokenizer;
mod common;
use common::raise_exception;
use common::render_real_chat_template_no_thinking;
use common::strftime_now;
use common::GraphedRunRecord;
use common::generate_graphed;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";

const SWEEP_MAX_NEW_64_MATCHES_THE_V3_RECORD_HARNESS: usize = 64;
const SWEEP_WARM_TOKENS_16_UNTIMED_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW: usize = 16;
const SWEEP_MAX_SEQ: usize = 2048;
const SWEEP_MIN_CREATIVE_CONTENT_TOKENS_48_MATCHES_THE_RECORD_BAR: usize = 48;
const SWEEP_MIN_DOC_CONTENT_TOKENS_32_SO_A_SHORT_REFUSAL_CANNOT_POSE_AS_A_MEASUREMENT: usize = 32;
const SWEEP_JUNK_DRAFT_TOKEN_ANY_POLICY_MUST_EMIT_THE_SAME_STREAM: u32 = 0;
const SWEEP_DEFAULT_KS: [usize; 4] = [2, 3, 4, 5];

const SWEEP_CREATIVE_PROMPTS_MATCH_THE_RECORD_BAR_HONESTY_BRACKET: [&str; 3] = [
    "Write a short story about a lighthouse keeper who discovers something unusual.",
    "Invent a new board game and explain its rules.",
    "Describe an imaginary city built inside a mountain.",
];

const SWEEP_WIKITEXT_RAW_CONTINUATION_PROMPTS_MIRROR_LLAMA_CPP_BENCH_SERVER_UNTEMPLATED_CORPUS:
    [&str; 3] = [
    " In 2000 Boulter had a guest @-@ starring role on the television series The Bill ; he \
     portrayed \" Scott Parry \" in the episode , \" In Safe Hands \" . Boulter starred as \
     \" Scott \" in the play Herons written by Simon Stephens , which was performed in 2001 at \
     the Royal Court Theatre . A review of Boulter 's performance in The Independent on Sunday \
     described him as \" horribly menacing \" in the role , and he received critical reviews in",
    " In the early 730s , he travelled in the Jiangsu / Zhejiang area ; his earliest surviving \
     poem , describing a poetry contest , is thought to date from the end of this period , \
     around 735 . In that year , he took the civil service exam , likely in Chang 'an . He \
     failed , to his surprise and that of centuries of later critics . Hung concludes that he \
     probably failed because his prose style at the time was too dense and obscure , while",
    " A second favourite epithet of Chinese critics is that of \" poet sage \" ( \u{8a69}\u{8056} \
     sh\u{12b} sh\u{e8}ng ) , a counterpart to the philosophical sage , Confucius . One of the \
     earliest surviving works , The Song of the Wagons ( from around 750 ) , gives voice to the \
     sufferings of a conscript soldier in the imperial army and a clear @-@ sighted \
     consciousness of suffering . These concerns are continuously articulated in",
];

const SWEEP_README_RAW_CONTINUATION_PROMPT_IS_THE_EXACT_LLAMA_CPP_92_96_PERCENT_CORPUS: [&str; 1] =
    ["# llama.cpp\n\n![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/\
      master/cover/llama-cpp/cover-llama-cpp-dark.svg)\n\n<div align=\"center\">\n\n<b>LLM \
      inference in C/C++</b>\n\n[![License: MIT](https://img.shields.io/badge/license-MIT-blue.\
      svg)](https://opensource.org/licenses/MIT)\n[![Release](https://img.shields.io/github/v/\
      release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)\n[![Server]\
      (https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://\
      github.com/ggml-org/llama.cpp/actions/workflows/server.yml)\n[![Docker](https://github.\
      com/ggml-org/llama.cpp/actions/workflows/docker.yml/badge.svg)](https://github.com/\
      ggml-org/llama.cpp/actions/workflows/docker.yml)\n[![Winget](https://github.com/ggml-org/\
      llama.cpp/actions/workflows/winget.yml/badge.svg)](https://github.com/ggml-org/"];

const SWEEP_DOC_PROMPTS_TARGET_THE_LLAMA_CPP_92_96_PERCENT_ACCEPT_REGIME: [&str; 2] = [
    "Continue this changelog with four more entries in exactly the same format, for versions \
     1.2.5 through 1.2.8:\n\n## 1.2.1\n- Fixed a crash when the config file is missing.\n- \
     Improved startup time by caching the model index.\n\n## 1.2.2\n- Fixed a crash when the \
     audio device is unplugged.\n- Improved startup time by lazy-loading the tokenizer.\n\n## \
     1.2.3\n- Fixed a crash when the network is unreachable.\n- Improved startup time by \
     deferring the license check.\n\n## 1.2.4\n- Fixed a crash when the cache directory is \
     read-only.\n- Improved startup time by precompiling the shaders.",
    "Copy the following paragraph word for word, exactly as written, three times in a row:\n\n\
     The quick brown fox jumps over the lazy dog while the calm gray cat watches from the warm \
     stone wall near the old wooden gate.",
];

fn ks_from_env() -> Vec<usize> {
    match std::env::var("NV_MTP_K_SWEEP") {
        Ok(raw) => {
            let ks: Vec<usize> = raw
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            assert!(
                !ks.is_empty() && ks.iter().all(|&k| (1..=7).contains(&k)),
                "NV_MTP_K_SWEEP must be a comma list of k in 1..=7, got {raw:?}"
            );
            ks
        }
        Err(_) => SWEEP_DEFAULT_KS.to_vec(),
    }
}

fn speaches_daemon_resident_mib_recorded_because_cotenancy_shifts_timings() -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            s.lines()
                .find(|l| l.contains("speaches-plus"))
                .and_then(|l| l.rsplit(',').next().map(|v| v.trim().to_string()))
        })
        .unwrap_or_else(|| "0".into())
}

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 checkpoint once per swept k (the graphed engine \
            holds one verify-lane shape for its lifetime and consumes the base by value, so a \
            new k means a new engine and a reload); set NV_Q38_MTP=1; NV_MTP_K_SWEEP picks the \
            ks (default 2,3,4,5); NV_MTP_SWEEP_EAGER=1 adds an eager v2-style arm whose accept \
            delta vs the graphed arm isolates what the m-row verify numerics cost acceptance; \
            NV_Q38_DRAFT_FP8=1 loads the drafter through the resident e4m3 arm for the drafter-cost \
            A/B; prompts are the 3 record-bar creative ones plus 2 doc-continuation ones \
            bracketing the high-accept regime; correctness bar per (k,prompt) is draft-policy \
            invariance through the graphed path"]
fn qwen38_mtp_k_sweep_accept_economics_on_the_graphed_v3_path() {
    if std::env::var("NV_Q38_MTP").as_deref() != Ok("1") {
        panic!("set NV_Q38_MTP=1 to run (it must never silently skip)");
    }
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
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tok.token_to_id(s))
        .collect();
    assert!(!stop.is_empty(), "tokenizer lost the im_end/endoftext stop tokens");

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let ks = ks_from_env();
    let eager_arm = std::env::var("NV_MTP_SWEEP_EAGER").as_deref() == Ok("1");
    let fp8_drafter = mtp_drafter_fp8_resident_from_env();
    let lmhead4_drafter =
        u8::from(nv_specdecode::qwen38_mtp::mtp_draft_lm_head_nvfp4_twin_from_env());
    let reanchor_flag =
        u8::from(nv_specdecode::qwen38_mtp::mtp_reanchor_post_norm_selected_from_env());
    let daemon_mib = speaches_daemon_resident_mib_recorded_because_cotenancy_shifts_timings();
    let max_new = SWEEP_MAX_NEW_64_MATCHES_THE_V3_RECORD_HARNESS;
    let warm_new = SWEEP_WARM_TOKENS_16_UNTIMED_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW;

    let prompts: Vec<(usize, &str, &str, usize, bool)> =
        SWEEP_CREATIVE_PROMPTS_MATCH_THE_RECORD_BAR_HONESTY_BRACKET
            .iter()
            .map(|q| {
                (
                    "creative",
                    *q,
                    SWEEP_MIN_CREATIVE_CONTENT_TOKENS_48_MATCHES_THE_RECORD_BAR,
                    false,
                )
            })
            .chain(
                SWEEP_DOC_PROMPTS_TARGET_THE_LLAMA_CPP_92_96_PERCENT_ACCEPT_REGIME
                    .iter()
                    .map(|q| {
                        (
                            "doc",
                            *q,
                            SWEEP_MIN_DOC_CONTENT_TOKENS_32_SO_A_SHORT_REFUSAL_CANNOT_POSE_AS_A_MEASUREMENT,
                            false,
                        )
                    }),
            )
            .chain(
                SWEEP_WIKITEXT_RAW_CONTINUATION_PROMPTS_MIRROR_LLAMA_CPP_BENCH_SERVER_UNTEMPLATED_CORPUS
                    .iter()
                    .map(|q| {
                        (
                            "wikitext",
                            *q,
                            SWEEP_MIN_DOC_CONTENT_TOKENS_32_SO_A_SHORT_REFUSAL_CANNOT_POSE_AS_A_MEASUREMENT,
                            true,
                        )
                    }),
            )
            .chain(
                SWEEP_README_RAW_CONTINUATION_PROMPT_IS_THE_EXACT_LLAMA_CPP_92_96_PERCENT_CORPUS
                    .iter()
                    .map(|q| {
                        (
                            "readme",
                            *q,
                            SWEEP_MIN_DOC_CONTENT_TOKENS_32_SO_A_SHORT_REFUSAL_CANNOT_POSE_AS_A_MEASUREMENT,
                            true,
                        )
                    }),
            )
            .enumerate()
            .map(|(pi, (bracket, q, min_content, raw_continuation))| {
                (pi, bracket, q, min_content, raw_continuation)
            })
            .collect();

    for &k in &ks {
        let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
        let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
        let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
        let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
            .expect("track-2 dense NVFP4 loader must load Qwen3.8-27B before this sweep can run");
        drop(weights);
        let mut engine =
            nv_models::graph_engine::GraphedQwen3Moe::new(base, &device, SWEEP_MAX_SEQ)
                .expect("graphed engine over the dense arm");
        let mtp = Qwen38DenseMtpHead::from_checkpoint(
            mtp_draft_dir_override_from_env().as_deref(),
            &dir,
            engine.underlying(),
            &device,
        )
        .expect("MTP head loads from the shipped model_mtp.safetensors");

        let mut k_new_toks = 0usize;
        let mut k_rounds = 0usize;
        let mut k_drafted = 0usize;
        let mut k_accepted = 0usize;
        let mut k_draft_ms = 0f64;
        let mut k_verify_ms = 0f64;
        let mut k_commit_ms = 0f64;
        let mut k_hot_ms = 0f64;
        let mut k_hot_toks = 0usize;

        let mut bracket_drafted: std::collections::BTreeMap<&str, usize> = Default::default();
        let mut bracket_accepted: std::collections::BTreeMap<&str, usize> = Default::default();
        let mut bracket_rounds: std::collections::BTreeMap<&str, usize> = Default::default();
        let mut bracket_emitted: std::collections::BTreeMap<&str, usize> = Default::default();

        for &(pi, bracket, question, min_content, raw_continuation) in &prompts {
            let prompt_text = if raw_continuation {
                question.to_string()
            } else {
                render_real_chat_template_no_thinking(&dir, question)
            };
            let prompt: Vec<u32> = tok
                .encode(prompt_text.as_str(), false)
                .expect("encode")
                .get_ids()
                .to_vec();

            generate_graphed(&mut engine, &mtp, k, &prompt, warm_new, &stop, None);
            let t0 = std::time::Instant::now();
            let spec = generate_graphed(&mut engine, &mtp, k, &prompt, max_new, &stop, None);
            let spec_s = t0.elapsed().as_secs_f64();
            let junk = generate_graphed(
                &mut engine,
                &mtp,
                k,
                &prompt,
                max_new,
                &stop,
                Some(SWEEP_JUNK_DRAFT_TOKEN_ANY_POLICY_MUST_EMIT_THE_SAME_STREAM),
            );
            assert_eq!(
                spec.ids, junk.ids,
                "draft-policy invariance broken at k={k} prompt={pi} through the GRAPHED verify: \
                 every emitted token is the batched verify's own greedy continuation of the \
                 committed prefix; a mismatch is a rollback/reanchor/lin-ckpt/graph-replay bug"
            );
            eprintln!(
                "[q38-mtp-ksweep] arm=graphed-IDS k={k} prompt={pi} bracket={bracket} \
                 ids_are_the_verify_arms_own_greedy_stream_so_a_cross_arm_diff_locates_the_first_argmax_flip={:?}",
                spec.ids
            );
            let content = spec.ids.iter().filter(|t| !stop.contains(t)).count();
            if raw_continuation {
                let text = tok.decode(&spec.ids, true).unwrap_or_default();
                eprintln!(
                    "[q38-mtp-ksweep] arm=graphed k={k} prompt={pi} bracket={bracket} \
                     text={:?}",
                    text.chars().take(180).collect::<String>()
                );
            }
            assert!(
                content >= min_content,
                "k={k} prompt={pi} ({bracket}) elicited only {content} content tokens of the >= \
                 {min_content} this sweep requires; rates measured on a stub answer flatter \
                 acceptance"
            );

            let stats = &spec.stats;
            let new_toks = spec.ids.len();
            let rounds = stats.rounds.max(1);
            let steady_ms_tok = (stats.draft_ms + stats.verify_ms) / new_toks as f64;
            let incl_commit_ms_tok =
                (stats.draft_ms + stats.verify_ms + stats.commit_ms) / new_toks as f64;
            let (hot_ms, hot_toks) = spec
                .round_wall_ms
                .iter()
                .zip(spec.round_emitted.iter())
                .skip(1)
                .fold((0f64, 0usize), |(ms, tk), (w, e)| (ms + w, tk + e));
            let hot_ms_tok = hot_ms / hot_toks.max(1) as f64;
            let capture_round0_ms = spec.round_wall_ms.first().copied().unwrap_or(0.0);
            let hist: Vec<String> = stats
                .accept_len_hist
                .iter()
                .map(|(len, n)| format!("{len}:{n}"))
                .collect();
            eprintln!(
                "[q38-mtp-ksweep] arm=graphed k={k} prompt={pi} bracket={bracket} \
                 prompt_toks={} new_toks={new_toks} content={content} rounds={} \
                 accept={:.3} pos0={:.3} tokens_per_round={:.2} steady_ms_tok={steady_ms_tok:.2} \
                 incl_commit_ms_tok={incl_commit_ms_tok:.2} hot_ms_tok={hot_ms_tok:.2} \
                 capture_round0_ms={capture_round0_ms:.1} draft_ms_per_round={:.2} \
                 draft_ms_per_draft_tok={:.2} verify_ms_per_round={:.2} \
                 commit_ms_per_round={:.2} accept_len_hist={} wall_s={spec_s:.2} \
                 fp8_drafter={} lmhead4_drafter={lmhead4_drafter} reanchor={reanchor_flag} \
                 daemon_resident_mib={daemon_mib} invariance=held",
                prompt.len(),
                stats.rounds,
                stats.accept_rate(),
                stats.pos0_accept_rate(),
                stats.tokens_per_round(),
                stats.draft_ms / rounds as f64,
                stats.draft_ms / stats.drafted.max(1) as f64,
                stats.verify_ms / rounds as f64,
                stats.commit_ms / rounds as f64,
                hist.join(","),
                u8::from(fp8_drafter),
            );

            k_new_toks += new_toks;
            k_rounds += stats.rounds;
            k_drafted += stats.drafted;
            k_accepted += stats.accepted;
            k_draft_ms += stats.draft_ms;
            k_verify_ms += stats.verify_ms;
            k_commit_ms += stats.commit_ms;
            k_hot_ms += hot_ms;
            k_hot_toks += hot_toks;
            *bracket_drafted.entry(bracket).or_default() += stats.drafted;
            *bracket_accepted.entry(bracket).or_default() += stats.accepted;
            *bracket_rounds.entry(bracket).or_default() += stats.rounds;
            *bracket_emitted.entry(bracket).or_default() += stats.emitted;

            if eager_arm {
                let eng = Qwen38MtpSelfSpecEngine::new(engine.underlying(), &mtp, k)
                    .expect("eager v2-style arm engine")
                    .with_stop_ids(stop.clone());
                eng.generate_greedy(&prompt, warm_new, SWEEP_MAX_SEQ)
                    .expect("untimed eager warmup");
                let (eids, estats) = eng
                    .generate_greedy(&prompt, max_new, SWEEP_MAX_SEQ)
                    .expect("eager spec arm");
                let e_new = eids.len();
                eprintln!(
                    "[q38-mtp-ksweep] arm=eager k={k} prompt={pi} bracket={bracket} \
                     new_toks={e_new} accept={:.3} pos0={:.3} tokens_per_round={:.2} \
                     steady_ms_tok={:.2} accept_delta_graphed_minus_eager={:+.3} \
                     fp8_drafter={} daemon_resident_mib={daemon_mib}",
                    estats.accept_rate(),
                    estats.pos0_accept_rate(),
                    estats.tokens_per_round(),
                    (estats.draft_ms + estats.verify_ms) / e_new.max(1) as f64,
                    stats.accept_rate() - estats.accept_rate(),
                    u8::from(fp8_drafter),
                );
            }
        }

        for (bracket, drafted) in &bracket_drafted {
            let accepted = bracket_accepted.get(bracket).copied().unwrap_or(0);
            let rounds = bracket_rounds.get(bracket).copied().unwrap_or(0);
            let emitted = bracket_emitted.get(bracket).copied().unwrap_or(0);
            eprintln!(
                "[q38-mtp-ksweep] arm=graphed-BRACKET k={k} bracket={bracket} \
                 accept={:.3} mean_len_llama_cpp_def={:.2} drafted={drafted} \
                 accepted={accepted} rounds={rounds} reanchor={reanchor_flag} \
                 llama_cpp_brackets=(chat 0.49-0.58 len 2.48-2.74; doc 0.92-0.96 len 3.71-3.88 \
                 on bartowski Q4_K_M n_max=3)",
                accepted as f64 / (*drafted).max(1) as f64,
                emitted as f64 / rounds.max(1) as f64,
            );
        }
        eprintln!(
            "[q38-mtp-ksweep] arm=graphed-AGGREGATE k={k} prompts={} new_toks={k_new_toks} \
             rounds={k_rounds} accept={:.3} tokens_per_round={:.2} steady_ms_tok={:.2} \
             incl_commit_ms_tok={:.2} hot_ms_tok={:.2} draft_ms_per_round={:.2} \
             verify_ms_per_round={:.2} commit_ms_per_round={:.2} draft_share_of_steady={:.3} \
             fp8_drafter={} lmhead4_drafter={lmhead4_drafter} reanchor={reanchor_flag} \
             daemon_resident_mib={daemon_mib} \
             basis=(model=unsloth/Qwen3.8-27B-NVFP4 commit-harness=k-sweep max_new={max_new} \
             warm={warm_new} template=official-no-thinking-chat-plus-raw-wikitext greedy \
             graphed m=k+1 verify)",
            prompts.len(),
            k_accepted as f64 / k_drafted.max(1) as f64,
            k_new_toks as f64 / k_rounds.max(1) as f64,
            (k_draft_ms + k_verify_ms) / k_new_toks.max(1) as f64,
            (k_draft_ms + k_verify_ms + k_commit_ms) / k_new_toks.max(1) as f64,
            k_hot_ms / k_hot_toks.max(1) as f64,
            k_draft_ms / k_rounds.max(1) as f64,
            k_verify_ms / k_rounds.max(1) as f64,
            k_commit_ms / k_rounds.max(1) as f64,
            k_draft_ms / (k_draft_ms + k_verify_ms).max(1e-9),
            u8::from(fp8_drafter),
        );
    }
}
