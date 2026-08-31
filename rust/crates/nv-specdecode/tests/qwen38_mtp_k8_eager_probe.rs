#![cfg(feature = "cuda")]

mod hub_dirs;

use candle_core::Device;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_specdecode::qwen38_mtp::{
    mtp_draft_dir_override_from_env, Qwen38DenseMtpHead, Qwen38MtpSelfSpecEngine,
    MTP_WEIGHTS_FILE_NAME, NV_Q38_MTP_EAGER_DEEP_K_ENV,
};
use nv_weights::{QuantizationConfig, WeightLoader};
use tokenizers::Tokenizer;
mod common;
use common::render_real_chat_template_no_thinking;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";

const PROBE_MAX_NEW_64_MATCHES_THE_K_SWEEP_AND_RECORD_HARNESS: usize = 64;
const PROBE_WARM_TOKENS_16_UNTIMED_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW: usize = 16;
const PROBE_MAX_SEQ: usize = 2048;
const PROBE_DEFAULT_KS_5_AND_8_BRACKET_THE_GRAPHED_CAP: [usize; 2] = [5, 8];

const PROBE_CREATIVE_PROMPTS_MATCH_THE_RECORD_BAR_HONESTY_BRACKET: [&str; 3] = [
    "Write a short story about a lighthouse keeper who discovers something unusual.",
    "Invent a new board game and explain its rules.",
    "Describe an imaginary city built inside a mountain.",
];

const PROBE_WIKITEXT_RAW_CONTINUATION_PROMPTS_MIRROR_LLAMA_CPP_BENCH_SERVER_UNTEMPLATED_CORPUS:
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

const PROBE_DOC_PROMPTS_TARGET_THE_LLAMA_CPP_92_96_PERCENT_ACCEPT_REGIME: [&str; 2] = [
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

const PROBE_README_RAW_CONTINUATION_PROMPT_IS_THE_EXACT_LLAMA_CPP_92_96_PERCENT_CORPUS: [&str; 1] =
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

fn ks_from_env() -> Vec<usize> {
    match std::env::var("NV_MTP_K_PROBE") {
        Ok(raw) => {
            let ks: Vec<usize> = raw
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            assert!(
                !ks.is_empty() && ks.iter().all(|&k| (1..=15).contains(&k)),
                "NV_MTP_K_PROBE must be a comma list of k in 1..=15, got {raw:?}"
            );
            ks
        }
        Err(_) => PROBE_DEFAULT_KS_5_AND_8_BRACKET_THE_GRAPHED_CAP.to_vec(),
    }
}

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 checkpoint ONCE (the eager engine borrows the \
            base, so every swept k shares one load); set NV_Q38_MTP=1; NV_MTP_K_PROBE picks the \
            ks (default 5,8); the point of this probe is k=8 accept economics through the EAGER \
            seq-(k+1) forward, which has no mk-kernel row template bound -- the graphed route is \
            capped at rows=8 by splitk_mk kMaxM and MK_VERIFY_MAX_ROWS; per (k,prompt) it also \
            runs the m=1 eager reference decode and reports the exact-prefix-match length \
            (batched verify rows are not required to be byte-identical to m=1, so a tail \
            divergence is drift, not a spec bug -- the hard gate lives in qwen38_mtp_real_ab)"]
fn qwen38_mtp_k8_accept_economics_on_the_eager_uncapped_path() {
    if std::env::var("NV_Q38_MTP").as_deref() != Ok("1") {
        panic!("set NV_Q38_MTP=1 to run (it must never silently skip)");
    }
    unsafe {
        std::env::set_var(NV_Q38_MTP_EAGER_DEEP_K_ENV, "1");
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
    let max_new = PROBE_MAX_NEW_64_MATCHES_THE_K_SWEEP_AND_RECORD_HARNESS;
    let warm_new = PROBE_WARM_TOKENS_16_UNTIMED_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW;

    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("track-2 dense NVFP4 loader must load Qwen3.8-27B before this probe can run");
    drop(weights);
    let mtp = Qwen38DenseMtpHead::from_checkpoint(
        mtp_draft_dir_override_from_env().as_deref(),
        &dir,
        &base,
        &device,
    )
    .expect("MTP head loads from the shipped model_mtp.safetensors");

    let prompts: Vec<(usize, &str, String)> = PROBE_CREATIVE_PROMPTS_MATCH_THE_RECORD_BAR_HONESTY_BRACKET
        .iter()
        .map(|q| ("creative", render_real_chat_template_no_thinking(&dir, q)))
        .chain(
            PROBE_DOC_PROMPTS_TARGET_THE_LLAMA_CPP_92_96_PERCENT_ACCEPT_REGIME
                .iter()
                .map(|q| ("doc", render_real_chat_template_no_thinking(&dir, q))),
        )
        .chain(
            PROBE_WIKITEXT_RAW_CONTINUATION_PROMPTS_MIRROR_LLAMA_CPP_BENCH_SERVER_UNTEMPLATED_CORPUS
                .iter()
                .map(|q| ("wikitext", q.to_string())),
        )
        .chain(
            PROBE_README_RAW_CONTINUATION_PROMPT_IS_THE_EXACT_LLAMA_CPP_92_96_PERCENT_CORPUS
                .iter()
                .map(|q| ("readme", q.to_string())),
        )
        .enumerate()
        .map(|(pi, (bracket, text))| (pi, bracket, text))
        .collect();

    for &k in &ks {
        let mut k_drafted = 0usize;
        let mut k_accepted = 0usize;
        let mut k_rounds = 0usize;
        let mut k_new = 0usize;
        let mut k_draft_ms = 0f64;
        let mut k_verify_ms = 0f64;

        for (pi, bracket, prompt_text) in &prompts {
            let prompt: Vec<u32> = tok
                .encode(prompt_text.as_str(), false)
                .expect("encode")
                .get_ids()
                .to_vec();
            let eng = Qwen38MtpSelfSpecEngine::new(&base, &mtp, k)
                .expect("eager engine at probe k")
                .with_stop_ids(stop.clone());
            eng.generate_greedy(&prompt, warm_new, PROBE_MAX_SEQ)
                .expect("untimed eager warmup");
            let t0 = std::time::Instant::now();
            let (ids, stats) = eng
                .generate_greedy(&prompt, max_new, PROBE_MAX_SEQ)
                .expect("eager spec generation");
            let spec_s = t0.elapsed().as_secs_f64();
            let (ref_ids, ref_stats) = eng
                .generate_reference(&prompt, max_new, PROBE_MAX_SEQ)
                .expect("eager m=1 reference generation");
            let ref_ms_tok: f64 = ref_stats.round_ms.iter().sum::<f64>()
                / ref_stats.emitted.max(1) as f64;
            let match_len = ids
                .iter()
                .zip(ref_ids.iter())
                .take_while(|(a, b)| a == b)
                .count();
            let new_toks = ids.len();
            let rounds = stats.rounds.max(1);
            eprintln!(
                "[q38-mtp-k8probe] arm=eager k={k} prompt={pi} bracket={bracket} \
                 prompt_toks={} new_toks={new_toks} rounds={} accept={:.3} pos0={:.3} \
                 tokens_per_round={:.2} steady_ms_tok={:.2} draft_ms_per_round={:.2} \
                 verify_ms_per_round={:.2} ref_m1_eager_ms_tok={ref_ms_tok:.2} \
                 ref_prefix_match={match_len}/{} accept_len_hist={} wall_s={spec_s:.2}",
                prompt.len(),
                stats.rounds,
                stats.accept_rate(),
                stats.pos0_accept_rate(),
                stats.tokens_per_round(),
                (stats.draft_ms + stats.verify_ms) / new_toks.max(1) as f64,
                stats.draft_ms / rounds as f64,
                stats.verify_ms / rounds as f64,
                new_toks.min(ref_ids.len()),
                stats
                    .accept_len_hist
                    .iter()
                    .map(|(len, n)| format!("{len}:{n}"))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            k_drafted += stats.drafted;
            k_accepted += stats.accepted;
            k_rounds += stats.rounds;
            k_new += new_toks;
            k_draft_ms += stats.draft_ms;
            k_verify_ms += stats.verify_ms;
        }
        eprintln!(
            "[q38-mtp-k8probe] arm=eager-AGGREGATE k={k} prompts={} new_toks={k_new} \
             rounds={k_rounds} accept={:.3} tokens_per_round={:.2} eager_steady_ms_tok={:.2} \
             draft_ms_per_round={:.2} verify_ms_per_round={:.2} \
             basis=(model=unsloth/Qwen3.8-27B-NVFP4 eager-seq-k-plus-1-verify max_new={max_new} \
             warm={warm_new} template=official-no-thinking-chat-plus-raw-wikitext greedy)",
            prompts.len(),
            k_accepted as f64 / k_drafted.max(1) as f64,
            k_new as f64 / k_rounds.max(1) as f64,
            (k_draft_ms + k_verify_ms) / k_new.max(1) as f64,
            k_draft_ms / k_rounds.max(1) as f64,
            k_verify_ms / k_rounds.max(1) as f64,
        );
    }
}
