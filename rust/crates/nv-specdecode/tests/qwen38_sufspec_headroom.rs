#![cfg(feature = "cuda")]

mod hub_dirs;
mod sufspec_classes;

use nv_specdecode::suffix_automaton::SuffixAutomaton;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const CHECKPOINT_LABEL: &str = "unsloth/Qwen3.8-27B-NVFP4";

const PROMPT_TOKENS_1024_PER_CLASS_KEEPS_ALL_THREE_CLASSES_IN_ONE_ENGINE_BUILD: usize = 1024;
const GEN_TOKENS_512_ENOUGH_ROUNDS_FOR_HIT_RATE_AND_ACCEPT_LEN_DISTRIBUTIONS: usize = 512;
const MAX_SEQ_4096_COVERS_PROMPT_GEN_AND_THE_VERIFY_GUARD: usize = 4096;
const SIM_PROPOSE_CAP_64_MEASURES_UNCLAMPED_HEADROOM_PAST_THE_8_ROW_VERIFY_CEILING: usize = 64;
const SIM_MIN_MATCH_GRID: [usize; 5] = [2, 4, 8, 16, 32];

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn require_gate(gate: &str) {
    if std::env::var(gate).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {gate}=1");
    }
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 plus the MTP head once; set \
            NV_QWEN38_SERVING_TEST=1 and NV_DRAFTER=mtp -- offline suffix-drafting headroom: \
            generates 512 greedy tokens per workload class (agentic-repetitive transcript, tiled \
            doc paragraph, free-prose corpus) through the graphed MTP session, then replays a \
            suffix-automaton drafter over each emitted stream WITHOUT touching the GPU again, \
            reporting hit rate and acceptable draft length per min_match against the session's \
            real MTP tokens-per-round; NV_Q38_SPEC_SUFFIX must be UNSET so the measured stream \
            is the plain-arm stream"]
fn qwen38_sufspec_offline_headroom_hit_rate_and_acceptable_len_per_class_vs_real_mtp_rounds() {
    use candle_core::Device;
    use nv_models::graph_engine::GraphedQwen3Moe;
    use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
    use nv_specdecode::qwen38_mtp::{
        mtp_chain_depth_from_env, mtp_draft_dir_override_from_env, mtp_drafter_selected_from_env,
        Qwen38DenseMtpHead, Qwen38MtpGraphedDecodeSession, MTP_WEIGHTS_FILE_NAME,
    };

    require_gate("NV_QWEN38_SERVING_TEST");
    assert!(
        mtp_drafter_selected_from_env(),
        "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_DRAFTER=mtp"
    );
    assert!(
        !nv_specdecode::qwen38_sufspec::spec_suffix_selected_from_env(),
        "headroom must measure the plain mtp stream: unset NV_Q38_SPEC_SUFFIX"
    );
    let dir = match std::env::var("NV_QWEN38_DIR") {
        Ok(d) if !d.is_empty() => std::path::PathBuf::from(d),
        _ => hub_dirs::snapshot(
            NVFP4_REPO,
            &["config.json", "tokenizer.json", MTP_WEIGHTS_FILE_NAME],
        )
        .unwrap_or_else(|| {
            panic!(
                "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: no hydrated {NVFP4_REPO} \
                 snapshot carrying the MTP shard; set NV_QWEN38_DIR"
            )
        }),
    };
    let tok =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let classes = sufspec_classes::three_class_prompts(
        &tok,
        PROMPT_TOKENS_1024_PER_CLASS_KEEPS_ALL_THREE_CLASSES_IN_ONE_ENGINE_BUILD,
    );

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = nv_weights::QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("track-2 dense NVFP4 loader");
    drop(weights);
    let mut engine =
        GraphedQwen3Moe::new(base, &device, MAX_SEQ_4096_COVERS_PROMPT_GEN_AND_THE_VERIFY_GUARD)
            .expect("graphed engine");
    let mtp = Qwen38DenseMtpHead::from_checkpoint(
        mtp_draft_dir_override_from_env().as_deref(),
        &dir,
        engine.underlying(),
        &device,
    )
    .expect("MTP head");
    let k = mtp_chain_depth_from_env();

    for (name, prompt) in &classes {
        let (gen, round_emits, mtp_rounds, mtp_drafted, mtp_accepted) = {
            let mut session = Qwen38MtpGraphedDecodeSession::start(&mut engine, &mtp, k, prompt)
                .unwrap_or_else(|e| panic!("mtp session start for class {name}: {e:#}"));
            let mut gen: Vec<u32> = vec![session.anchor()];
            let mut round_emits: Vec<usize> = Vec::new();
            while gen.len()
                < 1 + GEN_TOKENS_512_ENOUGH_ROUNDS_FOR_HIT_RATE_AND_ACCEPT_LEN_DISTRIBUTIONS
            {
                assert!(session.round_fits(), "kv headroom exhausted for class {name}");
                let emitted = session.round().expect("mtp round");
                round_emits.push(emitted.len());
                gen.extend(&emitted);
            }
            (
                gen,
                round_emits,
                session.stats.rounds,
                session.stats.drafted,
                session.stats.accepted,
            )
        };
        let mtp_tpr = (gen.len() - 1) as f64 / mtp_rounds as f64;
        let mtp_accept_rate = mtp_accepted as f64 / mtp_drafted.max(1) as f64;
        let preview: String = tok
            .decode(&gen[..gen.len().min(48)], false)
            .unwrap_or_default()
            .chars()
            .map(|c| if c == '\n' { '\u{23CE}' } else { c })
            .take(160)
            .collect();
        eprintln!(
            "SUFSPEC-GEN class={name} prompt_tokens={} gen_tokens={} mtp_rounds={mtp_rounds} \
             mtp_tokens_per_round={mtp_tpr:.2} mtp_accept_rate={mtp_accept_rate:.3} \
             checkpoint={CHECKPOINT_LABEL} preview={preview:?}",
            prompt.len(),
            gen.len() - 1,
        );

        for mm in SIM_MIN_MATCH_GRID {
            let mut sam = SuffixAutomaton::new();
            sam.extend_slice(prompt);
            sam.extend(gen[0]);
            let mut t = 1usize;
            let mut sim_rounds = 0usize;
            let mut hits = 0usize;
            let mut offered: usize = 0;
            let mut acc7_sum: usize = 0;
            let mut acc64_sum: usize = 0;
            let mut mtp_i = 0usize;
            while t < gen.len() {
                let p = sam.propose(
                    SIM_PROPOSE_CAP_64_MEASURES_UNCLAMPED_HEADROOM_PAST_THE_8_ROW_VERIFY_CEILING,
                    mm,
                );
                sim_rounds += 1;
                let adv = match p.filter(|p| {
                    p.tokens.len()
                        >= nv_specdecode::qwen38_sufspec::SUFFIX_MIN_DRAFT_4_A_SUFFIX_ROUND_MUST_OFFER_STRICTLY_MORE_THAN_THE_K3_MTP_CHAIN_ELSE_THE_MTP_ROUND_IS_THE_BETTER_BET
                }) {
                    Some(p) => {
                        hits += 1;
                        offered += p.tokens.len();
                        let a64 = lcp(&p.tokens, &gen[t..]);
                        let a7 = lcp(&p.tokens[..p.tokens.len().min(7)], &gen[t..]);
                        acc64_sum += a64;
                        acc7_sum += a7;
                        a7 + 1
                    }
                    None => {
                        let adv = round_emits[mtp_i % round_emits.len()];
                        mtp_i += 1;
                        adv
                    }
                };
                let end = (t + adv).min(gen.len());
                sam.extend_slice(&gen[t..end]);
                t = end;
            }
            let hit_rate = hits as f64 / sim_rounds as f64;
            let est_tpr = (gen.len() - 1) as f64 / sim_rounds as f64;
            eprintln!(
                "SUFSPEC-HEADROOM class={name} min_match={mm} sim_rounds={sim_rounds} \
                 hits={hits} hit_rate={hit_rate:.3} mean_offered_on_hit={:.2} \
                 mean_accepted7_on_hit={:.2} mean_accepted64_on_hit={:.2} \
                 est_tokens_per_round_hybrid={est_tpr:.2} mtp_tokens_per_round={mtp_tpr:.2}",
                offered as f64 / hits.max(1) as f64,
                acc7_sum as f64 / hits.max(1) as f64,
                acc64_sum as f64 / hits.max(1) as f64,
            );
        }
        engine.reset().expect("engine reset between classes");
    }
}
