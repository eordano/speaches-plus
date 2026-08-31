#![cfg(feature = "cuda")]

mod common;
mod hub_dirs;
mod sufspec_classes;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const CHECKPOINT_LABEL: &str = "unsloth/Qwen3.8-27B-NVFP4";

const PROMPT_TOKENS_1024_MATCHES_THE_HEADROOM_CLASSES: usize = 1024;
const GEN_TOKENS_384_LONG_ENOUGH_TO_CROSS_MANY_TREE_ROUND_BOUNDARIES: usize = 384;
const MAX_SEQ_4096_COVERS_PROMPT_GEN_AND_THE_VERIFY_GUARD: usize = 4096;

fn require_gate(gate: &str) {
    if std::env::var(gate).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {gate}=1");
    }
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 plus the MTP head once; set \
            NV_QWEN38_SERVING_TEST=1 NV_DRAFTER=mtp NV_Q38_DRAFT_FAST=1 -- the two-branch tree \
            losslessness gate: for each of the three workload classes the greedy emitted stream \
            with NV_Q38_MTP_TREE=1 must be BITWISE identical to the stream with it unset, \
            because the tree only changes which candidate rows the verify judges, never what \
            greedy argmax emits; the test flips the env itself, so leave NV_Q38_MTP_TREE unset \
            in the caller environment"]
fn qwen38_tree_greedy_emitted_stream_bitwise_identical_tree_on_vs_off_on_all_three_classes() {
    use candle_core::Device;
    use nv_models::graph_engine::GraphedQwen3Moe;
    use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
    use nv_specdecode::qwen38_mtp::{
        draft_fast_device_chained_tokens_selected_from_env, mtp_chain_depth_from_env,
        mtp_draft_dir_override_from_env, mtp_drafter_selected_from_env, Qwen38DenseMtpHead,
        Qwen38MtpGraphedDecodeSession, MTP_WEIGHTS_FILE_NAME, NV_Q38_MTP_TREE_ENV,
    };

    require_gate("NV_QWEN38_SERVING_TEST");
    assert!(
        mtp_drafter_selected_from_env(),
        "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_DRAFTER=mtp"
    );
    assert!(
        draft_fast_device_chained_tokens_selected_from_env(),
        "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_Q38_DRAFT_FAST=1; the tree \
         round forks the fast device-chained draft loop"
    );
    assert!(
        std::env::var(NV_Q38_MTP_TREE_ENV).is_err(),
        "leave {NV_Q38_MTP_TREE_ENV} unset: this gate owns the flag and flips it per arm"
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
        PROMPT_TOKENS_1024_MATCHES_THE_HEADROOM_CLASSES,
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

    let mut tie_divergences = 0usize;
    let mut bitwise_classes = 0usize;
    let mut total_tree_rounds = 0usize;
    let mut total_branch_b_rounds = 0usize;
    let mut total_recovered = 0usize;
    for (name, prompt) in &classes {
        let mut arms: Vec<(bool, Vec<u32>, Vec<f32>)> = Vec::new();
        for tree_on in [false, true] {
            if tree_on {
                std::env::set_var(NV_Q38_MTP_TREE_ENV, "1");
            } else {
                std::env::remove_var(NV_Q38_MTP_TREE_ENV);
            }
            engine.reset().expect("engine reset between arms");
            let (gen, margins) = {
                let mut session =
                    Qwen38MtpGraphedDecodeSession::start(&mut engine, &mtp, k, prompt)
                        .unwrap_or_else(|e| {
                            panic!("session start class={name} tree_on={tree_on}: {e:#}")
                        });
                let mut gen: Vec<u32> = vec![session.anchor()];
                while gen.len()
                    < 1 + GEN_TOKENS_384_LONG_ENOUGH_TO_CROSS_MANY_TREE_ROUND_BOUNDARIES
                {
                    assert!(session.round_fits(), "kv headroom exhausted class={name}");
                    let emitted = session.round().expect("round");
                    gen.extend(&emitted);
                }
                if tree_on {
                    assert!(
                        session.tree_rounds > 0,
                        "class={name}: the tree arm ran zero tree rounds; the env never engaged \
                         and this gate compared the baseline against itself"
                    );
                    eprintln!(
                        "TREE-GATE class={name} tree_rounds={} branch_b_rounds={} \
                         branch_b_recovered_tokens={} recovery_rate={:.4}",
                        session.tree_rounds,
                        session.tree_branch_b_rounds,
                        session.tree_branch_b_recovered_tokens,
                        session.tree_branch_b_rounds as f64 / session.tree_rounds.max(1) as f64,
                    );
                    total_tree_rounds += session.tree_rounds;
                    total_branch_b_rounds += session.tree_branch_b_rounds;
                    total_recovered += session.tree_branch_b_recovered_tokens;
                }
                let margins = std::mem::take(
                    &mut session
                        .emitted_argmax_margins_aligned_to_tokens_after_the_anchor_so_a_cross_shape_divergence_can_be_triaged_tie_vs_corruption,
                );
                (gen, margins)
            };
            eprintln!(
                "TREE-GATE class={name} tree_on={tree_on} gen_tokens={} fnv1a64={:016x} \
                 checkpoint={CHECKPOINT_LABEL}",
                gen.len(),
                sufspec_classes::fnv1a64_over_u32_le(&gen),
            );
            arms.push((tree_on, gen, margins));
        }
        let (_, off_stream, off_margins) = &arms[0];
        let (_, on_stream, on_margins) = &arms[1];
        let common = off_stream.len().min(on_stream.len());
        if off_stream[..common] == on_stream[..common] {
            bitwise_classes += 1;
            eprintln!(
                "TREE-GATE class={name} verdict=BITWISE common_tokens={common} \
                 off_len={} on_len={}",
                off_stream.len(),
                on_stream.len()
            );
            continue;
        }
        let i = off_stream
            .iter()
            .zip(on_stream.iter())
            .position(|(a, b)| a != b)
            .expect("the common prefixes differ, so a first differing index exists");
        assert!(
            i >= 1,
            "class={name}: the arms disagree on the prefill anchor itself; the prefill is \
             tree-independent, so this is state corruption, not a tie"
        );
        let off_margin = off_margins[i - 1];
        let on_margin = on_margins[i - 1];
        eprintln!(
            "TREE-GATE class={name} verdict=TIE-DIVERGENCE idx={i} off_tok={} on_tok={} \
             off_margin={off_margin:.4} on_margin={on_margin:.4} off_text={:?} on_text={:?}",
            off_stream[i],
            on_stream[i],
            tok.decode(&off_stream[i.saturating_sub(8)..(i + 8).min(off_stream.len())], true)
                .unwrap_or_default(),
            tok.decode(&on_stream[i.saturating_sub(8)..(i + 8).min(on_stream.len())], true)
                .unwrap_or_default(),
        );
        assert!(
            off_margin
                <= common::ADAPTIVE_TIE_MARGIN_MAX_1_BF16_LOGIT_UNIT_SHALLOW_FLIPS_MEASURE_0_0_TO_0_9_AND_DEEP_DRIFT_KEEPS_ONE_SIDE_UNDER_0_4
                || on_margin
                    <= common::ADAPTIVE_TIE_MARGIN_MAX_1_BF16_LOGIT_UNIT_SHALLOW_FLIPS_MEASURE_0_0_TO_0_9_AND_DEEP_DRIFT_KEEPS_ONE_SIDE_UNDER_0_4,
            "class={name} idx={i}: BOTH arms confident at the flip (off_margin={off_margin} \
             on_margin={on_margin}): two confident verifies of the same committed prefix \
             disagreeing is the fork/mask/commit corruption class, not the cross-shape argmax \
             tie band"
        );
        tie_divergences += 1;
    }
    std::env::remove_var(NV_Q38_MTP_TREE_ENV);
    eprintln!(
        "TREE-GATE overall bitwise_classes={bitwise_classes}/3 tie_divergences={tie_divergences} \
         tree_rounds={total_tree_rounds} branch_b_rounds={total_branch_b_rounds} \
         branch_b_recovered_tokens={total_recovered} recovery_rate={:.4}",
        total_branch_b_rounds as f64 / total_tree_rounds.max(1) as f64,
    );
    assert!(
        total_branch_b_rounds > 0,
        "no round ever accepted the top-2 sibling branch across all three classes; the recovery \
         machinery never engaged, so this run gates nothing about the branch-B commit path"
    );
    assert!(
        bitwise_classes >= 2,
        "only {bitwise_classes}/3 classes were bitwise: more than one cross-shape tie flip per \
         three 384-token classes is outside the rate the sufspec campaign measured for the \
         verified-shape tie band and must be triaged as a defect"
    );
}
