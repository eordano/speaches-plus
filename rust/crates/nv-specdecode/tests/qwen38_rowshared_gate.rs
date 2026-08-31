#![cfg(feature = "cuda")]

mod common;
mod hub_dirs;
mod sufspec_classes;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const CHECKPOINT_LABEL: &str = "unsloth/Qwen3.8-27B-NVFP4";

const PROMPT_TOKENS_1024_MATCHES_THE_TREE_AND_HEADROOM_GATES: usize = 1024;
const GEN_TOKENS_384_LONG_ENOUGH_TO_CROSS_MANY_VERIFY_ROUNDS: usize = 384;
const MAX_SEQ_4096_COVERS_PROMPT_GEN_AND_THE_VERIFY_GUARD: usize = 4096;

fn require_gate(gate: &str) {
    if std::env::var(gate).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {gate}=1");
    }
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 plus the MTP head once; set \
            NV_QWEN38_SERVING_TEST=1 NV_DRAFTER=mtp NV_Q38_DRAFT_FAST=1 -- the rowshared verify \
            arm losslessness gate: for each of the three workload classes the greedy emitted \
            stream with NV_Q38_MK_VERIFY_ROWSHARED=1 must be bitwise identical to the stream \
            with it unset, or diverge only inside the documented cross-arm argmax tie band, \
            because the arm only reorders the f32 online-softmax accumulation over the same fp8 \
            kv rows; the test flips the env itself, so leave NV_Q38_MK_VERIFY_ROWSHARED unset \
            in the caller environment"]
fn qwen38_rowshared_greedy_emitted_stream_bitwise_or_tie_band_vs_the_mk_arm_on_all_three_classes()
{
    use candle_core::Device;
    use nv_models::graph_engine::GraphedQwen3Moe;
    use nv_models::qwen3_5_moe::{
        Qwen3Moe, Qwen3_5DenseConfig,
        MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
    };
    use nv_specdecode::qwen38_mtp::{
        draft_fast_device_chained_tokens_selected_from_env, mtp_chain_depth_from_env,
        mtp_draft_dir_override_from_env, mtp_drafter_selected_from_env, Qwen38DenseMtpHead,
        Qwen38MtpGraphedDecodeSession, MTP_WEIGHTS_FILE_NAME,
    };

    require_gate("NV_QWEN38_SERVING_TEST");
    assert!(
        mtp_drafter_selected_from_env(),
        "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_DRAFTER=mtp"
    );
    assert!(
        draft_fast_device_chained_tokens_selected_from_env(),
        "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_Q38_DRAFT_FAST=1; the gate \
         compares the serving verify round"
    );
    assert!(
        std::env::var(
            MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING
        )
        .is_err(),
        "leave {MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING} \
         unset: this gate owns the flag and flips it per arm"
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
        PROMPT_TOKENS_1024_MATCHES_THE_TREE_AND_HEADROOM_GATES,
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
    for (name, prompt) in &classes {
        let mut arms: Vec<(bool, Vec<u32>, Vec<f32>)> = Vec::new();
        for rowshared_on in [false, true] {
            if rowshared_on {
                std::env::set_var(
                    MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
                    "1",
                );
            } else {
                std::env::remove_var(
                    MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
                );
            }
            engine.reset().expect("engine reset between arms");
            let (gen, margins) = {
                let mut session =
                    Qwen38MtpGraphedDecodeSession::start(&mut engine, &mtp, k, prompt)
                        .unwrap_or_else(|e| {
                            panic!("session start class={name} rowshared_on={rowshared_on}: {e:#}")
                        });
                let mut gen: Vec<u32> = vec![session.anchor()];
                while gen.len() < 1 + GEN_TOKENS_384_LONG_ENOUGH_TO_CROSS_MANY_VERIFY_ROUNDS {
                    assert!(session.round_fits(), "kv headroom exhausted class={name}");
                    let emitted = session.round().expect("round");
                    gen.extend(&emitted);
                }
                let margins = std::mem::take(
                    &mut session
                        .emitted_argmax_margins_aligned_to_tokens_after_the_anchor_so_a_cross_shape_divergence_can_be_triaged_tie_vs_corruption,
                );
                (gen, margins)
            };
            eprintln!(
                "ROWSHARED-GATE class={name} rowshared_on={rowshared_on} gen_tokens={} \
                 fnv1a64={:016x} checkpoint={CHECKPOINT_LABEL}",
                gen.len(),
                sufspec_classes::fnv1a64_over_u32_le(&gen),
            );
            arms.push((rowshared_on, gen, margins));
        }
        let (_, off_stream, off_margins) = &arms[0];
        let (_, on_stream, on_margins) = &arms[1];
        let common = off_stream.len().min(on_stream.len());
        if off_stream[..common] == on_stream[..common] {
            bitwise_classes += 1;
            eprintln!(
                "ROWSHARED-GATE class={name} verdict=BITWISE common_tokens={common} \
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
            "class={name}: the arms disagree on the prefill anchor itself; the prefill never \
             routes through verify_attention_fp8_mk, so this is state corruption, not a tie"
        );
        let off_margin = off_margins[i - 1];
        let on_margin = on_margins[i - 1];
        eprintln!(
            "ROWSHARED-GATE class={name} verdict=TIE-DIVERGENCE idx={i} off_tok={} on_tok={} \
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
             disagreeing is the indexing/masking corruption class, not the cross-arm \
             reduction-order tie band"
        );
        tie_divergences += 1;
    }
    std::env::remove_var(
        MK_VERIFY_ROWSHARED_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
    );
    eprintln!(
        "ROWSHARED-GATE overall bitwise_classes={bitwise_classes}/3 \
         tie_divergences={tie_divergences}"
    );
    assert_eq!(
        bitwise_classes + tie_divergences,
        3,
        "a class diverged outside the tie band; the per-class corruption assert above should \
         have caught it, so reaching here means this tally drifted from the loop"
    );
}
