#![cfg(feature = "cuda")]

mod hub_dirs;
mod sufspec_classes;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const CHECKPOINT_LABEL: &str = "unsloth/Qwen3.8-27B-NVFP4";

const PROMPT_TOKENS_1024_MATCHES_THE_HEADROOM_CLASSES: usize = 1024;
const GEN_TOKENS_384_LONG_ENOUGH_TO_CROSS_MANY_SUFFIX_AND_MTP_ROUND_BOUNDARIES: usize = 384;
const MAX_SEQ_4096_COVERS_PROMPT_GEN_AND_THE_VERIFY_GUARD: usize = 4096;

fn require_gate(gate: &str) {
    if std::env::var(gate).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {gate}=1");
    }
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 plus the MTP head once; set \
            NV_QWEN38_SERVING_TEST=1 and NV_DRAFTER=mtp -- the sufspec losslessness gate: for \
            each of the three workload classes (agentic-repetitive, tiled doc, free prose) the \
            greedy emitted stream with NV_Q38_SPEC_SUFFIX=1 must be BITWISE identical to the \
            stream with it unset, because suffix drafts only change which chains the verify \
            judges, never what greedy argmax emits; the test flips the env itself, so leave \
            NV_Q38_SPEC_SUFFIX unset in the caller environment"]
fn qwen38_sufspec_greedy_emitted_stream_bitwise_identical_suffix_on_vs_off_on_all_three_classes() {
    use candle_core::Device;
    use nv_models::graph_engine::GraphedQwen3Moe;
    use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
    use nv_specdecode::qwen38_mtp::{
        mtp_chain_depth_from_env, mtp_draft_dir_override_from_env, mtp_drafter_selected_from_env,
        Qwen38DenseMtpHead, Qwen38MtpGraphedDecodeSession, MTP_WEIGHTS_FILE_NAME,
    };
    use nv_specdecode::qwen38_sufspec::NV_Q38_SPEC_SUFFIX_ENV;

    require_gate("NV_QWEN38_SERVING_TEST");
    assert!(
        mtp_drafter_selected_from_env(),
        "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_DRAFTER=mtp"
    );
    assert!(
        std::env::var(NV_Q38_SPEC_SUFFIX_ENV).is_err(),
        "leave {NV_Q38_SPEC_SUFFIX_ENV} unset: this gate owns the flag and flips it per arm"
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

    let mut class_divergences: Vec<String> = Vec::new();
    for (name, prompt) in &classes {
        let mut arms: Vec<(bool, Vec<u32>, usize)> = Vec::new();
        for suffix_on in [false, true] {
            if suffix_on {
                std::env::set_var(NV_Q38_SPEC_SUFFIX_ENV, "1");
            } else {
                std::env::remove_var(NV_Q38_SPEC_SUFFIX_ENV);
            }
            engine.reset().expect("engine reset between arms");
            let (gen, suffix_rounds) = {
                let mut session =
                    Qwen38MtpGraphedDecodeSession::start(&mut engine, &mtp, k, prompt)
                        .unwrap_or_else(|e| {
                            panic!("session start class={name} suffix_on={suffix_on}: {e:#}")
                        });
                let mut gen: Vec<u32> = vec![session.anchor()];
                while gen.len()
                    < 1 + GEN_TOKENS_384_LONG_ENOUGH_TO_CROSS_MANY_SUFFIX_AND_MTP_ROUND_BOUNDARIES
                {
                    assert!(session.round_fits(), "kv headroom exhausted class={name}");
                    let emitted = session.round().expect("round");
                    gen.extend(&emitted);
                }
                let sr = session
                    .suffix_rounds_drafted_accepted()
                    .map(|(r, _, _)| r)
                    .unwrap_or(0);
                assert_eq!(
                    session.suffix_rounds_drafted_accepted().is_some(),
                    suffix_on,
                    "the suffix arm must exist exactly when the env selects it"
                );
                (gen, sr)
            };
            arms.push((suffix_on, gen, suffix_rounds));
        }
        std::env::remove_var(NV_Q38_SPEC_SUFFIX_ENV);
        let (_, off_stream, _) = &arms[0];
        let (_, on_stream, on_suffix_rounds) = &arms[1];
        let common = off_stream.len().min(on_stream.len());
        if off_stream[..common] != on_stream[..common] {
            let first_diff = off_stream[..common]
                .iter()
                .zip(&on_stream[..common])
                .position(|(a, b)| a != b)
                .unwrap_or(common);
            let line = format!(
                "class={name}: greedy emitted streams diverge between suffix-off and suffix-on \
                 at token {first_diff}/{common} (off tok={} on tok={}); suffix drafts changed \
                 WHAT was emitted, not just timing -- the hybrid is not lossless at this \
                 draft-max (fnv off={:016x} on={:016x})",
                off_stream[first_diff],
                on_stream[first_diff],
                sufspec_classes::fnv1a64_over_u32_le(&off_stream[..common]),
                sufspec_classes::fnv1a64_over_u32_le(&on_stream[..common]),
            );
            eprintln!("SUFSPEC-GATE RED {line}");
            class_divergences.push(line);
            continue;
        }
        if *name != "freeform" {
            assert!(
                *on_suffix_rounds > 0,
                "class={name}: the suffix arm never drafted on a repetitive class, so this gate \
                 compared two plain mtp runs and covered nothing; lower the thresholds or fix \
                 the arm before trusting the identity"
            );
        }
        eprintln!(
            "SUFSPEC-GATE class={name} common_tokens={common} fnv1a64={:016x} \
             suffix_rounds_on_arm={on_suffix_rounds} suffix_draft_max={} verify_tc={} \
             checkpoint={CHECKPOINT_LABEL}",
            sufspec_classes::fnv1a64_over_u32_le(&off_stream[..common]),
            nv_specdecode::qwen38_sufspec::suffix_draft_max_from_env(),
            std::env::var("NV_Q38_VERIFY_TC").unwrap_or_else(|_| "unset".into()),
        );
    }
    assert!(
        class_divergences.is_empty(),
        "the losslessness gate is red on {} of {} classes; every class is reported above so a \
         draft-max verdict can be class-dependent:\n{}",
        class_divergences.len(),
        classes.len(),
        class_divergences.join("\n")
    );
}
