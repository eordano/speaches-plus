#![cfg(feature = "cuda")]

mod hub_dirs;
mod sufspec_classes;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const CHECKPOINT_LABEL: &str = "unsloth/Qwen3.8-27B-NVFP4";

const PROMPT_TOKENS_1024_MATCHES_THE_HEADROOM_CLASSES: usize = 1024;
const GEN_TOKENS_512_ENOUGH_REJECTED_ROUNDS_PER_CLASS_FOR_A_RECOVERY_RATE: usize = 512;
const MAX_SEQ_4096_COVERS_PROMPT_GEN_AND_THE_VERIFY_GUARD: usize = 4096;
const TOP1_HOST_DEVICE_ARGMAX_MISMATCH_CEILING_5PCT_ABOVE_WHICH_THE_PROBE_READ_MISALIGNED_LOGITS:
    f64 = 0.05;

fn require_gate(gate: &str) {
    if std::env::var(gate).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {gate}=1");
    }
}

#[derive(Default)]
struct PositionRecovery {
    rejected: usize,
    verified_is_drafter_top2: usize,
    verified_in_drafter_top3: usize,
    top1_mismatch: usize,
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 plus the MTP head once; set \
            NV_QWEN38_SERVING_TEST=1 NV_DRAFTER=mtp NV_Q38_DRAFT_FAST=1 -- the PCTree headroom \
            probe: per workload class, at rounds whose first draft position was rejected, how \
            often the verify-decided token sits in the drafter's top-2/top-3 at that position; \
            that fraction is the mass a parent-conditioned draft tree could recover and a chain \
            forfeits; the test owns NV_Q38_MTP_ROUND_DEBUG, leave it and NV_Q38_SPEC_SUFFIX \
            unset in the caller environment"]
fn qwen38_pctree_headroom_drafter_top2_top3_recovery_at_rejected_positions_three_classes() {
    use candle_core::Device;
    use nv_models::graph_engine::GraphedQwen3Moe;
    use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
    use nv_specdecode::qwen38_mtp::{
        draft_fast_device_chained_tokens_selected_from_env, mtp_chain_depth_from_env,
        mtp_draft_dir_override_from_env, mtp_drafter_selected_from_env, Qwen38DenseMtpHead,
        Qwen38MtpGraphedDecodeSession, MTP_WEIGHTS_FILE_NAME, NV_Q38_MTP_ROUND_DEBUG_ENV,
    };
    use nv_specdecode::qwen38_sufspec::NV_Q38_SPEC_SUFFIX_ENV;

    require_gate("NV_QWEN38_SERVING_TEST");
    assert!(
        mtp_drafter_selected_from_env(),
        "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_DRAFTER=mtp"
    );
    assert!(
        draft_fast_device_chained_tokens_selected_from_env(),
        "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_Q38_DRAFT_FAST=1; the top3 \
         probe instruments only the fast device-chained draft path"
    );
    assert!(
        std::env::var(NV_Q38_SPEC_SUFFIX_ENV).is_err(),
        "leave {NV_Q38_SPEC_SUFFIX_ENV} unset: suffix rounds bypass the mtp chain and would \
         shrink the probe's sample without changing what it measures"
    );
    assert!(
        std::env::var(NV_Q38_MTP_ROUND_DEBUG_ENV).is_err(),
        "leave {NV_Q38_MTP_ROUND_DEBUG_ENV} unset: this probe owns the flag"
    );
    std::env::set_var(NV_Q38_MTP_ROUND_DEBUG_ENV, "1");

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

    let mut overall: Vec<PositionRecovery> = (0..k).map(|_| PositionRecovery::default()).collect();
    let mut overall_rounds = 0usize;
    let mut overall_full_accepts = 0usize;
    for (name, prompt) in &classes {
        engine.reset().expect("engine reset between classes");
        let (records, stream_fnv, gen_len) = {
            let mut session = Qwen38MtpGraphedDecodeSession::start(&mut engine, &mtp, k, prompt)
                .unwrap_or_else(|e| panic!("session start class={name}: {e:#}"));
            let mut gen: Vec<u32> = vec![session.anchor()];
            while gen.len() < 1 + GEN_TOKENS_512_ENOUGH_REJECTED_ROUNDS_PER_CLASS_FOR_A_RECOVERY_RATE
            {
                assert!(session.round_fits(), "kv headroom exhausted class={name}");
                let emitted = session.round().expect("round");
                gen.extend(&emitted);
            }
            let fnv = sufspec_classes::fnv1a64_over_u32_le(&gen);
            (
                std::mem::take(
                    &mut session
                        .tree_headroom_rounds_round_debug_fast_chain_only_drafter_top3_by_step_verify_greedy_rows_accepted,
                ),
                fnv,
                gen.len(),
            )
        };
        assert!(
            !records.is_empty(),
            "class={name}: the probe collected zero rounds, so the round-debug top3 arm never \
             engaged and this run measured nothing"
        );
        let mut class_pos: Vec<PositionRecovery> =
            (0..k).map(|_| PositionRecovery::default()).collect();
        let mut full_accepts = 0usize;
        for (top3_by_step, greedy_rows, accepted) in &records {
            assert_eq!(
                top3_by_step.len(),
                k,
                "class={name}: a fixed-k probe round drafted {} steps",
                top3_by_step.len()
            );
            assert_eq!(
                greedy_rows.len(),
                k + 1,
                "class={name}: a fixed-k probe round verified {} rows",
                greedy_rows.len()
            );
            if *accepted >= k {
                full_accepts += 1;
                continue;
            }
            let slot = &mut class_pos[*accepted];
            slot.rejected += 1;
            let verified = greedy_rows[*accepted];
            let t3 = &top3_by_step[*accepted];
            if verified == t3[0] {
                slot.top1_mismatch += 1;
            }
            if verified == t3[1] {
                slot.verified_is_drafter_top2 += 1;
            }
            if verified == t3[1] || verified == t3[2] {
                slot.verified_in_drafter_top3 += 1;
            }
        }
        for (p, s) in class_pos.iter().enumerate() {
            eprintln!(
                "PCTREE-HEADROOM class={name} reject_pos={} rejected_rounds={} \
                 verified_is_drafter_top2={} verified_in_drafter_top3={} top2_pct={:.1} \
                 top3_pct={:.1} top1_mismatch={}",
                p + 1,
                s.rejected,
                s.verified_is_drafter_top2,
                s.verified_in_drafter_top3,
                100.0 * s.verified_is_drafter_top2 as f64 / s.rejected.max(1) as f64,
                100.0 * s.verified_in_drafter_top3 as f64 / s.rejected.max(1) as f64,
                s.top1_mismatch,
            );
            overall[p].rejected += s.rejected;
            overall[p].verified_is_drafter_top2 += s.verified_is_drafter_top2;
            overall[p].verified_in_drafter_top3 += s.verified_in_drafter_top3;
            overall[p].top1_mismatch += s.top1_mismatch;
        }
        eprintln!(
            "PCTREE-HEADROOM class={name} rounds={} full_accept_rounds={} gen_tokens={gen_len} \
             fnv1a64={stream_fnv:016x} k={k} checkpoint={CHECKPOINT_LABEL}",
            records.len(),
            full_accepts,
        );
        overall_rounds += records.len();
        overall_full_accepts += full_accepts;
    }
    let rej1 = &overall[0];
    eprintln!(
        "PCTREE-HEADROOM overall rounds={overall_rounds} full_accept_rounds={overall_full_accepts} \
         pos1_rejected={} pos1_top2_pct={:.1} pos1_top3_pct={:.1}",
        rej1.rejected,
        100.0 * rej1.verified_is_drafter_top2 as f64 / rej1.rejected.max(1) as f64,
        100.0 * rej1.verified_in_drafter_top3 as f64 / rej1.rejected.max(1) as f64,
    );
    std::env::remove_var(NV_Q38_MTP_ROUND_DEBUG_ENV);
    let mismatch: usize = overall.iter().map(|s| s.top1_mismatch).sum();
    let rejected: usize = overall.iter().map(|s| s.rejected).sum();
    assert!(
        (mismatch as f64)
            <= TOP1_HOST_DEVICE_ARGMAX_MISMATCH_CEILING_5PCT_ABOVE_WHICH_THE_PROBE_READ_MISALIGNED_LOGITS
                * rejected.max(1) as f64,
        "host top-1 disagreed with the device draft argmax on {mismatch}/{rejected} rejected \
         rounds; the probe's top3 readback is not looking at the drafter's decision logits"
    );
}
