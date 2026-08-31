#![cfg(feature = "wgpu")]

mod common;
use common::q3d_lora_synth::{synth_adapter, write_adapter, Fill, Scratch};
use common::tiny_config_q3d_mixed_layers as tiny_config;
use common::tiny_weights_q3d as tiny_weights;

use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::{MtpHostWeights, Q3dLora, Qwen3_5DenseWgpu};
use nv_models::qwen3_5_moe::Qwen3_5DenseConfig;

static ENV_LOCK_THESE_TESTS_MUTATE_PROCESS_GLOBAL_ENV_SO_MUST_NOT_INTERLEAVE:
    std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK_THESE_TESTS_MUTATE_PROCESS_GLOBAL_ENV_SO_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

const DENSE_M_ENV: &str = "NV_WGPU_PREFILL_M";

const TINY_MAX_SEQ: usize = 64;

fn require_gpu() {
    assert!(
        common::have_gpu(),
        "no wgpu adapter: a skipped adapter-parity proof reads as a passed one, so this suite \
         fails instead of skipping"
    );
}

const VF_TINY_ARM_PINS_ONE_KERNEL_PER_STAGE_SO_THE_M16_AND_M8_GRAPHS_RECORD_IDENTICAL_ROUTES:
    [(&str, &str); 3] = [
    (q3d::PF_COOP_ENV_DEFAULT_ON_SET_0_FOR_THE_LEGACY_M16_GEMM_ARM, "0"),
    (q3d::PF_ATTN_TILED_ENV_DEFAULT_ON_SET_0_FOR_THE_SCORES_SLAB_ARM, "1"),
    (q3d::PF_SCAN_WY_ENV_DEFAULT_ON_SET_0_FOR_THE_TOKEN_SERIAL_SCAN, "1"),
];

const PADDED_TESTS_PIN_TILED_FLASH_BECAUSE_THE_SCORES_SLAB_ARM_SUMS_SCORES_IN_A_DIFFERENT_ORDER_THAN_M1_DECODE:
    &str =
    "the scores-slab pf attention arm reduces scores in a different order than the m=1 decode \
     attention, so off-grid activations land one bf16 ulp apart and the residual chain \
     amplifies that ulp into every logit (measured 1.5e-3 flat across the row); the base \
     fixtures sit on the exact grid and hide it, adapter deltas leave the grid and expose it. \
     A missing lora site measures orders of magnitude larger and fails per-module (the KV-row \
     probe pins that separately), so the bitwise adapter-vs-m1 gates pin the tiled flash arm, \
     where the m-row and m=1 reductions agree bit-for-bit";

fn padded_arm_env_set() {
    std::env::set_var(DENSE_M_ENV, "8");
    std::env::remove_var(q3d::VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM);
    std::env::set_var(
        q3d::PF_ATTN_TILED_ENV_DEFAULT_ON_SET_0_FOR_THE_SCORES_SLAB_ARM,
        "1",
    );
}

fn dedicated_arm_env_set() {
    std::env::set_var(DENSE_M_ENV, "16");
    std::env::set_var(
        q3d::VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM,
        "8",
    );
    for (k, v) in VF_TINY_ARM_PINS_ONE_KERNEL_PER_STAGE_SO_THE_M16_AND_M8_GRAPHS_RECORD_IDENTICAL_ROUTES {
        std::env::set_var(k, v);
    }
}

fn arm_env_clear() {
    std::env::remove_var(DENSE_M_ENV);
    std::env::remove_var(q3d::VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM);
    for (k, _) in VF_TINY_ARM_PINS_ONE_KERNEL_PER_STAGE_SO_THE_M16_AND_M8_GRAPHS_RECORD_IDENTICAL_ROUTES {
        std::env::remove_var(k);
    }
}

fn ids_from(vocab: usize, n: usize, salt: u32) -> Vec<u32> {
    (0..n)
        .map(|i| ((i as u32 * 7 + salt * 13 + 1) % (vocab as u32 - 1)) + 1)
        .collect()
}

fn engine(
    cfg: &Qwen3_5DenseConfig,
    hw: &q3d::HostDenseWeights,
    lora: Option<&Q3dLora>,
) -> Qwen3_5DenseWgpu {
    Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), hw, TINY_MAX_SEQ, lora).expect("build engine")
}

fn coarse_lora(cfg: &Qwen3_5DenseConfig, tag: &str, seed: u64) -> (Scratch, Q3dLora) {
    let synth = synth_adapter(cfg, seed, Fill::Coarse, None);
    let dir = write_adapter(tag, &synth);
    let lora = Q3dLora::from_peft_dir(&dir.0, cfg).expect("build q3d lora sites");
    (dir, lora)
}

use common::mtp_tiny_weights_q3d as mtp_tiny_weights;

fn prefill_like_serving(m: &mut Qwen3_5DenseWgpu, prompt: &[u32]) -> u32 {
    let (last, rest) = prompt.split_last().expect("non-empty prompt");
    let done = m.prefill_tokens(rest).expect("chunked prefill");
    for &t in &rest[done..] {
        m.prefill_step(t).expect("prefill step");
    }
    m.decode_step(*last).expect("anchor decode step")
}

fn bit_diff(want: &[f32], got: &[f32]) -> usize {
    want.iter()
        .zip(got)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

fn m1_reference_stream(
    cfg: &Qwen3_5DenseConfig,
    hw: &q3d::HostDenseWeights,
    lora: Option<&Q3dLora>,
    prompt: &[u32],
    n: usize,
) -> Vec<u32> {
    let mut m = engine(cfg, hw, lora);
    let mut last = prefill_like_serving(&mut m, prompt);
    let mut out = vec![last];
    while out.len() < n {
        last = m.decode_step(last).expect("reference decode step");
        out.push(last);
    }
    out
}

enum DraftPolicy {
    MtpOwn,
    Junk,
    Clairvoyant,
}

const ADAPTED_DRAFT_POLICY_INVARIANCE_IS_THE_BAR: &str =
    "every emitted token is gated by the adapted trunk's verify argmax, and every commit path \
     (full accept, partial accept with snapshot+replay or scan-replay, drafter KV \
     rewind+catch-up) must land in the state the adapted engine's pure M=1 stepping produces; \
     the served stream may not depend on WHAT was drafted, so junk drafts, the mtp head's own \
     drafts, and clairvoyant drafts must all emit the byte-identical adapted stream -- a \
     divergence means a verify or commit pass ran without its lora site";

#[allow(clippy::too_many_arguments)]
fn mtp_stream(
    cfg: &Qwen3_5DenseConfig,
    hw: &q3d::HostDenseWeights,
    lora: Option<&Q3dLora>,
    mtp: &MtpHostWeights,
    prompt: &[u32],
    n: usize,
    k: usize,
    policy: DraftPolicy,
    reference: &[u32],
) -> (Vec<u32>, usize) {
    let mut m = engine(cfg, hw, lora);
    m.mtp_attach_host(mtp).expect("attach mtp head");
    let mut last = prefill_like_serving(&mut m, prompt);
    let mut out = vec![last];
    let mut accepted_total = 0usize;
    while out.len() < n {
        let rows = m.verify_max_rows();
        assert!(rows >= 2, "{DENSE_M_ENV} did not reach the M-row prefill graph");
        if m.current_pos() + rows.max(m.prefill_chunk_len()) > TINY_MAX_SEQ {
            last = m.decode_step(last).expect("tail decode step");
            out.push(last);
            continue;
        }
        let want = k.min(rows - 1);
        let own = m.mtp_draft_round(last, want).expect("mtp draft round");
        let drafts: Vec<u32> = match policy {
            DraftPolicy::MtpOwn => own,
            DraftPolicy::Junk => (0..want as u32)
                .map(|j| (j * 5 + 2) % cfg.vocab_size as u32)
                .collect(),
            DraftPolicy::Clairvoyant => {
                let mut d = Vec::with_capacity(want);
                for j in 0..want {
                    d.push(*reference.get(out.len() + j).unwrap_or(&own[j.min(own.len() - 1)]));
                }
                d
            }
        };
        let mut batch = vec![last];
        batch.extend_from_slice(&drafts);
        let amax = m.verify_chain(&batch).expect("verify chain");
        let mut acc = 0usize;
        while acc < drafts.len() && amax[acc] == drafts[acc] {
            acc += 1;
        }
        let bonus = amax[acc];
        m.advance(acc + 1).expect("advance the accepted prefix");
        m.mtp_post_verify(&batch[1..=acc]).expect("mtp post verify");
        accepted_total += acc;
        out.extend_from_slice(&drafts[..acc]);
        out.push(bonus);
        last = bonus;
    }
    out.truncate(n);
    (out, accepted_total)
}

#[test]
fn no_adapter_engine_records_the_dedicated_verify_graph_bit_identical_to_plain() {
    let _env = env_lock();
    require_gpu();
    dedicated_arm_env_set();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9a3d_1001);
    let prompt = ids_from(cfg.vocab_size, 9, 3);
    let chain = ids_from(cfg.vocab_size, 5, 17);

    let mut plain = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("plain build");
    assert!(
        plain.verify_chain_rides_dedicated_graph(),
        "NV_Q3D_VERIFY_M=8 under NV_WGPU_PREFILL_M=16 must record the dedicated verify graph"
    );
    for &t in &prompt {
        plain.prefill_step(t).expect("plain prefill step");
    }
    let (want_tok, want_log) = plain.verify_chain_logits(&chain).expect("plain chain");
    drop(plain);

    let mut with_none = engine(&cfg, &hw, None);
    assert!(
        with_none.verify_chain_rides_dedicated_graph(),
        "a lora-capable build without an adapter must record the same dedicated verify graph"
    );
    for &t in &prompt {
        with_none.prefill_step(t).expect("lora-none prefill step");
    }
    let (got_tok, got_log) = with_none.verify_chain_logits(&chain).expect("lora-none chain");
    assert_eq!(want_tok, got_tok, "no-adapter dedicated verify argmax drifted");
    assert_eq!(
        bit_diff(&want_log, &got_log),
        0,
        "no-adapter dedicated verify logits drifted from the plain build; the lora-capable \
         build path must leave the base graph bit-identical"
    );
    arm_env_clear();
}

#[test]
fn adapter_on_batched_verify_rows_bit_match_the_adapters_own_m1_stepping() {
    let _env = env_lock();
    require_gpu();
    padded_arm_env_set();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9a3d_1002);
    let (_dir, lora) = coarse_lora(&cfg, "vfm1", 0xd317a2);
    let prompt = ids_from(cfg.vocab_size, 9, 3);

    for chain_len in [1usize, 3, 8] {
        let chain = ids_from(cfg.vocab_size, chain_len, 11 + chain_len as u32);

        let mut m1 = engine(&cfg, &hw, Some(&lora));
        for &t in &prompt {
            m1.prefill_step(t).expect("m1 prefill step");
        }
        let want: Vec<(u32, Vec<f32>)> = chain
            .iter()
            .map(|&t| m1.decode_step_logits(t).expect("m1 decode step"))
            .collect();

        let mut vf = engine(&cfg, &hw, Some(&lora));
        assert!(
            !vf.verify_chain_rides_dedicated_graph(),
            "this arm pins the padded prefill graph, the m1 twin the dedicated graph is \
             separately A/B'd against"
        );
        for &t in &prompt {
            vf.prefill_step(t).expect("verify prefill step");
        }
        let (toks, logits) = vf.verify_chain_logits(&chain).expect("verify_chain");
        for (r, (want_tok, want_row)) in want.iter().enumerate() {
            let got_row = &logits[r * cfg.vocab_size..(r + 1) * cfg.vocab_size];
            assert_eq!(
                bit_diff(want_row, got_row),
                0,
                "chain_len={chain_len} row {r}: adapter-on batched verify logits differ from \
                 the adapter-on M=1 decode step at the same position; a whole-row flat 1.5e-3 \
                 delta is the slab-arm mechanism, anything larger is a verify-path projection \
                 running without its lora site. \
                 {PADDED_TESTS_PIN_TILED_FLASH_BECAUSE_THE_SCORES_SLAB_ARM_SUMS_SCORES_IN_A_DIFFERENT_ORDER_THAN_M1_DECODE}"
            );
            assert_eq!(toks[r], *want_tok, "chain_len={chain_len} row {r} argmax");
        }
        vf.advance(chain_len).expect("advance the whole chain");
        assert_eq!(vf.current_pos(), m1.current_pos(), "chain_len={chain_len} pos");
    }
    arm_env_clear();
}

#[test]
fn adapter_on_partial_accepts_leave_the_stream_bit_identical_to_adapter_m1_stepping() {
    let _env = env_lock();
    require_gpu();
    padded_arm_env_set();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9a3d_1003);
    let (_dir, lora) = coarse_lora(&cfg, "vfpartial", 0xd317a3);
    let prompt = ids_from(cfg.vocab_size, 9, 5);
    let chain = ids_from(cfg.vocab_size, 6, 21);
    let tail = ids_from(cfg.vocab_size, 5, 31);

    for accepted in [0usize, 1, 3, 6] {
        let mut m1 = engine(&cfg, &hw, Some(&lora));
        for &t in &prompt {
            m1.prefill_step(t).expect("m1 prefill step");
        }
        for &t in &chain[..accepted] {
            m1.decode_step(t).expect("m1 accepted step");
        }
        let want: Vec<(u32, Vec<f32>)> = tail
            .iter()
            .map(|&t| m1.decode_step_logits(t).expect("m1 tail step"))
            .collect();

        let mut vf = engine(&cfg, &hw, Some(&lora));
        for &t in &prompt {
            vf.prefill_step(t).expect("verify prefill step");
        }
        vf.verify_chain(&chain).expect("verify_chain");
        vf.advance(accepted).expect("advance the accepted prefix");
        for (i, (want_tok, want_row)) in want.iter().enumerate() {
            let (got_tok, got_row) = vf.decode_step_logits(tail[i]).expect("tail step");
            assert_eq!(
                bit_diff(want_row, &got_row),
                0,
                "accepted={accepted} tail step {i}: the adapter stream drifted after a partial \
                 accept, so the rollback snapshot or the M=1 replay lost the lora deltas"
            );
            assert_eq!(got_tok, *want_tok, "accepted={accepted} tail step {i} argmax");
        }
    }
    arm_env_clear();
}

#[test]
fn adapter_on_dedicated_verify_graph_bit_matches_the_adapter_padded_graph_and_commits_partials() {
    let _env = env_lock();
    require_gpu();
    dedicated_arm_env_set();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9a3d_1004);
    let (_dir, lora) = coarse_lora(&cfg, "vfdedicated", 0xd317a4);
    let mut m = engine(&cfg, &hw, Some(&lora));
    assert!(
        m.verify_chain_rides_dedicated_graph(),
        "an adapter engine must record the dedicated verify graph: its passes carry the same \
         lora sites the prefill graph carries"
    );
    assert_eq!(m.verify_max_rows(), 8, "chain capacity is the dedicated graph's rows");
    let prompt = ids_from(cfg.vocab_size, 19, 7);
    let mut last = prefill_like_serving(&mut m, &prompt);

    let mut m1 = engine(&cfg, &hw, Some(&lora));
    let m1_last = prefill_like_serving(&mut m1, &prompt);
    assert_eq!(last, m1_last, "anchor token");

    for round in 0..3u32 {
        let mut batch = vec![last];
        batch.extend(ids_from(cfg.vocab_size, 3, 11 + round));
        let (a_tok, a_log) = m.verify_chain_logits(&batch).expect("dedicated-graph chain");
        m.advance(0).expect("rollback the dedicated-graph forward");
        let (b_tok, b_log) = m
            .debug_verify_chain_logits_via_the_padded_prefill_graph_for_test(&batch)
            .expect("padded-graph chain");
        assert_eq!(
            a_tok, b_tok,
            "round {round}: adapter verify argmax diverged between the dedicated and padded \
             graphs on arms pinned kernel-identical"
        );
        assert_eq!(
            bit_diff(&a_log, &b_log),
            0,
            "round {round}: adapter verify logits diverged between the dedicated 8-row graph \
             and the padded 16-row graph on arms pinned kernel-identical, so one of them \
             dropped a lora site"
        );
        m.advance(0).expect("rollback the padded-graph forward");

        let accept = 1 + (round as usize % 2);
        let (c_tok, c_log) = m.verify_chain_logits(&batch).expect("commit chain");
        m.advance(accept).expect("scan-replay commit of a partial prefix");
        assert_eq!(
            bit_diff(&a_log, &c_log),
            0,
            "round {round}: the re-run dedicated chain drifted before commit"
        );
        for &t in &batch[..accept] {
            m1.decode_step(t).expect("m1 twin step");
        }
        assert_eq!(m.current_pos(), m1.current_pos(), "round {round}: post-commit pos");
        let next_in = c_tok[accept - 1];
        let (vf_next, vf_row) = m.decode_step_logits(next_in).expect("post-commit decode");
        let (m1_next, m1_row) = m1.decode_step_logits(next_in).expect("m1 post-commit decode");
        assert_eq!(
            bit_diff(&vf_row, &m1_row),
            0,
            "round {round}: after a scan-replay partial commit of {accept} rows the adapter \
             stream drifted from pure adapter M=1 stepping, so the DeltaNet scan replay read \
             un-adapted projections"
        );
        assert_eq!(vf_next, m1_next, "round {round}: post-commit argmax");
        last = vf_next;
    }
    arm_env_clear();
}

#[test]
fn mtp_rounds_under_an_adapter_emit_the_adapter_m1_stream_under_every_draft_policy() {
    let _env = env_lock();
    require_gpu();
    padded_arm_env_set();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9a3d_1005);
    let mtp = mtp_tiny_weights(&cfg, 0xabcd_ef01_2345);
    let (_dir, lora) = coarse_lora(&cfg, "mtppadded", 0xd317a5);
    let prompt = ids_from(cfg.vocab_size, 19, 3);
    let n = 24;
    let reference = m1_reference_stream(&cfg, &hw, Some(&lora), &prompt, n);
    {
        let mut with = engine(&cfg, &hw, Some(&lora));
        let mut base = engine(&cfg, &hw, None);
        let (_, l_with) = with.decode_step_logits(prompt[0]).expect("adapter probe step");
        let (_, l_base) = base.decode_step_logits(prompt[0]).expect("base probe step");
        let shift = l_with
            .iter()
            .zip(&l_base)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            shift > 1e-3,
            "the coarse adapter left decode logits within {shift} of base, so every stream \
             identity below would hold with dead sites (the tiny fixture's argmax stream \
             saturates, which is why this probe reads logits, not tokens)"
        );
    }

    let (own, _) = mtp_stream(
        &cfg,
        &hw,
        Some(&lora),
        &mtp,
        &prompt,
        n,
        3,
        DraftPolicy::MtpOwn,
        &reference,
    );
    assert_eq!(own, reference, "{ADAPTED_DRAFT_POLICY_INVARIANCE_IS_THE_BAR}");

    let (junk, _) = mtp_stream(
        &cfg,
        &hw,
        Some(&lora),
        &mtp,
        &prompt,
        n,
        3,
        DraftPolicy::Junk,
        &reference,
    );
    assert_eq!(junk, reference, "{ADAPTED_DRAFT_POLICY_INVARIANCE_IS_THE_BAR}");

    let (clair, clair_accepted) = mtp_stream(
        &cfg,
        &hw,
        Some(&lora),
        &mtp,
        &prompt,
        n,
        3,
        DraftPolicy::Clairvoyant,
        &reference,
    );
    assert_eq!(clair, reference, "{ADAPTED_DRAFT_POLICY_INVARIANCE_IS_THE_BAR}");
    assert!(
        clair_accepted > 0,
        "clairvoyant drafts from the adapted reference stream never accepted, so the \
         full-accept commit and drafter catch-up paths never ran under the adapter and the \
         identities above are vacuous"
    );
    arm_env_clear();
}

#[test]
fn mtp_rounds_under_an_adapter_on_the_dedicated_verify_graph_emit_the_adapter_m1_stream() {
    let _env = env_lock();
    require_gpu();
    dedicated_arm_env_set();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9a3d_1006);
    let mtp = mtp_tiny_weights(&cfg, 0xabcd_ef01_2345);
    let (_dir, lora) = coarse_lora(&cfg, "mtpdedicated", 0xd317a6);
    let prompt = ids_from(cfg.vocab_size, 19, 3);
    let n = 24;
    let reference = m1_reference_stream(&cfg, &hw, Some(&lora), &prompt, n);
    let (own, _) = mtp_stream(
        &cfg,
        &hw,
        Some(&lora),
        &mtp,
        &prompt,
        n,
        3,
        DraftPolicy::MtpOwn,
        &reference,
    );
    assert_eq!(own, reference, "{ADAPTED_DRAFT_POLICY_INVARIANCE_IS_THE_BAR}");
    let (clair, clair_accepted) = mtp_stream(
        &cfg,
        &hw,
        Some(&lora),
        &mtp,
        &prompt,
        n,
        3,
        DraftPolicy::Clairvoyant,
        &reference,
    );
    assert_eq!(clair, reference, "{ADAPTED_DRAFT_POLICY_INVARIANCE_IS_THE_BAR}");
    assert!(
        clair_accepted > 0,
        "clairvoyant drafts never accepted on the dedicated graph, so its scan-replay commit \
         and drafter catch-up never ran under the adapter and the identities are vacuous"
    );
    arm_env_clear();
}

#[test]
fn verify_written_kv_rows_change_exactly_where_the_adapter_targets_say() {
    let _env = env_lock();
    require_gpu();
    padded_arm_env_set();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9a3d_1007);
    let prompt = ids_from(cfg.vocab_size, 9, 5);
    let chain = ids_from(cfg.vocab_size, 4, 23);
    let kv_row = cfg.num_key_value_heads * cfg.head_dim;

    let run = |lora: Option<&Q3dLora>| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut m = engine(&cfg, &hw, lora);
        for &t in &prompt {
            m.prefill_step(t).expect("prefill step");
        }
        let base_pos = m.current_pos();
        let (_, logits) = m.verify_chain_logits(&chain).expect("verify chain");
        let (k, v) = m
            .debug_read_attn_kv_bf16_rows_for_test(0, base_pos, chain.len())
            .expect("read verify-written kv rows");
        (k, v, logits)
    };

    let (k_base, v_base, log_base) = run(None);

    let kv_synth = synth_adapter(&cfg, 0xd317a7, Fill::Coarse, None)
        .retain_modules(|n| n.ends_with("self_attn.k_proj") || n.ends_with("self_attn.v_proj"));
    let kv_dir = write_adapter("kvonly", &kv_synth);
    let kv_lora = Q3dLora::from_peft_dir(&kv_dir.0, &cfg).expect("kv-only lora");
    let (k_kv, v_kv, _) = run(Some(&kv_lora));
    for r in 0..chain.len() {
        let kd = bit_diff(&k_base[r * kv_row..(r + 1) * kv_row], &k_kv[r * kv_row..(r + 1) * kv_row]);
        let vd = bit_diff(&v_base[r * kv_row..(r + 1) * kv_row], &v_kv[r * kv_row..(r + 1) * kv_row]);
        assert!(
            kd > 0,
            "row {r}: a k_proj-targeting adapter left the verify-written K row bit-identical \
             to base, so the drafted KV rows still embed base projections"
        );
        assert!(
            vd > 0,
            "row {r}: a v_proj-targeting adapter left the verify-written V row bit-identical \
             to base, so the drafted KV rows still embed base projections"
        );
    }

    let q_synth = synth_adapter(&cfg, 0xd317a8, Fill::Coarse, None)
        .retain_modules(|n| n.ends_with("self_attn.q_proj"));
    let q_dir = write_adapter("qonly", &q_synth);
    let q_lora = Q3dLora::from_peft_dir(&q_dir.0, &cfg).expect("q-only lora");
    let (k_q, v_q, log_q) = run(Some(&q_lora));
    assert_eq!(
        bit_diff(&k_base, &k_q) + bit_diff(&v_base, &v_q),
        0,
        "a q_proj-only adapter must leave every verify-written KV row bit-identical to base: \
         its site carries zero k/v slices and the packed read-modify-write adds exactly zero"
    );
    assert!(
        bit_diff(&log_base, &log_q) > 0,
        "the q_proj-only adapter never reached the verify logits, so the zero-KV-delta check \
         above is vacuous"
    );
    arm_env_clear();
}

#[test]
fn a_zero_delta_adapter_leaves_verify_and_mtp_paths_bit_identical_to_base() {
    let _env = env_lock();
    require_gpu();
    padded_arm_env_set();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9a3d_1008);
    let mtp = mtp_tiny_weights(&cfg, 0x1234_5678_9abc);
    let prompt = ids_from(cfg.vocab_size, 9, 5);
    let chain = ids_from(cfg.vocab_size, 5, 29);

    let zero = synth_adapter(&cfg, 0x2e41, Fill::ZeroB, None);
    let zdir = write_adapter("vfzero", &zero);
    let zlora = Q3dLora::from_peft_dir(&zdir.0, &cfg).expect("zero lora");

    let mut base = engine(&cfg, &hw, None);
    for &t in &prompt {
        base.prefill_step(t).expect("base prefill step");
    }
    let (base_tok, base_log) = base.verify_chain_logits(&chain).expect("base chain");
    drop(base);

    let mut z = engine(&cfg, &hw, Some(&zlora));
    for &t in &prompt {
        z.prefill_step(t).expect("zero prefill step");
    }
    let (z_tok, z_log) = z.verify_chain_logits(&chain).expect("zero chain");
    assert_eq!(base_tok, z_tok, "zero-delta adapter changed the verify argmax rows");
    assert_eq!(
        bit_diff(&base_log, &z_log),
        0,
        "a zero-delta adapter must leave batched verify logits bit-identical: its sites emit, \
         accumulate exactly zero, and the packed read-modify-write is the identity on a zero \
         delta"
    );
    drop(z);

    let n = 16;
    let base_stream = {
        let empty_ref: Vec<u32> = Vec::new();
        mtp_stream(
            &cfg, &hw, None, &mtp, &prompt, n, 3, DraftPolicy::MtpOwn, &empty_ref,
        )
        .0
    };
    let zero_stream = {
        let empty_ref: Vec<u32> = Vec::new();
        mtp_stream(
            &cfg,
            &hw,
            Some(&zlora),
            &mtp,
            &prompt,
            n,
            3,
            DraftPolicy::MtpOwn,
            &empty_ref,
        )
        .0
    };
    assert_eq!(
        zero_stream, base_stream,
        "a zero-delta adapter changed the mtp-drafted served stream"
    );
    arm_env_clear();
}
