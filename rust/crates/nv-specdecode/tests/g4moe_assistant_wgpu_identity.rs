#![cfg(feature = "wgpu")]

mod g4moe_assistant_common;
use g4moe_assistant_common as c;
use g4moe_assistant_common::official_template::OfficialTemplate;

const IDENTITY_DEPTH_256_ONE_RUNG_SUFFICES_BECAUSE_THE_ACCEPT_RULE_IS_DEPTH_BLIND: usize = 256;

const IDENTITY_EMITTED_TOKENS_160_CLEARS_THE_128_GATE_WITH_ROUND_GRANULARITY_SLACK: usize = 160;

const IDENTITY_MIN_COMPARED_128_IS_THE_GATE: usize = 128;

#[test]
#[ignore = "loads the 26B-A4B MoE trunk on wgpu plus the assistant drafter, primes one \
            chat-wrapped 256-token context twice, and asserts the speculative greedy stream \
            (assistant draft + verify_chain exact-u32 accept) is token-identical to the plain \
            decode_step greedy stream over >=128 emitted tokens -- speculative greedy with exact \
            verify is lossless, so any divergence is a verify/accept bug, no tolerance. \
            NV_G4WGPU_DRAFT_TEST=1 NV_G4DRAFT_ASSISTANT_DIR NV_G4DRAFT_CORPUS required; sets \
            NV_G4MOE_FLASH_DECODE=0 for BOTH streams because verify_chain refuses the flash \
            decode graph, so the identity read compares like against like"]
fn g4moe_wgpu_assistant_spec_stream_is_token_identical_to_plain_greedy_over_128_tokens() {
    c::gate_loud("NV_G4WGPU_DRAFT_TEST");
    c::force_the_nonflash_decode_graph_verify_chain_refuses_the_flash_default();
    let trunk_dir = c::trunk_snapshot_dir();
    let assistant_dir = c::assistant_ckpt_dir();
    let k = c::env_usize("NV_G4DRAFT_K", 4);
    let depth = IDENTITY_DEPTH_256_ONE_RUNG_SUFFICES_BECAUSE_THE_ACCEPT_RULE_IS_DEPTH_BLIND;
    let want = IDENTITY_EMITTED_TOKENS_160_CLEARS_THE_128_GATE_WITH_ROUND_GRANULARITY_SLACK;
    let tok = c::trunk_tokenizer(&trunk_dir);
    let tmpl = OfficialTemplate::load(&trunk_dir);
    let corpus_ids = c::corpus_ids(&tok);
    let prompt = c::chat_wrapped_prompt(&tmpl, &tok, &corpus_ids, 0, depth);
    let max_seq = prompt.len() + want * 2 + 64;
    let mut m = c::build_engine(&trunk_dir, max_seq);
    let mut host = c::load_host(&m, &trunk_dir, &assistant_dir);

    let plain_primed = c::prime_prompt(&mut m, &prompt).expect("plain prime");
    let mut plain: Vec<u32> = Vec::with_capacity(want);
    let mut cur = plain_primed.anchor;
    for _ in 0..want {
        cur = m.decode_step(cur).expect("plain decode step");
        plain.push(cur);
    }

    host.clear_mirrors();
    let spec_primed = c::prime_prompt(&mut m, &prompt).expect("spec prime");
    assert_eq!(
        spec_primed.anchor, plain_primed.anchor,
        "re-priming the same prompt produced a different anchor; the engine is not \
         deterministic and no identity read is possible"
    );
    let mut anchor = spec_primed.anchor;
    let mut hidden = spec_primed.hidden;
    let mut spec: Vec<u32> = Vec::with_capacity(want + k);
    let mut rounds = 0usize;
    let mut accepted_total = 0usize;
    while spec.len() < want {
        let out = c::spec_round(&mut m, &mut host, &mut anchor, &mut hidden, k)
            .unwrap_or_else(|e| panic!("spec round {rounds}: {e:#}"));
        accepted_total += out.accepted;
        spec.extend_from_slice(&out.emitted);
        rounds += 1;
    }

    let n = plain.len().min(spec.len());
    assert!(
        n >= IDENTITY_MIN_COMPARED_128_IS_THE_GATE,
        "only {n} tokens available for the identity read; the gate wants >= {}",
        IDENTITY_MIN_COMPARED_128_IS_THE_GATE
    );
    for i in 0..n {
        assert_eq!(
            plain[i], spec[i],
            "token stream diverged at emitted index {i}: plain={} spec={} \
             (plain_ctx={:?} spec_ctx={:?}); speculative greedy with exact verify is lossless, \
             so this is a verify/accept bug",
            plain[i],
            spec[i],
            &plain[i.saturating_sub(4)..(i + 4).min(plain.len())],
            &spec[i.saturating_sub(4)..(i + 4).min(spec.len())]
        );
    }
    eprintln!(
        "[G4WGPU-IDENTITY] depth={depth} compared={n} rounds={rounds} \
         accepted_drafts_total={accepted_total} mean_emitted_per_round={:.3} verdict=IDENTICAL \
         basis=(both streams on gemma4_moe_wgpu with NV_G4MOE_FLASH_DECODE=0, plain=decode_step \
         greedy, spec=assistant draft + verify_chain exact-u32 accept)",
        spec.len() as f64 / rounds as f64
    );
}
