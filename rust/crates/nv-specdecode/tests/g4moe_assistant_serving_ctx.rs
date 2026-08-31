#![cfg(feature = "wgpu")]

mod g4moe_assistant_common;
use g4moe_assistant_common as c;
use g4moe_assistant_common::official_template::OfficialTemplate;

const DEFAULT_DEPTHS_256_16384_196608_ARE_THE_BOARD_RUNGS: [usize; 3] = [256, 16384, 196_608];

const TIMED_EMITTED_TOKENS_256_DEFAULT_BOUNDS_ROUND_GRANULARITY_NOISE: usize = 256;

const WARMUP_STEPS_16_ABSORB_PIPELINE_AND_MIRROR_FIRST_TOUCH: usize = 16;

const COST_PROBE_CALLS_10_MEDIAN_EACH_FOR_THE_VERIFY_COST_STATEMENT: usize = 10;

fn depths_from_env() -> Vec<usize> {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) if !v.trim().is_empty() => v
            .split(',')
            .map(|s| {
                let s = s.trim();
                let (num, mult) = match s.strip_suffix('k') {
                    Some(n) => (n, 1024usize),
                    None => (s, 1usize),
                };
                num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
            })
            .collect(),
        _ => DEFAULT_DEPTHS_256_16384_196608_ARE_THE_BOARD_RUNGS.to_vec(),
    }
}

fn assistant_arm_on() -> bool {
    std::env::var("NV_G4MOE_ASSISTANT_ARM").ok().as_deref() == Some("1")
}

#[test]
#[ignore = "loads the 26B-A4B MoE trunk on wgpu, primes a real chat-wrapped corpus context per \
            NV_CTX_TOKENS depth through the engine's chunked prefill, and measures effective \
            emitted tok/s: NV_G4MOE_ASSISTANT_ARM unset or 0 times the plain decode_step loop on \
            the process env's decode graph (the null pair), NV_G4MOE_ASSISTANT_ARM=1 times the \
            assistant spec-round loop (draft on cuda + verify_chain) and also states the k-row \
            verify_chain, decode_step and draft medians for the verify-cost read; the assistant \
            arm additionally requires NV_G4MOE_FLASH_DECODE=0 in the arm env because \
            verify_chain refuses the flash decode graph. NV_G4MOE_ASSISTANT_SERVING_TEST=1 \
            NV_G4DRAFT_CORPUS required; NV_G4DRAFT_ASSISTANT_DIR required for the assistant arm"]
fn g4moe_wgpu_effective_decode_tok_s_vs_context_depth_real_chunked_prefill_assistant_vs_plain() {
    c::gate_loud("NV_G4MOE_ASSISTANT_SERVING_TEST");
    let assistant = assistant_arm_on();
    if assistant {
        assert!(
            !nv_models::gemma4_moe_wgpu::flash_decode_enabled(),
            "NV_G4MOE_ASSISTANT_ARM=1 needs NV_G4MOE_FLASH_DECODE=0 in the arm env: \
             verify_chain refuses the flash decode graph, and setting it here silently would \
             hide a real serving requirement from the ladder spec"
        );
    }
    let trunk_dir = c::trunk_snapshot_dir();
    let k = c::env_usize("NV_G4DRAFT_K", 4);
    let want = c::env_usize(
        "NV_TIMED_EMITTED_TOKENS",
        TIMED_EMITTED_TOKENS_256_DEFAULT_BOUNDS_ROUND_GRANULARITY_NOISE,
    );
    let depths = depths_from_env();
    let deepest = *depths.iter().max().expect("non-empty depths");
    let tok = c::trunk_tokenizer(&trunk_dir);
    let tmpl = OfficialTemplate::load(&trunk_dir);
    let corpus_ids = c::corpus_ids(&tok);
    let need = deepest + c::RETOKENIZATION_DRIFT_PAD_128_KEEPS_THE_CHAT_WRAP_AT_OR_ABOVE_THE_RUNG;
    assert!(
        corpus_ids.len() >= need,
        "corpus holds {} tokens but the deepest rung needs {need}",
        corpus_ids.len()
    );
    let max_seq = deepest
        + c::RETOKENIZATION_DRIFT_PAD_128_KEEPS_THE_CHAT_WRAP_AT_OR_ABOVE_THE_RUNG
        + 512
        + WARMUP_STEPS_16_ABSORB_PIPELINE_AND_MIRROR_FIRST_TOUCH * (k + 1)
        + want * 2
        + COST_PROBE_CALLS_10_MEDIAN_EACH_FOR_THE_VERIFY_COST_STATEMENT * (k + 2)
        + 64;
    let mut m = c::build_engine(&trunk_dir, max_seq);
    let mut host = if assistant {
        Some(c::load_host(&m, &trunk_dir, &c::assistant_ckpt_dir()))
    } else {
        None
    };

    for &depth in &depths {
        let prompt = c::chat_wrapped_prompt(&tmpl, &tok, &corpus_ids, 0, depth);
        if let Some(h) = host.as_mut() {
            h.clear_mirrors();
        }
        let primed = c::prime_prompt(&mut m, &prompt).expect("prime");
        let mut anchor = primed.anchor;
        let mut hidden = primed.hidden;

        if let Some(h) = host.as_mut() {
            for r in 0..2 {
                c::spec_round(&mut m, h, &mut anchor, &mut hidden, k)
                    .unwrap_or_else(|e| panic!("warmup spec round {r} at depth {depth}: {e:#}"));
            }
            let mut rounds = 0usize;
            let mut accepted = 0usize;
            let mut submitted = 0usize;
            let mut emitted = 0usize;
            let t0 = std::time::Instant::now();
            while emitted < want {
                let out = c::spec_round(&mut m, h, &mut anchor, &mut hidden, k)
                    .unwrap_or_else(|e| panic!("spec round {rounds} at depth {depth}: {e:#}"));
                rounds += 1;
                accepted += out.accepted;
                submitted += out.submitted;
                emitted += out.emitted.len();
            }
            let wall = t0.elapsed().as_secs_f64();
            let mut decode_ms: Vec<f64> = Vec::new();
            let mut cur = anchor;
            for _ in 0..COST_PROBE_CALLS_10_MEDIAN_EACH_FOR_THE_VERIFY_COST_STATEMENT {
                let t = std::time::Instant::now();
                cur = m.decode_step(cur).expect("cost decode step");
                decode_ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let chunk: Vec<u32> = std::iter::once(cur)
                .chain((0..k as u32).map(|i| 5 + i))
                .collect();
            let mut verify_ms: Vec<f64> = Vec::new();
            for _ in 0..COST_PROBE_CALLS_10_MEDIAN_EACH_FOR_THE_VERIFY_COST_STATEMENT {
                let t = std::time::Instant::now();
                let _ = m.verify_chain(&chunk).expect("cost verify chunk");
                verify_ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let mut draft_ms: Vec<f64> = Vec::new();
            for _ in 0..COST_PROBE_CALLS_10_MEDIAN_EACH_FOR_THE_VERIFY_COST_STATEMENT {
                let t = std::time::Instant::now();
                let _ = h
                    .propose_from_engine(&c::EngineTap(&m), cur, &hidden, k)
                    .expect("cost draft");
                draft_ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let decode_med = c::median_ms(&mut decode_ms);
            let verify_med = c::median_ms(&mut verify_ms);
            let draft_med = c::median_ms(&mut draft_ms);
            let median_ms_tok = wall * 1e3 / emitted as f64;
            eprintln!(
                "CTX-SCALING g4moe-assist-wgpu depth={depth} median_ms_tok={median_ms_tok:.3} \
                 tok_s={:.1} prime_s={:.1} prime_tok_s={:.1} prime=chunked steps={emitted} \
                 warmup_steps={} arm=assistant rounds={rounds} \
                 mean_emitted_per_round={:.3} accept_rate_over_submitted={:.4}",
                emitted as f64 / wall,
                primed.prefill_s,
                depth as f64 / primed.prefill_s,
                WARMUP_STEPS_16_ABSORB_PIPELINE_AND_MIRROR_FIRST_TOUCH,
                emitted as f64 / rounds as f64,
                accepted as f64 / submitted.max(1) as f64,
            );
            eprintln!(
                "[G4ASSIST-COST] depth={depth} k={k} decode_ms={decode_med:.2} \
                 verify_chunk_ms={verify_med:.2} draft_ms={draft_med:.2} \
                 verify_over_decode={:.3} breakeven_emitted_per_round={:.3} \
                 basis=(nonflash decode graph, verify_chain m-rows, drafter on {:?})",
                verify_med / decode_med,
                (draft_med + verify_med) / decode_med,
                h.device()
            );
        } else {
            let mut cur = anchor;
            for _ in 0..WARMUP_STEPS_16_ABSORB_PIPELINE_AND_MIRROR_FIRST_TOUCH {
                cur = m.decode_step(cur).expect("warmup decode step");
            }
            let mut step_ms: Vec<f64> = Vec::with_capacity(want);
            let t0 = std::time::Instant::now();
            for _ in 0..want {
                let t = std::time::Instant::now();
                cur = m.decode_step(cur).expect("timed decode step");
                step_ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let wall = t0.elapsed().as_secs_f64();
            let median = c::median_ms(&mut step_ms);
            eprintln!(
                "CTX-SCALING g4moe-assist-wgpu depth={depth} median_ms_tok={median:.3} \
                 tok_s={:.1} prime_s={:.1} prime_tok_s={:.1} prime=chunked steps={want} \
                 warmup_steps={} arm=plain",
                want as f64 / wall,
                primed.prefill_s,
                depth as f64 / primed.prefill_s,
                WARMUP_STEPS_16_ABSORB_PIPELINE_AND_MIRROR_FIRST_TOUCH,
            );
        }
    }
}
