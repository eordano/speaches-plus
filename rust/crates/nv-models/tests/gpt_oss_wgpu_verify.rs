#![cfg(feature = "wgpu")]

mod common;
use common::bf16_lin_gow_bias as bf16_lin;
use common::bit_diff;
use common::have_gpu;
use common::LcgSplitMix64TwoSided as Lcg;
use common::mx_stack;
use common::tiny_config_gpt_oss as tiny_config;
mod hub_snapshot;

use nv_models::gpt_oss_wgpu as gow;
use nv_models::gpt_oss_wgpu::{GptOssConfig, GptOssLayerType, ImageRowSplice};
use nv_quant::mxfp4::Mxfp4Tensor;

const MAX_SEQ: usize = 64;

const CHAIN_WIDTHS_COVER_ONE_ROW_A_SHORT_CHAIN_AND_THE_FULL_VERIFY_WIDTH: [usize; 3] = [1, 3, 6];

fn norm_vec(r: &mut Lcg, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| half::bf16::from_f32(1.0 + r.next_f32() * 0.1).to_bits())
        .collect()
}

fn tiny_weights(cfg: &GptOssConfig, seed: u64) -> gow::HostWeights {
    let mut r = Lcg::new(seed);
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let layers = (0..cfg.num_hidden_layers)
        .map(|_| gow::HostLayer {
            input_ln: norm_vec(&mut r, h),
            post_attn_ln: norm_vec(&mut r, h),
            attn: gow::HostAttn {
                q: bf16_lin(&mut r, cfg.num_attention_heads * hd, h, 0.2, true),
                k: bf16_lin(&mut r, cfg.num_key_value_heads * hd, h, 0.2, true),
                v: bf16_lin(&mut r, cfg.num_key_value_heads * hd, h, 0.2, true),
                o: bf16_lin(&mut r, h, cfg.num_attention_heads * hd, 0.2, true),
                sinks: (0..cfg.num_attention_heads)
                    .map(|_| 1.0 + r.next_f32() * 0.5)
                    .collect(),
            },
            moe: gow::HostMoe {
                router: bf16_lin(&mut r, cfg.num_local_experts, h, 0.2, true),
                gate_up: mx_stack(
                    &mut r,
                    cfg.num_local_experts,
                    2 * cfg.intermediate_size,
                    h,
                    0.2,
                ),
                down: mx_stack(&mut r, cfg.num_local_experts, h, cfg.intermediate_size, 0.2),
            },
        })
        .collect();
    gow::HostWeights {
        embed: r.bf16_vec(cfg.vocab_size * h, 0.6),
        final_norm: norm_vec(&mut r, h),
        lm_head: r.bf16_vec(cfg.vocab_size * h, 0.2),
        layers,
    }
}

fn tiny_weights_without_sinks(cfg: &GptOssConfig, seed: u64) -> gow::HostWeights {
    let mut out = tiny_weights(cfg, seed);
    for l in &mut out.layers {
        l.attn.sinks = l.attn.sinks.iter().map(|_| 0.0).collect();
    }
    out
}

fn ids(cfg: &GptOssConfig, n: usize, salt: u32) -> Vec<u32> {
    (0..n)
        .map(|i| ((i as u32 * 7 + salt * 13 + 1) % (cfg.vocab_size as u32 - 1)) + 1)
        .collect()
}

fn primed(cfg: &GptOssConfig, hw: &gow::HostWeights, prompt: &[u32]) -> gow::GptOssWgpu {
    let mut m = gow::GptOssWgpu::new(cfg.clone(), hw, MAX_SEQ).expect("build model");
    for t in prompt {
        m.prefill_step(*t).expect("prime with per-token prefill");
    }
    m
}

#[test]
fn verify_wgsl_validates_under_naga_without_a_gpu() {
    for (name, source) in gow::verify_audit_sources() {
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{name}: wgsl parse failed: {e}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: wgsl validation failed: {e:?}"));
        eprintln!("[naga-ok] {name} ({} bytes)", source.len());
    }
}

#[test]
fn mx_e2m1_alu_decode_is_bit_exact_where_the_substrate_shift_decode_loses_half_codes_to_ftz() {
    use nv_kernels::wgpu_backend::{compose, dispatch, WgpuContext};
    let ctx = match WgpuContext::shared() {
        Ok(ctx) => ctx,
        Err(e) => panic!("needs a wgpu adapter; a skipped decode proof reads as a passed one: {e}"),
    };
    let source = compose(include_str!("../../nv-kernels/wgsl/gow_mx.wgsl"));
    let w = dispatch::storage_from_slice(ctx, "e2m1-probe-w", &[0x7654_3210u32, 0xFEDC_BA98, 0, 0]);
    let y = dispatch::storage_zeroed(ctx, "e2m1-probe-y", 32 * 4);
    dispatch::run(
        ctx,
        "e2m1-probe",
        &source,
        "gow_mx_e2m1_probe",
        &[(10, &w), (14, &y)],
        (1, 1, 1),
    )
    .expect("run gow_mx_e2m1_probe");
    let out: Vec<f32> = dispatch::read_back(ctx, &y, 32).expect("read back");
    let mut shift_flushes = 0;
    for code in 0u8..16 {
        let want = nv_quant::mxfp4::decode_e2m1(code);
        let alu = out[code as usize];
        assert_eq!(
            alu.to_bits(),
            want.to_bits(),
            "code {code}: gow_mx_e2m1 produced {alu:e}, host decode_e2m1 says {want:e}; the ALU \
             placement must be a bit-exact stand-in for the old E2M1_TABLE lookup or the mx gemv \
             is no longer the audited numeric route"
        );
        let shifted = out[16 + code as usize];
        if code & 7 == 1 {
            assert!(
                shifted == want || shifted == 0.0,
                "code {code}: shift-decode*2^126 gave {shifted:e}, neither the true {want:e} nor \
                 the FTZ flush to zero; something other than denormal handling is wrong"
            );
            if shifted == 0.0 && want != 0.0 {
                shift_flushes += 1;
            }
        } else {
            assert_eq!(
                shifted.to_bits(),
                want.to_bits(),
                "code {code}: shift-decode*2^126 must be exact for every non-subnormal landing"
            );
        }
    }
    eprintln!(
        "[e2m1-probe] half-codes flushed by shift-decode restore: {shift_flushes} of 2 \
         (nonzero means this driver flushes f32 denormal multiplicands, which is why \
         gow_mx_e2m1 places bits arithmetically instead of using \
         e2m1_shift_decode_scale_must_carry_2pow126)"
    );
}

#[test]
fn verify_chain_rows_are_bit_identical_to_the_same_tokens_stepped_one_at_a_time() {
    if !have_gpu() {
        panic!("needs a wgpu adapter; a skipped identity proof reads as a passed one");
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xC0FFEE);
    let prompt = ids(&cfg, 11, 1);
    let probe = primed(&cfg, &hw, &prompt);
    let rows = probe.verify_max_rows();
    assert!(
        rows >= 2,
        "GptOssWgpu::verify_max_rows() is {rows}: without a multi-row verify epilogue every \
         speculative round degrades to one token per submission, which is exactly what this \
         suite exists to refuse"
    );
    drop(probe);

    let mut seen: Vec<(usize, Vec<u32>)> = Vec::new();
    for k in CHAIN_WIDTHS_COVER_ONE_ROW_A_SHORT_CHAIN_AND_THE_FULL_VERIFY_WIDTH {
        let k = k.min(rows);
        let chain = ids(&cfg, k, 5);
        let mut chained = primed(&cfg, &hw, &prompt);
        let pos0 = chained.current_pos();
        let (got_toks, got) = chained.verify_chain_logits(&chain).expect("verify_chain");
        assert_eq!(
            chained.current_pos(),
            pos0,
            "verify_chain committed {k} rows by itself; commit belongs to advance(n) so that a \
             partial accept costs nothing"
        );

        let mut stepped = primed(&cfg, &hw, &prompt);
        let mut want: Vec<f32> = Vec::new();
        let mut want_toks: Vec<u32> = Vec::new();
        for t in &chain {
            let (tok, l) = stepped.decode_step_logits(*t).expect("decode step");
            want_toks.push(tok);
            want.extend_from_slice(&l);
        }

        assert_eq!(got.len(), want.len(), "logit width changed at k={k}");
        for (row, (g, w)) in got
            .chunks(cfg.vocab_size)
            .zip(want.chunks(cfg.vocab_size))
            .enumerate()
        {
            let diff = bit_diff(g, w);
            assert_eq!(
                diff, 0,
                "k={k} row {row}: {diff} of {} logits differ bit-for-bit between the M-row verify \
                 forward and the same tokens stepped one at a time. The verify epilogue rides the \
                 prefill trunk, so a difference here is either the epilogue reading the wrong row \
                 (x_row_words / y_off_words on gow_v_lmhead) or the prefill attention dropping the \
                 per-head sink for rows above 0. Do NOT relax this to argmax or to a tolerance: a \
                 tiny model's logits barely depend on its context, so both of those oracles pass \
                 while every KV entry is wrong.",
                cfg.vocab_size
            );
        }
        assert_eq!(
            got_toks, want_toks,
            "k={k}: per-row argmax differs from the stepped argmax; the bit comparison above is \
             the gate that bites first"
        );
        seen.push((k, got_toks));
    }
    let widest = seen
        .iter()
        .max_by_key(|(k, _)| *k)
        .expect("at least one chain width ran");
    assert!(
        widest.1.iter().any(|t| *t != widest.1[0]),
        "the {}-row chain emitted the same token on every row {:?}; a verify epilogue that \
         broadcasts row 0 over every row would pass the identity assertions on a model whose \
         logits barely move, so this oracle refuses that shape",
        widest.0,
        widest.1
    );
    eprintln!("[verify-chain] (k, argmax rows) = {seen:?}");
}

#[test]
fn accepting_a_prefix_of_a_verified_chain_leaves_the_pure_one_token_stream() {
    if !have_gpu() {
        panic!("needs a wgpu adapter; a skipped losslessness proof reads as a passed one");
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x5EED);
    let prompt = ids(&cfg, 9, 2);
    let mut spec = primed(&cfg, &hw, &prompt);
    let rows = spec.verify_max_rows();
    assert!(
        rows >= 3,
        "verify width {rows} is too narrow for a partial accept"
    );
    let k = rows.min(6);
    let accept = k / 2;
    let chain = ids(&cfg, k, 4);
    let toks = spec.verify_chain(&chain).expect("verify_chain");
    spec.advance(accept).expect("advance the accepted prefix");
    assert_eq!(
        spec.current_pos(),
        prompt.len() + accept,
        "advance(n) must move exactly n rows"
    );

    let mut plain = primed(&cfg, &hw, &prompt);
    for t in &chain[..accept] {
        plain.decode_step(*t).expect("step the accepted prefix");
    }

    let tail = ids(&cfg, 5, 7);
    let mut spec_out: Vec<(u32, Vec<f32>)> = Vec::new();
    let mut plain_out: Vec<(u32, Vec<f32>)> = Vec::new();
    for t in &tail {
        spec_out.push(spec.decode_step_logits(*t).expect("spec continuation"));
        plain_out.push(plain.decode_step_logits(*t).expect("plain continuation"));
    }
    for (i, ((st, sl), (pt, pl))) in spec_out.iter().zip(&plain_out).enumerate() {
        let diff = bit_diff(sl, pl);
        assert_eq!(
            diff, 0,
            "continuation step {i}: {diff} logits differ (spec {st} vs plain {pt}) after a \
             partial accept. Rows written past the accepted prefix must be invisible: gpt-oss \
             attention bounds every row by base+t, so a difference here means the speculative KV \
             rows leaked into a later read"
        );
    }
    assert_eq!(
        toks.len(),
        k,
        "verify_chain returned {} rows for a {k}-token chain",
        toks.len()
    );
    eprintln!("[partial-accept] k={k} accepted={accept} rows={toks:?}");
}

#[test]
fn attention_sinks_stay_load_bearing_in_every_verify_row() {
    if !have_gpu() {
        panic!("needs a wgpu adapter; a skipped sink proof reads as a passed one");
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xBEEF);
    let flat = tiny_weights_without_sinks(&cfg, 0xBEEF);
    let prompt = ids(&cfg, 7, 3);
    let mut with = primed(&cfg, &hw, &prompt);
    let k = with.verify_max_rows().min(4);
    assert!(k >= 2, "verify width too narrow to exercise M>1 sinks");
    let chain = ids(&cfg, k, 6);
    let (_, got) = with.verify_chain_logits(&chain).expect("verify with sinks");
    let mut zero = primed(&cfg, &flat, &prompt);
    let (_, flatl) = zero
        .verify_chain_logits(&chain)
        .expect("verify without sinks");

    for (row, (a, b)) in got
        .chunks(cfg.vocab_size)
        .zip(flatl.chunks(cfg.vocab_size))
        .enumerate()
    {
        assert!(
            bit_diff(a, b) > 0,
            "row {row} is unchanged when every attention sink is zeroed, so the M-row verify \
             forward is not folding the sink into that row's softmax max and denominator (the \
             gow_pf_attn pad_max/pad_z fold). The bit-identity suite would still pass if BOTH \
             paths dropped the sink, which is why this oracle exists"
        );
    }

    let mut stepped = primed(&cfg, &flat, &prompt);
    let mut want: Vec<f32> = Vec::new();
    for t in &chain {
        want.extend_from_slice(&stepped.decode_step_logits(*t).expect("step").1);
    }
    assert_eq!(
        bit_diff(&flatl, &want),
        0,
        "the zero-sink weights break M-row / one-at-a-time identity, so the identity above rests \
         on a sink cancellation rather than on the graph"
    );
}

#[test]
fn splicing_the_models_own_embedding_rows_is_bit_identical_to_plain_prefill() {
    if !have_gpu() {
        panic!("needs a wgpu adapter; a skipped splice proof reads as a passed one");
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xD00D);
    let m = {
        let probe = gow::GptOssWgpu::new(cfg.clone(), &hw, MAX_SEQ).expect("build probe");
        probe.prefill_chunk_len()
    };
    assert!(
        m >= 2,
        "embed-row splice needs the chunked prefill graph (m={m})"
    );
    let n = 2 * m;
    let prompt = ids(&cfg, n, 8);
    let next = ids(&cfg, 1, 9)[0];
    let hidden = cfg.hidden_size;

    let mut plain = gow::GptOssWgpu::new(cfg.clone(), &hw, MAX_SEQ).expect("build plain");
    let done = plain.prefill_tokens(&prompt).expect("plain prefill");
    assert_eq!(done, n, "plain prefill consumed {done} of {n} tokens");
    let (_, want) = plain.decode_step_logits(next).expect("plain decode");

    let mut empty = gow::GptOssWgpu::new(cfg.clone(), &hw, MAX_SEQ).expect("build empty-splice");
    let done = empty
        .prefill_tokens_with_image_rows(&prompt, &[])
        .expect("zero-splice prefill");
    assert_eq!(done, n, "zero-splice prefill consumed {done} of {n} tokens");
    let (_, got) = empty.decode_step_logits(next).expect("decode");
    assert_eq!(
        bit_diff(&want, &got),
        0,
        "prefill_tokens_with_image_rows with no splices is not the plain prefill path"
    );

    let at = m - 2;
    let span = 5;
    let rows_bf16: Vec<u16> = prompt[at..at + span]
        .iter()
        .flat_map(|&t| {
            let base = t as usize * hidden;
            hw.embed[base..base + hidden].to_vec()
        })
        .collect();
    let mut spliced = gow::GptOssWgpu::new(cfg.clone(), &hw, MAX_SEQ).expect("build spliced");
    let done = spliced
        .prefill_tokens_with_image_rows(
            &prompt,
            &[ImageRowSplice {
                position: at,
                rows_bf16,
            }],
        )
        .expect("spliced prefill");
    assert_eq!(done, n, "spliced prefill consumed {done} of {n} tokens");
    let (_, got) = spliced.decode_step_logits(next).expect("decode");
    assert_eq!(
        bit_diff(&want, &got),
        0,
        "splicing rows that ARE the model's own embedding rows for the same token ids changed \
         the logits, so the splice pass is writing the wrong row, the wrong stride, or is \
         landing after the first rmsnorm instead of over the gathered rows. The splice spans a \
         chunk boundary at {at}..{} with chunk width {m}, so a per-chunk rel_pos error also \
         lands here",
        at + span
    );

    let noise: Vec<u16> = (0..span * hidden)
        .map(|i| half::bf16::from_f32(0.25 + (i % 7) as f32 * 0.01).to_bits())
        .collect();
    let mut other = gow::GptOssWgpu::new(cfg.clone(), &hw, MAX_SEQ).expect("build other");
    other
        .prefill_tokens_with_image_rows(
            &prompt,
            &[ImageRowSplice {
                position: at,
                rows_bf16: noise,
            }],
        )
        .expect("noise splice prefill");
    let (_, got) = other.decode_step_logits(next).expect("decode");
    assert!(
        bit_diff(&want, &got) > 0,
        "replacing {span} embedding rows with unrelated values changed nothing, so the splice \
         pass never runs and the identity above is vacuous"
    );
}

fn gptoss_snapshot() -> Option<std::path::PathBuf> {
    hub_snapshot::dir_from_env_or_hub("NV_GPTOSS_DIR", "openai/gpt-oss-20b", &["config.json"])
}

#[test]
#[ignore = "loads ~13 GB of MXFP4 weights; set NV_GPTOSS_WGPU_TEST=1"]
fn gptoss_real_weights_verify_chain_matches_per_token_decode() {
    if std::env::var("NV_GPTOSS_WGPU_TEST").is_err() {
        eprintln!("[skip] NV_GPTOSS_WGPU_TEST not set");
        return;
    }
    if !have_gpu() {
        panic!("real-weights test needs a wgpu adapter");
    }
    let Some(dir) = gptoss_snapshot() else {
        hub_snapshot::precondition_absent(
            "gptoss_real_weights_verify_chain_matches_per_token_decode",
            "no openai/gpt-oss-20b snapshot",
            "set NV_GPTOSS_DIR to a gpt-oss-20b snapshot dir with safetensors, or cache the repo",
        );
        return;
    };
    let cfg = GptOssConfig::from_hf_json_file(dir.join("config.json")).expect("config");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let max_seq: usize = std::env::var("NV_GPTOSS_MAX_SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let mut gpu = gow::GptOssWgpu::from_loader(cfg.clone(), &loader, max_seq).expect("build");
    let rows = gpu.verify_max_rows();
    assert!(rows >= 2, "verify epilogue absent on the real checkpoint");

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let text = std::env::var("NV_GPTOSS_PREFILL_TEXT").unwrap_or_else(|_| {
        "The verify forward must produce the same rows as one-at-a-time stepping. ".repeat(8)
    });
    let full: Vec<u32> = tok
        .encode(text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();
    let k = rows.min(8);
    assert!(full.len() > k + 4, "corpus too short for a {k}-row chain");
    let prompt = &full[..full.len() - k];
    let chain = &full[full.len() - k..];

    gpu.reset().expect("reset");
    let done = gpu.prefill_tokens(prompt).expect("chunked prefill");
    for t in &prompt[done..] {
        gpu.prefill_step(*t).expect("tail");
    }
    let got = gpu.verify_chain(chain).expect("verify_chain");

    gpu.reset().expect("reset");
    let done = gpu.prefill_tokens(prompt).expect("chunked prefill");
    for t in &prompt[done..] {
        gpu.prefill_step(*t).expect("tail");
    }
    let mut want: Vec<u32> = Vec::new();
    for t in chain {
        want.push(gpu.decode_step(*t).expect("decode step"));
    }
    assert_eq!(
        got, want,
        "real-weights verify_chain rows differ from the same tokens stepped one at a time"
    );
    eprintln!("[real-verify] k={k} rows={got:?}");
}
