#![cfg(feature = "wgpu")]

mod common;
use common::bf16_lin_gow_bias as bf16_lin;
use common::have_gpu;
use common::LcgSplitMix64TwoSided as Lcg;
use common::mx_stack;
use common::tiny_config_gpt_oss as tiny_config;
mod hub_snapshot;

use nv_models::gpt_oss_wgpu as gow;
use nv_models::gpt_oss_wgpu::{GptOssConfig, GptOssLayerType};
use nv_quant::mxfp4::Mxfp4Tensor;

const CHUNK_LENGTHS_COVER_FULL_TAIL_AND_SHORTER_THAN_M: [usize; 3] = [40, 33, 9];

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
                    .map(|_| r.next_f32() * 0.5)
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

fn prompt_ids(cfg: &GptOssConfig, n: usize, salt: u32) -> Vec<u32> {
    (0..n)
        .map(|i| ((i as u32 * 7 + salt * 13 + 1) % (cfg.vocab_size as u32 - 1)) + 1)
        .collect()
}

fn per_token_logits(cfg: &GptOssConfig, hw: &gow::HostWeights, ids: &[u32]) -> (u32, Vec<f32>) {
    let mut m = gow::GptOssWgpu::new(cfg.clone(), hw, 64).expect("build per-token model");
    let (last, rest) = ids.split_last().expect("non-empty prompt");
    for t in rest {
        m.prefill_step(*t).expect("per-token prefill step");
    }
    m.decode_step_logits(*last).expect("decode")
}

fn chunked_logits(
    cfg: &GptOssConfig,
    hw: &gow::HostWeights,
    ids: &[u32],
) -> (u32, Vec<f32>, usize, usize) {
    let mut m = gow::GptOssWgpu::new(cfg.clone(), hw, 64).expect("build chunked model");
    let chunk = m.prefill_chunk_len();
    assert!(
        chunk >= 2,
        "GptOssWgpu::prefill_chunk_len() is {chunk}: the decoder still replays one token per \
         command buffer, which is exactly what this suite exists to refuse. Do not relax this \
         bound -- a chunk of 1 is the per-token path wearing the chunked path's name."
    );
    let (last, rest) = ids.split_last().expect("non-empty prompt");
    let done = m.prefill_tokens(rest).expect("chunked prefill");
    for t in &rest[done..] {
        m.prefill_step(*t).expect("tail prefill step");
    }
    let (tok, logits) = m.decode_step_logits(*last).expect("decode");
    (tok, logits, done, chunk)
}

#[test]
fn chunked_prefill_logits_are_bit_identical_to_per_token_replay() {
    if !have_gpu() {
        panic!("needs a wgpu adapter; a skipped identity proof reads as a passed one");
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xC0FFEE);
    let mut seen: Vec<(usize, u32, usize)> = Vec::new();
    for (i, n) in CHUNK_LENGTHS_COVER_FULL_TAIL_AND_SHORTER_THAN_M
        .into_iter()
        .enumerate()
    {
        let ids = prompt_ids(&cfg, n, i as u32);
        let (want_tok, want) = per_token_logits(&cfg, &hw, &ids);
        let (got_tok, got, done, chunk) = chunked_logits(&cfg, &hw, &ids);
        assert_eq!(want.len(), got.len(), "logit width changed at n={n}");
        let diff = want
            .iter()
            .zip(&got)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diff,
            0,
            "n={n} (chunk={chunk}, {done} of {} prompt tokens went through prefill_tokens): {diff} \
             of {} logits differ bit-for-bit between chunked prefill and per-token replay. \
             Chunk size is bookkeeping, not semantics. The likely cause is the MoE slot axis: \
             gow_gemv_mx spends wid.z on experts, and the chunked arm spends the SAME axis on \
             m*k_top slots, taking the expert from sel[slot] and the chunk row from slot/k_top -- \
             get that divisor or either stride wrong and every chunk row reads token 0's \
             activations. Do NOT relax this to argmax or to a tolerance: a tiny model's logits \
             barely depend on its context, so both of those oracles pass while every KV entry is \
             wrong.",
            n - 1,
            want.len()
        );
        assert_eq!(
            want_tok, got_tok,
            "argmax token differs at n={n} (replay {want_tok} vs chunked {got_tok}); the bit \
             comparison above is the gate that bites first, and a mutation run that reaches HERE \
             instead is one the argmax oracle would also have caught"
        );
        seen.push((n, want_tok, done));
    }
    assert!(
        seen.iter().any(|(_, t, _)| *t != seen[0].1),
        "every prompt length produced the same argmax token {:?}; this oracle is not reading the \
         context at all and would pass with the prefill graph deleted",
        seen
    );
    assert!(
        seen.iter().all(|(_, _, done)| *done > 0),
        "prefill_tokens consumed nothing at some length, so the chunked graph never ran: {seen:?}"
    );
    eprintln!("[chunked-prefill] (len, argmax, tokens via prefill_tokens) = {seen:?}");
}

#[test]
fn wgpu_chunked_prefill_is_invariant_to_chunk_size() {
    if !have_gpu() {
        panic!("needs a wgpu adapter; a skipped invariance proof reads as a passed one");
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x5EED);
    let ids = prompt_ids(&cfg, 37, 3);
    let mut rows: Vec<(usize, u32, Vec<f32>)> = Vec::new();
    for m in [4usize, 8, 16] {
        let mut model = gow::GptOssWgpu::new(cfg.clone(), &hw, 64).expect("build");
        assert!(
            model.prefill_chunk_len() >= 2,
            "chunked prefill absent at m={m}"
        );
        let (last, rest) = ids.split_last().expect("non-empty");
        let mut done = 0usize;
        while done < rest.len() {
            let take = m.min(rest.len() - done);
            if take < 2 {
                break;
            }
            let got = model
                .prefill_tokens(&rest[done..done + take])
                .expect("chunked prefill");
            assert!(got > 0, "prefill_tokens made no progress at m={m}");
            done += got;
        }
        for t in &rest[done..] {
            model.prefill_step(*t).expect("tail");
        }
        let (tok, logits) = model.decode_step_logits(*last).expect("decode");
        rows.push((m, tok, logits));
    }
    let (m0, t0, l0) = &rows[0];
    for (m, t, l) in &rows[1..] {
        let diff = l0
            .iter()
            .zip(l)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diff, 0,
            "feeding the same prompt in {m0}-token and {m}-token slices produced {diff} differing \
             logits (argmax {t0} vs {t}). Chunk size is bookkeeping, not semantics."
        );
    }
    eprintln!(
        "[chunk-invariance] argmax {t0} identical across slice sizes {:?}",
        rows.iter().map(|(m, _, _)| *m).collect::<Vec<_>>()
    );
}

fn gptoss_snapshot() -> Option<std::path::PathBuf> {
    hub_snapshot::dir_from_env_or_hub("NV_GPTOSS_DIR", "openai/gpt-oss-20b", &["config.json"])
}

#[test]
#[ignore = "loads ~13 GB of MXFP4 weights; set NV_GPTOSS_WGPU_TEST=1"]
fn gptoss_real_weights_chunked_prefill_matches_per_token() {
    if std::env::var("NV_GPTOSS_WGPU_TEST").is_err() {
        eprintln!("[skip] NV_GPTOSS_WGPU_TEST not set");
        return;
    }
    if !have_gpu() {
        panic!("real-weights test needs a wgpu adapter");
    }
    let dir = gptoss_snapshot().expect("no gpt-oss-20b snapshot");
    let cfg = GptOssConfig::from_hf_json_file(dir.join("config.json")).expect("config");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let max_seq: usize = std::env::var("NV_GPTOSS_MAX_SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9216);
    let t0 = std::time::Instant::now();
    let mut gpu =
        gow::GptOssWgpu::from_loader(cfg.clone(), &loader, max_seq).expect("build from loader");
    eprintln!(
        "[real] built in {:.1}s, passes_per_token={} prefill_chunk_len={} prefill_passes={}",
        t0.elapsed().as_secs_f64(),
        gpu.pass_count(),
        gpu.prefill_chunk_len(),
        gpu.prefill_pass_count()
    );
    assert!(
        gpu.prefill_chunk_len() >= 2,
        "chunked prefill did not engage on the real checkpoint"
    );

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let corpus = std::env::var("NV_GPTOSS_PREFILL_TEXT").unwrap_or_else(|_| {
        std::iter::repeat_n(
            "The prefill path must produce the same continuation as the replay path. ",
            1400,
        )
        .collect::<String>()
    });
    let full: Vec<u32> = tok
        .encode(corpus.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();
    let lengths: Vec<usize> = std::env::var("NV_GPTOSS_PREFILL_LENS")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect::<Vec<usize>>()
        })
        .unwrap_or_else(|| vec![512, 2048, 8192]);
    let n_new: usize = std::env::var("NV_GPTOSS_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    for n in lengths {
        if n > full.len() || n + n_new + 2 > max_seq {
            eprintln!(
                "[real] len {n} unavailable (corpus {} tokens, max_seq {max_seq})",
                full.len()
            );
            continue;
        }
        let ids = &full[..n];
        let (last, rest) = ids.split_last().expect("non-empty");

        gpu.reset().expect("reset");
        let t_chunked = std::time::Instant::now();
        let done = gpu.prefill_tokens(rest).expect("chunked prefill");
        for t in &rest[done..] {
            gpu.prefill_step(*t).expect("tail");
        }
        let mut next = gpu.decode_step(*last).expect("decode");
        let chunked_ms = t_chunked.elapsed().as_secs_f64() * 1000.0;
        let mut chunked_out = vec![next];
        for _ in 1..n_new {
            next = gpu.decode_step(next).expect("decode");
            chunked_out.push(next);
        }

        gpu.reset().expect("reset");
        let t_replay = std::time::Instant::now();
        for t in rest {
            gpu.prefill_step(*t).expect("replay");
        }
        let mut next = gpu.decode_step(*last).expect("decode");
        let replay_ms = t_replay.elapsed().as_secs_f64() * 1000.0;
        let mut replay_out = vec![next];
        for _ in 1..n_new {
            next = gpu.decode_step(next).expect("decode");
            replay_out.push(next);
        }

        eprintln!(
            "[real] n={n} chunked {chunked_ms:.1} ms ({} via prefill_tokens) vs replay \
             {replay_ms:.1} ms = {:.2}x; chunked {chunked_out:?} replay {replay_out:?}",
            done,
            replay_ms / chunked_ms
        );
        assert_eq!(
            chunked_out, replay_out,
            "n={n}: chunked prefill and per-token replay produced different continuations on the \
             real checkpoint"
        );
    }
}
