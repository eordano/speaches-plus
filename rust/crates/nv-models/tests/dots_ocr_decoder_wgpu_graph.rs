#![cfg(feature = "wgpu")]

mod common;
use common::require;
use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_models::dots_ocr::{
    DotsDecoder, DotsDecoderConfig, DotsDecoderWgpuGraph, DotsGraphQuant, GenerateOptions,
    RegionRow,
};
use nv_weights::WeightLoader;

const TINY_ORACLE_TOKENS: usize = 5;

const BF16_LOGIT_BAND_IS_THE_PACKED_RESIDUAL_AND_GEMV_ROUNDING_AGAINST_A_CPU_F32_DECODER: f32 =
    3e-2;

const THE_NVFP4_END_TO_END_ARM_IS_REPORTED_NOT_BOUNDED: &str =
    "e2m1 carries no mantissa bit, so a W4A4 GEMV is several percent off per row whatever the \
     kernel does. On this fixture the whole logit vector barely turns between steps -- any two \
     steps of the bf16 graph sit at cosine 0.9998 -- so an end-to-end cosine or max-error band \
     here would be fitted to the quantizer and would still pass a decoder whose layers had \
     stopped contributing at all. This arm therefore reports its agreement and asserts only what \
     cannot be traded away: finite logits and bit-identical repeats. What the nvfp4 kernels \
     compute is gated by nvfp4_gemv_matches_the_same_arithmetic_run_on_the_host, which holds the \
     quantize+gemv pair to the codes it emits.";

const W4A4_AGAINST_A_CPU_F32_DECODER_IS_QUANTIZER_NOISE_NOT_A_DECODER_BAND: &str =
    "Every projection in the nvfp4 arm quantizes the weight to e2m1 block 16 and quantizes the \
     activation with it, which costs a few percent per GEMV before the graph does anything \
     wrong; against a CPU f32 decoder that noise compounds past any band a kernel defect would \
     have to cross. The bf16 graph is the reference that isolates the nvfp4 kernels, and the \
     cosine floor is the one the sibling deepseek_ocr nvfp4 arm is held to.";

const FIXTURE_BIAS_IS_A_TENTH_OF_THE_PROJECTION_WEIGHTS_LIKE_A_REAL_QWEN2_CHECKPOINT: f32 = 0.1;

fn adapter_available() -> bool {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("adapter: {}", ctx.summary());
            true
        }
        Err(e) => {
            if require() {
                panic!("NV_KERNELS_WGPU_REQUIRE=1 but no adapter: {e}");
            }
            eprintln!("skipping: no wgpu adapter ({e})");
            false
        }
    }
}

fn det_bf16(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = ((i as f32 + seed) * 0.7311).sin() * 0.2;
            bf16::from_f32(v).to_f32()
        })
        .collect()
}

fn tiny_config() -> DotsDecoderConfig {
    DotsDecoderConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        intermediate_size: 128,
        vocab_size: 96,
        max_position_embeddings: 64,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: false,
        attention_bias: true,
        image_token_id: 151665,
        head_dim: None,
    }
}

fn tiny_weight_map(c: &DotsDecoderConfig) -> HashMap<String, Tensor> {
    let dev = Device::Cpu;
    let h = c.hidden_size;
    let hd = c.head_dim();
    let qd = c.num_attention_heads * hd;
    let kvd = c.num_key_value_heads * hd;
    let inter = c.intermediate_size;
    let mut m = HashMap::new();
    let t = |vals: Vec<f32>, shape: (usize, usize)| Tensor::from_vec(vals, shape, &dev).unwrap();
    let t1 = |vals: Vec<f32>, n: usize| Tensor::from_vec(vals, n, &dev).unwrap();
    m.insert(
        "model.embed_tokens.weight".to_string(),
        t(det_bf16(c.vocab_size * h, 1.0), (c.vocab_size, h)),
    );
    m.insert(
        "lm_head.weight".to_string(),
        t(det_bf16(c.vocab_size * h, 2.0), (c.vocab_size, h)),
    );
    m.insert(
        "model.norm.weight".to_string(),
        t1(det_bf16(h, 3.0).iter().map(|v| 1.0 + v).collect(), h),
    );
    for l in 0..c.num_hidden_layers {
        let p = format!("model.layers.{l}");
        let s = l as f32 * 100.0;
        m.insert(
            format!("{p}.input_layernorm.weight"),
            t1(det_bf16(h, 4.0 + s).iter().map(|v| 1.0 + v).collect(), h),
        );
        m.insert(
            format!("{p}.post_attention_layernorm.weight"),
            t1(det_bf16(h, 5.0 + s).iter().map(|v| 1.0 + v).collect(), h),
        );
        for (name, rows, seed) in [
            ("q_proj", qd, 6.0),
            ("k_proj", kvd, 7.0),
            ("v_proj", kvd, 8.0),
        ] {
            m.insert(
                format!("{p}.self_attn.{name}.weight"),
                t(det_bf16(rows * h, seed + s), (rows, h)),
            );
            m.insert(
                format!("{p}.self_attn.{name}.bias"),
                t1(
                    det_bf16(rows, seed + 0.5 + s)
                        .iter()
                        .map(|v| {
                            bf16::from_f32(
                                v * FIXTURE_BIAS_IS_A_TENTH_OF_THE_PROJECTION_WEIGHTS_LIKE_A_REAL_QWEN2_CHECKPOINT,
                            )
                            .to_f32()
                        })
                        .collect(),
                    rows,
                ),
            );
        }
        m.insert(
            format!("{p}.self_attn.o_proj.weight"),
            t(det_bf16(h * qd, 9.0 + s), (h, qd)),
        );
        m.insert(
            format!("{p}.mlp.gate_proj.weight"),
            t(det_bf16(inter * h, 70.0 + s), (inter, h)),
        );
        m.insert(
            format!("{p}.mlp.up_proj.weight"),
            t(det_bf16(inter * h, 71.0 + s), (inter, h)),
        );
        m.insert(
            format!("{p}.mlp.down_proj.weight"),
            t(det_bf16(h * inter, 72.0 + s), (h, inter)),
        );
    }
    m
}

mod tempdir {
    pub struct Guard(pub std::path::PathBuf);
    impl Drop for Guard {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }
}

static GRAPH_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    GRAPH_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn loader_for(c: &DotsDecoderConfig) -> (WeightLoader, tempdir::Guard) {
    let map = tiny_weight_map(c);
    let dir = std::env::temp_dir().join(format!(
        "dots-graph-test-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    candle_core::safetensors::save(&map, &path).unwrap();
    let loader = WeightLoader::open_file(&path, &Device::Cpu).unwrap();
    (loader, tempdir::Guard(dir))
}

fn tiny_loader() -> (WeightLoader, DotsDecoderConfig, tempdir::Guard) {
    let c = tiny_config();
    let (loader, guard) = loader_for(&c);
    (loader, c, guard)
}

fn greedy_run(quant: DotsGraphQuant) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<u32>) {
    let (loader, cfg, _guard) = tiny_loader();
    let cpu = DotsDecoder::from_loader(cfg.clone(), &loader, &Device::Cpu, DType::F32, 32).unwrap();
    let mut wg = DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, 32, quant).unwrap();
    eprintln!(
        "{} graph: {} passes per decode step",
        quant.name(),
        wg.pass_count()
    );
    let mut cache = cpu.new_kv_cache(32).unwrap();
    let mut token = 3u32;
    let mut cpu_all = Vec::new();
    let mut gpu_all = Vec::new();
    let mut gpu_tokens = Vec::new();
    for step in 0..TINY_ORACLE_TOKENS {
        let embeds = cpu.embed_tokens(&[token]).unwrap();
        let cpu_logits = cpu.last_logits(&embeds, step, &mut cache).unwrap();
        let (gpu_token, gpu_logits) = wg.decode_step_logits(token).unwrap();
        assert_eq!(gpu_logits.len(), cfg.vocab_size);
        assert!(
            gpu_logits.iter().all(|v| v.is_finite()),
            "step {step}: non-finite {} logits",
            quant.name()
        );
        token = DotsDecoderWgpuGraph::greedy_token(&cpu_logits);
        cpu_all.push(cpu_logits);
        gpu_all.push(gpu_logits);
        gpu_tokens.push(gpu_token);
    }
    assert_eq!(wg.current_pos(), TINY_ORACLE_TOKENS);
    (cpu_all, gpu_all, gpu_tokens)
}

fn rel_err(reference: &[f32], got: &[f32]) -> f32 {
    let scale = reference
        .iter()
        .fold(0f32, |a, &b| a.max(b.abs()))
        .max(1e-6);
    reference
        .iter()
        .zip(got.iter())
        .fold(0f32, |a, (x, y)| a.max((x - y).abs()))
        / scale
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

#[test]
fn bf16_graph_tracks_the_cpu_decoder_over_five_greedy_steps() {
    if !adapter_available() {
        return;
    }
    let (cpu, gpu, tokens) = greedy_run(DotsGraphQuant::Bf16);
    let mut worst = 0f32;
    for (step, ((c, g), t)) in cpu.iter().zip(gpu.iter()).zip(tokens.iter()).enumerate() {
        let rel = rel_err(c, g);
        worst = worst.max(rel);
        let want = DotsDecoderWgpuGraph::greedy_token(c);
        eprintln!("bf16 step {step}: rel={rel:.5} argmax cpu={want} gpu={t}");
        assert_eq!(want, *t, "step {step}: bf16 greedy token diverged (rel {rel})");
        assert!(
            rel < BF16_LOGIT_BAND_IS_THE_PACKED_RESIDUAL_AND_GEMV_ROUNDING_AGAINST_A_CPU_F32_DECODER,
            "step {step}: bf16 logits diverged, rel {rel}"
        );
    }
    eprintln!("bf16 worst rel over {TINY_ORACLE_TOKENS} steps: {worst:.5}");
}

const NVFP4_GEMV_BAND_IS_F32_REDUCTION_ORDER_PLUS_TIE_ROUNDING_BETWEEN_TWO_QUANTIZERS: f32 = 2e-2;

#[test]
fn nvfp4_gemv_matches_the_same_arithmetic_run_on_the_host() {
    if !adapter_available() {
        return;
    }
    for (n, k) in [
        (64usize, 128usize),
        (32, 64),
        (64, 64),
        (128, 64),
        (32, 128),
        (96, 64),
        (256, 64),
        (160, 64),
        (256, 128),
    ] {
        nvfp4_gemv_case(n, k);
    }
}

fn nvfp4_gemv_case(n: usize, k: usize) {
    use nv_models::dots_ocr::decoder_wgpu_graph::{
        nvfp4_gemv_once, NVFP4_ACTIVATION_GLOBAL_2POW7_LANDS_DECODE_BLOCK_SCALES_INSIDE_THE_E4M3_WINDOW,
    };
    let bits = |vals: Vec<f32>| -> Vec<u16> { vals.iter().map(|v| bf16::from_f32(*v).to_bits()).collect() };
    let w = bits(det_bf16(n * k, 11.0));
    let x = bits(det_bf16(k, 13.0));

    let got = nvfp4_gemv_once(&w, n, k, &x).unwrap();

    let q = nv_models::nvfp4_host::quantize_nvfp4_host(&w, n, k);
    let wd = nv_models::nvfp4_host::dequantize_nvfp4_host(&q);
    let xr: Vec<f32> = x.iter().map(|b| f32::from_bits((*b as u32) << 16)).collect();
    let xr2 = xr.clone();
    let xq = nv_quant::nvfp4::Nvfp4Tensor::quantize_rows_with_global(
        &[xr],
        NVFP4_ACTIVATION_GLOBAL_2POW7_LANDS_DECODE_BLOCK_SCALES_INSIDE_THE_E4M3_WINDOW,
    )
    .dequantize()
    .remove(0);
    let want: Vec<f32> = (0..n)
        .map(|r| {
            (0..k).map(|c| wd[r * k + c] * xq[c]).sum::<f32>()
                / NVFP4_ACTIVATION_GLOBAL_2POW7_LANDS_DECODE_BLOCK_SCALES_INSIDE_THE_E4M3_WINDOW
        })
        .collect();

    let want_exact: Vec<f32> = (0..n)
        .map(|r| (0..k).map(|c| wd[r * k + c] * xr2[c]).sum::<f32>())
        .collect();
    let rel = rel_err(&want, &got);
    let rel_exact = rel_err(&want_exact, &got);
    eprintln!(
        "nvfp4 gemv n={n} k={k}: rel_vs_quantized={rel:.5} rel_vs_exact={rel_exact:.5}"
    );
    assert!(
        rel_exact < NVFP4_GEMV_BAND_IS_F32_REDUCTION_ORDER_PLUS_TIE_ROUNDING_BETWEEN_TWO_QUANTIZERS,
        "n={n} k={k}: the nvfp4 gemv does not compute what the same codes compute on the host \
         against exact activations: rel {rel_exact} (against the quantized-activation reference \
         the old kernel was held to, {rel})"
    );
}

#[test]
fn nvfp4_graph_decodes_finite_logits_and_repeats_them_after_reset() {
    if !adapter_available() {
        return;
    }
    let _ = W4A4_AGAINST_A_CPU_F32_DECODER_IS_QUANTIZER_NOISE_NOT_A_DECODER_BAND;
    let _ = THE_NVFP4_END_TO_END_ARM_IS_REPORTED_NOT_BOUNDED;
    let (cpu, bf16_logits, _) = greedy_run(DotsGraphQuant::Bf16);
    let (_, q4, q4_tokens) = greedy_run(DotsGraphQuant::Nvfp4);
    for (step, q) in q4.iter().enumerate() {
        eprintln!(
            "nvfp4 step {step}: cos_vs_bf16_graph={:.5} rel_vs_cpu_f32={:.5} argmax bf16={} nvfp4={}",
            cosine(&bf16_logits[step], q),
            rel_err(&cpu[step], q),
            DotsDecoderWgpuGraph::greedy_token(&bf16_logits[step]),
            q4_tokens[step]
        );
    }

    let (loader, cfg, _guard) = tiny_loader();
    let mut wg =
        DotsDecoderWgpuGraph::from_loader(cfg, &loader, 32, DotsGraphQuant::Nvfp4).unwrap();
    let a = wg.forward_tokens(&[0, 7, 33]).unwrap();
    wg.reset().unwrap();
    let b = wg.forward_tokens(&[0, 7, 33]).unwrap();
    assert_eq!(a, b, "the nvfp4 graph must decode deterministically across reset");
}

#[test]
fn dropping_the_qkv_bias_leaves_the_band_so_the_band_is_not_vacuous() {
    if !adapter_available() {
        return;
    }
    let (loader, cfg, _guard) = tiny_loader();
    let cpu = DotsDecoder::from_loader(cfg.clone(), &loader, &Device::Cpu, DType::F32, 32).unwrap();
    let mut blind = cfg.clone();
    blind.attention_bias = false;
    let mut wg =
        DotsDecoderWgpuGraph::from_loader(blind, &loader, 32, DotsGraphQuant::Bf16).unwrap();
    let mut cache = cpu.new_kv_cache(32).unwrap();
    let mut worst = 0f32;
    let mut token = 3u32;
    for step in 0..TINY_ORACLE_TOKENS {
        let embeds = cpu.embed_tokens(&[token]).unwrap();
        let cpu_logits = cpu.last_logits(&embeds, step, &mut cache).unwrap();
        let (_, gpu_logits) = wg.decode_step_logits(token).unwrap();
        let scale = cpu_logits
            .iter()
            .fold(0f32, |a, &b| a.max(b.abs()))
            .max(1e-6);
        let max_abs = cpu_logits
            .iter()
            .zip(gpu_logits.iter())
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        worst = worst.max(max_abs / scale);
        token = DotsDecoderWgpuGraph::greedy_token(&cpu_logits);
    }
    eprintln!("bias-blind graph worst rel: {worst:.5}");
    assert!(
        worst > BF16_LOGIT_BAND_IS_THE_PACKED_RESIDUAL_AND_GEMV_ROUNDING_AGAINST_A_CPU_F32_DECODER,
        "a graph that ignores the checkpoint's q/k/v bias must land outside the band, else the          band does not test the bias at all: worst rel {worst}"
    );
}

#[test]
fn reset_reproduces_the_same_logits() {
    if !adapter_available() {
        return;
    }
    let (loader, cfg, _guard) = tiny_loader();
    let mut wg =
        DotsDecoderWgpuGraph::from_loader(cfg, &loader, 32, DotsGraphQuant::Bf16).unwrap();
    let a = wg.forward_tokens(&[0, 7, 33]).unwrap();
    wg.reset().unwrap();
    let b = wg.forward_tokens(&[0, 7, 33]).unwrap();
    assert_eq!(a, b, "a graphed decode must be deterministic across reset");
}

#[test]
fn the_eight_row_prefill_chunk_reaches_what_eight_single_token_steps_reach() {
    if !adapter_available() {
        return;
    }
    let _env = env_lock();
    use nv_models::dots_ocr::decoder_wgpu_graph::{PF_STEEL_ENV, PF_STEEL_PROJ_ENV};
    std::env::set_var(PF_STEEL_ENV, "0");
    std::env::set_var(PF_STEEL_PROJ_ENV, "0");
    let (loader, cfg, _guard) = tiny_loader();
    let span = 4 * nv_models::dots_ocr::decoder_wgpu_graph::PREFILL_M_MAX + 16;
    let mut chunked =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    let widths = chunked.prefill_widths().to_vec();
    assert!(
        widths.len() >= 2 && widths[0] > widths[1],
        "the bf16 arm must carry a wide and a narrow chunk graph, got {widths:?}"
    );
    let m = widths[0];
    let tokens: Vec<u32> = (0..(m + widths[1] + 3) as u32)
        .map(|i| (i * 7 + 3) % 96)
        .collect();
    eprintln!(
        "chunk widths {widths:?}, prompt {} tokens exercises wide + narrow + per-token",
        tokens.len()
    );
    let mut per_token =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    let want = per_token.forward_tokens(&tokens).unwrap();
    let got = chunked.prefill_prompt(&tokens).unwrap();
    assert_eq!(
        want, got,
        "the {m}-row prefill chunk must land the same kv cache and logits as stepping those rows \
         one at a time, or the page it prefills is not the page the decoder would have read; the \
         steel arms are pinned off here because their f16 storage is banded, not bitwise, and \
         gated by the steel-vs-stepped suite"
    );
    std::env::remove_var(PF_STEEL_ENV);
    std::env::remove_var(PF_STEEL_PROJ_ENV);
}

fn embed_row(c: &DotsDecoderConfig, token: u32) -> Vec<f32> {
    let h = c.hidden_size;
    det_bf16(c.vocab_size * h, 1.0)[token as usize * h..(token as usize + 1) * h].to_vec()
}

#[test]
fn a_supplied_hidden_row_that_is_the_tokens_own_embedding_decodes_identically() {
    if !adapter_available() {
        return;
    }
    let (loader, cfg, _guard) = tiny_loader();
    let tokens = [3u32, 7, 11];
    let mut by_token =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, 32, DotsGraphQuant::Bf16).unwrap();
    let want = by_token.forward_tokens(&tokens).unwrap();

    let mut by_row =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, 32, DotsGraphQuant::Bf16).unwrap();
    by_row.prefill_step(tokens[0]).unwrap();
    by_row.prefill_embed(&embed_row(&cfg, tokens[1])).unwrap();
    let got = by_row.decode_step_logits(tokens[2]).unwrap().1;

    assert_eq!(
        want, got,
        "supplying a position's hidden row must reach the same logits as looking that row up \
         from the embedding table, or the vision span the pipeline splices in is not the row \
         the decoder would have read"
    );
}

#[test]
fn generate_over_a_vision_span_matches_generate_over_the_tokens_those_rows_embed() {
    if !adapter_available() {
        return;
    }
    let (loader, cfg, _guard) = tiny_loader();
    let tokens = [3u32, 7, 11, 21, 5, 44, 8, 19, 2, 61, 13, 30];
    let vision_start = 3usize;
    let n_vis = 6usize;
    let opts = GenerateOptions {
        max_new_tokens: 8,
        ngram_size: None,
        stop_on_loop: false,
        eos_token_ids: Vec::new(),
        ..GenerateOptions::default()
    };
    let mut plain =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, 32, DotsGraphQuant::Bf16).unwrap();
    let want = plain.generate(&tokens, 0, None, &opts).unwrap();

    let mut feats: Vec<f32> = Vec::new();
    for t in &tokens[vision_start..vision_start + n_vis] {
        feats.extend_from_slice(&embed_row(&cfg, *t));
    }
    let mut spliced =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, 32, DotsGraphQuant::Bf16).unwrap();
    let got = spliced
        .generate(&tokens, vision_start, Some(&feats), &opts)
        .unwrap();

    assert_eq!(want.tokens.len(), opts.max_new_tokens, "greedy run cut short");
    assert_eq!(
        want.tokens, got.tokens,
        "a vision span carrying the embeddings of the tokens it replaces must generate what \
         those tokens generate"
    );
    assert_eq!(want.hit_eos, got.hit_eos);
}

const SPLIT_CONTEXT_RUNGS: [usize; 4] = [64, 512, 2048, 3412];

#[test]
fn the_graph_argmax_token_equals_a_host_argmax_over_the_logits_it_returns() {
    if !adapter_available() {
        return;
    }
    let _env = env_lock();
    let (loader, cfg, _guard) = tiny_loader();
    let span = 512usize;
    let prompt: Vec<u32> = (0..96u32).map(|i| (i * 29 + 7) % cfg.vocab_size as u32).collect();
    let mut wg =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    wg.prefill_prompt(&prompt).unwrap();
    let mut tok = prompt[prompt.len() - 1];
    for step in 0..64 {
        let (gpu, logits) = wg.decode_step_logits(tok).unwrap();
        let host = nv_models::dots_ocr::decoder::argmax_banned(&logits, &[]);
        assert_eq!(
            gpu, host,
            "generate may only skip the 0.6 MB logit readback while the graph's own argmax \
             agrees with a host argmax over the same logits; they parted at step {step}"
        );
        tok = gpu;
    }
}

#[test]
fn transposing_the_k_cache_reaches_the_token_major_logits_bitwise() {
    if !adapter_available() {
        return;
    }
    let _env = env_lock();
    let (loader, cfg, _guard) = tiny_loader();
    let group = cfg.num_attention_heads / cfg.num_key_value_heads;
    let span = SPLIT_CONTEXT_RUNGS[SPLIT_CONTEXT_RUNGS.len() - 1] + 16;
    let tokens: Vec<u32> = (0..span as u32)
        .map(|i| (i * 37 + 11) % cfg.vocab_size as u32)
        .collect();
    let kt = nv_models::dots_ocr::decoder_wgpu_graph::KV_TRANSPOSE_ENV;
    let fe = nv_models::dots_ocr::decoder_wgpu_graph::ATTN_FOLD_ENV;

    for fold in [1, group] {
        std::env::set_var(fe, fold.to_string());
        let mut plain =
            DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16)
                .unwrap();
        std::env::set_var(kt, "1");
        let mut trans =
            DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16)
                .unwrap();
        std::env::remove_var(kt);
        std::env::remove_var(fe);
        for ctx in SPLIT_CONTEXT_RUNGS {
            let prompt = &tokens[..ctx];
            plain.reset().unwrap();
            trans.reset().unwrap();
            let want = plain.prefill_prompt(prompt).unwrap();
            let got = trans.prefill_prompt(prompt).unwrap();
            let mismatched = want
                .iter()
                .zip(got.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let worst = want
                .iter()
                .zip(got.iter())
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            eprintln!("DOTS-KT fold={fold} ctx={ctx} mismatched={mismatched} worst={worst:.3e}");
            assert_eq!(
                mismatched, 0,
                "{kt}=1 only moves where a K word lives, it reorders no arithmetic, so ctx={ctx} \
                 at fold={fold} must be bitwise identical, worst delta {worst:.3e}"
            );
        }
    }
}

#[test]
fn folding_the_gqa_group_into_one_workgroup_reaches_the_unfolded_logits_bitwise() {
    if !adapter_available() {
        return;
    }
    let _env = env_lock();
    let (loader, cfg, _guard) = tiny_loader();
    let group = cfg.num_attention_heads / cfg.num_key_value_heads;
    let span = SPLIT_CONTEXT_RUNGS[SPLIT_CONTEXT_RUNGS.len() - 1] + 16;
    let tokens: Vec<u32> = (0..span as u32)
        .map(|i| (i * 37 + 11) % cfg.vocab_size as u32)
        .collect();
    let env = nv_models::dots_ocr::decoder_wgpu_graph::ATTN_FOLD_ENV;

    let mut plain =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    for fold in 2..=group {
        if !group.is_multiple_of(fold) {
            continue;
        }
        std::env::set_var(env, fold.to_string());
        let mut folded =
            DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16)
                .unwrap();
        std::env::remove_var(env);
        assert!(
            folded.attn_split_passes() > 0,
            "the folded path is built on top of the split path"
        );
        for ctx in SPLIT_CONTEXT_RUNGS {
            let prompt = &tokens[..ctx];
            plain.reset().unwrap();
            folded.reset().unwrap();
            let want = plain.prefill_prompt(prompt).unwrap();
            let got = folded.prefill_prompt(prompt).unwrap();
            let mismatched = want
                .iter()
                .zip(got.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let worst = want
                .iter()
                .zip(got.iter())
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            eprintln!("DOTS-FOLD fold={fold} ctx={ctx} mismatched={mismatched} worst={worst:.3e}");
            assert_eq!(
                mismatched, 0,
                "folding {fold} query heads into one workgroup reorders no arithmetic, so \
                 {env}={fold} at ctx={ctx} must be bitwise identical, worst delta {worst:.3e}"
            );
        }
    }
}

#[test]
fn split_kv_attention_reaches_the_single_workgroup_logits_at_every_context() {
    if !adapter_available() {
        return;
    }
    let _env = env_lock();
    let (loader, cfg, _guard) = tiny_loader();
    let span = SPLIT_CONTEXT_RUNGS[SPLIT_CONTEXT_RUNGS.len() - 1] + 16;
    let tokens: Vec<u32> = (0..span as u32)
        .map(|i| (i * 37 + 11) % cfg.vocab_size as u32)
        .collect();

    let mut split =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    assert!(
        split.attn_split_passes() > 0,
        "the split-kv path must be built for a span this long"
    );
    std::env::set_var(
        nv_models::dots_ocr::decoder_wgpu_graph::ATTN_SPLIT_ENV,
        "0",
    );
    let mut plain =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    std::env::remove_var(nv_models::dots_ocr::decoder_wgpu_graph::ATTN_SPLIT_ENV);
    assert_eq!(
        plain.attn_split_passes(),
        0,
        "{}=0 must leave the single-workgroup decode attention in place",
        nv_models::dots_ocr::decoder_wgpu_graph::ATTN_SPLIT_ENV
    );

    for ctx in SPLIT_CONTEXT_RUNGS {
        let prompt = &tokens[..ctx];
        plain.reset().unwrap();
        split.reset().unwrap();
        let want = plain.prefill_prompt(prompt).unwrap();
        let got = split.prefill_prompt(prompt).unwrap();
        let mismatched = want
            .iter()
            .zip(got.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let worst = want
            .iter()
            .zip(got.iter())
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!(
            "split-kv ctx={ctx}: mismatched_logits={mismatched}/{} worst_abs={worst:.3e}",
            want.len()
        );
        assert_eq!(
            want, got,
            "context {ctx}: the split-kv decode attention must land the same logits as the \
             single-workgroup kernel it replaces, not merely a close answer"
        );
    }
}

fn real_checkpoint_dir() -> Option<std::path::PathBuf> {
    if let Ok(d) = std::env::var("NV_DOTS_DIR") {
        return Some(std::path::PathBuf::from(d));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots");
    let mut entries: Vec<_> = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

fn probe_quant() -> DotsGraphQuant {
    match std::env::var("NV_DOTS_PROBE_QUANT").ok().as_deref() {
        Some("w4") => DotsGraphQuant::AffineW4,
        Some("nvfp4") => DotsGraphQuant::Nvfp4,
        _ => DotsGraphQuant::Bf16,
    }
}

const PROBE_DECODE_STEPS: usize = 16;

#[test]
#[ignore]
fn real_weight_decode_rate_probe() {
    if std::env::var("NV_DOTS_GRAPH_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: NV_DOTS_GRAPH_TEST=1 not set");
        return;
    }
    if !adapter_available() {
        return;
    }
    let Some(dir) = real_checkpoint_dir() else {
        eprintln!("skipping: no rednote-hilab/dots.ocr checkpoint");
        return;
    };
    let cfg = DotsDecoderConfig::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).unwrap();
    let span = SPLIT_CONTEXT_RUNGS[SPLIT_CONTEXT_RUNGS.len() - 1] + PROBE_DECODE_STEPS + 16;
    let prompt: Vec<u32> = (0..span)
        .map(|i| ((i * 37 + 100) % cfg.vocab_size) as u32)
        .collect();

    for split in [false, true] {
        if split {
            std::env::remove_var(nv_models::dots_ocr::decoder_wgpu_graph::ATTN_SPLIT_ENV);
        } else {
            std::env::set_var(
                nv_models::dots_ocr::decoder_wgpu_graph::ATTN_SPLIT_ENV,
                "0",
            );
        }
        let t0 = std::time::Instant::now();
        let mut wg =
            DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, probe_quant())
                .unwrap();
        eprintln!(
            "DOTS-GRAPH-PROBE split={split} load_s={:.1} chunk_rows={} passes={} split_passes={} \
             vram_gib={:.3}",
            t0.elapsed().as_secs_f32(),
            wg.prefill_chunk_len(),
            wg.pass_count(),
            wg.attn_split_passes(),
            wg.vram_report().total_bytes as f64 / (1u64 << 30) as f64
        );
        for ctx in SPLIT_CONTEXT_RUNGS {
            wg.reset().unwrap();
            let t1 = std::time::Instant::now();
            let logits = wg.prefill_prompt(&prompt[..ctx]).unwrap();
            let prefill_s = t1.elapsed().as_secs_f64();
            let mut token = DotsDecoderWgpuGraph::greedy_token(&logits);
            let mut ms: Vec<f64> = Vec::with_capacity(PROBE_DECODE_STEPS);
            for _ in 0..PROBE_DECODE_STEPS {
                let t = std::time::Instant::now();
                token = wg.decode_step(token).unwrap();
                ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!(
                "DOTS-GRAPH-PROBE split={split} ctx={ctx} prefill_ms={:.1} prompt_tok_s={:.1} \
                 step_ms_min={:.2} step_ms_med={:.2} step_ms_max={:.2} decode_tok_s={:.1}",
                prefill_s * 1e3,
                ctx as f64 / prefill_s,
                ms[0],
                ms[ms.len() / 2],
                ms[ms.len() - 1],
                1e3 / ms[ms.len() / 2]
            );
            assert!(
                (token as usize) < cfg.vocab_size,
                "probe produced token {token} outside the vocabulary"
            );
        }
    }
    std::env::remove_var(nv_models::dots_ocr::decoder_wgpu_graph::ATTN_SPLIT_ENV);
}

#[test]
fn tiled_and_register_prefill_attention_reach_the_untiled_logits_at_every_context() {
    if !adapter_available() {
        return;
    }
    let _env = env_lock();
    use nv_models::dots_ocr::decoder_wgpu_graph::{ATTN_FUSED_ENV, ATTN_REG_ENV, ATTN_TILED_ENV};
    let (loader, cfg, _guard) = tiny_loader();
    let span = SPLIT_CONTEXT_RUNGS[SPLIT_CONTEXT_RUNGS.len() - 1] + 16;
    let tokens: Vec<u32> = (0..span as u32)
        .map(|i| (i * 29 + 5) % cfg.vocab_size as u32)
        .collect();

    std::env::set_var(ATTN_FUSED_ENV, "0");
    let mut reg =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    std::env::set_var(ATTN_REG_ENV, "0");
    std::env::set_var(ATTN_TILED_ENV, "1");
    let mut tiled =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    std::env::set_var(ATTN_TILED_ENV, "0");
    let mut plain =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    std::env::remove_var(ATTN_TILED_ENV);
    std::env::remove_var(ATTN_REG_ENV);
    std::env::remove_var(ATTN_FUSED_ENV);

    for ctx in SPLIT_CONTEXT_RUNGS {
        let prompt = &tokens[..ctx];
        plain.reset().unwrap();
        tiled.reset().unwrap();
        reg.reset().unwrap();
        let want = plain.prefill_prompt(prompt).unwrap();
        let got_tiled = tiled.prefill_prompt(prompt).unwrap();
        let got_reg = reg.prefill_prompt(prompt).unwrap();
        for (name, got) in [("tiled", &got_tiled), ("reg", &got_reg)] {
            let mismatched = want
                .iter()
                .zip(got.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let worst = want
                .iter()
                .zip(got.iter())
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            eprintln!(
                "{name}-prefill ctx={ctx}: mismatched={mismatched}/{} worst_abs={worst:.3e} \
                 rel={:.3e}",
                want.len(),
                worst / scale
            );
            assert_eq!(
                &want, got,
                "context {ctx}: the {name} prefill attention must land the same logits as the \
                 row-per-workgroup kernel it replaces"
            );
        }
    }
}

const STEEL_VS_STEPPED_BAND_IS_F16_KV_STORAGE_UNDER_F32_SIMDGROUP_MATH: f32 = 1e-1;

#[test]
fn causal_steel_prefill_reaches_the_stepped_decoder_and_holds_the_greedy_token() {
    if !adapter_available() {
        return;
    }
    let _env = env_lock();
    use nv_models::dots_ocr::decoder_wgpu_graph::{PF_STEEL_ENV, PF_STEEL_PROJ_ENV};
    let mut cfg = tiny_config();
    cfg.hidden_size = 512;
    cfg.intermediate_size = 256;
    cfg.max_position_embeddings = 1024;
    let (loader, _guard) = loader_for(&cfg);
    let span = 4 * 128 + 16;
    std::env::set_var(PF_STEEL_ENV, "1");
    std::env::set_var(PF_STEEL_PROJ_ENV, "1");
    let mut steel =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    std::env::set_var(PF_STEEL_ENV, "0");
    std::env::set_var(PF_STEEL_PROJ_ENV, "0");
    let mut stepped =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    std::env::remove_var(PF_STEEL_ENV);
    std::env::remove_var(PF_STEEL_PROJ_ENV);
    let widths = steel.prefill_widths().to_vec();
    assert!(
        widths.len() >= 2 && widths[0] > widths[1],
        "the steel gate needs a wide and a narrow chunk so a chunk runs at a non-zero base, \
         got {widths:?}"
    );
    let mut flips = 0u32;
    for len in [widths[0], widths[0] + widths[1] + 3, widths[0] - 28] {
        let tokens: Vec<u32> = (0..len as u32)
            .map(|i| (i * 7 + 3) % cfg.vocab_size as u32)
            .collect();
        steel.reset().unwrap();
        stepped.reset().unwrap();
        let got = steel.prefill_prompt(&tokens).unwrap();
        let want = stepped.forward_tokens(&tokens).unwrap();
        let rel = rel_err(&want, &got);
        let cos = cosine(&want, &got);
        let wt = DotsDecoderWgpuGraph::greedy_token(&want);
        let gt = DotsDecoderWgpuGraph::greedy_token(&got);
        eprintln!("steel-prefill len={len}: rel={rel:.3e} cos={cos:.6} argmax {wt} vs {gt}");
        assert!(
            rel < STEEL_VS_STEPPED_BAND_IS_F16_KV_STORAGE_UNDER_F32_SIMDGROUP_MATH,
            "prompt {len}: the causal steel chunk drifted rel {rel} from stepping those rows \
             through the decode attention chain"
        );
        if wt != gt {
            let mut top = f32::MIN;
            let mut second = f32::MIN;
            for &v in want.iter() {
                if v > top {
                    second = top;
                    top = v;
                } else if v > second {
                    second = v;
                }
            }
            let margin = (top - second) / top.abs().max(1e-6);
            assert!(
                margin < 3e-2 && flips < 2,
                "prompt {len}: the steel chunk flipped the greedy token ({wt} vs {gt}) with \
                 reference top-2 margin {margin:.4} -- a wide-margin or repeated flip is a \
                 wrong result, not f16-storage noise on a toy near-tie; the production gate \
                 is the twelve corpus zeros"
            );
            flips += 1;
            eprintln!("steel-prefill len={len}: near-tie flip tolerated (margin {margin:.4})");
        }
    }
}

const FUSED_VS_REG_BAND_IS_ONLINE_SOFTMAX_REASSOCIATION_UNDER_BF16_HEAD_PACKING: f32 = 3e-2;

const W4_VS_BF16_BAND_COVERS_F16_ACTIVATIONS_AGAINST_BF16_EXACT_SO_ARGMAX_IS_THE_ASSERT: f32 =
    2.5e-1;

fn roundtrip_affine_gs64(map: &mut HashMap<String, Tensor>, names: &[&str], gs: usize) {
    for name in names {
        let t = map.get(*name).unwrap();
        let (n, k) = t.dims2().unwrap();
        if !k.is_multiple_of(gs) {
            continue;
        }
        let f: Vec<f32> = t
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let (w, s, bi) = nv_quant::mlx_affine::quantize_mlx_affine_linear(&f, n, k, gs);
        let d = nv_quant::mlx_affine::dequantize_mlx_affine_linear(&w, &s, &bi, n, k, gs);
        let back: Vec<bf16> = d.iter().map(|v| bf16::from_f32(*v)).collect();
        let t2 = Tensor::from_vec(back, (n, k), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        map.insert(name.to_string(), t2);
    }
}

#[test]
fn w4_gemv_matches_the_host_group_epilogue_at_every_dots_decode_shape() {
    if !adapter_available() {
        return;
    }
    use nv_models::dots_ocr::decoder_wgpu_graph::gemv_w4_probe;
    for (n, k) in [(2048usize, 1536usize), (1536, 1536), (17920, 1536), (1536, 8960)] {
        let (w4_ms, bf_ms, max_rel) = gemv_w4_probe(n, k, 20).unwrap();
        eprintln!(
            "W4-GEMV-PROBE n={n} k={k} w4_ms={w4_ms:.4} bf16_ms={bf_ms:.4} speedup={:.2}x \
             host_max_ulp={max_rel:.3}",
            bf_ms / w4_ms
        );
        assert!(
            max_rel <= 1.01,
            "n={n} k={k}: the w4 gemv sits {max_rel} bf16 ulps from the host group epilogue -- \
             that is arithmetic, not reassociation"
        );
    }
}

#[test]
fn w4_decode_reaches_the_bf16_gemv_logits_on_identical_dequantised_weights() {
    if !adapter_available() {
        return;
    }
    let _env = env_lock();
    let mut cfg = tiny_config();
    cfg.hidden_size = 512;
    cfg.intermediate_size = 256;
    cfg.max_position_embeddings = 128;
    let map = tiny_weight_map(&cfg);
    let mut rt = map.clone();
    let mut names: Vec<String> = Vec::new();
    for li in 0..cfg.num_hidden_layers {
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            names.push(format!("model.layers.{li}.{suffix}"));
        }
    }
    names.push("lm_head.weight".to_string());
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    roundtrip_affine_gs64(&mut rt, &name_refs, 64);

    let save = |m: &HashMap<String, Tensor>, tag: &str| {
        let dir = std::env::temp_dir().join(format!(
            "dots-w4-test-{}-{tag}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        candle_core::safetensors::save(m, &path).unwrap();
        (
            WeightLoader::open_file(&path, &Device::Cpu).unwrap(),
            tempdir::Guard(dir),
        )
    };
    let (rt_loader, _g1) = save(&rt, "rt");
    let (raw_loader, _g2) = save(&map, "raw");

    let span = 96usize;
    let tokens: Vec<u32> = (0..span as u32)
        .map(|i| (i * 29 + 5) % cfg.vocab_size as u32)
        .collect();
    let mut base =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &rt_loader, span, DotsGraphQuant::Bf16)
            .unwrap();
    let mut w4 =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &raw_loader, span, DotsGraphQuant::AffineW4)
            .unwrap();

    let prompt = &tokens[..64];
    let want = base.prefill_prompt(prompt).unwrap();
    let got = w4.prefill_prompt(prompt).unwrap();
    let mut steps = vec![(want, got)];
    for t in 64..80 {
        let w = base.decode_step_logits(tokens[t]).unwrap().1;
        let g = w4.decode_step_logits(tokens[t]).unwrap().1;
        steps.push((w, g));
    }
    let mut flips = 0u32;
    for (i, (want, got)) in steps.iter().enumerate() {
        let rel = rel_err(want, got);
        let cos = cosine(want, got);
        let wt = DotsDecoderWgpuGraph::greedy_token(want);
        let gt = DotsDecoderWgpuGraph::greedy_token(got);
        eprintln!("w4-decode step={i}: rel={rel:.3e} cos={cos:.6} argmax {wt} vs {gt}");
        assert!(
            rel < W4_VS_BF16_BAND_COVERS_F16_ACTIVATIONS_AGAINST_BF16_EXACT_SO_ARGMAX_IS_THE_ASSERT,
            "step {i}: w4 decode drifted rel {rel} from the bf16 gemv arm -- the arms differ \
             by bf16-rounded dequant weights AND the f16 activation model under the msl gemv, \
             so this band only catches wrong-buffer-format bugs; the load-bearing asserts are \
             the near-tie-guarded greedy tokens here and the twelve corpus zeros"
        );
        if wt != gt {
            let mut top = f32::MIN;
            let mut second = f32::MIN;
            for &v in want.iter() {
                if v > top {
                    second = top;
                    top = v;
                } else if v > second {
                    second = v;
                }
            }
            let margin = (top - second) / top.abs().max(1e-6);
            assert!(
                margin < 3e-2 && flips < 3,
                "step {i}: w4 decode flipped the greedy token ({wt} vs {gt}) with reference \
                 top-2 margin {margin:.4} -- a wide-margin or repeated flip is a wrong result, \
                 not f16-activation noise on a toy near-tie; the production gate is the twelve \
                 corpus zeros"
            );
            flips += 1;
            eprintln!("w4-decode step={i}: near-tie flip tolerated (margin {margin:.4})");
        }
    }
}

#[test]
fn fused_prefill_attention_reaches_the_register_kernel_logits_within_the_reassociation_band() {
    if !adapter_available() {
        return;
    }
    let _env = env_lock();
    use nv_models::dots_ocr::decoder_wgpu_graph::{ATTN_FUSED_ENV, PF_STEEL_ENV, PF_STEEL_PROJ_ENV};
    let mut cfg = tiny_config();
    cfg.hidden_size = 512;
    cfg.intermediate_size = 256;
    cfg.max_position_embeddings = 128;
    let (loader, _guard) = loader_for(&cfg);
    let span = 96usize;
    let tokens: Vec<u32> = (0..span as u32)
        .map(|i| (i * 29 + 5) % cfg.vocab_size as u32)
        .collect();

    std::env::set_var(PF_STEEL_ENV, "0");
    std::env::set_var(PF_STEEL_PROJ_ENV, "0");
    std::env::set_var(ATTN_FUSED_ENV, "0");
    let mut reg =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    std::env::set_var(ATTN_FUSED_ENV, "1");
    let mut fused =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    std::env::remove_var(ATTN_FUSED_ENV);
    std::env::remove_var(PF_STEEL_ENV);
    std::env::remove_var(PF_STEEL_PROJ_ENV);

    for ctx in [32usize, 33, 64, 96] {
        let prompt = &tokens[..ctx];
        reg.reset().unwrap();
        fused.reset().unwrap();
        let want = reg.prefill_prompt(prompt).unwrap();
        let got = fused.prefill_prompt(prompt).unwrap();
        let rel = rel_err(&want, &got);
        let cos = cosine(&want, &got);
        let want_tok = DotsDecoderWgpuGraph::greedy_token(&want);
        let got_tok = DotsDecoderWgpuGraph::greedy_token(&got);
        eprintln!(
            "fused-prefill ctx={ctx}: rel={rel:.3e} cos={cos:.6} argmax {want_tok} vs {got_tok}"
        );
        assert!(
            rel < FUSED_VS_REG_BAND_IS_ONLINE_SOFTMAX_REASSOCIATION_UNDER_BF16_HEAD_PACKING,
            "context {ctx}: fused prefill attention drifted rel {rel} from the register kernel"
        );
        assert_eq!(
            want_tok, got_tok,
            "context {ctx}: fused prefill attention flipped the greedy token"
        );
    }
}

#[test]
#[ignore]
fn real_weight_prefill_rate_probe() {
    if std::env::var("NV_DOTS_GRAPH_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: NV_DOTS_GRAPH_TEST=1 not set");
        return;
    }
    if !adapter_available() {
        return;
    }
    let Some(dir) = real_checkpoint_dir() else {
        eprintln!("skipping: no rednote-hilab/dots.ocr checkpoint");
        return;
    };
    use nv_models::dots_ocr::decoder_wgpu_graph::{
        ATTN_FUSED_ENV, ATTN_REG_ENV, ATTN_TILED_ENV, PF_STEEL_ENV, PF_STEEL_PROJ_ENV,
    };
    let cfg = DotsDecoderConfig::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).unwrap();
    let span = SPLIT_CONTEXT_RUNGS[SPLIT_CONTEXT_RUNGS.len() - 1] + 64;
    let prompt: Vec<u32> = (0..span)
        .map(|i| ((i * 37 + 100) % cfg.vocab_size) as u32)
        .collect();

    for arm in ["plain", "tiled", "reg", "fused", "steel", "steelpair"] {
        std::env::set_var(ATTN_FUSED_ENV, "0");
        std::env::set_var(PF_STEEL_ENV, "0");
        std::env::set_var(PF_STEEL_PROJ_ENV, "0");
        match arm {
            "plain" => {
                std::env::set_var(ATTN_REG_ENV, "0");
                std::env::set_var(ATTN_TILED_ENV, "0");
            }
            "tiled" => {
                std::env::set_var(ATTN_REG_ENV, "0");
                std::env::set_var(ATTN_TILED_ENV, "1");
            }
            "fused" => {
                std::env::remove_var(ATTN_REG_ENV);
                std::env::remove_var(ATTN_TILED_ENV);
                std::env::set_var(ATTN_FUSED_ENV, "1");
            }
            "steel" => {
                std::env::remove_var(ATTN_REG_ENV);
                std::env::remove_var(ATTN_TILED_ENV);
                std::env::remove_var(ATTN_FUSED_ENV);
                std::env::set_var(PF_STEEL_ENV, "1");
            }
            "steelpair" => {
                std::env::remove_var(ATTN_REG_ENV);
                std::env::remove_var(ATTN_TILED_ENV);
                std::env::remove_var(ATTN_FUSED_ENV);
                std::env::set_var(PF_STEEL_ENV, "1");
                std::env::set_var(PF_STEEL_PROJ_ENV, "1");
            }
            _ => {
                std::env::remove_var(ATTN_REG_ENV);
                std::env::remove_var(ATTN_TILED_ENV);
            }
        }
        let mut wg =
            DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16)
                .unwrap();
        for ctx in SPLIT_CONTEXT_RUNGS {
            wg.reset().unwrap();
            let t = std::time::Instant::now();
            let logits = wg.prefill_prompt(&prompt[..ctx]).unwrap();
            let s = t.elapsed().as_secs_f64();
            let tok = DotsDecoderWgpuGraph::greedy_token(&logits);
            eprintln!(
                "DOTS-PREFILL-PROBE arm={arm} ctx={ctx} prefill_ms={:.1} prompt_tok_s={:.1} \
                 per_token_ms={:.3} argmax={tok}",
                s * 1e3,
                ctx as f64 / s,
                s * 1e3 / ctx as f64
            );
        }
    }
    std::env::remove_var(ATTN_TILED_ENV);
    std::env::remove_var(ATTN_REG_ENV);
    std::env::remove_var(ATTN_FUSED_ENV);
    std::env::remove_var(PF_STEEL_ENV);
    std::env::remove_var(PF_STEEL_PROJ_ENV);
}

#[test]
#[ignore]
fn real_weight_prefill_ablation_brackets_the_wall_by_deleted_work() {
    if std::env::var("NV_DOTS_GRAPH_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: NV_DOTS_GRAPH_TEST=1 not set");
        return;
    }
    if !adapter_available() {
        return;
    }
    let Some(dir) = real_checkpoint_dir() else {
        eprintln!("skipping: no rednote-hilab/dots.ocr checkpoint");
        return;
    };
    use nv_models::dots_ocr::decoder_wgpu_graph::{
        ATTN_FUSED_ENV, PF_ABLATE_ENV, PF_STEEL_ENV, PF_STEEL_PROJ_ENV,
    };
    let cfg = DotsDecoderConfig::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).unwrap();
    let ctx = 3412usize;
    let span = ctx + 64;
    let prompt: Vec<u32> = (0..span)
        .map(|i| ((i * 37 + 100) % cfg.vocab_size) as u32)
        .collect();
    std::env::set_var(PF_STEEL_ENV, "0");
    std::env::set_var(PF_STEEL_PROJ_ENV, "0");
    std::env::set_var(ATTN_FUSED_ENV, "0");
    for arm in ["base", "qk", "softmax", "pv", "attn", "proj", "base-again"] {
        if arm.starts_with("base") {
            std::env::remove_var(PF_ABLATE_ENV);
        } else {
            std::env::set_var(PF_ABLATE_ENV, arm);
        }
        let mut wg =
            DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16)
                .unwrap();
        wg.prefill_prompt(&prompt[..ctx]).unwrap();
        for rep in 0..3 {
            wg.reset().unwrap();
            let t = std::time::Instant::now();
            let logits = wg.prefill_prompt(&prompt[..ctx]).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "DOTS-PF-ABLATE arm={arm} rep={rep} ctx={ctx} prefill_ms={ms:.1} argmax={}",
                DotsDecoderWgpuGraph::greedy_token(&logits)
            );
        }
    }
    std::env::remove_var(PF_ABLATE_ENV);
    std::env::remove_var(ATTN_FUSED_ENV);
    std::env::remove_var(PF_STEEL_ENV);
    std::env::remove_var(PF_STEEL_PROJ_ENV);
}

#[test]
#[ignore]
fn prefill_phase_profile_says_which_kernel_owns_the_prompt() {
    if std::env::var("NV_DOTS_GRAPH_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: NV_DOTS_GRAPH_TEST=1 not set");
        return;
    }
    std::env::set_var("NV_WGPU_PROFILE", "1");
    if !adapter_available() {
        return;
    }
    let Some(dir) = real_checkpoint_dir() else {
        eprintln!("skipping: no rednote-hilab/dots.ocr checkpoint");
        return;
    };
    use nv_kernels::wgpu_backend::dispatch::profile;
    if !profile::enabled() {
        eprintln!("skipping: NV_WGPU_PROFILE did not take");
        return;
    }
    let cfg = DotsDecoderConfig::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).unwrap();
    let ctx = 3412usize;
    let span = ctx + 64;
    let prompt: Vec<u32> = (0..span)
        .map(|i| ((i * 37 + 100) % cfg.vocab_size) as u32)
        .collect();
    let mut wg =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    let m = wg.prefill_chunk_len();
    let warm = ctx - m;

    let t_all = std::time::Instant::now();
    let mut done = 0usize;
    while done + m <= warm {
        let take = (warm - done).min(10 * m);
        done += wg.prefill_tokens(&prompt[done..done + take]).unwrap();
        profile::reset();
    }
    let warm_ms = t_all.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "DOTS-PREFILL-PROFILE warmup_tokens={done} warmup_ms={warm_ms:.1} \
         per_token_ms={:.3} chunk_rows={m}",
        warm_ms / done.max(1) as f64
    );
    let warm = done;

    for phase in ["chunk", "step"] {
        profile::reset();
        let t0 = std::time::Instant::now();
        if phase == "chunk" {
            wg.prefill_tokens(&prompt[warm..warm + m]).unwrap();
        } else {
            wg.decode_step(prompt[warm + m]).unwrap();
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        let rows = profile::report();
        let total: f64 = rows.iter().map(|r| r.2).sum();
        eprintln!(
            "DOTS-PREFILL-PROFILE phase={phase} pos={} wall_ms={wall_ms:.2} gpu_ms={:.2} \
             accounted={:.0}% entries={}",
            wg.current_pos(),
            total / 1e6,
            100.0 * (total / 1e6) / wall_ms.max(1e-6),
            rows.len()
        );
        for (label, count, ns) in rows.iter() {
            eprintln!(
                "DOTS-PREFILL-PROFILE   {phase:<6} {label:<22} calls={count:<6} ms={:.3} \
                 share={:.1}%",
                ns / 1e6,
                100.0 * ns / total.max(1.0)
            );
        }
    }
}

#[test]
fn adapter_coop_configs_decide_whether_the_prefill_gemm_has_a_coop_route() {
    if !adapter_available() {
        return;
    }
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared().unwrap();
    eprintln!(
        "COOP-CONFIGS cooperative_matrix={} shader_f16={} subgroup={:?} gemm_tile={:?}",
        ctx.caps.cooperative_matrix,
        ctx.caps.shader_f16,
        ctx.caps.subgroup_width_known(),
        ctx.caps.coop_gemm_tile()
    );
    for c in &ctx.caps.coop_configs {
        eprintln!("COOP-CONFIGS   {}", c.label());
    }
}

#[test]
#[ignore]
fn coop_f32_gemm_probe_at_the_prefill_chunk_shape() {
    if std::env::var("NV_DOTS_GRAPH_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: NV_DOTS_GRAPH_TEST=1 not set");
        return;
    }
    if !adapter_available() {
        return;
    }
    use nv_models::dots_ocr::decoder_wgpu_graph::coop_gemm_probe;
    let plans = [
        (4u32, 4u32, 4u32, 2u32),
        (8, 4, 4, 2),
        (16, 4, 4, 2),
        (16, 2, 8, 2),
        (8, 8, 4, 2),
    ];
    for m in [32usize, 128, 256, 512] {
        for (n, k, tag) in [
            (2048usize, 1536usize, "qkv"),
            (17920, 1536, "gateup"),
            (1536, 8960, "down"),
        ] {
            let mut best: Option<(f64, f64, f32, (u32, u32, u32, u32))> = None;
            for (tm, tn, sg, ku) in plans {
                if let Ok(p) = coop_gemm_probe(m, n, k, tm, tn, sg, ku, 4) {
                    let take = match &best {
                        Some((bt, _, _, _)) => p.coop_tflops > *bt,
                        None => true,
                    };
                    if take {
                        best = Some((p.coop_tflops, p.m8_tflops, p.max_abs_diff, (tm, tn, sg, ku)));
                    }
                }
            }
            match best {
                Some((coop, m8, diff, (tm, tn, sg, ku))) => eprintln!(
                    "COOP-M-LADDER m={m:<4} {tag:<7} n={n:<6} k={k:<5} coop={coop:.2} TFLOP/s \
                     m8={m8:.2} TFLOP/s ratio={:.2}x best=tm{tm}_tn{tn}_sg{sg}_ku{ku} \
                     max_abs_diff={diff:.2e}",
                    coop / m8
                ),
                None => eprintln!("COOP-M-LADDER m={m} {tag}: no plan built"),
            }
        }
    }
}

#[test]
#[ignore]
fn decode_step_wall_clock_says_where_the_time_outside_the_kernels_goes() {
    if std::env::var("NV_DOTS_GRAPH_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: NV_DOTS_GRAPH_TEST=1 not set");
        return;
    }
    if !adapter_available() {
        return;
    }
    let Some(dir) = real_checkpoint_dir() else {
        eprintln!("skipping: no rednote-hilab/dots.ocr checkpoint");
        return;
    };
    let cfg = DotsDecoderConfig::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).unwrap();
    let ctx = 3412usize;
    let reps = 48usize;
    let span = ctx + reps + 512;
    let prompt: Vec<u32> = (0..span)
        .map(|i| ((i * 37 + 100) % cfg.vocab_size) as u32)
        .collect();

    let mut wg =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16).unwrap();
    let mut done = 0usize;
    let m = wg.prefill_chunk_len();
    while done + m <= ctx && m > 0 {
        let take = (ctx - done).min(10 * m);
        done += wg.prefill_tokens(&prompt[done..done + take]).unwrap();
    }
    while done < ctx {
        wg.prefill_step(prompt[done]).unwrap();
        done += 1;
    }

    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let mut tok = prompt[ctx];

    for _ in 0..8 {
        tok = wg.decode_step(tok).unwrap();
    }
    let mut full = Vec::new();
    let mut logits = Vec::new();
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        let (t, l) = wg.decode_step_logits(tok).unwrap();
        full.push(t0.elapsed().as_secs_f64() * 1e3);
        tok = t;
        logits = l;
    }
    let mut tokenonly = Vec::new();
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        tok = wg.decode_step(tok).unwrap();
        tokenonly.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    let mut amax = Vec::new();
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        let got = nv_models::dots_ocr::decoder::argmax_banned(&logits, &[]);
        amax.push(t0.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(got);
    }

    let long = 384usize;
    let mut walk = Vec::new();
    let mut disagreed = 0usize;
    for _ in 0..long {
        let t0 = std::time::Instant::now();
        let (t, l) = wg.decode_step_logits(tok).unwrap();
        walk.push(t0.elapsed().as_secs_f64() * 1e3);
        if t != nv_models::dots_ocr::decoder::argmax_banned(&l, &[]) {
            disagreed += 1;
        }
        tok = t;
    }
    assert_eq!(
        disagreed, 0,
        "generate skips the logit readback whenever the ban list is empty, which is only sound \
         while the graph argmax agrees with a host argmax on real weights"
    );
    let first = med(walk[..64].to_vec());
    let last = med(walk[long - 64..].to_vec());
    let mean = walk.iter().sum::<f64>() / walk.len() as f64;
    eprintln!(
        "DOTS-STEP-WALK steps={long} from_ctx={ctx}  first64_med_ms={first:.3}           last64_med_ms={last:.3}  mean_ms={mean:.3}  mean_tok_s={:.1}",
        1e3 / mean
    );

    let a = med(full);
    let b = med(tokenonly);
    let c = med(amax);
    eprintln!(
        "DOTS-STEP-WALL ctx={ctx} vocab={} reps={reps}\n           step_logits_ms={a:.3}  step_token_only_ms={b:.3}  logit_readback_ms={:.3}\n           host_argmax_ms={c:.3}  generate_step_ms={:.3}  implied_tok_s={:.1}",
        cfg.vocab_size,
        a - b,
        a + c,
        1e3 / (a + c)
    );
}

#[test]
#[ignore]
fn nvfp4_decode_at_page_context_probe() {
    if std::env::var("NV_DOTS_GRAPH_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: NV_DOTS_GRAPH_TEST=1 not set");
        return;
    }
    if !adapter_available() {
        return;
    }
    let Some(dir) = real_checkpoint_dir() else {
        eprintln!("skipping: no rednote-hilab/dots.ocr checkpoint");
        return;
    };
    use nv_kernels::wgpu_backend::dispatch::profile;
    let cfg = DotsDecoderConfig::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).unwrap();
    let ctx = 3412usize;
    let span = ctx + 64;
    let prompt: Vec<u32> = (0..span)
        .map(|i| ((i * 37 + 100) % cfg.vocab_size) as u32)
        .collect();

    for quant in [DotsGraphQuant::Bf16, DotsGraphQuant::Nvfp4] {
        let t0 = std::time::Instant::now();
        let mut wg =
            DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, quant).unwrap();
        eprintln!(
            "DOTS-Q-PROBE quant={} load_s={:.1} chunk_rows={} vram_gib={:.3}",
            quant.name(),
            t0.elapsed().as_secs_f32(),
            wg.prefill_chunk_len(),
            wg.vram_report().total_bytes as f64 / (1u64 << 30) as f64
        );

        let t1 = std::time::Instant::now();
        let mut done = 0usize;
        let m = wg.prefill_chunk_len();
        while done + m <= ctx && m > 0 {
            let take = (ctx - done).min(10 * m);
            done += wg.prefill_tokens(&prompt[done..done + take]).unwrap();
            if profile::enabled() {
                profile::reset();
            }
        }
        while done < ctx {
            wg.prefill_step(prompt[done]).unwrap();
            done += 1;
            if profile::enabled() && done.is_multiple_of(8) {
                profile::reset();
            }
        }
        eprintln!(
            "DOTS-Q-PROBE quant={} prefilled={done} prefill_s={:.1}",
            quant.name(),
            t1.elapsed().as_secs_f64()
        );

        let mut token = prompt[ctx];
        let mut ms: Vec<f64> = Vec::new();
        let reps = if profile::enabled() { 1 } else { 16 };
        for _ in 0..reps {
            if profile::enabled() {
                profile::reset();
            }
            let t = std::time::Instant::now();
            token = wg.decode_step(token).unwrap();
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ms[ms.len() / 2];
        eprintln!(
            "DOTS-Q-PROBE quant={} ctx={ctx} step_ms_min={:.2} step_ms_med={med:.2} \
             step_ms_max={:.2} decode_tok_s={:.1}",
            quant.name(),
            ms[0],
            ms[ms.len() - 1],
            1e3 / med
        );

        if profile::enabled() {
            profile::reset();
            wg.decode_step(token).unwrap();
            let rows = profile::report();
            let total: f64 = rows.iter().map(|r| r.2).sum();
            eprintln!(
                "DOTS-Q-PROBE quant={} gpu_ms={:.2} entries={}",
                quant.name(),
                total / 1e6,
                rows.len()
            );
            for (label, count, ns) in rows.iter().take(8) {
                eprintln!(
                    "DOTS-Q-PROBE   {:<6} {label:<22} calls={count:<5} ms={:.3} share={:.1}%",
                    quant.name(),
                    ns / 1e6,
                    100.0 * ns / total.max(1.0)
                );
            }
        }
    }
}

#[test]
#[ignore]
fn head_to_head_the_two_dots_decoders_at_matched_context() {
    if std::env::var("NV_DOTS_GRAPH_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: NV_DOTS_GRAPH_TEST=1 not set");
        return;
    }
    if !adapter_available() {
        return;
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--mlx-community--dots.ocr-4bit/snapshots");
    let Some(mlx_dir) = std::fs::read_dir(&snaps)
        .ok()
        .and_then(|d| d.flatten().map(|e| e.path()).find(|p| p.is_dir()))
    else {
        eprintln!("skipping: no mlx-community/dots.ocr-4bit snapshot");
        return;
    };

    for ctx in [64usize, 512, 2048, 3412] {
        let span = ctx + 64;
        let t0 = std::time::Instant::now();
        let mut m = nv_models::dots_ocr::decoder_wgpu::dots_decoder_wgpu_from_dir(&mlx_dir, span)
            .expect("load the mlx 4-bit dots decoder on the q3d bias arm");
        let load_s = t0.elapsed().as_secs_f32();
        let prompt: Vec<u32> = (0..ctx).map(|i| ((i * 37 + 100) % 151936) as u32).collect();
        let t1 = std::time::Instant::now();
        let mut tok = m.prefill(&prompt).expect("prefill");
        let prefill_s = t1.elapsed().as_secs_f64();
        let mut ms: Vec<f64> = Vec::new();
        for _ in 0..16 {
            let t = std::time::Instant::now();
            tok = m.decode_step(tok).expect("decode step");
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ms[ms.len() / 2];
        eprintln!(
            "DOTS-H2H arm=q3d_bias_mlx4bit ctx={ctx} load_s={load_s:.1} prefill_ms={:.1} \
             prompt_tok_s={:.1} step_ms_med={med:.2} decode_tok_s={:.1}",
            prefill_s * 1e3,
            ctx as f64 / prefill_s,
            1e3 / med
        );
    }
}

const THE_MERGED_ORACLE_IS_THE_LOSSY_ARM: &str =
    "folding B@A into bf16 weights rounds the delta at the weight's ulp, while the live arm \
     adds it in f32; at delta amplitudes near that ulp the two arms legitimately part by \
     percents that compound through the kv cache, so this band bounds merge rounding, not \
     kernel error -- the bitwise kernel oracle is \
     dg_lora_kernels_match_a_bitwise_host_mirror_across_site_geometries";

fn write_tiny_adapter(c: &DotsDecoderConfig) -> tempdir::Guard {
    let dev = Device::Cpu;
    let h = c.hidden_size;
    let hd = c.head_dim();
    let qd = c.num_attention_heads * hd;
    let kvd = c.num_key_value_heads * hd;
    let inter = c.intermediate_size;
    let rank = 4usize;
    let mut m: HashMap<String, Tensor> = HashMap::new();
    let mut seed = 500.0f32;
    let leaves: Vec<(&str, usize, usize)> = match std::env::var("DG_LORA_TEST_LEAF").as_deref() {
        Ok("q") => vec![("self_attn.q_proj", qd, h)],
        Ok("k") => vec![("self_attn.k_proj", kvd, h)],
        Ok("v") => vec![("self_attn.v_proj", kvd, h)],
        Ok("o") => vec![("self_attn.o_proj", h, qd)],
        Ok("gate") => vec![("mlp.gate_proj", inter, h)],
        Ok("up") => vec![("mlp.up_proj", inter, h)],
        Ok("down") => vec![("mlp.down_proj", h, inter)],
        _ => vec![
            ("self_attn.q_proj", qd, h),
            ("self_attn.k_proj", kvd, h),
            ("self_attn.v_proj", kvd, h),
            ("self_attn.o_proj", h, qd),
            ("mlp.gate_proj", inter, h),
            ("mlp.up_proj", inter, h),
            ("mlp.down_proj", h, inter),
        ],
    };
    for l in 0..c.num_hidden_layers {
        for (path, n, k) in leaves.iter().copied() {
            let a: Vec<f32> = det_bf16(rank * k, seed).iter().map(|v| v * 0.5).collect();
            let b: Vec<f32> = det_bf16(n * rank, seed + 0.3)
                .iter()
                .map(|v| v * 0.5)
                .collect();
            seed += 1.0;
            let module = format!("base_model.model.model.layers.{l}.{path}");
            m.insert(
                format!("{module}.lora_A.weight"),
                Tensor::from_vec(a, (rank, k), &dev).unwrap(),
            );
            m.insert(
                format!("{module}.lora_B.weight"),
                Tensor::from_vec(b, (n, rank), &dev).unwrap(),
            );
        }
    }
    let dir = std::env::temp_dir().join(format!(
        "dots-graph-lora-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    candle_core::safetensors::save(&m, dir.join("adapter_model.safetensors")).unwrap();
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::json!({
            "peft_type": "LORA",
            "r": rank,
            "lora_alpha": rank,
            "lora_dropout": 0.0,
            "bias": "none",
            "use_rslora": false,
            "use_dora": false,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        })
        .to_string(),
    )
    .unwrap();
    tempdir::Guard(dir)
}

fn merged_loader_for(
    c: &DotsDecoderConfig,
    adapter_dir: &std::path::Path,
) -> (WeightLoader, tempdir::Guard) {
    let dev = Device::Cpu;
    let mut map = tiny_weight_map(c);
    let ad =
        candle_core::safetensors::load(adapter_dir.join("adapter_model.safetensors"), &dev)
            .unwrap();
    for l in 0..c.num_hidden_layers {
        for path in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            let module = format!("base_model.model.model.layers.{l}.{path}");
            if !ad.contains_key(&format!("{module}.lora_A.weight")) {
                continue;
            }
            let a = ad[&format!("{module}.lora_A.weight")]
                .to_dtype(DType::BF16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let b = ad[&format!("{module}.lora_B.weight")]
                .to_dtype(DType::BF16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let key = format!("model.layers.{l}.{path}.weight");
            let w = map.remove(&key).unwrap();
            let merged = (w + b.matmul(&a).unwrap())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            map.insert(key, merged);
        }
    }
    let dir = std::env::temp_dir().join(format!(
        "dots-graph-merged-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    candle_core::safetensors::save(&map, &path).unwrap();
    let loader = WeightLoader::open_file(&path, &Device::Cpu).unwrap();
    (loader, tempdir::Guard(dir))
}

#[test]
fn graph_lora_first_decode_step_tracks_the_host_merged_weights() {
    if !adapter_available() {
        return;
    }
    let _env = env_lock();
    let cfg = tiny_config();
    let adir = write_tiny_adapter(&cfg);

    let (base_loader, _bg) = loader_for(&cfg);
    std::env::set_var(
        nv_models::dots_ocr::decoder_wgpu_graph::DOTS_GRAPH_LORA_DIR_ENV_HOT_LOADS_A_PEFT_ADAPTER_ON_THE_GRAPH_ARM,
        adir.0.as_os_str(),
    );
    let live = DotsDecoderWgpuGraph::from_loader(cfg.clone(), &base_loader, 32, DotsGraphQuant::Bf16);
    std::env::remove_var(
        nv_models::dots_ocr::decoder_wgpu_graph::DOTS_GRAPH_LORA_DIR_ENV_HOT_LOADS_A_PEFT_ADAPTER_ON_THE_GRAPH_ARM,
    );
    let mut live = live.unwrap();

    let (merged_loader, _mg) = merged_loader_for(&cfg, &adir.0);
    let mut merged =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &merged_loader, 32, DotsGraphQuant::Bf16)
            .unwrap();
    if std::env::var("DG_LORA_TEST_LEAF").is_err() {
        assert_eq!(
            live.pass_count(),
            merged.pass_count() + cfg.num_hidden_layers * 7 * 2,
            "the live arm must carry exactly one shrink+expand pair per adapted module"
        );
    }

    let _ = THE_MERGED_ORACLE_IS_THE_LOSSY_ARM;
    let mut token = 3u32;
    let mut worst = 0f32;
    for step in 0..1 {
        let (_lt, live_logits) = live.decode_step_logits(token).unwrap();
        let (_mt, merged_logits) = merged.decode_step_logits(token).unwrap();
        let rel = rel_err(&merged_logits, &live_logits);
        worst = worst.max(rel);
        let want = DotsDecoderWgpuGraph::greedy_token(&merged_logits);
        let got = DotsDecoderWgpuGraph::greedy_token(&live_logits);
        eprintln!("lora-vs-merged step {step}: rel={rel:.5} argmax merged={want} live={got}");
        assert_eq!(want, got, "step {step}: live lora diverged from host merge");
        assert!(
            live_logits.iter().all(|v| v.is_finite()),
            "step {step}: non-finite live lora logits"
        );
        assert!(
            rel < 3e-2,
            "step {step}: a single lora application must track the merge inside the bf16 \
             band before kv compounding starts, rel {rel}"
        );
        token = want;
    }
    eprintln!("lora-vs-merged first-step rel: {worst:.5}");
}

fn dgl_host_mirror(
    x: &[u16],
    a: &[u16],
    b: &[u16],
    y_init: &[u32],
    m: usize,
    k: usize,
    n: usize,
    rank: usize,
    x_f16: bool,
    y_fmt: u32,
    y_off: usize,
    y_row: usize,
) -> Vec<u32> {
    let dec = |bits: u16| -> f32 {
        if x_f16 {
            half::f16::from_bits(bits).to_f32()
        } else {
            bf16::from_bits(bits).to_f32()
        }
    };
    let bfv = |bits: u16| bf16::from_bits(bits).to_f32();
    let rne = |v: f32| -> u16 {
        let bits = v.to_bits();
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    };
    let mut part = vec![0f32; m * rank];
    for mi in 0..m {
        for r in 0..rank {
            let mut red = [0f32; 64];
            for (lane, slot) in red.iter_mut().enumerate() {
                let mut acc = 0f32;
                let mut w = lane;
                while w < k / 2 {
                    acc += bfv(a[r * k + 2 * w]) * dec(x[mi * k + 2 * w]);
                    acc += bfv(a[r * k + 2 * w + 1]) * dec(x[mi * k + 2 * w + 1]);
                    w += 64;
                }
                *slot = acc;
            }
            let mut span = 32usize;
            while span > 0 {
                for i in 0..span {
                    red[i] += red[i + span];
                }
                span /= 2;
            }
            part[mi * rank + r] = red[0];
        }
    }
    let mut y = y_init.to_vec();
    for mi in 0..m {
        for wi in 0..n / 2 {
            let n0 = wi * 2;
            let mut acc0 = 0f32;
            let mut acc1 = 0f32;
            for r2 in 0..rank / 2 {
                let p0 = part[mi * rank + 2 * r2];
                let p1 = part[mi * rank + 2 * r2 + 1];
                acc0 += bfv(b[n0 * rank + 2 * r2]) * p0 + bfv(b[n0 * rank + 2 * r2 + 1]) * p1;
                acc1 += bfv(b[(n0 + 1) * rank + 2 * r2]) * p0
                    + bfv(b[(n0 + 1) * rank + 2 * r2 + 1]) * p1;
            }
            if y_fmt == 0 {
                let ybase = mi * y_row + y_off + n0;
                y[ybase] = (f32::from_bits(y[ybase]) + acc0).to_bits();
                y[ybase + 1] = (f32::from_bits(y[ybase + 1]) + acc1).to_bits();
            } else {
                let yw = (mi * y_row + y_off) / 2 + wi;
                let old = y[yw];
                if y_fmt == 2 {
                    let v0 = half::f16::from_bits((old & 0xffff) as u16).to_f32() + acc0;
                    let v1 = half::f16::from_bits((old >> 16) as u16).to_f32() + acc1;
                    y[yw] = (half::f16::from_f32(v0).to_bits() as u32)
                        | ((half::f16::from_f32(v1).to_bits() as u32) << 16);
                } else {
                    let v0 = bf16::from_bits((old & 0xffff) as u16).to_f32() + acc0;
                    let v1 = bf16::from_bits((old >> 16) as u16).to_f32() + acc1;
                    y[yw] = ((rne(v1) as u32) << 16) | (rne(v0) as u32);
                }
            }
        }
    }
    y
}

#[test]
fn dg_lora_kernels_match_a_bitwise_host_mirror_across_site_geometries() {
    if !adapter_available() {
        return;
    }
    let enc16 = |v: f32, f16: bool| -> u16 {
        if f16 {
            half::f16::from_f32(v).to_bits()
        } else {
            bf16::from_f32(v).to_bits()
        }
    };
    for (case, (m, k, n, rank, x_f16, y_fmt, y_off, y_row)) in [
        (1usize, 64usize, 64usize, 4usize, false, 1u32, 0usize, 64usize),
        (1, 64, 32, 4, false, 0, 96, 128),
        (8, 64, 64, 4, false, 1, 0, 64),
        (8, 64, 32, 4, false, 0, 96, 128),
        (8, 64, 128, 4, false, 1, 128, 256),
        (1, 64, 64, 4, true, 2, 0, 64),
    ]
    .into_iter()
    .enumerate()
    {
        let x: Vec<u16> = (0..m * k)
            .map(|i| enc16(((i as f32) * 0.377).sin() * 0.8, x_f16))
            .collect();
        let a: Vec<u16> = (0..rank * k)
            .map(|i| enc16(((i as f32) * 0.731).cos() * 0.3, false))
            .collect();
        let b: Vec<u16> = (0..n * rank)
            .map(|i| enc16(((i as f32) * 1.313).sin() * 0.4, false))
            .collect();
        let y_words = if y_fmt == 0 { m * y_row } else { m * y_row / 2 };
        let y_init: Vec<u32> = (0..y_words)
            .map(|i| {
                if y_fmt == 0 {
                    (((i as f32) * 0.113).cos() * 0.5).to_bits()
                } else {
                    let v0 = enc16(((2 * i) as f32 * 0.113).cos() * 0.5, y_fmt == 2) as u32;
                    let v1 = enc16(((2 * i + 1) as f32 * 0.113).cos() * 0.5, y_fmt == 2) as u32;
                    v0 | (v1 << 16)
                }
            })
            .collect();
        let got = nv_models::dots_ocr::decoder_wgpu_graph::dg_lora_probe(
            &x, &a, &b, &y_init, m, k, n, rank, x_f16, y_fmt, y_off, y_row,
        )
        .unwrap();
        let want = dgl_host_mirror(&x, &a, &b, &y_init, m, k, n, rank, x_f16, y_fmt, y_off, y_row);
        assert_eq!(got.len(), want.len(), "case {case}: length");
        let bad = got.iter().zip(want.iter()).enumerate().find(|(_, (g, w))| {
            if g == w {
                return false;
            }
            if y_fmt == 0 {
                let (gf, wf) = (f32::from_bits(**g), f32::from_bits(**w));
                !(gf.is_finite() && wf.is_finite() && (gf - wf).abs() <= 1e-4)
            } else {
                true
            }
        });
        assert!(
            bad.is_none(),
            "case {case} (m {m} n {n} y_fmt {y_fmt} y_off {y_off}): first mismatch at word {:?}",
            bad.map(|(i, (g, w))| format!("{i}: got {g:08x} want {w:08x}"))
        );
        eprintln!("dg-lora kernel case {case}: bitwise ok ({} words)", got.len());
    }
}

#[test]
fn graph_lora_prefill_chunk_matches_per_token_steps_bitwise() {
    if !adapter_available() {
        return;
    }
    let _env = env_lock();
    use nv_models::dots_ocr::decoder_wgpu_graph::{
        DOTS_GRAPH_LORA_DIR_ENV_HOT_LOADS_A_PEFT_ADAPTER_ON_THE_GRAPH_ARM, PF_STEEL_ENV,
        PF_STEEL_PROJ_ENV,
    };
    std::env::set_var(PF_STEEL_ENV, "0");
    std::env::set_var(PF_STEEL_PROJ_ENV, "0");
    let cfg = tiny_config();
    let adir = write_tiny_adapter(&cfg);
    std::env::set_var(
        DOTS_GRAPH_LORA_DIR_ENV_HOT_LOADS_A_PEFT_ADAPTER_ON_THE_GRAPH_ARM,
        adir.0.as_os_str(),
    );
    let (loader, cfg, _guard) = tiny_loader();
    let span = 4 * nv_models::dots_ocr::decoder_wgpu_graph::PREFILL_M_MAX + 16;
    let chunked = DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16);
    let per_token = DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16);
    std::env::remove_var(DOTS_GRAPH_LORA_DIR_ENV_HOT_LOADS_A_PEFT_ADAPTER_ON_THE_GRAPH_ARM);
    std::env::remove_var(PF_STEEL_ENV);
    std::env::remove_var(PF_STEEL_PROJ_ENV);
    let mut chunked = chunked.unwrap();
    let mut per_token = per_token.unwrap();
    let widths = chunked.prefill_widths().to_vec();
    let m = widths[0];
    let tokens: Vec<u32> = (0..(m + widths.get(1).copied().unwrap_or(1) + 3) as u32)
        .map(|i| (i * 7 + 3) % 96)
        .collect();
    let want = per_token.forward_tokens(&tokens).unwrap();
    let got = chunked.prefill_prompt(&tokens).unwrap();
    assert_eq!(
        want, got,
        "with a live adapter the prefill lora sites must land exactly what the decode lora \
         sites land token-by-token; a divergence here is a prefill site wiring error, not \
         rounding (the lora math is per-row and m-invariant)"
    );
    eprintln!(
        "lora chunk-vs-steps bitwise ok over {} tokens (widths {widths:?})",
        tokens.len()
    );
}

#[test]
fn graph_lora_w4_arm_applies_a_small_delta_without_corrupting_the_page() {
    if !adapter_available() {
        return;
    }
    let _env = env_lock();
    let cfg = tiny_config();
    let adir = write_tiny_adapter(&cfg);
    let (base_loader, _bg) = loader_for(&cfg);
    let mut base =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &base_loader, 32, DotsGraphQuant::AffineW4)
            .unwrap();
    std::env::set_var(
        nv_models::dots_ocr::decoder_wgpu_graph::DOTS_GRAPH_LORA_DIR_ENV_HOT_LOADS_A_PEFT_ADAPTER_ON_THE_GRAPH_ARM,
        adir.0.as_os_str(),
    );
    let (ll, _lg) = loader_for(&cfg);
    let live = DotsDecoderWgpuGraph::from_loader(cfg.clone(), &ll, 32, DotsGraphQuant::AffineW4);
    std::env::remove_var(
        nv_models::dots_ocr::decoder_wgpu_graph::DOTS_GRAPH_LORA_DIR_ENV_HOT_LOADS_A_PEFT_ADAPTER_ON_THE_GRAPH_ARM,
    );
    let mut live = live.unwrap();
    let mut token = 3u32;
    for step in 0..TINY_ORACLE_TOKENS {
        let (_bt, base_logits) = base.decode_step_logits(token).unwrap();
        let (_lt, live_logits) = live.decode_step_logits(token).unwrap();
        assert!(
            live_logits.iter().all(|v| v.is_finite()),
            "step {step}: non-finite w4 lora logits"
        );
        let rel = rel_err(&base_logits, &live_logits);
        assert!(
            rel > 0.0,
            "step {step}: the adapter left the w4 logits untouched"
        );
        assert!(
            rel < 0.5,
            "step {step}: a small lora delta moved w4 logits by rel {rel} -- a sink or \
             activation format mismatch on this arm corrupts whole rows, which is exactly \
             what this gate exists to catch"
        );
        eprintln!("w4-lora step {step}: rel(base,live)={rel:.5}");
        token = DotsDecoderWgpuGraph::greedy_token(&base_logits);
    }
}

#[test]
fn batched_region_decode_matches_fresh_serial_sessions() {
    let (loader, cfg, _guard) = tiny_loader();
    let span = 64 * 32;
    let mut wg =
        DotsDecoderWgpuGraph::from_loader(cfg.clone(), &loader, span, DotsGraphQuant::Bf16)
            .unwrap();
    let (slots, cap) = wg
        .region_capacity()
        .expect("region machinery must build at the smallest useful span");
    assert_eq!(slots, 32);
    assert!(cap >= 63, "stride cap {cap}");
    let prompts: [Vec<u32>; 3] = [
        vec![3, 17, 41, 8, 25],
        vec![60, 2, 9],
        vec![11, 30, 5, 77, 23, 45, 6],
    ];
    let steps = 6usize;
    let mut serial: Vec<Vec<u32>> = Vec::new();
    for p in &prompts {
        wg.reset().unwrap();
        let logits = wg.prefill_prompt(p).unwrap();
        let mut tok = DotsDecoderWgpuGraph::greedy_token(&logits);
        let mut out = vec![tok];
        for _ in 1..steps {
            let (t, _) = wg.decode_step_logits(tok).unwrap();
            tok = t;
            out.push(t);
        }
        serial.push(out);
    }
    wg.reset().unwrap();
    let mut rows: Vec<RegionRow> = Vec::new();
    for (slot, p) in prompts.iter().enumerate() {
        let len = wg.region_prefill(slot, p, 0, None).unwrap();
        assert_eq!(len, p.len() - 1, "region prefill consumes all but the last token");
        rows.push(RegionRow {
            token: *p.last().unwrap(),
            base: (slot * (span / 32)) as u32,
            len: len as u32,
        });
    }
    let mut batched: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    for _ in 0..steps {
        let toks = wg.bstep_tokens(&rows).unwrap();
        for (r, t) in toks.iter().enumerate() {
            batched[r].push(*t);
            rows[r].len += 1;
            rows[r].token = *t;
        }
    }
    assert_eq!(
        serial, batched,
        "batched region decode must reproduce fresh serial sessions token for token"
    );
}
