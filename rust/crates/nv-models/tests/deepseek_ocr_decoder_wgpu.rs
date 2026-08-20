#![cfg(feature = "wgpu")]

mod common;
use common::require;
use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_models::deepseek_ocr::{
    DeepseekOcrDecoder, DeepseekOcrDecoderConfig, DeepseekOcrDecoderWgpu,
};
use nv_weights::WeightLoader;

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

fn tiny_config() -> DeepseekOcrDecoderConfig {
    DeepseekOcrDecoderConfig {
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        intermediate_size: 40,
        moe_intermediate_size: 24,
        n_routed_experts: 4,
        n_shared_experts: 2,
        num_experts_per_tok: 2,
        first_k_dense_replace: 1,
        moe_layer_freq: 1,
        vocab_size: 96,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        norm_topk_prob: false,
        routed_scaling_factor: 1.0,
        bos_token_id: 0,
        eos_token_id: 1,
    }
}

fn tiny_weight_map(c: &DeepseekOcrDecoderConfig) -> HashMap<String, Tensor> {
    let dev = Device::Cpu;
    let h = c.hidden_size;
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
        for (name, seed) in [
            ("q_proj", 6.0),
            ("k_proj", 7.0),
            ("v_proj", 8.0),
            ("o_proj", 9.0),
        ] {
            m.insert(
                format!("{p}.self_attn.{name}.weight"),
                t(det_bf16(h * h, seed + s), (h, h)),
            );
        }
        if c.is_moe_layer(l) {
            m.insert(
                format!("{p}.mlp.gate.weight"),
                t(
                    det_bf16(c.n_routed_experts * h, 10.0 + s),
                    (c.n_routed_experts, h),
                ),
            );
            for e in 0..c.n_routed_experts {
                let es = 20.0 + s + e as f32 * 7.0;
                let inter = c.moe_intermediate_size;
                m.insert(
                    format!("{p}.mlp.experts.{e}.gate_proj.weight"),
                    t(det_bf16(inter * h, es), (inter, h)),
                );
                m.insert(
                    format!("{p}.mlp.experts.{e}.up_proj.weight"),
                    t(det_bf16(inter * h, es + 1.0), (inter, h)),
                );
                m.insert(
                    format!("{p}.mlp.experts.{e}.down_proj.weight"),
                    t(det_bf16(h * inter, es + 2.0), (h, inter)),
                );
            }
            let si = c.shared_expert_intermediate_size();
            m.insert(
                format!("{p}.mlp.shared_experts.gate_proj.weight"),
                t(det_bf16(si * h, 60.0 + s), (si, h)),
            );
            m.insert(
                format!("{p}.mlp.shared_experts.up_proj.weight"),
                t(det_bf16(si * h, 61.0 + s), (si, h)),
            );
            m.insert(
                format!("{p}.mlp.shared_experts.down_proj.weight"),
                t(det_bf16(h * si, 62.0 + s), (h, si)),
            );
        } else {
            let inter = c.intermediate_size;
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
    }
    m
}

fn tiny_loader() -> (WeightLoader, DeepseekOcrDecoderConfig, tempdir::Guard) {
    let c = tiny_config();
    let map = tiny_weight_map(&c);
    let dir = std::env::temp_dir().join(format!(
        "dsocr-decoder-wgpu-test-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    candle_core::safetensors::save(&map, &path).unwrap();
    let loader = WeightLoader::open_file(&path, &Device::Cpu).unwrap();
    (loader, c, tempdir::Guard(dir))
}

mod tempdir {
    pub struct Guard(pub std::path::PathBuf);
    impl Drop for Guard {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }
}

#[test]
fn wgpu_decoder_matches_cpu_reference_on_greedy_steps() {
    if !adapter_available() {
        return;
    }
    let (loader, cfg, _guard) = tiny_loader();
    let cpu =
        DeepseekOcrDecoder::from_loader_with_dtype(cfg.clone(), &loader, &Device::Cpu, DType::F32)
            .unwrap();
    let mut wg = DeepseekOcrDecoderWgpu::from_loader(cfg.clone(), &loader, 16).unwrap();

    let tokens: Vec<u32> = vec![0, 5, 17, 42, 9];
    let mut cache = cpu.new_kv_cache(16).unwrap();
    let mut cpu_logits: Vec<f32> = Vec::new();
    for &t in &tokens {
        let l = cpu.forward_tokens(&[t], None, &mut cache).unwrap();
        cpu_logits = l.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    }
    let wg_logits = wg.forward_tokens(&tokens).unwrap();
    assert_eq!(wg.current_pos(), tokens.len());
    assert_eq!(wg_logits.len(), cfg.vocab_size);
    assert_eq!(cpu_logits.len(), cfg.vocab_size);

    let scale = cpu_logits
        .iter()
        .fold(0f32, |a, &b| a.max(b.abs()))
        .max(1e-6);
    let mut max_abs = 0f32;
    for (a, b) in cpu_logits.iter().zip(wg_logits.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    let rel = max_abs / scale;
    let cpu_argmax = DeepseekOcrDecoderWgpu::greedy_token(&cpu_logits);
    let wg_argmax = DeepseekOcrDecoderWgpu::greedy_token(&wg_logits);
    eprintln!(
        "logit agreement: max_abs={max_abs:.5} scale={scale:.4} rel={rel:.5} argmax cpu={cpu_argmax} wgpu={wg_argmax}"
    );
    assert_eq!(
        cpu_argmax, wg_argmax,
        "greedy token diverged (rel err {rel})"
    );
    assert!(rel < 3e-2, "logits diverged: rel {rel} >= 3e-2");
}

fn real_checkpoint_dir() -> Option<std::path::PathBuf> {
    if let Ok(d) = std::env::var("NV_OCR_DEEPSEEK_DIR") {
        return Some(std::path::PathBuf::from(d));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR-2/snapshots");
    let mut entries: Vec<_> = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

#[test]
#[ignore]
fn real_checkpoint_one_token_forward_on_wgpu() {
    if std::env::var("NV_DSOCR_WGPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: NV_DSOCR_WGPU_TEST=1 not set");
        return;
    }
    if !adapter_available() {
        return;
    }
    let Some(dir) = real_checkpoint_dir() else {
        eprintln!("skipping: no DeepSeek-OCR-2 checkpoint");
        return;
    };
    let cfg = DeepseekOcrDecoderConfig::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).unwrap();
    let t0 = std::time::Instant::now();
    let mut wg = DeepseekOcrDecoderWgpu::from_loader(cfg.clone(), &loader, 32).unwrap();
    eprintln!(
        "loaded real decoder host-side in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    let t1 = std::time::Instant::now();
    let logits = wg.forward_token(cfg.bos_token_id).unwrap();
    eprintln!(
        "one decode step on wgpu in {:.2}s, argmax={}",
        t1.elapsed().as_secs_f32(),
        DeepseekOcrDecoderWgpu::greedy_token(&logits)
    );
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
}

#[test]
fn wgpu_decoder_reset_reproduces_logits() {
    if !adapter_available() {
        return;
    }
    let (loader, cfg, _guard) = tiny_loader();
    let mut wg = DeepseekOcrDecoderWgpu::from_loader(cfg, &loader, 16).unwrap();
    let a = wg.forward_tokens(&[0, 7, 33]).unwrap();
    wg.reset();
    let b = wg.forward_tokens(&[0, 7, 33]).unwrap();
    assert_eq!(a, b, "wgpu decode must be deterministic across reset");
}
