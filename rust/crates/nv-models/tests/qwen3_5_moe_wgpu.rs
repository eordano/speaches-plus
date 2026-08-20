#![cfg(feature = "wgpu")]

mod common;
use common::bf16_lin_q3w as bf16_lin;
use common::have_gpu;
use common::LcgOddSeedShift33SignedUnit as Lcg;
use common::norm_vec;
use common::nozi_prof_dump;
use common::nvfp4;
use common::rel_err;
use common::tiny_config_qwen36_moe as tiny_config;
use common::tiny_weights;
use nv_models::qwen3_5_moe::{LayerType, Qwen3MoeConfig};
use nv_models::qwen3_5_moe_wgpu as q3w;

#[test]
fn tiny_wgpu_decode_matches_cpu_reference() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x51ee_d100_0001);
    let mut gpu = q3w::Qwen3MoeWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu model");
    eprintln!("[wgpu] recorded passes per token: {}", gpu.pass_count());

    let mut st = q3w::RefState::new(&cfg);
    let tokens: [u32; 6] = [3, 11, 5, 40, 2, 19];
    let mut worst_rel = 0f32;
    let mut top1_hits = 0usize;
    let mut all_logits: Vec<Vec<f32>> = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        let (arg, logits) = gpu.decode_step_logits(*t).expect("decode step");
        all_logits.push(logits.clone());
        let want = q3w::reference_step(&cfg, &hw, &mut st, *t).expect("reference step");
        let (abs, rel) = rel_err(&logits, &want);
        let ref_arg = want
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        if arg == ref_arg {
            top1_hits += 1;
        }
        worst_rel = worst_rel.max(rel);
        eprintln!(
            "step {i}: tok={t} gpu_argmax={arg} ref_argmax={ref_arg} max_abs={abs:.6} rel={rel:.6}"
        );
        assert!(
            rel < 0.05,
            "step {i}: logits diverged from CPU reference (rel {rel})"
        );
    }
    eprintln!("[wgpu] worst relative logit error over 6 steps: {worst_rel:.6}");
    assert_eq!(
        top1_hits,
        tokens.len(),
        "argmax disagreed with the CPU reference on {} of {} steps",
        tokens.len() - top1_hits,
        tokens.len()
    );

    let (spread, _) = rel_err(&all_logits[0], &all_logits[2]);
    eprintln!("[wgpu] logit spread between step 0 and step 2: {spread:.6}");
    assert!(
        spread > 1e-3,
        "logits are insensitive to the input token / recurrent state (spread {spread}); \
         the reference comparison would then be vacuous"
    );
}

#[test]
fn tiny_wgpu_recurrent_state_carries_and_resets() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x51ee_d100_0002);
    let mut gpu = q3w::Qwen3MoeWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu model");

    let (a0, l0) = gpu.decode_step_logits(7).expect("step");
    let (_a1, l1) = gpu.decode_step_logits(7).expect("step");
    let same = l0.iter().zip(l1.iter()).all(|(x, y)| (x - y).abs() <= 1e-6);
    assert!(
        !same,
        "feeding the same token twice produced identical logits: DeltaNet/KV state is not carried"
    );

    gpu.reset().expect("reset");
    let (a2, l2) = gpu.decode_step_logits(7).expect("step after reset");
    assert_eq!(a0, a2, "reset did not restore the first-token argmax");
    let (abs, _) = rel_err(&l0, &l2);
    assert!(
        abs <= 1e-5,
        "reset did not restore the first-token logits (max abs {abs})"
    );
    eprintln!("[wgpu] state carry verified; reset max_abs={abs:.8}");
}

#[test]
fn real_snapshot_config_is_supported_by_the_wgpu_module() {
    let snap = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots/e850c696e6d75f965367e816c16bc7dacd955ffa/config.json");
    if !snap.exists() {
        eprintln!("[skip] {} not present", snap.display());
        return;
    }
    let cfg = Qwen3MoeConfig::from_hf_json_file(&snap).expect("parse config");
    eprintln!(
        "[cfg] hidden={} layers={} experts={} topk={} head_dim={} rot={} lin_k={}x{} lin_v={}x{}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.head_dim,
        cfg.rotary_dim(),
        cfg.linear_num_key_heads,
        cfg.linear_key_head_dim,
        cfg.linear_num_value_heads,
        cfg.linear_value_head_dim,
    );
    assert_eq!(cfg.hidden_size % 32, 0);
    assert_eq!(cfg.moe_intermediate_size % 64, 0);
    assert_eq!(cfg.shared_expert_intermediate_size % 64, 0);
    assert!(cfg.num_experts <= 256);
    assert!(cfg.num_experts_per_tok <= 16);
    assert!(cfg.head_dim <= 256 && cfg.head_dim.is_multiple_of(2));
    assert!(cfg.linear_key_head_dim <= 128);
    assert!(cfg.linear_value_head_dim <= 128 && cfg.linear_value_head_dim.is_multiple_of(2));
    assert_eq!(cfg.rotary_dim() % 2, 0);
    assert_eq!(cfg.linear_num_value_heads % cfg.linear_num_key_heads, 0);
    assert_eq!(cfg.num_attention_heads % cfg.num_key_value_heads, 0);
    assert_eq!(cfg.layer_types.len(), cfg.num_hidden_layers);
}

#[test]
#[ignore = "loads ~20 GB of NVFP4 weights; set NV_QWEN36_WGPU_TEST=1"]
fn qwen36_wgpu_real_weights_decode() {
    if std::env::var("NV_QWEN36_WGPU_TEST").is_err() {
        eprintln!("[skip] NV_QWEN36_WGPU_TEST not set");
        return;
    }
    if !have_gpu() {
        panic!("real-weights test needs a wgpu adapter");
    }
    let dir = std::env::var("NV_QWEN36_DIR").unwrap_or_else(|_| {
        format!(
            "{}/.cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots/e850c696e6d75f965367e816c16bc7dacd955ffa",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let dir = std::path::PathBuf::from(dir);
    let mut cfg = Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    if let Some(n) = std::env::var("NV_QWEN36_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        assert!(n > 0 && n <= cfg.num_hidden_layers);
        cfg.num_hidden_layers = n;
        cfg.layer_types.truncate(n);
        eprintln!("[real] TRUNCATED to the first {n} layers (VRAM-limited partial load)");
    }
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let max_seq: usize = std::env::var("NV_QWEN36_MAX_SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let t0 = std::time::Instant::now();
    let mut gpu =
        q3w::Qwen3MoeWgpu::from_loader(cfg.clone(), &loader, max_seq).expect("build from loader");
    eprintln!(
        "[real] built in {:.1}s, {} passes/token",
        t0.elapsed().as_secs_f64(),
        gpu.pass_count()
    );

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let prompt = std::env::var("NV_QWEN36_PROMPT")
        .unwrap_or_else(|_| "The capital of France is".to_string());
    let enc = tok.encode(prompt.as_str(), false).expect("encode");
    let ids: Vec<u32> = enc.get_ids().to_vec();
    assert!(!ids.is_empty());

    let mut next = gpu.prefill(&ids).expect("prefill");
    let mut out = vec![next];
    let n_new: usize = std::env::var("NV_QWEN36_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let t1 = std::time::Instant::now();
    for _ in 1..n_new {
        next = gpu.decode_step(next).expect("decode");
        out.push(next);
    }
    let ms = t1.elapsed().as_secs_f64() * 1000.0 / (n_new.saturating_sub(1).max(1)) as f64;
    let text = tok.decode(&out, false).unwrap_or_default();
    eprintln!("[real] prompt={prompt:?}");
    eprintln!("[real] token_ids={out:?}");
    eprintln!("[real] continuation={text:?}");
    eprintln!("[real] {ms:.2} ms/tok decode");
    nozi_prof_dump();
    if std::env::var("NV_QWEN36_LAYERS").is_ok() {
        eprintln!("[real] truncated model: text coherence is NOT expected, only that it runs");
        return;
    }
    assert!(
        out.iter().any(|t| *t != out[0]),
        "generation collapsed to a single repeated token"
    );
}

fn gpu_mem_used_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

struct PeakSampler {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    peak: std::sync::Arc<std::sync::atomic::AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl PeakSampler {
    fn start() -> Self {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use std::sync::Arc;
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let (s, p) = (stop.clone(), peak.clone());
        let handle = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                if let Some(m) = gpu_mem_used_mib() {
                    p.fetch_max(m, Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_millis(40));
            }
        });
        Self {
            stop,
            peak,
            handle: Some(handle),
        }
    }

    fn finish(mut self) -> u64 {
        use std::sync::atomic::Ordering;
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.peak.load(Ordering::Relaxed)
    }
}

fn vram_config() -> Qwen3MoeConfig {
    let layers: usize = std::env::var("NV_Q3W_VRAM_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let mut layer_types = vec![LayerType::LinearAttention; layers];
    if layers > 2 {
        layer_types[2] = LayerType::FullAttention;
    }
    Qwen3MoeConfig {
        hidden_size: 1024,
        num_hidden_layers: layers,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        head_dim: 64,
        moe_intermediate_size: 512,
        shared_expert_intermediate_size: 512,
        num_experts: 256,
        num_experts_per_tok: 8,
        vocab_size: 2048,
        max_position_embeddings: 64,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 0.5,
        bos_token_id: 0,
        eos_token_id: 1,
        layer_types,
        linear_num_key_heads: 4,
        linear_num_value_heads: 8,
        linear_key_head_dim: 64,
        linear_value_head_dim: 64,
        linear_conv_kernel_dim: 4,
        attn_output_gate: true,
        tie_word_embeddings: false,
    }
}

fn vram_weights(cfg: &Qwen3MoeConfig, seed: u64) -> q3w::HostWeights {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let sinter = cfg.shared_expert_intermediate_size;
    let hd = cfg.head_dim;
    let n_k = cfg.linear_num_key_heads;
    let n_v = cfg.linear_num_value_heads;
    let d_k = cfg.linear_key_head_dim;
    let d_v = cfg.linear_value_head_dim;
    let key_dim = n_k * d_k;
    let value_dim = n_v * d_v;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;

    let gate_tpl = nvfp4(&mut r, inter, hidden, 0.15);
    let up_tpl = nvfp4(&mut r, inter, hidden, 0.15);
    let down_tpl = nvfp4(&mut r, hidden, inter, 0.15);
    let stack = |t: &q3w::HostNvfp4Lin, e: usize| {
        let mats: Vec<q3w::HostNvfp4Lin> = (0..e).map(|_| t.clone()).collect();
        q3w::stack_nvfp4_host(&mats)
    };

    let mut layers = Vec::new();
    for li in 0..cfg.num_hidden_layers {
        let mixer = match cfg.layer_types[li] {
            LayerType::LinearAttention => q3w::HostMixer::Delta(Box::new(q3w::HostDeltaNet {
                in_proj_qkv: bf16_lin(&mut r, conv_dim, hidden, 0.12),
                in_proj_z: bf16_lin(&mut r, value_dim, hidden, 0.12),
                in_proj_ab: bf16_lin(&mut r, 2 * n_v, hidden, 0.12),
                conv1d: r.f32_vec(conv_dim * ks, 0.4),
                a_log: r.f32_vec(n_v, 0.5),
                dt_bias: r.f32_vec(n_v, 0.5),
                norm_w: norm_vec(&mut r, d_v),
                out_proj: bf16_lin(&mut r, hidden, value_dim, 0.12),
            })),
            LayerType::FullAttention => {
                let q_out = cfg.num_attention_heads * hd * 2;
                let kv_out = cfg.num_key_value_heads * hd;
                q3w::HostMixer::Attn(Box::new(q3w::HostAttention {
                    q: nvfp4(&mut r, q_out, hidden, 0.12),
                    k: nvfp4(&mut r, kv_out, hidden, 0.12),
                    v: nvfp4(&mut r, kv_out, hidden, 0.12),
                    o: nvfp4(&mut r, hidden, cfg.num_attention_heads * hd, 0.12),
                    q_norm: norm_vec(&mut r, hd),
                    k_norm: norm_vec(&mut r, hd),
                }))
            }
        };
        layers.push(q3w::HostLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            mixer,
            moe: q3w::HostMoe {
                router: bf16_lin(&mut r, cfg.num_experts, hidden, 0.3),
                experts_gate: stack(&gate_tpl, cfg.num_experts),
                experts_up: stack(&up_tpl, cfg.num_experts),
                experts_down: stack(&down_tpl, cfg.num_experts),
                shared_gate: nvfp4(&mut r, sinter, hidden, 0.15),
                shared_up: nvfp4(&mut r, sinter, hidden, 0.15),
                shared_down: nvfp4(&mut r, hidden, sinter, 0.15),
                shared_expert_gate: bf16_lin(&mut r, 1, hidden, 0.3),
            },
        });
    }

    q3w::HostWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: norm_vec(&mut r, hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

#[test]
fn build_peak_vram_stays_near_declared_buffer_bytes() {
    if !have_gpu() {
        return;
    }
    let Some(baseline) = gpu_mem_used_mib() else {
        eprintln!("[skip] nvidia-smi unavailable");
        return;
    };
    let cfg = vram_config();
    let hw = vram_weights(&cfg, 0x5122_a10c_0001);

    let sampler = PeakSampler::start();
    let gpu = q3w::Qwen3MoeWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu model");
    let peak = sampler.finish();
    let after = gpu_mem_used_mib().unwrap_or(0);

    let report = gpu.vram_report().clone();
    let declared_mib = report.total_bytes / (1 << 20);
    let peak_delta = peak.saturating_sub(baseline);
    let after_delta = after.saturating_sub(baseline);
    eprint!("[vram] {}", report.render());
    eprintln!(
        "[vram] staging_flush={} declared={declared_mib} MiB  smi baseline={baseline} peak={peak} after={after} MiB  \
         peak_delta={peak_delta} MiB ({:.2}x declared)  after_delta={after_delta} MiB ({:.2}x declared)",
        q3w::staging_flush_enabled(),
        peak_delta as f64 / declared_mib.max(1) as f64,
        after_delta as f64 / declared_mib.max(1) as f64,
    );
    assert!(
        declared_mib > 512,
        "synthetic model too small to measure ({declared_mib} MiB)"
    );
    let transient = peak_delta.saturating_sub(after_delta);
    eprintln!(
        "[vram] transient build overhead = peak_delta - after_delta = {transient} MiB \
         ({:.2}x declared)",
        transient as f64 / declared_mib.max(1) as f64
    );
    if !q3w::staging_flush_enabled() {
        eprintln!("[vram] staging flush disabled -- reporting only, no assertion");
        return;
    }
    assert!(
        transient < 1024 && transient < declared_mib,
        "transient VRAM above the model's steady-state footprint reached {transient} MiB while \
         the model only declares {declared_mib} MiB; host->device staging buffers are \
         accumulating because pending writes are never submitted. Transient overhead must be \
         bounded by the flush threshold, not proportional to model size."
    );
}

const REPLICA_LAYERS_8_KEEPS_THE_REAL_3_TO_1_LINEAR_TO_FULL_RATIO_OF_THE_40_LAYER_35B: usize = 8;

fn qwen36_35b_shape_replica_config_reduced_layers_and_vocab_because_prefill_excludes_lm_head(
    layers: usize,
    max_seq: usize,
) -> Qwen3MoeConfig {
    let layer_types = (0..layers)
        .map(|i| {
            if i % 4 == 3 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    Qwen3MoeConfig {
        hidden_size: 2048,
        num_hidden_layers: layers,
        num_attention_heads: 16,
        num_key_value_heads: 2,
        head_dim: 256,
        moe_intermediate_size: 512,
        shared_expert_intermediate_size: 512,
        num_experts: 256,
        num_experts_per_tok: 8,
        vocab_size: 4096,
        max_position_embeddings: max_seq,
        rope_theta: 10_000_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 0.25,
        bos_token_id: 0,
        eos_token_id: 1,
        layer_types,
        linear_num_key_heads: 16,
        linear_num_value_heads: 32,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        attn_output_gate: true,
        tie_word_embeddings: false,
    }
}

fn pf_label_family(label: &str) -> &str {
    let rest = label.strip_prefix("q3w-pf-").unwrap_or(label);
    let cut = rest
        .char_indices()
        .find(|(_, c)| *c == '-')
        .map(|(i, _)| i)
        .unwrap_or(rest.len());
    &rest[..cut]
}

#[test]
#[ignore = "profiles the M-row prefill of a synthetic-weight replica at the real qwen3.6-35B \
            shapes (hidden 2048, hd256 16h/2kv, linear 16k/32v x128, 256 experts top-8, inter \
            512); set NV_Q3M_PF_ATTR_TEST=1 and NV_WGPU_PROFILE=1; kernel time is a function of \
            shape not value, so this attribution stands in while the real checkpoint snapshot \
            is a dangling symlink -- rates here are per-layer and must be scaled to 40 layers"]
fn qwen36_shape_replica_prefill_attribution_synthetic_weights() {
    if std::env::var("NV_Q3M_PF_ATTR_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_Q3M_PF_ATTR_TEST != 1");
        return;
    }
    if !have_gpu() {
        panic!("wgpu adapter required: this gated instrument must never silently skip");
    }
    use nv_kernels::wgpu_backend::dispatch::profile;
    let layers: usize = std::env::var("NV_Q3M_ATTR_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(REPLICA_LAYERS_8_KEEPS_THE_REAL_3_TO_1_LINEAR_TO_FULL_RATIO_OF_THE_40_LAYER_35B);
    let depth: usize = std::env::var("NV_Q3M_ATTR_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let cfg = qwen36_35b_shape_replica_config_reduced_layers_and_vocab_because_prefill_excludes_lm_head(
        layers,
        depth + 64,
    );
    let hw = vram_weights(&cfg, 0x517c_c1b7_2026);
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared().expect("shared wgpu context");
    let t_build = std::time::Instant::now();
    let mut gpu = q3w::Qwen3MoeWgpu::new(cfg, &hw, depth + 64).expect("build replica");
    drop(hw);
    let (mix_dense, mix_moe) = gpu.prefill_mrow_pass_mix();
    eprintln!(
        "[q3m-attr] built in {:.1}s layers={layers} mrow m={} passes/chunk={} mix=({mix_dense},{mix_moe})",
        t_build.elapsed().as_secs_f64(),
        gpu.prefill_mrow_chunk_len(),
        gpu.prefill_mrow_pass_count()
    );
    assert!(
        gpu.prefill_mrow_chunk_len() >= 2,
        "the M-row prefill list is off on the replica; the attribution would profile the \
         per-token replay instead of the shipping prefill path"
    );
    let ids: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 2000)).collect();
    profile::reset();
    let t0 = std::time::Instant::now();
    let done = gpu.prefill_tokens(&ids).expect("prefill_tokens");
    for t in &ids[done..] {
        gpu.prefill_step(*t).expect("tail prefill step");
    }
    ctx.poll_blocking().expect("drain prefill before stopping the clock");
    let prefill_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "[q3m-attr] PF-RATE depth={depth} layers={layers} prefill_s={prefill_s:.3} tok_s={:.1} \
         chunk_done={done} basis=synthetic_weights_real_shapes_release_profileflag={}",
        depth as f64 / prefill_s,
        profile::enabled()
    );
    if profile::enabled() {
        let rows = profile::report();
        let total: f64 = rows.iter().map(|r| r.2).sum();
        let mut fam: std::collections::BTreeMap<String, (u64, f64)> = Default::default();
        for (label, count, ns) in &rows {
            let e = fam.entry(pf_label_family(label).to_string()).or_default();
            e.0 += count;
            e.1 += ns;
        }
        let mut fam: Vec<(String, (u64, f64))> = fam.into_iter().collect();
        fam.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap_or(std::cmp::Ordering::Equal));
        for (family, (count, ns)) in &fam {
            eprintln!(
                "[q3m-attr] FAMILY {family} n={count} total_ms={:.1} share={:.1}%",
                ns / 1e6,
                100.0 * ns / total.max(1.0)
            );
        }
        for (label, count, ns) in rows.into_iter().take(48) {
            eprintln!(
                "[q3m-attr] {label} n={count} total_ms={:.1} share={:.1}%",
                ns / 1e6,
                100.0 * ns / total.max(1.0)
            );
        }
        eprintln!("[q3m-attr] TOTAL {:.1}ms GPU", total / 1e6);
    }
}

mod router_parity {
    use nv_kernels::wgpu_backend::{dispatch, WgpuContext};

    #[repr(C)]
    #[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
    struct RtParams {
        n_experts: u32,
        k: u32,
        pad0: u32,
        pad1: u32,
    }

    fn ctx_or_skip() -> Option<&'static WgpuContext> {
        match WgpuContext::shared() {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("skipping: no wgpu adapter ({e})");
                None
            }
        }
    }

    fn run(
        ctx: &WgpuContext,
        entry: &str,
        logits: &[f32],
        n: usize,
        k: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        let src = nv_models::qwen3_5_moe_wgpu::moe_source();
        let bits: Vec<u32> = logits.iter().map(|v| v.to_bits()).collect();
        let lg = dispatch::storage_from_slice(ctx, "rt-logits", &bits);
        let ids = dispatch::storage_zeroed(ctx, "rt-ids", (k * 4) as u64);
        let w = dispatch::storage_zeroed(ctx, "rt-w", (k * 4) as u64);
        let p = dispatch::uniform_from(
            ctx,
            "rt-p",
            &RtParams {
                n_experts: n as u32,
                k: k as u32,
                pad0: 0,
                pad1: 0,
            },
        );
        let pl = dispatch::cached_compute_pipeline(ctx, entry, &src, entry).unwrap();
        let bg = dispatch::bind_group(ctx, &pl, &[(0, &lg), (1, &ids), (2, &w), (3, &p)]);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pl);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit([enc.finish()]);
        let got_ids: Vec<u32> = dispatch::read_back(ctx, &ids, k).unwrap();
        let got_w: Vec<f32> = dispatch::read_back(ctx, &w, k).unwrap();
        (got_ids, got_w)
    }

    #[test]
    #[ignore]
    fn parallel_router_is_bit_identical_to_serial() {
        let Some(ctx) = ctx_or_skip() else { return };
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        let mut cases: Vec<(String, Vec<f32>, usize, usize)> = Vec::new();

        for (n, k) in [(256usize, 8usize), (128, 8), (64, 4), (256, 1), (32, 16)] {
            let v: Vec<f32> = (0..n)
                .map(|_| (next() >> 40) as f32 / 8192.0 - 1.0)
                .collect();
            cases.push((format!("random n={n} k={k}"), v, n, k));
        }

        cases.push(("all-equal n=256 k=8".into(), vec![0.5f32; 256], 256, 8));

        let mut dup = vec![-1.0f32; 256];
        for i in [3usize, 7, 11, 200, 201] {
            dup[i] = 9.0;
        }
        cases.push(("duplicate-max n=256 k=8".into(), dup, 256, 8));

        cases.push((
            "neg-inf n=64 k=4".into(),
            vec![f32::NEG_INFINITY; 64],
            64,
            4,
        ));
        let ramp: Vec<f32> = (0..256).map(|i| -(i as f32)).collect();
        cases.push(("descending n=256 k=8".into(), ramp, 256, 8));

        let mut failures = Vec::new();
        for (name, logits, n, k) in cases {
            let (sid, sw) = run(ctx, "q3w_router_topk", &logits, n, k);
            let (pid, pw) = run(ctx, "q3w_router_topk_par", &logits, n, k);
            let ids_match = sid == pid;
            let w_match = sw
                .iter()
                .zip(pw.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
            eprintln!(
                "{name:<26} ids {} weights {}",
                if ids_match { "match" } else { "DIFFER" },
                if w_match { "bit-identical" } else { "DIFFER" }
            );
            if !ids_match {
                failures.push(format!("{name}: ids {sid:?} vs {pid:?}"));
            }
            if !w_match {
                failures.push(format!("{name}: weights {sw:?} vs {pw:?}"));
            }
        }
        assert!(failures.is_empty(), "router parity failures: {failures:#?}");
    }
}

mod delta_parity {
    use nv_kernels::wgpu_backend::{dispatch, WgpuContext};

    #[repr(C)]
    #[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
    struct RecParams {
        heads: u32,
        d_k: u32,
        d_v: u32,
        pad0: u32,
    }

    struct Shape {
        heads: usize,
        d_k: usize,
        d_v: usize,
    }

    struct Inputs {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        g: Vec<f32>,
        beta: Vec<f32>,
        state0: Vec<f32>,
        steps: usize,
    }

    fn ctx_or_skip() -> Option<&'static WgpuContext> {
        match WgpuContext::shared() {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("skipping: no wgpu adapter ({e})");
                None
            }
        }
    }

    fn run(ctx: &WgpuContext, entry: &str, s: &Shape, x: &Inputs) -> (Vec<f32>, Vec<f32>) {
        let src = nv_models::qwen3_5_moe_wgpu::delta_source();
        let qb = dispatch::storage_from_slice(ctx, "dr-q", &x.q);
        let kb = dispatch::storage_from_slice(ctx, "dr-k", &x.k);
        let vb = dispatch::storage_from_slice(ctx, "dr-v", &x.v);
        let gb = dispatch::storage_from_slice(ctx, "dr-g", &x.g);
        let bb = dispatch::storage_from_slice(ctx, "dr-beta", &x.beta);
        let ob = dispatch::storage_zeroed(ctx, "dr-out", (s.heads * s.d_v * 4) as u64);
        let sb = dispatch::storage_from_slice(ctx, "dr-state", &x.state0);
        let p = dispatch::uniform_from(
            ctx,
            "dr-p",
            &RecParams {
                heads: s.heads as u32,
                d_k: s.d_k as u32,
                d_v: s.d_v as u32,
                pad0: 0,
            },
        );
        let pl = dispatch::cached_compute_pipeline(ctx, entry, &src, entry).unwrap();
        let bg = dispatch::bind_group(
            ctx,
            &pl,
            &[
                (30, &qb),
                (31, &kb),
                (32, &vb),
                (33, &gb),
                (34, &bb),
                (35, &ob),
                (36, &sb),
                (37, &p),
            ],
        );
        for _ in 0..x.steps {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pl);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(s.heads as u32, 1, 1);
            }
            ctx.queue.submit([enc.finish()]);
        }
        let out: Vec<f32> = dispatch::read_back(ctx, &ob, s.heads * s.d_v).unwrap();
        let st: Vec<f32> = dispatch::read_back(ctx, &sb, s.heads * s.d_k * s.d_v).unwrap();
        (out, st)
    }

    #[test]
    #[ignore]
    fn unrolled_delta_recurrent_is_bit_identical_to_serial() {
        let Some(ctx) = ctx_or_skip() else { return };
        let mut seed = 0x243f6a8885a308d3u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut unit = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| (next() >> 40) as f32 / 8388608.0 - 1.0)
                .collect()
        };

        let plans: [(&str, Shape, Option<f32>, Option<f32>); 5] = [
            (
                "large 32x128x128",
                Shape {
                    heads: 32,
                    d_k: 128,
                    d_v: 128,
                },
                None,
                None,
            ),
            (
                "tiny 4x16x16",
                Shape {
                    heads: 4,
                    d_k: 16,
                    d_v: 16,
                },
                None,
                None,
            ),
            (
                "ragged 3x6x8",
                Shape {
                    heads: 3,
                    d_k: 6,
                    d_v: 8,
                },
                None,
                None,
            ),
            (
                "g=0 8x32x32",
                Shape {
                    heads: 8,
                    d_k: 32,
                    d_v: 32,
                },
                Some(0.0),
                None,
            ),
            (
                "g=1,beta=0 8x32x32",
                Shape {
                    heads: 8,
                    d_k: 32,
                    d_v: 32,
                },
                Some(1.0),
                Some(0.0),
            ),
        ];

        let mut failures = Vec::new();
        for (name, s, gov, bov) in plans {
            let x = Inputs {
                q: unit(s.heads * s.d_k),
                k: unit(s.heads * s.d_k),
                v: unit(s.heads * s.d_v),
                g: match gov {
                    Some(c) => vec![c; s.heads],
                    None => unit(s.heads).iter().map(|t| 0.5 + 0.25 * t).collect(),
                },
                beta: match bov {
                    Some(c) => vec![c; s.heads],
                    None => unit(s.heads).iter().map(|t| 0.5 + 0.5 * t).collect(),
                },
                state0: unit(s.heads * s.d_k * s.d_v),
                steps: 3,
            };
            let (o_ref, s_ref) = run(ctx, "q3w_delta_recurrent", &s, &x);
            let (o_u4, s_u4) = run(ctx, "q3w_delta_recurrent_u4", &s, &x);
            let nonzero = o_ref.iter().filter(|v| **v != 0.0).count();
            let out_ok = o_ref
                .iter()
                .zip(o_u4.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
            let st_ok = s_ref
                .iter()
                .zip(s_u4.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
            eprintln!(
                "{name:<22} out {} state {} ({nonzero}/{} out words nonzero)",
                if out_ok { "bit-identical" } else { "DIFFER" },
                if st_ok { "bit-identical" } else { "DIFFER" },
                o_ref.len()
            );

            assert!(
                nonzero > 0,
                "{name}: the reference kernel produced all zeros, so this case measures nothing"
            );
            if !out_ok {
                let i = o_ref
                    .iter()
                    .zip(o_u4.iter())
                    .position(|(a, b)| a.to_bits() != b.to_bits())
                    .unwrap();
                failures.push(format!("{name}: out[{i}] {} vs {}", o_ref[i], o_u4[i]));
            }
            if !st_ok {
                let i = s_ref
                    .iter()
                    .zip(s_u4.iter())
                    .position(|(a, b)| a.to_bits() != b.to_bits())
                    .unwrap();
                failures.push(format!("{name}: state[{i}] {} vs {}", s_ref[i], s_u4[i]));
            }
        }
        assert!(failures.is_empty(), "delta parity failures: {failures:#?}");
    }
}
