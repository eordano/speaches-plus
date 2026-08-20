#![cfg(feature = "wgpu")]

mod common;
use common::envn;
use common::LcgCentered0p1Shift32 as Lcg;
use std::path::PathBuf;
use std::time::Instant;

use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_wgpu::{Gemma4Wgpu, HostBf16Lin, HostLayer, HostProj, HostWeights};
use common::config_json_gemma4_layers as config_json;

const CONTINUATION_TOKENS: usize = 8;

const TOLERANCE_OF_PEAK_ABS_LOGIT_BECAUSE_TILED_ONLINE_SOFTMAX_REASSOCIATES: f32 = 1e-4;

fn gpu_or_refuse() {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => eprintln!("[g4-pff] adapter: {}", ctx.summary()),
        Err(e) => panic!("gemma4 prefill flash tolerance suite needs a wgpu adapter: {e}"),
    }
}

fn host_weights_all_bf16(config: &Gemma4Config, seed: u64) -> HostWeights {
    let mut rng = Lcg(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let n_q = config.num_attention_heads;
    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        let kind = config.layer_kind(i);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );
        let qkv_rows = q_dim + kv_dim * if has_v { 2 } else { 1 };
        let bf16 = |rng: &mut Lcg, n: usize, k: usize| {
            HostProj::Bf16(HostBf16Lin {
                w: rng.bf16_vec(n * k),
                n,
                k,
            })
        };
        layers.push(HostLayer {
            kind,
            input_ln: rng.bf16_vec_around_one(hidden),
            post_attn_ln: rng.bf16_vec_around_one(hidden),
            pre_ff_ln: rng.bf16_vec_around_one(hidden),
            post_ff_ln: rng.bf16_vec_around_one(hidden),
            q_norm: rng.bf16_vec_around_one(hd),
            k_norm: rng.bf16_vec_around_one(hd),
            layer_scalar: 0.9,
            has_v,
            qkv: bf16(&mut rng, qkv_rows, hidden),
            o: bf16(&mut rng, hidden, q_dim),
            gate_up: bf16(&mut rng, 2 * inter, hidden),
            down: bf16(&mut rng, hidden, inter),
        });
    }
    HostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden),
        final_norm: rng.bf16_vec_around_one(hidden),
        layers,
    }
}

struct Arm {
    continuation: Vec<u32>,
    logits: Vec<f32>,
}

fn greedy_after_prefill(m: &mut Gemma4Wgpu, ids: &[u32], chunked: bool) -> Arm {
    m.reset();
    let (last, rest) = ids.split_last().expect("prompt");
    let mut done = 0usize;
    if chunked {
        done = m.prefill_tokens(rest).expect("prefill_tokens");
        assert!(
            done > 0 || rest.len() < m.prefill_chunk_len(),
            "chunked prefill consumed nothing on a {}-token prompt at m={}",
            rest.len(),
            m.prefill_chunk_len()
        );
    }
    for t in &rest[done..] {
        m.prefill_step(*t).expect("prefill step");
    }
    let (mut next, logits) = m.decode_step_logits(*last).expect("last prompt token");
    let mut continuation = Vec::with_capacity(CONTINUATION_TOKENS);
    for _ in 0..CONTINUATION_TOKENS {
        continuation.push(next);
        next = m.decode_step(next).expect("decode step");
    }
    Arm {
        continuation,
        logits,
    }
}

fn distinct(logits: &[f32]) -> usize {
    logits
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::HashSet<_>>()
        .len()
}

fn run_prompt_lengths_with_tolerance(m: &mut Gemma4Wgpu, vocab: usize, arm_name: &str) {
    let cm = m.prefill_chunk_len();
    for pp in [cm * 2 + cm / 2 + 1, cm + 3, cm.max(3) - 1] {
        if pp < 2 {
            continue;
        }
        let ids: Vec<u32> = (0..pp).map(|i| ((i * 7919 + 13) % vocab) as u32).collect();
        let chunked = greedy_after_prefill(m, &ids, true);
        let replay = greedy_after_prefill(m, &ids, false);
        let d = distinct(&replay.logits);
        assert!(
            d > (vocab / 4).min(1000),
            "{arm_name} pp={pp}: logits are degenerate ({d} distinct of {vocab}); the tolerance compare would be vacuous"
        );
        let peak = replay
            .logits
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let worst = chunked
            .logits
            .iter()
            .zip(replay.logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let bound = TOLERANCE_OF_PEAK_ABS_LOGIT_BECAUSE_TILED_ONLINE_SOFTMAX_REASSOCIATES * peak;
        eprintln!(
            "[g4-pff] {arm_name} pp={pp:3}: max |delta| {worst:.3e} vs bound {bound:.3e} (peak {peak:.3e}); chunked {:?} replay {:?}",
            chunked.continuation, replay.continuation
        );
        assert!(
            worst <= bound,
            "{arm_name} pp={pp}: flash-tiled chunked prefill drifted {worst:.3e} from the per-token \
             replay, over the {bound:.3e} bound (1e-4 of peak |logit| {peak:.3e}). Reassociation \
             across KV tiles explains bounded noise, not this; suspect causal bounds, ring/scale \
             indexing, or the mk scratch layout"
        );
        assert_eq!(
            chunked.continuation, replay.continuation,
            "{arm_name} pp={pp}: flash-tiled chunked prefill and per-token replay diverged at argmax level"
        );
    }
}

#[test]
fn both_flash_arm_sources_validate_under_naga_and_carry_their_entry_points() {
    for sg in [true, false] {
        let specs = nv_models::gemma4_wgpu::pf_flash_pipeline_specs_stage1_tiled_slotml_arm_matching_the_qwen_nll_signoff_then_stage2_pk_mk(sg);
        for (src, label, entry) in &specs {
            assert!(
                src.contains(&format!("fn {entry}(")),
                "{label}: entry {entry} missing from its composed source"
            );
            if !sg {
                assert!(
                    !src.contains("subgroup"),
                    "{label}: the portable arm must ship no subgroup intrinsics"
                );
            }
            let module = naga::front::wgsl::parse_str(src)
                .unwrap_or_else(|e| panic!("{label}: wgsl parse: {}", e.message()));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{label}: validate: {e}"));
        }
    }
}

fn pin_exact_e4m3_kv_decode_process_wide_because_this_suite_compares_chunk_written_to_step_written_kv(
) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::env::set_var("NV_G4_FLASH_SD", "0");
    });
    assert!(
        std::env::var("NV_G4_FLASH_SD").as_deref() == Ok("0"),
        "the 1e-4-of-peak bound isolates flash-prefill reassociation only when the KV decoder \
         is the exact e4m3 codec: the default-on shift twin decodes through the quantize-time \
         shared exponent, which differs between chunk-written and step-written KV groups, so its \
         (ppl-neutral, separately signed-off) drift would be misread here as a prefill defect"
    );
}

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn with_env<T>(env: &[(&str, &str)], f: impl FnOnce() -> T) -> T {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for (k, v) in env {
        std::env::set_var(k, v);
    }
    let out = f();
    for (k, _) in env {
        std::env::remove_var(k);
    }
    out
}

fn build_bf16_model(env: &[(&str, &str)], seed: u64) -> (Gemma4Wgpu, usize) {
    let layers = envn("NV_G4_PFF_LAYERS", 6);
    let hidden = envn("NV_G4_PFF_HIDDEN", 512);
    let inter = envn("NV_G4_PFF_INTER", 1024);
    let vocab = envn("NV_G4_PFF_VOCAB", 2048);
    let raw = config_json(layers, hidden, inter, vocab);
    let config = Gemma4Config::from_hf_json_str(&raw).expect("config");
    assert!(
        (0..config.num_hidden_layers)
            .any(|i| matches!(config.layer_kind(i), LayerType::FullAttention)),
        "the tiny config must contain a full-attention layer or the flash arm under test never runs"
    );
    let w = host_weights_all_bf16(&config, seed);
    let t = Instant::now();
    let m = with_env(env, || Gemma4Wgpu::new(config, &w, 512).expect("build"));
    eprintln!(
        "[g4-pff] built with {env:?} in {:.2}s: chunk m={}, {} prefill passes/chunk",
        t.elapsed().as_secs_f64(),
        m.prefill_chunk_len(),
        m.prefill_pass_count(),
    );
    (m, vocab)
}

#[test]
fn flash_tiled_full_attention_prefill_stays_within_1e4_of_peak_and_argmax_equal() {
    gpu_or_refuse();
    pin_exact_e4m3_kv_decode_process_wide_because_this_suite_compares_chunk_written_to_step_written_kv();
    let (m_off, _) = build_bf16_model(&[], 0xf1a5);
    assert!(
        m_off.prefill_chunk_len() >= 2,
        "chunked prefill is off on the tiny bf16 config; nothing under test"
    );
    let (mut m_on, vocab) = build_bf16_model(&[("NV_G4_WGPU_PF_FLASH", "1")], 0xf1a5);
    assert_eq!(
        m_on.prefill_chunk_len(),
        m_off.prefill_chunk_len(),
        "the flash arm must not change the chunk length"
    );
    assert!(
        m_on.prefill_pass_count() < m_off.prefill_pass_count(),
        "NV_G4_WGPU_PF_FLASH=1 recorded {} prefill passes vs {} without it; the tiled arm \
         replaces 2*m per-token dispatches per full-attention layer with 2, so an equal count \
         means it silently fell back",
        m_on.prefill_pass_count(),
        m_off.prefill_pass_count()
    );
    run_prompt_lengths_with_tolerance(&mut m_on, vocab, "flash-adapter-arm");
}

#[test]
fn flash_tiled_portable_arm_stays_within_1e4_of_peak_and_argmax_equal() {
    gpu_or_refuse();
    pin_exact_e4m3_kv_decode_process_wide_because_this_suite_compares_chunk_written_to_step_written_kv();
    let (m_off, _) = build_bf16_model(&[], 0xf1a6);
    assert!(
        m_off.prefill_chunk_len() >= 2,
        "chunked prefill is off on the tiny bf16 config; nothing under test"
    );
    let (mut m_port, vocab) = build_bf16_model(
        &[
            ("NV_G4_WGPU_PF_FLASH", "1"),
            ("NV_G4_WGPU_PF_FLASH_PORTABLE", "1"),
        ],
        0xf1a6,
    );
    assert!(
        m_port.prefill_pass_count() < m_off.prefill_pass_count(),
        "the forced-portable flash arm recorded {} prefill passes vs {} without it; equal count \
         means it silently fell back",
        m_port.prefill_pass_count(),
        m_off.prefill_pass_count()
    );
    run_prompt_lengths_with_tolerance(&mut m_port, vocab, "flash-portable-arm");
}

fn nv_ctx_tokens_last_entry_with_k_suffix(default: usize) -> usize {
    std::env::var("NV_CTX_TOKENS")
        .ok()
        .and_then(|v| {
            v.split(',')
                .map(|s| {
                    let s = s.trim();
                    let (num, mult) = match s.strip_suffix('k') {
                        Some(n) => (n, 1024usize),
                        None => (s, 1usize),
                    };
                    num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
                })
                .max()
        })
        .unwrap_or(default)
}

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_G4_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    let mut c: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("gemma4 snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file())
        .collect();
    c.sort();
    c.into_iter()
        .next()
        .expect("no gemma4 snapshot; set NV_G4_SNAPSHOT")
}

const CURVE_BIN_TOKENS: usize = 1024;

#[test]
#[ignore = "loads the 31B on wgpu; set NV_G4_PF_FLASH_CURVE_TEST=1 -- per-chunk prefill cost vs depth (max of NV_CTX_TOKENS, default 32k), one PF-CURVE line per 1024-token bin; run once plain and once with NV_G4_WGPU_PF_FLASH=1 to compare arms and extrapolate deep-context cost from the slope"]
fn gemma4_wgpu_prefill_per_chunk_cost_curve_flash_arm_from_env() {
    if std::env::var("NV_G4_PF_FLASH_CURVE_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_G4_PF_FLASH_CURVE_TEST != 1");
        return;
    }
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let flash_env = std::env::var("NV_G4_WGPU_PF_FLASH").unwrap_or_default();
    let depth = nv_ctx_tokens_last_entry_with_k_suffix(32 * 1024);
    let dir = snapshot_dir();
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("weights");
    let host =
        nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).expect("host staging");
    drop(loader);
    let mut m = Gemma4Wgpu::new(config, &host, depth + 64).expect("wgpu model");
    drop(host);
    let cm = m.prefill_chunk_len();
    assert!(
        cm >= 2,
        "chunked prefill is off on this config; there is no per-chunk cost to curve"
    );
    eprintln!(
        "PF-CURVE-HEAD flash_env={flash_env:?} chunk_m={cm} passes_per_chunk={} depth={depth} bin={CURVE_BIN_TOKENS}",
        m.prefill_pass_count()
    );
    let chunks = depth / cm;
    let mut bin_ms = 0.0f64;
    let mut bin_chunks = 0usize;
    let mut total_s = 0.0f64;
    for c in 0..chunks {
        let ids: Vec<u32> = (0..cm).map(|i| 2000 + ((c * cm + i) as u32 % 30000)).collect();
        let t0 = Instant::now();
        m.prefill_chunk(&ids)
            .unwrap_or_else(|e| panic!("prefill_chunk {c}: {e:#}"));
        ctx.poll_blocking().expect("drain the chunk before reading the clock");
        let dt = t0.elapsed().as_secs_f64();
        total_s += dt;
        bin_ms += dt * 1e3;
        bin_chunks += 1;
        let pos = (c + 1) * cm;
        if pos % CURVE_BIN_TOKENS == 0 || c + 1 == chunks {
            eprintln!(
                "PF-CURVE pos={pos} ms_per_chunk={:.3} bin_tok_s={:.1}",
                bin_ms / bin_chunks as f64,
                (bin_chunks * cm) as f64 / (bin_ms / 1e3)
            );
            bin_ms = 0.0;
            bin_chunks = 0;
        }
    }
    assert_eq!(m.current_pos(), chunks * cm, "prefill must land where it was pointed");
    eprintln!(
        "PF-CURVE-TOTAL flash_env={flash_env:?} depth={} prefill_s={total_s:.3} tok_s={:.1}",
        chunks * cm,
        (chunks * cm) as f64 / total_s
    );
}
