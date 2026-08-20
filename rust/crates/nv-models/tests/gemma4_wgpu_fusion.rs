#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip_no_require as ctx_or_skip;
use common::LcgCentered0p1Shift33 as Lcg;
use common::TINY_CONFIG;
use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_wgpu::{
    quantize_nvfp4_host, Gemma4Wgpu, HostBf16Lin, HostLayer, HostProj, HostWeights,
};
use common::EnvPins;

fn tiny_host_weights(config: &Gemma4Config, nvfp4_mlp: bool, seed: u64) -> HostWeights {
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
        let mk_proj = |rng: &mut Lcg, n: usize, k: usize, quant: bool| {
            let w = rng.bf16_vec(n * k);
            if quant {
                HostProj::Nvfp4(quantize_nvfp4_host(&w, n, k))
            } else {
                HostProj::Bf16(HostBf16Lin { w, n, k })
            }
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
            qkv: mk_proj(&mut rng, qkv_rows, hidden, false),
            o: mk_proj(&mut rng, hidden, q_dim, false),
            gate_up: mk_proj(&mut rng, 2 * inter, hidden, nvfp4_mlp),
            down: mk_proj(&mut rng, hidden, inter, nvfp4_mlp),
        });
    }
    HostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden),
        final_norm: rng.bf16_vec_around_one(hidden),
        layers,
    }
}

const STEPS: [u32; 12] = [7, 300, 42, 511, 0, 13, 99, 250, 8, 9, 10, 11];

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn run_cfg(nvfp4_mlp: bool, seed: u64, fuse: u32, attn_fp8: bool) -> (usize, Vec<u32>) {
    let config = Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap();
    let weights = tiny_host_weights(&config, nvfp4_mlp, seed);
    let fuse_s = fuse.to_string();
    let pins = EnvPins::pin(&[
        ("NV_WGPU_FUSE", Some(fuse_s.as_str())),
        ("NV_G4_WGPU_W8_FFN", Some("0")),
    ]);
    nv_models::gemma4_wgpu::set_attn_variant(Some(nv_models::gemma4_wgpu::AttnVariant {
        on: attn_fp8,
        ..nv_models::gemma4_wgpu::ATTN_VARIANT_DEFAULT
    }));
    let m = Gemma4Wgpu::new(config, &weights, 64);
    nv_models::gemma4_wgpu::set_attn_variant(None);
    drop(pins);
    let mut m = m.unwrap();
    let passes = m.pass_count();
    let mut bits = Vec::new();
    for t in STEPS {
        let (tok, logits) = m.decode_step_logits(t).unwrap();
        bits.push(tok);
        bits.extend(logits.iter().map(|v| v.to_bits()));
    }
    (passes, bits)
}

fn run_variant(nvfp4_mlp: bool, seed: u64) -> (usize, Vec<u32>) {
    run_cfg(nvfp4_mlp, seed, 7, false)
}

fn golden_path() -> std::path::PathBuf {
    match std::env::var("NV_FUSION_GOLDEN") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => std::env::temp_dir().join("g4w-fusion-golden.bin"),
    }
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn fused_matches_unfused_bit_identical() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    for (nvfp4_mlp, attn_fp8) in [(false, false), (true, false), (true, true)] {
        let (p_off, off) = run_cfg(nvfp4_mlp, 0x1dea, 0, attn_fp8);
        let (p_on, on) = run_cfg(nvfp4_mlp, 0x1dea, 7, attn_fp8);
        let diff = off.iter().zip(on.iter()).filter(|(x, y)| x != y).count();
        eprintln!(
            "nvfp4_mlp={nvfp4_mlp} attn_fp8={attn_fp8}: passes/token unfused {p_off} -> fused {p_on} ({:+}, {:.1}%), diff words {diff}/{}",
            p_on as i64 - p_off as i64,
            100.0 * (p_on as f64 - p_off as f64) / p_off as f64,
            off.len()
        );
        assert!(
            p_on < p_off,
            "fusion must cut dispatches: {p_off} -> {p_on}"
        );
        assert_eq!(
            diff, 0,
            "fused decode must be bit-identical to unfused (nvfp4_mlp={nvfp4_mlp} attn_fp8={attn_fp8})"
        );
    }
}

#[test]
fn each_fusion_alone_is_bit_identical_and_cuts_its_own_share() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap();
    let layers = config.num_hidden_layers;
    let (p_off, off) = run_cfg(true, 0x51ce, 0, false);
    for (mask, name, per_layer) in [
        (1u32, "head-prep", 6usize),
        (2, "norm-res-norm", 1),
        (4, "norm-add-norm", 2),
    ] {
        let (p, bits) = run_cfg(true, 0x51ce, mask, false);
        let diff = off.iter().zip(bits.iter()).filter(|(x, y)| x != y).count();
        eprintln!("mask={mask} ({name}): {p_off} -> {p} passes, diff words {diff}");
        assert_eq!(
            p_off - p,
            per_layer * layers,
            "{name} must remove {per_layer} dispatches per layer"
        );
        assert_eq!(diff, 0, "{name} alone must be bit-identical to unfused");
    }
}

#[test]
fn fusion_saves_nine_dispatches_per_layer() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap();
    let (p_off, _) = run_cfg(true, 0x1dea, 0, false);
    let (p_on, _) = run_cfg(true, 0x1dea, 7, false);
    let layers = config.num_hidden_layers;
    eprintln!(
        "layers {layers}: unfused {p_off}, fused {p_on}, saved {} = {} per layer",
        p_off - p_on,
        (p_off - p_on) as f64 / layers as f64
    );
    assert_eq!(
        p_off - p_on,
        9 * layers,
        "head-prep (-6), norm-res-norm (-1) and norm-add-norm (-2) must save 9 dispatches per layer"
    );
}

#[test]
fn logits_bit_identical_to_prefusion_golden() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let (p_bf16, a) = run_variant(false, 0x5eed);
    let (p_nvfp4, b) = run_variant(true, 0x5eed);
    eprintln!("passes per step: bf16-mlp {p_bf16}, nvfp4-mlp {p_nvfp4}");
    let mut all = a;
    all.extend(&b);
    let path = golden_path();
    if !path.exists() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, words_to_bytes(&all)).unwrap();
        eprintln!("golden written: {} ({} words)", path.display(), all.len());
        return;
    }
    let golden = bytes_to_words(&std::fs::read(&path).unwrap());
    assert_eq!(golden.len(), all.len(), "golden length mismatch");
    let diff = golden
        .iter()
        .zip(all.iter())
        .filter(|(g, n)| g != n)
        .count();
    if diff != 0 {
        let per_variant = all.len() / 2;
        let per_step = per_variant / STEPS.len();
        let mut shown = 0;
        for (i, (g, n)) in golden.iter().zip(all.iter()).enumerate() {
            if g != n {
                let variant = if i < per_variant {
                    "bf16-mlp"
                } else {
                    "nvfp4-mlp"
                };
                let j = i % per_variant;
                let (step, word) = (j / per_step, j % per_step);
                if shown < 12 {
                    eprintln!(
                        "  golden diff at {variant} step {step} word {word}: golden=0x{g:08x} now=0x{n:08x}"
                    );
                }
                shown += 1;
            }
        }
        let a_diff = golden[..per_variant]
            .iter()
            .zip(&all[..per_variant])
            .filter(|(g, n)| g != n)
            .count();
        eprintln!(
            "  golden diff split: bf16-mlp {a_diff}, nvfp4-mlp {} (of {per_variant} words each; golden mtime {:?})",
            diff - a_diff,
            std::fs::metadata(&path).and_then(|m| m.modified()).ok()
        );
    }
    assert_eq!(
        diff,
        0,
        "logits+argmax must be bit-identical to the pre-fusion golden ({} of {} words differ)",
        diff,
        all.len()
    );
    eprintln!(
        "golden compare: {} words bit-identical across bf16-mlp and nvfp4-mlp variants",
        all.len()
    );
}

#[test]
fn nvfp4_ffn_boots_with_fp8_attention_opted_out() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    assert!(
        std::env::var("NV_G4_WGPU_W8_FFN").is_err(),
        "NV_G4_WGPU_W8_FFN is set (leaked pin or ambient shell); this test \
         exercises the shipped \"all\" default and would be vacuous"
    );
    let config = Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap();
    let weights = tiny_host_weights(&config, true, 0x1dea);
    nv_models::gemma4_wgpu::set_attn_variant(Some(nv_models::gemma4_wgpu::AttnVariant {
        on: false,
        ..nv_models::gemma4_wgpu::ATTN_VARIANT_DEFAULT
    }));
    let m = Gemma4Wgpu::new(config, &weights, 64);
    nv_models::gemma4_wgpu::set_attn_variant(None);
    assert!(
        m.is_ok(),
        "A bf16-attn variant with nvfp4 FFN weights must boot at the shipped W8-FFN default. \
         It does not because w8_ffn_mode() (default \"all\" since 2026-08-09) converts nvfp4 \
         gate_up/down to HostProj::Fp8 while build_pipelines (gemma4_wgpu.rs:3490) gates \
         Fp8Pipelines on attn_fp8 alone, so upload_proj hits 'fp8 projection uploaded without \
         fp8 pipelines'. Fix belongs in gemma4_wgpu.rs: build Fp8Pipelines when w8_ffn_mode() \
         will convert any projection, not only when attn_fp8 is on. Err: {:?}",
        m.err()
    );
}

#[test]
fn bf16_checkpoint_with_attn_fp8_off_builds_no_q8_pipelines() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap();
    let weights = tiny_host_weights(&config, false, 0x1dea);
    let log = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("g4w-q8-gate-log-{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&log);
    let pins = EnvPins::pin(&[
        ("NV_G4_WGPU_W8_FFN", Some("all")),
        ("NV_WGPU_PIPELINE_LOG", Some(log.to_str().unwrap())),
    ]);
    nv_models::gemma4_wgpu::set_attn_variant(Some(nv_models::gemma4_wgpu::AttnVariant {
        on: false,
        ..nv_models::gemma4_wgpu::ATTN_VARIANT_DEFAULT
    }));
    let m = Gemma4Wgpu::new(config, &weights, 64);
    nv_models::gemma4_wgpu::set_attn_variant(None);
    drop(pins);
    assert!(
        m.is_ok(),
        "bf16 checkpoint with the attn variant off and NV_G4_WGPU_W8_FFN=all must boot: {:?}",
        m.err()
    );
    let logged = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_file(&log);
    assert!(
        logged.contains("[pipeline] g4w-gemv8-pk:"),
        "NV_WGPU_PIPELINE_LOG must have captured this boot's pipeline requests"
    );
    for q8 in [
        "g4w-gemv-fp8-pk",
        "g4w-gemv-int8-pk",
        "g4w-gemm-fp8-mk-pk",
        "g4w-gemm-int8-mk-pk",
    ] {
        assert!(
            !logged.contains(q8),
            "a bf16 checkpoint with fp8 attention opted out converts nothing to \
             HostProj::Fp8 (W8-FFN only converts nvfp4 sources), so the projection-derived \
             q8 gate must not build {q8}. The pre-#151 hand-maintained boolean over-built \
             these from NV_G4_WGPU_W8_FFN=all alone."
        );
    }
    eprintln!(
        "q8 gate derived from projection set: boot OK, {} pipeline requests, no q8 labels",
        logged.lines().count()
    );
}

#[test]
fn determinism_two_runs_bit_identical() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let (_, a) = run_variant(true, 0xf00d);
    let (_, b) = run_variant(true, 0xf00d);
    assert_eq!(
        a.iter().zip(b.iter()).filter(|(x, y)| x != y).count(),
        0,
        "decode must be bit-deterministic run-to-run"
    );
    eprintln!(
        "determinism: {} words bit-identical across two runs",
        a.len()
    );
}

#[test]
fn nvfp4_tree_and_sg_paths_bit_identical() {
    let _g = env_lock();
    let Some(ctx) = ctx_or_skip() else { return };
    if !nv_kernels::wgpu_backend::kernels::gemv_nvfp4::subgroup_ok(ctx) {
        eprintln!("skipping: adapter has no fixed-32 subgroups, tree==sg trivially");
        return;
    }
    std::env::set_var("NV_WGPU_NVFP4_TREE", "1");
    let (p_tree, a) = run_variant(true, 0xbeef);
    std::env::remove_var("NV_WGPU_NVFP4_TREE");
    let (p_sg, b) = run_variant(true, 0xbeef);
    eprintln!("passes: tree {p_tree}, sg {p_sg}");
    let diff = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    assert_eq!(
        diff, 0,
        "tree and sg nvfp4 decode paths must be bit-identical ({diff} words differ)"
    );
    eprintln!("tree-vs-sg: {} words bit-identical", a.len());
}

fn gpu_util() -> Option<u32> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

fn wait_idle(tag: &str) {
    let t0 = std::time::Instant::now();
    let mut streak = 0;
    loop {
        match gpu_util() {
            Some(u) if u <= 2 => streak += 1,
            Some(_) => streak = 0,
            None => return,
        }
        if streak >= 2 {
            eprintln!(
                "{tag}: gpu idle after {:.0}s wait",
                t0.elapsed().as_secs_f64()
            );
            return;
        }
        if t0.elapsed().as_secs_f64() > 600.0 {
            eprintln!("{tag}: WARNING gpu never went idle within 600s; measuring anyway");
            return;
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}

fn occupancy_note() -> String {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "nvidia-smi unavailable".to_string(),
    }
}

struct MsStats {
    median: f64,
    min: f64,
    p90: f64,
    n: usize,
}

impl MsStats {
    fn of(v: &[f64]) -> Self {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self {
            median: s[s.len() / 2],
            min: s[0],
            p90: s[(s.len() * 9 / 10).min(s.len() - 1)],
            n: s.len(),
        }
    }
}

impl std::fmt::Display for MsStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "median {:.2} min {:.2} p90 {:.2} (n={})",
            self.median, self.min, self.p90, self.n
        )
    }
}

fn hub_snapshot() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap();
    let base = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    std::fs::read_dir(&base)
        .expect("hub snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists())
        .expect("no snapshot with config.json")
}

#[test]
#[ignore]
fn real_gemma4_31b_fusion_ab_interleaved() {
    let _g = env_lock();
    if std::env::var("NV_GEMMA4_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_GEMMA4_WGPU_TEST=1 to run");
        return;
    }
    let Some(_ctx) = ctx_or_skip() else { return };
    let dir = hub_snapshot();
    eprintln!("loading Gemma4 config from {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).unwrap();
    drop(loader);
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let enc = tokenizer.encode("The capital of France is", false).unwrap();
    let mut prompt: Vec<u32> = vec![2];
    prompt.extend(enc.get_ids());

    let arms: Vec<(&str, u32, bool)> = match std::env::var("NV_FUSE_AB_ARMS") {
        Ok(_) => vec![
            ("fuse=0 bf16-attn", 0, false),
            ("fuse=1 head-prep", 1, false),
            ("fuse=2 norm-res-norm", 2, false),
            ("fuse=4 norm-add-norm", 4, false),
            ("fuse=7 bf16-attn", 7, false),
            ("fuse=7 fp8-attn", 7, true),
        ],
        Err(_) => vec![
            ("fuse=0 bf16-attn", 0, false),
            ("fuse=0 fp8-attn", 0, true),
            ("fuse=7 bf16-attn", 7, false),
            ("fuse=7 fp8-attn", 7, true),
        ],
    };
    let rounds: usize = std::env::var("NV_FUSE_AB_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let warmup = 8usize;
    let timed = 24usize;

    let mut times: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut passes = vec![0usize; arms.len()];
    let mut ids: Vec<Vec<u32>> = vec![Vec::new(); arms.len()];

    for round in 0..rounds {
        for (ai, (name, mask, fp8)) in arms.iter().enumerate() {
            std::env::set_var("NV_WGPU_FUSE", mask.to_string());
            nv_models::gemma4_wgpu::set_attn_variant(Some(nv_models::gemma4_wgpu::AttnVariant {
                on: *fp8,
                ..nv_models::gemma4_wgpu::ATTN_VARIANT_DEFAULT
            }));
            let built = Gemma4Wgpu::new(config.clone(), &host, 4096);
            std::env::remove_var("NV_WGPU_FUSE");
            nv_models::gemma4_wgpu::set_attn_variant(None);
            let mut m = built.unwrap();
            passes[ai] = m.pass_count();
            let mut last = 0u32;
            for t in &prompt {
                last = m.decode_step(*t).unwrap();
            }
            let mut got = vec![last];
            for _ in 0..warmup {
                last = m.decode_step(last).unwrap();
                got.push(last);
            }
            wait_idle(name);
            for _ in 0..timed {
                let t0 = std::time::Instant::now();
                last = m.decode_step(last).unwrap();
                times[ai].push(t0.elapsed().as_secs_f64() * 1000.0);
                got.push(last);
            }
            if round == 0 {
                ids[ai] = got;
            }
            eprintln!(
                "round {round} [{name}] {} passes, {}",
                passes[ai],
                MsStats::of(&times[ai][times[ai].len() - timed..])
            );
            drop(m);
        }
    }

    eprintln!("--- interleaved A/B summary ({rounds} rounds x {timed} timed tokens) ---");
    eprintln!("nvidia-smi: {}", occupancy_note());
    let base = MsStats::of(&times[0]).median;
    for (ai, (name, mask, fp8)) in arms.iter().enumerate() {
        let st = MsStats::of(&times[ai]);
        eprintln!(
            "{name:24} mask={mask} fp8={fp8} passes={:5} {st} -> {:+.1}% vs arm0",
            passes[ai],
            100.0 * (st.median - base) / base
        );
    }
    for (ai, (name, _, _)) in arms.iter().enumerate().skip(1) {
        let agree = ids[0]
            .iter()
            .zip(ids[ai].iter())
            .filter(|(a, b)| a == b)
            .count();
        eprintln!(
            "[{name}] argmax agreement vs arm0: {agree}/{} ; first divergence at {:?}",
            ids[0].len(),
            ids[0].iter().zip(ids[ai].iter()).position(|(a, b)| a != b)
        );
        eprintln!(
            "[{name}] text: {:?}",
            tokenizer.decode(&ids[ai], false).unwrap()
        );
    }
    eprintln!(
        "[arm0] text: {:?}",
        tokenizer.decode(&ids[0], false).unwrap()
    );
    assert!(times.iter().all(|t| !t.is_empty()));
}

#[test]
#[ignore]
fn real_gemma4_31b_fused_decode_ms_per_token() {
    if std::env::var("NV_GEMMA4_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_GEMMA4_WGPU_TEST=1 to run");
        return;
    }
    let _g = env_lock();
    let Some(_ctx) = ctx_or_skip() else { return };
    let home = std::env::var("HOME").unwrap();
    let base = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    let dir = std::fs::read_dir(&base)
        .expect("hub snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists())
        .expect("no snapshot with config.json");
    eprintln!("loading Gemma4 config from {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let t_load = std::time::Instant::now();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).unwrap();
    eprintln!(
        "host weight staging took {:.1}s",
        t_load.elapsed().as_secs_f64()
    );

    let t_up = std::time::Instant::now();
    let mut m = Gemma4Wgpu::new(config, &host, 4096).unwrap();
    drop(host);
    eprintln!(
        "device upload + pass build took {:.1}s ({} passes per step)",
        t_up.elapsed().as_secs_f64(),
        m.pass_count()
    );

    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let enc = tokenizer.encode("The capital of France is", false).unwrap();
    let mut prompt: Vec<u32> = vec![2];
    prompt.extend(enc.get_ids());
    eprintln!("prompt ids: {prompt:?}");
    let mut last = 0u32;
    for t in &prompt {
        last = m.decode_step(*t).unwrap();
    }
    eprintln!(
        "prompt fed ({} tokens), first generated token id {last}",
        prompt.len()
    );

    let warmup = 8usize;
    let mut warm = Vec::with_capacity(warmup);
    for _ in 0..warmup {
        last = m.decode_step(last).unwrap();
        warm.push(last);
    }
    wait_idle("pre-window");
    eprintln!("nvidia-smi before window: {}", occupancy_note());
    let timed = 32usize;
    let mut generated = Vec::with_capacity(timed);
    let t0 = std::time::Instant::now();
    for _ in 0..timed {
        last = m.decode_step(last).unwrap();
        generated.push(last);
    }
    let dt = t0.elapsed();
    eprintln!("nvidia-smi after window: {}", occupancy_note());
    let ms_per_tok = dt.as_secs_f64() * 1000.0 / timed as f64;
    let mut all: Vec<u32> = warm.clone();
    all.extend(&generated);
    eprintln!("generated ids: {all:?}");
    eprintln!(
        "generated text: {:?}",
        tokenizer.decode(&all, false).unwrap()
    );
    eprintln!(
        "wgpu Gemma4-31B NVFP4 decode: {timed} tokens in {:.3}s -> {ms_per_tok:.2} ms/token ({:.2} tok/s);
         same-session CONTAMINATED interleaved medians 2026-08-06: fuse=0/bf16-attn 38.19, fuse=0/fp8-attn 33.61, \
         fuse=7/bf16-attn 36.61, fuse=7/fp8-attn 32.87 (n=3 each); CUDA graphed 27.63",
        dt.as_secs_f64(),
        1000.0 / ms_per_tok
    );
    print_wgpu_profile_table_when_enabled();
    assert!(generated
        .iter()
        .all(|t| (*t as usize) < m.config().vocab_size));
}

fn print_wgpu_profile_table_when_enabled() {
    use nv_kernels::wgpu_backend::dispatch::profile;
    if !profile::enabled() {
        return;
    }
    let mut rows = profile::report();
    rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let mut total = 0.0;
    for (label, count, ns) in &rows {
        total += ns;
    }
    for (label, count, ns) in rows.iter().take(20) {
        eprintln!(
            "[prof] {label} n={count} total={:.3}ms share={:.1}% avg={:.2}us",
            ns / 1.0e6,
            100.0 * ns / total.max(1.0),
            ns / 1.0e3 / (*count).max(1) as f64
        );
    }
    eprintln!("[prof] TOTAL {:.3}ms over {} labels", total / 1.0e6, rows.len());
}
