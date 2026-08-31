#![cfg(feature = "wgpu")]

mod hub_dirs;

use std::path::PathBuf;
use std::time::Instant;

use nv_models::qwen3_5_dense_wgpu::{Q3dLora, Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use nv_specdecode::qwen38_mtp::{
    mtp_chain_depth_from_env, mtp_verify_replay_selected_from_env, run_mtp_verify_round,
};

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const CHECKPOINT_LABEL: &str = "unsloth/Qwen3.8-27B-NVFP4";

const TIMED_EMITTED_TOKENS_256_THE_SCOREBOARD_GENERATION_LENGTH: usize = 256;
const PLAIN_TIMED_STEPS_64_MATCHES_THE_CTX_SCALING_LADDERS: usize = 64;
const INTERLEAVED_PAIRS_DEFAULT_3_PER_POINT_PER_THE_QUALITY_VERDICT_METHOD: usize = 3;
const KV_SLOT_HEADROOM_1024_COVERS_ANCHOR_WARMUP_TIMED_OVERSHOOT_AND_THE_512_ROW_VERIFY_GUARD:
    usize = 1024;

const PLATEAU_WINDOW_4_STEPS_MEDIAN_VS_THE_PREVIOUS_4: usize = 4;
const PLATEAU_FLOOR_8_STEPS_THE_OLD_FIXED_WARMUP_THAT_DID_NOT_RAMP_CLOCKS: usize = 8;
const PLATEAU_RATIO_LAST_WINDOW_MEDIAN_WITHIN_10_PCT_OF_PREVIOUS: f64 = 0.10;
const PLATEAU_CAP_15_SECS_SO_A_SLOW_DEEP_RUNG_CANNOT_WARM_FOREVER: f64 = 15.0;
const PLAIN_ARM_WARMUP_CAP_200_STEPS_MATCHES_THE_CTX_SCALING_LADDERS: usize = 200;
const MTP_ARM_WARMUP_CAP_40_ROUNDS_BOUNDS_DEPTH_DRIFT_OF_THE_TIMED_WINDOW: usize = 40;

const KL_PROMPT_TOKENS_128_TILED_PARAGRAPH: usize = 128;
const KL_TIMED_STEPS_48_TEACHER_FORCED_ON_THE_BASE_STREAM: usize = 48;

const SYNTH_RANK_8_THE_TASK_CONTRACT_RANK: usize = 8;
const SYNTH_ALPHA_16_SO_SCALING_IS_2_MATCHING_COMMON_PEFT_DEFAULTS: f64 = 16.0;
const SYNTH_A_SCALE_0_02_KAIMING_LIKE_MAGNITUDE: f32 = 0.02;
const SYNTH_B_SCALE_DEFAULT_5_3E4_TARGETS_DELTA_W_STD_6E5_VIA_ALPHA_OVER_R_TIMES_AB_SQRT_R: f32 =
    5.3e-4;
const SYNTH_SEED: u64 = 0x9a38_ab01;

const REAL_PREFILL_ACCEPT_REGIME_MAX_DEPTH_512_DEEPER_PREFILL_IS_OUT_OF_BUDGET: usize = 512;

const TILED_PROMPT_PARAGRAPH_REPETITION_LANDS_IN_THE_HIGH_ACCEPT_DOC_REGIME_SO_ACCEPTANCE_HERE_IS_AN_UPPER_BRACKET_NOT_A_CHAT_NUMBER:
    &str = "Polynesian wayfinders crossed thousands of miles of open ocean by reading star \
     paths, swell patterns, and the flight of birds, long before the magnetic compass reached \
     Europe. Their voyaging canoes carried taro, breadfruit, and chickens between island chains \
     separated by weeks of sailing. ";

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn signed(&mut self, scale: f32) -> f32 {
        if self.next() & 1 == 0 {
            scale
        } else {
            -scale
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .map(|v| v.parse::<usize>().unwrap_or_else(|_| panic!("{key}={v} is not a usize")))
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .map(|v| v.parse::<f32>().unwrap_or_else(|_| panic!("{key}={v} is not an f32")))
        .unwrap_or(default)
}

fn require_gate(gate: &str) {
    if std::env::var(gate).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {gate}=1");
    }
}

fn single_depth_from_env_because_of_the_96_residual_race() -> usize {
    let raw = std::env::var("NV_CTX_TOKENS").unwrap_or_else(|_| "256".into());
    let vals: Vec<usize> = raw
        .split(',')
        .map(|s| {
            let s = s.trim();
            let (num, mult) = match s.strip_suffix(['k', 'K']) {
                Some(n) => (n, 1024usize),
                None => (s, 1usize),
            };
            num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
        })
        .collect();
    assert_eq!(
        vals.len(),
        1,
        "run ONE depth per process via NV_CTX_TOKENS (the #96 residual race), got {raw}"
    );
    vals[0]
}

fn snapshot_dir() -> PathBuf {
    match std::env::var("NV_QWEN38_DIR") {
        Ok(d) if !d.is_empty() => {
            let p = PathBuf::from(&d);
            assert!(p.is_dir(), "NV_QWEN38_DIR={d} is not a directory");
            p
        }
        _ => hub_dirs::snapshot(
            NVFP4_REPO,
            &["config.json", "tokenizer.json", "model_mtp.safetensors"],
        )
        .unwrap_or_else(|| {
            panic!(
                "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: no hydrated {NVFP4_REPO} \
                 snapshot carrying the MTP shard; set NV_QWEN38_DIR"
            )
        }),
    }
}

fn snapshot_rev_label(dir: &std::path::Path) -> String {
    dir.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn write_safetensors_f32(
    path: &std::path::Path,
    tensors: &mut [(String, Vec<usize>, Vec<f32>)],
) {
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape, vals) in tensors.iter() {
        let start = data.len();
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let mut hdr = serde_json::to_string(&serde_json::Value::Object(header))
        .unwrap()
        .into_bytes();
    while hdr.len() % 8 != 0 {
        hdr.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + hdr.len() + data.len());
    out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr);
    out.extend_from_slice(&data);
    std::fs::write(path, out).expect("write adapter_model.safetensors");
}

fn write_rank8_attn_mlp_adapter_no_mtp_modules(
    cfg: &Qwen3_5DenseConfig,
    b_scale: f32,
) -> PathBuf {
    use nv_models::qwen3_5_moe::LayerType;
    let dir = std::env::var("NV_LORA_AB_ADAPTER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::temp_dir().join(format!(
                "nv-lora-q38-ab-r{SYNTH_RANK_8_THE_TASK_CONTRACT_RANK}-seed{SYNTH_SEED:x}"
            ))
        });
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create adapter dir");
    let rank = SYNTH_RANK_8_THE_TASK_CONTRACT_RANK;
    let a_scale = SYNTH_A_SCALE_0_02_KAIMING_LIKE_MAGNITUDE;
    let mut rng = Rng(SYNTH_SEED);
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let n_h = cfg.num_attention_heads;
    let kv = cfg.num_key_value_heads * hd;
    let q_out = n_h * hd * if cfg.attn_output_gate { 2 } else { 1 };
    let n_vh = cfg.linear_num_value_heads;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = n_vh * cfg.linear_value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let emit = |rng: &mut Rng, tensors: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
                    name: String, k: usize, n: usize| {
        let a: Vec<f32> = (0..rank * k).map(|_| rng.signed(a_scale)).collect();
        let b: Vec<f32> = (0..n * rank).map(|_| rng.signed(b_scale)).collect();
        tensors.push((
            format!("base_model.model.{name}.lora_A.weight"),
            vec![rank, k],
            a,
        ));
        tensors.push((
            format!("base_model.model.{name}.lora_B.weight"),
            vec![n, rank],
            b,
        ));
    };
    for li in 0..cfg.num_hidden_layers {
        let p = format!("model.language_model.layers.{li}");
        if matches!(cfg.layer_types[li], LayerType::FullAttention) {
            emit(&mut rng, &mut tensors, format!("{p}.self_attn.q_proj"), hidden, q_out);
            emit(&mut rng, &mut tensors, format!("{p}.self_attn.k_proj"), hidden, kv);
            emit(&mut rng, &mut tensors, format!("{p}.self_attn.v_proj"), hidden, kv);
            emit(&mut rng, &mut tensors, format!("{p}.self_attn.o_proj"), n_h * hd, hidden);
        } else {
            emit(&mut rng, &mut tensors, format!("{p}.linear_attn.in_proj_qkv"), hidden, conv_dim);
            emit(&mut rng, &mut tensors, format!("{p}.linear_attn.in_proj_z"), hidden, value_dim);
            emit(&mut rng, &mut tensors, format!("{p}.linear_attn.in_proj_a"), hidden, n_vh);
            emit(&mut rng, &mut tensors, format!("{p}.linear_attn.in_proj_b"), hidden, n_vh);
            emit(&mut rng, &mut tensors, format!("{p}.linear_attn.out_proj"), value_dim, hidden);
        }
        emit(&mut rng, &mut tensors, format!("{p}.mlp.gate_proj"), hidden, inter);
        emit(&mut rng, &mut tensors, format!("{p}.mlp.up_proj"), hidden, inter);
        emit(&mut rng, &mut tensors, format!("{p}.mlp.down_proj"), inter, hidden);
    }
    let peft = serde_json::json!({
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": SYNTH_ALPHA_16_SO_SCALING_IS_2_MATCHING_COMMON_PEFT_DEFAULTS,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": false,
        "use_dora": false,
        "modules_to_save": serde_json::Value::Null,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                           "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                           "out_proj"],
    });
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&peft).unwrap(),
    )
    .expect("write adapter_config.json");
    write_safetensors_f32(&dir.join("adapter_model.safetensors"), &mut tensors);
    dir
}

fn median_of(ms: &[f64]) -> f64 {
    let mut sorted = ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

fn warmup_until_step_wall_plateaus(
    mut one_step_returning_tokens_emitted: impl FnMut() -> usize,
    step_cap: usize,
) -> (usize, usize) {
    let started = Instant::now();
    let mut ms: Vec<f64> = Vec::new();
    let mut tokens = 0usize;
    loop {
        let t0 = Instant::now();
        tokens += one_step_returning_tokens_emitted();
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
        let n = ms.len();
        let w = PLATEAU_WINDOW_4_STEPS_MEDIAN_VS_THE_PREVIOUS_4;
        if n >= PLATEAU_FLOOR_8_STEPS_THE_OLD_FIXED_WARMUP_THAT_DID_NOT_RAMP_CLOCKS && n >= 2 * w {
            let last = median_of(&ms[n - w..]);
            let prev = median_of(&ms[n - 2 * w..n - w]);
            if prev > 0.0
                && (last - prev).abs()
                    <= PLATEAU_RATIO_LAST_WINDOW_MEDIAN_WITHIN_10_PCT_OF_PREVIOUS * prev
            {
                return (n, tokens);
            }
        }
        if n >= step_cap
            || started.elapsed().as_secs_f64()
                >= PLATEAU_CAP_15_SECS_SO_A_SLOW_DEEP_RUNG_CANNOT_WARM_FOREVER
        {
            return (n, tokens);
        }
    }
}

fn tiled_prompt_ids(dir: &std::path::Path, len: usize) -> Vec<u32> {
    let tok =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let seed: Vec<u32> = tok
        .encode(
            TILED_PROMPT_PARAGRAPH_REPETITION_LANDS_IN_THE_HIGH_ACCEPT_DOC_REGIME_SO_ACCEPTANCE_HERE_IS_AN_UPPER_BRACKET_NOT_A_CHAT_NUMBER,
            false,
        )
        .expect("encode paragraph")
        .get_ids()
        .to_vec();
    assert!(!seed.is_empty(), "paragraph tokenized to nothing");
    (0..len).map(|i| seed[i % seed.len()]).collect()
}

fn prefill_like_serving(m: &mut Qwen3_5DenseWgpu, prompt: &[u32]) -> u32 {
    let (last, rest) = prompt.split_last().expect("non-empty prompt");
    let done = m.prefill_tokens(rest).expect("chunked prefill");
    for &t in &rest[done..] {
        m.prefill_step(t).expect("prefill step");
    }
    m.decode_step(*last).expect("anchor decode step")
}

fn synthfill_then_anchor(
    ctx: &nv_kernels::wgpu_backend::WgpuContext,
    gpu: &mut Qwen3_5DenseWgpu,
    depth: usize,
) -> u32 {
    gpu.reset().expect("reset before synthetic fill");
    gpu.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(depth)
        .unwrap_or_else(|e| panic!("synthetic state fill at depth {depth}: {e:#}"));
    ctx.poll_blocking().expect("drain synthetic fill writes");
    assert_eq!(gpu.current_pos(), depth, "synthetic fill must land at the requested depth");
    assert!(gpu.mtp_active(), "mtp head must be attached on every arm of this A/B");
    assert_eq!(
        gpu.mtp_len(),
        depth,
        "synthetic fill must leave the drafter KV length at the trunk depth"
    );
    gpu.decode_step(2000).expect("anchor decode step")
}

struct MtpWindow {
    rounds: usize,
    drafted: usize,
    accepted: usize,
    emitted: usize,
    warm_emitted: usize,
    wall_s: f64,
    round_ms_median: f64,
}

fn mtp_rounds_window(
    gpu: &mut Qwen3_5DenseWgpu,
    anchor: u32,
    k: usize,
    replay: bool,
    max_seq: usize,
    timed_target: usize,
    warm_cap: usize,
) -> MtpWindow {
    let mut last = anchor;
    let pos0 = gpu.current_pos();
    let mut warm_tally = (0usize, 0usize, 0usize);
    let round = |gpu: &mut Qwen3_5DenseWgpu,
                     last: &mut u32,
                     tally: &mut (usize, usize, usize)|
     -> usize {
        let rows = gpu.verify_max_rows();
        assert!(rows >= 2, "the M-row verify graph is absent");
        assert!(
            gpu.current_pos() + rows.max(gpu.prefill_chunk_len()) <= max_seq,
            "kv headroom exhausted mid-window at pos {}",
            gpu.current_pos()
        );
        let want = k.min(rows - 1);
        let drafts = gpu.mtp_draft_round(*last, want).expect("mtp draft round");
        let r = run_mtp_verify_round(gpu, *last, &drafts, replay).expect("mtp verify round");
        gpu.mtp_post_verify(&r.batch[1..r.accept.commit_len])
            .expect("mtp post verify");
        tally.0 += 1;
        tally.1 += drafts.len();
        tally.2 += r.accept.draft_accepted;
        *last = *r.emitted.last().expect("a round always emits the bonus token");
        r.emitted.len()
    };
    let (_warm_rounds, warm_emitted) = warmup_until_step_wall_plateaus(
        || round(gpu, &mut last, &mut warm_tally),
        warm_cap,
    );
    let mut timed = (0usize, 0usize, 0usize);
    let mut round_ms: Vec<f64> = Vec::new();
    let mut emitted = 0usize;
    let t0 = Instant::now();
    while emitted < timed_target {
        let r0 = Instant::now();
        emitted += round(gpu, &mut last, &mut timed);
        round_ms.push(r0.elapsed().as_secs_f64() * 1e3);
    }
    let wall_s = t0.elapsed().as_secs_f64();
    assert_eq!(
        gpu.current_pos(),
        pos0 + warm_emitted + emitted,
        "trunk pos must advance by exactly every emitted token"
    );
    assert_eq!(
        gpu.mtp_len(),
        gpu.current_pos(),
        "round catch-up must leave the drafter KV at the trunk committed length"
    );
    assert!(timed.0 > 0 && timed.1 > 0, "degenerate mtp window: no rounds or drafts");
    assert!(
        timed.2 > 0,
        "zero acceptance over {} drafts: effective tok/s would be plain decode wearing an mtp \
         label; refuse the row",
        timed.1
    );
    MtpWindow {
        rounds: timed.0,
        drafted: timed.1,
        accepted: timed.2,
        emitted,
        warm_emitted,
        wall_s,
        round_ms_median: median_of(&round_ms),
    }
}

fn plain_decode_window(gpu: &mut Qwen3_5DenseWgpu, anchor: u32, timed_steps: usize) -> (f64, usize) {
    let mut last = anchor;
    let (warm_steps, _warm_tokens) = warmup_until_step_wall_plateaus(
        || {
            last = gpu.decode_step(last).expect("warm decode step");
            1
        },
        PLAIN_ARM_WARMUP_CAP_200_STEPS_MATCHES_THE_CTX_SCALING_LADDERS,
    );
    let mut step_ms: Vec<f64> = Vec::new();
    for _ in 0..timed_steps {
        let t0 = Instant::now();
        last = gpu.decode_step(last).expect("timed decode step");
        step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    (median_of(&step_ms), warm_steps)
}

fn log_softmax_f64(row: &[f32]) -> Vec<f64> {
    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut lse = 0f64;
    for &v in row {
        lse += ((v as f64) - mx).exp();
    }
    let lse = mx + lse.ln();
    row.iter().map(|&v| (v as f64) - lse).collect()
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B dense hybrid TWICE on wgpu (base engine + rank-8 \
            synthetic-adapter engine, both with the shipped MTP head attached); set \
            NV_QWEN38_WGPU_TEST=1 -- interleaved same-session base/adapter pairs measuring (a) \
            plain decode ms/token and MTP effective tok/s at the NV_CTX_TOKENS depth over the \
            same synthetic state fill the ctx-scaling ladders use, (b) MTP acceptance \
            (accept_rate, tokens/round) base vs adapter, plus a real-chunked-prefill acceptance \
            regime at shallow depth, and (c) teacher-forced per-token KL(base||adapter) on the \
            base greedy stream; adapter targets attn+linear_attn+mlp, never mtp.*; run ONE depth \
            per process via NV_CTX_TOKENS (the #96 residual race)"]
fn qwen38_wgpu_lora_adapter_vs_base_mtp_decode_and_acceptance_interleaved_pairs() {
    require_gate("NV_QWEN38_WGPU_TEST");
    let depth = single_depth_from_env_because_of_the_96_residual_race();
    let dir = snapshot_dir();
    let rev = snapshot_rev_label(&dir);
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());

    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("qwen3.8 dense config");
    let max_seq = depth
        + KV_SLOT_HEADROOM_1024_COVERS_ANCHOR_WARMUP_TIMED_OVERSHOOT_AND_THE_512_ROW_VERIFY_GUARD;
    assert!(max_seq < cfg.max_position_embeddings, "depth {depth} does not fit max_pos");

    let b_scale = env_f32(
        "NV_LORA_AB_B_SCALE",
        SYNTH_B_SCALE_DEFAULT_5_3E4_TARGETS_DELTA_W_STD_6E5_VIA_ALPHA_OVER_R_TIMES_AB_SQRT_R,
    );
    let adapter_dir = write_rank8_attn_mlp_adapter_no_mtp_modules(&cfg, b_scale);
    let lora = Q3dLora::from_peft_dir(&adapter_dir, &cfg).expect("build q3d lora sites");
    let adapter_desc = format!(
        "adapter=synth_r{SYNTH_RANK_8_THE_TASK_CONTRACT_RANK}_alpha{}_a{SYNTH_A_SCALE_0_02_KAIMING_LIKE_MAGNITUDE}_b{b_scale}_seed{SYNTH_SEED:x} adapter_dir={} modules=attn+linear_attn+mlp_no_mtp",
        SYNTH_ALPHA_16_SO_SCALING_IS_2_MATCHING_COMMON_PEFT_DEFAULTS,
        adapter_dir.display()
    );

    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let t0 = Instant::now();
    let mut base = Qwen3_5DenseWgpu::from_loader(cfg.clone(), &loader, max_seq)
        .expect("base engine boot");
    base.mtp_attach(&loader).expect("attach mtp head to the base engine");
    let base_boot_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let mut adapted = Qwen3_5DenseWgpu::from_loader_with_lora(cfg.clone(), &loader, max_seq, Some(&lora))
        .expect("adapter engine boot");
    adapted.mtp_attach(&loader).expect("attach mtp head to the adapter engine");
    let adapted_boot_s = t1.elapsed().as_secs_f64();
    drop(loader);
    assert!(!base.lora_resident(), "base engine must carry no lora sites");
    assert!(adapted.lora_resident(), "adapter engine must carry resident lora sites");
    eprintln!(
        "LORA-AB-BOOT depth={depth} base_boot_s={base_boot_s:.1} adapted_boot_s={adapted_boot_s:.1} \
         max_seq={max_seq} checkpoint={CHECKPOINT_LABEL}@{rev} {adapter_desc}"
    );
    eprintln!(
        "LORA-AB-PASSES depth={depth} base_decode_passes={} adapter_decode_passes={} \
         lora_passes_per_token={} base_prefill_passes={} adapter_prefill_passes={} \
         checkpoint={CHECKPOINT_LABEL}@{rev}",
        base.pass_count(),
        adapted.pass_count(),
        lora.total_pass_count(),
        base.prefill_pass_count(),
        adapted.prefill_pass_count(),
    );

    let k = mtp_chain_depth_from_env();
    let replay = mtp_verify_replay_selected_from_env();
    let pairs = env_usize("NV_LORA_AB_PAIRS", INTERLEAVED_PAIRS_DEFAULT_3_PER_POINT_PER_THE_QUALITY_VERDICT_METHOD);
    let timed_target = env_usize(
        "NV_TIMED_EMITTED_TOKENS",
        TIMED_EMITTED_TOKENS_256_THE_SCOREBOARD_GENERATION_LENGTH,
    );

    if depth <= REAL_PREFILL_ACCEPT_REGIME_MAX_DEPTH_512_DEEPER_PREFILL_IS_OUT_OF_BUDGET {
        let prompt = tiled_prompt_ids(&dir, KL_PROMPT_TOKENS_128_TILED_PARAGRAPH);
        base.reset().expect("reset base for kl");
        let mut feed = prefill_like_serving(&mut base, &prompt);
        let mut feeds: Vec<u32> = Vec::new();
        let mut base_rows: Vec<Vec<f32>> = Vec::new();
        for _ in 0..KL_TIMED_STEPS_48_TEACHER_FORCED_ON_THE_BASE_STREAM {
            feeds.push(feed);
            let (tok, row) = base.decode_step_logits(feed).expect("base kl step");
            base_rows.push(row);
            feed = tok;
        }
        adapted.reset().expect("reset adapter for kl");
        let _ = prefill_like_serving(&mut adapted, &prompt);
        let mut kls: Vec<f64> = Vec::new();
        let mut agree = 0usize;
        for (i, &t) in feeds.iter().enumerate() {
            let (ltok, lrow) = adapted.decode_step_logits(t).expect("adapter kl step");
            let lp_b = log_softmax_f64(&base_rows[i]);
            let lp_l = log_softmax_f64(&lrow);
            let mut kl = 0f64;
            for (pb, pl) in lp_b.iter().zip(&lp_l) {
                kl += pb.exp() * (pb - pl);
            }
            kls.push(kl);
            let btok = base_rows[i]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(j, _)| j as u32)
                .unwrap();
            if btok == ltok {
                agree += 1;
            }
        }
        let mean = kls.iter().sum::<f64>() / kls.len() as f64;
        let mx = kls.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "LORA-AB-KL depth=0 basis=teacher_forced_on_base_greedy_stream_after_128tok_tiled_prefill \
             n={} kl_mean_nats={mean:.5} kl_median_nats={:.5} kl_max_nats={mx:.5} \
             top1_agree={agree}/{} checkpoint={CHECKPOINT_LABEL}@{rev} {adapter_desc}",
            kls.len(),
            median_of(&kls),
            kls.len(),
        );

        let vf_rows = base.verify_max_rows().min(adapted.verify_max_rows());
        if vf_rows >= 2 {
            let m = vf_rows.min(k + 1);
            base.reset().expect("reset base for vchain kl");
            let _ = prefill_like_serving(&mut base, &prompt);
            adapted.reset().expect("reset adapter for vchain kl");
            let _ = prefill_like_serving(&mut adapted, &prompt);
            let mut row_kls: Vec<f64> = Vec::new();
            let mut round_means: Vec<f64> = Vec::new();
            for chunk in feeds.chunks(m) {
                if chunk.len() < 2 {
                    break;
                }
                let (_, brows) = base
                    .verify_chain_logits(chunk)
                    .expect("base vchain logit rows");
                base.advance(chunk.len()).expect("base vchain full commit");
                let (_, lrows) = adapted
                    .verify_chain_logits(chunk)
                    .expect("adapter vchain logit rows");
                adapted.advance(chunk.len()).expect("adapter vchain full commit");
                let vocab = brows.len() / chunk.len();
                assert_eq!(
                    brows.len(),
                    lrows.len(),
                    "both arms must return one logit row per verify-chain slot"
                );
                let mut round: Vec<f64> = Vec::with_capacity(chunk.len());
                for i in 0..chunk.len() {
                    let lp_b = log_softmax_f64(&brows[i * vocab..(i + 1) * vocab]);
                    let lp_l = log_softmax_f64(&lrows[i * vocab..(i + 1) * vocab]);
                    let mut kl = 0f64;
                    for (pb, pl) in lp_b.iter().zip(&lp_l) {
                        kl += pb.exp() * (pb - pl);
                    }
                    round.push(kl);
                }
                let rmean = round.iter().sum::<f64>() / round.len() as f64;
                let rmax = round.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                eprintln!(
                    "LORA-AB-KL-VCHAIN depth=0 round={} rows={} kl_mean_nats={rmean:.5} \
                     kl_max_nats={rmax:.5} \
                     basis=teacher_forced_base_greedy_stream_through_the_m_row_verify_graph \
                     checkpoint={CHECKPOINT_LABEL}@{rev} {adapter_desc}",
                    round_means.len(),
                    round.len(),
                );
                round_means.push(rmean);
                row_kls.extend(round);
            }
            assert!(
                !round_means.is_empty(),
                "the vchain kl block forwarded no rounds: verify rows {vf_rows}, k {k}"
            );
            eprintln!(
                "LORA-AB-KL-VCHAIN-SUMMARY depth=0 rounds={} n={} kl_mean_nats={:.5} \
                 kl_round_median_nats={:.5} kl_max_nats={:.5} m_rows_per_round={m} \
                 basis=teacher_forced_base_greedy_stream_through_the_m_row_verify_graph \
                 checkpoint={CHECKPOINT_LABEL}@{rev} {adapter_desc}",
                round_means.len(),
                row_kls.len(),
                row_kls.iter().sum::<f64>() / row_kls.len() as f64,
                median_of(&round_means),
                row_kls.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            );
        } else {
            eprintln!(
                "LORA-AB-KL-VCHAIN-SKIP depth=0 verify_rows={vf_rows}: no M-row verify graph, \
                 so verify-chain logit rows cannot be captured on this boot"
            );
        }
    }

    for pair in 0..pairs {
        let order: [(&str, bool); 2] = if pair % 2 == 0 {
            [("base", false), ("lora", true)]
        } else {
            [("lora", true), ("base", false)]
        };
        for (arm, is_lora) in order {
            let gpu: &mut Qwen3_5DenseWgpu = if is_lora { &mut adapted } else { &mut base };
            let anchor = synthfill_then_anchor(&ctx, gpu, depth);
            let (plain_median, plain_warm) =
                plain_decode_window(gpu, anchor, PLAIN_TIMED_STEPS_64_MATCHES_THE_CTX_SCALING_LADDERS);
            eprintln!(
                "LORA-AB-PLAIN depth={depth} pair={pair} arm={arm} regime=synthfill \
                 median_ms_tok={plain_median:.3} tok_s={:.1} warmup_steps={plain_warm} \
                 timed_steps={PLAIN_TIMED_STEPS_64_MATCHES_THE_CTX_SCALING_LADDERS} \
                 checkpoint={CHECKPOINT_LABEL}@{rev}",
                1000.0 / plain_median,
            );
            let anchor = synthfill_then_anchor(&ctx, gpu, depth);
            let w = mtp_rounds_window(gpu, anchor, k, replay, max_seq, timed_target,
                MTP_ARM_WARMUP_CAP_40_ROUNDS_BOUNDS_DEPTH_DRIFT_OF_THE_TIMED_WINDOW);
            eprintln!(
                "LORA-AB-MTP depth={depth} pair={pair} arm={arm} regime=synthfill k={k} \
                 rounds={} drafted={} accepted={} accept_rate={:.3} tokens_per_round={:.2} \
                 emitted={} warm_emitted={} eff_ms_tok={:.3} eff_tok_s={:.1} \
                 round_ms_median={:.1} checkpoint={CHECKPOINT_LABEL}@{rev}",
                w.rounds,
                w.drafted,
                w.accepted,
                w.accepted as f64 / w.drafted as f64,
                w.emitted as f64 / w.rounds as f64,
                w.emitted,
                w.warm_emitted,
                1000.0 * w.wall_s / w.emitted as f64,
                w.emitted as f64 / w.wall_s,
                w.round_ms_median,
            );
        }
    }

    if depth <= REAL_PREFILL_ACCEPT_REGIME_MAX_DEPTH_512_DEEPER_PREFILL_IS_OUT_OF_BUDGET {
        let prompt = tiled_prompt_ids(&dir, depth);
        for pair in 0..pairs {
            let order: [(&str, bool); 2] = if pair % 2 == 0 {
                [("base", false), ("lora", true)]
            } else {
                [("lora", true), ("base", false)]
            };
            for (arm, is_lora) in order {
                let gpu: &mut Qwen3_5DenseWgpu = if is_lora { &mut adapted } else { &mut base };
                gpu.reset().expect("reset before real prefill");
                let t_pre = Instant::now();
                let anchor = prefill_like_serving(gpu, &prompt);
                let prefill_s = t_pre.elapsed().as_secs_f64();
                assert_eq!(gpu.current_pos(), depth, "real prefill must commit exactly the prompt");
                let w = mtp_rounds_window(gpu, anchor, k, replay, max_seq, timed_target,
                    PLATEAU_FLOOR_8_STEPS_THE_OLD_FIXED_WARMUP_THAT_DID_NOT_RAMP_CLOCKS);
                eprintln!(
                    "LORA-AB-MTP depth={depth} pair={pair} arm={arm} regime=real_tiled_prefill \
                     k={k} rounds={} drafted={} accepted={} accept_rate={:.3} \
                     tokens_per_round={:.2} emitted={} warm_emitted={} prefill_s={prefill_s:.1} \
                     eff_ms_tok={:.3} eff_tok_s={:.1} round_ms_median={:.1} \
                     checkpoint={CHECKPOINT_LABEL}@{rev}",
                    w.rounds,
                    w.drafted,
                    w.accepted,
                    w.accepted as f64 / w.drafted as f64,
                    w.emitted as f64 / w.rounds as f64,
                    w.emitted,
                    w.warm_emitted,
                    1000.0 * w.wall_s / w.emitted as f64,
                    w.emitted as f64 / w.wall_s,
                    w.round_ms_median,
                );
            }
        }
    } else {
        eprintln!(
            "LORA-AB-SKIP depth={depth} real_tiled_prefill_acceptance_regime: prefilling {depth} \
             real tokens per arm per pair is out of this run's budget; synthfill rows above carry \
             the timing contrast at this depth"
        );
    }
}
