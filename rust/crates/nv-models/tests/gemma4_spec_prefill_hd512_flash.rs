#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_layers::attn::{flash_attn_windowed, sdpa, AttnConfig};
use nv_models::gemma4::{Gemma4, Gemma4Config};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;

const HD512_AB_CHILD_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM: &str =
    "NV_G4_HD512AB_CHILD_OUT";
const HD512_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC: &str =
    "NV_G4_HD512AB_CHILD_DEPTH";
const HD512_AB_DEPTH_1200_CROSSES_THE_1024_VERIFY_PREFILL_CHUNK_SEAM_SO_THE_SHADOW_EXTEND_RUNS: usize =
    1200;
const HD512_AB_REBUILD_CONTROL_SEED_CEILING_1_BECAUSE_THE_VERIFY_PREFILL_PATH_MEASURED_BIT_DETERMINISTIC_ACROSS_FRESH_PROCESSES: f32 =
    1.0;
const HD512_AB_SEED_BAND_9_IS_2X_THE_MEASURED_4_46_FP8_TREEKERNEL_VS_BF16_SHADOW_IMPLEMENTATION_DELTA_AND_3X_UNDER_THE_28_58_HD512_FLASH_DEFECT: f32 =
    9.0;
const HD512_AB_MEAN_NLL_BAND_1_0_NATS_IS_1_6X_THE_MEASURED_0_61_BF16_ROUTE_DELTA_AN_NLL_IMPROVEMENT_WITH_SEED_MAX_ABS_3_78_UNDER_THE_EAGER_ESCAPES_4_46_AND_HALF_THE_2_02_HD512_FLASH_DEFECT_RECORD_ARGMAX_AND_SEED_STAY_PRIMARY_BECAUSE_THE_DEFECTS_OWN_NLL_DELTA_MEASURED_ONLY_0_34_ON_PG1342: f64 =
    1.0;

fn gemma4_31b_snapshot_dir() -> PathBuf {
    PathBuf::from(std::env::var("NV_G4_SNAPSHOT").unwrap_or_else(|_| {
        format!(
            "{}/.cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots/e5ef03afa233c35cb000323ff098d4291e1dd07c",
            std::env::var("HOME").unwrap_or_default()
        )
    }))
}

fn load_31b(dir: &PathBuf, device: &Device) -> (Gemma4, Gemma4Config) {
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(dir, device).expect("weights");
    let model =
        Gemma4::from_loader_quantized(cfg.clone(), &weights, &qcfg, device).expect("model");
    (model, cfg)
}

fn corpus_prompt_tokens(dir: &PathBuf, depth: usize) -> Vec<u32> {
    let corpus_path = std::env::var("NV_PPL_CORPUS")
        .expect("NV_PPL_CORPUS must point at the ppl corpus text (see ppl_gemma4.rs)");
    let text = std::fs::read_to_string(&corpus_path).expect("read NV_PPL_CORPUS");
    let tok =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let enc = tok.encode(text.as_str(), false).expect("encode corpus");
    let ids = enc.get_ids();
    assert!(
        ids.len() >= depth,
        "corpus tokenizes to {} tokens, need {depth}",
        ids.len()
    );
    let mut prompt = Vec::with_capacity(depth);
    prompt.push(bos);
    prompt.extend_from_slice(&ids[..depth - 1]);
    prompt
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        assert!(v.is_finite(), "non-finite logit at {i}");
        if v > val {
            val = v;
            best = i;
        }
    }
    best
}

fn lower_tri_mask(len: usize) -> Vec<u8> {
    let mut m = vec![0u8; len * len];
    for i in 0..len {
        for j in 0..=i {
            m[i * len + j] = 1;
        }
    }
    m
}

fn mean_nll_f64(rows: &[f32], vocab: usize, targets: &[u32]) -> f64 {
    let n = targets.len();
    assert_eq!(rows.len(), n * vocab, "row/target count mismatch");
    let mut sum = 0f64;
    for (i, &t) in targets.iter().enumerate() {
        let row = &rows[i * vocab..(i + 1) * vocab];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f64 = row.iter().map(|&x| ((x - m) as f64).exp()).sum();
        let lp = (row[t as usize] - m) as f64 - z.ln();
        sum -= lp;
    }
    sum / n as f64
}

struct ChildRows {
    seed: Vec<f32>,
    nll_rows: Vec<f32>,
    scored: usize,
    wall_s: f64,
}

fn hd512_ab_child_runs_the_serve_loop_prefill_and_writes_rows(out_path: &str, depth: usize) {
    let dir = gemma4_31b_snapshot_dir();
    let device = Device::new_cuda(0).expect("cuda");
    let prompt = corpus_prompt_tokens(&dir, depth);
    let (model, cfg) = load_31b(&dir, &device);
    let vocab = cfg.vocab_size;

    let stream_dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&stream_dev);

    let chunk = nv_models::gemma4::VERIFY_PREFILL_CHUNK;
    let mut cache = model.new_verify_cache(depth + 8).expect("verify cache");
    let t0 = std::time::Instant::now();
    let mut committed = 0usize;
    let mut seed_row: Vec<f32> = Vec::new();
    let mut tail_rows: Vec<f32> = Vec::new();
    let mut tail_row_count = 0usize;
    while committed < depth {
        let len = chunk.min(depth - committed);
        let last = committed + len == depth;
        let logit_rows = if last { len - 1 } else { 0 };
        let pmask_d = stream.clone_htod(&lower_tri_mask(len)).expect("mask htod");
        let ppos: Vec<i32> = (committed as i32..(committed + len) as i32).collect();
        let (logits, _aux) = model
            .forward_verify_tail(
                &prompt[committed..committed + len],
                &ppos,
                &pmask_d,
                committed,
                &[],
                &mut cache,
                logit_rows,
            )
            .expect("forward_verify_tail prefill chunk");
        if let Some(lg) = logits {
            let host: Vec<f32> = lg.flatten_all().expect("flatten").to_vec1().expect("host");
            assert_eq!(host.len(), logit_rows * vocab);
            seed_row = host[(logit_rows - 1) * vocab..].to_vec();
            tail_rows = host[..(logit_rows - 1) * vocab].to_vec();
            tail_row_count = logit_rows - 1;
        }
        committed += len;
    }
    device.synchronize().expect("sync");
    let wall_s = t0.elapsed().as_secs_f64();
    assert_eq!(seed_row.len(), vocab, "seed row must be one vocab row");
    assert!(tail_row_count > 100, "need a scored tail, got {tail_row_count}");

    let mut bytes = Vec::with_capacity(4 * (vocab + tail_rows.len()) + 24);
    bytes.extend_from_slice(&(vocab as u64).to_le_bytes());
    bytes.extend_from_slice(&(tail_row_count as u64).to_le_bytes());
    bytes.extend_from_slice(&wall_s.to_le_bytes());
    for &x in seed_row.iter().chain(tail_rows.iter()) {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(out_path, bytes).expect("write child rows");
    eprintln!(
        "hd512 AB child depth={depth} verify_prefill_wall_s={wall_s:.2} \
         seed_argmax={} scored_tail_rows={tail_row_count}",
        argmax(&seed_row)
    );
}

const HDIM512_FLASH_VS_SDPA_BOUND_1E2_IS_40X_THE_MEASURED_0_000244_CLASS_MAX_ABS_ON_CANDLE_011_AND_MATCHES_THE_HD256_EXACTNESS_BAND:
    f32 = 1e-2;

#[test]
#[ignore = "cuda + sm_120 synthetic regression canary, no model needed: candle 0.11's hdim512 \
            flash forward matches eager sdpa on sm_120 (f16 and bf16, causal and windowed \
            alike, 0.000244-class max-abs) just as hdim256 does, which is why hd-512 \
            prefill-shadow and E4B full layers take flash by default -- this test fails loudly \
            the day a candle-flash-attn bump REGRESSES the hdim512 forward, which is the \
            signal to re-contain: default the 31B shadow onto the NV_SPEC_PREFILL_HD512_BF16 \
            route and the E4B full layers onto NV_E4B_PREFILL_FULL_FLASH_HD512=0"]
fn candle_flash_hdim512_forward_stays_exact_on_sm120_regression_canary() {
    let device = Device::new_cuda(0).expect("cuda");
    let (n_q, n_kv, seq) = (32usize, 4usize, 128usize);
    let mk = |seed: f32, s: usize, h: usize, d: usize| -> Tensor {
        let n = s * h * d;
        let v: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i as f32 * 0.618_034 + seed).sin();
                x * 0.5
            })
            .collect();
        Tensor::from_vec(v, (1usize, s, h, d), &device).unwrap()
    };
    for &dtype in &[DType::F16, DType::BF16] {
        for &(hd, window_left) in &[
            (512usize, None),
            (512, Some(1023usize)),
            (256, None),
            (256, Some(1023)),
        ] {
            let q = mk(0.1, seq, n_q, hd).to_dtype(dtype).unwrap();
            let k = mk(0.2, seq, n_kv, hd).to_dtype(dtype).unwrap();
            let v = mk(0.3, seq, n_kv, hd).to_dtype(dtype).unwrap();
            let cfg = AttnConfig {
                num_heads: n_q,
                num_kv_heads: n_kv,
                head_dim: hd,
                softmax_scale: 1.0,
                causal: true,
            };
            let flash = flash_attn_windowed(&q, &k, &v, &cfg, window_left, Some(0)).unwrap();
            let eager = sdpa(&q, &k, &v, &cfg).unwrap();
            let d = (flash.to_dtype(DType::F32).unwrap() - eager.to_dtype(DType::F32).unwrap())
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            eprintln!(
                "hdim512-canary dtype={dtype:?} hd={hd} window_left={window_left:?} \
                 flash_vs_sdpa_max_abs={d:.6}"
            );
            assert!(
                d < HDIM512_FLASH_VS_SDPA_BOUND_1E2_IS_40X_THE_MEASURED_0_000244_CLASS_MAX_ABS_ON_CANDLE_011_AND_MATCHES_THE_HD256_EXACTNESS_BAND,
                "hdim{hd} flash forward diverged from sdpa by {d} for dtype {dtype:?} window \
                 {window_left:?}: the candle bump regressed the sm_120 forward -- re-contain \
                 the hd512 defaults (NV_SPEC_PREFILL_HD512_BF16 route on the 31B shadow, \
                 NV_E4B_PREFILL_FULL_FLASH_HD512=0 on E4B full layers) before shipping it"
            );
        }
    }
}

#[test]
#[ignore = "re-execs the test binary so EACH arm loads the ~18 GB Gemma-4-31B NVFP4 checkpoint in \
            a FRESH process and runs the PRODUCTION spec-decode prefill shape (eagle3/dflash \
            serve loops -> forward_verify_tail -> attention_tree_impl prefill shadow) over a real \
            NV_PPL_CORPUS slice crossing the 1024-token chunk seam; set NV_PPL_TEST=1 and \
            NV_PPL_CORPUS. Five arms: NV_SPEC_PREFILL_FLASH=0 twice (fresh-process rebuild-noise \
            control), the shipped DEFAULT (candle flash on the ten hd-512 full-attention layers \
            since candle 0.11 fixed the sm_120 hdim512 forward, hd256 sliding flash), \
            NV_SPEC_PREFILL_HD512_BF16=1 (the chunked bf16 tensor-core recovery escape), and \
            NV_SPEC_PREFILL_HD512_EAGER=1 (the chunked eager f32 reference escape). The DEFAULT \
            and both escapes must stay argmax-identical to the eager arms and inside the \
            measured implementation-delta bands. The mean-NLL absolute level here is the \
            raw-corpus regime issue #94 calls an invalid absolute eval -- only same-row \
            cross-arm deltas carry signal, which is why argmax and seed max-abs are the primary \
            assertions"]
fn g4_31b_spec_prefill_flash_default_and_both_hd512_escapes_stay_in_the_eager_band() {
    if std::env::var_os("NV_PPL_TEST").is_none() {
        panic!("set NV_PPL_TEST=1 (and NV_PPL_CORPUS) to run this 31B gate; refusing to skip silently");
    }
    if let Ok(out_path) = std::env::var(HD512_AB_CHILD_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM)
    {
        let depth: usize =
            std::env::var(HD512_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC)
                .expect("child depth env")
                .parse()
                .expect("child depth parse");
        hd512_ab_child_runs_the_serve_loop_prefill_and_writes_rows(&out_path, depth);
        return;
    }
    let dir = gemma4_31b_snapshot_dir();
    assert!(dir.is_dir(), "snapshot dir missing: {}", dir.display());
    let depth = HD512_AB_DEPTH_1200_CROSSES_THE_1024_VERIFY_PREFILL_CHUNK_SEAM_SO_THE_SHADOW_EXTEND_RUNS;
    let prompt = corpus_prompt_tokens(&dir, depth);

    let run_arm = |tag: &str, envs: &[(&str, &str)]| -> ChildRows {
        let exe = std::env::current_exe().expect("test binary path");
        let out_path = std::env::temp_dir().join(format!(
            "g4-hd512-ab-{}-{depth}-{tag}.rows",
            std::process::id()
        ));
        let mut cmd = std::process::Command::new(exe);
        cmd.args([
            "--ignored",
            "--exact",
            "g4_31b_spec_prefill_flash_default_and_both_hd512_escapes_stay_in_the_eager_band",
            "--nocapture",
        ]);
        cmd.env(
            HD512_AB_CHILD_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM,
            &out_path,
        );
        cmd.env(
            HD512_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC,
            depth.to_string(),
        );
        cmd.env_remove("NV_SPEC_PREFILL_FLASH");
        cmd.env_remove("NV_SPEC_PREFILL_HD512_BF16");
        cmd.env_remove("NV_SPEC_PREFILL_HD512_EAGER");
        cmd.env_remove("NV_DETERMINISTIC");
        cmd.env("NV_PREFILL_W4A4", "0");
        for (k, v) in envs {
            cmd.env(k, v);
        }
        let out = cmd.output().expect("spawn fresh-process arm");
        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        assert!(
            out.status.success(),
            "fresh-process arm {tag} failed:\n{}",
            String::from_utf8_lossy(&out.stdout)
        );
        let bytes = std::fs::read(&out_path)
            .unwrap_or_else(|e| panic!("child rows file {}: {e}", out_path.display()));
        std::fs::remove_file(&out_path).ok();
        let vocab = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let tail_rows = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let wall_s = f64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let body = &bytes[24..];
        assert_eq!(body.len(), 4 * vocab * (1 + tail_rows), "rows file shape");
        let all: Vec<f32> = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        ChildRows {
            seed: all[..vocab].to_vec(),
            nll_rows: all[vocab..].to_vec(),
            scored: tail_rows,
            wall_s,
        }
    };

    let eager0 = run_arm("eager0", &[("NV_SPEC_PREFILL_FLASH", "0")]);
    let eager1 = run_arm("eager1", &[("NV_SPEC_PREFILL_FLASH", "0")]);
    let default = run_arm("default", &[]);
    let eagerhd512 = run_arm("eagerhd512", &[("NV_SPEC_PREFILL_HD512_EAGER", "1")]);
    let bf16hd512 = run_arm("hd512bf16", &[("NV_SPEC_PREFILL_HD512_BF16", "1")]);

    let vocab = eager0.seed.len();
    assert_eq!(eager0.scored, eager1.scored);
    assert_eq!(eager0.scored, default.scored);
    assert_eq!(eager0.scored, eagerhd512.scored);
    assert_eq!(eager0.scored, bf16hd512.scored);
    let tail_start = depth - 1 - eager0.scored;
    let targets: Vec<u32> = prompt[tail_start + 1..depth].to_vec();
    assert_eq!(targets.len(), eager0.scored);

    let nll = |c: &ChildRows| mean_nll_f64(&c.nll_rows, vocab, &targets);
    let nll_e0 = nll(&eager0);
    let nll_e1 = nll(&eager1);
    let nll_def = nll(&default);
    let nll_ehd = nll(&eagerhd512);
    let nll_bhd = nll(&bf16hd512);

    let seed_control = max_abs(&eager0.seed, &eager1.seed);
    let seed_def = max_abs(&eager0.seed, &default.seed).max(max_abs(&eager1.seed, &default.seed));
    let seed_ehd =
        max_abs(&eager0.seed, &eagerhd512.seed).max(max_abs(&eager1.seed, &eagerhd512.seed));
    let seed_bhd =
        max_abs(&eager0.seed, &bf16hd512.seed).max(max_abs(&eager1.seed, &bf16hd512.seed));
    let nll_control = (nll_e0 - nll_e1).abs();
    let nll_def_delta = (nll_def - nll_e0).abs().max((nll_def - nll_e1).abs());
    let nll_ehd_delta = (nll_ehd - nll_e0).abs().max((nll_ehd - nll_e1).abs());
    let nll_bhd_delta = (nll_bhd - nll_e0).abs().max((nll_bhd - nll_e1).abs());

    eprintln!(
        "g4-31b hd512 AB depth={depth} scored={} corpus mean-NLL: eager0={nll_e0:.4} \
         eager1={nll_e1:.4} default={nll_def:.4} eagerhd512={nll_ehd:.4} \
         hd512bf16={nll_bhd:.4} (control_delta={nll_control:.4} \
         default_delta={nll_def_delta:.4} eagerhd512_delta={nll_ehd_delta:.4} \
         hd512bf16_delta={nll_bhd_delta:.4})",
        eager0.scored
    );
    eprintln!(
        "g4-31b hd512 AB seed-row max-abs: control={seed_control:.4} default={seed_def:.4} \
         eagerhd512={seed_ehd:.4} hd512bf16={seed_bhd:.4}; argmax eager0={} eager1={} \
         default={} eagerhd512={} hd512bf16={}",
        argmax(&eager0.seed),
        argmax(&eager1.seed),
        argmax(&default.seed),
        argmax(&eagerhd512.seed),
        argmax(&bf16hd512.seed)
    );
    eprintln!(
        "g4-31b hd512 AB verify-prefill wall: eager0={:.2}s eager1={:.2}s default={:.2}s \
         eagerhd512={:.2}s hd512bf16={:.2}s",
        eager0.wall_s, eager1.wall_s, default.wall_s, eagerhd512.wall_s, bf16hd512.wall_s
    );

    assert!(
        seed_control
            <= HD512_AB_REBUILD_CONTROL_SEED_CEILING_1_BECAUSE_THE_VERIFY_PREFILL_PATH_MEASURED_BIT_DETERMINISTIC_ACROSS_FRESH_PROCESSES,
        "eager rebuild control moved {seed_control}: the verify prefill path lost its measured \
         fresh-process determinism and this gate's bands need re-deriving"
    );
    assert_eq!(
        argmax(&eager0.seed),
        argmax(&eager1.seed),
        "eager rebuilds disagree on the seed argmax: the control itself is unstable"
    );
    assert_eq!(
        argmax(&eager0.seed),
        argmax(&default.seed),
        "DEFAULT spec prefill flipped the seed argmax vs the all-eager arm at depth {depth}"
    );
    assert!(
        seed_def
            <= HD512_AB_SEED_BAND_9_IS_2X_THE_MEASURED_4_46_FP8_TREEKERNEL_VS_BF16_SHADOW_IMPLEMENTATION_DELTA_AND_3X_UNDER_THE_28_58_HD512_FLASH_DEFECT,
        "DEFAULT spec prefill moved the seed row by {seed_def} vs the all-eager arm, outside the \
         implementation-delta band (rebuild control {seed_control})"
    );
    assert!(
        nll_def_delta
            <= HD512_AB_MEAN_NLL_BAND_1_0_NATS_IS_1_6X_THE_MEASURED_0_61_BF16_ROUTE_DELTA_AN_NLL_IMPROVEMENT_WITH_SEED_MAX_ABS_3_78_UNDER_THE_EAGER_ESCAPES_4_46_AND_HALF_THE_2_02_HD512_FLASH_DEFECT_RECORD_ARGMAX_AND_SEED_STAY_PRIMARY_BECAUSE_THE_DEFECTS_OWN_NLL_DELTA_MEASURED_ONLY_0_34_ON_PG1342,
        "DEFAULT spec prefill moved corpus mean-NLL by {nll_def_delta:.4} nats vs the all-eager \
         arm, outside the implementation-delta band (rebuild control {nll_control:.4})"
    );
    assert_eq!(
        argmax(&eager0.seed),
        argmax(&eagerhd512.seed),
        "NV_SPEC_PREFILL_HD512_EAGER escape flipped the seed argmax vs the all-eager arm at \
         depth {depth}"
    );
    assert!(
        seed_ehd
            <= HD512_AB_SEED_BAND_9_IS_2X_THE_MEASURED_4_46_FP8_TREEKERNEL_VS_BF16_SHADOW_IMPLEMENTATION_DELTA_AND_3X_UNDER_THE_28_58_HD512_FLASH_DEFECT,
        "NV_SPEC_PREFILL_HD512_EAGER escape moved the seed row by {seed_ehd} vs the all-eager \
         arm, outside the implementation-delta band (rebuild control {seed_control})"
    );
    assert!(
        nll_ehd_delta
            <= HD512_AB_MEAN_NLL_BAND_1_0_NATS_IS_1_6X_THE_MEASURED_0_61_BF16_ROUTE_DELTA_AN_NLL_IMPROVEMENT_WITH_SEED_MAX_ABS_3_78_UNDER_THE_EAGER_ESCAPES_4_46_AND_HALF_THE_2_02_HD512_FLASH_DEFECT_RECORD_ARGMAX_AND_SEED_STAY_PRIMARY_BECAUSE_THE_DEFECTS_OWN_NLL_DELTA_MEASURED_ONLY_0_34_ON_PG1342,
        "NV_SPEC_PREFILL_HD512_EAGER escape moved corpus mean-NLL by {nll_ehd_delta:.4} nats vs \
         the all-eager arm, outside the implementation-delta band (rebuild control \
         {nll_control:.4})"
    );
    assert_eq!(
        argmax(&eager0.seed),
        argmax(&bf16hd512.seed),
        "NV_SPEC_PREFILL_HD512_BF16 escape flipped the seed argmax vs the all-eager arm at \
         depth {depth}"
    );
    assert!(
        seed_bhd
            <= HD512_AB_SEED_BAND_9_IS_2X_THE_MEASURED_4_46_FP8_TREEKERNEL_VS_BF16_SHADOW_IMPLEMENTATION_DELTA_AND_3X_UNDER_THE_28_58_HD512_FLASH_DEFECT,
        "NV_SPEC_PREFILL_HD512_BF16 escape moved the seed row by {seed_bhd} vs the all-eager \
         arm, outside the implementation-delta band (rebuild control {seed_control})"
    );
    assert!(
        nll_bhd_delta
            <= HD512_AB_MEAN_NLL_BAND_1_0_NATS_IS_1_6X_THE_MEASURED_0_61_BF16_ROUTE_DELTA_AN_NLL_IMPROVEMENT_WITH_SEED_MAX_ABS_3_78_UNDER_THE_EAGER_ESCAPES_4_46_AND_HALF_THE_2_02_HD512_FLASH_DEFECT_RECORD_ARGMAX_AND_SEED_STAY_PRIMARY_BECAUSE_THE_DEFECTS_OWN_NLL_DELTA_MEASURED_ONLY_0_34_ON_PG1342,
        "NV_SPEC_PREFILL_HD512_BF16 escape moved corpus mean-NLL by {nll_bhd_delta:.4} nats vs \
         the all-eager arm, outside the implementation-delta band (rebuild control \
         {nll_control:.4})"
    );
}

const HD512_WALLS_DEFAULT_DEPTHS_1200_8192_32768_COVER_THE_FIVE_ARM_GATES_POINT_AND_THE_DEEP_LADDER:
    [usize; 3] = [1200, 8192, 32768];
const HD512_WALLS_WARMUP_RUNS_1_ABSORBS_FIRST_CALL_MODULE_AND_WORKSPACE_INIT: usize = 1;
const HD512_WALLS_TIMED_RUNS_2_GIVE_A_REPEAT_WITHOUT_DOUBLING_THE_DEEP_POINTS: usize = 2;

fn wall_depths_from_env_comma_list_k_suffix(default: &[usize]) -> Vec<usize> {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(s) => s
            .split(',')
            .map(|t| {
                let t = t.trim();
                match t.strip_suffix(['k', 'K']) {
                    Some(n) => n.parse::<usize>().expect("NV_CTX_TOKENS k entry") * 1024,
                    None => t.parse::<usize>().expect("NV_CTX_TOKENS entry"),
                }
            })
            .collect(),
        Err(_) => default.to_vec(),
    }
}

fn checkpoint_label_from_snapshot_dir(dir: &PathBuf) -> String {
    let rev = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into());
    let repo = dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into());
    format!("{repo}@{rev}")
}

fn hd512_route_arm_from_process_env() -> &'static str {
    if std::env::var("NV_SPEC_PREFILL_FLASH").as_deref() == Ok("0") {
        "eagerall"
    } else if std::env::var("NV_SPEC_PREFILL_HD512_EAGER").as_deref() == Ok("1") {
        "hd512eager"
    } else if std::env::var("NV_SPEC_PREFILL_HD512_BF16").as_deref() == Ok("1") {
        "hd512bf16"
    } else {
        "flash-default"
    }
}

#[test]
#[ignore = "loads the ~18 GB Gemma-4-31B NVFP4 checkpoint ONCE and walls the PRODUCTION \
            spec-decode verify-prefill shape (forward_verify_tail chunk-1024 loop, prefill \
            shadow on the ten hd-512 full-attention layers, logit_rows=1 so the lm_head tail \
            stays out of the route comparison) over a real NV_PPL_CORPUS slice at each \
            NV_CTX_TOKENS depth (comma list, k suffix ok, default 1200,8192,32768); set \
            NV_PPL_TEST=1 and NV_PPL_CORPUS; ONE route arm per process, selected by the \
            production envs (default flash route, NV_SPEC_PREFILL_HD512_BF16=1 bf16 recovery \
            escape, NV_SPEC_PREFILL_HD512_EAGER=1 chunked-eager escape) -- run 0 per depth is \
            warmup, \
            runs 1.. print CTX-PREFILL machine lines with the seed argmax so cross-arm \
            agreement is checkable across fresh-process rows"]
fn g4_31b_spec_prefill_hd512_route_verify_prefill_wall_vs_depth() {
    if std::env::var_os("NV_PPL_TEST").is_none() {
        panic!("set NV_PPL_TEST=1 (and NV_PPL_CORPUS) to run this 31B wall probe; refusing to skip silently");
    }
    let dir = gemma4_31b_snapshot_dir();
    assert!(dir.is_dir(), "snapshot dir missing: {}", dir.display());
    let label = checkpoint_label_from_snapshot_dir(&dir);
    let arm = hd512_route_arm_from_process_env();
    let device = Device::new_cuda(0).expect("cuda");
    let (model, cfg) = load_31b(&dir, &device);
    let vocab = cfg.vocab_size;
    let stream_dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&stream_dev);
    let chunk = nv_models::gemma4::VERIFY_PREFILL_CHUNK;

    let depths = wall_depths_from_env_comma_list_k_suffix(
        &HD512_WALLS_DEFAULT_DEPTHS_1200_8192_32768_COVER_THE_FIVE_ARM_GATES_POINT_AND_THE_DEEP_LADDER,
    );
    for depth in depths {
        let prompt = corpus_prompt_tokens(&dir, depth);
        for run in 0..HD512_WALLS_WARMUP_RUNS_1_ABSORBS_FIRST_CALL_MODULE_AND_WORKSPACE_INIT
            + HD512_WALLS_TIMED_RUNS_2_GIVE_A_REPEAT_WITHOUT_DOUBLING_THE_DEEP_POINTS
        {
            let mut cache = model.new_verify_cache(depth + 8).expect("verify cache");
            device.synchronize().expect("sync before starting the clock");
            let t0 = std::time::Instant::now();
            let mut committed = 0usize;
            let mut seed_row: Vec<f32> = Vec::new();
            while committed < depth {
                let len = chunk.min(depth - committed);
                let last = committed + len == depth;
                let logit_rows = if last { 1 } else { 0 };
                let pmask_d = stream.clone_htod(&lower_tri_mask(len)).expect("mask htod");
                let ppos: Vec<i32> = (committed as i32..(committed + len) as i32).collect();
                let (logits, _aux) = model
                    .forward_verify_tail(
                        &prompt[committed..committed + len],
                        &ppos,
                        &pmask_d,
                        committed,
                        &[],
                        &mut cache,
                        logit_rows,
                    )
                    .expect("forward_verify_tail prefill chunk");
                if let Some(lg) = logits {
                    seed_row = lg.flatten_all().expect("flatten").to_vec1().expect("host");
                }
                committed += len;
            }
            device.synchronize().expect("sync before stopping the clock");
            let dt = t0.elapsed().as_secs_f64();
            assert_eq!(seed_row.len(), vocab, "seed row must be one vocab row");
            let seed = argmax(&seed_row);
            let tok_s = depth as f64 / dt;
            if run >= HD512_WALLS_WARMUP_RUNS_1_ABSORBS_FIRST_CALL_MODULE_AND_WORKSPACE_INIT {
                println!(
                    "CTX-PREFILL checkpoint={label} depth={depth} prefill_s={dt:.3} \
                     prefill_tok_s={tok_s:.1} chunk={chunk} run={run} seed={seed} arm={arm} \
                     basis=verify_prefill_shadow_logit_rows_1"
                );
            } else {
                eprintln!(
                    "g4-31b hd512 walls warmup depth={depth} run={run} prefill_s={dt:.3} \
                     seed={seed} arm={arm}"
                );
            }
        }
    }
}
