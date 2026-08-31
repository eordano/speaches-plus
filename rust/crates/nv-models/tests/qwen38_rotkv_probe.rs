#![cfg(feature = "wgpu")]

mod hub_snapshot;
mod ppl_common;

use anyhow::Result;
use nv_kernels::wgpu_backend::kernels::kv_fp8;
use nv_kernels::wgpu_backend::kernels::kv_nvfp4;
use nv_models::hadamard_kv;
use nv_models::qwen3_5_dense_wgpu::{Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use nv_weights::WeightLoader;
use ppl_common::{
    checkpoint_label_from_snapshot_dir, corpus_text_from_nv_ppl_corpus_env_failing_loudly,
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control,
    first_n_corpus_tokens_after_tokenization,
    print_machine_line_with_acc_and_assert_real_beats_shuffled,
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip,
    TeacherForcedNllFp32SoftmaxF64Sum,
    PPL_CORPUS_SLICE_TOKENS_N_2048_CHOSEN_SO_31B_LOAD_PLUS_FOUR_EAGER_512_BLOCKS_FITS_2_TO_3_MIN,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const REPO: &str = "unsloth/Qwen3.8-27B-NVFP4";

const WGPU_GATE_ENV: &str = "NV_QWEN38_WGPU_TEST";

const ARM_ENV_NULL_K4V4SIM_PLAIN16_ROT16_ROT32_ROT64_OR_S_SUFFIX_FOR_K_SMOOTHING: &str =
    "NV_ROTKV_ARM";

const ARGMAX_OUT_ENV_WRITES_PER_POSITION_GREEDY_ARGMAX_JSON: &str = "NV_ROTKV_ARGMAX_OUT";

const ARGMAX_BASE_ENV_COMPARES_AGAINST_A_PRIOR_ARMS_ARGMAX_JSON: &str = "NV_ROTKV_ARGMAX_BASE";

const SINK_SLOTS_STAY_EXACT_MATCHING_THE_SHIPPED_KV_NVFP4_SUBSTRATE: usize =
    kv_nvfp4::KV_NVFP4_SINK_SLOTS_STAY_FP8_TO_ANCHOR_SOFTMAX as usize;

const K_BLOCK_TOKENS_MATCHING_THE_SHIPPED_K4V4_CHANNEL_BLOCK_GRAIN: usize =
    kv_nvfp4::KV_NVFP4_K_BLOCK_TOKENS_A_BLOCK_FINALIZES_THE_STEP_ITS_LAST_TOKEN_LANDS as usize;

const SMOOTH_ALPHA_HALF_SPLITS_THE_CHANNEL_AMAX_BETWEEN_K_AND_THE_Q_SIDE_COMPENSATION: f32 = 0.5;

const SMOOTH_AMAX_FLOOR_KEEPS_DEAD_CHANNELS_FROM_EXPLODING_THE_UNSMOOTH_MULTIPLY: f32 = 1e-3;

fn require_qwen38_wgpu_gate() {
    if std::env::var(WGPU_GATE_ENV).as_deref() != Ok("1") {
        panic!(
            "set {WGPU_GATE_ENV}=1 to run the real-weights Qwen3.8-27B-NVFP4 rotkv probe \
             (it must never silently skip; it boots the 22.57 GB checkpoint)"
        );
    }
}

fn qwen38_snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
        if !d.is_empty() {
            let p = PathBuf::from(&d);
            assert!(p.is_dir(), "NV_QWEN38_DIR={d} is not a directory");
            return p;
        }
    }
    hub_snapshot::snapshot_of(REPO, &["config.json", "tokenizer.json", "*.safetensors"])
        .unwrap_or_else(|| {
            panic!(
                "no complete {REPO} snapshot under the HF hub roots {:?}; set NV_QWEN38_DIR \
                 (this gated suite refuses to vacuously pass)",
                hub_snapshot::hub_roots()
            )
        })
}

fn boot_model_on_the_bf16_kv_read_arm(dir: &PathBuf, max_seq: usize) -> Qwen3_5DenseWgpu {
    std::env::set_var("NV_Q3D_KV_FP8", "0");
    std::env::set_var("NV_Q3D_VF_ATTN_FLASH", "0");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw)
        .expect("real Qwen3.8-27B-NVFP4 config.json must parse as a Qwen3_5 dense config");
    assert!(
        std::env::var("NV_QWEN38_LAYERS").is_err(),
        "NV_QWEN38_LAYERS truncates the trunk and would make every rotkv number non-canonical; \
         unset it"
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let loader = WeightLoader::open_dir(dir, &candle_core::Device::Cpu).expect("open weights");
    let m = Qwen3_5DenseWgpu::from_loader(cfg, &loader, max_seq)
        .expect("Qwen3_5DenseWgpu::from_loader on the real Qwen3.8-27B-NVFP4 checkpoint");
    drop(loader);
    m
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Null,
    K4v4Sim,
    Plain16,
    Rot { block: usize, smooth: bool },
}

fn arm_from_env() -> (Arm, String) {
    let raw = std::env::var(ARM_ENV_NULL_K4V4SIM_PLAIN16_ROT16_ROT32_ROT64_OR_S_SUFFIX_FOR_K_SMOOTHING)
        .unwrap_or_else(|_| {
            panic!(
                "{ARM_ENV_NULL_K4V4SIM_PLAIN16_ROT16_ROT32_ROT64_OR_S_SUFFIX_FOR_K_SMOOTHING} is \
                 unset; arms: null k4v4sim plain16 rot16 rot32 rot64 rot16s rot32s rot64s"
            )
        });
    let arm = match raw.as_str() {
        "null" => Arm::Null,
        "k4v4sim" => Arm::K4v4Sim,
        "plain16" => Arm::Plain16,
        "rot16" => Arm::Rot { block: 16, smooth: false },
        "rot32" => Arm::Rot { block: 32, smooth: false },
        "rot64" => Arm::Rot { block: 64, smooth: false },
        "rot16s" => Arm::Rot { block: 16, smooth: true },
        "rot32s" => Arm::Rot { block: 32, smooth: true },
        "rot64s" => Arm::Rot { block: 64, smooth: true },
        other => panic!(
            "{ARM_ENV_NULL_K4V4SIM_PLAIN16_ROT16_ROT32_ROT64_OR_S_SUFFIX_FOR_K_SMOOTHING}={other} \
             names no arm"
        ),
    };
    (arm, raw)
}

fn fp8_row_roundtrip_shipped_per_slot_head_scale(row: &mut [f32]) {
    let amax = row.iter().fold(0f32, |a, v| a.max(v.abs()));
    if amax <= 0.0 {
        return;
    }
    let scale = amax / kv_fp8::FP8_E4M3_MAX;
    let inv = kv_fp8::FP8_E4M3_MAX / amax;
    for v in row.iter_mut() {
        *v = kv_fp8::decode_e4m3(kv_fp8::encode_e4m3(*v * inv)) * scale;
    }
}

fn v4_row_roundtrip_shipped_e2m1_per_row_scale(row: &mut [f32]) {
    let amax = row.iter().fold(0f32, |a, v| a.max(v.abs()));
    if amax <= 0.0 {
        return;
    }
    let scale = amax / kv_nvfp4::E2M1_MAX;
    let inv = kv_nvfp4::E2M1_MAX / amax;
    for v in row.iter_mut() {
        let nib = kv_nvfp4::encode_e2m1(*v * inv);
        *v = kv_nvfp4::decode_e2m1(nib) * scale;
    }
}

fn nvfp4_16block_roundtrip_per_row_global(row: &mut [f32]) {
    let amax = row.iter().fold(0f32, |a, v| a.max(v.abs()));
    if amax <= 0.0 {
        return;
    }
    let stored = (448.0f32 * 6.0) / amax;
    for block in row.chunks_mut(nv_quant::nvfp4::BLOCK_SIZE) {
        let (packed, sb) = nv_quant::nvfp4::quantize_block_with_global(block, stored);
        let dq = nv_quant::nvfp4::dequantize_block(&packed, sb);
        for (o, q) in block.iter_mut().zip(dq) {
            *o = q / stored;
        }
    }
}

fn rotate_head_row_in_hadamard_blocks(row: &mut [f32], block: usize) {
    for chunk in row.chunks_mut(block) {
        hadamard_kv::rotate_row(chunk)
            .expect("rotkv blocks are 16/32/64, all powers of two, so rotation cannot fail");
    }
}

fn k4v4sim_channel_block_roundtrip(raw_rows: &[Vec<f32>], n_kv: usize, hd: usize) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = raw_rows.to_vec();
    for kvh in 0..n_kv {
        for d in 0..hd {
            let idx = kvh * hd + d;
            let amax = raw_rows.iter().fold(0f32, |a, r| a.max(r[idx].abs()));
            if amax <= 0.0 {
                continue;
            }
            let scale = amax / kv_nvfp4::E2M1_MAX;
            let inv = kv_nvfp4::E2M1_MAX / amax;
            for (o, r) in out.iter_mut().zip(raw_rows) {
                let nib = kv_nvfp4::encode_e2m1(r[idx] * inv);
                o[idx] = kv_nvfp4::decode_e2m1(nib) * scale;
            }
        }
    }
    out
}

struct SmoothState {
    running_amax_per_layer_channel: Vec<Vec<f32>>,
}

impl SmoothState {
    fn new(layers: usize, channels: usize) -> Self {
        Self {
            running_amax_per_layer_channel: vec![vec![0f32; channels]; layers],
        }
    }

    fn factors_from_rows_seen_so_far_so_the_transform_stays_causal(&self, layer: usize) -> Vec<f32> {
        self.running_amax_per_layer_channel[layer]
            .iter()
            .map(|a| {
                a.max(SMOOTH_AMAX_FLOOR_KEEPS_DEAD_CHANNELS_FROM_EXPLODING_THE_UNSMOOTH_MULTIPLY)
                    .powf(SMOOTH_ALPHA_HALF_SPLITS_THE_CHANNEL_AMAX_BETWEEN_K_AND_THE_Q_SIDE_COMPENSATION)
            })
            .collect()
    }

    fn absorb_raw_row(&mut self, layer: usize, row: &[f32]) {
        for (a, v) in self.running_amax_per_layer_channel[layer].iter_mut().zip(row) {
            *a = a.max(v.abs());
        }
    }
}

fn mangle_k_head_rows(arm: Arm, row: &mut [f32], hd: usize, smooth_factors: Option<&[f32]>) {
    match arm {
        Arm::Null | Arm::K4v4Sim => {}
        Arm::Plain16 => {
            for head_row in row.chunks_mut(hd) {
                nvfp4_16block_roundtrip_per_row_global(head_row);
            }
        }
        Arm::Rot { block, smooth } => {
            if smooth {
                let s = smooth_factors
                    .expect("smooth arms compute causal per-channel factors before mangling");
                for (v, f) in row.iter_mut().zip(s) {
                    *v /= f;
                }
            }
            for head_row in row.chunks_mut(hd) {
                rotate_head_row_in_hadamard_blocks(head_row, block);
                nvfp4_16block_roundtrip_per_row_global(head_row);
                rotate_head_row_in_hadamard_blocks(head_row, block);
            }
            if smooth {
                let s = smooth_factors.expect("checked above");
                for (v, f) in row.iter_mut().zip(s) {
                    *v *= f;
                }
            }
        }
    }
}

fn mangle_v_head_rows_v4row_for_every_quantizing_arm(arm: Arm, row: &mut [f32], hd: usize) {
    if arm == Arm::Null {
        return;
    }
    for head_row in row.chunks_mut(hd) {
        v4_row_roundtrip_shipped_e2m1_per_row_scale(head_row);
    }
}

struct Mangler {
    arm: Arm,
    n_kv: usize,
    hd: usize,
    raw_k_block_start_and_rows_per_layer: Vec<(usize, Vec<Vec<f32>>)>,
    smooth: SmoothState,
}

impl Mangler {
    fn new(arm: Arm, layers: usize, n_kv: usize, hd: usize) -> Self {
        Self {
            arm,
            n_kv,
            hd,
            raw_k_block_start_and_rows_per_layer: vec![(0, Vec::new()); layers],
            smooth: SmoothState::new(layers, n_kv * hd),
        }
    }

    fn reset_between_arms(&mut self) {
        for (start, rows) in &mut self.raw_k_block_start_and_rows_per_layer {
            *start = 0;
            rows.clear();
        }
        let layers = self.raw_k_block_start_and_rows_per_layer.len();
        self.smooth = SmoothState::new(layers, self.n_kv * self.hd);
    }

    fn after_step(&mut self, m: &mut Qwen3_5DenseWgpu, pos: usize) -> Result<()> {
        if self.arm == Arm::Null || pos < SINK_SLOTS_STAY_EXACT_MATCHING_THE_SHIPPED_KV_NVFP4_SUBSTRATE
        {
            return Ok(());
        }
        let bt = K_BLOCK_TOKENS_MATCHING_THE_SHIPPED_K4V4_CHANNEL_BLOCK_GRAIN;
        for layer in 0..m.attn_kv_layer_count() {
            let (mut k, mut v) = m.debug_read_attn_kv_bf16_rows_for_test(layer, pos, 1)?;
            mangle_v_head_rows_v4row_for_every_quantizing_arm(self.arm, &mut v, self.hd);
            m.debug_overwrite_attn_v_bf16_rows_for_test(layer, pos, &v)?;
            match self.arm {
                Arm::K4v4Sim => {
                    let b0 = (pos / bt * bt)
                        .max(SINK_SLOTS_STAY_EXACT_MATCHING_THE_SHIPPED_KV_NVFP4_SUBSTRATE);
                    let (start, rows) = &mut self.raw_k_block_start_and_rows_per_layer[layer];
                    if rows.is_empty() || *start != b0 {
                        *start = b0;
                        rows.clear();
                    }
                    rows.push(k.clone());
                    assert_eq!(
                        b0 + rows.len() - 1,
                        pos,
                        "k4v4sim raw-row ledger drifted from the cache position"
                    );
                    let quantized = k4v4sim_channel_block_roundtrip(rows, self.n_kv, self.hd);
                    let flat: Vec<f32> = quantized.concat();
                    m.debug_overwrite_attn_k_bf16_rows_for_test(layer, b0, &flat)?;
                }
                _ => {
                    let factors = if matches!(self.arm, Arm::Rot { smooth: true, .. }) {
                        Some(
                            self.smooth
                                .factors_from_rows_seen_so_far_so_the_transform_stays_causal(layer),
                        )
                    } else {
                        None
                    };
                    let raw_k = k.clone();
                    mangle_k_head_rows(self.arm, &mut k, self.hd, factors.as_deref());
                    self.smooth.absorb_raw_row(layer, &raw_k);
                    m.debug_overwrite_attn_k_bf16_rows_for_test(layer, pos, &k)?;
                }
            }
        }
        Ok(())
    }
}

fn score_slice_with_mangle(
    m: &mut Qwen3_5DenseWgpu,
    mangler: &mut Mangler,
    ids: &[u32],
) -> (TeacherForcedNllFp32SoftmaxF64Sum, Vec<u32>) {
    assert_eq!(
        m.current_pos(),
        0,
        "wgpu model must start each arm at position 0; call reset() between arms"
    );
    let vocab = m.config().vocab_size;
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    let mut argmaxes: Vec<u32> = Vec::with_capacity(ids.len() - 1);
    for p in 0..ids.len() - 1 {
        let (argmax, row) = m
            .decode_step_logits(ids[p])
            .unwrap_or_else(|err| panic!("wgpu decode_step_logits at position {p}: {err:#}"));
        mangler
            .after_step(m, p)
            .unwrap_or_else(|err| panic!("rotkv mangle at position {p}: {err:#}"));
        assert_eq!(row.len(), vocab);
        argmaxes.push(argmax);
        if (ids[p + 1] as usize) >= vocab {
            continue;
        }
        acc.add_position_full_vocab_row(&row, ids[p + 1]);
    }
    (acc, argmaxes)
}

#[test]
#[ignore = "boots the ~22.6 GB unsloth/Qwen3.8-27B-NVFP4 checkpoint on the wgpu bf16-KV read arm \
            and scores the canonical 2048-token slice while re-writing each freshly appended KV \
            row through the NV_ROTKV_ARM host quantization pipeline; set NV_QWEN38_WGPU_TEST=1, \
            NV_PPL_TEST=1, NV_PPL_CORPUS, NV_ROTKV_ARM"]
fn qwen38_rotkv_mangled_teacher_forced_ppl_beats_shuffled_control() {
    require_qwen38_wgpu_gate();
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let (arm, arm_name) = arm_from_env();

    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = qwen38_snapshot_dir();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let n = PPL_CORPUS_SLICE_TOKENS_N_2048_CHOSEN_SO_31B_LOAD_PLUS_FOUR_EAGER_512_BLOCKS_FITS_2_TO_3_MIN;
    let slice = first_n_corpus_tokens_after_tokenization(&tok, &corpus, n);

    let mut m = boot_model_on_the_bf16_kv_read_arm(&dir, n + 8);
    let cfg = m.config().clone();
    let mut mangler = Mangler::new(
        arm,
        m.attn_kv_layer_count(),
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    assert!(
        m.attn_kv_layer_count() > 0,
        "no full-attention layers exposed; the probe would mangle nothing and pass vacuously"
    );

    let label = checkpoint_label_from_snapshot_dir(&dir);
    eprintln!(
        "PPL basis: {label} corpus_slice_tokens={n} backend=wgpu batch=1 \
         path=decode_step_logits+per-step-kv-mangle arm={arm_name} kv_read=bf16 \
         sink_slots_exact={SINK_SLOTS_STAY_EXACT_MATCHING_THE_SHIPPED_KV_NVFP4_SUBSTRATE} \
         attn_layers={}",
        m.attn_kv_layer_count()
    );

    let (real, argmaxes) = score_slice_with_mangle(&mut m, &mut mangler, &slice);

    if let Ok(path) = std::env::var(ARGMAX_OUT_ENV_WRITES_PER_POSITION_GREEDY_ARGMAX_JSON) {
        let body: Vec<String> = argmaxes.iter().map(|a| a.to_string()).collect();
        std::fs::write(&path, format!("[{}]", body.join(",")))
            .unwrap_or_else(|e| panic!("writing {path}: {e}"));
        eprintln!("ROTKV-ARGMAX-OUT arm={arm_name} path={path} positions={}", argmaxes.len());
    }
    if let Ok(path) = std::env::var(ARGMAX_BASE_ENV_COMPARES_AGAINST_A_PRIOR_ARMS_ARGMAX_JSON) {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
        let base: Vec<u32> = text
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<u32>()
                    .unwrap_or_else(|e| panic!("parsing {path}: {e} at {s:?}"))
            })
            .collect();
        assert_eq!(
            base.len(),
            argmaxes.len(),
            "argmax base {path} scored a different position count; arms are not comparable"
        );
        let flips = base.iter().zip(&argmaxes).filter(|(a, b)| a != b).count();
        let first = base.iter().zip(&argmaxes).position(|(a, b)| a != b);
        eprintln!(
            "ROTKV-ARGMAX-FLIPS arm={arm_name} flips={flips}/{} first_flip_pos={first:?}",
            argmaxes.len()
        );
    }

    let mut shuffled = slice.clone();
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
        &mut shuffled,
    );
    m.reset().expect("reset between arms");
    mangler.reset_between_arms();
    let (shuffled, _) = score_slice_with_mangle(&mut m, &mut mangler, &shuffled);

    assert_eq!(
        real.scored_positions, shuffled.scored_positions,
        "real and shuffled arms scored different position counts"
    );
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        &format!("qwen3_8-dense-wgpu-rotkv-{arm_name}"),
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}

fn rel_rms(raw: &[f32], got: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in raw.iter().zip(got) {
        num += ((a - b) as f64).powi(2);
        den += (*a as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

#[test]
#[ignore = "boots the ~22.6 GB checkpoint on the wgpu bf16-KV arm, chunk-prefills 1024 corpus \
            tokens and reports elementwise K/V reconstruction error for every rotkv arm on three \
            layers spread across the stack; set NV_QWEN38_WGPU_TEST=1, NV_PPL_CORPUS"]
fn qwen38_rotkv_elementwise_error_probe_reports_every_arm() {
    require_qwen38_wgpu_gate();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = qwen38_snapshot_dir();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let depth = 1024usize;
    let slice = first_n_corpus_tokens_after_tokenization(&tok, &corpus, depth);

    let mut m = boot_model_on_the_bf16_kv_read_arm(&dir, depth + 8);
    let cfg = m.config().clone();
    let (n_kv, hd) = (cfg.num_key_value_heads, cfg.head_dim);
    let done = m.prefill_tokens(&slice).expect("chunked prefill of the probe slice");
    for t in &slice[done..] {
        m.prefill_step(*t).expect("tail prefill step");
    }
    assert_eq!(m.current_pos(), depth, "prefill did not reach probe depth");

    let layers = m.attn_kv_layer_count();
    assert!(layers >= 3, "expected at least three full-attention layers, got {layers}");
    let picks = [0usize, layers / 2, layers - 1];
    let sink = SINK_SLOTS_STAY_EXACT_MATCHING_THE_SHIPPED_KV_NVFP4_SUBSTRATE;
    let bt = K_BLOCK_TOKENS_MATCHING_THE_SHIPPED_K4V4_CHANNEL_BLOCK_GRAIN;

    for &layer in &picks {
        let (k_all, v_all) = m
            .debug_read_attn_kv_bf16_rows_for_test(layer, sink, depth - sink)
            .expect("bulk KV readback at probe depth");
        let rows = depth - sink;
        let row_len = n_kv * hd;
        assert_eq!(k_all.len(), rows * row_len);

        let mut channel_amax = vec![0f32; row_len];
        for r in 0..rows {
            for (c, a) in channel_amax.iter_mut().enumerate() {
                *a = a.max(k_all[r * row_len + c].abs());
            }
        }
        let mut spread_stats: Vec<f64> = Vec::new();
        for kvh in 0..n_kv {
            let mut ch: Vec<f32> = channel_amax[kvh * hd..(kvh + 1) * hd].to_vec();
            ch.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = ch[ch.len() / 2].max(1e-6);
            spread_stats.push((ch[ch.len() - 1] / med) as f64);
        }
        println!(
            "ROTKV-PROBE layer={layer} tensor=K channel_amax_max_over_median_per_head={:?}",
            spread_stats.iter().map(|s| format!("{s:.1}")).collect::<Vec<_>>()
        );

        let eval_k = |name: &str, f: &mut dyn FnMut(&mut [f32], usize)| {
            let mut got = k_all.clone();
            for r in 0..rows {
                f(&mut got[r * row_len..(r + 1) * row_len], r);
            }
            println!(
                "ROTKV-PROBE layer={layer} tensor=K arm={name} rel_rms={:.5}",
                rel_rms(&k_all, &got)
            );
        };

        eval_k("fp8row", &mut |row, _| {
            for hr in row.chunks_mut(hd) {
                fp8_row_roundtrip_shipped_per_slot_head_scale(hr);
            }
        });
        eval_k("plain16", &mut |row, _| {
            mangle_k_head_rows(Arm::Plain16, row, hd, None)
        });
        eval_k("v4row-on-K", &mut |row, _| {
            for hr in row.chunks_mut(hd) {
                v4_row_roundtrip_shipped_e2m1_per_row_scale(hr);
            }
        });
        for block in [16usize, 32, 64] {
            eval_k(&format!("rot{block}"), &mut |row, _| {
                mangle_k_head_rows(Arm::Rot { block, smooth: false }, row, hd, None)
            });
        }
        {
            for block in [32usize, 64] {
                let mut sm = SmoothState::new(1, row_len);
                let mut got = k_all.clone();
                for r in 0..rows {
                    let factors = sm.factors_from_rows_seen_so_far_so_the_transform_stays_causal(0);
                    let raw: Vec<f32> = got[r * row_len..(r + 1) * row_len].to_vec();
                    mangle_k_head_rows(
                        Arm::Rot { block, smooth: true },
                        &mut got[r * row_len..(r + 1) * row_len],
                        hd,
                        Some(&factors),
                    );
                    sm.absorb_raw_row(0, &raw);
                }
                println!(
                    "ROTKV-PROBE layer={layer} tensor=K arm=rot{block}s rel_rms={:.5}",
                    rel_rms(&k_all, &got)
                );
            }
        }
        {
            let mut got = k_all.clone();
            for b0 in (0..rows).step_by(bt) {
                let b1 = (b0 + bt).min(rows);
                let raw_rows: Vec<Vec<f32>> = (b0..b1)
                    .map(|r| k_all[r * row_len..(r + 1) * row_len].to_vec())
                    .collect();
                let q = k4v4sim_channel_block_roundtrip(&raw_rows, n_kv, hd);
                for (i, r) in (b0..b1).enumerate() {
                    got[r * row_len..(r + 1) * row_len].copy_from_slice(&q[i]);
                }
            }
            println!(
                "ROTKV-PROBE layer={layer} tensor=K arm=k4v4chan rel_rms={:.5}",
                rel_rms(&k_all, &got)
            );
        }

        let eval_v = |name: &str, f: &mut dyn FnMut(&mut [f32])| {
            let mut got = v_all.clone();
            for r in 0..rows {
                f(&mut got[r * row_len..(r + 1) * row_len]);
            }
            println!(
                "ROTKV-PROBE layer={layer} tensor=V arm={name} rel_rms={:.5}",
                rel_rms(&v_all, &got)
            );
        };
        eval_v("fp8row", &mut |row| {
            for hr in row.chunks_mut(hd) {
                fp8_row_roundtrip_shipped_per_slot_head_scale(hr);
            }
        });
        eval_v("v4row", &mut |row| {
            for hr in row.chunks_mut(hd) {
                v4_row_roundtrip_shipped_e2m1_per_row_scale(hr);
            }
        });
        eval_v("plain16", &mut |row| {
            for hr in row.chunks_mut(hd) {
                nvfp4_16block_roundtrip_per_row_global(hr);
            }
        });
        for block in [16usize, 32, 64] {
            eval_v(&format!("rot{block}"), &mut |row| {
                for hr in row.chunks_mut(hd) {
                    rotate_head_row_in_hadamard_blocks(hr, block);
                    nvfp4_16block_roundtrip_per_row_global(hr);
                    rotate_head_row_in_hadamard_blocks(hr, block);
                }
            });
        }
    }
}
