#![cfg(feature = "wgpu")]

mod common;
use common::have_gpu;
use common::prompt_ids;
use common::tiny_config_qwen36_moe as tiny_config;
use common::tiny_weights;
mod ppl_common;

use nv_models::qwen3_5_moe::{LayerType, Qwen3MoeConfig};
use nv_models::qwen3_5_moe_wgpu as q3w;

const MROW_ENV: &str = "NV_Q3_WGPU_PF_MROW";

const TOKENPAR_ENV: &str = "NV_Q3_WGPU_PF_TOKENPAR";

const FLASH_TILED_ENV: &str = "NV_Q3_WGPU_PF_FLASH_TILED";

static ENV_LOCK_THESE_TESTS_MUTATE_PROCESS_GLOBAL_ENV_SO_MUST_NOT_INTERLEAVE:
    std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK_THESE_TESTS_MUTATE_PROCESS_GLOBAL_ENV_SO_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

const PROMPT_LENGTHS_COVER_FULL_M_ROW_CHUNKS_A_LEGACY_TAIL_AND_SHORTER_THAN_M: [usize; 4] =
    [40, 33, 17, 9];

fn per_token_logits(cfg: &Qwen3MoeConfig, hw: &q3w::HostWeights, ids: &[u32]) -> (u32, Vec<f32>) {
    let mut m = q3w::Qwen3MoeWgpu::new(cfg.clone(), hw, 64).expect("build per-token model");
    let (last, rest) = ids.split_last().expect("non-empty prompt");
    for t in rest {
        m.prefill_step(*t).expect("per-token prefill step");
    }
    m.decode_step_logits(*last).expect("decode")
}

#[test]
fn new_m_row_wgsl_validates_under_naga_without_a_gpu() {
    for (name, source) in q3w::pf_mrow_audit_sources() {
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
fn m_row_prefill_logits_are_bit_identical_to_per_token_replay() {
    let _env = env_lock();
    if !have_gpu() {
        panic!("needs a wgpu adapter; a skipped identity proof reads as a passed one");
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x51ee_d5ee_d001);

    std::env::remove_var(MROW_ENV);
    std::env::remove_var(TOKENPAR_ENV);
    {
        let default_model =
            q3w::Qwen3MoeWgpu::new(cfg.clone(), &hw, 64).expect("build default model");
        assert!(
            default_model.prefill_mrow_chunk_len() >= 2,
            "the M-row list is default ON since real-weight first-token parity and \
             1.84x at 8k (417dd6a8c); it did not engage with the env unset"
        );
        assert!(
            default_model.prefill_chunk_len() >= 2,
            "the legacy chunked replay must remain available as the tail path"
        );
    }
    std::env::set_var(MROW_ENV, "0");
    {
        let off_model = q3w::Qwen3MoeWgpu::new(cfg.clone(), &hw, 64).expect("build off model");
        assert_eq!(
            off_model.prefill_mrow_chunk_len(),
            0,
            "NV_Q3_WGPU_PF_MROW=0 is the documented escape hatch and must disable the list"
        );
    }

    std::env::set_var(MROW_ENV, "1");
    let mut seen: Vec<(bool, usize, u32, usize)> = Vec::new();
    let arms_cover_token_parallel_the_copy_escape_and_a_16_plus_8_gemm_tile_split =
        [(true, "16"), (true, "24"), (false, "16")];
    for (token_parallel_arm, m_env) in
        arms_cover_token_parallel_the_copy_escape_and_a_16_plus_8_gemm_tile_split
    {
        std::env::set_var("NV_WGPU_PREFILL_M", m_env);
        if token_parallel_arm {
            std::env::remove_var(TOKENPAR_ENV);
        } else {
            std::env::set_var(TOKENPAR_ENV, "0");
        }
        for (i, n) in PROMPT_LENGTHS_COVER_FULL_M_ROW_CHUNKS_A_LEGACY_TAIL_AND_SHORTER_THAN_M
            .into_iter()
            .enumerate()
        {
            let ids = prompt_ids(&cfg, n, i as u32);
            let (want_tok, want) = per_token_logits(&cfg, &hw, &ids);

            let mut m = q3w::Qwen3MoeWgpu::new(cfg.clone(), &hw, 64).expect("build M-row model");
            let mm = m.prefill_mrow_chunk_len();
            assert!(
                mm >= 2,
                "NV_Q3_WGPU_PF_MROW=1 is set but the M-row list did not engage (m={mm}); \
                 a bail reason was printed on stderr during build, and this comparison would \
                 silently degrade into legacy-vs-legacy"
            );
            let (mrow_passes, copy_passes) = m.prefill_mrow_pass_mix();
            assert!(
                mrow_passes > 0,
                "the M-row list built zero M-row passes; it is not the list this suite certifies"
            );
            assert!(
                token_parallel_arm || copy_passes > 0,
                "NV_Q3_WGPU_PF_TOKENPAR=0 must fall back to the per-token sequential \
                 copies, but the list has none ({mrow_passes} M-row, {copy_passes} copies)"
            );
            let (last, rest) = ids.split_last().expect("non-empty prompt");
            let done = m.prefill_tokens(rest).expect("M-row prefill");
            for t in &rest[done..] {
                m.prefill_step(*t).expect("tail prefill step");
            }
            let (got_tok, got) = m.decode_step_logits(*last).expect("decode");

            assert_eq!(want.len(), got.len(), "logit width changed at n={n}");
            let diff = want
                .iter()
                .zip(&got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                diff,
                0,
                "n={n} tokenpar={token_parallel_arm} (m={mm}, {done} of {} prompt tokens \
                 through prefill_tokens): {diff} of {} logits differ bit-for-bit between \
                 the M-row prefill list and per-token replay. The M-row list batches every \
                 projection over the chunk and runs the conv/recurrent/attention-core/top-k \
                 families either token-parallel (chunked kernels, default) or as per-token \
                 sequential copies at 256B-strided offsets (NV_Q3_WGPU_PF_TOKENPAR=0) -- \
                 if this fires, a stride or bind offset disagrees with what a kernel \
                 actually reads, or a sequential pass ran out of token order. Do NOT relax \
                 this to argmax or a tolerance: a tiny model's logits barely depend on \
                 context, so both weaker oracles pass while the DeltaNet state and every \
                 KV entry are wrong. The chunked kernels replicate the per-token fma order \
                 exactly, so bit-identity is the contract, not a tolerance.",
                n - 1,
                want.len()
            );
            assert_eq!(
                want_tok, got_tok,
                "argmax token differs at n={n} (replay {want_tok} vs M-row {got_tok})"
            );
            if n - 1 >= mm {
                assert!(
                    done >= mm,
                    "prompt length {n} covers a full M-row chunk of {mm} but prefill_tokens \
                     consumed only {done}, so the M-row submission never ran"
                );
            }
            seen.push((token_parallel_arm, n, want_tok, done));
        }
    }
    std::env::remove_var(TOKENPAR_ENV);
    std::env::remove_var("NV_WGPU_PREFILL_M");

    let base_ids = prompt_ids(&cfg, 40, 0);
    let mut poked_ids = base_ids.clone();
    let poke_at = base_ids.len() - 2;
    poked_ids[poke_at] = (poked_ids[poke_at] % (cfg.vocab_size as u32 - 1)) + 1;
    assert_ne!(
        poked_ids[poke_at], base_ids[poke_at],
        "the poke must actually change the token"
    );
    let (_, base_logits) = per_token_logits(&cfg, &hw, &base_ids);
    let (_, poked_logits) = per_token_logits(&cfg, &hw, &poked_ids);
    assert!(
        base_logits
            .iter()
            .zip(&poked_logits)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "changing one interior prompt token changed no logit bit, so the final decode never \
         reads the prefill state and the bit-identity gate above is comparing constants; \
         this tiny model cannot certify the M-row path"
    );
    eprintln!("[q3w-mrow-prefill] (tokenpar, len, argmax, tokens via prefill_tokens) = {seen:?}");
    std::env::remove_var(MROW_ENV);
}

const TILED_FLASH_TOL_IS_1E_4_OF_PEAK_LOGIT: f32 = 1e-4;

fn real_geometry_config_hd256_16h_2kv_exercises_all_8_dim_lanes_and_deep_streams() -> Qwen3MoeConfig
{
    Qwen3MoeConfig {
        head_dim: 256,
        num_attention_heads: 16,
        num_key_value_heads: 2,
        max_position_embeddings: 2048,
        ..tiny_config()
    }
}

#[test]
fn tiled_flash_logits_stay_within_1e_4_of_peak_vs_the_8_row_groups_since_only_exp_sum_association_changes(
) {
    let _env = env_lock();
    if !have_gpu() {
        panic!("needs a wgpu adapter; a skipped tolerance proof reads as a passed one");
    }
    std::env::set_var(MROW_ENV, "1");
    std::env::remove_var(TOKENPAR_ENV);
    let arms_cover_short_tile_full_32_plus_16_tail_deep_streams_and_real_hd256_16h_geometry: [(
        fn() -> Qwen3MoeConfig,
        usize,
        usize,
    ); 4] = [
        (tiny_config, 16, 40),
        (tiny_config, 48, 50),
        (tiny_config, 16, 300),
        (
            real_geometry_config_hd256_16h_2kv_exercises_all_8_dim_lanes_and_deep_streams,
            64,
            1050,
        ),
    ];
    for (cfg_fn, m_env, n) in
        arms_cover_short_tile_full_32_plus_16_tail_deep_streams_and_real_hd256_16h_geometry
    {
        let cfg = cfg_fn();
        let hw = tiny_weights(&cfg, 0x51ee_d5ee_d001);
        std::env::set_var("NV_WGPU_PREFILL_M", m_env.to_string());
        let ids = prompt_ids(&cfg, n, 7);
        let run = |tiled: Option<&str>| {
            if let Some(arm) = tiled {
                std::env::set_var(FLASH_TILED_ENV, arm);
            } else {
                std::env::set_var(FLASH_TILED_ENV, "0");
                assert!(
                    !q3w::pf_flash_tiled_default_on_since_slotml_solo_nll_plus_p006_nats_under_p01_bound_and_557_vs_431_tok_s_user_signed_off(),
                    "the grouped control arm needs the per-8-row-group flash pairs; tiled ships \
                     default-on now, so unset would silently compare tiled to tiled"
                );
            }
            let mut m = q3w::Qwen3MoeWgpu::new(cfg.clone(), &hw, n + 16).expect("build model");
            let mm = m.prefill_mrow_chunk_len();
            assert!(
                mm >= 2,
                "the M-row list did not engage (m={mm}); this comparison would silently \
                 degrade into legacy-vs-legacy"
            );
            let passes = m.prefill_mrow_pass_count();
            let (last, rest) = ids.split_last().expect("non-empty prompt");
            let done = m.prefill_tokens(rest).expect("M-row prefill");
            assert!(
                done >= mm,
                "prompt length {n} covers a full M-row chunk of {mm} but prefill_tokens \
                 consumed only {done}, so the flash arm under test never ran"
            );
            for t in &rest[done..] {
                m.prefill_step(*t).expect("tail prefill step");
            }
            let (tok, logits) = m.decode_step_logits(*last).expect("decode");
            (tok, logits, passes)
        };
        let (tok_g, grouped, passes_grouped) = run(None);
        for arm in ["1", "2"] {
            let (tok_t, tiled, passes_tiled) = run(Some(arm));
            std::env::remove_var(FLASH_TILED_ENV);
            assert!(
                passes_tiled < passes_grouped,
                "NV_Q3_WGPU_PF_FLASH_TILED={arm} must replace the per-8-row-group flash \
                 dispatch pairs with one tiled stage1 plus one stage2 per attention layer, \
                 so the M-row pass count must drop; got {passes_tiled} vs {passes_grouped} \
                 at m={m_env} -- the tiled kernel never engaged and this tolerance gate \
                 certified nothing"
            );
            let peak = grouped.iter().fold(0f32, |a, b| a.max(b.abs()));
            assert!(peak > 0.0, "all-zero grouped logits cannot anchor a relative tolerance");
            let (worst_i, max_diff) = grouped
                .iter()
                .zip(&tiled)
                .map(|(a, b)| (a - b).abs())
                .enumerate()
                .fold((0usize, 0f32), |acc, (i, d)| if d > acc.1 { (i, d) } else { acc });
            eprintln!(
                "[q3w-tiled-flash-tol] arm={arm} m={m_env} n={n} peak={peak:.4} \
                 max_diff={max_diff:.3e} rel={:.3e} at logit {worst_i} \
                 (passes {passes_grouped} -> {passes_tiled})",
                max_diff / peak
            );
            assert!(
                max_diff <= TILED_FLASH_TOL_IS_1E_4_OF_PEAK_LOGIT * peak,
                "tiled flash arm {arm} diverged from the 8-row groups by {max_diff:.3e} at \
                 logit {worst_i} (peak |logit| {peak:.4}, m={m_env}, n={n}). Arm 1 only \
                 reassociates the online-softmax exp sums (block-of-8 max/exp instead of \
                 per-position, one warp per 4 rows instead of 8 warps per position); arm 2 \
                 keeps one m/l chain per j slot exactly like the grouped kernel's 8 warps \
                 and shares only the acc scale, so either way the perturbation is ~1e-6 \
                 relative; 1e-4 of peak is pure headroom, so a miss here is a mask, \
                 tile-offset, or scratch-slot indexing bug, not rounding"
            );
            assert_eq!(
                tok_g, tok_t,
                "argmax token differs at arm={arm}, m={m_env}, n={n} (grouped {tok_g} vs \
                 tiled {tok_t}) while every logit sits inside tolerance; investigate before \
                 trusting either"
            );
        }
    }
    std::env::remove_var("NV_WGPU_PREFILL_M");
    std::env::remove_var(MROW_ENV);
}

fn tiled_arm_from_outer_env_where_2_selects_the_slotml_variant() -> &'static str {
    match std::env::var(FLASH_TILED_ENV).ok().as_deref() {
        Some("2") => "2",
        _ => "1",
    }
}

fn qwen36_snapshot_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    std::env::var("NV_QWEN36_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(&home).join(
                ".cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots/e850c696e6d75f965367e816c16bc7dacd955ffa",
            )
        })
}

const CONTINUATION_GREEDY_TOKENS_64_LONG_ENOUGH_TO_SHOW_DIVERGENCE_OR_RECONVERGENCE: usize = 64;

#[test]
#[ignore = "loads the qwen3.6-35B MoE on wgpu twice; set NV_QWEN36_MROW_TEST=1 and \
            NV_PPL_CORPUS -- greedy-decodes 64 tokens from a tiled-prefilled vs a \
            grouped-prefilled real-corpus state and reports token-level agreement; \
            the grouped M-row state is bit-identical to the legacy replay, so this \
            measures exactly the tiled kernel's semantic effect on prefill state"]
fn real_weights_tiled_prefill_continuation_agreement_vs_grouped_greedy_64() {
    let _env = env_lock();
    if std::env::var("NV_QWEN36_MROW_TEST").is_err() {
        eprintln!("skip: NV_QWEN36_MROW_TEST not set");
        return;
    }
    let dir = qwen36_snapshot_dir();
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depth: usize = std::env::var("NV_MROW_PREFILL_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let cfg = Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    let corpus = ppl_common::corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let mut tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .expect("tokenizer.json beside the checkpoint");
    tok.with_truncation(None)
        .expect("clear the tokenizer's 4096 truncation so an 8k prefill slice exists");
    let ids = ppl_common::first_n_corpus_tokens_after_tokenization(&tok, &corpus, depth);
    let steps = CONTINUATION_GREEDY_TOKENS_64_LONG_ENOUGH_TO_SHOW_DIVERGENCE_OR_RECONVERGENCE;
    let arm = tiled_arm_from_outer_env_where_2_selects_the_slotml_variant();

    let run = |tiled: bool| -> Vec<u32> {
        std::env::set_var(MROW_ENV, "1");
        if tiled {
            std::env::set_var(FLASH_TILED_ENV, arm);
        } else {
            std::env::remove_var(FLASH_TILED_ENV);
        }
        let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
            .expect("open safetensors");
        let mut gpu = q3w::Qwen3MoeWgpu::from_loader(cfg.clone(), &loader, depth + steps + 16)
            .expect("build from loader");
        drop(loader);
        let mm = gpu.prefill_mrow_chunk_len();
        assert!(
            mm >= 2,
            "the M-row list did not engage (m={mm}); both arms must prefill through the \
             M-row list or this compares nothing"
        );
        let (last, rest) = ids.split_last().expect("non-empty corpus slice");
        let done = gpu.prefill_tokens(rest).expect("prefill_tokens");
        assert!(
            done >= mm,
            "prefill_tokens consumed only {done} of {} corpus tokens; the M-row \
             submission (and with it the flash arm under test) never ran",
            rest.len()
        );
        for t in &rest[done..] {
            gpu.prefill_step(*t).expect("tail prefill step");
        }
        let mut t = *last;
        let mut out = Vec::with_capacity(steps);
        for _ in 0..steps {
            t = gpu.decode_step(t).expect("greedy decode step");
            out.push(t);
        }
        std::env::remove_var(FLASH_TILED_ENV);
        out
    };

    let grouped = run(false);
    let tiled = run(true);
    std::env::remove_var(MROW_ENV);
    let agree = grouped.iter().zip(&tiled).filter(|(a, b)| a == b).count();
    let first_div = grouped.iter().zip(&tiled).position(|(a, b)| a != b);
    let suffix_agree = grouped
        .iter()
        .rev()
        .zip(tiled.iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let mut longest_common_run_at_any_offset_captures_shifted_reconvergence = 0usize;
    for ga in 0..grouped.len() {
        for ta in 0..tiled.len() {
            let mut run = 0usize;
            while ga + run < grouped.len()
                && ta + run < tiled.len()
                && grouped[ga + run] == tiled[ta + run]
            {
                run += 1;
            }
            longest_common_run_at_any_offset_captures_shifted_reconvergence =
                longest_common_run_at_any_offset_captures_shifted_reconvergence.max(run);
        }
    }
    eprintln!(
        "[q3w-tiled-continuation] arm={arm} depth={depth} steps={steps} agree={agree}/{steps} \
         first_divergence={first_div:?} elementwise_suffix_agree={suffix_agree} \
         longest_common_run_any_offset={longest_common_run_at_any_offset_captures_shifted_reconvergence}"
    );
    eprintln!("[q3w-tiled-continuation] grouped={grouped:?}");
    eprintln!("[q3w-tiled-continuation] tiled  ={tiled:?}");
    let decode = |seq: &[u32]| -> String {
        tok.decode(seq, false)
            .unwrap_or_else(|e| format!("<decode failed: {e}>"))
    };
    eprintln!("[q3w-tiled-continuation] grouped text: {:?}", decode(&grouped));
    eprintln!("[q3w-tiled-continuation] tiled text:   {:?}", decode(&tiled));
}

const NLL_SCORED_POSITIONS_512_MATCHES_THE_CANONICAL_PPL_BLOCK: usize = 512;

const NLL_QUALITY_NEUTRAL_DELTA_BOUND_NATS: f64 = 0.01;

#[test]
#[ignore = "loads the qwen3.6-35B MoE on wgpu twice; set NV_QWEN36_MROW_TEST=1 and \
            NV_PPL_CORPUS -- teacher-forced NLL over the next 512 corpus tokens after a \
            grouped-prefilled vs a tiled-prefilled state; the delta isolates prefill-state \
            fidelity under the tiled kernel's reassociation because both arms score the \
            same continuation with the same decode path"]
fn real_weights_tiled_vs_grouped_prefill_teacher_forced_nll_next_512() {
    let _env = env_lock();
    if std::env::var("NV_QWEN36_MROW_TEST").is_err() {
        eprintln!("skip: NV_QWEN36_MROW_TEST not set");
        return;
    }
    let dir = qwen36_snapshot_dir();
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depth: usize = std::env::var("NV_MROW_PREFILL_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let score_n = NLL_SCORED_POSITIONS_512_MATCHES_THE_CANONICAL_PPL_BLOCK;
    let cfg = Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    let corpus = ppl_common::corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let mut tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .expect("tokenizer.json beside the checkpoint");
    tok.with_truncation(None)
        .expect("clear the tokenizer's 4096 truncation so the prefill slice exists");
    let offset: usize = std::env::var("NV_MROW_CORPUS_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let all = ppl_common::first_n_corpus_tokens_after_tokenization(
        &tok,
        &corpus,
        offset + depth + score_n + 1,
    );
    let ids = &all[offset..];
    let arm = tiled_arm_from_outer_env_where_2_selects_the_slotml_variant();

    let run = |tiled: bool| -> ppl_common::TeacherForcedNllFp32SoftmaxF64Sum {
        std::env::set_var(MROW_ENV, "1");
        if tiled {
            std::env::set_var(FLASH_TILED_ENV, arm);
        } else {
            std::env::remove_var(FLASH_TILED_ENV);
        }
        let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
            .expect("open safetensors");
        let mut gpu = q3w::Qwen3MoeWgpu::from_loader(cfg.clone(), &loader, depth + score_n + 32)
            .expect("build from loader");
        drop(loader);
        let mm = gpu.prefill_mrow_chunk_len();
        assert!(
            mm >= 2,
            "the M-row list did not engage (m={mm}); both arms must prefill through the \
             M-row list or the delta measures nothing"
        );
        let prompt = &ids[..depth];
        let done = gpu.prefill_tokens(prompt).expect("prefill_tokens");
        assert!(
            done >= mm,
            "prefill_tokens consumed only {done} of {depth}; the M-row submission (and \
             with it the flash arm under test) never ran"
        );
        for t in &prompt[done..] {
            gpu.prefill_step(*t).expect("tail prefill step");
        }
        let mut acc = ppl_common::TeacherForcedNllFp32SoftmaxF64Sum::new();
        for i in depth..depth + score_n {
            let (_, logits) = gpu.decode_step_logits(ids[i]).expect("teacher-forced step");
            acc.add_position_full_vocab_row(&logits, ids[i + 1]);
        }
        std::env::remove_var(FLASH_TILED_ENV);
        assert_eq!(
            acc.scored_positions, score_n,
            "the teacher-forced loop must score exactly the canonical block"
        );
        acc
    };

    let grouped = run(false);
    let tiled = run(true);
    std::env::remove_var(MROW_ENV);
    let g_nll = grouped.sum_neg_ln_p_f64 / grouped.scored_positions as f64;
    let t_nll = tiled.sum_neg_ln_p_f64 / tiled.scored_positions as f64;
    let delta = t_nll - g_nll;
    let verdict = if delta.abs() <= NLL_QUALITY_NEUTRAL_DELTA_BOUND_NATS {
        "quality-neutral"
    } else if delta > 0.0 {
        "grouped better"
    } else {
        "tiled better"
    };
    eprintln!(
        "[q3w-tiled-nll] arm={arm} depth={depth} offset={offset} scored={score_n} \
         grouped_mean_nll={g_nll:.5} (ppl={:.3}, top1={:.4}) tiled_mean_nll={t_nll:.5} \
         (ppl={:.3}, top1={:.4}) delta_nats={delta:+.5} \
         bound={NLL_QUALITY_NEUTRAL_DELTA_BOUND_NATS} verdict={verdict}",
        grouped.perplexity_exp_of_mean_neg_ln_p(),
        grouped.top1_accuracy(),
        tiled.perplexity_exp_of_mean_neg_ln_p(),
        tiled.top1_accuracy(),
    );
}

const PIN_GROUPED_WHEN_UNSET_BECAUSE_THIS_GATE_ASSERTS_EXACT_MROW_MACHINERY_PARITY_AND_SLOTML_IS_ADJUDICATED_BY_THE_NLL_SUITE_NOT_ARGMAX_NEAR_TIES:
    &str = "0";

#[test]
#[ignore = "loads the qwen3.6-35B MoE on wgpu; set NV_QWEN36_MROW_TEST=1 -- prefill wall \
            time and first-token parity, M-row list vs legacy chunked replay"]
fn real_weights_m_row_prefill_time_and_first_token_parity() {
    let _env = env_lock();
    if std::env::var("NV_QWEN36_MROW_TEST").is_err() {
        eprintln!("skip: NV_QWEN36_MROW_TEST not set");
        return;
    }
    if std::env::var(FLASH_TILED_ENV).is_err() {
        std::env::set_var(
            FLASH_TILED_ENV,
            PIN_GROUPED_WHEN_UNSET_BECAUSE_THIS_GATE_ASSERTS_EXACT_MROW_MACHINERY_PARITY_AND_SLOTML_IS_ADJUDICATED_BY_THE_NLL_SUITE_NOT_ARGMAX_NEAR_TIES,
        );
    }
    let dir = qwen36_snapshot_dir();
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depth: usize = std::env::var("NV_MROW_PREFILL_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let cfg = Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    let ids: Vec<u32> = (0..depth as u32).map(|i| 2000 + (i % 30000)).collect();

    let run = |mrow: bool| -> (f64, u32, usize, usize, Vec<f32>) {
        if mrow {
            std::env::set_var(MROW_ENV, "1");
        } else {
            std::env::set_var(MROW_ENV, "0");
        }
        let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
            .expect("open safetensors");
        let mut gpu = q3w::Qwen3MoeWgpu::from_loader(cfg.clone(), &loader, depth + 16)
            .expect("build from loader");
        drop(loader);
        let mm = gpu.prefill_mrow_chunk_len();
        assert_eq!(
            mm >= 2,
            mrow,
            "the M-row list engagement must follow the env gate (mrow={mrow}, m={mm}); \
             a bail reason was printed on stderr during build"
        );
        let (last, rest) = ids.split_last().expect("non-empty prompt");
        let t0 = std::time::Instant::now();
        let done = gpu.prefill_tokens(rest).expect("prefill_tokens");
        for t in &rest[done..] {
            gpu.prefill_step(*t).expect("tail prefill step");
        }
        let (tok, logits) = gpu.decode_step_logits(*last).expect("first decode");
        let secs = t0.elapsed().as_secs_f64();
        let nan = logits.iter().filter(|v| v.is_nan()).count();
        let inf = logits.iter().filter(|v| v.is_infinite()).count();
        let peak = logits.iter().fold(0f32, |a, b| a.max(b.abs()));
        let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!(
            "[q3w-mrow-real-logits] mrow={mrow} nan={nan} inf={inf} peak={peak:.4} top={:?}",
            &top[..5]
        );
        (secs, tok, done, mm, logits)
    };

    let profiling_run_times_only_the_m_row_list =
        nv_kernels::wgpu_backend::dispatch::profile::enabled();
    let (legacy_s, legacy_tok, legacy_done, _, legacy_logits) =
        if profiling_run_times_only_the_m_row_list {
            (f64::NAN, 0, 0, 0, Vec::new())
        } else {
            run(false)
        };
    let (mrow_s, mrow_tok, mrow_done, mm, mrow_logits) = run(true);
    eprintln!(
        "[q3w-mrow-real] depth={depth} legacy_prefill_s={legacy_s:.1} (done={legacy_done}) \
         mrow_prefill_s={mrow_s:.1} (done={mrow_done}, m={mm}) first_token={mrow_tok}"
    );
    if legacy_logits.len() == mrow_logits.len() && !legacy_logits.is_empty() {
        let (worst_i, max_diff) = legacy_logits
            .iter()
            .zip(&mrow_logits)
            .map(|(a, b)| (a - b).abs())
            .enumerate()
            .fold((0usize, 0f32), |acc, (i, d)| if d > acc.1 { (i, d) } else { acc });
        eprintln!(
            "[q3w-mrow-real-diff] max|legacy-mrow| logit diff {max_diff:.3e} at {worst_i}"
        );
    }
    assert!(
        profiling_run_times_only_the_m_row_list || legacy_tok == mrow_tok,
        "first token after a {depth}-token prefill differs between the legacy chunked \
         replay ({legacy_tok}) and the M-row list ({mrow_tok}); the tiny-model \
         bit-identity suite localizes which pass family broke"
    );
    assert!(
        mrow_done >= mm,
        "the M-row model consumed only {mrow_done} tokens through prefill_tokens; the \
         M-row submission never ran"
    );
    {
        use nv_kernels::wgpu_backend::dispatch::profile;
        if profile::enabled() {
            profile::flush();
            let mut rows = profile::report();
            let total: f64 = rows.iter().map(|r| r.2).sum();
            rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            for (label, count, ns) in rows.into_iter().take(48) {
                eprintln!(
                    "[q3w-mrow-prof] {label} n={count} total={:.1}ms avg={:.2}us share={:.1}%",
                    ns / 1.0e6,
                    ns / 1.0e3 / count.max(1) as f64,
                    100.0 * ns / total.max(1.0)
                );
            }
            eprintln!("[q3w-mrow-prof] TOTAL {:.1}ms", total / 1.0e6);
        }
    }
    std::env::remove_var(MROW_ENV);
}
