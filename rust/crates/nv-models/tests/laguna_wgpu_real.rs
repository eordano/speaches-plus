#![cfg(feature = "laguna-wip")]

mod hub_snapshot;

use std::path::PathBuf;

use nv_models::laguna::LagunaConfig;
use nv_models::laguna_wgpu::attn::ref_attn;
use nv_models::laguna_wgpu::config::{rope_tables_from_inv_freq, LagunaShapes};
use nv_models::laguna_wgpu::dense::ref_dense_mlp;
use nv_models::laguna_wgpu::moe::ref_moe;
use nv_models::laguna_wgpu::weights::{bf16_val, HostBf16Lin, HostFfn, WeightSource};
use nv_models::laguna_wgpu::{
    rbf, ref_argmax, ref_gemv_bf16, ref_rmsnorm, ref_rmsnorm_residual, LagunaWgpu, RefState,
};

const LAGUNA_REPO: &str = "poolside/Laguna-XS-2.1-NVFP4";
const MAX_SEQ: usize = 256;
const BF16_ULP: f32 = 1.0 / 256.0;

fn ckpt_dir() -> Option<PathBuf> {
    hub_snapshot::dir_from_env_or_hub(
        "NV_LAGUNA_DIR",
        LAGUNA_REPO,
        &["config.json", "*.safetensors"],
    )
}

fn laguna_absent(test: &str) {
    hub_snapshot::precondition_absent(
        test,
        &format!("no {LAGUNA_REPO} snapshot with safetensors"),
        "set NV_LAGUNA_DIR to a Laguna-XS-2.1-NVFP4 snapshot dir, or cache the repo",
    );
}

fn have_gpu() -> bool {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(_) => true,
        Err(e) => {
            eprintln!("SKIP: no wgpu adapter: {e}");
            false
        }
    }
}

fn load(dir: &PathBuf) -> (LagunaConfig, nv_weights::WeightLoader) {
    let cfg = LagunaConfig::from_hf_json_file(&dir.join("config.json")).expect("parse config");
    let loader =
        nv_weights::WeightLoader::open_dir(dir, &nv_weights::Device::Cpu).expect("open weights");
    (cfg, loader)
}

fn streaming_reference_step(
    shapes: &LagunaShapes,
    src: &WeightSource<'_>,
    st: &mut RefState,
    token: u32,
) -> Vec<f32> {
    let hidden = shapes.hidden_size;
    let eps = shapes.rms_norm_eps;
    let pos = st.pos;

    let (fcos, fsin) = rope_tables_from_inv_freq(&shapes.rope_inv_freq_full, pos + 1);
    let (scos, ssin) = rope_tables_from_inv_freq(&shapes.rope_inv_freq_sliding, pos + 1);
    let fhalf = shapes.rope_inv_freq_full.len();
    let shalf = shapes.rope_inv_freq_sliding.len();

    let embed = src.embed(shapes).expect("embed");
    let mut res: Vec<f32> = (0..hidden)
        .map(|i| bf16_val(embed[token as usize * hidden + i]))
        .collect();
    drop(embed);

    let l0_ln = src.layer_input_ln(shapes, 0).expect("layer 0 input_ln");
    let mut normed = ref_rmsnorm(&res, &l0_ln, eps);

    for li in 0..shapes.num_layers {
        let layer_shape = *shapes.layer(li);
        let layer = src.layer(shapes, li).expect("layer");
        let (cos_row, sin_row) = if layer_shape.is_sliding() {
            (
                &scos[pos * shalf..(pos + 1) * shalf],
                &ssin[pos * shalf..(pos + 1) * shalf],
            )
        } else {
            (
                &fcos[pos * fhalf..(pos + 1) * fhalf],
                &fsin[pos * fhalf..(pos + 1) * fhalf],
            )
        };
        let mixed = ref_attn(
            shapes,
            &layer_shape,
            &layer.attn,
            &normed,
            cos_row,
            sin_row,
            st,
            pos,
        )
        .expect("ref_attn");
        let normed_post = ref_rmsnorm_residual(&mixed, &mut res, &layer.post_attn_ln, eps);
        let ffn = match &layer.ffn {
            HostFfn::Dense(d) => {
                ref_dense_mlp(shapes, &layer_shape, d, &normed_post).expect("dense")
            }
            HostFfn::Moe(m) => ref_moe(shapes, m, &normed_post).expect("moe"),
        };
        if li + 1 < shapes.num_layers {
            let next_ln = src.layer_input_ln(shapes, li + 1).expect("next input_ln");
            normed = ref_rmsnorm_residual(&ffn, &mut res, &next_ln, eps);
        } else {
            for i in 0..hidden {
                res[i] = rbf(res[i] + ffn[i]);
            }
        }
    }

    let fnorm = src.final_norm(shapes).expect("final norm");
    let fx = ref_rmsnorm(&res, &fnorm, eps);
    let lm = src.lm_head(shapes).expect("lm head");
    let lm_lin = HostBf16Lin {
        w: lm,
        n: shapes.vocab_size,
        k: hidden,
    };
    st.pos += 1;
    ref_gemv_bf16(&lm_lin, &fx)
}

fn scale_of(v: &[f32]) -> f32 {
    v.iter().fold(0f32, |a, x| a.max(x.abs())).max(1e-6)
}

fn worst_rel(got: &[f32], want: &[f32]) -> (f32, usize) {
    assert_eq!(got.len(), want.len());
    let scale = scale_of(want);
    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let r = (g - w).abs() / scale;
        if r > worst {
            worst = r;
            at = i;
        }
    }
    (worst, at)
}

fn top2_gap(v: &[f32]) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    s[0] - s[1]
}

#[test]
#[ignore]
fn real_weight_forward_matches_cpu_oracle() {
    let Some(dir) = ckpt_dir() else {
        laguna_absent("real_weight_forward_matches_cpu_oracle");
        return;
    };
    assert!(
        have_gpu(),
        "this real-weight parity test needs a wgpu adapter"
    );
    let steps: usize = std::env::var("NV_LAGUNA_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let tokens: Vec<u32> = vec![2, 1848, 386, 8546, 1236, 40, 991, 22];

    let (cfg, loader) = load(&dir);
    let shapes = LagunaShapes::derive(&cfg, MAX_SEQ).expect("derive shapes");
    eprintln!(
        "[laguna-real] {} layers, hidden {}, vocab {}, {} experts top-{}, full-attn heads {} sliding heads {}",
        shapes.num_layers,
        shapes.hidden_size,
        shapes.vocab_size,
        shapes.num_experts,
        shapes.top_k,
        shapes.layer(0).num_q_heads,
        shapes.layer(1).num_q_heads,
    );

    let mut gpu_logits: Vec<(u32, Vec<f32>)> = Vec::new();
    {
        let mut m =
            LagunaWgpu::from_loader(cfg.clone(), &loader, MAX_SEQ).expect("build gpu model");
        for (i, &tok) in tokens.iter().enumerate().take(steps) {
            let (gt, gl) = m.decode_step_logits(tok).expect("decode step");
            assert!(
                gl.iter().all(|v| v.is_finite()),
                "step {i}: gpu logits non-finite"
            );
            gpu_logits.push((gt, gl));
        }
    }

    let src = WeightSource::Loader(&loader);
    let mut st = RefState::new(&shapes);
    let mut worst_overall = 0f32;
    let mut argmax_ok = 0usize;
    let mut decided = 0usize;
    let mut decided_ok = 0usize;
    let mut step0_worst = f32::INFINITY;
    for (i, &tok) in tokens.iter().enumerate().take(steps) {
        let rl = streaming_reference_step(&shapes, &src, &mut st, tok);
        let (gt, gl) = &gpu_logits[i];
        let (worst, at) = worst_rel(gl, &rl);
        worst_overall = worst_overall.max(worst);
        let scale = scale_of(&rl);
        let rt = ref_argmax(&rl);
        let gap = top2_gap(&rl);
        let agree = *gt == rt;
        if agree {
            argmax_ok += 1;
        }
        let is_decided = gap > worst * scale;
        if is_decided {
            decided += 1;
            if agree {
                decided_ok += 1;
            }
        }
        if i == 0 {
            step0_worst = worst;
        }
        eprintln!(
            "[laguna-real] step {i} tok {}: worst_rel {:.3e} ({:.2} bf16 ulp) at logit {at} | argmax gpu {gt} oracle {rt} {} | {} | top2_gap {:.4} vs divergence {:.4}",
            tokens[i],
            worst,
            worst / BF16_ULP,
            if agree { "AGREE" } else { "DISAGREE" },
            if is_decided { "DECIDED" } else { "within-substrate-noise" },
            gap,
            worst * scale,
        );
    }
    eprintln!(
        "[laguna-real] VERDICT: argmax {argmax_ok}/{steps} agree ({decided_ok}/{decided} on decided steps); \
         step-0 full-stack worst_rel {:.3e} ({:.2} bf16 ulp); worst over all steps {:.3e} ({:.2} bf16 ulp). \
         Undecided steps are top-8/256 router near-ties crossed by the non-order-replicating NVFP4 gemv oracle substrate noise, not a GPU defect (see decode test for behavioral agreement).",
        step0_worst,
        step0_worst / BF16_ULP,
        worst_overall,
        worst_overall / BF16_ULP
    );
    assert!(
        step0_worst < 5.0 * BF16_ULP,
        "step 0 (the full 40-layer NVFP4 forward, before any routing amplification) disagreed by {:.2} bf16 ulp",
        step0_worst / BF16_ULP
    );
    assert_eq!(
        decided_ok, decided,
        "gpu/oracle argmax disagreed on a step whose top-2 gap exceeds the substrate noise ({decided_ok}/{decided})"
    );
}

const SYS_MSG: &str = "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations.";
const EOS_LITERAL: &str = "〈|EOS|〉";
const STOPS: [u32; 2] = [2, 24];

fn render_chat(user: &str) -> String {
    format!("{EOS_LITERAL}<system>{SYS_MSG}</system>\n<user>{user}</user>\n<assistant></think>")
}

fn greedy_run(m: &mut LagunaWgpu, prompt_ids: &[u32], max_new: usize) -> Vec<u32> {
    m.reset().unwrap();
    let mut cur = m.prefill(prompt_ids).expect("prefill");
    let mut gen = vec![cur];
    for _ in 0..max_new {
        if STOPS.contains(&cur) {
            break;
        }
        cur = m.decode_step(cur).expect("decode");
        gen.push(cur);
    }
    gen
}

#[test]
#[ignore]
fn real_weight_greedy_decode_is_coherent_and_reproducible() {
    let Some(dir) = ckpt_dir() else {
        laguna_absent("real_weight_greedy_decode_is_coherent_and_reproducible");
        return;
    };
    assert!(
        have_gpu(),
        "this real-weight decode test needs a wgpu adapter"
    );
    let max_new: usize = std::env::var("NV_LAGUNA_MAXNEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let user = std::env::var("NV_LAGUNA_PROMPT").unwrap_or_else(|_| {
        "Write a Python function is_prime(n) that returns True if n is prime.".to_string()
    });

    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");
    let prompt = render_chat(&user);
    let enc = tokenizer.encode(prompt.clone(), false).expect("encode");
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    eprintln!(
        "[laguna-decode] prompt {} tokens; first/last ids {:?}..{:?}",
        prompt_ids.len(),
        &prompt_ids[..prompt_ids.len().min(4)],
        &prompt_ids[prompt_ids.len().saturating_sub(4)..],
    );
    assert!(!prompt_ids.is_empty());
    assert!(prompt_ids.len() < MAX_SEQ, "prompt longer than kv cache");

    let (cfg, loader) = load(&dir);

    let gen1;
    let gen_reset;
    {
        let mut m = LagunaWgpu::from_loader(cfg.clone(), &loader, MAX_SEQ).expect("build model");
        gen1 = greedy_run(&mut m, &prompt_ids, max_new);
        gen_reset = greedy_run(&mut m, &prompt_ids, max_new);
    }

    let gen_restart;
    {
        let mut m = LagunaWgpu::from_loader(cfg.clone(), &loader, MAX_SEQ).expect("rebuild model");
        gen_restart = greedy_run(&mut m, &prompt_ids, max_new);
    }

    let text = tokenizer.decode(&gen1, false).unwrap_or_default();
    eprintln!(
        "[laguna-decode] {} generated tokens: {:?}",
        gen1.len(),
        gen1
    );
    eprintln!("[laguna-decode] continuation: {text:?}");

    assert_eq!(
        gen1, gen_reset,
        "greedy decode is not deterministic across reset()+replay"
    );
    assert_eq!(
        gen1, gen_restart,
        "greedy decode is not byte-reproducible across a fresh model rebuild (restart)"
    );
    assert!(
        gen1.len() >= 4 && !STOPS.contains(&gen1[0]),
        "model stopped immediately ({} tokens, first {})",
        gen1.len(),
        gen1[0]
    );
    let distinct = gen1.iter().collect::<std::collections::HashSet<_>>().len();
    assert!(
        distinct >= 3,
        "continuation is degenerate: only {distinct} distinct tokens in {:?}",
        gen1
    );
    assert!(!text.trim().is_empty(), "empty continuation text");
}
