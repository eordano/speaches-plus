#![cfg(feature = "wgpu")]

mod common;
use common::bf16_lin_q3w as bf16_lin;
use common::bit_diff;
use common::have_gpu;
use common::LcgOddSeedShift33SignedUnit as Lcg;
use common::norm_vec;
use common::nvfp4;
use common::prompt_ids;
use common::tiny_config_qwen36_moe as tiny_config;
use nv_models::qwen3_5_moe::{LayerType, Qwen3MoeConfig};
use nv_models::qwen3_5_moe_wgpu as q3w;
use q3w::EmbedRowsSplice;

const PREFILL_M_ENV: &str = "NV_WGPU_PREFILL_M";
const MROW_ENV: &str = "NV_Q3_WGPU_PF_MROW";

static ENV_LOCK_THESE_TESTS_MUTATE_PROCESS_GLOBAL_ENV_SO_MUST_NOT_INTERLEAVE:
    std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK_THESE_TESTS_MUTATE_PROCESS_GLOBAL_ENV_SO_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

const DELTA_A_LOG_BIAS_KEEPS_RECURRENT_STATE_OBSERVABLE_PAST_A_20_TOKEN_PROMPT: f32 = -4.0;

fn slow_decay_a_log(r: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            half::bf16::from_f32(
                DELTA_A_LOG_BIAS_KEEPS_RECURRENT_STATE_OBSERVABLE_PAST_A_20_TOKEN_PROMPT
                    + r.next_f32() * 0.3,
            )
            .to_f32()
        })
        .collect()
}

fn tiny_weights(cfg: &Qwen3MoeConfig, seed: u64) -> q3w::HostWeights {
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

    let mut layers = Vec::new();
    for li in 0..cfg.num_hidden_layers {
        let mixer = match cfg.layer_types[li] {
            LayerType::LinearAttention => q3w::HostMixer::Delta(Box::new(q3w::HostDeltaNet {
                in_proj_qkv: bf16_lin(&mut r, conv_dim, hidden, 0.12),
                in_proj_z: bf16_lin(&mut r, value_dim, hidden, 0.12),
                in_proj_ab: bf16_lin(&mut r, 2 * n_v, hidden, 0.12),
                conv1d: r.f32_vec(conv_dim * ks, 0.4),
                a_log: slow_decay_a_log(&mut r, n_v),
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
        let gates: Vec<_> = (0..cfg.num_experts)
            .map(|_| nvfp4(&mut r, inter, hidden, 0.15))
            .collect();
        let ups: Vec<_> = (0..cfg.num_experts)
            .map(|_| nvfp4(&mut r, inter, hidden, 0.15))
            .collect();
        let downs: Vec<_> = (0..cfg.num_experts)
            .map(|_| nvfp4(&mut r, hidden, inter, 0.15))
            .collect();
        layers.push(q3w::HostLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            mixer,
            moe: q3w::HostMoe {
                router: bf16_lin(&mut r, cfg.num_experts, hidden, 0.3),
                experts_gate: q3w::stack_nvfp4_host(&gates),
                experts_up: q3w::stack_nvfp4_host(&ups),
                experts_down: q3w::stack_nvfp4_host(&downs),
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

fn embed_row(hw: &q3w::HostWeights, hidden: usize, id: u32) -> Vec<u16> {
    hw.embed[id as usize * hidden..(id as usize + 1) * hidden].to_vec()
}

fn splices_from_prompt(
    hw: &q3w::HostWeights,
    hidden: usize,
    ids: &[u32],
    runs: &[(usize, usize)],
) -> Vec<EmbedRowsSplice> {
    runs.iter()
        .map(|&(pos, n_rows)| {
            let mut rows_bf16 = Vec::with_capacity(n_rows * hidden);
            for r in 0..n_rows {
                rows_bf16.extend_from_slice(&embed_row(hw, hidden, ids[pos + r]));
            }
            EmbedRowsSplice { position: pos, rows_bf16 }
        })
        .collect()
}

fn plain_prefill_logits(
    cfg: &Qwen3MoeConfig,
    hw: &q3w::HostWeights,
    ids: &[u32],
) -> (u32, Vec<f32>) {
    let mut m = q3w::Qwen3MoeWgpu::new(cfg.clone(), hw, 64).expect("build plain model");
    let (last, rest) = ids.split_last().expect("non-empty prompt");
    let done = m.prefill_tokens(rest).expect("prefill_tokens");
    for t in &rest[done..] {
        m.prefill_step(*t).expect("tail prefill step");
    }
    m.decode_step_logits(*last).expect("decode")
}

fn splice_prefill_logits(
    cfg: &Qwen3MoeConfig,
    hw: &q3w::HostWeights,
    ids: &[u32],
    splices: &[EmbedRowsSplice],
) -> (u32, Vec<f32>) {
    let mut m = q3w::Qwen3MoeWgpu::new(cfg.clone(), hw, 64).expect("build splice model");
    let tok = m
        .prefill_with_splices(ids, splices)
        .expect("prefill_with_splices");
    let logits = m.last_decode_logits().expect("read logits");
    (tok, logits)
}

#[test]
fn splice_prefill_is_bit_identical_to_token_prefill_and_spliced_rows_actually_land() {
    let _env = env_lock();
    if !have_gpu() {
        std::env::remove_var(PREFILL_M_ENV);
        std::env::remove_var(MROW_ENV);
        panic!("needs a wgpu adapter; a skipped identity proof reads as a passed one");
    }
    let cfg = tiny_config();
    let hidden = cfg.hidden_size;
    let hw = tiny_weights(&cfg, 0x51ee_dC0F_FEE1);

    std::env::set_var(PREFILL_M_ENV, "8");
    std::env::remove_var(MROW_ENV);

    let cases: &[(&str, usize, &[(usize, usize)])] = &[
        ("splice_inside_one_m_chunk", 20, &[(2, 3)]),
        ("splice_spanning_a_chunk_boundary", 20, &[(6, 4)]),
        ("splice_in_the_per_token_tail", 20, &[(17, 1)]),
        ("splice_at_position_0", 20, &[(0, 2)]),
        ("two_splices_across_chunks", 24, &[(3, 2), (11, 3)]),
    ];

    let engaged_mrow = q3w::Qwen3MoeWgpu::new(cfg.clone(), &hw, 64)
        .expect("probe build")
        .prefill_mrow_chunk_len()
        >= 2;
    assert!(
        engaged_mrow,
        "NV_WGPU_PREFILL_M=8 did not engage the M-row list; the boundary-spanning splice case \
         would not exercise the two-submission path this suite exists to prove"
    );

    for (name, len, runs) in cases {
        let ids = prompt_ids(&cfg, *len, 1);
        let splices = splices_from_prompt(&hw, hidden, &ids, runs);

        let (want_tok, want) = plain_prefill_logits(&cfg, &hw, &ids);
        let (got_tok, got) = splice_prefill_logits(&cfg, &hw, &ids, &splices);
        assert_eq!(want.len(), got.len(), "{name}: logit width changed");
        let diff = bit_diff(&want, &got);
        assert_eq!(
            diff, 0,
            "{name}: {diff} of {} logits differ bit-for-bit between the splice-prefill entry fed \
             the gathered embed rows verbatim and the plain token prefill. The gather copies raw \
             bf16 with no scaling, so a host write of those same rows into res/pfm.res MUST be \
             bit-identical; if this fires the splice injected the wrong rows, wrote them at the \
             wrong offset, or the two-submission M-row split let the gathers overwrite them. Do \
             NOT relax this to argmax or a tolerance.",
            want.len()
        );
        assert_eq!(
            want_tok, got_tok,
            "{name}: argmax token differs (plain {want_tok} vs splice {got_tok})"
        );

        let zero_splices: Vec<EmbedRowsSplice> = runs
            .iter()
            .map(|&(pos, n_rows)| EmbedRowsSplice {
                position: pos,
                rows_bf16: vec![0u16; n_rows * hidden],
            })
            .collect();
        let (_zt, zero_logits) = splice_prefill_logits(&cfg, &hw, &ids, &zero_splices);
        assert!(
            bit_diff(&want, &zero_logits) > 0,
            "{name}: feeding all-zero embedding rows at the spliced positions produced logits \
             bit-identical to the plain prompt, so the splice rows are being ignored and the \
             gathered token rows used instead -- the parity assertion above would pass an \
             implementation that never reads rows_bf16. This oracle-strength guard exists to kill \
             exactly that."
        );
    }

    let ids = prompt_ids(&cfg, 20, 2);
    let empty: Vec<EmbedRowsSplice> = Vec::new();
    let (plain_tok, plain) = plain_prefill_logits(&cfg, &hw, &ids);
    let (deleg_tok, deleg) = splice_prefill_logits(&cfg, &hw, &ids, &empty);
    assert_eq!(
        bit_diff(&plain, &deleg),
        0,
        "empty-splices prefill_with_splices is not bit-identical to prefill; the empty case must \
         delegate to prefill by construction"
    );
    assert_eq!(plain_tok, deleg_tok, "empty-splices delegation argmax differs");

    {
        let ids = prompt_ids(&cfg, 20, 3);
        let good = splices_from_prompt(&hw, hidden, &ids, &[(4, 2)]);
        let overlapping = vec![
            EmbedRowsSplice { position: 4, rows_bf16: good[0].rows_bf16.clone() },
            EmbedRowsSplice { position: 5, rows_bf16: embed_row(&hw, hidden, ids[5]) },
        ];
        let mut m = q3w::Qwen3MoeWgpu::new(cfg.clone(), &hw, 64).expect("build");
        assert!(
            m.prefill_with_splices(&ids, &overlapping).is_err(),
            "overlapping splices must return Err"
        );

        let covers_last = vec![EmbedRowsSplice {
            position: ids.len() - 1,
            rows_bf16: embed_row(&hw, hidden, ids[ids.len() - 1]),
        }];
        let mut m = q3w::Qwen3MoeWgpu::new(cfg.clone(), &hw, 64).expect("build");
        assert!(
            m.prefill_with_splices(&ids, &covers_last).is_err(),
            "a splice covering the final token must return Err (the last token is decoded via \
             decode_step, which replays the embed gathers)"
        );

        let ragged = vec![EmbedRowsSplice {
            position: 3,
            rows_bf16: vec![0u16; hidden + 7],
        }];
        let mut m = q3w::Qwen3MoeWgpu::new(cfg.clone(), &hw, 64).expect("build");
        assert!(
            m.prefill_with_splices(&ids, &ragged).is_err(),
            "rows_bf16 not a multiple of hidden_size must return Err"
        );
    }

    {
        std::env::set_var(MROW_ENV, "0");
        let ids = prompt_ids(&cfg, 20, 4);
        let mrow_off = q3w::Qwen3MoeWgpu::new(cfg.clone(), &hw, 64).expect("build mrow-off");
        assert_eq!(
            mrow_off.prefill_mrow_chunk_len(),
            0,
            "NV_Q3_WGPU_PF_MROW=0 must disable the M-row list so the per-token / pf_list fallback \
             routing is what carries the splice"
        );
        let splices = splices_from_prompt(&hw, hidden, &ids, &[(3, 2)]);
        let (want_tok, want) = plain_prefill_logits(&cfg, &hw, &ids);
        let (got_tok, got) = splice_prefill_logits(&cfg, &hw, &ids, &splices);
        assert_eq!(
            bit_diff(&want, &got),
            0,
            "with the M-row list off, the splice path (prefill_step_embed_row + pf_list chunks) is \
             not bit-identical to the plain token prefill"
        );
        assert_eq!(want_tok, got_tok, "mrow-off splice argmax differs");
        std::env::remove_var(MROW_ENV);
    }

    std::env::remove_var(PREFILL_M_ENV);
    std::env::remove_var(MROW_ENV);
}

fn qwen36_visual_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    std::env::var("NV_QWEN36_VISUAL_DIR")
        .or_else(|_| std::env::var("NV_QWEN36_DIR"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(&home).join(
                ".cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots/e850c696e6d75f965367e816c16bc7dacd955ffa",
            )
        })
}

fn assert_real_visual_weights(dir: &std::path::Path) {
    let visual = dir.join("model_visual.safetensors");
    let sz = std::fs::metadata(&visual)
        .unwrap_or_else(|e| panic!("stat {}: {e}", visual.display()))
        .len();
    assert!(
        sz > 500_000_000,
        "model_visual.safetensors at {} is {sz} bytes -- that is the LFS pointer stub, not the \
         ~893MB real tensor file. Fetch it (nix build .#speaches-models-hub-mm withMultimodal, or \
         `unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; HF_HUB_DISABLE_XET=1 hf download \
         RedHatAI/Qwen3.6-35B-A3B-NVFP4 model_visual.safetensors`), then symlink config.json and \
         the real model_visual.safetensors into a scratch dir and point NV_QWEN36_VISUAL_DIR at \
         it (the /nix/store copy is immutable).",
        visual.display()
    );
}

const RESCALE: f32 = 1.0 / 255.0;
const MEAN: f32 = 0.5;
const STD: f32 = 0.5;

fn norm_channel(v: f32) -> f32 {
    (v * RESCALE - MEAN) / STD
}

fn pixel_tensor(
    device: &candle_core::Device,
    h: usize,
    w: usize,
    rgb_at: impl Fn(usize, usize) -> [f32; 3],
) -> candle_core::Tensor {
    let mut data = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let rgb = rgb_at(y, x);
            for c in 0..3 {
                data[c * h * w + y * w + x] = norm_channel(rgb[c]);
            }
        }
    }
    candle_core::Tensor::from_vec(data, (1, 3, h, w), device).expect("pixel tensor")
}

fn rows_l2(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter()
        .map(|r| r.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect()
}

fn tensor_rows(t: &candle_core::Tensor) -> (usize, usize, Vec<Vec<f32>>) {
    let (n, d) = t.dims2().expect("[N, D] tower output");
    let flat: Vec<f32> = t
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("to_vec1");
    let rows = (0..n).map(|i| flat[i * d..(i + 1) * d].to_vec()).collect();
    (n, d, rows)
}

fn mean_row(rows: &[Vec<f32>]) -> Vec<f32> {
    let d = rows[0].len();
    let mut acc = vec![0f32; d];
    for r in rows {
        for (a, v) in acc.iter_mut().zip(r) {
            *a += v;
        }
    }
    for a in &mut acc {
        *a /= rows.len() as f32;
    }
    acc
}

fn l2_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[test]
#[ignore = "real Qwen3.6 vision tower; set NV_QWEN36_VISION_TEST=1 and provide the real \
            model_visual.safetensors via NV_QWEN36_VISUAL_DIR"]
fn real_weights_tower_rows_are_finite_and_content_bearing() {
    assert!(
        std::env::var("NV_QWEN36_VISION_TEST").is_ok(),
        "NV_QWEN36_VISION_TEST not set: this ignored test never skip-passes"
    );
    let dir = qwen36_visual_dir();
    assert_real_visual_weights(&dir);
    let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
    let tower =
        nv_omni::qwen3_vision::Qwen3VisionTower::try_load(&dir, &device).expect("load tower");

    let (h, w) = (256usize, 128usize);
    let half_red_half_blue = pixel_tensor(&device, h, w, |y, _| {
        if y < h / 2 {
            [255.0, 0.0, 0.0]
        } else {
            [0.0, 0.0, 255.0]
        }
    });
    let out = tower.forward(&half_red_half_blue).expect("tower forward");
    let (n, d, rows) = tensor_rows(&out);
    assert_eq!(
        (n, d),
        ((h / 16 * (w / 16)) / 4, 2048),
        "tower output shape must be [(H/16 * W/16)/4, 2048]"
    );
    for (i, r) in rows.iter().enumerate() {
        assert!(
            r.iter().all(|v| v.is_finite()),
            "row {i} has a non-finite value"
        );
    }
    let l2 = rows_l2(&rows);
    assert!(l2.iter().all(|&v| v > 0.0), "some row has zero L2 norm");

    let half = n / 2;
    let top_mean = mean_row(&rows[..half]);
    let bot_mean = mean_row(&rows[half..]);
    assert!(
        l2_diff(&top_mean, &bot_mean) > 1e-3,
        "mean of the red-region rows equals the mean of the blue-region rows; the tower is not \
         content-bearing (degenerate embeddings)"
    );

    let solid = |rgb: [f32; 3]| {
        let px = pixel_tensor(&device, 128, 128, |_, _| rgb);
        let out = tower.forward(&px).expect("solid forward");
        tensor_rows(&out).2
    };
    let red = solid([255.0, 0.0, 0.0]);
    let blue = solid([0.0, 0.0, 255.0]);
    let red_mean = mean_row(&red);
    let blue_mean = mean_row(&blue);
    assert!(
        l2_diff(&red_mean, &blue_mean) > 1e-3,
        "a solid-red image and a solid-blue image produced the same tower rows"
    );
    eprintln!(
        "[qwen36-tower] rows=[{n},{d}] top/bottom L2 diff {:.4}, red/blue mean L2 diff {:.4}",
        l2_diff(&top_mean, &bot_mean),
        l2_diff(&red_mean, &blue_mean)
    );
}

const VISION_START: u32 = 248053;
const VISION_END: u32 = 248054;
const IMAGE_PAD: u32 = 248056;

fn tensor_to_bf16_bits(t: &candle_core::Tensor) -> Vec<u16> {
    let flat: Vec<f32> = t
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("to_vec1");
    flat.iter()
        .map(|&v| half::bf16::from_f32(v).to_bits())
        .collect()
}

fn encode_ids(tok: &tokenizers::Tokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, false)
        .expect("encode")
        .get_ids()
        .to_vec()
}

#[test]
#[ignore = "full e2e on the real 22.4GB Qwen3.6 MoE trunk + real vision tower; set \
            NV_QWEN36_MM_TEST=1 and provide the real model_visual.safetensors"]
fn real_weights_mm_greedy_decode_names_the_color_and_text_only_entry_is_unchanged() {
    let _env = env_lock();
    assert!(
        std::env::var("NV_QWEN36_MM_TEST").is_ok(),
        "NV_QWEN36_MM_TEST not set: this ignored test never skip-passes"
    );
    if !have_gpu() {
        panic!("needs a wgpu adapter");
    }
    let dir = qwen36_visual_dir();
    assert_real_visual_weights(&dir);
    let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
    let tower =
        nv_omni::qwen3_vision::Qwen3VisionTower::try_load(&dir, &device).expect("load tower");

    let px = pixel_tensor(&device, 128, 128, |_, _| [255.0, 0.0, 0.0]);
    let emb = tower.forward(&px).expect("tower forward");
    let (n_rows, out_h) = emb.dims2().expect("[N,2048]");
    assert_eq!((n_rows, out_h), (16, 2048), "128x128 red image -> [16, 2048]");
    let rows_bf16 = tensor_to_bf16_bits(&emb);

    let mut tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .expect("tokenizer.json beside the checkpoint");
    tok.with_truncation(None).expect("clear truncation");

    let prefix = encode_ids(&tok, "<|im_start|>user\n");
    let suffix = encode_ids(
        &tok,
        "What color is this image? Answer with one word.<|im_end|>\n<|im_start|>assistant\n",
    );
    let mut ids: Vec<u32> = Vec::new();
    ids.extend_from_slice(&prefix);
    ids.push(VISION_START);
    let run_start = ids.len();
    ids.extend_from_slice(&vec![IMAGE_PAD; n_rows]);
    ids.push(VISION_END);
    ids.extend_from_slice(&suffix);

    let pad_runs = ids.iter().filter(|&&t| t == IMAGE_PAD).count();
    assert_eq!(pad_runs, n_rows, "prompt must reserve exactly {n_rows} image_pad slots");
    let splices = tower
        .build_splices(&ids, IMAGE_PAD, std::slice::from_ref(&emb))
        .expect("build_splices");
    assert_eq!(splices.len(), 1, "one image run");
    assert_eq!(splices[0].position, run_start, "splice at the image-pad run start");

    let cfg = Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open trunk safetensors");
    let mut engine =
        q3w::Qwen3MoeWgpu::from_loader(cfg.clone(), &loader, ids.len() + 64).expect("build engine");
    drop(loader);

    let mut t = engine
        .prefill_with_splices(
            &ids,
            &[EmbedRowsSplice { position: run_start, rows_bf16 }],
        )
        .expect("mm prefill");
    let mut out = vec![t];
    for _ in 0..23 {
        t = engine.decode_step(t).expect("greedy decode");
        out.push(t);
    }
    let text = tok.decode(&out, true).expect("decode transcript");
    eprintln!("[qwen36-mm] transcript: {text:?}");
    assert!(
        text.to_lowercase().contains("red"),
        "greedy decode did not name the color red; got {text:?}"
    );

    let mut ids_text: Vec<u32> = Vec::new();
    ids_text.extend_from_slice(&prefix);
    ids_text.extend_from_slice(&suffix);

    engine.reset().expect("reset");
    let ta = engine
        .prefill_with_splices(&ids_text, &[])
        .expect("text prefill via splice entry");
    let la = engine.last_decode_logits().expect("logits a");
    let mut seq_a = vec![ta];
    let mut cur = ta;
    for _ in 0..23 {
        cur = engine.decode_step(cur).expect("decode a");
        seq_a.push(cur);
    }

    engine.reset().expect("reset");
    let tb = engine.prefill(&ids_text).expect("text prefill via prefill");
    let lb = engine.last_decode_logits().expect("logits b");
    let mut seq_b = vec![tb];
    let mut cur = tb;
    for _ in 0..23 {
        cur = engine.decode_step(cur).expect("decode b");
        seq_b.push(cur);
    }

    assert_eq!(
        bit_diff(&la, &lb),
        0,
        "text-only prefill_with_splices(&[]) and prefill diverge on first-token logits; the empty \
         splice path must be bit-identical to the pre-change text path"
    );
    assert_eq!(seq_a, seq_b, "text-only 24-token greedy sequences differ between the two entries");
    eprintln!("[qwen36-mm] text-only invariance holds over 24 greedy tokens");
}
