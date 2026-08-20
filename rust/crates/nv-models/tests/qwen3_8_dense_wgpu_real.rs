#![cfg(feature = "wgpu")]

mod hub_snapshot;
mod ppl_common;

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
use std::time::Instant;
use tokenizers::Tokenizer;

const REPO: &str = "unsloth/Qwen3.8-27B-NVFP4";

const WGPU_GATE_ENV: &str = "NV_QWEN38_WGPU_TEST";

fn require_qwen38_wgpu_gate() {
    if std::env::var(WGPU_GATE_ENV).as_deref() != Ok("1") {
        panic!(
            "set {WGPU_GATE_ENV}=1 to run the real-weights Qwen3.8-27B-NVFP4 wgpu suite \
             (it must never silently skip; this arm boots the 22.57 GB checkpoint)"
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

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .map(|v| {
            v.parse::<usize>()
                .unwrap_or_else(|_| panic!("{key}={v} is not a usize"))
        })
        .unwrap_or(default)
}

fn parse_config_with_optional_layer_truncation(dir: &PathBuf) -> (Qwen3_5DenseConfig, Option<usize>) {
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let mut cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw)
        .expect("real Qwen3.8-27B-NVFP4 config.json must parse as a Qwen3_5 dense config");
    let full_layers = cfg.num_hidden_layers;
    let truncated = std::env::var("NV_QWEN38_LAYERS").ok().map(|v| {
        let n: usize = v.parse().unwrap_or_else(|_| panic!("NV_QWEN38_LAYERS={v} is not a usize"));
        assert!(n >= 1 && n <= full_layers, "NV_QWEN38_LAYERS={n} out of 1..={full_layers}");
        cfg.layer_types.truncate(n);
        cfg.num_hidden_layers = n;
        n
    });
    (cfg, truncated.filter(|n| *n != full_layers))
}

fn assert_text_path_is_vision_free(loader: &WeightLoader) {
    let names = loader.names();
    let vision: Vec<&String> = names
        .iter()
        .filter(|n| n.contains("visual") || n.contains("vision"))
        .collect();
    assert!(
        !vision.is_empty(),
        "checkpoint exposes no vision-tower tensors: this is meant to be the multimodal \
         Qwen3.8-27B checkpoint whose text-only decode path we are proving self-sufficient"
    );
    assert!(
        loader.has("model.language_model.embed_tokens.weight"),
        "text embed under model.language_model.* is absent; the wgpu decoder resolves \
         model.language_model.*/model.layers.*/lm_head.weight/model.norm.weight candidates only, \
         never model.visual.* -- a successful boot below therefore requests zero vision tensors"
    );
    for v in &vision {
        assert!(
            !v.starts_with("model.language_model.") && !v.starts_with("model.layers."),
            "vision tensor {v} sits under a text prefix; the vision-free claim would be unsound"
        );
    }
}

fn qwen_chat_prompt_ids(tok: &Tokenizer) -> Vec<u32> {
    let q = std::env::var("NV_QWEN38_PROMPT")
        .unwrap_or_else(|_| "Explain, in a few sentences, why the sky appears blue.".into());
    let text =
        format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    tok.encode(text.as_str(), false)
        .expect("encode chat prompt")
        .get_ids()
        .to_vec()
}

fn boot_model(dir: &PathBuf, cfg: Qwen3_5DenseConfig, max_seq: usize) -> Qwen3_5DenseWgpu {
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let loader = WeightLoader::open_dir(dir, &candle_core::Device::Cpu).expect("open weights");
    assert_text_path_is_vision_free(&loader);
    let m = Qwen3_5DenseWgpu::from_loader(cfg, &loader, max_seq)
        .expect("Qwen3_5DenseWgpu::from_loader on the real Qwen3.8-27B-NVFP4 checkpoint");
    drop(loader);
    m
}

#[test]
#[ignore = "boots the ~22.6 GB unsloth/Qwen3.8-27B-NVFP4 checkpoint on the wgpu backend; set \
            NV_QWEN38_WGPU_TEST=1 (NV_QWEN38_DIR/LAYERS/MAX_SEQ/PROMPT/NEW optional)"]
fn qwen3_8_27b_nvfp4_wgpu_boots_and_greedily_decodes_at_least_64_tokens() {
    require_qwen38_wgpu_gate();
    let dir = qwen38_snapshot_dir();
    let (cfg, truncated) = parse_config_with_optional_layer_truncation(&dir);
    let vocab = cfg.vocab_size;
    let n_layers = cfg.num_hidden_layers;

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let prompt = qwen_chat_prompt_ids(&tok);
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tok.token_to_id(s))
        .collect();

    let new_tokens = env_usize("NV_QWEN38_NEW", 96);
    assert!(
        new_tokens >= 64,
        "NV_QWEN38_NEW={new_tokens} < 64: the decode-sanity gate requires at least 64 generated tokens"
    );
    let max_seq = env_usize("NV_QWEN38_MAX_SEQ", prompt.len() + new_tokens + 16);
    assert!(
        max_seq >= prompt.len() + new_tokens,
        "NV_QWEN38_MAX_SEQ={max_seq} too small for prompt {} + new {new_tokens}",
        prompt.len()
    );

    let mut m = boot_model(&dir, cfg, max_seq);

    let first = m.prefill(&prompt).expect("prefill the chat prompt");
    let mut generated: Vec<u32> = vec![first];
    let mut hit_stop_at: Option<usize> = None;
    let t0 = Instant::now();
    let mut tok_prev = first;
    for i in 1..new_tokens {
        if stop.contains(&tok_prev) {
            hit_stop_at = Some(i);
            break;
        }
        let next = m.decode_step(tok_prev).expect("greedy decode step");
        generated.push(next);
        tok_prev = next;
    }
    let elapsed = t0.elapsed();
    let stepped = generated.len();
    let ms_per_token = elapsed.as_secs_f64() * 1e3 / stepped.max(1) as f64;

    for (i, t) in generated.iter().enumerate() {
        assert!(
            (*t as usize) < vocab,
            "generated token {t} at step {i} is outside the vocab width {vocab}"
        );
    }
    if nv_kernels::wgpu_backend::dispatch::profile::enabled() {
        let rows = nv_kernels::wgpu_backend::dispatch::profile::report();
        eprintln!("== qwen38 decode per-dispatch GPU profile ({stepped} steps) ==");
        for (lbl, count, ns) in &rows {
            eprintln!(
                "{lbl:<52} {count:>8}  {:>10.3} ms  {:>8.2} us/step",
                ns / 1e6,
                ns / 1e3 / stepped.max(1) as f64
            );
        }
        let gpu_ms = nv_kernels::wgpu_backend::dispatch::profile::total_ns() / 1e6;
        eprintln!(
            "GPU-attributed {gpu_ms:.1} ms = {:.2} ms/step vs wall {ms_per_token:.2} ms/step",
            gpu_ms / stepped.max(1) as f64
        );
    }
    let text = tok.decode(&generated, true).unwrap_or_default();
    let label = checkpoint_label_from_snapshot_dir(&dir);
    let log = std::env::var("NV_QWEN38_LOG").unwrap_or_else(|_| "stdout".into());

    println!(
        "QWEN38-WGPU-DECODE checkpoint={label} backend=wgpu batch=1 layers={n_layers}{} \
         prompt_tokens={} generated_tokens={stepped} ms_per_token={ms_per_token:.2} \
         stop_at={:?} log={log}",
        if truncated.is_some() { "(TRUNCATED)" } else { "" },
        prompt.len(),
        hit_stop_at
    );
    println!("QWEN38-WGPU-DECODE-TEXT {label} :: {text:?}");

    if truncated.is_some() {
        eprintln!(
            "NV_QWEN38_LAYERS truncated the trunk to {n_layers} layers: this is a boot/timing \
             iteration only; the >=64-token and non-degenerate coherence assertions are skipped"
        );
        assert!(
            stepped >= 2,
            "even a truncated trunk must emit more than the prefill token"
        );
        return;
    }

    assert!(
        stepped >= 64,
        "full-depth greedy decode produced only {stepped} tokens before stop {hit_stop_at:?}; \
         the sanity gate demands at least 64"
    );
    let distinct: std::collections::BTreeSet<u32> = generated.iter().copied().collect();
    assert!(
        distinct.len() >= 8,
        "full-depth greedy decode is degenerate: only {} distinct tokens across {stepped} steps \
         ({text:?})",
        distinct.len()
    );
}

#[test]
#[ignore = "loads the 27B on wgpu; set NV_QWEN38_WGPU_TEST=1 -- chunked-prefill tok/s at NV_CTX_TOKENS prompt length (plain integer, default 2048), chat prompt padded with its own last token"]
fn qwen38_wgpu_chunked_prefill_rate_vs_prompt_len() {
    require_qwen38_wgpu_gate();
    let dir = qwen38_snapshot_dir();
    let (cfg, truncated) = parse_config_with_optional_layer_truncation(&dir);
    let n_layers = cfg.num_hidden_layers;
    let vocab = cfg.vocab_size;
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let mut prompt = qwen_chat_prompt_ids(&tok);
    let n = env_usize("NV_CTX_TOKENS", 2048);
    assert!(
        n >= prompt.len(),
        "NV_CTX_TOKENS={n} shorter than the {}-token chat prompt",
        prompt.len()
    );
    let filler = *prompt.last().expect("chat prompt is never empty");
    prompt.resize(n, filler);
    let mut m = boot_model(&dir, cfg, n + 8);
    let t0 = Instant::now();
    let first = m.prefill(&prompt).expect("prefill the padded prompt");
    let s = t0.elapsed().as_secs_f64();
    assert!(
        (first as usize) < vocab,
        "prefill emitted token {first} outside the vocab width {vocab}"
    );
    if nv_kernels::wgpu_backend::dispatch::profile::enabled() {
        let rows = nv_kernels::wgpu_backend::dispatch::profile::report();
        eprintln!("== qwen38 prefill per-dispatch GPU profile ({n} tokens) ==");
        for (lbl, count, ns) in &rows {
            eprintln!("{lbl:<52} {count:>8}  {:>10.3} ms", ns / 1e6);
        }
        let gpu_ms = nv_kernels::wgpu_backend::dispatch::profile::total_ns() / 1e6;
        eprintln!(
            "GPU-attributed {gpu_ms:.1} ms vs wall {:.1} ms",
            s * 1e3
        );
    }
    let label = checkpoint_label_from_snapshot_dir(&dir);
    println!(
        "QWEN38-WGPU-PREFILL checkpoint={label} backend=wgpu layers={n_layers}{} \
         prompt_tokens={n} prefill_s={s:.2} tok_s={:.1} chunk_m={} passes_per_chunk={}",
        if truncated.is_some() { "(TRUNCATED)" } else { "" },
        n as f64 / s,
        m.prefill_chunk_len(),
        m.prefill_pass_count()
    );
}

mod pf_coop_tiny {
    use nv_models::qwen3_5_dense_wgpu as q3d;
    use nv_models::qwen3_5_moe::LayerType;
    use nv_models::qwen3_5_moe_wgpu::{quantize_nvfp4_host, HostBf16Lin, HostDeltaNet};
    use q3d::Qwen3_5DenseConfig;

    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32 - 1.0
        }
        fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
            (0..n)
                .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
                .collect()
        }
        fn f32_vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n)
                .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_f32())
                .collect()
        }
    }

    fn norm_vec(r: &mut Lcg, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(1.0 + 0.1 * r.next_f32()).to_bits())
            .collect()
    }

    fn aligned_tiny_config() -> Qwen3_5DenseConfig {
        Qwen3_5DenseConfig {
            hidden_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 32,
            intermediate_size: 192,
            vocab_size: 64,
            max_position_embeddings: 96,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            partial_rotary_factor: 0.25,
            bos_token_id: None,
            eos_token_id: 1,
            layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
            linear_num_key_heads: 2,
            linear_num_value_heads: 8,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            attn_output_gate: true,
            tie_word_embeddings: false,
        }
    }

    fn fp8_pair(r: &mut Lcg, n: usize, k: usize, scale: f32) -> (q3d::HostFp8Lin, HostBf16Lin) {
        let base: Vec<half::bf16> = (0..n * k)
            .map(|_| half::bf16::from_f32(r.next_f32() * scale))
            .collect();
        let (bytes, scales) = nv_quant::fp8::quantize_e4m3_per_row(&base, n, k).expect("fp8 quant");
        let deq = nv_quant::fp8::dequantize_e4m3_per_row(&bytes, n, k, &scales).expect("fp8 deq");
        let packed: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (
            q3d::HostFp8Lin {
                packed,
                scales,
                n,
                k,
            },
            HostBf16Lin {
                w: deq
                    .iter()
                    .map(|v| half::bf16::from_f32(*v).to_bits())
                    .collect(),
                n,
                k,
            },
        )
    }

    fn fp8_dense(r: &mut Lcg, n: usize, k: usize, scale: f32) -> q3d::HostDenseLin {
        let (fp8, bf16) = fp8_pair(r, n, k, scale);
        q3d::HostDenseLin::Fp8 { fp8, bf16 }
    }

    fn nvfp4(r: &mut Lcg, n: usize, k: usize, scale: f32) -> q3d::HostDenseLin {
        let w = r.bf16_vec(n * k, scale);
        q3d::HostDenseLin::Nvfp4(quantize_nvfp4_host(&w, n, k))
    }

    fn tiny_fp8_nvfp4_weights(cfg: &Qwen3_5DenseConfig, seed: u64) -> q3d::HostDenseWeights {
        let mut r = Lcg(seed | 1);
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
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
        for lt in &cfg.layer_types {
            let mut delta_fp8 = q3d::DeltaFp8::default();
            let mixer = match lt {
                LayerType::LinearAttention => {
                    let (f_qkv, t_qkv) = fp8_pair(&mut r, conv_dim, hidden, 0.12);
                    let (f_z, t_z) = fp8_pair(&mut r, value_dim, hidden, 0.12);
                    let (f_out, t_out) = fp8_pair(&mut r, hidden, value_dim, 0.12);
                    delta_fp8 = q3d::DeltaFp8 {
                        qkv: Some(f_qkv),
                        z: Some(f_z),
                        out: Some(f_out),
                    };
                    q3d::HostDenseMixer::Delta(Box::new(HostDeltaNet {
                        in_proj_qkv: t_qkv,
                        in_proj_z: t_z,
                        in_proj_ab: HostBf16Lin {
                            w: r.bf16_vec(2 * n_v * hidden, 0.12),
                            n: 2 * n_v,
                            k: hidden,
                        },
                        conv1d: r.f32_vec(conv_dim * ks, 0.4),
                        a_log: r.f32_vec(n_v, 0.5),
                        dt_bias: r.f32_vec(n_v, 0.5),
                        norm_w: norm_vec(&mut r, d_v),
                        out_proj: t_out,
                    }))
                }
                LayerType::FullAttention => {
                    let q_out = cfg.num_attention_heads * hd * 2;
                    let kv_out = cfg.num_key_value_heads * hd;
                    q3d::HostDenseMixer::Attn(Box::new(q3d::HostDenseAttention {
                        q: fp8_dense(&mut r, q_out, hidden, 0.12),
                        k: fp8_dense(&mut r, kv_out, hidden, 0.12),
                        v: fp8_dense(&mut r, kv_out, hidden, 0.12),
                        o: fp8_dense(&mut r, hidden, cfg.num_attention_heads * hd, 0.12),
                        q_norm: norm_vec(&mut r, hd),
                        k_norm: norm_vec(&mut r, hd),
                    }))
                }
            };
            layers.push(q3d::HostDenseLayer {
                input_ln: norm_vec(&mut r, hidden),
                post_attn_ln: norm_vec(&mut r, hidden),
                mixer,
                mlp: q3d::HostDenseMlp {
                    gate: nvfp4(&mut r, inter, hidden, 0.15),
                    up: nvfp4(&mut r, inter, hidden, 0.15),
                    down: nvfp4(&mut r, hidden, inter, 0.15),
                },
                delta_fp8,
            });
        }
        q3d::HostDenseWeights {
            embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
            final_norm: norm_vec(&mut r, hidden),
            lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
            layers,
        }
    }

    static COOP_ENV_ARMS_SERIALIZE_BECAUSE_PF_COOP_READS_PROCESS_ENV: std::sync::Mutex<()> =
        std::sync::Mutex::new(());

    fn chunked_prefill_logits(gpu: &mut q3d::Qwen3_5DenseWgpu, tokens: &[u32]) -> (u32, Vec<f32>) {
        gpu.reset().expect("reset");
        let (last, rest) = tokens.split_last().expect("prompt non-empty");
        let done = gpu.prefill_tokens(rest).expect("prefill_tokens");
        for t in &rest[done..] {
            gpu.prefill_step(*t).expect("tail prefill step");
        }
        gpu.decode_step_logits(*last).expect("last step")
    }

    fn spliced_chunked_prefill_logits(
        gpu: &mut q3d::Qwen3_5DenseWgpu,
        tokens: &[u32],
        splices: &[q3d::ImageRowSplice],
    ) -> (u32, Vec<f32>) {
        gpu.reset().expect("reset");
        let (last, rest) = tokens.split_last().expect("prompt non-empty");
        let done = gpu
            .prefill_tokens_with_image_rows(rest, splices)
            .expect("spliced prefill");
        assert_eq!(done, rest.len(), "spliced prefill must consume every prompt token");
        gpu.decode_step_logits(*last).expect("last step")
    }

    pub const SPLICE_TWIN_OBSERVES_THE_CONV_WINDOW_BECAUSE_THIS_TINY_NET_HIDES_ATTENTION: &str =
        "the tiny fp8/nvfp4 net cannot surface a splice through its full-attention layer: the \
         attn output gate saturates near zero and the bf16 residual add rounds the remainder \
         away, while the DeltaNet scan decays a perturbation to nothing within ~7 rows. The \
         only decode-visible channel is the causal-conv tail (the last linear_conv_kernel_dim \
         rows of the final chunk), so the e2e half of this twin splices exactly there, and the \
         buffer-level half proves the splice pass bit-exactly for every masked row instead.";

    #[test]
    fn pf_coop_arm_matches_the_legacy_arm_through_the_image_row_splice_path() {
        let _env = COOP_ENV_ARMS_SERIALIZE_BECAUSE_PF_COOP_READS_PROCESS_ENV
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if nv_kernels::wgpu_backend::WgpuContext::shared().is_err() {
            if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() == Ok("1") {
                eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): no wgpu adapter");
                return;
            }
            panic!("no wgpu adapter; this gate must never silently skip");
        }
        let cfg = aligned_tiny_config();
        let hw = tiny_fp8_nvfp4_weights(&cfg, 0xc00b_0002);
        let hidden = cfg.hidden_size;
        let tokens: Vec<u32> = (0..33u32).map(|i| (i * 5 + 2) % 64).collect();

        let run_arm = |label: &str| -> ((u32, Vec<f32>), (u32, Vec<f32>)) {
            let mut gpu =
                q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 96).expect("build arm model");
            let m = gpu.prefill_chunk_len();
            assert!(m >= 16, "{label}: chunked prefill graph missing (m={m})");

            let chunk: Vec<u32> = (0..m as u32).map(|i| (i * 5 + 2) % 64).collect();
            let mut r = Lcg(0x51ce_0001);
            let rows: Vec<(usize, Vec<u16>)> = (0..m)
                .step_by(3)
                .map(|rel| (rel, r.bf16_vec(hidden, 0.4)))
                .collect();
            let got = gpu
                .debug_pf_embed_splice_rows_for_test(&chunk, &rows)
                .expect("embed+splice readback");
            for rel in 0..m {
                let row = &got[rel * hidden..(rel + 1) * hidden];
                match rows.iter().find(|(rr, _)| *rr == rel) {
                    Some((_, want)) => assert_eq!(
                        row,
                        want.as_slice(),
                        "{label}: spliced row {rel} is not the host image row bit-for-bit"
                    ),
                    None => {
                        let tok = chunk[rel] as usize;
                        assert_eq!(
                            row,
                            &hw.embed[tok * hidden..(tok + 1) * hidden],
                            "{label}: unspliced row {rel} no longer matches the embed of token \
                             {tok}: the splice pass leaked past its mask"
                        );
                    }
                }
            }

            gpu.reset().expect("reset after readback");
            let mut r2 = Lcg(0x51ce_0002);
            let splices = vec![q3d::ImageRowSplice {
                position: 28,
                rows_bf16: r2.bf16_vec(4 * hidden, 0.4),
            }];
            let spliced = spliced_chunked_prefill_logits(&mut gpu, &tokens, &splices);
            gpu.reset().expect("reset between e2e arms");
            let plain = chunked_prefill_logits(&mut gpu, &tokens);
            (spliced, plain)
        };

        std::env::set_var("NV_Q3D_PF_COOP", "0");
        std::env::remove_var("NV_WGPU_PREFILL_M");
        let ((arg_base, logits_base), (arg_plain, logits_plain)) = run_arm("base m=16");
        assert!(
            logits_plain
                .iter()
                .zip(logits_base.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "{}",
            SPLICE_TWIN_OBSERVES_THE_CONV_WINDOW_BECAUSE_THIS_TINY_NET_HIDES_ATTENTION
        );

        std::env::set_var("NV_Q3D_PF_COOP", "1");
        std::env::set_var("NV_WGPU_PREFILL_M", "16");
        let ((arg_coop, logits_coop), _) = run_arm("coop m=16");

        std::env::set_var("NV_WGPU_PREFILL_M", "64");
        let ((arg_c64, logits_c64), _) = run_arm("coop m=64");
        std::env::remove_var("NV_WGPU_PREFILL_M");
        std::env::remove_var("NV_Q3D_PF_COOP");

        let bitdiff = logits_base
            .iter()
            .zip(logits_coop.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let scale = logits_base
            .iter()
            .fold(0f32, |a, v| a.max(v.abs()))
            .max(1e-6);
        let rel = |xs: &[f32]| {
            xs.iter()
                .zip(logits_base.iter())
                .fold(0f32, |a, (c, b)| a.max((c - b).abs() / scale))
        };
        let rel_coop = rel(&logits_coop);
        let rel_c64 = rel(&logits_c64);
        eprintln!(
            "[pf-coop-splice-tiny] base vs coop m=16: {bitdiff}/{} lanes differ, rel \
             {rel_coop:.5}; coop m=64 rel {rel_c64:.5}; argmax plain/base/coop/c64 \
             {arg_plain}/{arg_base}/{arg_coop}/{arg_c64}",
            logits_base.len()
        );
        assert!(
            bitdiff > 0,
            "the coop arm produced bit-identical logits to the legacy spliced arm: the \
             w8a16/w4a16 route did not actually run under the splice graph"
        );
        assert_eq!(arg_base, arg_coop, "coop arm flipped the spliced greedy argmax");
        assert_eq!(arg_base, arg_c64, "coop m=64 arm flipped the spliced greedy argmax");
        assert!(
            rel_coop < 0.05 && rel_c64 < 0.05,
            "coop spliced prefill drifted past 5% of the legacy arm (m16 {rel_coop}, m64 {rel_c64})"
        );
    }

    #[test]
    fn pf_coop_arm_is_a_distinct_numeric_path_within_tolerance_and_lifts_m_past_16() {
        let _env = COOP_ENV_ARMS_SERIALIZE_BECAUSE_PF_COOP_READS_PROCESS_ENV
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if nv_kernels::wgpu_backend::WgpuContext::shared().is_err() {
            if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() == Ok("1") {
                eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): no wgpu adapter");
                return;
            }
            panic!("no wgpu adapter; this gate must never silently skip");
        }
        let cfg = aligned_tiny_config();
        let hw = tiny_fp8_nvfp4_weights(&cfg, 0xc00b_0001);
        let tokens: Vec<u32> = (0..33u32).map(|i| (i * 7 + 3) % 64).collect();

        std::env::set_var("NV_Q3D_PF_COOP", "0");
        std::env::remove_var("NV_WGPU_PREFILL_M");
        let mut base = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 96).expect("build base arm");
        assert_eq!(base.prefill_chunk_len(), 16);
        let (arg_base, logits_base) = chunked_prefill_logits(&mut base, &tokens);
        drop(base);

        std::env::set_var("NV_Q3D_PF_COOP", "1");
        std::env::set_var("NV_WGPU_PREFILL_M", "16");
        let mut coop = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 96).expect("build coop arm");
        assert_eq!(coop.prefill_chunk_len(), 16);
        let (arg_coop, logits_coop) = chunked_prefill_logits(&mut coop, &tokens);
        drop(coop);

        std::env::set_var("NV_WGPU_PREFILL_M", "64");
        let mut coop64 =
            q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 96).expect("build coop m=64 arm");
        assert_eq!(
            coop64.prefill_chunk_len(),
            64,
            "NV_Q3D_PF_COOP=1 must lift the NV_WGPU_PREFILL_M clamp past 16"
        );
        let (arg_c64, logits_c64) = chunked_prefill_logits(&mut coop64, &tokens);
        drop(coop64);
        std::env::remove_var("NV_WGPU_PREFILL_M");
        std::env::remove_var("NV_Q3D_PF_COOP");

        let bitdiff = logits_base
            .iter()
            .zip(logits_coop.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let scale = logits_base
            .iter()
            .fold(0f32, |a, v| a.max(v.abs()))
            .max(1e-6);
        let rel = |xs: &[f32]| {
            xs.iter()
                .zip(logits_base.iter())
                .fold(0f32, |a, (c, b)| a.max((c - b).abs() / scale))
        };
        let rel_coop = rel(&logits_coop);
        let rel_c64 = rel(&logits_c64);
        eprintln!(
            "[pf-coop-tiny] arm A vs coop m=16: {bitdiff}/{} lanes differ, rel {rel_coop:.5}; \
             coop m=64 rel {rel_c64:.5}; argmax {arg_base}/{arg_coop}/{arg_c64}",
            logits_base.len()
        );
        assert!(
            bitdiff > 0,
            "the coop arm produced bit-identical logits to the legacy m-row arm: the w8a16/w4a16 \
             route did not actually run, or its output was discarded"
        );
        assert_eq!(arg_base, arg_coop, "coop arm flipped the greedy argmax");
        assert_eq!(arg_base, arg_c64, "coop m=64 arm flipped the greedy argmax");
        assert!(
            rel_coop < 0.05 && rel_c64 < 0.05,
            "coop prefill drifted past 5% of the legacy arm (m16 {rel_coop}, m64 {rel_c64})"
        );
    }
}

fn score_suffix_after_chunked_prefill_of_prefix(
    m: &mut Qwen3_5DenseWgpu,
    ids: &[u32],
    prefix: usize,
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    assert_eq!(
        m.current_pos(),
        0,
        "wgpu model must start each arm at position 0; call reset() between arms"
    );
    let chunk_m = m.prefill_chunk_len();
    assert!(
        chunk_m >= 2,
        "this gate scores the CHUNKED prefill graph's states; NV_WGPU_PREFILL_M must be >= 2"
    );
    let done = m
        .prefill_tokens(&ids[..prefix])
        .expect("chunked prefill of the prefix");
    assert_eq!(
        done, prefix,
        "prefill_tokens must consume the whole {prefix}-token prefix through the pf graph"
    );
    assert_eq!(m.current_pos(), prefix, "prefill advanced to an unexpected position");
    let vocab = m.config().vocab_size;
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    for p in prefix..ids.len() - 1 {
        let (_argmax, row) = m
            .decode_step_logits(ids[p])
            .unwrap_or_else(|err| panic!("wgpu decode_step_logits at position {p}: {err:#}"));
        assert_eq!(row.len(), vocab);
        if (ids[p + 1] as usize) >= vocab {
            continue;
        }
        acc.add_position_full_vocab_row(&row, ids[p + 1]);
    }
    acc
}

#[test]
#[ignore = "loads the ~22.6 GB checkpoint, chunk-prefills a 1024-token prefix through the pf graph \
            (the path NV_Q3D_PF_COOP reroutes) and teacher-forces the remaining 1023 positions; set \
            NV_QWEN38_WGPU_TEST=1, NV_PPL_TEST=1, NV_PPL_CORPUS; arm with NV_Q3D_PF_COOP/NV_WGPU_PREFILL_M"]
fn qwen38_wgpu_chunked_prefill_then_teacher_forced_ppl_beats_shuffled_control() {
    require_qwen38_wgpu_gate();
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    assert!(
        std::env::var("NV_QWEN38_LAYERS").is_err(),
        "NV_QWEN38_LAYERS truncates the trunk; unset it for the prefill ppl gate"
    );

    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = qwen38_snapshot_dir();
    let (cfg, _) = parse_config_with_optional_layer_truncation(&dir);

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let n = PPL_CORPUS_SLICE_TOKENS_N_2048_CHOSEN_SO_31B_LOAD_PLUS_FOUR_EAGER_512_BLOCKS_FITS_2_TO_3_MIN;
    let prefix = n / 2;
    let slice = first_n_corpus_tokens_after_tokenization(&tok, &corpus, n);

    let mut m = boot_model(&dir, cfg, n + 8);
    let label = checkpoint_label_from_snapshot_dir(&dir);
    let coop = std::env::var("NV_Q3D_PF_COOP").unwrap_or_else(|_| "unset".into());
    let pfm = m.prefill_chunk_len();
    eprintln!(
        "PPL basis: {label} corpus_slice_tokens={n} prefix_via_chunked_prefill={prefix} \
         backend=wgpu batch=1 path=prefill_tokens+decode_step_logits NV_Q3D_PF_COOP={coop} \
         chunk_m={pfm} note=scores-only-the-suffix-positions-so-not-comparable-to-the-full-decode-ppl"
    );

    let real = score_suffix_after_chunked_prefill_of_prefix(&mut m, &slice, prefix);

    let mut shuffled = slice.clone();
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
        &mut shuffled,
    );
    m.reset().expect("reset between arms");
    let shuffled = score_suffix_after_chunked_prefill_of_prefix(&mut m, &shuffled, prefix);

    assert_eq!(
        real.scored_positions, shuffled.scored_positions,
        "real and shuffled arms scored different position counts"
    );
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        &format!("qwen3_8-dense-wgpu-pfchunk-coop{coop}-m{pfm}"),
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}

fn score_slice_via_wgpu_decode_step_logits(
    m: &mut Qwen3_5DenseWgpu,
    ids: &[u32],
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    assert_eq!(
        m.current_pos(),
        0,
        "wgpu model must start each arm at position 0; call reset() between arms"
    );
    let vocab = m.config().vocab_size;
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    let mut skipped_beyond_logit_width = 0usize;
    for p in 0..ids.len() - 1 {
        let (_argmax, row) = m
            .decode_step_logits(ids[p])
            .unwrap_or_else(|err| panic!("wgpu decode_step_logits at position {p}: {err:#}"));
        assert_eq!(
            row.len(),
            vocab,
            "decode_step_logits must return the full vocab row, not a top-k slice"
        );
        if (ids[p + 1] as usize) >= vocab {
            skipped_beyond_logit_width += 1;
            continue;
        }
        acc.add_position_full_vocab_row(&row, ids[p + 1]);
    }
    if skipped_beyond_logit_width > 0 {
        eprintln!(
            "PPL-SKIPPED {skipped_beyond_logit_width} positions whose true token id exceeds the \
             model's logit width; excluded identically on both arms"
        );
    }
    acc
}

#[test]
#[ignore = "loads the ~22.6 GB unsloth/Qwen3.8-27B-NVFP4 checkpoint on the wgpu backend and scores \
            a 2048-token corpus slice; set NV_QWEN38_WGPU_TEST=1, NV_PPL_TEST=1, NV_PPL_CORPUS"]
fn qwen3_8_27b_nvfp4_wgpu_teacher_forced_ppl_beats_shuffled_control() {
    require_qwen38_wgpu_gate();
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    assert!(
        std::env::var("NV_QWEN38_LAYERS").is_err(),
        "NV_QWEN38_LAYERS truncates the trunk and would make the ppl number non-canonical; \
         unset it for the perplexity gate"
    );

    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = qwen38_snapshot_dir();
    let (cfg, _) = parse_config_with_optional_layer_truncation(&dir);

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let n = PPL_CORPUS_SLICE_TOKENS_N_2048_CHOSEN_SO_31B_LOAD_PLUS_FOUR_EAGER_512_BLOCKS_FITS_2_TO_3_MIN;
    let slice = first_n_corpus_tokens_after_tokenization(&tok, &corpus, n);

    let mut m = boot_model(&dir, cfg, n + 8);

    let label = checkpoint_label_from_snapshot_dir(&dir);
    eprintln!(
        "PPL basis: {label} corpus_slice_tokens={n} backend=wgpu batch=1 path=decode_step_logits \
         quant=nvfp4 note=architectural-vs-own-trunk-not-parity-with-parent-python"
    );

    let real = score_slice_via_wgpu_decode_step_logits(&mut m, &slice);

    let mut shuffled = slice.clone();
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
        &mut shuffled,
    );
    m.reset().expect("reset between arms");
    let shuffled = score_slice_via_wgpu_decode_step_logits(&mut m, &shuffled);

    assert_eq!(
        real.scored_positions, shuffled.scored_positions,
        "real and shuffled arms scored different position counts"
    );
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        "qwen3_8-dense-wgpu",
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}

const GEN_ARM_PREFIXES_OF_ONE_FREERUN_SO_ALL_ARMS_SHARE_DEPTH_AND_BOOT: [usize; 3] =
    [256, 1024, 2048];

#[test]
#[ignore = "loads the ~22.6 GB checkpoint, chunk-prefills NV_CTX_TOKENS prompt tokens (plain \
            integer, default 2048), then greedily free-runs 2048 tokens timing every step; the \
            256/1024/2048 gen arms are prefixes of that one freerun so every arm shares the same \
            boot, depth and clock state; set NV_QWEN38_WGPU_TEST=1"]
fn qwen38_wgpu_ttft_then_greedy_freerun_gen_arms_are_prefixes_of_one_2048_token_run() {
    require_qwen38_wgpu_gate();
    let dir = qwen38_snapshot_dir();
    let (cfg, truncated) = parse_config_with_optional_layer_truncation(&dir);
    assert!(
        truncated.is_none(),
        "NV_QWEN38_LAYERS truncates the trunk and would make gen-arm numbers non-canonical; unset it"
    );
    let vocab = cfg.vocab_size;
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let mut prompt = qwen_chat_prompt_ids(&tok);
    let depth = env_usize("NV_CTX_TOKENS", 2048);
    assert!(
        depth >= prompt.len(),
        "NV_CTX_TOKENS={depth} shorter than the {}-token chat prompt",
        prompt.len()
    );
    let filler = *prompt.last().expect("chat prompt is never empty");
    prompt.resize(depth, filler);
    let gen_total = *GEN_ARM_PREFIXES_OF_ONE_FREERUN_SO_ALL_ARMS_SHARE_DEPTH_AND_BOOT
        .last()
        .expect("arm ladder is never empty");
    let mut m = boot_model(&dir, cfg, depth + gen_total + 8);
    let label = checkpoint_label_from_snapshot_dir(&dir);

    let t0 = Instant::now();
    let first = m.prefill(&prompt).expect("prefill the padded chat prompt");
    let ttft_s = t0.elapsed().as_secs_f64();
    assert!(
        (first as usize) < vocab,
        "prefill emitted token {first} outside the vocab width {vocab}"
    );

    let mut step_ms: Vec<f64> = Vec::with_capacity(gen_total - 1);
    let mut tok_prev = first;
    for i in 1..gen_total {
        let t = Instant::now();
        let next = m
            .decode_step(tok_prev)
            .unwrap_or_else(|e| panic!("freerun decode step {i} at depth {depth}: {e:#}"));
        step_ms.push(t.elapsed().as_secs_f64() * 1e3);
        assert!(
            (next as usize) < vocab,
            "freerun step {i} emitted token {next} outside the vocab width {vocab}"
        );
        tok_prev = next;
    }

    for &arm in &GEN_ARM_PREFIXES_OF_ONE_FREERUN_SO_ALL_ARMS_SHARE_DEPTH_AND_BOOT {
        let decode_steps = arm - 1;
        let wall_ms: f64 = step_ms[..decode_steps].iter().sum();
        let mut sorted = step_ms[..decode_steps].to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = sorted[sorted.len() / 2];
        println!(
            "GEN-ARM qwen38-wgpu checkpoint={label} backend=wgpu batch=1 prefill_tokens={depth} \
             gen={arm} ttft_s={ttft_s:.3} prefill_tok_s={:.1} decode_steps={decode_steps} \
             decode_median_ms={median_ms:.3} decode_tok_s={:.1} basis=greedy_freerun_prefix_arms_share_one_boot_no_warmup",
            depth as f64 / ttft_s,
            decode_steps as f64 / (wall_ms / 1e3)
        );
    }
}

const FUSED_DECODE_ENVS: [&str; 4] =
    ["NV_Q3D_FUSE_DN", "NV_Q3D_FUSE_ATTN", "NV_Q3D_FUSE_DN_GEMV", "NV_Q3D_FUSE_MLP"];

const FUSED2_DECODE_ENVS_DEFAULT_OFF: [&str; 2] = ["NV_Q3D_FUSE_MLP_GEMV", "NV_Q3D_FUSE_KVW"];

const NVFP4_MLP_LAYERS_ARE_THE_FIRST_56_ON_THE_27B_CHECKPOINT_LAYERS_56_TO_63_SHIP_FP8_MLP:
    usize = 56;

const KVW_FOLD_REMOVES_KV_WRITE_AND_ONE_FP8_QUANTIZER_PER_ATTN_LAYER: usize = 2;

const MLP_GEMV_2W_REMOVES_ONE_GEMV_PER_NVFP4_MLP_LAYER: usize = 1;

#[test]
#[ignore = "boots the ~22.6 GB checkpoint three times (all fusion envs off, round-1 defaults, \
            round-1 plus the round-2 arms) and compares greedy decode logits bit-for-bit; the \
            merged fp8 DN gemv, the nvfp4 gate+up 2w gemv and the kv-write+fp8-quant fold only \
            engage on real weights, so this is their only full-graph bit gate; set \
            NV_QWEN38_WGPU_TEST=1 (NV_QWEN38_DIR/LAYERS/PROMPT and NV_QWEN38_FUSED_AB_STEPS \
            optional)"]
fn fused_decode_envs_are_bit_identical_on_the_real_checkpoint() {
    require_qwen38_wgpu_gate();
    let dir = qwen38_snapshot_dir();
    let (cfg, truncated) = parse_config_with_optional_layer_truncation(&dir);
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let prompt = qwen_chat_prompt_ids(&tok);
    let steps = env_usize("NV_QWEN38_FUSED_AB_STEPS", 32);
    let max_seq = prompt.len() + steps + 16;

    let mut run = |fuse1: bool, fuse2: bool| -> (usize, Vec<(u32, Vec<u32>)>) {
        for k in FUSED_DECODE_ENVS {
            std::env::set_var(k, if fuse1 { "1" } else { "0" });
        }
        for k in FUSED2_DECODE_ENVS_DEFAULT_OFF {
            std::env::set_var(k, if fuse2 { "1" } else { "0" });
        }
        let mut m = boot_model(&dir, cfg.clone(), max_seq);
        let passes = m.pass_count();
        let first = m.prefill(&prompt).expect("prefill the chat prompt");
        let mut out = Vec::new();
        let mut t = first;
        for i in 0..steps {
            let (arg, logits) = m
                .decode_step_logits(t)
                .unwrap_or_else(|e| panic!("decode step {i} (fuse1={fuse1} fuse2={fuse2}): {e:#}"));
            out.push((arg, logits.iter().map(|x| x.to_bits()).collect()));
            t = arg;
        }
        for k in FUSED_DECODE_ENVS.iter().chain(FUSED2_DECODE_ENVS_DEFAULT_OFF.iter()) {
            std::env::remove_var(k);
        }
        (passes, out)
    };
    let (base_passes, base) = run(false, false);
    let (fused_passes, fused) = run(true, false);
    let (fused2_passes, fused2) = run(true, true);
    assert!(
        fused_passes < base_passes,
        "the fused build recorded {fused_passes} decode passes against {base_passes} unfused; \
         the fusion envs did not reach the builder and this A/B compared the arm to itself"
    );
    let attn_layers = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, nv_models::qwen3_5_moe::LayerType::FullAttention))
        .count();
    let nvfp4_mlp_layers = cfg
        .num_hidden_layers
        .min(NVFP4_MLP_LAYERS_ARE_THE_FIRST_56_ON_THE_27B_CHECKPOINT_LAYERS_56_TO_63_SHIP_FP8_MLP);
    assert_eq!(
        fused_passes - fused2_passes,
        attn_layers * KVW_FOLD_REMOVES_KV_WRITE_AND_ONE_FP8_QUANTIZER_PER_ATTN_LAYER
            + nvfp4_mlp_layers * MLP_GEMV_2W_REMOVES_ONE_GEMV_PER_NVFP4_MLP_LAYER,
        "the round-2 arms must drop exactly {}x{attn_layers} kv-fold plus \
         {}x{nvfp4_mlp_layers} gate+up-merge dispatches (round-1 {fused_passes} passes, \
         round-2 {fused2_passes}); a shortfall means an arm silently fell back to its chain \
         and this A/B partly compared an arm to itself",
        KVW_FOLD_REMOVES_KV_WRITE_AND_ONE_FP8_QUANTIZER_PER_ATTN_LAYER,
        MLP_GEMV_2W_REMOVES_ONE_GEMV_PER_NVFP4_MLP_LAYER
    );
    for (arm, got) in [("round-1", &fused), ("round-2", &fused2)] {
        for (i, ((ba, bl), (fa, fl))) in base.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                ba, fa,
                "{arm}: greedy argmax diverged at step {i}: unfused {ba} vs fused {fa}"
            );
            let diff = bl
                .iter()
                .zip(fl.iter())
                .enumerate()
                .find(|(_, (b, f))| b != f);
            assert!(
                diff.is_none(),
                "{arm}: logit bits diverged at step {i}, first at vocab index {:?}: the fused \
                 decode arms preserve per-element arithmetic order, so anything short of bit \
                 identity on the real checkpoint is a fusion defect, not rounding",
                diff.map(|(j, _)| j)
            );
        }
    }
    println!(
        "QWEN38-WGPU-FUSED-AB truncated={} steps={steps} passes_unfused={base_passes} \
         passes_fused={fused_passes} passes_fused2={fused2_passes} verdict=bit_identical",
        truncated.is_some()
    );
}
