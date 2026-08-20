#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::kernels::kv_nvfp4 as kv4;
use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseConfig;
use nv_models::qwen3_5_moe::LayerType;
use nv_models::qwen3_5_moe_wgpu::HostBf16Lin;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32) - 1.0
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

fn bf16_lin(r: &mut Lcg, n: usize, k: usize, scale: f32) -> HostBf16Lin {
    HostBf16Lin {
        w: r.bf16_vec(n * k, scale),
        n,
        k,
    }
}

fn norm_vec(r: &mut Lcg, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| half::bf16::from_f32(1.0 + 0.1 * r.next_f32()).to_bits())
        .collect()
}

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static L: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn gpu_present_or_explicit_skip() -> bool {
    if nv_kernels::wgpu_backend::WgpuContext::shared().is_ok() {
        return true;
    }
    if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() == Ok("1") {
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): no wgpu adapter");
        return false;
    }
    panic!("no wgpu adapter; this gate must never silently skip");
}

fn tiny_config() -> Qwen3_5DenseConfig {
    Qwen3_5DenseConfig {
        hidden_size: 128,
        num_hidden_layers: 4,
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
        layer_types: vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ],
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        attn_output_gate: true,
        tie_word_embeddings: false,
    }
}

fn tiny_weights(cfg: &Qwen3_5DenseConfig, seed: u64) -> q3d::HostDenseWeights {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;

    let mut layers = Vec::new();
    for li in 0..cfg.num_hidden_layers {
        let mixer = match cfg.layer_types[li] {
            LayerType::LinearAttention => {
                q3d::HostDenseMixer::Delta(Box::new(nv_models::qwen3_5_moe_wgpu::HostDeltaNet {
                    in_proj_qkv: bf16_lin(&mut r, conv_dim, hidden, 0.12),
                    in_proj_z: bf16_lin(&mut r, value_dim, hidden, 0.12),
                    in_proj_ab: bf16_lin(&mut r, 2 * cfg.linear_num_value_heads, hidden, 0.12),
                    conv1d: r.f32_vec(conv_dim * ks, 0.4),
                    a_log: r.f32_vec(cfg.linear_num_value_heads, 0.5),
                    dt_bias: r.f32_vec(cfg.linear_num_value_heads, 0.5),
                    norm_w: norm_vec(&mut r, cfg.linear_value_head_dim),
                    out_proj: bf16_lin(&mut r, hidden, value_dim, 0.12),
                }))
            }
            LayerType::FullAttention => {
                let q_out = cfg.num_attention_heads * hd * 2;
                let kv_out = cfg.num_key_value_heads * hd;
                q3d::HostDenseMixer::Attn(Box::new(q3d::HostDenseAttention {
                    q: bf16_lin(&mut r, q_out, hidden, 0.12).into(),
                    k: bf16_lin(&mut r, kv_out, hidden, 0.12).into(),
                    v: bf16_lin(&mut r, kv_out, hidden, 0.12).into(),
                    o: bf16_lin(&mut r, hidden, cfg.num_attention_heads * hd, 0.12).into(),
                    q_norm: norm_vec(&mut r, hd),
                    k_norm: norm_vec(&mut r, hd),
                }))
            }
        };
        layers.push(q3d::HostDenseLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            mixer,
            delta_fp8: Default::default(),
            mlp: q3d::HostDenseMlp {
                gate: bf16_lin(&mut r, inter, hidden, 0.15).into(),
                up: bf16_lin(&mut r, inter, hidden, 0.15).into(),
                down: bf16_lin(&mut r, hidden, inter, 0.15).into(),
            },
        });
    }

    q3d::HostDenseWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: norm_vec(&mut r, hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

const ARM_ENV: &str = q3d::KV_NVFP4_ENV_OFF_BY_DEFAULT_FP8_STAYS_THE_SHIPPED_KV_DECODE_FORMAT;

const MAX_SEQ: usize = 96;

const PROMPT_LEN_CROSSES_A_32_TOKEN_K_BLOCK_AND_A_16_ROW_CHUNK: usize = 39;

const GREEDY_STEPS: usize = 8;

fn build(cfg: &Qwen3_5DenseConfig, hw: &q3d::HostDenseWeights, arm: Option<&str>) -> q3d::Qwen3_5DenseWgpu {
    if let Some(a) = arm {
        std::env::set_var(ARM_ENV, a);
    }
    let gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), hw, MAX_SEQ).expect("build");
    std::env::remove_var(ARM_ENV);
    gpu
}

fn drive(gpu: &mut q3d::Qwen3_5DenseWgpu) -> (usize, Vec<u32>) {
    let tokens: Vec<u32> = (0..PROMPT_LEN_CROSSES_A_32_TOKEN_K_BLOCK_AND_A_16_ROW_CHUNK as u32)
        .map(|i| (i * 7 + 3) % 64)
        .collect();
    let done = gpu.prefill_tokens(&tokens).expect("prefill_tokens");
    for t in &tokens[done..] {
        gpu.prefill_step(*t).expect("tail prefill step");
    }
    let mut ids = Vec::new();
    let mut next = 2u32;
    for _ in 0..GREEDY_STEPS {
        next = gpu.decode_step(next).expect("greedy step");
        assert!((next as usize) < 64, "greedy token out of vocab");
        ids.push(next);
    }
    (tokens.len() + GREEDY_STEPS, ids)
}

fn attn_state_groups(
    bufs: &[(u64, Vec<u32>)],
    n_kv: usize,
    hd: usize,
    k4: bool,
) -> Vec<Vec<&Vec<u32>>> {
    let kc = (MAX_SEQ * n_kv * hd * 2) as u64;
    let kc8 = (MAX_SEQ * n_kv * hd) as u64;
    let sc = (MAX_SEQ * n_kv * 4) as u64;
    let p4 = (MAX_SEQ * n_kv * hd / 2) as u64;
    let k4s = (kv4::k_scale_blocks(MAX_SEQ) * n_kv * hd * 4) as u64;
    let mut want = vec![kc, kc, kc8, kc8, sc, sc, p4, sc];
    if k4 {
        want.extend([p4, k4s]);
    }
    let mut groups = Vec::new();
    let mut i = 0;
    while i + want.len() <= bufs.len() {
        if (0..want.len()).all(|j| bufs[i + j].0 == want[j]) {
            groups.push(bufs[i..i + want.len()].iter().map(|(_, w)| w).collect());
            i += want.len();
        } else {
            i += 1;
        }
    }
    groups
}

fn as_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn as_f32(words: &[u32]) -> Vec<f32> {
    words.iter().map(|w| f32::from_bits(*w)).collect()
}

fn as_bf16(words: &[u32]) -> Vec<u16> {
    words
        .iter()
        .flat_map(|w| [(*w & 0xffff) as u16, (*w >> 16) as u16])
        .collect()
}

#[test]
fn the_nvfp4_caches_hold_exactly_the_cpu_reference_quantization_of_the_bf16_cache() {
    let _g = env_lock();
    if !gpu_present_or_explicit_skip() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x4b1d_0001);
    let n_kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    for arm in ["v4k8", "k4v4"] {
        let k4 = arm == "k4v4";
        let mut gpu = build(&cfg, &hw, Some(arm));
        let (pos, ids) = drive(&mut gpu);
        eprintln!("[kv-nvfp4] arm {arm}: {pos} positions landed, greedy tail {ids:?}");
        let bufs = gpu.debug_state_buffer_words_for_test();
        let groups = attn_state_groups(&bufs, n_kv, hd, k4);
        let n_attn = cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::FullAttention))
            .count();
        assert_eq!(
            groups.len(),
            n_attn,
            "arm {arm}: expected one bf16+fp8+nvfp4 state group per full-attention layer"
        );
        for (gi, g) in groups.iter().enumerate() {
            let kc = as_bf16(g[0]);
            let vc = as_bf16(g[1]);
            let v4w = as_bytes(g[6]);
            let v4s = as_f32(g[7]);
            let mut want_v4w = vec![0u8; v4w.len()];
            let mut want_v4s = vec![0f32; v4s.len()];
            kv4::cpu_quantize_kv_nvfp4_v_rows(
                &vc, &mut want_v4w, &mut want_v4s, 0, pos, n_kv, hd, MAX_SEQ,
            );
            let live_bytes = pos * n_kv * hd / 2;
            assert_eq!(
                &v4w[..live_bytes],
                &want_v4w[..live_bytes],
                "arm {arm} layer group {gi}: V nibbles diverge from the CPU reference"
            );
            let live_scales = pos * n_kv;
            assert_eq!(
                v4s[..live_scales]
                    .iter()
                    .map(|s| s.to_bits())
                    .collect::<Vec<_>>(),
                want_v4s[..live_scales]
                    .iter()
                    .map(|s| s.to_bits())
                    .collect::<Vec<_>>(),
                "arm {arm} layer group {gi}: V row scales diverge from the CPU reference"
            );
            assert!(
                want_v4s[..live_scales].iter().any(|s| *s != 1.0 && *s != 0.0),
                "arm {arm} layer group {gi}: every V scale is trivial; the gate would be vacuous"
            );
            if k4 {
                let k4w = as_bytes(g[8]);
                let k4s = as_f32(g[9]);
                let mut want_k4w = vec![0u8; k4w.len()];
                let mut want_k4s = vec![0f32; k4s.len()];
                kv4::cpu_quantize_kv_nvfp4_k_channel_blocks(
                    &kc, &mut want_k4w, &mut want_k4s, 0, pos, n_kv, hd, MAX_SEQ,
                );
                assert_eq!(
                    &k4w[..live_bytes],
                    &want_k4w[..live_bytes],
                    "arm {arm} layer group {gi}: K nibbles diverge from the CPU reference; the \
                     streaming block re-quantize must end bit-identical to one whole-range pass"
                );
                let live_blocks = pos.div_ceil(32);
                let live_ch = live_blocks * n_kv * hd;
                assert_eq!(
                    k4s[..live_ch]
                        .iter()
                        .map(|s| s.to_bits())
                        .collect::<Vec<_>>(),
                    want_k4s[..live_ch]
                        .iter()
                        .map(|s| s.to_bits())
                        .collect::<Vec<_>>(),
                    "arm {arm} layer group {gi}: K per-channel block scales diverge from the CPU \
                     reference"
                );
            }
        }
    }
}

#[test]
fn nothing_moves_with_the_env_unset_and_each_arm_adds_exactly_its_own_state() {
    let _g = env_lock();
    if !gpu_present_or_explicit_skip() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x4b1d_0002);
    let n_attn = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::FullAttention))
        .count();
    let count = |arm: Option<&str>| build(&cfg, &hw, arm).debug_state_buffer_words_for_test().len();
    let base = count(None);
    assert_eq!(
        count(Some("v4k8")),
        base + 2 * n_attn,
        "v4k8 adds one payload + one row-scale state buffer per attention layer and nothing else"
    );
    assert_eq!(
        count(Some("k4v4")),
        base + 4 * n_attn,
        "k4v4 adds K and V payload + scale state buffers per attention layer and nothing else"
    );
}

#[test]
fn each_arm_decodes_deterministically_across_two_builds() {
    let _g = env_lock();
    if !gpu_present_or_explicit_skip() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x4b1d_0003);
    for arm in ["v4k8", "k4v4"] {
        let (_, a) = drive(&mut build(&cfg, &hw, Some(arm)));
        let (_, b) = drive(&mut build(&cfg, &hw, Some(arm)));
        assert_eq!(a, b, "arm {arm}: greedy continuation must be build-deterministic");
    }
}
