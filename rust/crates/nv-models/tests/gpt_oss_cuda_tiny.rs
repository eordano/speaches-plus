#![cfg(feature = "cuda")]

use candle_core::Device;
use nv_models::gpt_oss as go;
use nv_models::gpt_oss::{GptOssConfig, GptOssLayerType};
use nv_models::gpt_oss_cuda::GptOssCuda;
use nv_quant::mxfp4::Mxfp4Tensor;

const CROSS_M_DEVIATION_IS_CUBLAS_TILING_NOT_A_DECODE_DEFECT: &str =
    "an M>1 chunk and M successive M=1 steps run the same weights through different cublas GEMM \
     shapes, so their accumulation order over the k axis differs and the logits cannot be bit \
     identical; what must hold is that every argmax agrees and the logit deviation stays at the \
     bf16 rounding scale of the trunk.";

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let v = ((z >> 40) as u32) as f32 / (1u64 << 23) as f32;
        v - 1.0
    }
    fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
            .collect()
    }
    fn f32_rows(&mut self, rows: usize, cols: usize, scale: f32) -> Vec<Vec<f32>> {
        (0..rows)
            .map(|_| (0..cols).map(|_| self.next_f32() * scale).collect())
            .collect()
    }
}

fn tiny_config() -> GptOssConfig {
    GptOssConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        intermediate_size: 32,
        num_local_experts: 4,
        num_experts_per_tok: 2,
        vocab_size: 64,
        max_position_embeddings: 64,
        sliding_window: 4,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        swiglu_limit: 7.0,
        layer_types: vec![GptOssLayerType::Sliding, GptOssLayerType::Full],
        yarn_factor: 4.0,
        yarn_beta_fast: 32.0,
        yarn_beta_slow: 1.0,
        yarn_original_max: 16,
        tie_word_embeddings: false,
    }
}

fn bf16_lin(r: &mut Lcg, n: usize, k: usize, scale: f32, bias: bool) -> go::HostBf16Lin {
    go::HostBf16Lin {
        w: r.bf16_vec(n * k, scale),
        bias: if bias {
            r.bf16_vec(n, scale)
        } else {
            Vec::new()
        },
        n,
        k,
    }
}

fn norm_vec(r: &mut Lcg, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| half::bf16::from_f32(1.0 + 0.1 * r.next_f32()).to_bits())
        .collect()
}

fn mx_stack(r: &mut Lcg, e: usize, n: usize, k: usize, scale: f32) -> go::HostMxStack {
    let mats: Vec<Mxfp4Tensor> = (0..e)
        .map(|_| Mxfp4Tensor::quantize_rows(&r.f32_rows(n, k, scale)))
        .collect();
    let biases: Vec<Vec<u16>> = (0..e).map(|_| r.bf16_vec(n, scale)).collect();
    go::stack_mx_host(&mats, &biases)
}

fn tiny_weights(cfg: &GptOssConfig, seed: u64) -> go::HostWeights {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let q_out = cfg.num_attention_heads * hd;
    let kv_out = cfg.num_key_value_heads * hd;

    let mut layers = Vec::new();
    for _ in 0..cfg.num_hidden_layers {
        layers.push(go::HostLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            attn: go::HostAttn {
                q: bf16_lin(&mut r, q_out, hidden, 0.12, true),
                k: bf16_lin(&mut r, kv_out, hidden, 0.12, true),
                v: bf16_lin(&mut r, kv_out, hidden, 0.12, true),
                o: bf16_lin(&mut r, hidden, q_out, 0.12, true),
                sinks: (0..cfg.num_attention_heads)
                    .map(|_| r.next_f32() * 0.5)
                    .collect(),
            },
            moe: go::HostMoe {
                router: bf16_lin(&mut r, cfg.num_local_experts, hidden, 0.3, true),
                gate_up: mx_stack(&mut r, cfg.num_local_experts, 2 * inter, hidden, 0.15),
                down: mx_stack(&mut r, cfg.num_local_experts, hidden, inter, 0.15),
            },
        });
    }

    go::HostWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: norm_vec(&mut r, hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

fn cuda_device() -> Option<Device> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("[skip] no cuda device: {e}");
            None
        }
    }
}

fn spread(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_abs = 0f32;
    let mut denom = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        max_abs = max_abs.max((x - y).abs());
        denom = denom.max(x.abs().max(y.abs()));
    }
    (max_abs, max_abs / denom.max(1e-6))
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

#[test]
fn cuda_decode_tracks_the_gpt_oss_host_reference_including_the_attention_sink_fold() {
    let Some(device) = cuda_device() else { return };
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xA51E_1234);
    let model = GptOssCuda::from_host(cfg.clone(), &hw, &device).unwrap();
    let mut cache = model.new_kv_cache(32).unwrap();
    let mut refst = go::RefState::new(&cfg);

    let tokens: Vec<u32> = (0..12u32).map(|i| (i * 7 + 3) % cfg.vocab_size as u32).collect();
    let mut worst_rel = 0f32;
    for (step, tok) in tokens.iter().enumerate() {
        let got = model
            .forward_last_logits(&[*tok], &[step as u32], &mut cache)
            .unwrap();
        let want = go::reference_step(&cfg, &hw, &mut refst, *tok).unwrap();
        assert_eq!(got.len(), want.len());
        let (abs, rel) = spread(&got, &want);
        worst_rel = worst_rel.max(rel);
        assert_eq!(
            argmax(&got),
            argmax(&want),
            "step {step}: cuda argmax {} != host reference argmax {} (max abs {abs})",
            argmax(&got),
            argmax(&want)
        );
        assert!(
            rel < 0.05,
            "step {step}: cuda vs host-reference logits diverged by {rel} relative ({abs} abs); \
             the two paths agree in algebra and differ only in where bf16 rounding lands \
             (candle bf16 GEMM output rounding vs the reference's f32 accumulate with explicit \
             rbf sites), so anything past a few percent is a decode defect, not rounding"
        );
    }
    eprintln!("[gpt-oss cuda] worst relative logit deviation vs host reference: {worst_rel}");
    assert!(
        worst_rel > 0.0,
        "a zero deviation across 12 steps means the reference and the model are the same code path"
    );
}

#[test]
fn a_prefilled_chunk_and_the_same_tokens_stepped_one_at_a_time_agree_on_every_argmax() {
    let Some(device) = cuda_device() else { return };
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x5EED_9001);
    let model = GptOssCuda::from_host(cfg.clone(), &hw, &device).unwrap();
    let tokens: Vec<u32> = vec![5, 17, 42, 8, 23, 61, 2];
    let positions: Vec<u32> = (0..tokens.len() as u32).collect();

    let mut chunk_cache = model.new_kv_cache(32).unwrap();
    let chunked = model
        .forward_all_logits(&tokens, &positions, &mut chunk_cache)
        .unwrap()
        .to_vec3::<f32>()
        .unwrap()
        .remove(0);

    let mut step_cache = model.new_kv_cache(32).unwrap();
    let mut stepped = Vec::new();
    for (i, tok) in tokens.iter().enumerate() {
        stepped.push(
            model
                .forward_last_logits(&[*tok], &[i as u32], &mut step_cache)
                .unwrap(),
        );
    }

    let mut worst = 0f32;
    for (i, (c, s)) in chunked.iter().zip(stepped.iter()).enumerate() {
        let (abs, rel) = spread(c, s);
        worst = worst.max(rel);
        assert_eq!(
            argmax(c),
            argmax(s),
            "row {i}: chunk argmax {} != stepped argmax {} ({abs} abs). \
             {CROSS_M_DEVIATION_IS_CUBLAS_TILING_NOT_A_DECODE_DEFECT}",
            argmax(c),
            argmax(s)
        );
        assert_eq!(
            c, s,
            "row {i}: at this config the M>1 chunk is bit identical to M successive M=1 steps \
             ({abs} abs, {rel} relative apart). If a future shape breaks that, the reason is \
             allowed to be only one thing: {CROSS_M_DEVIATION_IS_CUBLAS_TILING_NOT_A_DECODE_DEFECT}"
        );
    }
    eprintln!("[gpt-oss cuda] chunk-vs-step relative logit deviation: {worst} (bit identical)");
}

#[test]
fn the_sliding_layer_stops_seeing_keys_that_fall_out_of_its_window() {
    let Some(device) = cuda_device() else { return };
    let mut cfg = tiny_config();
    cfg.layer_types = vec![GptOssLayerType::Sliding, GptOssLayerType::Sliding];
    cfg.sliding_window = 2;
    let hw = tiny_weights(&cfg, 0xBEEF_0007);
    let model = GptOssCuda::from_host(cfg.clone(), &hw, &device).unwrap();

    let tail: Vec<u32> = vec![11, 29, 6];
    let long: Vec<u32> = vec![63, 1, 44, 11, 29, 6];
    let mut c_long = model.new_kv_cache(32).unwrap();
    let mut c_tail = model.new_kv_cache(32).unwrap();
    let mut got_long = Vec::new();
    for (i, t) in long.iter().enumerate() {
        got_long = model
            .forward_last_logits(&[*t], &[i as u32], &mut c_long)
            .unwrap();
    }
    let mut got_tail = Vec::new();
    for (i, t) in tail.iter().enumerate() {
        got_tail = model
            .forward_last_logits(&[*t], &[(long.len() - tail.len() + i) as u32], &mut c_tail)
            .unwrap();
    }
    let (abs, rel) = spread(&got_long, &got_tail);
    assert_eq!(
        argmax(&got_long),
        argmax(&got_tail),
        "with sliding_window=2 on both layers the final logit's receptive field is three tokens \
         wide, so a 6-token history and its 3-token tail must agree ({abs} abs apart)"
    );
    assert!(
        rel < 1e-3,
        "the two histories feed the sliding layers identical live keys, so only the masked-out \
         columns differ and the logits must agree to rounding: got {rel} relative ({abs} abs)"
    );
}
