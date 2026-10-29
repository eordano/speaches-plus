use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

const HEADS: usize = 12;
const HEAD_DIM: usize = 256;

pub const DECODE_CAPABLE_TINY_QWEN4_EXP_IS_12_HEADS_OF_256_BECAUSE_THE_ROW_ATTENTION_KERNEL_IS_BUILT_FOR_THAT_SHAPE: usize = HEADS * HEAD_DIM;

pub fn config() -> String {
    config_json()
}

pub fn weights(seed: u64) -> HashMap<String, Tensor> {
    tensors(seed)
}

pub fn vocab() -> usize {
    VOCAB
}

const H_512_BECAUSE_THE_HC_NORM_KERNEL_TILES_HC_TIMES_HIDDEN_BY_2048: usize = 512;
const H: usize = H_512_BECAUSE_THE_HC_NORM_KERNEL_TILES_HC_TIMES_HIDDEN_BY_2048;
const HC: usize = 4;
const LOWRANK: usize = 32;
const GDN_HEADS: usize = 2;
const GDN_DIM: usize = 128;
const EXPERTS: usize = 8;
const MOE_INTER: usize = 128;
const SHARED_INTER: usize = 128;
const IDX_HEADS: usize = 2;
const IDX_DIM: usize = 128;
const VOCAB: usize = 512;
const LAYERS_GDN_THEN_ATTENTION: [&str; 2] = ["linear_attention", "full_attention"];
const E4M3_HALF_IS_A_FINITE_BLOCK_SCALE_FOR_EVERY_NVFP4_GROUP: u8 = 0x30;

fn config_json() -> String {
    config_json_with_max_pos(4096)
}

fn config_json_with_max_pos(max_pos: usize) -> String {
    format!(
        r#"{{"text_config": {{
  "layer_types": ["{}", "{}"],
  "rope_parameters": {{"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}},
  "head_dim": {HEAD_DIM}, "hidden_size": {H}, "num_hidden_layers": 2,
  "num_attention_heads": {HEADS}, "num_key_value_heads": 1, "rms_norm_eps": 1e-6,
  "vocab_size": {VOCAB}, "max_position_embeddings": {max_pos},
  "linear_num_key_heads": {GDN_HEADS}, "linear_num_value_heads": {GDN_HEADS},
  "linear_key_head_dim": {GDN_DIM}, "linear_value_head_dim": {GDN_DIM}, "linear_conv_kernel_dim": 4,
  "num_experts": {EXPERTS}, "num_experts_per_tok": 2, "moe_intermediate_size": {MOE_INTER},
  "shared_expert_intermediate_size": {SHARED_INTER}, "hc_count": {HC}, "hc_lowrank": {LOWRANK},
  "indexer_n_heads": {IDX_HEADS}, "indexer_kv_heads": 1, "indexer_head_dim": {IDX_DIM},
  "indexer_budget": 2048, "indexer_compress_ratio": 4, "eos_token_id": 1,
  "output_gate_type": "sigmoid"
}}}}"#,
        LAYERS_GDN_THEN_ATTENTION[0], LAYERS_GDN_THEN_ATTENTION[1]
    )
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as u32 as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }

    fn bf16(&mut self, shape: &[usize], scale: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| self.next() * scale).collect();
        Tensor::from_vec(v, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }

    fn bytes(&mut self, n: usize) -> Tensor {
        let v: Vec<u8> = (0..n).map(|_| (self.next().to_bits() >> 7) as u8).collect();
        Tensor::from_vec(v, n, &Device::Cpu).unwrap()
    }
}

fn small(n: usize) -> Tensor {
    Tensor::from_vec(vec![0.05f32; n], n, &Device::Cpu).unwrap()
}

fn scalar(v: f32) -> Tensor {
    Tensor::from_vec(vec![v], 1, &Device::Cpu).unwrap()
}

fn hyper(t: &mut HashMap<String, Tensor>, r: &mut Lcg, p: &str, inject: bool) {
    let hh = HC * H;
    t.insert(format!("{p}.hc_norm.weight"), small(hh));
    t.insert(
        format!("{p}.input_mix_weight_down.weight"),
        r.bf16(&[LOWRANK, hh], 0.05),
    );
    t.insert(
        format!("{p}.input_mix_weight_up.weight"),
        r.bf16(&[hh, LOWRANK], 0.05),
    );
    if inject {
        t.insert(
            format!("{p}.block_inject_weight.weight"),
            r.bf16(&[HC, hh], 0.05),
        );
    }
}

fn tensors(seed: u64) -> HashMap<String, Tensor> {
    let mut r = Lcg(seed | 1);
    let mut t = HashMap::new();
    t.insert(
        "model.language_model.embed_tokens.weight".into(),
        r.bf16(&[VOCAB, H], 0.5),
    );
    t.insert("lm_head.weight".into(), r.bf16(&[VOCAB, H], 0.1));
    hyper(
        &mut t,
        &mut r,
        "model.language_model.hyper_connection_mixer",
        false,
    );
    for (i, kind) in LAYERS_GDN_THEN_ATTENTION.iter().enumerate() {
        let p = format!("model.language_model.layers.{i}");
        hyper(&mut t, &mut r, &format!("{p}.attn_hyper_connection"), true);
        hyper(&mut t, &mut r, &format!("{p}.mlp_hyper_connection"), true);
        if *kind == "linear_attention" {
            let q = format!("{p}.linear_attn");
            let key = GDN_HEADS * GDN_DIM;
            let conv = 2 * key + key;
            t.insert(format!("{q}.in_proj_qkv.weight"), r.bf16(&[conv, H], 0.05));
            t.insert(format!("{q}.in_proj_z.weight"), r.bf16(&[key, H], 0.05));
            t.insert(
                format!("{q}.in_proj_a.weight"),
                r.bf16(&[GDN_HEADS, H], 0.05),
            );
            t.insert(
                format!("{q}.in_proj_b.weight"),
                r.bf16(&[GDN_HEADS, H], 0.05),
            );
            t.insert(format!("{q}.conv1d.weight"), r.bf16(&[conv, 1, 4], 0.3));
            t.insert(format!("{q}.A_log"), r.bf16(&[GDN_HEADS], 0.5));
            t.insert(format!("{q}.dt_bias"), r.bf16(&[GDN_HEADS], 0.5));
            t.insert(
                format!("{q}.norm.weight"),
                r.bf16(&[GDN_DIM], 0.1).affine(1.0, 1.0).unwrap(),
            );
            t.insert(format!("{q}.out_proj.weight"), r.bf16(&[H, key], 0.05));
        } else {
            let a = format!("{p}.self_attn");
            t.insert(
                format!("{a}.q_proj.weight"),
                r.bf16(&[HEADS * HEAD_DIM * 2, H], 0.05),
            );
            t.insert(format!("{a}.k_proj.weight"), r.bf16(&[HEAD_DIM, H], 0.05));
            t.insert(format!("{a}.v_proj.weight"), r.bf16(&[HEAD_DIM, H], 0.05));
            t.insert(
                format!("{a}.o_proj.weight"),
                r.bf16(&[H, HEADS * HEAD_DIM], 0.05),
            );
            t.insert(format!("{a}.q_norm.weight"), small(HEAD_DIM));
            t.insert(format!("{a}.k_norm.weight"), small(HEAD_DIM));
            t.insert(
                format!("{a}.indexer.index_qk_proj.weight"),
                r.bf16(&[(IDX_HEADS + 1) * IDX_DIM, H], 0.05),
            );
            t.insert(format!("{a}.indexer.q_layernorm.weight"), small(IDX_DIM));
            t.insert(format!("{a}.indexer.k_layernorm.weight"), small(IDX_DIM));
        }
        let m = format!("{p}.mlp");
        t.insert(format!("{m}.gate.weight"), r.bf16(&[EXPERTS, H], 0.1));
        for e in 0..EXPERTS {
            for (proj, rows, k) in [
                ("gate_proj", MOE_INTER, H),
                ("up_proj", MOE_INTER, H),
                ("down_proj", H, MOE_INTER),
            ] {
                let b = format!("{m}.experts.{e}.{proj}");
                t.insert(format!("{b}.weight"), r.bytes(rows * k / 2));
                t.insert(
                    format!("{b}.weight_scale"),
                    Tensor::from_vec(
                        vec![
                            E4M3_HALF_IS_A_FINITE_BLOCK_SCALE_FOR_EVERY_NVFP4_GROUP;
                            rows * k / 16
                        ],
                        rows * k / 16,
                        &Device::Cpu,
                    )
                    .unwrap(),
                );
                t.insert(format!("{b}.weight_scale_2"), scalar(0.02));
                t.insert(format!("{b}.input_scale"), scalar(1.0));
            }
        }
        t.insert(
            format!("{m}.shared_expert.gate_proj.weight"),
            r.bf16(&[SHARED_INTER, H], 0.05),
        );
        t.insert(
            format!("{m}.shared_expert.up_proj.weight"),
            r.bf16(&[SHARED_INTER, H], 0.05),
        );
        t.insert(
            format!("{m}.shared_expert.down_proj.weight"),
            r.bf16(&[H, SHARED_INTER], 0.05),
        );
        t.insert(
            format!("{m}.shared_expert_gate.weight"),
            r.bf16(&[1, H], 0.05),
        );
    }
    t
}
