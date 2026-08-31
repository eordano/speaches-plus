use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_specdecode::qwen38_mtp::{
    merge_speculator_lora_into_mtp_map_before_any_quant_arm_so_the_drafter_serves_at_base_speed,
    mtp_speculator_dir_if_advertised,
    MTP_SPECULATOR_LORA_MODULES_MATCH_Q38_MTP_TRAIN_HEAD_PROJECTIONS_NORMS_EMBED_LM_HEAD_D2T_T2D_ARE_NEVER_TARGETS,
    QWEN38_MTP_SPECULATOR_KIND_MIRRORS_NV_TRAIN,
    SPECULATOR_KIND_KEY_READ_FROM_THE_MAIN_ADAPTER_CONFIG_BECAUSE_EAGLE3_SUBDIR_CONFIGS_OMIT_IT,
    SPECULATOR_SUBDIR_MIRRORS_NV_TRAIN_SPECULATOR_SUBDIR,
};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

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
        ((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| bf16::from_f32(self.next_f32() * scale).to_f32())
            .collect()
    }
}

fn bf16_cpu(vals: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(vals.to_vec(), shape, &Device::Cpu)
        .expect("cpu tensor")
        .to_dtype(DType::BF16)
        .expect("bf16")
}

fn bits(t: &Tensor) -> Vec<u16> {
    t.flatten_all()
        .expect("flatten")
        .to_vec1::<bf16>()
        .expect("host bf16")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn f32v(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("host f32")
}

const TINY_MODULE_DIMS_EXERCISE_GATED_Q_ROWS_AND_THE_2X_HIDDEN_FC_IN: [(&str, usize, usize); 8] = [
    ("mtp.fc", 8, 16),
    ("mtp.layers.0.self_attn.q_proj", 12, 8),
    ("mtp.layers.0.self_attn.k_proj", 4, 8),
    ("mtp.layers.0.self_attn.v_proj", 4, 8),
    ("mtp.layers.0.self_attn.o_proj", 8, 6),
    ("mtp.layers.0.mlp.gate_proj", 10, 8),
    ("mtp.layers.0.mlp.up_proj", 10, 8),
    ("mtp.layers.0.mlp.down_proj", 8, 10),
];

fn tiny_map(seed: u64) -> HashMap<String, Tensor> {
    let mut r = Lcg::new(seed);
    TINY_MODULE_DIMS_EXERCISE_GATED_Q_ROWS_AND_THE_2X_HIDDEN_FC_IN
        .iter()
        .map(|(m, out_f, in_f)| {
            (
                format!("{m}.weight"),
                bf16_cpu(&r.vec(out_f * in_f, 0.12), &[*out_f, *in_f]),
            )
        })
        .collect()
}

fn map_bits(map: &HashMap<String, Tensor>) -> BTreeMap<String, Vec<u16>> {
    map.iter().map(|(k, t)| (k.clone(), bits(t))).collect()
}

fn fresh_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "q38-spec-merge-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("mk fresh dir");
    dir
}

fn landed_peft_config_json(r: usize, alpha: f64, targets: &[&str], kind: Option<&str>) -> String {
    let mut cfg = serde_json::json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": false,
        "use_dora": false,
        "modules_to_save": null,
        "fan_in_fan_out": false,
        "inference_mode": true,
        "target_modules": targets,
        "base_model_name_or_path": "synthetic-base",
    });
    if let Some(kind) = kind {
        cfg.as_object_mut().unwrap().insert(
            SPECULATOR_KIND_KEY_READ_FROM_THE_MAIN_ADAPTER_CONFIG_BECAUSE_EAGLE3_SUBDIR_CONFIGS_OMIT_IT
                .to_string(),
            serde_json::Value::String(kind.to_string()),
        );
    }
    serde_json::to_string_pretty(&cfg).unwrap()
}

fn write_speculator_dir(
    dir: &Path,
    tensors: &HashMap<String, Tensor>,
    r: usize,
    alpha: f64,
    targets: &[&str],
) {
    std::fs::create_dir_all(dir).expect("mk speculator dir");
    std::fs::write(
        dir.join("adapter_config.json"),
        landed_peft_config_json(
            r,
            alpha,
            targets,
            Some(QWEN38_MTP_SPECULATOR_KIND_MIRRORS_NV_TRAIN),
        ),
    )
    .expect("write speculator config");
    candle_core::safetensors::save(tensors, dir.join("adapter_model.safetensors"))
        .expect("write speculator tensors");
}

fn pair_tensors(entries: &[(&str, Tensor, Tensor)]) -> HashMap<String, Tensor> {
    let mut t = HashMap::new();
    for (m, a, b) in entries {
        t.insert(
            format!("base_model.model.{m}.lora_A.weight"),
            a.to_dtype(DType::BF16).unwrap().contiguous().unwrap(),
        );
        t.insert(
            format!("base_model.model.{m}.lora_B.weight"),
            b.to_dtype(DType::BF16).unwrap().contiguous().unwrap(),
        );
    }
    t
}

fn merge(
    map: &mut HashMap<String, Tensor>,
    dir: &Path,
) -> anyhow::Result<Vec<String>> {
    merge_speculator_lora_into_mtp_map_before_any_quant_arm_so_the_drafter_serves_at_base_speed(
        map, dir,
    )
}

fn write_main_adapter_config(dir: &Path, kind: Option<&str>) {
    std::fs::create_dir_all(dir).expect("mk adapter dir");
    std::fs::write(
        dir.join("adapter_config.json"),
        landed_peft_config_json(2, 2.0, &["q_proj"], kind),
    )
    .expect("write main adapter config");
}

#[test]
fn speculator_advertising_gate_keys_on_the_main_config_and_the_subdir_together() {
    let plain = fresh_dir("gate-plain");
    write_main_adapter_config(&plain, None);
    assert!(
        mtp_speculator_dir_if_advertised(&plain)
            .expect("an artifact without speculator/ must load exactly as today")
            .is_none(),
        "no kind key and no subdir must mean no speculator"
    );

    let kinded_no_subdir = fresh_dir("gate-kind-no-subdir");
    write_main_adapter_config(&kinded_no_subdir, Some(QWEN38_MTP_SPECULATOR_KIND_MIRRORS_NV_TRAIN));
    let err = mtp_speculator_dir_if_advertised(&kinded_no_subdir).unwrap_err();
    assert!(
        format!("{err:#}").contains("advertises speculator kind"),
        "kind without subdir must refuse, got: {err:#}"
    );

    let subdir_no_kind = fresh_dir("gate-subdir-no-kind");
    write_main_adapter_config(&subdir_no_kind, None);
    std::fs::create_dir_all(subdir_no_kind.join(SPECULATOR_SUBDIR_MIRRORS_NV_TRAIN_SPECULATOR_SUBDIR))
        .unwrap();
    let err = mtp_speculator_dir_if_advertised(&subdir_no_kind).unwrap_err();
    assert!(
        format!("{err:#}").contains("authoritative in the main config"),
        "subdir without a main-config kind must refuse, got: {err:#}"
    );

    let eagle = fresh_dir("gate-eagle3");
    write_main_adapter_config(&eagle, Some("eagle3"));
    std::fs::create_dir_all(eagle.join(SPECULATOR_SUBDIR_MIRRORS_NV_TRAIN_SPECULATOR_SUBDIR)).unwrap();
    let err = mtp_speculator_dir_if_advertised(&eagle).unwrap_err();
    assert!(
        format!("{err:#}").contains("eagle3"),
        "an eagle3 speculator must refuse on the MTP surface, got: {err:#}"
    );

    let good = fresh_dir("gate-good");
    write_main_adapter_config(&good, Some(QWEN38_MTP_SPECULATOR_KIND_MIRRORS_NV_TRAIN));
    let sub = good.join(SPECULATOR_SUBDIR_MIRRORS_NV_TRAIN_SPECULATOR_SUBDIR);
    std::fs::create_dir_all(&sub).unwrap();
    assert_eq!(
        mtp_speculator_dir_if_advertised(&good).expect("kinded artifact with subdir"),
        Some(sub)
    );

    for d in [plain, kinded_no_subdir, subdir_no_kind, eagle, good] {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[test]
fn zero_deltas_merge_to_bitwise_identical_weights_across_all_eight_modules() {
    let mut map = tiny_map(0x5eed_0001);
    let before = map_bits(&map);
    let mut r = Lcg::new(0x5eed_0002);
    let rank = 2usize;
    let entries: Vec<(&str, Tensor, Tensor)> =
        TINY_MODULE_DIMS_EXERCISE_GATED_Q_ROWS_AND_THE_2X_HIDDEN_FC_IN
            .iter()
            .map(|(m, out_f, in_f)| {
                (
                    *m,
                    bf16_cpu(&r.vec(rank * in_f, 0.3), &[rank, *in_f]),
                    Tensor::zeros((*out_f, rank), DType::BF16, &Device::Cpu).unwrap(),
                )
            })
            .collect();
    let dir = fresh_dir("zero-delta");
    write_speculator_dir(&dir, &pair_tensors(&entries), rank, rank as f64, &["fc"]);
    let merged = merge(&mut map, &dir).expect("zero deltas must merge");
    assert_eq!(
        merged.len(),
        MTP_SPECULATOR_LORA_MODULES_MATCH_Q38_MTP_TRAIN_HEAD_PROJECTIONS_NORMS_EMBED_LM_HEAD_D2T_T2D_ARE_NEVER_TARGETS
            .len()
    );
    assert_eq!(
        map_bits(&map),
        before,
        "an untrained (B=0) speculator must merge to bitwise-identical drafter weights: the \
         rounding order is bf16 -> f32 -> +0.0 -> bf16, a round-trip"
    );
    let _ = std::fs::remove_dir_all(dir);
}

fn naive_expected_bits(w: &Tensor, a: &Tensor, b: &Tensor, scaling: f32) -> Vec<u16> {
    let (out_f, in_f) = w.dims2().unwrap();
    let (rank, _) = a.dims2().unwrap();
    let (wv, av, bv) = (f32v(w), f32v(a), f32v(b));
    (0..out_f * in_f)
        .map(|idx| {
            let (o, i) = (idx / in_f, idx % in_f);
            let mut acc = 0f32;
            for k in 0..rank {
                acc += bv[o * rank + k] * av[k * in_f + i];
            }
            bf16::from_f32(wv[idx] + scaling * acc).to_bits()
        })
        .collect()
}

const RANK_1_KEEPS_THE_NAIVE_REFERENCE_BITWISE_COMPARABLE_NO_ACCUMULATION_ORDER_EXISTS: usize = 1;

#[test]
fn nonzero_deltas_match_the_f32_reference_with_one_final_bf16_rounding() {
    let mut map = tiny_map(0x5eed_0011);
    let rank = RANK_1_KEEPS_THE_NAIVE_REFERENCE_BITWISE_COMPARABLE_NO_ACCUMULATION_ORDER_EXISTS;
    let mut r = Lcg::new(0x5eed_0012);
    let entries: Vec<(&str, Tensor, Tensor)> =
        TINY_MODULE_DIMS_EXERCISE_GATED_Q_ROWS_AND_THE_2X_HIDDEN_FC_IN
            .iter()
            .map(|(m, out_f, in_f)| {
                (
                    *m,
                    bf16_cpu(&r.vec(rank * in_f, 0.4), &[rank, *in_f]),
                    bf16_cpu(&r.vec(out_f * rank, 0.4), &[*out_f, rank]),
                )
            })
            .collect();
    let dir = fresh_dir("nonzero-delta");
    write_speculator_dir(&dir, &pair_tensors(&entries), rank, rank as f64, &["fc"]);
    let before = map_bits(&map);
    merge(&mut map, &dir).expect("well-formed deltas must merge");
    let pairs = pair_tensors(&entries);
    let mut any_changed = false;
    for (m, _, _) in &entries {
        let key = format!("{m}.weight");
        let a = &pairs[&format!("base_model.model.{m}.lora_A.weight")];
        let b = &pairs[&format!("base_model.model.{m}.lora_B.weight")];
        let w_before = bf16_cpu_from_bits(&before[&key], map[&key].dims());
        let expected = naive_expected_bits(&w_before, a, b, 1.0);
        assert_eq!(
            bits(&map[&key]),
            expected,
            "{m}: merged weight must equal bf16(f32(w) + B*A) with exactly one final rounding"
        );
        any_changed |= expected != before[&key];
    }
    assert!(any_changed, "the nonzero fixture must actually move some weight");
    let _ = std::fs::remove_dir_all(dir);
}

fn bf16_cpu_from_bits(raw: &[u16], dims: &[usize]) -> Tensor {
    let vals: Vec<f32> = raw.iter().map(|b| bf16::from_bits(*b).to_f32()).collect();
    bf16_cpu(&vals, dims)
}

#[test]
fn alpha_unequal_r_scales_the_delta_in_f32_after_the_matmul_never_folding_into_bf16_b() {
    let mut map = tiny_map(0x5eed_0021);
    let rank = RANK_1_KEEPS_THE_NAIVE_REFERENCE_BITWISE_COMPARABLE_NO_ACCUMULATION_ORDER_EXISTS;
    let alpha = 1.5f64;
    let mut r = Lcg::new(0x5eed_0022);
    let (m, out_f, in_f) = TINY_MODULE_DIMS_EXERCISE_GATED_Q_ROWS_AND_THE_2X_HIDDEN_FC_IN[0];
    let a = bf16_cpu(&r.vec(rank * in_f, 0.37), &[rank, in_f]);
    let b = bf16_cpu(&r.vec(out_f * rank, 0.37), &[out_f, rank]);
    let entries = vec![(m, a.clone(), b.clone())];
    let dir = fresh_dir("alpha-ne-r");
    write_speculator_dir(&dir, &pair_tensors(&entries), rank, alpha, &["fc"]);
    let key = format!("{m}.weight");
    let w_before = map[&key].clone();
    merge(&mut map, &dir).expect("alpha != r is a legal artifact");

    let exact = naive_expected_bits(&w_before, &a, &b, alpha as f32);
    let b_folded_lossy: Vec<f32> = f32v(&b)
        .iter()
        .map(|x| bf16::from_f32(x * alpha as f32).to_f32())
        .collect();
    let lossy = naive_expected_bits(
        &w_before,
        &a,
        &bf16_cpu(&b_folded_lossy, &[out_f, rank]),
        1.0,
    );
    assert_ne!(
        exact, lossy,
        "fixture must expose the extra bf16 rounding a LoraAdapter::optimize-style fold adds"
    );
    assert_eq!(
        bits(&map[&key]),
        exact,
        "scaling = lora_alpha/r must multiply the f32 delta after the matmul; folding it into \
         bf16 lora_b first adds a rounding the trainer's f32 math never had"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_speculator_naming_anything_but_the_eight_projections_refuses_without_partial_merge() {
    let rank = 2usize;
    let mut r = Lcg::new(0x5eed_0031);

    let norm_target = fresh_dir("refuse-norm");
    let mut t = HashMap::new();
    t.insert(
        "base_model.model.mtp.norm.lora_A.weight".to_string(),
        bf16_cpu(&r.vec(rank * 8, 0.3), &[rank, 8]),
    );
    t.insert(
        "base_model.model.mtp.norm.lora_B.weight".to_string(),
        bf16_cpu(&r.vec(8 * rank, 0.3), &[8, rank]),
    );
    write_speculator_dir(&norm_target, &t, rank, rank as f64, &["norm"]);
    let mut map = tiny_map(0x5eed_0032);
    let err = merge(&mut map, &norm_target).unwrap_err();
    assert!(
        format!("{err:#}").contains("not one of the 8"),
        "a norm target must refuse, got: {err:#}"
    );

    let embed = fresh_dir("refuse-embed");
    let mut t = HashMap::new();
    t.insert(
        "base_model.model.embed_tokens.lora_embedding_A".to_string(),
        bf16_cpu(&r.vec(rank * 8, 0.3), &[rank, 8]),
    );
    write_speculator_dir(&embed, &t, rank, rank as f64, &["embed_tokens"]);
    let err = merge(&mut map, &embed).unwrap_err();
    assert!(
        format!("{err:#}").contains("never MTP speculator targets"),
        "an embedding-form tensor must refuse, got: {err:#}"
    );

    for d in [norm_target, embed] {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[test]
fn a_shape_mismatch_on_any_module_refuses_the_whole_merge_leaving_the_map_untouched() {
    let rank = 2usize;
    let mut r = Lcg::new(0x5eed_0041);
    let (good_m, good_out, good_in) = TINY_MODULE_DIMS_EXERCISE_GATED_Q_ROWS_AND_THE_2X_HIDDEN_FC_IN[1];
    let misshapen_v_proj_sorts_after_valid_q_proj_so_insert_as_you_go_would_leave_q_proj_merged =
        "mtp.layers.0.self_attn.v_proj";
    let (v_out, _v_in) = (4usize, 8usize);
    let wrong_v_proj_in_16_instead_of_8 = 16usize;
    let entries = vec![
        (
            good_m,
            bf16_cpu(&r.vec(rank * good_in, 0.3), &[rank, good_in]),
            bf16_cpu(&r.vec(good_out * rank, 0.3), &[good_out, rank]),
        ),
        (
            misshapen_v_proj_sorts_after_valid_q_proj_so_insert_as_you_go_would_leave_q_proj_merged,
            bf16_cpu(&r.vec(rank * wrong_v_proj_in_16_instead_of_8, 0.3), &[rank, wrong_v_proj_in_16_instead_of_8]),
            bf16_cpu(&r.vec(v_out * rank, 0.3), &[v_out, rank]),
        ),
    ];
    let dir = fresh_dir("refuse-shape");
    write_speculator_dir(&dir, &pair_tensors(&entries), rank, rank as f64, &["q_proj", "v_proj"]);
    let mut map = tiny_map(0x5eed_0042);
    let before = map_bits(&map);
    let err = merge(&mut map, &dir).unwrap_err();
    assert!(
        format!("{err:#}").contains("trained against a different drafter"),
        "an in-features mismatch must refuse as a mismatched-drafter artifact, got: {err:#}"
    );
    assert_eq!(
        map_bits(&map),
        before,
        "a refused merge must leave every weight untouched, including modules that validated"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn rank_disagreeing_with_the_config_a_missing_pair_half_and_an_empty_pair_all_refuse() {
    let mut r = Lcg::new(0x5eed_0051);
    let (m, out_f, in_f) = TINY_MODULE_DIMS_EXERCISE_GATED_Q_ROWS_AND_THE_2X_HIDDEN_FC_IN[0];
    let mut map = tiny_map(0x5eed_0052);

    let rank_mismatch = fresh_dir("refuse-rank");
    let entries = vec![(
        m,
        bf16_cpu(&r.vec(2 * in_f, 0.3), &[2, in_f]),
        bf16_cpu(&r.vec(out_f * 2, 0.3), &[out_f, 2]),
    )];
    write_speculator_dir(&rank_mismatch, &pair_tensors(&entries), 4, 4.0, &["fc"]);
    let err = merge(&mut map, &rank_mismatch).unwrap_err();
    assert!(
        format!("{err:#}").contains("a rank the config does not claim"),
        "tensor rank must match adapter_config r, got: {err:#}"
    );

    let missing_b = fresh_dir("refuse-missing-b");
    let mut t = HashMap::new();
    t.insert(
        format!("base_model.model.{m}.lora_A.weight"),
        bf16_cpu(&r.vec(2 * in_f, 0.3), &[2, in_f]),
    );
    write_speculator_dir(&missing_b, &t, 2, 2.0, &["fc"]);
    let err = merge(&mut map, &missing_b).unwrap_err();
    assert!(
        format!("{err:#}").contains("missing lora_B"),
        "an A without its B must refuse, got: {err:#}"
    );

    let empty = fresh_dir("refuse-empty");
    write_speculator_dir(&empty, &HashMap::new(), 2, 2.0, &["fc"]);
    let err = merge(&mut map, &empty).unwrap_err();
    assert!(
        format!("{err:#}").contains("holds no lora_A/lora_B"),
        "an empty speculator pair must refuse, got: {err:#}"
    );

    for d in [rank_mismatch, missing_b, empty] {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[cfg(feature = "cuda")]
mod loader_level_cuda {
    use super::*;
    use nv_models::qwen3_5_moe::{LayerType, Qwen3Moe, Qwen3_5DenseConfig};
    use nv_specdecode::qwen38_mtp::{Qwen38DenseMtpHead, MTP_WEIGHTS_FILE_NAME};

    fn tiny_q38_config() -> Qwen3_5DenseConfig {
        let layer_types: Vec<LayerType> = (0..8)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();
        Qwen3_5DenseConfig {
            hidden_size: 128,
            num_hidden_layers: 8,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            head_dim: 64,
            intermediate_size: 192,
            vocab_size: 64,
            max_position_embeddings: 64,
            rope_theta: 10_000_000.0,
            rms_norm_eps: 1e-6,
            partial_rotary_factor: 0.25,
            bos_token_id: None,
            eos_token_id: 1,
            layer_types,
            linear_num_key_heads: 2,
            linear_num_value_heads: 6,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            attn_output_gate: true,
            tie_word_embeddings: false,
        }
    }

    fn norm_stored_minus_one(r: &mut Lcg, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| bf16::from_f32(1.0 + 0.1 * r.next_f32()).to_f32() - 1.0)
            .collect()
    }

    fn write_tiny_trunk(cfg: &Qwen3_5DenseConfig, r: &mut Lcg, dir: &Path) {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let hd = cfg.head_dim;
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let ks = cfg.linear_conv_kernel_dim;
        let n_v = cfg.linear_num_value_heads;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert(
            "model.language_model.embed_tokens.weight".into(),
            bf16_cpu(&r.vec(cfg.vocab_size * hidden, 0.6), &[cfg.vocab_size, hidden]),
        );
        t.insert(
            "model.language_model.norm.weight".into(),
            bf16_cpu(&norm_stored_minus_one(r, hidden), &[hidden]),
        );
        t.insert(
            "lm_head.weight".into(),
            bf16_cpu(&r.vec(cfg.vocab_size * hidden, 0.2), &[cfg.vocab_size, hidden]),
        );
        for (i, lt) in cfg.layer_types.iter().enumerate() {
            let p = format!("model.language_model.layers.{i}");
            t.insert(
                format!("{p}.input_layernorm.weight"),
                bf16_cpu(&norm_stored_minus_one(r, hidden), &[hidden]),
            );
            t.insert(
                format!("{p}.post_attention_layernorm.weight"),
                bf16_cpu(&norm_stored_minus_one(r, hidden), &[hidden]),
            );
            match lt {
                LayerType::LinearAttention => {
                    let q = format!("{p}.linear_attn");
                    t.insert(
                        format!("{q}.in_proj_qkv.weight"),
                        bf16_cpu(&r.vec(conv_dim * hidden, 0.12), &[conv_dim, hidden]),
                    );
                    t.insert(
                        format!("{q}.in_proj_z.weight"),
                        bf16_cpu(&r.vec(value_dim * hidden, 0.12), &[value_dim, hidden]),
                    );
                    t.insert(
                        format!("{q}.in_proj_a.weight"),
                        bf16_cpu(&r.vec(n_v * hidden, 0.12), &[n_v, hidden]),
                    );
                    t.insert(
                        format!("{q}.in_proj_b.weight"),
                        bf16_cpu(&r.vec(n_v * hidden, 0.12), &[n_v, hidden]),
                    );
                    t.insert(
                        format!("{q}.conv1d.weight"),
                        bf16_cpu(&r.vec(conv_dim * ks, 0.4), &[conv_dim, 1, ks]),
                    );
                    t.insert(format!("{q}.A_log"), bf16_cpu(&r.vec(n_v, 0.5), &[n_v]));
                    t.insert(format!("{q}.dt_bias"), bf16_cpu(&r.vec(n_v, 0.5), &[n_v]));
                    t.insert(
                        format!("{q}.norm.weight"),
                        bf16_cpu(
                            &(0..cfg.linear_value_head_dim)
                                .map(|_| bf16::from_f32(1.0 + 0.1 * r.next_f32()).to_f32())
                                .collect::<Vec<f32>>(),
                            &[cfg.linear_value_head_dim],
                        ),
                    );
                    t.insert(
                        format!("{q}.out_proj.weight"),
                        bf16_cpu(&r.vec(hidden * value_dim, 0.12), &[hidden, value_dim]),
                    );
                }
                LayerType::FullAttention => {
                    let a = format!("{p}.self_attn");
                    let q_out = cfg.num_attention_heads * hd * 2;
                    let kv_out = cfg.num_key_value_heads * hd;
                    t.insert(
                        format!("{a}.q_proj.weight"),
                        bf16_cpu(&r.vec(q_out * hidden, 0.12), &[q_out, hidden]),
                    );
                    t.insert(
                        format!("{a}.k_proj.weight"),
                        bf16_cpu(&r.vec(kv_out * hidden, 0.12), &[kv_out, hidden]),
                    );
                    t.insert(
                        format!("{a}.v_proj.weight"),
                        bf16_cpu(&r.vec(kv_out * hidden, 0.12), &[kv_out, hidden]),
                    );
                    t.insert(
                        format!("{a}.o_proj.weight"),
                        bf16_cpu(
                            &r.vec(hidden * cfg.num_attention_heads * hd, 0.12),
                            &[hidden, cfg.num_attention_heads * hd],
                        ),
                    );
                    t.insert(
                        format!("{a}.q_norm.weight"),
                        bf16_cpu(&norm_stored_minus_one(r, hd), &[hd]),
                    );
                    t.insert(
                        format!("{a}.k_norm.weight"),
                        bf16_cpu(&norm_stored_minus_one(r, hd), &[hd]),
                    );
                }
            }
            t.insert(
                format!("{p}.mlp.gate_proj.weight"),
                bf16_cpu(&r.vec(inter * hidden, 0.15), &[inter, hidden]),
            );
            t.insert(
                format!("{p}.mlp.up_proj.weight"),
                bf16_cpu(&r.vec(inter * hidden, 0.15), &[inter, hidden]),
            );
            t.insert(
                format!("{p}.mlp.down_proj.weight"),
                bf16_cpu(&r.vec(hidden * inter, 0.15), &[hidden, inter]),
            );
        }
        candle_core::safetensors::save(&t, dir.join("model.safetensors"))
            .expect("save tiny trunk");
    }

    fn write_tiny_mtp_weights(base: &Qwen3Moe, r: &mut Lcg, dir: &Path) {
        let g = Qwen38DenseMtpHead::geometry_from_dense_base(base).expect("geometry");
        let mut m: HashMap<String, Tensor> = HashMap::new();
        for (name, shape) in g.expected_tensor_shapes() {
            let n: usize = shape.iter().product();
            let vals = if name.ends_with("norm.weight")
                || name.contains("norm_embedding")
                || name.contains("norm_hidden")
            {
                norm_stored_minus_one(r, n)
            } else {
                r.vec(n, 0.12)
            };
            m.insert(name, bf16_cpu(&vals, &shape));
        }
        candle_core::safetensors::save(&m, dir.join(MTP_WEIGHTS_FILE_NAME))
            .expect("save tiny mtp head");
    }

    fn draft_logit_bits(
        head: &Qwen38DenseMtpHead,
        base: &Qwen3Moe,
        device: &Device,
    ) -> Vec<u32> {
        let mut cache = head.new_kv_cache(8, device).expect("kv cache");
        let hidden = Tensor::ones(
            (1usize, 1usize, base.config().hidden_size),
            base.dtype(),
            device,
        )
        .expect("anchor hidden");
        let (logits, _h) = head
            .forward_draft(base, &hidden, 3, &mut cache)
            .expect("draft step");
        f32v(&logits).iter().map(|v| v.to_bits()).collect()
    }

    fn mtp_speculator_entries(
        base: &Qwen3Moe,
        rank: usize,
        r: &mut Lcg,
        b_scale: f32,
    ) -> Vec<(String, Tensor, Tensor)> {
        let shapes = Qwen38DenseMtpHead::geometry_from_dense_base(base)
            .expect("geometry")
            .expected_tensor_shapes();
        MTP_SPECULATOR_LORA_MODULES_MATCH_Q38_MTP_TRAIN_HEAD_PROJECTIONS_NORMS_EMBED_LM_HEAD_D2T_T2D_ARE_NEVER_TARGETS
            .iter()
            .map(|m| {
                let shape = &shapes[&format!("{m}.weight")];
                let (out_f, in_f) = (shape[0], shape[1]);
                let b = if b_scale == 0.0 {
                    Tensor::zeros((out_f, rank), DType::BF16, &Device::Cpu).unwrap()
                } else {
                    bf16_cpu(&r.vec(out_f * rank, b_scale), &[out_f, rank])
                };
                (
                    m.to_string(),
                    bf16_cpu(&r.vec(rank * in_f, 0.3), &[rank, in_f]),
                    b,
                )
            })
            .collect()
    }

    fn write_artifact(tag: &str, kind: &str, entries: &[(String, Tensor, Tensor)], rank: usize) -> PathBuf {
        let dir = fresh_dir(tag);
        write_main_adapter_config(&dir, Some(kind));
        let sub = dir.join(SPECULATOR_SUBDIR_MIRRORS_NV_TRAIN_SPECULATOR_SUBDIR);
        let borrowed: Vec<(&str, Tensor, Tensor)> = entries
            .iter()
            .map(|(m, a, b)| (m.as_str(), a.clone(), b.clone()))
            .collect();
        write_speculator_dir(&sub, &pair_tensors(&borrowed), rank, rank as f64, &["fc"]);
        dir
    }

    #[test]
    fn loader_merges_speculator_deltas_and_a_plain_artifact_stays_byte_identical() {
        let Ok(device) = Device::new_cuda_with_stream(0) else {
            eprintln!("[skip] no cuda device; the speculator loader suite needs the card");
            return;
        };
        let cfg = tiny_q38_config();
        let mut r = Lcg::new(0x5eed_c0da);
        let ckpt = fresh_dir("loader-ckpt");
        write_tiny_trunk(&cfg, &mut r, &ckpt);
        let weights = nv_weights::WeightLoader::open_dir(&ckpt, &device).expect("open tiny dir");
        let base = Qwen3Moe::from_loader_dense(cfg, &weights, &device).expect("tiny cuda trunk");
        drop(weights);
        write_tiny_mtp_weights(&base, &mut r, &ckpt);

        let plain = Qwen38DenseMtpHead::from_checkpoint(None, &ckpt, &base, &device)
            .expect("plain head");
        let plain_bits = draft_logit_bits(&plain, &base, &device);

        let none_head = Qwen38DenseMtpHead::from_checkpoint_with_speculator(
            None, &ckpt, None, &base, &device,
        )
        .expect("None adapter dir");
        assert_eq!(
            draft_logit_bits(&none_head, &base, &device),
            plain_bits,
            "a None adapter dir must load bitwise-identically to from_checkpoint"
        );

        let adapter_without_spec = fresh_dir("loader-plain-adapter");
        write_main_adapter_config(&adapter_without_spec, None);
        let no_spec = Qwen38DenseMtpHead::from_checkpoint_with_speculator(
            None,
            &ckpt,
            Some(&adapter_without_spec),
            &base,
            &device,
        )
        .expect("adapter without speculator/");
        assert_eq!(
            draft_logit_bits(&no_spec, &base, &device),
            plain_bits,
            "an artifact without speculator/ must behave byte-identically to today"
        );

        let rank = 2usize;
        let zero = write_artifact(
            "loader-zero",
            nv_specdecode::qwen38_mtp::QWEN38_MTP_SPECULATOR_KIND_MIRRORS_NV_TRAIN,
            &mtp_speculator_entries(&base, rank, &mut r, 0.0),
            rank,
        );
        let zero_head = Qwen38DenseMtpHead::from_checkpoint_with_speculator(
            None, &ckpt, Some(&zero), &base, &device,
        )
        .expect("zero-delta speculator");
        assert_eq!(
            draft_logit_bits(&zero_head, &base, &device),
            plain_bits,
            "zero deltas must merge to a bitwise-identical drafter"
        );

        let trained = write_artifact(
            "loader-trained",
            nv_specdecode::qwen38_mtp::QWEN38_MTP_SPECULATOR_KIND_MIRRORS_NV_TRAIN,
            &mtp_speculator_entries(&base, rank, &mut r, 0.3),
            rank,
        );
        let trained_head = Qwen38DenseMtpHead::from_checkpoint_with_speculator(
            None, &ckpt, Some(&trained), &base, &device,
        )
        .expect("trained speculator merges at load");
        assert_ne!(
            draft_logit_bits(&trained_head, &base, &device),
            plain_bits,
            "nonzero speculator deltas must change the drafter's function"
        );

        let eagle = write_artifact(
            "loader-eagle3",
            "eagle3",
            &mtp_speculator_entries(&base, rank, &mut r, 0.3),
            rank,
        );
        let err = match Qwen38DenseMtpHead::from_checkpoint_with_speculator(
            None, &ckpt, Some(&eagle), &base, &device,
        ) {
            Ok(_) => panic!("an eagle3-kind artifact must not load on the MTP surface"),
            Err(e) => e,
        };
        assert!(
            format!("{err:#}").contains("eagle3"),
            "an eagle3-kind artifact must refuse on the MTP loader, got: {err:#}"
        );

        let mut mismatched_entries = mtp_speculator_entries(&base, rank, &mut r, 0.3);
        let h = base.config().hidden_size;
        let v_proj_sorts_last_in_the_merge_btreemap_so_every_other_module_validates_first =
            mismatched_entries
                .iter()
                .position(|(m, _, _)| m == "mtp.layers.0.self_attn.v_proj")
                .expect("the allowlist fixture must carry v_proj");
        mismatched_entries[v_proj_sorts_last_in_the_merge_btreemap_so_every_other_module_validates_first]
            .1 = bf16_cpu(&r.vec(rank * (h + 1), 0.3), &[rank, h + 1]);
        let mismatched = write_artifact(
            "loader-mismatch",
            nv_specdecode::qwen38_mtp::QWEN38_MTP_SPECULATOR_KIND_MIRRORS_NV_TRAIN,
            &mismatched_entries,
            rank,
        );
        let err = match Qwen38DenseMtpHead::from_checkpoint_with_speculator(
            None, &ckpt, Some(&mismatched), &base, &device,
        ) {
            Ok(_) => panic!("misshapen speculator deltas must not load"),
            Err(e) => e,
        };
        assert!(
            format!("{err:#}").contains("trained against a different drafter"),
            "misshapen deltas must refuse the whole load, got: {err:#}"
        );

        for d in [ckpt, adapter_without_spec, zero, trained, eagle, mismatched] {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}
