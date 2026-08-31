use candle_core::{DType, Device, Tensor, D};
use nv_train::qwen38::{
    build_rope, flat_positions, run_jsonl_with_mtp_cotrain, HostDequantWeights, Q38Config,
    Q38JsonlArgs, Q38MtpCoTrainArgs, Q38MtpTrainHead, TrainLayer, EMBED_TOKENS, FINAL_NORM,
    LM_HEAD, MTP_WEIGHTS_FILE_NAME,
};
use nv_train::qwen38_fixture::{write_tiny_checkpoint, FixtureLcg};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

mod mtp_fixture;

const THIS_REFERENCE_IS_TRANSCRIBED_FROM_THE_SERVING_DRAFTER_NOT_FROM_THE_TRAINING_HEAD: &str =
    "Every convention below is copied from nv-specdecode/src/qwen38_mtp.rs (\
     layer_input_from_token_ids_already_on_device cats the normed embedding FIRST, \
     prefill_prompt_kv_shifted_by_one_with_position_zero_on_a_zero_hidden builds \
     cat(zero, trunk_hidden[..seq-1])) and nv-models/src/qwen3_5_moe.rs project_qkv \
     (q splits value-first gate-second, q/k norm before the partial rope, sigmoid gate \
     after sdpa), using only raw safetensors bytes and plain candle ops. It must NOT be \
     rewritten to call Q38MtpTrainHead, TrainAttn, RmsNorm or Rope: sharing code with \
     the function under test would turn this suite back into the self-consistency check \
     the serving-parity review flagged as unable to catch a swapped cat, a wrong-direction \
     shift, a dropped zero row, or a post-final-norm tap.";

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nv-train-mtporacle-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct RawWeights(HashMap<String, Tensor>);

impl RawWeights {
    fn load(base: &Path) -> Self {
        let mut all = candle_core::safetensors::load(base.join("model.safetensors"), &Device::Cpu)
            .unwrap();
        all.extend(
            candle_core::safetensors::load(base.join(MTP_WEIGHTS_FILE_NAME), &Device::Cpu)
                .unwrap(),
        );
        Self(all)
    }

    fn f32(&self, name: &str) -> Tensor {
        self.0
            .get(name)
            .unwrap_or_else(|| panic!("{name} missing from the fixture checkpoint"))
            .to_dtype(DType::F32)
            .unwrap()
    }

    fn gamma_stored_zero_centered(&self, name: &str) -> Tensor {
        self.f32(name).affine(1.0, 1.0).unwrap()
    }
}

const RMS_EPS: f64 = 1e-6;

fn rms_ref(x: &Tensor, gamma: &Tensor) -> Tensor {
    let mean_sq = x.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
    let eps = Tensor::new(RMS_EPS as f32, x.device()).unwrap();
    let denom = mean_sq.broadcast_add(&eps).unwrap().sqrt().unwrap();
    x.broadcast_div(&denom)
        .unwrap()
        .broadcast_mul(gamma)
        .unwrap()
}

fn matmul_wt(x: &Tensor, w: &Tensor) -> Tensor {
    x.matmul(&w.t().unwrap().contiguous().unwrap()).unwrap()
}

fn sigmoid_ref(x: &Tensor) -> Tensor {
    x.affine(-1.0, 0.0)
        .unwrap()
        .exp()
        .unwrap()
        .affine(1.0, 1.0)
        .unwrap()
        .recip()
        .unwrap()
}

fn silu_ref(x: &Tensor) -> Tensor {
    x.mul(&sigmoid_ref(x)).unwrap()
}

fn softmax_ref(scores: &Tensor) -> Tensor {
    let m = scores.max_keepdim(D::Minus1).unwrap();
    let e = scores.broadcast_sub(&m).unwrap().exp().unwrap();
    let s = e.sum_keepdim(D::Minus1).unwrap();
    e.broadcast_div(&s).unwrap()
}

fn rope_tables_ref(theta: f32, rotary_dim: usize, t: usize) -> (Tensor, Tensor) {
    let half = rotary_dim / 2;
    let mut cos = vec![0f32; t * half];
    let mut sin = vec![0f32; t * half];
    for p in 0..t {
        for i in 0..half {
            let inv_freq = 1.0 / theta.powf((i as f32 * 2.0) / (rotary_dim as f32));
            let angle = p as f32 * inv_freq;
            cos[p * half + i] = angle.cos();
            sin[p * half + i] = angle.sin();
        }
    }
    (
        Tensor::from_vec(cos, (t, 1, half), &Device::Cpu).unwrap(),
        Tensor::from_vec(sin, (t, 1, half), &Device::Cpu).unwrap(),
    )
}

fn rope_ref(x_t_heads_rd: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let rd = x_t_heads_rd.dims()[2];
    let half = rd / 2;
    let lo = x_t_heads_rd.narrow(2, 0, half).unwrap();
    let hi = x_t_heads_rd.narrow(2, half, half).unwrap();
    let out_lo = lo
        .broadcast_mul(cos)
        .unwrap()
        .sub(&hi.broadcast_mul(sin).unwrap())
        .unwrap();
    let out_hi = lo
        .broadcast_mul(sin)
        .unwrap()
        .add(&hi.broadcast_mul(cos).unwrap())
        .unwrap();
    Tensor::cat(&[&out_lo, &out_hi], 2).unwrap()
}

fn attention_ref(raw: &RawWeights, cfg: &Q38Config, x_t_h: &Tensor) -> Tensor {
    let t = x_t_h.dims()[0];
    let hd = cfg.head_dim;
    let heads = cfg.num_attention_heads;
    let kv_heads = cfg.num_key_value_heads;
    let rd = cfg.rotary_dim();
    assert!(
        cfg.attn_output_gate && rd < hd,
        "the fixture must exercise the gated-attention and partial-rotary serving paths, or \
         this oracle stops covering them"
    );
    let p = "mtp.layers.0.self_attn";
    let q_raw = matmul_wt(x_t_h, &raw.f32(&format!("{p}.q_proj.weight")))
        .reshape((t, heads, 2 * hd))
        .unwrap();
    let q_value = q_raw.narrow(2, 0, hd).unwrap().contiguous().unwrap();
    let q_gate = q_raw.narrow(2, hd, hd).unwrap().contiguous().unwrap();
    let k = matmul_wt(x_t_h, &raw.f32(&format!("{p}.k_proj.weight")))
        .reshape((t, kv_heads, hd))
        .unwrap();
    let v = matmul_wt(x_t_h, &raw.f32(&format!("{p}.v_proj.weight")))
        .reshape((t, kv_heads, hd))
        .unwrap();
    let qn = rms_ref(
        &q_value,
        &raw.gamma_stored_zero_centered(&format!("{p}.q_norm.weight")),
    );
    let kn = rms_ref(
        &k,
        &raw.gamma_stored_zero_centered(&format!("{p}.k_norm.weight")),
    );
    let (cos, sin) = rope_tables_ref(cfg.rope_theta, rd, t);
    let roll = |x: &Tensor| -> Tensor {
        let rot = rope_ref(&x.narrow(2, 0, rd).unwrap().contiguous().unwrap(), &cos, &sin);
        let pass = x.narrow(2, rd, hd - rd).unwrap();
        Tensor::cat(&[&rot, &pass], 2).unwrap()
    };
    let q_final = roll(&qn);
    let k_final = roll(&kn);
    let mut causal = vec![0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if j > i {
                causal[i * t + j] = f32::NEG_INFINITY;
            }
        }
    }
    let causal = Tensor::from_vec(causal, (t, t), &Device::Cpu).unwrap();
    let scale = 1.0 / (hd as f64).sqrt();
    let mut per_head: Vec<Tensor> = Vec::with_capacity(heads);
    for h in 0..heads {
        let kv = h / (heads / kv_heads);
        let qh = q_final
            .narrow(1, h, 1)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((t, hd))
            .unwrap();
        let kh = k_final
            .narrow(1, kv, 1)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((t, hd))
            .unwrap();
        let vh = v
            .narrow(1, kv, 1)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((t, hd))
            .unwrap();
        let scores = qh
            .matmul(&kh.t().unwrap().contiguous().unwrap())
            .unwrap()
            .affine(scale, 0.0)
            .unwrap()
            .add(&causal)
            .unwrap();
        per_head.push(
            softmax_ref(&scores)
                .matmul(&vh)
                .unwrap()
                .reshape((t, 1, hd))
                .unwrap(),
        );
    }
    let concat = Tensor::cat(&per_head, 1)
        .unwrap()
        .reshape((t, heads * hd))
        .unwrap();
    let gate = sigmoid_ref(&q_gate.reshape((t, heads * hd)).unwrap());
    matmul_wt(
        &concat.mul(&gate).unwrap(),
        &raw.f32(&format!("{p}.o_proj.weight")),
    )
}

fn mlp_ref(raw: &RawWeights, x_t_h: &Tensor) -> Tensor {
    let p = "mtp.layers.0.mlp";
    let gate = silu_ref(&matmul_wt(x_t_h, &raw.f32(&format!("{p}.gate_proj.weight"))));
    let up = matmul_wt(x_t_h, &raw.f32(&format!("{p}.up_proj.weight")));
    matmul_wt(
        &gate.mul(&up).unwrap(),
        &raw.f32(&format!("{p}.down_proj.weight")),
    )
}

fn ce_ref(logits_t_vocab: &Tensor, tgt: &[u32], mask: &[f32]) -> f32 {
    let t = tgt.len();
    let m = logits_t_vocab.max_keepdim(D::Minus1).unwrap();
    let lse = logits_t_vocab
        .broadcast_sub(&m)
        .unwrap()
        .exp()
        .unwrap()
        .sum_keepdim(D::Minus1)
        .unwrap()
        .log()
        .unwrap()
        .add(&m)
        .unwrap()
        .reshape(t)
        .unwrap();
    let tgt_t = Tensor::from_vec(tgt.to_vec(), (t, 1), &Device::Cpu).unwrap();
    let picked = logits_t_vocab
        .gather(&tgt_t, 1)
        .unwrap()
        .reshape(t)
        .unwrap();
    let per_pos = lse.sub(&picked).unwrap().to_vec1::<f32>().unwrap();
    let weight_sum: f32 = mask.iter().sum();
    per_pos
        .iter()
        .zip(mask)
        .map(|(l, w)| l * w)
        .sum::<f32>()
        / weight_sum
}

fn serving_shift_ref(hidden_t_h: &Tensor) -> Tensor {
    let (t, h) = hidden_t_h.dims2().unwrap();
    let zero = Tensor::zeros((1, h), DType::F32, &Device::Cpu).unwrap();
    Tensor::cat(&[&zero, &hidden_t_h.narrow(0, 0, t - 1).unwrap()], 0)
        .unwrap()
        .contiguous()
        .unwrap()
}

fn future_shift_ref(hidden_t_h: &Tensor) -> Tensor {
    let (t, h) = hidden_t_h.dims2().unwrap();
    let zero = Tensor::zeros((1, h), DType::F32, &Device::Cpu).unwrap();
    Tensor::cat(&[&hidden_t_h.narrow(0, 1, t - 1).unwrap(), &zero], 0)
        .unwrap()
        .contiguous()
        .unwrap()
}

fn reference_head_ce(
    raw: &RawWeights,
    cfg: &Q38Config,
    ids: &[u32],
    shifted_hidden_t_h: &Tensor,
    tgt: &[u32],
    mask: &[f32],
    cat_hidden_first_which_is_the_mutation_not_serving: bool,
) -> f32 {
    let _ = THIS_REFERENCE_IS_TRANSCRIBED_FROM_THE_SERVING_DRAFTER_NOT_FROM_THE_TRAINING_HEAD;
    let t = ids.len();
    let ids_t = Tensor::from_vec(ids.to_vec(), t, &Device::Cpu).unwrap();
    let emb = raw.f32(EMBED_TOKENS).index_select(&ids_t, 0).unwrap();
    let norm_emb = rms_ref(
        &emb,
        &raw.gamma_stored_zero_centered("mtp.pre_fc_norm_embedding.weight"),
    );
    let norm_hid = rms_ref(
        shifted_hidden_t_h,
        &raw.gamma_stored_zero_centered("mtp.pre_fc_norm_hidden.weight"),
    );
    let fused = if cat_hidden_first_which_is_the_mutation_not_serving {
        Tensor::cat(&[&norm_hid, &norm_emb], 1).unwrap()
    } else {
        Tensor::cat(&[&norm_emb, &norm_hid], 1).unwrap()
    };
    let h0 = matmul_wt(&fused.contiguous().unwrap(), &raw.f32("mtp.fc.weight"));
    let normed = rms_ref(
        &h0,
        &raw.gamma_stored_zero_centered("mtp.layers.0.input_layernorm.weight"),
    );
    let x = h0.add(&attention_ref(raw, cfg, &normed)).unwrap();
    let normed_post = rms_ref(
        &x,
        &raw.gamma_stored_zero_centered("mtp.layers.0.post_attention_layernorm.weight"),
    );
    let x = x.add(&mlp_ref(raw, &normed_post)).unwrap();
    let normed_final = rms_ref(&x, &raw.gamma_stored_zero_centered("mtp.norm.weight"));
    let logits = matmul_wt(&normed_final, &raw.f32("lm_head.weight"));
    ce_ref(&logits, tgt, mask)
}

const EQUALITY_TOL_REL: f32 = 1e-4;
const MUTATION_MUST_MOVE_THE_CE_BY_AT_LEAST: f32 = 1e-3;

fn rel_diff(a: f32, b: f32) -> f32 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-12)
}

#[test]
fn the_head_ce_equals_an_independent_replica_of_the_serving_drafter_and_each_named_mutation_moves_it(
) {
    let root = scratch("unit");
    let base = root.join("base");
    let cfg = write_tiny_checkpoint(&base);
    mtp_fixture::write_tiny_mtp_head(&base, &cfg);
    let device = Device::Cpu;

    let t = 9usize;
    let ids_full: Vec<u32> = (0..=t as u32)
        .map(|i| (i * 7 + 3) % cfg.vocab_size as u32)
        .collect();
    let ids = &ids_full[..t];
    let tgt = &ids_full[1..];
    let mut mask = vec![1.0f32; t];
    mask[0] = 0.0;
    mask[1] = 0.0;
    let trunk_hidden = FixtureLcg(0xB0A7_5EED).tensor(&[t, cfg.hidden_size], 1.0);

    let src = HostDequantWeights::open_dir(&base, &device).unwrap();
    let head = Q38MtpTrainHead::from_checkpoint(&base, &cfg, DType::F32, &device).unwrap();
    let lm_head = src.dense_linear(LM_HEAD, DType::F32).unwrap();
    let rope = build_rope(&cfg, t, &device).unwrap();
    let positions = flat_positions(1, t, &device).unwrap();
    let emb = src
        .embed_rows(EMBED_TOKENS, ids, cfg.hidden_size)
        .unwrap()
        .reshape((1, t, cfg.hidden_size))
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let tgt_t = Tensor::from_vec(tgt.to_vec(), t, &device).unwrap();
    let got = head
        .shifted_draft_ce(
            &emb,
            &trunk_hidden.reshape((1, t, cfg.hidden_size)).unwrap(),
            &rope,
            &positions,
            &lm_head,
            &tgt_t,
            &mask,
            cfg.vocab_size,
        )
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    let raw = RawWeights::load(&base);
    let serving = reference_head_ce(
        &raw,
        &cfg,
        ids,
        &serving_shift_ref(&trunk_hidden),
        tgt,
        &mask,
        false,
    );
    assert!(
        rel_diff(got, serving) < EQUALITY_TOL_REL,
        "shifted_draft_ce diverged from the from-scratch serving-drafter replica: trainer {got} \
         vs replica {serving}; the head is training a function NV_DRAFTER=mtp does not serve"
    );

    let hidden_first = reference_head_ce(
        &raw,
        &cfg,
        ids,
        &serving_shift_ref(&trunk_hidden),
        tgt,
        &mask,
        true,
    );
    let unshifted_no_zero_row = reference_head_ce(
        &raw, &cfg, ids, &trunk_hidden, tgt, &mask, false,
    );
    let shifted_toward_the_future = reference_head_ce(
        &raw,
        &cfg,
        ids,
        &future_shift_ref(&trunk_hidden),
        tgt,
        &mask,
        false,
    );
    for (mutation, wrong) in [
        ("fc cat order swapped to hidden-first", hidden_first),
        ("zero-hidden row dropped (unshifted tap)", unshifted_no_zero_row),
        ("hidden shifted the wrong direction", shifted_toward_the_future),
    ] {
        assert!(
            (wrong - serving).abs() > MUTATION_MUST_MOVE_THE_CE_BY_AT_LEAST,
            "{mutation} leaves the fixture CE at {wrong} vs serving {serving}, so the equality \
             assertion above is vacuous for that mutation and the fixture must be re-seeded"
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_cotrain_arm_taps_the_pre_final_norm_trunk_hidden_the_serving_drafter_consumes() {
    let root = scratch("e2e");
    let base = root.join("base");
    let cfg = write_tiny_checkpoint(&base);
    mtp_fixture::write_tiny_mtp_head(&base, &cfg);
    let device = Device::Cpu;

    let data = root.join("train.jsonl");
    std::fs::write(
        &data,
        format!(
            "{}\n{}\n",
            serde_json::json!({ "text": "planning and marksmanship in a single turn" }),
            serde_json::json!({ "text": "the score was praised, the story was not" })
        ),
    )
    .unwrap();

    let args = Q38JsonlArgs {
        base: base.clone(),
        data: data.clone(),
        eval_data: None,
        out: root.join("adapter"),
        grad_layers: 2,
        rank: 4,
        alpha: 4.0,
        targets: vec!["q_proj".into(), "v_proj".into()],
        steps: 1,
        lr: 5e-3,
        b_lr_ratio: 1.0,
        seed: 19,
    };
    let (_summary, co) = run_jsonl_with_mtp_cotrain(
        &args,
        &Q38MtpCoTrainArgs {
            rank: 4,
            alpha: 4.0,
            speculator_lr_ratio: 4.0,
        },
        &device,
    )
    .unwrap();

    let rows = nv_train::load_jsonl_token_dataset(
        &data,
        &base,
        cfg.vocab_size,
        nv_train::JSONL_MAX_SEQ_IS_512_SHARED_BY_BOTH_ENDPOINT_ARMS_BECAUSE_THE_DEFAULT_QWEN_WINDOWS_GDN_GRAPH_AT_512_IS_ABOUT_14_GB,
    )
    .unwrap();
    let src = HostDequantWeights::open_dir(&base, &device).unwrap();
    let raw = RawWeights::load(&base);
    let max_t = rows.iter().map(|r| r.ids.len() - 1).max().unwrap();
    let rope = build_rope(&cfg, max_t, &device).unwrap();
    let layers: Vec<TrainLayer> = (0..cfg.num_hidden_layers)
        .map(|li| TrainLayer::from_weights(&src, &cfg, li, DType::F32).unwrap())
        .collect();

    let mut expected_sum = 0f32;
    let mut wrong_tap_sum = 0f32;
    for row in &rows {
        let t = row.ids.len() - 1;
        let ids = &row.ids[..t];
        let tgt = &row.ids[1..];
        let mask = row.target_mask_where_predicting_a_prompt_token_weighs_zero();
        let positions = flat_positions(1, t, &device).unwrap();
        let mut x = src
            .embed_rows(EMBED_TOKENS, ids, cfg.hidden_size)
            .unwrap()
            .reshape((1, t, cfg.hidden_size))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        for layer in &layers {
            x = layer.forward(&x, &rope, &positions).unwrap();
        }
        let pre_final_norm = x.reshape((t, cfg.hidden_size)).unwrap();
        let post_final_norm = rms_ref(
            &pre_final_norm,
            &raw.gamma_stored_zero_centered(FINAL_NORM),
        );
        expected_sum += reference_head_ce(
            &raw,
            &cfg,
            ids,
            &serving_shift_ref(&pre_final_norm),
            tgt,
            &mask,
            false,
        );
        wrong_tap_sum += reference_head_ce(
            &raw,
            &cfg,
            ids,
            &serving_shift_ref(&post_final_norm),
            tgt,
            &mask,
            false,
        );
    }
    let inv = 1.0 / rows.len() as f32;
    let expected = expected_sum * inv;
    let wrong_tap = wrong_tap_sum * inv;

    assert!(
        rel_diff(co.head_losses[0], expected) < EQUALITY_TOL_REL,
        "at step 0 every LoRA B is zero, so the co-train head loss must equal the serving \
         replica scored on the base trunk's PRE-final-norm output: got {} vs replica {expected}; \
         the trainer is handing the head a hidden the serving drafter never sees",
        co.head_losses[0]
    );
    assert!(
        (wrong_tap - expected).abs() > MUTATION_MUST_MOVE_THE_CE_BY_AT_LEAST,
        "the post-final-norm tap scores {wrong_tap} vs pre-final-norm {expected} on this \
         fixture, so the tap assertion above is vacuous and the fixture must be re-seeded"
    );
    let _ = std::fs::remove_dir_all(&root);
}
