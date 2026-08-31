use candle_core::{DType, Device, Tensor, Var};
use nv_layers::linear::Linear;
use nv_layers::LinearAttention;
use nv_train::qwen38::{
    build_rope, gdn_scan, layer_prefix, HostDequantWeights, Q38Config, Q38JsonlArgs, Q38TrainArgs,
    TrainGdn, FINAL_NORM,
};
use nv_train::qwen38_fixture::{tiny_config_json, write_tiny_checkpoint, FixtureLcg as Lcg};
use std::path::{Path, PathBuf};

const REAL_CORPUS_FILE: &str = "/home/user/tmp/wf/q38/xppl/wikitext-2-raw/wiki.train.raw";

const THIS_PROSE_IS_A_REAL_CORPUS_SO_A_RUN_WITHOUT_WIKITEXT_STILL_TRAINS_ON_ENGLISH_AND_NOT_ON_NOISE: &str =
    "The Valkyria series began as a tactical role playing franchise for home consoles, and the \
     third entry was released only in Japan. Its battle system alternates a top down command map \
     with a third person action phase, so a single turn mixes planning and marksmanship. Critics \
     praised the art direction and the score, while faulting a story that reuses the structure of \
     the first game. A localisation was never announced, and fan translations circulated instead. \
     The engine reuses assets from the second entry, which shortened development and narrowed the \
     visual range of the maps. Later ports added a training mode and a gallery of illustrations.";

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nv-train-q38-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn corpus_file(dir: &Path) -> PathBuf {
    let real = PathBuf::from(REAL_CORPUS_FILE);
    if real.is_file() {
        return real;
    }
    let p = dir.join("corpus.txt");
    let mut body = String::new();
    while body.len() < 8192 {
        body.push_str(
            THIS_PROSE_IS_A_REAL_CORPUS_SO_A_RUN_WITHOUT_WIKITEXT_STILL_TRAINS_ON_ENGLISH_AND_NOT_ON_NOISE,
        );
        body.push(' ');
    }
    std::fs::write(&p, body).unwrap();
    p
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.to_dtype(DType::F32).unwrap();
    let b = b.to_dtype(DType::F32).unwrap();
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

fn amax(a: &Tensor) -> f32 {
    a.to_dtype(DType::F32)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

#[test]
fn the_trainable_gdn_forward_equals_the_serving_linear_attention_on_the_same_weights() {
    let dev = Device::Cpu;
    let cfg = nv_layers::LinearAttentionConfig {
        hidden_size: 16,
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 4,
        mamba_ssm_dtype: DType::F32,
        rms_eps: 1e-6,
    };
    let mut rng = Lcg(0xA11CE);
    let h = cfg.hidden_size;
    let n_v = cfg.linear_num_value_heads;
    let w_qkv = rng.tensor(&[cfg.conv_dim(), h], 0.2);
    let w_z = rng.tensor(&[cfg.value_dim(), h], 0.2);
    let w_a = rng.tensor(&[n_v, h], 0.2);
    let w_b = rng.tensor(&[n_v, h], 0.2);
    let conv = rng.tensor(&[cfg.conv_dim(), 1, cfg.linear_conv_kernel_dim], 0.4);
    let a_log = rng.tensor(&[n_v], 0.3);
    let dt_bias = rng.tensor(&[n_v], 0.3);
    let norm_w = rng.tensor(&[cfg.linear_value_head_dim], 0.1).affine(1.0, 1.0).unwrap();
    let w_out = rng.tensor(&[h, cfg.value_dim()], 0.2);

    let lin = |w: &Tensor| Linear::new(w.clone(), None).unwrap();
    let serving = LinearAttention::new(
        cfg,
        lin(&w_qkv),
        lin(&w_z),
        lin(&w_a),
        lin(&w_b),
        conv.clone(),
        a_log.clone(),
        dt_bias.clone(),
        norm_w.clone(),
        lin(&w_out),
    )
    .unwrap();
    let trainable = TrainGdn::new(
        cfg,
        lin(&w_qkv),
        lin(&w_z),
        lin(&w_a),
        lin(&w_b),
        conv,
        a_log,
        dt_bias,
        norm_w,
        lin(&w_out),
    );

    let x = rng.tensor(&[1, 7, h], 0.5).to_device(&dev).unwrap();
    let want = serving.forward(&x).unwrap();
    let got = trainable.forward(&x).unwrap();
    assert_eq!(want.dims(), got.dims());
    let scale = amax(&want).max(1e-6);
    let d = max_abs_diff(&want, &got);
    assert!(
        d / scale < 1e-4,
        "the differentiable GDN forward is not the served function: max abs diff {d} against a \
         reference amax of {scale}. Anything trained through it would optimise a different model \
         than the one nvk-lora-merge later serves."
    );
}

#[test]
fn the_trainer_runs_the_served_sdpa_itself_and_it_reaches_q_and_k_with_a_gradient() {
    let (b, s, h, hkv, d) = (1usize, 6usize, 2usize, 1usize, 8usize);
    let mut rng = Lcg(0xF00D);
    let q0 = rng.tensor(&[b, s, h, d], 0.5);
    let k0 = rng.tensor(&[b, s, hkv, d], 0.5);
    let v0 = rng.tensor(&[b, s, hkv, d], 0.5);
    let scale = 1.0 / (d as f32).sqrt();
    let cfg = || nv_layers::attn::AttnConfig {
        num_heads: h,
        num_kv_heads: hkv,
        head_dim: d,
        softmax_scale: scale,
        causal: true,
    };

    let served = nv_layers::attn::sdpa(&q0, &k0, &v0, &cfg()).unwrap();

    let qv = Var::from_tensor(&q0).unwrap();
    let kv = Var::from_tensor(&k0).unwrap();
    let vv = Var::from_tensor(&v0).unwrap();
    let out =
        nv_layers::attn::sdpa(qv.as_tensor(), kv.as_tensor(), vv.as_tensor(), &cfg()).unwrap();
    assert!(
        max_abs_diff(&served, &out) < 1e-5,
        "the tracked sdpa forward must stay the served function; max abs diff {}",
        max_abs_diff(&served, &out)
    );

    let g = out.sum_all().unwrap().backward().unwrap();
    for (label, var) in [("q", &qv), ("k", &kv), ("v", &vv)] {
        let grad = g.get(var).unwrap_or_else(|| {
            panic!(
                "{label} received no gradient entry at all from nv_layers::attn::sdpa. That was \
                 this file's original defect -- candle_nn::ops::softmax_last_dim is installed \
                 with apply_op1_no_bwd, so its output is a graph leaf and q/k train nowhere while \
                 v still does -- and it is what nv_layers::attn::\
                 softmax_last_dim_that_keeps_q_and_k_on_the_graph fixes. The trainer no longer \
                 carries its own attention, so a regression here mistrains every adapter silently."
            )
        });
        let host: Vec<f32> = grad.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            host.iter().all(|x| x.is_finite()) && host.iter().any(|x| *x != 0.0),
            "{label} gradient is all zero or non-finite"
        );
    }
}

#[test]
fn the_gdn_scan_gradient_matches_a_central_difference_in_f64() {
    let dev = Device::Cpu;
    let (b, t, n_v, d_k, d_v) = (1usize, 5usize, 3usize, 4usize, 4usize);
    let mut rng = Lcg(0xBEEF);
    let f64t = |x: Tensor| x.to_dtype(DType::F64).unwrap();
    let q = f64t(rng.tensor(&[b, t, n_v, d_k], 0.4));
    let k = f64t(rng.tensor(&[b, t, n_v, d_k], 0.4));
    let v0 = f64t(rng.tensor(&[b, t, n_v, d_v], 0.4));
    let g = f64t(
        rng.tensor(&[b, t, n_v], 0.1)
            .affine(1.0, 0.8)
            .unwrap(),
    );
    let beta = f64t(rng.tensor(&[b, t, n_v], 0.1).affine(1.0, 0.5).unwrap());
    let probe = f64t(rng.tensor(&[b, t, n_v, d_v], 1.0));

    let v = Var::from_tensor(&v0).unwrap();
    let out = gdn_scan(&q, &k, v.as_tensor(), &g, &beta, b, t, n_v, d_k, d_v).unwrap();
    let loss = out.mul(&probe).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let dv = grads
        .get(&v)
        .expect("the scan input received no gradient at all, so nothing upstream of it trains")
        .flatten_all()
        .unwrap()
        .to_vec1::<f64>()
        .unwrap();

    let n = b * t * n_v * d_v;
    let host: Vec<f64> = v0.flatten_all().unwrap().to_vec1().unwrap();
    let eval = |vals: &[f64]| -> f64 {
        let vt = Tensor::from_vec(vals.to_vec(), (b, t, n_v, d_v), &dev).unwrap();
        gdn_scan(&q, &k, &vt, &g, &beta, b, t, n_v, d_k, d_v)
            .unwrap()
            .mul(&probe)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f64>()
            .unwrap()
    };
    let eps = 1e-6f64;
    for idx in [0usize, 1, n / 2, n - d_v - 1, n - 1] {
        let mut hi = host.clone();
        hi[idx] += eps;
        let mut lo = host.clone();
        lo[idx] -= eps;
        let numeric = (eval(&hi) - eval(&lo)) / (2.0 * eps);
        let got = dv[idx];
        let tol = 1e-5 * numeric.abs().max(1.0);
        assert!(
            (numeric - got).abs() < tol,
            "d(loss)/dv[{idx}]: the gated-delta-net scan backward says {got}, an independent f64 \
             central difference says {numeric}"
        );
    }
}

#[test]
fn the_scan_is_not_a_constant_of_its_gate_and_decay_inputs() {
    let (b, t, n_v, d_k, d_v) = (1usize, 4usize, 2usize, 4usize, 4usize);
    let mut rng = Lcg(0xC0DE);
    let q = rng.tensor(&[b, t, n_v, d_k], 0.4);
    let k = rng.tensor(&[b, t, n_v, d_k], 0.4);
    let v = rng.tensor(&[b, t, n_v, d_v], 0.4);
    let g = rng.tensor(&[b, t, n_v], 0.05).affine(1.0, 0.9).unwrap();
    let beta = rng.tensor(&[b, t, n_v], 0.05).affine(1.0, 0.5).unwrap();
    let base = gdn_scan(&q, &k, &v, &g, &beta, b, t, n_v, d_k, d_v).unwrap();

    let g_zero = Tensor::zeros(g.shape(), g.dtype(), g.device()).unwrap();
    let no_memory = gdn_scan(&q, &k, &v, &g_zero, &beta, b, t, n_v, d_k, d_v).unwrap();
    assert!(
        max_abs_diff(&base, &no_memory) > 1e-4,
        "zeroing the decay must change the scan output, or the recurrence is not carrying state \
         across timesteps and the parity gate is comparing two copies of the same bug"
    );

    let beta_zero = Tensor::zeros(beta.shape(), beta.dtype(), beta.device()).unwrap();
    let never_written = gdn_scan(&q, &k, &v, &g, &beta_zero, b, t, n_v, d_k, d_v).unwrap();
    assert!(
        amax(&never_written) < 1e-6,
        "with beta zero no value is ever written into the state, so every output must be exactly \
         zero; got amax {}",
        amax(&never_written)
    );
}

#[test]
fn a_qwen38_shaped_checkpoint_trains_lora_with_monotone_loss_finite_grads_and_nonzero_deltas() {
    let root = scratch("train");
    let base = root.join("base");
    let cfg = write_tiny_checkpoint(&base);
    assert_eq!(cfg.num_hidden_layers, 4);
    let corpus = corpus_file(&root);
    let out = root.join("adapter");

    let args = Q38TrainArgs {
        base: base.clone(),
        out: out.clone(),
        corpus,
        corpus_bytes: 4096,
        seq_len: 24,
        examples: 2,
        grad_layers: 2,
        rank: 4,
        alpha: 4.0,
        lr: 5e-3,
        steps: 8,
        targets: Q38TrainArgs::default_targets(),
        seed: 11,
        dtype: DType::F32,
    };
    let s = nv_train::qwen38::run(&args, &Device::Cpu).unwrap();

    assert_eq!(s.window_start, 2, "grad_layers 2 of 4 starts the window at 2");
    assert_eq!(
        s.window_kinds,
        vec!["gdn", "full_attn"],
        "the window must span both mixer kinds, or the GDN autograd path is untested here"
    );
    assert_eq!(
        s.modules.len(),
        13,
        "the GDN layer offers 3 mixer projections plus 3 MLP and the full-attention layer offers \
         4 plus 3, got {:?}",
        s.modules
    );
    assert!(
        s.all_grads_finite,
        "a non-finite gradient reached the optimiser"
    );
    assert!(
        s.losses_are_monotonically_decreasing(),
        "loss must fall on every step of a fixed-batch overfit run; got {:?}",
        s.losses
    );
    assert!(
        s.total_drop() > 0.05,
        "loss fell by only {} over {} steps, which is inside the noise of doing nothing; losses \
         {:?}",
        s.total_drop(),
        s.losses.len(),
        s.losses
    );
    assert!(
        s.min_module_delta_amax > 0.0,
        "at least one targeted module ended with an all-zero B @ A delta, so it was never trained"
    );

    let cfg_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("adapter_config.json")).unwrap())
            .unwrap();
    assert_eq!(cfg_json["peft_type"], "LORA");
    assert_eq!(cfg_json["r"], 4);
    let st =
        candle_core::safetensors::load(out.join("adapter_model.safetensors"), &Device::Cpu).unwrap();
    for m in &s.modules {
        assert!(
            st.contains_key(&format!("base_model.model.{m}.lora_A.weight")),
            "adapter is missing lora_A for {m}, so nvk-lora-merge would not see it"
        );
        assert!(
            st.contains_key(&format!("base_model.model.{m}.lora_B.weight")),
            "adapter is missing lora_B for {m}"
        );
    }
    for m in &s.modules {
        let layer: usize = m
            .trim_start_matches("model.language_model.layers.")
            .split('.')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!(
            layer >= s.window_start,
            "{m} is below the frozen-prefix boundary {}; its gradient would have been taken \
             against a cached hidden state that never sees it",
            s.window_start
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_adapter_the_trainer_writes_merges_into_the_checkpoint_it_was_trained_on() {
    let root = scratch("merge");
    let base = root.join("base");
    write_tiny_checkpoint(&base);
    let corpus = corpus_file(&root);
    let out = root.join("adapter");

    let args = Q38TrainArgs {
        base: base.clone(),
        out: out.clone(),
        corpus,
        corpus_bytes: 4096,
        seq_len: 16,
        examples: 1,
        grad_layers: 1,
        rank: 4,
        alpha: 8.0,
        lr: 5e-3,
        steps: 3,
        targets: vec!["q_proj".to_string(), "down_proj".to_string()],
        seed: 3,
        dtype: DType::F32,
    };
    let s = nv_train::qwen38::run(&args, &Device::Cpu).unwrap();
    assert_eq!(s.modules.len(), 2, "got {:?}", s.modules);

    let merged_dir = root.join("merged");
    let report = nv_weights::lora_merge::merge_lora_adapter_into_quantized_snapshot(
        &base,
        &out,
        &merged_dir,
    )
    .expect(
        "the adapter the trainer writes must load in the same merge core nvk-lora-merge calls; \
         a format the merge tool refuses cannot be served",
    );
    assert_eq!(report.modules.len(), 2);

    let base_src = HostDequantWeights::open_dir(&base, &Device::Cpu).unwrap();
    let merged_src = HostDequantWeights::open_dir(&merged_dir, &Device::Cpu).unwrap();
    let adapter =
        candle_core::safetensors::load(out.join("adapter_model.safetensors"), &Device::Cpu)
            .unwrap();
    let scaling = args.alpha / args.rank as f64;
    for m in &s.modules {
        let w0 = base_src.dense_matrix(m, DType::F32).unwrap();
        let w1 = merged_src.dense_matrix(m, DType::F32).unwrap();
        let a = adapter[&format!("base_model.model.{m}.lora_A.weight")]
            .to_dtype(DType::F32)
            .unwrap();
        let b = adapter[&format!("base_model.model.{m}.lora_B.weight")]
            .to_dtype(DType::F32)
            .unwrap();
        let delta = b.matmul(&a).unwrap().affine(scaling, 0.0).unwrap();
        let want = w0.add(&delta).unwrap();
        let step = amax(&w1).max(1e-6) * 0.02;
        assert!(
            max_abs_diff(&want, &w1) < step,
            "{m}: merged weight is not base + {scaling} * B @ A within the bf16 storage step \
             ({} vs tolerance {step})",
            max_abs_diff(&want, &w1)
        );
        assert!(
            amax(&delta) > 0.0,
            "{m}: the merge folded an all-zero delta, which proves nothing about the round trip"
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_target_below_the_window_is_impossible_to_request_because_the_prefix_is_cached() {
    let root = scratch("window");
    let base = root.join("base");
    write_tiny_checkpoint(&base);
    let corpus = corpus_file(&root);
    let args = Q38TrainArgs {
        base: base.clone(),
        out: root.join("adapter"),
        corpus,
        corpus_bytes: 2048,
        seq_len: 12,
        examples: 1,
        grad_layers: 1,
        rank: 2,
        alpha: 2.0,
        lr: 1e-3,
        steps: 1,
        targets: vec!["in_proj_qkv".to_string()],
        seed: 5,
        dtype: DType::F32,
    };
    let err = nv_train::qwen38::run(&args, &Device::Cpu)
        .expect_err("layer 3 is full attention and carries no in_proj_qkv, so nothing matches");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("no window module matched targets"),
        "expected a loud refusal naming the targets, got: {msg}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

const A_CORRECT_QWEN38_TRUNK_SCORES_WIKITEXT_WELL_UNDER_LN_VOCAB_AND_A_BROKEN_ONE_SITS_JUST_ABOVE_IT:
    f32 = 4.0;

#[cfg(feature = "cuda")]
#[test]
#[ignore = "host-dequants the real ~22.6 GB unsloth/Qwen3.8-27B-NVFP4 and trains a LoRA on the \
            top layers; set NV_Q38_LORA_TRAIN_TEST=1"]
fn the_real_qwen38_27b_trains_a_lora_on_its_top_layers_from_a_starting_loss_only_a_correct_trunk_can_produce(
) {
    if std::env::var("NV_Q38_LORA_TRAIN_TEST").as_deref() != Ok("1") {
        panic!(
            "set NV_Q38_LORA_TRAIN_TEST=1 to run the real-checkpoint trainer; a suite that \
             silently skipped when invoked explicitly would report green on a trainer that never \
             touched the checkpoint"
        );
    }
    let base = std::env::var("NV_QWEN38_DIR")
        .map(PathBuf::from)
        .expect("set NV_QWEN38_DIR to a hydrated unsloth/Qwen3.8-27B-NVFP4 snapshot");
    let corpus = PathBuf::from(REAL_CORPUS_FILE);
    assert!(
        corpus.is_file(),
        "this gate teacher-forces real wikitext-2-raw; {} is missing",
        corpus.display()
    );
    let root = scratch("real");
    let args = Q38TrainArgs {
        base,
        out: root.join("adapter"),
        corpus,
        corpus_bytes: 16384,
        seq_len: 256,
        examples: 2,
        grad_layers: 4,
        rank: 16,
        alpha: 16.0,
        lr: 1e-4,
        steps: 8,
        targets: Q38TrainArgs::default_targets(),
        seed: 7,
        dtype: DType::BF16,
    };
    let dev = Device::new_cuda(0).expect("cuda device 0");
    let s = nv_train::qwen38::run(&args, &dev).unwrap();
    eprintln!(
        "[q38-lora-train] window_start={} kinds={:?} modules={} losses={:?} tok_per_s={:.1}",
        s.window_start,
        s.window_kinds,
        s.modules.len(),
        s.losses,
        s.tokens_per_second()
    );
    assert!(
        s.losses[0] < A_CORRECT_QWEN38_TRUNK_SCORES_WIKITEXT_WELL_UNDER_LN_VOCAB_AND_A_BROKEN_ONE_SITS_JUST_ABOVE_IT,
        "step 0 has a zero LoRA delta, so its loss IS the base model's teacher-forced cross \
         entropy on wikitext-2; {} nats against ln(248320) = 12.42 means the trunk is wrong, not \
         merely untrained. Reading the trunk RMSNorm gammas raw instead of stored + 1 lands here \
         at about 13.5 while still producing a monotone falling curve",
        s.losses[0]
    );
    assert!(s.all_grads_finite);
    assert!(
        s.losses_are_monotonically_decreasing(),
        "losses {:?}",
        s.losses
    );
    assert!(s.total_drop() > 0.05, "losses {:?}", s.losses);
    assert!(s.min_module_delta_amax > 0.0);
    assert_eq!(s.window_kinds, vec!["gdn", "gdn", "gdn", "full_attn"]);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_trunk_norms_take_a_plus_one_on_load_and_the_gdn_inner_norm_does_not() {
    let root = scratch("norms");
    let base = root.join("base");
    write_tiny_checkpoint(&base);
    let src = HostDequantWeights::open_dir(&base, &Device::Cpu).unwrap();
    let p = layer_prefix(3);
    for name in [
        format!("{p}.input_layernorm.weight"),
        format!("{p}.post_attention_layernorm.weight"),
        format!("{p}.self_attn.q_norm.weight"),
        format!("{p}.self_attn.k_norm.weight"),
        FINAL_NORM.to_string(),
    ] {
        let raw = src.plain(&name, DType::F32).unwrap();
        let used = src.trunk_rmsnorm_gamma(&name, DType::F32).unwrap();
        assert!(
            max_abs_diff(&used, &raw.affine(1.0, 1.0).unwrap()) == 0.0,
            "{name} must be loaded as stored + 1, matching qwen3_5_moe::load_rmsnorm_plus_one. \
             These gammas ship centred on zero: read raw they scale the residual stream to \
             nothing and the trainer reports a falling loss on a destroyed model"
        );
        assert!(
            amax(&raw) < 0.5,
            "{name} in the fixture must be centred on zero like the real checkpoint, or this \
             gate cannot distinguish the two conventions"
        );
    }
    let gdn = format!("{}.linear_attn.norm.weight", layer_prefix(2));
    let raw = src.plain(&gdn, DType::F32).unwrap();
    assert!(
        amax(&raw) > 0.5,
        "linear_attn.norm.weight is loaded raw by nv_layers::LinearAttention::from_loader, so it \
         ships centred on its true value"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_jsonl_trainer_reports_heldout_adapter_and_base_losses_the_rollover_gate_reads() {
    let root = scratch("jsonl");
    let base = root.join("base");
    write_tiny_checkpoint(&base);
    let prose =
        THIS_PROSE_IS_A_REAL_CORPUS_SO_A_RUN_WITHOUT_WIKITEXT_STILL_TRAINS_ON_ENGLISH_AND_NOT_ON_NOISE;
    let row = |s: &str| format!("{}\n", serde_json::json!({ "text": s }));
    let data = root.join("train.jsonl");
    std::fs::write(
        &data,
        format!(
            "{}{}{}",
            row(&prose[0..40]),
            row(&prose[40..90]),
            row(&prose[90..120])
        ),
    )
    .unwrap();
    let eval = root.join("eval.jsonl");
    std::fs::write(&eval, format!("{}{}", row(&prose[120..160]), row(&prose[160..200]))).unwrap();

    let servable: Vec<String> = [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let args = Q38JsonlArgs {
        base: base.clone(),
        data: data.clone(),
        eval_data: Some(eval),
        out: root.join("adapter"),
        grad_layers: 2,
        rank: 4,
        alpha: 4.0,
        targets: servable,
        steps: 6,
        lr: 5e-3,
        b_lr_ratio: 1.0,
        seed: 11,
    };
    let s = nv_train::qwen38::run_jsonl(&args, &Device::Cpu).unwrap();

    assert_eq!(s.window_start, 2);
    assert_eq!(s.num_examples, 3);
    assert_eq!(s.num_eval_examples, 2);
    assert_eq!(
        s.modules.len(),
        10,
        "servable targets in a gdn+full window are 3 MLP modules plus 4+3, got {:?}",
        s.modules
    );
    assert!(
        s.modules.iter().all(|m| !m.contains("linear_attn")),
        "the servable target set must never produce a linear_attn adapter module: {:?}",
        s.modules
    );
    assert!(
        s.losses.first().unwrap() > s.losses.last().unwrap(),
        "an overfit run on a fixed batch must reduce the loss; got {:?}",
        s.losses
    );
    let la = s
        .eval_loss_adapter
        .expect("validation rows were supplied, so the adapter-attached held-out CE must exist");
    let lb = s
        .eval_loss_base
        .expect("validation rows were supplied, so the adapter-detached held-out CE must exist");
    assert!(la.is_finite() && lb.is_finite(), "la {la} lb {lb}");
    assert!(
        (la - lb).abs() > 0.0,
        "after training the adapter delta is nonzero, so attached and detached held-out CE \
         cannot coincide; both read {la}"
    );
    assert_eq!(
        (s.eval_ce_adapter.len(), s.eval_ce_base.len()),
        (2, 2),
        "per-row paired CE must carry one entry per loaded held-out row; the loader drops \
         unusable jsonl lines, so indices are loaded-dataset order, not file line numbers"
    );
    assert_eq!(
        s.eval_ce_adapter.iter().sum::<f32>(),
        la,
        "eval_loss_adapter must stay exactly the sum of the per-row adapter CE means"
    );
    assert_eq!(
        s.eval_ce_base.iter().sum::<f32>(),
        lb,
        "eval_loss_base must stay exactly the sum of the per-row base CE means"
    );
    assert!(
        s.eval_ce_adapter
            .iter()
            .zip(&s.eval_ce_base)
            .any(|(a, b)| a != b),
        "the paired per-row deltas cannot all be zero after training moved the scalar"
    );
    assert!(
        s.eval_ce_adapter
            .iter()
            .chain(&s.eval_ce_base)
            .all(|x| x.is_finite()),
        "per-row CE must be finite: adapter {:?} base {:?}",
        s.eval_ce_adapter,
        s.eval_ce_base
    );
    assert!(s.adapter_path.is_file() && s.config_path.is_file());
    assert!(
        s.all_grads_finite,
        "a healthy tiny-fixture run must surface all_grads_finite=true; the endpoint fails the \
         job on false instead of hot-loading a NaN adapter"
    );

    let no_eval = Q38JsonlArgs {
        eval_data: None,
        out: root.join("adapter2"),
        ..args
    };
    let s2 = nv_train::qwen38::run_jsonl(&no_eval, &Device::Cpu).unwrap();
    assert_eq!(s2.eval_loss_adapter, None);
    assert_eq!(s2.eval_loss_base, None);
    assert!(
        s2.eval_ce_adapter.is_empty() && s2.eval_ce_base.is_empty(),
        "no eval rows means no per-row CE, never a fabricated vector"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn eval_ce_honors_the_prompt_boundary_row_for_row_on_identical_token_streams() {
    let root = scratch("evalmask");
    let base = root.join("base");
    write_tiny_checkpoint(&base);
    let prompt = "The Valkyria series began ";
    let completion = "as a tactical role playing franchise.";
    let whole = format!("{prompt}{completion}");
    let data = root.join("train.jsonl");
    std::fs::write(
        &data,
        format!("{}\n", serde_json::json!({ "text": whole.as_str() })),
    )
    .unwrap();
    let eval = root.join("eval.jsonl");
    std::fs::write(
        &eval,
        format!(
            "{}\n{}\n{}\n",
            serde_json::json!({ "text": whole.as_str() }),
            serde_json::json!({ "prompt": "", "completion": whole.as_str() }),
            serde_json::json!({ "prompt": prompt, "completion": completion }),
        ),
    )
    .unwrap();
    let args = Q38JsonlArgs {
        base,
        data,
        eval_data: Some(eval),
        out: root.join("adapter"),
        grad_layers: 2,
        rank: 2,
        alpha: 2.0,
        targets: vec!["q_proj".into(), "down_proj".into()],
        steps: 1,
        lr: 1e-3,
        b_lr_ratio: 1.0,
        seed: 3,
    };
    let s = nv_train::qwen38::run_jsonl(&args, &Device::Cpu).unwrap();
    assert_eq!(s.num_eval_examples, 3);
    let ce = &s.eval_ce_base;
    assert_eq!(
        ce[0], ce[1],
        "byte tokenization makes rows 0 and 1 the same ids with the same all-positions mask, so \
         their per-row CE must match bitwise"
    );
    assert_ne!(
        ce[2], ce[0],
        "row 2 carries the same ids but a completion-only mask; equal CE would mean the prompt \
         boundary is ignored and eval scores prompt positions"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_row_at_the_512_token_cap_trains_through_the_jsonl_arm_end_to_end() {
    let root = scratch("cap512");
    let base = root.join("base");
    let cfg = write_tiny_checkpoint(&base);
    let long = THIS_PROSE_IS_A_REAL_CORPUS_SO_A_RUN_WITHOUT_WIKITEXT_STILL_TRAINS_ON_ENGLISH_AND_NOT_ON_NOISE
        .repeat(2);
    assert!(long.len() > 600);
    let data = root.join("train.jsonl");
    std::fs::write(&data, format!("{}\n", serde_json::json!({ "text": long }))).unwrap();

    let cap =
        nv_train::JSONL_MAX_SEQ_IS_512_SHARED_BY_BOTH_ENDPOINT_ARMS_BECAUSE_THE_DEFAULT_QWEN_WINDOWS_GDN_GRAPH_AT_512_IS_ABOUT_14_GB;
    assert_eq!(cap, 512);
    let rows = nv_train::load_jsonl_token_dataset(&data, &base, cfg.vocab_size, cap).unwrap();
    assert_eq!(
        rows[0].ids.len(),
        512,
        "the loader must feed the full 512-token cap, not the old 32"
    );

    let args = Q38JsonlArgs {
        base: base.clone(),
        data,
        eval_data: None,
        out: root.join("adapter"),
        grad_layers: 1,
        rank: 2,
        alpha: 2.0,
        targets: vec!["q_proj".into()],
        steps: 1,
        lr: 1e-3,
        b_lr_ratio: 1.0,
        seed: 5,
    };
    let s = nv_train::qwen38::run_jsonl(&args, &Device::Cpu).unwrap();
    assert_eq!(s.num_examples, 1);
    assert!(s.losses[0].is_finite());
    assert!(s.all_grads_finite);
    assert!(s.adapter_path.is_file());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_b_lr_ratio_of_one_is_exactly_the_single_optimizer_and_above_one_changes_the_trajectory() {
    let root = scratch("loraplus");
    let base = root.join("base");
    write_tiny_checkpoint(&base);
    let data = root.join("train.jsonl");
    let prose =
        THIS_PROSE_IS_A_REAL_CORPUS_SO_A_RUN_WITHOUT_WIKITEXT_STILL_TRAINS_ON_ENGLISH_AND_NOT_ON_NOISE;
    std::fs::write(
        &data,
        format!(
            "{}\n{}\n",
            serde_json::json!({ "text": &prose[0..60] }),
            serde_json::json!({ "text": &prose[60..120] })
        ),
    )
    .unwrap();
    let args = |out: &str, ratio: f64| Q38JsonlArgs {
        base: base.clone(),
        data: data.clone(),
        eval_data: None,
        out: root.join(out),
        grad_layers: 1,
        rank: 4,
        alpha: 4.0,
        targets: vec!["q_proj".into(), "down_proj".into()],
        steps: 3,
        lr: 5e-3,
        b_lr_ratio: ratio,
        seed: 23,
    };
    let one_a = nv_train::qwen38::run_jsonl(&args("a", 1.0), &Device::Cpu).unwrap();
    let one_b = nv_train::qwen38::run_jsonl(&args("b", 1.0), &Device::Cpu).unwrap();
    assert_eq!(
        one_a.losses, one_b.losses,
        "ratio 1.0 must be the exact single-optimizer path, reproducible bit for bit"
    );
    let hot = nv_train::qwen38::run_jsonl(&args("c", 8.0), &Device::Cpu).unwrap();
    assert_eq!(
        hot.losses[0], one_a.losses[0],
        "step 0 is scored before any optimiser update, so the ratio cannot touch it"
    );
    let (a_last, hot_last) = (*one_a.losses.last().unwrap(), *hot.losses.last().unwrap());
    assert!(
        (a_last - hot_last).abs() / a_last.abs().max(1e-6) > 1e-6,
        "an 8x B learning rate left the loss trajectory untouched ({a_last} vs {hot_last}), so \
         the split optimiser is not stepping B differently"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_rope_table_is_built_for_the_partial_rotary_dimension_not_the_head_dimension() {
    let cfg = Q38Config::from_hf_json_str(&tiny_config_json()).unwrap();
    assert_eq!(cfg.head_dim, 8);
    assert_eq!(
        cfg.rotary_dim(),
        4,
        "partial_rotary_factor 0.5 of head_dim 8 rotates only the first 4 lanes"
    );
    let rope = build_rope(&cfg, 32, &Device::Cpu).unwrap();
    assert_eq!(
        rope.cos().dims(),
        &[32, 2],
        "the table is rotary_dim/2 wide; a head_dim-wide table would rotate lanes the served \
         model leaves alone"
    );
}
