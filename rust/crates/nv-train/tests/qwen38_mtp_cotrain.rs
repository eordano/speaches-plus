use candle_core::{DType, Device, Tensor};
use nv_train::qwen38::{
    run_jsonl, run_jsonl_with_mtp_cotrain, run_with_mtp_cotrain, Q38JsonlArgs, Q38MtpCoTrainArgs,
    Q38TrainArgs, QWEN38_MTP_SPECULATOR_KIND,
};
use nv_train::qwen38_fixture::write_tiny_checkpoint;
use nv_train::{SPECULATOR_KIND_KEY, SPECULATOR_SUBDIR};
use std::path::{Path, PathBuf};

mod mtp_fixture;

const PROSE: &str =
    "The battle system alternates a top down command map with a third person action phase, so a \
     single turn mixes planning and marksmanship. Critics praised the art direction and the \
     score, while faulting a story that reuses the structure of the first game.";

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nv-train-mtpct-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_jsonl(root: &Path) -> PathBuf {
    let row = |s: &str| format!("{}\n", serde_json::json!({ "text": s }));
    let data = root.join("train.jsonl");
    std::fs::write(
        &data,
        format!(
            "{}{}{}",
            row(&PROSE[0..48]),
            row(&PROSE[48..110]),
            row(&PROSE[110..150])
        ),
    )
    .unwrap();
    data
}

fn jsonl_args(base: &Path, data: &Path, out: PathBuf) -> Q38JsonlArgs {
    Q38JsonlArgs {
        base: base.to_path_buf(),
        data: data.to_path_buf(),
        eval_data: None,
        out,
        grad_layers: 2,
        rank: 4,
        alpha: 4.0,
        targets: vec![
            "q_proj".into(),
            "v_proj".into(),
            "gate_proj".into(),
            "down_proj".into(),
        ],
        steps: 5,
        lr: 5e-3,
        b_lr_ratio: 1.0,
        seed: 19,
    }
}

fn co_args() -> Q38MtpCoTrainArgs {
    Q38MtpCoTrainArgs {
        rank: 4,
        alpha: 4.0,
        speculator_lr_ratio: 4.0,
    }
}

fn tensor_bits_equal(a: &Tensor, b: &Tensor) -> bool {
    a.dims() == b.dims()
        && a.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            == b.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
}

const EXPECTED_HEAD_MODULES: [&str; 8] = [
    "mtp.fc",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.mlp.down_proj",
];

#[test]
fn the_defaults_are_the_medusa2_recipe_rank_32_and_speculator_lr_ratio_4() {
    let d = Q38MtpCoTrainArgs::default();
    assert_eq!(d.rank, 32);
    assert_eq!(d.alpha, 32.0);
    assert_eq!(d.speculator_lr_ratio, 4.0);
}

#[test]
fn cotraining_trains_head_loras_and_ships_the_speculator_pair_inside_the_adapter_dir() {
    let root = scratch("jsonl-cotrain");
    let base = root.join("base");
    let cfg = write_tiny_checkpoint(&base);
    mtp_fixture::write_tiny_mtp_head(&base, &cfg);
    let data = write_jsonl(&root);
    let out = root.join("adapter");

    let (summary, co) = run_jsonl_with_mtp_cotrain(
        &jsonl_args(&base, &data, out.clone()),
        &co_args(),
        &Device::Cpu,
    )
    .unwrap();

    assert_eq!(co.speculator_kind, QWEN38_MTP_SPECULATOR_KIND);
    assert_eq!(co.speculator_dir, out.join(SPECULATOR_SUBDIR));
    assert_eq!(co.head_modules.len(), 8, "got {:?}", co.head_modules);
    for m in EXPECTED_HEAD_MODULES {
        assert!(
            co.head_modules.iter().any(|x| x == m),
            "head module {m} missing from {:?}",
            co.head_modules
        );
    }
    assert_eq!(
        co.head_trainable_vars, 16,
        "8 head modules with an A and a B factor each"
    );
    assert!(co.all_head_grads_finite);
    assert_eq!(co.head_losses.len(), summary.losses.len());
    assert!(
        co.head_losses.iter().all(|l| l.is_finite()),
        "head losses {:?}",
        co.head_losses
    );
    assert!(
        co.head_losses.first().unwrap() > co.head_losses.last().unwrap(),
        "a fixed-batch overfit run must reduce the head CE, or the head optimiser never moved a \
         head-LoRA var; got {:?}",
        co.head_losses
    );

    let sdir = co.speculator_dir;
    let scfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(sdir.join("adapter_config.json")).unwrap())
            .unwrap();
    assert_eq!(scfg["peft_type"], "LORA");
    assert_eq!(scfg["r"], 4);
    assert_eq!(scfg[SPECULATOR_KIND_KEY], QWEN38_MTP_SPECULATOR_KIND);
    let st =
        candle_core::safetensors::load(sdir.join("adapter_model.safetensors"), &Device::Cpu)
            .unwrap();
    let mut trained_b = 0usize;
    for m in EXPECTED_HEAD_MODULES {
        assert!(st.contains_key(&format!("base_model.model.{m}.lora_A.weight")));
        let b = &st[&format!("base_model.model.{m}.lora_B.weight")];
        let bmax = b
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        if bmax > 0.0 {
            trained_b += 1;
        }
    }
    assert!(
        trained_b > 0,
        "every head lora_B is still exactly zero, so no head gradient ever reached an optimiser \
         step and the speculator artifact is untrained"
    );

    let main_cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("adapter_config.json")).unwrap())
            .unwrap();
    assert_eq!(
        main_cfg[SPECULATOR_KIND_KEY], QWEN38_MTP_SPECULATOR_KIND,
        "the main adapter config must name the speculator kind so the pair ships as one artifact"
    );
    assert_eq!(main_cfg["peft_type"], "LORA");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn without_cotraining_the_signature_artifact_and_losses_are_the_plain_trainers_bit_for_bit() {
    let root = scratch("regression");
    let base = root.join("base");
    let cfg = write_tiny_checkpoint(&base);
    mtp_fixture::write_tiny_mtp_head(&base, &cfg);
    let data = write_jsonl(&root);

    let plain = run_jsonl(&jsonl_args(&base, &data, root.join("plain")), &Device::Cpu).unwrap();
    let (co_trunk, co) = run_jsonl_with_mtp_cotrain(
        &jsonl_args(&base, &data, root.join("cotrain")),
        &co_args(),
        &Device::Cpu,
    )
    .unwrap();

    assert_eq!(
        plain.losses, co_trunk.losses,
        "the head CE consumes the window output detached, so the target trajectory under \
         co-training must be the plain trainer's bit for bit; a drift here means head loss is \
         leaking into the target adapter"
    );
    assert!(!co.head_losses.is_empty());

    let plain_cfg_raw =
        std::fs::read_to_string(root.join("plain").join("adapter_config.json")).unwrap();
    let plain_cfg: serde_json::Value = serde_json::from_str(&plain_cfg_raw).unwrap();
    assert!(
        plain_cfg.get(SPECULATOR_KIND_KEY).is_none(),
        "the plain signature must not name a speculator kind"
    );
    assert!(
        !root.join("plain").join(SPECULATOR_SUBDIR).exists(),
        "the plain signature must not write a speculator/ subdir"
    );

    let co_cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("cotrain").join("adapter_config.json")).unwrap(),
    )
    .unwrap();
    let plain_obj = plain_cfg.as_object().unwrap();
    let co_obj = co_cfg.as_object().unwrap();
    for (k, v) in plain_obj {
        assert_eq!(
            co_obj.get(k),
            Some(v),
            "main config key {k} must be unchanged by co-training"
        );
    }
    assert_eq!(
        co_obj.len(),
        plain_obj.len() + 1,
        "co-training adds exactly the speculator-kind key to the main config"
    );

    let a = candle_core::safetensors::load(
        root.join("plain").join("adapter_model.safetensors"),
        &Device::Cpu,
    )
    .unwrap();
    let b = candle_core::safetensors::load(
        root.join("cotrain").join("adapter_model.safetensors"),
        &Device::Cpu,
    )
    .unwrap();
    assert_eq!(a.len(), b.len());
    for (k, ta) in &a {
        let tb = b.get(k).unwrap_or_else(|| panic!("{k} missing from the co-trained adapter"));
        assert!(
            tensor_bits_equal(ta, tb),
            "{k}: the co-trained target delta differs from the plain trainer's, so head \
             gradients reached a trunk LoRA var"
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_checkpoint_without_mtp_weights_refuses_cotraining_naming_the_missing_tensors() {
    let root = scratch("refuse");
    let base = root.join("base");
    write_tiny_checkpoint(&base);
    let data = write_jsonl(&root);
    let err = run_jsonl_with_mtp_cotrain(
        &jsonl_args(&base, &data, root.join("adapter")),
        &co_args(),
        &Device::Cpu,
    )
    .expect_err("the tiny fixture ships no mtp.* tensors, so co-training has no head to train");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("mtp.fc.weight") && msg.contains("no complete MTP head"),
        "the refusal must name the missing tensors so the caller can tell a wrong base from a \
         wrong request; got: {msg}"
    );
    assert!(
        !root.join("adapter").join("adapter_config.json").exists(),
        "a refused co-training request must not leave a partial adapter behind"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_corpus_arm_gains_the_same_cotrain_signature_with_an_untouched_target_trajectory() {
    let root = scratch("corpus-arm");
    let base = root.join("base");
    let cfg = write_tiny_checkpoint(&base);
    mtp_fixture::write_tiny_mtp_head(&base, &cfg);
    let corpus = root.join("corpus.txt");
    let mut body = String::new();
    while body.len() < 4096 {
        body.push_str(PROSE);
        body.push(' ');
    }
    std::fs::write(&corpus, body).unwrap();

    let args = |out: &str| Q38TrainArgs {
        base: base.clone(),
        out: root.join(out),
        corpus: corpus.clone(),
        corpus_bytes: 2048,
        seq_len: 16,
        examples: 2,
        grad_layers: 2,
        rank: 4,
        alpha: 4.0,
        lr: 5e-3,
        steps: 4,
        targets: Q38TrainArgs::default_targets(),
        seed: 29,
        dtype: DType::F32,
    };
    let plain = nv_train::qwen38::run(&args("plain"), &Device::Cpu).unwrap();
    let (co_trunk, co) =
        run_with_mtp_cotrain(&args("cotrain"), &co_args(), &Device::Cpu).unwrap();

    assert_eq!(plain.losses, co_trunk.losses);
    assert_eq!(co.head_losses.len(), plain.losses.len());
    assert!(co.all_head_grads_finite);
    assert!(
        co.head_losses.first().unwrap() > co.head_losses.last().unwrap(),
        "head losses {:?}",
        co.head_losses
    );
    assert!(co
        .speculator_dir
        .join("adapter_model.safetensors")
        .is_file());
    assert!(co.speculator_dir.join("adapter_config.json").is_file());
    let _ = std::fs::remove_dir_all(&root);
}
