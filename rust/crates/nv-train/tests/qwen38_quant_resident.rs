use candle_core::{DType, Device, Tensor};
use nv_train::qwen38::{
    run_jsonl, Q38JsonlArgs, Q38TrainArgs,
    NV_TRAIN_QUANT_RESIDENT_KEEPS_NVFP4_AND_FP8_MODULES_RESIDENT_SO_TRAINING_RUNS_THE_SERVING_QUANT_GEMM as ARM,
};
use nv_train::qwen38_fixture::write_tiny_checkpoint;
use nv_train::{LoraConfig, LoraTrainable};
use std::path::PathBuf;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

const PROSE: &str =
    "The Valkyria series began as a tactical role playing franchise for home consoles, and the \
     third entry was released only in Japan. Its battle system alternates a top down command map \
     with a third person action phase, so a single turn mixes planning and marksmanship. Critics \
     praised the art direction and the score, while faulting a story that reuses the structure \
     of the first game.";

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nv-train-qres-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn jsonl_args(root: &PathBuf, out_leaf: &str) -> Q38JsonlArgs {
    let base = root.join("base");
    if !base.join("config.json").is_file() {
        write_tiny_checkpoint(&base);
    }
    let row = |s: &str| format!("{}\n", serde_json::json!({ "text": s }));
    let data = root.join("train.jsonl");
    std::fs::write(&data, format!("{}{}", row(&PROSE[0..60]), row(&PROSE[60..130]))).unwrap();
    let eval = root.join("eval.jsonl");
    std::fs::write(&eval, format!("{}{}", row(&PROSE[130..190]), row(&PROSE[190..250]))).unwrap();
    Q38JsonlArgs {
        base,
        data,
        eval_data: Some(eval),
        out: root.join(out_leaf),
        grad_layers: 2,
        rank: 2,
        alpha: 2.0,
        targets: vec![
            "q_proj".into(),
            "o_proj".into(),
            "out_proj".into(),
            "down_proj".into(),
        ],
        steps: 2,
        lr: 1e-3,
        b_lr_ratio: 1.0,
        seed: 11,
    }
}

#[test]
fn the_default_path_is_deterministically_byte_identical_and_reports_no_resident_base() {
    let _g = ENV_LOCK.lock().unwrap();
    std::env::remove_var(ARM);
    let root = scratch("default");
    let a = run_jsonl(&jsonl_args(&root, "adapter-a"), &Device::Cpu).unwrap();
    let b = run_jsonl(&jsonl_args(&root, "adapter-b"), &Device::Cpu).unwrap();
    assert!(!a.resident_base && !b.resident_base);
    assert_eq!(
        a.base_dtype, "F32",
        "the default arm's base_dtype string must not carry the resident marker"
    );
    assert_eq!(a.losses, b.losses, "same fixture, same seed, same losses");
    let bytes = |leaf: &str| std::fs::read(root.join(leaf).join("adapter_model.safetensors")).unwrap();
    assert_eq!(
        bytes("adapter-a"),
        bytes("adapter-b"),
        "the env-unset trainer must stay byte-identical run to run"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn from_features_seeds_the_same_factors_as_the_dense_base_constructor() {
    let base = Tensor::zeros((24usize, 16usize), DType::F32, &Device::Cpu).unwrap();
    let cfg = LoraConfig {
        r: 4,
        alpha: 8.0,
        dropout: 0.0,
    };
    let with_base = LoraTrainable::new(&base, cfg, 77, &Device::Cpu).unwrap();
    let dims_only = LoraTrainable::from_features(24, 16, cfg, 77, &Device::Cpu).unwrap();
    let host = |t: &Tensor| -> Vec<f32> { t.flatten_all().unwrap().to_vec1().unwrap() };
    assert_eq!(
        host(with_base.a_tensor()),
        host(dims_only.a_tensor()),
        "A init depends only on (seed, in_features); a drift here silently changes every \
         adapter trained after the resident wiring"
    );
    assert_eq!(host(with_base.b_tensor()), host(dims_only.b_tensor()));
    assert!(with_base.base_tensor().is_some());
    assert!(dims_only.base_tensor().is_none());
}

#[test]
fn the_resident_arm_refuses_an_all_bf16_checkpoint_loudly_before_touching_a_device() {
    let _g = ENV_LOCK.lock().unwrap();
    let root = scratch("refuse-bf16");
    std::env::set_var(ARM, "1");
    let err = format!(
        "{:#}",
        run_jsonl(&jsonl_args(&root, "adapter-resident"), &Device::Cpu).unwrap_err()
    );
    std::env::remove_var(ARM);
    assert!(
        err.contains("NV_TRAIN_QUANT_RESIDENT") && err.contains("nvfp4-packed or fp8-rowscale"),
        "a bf16-only checkpoint must refuse the arm by name instead of silently running the \
         host-dequant path, got: {err}"
    );
    assert!(
        !root.join("adapter-resident").join("adapter_model.safetensors").exists(),
        "a refused arm must not leave an adapter behind"
    );
    let _ = std::fs::remove_dir_all(&root);
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore]
fn resident_arm_on_a_real_packed_base_trains_and_writes_the_ab_sidecar() {
    if std::env::var("NV_Q38_RESIDENT_AB_TEST").as_deref() != Ok("1") {
        eprintln!("skip: NV_Q38_RESIDENT_AB_TEST unset");
        return;
    }
    let base = PathBuf::from(std::env::var("NV_Q38_AB_BASE").expect("NV_Q38_AB_BASE"));
    let data = PathBuf::from(std::env::var("NV_Q38_AB_DATA").expect("NV_Q38_AB_DATA"));
    let eval = PathBuf::from(std::env::var("NV_Q38_AB_EVAL_DATA").expect("NV_Q38_AB_EVAL_DATA"));
    let out = PathBuf::from(std::env::var("NV_Q38_AB_OUT").expect("NV_Q38_AB_OUT"));
    let want_resident = std::env::var(ARM).as_deref() == Ok("1");
    let args = Q38JsonlArgs {
        base,
        data,
        eval_data: Some(eval),
        out: out.clone(),
        grad_layers: env_or("NV_Q38_AB_GRAD_LAYERS", 2),
        rank: env_or("NV_Q38_AB_RANK", 8),
        alpha: env_or("NV_Q38_AB_ALPHA", 8.0),
        targets: Q38TrainArgs::default_targets(),
        steps: env_or("NV_Q38_AB_STEPS", 4),
        lr: env_or("NV_Q38_AB_LR", 1e-4),
        b_lr_ratio: 1.0,
        seed: env_or("NV_Q38_AB_SEED", 7),
    };
    let t0 = std::time::Instant::now();
    let device = Device::new_cuda(0).expect("the resident A/B needs the CUDA device");
    let s = run_jsonl(&args, &device).unwrap();
    let wall_seconds = t0.elapsed().as_secs_f64();
    assert!(s.all_grads_finite, "a NaN-gradient run is not an A/B datum");
    assert_eq!(
        s.resident_base, want_resident,
        "the summary must carry the arm identity the env selected"
    );
    let sidecar = serde_json::json!({
        "resident_base": s.resident_base,
        "base_dtype": s.base_dtype,
        "train_losses": s.losses,
        "eval_loss_adapter": s.eval_loss_adapter,
        "eval_loss_base": s.eval_loss_base,
        "eval_ce_adapter": s.eval_ce_adapter,
        "eval_ce_base": s.eval_ce_base,
        "num_examples": s.num_examples,
        "num_eval_examples": s.num_eval_examples,
        "window_start": s.window_start,
        "checkpointed": s.checkpointed,
        "steps": args.steps,
        "seed": args.seed,
        "wall_seconds": wall_seconds,
    });
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(
        out.join("resident_train_summary.json"),
        serde_json::to_string_pretty(&sidecar).unwrap(),
    )
    .unwrap();
    println!("[resident-ab] {}", serde_json::to_string(&sidecar).unwrap());
}
