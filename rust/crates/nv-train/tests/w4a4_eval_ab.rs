use candle_core::Device;
use nv_train::qwen38::{run_jsonl, Q38JsonlArgs, Q38TrainArgs};
use nv_train::qwen38_fixture::write_tiny_checkpoint;
use std::path::PathBuf;

const PROSE: &str =
    "The Valkyria series began as a tactical role playing franchise for home consoles, and the \
     third entry was released only in Japan. Its battle system alternates a top down command map \
     with a third person action phase, so a single turn mixes planning and marksmanship. Critics \
     praised the art direction and the score, while faulting a story that reuses the structure \
     of the first game.";

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nv-train-w4a4ab-{name}-{}", std::process::id()));
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
        steps: 1,
        lr: 1e-3,
        b_lr_ratio: 1.0,
        seed: 11,
    }
}

#[test]
fn the_w4a4_eval_arm_is_opt_in_and_refuses_to_pretend_without_the_serving_gemm() {
    let root = scratch("optin");
    std::env::remove_var("NV_Q38_EVAL_W4A4");
    let args = jsonl_args(&root, "adapter-plain");
    let s = run_jsonl(&args, &Device::Cpu).unwrap();
    assert!(s.eval_loss_adapter.is_some() && s.eval_loss_base.is_some());
    assert_eq!(s.eval_ce_adapter.len(), s.num_eval_examples);
    assert_eq!(s.eval_ce_base.len(), s.num_eval_examples);
    assert!(
        !args.out.join("w4a4_eval.json").exists(),
        "the w4a4 sidecar must not appear without NV_Q38_EVAL_W4A4=1"
    );

    std::env::set_var("NV_Q38_EVAL_W4A4", "1");
    let args = jsonl_args(&root, "adapter-w4a4-cpu");
    let err = format!("{:#}", run_jsonl(&args, &Device::Cpu).unwrap_err());
    std::env::remove_var("NV_Q38_EVAL_W4A4");
    assert!(
        err.contains("W4A4"),
        "a CPU or cuda-less run under the env arm must fail loudly naming the arm, got: {err}"
    );
    assert!(
        !args.out.join("w4a4_eval.json").exists(),
        "a refused arm must not leave a sidecar behind"
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
fn w4a4_ab_on_a_real_packed_base_scores_the_adapter_under_both_forwards() {
    if std::env::var("NV_Q38_W4A4_AB_TEST").as_deref() != Ok("1") {
        eprintln!("skip: NV_Q38_W4A4_AB_TEST unset");
        return;
    }
    let base = PathBuf::from(std::env::var("NV_Q38_AB_BASE").expect("NV_Q38_AB_BASE"));
    let data = PathBuf::from(std::env::var("NV_Q38_AB_DATA").expect("NV_Q38_AB_DATA"));
    let eval = PathBuf::from(std::env::var("NV_Q38_AB_EVAL_DATA").expect("NV_Q38_AB_EVAL_DATA"));
    let out = PathBuf::from(std::env::var("NV_Q38_AB_OUT").expect("NV_Q38_AB_OUT"));
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
    std::env::set_var("NV_Q38_EVAL_W4A4", "1");
    let device = Device::new_cuda(0).expect("the w4a4 A/B needs the CUDA device");
    let s = run_jsonl(&args, &device).unwrap();
    std::env::remove_var("NV_Q38_EVAL_W4A4");
    assert!(s.all_grads_finite, "a NaN-gradient run is not an A/B datum");
    let sidecar = std::fs::read_to_string(out.join("w4a4_eval.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&sidecar).unwrap();
    for key in [
        "eval_loss_base_bf16",
        "eval_loss_adapter_bf16",
        "eval_loss_base_w4a4",
        "eval_loss_adapter_w4a4",
    ] {
        assert!(
            v[key].as_f64().map(|x| x.is_finite()).unwrap_or(false),
            "sidecar {key} missing or non-finite: {sidecar}"
        );
    }
    let n = v["num_eval_examples"].as_u64().unwrap() as usize;
    for (vec_key, sum_key) in [
        ("eval_ce_base_bf16", "eval_loss_base_bf16"),
        ("eval_ce_adapter_bf16", "eval_loss_adapter_bf16"),
        ("eval_ce_base_w4a4", "eval_loss_base_w4a4"),
        ("eval_ce_adapter_w4a4", "eval_loss_adapter_w4a4"),
    ] {
        let rows = v[vec_key]
            .as_array()
            .unwrap_or_else(|| panic!("sidecar {vec_key} missing: {sidecar}"));
        assert_eq!(
            rows.len(),
            n,
            "sidecar {vec_key} must carry one paired CE per held-out row"
        );
        let sum: f64 = rows.iter().map(|x| x.as_f64().unwrap()).sum();
        let scalar = v[sum_key].as_f64().unwrap();
        assert!(
            (sum - scalar).abs() <= 1e-3 * scalar.abs().max(1.0),
            "sidecar {sum_key} {scalar} is not the sum {sum} of {vec_key}"
        );
    }
    println!("[w4a4-ab] {}", serde_json::to_string(&v).unwrap());
}
