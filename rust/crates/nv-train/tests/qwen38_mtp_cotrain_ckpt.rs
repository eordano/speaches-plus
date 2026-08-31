use candle_core::Device;
use nv_train::qwen38::{run_jsonl_with_mtp_cotrain, Q38JsonlArgs, Q38MtpCoTrainArgs};
use nv_train::qwen38_fixture::write_tiny_checkpoint;
use std::path::PathBuf;

mod mtp_fixture;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nv-train-mtpckpt-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn the_checkpointed_window_hands_the_head_the_same_detached_output_as_the_plain_graph() {
    let root = scratch("parity");
    let base = root.join("base");
    let cfg = write_tiny_checkpoint(&base);
    mtp_fixture::write_tiny_mtp_head(&base, &cfg);
    let data = root.join("train.jsonl");
    std::fs::write(
        &data,
        format!(
            "{}\n{}\n",
            serde_json::json!({"text": "the battle system alternates a command map"}),
            serde_json::json!({"prompt": "Q: was it localised? A:", "completion": " no, fan translations circulated"}),
        ),
    )
    .unwrap();

    let args = |out: &str| Q38JsonlArgs {
        base: base.clone(),
        data: data.clone(),
        eval_data: None,
        out: root.join(out),
        grad_layers: 3,
        rank: 4,
        alpha: 4.0,
        targets: vec!["q_proj".into(), "v_proj".into(), "down_proj".into()],
        steps: 4,
        lr: 5e-3,
        b_lr_ratio: 1.0,
        seed: 17,
    };
    let co = Q38MtpCoTrainArgs {
        rank: 4,
        alpha: 4.0,
        speculator_lr_ratio: 4.0,
    };

    std::env::set_var("NV_TRAIN_CKPT", "0");
    let (plain, plain_co) =
        run_jsonl_with_mtp_cotrain(&args("adapter-plain"), &co, &Device::Cpu).unwrap();
    std::env::set_var("NV_TRAIN_CKPT", "1");
    let (ckpt, ckpt_co) =
        run_jsonl_with_mtp_cotrain(&args("adapter-ckpt"), &co, &Device::Cpu).unwrap();
    std::env::remove_var("NV_TRAIN_CKPT");

    assert!(!plain.checkpointed && ckpt.checkpointed);
    assert_eq!(plain_co.head_losses.len(), ckpt_co.head_losses.len());
    for (step, (a, b)) in plain_co
        .head_losses
        .iter()
        .zip(ckpt_co.head_losses.iter())
        .enumerate()
    {
        let rel = (a - b).abs() / a.abs().max(1e-6);
        assert!(
            rel < 1e-4,
            "step {step}: plain head CE {a} vs checkpointed {b} (rel {rel:.3e}); the \
             checkpointed backward is feeding the head a different window output than the exact \
             graph, so co-training under checkpointing would train the drafter on hiddens the \
             served trunk never produces"
        );
    }
    for (step, (a, b)) in plain.losses.iter().zip(ckpt.losses.iter()).enumerate() {
        let rel = (a - b).abs() / a.abs().max(1e-6);
        assert!(rel < 1e-4, "step {step}: trunk {a} vs {b} (rel {rel:.3e})");
    }
    assert!(plain_co.all_head_grads_finite && ckpt_co.all_head_grads_finite);
    assert!(
        ckpt_co
            .speculator_dir
            .join("adapter_model.safetensors")
            .is_file(),
        "the checkpointed arm must ship the same speculator pair"
    );
    let _ = std::fs::remove_dir_all(&root);
}
