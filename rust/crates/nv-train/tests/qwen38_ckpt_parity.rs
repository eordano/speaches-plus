use candle_core::Device;
use nv_train::qwen38::{run_jsonl, Q38JsonlArgs};
use nv_train::qwen38_fixture::write_tiny_checkpoint;
use std::path::PathBuf;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nv-train-ckpt-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn the_checkpointed_window_backward_reproduces_the_plain_graph_losses_on_the_same_seed() {
    let root = scratch("parity");
    let base = root.join("base");
    write_tiny_checkpoint(&base);
    let data = root.join("train.jsonl");
    std::fs::write(
        &data,
        format!(
            "{}\n{}\n{}\n",
            serde_json::json!({"text": "the battle system alternates a command map"}),
            serde_json::json!({"prompt": "Q: was it localised? A:", "completion": " no, fan translations circulated"}),
            serde_json::json!({"text": "critics praised the art direction and the score"}),
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
        targets: vec![
            "q_proj".into(),
            "v_proj".into(),
            "in_proj_qkv".into(),
            "down_proj".into(),
        ],
        steps: 4,
        lr: 5e-3,
        b_lr_ratio: 1.0,
        seed: 17,
    };

    std::env::set_var("NV_TRAIN_CKPT", "0");
    let plain = run_jsonl(&args("adapter-plain"), &Device::Cpu).unwrap();
    std::env::set_var("NV_TRAIN_CKPT", "1");
    let ckpt = run_jsonl(&args("adapter-ckpt"), &Device::Cpu).unwrap();
    std::env::remove_var("NV_TRAIN_CKPT");

    assert!(!plain.checkpointed, "NV_TRAIN_CKPT=0 must force the plain graph");
    assert!(ckpt.checkpointed, "NV_TRAIN_CKPT=1 must force the checkpointed backward");
    assert_eq!(plain.losses.len(), ckpt.losses.len());
    for (step, (a, b)) in plain.losses.iter().zip(ckpt.losses.iter()).enumerate() {
        let rel = (a - b).abs() / a.abs().max(1e-6);
        assert!(
            rel < 1e-4,
            "step {step}: plain {a} vs checkpointed {b} (rel {rel:.3e}); the checkpointed \
             backward is optimising a different function than the exact graph"
        );
    }
    assert!(plain.all_grads_finite && ckpt.all_grads_finite);
    assert!(
        plain.losses.first().unwrap() > plain.losses.last().unwrap(),
        "the fixed-batch overfit run must reduce the loss; {:?}",
        plain.losses
    );
    let _ = std::fs::remove_dir_all(&root);
}
