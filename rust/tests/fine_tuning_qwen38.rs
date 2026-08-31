use speaches_plus::oapi::fine_tuning::{create_job, CreateFineTuningJobRequest, Hyperparameters};

const TRAIN_ROWS: [&str; 3] = [
    "The battle system alternates a top down command map with an action phase.",
    "Critics praised the art direction and the score of the third entry.",
    "A localisation was never announced, and fan translations circulated instead.",
];

const EVAL_ROWS: [&str; 2] = [
    "The engine reuses assets from the second entry, which shortened development.",
    "Later ports added a training mode and a gallery of illustrations.",
];

#[test]
fn a_fine_tuning_job_on_a_qwen38_base_trains_the_windowed_trainer_and_reports_the_gate_losses() {
    let root = std::env::temp_dir().join(format!("ft-q38-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::env::set_var("NV_FINE_TUNING_DIR", root.join("jobs"));
    std::env::remove_var("NV_TRAIN_DEVICE");

    let base = root.join("base");
    nv_train::qwen38_fixture::write_tiny_checkpoint(&base);

    let req = CreateFineTuningJobRequest {
        model: base.display().to_string(),
        training_file: None,
        training_data: Some(
            TRAIN_ROWS
                .iter()
                .map(|t| serde_json::json!({ "text": t }))
                .chain(std::iter::once(serde_json::json!({
                    "prompt": "Q: which entry got a localisation? A:",
                    "completion": " none was announced; fan translations circulated instead."
                })))
                .collect(),
        ),
        validation_file: None,
        validation_data: Some(
            EVAL_ROWS
                .iter()
                .map(|t| serde_json::json!({ "text": t }))
                .collect(),
        ),
        suffix: Some("q38e2e".into()),
        output_dir: None,
        hyperparameters: Some(Hyperparameters {
            rank: Some(4),
            alpha: Some(4.0),
            steps: Some(4),
            n_steps: None,
            lr: Some(5e-3),
            learning_rate: None,
            target: None,
            seed: Some(11),
            grad_layers: None,
            b_lr_ratio: None,
        }),
    };

    let job = create_job(&req).expect("job creation must succeed on a valid tiny checkpoint");
    assert_eq!(
        job.status,
        "succeeded",
        "job error: {:?}",
        job.error
    );
    let ft = job
        .fine_tuned_model
        .as_deref()
        .expect("a succeeded job must name its adapter");
    assert!(ft.starts_with("ft:base:q38e2e:"), "{ft}");

    assert_eq!(
        job.hyperparameters.grad_layers,
        Some(4),
        "the default window depth is the GDN-bounded 4 and must surface on the stored job"
    );

    let metrics = job.metrics.as_ref().expect("succeeded jobs carry metrics");
    assert_eq!(metrics["trainer"], "qwen38_windowed");
    assert_eq!(
        metrics["checkpointed"], false,
        "grad_layers 4 sits at the checkpoint threshold, so the default job trains on the \
         plain full-graph path"
    );
    let la = metrics["eval_loss_adapter"]
        .as_f64()
        .expect("the roll-over gate reads eval_loss_adapter from the qwen path");
    let lb = metrics["eval_loss_base"]
        .as_f64()
        .expect("the roll-over gate reads eval_loss_base from the qwen path");
    assert!(la.is_finite() && lb.is_finite(), "la {la} lb {lb}");
    assert!(
        (la - lb).abs() > 0.0,
        "a trained adapter's held-out CE cannot coincide with the base's; both read {la}"
    );
    assert!(metrics["loss_first"].as_f64().unwrap() > metrics["loss_last"].as_f64().unwrap());

    let modules = metrics["modules"].as_array().unwrap();
    assert!(!modules.is_empty());
    assert!(
        modules
            .iter()
            .any(|m| m.as_str().unwrap().contains("linear_attn")),
        "qwen defaults include the GDN projections because qwen3_5_dense_wgpu now carries \
         deltanet lora sites; a default-target adapter over a window with a linear-attention \
         layer must adapt it: {modules:?}"
    );

    for f in &job.result_files {
        assert!(
            std::path::Path::new(f).exists(),
            "result file {f} is missing"
        );
    }
    let adapter_cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(std::path::Path::new(&job.result_files[0]).join(
            speaches_plus::oapi::lora::ADAPTER_CONFIG_FILE,
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        adapter_cfg[speaches_plus::oapi::lora::STAMPED_ID_KEY], ft,
        "run_job must stamp the fine_tuned_model id for restart rediscovery"
    );
    assert!(
        speaches_plus::oapi::lora::registered().get(ft).is_some(),
        "the qwen path must still register the adapter in the catalog"
    );
    assert_eq!(metrics["registered"], true);
    let _ = std::fs::remove_dir_all(&root);
}
