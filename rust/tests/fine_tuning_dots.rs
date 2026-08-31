use speaches_plus::oapi::fine_tuning::{create_job, CreateFineTuningJobRequest, Hyperparameters};

#[test]
fn a_fine_tuning_job_on_a_dots_base_trains_the_qlora_trainer_and_registers_the_adapter() {
    let root = std::env::temp_dir().join(format!("ft-dots-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::env::set_var("NV_FINE_TUNING_DIR", root.join("jobs"));
    std::env::set_var("NV_TRAIN_DEVICE", "cpu");

    let base = root.join("base");
    nv_train::dots_fixture::write_tiny_checkpoint(&base).unwrap();
    let train_dir = root.join("dump-train");
    let eval_dir = root.join("dump-eval");
    nv_train::dots_fixture::write_dump_dir(&train_dir, 3, 3).unwrap();
    nv_train::dots_fixture::write_dump_dir(&eval_dir, 2, 9).unwrap();

    let req = CreateFineTuningJobRequest {
        model: base.display().to_string(),
        training_file: Some(train_dir.display().to_string()),
        training_data: None,
        validation_file: Some(eval_dir.display().to_string()),
        validation_data: None,
        suffix: Some("dotse2e".into()),
        output_dir: None,
        hyperparameters: Some(Hyperparameters {
            rank: Some(4),
            alpha: Some(8.0),
            steps: Some(4),
            n_steps: None,
            lr: Some(1e-2),
            learning_rate: None,
            target: None,
            seed: Some(11),
            grad_layers: Some(2),
            b_lr_ratio: None,
        }),
    };

    let job = create_job(&req).expect("job creation must succeed on a valid tiny dots checkpoint");
    assert_eq!(job.status, "succeeded", "job error: {:?}", job.error);
    let ft = job
        .fine_tuned_model
        .as_deref()
        .expect("a succeeded job must name its adapter");
    assert!(ft.starts_with("ft:base:dotse2e:"), "{ft}");
    assert_eq!(job.hyperparameters.grad_layers, Some(2));
    assert_eq!(job.hyperparameters.b_lr_ratio, Some(1.0));

    let metrics = job.metrics.as_ref().expect("succeeded jobs carry metrics");
    assert_eq!(metrics["trainer"], "dots_qlora_windowed");
    let la = metrics["eval_loss_adapter"].as_f64().unwrap();
    let lb = metrics["eval_loss_base"].as_f64().unwrap();
    assert!(la.is_finite() && lb.is_finite(), "la {la} lb {lb}");
    assert!(
        (la - lb).abs() > 0.0,
        "a trained adapter's held-out CE cannot coincide with the base's; both read {la}"
    );
    assert!(metrics["loss_first"].as_f64().unwrap() > metrics["loss_last"].as_f64().unwrap());
    assert_eq!(metrics["all_grads_finite"], true);

    let modules = metrics["modules"].as_array().unwrap();
    assert!(!modules.is_empty());
    assert!(
        modules
            .iter()
            .all(|m| m.as_str().unwrap().starts_with("language_model.model.layers.")),
        "dots adapters save under the checkpoint's own mlx trunk names: {modules:?}"
    );

    for f in &job.result_files {
        assert!(
            std::path::Path::new(f).exists(),
            "result file {f} is missing"
        );
    }
    let adapter_cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(&job.result_files[0])
                .join(speaches_plus::oapi::lora::ADAPTER_CONFIG_FILE),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        adapter_cfg[speaches_plus::oapi::lora::STAMPED_ID_KEY], ft,
        "run_job must stamp the fine_tuned_model id for restart rediscovery"
    );
    assert!(
        speaches_plus::oapi::lora::registered().get(ft).is_none(),
        "a dots adapter serves through the OCR pipeline, never the chat catalog"
    );
    assert_eq!(metrics["registered"], false);
    assert_eq!(metrics["hot_loaded"], false);
    assert_eq!(metrics["serving_route"], "ocr");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_dots_family_rejects_inline_data_a_jsonl_file_and_every_speculator_kind() {
    let root = std::env::temp_dir().join(format!("ft-dots-rejects-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::env::set_var("NV_FINE_TUNING_DIR", root.join("jobs"));

    let base = root.join("base");
    nv_train::dots_fixture::write_tiny_checkpoint(&base).unwrap();

    let blank = |model: String| CreateFineTuningJobRequest {
        model,
        training_file: None,
        training_data: None,
        validation_file: None,
        validation_data: None,
        suffix: None,
        output_dir: None,
        hyperparameters: None,
    };

    let mut inline = blank(base.display().to_string());
    inline.training_data = Some(vec![serde_json::json!({"text": "x"})]);
    let err = create_job(&inline).expect_err("inline rows cannot carry vision-tower rows");
    assert!(err.1.contains("nvk-dots-vision-dump"), "{}", err.1);

    let jsonl = root.join("rows.jsonl");
    std::fs::write(&jsonl, "{\"text\": \"x\"}\n").unwrap();
    let mut file_req = blank(base.display().to_string());
    file_req.training_file = Some(jsonl.display().to_string());
    let err = create_job(&file_req).expect_err("a jsonl file is not a dump directory");
    assert!(err.1.contains("dump"), "{}", err.1);

    let dump = root.join("dump");
    nv_train::dots_fixture::write_dump_dir(&dump, 1, 1).unwrap();
    for kind in ["mtp", "eagle3"] {
        let mut spec_req = blank(base.display().to_string());
        spec_req.training_file = Some(dump.display().to_string());
        let spec = Some(speaches_plus::oapi::fine_tuning::SpeculatorSpec {
            kind: kind.into(),
            drafter_dir: None,
            rank: None,
            lr_ratio: None,
        });
        let err = speaches_plus::oapi::fine_tuning::create_job_with_speculator(&spec_req, &spec)
            .expect_err("dots ships no speculator to co-train");
        assert!(err.1.contains("dots"), "{kind}: {}", err.1);
    }

    let _ = std::fs::remove_dir_all(&root);
}
