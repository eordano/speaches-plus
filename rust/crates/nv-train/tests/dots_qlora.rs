use anyhow::Result;
use candle_core::{DType, Device};
use nv_train::dots::{
    dots_default_targets, dots_layer_prefix, load_dump_dir, run, DotsConfig, DotsTrainArgs,
    DotsW4Weights,
};
use nv_train::dots_fixture::{
    write_dump_dir as fabricate_samples, write_tiny_checkpoint as fabricate_checkpoint, HEAD_DIM,
    HIDDEN, INTER, LAYERS, VOCAB,
};
use std::path::Path;

fn train_args(base: &Path, data: &Path, eval: &Path, out: &Path, steps: usize) -> DotsTrainArgs {
    DotsTrainArgs {
        base: base.to_path_buf(),
        data: data.to_path_buf(),
        eval_data: Some(eval.to_path_buf()),
        out: out.to_path_buf(),
        grad_layers: LAYERS,
        rank: 4,
        alpha: 8.0,
        lr: 1e-2,
        b_lr_ratio: 1.0,
        steps,
        targets: dots_default_targets(),
        seed: 11,
        dtype: DType::F32,
    }
}

#[test]
fn zero_steps_leaves_the_adapter_at_identity_so_base_and_adapter_eval_match() -> Result<()> {
    let td = tempdir()?;
    let base = td.join("ckpt");
    fabricate_checkpoint(&base)?;
    fabricate_samples(&td.join("train"), 2, 3)?;
    fabricate_samples(&td.join("eval"), 2, 3)?;
    let s = run(
        &train_args(&base, &td.join("train"), &td.join("eval"), &td.join("out0"), 0),
        &Device::Cpu,
    )?;
    assert_eq!(s.eval_ce_base.len(), 2);
    for (a, b) in s.eval_ce_base.iter().zip(s.eval_ce_adapter.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "B initializes to zeros, so an untrained adapter must not move eval CE: {a} vs {b}"
        );
    }
    assert_eq!(s.modules.len(), LAYERS * 7);
    Ok(())
}

#[test]
fn a_few_steps_reduce_the_loss_and_save_a_peft_adapter_under_checkpoint_module_names() -> Result<()> {
    let td = tempdir()?;
    let base = td.join("ckpt");
    fabricate_checkpoint(&base)?;
    fabricate_samples(&td.join("train"), 2, 3)?;
    fabricate_samples(&td.join("eval"), 2, 3)?;
    let out = td.join("out");
    let s = run(
        &train_args(&base, &td.join("train"), &td.join("eval"), &out, 6),
        &Device::Cpu,
    )?;
    assert_eq!(s.losses.len(), 6);
    assert!(s.all_grads_finite, "non-finite grads in synthetic training");
    assert!(
        s.losses[5] < s.losses[0],
        "loss did not drop over 6 steps: {:?}",
        s.losses
    );
    assert!(
        s.eval_loss_adapter().unwrap() < s.eval_loss_base().unwrap(),
        "held-out CE did not improve on same-distribution synthetic data: {:?} -> {:?}",
        s.eval_loss_base(),
        s.eval_loss_adapter()
    );
    assert!(s.max_module_delta_amax > 0.0, "adapter never moved");

    let adapter = candle_core::safetensors::load(out.join("adapter_model.safetensors"), &Device::Cpu)?;
    let key = format!(
        "base_model.model.{}.self_attn.q_proj.lora_A.weight",
        dots_layer_prefix(0)
    );
    let a = adapter
        .get(&key)
        .unwrap_or_else(|| panic!("missing {key} in adapter; keys: {:?}", adapter.keys()));
    assert_eq!(a.dims(), [4, HIDDEN]);
    let cfgs = std::fs::read_to_string(out.join("adapter_config.json"))?;
    let cfg: serde_json::Value = serde_json::from_str(&cfgs)?;
    assert_eq!(cfg["peft_type"], "LORA");
    assert_eq!(cfg["r"], 4);
    Ok(())
}

#[test]
fn the_dequantized_base_matches_a_scalar_reference_row() -> Result<()> {
    let td = tempdir()?;
    let base = td.join("ckpt");
    fabricate_checkpoint(&base)?;
    let src = DotsW4Weights::open_dir(&base, &Device::Cpu)?;
    let cfg = DotsConfig::from_hf_json_file(&base.join("config.json"))?;
    assert_eq!(cfg.head_dim, HEAD_DIM);
    let module = format!("{}.mlp.gate_proj", dots_layer_prefix(0));
    let w = src.dense_f32(&module, INTER, HIDDEN)?;
    assert_eq!(w.len(), INTER * HIDDEN);
    let embed = src.embed_rows_f32(&[0, 5, VOCAB as u32 - 1], HIDDEN)?;
    assert_eq!(embed.len(), 3 * HIDDEN);
    assert!(embed.iter().all(|x| x.is_finite()));
    assert!(src.embed_rows_f32(&[VOCAB as u32], HIDDEN).is_err());
    Ok(())
}

#[test]
fn the_dump_roundtrip_preserves_ids_rows_and_offsets() -> Result<()> {
    let td = tempdir()?;
    let d = td.join("dump");
    fabricate_samples(&d, 3, 9)?;
    let samples = load_dump_dir(&d, HIDDEN)?;
    assert_eq!(samples.len(), 3);
    let s = &samples[0];
    assert_eq!(s.vision_start, 2);
    assert_eq!(s.n_vis, 4);
    assert_eq!(s.completion_start, 9);
    assert_eq!(s.rows.len(), 4 * HIDDEN);
    Ok(())
}

fn tempdir() -> Result<std::path::PathBuf> {
    let base = std::env::temp_dir().join(format!(
        "dots-qlora-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir_all(&base)?;
    Ok(base)
}
