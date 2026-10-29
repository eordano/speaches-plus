use super::*;
use candle_core::backprop::GradStore;

pub(super) fn enabled() -> bool {
    std::env::var("NV_LAGUNA_TRAIN_CHECKPOINT").as_deref() == Ok("1")
}

const HEAD_TOKEN_CHUNK: usize = 256;

fn head_loss_and_gradient(
    model: &LagunaTrainModel,
    hidden: &Tensor,
    row: &TokenRow,
    chunk: usize,
    backward: bool,
) -> Result<(f32, Option<Tensor>)> {
    anyhow::ensure!(
        chunk > 0 && row.ids.len() > 1,
        "head loss needs a nonempty chunk and targets"
    );
    let mask = row.target_mask_where_predicting_a_prompt_token_weighs_zero();
    let count: f32 = mask.iter().sum();
    anyhow::ensure!(count > 0.0, "every target position is masked");
    let mut value = 0.0;
    let mut gradients = Vec::new();
    for start in (0..mask.len()).step_by(chunk) {
        let len = chunk.min(mask.len() - start);
        let slice = hidden.narrow(1, start, len)?.contiguous()?;
        let local_count: f32 = mask[start..start + len].iter().sum();
        if local_count == 0.0 {
            if backward {
                gradients.push(slice.zeros_like()?);
            }
            continue;
        }
        let input = Var::from_tensor(&slice.detach())?;
        let logits = model.final_logits(&input)?.squeeze(0)?;
        let targets = Tensor::from_vec(
            row.ids[start + 1..start + len + 1].to_vec(),
            len,
            hidden.device(),
        )?;
        let loss = ce_mean_over_unmasked_positions(&logits, &targets, &mask[start..start + len])?
            .affine(local_count as f64 / count as f64, 0.0)?;
        value += loss.to_scalar::<f32>()?;
        if backward {
            let mut grads = loss.backward()?;
            gradients.push(
                grads
                    .remove(&input)
                    .context("missing chunked head gradient")?
                    .detach(),
            );
        }
    }
    let gradient = if backward {
        gradients.push(hidden.narrow(1, mask.len(), 1)?.zeros_like()?);
        Some(Tensor::cat(&gradients, 1)?)
    } else {
        None
    };
    Ok((value, gradient))
}

pub(super) fn eval_row_ce(model: &LagunaTrainModel, row: &TokenRow) -> Result<f32> {
    let seq = row.ids.len();
    let dev = model.device();
    let tokens = Tensor::from_vec(row.ids.clone(), (1usize, seq), dev)?;
    let positions = Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), seq, dev)?;
    let mut hidden = model.embed_tokens(&tokens)?.detach();
    for layer in &model.layers {
        hidden = model
            .forward_layer(layer, &hidden, &positions, seq)?
            .detach();
    }
    Ok(head_loss_and_gradient(model, &hidden, row, HEAD_TOKEN_CHUNK, false)?.0)
}

pub(super) fn row_gradients(
    model: &LagunaTrainModel,
    row: &TokenRow,
    vars: &[Var],
) -> Result<(f32, GradStore)> {
    let dev = model.device();
    let seq = row.ids.len();
    let tokens = Tensor::from_vec(row.ids.clone(), (1usize, seq), dev)?;
    let positions = Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), seq, dev)?;
    let mut boundaries = vec![model.embed_tokens(&tokens)?.detach()];
    nv_train::train_prof::phase(dev, "checkpoint_forward_row", || {
        for layer in &model.layers {
            let output = model.forward_layer(layer, boundaries.last().unwrap(), &positions, seq)?;
            boundaries.push(output.detach());
        }
        Ok(())
    })?;
    let (value, incoming) = head_loss_and_gradient(
        model,
        boundaries.last().unwrap(),
        row,
        HEAD_TOKEN_CHUNK,
        true,
    )?;
    let mut incoming = incoming.context("missing head boundary gradient")?;
    let mut result = GradStore::default();
    nv_train::train_prof::phase(dev, "checkpoint_backward_row", || {
        for (index, layer) in model.layers.iter().enumerate().rev() {
            boundaries.pop();
            let input = Var::from_tensor(boundaries.last().context("missing layer checkpoint")?)?;
            let output = model.forward_layer(layer, &input, &positions, seq)?;
            let local_loss = output.mul(&incoming)?.sum_all()?;
            let mut grads = local_loss.backward()?;
            for var in vars {
                if let Some(gradient) = grads.remove(var) {
                    anyhow::ensure!(
                        result.get(var).is_none(),
                        "adapter variable shared between checkpoint layers"
                    );
                    result.insert(var, gradient.detach());
                }
            }
            if index > 0 {
                incoming = grads
                    .remove(&input)
                    .context("missing checkpoint input gradient")?
                    .detach();
            }
        }
        Ok(())
    })?;
    for var in vars {
        anyhow::ensure!(
            result.get(var).is_some(),
            "checkpoint backward omitted an adapter variable"
        );
    }
    Ok((value, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_head_and_checkpointed_graph_preserve_the_masked_loss_and_every_gradient() -> Result<()> {
        let root = std::env::temp_dir().join(format!("laguna-chunked-head-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        write_tiny_laguna_checkpoint(&root);
        let model = LagunaTrainModel::load(&root, &Device::Cpu)?;
        let hidden = Var::from_tensor(
            &Tensor::arange(0f32, 9.0 * model.config.hidden_size as f32, &Device::Cpu)?
                .reshape((1, 9, model.config.hidden_size))?
                .affine(0.01, -0.2)?,
        )?;
        for completion_start in [0, 5, 8] {
            let row = TokenRow {
                ids: vec![1, 4, 7, 2, 9, 3, 8, 5, 6],
                completion_start,
            };
            let loss = row_loss_from_logits(&model.final_logits(&hidden)?, &row)?;
            let value = loss.to_scalar::<f32>()?;
            let expected = loss
                .backward()?
                .remove(&hidden)
                .context("missing reference hidden gradient")?;
            for chunk in [1, 3, 7, 256] {
                let (got, gradient) = head_loss_and_gradient(&model, &hidden, &row, chunk, true)?;
                assert!(
                    (got - value).abs() < 1e-5,
                    "chunk {chunk}, mask {completion_start}: loss mismatch"
                );
                let error = (gradient.unwrap() - &expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(
                    error < 1e-5,
                    "chunk {chunk}, mask {completion_start}: hidden gradient error {error}"
                );
                let (eval, gradient) = head_loss_and_gradient(&model, &hidden, &row, chunk, false)?;
                assert_eq!(eval, got);
                assert!(gradient.is_none());
            }
        }
        let device = Device::Cpu;
        let sites = build_sites(&model, &LagunaTrainArgs::default_targets(), 37)?;
        let mut vars = Vec::new();
        for site in &sites {
            let lora = Arc::new(LoraTrainable::from_features(
                site.lin.out_features(),
                site.lin.in_features(),
                LoraConfig {
                    r: 4,
                    alpha: 4.0,
                    dropout: 0.0,
                },
                site.seed,
                &device,
            )?);
            let trainable = lora.trainable_vars();
            trainable[1].set(&trainable[1].ones_like()?.affine(0.025, 0.0)?)?;
            vars.extend(trainable);
            site.lin
                .attach_lora(Arc::new(TrainingLoraHook::new(lora, true)))?;
        }
        for completion_start in [0, 5] {
            let row = TokenRow {
                ids: vec![1, 4, 7, 2, 9, 3, 8, 5, 6],
                completion_start,
            };
            let loss = solo_row_ce(&model, &row)?;
            let expected_loss = loss.to_scalar::<f32>()?;
            assert!(
                (eval_row_ce(&model, &row)? - expected_loss).abs() < 1e-6,
                "detached evaluation must preserve the masked objective"
            );
            let expected = loss.backward()?;
            let (actual_loss, actual) = row_gradients(&model, &row, &vars)?;
            assert!(
                (actual_loss - expected_loss).abs() < 1e-6,
                "checkpoint forward must preserve the masked objective"
            );
            for (index, var) in vars.iter().enumerate() {
                let want = expected.get(var).context("reference gradient missing")?;
                let got = actual.get(var).context("checkpoint gradient missing")?;
                let error = (want - got)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                let scale = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                assert!(
                    error <= 1e-6 + scale * 1e-4,
                    "adapter {index}: max gradient error {error}, reference scale {scale}"
                );
            }
        }
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
