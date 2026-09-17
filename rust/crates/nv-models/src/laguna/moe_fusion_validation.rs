use super::*;
use anyhow::{Context, Result};
use cudarc::driver::CudaStream;
use half::bf16;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Seek, Write};
use std::sync::Arc;

std::thread_local! {
    static CAPTURE: RefCell<Option<BTreeMap<usize, Vec<bf16>>>> = const { RefCell::new(None) };
}

pub(super) fn capture_input(layer: usize, input: &Tensor, rows: usize) -> Result<()> {
    if rows != 3 {
        return Ok(());
    }
    CAPTURE.with(|slot| -> Result<()> {
        let mut slot = slot.borrow_mut();
        if let Some(values) = slot.as_mut() {
            let words = input.flatten_all()?.to_vec1::<bf16>()?;
            anyhow::ensure!(
                words.iter().all(|v| v.is_finite()),
                "nonfinite real hidden input"
            );
            anyhow::ensure!(
                values.insert(layer, words).is_none(),
                "duplicate layer capture"
            );
        }
        Ok(())
    })
}

fn host(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?)
}

fn words(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn blob(file: &mut File, bytes: &[u8], dtype: &str, shape: &[usize]) -> Result<Value> {
    let offset = file.stream_position()?;
    file.write_all(bytes)?;
    Ok(json!({"offset": offset, "bytes": bytes.len(), "dtype": dtype, "shape": shape}))
}

fn difference(a: &[f32], b: &[f32]) -> Value {
    assert_eq!(a.len(), b.len());
    let mut error = Vec::new();
    let (mut squared, mut reference, mut different) = (0.0f64, 0.0f64, 0usize);
    for (&a, &b) in a.iter().zip(b) {
        different += usize::from(a.to_bits() != b.to_bits());
        if a.is_finite() && b.is_finite() {
            let d = b as f64 - a as f64;
            squared += d * d;
            reference += (a as f64) * (a as f64);
            error.push(d.abs());
        }
    }
    error.sort_by(f64::total_cmp);
    let percentile = |p: usize| error.get(error.len().saturating_sub(1) * p / 100).copied();
    json!({"values": a.len(), "different_words": different,
        "finite_reference": a.iter().filter(|v| v.is_finite()).count(),
        "finite_candidate": b.iter().filter(|v| v.is_finite()).count(),
        "relative_l2": if reference > 0.0 { Some((squared/reference).sqrt()) } else { None },
        "absolute_l2": squared.sqrt(), "abs_p50": percentile(50),
        "abs_p95": percentile(95), "abs_p99": percentile(99), "abs_max": error.last()})
}

struct Arm {
    output: Tensor,
    values: Vec<f32>,
    ids: Vec<i32>,
    weights: Vec<f32>,
    fused: usize,
    shared_ids: Vec<i32>,
    shared_weights: Vec<f32>,
}

fn arm(
    model: &Laguna,
    layer: usize,
    moe: &LagunaMoe,
    input: &Tensor,
    m: usize,
    fused: bool,
    stream: &Arc<CudaStream>,
) -> Result<Arm> {
    model.take_batch_moe_fused_dispatches();
    let output = model
        .moe_forward_verify_grouped(layer, moe, input, m, fused)?
        .context("real packed grouped route unavailable")?;
    stream.synchronize()?;
    let values = host(&output)?;
    let map = model
        .moe_verify_ctx
        .lock()
        .map_err(|e| anyhow::anyhow!("context lock: {e}"))?;
    let ctx = map.get(&m).context("context absent after actual forward")?;
    #[allow(deprecated)]
    let ids = stream.memcpy_dtov(&ctx.topk_ids)?;
    #[allow(deprecated)]
    let weights = stream.memcpy_dtov(&ctx.topk_weights)?;
    let expected_fold = std::env::var("NV_LAGUNA_PREPARED_SHARED_FOLD").as_deref() == Ok("1");
    anyhow::ensure!(
        ctx.folds_shared() == expected_fold,
        "diagnostic must exercise the requested shared-fold policy"
    );
    let stride = moe.top_k + usize::from(expected_fold);
    anyhow::ensure!(
        ids.len() == m * stride && weights.len() == m * stride,
        "actual route stride differs"
    );
    let mut routed_ids = Vec::with_capacity(m * moe.top_k);
    let mut routed_weights = Vec::with_capacity(m * moe.top_k);
    let mut shared_ids = Vec::new();
    let mut shared_weights = Vec::new();
    for row in 0..m {
        let start = row * stride;
        routed_ids.extend_from_slice(&ids[start..start + moe.top_k]);
        routed_weights.extend_from_slice(&weights[start..start + moe.top_k]);
        if expected_fold {
            let tail = start + moe.top_k;
            anyhow::ensure!(
                ids[tail] == moe.num_experts as i32 && weights[tail].to_bits() == 1.0f32.to_bits(),
                "folded shared tail must be expert E with exact weight one"
            );
            shared_ids.push(ids[tail]);
            shared_weights.push(weights[tail]);
        }
    }
    let ids = routed_ids;
    let weights = routed_weights;
    let observed = model.take_batch_moe_fused_dispatches();
    anyhow::ensure!(
        observed == usize::from(fused),
        "actual fused engagement {observed} != {}",
        usize::from(fused)
    );
    Ok(Arm {
        output,
        values,
        ids,
        weights,
        fused: observed,
        shared_ids,
        shared_weights,
    })
}

fn route_difference(a: &Arm, b: &Arm, m: usize, k: usize) -> Value {
    let overlaps: Vec<usize> = (0..m)
        .map(|row| {
            a.ids[row * k..(row + 1) * k]
                .iter()
                .filter(|id| b.ids[row * k..(row + 1) * k].contains(id))
                .count()
        })
        .collect();
    let mut reference = Vec::new();
    let mut candidate = Vec::new();
    for row in 0..m {
        for slot in 0..k {
            let index = row * k + slot;
            if let Some(other) = b.ids[row * k..(row + 1) * k]
                .iter()
                .position(|id| *id == a.ids[index])
            {
                reference.push(a.weights[index]);
                candidate.push(b.weights[row * k + other]);
            }
        }
    }
    json!({"topk_overlap_per_row": overlaps, "matched_expert_weight_delta": difference(&reference, &candidate)})
}

fn compare(
    model: &Laguna,
    layer: usize,
    input_words: &[bf16],
    m: usize,
    case: &str,
    input_blob: Value,
    report: &mut File,
    outputs: &mut File,
) -> Result<()> {
    let LagunaFfn::Moe(moe) = &model.layers[layer].ffn else {
        anyhow::bail!("captured non-MoE layer")
    };
    let Device::Cuda(dev) = model.device() else {
        anyhow::bail!("real CUDA required")
    };
    let stream = dev.cuda_stream();
    anyhow::ensure!(
        !stream.cu_stream().is_null() && !nv_layers::cuda_stream::is_overridden(),
        "actual explicit model stream required"
    );
    let h = model.config.hidden_size;
    let input = Tensor::from_vec(input_words.to_vec(), (m, h), model.device())?;
    let shared = host(&moe.shared_expert.forward(&input)?)?;
    let before = arm(model, layer, moe, &input, m, false, &stream)?;
    let fused = arm(model, layer, moe, &input, m, true, &stream)?;
    let routed_owned = {
        let map = model
            .moe_verify_ctx
            .lock()
            .map_err(|e| anyhow::anyhow!("context lock: {e}"))?;
        map[&m].copy_scattered_f32_into_owned_tensor(model.device(), &stream)?
    };
    let routed_words = words(&host(&routed_owned)?);
    let after = arm(model, layer, moe, &input, m, false, &stream)?;
    let poisoned: Vec<bf16> = input_words
        .iter()
        .map(|v| bf16::from_bits(v.to_bits() ^ 0x8000))
        .collect();
    let overwrite = Tensor::from_vec(poisoned, (m, h), model.device())?;
    let overwritten = arm(model, layer, moe, &overwrite, m, true, &stream)?;
    {
        let mut map = model
            .moe_verify_ctx
            .lock()
            .map_err(|e| anyhow::anyhow!("context lock: {e}"))?;
        stream.memcpy_htod(
            &vec![12345.0f32; m * h],
            &mut map.get_mut(&m).unwrap().y_acc,
        )?;
    }
    stream.synchronize()?;
    let retained = [&before, &fused, &after]
        .iter()
        .map(|a| host(&a.output).map(|now| words(&now) == words(&a.values)))
        .collect::<Result<Vec<_>>>()?;
    let routed_retained = words(&host(&routed_owned)?) == routed_words;
    let mut arms = Vec::new();
    for (label, arm) in [
        ("generic_before", &before),
        ("fused", &fused),
        ("generic_after", &after),
    ] {
        let bytes: Vec<u8> = arm.values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let output = blob(outputs, &bytes, "f32le", &[m, h])?;
        arms.push(json!({"arm": label, "output_blob": output, "route_ids": arm.ids,
            "route_weight_bits": words(&arm.weights), "fused_dispatches": arm.fused,
            "shared_tail_ids": arm.shared_ids, "shared_tail_weight_bits": words(&arm.shared_weights)}));
    }
    let finite = [&before, &fused, &after, &overwritten]
        .iter()
        .all(|a| a.values.iter().chain(&a.weights).all(|v| v.is_finite()))
        && shared.iter().all(|v| v.is_finite());
    let valid_routes = [&before, &fused, &after, &overwritten].iter().all(|a| {
        a.ids.len() == m * moe.top_k
            && a.weights.len() == m * moe.top_k
            && a.ids.chunks_exact(moe.top_k).all(|ids| {
                ids.iter().enumerate().all(|(i, id)| {
                    *id >= 0 && (*id as usize) < moe.num_experts && !ids[..i].contains(id)
                })
            })
    });
    let feature_argmax = |a: &[f32]| -> Vec<Option<u32>> {
        a.chunks_exact(h)
            .map(nv_layers::sampler::argmax_checked)
            .collect()
    };
    let row = json!({"kind":"layer_comparison", "case":case, "layer":layer, "m":m,
        "input_blob":input_blob, "prepared_shared_fold":!fused.shared_ids.is_empty(), "mode":1, "softcap":moe.softcap, "norm_topk":moe.norm_topk,
        "routed_scaling":moe.routed_scaling, "correction_bias_bits":words(&host(&moe.selection_bias)?),
        "shared_output_bits":words(&shared), "arms":arms,
        "generic_fused":difference(&before.values,&fused.values),
        "null_control":difference(&before.values,&after.values),
        "route_comparison":route_difference(&before,&fused,m,moe.top_k),
        "null_route_comparison":route_difference(&before,&after,m,moe.top_k),
        "feature_argmax_generic":feature_argmax(&before.values),
        "feature_argmax_fused":feature_argmax(&fused.values),
        "argmax_basis":"hidden features, not vocabulary tokens", "outputs_retained":retained,
        "routed_owned_copy_retained":routed_retained, "finite":finite, "routes_valid":valid_routes,
        "overwrite_fused_dispatches":overwritten.fused, "quality_threshold":Value::Null});
    writeln!(report, "{}", row)?;
    report.flush()?;
    outputs.flush()?;
    anyhow::ensure!(
        finite && valid_routes,
        "nonfinite values or invalid routes; diagnostics retained"
    );
    anyhow::ensure!(
        retained.iter().all(|v| *v) && routed_retained,
        "owned output changed after scratch overwrite"
    );
    Ok(())
}

pub(crate) fn run(model: &Laguna, tokenizer: &tokenizers::Tokenizer) -> Result<()> {
    anyhow::ensure!(
        std::env::var("NV_LAGUNA_MOE_FUSION_TEST").as_deref() == Ok("1"),
        "explicit real-weight diagnostic opt-in required"
    );
    anyhow::ensure!(
        std::env::var("NV_LAGUNA_SPEC_MOE_FUSED").as_deref() == Ok("1"),
        "candidate arm must be explicitly selected"
    );
    anyhow::ensure!(
        std::env::var_os("NV_LAGUNA_HOST_VERIFY").is_none() && !model.host_moe(),
        "device verify routing required"
    );
    model.set_device_verify_routing(true);
    let out = std::path::PathBuf::from(std::env::var("NV_LAGUNA_MOE_FUSION_REPORT_DIR")?);
    std::fs::create_dir(&out)?;
    let create = |name: &str| -> Result<File> {
        Ok(OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(out.join(name))?)
    };
    let mut report = create("comparisons.jsonl")?;
    let mut inputs = create("inputs.bf16le")?;
    let mut outputs = create("outputs.f32le")?;
    let expected: Vec<usize> = model
        .layers
        .iter()
        .enumerate()
        .filter_map(|(i, l)| matches!(l.ffn, LagunaFfn::Moe(_)).then_some(i))
        .collect();
    anyhow::ensure!(!expected.is_empty(), "real target has no routed layers");
    for &layer in &expected {
        let LagunaFfn::Moe(moe) = &model.layers[layer].ffn else {
            unreachable!()
        };
        let bank = model
            .grouped_weights(moe)?
            .context("native packed bank required")?;
        anyhow::ensure!(
            (
                bank.hidden_size,
                bank.intermediate_size,
                bank.num_experts,
                moe.top_k
            ) == (2048, 512, 256, 8),
            "pinned XS geometry required"
        );
        anyhow::ensure!(
            [
                moe.shared_expert.gate_proj(),
                moe.shared_expert.up_proj(),
                moe.shared_expert.down_proj()
            ]
            .iter()
            .all(|linear| linear.nvfp4_parts().is_some()),
            "shared expert must remain native NVFP4"
        );
    }
    let expected_fold = std::env::var("NV_LAGUNA_PREPARED_SHARED_FOLD").as_deref() == Ok("1");
    anyhow::ensure!(
        !expected_fold || nv_layers::moe_grouped::folded_shared_enabled(),
        "folded diagnostic requires shared payloads in grouped banks"
    );
    let mut captured = Vec::new();
    for (lane, text) in [
        "A copper sphere rolls across the wooden table. ",
        "Explain why each cache retains its own history. ",
    ]
    .into_iter()
    .enumerate()
    {
        let ids = tokenizer
            .encode(text.repeat(128), true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec();
        anyhow::ensure!(ids.len() >= 259, "source prompt is too short");
        writeln!(
            report,
            "{}",
            json!({"kind":"capture_source","lane":lane,"prefix_token_ids":&ids[..256],"suffix_token_ids":&ids[256..259],"source_kv":"bf16","source_route":"generic device grouped MoE"})
        )?;
        report.flush()?;
        let mut cache = model.new_kv_cache(1024)?;
        let tokens = Tensor::from_vec(ids[..256].to_vec(), (1, 256), model.device())?;
        let pos = Tensor::from_vec((0..256i32).collect::<Vec<_>>(), 256, model.device())?;
        let _ = model.forward_hidden_with_cache_aux(&tokens, &pos, &mut cache, &[])?;
        CAPTURE.with(|slot| *slot.borrow_mut() = Some(BTreeMap::new()));
        let tokens = Tensor::from_vec(ids[256..259].to_vec(), (1, 3), model.device())?;
        let pos = Tensor::from_vec(vec![256i32, 257, 258], 3, model.device())?;
        let result = model.forward_hidden_with_cache_aux(&tokens, &pos, &mut cache, &[]);
        let values = CAPTURE.with(|slot| slot.borrow_mut().take().expect("capture armed"));
        result?;
        anyhow::ensure!(
            values.keys().copied().collect::<Vec<_>>() == expected,
            "real model execution did not capture every routed layer"
        );
        anyhow::ensure!(
            model.device_verify_routing(),
            "real hidden acquisition fell back from device routing"
        );
        captured.push(values);
    }
    let h = model.config.hidden_size;
    for layer in &expected {
        let mut refs = Vec::new();
        for lane in 0..2 {
            let data = &captured[lane][layer];
            anyhow::ensure!(data.len() == 3 * h, "real input shape differs");
            let bytes: Vec<u8> = data
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            let reference = blob(&mut inputs, &bytes, "bf16le", &[3, h])?;
            compare(
                model,
                *layer,
                data,
                3,
                &format!("real_lane_{lane}"),
                reference.clone(),
                &mut report,
                &mut outputs,
            )?;
            refs.push(reference);
        }
        let joined: Vec<bf16> = captured[0][layer]
            .iter()
            .chain(&captured[1][layer])
            .copied()
            .collect();
        compare(
            model,
            *layer,
            &joined,
            6,
            "two_real_lanes",
            json!({"concatenated":refs,"shape":[6,h]}),
            &mut report,
            &mut outputs,
        )?;
    }
    let summary = json!({"kind":"completed","layers":expected.len(),"cases":expected.len()*3,
        "m3_cases":expected.len()*2,"m6_cases":expected.len(),"comparison_fused_dispatches":expected.len()*3,
        "overwrite_fused_dispatches":expected.len()*3,"real_prefix_tokens_per_lane":256,"suffix_rows_per_lane":3,
        "source":"actual model attention/residual/post-attention norm, frozen before MoE",
        "prepared_shared_fold":expected_fold,"quality_threshold":Value::Null,"model_quality_accepted":false,"performance_measurement":false});
    writeln!(report, "{}", summary)?;
    report.flush()?;
    inputs.flush()?;
    outputs.flush()?;
    eprintln!("LAGUNA-M3-M6-MOE-DIAGNOSTIC passed layers={} cases={} finite=true owned_outputs=true fused_engagement=true quality_budget=none",expected.len(),expected.len()*3);
    Ok(())
}
