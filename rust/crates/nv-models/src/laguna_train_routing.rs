use anyhow::{ensure, Result};
use candle_core::{DType, Tensor};
use std::sync::Arc;

use crate::laguna::LagunaMoe;

pub const THE_EXPERT_LIST_IS_EMPTY_ONCE_THE_GROUPED_STACKS_OWN_THE_WEIGHTS: usize = 0;

pub enum RouteIds {
    Host(Vec<u32>),
    Device(Tensor),
}

impl RouteIds {
    fn host(&self) -> Result<Vec<u32>> {
        Ok(match self {
            RouteIds::Host(v) => v.clone(),
            RouteIds::Device(t) => t.flatten_all()?.to_vec1::<u32>()?,
        })
    }
}

fn device_route_default_on_0_restores_the_host_topk() -> bool {
    nv_config::env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call("NV_TRAIN_MOE_DEVICE_ROUTE")
}

fn selected_weights(moe: &LagunaMoe, x: &Tensor) -> Result<(RouteIds, Tensor)> {
    let (tokens, hidden) = x.dims2()?;
    ensure!(
        tokens > 0 && hidden > 0,
        "training MoE input must be nonempty"
    );
    ensure!(
        moe.num_experts > 0
            && (moe.experts.len() == moe.num_experts
                || THE_EXPERT_LIST_IS_EMPTY_ONCE_THE_GROUPED_STACKS_OWN_THE_WEIGHTS
                    == moe.experts.len())
            && moe.top_k > 0
            && moe.top_k <= moe.num_experts,
        "training MoE has incompatible expert count or top-k"
    );
    ensure!(
        hidden == moe.gate.in_features()
            && moe.gate.out_features() == moe.num_experts
            && (moe.selection_bias.dims() == [moe.num_experts]
                || moe.selection_bias.dims() == [1, moe.num_experts]),
        "training MoE router dimensions disagree"
    );
    let selections = tokens
        .checked_mul(moe.top_k)
        .ok_or_else(|| anyhow::anyhow!("training MoE route count overflow"))?;
    ensure!(
        selections <= u32::MAX as usize && moe.num_experts <= u32::MAX as usize,
        "training MoE indices exceed u32"
    );
    ensure!(
        moe.softcap.is_finite() && moe.routed_scaling.is_finite(),
        "training MoE softcap and routed scaling must be finite"
    );
    let mut logits = moe.gate.forward(x)?.to_dtype(DType::F32)?.contiguous()?;
    if moe.softcap > 0.0 {
        logits = logits
            .affine(1.0 / moe.softcap as f64, 0.0)?
            .tanh()?
            .affine(moe.softcap as f64, 0.0)?;
    }
    let scores = candle_nn::ops::sigmoid(&logits)?;
    #[cfg(feature = "cuda")]
    if matches!(x.device(), candle_core::Device::Cuda(_)) && device_route_default_on_0_restores_the_host_topk() {
        let _ = nv_layers::moe_train::BLOCK8_ALIGNMENT_ON_THE_DEVICE_WHEN_THE_IDS_ARE_THERE_SO_THE_ROUTER_NEVER_DRAINS_THE_STREAM;
        let ids = nv_layers::moe_grouped::route_topk_ids_on_the_device_as_u32_tokens_by_k(
            &logits.detach(),
            Some(&moe.selection_bias),
            tokens,
            moe.num_experts,
            moe.top_k,
            1,
            0.0,
            x.device(),
        )?;
        let mut weights = scores.gather(&ids, 1)?.contiguous()?;
        if moe.norm_topk {
            let sums = weights.sum_keepdim(1)?;
            let denominator = sums.gt(0f64)?.where_cond(&sums, &sums.ones_like()?)?;
            weights = weights.broadcast_div(&denominator)?;
        }
        return Ok((RouteIds::Device(ids), weights));
    }
    let scores_host = scores.detach().flatten_all()?.to_vec1::<f32>()?;
    let bias = moe
        .selection_bias
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    ensure!(
        scores_host.iter().chain(&bias).all(|v| v.is_finite()),
        "training MoE router produced nonfinite scores or selection bias"
    );
    let mut ids = Vec::with_capacity(selections);
    let mut selection = vec![0f32; moe.num_experts];
    for row in scores_host.chunks_exact(moe.num_experts) {
        for (e, value) in selection.iter_mut().enumerate() {
            *value = row[e] + bias[e];
            ensure!(value.is_finite(), "training MoE selection score overflow");
        }
        for _ in 0..moe.top_k {
            let mut best = f32::NEG_INFINITY;
            let mut best_id = None;
            for (e, &value) in selection.iter().enumerate() {
                if value > best {
                    best = value;
                    best_id = Some(e);
                }
            }
            let e =
                best_id.ok_or_else(|| anyhow::anyhow!("training MoE has no selectable expert"))?;
            ids.push(e as u32);
            selection[e] = f32::NEG_INFINITY;
        }
    }
    let indices = Tensor::from_vec(ids.clone(), (tokens, moe.top_k), x.device())?;
    let mut weights = scores.gather(&indices, 1)?.contiguous()?;
    if moe.norm_topk {
        let sums = weights.sum_keepdim(1)?;
        let denominator = sums.gt(0f64)?.where_cond(&sums, &sums.ones_like()?)?;
        weights = weights.broadcast_div(&denominator)?;
    }
    Ok((RouteIds::Host(ids), weights))
}

pub const ONE_GATHER_PER_LAYER_BECAUSE_INDEX_ADD_PER_EXPERT_KEPT_A_FULL_TOKENS_X_HIDDEN_COPY_ON_THE_GRAPH_FOR_EVERY_TOUCHED_EXPERT_AND_A_SCATTER_WITH_REPEATED_ROWS_HAS_NO_DUPLICATE_CONTRACT: usize = 1;

pub const NV_TRAIN_MOE_GROUPED_ENV_DEFAULT_ON_0_RESTORES_THE_PER_EXPERT_QUANT_LINEAR_PATH: &str =
    "NV_TRAIN_MOE_GROUPED";

#[cfg(feature = "cuda")]
pub(crate) fn grouped_train_stacks(
    moe: &LagunaMoe,
    hidden: usize,
    device: &candle_core::Device,
) -> Option<Arc<nv_layers::moe_train::MarlinMoeTrainStacks>> {
    let candle_core::Device::Cuda(dev) = device else {
        return None;
    };
    if !nv_config::env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(
        NV_TRAIN_MOE_GROUPED_ENV_DEFAULT_ON_0_RESTORES_THE_PER_EXPERT_QUANT_LINEAR_PATH,
    ) {
        return None;
    }
    let mut slot = moe.train_stacks.lock().ok()?;
    if let Some(s) = slot.as_ref() {
        return Some(s.clone());
    }
    let inter = moe.experts.first()?.gate_proj().out_features();
    let stream = nv_layers::cuda_stream::current_stream(dev);
    let t0 = std::time::Instant::now();
    match nv_layers::moe_train::MarlinMoeTrainStacks::build_from_experts(
        &moe.experts,
        hidden,
        inter,
        &stream,
    ) {
        Ok(st) => {
            eprintln!(
                "[moe_train] grouped expert stacks built: experts={} hidden={hidden} inter={inter} in {:.1} s",
                st.experts,
                t0.elapsed().as_secs_f64()
            );
            let st = Arc::new(st);
            *slot = Some(st.clone());
            Some(st)
        }
        Err(e) => {
            eprintln!("[moe_train] grouped expert stacks unavailable ({e:#}); per-expert path");
            None
        }
    }
}

pub(crate) fn forward(moe: &LagunaMoe, x_flat: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    let span_stream = match x_flat.device() {
        candle_core::Device::Cuda(d) => Some(nv_layers::cuda_stream::current_stream(d)),
        _ => None,
    };
    #[cfg(feature = "cuda")]
    let router_span = span_stream.as_ref().and_then(|s| nv_layers::prof_events::span("fwd_moe_router_chain_with_host_topk", s));
    let (top_ids, top_weights) = selected_weights(moe, x_flat)?;
    #[cfg(feature = "cuda")]
    drop(router_span);
    let (tokens, hidden) = x_flat.dims2()?;
    let device = x_flat.device();
    #[cfg(feature = "cuda")]
    if let Some(stacks) = grouped_train_stacks(moe, hidden, device) {
        let _ = nv_layers::moe_train::GROUPED_EXPERT_TRAINING_RUNS_FOUR_MARLIN_MOE_LAUNCHES_PER_LAYER_PER_DIRECTION_INSTEAD_OF_THREE_PER_TOUCHED_EXPERT;
        let op = match &top_ids {
            RouteIds::Device(ids) => nv_layers::moe_train::MoeExpertsMarlinWeightedTrainOp::new_with_device_ids(
                stacks, ids.clone(), tokens, moe.top_k,
            ),
            RouteIds::Host(ids) => nv_layers::moe_train::MoeExpertsMarlinWeightedTrainOp::new(
                stacks, ids, tokens, moe.top_k,
            ),
        };
        let w = top_weights
            .flatten_all()?
            .to_dtype(DType::F32)?
            .reshape((tokens, moe.top_k))?
            .contiguous()?;
        let weighted_span = span_stream.as_ref().and_then(|s| nv_layers::prof_events::span("fwd_moe_weighted_op_with_cast", s));
        let mut acc = x_flat
            .to_dtype(DType::BF16)?
            .contiguous()?
            .apply_op2(&w, op)?;
        drop(weighted_span);
        let shared_span = span_stream.as_ref().and_then(|s| nv_layers::prof_events::span("fwd_moe_shared_expert_and_combine", s));
        if moe.routed_scaling != 1.0 {
            acc = acc.affine(moe.routed_scaling as f64, 0.0)?;
        }
        let shared = moe.shared_expert.forward(x_flat)?.to_dtype(DType::F32)?;
        let out = acc.add(&shared)?;
        drop(shared_span);
        return Ok(out);
    }
    let top_ids = top_ids.host()?;
    let mut expert_rows: Vec<Vec<u32>> = vec![Vec::new(); moe.num_experts];
    let mut expert_weight_indices: Vec<Vec<u32>> = vec![Vec::new(); moe.num_experts];
    for (slot, &expert) in top_ids.iter().enumerate() {
        expert_rows[expert as usize].push((slot / moe.top_k) as u32);
        expert_weight_indices[expert as usize].push(slot as u32);
    }
    let weights = top_weights.flatten_all()?;
    let _ = ONE_GATHER_PER_LAYER_BECAUSE_INDEX_ADD_PER_EXPERT_KEPT_A_FULL_TOKENS_X_HIDDEN_COPY_ON_THE_GRAPH_FOR_EVERY_TOUCHED_EXPERT_AND_A_SCATTER_WITH_REPEATED_ROWS_HAS_NO_DUPLICATE_CONTRACT;
    let mut weighted_rows_in_expert_order: Vec<Tensor> = Vec::new();
    let mut stacked_position_of_slot: Vec<u32> = vec![u32::MAX; top_ids.len()];
    let mut stacked_len = 0u32;
    for (e, rows) in expert_rows.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let indices = Tensor::from_vec(rows.clone(), rows.len(), device)?;
        let gathered = x_flat.index_select(&indices, 0)?.contiguous()?;
        let y = moe.experts[e].forward(&gathered)?.to_dtype(DType::F32)?;
        let weight_indices =
            Tensor::from_vec(expert_weight_indices[e].clone(), rows.len(), device)?;
        let w = weights
            .index_select(&weight_indices, 0)?
            .reshape((rows.len(), 1))?;
        weighted_rows_in_expert_order.push(y.broadcast_mul(&w)?);
        for &slot in &expert_weight_indices[e] {
            stacked_position_of_slot[slot as usize] = stacked_len;
            stacked_len += 1;
        }
    }
    anyhow::ensure!(
        stacked_len as usize == top_ids.len()
            && stacked_position_of_slot.iter().all(|&p| p != u32::MAX),
        "laguna routing: {} stacked expert rows do not cover the {} (token, slot) assignments",
        stacked_len,
        top_ids.len()
    );
    let stacked = Tensor::cat(&weighted_rows_in_expert_order, 0)?;
    let perm = Tensor::from_vec(stacked_position_of_slot, top_ids.len(), device)?;
    let mut acc = stacked
        .index_select(&perm, 0)?
        .reshape((tokens, moe.top_k, hidden))?
        .sum(1)?;
    if moe.routed_scaling != 1.0 {
        acc = acc.affine(moe.routed_scaling as f64, 0.0)?;
    }
    let shared = moe.shared_expert.forward(x_flat)?.to_dtype(DType::F32)?;
    Ok(acc.add(&shared)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};
    use nv_layers::{linear::Linear, mlp::Mlp};

    const ROUTER: [f32; 6] = [0.9, -0.2, -0.3, 0.8, 0.4, 0.5];
    const BIAS: [f32; 3] = [0.1, -0.03, 0.02];
    const X: [f32; 6] = [0.7, -0.4, -0.5, 0.9, 0.3, 0.6];
    const U: [f32; 6] = [0.7, -0.8, 0.4, 0.3, -0.6, 0.9];

    fn matrices(e: usize) -> [[f32; 4]; 3] {
        let f = e as f32 + 1.0;
        [
            [0.2 * f, -0.3, 0.5, 0.1 * f],
            [0.4, 0.2 * f, -0.1 * f, 0.6],
            [0.7 * f, -0.2, 0.3, 0.4 * f],
        ]
    }

    fn mlp(e: usize) -> Mlp {
        let [g, u, d] = matrices(e).map(|w| {
            Linear::new(
                Tensor::from_vec(w.to_vec(), (2, 2), &Device::Cpu).unwrap(),
                None,
            )
            .unwrap()
        });
        Mlp::new(g, u, d).unwrap()
    }

    fn fixture(norm_topk: bool, softcap: f32) -> LagunaMoe {
        LagunaMoe {
            num_experts: 3,
            top_k: 2,
            norm_topk,
            routed_scaling: 1.3,
            softcap,
            gate: Linear::new(
                Tensor::from_vec(ROUTER.to_vec(), (3, 2), &Device::Cpu).unwrap(),
                None,
            )
            .unwrap(),
            selection_bias: Tensor::from_vec(BIAS.to_vec(), (1, 3), &Device::Cpu).unwrap(),
            experts: (0..3).map(mlp).collect(),
            shared_expert: mlp(3),
            #[cfg(feature = "cuda")]
            grouped: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            train_stacks: std::sync::Mutex::new(None),
        }
    }

    fn expert64(e: usize, x: &[f64]) -> [f64; 2] {
        let [g, u, d] = matrices(e);
        let dot = |w: &[f32], row: usize| w[row * 2] as f64 * x[0] + w[row * 2 + 1] as f64 * x[1];
        let a: [f64; 2] = std::array::from_fn(|row| {
            let z = dot(&g, row);
            z / (1.0 + (-z).exp()) * dot(&u, row)
        });
        std::array::from_fn(|row| d[row * 2] as f64 * a[0] + d[row * 2 + 1] as f64 * a[1])
    }

    fn oracle(x: &[f64], norm: bool, cap: f32) -> (f64, Vec<u32>) {
        let mut loss = 0.0;
        let mut all_ids = Vec::new();
        for (t, row) in x.chunks_exact(2).enumerate() {
            let scores: [f64; 3] = std::array::from_fn(|e| {
                let z = ROUTER[e * 2] as f64 * row[0] + ROUTER[e * 2 + 1] as f64 * row[1];
                let z = if cap > 0.0 {
                    (z / cap as f64).tanh() * cap as f64
                } else {
                    z
                };
                1.0 / (1.0 + (-z).exp())
            });
            let mut ids = [0usize, 1, 2];
            ids.sort_by(|&a, &b| {
                (scores[b] + BIAS[b] as f64)
                    .total_cmp(&(scores[a] + BIAS[a] as f64))
                    .then(a.cmp(&b))
            });
            let sum = if norm {
                scores[ids[0]] + scores[ids[1]]
            } else {
                1.0
            };
            let mut out = expert64(3, row);
            for &e in &ids[..2] {
                all_ids.push(e as u32);
                let y = expert64(e, row);
                for j in 0..2 {
                    out[j] += (1.3f32 as f64) * scores[e] / sum * y[j];
                }
            }
            loss += out[0] * U[t * 2] as f64 + out[1] * U[t * 2 + 1] as f64;
        }
        (loss, all_ids)
    }

    #[test]
    fn selected_router_gradients_match_full_moe_finite_differences() {
        for norm in [false, true] {
            for cap in [0.0, 1.5] {
                let moe = fixture(norm, cap);
                let x = Var::from_vec(X.to_vec(), (3, 2), &Device::Cpu).unwrap();
                let u = Tensor::from_vec(U.to_vec(), (3, 2), &Device::Cpu).unwrap();
                let loss = forward(&moe, x.as_tensor())
                    .unwrap()
                    .mul(&u)
                    .unwrap()
                    .sum_all()
                    .unwrap();
                let grads = loss.backward().unwrap();
                let dx = grads
                    .get(&x)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let values: Vec<f64> = X.iter().map(|&v| v as f64).collect();
                let (expected, ids) = oracle(&values, norm, cap);
                assert!((loss.to_scalar::<f32>().unwrap() as f64 - expected).abs() < 1e-5);
                assert_eq!(selected_weights(&moe, x.as_tensor()).unwrap().0.host().unwrap(), ids);
                assert_eq!(moe.route_host(&x.as_tensor().detach(), 3).unwrap().0, ids);
                for i in 0..values.len() {
                    let mut plus = values.clone();
                    let mut minus = values.clone();
                    plus[i] += 1e-4;
                    minus[i] -= 1e-4;
                    let (lp, ip) = oracle(&plus, norm, cap);
                    let (lm, im) = oracle(&minus, norm, cap);
                    assert_eq!(ip, ids, "finite difference crossed a top-k boundary");
                    assert_eq!(im, ids, "finite difference crossed a top-k boundary");
                    let fd = (lp - lm) / 2e-4;
                    assert!(dx[i].is_finite());
                    assert!(
                        (dx[i] as f64 - fd).abs() < 2e-4 + 0.004 * fd.abs(),
                        "norm={norm} cap={cap} input={i}: {} != {fd}",
                        dx[i]
                    );
                }
                let frozen = moe
                    .forward(x.as_tensor())
                    .unwrap()
                    .mul(&u)
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .backward()
                    .unwrap();
                let frozen_dx = frozen
                    .get(&x)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                assert!(
                    dx.iter().zip(frozen_dx).any(|(a, b)| (*a - b).abs() > 1e-4),
                    "router contribution is absent from the discriminator"
                );
            }
        }
    }

    #[test]
    fn selection_bias_only_ranks_experts_and_ties_choose_lower_ids() {
        let mut moe = fixture(false, 0.0);
        let x = Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap();
        moe.selection_bias = Tensor::zeros(3, DType::F32, &Device::Cpu).unwrap();
        let (ids, w) = selected_weights(&moe, &x).unwrap();
        let ids = ids.host().unwrap();
        assert_eq!(ids, [0, 1]);
        assert_eq!(
            w.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [0.5, 0.5]
        );
        moe.selection_bias = Tensor::from_vec(vec![2f32, -1.0, 3.0], 3, &Device::Cpu).unwrap();
        let (ids, w) = selected_weights(&moe, &x).unwrap();
        let ids = ids.host().unwrap();
        assert_eq!(ids, [2, 0]);
        assert_eq!(
            w.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [0.5, 0.5]
        );
    }

    #[test]
    fn zero_selected_scores_do_not_divide_by_zero() {
        let mut moe = fixture(true, 0.0);
        moe.gate =
            Linear::new(Tensor::full(-1000f32, (3, 2), &Device::Cpu).unwrap(), None).unwrap();
        let x = Var::from_vec(vec![1f32, 1.0], (1, 2), &Device::Cpu).unwrap();
        let (_, w) = selected_weights(&moe, x.as_tensor()).unwrap();
        assert_eq!(
            w.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [0.0, 0.0]
        );
        let y = forward(&moe, x.as_tensor()).unwrap();
        assert_eq!(
            y.to_vec2::<f32>().unwrap(),
            moe.shared_expert
                .forward(x.as_tensor())
                .unwrap()
                .to_vec2::<f32>()
                .unwrap()
        );
        let grads = y.sum_all().unwrap().backward().unwrap();
        assert!(grads
            .get(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn malformed_and_nonfinite_routes_return_errors() {
        let x = Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap();
        let mut moe = fixture(true, 0.0);
        moe.top_k = 0;
        assert!(forward(&moe, &x).is_err());
        moe.top_k = 4;
        assert!(forward(&moe, &x).is_err());
        moe.top_k = 2;
        moe.selection_bias = Tensor::from_vec(vec![0f32, f32::NAN, 0.0], 3, &Device::Cpu).unwrap();
        assert!(forward(&moe, &x).is_err());
        moe.selection_bias = Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap();
        assert!(forward(&moe, &x).is_err());
        moe.selection_bias = Tensor::zeros(3, DType::F32, &Device::Cpu).unwrap();
        let invalid = Tensor::from_vec(vec![f32::NAN, 0.0], (1, 2), &Device::Cpu).unwrap();
        assert!(forward(&moe, &invalid).is_err());
        moe.experts.pop();
        assert!(forward(&moe, &x).is_err());
    }
}
