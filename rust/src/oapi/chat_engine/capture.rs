use super::*;
use crate::oapi::capture::{
    capture_record, capture_runs, checked_max_context, CaptureInfo, LayerCaptureRequest,
    LayerCaptureResponse,
};
use candle_core::Tensor;

pub(crate) const CAPTURE_WAITS_120_S_FOR_THE_SERVING_SLOT_BEFORE_A_503_THE_CLIENT_RETRIES:
    std::time::Duration = std::time::Duration::from_secs(120);

pub(crate) fn capture_family(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::Qwen3 => "qwen3",
        ModelFamily::Gemma4 => "gemma4",
        ModelFamily::Gemma4E4b => "gemma4-e4b",
        ModelFamily::Gemma4Moe => "gemma4-moe",
        ModelFamily::Qwen3_5Moe => "qwen3.5",
        ModelFamily::Laguna => "laguna",
        ModelFamily::Omni => "omni",
        ModelFamily::GptOss => "gpt-oss",
        ModelFamily::Qwen4Exp => "qwen4-exp",
    }
}

pub(crate) fn capture_info(engine: &NvEngineChat) -> CaptureInfo {
    let id = engine.inner.model_id.as_str();
    let family = capture_family(engine.inner.family);
    let ok = |n: usize, h: usize| CaptureInfo::available(id, family, n, h);
    let no = |why: &str| CaptureInfo::unavailable(id, family, why);
    match &engine.inner.model {
        LoadedModel::Laguna(m) => ok(m.config().num_hidden_layers, m.config().hidden_size),
        LoadedModel::Gemma4(m) => ok(m.config().num_hidden_layers, m.config().hidden_size),
        LoadedModel::Gemma4Moe(m) => ok(
            m.config().base.num_hidden_layers,
            m.config().base.hidden_size,
        ),
        LoadedModel::Qwen4Exp(m) => ok(m.config().num_hidden_layers, m.config().hidden_size),
        LoadedModel::Qwen3_5Moe(QwenMoeShared::Batch(_)) => no(
            "NV_Q38_BATCH=1 hands the model to the lane scheduler; capture needs the solo eager \
             or graphed engine",
        ),
        LoadedModel::Qwen3_5Moe(_) => match engine.inner.capture_shape {
            Some((n, h)) => ok(n, h),
            None => no("the trunk shape was not recorded at load"),
        },
        LoadedModel::Gemma4E4b(_) => {
            no("the E4B per-layer-input trunk has no residual capture path")
        }
        LoadedModel::Qwen3(_) | LoadedModel::Omni(_) | LoadedModel::GptOss(_) => {
            no("this family has no residual capture path")
        }
    }
}

enum Target {
    Laguna {
        model: LagunaShared,
        _lock: tokio::sync::OwnedRwLockWriteGuard<()>,
    },
    Gemma4 {
        model: Arc<nv_models::gemma4::Gemma4>,
        _permit: tokio::sync::OwnedSemaphorePermit,
    },
    Gemma4Moe(Arc<nv_models::gemma4_moe::Gemma4Moe>),
    Qwen4Exp {
        model: Arc<nv_models::qwen4_exp::Qwen4Exp>,
        _permit: tokio::sync::OwnedSemaphorePermit,
    },
    QwenEager(
        tokio::sync::OwnedMutexGuard<nv_models::qwen3_5_moe::Qwen3Moe>,
        Option<Arc<nv_models::qwen3_5_moe::GroupedMoeDispatch>>,
    ),
    QwenGraphed(tokio::sync::OwnedMutexGuard<nv_models::graph_engine::GraphedQwen3Moe>),
}

enum Scratch {
    Laguna(nv_models::laguna::LagunaStudyCaptureState),
    Gemma4(nv_models::gemma4::Gemma4KvCache),
    Qwen4Exp(nv_models::qwen4_exp::Qwen4ExpState),
    Qwen(nv_models::qwen3_5_moe::Qwen3MoeKvCache),
}

fn chunked_rows(
    ids: &[u32],
    chunk: usize,
    device: &candle_core::Device,
    mut f: impl FnMut(&Tensor, &Tensor) -> anyhow::Result<Tensor>,
) -> anyhow::Result<(Tensor, usize)> {
    let mut rows = Vec::new();
    let mut off = 0usize;
    while off < ids.len() {
        let c = (ids.len() - off).min(chunk);
        let tokens = Tensor::from_vec(ids[off..off + c].to_vec(), (1usize, c), device)?;
        let positions: Vec<i32> = (off as i32..(off + c) as i32).collect();
        let positions = Tensor::from_vec(positions, c, device)?;
        rows.push(f(&tokens, &positions)?);
        off += c;
    }
    Ok((Tensor::cat(&rows, 0)?, rows.len()))
}

impl Target {
    fn scratch(&self, layer: usize, max_context: usize) -> anyhow::Result<Scratch> {
        Ok(match self {
            Target::Laguna { model: m, .. } => Scratch::Laguna(m.new_study_capture_state(layer, max_context)?),
            Target::Gemma4 { model: m, .. } => Scratch::Gemma4(m.new_kv_cache(max_context)?),
            Target::Gemma4Moe(m) => Scratch::Gemma4(m.new_kv_cache(max_context)?),
            Target::Qwen4Exp { model: m, .. } => Scratch::Qwen4Exp(m.new_state(max_context)?),
            Target::QwenEager(m, _) => Scratch::Qwen(m.new_kv_cache(max_context)?),
            Target::QwenGraphed(g) => Scratch::Qwen(g.underlying().new_kv_cache(max_context)?),
        })
    }

    fn rows(
        &self,
        ids: &[u32],
        layer: usize,
        scratch: &mut Scratch,
    ) -> anyhow::Result<(Tensor, usize)> {
        match (self, scratch) {
            (Target::Laguna { model: m, .. }, Scratch::Laguna(state)) => {
                let (rows, _calls, chunks) = m.study_capture_record_tokens(ids, state)?;
                Ok((rows, chunks))
            }
            (Target::Gemma4 { model: m, .. }, Scratch::Gemma4(cache)) => {
                cache.reset();
                chunked_rows(
                    ids,
                    nv_models::gemma4::VERIFY_PREFILL_CHUNK,
                    m.device(),
                    |t, p| m.residual_rows_after_layer(t, p, cache, layer),
                )
            }
            (Target::Gemma4Moe(m), Scratch::Gemma4(cache)) => {
                cache.reset();
                chunked_rows(
                    ids,
                    GEMMA4_MOE_CAPTURE_CHUNK_MATCHES_THE_SERVING_PREFILL_CHUNK,
                    m.device(),
                    |t, p| m.residual_rows_after_layer(t, p, cache, layer),
                )
            }
            (Target::Qwen4Exp { model: m, .. }, Scratch::Qwen4Exp(state)) => {
                let rows = m.residual_rows_after_layer(ids, layer, state)?;
                Ok((rows, ids.len().div_ceil(nv_models::qwen4_exp::PREFILL_CHUNK_2048_IS_THE_PREFIX_CHECKPOINT_GRID_SO_A_RESUMED_PREFILL_CHUNKS_EXACTLY_AS_A_COLD_ONE)))
            }
            (Target::QwenEager(m, d), Scratch::Qwen(cache)) => {
                cache.reset();
                let disp = d
                    .as_deref()
                    .map(|d| d as &dyn nv_models::qwen3_5_moe::MoeDispatch);
                chunked_rows(ids, ids.len(), m.device(), |t, p| {
                    m.residual_rows_after_layer(t, p, cache, disp, layer)
                })
            }
            (Target::QwenGraphed(g), Scratch::Qwen(cache)) => {
                cache.reset();
                let m = g.underlying();
                chunked_rows(ids, ids.len(), m.device(), |t, p| {
                    m.residual_rows_after_layer(t, p, cache, g.moe_dispatch_ref(), layer)
                })
            }
            _ => anyhow::bail!("capture scratch does not match its engine"),
        }
    }
}

pub(crate) const GEMMA4_MOE_CAPTURE_CHUNK_MATCHES_THE_SERVING_PREFILL_CHUNK: usize = 256;

async fn acquire_capture_target(engine: &NvEngineChat) -> anyhow::Result<Target> {
    let wait = CAPTURE_WAITS_120_S_FOR_THE_SERVING_SLOT_BEFORE_A_503_THE_CLIENT_RETRIES;
    Ok(match &engine.inner.model {
        LoadedModel::Laguna(m) => Target::Laguna {
            model: m.clone(),
            _lock: m.1.acquire_exclusive().await?,
        },
        LoadedModel::Gemma4(m) => {
            let g = chat_gate();
            Target::Gemma4 {
                model: m.clone(),
                _permit: acquire_permit_bounded(g.sem.clone(), g.permits, wait).await?,
            }
        }
        LoadedModel::Gemma4Moe(m) => Target::Gemma4Moe(m.clone()),
        LoadedModel::Qwen4Exp(m) => Target::Qwen4Exp {
            model: m.clone(),
            _permit: qwen4_exp_loop::acquire_qwen4_exp_permit(wait).await?,
        },
        LoadedModel::Qwen3_5Moe(QwenMoeShared::Eager(m)) => Target::QwenEager(
            m.clone().lock_owned().await,
            engine.inner.qwen_moe_dispatch.clone(),
        ),
        LoadedModel::Qwen3_5Moe(QwenMoeShared::Graphed(g)) => {
            Target::QwenGraphed(g.clone().lock_owned().await)
        }
        _ => {
            capture_info(engine).shape()?;
            anyhow::bail!("{}", crate::oapi::capture::CAPTURE_NOT_AVAILABLE)
        }
    })
}

pub(crate) fn bos_prefix(family: ModelFamily, bos: Option<u32>) -> Option<u32> {
    match family {
        ModelFamily::Gemma4 | ModelFamily::Gemma4Moe => bos,
        _ => None,
    }
}

pub(crate) async fn capture_layer(
    engine: &NvEngineChat,
    req: LayerCaptureRequest,
) -> anyhow::Result<LayerCaptureResponse> {
    let (num_layers, hidden_size) = capture_info(engine).shape()?;
    let mut limits = vec![engine.inner.kv_max_seq_len];
    match &engine.inner.model {
        LoadedModel::Laguna(m) => limits.push(m.config().max_position_embeddings),
        LoadedModel::Qwen4Exp(m) => limits.push(m.config().max_position_embeddings),
        _ => {}
    }
    let max_context = checked_max_context(&req, num_layers, &limits)?;
    let runs = capture_runs(
        &engine.inner.tokenizer,
        &req,
        bos_prefix(engine.inner.family, engine.inner.bos_token_id),
        max_context,
    )?;
    let target = acquire_capture_target(engine).await?;
    let layer = req.layer;
    let want_tokens = req.tokens;
    let start = std::time::Instant::now();
    let records = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<_>> {
        let mut scratch = target.scratch(layer, max_context)?;
        let mut out = Vec::with_capacity(runs.len());
        for (ids, skip) in runs {
            let (rows, chunks) = target.rows(&ids, layer, &mut scratch)?;
            let rows = rows.narrow(0, skip, ids.len() - skip)?;
            out.push(capture_record(
                ids[skip..].to_vec(),
                &rows,
                chunks,
                want_tokens,
            )?);
        }
        drop(scratch);
        drop(target);
        Ok(out)
    })
    .await??;
    Ok(LayerCaptureResponse {
        model: engine.inner.model_id.clone(),
        layer,
        num_layers,
        hidden_size,
        records,
        seconds: start.elapsed().as_secs_f64(),
    })
}
