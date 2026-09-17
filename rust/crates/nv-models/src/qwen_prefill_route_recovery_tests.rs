use super::*;
use half::bf16;
use std::{cell::Cell, marker::PhantomData, rc::Rc};

thread_local! {
    static FAIL_REPLACEMENT: Cell<bool> = const { Cell::new(false) };
}

const INJECTED: &str = "injected Qwen prefill replacement failure after graph invalidation";
const CHUNK: usize = 4096;

pub(super) fn fail_replacement_if_armed() -> Result<()> {
    anyhow::ensure!(!FAIL_REPLACEMENT.with(|flag| flag.replace(false)), INJECTED);
    Ok(())
}

struct FaultGuard(PhantomData<Rc<()>>);

impl FaultGuard {
    fn arm() -> Self {
        assert!(!FAIL_REPLACEMENT.with(|flag| flag.replace(true)));
        Self(PhantomData)
    }
}

impl Drop for FaultGuard {
    fn drop(&mut self) {
        FAIL_REPLACEMENT.with(|flag| flag.set(false));
    }
}

#[test]
fn replacement_fault_is_thread_local_one_shot_and_unwind_safe() {
    {
        let _fault = FaultGuard::arm();
        assert!(std::thread::spawn(|| fail_replacement_if_armed().is_ok())
            .join()
            .unwrap());
        assert_eq!(
            fail_replacement_if_armed().unwrap_err().to_string(),
            INJECTED
        );
        assert!(fail_replacement_if_armed().is_ok());
    }
    let failure: Result<()> = (|| {
        let _fault = FaultGuard::arm();
        anyhow::bail!("early test error")
    })();
    assert!(failure.is_err());
    assert!(fail_replacement_if_armed().is_ok());
    assert!(std::panic::catch_unwind(|| {
        let _fault = FaultGuard::arm();
        panic!("test unwind");
    })
    .is_err());
    assert!(fail_replacement_if_armed().is_ok());
}

struct SessionGuard<'a>(&'a Device, Qwen38PrefillGraphSession);

impl Drop for SessionGuard<'_> {
    fn drop(&mut self) {
        let _ = self.0.synchronize();
        self.1.invalidate();
    }
}

fn hidden_bits(t: &Tensor) -> Result<Vec<u16>> {
    anyhow::ensure!(t.dtype() == DType::BF16, "hidden rows must remain BF16");
    let rows = t.flatten_all()?.to_vec1::<bf16>()?;
    anyhow::ensure!(
        rows.iter().all(|x| x.to_f32().is_finite()),
        "nonfinite hidden"
    );
    anyhow::ensure!(rows.iter().any(|x| x.to_f32() != 0.), "zero hidden");
    Ok(rows.into_iter().map(bf16::to_bits).collect())
}

fn logits_bits(t: &Tensor) -> Result<Vec<u32>> {
    anyhow::ensure!(t.dims3()?.1 == 1, "expected one last-row logit tensor");
    let row = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    anyhow::ensure!(row.iter().all(|x| x.is_finite()), "nonfinite logits");
    anyhow::ensure!(row.iter().any(|x| *x != 0.), "zero logits");
    Ok(row.into_iter().map(f32::to_bits).collect())
}

fn same<T: Eq + std::fmt::Debug>(a: &[T], b: &[T], label: &str) -> Result<()> {
    anyhow::ensure!(
        a.len() == b.len(),
        "{label}: length {} != {}",
        a.len(),
        b.len()
    );
    if let Some((i, (a, b))) = a.iter().zip(b).enumerate().find(|(_, (a, b))| a != b) {
        anyhow::bail!("{label}: first mismatch at {i}: {a:?} != {b:?}");
    }
    Ok(())
}

fn inputs(ids: &[u32], start: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    Ok((
        Tensor::from_vec(ids.to_vec(), (1, ids.len()), device)?,
        Tensor::from_vec(
            (start as i32..(start + ids.len()) as i32).collect::<Vec<_>>(),
            ids.len(),
            device,
        )?,
    ))
}

fn first_decode(model: &Qwen3Moe, cache: &mut Qwen3MoeKvCache) -> Result<Vec<u32>> {
    let (tokens, positions) = inputs(&[4242], cache.current_len(), &model.device)?;
    logits_bits(&model.forward_with_cache_dispatched_rows(
        &tokens,
        &positions,
        cache,
        None,
        Some(1),
    )?)
}

struct Reference {
    hidden: Vec<Vec<u16>>,
    logits: Vec<u32>,
    decode: Vec<u32>,
}

fn eager_reference(model: &Qwen3Moe, ids: &[u32]) -> Result<Reference> {
    let mut cache = model.new_kv_cache(ids.len() + 8)?;
    let mut hidden = Vec::new();
    let mut last = None;
    for piece in ids.chunks(CHUNK) {
        let (tokens, positions) = inputs(piece, cache.current_len(), &model.device)?;
        let (logits, rows) = model.forward_with_cache_dispatched_hidden_rows(
            &tokens,
            &positions,
            &mut cache,
            None,
            Some(1),
        )?;
        hidden.push(hidden_bits(&rows)?);
        last = Some(logits_bits(&logits)?);
    }
    anyhow::ensure!(cache.current_len() == ids.len(), "eager frontier");
    Ok(Reference {
        hidden,
        logits: last.unwrap(),
        decode: first_decode(model, &mut cache)?,
    })
}

fn select_route(depth: usize) {
    nv_layers::linear::nvfp4_cfx_set_prefill_total_tokens_hint(depth);
    assert_eq!(
        nv_layers::linear::nvfp4_cfx_effective_route_selected_for_current_prefill_hint(),
        depth == 32768,
        "required shallow/deep CFX route was overridden"
    );
}

#[test]
#[ignore = "requires real dense Qwen27B NVFP4, explicit CUDA stream and NV_QWEN38_SERVING_TEST=1"]
fn qwen38_cuda_prefill_route_failure_recovers_cold_then_recaptures() -> Result<()> {
    anyhow::ensure!(
        std::env::var("NV_QWEN38_SERVING_TEST").as_deref() == Ok("1"),
        "GPU gate must execute"
    );
    anyhow::ensure!(
        q38_prefill_chunk_tokens_from_env() == Some(CHUNK),
        "requires chunk4096"
    );
    anyhow::ensure!(prefill_graph_capture_hostile_env_because_the_inner_gdn_graph_arm_nests_a_capture_and_the_linattn_profiler_syncs_every_lap().is_none(), "unset capture-hostile env");
    for key in [
        "NV_Q38_FP8_DEQUANT_BF16",
        "NV_Q38_MLP_DEQUANT_BF16",
        "NV_Q38_GDN_AB_FP8",
    ] {
        anyhow::ensure!(
            std::env::var(key).as_deref() != Ok("1"),
            "unexpected precision arm {key}"
        );
    }
    select_route(8192);
    select_route(32768);
    let device = Device::new_cuda_with_stream(0)?;
    let dir = std::path::PathBuf::from(std::env::var("NV_QWEN38_DIR")?);
    let raw = std::fs::read_to_string(dir.join("config.json"))?;
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw)?;
    let quant = QuantizationConfig::from_hf_json_str(&raw)?;
    anyhow::ensure!(
        quant.scheme == nv_weights::QuantScheme::Nvfp4,
        "requires NVFP4 checkpoint"
    );
    let model = {
        let weights = WeightLoader::open_dir(&dir, &device)?;
        Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &quant, &device)?
    };
    anyhow::ensure!(
        model.is_dense() && model.config.hidden_size == 5120 && model.layers.len() == 64,
        "requires dense27B geometry"
    );
    let packed = model
        .layers
        .iter()
        .filter(|l| match &l.ffn {
            LayerFfn::Dense(m) => [m.gate_proj(), m.up_proj(), m.down_proj()]
                .iter()
                .all(|p| p.nvfp4_parts().is_some()),
            _ => false,
        })
        .count();
    anyhow::ensure!(
        packed >= 56,
        "requires packed NVFP4 trunk, got {packed} layers"
    );
    let mut cache = model.new_kv_cache(32768 + 8)?;
    assert_eq!(cache.lin_attn_states.len(), 48);
    let mut session = SessionGuard(&device, model.new_prefill_graph_session(CHUNK)?);
    let original_stage = session.1.tok_stage.id();
    for (stage, depth) in [8192, 8192, 32768, 8192].into_iter().enumerate() {
        if stage == 1 {
            device.synchronize()?;
            cache.reset();
            assert_eq!(session.1.captured_segment_count(), 16);
            select_route(32768);
            let fault = FaultGuard::arm();
            let failed = model.prefill_chunks_graphed_visiting_hidden_rows_serving_last_row_logits(
                &vec![2222; 32768],
                CHUNK,
                &mut cache,
                &mut session.1,
                |_, _, _| anyhow::bail!("visitor ran before injected failure"),
            );
            drop(fault);
            assert_eq!(failed.unwrap_err().to_string(), INJECTED);
            assert!(fail_replacement_if_armed().is_ok());
            assert_eq!(cache.current_len(), 0);
            assert!(cache.lin_attn_states.iter().all(Option::is_none));
            assert_eq!(session.1.captured_segment_count(), 0);
            assert_eq!(session.1.captured_cfx_route_decision(), Some(false));
            assert_eq!(session.1.tok_stage.id(), original_stage);
            assert!(session.1.capture_failed);
            assert!(
                session
                    .1
                    .warmed_one_eager_full_chunk_so_plans_and_workspaces_exist
            );
            eprintln!("[q38-route-recovery] fault-after-invalidation segments=0 frontier=0");
        }
        select_route(depth);
        let ids: Vec<u32> = (0..depth)
            .map(|i| 2000 + ((i + stage * 7919) % 30000) as u32)
            .collect();
        let expected = eager_reference(&model, &ids)?;
        for replay in 0..2 {
            device.synchronize()?;
            cache.reset();
            select_route(depth);
            let mut visited = 0;
            let logits = model
                .prefill_chunks_graphed_visiting_hidden_rows_serving_last_row_logits(
                    &ids,
                    CHUNK,
                    &mut cache,
                    &mut session.1,
                    |_, piece, rows| {
                        anyhow::ensure!(visited < expected.hidden.len(), "extra hidden callback");
                        anyhow::ensure!(
                            piece == &ids[visited * CHUNK..(visited + 1) * CHUNK],
                            "callback token alignment"
                        );
                        same(
                            &hidden_bits(rows)?,
                            &expected.hidden[visited],
                            &format!("stage{stage} replay{replay} hidden{visited}"),
                        )?;
                        visited += 1;
                        Ok(())
                    },
                )?;
            assert_eq!(visited, expected.hidden.len());
            assert_eq!(cache.current_len(), depth);
            assert_eq!(
                session.1.captured_segment_count(),
                if stage == 1 { 0 } else { 16 }
            );
            assert_eq!(session.1.capture_failed, stage == 1);
            assert_eq!(
                session.1.captured_cfx_route_decision(),
                Some(depth == 32768)
            );
            if stage == 1 {
                assert_eq!(session.1.tok_stage.id(), original_stage);
                for (current, pinned) in cache.lin_attn_states.iter().zip(
                    &session.1.pinned_lin_states_stable_across_prefills_because_segment_graphs_bake_their_pointers,
                ) {
                    assert_ne!(current.as_ref().unwrap().recurrent_state().id(), pinned.as_ref().unwrap().recurrent_state().id(), "failed session installed retired pinned state");
                }
            } else if stage >= 2 {
                assert_ne!(
                    session.1.tok_stage.id(),
                    original_stage,
                    "replacement must own new staging"
                );
            }
            same(&logits_bits(&logits)?, &expected.logits, "last-row logits")?;
            same(
                &first_decode(&model, &mut cache)?,
                &expected.decode,
                "first decode logits",
            )?;
            assert_eq!(cache.current_len(), depth + 1);
            eprintln!("[q38-route-recovery] stage={stage} depth={depth} replay={replay} segments={} hidden=exact logits=exact decode=exact", session.1.captured_segment_count());
        }
    }
    device.synchronize()?;
    Ok(())
}
