use super::*;
use nv_layers::moe_grouped::{
    forward_grouped, forward_grouped_decode, GroupedDecodeContext, MoeGroupedWeights,
};

fn words(t: Tensor) -> Vec<u32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(|x| {
            assert!(x.is_finite());
            x.to_bits()
        })
        .collect()
}

fn outputs(w: &MoeGroupedWeights, moe: &LagunaMoe, device: &Device) -> Vec<Vec<u32>> {
    let Device::Cuda(d) = device else {
        panic!("CUDA required")
    };
    let stream = d.cuda_stream();
    let mut result = Vec::new();
    for m in [1usize, 3, 256] {
        let x = Tensor::from_vec(
            (0..m * w.hidden_size)
                .map(|i| ((i % 127) as f32 - 63.0) / 128.0)
                .collect::<Vec<_>>(),
            (m, w.hidden_size),
            device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let y = if m == 256 {
            let ids: Vec<_> = (0..m * moe.top_k)
                .map(|i| (i % moe.num_experts) as u32)
                .collect();
            let weights = vec![1.0f32 / moe.top_k as f32; ids.len()];
            forward_grouped(w, w, &x, &ids, &weights, m, moe.top_k, device).unwrap()
        } else {
            let mut ctx = if m == 1 && w.folded_shared {
                GroupedDecodeContext::new_folded_shared(
                    w.hidden_size,
                    w.intermediate_size,
                    moe.top_k,
                    moe.num_experts,
                    m,
                    &stream,
                )
                .unwrap()
            } else {
                GroupedDecodeContext::new_multi(
                    w.hidden_size,
                    w.intermediate_size,
                    moe.top_k,
                    moe.num_experts,
                    m,
                    &stream,
                )
                .unwrap()
            };
            let logits = moe.gate.forward(&x).unwrap();
            forward_grouped_decode(
                w,
                &mut ctx,
                &x,
                &logits,
                Some(&moe.selection_bias),
                1,
                moe.softcap,
                moe.norm_topk,
                moe.routed_scaling,
                device,
            )
            .unwrap()
        };
        result.push(words(y));
        result.push(words(moe.shared_expert.forward(&x).unwrap()));
        stream.synchronize().unwrap();
    }
    result
}

#[test]
#[ignore = "NV_LAGUNA_TEST=1 and NV_LAGUNA_DIR; real packed layer, all grouped bytes and M1/M3/M256 outputs before/after release"]
fn real_grouped_bank_and_outputs_survive_original_release() {
    assert_eq!(std::env::var("NV_LAGUNA_TEST").as_deref(), Ok("1"));
    assert!(std::env::var_os("NV_LAGUNA_HOST_MOE").is_none());
    let dir = std::path::PathBuf::from(std::env::var("NV_LAGUNA_DIR").unwrap());
    let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let cfg = LagunaConfig::from_hf_json_str(&raw).unwrap();
    let quant = QuantizationConfig::from_hf_json_str(&raw).unwrap();
    assert_eq!(quant.scheme, QuantScheme::Nvfp4);
    assert_eq!(
        (
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.num_experts,
            cfg.num_experts_per_tok,
            cfg.shared_expert_intermediate_size
        ),
        (2048, 512, 256, 8, 512)
    );
    let device = Device::new_cuda(0).unwrap();
    let Device::Cuda(d) = &device else {
        unreachable!()
    };
    let stream = d.cuda_stream();
    let runner = Some(Arc::new(Mutex::new(
        nv_quant::nvfp4::Nvfp4GemmRunner::new(stream.clone()).unwrap(),
    )));
    let weights = WeightLoader::open_dir(&dir, &device).unwrap();
    let idx = (0..cfg.num_hidden_layers)
        .find(|&i| cfg.is_moe_layer(i))
        .unwrap();
    let fp8_runner = None;
    let mut layer = LagunaLayer::from_loader(
        &cfg,
        idx,
        &weights,
        &quant,
        &runner,
        &fp8_runner,
        &device,
        DType::BF16,
    )
    .unwrap();
    drop(weights);
    let LagunaFfn::Moe(moe) = &mut layer.ffn else {
        panic!("selected routed layer missing")
    };
    let folded = nv_layers::moe_grouped::folded_shared_enabled();
    let reference = if folded {
        MoeGroupedWeights::build_from_experts_folding_shared_as_a_fixed_extra_tile(
            &moe.experts,
            &moe.shared_expert,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            &stream,
        )
        .unwrap()
    } else {
        MoeGroupedWeights::build_from_experts(
            &moe.experts,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            &stream,
        )
        .unwrap()
    };
    let expected = outputs(&reference, moe, &device);
    assert_eq!(
        expected,
        outputs(&reference, moe, &device),
        "exact null output control"
    );
    let released = prebuild_grouped_and_release_routed_experts(
        moe,
        cfg.hidden_size,
        cfg.moe_intermediate_size,
        &stream,
    )
    .unwrap();
    assert_eq!(released, 256 * 3 * (524288 + 65536));
    assert!(moe.experts.is_empty());
    let actual = moe
        .grouped
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap()
        .clone();
    for (a, b) in [
        (&reference.gate_w, &actual.gate_w),
        (&reference.up_w, &actual.up_w),
        (&reference.down_w, &actual.down_w),
        (&reference.gate_w_scales, &actual.gate_w_scales),
        (&reference.up_w_scales, &actual.up_w_scales),
        (&reference.down_w_scales, &actual.down_w_scales),
    ] {
        #[allow(deprecated)]
        let a = stream.memcpy_dtov(a).unwrap();
        #[allow(deprecated)]
        let b = stream.memcpy_dtov(b).unwrap();
        assert_eq!(a, b, "complete packed bank bytes differ");
    }
    for (a, b) in [
        (&reference.gate_alphas, &actual.gate_alphas),
        (&reference.up_alphas, &actual.up_alphas),
        (&reference.down_alphas, &actual.down_alphas),
        (
            &reference.input_globals_gate_up,
            &actual.input_globals_gate_up,
        ),
        (&reference.input_globals_down, &actual.input_globals_down),
    ] {
        #[allow(deprecated)]
        let a: Vec<u32> = stream
            .memcpy_dtov(a)
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect();
        #[allow(deprecated)]
        let b: Vec<u32> = stream
            .memcpy_dtov(b)
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_eq!(a, b, "grouped scalar bits differ");
    }
    assert_eq!(
        expected,
        outputs(&actual, moe, &device),
        "released bank numerical outputs"
    );
    assert!(
        prebuild_grouped_and_release_routed_experts(
            moe,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            &stream
        )
        .is_err(),
        "repeated release must fail before bank mutation"
    );
    let retained = moe
        .grouped
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap()
        .clone();
    assert!(Arc::ptr_eq(&actual, &retained));
    assert_eq!(
        expected,
        outputs(&actual, moe, &device),
        "healthy bank after refused second release"
    );
    let x = Tensor::zeros((1usize, cfg.hidden_size), DType::BF16, &device).unwrap();
    assert!(moe
        .forward(&x)
        .unwrap_err()
        .to_string()
        .contains("individual routed experts unavailable"));
    stream.synchronize().unwrap();
    drop((actual, retained, reference, layer, runner));
    stream.synchronize().unwrap();
    eprintln!("LAGUNA-GROUPED-RESIDENCY-GATE layer={idx} folded_shared={folded} packed_banks=bitwise scalar_bits=bitwise rows=1,3,256 null=bitwise post_release=bitwise repeat_refusal=preserves_bank released_payload_bytes={released} whole_model=unqualified");
}
