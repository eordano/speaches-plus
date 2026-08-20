#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_specdecode::eagle3_loader::{DrafterKvCache, Eagle3SpeculatorConfig, LoadedEagle3Scorer};

fn synthetic_scorer(dev: &Device) -> anyhow::Result<LoadedEagle3Scorer> {
    synthetic_scorer_max_pos(dev, 64)
}

fn synthetic_scorer_max_pos(dev: &Device, max_pos: usize) -> anyhow::Result<LoadedEagle3Scorer> {
    let cfg = Eagle3SpeculatorConfig {
        hidden_size: 16,
        draft_vocab_size: 32,
        target_vocab_size: 64,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        intermediate_size: 32,
        max_position_embeddings: max_pos,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        norm_before_residual: true,
        norm_before_fc: false,
        eagle_aux_hidden_state_layer_ids: vec![0],
    };
    let dtype = DType::BF16;
    let h = cfg.hidden_size;

    let mk_linear = |out: usize, inp: usize, seed: f32| -> anyhow::Result<Linear> {
        let total = out * inp;
        let data: Vec<bf16> = (0..total)
            .map(|i| bf16::from_f32(((i as f32) * 0.001 + seed).sin() * 0.05))
            .collect();
        let t = Tensor::from_vec(data, (out, inp), dev)?;
        Linear::new(t, None)
    };
    let mk_rms = |dim: usize| -> anyhow::Result<RmsNorm> {
        let data: Vec<bf16> = (0..dim).map(|_| bf16::from_f32(1.0)).collect();
        let t = Tensor::from_vec(data, dim, dev)?;
        Ok(RmsNorm::new(t, cfg.rms_norm_eps))
    };

    let embed_total = cfg.target_vocab_size * h;
    let embed_data: Vec<bf16> = (0..embed_total)
        .map(|i| bf16::from_f32(((i as f32) * 0.013).cos() * 0.05))
        .collect();
    let embed = Tensor::from_vec(embed_data, (cfg.target_vocab_size, h), dev)?;

    let fc = mk_linear(h, cfg.fc_in_dim(), 0.1)?;
    let input_ln = mk_rms(h)?;
    let hidden_ln = mk_rms(h)?;
    let post_ln = mk_rms(h)?;
    let q = mk_linear(cfg.q_out_dim(), cfg.block_in_dim(), 0.2)?;
    let k = mk_linear(cfg.kv_out_dim(), cfg.block_in_dim(), 0.3)?;
    let v = mk_linear(cfg.kv_out_dim(), cfg.block_in_dim(), 0.4)?;
    let o = mk_linear(h, cfg.q_out_dim(), 0.5)?;
    let gate = mk_linear(cfg.intermediate_size, h, 0.6)?;
    let up = mk_linear(cfg.intermediate_size, h, 0.7)?;
    let down = mk_linear(h, cfg.intermediate_size, 0.8)?;
    let norm = mk_rms(h)?;
    let lm_head = mk_linear(cfg.draft_vocab_size, h, 0.9)?;

    let d2t: Vec<u32> = (0..cfg.draft_vocab_size as u32).collect();
    let t2d: Vec<bool> = (0..cfg.target_vocab_size)
        .map(|i| i < cfg.draft_vocab_size)
        .collect();

    LoadedEagle3Scorer::from_parts(
        cfg,
        dev.clone(),
        dtype,
        embed,
        fc,
        input_ln,
        hidden_ln,
        post_ln,
        q,
        k,
        v,
        o,
        gate,
        up,
        down,
        norm,
        lm_head,
        d2t,
        t2d,
    )
}

fn synthetic_aux(scorer: &LoadedEagle3Scorer, rows: usize, dev: &Device) -> Tensor {
    let n = rows * scorer.config().fc_in_dim();
    let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.007).sin() * 0.3).collect();
    Tensor::from_vec(data, (rows, scorer.config().fc_in_dim()), dev).unwrap()
}

#[test]
fn graphed_shift_chain_matches_eager_across_rounds() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let scorer = synthetic_scorer(&dev)?;
    let kd = 3usize;
    let mut g = scorer.new_chain_graph(48, kd)?;

    let mut cache_g = DrafterKvCache::new();
    let mut cache_e = DrafterKvCache::new();

    let mut context: Vec<u32> = vec![1, 5, 9, 13, 2];
    let mut bonus: u32 = 7;
    let mut node_count_after_capture = 0usize;
    for round in 0..5 {
        let aux = synthetic_aux(&scorer, context.len(), &dev);
        let proj = scorer.project_aux(&aux)?;

        let got_g = scorer.chain_draft_cached_shift_graphed(
            &mut cache_g,
            &mut g,
            &context,
            &proj,
            kd,
            bonus,
        )?;
        let got_e =
            scorer.chain_draft_cached_cond(&mut cache_e, &context, &proj, kd, Some(bonus), true)?;

        assert_eq!(got_g.len(), kd, "round {round}: graphed draft count");
        assert_eq!(
            got_g,
            got_e,
            "round {round}: graphed drafts diverge from eager shift chain (ctx={})",
            context.len()
        );
        assert!(
            !g.disabled(),
            "round {round}: graph fell back (capture failed)"
        );
        if round == 0 {
            node_count_after_capture = g.graph_node_count();
            assert!(
                node_count_after_capture > 0,
                "graph did not capture on round 0"
            );
        } else {
            assert_eq!(
                g.graph_node_count(),
                node_count_after_capture,
                "round {round}: graph was re-captured (shape churn)"
            );
        }

        context.push(bonus);
        context.push(got_e[0]);
        bonus = got_e[1] % 32;
    }
    Ok(())
}

#[test]
fn graphed_chain_replays_stable_200_rounds() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let scorer = synthetic_scorer_max_pos(&dev, 768)?;
    let kd = 3usize;
    let mut g = scorer.new_chain_graph(768, kd)?;
    let mut g_ref = scorer.new_chain_graph(768, kd)?;

    let mut cache_g = DrafterKvCache::new();
    let mut cache_b = DrafterKvCache::new();

    let mut context: Vec<u32> = vec![1, 5, 9, 13, 2];
    let mut bonus: u32 = 7;
    let mut nodes = 0usize;
    for round in 0..201 {
        let aux = synthetic_aux(&scorer, context.len(), &dev);
        let proj = scorer.project_aux(&aux)?;

        let got_g = scorer.chain_draft_cached_shift_graphed_mode(
            &mut cache_g,
            &mut g,
            &context,
            &proj,
            kd,
            bonus,
            false,
        )?;
        let got_b = scorer.chain_draft_cached_shift_graphed_mode(
            &mut cache_b,
            &mut g_ref,
            &context,
            &proj,
            kd,
            bonus,
            true,
        )?;

        assert_eq!(
            got_g,
            got_b,
            "round {round}: replayed drafts diverge from the uncaptured body (ctx={})",
            context.len()
        );
        assert!(!g.disabled(), "round {round}: graph fell back");
        assert!(!g_ref.disabled(), "round {round}: eager body fell back");
        if round == 0 {
            nodes = g.graph_node_count();
            assert!(nodes > 0, "graph did not capture on round 0");
        } else {
            assert_eq!(
                g.graph_node_count(),
                nodes,
                "round {round}: graph was re-captured"
            );
        }

        context.push(bonus);
        context.push(got_g[0]);
        bonus = got_g[1] % 32;
    }
    Ok(())
}

#[test]
fn single_query_attn_kernel_matches_sdpa() -> anyhow::Result<()> {
    use cudarc::driver::DevicePtr;
    use cudarc::driver::DevicePtrMut;
    use nv_layers::attn::{sdpa, AttnConfig};

    let dev = Device::new_cuda(0)?;
    let cud = match &dev {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = cud.cuda_stream();
    let (nh, nkv, hd, n) = (4usize, 2usize, 8usize, 7usize);
    let scale = 1.0f64 / (hd as f64).sqrt();

    let q = Tensor::randn(0f32, 1f32, (1usize, 1usize, nh, hd), &dev)?.to_dtype(DType::BF16)?;
    let kt = Tensor::randn(0f32, 1f32, (1usize, n, nkv, hd), &dev)?.to_dtype(DType::BF16)?;
    let vt = Tensor::randn(0f32, 1f32, (1usize, n, nkv, hd), &dev)?.to_dtype(DType::BF16)?;

    let cfg = AttnConfig {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        softmax_scale: scale as f32,
        causal: false,
    };
    let want: Vec<f32> = sdpa(&q, &kt, &vt, &cfg)?
        .reshape((nh * hd,))?
        .to_dtype(DType::F32)?
        .to_vec1()?;

    let stride = nkv * hd;
    let mut kc = stream.alloc_zeros::<half::bf16>(16 * stride)?;
    let mut vc = stream.alloc_zeros::<half::bf16>(16 * stride)?;
    {
        let kcont = kt.contiguous()?;
        let vcont = vt.contiguous()?;
        let (ks, _) = kcont.storage_and_layout();
        let (vs, _) = vcont.storage_and_layout();
        let kcu = match &*ks {
            candle_core::Storage::Cuda(c) => c,
            _ => unreachable!(),
        };
        let vcu = match &*vs {
            candle_core::Storage::Cuda(c) => c,
            _ => unreachable!(),
        };
        let ksl = kcu.as_cuda_slice::<half::bf16>()?;
        let vsl = vcu.as_cuda_slice::<half::bf16>()?;
        let mut kdst = kc.slice_mut(0..n * stride);
        let mut vdst = vc.slice_mut(0..n * stride);
        stream.memcpy_dtod(&ksl.slice(0..n * stride), &mut kdst)?;
        stream.memcpy_dtod(&vsl.slice(0..n * stride), &mut vdst)?;
    }
    let n_dev = stream.clone_htod(&[n as i32])?;
    let mask = stream.alloc_zeros::<u8>(1)?;
    let mut out = stream.alloc_zeros::<half::bf16>(nh * hd)?;
    let q_s = (q.reshape((1usize, nh * hd))? * scale)?.contiguous()?;
    {
        let (qs, _) = q_s.storage_and_layout();
        let qcu = match &*qs {
            candle_core::Storage::Cuda(c) => c,
            _ => unreachable!(),
        };
        let qsl = qcu.as_cuda_slice::<half::bf16>()?;
        let (qp, _g1) = qsl.device_ptr(&stream);
        let (kp, _g2) = kc.device_ptr(&stream);
        let (vp, _g3) = vc.device_ptr(&stream);
        let (np, _g4) = n_dev.device_ptr(&stream);
        let (mp, _g5) = mask.device_ptr(&stream);
        let (op, _g6) = out.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::tree_verify_attn_bf16(
                stream.cu_stream() as *mut _,
                qp as *const u16,
                kp as *const u16,
                vp as *const u16,
                np as *const i32,
                mp as *const u8,
                std::ptr::null(),
                op as *mut u16,
                nh as i32,
                nkv as i32,
                hd as i32,
                1,
                0,
            )
        };
        assert_eq!(rc, 0, "tree_verify_attn_bf16 rc");
    }
    stream.synchronize()?;
    let got_bf: Vec<half::bf16> = stream.clone_dtoh(&out)?;
    let got: Vec<f32> = got_bf.iter().map(|x| x.to_f32()).collect();
    let mut max_abs = 0f32;
    for (a, b) in want.iter().zip(got.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs < 0.05,
        "kernel attention diverges from sdpa: max_abs={max_abs}\nwant[..8]={:?}\ngot[..8]={:?}",
        &want[..8],
        &got[..8]
    );
    Ok(())
}

#[test]
fn graphed_chain_falls_back_beyond_capacity() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let scorer = synthetic_scorer(&dev)?;
    let kd = 2usize;

    let mut g = scorer.new_chain_graph(4, kd)?;
    let context: Vec<u32> = vec![1, 5, 9, 13, 2];
    let aux = synthetic_aux(&scorer, context.len(), &dev);
    let proj = scorer.project_aux(&aux)?;
    let mut cache_g = DrafterKvCache::new();
    let mut cache_e = DrafterKvCache::new();
    let got_g =
        scorer.chain_draft_cached_shift_graphed(&mut cache_g, &mut g, &context, &proj, kd, 7)?;
    let got_e = scorer.chain_draft_cached_cond(&mut cache_e, &context, &proj, kd, Some(7), true)?;
    assert_eq!(got_g, got_e);
    assert_eq!(
        g.graph_node_count(),
        0,
        "no graph should have been captured"
    );
    Ok(())
}

#[test]
fn graphed_shift_chain_matches_eager_large_ctx_uncapped() -> anyhow::Result<()> {
    use nv_specdecode::eagle3_loader::DRAFTER_KV_CAP_SLACK;
    let dev = Device::new_cuda(0)?;
    let scorer = synthetic_scorer_max_pos(&dev, 768)?;
    let kd = 3usize;
    let (sink, window) = (2usize, 8usize);
    let bound = sink + window + DRAFTER_KV_CAP_SLACK;
    let mut g = scorer.new_chain_graph(bound + 64, kd)?;
    let mut cache_g = DrafterKvCache::new();
    let mut cache_e = DrafterKvCache::new();
    let mut context: Vec<u32> = (0..bound - 4).map(|i| ((i * 7 + 3) % 32) as u32).collect();
    let mut bonus: u32 = 7;
    for round in 0..12 {
        let aux = synthetic_aux(&scorer, context.len(), &dev);
        let proj = scorer.project_aux(&aux)?;
        let got_g = scorer.chain_draft_cached_shift_graphed(
            &mut cache_g,
            &mut g,
            &context,
            &proj,
            kd,
            bonus,
        )?;
        let got_e =
            scorer.chain_draft_cached_cond(&mut cache_e, &context, &proj, kd, Some(bonus), true)?;
        assert_eq!(
            got_g,
            got_e,
            "round {round}: uncapped graphed drafts diverge from uncapped eager (ctx={})",
            context.len()
        );
        context.push(bonus);
        context.push(got_e[0]);
        bonus = got_e[1] % 32;
    }
    Ok(())
}

#[test]
fn graphed_shift_chain_matches_eager_with_kv_cap() -> anyhow::Result<()> {
    use nv_specdecode::eagle3_loader::DRAFTER_KV_CAP_SLACK;
    let dev = Device::new_cuda(0)?;
    let scorer = synthetic_scorer_max_pos(&dev, 768)?;
    let kd = 3usize;
    let (sink, window) = (2usize, 8usize);
    let bound = sink + window + DRAFTER_KV_CAP_SLACK;
    let mut g = scorer.new_chain_graph(bound + 16, kd)?;
    let mut cache_g = DrafterKvCache::with_kv_cap(sink, window);
    let mut cache_e = DrafterKvCache::with_kv_cap(sink, window);
    let mut context: Vec<u32> = (0..bound - 4).map(|i| ((i * 7 + 3) % 32) as u32).collect();
    let mut bonus: u32 = 7;
    for round in 0..12 {
        let aux = synthetic_aux(&scorer, context.len(), &dev);
        let proj = scorer.project_aux(&aux)?;
        let got_g = scorer.chain_draft_cached_shift_graphed(
            &mut cache_g,
            &mut g,
            &context,
            &proj,
            kd,
            bonus,
        )?;
        let got_e =
            scorer.chain_draft_cached_cond(&mut cache_e, &context, &proj, kd, Some(bonus), true)?;
        assert_eq!(
            got_g,
            got_e,
            "round {round}: capped graphed drafts diverge from capped eager (ctx={})",
            context.len()
        );
        assert!(!g.disabled(), "round {round}: graph fell back");
        assert_eq!(cache_g.len(), context.len());
        assert_eq!(cache_g.phys_len() + cache_g.evicted(), context.len());
        assert!(
            cache_g.phys_len() <= bound,
            "round {round}: phys {} exceeds bound {bound}",
            cache_g.phys_len()
        );
        assert_eq!(cache_g.evicted(), cache_e.evicted(), "round {round}");
        context.push(bonus);
        context.push(got_e[0]);
        bonus = got_e[1] % 32;
    }
    assert!(cache_g.compactions() >= 1, "cap never triggered");
    assert!(cache_g.evicted() > 0);
    Ok(())
}
