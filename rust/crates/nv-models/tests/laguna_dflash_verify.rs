#![cfg(feature = "cuda")]

#[path = "laguna_prompts.rs"]
mod prompts;

use candle_core::{Device, Tensor};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use nv_models::laguna::{Laguna, LagunaConfig};
use nv_models::laguna_dflash::argmax_row;
use prompts::LagunaEval;

fn rows_of(logits: &Tensor, m: usize, vocab: usize) -> Vec<Vec<f32>> {
    let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(flat.len(), m * vocab);
    (0..m)
        .map(|i| flat[i * vocab..(i + 1) * vocab].to_vec())
        .collect()
}

#[test]
#[ignore]
fn laguna_dflash_verify_device_routed_moe_ab() {
    if std::env::var("NV_LAGUNA_TEST").is_err() || std::env::var("NV_LAGUNA_DFLASH").is_err() {
        eprintln!("set NV_LAGUNA_TEST=1 and NV_LAGUNA_DFLASH=1 to run");
        return;
    }
    let ev = LagunaEval::open().expect("laguna snapshot + prompt pack");
    eprintln!("{}", ev.describe());
    let tdir = ev.dir.clone();
    let device = Device::new_cuda(0).expect("cuda device");
    let raw_cfg = std::fs::read_to_string(tdir.join("config.json")).expect("read config");
    let config = LagunaConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qconfig =
        nv_weights::QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse quant config");
    let weights = nv_weights::WeightLoader::open_dir(&tdir, &device).expect("open weights");
    let target =
        Laguna::from_loader_quantized(config, &weights, &qconfig, &device).expect("load target");
    let vocab = target.config().vocab_size;

    let ids: Vec<u32> = ev.ids("openended-code").expect("pack prompt");
    let seq = ids.len();
    let mask_id: u32 = 12;

    for &m in &[5usize, 8usize, 16usize] {
        let max_seq = seq + 2 * m + 8;
        let mut cache = target.new_kv_cache(max_seq).expect("cache");
        let tokens = Tensor::from_vec(ids.clone(), (1usize, seq), &device).unwrap();
        let positions =
            Tensor::from_vec((0..seq as i32).collect::<Vec<i32>>(), seq, &device).unwrap();
        target.set_device_verify_routing(false);
        let logits = target
            .forward_with_cache(&tokens, &positions, &mut cache)
            .expect("prefill");
        let last: Vec<f32> = logits
            .narrow(1, seq - 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let anchor = argmax_row(&last);

        let mut block = vec![mask_id; m];
        block[0] = anchor;
        let bt = Tensor::from_vec(block.clone(), (1usize, m), &device).unwrap();
        let bp = Tensor::from_vec(
            (0..m).map(|i| (seq + i) as i32).collect::<Vec<i32>>(),
            m,
            &device,
        )
        .unwrap();

        target.set_device_verify_routing(false);
        let _ = target
            .forward_with_cache(&bt, &bp, &mut cache)
            .expect("warm host");
        cache.rollback(m).expect("rollback");
        target.set_device_verify_routing(true);
        let _ = target
            .forward_with_cache(&bt, &bp, &mut cache)
            .expect("warm dev");
        cache.rollback(m).expect("rollback");
        device.synchronize().ok();

        target.set_device_verify_routing(false);
        let t0 = std::time::Instant::now();
        let logits_h = target
            .forward_with_cache(&bt, &bp, &mut cache)
            .expect("host verify");
        device.synchronize().ok();
        let host_ms = 1000.0 * t0.elapsed().as_secs_f64();
        cache.rollback(m).expect("rollback");

        target.set_device_verify_routing(true);
        let t1 = std::time::Instant::now();
        let logits_d = target
            .forward_with_cache(&bt, &bp, &mut cache)
            .expect("device verify");
        device.synchronize().ok();
        let dev_ms = 1000.0 * t1.elapsed().as_secs_f64();
        cache.rollback(m).expect("rollback");
        let logits_d2 = target
            .forward_with_cache(&bt, &bp, &mut cache)
            .expect("device verify repeat");
        cache.rollback(m).expect("rollback");
        assert!(
            target.device_verify_routing(),
            "device verify routing disabled itself (fell back) at m={m}"
        );

        let rows_h = rows_of(&logits_h, m, vocab);
        let rows_d = rows_of(&logits_d, m, vocab);
        let rows_d2 = rows_of(&logits_d2, m, vocab);
        assert_eq!(
            rows_d, rows_d2,
            "m={m}: device-routed verify must be bitwise deterministic"
        );
        let mut mismatches = 0usize;
        for i in 0..m {
            let ah = argmax_row(&rows_h[i]);
            let ad = argmax_row(&rows_d[i]);
            if ah != ad {
                mismatches += 1;
                eprintln!(
                    "m={m} slot {i}: host argmax {ah} (h {:.3} / d {:.3}) vs device {ad} (h {:.3} / d {:.3})",
                    rows_h[i][ah as usize],
                    rows_d[i][ah as usize],
                    rows_h[i][ad as usize],
                    rows_d[i][ad as usize],
                );
            }
        }
        eprintln!(
            "m={m}: per-slot argmax agree {}/{m}, single-forward host {host_ms:.2} ms vs device {dev_ms:.2} ms",
            m - mismatches
        );

        assert!(
            mismatches <= 1,
            "m={m}: {mismatches} argmax mismatches vs host-routed (bar <=1: route_host \
             breaks exact bf16 gate-logit ties to the lower expert index exactly like the \
             moe_route_topk kernel, so on identical input the expert sets are identical -- \
             NV_LAGUNA_ROUTE_AB_PROBE=1 prints any residual per-layer routing divergence; \
             more than one flip means either that tie alignment broke or the ~1e-5 \
             inter-stack drift class between forward_grouped and forward_grouped_decode \
             grew past a bf16 gate-logit rounding boundary on more than one slot)"
        );
    }
}

#[test]
#[ignore]
fn laguna_dflash_verify_single_layer_moe_ab() {
    if std::env::var("NV_LAGUNA_TEST").is_err() || std::env::var("NV_LAGUNA_DFLASH").is_err() {
        eprintln!("set NV_LAGUNA_TEST=1 and NV_LAGUNA_DFLASH=1 to run");
        return;
    }
    let ev = LagunaEval::open().expect("laguna snapshot + prompt pack");
    eprintln!("{}", ev.describe());
    let tdir = ev.dir.clone();
    let device = Device::new_cuda(0).expect("cuda device");
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let raw_cfg = std::fs::read_to_string(tdir.join("config.json")).expect("read config");
    let config = LagunaConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qconfig =
        nv_weights::QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse quant config");
    let weights = nv_weights::WeightLoader::open_dir(&tdir, &device).expect("open weights");
    let target =
        Laguna::from_loader_quantized(config, &weights, &qconfig, &device).expect("load target");
    let hidden = target.config().hidden_size;
    let inter = target.config().moe_intermediate_size;
    let k = target.config().num_experts_per_tok;
    let num_experts = target.config().num_experts;
    let n_tokens = 16usize;

    let mut checked_layers = 0usize;
    for (li, layer) in target.layers().iter().enumerate() {
        let moe = match &layer.ffn {
            nv_models::laguna::LagunaFfn::Moe(m) => m,
            _ => continue,
        };
        if checked_layers >= 3 {
            break;
        }
        checked_layers += 1;
        let grouped = moe
            .grouped
            .lock()
            .unwrap()
            .as_ref()
            .expect("grouped weights not prebuilt")
            .as_ref()
            .expect("grouped weights failed to build")
            .clone();
        let stream = dev.cuda_stream();
        let mut ctx = nv_layers::moe_grouped::GroupedDecodeContext::new_multi(
            hidden,
            inter,
            k,
            num_experts,
            n_tokens,
            &stream,
        )
        .expect("verify ctx");

        for trial in 0..4 {
            let mut state = 0x243f6a8885a308d3u64 ^ ((li as u64) << 32) ^ trial;
            let mut next_f = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32) / (1u32 << 24) as f32 * 2.0 - 1.0
            };
            let x_host: Vec<f32> = (0..n_tokens * hidden).map(|_| next_f()).collect();
            let x = Tensor::from_vec(x_host, (n_tokens, hidden), &device)
                .unwrap()
                .to_dtype(candle_core::DType::BF16)
                .unwrap();

            let (ids_h, w_h) = moe.route_host(&x, n_tokens).expect("route_host");
            let out_h = nv_layers::moe_grouped::forward_grouped(
                &grouped, &grouped, &x, &ids_h, &w_h, n_tokens, k, &device,
            )
            .expect("host-routed forward_grouped")
            .affine(moe.routed_scaling as f64, 0.0)
            .unwrap();

            let logits = moe.gate.forward(&x).expect("gate");
            let out_d = nv_layers::moe_grouped::forward_grouped_decode(
                &grouped,
                &mut ctx,
                &x,
                &logits,
                Some(&moe.selection_bias),
                1,
                moe.softcap,
                moe.norm_topk,
                moe.routed_scaling,
                &device,
            )
            .expect("device-routed forward_grouped_decode");

            #[allow(deprecated)]
            let ids_d: Vec<i32> = stream.memcpy_dtov(&ctx.topk_ids).unwrap();
            #[allow(deprecated)]
            let w_d: Vec<f32> = stream.memcpy_dtov(&ctx.topk_weights).unwrap();
            let logits_f32: Vec<f32> = logits
                .to_dtype(candle_core::DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let bias_f32: Vec<f32> = moe
                .selection_bias
                .to_dtype(candle_core::DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let mut set_mismatches = 0usize;
            let mut token_matched = vec![true; n_tokens];
            for t in 0..n_tokens {
                let mut set_h: Vec<u32> = ids_h[t * k..(t + 1) * k].to_vec();
                let mut set_d: Vec<u32> = ids_d[t * k..(t + 1) * k]
                    .iter()
                    .map(|&v| v as u32)
                    .collect();
                set_h.sort_unstable();
                set_d.sort_unstable();
                if set_h != set_d {
                    set_mismatches += 1;
                    token_matched[t] = false;
                    let only_h: Vec<u32> = set_h
                        .iter()
                        .filter(|e| !set_d.contains(e))
                        .copied()
                        .collect();
                    let only_d: Vec<u32> = set_d
                        .iter()
                        .filter(|e| !set_h.contains(e))
                        .copied()
                        .collect();
                    let sel_of = |e: u32| {
                        let l = logits_f32[t * num_experts + e as usize];
                        let s = 1.0f32 / (1.0 + (-l).exp());
                        s + bias_f32[e as usize]
                    };
                    for &e in only_h.iter().chain(only_d.iter()) {
                        let l = logits_f32[t * num_experts + e as usize];
                        eprintln!(
                            "layer {li} trial {trial} token {t}: expert {e} ({}) logit {l:.6} sel {:.9}",
                            if only_h.contains(&e) { "host-only" } else { "device-only" },
                            sel_of(e)
                        );
                    }

                    let mut sels_h: Vec<u32> =
                        only_h.iter().map(|&e| sel_of(e).to_bits()).collect();
                    let mut sels_d: Vec<u32> =
                        only_d.iter().map(|&e| sel_of(e).to_bits()).collect();
                    sels_h.sort_unstable();
                    sels_d.sort_unstable();
                    assert_eq!(
                        sels_h, sels_d,
                        "layer {li} trial {trial} token {t}: expert set difference is NOT an exact tie"
                    );
                    continue;
                }
                for j in 0..k {
                    let e = ids_h[t * k + j];
                    let wh = w_h[t * k + j] * moe.routed_scaling;
                    let pos = ids_d[t * k..(t + 1) * k]
                        .iter()
                        .position(|&v| v as u32 == e)
                        .unwrap();
                    let wd = w_d[t * k + pos];
                    let rel = (wh - wd).abs() / wh.abs().max(1e-6);
                    assert!(
                        rel < 1e-4,
                        "layer {li} trial {trial} token {t} expert {e}: weight {wh} vs {wd}"
                    );
                }
            }
            assert!(
                set_mismatches <= 2,
                "layer {li} trial {trial}: {set_mismatches}/16 tokens with differing expert sets"
            );

            let h_flat: Vec<f32> = out_h.flatten_all().unwrap().to_vec1().unwrap();
            let d_flat: Vec<f32> = out_d.flatten_all().unwrap().to_vec1().unwrap();
            let mut max_rel = 0f32;
            let mut max_abs = 0f32;
            for t in 0..n_tokens {
                if !token_matched[t] {
                    continue;
                }
                for h in 0..hidden {
                    let a = h_flat[t * hidden + h];
                    let b = d_flat[t * hidden + h];
                    let abs = (a - b).abs();
                    let rel = abs / a.abs().max(b.abs()).max(1e-2);
                    if rel > max_rel {
                        max_rel = rel;
                    }
                    if abs > max_abs {
                        max_abs = abs;
                    }
                }
            }
            eprintln!(
                "layer {li} trial {trial}: expert sets equal {}/16, routed out (matched rows) max_rel {max_rel:.2e} max_abs {max_abs:.2e}",
                n_tokens - set_mismatches
            );
            assert!(
                max_rel < 1e-3,
                "layer {li} trial {trial}: routed output rel diff too large ({max_rel})"
            );
        }
    }
    assert!(checked_layers > 0, "no MoE layers found");
}

#[test]
#[ignore]
fn laguna_dflash_accept_kernel_matches_host_chain() {
    if std::env::var("NV_LAGUNA_TEST").is_err() || std::env::var("NV_LAGUNA_DFLASH").is_err() {
        eprintln!("set NV_LAGUNA_TEST=1 and NV_LAGUNA_DFLASH=1 to run");
        return;
    }
    let device = Device::new_cuda(0).expect("cuda device");
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = dev.cuda_stream();
    let vocab = 100352usize;
    let m = 16usize;
    let k = m - 1;
    let parts = nv_kernels::cuda::dflash_accept_parts();
    assert!(parts > 0);

    let mut rng_state = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    for &target_accept in &[0usize, 3, k - 1, k] {
        let mut logits_host = vec![0f32; m * vocab];
        for v in logits_host.iter_mut() {
            *v = ((next() >> 40) as f32) / (1u32 << 24) as f32 * 8.0 - 4.0;
        }

        for row in [1usize, 5, 9] {
            let base = row * vocab;
            let peak_at = (next() as usize) % (vocab - 100);
            logits_host[base + peak_at] = 20.0;
            logits_host[base + peak_at + 37] = 20.0;
        }
        let row_argmax_host: Vec<u32> = (0..m)
            .map(|i| argmax_row(&logits_host[i * vocab..(i + 1) * vocab]))
            .collect();
        let drafts: Vec<u32> = (0..k)
            .map(|i| {
                if i < target_accept {
                    row_argmax_host[i]
                } else {
                    row_argmax_host[i].wrapping_add(1) % vocab as u32
                }
            })
            .collect();

        let mut a_ref = 0usize;
        while a_ref < k && row_argmax_host[a_ref] == drafts[a_ref] {
            a_ref += 1;
        }
        let mut emitted_ref: Vec<u32> = drafts[..a_ref].to_vec();
        emitted_ref.push(row_argmax_host[a_ref]);
        assert_eq!(a_ref, target_accept);

        #[allow(deprecated)]
        let logits_dev = stream.memcpy_stod(&logits_host).unwrap();
        #[allow(deprecated)]
        let drafts_dev = stream.memcpy_stod(&drafts).unwrap();
        let mut row_argmax_dev = stream.alloc_zeros::<u32>(m).unwrap();
        let mut out_dev = stream.alloc_zeros::<u32>(m + 1).unwrap();
        let mut pv = stream.alloc_zeros::<f32>(m * parts).unwrap();
        let mut pi = stream.alloc_zeros::<i32>(m * parts).unwrap();
        let rc = {
            let (lp, _g0) = logits_dev.device_ptr(&stream);
            let (dp, _g1) = drafts_dev.device_ptr(&stream);
            let (rp, _g2) = row_argmax_dev.device_ptr_mut(&stream);
            let (op, _g3) = out_dev.device_ptr_mut(&stream);
            let (pvp, _g4) = pv.device_ptr_mut(&stream);
            let (pip, _g5) = pi.device_ptr_mut(&stream);
            unsafe {
                nv_kernels::cuda::dflash_accept_f32(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    lp as *const f32,
                    dp as *const u32,
                    rp as *mut u32,
                    op as *mut u32,
                    pvp as *mut f32,
                    pip as *mut i32,
                    m as i32,
                    vocab as i32,
                )
            }
        };
        assert_eq!(rc, 0);
        #[allow(deprecated)]
        let row_argmax_gpu: Vec<u32> = stream.memcpy_dtov(&row_argmax_dev).unwrap();
        #[allow(deprecated)]
        let out_gpu: Vec<u32> = stream.memcpy_dtov(&out_dev).unwrap();
        assert_eq!(
            row_argmax_gpu, row_argmax_host,
            "per-row argmax (first-max)"
        );
        let a_gpu = out_gpu[0] as usize;
        assert_eq!(a_gpu, a_ref, "accept count");
        assert_eq!(
            &out_gpu[1..2 + a_gpu],
            emitted_ref.as_slice(),
            "emitted tokens"
        );
    }
    eprintln!(
        "accept kernel matches host chain for accept prefixes 0/3/{}/{k}",
        k - 1
    );
}
