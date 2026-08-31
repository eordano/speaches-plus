#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const Q36_GDN_GEOMETRY_NK16_NV32_DK128_DV128_K4_HIDDEN2048: (usize, usize, usize, usize, usize, usize) =
    (16, 32, 128, 128, 4, 2048);
const Q36_ATTN_GEOMETRY_NQ16_NKV2_HD256_ROT64: (usize, usize, usize, usize) = (16, 2, 256, 64);
const MROW_PARITY_ABS_TOL_0_02_THE_Q38_GATE_CLASS_FOR_CHUNK_VS_PER_ROW_REORDER: f32 = 0.02;
const WARM_STEPS_3_SO_PARITY_IS_MEASURED_FROM_A_NONZERO_STATE: usize = 3;
const FORCED_CHAIN_ROUNDS_5_COMPOUND_STATE_DRIFT_ACROSS_COMMITS: usize = 5;
const ENGINE_ROW_REL_L2_CEILING_3E1_REGRESSION_TRIP_ONLY_CHAIN_IDENTITY_IS_THE_CORRECTNESS_GATE:
    f64 = 3e-1;

fn host_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flat")
        .to_vec1()
        .expect("host")
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn qwen36_geometry_gdn_mrow_verify_matches_the_per_row_fused_loop_chunk_arm_on_and_off() {
    use nv_layers::linear::Linear;
    use nv_layers::linear_attn::{LinAttnState, LinearAttention, LinearAttentionConfig};

    let Ok(device) = Device::new_cuda_with_stream(0) else {
        eprintln!("[skip] no cuda device; the qwen36 GDN mrow parity needs the card");
        return;
    };
    let (n_k, n_v, d_k, d_v, kernel, hidden) = Q36_GDN_GEOMETRY_NK16_NV32_DK128_DV128_K4_HIDDEN2048;
    let cfg = LinearAttentionConfig {
        hidden_size: hidden,
        linear_num_key_heads: n_k,
        linear_num_value_heads: n_v,
        linear_key_head_dim: d_k,
        linear_value_head_dim: d_v,
        linear_conv_kernel_dim: kernel,
        mamba_ssm_dtype: DType::F32,
        rms_eps: 1e-6,
    };
    let conv_dim = cfg.conv_dim();
    let value_dim = cfg.value_dim();
    let bf = |t: Tensor| t.to_dtype(DType::BF16).expect("bf16 cast");
    let rnd = |shape: &[usize]| bf(Tensor::randn(0f32, 0.05f32, shape, &device).expect("randn"));
    let la = LinearAttention::new(
        cfg,
        Linear::new(rnd(&[conv_dim, hidden]), None).expect("qkv"),
        Linear::new(rnd(&[value_dim, hidden]), None).expect("z"),
        Linear::new(rnd(&[n_v, hidden]), None).expect("a"),
        Linear::new(rnd(&[n_v, hidden]), None).expect("b"),
        rnd(&[conv_dim, 1, kernel]),
        rnd(&[n_v]),
        rnd(&[n_v]),
        rnd(&[d_v]),
        Linear::new(rnd(&[hidden, value_dim]), None).expect("out"),
    )
    .expect("synthetic GDN layer at the qwen3.6 head geometry");
    assert!(
        la.fused_decode_supported(),
        "the qwen36 geometry must take the fused decode path or this parity proves nothing"
    );

    for &(chunk_arm, m) in &[(false, 3usize), (true, 3usize), (true, 5usize), (true, 8usize)] {
        if chunk_arm {
            std::env::set_var("NV_Q38_VERIFY_GDN_CHUNK", "1");
        } else {
            std::env::remove_var("NV_Q38_VERIFY_GDN_CHUNK");
        }
        let st_a = la.new_fused_state(&device).expect("state A");
        let st_b = la.new_fused_state(&device).expect("state B");
        for step in 0..WARM_STEPS_3_SO_PARITY_IS_MEASURED_FROM_A_NONZERO_STATE {
            let x = rnd(&[1, 1, hidden]);
            la.forward_decode_fused(&x, &st_a)
                .expect("warm A")
                .unwrap_or_else(|| panic!("warm step {step}: fused decode refused arm A"));
            la.forward_decode_fused(&x, &st_b)
                .expect("warm B")
                .unwrap_or_else(|| panic!("warm step {step}: fused decode refused arm B"));
        }
        device.synchronize().expect("sync after warm");

        let x_rows = rnd(&[1, m, hidden]);
        let ckpts_a: Vec<LinAttnState> = (0..m)
            .map(|_| la.new_fused_state(&device).expect("ckpt A"))
            .collect();
        let ckpts_b: Vec<LinAttnState> = la
            .new_fused_verify_ckpt_rows_off_one_slab_so_chunk_kernels_write_ckpts_in_place(
                &device, m,
            )
            .expect("ckpt B slab matches what the serving verify lane preallocates");

        let mut outs_a: Vec<Tensor> = Vec::with_capacity(m);
        for j in 0..m {
            let xj = x_rows.narrow(1, j, 1).expect("row").copy().expect("copy");
            let out_j = la
                .forward_decode_fused(&xj, &st_a)
                .expect("per-row fused decode")
                .unwrap_or_else(|| panic!("row {j}: fused decode refused"));
            ckpts_a[j].copy_data_from(&st_a).expect("ckpt A copy");
            outs_a.push(out_j);
        }
        let refs: Vec<&Tensor> = outs_a.iter().collect();
        let out_a = Tensor::cat(&refs, 1).expect("cat A");

        let out_b = la
            .forward_verify_mrow_projections_once_because_per_row_fused_steps_reread_every_gdn_weight(
                &x_rows, &st_b, &ckpts_b,
                None,
            )
            .expect("m-row verify")
            .expect("the m-row primitive refused the qwen36 geometry it now serves");
        device.synchronize().expect("sync after both arms");

        let tol = MROW_PARITY_ABS_TOL_0_02_THE_Q38_GATE_CLASS_FOR_CHUNK_VS_PER_ROW_REORDER;
        let d_out = max_abs(&host_f32(&out_a), &host_f32(&out_b));
        let d_live_rec = max_abs(
            &host_f32(st_a.recurrent_state()),
            &host_f32(st_b.recurrent_state()),
        );
        let d_live_conv = max_abs(&host_f32(st_a.conv_state()), &host_f32(st_b.conv_state()));
        let mut d_ck_rec = 0f32;
        let mut d_ck_conv = 0f32;
        for j in 0..m {
            d_ck_rec = d_ck_rec.max(max_abs(
                &host_f32(ckpts_a[j].recurrent_state()),
                &host_f32(ckpts_b[j].recurrent_state()),
            ));
            d_ck_conv = d_ck_conv.max(max_abs(
                &host_f32(ckpts_a[j].conv_state()),
                &host_f32(ckpts_b[j].conv_state()),
            ));
        }
        eprintln!(
            "Q36 gdn-mrow parity basis=(synthetic qwen3.6 geometry n_k={n_k} n_v={n_v} \
             d_k={d_k} d_v={d_v} k={kernel} hidden={hidden}, warm {} steps, m={m}, \
             chunk_arm={chunk_arm}): max_abs out={d_out:.5} live_rec={d_live_rec:.5} \
             live_conv={d_live_conv:.5} ckpt_rec={d_ck_rec:.5} ckpt_conv={d_ck_conv:.5} tol={tol}",
            WARM_STEPS_3_SO_PARITY_IS_MEASURED_FROM_A_NONZERO_STATE
        );
        for (name, d) in [
            ("out", d_out),
            ("live_rec", d_live_rec),
            ("live_conv", d_live_conv),
            ("ckpt_rec", d_ck_rec),
            ("ckpt_conv", d_ck_conv),
        ] {
            assert!(
                d <= tol,
                "m={m} chunk_arm={chunk_arm}: m-row verify diverged from the per-row fused \
                 loop on {name}: {d} > {tol}"
            );
        }
    }
    std::env::remove_var("NV_Q38_VERIFY_GDN_CHUNK");
}

#[test]
fn qkv_prep_rows_kernel_is_bitwise_the_m1_record_kernel_run_per_row() {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    use std::ffi::c_void;

    let Ok(device) = Device::new_cuda_with_stream(0) else {
        eprintln!("[skip] no cuda device; the qkv rows-kernel parity needs the card");
        return;
    };
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let (n_q, n_kv, hd, rotary) = Q36_ATTN_GEOMETRY_NQ16_NKV2_HD256_ROT64;
    let q_stride = 2 * hd;
    let m = 3usize;
    let base_pos = 7usize;
    let max_pos = base_pos + m + 2;
    let half = rotary / 2;
    let eps = 1e-6f32;

    let host_rand_bf16 = |n: usize, seed: u64| -> Vec<bf16> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                bf16::from_f32(((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4)
            })
            .collect()
    };
    let host_rand_f32 = |n: usize, seed: u64| -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect()
    };

    let q_raw = host_rand_bf16(m * n_q * q_stride, 0x1111);
    let k_raw = host_rand_bf16(m * n_kv * hd, 0x2222);
    let v_raw = host_rand_bf16(m * n_kv * hd, 0x3333);
    let qw = host_rand_bf16(hd, 0x4444);
    let kw = host_rand_bf16(hd, 0x5555);
    let cos_tab = host_rand_f32(max_pos * half, 0x6666);
    let sin_tab = host_rand_f32(max_pos * half, 0x7777);
    let positions: Vec<i32> = (0..m).map(|j| (base_pos + j) as i32).collect();

    let up = |v: &[bf16]| stream.clone_htod(v).expect("htod");
    let upf = |v: &[f32]| stream.clone_htod(v).expect("htod f32");
    let upi = |v: &[i32]| stream.clone_htod(v).expect("htod i32");
    #[allow(deprecated)]
    let (q_d, k_d, v_d) = (up(&q_raw), up(&k_raw), up(&v_raw));
    #[allow(deprecated)]
    let (qw_d, kw_d) = (up(&qw), up(&kw));
    #[allow(deprecated)]
    let (cos_d, sin_d) = (upf(&cos_tab), upf(&sin_tab));
    #[allow(deprecated)]
    let pos_d = upi(&positions);

    let cache_elems = max_pos * n_kv * hd;
    let scale_elems = max_pos * n_kv;
    let mut mk_outputs = || -> (
        cudarc::driver::CudaSlice<u8>,
        cudarc::driver::CudaSlice<u8>,
        cudarc::driver::CudaSlice<f32>,
        cudarc::driver::CudaSlice<f32>,
        cudarc::driver::CudaSlice<bf16>,
        cudarc::driver::CudaSlice<bf16>,
    ) {
        (
            stream.alloc_zeros::<u8>(cache_elems).expect("kc"),
            stream.alloc_zeros::<u8>(cache_elems).expect("vc"),
            stream.alloc_zeros::<f32>(scale_elems).expect("ks"),
            stream.alloc_zeros::<f32>(scale_elems).expect("vs"),
            stream.alloc_zeros::<bf16>(m * n_q * hd).expect("qo"),
            stream.alloc_zeros::<bf16>(m * n_q * hd).expect("qs"),
        )
    };
    let (mut kc_a, mut vc_a, mut ks_a, mut vs_a, mut qo_a, mut qs_a) = mk_outputs();
    let (mut kc_b, mut vc_b, mut ks_b, mut vs_b, mut qo_b, mut qs_b) = mk_outputs();

    {
        let (qp, _g0) = q_d.device_ptr(&stream);
        let (kp, _g1) = k_d.device_ptr(&stream);
        let (vp, _g2) = v_d.device_ptr(&stream);
        let (qwp, _g3) = qw_d.device_ptr(&stream);
        let (kwp, _g4) = kw_d.device_ptr(&stream);
        let (cp, _g5) = cos_d.device_ptr(&stream);
        let (sp, _g6) = sin_d.device_ptr(&stream);
        let (pp, _g7) = pos_d.device_ptr(&stream);
        let (kca, _g8) = kc_a.device_ptr_mut(&stream);
        let (vca, _g9) = vc_a.device_ptr_mut(&stream);
        let (ksa, _g10) = ks_a.device_ptr_mut(&stream);
        let (vsa, _g11) = vs_a.device_ptr_mut(&stream);
        let (qoa, _g12) = qo_a.device_ptr_mut(&stream);
        let (qsa, _g13) = qs_a.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::qkv_norm_rope_kvstore_fp8_rows(
                stream.cu_stream() as *mut c_void,
                qp as *const u16,
                kp as *const u16,
                vp as *const u16,
                qwp as *const u16,
                kwp as *const u16,
                cp as *const f32,
                sp as *const f32,
                pp as *const i32,
                kca as *mut u8,
                vca as *mut u8,
                ksa as *mut f32,
                vsa as *mut f32,
                qoa as *mut u16,
                qsa as *mut u16,
                m as i32,
                n_q as i32,
                n_kv as i32,
                hd as i32,
                q_stride as i32,
                rotary as i32,
                eps,
            )
        };
        assert_eq!(rc, 0, "qkv_norm_rope_kvstore_fp8_rows rc={rc}");
    }

    for j in 0..m {
        let pos_j: Vec<i32> = vec![positions[j]];
        #[allow(deprecated)]
        let pos_j_d = upi(&pos_j);
        let (qp, _g0) = q_d.device_ptr(&stream);
        let (kp, _g1) = k_d.device_ptr(&stream);
        let (vp, _g2) = v_d.device_ptr(&stream);
        let (qwp, _g3) = qw_d.device_ptr(&stream);
        let (kwp, _g4) = kw_d.device_ptr(&stream);
        let (cp, _g5) = cos_d.device_ptr(&stream);
        let (sp, _g6) = sin_d.device_ptr(&stream);
        let (pp, _g7) = pos_j_d.device_ptr(&stream);
        let (kcb, _g8) = kc_b.device_ptr_mut(&stream);
        let (vcb, _g9) = vc_b.device_ptr_mut(&stream);
        let (ksb, _g10) = ks_b.device_ptr_mut(&stream);
        let (vsb, _g11) = vs_b.device_ptr_mut(&stream);
        let (qob, _g12) = qo_b.device_ptr_mut(&stream);
        let (qsb, _g13) = qs_b.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::qkv_norm_rope_kvstore_fp8_decode(
                stream.cu_stream() as *mut c_void,
                (qp + (j * n_q * q_stride * 2) as u64) as *const u16,
                (kp + (j * n_kv * hd * 2) as u64) as *const u16,
                (vp + (j * n_kv * hd * 2) as u64) as *const u16,
                qwp as *const u16,
                kwp as *const u16,
                cp as *const f32,
                sp as *const f32,
                pp as *const i32,
                kcb as *mut u8,
                vcb as *mut u8,
                ksb as *mut f32,
                vsb as *mut f32,
                (qob + (j * n_q * hd * 2) as u64) as *mut u16,
                (qsb + (j * n_q * hd * 2) as u64) as *mut u16,
                n_q as i32,
                n_kv as i32,
                hd as i32,
                q_stride as i32,
                rotary as i32,
                eps,
            )
        };
        assert_eq!(rc, 0, "qkv_norm_rope_kvstore_fp8_decode row {j} rc={rc}");
        stream.synchronize().expect("row sync");
    }
    stream.synchronize().expect("final sync");

    #[allow(deprecated)]
    let read_u8 = |s: &cudarc::driver::CudaSlice<u8>| stream.memcpy_dtov(s).expect("dtoh u8");
    #[allow(deprecated)]
    let read_f32 = |s: &cudarc::driver::CudaSlice<f32>| stream.memcpy_dtov(s).expect("dtoh f32");
    #[allow(deprecated)]
    let read_bf = |s: &cudarc::driver::CudaSlice<bf16>| stream.memcpy_dtov(s).expect("dtoh bf16");

    assert_eq!(read_u8(&kc_a), read_u8(&kc_b), "k fp8 cache rows differ");
    assert_eq!(read_u8(&vc_a), read_u8(&vc_b), "v fp8 cache rows differ");
    assert_eq!(
        read_f32(&ks_a).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        read_f32(&ks_b).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "k scales differ"
    );
    assert_eq!(
        read_f32(&vs_a).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        read_f32(&vs_b).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "v scales differ"
    );
    assert_eq!(
        read_bf(&qo_a).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        read_bf(&qo_b).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "q_out differs"
    );
    assert_eq!(
        read_bf(&qs_a).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        read_bf(&qs_b).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "q_sig differs"
    );
    eprintln!(
        "Q36 qkv-prep rows-kernel parity basis=(synthetic qwen3.6 attention geometry \
         n_q={n_q} n_kv={n_kv} hd={hd} rot={rotary} m={m} gated): rows kernel BITWISE equals \
         the m=1 record kernel run per row (q_out, q_sig, fp8 kv rows, scales)"
    );
}

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN36_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("qwen3.6 snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no qwen3.6 snapshot; set NV_QWEN36_DIR")
}

const STORY_SEED: &str = "The lighthouse keeper had counted nine hundred and twelve storms from \
    the gallery deck, and he wrote each one into the ledger with the same steel pen. On the \
    morning the sea turned the color of old brass, he found a rowboat lodged in the rocks below \
    the light, its oars shipped neatly, its hull dry on the inside. There was no name on the \
    stern. There was a brass key on the center thwart, and under the key a note in his own \
    handwriting, dated forty years ahead. ";

fn prompt_of_depth(tok: &Tokenizer, depth: usize) -> Vec<u32> {
    let enc = |s: &str| -> Vec<u32> { tok.encode(s, false).expect("encode").get_ids().to_vec() };
    let head = enc("<|im_start|>user\n");
    let tail = enc(
        "\n\nContinue this story in the same style.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    );
    let seed = enc(STORY_SEED);
    let ctx_needed = depth
        .checked_sub(head.len() + tail.len())
        .expect("depth must fit the template frame");
    let mut out = head;
    let mut ctx: Vec<u32> = Vec::with_capacity(ctx_needed + seed.len());
    while ctx.len() < ctx_needed {
        ctx.extend_from_slice(&seed);
    }
    ctx.truncate(ctx_needed);
    out.extend_from_slice(&ctx);
    out.extend_from_slice(&tail);
    assert_eq!(out.len(), depth);
    out
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i as u32;
        }
    }
    best
}

fn prefill_host_routed(engine: &mut GraphedQwen3Moe, prompt: &[u32]) -> Vec<f32> {
    let stashed = engine
        .take_moe_dispatch_for_host_routed_prefill_then_restore_reinstalling_would_drop_and_rebuild_the_folded_weights_a_torn_down_graph_baked();
    let mut fed = 0usize;
    let mut last = Vec::new();
    while fed < prompt.len() {
        let n = 512.min(prompt.len() - fed);
        last = engine
            .prefill(&prompt[fed..fed + n])
            .unwrap_or_else(|e| panic!("prefill chunk at {fed}: {e:#}"));
        fed += n;
    }
    match stashed {
        Some(hook) => engine.set_moe_dispatch(Some(hook)),
        None => engine.install_grouped_moe().expect("grouped install"),
    }
    engine.set_device_routing(true);
    last
}

#[test]
#[ignore = "loads the ~22 GB Qwen3.6-35B NVFP4; set NV_QWEN36_TRUNK_PARITY=1 and run in \
            --release under the GPU lock -- eager (NV_GRAPH_OFF) engine-level A/B of the \
            verify trunk arms (NV_QWEN36_TRUNK_PARITY_ARMS subset of mrow,chunk,prep) on the \
            record stack: an m=1 greedy decode stream is the reference; both arm settings \
            verify the identical forced chains from that stream; the gates are (a) arms-on \
            verify-vs-decode argmax agreement >= arms-off at every rows shape (acceptance \
            must not regress) and (b) row logit rel-L2 on identical chains stays under a \
            loose regression ceiling -- chain identity vs the served stream lives in the \
            evidence/serving gates"]
fn qwen36_engine_verify_forced_chain_identity_across_trunk_arms() {
    if std::env::var("NV_QWEN36_TRUNK_PARITY").as_deref() != Ok("1") {
        panic!("set NV_QWEN36_TRUNK_PARITY=1 to run (it must never silently skip)");
    }
    for (k, v) in [
        ("NV_Q36_GRAPHED_DECODE_FIX", "1"),
        ("NV_Q36_GDN_FP8", "1"),
        ("NV_Q36_LM_HEAD_FP8", "1"),
        ("NV_Q38_GDN_SPLIT", "1"),
        ("NV_Q36_MOE_FUSED_CHAIN", "1"),
        ("NV_MOE_ROUTE_FUSE", "1"),
        ("NV_Q36_MOE_SHARED_FOLD", "1"),
        ("NV_MOE_FP4_DECODE_GEMV", "1"),
        ("NV_Q36_MOE_VERIFY_CHAIN", "1"),
    ] {
        std::env::set_var(k, v);
    }
    let captured_mode = std::env::var("NV_QWEN36_TRUNK_PARITY_GRAPH").as_deref() == Ok("1");
    if captured_mode {
        std::env::remove_var("NV_GRAPH_OFF");
    } else {
        std::env::set_var("NV_GRAPH_OFF", "1");
    }
    for k in [
        "NV_Q38_MROW_VERIFY",
        "NV_Q38_VERIFY_GDN_CHUNK",
        "NV_Q36_ATTN_VERIFY_PREP",
    ] {
        std::env::remove_var(k);
    }

    let dir = snapshot_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw).expect("config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let depth = 256usize;
    let rows_sweep = [2usize, 3, 5];
    let rounds = FORCED_CHAIN_ROUNDS_5_COMPOUND_STATE_DRIFT_ACROSS_COMMITS;
    let slots = depth + 64 + rounds * rows_sweep.len() * 2 + 16;
    let mut engine = GraphedQwen3Moe::new(model, &device, slots).expect("graphed engine");
    let vocab = engine.vocab_size();
    eprintln!(
        "[trunk-parity] basis: checkpoint=RedHatAI/Qwen3.6-35B-A3B-NVFP4 backend=cuda \
         record stack + verify chain, captured_mode={captured_mode}, depth={depth}, \
         rows sweep {rows_sweep:?}, {rounds} forced rounds per shape"
    );

    let prompt = prompt_of_depth(&tok, depth);
    let last = prefill_host_routed(&mut engine, &prompt);
    let anchor0 = argmax(&last);

    let stream_len = rounds * rows_sweep.iter().max().copied().unwrap_or(5) + 4;
    let mut decoded: Vec<u32> = vec![anchor0];
    for i in 0..stream_len {
        let prev = decoded[i];
        engine.forward_decode(prev).expect("m1 decode");
        decoded.push(argmax(&engine.logits_host().expect("m1 logits")));
    }
    eprintln!(
        "[trunk-parity] m=1 greedy reference stream of {} tokens recorded",
        decoded.len()
    );

    let arms = std::env::var("NV_QWEN36_TRUNK_PARITY_ARMS")
        .unwrap_or_else(|_| "mrow,chunk,prep".to_string());
    let mut per_config: Vec<(bool, Vec<(usize, usize, usize, Vec<Vec<f32>>)>)> = Vec::new();
    for arms_on in [false, true] {
        for (tag, k) in [
            ("mrow", "NV_Q38_MROW_VERIFY"),
            ("chunk", "NV_Q38_VERIFY_GDN_CHUNK"),
            ("prep", "NV_Q36_ATTN_VERIFY_PREP"),
        ] {
            let selected = arms_on && arms.split(',').any(|a| a.trim() == tag);
            if selected {
                std::env::set_var(k, "1");
            } else {
                std::env::remove_var(k);
            }
        }
        let mut per_rows = Vec::new();
        for &rows in &rows_sweep {
            engine.reset().expect("reset before walk");
            let last_w = prefill_host_routed(&mut engine, &prompt);
            assert_eq!(
                argmax(&last_w),
                anchor0,
                "prefill anchor changed (arms_on={arms_on}); the arms must not touch prefill"
            );
            engine.ensure_verify_lane(rows).expect("verify lane");
            let commit_step = std::env::var("NV_QWEN36_TRUNK_PARITY_COMMIT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(rows)
                .min(rows);
            let mut agree = 0usize;
            let mut total = 0usize;
            let mut logits_rounds: Vec<Vec<f32>> = Vec::new();
            let mut p = 0usize;
            while p + rows < decoded.len() {
                let chain = &decoded[p..p + rows];
                engine.forward_verify_chain(chain).expect("verify chain");
                let flat = engine.verify_logits_host().expect("verify logits");
                for r in 0..rows {
                    let pred = argmax(&flat[r * vocab..(r + 1) * vocab]);
                    total += 1;
                    if pred == decoded[p + r + 1] {
                        agree += 1;
                    }
                }
                logits_rounds.push(flat);
                engine
                    .commit_verify_consumed(commit_step)
                    .expect("commit walk");
                p += commit_step;
            }
            eprintln!(
                "[trunk-parity] arms_on={arms_on} arms={arms} rows={rows}: verify-vs-decode \
                 agreement {agree}/{total}"
            );
            per_rows.push((rows, agree, total, logits_rounds));
        }
        per_config.push((arms_on, per_rows));
    }

    let (_, off_rows) = &per_config[0];
    let (_, on_rows) = &per_config[1];
    let mut worst_rel = 0f64;
    for ((rows_o, agree_o, total_o, logits_o), (rows_n, agree_n, total_n, logits_n)) in
        off_rows.iter().zip(on_rows.iter())
    {
        assert_eq!(rows_o, rows_n);
        assert_eq!(total_o, total_n);
        for (flat_a, flat_b) in logits_o.iter().zip(logits_n.iter()) {
            for r in 0..*rows_o {
                let a = &flat_a[r * vocab..(r + 1) * vocab];
                let b = &flat_b[r * vocab..(r + 1) * vocab];
                let mut sum_sq = 0f64;
                let mut ref_sq = 0f64;
                for (x, y) in a.iter().zip(b.iter()) {
                    let d = (*x - *y) as f64;
                    sum_sq += d * d;
                    ref_sq += (*x as f64) * (*x as f64);
                }
                let rel = (sum_sq / ref_sq.max(1e-12)).sqrt();
                worst_rel = worst_rel.max(rel);
                assert!(
                    rel <= ENGINE_ROW_REL_L2_CEILING_3E1_REGRESSION_TRIP_ONLY_CHAIN_IDENTITY_IS_THE_CORRECTNESS_GATE,
                    "rows={rows_o} row={r}: rel_l2={rel:.3e} exceeds the regression ceiling \
                     on identical forced chains"
                );
            }
        }
        assert!(
            agree_n >= agree_o,
            "rows={rows_o}: the trunk arms LOWER verify-vs-decode agreement \
             ({agree_n}/{total_n} vs {agree_o}/{total_o}); that is a real acceptance \
             regression, not ulp reordering"
        );
    }
    eprintln!(
        "[trunk-parity] PASS arms={arms} worst row rel_l2={worst_rel:.3e}; arms-on \
         verify-vs-decode agreement is >= arms-off at every rows shape"
    );
}
