#![cfg(feature = "cuda")]

mod hub_dirs;
mod sufspec_classes;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const CHECKPOINT_LABEL: &str = "unsloth/Qwen3.8-27B-NVFP4";

const PROMPT_TOKENS_1024_MATCHES_THE_M_INVARIANCE_AND_TREE_GATES: usize = 1024;
const MAX_SEQ_2048_COVERS_PROMPT_CHAIN_BUILD_AND_THE_M4_VERIFY: usize = 2048;
const VERIFY_ROWS_M4_THE_CHAIN_SHAPE_THE_ROUTE_SERVES: usize = 4;

const LIVE_TENSOR_CROSS_ARM_BAND_ABS_2E2_REL_HALF_PCT_IS_5X_THE_MEASURED_2_BF16_ULP_SCORE_REORDER:
    (f32, f32) = (2e-2, 0.005);

const STREAM_IDENTITY_IS_NOT_GATEABLE_FOR_SCORE_REORDERING_ARMS: &str =
    "The mma QK tile sums exact f16xf16 products in a different association than the mk \
     kernel's f32 fma chain, so per-position scores differ at f32-ulp scale and the bf16 \
     attention output moves by 1-2 ulp per layer. The MoE router is discrete: a 1-ulp router \
     input shift near a top-k tie picks a different expert, and 48 routed layers amplify that \
     into confidently different greedy streams -- the same mechanism moves the shipped \
     rowshared arm's m=8 row-0 logits, where warps_on_mi=1 reorders the l/acc position sums. \
     Two confident streams disagreeing is therefore not adjudicable as corruption-vs-tie at \
     stream level for this arm; the gateable losslessness statement is per-layer cross-arm \
     parity on live serving tensors, which this suite asserts, and the arm ships default-off \
     with no knee route because the emitted-stream discipline cannot certify it.";

fn require_gate(gate: &str) {
    if std::env::var(gate).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {gate}=1");
    }
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 once; set NV_QWEN38_SERVING_TEST=1 -- the \
            tctile verify arm losslessness gate: a real m=4 verify forward at depth 1024 dumps \
            every full-attention slot's live q/kv tensors, and each slot is replayed through \
            the mk and tctile launchers, which must agree per row within the documented \
            2-bf16-ulp score-reorder band; STREAM_IDENTITY_IS_NOT_GATEABLE_FOR_SCORE_REORDERING_ARMS \
            states why this, and not emitted-stream identity, is the gateable property"]
fn qwen38_tctile_per_slot_live_tensor_parity_vs_the_mk_arm_on_a_real_verify_forward() {
    use candle_core::Device;
    use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
    use nv_models::graph_engine::GraphedQwen3Moe;
    use nv_models::qwen3_5_moe::{
        Qwen3Moe, Qwen3_5DenseConfig,
        MK_VERIFY_DUMP_DIR_ENV_WRITES_EACH_SLOTS_QKV_ONCE_SO_A_PARITY_GATE_CAN_REPLAY_LIVE_SERVING_TENSORS_THROUGH_BOTH_ARMS,
    };

    require_gate("NV_QWEN38_SERVING_TEST");
    let dir = match std::env::var("NV_QWEN38_DIR") {
        Ok(d) if !d.is_empty() => std::path::PathBuf::from(d),
        _ => hub_dirs::snapshot(NVFP4_REPO, &["config.json", "tokenizer.json"]).unwrap_or_else(
            || {
                panic!(
                    "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: no hydrated {NVFP4_REPO} \
                     snapshot; set NV_QWEN38_DIR"
                )
            },
        ),
    };
    let tok =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let depth = PROMPT_TOKENS_1024_MATCHES_THE_M_INVARIANCE_AND_TREE_GATES;
    let prompt = sufspec_classes::tokens_of(
        &tok,
        &sufspec_classes::agentic_transcript_text(4 + depth / 128),
        depth,
        "agentic",
    );

    let dump_dir = std::env::temp_dir().join(format!("q38-tctile-parity-{}", std::process::id()));
    std::fs::create_dir_all(&dump_dir).expect("dump dir");
    std::env::set_var(
        MK_VERIFY_DUMP_DIR_ENV_WRITES_EACH_SLOTS_QKV_ONCE_SO_A_PARITY_GATE_CAN_REPLAY_LIVE_SERVING_TENSORS_THROUGH_BOTH_ARMS,
        &dump_dir,
    );

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = nv_weights::QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("track-2 dense NVFP4 loader");
    drop(weights);
    let mut engine = GraphedQwen3Moe::new(
        base,
        &device,
        MAX_SEQ_2048_COVERS_PROMPT_CHAIN_BUILD_AND_THE_M4_VERIFY,
    )
    .expect("graphed engine");
    engine.install_grouped_moe().expect("routing install");
    engine.set_device_routing(false);

    let m = VERIFY_ROWS_M4_THE_CHAIN_SHAPE_THE_ROUTE_SERVES;
    engine.ensure_verify_lane(m).expect("verify lane");
    engine.synchronize().expect("post-lane sync");
    let mut fed = 0usize;
    let mut last: Vec<f32> = Vec::new();
    while fed < prompt.len() {
        let n = 512.min(prompt.len() - fed);
        last = engine
            .prefill(&prompt[fed..fed + n])
            .unwrap_or_else(|e| panic!("prefill chunk at {fed}: {e:#}"));
        fed += n;
    }
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32
    };
    let mut chain: Vec<u32> = vec![argmax(&last)];
    for _ in 0..m - 1 {
        engine
            .forward_decode(*chain.last().unwrap())
            .expect("chain-build decode");
        chain.push(argmax(&engine.logits_host().expect("chain-build logits")));
    }
    engine
        .forward_verify_chain(&chain)
        .expect("dumping verify forward");
    std::env::remove_var(
        MK_VERIFY_DUMP_DIR_ENV_WRITES_EACH_SLOTS_QKV_ONCE_SO_A_PARITY_GATE_CAN_REPLAY_LIVE_SERVING_TENSORS_THROUGH_BOTH_ARMS,
    );

    let ctx = CudaContext::new(0).expect("cuda ctx");
    let stream = ctx.default_stream();
    let mut slots_checked = 0usize;
    let mut worst_overall = 0f32;
    for slot in 0..256usize {
        let meta_path = dump_dir.join(format!("slot{slot}.meta"));
        let Ok(meta) = std::fs::read_to_string(&meta_path) else {
            continue;
        };
        let field = |k: &str| -> f64 {
            meta.split_whitespace()
                .find_map(|t| t.strip_prefix(&format!("{k}=")))
                .unwrap_or_else(|| panic!("meta field {k} in {meta}"))
                .parse::<f64>()
                .unwrap()
        };
        let (total, mm, n_q, n_kv, hd) = (
            field("total") as usize,
            field("m") as usize,
            field("n_q") as usize,
            field("n_kv") as usize,
            field("hd") as usize,
        );
        let scaling = field("scaling") as f32;
        assert_eq!(mm, m, "slot{slot}: dumped m");
        let rd = |sfx: &str| std::fs::read(dump_dir.join(format!("slot{slot}.{sfx}.bin"))).unwrap();
        let hq: Vec<u16> = rd("q")
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let hk = rd("k");
        let hv = rd("v");
        let f32s = |b: Vec<u8>| -> Vec<f32> {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let hks = f32s(rd("ks"));
        let hvs = f32s(rd("vs"));

        let dq: CudaSlice<u16> = stream.clone_htod(&hq).unwrap();
        let dk: CudaSlice<u8> = stream.clone_htod(&hk).unwrap();
        let dv: CudaSlice<u8> = stream.clone_htod(&hv).unwrap();
        let dks: CudaSlice<f32> = stream.clone_htod(&hks).unwrap();
        let dvs: CudaSlice<f32> = stream.clone_htod(&hvs).unwrap();
        let dnc: CudaSlice<i32> = stream.clone_htod(&[total as i32]).unwrap();
        let mut dscr: CudaSlice<f32> =
            stream.alloc_zeros::<f32>(n_q * 8 * 32 * (hd + 2)).unwrap();
        let mut dfan: CudaSlice<u32> = stream.alloc_zeros::<u32>(n_q).unwrap();
        let mut dmk: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n_q * hd).unwrap();
        let mut dtc: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n_q * hd).unwrap();
        {
            let (pq, _a) = dq.device_ptr(&stream);
            let (pk, _b) = dk.device_ptr(&stream);
            let (pv, _c) = dv.device_ptr(&stream);
            let (pks, _d) = dks.device_ptr(&stream);
            let (pvs, _e) = dvs.device_ptr(&stream);
            let (pn, _f) = dnc.device_ptr(&stream);
            let (ps, _g) = dscr.device_ptr_mut(&stream);
            let (pf, _h) = dfan.device_ptr_mut(&stream);
            let (pmk, _i) = dmk.device_ptr_mut(&stream);
            let (ptc, _j) = dtc.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::flash_decode_fused_fp8kv_mk(
                    stream.cu_stream() as *mut _,
                    pq as *const u16,
                    pk as *const u8,
                    pv as *const u8,
                    pks as *const f32,
                    pvs as *const f32,
                    pmk as *mut u16,
                    pn as *const i32,
                    0,
                    m as i32,
                    ps as *mut f32,
                    pf as *mut u32,
                    n_q as i32,
                    n_kv as i32,
                    hd as i32,
                    0,
                    0,
                    scaling,
                )
            };
            assert_eq!(rc, 0, "slot{slot}: mk launcher refused");
            let rc = unsafe {
                nv_kernels::cuda::flash_decode_rowshared_tctile_fp8kv_mk_paged(
                    stream.cu_stream() as *mut _,
                    pq as *const u16,
                    pk as *const u8,
                    pv as *const u8,
                    pks as *const f32,
                    pvs as *const f32,
                    ptc as *mut u16,
                    pn as *const i32,
                    ps as *mut f32,
                    pf as *mut u32,
                    n_q as i32,
                    n_kv as i32,
                    hd as i32,
                    0,
                    0,
                    0,
                    (8 * 32 / m) as i32,
                    scaling,
                    std::ptr::null(),
                    0,
                    0,
                    m as i32,
                    m as i32,
                    0,
                    0,
                )
            };
            assert_eq!(rc, 0, "slot{slot}: tctile launcher refused");
        }
        stream.synchronize().unwrap();
        let out_mk = stream.clone_dtoh(&dmk).unwrap();
        let out_tc = stream.clone_dtoh(&dtc).unwrap();
        let (abs_band, rel_band) =
            LIVE_TENSOR_CROSS_ARM_BAND_ABS_2E2_REL_HALF_PCT_IS_5X_THE_MEASURED_2_BF16_ULP_SCORE_REORDER;
        for r in 0..m {
            let seg = r * n_q * hd..(r + 1) * n_q * hd;
            let max_mag = out_mk[seg.clone()]
                .iter()
                .map(|b| half::bf16::from_bits(*b).to_f32().abs())
                .fold(0f32, f32::max);
            let worst = out_mk[seg.clone()]
                .iter()
                .zip(out_tc[seg].iter())
                .map(|(a, b)| {
                    (half::bf16::from_bits(*a).to_f32() - half::bf16::from_bits(*b).to_f32()).abs()
                })
                .fold(0f32, f32::max);
            worst_overall = worst_overall.max(worst);
            assert!(
                worst.is_finite() && worst <= abs_band + rel_band * max_mag,
                "slot{slot} row{r}: tctile vs mk on the live serving tensors differs by {worst} \
                 (max|mk|={max_mag}), beyond the 2-bf16-ulp score-reorder band; on identical \
                 inputs this is an indexing or masking bug, not chaos -- see \
                 STREAM_IDENTITY_IS_NOT_GATEABLE_FOR_SCORE_REORDERING_ARMS for the discipline"
            );
        }
        slots_checked += 1;
    }
    let _ = std::fs::remove_dir_all(&dump_dir);
    assert!(
        slots_checked >= 8,
        "only {slots_checked} full-attention slots were dumped and replayed; the qwen38 trunk \
         carries more, so the dump hook or the walk is broken and this gate proved nothing"
    );
    eprintln!(
        "TCTILE-PARITY slots={slots_checked} m={m} depth={depth} worst_cross_arm={worst_overall:.4} \
         checkpoint={CHECKPOINT_LABEL}"
    );
}
