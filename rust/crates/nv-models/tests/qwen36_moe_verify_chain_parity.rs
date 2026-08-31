#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::qwen3_5_moe::{GroupedMoeDispatch, MoeDispatch, Qwen3Moe, Qwen3MoeConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;

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

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn host_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flat")
        .to_vec1()
        .expect("host")
}

#[test]
#[ignore = "loads the ~22 GB Qwen3.6-35B NVFP4; set NV_QWEN36_VCHAIN_PARITY=1 and run in \
            --release under the GPU lock -- dispatch-level A/B of the m<=8 fused verify chain: \
            (a) per-row agreement with the proven m=1 fused chain within gate-GEMM ulp jitter \
            (the MoE layer is per-token pointwise; only the router GEMM shape differs), \
            (b) against the generic grouped verify at the same m: BITWISE when the shared \
            expert is NOT folded, loose rel-L2 under NV_Q36_MOE_SHARED_FOLD=1 (fold routes the \
            shared expert through the quantized grouped tile, an accepted arm-level change)"]
fn qwen36_moe_verify_chain_dispatch_ab_parity() {
    if std::env::var("NV_QWEN36_VCHAIN_PARITY").as_deref() != Ok("1") {
        panic!("set NV_QWEN36_VCHAIN_PARITY=1 to run (it must never silently skip)");
    }
    let fold = std::env::var("NV_Q36_MOE_SHARED_FOLD").as_deref() == Ok("1");
    for (k, v) in [
        ("NV_MOE_ROUTE_FUSE", "1"),
        ("NV_Q36_MOE_FUSED_CHAIN", "1"),
        ("NV_MOE_FP4_DECODE_GEMV", "1"),
    ] {
        std::env::set_var(k, v);
    }

    let dir = snapshot_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw).expect("config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);
    let hidden = model.config().hidden_size;
    let n_layers = model.config().num_hidden_layers;
    let mut dispatch = GroupedMoeDispatch::from_model(&model).expect("dispatch");
    eprintln!(
        "[vchain-parity] basis: checkpoint=RedHatAI/Qwen3.6-35B-A3B-NVFP4 backend=cuda \
         shared_fold={fold} arms=route_fuse+fused_chain+decode_gemv, synthetic BF16 rows \
         (dispatch-level parity is routing-agnostic), layers 0 and {}",
        n_layers - 1
    );

    let mut worst_rel = 0f64;
    for &layer_idx in &[0usize, n_layers - 1] {
        for &m in &[2usize, 3, 5, 8] {
            let mut rng = Rng(0x9e3779b97f4a7c15 ^ ((m as u64) << 8) ^ ((layer_idx as u64) << 16));
            let x_host: Vec<f32> = (0..m * hidden).map(|_| rng.unit() - 0.5).collect();
            let x = Tensor::from_vec(x_host, (m, hidden), &device)
                .expect("x")
                .to_dtype(DType::BF16)
                .expect("bf16");
            let moe = model
                .moe_block_for_dispatch_parity_tests(layer_idx)
                .expect("layer")
                .as_moe()
                .expect("moe layer");

            dispatch.ab_toggle_verify_chain_small_m_for_parity_tests(true);
            let y_chain = dispatch.forward(layer_idx, moe, &x).expect("chain forward");
            assert_eq!(y_chain.dims(), &[m, hidden]);
            let chain = host_f32(&y_chain);
            let nonfinite = chain.iter().filter(|v| !v.is_finite()).count();
            assert_eq!(nonfinite, 0, "m={m} layer={layer_idx}: chain output non-finite");

            let mut worst_row_rel = 0f64;
            let mut rerouted_rows = 0usize;
            for r in 0..m {
                let xr = x.narrow(0, r, 1).expect("row").contiguous().expect("c");
                let yr = dispatch.forward(layer_idx, moe, &xr).expect("m1 forward");
                let m1 = host_f32(&yr);
                let mut sum_sq = 0f64;
                let mut ref_sq = 0f64;
                for h in 0..hidden {
                    let d = (chain[r * hidden + h] - m1[h]) as f64;
                    sum_sq += d * d;
                    ref_sq += (m1[h] as f64) * (m1[h] as f64);
                }
                let rel = (sum_sq / ref_sq.max(1e-12)).sqrt();
                if rel > 2e-2 {
                    rerouted_rows += 1;
                } else {
                    worst_row_rel = worst_row_rel.max(rel);
                }
            }
            assert!(
                rerouted_rows <= 1,
                "m={m} layer={layer_idx}: {rerouted_rows} rows diverge from the m=1 fused \
                 chain; the m-row gate GEMM's ulp jitter can flip a near-tie top-k boundary \
                 on at most the occasional row -- more than one is a chain defect (the \
                 bitwise chain-vs-generic assert below is the authoritative gate)"
            );

            dispatch.ab_toggle_verify_chain_small_m_for_parity_tests(false);
            let y_gen = dispatch.forward(layer_idx, moe, &x).expect("generic verify");
            let generic = host_f32(&y_gen);
            let mut sum_sq = 0f64;
            let mut ref_sq = 0f64;
            let mut bitwise = true;
            for (a, b) in chain.iter().zip(generic.iter()) {
                if a.to_bits() != b.to_bits() {
                    bitwise = false;
                }
                sum_sq += ((a - b) as f64) * ((a - b) as f64);
                ref_sq += (*b as f64) * (*b as f64);
            }
            let rel_l2 = (sum_sq / ref_sq.max(1e-12)).sqrt();
            worst_rel = worst_rel.max(rel_l2);
            eprintln!(
                "[vchain-parity] layer={layer_idx} m={m} rows_vs_m1_worst_rel_l2={worst_row_rel:.3e} \
                 rerouted_rows={rerouted_rows} chain_vs_generic bitwise={bitwise} rel_l2={rel_l2:.3e}"
            );
            if !fold {
                assert!(
                    bitwise,
                    "m={m} layer={layer_idx}: without the shared fold the chain arithmetic is \
                     the generic path's reordered-not-changed sum, so any bit difference is a \
                     defect (rel_l2={rel_l2:.3e})"
                );
            } else {
                assert!(
                    rel_l2 <= 5e-2,
                    "m={m} layer={layer_idx}: folded chain vs generic verify rel_l2={rel_l2} \
                     exceeds the arm-level band"
                );
            }
        }
    }
    eprintln!("[vchain-parity] PASS worst chain_vs_generic rel_l2={worst_rel:.3e} fold={fold}");
}

#[test]
#[ignore = "loads the ~22 GB Qwen3.6-35B NVFP4; set NV_QWEN36_MCHAIN_PARITY=1 and run in \
            --release under the GPU lock -- with NV_Q36_MOE_ROUTER_GEMV + NV_MOE_FP4_GEMV_MROW \
            the m<=8 chain's router logits are per-row bitwise the m=1 gemv logits and the \
            grouped gemvs are per-tile bitwise the m1 kernel, so every chain row must equal \
            its m=1 forward BIT-FOR-BIT; also proves the router arm leaves m=1 itself bitwise \
            unchanged (arm-on m=1 vs weights-swapped-out m=1)"]
fn qwen36_moe_mchain_arms_make_the_chain_a_per_row_bitwise_twin() {
    if std::env::var("NV_QWEN36_MCHAIN_PARITY").as_deref() != Ok("1") {
        panic!("set NV_QWEN36_MCHAIN_PARITY=1 to run (it must never silently skip)");
    }
    let fold = std::env::var("NV_Q36_MOE_SHARED_FOLD").as_deref() == Ok("1");
    for (k, v) in [
        ("NV_MOE_ROUTE_FUSE", "1"),
        ("NV_Q36_MOE_FUSED_CHAIN", "1"),
        ("NV_MOE_FP4_DECODE_GEMV", "1"),
        ("NV_Q36_MOE_VERIFY_CHAIN", "1"),
        ("NV_Q36_MOE_ROUTER_GEMV", "1"),
        ("NV_MOE_FP4_GEMV_MROW", "1"),
    ] {
        std::env::set_var(k, v);
    }

    let dir = snapshot_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw).expect("config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);
    let hidden = model.config().hidden_size;
    let n_layers = model.config().num_hidden_layers;
    let mut dispatch = GroupedMoeDispatch::from_model(&model).expect("dispatch");
    eprintln!(
        "[mchain-twin] basis: checkpoint=RedHatAI/Qwen3.6-35B-A3B-NVFP4 backend=cuda \
         shared_fold={fold} arms=route_fuse+fused_chain+decode_gemv+verify_chain+\
         router_gemv+gemv_mrow, synthetic BF16 rows, layers 0 and {}",
        n_layers - 1
    );

    for &layer_idx in &[0usize, n_layers - 1] {
        for &m in &[2usize, 3, 5, 8] {
            let mut rng = Rng(0x51c36a17 ^ ((m as u64) << 8) ^ ((layer_idx as u64) << 16));
            let x_host: Vec<f32> = (0..m * hidden).map(|_| rng.unit() - 0.5).collect();
            let x = Tensor::from_vec(x_host, (m, hidden), &device)
                .expect("x")
                .to_dtype(DType::BF16)
                .expect("bf16");
            let moe = model
                .moe_block_for_dispatch_parity_tests(layer_idx)
                .expect("layer")
                .as_moe()
                .expect("moe layer");

            let y_chain = dispatch.forward(layer_idx, moe, &x).expect("chain forward");
            let chain = host_f32(&y_chain);
            let nonfinite = chain.iter().filter(|v| !v.is_finite()).count();
            assert_eq!(nonfinite, 0, "m={m} layer={layer_idx}: chain output non-finite");

            for r in 0..m {
                let xr = x.narrow(0, r, 1).expect("row").contiguous().expect("c");
                let yr = dispatch.forward(layer_idx, moe, &xr).expect("m1 forward");
                let m1 = host_f32(&yr);
                let diffs = (0..hidden)
                    .filter(|&h| chain[r * hidden + h].to_bits() != m1[h].to_bits())
                    .count();
                assert_eq!(
                    diffs, 0,
                    "m={m} layer={layer_idx} row={r}: chain row diverges from its m=1 \
                     forward in {diffs} of {hidden} elements -- with the router gemv arm the \
                     per-row logits are bitwise the m=1 logits, so any difference is a defect"
                );

                let stashed = dispatch.ab_swap_router_gemv_weights_for_parity_tests(None);
                assert!(
                    stashed.is_some(),
                    "router gemv arm was requested but from_model built no weights"
                );
                let yr_off = dispatch.forward(layer_idx, moe, &xr).expect("m1 arm-off");
                let m1_off = host_f32(&yr_off);
                dispatch.ab_swap_router_gemv_weights_for_parity_tests(stashed);
                let off_diffs = (0..hidden)
                    .filter(|&h| m1_off[h].to_bits() != m1[h].to_bits())
                    .count();
                assert_eq!(
                    off_diffs, 0,
                    "m=1 layer={layer_idx} row={r}: the router gemv arm changes the m=1 \
                     chain output ({off_diffs} of {hidden} elements) -- it must be bitwise \
                     the gemv_bf16 + cast + dtod sequence it replaces"
                );
            }
            eprintln!(
                "[mchain-twin] layer={layer_idx} m={m} all rows BITWISE == their m=1 \
                 forwards; m=1 arm-on == arm-off BITWISE"
            );
        }
    }
    eprintln!("[mchain-twin] PASS fold={fold}");
}
