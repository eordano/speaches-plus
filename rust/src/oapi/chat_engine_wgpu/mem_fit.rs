use std::path::{Path, PathBuf};

use super::{classify_wgpu_model, WgpuModelKind};

pub const MARGINAL_WEIGHT_FACTOR: f64 = 0.983;
pub const PROCESS_FIXED_GIB: f64 = 3.612;

pub const REPACK_LOAD_PEAK_FACTOR: f64 = 2.34;

pub const NATIVE_PACKED_LOAD_PEAK_FACTOR: f64 = 1.10;

const WEIGHT_EXTENSIONS: [&str; 5] = ["safetensors", "bin", "gguf", "pt", "pth"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadPath {
    NativePacked,
    HostDequant,
    PlainWeights,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fit {
    Fits,
    WontFit {
        estimated_gib: f64,
        available_gib: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelEstimate {
    pub weight_gib: f64,
    pub steady_gib: f64,
    pub load_peak_gib: f64,
    pub prefill_scratch_gib: f64,
}

pub fn decide(estimates: &[Option<ModelEstimate>], available_gib: f64) -> Vec<Fit> {
    let mut resident = PROCESS_FIXED_GIB;
    let mut out = Vec::with_capacity(estimates.len());
    for est in estimates {
        match est {
            None => out.push(Fit::Fits),
            Some(e) => {
                let peak = e.load_peak_gib + e.prefill_scratch_gib;
                if resident + peak <= available_gib {
                    resident += e.steady_gib + e.prefill_scratch_gib;
                    out.push(Fit::Fits);
                } else {
                    out.push(Fit::WontFit {
                        estimated_gib: peak,
                        available_gib,
                    });
                }
            }
        }
    }
    out
}

fn weight_file_bytes(dir: &Path) -> Option<u64> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut total = 0u64;
    let mut saw_any = false;
    for entry in entries.flatten() {
        let path = entry.path();
        let is_weight = matches!(
            path.extension().and_then(|e| e.to_str()),
            Some(ext) if WEIGHT_EXTENSIONS.contains(&ext)
        );
        if !is_weight {
            continue;
        }
        if let Ok(meta) = std::fs::metadata(&path) {
            total += meta.len();
            saw_any = true;
        }
    }
    saw_any.then_some(total)
}

pub fn load_path(dir: &Path) -> LoadPath {
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return LoadPath::HostDequant;
    };
    if !(text.contains("quantization_config") || text.contains("quant_method")) {
        return LoadPath::PlainWeights;
    }
    match classify_wgpu_model(&text) {
        Ok(
            WgpuModelKind::Qwen3_5Moe
            | WgpuModelKind::Qwen3_5Dense
            | WgpuModelKind::GptOss
            | WgpuModelKind::Laguna
            | WgpuModelKind::Gemma4E4b,
        ) => LoadPath::NativePacked,
        Ok(WgpuModelKind::Gemma4Dense | WgpuModelKind::Gemma4Moe) | Err(_) => LoadPath::HostDequant,
    }
}

fn prefill_scratch_gib(dir: &Path, max_seq: usize) -> f64 {
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return 0.0;
    };
    match classify_wgpu_model(&text) {
        Ok(WgpuModelKind::GptOss) => {
            let m = nv_models::gpt_oss_wgpu::prefill_m();
            if m == 0 || max_seq == 0 {
                return 0.0;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
                return 0.0;
            };
            let n_heads = v
                .get("num_attention_heads")
                .and_then(|x| x.as_u64())
                .unwrap_or(0) as usize;
            if n_heads == 0 {
                return 0.0;
            }
            (m * n_heads * max_seq * 4) as f64 / (1u64 << 30) as f64
        }
        Ok(WgpuModelKind::Qwen3_5Moe) => {
            let m = nv_models::qwen3_5_moe_wgpu::prefill_m();
            if m == 0 {
                return 0.0;
            }
            let legacy = m
                * nv_models::qwen3_5_moe_wgpu::prefill_list_bytes_per_token_charged_by_mem_fit();
            let mrow = match nv_models::qwen3_5_moe::Qwen3MoeConfig::from_hf_json_file(
                &dir.join("config.json"),
            ) {
                Ok(cfg) => nv_models::qwen3_5_moe_wgpu::pf_mrow_scratch_bytes_upper_bound_for_mem_fit(
                    &cfg,
                ),
                Err(_) => 0,
            };
            (legacy + mrow) as f64 / (1u64 << 30) as f64
        }
        Ok(WgpuModelKind::Laguna) => {
            let m = nv_models::laguna_wgpu::prefill_m();
            if m == 0 {
                return 0.0;
            }
            (m * nv_models::laguna_wgpu::prefill_list_bytes_per_token_charged_by_mem_fit()) as f64
                / (1u64 << 30) as f64
        }
        _ => 0.0,
    }
}

pub fn estimate_model(dir: &Path) -> Option<ModelEstimate> {
    estimate_model_with_max_seq(dir, 0)
}

pub fn estimate_model_with_max_seq(dir: &Path, max_seq: usize) -> Option<ModelEstimate> {
    let weight_gib = weight_file_bytes(dir)? as f64 / (1u64 << 30) as f64;
    let steady_gib = weight_gib * MARGINAL_WEIGHT_FACTOR;
    let load_peak_gib = match load_path(dir) {
        LoadPath::HostDequant => weight_gib * REPACK_LOAD_PEAK_FACTOR,
        LoadPath::NativePacked => weight_gib * NATIVE_PACKED_LOAD_PEAK_FACTOR,
        LoadPath::PlainWeights => steady_gib,
    };
    Some(ModelEstimate {
        weight_gib,
        steady_gib,
        load_peak_gib,
        prefill_scratch_gib: prefill_scratch_gib(dir, max_seq),
    })
}

pub fn estimate_model_gib(dir: &Path) -> Option<f64> {
    estimate_model(dir).map(|e| e.steady_gib)
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn mach_host_self() -> libc::host_t;
}

#[cfg(target_os = "macos")]
pub fn available_memory_gib() -> Option<f64> {
    let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = (std::mem::size_of::<libc::vm_statistics64>() / std::mem::size_of::<i32>())
        as libc::mach_msg_type_number_t;
    let rc = unsafe {
        libc::host_statistics64(
            mach_host_self(),
            libc::HOST_VM_INFO64,
            &mut stats as *mut _ as *mut i32,
            &mut count,
        )
    };
    if rc != libc::KERN_SUCCESS {
        return None;
    }
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as f64;
    let reclaimable = stats.free_count as f64
        + stats.inactive_count as f64
        + stats.purgeable_count as f64
        + stats.external_page_count as f64;
    Some(reclaimable * page / (1u64 << 30) as f64)
}

#[cfg(not(target_os = "macos"))]
pub fn available_memory_gib() -> Option<f64> {
    None
}

fn guard_disabled() -> bool {
    matches!(
        std::env::var("NV_MEM_FIT").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub fn plan(dirs: &[PathBuf]) -> Option<(Vec<Fit>, f64)> {
    if guard_disabled() {
        return None;
    }
    let available = available_memory_gib()?;
    let estimates: Vec<Option<ModelEstimate>> = dirs
        .iter()
        .map(|d| estimate_model_with_max_seq(d, super::WgpuChatEngine::default_max_seq_for(d)))
        .collect();
    Some((decide(&estimates, available), available))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(weight_gib: f64) -> Option<ModelEstimate> {
        let steady = weight_gib * MARGINAL_WEIGHT_FACTOR;
        Some(ModelEstimate {
            weight_gib,
            steady_gib: steady,
            load_peak_gib: steady,
            prefill_scratch_gib: 0.0,
        })
    }

    fn repacked(weight_gib: f64) -> Option<ModelEstimate> {
        Some(ModelEstimate {
            weight_gib,
            steady_gib: weight_gib * MARGINAL_WEIGHT_FACTOR,
            load_peak_gib: weight_gib * REPACK_LOAD_PEAK_FACTOR,
            prefill_scratch_gib: 0.0,
        })
    }

    fn native(weight_gib: f64) -> Option<ModelEstimate> {
        Some(ModelEstimate {
            weight_gib,
            steady_gib: weight_gib * MARGINAL_WEIGHT_FACTOR,
            load_peak_gib: weight_gib * NATIVE_PACKED_LOAD_PEAK_FACTOR,
            prefill_scratch_gib: 0.0,
        })
    }

    fn fixture_dir(name: &str, config: Option<&str>) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mem-fit-fixture-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(text) = config {
            std::fs::write(dir.join("config.json"), text).unwrap();
        }
        dir
    }

    #[test]
    fn all_fit_under_budget() {
        let out = decide(&[bf16(10.0), bf16(10.0)], 25.0);
        assert_eq!(out, vec![Fit::Fits, Fit::Fits]);
    }

    #[test]
    fn second_model_skipped_when_over_budget() {
        let out = decide(&[bf16(15.0), bf16(17.0)], 24.0);
        assert_eq!(out[0], Fit::Fits);
        assert!(matches!(out[1], Fit::WontFit { .. }));
    }

    #[test]
    fn unknown_estimate_never_blocks() {
        let out = decide(&[None, bf16(1000.0)], 1.0);
        assert_eq!(out[0], Fit::Fits);
        assert!(matches!(out[1], Fit::WontFit { .. }));
    }

    #[test]
    fn measured_four_model_boot_is_not_over_charged() {
        let weights = [14.894_f64, 30.392, 23.324, 20.095];
        let predicted: f64 = PROCESS_FIXED_GIB
            + weights
                .iter()
                .map(|w| w * MARGINAL_WEIGHT_FACTOR)
                .sum::<f64>();
        let measured = 90.16;
        assert!(
            (predicted - measured).abs() < 2.0,
            "additive model predicted {predicted:.2} GiB against {measured:.2} measured"
        );
        let old_model: f64 = weights.iter().map(|w| w * 1.3).sum();
        assert!(
            old_model > measured + 20.0,
            "the old 1.3x model should visibly over-charge: {old_model:.2} vs {measured:.2}"
        );
    }

    #[test]
    fn load_transient_is_refused_even_when_steady_state_would_fit() {
        let est = [repacked(30.392)];
        let steady_only_budget = 40.0;
        assert!(
            matches!(decide(&est, steady_only_budget)[0], Fit::WontFit { .. }),
            "a budget covering only the steady state must be refused"
        );
        assert_eq!(decide(&est, 75.0), vec![Fit::Fits]);
    }

    #[test]
    fn bf16_is_not_charged_the_repack_transient() {
        assert_eq!(decide(&[bf16(14.894)], 20.0), vec![Fit::Fits]);
        assert!(matches!(
            decide(&[repacked(14.894)], 20.0)[0],
            Fit::WontFit { .. }
        ));
    }

    #[test]
    fn process_fixed_cost_is_charged_once() {
        let one = decide(&[bf16(10.0)], PROCESS_FIXED_GIB + 9.83 + 0.01);
        assert_eq!(one, vec![Fit::Fits]);
        let two = decide(&[bf16(10.0), bf16(10.0)], PROCESS_FIXED_GIB + 19.66 + 0.01);
        assert_eq!(two, vec![Fit::Fits, Fit::Fits]);
    }

    #[test]
    fn native_packed_is_not_charged_the_repack_transient() {
        let est = [native(23.324)];
        assert_eq!(decide(&est, 32.4), vec![Fit::Fits]);
        assert!(matches!(
            decide(&[repacked(23.324)], 32.4)[0],
            Fit::WontFit { .. }
        ));
        assert!(matches!(decide(&est, 20.0)[0], Fit::WontFit { .. }));
    }

    #[test]
    fn prefill_scratch_charges_gptoss_by_max_seq_and_qwen_moe_by_pf_list_only() {
        let cfg = r#"{"model_type":"gpt_oss","num_attention_heads":64,"quantization_config":{"quant_method":"mxfp4"}}"#;
        let oss = fixture_dir("gpt-oss-scratch", Some(cfg));
        let m = nv_models::gpt_oss_wgpu::prefill_m();
        assert!(m > 0, "default prefill_m must be non-zero for this test to mean anything");

        let at = |max_seq: usize| prefill_scratch_gib(&oss, max_seq);
        let one_k = at(1024);
        let expect = (m * 64 * 1024 * 4) as f64 / (1u64 << 30) as f64;
        assert!(
            (one_k - expect).abs() < 1e-12,
            "scratch {one_k} != m*heads*max_seq*4 {expect}"
        );
        assert!(one_k > 0.0, "the term must actually fire, not silently yield zero");
        assert!(
            (at(2048) - 2.0 * one_k).abs() < 1e-12,
            "scratch must scale linearly with max_seq"
        );
        assert_eq!(at(0), 0.0, "no max_seq known -> charge nothing rather than guess");

        let qwen = fixture_dir(
            "qwen-moe-scratch",
            Some(r#"{"model_type":"qwen3_5_moe","num_attention_heads":64}"#),
        );
        let qm = nv_models::qwen3_5_moe_wgpu::prefill_m();
        assert!(
            qm > 0,
            "default qwen prefill_m must be non-zero for this test to mean anything"
        );
        let q_expect = (qm
            * nv_models::qwen3_5_moe_wgpu::prefill_list_bytes_per_token_charged_by_mem_fit())
            as f64
            / (1u64 << 30) as f64;
        let q_one_k = prefill_scratch_gib(&qwen, 1024);
        assert!(
            (q_one_k - q_expect).abs() < 1e-15 && q_one_k > 0.0,
            "qwen moe scratch {q_one_k} != m*list_bytes_per_token {q_expect}; the pf list is \
             small but resident from build, so it must be charged, not rounded to zero"
        );
        assert_eq!(
            prefill_scratch_gib(&qwen, 2048),
            q_one_k,
            "the qwen pf list holds per-chunk params only, so it must not scale with max_seq"
        );
        let laguna = fixture_dir("laguna-scratch", Some(r#"{"model_type":"laguna"}"#));
        let lm = nv_models::laguna_wgpu::prefill_m();
        assert!(
            lm > 0,
            "default laguna prefill_m must be non-zero for this test to mean anything"
        );
        let l_expect = (lm
            * nv_models::laguna_wgpu::prefill_list_bytes_per_token_charged_by_mem_fit())
            as f64
            / (1u64 << 30) as f64;
        let l_one_k = prefill_scratch_gib(&laguna, 1024);
        assert!(
            (l_one_k - l_expect).abs() < 1e-15 && l_one_k > 0.0,
            "laguna scratch {l_one_k} != m*list_bytes_per_token {l_expect}; the pf list is small \
             but resident from build, so it must be charged, not rounded to zero"
        );
        assert_eq!(
            prefill_scratch_gib(&laguna, 2048),
            l_one_k,
            "the laguna pf list holds per-chunk StepUniform records only, so it must not scale \
             with max_seq"
        );
    }

    #[test]
    fn load_path_native_for_packed_qwen_moe_and_gpt_oss() {
        let qwen = fixture_dir(
            "qwen-moe",
            Some(
                r#"{"model_type":"qwen3_5_moe","quantization_config":{"quant_method":"compressed-tensors","format":"nvfp4-pack-quantized"}}"#,
            ),
        );
        assert_eq!(load_path(&qwen), LoadPath::NativePacked);
        let oss = fixture_dir(
            "gpt-oss",
            Some(r#"{"model_type":"gpt_oss","quantization_config":{"quant_method":"mxfp4"}}"#),
        );
        assert_eq!(load_path(&oss), LoadPath::NativePacked);
    }

    #[test]
    fn load_path_conservative_for_unknown_and_unreadable() {
        let unknown = fixture_dir(
            "unknown-arch",
            Some(r#"{"model_type":"mystery","quantization_config":{"quant_method":"awq"}}"#),
        );
        assert_eq!(load_path(&unknown), LoadPath::HostDequant);
        let missing = fixture_dir("no-config", None);
        assert_eq!(load_path(&missing), LoadPath::HostDequant);
    }

    #[test]
    fn load_path_plain_for_unquantized_checkpoint() {
        let plain = fixture_dir("plain-bf16", Some(r#"{"model_type":"qwen3_5_moe"}"#));
        assert_eq!(load_path(&plain), LoadPath::PlainWeights);
    }
}
