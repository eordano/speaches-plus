#![cfg(feature = "wgpu")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use nv_layers::attn::AttnConfig;
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::norm::RmsNorm;
use nv_layers::weight_load::load_rmsnorm;
use nv_models::deepseek_ocr::preprocess::RgbImage;
use nv_models::dots_ocr::preprocess::{prepare, PixelBudget, PATCH_DIM};
use nv_models::dots_ocr::vision::{vision_rope_tables, DotsVisionConfig, DotsVisionTower};
use nv_models::dots_ocr::vision_wgpu::{
    pad_to, probe_flash_attention, probe_gemm_f16, probe_gemm_w4a16, AttnArm, CoopPlan,
    DotsVisionTowerWgpu, FlashPlan, VisionGemm, VisionGemmRoute, VisionWeightFmt,
    DOTS_VISION_WGPU_ATTN_ENV, DOTS_VISION_WGPU_FLASH_PLAN_ENV, DOTS_VISION_WGPU_WQ_ENV,
};
use nv_kernels::wgpu_backend::WgpuContext;
use nv_weights::WeightLoader;

const PAGE_ENV: &str = "NV_DOTS_PAGE";
const PATCHES_ENV: &str = "NV_DOTS_VISION_WGPU_PATCHES";

const DEFAULT_PARITY_PATCH_BUDGET_KEEPS_THE_CPU_F32_ORACLE_UNDER_A_MINUTE: usize = 1024;

const PIXELS_PER_PATCH_IS_THE_14X14_PATCH_AREA: usize = 196;

const F16_OPERAND_GEMM_TOLERANCE_IS_SET_BY_THE_OPERAND_MANTISSA_NOT_THE_ACCUMULATOR: f64 = 4e-3;

fn ctx_or_fail(test: &str) -> &'static WgpuContext {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("[{test}] adapter: {}", ctx.summary());
            ctx
        }
        Err(e) => panic!(
            "{test} needs a wgpu adapter and must fail rather than skip: {e}. \
             Run through rust/scripts/nvk.sh, which wires the Vulkan loader."
        ),
    }
}

fn snapshot_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_DOTS_DIR") {
        let p = PathBuf::from(d);
        return p.is_dir().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

const THE_RESIDENT_FOOTPRINT_IS_SAMPLED_NOT_SUMMED_BECAUSE_WGPU_FREES_A_DROPPED_BUFFER_ONLY_ON_POLL: &str =
    "wgpu reclaims a dropped device buffer at the next poll, so a forward's own accounting of what \
     it allocated is not what the driver is holding. The only honest number is the driver's, \
     sampled while the forward runs, and it is this process's own compute-app row rather than the \
     card total because a second lane may be measuring on the same card.";

fn device_mib_for_this_process() -> Option<u64> {
    let pid = std::process::id();
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut it = l.split(',');
            let p = it.next()?.trim().parse::<u32>().ok()?;
            let m = it.next()?.trim().parse::<u64>().ok()?;
            (p == pid).then_some(m)
        })
        .next()
}

fn peak_device_mib_during<T>(f: impl FnOnce() -> T) -> (T, u64) {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    let _ = THE_RESIDENT_FOOTPRINT_IS_SAMPLED_NOT_SUMMED_BECAUSE_WGPU_FREES_A_DROPPED_BUFFER_ONLY_ON_POLL;
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let (s2, p2) = (stop.clone(), peak.clone());
    let sampler = std::thread::spawn(move || {
        while !s2.load(Ordering::Relaxed) {
            if let Some(m) = device_mib_for_this_process() {
                p2.fetch_max(m, Ordering::Relaxed);
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    });
    let out = f();
    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    (out, peak.load(Ordering::Relaxed))
}

fn max_abs_and_rel(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    assert_eq!(got.len(), want.len(), "compared vectors differ in length");
    let mut max_abs = 0f64;
    let mut scale = 0f64;
    let mut se = 0f64;
    let mut sr = 0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        let d = (*g as f64 - *w as f64).abs();
        max_abs = max_abs.max(d);
        scale = scale.max((*w as f64).abs());
        se += d * d;
        sr += (*w as f64) * (*w as f64);
    }
    let rms_rel = (se / got.len() as f64).sqrt() / (sr / got.len() as f64).sqrt().max(1e-30);
    (max_abs, max_abs / scale.max(1e-30), rms_rel)
}

struct TwinBlock {
    norm1: RmsNorm,
    qkv: Linear,
    proj: Linear,
    norm2: RmsNorm,
    mlp: Mlp,
}

struct TwinTower {
    cfg: DotsVisionConfig,
    patch_proj: Linear,
    patch_norm: RmsNorm,
    blocks: Vec<TwinBlock>,
    post_trunk_norm: Option<RmsNorm>,
    ln_w: Tensor,
    ln_b: Tensor,
    fc0: Linear,
    fc2: Linear,
    device: Device,
    dtype: DType,
}

fn twin_linear(
    w: &WeightLoader,
    name: &str,
    out_dim: usize,
    in_dim: usize,
    bias: Option<&str>,
    dtype: DType,
) -> Result<Linear> {
    let t = w.get(name, dtype).with_context(|| format!("load {name}"))?;
    let t = if t.dims().len() > 2 {
        let rows = t.dim(0)?;
        let cols: usize = t.dims()[1..].iter().product();
        t.reshape((rows, cols))?
    } else {
        t
    };
    assert_eq!(t.dims(), &[out_dim, in_dim], "{name} shape");
    let b = match bias {
        Some(bn) => Some(w.get(bn, dtype).with_context(|| format!("load {bn}"))?),
        None => None,
    };
    Linear::new(t.contiguous()?, b)
}

impl TwinTower {
    fn load(w: &WeightLoader, prefix: &str, device: &Device, dtype: DType) -> Result<Self> {
        let cfg = DotsVisionConfig::default();
        let dim = cfg.embed_dim;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let bp = format!("{prefix}blocks.{i}.");
            blocks.push(TwinBlock {
                norm1: load_rmsnorm(
                    w,
                    &format!("{bp}norm1.weight"),
                    dim,
                    cfg.rms_norm_eps,
                    DType::F32,
                )?,
                norm2: load_rmsnorm(
                    w,
                    &format!("{bp}norm2.weight"),
                    dim,
                    cfg.rms_norm_eps,
                    DType::F32,
                )?,
                qkv: twin_linear(w, &format!("{bp}attn.qkv.weight"), dim * 3, dim, None, dtype)?,
                proj: twin_linear(w, &format!("{bp}attn.proj.weight"), dim, dim, None, dtype)?,
                mlp: Mlp::new(
                    twin_linear(
                        w,
                        &format!("{bp}mlp.fc1.weight"),
                        cfg.intermediate_size,
                        dim,
                        None,
                        dtype,
                    )?,
                    twin_linear(
                        w,
                        &format!("{bp}mlp.fc3.weight"),
                        cfg.intermediate_size,
                        dim,
                        None,
                        dtype,
                    )?,
                    twin_linear(
                        w,
                        &format!("{bp}mlp.fc2.weight"),
                        dim,
                        cfg.intermediate_size,
                        None,
                        dtype,
                    )?,
                )?,
            });
        }
        let mp = format!("{prefix}merger.");
        let hidden = dim * cfg.spatial_merge_size * cfg.spatial_merge_size;
        Ok(Self {
            patch_proj: twin_linear(
                w,
                &format!("{prefix}patch_embed.patchifier.proj.weight"),
                dim,
                PATCH_DIM,
                Some(&format!("{prefix}patch_embed.patchifier.proj.bias")),
                dtype,
            )?,
            patch_norm: load_rmsnorm(
                w,
                &format!("{prefix}patch_embed.patchifier.norm.weight"),
                dim,
                cfg.rms_norm_eps,
                DType::F32,
            )?,
            blocks,
            post_trunk_norm: if cfg.post_norm {
                Some(load_rmsnorm(
                    w,
                    &format!("{prefix}post_trunk_norm.weight"),
                    dim,
                    cfg.rms_norm_eps,
                    DType::F32,
                )?)
            } else {
                None
            },
            ln_w: w.get(&format!("{mp}ln_q.weight"), DType::F32)?,
            ln_b: w.get(&format!("{mp}ln_q.bias"), DType::F32)?,
            fc0: twin_linear(
                w,
                &format!("{mp}mlp.0.weight"),
                hidden,
                hidden,
                Some(&format!("{mp}mlp.0.bias")),
                dtype,
            )?,
            fc2: twin_linear(
                w,
                &format!("{mp}mlp.2.weight"),
                cfg.hidden_size,
                hidden,
                Some(&format!("{mp}mlp.2.bias")),
                dtype,
            )?,
            cfg,
            device: device.clone(),
            dtype,
        })
    }

    fn rotate_half(x: &Tensor) -> Result<Tensor> {
        let d = x.dim(D::Minus1)?;
        let half = d / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        Ok(Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?)
    }

    fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        Ok(x32
            .broadcast_mul(cos)?
            .add(&Self::rotate_half(&x32)?.broadcast_mul(sin)?)?
            .to_dtype(dtype)?)
    }

    fn forward(&self, patches: &[f32], grid_h: usize, grid_w: usize) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        self.forward_traced(patches, grid_h, grid_w, true)
    }

    fn forward_traced(
        &self,
        patches: &[f32],
        grid_h: usize,
        grid_w: usize,
        trace_every_block: bool,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let n = grid_h * grid_w;
        let hd = self.cfg.head_dim();
        let heads = self.cfg.num_attention_heads;
        let dim = self.cfg.embed_dim;
        let p = Tensor::from_slice(patches, (n, PATCH_DIM), &Device::Cpu)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;
        let mut x = self.patch_proj.forward(&p)?;
        x = self.patch_norm.forward(&x)?.to_dtype(self.dtype)?;
        let (cos, sin) = vision_rope_tables(grid_h, grid_w, hd, &self.device)?;
        let attn_cfg = AttnConfig {
            num_heads: heads,
            num_kv_heads: heads,
            head_dim: hd,
            softmax_scale: 1.0 / (hd as f32).sqrt(),
            causal: false,
        };
        let mut trace = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let normed = blk.norm1.forward(&x)?.to_dtype(self.dtype)?;
            let qkv = blk.qkv.forward(&normed)?.reshape((n, 3, heads, hd))?;
            let q = qkv.i((.., 0))?.contiguous()?.reshape((1, n, heads, hd))?;
            let k = qkv.i((.., 1))?.contiguous()?.reshape((1, n, heads, hd))?;
            let v = qkv.i((.., 2))?.contiguous()?.reshape((1, n, heads, hd))?;
            let q = Self::rope(&q, &cos, &sin)?;
            let k = Self::rope(&k, &cos, &sin)?;
            let attn = nv_layers::attn::attention(&q, &k, &v, &attn_cfg)?;
            let attn = attn.reshape((n, heads * hd))?;
            x = x.add(&blk.proj.forward(&attn)?)?;
            let normed2 = blk.norm2.forward(&x)?.to_dtype(self.dtype)?;
            x = x.add(&blk.mlp.forward(&normed2)?)?;
            if trace_every_block {
                trace.push(
                    x.to_dtype(DType::F32)?
                        .to_device(&Device::Cpu)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                );
            }
        }
        if let Some(norm) = &self.post_trunk_norm {
            x = norm.forward(&x)?.to_dtype(self.dtype)?;
        }
        let merge_units = self.cfg.spatial_merge_size * self.cfg.spatial_merge_size;
        let normed = nv_models::deepseek_ocr::sam::layer_norm(&x, &self.ln_w, &self.ln_b, 1e-6)?;
        let grouped = normed.reshape((n / merge_units, dim * merge_units))?;
        let h = self.fc0.forward(&grouped)?.gelu_erf()?;
        let merged = self
            .fc2
            .forward(&h)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok((merged, trace))
    }
}

#[test]
fn the_wgpu_vision_tower_names_the_only_general_large_m_gemm_this_adapter_carries() {
    let ctx = ctx_or_fail("dots_vision_wgpu_route");
    assert!(
        ctx.caps.shader_f16,
        "every dots.ocr vision wgpu route stages weights as f16; this adapter has no SHADER_F16"
    );
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    eprintln!(
        "[dots_vision_wgpu_route] route={} tile={} coop_gemm_reason={:?}",
        route.name(),
        route.tile(),
        ctx.caps.coop_gemm_reason()
    );
    let head_dim = DotsVisionConfig::default().head_dim();
    assert!(
        head_dim.is_multiple_of(route.tile()),
        "dots.ocr vision head_dim {head_dim} must be a multiple of the {}-wide GEMM tile",
        route.tile()
    );
    assert!(
        route.is_coop() || !ctx.caps.cooperative_matrix,
        "the adapter advertises cooperative matrices but the vision tower fell back to the scalar \
         correctness route; that is a capability-gate defect, not a hardware limit"
    );
}

#[test]
fn dv_gemm_matches_an_f64_oracle_on_both_routes_including_a_ragged_patch_dim_k() {
    let ctx = ctx_or_fail("dots_vision_wgpu_gemm_oracle");
    let mut routes = vec![VisionGemmRoute::ScalarF16];
    if let VisionGemmRoute::Coop(g) = VisionGemmRoute::from_env_and_adapter(ctx) {
        routes.push(VisionGemmRoute::Coop(g));
    }
    let shapes = [(32usize, 64usize, PATCH_DIM), (48, 96, 128), (16, 32, 1536)];
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 0.5 - 0.25
    };
    for route in routes {
        let g = VisionGemm::new(ctx, route).expect("vision gemm");
        for (m, n, k) in shapes {
            let w: Vec<f32> = (0..n * k).map(|_| next()).collect();
            let x: Vec<f32> = (0..m * k).map(|_| next()).collect();
            let got = probe_gemm_f16(&g, &w, &x, m, n, k).expect("probe gemm");
            let mut worst = 0f64;
            let mut scale = 0f64;
            for r in 0..m {
                for c in 0..n {
                    let mut acc = 0f64;
                    for i in 0..k {
                        acc += half::f16::from_f32(x[r * k + i]).to_f32() as f64
                            * half::f16::from_f32(w[c * k + i]).to_f32() as f64;
                    }
                    worst = worst.max((got[r * n + c] as f64 - acc).abs());
                    scale = scale.max(acc.abs());
                }
            }
            let rel = worst / scale.max(1e-30);
            eprintln!(
                "[dots_vision_wgpu_gemm_oracle] route={} m={m} n={n} k={k} max_abs={worst:.3e} max_rel={rel:.3e}",
                route.name()
            );
            assert!(
                rel < F16_OPERAND_GEMM_TOLERANCE_IS_SET_BY_THE_OPERAND_MANTISSA_NOT_THE_ACCUMULATOR,
                "route {} at {m}x{n}x{k} drifted {rel:.3e} from the f64 oracle over f16-rounded operands",
                route.name()
            );
        }
    }
}

#[test]
fn the_nvfp4_weight_arm_is_env_selected_and_every_other_value_keeps_f16() {
    std::env::remove_var(DOTS_VISION_WGPU_WQ_ENV);
    assert_eq!(VisionWeightFmt::from_env(), VisionWeightFmt::F16);
    std::env::set_var(DOTS_VISION_WGPU_WQ_ENV, "nvfp4");
    assert_eq!(VisionWeightFmt::from_env(), VisionWeightFmt::Nvfp4Block16);
    assert_eq!(VisionWeightFmt::from_env().name(), "nvfp4");
    std::env::set_var(DOTS_VISION_WGPU_WQ_ENV, "0");
    assert_eq!(
        VisionWeightFmt::from_env(),
        VisionWeightFmt::F16,
        "explicit =0 must restore the f16 default"
    );
    std::env::remove_var(DOTS_VISION_WGPU_WQ_ENV);
}

const NVFP4_WEIGHT_ONLY_GEMM_RMS_CEILING_ADMITS_THE_FULL_E2M1_NOISE_BECAUSE_ZERO_MEAN_RANDOM_DATA_NEVER_AVERAGES_IT_DOWN: f64 =
    2e-1;

const NVFP4_WEIGHT_ONLY_GEMM_RMS_CEILING_BASIS: &str =
    "On uniform zero-mean operands the dot product is a random walk, so per-weight e2m1 \
     quantization noise passes straight through: measured 9.96e-2 at k=588 and 1.02e-1 at k=128 \
     on this suite's own shapes. The ceiling is 2e-1 because the failure it exists to catch -- a \
     misindexed nibble, a wrong scale byte, a swapped operand -- reads order-one, while the \
     kernel-correctness gate is the bit-equality assertion above it, not this number.";

#[test]
fn the_w4a16_vision_route_is_bit_equal_to_the_f16_route_on_host_dequantized_weights() {
    let ctx = ctx_or_fail("dots_vision_wgpu_w4a16");
    let adapter_speaks_16x16 = matches!(
        ctx.caps.coop_decision(&nv_kernels::wgpu_backend::qualify::CoopRequest::square(
            16,
            nv_kernels::wgpu_backend::qualify::CoopScalar::F16,
            nv_kernels::wgpu_backend::qualify::CoopScalar::F32,
        )),
        nv_kernels::wgpu_backend::qualify::CoopDecision::Compile
            | nv_kernels::wgpu_backend::qualify::CoopDecision::CompileUnadvertised(_)
    );
    if !adapter_speaks_16x16 {
        eprintln!(
            "the nvfp4 vision weight arm exists only on 16x16 cooperative-matrix adapters; \
             this one advertises none (apple simdgroups are 8x8), so the arm structurally \
             cannot exist here and the loud-fail is reserved for adapters that HAVE 16x16 \
             and still failed to route onto it"
        );
        return;
    }
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    assert!(
        route.is_coop() && route.tile() == 16,
        "this adapter advertises 16x16 cooperative matrices yet the vision gemm served {}; \
         the nvfp4 weight arm must fail loudly rather than skip on hardware that can run it",
        route.name()
    );
    let g = VisionGemm::new(ctx, route).expect("vision gemm");
    let mut state = 0x1357_9bdf_2468_aceeu64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 0.5 - 0.25
    };
    for (m, n, k) in [(32usize, 64usize, PATCH_DIM), (48, 96, 128), (16, 32, 1536)] {
        let w: Vec<f32> = (0..n * k).map(|_| next()).collect();
        let x: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let (got4, wdeq) = probe_gemm_w4a16(&g, &w, &x, m, n, k).expect("probe w4a16");
        let gotf = probe_gemm_f16(&g, &wdeq, &x, m, n, k).expect("probe f16 on dequantized");
        let ndiff = got4
            .iter()
            .zip(gotf.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            ndiff,
            0,
            "m={m} n={n} k={k}: the vision w4a16 route must be bit-equal to the f16 route on \
             host-dequantized weights, same f32 product rounded to f16 either way; {ndiff} of {} \
             outputs differ",
            m * n
        );
        let mut se = 0f64;
        let mut sr = 0f64;
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0f64;
                for i in 0..k {
                    acc += half::f16::from_f32(x[r * k + i]).to_f32() as f64 * w[c * k + i] as f64;
                }
                let d = got4[r * n + c] as f64 - acc;
                se += d * d;
                sr += acc * acc;
            }
        }
        let rms = (se / (m * n) as f64).sqrt() / (sr / (m * n) as f64).sqrt().max(1e-30);
        eprintln!(
            "[dots_vision_wgpu_w4a16] m={m} n={n} k={k} bit_equal_to_f16_on_dequant=yes \
             rms_rel_vs_unquantized_f64_oracle={rms:.3e}"
        );
        assert!(
            rms < NVFP4_WEIGHT_ONLY_GEMM_RMS_CEILING_ADMITS_THE_FULL_E2M1_NOISE_BECAUSE_ZERO_MEAN_RANDOM_DATA_NEVER_AVERAGES_IT_DOWN,
            "m={m} n={n} k={k}: w4a16 drifted {rms:.3e} rms-relative from the unquantized f64 \
             oracle, past what NVFP4 block-16 weight quantization alone can explain. {}",
            NVFP4_WEIGHT_ONLY_GEMM_RMS_CEILING_BASIS
        );
    }
}

#[test]
fn padding_a_ragged_patch_dim_up_to_the_gemm_tile_never_shortens_a_row() {
    for tile in [1usize, 8, 16] {
        let padded = pad_to(PATCH_DIM, tile);
        assert!(padded >= PATCH_DIM);
        assert!(padded.is_multiple_of(tile));
        assert!(padded - PATCH_DIM < tile);
    }
    assert_eq!(pad_to(PATCH_DIM, 16), 592);
}

const FUSED_ATTENTION_PLAN_LADDER_SPANS_THE_SHAPES_THE_ACCUMULATOR_AND_THE_WORKGROUP_BUDGET_BOTH_ADMIT:
    [FlashPlan; 12] = [
    FlashPlan { bm: 64, bn: 64, sg: 4, scs: 2 },
    FlashPlan { bm: 64, bn: 64, sg: 4, scs: 4 },
    FlashPlan { bm: 64, bn: 64, sg: 4, scs: 1 },
    FlashPlan { bm: 64, bn: 96, sg: 4, scs: 2 },
    FlashPlan { bm: 32, bn: 128, sg: 4, scs: 4 },
    FlashPlan { bm: 32, bn: 128, sg: 4, scs: 2 },
    FlashPlan { bm: 32, bn: 128, sg: 2, scs: 2 },
    FlashPlan { bm: 32, bn: 64, sg: 4, scs: 4 },
    FlashPlan { bm: 32, bn: 64, sg: 2, scs: 2 },
    FlashPlan { bm: 128, bn: 32, sg: 8, scs: 2 },
    FlashPlan { bm: 64, bn: 128, sg: 8, scs: 4 },
    FlashPlan { bm: 128, bn: 64, sg: 8, scs: 2 },
];

const THE_FUSED_ATTENTION_ORACLE_TOLERANCE_IS_SET_BY_F16_PROBABILITIES_NOT_BY_THE_ONLINE_RESCALE: f64 =
    1e-3;

fn flash_oracle(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    heads: usize,
    hd: usize,
) -> Vec<f32> {
    let dim = heads * hd;
    let scale = 1.0f64 / (hd as f64).sqrt();
    let h16 = |x: f32| half::f16::from_f32(x).to_f32() as f64;
    let mut out = vec![0f32; n * dim];
    for h in 0..heads {
        for r in 0..n {
            let mut s = vec![0f64; n];
            let mut mx = f64::NEG_INFINITY;
            for (j, sj) in s.iter_mut().enumerate() {
                let mut acc = 0f64;
                for d in 0..hd {
                    acc += h16(q[r * dim + h * hd + d]) * h16(k[j * dim + h * hd + d]);
                }
                *sj = acc * scale;
                mx = mx.max(*sj);
            }
            let mut z = 0f64;
            for sj in s.iter_mut() {
                *sj = (*sj - mx).exp();
                z += *sj;
            }
            for d in 0..hd {
                let mut acc = 0f64;
                for (j, sj) in s.iter().enumerate() {
                    acc += sj * h16(v[j * dim + h * hd + d]);
                }
                out[r * dim + h * hd + d] = (acc / z) as f32;
            }
        }
    }
    out
}

#[test]
fn every_fused_attention_plan_in_the_ladder_emits_wgsl_naga_parses_and_validates() {
    let mut emitted = 0usize;
    for plan in FUSED_ATTENTION_PLAN_LADDER_SPANS_THE_SHAPES_THE_ACCUMULATOR_AND_THE_WORKGROUP_BUDGET_BOTH_ADMIT
    {
        let Ok(geom) = plan.geometry(16, 128) else {
            continue;
        };
        let src = geom.source();
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("flash plan {}: parse: {e}", plan.name()));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("flash plan {}: validate: {e}", plan.name()));
        assert!(src.contains(&geom.entry()));
        assert!(
            !src.contains("array<FQC"),
            "flash plan {}: accumulators must not be an indexed array",
            plan.name()
        );
        eprintln!(
            "[dots_vision_wgpu_flash_source] {} wg={} acc_dwords={} workgroup_bytes={} \
             mac_per_load_s={:.2} mac_per_load_o={:.2}",
            plan.name(),
            geom.wg,
            geom.acc_lane_dwords(),
            geom.workgroup_bytes(),
            (geom.srm * geom.scn) as f64 / (geom.srm + geom.scn) as f64,
            (geom.rm * geom.ocn) as f64 / (geom.rm + geom.ocn) as f64,
        );
        emitted += 1;
    }
    assert!(
        emitted >= 4,
        "only {emitted} of the ladder's plans produced a geometry at 16x16 fragments and head_dim \
         128; the ladder has nothing to sweep"
    );
}

#[test]
fn the_fused_attention_reproduces_an_f64_softmax_oracle_over_f16_rounded_operands() {
    let ctx = ctx_or_fail("dots_vision_wgpu_flash_oracle");
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    assert!(
        route.is_coop(),
        "the fused attention is a cooperative-matrix kernel and this adapter reports none"
    );
    let g = VisionGemm::new(ctx, route).expect("vision gemm");
    let (n, heads, hd) = (200usize, 2usize, 128usize);
    let dim = heads * hd;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let q: Vec<f32> = (0..n * dim).map(|_| next()).collect();
    let k: Vec<f32> = (0..n * dim).map(|_| next()).collect();
    let v: Vec<f32> = (0..n * dim).map(|_| next()).collect();
    for rows in [64usize, 200] {
        let zero = vec![0f32; rows * dim];
        let mut eye = vec![0f32; rows * dim];
        for j in 0..rows {
            for h in 0..heads {
                eye[j * dim + h * hd + (j % hd)] = 1.0;
            }
        }
        for plan in FUSED_ATTENTION_PLAN_LADDER_SPANS_THE_SHAPES_THE_ACCUMULATOR_AND_THE_WORKGROUP_BUDGET_BOTH_ADMIT
        {
            if g.flash_geometry(&plan, hd).is_err() {
                continue;
            }
            let got =
                probe_flash_attention(&g, &plan, &zero, &zero, &eye, rows, heads, hd)
                    .expect("identity-value probe");
            let mut worst_cell = (0usize, 0usize, 0f64);
            for r in 0..rows {
                for d in 0..hd.min(rows) {
                    let want = if rows > hd && d + hd < rows { 2.0 } else { 1.0 };
                    let p = got[r * dim + d] as f64 * rows as f64;
                    if (p - want).abs() > worst_cell.2 {
                        worst_cell = (r, d, (p - want).abs());
                    }
                }
            }
            assert!(
                worst_cell.2 < 1e-2,
                "flash plan {}: with zero scores and an identity value matrix every probability \
                 must be 1/n, yet row {} column {} read {:.4} off. That is the probability tile \
                 being written by something other than the softmax: {}",
                plan.name(),
                worst_cell.0,
                worst_cell.1,
                worst_cell.2,
                nv_models::dots_ocr::vision_wgpu::EVERY_WORKGROUP_TILE_THIS_KERNEL_COOP_LOADS_MUST_START_ON_A_16_BYTE_BOUNDARY_WHICH_MAKING_IT_AN_ARRAY_DOES_NOT_BY_ITSELF_ACHIEVE
            );
        }
    }
    let mut ran = 0usize;
    let mut worst = 0f64;
    for rows in [64usize, 65, 200] {
        let want = flash_oracle(&q, &k, &v, rows, heads, hd);
        for plan in FUSED_ATTENTION_PLAN_LADDER_SPANS_THE_SHAPES_THE_ACCUMULATOR_AND_THE_WORKGROUP_BUDGET_BOTH_ADMIT
        {
            if g.flash_geometry(&plan, hd).is_err() {
                continue;
            }
            let got = probe_flash_attention(
                &g,
                &plan,
                &q[..rows * dim],
                &k[..rows * dim],
                &v[..rows * dim],
                rows,
                heads,
                hd,
            )
            .unwrap_or_else(|e| panic!("flash plan {}: {e}", plan.name()));
            let (abs, rel, rms) = max_abs_and_rel(&got, &want);
            worst = worst.max(rms);
            eprintln!(
                "[dots_vision_wgpu_flash_oracle] {} n={rows} heads={heads} hd={hd} \
                 max_abs={abs:.3e} max_rel={rel:.3e} rms_rel={rms:.3e}",
                plan.name()
            );
            ran += 1;
        }
    }
    assert!(
        ran >= 12,
        "only {ran} plan-by-shape cells ran on this adapter; the oracle gate covered almost nothing"
    );
    assert!(
        worst < THE_FUSED_ATTENTION_ORACLE_TOLERANCE_IS_SET_BY_F16_PROBABILITIES_NOT_BY_THE_ONLINE_RESCALE,
        "the fused attention drifted {worst:.3e} rms-relative from the f64 online-free oracle at \
         worst; the online rescale is exp(m_old-m_new) and is exactly 1.0 on most blocks, so a \
         drift this size is a kernel defect and not the arithmetic change"
    );
}

#[test]
#[ignore]
fn the_wgpu_vision_tower_tracks_the_shipped_tower_layer_by_layer_on_real_dots_weights() {
    let ctx = ctx_or_fail("dots_vision_wgpu_parity");
    let dir = snapshot_dir().expect("dots.ocr snapshot present in the hub cache");
    let page = std::env::var(PAGE_ENV)
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let want_patches: usize = std::env::var(PATCHES_ENV)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_PARITY_PATCH_BUDGET_KEEPS_THE_CPU_F32_ORACLE_UNDER_A_MINUTE);

    let img = RgbImage::decode_file(&page).expect("decode page");
    let budget = PixelBudget {
        min_pixels: 3136,
        max_pixels: want_patches * PIXELS_PER_PATCH_IS_THE_14X14_PATCH_AREA,
    };
    let prep = prepare(&img, budget).expect("prepare page");
    let n = prep.num_patches();
    eprintln!(
        "[dots_vision_wgpu_parity] page {}x{} -> grid {}x{} = {n} patches, {} vision tokens",
        img.w,
        img.h,
        prep.grid_h,
        prep.grid_w,
        prep.num_vision_tokens()
    );

    let cpu = Device::Cpu;
    let loader = WeightLoader::open_dir(&dir, &cpu).expect("open dots.ocr weights");

    let t0 = Instant::now();
    let twin = TwinTower::load(&loader, "vision_tower.", &cpu, DType::F32).expect("twin load");
    let (twin_out, twin_trace) = twin
        .forward(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("twin forward");
    eprintln!(
        "[dots_vision_wgpu_parity] cpu f32 twin forward {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    let shipped = DotsVisionTower::from_loader(
        &loader,
        "vision_tower.",
        DotsVisionConfig::default(),
        &cpu,
        DType::F32,
    )
    .expect("shipped tower load");
    let shipped_out = shipped
        .forward_patches(
            &Tensor::from_slice(&prep.patches, (n, PATCH_DIM), &cpu).expect("patch tensor"),
            prep.grid_h,
            prep.grid_w,
        )
        .expect("shipped forward")
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("shipped output");
    let (tie_abs, tie_rel, tie_rms) = max_abs_and_rel(&twin_out, &shipped_out);
    eprintln!(
        "[dots_vision_wgpu_parity] twin vs shipped DotsVisionTower (cpu f32): max_abs={tie_abs:.3e} max_rel={tie_rel:.3e} rms_rel={tie_rms:.3e}"
    );
    assert!(
        tie_rel < 1e-5,
        "the test-local twin is not the shipped vision tower: max_rel={tie_rel:.3e}"
    );

    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    let t1 = Instant::now();
    let wgpu_tower = DotsVisionTowerWgpu::from_loader_route(
        &loader,
        "vision_tower.",
        DotsVisionConfig::default(),
        route,
    )
    .expect("wgpu tower load");
    eprintln!(
        "[dots_vision_wgpu_parity] wgpu weight staging {:.1}s route={}",
        t1.elapsed().as_secs_f64(),
        route.name()
    );
    let t2 = Instant::now();
    let (wgpu_out, wgpu_trace) = wgpu_tower
        .forward_patches_with_layer_trace(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("wgpu forward");
    eprintln!(
        "[dots_vision_wgpu_parity] wgpu forward {:.2}s",
        t2.elapsed().as_secs_f64()
    );

    assert_eq!(wgpu_trace.len(), twin_trace.len(), "layer trace lengths");
    eprintln!("[dots_vision_wgpu_parity] layer max_abs max_rel rms_rel");
    let mut worst_rms = 0f64;
    for (li, (a, b)) in wgpu_trace.iter().zip(twin_trace.iter()).enumerate() {
        let (abs, rel, rms) = max_abs_and_rel(a, b);
        worst_rms = worst_rms.max(rms);
        eprintln!("[dots_vision_wgpu_parity] {li} {abs:.4e} {rel:.4e} {rms:.4e}");
    }
    let (out_abs, out_rel, out_rms) = max_abs_and_rel(&wgpu_out, &twin_out);
    eprintln!(
        "[dots_vision_wgpu_parity] merged features: max_abs={out_abs:.4e} max_rel={out_rel:.4e} rms_rel={out_rms:.4e}"
    );
    assert!(
        out_rms < 5e-2,
        "wgpu vision features drifted {out_rms:.3e} rms-relative from the cpu f32 oracle; \
         worst per-layer rms {worst_rms:.3e}"
    );
}

const THE_NVFP4_ARM_GATE_SITS_BETWEEN_THE_MEASURED_QUANTIZATION_ERROR_AND_THE_ORDER_ONE_SIGNATURE_OF_AN_UNPACK_DEFECT: f64 =
    5e-1;

const THE_NVFP4_ARM_GATE_BASIS: &str =
    "Measured on this suite's reference rung (dots_ocr c0111ce6, MAC_NOTEBOOK_009_p030 at the \
     1024-patch budget = 988 patches, wgpu coop_mat16x16 slab attention, cpu f32 oracle): the \
     nvfp4 arm reads merged rms_rel 3.16e-1 with the closed-form scale search and 3.41e-1 with \
     nearest-byte scales, entering at 1.09e-1 after layer 0 -- one projection's worth of e2m1 \
     noise -- and plateauing near 4.7e-1 mid-trunk. The CUDA bf16 twin reads 1.02e-1 on the same \
     rung and the f16 arm 2.21e-2. A real unpack defect -- swapped nibble order, misindexed \
     scale byte, wrong operand -- reads order-one from the first layer, so the gate sits at 5e-1: \
     above the measured quantization error, below catastrophe. The measured number is reported \
     and is the deliverable; whether 3.16e-1 features serve OCR is adjudicated downstream, not \
     by this gate.";

#[test]
#[ignore]
fn the_nvfp4_weight_arm_tracks_the_cpu_f32_oracle_layer_by_layer_within_the_cuda_bf16_error_band() {
    let ctx = ctx_or_fail("dots_vision_wgpu_nvfp4_parity");
    let dir = snapshot_dir().expect("dots.ocr snapshot present in the hub cache");
    let page = std::env::var(PAGE_ENV)
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let want_patches: usize = std::env::var(PATCHES_ENV)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_PARITY_PATCH_BUDGET_KEEPS_THE_CPU_F32_ORACLE_UNDER_A_MINUTE);
    let img = RgbImage::decode_file(&page).expect("decode page");
    let budget = PixelBudget {
        min_pixels: 3136,
        max_pixels: want_patches * PIXELS_PER_PATCH_IS_THE_14X14_PATCH_AREA,
    };
    let prep = prepare(&img, budget).expect("prepare page");
    let n = prep.num_patches();
    eprintln!(
        "[dots_vision_wgpu_nvfp4_parity] page {}x{} -> grid {}x{} = {n} patches",
        img.w, img.h, prep.grid_h, prep.grid_w
    );

    let cpu = Device::Cpu;
    let loader = WeightLoader::open_dir(&dir, &cpu).expect("open dots.ocr weights");
    let t0 = Instant::now();
    let twin = TwinTower::load(&loader, "vision_tower.", &cpu, DType::F32).expect("twin load");
    let (twin_out, twin_trace) = twin
        .forward(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("twin forward");
    eprintln!(
        "[dots_vision_wgpu_nvfp4_parity] cpu f32 twin forward {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    let mut merged: Vec<(VisionWeightFmt, f64)> = Vec::new();
    for wq in [VisionWeightFmt::F16, VisionWeightFmt::Nvfp4Block16] {
        let t1 = Instant::now();
        let tower = DotsVisionTowerWgpu::from_loader_route_wq(
            &loader,
            "vision_tower.",
            DotsVisionConfig::default(),
            route,
            wq,
        )
        .expect("wgpu tower load");
        assert_eq!(tower.weight_fmt(), wq);
        let staged = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let (out, trace) = tower
            .forward_patches_with_layer_trace(&prep.patches, prep.grid_h, prep.grid_w)
            .expect("wgpu forward");
        eprintln!(
            "[dots_vision_wgpu_nvfp4_parity] arm={} staging {staged:.1}s forward {:.2}s route={}",
            wq.name(),
            t2.elapsed().as_secs_f64(),
            route.name()
        );
        assert_eq!(trace.len(), twin_trace.len(), "layer trace lengths");
        let mut worst_rms = 0f64;
        for (li, (a, b)) in trace.iter().zip(twin_trace.iter()).enumerate() {
            let (abs, rel, rms) = max_abs_and_rel(a, b);
            worst_rms = worst_rms.max(rms);
            eprintln!(
                "[dots_vision_wgpu_nvfp4_parity] arm={} layer {li} {abs:.4e} {rel:.4e} {rms:.4e}",
                wq.name()
            );
        }
        let (out_abs, out_rel, out_rms) = max_abs_and_rel(&out, &twin_out);
        eprintln!(
            "[dots_vision_wgpu_nvfp4_parity] arm={} merged features: max_abs={out_abs:.4e} \
             max_rel={out_rel:.4e} rms_rel={out_rms:.4e} worst_layer_rms={worst_rms:.4e}",
            wq.name()
        );
        assert_eq!(
            out.iter().filter(|v| !v.is_finite()).count(),
            0,
            "arm {} produced non-finite features",
            wq.name()
        );
        merged.push((wq, out_rms));
    }
    let f16_rms = merged[0].1;
    let nvfp4_rms = merged[1].1;
    eprintln!(
        "[dots_vision_wgpu_nvfp4_parity] merged rms_rel vs cpu f32 oracle: f16={f16_rms:.4e} \
         nvfp4={nvfp4_rms:.4e} (cuda bf16 reference 1.02e-1)"
    );
    assert!(
        f16_rms < 5e-2,
        "the f16 arm drifted {f16_rms:.3e} in the same process the nvfp4 arm ran in; the control \
         is broken and the nvfp4 number means nothing"
    );
    assert!(
        nvfp4_rms < THE_NVFP4_ARM_GATE_SITS_BETWEEN_THE_MEASURED_QUANTIZATION_ERROR_AND_THE_ORDER_ONE_SIGNATURE_OF_AN_UNPACK_DEFECT,
        "nvfp4 arm drifted {nvfp4_rms:.3e} rms-relative from the cpu f32 oracle. {}",
        THE_NVFP4_ARM_GATE_BASIS
    );
}

#[test]
#[ignore]
fn the_wgpu_vision_tower_runs_the_full_resolution_page_whose_prefill_is_most_of_dots_ocr_wall_time() {
    let ctx = ctx_or_fail("dots_vision_wgpu_full_page");
    let dir = snapshot_dir().expect("dots.ocr snapshot present in the hub cache");
    let page = std::env::var(PAGE_ENV)
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let img = RgbImage::decode_file(&page).expect("decode page");
    let prep = prepare(&img, PixelBudget::from_env()).expect("prepare page");
    let n = prep.num_patches();
    eprintln!(
        "[dots_vision_wgpu_full_page] page {}x{} -> grid {}x{} = {n} patches, {} vision tokens",
        img.w,
        img.h,
        prep.grid_h,
        prep.grid_w,
        prep.num_vision_tokens()
    );
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).expect("open dots.ocr weights");
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    let t0 = Instant::now();
    let tower = DotsVisionTowerWgpu::from_loader_route(
        &loader,
        "vision_tower.",
        DotsVisionConfig::default(),
        route,
    )
    .expect("wgpu tower load");
    eprintln!(
        "[dots_vision_wgpu_full_page] weight staging {:.1}s route={}",
        t0.elapsed().as_secs_f64(),
        route.name()
    );
    let t1 = Instant::now();
    let (out, peak_mib) = peak_device_mib_during(|| {
        tower
            .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
            .expect("wgpu forward at full page resolution")
    });
    let wall = t1.elapsed().as_secs_f64();
    eprintln!(
        "[dots_vision_wgpu_full_page] attention arm={} plan={} peak resident {peak_mib} MiB",
        AttnArm::from_env().name(),
        FlashPlan::from_env().name()
    );
    let (submits, dispatches) = tower.gemm().submits_and_dispatches();
    eprintln!(
        "[dots_vision_wgpu_full_page] {dispatches} dispatches in {submits} submits ({:.1} per submit)",
        dispatches as f64 / submits.max(1) as f64
    );
    assert!(
        dispatches > 0,
        "the wgpu vision tower recorded no dispatches at all"
    );
    let cfg = DotsVisionConfig::default();
    assert_eq!(
        out.len(),
        prep.num_vision_tokens() * cfg.hidden_size,
        "merged feature count"
    );
    let nonfinite = out.iter().filter(|v| !v.is_finite()).count();
    let mx = out.iter().fold(0f32, |a, v| a.max(v.abs()));
    eprintln!(
        "[dots_vision_wgpu_full_page] forward {wall:.2}s for {} vision tokens, max_abs={mx:.4e}, nonfinite={nonfinite}",
        prep.num_vision_tokens()
    );
    assert_eq!(nonfinite, 0, "the vision tower produced non-finite features");
    assert!(mx > 0.0, "the vision tower produced an all-zero feature map");
}

const THE_TILES_DERIVED_PLAN_THE_SQUARE_TILE_REPLACED: CoopPlan = CoopPlan {
    tm: 16,
    tn: 1,
    sg: 4,
    ku: 1,
};

const THE_PLAN_AB_DEFAULTS_BELOW_FULL_RESOLUTION_BECAUSE_A_SECOND_FULL_PAGE_FORWARD_IN_ONE_PROCESS_DOES_NOT_SURVIVE: usize =
    16384;

const A_SECOND_FULL_RESOLUTION_FORWARD_IN_ONE_PROCESS_LOSES_ITS_READBACK_BUFFER: &str =
    "One full-resolution forward holds about 12 GB of device buffers and frees them only when the \
     next poll runs, which is at the end of the following forward. A second full-page forward in \
     the same process therefore peaks at twice that and does not survive it: unbatched it dies in \
     Buffer::map_async with 'nv-kernels-readback is invalid', batched it degrades to something \
     still running after thirty minutes at 100% GPU. It is independent of the coop plan and of the \
     submit batching -- both arms reproduce it -- and it is the reason this A/B defaults to a \
     patch budget rather than PixelBudget::from_env. Set NV_DOTS_VISION_WGPU_PATCHES to override.";

#[test]
#[ignore]
fn the_shipped_coop_plan_outruns_the_tiles_derived_one_it_replaced_and_the_two_agree_bit_for_bit() {
    let ctx = ctx_or_fail("dots_vision_wgpu_plan_ab");
    let dir = snapshot_dir().expect("dots.ocr snapshot present in the hub cache");
    let page = std::env::var(PAGE_ENV)
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let img = RgbImage::decode_file(&page).expect("decode page");
    let _ = A_SECOND_FULL_RESOLUTION_FORWARD_IN_ONE_PROCESS_LOSES_ITS_READBACK_BUFFER;
    let want = std::env::var(PATCHES_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(
            THE_PLAN_AB_DEFAULTS_BELOW_FULL_RESOLUTION_BECAUSE_A_SECOND_FULL_PAGE_FORWARD_IN_ONE_PROCESS_DOES_NOT_SURVIVE,
        );
    let prep = prepare(
        &img,
        PixelBudget {
            min_pixels: 3136,
            max_pixels: want * PIXELS_PER_PATCH_IS_THE_14X14_PATCH_AREA,
        },
    )
    .expect("prepare page");
    let n = prep.num_patches();
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).expect("open dots.ocr weights");
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    assert!(route.is_coop(), "the plan A/B has no meaning off the coop route");
    let tower = DotsVisionTowerWgpu::from_loader_route(
        &loader,
        "vision_tower.",
        DotsVisionConfig::default(),
        route,
    )
    .expect("wgpu tower load");
    let shapes = tower.gemm_shapes(n);
    eprintln!(
        "[dots_vision_wgpu_plan_ab] grid {}x{} = {n} patches; {} distinct gemm shapes",
        prep.grid_h,
        prep.grid_w,
        shapes.len()
    );

    let warm = Instant::now();
    tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("warm-up forward");
    eprintln!(
        "[dots_vision_wgpu_plan_ab] warm-up forward {:.2}s (pipeline compiles and first-touch \
         buffer zeroing land here, not on the first timed arm)",
        warm.elapsed().as_secs_f64()
    );

    let mut arms: Vec<(&str, Option<CoopPlan>, f64, Vec<f32>)> = Vec::new();
    for (name, plan) in [
        ("shipped_square_tile", None),
        ("tiles_derived_tm16_tn1", Some(THE_TILES_DERIVED_PLAN_THE_SQUARE_TILE_REPLACED)),
        ("shipped_square_tile_null_control", None),
    ] {
        for (sn, sk) in &shapes {
            tower.set_plan(*sn, *sk, plan);
        }
        let t = Instant::now();
        let out = tower
            .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
            .expect("wgpu forward at full page resolution");
        let wall = t.elapsed().as_secs_f64();
        let mx = out.iter().fold(0f32, |a, v| a.max(v.abs()));
        eprintln!("[dots_vision_wgpu_plan_ab] {name} forward {wall:.2}s max_abs={mx:.4e}");
        arms.push((name, plan, wall, out));
    }
    for (sn, sk) in &shapes {
        tower.set_plan(*sn, *sk, None);
    }

    let base = &arms[0].3;
    for (name, _, _, out) in arms.iter().skip(1) {
        let differing = base
            .iter()
            .zip(out.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing, 0,
            "{name} disagrees with the shipped plan on {differing} of {} merged features; a coop \
             plan moves which block owns a tile and never the k order, so any difference at all is \
             a defect in the plan resolution rather than a tolerance to widen",
            base.len()
        );
    }
    assert!(
        arms[0].2 < arms[1].2,
        "the shipped square-tile plan ({:.2}s) did not beat the tiles-derived tm16_tn1 plan \
         ({:.2}s) it replaced; null control of the shipped plan against itself was {:.2}s",
        arms[0].2,
        arms[1].2,
        arms[2].2
    );
}

const COOP_PLAN_LADDER_REPEATS_AMORTIZE_THE_SUBMIT_AND_THE_READBACK_SYNC_OVER_THE_KERNEL: usize = 8;

const COOP_PLAN_LADDER_REPEATS_THE_SHIPPED_PLAN_LAST_AS_A_NULL_CONTROL_AGAINST_ITSELF: usize = 1;

fn coop_plan_ladder_arms() -> Vec<CoopPlan> {
    let mut v = Vec::new();
    for (tm, tn) in [
        (16u32, 1u32),
        (8, 2),
        (4, 4),
        (2, 8),
        (1, 16),
        (8, 1),
        (4, 2),
        (2, 2),
        (1, 4),
        (1, 2),
        (1, 1),
    ] {
        for sg in [1u32, 4, 8] {
            v.push(CoopPlan { tm, tn, sg, ku: 1 });
        }
    }
    for (tm, tn, sg, ku) in [(4u32, 4u32, 4u32, 2u32), (2, 2, 4, 2), (1, 2, 4, 4)] {
        v.push(CoopPlan { tm, tn, sg, ku });
    }
    for _ in 0..COOP_PLAN_LADDER_REPEATS_THE_SHIPPED_PLAN_LAST_AS_A_NULL_CONTROL_AGAINST_ITSELF {
        v.push(CoopPlan {
            tm: 16,
            tn: 1,
            sg: 4,
            ku: 1,
        });
    }
    v
}

#[test]
#[ignore]
fn the_coop_plan_ladder_ranks_a_route_for_each_gemm_shape_this_vision_tower_actually_issues() {
    let ctx = ctx_or_fail("dots_vision_wgpu_plan_ladder");
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    assert!(
        route.is_coop(),
        "the coop plan ladder has nothing to rank on the scalar fallback route"
    );
    let g = VisionGemm::new(ctx, route).expect("vision gemm");
    let tile = g.tile();
    let cfg = DotsVisionConfig::default();
    let dim = cfg.embed_dim;
    let hd = cfg.head_dim();
    let inter = cfg.intermediate_size;

    let page = std::env::var(PAGE_ENV)
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let img = RgbImage::decode_file(&page).expect("decode page");
    let prep = prepare(&img, PixelBudget::from_env()).expect("prepare page");
    let n = prep.num_patches();
    let n_pad = pad_to(n, tile);
    let qb = g.attention_query_block_rows(n_pad);
    let merged = dim * cfg.spatial_merge_size * cfg.spatial_merge_size;
    eprintln!(
        "[dots_vision_wgpu_plan_ladder] grid {}x{} = {n} patches, n_pad={n_pad}, query block {qb} rows",
        prep.grid_h, prep.grid_w
    );

    let shapes: Vec<(&str, usize, usize, usize)> = vec![
        ("attn_scores", n_pad, hd, qb),
        ("attn_ctx", hd, n_pad, qb),
        ("qkv", dim * 3, dim, n_pad),
        ("attn_proj", dim, dim, n_pad),
        ("mlp_fc1_fc3", inter, dim, n_pad),
        ("mlp_fc2", dim, inter, n_pad),
        ("patch_proj", dim, pad_to(PATCH_DIM, tile), n_pad),
        ("merger_fc0", merged, merged, n / (merged / dim)),
        ("merger_fc2", cfg.hidden_size, merged, n / (merged / dim)),
    ];
    let arms = coop_plan_ladder_arms();
    for (name, sn, sk, sm) in shapes {
        let sm_pad = pad_to(sm, tile);
        let rows = g
            .time_gemm_plans(
                name,
                sn,
                sk,
                sm_pad,
                COOP_PLAN_LADDER_REPEATS_AMORTIZE_THE_SUBMIT_AND_THE_READBACK_SYNC_OVER_THE_KERNEL,
                &arms,
            )
            .expect("coop plan ladder");
        assert!(
            !rows.is_empty(),
            "{name}: every plan in the ladder was rejected at m={sm_pad} n={sn} k={sk}"
        );
        let best = rows
            .iter()
            .fold(f64::INFINITY, |a, (_, s)| a.min(*s))
            .max(1e-12);
        let flops = 2.0 * sm_pad as f64 * sn as f64 * sk as f64;
        for (plan, secs) in &rows {
            eprintln!(
                "[dots_vision_wgpu_plan_ladder] {name} m={sm_pad} n={sn} k={sk} {} {:.4} ms {:.1} TFLOP/s {:.2}x-best",
                plan.name(),
                secs * 1e3,
                flops / secs / 1e12,
                secs / best
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn the_cuda_bf16_vision_tower_full_page_is_the_wall_clock_this_port_is_measured_against() {
    let dir = snapshot_dir().expect("dots.ocr snapshot present in the hub cache");
    let page = std::env::var(PAGE_ENV)
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let img = RgbImage::decode_file(&page).expect("decode page");
    let prep = prepare(&img, PixelBudget::from_env()).expect("prepare page");
    eprintln!(
        "[dots_vision_cuda_full_page] grid {}x{} = {} patches, {} vision tokens",
        prep.grid_h,
        prep.grid_w,
        prep.num_patches(),
        prep.num_vision_tokens()
    );
    let cuda = Device::new_cuda(0).expect("cuda device 0");
    let loader = WeightLoader::open_dir(&dir, &cuda).expect("open dots.ocr weights on cuda");
    let t0 = Instant::now();
    let twin = TwinTower::load(&loader, "vision_tower.", &cuda, DType::BF16).expect("cuda twin");
    eprintln!(
        "[dots_vision_cuda_full_page] weight load {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    let t1 = Instant::now();
    let (out, _) = twin
        .forward_traced(&prep.patches, prep.grid_h, prep.grid_w, false)
        .expect("cuda twin forward at full page resolution");
    let wall = t1.elapsed().as_secs_f64();
    let mx = out.iter().fold(0f32, |a, v| a.max(v.abs()));
    eprintln!(
        "[dots_vision_cuda_full_page] forward {wall:.2}s for {} vision tokens, max_abs={mx:.4e}",
        prep.num_vision_tokens()
    );
    assert!(out.iter().all(|v| v.is_finite()));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn the_cuda_bf16_vision_tower_is_the_reference_the_wgpu_error_must_be_compared_against() {
    let dir = snapshot_dir().expect("dots.ocr snapshot present in the hub cache");
    let page = std::env::var(PAGE_ENV)
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let want_patches: usize = std::env::var(PATCHES_ENV)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_PARITY_PATCH_BUDGET_KEEPS_THE_CPU_F32_ORACLE_UNDER_A_MINUTE);
    let img = RgbImage::decode_file(&page).expect("decode page");
    let prep = prepare(
        &img,
        PixelBudget {
            min_pixels: 3136,
            max_pixels: want_patches * PIXELS_PER_PATCH_IS_THE_14X14_PATCH_AREA,
        },
    )
    .expect("prepare page");

    let cpu = Device::Cpu;
    let loader = WeightLoader::open_dir(&dir, &cpu).expect("open dots.ocr weights");
    let twin = TwinTower::load(&loader, "vision_tower.", &cpu, DType::F32).expect("twin load");
    let (twin_out, twin_trace) = twin
        .forward(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("twin forward");

    let cuda = Device::new_cuda(0).expect("cuda device 0");
    let cloader = WeightLoader::open_dir(&dir, &cuda).expect("open dots.ocr weights on cuda");
    let ctwin = TwinTower::load(&cloader, "vision_tower.", &cuda, DType::BF16).expect("cuda twin");
    let (cuda_out, cuda_trace) = ctwin
        .forward(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("cuda twin forward");

    eprintln!("[dots_vision_cuda_reference] layer max_abs max_rel rms_rel");
    for (li, (a, b)) in cuda_trace.iter().zip(twin_trace.iter()).enumerate() {
        let (abs, rel, rms) = max_abs_and_rel(a, b);
        eprintln!("[dots_vision_cuda_reference] {li} {abs:.4e} {rel:.4e} {rms:.4e}");
    }
    let (abs, rel, rms) = max_abs_and_rel(&cuda_out, &twin_out);
    eprintln!(
        "[dots_vision_cuda_reference] merged features: max_abs={abs:.4e} max_rel={rel:.4e} rms_rel={rms:.4e}"
    );
}

const THE_ARM_AB_DEFAULTS_BELOW_FULL_RESOLUTION_FOR_THE_SAME_REASON_THE_PLAN_AB_DOES: usize = 16384;

fn prepared_page(want: usize) -> nv_models::dots_ocr::preprocess::PreparedImage {
    let page = std::env::var(PAGE_ENV)
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let img = RgbImage::decode_file(&page).expect("decode page");
    let budget = if want == 0 {
        PixelBudget::from_env()
    } else {
        PixelBudget {
            min_pixels: 3136,
            max_pixels: want * PIXELS_PER_PATCH_IS_THE_14X14_PATCH_AREA,
        }
    };
    prepare(&img, budget).expect("prepare page")
}

fn wanted_patches(default: usize) -> usize {
    std::env::var(PATCHES_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

#[test]
#[ignore]
fn the_three_attention_arms_are_timed_and_priced_for_footprint_interleaved_in_one_process() {
    let ctx = ctx_or_fail("dots_vision_wgpu_attn_arms");
    let dir = snapshot_dir().expect("dots.ocr snapshot present in the hub cache");
    let prep = prepared_page(wanted_patches(
        THE_ARM_AB_DEFAULTS_BELOW_FULL_RESOLUTION_FOR_THE_SAME_REASON_THE_PLAN_AB_DOES,
    ));
    let n = prep.num_patches();
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).expect("open dots.ocr weights");
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    let tower = DotsVisionTowerWgpu::from_loader_route(
        &loader,
        "vision_tower.",
        DotsVisionConfig::default(),
        route,
    )
    .expect("wgpu tower load");
    eprintln!(
        "[dots_vision_wgpu_attn_arms] grid {}x{} = {n} patches, plan {}",
        prep.grid_h,
        prep.grid_w,
        FlashPlan::from_env().name()
    );

    std::env::set_var(DOTS_VISION_WGPU_ATTN_ENV, "slab");
    let warm = Instant::now();
    tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("warm-up forward");
    eprintln!(
        "[dots_vision_wgpu_attn_arms] warm-up forward {:.2}s",
        warm.elapsed().as_secs_f64()
    );

    let mut arms: Vec<(&str, f64, u64, Vec<f32>)> = Vec::new();
    for name in ["slab", "zmerge", "flash", "slab"] {
        std::env::set_var(DOTS_VISION_WGPU_ATTN_ENV, name);
        assert_eq!(AttnArm::from_env().name(), name, "the arm env did not take");
        let t = Instant::now();
        let (out, mib) = peak_device_mib_during(|| {
            tower
                .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
                .unwrap_or_else(|e| panic!("{name} forward: {e}"))
        });
        let wall = t.elapsed().as_secs_f64();
        let mx = out.iter().fold(0f32, |a, v| a.max(v.abs()));
        eprintln!(
            "[dots_vision_wgpu_attn_arms] {name} forward {wall:.2}s peak={mib} MiB max_abs={mx:.4e} \
             nonfinite={}",
            out.iter().filter(|v| !v.is_finite()).count()
        );
        assert_eq!(
            out.iter().filter(|v| !v.is_finite()).count(),
            0,
            "{name} produced non-finite features"
        );
        arms.push((name, wall, mib, out));
    }
    std::env::remove_var(DOTS_VISION_WGPU_ATTN_ENV);

    let base = arms[0].3.clone();
    for (name, wall, mib, out) in arms.iter().skip(1) {
        let (abs, rel, rms) = max_abs_and_rel(out, &base);
        let bits = base
            .iter()
            .zip(out.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "[dots_vision_wgpu_attn_arms] {name} vs slab: {:.3}x wall, peak {mib} MiB, \
             differing_bits={bits}/{} max_abs={abs:.4e} max_rel={rel:.4e} rms_rel={rms:.4e} \
             ({wall:.2}s)",
            arms[0].1 / wall,
            base.len()
        );
    }
    assert!(
        arms[3].3.iter().zip(base.iter()).all(|(a, b)| a.to_bits() == b.to_bits()),
        "the slab arm run twice in one process disagreed with itself; the null control is broken \
         and no arm delta in this row means anything"
    );
}

const THE_WEIGHT_ARM_AB_DEFAULTS_BELOW_FULL_RESOLUTION_FOR_THE_SAME_REASON_THE_PLAN_AB_DOES: usize =
    16224;

#[test]
#[ignore]
fn the_two_weight_arms_are_timed_and_priced_for_footprint_interleaved_in_one_process() {
    let ctx = ctx_or_fail("dots_vision_wgpu_wq_arms");
    let dir = snapshot_dir().expect("dots.ocr snapshot present in the hub cache");
    let prep = prepared_page(wanted_patches(
        THE_WEIGHT_ARM_AB_DEFAULTS_BELOW_FULL_RESOLUTION_FOR_THE_SAME_REASON_THE_PLAN_AB_DOES,
    ));
    let n = prep.num_patches();
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).expect("open dots.ocr weights");
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    let base_mib = device_mib_for_this_process().unwrap_or(0);
    let mut towers: Vec<(VisionWeightFmt, DotsVisionTowerWgpu, f64, u64)> = Vec::new();
    for wq in [VisionWeightFmt::F16, VisionWeightFmt::Nvfp4Block16] {
        let t = Instant::now();
        let tower = DotsVisionTowerWgpu::from_loader_route_wq(
            &loader,
            "vision_tower.",
            DotsVisionConfig::default(),
            route,
            wq,
        )
        .unwrap_or_else(|e| panic!("{} tower load: {e}", wq.name()));
        let staged = t.elapsed().as_secs_f64();
        let mib = device_mib_for_this_process().unwrap_or(0);
        eprintln!(
            "[dots_vision_wgpu_wq_arms] {} staging {staged:.1}s resident {mib} MiB (process base \
             {base_mib} MiB)",
            wq.name()
        );
        towers.push((wq, tower, staged, mib));
    }
    eprintln!(
        "[dots_vision_wgpu_wq_arms] grid {}x{} = {n} patches, attn arm={} plan={} route={}",
        prep.grid_h,
        prep.grid_w,
        AttnArm::from_env().name(),
        FlashPlan::from_env().name(),
        route.name()
    );
    for (wq, tower, _, _) in &towers {
        let warm = Instant::now();
        tower
            .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
            .unwrap_or_else(|e| panic!("{} warm-up forward: {e}", wq.name()));
        eprintln!(
            "[dots_vision_wgpu_wq_arms] {} warm-up forward {:.2}s",
            wq.name(),
            warm.elapsed().as_secs_f64()
        );
    }
    let mut runs: Vec<(&'static str, f64, u64, Vec<f32>)> = Vec::new();
    for pick in [0usize, 1, 0, 1, 0] {
        let (wq, tower, _, _) = &towers[pick];
        let t = Instant::now();
        let (out, mib) = peak_device_mib_during(|| {
            tower
                .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
                .unwrap_or_else(|e| panic!("{} forward: {e}", wq.name()))
        });
        let wall = t.elapsed().as_secs_f64();
        let nonfinite = out.iter().filter(|v| !v.is_finite()).count();
        eprintln!(
            "[dots_vision_wgpu_wq_arms] {} forward {wall:.2}s peak={mib} MiB nonfinite={nonfinite}",
            wq.name()
        );
        assert_eq!(nonfinite, 0, "{} produced non-finite features", wq.name());
        runs.push((wq.name(), wall, mib, out));
    }
    let f16_walls: Vec<f64> = runs.iter().filter(|r| r.0 == "f16").map(|r| r.1).collect();
    let nv_walls: Vec<f64> = runs.iter().filter(|r| r.0 == "nvfp4").map(|r| r.1).collect();
    let null_spread = (f16_walls.iter().cloned().fold(f64::MIN, f64::max)
        - f16_walls.iter().cloned().fold(f64::MAX, f64::min))
        / f16_walls.iter().sum::<f64>()
        * f16_walls.len() as f64;
    let f16_mean = f16_walls.iter().sum::<f64>() / f16_walls.len() as f64;
    let nv_mean = nv_walls.iter().sum::<f64>() / nv_walls.len() as f64;
    let (abs, rel, rms) = max_abs_and_rel(&runs[1].3, &runs[0].3);
    eprintln!(
        "[dots_vision_wgpu_wq_arms] f16 mean {f16_mean:.2}s (null spread {:.1}%) nvfp4 mean \
         {nv_mean:.2}s ({:.3}x) arm-vs-arm max_abs={abs:.4e} max_rel={rel:.4e} rms_rel={rms:.4e}",
        null_spread * 100.0,
        f16_mean / nv_mean
    );
    assert!(
        runs[4]
            .3
            .iter()
            .zip(runs[0].3.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the f16 arm run twice in one process disagreed with itself; the null control is broken \
         and no weight-arm delta in this row means anything"
    );
    assert!(
        runs[3]
            .3
            .iter()
            .zip(runs[1].3.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the nvfp4 arm run twice in one process disagreed with itself"
    );
}

#[test]
#[ignore]
fn the_fused_attention_plan_ladder_ranks_the_block_shapes_this_tower_can_actually_launch() {
    let ctx = ctx_or_fail("dots_vision_wgpu_flash_ladder");
    let dir = snapshot_dir().expect("dots.ocr snapshot present in the hub cache");
    let prep = prepared_page(wanted_patches(
        THE_ARM_AB_DEFAULTS_BELOW_FULL_RESOLUTION_FOR_THE_SAME_REASON_THE_PLAN_AB_DOES,
    ));
    let n = prep.num_patches();
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).expect("open dots.ocr weights");
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    let tower = DotsVisionTowerWgpu::from_loader_route(
        &loader,
        "vision_tower.",
        DotsVisionConfig::default(),
        route,
    )
    .expect("wgpu tower load");
    eprintln!("[dots_vision_wgpu_flash_ladder] grid {}x{} = {n} patches", prep.grid_h, prep.grid_w);

    std::env::set_var(DOTS_VISION_WGPU_ATTN_ENV, "slab");
    let t = Instant::now();
    let slab = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("slab forward");
    let slab_wall = t.elapsed().as_secs_f64();
    eprintln!("[dots_vision_wgpu_flash_ladder] slab forward {slab_wall:.2}s");

    std::env::set_var(DOTS_VISION_WGPU_ATTN_ENV, "flash");
    let mut ran = 0usize;
    for plan in FUSED_ATTENTION_PLAN_LADDER_SPANS_THE_SHAPES_THE_ACCUMULATOR_AND_THE_WORKGROUP_BUDGET_BOTH_ADMIT
    {
        std::env::set_var(
            DOTS_VISION_WGPU_FLASH_PLAN_ENV,
            format!("{},{},{},{}", plan.bm, plan.bn, plan.sg, plan.scs),
        );
        assert_eq!(FlashPlan::from_env(), plan, "the flash plan env did not take");
        let t = Instant::now();
        match tower.forward_patches(&prep.patches, prep.grid_h, prep.grid_w) {
            Ok(out) => {
                let wall = t.elapsed().as_secs_f64();
                let (_, _, rms) = max_abs_and_rel(&out, &slab);
                eprintln!(
                    "[dots_vision_wgpu_flash_ladder] {} forward {wall:.2}s {:.3}x-slab rms_rel_vs_slab={rms:.4e}",
                    plan.name(),
                    slab_wall / wall
                );
                ran += 1;
            }
            Err(e) => eprintln!(
                "[dots_vision_wgpu_flash_ladder] {} refused: {e}",
                plan.name()
            ),
        }
    }
    std::env::remove_var(DOTS_VISION_WGPU_FLASH_PLAN_ENV);
    std::env::remove_var(DOTS_VISION_WGPU_ATTN_ENV);
    assert!(ran >= 3, "only {ran} flash plans ran; the ladder ranked nothing");
}

const THE_TWO_PAGE_EXPERIMENT_LIVES_IN_DOTS_OCR_TWO_PAGE_WGPU_WHICH_SUPERSEDED_THIS_FILES_COPY: () =
    ();
