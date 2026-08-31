#![cfg(feature = "wgpu")]

mod common;
use common::have_gpu;
use common::q3d_lora_synth::{synth_adapter, write_adapter, Fill};
use common::rel_err;
use common::tiny_config_q3d_mixed_layers as tiny_config;
use common::tiny_weights_q3d as tiny_weights;
use half::bf16;
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::{compose, dispatch};
use nv_models::qwen3_5_dense_wgpu as q3d;
use q3d::{Q3dLora, Qwen3_5DenseWgpu};

const GEMV_W4A8_WGSL: &str = include_str!("../../nv-kernels/wgsl/gemv_nvfp4_w4a8.wgsl");
const W4A8_PRODUCER_WGSL: &str = include_str!("../../nv-kernels/wgsl/w4a8_producer_rowquant.wgsl");
const GEMV_BF16_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3d_gemv_bf16.wgsl");

const W4A8_ENV: &str =
    q3d::MLP_W4A8_ENV_DEFAULT_ON_COSTS_0_0070_NATS_OF_TEACHER_FORCED_FLOOR_FOR_2PCT_DECODE_MEASURED_POST_FLIP_THE_FLIP_CLAIM_HAD_NO_LANDED_ROW_SET_0_FOR_THE_BF16_ACTIVATION_MLP_ARM;
const FOLD_ENV: &str =
    q3d::LORA_GEMV_FOLD_ENV_EXPAND_RIDES_THE_W4A8_GEMV_EPILOGUE_ADDING_DELTA_BEFORE_THE_BF16_ENCODE_DEFAULT_ON_SET_0_FOR_THE_TWO_PASS_SITE;

#[test]
fn the_w4a8_lora_fold_shaders_validate_without_a_gpu() {
    for (what, src, entries) in [
        (
            "gemv_nvfp4_w4a8",
            compose(GEMV_W4A8_WGSL),
            vec!["gemv_nvfp4_w4a8_mrow2", "gemv_nvfp4_w4a8_mrow2_lora_fold"],
        ),
        (
            "w4a8_producer_rowquant",
            compose(W4A8_PRODUCER_WGSL),
            vec!["silu_mul_rowquant_evenodd", "silu_mul_rowquant_evenodd_act_bf16"],
        ),
        (
            "q3d_gemv_bf16",
            compose(GEMV_BF16_WGSL),
            vec![
                "q3w_gemv_fp8_rowscale",
                "q3w_gemv_fp8_rowscale_lora_fold",
                "q3w_gemv_dn_merged_fp8_qkv_fp8_z_bf16_ab",
                "q3w_gemv_dn_merged_fp8_qkv_fp8_z_bf16_ab_lora_fold",
            ],
        ),
    ] {
        for entry in &entries {
            assert!(
                src.contains(&format!("fn {entry}(")),
                "{what}: {entry} entry point vanished"
            );
        }
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("{what} wgsl parse: {}", e.message()));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{what} wgsl validate: {e:?}"));
    }
}

fn bf16_ulp_key(bits: u16) -> i32 {
    let mag = (bits & 0x7fff) as i32;
    if bits & 0x8000 != 0 {
        -mag
    } else {
        mag
    }
}

fn bf16_spacing_at(v: f32) -> f32 {
    let m = v.abs();
    if !m.is_finite() {
        return f32::INFINITY;
    }
    if m == 0.0 {
        return f32::MIN_POSITIVE;
    }
    let e = ((m.to_bits() >> 23) & 0xff) as i32 - 127;
    2f32.powi(e - 7)
}

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }

    fn unit(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn bf16_bits(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| bf16::from_f32(self.unit() * scale).to_bits())
            .collect()
    }

    fn words(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.next_u32()).collect()
    }

    fn scale_words(&mut self, n: usize) -> Vec<u32> {
        (0..n)
            .map(|_| {
                let mut w = 0u32;
                for b in 0..4 {
                    let exp = 1 + (self.next_u32() % 8);
                    let mant = self.next_u32() % 8;
                    w |= ((exp << 3) | mant) << (8 * b);
                }
                w
            })
            .collect()
    }
}

fn oracle_slice_deltas(
    x: &[u16],
    a: &[u16],
    b: &[u16],
    widths: &[usize],
    slice_id: usize,
    rank: usize,
    k: usize,
) -> Vec<f64> {
    let dec = |u: u16| bf16::from_bits(u).to_f32();
    let mut h = vec![0f64; rank];
    for (r, hr) in h.iter_mut().enumerate() {
        let a_base = (slice_id * rank + r) * k;
        let mut s = 0f64;
        for kk in 0..k {
            s += dec(x[kk]) as f64 * dec(a[a_base + kk]) as f64;
        }
        *hr = s;
    }
    let b_off: usize = widths[..slice_id].iter().map(|w| w * rank).sum();
    (0..widths[slice_id])
        .map(|j| {
            let mut d = 0f64;
            for (r, hr) in h.iter().enumerate() {
                d += hr * dec(b[b_off + j * rank + r]) as f64;
            }
            d
        })
        .collect()
}

fn assert_fused_rows_match_the_scaled_ulp_bound(
    tag: &str,
    y_base: &[u16],
    y_fused: &[u16],
    deltas: &[f64],
) -> usize {
    let dec = |u: u16| bf16::from_bits(u).to_f32();
    let mut live_rows = 0usize;
    for (j, d) in deltas.iter().enumerate() {
        let base = dec(y_base[j]);
        assert!(base.is_finite(), "{tag} row {j}: base gemv is not finite");
        let want = base as f64 + d;
        let fused = dec(y_fused[j]);
        let tol = bf16_spacing_at(base) as f64
            + 4.0 * bf16_spacing_at(*d as f32) as f64
            + bf16_spacing_at(want as f32) as f64;
        assert!(
            (fused as f64 - want).abs() <= tol,
            "{tag} row {j}: fused {fused} vs base+oracle-delta {want} beyond \
             {tol} = one bf16 ulp of the base (the fold adds delta before the encode the \
             base output already paid) + four of the delta (f32 shrink accumulation) + one \
             of the result; a wrong part_base/b_base or a dropped tile shifts delta mass \
             far past this"
        );
        if bf16::from_f32(base + *d as f32).to_bits() != y_base[j] {
            live_rows += 1;
        }
    }
    live_rows
}

fn assert_live_rows(tag: &str, live_rows: usize, n: usize) {
    assert!(
        live_rows > n / 4,
        "{tag}: only {live_rows}/{n} rows carry a delta visible above bf16 rounding; \
         the oracle cannot distinguish a dead epilogue from a live one at this magnitude"
    );
}

#[test]
fn the_fold_epilogue_matches_the_base_gemv_plus_an_f64_delta_oracle() {
    if !have_gpu() {
        return;
    }
    let cases: [(Vec<usize>, usize, usize, usize, bool); 3] = [
        (vec![512, 1024], 1, 2048, 8, false),
        (vec![768], 0, 1024, 63, false),
        (vec![512], 0, 512, 8, true),
    ];
    for (case, (widths, slice_id, k, rank, zero_b)) in cases.iter().enumerate() {
        let n = widths[*slice_id];
        let mut rng = Lcg(0x9a48_f01d ^ (case as u64) << 8);
        let x = rng.bf16_bits(*k, 1.0);
        let a = rng.bf16_bits(widths.len() * rank * k, 0.02);
        let b_len: usize = widths.iter().map(|w| w * rank).sum();
        let b = if *zero_b {
            vec![0u16; b_len]
        } else {
            rng.bf16_bits(b_len, 0.25)
        };
        let kb = k / 16;
        let w_packed = rng.words(n * k / 8);
        let w_scales = rng.scale_words(wk::gemv_nvfp4::swizzled_scale_len(n, kb).div_ceil(4));
        let alpha = 2e-4f32;
        let (y_base, y_fused) = q3d::w4a8_lora_fold_probe(
            &x, &a, &b, widths, *slice_id, &w_packed, &w_scales, alpha, n, *k, *rank,
        )
        .expect("fold probe");

        if *zero_b {
            for j in 0..n {
                assert_eq!(
                    y_base[j], y_fused[j],
                    "case {case} row {j}: a zero-B adapter must leave the fold output \
                     bit-identical to the base gemv"
                );
            }
            continue;
        }

        let deltas = oracle_slice_deltas(&x, &a, &b, widths, *slice_id, *rank, *k);
        let live = assert_fused_rows_match_the_scaled_ulp_bound(
            &format!("case {case}"),
            &y_base,
            &y_fused,
            &deltas,
        );
        assert_live_rows(&format!("case {case}"), live, n);
    }
}

fn e4m3_decode(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0f) as i32;
    let mant = (byte & 0x07) as f32;
    if exp == 0 {
        sign * (mant / 8.0) * 2f32.powi(-6)
    } else {
        sign * (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
    }
}

fn synth_fp8_words(rng: &mut Lcg, n: usize, k: usize) -> Vec<u32> {
    (0..n * k / 4)
        .map(|_| {
            let mut w = 0u32;
            for b in 0..4 {
                let sign = (rng.next_u32() & 1) << 7;
                let exp = 2 + (rng.next_u32() % 9);
                let mant = rng.next_u32() % 8;
                w |= (sign | (exp << 3) | mant) << (8 * b);
            }
            w
        })
        .collect()
}

fn synth_fp8_scales(rng: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| 0.01 + 0.03 * (rng.next_u32() % 100) as f32 / 100.0)
        .collect()
}

#[test]
fn the_fp8_rowscale_fold_epilogue_matches_the_base_gemv_plus_an_f64_delta_oracle() {
    if !have_gpu() {
        return;
    }
    let cases: [(Vec<usize>, usize, usize, usize, bool); 3] = [
        (vec![384, 256], 1, 512, 8, false),
        (vec![256], 0, 256, 63, false),
        (vec![128], 0, 128, 8, true),
    ];
    for (case, (widths, slice_id, k, rank, zero_b)) in cases.iter().enumerate() {
        let n = widths[*slice_id];
        let mut rng = Lcg(0xf8f8_0a11 ^ (case as u64) << 8);
        let x = rng.bf16_bits(*k, 1.0);
        let a = rng.bf16_bits(widths.len() * rank * k, 0.02);
        let b_len: usize = widths.iter().map(|w| w * rank).sum();
        let b = if *zero_b {
            vec![0u16; b_len]
        } else {
            rng.bf16_bits(b_len, 0.25)
        };
        let w = q3d::HostFp8Lin {
            packed: synth_fp8_words(&mut rng, n, *k),
            scales: synth_fp8_scales(&mut rng, n),
            n,
            k: *k,
        };
        let (y_base, y_fused) =
            q3d::fp8_rowscale_lora_fold_probe(&x, &a, &b, widths, *slice_id, &w, *rank)
                .expect("fp8 fold probe");

        if *zero_b {
            for j in 0..n {
                assert_eq!(
                    y_base[j], y_fused[j],
                    "case {case} row {j}: a zero-B adapter must leave the fp8 fold output \
                     bit-identical to the base gemv"
                );
            }
            continue;
        }
        let deltas = oracle_slice_deltas(&x, &a, &b, widths, *slice_id, *rank, *k);
        let live = assert_fused_rows_match_the_scaled_ulp_bound(
            &format!("fp8 case {case}"),
            &y_base,
            &y_fused,
            &deltas,
        );
        assert_live_rows(&format!("fp8 case {case}"), live, n);
    }
}

#[test]
fn the_dn_merged_fold_matches_the_base_gemv_plus_an_f64_delta_oracle_on_every_output() {
    if !have_gpu() {
        return;
    }
    for (case, zero_b) in [(0usize, false), (1, true)] {
        let (k, rank) = (128usize, 8usize);
        let widths = [192usize, 128, 8, 8];
        let mut rng = Lcg(0xd0d0_31ab ^ (case as u64) << 8);
        let x = rng.bf16_bits(k, 1.0);
        let a = rng.bf16_bits(widths.len() * rank * k, 0.02);
        let b_len: usize = widths.iter().map(|w| w * rank).sum();
        let b = if zero_b {
            vec![0u16; b_len]
        } else {
            rng.bf16_bits(b_len, 0.25)
        };
        let qkv = q3d::HostFp8Lin {
            packed: synth_fp8_words(&mut rng, widths[0], k),
            scales: synth_fp8_scales(&mut rng, widths[0]),
            n: widths[0],
            k,
        };
        let z = q3d::HostFp8Lin {
            packed: synth_fp8_words(&mut rng, widths[1], k),
            scales: synth_fp8_scales(&mut rng, widths[1]),
            n: widths[1],
            k,
        };
        let ab = nv_models::qwen3_5_moe_wgpu::HostBf16Lin {
            w: rng.bf16_bits((widths[2] + widths[3]) * k, 0.12),
            n: widths[2] + widths[3],
            k,
        };
        let (ys_base, ys_fused) =
            q3d::dn_merged_lora_fold_probe(&x, &a, &b, &widths, &qkv, &z, &ab, rank)
                .expect("merged fold probe");

        if zero_b {
            for (o, (yb, yf)) in ys_base.iter().zip(ys_fused.iter()).enumerate() {
                for (j, (p, q)) in yb.iter().zip(yf.iter()).enumerate() {
                    assert_eq!(
                        p, q,
                        "merged output {o} row {j}: a zero-B adapter must leave the merged \
                         fold bit-identical to the base gemv"
                    );
                }
            }
            continue;
        }

        let d_qkv = oracle_slice_deltas(&x, &a, &b, &widths, 0, rank, k);
        let d_z = oracle_slice_deltas(&x, &a, &b, &widths, 1, rank, k);
        let d_a = oracle_slice_deltas(&x, &a, &b, &widths, 2, rank, k);
        let d_b = oracle_slice_deltas(&x, &a, &b, &widths, 3, rank, k);
        let mut d_ab = d_a;
        d_ab.extend_from_slice(&d_b);
        let mut live = 0usize;
        live += assert_fused_rows_match_the_scaled_ulp_bound(
            "merged qkv",
            &ys_base[0],
            &ys_fused[0],
            &d_qkv,
        );
        live +=
            assert_fused_rows_match_the_scaled_ulp_bound("merged z", &ys_base[1], &ys_fused[1], &d_z);
        live += assert_fused_rows_match_the_scaled_ulp_bound(
            "merged ab",
            &ys_base[2],
            &ys_fused[2],
            &d_ab,
        );
        assert_live_rows("merged", live, widths.iter().sum());
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ProdParams {
    k_blocks: u32,
    words_per_row: u32,
    hidden: u32,
    eps: f32,
}

#[test]
fn the_act_storing_silu_entry_keeps_quantization_bit_identical_and_writes_the_bf16_act() {
    if !have_gpu() {
        return;
    }
    let ctx = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(c) => c,
        Err(_) => return,
    };
    let k = 4096usize;
    let mut rng = Lcg(0x51_1u64);
    let gate = rng.bf16_bits(k, 2.0);
    let up = rng.bf16_bits(k, 2.0);
    let pack = |v: &[u16]| -> Vec<u32> {
        v.chunks(2)
            .map(|c| (c[0] as u32) | ((c[1] as u32) << 16))
            .collect()
    };
    let src = compose(W4A8_PRODUCER_WGSL);
    let mk = |label: &str, entry: &str| {
        dispatch::cached_compute_pipeline(ctx, label, &src, entry).expect("producer pipeline")
    };
    let base_pipe = mk("silu-base", "silu_mul_rowquant_evenodd");
    let act_pipe = mk("silu-act", "silu_mul_rowquant_evenodd_act_bf16");
    let g = dispatch::GpuTensor::<u32>::upload(ctx, "silu-g", &pack(&gate));
    let u = dispatch::GpuTensor::<u32>::upload(ctx, "silu-u", &pack(&up));
    let p = dispatch::GpuUniform::new(
        ctx,
        "silu-p",
        &ProdParams {
            k_blocks: (k / 16) as u32,
            words_per_row: (k / 2) as u32,
            hidden: k as u32,
            eps: 0.0,
        },
    );
    let xq_a = dispatch::GpuTensor::<u32>::zeroed(ctx, "silu-xqa", k / 4);
    let xs_a = dispatch::GpuTensor::<f32>::zeroed(ctx, "silu-xsa", 1);
    let xq_b = dispatch::GpuTensor::<u32>::zeroed(ctx, "silu-xqb", k / 4);
    let xs_b = dispatch::GpuTensor::<f32>::zeroed(ctx, "silu-xsb", 1);
    let act = dispatch::GpuTensor::<u32>::zeroed(ctx, "silu-act", k / 2);
    let base_bind = dispatch::bind_group(
        ctx,
        &base_pipe,
        &[
            (4, p.raw()),
            (5, xq_a.raw()),
            (6, xs_a.raw()),
            (7, g.raw()),
            (8, u.raw()),
        ],
    );
    let act_bind = dispatch::bind_group(
        ctx,
        &act_pipe,
        &[
            (3, act.raw()),
            (4, p.raw()),
            (5, xq_b.raw()),
            (6, xs_b.raw()),
            (7, g.raw()),
            (8, u.raw()),
        ],
    );
    dispatch::submit_pass_list(
        ctx,
        [
            (base_pipe.as_ref(), &base_bind, (1u32, 1u32, 1u32)),
            (act_pipe.as_ref(), &act_bind, (1u32, 1u32, 1u32)),
        ],
    );
    ctx.poll_blocking().expect("silu poll");
    assert_eq!(
        xq_a.download(ctx).expect("xq base"),
        xq_b.download(ctx).expect("xq act"),
        "the act-storing entry changed the quantized activation stream"
    );
    assert_eq!(
        xs_a.download(ctx).expect("xs base")[0].to_bits(),
        xs_b.download(ctx).expect("xs act")[0].to_bits(),
        "the act-storing entry changed the per-tensor scale"
    );
    let act_words = act.download(ctx).expect("act");
    for i in 0..k {
        let bits = if i.is_multiple_of(2) {
            (act_words[i / 2] & 0xffff) as u16
        } else {
            (act_words[i / 2] >> 16) as u16
        };
        let gv = bf16::from_bits(gate[i]).to_f32();
        let uv = bf16::from_bits(up[i]).to_f32();
        let want =
            bf16::from_f32(bf16::from_f32(gv / (1.0 + (-gv).exp())).to_f32() * uv).to_bits();
        assert!(
            (bf16_ulp_key(bits) - bf16_ulp_key(want)).abs() <= 2,
            "act element {i}: stored {bits:#06x} vs host silu*up oracle {want:#06x} beyond 2 \
             bf16 ulps (gate {gv}, up {uv})"
        );
    }
}

fn zero_bf16_lin(n: usize, k: usize) -> q3d::HostDenseLin {
    q3d::HostDenseLin::Bf16(nv_models::qwen3_5_moe_wgpu::HostBf16Lin {
        w: vec![0u16; n * k],
        n,
        k,
    })
}

const TINY_INTER_192_IS_THE_SMALLEST_DIM_THE_MROW2_BLOCK_PAIR_ROUTE_ADMITS: usize = 192;

fn tiny_nvfp4_mlp_setup() -> (q3d::Qwen3_5DenseConfig, q3d::HostDenseWeights) {
    let mut cfg = tiny_config();
    cfg.intermediate_size = TINY_INTER_192_IS_THE_SMALLEST_DIM_THE_MROW2_BLOCK_PAIR_ROUTE_ADMITS;
    cfg.layer_types =
        vec![nv_models::qwen3_5_moe::LayerType::FullAttention; cfg.num_hidden_layers];
    let mut hw = tiny_weights(&cfg, 0xf01d_9b00_0048);
    let mut r = common::LcgOddSeedShift33SignedUnit::new(0xf01d_9b00_0049);
    let (hidden, inter) = (cfg.hidden_size, cfg.intermediate_size);
    let o_k = cfg.num_attention_heads * cfg.head_dim;
    for layer in &mut hw.layers {
        layer.mlp.gate = common::nvfp4_dense_lin(common::bf16_lin(&mut r, inter, hidden, 0.25));
        layer.mlp.up = common::nvfp4_dense_lin(common::bf16_lin(&mut r, inter, hidden, 0.25));
        layer.mlp.down = common::nvfp4_dense_lin(common::bf16_lin(&mut r, hidden, inter, 0.25));
        let q3d::HostDenseMixer::Attn(a) = &mut layer.mixer else {
            panic!("tiny_nvfp4_mlp_setup forces FullAttention on every layer");
        };
        a.o = zero_bf16_lin(hidden, o_k);
    }
    (cfg, hw)
}

const STEP_TOKENS: [u32; 6] = [3, 11, 5, 40, 2, 19];

fn run_steps(m: &mut Qwen3_5DenseWgpu) -> (Vec<u32>, Vec<Vec<f32>>) {
    let mut args = Vec::new();
    let mut logits = Vec::new();
    for (i, t) in STEP_TOKENS.iter().enumerate() {
        let (a, l) = m.decode_step_logits(*t).expect("decode step");
        assert!(
            l.iter().all(|v| v.is_finite()),
            "step {i}: logits contain non-finite values; rel_err folds NaN to zero and every \
             comparison in this gate would be vacuous"
        );
        args.push(a);
        logits.push(l);
    }
    (args, logits)
}

const ARM_VS_ARM_FOLD_REL_TOL_COVERS_ONE_PRE_VS_POST_ENCODE_ROUNDING_PER_SITE: f32 = 0.02;

#[test]
fn the_w4a8_lora_arm_keeps_the_base_fusions_and_the_fold_halves_the_lora_passes() {
    if !have_gpu() {
        return;
    }
    for var in [W4A8_ENV, FOLD_ENV] {
        assert!(
            std::env::var(var).is_err(),
            "{var} preset in the environment; this test owns that env var"
        );
    }
    let (cfg, hw) = tiny_nvfp4_mlp_setup();
    let layers = cfg.num_hidden_layers;

    std::env::set_var(W4A8_ENV, "1");
    let plain_w4a8 = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32);
    std::env::remove_var(W4A8_ENV);
    let mut plain_w4a8 = plain_w4a8.expect("plain w4a8 build");
    let p0 = plain_w4a8.pass_count();
    let pf0 = plain_w4a8.prefill_pass_count();
    assert!(pf0 > 0, "this gate needs the chunked prefill graph");
    let (plain_args, plain_logits) = run_steps(&mut plain_w4a8);
    drop(plain_w4a8);

    std::env::set_var(W4A8_ENV, "0");
    let plain_legacy = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32);
    std::env::remove_var(W4A8_ENV);
    let plain_legacy = plain_legacy.expect("plain legacy build");
    assert_eq!(
        plain_legacy.pass_count(),
        p0 + layers,
        "the w4a8 arm must keep dropping exactly the standalone quant dispatch per layer"
    );
    let pl0 = plain_legacy.pass_count();
    drop(plain_legacy);

    let synth = synth_adapter(&cfg, 0xf01d_c0a5, Fill::Coarse, None);
    let dir = write_adapter("w48fold", &synth);
    let lora = Q3dLora::from_peft_dir(&dir.0, &cfg).expect("build lora sites");
    assert_eq!(
        lora.total_pass_count(),
        8 * layers,
        "the all-attention fixture carries qkv+o+gate_up+down sites per layer at two legacy \
         passes each"
    );

    std::env::set_var(W4A8_ENV, "1");
    let ad_fold = Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), &hw, 32, Some(&lora));
    std::env::remove_var(W4A8_ENV);
    let mut ad_fold = ad_fold.expect("adapter fold build");
    assert_eq!(
        ad_fold.pass_count(),
        p0 + 6 * layers,
        "the folded adapter decode graph must keep every base w4a8 fusion and add one shrink \
         per MLP site plus the two legacy attention sites (qkv 2 + o 2 + gate_up 1 + down 1)"
    );
    assert_eq!(
        ad_fold.prefill_pass_count(),
        pf0 + lora.total_pass_count(),
        "prefill sites stay on the two-pass emitter, so the chunked prefill list grows by the \
         full legacy lora pass count"
    );
    let (fold_args, fold_logits) = run_steps(&mut ad_fold);
    drop(ad_fold);

    std::env::set_var(W4A8_ENV, "1");
    std::env::set_var(FOLD_ENV, "0");
    let ad_nofold = Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), &hw, 32, Some(&lora));
    std::env::remove_var(FOLD_ENV);
    std::env::remove_var(W4A8_ENV);
    let mut ad_nofold = ad_nofold.expect("adapter nofold build");
    assert_eq!(
        ad_nofold.pass_count(),
        p0 + 8 * layers,
        "{FOLD_ENV}=0 must restore the two-pass site emitter on the w4a8 route"
    );
    let (nofold_args, nofold_logits) = run_steps(&mut ad_nofold);
    drop(ad_nofold);

    std::env::set_var(W4A8_ENV, "0");
    let ad_legacy = Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), &hw, 32, Some(&lora));
    std::env::remove_var(W4A8_ENV);
    let mut ad_legacy = ad_legacy.expect("adapter legacy build");
    assert_eq!(
        ad_legacy.pass_count(),
        pl0 + 9 * layers,
        "the pre-fold adapter route pays eight legacy site passes plus the de-fused \
         silu/dquant pair per layer; this pins the structural cost the w4a8 arm recovers"
    );
    let (legacy_args, legacy_logits) = run_steps(&mut ad_legacy);
    drop(ad_legacy);

    std::env::set_var(W4A8_ENV, "1");
    let zero = synth_adapter(&cfg, 0xf01d_0000, Fill::ZeroB, None);
    let zdir = write_adapter("w48foldzero", &zero);
    let zlora = Q3dLora::from_peft_dir(&zdir.0, &cfg).expect("build zero lora sites");
    let ad_zero = Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), &hw, 32, Some(&zlora));
    std::env::remove_var(W4A8_ENV);
    let mut ad_zero = ad_zero.expect("adapter zero build");
    let (zero_args, zero_logits) = run_steps(&mut ad_zero);
    drop(ad_zero);
    assert_eq!(plain_args, zero_args, "zero-B adapter argmax drifted on the w4a8 fold route");
    for (i, (a, b)) in plain_logits.iter().zip(zero_logits.iter()).enumerate() {
        for (j, (p, q)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                p.to_bits() == q.to_bits(),
                "step {i} logit {j}: a zero-B adapter must leave the folded w4a8 graph \
                 bit-identical ({p} vs {q})"
            );
        }
    }

    let mut shift = 0f32;
    for (a, b) in plain_logits.iter().zip(fold_logits.iter()) {
        for (p, q) in a.iter().zip(b.iter()) {
            shift = shift.max((p - q).abs());
        }
    }
    assert!(
        shift > 1e-3,
        "a coarse adapter left the folded logits within {shift} of base; the folded sites are \
         wired but inert"
    );

    for (i, ((fa, fl), (na, nl))) in fold_args
        .iter()
        .zip(fold_logits.iter())
        .zip(nofold_args.iter().zip(nofold_logits.iter()))
        .enumerate()
    {
        assert_eq!(fa, na, "step {i}: argmax diverged between the fold and two-pass arms");
        let (_, rel) = rel_err(fl, nl);
        assert!(
            rel < ARM_VS_ARM_FOLD_REL_TOL_COVERS_ONE_PRE_VS_POST_ENCODE_ROUNDING_PER_SITE,
            "step {i}: fold logits drifted rel {rel} from the two-pass site arm; the fold is \
             an order-only change of the same delta"
        );
    }
    for (i, (fa, la)) in fold_args.iter().zip(legacy_args.iter()).enumerate() {
        let (_, rel) = rel_err(&fold_logits[i], &legacy_logits[i]);
        eprintln!(
            "step {i}: fold_arg={fa} legacy_arg={la} fold_vs_legacy rel={rel:.6}"
        );
    }
}

fn synth_fp8_lin(
    rng: &mut Lcg,
    n: usize,
    k: usize,
) -> (q3d::HostFp8Lin, nv_models::qwen3_5_moe_wgpu::HostBf16Lin) {
    let packed = synth_fp8_words(rng, n, k);
    let scales = synth_fp8_scales(rng, n);
    let mut w = vec![0u16; n * k];
    for (e, wi) in w.iter_mut().enumerate() {
        let byte = ((packed[e / 4] >> (8 * (e % 4))) & 0xff) as u8;
        *wi = bf16::from_f32(e4m3_decode(byte) * scales[e / k]).to_bits();
    }
    (
        q3d::HostFp8Lin {
            packed,
            scales,
            n,
            k,
        },
        nv_models::qwen3_5_moe_wgpu::HostBf16Lin { w, n, k },
    )
}

const TINY_FP8_N_VH_8_KEEPS_THE_AB_TWIN_16_ALIGNED_FOR_THE_COOP_PREFILL_THE_FP8_WEIGHTS_ENGAGE:
    usize = 8;

fn tiny_fp8_attn_dn_setup() -> (q3d::Qwen3_5DenseConfig, q3d::HostDenseWeights) {
    let mut cfg = tiny_config();
    cfg.intermediate_size = TINY_INTER_192_IS_THE_SMALLEST_DIM_THE_MROW2_BLOCK_PAIR_ROUTE_ADMITS;
    cfg.linear_num_value_heads =
        TINY_FP8_N_VH_8_KEEPS_THE_AB_TWIN_16_ALIGNED_FOR_THE_COOP_PREFILL_THE_FP8_WEIGHTS_ENGAGE;
    let mut hw = tiny_weights(&cfg, 0xf8a7_7d00_0050);
    let mut r = common::LcgOddSeedShift33SignedUnit::new(0xf8a7_7d00_0051);
    let mut f8 = Lcg(0xf8a7_7d00_0052);
    let (hidden, inter) = (cfg.hidden_size, cfg.intermediate_size);
    let q_out = cfg.num_attention_heads * cfg.head_dim * 2;
    let kv_out = cfg.num_key_value_heads * cfg.head_dim;
    let o_k = cfg.num_attention_heads * cfg.head_dim;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    for layer in &mut hw.layers {
        layer.mlp.gate = common::nvfp4_dense_lin(common::bf16_lin(&mut r, inter, hidden, 0.25));
        layer.mlp.up = common::nvfp4_dense_lin(common::bf16_lin(&mut r, inter, hidden, 0.25));
        layer.mlp.down = common::nvfp4_dense_lin(common::bf16_lin(&mut r, hidden, inter, 0.25));
        match &mut layer.mixer {
            q3d::HostDenseMixer::Attn(a) => {
                let (fq, tq) = synth_fp8_lin(&mut f8, q_out, hidden);
                a.q = q3d::HostDenseLin::Fp8 { fp8: fq, bf16: tq };
                let (fk, tk) = synth_fp8_lin(&mut f8, kv_out, hidden);
                a.k = q3d::HostDenseLin::Fp8 { fp8: fk, bf16: tk };
                let (fv, tv) = synth_fp8_lin(&mut f8, kv_out, hidden);
                a.v = q3d::HostDenseLin::Fp8 { fp8: fv, bf16: tv };
                let (fo, to) = synth_fp8_lin(&mut f8, hidden, o_k);
                a.o = q3d::HostDenseLin::Fp8 { fp8: fo, bf16: to };
            }
            q3d::HostDenseMixer::Delta(d) => {
                let (fqkv, tqkv) = synth_fp8_lin(&mut f8, conv_dim, hidden);
                d.in_proj_qkv = tqkv;
                let (fz, tz) = synth_fp8_lin(&mut f8, value_dim, hidden);
                d.in_proj_z = tz;
                let (fo, to) = synth_fp8_lin(&mut f8, hidden, value_dim);
                d.out_proj = to;
                layer.delta_fp8 = q3d::DeltaFp8 {
                    qkv: Some(fqkv),
                    z: Some(fz),
                    out: Some(fo),
                };
            }
        }
    }
    (cfg, hw)
}

const ATTN_DN_ENV: &str =
    q3d::LORA_GEMV_FOLD_ATTN_DN_ENV_OPT_IN_MEASURED_PLUS_0_9MS_AT_D256_BECAUSE_2ROW_FP8_WORKGROUPS_RESTAGE_H_4X_DENSER_THAN_W4A8_AND_THE_SHRINK_BECOMES_A_CRITICAL_PATH_PREDECESSOR_OF_THE_WIDEST_GEMVS;

#[test]
fn the_fp8_attn_and_dn_folds_collapse_every_site_to_its_shrink_and_keep_zero_b_bit_identity() {
    if !have_gpu() {
        return;
    }
    for var in [W4A8_ENV, FOLD_ENV, ATTN_DN_ENV] {
        assert!(
            std::env::var(var).is_err(),
            "{var} preset in the environment; this test owns that env var"
        );
    }
    let (cfg, hw) = tiny_fp8_attn_dn_setup();
    let layers = cfg.num_hidden_layers;
    let attn_layers = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, nv_models::qwen3_5_moe::LayerType::FullAttention))
        .count();
    let dn_layers = layers - attn_layers;
    assert!(attn_layers > 0 && dn_layers > 0, "fixture must exercise both mixer folds");

    std::env::set_var(W4A8_ENV, "1");
    let plain = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32);
    std::env::remove_var(W4A8_ENV);
    let mut plain = plain.expect("plain fp8 build");
    let p0 = plain.pass_count();
    let pf0 = plain.prefill_pass_count();
    assert!(pf0 > 0, "this gate needs the chunked prefill graph");
    let (plain_args, plain_logits) = run_steps(&mut plain);
    drop(plain);

    let synth = synth_adapter(&cfg, 0xf8a7_c0a5, Fill::Coarse, None);
    let dir = write_adapter("fp8fold", &synth);
    let lora = Q3dLora::from_peft_dir(&dir.0, &cfg).expect("build lora sites");
    assert_eq!(
        lora.total_pass_count(),
        8 * layers,
        "every layer carries a mixer site pair plus the two MLP sites at two legacy passes each"
    );

    std::env::set_var(W4A8_ENV, "1");
    let ad_default = Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), &hw, 32, Some(&lora));
    std::env::remove_var(W4A8_ENV);
    let ad_default = ad_default.expect("adapter default build");
    assert_eq!(
        ad_default.pass_count(),
        p0 + 6 * layers,
        "the default fold scope is the MLP w4a8 lever alone: gate_up/down shrink (2) plus \
         two-pass attn qkv/o or dn_in/dn_out (4); the attn/DN folds stay behind {ATTN_DN_ENV}"
    );
    drop(ad_default);

    std::env::set_var(W4A8_ENV, "1");
    std::env::set_var(ATTN_DN_ENV, "1");
    let ad_fold = Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), &hw, 32, Some(&lora));
    std::env::remove_var(ATTN_DN_ENV);
    std::env::remove_var(W4A8_ENV);
    let mut ad_fold = ad_fold.expect("adapter fold build");
    assert_eq!(
        ad_fold.pass_count(),
        p0 + 4 * layers,
        "every site must collapse to its shrink: attn qkv/o on the fp8 rowscale entries, dn_in \
         on the merged DN gemv, dn_out on the fp8 rowscale entry, gate_up/down on the w4a8 route"
    );
    assert_eq!(
        ad_fold.prefill_pass_count(),
        pf0 + lora.total_pass_count(),
        "prefill sites stay on the two-pass emitter"
    );
    let (fold_args, fold_logits) = run_steps(&mut ad_fold);
    drop(ad_fold);

    std::env::set_var(W4A8_ENV, "1");
    std::env::set_var(FOLD_ENV, "0");
    let ad_nofold = Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), &hw, 32, Some(&lora));
    std::env::remove_var(FOLD_ENV);
    std::env::remove_var(W4A8_ENV);
    let mut ad_nofold = ad_nofold.expect("adapter nofold build");
    assert_eq!(
        ad_nofold.pass_count(),
        p0 + 8 * layers,
        "{FOLD_ENV}=0 must restore the two-pass emitter on every fp8 and w4a8 site"
    );
    let (nofold_args, nofold_logits) = run_steps(&mut ad_nofold);
    drop(ad_nofold);

    std::env::set_var(W4A8_ENV, "1");
    std::env::set_var(ATTN_DN_ENV, "1");
    let zero = synth_adapter(&cfg, 0xf8a7_0000, Fill::ZeroB, None);
    let zdir = write_adapter("fp8foldzero", &zero);
    let zlora = Q3dLora::from_peft_dir(&zdir.0, &cfg).expect("build zero lora sites");
    let ad_zero = Qwen3_5DenseWgpu::new_with_lora(cfg.clone(), &hw, 32, Some(&zlora));
    std::env::remove_var(ATTN_DN_ENV);
    std::env::remove_var(W4A8_ENV);
    let mut ad_zero = ad_zero.expect("adapter zero build");
    let (zero_args, zero_logits) = run_steps(&mut ad_zero);
    drop(ad_zero);
    assert_eq!(
        plain_args, zero_args,
        "zero-B adapter argmax drifted on the fp8 fold route"
    );
    for (i, (a, b)) in plain_logits.iter().zip(zero_logits.iter()).enumerate() {
        for (j, (p, q)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                p.to_bits() == q.to_bits(),
                "step {i} logit {j}: a zero-B adapter must leave the folded fp8 graph \
                 bit-identical ({p} vs {q})"
            );
        }
    }

    let mut shift = 0f32;
    for (a, b) in plain_logits.iter().zip(fold_logits.iter()) {
        for (p, q) in a.iter().zip(b.iter()) {
            shift = shift.max((p - q).abs());
        }
    }
    assert!(
        shift > 1e-3,
        "a coarse adapter left the folded logits within {shift} of base; the folded sites are \
         wired but inert"
    );

    for (i, ((fa, fl), (na, nl))) in fold_args
        .iter()
        .zip(fold_logits.iter())
        .zip(nofold_args.iter().zip(nofold_logits.iter()))
        .enumerate()
        .take(ARM_VS_ARM_FOLD_STEPS_2_THE_DN_RECURRENCE_AMPLIFIES_ONE_ULP_REENCODE_DRIFT_CHAOTICALLY_WITH_DEPTH)
    {
        assert_eq!(fa, na, "step {i}: argmax diverged between the fold and two-pass arms");
        let (_, rel) = rel_err(fl, nl);
        assert!(
            rel < ARM_VS_ARM_FOLD_REL_TOL_COVERS_ONE_PRE_VS_POST_ENCODE_ROUNDING_PER_SITE,
            "step {i}: fold logits drifted rel {rel} from the two-pass site arm; the fold is \
             an order-only change of the same delta"
        );
    }
}

const ARM_VS_ARM_FOLD_STEPS_2_THE_DN_RECURRENCE_AMPLIFIES_ONE_ULP_REENCODE_DRIFT_CHAOTICALLY_WITH_DEPTH:
    usize = 2;
