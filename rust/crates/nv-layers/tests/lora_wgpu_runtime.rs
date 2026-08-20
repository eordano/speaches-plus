#![cfg(feature = "wgpu")]

use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_kernels::wgpu_backend::WgpuContext;
use nv_layers::linear::Linear;
use nv_layers::lora_slots::{
    LoraAdapter, LoraModuleSpec, LoraModuleWeights, LoraSlotManager, WgpuLoraDispatch,
    WgpuLoraHook, WgpuLoraPath,
};
use std::collections::HashMap;
use std::sync::Arc;

#[path = "wgpu_common.rs"]
mod wgpu_common;

use wgpu_common::wgpu_ctx_or_skip as ctx_or_skip;

fn wgsl_bf16_encode(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let b = x.to_bits();
    let r = 0x7fffu32 + ((b >> 16) & 1);
    (b.wrapping_add(r) >> 16) as u16
}

fn wgsl_bf16_decode(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

const LADDER: [f32; 8] = [-1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0];

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E3779B97F4A7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn ladder(&mut self) -> f32 {
        LADDER[(self.next_u64() >> 33) as usize % LADDER.len()]
    }
}

fn exact_bf16(v: f32, what: &str) -> f32 {
    assert_eq!(
        bf16::from_f32(v).to_f32(),
        v,
        "{what}: {v} is not exactly representable in bf16; fixture is not an exact oracle"
    );
    v
}

fn ladder_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.ladder()).collect()
}

fn to_bf16_tensor(v: &[f32], rows: usize, cols: usize, dev: &Device) -> Tensor {
    let b: Vec<bf16> = v.iter().map(|x| bf16::from_f32(*x)).collect();
    Tensor::from_vec(b, (rows, cols), dev).unwrap()
}

fn bits_of(t: &Tensor) -> Vec<u16> {
    t.flatten_all()
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_vec1::<bf16>()
        .unwrap()
        .into_iter()
        .map(|v| v.to_bits())
        .collect()
}

#[derive(Clone)]
struct Adapter {
    rank: usize,
    k: usize,
    n: usize,
    a: Vec<f32>,
    b_folded: Vec<f32>,
}

fn make_adapter(seed: u64, rank: usize, k: usize, n: usize, scale: f32) -> Adapter {
    let a = ladder_vec(seed ^ 0xA, rank * k);
    let b_unscaled = ladder_vec(seed ^ 0xB, n * rank);
    let b_folded: Vec<f32> = b_unscaled
        .iter()
        .map(|v| exact_bf16(v * scale, "scaling folded into B"))
        .collect();
    Adapter {
        rank,
        k,
        n,
        a,
        b_folded,
    }
}

fn delta_row(ad: &Adapter, x: &[f32]) -> Vec<f32> {
    let mut h32 = vec![0f32; ad.rank];
    let mut h64 = vec![0f64; ad.rank];
    for r in 0..ad.rank {
        let mut acc = 0f32;
        let mut acc64 = 0f64;
        for kk in 0..ad.k {
            let aw = ad.a[r * ad.k + kk];
            acc = x[kk].mul_add(aw, acc);
            acc64 += (x[kk] as f64) * (aw as f64);
        }
        assert_eq!(
            acc as f64, acc64,
            "shrink accumulation left the exact f32 grid; tighten the fixture magnitudes"
        );
        h32[r] = acc;
        h64[r] = acc64;
    }
    let mut d = vec![0f32; ad.n];
    for nn in 0..ad.n {
        let mut acc = 0f32;
        let mut acc64 = 0f64;
        for r in 0..ad.rank {
            let bw = ad.b_folded[nn * ad.rank + r];
            acc = h32[r].mul_add(bw, acc);
            acc64 += h64[r] * (bw as f64);
        }
        assert_eq!(
            acc as f64, acc64,
            "expand accumulation left the exact f32 grid; tighten the fixture magnitudes"
        );
        d[nn] = acc;
    }
    d
}

struct Fixture {
    device: Device,
    dispatch: Arc<WgpuLoraDispatch>,
    hook: Arc<WgpuLoraHook>,
    mgr: LoraSlotManager,
    adapters: Vec<Vec<Option<Adapter>>>,
    widths: Vec<usize>,
    slice_start: Vec<usize>,
    names: Vec<String>,
    k: usize,
    out_features: usize,
}

fn slice_names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("s{i}")).collect()
}

fn build_fixture(
    ctx: &'static WgpuContext,
    k: usize,
    widths: &[usize],
    max_rank: usize,
    slot_ranks: &[Option<usize>],
    scale: f32,
    max_tokens: usize,
    seed: u64,
) -> Fixture {
    let device = Device::Cpu;
    let names = slice_names(widths.len());
    let specs: Vec<LoraModuleSpec> = names
        .iter()
        .zip(widths)
        .map(|(n, &w)| LoraModuleSpec::new(n.clone(), k, w))
        .collect();
    let max_loras = slot_ranks.len();
    let mut mgr = LoraSlotManager::new(max_loras, max_rank, &specs, DType::BF16, &device).unwrap();

    let mut adapters: Vec<Vec<Option<Adapter>>> = vec![Vec::new(); max_loras];
    for (s, rank) in slot_ranks.iter().enumerate() {
        let mut modules = HashMap::new();
        for (i, (name, &w)) in names.iter().zip(widths).enumerate() {
            match rank {
                Some(r) => {
                    let ad = make_adapter(seed ^ ((s as u64) << 16) ^ (i as u64), *r, k, w, scale);
                    modules.insert(
                        name.clone(),
                        LoraModuleWeights {
                            a: to_bf16_tensor(&ad.a, *r, k, &device),
                            b: to_bf16_tensor(&ad.b_folded, w, *r, &device),
                        },
                    );
                    adapters[s].push(Some(ad));
                }
                None => adapters[s].push(None),
            }
        }
        let adapter = LoraAdapter {
            scaling: 1.0,
            modules,
        };
        let got = mgr.activate(100 + s as u64, &adapter).unwrap();
        assert_eq!(got, s, "slot allocation order");
    }

    let dispatch = WgpuLoraDispatch::with_context(ctx, &device, max_tokens, max_loras).unwrap();
    let stacks: Vec<&_> = names.iter().map(|n| mgr.stack(n).unwrap()).collect();
    let hook = WgpuLoraHook::from_stacks(dispatch.clone(), &stacks).unwrap();
    drop(stacks);

    let mut acc = 0usize;
    let slice_start: Vec<usize> = widths
        .iter()
        .map(|&w| {
            let s = acc;
            acc += w;
            s
        })
        .collect();

    Fixture {
        device,
        dispatch,
        hook,
        mgr,
        adapters,
        widths: widths.to_vec(),
        slice_start,
        names,
        k,
        out_features: acc,
    }
}

impl Fixture {
    fn oracle(
        &self,
        mapping: &[i32],
        x: &[f32],
        y_base_bits: &[u16],
        win: Option<(usize, usize)>,
    ) -> Vec<u16> {
        let (win_off, win_len) = win.unwrap_or((0, self.out_features));
        let m = mapping.len();
        assert_eq!(y_base_bits.len(), m * win_len);
        let mut acc: Vec<f32> = y_base_bits.iter().map(|&b| wgsl_bf16_decode(b)).collect();
        for (tok, &slot) in mapping.iter().enumerate() {
            if slot < 0 {
                continue;
            }
            let xr = &x[tok * self.k..(tok + 1) * self.k];
            for (s, &w) in self.widths.iter().enumerate() {
                let Some(ad) = &self.adapters[slot as usize][s] else {
                    continue;
                };
                let d = delta_row(ad, xr);
                for j in 0..w {
                    let col = self.slice_start[s] + j;
                    if col < win_off || col >= win_off + win_len {
                        continue;
                    }
                    acc[tok * win_len + (col - win_off)] += d[j];
                }
            }
        }
        acc.iter().map(|&v| wgsl_bf16_encode(v)).collect()
    }

    fn run(
        &self,
        mapping: &[i32],
        x: &[f32],
        y_base_bits: &[u16],
        win: Option<(usize, usize)>,
    ) -> Vec<u16> {
        let (_, win_len) = win.unwrap_or((0, self.out_features));
        let m = mapping.len();
        let x_t = to_bf16_tensor(x, m, self.k, &self.device);
        let y_vals: Vec<bf16> = y_base_bits.iter().map(|&b| bf16::from_bits(b)).collect();
        let y_t = Tensor::from_vec(y_vals, (m, win_len), &self.device).unwrap();
        self.dispatch.set_mapping(mapping).unwrap();
        self.hook.apply(&x_t, &y_t, win).unwrap();
        bits_of(&y_t)
    }
}

fn base_bits(seed: u64, n: usize) -> Vec<u16> {
    ladder_vec(seed, n)
        .iter()
        .map(|&v| bf16::from_f32(v).to_bits())
        .collect()
}

fn assert_bitwise(got: &[u16], want: &[u16], tag: &str) {
    assert_eq!(got.len(), want.len(), "{tag}: length");
    let mut bad = 0usize;
    let mut first = None;
    let mut max_abs = 0f32;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let e = (wgsl_bf16_decode(g) - wgsl_bf16_decode(w)).abs();
        if e > max_abs {
            max_abs = e;
        }
        if g != w {
            bad += 1;
            if first.is_none() {
                first = Some((i, g, w));
            }
        }
    }
    if bad != 0 {
        let (i, g, w) = first.unwrap();
        panic!(
            "{tag}: {bad}/{} elements differ; first at {i}: got 0x{g:04x} ({}) want 0x{w:04x} ({}), max abs err {max_abs}",
            got.len(),
            wgsl_bf16_decode(g),
            wgsl_bf16_decode(w)
        );
    }
    eprintln!(
        "{tag}: {} elements bitwise-equal (max abs err {max_abs})",
        got.len()
    );
}

fn differs(got: &[u16], want: &[u16]) -> usize {
    got.iter().zip(want.iter()).filter(|(a, b)| a != b).count()
}

#[test]
fn rank1_alpha4_decode_m1_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("rank1_alpha4_decode_m1") else {
        return;
    };
    let (k, w, m) = (32usize, 24usize, 1usize);
    let fix = build_fixture(ctx, k, &[w], 1, &[Some(1)], 4.0, 8, 0x11);
    assert_eq!(fix.hook.plan(m, None).unwrap(), WgpuLoraPath::Fused);
    let x = ladder_vec(0x1a, m * k);
    let y0 = base_bits(0x1b, m * w);
    let got = fix.run(&[0], &x, &y0, None);
    let want = fix.oracle(&[0], &x, &y0, None);
    assert!(differs(&got, &y0) > 0, "rank-1 delta must move the output");
    assert_bitwise(&got, &want, "rank1_alpha4_decode_m1");
}

#[test]
fn rank16_alpha32_prefill_fused_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("rank16_alpha32_prefill") else {
        return;
    };
    let (k, w, m) = (64usize, 32usize, 8usize);
    let fix = build_fixture(ctx, k, &[w], 16, &[Some(16)], 2.0, 128, 0x21);
    assert_eq!(fix.hook.plan(m, None).unwrap(), WgpuLoraPath::Fused);
    let x = ladder_vec(0x2a, m * k);
    let y0 = base_bits(0x2b, m * w);
    let got = fix.run(&vec![0i32; m], &x, &y0, None);
    let want = fix.oracle(&vec![0i32; m], &x, &y0, None);
    assert_bitwise(&got, &want, "rank16_alpha32_prefill_fused");
}

#[test]
fn rank32_alpha8_large_batch_takes_grouped_path() {
    let Some(ctx) = ctx_or_skip("rank32_alpha8_grouped") else {
        return;
    };
    let (k, w, m) = (64usize, 48usize, 96usize);
    let fix = build_fixture(ctx, k, &[w], 32, &[Some(32)], 0.25, 256, 0x31);
    assert_eq!(
        fix.hook.plan(m, None).unwrap(),
        WgpuLoraPath::Grouped,
        "m>64 full-window must route to shrink+expand"
    );
    let x = ladder_vec(0x3a, m * k);
    let y0 = base_bits(0x3b, m * w);
    let got = fix.run(&vec![0i32; m], &x, &y0, None);
    let want = fix.oracle(&vec![0i32; m], &x, &y0, None);
    assert_bitwise(&got, &want, "rank32_alpha8_grouped_m96");
}

#[test]
fn multi_slot_token_mapping_selects_per_token_adapter() {
    let Some(ctx) = ctx_or_skip("multi_slot_mapping") else {
        return;
    };
    let (k, w) = (48usize, 24usize);
    let fix = build_fixture(
        ctx,
        k,
        &[w],
        16,
        &[Some(1), Some(8), Some(16)],
        2.0,
        256,
        0x41,
    );
    for (m, want_path) in [(13usize, WgpuLoraPath::Fused), (80, WgpuLoraPath::Grouped)] {
        assert_eq!(fix.hook.plan(m, None).unwrap(), want_path, "path for m={m}");
        let mapping: Vec<i32> = (0..m)
            .map(|i| match i % 4 {
                0 => 0,
                1 => -1,
                2 => 1,
                _ => 2,
            })
            .collect();
        let x = ladder_vec(0x4a ^ m as u64, m * k);
        let y0 = base_bits(0x4b ^ m as u64, m * w);
        let got = fix.run(&mapping, &x, &y0, None);
        let want = fix.oracle(&mapping, &x, &y0, None);
        assert_bitwise(&got, &want, &format!("multi_slot_m{m}"));
        for (tok, &slot) in mapping.iter().enumerate() {
            if slot == -1 {
                assert_eq!(
                    &got[tok * w..(tok + 1) * w],
                    &y0[tok * w..(tok + 1) * w],
                    "m={m}: -1 token row {tok} must be bitwise untouched"
                );
            }
        }
        let with = (0..m)
            .filter(|&t| mapping[t] != -1)
            .any(|t| got[t * w..(t + 1) * w] != y0[t * w..(t + 1) * w]);
        assert!(with, "m={m}: adapted rows must actually move");
    }
}

#[test]
fn no_adapter_batch_and_disarm_are_bit_identical_to_base() {
    let Some(ctx) = ctx_or_skip("no_adapter_batch") else {
        return;
    };
    let (k, w, m) = (32usize, 16usize, 6usize);
    let fix = build_fixture(ctx, k, &[w], 8, &[Some(8)], 2.0, 64, 0x51);
    let x = ladder_vec(0x5a, m * k);
    let y0 = base_bits(0x5b, m * w);

    let all_none = vec![-1i32; m];
    let got = fix.run(&all_none, &x, &y0, None);
    assert!(
        !fix.dispatch.armed(),
        "an all -1 mapping must report armed() == false"
    );
    assert_bitwise(&got, &y0, "all_minus_one_mapping_is_base");

    fix.dispatch.set_mapping(&vec![0i32; m]).unwrap();
    assert!(fix.dispatch.armed());
    fix.dispatch.disarm();
    assert!(!fix.dispatch.armed());
    let x_t = to_bf16_tensor(&x, m, k, &fix.device);
    let y_vals: Vec<bf16> = y0.iter().map(|&b| bf16::from_bits(b)).collect();
    let y_t = Tensor::from_vec(y_vals, (m, w), &fix.device).unwrap();
    fix.hook.apply(&x_t, &y_t, None).unwrap();
    assert_bitwise(&bits_of(&y_t), &y0, "disarmed_apply_is_base");
}

#[test]
fn unequal_qkv_slice_widths_and_row_windows() {
    let Some(ctx) = ctx_or_skip("unequal_qkv_windows") else {
        return;
    };
    let (k, m) = (48usize, 5usize);
    let widths = [24usize, 8, 8];
    let fix = build_fixture(ctx, k, &widths, 8, &[Some(8), Some(4)], 2.0, 64, 0x61);
    let n_total: usize = widths.iter().sum();
    let mapping: Vec<i32> = (0..m).map(|i| (i % 3) as i32 - 1).collect();
    let x = ladder_vec(0x6a, m * k);
    let y_full = base_bits(0x6b, m * n_total);

    let got = fix.run(&mapping, &x, &y_full, None);
    let want = fix.oracle(&mapping, &x, &y_full, None);
    assert_bitwise(&got, &want, "qkv_full_row");

    for (s, (&off, &len)) in fix.slice_start.iter().zip(widths.iter()).enumerate() {
        let y_win: Vec<u16> = (0..m)
            .flat_map(|t| y_full[t * n_total + off..t * n_total + off + len].to_vec())
            .collect();
        let win = Some((off, len));
        assert_eq!(fix.hook.plan(m, win).unwrap(), WgpuLoraPath::Fused);
        let g = fix.run(&mapping, &x, &y_win, win);
        let wnt = fix.oracle(&mapping, &x, &y_win, win);
        assert_bitwise(&g, &wnt, &format!("qkv_window_slice{s}"));
        let full_slice: Vec<u16> = (0..m)
            .flat_map(|t| want[t * n_total + off..t * n_total + off + len].to_vec())
            .collect();
        assert_eq!(
            g, full_slice,
            "window slice{s} must equal the same columns of the full-row result"
        );
    }
}

#[test]
fn falsification_single_weight_perturbation_breaks_the_gate() {
    let Some(ctx) = ctx_or_skip("falsification") else {
        return;
    };
    let (k, w, m) = (32usize, 16usize, 4usize);
    let mut fix = build_fixture(ctx, k, &[w], 8, &[Some(8)], 2.0, 64, 0x71);
    let x = ladder_vec(0x7a, m * k);
    let y0 = base_bits(0x7b, m * w);
    let mapping = vec![0i32; m];

    let got = fix.run(&mapping, &x, &y0, None);
    let want = fix.oracle(&mapping, &x, &y0, None);
    assert_bitwise(&got, &want, "falsification_baseline");

    let clean = fix.adapters[0][0].clone().unwrap();
    let mut a_only = clean.clone();
    a_only.a[7] = exact_bf16(a_only.a[7] + 2.0, "perturbed A entry");
    fix.adapters[0][0] = Some(a_only);
    let want_a = fix.oracle(&mapping, &x, &y0, None);
    let moved_a = differs(&want_a, &want);
    assert!(moved_a > 0, "perturbing A[7] must change the oracle");
    assert_ne!(
        got, want_a,
        "FALSIFICATION FAILED (shrink): oracle comparison passed against a perturbed A"
    );
    eprintln!("falsification: perturbing A[7] moved {moved_a} oracle elements; gate rejects it");

    let mut perturbed = clean;
    let idx = 3usize;
    perturbed.b_folded[idx] = exact_bf16(perturbed.b_folded[idx] + 4.0, "perturbed B entry");
    fix.adapters[0][0] = Some(perturbed.clone());
    let want_perturbed = fix.oracle(&mapping, &x, &y0, None);
    let n_moved = differs(&want_perturbed, &want);
    assert!(
        n_moved > 0,
        "the perturbation must change the oracle, otherwise the falsification proves nothing"
    );
    let got2 = fix.run(&mapping, &x, &y0, None);
    assert_eq!(
        got2, got,
        "device weights were not touched, so the device output must not move"
    );
    assert_ne!(
        got2, want_perturbed,
        "FALSIFICATION FAILED: the oracle comparison passed against perturbed weights"
    );
    eprintln!(
        "falsification: perturbing B[{idx}] moved {n_moved} oracle elements; gate rejects it"
    );

    let modules = HashMap::from([(
        fix.names[0].clone(),
        LoraModuleWeights {
            a: to_bf16_tensor(&perturbed.a, perturbed.rank, k, &fix.device),
            b: to_bf16_tensor(&perturbed.b_folded, w, perturbed.rank, &fix.device),
        },
    )]);
    assert_eq!(fix.mgr.deactivate(100), Some(0));
    assert_eq!(
        fix.mgr
            .activate(
                100,
                &LoraAdapter {
                    scaling: 1.0,
                    modules,
                },
            )
            .unwrap(),
        0
    );
    let got3 = fix.run(&mapping, &x, &y0, None);
    assert_bitwise(&got3, &want_perturbed, "reactivated_perturbed_slot");
}

#[test]
fn linear_forward_matches_oracle_and_disarm_restores_base_bitwise() {
    let Some(ctx) = ctx_or_skip("linear_forward") else {
        return;
    };
    let (k, w, m) = (32usize, 24usize, 6usize);
    let device = Device::Cpu;
    let weight = to_bf16_tensor(&ladder_vec(0x81, w * k), w, k, &device);
    let linear = Linear::new(weight, None).unwrap();

    let fix = build_fixture(ctx, k, &[w], 8, &[Some(8), Some(4)], 2.0, 64, 0x82);
    let x = ladder_vec(0x83, m * k);
    let x_t = to_bf16_tensor(&x, m, k, &device);

    let base = linear.forward(&x_t).unwrap();
    assert_eq!(base.dtype(), DType::BF16, "base output must be bf16");
    let base_bits = bits_of(&base);

    linear.attach_lora(fix.hook.clone()).unwrap();
    let disarmed = bits_of(&linear.forward(&x_t).unwrap());
    assert_bitwise(&disarmed, &base_bits, "linear_disarmed_equals_base");

    let mapping: Vec<i32> = (0..m).map(|i| (i % 3) as i32 - 1).collect();
    fix.dispatch.set_mapping(&mapping).unwrap();
    let got = bits_of(&linear.forward(&x_t).unwrap());
    let want = fix.oracle(&mapping, &x, &base_bits, None);
    assert_bitwise(&got, &want, "linear_forward_with_lora");

    let got_dense = bits_of(&linear.forward_dense(&x_t).unwrap());
    assert_bitwise(&got_dense, &want, "linear_forward_dense_with_lora");

    fix.dispatch.disarm();
    let after = bits_of(&linear.forward(&x_t).unwrap());
    assert_bitwise(&after, &base_bits, "linear_disarm_restores_base");
}

#[test]
fn mapping_length_mismatch_is_loud() {
    let Some(ctx) = ctx_or_skip("mapping_length_mismatch") else {
        return;
    };
    let (k, w) = (32usize, 16usize);
    let device = Device::Cpu;
    let weight = to_bf16_tensor(&ladder_vec(0x91, w * k), w, k, &device);
    let linear = Linear::new(weight, None).unwrap();
    let fix = build_fixture(ctx, k, &[w], 8, &[Some(8)], 2.0, 8, 0x92);
    linear.attach_lora(fix.hook.clone()).unwrap();

    fix.dispatch.set_mapping(&[0i32; 4]).unwrap();
    let x_t = to_bf16_tensor(&ladder_vec(0x93, 8 * k), 8, k, &device);
    let err = linear.forward(&x_t);
    assert!(
        err.is_err(),
        "mismatched mapping length must error, not silently skip"
    );
    let msg = format!("{:#}", err.unwrap_err());
    assert!(
        msg.contains("armed mapping length"),
        "unexpected error: {msg}"
    );

    assert!(
        fix.dispatch.set_mapping(&[0; 16]).is_err(),
        "overlong mapping must fail"
    );
    assert!(
        fix.dispatch.set_mapping(&[1]).is_err(),
        "out-of-range slot must fail"
    );
    assert!(
        fix.dispatch.set_mapping(&[]).is_err(),
        "empty mapping must fail"
    );
}

#[test]
fn grid_bound_policy_picks_a_fitting_kernel() {
    let Some(ctx) = ctx_or_skip("grid_bound_policy") else {
        return;
    };
    let limit = ctx.caps.max_compute_workgroups_per_dimension as usize;
    let fix = build_fixture(ctx, 16, &[512], 8, &[Some(8)], 2.0, 8, 0xA1);

    assert_eq!(fix.hook.plan(1, None).unwrap(), WgpuLoraPath::Fused);
    assert_eq!(fix.hook.plan(64, None).unwrap(), WgpuLoraPath::Fused);
    assert_eq!(fix.hook.plan(65, None).unwrap(), WgpuLoraPath::Grouped);
    assert_eq!(
        fix.hook.plan(65, Some((0, 256))).unwrap(),
        WgpuLoraPath::Fused,
        "a windowed call has no expand path and must stay fused"
    );

    let grouped_break = (limit / 32 + 1) * 16;
    assert!(
        grouped_break.div_ceil(16) * 32 > limit,
        "fixture must break the grouped grid"
    );
    assert!(
        grouped_break < limit,
        "fixture must still fit the fused grid"
    );
    assert_eq!(
        fix.hook.plan(grouped_break, None).unwrap(),
        WgpuLoraPath::Fused,
        "an oversized grouped grid must fall back to fused, not error"
    );

    let both_break = limit + 16;
    let err = fix.hook.plan(both_break, None).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("exceeds the wgpu grid bound"),
        "unexpected error: {msg}"
    );
}
