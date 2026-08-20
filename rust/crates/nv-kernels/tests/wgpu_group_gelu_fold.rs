#![cfg(feature = "wgpu")]

mod common;
use common::ctx;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::quant_gemv::{bf16_to_f32, QFormat};
use nv_kernels::wgpu_backend::kernels::{gelu_tanh_mul, gemv_nvfp4, quant_gemv};

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / 8_388_608.0) - 1.0
    }
    fn bf16_vec(&mut self, len: usize, scale: f32) -> Vec<u16> {
        (0..len)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
            .collect()
    }
}

fn gelu_ref(gate: f32, up: f32) -> f64 {
    let g = gate as f64;
    let inner = 0.797_884_560_802_865_4 * (g + 0.044_715 * g * g * g);
    (0.5 * g * (1.0 + inner.tanh())) * up as f64
}

fn ref_decode_e4m3(code: u8) -> f64 {
    let mag = code & 0x7f;
    assert_ne!(mag, 0x7f, "quantizer emitted the e4m3 NaN code");
    let e = (mag >> 3) as i32;
    let m = (mag & 7) as f64;
    let v = if e == 0 {
        m * 2f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    if code & 0x80 != 0 {
        -v
    } else {
        v
    }
}

fn cpu_gemv_ref(
    wq: &[u32],
    scales: &[f32],
    x: &[u16],
    n: usize,
    k: usize,
    group: usize,
    fmt: QFormat,
) -> Vec<f64> {
    let g = if group == 0 { k } else { group };
    let per_row = k / g;
    (0..n)
        .map(|r| {
            let mut acc = 0f64;
            for gi in 0..per_row {
                let mut d = 0f64;
                for i in 0..g {
                    let idx = r * k + gi * g + i;
                    let byte = ((wq[idx / 4] >> (8 * (idx % 4))) & 0xff) as u8;
                    let v = match fmt {
                        QFormat::E4m3 => ref_decode_e4m3(byte),
                        QFormat::Int8 => (byte as i8) as f64,
                    };
                    d += v * bf16_to_f32(x[gi * g + i]) as f64;
                }
                acc += d * scales[r * per_row + gi] as f64;
            }
            acc
        })
        .collect()
}

struct Case {
    inter: usize,
    k: usize,
    group: usize,
}

const CASES: &[Case] = &[
    Case {
        inter: 512,
        k: 256,
        group: 16,
    },
    Case {
        inter: 512,
        k: 256,
        group: 0,
    },
    Case {
        inter: 1024,
        k: 512,
        group: 64,
    },
    Case {
        inter: 300,
        k: 256,
        group: 16,
    },
    Case {
        inter: 129,
        k: 128,
        group: 32,
    },
];

fn run_case(
    ctx: &WgpuContext,
    c: &Case,
    fmt: QFormat,
    fold: quant_gemv::GeluFold,
    seed: u64,
) -> (usize, f64) {
    let n = 2 * c.inter;
    let mut rng = Lcg(seed);
    let w = rng.bf16_vec(n * c.k, 1.0);
    let x = rng.bf16_vec(c.k, 1.0);
    let (wq, scales) = quant_gemv::quantize_groups(&w, n, c.k, c.group, fmt);

    let mut rows = vec![0u16; n];
    quant_gemv::gemv_group_bf16(ctx, &wq, &scales, &x, &mut rows, n, c.k, c.group, fmt)
        .expect("split gemv");

    let mut split = vec![0u16; c.inter];
    gelu_tanh_mul::gelu_tanh_mul_bf16(ctx, &rows[..c.inter], &rows[c.inter..], &mut split, c.inter)
        .expect("split gelu");

    let mut fused = vec![0u16; c.inter];
    quant_gemv::gemv_group_gelu_bf16(
        ctx, &wq, &scales, &x, &mut fused, n, c.k, c.group, fmt, fold,
    )
    .expect("fused gemv+gelu");

    let diff = split
        .iter()
        .zip(fused.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff,
        0,
        "inter={} k={} group={} fold={fold:?}: fused differs from gemv+gelu in {diff}/{} elements \
         (first mismatch at {:?})",
        c.inter,
        c.k,
        c.group,
        c.inter,
        split
            .iter()
            .zip(fused.iter())
            .position(|(a, b)| a != b)
            .map(|i| (i, split[i], fused[i]))
    );

    let nonzero = fused.iter().filter(|v| **v != 0).count();
    assert!(
        nonzero * 4 > c.inter * 3,
        "inter={} group={}: {nonzero}/{} outputs are zero -- the fixture measured nothing",
        c.inter,
        c.group,
        c.inter
    );

    let mut swapped = vec![0u16; c.inter];
    gelu_tanh_mul::gelu_tanh_mul_bf16(
        ctx,
        &rows[c.inter..],
        &rows[..c.inter],
        &mut swapped,
        c.inter,
    )
    .expect("swapped gelu");
    assert!(
        swapped.iter().zip(fused.iter()).any(|(a, b)| a != b),
        "inter={} group={}: gelu(gate)*up and gelu(up)*gate agree everywhere -- \
         the fixture cannot detect a transposed pairing",
        c.inter,
        c.group
    );

    let cpu = cpu_gemv_ref(&wq, &scales, &x, n, c.k, c.group, fmt);
    let mut worst = 0f64;
    for i in 0..c.inter {
        let want = gelu_ref(cpu[i] as f32, cpu[c.inter + i] as f32);
        let got = bf16_to_f32(fused[i]) as f64;
        let denom = want.abs().max(1e-3);
        worst = worst.max((got - want).abs() / denom);
    }
    (diff, worst)
}

#[test]
fn fused_gate_up_gelu_is_bit_exact_against_the_split_gemv_plus_gelu() {
    let Some(ctx) = ctx("int8_gelu_fold") else {
        return;
    };

    let sg_ok = gemv_nvfp4::sg32_ok(ctx);
    eprintln!(
        "int8_gelu_fold: sg32_ok={sg_ok} (strict subgroup_ok={}, probed width={:?})",
        gemv_nvfp4::subgroup_ok(ctx),
        ctx.subgroup_width()
    );
    let folds: &[quant_gemv::GeluFold] = if sg_ok {
        &[quant_gemv::GeluFold::Tree, quant_gemv::GeluFold::Subgroup]
    } else {
        &[quant_gemv::GeluFold::Tree]
    };
    let mut cells = 0;
    for fmt in [QFormat::Int8, QFormat::E4m3] {
        for c in CASES {
            for fold in folds {
                let (diff, worst) = run_case(
                    ctx,
                    c,
                    fmt,
                    *fold,
                    0xc0ffee ^ (c.inter as u64) << 8 ^ c.k as u64 ^ (fmt as u64) << 40,
                );
                eprintln!(
                    "{} gelu fold inter={:<5} k={:<5} group={:<3} fold={fold:?} | bit-exact ({diff} diffs) \
                     | worst rel vs f64 CPU {worst:.4}",
                    fmt.label(), c.inter, c.k, c.group
                );
                assert!(
                    worst < 0.08,
                    "fmt={} inter={} k={} group={} fold={fold:?}: worst relative error {worst} \
                     against the f64 CPU reference -- both GPU arms agree but on the wrong value",
                    fmt.label(),
                    c.inter,
                    c.k,
                    c.group
                );
                cells += 1;
            }
        }
    }
    let want = 2 * CASES.len() * folds.len();
    assert_eq!(cells, want, "expected {want} cells, ran {cells}");
}

struct Rig {
    pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    group: wgpu::BindGroup,
    grid: (u32, u32, u32),
}

fn bench(ctx: &WgpuContext, rig: &Rig, iters: usize) -> f64 {
    let submit = |count: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&rig.pipeline);
            pass.set_bind_group(0, &rig.group, &[]);
            for _ in 0..count {
                pass.dispatch_workgroups(rig.grid.0, rig.grid.1, rig.grid.2);
            }
        }
        ctx.queue.submit([enc.finish()]);
    };
    submit(8);
    ctx.poll_blocking().expect("warmup poll");
    let start = std::time::Instant::now();
    submit(iters);
    ctx.poll_blocking().expect("timed poll");
    start.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn fold_rate(ctx: &WgpuContext, fmt: QFormat) {
    let sg = gemv_nvfp4::subgroup_ok(ctx);
    let inter = 21504usize;
    let k = 5376usize;
    let group = 16usize;
    let n = 2 * inter;
    let mut rng = Lcg(0x9e3779b9);
    let w = rng.bf16_vec(n * k, 1.0);
    let x = rng.bf16_vec(k, 1.0);
    let (wq, scales) = quant_gemv::quantize_groups(&w, n, k, group, fmt);

    let bytes = (wq.len() + scales.len()) * 4;

    let src = quant_gemv::source();
    let rows_per_group = if sg {
        quant_gemv::SG_ROWS_PER_GROUP
    } else {
        quant_gemv::TREE_ROWS_PER_GROUP
    };
    let grid = dispatch::workgroup_count_1d(ctx, n as u64, rows_per_group);
    let params = quant_gemv::params_for(n, k, group, grid.0);
    let w_buf = dispatch::storage_from_slice(ctx, "gf-w", &wq);
    let s_buf = dispatch::storage_from_slice(ctx, "gf-s", &scales);
    let x_buf = dispatch::storage_from_slice(ctx, "gf-x", &quant_gemv::pack_x_bf16(&x));
    let y_buf = dispatch::storage_zeroed(ctx, "gf-y", (n * 4) as u64);
    let p_buf = dispatch::uniform_from(ctx, "gf-p", &params);

    let mk = |label: &str, entry: &str| {
        let pipeline =
            dispatch::cached_compute_pipeline(ctx, label, &src, entry).expect("pipeline");
        let group = dispatch::bind_group(
            ctx,
            &pipeline,
            &[
                (0, &w_buf),
                (1, &s_buf),
                (2, &x_buf),
                (3, &y_buf),
                (4, &p_buf),
            ],
        );
        Rig {
            pipeline,
            group,
            grid,
        }
    };
    let plain_entry = match (sg, fmt) {
        (true, QFormat::Int8) => quant_gemv::INT8_GROUP_SG_ENTRY,
        (false, QFormat::Int8) => quant_gemv::INT8_GROUP_ENTRY,
        (true, QFormat::E4m3) => quant_gemv::FP8_GROUP_SG_ENTRY,
        (false, QFormat::E4m3) => quant_gemv::FP8_GROUP_ENTRY,
    };
    let plain = mk("gf-plain", plain_entry);

    let null = mk("gf-null", plain_entry);
    let fold = if sg {
        quant_gemv::GeluFold::Subgroup
    } else {
        quant_gemv::GeluFold::Tree
    };
    let fused = mk("gf-fused", fold.entry(fmt));

    let gsrc = gelu_tanh_mul::source();
    let gpipe =
        dispatch::cached_compute_pipeline(ctx, "gf-gelu", &gsrc, gelu_tanh_mul::ENTRY_FUSED_EVEN)
            .expect("gelu pipeline");

    let gsrc_words = quant_gemv::pack_x_bf16(&rng.bf16_vec(n, 1.0));
    let gsrc_buf = dispatch::storage_from_slice(ctx, "gf-gsrc", &gsrc_words);
    let gy_buf = dispatch::storage_zeroed(ctx, "gf-gy", (inter / 2 * 4) as u64);
    let gp_buf = dispatch::uniform_from(
        ctx,
        "gf-gp",
        &gelu_tanh_mul::GeluParams {
            inter: inter as u32,
            inter_words: (inter / 2) as u32,
            rows: 1,
            tot_pairs: inter as u32,
        },
    );
    let ggrid =
        dispatch::workgroup_count_1d(ctx, (inter / 2) as u64, gelu_tanh_mul::WORKGROUP_SIZE);
    let gelu = Rig {
        group: dispatch::bind_group(ctx, &gpipe, &[(3, &gsrc_buf), (4, &gy_buf), (5, &gp_buf)]),
        pipeline: gpipe,
        grid: (ggrid.0, 1, 1),
    };

    let iters = 200;
    let mut best = [f64::MAX; 4];
    for _ in 0..9 {
        for (i, rig) in [&plain, &fused, &gelu, &null].iter().enumerate() {
            let us = bench(ctx, rig, iters);
            if us < best[i] {
                best[i] = us;
            }
        }
    }
    let null_us = (best[3] - best[0]).abs();
    eprintln!(
        "{} gate_up fold n={n} k={k} group={group} sg={sg}\n  \
         gemv             {:9.3} us  ({:.1} GB/s over {:.1} MB)\n  \
         gemv+gelu fused  {:9.3} us  (fold cost {:+.3}, net {:+.3} us/layer)\n  \
         standalone gelu  {:9.3} us\n  \
         null control     {:9.3} us  (|delta| {:.3} us, {:.2}%)",
        fmt.label(),
        best[0],
        bytes as f64 / best[0] * 1e-3,
        bytes as f64 / 1e6,
        best[1],
        best[1] - best[0],
        (best[1] - best[0]) - best[2],
        best[2],
        best[3],
        null_us,
        null_us / best[0] * 100.0,
    );
    assert!(best.iter().all(|v| *v > 0.0));
}

#[test]
#[ignore = "kernel-rate suite: run alone, one per process"]
fn fusing_the_gelu_pass_costs_less_than_the_dispatch_it_deletes() {
    let Some(ctx) = ctx("int8_gelu_fold_rate") else {
        return;
    };
    fold_rate(ctx, QFormat::Int8);
}

#[test]
#[ignore = "kernel-rate suite: run alone, one per process"]
fn fusing_the_gelu_pass_costs_less_than_the_dispatch_it_deletes_on_e4m3() {
    let Some(ctx) = ctx("fp8_gelu_fold_rate") else {
        return;
    };
    fold_rate(ctx, QFormat::E4m3);
}
