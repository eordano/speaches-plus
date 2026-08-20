#![cfg(feature = "wgpu")]

mod common;
use common::d;
use common::LcgShift33W4a16Packs as Lcg;
use common::OffParams;
use common::pack_u16;
use common::Params;
use common::tree_sum;
use common::widen_u16;
use half::bf16;
use nv_kernels::wgpu_backend::compose;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemv_w4a16 as gw;
use nv_kernels::wgpu_backend::kernels::gemv_w4a16::ScaleGrain;

fn ctx(test: &str) -> &'static WgpuContext {
    let ctx = WgpuContext::shared().unwrap_or_else(|e| panic!("{test}: no wgpu adapter: {e}"));
    eprintln!("{test}: {}", ctx.summary());
    let st = ctx.qualify();
    assert!(
        st.qualified,
        "{test}: adapter not qualified: {:?}",
        st.reason
    );
    ctx
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sched {
    Lane32Fma,

    Lane16Fma,

    Block256Pairwise,
}

struct Case<'a> {
    packed: &'a [u32],
    scales: &'a [u16],
    x: &'a [u16],
    k: usize,
    gs: usize,
}

impl Case<'_> {
    fn w(&self, row: usize, kk: usize) -> f32 {
        let word = self.packed[row * (self.k / 8) + kk / 8];
        let nib = (word >> (4 * (kk % 8))) & 0xf;
        let sc = d(self.scales[row * (self.k / self.gs) + kk / self.gs]);
        ((nib as i32 - 8) as f32) * sc
    }

    fn row_f64(&self, row: usize) -> f64 {
        (0..self.k)
            .map(|kk| self.w(row, kk) as f64 * d(self.x[kk]) as f64)
            .sum()
    }

    fn wide(&self) -> bool {
        self.gs.is_multiple_of(32)
    }

    fn scale_at(&self, row: usize, kk: usize) -> f32 {
        d(self.scales[row * (self.k / self.gs) + kk / self.gs])
    }

    fn dot8_fma(&self, row: usize, kb: usize, acc_in: f32) -> f32 {
        let mut acc = acc_in;
        let word = self.packed[row * (self.k / 8) + kb / 8];
        for i in 0..8 {
            let nib = (word >> (4 * i)) & 0xf;
            acc = ((nib as i32 - 8) as f32).mul_add(d(self.x[kb + i]), acc);
        }
        acc
    }

    fn dot8_pairwise(&self, row: usize, kb: usize) -> f32 {
        let word = self.packed[row * (self.k / 8) + kb / 8];
        let lvl = |i: usize| (((word >> (4 * i)) & 0xf) as i32 - 8) as f32;
        let mut a = 0f32;
        for i in 0..4 {
            a += lvl(2 * i) * d(self.x[kb + 2 * i]) + lvl(2 * i + 1) * d(self.x[kb + 2 * i + 1]);
        }
        a
    }

    fn lane_acc(&self, row: usize, lane: usize, lanes: usize, sched: Sched) -> f32 {
        let mut acc = 0f32;
        let mut v = lane;
        while v < self.k / 32 {
            let kbase = v * 32;
            match (sched, self.wide()) {
                (Sched::Block256Pairwise, true) => {
                    let sc = self.scale_at(row, kbase);
                    let mut block = 0f32;
                    for j in 0..4 {
                        block += self.dot8_pairwise(row, kbase + j * 8);
                    }
                    acc = sc.mul_add(block, acc);
                }
                (Sched::Block256Pairwise, false) => {
                    let mut block = 0f32;
                    for j in 0..4 {
                        let kb = kbase + j * 8;
                        block = self
                            .dot8_pairwise(row, kb)
                            .mul_add(self.scale_at(row, kb), block);
                    }
                    acc += block;
                }
                (_, true) => {
                    let sc = self.scale_at(row, kbase);
                    let mut a = 0f32;
                    for j in 0..4 {
                        a = self.dot8_fma(row, kbase + j * 8, a);
                    }
                    acc = sc.mul_add(a, acc);
                }
                (_, false) => {
                    for j in 0..4 {
                        let kb = kbase + j * 8;
                        let a = self.dot8_fma(row, kb, 0.0);
                        acc = a.mul_add(self.scale_at(row, kb), acc);
                    }
                }
            }
            v += lanes;
        }
        acc
    }

    fn row_exact(&self, row: usize, sched: Sched) -> f32 {
        match sched {
            Sched::Lane32Fma => {
                let parts: Vec<f32> = (0..32).map(|l| self.lane_acc(row, l, 32, sched)).collect();
                tree_sum(&parts)
            }
            Sched::Lane16Fma => {
                let parts: Vec<f32> = (0..16).map(|l| self.lane_acc(row, l, 16, sched)).collect();
                tree_sum(&parts)
            }
            Sched::Block256Pairwise => {
                let parts: Vec<f32> = (0..256)
                    .map(|t| self.lane_acc(row, t, 256, sched))
                    .collect();
                let warps: Vec<f32> = (0..8)
                    .map(|w| tree_sum(&parts[w * 32..(w + 1) * 32]))
                    .collect();
                tree_sum(&warps)
            }
        }
    }
}

fn run_entry(
    ctx: &WgpuContext,
    src: &str,
    entry: &str,
    case: &Case<'_>,
    n: usize,
    rows_per_group: u32,
    packed_y: bool,
) -> Vec<u16> {
    let k = case.k;
    let gs = case.gs;
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, rows_per_group);
    let params = Params {
        n_rows: n as u32,
        k_elems: k as u32,
        gs: gs as u32,
        w_row_words: (k / 8) as u32,
        scale_row_stride: (k / gs) as u32,
        groups_x: groups.0,
    };
    let p = dispatch::uniform_from(ctx, "gsw-p", &params);
    let w = dispatch::storage_from_slice(ctx, "gsw-w", case.packed);
    let scale_words = if packed_y {
        gw::pack_scale_words(case.scales)
    } else {
        widen_u16(case.scales)
    };
    let s = dispatch::storage_from_slice(ctx, "gsw-s", &scale_words);
    let xb = dispatch::storage_from_slice(ctx, "gsw-x", &pack_u16(case.x));
    let y_words = if packed_y { n / 2 } else { n };
    let y = dispatch::storage_zeroed(ctx, "gsw-y", (y_words * 4) as u64);
    let off = dispatch::uniform_from(ctx, "gsw-off", &OffParams::default());
    let pipeline = dispatch::cached_compute_pipeline(ctx, "w4a16-gsw", src, entry)
        .unwrap_or_else(|e| panic!("pipeline {entry}: {e}"));
    let binds: Vec<(u32, &wgpu::Buffer)> = if entry == gw::SG_PK_ENTRY {
        vec![(1, &s), (3, &y), (4, &p), (6, &w), (7, &xb), (30, &off)]
    } else if entry == gw::V4_ENTRY {
        vec![
            (1, &s),
            (3, &y),
            (4, &p),
            (gw::V4_PACKED_SLOT, &w),
            (gw::V4_X_SLOT, &xb),
        ]
    } else {
        vec![(0, &w), (1, &s), (2, &xb), (3, &y), (4, &p)]
    };
    let bg = dispatch::bind_group(ctx, &pipeline, &binds);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }
    ctx.queue.submit([enc.finish()]);
    ctx.poll_blocking().expect("poll");
    let words: Vec<u32> = dispatch::read_back(ctx, &y, y_words).expect("read_back");
    if packed_y {
        let mut out = vec![0u16; n];
        for (i, word) in words.iter().enumerate() {
            out[2 * i] = (*word & 0xffff) as u16;
            out[2 * i + 1] = (*word >> 16) as u16;
        }
        out
    } else {
        words.iter().map(|word| (*word & 0xffff) as u16).collect()
    }
}

struct Verdict {
    mismatch: usize,
    max_rel: f64,
}

fn verify(case: &Case<'_>, n: usize, sched: Sched, got: &[u16], tag: &str) -> Verdict {
    let mut mismatch = 0usize;
    let mut max_abs = 0f64;
    let mut ref_amp = 0f64;
    let mut schedule_gap = 0f64;
    for row in 0..n {
        let exact = case.row_exact(row, sched);
        if bf16::from_f32(exact).to_bits() != got[row] {
            mismatch += 1;
        }
        let f64ref = case.row_f64(row);
        ref_amp = ref_amp.max(f64ref.abs());
        max_abs = max_abs.max((d(got[row]) as f64 - f64ref).abs());
        schedule_gap = schedule_gap.max((exact as f64 - f64ref).abs());
    }
    let denom = ref_amp.max(1e-9);

    assert!(
        schedule_gap / denom < 0.05,
        "{tag}: the order-matching reference disagrees with the independent f64 sum by {:.3} rel -- the reference is wrong, not the kernel",
        schedule_gap / denom
    );
    Verdict {
        mismatch,
        max_rel: max_abs / denom,
    }
}

const GROUP_SIZES: [usize; 12] = [8, 16, 24, 32, 40, 48, 56, 64, 72, 96, 128, 192];

const K_ALL: usize = 40320;

#[test]
fn every_accepted_group_size_is_numerically_correct_in_every_reachable_variant() {
    let ctx = ctx("w4a16_group_size_contract");
    let n = 64usize;
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();
    let mut bad = Vec::new();
    for gs in GROUP_SIZES {
        for k in [K_ALL, 5376usize] {
            if !k.is_multiple_of(gs) {
                continue;
            }
            if gw::shape_rule(k, gs).is_err() {
                rejected.push((gs, k));
                continue;
            }
            accepted.push((gs, k));
            let mut rng = Lcg(0x9611 ^ (gs as u64) ^ ((k as u64) << 13));
            let packed = rng.packed(n * (k / 8));
            let scales = rng.scales(n * (k / gs));
            let x = rng.bf16_words(k, 1.5);
            let case = Case {
                packed: &packed,
                scales: &scales,
                x: &x,
                k,
                gs,
            };
            let wgsl = compose(gw::WGSL);
            let mut check = |label: String, sched: Sched, got: Vec<u16>| {
                let v = verify(&case, n, sched, &got, &label);
                eprintln!(
                    "w4a16-gs gs={gs:<4} k={k:<6} {label:<26} | {:>3}/{n} words off, max rel vs independent f64 {:.3e}",
                    v.mismatch, v.max_rel
                );
                if v.mismatch != 0 || v.max_rel > 0.02 {
                    bad.push(format!(
                        "{label} gs={gs} k={k}: {}/{n} words off, max rel {:.3e}",
                        v.mismatch, v.max_rel
                    ));
                }
            };
            check(
                gw::BLOCK_ENTRY.into(),
                Sched::Lane32Fma,
                run_entry(
                    ctx,
                    &wgsl,
                    gw::BLOCK_ENTRY,
                    &case,
                    n,
                    gw::ROWS_PER_GROUP,
                    false,
                ),
            );
            check(
                gw::V4_ENTRY.into(),
                Sched::Lane32Fma,
                run_entry(
                    ctx,
                    &wgsl,
                    gw::V4_ENTRY,
                    &case,
                    n,
                    gw::ROWS_PER_GROUP,
                    false,
                ),
            );
            check(
                gw::ROW_ENTRY.into(),
                Sched::Block256Pairwise,
                run_entry(ctx, &wgsl, gw::ROW_ENTRY, &case, n, 1, false),
            );
            if let Some(grain) = ScaleGrain::for_group_size(gs) {
                let src = gw::sg_pk_source_grain(grain);
                check(
                    format!("sg_pk/{grain:?}"),
                    Sched::Lane16Fma,
                    run_entry(ctx, &src, gw::SG_PK_ENTRY, &case, n, gw::SG_PK_ROWS, true),
                );
            }
        }
    }
    eprintln!("w4a16-gs accepted {accepted:?}\nw4a16-gs rejected {rejected:?}");
    assert!(
        bad.is_empty(),
        "a group size the shape rule accepts computed wrong results:\n  {}",
        bad.join("\n  ")
    );
}

#[test]
fn the_route_never_sends_an_inexpressible_group_size_to_the_sg_body() {
    let mut bad = Vec::new();
    for gs in (8..=512).step_by(8) {
        let route = gw::w4_route(4096, 5376, gs, true, false);
        if route == gw::W4Route::Sg16 && !ScaleGrain::Ge32.accepts(gs) {
            bad.push(gs);
        }
        if let Some((r, grain)) = gw::w4_route_grain(4096, 5376, gs, true, false) {
            assert!(
                grain.accepts(gs),
                "w4_route_grain handed back {grain:?} for gs={gs}"
            );
            if r == gw::W4Route::Sg16 {
                gw::require_grain(grain, gs).expect("routed sg16 must pass its own guard");
            }
        }
    }
    assert!(
        bad.is_empty(),
        "w4_route sends these group sizes to the sg body, which folds 32 weights under one scale and cannot express them: {bad:?}"
    );
}
