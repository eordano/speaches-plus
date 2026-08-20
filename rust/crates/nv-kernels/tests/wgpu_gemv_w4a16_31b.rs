#![cfg(feature = "wgpu")]

mod common;
use common::d;
use common::dot8;
use common::LcgShift33W4a16Packs as Lcg;
use common::OffParams;
use common::pack_u16;
use common::Params;
use common::q;
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
    let width = ctx.subgroup_width();
    eprintln!("{test}: probed subgroup width {width:?}");
    assert!(
        gw::sg_pk_supported(width),
        "{test}: probed subgroup width {width:?} cannot host the 16-lane sg body"
    );
    ctx
}

fn oracle_row(
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    xoff: usize,
    row: usize,
    k: usize,
    gs: usize,
    grain: ScaleGrain,
) -> f32 {
    let lanes = gw::SG_PK_LANES as usize;
    let kv = k / 32;
    let wbase = row * (k / 8);
    let sbase = row * (k / gs);
    let accs: Vec<f32> = (0..lanes)
        .map(|lane| {
            let mut acc = 0f32;
            let mut v = lane;
            while v < kv {
                let kbase = v * 32;
                match grain {
                    ScaleGrain::Ge32 | ScaleGrain::Ge32Fixed(_) => {
                        let sc = d(scales[sbase + kbase / gs]);
                        let mut a = 0f32;
                        for j in 0..4 {
                            a = dot8(packed[wbase + v * 4 + j], x, xoff + kbase + j * 8, a);
                        }
                        acc = sc.mul_add(a, acc);
                    }
                    ScaleGrain::G16 => {
                        let mut a0 = 0f32;
                        a0 = dot8(packed[wbase + v * 4], x, xoff + kbase, a0);
                        a0 = dot8(packed[wbase + v * 4 + 1], x, xoff + kbase + 8, a0);
                        let mut a1 = 0f32;
                        a1 = dot8(packed[wbase + v * 4 + 2], x, xoff + kbase + 16, a1);
                        a1 = dot8(packed[wbase + v * 4 + 3], x, xoff + kbase + 24, a1);
                        acc = d(scales[sbase + 2 * v]).mul_add(a0, acc);
                        acc = d(scales[sbase + 2 * v + 1]).mul_add(a1, acc);
                    }
                }
                v += lanes;
            }
            acc
        })
        .collect();
    tree_sum(&accs)
}

fn oracle_packed(
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
    gs: usize,
    grain: ScaleGrain,
) -> Vec<u32> {
    (0..n / 2)
        .map(|p| {
            let lo = bf16::from_f32(oracle_row(packed, scales, x, 0, 2 * p, k, gs, grain)).to_bits()
                as u32;
            let hi = bf16::from_f32(oracle_row(packed, scales, x, 0, 2 * p + 1, k, gs, grain))
                .to_bits() as u32;
            lo | (hi << 16)
        })
        .collect()
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SplitParams {
    q_rows: u32,
    kv_rows: u32,
    v_off: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct MkParams {
    m: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    dst_word_off: u32,
}

#[test]
fn q2a_the_ge32_sg_body_is_silently_wrong_at_group_16() {
    let ctx = ctx("q2a_ge32_at_group_16");
    let (n, k, gs) = (2048usize, 5376usize, 16usize);
    let mut rng = Lcg(0x9161);
    let packed = rng.packed(n * (k / 8));
    let scales = rng.scales(n * (k / gs));
    let x = rng.bf16_words(k, 1.5);
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, gw::SG_PK_ROWS);
    let params = Params {
        n_rows: n as u32,
        k_elems: k as u32,
        gs: gs as u32,
        w_row_words: (k / 8) as u32,
        scale_row_stride: (k / gs) as u32,
        groups_x: groups.0,
    };
    let want = oracle_packed(&packed, &scales, &x, n, k, gs, ScaleGrain::G16);
    let mut seen = Vec::new();
    for grain in [ScaleGrain::Ge32, ScaleGrain::G16] {
        let p = dispatch::uniform_from(ctx, "g16-params", &params);
        let w = dispatch::storage_from_slice(ctx, "g16-w", &packed);
        let s = dispatch::storage_from_slice(ctx, "g16-s", &gw::pack_scale_words(&scales));
        let xb = dispatch::storage_from_slice(ctx, "g16-x", &pack_u16(&x));
        let y = dispatch::storage_zeroed(ctx, "g16-y", (n / 2 * 4) as u64);
        let off = dispatch::uniform_from(ctx, "g16-off", &OffParams::default());
        let src = gw::sg_pk_source_grain(grain);
        let pipeline = dispatch::cached_compute_pipeline(ctx, "w4a16-g16", &src, gw::SG_PK_ENTRY)
            .expect("pipeline");
        let bg = dispatch::bind_group(
            ctx,
            &pipeline,
            &[(1, &s), (3, &y), (4, &p), (6, &w), (7, &xb), (30, &off)],
        );
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        ctx.queue.submit([enc.finish()]);
        ctx.poll_blocking().expect("poll");
        let got: Vec<u32> = dispatch::read_back(ctx, &y, n / 2).expect("read_back");
        let mismatch = want.iter().zip(got.iter()).filter(|(a, b)| a != b).count();
        eprintln!(
            "g16-hazard n={n} k={k} gs={gs} grain={grain:?} | {mismatch}/{} output words differ from the true group-16 oracle",
            n / 2
        );
        seen.push((grain, mismatch));
    }
    assert!(
        seen[0].1 * 10 > (n / 2) * 9,
        "the Ge32 body must be shown wrong at gs=16 -- if it nearly matches, the oracle is not a group-16 oracle ({}/{})",
        seen[0].1,
        n / 2
    );
    assert_eq!(
        seen[1].1, 0,
        "the G16 body must be bit-exact against the true group-16 oracle"
    );
}

#[test]
fn q2b_g16_is_bit_exact_on_every_31b_shape_and_costs_only_scale_bytes() {
    let ctx = ctx("q2b_g16_bit_exact_and_rate");
    for (tag, n, k) in [
        ("gate_up", 4096usize, 5376usize),
        ("down", 5376, 4096),
        ("qkv", 4096, 5376),
        ("o", 5376, 2048),
        ("ragged", 42, 96),
    ] {
        gw::g16_shape_rule(k).expect("legal g16 shape");
        let mut rng = Lcg(0x9162 ^ (n as u64) ^ ((k as u64) << 21));
        let packed = rng.packed(n * (k / 8));
        let scales = rng.scales(n * (k / 16));
        let x = rng.bf16_words(k, 1.5);
        let groups = dispatch::workgroup_count_1d(ctx, n as u64, gw::SG_PK_ROWS);
        let params = Params {
            n_rows: n as u32,
            k_elems: k as u32,
            gs: 16,
            w_row_words: (k / 8) as u32,
            scale_row_stride: (k / 16) as u32,
            groups_x: groups.0,
        };
        let p = dispatch::uniform_from(ctx, "g16b-params", &params);
        let w = dispatch::storage_from_slice(ctx, "g16b-w", &packed);
        let s = dispatch::storage_from_slice(ctx, "g16b-s", &gw::pack_scale_words(&scales));
        let xb = dispatch::storage_from_slice(ctx, "g16b-x", &pack_u16(&x));
        let y = dispatch::storage_zeroed(ctx, "g16b-y", (n.div_ceil(2) * 4) as u64);
        let off = dispatch::uniform_from(ctx, "g16b-off", &OffParams::default());
        let src = gw::sg_pk_source_grain(ScaleGrain::G16);
        let pipeline = dispatch::cached_compute_pipeline(ctx, "w4a16-g16b", &src, gw::SG_PK_ENTRY)
            .expect("pipeline");
        let bg = dispatch::bind_group(
            ctx,
            &pipeline,
            &[(1, &s), (3, &y), (4, &p), (6, &w), (7, &xb), (30, &off)],
        );
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        ctx.queue.submit([enc.finish()]);
        ctx.poll_blocking().expect("poll");
        let got: Vec<u32> = dispatch::read_back(ctx, &y, n / 2).expect("read_back");
        let want = oracle_packed(&packed, &scales, &x, n, k, 16, ScaleGrain::G16);
        let mismatch = want.iter().zip(got.iter()).filter(|(a, b)| a != b).count();
        eprintln!(
            "g16-exact {tag:<8} n={n:<6} k={k:<6} | {mismatch}/{} words off",
            n / 2
        );
        assert_eq!(mismatch, 0, "{tag}: G16 must be bit-exact");
    }
}

#[test]
fn q2f_every_g16_dispatch_arm_agrees_with_the_m1_decode_arm() {
    let ctx = ctx("q2f_g16_dispatch_arms");
    for (tag, q_rows, kv_rows, k, mk_max, m) in [
        (
            "qkv-split-m1",
            4096usize,
            1024usize,
            5376usize,
            8u32,
            1usize,
        ),
        ("qkv-split-m8", 4096, 1024, 5376, 8, 8),
        ("ragged-m3", 96, 32, 128, 4, 3),
    ] {
        let n = q_rows + 2 * kv_rows;
        let v_off = q_rows + kv_rows;
        let gs = 16usize;
        let grain = ScaleGrain::G16;
        let mut rng = Lcg(0x9166 ^ (n as u64) ^ ((k as u64) << 17));
        let packed = rng.packed(n * (k / 8));
        let scales = rng.scales(n * (k / gs));
        let x = rng.bf16_words(m * k, 1.5);
        let scale_words = gw::pack_scale_words(&scales);
        let want = m1_reference(ctx, &packed, &scale_words, &x, n, k, gs, m, grain);

        let groups = dispatch::workgroup_count_1d(ctx, n as u64, gw::SG_PK_ROWS);
        let params = Params {
            n_rows: n as u32,
            k_elems: k as u32,
            gs: gs as u32,
            w_row_words: (k / 8) as u32,
            scale_row_stride: (k / gs) as u32,
            groups_x: groups.0,
        };
        let w = dispatch::storage_from_slice(ctx, "arm-w", &packed);
        let s = dispatch::storage_from_slice(ctx, "arm-s", &scale_words);
        let p = dispatch::uniform_from(ctx, "arm-p", &params);
        let sp = dispatch::uniform_from(
            ctx,
            "arm-split",
            &SplitParams {
                q_rows: q_rows as u32,
                kv_rows: kv_rows as u32,
                v_off: v_off as u32,
                pad0: 0,
            },
        );
        let yq = dispatch::storage_zeroed(ctx, "arm-q", (m * q_rows / 2 * 4) as u64);
        let yk = dispatch::storage_zeroed(ctx, "arm-k", (m * kv_rows / 2 * 4) as u64);
        let yv = dispatch::storage_zeroed(ctx, "arm-v", (m * kv_rows / 2 * 4) as u64);

        let (src, entry, xbuf) = if m == 1 {
            (
                gw::sg_pk_source_grain(grain),
                gw::SG_PK3_ENTRY,
                dispatch::storage_from_slice(ctx, "arm-x", &pack_u16(&x)),
            )
        } else {
            (
                gw::sg_mk_unrolled_source_grain(mk_max, grain),
                gw::SG_MK_PK3_ENTRY,
                dispatch::storage_from_slice(ctx, "arm-x", &pack_u16(&x)),
            )
        };
        let mkp = dispatch::uniform_from(
            ctx,
            "arm-mkp",
            &MkParams {
                m: m as u32,
                x_stride_words: (k / 2) as u32,
                y_stride_words: 0,
                dst_word_off: 0,
            },
        );
        let pipeline = dispatch::cached_compute_pipeline(ctx, "w4a16-arm", &src, entry)
            .unwrap_or_else(|e| panic!("{tag}: pipeline {entry}: {e}"));
        let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![
            (4, &p),
            (6, &w),
            (7, &xbuf),
            (1, &s),
            (31, &yq),
            (32, &yk),
            (33, &yv),
            (34, &sp),
        ];
        if m > 1 {
            binds.push((35, &mkp));
        }
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
        let got_q: Vec<u32> = dispatch::read_back(ctx, &yq, m * q_rows / 2).unwrap();
        let got_k: Vec<u32> = dispatch::read_back(ctx, &yk, m * kv_rows / 2).unwrap();
        let got_v: Vec<u32> = dispatch::read_back(ctx, &yv, m * kv_rows / 2).unwrap();
        let mut mismatch = 0usize;
        for (t, row) in want.iter().enumerate() {
            for i in 0..q_rows / 2 {
                mismatch += usize::from(got_q[t * (q_rows / 2) + i] != row[i]);
            }
            for i in 0..kv_rows / 2 {
                mismatch += usize::from(got_k[t * (kv_rows / 2) + i] != row[q_rows / 2 + i]);
                mismatch += usize::from(got_v[t * (kv_rows / 2) + i] != row[v_off / 2 + i]);
            }
        }
        eprintln!(
            "g16-arms {tag:<14} q={q_rows} kv={kv_rows} k={k} m={m} entry={entry} | {mismatch} words off"
        );
        assert_eq!(
            mismatch, 0,
            "{tag}: {entry} at gs=16 must match the M=1 decode arm"
        );
    }
}

#[test]
fn q2e_fixed_shift_grain_is_bit_identical_to_the_runtime_divide() {
    let ctx = ctx("q2e_fixed_shift_grain");
    for (tag, n, k, gs) in [
        ("gate_up-g32", 4096usize, 5376usize, 32usize),
        ("down-g64", 5376, 4096, 64),
        ("qkv-g128", 4096, 5376, 128),
        ("ragged-g32", 42, 96, 32),
    ] {
        let shift = (gs / 32).trailing_zeros();
        let mut rng = Lcg(0x9165 ^ (n as u64) ^ ((k as u64) << 19) ^ gs as u64);
        let packed = rng.packed(n * (k / 8));
        let scales = rng.scales(n * (k / gs));
        let x = rng.bf16_words(k, 1.5);
        let scale_words = gw::pack_scale_words(&scales);
        let groups = dispatch::workgroup_count_1d(ctx, n as u64, gw::SG_PK_ROWS);
        let params = Params {
            n_rows: n as u32,
            k_elems: k as u32,
            gs: gs as u32,
            w_row_words: (k / 8) as u32,
            scale_row_stride: (k / gs) as u32,
            groups_x: groups.0,
        };
        let mut out = Vec::new();
        for grain in [ScaleGrain::Ge32, ScaleGrain::Ge32Fixed(shift)] {
            assert!(
                grain.accepts(gs),
                "{tag}: {grain:?} does not accept gs={gs}"
            );
            let p = dispatch::uniform_from(ctx, "fx-params", &params);
            let w = dispatch::storage_from_slice(ctx, "fx-w", &packed);
            let s = dispatch::storage_from_slice(ctx, "fx-s", &scale_words);
            let xb = dispatch::storage_from_slice(ctx, "fx-x", &pack_u16(&x));
            let y = dispatch::storage_zeroed(ctx, "fx-y", (n.div_ceil(2) * 4) as u64);
            let off = dispatch::uniform_from(ctx, "fx-off", &OffParams::default());
            let src = gw::sg_pk_source_grain(grain);
            let pipeline =
                dispatch::cached_compute_pipeline(ctx, "w4a16-fixed", &src, gw::SG_PK_ENTRY)
                    .expect("pipeline");
            let bg = dispatch::bind_group(
                ctx,
                &pipeline,
                &[(1, &s), (3, &y), (4, &p), (6, &w), (7, &xb), (30, &off)],
            );
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
            ctx.queue.submit([enc.finish()]);
            ctx.poll_blocking().expect("poll");
            out.push(dispatch::read_back::<u32>(ctx, &y, n / 2).expect("read_back"));
        }
        let want = oracle_packed(&packed, &scales, &x, n, k, gs, ScaleGrain::Ge32);
        let vs_oracle = want
            .iter()
            .zip(out[1].iter())
            .filter(|(a, b)| a != b)
            .count();
        let vs_runtime = out[0]
            .iter()
            .zip(out[1].iter())
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "fixed-shift {tag:<12} n={n:<6} k={k:<6} gs={gs:<3} shift={shift} | {vs_runtime}/{} vs runtime-divide, {vs_oracle}/{} vs cpu oracle",
            n / 2,
            n / 2
        );
        assert_eq!(vs_runtime, 0, "{tag}: shift grain diverged from the divide");
        assert_eq!(vs_oracle, 0, "{tag}: shift grain diverged from the oracle");
    }
}

fn m1_reference(
    ctx: &WgpuContext,
    packed: &[u32],
    scale_words: &[u32],
    x: &[u16],
    n: usize,
    k: usize,
    gs: usize,
    m: usize,
    grain: ScaleGrain,
) -> Vec<Vec<u32>> {
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, gw::SG_PK_ROWS);
    let params = Params {
        n_rows: n as u32,
        k_elems: k as u32,
        gs: gs as u32,
        w_row_words: (k / 8) as u32,
        scale_row_stride: (k / gs) as u32,
        groups_x: groups.0,
    };
    let src = gw::sg_pk_source_grain(grain);
    let w = dispatch::storage_from_slice(ctx, "mk-w", packed);
    let s = dispatch::storage_from_slice(ctx, "mk-s", scale_words);
    let p = dispatch::uniform_from(ctx, "mk-p", &params);
    let off = dispatch::uniform_from(ctx, "mk-off", &OffParams::default());
    (0..m)
        .map(|t| {
            let xt = pack_u16(&x[t * k..(t + 1) * k]);
            let xb = dispatch::storage_from_slice(ctx, "mk-xt", &xt);
            let y = dispatch::storage_zeroed(ctx, "mk-yt", (n / 2 * 4) as u64);
            let pipeline =
                dispatch::cached_compute_pipeline(ctx, "w4a16-mk-ref", &src, gw::SG_PK_ENTRY)
                    .expect("pipeline");
            let bg = dispatch::bind_group(
                ctx,
                &pipeline,
                &[(1, &s), (3, &y), (4, &p), (6, &w), (7, &xb), (30, &off)],
            );
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
            ctx.queue.submit([enc.finish()]);
            ctx.poll_blocking().expect("poll");
            dispatch::read_back(ctx, &y, n / 2).expect("read_back")
        })
        .collect()
}

#[test]
fn q3_m_row_twin_is_bit_exact_against_the_m1_kernel_on_31b_shapes_and_both_grains() {
    let ctx = ctx("q3_m_row_twin");
    for (tag, n, k, gs, mk_max, m, fixed) in [
        (
            "31b-gate_up-43008-g32-m8",
            43008usize,
            5376usize,
            32usize,
            8u32,
            8usize,
            false,
        ),
        ("31b-down-g32-m16", 5376, 21504, 32, 16, 16, false),
        ("31b-qkv-g32-m5of8", 16384, 5376, 32, 8, 5, false),
        ("31b-o-g32-m4", 5376, 8192, 32, 4, 4, false),
        ("31b-gate_up-43008-g32fix-m8", 43008, 5376, 32, 8, 8, true),
        ("31b-gate_up-43008-g16-m8", 43008, 5376, 16, 8, 8, false),
        ("31b-gate_up-8192-g32-m8", 8192, 5376, 32, 8, 8, false),
        ("31b-down-g64fix-m16", 5376, 21504, 64, 16, 16, true),
        ("31b-qkv-g128fix-m4of8", 16384, 5376, 128, 8, 4, true),
        ("31b-gate_up-8192-g16-m8", 8192, 5376, 16, 8, 8, false),
        ("31b-down-g16-m16", 5376, 21504, 16, 16, 16, false),
        ("31b-qkv-g16-m1of16", 16384, 5376, 16, 16, 1, false),
    ] {
        let grain = if fixed {
            ScaleGrain::fastest_for_group_size(gs)
        } else {
            ScaleGrain::for_group_size(gs)
        }
        .expect("supported grain");
        assert!(grain.accepts(gs), "{tag}: {grain:?} rejects gs={gs}");
        let mut rng = Lcg(0x5316 ^ (n as u64) ^ ((k as u64) << 22) ^ (m as u64));
        let packed = rng.packed(n * (k / 8));
        let scales = rng.scales(n * (k / gs));
        let x = rng.bf16_words(m * k, 1.5);
        let scale_words = gw::pack_scale_words(&scales);
        let want = m1_reference(ctx, &packed, &scale_words, &x, n, k, gs, m, grain);

        let groups = ((n as u32).div_ceil(gw::SG_PK_ROWS), 1, 1);
        let params = Params {
            n_rows: n as u32,
            k_elems: k as u32,
            gs: gs as u32,
            w_row_words: (k / 8) as u32,
            scale_row_stride: (k / gs) as u32,
            groups_x: groups.0,
        };
        let word_off = 3usize;
        let y_stride_words = n / 2 + 5;
        let w = dispatch::storage_from_slice(ctx, "mk-w", &packed);
        let s = dispatch::storage_from_slice(ctx, "mk-s", &scale_words);
        let xb = dispatch::storage_from_slice(ctx, "mk-x", &pack_u16(&x));
        let p = dispatch::uniform_from(ctx, "mk-p", &params);
        let mkp = dispatch::uniform_from(
            ctx,
            "mk-mkp",
            &MkParams {
                m: m as u32,
                x_stride_words: (k / 2) as u32,
                y_stride_words: y_stride_words as u32,
                dst_word_off: word_off as u32,
            },
        );
        let y_words = word_off + m * y_stride_words;
        let y = dispatch::storage_zeroed(ctx, "mk-y", (y_words * 4) as u64);
        let src = gw::sg_mk_unrolled_source_grain(mk_max, grain);
        let pipeline = dispatch::cached_compute_pipeline(ctx, "w4a16-mk", &src, gw::SG_MK_PK_ENTRY)
            .expect("pipeline");
        let bg = dispatch::bind_group(
            ctx,
            &pipeline,
            &[(1, &s), (3, &y), (4, &p), (6, &w), (7, &xb), (35, &mkp)],
        );
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        ctx.queue.submit([enc.finish()]);
        ctx.poll_blocking().expect("poll");
        let got: Vec<u32> = dispatch::read_back(ctx, &y, y_words).expect("read_back");
        assert!(
            got[..word_off].iter().all(|w| *w == 0),
            "{tag}: words below dst_word_off must stay untouched"
        );
        let mut mismatch = 0usize;
        for (t, row) in want.iter().enumerate() {
            let base = word_off + t * y_stride_words;
            mismatch += row
                .iter()
                .zip(got[base..base + n / 2].iter())
                .filter(|(a, b)| a != b)
                .count();
        }
        eprintln!(
            "mk-exact {tag:<22} n={n:<6} k={k:<6} gs={gs:<3} mk_max={mk_max:<3} m={m:<3} | {mismatch}/{} words off",
            m * n / 2
        );
        assert_eq!(
            mismatch, 0,
            "{tag}: the M-row twin must be bit-identical to M=1 row by row"
        );
    }
}

#[test]
fn q2d_nvfp4_to_affine_int4_is_lossy_at_every_group_size() {
    const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    fn nvfp4_round(v: f32, s: f32) -> f32 {
        if s == 0.0 {
            return 0.0;
        }
        let a = (v / s).abs();
        let mut best = E2M1[0];
        for c in E2M1 {
            if (c - a).abs() < (best - a).abs() {
                best = c;
            }
        }
        best * s * v.signum()
    }

    fn affine(vals: &[f32], lo: f32, hi_lvl: f32) -> Vec<f32> {
        let amp = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
        if amp == 0.0 {
            return vec![0.0; vals.len()];
        }
        let mut best = (f32::MAX, Vec::new());
        for step in 1..=64 {
            let s = amp / hi_lvl * (step as f32 / 32.0);
            let deq: Vec<f32> = vals
                .iter()
                .map(|v| ((v / s).round().clamp(lo, hi_lvl)) * s)
                .collect();
            let err: f32 = vals.iter().zip(&deq).map(|(a, b)| (a - b) * (a - b)).sum();
            if err < best.0 {
                best = (err, deq);
            }
        }
        best.1
    }
    let mut rng = Lcg(0xf00d);
    let mut rows = Vec::new();
    for (label, gs, lo, hi_lvl) in [
        ("int4 g16 (source-matched)", 16usize, -8.0f32, 7.0f32),
        ("int4 g32 (in-tree)", 32, -8.0, 7.0),
        ("int8 g32 (calibration)", 32, -128.0, 127.0),
        ("int8 g128 (shipping 31B)", 128, -128.0, 127.0),
    ] {
        let (mut num, mut den, mut worst) = (0f64, 0f64, 0f64);
        for _ in 0..(65536 / gs) {
            let src: Vec<f32> = (0..gs)
                .map(|_| {
                    let (u1, u2) = (
                        (rng.next_u32() as f32 / 4294967296.0).max(1e-7),
                        rng.next_u32() as f32 / 4294967296.0,
                    );
                    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
                })
                .collect();
            let mut nv = Vec::with_capacity(gs);
            for blk in src.chunks(16) {
                let s = blk.iter().fold(0f32, |m, v| m.max(v.abs())) / 6.0;
                nv.extend(blk.iter().map(|v| nvfp4_round(*v, s)));
            }
            let re = affine(&nv, lo, hi_lvl);
            let e: f64 = nv
                .iter()
                .zip(&re)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum();
            let p: f64 = nv.iter().map(|a| (*a as f64).powi(2)).sum();
            num += e;
            den += p;
            worst = worst.max((e / p.max(1e-30)).sqrt());
        }
        let rel = (num / den).sqrt();
        eprintln!("nvfp4-requant {label:<26} | relative rms {rel:.5} | worst group {worst:.5}");
        rows.push((label, rel));
    }
    assert!(
        rows[0].1 > 1.0e-2 && rows[1].1 > 1.0e-2,
        "int4 requant landed under 1e-2 rms, which would contradict the non-uniform-code argument: {rows:?}"
    );
    assert!(
        rows[3].1 * 8.0 < rows[0].1,
        "int8 g128 must be far cleaner than int4 g16 for the comparison to mean anything: {rows:?}"
    );
}

const WIRED: [(&str, &str, usize, usize, usize); 16] = [
    ("e4b", "gate_up", 20480, 2560, 200),
    ("e4b", "down", 2560, 10240, 300),
    ("31b", "gate_up", 43008, 5376, 60),
    ("31b", "down", 5376, 21504, 60),
    ("31b", "qkv", 16384, 5376, 80),
    ("31b", "o", 5376, 8192, 150),
    ("qwen-a3b", "dn_in_qkv", 8192, 2048, 400),
    ("qwen-a3b", "dn_in_z", 4096, 2048, 600),
    ("qwen-a3b", "dn_in_ab", 64, 2048, 2000),
    ("qwen-a3b", "dn_out", 2048, 4096, 600),
    ("qwen-a3b", "attn_q", 4096, 2048, 600),
    ("qwen-a3b", "attn_kv", 512, 2048, 2000),
    ("qwen-a3b", "expert_gate_up", 1024, 2048, 2000),
    ("qwen-a3b", "expert_down", 2048, 512, 2000),
    ("qwen-a3b", "lm_head", 248320, 2048, 30),
    ("q35-9b", "lm_head", 248320, 4096, 20),
];

fn sg_pk_once(
    ctx: &WgpuContext,
    packed: &[u32],
    scale_words: &[u32],
    x: &[u16],
    n: usize,
    k: usize,
    gs: usize,
    grain: ScaleGrain,
) -> Vec<u32> {
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, gw::SG_PK_ROWS);
    let params = Params {
        n_rows: n as u32,
        k_elems: k as u32,
        gs: gs as u32,
        w_row_words: (k / 8) as u32,
        scale_row_stride: (k / gs) as u32,
        groups_x: groups.0,
    };
    let p = dispatch::uniform_from(ctx, "wire-p", &params);
    let w = dispatch::storage_from_slice(ctx, "wire-w", packed);
    let s = dispatch::storage_from_slice(ctx, "wire-s", scale_words);
    let xb = dispatch::storage_from_slice(ctx, "wire-x", &pack_u16(x));
    let y = dispatch::storage_zeroed(ctx, "wire-y", (n / 2 * 4) as u64);
    let off = dispatch::uniform_from(ctx, "wire-off", &OffParams::default());
    let src = gw::sg_pk_source_grain(grain);
    let pipeline = dispatch::cached_compute_pipeline(ctx, "w4a16-wired", &src, gw::SG_PK_ENTRY)
        .unwrap_or_else(|e| panic!("pipeline {grain:?}: {e}"));
    let bg = dispatch::bind_group(
        ctx,
        &pipeline,
        &[(1, &s), (3, &y), (4, &p), (6, &w), (7, &xb), (30, &off)],
    );
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }
    ctx.queue.submit([enc.finish()]);
    ctx.poll_blocking().expect("poll");
    dispatch::read_back(ctx, &y, n / 2).expect("read_back")
}

fn oracle_row_wide(
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    row: usize,
    k: usize,
    gs: usize,
) -> f32 {
    let lanes = 32usize;
    let kv = k / 32;
    let wbase = row * (k / 8);
    let sbase = row * (k / gs);
    let accs: Vec<f32> = (0..lanes)
        .map(|lane| {
            let mut acc = 0f32;
            let mut v = lane;
            while v < kv {
                let kbase = v * 32;
                if gs >= 32 {
                    let sc = d(scales[sbase + kbase / gs]);
                    let mut a = 0f32;
                    for j in 0..4 {
                        a = dot8(packed[wbase + v * 4 + j], x, kbase + j * 8, a);
                    }
                    acc = sc.mul_add(a, acc);
                } else {
                    for j in 0..4 {
                        let kb = kbase + j * 8;
                        let sc = d(scales[sbase + kb / gs]);
                        let a = dot8(packed[wbase + v * 4 + j], x, kb, 0.0);
                        acc = a.mul_add(sc, acc);
                    }
                }
                v += lanes;
            }
            acc
        })
        .collect();
    tree_sum(&accs)
}

fn wide_once(
    ctx: &WgpuContext,
    entry: &str,
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
    gs: usize,
) -> Vec<u32> {
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, gw::ROWS_PER_GROUP);
    let params = Params {
        n_rows: n as u32,
        k_elems: k as u32,
        gs: gs as u32,
        w_row_words: (k / 8) as u32,
        scale_row_stride: (k / gs) as u32,
        groups_x: groups.0,
    };
    let p = dispatch::uniform_from(ctx, "wide-p", &params);
    let w = dispatch::storage_from_slice(ctx, "wide-w", packed);
    let s = dispatch::storage_from_slice(ctx, "wide-s", &widen_u16(scales));
    let xb = dispatch::storage_from_slice(ctx, "wide-x", &pack_u16(x));
    let y = dispatch::storage_zeroed(ctx, "wide-y", (n * 4) as u64);
    let src = compose(gw::WGSL);
    let pipeline = dispatch::cached_compute_pipeline(ctx, "w4a16-wide", &src, entry)
        .unwrap_or_else(|e| panic!("pipeline {entry}: {e}"));
    let bindings: Vec<(u32, &wgpu::Buffer)> = if entry == gw::V4_ENTRY {
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
    let bg = dispatch::bind_group(ctx, &pipeline, &bindings);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }
    ctx.queue.submit([enc.finish()]);
    ctx.poll_blocking().expect("poll");
    dispatch::read_back(ctx, &y, n).expect("read_back")
}

fn oracle_probe_words(n: usize, k: usize) -> Vec<usize> {
    let words = n / 2;
    if n * k <= (8 << 20) {
        return (0..words).collect();
    }
    let mut v: Vec<usize> = (0..words.min(256)).collect();
    v.extend(words.saturating_sub(256)..words);
    let stride = (words / 512).max(1);
    v.extend((0..words).step_by(stride));
    v.sort_unstable();
    v.dedup();
    v
}

#[test]
fn q5_every_wired_shape_is_bit_exact_through_its_routed_variant() {
    let ctx = ctx("q5_wired_bit_exact");
    for (model, tensor, n, k, _) in WIRED {
        for gs in [32usize, 16] {
            let tag = format!("{model}/{tensor}-g{gs}");
            if gs == 16 {
                gw::g16_shape_rule(k).unwrap_or_else(|e| panic!("{tag}: {e}"));
            }
            let (route, grain) = gw::w4_route_grain(n, k, gs, true, false)
                .unwrap_or_else(|| panic!("{tag}: no route"));
            assert!(grain.accepts(gs), "{tag}: {grain:?} rejects gs={gs}");
            let mut rng = Lcg(0xa16 ^ (n as u64) ^ ((k as u64) << 17) ^ (gs as u64) << 40);
            let packed = rng.packed(n * (k / 8));
            let scales = rng.scales(n * (k / gs));
            let x = rng.bf16_words(k, 1.5);
            let scale_words = gw::pack_scale_words(&scales);
            let probe_words = oracle_probe_words(n, k);
            let mut mismatch = 0usize;
            match route {
                gw::W4Route::Sg16 => {
                    let got = sg_pk_once(ctx, &packed, &scale_words, &x, n, k, gs, grain);
                    for p in &probe_words {
                        let lo = bf16::from_f32(oracle_row(
                            &packed,
                            &scales,
                            &x,
                            0,
                            2 * p,
                            k,
                            gs,
                            grain,
                        ))
                        .to_bits() as u32;
                        let hi = bf16::from_f32(oracle_row(
                            &packed,
                            &scales,
                            &x,
                            0,
                            2 * p + 1,
                            k,
                            gs,
                            grain,
                        ))
                        .to_bits() as u32;
                        if got[*p] != (lo | (hi << 16)) {
                            mismatch += 1;
                        }
                    }
                }

                other => {
                    let entry = if other == gw::W4Route::V4 {
                        gw::V4_ENTRY
                    } else {
                        gw::BLOCK_ENTRY
                    };
                    let got = wide_once(ctx, entry, &packed, &scales, &x, n, k, gs);
                    for p in &probe_words {
                        for row in [2 * p, 2 * p + 1] {
                            let want =
                                bf16::from_f32(oracle_row_wide(&packed, &scales, &x, row, k, gs))
                                    .to_bits() as u32;
                            if got[row] != want {
                                mismatch += 1;
                            }
                        }
                    }
                }
            }
            eprintln!(
                "wired-exact {tag:<28} n={n:<6} k={k:<6} {route:?}/{grain:?} | {mismatch}/{} sampled words off ({} of {} total)",
                probe_words.len(),
                probe_words.len(),
                n / 2
            );
            assert_eq!(
                mismatch, 0,
                "{tag}: routed variant diverged from the oracle"
            );
        }
    }
}

#[test]
fn q6_v4_and_block_agree_at_group_sixteen() {
    let ctx = ctx("q6_v4_block_g16");
    for (tag, n, k) in [
        ("31b-o", 5376usize, 8192usize),
        ("qwen-dn_out", 2048, 4096),
        ("ragged", 40, 96),
    ] {
        let mut rng = Lcg(0x604 ^ (n as u64) ^ ((k as u64) << 13));
        let packed = rng.packed(n * (k / 8));
        let mut out = Vec::new();
        for gs in [16usize, 32] {
            let scales = rng.scales(n * (k / gs));
            let x = rng.bf16_words(k, 1.5);
            let got: Vec<Vec<u32>> = [gw::BLOCK_ENTRY, gw::V4_ENTRY]
                .into_iter()
                .map(|entry| wide_once(ctx, entry, &packed, &scales, &x, n, k, gs))
                .collect();

            let vs_oracle = (0..n)
                .filter(|row| {
                    let want = bf16::from_f32(oracle_row_wide(&packed, &scales, &x, *row, k, gs))
                        .to_bits() as u32;
                    got[0][*row] != want
                })
                .count();
            let diff = got[0]
                .iter()
                .zip(got[1].iter())
                .filter(|(a, b)| a != b)
                .count();
            eprintln!(
                "v4-vs-block {tag:<14} n={n:<6} k={k:<6} gs={gs:<3} | {diff}/{n} rows differ, block {vs_oracle}/{n} off the cpu oracle"
            );
            assert_eq!(
                vs_oracle, 0,
                "{tag}: block diverged from the oracle at gs={gs}"
            );
            out.push((gs, diff));
        }
        for (gs, diff) in out {
            assert_eq!(diff, 0, "{tag}: V4 and block disagree at gs={gs}");
        }
    }
}

#[test]
fn q7_the_mk_and_mr_generators_are_bit_exact_at_group_sixteen() {
    let ctx = ctx("q7_mk_mr_g16");
    for (tag, n, k, gs) in [
        ("31b-o", 5376usize, 8192usize, 16usize),
        ("31b-o-g32", 5376, 8192, 32),
        ("qwen-dn_out", 2048, 4096, 16),
        ("qwen-lm_head", 8192, 2048, 16),
    ] {
        let grain = ScaleGrain::for_group_size(gs).expect("grain");
        assert!(grain.accepts(gs), "{tag}: {grain:?} rejects gs={gs}");
        let mut rng = Lcg(0x7c7 ^ (n as u64) ^ ((k as u64) << 11) ^ (gs as u64));
        let packed = rng.packed(n * (k / 8));
        let scales = rng.scales(n * (k / gs));
        let scale_words = gw::pack_scale_words(&scales);
        let m = 4usize;
        let x = rng.bf16_words(m * k, 1.5);
        let want = m1_reference(ctx, &packed, &scale_words, &x, n, k, gs, m, grain);

        let groups = ((n as u32).div_ceil(gw::SG_PK_ROWS), 1, 1);
        let params = Params {
            n_rows: n as u32,
            k_elems: k as u32,
            gs: gs as u32,
            w_row_words: (k / 8) as u32,
            scale_row_stride: (k / gs) as u32,
            groups_x: groups.0,
        };
        let w = dispatch::storage_from_slice(ctx, "q7-w", &packed);
        let s = dispatch::storage_from_slice(ctx, "q7-s", &scale_words);
        let xb = dispatch::storage_from_slice(ctx, "q7-x", &pack_u16(&x));
        let p = dispatch::uniform_from(ctx, "q7-p", &params);
        let y_stride_words = n / 2;
        let mkp = dispatch::uniform_from(
            ctx,
            "q7-mkp",
            &MkParams {
                m: m as u32,
                x_stride_words: (k / 2) as u32,
                y_stride_words: y_stride_words as u32,
                dst_word_off: 0,
            },
        );
        let y = dispatch::storage_zeroed(ctx, "q7-y", ((m * y_stride_words) * 4) as u64);
        let src = gw::sg_mk_source_grain(m as u32, grain);
        let pipeline =
            dispatch::cached_compute_pipeline(ctx, "w4a16-q7-mk", &src, gw::SG_MK_PK_ENTRY)
                .unwrap_or_else(|e| panic!("{tag}: mk pipeline: {e}"));
        let bg = dispatch::bind_group(
            ctx,
            &pipeline,
            &[(1, &s), (3, &y), (4, &p), (6, &w), (7, &xb), (35, &mkp)],
        );
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        ctx.queue.submit([enc.finish()]);
        ctx.poll_blocking().expect("poll");
        let got: Vec<u32> = dispatch::read_back(ctx, &y, m * y_stride_words).expect("read_back");
        let mk_off: usize = want
            .iter()
            .enumerate()
            .map(|(t, row)| {
                let base = t * y_stride_words;
                row.iter()
                    .zip(got[base..base + n / 2].iter())
                    .filter(|(a, b)| a != b)
                    .count()
            })
            .sum();

        let mut mr_off = 0usize;
        for mr in [2u32, 4, 8] {
            let rows_per_wg = gw::SG_PK_ROWS * mr;
            let mrg = ((n as u32).div_ceil(rows_per_wg), 1, 1);
            let mrp = Params {
                groups_x: mrg.0,
                ..params
            };
            let pm = dispatch::uniform_from(ctx, "q7-mrp", &mrp);
            let x0 = dispatch::storage_from_slice(ctx, "q7-mrx", &pack_u16(&x[..k]));
            let ym = dispatch::storage_zeroed(ctx, "q7-mry", (n / 2 * 4) as u64);
            let off = dispatch::uniform_from(ctx, "q7-mroff", &OffParams::default());
            let srcm = gw::sg_pk_mr_source_grain(mr, grain);
            let pipe =
                dispatch::cached_compute_pipeline(ctx, "w4a16-q7-mr", &srcm, gw::SG_PKM_ENTRY)
                    .unwrap_or_else(|e| panic!("{tag}: mr{mr} pipeline: {e}"));
            let bgm = dispatch::bind_group(
                ctx,
                &pipe,
                &[(1, &s), (3, &ym), (4, &pm), (6, &w), (7, &x0), (30, &off)],
            );
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipe);
                pass.set_bind_group(0, &bgm, &[]);
                pass.dispatch_workgroups(mrg.0, mrg.1, mrg.2);
            }
            ctx.queue.submit([enc.finish()]);
            ctx.poll_blocking().expect("poll");
            let gm: Vec<u32> = dispatch::read_back(ctx, &ym, n / 2).expect("read_back");
            let d = want[0]
                .iter()
                .zip(gm.iter())
                .filter(|(a, b)| a != b)
                .count();
            eprintln!(
                "mr-exact    {tag:<14} n={n:<6} k={k:<6} gs={gs:<3} mr={mr:<2} | {d}/{} words off",
                n / 2
            );
            mr_off += d;
        }
        eprintln!(
            "mk-exact    {tag:<14} n={n:<6} k={k:<6} gs={gs:<3} m={m:<2} {grain:?} | {mk_off}/{} words off\n",
            m * n / 2
        );
        assert_eq!(mk_off, 0, "{tag}: sg_mk_source_grain diverged at gs={gs}");
        assert_eq!(
            mr_off, 0,
            "{tag}: sg_pk_mr_source_grain diverged at gs={gs}"
        );
    }
}
