#![cfg(feature = "wgpu")]

mod common;
use common::LcgShift32TwoSided as Lcg;
use common::wgpu_allow_skip;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemv_bf16 as g;

fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("{test}: {}", ctx.summary());
            Some(ctx)
        }
        Err(e) => {
            if !wgpu_allow_skip() {
                panic!(
                    "{test}: no wgpu adapter: {e}. This gate refuses to report success without \
                     running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
                );
            }
            eprintln!("{test}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) no wgpu adapter: {e}");
            None
        }
    }
}

fn sg32_or_skip(test: &str, ctx: &WgpuContext) -> bool {
    if !g::sg32_ok(ctx) {
        if !wgpu_allow_skip() {
            panic!(
                "{test}: probed subgroup width {:?} is not 32, so every sg arm here is unreachable \
                 and this gate would report success having compared nothing. Set \
                 NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose.",
                ctx.subgroup_width()
            );
        }
        eprintln!("{test}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) probed subgroup width is not 32");
        return false;
    }
    true
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_rows: u32,
    k_elems: u32,
    w_row_words: u32,
    groups_x: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OffParams {
    dst_word_off: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

fn pack_u16(src: &[u16]) -> Vec<u32> {
    src.chunks_exact(2)
        .map(|c| (c[0] as u32) | ((c[1] as u32) << 16))
        .collect()
}

fn run_sg_pk(
    ctx: &WgpuContext,
    w: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
    wg: u32,
    word_off: usize,
) -> Vec<u32> {
    let (entry, rows_per_group) = g::sg_pk_entry(wg);
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, rows_per_group);
    let params = Params {
        n_rows: n as u32,
        k_elems: k as u32,
        w_row_words: (k / 2) as u32,
        groups_x: groups.0,
    };
    let off = OffParams {
        dst_word_off: word_off as u32,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let n_words = word_off + n.div_ceil(2);
    let wbuf = dispatch::storage_from_slice(ctx, "sgpk-w", &pack_u16(w));
    let xbuf = dispatch::storage_from_slice(ctx, "sgpk-x", &pack_u16(x));
    let ybuf = dispatch::storage_zeroed(ctx, "sgpk-y", (n_words * 4) as u64);
    let pbuf = dispatch::uniform_from(ctx, "sgpk-p", &params);
    let obuf = dispatch::uniform_from(ctx, "sgpk-o", &off);
    dispatch::run(
        ctx,
        "gemv_bf16_sg_pk_test",
        &g::sg_pk_source(),
        entry,
        &[(0, &wbuf), (1, &xbuf), (2, &ybuf), (3, &pbuf), (30, &obuf)],
        groups,
    )
    .unwrap();
    dispatch::read_back(ctx, &ybuf, n_words).unwrap()
}

fn pack_reference(y: &[u16], word_off: usize) -> Vec<u32> {
    let mut out = vec![0u32; word_off + y.len().div_ceil(2)];
    for (i, word) in out.iter_mut().enumerate().skip(word_off) {
        let r = 2 * (i - word_off);
        let lo = y[r] as u32;
        let hi = if r + 1 < y.len() { y[r + 1] as u32 } else { 0 };
        *word = lo | (hi << 16);
    }
    out
}

#[test]
#[ignore]
fn perf_sg_pk_vs_tree_at_lm_head_shape() {
    let Some(ctx) = ctx_or_skip("sg_pk_perf") else {
        return;
    };
    if !sg32_or_skip("sg_pk_perf", ctx) {
        return;
    }
    let (n, k) = (262144usize, 2048usize);
    let mut rng = Lcg(0xbe9c);
    let w = rng.bf16_vec(n * k, 0.05);
    let x = rng.bf16_vec(k, 0.5);
    let bytes = (n as f64) * (k as f64) * 2.0;
    let iters = 40usize;
    for round in 0..3 {
        for (name, kern) in [
            ("tree_vec8", g::GemvKernel::TreeVec8),
            ("sg_u32_wg256", g::GemvKernel::SgU32),
            ("sg_v4_wg256", g::GemvKernel::SgV4 { wg: 256 }),
            ("sg_v4_wg512", g::GemvKernel::SgV4 { wg: 512 }),
        ] {
            let (_, secs) = g::gemv_bf16_probe(ctx, &w, &x, n, k, 5, iters, kern).unwrap();
            let per = secs / iters as f64;
            eprintln!(
                "round {round} {name}: {:.3} ms/iter, {:.1} GB/s",
                per * 1e3,
                bytes / per / 1e9
            );
        }
    }
}

#[test]
fn sg_pk_entries_match_the_tree_vec8_kernel_bitwise() {
    let Some(ctx) = ctx_or_skip("sg_pk_bitwise") else {
        return;
    };
    if !sg32_or_skip("sg_pk_bitwise", ctx) {
        return;
    }
    let shapes = [
        (4096usize, 2048usize),
        (511, 2048),
        (300, 256),
        (33, 4096),
        (8, 64),
        (2, 2048),
    ];
    for (n, k) in shapes {
        let mut rng = Lcg(0x5657_9a17 ^ (n as u64) ^ ((k as u64) << 24));
        let w = rng.bf16_vec(n * k, 0.25);
        let x = rng.bf16_vec(k, 1.0);
        let (y_tree, _) =
            g::gemv_bf16_probe(ctx, &w, &x, n, k, 1, 1, g::GemvKernel::TreeVec8).unwrap();
        let (y_sg, _) =
            g::gemv_bf16_probe(ctx, &w, &x, n, k, 1, 1, g::GemvKernel::SgV4 { wg: 128 }).unwrap();
        assert_eq!(
            y_sg, y_tree,
            "SgV4 unpacked diverges from tree at n={n} k={k}"
        );
        for word_off in [0usize, 3, 128] {
            let want = pack_reference(&y_tree, word_off);
            for wg in [128u32, 256] {
                let got = run_sg_pk(ctx, &w, &x, n, k, wg, word_off);
                assert_eq!(
                    got, want,
                    "sg pk wg{wg} diverges from packed tree at n={n} k={k} off={word_off}"
                );
            }
        }
    }
}
