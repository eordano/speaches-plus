
#![cfg(feature = "wgpu")]

mod common;
use common::bf16_bits;
use common::env_usize;
use common::Rng;
use nv_kernels::wgpu_backend::dispatch::{self, Recorded};
use nv_kernels::wgpu_backend::WgpuContext;
use nv_models::gemma4_moe_wgpu::{bench, GemvBf16Params};

const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("attn q  (sliding)", 4096, 2816, 25),
    ("attn o  (sliding)", 2816, 4096, 25),
    ("mlp down", 2816, 2112, 30),
    ("lm_head (tied embed)", 262144, 2816, 1),
];

const ASYMPTOTE_GBS: f64 = 755.0;

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn spread(v: &[f64]) -> f64 {
    let lo = v.iter().cloned().fold(f64::MAX, f64::min);
    let hi = v.iter().cloned().fold(0f64, f64::max);
    (hi - lo) / median(v)
}

fn pack_pairs(src: &[u16]) -> Vec<u32> {
    let words = src.len().div_ceil(2).max(1);
    let mut out = vec![0u32; words.next_multiple_of(4)];
    for (i, v) in src.iter().enumerate() {
        out[i / 2] |= (*v as u32) << (16 * (i % 2));
    }
    out
}

struct Arm {
    rec: Recorded,
    _bufs: Vec<wgpu::Buffer>,
    bytes: f64,
    dispatches: usize,
}

impl Arm {
    fn timed(&mut self, ctx: &WgpuContext, inner: usize) -> f64 {
        let t0 = std::time::Instant::now();
        self.rec.replay_n(ctx, inner).expect("replay");
        ctx.poll_blocking().expect("drain");
        t0.elapsed().as_secs_f64()
    }
}

#[allow(clippy::too_many_arguments)]
fn build_arm(
    ctx: &WgpuContext,
    tag: &str,
    ws: &[Vec<u16>],
    n: usize,
    k: usize,
    chunk_rows: usize,
    x: &wgpu::Buffer,
) -> Arm {
    let src = bench::gemv_bf16_source();
    let mut rec = Recorded::new();
    let mut bufs = Vec::new();
    let y = dispatch::storage_zeroed(ctx, "nozi-y", (ws.len() * n * 4) as u64);
    let mut bytes = 0f64;
    let mut dispatches = 0usize;
    for (c, w) in ws.iter().enumerate() {
        let mut row0 = 0usize;
        while row0 < n {
            let rows = chunk_rows.min(n - row0);
            let wbuf = dispatch::storage_from_slice(
                ctx,
                &format!("{tag}-w{c}-{row0}"),
                &pack_pairs(&w[row0 * k..(row0 + rows) * k]),
            );
            let grid = dispatch::workgroup_count_1d(ctx, rows.div_ceil(2) as u64, 1);
            let p = dispatch::uniform_from(
                ctx,
                "nozi-p",
                &GemvBf16Params {
                    n_rows: rows as u32,
                    k_words: (k / 2) as u32,
                    groups_x: grid.0,
                    out_f32: 1,
                    w_row_words: (k / 2) as u32,
                    x_off_words: 0,
                    y_off_words: (c * n + row0) as u32,
                    wide: 1,
                    alpha: 1.0,
                    ..Default::default()
                },
            );
            rec.push(
                ctx,
                tag,
                &src,
                bench::GEMV_BF16_ENTRY,
                &[(0, &wbuf), (1, x), (2, &p), (3, &y)],
                grid,
            )
            .expect("record dispatch");
            bytes += (rows * k * 2) as f64;
            dispatches += 1;
            bufs.push(wbuf);
            bufs.push(p);
            row0 += rows;
        }
    }
    bufs.push(y);
    Arm {
        rec,
        _bufs: bufs,
        bytes,
        dispatches,
    }
}

#[test]
#[ignore = "wires ~1 GB per shape on a real adapter; --ignored --release"]
fn the_nozi_exemption_is_what_an_unlisted_dense_gemv_entry_would_pay() {
    let ctx = WgpuContext::shared().expect("this A/B needs a wgpu adapter; there is no skip path");
    assert!(
        std::env::var("NV_WGPU_NOZI").is_err(),
        "NV_WGPU_NOZI is pre-set on the runner; this test owns that knob and setting it \
         outside makes both arms the same pipeline"
    );
    assert!(
        nv_kernels::wgpu_backend::dispatch::nozi_entry_listed(bench::GEMV_BF16_ENTRY, false),
        "{} is no longer on NOZI_AUDITED_ENTRIES, so there is no exemption to price and the \
         shipped dense GEMV has quietly started paying this handicap itself",
        bench::GEMV_BF16_ENTRY
    );
    eprintln!("[nozi] adapter: {}", ctx.info.name);

    let target_mb = env_usize("NV_G4MOE_NOZI_STACK_MB", 768);
    let reps = env_usize("NV_G4MOE_NOZI_REPS", 9);
    let warm = env_usize("NV_G4MOE_NOZI_WARM", 3);
    assert!(reps >= 5, "fewer than 5 reps cannot show a spread");

    let mut token_exempt = 0f64;
    let mut token_zi = 0f64;
    eprintln!(
        "[nozi] {:<22} {:>9} {:>9} {:>8} {:>7} {:>7} {:>9}",
        "shape [out,in]", "exempt", "zero-init", "handicap", "sprd%", "ctl%", "workgroups"
    );

    for &(name, n, k, per_token) in SHAPES {
        let bytes_shape = (n * k * 2) as f64;
        let copies = (((target_mb as f64 * 1e6) / bytes_shape).ceil() as usize).max(1);
        let cap = ctx
            .caps
            .max_storage_buffer_binding_size
            .min(1u64 << 30)
            .max(4 << 20);
        let chunk_rows = (((cap / (k as u64 * 2)) as usize).clamp(2, n) & !1usize).clamp(2, n);

        let mut xr = Rng(0x51ed_270b_ee98_1c3f ^ (n as u64) << 20 ^ k as u64);
        let xh: Vec<u16> = (0..k).map(|_| bf16_bits(xr.next_f32())).collect();
        let x = dispatch::storage_from_slice(&ctx, "nozi-x", &pack_pairs(&xh));

        let ws: Vec<Vec<u16>> = (0..copies)
            .map(|c| {
                let mut rng = Rng(0xA1B2_C3D4u64.wrapping_add(c as u64 * 0x9e37_79b9_7f4a_7c15));
                (0..n * k)
                    .map(|_| bf16_bits(rng.next_f32() * 0.1))
                    .collect()
            })
            .collect();

        unsafe { std::env::remove_var("NV_WGPU_NOZI") };
        let mut exempt = build_arm(&ctx, "nozi-exempt", &ws, n, k, chunk_rows, &x);
        let mut ctl = build_arm(&ctx, "nozi-ctl", &ws, n, k, chunk_rows, &x);
        unsafe { std::env::set_var("NV_WGPU_NOZI", "0") };
        let mut zi = build_arm(&ctx, "nozi-zi", &ws, n, k, chunk_rows, &x);
        unsafe { std::env::remove_var("NV_WGPU_NOZI") };
        assert_eq!(
            exempt.dispatches, zi.dispatches,
            "{name}: the arms issue different dispatch counts"
        );

        let inner = env_usize(
            "NV_G4MOE_NOZI_INNER",
            ((2.0e9 / exempt.bytes).ceil() as usize).max(1),
        );
        for _ in 0..warm {
            exempt.timed(&ctx, 1);
            zi.timed(&ctx, 1);
            ctl.timed(&ctx, 1);
        }

        let mut r = Vec::with_capacity(reps);
        let mut te = Vec::with_capacity(reps);
        let mut tz = Vec::with_capacity(reps);
        for _ in 0..reps {
            let a = exempt.timed(&ctx, inner);
            let b = zi.timed(&ctx, inner);
            r.push(b / a);
            te.push(a);
            tz.push(b);
        }
        let mut rc = Vec::with_capacity(reps);
        for _ in 0..reps {
            let a = exempt.timed(&ctx, inner);
            let c = ctl.timed(&ctx, inner);
            rc.push(c / a);
        }
        let handicap = median(&r);
        let ctl_off = (median(&rc) - 1.0).abs();
        let sprd = spread(&te).max(spread(&tz));
        let rate_e = exempt.bytes * inner as f64 / median(&te) / 1e9;
        let rate_z = rate_e / handicap;
        assert!(
            rate_e < ASYMPTOTE_GBS,
            "{name}: the exempt arm read {rate_e:.1} GB/s, above the {ASYMPTOTE_GBS} GB/s \
             asymptote -- the stack is cache-resident and every rate here is fiction"
        );
        assert!(
            ctl_off < 0.05,
            "{name}: the null control ran {:.1}% off the exempt arm; this box moved under \
             the measurement",
            ctl_off * 100.0
        );
        eprintln!(
            "[nozi] {name:<22} {rate_e:>8.1}G {rate_z:>8.1}G {handicap:>8.4} {:>7.1} \
             {:>7.1} {:>9}   ({copies} x [{n},{k}], {} disp, inner {inner})",
            sprd * 100.0,
            ctl_off * 100.0,
            exempt.dispatches * inner,
            exempt.dispatches
        );
        let unit = |t: f64| t / inner as f64 / copies as f64 * per_token as f64;
        token_exempt += unit(median(&te));
        token_zi += unit(median(&te) * handicap);
    }

    eprintln!(
        "[nozi] over one token's dense GEMVs at these four shapes: exempt {:.3} ms, zero-init \
         {:.3} ms -- an unlisted entry pays {:.4}x before it computes anything different.",
        token_exempt * 1e3,
        token_zi * 1e3,
        token_zi / token_exempt
    );
    eprintln!(
        "[nozi] int8-candidate speedups were measured with this handicap equalized away \
         (current numbers: perf/runs.jsonl). Routed as new entries they would keep the \
         handicap, so divide any candidate ratio by {:.4} before planning against it.",
        token_zi / token_exempt
    );
    assert!(
        token_zi >= token_exempt * 0.999,
        "zero-init measured FASTER than the exemption ({:.4}x); a memset cannot make a \
         kernel faster, so this run measured contention rather than a build option",
        token_zi / token_exempt
    );
}
