
#![cfg(feature = "wgpu")]

mod common;
use common::bf16_bits;
use common::bf16_val;
use common::env_usize;
use common::Rng;
use nv_kernels::wgpu_backend::dispatch::{self, Recorded};
use nv_kernels::wgpu_backend::WgpuContext;
use nv_models::gemma4_moe_wgpu::{
    bench, dequantize_i8_row, quantize_i8_host, GemvBf16Params, GemvI8Params, HostI8Mat, I8_GS,
};

const ASYMPTOTE_GBS: f64 = 755.0;

const BF16_BPW: f64 = 2.0;
const I8_BPW: f64 = 1.0 + 2.0 / I8_GS as f64;

fn bar() -> f64 {
    I8_BPW / BF16_BPW
}

const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("attn q  (sliding)", 4096, 2816, 25),
    ("attn k  (sliding)", 2048, 2816, 25),
    ("attn v  (sliding)", 2048, 2816, 25),
    ("attn o  (sliding)", 2816, 4096, 25),
    ("attn q  (full)", 8192, 2816, 5),
    ("attn k  (full)", 1024, 2816, 5),
    ("attn o  (full)", 2816, 8192, 5),
    ("mlp gate", 2112, 2816, 30),
    ("mlp up", 2112, 2816, 30),
    ("mlp down", 2816, 2112, 30),
    ("lm_head (tied embed)", 262144, 2816, 1),
];

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn spread(v: &[f64]) -> f64 {
    let lo = v.iter().cloned().fold(f64::MAX, f64::min);
    let hi = v.iter().cloned().fold(f64::MIN, f64::max);
    (hi - lo) / median(v)
}

fn pack_pairs(src: &[u16]) -> Vec<u32> {
    let mut out = vec![0u32; src.len().div_ceil(2).max(1)];
    for (i, v) in src.iter().enumerate() {
        out[i / 2] |= (*v as u32) << (16 * (i % 2));
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fmt {
    Bf16,
    Bf16V4,
    I8,
    I8V4,
}

impl Fmt {
    fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Bf16V4 => "bf16.v4",
            Self::I8 => "int8",
            Self::I8V4 => "int8.v4",
        }
    }

    fn is_int8(self) -> bool {
        matches!(self, Self::I8 | Self::I8V4)
    }

    fn bpw(self) -> f64 {
        if self.is_int8() {
            I8_BPW
        } else {
            BF16_BPW
        }
    }

    fn source(self) -> String {
        match self {
            Self::Bf16 => bench::gemv_bf16_source(),
            Self::Bf16V4 => bench::gemv_bf16_v4_source(),
            Self::I8 => bench::gemv_i8_source(),
            Self::I8V4 => bench::gemv_i8_v4_source(),
        }
    }

    fn entry(self) -> &'static str {
        match self {
            Self::Bf16 => bench::GEMV_BF16_ENTRY,
            Self::Bf16V4 => bench::GEMV_BF16_V4_ENTRY,
            Self::I8 => bench::GEMV_I8_ENTRY,
            Self::I8V4 => bench::GEMV_I8_V4_ENTRY,
        }
    }

    fn row_words(self, k: usize) -> usize {
        match self {
            Self::Bf16 | Self::Bf16V4 => k / 2,
            Self::I8 => k / 4,
            Self::I8V4 => k / 16,
        }
    }

    fn wide(self) -> u32 {
        u32::from(self == Self::Bf16V4)
    }
}

struct Arm {
    fmt: Fmt,
    rec: Recorded,

    _bufs: Vec<wgpu::Buffer>,
    y: wgpu::Buffer,
    bytes_per_replay: f64,
    dispatches: usize,
}

impl Arm {
    fn timed(&mut self, ctx: &WgpuContext, inner: usize) -> f64 {
        let t0 = std::time::Instant::now();
        self.rec.replay_n(ctx, inner).expect("replay");
        ctx.poll_blocking().expect("drain");
        t0.elapsed().as_secs_f64()
    }

    fn rate_gbs(&self, secs: f64, inner: usize) -> f64 {
        self.bytes_per_replay * inner as f64 / secs / 1e9
    }
}

#[allow(clippy::too_many_arguments)]
fn build_arm(
    ctx: &WgpuContext,
    fmt: Fmt,
    tag: &str,
    ws: &[Vec<u16>],
    qs: &[HostI8Mat],
    n: usize,
    k: usize,
    chunk_rows: usize,
    x: &wgpu::Buffer,
) -> Arm {
    let src = fmt.source();
    let entry = fmt.entry();
    let groups = k / I8_GS;
    let copies = ws.len();
    let mut rec = Recorded::new();
    let mut bufs = Vec::new();
    let y = dispatch::storage_zeroed(ctx, "i8ab-y", (copies * n * 4) as u64);
    let mut bytes = 0f64;
    let mut dispatches = 0usize;
    for c in 0..copies {
        let mut row0 = 0usize;
        while row0 < n {
            let rows = chunk_rows.min(n - row0);
            let wbuf = if fmt.is_int8() {
                dispatch::storage_from_slice(
                    ctx,
                    &format!("{tag}-w{c}-{row0}"),
                    &qs[c].q[row0 * (k / 4)..(row0 + rows) * (k / 4)],
                )
            } else {
                dispatch::storage_from_slice(
                    ctx,
                    &format!("{tag}-w{c}-{row0}"),
                    &pack_pairs(&ws[c][row0 * k..(row0 + rows) * k]),
                )
            };
            let pairs = rows.div_ceil(2) as u64;
            let grid = dispatch::workgroup_count_1d(ctx, pairs, 1);
            let rw = fmt.row_words(k) as u32;
            let p = if fmt.is_int8() {
                dispatch::uniform_from(
                    ctx,
                    "i8ab-p",
                    &GemvI8Params {
                        n_rows: rows as u32,
                        k_words: rw,
                        groups_x: grid.0,
                        out_f32: 1,
                        w_row_words: rw,
                        x_off_words: 0,
                        y_off_words: (c * n + row0) as u32,
                        s_row_elems: groups as u32,
                        alpha: 1.0,
                        ..Default::default()
                    },
                )
            } else {
                dispatch::uniform_from(
                    ctx,
                    "i8ab-p",
                    &GemvBf16Params {
                        n_rows: rows as u32,
                        k_words: rw,
                        groups_x: grid.0,
                        out_f32: 1,
                        w_row_words: rw,
                        x_off_words: 0,
                        y_off_words: (c * n + row0) as u32,
                        wide: fmt.wide(),
                        alpha: 1.0,
                        ..Default::default()
                    },
                )
            };
            if fmt.is_int8() {
                let sbuf = dispatch::storage_from_slice(
                    ctx,
                    &format!("{tag}-s{c}-{row0}"),
                    &pack_pairs(&qs[c].scales[row0 * groups..(row0 + rows) * groups]),
                );
                rec.push(
                    ctx,
                    tag,
                    &src,
                    entry,
                    &[(0, &wbuf), (1, x), (2, &p), (3, &y), (4, &sbuf)],
                    grid,
                )
                .expect("record int8 dispatch");
                bufs.push(sbuf);
            } else {
                rec.push(
                    ctx,
                    tag,
                    &src,
                    entry,
                    &[(0, &wbuf), (1, x), (2, &p), (3, &y)],
                    grid,
                )
                .expect("record bf16 dispatch");
            }
            bytes += (rows * k) as f64 * fmt.bpw();
            dispatches += 1;
            bufs.push(wbuf);
            bufs.push(p);
            row0 += rows;
        }
    }
    Arm {
        fmt,
        rec,
        _bufs: bufs,
        y,
        bytes_per_replay: bytes,
        dispatches,
    }
}

fn ref_bf16(w: &[u16], k: usize, x: &[u16], probe: &[usize]) -> Vec<f64> {
    probe
        .iter()
        .map(|&r| {
            (0..k)
                .map(|j| bf16_val(w[r * k + j]) as f64 * bf16_val(x[j]) as f64)
                .sum()
        })
        .collect()
}

fn ref_i8(q: &HostI8Mat, x: &[u16], probe: &[usize]) -> Vec<f64> {
    probe
        .iter()
        .map(|&r| {
            let row = dequantize_i8_row(q, r);
            row.iter()
                .enumerate()
                .map(|(j, v)| *v as f64 * bf16_val(x[j]) as f64)
                .sum()
        })
        .collect()
}

fn assert_matches(label: &str, got: &[f32], reference: &[f64], probe: &[usize], tol: f64) {
    let mag = reference.iter().fold(0f64, |a, b| a.max(b.abs()));
    assert!(
        mag > 1e-6,
        "{label}: the f64 reference is degenerate (max |ref| {mag:.3e}); this comparison \
         would pass on zeros"
    );
    let out_mag = probe.iter().fold(0f64, |a, &r| a.max(got[r].abs() as f64));
    assert!(
        out_mag > 1e-6,
        "{label}: the kernel wrote nothing but zeros (max |y| {out_mag:.3e})"
    );
    let mut worst = 0f64;
    let mut worst_at = 0usize;
    for (i, &r) in probe.iter().enumerate() {
        let rel = (got[r] as f64 - reference[i]).abs() / mag;
        if rel > worst {
            worst = rel;
            worst_at = r;
        }
    }
    assert!(
        worst < tol,
        "{label}: row {worst_at} is {worst:.3e} off the f64 reference (tolerance {tol:.1e}); \
         the kernel does not compute the dot product it claims to"
    );
}
#[test]
#[ignore = "wires several GB per shape on a real adapter; --ignored --release"]
fn int8_dense_gemv_beats_the_bf16_byte_ratio_at_every_26b_shape() {
    unsafe { std::env::set_var("NV_WGPU_NOZI", "0") };

    let ctx = WgpuContext::shared().expect("this A/B needs a wgpu adapter; there is no skip path");
    eprintln!(
        "[i8ab] adapter: {} | max storage binding {:.2} GiB | NOZI equalized OFF for all arms",
        ctx.info.name,
        ctx.caps.max_storage_buffer_binding_size as f64 / (1u64 << 30) as f64
    );

    let target_mb = env_usize("NV_G4MOE_I8AB_STACK_MB", 768);
    let reps = env_usize("NV_G4MOE_I8AB_REPS", 9);
    let warm = env_usize("NV_G4MOE_I8AB_WARM", 3);
    let probe_rows = env_usize("NV_G4MOE_I8AB_PROBE", 48);
    assert!(reps >= 5, "fewer than 5 reps cannot show a spread");

    const ARMS: [Fmt; 4] = [Fmt::Bf16, Fmt::Bf16V4, Fmt::I8, Fmt::I8V4];
    let mut losses: Vec<String> = Vec::new();
    let mut by_shape: std::collections::HashMap<&str, f64> = Default::default();
    let mut shapes_seen = 0usize;
    let mut token: [f64; 4] = [0.0; 4];
    let mut token_best_bf16 = 0f64;
    let mut token_best_i8 = 0f64;

    eprintln!(
        "[i8ab] {:<22} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8} {:>7} {:>7} {:>7}",
        "shape [out,in]",
        "bf16",
        "bf16.v4",
        "int8",
        "int8.v4",
        "R best",
        "bar",
        "sprd%",
        "A/A'%",
        "unpr%"
    );

    for &(name, n, k, per_token) in SHAPES {
        assert!(
            k.is_multiple_of(I8_GS),
            "{name}: k={k} not a multiple of 32"
        );
        assert!(
            k.is_multiple_of(16),
            "{name}: k={k} cannot feed a vec4 int8 load"
        );
        let bytes_bf16 = (n * k) as f64 * BF16_BPW;
        let copies = (((target_mb as f64 * 1e6) / bytes_bf16).ceil() as usize).max(1);
        let cap = ctx
            .caps
            .max_storage_buffer_binding_size
            .min(1u64 << 30)
            .max(4 << 20);
        let chunk_rows = (((cap / (k as u64 * 2)) as usize).clamp(2, n) & !1usize).clamp(2, n);

        let mut xr = Rng(0x51ed_270b_ee98_1c3f ^ (n as u64) << 20 ^ k as u64);
        let xh: Vec<u16> = (0..k).map(|_| bf16_bits(xr.next_f32())).collect();
        let x = dispatch::storage_from_slice(&ctx, "i8ab-x", &pack_pairs(&xh));

        let ws: Vec<Vec<u16>> = (0..copies)
            .map(|c| {
                let mut rng = Rng(0xA1B2_C3D4u64.wrapping_add(c as u64 * 0x9e37_79b9_7f4a_7c15));
                (0..n * k)
                    .map(|_| bf16_bits(rng.next_f32() * 0.1))
                    .collect()
            })
            .collect();
        let qs: Vec<HostI8Mat> = ws.iter().map(|w| quantize_i8_host(w, n, k)).collect();

        let mut arms: Vec<Arm> = ARMS
            .iter()
            .map(|&f| build_arm(&ctx, f, "i8ab", &ws, &qs, n, k, chunk_rows, &x))
            .collect();
        let mut ctl = build_arm(&ctx, Fmt::Bf16, "i8ab-ctl", &ws, &qs, n, k, chunk_rows, &x);
        for a in &arms {
            assert_eq!(
                a.dispatches, arms[0].dispatches,
                "{name}: {:?} issues a different dispatch count, so the fixed per-dispatch \
                 cost does not cancel in the ratio",
                a.fmt
            );
        }

        let inner = env_usize(
            "NV_G4MOE_I8AB_INNER",
            ((2.0e9 / arms[0].bytes_per_replay).ceil() as usize).max(1),
        );

        for _ in 0..warm {
            for a in arms.iter_mut() {
                a.timed(&ctx, 1);
            }
            ctl.timed(&ctx, 1);
        }

        let stride = (n / probe_rows).max(1);
        let probe: Vec<usize> = (0..n).step_by(stride).take(probe_rows).collect();
        let ra = ref_bf16(&ws[0], k, &xh, &probe);
        let rb = ref_i8(&qs[0], &xh, &probe);
        for a in &arms {
            let y: Vec<f32> = dispatch::read_back(&ctx, &a.y, n).expect("read y");
            let reference = if a.fmt.is_int8() { &rb } else { &ra };
            assert_matches(
                &format!("{name} {}", a.fmt.label()),
                &y,
                reference,
                &probe,
                2e-3,
            );
        }
        let mut num = 0f64;
        let mut den = 0f64;
        for i in 0..probe.len() {
            num += (ra[i] - rb[i]).powi(2);
            den += ra[i] * ra[i];
        }
        let dot_rel_rms = (num / den.max(1e-30)).sqrt();

        let mut speedup = vec![1.0f64; 4];
        let mut secs = vec![0.0f64; 4];
        let mut sprd = 0f64;
        for i in 1..4 {
            let (lo, hi) = arms.split_at_mut(i);
            let a0 = &mut lo[0];
            let ai = &mut hi[0];
            let mut r = Vec::with_capacity(reps);
            let mut t0v = Vec::with_capacity(reps);
            let mut tiv = Vec::with_capacity(reps);
            for _ in 0..reps {
                let x0 = a0.timed(&ctx, inner);
                let xi = ai.timed(&ctx, inner);
                r.push(x0 / xi);
                t0v.push(x0);
                tiv.push(xi);
            }
            speedup[i] = median(&r);
            secs[i] = median(&tiv);
            sprd = sprd.max(spread(&tiv)).max(spread(&t0v));
            if i == 1 {
                secs[0] = median(&t0v);
            }
        }
        let mut rc = Vec::with_capacity(reps);
        for _ in 0..reps {
            let x0 = arms[0].timed(&ctx, inner);
            let xc = ctl.timed(&ctx, inner);
            rc.push(x0 / xc);
        }
        let drift = (median(&rc) - 1.0).abs();
        let raw: Vec<f64> = arms
            .iter()
            .zip(secs.iter())
            .map(|(a, s)| a.rate_gbs(*s, inner))
            .collect();
        let secs: Vec<f64> = speedup.iter().map(|s| secs[0] / s).collect();
        let rates: Vec<f64> = arms
            .iter()
            .zip(secs.iter())
            .map(|(a, s)| a.rate_gbs(*s, inner))
            .collect();
        let unpaired = (0..4)
            .map(|i| (raw[i] - rates[i]).abs() / rates[i])
            .fold(0f64, f64::max);

        for (a, r) in arms.iter().zip(raw.iter()) {
            assert!(
                *r < ASYMPTOTE_GBS,
                "{name}: the {} arm read {r:.1} GB/s, above the {ASYMPTOTE_GBS} GB/s \
                 asymptote -- the stack ({:.0} MB over {} dispatches) is cache-resident and \
                 every rate here is fiction",
                a.fmt.label(),
                a.bytes_per_replay / 1e6,
                a.dispatches
            );
        }

        let bf16_i = if speedup[1] > speedup[0] { 1 } else { 0 };
        let i8_i = if speedup[3] > speedup[2] { 3 } else { 2 };
        let win = speedup[i8_i] / speedup[bf16_i];
        let ratio = bar() * win;

        eprintln!(
            "[i8ab] {name:<22} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {ratio:>8.3} {:>8.3} \
             {:>7.1} {:>7.1} {:>7.1}   ({copies} x [{n},{k}], {:.0} MB bf16, {} disp, inner \
             {inner}, best {} -> {} {:.3}x, dot rel-rms {dot_rel_rms:.2e})",
            rates[0],
            rates[1],
            rates[2],
            rates[3],
            bar(),
            sprd * 100.0,
            drift * 100.0,
            unpaired * 100.0,
            arms[0].bytes_per_replay / 1e6,
            arms[0].dispatches,
            arms[bf16_i].fmt.label(),
            arms[i8_i].fmt.label(),
            win,
        );

        let unit = |i: usize| secs[i] / inner as f64 / copies as f64 * per_token as f64;
        for i in 0..4 {
            token[i] += unit(i);
        }
        token_best_bf16 += unit(bf16_i);
        token_best_i8 += unit(i8_i);
        by_shape.insert(name, ratio);
        shapes_seen += 1;
        if ratio <= bar() {
            losses.push(format!(
                "{name}: best-bf16 {} is {win:.3}x best-int8 {} -- R ratio {ratio:.3} <= bar \
                 {:.5}",
                arms[bf16_i].fmt.label(),
                arms[i8_i].fmt.label(),
                bar()
            ));
        }
        assert!(
            drift < 0.08,
            "{name}: the null control ran {:.1}% off arm 0 on paired reps; this box moved \
             under the measurement and nothing here is attributable",
            drift * 100.0
        );
    }

    assert_eq!(shapes_seen, SHAPES.len(), "a shape was skipped");
    eprintln!(
        "[i8ab] one token's dense-GEMV seconds, by arm: bf16 {:.3} ms | bf16.v4 {:.3} ms | \
         int8 {:.3} ms | int8.v4 {:.3} ms",
        token[0] * 1e3,
        token[1] * 1e3,
        token[2] * 1e3,
        token[3] * 1e3
    );
    eprintln!(
        "[i8ab] VERDICT best-bf16 {:.3} ms -> best-int8 {:.3} ms = {:.3}x on the dense stream \
         alone (attention, the MoE, the norms and the router are unchanged and not in it)",
        token_best_bf16 * 1e3,
        token_best_i8 * 1e3,
        token_best_bf16 / token_best_i8
    );
    for l in &losses {
        eprintln!("[i8ab] AT-OR-UNDER-BAR {l}");
    }

    for (a, b) in [
        ("attn k  (sliding)", "attn v  (sliding)"),
        ("mlp gate", "mlp up"),
    ] {
        let (ra, rb) = (by_shape[a], by_shape[b]);
        let disagree = (ra - rb).abs() / ra.min(rb);
        eprintln!(
            "[i8ab] replicate {a} {ra:.3} vs {b} {rb:.3} -> {:.1}%",
            disagree * 100.0
        );
        assert!(
            disagree < 0.10,
            "{a} and {b} are the same shape and measured {ra:.3} vs {rb:.3} ({:.1}% apart); \
             this run cannot resolve a kernel difference and every ratio in it is weather",
            disagree * 100.0
        );
    }

    let agg = bar() * (token_best_bf16 / token_best_i8);
    eprintln!(
        "[i8ab] AGGREGATE R_int8/R_bf16 = {agg:.4} against bar {:.5}; {} of {} shapes at or \
         under the bar",
        bar(),
        losses.len(),
        SHAPES.len()
    );
    assert!(
        agg > bar(),
        "over a whole token's dense GEMVs int8 realizes {agg:.4} of the bf16 rate against a \
         {:.5} byte ratio -- it is a byte-cheaper SLOWER path, which is the failure the rules \
         table already records twice",
        bar()
    );
    const MAX_AT_BAR_SHAPES: usize = 1;
    assert!(
        losses.len() <= MAX_AT_BAR_SHAPES,
        "{} of {} dense shapes are at or under the crossover bar {:.5}, was {MAX_AT_BAR_SHAPES}",
        losses.len(),
        SHAPES.len(),
        bar()
    );
}

#[test]
fn gemma4_moe_int8_dense_shapes_match_the_checkpoint() {
    const HIDDEN: usize = 2816;
    const N_Q: usize = 16;
    const HD_SLIDING: usize = 256;
    const HD_FULL: usize = 512;
    const KV_SLIDING: usize = 8;
    const KV_FULL: usize = 2;
    const INTER: usize = 2112;
    const VOCAB: usize = 262144;
    const LAYERS: usize = 30;
    const FULL_LAYERS: usize = 5;
    let sliding = LAYERS - FULL_LAYERS;

    let want: Vec<(&str, usize, usize, usize)> = vec![
        ("attn q  (sliding)", N_Q * HD_SLIDING, HIDDEN, sliding),
        (
            "attn k  (sliding)",
            KV_SLIDING * HD_SLIDING,
            HIDDEN,
            sliding,
        ),
        (
            "attn v  (sliding)",
            KV_SLIDING * HD_SLIDING,
            HIDDEN,
            sliding,
        ),
        ("attn o  (sliding)", HIDDEN, N_Q * HD_SLIDING, sliding),
        ("attn q  (full)", N_Q * HD_FULL, HIDDEN, FULL_LAYERS),
        ("attn k  (full)", KV_FULL * HD_FULL, HIDDEN, FULL_LAYERS),
        ("attn o  (full)", HIDDEN, N_Q * HD_FULL, FULL_LAYERS),
        ("mlp gate", INTER, HIDDEN, LAYERS),
        ("mlp up", INTER, HIDDEN, LAYERS),
        ("mlp down", HIDDEN, INTER, LAYERS),
        ("lm_head (tied embed)", VOCAB, HIDDEN, 1),
    ];
    assert_eq!(want.len(), SHAPES.len(), "shape table lost an entry");
    for (a, b) in want.iter().zip(SHAPES.iter()) {
        assert_eq!(a, b, "shape table disagrees with the derived geometry");
    }

    let bf16: f64 = SHAPES
        .iter()
        .map(|(_, n, k, c)| (n * k * c) as f64 * BF16_BPW)
        .sum();
    let i8b: f64 = SHAPES
        .iter()
        .map(|(_, n, k, c)| (n * k * c) as f64 * I8_BPW)
        .sum();
    assert!(
        (bf16 / 1e9 - 4.767).abs() < 0.01,
        "dense bf16 bytes per token came out {:.4} GB, not the graph's 4.767",
        bf16 / 1e9
    );
    assert!(
        (i8b / bf16 - bar()).abs() < 1e-12,
        "the byte ratio is not the encoding ratio"
    );
    eprintln!(
        "[i8ab] dense per token: bf16 {:.4} GB -> int8 {:.4} GB (saves {:.4} GB, {:.4}x); \
         whole token 5.616 -> {:.4} GB, ceiling at 738.5 GB/s moves {:.1} -> {:.1} tok/s",
        bf16 / 1e9,
        i8b / 1e9,
        (bf16 - i8b) / 1e9,
        i8b / bf16,
        (5.616e9 - (bf16 - i8b)) / 1e9,
        738.5e9 / 5.616e9,
        738.5e9 / (5.616e9 - (bf16 - i8b)),
    );
}
