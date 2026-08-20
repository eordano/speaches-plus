#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemv_nvfp4 as g;
use nv_kernels::wgpu_backend::kernels::gemv_nvfp4::lin;
mod common;
use common::LcgShift33W4a16Packs as Lcg;
use common::Params as W4Params;

fn ctx(test: &str) -> &'static WgpuContext {
    let c = WgpuContext::shared().unwrap_or_else(|e| panic!("{test}: no wgpu adapter: {e}"));
    let q = c.qualify();
    assert!(q.qualified, "{test}: adapter not qualified: {:?}", q.reason);
    eprintln!("{test}: {}", c.summary());
    c
}

struct Inputs {
    w_words: Vec<u32>,
    ws_bytes: Vec<u8>,
    ws_words: Vec<u32>,
    ws_lin_words: Vec<u32>,
    x_words: Vec<u32>,
    x_i8_words: Vec<u32>,
    xs_words: Vec<u32>,
    weight_bytes: f64,
}

fn make_inputs(n: usize, k: usize, seed: u64) -> Inputs {
    let mut rng = Lcg(seed);
    let k_blocks = k / 16;
    let w_words: Vec<u32> = (0..n * k / 8).map(|_| rng.next_u32()).collect();
    let scale_len = g::swizzled_scale_len(n, k_blocks);
    let scale_byte = |r: &mut Lcg| (0x30u32 | (r.next_u32() & 0x0f)) as u8;
    let ws_bytes: Vec<u8> = (0..scale_len).map(|_| scale_byte(&mut rng)).collect();
    let pack = |c: &[u8]| {
        let mut w = 0u32;
        for (i, b) in c.iter().enumerate() {
            w |= (*b as u32) << (8 * i);
        }
        w
    };
    let ws_words: Vec<u32> = ws_bytes.chunks(4).map(pack).collect();
    let ws_lin_words = lin::linear_scales_from_swizzled(&ws_bytes, n, k_blocks);
    let x_words: Vec<u32> = (0..k / 8).map(|_| rng.next_u32()).collect();
    let x_i8_words = lin::x_i8_from_packed(&x_words);
    let xs_bytes: Vec<u8> = (0..k_blocks.div_ceil(4) * 4)
        .map(|_| scale_byte(&mut rng))
        .collect();
    let xs_words: Vec<u32> = xs_bytes.chunks(4).map(pack).collect();
    let weight_bytes = (n * k / 2 + scale_len + k / 2 + k_blocks) as f64;
    Inputs {
        w_words,
        ws_bytes,
        ws_words,
        ws_lin_words,
        x_words,
        x_i8_words,
        xs_words,
        weight_bytes,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cand {
    Tree,
    Sg,
    Swz,
    Lin,
    NoScale,
    NoDec,
    XPre,
    V3,
    V3NoDec,
    V3Stream,
}

impl Cand {
    fn entry(self) -> &'static str {
        match self {
            Self::Tree => g::GEMV_ENTRY,
            Self::Sg => g::SG_GEMV_ENTRY,
            Self::Swz => lin::SWZ_ENTRY,
            Self::Lin => lin::LIN_ENTRY,
            Self::NoScale => lin::NOSCALE_ENTRY,
            Self::NoDec => lin::NODEC_ENTRY,
            Self::XPre => lin::XPRE_ENTRY,
            Self::V3 => lin::V3_ENTRY,
            Self::V3NoDec => lin::V3_NODEC_ENTRY,
            Self::V3Stream => lin::V3_STREAM_ENTRY,
        }
    }

    fn source(self) -> String {
        match self {
            Self::Tree => g::gemv_source(),
            Self::Sg => g::sg_gemv_source(),
            _ => lin::source(),
        }
    }

    fn rows_per_group(self) -> u32 {
        match self {
            Self::Tree => 1,
            Self::Sg => g::SG_ROWS_PER_GROUP,
            _ => lin::ROWS_PER_GROUP,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Tree => "tree  wg256 stride256 vec2 swz-scales",
            Self::Sg => "sg    wg128 stride256 vec2 swz-scales",
            Self::Swz => "swz   wg128 stride256 vec2 swz-scales (ctl)",
            Self::Lin => "lin   wg128 stride256 vec2 LIN-scales",
            Self::NoScale => "P:noscale  no w-scale load  (INVALID)",
            Self::NoDec => "P:nodec    no e2m1 SWAR      (INVALID)",
            Self::XPre => "P:xpre     no x-side SWAR    (INVALID)",
            Self::V3 => "v3    wg128 stride32  vec4 LIN+xi8",
            Self::V3NoDec => "P:v3-nodec no e2m1 SWAR      (INVALID)",
            Self::V3Stream => "P:v3-stream loads only       (INVALID)",
        }
    }

    fn valid_numerics(self) -> bool {
        !matches!(
            self,
            Self::NoScale | Self::NoDec | Self::XPre | Self::V3NoDec | Self::V3Stream
        )
    }

    fn bit_exact_with_tree(self) -> bool {
        matches!(self, Self::Tree | Self::Sg | Self::Swz | Self::Lin)
    }
}

struct Bufs {
    w: wgpu::Buffer,
    w4: wgpu::Buffer,
    ws: wgpu::Buffer,
    wl: wgpu::Buffer,
}

fn probe(
    ctx: &WgpuContext,
    inputs: &Inputs,
    variant: Cand,
    n: usize,
    k: usize,
    warmup: usize,
    iters: usize,
) -> (Vec<u16>, f64) {
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, variant.rows_per_group());
    let params = lin::params(1.0, n, k, groups.0);
    let pbuf = dispatch::uniform_from(ctx, "lin-params", &params);
    let mk_bufs = |tag: &str| Bufs {
        w: dispatch::storage_from_slice(ctx, tag, &inputs.w_words),
        w4: dispatch::storage_from_slice(ctx, tag, &inputs.w_words),
        ws: dispatch::storage_from_slice(ctx, tag, &inputs.ws_words),
        wl: dispatch::storage_from_slice(ctx, tag, &inputs.ws_lin_words),
    };
    let b0 = mk_bufs("lin-rep0");
    let b1 = mk_bufs("lin-rep1");
    let x = dispatch::storage_from_slice(ctx, "lin-x", &inputs.x_words);
    let xi8 = dispatch::storage_from_slice(ctx, "lin-xi8", &inputs.x_i8_words);
    let xs = dispatch::storage_from_slice(ctx, "lin-xs", &inputs.xs_words);
    let y = dispatch::storage_zeroed(ctx, "lin-y", (n * 4) as u64);

    let pipeline =
        dispatch::cached_compute_pipeline(ctx, variant.label(), &variant.source(), variant.entry())
            .expect("pipeline");

    let mk = |b: &Bufs| {
        let mut e: Vec<(u32, &wgpu::Buffer)> = Vec::new();
        match variant {
            Cand::V3 | Cand::V3NoDec | Cand::V3Stream => {
                e.push((3, &xs));
                e.push((4, &pbuf));
                e.push((5, &y));
                e.push((6, &b.wl));
                e.push((7, &b.w4));
                e.push((8, &xi8));
            }
            Cand::NoScale => {
                e.push((0, &b.w));
                e.push((2, &x));
                e.push((3, &xs));
                e.push((4, &pbuf));
                e.push((5, &y));
            }
            Cand::Lin | Cand::NoDec | Cand::XPre => {
                e.push((0, &b.w));
                e.push((2, &x));
                e.push((3, &xs));
                e.push((4, &pbuf));
                e.push((5, &y));
                e.push((6, &b.wl));
            }
            _ => {
                e.push((0, &b.w));
                e.push((1, &b.ws));
                e.push((2, &x));
                e.push((3, &xs));
                e.push((4, &pbuf));
                e.push((5, &y));
            }
        }
        dispatch::bind_group(ctx, &pipeline, &e)
    };
    let group0 = mk(&b0);
    let group1 = mk(&b1);

    let submit = |count: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for i in 0..count {
                pass.set_bind_group(0, if i % 2 == 0 { &group0 } else { &group1 }, &[]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
        }
        ctx.queue.submit([enc.finish()]);
    };
    submit(warmup.max(1));
    ctx.poll_blocking().expect("warmup poll");

    let start = std::time::Instant::now();
    submit(iters);
    ctx.poll_blocking().expect("timed poll");
    let secs = start.elapsed().as_secs_f64();

    let words: Vec<u32> = dispatch::read_back(ctx, &y, n).expect("read back");
    let out = words.iter().map(|w| (*w & 0xffff) as u16).collect();
    (out, secs)
}

fn probe_int8(ctx: &WgpuContext, n: usize, k: usize, iters: usize, seed: u64) -> (f64, f64) {
    use nv_kernels::wgpu_backend::kernels::quant_gemv as q;
    let mut rng = Lcg(seed);
    let w_words: Vec<u32> = (0..n * k / 4).map(|_| rng.next_u32()).collect();
    let x_words: Vec<u32> = (0..k / 2).map(|_| rng.next_u32() & 0x3f80_3f80).collect();
    let scales: Vec<f32> = (0..n).map(|_| 1.0 / 127.0).collect();
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, q::SG_ROWS_PER_GROUP);
    let params = q::params_for(n, k, 0, groups.0);
    let pbuf = dispatch::uniform_from(ctx, "i8-p", &params);
    let w0 = dispatch::storage_from_slice(ctx, "i8-w0", &w_words);
    let w1 = dispatch::storage_from_slice(ctx, "i8-w1", &w_words);
    let s0 = dispatch::storage_from_slice(ctx, "i8-s0", &scales);
    let s1 = dispatch::storage_from_slice(ctx, "i8-s1", &scales);
    let x = dispatch::storage_from_slice(ctx, "i8-x", &x_words);
    let y = dispatch::storage_zeroed(ctx, "i8-y", (n * 4) as u64);
    let pipeline = dispatch::cached_compute_pipeline(ctx, "i8-sg", &q::source(), q::INT8_SG_ENTRY)
        .expect("int8 pipeline");
    let g0 = dispatch::bind_group(
        ctx,
        &pipeline,
        &[(0, &w0), (1, &s0), (2, &x), (3, &y), (4, &pbuf)],
    );
    let g1 = dispatch::bind_group(
        ctx,
        &pipeline,
        &[(0, &w1), (1, &s1), (2, &x), (3, &y), (4, &pbuf)],
    );
    let submit = |count: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for i in 0..count {
                pass.set_bind_group(0, if i % 2 == 0 { &g0 } else { &g1 }, &[]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
        }
        ctx.queue.submit([enc.finish()]);
    };
    submit(10);
    ctx.poll_blocking().expect("warmup");
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        submit(iters);
        ctx.poll_blocking().expect("timed");
        best = best.min(t.elapsed().as_secs_f64());
    }
    let bytes = (n * k + n * 4 + k * 2) as f64;
    (
        best * 1e3 / iters as f64,
        bytes * iters as f64 / best / 1.0e9,
    )
}

fn probe_w4a16(
    ctx: &WgpuContext,
    n: usize,
    k: usize,
    lanes: u32,
    wg: u32,
    iters: usize,
    seed: u64,
) -> (f64, f64) {
    use nv_kernels::wgpu_backend::kernels::gemv_w4a16 as gw;
    const GS: usize = 32;
    let mut rng = Lcg(seed);
    let packed: Vec<u32> = (0..n * k / 8).map(|_| rng.next_u32()).collect();
    let scales: Vec<u32> = (0..n * k / GS)
        .map(|_| 0x3f00 | (rng.next_u32() & 0x7f))
        .collect();
    let x: Vec<u32> = (0..k / 2).map(|_| rng.next_u32() & 0x3f80_3f80).collect();
    let rows_per_group = wg / lanes;
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, rows_per_group);
    let params = W4Params {
        n_rows: n as u32,
        k_elems: k as u32,
        gs: GS as u32,
        w_row_words: (k / 8) as u32,
        scale_row_stride: (k / GS) as u32,
        groups_x: groups.0,
    };
    let p = dispatch::uniform_from(ctx, "w4-p", &params);
    let w0 = dispatch::storage_from_slice(ctx, "w4-w0", &packed);
    let w1 = dispatch::storage_from_slice(ctx, "w4-w1", &packed);
    let s0 = dispatch::storage_from_slice(ctx, "w4-s0", &scales);
    let s1 = dispatch::storage_from_slice(ctx, "w4-s1", &scales);
    let xb = dispatch::storage_from_slice(ctx, "w4-x", &x);
    let y = dispatch::storage_zeroed(ctx, "w4-y", (n * 4) as u64);
    let src = gw::sg_source(lanes, wg);
    let pipeline = dispatch::cached_compute_pipeline(ctx, "w4-sg", &src, gw::SG_ENTRY)
        .expect("w4a16 sg pipeline");
    let g0 = dispatch::bind_group(
        ctx,
        &pipeline,
        &[(0, &w0), (1, &s0), (2, &xb), (3, &y), (4, &p)],
    );
    let g1 = dispatch::bind_group(
        ctx,
        &pipeline,
        &[(0, &w1), (1, &s1), (2, &xb), (3, &y), (4, &p)],
    );
    let submit = |count: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for i in 0..count {
                pass.set_bind_group(0, if i % 2 == 0 { &g0 } else { &g1 }, &[]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
        }
        ctx.queue.submit([enc.finish()]);
    };
    submit(10);
    ctx.poll_blocking().expect("warmup");
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        submit(iters);
        ctx.poll_blocking().expect("timed");
        best = best.min(t.elapsed().as_secs_f64());
    }
    let bytes = (n * k / 2 + n * (k / GS) * 4 + k * 2) as f64;
    (
        best * 1e3 / iters as f64,
        bytes * iters as f64 / best / 1.0e9,
    )
}

#[test]
fn affine_int4_clears_the_crossover_on_the_same_shapes_where_nvfp4_cannot() {
    let ctx = ctx("w4a16_vs_nvfp4_31b");
    eprintln!("NOTE: run this test alone; see the ladder test's note on replica-buffer pressure.");
    assert!(g::sg32_ok(ctx), "adapter is not 32-wide by probe");
    let shapes = [
        ("gate_up", 43008usize, 5376usize, 50usize),
        ("down", 5376, 21504, 50),
        ("qkv", 16384, 5376, 50),
        ("o", 5376, 8192, 100),
    ];
    let mut cleared = 0usize;
    for (tag, n, k, iters) in shapes {
        let mut best = (0.0f64, 0.0f64, 0u32);
        for lanes in [8u32, 16, 32] {
            let (ms, gbps) = probe_w4a16(ctx, n, k, lanes, 256, iters, 0xbeef ^ (n as u64));
            eprintln!(
                "w4a16 {tag:<8} n={n:<6} k={k:<6} gs=32 | sg_x{lanes}_wg256 {ms:>8.4} ms {gbps:>7.1} GB/s {:>5.1}% roofline",
                100.0 * gbps / ROOFLINE_GBPS
            );
            if gbps > best.1 {
                best = (ms, gbps, lanes);
            }
        }
        if best.1 > CROSSOVER_GBPS {
            cleared += 1;
        }
        eprintln!(
            "w4a16 {tag:<8} BEST sg_x{} {:.1} GB/s  vs 279 crossover -> {}",
            best.2,
            best.1,
            if best.1 > CROSSOVER_GBPS {
                "CLEARS"
            } else {
                "below"
            }
        );
    }
    eprintln!("w4a16 cleared the {CROSSOVER_GBPS} GB/s crossover on {cleared}/4 gemma4-31B shapes");
}

const ROOFLINE_GBPS: f64 = 738.5;
const CROSSOVER_GBPS: f64 = 279.0;

fn bf16_encode(x: f32) -> u16 {
    let b = x.to_bits();
    let r = 0x7fff + ((b >> 16) & 1);
    if x.is_nan() {
        0x7fc0
    } else {
        ((b.wrapping_add(r)) >> 16) as u16
    }
}

fn butterfly(v: &mut [f32; 32]) {
    for d in [16usize, 8, 4, 2, 1] {
        let src = *v;
        for l in 0..32 {
            v[l] = src[l] + src[l ^ d];
        }
    }
}

fn oracle_v3(inputs: &Inputs, n: usize, k: usize) -> Vec<u16> {
    let k_blocks = k / 16;
    let stride = lin::ws_row_stride(k_blocks);
    let row_words = k / 8;
    let pairs = k_blocks / 2;
    let mut y = vec![0u16; n];
    for row in 0..n {
        let mut acc = [0.0f32; 32];
        for (lane, a) in acc.iter_mut().enumerate() {
            let mut v = lane;
            while v < pairs {
                let kb = 2 * v;
                for (j, kbj) in [kb, kb + 1].into_iter().enumerate() {
                    let si = row * stride + kbj;
                    let ws = (inputs.ws_lin_words[si / 4] >> (8 * (si % 4))) & 0xff;
                    let xs = (inputs.xs_words[kbj / 4] >> (8 * (kbj % 4))) & 0xff;
                    let s = lin::ue4m3_decode(ws) * lin::ue4m3_decode(xs);
                    let mut d = 0.0f32;
                    for h in 0..2 {
                        let ww = inputs.w_words[row * row_words + 4 * v + 2 * j + h];
                        let xlo = inputs.x_i8_words[4 * kbj + 2 * h];
                        let xhi = inputs.x_i8_words[4 * kbj + 2 * h + 1];
                        let p = dot4i8(lin::i8map(ww), xlo) + dot4i8(lin::i8map(ww >> 4), xhi);
                        d += p as f32 * 0.25;
                    }
                    *a = s.mul_add(d, *a);
                }
                v += 32;
            }
        }
        butterfly(&mut acc);
        y[row] = bf16_encode(acc[0]);
    }
    y
}

fn dot4i8(a: u32, b: u32) -> i32 {
    (0..4)
        .map(|i| {
            let x = ((a >> (8 * i)) & 0xff) as u8 as i8 as i32;
            let y = ((b >> (8 * i)) & 0xff) as u8 as i8 as i32;
            x * y
        })
        .sum()
}

fn oracle_tree(inputs: &Inputs, n: usize, k: usize) -> Vec<u16> {
    let k_blocks = k / 16;
    let k_tiles = g::k_tiles(k_blocks);
    let row_words = k / 8;
    let mut y = vec![0u16; n];
    for row in 0..n {
        let mut part = [0.0f32; 256];
        for (tid, p) in part.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            let mut kb = tid;
            while kb < k_blocks {
                let m_tile = row / 128;
                let d2 = (row / 32) % 4;
                let d3 = row % 32;
                let si = ((m_tile * k_tiles + kb / 4) * 32 + d3) * 16 + d2 * 4 + kb % 4;
                let ws = inputs.ws_bytes[si] as u32;
                let xs = (inputs.xs_words[kb / 4] >> (8 * (kb % 4))) & 0xff;
                let s = lin::ue4m3_decode(ws) * lin::ue4m3_decode(xs);
                let mut d = 0.0f32;
                for h in 0..2 {
                    let ww = inputs.w_words[row * row_words + 2 * kb + h];
                    let xw = inputs.x_words[2 * kb + h];
                    let p = dot4i8(lin::i8map(ww), lin::i8map(xw))
                        + dot4i8(lin::i8map(ww >> 4), lin::i8map(xw >> 4));
                    d += p as f32 * 0.25;
                }
                acc = s.mul_add(d, acc);
                kb += 256;
            }
            *p = acc;
        }
        for step in 0..8usize {
            let stride = [16usize, 8, 4, 2, 1, 128, 64, 32][step];
            let src = part;
            for tid in 0..256 {
                let taking = step < 5 || (tid & 31) == 0;
                if taking && (tid & stride) == 0 {
                    part[tid] = src[tid] + src[tid + stride];
                }
            }
        }
        y[row] = bf16_encode(part[0]);
    }
    y
}

#[test]
fn the_nvfp4_gemv_structure_ladder_on_the_gemma4_31b_shapes() {
    let ctx = ctx("nvfp4_structure_ladder");
    eprintln!(
        "NOTE: run this test alone. Four heavy suites in one process share ~1.5 GB of replica \
         buffers and every arm reads materially low when combined (current numbers: \
         perf/runs.jsonl)."
    );
    assert!(
        g::sg32_ok(ctx),
        "adapter is not 32-wide by probe; this suite measures the subgroup family"
    );
    let shapes = [
        ("gate_up", 43008usize, 5376usize, 100usize),
        ("down", 5376, 21504, 100),
        ("qkv", 16384, 5376, 100),
        ("o", 5376, 8192, 200),
    ];
    let cands = [
        Cand::Tree,
        Cand::Sg,
        Cand::Swz,
        Cand::Lin,
        Cand::NoScale,
        Cand::NoDec,
        Cand::XPre,
        Cand::V3,
        Cand::V3NoDec,
        Cand::V3Stream,
    ];
    let mut v3_over_crossover = 0usize;
    for (tag, n, k, iters) in shapes {
        assert!(lin::v3_shape_ok(k), "{tag}: k={k} not v3-shaped");
        let inputs = make_inputs(n, k, 0x5eed_0001 ^ ((n as u64) << 24) ^ k as u64);
        let mut baseline: Option<Vec<u16>> = None;
        let mut tree_gbps = 0.0f64;
        let mut v3_y: Vec<u16> = Vec::new();
        for &variant in &cands {
            let mut best = f64::MAX;
            let mut y = Vec::new();
            for _ in 0..3 {
                let (out, secs) = probe(ctx, &inputs, variant, n, k, 10, iters);
                best = best.min(secs);
                y = out;
            }
            let ms = best * 1e3 / iters as f64;
            let gbps = inputs.weight_bytes * iters as f64 / best / 1.0e9;
            let note = if !variant.valid_numerics() {
                "probe-only".to_string()
            } else if variant.bit_exact_with_tree() {
                match &baseline {
                    None => {
                        baseline = Some(y.clone());
                        tree_gbps = gbps;
                        "ref".to_string()
                    }
                    Some(b) => {
                        let diff = b.iter().zip(y.iter()).filter(|(a, c)| a != c).count();
                        assert_eq!(
                            diff,
                            0,
                            "{tag} n={n} k={k}: {} differs from tree in {diff}/{n} rows",
                            variant.label()
                        );
                        "bit-exact".to_string()
                    }
                }
            } else {
                let b = baseline.as_ref().expect("tree first");
                let diff = b.iter().zip(y.iter()).filter(|(a, c)| a != c).count();
                let maxulp = b
                    .iter()
                    .zip(y.iter())
                    .map(|(a, c)| (*a as i32 - *c as i32).unsigned_abs())
                    .max()
                    .unwrap_or(0);
                v3_y = y.clone();
                format!("reorder: {diff}/{n} rows differ, max {maxulp} bf16-ulp")
            };
            if variant == Cand::V3 && gbps > CROSSOVER_GBPS {
                v3_over_crossover += 1;
            }
            eprintln!(
                "nvfp4 {tag:<8} n={n:<6} k={k:<6} | {:<44} {ms:>8.4} ms {gbps:>7.1} GB/s {:>5.1}% roofline | {note}",
                variant.label(),
                100.0 * gbps / ROOFLINE_GBPS,
            );
        }
        let (i8ms, i8gbps) = probe_int8(ctx, n, k, iters, 0x1234 ^ (n as u64));
        eprintln!(
            "nvfp4 {tag:<8} n={n:<6} k={k:<6} | {:<44} {i8ms:>8.4} ms {i8gbps:>7.1} GB/s {:>5.1}% roofline | REFERENCE int8",
            "int8  wg128 stride32  vec4 rowscale",
            100.0 * i8gbps / ROOFLINE_GBPS,
        );
        let _ = (tree_gbps, &v3_y);
    }
    eprintln!(
        "v3 cleared the {CROSSOVER_GBPS} GB/s 4-bit crossover on {v3_over_crossover}/4 shapes"
    );
}

#[test]
fn linear_scales_are_bit_exact_with_the_swizzled_tree_on_awkward_shapes() {
    let ctx = ctx("nvfp4_lin_awkward");
    assert!(g::sg32_ok(ctx), "adapter is not 32-wide by probe");
    let mut cells = 0usize;
    for (n, k) in [
        (1usize, 16usize),
        (3, 32),
        (7, 4112),
        (129, 64),
        (37, 16 * 257),
        (255, 4096),
        (256, 5376),
        (64, 21504),
        (300, 336),
    ] {
        let inputs = make_inputs(n, k, 0xabcd ^ ((n as u64) << 20) ^ k as u64);
        let (yt, _) = probe(ctx, &inputs, Cand::Tree, n, k, 1, 1);
        let (yl, _) = probe(ctx, &inputs, Cand::Lin, n, k, 1, 1);
        let diff = yt.iter().zip(yl.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "n={n} k={k}: lin differs from tree in {diff}/{n}");
        eprintln!("awkward n={n} k={k}: lin bit-exact vs tree");
        cells += 1;
    }
    assert_eq!(cells, 9);
}

#[test]
fn v3_is_bit_exact_against_a_cpu_oracle_that_replicates_its_order() {
    let ctx = ctx("nvfp4_v3_oracle");
    assert!(g::sg32_ok(ctx), "adapter is not 32-wide by probe");
    let mut cells = 0usize;
    for (n, k) in [
        (1usize, 32usize),
        (5, 64),
        (129, 512),
        (255, 4096),
        (256, 5376),
        (64, 21504),
        (37, 2048),
        (300, 8192),
    ] {
        assert!(lin::v3_shape_ok(k));
        let inputs = make_inputs(n, k, 0x7777 ^ ((n as u64) << 18) ^ k as u64);
        let (y_tree_gpu, _) = probe(ctx, &inputs, Cand::Tree, n, k, 1, 1);
        let y_tree_cpu = oracle_tree(&inputs, n, k);
        let td = y_tree_gpu
            .iter()
            .zip(y_tree_cpu.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(td, 0, "n={n} k={k}: tree oracle disagrees with tree GPU");

        let (y_v3_gpu, _) = probe(ctx, &inputs, Cand::V3, n, k, 1, 1);
        let y_v3_cpu = oracle_v3(&inputs, n, k);
        let vd = y_v3_gpu
            .iter()
            .zip(y_v3_cpu.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(vd, 0, "n={n} k={k}: v3 oracle disagrees with v3 GPU");

        let rows_diff = y_tree_gpu
            .iter()
            .zip(y_v3_gpu.iter())
            .filter(|(a, b)| a != b)
            .count();
        let maxulp = y_tree_gpu
            .iter()
            .zip(y_v3_gpu.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
            .max()
            .unwrap_or(0);
        eprintln!(
            "oracle n={n:<5} k={k:<6}: tree==oracle_tree, v3==oracle_v3 | v3 vs tree: {rows_diff}/{n} rows, max {maxulp} bf16-ulp"
        );
        cells += 1;
    }
    assert_eq!(cells, 8);
}
