#![cfg(feature = "wgpu")]

mod common;
use common::wgpu_allow_skip;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemv_nvfp4 as g4;
use nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2 as v2;
use common::bf16_enc;

const ROOFLINE_GBPS: f64 = 800.0;
const REPLICA_BUDGET: f64 = 768.0 * 1024.0 * 1024.0;
const WORK_BUDGET: f64 = 1.5 * 1024.0 * 1024.0 * 1024.0;

fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(c) if c.qualify().qualified => {
            eprintln!("{test}: {}", c.summary());
            Some(c)
        }
        Ok(c) => {
            if !wgpu_allow_skip() {
                panic!(
                    "{test}: wgpu adapter not qualified: {:?}. This gate refuses to report \
                     success without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on \
                     purpose.",
                    c.qualify().reason
                );
            }
            eprintln!(
                "{test}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) adapter not qualified: {:?}",
                c.qualify().reason
            );
            None
        }
        Err(e) => {
            if !wgpu_allow_skip() {
                panic!(
                    "{test}: no wgpu adapter: {e}. This gate refuses to report success \
                     without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
                );
            }
            eprintln!("{test}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) no wgpu adapter: {e}");
            None
        }
    }
}

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn scale_word(&mut self) -> u32 {
        let mut w = 0u32;
        for b in 0..4 {
            w |= (0x30u32 | (self.next_u32() & 0x0f)) << (8 * b);
        }
        w
    }
}

#[inline]
fn byte_at(word: u32, idx: usize) -> u32 {
    (word >> (8 * (idx & 3))) & 0xff
}

#[inline]
fn ue4m3(bits: u32) -> f32 {
    f32::from_bits(((bits & 127) << 20).wrapping_add(0x3c00_0000))
}

#[inline]
fn i8map(s: u32) -> u32 {
    let k = s & 0x0707_0707;
    let hm = ((k >> 2) & 0x0101_0101).wrapping_mul(255);
    let e7 = (k & (k >> 1) & (k >> 2)) & 0x0101_0101;
    let m = k.wrapping_add((k & 0x0303_0303) & hm).wrapping_add(e7 << 1);
    let sb = (s & (k.wrapping_add(0x0707_0707) & 0x0808_0808)) >> 3;
    (m ^ sb.wrapping_mul(255)).wrapping_add(sb)
}

#[inline]
fn dot4i8(a: u32, b: u32) -> i32 {
    let mut s = 0i32;
    for i in 0..4 {
        let x = ((a >> (8 * i)) & 0xff) as u8 as i8 as i32;
        let y = ((b >> (8 * i)) & 0xff) as u8 as i8 as i32;
        s += x * y;
    }
    s
}

#[inline]
fn idot(w: u32, x: u32) -> i32 {
    dot4i8(i8map(w), i8map(x)) + dot4i8(i8map(w >> 4), i8map(x >> 4))
}

#[inline]
fn iblock(w0: u32, w1: u32, x0: u32, x1: u32) -> f32 {
    (idot(w0, x0) + idot(w1, x1)) as f32 * 0.25
}

#[inline]
fn e2m1(n: u32) -> f32 {
    let k = n & 7;
    let sgn = (n & 8) << 28;
    let mag = if k < 2 {
        (k & 1) * 0x3f00_0000
    } else {
        (k + 252) << 22
    };
    f32::from_bits(sgn | mag)
}

#[inline]
fn even4(w: u32) -> [f32; 4] {
    [
        e2m1(w & 0xf),
        e2m1((w >> 8) & 0xf),
        e2m1((w >> 16) & 0xf),
        e2m1((w >> 24) & 0xf),
    ]
}

#[inline]
fn odd4(w: u32) -> [f32; 4] {
    [
        e2m1((w >> 4) & 0xf),
        e2m1((w >> 12) & 0xf),
        e2m1((w >> 20) & 0xf),
        e2m1((w >> 28) & 0xf),
    ]
}

#[inline]
fn scale4(v: [f32; 4], s: f32) -> [f32; 4] {
    [v[0] * s, v[1] * s, v[2] * s, v[3] * s]
}

#[inline]
fn fdot8(w: u32, x: u32, acc_in: f32) -> f32 {
    let we = even4(w);
    let wo = odd4(w);
    let xe = even4(x);
    let xo = odd4(x);
    let mut s = acc_in;
    for i in 0..4 {
        s = we[i].mul_add(xe[i], s);
        s = wo[i].mul_add(xo[i], s);
    }
    s
}

#[inline]
fn fblock(w0: u32, w1: u32, x0: u32, x1: u32) -> f32 {
    fdot8(w1, x1, fdot8(w0, x0, 0.0))
}

#[inline]
fn pdot8(w: u32, xe: [f32; 4], xo: [f32; 4], acc_in: f32) -> f32 {
    let we = even4(w);
    let wo = odd4(w);
    let mut s = acc_in;
    for i in 0..4 {
        s = we[i].mul_add(xe[i], s);
        s = wo[i].mul_add(xo[i], s);
    }
    s
}

fn bfly(lanes: [f32; 32]) -> f32 {
    let mut a = lanes;
    for stride in [16usize, 8, 4, 2, 1] {
        let src = a;
        for (i, slot) in a.iter_mut().enumerate() {
            *slot = src[i] + src[i ^ stride];
        }
    }
    a[0]
}

fn ws_idx(row: usize, block: usize, k_tiles: usize) -> usize {
    let m_tile = row / 128;
    let d2 = (row / 32) % 4;
    let d3 = row % 32;
    let k_tile = block / 4;
    let d5 = block % 4;
    ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5
}

struct Data {
    n: usize,
    k_blocks: usize,
    k_tiles: usize,
    w: Vec<u32>,
    ws: Vec<u32>,
    x: Vec<u32>,
    xs: Vec<u32>,
}

impl Data {
    fn new(n: usize, k: usize) -> Self {
        let k_blocks = k / 16;
        let k_tiles = g4::k_tiles(k_blocks);
        let scale_len = g4::swizzled_scale_len(n, k_blocks);
        let mut rng = Lcg(0x0806_2026 ^ ((n as u64) << 24) ^ k as u64);
        Self {
            n,
            k_blocks,
            k_tiles,
            w: (0..n * k / 8).map(|_| rng.next_u32()).collect(),
            ws: (0..scale_len.div_ceil(4))
                .map(|_| rng.scale_word())
                .collect(),
            x: (0..k / 8).map(|_| rng.next_u32()).collect(),
            xs: (0..k_blocks.div_ceil(4))
                .map(|_| rng.scale_word())
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Oracle {
    Tree,
    Warp,
    Quad,
    QuadFloat,
    Pair,
    PairPrescale,
}

fn oracle_tree(d: &Data, row: usize) -> u16 {
    let wbase = row * d.k_blocks;
    let mut part = [0f32; 256];
    for (tid, slot) in part.iter_mut().enumerate() {
        let mut acc = 0f32;
        let mut kb = tid;
        while kb < d.k_blocks {
            let i = ws_idx(row, kb, d.k_tiles);
            let bs = ue4m3(byte_at(d.ws[i >> 2], i)) * ue4m3(byte_at(d.xs[kb >> 2], kb));
            let o = 2 * (wbase + kb);
            let mut dot = idot(d.w[o], d.x[2 * kb]) as f32 * 0.25;
            dot += idot(d.w[o + 1], d.x[2 * kb + 1]) as f32 * 0.25;
            acc = bs.mul_add(dot, acc);
            kb += 256;
        }
        *slot = acc;
    }
    for (step, stride) in [16usize, 8, 4, 2, 1, 128, 64, 32].into_iter().enumerate() {
        let src = part;
        for tid in 0..256 {
            let taking = step < 5 || (tid & 31) == 0;
            if taking && (tid & stride) == 0 {
                part[tid] = src[tid] + src[tid + stride];
            }
        }
    }
    bf16_enc(part[0])
}

fn oracle_warp(d: &Data, row: usize) -> u16 {
    let wbase = row * d.k_blocks;
    let mut lanes = [0f32; 32];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut acc = 0f32;
        let mut kb = lane;
        while kb < d.k_blocks {
            let i = ws_idx(row, kb, d.k_tiles);
            let bs = ue4m3(byte_at(d.ws[i >> 2], i)) * ue4m3(byte_at(d.xs[kb >> 2], kb));
            let o = 2 * (wbase + kb);
            acc = bs.mul_add(
                iblock(d.w[o], d.w[o + 1], d.x[2 * kb], d.x[2 * kb + 1]),
                acc,
            );
            kb += 32;
        }
        *slot = acc;
    }
    bf16_enc(bfly(lanes))
}

fn oracle_quad(d: &Data, row: usize) -> u16 {
    let quads = d.k_blocks / 4;
    let w4b = row * (d.k_blocks / 2);
    let mut lanes = [0f32; 32];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut acc = 0f32;
        let mut q = lane;
        while q < quads {
            let i = ws_idx(row, q * 4, d.k_tiles);
            let wsw = d.ws[i >> 2];
            let xsw = d.xs[q];
            let wo = 4 * (w4b + 2 * q);
            let xo = 8 * q;
            for j in 0..4 {
                let bs = ue4m3(byte_at(wsw, j)) * ue4m3(byte_at(xsw, j));
                let dot = iblock(
                    d.w[wo + 2 * j],
                    d.w[wo + 2 * j + 1],
                    d.x[xo + 2 * j],
                    d.x[xo + 2 * j + 1],
                );
                acc = bs.mul_add(dot, acc);
            }
            q += 32;
        }
        *slot = acc;
    }
    bf16_enc(bfly(lanes))
}

fn oracle_quad_float(d: &Data, row: usize) -> u16 {
    let quads = d.k_blocks / 4;
    let w4b = row * (d.k_blocks / 2);
    let mut lanes = [0f32; 32];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut acc = 0f32;
        let mut q = lane;
        while q < quads {
            let i = ws_idx(row, q * 4, d.k_tiles);
            let wsw = d.ws[i >> 2];
            let xsw = d.xs[q];
            let wo = 4 * (w4b + 2 * q);
            let xo = 8 * q;
            for j in 0..4 {
                let bs = ue4m3(byte_at(wsw, j)) * ue4m3(byte_at(xsw, j));
                let dot = fblock(
                    d.w[wo + 2 * j],
                    d.w[wo + 2 * j + 1],
                    d.x[xo + 2 * j],
                    d.x[xo + 2 * j + 1],
                );
                acc = bs.mul_add(dot, acc);
            }
            q += 32;
        }
        *slot = acc;
    }
    bf16_enc(bfly(lanes))
}

fn oracle_pair(d: &Data, row: usize) -> u16 {
    let pairs = d.k_blocks / 2;
    let base = row * (d.k_blocks / 2);
    let mut lanes = [0f32; 32];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut acc = 0f32;
        let mut p = lane;
        while p < pairs {
            let b0 = 2 * p;
            let xsw = d.xs[b0 >> 2];
            let xs0 = ue4m3(byte_at(xsw, b0));
            let xs1 = ue4m3(byte_at(xsw, b0 + 1));
            let wo = 4 * (base + p);
            let xo = 4 * p;
            let i = ws_idx(row, b0, d.k_tiles);
            let wsw = d.ws[i >> 2];
            let d0 = idot(d.w[wo], d.x[xo]) + idot(d.w[wo + 1], d.x[xo + 1]);
            let d1 = idot(d.w[wo + 2], d.x[xo + 2]) + idot(d.w[wo + 3], d.x[xo + 3]);
            acc = (ue4m3(byte_at(wsw, i)) * xs0).mul_add(d0 as f32 * 0.25, acc);
            acc = (ue4m3(byte_at(wsw, i + 1)) * xs1).mul_add(d1 as f32 * 0.25, acc);
            p += 32;
        }
        *slot = acc;
    }
    bf16_enc(bfly(lanes))
}

fn oracle_pair_prescale(d: &Data, row: usize) -> u16 {
    let pairs = d.k_blocks / 2;
    let base = row * (d.k_blocks / 2);
    let mut lanes = [0f32; 32];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut acc = 0f32;
        let mut p = lane;
        while p < pairs {
            let b0 = 2 * p;
            let xsw = d.xs[b0 >> 2];
            let xs0 = ue4m3(byte_at(xsw, b0));
            let xs1 = ue4m3(byte_at(xsw, b0 + 1));
            let wo = 4 * (base + p);
            let xo = 4 * p;
            let xe0 = scale4(even4(d.x[xo]), xs0);
            let xq0 = scale4(odd4(d.x[xo]), xs0);
            let xe1 = scale4(even4(d.x[xo + 1]), xs0);
            let xq1 = scale4(odd4(d.x[xo + 1]), xs0);
            let xe2 = scale4(even4(d.x[xo + 2]), xs1);
            let xq2 = scale4(odd4(d.x[xo + 2]), xs1);
            let xe3 = scale4(even4(d.x[xo + 3]), xs1);
            let xq3 = scale4(odd4(d.x[xo + 3]), xs1);
            let i = ws_idx(row, b0, d.k_tiles);
            let wsw = d.ws[i >> 2];
            let d0 = pdot8(d.w[wo + 1], xe1, xq1, pdot8(d.w[wo], xe0, xq0, 0.0));
            let d1 = pdot8(d.w[wo + 3], xe3, xq3, pdot8(d.w[wo + 2], xe2, xq2, 0.0));
            acc = ue4m3(byte_at(wsw, i)).mul_add(d0, acc);
            acc = ue4m3(byte_at(wsw, i + 1)).mul_add(d1, acc);
            p += 32;
        }
        *slot = acc;
    }
    bf16_enc(bfly(lanes))
}

fn oracle_rows(d: &Data, kind: Oracle) -> Vec<u16> {
    let f: fn(&Data, usize) -> u16 = match kind {
        Oracle::Tree => oracle_tree,
        Oracle::Warp => oracle_warp,
        Oracle::Quad => oracle_quad,
        Oracle::QuadFloat => oracle_quad_float,
        Oracle::Pair => oracle_pair,
        Oracle::PairPrescale => oracle_pair_prescale,
    };
    let threads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(8)
        .min(24);
    let chunk = d.n.div_ceil(threads).max(1);
    let mut out = vec![0u16; d.n];
    std::thread::scope(|s| {
        for (ci, part) in out.chunks_mut(chunk).enumerate() {
            let d = &*d;
            s.spawn(move || {
                let start = ci * chunk;
                for (j, o) in part.iter_mut().enumerate() {
                    *o = f(d, start + j);
                }
            });
        }
    });
    out
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Tree,
    V2(v2::V2Kernel),
}

#[derive(Clone, Copy, Debug)]
struct Variant {
    name: &'static str,
    kind: Kind,
    cfg: v2::V2Config,
    oracle: Oracle,
}

impl Variant {
    fn label(&self) -> String {
        match self.kind {
            Kind::Tree => "tree wg256 r1 (SHIPPING)".to_string(),
            Kind::V2(k) if k.multi_row() => {
                format!("{} wg{} mr{}", self.name, self.cfg.wg, self.cfg.mr)
            }
            Kind::V2(_) => format!("{} wg{}", self.name, self.cfg.wg),
        }
    }
    fn entry(&self) -> &'static str {
        match self.kind {
            Kind::Tree => g4::GEMV_ENTRY,
            Kind::V2(k) => k.entry(),
        }
    }
    fn source(&self) -> String {
        match self.kind {
            Kind::Tree => g4::gemv_source(),
            Kind::V2(_) => v2::source(self.cfg),
        }
    }
    fn rows_per_group(&self) -> u32 {
        match self.kind {
            Kind::Tree => 1,
            Kind::V2(k) => self.cfg.rows_per_group(k),
        }
    }
    fn vec4_slots(&self) -> bool {
        match self.kind {
            Kind::Tree => false,
            Kind::V2(k) => k.vec4_slots(),
        }
    }
    fn shape_ok(&self, k: usize) -> bool {
        match self.kind {
            Kind::Tree => true,
            Kind::V2(kern) => kern.shape_ok(k),
        }
    }
}

fn matrix() -> Vec<Variant> {
    use v2::V2Kernel::*;
    let mut v = vec![Variant {
        name: "tree",
        kind: Kind::Tree,
        cfg: v2::V2Config::new(256, 1),
        oracle: Oracle::Tree,
    }];
    for wg in [256u32, 128, 64] {
        v.push(Variant {
            name: "A warp",
            kind: Kind::V2(Warp),
            cfg: v2::V2Config::new(wg, 1),
            oracle: Oracle::Warp,
        });
    }
    for wg in [256u32, 128] {
        v.push(Variant {
            name: "B warpq",
            kind: Kind::V2(WarpQ),
            cfg: v2::V2Config::new(wg, 1),
            oracle: Oracle::Quad,
        });
    }
    for wg in [256u32, 128] {
        v.push(Variant {
            name: "C fdec",
            kind: Kind::V2(FDec),
            cfg: v2::V2Config::new(wg, 1),
            oracle: Oracle::QuadFloat,
        });
    }
    for (wg, mr) in [
        (256u32, 2u32),
        (256, 4),
        (256, 8),
        (128, 2),
        (128, 4),
        (64, 4),
    ] {
        v.push(Variant {
            name: "D mrow",
            kind: Kind::V2(MRow),
            cfg: v2::V2Config::new(wg, mr),
            oracle: Oracle::Pair,
        });
        v.push(Variant {
            name: "E fmrow",
            kind: Kind::V2(FMRow),
            cfg: v2::V2Config::new(wg, mr),
            oracle: Oracle::PairPrescale,
        });
        v.push(Variant {
            name: "F fmlut",
            kind: Kind::V2(FMLut),
            cfg: v2::V2Config::new(wg, mr),
            oracle: Oracle::PairPrescale,
        });
    }
    for (wg, mr) in [(128u32, 1u32), (64, 2)] {
        v.push(Variant {
            name: "D mrow",
            kind: Kind::V2(MRow),
            cfg: v2::V2Config::new(wg, mr),
            oracle: Oracle::Pair,
        });
    }
    for wg in [256u32, 128, 64] {
        v.push(Variant {
            name: "G mrow2",
            kind: Kind::V2(MRow2),
            cfg: v2::V2Config::new(wg, 2),
            oracle: Oracle::Pair,
        });
        v.push(Variant {
            name: "H mrowq",
            kind: Kind::V2(MRowQ),
            cfg: v2::V2Config::new(wg, 2),
            oracle: Oracle::Quad,
        });
    }
    v
}

struct Timing {
    ms: f64,
    gbps: f64,
    spread_pct: f64,
}

fn replicas(bytes: f64) -> usize {
    ((REPLICA_BUDGET / bytes).ceil() as usize).clamp(2, 512)
}

fn iters(bytes: f64) -> usize {
    ((WORK_BUDGET / bytes).round() as usize).clamp(20, 4000)
}

fn run_timed(
    ctx: &WgpuContext,
    pipeline: &wgpu::ComputePipeline,
    groups: (u32, u32, u32),
    binds: &[wgpu::BindGroup],
    n_iters: usize,
    bytes: f64,
) -> Timing {
    let submit = |count: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            for i in 0..count {
                pass.set_bind_group(0, &binds[i % binds.len()], &[]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
        }
        ctx.queue.submit([enc.finish()]);
    };
    submit(binds.len().clamp(4, 32));
    ctx.poll_blocking().expect("warmup poll");

    let mut runs = Vec::new();
    for _ in 0..3 {
        let start = std::time::Instant::now();
        submit(n_iters);
        ctx.poll_blocking().expect("timed poll");
        runs.push(start.elapsed().as_secs_f64());
    }
    let best = runs.iter().cloned().fold(f64::MAX, f64::min);
    let worst = runs.iter().cloned().fold(0.0, f64::max);
    Timing {
        ms: best * 1e3 / n_iters as f64,
        gbps: bytes * n_iters as f64 / best / 1.0e9,
        spread_pct: 100.0 * (worst - best) / best,
    }
}

fn diff_report(want: &[u16], got: &[u16]) -> (bool, String) {
    let mut off = 0usize;
    let mut max_ulp = 0i32;
    let mut first = usize::MAX;
    for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
        if a != b {
            if off == 0 {
                first = i;
            }
            off += 1;
            max_ulp = max_ulp.max((*a as i32 - *b as i32).abs());
        }
    }
    if off == 0 {
        (true, "BIT-EXACT".to_string())
    } else {
        (
            false,
            format!(
                "MISMATCH {off}/{} rows max_ulp={max_ulp} first_row={first}",
                want.len()
            ),
        )
    }
}

fn shape_matrix(ctx: &WgpuContext, tag: &str, n: usize, k: usize, failures: &mut Vec<String>) {
    let d = Data::new(n, k);
    let bytes = (n * k / 2 + n * d.k_blocks) as f64;
    let reps = replicas(bytes);
    let n_iters = iters(bytes);

    let wbufs: Vec<wgpu::Buffer> = (0..reps)
        .map(|_| dispatch::storage_from_slice(ctx, "v2-w", &d.w))
        .collect();
    let wsbufs: Vec<wgpu::Buffer> = (0..reps)
        .map(|_| dispatch::storage_from_slice(ctx, "v2-ws", &d.ws))
        .collect();
    let xbuf = dispatch::storage_from_slice(ctx, "v2-x", &d.x);
    let xsbuf = dispatch::storage_from_slice(ctx, "v2-xs", &d.xs);
    let ybuf = dispatch::storage_zeroed(ctx, "v2-y", (n * 4) as u64);

    let mut cache: Vec<(Oracle, Vec<u16>)> = Vec::new();
    let mut ran = 0usize;
    for v in matrix() {
        if !v.shape_ok(k) {
            eprintln!("{tag:<10} n={n:<6} k={k:<6} | {:<22} SKIP shape", v.label());
            continue;
        }
        ran += 1;
        let groups = dispatch::workgroup_count_1d(ctx, n as u64, v.rows_per_group());
        let params = g4::gemv_params(1.0, n, k, groups.0);
        let pbuf = dispatch::uniform_from(ctx, "v2-p", &params);
        let pipeline =
            match dispatch::cached_compute_pipeline(ctx, "nvfp4-v2", &v.source(), v.entry()) {
                Ok(p) => p,
                Err(e) => {
                    let msg = format!("{tag} {} PIPELINE FAIL {e}", v.label());
                    eprintln!("{msg}");
                    failures.push(msg);
                    continue;
                }
            };
        let binds: Vec<wgpu::BindGroup> = (0..reps)
            .map(|i| {
                let b: Vec<(u32, &wgpu::Buffer)> = if v.vec4_slots() {
                    vec![
                        (v2::WS_SLOT, &wsbufs[i]),
                        (v2::XS_SLOT, &xsbuf),
                        (v2::PARAMS_SLOT, &pbuf),
                        (v2::Y_SLOT, &ybuf),
                        (v2::W4_SLOT, &wbufs[i]),
                        (v2::X4_SLOT, &xbuf),
                    ]
                } else {
                    vec![
                        (v2::W2_SLOT, &wbufs[i]),
                        (v2::WS_SLOT, &wsbufs[i]),
                        (v2::X2_SLOT, &xbuf),
                        (v2::XS_SLOT, &xsbuf),
                        (v2::PARAMS_SLOT, &pbuf),
                        (v2::Y_SLOT, &ybuf),
                    ]
                };
                dispatch::bind_group(ctx, &pipeline, &b)
            })
            .collect();

        let t = run_timed(ctx, &pipeline, groups, &binds, n_iters, bytes);
        let words: Vec<u32> = dispatch::read_back(ctx, &ybuf, n).expect("read_back");
        let got: Vec<u16> = words.iter().map(|w| (*w & 0xffff) as u16).collect();

        if !cache.iter().any(|(o, _)| *o == v.oracle) {
            cache.push((v.oracle, oracle_rows(&d, v.oracle)));
        }
        let want = &cache.iter().find(|(o, _)| *o == v.oracle).unwrap().1;
        let (ok, verdict) = diff_report(want, &got);
        if !ok {
            failures.push(format!("{tag} n={n} k={k} {} {verdict}", v.label()));
        }
        eprintln!(
            "{tag:<10} n={n:<6} k={k:<6} | {:<22} {:>9.4} ms {:>7.1} GB/s {:>5.1}% roof (spread {:>4.1}%, reps {reps}, iters {n_iters}) | {verdict}",
            v.label(),
            t.ms,
            t.gbps,
            100.0 * t.gbps / ROOFLINE_GBPS,
            t.spread_pct,
        );
    }
    assert!(
        ran > 0,
        "{tag} n={n} k={k}: every variant in the matrix was shape-skipped, so this cell \
         compared nothing"
    );
}

const CHEAP_DECODE: &str = r#"
fn nv2_i8map(s: u32) -> u32 { return s & 0x0f0f0f0fu; }
fn nv2_dec4(n: vec4<u32>) -> vec4<f32> {
    return bitcast<vec4<f32>>((n & vec4<u32>(15u)) << vec4<u32>(23u));
}
fn nv2_mdec4(n: vec4<u32>) -> vec4<f32> {
    return bitcast<vec4<f32>>((n & vec4<u32>(15u)) << vec4<u32>(23u));
}
"#;

fn ablation_shape(ctx: &WgpuContext, tag: &str, n: usize, k: usize) {
    let d = Data::new(n, k);
    let bytes = (n * k / 2 + n * d.k_blocks) as f64;
    let reps = replicas(bytes);
    let n_iters = iters(bytes);
    let wbufs: Vec<wgpu::Buffer> = (0..reps)
        .map(|_| dispatch::storage_from_slice(ctx, "ab-w", &d.w))
        .collect();
    let wsbufs: Vec<wgpu::Buffer> = (0..reps)
        .map(|_| dispatch::storage_from_slice(ctx, "ab-ws", &d.ws))
        .collect();
    let xbuf = dispatch::storage_from_slice(ctx, "ab-x", &d.x);
    let xsbuf = dispatch::storage_from_slice(ctx, "ab-xs", &d.xs);
    let ybuf = dispatch::storage_zeroed(ctx, "ab-y", (n * 4) as u64);

    for v in matrix() {
        let Kind::V2(kern) = v.kind else { continue };
        if !v.shape_ok(k) {
            continue;
        }
        let groups = dispatch::workgroup_count_1d(ctx, n as u64, v.rows_per_group());
        let params = g4::gemv_params(1.0, n, k, groups.0);
        let pbuf = dispatch::uniform_from(ctx, "ab-p", &params);
        let real = v.source();
        let cheap = v2::with_decode(&real, CHEAP_DECODE).expect("decode markers");
        let mut out = Vec::new();
        for (label, src) in [("A0 real", &real), ("A1 cheap-decode", &cheap)] {
            let pipeline = dispatch::cached_compute_pipeline(ctx, "nvfp4-v2-ab", src, v.entry())
                .expect("pipeline");
            let binds: Vec<wgpu::BindGroup> = (0..reps)
                .map(|i| {
                    let b: Vec<(u32, &wgpu::Buffer)> = if kern.vec4_slots() {
                        vec![
                            (v2::WS_SLOT, &wsbufs[i]),
                            (v2::XS_SLOT, &xsbuf),
                            (v2::PARAMS_SLOT, &pbuf),
                            (v2::Y_SLOT, &ybuf),
                            (v2::W4_SLOT, &wbufs[i]),
                            (v2::X4_SLOT, &xbuf),
                        ]
                    } else {
                        vec![
                            (v2::W2_SLOT, &wbufs[i]),
                            (v2::WS_SLOT, &wsbufs[i]),
                            (v2::X2_SLOT, &xbuf),
                            (v2::XS_SLOT, &xsbuf),
                            (v2::PARAMS_SLOT, &pbuf),
                            (v2::Y_SLOT, &ybuf),
                        ]
                    };
                    dispatch::bind_group(ctx, &pipeline, &b)
                })
                .collect();
            out.push((
                label,
                run_timed(ctx, &pipeline, groups, &binds, n_iters, bytes),
            ));
        }
        eprintln!(
            "ablate {tag:<9} n={n:<6} k={k:<6} | {:<22} A0 {:>8.4} ms {:>6.1} GB/s -> A1 {:>8.4} ms {:>6.1} GB/s  ({:.2}x headroom)",
            v.label(),
            out[0].1.ms,
            out[0].1.gbps,
            out[1].1.ms,
            out[1].1.gbps,
            out[0].1.ms / out[1].1.ms,
        );
    }
}

#[test]
fn nvfp4_variant_decode_alu_ablation() {
    let Some(ctx) = ctx_or_skip("nvfp4_variant_decode_alu_ablation") else {
        return;
    };
    eprintln!("=== decode-ALU ablation (A1 is numerically WRONG, timing only) ===");
    for (tag, n, k) in [
        ("gate_up", 43008usize, 5376usize),
        ("down", 5376, 21504),
        ("attn_q", 8192, 2048),
        ("q38_gate_up", 17408, 5120),
        ("q38_down", 5120, 17408),
    ] {
        ablation_shape(ctx, tag, n, k);
    }
}

const MOE_SLOTS: usize = 9;

#[test]
fn nvfp4_variant_matrix_qwen3_moe_routed_dispatch() {
    let Some(ctx) = ctx_or_skip("nvfp4_variant_matrix_qwen3_moe_routed_dispatch") else {
        return;
    };
    eprintln!(
        "=== Qwen3.6-35B-A3B MoE expert GEMV at the routed dispatch size ({MOE_SLOTS} slots) ==="
    );
    eprintln!(
        "stream-read bound for these two dispatches, same addressing, zero decode: \
         perf/runs.jsonl carries the current gate_up/down bounds (down improves with 2 rows \
         per subgroup)"
    );
    let mut failures = Vec::new();
    for (tag, n, k) in [
        ("moe_gate_up", MOE_SLOTS * 1024, 2048usize),
        ("moe_down", MOE_SLOTS * 2048, 512),
    ] {
        shape_matrix(ctx, tag, n, k, &mut failures);
    }
    assert!(failures.is_empty(), "non-bit-exact variants: {failures:#?}");
}

struct RouteArm {
    label: &'static str,
    kind: v2::V2Kernel,
    cfg: v2::V2Config,
    oracle: Oracle,

    pk: bool,
}

impl RouteArm {
    fn entry(&self) -> &'static str {
        if self.pk {
            self.kind.pk_entry().expect("pk entry for arm")
        } else {
            self.kind.entry()
        }
    }
}

const ROUTE_ROUNDS: usize = 9;

fn med(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn lo(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::MAX, f64::min)
}

fn hi(v: &[f64]) -> f64 {
    v.iter().cloned().fold(0.0, f64::max)
}

fn spread_pct(v: &[f64]) -> f64 {
    100.0 * (hi(v) - lo(v)) / lo(v)
}

struct ArmState {
    pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    binds: Vec<wgpu::BindGroup>,
    groups: (u32, u32, u32),
}

impl ArmState {
    fn command_buffer(&self, ctx: &WgpuContext, n_iters: usize) -> wgpu::CommandBuffer {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            for i in 0..n_iters {
                pass.set_bind_group(0, &self.binds[i % self.binds.len()], &[]);
                pass.dispatch_workgroups(self.groups.0, self.groups.1, self.groups.2);
            }
        }
        enc.finish()
    }

    fn time(&self, ctx: &WgpuContext, n_iters: usize, bytes: f64) -> f64 {
        let cb = self.command_buffer(ctx, n_iters);
        let t0 = std::time::Instant::now();
        ctx.queue.submit([cb]);
        ctx.poll_blocking().expect("timed poll");
        bytes * n_iters as f64 / t0.elapsed().as_secs_f64() / 1.0e9
    }
}

#[allow(clippy::too_many_arguments)]
fn arm_state(
    ctx: &WgpuContext,
    d: &Data,
    a: &RouteArm,
    wbufs: &[wgpu::Buffer],
    wsbufs: &[wgpu::Buffer],
    xbuf: &wgpu::Buffer,
    xsbuf: &wgpu::Buffer,
    ybuf: &wgpu::Buffer,
) -> ArmState {
    let rpg = a.cfg.rows_per_group(a.kind);
    let groups = dispatch::workgroup_count_1d(ctx, d.n as u64, rpg);
    let params = g4::gemv_params(1.0, d.n, d.k_blocks * 16, groups.0);
    let pbuf = dispatch::uniform_from(ctx, "rt-p", &params);
    let src = v2::source(a.cfg);
    let pipeline = dispatch::cached_compute_pipeline(ctx, "nvfp4-route", &src, a.entry())
        .expect("route pipeline");
    let binds: Vec<wgpu::BindGroup> = (0..wbufs.len())
        .map(|i| {
            let b: Vec<(u32, &wgpu::Buffer)> = if a.kind.vec4_slots() {
                vec![
                    (v2::WS_SLOT, &wsbufs[i]),
                    (v2::XS_SLOT, xsbuf),
                    (v2::PARAMS_SLOT, &pbuf),
                    (v2::Y_SLOT, ybuf),
                    (v2::W4_SLOT, &wbufs[i]),
                    (v2::X4_SLOT, xbuf),
                ]
            } else {
                vec![
                    (v2::W2_SLOT, &wbufs[i]),
                    (v2::WS_SLOT, &wsbufs[i]),
                    (v2::X2_SLOT, xbuf),
                    (v2::XS_SLOT, xsbuf),
                    (v2::PARAMS_SLOT, &pbuf),
                    (v2::Y_SLOT, ybuf),
                ]
            };
            dispatch::bind_group(ctx, &pipeline, &b)
        })
        .collect();

    std::mem::forget(pbuf);
    ArmState {
        pipeline,
        binds,
        groups,
    }
}

fn ulp_report(a: &[u16], b: &[u16]) -> String {
    let mut off = 0usize;
    let mut max_ulp = 0i32;
    let mut sum_ulp = 0i64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as i32 - *y as i32).abs();
        if d != 0 {
            off += 1;
            sum_ulp += d as i64;
            max_ulp = max_ulp.max(d);
        }
    }
    if off == 0 {
        "bit-identical".to_string()
    } else {
        format!(
            "{off}/{} rows differ, max {max_ulp} ulp, mean {:.2} ulp over the differing rows",
            a.len(),
            sum_ulp as f64 / off as f64
        )
    }
}

fn widen_scales(d: &mut Data, seed: u64) {
    let mut rng = Lcg(seed);
    let word = |r: &mut Lcg| {
        let mut w = 0u32;
        for b in 0..4 {
            w |= ((r.next_u32() & 0x7f) | 0x08) << (8 * b);
        }
        w
    };
    for v in d.ws.iter_mut() {
        *v = word(&mut rng);
    }
    for v in d.xs.iter_mut() {
        *v = word(&mut rng);
    }
}

fn route_ab(ctx: &WgpuContext, tag: &str, n: usize, k: usize, arms: &[RouteArm]) {
    let d = Data::new(n, k);
    let bytes = (n * k / 2 + n * d.k_blocks) as f64;
    let reps = replicas(bytes);

    let n_iters = iters(bytes);
    let wbufs: Vec<wgpu::Buffer> = (0..reps)
        .map(|_| dispatch::storage_from_slice(ctx, "rt-w", &d.w))
        .collect();
    let wsbufs: Vec<wgpu::Buffer> = (0..reps)
        .map(|_| dispatch::storage_from_slice(ctx, "rt-ws", &d.ws))
        .collect();
    let xbuf = dispatch::storage_from_slice(ctx, "rt-x", &d.x);
    let xsbuf = dispatch::storage_from_slice(ctx, "rt-xs", &d.xs);
    let ybuf = dispatch::storage_zeroed(ctx, "rt-y", (n * 4) as u64);

    let mut order: Vec<&RouteArm> = arms.iter().collect();
    order.push(&arms[0]);
    let states: Vec<ArmState> = order
        .iter()
        .map(|a| arm_state(ctx, &d, a, &wbufs, &wsbufs, &xbuf, &xsbuf, &ybuf))
        .collect();

    eprintln!(
        "--- {tag} n={n} k={k}  {:.2} MiB/dispatch  x{n_iters} dispatches/rep, {reps} replicas, \
         {ROUTE_ROUNDS} interleaved rounds ---",
        bytes / (1024.0 * 1024.0)
    );

    let mut outs: Vec<Vec<u16>> = Vec::new();
    for (a, st) in order.iter().zip(&states) {
        ctx.queue.submit([st.command_buffer(ctx, 1)]);
        ctx.poll_blocking().expect("correctness poll");
        if a.pk {
            let words: Vec<u32> = dispatch::read_back(ctx, &ybuf, n / 2).expect("read_back");
            let mut y = Vec::with_capacity(n);
            for w in words {
                y.push((w & 0xffff) as u16);
                y.push((w >> 16) as u16);
            }
            outs.push(y);
        } else {
            let words: Vec<u32> = dispatch::read_back(ctx, &ybuf, n).expect("read_back");
            let y: Vec<u16> = words.iter().map(|w| (*w & 0xffff) as u16).collect();
            let (ok, verdict) = diff_report(&oracle_rows(&d, a.oracle), &y);
            assert!(ok, "{tag} {} is not bit-exact: {verdict}", a.label);
            outs.push(y);
        }
    }

    let mut rounds: Vec<Vec<f64>> = vec![Vec::new(); states.len()];
    for st in &states {
        for _ in 0..2 {
            ctx.queue.submit([st.command_buffer(ctx, n_iters)]);
            ctx.poll_blocking().expect("warmup");
        }
    }
    for _ in 0..ROUTE_ROUNDS {
        for (i, st) in states.iter().enumerate() {
            rounds[i].push(st.time(ctx, n_iters, bytes));
        }
    }

    let base = &rounds[0];
    for (i, a) in order.iter().enumerate() {
        let r = &rounds[i];
        let ratio: Vec<f64> = r.iter().zip(base).map(|(x, b)| x / b).collect();
        let tag_i = if i + 1 == order.len() {
            format!("{} [NULL]", a.label)
        } else {
            a.label.to_string()
        };
        eprintln!(
            "  {:<28} {:>7.1} GB/s med (min {:>6.1} max {:>6.1}, spread {:>5.1}%) | \
             per-round vs arm0: {:.3}x med ({:.3}-{:.3}) | {}",
            tag_i,
            med(r),
            lo(r),
            hi(r),
            spread_pct(r),
            med(&ratio),
            lo(&ratio),
            hi(&ratio),
            if i == 0 {
                "reference".to_string()
            } else {
                ulp_report(&outs[0], &outs[i])
            }
        );
    }
}

#[test]
#[ignore = "GPU rate measurement; run explicitly with --nocapture"]
fn moe_expert_route_ab() {
    let Some(ctx) = ctx_or_skip("moe_expert_route_ab") else {
        return;
    };
    eprintln!("=== Qwen3.6 MoE expert GEMV: which v2 kernel the route should pick ===");
    eprintln!(
        "stream-read bound at the same addressing with zero decode: perf/runs.jsonl carries \
         the current gate_up/down bounds. Anything below the bound is kernel, not memory."
    );
    route_ab(
        ctx,
        "moe gate/up (routed at n=1024)",
        MOE_SLOTS * 1024,
        2048,
        &[
            RouteArm {
                label: "fdec wg128  (TODAY)",
                kind: v2::V2Kernel::FDec,
                cfg: v2::V2Config::new(128, 1),
                oracle: Oracle::QuadFloat,
                pk: false,
            },
            RouteArm {
                label: "fmlut wg128 mr4",
                kind: v2::V2Kernel::FMLut,
                cfg: v2::V2Config::new(128, 4),
                oracle: Oracle::PairPrescale,
                pk: false,
            },
            RouteArm {
                label: "fmlut wg256 mr8",
                kind: v2::V2Kernel::FMLut,
                cfg: v2::V2Config::new(256, 8),
                oracle: Oracle::PairPrescale,
                pk: false,
            },
            RouteArm {
                label: "warp wg64",
                kind: v2::V2Kernel::Warp,
                cfg: v2::V2Config::new(64, 1),
                oracle: Oracle::Warp,
                pk: false,
            },
        ],
    );

    route_ab(
        ctx,
        "moe gate/up, PACKED entries",
        MOE_SLOTS * 1024,
        2048,
        &[
            RouteArm {
                label: "fdec_pk wg128  (TODAY)",
                kind: v2::V2Kernel::FDec,
                cfg: v2::V2Config::new(128, 1),
                oracle: Oracle::QuadFloat,
                pk: true,
            },
            RouteArm {
                label: "fmlut_pk wg128 mr4",
                kind: v2::V2Kernel::FMLut,
                cfg: v2::V2Config::new(128, 4),
                oracle: Oracle::PairPrescale,
                pk: true,
            },
        ],
    );

    route_ab(
        ctx,
        "moe gate or up unfused (n=512)",
        MOE_SLOTS * 512,
        2048,
        &[
            RouteArm {
                label: "fdec wg128  (TODAY)",
                kind: v2::V2Kernel::FDec,
                cfg: v2::V2Config::new(128, 1),
                oracle: Oracle::QuadFloat,
                pk: false,
            },
            RouteArm {
                label: "fmlut wg128 mr4",
                kind: v2::V2Kernel::FMLut,
                cfg: v2::V2Config::new(128, 4),
                oracle: Oracle::PairPrescale,
                pk: false,
            },
        ],
    );
    route_ab(
        ctx,
        "moe down (routed at n=2048)",
        MOE_SLOTS * 2048,
        512,
        &[
            RouteArm {
                label: "warp wg64   (TODAY)",
                kind: v2::V2Kernel::Warp,
                cfg: v2::V2Config::new(64, 1),
                oracle: Oracle::Warp,
                pk: false,
            },
            RouteArm {
                label: "fmlut wg128 mr4",
                kind: v2::V2Kernel::FMLut,
                cfg: v2::V2Config::new(128, 4),
                oracle: Oracle::PairPrescale,
                pk: false,
            },
            RouteArm {
                label: "fmlut wg256 mr4",
                kind: v2::V2Kernel::FMLut,
                cfg: v2::V2Config::new(256, 4),
                oracle: Oracle::PairPrescale,
                pk: false,
            },
            RouteArm {
                label: "fdec wg128",
                kind: v2::V2Kernel::FDec,
                cfg: v2::V2Config::new(128, 1),
                oracle: Oracle::QuadFloat,
                pk: false,
            },
        ],
    );
}

#[test]
fn moe_expert_shapes_are_route_invariant_bit_for_bit() {
    let ctx = ctx_or_skip("moe_expert_shapes_are_route_invariant_bit_for_bit")
        .expect("no qualified wgpu adapter; the route-invariance gate cannot be waived");
    let arms = [
        Variant {
            name: "C fdec",
            kind: Kind::V2(v2::V2Kernel::FDec),
            cfg: v2::V2Config::new(128, 1),
            oracle: Oracle::QuadFloat,
        },
        Variant {
            name: "F fmlut",
            kind: Kind::V2(v2::V2Kernel::FMLut),
            cfg: v2::V2Config::new(128, 4),
            oracle: Oracle::PairPrescale,
        },
        Variant {
            name: "F fmlut",
            kind: Kind::V2(v2::V2Kernel::FMLut),
            cfg: v2::V2Config::new(256, 8),
            oracle: Oracle::PairPrescale,
        },
        Variant {
            name: "A warp",
            kind: Kind::V2(v2::V2Kernel::Warp),
            cfg: v2::V2Config::new(64, 1),
            oracle: Oracle::Warp,
        },
    ];

    for (tag, n, k, gate) in [
        ("moe gate/up", MOE_SLOTS * 1024, 2048usize, true),
        ("moe down", MOE_SLOTS * 2048, 512, false),
    ] {
        for wide in [false, true] {
            let mut d = Data::new(n, k);
            if wide {
                widen_scales(&mut d, 0xd1ce ^ (k as u64) ^ (n as u64) << 20);
            }
            let d = d;
            let base = run_once(ctx, &arms[0], &d, n, k);
            for v in &arms[1..] {
                if !v.shape_ok(k) {
                    continue;
                }
                let got = run_once(ctx, v, &d, n, k);
                let verdict = ulp_report(&base, &got);
                eprintln!(
                    "{tag:<12} n={n:<6} k={k:<5} scales={:<6} {:<22} vs {:<22} {verdict}",
                    if wide { "wide" } else { "narrow" },
                    v.label(),
                    arms[0].label()
                );
                assert!(
                    !gate || base == got,
                    "{tag} n={n} k={k} wide={wide}: {} moved the output away from {}; \
                     rerouting this shape is not free",
                    v.label(),
                    arms[0].label()
                );
            }
        }
    }
}

#[test]
#[ignore = "GPU rate measurement; run explicitly with --nocapture"]
fn single_slot_shapes_the_same_threshold_decides() {
    let Some(ctx) = ctx_or_skip("single_slot_shapes_the_same_threshold_decides") else {
        return;
    };
    eprintln!("=== shapes a lowered n-threshold would also re-route (slots = 1) ===");
    for (tag, n, k) in [
        ("attn_kv", 512usize, 2048usize),
        ("moe gate/up unfused", 1024, 2048),
    ] {
        route_ab(
            ctx,
            tag,
            n,
            k,
            &[
                RouteArm {
                    label: "fdec wg128  (TODAY)",
                    kind: v2::V2Kernel::FDec,
                    cfg: v2::V2Config::new(128, 1),
                    oracle: Oracle::QuadFloat,
                    pk: false,
                },
                RouteArm {
                    label: "fmlut wg128 mr4",
                    kind: v2::V2Kernel::FMLut,
                    cfg: v2::V2Config::new(128, 4),
                    oracle: Oracle::PairPrescale,
                    pk: false,
                },
            ],
        );
    }
}

#[test]
#[ignore = "GPU rate measurement; run explicitly with --nocapture"]
fn dense_single_slot_route_ab() {
    let Some(ctx) = ctx_or_skip("dense_single_slot_route_ab") else {
        return;
    };
    eprintln!("=== dense single-slot GEMV: shipping mrow vs scalar/wide rungs ===");
    for (tag, n, k) in [
        ("q38 gate/up", 17408usize, 5120usize),
        ("q38 down", 5120, 17408),
        ("31b gate_up", 43008, 5376),
        ("31b down", 5376, 21504),
    ] {
        route_ab(
            ctx,
            tag,
            n,
            k,
            &[
                RouteArm {
                    label: "mrow wg128 mr2 (TODAY)",
                    kind: v2::V2Kernel::MRow,
                    cfg: v2::V2Config::new(128, 2),
                    oracle: Oracle::Pair,
                    pk: false,
                },
                RouteArm {
                    label: "mrow2 wg128",
                    kind: v2::V2Kernel::MRow2,
                    cfg: v2::V2Config::new(128, 2),
                    oracle: Oracle::Pair,
                    pk: false,
                },
                RouteArm {
                    label: "mrow2 wg256",
                    kind: v2::V2Kernel::MRow2,
                    cfg: v2::V2Config::new(256, 2),
                    oracle: Oracle::Pair,
                    pk: false,
                },
                RouteArm {
                    label: "mrowq wg128",
                    kind: v2::V2Kernel::MRowQ,
                    cfg: v2::V2Config::new(128, 2),
                    oracle: Oracle::Quad,
                    pk: false,
                },
                RouteArm {
                    label: "mrowq wg256",
                    kind: v2::V2Kernel::MRowQ,
                    cfg: v2::V2Config::new(256, 2),
                    oracle: Oracle::Quad,
                    pk: false,
                },
                RouteArm {
                    label: "warpq wg256",
                    kind: v2::V2Kernel::WarpQ,
                    cfg: v2::V2Config::new(256, 1),
                    oracle: Oracle::Quad,
                    pk: false,
                },
                RouteArm {
                    label: "mrow wg64 mr2",
                    kind: v2::V2Kernel::MRow,
                    cfg: v2::V2Config::new(64, 2),
                    oracle: Oracle::Pair,
                    pk: false,
                },
            ],
        );
    }
}

#[test]
fn nvfp4_moe_decode_alu_ablation_at_routed_dispatch() {
    let Some(ctx) = ctx_or_skip("nvfp4_moe_decode_alu_ablation_at_routed_dispatch") else {
        return;
    };
    eprintln!("=== MoE decode-ALU ablation (A1 is numerically WRONG, timing only) ===");
    for (tag, n, k) in [
        ("moe_gate_up", MOE_SLOTS * 1024, 2048usize),
        ("moe_down", MOE_SLOTS * 2048, 512),
    ] {
        ablation_shape(ctx, tag, n, k);
    }
}

#[test]
fn the_int_and_float_quad_oracles_agree_exactly() {
    for (n, k) in [(64usize, 1024usize), (33, 2048)] {
        let d = Data::new(n, k);
        assert_eq!(
            oracle_rows(&d, Oracle::Quad),
            oracle_rows(&d, Oracle::QuadFloat),
            "n={n} k={k}: e2m1 products are exact, so the int8 and f32 dots must coincide"
        );
    }
}

const SHIPPING_TREE_ENTRIES: &[&str] = &["g4w_gemv_nvfp4_pk", "q3w_gemv_nvfp4"];

#[test]
fn the_ladder_tree_arm_only_matches_the_shipping_kernel_under_the_pending_gate() {
    for e in SHIPPING_TREE_ENTRIES {
        assert!(
            dispatch::nozi_entry_listed(e, false),
            "{e} runs the tree kernel zero-init-free in production today"
        );
    }
    assert!(
        !dispatch::nozi_entry_listed(g4::GEMV_ENTRY, false),
        "{} is unaudited by default, so the ladder baseline zero-inits gemv_partial[256] \
         that the shipping kernel does not",
        g4::GEMV_ENTRY
    );
    assert!(
        dispatch::nozi_entry_listed(g4::GEMV_ENTRY, true),
        "NV_WGPU_NOZI_NVFP4_V2=1 must put the ladder baseline on the shipping policy"
    );
}

#[test]
fn every_timed_v2_arm_is_zero_init_free_without_the_gate() {
    for v in matrix() {
        let Kind::V2(_) = v.kind else { continue };
        let e = v.entry();
        assert!(
            !dispatch::nozi_entry_listed(e, false) && !dispatch::nozi_entry_listed(e, true),
            "{e} is listed; the timed v2 entries declare no workgroup storage, so listing one \
             would only hide which arm the option actually moved"
        );
    }
}

#[test]
fn the_only_v2_workgroup_array_is_untouched_by_fmlut_pk() {
    let src = v2::source(v2::V2Config::new(128, 4));
    assert_eq!(
        src.matches("var<workgroup>").count(),
        1,
        "a second workgroup array would invalidate the zero-init reasoning below"
    );
    let (_, tail) = src
        .split_once(&format!("fn {}(", v2::FMLUT_PK_ENTRY))
        .expect("fmlut_pk entry present");
    let body = tail.split("@compute").next().unwrap_or(tail);
    assert!(
        !body.contains("nv2_pk_bits"),
        "{} reaches no workgroup storage, so listing it is inert rather than a speedup",
        v2::FMLUT_PK_ENTRY
    );
}

#[test]
fn nvfp4_variant_matrix_gemma4_31b() {
    let Some(ctx) = ctx_or_skip("nvfp4_variant_matrix_gemma4_31b") else {
        return;
    };
    eprintln!("=== Gemma-4-31B-IT-NVFP4 NVFP4 GEMV variant matrix ===");
    let mut failures = Vec::new();
    for (tag, n, k) in [("gate_up", 43008usize, 5376usize), ("down", 5376, 21504)] {
        shape_matrix(ctx, tag, n, k, &mut failures);
    }
    assert!(failures.is_empty(), "non-bit-exact variants: {failures:#?}");
}

#[test]
fn nvfp4_variant_matrix_qwen38_dense() {
    let Some(ctx) = ctx_or_skip("nvfp4_variant_matrix_qwen38_dense") else {
        return;
    };
    eprintln!("=== Qwen3.8-27B-NVFP4 dense single-slot GEMV variant matrix ===");
    let mut failures = Vec::new();
    for (tag, n, k) in [("q38_gate_up", 17408usize, 5120usize), ("q38_down", 5120, 17408)] {
        shape_matrix(ctx, tag, n, k, &mut failures);
    }
    assert!(failures.is_empty(), "non-bit-exact variants: {failures:#?}");
}

#[test]
fn nvfp4_variant_matrix_qwen3_moe() {
    let Some(ctx) = ctx_or_skip("nvfp4_variant_matrix_qwen3_moe") else {
        return;
    };
    eprintln!("=== Qwen3.6-35B-A3B-NVFP4 NVFP4 GEMV variant matrix ===");
    let mut failures = Vec::new();
    for (tag, n, k) in [
        ("moe_gate", 512usize, 2048usize),
        ("moe_down", 2048, 512),
        ("attn_q", 8192, 2048),
        ("attn_kv", 512, 2048),
        ("attn_o", 2048, 4096),
    ] {
        shape_matrix(ctx, tag, n, k, &mut failures);
    }
    assert!(failures.is_empty(), "non-bit-exact variants: {failures:#?}");
}

#[test]
fn nvfp4_variants_are_bit_exact_on_ragged_shapes() {
    let Some(ctx) = ctx_or_skip("nvfp4_variants_ragged") else {
        return;
    };
    let mut failures = Vec::new();
    for (n, k) in [(300usize, 1024usize), (37, 2048), (129, 512), (1000, 4096)] {
        let d = Data::new(n, k);
        let wb = dispatch::storage_from_slice(ctx, "rg-w", &d.w);
        let wsb = dispatch::storage_from_slice(ctx, "rg-ws", &d.ws);
        let xb = dispatch::storage_from_slice(ctx, "rg-x", &d.x);
        let xsb = dispatch::storage_from_slice(ctx, "rg-xs", &d.xs);
        let yb = dispatch::storage_zeroed(ctx, "rg-y", (n * 4) as u64);
        for v in matrix() {
            if !v.shape_ok(k) {
                continue;
            }
            let groups = dispatch::workgroup_count_1d(ctx, n as u64, v.rows_per_group());
            let params = g4::gemv_params(1.0, n, k, groups.0);
            let pbuf = dispatch::uniform_from(ctx, "rg-p", &params);
            let pipeline =
                dispatch::cached_compute_pipeline(ctx, "nvfp4-v2-rg", &v.source(), v.entry())
                    .expect("pipeline");
            let b: Vec<(u32, &wgpu::Buffer)> = if v.vec4_slots() {
                vec![
                    (v2::WS_SLOT, &wsb),
                    (v2::XS_SLOT, &xsb),
                    (v2::PARAMS_SLOT, &pbuf),
                    (v2::Y_SLOT, &yb),
                    (v2::W4_SLOT, &wb),
                    (v2::X4_SLOT, &xb),
                ]
            } else {
                vec![
                    (v2::W2_SLOT, &wb),
                    (v2::WS_SLOT, &wsb),
                    (v2::X2_SLOT, &xb),
                    (v2::XS_SLOT, &xsb),
                    (v2::PARAMS_SLOT, &pbuf),
                    (v2::Y_SLOT, &yb),
                ]
            };
            dispatch::dispatch(ctx, &pipeline, &b, groups).expect("dispatch");
            let words: Vec<u32> = dispatch::read_back(ctx, &yb, n).expect("read_back");
            let got: Vec<u16> = words.iter().map(|w| (*w & 0xffff) as u16).collect();
            let want = oracle_rows(&d, v.oracle);
            let (ok, verdict) = diff_report(&want, &got);
            eprintln!("ragged n={n:<5} k={k:<5} | {:<22} {verdict}", v.label());
            if !ok {
                failures.push(format!("ragged n={n} k={k} {} {verdict}", v.label()));
            }
        }
    }
    assert!(failures.is_empty(), "non-bit-exact variants: {failures:#?}");
}

fn run_once(ctx: &WgpuContext, v: &Variant, d: &Data, n: usize, k: usize) -> Vec<u16> {
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, v.rows_per_group());
    let params = g4::gemv_params(1.0, n, k, groups.0);
    let pbuf = dispatch::uniform_from(ctx, "xk-p", &params);
    let wb = dispatch::storage_from_slice(ctx, "xk-w", &d.w);
    let wsb = dispatch::storage_from_slice(ctx, "xk-ws", &d.ws);
    let xb = dispatch::storage_from_slice(ctx, "xk-x", &d.x);
    let xsb = dispatch::storage_from_slice(ctx, "xk-xs", &d.xs);
    let yb = dispatch::storage_zeroed(ctx, "xk-y", (n * 4) as u64);
    let pipeline =
        dispatch::cached_compute_pipeline(ctx, "xk", &v.source(), v.entry()).expect("pipeline");
    let b: Vec<(u32, &wgpu::Buffer)> = if v.vec4_slots() {
        vec![
            (v2::WS_SLOT, &wsb),
            (v2::XS_SLOT, &xsb),
            (v2::PARAMS_SLOT, &pbuf),
            (v2::Y_SLOT, &yb),
            (v2::W4_SLOT, &wb),
            (v2::X4_SLOT, &xb),
        ]
    } else {
        vec![
            (v2::W2_SLOT, &wb),
            (v2::WS_SLOT, &wsb),
            (v2::X2_SLOT, &xb),
            (v2::XS_SLOT, &xsb),
            (v2::PARAMS_SLOT, &pbuf),
            (v2::Y_SLOT, &yb),
        ]
    };
    dispatch::dispatch(ctx, &pipeline, &b, groups).expect("dispatch");
    let words: Vec<u32> = dispatch::read_back(ctx, &yb, n).expect("read_back");
    words.iter().map(|w| (*w & 0xffff) as u16).collect()
}

#[test]
fn mrow2_pk_packs_the_scalar_pair_bit_for_bit() {
    let Some(ctx) = ctx_or_skip("mrow2_pk_packs_the_scalar_pair_bit_for_bit") else {
        return;
    };
    for (n, k) in [(300usize, 1024usize), (37, 2048), (129, 512)] {
        let d = Data::new(n, k);
        let want = oracle_rows(&d, Oracle::Pair);
        let cfg = v2::V2Config::new(128, 2);
        let kern = v2::V2Kernel::MRow2;
        let wb = dispatch::storage_from_slice(ctx, "pk2-w", &d.w);
        let wsb = dispatch::storage_from_slice(ctx, "pk2-ws", &d.ws);
        let xb = dispatch::storage_from_slice(ctx, "pk2-x", &d.x);
        let xsb = dispatch::storage_from_slice(ctx, "pk2-xs", &d.xs);
        let yb = dispatch::storage_zeroed(ctx, "pk2-y", (n.div_ceil(2) * 4) as u64);
        let groups = dispatch::workgroup_count_1d(ctx, n as u64, cfg.rows_per_group(kern));
        let params = g4::gemv_params(1.0, n, k, groups.0);
        let pbuf = dispatch::uniform_from(ctx, "pk2-p", &params);
        let entry = kern.pk_entry().expect("mrow2 pk entry");
        let pipeline =
            dispatch::cached_compute_pipeline(ctx, "nvfp4-v2-pk2", &v2::source(cfg), entry)
                .expect("pipeline");
        let b: Vec<(u32, &wgpu::Buffer)> = vec![
            (v2::WS_SLOT, &wsb),
            (v2::XS_SLOT, &xsb),
            (v2::PARAMS_SLOT, &pbuf),
            (v2::Y_SLOT, &yb),
            (v2::W4_SLOT, &wb),
            (v2::X4_SLOT, &xb),
        ];
        dispatch::dispatch(ctx, &pipeline, &b, groups).expect("dispatch");
        let words: Vec<u32> = dispatch::read_back(ctx, &yb, n.div_ceil(2)).expect("read_back");
        let mut got = Vec::with_capacity(n);
        for w in words {
            got.push((w & 0xffff) as u16);
            got.push((w >> 16) as u16);
        }
        got.truncate(n);
        assert_eq!(
            want, got,
            "n={n} k={k}: {entry} must pack exactly what the scalar pair oracle produces"
        );
    }
}

const Q3M_V2_WGSL: &str = include_str!("../wgsl/q3m_gemv_nvfp4_v2.wgsl");

#[test]
fn every_q3w_v2_entry_builds_a_pipeline_from_the_model_composition() {
    let Some(ctx) = ctx_or_skip("every_q3w_v2_entry_builds_a_pipeline_from_the_model_composition")
    else {
        return;
    };
    let cfg = v2::V2Config::new(128, 2);
    let src = nv_kernels::wgpu_backend::compose(&format!(
        "{}\n{}",
        v2::helpers(cfg),
        Q3M_V2_WGSL
    ));
    for entry in [
        "q3w_gemv_nvfp4_fmlut",
        "q3w_gemv_nvfp4_mrow",
        "q3w_gemv_nvfp4_mrow2",
        "q3w_gemv_nvfp4_fdec",
        "q3w_gemv_nvfp4_warp",
    ] {
        dispatch::cached_compute_pipeline(ctx, "q3m-v2-entries", &src, entry)
            .unwrap_or_else(|e| panic!("{entry}: the composed q3m module must build: {e}"));
    }
}

#[test]
fn tree_and_fmlut_differ_only_by_block_summation_order() {
    let Some(ctx) = ctx_or_skip("tree_and_fmlut_differ_only_by_block_summation_order") else {
        return;
    };
    let tree = Variant {
        name: "tree",
        kind: Kind::Tree,
        cfg: v2::V2Config::new(256, 1),
        oracle: Oracle::Tree,
    };
    let fmlut = Variant {
        name: "F fmlut",
        kind: Kind::V2(v2::V2Kernel::FMLut),
        cfg: v2::V2Config::new(128, 4),
        oracle: Oracle::PairPrescale,
    };
    let wide = |d: &mut Data, seed: u64| {
        let mut rng = Lcg(seed);
        let word = |r: &mut Lcg| {
            let mut w = 0u32;
            for b in 0..4 {
                w |= ((r.next_u32() & 0x7f) | 0x08) << (8 * b);
            }
            w
        };
        for v in d.ws.iter_mut() {
            *v = word(&mut rng);
        }
        for v in d.xs.iter_mut() {
            *v = word(&mut rng);
        }
    };
    let mut worst = 0i32;
    for (tag, n, k) in [
        ("31b gate_up", 8192usize, 5376usize),
        ("31b down", 5376, 21504),
        ("moe q", 4096, 2048),
        ("moe o", 2048, 4096),
        ("31b down wide-sf", 5376, 21504),
        ("31b gate_up wide-sf", 8192, 5376),
    ] {
        let mut d = Data::new(n, k);
        if tag.ends_with("wide-sf") {
            wide(&mut d, 0xd1ce ^ (k as u64));
        }
        let d = d;
        let a = run_once(ctx, &tree, &d, n, k);
        let b = run_once(ctx, &fmlut, &d, n, k);
        let mut off = 0usize;
        let mut max_ulp = 0i32;
        for (x, y) in a.iter().zip(b.iter()) {
            if x != y {
                off += 1;
                max_ulp = max_ulp.max((*x as i32 - *y as i32).abs());
            }
        }
        worst = worst.max(max_ulp);
        eprintln!(
            "xkernel {tag:<12} n={n:<6} k={k:<6} k_blocks={:<5} rows_differing={off}/{n} ({:.3}%) max_bf16_ulp={max_ulp}",
            d.k_blocks,
            100.0 * off as f64 / n as f64
        );
    }
    assert!(
        worst <= 1,
        "tree and fmlut disagree by more than one bf16 ulp (max {worst}); that is an indexing bug, not a summation-order effect"
    );
}
