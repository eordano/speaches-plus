#![cfg(feature = "wgpu")]

mod common;
use common::bf16_enc;
use common::ctx_or_skip_quiet_unqualified as ctx_or_skip;
use half::bf16;
use nv_kernels::wgpu_backend::compose;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemv_nvfp4 as g4;
use nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2 as v2;
use nv_kernels::wgpu_backend::kernels::quantize_nvfp4_bf16 as qz;

const W48_WGSL: &str = include_str!("../wgsl/gemv_nvfp4_w4a8.wgsl");
const QUANT_ENTRY: &str = "w48_quantize_x_evenodd";
const GEMV_ENTRY: &str = "gemv_nvfp4_w4a8_mrow2";
const ROWS_PER_GROUP: u32 = 8;
const SCALE_INDEX_ANCHOR: &str = "nvfp4_scale_byte_index(sr0, b0, w48_p.k_tiles)";

const ROWQUANT_D_REL_TOL_1E6_IS_VULKAN_DIVISION_ULP_SLACK_ON_AMAX_OVER_127: f32 = 1e-6;
const ROWQUANT_CODE_SLACK_ALLOWS_ONE_CODE_OF_TIE_FLIP: i32 = 1;
const E2E_VS_F64_ORACLE_REL_TOL_BUDGETS_PER_TENSOR_Q8_NOISE_AND_BF16_OUTPUT_ROUNDING: f32 = 3e-2;
const PLANTED_SWAP_MUST_MOVE_ROWS_BEYOND_ANY_ROUNDING: f32 = 1e-3;

fn w48_src() -> String {
    compose(W48_WGSL)
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_unit_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn scale_word(&mut self) -> u32 {
        let mut w = 0u32;
        for b in 0..4 {
            let exp = 1 + (self.next_u32() % 8);
            let mant = self.next_u32() % 8;
            w |= ((exp << 3) | mant) << (8 * b);
        }
        w
    }
}

fn gen_x_bf16(k: usize, seed: u64) -> Vec<u16> {
    let mut rng = Lcg::new(seed);
    (0..k)
        .map(|_| bf16::from_f32(rng.next_unit_f32()).to_bits())
        .collect()
}

fn pack_bf16(v: &[u16]) -> Vec<u32> {
    v.chunks(2)
        .map(|c| (c[0] as u32) | ((*c.get(1).unwrap_or(&0) as u32) << 16))
        .collect()
}

struct W48Data {
    n: usize,
    k_blocks: usize,
    k_tiles: usize,
    w: Vec<u32>,
    ws: Vec<u32>,
}

impl W48Data {
    fn new(n: usize, k: usize, seed: u64) -> Self {
        assert!(k % 32 == 0, "mrow2 family consumes block pairs, k must be 32-aligned");
        let k_blocks = k / 16;
        let mut rng = Lcg::new(seed);
        Self {
            n,
            k_blocks,
            k_tiles: g4::k_tiles(k_blocks),
            w: (0..n * k / 8).map(|_| rng.next_u32()).collect(),
            ws: (0..g4::swizzled_scale_len(n, k_blocks).div_ceil(4))
                .map(|_| rng.scale_word())
                .collect(),
        }
    }
}

fn i8map(s: u32) -> u32 {
    let k = s & 0x0707_0707;
    let hm = ((k >> 2) & 0x0101_0101).wrapping_mul(255);
    let e7 = (k & (k >> 1) & (k >> 2)) & 0x0101_0101;
    let m = k.wrapping_add((k & 0x0303_0303) & hm).wrapping_add(e7 << 1);
    let sb = (s & (k.wrapping_add(0x0707_0707) & 0x0808_0808)) >> 3;
    (m ^ sb.wrapping_mul(255)).wrapping_add(sb)
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

fn ue4m3(bits: u32) -> f32 {
    let b = bits & 127;
    if b < 8 {
        b as f32 * 0.001953125
    } else {
        f32::from_bits((b << 20).wrapping_add(0x3c00_0000))
    }
}

fn byte_at(word: u32, idx: usize) -> u32 {
    (word >> (8 * (idx & 3))) & 0xff
}

fn ws_idx(row: usize, block: usize, k_tiles: usize) -> usize {
    let m_tile = row / 128;
    let d2 = (row / 32) % 4;
    let d3 = row % 32;
    let k_tile = block / 4;
    let d5 = block % 4;
    ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5
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

struct QuantX {
    xq: Vec<u32>,
    d: f32,
}

fn host_quantize_evenodd(x: &[u16]) -> QuantX {
    assert!(x.len() % 8 == 0);
    let vals: Vec<f32> = x.iter().map(|b| bf16::from_bits(*b).to_f32()).collect();
    let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let dinv = if d != 0.0 { 1.0 / d } else { 0.0 };
    let code = |v: f32| -> u32 {
        let q = (v * dinv).clamp(-127.0, 127.0).round_ties_even() as i32;
        (q as u8) as u32
    };
    let mut xq = Vec::with_capacity(x.len() / 4);
    for g in 0..x.len() / 8 {
        let mut ev = 0u32;
        let mut od = 0u32;
        for j in 0..4 {
            ev |= code(vals[8 * g + 2 * j]) << (8 * j);
            od |= code(vals[8 * g + 2 * j + 1]) << (8 * j);
        }
        xq.push(ev);
        xq.push(od);
    }
    QuantX { xq, d }
}

fn oracle_row(d: &W48Data, xq: &[u32], xd: f32, alpha: f32, row: usize) -> u16 {
    let pairs = d.k_blocks / 2;
    let stride = d.k_blocks / 2;
    let mut lanes = [0f32; 32];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut acc = 0f32;
        let mut p = lane;
        while p < pairs {
            let b0 = 2 * p;
            let wo = 4 * (row * stride + p);
            let xo = 8 * p;
            let wsi = ws_idx(row, b0, d.k_tiles);
            let wsw = d.ws[wsi >> 2];
            let a0 = dot4i8(i8map(d.w[wo]), xq[xo])
                + dot4i8(i8map(d.w[wo] >> 4), xq[xo + 1])
                + dot4i8(i8map(d.w[wo + 1]), xq[xo + 2])
                + dot4i8(i8map(d.w[wo + 1] >> 4), xq[xo + 3]);
            let a1 = dot4i8(i8map(d.w[wo + 2]), xq[xo + 4])
                + dot4i8(i8map(d.w[wo + 2] >> 4), xq[xo + 5])
                + dot4i8(i8map(d.w[wo + 3]), xq[xo + 6])
                + dot4i8(i8map(d.w[wo + 3] >> 4), xq[xo + 7]);
            acc = ue4m3(byte_at(wsw, wsi)).mul_add(a0 as f32, acc);
            acc = ue4m3(byte_at(wsw, wsi + 1)).mul_add(a1 as f32, acc);
            p += 32;
        }
        *slot = acc;
    }
    let ep = (alpha * xd) * 0.5;
    bf16_enc(bfly(lanes) * ep)
}

fn oracle_rows(d: &W48Data, xq: &[u32], xd: f32, alpha: f32) -> Vec<u16> {
    let threads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(8)
        .min(24);
    let chunk = d.n.div_ceil(threads).max(1);
    let mut out = vec![0u16; d.n];
    std::thread::scope(|s| {
        for (ci, part) in out.chunks_mut(chunk).enumerate() {
            s.spawn(move || {
                for (j, o) in part.iter_mut().enumerate() {
                    *o = oracle_row(d, xq, xd, alpha, ci * chunk + j);
                }
            });
        }
    });
    out
}

fn dequant_oracle_f64(d: &W48Data, x: &[u16], alpha: f32) -> Vec<f64> {
    let k = d.k_blocks * 16;
    let xf: Vec<f64> = x.iter().map(|b| bf16::from_bits(*b).to_f64()).collect();
    (0..d.n)
        .map(|row| {
            let mut acc = 0f64;
            for kk in 0..k {
                let word = d.w[row * (k / 8) + kk / 8];
                let nib = (word >> (4 * (kk % 8))) & 0xf;
                let wsi = ws_idx(row, kk / 16, d.k_tiles);
                let s = ue4m3(byte_at(d.ws[wsi >> 2], wsi)) as f64;
                acc += common::reference_e2m1(nib as u8) as f64 * s * xf[kk];
            }
            acc * alpha as f64
        })
        .collect()
}

fn quant_groups(k: usize) -> (u32, u32, u32) {
    ((k / 8).div_ceil(256) as u32, 1, 1)
}

fn run_quantize(ctx: &WgpuContext, src: &str, x: &[u16], k_blocks: usize) -> QuantX {
    let params = g4::gemv_params(1.0, 1, k_blocks * 16, 1);
    let xb = dispatch::storage_from_slice(ctx, "w48-xin", &pack_bf16(x));
    let qb = dispatch::storage_from_slice(ctx, "w48-xq", &vec![0u32; x.len() / 4]);
    let sb = dispatch::storage_from_slice(ctx, "w48-xs", &[f32::NAN]);
    let ub = dispatch::uniform_from(ctx, "w48-qp", &params);
    dispatch::run(
        ctx,
        "w48-quant",
        src,
        QUANT_ENTRY,
        &[(6, &xb), (7, &qb), (8, &sb), (4, &ub)],
        quant_groups(k_blocks * 16),
    )
    .expect("quant dispatch");
    QuantX {
        xq: dispatch::read_back(ctx, &qb, x.len() / 4).expect("xq read back"),
        d: dispatch::read_back::<f32>(ctx, &sb, 1).expect("xs read back")[0],
    }
}

fn run_gemv(ctx: &WgpuContext, src: &str, d: &W48Data, q: &QuantX, alpha: f32) -> Vec<u16> {
    let groups = dispatch::workgroup_count_1d(ctx, d.n as u64, ROWS_PER_GROUP);
    let params = g4::gemv_params(alpha, d.n, d.k_blocks * 16, groups.0);
    let wb = dispatch::storage_from_slice(ctx, "w48-w", &d.w);
    let wsb = dispatch::storage_from_slice(ctx, "w48-ws", &d.ws);
    let qb = dispatch::storage_from_slice(ctx, "w48-x", &q.xq);
    let sb = dispatch::storage_from_slice(ctx, "w48-s", &[q.d]);
    let yb = dispatch::storage_from_slice(ctx, "w48-y", &vec![0xffffffffu32; d.n.div_ceil(2)]);
    let ub = dispatch::uniform_from(ctx, "w48-gp", &params);
    dispatch::run(
        ctx,
        "w48-gemv",
        src,
        GEMV_ENTRY,
        &[(0, &wb), (1, &wsb), (2, &qb), (3, &sb), (4, &ub), (5, &yb)],
        groups,
    )
    .expect("gemv dispatch");
    let words: Vec<u32> =
        dispatch::read_back(ctx, &yb, d.n.div_ceil(2)).expect("y read back");
    (0..d.n)
        .map(|r| {
            let w = words[r / 2];
            if r % 2 == 0 {
                (w & 0xffff) as u16
            } else {
                (w >> 16) as u16
            }
        })
        .collect()
}

fn ulp_stats(got: &[u16], want: &[u16]) -> (usize, i32, usize) {
    let mut off = 0usize;
    let mut max_ulp = 0i32;
    let mut first = usize::MAX;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g != w {
            if off == 0 {
                first = i;
            }
            off += 1;
            max_ulp = max_ulp.max((*g as i32 - *w as i32).abs());
        }
    }
    (off, max_ulp, first)
}

const PARITY_SHAPES: [(usize, usize, u64); 3] =
    [(17408, 5120, 0x51), (5120, 17408, 0x52), (333, 736, 0x53)];

#[test]
fn activation_rowquant_matches_host_emulation_within_division_ulp_slack() {
    let Some(ctx) = ctx_or_skip("w4a8_rowquant_parity") else {
        return;
    };
    let src = w48_src();
    for &(k, seed) in &[(5120usize, 0x41u64), (17408, 0x42), (736, 0x43)] {
        let x = gen_x_bf16(k, seed);
        let got = run_quantize(ctx, &src, &x, k / 16);
        let want = host_quantize_evenodd(&x);
        let rel = (got.d - want.d).abs() / want.d.abs().max(1e-30);
        assert!(
            rel <= ROWQUANT_D_REL_TOL_1E6_IS_VULKAN_DIVISION_ULP_SLACK_ON_AMAX_OVER_127,
            "k={k} seed={seed}: per-tensor scale d gpu={:e} host={:e} rel={rel:e}",
            got.d,
            want.d
        );
        for (i, (g, w)) in got.xq.iter().zip(&want.xq).enumerate() {
            for j in 0..4 {
                let gq = ((g >> (8 * j)) & 0xff) as u8 as i8 as i32;
                let wq = ((w >> (8 * j)) & 0xff) as u8 as i8 as i32;
                assert!(
                    (gq - wq).abs() <= ROWQUANT_CODE_SLACK_ALLOWS_ONE_CODE_OF_TIE_FLIP,
                    "k={k} seed={seed}: word {i} lane {j} gpu={gq} host={wq}: more than one \
                     code apart cannot come from Vulkan's 2.5-ULP division slack"
                );
            }
        }
    }
}

#[test]
fn gemv_is_bitexact_against_the_lane_lockstep_oracle_on_host_quantized_activations() {
    let Some(ctx) = ctx_or_skip("w4a8_gemv_parity") else {
        return;
    };
    let src = w48_src();
    let alpha = 0.75f32;
    for &(n, k, seed) in &PARITY_SHAPES {
        let d = W48Data::new(n, k, seed);
        let x = gen_x_bf16(k, seed ^ 0xfeed);
        let q = host_quantize_evenodd(&x);
        let got = run_gemv(ctx, &src, &d, &q, alpha);
        let want = oracle_rows(&d, &q.xq, q.d, alpha);
        let (off, max_ulp, first) = ulp_stats(&got, &want);
        assert_eq!(
            off, 0,
            "n={n} k={k}: {off} rows differ (max {max_ulp} ulp, first row {first}); both \
             sides do exact int dots in lockstep lane order, so any drift is an indexing \
             or scale-folding bug"
        );
    }
}

#[test]
fn gpu_rowquant_feeding_gemv_is_bitexact_on_its_own_stream_and_tracks_the_f64_oracle() {
    let Some(ctx) = ctx_or_skip("w4a8_e2e") else {
        return;
    };
    let src = w48_src();
    let alpha = 1.0f32;
    for &(n, k, seed) in &PARITY_SHAPES {
        let d = W48Data::new(n, k, seed ^ 0xe2e);
        let x = gen_x_bf16(k, seed ^ 0xabcd);
        let q = run_quantize(ctx, &src, &x, k / 16);
        let got = run_gemv(ctx, &src, &d, &q, alpha);
        let int_want = oracle_rows(&d, &q.xq, q.d, alpha);
        let (off, max_ulp, first) = ulp_stats(&got, &int_want);
        assert_eq!(
            off, 0,
            "n={n} k={k}: {off} rows differ from the int oracle on the GPU-quantized \
             stream (max {max_ulp} ulp, first row {first})"
        );
        let want = dequant_oracle_f64(&d, &x, alpha);
        let rms = (want.iter().map(|v| v * v).sum::<f64>() / want.len() as f64).sqrt();
        let mut worst = (0f64, 0usize);
        for (row, (g, w)) in got.iter().zip(&want).enumerate() {
            let gv = bf16::from_bits(*g).to_f64();
            let rel = (gv - w).abs() / w.abs().max(rms);
            if rel > worst.0 {
                worst = (rel, row);
            }
        }
        assert!(
            worst.0 < E2E_VS_F64_ORACLE_REL_TOL_BUDGETS_PER_TENSOR_Q8_NOISE_AND_BF16_OUTPUT_ROUNDING as f64,
            "n={n} k={k}: row {} rel {:.3e} vs the f64 dequant oracle exceeds the \
             per-tensor q8 activation budget",
            worst.1,
            worst.0
        );
        eprintln!("w4a8 e2e n={n} k={k}: worst rel vs f64 oracle {:.3e}", worst.0);
    }
}

#[test]
fn planted_scale_block_swap_is_caught_by_the_parity_harness() {
    let Some(ctx) = ctx_or_skip("w4a8_planted_bug") else {
        return;
    };
    let src = w48_src();
    let bad = src.replacen(
        SCALE_INDEX_ANCHOR,
        "nvfp4_scale_byte_index(sr0, b0 + 1u, w48_p.k_tiles)",
        1,
    );
    assert_ne!(
        bad, src,
        "scale-index anchor missing from gemv_nvfp4_w4a8.wgsl; the planted-bug gate is vacuous"
    );
    let (n, k, seed) = (65usize, 1024usize, 0x71u64);
    let d = W48Data::new(n, k, seed);
    let x = gen_x_bf16(k, seed ^ 0xbad);
    let q = host_quantize_evenodd(&x);
    let want = oracle_rows(&d, &q.xq, q.d, 1.0);
    let got = run_gemv(ctx, &bad, &d, &q, 1.0);
    let mut worst = 0f32;
    for (g, w) in got.iter().zip(&want) {
        let gv = bf16::from_bits(*g).to_f32();
        let wv = bf16::from_bits(*w).to_f32();
        worst = worst.max((gv - wv).abs() / wv.abs().max(1e-3));
    }
    assert!(
        worst >= PLANTED_SWAP_MUST_MOVE_ROWS_BEYOND_ANY_ROUNDING,
        "swapped scale-block index survived parity (worst rel {worst:.3e}); per-block \
         scales vary by construction, so the harness would miss a real scale-indexing \
         regression"
    );
}

const DRAM_HONEST_WEIGHT_REPLICAS_OVERFLOW_THE_128MB_BLACKWELL_L2: usize = 8;
const LADDER_ROUNDS: usize = 9;
const GEMV_DISPATCHES_PER_ROUND: usize = 400;
const STEP_ITERS_PER_ROUND: usize = 100;

struct Pass {
    pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    binds: Vec<wgpu::BindGroup>,
    groups: (u32, u32, u32),
}

struct Arm {
    label: String,
    passes: Vec<Pass>,
}

fn arm_cb(ctx: &WgpuContext, arm: &Arm, iters: usize) -> wgpu::CommandBuffer {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for i in 0..iters {
            for p in &arm.passes {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.binds[i % p.binds.len()], &[]);
                pass.dispatch_workgroups(p.groups.0, p.groups.1, p.groups.2);
            }
        }
    }
    enc.finish()
}

fn time_arm(ctx: &WgpuContext, arm: &Arm, iters: usize) -> f64 {
    let cb = arm_cb(ctx, arm, iters);
    let t0 = std::time::Instant::now();
    ctx.queue.submit([cb]);
    ctx.poll_blocking().expect("timed poll");
    t0.elapsed().as_secs_f64()
}

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

fn ladder(ctx: &WgpuContext, tag: &str, arms: &[Arm], iters: usize, bytes_per_iter: f64) {
    for arm in arms {
        for _ in 0..2 {
            time_arm(ctx, arm, iters);
        }
    }
    let mut rounds: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for _ in 0..LADDER_ROUNDS {
        for (i, arm) in arms.iter().enumerate() {
            rounds[i].push(time_arm(ctx, arm, iters));
        }
    }
    let base = &rounds[0];
    for (i, arm) in arms.iter().enumerate() {
        let rates: Vec<f64> = rounds[i]
            .iter()
            .map(|t| bytes_per_iter * iters as f64 / t / 1e9)
            .collect();
        let ratio: Vec<f64> = base
            .iter()
            .zip(&rounds[i])
            .map(|(b, t)| b / t)
            .collect();
        let ms: Vec<f64> = rounds[i].iter().map(|t| t * 1e3 / iters as f64).collect();
        eprintln!(
            "{tag} | {:<26} {:>8.4} ms/iter med  {:>7.1} GB/s med (min {:>6.1} max {:>6.1}) | \
             vs arm0 per-round {:.3}x med ({:.3}-{:.3})",
            arm.label,
            med(&ms),
            med(&rates),
            lo(&rates),
            hi(&rates),
            med(&ratio),
            lo(&ratio),
            hi(&ratio)
        );
    }
}

struct GemvBuffers {
    wbufs: Vec<wgpu::Buffer>,
    wsbufs: Vec<wgpu::Buffer>,
    ybuf: wgpu::Buffer,
    pbuf: wgpu::Buffer,
    groups: (u32, u32, u32),
}

fn gemv_buffers(ctx: &WgpuContext, d: &W48Data) -> GemvBuffers {
    let groups = dispatch::workgroup_count_1d(ctx, d.n as u64, ROWS_PER_GROUP);
    let params = g4::gemv_params(1.0, d.n, d.k_blocks * 16, groups.0);
    GemvBuffers {
        wbufs: (0..DRAM_HONEST_WEIGHT_REPLICAS_OVERFLOW_THE_128MB_BLACKWELL_L2)
            .map(|r| dispatch::storage_from_slice(ctx, &format!("w48b-w{r}"), &d.w))
            .collect(),
        wsbufs: (0..DRAM_HONEST_WEIGHT_REPLICAS_OVERFLOW_THE_128MB_BLACKWELL_L2)
            .map(|r| dispatch::storage_from_slice(ctx, &format!("w48b-ws{r}"), &d.ws))
            .collect(),
        ybuf: dispatch::storage_from_slice(ctx, "w48b-y", &vec![0u32; d.n]),
        pbuf: dispatch::uniform_from(ctx, "w48b-p", &params),
        groups,
    }
}

fn incumbent_pass(
    ctx: &WgpuContext,
    b: &GemvBuffers,
    xnib: &wgpu::Buffer,
    xs4: &wgpu::Buffer,
) -> Pass {
    let src = v2::source(v2::V2Config::new(128, 2));
    let pipeline = dispatch::cached_compute_pipeline(ctx, "w48b-mrow2", &src, v2::MROW2_ENTRY)
        .expect("incumbent pipeline");
    let binds = b
        .wbufs
        .iter()
        .zip(&b.wsbufs)
        .map(|(wb, wsb)| {
            dispatch::bind_group(
                ctx,
                &pipeline,
                &[
                    (v2::WS_SLOT, wsb),
                    (v2::XS_SLOT, xs4),
                    (v2::PARAMS_SLOT, &b.pbuf),
                    (v2::Y_SLOT, &b.ybuf),
                    (v2::W4_SLOT, wb),
                    (v2::X4_SLOT, xnib),
                ],
            )
        })
        .collect();
    Pass {
        pipeline,
        binds,
        groups: b.groups,
    }
}

fn w4a8_pass(
    ctx: &WgpuContext,
    src: &str,
    b: &GemvBuffers,
    xq: &wgpu::Buffer,
    xs: &wgpu::Buffer,
) -> Pass {
    let pipeline = dispatch::cached_compute_pipeline(ctx, "w48b-w4a8", src, GEMV_ENTRY)
        .expect("w4a8 pipeline");
    let binds = b
        .wbufs
        .iter()
        .zip(&b.wsbufs)
        .map(|(wb, wsb)| {
            dispatch::bind_group(
                ctx,
                &pipeline,
                &[
                    (0, wb),
                    (1, wsb),
                    (2, xq),
                    (3, xs),
                    (4, &b.pbuf),
                    (5, &b.ybuf),
                ],
            )
        })
        .collect();
    Pass {
        pipeline,
        binds,
        groups: b.groups,
    }
}

fn quant_pass(
    ctx: &WgpuContext,
    src: &str,
    k: usize,
    pbuf: &wgpu::Buffer,
    xin: &wgpu::Buffer,
    xq: &wgpu::Buffer,
    xs: &wgpu::Buffer,
) -> Pass {
    let pipeline = dispatch::cached_compute_pipeline(ctx, "w48b-quant", src, QUANT_ENTRY)
        .expect("quant pipeline");
    let binds = vec![dispatch::bind_group(
        ctx,
        &pipeline,
        &[(6, xin), (7, xq), (8, xs), (4, pbuf)],
    )];
    Pass {
        pipeline,
        binds,
        groups: quant_groups(k),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct QzActParams {
    global_scale: f32,
    k_blocks: u32,
    pad0: u32,
    pad1: u32,
}

fn w4a4_act_quant_pass(ctx: &WgpuContext, k: usize, xin: &wgpu::Buffer) -> Pass {
    let k_blocks = k / 16;
    let params = QzActParams {
        global_scale: 1.0,
        k_blocks: k_blocks as u32,
        ..Default::default()
    };
    let pbuf = dispatch::uniform_from(ctx, "w48b-qzp", &params);
    let packed = dispatch::storage_from_slice(ctx, "w48b-qzpk", &vec![0u32; k / 8]);
    let scales = dispatch::storage_from_slice(ctx, "w48b-qzsc", &vec![0u32; k_blocks]);
    let pipeline = dispatch::cached_compute_pipeline(
        ctx,
        "w48b-qzact",
        &qz::act_grid_source(),
        qz::ACT_GRID_ENTRY,
    )
    .expect("act grid pipeline");
    let binds = vec![dispatch::bind_group(
        ctx,
        &pipeline,
        &[(0, xin), (1, &pbuf), (2, &packed), (3, &scales)],
    )];
    std::mem::forget((pbuf, packed, scales));
    Pass {
        pipeline,
        binds,
        groups: (qz::act_grid_groups(k_blocks as u32, qz::ACT_GRID_WG), 1, 1),
    }
}

fn w4a4_x(ctx: &WgpuContext, k: usize, seed: u64) -> (wgpu::Buffer, wgpu::Buffer) {
    let mut rng = Lcg::new(seed);
    let xnib: Vec<u32> = (0..k / 8).map(|_| rng.next_u32()).collect();
    let xs4: Vec<u32> = (0..(k / 16).div_ceil(4)).map(|_| rng.scale_word()).collect();
    (
        dispatch::storage_from_slice(ctx, "w48b-xnib", &xnib),
        dispatch::storage_from_slice(ctx, "w48b-xs4", &xs4),
    )
}

const W4A8_PAYS_ONLY_WHEN_ROWQUANT_FUSES_INTO_ITS_PRODUCERS: &str =
    "w4a8 out-streams shipping w4a4 mrow2 at DRAM on both dense q38 shapes (gate/up ~1.05x, \
     down ~1.11x, NULL band ~+/-0.04x) but its standalone rowquant dispatch (~2.5-4us) \
     returns the 3-gemv MLP step to parity when charged unfused; against a w4a4 arm charged \
     the generic act-grid nvfp4 activation quantizer the w4a8 step wins ~1.08x. Routing arm \
     (orchestrator decision, default OFF, e.g. NV_NVFP4_V2_SELECT=w4a8_mrow2) should land \
     together with rowquant fused into the rmsnorm and silu-mul producers, as the CUDA \
     precedent does in rmsnorm_residual_writeout_rowquant_i8_m1 and silu_mul_rowquant_i8_m1.";

#[test]
#[ignore = "GPU rate measurement; run via nvk.sh probe with --nocapture"]
fn w4a8_beats_w4a4_mrow2_at_dram_but_the_unfused_rowquant_returns_the_step_to_parity_bench() {
    let Some(ctx) = ctx_or_skip("w4a8_vs_mrow2_bench") else {
        return;
    };
    eprintln!("{W4A8_PAYS_ONLY_WHEN_ROWQUANT_FUSES_INTO_ITS_PRODUCERS}");
    let src = w48_src();
    for &(n, k, seed) in &[(17408usize, 5120usize, 0xb1u64), (5120, 17408, 0xb2)] {
        let d = W48Data::new(n, k, seed);
        let bytes = (n * k / 2 + n * d.k_blocks) as f64;
        let x = gen_x_bf16(k, seed ^ 0xace);
        let q = host_quantize_evenodd(&x);
        let b = gemv_buffers(ctx, &d);
        let (xnib, xs4) = w4a4_x(ctx, k, seed ^ 0x44);
        let xqb = dispatch::storage_from_slice(ctx, "w48b-xq", &q.xq);
        let xsb = dispatch::storage_from_slice(ctx, "w48b-xs", &[q.d]);
        let xinb = dispatch::storage_from_slice(ctx, "w48b-xin", &pack_bf16(&x));
        let arms = vec![
            Arm {
                label: "w4a4 mrow2(128,2) SHIPPING".into(),
                passes: vec![incumbent_pass(ctx, &b, &xnib, &xs4)],
            },
            Arm {
                label: "w4a8 mrow2(128)".into(),
                passes: vec![w4a8_pass(ctx, &src, &b, &xqb, &xsb)],
            },
            Arm {
                label: "w4a8 quant+gemv (unamortized)".into(),
                passes: vec![
                    quant_pass(ctx, &src, k, &b.pbuf, &xinb, &xqb, &xsb),
                    w4a8_pass(ctx, &src, &b, &xqb, &xsb),
                ],
            },
            Arm {
                label: "w4a4 mrow2 [NULL]".into(),
                passes: vec![incumbent_pass(ctx, &b, &xnib, &xs4)],
            },
        ];
        eprintln!(
            "--- n={n} k={k}  {:.1} MiB weights+scales/iter  x{} replicas, {} dispatches/round, \
             {LADDER_ROUNDS} interleaved rounds ---",
            bytes / (1024.0 * 1024.0),
            DRAM_HONEST_WEIGHT_REPLICAS_OVERFLOW_THE_128MB_BLACKWELL_L2,
            GEMV_DISPATCHES_PER_ROUND
        );
        ladder(
            ctx,
            &format!("n={n} k={k}"),
            &arms,
            GEMV_DISPATCHES_PER_ROUND,
            bytes,
        );
    }

    let (hidden, inter, seed) = (5120usize, 17408usize, 0xc0u64);
    let gate = W48Data::new(inter, hidden, seed);
    let up = W48Data::new(inter, hidden, seed ^ 1);
    let down = W48Data::new(hidden, inter, seed ^ 2);
    let bg = gemv_buffers(ctx, &gate);
    let bu = gemv_buffers(ctx, &up);
    let bd = gemv_buffers(ctx, &down);
    let xh = gen_x_bf16(hidden, seed ^ 0x10);
    let xi = gen_x_bf16(inter, seed ^ 0x11);
    let qh = host_quantize_evenodd(&xh);
    let qi = host_quantize_evenodd(&xi);
    let (xnib_h, xs4_h) = w4a4_x(ctx, hidden, seed ^ 0x20);
    let (xnib_i, xs4_i) = w4a4_x(ctx, inter, seed ^ 0x21);
    let xqh = dispatch::storage_from_slice(ctx, "w48b-xqh", &qh.xq);
    let xsh = dispatch::storage_from_slice(ctx, "w48b-xsh", &[qh.d]);
    let xinh = dispatch::storage_from_slice(ctx, "w48b-xinh", &pack_bf16(&xh));
    let xqi = dispatch::storage_from_slice(ctx, "w48b-xqi", &qi.xq);
    let xsi = dispatch::storage_from_slice(ctx, "w48b-xsi", &[qi.d]);
    let xini = dispatch::storage_from_slice(ctx, "w48b-xini", &pack_bf16(&xi));
    let step_bytes = ((2 * (inter * hidden / 2 + inter * gate.k_blocks))
        + (hidden * inter / 2 + hidden * down.k_blocks)) as f64;
    let arms = vec![
        Arm {
            label: "w4a4 3-gemv step (quant-free)".into(),
            passes: vec![
                incumbent_pass(ctx, &bg, &xnib_h, &xs4_h),
                incumbent_pass(ctx, &bu, &xnib_h, &xs4_h),
                incumbent_pass(ctx, &bd, &xnib_i, &xs4_i),
            ],
        },
        Arm {
            label: "w4a4 2-actgrid+3-gemv step".into(),
            passes: vec![
                w4a4_act_quant_pass(ctx, hidden, &xinh),
                incumbent_pass(ctx, &bg, &xnib_h, &xs4_h),
                incumbent_pass(ctx, &bu, &xnib_h, &xs4_h),
                w4a4_act_quant_pass(ctx, inter, &xini),
                incumbent_pass(ctx, &bd, &xnib_i, &xs4_i),
            ],
        },
        Arm {
            label: "w4a8 2-quant+3-gemv step".into(),
            passes: vec![
                quant_pass(ctx, &src, hidden, &bg.pbuf, &xinh, &xqh, &xsh),
                w4a8_pass(ctx, &src, &bg, &xqh, &xsh),
                w4a8_pass(ctx, &src, &bu, &xqh, &xsh),
                quant_pass(ctx, &src, inter, &bd.pbuf, &xini, &xqi, &xsi),
                w4a8_pass(ctx, &src, &bd, &xqi, &xsi),
            ],
        },
        Arm {
            label: "w4a4 3-gemv step [NULL]".into(),
            passes: vec![
                incumbent_pass(ctx, &bg, &xnib_h, &xs4_h),
                incumbent_pass(ctx, &bu, &xnib_h, &xs4_h),
                incumbent_pass(ctx, &bd, &xnib_i, &xs4_i),
            ],
        },
    ];
    eprintln!(
        "--- MLP step hidden={hidden} inter={inter}  {:.1} MiB weights+scales/step  \
         x{} replicas, {} steps/round, {LADDER_ROUNDS} interleaved rounds, \
         arm0 charges w4a4 no activation quantize, arm1 charges it the generic \
         act-grid nvfp4 quantizer ---",
        step_bytes / (1024.0 * 1024.0),
        DRAM_HONEST_WEIGHT_REPLICAS_OVERFLOW_THE_128MB_BLACKWELL_L2,
        STEP_ITERS_PER_ROUND
    );
    ladder(ctx, "mlp_step", &arms, STEP_ITERS_PER_ROUND, step_bytes);
}
