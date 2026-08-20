#![cfg(feature = "wgpu")]

mod common;
use common::ctx;
use common::require;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemv_nvfp4 as g;
use common::LcgShift33W4a16Packs as Lcg;
use common::gpu_util;

const DRAM_PEAK_GBPS: f64 = 1792.0;

pub const DECODE_BASELINE: &str = r#"
fn gemv_ue4m3_decode(bits: u32) -> f32 {
    let b = bits & 255u;
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    return select(
        bitcast<f32>(((e + 120u) << 23u) | (m << 20u)),
        f32(m) * UE4M3_SUBNORMAL_STEP,
        e == 0u
    );
}

fn gemv_e2m1_decode(nibble: u32) -> f32 {
    let n = nibble & 15u;
    let s = (n >> 3u) << 31u;
    let e = (n >> 1u) & 3u;
    let m = n & 1u;
    let bits = select(m * 0x3f000000u, ((126u + e) << 23u) | (m << 22u), e != 0u);
    return bitcast<f32>(s | bits);
}

fn gemv_dot8(ww: u32, xw: u32, dot_in: f32) -> f32 {
    var dot = dot_in;
    for (var e = 0u; e < 8u; e = e + 2u) {
        let w_lo = gemv_e2m1_decode(nvfp4_nibble(ww, e));
        let w_hi = gemv_e2m1_decode(nvfp4_nibble(ww, e + 1u));
        let x_lo = gemv_e2m1_decode(nvfp4_nibble(xw, e));
        let x_hi = gemv_e2m1_decode(nvfp4_nibble(xw, e + 1u));
        dot = dot + fma(w_lo, x_lo, w_hi * x_hi);
    }
    return dot;
}
"#;

const UE4M3_KEEP: &str = r#"
fn gemv_ue4m3_decode(bits: u32) -> f32 {
    let b = bits & 255u;
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    return select(
        bitcast<f32>(((e + 120u) << 23u) | (m << 20u)),
        f32(m) * UE4M3_SUBNORMAL_STEP,
        e == 0u
    );
}
"#;

const E2M1_KEEP: &str = r#"
fn gemv_e2m1_decode(nibble: u32) -> f32 {
    let n = nibble & 15u;
    let k = n & 7u;
    let mag = select((k + 252u) << 22u, (k & 1u) * 0x3f000000u, k < 2u);
    return bitcast<f32>(((n & 8u) << 28u) | mag);
}
"#;

const DOT8_KEEP: &str = r#"
fn gemv_dot8(ww: u32, xw: u32, dot_in: f32) -> f32 {
    var dot = dot_in;
    for (var e = 0u; e < 8u; e = e + 2u) {
        let w_lo = gemv_e2m1_decode(nvfp4_nibble(ww, e));
        let w_hi = gemv_e2m1_decode(nvfp4_nibble(ww, e + 1u));
        let x_lo = gemv_e2m1_decode(nvfp4_nibble(xw, e));
        let x_hi = gemv_e2m1_decode(nvfp4_nibble(xw, e + 1u));
        dot = dot + fma(w_lo, x_lo, w_hi * x_hi);
    }
    return dot;
}
"#;

const UE4M3_NULL: &str = r#"
fn gemv_ue4m3_decode(bits: u32) -> f32 {
    return f32(bits & 1u);
}
"#;

const DOT8_NULL: &str = r#"
fn gemv_dot8(ww: u32, xw: u32, dot_in: f32) -> f32 {
    return dot_in + f32((ww ^ xw) & 1u);
}
"#;

const DOT8_HALF_W: &str = r#"
fn gemv_dot8(ww: u32, xw: u32, dot_in: f32) -> f32 {
    var dot = dot_in;
    for (var e = 0u; e < 8u; e = e + 2u) {
        let w_lo = gemv_e2m1_decode(nvfp4_nibble(ww, e));
        let w_hi = gemv_e2m1_decode(nvfp4_nibble(ww, e + 1u));
        let x_lo = f32(nvfp4_nibble(xw, e));
        let x_hi = f32(nvfp4_nibble(xw, e + 1u));
        dot = dot + fma(w_lo, x_lo, w_hi * x_hi);
    }
    return dot;
}
"#;

const UE4M3_LEAN: &str = r#"
fn gemv_ue4m3_decode(bits: u32) -> f32 {
    let b = bits & 127u;
    return select(
        bitcast<f32>((b << 20u) + 0x3c000000u),
        f32(b) * UE4M3_SUBNORMAL_STEP,
        b < 8u
    );
}
"#;

const E2M1_AT: &str = r#"
fn gemv_e2m1_decode(nibble: u32) -> f32 {
    let n = nibble & 15u;
    let k = n & 7u;
    let mag = select((k + 252u) << 22u, (k & 1u) * 0x3f000000u, k < 2u);
    return bitcast<f32>(((n & 8u) << 28u) | mag);
}

fn gemv_e2m1_at(w: u32, sh: u32) -> f32 {
    let k = (w >> sh) & 7u;
    let mag = select((k + 252u) << 22u, (k & 1u) * 0x3f000000u, k < 2u);
    return bitcast<f32>(((w << (28u - sh)) & 0x80000000u) | mag);
}
"#;

const DOT8_AT: &str = r#"
fn gemv_dot8(ww: u32, xw: u32, dot_in: f32) -> f32 {
    var dot = dot_in;
    for (var e = 0u; e < 8u; e = e + 2u) {
        let s0 = 4u * e;
        let s1 = s0 + 4u;
        let w_lo = gemv_e2m1_at(ww, s0);
        let w_hi = gemv_e2m1_at(ww, s1);
        let x_lo = gemv_e2m1_at(xw, s0);
        let x_hi = gemv_e2m1_at(xw, s1);
        dot = dot + fma(w_lo, x_lo, w_hi * x_hi);
    }
    return dot;
}
"#;

const E2M1_MAG: &str = r#"
fn gemv_e2m1_decode(nibble: u32) -> f32 {
    let n = nibble & 15u;
    let k = n & 7u;
    let mag = select((k + 252u) << 22u, (k & 1u) * 0x3f000000u, k < 2u);
    return bitcast<f32>(((n & 8u) << 28u) | mag);
}

fn gemv_e2m1_mag(w: u32, sh: u32) -> f32 {
    let k = (w >> sh) & 7u;
    return bitcast<f32>(select((k + 252u) << 22u, (k & 1u) * 0x3f000000u, k < 2u));
}
"#;

const DOT8_XORSIGN: &str = r#"
fn gemv_dot8(ww: u32, xw: u32, dot_in: f32) -> f32 {
    let sx = ww ^ xw;
    var dot = dot_in;
    for (var e = 0u; e < 8u; e = e + 2u) {
        let s0 = 4u * e;
        let s1 = s0 + 4u;
        let m_lo = gemv_e2m1_mag(ww, s0) * gemv_e2m1_mag(xw, s0);
        let m_hi = gemv_e2m1_mag(ww, s1) * gemv_e2m1_mag(xw, s1);
        let p_lo = bitcast<f32>(bitcast<u32>(m_lo) | ((sx << (28u - s0)) & 0x80000000u));
        let p_hi = bitcast<f32>(bitcast<u32>(m_hi) | ((sx << (28u - s1)) & 0x80000000u));
        dot = dot + (p_lo + p_hi);
    }
    return dot;
}
"#;

const E2M1_I8: &str = r#"
fn gemv_e2m1_decode(nibble: u32) -> f32 {
    let n = nibble & 15u;
    let k = n & 7u;
    let mag = select((k + 252u) << 22u, (k & 1u) * 0x3f000000u, k < 2u);
    return bitcast<f32>(((n & 8u) << 28u) | mag);
}

fn gemv_i8x4(w: u32, sh: u32) -> u32 {
    let a = (w >> sh) & 0xffffu;
    let s = (a | (a << 12u)) & 0x0f0f0f0fu;
    let k = s & 0x07070707u;
    let k1 = k >> 1u;
    let k2 = k >> 2u;
    let hi = k2 & 0x01010101u;
    let e7 = (k & k1 & k2) & 0x01010101u;
    let m = k + (k & (hi * 255u)) - (hi << 2u) + (e7 << 1u);
    let nz = (k | k1 | k2) & 0x01010101u;
    let sneg = ((s >> 3u) & 0x01010101u) & nz;
    return (m ^ (sneg * 255u)) + sneg;
}
"#;

const DOT8_DP4A: &str = r#"
fn gemv_dot8(ww: u32, xw: u32, dot_in: f32) -> f32 {
    let d = dot4I8Packed(gemv_i8x4(ww, 0u), gemv_i8x4(xw, 0u))
        + dot4I8Packed(gemv_i8x4(ww, 16u), gemv_i8x4(xw, 16u));
    return dot_in + f32(d) * 0.25;
}
"#;

const E2M1_I8_SPLIT: &str = r#"
fn gemv_e2m1_decode(nibble: u32) -> f32 {
    let n = nibble & 15u;
    let k = n & 7u;
    let mag = select((k + 252u) << 22u, (k & 1u) * 0x3f000000u, k < 2u);
    return bitcast<f32>(((n & 8u) << 28u) | mag);
}

fn gemv_i8map(s: u32) -> u32 {
    let k = s & 0x07070707u;
    let k1 = k >> 1u;
    let k2 = k >> 2u;
    let hm = (k2 & 0x01010101u) * 255u;
    let e7 = (k & k1 & k2) & 0x01010101u;
    let m = k + ((k & 0x03030303u) & hm) + (e7 << 1u);
    let nz = (k | k1 | k2) & 0x01010101u;
    let sb = ((s >> 3u) & 0x01010101u) & nz;
    return (m ^ (sb * 255u)) + sb;
}
"#;

const DOT8_DP4A_SPLIT: &str = r#"
fn gemv_dot8(ww: u32, xw: u32, dot_in: f32) -> f32 {
    let d = dot4I8Packed(gemv_i8map(ww & 0x0f0f0f0fu), gemv_i8map(xw & 0x0f0f0f0fu))
        + dot4I8Packed(gemv_i8map((ww >> 4u) & 0x0f0f0f0fu), gemv_i8map((xw >> 4u) & 0x0f0f0f0fu));
    return dot_in + f32(d) * 0.25;
}
"#;

const DOT8_DP4A_BARE: &str = r#"
fn gemv_dot8(ww: u32, xw: u32, dot_in: f32) -> f32 {
    let d = dot4I8Packed(gemv_i8map(ww), gemv_i8map(xw))
        + dot4I8Packed(gemv_i8map(ww >> 4u), gemv_i8map(xw >> 4u));
    return dot_in + f32(d) * 0.25;
}
"#;

const E2M1_I8_NZ3: &str = r#"
fn gemv_e2m1_decode(nibble: u32) -> f32 {
    let n = nibble & 15u;
    let k = n & 7u;
    let mag = select((k + 252u) << 22u, (k & 1u) * 0x3f000000u, k < 2u);
    return bitcast<f32>(((n & 8u) << 28u) | mag);
}

fn gemv_i8map(s: u32) -> u32 {
    let k = s & 0x07070707u;
    let hm = ((k >> 2u) & 0x01010101u) * 255u;
    let e7 = (k & (k >> 1u) & (k >> 2u)) & 0x01010101u;
    let m = k + ((k & 0x03030303u) & hm) + (e7 << 1u);
    let sb = (s & ((k + 0x07070707u) & 0x08080808u)) >> 3u;
    return (m ^ (sb * 255u)) + sb;
}
"#;

struct Cand {
    name: &'static str,
    decode: String,
    exact: bool,
    subs: Vec<(&'static str, &'static str)>,
}

fn cand(name: &'static str, decode: String, exact: bool) -> Cand {
    Cand {
        name,
        decode,
        exact,
        subs: Vec::new(),
    }
}

fn exact_ladder() -> Vec<Cand> {
    vec![
        cand(
            "P0 baseline (pre-lane decode)",
            DECODE_BASELINE.to_string(),
            true,
        ),
        cand(
            "P1 +lean e2m1",
            format!("{UE4M3_KEEP}\n{E2M1_KEEP}\n{DOT8_KEEP}"),
            true,
        ),
        cand(
            "P2 +lean ue4m3",
            format!("{UE4M3_LEAN}\n{E2M1_KEEP}\n{DOT8_KEEP}"),
            true,
        ),
        cand(
            "P3 +in-word nibble+sign (no BFE)",
            format!("{UE4M3_LEAN}\n{E2M1_AT}\n{DOT8_AT}"),
            true,
        ),
        cand(
            "P4 +hoisted xor sign",
            format!("{UE4M3_LEAN}\n{E2M1_MAG}\n{DOT8_XORSIGN}"),
            true,
        ),
        cand(
            "P5 swar i8 spread + dot4I8Packed",
            format!("{UE4M3_LEAN}\n{E2M1_I8}\n{DOT8_DP4A}"),
            true,
        ),
        cand(
            "P6 swar i8 nibble-split + dp4a",
            format!("{UE4M3_LEAN}\n{E2M1_I8_SPLIT}\n{DOT8_DP4A_SPLIT}"),
            true,
        ),
        cand(
            "P7 P6 + no pre-mask",
            format!("{UE4M3_LEAN}\n{E2M1_I8_SPLIT}\n{DOT8_DP4A_BARE}"),
            true,
        ),
        cand(
            "P8 P7 + carry-free sign guard",
            format!("{UE4M3_LEAN}\n{E2M1_I8_NZ3}\n{DOT8_DP4A_BARE}"),
            true,
        ),
    ]
}

fn diagnostics() -> Vec<Cand> {
    let mut out = vec![
        cand(
            "D1 DIAG no element decode",
            format!("{UE4M3_KEEP}\n{E2M1_KEEP}\n{DOT8_NULL}"),
            false,
        ),
        cand(
            "D2 DIAG no block-scale decode",
            format!("{UE4M3_NULL}\n{E2M1_KEEP}\n{DOT8_KEEP}"),
            false,
        ),
        cand(
            "D3 DIAG no decode at all",
            format!("{UE4M3_NULL}\n{E2M1_KEEP}\n{DOT8_NULL}"),
            false,
        ),
        cand(
            "D4 DIAG x-side decode removed",
            format!("{UE4M3_KEEP}\n{E2M1_KEEP}\n{DOT8_HALF_W}"),
            false,
        ),
    ];
    let mut noload = cand(
        "D5 DIAG scale loads pinned to index 0",
        format!("{UE4M3_KEEP}\n{E2M1_KEEP}\n{DOT8_KEEP}"),
        false,
    );
    noload.subs = vec![
        (
            "let ws = byte_at(gemv_w_scales[ws_idx >> 2u], ws_idx);",
            "let ws = byte_at(gemv_w_scales[0], ws_idx);",
        ),
        (
            "let xs = byte_at(gemv_x_scales[kb >> 2u], kb);",
            "let xs = byte_at(gemv_x_scales[0], kb);",
        ),
    ];
    out.push(noload);
    let mut noxw = cand(
        "D6 DIAG x vector pinned to block 0",
        format!("{UE4M3_KEEP}\n{E2M1_KEEP}\n{DOT8_KEEP}"),
        false,
    );
    noxw.subs = vec![("let xv = gemv_x_packed[kb];", "let xv = gemv_x_packed[0];")];
    out.push(noxw);
    out
}

fn cands() -> Vec<Cand> {
    let mut out = exact_ladder();
    out.extend(diagnostics());
    out
}

fn source_for(c: &Cand) -> String {
    let base = g::sg_gemv_source();
    let mut src = g::with_decode(&base, &c.decode).expect("decode markers present in sg source");
    for (from, to) in &c.subs {
        assert!(
            src.contains(from),
            "{}: substitution anchor missing: {from}",
            c.name
        );
        src = src.replace(from, to);
    }
    src
}

fn gpu_occupancy(tag: &str) {
    let smi = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ])
        .output();
    if let Ok(s) = smi {
        eprintln!(
            "occupancy[{tag}]: {}",
            String::from_utf8_lossy(&s.stdout).trim()
        );
    }
}

fn wait_idle(tag: &str) {
    let t0 = std::time::Instant::now();
    let mut streak = 0;
    loop {
        match gpu_util() {
            Some(u) if u <= 4 => streak += 1,
            Some(_) => streak = 0,
            None => return,
        }
        if streak >= 2 {
            eprintln!("{tag}: gpu idle after {:.0}s", t0.elapsed().as_secs_f64());
            return;
        }
        if t0.elapsed().as_secs_f64() > 300.0 {
            eprintln!("{tag}: WARNING gpu never idle within 300s; measuring anyway");
            return;
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}

struct Inputs {
    w_words: Vec<u32>,
    ws_words: Vec<u32>,
    x_words: Vec<u32>,
    xs_words: Vec<u32>,
    weight_bytes: f64,
}

fn make_inputs(n: usize, k: usize, seed: u64) -> Inputs {
    let mut rng = Lcg(seed);
    let k_blocks = k / 16;
    let w_words: Vec<u32> = (0..n * k / 8).map(|_| rng.next_u32()).collect();
    let scale_len = g::swizzled_scale_len(n, k_blocks);
    let scale_byte = |r: &mut Lcg| 0x30u32 | (r.next_u32() & 0x0f);
    let mk_scales = |r: &mut Lcg, len: usize| -> Vec<u32> {
        (0..len)
            .map(|_| {
                let mut w = 0u32;
                for b in 0..4 {
                    w |= scale_byte(r) << (8 * b);
                }
                w
            })
            .collect()
    };
    let ws_words = mk_scales(&mut rng, scale_len.div_ceil(4));
    let x_words: Vec<u32> = (0..k / 8).map(|_| rng.next_u32()).collect();
    let xs_words = mk_scales(&mut rng, k_blocks.div_ceil(4));
    let weight_bytes = (n * k / 2 + scale_len + k / 2 + k_blocks) as f64;
    Inputs {
        w_words,
        ws_words,
        x_words,
        xs_words,
        weight_bytes,
    }
}

struct Bufs {
    w0: wgpu::Buffer,
    w1: wgpu::Buffer,
    ws0: wgpu::Buffer,
    ws1: wgpu::Buffer,
    x: wgpu::Buffer,
    xs: wgpu::Buffer,
    y: wgpu::Buffer,
}

fn upload(ctx: &WgpuContext, inputs: &Inputs, n: usize) -> Bufs {
    Bufs {
        w0: dispatch::storage_from_slice(ctx, "dec-w0", &inputs.w_words),
        w1: dispatch::storage_from_slice(ctx, "dec-w1", &inputs.w_words),
        ws0: dispatch::storage_from_slice(ctx, "dec-ws0", &inputs.ws_words),
        ws1: dispatch::storage_from_slice(ctx, "dec-ws1", &inputs.ws_words),
        x: dispatch::storage_from_slice(ctx, "dec-x", &inputs.x_words),
        xs: dispatch::storage_from_slice(ctx, "dec-xs", &inputs.xs_words),
        y: dispatch::storage_zeroed(ctx, "dec-y", (n * 4) as u64),
    }
}

struct Rig {
    name: &'static str,
    exact: bool,
    pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    bg0: wgpu::BindGroup,
    bg1: wgpu::BindGroup,
    groups: (u32, u32, u32),
}

fn build(ctx: &WgpuContext, bufs: &Bufs, c: &Cand, n: usize, k: usize) -> Rig {
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, g::SG_ROWS_PER_GROUP);
    let params = g::gemv_params(1.0, n, k, groups.0);
    let pbuf = dispatch::uniform_from(ctx, "dec-params", &params);
    let src = source_for(c);
    let pipeline = dispatch::cached_compute_pipeline(ctx, c.name, &src, g::SG_GEMV_ENTRY)
        .unwrap_or_else(|e| panic!("pipeline {}: {e}", c.name));
    let mk = |w: &wgpu::Buffer, ws: &wgpu::Buffer| {
        dispatch::bind_group(
            ctx,
            &pipeline,
            &[
                (0, w),
                (1, ws),
                (2, &bufs.x),
                (3, &bufs.xs),
                (4, &pbuf),
                (5, &bufs.y),
            ],
        )
    };
    Rig {
        name: c.name,
        exact: c.exact,
        bg0: mk(&bufs.w0, &bufs.ws0),
        bg1: mk(&bufs.w1, &bufs.ws1),
        pipeline,
        groups,
    }
}

fn dispatch_n(ctx: &WgpuContext, rig: &Rig, count: usize) {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&rig.pipeline);
        for i in 0..count {
            pass.set_bind_group(0, if i % 2 == 0 { &rig.bg0 } else { &rig.bg1 }, &[]);
            pass.dispatch_workgroups(rig.groups.0, rig.groups.1, rig.groups.2);
        }
    }
    ctx.queue.submit([enc.finish()]);
}

fn time_rig(ctx: &WgpuContext, rig: &Rig, iters: usize) -> f64 {
    dispatch_n(ctx, rig, 10);
    ctx.poll_blocking().expect("warmup poll");
    let start = std::time::Instant::now();
    dispatch_n(ctx, rig, iters);
    ctx.poll_blocking().expect("timed poll");
    start.elapsed().as_secs_f64()
}

fn read_y(ctx: &WgpuContext, bufs: &Bufs, n: usize) -> Vec<u16> {
    let words: Vec<u32> = dispatch::read_back(ctx, &bufs.y, n).expect("read back");
    words.iter().map(|w| (*w & 0xffff) as u16).collect()
}

fn sweep(ctx: &'static WgpuContext, tag: &str, n: usize, k: usize, iters: usize) {
    let inputs = make_inputs(n, k, 0x51de ^ ((n as u64) << 24) ^ k as u64);
    let cs = cands();
    let bufs = upload(ctx, &inputs, n);
    let rigs: Vec<Rig> = cs.iter().map(|c| build(ctx, &bufs, c, n, k)).collect();

    let mut reference: Option<Vec<u16>> = None;
    for rig in &rigs {
        dispatch_n(ctx, rig, 1);
        ctx.poll_blocking().expect("exact poll");
        let y = read_y(ctx, &bufs, n);
        match &reference {
            None => {
                let nz = y.iter().filter(|v| **v != 0).count();
                assert!(nz * 4 >= n * 3, "{tag}: reference mostly zero, vacuous");
                reference = Some(y);
            }
            Some(r) if rig.exact => {
                let diff = r.iter().zip(y.iter()).filter(|(a, b)| a != b).count();
                assert_eq!(diff, 0, "{tag} n={n} k={k}: {} not bit-exact", rig.name);
            }
            _ => {}
        }
    }

    let mut best = vec![f64::MAX; rigs.len()];
    for _ in 0..5 {
        for (i, rig) in rigs.iter().enumerate() {
            best[i] = best[i].min(time_rig(ctx, rig, iters));
        }
    }
    let base_ms = best[0] * 1e3 / iters as f64;
    eprintln!(
        "-- {tag:<8} n={n:<7} k={k:<6} k_blocks={:<5} bytes/dispatch={:.3} GB",
        k / 16,
        inputs.weight_bytes / 1e9
    );
    for (rig, secs) in rigs.iter().zip(best.iter()) {
        let ms = secs * 1e3 / iters as f64;
        let gbps = inputs.weight_bytes * iters as f64 / secs / 1.0e9;
        eprintln!(
            "   {:<38} {:>9.4} ms {:>8.1} GB/s {:>5.1}% dram {:>+7.1}% vs A0{}",
            rig.name,
            ms,
            gbps,
            100.0 * gbps / DRAM_PEAK_GBPS,
            100.0 * (ms - base_ms) / base_ms,
            if rig.exact { "" } else { "  [INEXACT DIAG]" }
        );
    }
}

fn gemma4_shapes() -> Vec<(&'static str, usize, usize, usize)> {
    vec![
        ("gate_up", 43008, 5376, 60),
        ("down", 5376, 21504, 60),
        ("qkv", 16384, 5376, 100),
        ("o", 5376, 8192, 150),
    ]
}

#[test]
#[ignore]
fn decode_ablation_sweep() {
    let Some(ctx) = ctx("decode_ablation_sweep") else {
        return;
    };
    if !g::subgroup_ok(ctx) {
        eprintln!("decode_ablation_sweep: SKIP no fixed 32-wide subgroups");
        return;
    }
    eprintln!("decode_ablation_sweep: CONTAMINATED kernel-level microbenchmark, not end-to-end");
    wait_idle("decode_ablation_sweep");
    gpu_occupancy("before");
    for (tag, n, k, iters) in gemma4_shapes() {
        wait_idle(tag);
        sweep(ctx, tag, n, k, iters);
    }
    gpu_occupancy("after");
}

fn e2m1_baseline_bits(n: u32) -> u32 {
    let n = n & 15;
    let s = (n >> 3) << 31;
    let e = (n >> 1) & 3;
    let m = n & 1;
    let bits = if e != 0 {
        ((126 + e) << 23) | (m << 22)
    } else {
        m * 0x3f00_0000
    };
    s | bits
}

fn e2m1_lean_bits(n: u32) -> u32 {
    let n = n & 15;
    let k = n & 7;
    let mag = if k < 2 {
        (k & 1) * 0x3f00_0000
    } else {
        (k + 252) << 22
    };
    ((n & 8) << 28) | mag
}

#[test]
fn lean_e2m1_decode_is_bit_identical_on_all_16_codes() {
    let want = [
        0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    for n in 0u32..16 {
        let b = e2m1_baseline_bits(n);
        let l = e2m1_lean_bits(n);
        assert_eq!(
            b, l,
            "nibble {n}: baseline bits {b:#010x} != lean bits {l:#010x}"
        );
        assert_eq!(
            l,
            want[n as usize].to_bits(),
            "nibble {n}: lean decodes to {} not {}",
            f32::from_bits(l),
            want[n as usize]
        );
    }
}

#[test]
fn the_shipped_decode_block_is_the_lean_one_and_is_marker_wrapped() {
    let block = g::decode_block();
    assert!(
        block.contains("fn gemv_e2m1_decode("),
        "decode markers lost"
    );
    assert!(block.contains("fn gemv_dot8("));
    assert!(block.contains("fn gemv_ue4m3_decode("));
    assert!(
        block.contains("dot4I8Packed"),
        "shipped dot8 is not the packed integer form"
    );
    assert!(
        block.contains("let b = bits & 127u;"),
        "shipped ue4m3 is not the lean form"
    );
    assert!(
        !block.contains("E2M1_TABLE"),
        "hot decode regressed to a table lookup"
    );
    for src in [
        g::gemv_source(),
        g::sg_gemv_source(),
        g::sgw_source(g::SGW_DEEP),
    ] {
        let swapped = g::with_decode(&src, DECODE_BASELINE).expect("markers in composed source");
        assert!(swapped.contains("let bits = select(m * 0x3f000000u,"));
        assert!(!swapped.contains("let mag = select((k + 252u) << 22u,"));
    }
}

#[test]
fn every_ablation_candidate_compiles_to_a_distinct_source() {
    let mut seen: Vec<String> = Vec::new();
    for c in cands() {
        let src = source_for(&c);
        assert!(src.contains(g::SG_GEMV_ENTRY), "{}: entry lost", c.name);
        assert!(
            !seen.contains(&src),
            "{}: ablation source duplicates an earlier candidate",
            c.name
        );
        seen.push(src);
    }
    assert_eq!(seen.len(), 15);
}

fn ue4m3_baseline(bits: u32) -> f32 {
    let b = bits & 255;
    let e = (b >> 3) & 15;
    let m = b & 7;
    if e == 0 {
        m as f32 * 0.001953125
    } else {
        f32::from_bits(((e + 120) << 23) | (m << 20))
    }
}

fn ue4m3_lean(bits: u32) -> f32 {
    let b = bits & 127;
    if b < 8 {
        b as f32 * 0.001953125
    } else {
        f32::from_bits((b << 20) + 0x3c00_0000)
    }
}

#[test]
fn lean_ue4m3_decode_is_bit_identical_on_all_256_codes() {
    for b in 0u32..256 {
        assert_eq!(
            ue4m3_baseline(b).to_bits(),
            ue4m3_lean(b).to_bits(),
            "ue4m3 byte {b}: {} vs {}",
            ue4m3_baseline(b),
            ue4m3_lean(b)
        );
    }
}

fn swar_i8x4(w: u32) -> u32 {
    let a = w & 0xffff;
    let s = (a | (a << 12)) & 0x0f0f_0f0f;
    let k = s & 0x0707_0707;
    let k1 = k >> 1;
    let k2 = k >> 2;
    let hi = k2 & 0x0101_0101;
    let e7 = (k & k1 & k2) & 0x0101_0101;
    let m = k
        .wrapping_add(k & hi.wrapping_mul(255))
        .wrapping_sub(hi << 2)
        .wrapping_add(e7 << 1);
    let nz = (k | k1 | k2) & 0x0101_0101;
    let sneg = ((s >> 3) & 0x0101_0101) & nz;
    (m ^ sneg.wrapping_mul(255)).wrapping_add(sneg)
}

#[test]
fn the_swar_i8_map_is_exactly_two_times_the_e2m1_value() {
    for n0 in 0u32..16 {
        for n1 in 0u32..16 {
            for n2 in 0u32..16 {
                for n3 in 0u32..16 {
                    let w = n0 | (n1 << 4) | (n2 << 8) | (n3 << 12);
                    let packed = swar_i8x4(w);
                    let got = [
                        packed as u8 as i8,
                        (packed >> 8) as u8 as i8,
                        (packed >> 16) as u8 as i8,
                        (packed >> 24) as u8 as i8,
                    ];
                    let want = [n0, n2, n1, n3]
                        .map(|n| (2.0 * f32::from_bits(e2m1_baseline_bits(n))) as i8);
                    assert_eq!(got, want, "word {w:#06x} nibbles {n0} {n1} {n2} {n3}");
                }
            }
        }
    }
}

#[test]
fn the_dp4a_dot_is_exact_for_every_nibble_pair() {
    for wn in 0u32..16 {
        for xn in 0u32..16 {
            let w = wn * 0x1111;
            let x = xn * 0x1111;
            let vw = swar_i8x4(w);
            let vx = swar_i8x4(x);
            let mut d = 0i32;
            for b in 0..4 {
                d += ((vw >> (8 * b)) as u8 as i8 as i32) * ((vx >> (8 * b)) as u8 as i8 as i32);
            }
            let dp = d as f32 * 0.25;
            let wf = f32::from_bits(e2m1_baseline_bits(wn));
            let xf = f32::from_bits(e2m1_baseline_bits(xn));
            let mut want = 0.0f32;
            for _ in 0..4 {
                want += wf * xf;
            }
            assert_eq!(
                dp.to_bits(),
                want.to_bits(),
                "w nibble {wn} x nibble {xn}: dp4a {dp} vs fp {want}"
            );
        }
    }
}

fn swar_i8map(s: u32) -> u32 {
    let k = s & 0x0707_0707;
    let k1 = k >> 1;
    let k2 = k >> 2;
    let hm = (k2 & 0x0101_0101).wrapping_mul(255);
    let e7 = (k & k1 & k2) & 0x0101_0101;
    let m = k.wrapping_add((k & 0x0303_0303) & hm).wrapping_add(e7 << 1);
    let nz = (k | k1 | k2) & 0x0101_0101;
    let sb = ((s >> 3) & 0x0101_0101) & nz;
    (m ^ sb.wrapping_mul(255)).wrapping_add(sb)
}

fn swar_i8map_nz3(s: u32) -> u32 {
    let k = s & 0x0707_0707;
    let hm = ((k >> 2) & 0x0101_0101).wrapping_mul(255);
    let e7 = (k & (k >> 1) & (k >> 2)) & 0x0101_0101;
    let m = k.wrapping_add((k & 0x0303_0303) & hm).wrapping_add(e7 << 1);
    let sb = (s & (k.wrapping_add(0x0707_0707) & 0x0808_0808)) >> 3;
    (m ^ sb.wrapping_mul(255)).wrapping_add(sb)
}

#[test]
fn the_split_swar_maps_agree_with_e2m1_on_every_32_bit_nibble_word() {
    let mut rng = Lcg(0xfeed_5eed);
    let mut checked = 0usize;
    let mut words: Vec<u32> = (0..1u32 << 16).collect();
    words.extend((0..200_000).map(|_| rng.next_u32()));
    for w in words {
        for (tag, half, shift) in [("lo", w & 0x0f0f_0f0f, 0u32), ("hi", w >> 4, 4)] {
            for (name, got) in [
                ("split", swar_i8map(half)),
                ("bare", swar_i8map(if shift == 0 { w } else { w >> 4 })),
                ("nz3", swar_i8map_nz3(if shift == 0 { w } else { w >> 4 })),
            ] {
                for b in 0..4u32 {
                    let nib = (w >> (8 * b + shift)) & 15;
                    let want = (2.0 * f32::from_bits(e2m1_baseline_bits(nib))) as i8;
                    let g = (got >> (8 * b)) as u8 as i8;
                    assert_eq!(
                        g, want,
                        "{name}/{tag} word {w:#010x} byte {b} nibble {nib}: {g} != {want}"
                    );
                    checked += 1;
                }
            }
        }
    }
    eprintln!("split_swar: {checked} byte lanes verified exact");
}

fn exactness_shapes() -> Vec<(usize, usize)> {
    vec![
        (1, 16),
        (3, 32),
        (7, 4112),
        (129, 64),
        (37, 16 * 257),
        (255, 4096),
        (256, 5376),
        (1024, 21504),
        (67, 8192),
        (2048, 2048),
        (4096, 16),
        (5376, 8192),
    ]
}

#[test]
fn the_shipped_decode_is_bit_exact_against_the_pre_lane_decode() {
    let Some(ctx) = ctx("shipped_decode_bit_exact") else {
        return;
    };
    if !g::subgroup_ok(ctx) {
        if require() {
            panic!("shipped_decode_bit_exact: adapter lacks fixed 32-wide subgroups");
        }
        eprintln!("shipped_decode_bit_exact: SKIP no fixed 32-wide subgroups");
        return;
    }
    let shipped = cand(
        "shipped (markers untouched)",
        g::decode_block().to_string(),
        true,
    );
    let old = cand(
        "P0 baseline (pre-lane decode)",
        DECODE_BASELINE.to_string(),
        true,
    );
    let mut rows = 0usize;
    for (n, k) in exactness_shapes() {
        let inputs = make_inputs(n, k, 0xc0ffee ^ ((n as u64) << 20) ^ k as u64);
        let bufs = upload(ctx, &inputs, n);
        let want = {
            let rig = build(ctx, &bufs, &old, n, k);
            dispatch_n(ctx, &rig, 1);
            ctx.poll_blocking().expect("ref poll");
            read_y(ctx, &bufs, n)
        };
        let nz = want.iter().filter(|v| **v != 0).count();
        assert!(
            nz * 4 >= n * 3,
            "n={n} k={k}: reference mostly zero, vacuous"
        );
        let rig = build(ctx, &bufs, &shipped, n, k);
        dispatch_n(ctx, &rig, 1);
        ctx.poll_blocking().expect("got poll");
        let got = read_y(ctx, &bufs, n);
        let diff = want.iter().zip(got.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(
            diff, 0,
            "n={n} k={k}: {diff}/{n} rows differ from the old decode"
        );
        rows += n;
    }
    eprintln!(
        "shipped_decode_bit_exact: {rows} rows across {} shapes, 0 mismatches",
        exactness_shapes().len()
    );
}

struct VariantRig {
    name: &'static str,
    source: String,
    entry: &'static str,
    rpg: u32,
}

fn variant_rigs() -> Vec<VariantRig> {
    vec![
        VariantRig {
            name: "Tree wg256 r1",
            source: g::gemv_source(),
            entry: g::GEMV_ENTRY,
            rpg: 1,
        },
        VariantRig {
            name: "Sg   wg128 r4",
            source: g::sg_gemv_source(),
            entry: g::SG_GEMV_ENTRY,
            rpg: g::SG_ROWS_PER_GROUP,
        },
    ]
}

fn build_variant(ctx: &WgpuContext, bufs: &Bufs, v: &VariantRig, n: usize, k: usize) -> Rig {
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, v.rpg);
    let params = g::gemv_params(1.0, n, k, groups.0);
    let pbuf = dispatch::uniform_from(ctx, "var-params", &params);
    let pipeline = dispatch::cached_compute_pipeline(ctx, v.name, &v.source, v.entry)
        .unwrap_or_else(|e| panic!("pipeline {}: {e}", v.name));
    let mk = |w: &wgpu::Buffer, ws: &wgpu::Buffer| {
        dispatch::bind_group(
            ctx,
            &pipeline,
            &[
                (0, w),
                (1, ws),
                (2, &bufs.x),
                (3, &bufs.xs),
                (4, &pbuf),
                (5, &bufs.y),
            ],
        )
    };
    Rig {
        name: v.name,
        exact: true,
        bg0: mk(&bufs.w0, &bufs.ws0),
        bg1: mk(&bufs.w1, &bufs.ws1),
        pipeline,
        groups,
    }
}

#[test]
#[ignore]
fn tree_vs_sg_under_the_new_decode() {
    let Some(ctx) = ctx("tree_vs_sg") else {
        return;
    };
    if !g::subgroup_ok(ctx) {
        eprintln!("tree_vs_sg: SKIP no fixed 32-wide subgroups");
        return;
    }
    eprintln!("tree_vs_sg: CONTAMINATED kernel-level microbenchmark, not end-to-end");
    wait_idle("tree_vs_sg");
    gpu_occupancy("before");
    let vs = variant_rigs();
    for (tag, n, k, iters) in gemma4_shapes() {
        wait_idle(tag);
        let inputs = make_inputs(n, k, 0x1234 ^ ((n as u64) << 24) ^ k as u64);
        let bufs = upload(ctx, &inputs, n);
        let rigs: Vec<Rig> = vs
            .iter()
            .map(|v| build_variant(ctx, &bufs, v, n, k))
            .collect();
        let mut reference: Option<Vec<u16>> = None;
        for rig in &rigs {
            dispatch_n(ctx, rig, 1);
            ctx.poll_blocking().expect("poll");
            let y = read_y(ctx, &bufs, n);
            match &reference {
                None => reference = Some(y),
                Some(r) => {
                    let d = r.iter().zip(y.iter()).filter(|(a, b)| a != b).count();
                    assert_eq!(d, 0, "{tag}: {} not bit-exact vs tree", rig.name);
                }
            }
        }
        let mut best = vec![f64::MAX; rigs.len()];
        for _ in 0..5 {
            for (i, rig) in rigs.iter().enumerate() {
                best[i] = best[i].min(time_rig(ctx, rig, iters));
            }
        }
        let base = best[0] * 1e3 / iters as f64;
        eprintln!(
            "-- {tag:<8} n={n:<7} k={k:<6} blocks/lane={}",
            g::blocks_per_lane(k)
        );
        for (rig, secs) in rigs.iter().zip(best.iter()) {
            let ms = secs * 1e3 / iters as f64;
            let gbps = inputs.weight_bytes * iters as f64 / secs / 1.0e9;
            eprintln!(
                "   {:<20} {:>9.4} ms {:>8.1} GB/s {:>5.1}% dram {:>+7.1}% vs tree",
                rig.name,
                ms,
                gbps,
                100.0 * gbps / DRAM_PEAK_GBPS,
                100.0 * (ms - base) / base
            );
        }
    }
    gpu_occupancy("after");
}
