#![cfg(feature = "wgpu")]

use std::time::Instant;

use half::f16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::gemm_coop_f16 as coop;
use nv_kernels::wgpu_backend::kernels::gemm_spirv as spv;
use nv_kernels::wgpu_backend::{compose, dispatch};
mod common;
use common::ctx_or_panic as ctx;

const SLC_DEFEAT_BYTES: u64 = 1_800_000_000;

const SHIPPING_SLAB: u32 = 16;

fn require_coop(ctx: &WgpuContext) -> coop::CoopGemm {
    match coop::select(ctx, coop::Operand::F16) {
        Ok(g) => {
            eprintln!(
                "coop fragment selected from the adapter's advertised list: {}",
                g.request().label()
            );
            g
        }
        Err(why) => panic!("cooperative-matrix GEMM unavailable on this adapter: {why}"),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct MrowParams {
    n_rows: u32,
    k_elems: u32,
    row_words: u32,
    groups_x: u32,
    m: u32,
    y_stride: u32,
    pad0: u32,
    pad1: u32,
}

fn mrow_source(m: u32) -> String {
    use std::fmt::Write as _;
    let mut b = String::new();
    b.push_str("struct MrowParams {\n    n_rows: u32,\n    k_elems: u32,\n    row_words: u32,\n    groups_x: u32,\n    m: u32,\n    y_stride: u32,\n    pad0: u32,\n    pad1: u32,\n};\n\n");
    b.push_str("@group(0) @binding(0) var<storage, read> mr_w: array<u32>;\n");
    b.push_str("@group(0) @binding(1) var<storage, read> mr_x: array<u32>;\n");
    b.push_str("@group(0) @binding(2) var<storage, read_write> mr_y: array<f32>;\n");
    b.push_str("@group(0) @binding(3) var<uniform> mr_p: MrowParams;\n\n");
    b.push_str("const MR_LANES: u32 = 32u;\nconst MR_ROWS: u32 = 8u;\n\n");
    b.push_str("var<workgroup> mr_partial: array<f32, 256>;\n\n");
    b.push_str("fn mr_reduce(tid: u32, lane: u32, acc: f32) -> f32 {\n    workgroupBarrier();\n    mr_partial[tid] = acc;\n    workgroupBarrier();\n    for (var stride = MR_LANES >> 1u; stride > 0u; stride = stride >> 1u) {\n        if (lane < stride) {\n            mr_partial[tid] = mr_partial[tid] + mr_partial[tid + stride];\n        }\n        workgroupBarrier();\n    }\n    return mr_partial[tid - lane];\n}\n\n");
    b.push_str("@compute @workgroup_size(256)\n");
    writeln!(b, "fn mrow_bf16_m{m}(").unwrap();
    b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>\n) {\n");
    b.push_str("    let tid = lid.x;\n    let lane = tid & (MR_LANES - 1u);\n    let warp = tid / MR_LANES;\n");
    b.push_str("    let row = (wid.x + wid.y * mr_p.groups_x) * MR_ROWS + warp;\n");
    b.push_str("    let live = row < mr_p.n_rows;\n");
    b.push_str("    let kv = select(0u, mr_p.k_elems >> 3u, live);\n");
    b.push_str("    let w_base = select(0u, row * mr_p.row_words, live);\n");
    for t in 0..m {
        writeln!(b, "    var acc{t} = 0.0;").unwrap();
    }
    b.push_str("    for (var v = lane; v < kv; v = v + MR_LANES) {\n");
    b.push_str("        let wo = w_base + (v << 2u);\n        let xo = v << 2u;\n");
    b.push_str("        for (var j = 0u; j < 4u; j = j + 1u) {\n");
    b.push_str("            let ww = mr_w[wo + j];\n            let wl = bf16_lo(ww);\n            let wh = bf16_hi(ww);\n");
    for t in 0..m {
        writeln!(
            b,
            "            let xw{t} = mr_x[{t}u * mr_p.row_words + xo + j];\n            acc{t} = acc{t} + (wl * bf16_lo(xw{t}) + wh * bf16_hi(xw{t}));"
        )
        .unwrap();
    }
    b.push_str("        }\n    }\n");
    for t in 0..m {
        writeln!(b, "    {{\n        let total{t} = mr_reduce(tid, lane, acc{t});\n        if (lane == 0u && live) {{ mr_y[{t}u * mr_p.y_stride + row] = total{t}; }}\n    }}").unwrap();
    }
    b.push_str("}\n");
    compose(&b)
}

const F32_TOWARD_ZERO_UNIT: f64 = 1.0 / 8_388_608.0;

const OWN_ORDER_SLACK: f64 = 2.0;

const MROW_LANES: usize = 32;

fn oracle_f64(x: &[f32], w: &[f32], m: u32, n: u32, k: u32) -> Vec<f64> {
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let mut o = vec![0f64; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut s = 0f64;
            for j in 0..k {
                s += x[mi * k + j] as f64 * w[ni * k + j] as f64;
            }
            o[mi * n + ni] = s;
        }
    }
    o
}

fn rel_rms(got: &[f32], oracle: &[f64]) -> f64 {
    assert_eq!(got.len(), oracle.len());
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, o) in got.iter().zip(oracle.iter()) {
        let d = *g as f64 - *o;
        num += d * d;
        den += o * o;
    }
    assert!(
        den > 0.0,
        "the oracle is all zeros; the comparison is vacuous"
    );
    (num / den).sqrt()
}

fn max_abs_err(got: &[f32], oracle: &[f64]) -> f64 {
    got.iter()
        .zip(oracle.iter())
        .fold(0f64, |a, (g, o)| a.max((*g as f64 - *o).abs()))
}

fn round_toward_zero(v: f64) -> f32 {
    let r = v as f32;
    if r == 0.0 || (r as f64) == v || (r as f64).abs() < v.abs() {
        return r;
    }
    f32::from_bits(r.to_bits() - 1)
}

fn coop_order(x: &[f32], w: &[f32], m: u32, n: u32, k: u32, tile: u32, rn: bool) -> Vec<f32> {
    let (m, n, k, t) = (m as usize, n as usize, k as usize, tile as usize);
    let mut y = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut c = 0f32;
            for s in 0..k / t {
                let mut p = 0f64;
                for j in 0..t {
                    let e = s * t + j;
                    p += x[mi * k + e] as f64 * w[ni * k + e] as f64;
                }
                let sum = c as f64 + p;
                c = if rn {
                    sum as f32
                } else {
                    round_toward_zero(sum)
                };
            }
            y[mi * n + ni] = c;
        }
    }
    y
}

fn coop_order_f32(x: &[f32], w: &[f32], m: u32, n: u32, k: u32, tile: u32) -> Vec<f32> {
    coop_order(x, w, m, n, k, tile, true)
}

fn coop_order_rz(x: &[f32], w: &[f32], m: u32, n: u32, k: u32, tile: u32) -> Vec<f32> {
    coop_order(x, w, m, n, k, tile, false)
}

fn mrow_order_f32(x: &[f32], w: &[f32], m: u32, n: u32, k: u32) -> Vec<f32> {
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let mut y = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut lanes = [0f32; MROW_LANES];
            for (lane, acc) in lanes.iter_mut().enumerate() {
                let mut v = lane;
                while v < k / 8 {
                    for j in 0..4 {
                        let e = v * 8 + j * 2;
                        let p = w[ni * k + e] as f64 * x[mi * k + e] as f64
                            + w[ni * k + e + 1] as f64 * x[mi * k + e + 1] as f64;
                        *acc = (*acc as f64 + p) as f32;
                    }
                    v += MROW_LANES;
                }
            }
            let mut stride = MROW_LANES / 2;
            while stride > 0 {
                for l in 0..stride {
                    lanes[l] += lanes[l + stride];
                }
                stride >>= 1;
            }
            y[mi * n + ni] = lanes[0];
        }
    }
    y
}

fn coop_accum_units(
    x: &[f32],
    w: &[f32],
    m: u32,
    n: u32,
    k: u32,
    tile: u32,
    got: &[f32],
) -> (f64, f64) {
    let (mu, nu, ku, t) = (m as usize, n as usize, k as usize, tile as usize);
    assert_eq!(got.len(), mu * nu);
    assert!(
        ku >= t && ku % t == 0,
        "K={ku} is not whole {t}-wide fragments"
    );
    let mut mx = 0f64;
    let mut num = 0f64;
    let mut den = 0f64;
    for mi in 0..mu {
        for ni in 0..nu {
            let mut c = 0f64;
            let mut bound = 0f64;
            for s in 0..ku / t {
                let before = c;
                let mut frag_abs = 0f64;
                for j in 0..t {
                    let e = s * t + j;
                    let p = x[mi * ku + e] as f64 * w[ni * ku + e] as f64;
                    c += p;
                    frag_abs += p.abs();
                }
                bound += before.abs().max(c.abs()).max(frag_abs);
            }
            let d = (got[mi * nu + ni] as f64 - c).abs();
            let allowed = F32_TOWARD_ZERO_UNIT * bound;
            assert!(
                allowed > 0.0,
                "element ({mi},{ni}) has a zero rounding bound"
            );
            mx = mx.max(d / allowed);
            num += d * d;
            den += allowed * allowed;
        }
    }
    (mx, (num / den).sqrt())
}

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_unit(&mut self) -> f32 {
        ((self.next_u32() & 0xff_ffff) as f32 / 16777216.0) * 2.0 - 1.0
    }
}

fn shared_values(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|_| {
            let v = r.next_unit() * 0.5;
            let b = half::bf16::from_f32(v);
            b.to_f32()
        })
        .collect()
}

fn pack_bf16(v: &[f32]) -> Vec<u32> {
    let mut out = vec![0u32; v.len().div_ceil(2)];
    for (i, x) in v.iter().enumerate() {
        let bits = half::bf16::from_f32(*x).to_bits() as u32;
        out[i / 2] |= bits << (16 * (i % 2));
    }
    out
}

fn to_f16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| f16::from_f32(*x).to_bits()).collect()
}

struct Rig {
    pipeline: wgpu::ComputePipeline,
    group: Vec<wgpu::BindGroup>,
    groups: (u32, u32, u32),
    y: wgpu::Buffer,
    tm: u32,
    tn: u32,
    #[allow(dead_code)]
    keep: Vec<wgpu::Buffer>,
}

struct Replicas {
    buf: wgpu::Buffer,
    stride: u64,
    count: u64,
}

fn replicas<T: bytemuck::Pod>(ctx: &WgpuContext, label: &str, one: &[T], target: u64) -> Replicas {
    let bytes = std::mem::size_of_val(one) as u64;
    let align = 256u64;
    let stride = bytes.div_ceil(align) * align;
    let count = target.div_ceil(stride).clamp(1, 64);
    let max = ctx
        .caps
        .max_buffer_size
        .min(ctx.caps.max_storage_buffer_binding_size);
    let count = count.min((max / stride).max(1));
    let mut flat = vec![0u8; (stride * count) as usize];
    let src = bytemuck::cast_slice::<T, u8>(one);
    for i in 0..count {
        let o = (i * stride) as usize;
        flat[o..o + src.len()].copy_from_slice(src);
    }
    let buf = dispatch::storage_from_slice(ctx, label, &flat);
    Replicas { buf, stride, count }
}

fn build_coop_ab(
    ctx: &WgpuContext,
    g: coop::CoopGemm,
    w: &Replicas,
    x: &[f32],
    m: u32,
    n: u32,
    k: u32,
    cfg: (u32, u32, u32, u32),
) -> Rig {
    let ab = g.ab;
    let (tm, tn, sg, ku) = cfg;
    g.check_shape(m, n, k)
        .unwrap_or_else(|e| panic!("shape M={m} N={n} K={k} is not valid for {:?}: {e}", g));
    let src = g.source(tm, tn, sg, ku);
    let entry = g.entry(tm, tn, sg, ku);
    let pipeline = dispatch::compute_pipeline(ctx, "coop-gemm", &src, &entry).unwrap_or_else(|e| {
        panic!(
            "coop gemm pipeline ({} tm={tm} tn={tn} sg={sg} ku={ku}): {e}",
            g.request().label()
        )
    });
    let xbuf = match ab {
        coop::Operand::F16 => dispatch::storage_from_slice(ctx, "coop-x", &to_f16(x)),
        coop::Operand::F32 => dispatch::storage_from_slice(ctx, "coop-x", x),
    };
    let y = dispatch::storage_zeroed(ctx, "coop-y", (m as u64) * (n as u64) * 4);
    let zero = dispatch::storage_from_slice(ctx, "coop-zero", &vec![0f32; g.zero_elems()]);
    let (bm, bn) = g.grid(m, n, tm, tn, sg);
    let groups = dispatch::workgroup_count_1d(ctx, (bm as u64) * (bn as u64), 1);
    let p = coop::CoopGemmParams {
        n_rows: n,
        k_elems: k,
        m_rows: m,
        blocks_n: bn,
        y_stride: n,
        groups_x: groups.0,
        pad0: 0,
        pad1: 0,
    };
    let pbuf = dispatch::uniform_from(ctx, "coop-p", &p);
    let group = (0..w.count)
        .map(|i| {
            dispatch::bind_group_offsets(
                ctx,
                &pipeline,
                &[
                    (0, &w.buf, i * w.stride),
                    (1, &xbuf, 0),
                    (2, &y, 0),
                    (3, &pbuf, 0),
                    (4, &zero, 0),
                ],
            )
        })
        .collect();
    Rig {
        pipeline,
        group,
        groups,
        y,
        tm,
        tn,

        keep: vec![xbuf, zero, pbuf],
    }
}

fn build_mrow(ctx: &WgpuContext, w: &Replicas, x: &[f32], m: u32, n: u32, k: u32) -> Rig {
    let src = mrow_source(m);
    let entry = format!("mrow_bf16_m{m}");
    let pipeline = dispatch::compute_pipeline(ctx, "mrow-gemm", &src, &entry)
        .unwrap_or_else(|e| panic!("mrow pipeline m={m}: {e}"));
    let xbuf = dispatch::storage_from_slice(ctx, "mrow-x", &pack_bf16(x));
    let y = dispatch::storage_zeroed(ctx, "mrow-y", (m as u64) * (n as u64) * 4);
    let groups = dispatch::workgroup_count_1d(ctx, n.div_ceil(8) as u64, 1);
    let p = MrowParams {
        n_rows: n,
        k_elems: k,
        row_words: k / 2,
        groups_x: groups.0,
        m,
        y_stride: n,
        pad0: 0,
        pad1: 0,
    };
    let pbuf = dispatch::uniform_from(ctx, "mrow-p", &p);
    let group = (0..w.count)
        .map(|i| {
            dispatch::bind_group_offsets(
                ctx,
                &pipeline,
                &[
                    (0, &w.buf, i * w.stride),
                    (1, &xbuf, 0),
                    (2, &y, 0),
                    (3, &pbuf, 0),
                ],
            )
        })
        .collect();
    Rig {
        pipeline,
        group,
        groups,
        y,
        tm: 0,
        tn: 0,
        keep: vec![xbuf, pbuf],
    }
}

fn bench(ctx: &WgpuContext, rig: &Rig, passes: usize, reps: usize) -> f64 {
    let pipeline = &rig.pipeline;
    let groups = rig.groups;
    let submit = |n: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            for i in 0..n {
                pass.set_bind_group(0, &rig.group[i % rig.group.len()], &[]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
        }
        ctx.queue.submit([enc.finish()]);
        ctx.poll_blocking().unwrap();
    };
    submit(2);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        submit(passes);
        best = best.min(t0.elapsed().as_secs_f64() / passes as f64);
    }
    best
}

#[test]
fn coop_gemm_matches_f64_oracle_and_the_m_row_path() {
    let ctx = ctx();
    let g = require_coop(ctx);
    eprintln!("adapter: {}", ctx.summary());
    for (m, n, k) in [
        (16u32, 64u32, 128u32),
        (32, 128, 256),
        (64, 9 * g.tile, 512),
    ] {
        let w = shared_values((n * k) as usize, 0x5eed_0001);
        let x = shared_values((m * k) as usize, 0x5eed_0002);
        let wf16 = replicas(ctx, "oracle-w-f16", &to_f16(&w), 0);
        let wbf16 = replicas(ctx, "oracle-w-bf16", &pack_bf16(&w), 0);
        let (tm, tn) = g.tiles(m, coop::ACC_FRAGS);
        let cr = build_coop_ab(ctx, g, &wf16, &x, m, n, k, (tm, tn, 4, 1));
        let mr = build_mrow(ctx, &wbf16, &x, m, n, k);
        for r in [&cr, &mr] {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&r.pipeline);
                pass.set_bind_group(0, &r.group[0], &[]);
                pass.dispatch_workgroups(r.groups.0, r.groups.1, r.groups.2);
            }
            ctx.queue.submit([enc.finish()]);
        }
        ctx.poll_blocking().unwrap();
        let yc: Vec<f32> = dispatch::read_back(ctx, &cr.y, (m * n) as usize).unwrap();
        let ym: Vec<f32> = dispatch::read_back(ctx, &mr.y, (m * n) as usize).unwrap();

        let oracle = oracle_f64(&x, &w, m, n, k);
        let xh: Vec<f32> = x.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
        let wh: Vec<f32> = w.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
        assert_eq!(
            oracle_f64(&xh, &wh, m, n, k),
            oracle,
            "shared_values() must stay exactly representable in BOTH f16 and bf16: the coop path \
             feeds f16 operands and the m-row path bf16 ones, and every model below attributes the \
             whole gap between them to summation order. Once f16 rounds an operand that premise is \
             gone and these bounds need an operand term"
        );
        let xb: Vec<f32> = x
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_f32())
            .collect();
        let wb: Vec<f32> = w
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_f32())
            .collect();
        assert_eq!(
            oracle_f64(&xb, &wb, m, n, k),
            oracle,
            "bf16 rounds this operand set"
        );

        let crms = rel_rms(&yc, &oracle);
        let mrms = rel_rms(&ym, &oracle);
        let cpu_rn = rel_rms(&coop_order_f32(&x, &w, m, n, k, g.tile), &oracle);
        let cpu_rz = rel_rms(&coop_order_rz(&x, &w, m, n, k, g.tile), &oracle);
        let cpu_mrow = rel_rms(&mrow_order_f32(&x, &w, m, n, k), &oracle);
        let (units_max, units_rms) = coop_accum_units(&x, &w, m, n, k, g.tile, &yc);
        let cross = yc
            .iter()
            .zip(ym.iter())
            .fold(0f64, |a, (p, q)| a.max((*p as f64 - *q as f64).abs()));
        eprintln!(
            "  M={m:<3} N={n:<5} K={k:<5}  coop tm={} tn={}  |vs f64| max {:.3e} rel-rms {crms:.3e}   m-row max {:.3e} rel-rms {mrms:.3e}   max|coop-mrow| {cross:.3e}",
            cr.tm,
            cr.tn,
            max_abs_err(&yc, &oracle),
            max_abs_err(&ym, &oracle)
        );
        eprintln!(
            "        own-order emulation: coop round-to-nearest {cpu_rn:.3e}, coop round-toward-zero {cpu_rz:.3e} (gpu/rz {:.3}), m-row {cpu_mrow:.3e} (gpu/mrow {:.4});  coop accumulator-rounding units: max {units_max:.4} rms {units_rms:.4} of a 1.0 ceiling",
            crms / cpu_rz,
            mrms / cpu_mrow
        );
        assert!(
            crms < 3e-3,
            "coop GEMM rel-rms {crms:.3e} against the f64 oracle is not a rounding-order difference"
        );
        assert!(
            units_max <= 1.0,
            "coop GEMM at M={m} N={n} K={k} is {units_max:.4} f32 accumulator-rounding units off \
             the f64 oracle. The ceiling is 1.0 by the triangle inequality over the {} sequential \
             accumulate steps this kernel performs, so the kernel lost more than its own \
             accumulation order can explain -- a real precision defect, not the gate",
            k / g.tile
        );
        eprintln!(
            "        coop is {:.2}x the m-row path here, and that is reduction shape, not accuracy: both operand sets are exactly representable (asserted above), the coop path folds K/{} fragments into ONE f32 accumulator that rounds toward zero, so its error grows linearly in K; the m-row path uses a 32-lane pairwise tree with round-to-nearest, so its error grows like sqrt(log K). A bare ratio between the two measures the two reduction trees -- the defect task #54 found in the marlin gate.",
            crms / mrms,
            g.tile
        );
        assert!(
            crms > cpu_rn,
            "the coop path at M={m} N={n} K={k} is now {crms:.3e}, no worse than a ROUND-TO-NEAREST \
             emulation of its own summation order ({cpu_rn:.3e}). F32_TOWARD_ZERO_UNIT is a whole \
             2^-23 only because this adapter's cooperative-matrix accumulate truncates; if that \
             changed, halve the constant to 2^-24, re-derive the units below, and delete \
             coop_order_rz"
        );
        assert!(
            crms <= OWN_ORDER_SLACK * cpu_rz,
            "coop GEMM ({crms:.3e}) is more than {OWN_ORDER_SLACK}x a CPU emulation of its OWN \
             summation order and rounding mode ({cpu_rz:.3e}) at M={m} N={n} K={k}. The GPU \
             measured 0.91-0.95x of that emulation when this bound was set, so {OWN_ORDER_SLACK}x \
             is headroom, not a fitted threshold"
        );
        assert!(
            mrms <= OWN_ORDER_SLACK * cpu_mrow,
            "m-row GEMM ({mrms:.3e}) is more than {OWN_ORDER_SLACK}x a CPU emulation of its OWN \
             summation order ({cpu_mrow:.3e}) at M={m} N={n} K={k}. mrow_order_f32 reproduces this \
             kernel to four digits (gpu/mrow 0.9998-1.0006), so a miss here is the kernel, not the \
             model"
        );
    }
}

#[test]
fn the_accumulator_rounding_unit_metric_is_calibrated_and_can_fail() {
    assert_eq!(
        F32_TOWARD_ZERO_UNIT,
        f32::EPSILON as f64,
        "one f32 accumulate step that rounds TOWARD ZERO can lose a whole 2^-23, not the 2^-24 a \
         round-to-nearest step would: this adapter's cooperative-matrix accumulate truncates, \
         which is why coop_order_rz tracks the GPU within 10% at every K while coop_order_f32 -- \
         round-to-nearest over the identical summation order -- is 2.1x/2.9x/4.6x optimistic at \
         K=128/256/512"
    );
    let tile = 16u32;
    let k = 2 * tile;
    let mut x = vec![0f32; k as usize];
    let mut w = vec![0f32; k as usize];
    x[0] = 4.0;
    w[0] = 1.0;
    x[tile as usize] = -2.0;
    w[tile as usize] = 1.0;

    let want = 2.0f32;
    let bound = F32_TOWARD_ZERO_UNIT * (4.0 + 4.0);
    assert_eq!(
        coop_accum_units(&x, &w, 1, 1, k, tile, &[want]),
        (0.0, 0.0),
        "an exact result must read as zero units"
    );

    for mult in [0.25f64, 0.5, 1.0, 2.0, 4.0] {
        let got = (want as f64 + mult * bound) as f32;
        assert_eq!(
            got as f64 - want as f64,
            mult * bound,
            "test setup is not exact at mult={mult}"
        );
        let (mx, rms) = coop_accum_units(&x, &w, 1, 1, k, tile, &[got]);
        assert!(
            (mx - mult).abs() < 1e-6 && (rms - mult).abs() < 1e-6,
            "metric is miscalibrated: {mult} units of error read as max {mx} / rms {rms}"
        );
        assert_eq!(
            mx > 1.0,
            mult > 1.0,
            "the 1.0 ceiling must trip exactly when the error exceeds what {} accumulate steps \
             can round away (mult={mult}, units={mx})",
            k / tile
        );
    }
}

#[test]
fn coop_gemm_prefill_sweep() {
    let ctx = ctx();
    let g = require_coop(ctx);
    eprintln!("adapter: {}", ctx.summary());

    let g32 = match coop::select(ctx, coop::Operand::F32) {
        Ok(g32) => Some(g32),
        Err(why) => {
            eprintln!("  SKIP coopF32 rows: {why}");
            None
        }
    };
    let f32_arm = g32.is_some();
    let trace = std::env::var("NV_COOP_SWEEP_TRACE").ok().as_deref() == Some("1");
    eprintln!(
        "shape                 M    path       ms/dispatch   GB/s    TFLOP/s   ms/prompt-token   speedup"
    );
    let shapes: [(&str, u32, u32); 7] = [
        ("E4B gate_up 10240x2560", 10240, 2560),
        ("E4B down    2560x10240", 2560, 10240),
        ("31B square   5376x5376", 5376, 5376),
        ("MoE expert   2048x768 ", 2048, 768),
        ("Q38 gate_up 17408x5120", 17408, 5120),
        ("Q38 down    5120x17408", 5120, 17408),
        ("Q38 attn_q   6144x5120", 6144, 5120),
    ];
    for (label, n, k) in shapes {
        let w = shared_values((n * k) as usize, 0x1111_2222);
        let wf16 = replicas(ctx, "sweep-w-f16", &to_f16(&w), SLC_DEFEAT_BYTES);
        let wbf16 = replicas(ctx, "sweep-w-bf16", &pack_bf16(&w), SLC_DEFEAT_BYTES);
        let wf32 = replicas(ctx, "sweep-w-f32", &w, SLC_DEFEAT_BYTES);
        eprintln!(
            "  {label}: {} weight replicas x {:.1} MiB = {:.2} GiB cycled per dispatch",
            wf16.count,
            wf16.stride as f64 / 1048576.0,
            (wf16.count * wf16.stride) as f64 / 1073741824.0
        );
        drop(w);
        for m in [16u32, 32, 64, 128, 256] {
            let x = shared_values((m * k) as usize, 0x3333_4444);
            let passes = if (n as u64) * (k as u64) > 20_000_000 {
                8
            } else {
                30
            };
            let bytes = (n as f64) * (k as f64) * 2.0
                + (m as f64) * (k as f64) * 2.0
                + (m as f64) * (n as f64) * 4.0;
            let flops = 2.0 * m as f64 * n as f64 * k as f64;

            let t_mono = if m <= 64 {
                let mr = build_mrow(ctx, &wbf16, &x, m, n, k);
                bench(ctx, &mr, passes, 3)
            } else {
                f64::INFINITY
            };

            let slabs = m.div_ceil(SHIPPING_SLAB) as f64;
            let x16 = shared_values((SHIPPING_SLAB.min(m) * k) as usize, 0x3333_4444);
            let mr16 = build_mrow(ctx, &wbf16, &x16, SHIPPING_SLAB.min(m), n, k);
            let t_slab = bench(ctx, &mr16, passes, 3) * slabs;
            let tm_ = t_mono.min(t_slab);

            let mut best_coop = f64::INFINITY;
            let mut best_cfg = (0u32, 0u32, 0u32, 0u32);
            let mut best_f32 = f64::INFINITY;
            let mut best_f32_cfg = (0u32, 0u32, 0u32, 0u32);
            for acc in [4u32, 8, 16, 32] {
                let (tm, tn) = g.tiles(m, acc);
                if tm * tn > 32 || !g.acc_fits_a_register_file(tm, tn) {
                    continue;
                }
                for sg in [1u32, 2, 4, 8] {
                    for ku in [1u32, 2, 4] {
                        if !k.is_multiple_of(g.tile * ku) {
                            continue;
                        }
                        let cfg = (tm, tn, sg, ku);
                        if trace {
                            eprintln!(
                                "    trace {label} M={m} tm={tm} tn={tn} sg={sg} ku={ku} \
                                 ({} accumulator fragments of {}x{})",
                                tm * tn,
                                g.tile,
                                g.tile
                            );
                        }
                        let cr = build_coop_ab(ctx, g, &wf16, &x, m, n, k, cfg);
                        let t = bench(ctx, &cr, passes, 3);
                        if t < best_coop {
                            best_coop = t;
                            best_cfg = cfg;
                        }
                        if let Some(g32) = g32 {
                            let cf = build_coop_ab(ctx, g32, &wf32, &x, m, n, k, cfg);
                            let tf = bench(ctx, &cf, passes, 3);
                            if tf < best_f32 {
                                best_f32 = tf;
                                best_f32_cfg = cfg;
                            }
                        }
                    }
                }
            }
            eprintln!(
                "{label}  {m:<4} m-row mono         {:>9.3}   {:>6.1}  {:>7.2}   {:>9.4}",
                t_mono * 1e3,
                bytes / t_mono / 1e9,
                flops / t_mono / 1e12,
                t_mono * 1e3 / m as f64
            );
            eprintln!(
                "{label}  {m:<4} m-row slab16 x{slabs:<3.0} {:>9.3}   {:>6.1}  {:>7.2}   {:>9.4}",
                t_slab * 1e3,
                bytes * slabs / t_slab / 1e9,
                flops / t_slab / 1e12,
                t_slab * 1e3 / m as f64
            );
            eprintln!(
                "{label}  {m:<4} coop{} {}x{} sg{} ku{}  {:>9.3}   {:>6.1}  {:>7.2}   {:>9.4}     {:>5.2}x",
                g.tile,
                best_cfg.0,
                best_cfg.1,
                best_cfg.2,
                best_cfg.3,
                best_coop * 1e3,
                bytes / best_coop / 1e9,
                flops / best_coop / 1e12,
                best_coop * 1e3 / m as f64,
                tm_ / best_coop
            );
            if !f32_arm {
                continue;
            }
            let bytes32 = (n as f64) * (k as f64) * 4.0
                + (m as f64) * (k as f64) * 4.0
                + (m as f64) * (n as f64) * 4.0;
            eprintln!(
                "{label}  {m:<4} coopF32 {}x{} sg{} ku{} {:>9.3}   {:>6.1}  {:>7.2}   {:>9.4}     {:>5.2}x",
                best_f32_cfg.0,
                best_f32_cfg.1,
                best_f32_cfg.2,
                best_f32_cfg.3,
                best_f32 * 1e3,
                bytes32 / best_f32 / 1e9,
                flops / best_f32 / 1e12,
                best_f32 * 1e3 / m as f64,
                tm_ / best_f32
            );
        }
    }
}

#[test]
fn coop_gemm_agrees_with_the_m_row_path_at_prefill_scale() {
    let ctx = ctx();
    let g = require_coop(ctx);
    for (n, k, m) in [
        (10240u32, 2560u32, 32u32),
        (2560, 10240, 64),
        (5376, 5376, 16),
    ] {
        let w = shared_values((n * k) as usize, 0x7777_8888);
        let x = shared_values((m * k) as usize, 0x9999_aaaa);
        let wf16 = replicas(ctx, "scale-w-f16", &to_f16(&w), 0);
        let wbf16 = replicas(ctx, "scale-w-bf16", &pack_bf16(&w), 0);
        let (tm, tn) = g.tiles(m, coop::ACC_FRAGS);
        let cr = build_coop_ab(ctx, g, &wf16, &x, m, n, k, (tm, tn, 4, 1));
        let mr = build_mrow(ctx, &wbf16, &x, m, n, k);
        for r in [&cr, &mr] {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&r.pipeline);
                pass.set_bind_group(0, &r.group[0], &[]);
                pass.dispatch_workgroups(r.groups.0, r.groups.1, r.groups.2);
            }
            ctx.queue.submit([enc.finish()]);
        }
        ctx.poll_blocking().unwrap();
        let yc: Vec<f32> = dispatch::read_back(ctx, &cr.y, (m * n) as usize).unwrap();
        let ym: Vec<f32> = dispatch::read_back(ctx, &mr.y, (m * n) as usize).unwrap();
        let mut worst = 0f64;
        let mut scale = 0f64;
        let mut zero_rows = 0usize;
        for (a, b) in yc.iter().zip(ym.iter()) {
            worst = worst.max((*a as f64 - *b as f64).abs());
            scale = scale.max((*b as f64).abs());
            if *a == 0.0 {
                zero_rows += 1;
            }
        }
        let rel = worst / scale.max(1e-30);
        eprintln!(
            "  M={m:<3} N={n:<6} K={k:<6}  coop{} {tm}x{tn}: max|coop - m-row| {worst:.3e}  (max|y| {scale:.3e}, rel {rel:.3e}), coop zeros {zero_rows}/{}",
            g.tile,
            yc.len()
        );
        assert!(
            zero_rows * 100 < yc.len(),
            "coop output is mostly zero at N={n} K={k} M={m}: the kernel did not write"
        );
        assert!(
            rel < 2e-3,
            "coop and m-row disagree by {rel:.3e} relative at N={n} K={k} M={m}"
        );
    }
}

fn build_w4a16(
    ctx: &WgpuContext,
    wb: &Replicas,
    wsf: &Replicas,
    x: &[f32],
    m: u32,
    n: u32,
    k: u32,
    cfg: (u32, u32, u32, u32),
) -> Rig {
    use nv_kernels::wgpu_backend::kernels::gemm_coop_f16 as cf;
    let (tm, tn, sg, ku) = cfg;
    let src = cf::source_w4a16(tm, tn, sg, ku);
    let entry = cf::entry_w4a16(tm, tn, sg, ku);
    let pipeline = dispatch::compute_pipeline(ctx, "w4a16-coop", &src, &entry)
        .unwrap_or_else(|e| panic!("w4a16 pipeline tm={tm} tn={tn} sg={sg} ku={ku}: {e}"));
    let xbuf = dispatch::storage_from_slice(ctx, "w4a16-x", &to_f16(x));
    let y = dispatch::storage_zeroed(ctx, "w4a16-y", (m as u64) * (n as u64) * 4);
    let zero = dispatch::storage_from_slice(ctx, "w4a16-zero", &vec![0f32; 256]);
    let cols = 16 * tn * sg;
    let bn = n.div_ceil(cols);
    let bm = m.div_ceil(16 * tm);
    let groups = dispatch::workgroup_count_1d(ctx, (bm as u64) * (bn as u64), 1);
    let p = coop::CoopGemmParams {
        n_rows: n,
        k_elems: k,
        m_rows: m,
        blocks_n: bn,
        y_stride: n,
        groups_x: groups.0,
        pad0: 0,
        pad1: 0,
    };
    let pbuf = dispatch::uniform_from(ctx, "w4a16-p", &p);
    let group = (0..wb.count)
        .map(|i| {
            dispatch::bind_group_offsets(
                ctx,
                &pipeline,
                &[
                    (0, &wb.buf, i * wb.stride),
                    (1, &xbuf, 0),
                    (2, &y, 0),
                    (3, &pbuf, 0),
                    (4, &zero, 0),
                    (5, &wsf.buf, (i % wsf.count) * wsf.stride),
                ],
            )
        })
        .collect();
    Rig {
        pipeline,
        group,
        groups,
        y,
        tm,
        tn,
        keep: vec![xbuf, zero, pbuf],
    }
}

fn nvfp4_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0), *c.get(3).unwrap_or(&0)]))
        .collect()
}

#[test]
fn w4a16_coop_matches_the_f16_coop_path_on_dequantized_weights() {
    use nv_quant::nvfp4::Nvfp4Tensor;
    let ctx = ctx();
    let g = require_coop(ctx);
    assert!(g.tile == 16, "w4a16 stages 16-code blocks; the adapter must serve 16x16 fragments");
    for (m, n, k, cfg) in [
        (32u32, 128u32, 256u32, (2u32, 2u32, 2u32, 1u32)),
        (64, 256, 512, (4, 4, 2, 2)),
        (128, 512, 1024, (8, 2, 2, 4)),
    ] {
        let flat = shared_values((n * k) as usize, 0x9999_aaaa);
        let wrows: Vec<Vec<f32>> = flat.chunks(k as usize).map(|c| c.to_vec()).collect();
        let wq = Nvfp4Tensor::quantize_rows(&wrows);
        let wdeq2d = wq.dequantize();
        let wdeq: Vec<f32> = wdeq2d.into_iter().flatten().collect();
        let x = shared_values((m * k) as usize, 0xbbbb_cccc);

        let wb = replicas(ctx, "w4a16-par-w", &nvfp4_words(&wq.data), 1);
        let wsf = replicas(ctx, "w4a16-par-sf", &nvfp4_words(&wq.scales), 1);
        let rig4 = build_w4a16(ctx, &wb, &wsf, &x, m, n, k, cfg);
        let _ = bench(ctx, &rig4, 1, 1);
        let got4: Vec<f32> = dispatch::read_back(ctx, &rig4.y, (m * n) as usize)
            .unwrap()
            .iter()
            .map(|w| f32::from_bits(*w))
            .collect();

        let wf16 = replicas(ctx, "w4a16-par-wf16", &to_f16(&wdeq), 1);
        let (tm, tn, sg, _) = cfg;
        let rigf = build_coop_ab(ctx, g, &wf16, &x, m, n, k, (tm, tn, sg, 1));
        let _ = bench(ctx, &rigf, 1, 1);
        let gotf: Vec<f32> = dispatch::read_back(ctx, &rigf.y, (m * n) as usize)
            .unwrap()
            .iter()
            .map(|w| f32::from_bits(*w))
            .collect();

        let ndiff = got4
            .iter()
            .zip(gotf.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            ndiff,
            0,
            "M={m} N={n} K={k} cfg={cfg:?}: w4a16 in-kernel unpack must be bit-equal to the \
             f16 path on host-dequantized weights (same f32 product rounded to f16 either way); \
             {ndiff} of {} outputs differ",
            m * n
        );
    }
}

#[test]
fn w8a16_coop_matches_the_f16_coop_path_on_dequantized_fp8_weights() {
    use nv_kernels::wgpu_backend::kernels::gemm_coop_f16 as cf;
    let ctx = ctx();
    let g = require_coop(ctx);
    assert!(g.tile == 16);
    for (m, n, k, cfg) in [
        (32u32, 128u32, 256u32, (2u32, 2u32, 2u32, 1u32)),
        (64, 256, 512, (4, 4, 2, 2)),
        (128, 512, 1024, (8, 2, 2, 4)),
    ] {
        let flat = shared_values((n * k) as usize, 0xdddd_eeee);
        let wb16: Vec<half::bf16> = flat.iter().map(|v| half::bf16::from_f32(*v)).collect();
        let (bytes, scales) =
            nv_quant::fp8::quantize_e4m3_per_row(&wb16, n as usize, k as usize).unwrap();
        let wdeq =
            nv_quant::fp8::dequantize_e4m3_per_row(&bytes, n as usize, k as usize, &scales)
                .unwrap();
        let x = shared_values((m * k) as usize, 0xbbbb_cccc);

        let wb = replicas(ctx, "w8a16-par-w", &nvfp4_words(&bytes), 1);
        let wsf = replicas(ctx, "w8a16-par-sf", &scales, 1);
        let (tm, tn, sg, ku) = cfg;
        let src = cf::source_wq16(cf::WqFmt::Fp8RowscalePlain, tm, tn, sg, ku);
        let entry = cf::entry_wq16(cf::WqFmt::Fp8RowscalePlain, tm, tn, sg, ku);
        let pipeline = dispatch::compute_pipeline(ctx, "w8a16-coop", &src, &entry)
            .unwrap_or_else(|e| panic!("w8a16 pipeline {cfg:?}: {e}"));
        let xbuf = dispatch::storage_from_slice(ctx, "w8a16-x", &to_f16(&x));
        let y = dispatch::storage_zeroed(ctx, "w8a16-y", (m as u64) * (n as u64) * 4);
        let zero = dispatch::storage_from_slice(ctx, "w8a16-zero", &vec![0f32; 256]);
        let cols = 16 * tn * sg;
        let bn = n.div_ceil(cols);
        let bm = m.div_ceil(16 * tm);
        let groups = dispatch::workgroup_count_1d(ctx, (bm as u64) * (bn as u64), 1);
        let p = coop::CoopGemmParams {
            n_rows: n,
            k_elems: k,
            m_rows: m,
            blocks_n: bn,
            y_stride: n,
            groups_x: groups.0,
            pad0: 0,
            pad1: 0,
        };
        let pbuf = dispatch::uniform_from(ctx, "w8a16-p", &p);
        let group = vec![dispatch::bind_group_offsets(
            ctx,
            &pipeline,
            &[
                (0, &wb.buf, 0),
                (1, &xbuf, 0),
                (2, &y, 0),
                (3, &pbuf, 0),
                (4, &zero, 0),
                (5, &wsf.buf, 0),
            ],
        )];
        let rig8 = Rig {
            pipeline,
            group,
            groups,
            y,
            tm,
            tn,
            keep: vec![xbuf, zero, pbuf],
        };
        let _ = bench(ctx, &rig8, 1, 1);
        let got8: Vec<f32> = dispatch::read_back(ctx, &rig8.y, (m * n) as usize)
            .unwrap()
            .iter()
            .map(|w| f32::from_bits(*w))
            .collect();

        let wf16 = replicas(ctx, "w8a16-par-wf16", &to_f16(&wdeq), 1);
        let rigf = build_coop_ab(ctx, g, &wf16, &x, m, n, k, (tm, tn, sg, 1));
        let _ = bench(ctx, &rigf, 1, 1);
        let gotf: Vec<f32> = dispatch::read_back(ctx, &rigf.y, (m * n) as usize)
            .unwrap()
            .iter()
            .map(|w| f32::from_bits(*w))
            .collect();

        let ndiff = got8
            .iter()
            .zip(gotf.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            ndiff,
            0,
            "M={m} N={n} K={k} cfg={cfg:?}: w8a16 in-kernel shift-decode must be bit-equal to \
             the f16 path on host-dequantized fp8 weights; {ndiff} of {} outputs differ",
            m * n
        );
    }
}

fn bf16_encode_prelude(x: f32) -> u32 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let b = x.to_bits();
    let r = 0x7fffu32 + ((b >> 16) & 1);
    b.wrapping_add(r) >> 16
}

#[test]
fn w4a16_y16_epilogue_is_bit_equal_to_the_f32_store_plus_host_pack() {
    use nv_kernels::wgpu_backend::kernels::gemm_coop_f16 as cf;
    use nv_quant::nvfp4::Nvfp4Tensor;
    let ctx = ctx();
    let g = require_coop(ctx);
    assert!(g.tile == 16);
    let (m, n, k) = (64u32, 256u32, 512u32);
    let cfg = (4u32, 4u32, 2u32, 2u32);
    let flat = shared_values((n * k) as usize, 0x9999_aaaa);
    let wrows: Vec<Vec<f32>> = flat.chunks(k as usize).map(|c| c.to_vec()).collect();
    let wq = Nvfp4Tensor::quantize_rows(&wrows);
    let x = shared_values((m * k) as usize, 0xbbbb_cccc);
    let wb = replicas(ctx, "y16-w", &nvfp4_words(&wq.data), 1);
    let wsf = replicas(ctx, "y16-sf", &nvfp4_words(&wq.scales), 1);

    let rig32 = build_w4a16(ctx, &wb, &wsf, &x, m, n, k, cfg);
    let _ = bench(ctx, &rig32, 1, 1);
    let got32: Vec<f32> = dispatch::read_back(ctx, &rig32.y, (m * n) as usize)
        .unwrap()
        .iter()
        .map(|w| f32::from_bits(*w))
        .collect();

    let (tm, tn, sg, ku) = cfg;
    let src = cf::source_wq16_act_y16(cf::WqFmt::Nvfp4Block16, cf::WqAct::F16, tm, tn, sg, ku);
    let entry = cf::entry_wq16_act_y16(cf::WqFmt::Nvfp4Block16, cf::WqAct::F16, tm, tn, sg, ku);
    let pipeline = dispatch::compute_pipeline(ctx, "w4a16-y16", &src, &entry)
        .unwrap_or_else(|e| panic!("y16 pipeline: {e}"));
    let xbuf = dispatch::storage_from_slice(ctx, "y16-x", &to_f16(&x));
    let y = dispatch::storage_zeroed(ctx, "y16-y", (m as u64) * (n as u64) / 2 * 4);
    let zero = dispatch::storage_from_slice(ctx, "y16-zero", &vec![0f32; 256]);
    let cols = 16 * tn * sg;
    let bn = n.div_ceil(cols);
    let bm = m.div_ceil(16 * tm);
    let groups = dispatch::workgroup_count_1d(ctx, (bm as u64) * (bn as u64), 1);
    let alpha = 1.5f32;
    let p = coop::CoopGemmParams {
        n_rows: n,
        k_elems: k,
        m_rows: m,
        blocks_n: bn,
        y_stride: n,
        groups_x: groups.0,
        pad0: alpha.to_bits(),
        pad1: 0,
    };
    let pbuf = dispatch::uniform_from(ctx, "y16-p", &p);
    let bind = dispatch::bind_group_offsets(
        ctx,
        &pipeline,
        &[
            (0, &wb.buf, 0),
            (1, &xbuf, 0),
            (2, &y, 0),
            (3, &pbuf, 0),
            (4, &zero, 0),
            (5, &wsf.buf, 0),
        ],
    );
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }
    ctx.queue.submit([enc.finish()]);
    ctx.poll_blocking().unwrap();
    let got16: Vec<u32> = dispatch::read_back(ctx, &y, (m * n / 2) as usize).unwrap();

    let mut ndiff = 0usize;
    for i in 0..(m * n) as usize {
        let want = bf16_encode_prelude(got32[i] * alpha);
        let word = got16[i / 2];
        let got = if i % 2 == 0 { word & 0xffff } else { word >> 16 };
        if got != want {
            ndiff += 1;
        }
    }
    assert_eq!(
        ndiff,
        0,
        "the in-kernel bf16 epilogue must equal the f32 store followed by the prelude's \
         round-to-nearest-even bf16 encode; {ndiff} of {} differ",
        m * n
    );
}

#[test]
#[ignore]
fn w4a16_coop_rate_at_qwen_shapes() {
    use nv_quant::nvfp4::Nvfp4Tensor;
    let ctx = ctx();
    let g = require_coop(ctx);
    assert!(g.tile == 16);
    eprintln!("adapter: {}", ctx.summary());
    eprintln!("shape                 M    w4a16-coop  ms/dispatch  w-GB/s   TFLOP/s   ms/prompt-token");
    for (label, n, k) in [
        ("Q38 gate_up 17408x5120", 17408u32, 5120u32),
        ("Q38 down    5120x17408", 5120, 17408),
    ] {
        let flat = shared_values((n * k) as usize, 0x5555_6666);
        let wrows: Vec<Vec<f32>> = flat.chunks(k as usize).map(|c| c.to_vec()).collect();
        drop(flat);
        let wq = Nvfp4Tensor::quantize_rows(&wrows);
        drop(wrows);
        let wb = replicas(ctx, "w4a16-w", &nvfp4_words(&wq.data), SLC_DEFEAT_BYTES);
        let wsf = replicas(ctx, "w4a16-sf", &nvfp4_words(&wq.scales), SLC_DEFEAT_BYTES / 8);
        eprintln!(
            "  {label}: {} weight replicas x {:.1} MiB",
            wb.count,
            wb.stride as f64 / 1048576.0
        );
        let w_bytes = (n as f64) * (k as f64) / 2.0 + wq.scales.len() as f64;
        for m in [32u32, 64, 128, 256] {
            let x = shared_values((m * k) as usize, 0x7777_8888);
            let mut best = f64::INFINITY;
            let mut best_cfg = (0u32, 0u32, 0u32, 0u32);
            for cfg in [
                (2u32, 2u32, 2u32, 1u32),
                (2, 2, 2, 3),
                (4, 4, 2, 2),
                (8, 2, 2, 1),
                (8, 2, 2, 2),
                (8, 1, 1, 4),
                (4, 2, 4, 2),
                (2, 4, 4, 1),
            ] {
                let (tm, tn, sg, ku) = cfg;
                if m > 16 * tm * 8 || tm * tn > 32 || !(k as usize).is_multiple_of(16 * ku as usize) {
                    continue;
                }
                if 16 * tn * sg * 16 * ku > nv_kernels::wgpu_backend::kernels::gemm_coop_f16::W4A16_STAGE_BUDGET_F16_IS_HALF_THE_48K_WORKGROUP_LIMIT {
                    continue;
                }
                let rig = build_w4a16(ctx, &wb, &wsf, &x, m, n, k, cfg);
                let passes = 8;
                let t = bench(ctx, &rig, passes, 3);
                if t < best {
                    best = t;
                    best_cfg = cfg;
                }
            }
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            eprintln!(
                "{label}  {m:<4} w4a16 {}x{} sg{} ku{}  {:>9.3}   {:>6.1}  {:>7.2}   {:>9.4}",
                best_cfg.0,
                best_cfg.1,
                best_cfg.2,
                best_cfg.3,
                best * 1e3,
                w_bytes / best / 1e9,
                flops / best / 1e12,
                best * 1e3 / m as f64
            );
        }
    }
}

enum ActPayload {
    F16(Vec<u16>),
    Fp8 { words: Vec<u32>, scales: Vec<f32> },
    Nvfp4 { words: Vec<u32>, scales: Vec<u32> },
}

fn quantize_act(act: coop::WqAct, x: &[f32], m: u32, k: u32) -> (ActPayload, Vec<f32>) {
    match act {
        coop::WqAct::F16 => (ActPayload::F16(to_f16(x)), x.to_vec()),
        coop::WqAct::Fp8Rowscale => {
            let xb: Vec<half::bf16> = x.iter().map(|v| half::bf16::from_f32(*v)).collect();
            let (bytes, scales) =
                nv_quant::fp8::quantize_e4m3_per_row(&xb, m as usize, k as usize).unwrap();
            let deq =
                nv_quant::fp8::dequantize_e4m3_per_row(&bytes, m as usize, k as usize, &scales)
                    .unwrap();
            (
                ActPayload::Fp8 {
                    words: nvfp4_words(&bytes),
                    scales,
                },
                deq,
            )
        }
        coop::WqAct::Nvfp4Block16 => {
            use nv_quant::nvfp4::Nvfp4Tensor;
            let rows: Vec<Vec<f32>> = x.chunks(k as usize).map(|c| c.to_vec()).collect();
            let q = Nvfp4Tensor::quantize_rows(&rows);
            let deq: Vec<f32> = q.dequantize().into_iter().flatten().collect();
            (
                ActPayload::Nvfp4 {
                    words: nvfp4_words(&q.data),
                    scales: nvfp4_words(&q.scales),
                },
                deq,
            )
        }
    }
}

fn build_wq16_act(
    ctx: &WgpuContext,
    fmt: coop::WqFmt,
    act: coop::WqAct,
    wb: &Replicas,
    wsf: &Replicas,
    payload: &ActPayload,
    m: u32,
    n: u32,
    k: u32,
    cfg: (u32, u32, u32, u32),
) -> Rig {
    let (tm, tn, sg, ku) = cfg;
    let src = coop::source_wq16_act(fmt, act, tm, tn, sg, ku);
    let entry = coop::entry_wq16_act(fmt, act, tm, tn, sg, ku);
    let pipeline = dispatch::compute_pipeline(ctx, "wq16-act-coop", &src, &entry)
        .unwrap_or_else(|e| panic!("{fmt:?}x{act:?} pipeline tm={tm} tn={tn} sg={sg} ku={ku}: {e}"));
    let (xbuf, xsfbuf) = match payload {
        ActPayload::F16(h) => (dispatch::storage_from_slice(ctx, "wq16-x", h), None),
        ActPayload::Fp8 { words, scales } => (
            dispatch::storage_from_slice(ctx, "wq16-x", words),
            Some(dispatch::storage_from_slice(ctx, "wq16-xsf", scales)),
        ),
        ActPayload::Nvfp4 { words, scales } => (
            dispatch::storage_from_slice(ctx, "wq16-x", words),
            Some(dispatch::storage_from_slice(ctx, "wq16-xsf", scales)),
        ),
    };
    let y = dispatch::storage_zeroed(ctx, "wq16-y", (m as u64) * (n as u64) * 4);
    let zero = dispatch::storage_from_slice(ctx, "wq16-zero", &vec![0f32; 256]);
    let cols = 16 * tn * sg;
    let bn = n.div_ceil(cols);
    let bm = m.div_ceil(16 * tm);
    let groups = dispatch::workgroup_count_1d(ctx, (bm as u64) * (bn as u64), 1);
    let p = coop::CoopGemmParams {
        n_rows: n,
        k_elems: k,
        m_rows: m,
        blocks_n: bn,
        y_stride: n,
        groups_x: groups.0,
        pad0: 0,
        pad1: 0,
    };
    let pbuf = dispatch::uniform_from(ctx, "wq16-p", &p);
    let group = (0..wb.count)
        .map(|i| {
            let mut entries = vec![
                (0u32, &wb.buf, i * wb.stride),
                (1, &xbuf, 0),
                (2, &y, 0),
                (3, &pbuf, 0),
                (4, &zero, 0),
                (5, &wsf.buf, (i % wsf.count) * wsf.stride),
            ];
            if let Some(sf) = xsfbuf.as_ref() {
                entries.push((6, sf, 0));
            }
            dispatch::bind_group_offsets(ctx, &pipeline, &entries)
        })
        .collect();
    let mut keep = vec![xbuf, zero, pbuf];
    if let Some(sf) = xsfbuf {
        keep.push(sf);
    }
    Rig {
        pipeline,
        group,
        groups,
        y,
        tm,
        tn,
        keep,
    }
}

fn quantize_weights(fmt: coop::WqFmt, w: &[f32], n: u32, k: u32, target: u64, ctx: &WgpuContext)
    -> (Replicas, Replicas, Vec<f32>, f64) {
    match fmt {
        coop::WqFmt::Nvfp4Block16 => {
            use nv_quant::nvfp4::Nvfp4Tensor;
            let rows: Vec<Vec<f32>> = w.chunks(k as usize).map(|c| c.to_vec()).collect();
            let q = Nvfp4Tensor::quantize_rows(&rows);
            let deq: Vec<f32> = q.dequantize().into_iter().flatten().collect();
            let bytes = (n as f64) * (k as f64) / 2.0 + q.scales.len() as f64;
            (
                replicas(ctx, "wq16-w", &nvfp4_words(&q.data), target),
                replicas(ctx, "wq16-wsf", &nvfp4_words(&q.scales), target / 8),
                deq,
                bytes,
            )
        }
        coop::WqFmt::Fp8RowscalePlain => {
            let wb16: Vec<half::bf16> = w.iter().map(|v| half::bf16::from_f32(*v)).collect();
            let (bytes8, scales) =
                nv_quant::fp8::quantize_e4m3_per_row(&wb16, n as usize, k as usize).unwrap();
            let deq =
                nv_quant::fp8::dequantize_e4m3_per_row(&bytes8, n as usize, k as usize, &scales)
                    .unwrap();
            let bytes = (n as f64) * (k as f64) + (n as f64) * 4.0;
            (
                replicas(ctx, "wq16-w", &nvfp4_words(&bytes8), target),
                replicas(ctx, "wq16-wsf", &scales, target / 4),
                deq,
                bytes,
            )
        }
    }
}

fn rel_rms_f32(got: &[f32], want: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, o) in got.iter().zip(want.iter()) {
        let d = *g as f64 - *o as f64;
        num += d * d;
        den += (*o as f64) * (*o as f64);
    }
    (num / den.max(1e-30)).sqrt()
}

const WQ16_ACT_ARMS: [(coop::WqFmt, coop::WqAct); 3] = [
    (coop::WqFmt::Nvfp4Block16, coop::WqAct::Fp8Rowscale),
    (coop::WqFmt::Nvfp4Block16, coop::WqAct::Nvfp4Block16),
    (coop::WqFmt::Fp8RowscalePlain, coop::WqAct::Fp8Rowscale),
];

#[test]
fn wq16_act_arms_are_bit_equal_to_the_f16_act_path_on_dequantized_activations() {
    let ctx = ctx();
    let g = require_coop(ctx);
    assert!(g.tile == 16);
    for (fmt, act) in WQ16_ACT_ARMS {
        for (m, n, k, cfg) in [
            (32u32, 128u32, 256u32, (2u32, 2u32, 2u32, 1u32)),
            (64, 256, 512, (4, 4, 2, 2)),
            (128, 512, 1024, (8, 2, 2, 4)),
        ] {
            let w = shared_values((n * k) as usize, 0x9999_aaaa);
            let (wb, wsf, _wdeq, _) = quantize_weights(fmt, &w, n, k, 1, ctx);
            let x = shared_values((m * k) as usize, 0xbbbb_cccc);
            let (payload, xdeq) = quantize_act(act, &x, m, k);
            eprintln!(
                "  {fmt:?}x{act:?} M={m} N={n} K={k}: act quantization rel-rms {:.3e}",
                rel_rms_f32(&xdeq, &x)
            );

            let rigq = build_wq16_act(ctx, fmt, act, &wb, &wsf, &payload, m, n, k, cfg);
            let _ = bench(ctx, &rigq, 1, 1);
            let gotq: Vec<f32> = dispatch::read_back(ctx, &rigq.y, (m * n) as usize).unwrap();

            let ref_payload = ActPayload::F16(to_f16(&xdeq));
            let rigf =
                build_wq16_act(ctx, fmt, coop::WqAct::F16, &wb, &wsf, &ref_payload, m, n, k, cfg);
            let _ = bench(ctx, &rigf, 1, 1);
            let gotf: Vec<f32> = dispatch::read_back(ctx, &rigf.y, (m * n) as usize).unwrap();

            let ndiff = gotq
                .iter()
                .zip(gotf.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                ndiff,
                0,
                "{fmt:?}x{act:?} M={m} N={n} K={k} cfg={cfg:?}: the in-kernel A stage must be \
                 bit-equal to the F16-act path fed host-dequantized activations (same f32 \
                 product rounded to the same f16 tile either way); {ndiff} of {} outputs differ",
                m * n
            );
        }
    }
}

#[test]
#[ignore]
fn wq16_act_rate_at_qwen_shapes() {
    let ctx = ctx();
    let g = require_coop(ctx);
    assert!(g.tile == 16);
    eprintln!("adapter: {}", ctx.summary());
    eprintln!(
        "shape                 M    arm    cfg              ms/dispatch  w-GB/s  A-GB/disp  A-GB/s  TFLOP/s  ms/prompt-token  vs-w4a16"
    );
    let arms: [(coop::WqFmt, coop::WqAct); 4] = [
        (coop::WqFmt::Nvfp4Block16, coop::WqAct::F16),
        (coop::WqFmt::Nvfp4Block16, coop::WqAct::Fp8Rowscale),
        (coop::WqFmt::Nvfp4Block16, coop::WqAct::Nvfp4Block16),
        (coop::WqFmt::Fp8RowscalePlain, coop::WqAct::Fp8Rowscale),
    ];
    for (label, n, k) in [
        ("Q38 gate_up 17408x5120", 17408u32, 5120u32),
        ("Q38 down    5120x17408", 5120, 17408),
    ] {
        let w = shared_values((n * k) as usize, 0x5555_6666);
        let mut packs = Vec::new();
        for fmt in [coop::WqFmt::Nvfp4Block16, coop::WqFmt::Fp8RowscalePlain] {
            let (wb, wsf, _deq, bytes) = quantize_weights(fmt, &w, n, k, SLC_DEFEAT_BYTES, ctx);
            eprintln!(
                "  {label} {fmt:?}: {} weight replicas x {:.1} MiB",
                wb.count,
                wb.stride as f64 / 1048576.0
            );
            packs.push((fmt, wb, wsf, bytes));
        }
        drop(w);
        for m in [64u32, 128, 256, 512] {
            let x = shared_values((m * k) as usize, 0x7777_8888);
            let mut baseline = f64::INFINITY;
            for (fmt, act) in arms {
                let (_, wb, wsf, w_bytes) = packs.iter().find(|p| p.0 == fmt).unwrap();
                let (payload, _) = quantize_act(act, &x, m, k);
                let mut best = f64::INFINITY;
                let mut best_cfg = (0u32, 0u32, 0u32, 0u32);
                for cfg in [
                    (2u32, 2u32, 2u32, 1u32),
                    (2, 2, 2, 3),
                    (4, 4, 2, 2),
                    (8, 2, 2, 1),
                    (8, 2, 2, 2),
                    (8, 1, 1, 4),
                    (4, 2, 4, 2),
                    (2, 4, 4, 1),
                ] {
                    let (tm, tn, sg, ku) = cfg;
                    if m > 16 * tm * 8
                        || tm * tn > 32
                        || !(k as usize).is_multiple_of(16 * ku as usize)
                    {
                        continue;
                    }
                    if coop::wq16_act_stage_elems(act, tm, tn, sg, ku)
                        > coop::W4A16_STAGE_BUDGET_F16_IS_HALF_THE_48K_WORKGROUP_LIMIT
                    {
                        continue;
                    }
                    let rig = build_wq16_act(ctx, fmt, act, wb, wsf, &payload, m, n, k, cfg);
                    let t = bench(ctx, &rig, 8, 3);
                    if t < best {
                        best = t;
                        best_cfg = cfg;
                    }
                }
                if fmt == coop::WqFmt::Nvfp4Block16 && act == coop::WqAct::F16 {
                    baseline = best;
                }
                let bn = n.div_ceil(16 * best_cfg.1 * best_cfg.2) as f64;
                let a_bytes = act.a_bytes(m, k) * bn;
                let flops = 2.0 * m as f64 * n as f64 * k as f64;
                eprintln!(
                    "{label}  {m:<4} {}{} {}x{} sg{} ku{}  {:>9.3}   {:>6.1}   {:>7.3}  {:>6.1}  {:>7.2}   {:>9.4}   {:>5.2}x",
                    match fmt {
                        coop::WqFmt::Nvfp4Block16 => "w4",
                        coop::WqFmt::Fp8RowscalePlain => "w8",
                    },
                    match act {
                        coop::WqAct::F16 => "a16",
                        coop::WqAct::Fp8Rowscale => "a8 ",
                        coop::WqAct::Nvfp4Block16 => "a4 ",
                    },
                    best_cfg.0,
                    best_cfg.1,
                    best_cfg.2,
                    best_cfg.3,
                    best * 1e3,
                    w_bytes / best / 1e9,
                    a_bytes / 1e9,
                    a_bytes / best / 1e9,
                    flops / best / 1e12,
                    best * 1e3 / m as f64,
                    baseline / best
                );
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct W4a4Params {
    alpha: f32,
    m: u32,
    n: u32,
    k: u32,
    row_words: u32,
    k_tiles: u32,
    tiles_n: u32,
    groups_x: u32,
}

fn build_w4a4_coop(
    ctx: &WgpuContext,
    wb: &Replicas,
    wsf: &Replicas,
    a_packed: &[u8],
    a_sf: &[u8],
    m: u32,
    n: u32,
    k: u32,
) -> Rig {
    use nv_kernels::wgpu_backend::kernels::gemm_nvfp4 as g4;
    let src = g4::coop_source();
    let pipeline = dispatch::compute_pipeline(ctx, "w4a4-coop", &src, g4::COOP_ENTRY)
        .unwrap_or_else(|e| panic!("w4a4 coop pipeline: {e}"));
    let aw: Vec<u32> = a_packed
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], *c.get(2).unwrap_or(&0), *c.get(3).unwrap_or(&0)]))
        .collect();
    let asfw: Vec<u32> = a_sf
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], *c.get(2).unwrap_or(&0), *c.get(3).unwrap_or(&0)]))
        .collect();
    let abuf = dispatch::storage_from_slice(ctx, "w4a4-a", &aw);
    let asfbuf = dispatch::storage_from_slice(ctx, "w4a4-a-sf", &asfw);
    let y = dispatch::storage_zeroed(ctx, "w4a4-d", (m as u64) * (n as u64) * 4);
    let blocks_n = (n as usize).div_ceil(g4::COOP_BLOCK_N) as u32;
    let groups = dispatch::workgroup_count_1d(
        ctx,
        ((m as usize).div_ceil(g4::COOP_BLOCK_M) as u64) * (blocks_n as u64),
        1,
    );
    let k_blocks = (k as usize) / 16;
    let p = W4a4Params {
        alpha: 1.0,
        m,
        n,
        k,
        row_words: k / 8,
        k_tiles: g4::k_tiles(k_blocks) as u32,
        tiles_n: blocks_n,
        groups_x: groups.0,
    };
    let pbuf = dispatch::uniform_from(ctx, "w4a4-p", &p);
    let group = (0..wb.count)
        .map(|i| {
            dispatch::bind_group_offsets(
                ctx,
                &pipeline,
                &[
                    (0, &abuf, 0),
                    (1, &asfbuf, 0),
                    (2, &wb.buf, i * wb.stride),
                    (3, &wsf.buf, (i % wsf.count) * wsf.stride),
                    (4, &pbuf, 0),
                    (5, &y, 0),
                ],
            )
        })
        .collect();
    Rig {
        pipeline,
        group,
        groups,
        y,
        tm: 0,
        tn: 0,
        keep: vec![abuf, asfbuf, pbuf],
    }
}

#[test]
#[ignore]
fn w4a4_coop_rate_at_qwen_shapes() {
    use nv_kernels::wgpu_backend::kernels::gemm_nvfp4 as g4;
    use nv_quant::nvfp4::{swizzle_scales, Nvfp4Tensor};
    let ctx = ctx();
    let _ = require_coop(ctx);
    assert!(
        g4::resolve_path(ctx, g4::GemmPath::CoopMat).is_ok(),
        "this rate probe refuses to fall back silently: the adapter must serve the coop path"
    );
    eprintln!("adapter: {}", ctx.summary());
    eprintln!("shape                 M    w4a4-coop  ms/dispatch  w-GB/s   TFLOP/s   ms/prompt-token");
    for (label, n, k) in [
        ("Q38 gate_up 17408x5120", 17408u32, 5120u32),
        ("Q38 down    5120x17408", 5120, 17408),
        ("E4B gate_up 10240x2560", 10240, 2560),
    ] {
        let flat = shared_values((n * k) as usize, 0x5555_6666);
        let wrows: Vec<Vec<f32>> = flat.chunks(k as usize).map(|c| c.to_vec()).collect();
        drop(flat);
        let wq = Nvfp4Tensor::quantize_rows(&wrows);
        drop(wrows);
        let wsf_bytes = swizzle_scales(&wq.scales, n as usize, (k / 16) as usize);
        let wb_words: Vec<u32> = wq
            .data
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], *c.get(2).unwrap_or(&0), *c.get(3).unwrap_or(&0)]))
            .collect();
        let wsf_words: Vec<u32> = wsf_bytes
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], *c.get(2).unwrap_or(&0), *c.get(3).unwrap_or(&0)]))
            .collect();
        let wb = replicas(ctx, "w4a4-w", &wb_words, SLC_DEFEAT_BYTES);
        let wsf = replicas(ctx, "w4a4-w-sf", &wsf_words, SLC_DEFEAT_BYTES / 8);
        eprintln!(
            "  {label}: {} weight replicas x {:.1} MiB",
            wb.count,
            wb.stride as f64 / 1048576.0
        );
        let w_bytes = (n as f64) * (k as f64) / 2.0 + wsf_bytes.len() as f64;
        for m in [16u32, 32, 64, 128, 256] {
            let aflat = shared_values((m * k) as usize, 0x7777_8888);
            let arows: Vec<Vec<f32>> = aflat.chunks(k as usize).map(|c| c.to_vec()).collect();
            let aq = Nvfp4Tensor::quantize_rows(&arows);
            let a_sf = swizzle_scales(&aq.scales, m as usize, (k / 16) as usize);
            let rig = build_w4a4_coop(ctx, &wb, &wsf, &aq.data, &a_sf, m, n, k);
            let passes = if (n as u64) * (k as u64) > 20_000_000 { 8 } else { 30 };
            let t = bench(ctx, &rig, passes, 3);
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            eprintln!(
                "{label}  {m:<4} w4a4-coop  {:>9.3}   {:>6.1}  {:>7.2}   {:>9.4}",
                t * 1e3,
                w_bytes / t / 1e9,
                flops / t / 1e12,
                t * 1e3 / m as f64
            );
        }
    }
}

fn require_spirv(ctx: &WgpuContext) {
    if let Some(why) = spv::preflight(ctx) {
        panic!("spirv mulmm route closed on this adapter: {why}");
    }
}

fn spv_fmt(fmt: coop::WqFmt) -> spv::SpvWq {
    match fmt {
        coop::WqFmt::Nvfp4Block16 => spv::SpvWq::Nvfp4Block16,
        coop::WqFmt::Fp8RowscalePlain => spv::SpvWq::Fp8RowscalePlain,
    }
}

fn build_spirv_mulmm(
    ctx: &WgpuContext,
    blob: &'static spv::SpvBlob,
    wb: &Replicas,
    wsf: &Replicas,
    x: &[f32],
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
) -> Rig {
    spv::check_shape(m, n, k, blob.blocking)
        .unwrap_or_else(|e| panic!("{}: M={m} N={n} K={k}: {e}", blob.name));
    let g = spv::pipeline(ctx, blob).unwrap_or_else(|e| panic!("{}: {e}", blob.name));
    let xbuf = dispatch::storage_from_slice(ctx, "spv-x", &to_f16(x));
    let y_words = match blob.out {
        spv::SpvOut::F32 => (m as u64) * (n as u64),
        spv::SpvOut::Bf16Alpha => (m as u64) * (n as u64) / 2,
    };
    let y = dispatch::storage_zeroed(ctx, "spv-y", y_words * 4);
    let p = spv::SpirvGemmParams {
        n_rows: n,
        k_elems: k,
        m_rows: m,
        y_stride: n,
        alpha,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let pbuf = dispatch::uniform_from(ctx, "spv-p", &p);
    let group = (0..wb.count)
        .map(|i| {
            dispatch::bind_group_offsets(
                ctx,
                &g.pipeline,
                &[
                    (0, &wb.buf, i * wb.stride),
                    (1, &xbuf, 0),
                    (2, &y, 0),
                    (3, &pbuf, 0),
                    (4, &wsf.buf, (i % wsf.count) * wsf.stride),
                ],
            )
        })
        .collect();
    Rig {
        pipeline: g.pipeline.clone(),
        group,
        groups: g.grid(m, n),
        y,
        tm: 0,
        tn: 0,
        keep: vec![xbuf, pbuf],
    }
}

const SPIRV_PARITY_SHAPES: [(u32, u32, u32); 3] =
    [(32, 128, 256), (64, 256, 512), (128, 512, 1024)];

#[test]
fn spirv_mulmm_arms_match_the_wgsl_coop_arm_bit_for_bit_and_stay_inside_the_oracle_units() {
    let ctx = ctx();
    require_spirv(ctx);
    let g = require_coop(ctx);
    assert!(g.tile == 16);
    for fmt in [coop::WqFmt::Nvfp4Block16, coop::WqFmt::Fp8RowscalePlain] {
        for blob in spv::blobs()
            .iter()
            .filter(|b| b.wq == spv_fmt(fmt) && b.out == spv::SpvOut::F32)
        {
            for (m, n, k) in SPIRV_PARITY_SHAPES {
                if !k.is_multiple_of(blob.blocking.bk) {
                    continue;
                }
                let w = shared_values((n * k) as usize, 0x9999_aaaa);
                let (wb, wsf, wdeq, _) = quantize_weights(fmt, &w, n, k, 1, ctx);
                let x = shared_values((m * k) as usize, 0xbbbb_cccc);

                let rig_s = build_spirv_mulmm(ctx, blob, &wb, &wsf, &x, m, n, k, 1.0);
                let _ = bench(ctx, &rig_s, 1, 1);
                let got_s: Vec<f32> = dispatch::read_back(ctx, &rig_s.y, (m * n) as usize).unwrap();

                let payload = ActPayload::F16(to_f16(&x));
                let rig_w =
                    build_wq16_act(ctx, fmt, coop::WqAct::F16, &wb, &wsf, &payload, m, n, k, (2, 2, 2, 1));
                let _ = bench(ctx, &rig_w, 1, 1);
                let got_w: Vec<f32> = dispatch::read_back(ctx, &rig_w.y, (m * n) as usize).unwrap();

                let ndiff = got_s
                    .iter()
                    .zip(got_w.iter())
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    ndiff,
                    0,
                    "{} M={m} N={n} K={k}: the spirv mul_mm arm and the WGSL coop arm stage \
                     identical f16 operands and accumulate k-sequentially in the same 16-deep \
                     fragment steps on the same tensor cores, so any bit difference is a staging \
                     or indexing defect, not reduction order; {ndiff} of {} differ",
                    blob.name,
                    m * n
                );

                let xh: Vec<f32> = x.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
                let wh: Vec<f32> = wdeq.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
                let (units_max, units_rms) = coop_accum_units(&xh, &wh, m, n, k, 16, &got_s);
                eprintln!(
                    "  {} {fmt:?} M={m} N={n} K={k}: bit-equal to WGSL arm; oracle units max \
                     {units_max:.4} rms {units_rms:.4} of the 1.0 truncating-accumulate ceiling",
                    blob.name
                );
                assert!(
                    units_max <= 1.0,
                    "{} M={m} N={n} K={k}: {units_max:.4} f32 accumulator-rounding units off the \
                     f64 oracle; the ceiling is 1.0 by the triangle inequality over the {} \
                     sequential accumulate steps, each losing at most 2^-23 of its largest \
                     operand (running sum in, fragment dot magnitude, running sum out) -- the \
                     operand scale, not the post-cancellation running sum, is the loss unit on a \
                     truncating WMMA accumulator (wgpu_wmma_accum_model.rs pins this per step), \
                     so exceeding it means the kernel lost more than its own accumulation order \
                     can explain",
                    blob.name,
                    k / 16
                );
            }
        }
    }
}

#[test]
fn spirv_mulmm_is_exact_when_every_partial_sum_is_integer_representable() {
    let ctx = ctx();
    require_spirv(ctx);
    let class = common::wmma_accum_class(ctx);
    let (m, n, k) = (64u32, 128u32, 512u32);
    let e2m1: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let x: Vec<f32> = (0..m * k)
        .map(|i| (((i * 5 + 1) % 9) as f32) - 4.0)
        .collect();

    let codes: Vec<u8> = (0..n * k)
        .map(|i| ((i as u64 * 7 + 3) % 16) as u8)
        .collect();
    let mut packed = vec![0u8; (n * k / 2) as usize];
    for (i, c) in codes.iter().enumerate() {
        packed[i / 2] |= c << (4 * (i % 2));
    }
    let scale_one_ue4m3 = 0x38u8;
    let scales = vec![scale_one_ue4m3; (n * k / 16) as usize];
    let wb4 = replicas(ctx, "spv-int-w4", &nvfp4_words(&packed), 1);
    let wsf4 = replicas(ctx, "spv-int-sf4", &nvfp4_words(&scales), 1);
    let blob4 = spv::default_blob(spv::SpvWq::Nvfp4Block16, spv::SpvOut::F32);
    let rig4 = build_spirv_mulmm(ctx, blob4, &wb4, &wsf4, &x, m, n, k, 1.0);
    let _ = bench(ctx, &rig4, 1, 1);
    let got4: Vec<f32> = dispatch::read_back(ctx, &rig4.y, (m * n) as usize).unwrap();
    let mut ndiff4 = 0usize;
    for mi in 0..m as usize {
        for ni in 0..n as usize {
            let mut acc = 0f64;
            for j in 0..k as usize {
                acc += x[mi * k as usize + j] as f64
                    * e2m1[codes[ni * k as usize + j] as usize] as f64;
            }
            if got4[mi * n as usize + ni] != acc as f32 {
                ndiff4 += 1;
            }
        }
    }
    match class {
        common::WmmaAccumClass::IeeeExact => assert_eq!(
            ndiff4, 0,
            "w4a16 spirv mulmm must be exact on unit-scale integer-and-half operands: every \
             partial sum is a multiple of 0.5 below 2^23, so on an IEEE-exact WMMA accumulator \
             all {} accumulate steps round nothing away in any order",
            k / 16
        ),
        common::WmmaAccumClass::TruncatingDot2 => {
            let wdeq4: Vec<f32> = codes.iter().map(|c| e2m1[*c as usize]).collect();
            let (u4_max, u4_rms) = coop_accum_units(&x, &wdeq4, m, n, k, 16, &got4);
            eprintln!(
                "  w4a16 on TruncatingDot2 wmma: {ndiff4} of {} not exact, oracle units max \
                 {u4_max:.4} rms {u4_rms:.4}",
                m * n
            );
            assert!(
                u4_max <= 1.0,
                "w4a16 spirv mulmm on a truncating WMMA accumulator: exactness on \
                 integer-representable partials is a property of IEEE accumulation and this \
                 adapter class provably does not have it (wgpu_wmma_accum_model.rs pins the \
                 minimal dot2 pair (3,-1) -> 2-2^-23), but each step still loses at most 2^-23 \
                 of its largest operand; {u4_max:.4} units exceeds that ceiling, which a staging \
                 or indexing defect would"
            );
        }
    }

    let bytes8: Vec<u8> = (0..n * k)
        .map(|i| {
            let mag = 0x50u8 | ((i % 8) as u8);
            if (i / 8) % 2 == 1 {
                mag | 0x80
            } else {
                mag
            }
        })
        .collect();
    let sf8 = vec![0.25f32; n as usize];
    let wb8 = replicas(ctx, "spv-int-w8", &nvfp4_words(&bytes8), 1);
    let wsf8 = replicas(ctx, "spv-int-sf8", &sf8, 1);
    let blob8 = spv::default_blob(spv::SpvWq::Fp8RowscalePlain, spv::SpvOut::F32);
    let rig8 = build_spirv_mulmm(ctx, blob8, &wb8, &wsf8, &x, m, n, k, 1.0);
    let _ = bench(ctx, &rig8, 1, 1);
    let got8: Vec<f32> = dispatch::read_back(ctx, &rig8.y, (m * n) as usize).unwrap();
    let decode8 = |b: u8| -> f64 {
        let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
        let e = ((b >> 3) & 15) as i32;
        let mm = (b & 7) as f64;
        s * (8.0 + mm) * (2f64).powi(e - 10)
    };
    let mut ndiff8 = 0usize;
    for mi in 0..m as usize {
        for ni in 0..n as usize {
            let mut acc = 0f64;
            for j in 0..k as usize {
                acc += x[mi * k as usize + j] as f64
                    * (decode8(bytes8[ni * k as usize + j]) * 0.25);
            }
            if got8[mi * n as usize + ni] != acc as f32 {
                ndiff8 += 1;
            }
        }
    }
    match class {
        common::WmmaAccumClass::IeeeExact => assert_eq!(
            ndiff8, 0,
            "w8a16 spirv mulmm must be exact on quarter-step operands: every partial sum is a \
             multiple of 0.25 below 2^23, so on an IEEE-exact WMMA accumulator no accumulate \
             step rounds in any order"
        ),
        common::WmmaAccumClass::TruncatingDot2 => {
            let wdeq8: Vec<f32> = bytes8.iter().map(|b| (decode8(*b) * 0.25) as f32).collect();
            let (u8_max, u8_rms) = coop_accum_units(&x, &wdeq8, m, n, k, 16, &got8);
            eprintln!(
                "  w8a16 on TruncatingDot2 wmma: {ndiff8} of {} not exact, oracle units max \
                 {u8_max:.4} rms {u8_rms:.4}",
                m * n
            );
            assert!(
                u8_max <= 1.0,
                "w8a16 spirv mulmm on a truncating WMMA accumulator: quarter-step exactness is \
                 an IEEE-accumulation property this adapter class provably lacks \
                 (wgpu_wmma_accum_model.rs), but each step still loses at most 2^-23 of its \
                 largest operand; {u8_max:.4} units exceeds that ceiling, which a staging or \
                 indexing defect would"
            );
        }
    }
}

#[test]
fn spirv_mulmm_y16_epilogue_is_bit_equal_to_the_f32_blob_plus_host_pack() {
    let ctx = ctx();
    require_spirv(ctx);
    let (m, n, k) = (64u32, 256u32, 512u32);
    let alpha = 1.5f32;
    for fmt in [coop::WqFmt::Nvfp4Block16, coop::WqFmt::Fp8RowscalePlain] {
        let w = shared_values((n * k) as usize, 0x9999_aaaa);
        let (wb, wsf, _wdeq, _) = quantize_weights(fmt, &w, n, k, 1, ctx);
        let x = shared_values((m * k) as usize, 0xbbbb_cccc);

        let blob32 = spv::default_blob(spv_fmt(fmt), spv::SpvOut::F32);
        let rig32 = build_spirv_mulmm(ctx, blob32, &wb, &wsf, &x, m, n, k, 1.0);
        let _ = bench(ctx, &rig32, 1, 1);
        let got32: Vec<f32> = dispatch::read_back(ctx, &rig32.y, (m * n) as usize).unwrap();

        let blob16 = spv::default_blob(spv_fmt(fmt), spv::SpvOut::Bf16Alpha);
        let rig16 = build_spirv_mulmm(ctx, blob16, &wb, &wsf, &x, m, n, k, alpha);
        let _ = bench(ctx, &rig16, 1, 1);
        let got16: Vec<u32> = dispatch::read_back(ctx, &rig16.y, (m * n / 2) as usize).unwrap();

        let mut ndiff = 0usize;
        for i in 0..(m * n) as usize {
            let want = bf16_encode_prelude(got32[i] * alpha);
            let word = got16[i / 2];
            let got = if i % 2 == 0 { word & 0xffff } else { word >> 16 };
            if got != want {
                ndiff += 1;
            }
        }
        assert_eq!(
            ndiff,
            0,
            "{fmt:?}: the spirv bf16 epilogue must equal the f32 blob followed by the prelude's \
             round-to-nearest-even bf16 encode with the alpha multiplier; {ndiff} of {} differ",
            m * n
        );
    }
}

#[test]
#[ignore]
fn spirv_mulmm_rate_at_qwen_shapes() {
    let ctx = ctx();
    require_spirv(ctx);
    eprintln!("adapter: {}", ctx.summary());
    eprintln!(
        "shape                 M    blob                                          ms/dispatch  w-GB/s   TFLOP/s   ms/prompt-token"
    );
    eprintln!(
        "  wgsl w4a16 baselines (w4a16_coop_rate_at_qwen_shapes, SLC-defeat, dispatch-only) \
         and the llama.cpp KHR-class end-to-end rate live in perf/runs.jsonl"
    );
    for (label, n, k) in [
        ("Q38 gate_up 17408x5120", 17408u32, 5120u32),
        ("Q38 down    5120x17408", 5120, 17408),
    ] {
        let w = shared_values((n * k) as usize, 0x5555_6666);
        for fmt in [coop::WqFmt::Nvfp4Block16, coop::WqFmt::Fp8RowscalePlain] {
            let (wb, wsf, _deq, w_bytes) = quantize_weights(fmt, &w, n, k, SLC_DEFEAT_BYTES, ctx);
            eprintln!(
                "  {label} {fmt:?}: {} weight replicas x {:.1} MiB",
                wb.count,
                wb.stride as f64 / 1048576.0
            );
            for m in [64u32, 128, 256, 512] {
                let x = shared_values((m * k) as usize, 0x7777_8888);
                for blob in spv::blobs()
                    .iter()
                    .filter(|b| b.wq == spv_fmt(fmt) && b.out == spv::SpvOut::F32)
                {
                    if !k.is_multiple_of(blob.blocking.bk) {
                        continue;
                    }
                    let rig = build_spirv_mulmm(ctx, blob, &wb, &wsf, &x, m, n, k, 1.0);
                    let t = bench(ctx, &rig, 8, 3);
                    let flops = 2.0 * m as f64 * n as f64 * k as f64;
                    eprintln!(
                        "{label}  {m:<4} {:<45} {:>9.3}   {:>6.1}  {:>7.2}   {:>9.4}",
                        blob.name,
                        t * 1e3,
                        w_bytes / t / 1e9,
                        flops / t / 1e12,
                        t * 1e3 / m as f64
                    );
                }
            }
        }
        drop(w);
    }
}
