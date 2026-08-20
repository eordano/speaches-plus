#![cfg(feature = "wgpu")]

use std::time::Instant;

use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::qualify;
mod common;
use common::ctx_or_panic as ctx;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    iters: u32,
    stride: u32,
    pad0: u32,
    pad1: u32,
}

fn to_msl(source: &str) -> Result<String, String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| format!("wgsl parse: {e:?}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("validate: {e:?}"))?;
    let opts = naga::back::msl::Options {
        lang_version: (3, 0),
        ..Default::default()
    };
    naga::back::msl::write_string(
        &module,
        &info,
        &opts,
        &naga::back::msl::PipelineOptions::default(),
    )
    .map(|(s, _)| s)
    .map_err(|e| format!("msl-out: {e:?}"))
}

fn compile_probe_src(tile: u32, ab: &str, c: &str, space: &str) -> String {
    let (decl, load_a, load_b, load_c, store) = match space {
        "workgroup" => (
            format!(
                "var<workgroup> wa: array<{ab}, {n}>;\nvar<workgroup> wb: array<{ab}, {n}>;\nvar<workgroup> wc: array<{c}, {n}>;\n",
                n = tile * tile
            ),
            "coopLoadT<CA>(&wa[0], TILE)".to_string(),
            "coopLoadT<CB>(&wb[0], TILE)".to_string(),
            "coopLoadT<CC>(&wc[0], TILE)".to_string(),
            "coopStoreT(acc, &wc[0], TILE)".to_string(),
        ),
        _ => (
            String::new(),
            "coopLoadT<CA>(&sa[0], TILE)".to_string(),
            "coopLoadT<CB>(&sb[0], TILE)".to_string(),
            "coopLoadT<CC>(&sc[0], TILE)".to_string(),
            "coopStoreT(acc, &sc[0], TILE)".to_string(),
        ),
    };
    format!(
        "enable f16;\nenable wgpu_cooperative_matrix;\n\
         alias CA = coop_mat{tile}x{tile}<{ab}, A>;\n\
         alias CB = coop_mat{tile}x{tile}<{ab}, B>;\n\
         alias CC = coop_mat{tile}x{tile}<{c}, C>;\n\
         const TILE: u32 = {tile}u;\n\
         @group(0) @binding(0) var<storage, read> sa: array<{ab}>;\n\
         @group(0) @binding(1) var<storage, read> sb: array<{ab}>;\n\
         @group(0) @binding(2) var<storage, read_write> sc: array<{c}>;\n\
         {decl}\
         @compute @workgroup_size(32)\n\
         fn probe() {{\n\
         \x20   let a = {load_a};\n\
         \x20   let b = {load_b};\n\
         \x20   let acc = coopMultiplyAdd(a, b, {load_c});\n\
         \x20   {store};\n\
         }}\n"
    )
}

#[test]
fn coop_probe_1_adapter_configs() {
    let ctx = ctx();
    eprintln!("adapter: {}", ctx.summary());
    let props = ctx.adapter.cooperative_matrix_properties();
    eprintln!(
        "EXPERIMENTAL_COOPERATIVE_MATRIX granted on device: {}",
        ctx.caps.cooperative_matrix
    );
    eprintln!(
        "adapter-reported cooperative_matrix_properties: {} config(s)",
        props.len()
    );
    for (i, p) in props.iter().enumerate() {
        eprintln!(
            "  cfg[{i}]  M={:<3} N={:<3} K={:<3}  AB={:?}  CR={:?}  sat_accum={}",
            p.m_size, p.n_size, p.k_size, p.ab_type, p.cr_type, p.saturating_accumulation
        );
    }
    eprintln!(
        "in-tree gate coop_gemm_tile() = {:?}  reason = {:?}",
        ctx.caps.coop_gemm_tile(),
        ctx.caps.coop_gemm_reason()
    );
    eprintln!("runtime subgroup width probe = {:?}", ctx.subgroup_width());
    assert!(!props.is_empty(), "adapter reported zero coop_mat configs");
}

#[test]
fn coop_probe_2_compile_matrix() {
    let owned = WgpuContext::new().expect("no wgpu adapter");
    let ctx = &owned;
    eprintln!("adapter: {}", ctx.summary());

    let seen: std::sync::Arc<std::sync::Mutex<Vec<String>>> = Default::default();
    let sink = seen.clone();
    ctx.device
        .on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| {
            sink.lock().unwrap().push(e.to_string());
        }));
    let mut ok = Vec::new();
    let mut fail = Vec::new();
    let mut skipped = Vec::new();
    let unsafe_sweep = qualify::coop_unsafe_sweep_enabled();

    for tile in [8u32, 16] {
        for (ab, c) in [("f16", "f16"), ("f16", "f32"), ("f32", "f32")] {
            for space in ["storage", "workgroup"] {
                let src = compile_probe_src(tile, ab, c, space);
                let label = format!("coop{tile}x{tile}-{ab}{c}-{space}");
                let req = qualify::CoopRequest::square(
                    tile,
                    qualify::CoopScalar::from_wgsl(ab).expect("ab scalar"),
                    qualify::CoopScalar::from_wgsl(c).expect("cr scalar"),
                );
                seen.lock().unwrap().clear();
                let naga_msl = to_msl(&src);
                let decision = qualify::coop_decide(&req, &ctx.caps.coop_configs, unsafe_sweep);
                if let qualify::CoopDecision::Skip(why) = &decision {
                    eprintln!(
                        "  {label:<34} SKIP    unadvertised: {why}  (naga-msl {}) -- set {}=1 to \
                         compile it anyway",
                        if naga_msl.is_ok() { "ok" } else { "fail" },
                        qualify::COOP_UNSAFE_SWEEP_ENV
                    );
                    skipped.push(label);
                    continue;
                }
                if let qualify::CoopDecision::CompileUnadvertised(why) = &decision {
                    eprintln!("  {label:<34} UNSAFE  {why}");
                }
                let pipe = dispatch::compute_pipeline(ctx, &label, &src, "probe");
                let uncaptured = seen.lock().unwrap().join(" ;; ");
                match (&naga_msl, &pipe, uncaptured.is_empty()) {
                    (Ok(m), Ok(_), true) => {
                        let sg = m
                            .lines()
                            .filter(|l| l.contains("simdgroup_"))
                            .take(2)
                            .map(str::trim)
                            .collect::<Vec<_>>()
                            .join(" | ");
                        eprintln!("  {label:<34} OK      msl: {sg}");
                        ok.push(label);
                    }
                    (Err(e), _, _) => {
                        eprintln!("  {label:<34} NAGA-MSL FAIL: {}", first_line(e));
                        fail.push(label);
                    }
                    (Ok(_), Err(e), _) => {
                        eprintln!(
                            "  {label:<34} PIPELINE FAIL: {}",
                            first_line(&e.to_string())
                        );
                        fail.push(label);
                    }
                    (Ok(_), Ok(_), false) => {
                        eprintln!("  {label:<34} METAL FAIL: {}", first_line(&uncaptured));
                        fail.push(label);
                    }
                }
            }
        }
    }
    eprintln!("compiled: {ok:?}");
    eprintln!("rejected: {fail:?}");
    eprintln!("skipped (unadvertised): {skipped:?}");
    assert!(
        !ok.is_empty(),
        "no cooperative-matrix configuration compiled on this adapter; {} of {} rows were skipped \
         as unadvertised, so the adapter advertises nothing this probe knows how to emit \
         (advertised: {:?})",
        skipped.len(),
        skipped.len() + ok.len() + fail.len(),
        ctx.caps.coop_configs
    );
}

fn first_line(s: &str) -> String {
    let one: String = s.replace('\n', " ⏎ ");
    if one.len() > 400 {
        format!("{}…", &one[..400])
    } else {
        one
    }
}

#[test]
fn coop_probe_3_emitted_msl() {
    let ctx = ctx();
    let _ = ctx;
    let src = compile_probe_src(8, "f16", "f32", "storage");
    match to_msl(&src) {
        Ok(msl) => {
            eprintln!("--- emitted MSL for coop8x8 f16xf16->f32, storage space ---");
            for l in msl.lines() {
                if l.contains("simdgroup") || l.contains("Naga") || l.contains("kernel void") {
                    eprintln!("{l}");
                }
            }
            assert!(
                msl.contains("simdgroup_multiply_accumulate"),
                "MSL has no simdgroup_multiply_accumulate; coop_mat did not lower to matrix hw"
            );
            assert!(msl.contains("simdgroup_load"), "MSL has no simdgroup_load");
        }
        Err(e) => panic!("naga could not emit MSL for coop8x8 f16->f32: {e}"),
    }
}

const THRU_SLOTS: u32 = 16;
const THRU_ACC: u32 = 8;
const THRU_OUT_SLOTS: u32 = 256;

fn thru_src(tile: u32) -> String {
    use std::fmt::Write as _;
    let te = tile * tile;
    let mut b = String::new();
    writeln!(b, "enable f16;\nenable wgpu_cooperative_matrix;\n").unwrap();
    writeln!(b, "alias CA = coop_mat{tile}x{tile}<f16, A>;").unwrap();
    writeln!(b, "alias CB = coop_mat{tile}x{tile}<f16, B>;").unwrap();
    writeln!(b, "alias CC = coop_mat{tile}x{tile}<f32, C>;\n").unwrap();
    b.push_str("struct P { iters: u32, stride: u32, pad0: u32, pad1: u32 };\n\n");
    b.push_str("@group(0) @binding(0) var<storage, read> sh: array<f16>;\n");
    b.push_str("@group(0) @binding(1) var<storage, read> sz: array<f32>;\n");
    b.push_str("@group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n");
    b.push_str("@group(0) @binding(3) var<uniform> pp: P;\n\n");
    b.push_str("@compute @workgroup_size(256)\n");
    b.push_str("fn coop_thru(@builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>) {\n");
    b.push_str("    let sg = lidx / 32u;\n");
    writeln!(b, "    let slot = (wid.x * 8u + sg) & {}u;", THRU_SLOTS - 1).unwrap();
    writeln!(b, "    let a = coopLoadT<CA>(&sh[slot * {te}u], {tile}u);").unwrap();
    writeln!(
        b,
        "    let b = coopLoadT<CB>(&sh[{}u + slot * {te}u], {tile}u);",
        THRU_SLOTS * te
    )
    .unwrap();
    for c in 0..THRU_ACC {
        writeln!(
            b,
            "    var c{c} = coopLoadT<CC>(&sz[{}u], {tile}u);",
            c * te
        )
        .unwrap();
    }
    b.push_str("    for (var i = 0u; i < pp.iters; i = i + 1u) {\n");
    for c in 0..THRU_ACC {
        writeln!(b, "        c{c} = coopMultiplyAdd(a, b, c{c});").unwrap();
    }
    b.push_str("    }\n");
    writeln!(
        b,
        "    let o = ((wid.x * 8u + sg) & {}u) * {}u;",
        THRU_OUT_SLOTS - 1,
        THRU_ACC * te
    )
    .unwrap();
    for c in 0..THRU_ACC {
        writeln!(b, "    coopStoreT(c{c}, &dst[o + {}u], {tile}u);", c * te).unwrap();
    }
    b.push_str("}\n");
    b
}

const SCALAR_SRC: &str = r#"
struct P { iters: u32, stride: u32, pad0: u32, pad1: u32 };

@group(0) @binding(0) var<storage, read> sh: array<f32>;
@group(0) @binding(1) var<storage, read> sz: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> pp: P;

@compute @workgroup_size(256)
fn scalar_thru(@builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let gid = wid.x * 256u + lidx;
    let a = sh[gid & 1023u];
    let b = sh[(gid + 7u) & 1023u];
    var c0 = sz[0]; var c1 = sz[1]; var c2 = sz[2]; var c3 = sz[3];
    var c4 = sz[4]; var c5 = sz[5]; var c6 = sz[6]; var c7 = sz[7];
    var c8 = sz[8]; var c9 = sz[9]; var ca = sz[10]; var cb = sz[11];
    var cc = sz[12]; var cd = sz[13]; var ce = sz[14]; var cf = sz[15];
    for (var i = 0u; i < pp.iters; i = i + 1u) {
        c0 = fma(a, b, c0); c1 = fma(a, b, c1); c2 = fma(a, b, c2); c3 = fma(a, b, c3);
        c4 = fma(a, b, c4); c5 = fma(a, b, c5); c6 = fma(a, b, c6); c7 = fma(a, b, c7);
        c8 = fma(a, b, c8); c9 = fma(a, b, c9); ca = fma(a, b, ca); cb = fma(a, b, cb);
        cc = fma(a, b, cc); cd = fma(a, b, cd); ce = fma(a, b, ce); cf = fma(a, b, cf);
    }
    dst[gid] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + ca + cb + cc + cd + ce + cf;
}
"#;

fn bench_groups(
    ctx: &WgpuContext,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    groups: u32,
    passes: usize,
    reps: usize,
) -> f64 {
    let submit = |n: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            for _ in 0..n {
                pass.dispatch_workgroups(groups, 1, 1);
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
fn coop_probe_4_mma_throughput() {
    let ctx = ctx();
    eprintln!("adapter: {}", ctx.summary());
    let groups = 2048u32;
    let sg_per_group = 8u32;
    let acc_per_sg = THRU_ACC;

    let tiles: Vec<u32> = ctx
        .caps
        .coop_configs
        .iter()
        .filter(|c| c.ab_f16() && c.cr_f32() && c.m == c.n && c.n == c.k)
        .map(|c| c.m)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    assert!(
        !tiles.is_empty(),
        "adapter reports {} cooperative-matrix configs but no square f16xf16->f32 one, so there \
         is no MMA shape this probe may compile and the throughput arm would measure nothing: {:?}",
        ctx.caps.coop_configs.len(),
        ctx.caps.coop_configs
    );

    let max_te = tiles.iter().map(|t| t * t).max().expect("a tile");
    let mut hbits = vec![0u16; (2 * THRU_SLOTS * max_te) as usize];
    for (i, v) in hbits.iter_mut().enumerate() {
        *v = half::f16::from_f32(0.5 + (i % 13) as f32 * 0.0125).to_bits();
    }
    let sh = dispatch::storage_from_slice(ctx, "coop-thru-h", &hbits);
    let shf = dispatch::storage_from_slice(ctx, "coop-thru-f", &vec![0.37f32; 2048]);
    let sz = dispatch::storage_from_slice(
        ctx,
        "coop-thru-z",
        &vec![0.0f32; (THRU_ACC * max_te) as usize],
    );
    let dst_elems = (THRU_OUT_SLOTS * THRU_ACC * max_te).max(groups * 256);
    let dst = dispatch::storage_zeroed(ctx, "coop-thru-dst", (dst_elems * 4) as u64);

    let scalar = dispatch::compute_pipeline(ctx, "scalar-thru", SCALAR_SRC, "scalar_thru")
        .expect("scalar throughput pipeline");

    for tile in tiles {
        let name = format!("coop {tile}x{tile}x{tile} f16->f32");
        let src = thru_src(tile);
        let coop = dispatch::compute_pipeline(ctx, "coop-thru", &src, "coop_thru")
            .unwrap_or_else(|e| panic!("{name} throughput pipeline: {e}"));
        let (iters_lo, iters_hi) = (256u32, 512u32);
        let mut t = [0f64; 2];
        for (slot, iters) in [iters_lo, iters_hi].into_iter().enumerate() {
            let p = dispatch::uniform_from(
                ctx,
                "coop-thru-p",
                &Params {
                    iters,
                    stride: tile,
                    pad0: 0,
                    pad1: 0,
                },
            );
            let bg = dispatch::bind_group(ctx, &coop, &[(0, &sh), (1, &sz), (2, &dst), (3, &p)]);
            t[slot] = bench_groups(ctx, &coop, &bg, groups, 20, 5);
        }
        let d_iters = (iters_hi - iters_lo) as f64;
        let d_t = t[1] - t[0];
        let mmas = (groups * sg_per_group * acc_per_sg) as f64 * d_iters;
        let flops = mmas * 2.0 * (tile as f64).powi(3);
        eprintln!(
            "  {name}: iters {iters_lo} -> {:.3} ms, iters {iters_hi} -> {:.3} ms, slope {:.3} ms",
            t[0] * 1e3,
            t[1] * 1e3,
            d_t * 1e3
        );
        eprintln!(
            "  {name}: marginal {:.2} TFLOP/s  ({:.0} MMA {tile}x{tile}x{tile} per marginal step)",
            flops / d_t / 1e12,
            mmas
        );
        assert!(
            d_t > 0.2 * t[0],
            "{name}: doubling the trip count did not double the time ({:.3} -> {:.3} ms); the loop was folded",
            t[0] * 1e3,
            t[1] * 1e3
        );
    }

    {
        let mut t = [0f64; 2];
        for (slot, iters) in [256u32, 512].into_iter().enumerate() {
            let p = dispatch::uniform_from(
                ctx,
                "scalar-thru-p",
                &Params {
                    iters,
                    stride: 8,
                    pad0: 0,
                    pad1: 0,
                },
            );
            let bg = dispatch::bind_group(ctx, &scalar, &[(0, &shf), (1, &sz), (2, &dst), (3, &p)]);
            t[slot] = bench_groups(ctx, &scalar, &bg, groups, 20, 5);
        }
        let d_t = t[1] - t[0];
        let flops = (groups * 256) as f64 * 16.0 * 256.0 * 2.0;
        eprintln!(
            "  scalar f32 fma x16: iters 256 -> {:.3} ms, 512 -> {:.3} ms, slope {:.3} ms",
            t[0] * 1e3,
            t[1] * 1e3,
            d_t * 1e3
        );
        eprintln!(
            "  scalar f32 fma x16: marginal {:.2} TFLOP/s",
            flops / d_t / 1e12
        );
        assert!(
            d_t > 0.2 * t[0],
            "scalar: loop folded ({:.3} -> {:.3} ms)",
            t[0] * 1e3,
            t[1] * 1e3
        );
    }
    eprintln!(
        "  every number above was measured on `{}` ({}); the 27.2 TFLOP/s FP32 FMA figure this \
         line used to carry is a spec-sheet number for a different (Apple silicon) machine and is \
         not a reference for this one",
        ctx.caps.adapter_name, ctx.caps.driver
    );
}

#[test]
fn coop_probe_5_mma_matches_cpu_oracle() {
    let ctx = ctx();
    let tiles: Vec<u32> = ctx
        .caps
        .coop_configs
        .iter()
        .filter(|c| c.ab_f16() && c.cr_f32() && c.m == c.n && c.n == c.k)
        .map(|c| c.m)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    assert!(
        !tiles.is_empty(),
        "adapter reports {} cooperative-matrix configs but no square f16xf16->f32 one, so the \
         only arithmetic check in this file would have verified nothing: {:?}",
        ctx.caps.coop_configs.len(),
        ctx.caps.coop_configs
    );
    eprintln!("  checking MMA arithmetic for square f16->f32 tiles {tiles:?}");
    for tile in tiles {
        check_mma_against_oracle(ctx, tile);
    }
}

fn check_mma_against_oracle(ctx: &WgpuContext, tile: u32) {
    let t = tile as usize;
    let n = t * t;
    let src = format!(
        "enable f16;\nenable wgpu_cooperative_matrix;\n\
         alias CA = coop_mat{tile}x{tile}<f16, A>;\n\
         alias CB = coop_mat{tile}x{tile}<f16, B>;\n\
         alias CC = coop_mat{tile}x{tile}<f32, C>;\n\
         const TILE: u32 = {tile}u;\n\
         @group(0) @binding(0) var<storage, read> ma: array<f16>;\n\
         @group(0) @binding(1) var<storage, read> mb: array<f16>;\n\
         @group(0) @binding(2) var<storage, read_write> md: array<f32>;\n\
         @compute @workgroup_size(32)\n\
         fn mma_once() {{\n\
         let a = coopLoadT<CA>(&ma[0], TILE);\n\
         let b = coopLoadT<CB>(&mb[0], TILE);\n\
         let c = coopLoadT<CC>(&md[{n}], TILE);\n\
         let d = coopMultiplyAdd(a, b, c);\n\
         coopStoreT(d, &md[0], TILE);\n\
         }}\n"
    );
    let mut a = vec![0f32; n];
    let mut b = vec![0f32; n];
    for i in 0..n {
        a[i] = ((i % 7) as f32 - 3.0) * 0.25;

        let (row, col) = (i / t, i % t);
        b[i] = ((row % 5) as f32 - 2.0) * 0.5 + ((col % 3) as f32 - 1.0) * 0.25;
    }
    let ah: Vec<u16> = a
        .iter()
        .map(|v| half::f16::from_f32(*v).to_bits())
        .collect();
    let bh: Vec<u16> = b
        .iter()
        .map(|v| half::f16::from_f32(*v).to_bits())
        .collect();
    let abuf = dispatch::storage_from_slice(ctx, "mma-a", &ah);
    let bbuf = dispatch::storage_from_slice(ctx, "mma-b", &bh);
    let dbuf = dispatch::storage_from_slice(ctx, "mma-d", &vec![0f32; 2 * n]);
    let label = format!("mma-once-{tile}");
    let pipe = dispatch::compute_pipeline(ctx, &label, &src, "mma_once").expect("mma pipeline");
    let bg = dispatch::bind_group(ctx, &pipe, &[(0, &abuf), (1, &bbuf), (2, &dbuf)]);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&pipe);
        p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(1, 1, 1);
    }
    ctx.queue.submit([enc.finish()]);
    ctx.poll_blocking().unwrap();
    let got: Vec<f32> = dispatch::read_back(ctx, &dbuf, n).unwrap();

    let mut ab = vec![0f32; n];
    let mut abt = vec![0f32; n];
    for i in 0..t {
        for j in 0..t {
            let mut s1 = 0f32;
            let mut s2 = 0f32;
            for k in 0..t {
                s1 += a[i * t + k] * b[k * t + j];
                s2 += a[i * t + k] * b[j * t + k];
            }
            ab[i * t + j] = s1;
            abt[i * t + j] = s2;
        }
    }
    let err = |o: &[f32]| -> f32 {
        got.iter()
            .zip(o.iter())
            .map(|(g, r)| (g - r).abs())
            .fold(0f32, f32::max)
    };
    let layout_gap = ab
        .iter()
        .zip(abt.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        layout_gap > 1e-3,
        "{tile}x{tile}: the two candidate oracles are identical (max|A*B - A*B^T| = \
         {layout_gap:.3e}), so this arm cannot tell a correct MMA from a transposed one. Fix the \
         input pattern, not the assertion."
    );
    let (e_ab, e_abt) = (err(&ab), err(&abt));
    eprintln!("  {tile}x{tile}: max|A*B - gpu| = {e_ab:.3e}, max|A*B^T - gpu| = {e_abt:.3e}");

    assert!(
        got.iter().any(|v| *v != 0.0),
        "{tile}x{tile} MMA wrote all zeros: the pipeline was created but the dispatch computed \
         nothing. Check stderr for a driver-side compile failure (this adapter prints `NVVM \
         compilation failed` and wgpu does not surface it). A config that probe 2 lists as OK can \
         still fail here -- probe 2 only creates the pipeline."
    );
    assert!(
        e_ab < 1e-4 || e_abt < 1e-4,
        "{tile}x{tile} MMA matched neither A*B ({e_ab:.3e}) nor A*B^T ({e_abt:.3e}); \
         gpu[0..4]={:?} A*B[0..4]={:?} A*B^T[0..4]={:?}",
        &got[..4.min(n)],
        &ab[..4.min(n)],
        &abt[..4.min(n)]
    );
}
