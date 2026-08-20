#![cfg(feature = "wgpu")]

mod common;
use common::widen_u16;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::gemv_w4a16_m1_proto as proto;
use nv_kernels::wgpu_backend::{compose, dispatch};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

include!("support/w4a16_host_oracle.rs");

const CANDIDATE_SUBDIR: &str = "wgsl/candidates";
const SHIPPING_SHADER_SUBPATH: &str = "wgsl/gemv_w4a16_m1_proto.wgsl";
const RUNNER_SUBPATH: &str = "tools/kernel-forge.sh";

const FORGE_JUDGES_CANDIDATES_AT_VARIANT_0_GEOMETRY_WARPS8_SPLIT1: u32 = 0;
const FORGE_ENTRY_SUFFIX_MATCHES_THE_PROTO_WARP8_ENTRY: &str = "_w8";
const PLANTED_BUG_STEM_MARKER_MEANS_THE_GATE_MUST_REJECT: &str = "planted_bug";
const CANDIDATES_ENV: &str = "NV_KERNEL_FORGE_CANDIDATES";
const FAILURE_LOG_ENV: &str = "NV_KERNEL_FORGE_FAILURE_LOG";

const GATE_SHAPES_ARE_MULTI_GROUP_SO_SCALE_INDEXING_IS_OBSERVABLE: [(usize, usize); 3] =
    [(67, 320), (96, 2560), (33, 10240)];
const BENCH_SHAPES_ARE_THE_E4B_GATE_UP_AND_DOWN_PROJECTIONS: [(&str, usize, usize); 2] =
    [("gate_up", 20480, 2560), ("down", 2560, 10240)];
const BENCH_WARMUP_DISPATCHES: usize = 8;
const BENCH_TIMED_DISPATCHES: usize = 200;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ForgeParams {
    n_rows: u32,
    kv: u32,
    w_row_words: u32,
    split: u32,
    rows_per_group: u32,
    max_v: u32,
    groups_x: u32,
    reserved: u32,
}

struct Candidate {
    stem: String,
    entry: String,
    source: String,
    must_be_rejected: bool,
}

enum Verdict {
    Pass(f32),
    Reject(String),
}

fn manifest(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(sub)
}

fn failure_log_path() -> PathBuf {
    std::env::var(FAILURE_LOG_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("kernel-forge-failures.tsv"))
}

fn recycle_prior_failures() {
    let p = failure_log_path();
    match std::fs::read_to_string(&p) {
        Ok(prior) if !prior.trim().is_empty() => {
            eprintln!(
                "[forge] {} prior failure line(s) recycled from {} into this attempt:",
                prior.lines().count(),
                p.display()
            );
            for line in prior.lines() {
                eprintln!("[forge]   {line}");
            }
        }
        _ => eprintln!("[forge] no prior failures recorded at {}", p.display()),
    }
}

fn record_failure(stem: &str, stage: &str, detail: &str) {
    let p = failure_log_path();
    let one_line = detail.replace('\n', " | ");
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&p)
        .unwrap_or_else(|e| panic!("forge cannot append to the failure log {}: {e}", p.display()));
    writeln!(f, "{stem}\t{stage}\t{one_line}")
        .unwrap_or_else(|e| panic!("forge cannot write the failure log {}: {e}", p.display()));
    eprintln!("[forge] {stem}: {stage} FAILED, recycled to {}", p.display());
}

fn selected() -> Option<Vec<String>> {
    let raw = std::env::var(CANDIDATES_ENV).ok()?;
    let picked: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if picked.is_empty() {
        return None;
    }
    Some(picked)
}

fn candidates() -> Vec<Candidate> {
    let dir = manifest(CANDIDATE_SUBDIR);
    let rd = std::fs::read_dir(&dir).unwrap_or_else(|e| {
        panic!(
            "the forge candidate directory {} is missing: {e}. A generate-and-gate cycle with \
             no candidates on disk gates nothing.",
            dir.display()
        )
    });
    let pick = selected();
    let mut out: Vec<Candidate> = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("wgsl") {
            continue;
        }
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        if let Some(list) = &pick {
            if !list.iter().any(|s| *s == stem) {
                continue;
            }
        }
        let source = std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("cannot read candidate {}: {e}", p.display()));
        out.push(Candidate {
            entry: format!("{stem}{FORGE_ENTRY_SUFFIX_MATCHES_THE_PROTO_WARP8_ENTRY}"),
            must_be_rejected: stem.contains(PLANTED_BUG_STEM_MARKER_MEANS_THE_GATE_MUST_REJECT),
            stem,
            source,
        });
    }
    out.sort_by(|a, b| a.stem.cmp(&b.stem));
    assert!(
        !out.is_empty(),
        "no candidate matched {CANDIDATES_ENV}; the gate would have reported success over an \
         empty set"
    );
    out
}

fn compile_host_side(source: &str, entry: &str) -> Result<(), String> {
    let full = compose(source);
    let module = naga::front::wgsl::parse_str(&full).map_err(|e| e.emit_to_string(&full))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("{e:?}"))?;
    if !module
        .entry_points
        .iter()
        .any(|ep| ep.name == entry && ep.workgroup_size == [256, 1, 1])
    {
        return Err(format!(
            "no @compute @workgroup_size(256) entry named {entry}; the forge dispatches every \
             candidate at the proto warps8/split1 geometry"
        ));
    }
    Ok(())
}

fn pack_u16(src: &[u16]) -> Vec<u32> {
    (0..src.len() / 2)
        .map(|i| src[2 * i] as u32 | ((src[2 * i + 1] as u32) << 16))
        .collect()
}

fn geometry(ctx: &WgpuContext, n: usize, k: usize) -> (ForgeParams, (u32, u32, u32)) {
    let v = proto::variant_for(FORGE_JUDGES_CANDIDATES_AT_VARIANT_0_GEOMETRY_WARPS8_SPLIT1)
        .expect("variant 0 is the shipping proto geometry");
    let kv = (k / 32) as u32;
    let rows_per_group = v.warps / v.split;
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, rows_per_group);
    (
        ForgeParams {
            n_rows: n as u32,
            kv,
            w_row_words: (k / 8) as u32,
            split: v.split,
            rows_per_group,
            max_v: proto::max_v_for(kv, v.split).expect("shape is inside the proto v-step ladder"),
            groups_x: groups.0,
            reserved: 0,
        },
        groups,
    )
}

fn run_shader(
    ctx: &WgpuContext,
    label: &str,
    source: &str,
    entry: &str,
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
) -> Result<Vec<u16>, String> {
    let (params, groups) = geometry(ctx, n, k);
    let pb = dispatch::storage_from_slice(ctx, "forge-packed", packed);
    let sb = dispatch::storage_from_slice(ctx, "forge-scale", &widen_u16(scales));
    let xb = dispatch::storage_from_slice(ctx, "forge-x", &pack_u16(x));
    let yb = dispatch::storage_from_slice(ctx, "forge-y", &vec![0x7fc0u32; n]);
    let ub = dispatch::uniform_from(ctx, "forge-params", &params);
    dispatch::run(
        ctx,
        label,
        &compose(source),
        entry,
        &[(0, &pb), (1, &sb), (2, &xb), (3, &yb), (4, &ub)],
        groups,
    )
    .map_err(|e| e.to_string())?;
    let words: Vec<u32> = dispatch::read_back(ctx, &yb, n).map_err(|e| e.to_string())?;
    Ok(words.iter().map(|w| (*w & 0xffff) as u16).collect())
}

fn judge(
    got: &[u16],
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
) -> Verdict {
    let gs = proto::GROUP_SIZE;
    let arm_row_major = ref_row_major(packed, scales, x, n, k, gs);
    let arm_group_major = ref_group_major_f64(packed, scales, x, n, k, gs, PlantedBug::None);
    let drift = max_rel_diff(&arm_row_major, &arm_group_major);
    assert!(
        drift < CROSS_DECOMPOSITION_REL_TOL_BOUNDS_F32_VS_F64_ACCUMULATION,
        "the two host decompositions disagree by {drift:.3e} before any candidate was judged \
         (n={n} k={k} gs={gs}); the forge refuses to grade against a broken oracle"
    );
    if let Some(i) = got
        .iter()
        .position(|&g| !bf16::from_bits(g).to_f32().is_finite())
    {
        return Verdict::Reject(format!(
            "row {i} is non-finite: the NaN poison survived, so the row was never written \
             (n={n} k={k})"
        ));
    }
    let worst = max_rel_bf16_output(got, &arm_row_major)
        .max(max_rel_bf16_output(got, &arm_group_major));
    if worst <= KERNEL_VS_HOST_ORACLE_REL_TOL_PINNED_BY_GEMV_W4A16_SUITES {
        Verdict::Pass(worst)
    } else {
        Verdict::Reject(format!(
            "max rel {worst:.3e} over both oracle arms exceeds the bound \
             {KERNEL_VS_HOST_ORACLE_REL_TOL_PINNED_BY_GEMV_W4A16_SUITES:.0e} that \
             gemv_w4a16_cpu_ref and gemv_w4a16_m1_proto already pin (n={n} k={k})"
        ))
    }
}

fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("{test}: {}", ctx.summary());
            let st = ctx.qualify();
            if st.qualified {
                return Some(ctx);
            }
            if std::env::var("NV_KERNELS_WGPU_ALLOW_SKIP").as_deref() != Ok("1") {
                panic!("{test}: adapter not qualified: {:?}", st.reason);
            }
            eprintln!("{test}: SKIP adapter not qualified: {:?}", st.reason);
            None
        }
        Err(e) => {
            if std::env::var("NV_KERNELS_WGPU_ALLOW_SKIP").as_deref() != Ok("1") {
                panic!(
                    "{test}: no wgpu adapter: {e}. This gate refuses to report success without \
                     running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
                );
            }
            eprintln!("{test}: SKIP no wgpu adapter: {e}");
            None
        }
    }
}

#[test]
fn the_forge_gate_rejects_every_planted_dequant_bug_before_it_grades_a_candidate() {
    let (n, k, gs) = (7usize, 512usize, proto::GROUP_SIZE);
    let (packed, scales, x) = gen_inputs(n, k, gs, 0x5eed);
    let good = ref_group_major_f64(&packed, &scales, &x, n, k, gs, PlantedBug::None);
    let as_bits = |v: &[f32]| -> Vec<u16> { v.iter().map(|f| bf16::from_f32(*f).to_bits()).collect() };
    match judge(&as_bits(&good), &packed, &scales, &x, n, k) {
        Verdict::Pass(worst) => assert!(
            worst <= KERNEL_VS_HOST_ORACLE_REL_TOL_PINNED_BY_GEMV_W4A16_SUITES,
            "control arm reported {worst}"
        ),
        Verdict::Reject(why) => panic!(
            "the control arm -- the oracle's own output -- was rejected: {why}. A gate that \
             fails its own reference grades nothing."
        ),
    }
    for bug in [
        PlantedBug::ScaleIndexOffByOneGroup,
        PlantedBug::NibbleOrderReversed,
        PlantedBug::MissingSignOffset,
    ] {
        let bad = ref_group_major_f64(&packed, &scales, &x, n, k, gs, bug);
        assert!(
            matches!(
                judge(&as_bits(&bad), &packed, &scales, &x, n, k),
                Verdict::Reject(_)
            ),
            "a planted dequant bug passed the forge acceptance rule at n={n} k={k} gs={gs}; \
             the harness would sign off on a wrong candidate (05.2 planted-bug protocol)"
        );
    }
}

#[test]
fn every_candidate_compiles_and_declares_the_proto_binding_contract() {
    recycle_prior_failures();
    let cands = candidates();
    let mut rejected = 0usize;
    for c in &cands {
        match compile_host_side(&c.source, &c.entry) {
            Ok(()) => eprintln!("[forge] {}: compiles, entry {} present", c.stem, c.entry),
            Err(msg) => {
                record_failure(&c.stem, "compile", &msg);
                rejected += 1;
            }
        }
        for binding in [
            "@group(0) @binding(0)",
            "@group(0) @binding(1)",
            "@group(0) @binding(2)",
            "@group(0) @binding(3)",
            "@group(0) @binding(4)",
        ] {
            assert!(
                c.source.contains(binding),
                "{}: candidate is missing {binding}; the forge binds packed/scale/x/y/params \
                 exactly as the shipping proto launcher does",
                c.stem
            );
        }
    }
    assert!(
        selected().is_some() || cands.len() >= 4,
        "the unfiltered candidate set shrank to {} files; a cycle ships generated variants plus \
         at least one planted-bug negative control, and a set without the control proves nothing",
        cands.len()
    );
    assert_eq!(
        rejected, 0,
        "{rejected} candidate(s) failed host-side compilation; the diagnostics are recycled to \
         {} for the next generation attempt",
        failure_log_path().display()
    );
}

#[test]
fn every_candidate_meets_or_misses_the_hardened_oracle_exactly_as_its_name_promises() {
    let Some(ctx) = ctx_or_skip("kernel_forge_gemv_w4a16") else {
        return;
    };
    let cands = candidates();
    for c in &cands {
        for &(n, k) in &GATE_SHAPES_ARE_MULTI_GROUP_SO_SCALE_INDEXING_IS_OBSERVABLE {
            let (packed, scales, x) = gen_inputs(n, k, proto::GROUP_SIZE, 0x1234 + (n * k) as u64);
            let got = match run_shader(
                ctx,
                "nv_kernels_kernel_forge",
                &c.source,
                &c.entry,
                &packed,
                &scales,
                &x,
                n,
                k,
            ) {
                Ok(v) => v,
                Err(e) => {
                    record_failure(&c.stem, "dispatch", &e);
                    panic!("{}: dispatch failed at n={n} k={k}: {e}", c.stem);
                }
            };
            match (judge(&got, &packed, &scales, &x, n, k), c.must_be_rejected) {
                (Verdict::Pass(worst), false) => {
                    eprintln!("[forge] {} n={n} k={k}: PASS max rel {worst:.3e}", c.stem)
                }
                (Verdict::Reject(why), true) => {
                    eprintln!("[forge] {} n={n} k={k}: REJECTED as designed: {why}", c.stem)
                }
                (Verdict::Reject(why), false) => {
                    record_failure(&c.stem, "oracle", &why);
                    panic!("{}: n={n} k={k}: {why}", c.stem);
                }
                (Verdict::Pass(worst), true) => panic!(
                    "{}: the planted-bug candidate PASSED at n={n} k={k} (max rel {worst:.3e}); \
                     the forge gate has no teeth and every green candidate above it is void",
                    c.stem
                ),
            }
        }
    }
}

#[test]
fn the_runner_script_gates_the_gpu_on_idle_and_names_this_suite() {
    let p = manifest(RUNNER_SUBPATH);
    let src = std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!(
            "the forge runner {} is missing: {e}; the pipeline is the runner plus this suite, \
             not this suite alone",
            p.display()
        )
    });
    for needle in [
        "kernel_forge_gemv_w4a16",
        "nvk.sh",
        "NVK_LANE",
        FAILURE_LOG_ENV,
        CANDIDATES_ENV,
        "memory.used",
    ] {
        assert!(
            src.contains(needle),
            "{} no longer mentions {needle}; a latency number the runner took without the idle \
             gate, or a gate the runner never ran, is not a number",
            p.display()
        );
    }
}

fn bench_one(
    ctx: &WgpuContext,
    label: &str,
    source: &str,
    entry: &str,
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
) -> f64 {
    let (params, groups) = geometry(ctx, n, k);
    let pb = dispatch::storage_from_slice(ctx, "forge-bench-packed", packed);
    let sb = dispatch::storage_from_slice(ctx, "forge-bench-scale", &widen_u16(scales));
    let xb = dispatch::storage_from_slice(ctx, "forge-bench-x", &pack_u16(x));
    let yb = dispatch::storage_zeroed(ctx, "forge-bench-y", (n * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "forge-bench-params", &params);
    let pipeline = dispatch::cached_compute_pipeline(ctx, label, &compose(source), entry)
        .unwrap_or_else(|e| panic!("{label}:{entry} pipeline: {e}"));
    let bindings: Vec<(u32, &wgpu::Buffer)> =
        vec![(0, &pb), (1, &sb), (2, &xb), (3, &yb), (4, &ub)];
    let group = dispatch::bind_group(ctx, &pipeline, &bindings);
    let submit = |count: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            for _ in 0..count {
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
        }
        ctx.queue.submit([enc.finish()]);
    };
    submit(BENCH_WARMUP_DISPATCHES);
    ctx.poll_blocking().expect("warmup poll");
    let t0 = Instant::now();
    submit(BENCH_TIMED_DISPATCHES);
    ctx.poll_blocking().expect("timed poll");
    t0.elapsed().as_secs_f64() / BENCH_TIMED_DISPATCHES as f64
}

#[test]
#[ignore]
fn latency_against_the_shipping_proto_shader() {
    let Some(ctx) = ctx_or_skip("kernel_forge_gemv_w4a16 bench") else {
        return;
    };
    let shipping = std::fs::read_to_string(manifest(SHIPPING_SHADER_SUBPATH))
        .expect("the shipping proto shader is the baseline every candidate is measured against");
    let shipping_entry = proto::entry_for(8).expect("warps8 entry");
    let cands = candidates();
    for &(name, n, k) in &BENCH_SHAPES_ARE_THE_E4B_GATE_UP_AND_DOWN_PROJECTIONS {
        let (packed, scales, x) = gen_inputs(n, k, proto::GROUP_SIZE, 0xBEEF);
        let bytes = (n * k / 2 + n * (k / proto::GROUP_SIZE) * 2) as f64;
        let base = bench_one(
            ctx,
            "nv_kernels_kernel_forge_baseline",
            &shipping,
            shipping_entry,
            &packed,
            &scales,
            &x,
            n,
            k,
        );
        println!(
            "=== {name} N={n} K={k} weights+scales {:.1} MB ===",
            bytes / 1e6
        );
        println!(
            "  gemv_w4a16_m1_proto (shipping): {:8.2} us  {:5.2} TB/s",
            base * 1e6,
            bytes / base / 1e12
        );
        for c in &cands {
            if c.must_be_rejected {
                continue;
            }
            let secs = bench_one(
                ctx,
                "nv_kernels_kernel_forge_candidate",
                &c.source,
                &c.entry,
                &packed,
                &scales,
                &x,
                n,
                k,
            );
            println!(
                "  {:<34} {:8.2} us  {:5.2} TB/s  {:+6.1}% vs shipping",
                c.stem,
                secs * 1e6,
                bytes / secs / 1e12,
                (secs / base - 1.0) * 100.0
            );
        }
    }
}
