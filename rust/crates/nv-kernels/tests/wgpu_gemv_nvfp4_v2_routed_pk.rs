#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::kernels::{gemv_nvfp4 as g4, gemv_nvfp4_v2 as v2};
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
mod common;
use common::lcg_hi33_u32 as lcg;
use common::pipeline;

fn ctx(what: &str) -> &'static WgpuContext {
    let ctx = WgpuContext::shared()
        .unwrap_or_else(|e| panic!("{what}: no wgpu adapter, this proof cannot run: {e}"));
    eprintln!("{what}: {}", ctx.summary());
    ctx
}

struct Routed {
    label: &'static str,
    n: usize,
    k: usize,
    slots: usize,

    per_token: usize,
}

const NVFP4_GEMVS_PER_TOKEN: usize = 110;

const ROUTED: &[Routed] = &[
    Routed {
        label: "at-qproj",
        n: 8192,
        k: 2048,
        slots: 1,
        per_token: 10,
    },
    Routed {
        label: "at-vkproj(rowcat)",
        n: 1024,
        k: 2048,
        slots: 1,
        per_token: 10,
    },
    Routed {
        label: "at-oproj",
        n: 2048,
        k: 4096,
        slots: 1,
        per_token: 10,
    },
    Routed {
        label: "moe-gateup(rowcat)",
        n: 1024,
        k: 2048,
        slots: 9,
        per_token: 40,
    },
    Routed {
        label: "moe-down",
        n: 2048,
        k: 512,
        slots: 9,
        per_token: 40,
    },
    Routed {
        label: "at-vproj(kv unfused)",
        n: 512,
        k: 2048,
        slots: 1,
        per_token: 0,
    },
    Routed {
        label: "at-kproj(kv unfused)",
        n: 512,
        k: 2048,
        slots: 1,
        per_token: 0,
    },
    Routed {
        label: "moe-gate(unfused)",
        n: 512,
        k: 2048,
        slots: 9,
        per_token: 0,
    },
    Routed {
        label: "moe-up(unfused)",
        n: 512,
        k: 2048,
        slots: 9,
        per_token: 0,
    },
    Routed {
        label: "moe-sgate(fold off)",
        n: 512,
        k: 2048,
        slots: 1,
        per_token: 0,
    },
    Routed {
        label: "moe-sup(fold off)",
        n: 512,
        k: 2048,
        slots: 1,
        per_token: 0,
    },
    Routed {
        label: "moe-sdown(fold off)",
        n: 2048,
        k: 512,
        slots: 1,
        per_token: 0,
    },
];

fn reads_neighbour_slot(kernel: v2::V2Kernel) -> bool {
    matches!(kernel, v2::V2Kernel::Warp | v2::V2Kernel::FDec)
}

const SG_PROBE: &str = "
@group(0) @binding(0) var<storage, read_write> sgw: array<atomic<u32>, 2>;
@compute @workgroup_size(WGSIZE)
fn probe_width(@builtin(subgroup_size) s: u32) {
    atomicMin(&sgw[0], s);
    atomicMax(&sgw[1], s);
}
";

fn probe_width_at(ctx: &WgpuContext, wg: u32) -> (u32, u32) {
    let src = SG_PROBE.replace("WGSIZE", &wg.to_string());
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("v2-sgw"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
    let pl = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("v2-sgw"),
            layout: None,
            module: &module,
            entry_point: Some("probe_width"),
            compilation_options: Default::default(),
            cache: None,
        });
    if let Some(e) = pollster::block_on(scope.pop()) {
        panic!("subgroup probe at workgroup_size({wg}) failed to compile: {e}");
    }
    let buf = dispatch::storage_from_slice(ctx, "v2-sgw", &[u32::MAX, 0u32]);
    let bind = dispatch::bind_group(ctx, &pl, &[(0, &buf)]);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pl);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(64, 1, 1);
    }
    ctx.queue.submit([enc.finish()]);
    ctx.poll_blocking().expect("poll");
    let got = dispatch::read_back::<u32>(ctx, &buf, 2).expect("probe readback");
    (got[0], got[1])
}

#[test]
fn the_subgroup_width_is_32_at_every_workgroup_size_the_route_emits() {
    let ctx = ctx("v2-routed-sgwidth");
    assert!(
        ctx.caps.subgroup,
        "the v2 entries reduce with subgroup shuffles; without the SUBGROUP feature the route \
         is unreachable and this proof cannot pass"
    );
    let shipped = ctx.subgroup_width();
    assert_eq!(
        shipped,
        Some(v2::NV2_LANES),
        "the shipped probe reports {shipped:?}, so subgroup32_ok would already refuse the route"
    );

    let mut widths: Vec<u32> = ROUTED
        .iter()
        .map(|r| v2::select_slots(r.n, r.k, r.slots).1.wg)
        .collect();
    widths.sort_unstable();
    widths.dedup();
    assert!(
        !widths.is_empty(),
        "no routed shape selected a config, so this test probed nothing"
    );
    for wg in widths {
        let (lo, hi) = probe_width_at(ctx, wg);
        eprintln!("subgroup width at workgroup_size({wg:>3}): min {lo} max {hi}");
        assert_eq!(
            (lo, hi),
            (v2::NV2_LANES, v2::NV2_LANES),
            "at workgroup_size({wg}) this adapter runs {lo}..{hi}-wide subgroups, not {}. \
             NV2_SGS = NV2_WG/32 is compiled into nv2_pk_bits' length and into the row map, so \
             at this width some subgroup never writes its slot and gemv_nvfp4_{{fdec,warp}}_pk \
             pair-pack a word out of workgroup memory nobody wrote -- and both entries ship with \
             zero-init disabled. The shipped subgroup32_ok probe only ever compiled \
             @workgroup_size(256)",
            v2::NV2_LANES
        );
    }
}

#[test]
fn every_routed_shape_pair_packs_inside_nv2_pk_bits() {
    let mut neighbour_cells = 0usize;
    let (mut total, mut on_pack) = (0usize, 0usize);
    for r in ROUTED {
        let (kernel, cfg) = v2::select_slots(r.n, r.k, r.slots);
        let (pk_kernel, pk_cfg, entry) =
            v2::select_pk_slots(r.n, r.k, r.slots).unwrap_or_else(|| {
                panic!(
                "{}: n={} k={} slots={} takes no pair-packed route, but the graph binds a packed \
                 y buffer for every nvfp4 GEMV",
                r.label, r.n, r.k, r.slots
            )
            });
        assert_eq!((pk_kernel, pk_cfg), (kernel, cfg), "{}", r.label);
        let sgs = cfg.subgroups();
        let rpg = cfg.rows_per_group(kernel);
        eprintln!(
            "{:<20} n={:<5} k={:<5} slots={} -> {entry:<22} wg={} mr={} subgroups={sgs} \
             rows/group={rpg} groups={}",
            r.label,
            r.n,
            r.k,
            r.slots,
            cfg.wg,
            cfg.mr,
            r.n / rpg as usize
        );
        assert!(
            rpg.is_multiple_of(2),
            "{}: {rpg} rows per workgroup is odd, so two workgroups pack the same y word",
            r.label
        );
        assert_eq!(
            r.n % rpg as usize,
            0,
            "{}: n={} is not a whole number of {rpg}-row workgroups; the tail group runs \
             subgroups that never write their nv2_pk_bits slot",
            r.label,
            r.n
        );
        total += r.per_token;
        if !reads_neighbour_slot(kernel) {
            continue;
        }
        on_pack += r.per_token;
        neighbour_cells += 1;
        assert!(
            sgs.is_multiple_of(2),
            "{}: {entry} reads nv2_pk_bits[sgid + 1] from every even subgroup and this config \
             holds {sgs} of them -- the last even subgroup indexes one past the array",
            r.label
        );
        let last_even = sgs - 2;
        assert!(
            last_even + 1 < sgs,
            "{}: sgid={last_even} would read nv2_pk_bits[{}] out of {sgs}",
            r.label,
            last_even + 1
        );
    }
    assert!(
        neighbour_cells >= 4,
        "only {neighbour_cells} routed shapes reached the cross-subgroup pack; this test stopped \
         covering the thing it exists for"
    );
    assert_eq!(
        total, NVFP4_GEMVS_PER_TOKEN,
        "this table accounts for {total} nvfp4 GEMVs per token and the graph builds \
         {NVFP4_GEMVS_PER_TOKEN}; a shape it does not list is a shape nothing here checked"
    );

    eprintln!(
        "cross-subgroup pack carries {on_pack} of {total} nvfp4 GEMVs per token \
         ({:.0}%); before 8eb9ff466 it carried {} ({:.0}%)",
        100.0 * on_pack as f64 / total as f64,
        on_pack + 40,
        100.0 * (on_pack + 40) as f64 / total as f64
    );
    assert_eq!(
        on_pack, 50,
        "the share of the token that lands on the cross-subgroup pair-pack moved to {on_pack} of \
         {total}; that is the exposure the corruption event's bound was measured against"
    );
}

#[test]
fn an_odd_subgroup_count_is_refused_rather_than_read_out_of_range() {
    for kernel in [v2::V2Kernel::Warp, v2::V2Kernel::FDec] {
        for wg in [32u32, 96, 160, 224] {
            let cfg = v2::V2Config::new(wg, 1);
            assert!(cfg.valid(), "wg={wg} is a legal workgroup size");
            assert!(
                !cfg.subgroups().is_multiple_of(2),
                "wg={wg} was chosen to give an odd subgroup count"
            );
            assert!(
                !v2::pk_capable(kernel, cfg),
                "{:?} at wg={wg} holds {} subgroups and pk_capable still allows the pair-pack; \
                 nv2_pk_bits[sgid + 1u] then indexes past the array",
                kernel,
                cfg.subgroups()
            );
        }
        assert!(
            v2::pk_capable(kernel, v2::V2Config::new(64, 1)),
            "the guard must not reject the even counts the route depends on"
        );
    }
}

const SCALE_LO: u32 = 56;
const SCALE_HI: u32 = 126;

fn scale_bytes(words: usize, seed: &mut u64) -> Vec<u32> {
    (0..words)
        .map(|_| {
            let mut w = 0u32;
            for b in 0..4 {
                let v = SCALE_LO + (lcg(seed) % (SCALE_HI - SCALE_LO + 1));
                w |= v << (8 * b);
            }
            w
        })
        .collect()
}

const POISON_SRC: &str = "
@compute @workgroup_size(NV2_WG)
fn nv2r_poison(@builtin(local_invocation_id) tid: vec3<u32>) {
    if (tid.x < NV2_SGS) {
        nv2_pk_bits[tid.x] = (0xdead0000u | tid.x) ^ (nv2_y[0] & 0xffu);
    }
}

@compute @workgroup_size(NV2_WG)
fn nv2r_tripwire(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    if (sgid == 0u && lane == 0u) {
        nv2_pk_bits[0] = 1u;
    }
    workgroupBarrier();
    let row = (wid.x + wid.y * nv2_p.groups_x) * NV2_SGS + sgid;
    if (lane == 0u && row < nv2_p.n_rows) {
        nv2_y[row] = nv2_pk_bits[sgid];
    }
}
";

struct Cell {
    w: Vec<u32>,
    ws: Vec<u32>,
    x: Vec<u32>,
    xs: Vec<u32>,
}

fn cell(n: usize, k: usize, seed: u64) -> Cell {
    let k_blocks = k / 16;
    let mut s = seed;
    Cell {
        w: (0..n * k_blocks * 2).map(|_| lcg(&mut s)).collect(),
        ws: scale_bytes(g4::swizzled_scale_len(n, k_blocks) / 4, &mut s),
        x: (0..k_blocks * 2).map(|_| lcg(&mut s)).collect(),
        xs: scale_bytes(k_blocks.div_ceil(4), &mut s),
    }
}

struct Bound {
    p: wgpu::Buffer,
    w: wgpu::Buffer,
    ws: wgpu::Buffer,
    x: wgpu::Buffer,
    xs: wgpu::Buffer,
}

fn upload(ctx: &WgpuContext, c: &Cell, n: usize, k: usize, groups_x: u32) -> Bound {
    Bound {
        p: dispatch::uniform_from(ctx, "v2r-p", &g4::gemv_params(1.0, n, k, groups_x)),
        w: dispatch::storage_from_slice(ctx, "v2r-w", &c.w),
        ws: dispatch::storage_from_slice(ctx, "v2r-ws", &c.ws),
        x: dispatch::storage_from_slice(ctx, "v2r-x", &c.x),
        xs: dispatch::storage_from_slice(ctx, "v2r-xs", &c.xs),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Slots {
    Vec2,
    Vec4,
    ParamsOnly,
}

fn slots_of(kernel: v2::V2Kernel) -> Slots {
    if kernel.vec4_slots() {
        Slots::Vec4
    } else {
        Slots::Vec2
    }
}

fn binds<'a>(b: &'a Bound, y: &'a wgpu::Buffer, s: Slots) -> Vec<(u32, &'a wgpu::Buffer)> {
    match s {
        Slots::ParamsOnly => vec![(v2::PARAMS_SLOT, &b.p), (v2::Y_SLOT, y)],
        Slots::Vec4 => vec![
            (v2::WS_SLOT, &b.ws),
            (v2::XS_SLOT, &b.xs),
            (v2::PARAMS_SLOT, &b.p),
            (v2::Y_SLOT, y),
            (v2::W4_SLOT, &b.w),
            (v2::X4_SLOT, &b.x),
        ],
        Slots::Vec2 => vec![
            (v2::W2_SLOT, &b.w),
            (v2::WS_SLOT, &b.ws),
            (v2::X2_SLOT, &b.x),
            (v2::XS_SLOT, &b.xs),
            (v2::PARAMS_SLOT, &b.p),
            (v2::Y_SLOT, y),
        ],
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    ctx: &WgpuContext,
    pl: &wgpu::ComputePipeline,
    poison: Option<&wgpu::ComputePipeline>,
    b: &Bound,
    slots: Slots,
    words: usize,
    grid: (u32, u32, u32),
    reps: usize,
) -> Vec<u32> {
    let y = dispatch::storage_zeroed(ctx, "v2r-y", (words * 4) as u64);
    let bind = dispatch::bind_group(ctx, pl, &binds(b, &y, slots));
    let pbind = poison.map(|p| dispatch::bind_group(ctx, p, &[(v2::Y_SLOT, &y)]));
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for _ in 0..reps {
            if let (Some(p), Some(pb)) = (poison, pbind.as_ref()) {
                pass.set_pipeline(p);
                pass.set_bind_group(0, pb, &[]);
                pass.dispatch_workgroups(grid.0, grid.1, grid.2);
            }
            pass.set_pipeline(pl);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(grid.0, grid.1, grid.2);
        }
    }
    ctx.queue.submit([enc.finish()]);
    ctx.poll_blocking().expect("poll");
    dispatch::read_back::<u32>(ctx, &y, words).expect("y readback")
}

fn moved(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

fn first_diffs(a: &[u32], b: &[u32]) -> Vec<String> {
    a.iter()
        .zip(b.iter())
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .take(4)
        .map(|(i, (x, y))| format!("[{i}] {x:#010x} vs {y:#010x}"))
        .collect()
}

#[test]
fn the_routed_pk_entries_are_write_before_read_at_their_routed_config() {
    let ctx = ctx("v2-routed-nozi");
    assert_eq!(
        ctx.subgroup_width(),
        Some(v2::NV2_LANES),
        "nv2_pk_bits is sized NV2_WG/32; the write-before-read argument is only valid at a \
         32-wide subgroup"
    );

    let mut cells = 0usize;
    let mut seen: Vec<(u32, u32, &str)> = Vec::new();
    for r in ROUTED {
        let (kernel, cfg) = v2::select_slots(r.n, r.k, r.slots);
        if !reads_neighbour_slot(kernel) {
            continue;
        }
        let entry = kernel.pk_entry().expect("routed pk entry");
        if seen.contains(&(cfg.wg, r.n as u32, entry)) {
            continue;
        }
        seen.push((cfg.wg, r.n as u32, entry));

        let src = format!("{}\n{}", v2::source(cfg), POISON_SRC);
        let pl_poison = pipeline(ctx, "v2r-poison", &src, "nv2r_poison", false);
        let grid = dispatch::workgroup_count_1d(ctx, r.n as u64, cfg.rows_per_group(kernel));
        let c = cell(r.n, r.k, 0x51ce_0f4b ^ (r.n as u64) ^ ((r.k as u64) << 20));
        let b = upload(ctx, &c, r.n, r.k, grid.0);

        let tw_zi = pipeline(ctx, "v2r-tw-zi", &src, "nv2r_tripwire", true);
        let tw_no = pipeline(ctx, "v2r-tw-no", &src, "nv2r_tripwire", false);
        let mut tw_moved = 0usize;
        for _ in 0..16 {
            let a = run(ctx, &tw_zi, None, &b, Slots::ParamsOnly, r.n, grid, 1);
            let p = run(
                ctx,
                &tw_no,
                Some(&pl_poison),
                &b,
                Slots::ParamsOnly,
                r.n,
                grid,
                1,
            );
            tw_moved = moved(&a, &p);
            if tw_moved > 0 {
                break;
            }
        }
        assert!(
            tw_moved > 0,
            "wg={} n={}: the tripwire read {} words it never wrote and saw the same value with \
             workgroup zero-init on as with it off under poison. The poison is not reaching \
             workgroup memory, so nothing below can discharge {entry}'s audit entry",
            cfg.wg,
            r.n,
            r.n
        );

        let pl_zi = pipeline(ctx, "v2r-zi", &src, entry, true);
        let pl_no = pipeline(ctx, "v2r-no", &src, entry, false);
        let sl = slots_of(kernel);
        let words = r.n.div_ceil(2);
        let zi = run(ctx, &pl_zi, None, &b, sl, words, grid, 1);
        let no = run(ctx, &pl_no, Some(&pl_poison), &b, sl, words, grid, 1);
        assert!(
            zi.iter().any(|v| *v != 0),
            "{entry} wg={} n={} produced an all-zero y, so parity is vacuous",
            cfg.wg,
            r.n
        );
        let diffs = first_diffs(&zi, &no);
        assert!(
            diffs.is_empty(),
            "{entry} at the ROUTED config wg={} (NV2_SGS={}) n={} k={} read workgroup memory it \
             had not written: {diffs:?}. The shipped audit only ever proved this at wg=256 \
             (NV2_SGS=8)",
            cfg.wg,
            cfg.subgroups(),
            r.n,
            r.k
        );
        eprintln!(
            "nozi-parity {entry:<22} wg={:<3} sgs={} n={:<5} k={:<5} | bit-identical under \
             poison, tripwire moved {tw_moved}/{} words",
            cfg.wg,
            cfg.subgroups(),
            r.n,
            r.k,
            r.n
        );
        cells += 1;
    }
    assert!(
        cells >= 2,
        "ran {cells} routed poison cells; the route stopped reaching the cross-subgroup pack"
    );
}

fn reps_env(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
        .max(1)
}

#[test]
fn the_packed_entry_equals_its_scalar_twin_on_every_routed_shape() {
    let ctx = ctx("v2-routed-pack");
    assert_eq!(
        ctx.subgroup_width(),
        Some(v2::NV2_LANES),
        "32-wide subgroups"
    );
    let reps = reps_env("NV_V2_PK_REPS", 64);

    let mut cells = 0usize;
    let mut seen: Vec<(&str, usize, usize)> = Vec::new();
    for r in ROUTED {
        let (kernel, cfg) = v2::select_slots(r.n, r.k, r.slots);
        let pk = kernel.pk_entry().expect("routed pk entry");
        if seen.contains(&(pk, r.n, r.k)) {
            continue;
        }
        seen.push((pk, r.n, r.k));

        let src = v2::source(cfg);
        let scalar = pipeline(ctx, "v2r-scalar", &src, kernel.entry(), true);
        let packed = pipeline(ctx, "v2r-packed", &src, pk, true);
        let grid = dispatch::workgroup_count_1d(ctx, r.n as u64, cfg.rows_per_group(kernel));
        let c = cell(r.n, r.k, 0x9e37_79b9 ^ (r.n as u64) ^ ((r.k as u64) << 17));
        let b = upload(ctx, &c, r.n, r.k, grid.0);
        let sl = slots_of(kernel);

        let s = run(ctx, &scalar, None, &b, sl, r.n, grid, 1);
        assert!(
            s.iter().any(|v| *v != 0),
            "{}: the scalar twin produced an all-zero y at n={} k={}",
            kernel.entry(),
            r.n,
            r.k
        );
        let want: Vec<u32> = (0..r.n / 2)
            .map(|i| (s[2 * i] & 0xffff) | ((s[2 * i + 1] & 0xffff) << 16))
            .collect();

        let p1 = run(ctx, &packed, None, &b, sl, r.n / 2, grid, 1);
        let diffs = first_diffs(&want, &p1);
        assert!(
            diffs.is_empty(),
            "{pk} n={} k={} wg={}: the pair-pack disagrees with the scalar twin it is supposed \
             to be a store rewrite of: {diffs:?}",
            r.n,
            r.k,
            cfg.wg
        );

        for i in 0..reps {
            let got = run(ctx, &packed, None, &b, sl, r.n / 2, grid, 2);
            let diffs = first_diffs(&p1, &got);
            assert!(
                diffs.is_empty(),
                "{pk} n={} k={} wg={}: rep {i} of {reps} disagreed with rep 0 on the same \
                 inputs, so the entry is not a function of its input: {diffs:?}",
                r.n,
                r.k,
                cfg.wg
            );
        }
        eprintln!(
            "pack-equiv  {pk:<22} wg={:<3} n={:<5} k={:<5} | {} rows match the scalar twin, \
             identical across {reps} submits x 2 back-to-back dispatches",
            cfg.wg, r.n, r.k, r.n
        );
        cells += 1;
    }
    assert!(cells >= 3, "ran {cells} pack-equivalence cells");
}
