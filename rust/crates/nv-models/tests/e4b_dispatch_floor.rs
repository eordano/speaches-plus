#![cfg(feature = "wgpu")]

mod common;
use common::ctx;
use common::env_usize;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct P {
    elems: u32,
    chain: u32,
    never: u32,
    pad: u32,
}

const WGSL: &str = r#"
struct P { elems: u32, chain: u32, never: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> df_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> df_dst: array<u32>;
@group(0) @binding(2) var<uniform> df_p: P;

var<workgroup> df_red: array<f32, 256>;
var<workgroup> df_big: array<f32, 4096>;

fn df_sink(x: u32) {
    if (df_p.never == 0xffffffffu) {
        df_dst[0] = x + df_src[0];
    }
}

@compute @workgroup_size(256)
fn df_nul(@builtin(local_invocation_id) t: vec3<u32>) {
    df_sink(t.x);
}

@compute @workgroup_size(256)
fn df_bar1(@builtin(local_invocation_id) t: vec3<u32>) {
    workgroupBarrier();
    df_sink(t.x);
}

@compute @workgroup_size(256)
fn df_bar8(@builtin(local_invocation_id) t: vec3<u32>) {
    for (var i = 0u; i < 8u; i = i + 1u) {
        workgroupBarrier();
    }
    df_sink(t.x);
}

@compute @workgroup_size(256)
fn df_red256(@builtin(local_invocation_id) t: vec3<u32>) {
    let lid = t.x;
    df_red[lid] = f32(lid) + f32(df_p.elems);
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            df_red[lid] = df_red[lid] + df_red[lid + s];
        }
        workgroupBarrier();
    }
    df_sink(u32(df_red[0]));
}

@compute @workgroup_size(256)
fn df_chain(@builtin(global_invocation_id) g: vec3<u32>) {
    var i = g.x % df_p.elems;
    for (var k = 0u; k < df_p.chain; k = k + 1u) {
        i = df_src[i];
    }
    df_sink(i);
}

@compute @workgroup_size(256)
fn df_wgmem(@builtin(local_invocation_id) t: vec3<u32>) {
    if (df_p.never == 0xffffffffu) {
        df_big[t.x] = f32(t.x);
        df_dst[0] = u32(df_big[t.x ^ 1u]);
    }
    df_sink(t.x);
}
"#;

struct Bufs {
    src: wgpu::Buffer,
    dst: wgpu::Buffer,
    p: wgpu::Buffer,
}

fn make_bufs(ctx: &WgpuContext, elems: usize) -> Bufs {
    let mut v = vec![0u32; elems];
    let step = 0x9e37_79b9u64;
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = (((i as u64).wrapping_mul(step).wrapping_add(0x1234_5)) % elems as u64) as u32;
    }
    let fixed = v
        .iter()
        .enumerate()
        .filter(|(i, x)| *i as u32 == **x)
        .count();
    assert!(
        fixed * 1000 < elems,
        "chase table has {fixed} fixed points of {elems}; a walk that stands still measures no \
         memory latency"
    );
    Bufs {
        src: dispatch::storage_from_slice(ctx, "df-src", &v),
        dst: dispatch::storage_zeroed(ctx, "df-dst", 256),
        p: dispatch::uniform_from(
            ctx,
            "df-p",
            &P {
                elems: elems as u32,
                chain: 0,
                never: 0,
                pad: 0,
            },
        ),
    }
}

fn params(ctx: &WgpuContext, elems: usize, chain: u32) -> wgpu::Buffer {
    dispatch::uniform_from(
        ctx,
        "df-p",
        &P {
            elems: elems as u32,
            chain,
            never: 0,
            pad: 0,
        },
    )
}

fn submit_copies(
    ctx: &WgpuContext,
    pl: &wgpu::ComputePipeline,
    bind: &wgpu::BindGroup,
    grid: u32,
    copies: usize,
    reps: usize,
) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut worst: f64 = 0.0;
    for r in 0..reps + 1 {
        let t0 = Instant::now();
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pl);
            for _ in 0..copies {
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(grid, 1, 1);
            }
        }
        ctx.queue.submit([enc.finish()]);
        ctx.poll_blocking().expect("drain");
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if r == 0 {
            continue;
        }
        best = best.min(ms);
        worst = worst.max(ms);
    }
    (best, worst)
}

struct Arm {
    us: f64,
    drift_pct: f64,
    spread_pct: f64,
}

fn price(
    ctx: &WgpuContext,
    pl: &wgpu::ComputePipeline,
    bind: &wgpu::BindGroup,
    grid: u32,
    lo: usize,
    hi: usize,
    reps: usize,
) -> Arm {
    let (a, aw) = submit_copies(ctx, pl, bind, grid, lo, reps);
    let (h, _) = submit_copies(ctx, pl, bind, grid, hi, reps);
    let (a2, _) = submit_copies(ctx, pl, bind, grid, lo, reps);
    Arm {
        us: (h - 0.5 * (a + a2)) / (hi - lo) as f64 * 1e3,
        drift_pct: 100.0 * (a2 - a) / a,
        spread_pct: 100.0 * (aw - a) / a,
    }
}

fn build(ctx: &'static WgpuContext, entry: &str, zero_init: bool) -> wgpu::ComputePipeline {
    dispatch::compute_pipeline_opts(
        ctx,
        &format!("df-{entry}-{}", if zero_init { "zi" } else { "nozi" }),
        WGSL,
        entry,
        !zero_init,
    )
    .unwrap_or_else(|e| panic!("{entry}: {e}"))
}

#[test]
#[ignore = "timing instrument; set NV_E4B_DISPATCH_FLOOR=1"]
fn e4b_dispatch_floor_decomposition() {
    assert_eq!(
        std::env::var("NV_E4B_DISPATCH_FLOOR").ok().as_deref(),
        Some("1"),
        "set NV_E4B_DISPATCH_FLOOR=1 -- a silent skip here would report a pass"
    );
    let ctx = ctx();
    eprintln!("adapter: {}", ctx.summary());
    if let Ok(o) = std::process::Command::new("uptime").output() {
        eprintln!("host: {}", String::from_utf8_lossy(&o.stdout).trim());
    }
    let reps = env_usize("NV_E4B_DISPATCH_FLOOR_REPS", 12);
    let elems = env_usize("NV_E4B_DISPATCH_FLOOR_ELEMS", 16 << 20);
    let b = make_bufs(ctx, elems);

    let nul = build(ctx, "df_nul", false);
    let bind_of = |pl: &wgpu::ComputePipeline, p: &wgpu::Buffer| {
        dispatch::bind_group(ctx, pl, &[(0, &b.src), (1, &b.dst), (2, p)])
    };

    eprintln!(
        "\n==== an empty 256-thread dispatch, by grid ====\n\
         the graph's real grids are 1 (fused_norm_a), 2 (fused_attn_k), 8 (flash_stage2),\n\
         16/160 (per-layer w4), 128 (flash_stage1), 1280 (gate_up), 16384 (lm_head)"
    );
    eprintln!(
        "{:>8}  {:>10}  {:>9}  {:>9}",
        "wgs", "us/disp", "drift%", "spread%"
    );
    let bind_nul = bind_of(&nul, &b.p);
    let mut nul_us = std::collections::BTreeMap::new();
    for g in [
        1u32, 2, 4, 8, 16, 20, 42, 76, 128, 160, 256, 512, 1280, 16384,
    ] {
        let a = price(ctx, &nul, &bind_nul, g, 8, 72, reps);
        eprintln!(
            "{g:>8}  {:>10.2}  {:>+9.2}  {:>9.2}",
            a.us, a.drift_pct, a.spread_pct
        );
        nul_us.insert(g, a.us);
    }
    let floor_us = *nul_us.get(&8).expect("grid 8");

    eprintln!(
        "\n==== one body feature at a time, all at grid 8 (flash_stage2's grid) ====\n\
         'over empty' is the feature's own cost; the empty arm at this grid is {floor_us:.2} us"
    );
    eprintln!(
        "{:<34} {:>10}  {:>11}  {:>9}",
        "arm", "us/disp", "over empty", "drift%"
    );
    let feature = |name: &str, pl: &wgpu::ComputePipeline, p: &wgpu::Buffer| -> f64 {
        let bind = bind_of(pl, p);
        let a = price(ctx, pl, &bind, 8, 8, 72, reps);
        eprintln!(
            "{name:<34} {:>10.2}  {:>11.2}  {:>+9.2}",
            a.us,
            a.us - floor_us,
            a.drift_pct
        );
        a.us
    };
    let bar1 = build(ctx, "df_bar1", false);
    let bar8 = build(ctx, "df_bar8", false);
    let red = build(ctx, "df_red256", false);
    let chain = build(ctx, "df_chain", false);
    let wg_zi = build(ctx, "df_wgmem", true);
    let wg_no = build(ctx, "df_wgmem", false);
    let us_bar1 = feature("1 workgroupBarrier", &bar1, &b.p);
    let us_bar8 = feature("8 workgroupBarriers", &bar8, &b.p);
    let us_red = feature("256-lane tree reduction (8 rounds)", &red, &b.p);
    let mut chain_us = Vec::new();
    for k in [0u32, 1, 2, 4, 8, 16, 32] {
        let p = params(ctx, elems, k);
        chain_us.push(feature(
            &format!("dependent global loads, chain={k}"),
            &chain,
            &p,
        ));
    }
    let us_zi = feature("16 KiB workgroup mem, zero-init ON", &wg_zi, &b.p);
    let us_no = feature("16 KiB workgroup mem, zero-init OFF", &wg_no, &b.p);

    let per_load = (chain_us[6] - chain_us[1]) / 31.0;
    eprintln!(
        "\none dependent global load costs {:.3} us at grid 8; the 8-round tree reduction costs \
         {:.2} us and one bare barrier {:.2} us. The per-launch memset of 16 KiB of workgroup \
         memory costs {:.2} us ({:.2} vs {:.2}).",
        per_load,
        us_red - floor_us,
        us_bar1 - floor_us,
        us_zi - us_no,
        us_zi,
        us_no
    );

    assert!(
        chain_us[6] > chain_us[0] + 1.0,
        "a 32-deep pointer chase ({:.2} us) is not measurably above a 0-deep one ({:.2} us) -- the \
         compiler folded the loads and every latency number here is fiction",
        chain_us[6],
        chain_us[0]
    );
    assert!(
        us_bar8 > 0.0 && us_red > floor_us,
        "the reduction arm ({us_red:.2} us) did not clear the empty arm ({floor_us:.2} us); the \
         workgroup array was optimized out and this table prices nothing"
    );

    eprintln!(
        "\n==== the serial premium: D dispatches of grid G vs one dispatch of grid D*G ====\n\
         identical per-workgroup work either way. This difference is what merging D independent\n\
         small passes into one wider pass can buy, and nothing more."
    );
    for (name, pl, p) in [
        ("256-lane tree reduction", &red, &b.p),
        ("chain=8 dependent loads", &chain, &params(ctx, elems, 8)),
    ] {
        eprintln!("\n  {name}");
        eprintln!(
            "{:>6} {:>6}  {:>12}  {:>12}  {:>10}",
            "G", "D", "D serial us", "1 wide us", "merge x"
        );
        let bind = bind_of(pl, p);
        for g in [1u32, 2, 8] {
            let one = price(ctx, pl, &bind, g, 8, 72, reps);
            for d in [2u32, 4, 8, 16] {
                let wide = price(ctx, pl, &bind, g * d, 8, 72, reps);
                eprintln!(
                    "{g:>6} {d:>6}  {:>12.2}  {:>12.2}  {:>10.2}",
                    one.us * d as f64,
                    wide.us,
                    one.us * d as f64 / wide.us
                );
            }
        }
    }

    let seen = dispatch::read_back::<u32>(ctx, &b.dst, 4).expect("readback");
    assert_eq!(
        seen[0], 0,
        "the never-taken sink fired; the arms above measured a store, not the feature under test"
    );
}
