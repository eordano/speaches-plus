#![cfg(feature = "wgpu")]

mod common;
use common::env_usize;
use common::med;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use nv_models::qwen3_5_moe_wgpu as q3w;
use std::time::Instant;

const NVFP4_BLOCK: usize = 16;

fn burst(
    ctx: &WgpuContext,
    pl: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    grid: (u32, u32, u32),
    copies: usize,
) -> f64 {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pl);
        pass.set_bind_group(0, bg, &[]);
        for _ in 0..copies {
            pass.dispatch_workgroups(grid.0, grid.1, grid.2);
        }
    }
    let s = Instant::now();
    ctx.queue.submit([enc.finish()]);
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    s.elapsed().as_secs_f64() * 1e3
}

fn bf16(v: f32) -> u32 {
    let b = v.to_bits();
    let r = (b >> 16) & 1;
    ((b + 0x7fff + r) >> 16) & 0xffff
}

fn pack_bf16(vals: &[f32]) -> Vec<u32> {
    assert!(vals.len().is_multiple_of(2), "bf16 pack needs even length");
    vals.chunks(2)
        .map(|c| bf16(c[0]) | (bf16(c[1]) << 16))
        .collect()
}

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn logval(&mut self) -> f32 {
        let u = self.next_u64();
        if u % 512 == 0 {
            return match (u >> 9) % 3 {
                0 => f32::INFINITY,
                1 => f32::NEG_INFINITY,
                _ => f32::NAN,
            };
        }
        let mag = 2f32.powi(((u >> 40) % 40) as i32 - 24);
        let frac = ((u >> 8) & 0xffff) as f32 / 65536.0;
        let s = if u & 1 == 0 { 1.0 } else { -1.0 };
        s * mag * (0.5 + frac)
    }
}

struct Arm {
    entry: &'static str,
    us: f64,
    null_us: f64,
}

struct Variant {
    entry: &'static str,
    wg_blocks: usize,
    ref_of: &'static str,
}

fn variants() -> Vec<Variant> {
    let mut v = vec![
        Variant {
            entry: "q3w_quant_rows",
            wg_blocks: 256,
            ref_of: "q3w_quant_rows",
        },
        Variant {
            entry: "q3w_silu_mul_quant",
            wg_blocks: 256,
            ref_of: "q3w_silu_mul_quant",
        },
    ];
    for (e, r, wg) in q3w::quant_lane_entries() {
        v.push(Variant {
            entry: e,
            wg_blocks: wg / 8,
            ref_of: r,
        });
    }
    v
}

struct Bench {
    x: wgpu::Buffer,
    gu: wgpu::Buffer,
    sel: wgpu::Buffer,
    glob: wgpu::Buffer,
    packed: wgpu::Buffer,
    scales: wgpu::Buffer,
    src: String,
}

impl Bench {
    fn new(ctx: &'static WgpuContext, max_kb: usize, max_slots: usize, seed: u64) -> Self {
        let max_k = max_kb * NVFP4_BLOCK;
        let mut rng = Rng(seed);
        let gu: Vec<f32> = (0..max_slots * 2 * max_k).map(|_| rng.logval()).collect();
        let xrow: Vec<f32> = (0..max_slots * max_k).map(|_| rng.logval()).collect();
        let globals: Vec<f32> = (0..257)
            .map(|i| 2f32.powi((i as i32 % 17) - 8) * (1.0 + (i % 7) as f32 / 8.0))
            .collect();
        let sel: Vec<u32> = (0..max_slots).map(|i| ((i * 37) % 257) as u32).collect();
        Self {
            x: dispatch::storage_from_slice(ctx, "q-x", &pack_bf16(&xrow)),
            gu: dispatch::storage_from_slice(ctx, "q-gu", &pack_bf16(&gu)),
            sel: dispatch::storage_from_slice(ctx, "q-sel", &sel),
            glob: dispatch::storage_from_slice(ctx, "q-glob", &globals),
            packed: dispatch::storage_zeroed(ctx, "q-packed", (max_slots * max_kb * 2 * 4) as u64),
            scales: dispatch::storage_zeroed(ctx, "q-scales", (max_slots * max_kb / 4 * 4) as u64),
            src: q3w::nvfp4_quant_source(),
        }
    }

    #[allow(clippy::type_complexity)]
    fn build(
        &self,
        ctx: &'static WgpuContext,
        v: &Variant,
        kb: usize,
        slots: usize,
        x_per_slot: bool,
    ) -> (
        std::sync::Arc<wgpu::ComputePipeline>,
        wgpu::BindGroup,
        (u32, u32, u32),
        wgpu::Buffer,
        wgpu::Buffer,
    ) {
        let k = kb * NVFP4_BLOCK;
        let is_silu = v.entry.contains("silu");
        let p = dispatch::uniform_from(
            ctx,
            "q-p",
            &q3w::QuantRowsParams {
                k_blocks: kb as u32,
                n_slots: slots as u32,
                use_sel: 1,
                x_slot_stride_elems: match (is_silu, x_per_slot) {
                    (true, _) => 2 * k as u32,
                    (false, true) => k as u32,
                    (false, false) => 0,
                },
            },
        );
        let pp = dispatch::uniform_from(
            ctx,
            "q-pp",
            &q3w::SiluPairParams {
                u_off_elems: if is_silu { k as u32 } else { 0 },
                ..Default::default()
            },
        );
        let pl = dispatch::cached_compute_pipeline(ctx, v.entry, &self.src, v.entry)
            .unwrap_or_else(|e| panic!("{}: {e}", v.entry));
        let binds: Vec<(u32, &wgpu::Buffer)> = if is_silu {
            vec![
                (11, &p),
                (12, &self.packed),
                (13, &self.scales),
                (14, &self.sel),
                (15, &self.glob),
                (16, &self.gu),
                (17, &self.gu),
                (18, &pp),
            ]
        } else {
            vec![
                (10, &self.x),
                (11, &p),
                (12, &self.packed),
                (13, &self.scales),
                (14, &self.sel),
                (15, &self.glob),
            ]
        };
        let bg = dispatch::bind_group(ctx, &pl, &binds);
        let gx = kb.div_ceil(v.wg_blocks).max(1) as u32;
        (pl, bg, (gx, slots as u32, 1), p, pp)
    }
}

fn quant_outputs(
    ctx: &'static WgpuContext,
    b: &Bench,
    vs: &[Variant],
    kb: usize,
    slots: usize,
    x_per_slot: bool,
) -> Vec<(Vec<u32>, Vec<u32>)> {
    let n_packed = slots * kb * 2;
    let n_scales = slots * kb / 4;
    let mut out = Vec::with_capacity(vs.len());
    for v in vs {
        let (pl, bg, grid, _p, _pp) = b.build(ctx, v, kb, slots, x_per_slot);
        ctx.queue
            .write_buffer(&b.packed, 0, &vec![0x5au8; n_packed * 4]);
        ctx.queue
            .write_buffer(&b.scales, 0, &vec![0xabu8; n_scales * 4]);
        burst(ctx, &pl, &bg, grid, 1);
        out.push((
            dispatch::read_back::<u32>(ctx, &b.packed, n_packed).unwrap(),
            dispatch::read_back::<u32>(ctx, &b.scales, n_scales).unwrap(),
        ));
    }
    out
}

#[test]
fn lane_split_quantize_matches_the_shipped_quantize_bit_for_bit() {
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    let vs = variants();
    let seeds = env_usize("NV_QWEN36_QUANT_SEEDS", 32);
    assert!(seeds > 0, "a zero-fixture sweep is not a gate");
    let cells = [(128, 9), (32, 9), (128, 1), (256, 1), (8, 9), (4, 3)];
    let mut compared = 0usize;
    for s in 0..seeds {
        let b = Bench::new(
            ctx,
            256,
            9,
            0x243f_6a88_8530_8d31 ^ (s as u64).wrapping_mul(0x9e37_79b9),
        );
        for (kb, slots) in cells {
            let got = quant_outputs(ctx, &b, &vs, kb, slots, s % 2 == 0);
            for (i, v) in vs.iter().enumerate() {
                let (p, sc) = &got[i];
                assert!(
                    p.iter().filter(|w| **w != 0).count() * 4 > p.len(),
                    "{} at kb={kb} slots={slots} seed={s}: only {} of {} packed words are \
                     non-zero -- the fixture underflowed nvfp4 and this gate is blind",
                    v.entry,
                    p.iter().filter(|w| **w != 0).count(),
                    p.len()
                );
                assert!(
                    sc.iter().collect::<std::collections::HashSet<_>>().len() > 2,
                    "{} at kb={kb} slots={slots} seed={s}: scale words take {} distinct values \
                     -- degenerate",
                    v.entry,
                    sc.iter().collect::<std::collections::HashSet<_>>().len()
                );
                if v.entry == v.ref_of {
                    continue;
                }
                let r = vs
                    .iter()
                    .position(|w| w.entry == v.ref_of)
                    .expect("reference arm");
                let bad = p
                    .iter()
                    .zip(got[r].0.iter())
                    .position(|(a, c)| a != c)
                    .map(|w| format!("packed word {w}: {:#010x} vs {:#010x}", p[w], got[r].0[w]))
                    .or_else(|| {
                        sc.iter()
                            .zip(got[r].1.iter())
                            .position(|(a, c)| a != c)
                            .map(|w| {
                                format!("scale word {w}: {:#010x} vs {:#010x}", sc[w], got[r].1[w])
                            })
                    });
                assert!(
                    bad.is_none(),
                    "{} differs from {} at kb={kb} slots={slots} seed={s}: {}",
                    v.entry,
                    v.ref_of,
                    bad.unwrap()
                );
                compared += 1;
            }
        }
    }
    eprintln!(
        "lane-split quantize: {compared} arm-shape-seed comparisons bit-identical over \
         {seeds} fixtures x {} shapes",
        cells.len()
    );
    assert!(
        compared >= 4 * cells.len(),
        "only {compared} comparisons ran; the sweep collapsed"
    );
}

#[test]
#[ignore = "needs a GPU; set NV_QWEN36_QUANT_RATE=1"]
fn quant_pass_cost_per_dispatch() {
    assert_eq!(
        std::env::var("NV_QWEN36_QUANT_RATE").ok().as_deref(),
        Some("1"),
        "set NV_QWEN36_QUANT_RATE=1 -- a silent skip here would report a pass"
    );
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    eprintln!("adapter: {}", ctx.summary());
    if let Ok(o) = std::process::Command::new("uptime").output() {
        eprintln!("host: {}", String::from_utf8_lossy(&o.stdout).trim());
    }

    let max_kb = 128usize;
    let max_slots = 72usize;
    let max_k = max_kb * NVFP4_BLOCK;

    let mut rng = Rng(0x243f6a8885a308d3);
    let gu: Vec<f32> = (0..max_slots * 2 * max_k).map(|_| rng.logval()).collect();
    let xrow: Vec<f32> = (0..max_k).map(|_| rng.logval()).collect();
    let globals: Vec<f32> = (0..257)
        .map(|i| 2f32.powi((i as i32 % 9) - 4) * (1.0 + (i % 7) as f32 / 8.0))
        .collect();
    let sel: Vec<u32> = (0..max_slots).map(|i| ((i * 37) % 257) as u32).collect();

    let x_buf = dispatch::storage_from_slice(ctx, "q-x", &pack_bf16(&xrow));
    let gu_buf = dispatch::storage_from_slice(ctx, "q-gu", &pack_bf16(&gu));
    let sel_buf = dispatch::storage_from_slice(ctx, "q-sel", &sel);
    let glob_buf = dispatch::storage_from_slice(ctx, "q-glob", &globals);
    let packed = dispatch::storage_zeroed(ctx, "q-packed", (max_slots * max_kb * 2 * 4) as u64);
    let scales = dispatch::storage_zeroed(ctx, "q-scales", (max_slots * max_kb / 4 * 4) as u64);

    let src = q3w::nvfp4_quant_source();
    let vs = variants();
    let reps = env_usize("NV_QWEN36_QUANT_REPS", 24);
    let (lo, hi) = (8usize, 64usize);

    let cells: Vec<(usize, usize)> = vec![
        (128, 9),
        (32, 9),
        (128, 1),
        (32, 1),
        (128, 36),
        (32, 36),
        (128, 72),
        (32, 72),
    ];

    eprint!(
        "\n==== nvfp4 quantize passes, us per dispatch (slope of {lo} vs {hi} copies in one \
         pass, median of {reps} interleaved reps) ====\n{:<14}{:>8}",
        "kb / slots", "wgs"
    );
    for v in &vs {
        eprint!(" {:>22}", v.entry.trim_start_matches("q3w_"));
    }
    eprintln!(" {:>8}", "null us");

    let mut served: Vec<(usize, usize, Vec<f64>)> = Vec::new();
    for &(kb, slots) in &cells {
        let k = kb * NVFP4_BLOCK;
        let built: Vec<_> = vs
            .iter()
            .map(|v| {
                let is_silu = v.entry.contains("silu");
                let p = dispatch::uniform_from(
                    ctx,
                    "q-p",
                    &q3w::QuantRowsParams {
                        k_blocks: kb as u32,
                        n_slots: slots as u32,
                        use_sel: 1,
                        x_slot_stride_elems: if is_silu { 2 * k as u32 } else { 0 },
                    },
                );
                let pp = dispatch::uniform_from(
                    ctx,
                    "q-pp",
                    &q3w::SiluPairParams {
                        u_off_elems: if is_silu { k as u32 } else { 0 },
                        ..Default::default()
                    },
                );
                let pl = dispatch::cached_compute_pipeline(ctx, v.entry, &src, v.entry)
                    .unwrap_or_else(|e| panic!("{}: {e}", v.entry));
                let binds: Vec<(u32, &wgpu::Buffer)> = if is_silu {
                    vec![
                        (11, &p),
                        (12, &packed),
                        (13, &scales),
                        (14, &sel_buf),
                        (15, &glob_buf),
                        (16, &gu_buf),
                        (17, &gu_buf),
                        (18, &pp),
                    ]
                } else {
                    vec![
                        (10, &x_buf),
                        (11, &p),
                        (12, &packed),
                        (13, &scales),
                        (14, &sel_buf),
                        (15, &glob_buf),
                    ]
                };
                let bg = dispatch::bind_group(ctx, &pl, &binds);
                let gx = (kb.div_ceil(v.wg_blocks)).max(1) as u32;
                (v, pl, bg, (gx, slots as u32, 1u32), p, pp)
            })
            .collect();

        let n_packed = slots * kb * 2;
        let n_scales = slots * kb / 4;
        let mut refs: std::collections::HashMap<&str, (Vec<u32>, Vec<u32>)> =
            std::collections::HashMap::new();
        for (v, pl, bg, grid, _, _) in &built {
            ctx.queue
                .write_buffer(&packed, 0, &vec![0x5au8; n_packed * 4]);
            ctx.queue
                .write_buffer(&scales, 0, &vec![0xabu8; n_scales * 4]);
            burst(ctx, pl, bg, *grid, 1);
            let got = (
                dispatch::read_back::<u32>(ctx, &packed, n_packed).unwrap(),
                dispatch::read_back::<u32>(ctx, &scales, n_scales).unwrap(),
            );
            assert!(
                got.0.iter().filter(|w| **w != 0).count() * 4 > n_packed,
                "{} at kb={kb} slots={slots}: packed nibbles are degenerate ({} of {n_packed} \
                 words non-zero) -- the fixture underflowed the format and this gate is blind",
                v.entry,
                got.0.iter().filter(|w| **w != 0).count()
            );
            assert!(
                got.1.iter().collect::<std::collections::HashSet<_>>().len() > 2,
                "{} at kb={kb} slots={slots}: scale bytes take {} distinct values -- degenerate",
                v.entry,
                got.1.iter().collect::<std::collections::HashSet<_>>().len()
            );
            if v.entry == v.ref_of {
                refs.insert(v.entry, got);
            } else {
                let want = refs
                    .get(v.ref_of)
                    .unwrap_or_else(|| panic!("{} has no reference {}", v.entry, v.ref_of));
                assert_eq!(
                    got.0, want.0,
                    "{} at kb={kb} slots={slots}: packed nibbles differ from {}",
                    v.entry, v.ref_of
                );
                assert_eq!(
                    got.1, want.1,
                    "{} at kb={kb} slots={slots}: scale bytes differ from {}",
                    v.entry, v.ref_of
                );
            }
        }

        for (_, pl, bg, grid, _, _) in &built {
            for _ in 0..3 {
                burst(ctx, pl, bg, *grid, hi);
            }
        }
        let mut d: Vec<Vec<f64>> = vec![Vec::new(); built.len()];
        let mut z: Vec<Vec<f64>> = vec![Vec::new(); built.len()];
        for _ in 0..reps {
            for (i, (_, pl, bg, grid, _, _)) in built.iter().enumerate() {
                let a = burst(ctx, pl, bg, *grid, lo);
                let b = burst(ctx, pl, bg, *grid, hi);
                let a2 = burst(ctx, pl, bg, *grid, lo);
                d[i].push(1e3 * (b - (a + a2) / 2.0) / (hi - lo) as f64);
                z[i].push(1e3 * (a - a2).abs() / (hi - lo) as f64);
            }
        }
        let arms: Vec<Arm> = built
            .iter()
            .enumerate()
            .map(|(i, (v, _, _, _, _, _))| Arm {
                entry: v.entry,
                us: med(&mut d[i]),
                null_us: med(&mut z[i]),
            })
            .collect();
        let null = arms.iter().map(|a| a.null_us).fold(0.0, f64::max);
        eprint!(
            "{:<14}{:>8}",
            format!("{kb} / {slots}"),
            built
                .iter()
                .map(|(_, _, _, g, _, _)| g.0 * g.1)
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join("/")
        );
        for a in &arms {
            eprint!(" {:>22.2}", a.us);
        }
        eprintln!(
            " {:>8.2}{}",
            null,
            if arms.iter().all(|a| a.us > 2.0 * a.null_us) {
                ""
            } else {
                "   UNRESOLVED -- do not quote"
            }
        );
        for a in &arms {
            assert!(
                a.us > 2.0 * a.null_us,
                "{} at kb={kb} slots={slots} is not resolved against its own null \
                 ({:.2} vs {:.2} us) -- the box moved inside a rep; re-run rather than quoting it",
                a.entry,
                a.us,
                a.null_us
            );
        }
        served.push((kb, slots, arms.iter().map(|a| a.us).collect()));
    }

    let at = |kb: usize, slots: usize| -> &Vec<f64> {
        &served
            .iter()
            .find(|(a, b, _)| *a == kb && *b == slots)
            .unwrap_or_else(|| panic!("cell {kb}/{slots}"))
            .2
    };
    eprintln!(
        "\nslot scaling at fixed per-thread depth (1 -> 9 -> 36 -> 72 slots is 1 -> 9 -> 36 -> 72\n\
         workgroups for the shipped entries): a pass whose cost is the DEPTH of one thread is\n\
         flat here, and one that is throughput-bound is linear."
    );
    for (i, v) in vs.iter().enumerate() {
        for kb in [128usize, 32] {
            eprintln!(
                "  {:<26} kb={kb:<4} {:>7.2} {:>7.2} {:>7.2} {:>7.2} us  ({:.2}x from 1 to 72 slots)",
                v.entry,
                at(kb, 1)[i],
                at(kb, 9)[i],
                at(kb, 36)[i],
                at(kb, 72)[i],
                at(kb, 72)[i] / at(kb, 1)[i]
            );
        }
    }

    eprintln!("\nserved cells, 40 dispatches per token each:");
    let mut best_gain = 0.0f64;
    for (kb, label, ship) in [
        (128usize, "xquant", "q3w_quant_rows"),
        (32, "siluq", "q3w_silu_mul_quant"),
    ] {
        let row = at(kb, 9);
        let bi = vs
            .iter()
            .position(|v| v.entry == ship)
            .expect("shipped arm");
        let base = row[bi];
        let mut pass_best = base;
        eprintln!(
            "  {label} kb={kb} slots=9: shipped {base:.2} us = {:.3} ms/token",
            40.0 * base / 1e3
        );
        for (i, v) in vs.iter().enumerate() {
            if v.entry == v.ref_of || v.ref_of != ship {
                continue;
            }
            eprintln!(
                "    {:<26} {:>7.2} us  {:>6.3}x  {:>+7.3} ms/token",
                v.entry,
                row[i],
                row[i] / base,
                40.0 * (row[i] - base) / 1e3
            );
            pass_best = pass_best.min(row[i]);
        }
        best_gain += (base - pass_best).max(0.0);
    }
    eprintln!(
        "\nbest lane-split arm on each served pass is worth {:.3} ms/token together, against a\n\
         16.806 ms token. This is a PER-DISPATCH slope; it becomes a wall-clock claim only after\n\
         an A/B/A on the whole graph clears its own drift.",
        40.0 * best_gain / 1e3
    );
}
