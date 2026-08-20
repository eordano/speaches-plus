#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RecParams {
    heads: u32,
    d_k: u32,
    d_v: u32,
    pad0: u32,
}

#[derive(Clone, Copy)]
struct Shape {
    heads: usize,
    d_k: usize,
    d_v: usize,
}

#[derive(Clone, Copy)]
struct Arm {
    label: &'static str,
    entry: &'static str,
    lanes_per_wg: usize,
}

const ARMS: [Arm; 10] = [
    Arm {
        label: "serial wg128",
        entry: "q3w_delta_recurrent",
        lanes_per_wg: 128,
    },
    Arm {
        label: "u4     wg128",
        entry: "q3w_delta_recurrent_u4",
        lanes_per_wg: 128,
    },
    Arm {
        label: "serial wg32 ",
        entry: "q3w_delta_recurrent_l32",
        lanes_per_wg: 32,
    },
    Arm {
        label: "u4     wg32 ",
        entry: "q3w_delta_recurrent_u4l32",
        lanes_per_wg: 32,
    },
    Arm {
        label: "u8     wg128",
        entry: "q3w_delta_recurrent_u8",
        lanes_per_wg: 128,
    },
    Arm {
        label: "u8     wg32 ",
        entry: "q3w_delta_recurrent_u8l32",
        lanes_per_wg: 32,
    },
    Arm {
        label: "u16    wg128",
        entry: "q3w_delta_recurrent_u16",
        lanes_per_wg: 128,
    },
    Arm {
        label: "u16    wg32 ",
        entry: "q3w_delta_recurrent_u16l32",
        lanes_per_wg: 32,
    },
    Arm {
        label: "u32    wg128",
        entry: "q3w_delta_recurrent_u32",
        lanes_per_wg: 128,
    },
    Arm {
        label: "u32    wg32 ",
        entry: "q3w_delta_recurrent_u32l32",
        lanes_per_wg: 32,
    },
];

fn grid(a: &Arm, s: &Shape) -> (u32, u32, u32) {
    (s.heads as u32, (s.d_v.div_ceil(a.lanes_per_wg)) as u32, 1)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| (self.next() >> 40) as f32 / 8388608.0 - 1.0)
            .collect()
    }
}

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect("no wgpu adapter -- this suite measures nothing without one")
}

struct Bufs {
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    g: wgpu::Buffer,
    beta: wgpu::Buffer,
    out: wgpu::Buffer,
    p: wgpu::Buffer,
}

fn bufs(
    ctx: &WgpuContext,
    s: &Shape,
    r: &mut Rng,
    g_const: Option<f32>,
    b_const: Option<f32>,
) -> Bufs {
    let g: Vec<f32> = match g_const {
        Some(c) => vec![c; s.heads],
        None => r.unit(s.heads).iter().map(|t| 0.5 + 0.25 * t).collect(),
    };
    let beta: Vec<f32> = match b_const {
        Some(c) => vec![c; s.heads],
        None => r.unit(s.heads).iter().map(|t| 0.5 + 0.5 * t).collect(),
    };
    Bufs {
        q: dispatch::storage_from_slice(ctx, "dr-q", &r.unit(s.heads * s.d_k)),
        k: dispatch::storage_from_slice(ctx, "dr-k", &r.unit(s.heads * s.d_k)),
        v: dispatch::storage_from_slice(ctx, "dr-v", &r.unit(s.heads * s.d_v)),
        g: dispatch::storage_from_slice(ctx, "dr-g", &g),
        beta: dispatch::storage_from_slice(ctx, "dr-beta", &beta),
        out: dispatch::storage_zeroed(ctx, "dr-out", (s.heads * s.d_v * 4) as u64),
        p: dispatch::uniform_from(
            ctx,
            "dr-p",
            &RecParams {
                heads: s.heads as u32,
                d_k: s.d_k as u32,
                d_v: s.d_v as u32,
                pad0: 0,
            },
        ),
    }
}

fn bind(
    ctx: &WgpuContext,
    pl: &wgpu::ComputePipeline,
    b: &Bufs,
    state: &wgpu::Buffer,
) -> wgpu::BindGroup {
    dispatch::bind_group(
        ctx,
        pl,
        &[
            (30, &b.q),
            (31, &b.k),
            (32, &b.v),
            (33, &b.g),
            (34, &b.beta),
            (35, &b.out),
            (36, state),
            (37, &b.p),
        ],
    )
}

#[test]
fn lane_split_delta_recurrent_is_bit_identical_to_serial() {
    let ctx = ctx();
    let src = nv_models::qwen3_5_moe_wgpu::delta_source();

    let plans: [(&str, Shape, Option<f32>, Option<f32>); 5] = [
        (
            "served 32x128x128",
            Shape {
                heads: 32,
                d_k: 128,
                d_v: 128,
            },
            None,
            None,
        ),
        (
            "tiny 4x16x16",
            Shape {
                heads: 4,
                d_k: 16,
                d_v: 16,
            },
            None,
            None,
        ),
        (
            "ragged 3x6x40",
            Shape {
                heads: 3,
                d_k: 6,
                d_v: 40,
            },
            None,
            None,
        ),
        (
            "g=0 8x32x32",
            Shape {
                heads: 8,
                d_k: 32,
                d_v: 32,
            },
            Some(0.0),
            None,
        ),
        (
            "g=1,beta=0 8x32x32",
            Shape {
                heads: 8,
                d_k: 32,
                d_v: 32,
            },
            Some(1.0),
            Some(0.0),
        ),
    ];

    let mut failures = Vec::new();
    for (name, s, gov, bov) in plans {
        let state0 = Rng(0x9e3779b97f4a7c15).unit(s.heads * s.d_k * s.d_v);
        let mut results = Vec::new();
        for a in ARMS {
            let mut r = Rng(0x243f6a8885a308d3);
            let b = bufs(ctx, &s, &mut r, gov, bov);
            let st = dispatch::storage_from_slice(ctx, "dr-state", &state0);
            let pl = dispatch::cached_compute_pipeline(ctx, a.entry, &src, a.entry).unwrap();
            let bg = bind(ctx, &pl, &b, &st);
            let gd = grid(&a, &s);
            for _ in 0..3 {
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pl);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(gd.0, gd.1, gd.2);
                }
                ctx.queue.submit([enc.finish()]);
            }
            let out: Vec<f32> = dispatch::read_back(ctx, &b.out, s.heads * s.d_v).unwrap();
            let stv: Vec<f32> = dispatch::read_back(ctx, &st, s.heads * s.d_k * s.d_v).unwrap();
            results.push((a.label, out, stv));
        }

        let (_, ref o_ref, ref s_ref) = results[0];
        let nonzero = o_ref.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero > 0,
            "{name}: the reference kernel produced all zeros, so this case measures nothing"
        );
        for (label, out, stv) in results.iter().skip(1) {
            let obad = o_ref
                .iter()
                .zip(out.iter())
                .position(|(a, b)| a.to_bits() != b.to_bits());
            let sbad = s_ref
                .iter()
                .zip(stv.iter())
                .position(|(a, b)| a.to_bits() != b.to_bits());
            eprintln!(
                "{name:<20} {label}  out {}  state {}  ({nonzero}/{} out words nonzero)",
                if obad.is_none() {
                    "bit-identical"
                } else {
                    "DIFFER"
                },
                if sbad.is_none() {
                    "bit-identical"
                } else {
                    "DIFFER"
                },
                o_ref.len()
            );
            if let Some(i) = obad {
                failures.push(format!(
                    "{name}/{label}: out[{i}] {} vs {}",
                    o_ref[i], out[i]
                ));
            }
            if let Some(i) = sbad {
                failures.push(format!(
                    "{name}/{label}: state[{i}] {} vs {}",
                    s_ref[i], stv[i]
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "lane-split parity failures: {failures:#?}"
    );
}

fn pct(xs: &[f64], p: f64) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[(((s.len() - 1) as f64) * p).round() as usize]
}

fn recurrent_cost_ladder(s: Shape, layers: usize) {
    let ctx = ctx();
    let src = nv_models::qwen3_5_moe_wgpu::delta_source();
    let state_bytes = (s.heads * s.d_k * s.d_v * 4) as u64;
    let iters: usize = std::env::var("NV_DELTA_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    eprintln!("adapter: {:?}", ctx.adapter.get_info().name);
    eprintln!(
        "shape heads={} d_k={} d_v={}  state={} MiB x {} layers = {} MiB/step  iters={}",
        s.heads,
        s.d_k,
        s.d_v,
        state_bytes / (1 << 20),
        layers,
        state_bytes * layers as u64 / (1 << 20),
        iters
    );

    let mut r = Rng(0x243f6a8885a308d3);

    let b = {
        let g: Vec<f32> = (0..s.heads).map(|_| 0.975 + 0.025 * r.unit(1)[0]).collect();
        let beta: Vec<f32> = (0..s.heads).map(|_| 0.05 + 0.05 * r.unit(1)[0]).collect();
        Bufs {
            q: dispatch::storage_from_slice(ctx, "dr-q", &r.unit(s.heads * s.d_k)),
            k: dispatch::storage_from_slice(ctx, "dr-k", &r.unit(s.heads * s.d_k)),
            v: dispatch::storage_from_slice(ctx, "dr-v", &r.unit(s.heads * s.d_v)),
            g: dispatch::storage_from_slice(ctx, "dr-g", &g),
            beta: dispatch::storage_from_slice(ctx, "dr-beta", &beta),
            out: dispatch::storage_zeroed(ctx, "dr-out", (s.heads * s.d_v * 4) as u64),
            p: dispatch::uniform_from(
                ctx,
                "dr-p",
                &RecParams {
                    heads: s.heads as u32,
                    d_k: s.d_k as u32,
                    d_v: s.d_v as u32,
                    pad0: 0,
                },
            ),
        }
    };
    let state0 = r.unit(s.heads * s.d_k * s.d_v);
    let states: Vec<wgpu::Buffer> = (0..layers)
        .map(|_| dispatch::storage_from_slice(ctx, "dr-state", &state0))
        .collect();

    println!();
    println!(
        "{:<14} {:>7} {:>10} {:>10} {:>9} {:>10} {:>10} {:>9}",
        "arm", "wgs", "1x med ms", "8x med ms", "enc ms", "us/disp", "GB/s", "vs base"
    );
    let mut base_us: Option<f64> = None;
    for a in ARMS {
        let pl = dispatch::cached_compute_pipeline(ctx, a.entry, &src, a.entry).unwrap();
        let bgs: Vec<wgpu::BindGroup> = states.iter().map(|st| bind(ctx, &pl, &b, st)).collect();
        let gd = grid(&a, &s);

        let run = |rounds: usize| -> (f64, f64) {
            let mut walls = Vec::with_capacity(iters);
            let mut encs = Vec::with_capacity(iters);
            for it in 0..iters + 5 {
                let t0 = Instant::now();
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pl);
                    for _ in 0..rounds {
                        for bg in &bgs {
                            pass.set_bind_group(0, bg, &[]);
                            pass.dispatch_workgroups(gd.0, gd.1, gd.2);
                        }
                    }
                }
                ctx.queue.submit([enc.finish()]);
                let te = t0.elapsed().as_secs_f64() * 1e3;
                ctx.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
                if it >= 5 {
                    walls.push(t0.elapsed().as_secs_f64() * 1e3);
                    encs.push(te);
                }
            }
            (pct(&walls, 0.5), pct(&encs, 0.5))
        };

        let (t1, e1) = run(1);
        let (t4, e4) = run(8);
        let per_disp_us = ((t4 - e4) - (t1 - e1)) / (7.0 * layers as f64) * 1e3;

        let passes = if a.entry.contains("_u") { 3.0 } else { 4.0 };
        let gbs = state_bytes as f64 * passes / (per_disp_us * 1e-6) / 1e9;
        let rel = match base_us {
            None => {
                base_us = Some(per_disp_us);
                String::from("--")
            }
            Some(b0) => format!("{:.2}x", b0 / per_disp_us),
        };
        println!(
            "{:<14} {:>7} {:>10.3} {:>10.3} {:>9.3} {:>10.2} {:>10.1} {:>9}",
            a.label,
            gd.0 * gd.1,
            t1,
            t4,
            e4,
            per_disp_us,
            gbs,
            rel
        );
    }
    println!();
    println!("per-step cost = us/disp x {layers} linear-attention layers");
}

#[test]
#[ignore]
fn delta_recurrent_lane_split_cost() {
    recurrent_cost_ladder(
        Shape {
            heads: 32,
            d_k: 128,
            d_v: 128,
        },
        30,
    );
}

#[test]
#[ignore]
fn delta_recurrent_cost_at_qwen38_geometry() {
    if std::env::var("NV_QWEN38_DELTA_BENCH").ok().as_deref() != Some("1") {
        eprintln!(
            "delta_recurrent_cost_at_qwen38_geometry: NV_QWEN38_DELTA_BENCH != 1, measuring \
             nothing -- a run without this line's absence produced no numbers"
        );
        return;
    }
    recurrent_cost_ladder(
        Shape {
            heads: 48,
            d_k: 128,
            d_v: 128,
        },
        48,
    );
}

mod graph_arms {
    use nv_models::qwen3_5_moe::{LayerType, Qwen3MoeConfig};
    use nv_models::qwen3_5_moe_wgpu as q3w;

    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32) - 1.0
        }
        fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
            (0..n)
                .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
                .collect()
        }
        fn f32_vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n)
                .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_f32())
                .collect()
        }
        fn norm_vec(&mut self, n: usize) -> Vec<u16> {
            (0..n)
                .map(|_| half::bf16::from_f32(1.0 + 0.1 * self.next_f32()).to_bits())
                .collect()
        }
    }

    pub(super) fn cfg() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            hidden_size: 256,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 32,
            moe_intermediate_size: 64,
            shared_expert_intermediate_size: 64,
            num_experts: 8,
            num_experts_per_tok: 2,
            vocab_size: 64,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            partial_rotary_factor: 0.5,
            bos_token_id: 0,
            eos_token_id: 1,
            layer_types: vec![
                LayerType::LinearAttention,
                LayerType::FullAttention,
                LayerType::LinearAttention,
            ],
            linear_num_key_heads: 4,
            linear_num_value_heads: 8,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            attn_output_gate: true,
            tie_word_embeddings: false,
        }
    }

    pub(super) fn weights(c: &Qwen3MoeConfig) -> q3w::HostWeights {
        weights_seeded(c, 0x51ee_d100_0007)
    }

    pub(super) fn weights_seeded(c: &Qwen3MoeConfig, seed: u64) -> q3w::HostWeights {
        let mut r = Lcg(seed);
        let h = c.hidden_size;
        let inter = c.moe_intermediate_size;
        let sinter = c.shared_expert_intermediate_size;
        let key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
        let value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let bf = |r: &mut Lcg, n: usize, k: usize, s: f32| q3w::HostBf16Lin {
            w: r.bf16_vec(n * k, s),
            n,
            k,
        };
        let nv = |r: &mut Lcg, n: usize, k: usize, s: f32| {
            q3w::quantize_nvfp4_host(&r.bf16_vec(n * k, s), n, k)
        };
        let mut layers = Vec::new();
        for li in 0..c.num_hidden_layers {
            let mixer = match c.layer_types[li] {
                LayerType::LinearAttention => q3w::HostMixer::Delta(Box::new(q3w::HostDeltaNet {
                    in_proj_qkv: bf(&mut r, conv_dim, h, 0.12),
                    in_proj_z: bf(&mut r, value_dim, h, 0.12),
                    in_proj_ab: bf(&mut r, 2 * c.linear_num_value_heads, h, 0.12),
                    conv1d: r.f32_vec(conv_dim * c.linear_conv_kernel_dim, 0.4),
                    a_log: r.f32_vec(c.linear_num_value_heads, 0.5),
                    dt_bias: r.f32_vec(c.linear_num_value_heads, 0.5),
                    norm_w: r.norm_vec(c.linear_value_head_dim),
                    out_proj: bf(&mut r, h, value_dim, 0.12),
                })),
                LayerType::FullAttention => q3w::HostMixer::Attn(Box::new(q3w::HostAttention {
                    q: nv(&mut r, c.num_attention_heads * c.head_dim * 2, h, 0.12),
                    k: nv(&mut r, c.num_key_value_heads * c.head_dim, h, 0.12),
                    v: nv(&mut r, c.num_key_value_heads * c.head_dim, h, 0.12),
                    o: nv(&mut r, h, c.num_attention_heads * c.head_dim, 0.12),
                    q_norm: r.norm_vec(c.head_dim),
                    k_norm: r.norm_vec(c.head_dim),
                })),
            };
            let g: Vec<_> = (0..c.num_experts)
                .map(|_| nv(&mut r, inter, h, 0.15))
                .collect();
            let u: Vec<_> = (0..c.num_experts)
                .map(|_| nv(&mut r, inter, h, 0.15))
                .collect();
            let d: Vec<_> = (0..c.num_experts)
                .map(|_| nv(&mut r, h, inter, 0.15))
                .collect();
            layers.push(q3w::HostLayer {
                input_ln: r.norm_vec(h),
                post_attn_ln: r.norm_vec(h),
                mixer,
                moe: q3w::HostMoe {
                    router: bf(&mut r, c.num_experts, h, 0.3),
                    experts_gate: q3w::stack_nvfp4_host(&g),
                    experts_up: q3w::stack_nvfp4_host(&u),
                    experts_down: q3w::stack_nvfp4_host(&d),
                    shared_gate: nv(&mut r, sinter, h, 0.15),
                    shared_up: nv(&mut r, sinter, h, 0.15),
                    shared_down: nv(&mut r, h, sinter, 0.15),
                    shared_expert_gate: bf(&mut r, 1, h, 0.3),
                },
            });
        }
        q3w::HostWeights {
            embed: r.bf16_vec(c.vocab_size * h, 0.6),
            final_norm: r.norm_vec(h),
            lm_head: r.bf16_vec(c.vocab_size * h, 0.2),
            layers,
        }
    }

    #[test]
    fn whole_graph_decode_is_bit_identical_across_delta_unroll_arms() {
        let c = cfg();
        let hw = weights(&c);
        let tokens: [u32; 8] = [3, 11, 5, 40, 2, 19, 7, 33];
        let arms = ["0", "4", "8", "16", "32", "l32", "4l32", "8l32", "16l32", "32l32"];

        let mut reference: Option<Vec<Vec<f32>>> = None;
        let mut failures = Vec::new();
        for arm in arms {
            std::env::set_var("NV_Q3_WGPU_DELTA_UNROLL", arm);
            let entry = q3w::delta_recurrent_kernel();
            let mut m = q3w::Qwen3MoeWgpu::new(c.clone(), &hw, 32).expect("build the decode graph");
            let mut got = Vec::new();
            for t in tokens {
                let (_, logits) = m.decode_step_logits(t).expect("decode step");
                got.push(logits);
            }
            let nonzero = got[0].iter().filter(|v| **v != 0.0).count();
            match &reference {
                None => {
                    assert!(
                        nonzero > 0,
                        "the baseline graph produced all-zero logits, so this test measures nothing"
                    );
                    eprintln!(
                        "arm {arm:<6} {:<26} baseline, {}/{} logit words nonzero, {} layers unrolled",
                        entry.0,
                        nonzero,
                        got[0].len(),
                        m.delta_unrolled_layers().0
                    );
                    reference = Some(got);
                }
                Some(refv) => {
                    let bad = refv
                        .iter()
                        .zip(got.iter())
                        .enumerate()
                        .find_map(|(s, (a, b))| {
                            a.iter()
                                .zip(b.iter())
                                .position(|(x, y)| x.to_bits() != y.to_bits())
                                .map(|i| (s, i, a[i], b[i]))
                        });
                    eprintln!(
                        "arm {arm:<6} {:<26} {} ({} layers unrolled)",
                        entry.0,
                        if bad.is_none() {
                            "bit-identical"
                        } else {
                            "DIFFER"
                        },
                        m.delta_unrolled_layers().0
                    );
                    if let Some((s, i, x, y)) = bad {
                        failures.push(format!("{arm}: step {s} logit[{i}] {x} vs {y}"));
                    }
                }
            }
        }
        std::env::remove_var("NV_Q3_WGPU_DELTA_UNROLL");
        assert!(
            failures.is_empty(),
            "graph-level parity failures: {failures:#?}"
        );
    }
}

mod census {
    use nv_models::qwen3_5_moe::LayerType;
    use std::collections::BTreeMap;

    #[test]
    fn dispatch_census_at_the_served_layer_count() {
        let mut c = super::graph_arms::cfg();
        c.num_hidden_layers = 40;
        c.layer_types = (0..40)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();
        c.num_experts_per_tok = 8;
        c.num_experts = 8;
        let hw = super::graph_arms::weights(&c);
        let m = nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu::new(c.clone(), &hw, 32)
            .expect("build the decode graph");

        let labels = m.pass_labels();
        let head = m.head_pass_start();
        let mut per: BTreeMap<&str, usize> = BTreeMap::new();
        for l in &labels[..head] {
            *per.entry(l.split(':').next().unwrap()).or_default() += 1;
        }
        let mut tail: BTreeMap<&str, usize> = BTreeMap::new();
        for l in &labels[head..] {
            *tail.entry(l.split(':').next().unwrap()).or_default() += 1;
        }

        let linear = c
            .layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::LinearAttention))
            .count();
        let full = c.num_hidden_layers - linear;
        println!();
        println!(
            "{} dispatches/token over {} layers ({linear} linear-attention, {full} full-attention), \
             {} in the body and {} in the head",
            labels.len(),
            c.num_hidden_layers,
            head,
            labels.len() - head
        );
        println!();
        let mut rows: Vec<(&str, usize)> = per.into_iter().collect();
        rows.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        println!(
            "{:<26} {:>6} {:>10} {:>9}",
            "label", "count", "per layer", "share"
        );
        for (l, n) in &rows {
            let denom = if l.contains("-dn-") {
                linear
            } else if l.contains("-at-") {
                full
            } else {
                c.num_hidden_layers
            };
            println!(
                "{l:<26} {n:>6} {:>10.2} {:>8.1}%",
                *n as f64 / denom as f64,
                100.0 * *n as f64 / labels.len() as f64
            );
        }
        println!();
        for (l, n) in tail {
            println!("head  {l:<20} {n:>6}");
        }
        assert!(labels.len() > 400, "census did not build the whole graph");
    }
}

mod chain {

    #[test]
    fn decode_chain_matches_the_per_step_token_sequence() {
        let mut c = super::graph_arms::cfg();
        c.vocab_size = c.hidden_size;
        let prompt: u32 = 3;
        let n = 12usize;

        let selector_head = |c: &nv_models::qwen3_5_moe::Qwen3MoeConfig, seed: u64| {
            let mut w = super::graph_arms::weights_seeded(c, seed);
            let h = c.hidden_size;
            for v in 0..c.vocab_size {
                for j in 0..h {
                    let base = half::bf16::from_bits(w.lm_head[v * h + j]).to_f32() * 0.05;
                    let hit = if j == v % h { 1.0 } else { 0.0 };
                    w.lm_head[v * h + j] = half::bf16::from_f32(base + hit).to_bits();
                }
            }
            w
        };

        let mut chosen = None;
        for seed in [
            0x51ee_d100_0007u64,
            0xa5a5_1234_0001,
            0x0bad_c0de_0002,
            0xfeed_face_0003,
        ] {
            let hw = selector_head(&c, seed);
            let mut a = nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu::new(c.clone(), &hw, 32)
                .expect("build the decode graph");
            let mut serial = Vec::with_capacity(n);
            let mut t = prompt;
            for _ in 0..n {
                t = a.decode_step(t).expect("decode step");
                serial.push(t);
            }
            let distinct: std::collections::BTreeSet<u32> = serial.iter().copied().collect();
            eprintln!("seed {seed:#x}: {} distinct tokens in {n}", distinct.len());
            if distinct.len() > 1 {
                chosen = Some((hw, a, serial));
                break;
            }
        }
        let (hw, a, serial) =
            chosen.expect("no seed produced a moving token stream; the test would be vacuous");

        let mut failures = Vec::new();
        for k in [2usize, 3, 4, 6, 12] {
            let mut m = nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu::new(c.clone(), &hw, 32)
                .expect("build the decode graph");
            let mut got = Vec::with_capacity(n);
            let mut feed = prompt;
            while got.len() < n {
                let chunk = m.decode_chain(feed, k).expect("decode chain");
                assert_eq!(
                    chunk.len(),
                    k,
                    "chain returned {} tokens, want {k}",
                    chunk.len()
                );
                feed = *chunk.last().expect("non-empty chunk");
                got.extend(chunk);
            }
            let same = got == serial;
            eprintln!(
                "k={k:<3} {} pos={} first8={:?}",
                if same {
                    "identical to per-step"
                } else {
                    "DIFFERS"
                },
                m.current_pos(),
                &got[..8]
            );
            if !same {
                let i = got
                    .iter()
                    .zip(serial.iter())
                    .position(|(x, y)| x != y)
                    .unwrap_or(0);
                failures.push(format!("k={k}: token {i} was {} not {}", got[i], serial[i]));
            }
            if m.current_pos() != a.current_pos() {
                failures.push(format!(
                    "k={k}: pos {} not {}",
                    m.current_pos(),
                    a.current_pos()
                ));
            }
        }
        eprintln!("per-step reference: {serial:?}");
        assert!(failures.is_empty(), "chain parity failures: {failures:#?}");
    }
}

mod chain_cost {
    use nv_models::qwen3_5_moe::LayerType;
    use std::time::Instant;

    fn pct(xs: &[f64], p: f64) -> f64 {
        let mut s = xs.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[(((s.len() - 1) as f64) * p).round() as usize]
    }

    #[test]
    #[ignore]
    fn chain_removes_the_per_token_drain() {
        let mut c = super::graph_arms::cfg();
        c.num_hidden_layers = 40;
        c.layer_types = (0..40)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();
        c.num_experts_per_tok = 8;
        c.num_experts = 8;
        let hw = super::graph_arms::weights(&c);
        let steps: usize = std::env::var("NV_CHAIN_BENCH_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(240);
        let mut m = nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu::new(c.clone(), &hw, steps + 8)
            .expect("build the decode graph");
        println!();
        println!(
            "{} dispatches/token, {} steps per arm",
            m.pass_count(),
            steps
        );
        println!("{:>4} {:>12} {:>12} {:>12}", "k", "ms/token", "p10", "p90");

        let mut rows = Vec::new();
        for k in [1usize, 2, 4, 8] {
            m.reset().expect("reset");
            let mut per = Vec::new();
            let mut feed = 3u32;

            for _ in 0..2 {
                let out = m.decode_chain(feed, k).expect("chain");
                feed = *out.last().unwrap();
            }
            m.reset().expect("reset");
            let mut done = 0usize;
            while done < steps {
                let t0 = Instant::now();
                let out = m.decode_chain(feed, k).expect("chain");
                per.push(t0.elapsed().as_secs_f64() * 1e3 / k as f64);
                feed = *out.last().unwrap();
                done += k;
            }
            let med = pct(&per, 0.5);
            println!(
                "{k:>4} {med:>12.4} {:>12.4} {:>12.4}",
                pct(&per, 0.1),
                pct(&per, 0.9)
            );
            rows.push((k as f64, med));
        }

        let n = rows.len() as f64;
        let sx: f64 = rows.iter().map(|(k, _)| 1.0 / k).sum();
        let sy: f64 = rows.iter().map(|(_, y)| *y).sum();
        let sxx: f64 = rows.iter().map(|(k, _)| 1.0 / (k * k)).sum();
        let sxy: f64 = rows.iter().map(|(k, y)| y / k).sum();
        let h = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let g = (sy - h * sx) / n;
        println!();
        println!("decode(k) = G + H/k   G = {g:.4} ms   H = {h:.4} ms/token");
        for (k, y) in &rows {
            println!("  k={k:<4} measured {y:.4}  fit {:.4}", g + h / k);
        }
        assert!(h.is_finite());
    }
}
