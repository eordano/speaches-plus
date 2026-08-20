#![cfg(feature = "wgpu")]

mod common;
use common::FdParams;
use nv_kernels::wgpu_backend::kernels::{
    flash_decode, fused_norm_chain, kv_fp8, rmsnorm, rmsnorm_residual,
};
use nv_kernels::wgpu_backend::{compose, dispatch, WgpuContext};
use common::lcg_hi32_u32 as lcg;

fn build_pipeline_entry(
    ctx: &WgpuContext,
    src: &str,
    entry: &str,
    zero_init: bool,
) -> wgpu::ComputePipeline {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flash-ab"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
    ctx.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flash-ab"),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[],
                zero_initialize_workgroup_memory: zero_init,
            },
            cache: None,
        })
}

fn build_pipeline(ctx: &WgpuContext, src: &str, zero_init: bool) -> wgpu::ComputePipeline {
    build_pipeline_entry(ctx, src, flash_decode::ENTRY_STAGE1_FP8, zero_init)
}

const POISON_WGSL: &str = "
@compute @workgroup_size(256)
fn fd_poison_wg(@builtin(local_invocation_id) tid: vec3<u32>) {
    let p = bitcast<f32>(0x7fc0deadu | fd_out[0]);
    for (var i = tid.x; i < 512u; i = i + 256u) {
        fd_qsh[i] = p;
    }
    fd_red[tid.x] = p;
    if (tid.x < 8u) {
        fd_sm[tid.x] = p;
        fd_sl[tid.x] = p;
    }
    for (var i = tid.x; i < 4096u; i = i + 256u) {
        fd_sacc[i] = p;
    }
    workgroupBarrier();
    if (tid.x == 0u) {
        var n = 0u;
        for (var i = 0u; i < 4096u; i = i + 1u) {
            if (bitcast<u32>(fd_sacc[i]) == bitcast<u32>(p)) {
                n = n + 1u;
            }
        }
        fd_out[1] = n;
    }
}

@compute @workgroup_size(256)
fn fd_leak_probe(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let base = wg.x * 8192u;
    for (var i = tid.x; i < 4096u; i = i + 256u) {
        fd_scratch[base + i] = fd_sacc[i];
    }
    for (var i = tid.x; i < 512u; i = i + 256u) {
        fd_scratch[base + 4096u + i] = fd_qsh[i];
    }
    fd_scratch[base + 4608u + tid.x] = fd_red[tid.x];
    if (tid.x < 8u) {
        fd_scratch[base + 4864u + tid.x] = fd_sm[tid.x];
        fd_scratch[base + 4872u + tid.x] = fd_sl[tid.x];
    }
}
";

#[test]
fn uninitialized_workgroup_reads_see_zero_on_this_adapter() {
    let ctx = WgpuContext::shared()
        .unwrap_or_else(|e| panic!("no wgpu adapter, this proof cannot pass: {e}"));
    let src = format!("{}\n{}", compose(flash_decode::WGSL), POISON_WGSL);
    let pl_poison = build_pipeline_entry(ctx, &src, "fd_poison_wg", false);
    let pl_probe_nozi = build_pipeline_entry(ctx, &src, "fd_leak_probe", false);
    let pl_probe_zi = build_pipeline_entry(ctx, &src, "fd_leak_probe", true);
    let out_buf = dispatch::storage_zeroed(ctx, "lp-out", 8);
    let scratch_len = 8usize * 8192;
    let scratch = dispatch::storage_zeroed(ctx, "lp-scr", (scratch_len * 4) as u64);
    let poison_bind = dispatch::bind_group(ctx, &pl_poison, &[(3, &out_buf)]);

    let run = |probe: &wgpu::ComputePipeline| -> (u32, usize) {
        let bind = dispatch::bind_group(ctx, probe, &[(7, &scratch)]);
        ctx.queue
            .write_buffer(&scratch, 0, &vec![0u8; scratch_len * 4]);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pl_poison);
            pass.set_bind_group(0, &poison_bind, &[]);
            pass.dispatch_workgroups(8, 16, 1);
            pass.set_pipeline(probe);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(8, 1, 1);
        }
        ctx.queue.submit([enc.finish()]);
        let self_check: Vec<u32> = dispatch::read_back(ctx, &out_buf, 2).unwrap();
        let words: Vec<f32> = dispatch::read_back(ctx, &scratch, scratch_len).unwrap();
        let nonzero = words.iter().filter(|w| w.to_bits() != 0).count();
        (self_check[1], nonzero)
    };

    let (poisoned_zi, stale_zi) = run(&pl_probe_zi);
    let (poisoned_nozi, stale_nozi) = run(&pl_probe_nozi);
    eprintln!(
        "poison self-check {poisoned_zi}/{poisoned_nozi} of 4096; stale words seen by next dispatch: zi {stale_zi}, nozi {stale_nozi} of {scratch_len}"
    );
    assert_eq!(
        poisoned_zi, 4096,
        "poison kernel did not stick within its own dispatch"
    );
    assert_eq!(
        poisoned_nozi, 4096,
        "poison kernel did not stick within its own dispatch"
    );
    assert_eq!(stale_zi, 0, "zero-init failed to clear workgroup memory");

    assert!(
        stale_nozi > 0 || stale_zi == 0,
        "impossible: nozi saw no leak while zi did"
    );
    eprintln!(
        "[nozi] this adapter leaks {stale_nozi}/{scratch_len} workgroup words across dispatches \
         ({:.0}%). Every NOZI_AUDITED_ENTRIES member needs a poison-parity test; zero-equivalence \
         is not a safety property here.",
        100.0 * stale_nozi as f64 / scratch_len as f64
    );
}

#[test]
fn flash_stage1_nozi_bitwise_parity_poisoned() {
    let ctx = WgpuContext::shared()
        .unwrap_or_else(|e| panic!("no wgpu adapter, this proof cannot pass: {e}"));
    let n_heads = 8usize;
    let nkv = 2usize;
    let hd = 256usize;
    let splits = 16u32;
    let max_seq = 1024usize;

    let src = format!("{}\n{}", compose(flash_decode::WGSL), POISON_WGSL);
    let pl_poison = build_pipeline_entry(ctx, &src, "fd_poison_wg", false);

    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut step = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 32) as u32
    };
    let q: Vec<f32> = (0..n_heads * hd)
        .map(|_| (step() as f32 / u32::MAX as f32) - 0.5)
        .collect();
    let kv_words: Vec<u32> = (0..max_seq * nkv * hd / 4)
        .map(|_| step() & 0x3f3f3f3f)
        .collect();
    let scales: Vec<f32> = (0..max_seq * nkv)
        .map(|_| (step() as f32 / u32::MAX as f32) * 0.5 + 0.5)
        .collect();

    let q_buf = dispatch::storage_from_slice(ctx, "pp-q", &q);
    let k_buf = dispatch::storage_from_slice(ctx, "pp-k", &kv_words);
    let v_buf = dispatch::storage_from_slice(ctx, "pp-v", &kv_words);
    let ks_buf = dispatch::storage_from_slice(ctx, "pp-ks", &scales);
    let vs_buf = dispatch::storage_from_slice(ctx, "pp-vs", &scales);
    let out_buf = dispatch::storage_zeroed(ctx, "pp-out", 8);
    let scratch_len = n_heads * splits as usize * (hd + 2);
    let scratch = dispatch::storage_zeroed(ctx, "pp-scr", (scratch_len * 4) as u64);
    let poison_bind = dispatch::bind_group(ctx, &pl_poison, &[(3, &out_buf)]);

    let run = |pl: &wgpu::ComputePipeline,
               fp8: bool,
               total: u32,
               start: u32,
               fill: u8,
               poison: bool|
     -> Vec<f32> {
        let fd = FdParams {
            n_heads: n_heads as u32,
            n_kv: nkv as u32,
            head_dim: hd as u32,
            total,
            start,
            splits,
            ring: 0,
            out_bf16: 1,
            scaling: 1.0 / (hd as f32).sqrt(),
            m_rows: 1,
            ..Default::default()
        };
        let fd_buf = dispatch::uniform_from(ctx, "pp-fd", &fd);
        let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![
            (0, &q_buf),
            (4, &fd_buf),
            (5, &k_buf),
            (6, &v_buf),
            (7, &scratch),
        ];
        if fp8 {
            binds.push((8, &ks_buf));
            binds.push((9, &vs_buf));
        }
        let bind = dispatch::bind_group(ctx, pl, &binds);
        ctx.queue
            .write_buffer(&scratch, 0, &vec![fill; scratch_len * 4]);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            if poison {
                pass.set_pipeline(&pl_poison);
                pass.set_bind_group(0, &poison_bind, &[]);
                pass.dispatch_workgroups(n_heads as u32, splits, 1);
            }
            pass.set_pipeline(pl);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(n_heads as u32, splits, 1);
        }
        ctx.queue.submit([enc.finish()]);
        dispatch::read_back(ctx, &scratch, scratch_len).unwrap()
    };

    let mut cells = 0usize;
    for (entry, fp8) in [
        (flash_decode::ENTRY_STAGE1_FP8, true),
        (flash_decode::ENTRY_STAGE1_BF16, false),
    ] {
        let pl_zi = build_pipeline_entry(ctx, &src, entry, true);
        let pl_nozi = build_pipeline_entry(ctx, &src, entry, false);
        for (total, start) in [
            (1u32, 0u32),
            (5, 0),
            (8, 0),
            (15, 0),
            (17, 0),
            (127, 0),
            (128, 0),
            (129, 0),
            (403, 0),
            (600, 88),
            (1024, 512),
        ] {
            let a = run(&pl_zi, fp8, total, start, 0xaa, false);
            let b = run(&pl_nozi, fp8, total, start, 0xbb, true);
            assert!(
                a.iter().any(|v| *v != 0.0),
                "{entry} total={total}: the zero-init arm wrote nothing, parity is vacuous"
            );
            let mismatches = a
                .iter()
                .zip(b.iter())
                .enumerate()
                .filter(|(_, (x, y))| x.to_bits() != y.to_bits())
                .take(4)
                .map(|(i, (x, y))| format!("[{i}] {:08x} vs {:08x}", x.to_bits(), y.to_bits()))
                .collect::<Vec<_>>();
            assert!(
                mismatches.is_empty(),
                "{entry} total={total} start={start}: nozi+poison diverged from zero-init: \
                 {mismatches:?}"
            );
            cells += 1;
        }
        eprintln!("{entry}: 11 shapes x {scratch_len} scratch words bit-identical under poison");
    }
    assert_eq!(cells, 22, "ran {cells} stage1 cells");
}

const POISON_MK_WGSL: &str = "
@compute @workgroup_size(256)
fn fd_poison_wg_mk(@builtin(local_invocation_id) tid: vec3<u32>) {
    let p = bitcast<f32>(0x7fc0deadu | fd_out[0]);
    for (var i = tid.x; i < 512u; i = i + 256u) {
        fd_qsh[i] = p;
    }
    for (var i = tid.x; i < 2048u; i = i + 256u) {
        fd_qsh_mk[i] = p;
    }
    fd_red[tid.x] = p;
    if (tid.x < 8u) {
        fd_sm[tid.x] = p;
        fd_sl[tid.x] = p;
    }
    for (var i = tid.x; i < 4096u; i = i + 256u) {
        fd_sacc[i] = p;
    }
}
";

#[test]
fn flash_stage1_mk_nozi_bitwise_parity_poisoned() {
    let ctx = WgpuContext::shared()
        .unwrap_or_else(|e| panic!("no wgpu adapter, this proof cannot pass: {e}"));
    let n_heads = 8usize;
    let nkv = 2usize;
    let hd = 256usize;
    let splits = 16u32;
    let max_seq = 1024usize;

    let src = format!("{}\n{}", compose(flash_decode::WGSL), POISON_MK_WGSL);
    let pl_poison = build_pipeline_entry(ctx, &src, "fd_poison_wg_mk", false);

    let mut seed = 0x243f6a8885a308d3u64;
    let mut step = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 32) as u32
    };
    let q: Vec<f32> = (0..8 * n_heads * hd)
        .map(|_| (step() as f32 / u32::MAX as f32) - 0.5)
        .collect();
    let kv_words: Vec<u32> = (0..max_seq * nkv * hd / 4)
        .map(|_| step() & 0x3f3f3f3f)
        .collect();
    let scales: Vec<f32> = (0..max_seq * nkv)
        .map(|_| (step() as f32 / u32::MAX as f32) * 0.5 + 0.5)
        .collect();

    let q_buf = dispatch::storage_from_slice(ctx, "ppmk-q", &q);
    let k_buf = dispatch::storage_from_slice(ctx, "ppmk-k", &kv_words);
    let v_buf = dispatch::storage_from_slice(ctx, "ppmk-v", &kv_words);
    let ks_buf = dispatch::storage_from_slice(ctx, "ppmk-ks", &scales);
    let vs_buf = dispatch::storage_from_slice(ctx, "ppmk-vs", &scales);
    let out_buf = dispatch::storage_zeroed(ctx, "ppmk-out", 8);
    let poison_bind = dispatch::bind_group(ctx, &pl_poison, &[(3, &out_buf)]);

    let mut cells = 0;
    for (entry, fp8) in [
        (flash_decode::ENTRY_STAGE1_BF16_MK, false),
        (flash_decode::ENTRY_STAGE1_BF16_MK_U, false),
        (flash_decode::ENTRY_STAGE1_FP8_MK, true),
        (flash_decode::ENTRY_STAGE1_FP8_MK_U, true),
    ] {
        let pl_zi = build_pipeline_entry(ctx, &src, entry, true);
        let pl_nozi = build_pipeline_entry(ctx, &src, entry, false);
        for m in [1usize, 3, 8] {
            let scratch_len = n_heads * m * splits as usize * (hd + 2);
            let scratch = dispatch::storage_zeroed(ctx, "ppmk-scr", (scratch_len * 4) as u64);
            let run =
                |pl: &wgpu::ComputePipeline, total: u32, fill: u8, poison: bool| -> Vec<f32> {
                    let fd = FdParams {
                        n_heads: n_heads as u32,
                        n_kv: nkv as u32,
                        head_dim: hd as u32,
                        total,
                        start: 0,
                        splits,
                        ring: 0,
                        out_bf16: 1,
                        scaling: 1.0 / (hd as f32).sqrt(),
                        fused: 1,
                        m_rows: m as u32,
                        ..Default::default()
                    };
                    let fd_buf = dispatch::uniform_from(ctx, "ppmk-fd", &fd);
                    let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![
                        (0, &q_buf),
                        (4, &fd_buf),
                        (5, &k_buf),
                        (6, &v_buf),
                        (7, &scratch),
                    ];
                    if fp8 {
                        binds.push((8, &ks_buf));
                        binds.push((9, &vs_buf));
                    }
                    let bind = dispatch::bind_group(ctx, pl, &binds);
                    ctx.queue
                        .write_buffer(&scratch, 0, &vec![fill; scratch_len * 4]);
                    let mut enc = ctx.device.create_command_encoder(&Default::default());
                    {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        if poison {
                            pass.set_pipeline(&pl_poison);
                            pass.set_bind_group(0, &poison_bind, &[]);
                            pass.dispatch_workgroups(n_heads as u32, splits, 1);
                        }
                        pass.set_pipeline(pl);
                        pass.set_bind_group(0, &bind, &[]);
                        pass.dispatch_workgroups(n_heads as u32, splits, 1);
                    }
                    ctx.queue.submit([enc.finish()]);
                    dispatch::read_back(ctx, &scratch, scratch_len).unwrap()
                };
            for total in [8u32, 17, 128, 403, 1024] {
                let a = run(&pl_zi, total, 0xaa, false);
                let b = run(&pl_nozi, total, 0xbb, true);
                let live = a.iter().filter(|v| **v != 0.0 && v.is_finite()).count();

                let bar = if total as usize >= splits as usize * 8 {
                    scratch_len / 4
                } else {
                    0
                };
                assert!(
                    live > bar,
                    "{entry} m={m} total={total}: {live}/{scratch_len} live scratch words -- the poison fixture measured nothing"
                );
                let mismatches = a
                    .iter()
                    .zip(b.iter())
                    .enumerate()
                    .filter(|(_, (x, y))| x.to_bits() != y.to_bits())
                    .take(4)
                    .map(|(i, (x, y))| format!("[{i}] {:08x} vs {:08x}", x.to_bits(), y.to_bits()))
                    .collect::<Vec<_>>();
                assert!(
                    mismatches.is_empty(),
                    "{entry} m={m} total={total}: nozi+poison diverged from zero-init: {mismatches:?}"
                );
                cells += 1;
            }
        }
    }
    assert_eq!(cells, 4 * 3 * 5, "ran {cells} cells");
    eprintln!("flash stage1 mk nozi parity: {cells} cells bit-identical under a poisoned launch");
}

#[test]
#[ignore = "GPU timing A/B; run explicitly"]
fn flash_stage1_zero_init_ab() {
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    assert!(ctx.caps.timestamp_query, "needs TIMESTAMP_QUERY");
    let period = ctx.queue.get_timestamp_period() as f64;

    let n_heads = 8usize;
    let nkv = 2usize;
    let hd = 256usize;
    let splits = 16u32;
    let max_seq = 1024usize;
    let dispatches_per_token = 42usize;

    let src = compose(flash_decode::WGSL);
    let pl_zi = build_pipeline(ctx, &src, true);
    let pl_nozi = build_pipeline(ctx, &src, false);

    let mut seed = 0x243f6a8885a308d3u64;
    let mut step = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 32) as u32
    };
    let q: Vec<f32> = (0..n_heads * hd)
        .map(|_| (step() as f32 / u32::MAX as f32) - 0.5)
        .collect();
    let kv_words: Vec<u32> = (0..max_seq * nkv * hd / 4)
        .map(|_| step() & 0x3f3f3f3f)
        .collect();
    let scales: Vec<f32> = (0..max_seq * nkv)
        .map(|_| (step() as f32 / u32::MAX as f32) * 0.5 + 0.5)
        .collect();

    let q_buf = dispatch::storage_from_slice(ctx, "ab-q", &q);
    let k_buf = dispatch::storage_from_slice(ctx, "ab-k", &kv_words);
    let v_buf = dispatch::storage_from_slice(ctx, "ab-v", &kv_words);
    let ks_buf = dispatch::storage_from_slice(ctx, "ab-ks", &scales);
    let vs_buf = dispatch::storage_from_slice(ctx, "ab-vs", &scales);
    let scratch_len = n_heads * splits as usize * (hd + 2);
    let scratch = dispatch::storage_zeroed(ctx, "ab-scr", (scratch_len * 4) as u64);

    let qs = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("ab-qs"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ab-res"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ab-stg"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let run = |name: &str, pl: &wgpu::ComputePipeline, total: u32| -> Vec<f32> {
        let fd = FdParams {
            n_heads: n_heads as u32,
            n_kv: nkv as u32,
            head_dim: hd as u32,
            total,
            start: 0,
            splits,
            ring: 0,
            out_bf16: 1,
            scaling: 1.0 / (hd as f32).sqrt(),
            m_rows: 1,
            ..Default::default()
        };
        let fd_buf = dispatch::uniform_from(ctx, "ab-fd", &fd);
        let bind = dispatch::bind_group(
            ctx,
            pl,
            &[
                (0, &q_buf),
                (4, &fd_buf),
                (5, &k_buf),
                (6, &v_buf),
                (7, &scratch),
                (8, &ks_buf),
                (9, &vs_buf),
            ],
        );
        let warm = 3usize;
        let reps = 10usize;
        let mut gpu_ms = 0f64;
        for it in 0..(warm + reps) {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &qs,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });
                for _ in 0..dispatches_per_token {
                    pass.set_pipeline(pl);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(n_heads as u32, splits, 1);
                }
            }
            enc.resolve_query_set(&qs, 0..2, &resolve, 0);
            enc.copy_buffer_to_buffer(&resolve, 0, &staging, 0, 16);
            ctx.queue.submit([enc.finish()]);
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            ctx.poll_blocking().unwrap();
            rx.recv().unwrap().unwrap();
            let ticks: Vec<u64> = {
                let view = slice.get_mapped_range().unwrap();
                bytemuck::cast_slice::<u8, u64>(&view).to_vec()
            };
            staging.unmap();
            if it < warm {
                continue;
            }
            gpu_ms += ticks[1].saturating_sub(ticks[0]) as f64 * period / 1e6;
        }
        let out: Vec<f32> = dispatch::read_back(ctx, &scratch, scratch_len).unwrap();
        eprintln!(
            "{name} total={total}: {dispatches_per_token} dispatches -> gpu {:.3} ms ({:.2} us/dispatch)",
            gpu_ms / reps as f64,
            gpu_ms / reps as f64 * 1e3 / dispatches_per_token as f64
        );
        out
    };

    for total in [25u32, 512, 1024] {
        let a = run("zero-init ON (production)", &pl_zi, total);
        let b = run("zero-init OFF", &pl_nozi, total);
        let bitwise = a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits());
        eprintln!("outputs bit-identical: {bitwise}");
        assert!(bitwise, "zero-init off changed stage1 output");
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FncParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct KvFp8Params {
    n_tokens: u32,
    n_kv: u32,
    head_dim: u32,
    ring: u32,
    pairs: u32,
    start: u32,
    slots: u32,
    reserved: u32,
}

fn bf16_words(n: usize, seed: &mut u64) -> Vec<u32> {
    (0..n).map(|_| lcg(seed) & 0x3f3f3f3f).collect()
}

fn assert_bitwise_eq(a: &[u32], b: &[u32], what: &str) {
    let mismatches = a
        .iter()
        .zip(b.iter())
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .take(4)
        .map(|(i, (x, y))| format!("[{i}] {x:08x} vs {y:08x}"))
        .collect::<Vec<_>>();
    assert!(
        mismatches.is_empty(),
        "{what}: nozi+poison diverged from zero-init: {mismatches:?}"
    );
}

const FNC_POISON_WGSL: &str = "
@compute @workgroup_size(256)
fn fnc_poison_wg(@builtin(local_invocation_id) tid: vec3<u32>) {
    let p = bitcast<f32>(0x7fc0deadu | fnc_out[0]);
    fnc_scratch[tid.x] = p;
    if (tid.x == 0u) {
        fnc_shared = p;
    }
}
";

#[test]
fn fnc_nozi_bitwise_parity_poisoned() {
    let ctx = WgpuContext::shared()
        .unwrap_or_else(|e| panic!("no wgpu adapter, this proof cannot pass: {e}"));
    let src = format!("{}\n{}", compose(fused_norm_chain::WGSL), FNC_POISON_WGSL);
    let pl_poison = build_pipeline_entry(ctx, &src, "fnc_poison_wg", false);
    for entry in [
        fused_norm_chain::ENTRY_RMS_RES_RMS,
        fused_norm_chain::ENTRY_RES_OF_RMS,
        fused_norm_chain::ENTRY_RMS_RES_RMS_NEXT,
    ] {
        let pl_zi = build_pipeline_entry(ctx, &src, entry, true);
        let pl_nozi = build_pipeline_entry(ctx, &src, entry, false);
        for batch in [1usize, 3] {
            let hidden = 2048usize;
            let words = hidden / 2;
            let mut seed = 0x1234_5678_9abc_def0u64 ^ (batch as u64);
            let x = bf16_words(batch * words, &mut seed);
            let res0 = bf16_words(batch * words, &mut seed);
            let w1 = bf16_words(words, &mut seed);
            let w2 = bf16_words(words, &mut seed);
            let params = FncParams {
                hidden: hidden as u32,
                batch: batch as u32,
                eps: 1e-6,
                words_per_row: words as u32,
                scale: 0.5,
                ..Default::default()
            };
            let p_buf = dispatch::uniform_from(ctx, "fnc-p", &params);
            let run = |pl: &wgpu::ComputePipeline, poison: bool| -> Vec<Vec<u32>> {
                let x_buf = dispatch::storage_from_slice(ctx, "fnc-x", &x);
                let res_buf = dispatch::storage_from_slice(ctx, "fnc-res", &res0);
                let w1_buf = dispatch::storage_from_slice(ctx, "fnc-w1", &w1);
                let w2_buf = dispatch::storage_from_slice(ctx, "fnc-w2", &w2);
                let out_buf = dispatch::storage_zeroed(ctx, "fnc-out", (batch * words * 4) as u64);
                let out2_buf =
                    dispatch::storage_zeroed(ctx, "fnc-out2", (batch * words * 4) as u64);
                let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![
                    (0, &x_buf),
                    (1, &res_buf),
                    (2, &w1_buf),
                    (4, &out_buf),
                    (5, &p_buf),
                ];
                if entry != fused_norm_chain::ENTRY_RES_OF_RMS {
                    binds.push((3, &w2_buf));
                }
                if entry == fused_norm_chain::ENTRY_RMS_RES_RMS_NEXT {
                    binds.push((6, &out2_buf));
                }
                let bind = dispatch::bind_group(ctx, pl, &binds);
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    if poison {
                        let pbind = dispatch::bind_group(ctx, &pl_poison, &[(4, &out_buf)]);
                        pass.set_pipeline(&pl_poison);
                        pass.set_bind_group(0, &pbind, &[]);
                        pass.dispatch_workgroups(64, 1, 1);
                    }
                    pass.set_pipeline(pl);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(batch as u32, 1, 1);
                }
                ctx.queue.submit([enc.finish()]);
                vec![
                    dispatch::read_back(ctx, &res_buf, batch * words).unwrap(),
                    dispatch::read_back(ctx, &out_buf, batch * words).unwrap(),
                    dispatch::read_back(ctx, &out2_buf, batch * words).unwrap(),
                ]
            };
            let a = run(&pl_zi, false);
            let b = run(&pl_nozi, true);
            for (name, (x, y)) in ["res", "out", "out2"].iter().zip(a.iter().zip(b.iter())) {
                assert_bitwise_eq(x, y, &format!("fnc {entry} batch={batch} {name}"));
            }
            eprintln!("fnc {entry} batch={batch}: res/out/out2 bit-identical");
        }
    }
}

const RMS_POISON_WGSL: &str = "
@compute @workgroup_size(256)
fn rms_poison_wg(@builtin(local_invocation_id) tid: vec3<u32>) {
    let p = bitcast<f32>(0x7fc0deadu | rms_y[0]);
    rms_scratch[tid.x] = p;
    if (tid.x == 0u) {
        rms_shared = p;
    }
}
";

const RMSRES_POISON_WGSL: &str = "
@compute @workgroup_size(256)
fn rmsres_poison_wg(@builtin(local_invocation_id) tid: vec3<u32>) {
    let p = bitcast<f32>(0x7fc0deadu | rmsres_out[0]);
    rmsres_scratch[tid.x] = p;
    if (tid.x == 0u) {
        rmsres_shared = p;
    }
}
";

#[test]
fn rmsnorm_nozi_bitwise_parity_poisoned() {
    let ctx = WgpuContext::shared()
        .unwrap_or_else(|e| panic!("no wgpu adapter, this proof cannot pass: {e}"));
    let hidden = 2048usize;
    let words = hidden / 2;
    for batch in [1usize, 3] {
        let mut seed = 0xfeed_beef_0000_0001u64 ^ (batch as u64);
        let x = bf16_words(batch * words, &mut seed);
        let res0 = bf16_words(batch * words, &mut seed);
        let w = bf16_words(words, &mut seed);
        let params = RmsParams {
            hidden: hidden as u32,
            batch: batch as u32,
            eps: 1e-6,
            words_per_row: words as u32,
        };
        let p_buf = dispatch::uniform_from(ctx, "rms-p", &params);

        let src = format!("{}\n{}", compose(rmsnorm::WGSL), RMS_POISON_WGSL);
        let pl_poison = build_pipeline_entry(ctx, &src, "rms_poison_wg", false);
        let pl_zi = build_pipeline_entry(ctx, &src, "rmsnorm_bf16", true);
        let pl_nozi = build_pipeline_entry(ctx, &src, "rmsnorm_bf16", false);
        let run = |pl: &wgpu::ComputePipeline, poison: bool| -> Vec<u32> {
            let x_buf = dispatch::storage_from_slice(ctx, "rms-x", &x);
            let w_buf = dispatch::storage_from_slice(ctx, "rms-w", &w);
            let y_buf = dispatch::storage_zeroed(ctx, "rms-y", (batch * words * 4) as u64);
            let bind = dispatch::bind_group(
                ctx,
                pl,
                &[(0, &x_buf), (1, &w_buf), (2, &y_buf), (3, &p_buf)],
            );
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                if poison {
                    let pbind = dispatch::bind_group(ctx, &pl_poison, &[(2, &y_buf)]);
                    pass.set_pipeline(&pl_poison);
                    pass.set_bind_group(0, &pbind, &[]);
                    pass.dispatch_workgroups(64, 1, 1);
                }
                pass.set_pipeline(pl);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(batch as u32, 1, 1);
            }
            ctx.queue.submit([enc.finish()]);
            dispatch::read_back(ctx, &y_buf, batch * words).unwrap()
        };
        assert_bitwise_eq(
            &run(&pl_zi, false),
            &run(&pl_nozi, true),
            &format!("rmsnorm batch={batch}"),
        );
        eprintln!("rmsnorm_bf16 batch={batch}: bit-identical");

        let src_r = format!(
            "{}\n{}",
            compose(rmsnorm_residual::WGSL),
            RMSRES_POISON_WGSL
        );
        let plr_poison = build_pipeline_entry(ctx, &src_r, "rmsres_poison_wg", false);
        let plr_zi = build_pipeline_entry(ctx, &src_r, "rmsnorm_residual_bf16", true);
        let plr_nozi = build_pipeline_entry(ctx, &src_r, "rmsnorm_residual_bf16", false);
        let run_r = |pl: &wgpu::ComputePipeline, poison: bool| -> Vec<Vec<u32>> {
            let x_buf = dispatch::storage_from_slice(ctx, "rmsres-x", &x);
            let res_buf = dispatch::storage_from_slice(ctx, "rmsres-res", &res0);
            let w_buf = dispatch::storage_from_slice(ctx, "rmsres-w", &w);
            let out_buf = dispatch::storage_zeroed(ctx, "rmsres-out", (batch * words * 4) as u64);
            let bind = dispatch::bind_group(
                ctx,
                pl,
                &[
                    (0, &x_buf),
                    (1, &res_buf),
                    (2, &w_buf),
                    (3, &out_buf),
                    (4, &p_buf),
                ],
            );
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                if poison {
                    let pbind = dispatch::bind_group(ctx, &plr_poison, &[(3, &out_buf)]);
                    pass.set_pipeline(&plr_poison);
                    pass.set_bind_group(0, &pbind, &[]);
                    pass.dispatch_workgroups(64, 1, 1);
                }
                pass.set_pipeline(pl);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(batch as u32, 1, 1);
            }
            ctx.queue.submit([enc.finish()]);
            vec![
                dispatch::read_back(ctx, &res_buf, batch * words).unwrap(),
                dispatch::read_back(ctx, &out_buf, batch * words).unwrap(),
            ]
        };
        let a = run_r(&plr_zi, false);
        let b = run_r(&plr_nozi, true);
        for (name, (x, y)) in ["res", "out"].iter().zip(a.iter().zip(b.iter())) {
            assert_bitwise_eq(x, y, &format!("rmsnorm_residual batch={batch} {name}"));
        }
        eprintln!("rmsnorm_residual_bf16 batch={batch}: res/out bit-identical");
    }
}

const KV_POISON_WGSL: &str = "
@compute @workgroup_size(256)
fn kv_poison_wg(@builtin(local_invocation_id) tid: vec3<u32>) {
    let p = bitcast<f32>(0x7fc0deadu | kvq_out[0]);
    kv_scratch[tid.x] = p;
    if (tid.x == 0u) {
        kv_amax = p;
    }
}
";

#[test]
fn kv_quant_nozi_bitwise_parity_poisoned() {
    let ctx = WgpuContext::shared()
        .unwrap_or_else(|e| panic!("no wgpu adapter, this proof cannot pass: {e}"));
    let src = format!("{}\n{}", compose(kv_fp8::WGSL), KV_POISON_WGSL);
    let pl_poison = build_pipeline_entry(ctx, &src, "kv_poison_wg", false);
    let pl_zi = build_pipeline_entry(ctx, &src, kv_fp8::QUANTIZE_ENTRY, true);
    let pl_nozi = build_pipeline_entry(ctx, &src, kv_fp8::QUANTIZE_ENTRY, false);
    let n_kv = 2usize;
    let head_dim = 256usize;
    for (n_tokens, start) in [(1usize, 0i32), (3, 5)] {
        let pairs = n_tokens * n_kv;
        let slots = start as usize + n_tokens;
        let mut seed = 0x0dd0_c0de_0000_0002u64 ^ (n_tokens as u64);
        let x = bf16_words(n_tokens * n_kv * head_dim / 2, &mut seed);
        let params = KvFp8Params {
            n_tokens: n_tokens as u32,
            n_kv: n_kv as u32,
            head_dim: head_dim as u32,
            ring: 0,
            pairs: pairs as u32,
            start: 0,
            slots: slots as u32,
            reserved: 0,
        };
        let p_buf = dispatch::uniform_from(ctx, "kv-p", &params);
        let start_buf = dispatch::storage_from_slice(ctx, "kv-start", &[start]);
        let run = |pl: &wgpu::ComputePipeline, poison: bool| -> Vec<Vec<u32>> {
            let x_buf = dispatch::storage_from_slice(ctx, "kv-x", &x);
            let out_buf = dispatch::storage_zeroed(ctx, "kv-out", (slots * n_kv * head_dim) as u64);
            let sc_buf = dispatch::storage_zeroed(ctx, "kv-sc", (slots * n_kv * 4) as u64);
            let bind = dispatch::bind_group(
                ctx,
                pl,
                &[
                    (0, &x_buf),
                    (1, &out_buf),
                    (2, &sc_buf),
                    (3, &start_buf),
                    (4, &p_buf),
                ],
            );
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                if poison {
                    let pbind = dispatch::bind_group(ctx, &pl_poison, &[(1, &out_buf)]);
                    pass.set_pipeline(&pl_poison);
                    pass.set_bind_group(0, &pbind, &[]);
                    pass.dispatch_workgroups(64, 1, 1);
                }
                pass.set_pipeline(pl);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(pairs as u32, 1, 1);
            }
            ctx.queue.submit([enc.finish()]);
            vec![
                dispatch::read_back(ctx, &out_buf, slots * n_kv * head_dim / 4).unwrap(),
                dispatch::read_back(ctx, &sc_buf, slots * n_kv).unwrap(),
            ]
        };
        let a = run(&pl_zi, false);
        let b = run(&pl_nozi, true);
        for (name, (x, y)) in ["out", "scales"].iter().zip(a.iter().zip(b.iter())) {
            assert_bitwise_eq(
                x,
                y,
                &format!("kv_quant n_tokens={n_tokens} start={start} {name}"),
            );
        }
        eprintln!("quantize_kv_fp8 n_tokens={n_tokens} start={start}: out/scales bit-identical");
    }
}

#[test]
fn flash_stage2_nozi_bitwise_parity_poisoned() {
    let ctx = WgpuContext::shared()
        .unwrap_or_else(|e| panic!("no wgpu adapter, this proof cannot pass: {e}"));
    let src = format!("{}\n{}", compose(flash_decode::WGSL), POISON_WGSL);
    let pl_poison = build_pipeline_entry(ctx, &src, "fd_poison_wg", false);
    let pl_zi = build_pipeline_entry(ctx, &src, flash_decode::ENTRY_STAGE2, true);
    let pl_nozi = build_pipeline_entry(ctx, &src, flash_decode::ENTRY_STAGE2, false);
    let n_heads = 8usize;
    let hd = 256usize;
    for splits in [4u32, 16] {
        let scratch_len = n_heads * splits as usize * (hd + 2);
        let mut seed = 0x5ca1_ab1e_0000_0003u64 ^ (splits as u64);
        let scratch_data: Vec<f32> = (0..scratch_len)
            .map(|_| (lcg(&mut seed) as f32 / u32::MAX as f32) - 0.5)
            .collect();
        let fd = FdParams {
            n_heads: n_heads as u32,
            n_kv: 2,
            head_dim: hd as u32,
            total: 128,
            splits,
            out_bf16: 1,
            scaling: 1.0,
            m_rows: 1,
            ..Default::default()
        };
        let fd_buf = dispatch::uniform_from(ctx, "st2-fd", &fd);
        let run = |pl: &wgpu::ComputePipeline, poison: bool| -> Vec<u32> {
            let scratch = dispatch::storage_from_slice(ctx, "st2-scr", &scratch_data);
            let out_buf = dispatch::storage_zeroed(ctx, "st2-out", (n_heads * hd * 4) as u64);
            let bind = dispatch::bind_group(ctx, pl, &[(3, &out_buf), (4, &fd_buf), (7, &scratch)]);
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                if poison {
                    let pbind = dispatch::bind_group(ctx, &pl_poison, &[(3, &out_buf)]);
                    pass.set_pipeline(&pl_poison);
                    pass.set_bind_group(0, &pbind, &[]);
                    pass.dispatch_workgroups(64, 1, 1);
                }
                pass.set_pipeline(pl);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(n_heads as u32, 1, 1);
            }
            ctx.queue.submit([enc.finish()]);
            dispatch::read_back(ctx, &out_buf, n_heads * hd).unwrap()
        };
        assert_bitwise_eq(
            &run(&pl_zi, false),
            &run(&pl_nozi, true),
            &format!("flash_splitk_stage2 splits={splits}"),
        );
        eprintln!(
            "flash_splitk_stage2 splits={splits}: {} out words bit-identical",
            n_heads * hd
        );
    }
}
