#![cfg(feature = "cuda")]

mod common;
use common::assert_close;
use common::frand;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::graph::CudaGraphRunner;
use nv_kernels::lora::{self, LoraKernelMeta};
use std::cell::Cell;
use std::ffi::c_void;
use std::sync::Arc;
use common::Case;
use common::build_a;
use common::build_b;
use common::build_y_base;
use common::lora_oracle as oracle;

fn build_x(case: &Case) -> Vec<bf16> {
    (0..case.t * case.k)
        .map(|i| bf16::from_f32(frand(case.seed ^ 0x11, i)))
        .collect()
}

struct DevCase {
    x: CudaSlice<bf16>,
    a: Vec<CudaSlice<bf16>>,
    b: Vec<CudaSlice<bf16>>,
    a_ptrs: CudaSlice<u64>,
    b_ptrs: CudaSlice<u64>,
    y: CudaSlice<bf16>,
    map_d: CudaSlice<i32>,
    sorted_d: CudaSlice<i32>,
    counts_d: CudaSlice<i32>,
    start_d: CudaSlice<i32>,
    active_d: CudaSlice<i32>,
    slice_n_d: CudaSlice<i32>,
    slice_start_d: CudaSlice<i32>,
    b_stride_d: CudaSlice<i64>,
    buf: CudaSlice<f32>,
}

fn setup(
    stream: &Arc<CudaStream>,
    case: &Case,
    x_host: &[bf16],
    a_host: &[Vec<bf16>],
    b_host: &[Vec<bf16>],
    y_base: &[bf16],
) -> DevCase {
    let x = stream.clone_htod(x_host).unwrap();
    let a: Vec<CudaSlice<bf16>> = a_host
        .iter()
        .map(|v| stream.clone_htod(v).unwrap())
        .collect();
    let b: Vec<CudaSlice<bf16>> = b_host
        .iter()
        .map(|v| stream.clone_htod(v).unwrap())
        .collect();

    let a_addrs: Vec<u64> = a
        .iter()
        .map(|s| {
            let (p, g) = s.device_ptr(stream);
            drop(g);
            p as u64
        })
        .collect();
    let b_addrs: Vec<u64> = b
        .iter()
        .map(|s| {
            let (p, g) = s.device_ptr(stream);
            drop(g);
            p as u64
        })
        .collect();
    let a_ptrs = stream.clone_htod(&a_addrs).unwrap();
    let b_ptrs = stream.clone_htod(&b_addrs).unwrap();

    let y = stream.clone_htod(y_base).unwrap();

    let map_d = stream.alloc_zeros::<i32>(case.t).unwrap();
    let sorted_d = stream.alloc_zeros::<i32>(case.t).unwrap();
    let counts_d = stream.alloc_zeros::<i32>(case.max_loras + 1).unwrap();
    let start_d = stream.alloc_zeros::<i32>(case.max_loras + 2).unwrap();
    let active_d = stream.alloc_zeros::<i32>(case.max_loras + 1).unwrap();

    let slice_n: Vec<i32> = case.widths.iter().map(|&w| w as i32).collect();
    let slice_start: Vec<i32> = case.slice_starts().iter().map(|&s| s as i32).collect();
    let b_stride: Vec<i64> = case
        .widths
        .iter()
        .map(|&w| (w * case.rank) as i64)
        .collect();
    let slice_n_d = stream.clone_htod(&slice_n).unwrap();
    let slice_start_d = stream.clone_htod(&slice_start).unwrap();
    let b_stride_d = stream.clone_htod(&b_stride).unwrap();
    let buf = stream
        .alloc_zeros::<f32>(case.widths.len() * case.t * case.rank)
        .unwrap();

    DevCase {
        x,
        a,
        b,
        a_ptrs,
        b_ptrs,
        y,
        map_d,
        sorted_d,
        counts_d,
        start_d,
        active_d,
        slice_n_d,
        slice_start_d,
        b_stride_d,
        buf,
    }
}

impl DevCase {
    fn upload_meta(&mut self, stream: &Arc<CudaStream>, meta: &LoraKernelMeta) {
        stream
            .memcpy_htod(&meta.token_lora_mapping, &mut self.map_d)
            .unwrap();
        stream
            .memcpy_htod(&meta.token_indices_sorted, &mut self.sorted_d)
            .unwrap();
        stream
            .memcpy_htod(&meta.num_tokens_per_lora, &mut self.counts_d)
            .unwrap();
        stream
            .memcpy_htod(&meta.lora_token_start_loc, &mut self.start_d)
            .unwrap();
        stream
            .memcpy_htod(&meta.active_lora_ids, &mut self.active_d)
            .unwrap();
    }
}

#[derive(Clone, Copy)]
struct RawPtrs {
    x: u64,
    a_ptrs: u64,
    b_ptrs: u64,
    y: u64,
    map: u64,
    sorted: u64,
    counts: u64,
    start: u64,
    active: u64,
    slice_n: u64,
    slice_start: u64,
    b_stride: u64,
    buf: u64,
}

impl DevCase {
    fn raw(&mut self, stream: &Arc<CudaStream>) -> RawPtrs {
        fn ro<T>(s: &CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
            let (p, g) = s.device_ptr(stream);
            drop(g);
            p as u64
        }
        fn rw<T>(s: &mut CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
            let (p, g) = s.device_ptr_mut(stream);
            drop(g);
            p as u64
        }
        RawPtrs {
            x: ro(&self.x, stream),
            a_ptrs: ro(&self.a_ptrs, stream),
            b_ptrs: ro(&self.b_ptrs, stream),
            y: rw(&mut self.y, stream),
            map: ro(&self.map_d, stream),
            sorted: ro(&self.sorted_d, stream),
            counts: ro(&self.counts_d, stream),
            start: ro(&self.start_d, stream),
            active: ro(&self.active_d, stream),
            slice_n: ro(&self.slice_n_d, stream),
            slice_start: ro(&self.slice_start_d, stream),
            b_stride: ro(&self.b_stride_d, stream),
            buf: rw(&mut self.buf, stream),
        }
    }
}

fn launch_fused_raw(stream: &Arc<CudaStream>, r: &RawPtrs, case: &Case) -> anyhow::Result<()> {
    let (win_off, win_len) = case.window();
    let rc = unsafe {
        lora::lora_fused(
            stream.cu_stream() as *mut c_void,
            r.x as *const u16,
            r.a_ptrs as *const u64,
            r.b_ptrs as *const u64,
            r.y as *mut u16,
            r.sorted as *const i32,
            r.counts as *const i32,
            r.start as *const i32,
            r.active as *const i32,
            r.slice_n as *const i32,
            r.slice_start as *const i32,
            r.b_stride as *const i64,
            case.t as i32,
            case.rank as i32,
            case.k as i32,
            case.max_n() as i32,
            case.widths.len() as i32,
            (case.max_loras + 1) as i32,
            (case.rank * case.k) as i64,
            win_off as i32,
            win_len as i32,
            win_len as i32,
            case.scale,
        )
    };
    if rc != 0 {
        anyhow::bail!("lora_fused rc={rc}");
    }
    Ok(())
}

fn launch_fused(stream: &Arc<CudaStream>, dev: &mut DevCase, case: &Case) -> anyhow::Result<()> {
    let raw = dev.raw(stream);
    launch_fused_raw(stream, &raw, case)
}

fn launch_two_raw(stream: &Arc<CudaStream>, r: &RawPtrs, case: &Case) -> anyhow::Result<()> {
    let rc = unsafe {
        lora::lora_shrink(
            stream.cu_stream() as *mut c_void,
            r.x as *const u16,
            r.a_ptrs as *const u64,
            r.buf as *mut f32,
            r.map as *const i32,
            r.sorted as *const i32,
            r.counts as *const i32,
            r.start as *const i32,
            r.active as *const i32,
            case.t as i32,
            case.rank as i32,
            case.k as i32,
            case.widths.len() as i32,
            (case.max_loras + 1) as i32,
            (case.rank * case.k) as i64,
            case.scale,
        )
    };
    if rc != 0 {
        anyhow::bail!("lora_shrink rc={rc}");
    }
    let rc = unsafe {
        lora::lora_expand(
            stream.cu_stream() as *mut c_void,
            r.buf as *const f32,
            r.b_ptrs as *const u64,
            r.y as *mut u16,
            r.map as *const i32,
            r.sorted as *const i32,
            r.counts as *const i32,
            r.start as *const i32,
            r.active as *const i32,
            r.slice_n as *const i32,
            r.slice_start as *const i32,
            case.t as i32,
            case.rank as i32,
            case.max_n() as i32,
            case.widths.len() as i32,
            (case.max_loras + 1) as i32,
            case.sum_n() as i32,
        )
    };
    if rc != 0 {
        anyhow::bail!("lora_expand rc={rc}");
    }
    Ok(())
}

fn launch_two(stream: &Arc<CudaStream>, dev: &mut DevCase, case: &Case) -> anyhow::Result<()> {
    let raw = dev.raw(stream);
    launch_two_raw(stream, &raw, case)
}

fn run_case(case: &Case, tag: &str) -> Option<(Vec<bf16>, Vec<bf16>, Vec<bf16>)> {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "lora_fused: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("lora_fused: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return None;
    };
    let stream = ctx.default_stream();

    let x_host = build_x(case);
    let a_host = build_a(case, case.seed);
    let b_host = build_b(case, case.seed);
    let y_base = build_y_base(case);

    let mut dev = setup(&stream, case, &x_host, &a_host, &b_host, &y_base);
    let meta = LoraKernelMeta::prepare(&case.mapping, case.max_loras);
    dev.upload_meta(&stream, &meta);

    launch_fused(&stream, &mut dev, case).unwrap();
    stream.synchronize().unwrap();

    let got: Vec<bf16> = stream.memcpy_dtov(&dev.y).unwrap();
    let want = oracle(case, &x_host, &a_host, &b_host, &y_base);
    assert_close(&got, &want, 2e-2, 2e-2, tag);
    Some((got, want, y_base))
}

#[test]
fn fused_one_adapter_prefill() {
    let case = Case {
        t: 64,
        k: 128,
        rank: 16,
        widths: vec![64],
        max_loras: 2,
        mapping: vec![0; 64],
        slot_ranks: vec![16, 16],
        scale: 1.0,
        seed: 1,
        win: None,
    };
    run_case(&case, "fused_one_adapter_prefill");
}

#[test]
fn fused_interleaved_adapters_multi_slice() {
    let mapping: Vec<i32> = (0..33).map(|i| (i % 4) as i32).collect();
    let case = Case {
        t: 33,
        k: 96,
        rank: 16,
        widths: vec![48, 32],
        max_loras: 4,
        mapping,
        slot_ranks: vec![16, 16, 16, 16],
        scale: 0.5,
        seed: 2,
        win: None,
    };
    run_case(&case, "fused_interleaved_adapters_multi_slice");
}

#[test]
fn fused_mixed_adapted_and_unadapted() {
    let mapping: Vec<i32> = (0..24)
        .map(|i| match i % 3 {
            0 => -1,
            1 => 0,
            _ => 2,
        })
        .collect();
    let case = Case {
        t: 24,
        k: 64,
        rank: 8,
        widths: vec![32, 16],
        max_loras: 3,
        mapping: mapping.clone(),
        slot_ranks: vec![8, 8, 8],
        scale: 1.0,
        seed: 3,
        win: None,
    };
    let (got, _want, y_base) = run_case(&case, "fused_mixed_adapted_and_unadapted").unwrap();
    let sum_n = case.sum_n();
    for (tok, &slot) in mapping.iter().enumerate() {
        if slot != -1 {
            continue;
        }
        for n in 0..sum_n {
            assert_eq!(
                got[tok * sum_n + n],
                y_base[tok * sum_n + n],
                "y row for -1 token {tok} was touched"
            );
        }
    }
}

#[test]
fn fused_decode_group_of_one() {
    let case = Case {
        t: 1,
        k: 128,
        rank: 16,
        widths: vec![64, 32],
        max_loras: 4,
        mapping: vec![2],
        slot_ranks: vec![16, 16, 16, 16],
        scale: 1.0,
        seed: 4,
        win: None,
    };
    run_case(&case, "fused_decode_t1");

    let case = Case {
        t: 4,
        k: 64,
        rank: 8,
        widths: vec![32],
        max_loras: 4,
        mapping: vec![3, 0, 2, 1],
        slot_ranks: vec![8, 8, 8, 8],
        scale: 1.0,
        seed: 5,
        win: None,
    };
    run_case(&case, "fused_decode_t4_distinct");
}

#[test]
fn fused_wide_slice_needs_chunking() {
    let case = Case {
        t: 5,
        k: 256,
        rank: 32,
        widths: vec![640, 320],
        max_loras: 2,
        mapping: vec![0, 1, -1, 0, 1],
        slot_ranks: vec![32, 32],
        scale: 1.0,
        seed: 6,
        win: None,
    };
    run_case(&case, "fused_wide_slice_needs_chunking");
}

#[test]
fn fused_rank_below_slot_rank_and_max_rank_64() {
    let case = Case {
        t: 16,
        k: 64,
        rank: 64,
        widths: vec![96, 32],
        max_loras: 3,
        mapping: (0..16).map(|i| (i % 2) as i32).collect(),
        slot_ranks: vec![64, 4, 64],
        scale: 1.0,
        seed: 7,
        win: None,
    };
    run_case(&case, "fused_rank_below_slot_rank_and_max_rank_64");
}

#[test]
fn fused_all_zero_slot_leaves_y_untouched() {
    let case = Case {
        t: 8,
        k: 64,
        rank: 8,
        widths: vec![32],
        max_loras: 2,
        mapping: vec![1; 8],
        slot_ranks: vec![8, 0],
        scale: 1.0,
        seed: 8,
        win: None,
    };
    let (got, _want, y_base) = run_case(&case, "fused_all_zero_slot").unwrap();
    assert_eq!(got, y_base, "all-zero slot must leave y bitwise unchanged");
}

#[test]
fn fused_row_window_matches_full_slice() {
    let full = Case {
        t: 9,
        k: 96,
        rank: 16,
        widths: vec![128, 32, 32],
        max_loras: 2,
        mapping: vec![0, 1, -1, 0, 1, 0, -1, 1, 0],
        slot_ranks: vec![16, 16],
        scale: 1.0,
        seed: 9,
        win: None,
    };
    let (full_got, _w, _y) = run_case(&full, "fused_window_reference_full").unwrap();

    for (win_off, win_len, tag) in [
        (0usize, 128usize, "fused_window_q"),
        (128, 32, "fused_window_k"),
        (160, 32, "fused_window_v"),
        (100, 60, "fused_window_straddles_slices"),
    ] {
        let mut case = full.clone();
        case.win = Some((win_off, win_len));
        let (got, _want, _yb) = run_case(&case, tag).unwrap();
        let sum_n = full.sum_n();
        let full_y_base = build_y_base(&full);
        let win_y_base = build_y_base(&case);
        for tok in 0..case.t {
            for c in 0..win_len {
                let g = got[tok * win_len + c].to_f32();
                let base_delta = full_got[tok * sum_n + win_off + c].to_f32()
                    - full_y_base[tok * sum_n + win_off + c].to_f32();
                let want = win_y_base[tok * win_len + c].to_f32() + base_delta;
                assert!(
                    (g - want).abs() <= 4e-2 + 2e-2 * want.abs(),
                    "{tag}: tok {tok} col {c}: got {g} want {want}"
                );
            }
        }
    }
}

#[test]
fn fused_matches_two_launch_path() {
    let case = Case {
        t: 40,
        k: 128,
        rank: 32,
        widths: vec![96, 48],
        max_loras: 3,
        mapping: (0..40).map(|i| [0i32, 2, -1, 1][i % 4]).collect(),
        slot_ranks: vec![32, 32, 32],
        scale: 0.75,
        seed: 10,
        win: None,
    };
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "lora_fused: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("lora_fused: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let x_host = build_x(&case);
    let a_host = build_a(&case, case.seed);
    let b_host = build_b(&case, case.seed);
    let y_base = build_y_base(&case);
    let meta = LoraKernelMeta::prepare(&case.mapping, case.max_loras);

    let mut dev1 = setup(&stream, &case, &x_host, &a_host, &b_host, &y_base);
    dev1.upload_meta(&stream, &meta);
    launch_fused(&stream, &mut dev1, &case).unwrap();
    stream.synchronize().unwrap();
    let y_fused: Vec<bf16> = stream.memcpy_dtov(&dev1.y).unwrap();

    let mut dev2 = setup(&stream, &case, &x_host, &a_host, &b_host, &y_base);
    dev2.upload_meta(&stream, &meta);
    launch_two(&stream, &mut dev2, &case).unwrap();
    stream.synchronize().unwrap();
    let y_two: Vec<bf16> = stream.memcpy_dtov(&dev2.y).unwrap();

    let want = oracle(&case, &x_host, &a_host, &b_host, &y_base);
    assert_close(&y_fused, &want, 2e-2, 2e-2, "fused_vs_oracle");
    assert_close(&y_two, &want, 2e-2, 2e-2, "two_launch_vs_oracle");
}

#[test]
fn fused_graph_replay_sees_inplace_mutation() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "lora_fused: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("lora_fused: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.new_stream().expect("new_stream");

    let mut case = Case {
        t: 8,
        k: 64,
        rank: 16,
        widths: vec![64, 32],
        max_loras: 4,
        mapping: vec![0; 8],
        slot_ranks: vec![16, 16, 16, 16],
        scale: 1.0,
        seed: 100,
        win: None,
    };

    let x_host = build_x(&case);
    let y_base = build_y_base(&case);
    let a1 = build_a(&case, 100);
    let b1 = build_b(&case, 100);

    let mut dev = setup(&stream, &case, &x_host, &a1, &b1, &y_base);
    let meta1 = LoraKernelMeta::prepare(&case.mapping, case.max_loras);
    dev.upload_meta(&stream, &meta1);

    let captures = Cell::new(0u32);
    let mut runner = CudaGraphRunner::new(stream.clone());

    let case_c = case.clone();
    let raw = dev.raw(&stream);
    runner
        .run(7, |s| {
            captures.set(captures.get() + 1);
            launch_fused_raw(s, &raw, &case_c)
        })
        .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(captures.get(), 1, "first run must capture exactly once");
    assert!(runner.has_cached());
    let nodes = runner.cached_node_count();

    let y1: Vec<bf16> = stream.memcpy_dtov(&dev.y).unwrap();
    let want1 = oracle(&case, &x_host, &a1, &b1, &y_base);
    assert_close(&y1, &want1, 2e-2, 2e-2, "fused_graph_capture");

    let map2: Vec<i32> = vec![1, 3, -1, 1, 3, -1, 1, 3];
    let a2 = build_a(&case, 200);
    let b2 = build_b(&case, 200);
    for (s, host) in a2.iter().enumerate() {
        stream.memcpy_htod(host, &mut dev.a[s]).unwrap();
    }
    for (s, host) in b2.iter().enumerate() {
        stream.memcpy_htod(host, &mut dev.b[s]).unwrap();
    }
    let meta2 = LoraKernelMeta::prepare(&map2, case.max_loras);
    dev.upload_meta(&stream, &meta2);
    stream.memcpy_htod(&y_base, &mut dev.y).unwrap();

    runner
        .run(7, |s| {
            captures.set(captures.get() + 1);
            launch_fused_raw(s, &raw, &case_c)
        })
        .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(captures.get(), 1, "second run must REPLAY, not recapture");
    assert_eq!(runner.cached_node_count(), nodes);

    let y2: Vec<bf16> = stream.memcpy_dtov(&dev.y).unwrap();
    case.mapping = map2;
    let want2 = oracle(&case, &x_host, &a2, &b2, &y_base);
    assert_close(&y2, &want2, 2e-2, 2e-2, "fused_graph_replay_new_state");

    let map3 = vec![-1i32; case.t];
    let meta3 = LoraKernelMeta::prepare(&map3, case.max_loras);
    dev.upload_meta(&stream, &meta3);
    stream.memcpy_htod(&y_base, &mut dev.y).unwrap();
    runner
        .run(7, |s| {
            captures.set(captures.get() + 1);
            launch_fused_raw(s, &raw, &case_c)
        })
        .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(captures.get(), 1, "third run must REPLAY");
    let y3: Vec<bf16> = stream.memcpy_dtov(&dev.y).unwrap();
    assert_eq!(y3, y_base, "all -1 replay must leave y bitwise untouched");
}

#[test]
fn bench_fused_vs_two_launch() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "lora_fused: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("lora_fused: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    for (t, tag) in [
        (1usize, "decode_t1"),
        (8, "spec_t8"),
        (64, "m64"),
        (256, "prefill_t256"),
    ] {
        let case = Case {
            t,
            k: 5376,
            rank: 64,
            widths: vec![4096, 1024, 1024],
            max_loras: 4,
            mapping: (0..t).map(|i| (i % 2) as i32).collect(),
            slot_ranks: vec![64, 64, 64, 64],
            scale: 1.0,
            seed: 42,
            win: None,
        };
        let x_host = build_x(&case);
        let a_host = build_a(&case, case.seed);
        let b_host = build_b(&case, case.seed);
        let y_base = build_y_base(&case);
        let meta = LoraKernelMeta::prepare(&case.mapping, case.max_loras);
        let mut dev = setup(&stream, &case, &x_host, &a_host, &b_host, &y_base);
        dev.upload_meta(&stream, &meta);

        let raw = dev.raw(&stream);
        for _ in 0..10 {
            launch_fused_raw(&stream, &raw, &case).unwrap();
            launch_two_raw(&stream, &raw, &case).unwrap();
        }
        stream.synchronize().unwrap();

        let iters = 200;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            launch_fused_raw(&stream, &raw, &case).unwrap();
        }
        stream.synchronize().unwrap();
        let fused_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            launch_two_raw(&stream, &raw, &case).unwrap();
        }
        stream.synchronize().unwrap();
        let two_us = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;

        eprintln!(
            "[bench {tag}] t={t} k=5376 rank=64 widths=[4096,1024,1024]: fused {fused_us:.1} us/iter, shrink+expand {two_us:.1} us/iter, ratio {:.2}x",
            two_us / fused_us
        );
    }
}
