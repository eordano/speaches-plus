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

#[derive(Clone)]
struct Case {
    t: usize,
    k: usize,
    rank: usize,
    widths: Vec<usize>,
    max_loras: usize,
    mapping: Vec<i32>,
    slot_ranks: Vec<usize>,
    scale: f32,
    seed: u64,
}

impl Case {
    fn sum_n(&self) -> usize {
        self.widths.iter().sum()
    }
    fn max_n(&self) -> usize {
        *self.widths.iter().max().unwrap()
    }
    fn slice_starts(&self) -> Vec<usize> {
        let mut acc = 0usize;
        self.widths
            .iter()
            .map(|w| {
                let s = acc;
                acc += w;
                s
            })
            .collect()
    }
}

fn build_x(case: &Case) -> Vec<bf16> {
    (0..case.t * case.k)
        .map(|i| bf16::from_f32(frand(case.seed ^ 0x11, i)))
        .collect()
}

fn build_y_base(case: &Case) -> Vec<bf16> {
    (0..case.t * case.sum_n())
        .map(|i| bf16::from_f32(frand(case.seed ^ 0x22, i) * 2.0))
        .collect()
}

fn build_a(case: &Case, wseed: u64) -> Vec<Vec<bf16>> {
    (0..case.widths.len())
        .map(|s| {
            let mut v = vec![bf16::from_f32(0.0); case.max_loras * case.rank * case.k];
            for slot in 0..case.max_loras {
                let occ = case.slot_ranks[slot];
                for r in 0..occ.min(case.rank) {
                    for kk in 0..case.k {
                        let idx = (slot * case.rank + r) * case.k + kk;
                        v[idx] = bf16::from_f32(
                            frand(
                                wseed ^ ((s as u64) << 8) ^ ((slot as u64) << 16),
                                r * case.k + kk,
                            ) * 0.25,
                        );
                    }
                }
            }
            v
        })
        .collect()
}

fn build_b(case: &Case, wseed: u64) -> Vec<Vec<bf16>> {
    case.widths
        .iter()
        .enumerate()
        .map(|(s, &w)| {
            let mut v = vec![bf16::from_f32(0.0); case.max_loras * w * case.rank];
            for slot in 0..case.max_loras {
                let occ = case.slot_ranks[slot];
                for n in 0..w {
                    for r in 0..occ.min(case.rank) {
                        let idx = (slot * w + n) * case.rank + r;
                        v[idx] = bf16::from_f32(
                            frand(
                                wseed ^ 0x33 ^ ((s as u64) << 8) ^ ((slot as u64) << 16),
                                n * case.rank + r,
                            ) * 0.25,
                        );
                    }
                }
            }
            v
        })
        .collect()
}

fn oracle(case: &Case, x: &[bf16], a: &[Vec<bf16>], b: &[Vec<bf16>], y_base: &[bf16]) -> Vec<bf16> {
    let sum_n = case.sum_n();
    let starts = case.slice_starts();
    let mut y = y_base.to_vec();
    for tok in 0..case.t {
        let slot = case.mapping[tok];
        if slot < 0 {
            continue;
        }
        let slot = slot as usize;
        for (s, &w) in case.widths.iter().enumerate() {
            let mut tmp = vec![0f32; case.rank];
            for r in 0..case.rank {
                let mut acc = 0f32;
                for kk in 0..case.k {
                    acc += x[tok * case.k + kk].to_f32()
                        * a[s][(slot * case.rank + r) * case.k + kk].to_f32();
                }
                tmp[r] = acc * case.scale;
            }
            for n in 0..w {
                let mut acc = 0f32;
                for r in 0..case.rank {
                    acc += tmp[r] * b[s][(slot * w + n) * case.rank + r].to_f32();
                }
                let yi = tok * sum_n + starts[s] + n;
                y[yi] = bf16::from_f32(y[yi].to_f32() + acc);
            }
        }
    }
    y
}

#[derive(Clone, Copy)]
struct Dims {
    m: i32,
    rank: i32,
    k: i32,
    n_slices: i32,
    grid_loras: i32,
    max_n: i32,
    y_row_stride: i32,
    a_d0: i64,
    scale: f32,
}

#[derive(Clone, Copy)]
struct Raw {
    x: u64,
    a_ptrs: u64,
    b_ptrs: u64,
    buf: u64,
    y: u64,
    map: u64,
    sorted: u64,
    counts: u64,
    start: u64,
    active: u64,
    slice_n: u64,
    slice_start: u64,
}

struct DevCase {
    x: CudaSlice<bf16>,
    a: Vec<CudaSlice<bf16>>,
    b: Vec<CudaSlice<bf16>>,
    a_ptrs: CudaSlice<u64>,
    b_ptrs: CudaSlice<u64>,
    buf: CudaSlice<f32>,
    y: CudaSlice<bf16>,
    map_d: CudaSlice<i32>,
    sorted_d: CudaSlice<i32>,
    counts_d: CudaSlice<i32>,
    start_d: CudaSlice<i32>,
    active_d: CudaSlice<i32>,
    slice_n_d: CudaSlice<i32>,
    slice_start_d: CudaSlice<i32>,
    dims: Dims,
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

    let n_slices = case.widths.len();
    let buf = stream
        .alloc_zeros::<f32>(n_slices * case.t * case.rank)
        .unwrap();
    let y = stream.clone_htod(y_base).unwrap();

    let map_d = stream.alloc_zeros::<i32>(case.t).unwrap();
    let sorted_d = stream.alloc_zeros::<i32>(case.t).unwrap();
    let counts_d = stream.alloc_zeros::<i32>(case.max_loras + 1).unwrap();
    let start_d = stream.alloc_zeros::<i32>(case.max_loras + 2).unwrap();
    let active_d = stream.alloc_zeros::<i32>(case.max_loras + 1).unwrap();

    let slice_n: Vec<i32> = case.widths.iter().map(|&w| w as i32).collect();
    let slice_start: Vec<i32> = case.slice_starts().iter().map(|&s| s as i32).collect();
    let slice_n_d = stream.clone_htod(&slice_n).unwrap();
    let slice_start_d = stream.clone_htod(&slice_start).unwrap();

    let dims = Dims {
        m: case.t as i32,
        rank: case.rank as i32,
        k: case.k as i32,
        n_slices: n_slices as i32,
        grid_loras: (case.max_loras + 1) as i32,
        max_n: case.max_n() as i32,
        y_row_stride: case.sum_n() as i32,
        a_d0: (case.rank * case.k) as i64,
        scale: case.scale,
    };

    DevCase {
        x,
        a,
        b,
        a_ptrs,
        b_ptrs,
        buf,
        y,
        map_d,
        sorted_d,
        counts_d,
        start_d,
        active_d,
        slice_n_d,
        slice_start_d,
        dims,
    }
}

impl DevCase {
    fn raw(&mut self, stream: &Arc<CudaStream>) -> Raw {
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
        Raw {
            x: ro(&self.x, stream),
            a_ptrs: ro(&self.a_ptrs, stream),
            b_ptrs: ro(&self.b_ptrs, stream),
            buf: rw(&mut self.buf, stream),
            y: rw(&mut self.y, stream),
            map: ro(&self.map_d, stream),
            sorted: ro(&self.sorted_d, stream),
            counts: ro(&self.counts_d, stream),
            start: ro(&self.start_d, stream),
            active: ro(&self.active_d, stream),
            slice_n: ro(&self.slice_n_d, stream),
            slice_start: ro(&self.slice_start_d, stream),
        }
    }

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

fn launch_pair(stream: &Arc<CudaStream>, r: &Raw, d: &Dims) -> anyhow::Result<()> {
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
            d.m,
            d.rank,
            d.k,
            d.n_slices,
            d.grid_loras,
            d.a_d0,
            d.scale,
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
            d.m,
            d.rank,
            d.max_n,
            d.n_slices,
            d.grid_loras,
            d.y_row_stride,
        )
    };
    if rc != 0 {
        anyhow::bail!("lora_expand rc={rc}");
    }
    Ok(())
}

fn run_case(case: &Case, tag: &str) -> Option<(Vec<bf16>, Vec<f32>, Vec<bf16>, Vec<bf16>)> {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "lora_grouped: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("lora_grouped: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
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

    let raw = dev.raw(&stream);
    let dims = dev.dims;
    launch_pair(&stream, &raw, &dims).unwrap();
    stream.synchronize().unwrap();

    let got: Vec<bf16> = stream.memcpy_dtov(&dev.y).unwrap();
    let buf: Vec<f32> = stream.memcpy_dtov(&dev.buf).unwrap();
    let want = oracle(case, &x_host, &a_host, &b_host, &y_base);
    assert_close(&got, &want, 2e-2, 2e-2, tag);
    Some((got, buf, want, y_base))
}

#[test]
fn meta_prepare_matches_vllm_semantics() {
    let meta = LoraKernelMeta::prepare(&[2, -1, 0, 2, -1, 2], 4);
    assert_eq!(meta.token_indices_sorted, vec![1, 4, 2, 0, 3, 5]);
    assert_eq!(meta.active_lora_ids, vec![-1, 0, 2, -1, -1]);
    assert_eq!(meta.num_tokens_per_lora, vec![2, 1, 3, 0, 0]);
    assert_eq!(meta.lora_token_start_loc, vec![0, 2, 3, 6, 0, 0]);
    assert_eq!(meta.num_active_loras, 3);
    assert!(!meta.no_lora);

    let meta = LoraKernelMeta::prepare(&[-1, -1], 2);
    assert!(meta.no_lora);
    assert_eq!(meta.active_lora_ids, vec![-1, -1, -1]);

    let meta = LoraKernelMeta::prepare(&[1, 1, 1], 2);
    assert!(!meta.no_lora);
    assert_eq!(meta.active_lora_ids, vec![1, -1, -1]);
    assert_eq!(meta.num_tokens_per_lora, vec![3, 0, 0]);
    assert_eq!(meta.lora_token_start_loc, vec![0, 3, 0, 0]);
}

#[test]
fn one_adapter_prefill() {
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
    };
    run_case(&case, "one_adapter_prefill");
}

#[test]
fn interleaved_adapters() {
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
    };
    run_case(&case, "interleaved_adapters");
}

#[test]
fn mixed_adapted_and_unadapted() {
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
    };
    let (got, buf, _want, y_base) = run_case(&case, "mixed_adapted_and_unadapted").unwrap();

    let sum_n = case.sum_n();
    for (tok, &slot) in mapping.iter().enumerate() {
        if slot != -1 {
            continue;
        }
        for s in 0..case.widths.len() {
            for r in 0..case.rank {
                let v = buf[s * case.t * case.rank + tok * case.rank + r];
                assert_eq!(v, 0.0, "buffer row for -1 token {tok} slice {s} not zero");
            }
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
fn decode_group_of_one() {
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
    };
    run_case(&case, "decode_group_of_one_t1");

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
    };
    run_case(&case, "decode_group_of_one_t4_distinct");
}

#[test]
fn large_group_prefill() {
    let mut mapping = vec![0i32; 512];
    mapping[0] = -1;
    mapping[511] = -1;
    let case = Case {
        t: 512,
        k: 128,
        rank: 32,
        widths: vec![96, 48],
        max_loras: 2,
        mapping,
        slot_ranks: vec![32, 32],
        scale: 1.0,
        seed: 6,
    };
    run_case(&case, "large_group_prefill");
}

#[test]
fn rank_below_slot_rank() {
    let case = Case {
        t: 16,
        k: 64,
        rank: 16,
        widths: vec![32, 16],
        max_loras: 3,
        mapping: (0..16).map(|i| (i % 2) as i32).collect(),
        slot_ranks: vec![16, 4, 16],
        scale: 1.0,
        seed: 7,
    };
    run_case(&case, "rank_below_slot_rank");
}

#[test]
fn all_zero_slot() {
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
    };
    let (got, _buf, _want, y_base) = run_case(&case, "all_zero_slot").unwrap();
    assert_eq!(got, y_base, "all-zero slot must leave y bitwise unchanged");
}

#[test]
fn unequal_slice_widths_gqa() {
    let mapping: Vec<i32> = (0..21)
        .map(|i| match i % 4 {
            0 => 0,
            1 => 2,
            2 => -1,
            _ => 1,
        })
        .collect();
    let case = Case {
        t: 21,
        k: 96,
        rank: 16,
        widths: vec![128, 32, 32],
        max_loras: 3,
        mapping,
        slot_ranks: vec![16, 16, 16],
        scale: 1.0,
        seed: 9,
    };
    run_case(&case, "unequal_slice_widths_gqa");
}

#[test]
fn graph_replay_sees_inplace_metadata_and_weight_mutation() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "lora_grouped: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("lora_grouped: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
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
    };

    let x_host = build_x(&case);
    let y_base = build_y_base(&case);
    let a1 = build_a(&case, 100);
    let b1 = build_b(&case, 100);

    let mut dev = setup(&stream, &case, &x_host, &a1, &b1, &y_base);
    let meta1 = LoraKernelMeta::prepare(&case.mapping, case.max_loras);
    dev.upload_meta(&stream, &meta1);

    let raw = dev.raw(&stream);
    let dims = dev.dims;

    let captures = Cell::new(0u32);
    let mut runner = CudaGraphRunner::new(stream.clone());

    runner
        .run(42, |s| {
            captures.set(captures.get() + 1);
            launch_pair(s, &raw, &dims)
        })
        .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(captures.get(), 1, "first run must capture exactly once");
    assert!(runner.has_cached());
    let nodes_after_capture = runner.cached_node_count();

    let y1: Vec<bf16> = stream.memcpy_dtov(&dev.y).unwrap();
    let want1 = oracle(&case, &x_host, &a1, &b1, &y_base);
    assert_close(&y1, &want1, 2e-2, 2e-2, "graph_capture_map1_w1");

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
        .run(42, |s| {
            captures.set(captures.get() + 1);
            launch_pair(s, &raw, &dims)
        })
        .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(captures.get(), 1, "second run must REPLAY, not recapture");
    assert_eq!(runner.cached_node_count(), nodes_after_capture);

    let y2: Vec<bf16> = stream.memcpy_dtov(&dev.y).unwrap();
    case.mapping = map2.clone();
    let want2 = oracle(&case, &x_host, &a2, &b2, &y_base);
    assert_close(&y2, &want2, 2e-2, 2e-2, "graph_replay_map2_w2");
    assert_ne!(
        y2, y1,
        "replay output must change after in-place metadata/weight mutation"
    );
    let stale = oracle(&case, &x_host, &a1, &b1, &y_base);
    let mut differs_from_stale = false;
    for (g, s) in y2.iter().zip(stale.iter()) {
        if (g.to_f32() - s.to_f32()).abs() > 5e-2 {
            differs_from_stale = true;
            break;
        }
    }
    assert!(
        differs_from_stale,
        "replay output still matches OLD weights: graph baked stale state"
    );

    let map3 = vec![-1i32; case.t];
    let meta3 = LoraKernelMeta::prepare(&map3, case.max_loras);
    dev.upload_meta(&stream, &meta3);
    stream.memcpy_htod(&y_base, &mut dev.y).unwrap();

    runner
        .run(42, |s| {
            captures.set(captures.get() + 1);
            launch_pair(s, &raw, &dims)
        })
        .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(captures.get(), 1, "third run must REPLAY, not recapture");

    let y3: Vec<bf16> = stream.memcpy_dtov(&dev.y).unwrap();
    assert_eq!(
        y3, y_base,
        "all -1 mapping on replay must leave y bitwise untouched"
    );
}
