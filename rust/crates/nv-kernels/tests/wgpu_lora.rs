#![cfg(feature = "wgpu")]

mod common;
use common::assert_close;
use common::ctx_or_skip;
use common::frand;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::lora::{self, LoraMeta};
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

fn bits(v: &[bf16]) -> Vec<u16> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn from_bits(v: &[u16]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_bits(x)).collect()
}

fn slice_refs(v: &[Vec<u16>]) -> Vec<&[u16]> {
    v.iter().map(|s| s.as_slice()).collect()
}

#[allow(dead_code)]
fn report_bits(tag: &str, got: &[u16], want: &[u16]) -> (usize, i32) {
    assert_eq!(got.len(), want.len(), "{tag}: length mismatch");
    let mut mismatch = 0usize;
    let mut max_ulp = 0i32;
    for (g, w) in got.iter().zip(want.iter()) {
        if g != w {
            mismatch += 1;
            max_ulp = max_ulp.max((*g as i32 - *w as i32).abs());
        }
    }
    eprintln!(
        "{tag}: {mismatch}/{} words differ, max_ulp={max_ulp}",
        got.len()
    );
    (mismatch, max_ulp)
}

struct WgpuOut {
    y_grouped: Option<Vec<bf16>>,
    buf_grouped: Option<Vec<f32>>,
    y_fused: Vec<bf16>,
    y_base: Vec<bf16>,
}

fn run_wgpu(ctx: &WgpuContext, case: &Case, tag: &str) -> WgpuOut {
    let x = build_x(case);
    let a = build_a(case, case.seed);
    let b = build_b(case, case.seed);
    let y_base = build_y_base(case);
    let meta = LoraMeta::prepare(&case.mapping, case.max_loras);
    let (win_off, win_len) = case.window();

    let x_bits = bits(&x);
    let a_bits: Vec<Vec<u16>> = a.iter().map(|v| bits(v)).collect();
    let b_bits: Vec<Vec<u16>> = b.iter().map(|v| bits(v)).collect();
    let want = oracle(case, &x, &a, &b, &y_base);

    let mut y_grouped = None;
    let mut buf_grouped = None;
    if case.win.is_none() {
        let mut y = bits(&y_base);
        let mut buf = vec![0f32; case.widths.len() * case.t * case.rank];
        lora::lora_grouped(
            ctx,
            &x_bits,
            &slice_refs(&a_bits),
            &slice_refs(&b_bits),
            &mut y,
            &meta,
            &case.widths,
            case.t,
            case.rank,
            case.k,
            case.sum_n(),
            case.scale,
            Some(&mut buf),
        )
        .expect("wgpu lora_grouped");
        let y = from_bits(&y);
        assert_close(&y, &want, 2e-2, 2e-2, &format!("{tag}_wgpu_grouped"));
        y_grouped = Some(y);
        buf_grouped = Some(buf);
    }

    let mut y = bits(&y_base);
    lora::lora_fused(
        ctx,
        &x_bits,
        &slice_refs(&a_bits),
        &slice_refs(&b_bits),
        &mut y,
        &meta,
        &case.widths,
        case.t,
        case.rank,
        case.k,
        win_off,
        win_len,
        win_len,
        case.scale,
    )
    .expect("wgpu lora_fused");
    let y_fused = from_bits(&y);
    assert_close(&y_fused, &want, 2e-2, 2e-2, &format!("{tag}_wgpu_fused"));

    #[cfg(feature = "cuda")]
    cuda_ref::cross_check(
        case,
        tag,
        &x,
        &a,
        &b,
        &y_base,
        &y_grouped,
        &buf_grouped,
        &y_fused,
    );

    WgpuOut {
        y_grouped,
        buf_grouped,
        y_fused,
        y_base,
    }
}

#[cfg(feature = "cuda")]
mod cuda_ref {
    use super::*;
    use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use nv_kernels::lora as culora;
    use std::ffi::c_void;
    use std::sync::Arc;

    struct Dev {
        _x: CudaSlice<bf16>,
        _a: Vec<CudaSlice<bf16>>,
        _b: Vec<CudaSlice<bf16>>,
        _a_ptrs: CudaSlice<u64>,
        _b_ptrs: CudaSlice<u64>,
        y: CudaSlice<bf16>,
        buf: CudaSlice<f32>,
        _map: CudaSlice<i32>,
        _sorted: CudaSlice<i32>,
        _counts: CudaSlice<i32>,
        _start: CudaSlice<i32>,
        _active: CudaSlice<i32>,
        _slice_n: CudaSlice<i32>,
        _slice_start: CudaSlice<i32>,
        _b_stride: CudaSlice<i64>,
        raw: [u64; 13],
    }

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

    fn setup(
        stream: &Arc<CudaStream>,
        case: &Case,
        x: &[bf16],
        a: &[Vec<bf16>],
        b: &[Vec<bf16>],
        y_base: &[bf16],
    ) -> Dev {
        let x_d = stream.clone_htod(x).unwrap();
        let a_d: Vec<CudaSlice<bf16>> = a.iter().map(|v| stream.clone_htod(v).unwrap()).collect();
        let b_d: Vec<CudaSlice<bf16>> = b.iter().map(|v| stream.clone_htod(v).unwrap()).collect();
        let a_addrs: Vec<u64> = a_d.iter().map(|s| ro(s, stream)).collect();
        let b_addrs: Vec<u64> = b_d.iter().map(|s| ro(s, stream)).collect();
        let a_ptrs = stream.clone_htod(&a_addrs).unwrap();
        let b_ptrs = stream.clone_htod(&b_addrs).unwrap();
        let mut y = stream.clone_htod(y_base).unwrap();
        let mut buf = stream
            .alloc_zeros::<f32>(case.widths.len() * case.t * case.rank)
            .unwrap();

        let meta = culora::LoraKernelMeta::prepare(&case.mapping, case.max_loras);
        let map = stream.clone_htod(&meta.token_lora_mapping).unwrap();
        let sorted = stream.clone_htod(&meta.token_indices_sorted).unwrap();
        let counts = stream.clone_htod(&meta.num_tokens_per_lora).unwrap();
        let start = stream.clone_htod(&meta.lora_token_start_loc).unwrap();
        let active = stream.clone_htod(&meta.active_lora_ids).unwrap();
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

        let raw = [
            ro(&x_d, stream),
            ro(&a_ptrs, stream),
            ro(&b_ptrs, stream),
            rw(&mut y, stream),
            rw(&mut buf, stream),
            ro(&map, stream),
            ro(&sorted, stream),
            ro(&counts, stream),
            ro(&start, stream),
            ro(&active, stream),
            ro(&slice_n_d, stream),
            ro(&slice_start_d, stream),
            ro(&b_stride_d, stream),
        ];

        Dev {
            _x: x_d,
            _a: a_d,
            _b: b_d,
            _a_ptrs: a_ptrs,
            _b_ptrs: b_ptrs,
            y,
            buf,
            _map: map,
            _sorted: sorted,
            _counts: counts,
            _start: start,
            _active: active,
            _slice_n: slice_n_d,
            _slice_start: slice_start_d,
            _b_stride: b_stride_d,
            raw,
        }
    }

    fn run(
        case: &Case,
        x: &[bf16],
        a: &[Vec<bf16>],
        b: &[Vec<bf16>],
        y_base: &[bf16],
        fused: bool,
    ) -> Option<(Vec<bf16>, Vec<f32>)> {
        let Ok(ctx) = CudaContext::new(0) else {
            if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
                panic!(
                    "wgpu_lora: no CUDA device 0. This gate refuses to report success without \
                     running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
                );
            }
            eprintln!("wgpu_lora: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
            return None;
        };
        let stream = ctx.default_stream();
        let dev = setup(&stream, case, x, a, b, y_base);
        let r = &dev.raw;
        let (win_off, win_len) = case.window();
        let max_n = *case.widths.iter().max().unwrap();

        if fused {
            let rc = unsafe {
                culora::lora_fused(
                    stream.cu_stream() as *mut c_void,
                    r[0] as *const u16,
                    r[1] as *const u64,
                    r[2] as *const u64,
                    r[3] as *mut u16,
                    r[6] as *const i32,
                    r[7] as *const i32,
                    r[8] as *const i32,
                    r[9] as *const i32,
                    r[10] as *const i32,
                    r[11] as *const i32,
                    r[12] as *const i64,
                    case.t as i32,
                    case.rank as i32,
                    case.k as i32,
                    max_n as i32,
                    case.widths.len() as i32,
                    (case.max_loras + 1) as i32,
                    (case.rank * case.k) as i64,
                    win_off as i32,
                    win_len as i32,
                    win_len as i32,
                    case.scale,
                )
            };
            assert_eq!(rc, 0, "cuda lora_fused rc={rc}");
        } else {
            let rc = unsafe {
                culora::lora_shrink(
                    stream.cu_stream() as *mut c_void,
                    r[0] as *const u16,
                    r[1] as *const u64,
                    r[4] as *mut f32,
                    r[5] as *const i32,
                    r[6] as *const i32,
                    r[7] as *const i32,
                    r[8] as *const i32,
                    r[9] as *const i32,
                    case.t as i32,
                    case.rank as i32,
                    case.k as i32,
                    case.widths.len() as i32,
                    (case.max_loras + 1) as i32,
                    (case.rank * case.k) as i64,
                    case.scale,
                )
            };
            assert_eq!(rc, 0, "cuda lora_shrink rc={rc}");
            let rc = unsafe {
                culora::lora_expand(
                    stream.cu_stream() as *mut c_void,
                    r[4] as *const f32,
                    r[2] as *const u64,
                    r[3] as *mut u16,
                    r[5] as *const i32,
                    r[6] as *const i32,
                    r[7] as *const i32,
                    r[8] as *const i32,
                    r[9] as *const i32,
                    r[10] as *const i32,
                    r[11] as *const i32,
                    case.t as i32,
                    case.rank as i32,
                    max_n as i32,
                    case.widths.len() as i32,
                    (case.max_loras + 1) as i32,
                    case.sum_n() as i32,
                )
            };
            assert_eq!(rc, 0, "cuda lora_expand rc={rc}");
        }
        stream.synchronize().unwrap();
        let y: Vec<bf16> = stream.memcpy_dtov(&dev.y).unwrap();
        let buf: Vec<f32> = stream.memcpy_dtov(&dev.buf).unwrap();
        Some((y, buf))
    }

    pub fn cross_check(
        case: &Case,
        tag: &str,
        x: &[bf16],
        a: &[Vec<bf16>],
        b: &[Vec<bf16>],
        y_base: &[bf16],
        y_grouped: &Option<Vec<bf16>>,
        buf_grouped: &Option<Vec<f32>>,
        y_fused: &[bf16],
    ) {
        if let (Some(yg), Some(bg)) = (y_grouped, buf_grouped) {
            if let Some((cy, cbuf)) = run(case, x, a, b, y_base, false) {
                let (ym, yulp) = report_bits(
                    &format!("{tag}_grouped_wgpu_vs_cuda_y"),
                    &bits(yg),
                    &bits(&cy),
                );
                assert_eq!(
                    (ym, yulp),
                    (0, 0),
                    "{tag}: grouped y must be bit-exact vs cuda"
                );
                let mut fm = 0usize;
                let mut fmax = 0f32;
                for (g, c) in bg.iter().zip(cbuf.iter()) {
                    if g.to_bits() != c.to_bits() {
                        fm += 1;
                        fmax = fmax.max((g - c).abs());
                    }
                }
                eprintln!(
                    "{tag}_shrink_buffer_wgpu_vs_cuda: {fm}/{} f32 words differ, max_abs={fmax:e}",
                    bg.len()
                );
                assert_eq!(fm, 0, "{tag}: shrink f32 buffer must be bit-exact vs cuda");
            }
        }
        if let Some((cy, _)) = run(case, x, a, b, y_base, true) {
            let (ym, yulp) = report_bits(
                &format!("{tag}_fused_wgpu_vs_cuda_y"),
                &bits(y_fused),
                &bits(&cy),
            );
            assert_eq!(
                (ym, yulp),
                (0, 0),
                "{tag}: fused y must be bit-exact vs cuda"
            );
        }
    }

    #[test]
    fn meta_prepare_matches_the_cuda_side() {
        for (mapping, max_loras) in [
            (vec![2i32, -1, 0, 2, -1, 2], 4usize),
            (vec![-1, -1], 2),
            (vec![1, 1, 1], 2),
            ((0..33).map(|i| i % 4).collect::<Vec<_>>(), 4),
        ] {
            let w = LoraMeta::prepare(&mapping, max_loras);
            let c = culora::LoraKernelMeta::prepare(&mapping, max_loras);
            assert_eq!(w.token_indices_sorted, c.token_indices_sorted);
            assert_eq!(w.num_tokens_per_lora, c.num_tokens_per_lora);
            assert_eq!(w.lora_token_start_loc, c.lora_token_start_loc);
            assert_eq!(w.active_lora_ids, c.active_lora_ids);
            assert_eq!(w.no_lora, c.no_lora);
        }
    }
}

#[test]
fn one_adapter_prefill() {
    let Some(ctx) = ctx_or_skip("one_adapter_prefill") else {
        return;
    };
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
    run_wgpu(ctx, &case, "one_adapter_prefill");
}

#[test]
fn interleaved_adapters() {
    let Some(ctx) = ctx_or_skip("interleaved_adapters") else {
        return;
    };
    let mapping: Vec<i32> = (0..33).map(|i| i % 4).collect();
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
    run_wgpu(ctx, &case, "interleaved_adapters");
}

#[test]
fn mixed_adapted_and_unadapted() {
    let Some(ctx) = ctx_or_skip("mixed_adapted_and_unadapted") else {
        return;
    };
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
    let out = run_wgpu(ctx, &case, "mixed_adapted_and_unadapted");
    let buf = out.buf_grouped.unwrap();
    let y_grouped = out.y_grouped.unwrap();
    let sum_n = case.sum_n();
    for (tok, &slot) in mapping.iter().enumerate() {
        if slot != -1 {
            continue;
        }
        for s in 0..case.widths.len() {
            for r in 0..case.rank {
                let v = buf[(s * case.t + tok) * case.rank + r];
                assert_eq!(v, 0.0, "buffer row for -1 token {tok} slice {s} not zero");
            }
        }
        for n in 0..sum_n {
            assert_eq!(
                y_grouped[tok * sum_n + n],
                out.y_base[tok * sum_n + n],
                "grouped y row for -1 token {tok} was touched"
            );
            assert_eq!(
                out.y_fused[tok * sum_n + n],
                out.y_base[tok * sum_n + n],
                "fused y row for -1 token {tok} was touched"
            );
        }
    }
}

#[test]
fn decode_group_of_one() {
    let Some(ctx) = ctx_or_skip("decode_group_of_one") else {
        return;
    };
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
    run_wgpu(ctx, &case, "decode_group_of_one_t1");

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
    run_wgpu(ctx, &case, "decode_group_of_one_t4_distinct");
}

#[test]
fn large_group_prefill() {
    let Some(ctx) = ctx_or_skip("large_group_prefill") else {
        return;
    };
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
        win: None,
    };
    run_wgpu(ctx, &case, "large_group_prefill");
}

#[test]
fn rank_below_slot_rank() {
    let Some(ctx) = ctx_or_skip("rank_below_slot_rank") else {
        return;
    };
    let case = Case {
        t: 16,
        k: 64,
        rank: 16,
        widths: vec![32, 16],
        max_loras: 3,
        mapping: (0..16).map(|i| i % 2).collect(),
        slot_ranks: vec![16, 4, 16],
        scale: 1.0,
        seed: 7,
        win: None,
    };
    run_wgpu(ctx, &case, "rank_below_slot_rank");
}

#[test]
fn all_zero_slot() {
    let Some(ctx) = ctx_or_skip("all_zero_slot") else {
        return;
    };
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
    let out = run_wgpu(ctx, &case, "all_zero_slot");
    assert_eq!(
        out.y_grouped.unwrap(),
        out.y_base,
        "all-zero slot must leave grouped y bitwise unchanged"
    );
    assert_eq!(
        out.y_fused, out.y_base,
        "all-zero slot must leave fused y bitwise unchanged"
    );
}

#[test]
fn unequal_slice_widths_gqa() {
    let Some(ctx) = ctx_or_skip("unequal_slice_widths_gqa") else {
        return;
    };
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
        win: None,
    };
    run_wgpu(ctx, &case, "unequal_slice_widths_gqa");
}

#[test]
fn fused_max_rank_64_and_rank_below_slot() {
    let Some(ctx) = ctx_or_skip("fused_max_rank_64_and_rank_below_slot") else {
        return;
    };
    let case = Case {
        t: 16,
        k: 64,
        rank: 64,
        widths: vec![96, 32],
        max_loras: 3,
        mapping: (0..16).map(|i| i % 2).collect(),
        slot_ranks: vec![64, 4, 64],
        scale: 1.0,
        seed: 7,
        win: None,
    };
    run_wgpu(ctx, &case, "fused_max_rank_64");
}

#[test]
fn fused_wide_slice_needs_chunking() {
    let Some(ctx) = ctx_or_skip("fused_wide_slice_needs_chunking") else {
        return;
    };
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
    run_wgpu(ctx, &case, "fused_wide_slice_needs_chunking");
}

#[test]
fn fused_row_window_matches_full_slice() {
    let Some(ctx) = ctx_or_skip("fused_row_window_matches_full_slice") else {
        return;
    };
    let full = Case {
        t: 9,
        k: 96,
        rank: 16,
        widths: vec![128, 32, 32],
        max_loras: 2,
        mapping: vec![0, 1, -1, 0, 1, 0, -1, 1, 0],
        slot_ranks: vec![16, 16],
        scale: 1.0,
        seed: 10,
        win: None,
    };
    let full_out = run_wgpu(ctx, &full, "fused_window_reference_full");
    let full_y_base = build_y_base(&full);
    let sum_n = full.sum_n();

    for (win_off, win_len, tag) in [
        (0usize, 128usize, "fused_window_q"),
        (128, 32, "fused_window_k"),
        (160, 32, "fused_window_v"),
        (100, 60, "fused_window_straddles_slices"),
    ] {
        let mut case = full.clone();
        case.win = Some((win_off, win_len));
        let out = run_wgpu(ctx, &case, tag);
        let win_y_base = build_y_base(&case);
        for tok in 0..case.t {
            for c in 0..win_len {
                let g = out.y_fused[tok * win_len + c].to_f32();
                let base_delta = full_out.y_fused[tok * sum_n + win_off + c].to_f32()
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
fn shrink_expand_roundtrip_matches_grouped() {
    let Some(ctx) = ctx_or_skip("shrink_expand_roundtrip_matches_grouped") else {
        return;
    };
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
    let x = build_x(&case);
    let a = build_a(&case, case.seed);
    let b = build_b(&case, case.seed);
    let y_base = build_y_base(&case);
    let meta = LoraMeta::prepare(&case.mapping, case.max_loras);

    let x_bits = bits(&x);
    let a_bits: Vec<Vec<u16>> = a.iter().map(|v| bits(v)).collect();
    let b_bits: Vec<Vec<u16>> = b.iter().map(|v| bits(v)).collect();

    let mut buf = vec![0f32; case.widths.len() * case.t * case.rank];
    lora::lora_shrink(
        ctx,
        &x_bits,
        &slice_refs(&a_bits),
        &mut buf,
        &meta,
        case.t,
        case.rank,
        case.k,
        case.scale,
    )
    .expect("wgpu lora_shrink");

    let mut y_split = bits(&y_base);
    lora::lora_expand(
        ctx,
        &buf,
        &slice_refs(&b_bits),
        &mut y_split,
        &meta,
        &case.widths,
        case.t,
        case.rank,
        case.sum_n(),
    )
    .expect("wgpu lora_expand");

    let mut y_joint = bits(&y_base);
    let mut buf_joint = vec![0f32; buf.len()];
    lora::lora_grouped(
        ctx,
        &x_bits,
        &slice_refs(&a_bits),
        &slice_refs(&b_bits),
        &mut y_joint,
        &meta,
        &case.widths,
        case.t,
        case.rank,
        case.k,
        case.sum_n(),
        case.scale,
        Some(&mut buf_joint),
    )
    .expect("wgpu lora_grouped");

    assert_eq!(
        y_split, y_joint,
        "host-roundtrip shrink+expand must match the on-gpu two-launch path bitwise"
    );
    for (i, (s, j)) in buf.iter().zip(buf_joint.iter()).enumerate() {
        assert_eq!(s.to_bits(), j.to_bits(), "shrink buffer differs at {i}");
    }
    let want = oracle(&case, &x, &a, &b, &y_base);
    assert_close(
        &from_bits(&y_split),
        &want,
        2e-2,
        2e-2,
        "roundtrip_vs_oracle",
    );
}

#[test]
fn fused_rank_above_64_is_rejected() {
    let Some(ctx) = ctx_or_skip("fused_rank_above_64_is_rejected") else {
        return;
    };
    let meta = LoraMeta::prepare(&[0], 1);
    let a = vec![vec![0u16; 65 * 8]];
    let b = vec![vec![0u16; 4 * 65]];
    let mut y = vec![0u16; 4];
    let err = lora::lora_fused(
        ctx,
        &[0u16; 8],
        &slice_refs(&a),
        &slice_refs(&b),
        &mut y,
        &meta,
        &[4],
        1,
        65,
        8,
        0,
        4,
        4,
        1.0,
    )
    .expect_err("rank 65 must be rejected like the cuda host");
    assert!(format!("{err}").contains("FUSED_MAX_RANK"), "{err}");
}
