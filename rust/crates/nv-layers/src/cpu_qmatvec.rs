use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use candle_core::quantized::k_quants::{BlockQ4K, BlockQ8_0, BlockQ8K, GgmlType, QK_K, QK8_0};
use candle_core::quantized::{GgmlDType, QMatMul};
use candle_core::{DType, Device, Tensor};
use half::f16;
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arm {
    Avx512,
    Avx2,
    Scalar,
}

pub fn arms_available() -> Vec<Arm> {
    let mut arms = vec![Arm::Scalar];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            arms.push(Arm::Avx2);
        }
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vbmi")
            && is_x86_feature_detected!("avx512vnni")
        {
            arms.push(Arm::Avx512);
        }
    }
    arms
}

pub fn arm() -> Arm {
    static ARM: OnceLock<Arm> = OnceLock::new();
    *ARM.get_or_init(|| *arms_available().last().unwrap())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Q4K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; QK_K / 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Q8K {
    d: f32,
    qs: [i8; QK_K],
    bsums: [i16; QK_K / 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Q80 {
    d: u16,
    qs: [i8; QK8_0],
}

const _: () = assert!(size_of::<Q4K>() == size_of::<BlockQ4K>());

pub const ROWS_PER_TASK: usize = 32;

static CALLS: AtomicUsize = AtomicUsize::new(0);

pub fn calls() -> usize {
    CALLS.load(Ordering::Relaxed)
}

pub fn matmul(qm: &QMatMul, x: &Tensor) -> Result<Option<Tensor>> {
    let QMatMul::QTensor(qt) = qm else {
        return Ok(None);
    };
    let (n, k) = qt.shape().dims2()?;
    let (m, xk) = x.dims2()?;
    if !qt.device().is_cpu() || xk != k || m == 0 {
        return Ok(None);
    }
    let dtype = qt.dtype();
    if !matches!(dtype, GgmlDType::Q4K | GgmlDType::Q8_0) {
        return Ok(None);
    }
    CALLS.fetch_add(1, Ordering::Relaxed);
    let xf = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let data = qt.data()?;
    let mut out_t = vec![0f32; n * m];
    matmul_rows(arm(), dtype, &data, k, &xf, &mut out_t)?;
    let y = Tensor::from_vec(out_t, (n, m), &Device::Cpu)?;
    Ok(Some(if m == 1 {
        y.reshape((1, n))?
    } else {
        y.t()?.contiguous()?
    }))
}

fn cast_rows<T>(rows: &[u8], n: usize, kb: usize) -> Result<&[T]> {
    anyhow::ensure!(
        rows.len() == n * kb * size_of::<T>()
            && (rows.as_ptr() as usize).is_multiple_of(align_of::<T>()),
        "cpu_qmatvec: {} bytes do not hold {n} rows of {kb} blocks",
        rows.len()
    );
    Ok(unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const T, n * kb) })
}

fn quantize_rows<B: GgmlType + Send + Sync>(x: &[f32], k: usize) -> Vec<B> {
    let m = x.len() / k;
    let kb = k / B::BLCK_SIZE;
    let mut xq = vec![B::zeros(); m * kb];
    let each = |(q, x): (&mut [B], &[f32])| B::from_float(x, q);
    if m < INPUT_ROWS_PER_PASS_SHARE_ONE_UNPACK_OF_EACH_WEIGHT_BLOCK_AND_KEEP_ONE_ACCUMULATOR_PER_ROW {
        xq.chunks_mut(kb).zip(x.chunks(k)).for_each(each);
    } else {
        pool().install(|| xq.par_chunks_mut(kb).zip(x.par_chunks(k)).for_each(each));
    }
    xq
}

fn cast_blocks<B, Q>(v: &[B]) -> &[Q] {
    const { assert!(size_of::<B>() == size_of::<Q>() && align_of::<B>() == align_of::<Q>()) };
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const Q, v.len()) }
}

pub fn matmul_rows(
    arm: Arm,
    dtype: GgmlDType,
    rows: &[u8],
    k: usize,
    x: &[f32],
    out_t: &mut [f32],
) -> Result<()> {
    anyhow::ensure!(
        k > 0 && x.len().is_multiple_of(k),
        "cpu_qmatvec: x has {} elements, k = {k}",
        x.len()
    );
    let m = x.len() / k;
    anyhow::ensure!(
        m > 0 && out_t.len().is_multiple_of(m),
        "cpu_qmatvec: out holds {} values for m = {m}",
        out_t.len()
    );
    let n = out_t.len() / m;
    match dtype {
        GgmlDType::Q4K => {
            let kb = k / QK_K;
            let w: &[Q4K] = cast_rows(rows, n, kb)?;
            let xb = quantize_rows::<BlockQ8K>(x, k);
            let xq: &[Q8K] = cast_blocks(&xb);
            let xperm = permute_q8k(xq, arm);
            run_rows(out_t, m, kb, |r, o, j0, streaming| {
                let wr = &w[r];
                let (xq, xp) = (&xq[j0 * kb..], &xperm[j0 * kb * QK_K..]);
                match arm {
                    #[cfg(target_arch = "x86_64")]
                    Arm::Avx2 => unsafe { dot_q4k_avx2_rows(wr, xq, xp, o, streaming) },
                    #[cfg(target_arch = "x86_64")]
                    Arm::Avx512 => unsafe { dot_q4k_avx512_rows(wr, xq, xp, o, streaming) },
                    _ => {
                        for (j, y) in o.iter_mut().enumerate() {
                            *y = dot_q4k_scalar(wr, &xq[j * kb..(j + 1) * kb]);
                        }
                    }
                }
            });
        }
        GgmlDType::Q8_0 => {
            let kb = k / QK8_0;
            let w: &[Q80] = cast_rows(rows, n, kb)?;
            let xb = quantize_rows::<BlockQ8_0>(x, k);
            let xq: &[Q80] = cast_blocks(&xb);
            let xd: Vec<f32> = xq.iter().map(|y| f16::from_bits(y.d).to_f32()).collect();
            run_rows(out_t, m, kb, |r, o, j0, streaming| {
                let wr = &w[r];
                let xq = &xq[j0 * kb..];
                match arm {
                    #[cfg(target_arch = "x86_64")]
                    Arm::Avx2 | Arm::Avx512 => unsafe {
                        dot_q80_avx2_rows(wr, xq, &xd[j0 * kb..], o, streaming)
                    },
                    _ => {
                        for (j, y) in o.iter_mut().enumerate() {
                            *y = dot_q80_scalar(wr, &xq[j * kb..(j + 1) * kb]);
                        }
                    }
                }
            });
        }
        other => anyhow::bail!("cpu_qmatvec: no CPU kernel for {other:?}"),
    }
    Ok(())
}

pub const MATVEC_THREADS_ARE_HALF_THE_LOGICAL_CPUS_CAPPED_AT_16_BECAUSE_SMT_SIBLINGS_SHARE_THE_LOAD_PORTS: &str =
    "the decoder_step_floor_probe reads the 1.93 GB q4_k weight set in 48-50 ms on 8 or 16 \
     threads and 68 ms on all 32 SMT threads of a 16-core Zen 5: two siblings streaming the same \
     core's loads add contention, not bandwidth, so the row loop runs on its own pool of one \
     thread per physical core, at most sixteen";

pub const THREADS_PIN_TO_THEIR_CCD_AND_THE_LARGEST_L3_SERVES_THE_HOT_ROWS_FIRST: &str =
    "each pool thread is bound to the logical cpus of one CCD, the CCDs ordered by L3 size, so \
     a row block served by the same CCD on every call stays in that CCD's L3 across calls; a \
     Placement names the share of leading rows that is hot and whether the rest streams with \
     non-temporal prefetch, hot rows queue to the large-L3 threads first and cold rows to the \
     others, and every thread drains the other queue when its own is empty";

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Placement {
    pub hot_share: f32,
    pub streaming: bool,
}

thread_local! {
    static PLACEMENT: Cell<Placement> = const { Cell::new(Placement { hot_share: 0.0, streaming: false }) };
}

pub fn with_placement<T>(p: Placement, f: impl FnOnce() -> T) -> T {
    let prev = PLACEMENT.replace(p);
    let out = f();
    PLACEMENT.set(prev);
    out
}

struct Ccd {
    l3_bytes: usize,
    cpus: Vec<usize>,
    cores: usize,
}

fn cpu_list(path: &str) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for part in std::fs::read_to_string(path).ok()?.trim().split(',') {
        let (a, b) = part.split_once('-').unwrap_or((part, part));
        out.extend(a.parse::<usize>().ok()?..=b.parse::<usize>().ok()?);
    }
    Some(out)
}

fn ccds_by_l3_size() -> Vec<Ccd> {
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut by_l3: BTreeMap<Vec<usize>, (usize, usize)> = BTreeMap::new();
    for cpu in 0..logical {
        let base = format!("/sys/devices/system/cpu/cpu{cpu}/");
        let (Some(shared), Some(siblings)) = (
            cpu_list(&format!("{base}cache/index3/shared_cpu_list")),
            cpu_list(&format!("{base}topology/thread_siblings_list")),
        ) else {
            return Vec::new();
        };
        let size = std::fs::read_to_string(format!("{base}cache/index3/size"))
            .ok()
            .and_then(|s| s.trim().trim_end_matches('K').parse::<usize>().ok())
            .unwrap_or(0)
            << 10;
        let e = by_l3.entry(shared).or_insert((size, 0));
        e.1 += (siblings.first() == Some(&cpu)) as usize;
    }
    let mut ccds: Vec<Ccd> = by_l3
        .into_iter()
        .map(|(cpus, (l3_bytes, cores))| Ccd { l3_bytes, cpus, cores })
        .collect();
    ccds.sort_by_key(|c| std::cmp::Reverse(c.l3_bytes));
    ccds
}

#[cfg(target_os = "linux")]
fn pin_to(cpus: &[usize]) {
    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
    }
    let mut mask = [0u64; 16];
    for &c in cpus.iter().filter(|&&c| c < 1024) {
        mask[c / 64] |= 1 << (c % 64);
    }
    unsafe { sched_setaffinity(0, size_of_val(&mask), mask.as_ptr()) };
}

#[cfg(not(target_os = "linux"))]
fn pin_to(_cpus: &[usize]) {}

struct Pool {
    pool: rayon::ThreadPool,
    hot_threads: usize,
    hot_l3_bytes: usize,
}

fn pool_with_topology() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let _ = MATVEC_THREADS_ARE_HALF_THE_LOGICAL_CPUS_CAPPED_AT_16_BECAUSE_SMT_SIBLINGS_SHARE_THE_LOAD_PORTS;
        let _ = THREADS_PIN_TO_THEIR_CCD_AND_THE_LARGEST_L3_SERVES_THE_HOT_ROWS_FIRST;
        let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let ccds = ccds_by_l3_size();
        let cores: usize = ccds.iter().map(|c| c.cores).sum();
        let threads = if cores == 0 { logical / 2 } else { cores }.clamp(1, 16);
        let masks: Vec<Vec<usize>> = ccds
            .iter()
            .flat_map(|c| std::iter::repeat_n(c.cpus.clone(), c.cores))
            .take(threads)
            .collect();
        let split = ccds.len() >= 2;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .start_handler(move |i| {
                if let Some(m) = masks.get(i) {
                    pin_to(m);
                }
            })
            .build()
            .expect("cpu_qmatvec thread pool");
        Pool {
            pool,
            hot_threads: if split { ccds[0].cores.min(threads) } else { threads },
            hot_l3_bytes: if split { ccds[0].l3_bytes } else { 0 },
        }
    })
}

pub fn pool() -> &'static rayon::ThreadPool {
    &pool_with_topology().pool
}

pub fn hot_l3_bytes() -> usize {
    pool_with_topology().hot_l3_bytes
}

pub fn par_chunks_mut(out: &mut [f32], chunk: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    pool().install(|| out.par_chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c)))
}

struct OutPtr(*mut f32);

unsafe impl Sync for OutPtr {}

impl OutPtr {
    unsafe fn rows<'a>(&self, at: usize, len: usize) -> &'a mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(at), len) }
    }
}

fn run_rows(
    out_t: &mut [f32],
    m: usize,
    kb: usize,
    f: impl Fn(std::ops::Range<usize>, &mut [f32], usize, bool) + Sync,
) {
    const J: usize = INPUT_ROWS_PER_PASS_SHARE_ONE_UNPACK_OF_EACH_WEIGHT_BLOCK_AND_KEEP_ONE_ACCUMULATOR_PER_ROW;
    let n = out_t.len() / m;
    let p = PLACEMENT.get();
    let chunks = n.div_ceil(ROWS_PER_TASK);
    let hot_chunks = ((n as f32 * p.hot_share) as usize / ROWS_PER_TASK).min(chunks);
    let hot_rows = hot_chunks * ROWS_PER_TASK;
    let queues = [(AtomicUsize::new(0), hot_chunks), (AtomicUsize::new(hot_chunks), chunks)];
    let pool = pool_with_topology();
    let out = OutPtr(out_t.as_mut_ptr());
    let drain = |thread: usize| {
        let home = (thread >= pool.hot_threads) as usize;
        for (next, end) in [&queues[home], &queues[1 - home]] {
            loop {
                let c = next.fetch_add(1, Ordering::Relaxed);
                if c >= *end {
                    break;
                }
                let rows = c * ROWS_PER_TASK..((c + 1) * ROWS_PER_TASK).min(n);
                let streaming = p.streaming && m <= J && rows.start >= hot_rows;
                for j0 in (0..m).step_by(J) {
                    let j1 = (j0 + J).min(m);
                    for r in rows.clone() {
                        f(r * kb..(r + 1) * kb, unsafe { out.rows(r * m + j0, j1 - j0) }, j0, streaming);
                    }
                }
            }
        }
    };
    let _ = A_WORKER_PER_THREAD_IS_SPAWNED_NOT_BROADCAST_SO_A_DESCHEDULED_THREAD_NEVER_HOLDS_THE_CALL;
    pool.pool.install(|| {
        rayon::scope(|s| {
            for _ in 0..pool.pool.current_num_threads() {
                s.spawn(|_| drain(rayon::current_thread_index().unwrap_or(0)));
            }
        })
    });
}

pub const A_WORKER_PER_THREAD_IS_SPAWNED_NOT_BROADCAST_SO_A_DESCHEDULED_THREAD_NEVER_HOLDS_THE_CALL: &str =
    "on a loaded box a pool thread can sit descheduled for a timeslice; a broadcast waits for      it, a spawned worker is stolen by whichever thread is running, and a worker that finds both      queues empty returns at once";

pub const STREAMING_ROWS_PREFETCH_NON_TEMPORAL_ONE_KIB_AHEAD_SO_THEY_SKIP_THE_L3_THAT_HOLDS_THE_HOT_ROWS: usize =
    1024;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prefetch_stream(p: *const u8, bytes: usize) {
    unsafe {
        let mut off = 0;
        while off < bytes {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_NTA }>(
                p.add(STREAMING_ROWS_PREFETCH_NON_TEMPORAL_ONE_KIB_AHEAD_SO_THEY_SKIP_THE_L3_THAT_HOLDS_THE_HOT_ROWS + off)
                    as *const i8,
            );
            off += 64;
        }
    }
}

fn scales_mins(sc: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let u = |i: usize| u32::from_le_bytes([sc[4 * i], sc[4 * i + 1], sc[4 * i + 2], sc[4 * i + 3]]);
    let (u0, u1, u2) = (u(0), u(1), u(2));
    let t3 = ((u2 >> 4) & 0x0f0f0f0f) | (((u1 >> 6) & 0x03030303) << 4);
    let t1 = (u2 & 0x0f0f0f0f) | (((u0 >> 6) & 0x03030303) << 4);
    let t2 = u1 & 0x3f3f3f3f;
    let t0 = u0 & 0x3f3f3f3f;
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    scales[..4].copy_from_slice(&t0.to_le_bytes());
    scales[4..].copy_from_slice(&t1.to_le_bytes());
    mins[..4].copy_from_slice(&t2.to_le_bytes());
    mins[4..].copy_from_slice(&t3.to_le_bytes());
    (scales, mins)
}

fn mins_dot_bsums(mins: &[u8; 8], y: &Q8K) -> i32 {
    (0..QK_K / 16)
        .map(|j| y.bsums[j] as i32 * mins[j / 2] as i32)
        .sum()
}

fn q4k_nibble(w: &Q4K, idx: usize) -> u8 {
    let raw = w.qs[32 * (idx / 64) + idx % 32];
    if idx % 64 < 32 { raw & 0xF } else { raw >> 4 }
}

fn finish_q4k(sumf: f32, sums: [f32; 8]) -> f32 {
    sumf + sums.iter().sum::<f32>()
}

fn dot_q4k_scalar(row: &[Q4K], xq: &[Q8K]) -> f32 {
    let mut sums = [0f32; 8];
    let mut sumf = 0f32;
    for (w, y) in row.iter().zip(xq) {
        let (scales, mins) = scales_mins(&w.scales);
        let mut aux32 = [0i32; 8];
        for (s, scale) in scales.iter().enumerate() {
            for g in 0..4 {
                for l in 0..8 {
                    let idx = 32 * s + 8 * g + l;
                    let p = y.qs[idx] as i16 * q4k_nibble(w, idx) as i16;
                    aux32[l] += *scale as i32 * p as i32;
                }
            }
        }
        let d = f16::from_bits(w.d).to_f32() * y.d;
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
        let dmin = f16::from_bits(w.dmin).to_f32() * y.d;
        sumf -= dmin * mins_dot_bsums(&mins, y) as f32;
    }
    finish_q4k(sumf, sums)
}

fn permute_q8k(xq: &[Q8K], arm: Arm) -> Vec<u8> {
    let sub_offset = |s: usize| match arm {
        Arm::Avx512 => 128 * (s / 4) + 32 * [0, 2, 1, 3][s % 4],
        _ => 32 * s,
    };
    let mut out = vec![0u8; xq.len() * QK_K];
    for (b, y) in xq.iter().enumerate() {
        for s in 0..8 {
            let dst = &mut out[b * QK_K + sub_offset(s)..][..32];
            for k in 0..8 {
                for j in 0..4 {
                    dst[4 * k + j] = y.qs[32 * s + k + 8 * j] as u8;
                }
            }
        }
    }
    out
}

fn dot_q80_scalar(row: &[Q80], xq: &[Q80]) -> f32 {
    let mut sumf = 0f32;
    dot_q80_scalar_into(row, xq, &mut sumf);
    sumf
}

fn dot_q80_scalar_into(row: &[Q80], xq: &[Q80], sumf: &mut f32) {
    for (w, y) in row.iter().zip(xq) {
        let sum_i: i32 =
            w.qs.iter()
                .zip(&y.qs)
                .map(|(&a, &b)| a as i32 * b as i32)
                .sum();
        *sumf += sum_i as f32 * f16::from_bits(w.d).to_f32() * f16::from_bits(y.d).to_f32();
    }
}

pub const INPUT_ROWS_PER_PASS_SHARE_ONE_UNPACK_OF_EACH_WEIGHT_BLOCK_AND_KEEP_ONE_ACCUMULATOR_PER_ROW: usize = 8;

pub const MINS_DOTS_OF_FOUR_OR_MORE_INPUT_ROWS_REDUCE_AS_ONE_TRANSPOSE: usize = 4;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mins_lanes_avx2(mins: &[u8; 8]) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let m = _mm_set1_epi64x(i64::from_le_bytes(*mins));
    _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(m, m))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mins_dot_bsums_avx2(m16: std::arch::x86_64::__m256i, y: &Q8K) -> i32 {
    unsafe {
        use std::arch::x86_64::*;
        let b = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
        let p = _mm256_madd_epi16(b, m16);
        let s = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
        let s = _mm_hadd_epi32(s, s);
        _mm_cvtsi128_si32(_mm_hadd_epi32(s, s))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4k_row_finish(sums: &[std::arch::x86_64::__m256], sumf: &[f32], o: &mut [f32]) {
    unsafe {
        for ((y, s), f) in o.iter_mut().zip(sums).zip(sumf) {
            let mut v = [0f32; 8];
            std::arch::x86_64::_mm256_storeu_ps(v.as_mut_ptr(), *s);
            *y = finish_q4k(*f, v);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4k_avx2_rows(row: &[Q4K], xq: &[Q8K], xperm: &[u8], out: &mut [f32], streaming: bool) {
    unsafe {
        use std::arch::x86_64::*;
        const J: usize = INPUT_ROWS_PER_PASS_SHARE_ONE_UNPACK_OF_EACH_WEIGHT_BLOCK_AND_KEEP_ONE_ACCUMULATOR_PER_ROW;
        let shuf = _mm256_setr_epi8(
            0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12,
            5, 13, 6, 14, 7, 15,
        );
        let m4 = _mm256_set1_epi8(0xF);
        let kb = row.len();
        for (c, o) in out.chunks_mut(J).enumerate() {
            let mut sums = [_mm256_setzero_ps(); J];
            let mut sumf = [0f32; J];
            for (b, w) in row.iter().enumerate() {
                if streaming {
                    prefetch_stream(w as *const Q4K as *const u8, size_of::<Q4K>());
                }
                let (scales, mins) = scales_mins(&w.scales);
                let m16 = mins_lanes_avx2(&mins);
                let (wd, wmin) = (
                    f16::from_bits(w.d).to_f32(),
                    f16::from_bits(w.dmin).to_f32(),
                );
                let mut lo = [_mm256_setzero_si256(); 4];
                let mut hi = lo;
                for i in 0..4 {
                    let raw = _mm256_loadu_si256(w.qs.as_ptr().add(32 * i) as *const __m256i);
                    let t = _mm256_shuffle_epi8(raw, shuf);
                    let u = _mm256_permute4x64_epi64(t, 0x4E);
                    let perm = _mm256_permute2x128_si256(
                        _mm256_unpacklo_epi16(t, u),
                        _mm256_unpackhi_epi16(t, u),
                        0x20,
                    );
                    lo[i] = _mm256_and_si256(perm, m4);
                    hi[i] = _mm256_and_si256(_mm256_srli_epi16(perm, 4), m4);
                }
                for j in 0..o.len() {
                    let y = &xq[(c * J + j) * kb + b];
                    let xp = xperm.as_ptr().add(((c * J + j) * kb + b) * QK_K);
                    let mut acc = _mm256_setzero_si256();
                    for i in 0..4 {
                        let qa = _mm256_loadu_si256(xp.add(64 * i) as *const __m256i);
                        let qb = _mm256_loadu_si256(xp.add(64 * i + 32) as *const __m256i);
                        let pa = _mm256_maddubs_epi16(lo[i], qa);
                        let pb = _mm256_maddubs_epi16(hi[i], qb);
                        acc = _mm256_add_epi32(
                            acc,
                            _mm256_madd_epi16(pa, _mm256_set1_epi16(scales[2 * i] as i16)),
                        );
                        acc = _mm256_add_epi32(
                            acc,
                            _mm256_madd_epi16(pb, _mm256_set1_epi16(scales[2 * i + 1] as i16)),
                        );
                    }
                    let d = wd * y.d;
                    sums[j] = _mm256_add_ps(
                        sums[j],
                        _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(acc)),
                    );
                    let dmin = wmin * y.d;
                    sumf[j] -= dmin * mins_dot_bsums_avx2(m16, y) as f32;
                }
            }
            q4k_row_finish(&sums, &sumf, o);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vnni")]
#[inline]
unsafe fn dpwssd(
    acc: std::arch::x86_64::__m512i,
    a: std::arch::x86_64::__m512i,
    b: std::arch::x86_64::__m512i,
) -> std::arch::x86_64::__m512i {
    let mut acc = acc;
    unsafe {
        std::arch::asm!(
            "vpdpwssd {acc}, {a}, {b}",
            acc = inout(zmm_reg) acc,
            a = in(zmm_reg) a,
            b = in(zmm_reg) b,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vnni")]
unsafe fn dot_q4k_avx512_rows(row: &[Q4K], xq: &[Q8K], xperm: &[u8], out: &mut [f32], streaming: bool) {
    unsafe {
        use std::arch::x86_64::*;
        const J: usize = INPUT_ROWS_PER_PASS_SHARE_ONE_UNPACK_OF_EACH_WEIGHT_BLOCK_AND_KEEP_ONE_ACCUMULATOR_PER_ROW;
        let mut idx = [0u8; 64];
        for g in 0..2 {
            for k in 0..8 {
                for j in 0..4 {
                    idx[32 * g + 4 * k + j] = (32 * g + k + 8 * j) as u8;
                }
            }
        }
        let idx = _mm512_loadu_si512(idx.as_ptr() as *const __m512i);
        let m4 = _mm512_set1_epi8(0xF);
        let pair = |a: u8, b: u8| {
            _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi16(a as i16)),
                _mm256_set1_epi16(b as i16),
                1,
            )
        };
        let kb = row.len();
        for (c, o) in out.chunks_mut(J).enumerate() {
            let mut sums = [_mm256_setzero_ps(); J];
            let mut sumf = [0f32; J];
            let mut sumf_lanes = _mm256_setzero_ps();
            let lanes = o.len() >= MINS_DOTS_OF_FOUR_OR_MORE_INPUT_ROWS_REDUCE_AS_ONE_TRANSPOSE;
            let at: [usize; J] = std::array::from_fn(|j| (c * J + j.min(o.len() - 1)) * kb);
            for (b, w) in row.iter().enumerate() {
                if streaming {
                    prefetch_stream(w as *const Q4K as *const u8, size_of::<Q4K>());
                }
                let (scales, mins) = scales_mins(&w.scales);
                let m16 = mins_lanes_avx2(&mins);
                let (wd, wmin) = (
                    f16::from_bits(w.d).to_f32(),
                    f16::from_bits(w.dmin).to_f32(),
                );
                let sc = [
                    pair(scales[0], scales[2]),
                    pair(scales[1], scales[3]),
                    pair(scales[4], scales[6]),
                    pair(scales[5], scales[7]),
                ];
                let mut lo = [_mm512_setzero_si512(); 2];
                let mut hi = lo;
                for i in 0..2 {
                    let raw = _mm512_loadu_si512(w.qs.as_ptr().add(64 * i) as *const __m512i);
                    let perm = _mm512_permutexvar_epi8(idx, raw);
                    lo[i] = _mm512_and_si512(perm, m4);
                    hi[i] = _mm512_and_si512(_mm512_srli_epi16(perm, 4), m4);
                }
                let mut mind = [_mm256_setzero_si256(); J];
                let mut yd = [0f32; J];
                for j in 0..o.len() {
                    let y = &xq[at[j] + b];
                    let xp = xperm.as_ptr().add((at[j] + b) * QK_K);
                    let mut acc = _mm512_setzero_si512();
                    for i in 0..2 {
                        let qa = _mm512_loadu_si512(xp.add(128 * i) as *const __m512i);
                        let qb = _mm512_loadu_si512(xp.add(128 * i + 64) as *const __m512i);
                        let pa = _mm512_maddubs_epi16(lo[i], qa);
                        let pb = _mm512_maddubs_epi16(hi[i], qb);
                        acc = dpwssd(acc, pa, sc[2 * i]);
                        acc = dpwssd(acc, pb, sc[2 * i + 1]);
                    }
                    let aux = _mm256_add_epi32(
                        _mm512_castsi512_si256(acc),
                        _mm512_extracti64x4_epi64(acc, 1),
                    );
                    let d = wd * y.d;
                    sums[j] = _mm256_add_ps(
                        sums[j],
                        _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(aux)),
                    );
                    if lanes {
                        yd[j] = y.d;
                        mind[j] = _mm256_madd_epi16(
                            _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i),
                            m16,
                        );
                    } else {
                        let dmin = wmin * y.d;
                        sumf[j] -= dmin * mins_dot_bsums_avx2(m16, y) as f32;
                    }
                }
                if lanes {
                    let h = |a: __m256i, b: __m256i| _mm256_hadd_epi32(a, b);
                    let q0 = h(h(mind[0], mind[1]), h(mind[2], mind[3]));
                    let q1 = h(h(mind[4], mind[5]), h(mind[6], mind[7]));
                    let dots = _mm256_add_epi32(
                        _mm256_permute2x128_si256(q0, q1, 0x20),
                        _mm256_permute2x128_si256(q0, q1, 0x31),
                    );
                    let dmin = _mm256_mul_ps(_mm256_set1_ps(wmin), _mm256_loadu_ps(yd.as_ptr()));
                    sumf_lanes = _mm256_sub_ps(sumf_lanes, _mm256_mul_ps(dmin, _mm256_cvtepi32_ps(dots)));
                }
            }
            if lanes {
                _mm256_storeu_ps(sumf.as_mut_ptr(), sumf_lanes);
            }
            q4k_row_finish(&sums, &sumf, o);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q80_avx2_rows(row: &[Q80], xq: &[Q80], xd: &[f32], out: &mut [f32], streaming: bool) {
    unsafe {
        use std::arch::x86_64::*;
        const J: usize = INPUT_ROWS_PER_PASS_SHARE_ONE_UNPACK_OF_EACH_WEIGHT_BLOCK_AND_KEEP_ONE_ACCUMULATOR_PER_ROW;
        let kb = row.len();
        let quads = kb / 4 * 4;
        for (c, o) in out.chunks_mut(J).enumerate() {
            let mut sumf = [0f32; J];
            for b in (0..quads).step_by(4) {
                if streaming {
                    prefetch_stream(row.as_ptr().add(b) as *const u8, 4 * size_of::<Q80>());
                }
                let wa: [__m256i; 4] = std::array::from_fn(|t| {
                    _mm256_loadu_si256(row[b + t].qs.as_ptr() as *const __m256i)
                });
                let wabs: [__m256i; 4] = std::array::from_fn(|t| _mm256_abs_epi8(wa[t]));
                let wd: [f32; 4] = std::array::from_fn(|t| f16::from_bits(row[b + t].d).to_f32());
                for (j, f) in sumf.iter_mut().enumerate().take(o.len()) {
                    let at = (c * J + j) * kb + b;
                    let v: [__m256i; 4] = std::array::from_fn(|t| {
                        let y = _mm256_loadu_si256(xq[at + t].qs.as_ptr() as *const __m256i);
                        let p = _mm256_maddubs_epi16(wabs[t], _mm256_sign_epi8(y, wa[t]));
                        _mm256_madd_epi16(p, _mm256_set1_epi16(1))
                    });
                    let h = _mm256_hadd_epi32(
                        _mm256_hadd_epi32(v[0], v[1]),
                        _mm256_hadd_epi32(v[2], v[3]),
                    );
                    let s4 =
                        _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
                    let mut si = [0i32; 4];
                    _mm_storeu_si128(si.as_mut_ptr() as *mut __m128i, s4);
                    for t in 0..4 {
                        *f += si[t] as f32 * wd[t] * xd[at + t];
                    }
                }
            }
            for (j, (y, f)) in o.iter_mut().zip(sumf).enumerate() {
                let mut f = f;
                let at = (c * J + j) * kb;
                dot_q80_scalar_into(&row[quads..], &xq[at + quads..at + kb], &mut f);
                *y = f;
            }
        }
    }
}

pub const BF16_ROWS_EXPAND_EACH_WEIGHT_ONCE_INTO_F32_LANES_AND_KEEP_ONE_ACCUMULATOR_PER_INPUT_ROW: &str =
    "a bf16 weight is exact in f32, so the dot product is f32 fused multiply-adds over the \
     expanded lanes with no quantization of the input; sixteen weights load once per pass and \
     feed one accumulator per input row, so M rows cost one weight read like the k-quant rows; \
     f32 rows take the same loop over unexpanded lanes";

pub trait F32Lanes: Copy + Send + Sync {
    fn to_f32(self) -> f32;
    #[cfg(target_arch = "x86_64")]
    unsafe fn load16(p: *const Self) -> std::arch::x86_64::__m512;
    #[cfg(target_arch = "x86_64")]
    unsafe fn load8(p: *const Self) -> std::arch::x86_64::__m256;
}

impl F32Lanes for u16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from_bits((self as u32) << 16)
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load16(p: *const Self) -> std::arch::x86_64::__m512 {
        unsafe {
            use std::arch::x86_64::*;
            _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm256_loadu_si256(p as *const __m256i)),
                16,
            ))
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load8(p: *const Self) -> std::arch::x86_64::__m256 {
        unsafe {
            use std::arch::x86_64::*;
            _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128(p as *const __m128i)),
                16,
            ))
        }
    }
}

impl F32Lanes for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load16(p: *const Self) -> std::arch::x86_64::__m512 {
        unsafe { std::arch::x86_64::_mm512_loadu_ps(p) }
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn load8(p: *const Self) -> std::arch::x86_64::__m256 {
        unsafe { std::arch::x86_64::_mm256_loadu_ps(p) }
    }
}

pub fn matmul_rows_bf16(
    arm: Arm,
    rows: &[u16],
    k: usize,
    x: &[f32],
    out_t: &mut [f32],
) -> Result<()> {
    matmul_rows_lanes(arm, rows, k, x, out_t)
}

pub fn matmul_rows_f32(
    arm: Arm,
    rows: &[f32],
    k: usize,
    x: &[f32],
    out_t: &mut [f32],
) -> Result<()> {
    matmul_rows_lanes(arm, rows, k, x, out_t)
}

fn matmul_rows_lanes<W: F32Lanes>(
    arm: Arm,
    rows: &[W],
    k: usize,
    x: &[f32],
    out_t: &mut [f32],
) -> Result<()> {
    let _ = BF16_ROWS_EXPAND_EACH_WEIGHT_ONCE_INTO_F32_LANES_AND_KEEP_ONE_ACCUMULATOR_PER_INPUT_ROW;
    anyhow::ensure!(
        k > 0 && x.len().is_multiple_of(k),
        "cpu_qmatvec lanes: x has {} elements, k = {k}",
        x.len()
    );
    let m = x.len() / k;
    anyhow::ensure!(
        m > 0 && out_t.len().is_multiple_of(m),
        "cpu_qmatvec lanes: out holds {} values for m = {m}",
        out_t.len()
    );
    let n = out_t.len() / m;
    anyhow::ensure!(
        rows.len() == n * k,
        "cpu_qmatvec lanes: {} weights do not hold {n} rows of {k}",
        rows.len()
    );
    run_rows(out_t, m, k, |r, o, j0, streaming| {
        let (w, x) = (&rows[r], &x[j0 * k..]);
        match arm {
            #[cfg(target_arch = "x86_64")]
            Arm::Avx512 => unsafe { dot_lanes_avx512_rows(w, x, o, streaming) },
            #[cfg(target_arch = "x86_64")]
            Arm::Avx2 => unsafe { dot_lanes_avx2_rows(w, x, o, streaming) },
            _ => dot_lanes_scalar_rows(w, x, o),
        }
    });
    Ok(())
}

fn dot_lanes_scalar_rows<W: F32Lanes>(w: &[W], x: &[f32], out: &mut [f32]) {
    let k = w.len();
    for (j, o) in out.iter_mut().enumerate() {
        let xr = &x[j * k..(j + 1) * k];
        let mut lanes = [0f32; 16];
        for (i, (&wi, &xi)) in w.iter().zip(xr).enumerate() {
            lanes[i % 16] = wi.to_f32().mul_add(xi, lanes[i % 16]);
        }
        *o = lanes.iter().sum();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_lanes_avx512_rows<W: F32Lanes>(w: &[W], x: &[f32], out: &mut [f32], streaming: bool) {
    unsafe {
        use std::arch::x86_64::*;
        const J: usize = INPUT_ROWS_PER_PASS_SHARE_ONE_UNPACK_OF_EACH_WEIGHT_BLOCK_AND_KEEP_ONE_ACCUMULATOR_PER_ROW;
        let k = w.len();
        let main = k / 16 * 16;
        for (c, o) in out.chunks_mut(J).enumerate() {
            let mut acc = [_mm512_setzero_ps(); J];
            for i in (0..main).step_by(16) {
                if streaming {
                    prefetch_stream(w.as_ptr().add(i) as *const u8, 16 * size_of::<W>());
                }
                let wv = W::load16(w.as_ptr().add(i));
                for j in 0..o.len() {
                    acc[j] = _mm512_fmadd_ps(
                        wv,
                        _mm512_loadu_ps(x.as_ptr().add((c * J + j) * k + i)),
                        acc[j],
                    );
                }
            }
            for (j, y) in o.iter_mut().enumerate() {
                let mut s = _mm512_reduce_add_ps(acc[j]);
                for i in main..k {
                    s = w[i].to_f32().mul_add(x[(c * J + j) * k + i], s);
                }
                *y = s;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_lanes_avx2_rows<W: F32Lanes>(w: &[W], x: &[f32], out: &mut [f32], streaming: bool) {
    unsafe {
        use std::arch::x86_64::*;
        const J: usize = INPUT_ROWS_PER_PASS_SHARE_ONE_UNPACK_OF_EACH_WEIGHT_BLOCK_AND_KEEP_ONE_ACCUMULATOR_PER_ROW;
        let k = w.len();
        let main = k / 8 * 8;
        for (c, o) in out.chunks_mut(J).enumerate() {
            let mut acc = [_mm256_setzero_ps(); J];
            for i in (0..main).step_by(8) {
                if streaming {
                    prefetch_stream(w.as_ptr().add(i) as *const u8, 8 * size_of::<W>());
                }
                let wv = W::load8(w.as_ptr().add(i));
                for j in 0..o.len() {
                    acc[j] = _mm256_fmadd_ps(
                        wv,
                        _mm256_loadu_ps(x.as_ptr().add((c * J + j) * k + i)),
                        acc[j],
                    );
                }
            }
            for (j, y) in o.iter_mut().enumerate() {
                let mut v = [0f32; 8];
                _mm256_storeu_ps(v.as_mut_ptr(), acc[j]);
                let mut s = v.iter().sum::<f32>();
                for i in main..k {
                    s = w[i].to_f32().mul_add(x[(c * J + j) * k + i], s);
                }
                *y = s;
            }
        }
    }
}
