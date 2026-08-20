#![allow(dead_code)]
#![allow(unused_imports)]

use half::bf16;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
#[cfg(feature = "cuda")]
#[cfg(feature = "wgpu")]
use nv_kernels::cuda;
#[cfg(feature = "wgpu")]
use nv_kernels::wgpu_backend::device::WgpuContext;
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "wgpu")]
use nv_kernels::wgpu_backend::device::shared_or_reason;

pub fn assert_close(got: &[bf16], want: &[bf16], rtol: f32, atol: f32, tag: &str) {
    assert_eq!(got.len(), want.len(), "{tag}: length mismatch");
    let mut max_err = 0f32;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let gf = g.to_f32();
        let wf = w.to_f32();
        let err = (gf - wf).abs();
        let tol = atol + rtol * wf.abs();
        assert!(
            err <= tol,
            "{tag}: mismatch at {i}: got {gf} want {wf} err {err} tol {tol}"
        );
        if err > max_err {
            max_err = err;
        }
    }
    eprintln!("{tag}: max abs err {max_err}");
}

#[cfg(feature = "cuda")]
pub fn assert_u16_bits(name: &str, a: &[u16], b: &[u16]) {
    assert_eq!(a.len(), b.len(), "{name}: length");
    let mut diff = 0usize;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            diff += 1;
            if first.is_none() {
                first = Some((i, *x, *y));
            }
        }
    }
    assert_eq!(
        diff,
        0,
        "{name}: {diff}/{} bf16 words differ, first {first:?}",
        a.len()
    );
}

#[cfg(feature = "cuda")]
#[cfg(feature = "wgpu")]
pub fn backends(test: &str) -> Option<(Arc<CudaStream>, &'static WgpuContext)> {
    let cu = match CudaContext::new(0) {
        Ok(c) => c.default_stream(),
        Err(e) => {
            if require_parity_opt_in() {
                panic!("{test}: no CUDA device 0: {e}");
            }
            eprintln!("{test}: SKIP no CUDA device 0: {e}");
            return None;
        }
    };
    let wg = match WgpuContext::shared() {
        Ok(ctx) if ctx.qualify().qualified => ctx,
        Ok(ctx) => {
            if require_parity_opt_in() {
                panic!(
                    "{test}: wgpu adapter not qualified: {:?}",
                    ctx.qualify().reason
                );
            }
            eprintln!("{test}: SKIP adapter not qualified");
            return None;
        }
        Err(e) => {
            if require_parity_opt_in() {
                panic!("{test}: no wgpu adapter: {e}");
            }
            eprintln!("{test}: SKIP no wgpu adapter: {e}");
            return None;
        }
    };
    eprintln!("{test}: cuda dev0 vs {}", wg.summary());
    Some((cu, wg))
}

#[cfg(feature = "wgpu")]
pub fn bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| bf16::from_f32(*x).to_bits()).collect()
}

#[cfg(feature = "wgpu")]
pub fn ctx(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(c) if c.qualify().qualified => {
            eprintln!("{test}: {}", c.summary());
            Some(c)
        }
        Ok(c) => {
            if require() {
                panic!("{test}: adapter not qualified: {:?}", c.qualify().reason);
            }
            eprintln!("{test}: SKIP adapter not qualified");
            None
        }
        Err(e) => {
            if require() {
                panic!("{test}: no wgpu adapter: {e}");
            }
            eprintln!("{test}: SKIP no wgpu adapter: {e}");
            None
        }
    }
}

#[cfg(feature = "wgpu")]
pub fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("{test}: {}", ctx.summary());
            let st = ctx.qualify();
            if !st.qualified {
                if require() {
                    panic!("adapter not qualified: {:?}", st.reason);
                }
                eprintln!("{test}: SKIP adapter not qualified: {:?}", st.reason);
                return None;
            }
            Some(ctx)
        }
        Err(e) => {
            if require() {
                panic!(
                    "{test}: no wgpu adapter: {e}. This gate refuses to report \
                     success without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to \
                     skip on purpose."
                );
            }
            eprintln!("{test}: SKIP no wgpu adapter: {e}");
            None
        }
    }
}

#[cfg(feature = "wgpu")]
pub fn d(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

#[cfg(feature = "wgpu")]
pub fn dot8(word: u32, x: &[u16], kb: usize, acc_in: f32) -> f32 {
    let mut acc = acc_in;
    for i in 0..8 {
        acc = q(word, i).mul_add(d(x[kb + i]), acc);
    }
    acc
}

#[cfg(feature = "cuda")]
pub fn dtoh_u16(stream: &Arc<CudaStream>, d: &CudaSlice<u16>) -> Vec<u16> {
    #[allow(deprecated)]
    let v = stream.memcpy_dtov(d).unwrap();
    v
}

#[cfg(feature = "cuda")]
pub fn e4m3_decode(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = (b >> 3) & 0xf;
    let m = (b & 7) as f32;
    if (b & 0x7f) == 0x7f {
        return f32::NAN;
    }
    if e == 0 {
        s * m * (1.0 / 8.0) * 2f32.powi(-6)
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e as i32 - 7)
    }
}

#[cfg(feature = "wgpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FdParams {
    pub n_heads: u32,
    pub n_kv: u32,
    pub head_dim: u32,
    pub total: u32,
    pub start: u32,
    pub splits: u32,
    pub ring: u32,
    pub out_bf16: u32,
    pub scaling: f32,
    pub pad0: u32,
    pub fused: u32,
    pub pad2: u32,
    pub m_rows: u32,
    pub window: u32,
    pub pad3: u32,
    pub pad4: u32,
}

pub fn frand(seed: u64, i: usize) -> f32 {
    let mut z = seed
        .wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1))
        .wrapping_mul(0xBF58476D1CE4E5B9);
    z ^= z >> 29;
    z = z.wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 32;
    ((z & 0xFFFF) as f32 / 65535.0) - 0.5
}

#[cfg(feature = "cuda")]
pub fn htod_f32(stream: &Arc<CudaStream>, v: &[f32]) -> CudaSlice<f32> {
    #[allow(deprecated)]
    let d = stream.clone_htod(&v.to_vec()).unwrap();
    d
}

#[cfg(feature = "cuda")]
pub fn htod_u16(stream: &Arc<CudaStream>, v: &[u16]) -> CudaSlice<u16> {
    #[allow(deprecated)]
    let d = stream.clone_htod(&v.to_vec()).unwrap();
    d
}

#[cfg(feature = "cuda")]
pub fn lcg_unit_f32(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*seed >> 40) as u32;
    (bits as f32) / (1u32 << 24) as f32
}

pub struct LcgShift32TwoSided(pub u64);

impl LcgShift32TwoSided {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    pub fn bf16_vec(&mut self, len: usize, scale: f32) -> Vec<u16> {
        (0..len)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
            .collect()
    }
    pub fn f32_vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_f32() * scale).collect()
    }
    pub fn next_fp8(&mut self) -> u8 {
        loop {
            let b = (self.next_u32() & 0xff) as u8;
            if b & 0x7f != 0x7f {
                return b;
            }
        }
    }
    pub fn fp8_vec(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_fp8()).collect()
    }
    pub fn scale_vec(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| 0.002 + self.next_f32().abs() * 0.03)
            .collect()
    }
}

pub fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

pub struct LcgShift33W4a16Packs(pub u64);

impl LcgShift33W4a16Packs {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / 2147483648.0) - 1.0
    }
    pub fn packed(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.next_u32()).collect()
    }
    pub fn bf16_words(&mut self, n: usize, gain: f32) -> Vec<u16> {
        (0..n)
            .map(|_| bf16::from_f32(self.next_f32() * gain).to_bits())
            .collect()
    }
    pub fn scales(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| bf16::from_f32(0.002 + self.next_f32().abs() * 0.03).to_bits())
            .collect()
    }
}

pub struct LcgOddSeedShift32F64TwoSided(pub u64);

impl LcgOddSeedShift32F64TwoSided {
    pub fn new(seed: u64) -> Self {
        LcgOddSeedShift32F64TwoSided(seed | 1)
    }
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / u32::MAX as f64) as f32 * 2.0 - 1.0
    }
}

pub struct LcgMask23TwoSided(pub u64);

impl LcgMask23TwoSided {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u32() & 0x7f_ffff) as f32 / 8388608.0) * 2.0 - 1.0
    }
    pub fn bf16_words(&mut self, n: usize, gain: f32) -> Vec<u16> {
        (0..n)
            .map(|_| bf16::from_f32(self.next_f32() * gain).to_bits())
            .collect()
    }
    pub fn packed_nibbles(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next_u32() & 0xff) as u8).collect()
    }
    pub fn plausible_ue4m3_scale_bytes_biased_to_small_exponents(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                let exp = 1 + (self.next_u32() % 8) as u8;
                let mant = (self.next_u32() % 8) as u8;
                (exp << 3) | mant
            })
            .collect()
    }
}

pub fn lcg_f32(state: &mut u64) -> f32 {
    ((lcg(state) >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
}

#[cfg(feature = "wgpu")]
pub fn max_abs_err(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want.iter())
        .fold(0f32, |m, (g, e)| m.max((g - e).abs()))
}

#[cfg(feature = "wgpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OffParams {
    pub dst_word_off: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[cfg(feature = "wgpu")]
pub fn pack_u16(src: &[u16]) -> Vec<u32> {
    let mut out = vec![0u32; src.len() / 2];
    for (i, w) in out.iter_mut().enumerate() {
        *w = src[2 * i] as u32 | ((src[2 * i + 1] as u32) << 16);
    }
    out
}

#[cfg(feature = "wgpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub n_rows: u32,
    pub k_elems: u32,
    pub gs: u32,
    pub w_row_words: u32,
    pub scale_row_stride: u32,
    pub groups_x: u32,
}

#[cfg(feature = "wgpu")]
pub fn q(word: u32, elem: usize) -> f32 {
    (((word >> (4 * elem)) & 0xf) as i32 - 8) as f32
}

#[cfg(feature = "cuda")]
pub fn rand_bf16(seed: &mut u64, n: usize, lo: f32, hi: f32) -> Vec<u16> {
    (0..n)
        .map(|_| half::bf16::from_f32(lo + lcg_unit_f32(seed) * (hi - lo)).to_bits())
        .collect()
}

#[cfg(feature = "cuda")]
pub fn require_parity_opt_in() -> bool {
    std::env::var("NV_KERNELS_PARITY_REQUIRE").as_deref() == Ok("1")
}

pub const REFUSE_TO_SKIP_IS_THE_DEFAULT_BECAUSE_AN_OPT_IN_GATE_ONCE_REPORTED_A_WHOLE_BACKEND_PASSED_ON_NOTHING:
    &str = "NV_KERNELS_WGPU_ALLOW_SKIP";

#[cfg(feature = "wgpu")]
pub fn require() -> bool {
    std::env::var("NV_KERNELS_WGPU_ALLOW_SKIP").as_deref() != Ok("1")
}

pub fn rnd_f(state: &mut u64) -> f32 {
    ((lcg(state) >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
}

#[cfg(feature = "cuda")]
pub fn stream(test: &str) -> Option<Arc<CudaStream>> {
    match CudaContext::new(0) {
        Ok(c) => Some(c.default_stream()),
        Err(e) => {
            if require_parity_opt_in() {
                panic!("{test}: no CUDA device 0: {e}");
            }
            eprintln!("{test}: SKIP no CUDA device 0: {e}");
            None
        }
    }
}

#[cfg(feature = "wgpu")]
pub fn to_bf16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| bf16::from_f32(*x).to_bits()).collect()
}

#[cfg(feature = "wgpu")]
pub fn tree_sum(vals: &[f32]) -> f32 {
    let mut s = vals.to_vec();
    let mut stride = s.len() / 2;
    while stride > 0 {
        for i in 0..stride {
            s[i] += s[i + stride];
        }
        stride >>= 1;
    }
    s[0]
}

#[cfg(feature = "wgpu")]
pub fn wgpu_allow_skip() -> bool {
    std::env::var("NV_KERNELS_WGPU_ALLOW_SKIP").as_deref() == Ok("1")
}

#[cfg(feature = "wgpu")]
pub fn widen_u16(src: &[u16]) -> Vec<u32> {
    src.iter().map(|v| *v as u32).collect()
}

#[cfg(feature = "cuda")]
pub fn xorshift(state: &mut u32) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    (x as f32 / u32::MAX as f32) * 2.0 - 1.0
}

pub fn idle_pct() -> Option<u32> {
    let out = std::process::Command::new("top")
        .args(["-l", "1", "-n", "8", "-o", "cpu"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if line.contains("CPU usage") {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let pct = fields.get(6)?.trim_end_matches('%');
            return pct.parse::<f32>().ok().map(|v| v as u32);
        }
    }
    None
}

pub fn wait_for_idle(min_pct: u32, timeout: std::time::Duration) -> bool {
    let start = std::time::Instant::now();
    loop {
        match idle_pct() {
            Some(p) if p >= min_pct => {
                eprintln!("idle gate: {p}% idle, proceeding");
                return true;
            }
            Some(p) => eprintln!("idle gate: {p}% idle, waiting"),
            None => eprintln!("idle gate: top unavailable, proceeding without gate"),
        }
        if idle_pct().is_none() || start.elapsed() >= timeout {
            return idle_pct().unwrap_or(100) >= min_pct;
        }
        std::thread::sleep(std::time::Duration::from_secs(20));
    }
}

pub fn time_calls<F: FnMut()>(mut f: F, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let start = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64()
}

pub fn gpu_util() -> Option<u32> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

pub fn swizzled_dst(m: usize, kb: usize, k_blocks: usize) -> usize {
    let k_tiles = k_blocks.div_ceil(4);
    let m_tile = m / 128;
    let d2 = (m / 32) & 3;
    let d3 = m & 31;
    let k_tile = kb / 4;
    let d5 = kb & 3;
    ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5
}

pub fn swizzled_scale_dst(m: usize, kb: usize, k_blocks: usize) -> usize {
    let k_tiles = k_blocks.div_ceil(4);
    let m_tile = m / 128;
    let d2 = (m / 32) % 4;
    let d3 = m % 32;
    let k_tile = kb / 4;
    let d5 = kb % 4;
    ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5
}

#[cfg(feature = "wgpu")]
pub fn ctx_or_skip_quiet_unqualified(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("{test}: {}", ctx.summary());
            let st = ctx.qualify();
            if !st.qualified {
                if !wgpu_allow_skip() {
                    panic!(
                        "{test}: wgpu adapter not qualified: {:?}. This gate refuses to \
                         report success without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to \
                         skip on purpose.",
                        st.reason
                    );
                }
                return None;
            }
            Some(ctx)
        }
        Err(e) => {
            if !wgpu_allow_skip() {
                panic!("{test}: no wgpu adapter: {e}");
            }
            eprintln!("{test}: SKIP no wgpu adapter: {e}");
            None
        }
    }
}

#[cfg(feature = "wgpu")]
pub fn ctx_or_skip_reasoned(what: &str) -> Option<&'static WgpuContext> {
    match shared_or_reason() {
        Ok(ctx) => {
            let q = ctx.qualify();
            if !q.qualified {
                if !wgpu_allow_skip() {
                    panic!(
                        "{what}: wgpu adapter not qualified: {:?}. This gate refuses to \
                         report success without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to \
                         skip on purpose.",
                        q.reason
                    );
                }
                eprintln!(
                    "{what}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) adapter not qualified: {:?}",
                    q.reason
                );
                return None;
            }
            eprintln!("{what}: {}", ctx.summary());
            Some(ctx)
        }
        Err(reason) => {
            if !wgpu_allow_skip() {
                panic!(
                    "{what}: no wgpu adapter: {reason}. This gate refuses to report success \
                     without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
                );
            }
            eprintln!("{what}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) no wgpu adapter: {reason}");
            None
        }
    }
}

#[cfg(feature = "wgpu")]
pub fn ctx_or_panic() -> &'static WgpuContext {
    match WgpuContext::shared() {
        Ok(c) => c,
        Err(e) => panic!("no wgpu adapter: {e}"),
    }
}

#[cfg(feature = "wgpu")]
pub fn pipeline(
    ctx: &WgpuContext,
    label: &str,
    src: &str,
    entry: &str,
    zero_init: bool,
) -> wgpu::ComputePipeline {
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
    let pl = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[],
                zero_initialize_workgroup_memory: zero_init,
            },
            cache: None,
        });
    if let Some(e) = pollster::block_on(scope.pop()) {
        panic!("{label}:{entry} failed to compile: {e}");
    }
    pl
}

pub fn lcg_hi33_u32(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as u32
}

pub fn lcg_hi32_u32(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 32) as u32
}

pub fn rel_rms_f32(got: &[f32], want: &[f32]) -> f32 {
    let num: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| ((a - b) as f64) * ((a - b) as f64))
        .sum();
    let den: f64 = want.iter().map(|b| (*b as f64) * (*b as f64)).sum();
    (num.sqrt() / den.sqrt()) as f32
}

pub fn rel_rms_bf16(got: &[bf16], expect: &[bf16]) -> f64 {
    let mut sum_sq = 0f64;
    let mut sum_expect_sq = 0f64;
    for (g, e) in got.iter().zip(expect.iter()) {
        let d = (g.to_f32() - e.to_f32()) as f64;
        sum_sq += d * d;
        sum_expect_sq += (e.to_f32() as f64).powi(2);
    }
    (sum_sq / sum_expect_sq.max(1e-12)).sqrt()
}

pub fn rnd_fp8(state: &mut u64) -> u8 {
    let b = (lcg(state) >> 40) as u8;
    if (b & 0x7f) == 0x7f {
        b & 0x7e
    } else {
        b
    }
}

pub fn fp8_e4m3_to_f64(b: u8) -> f64 {
    let sign = if b & 0x80 != 0 { -1.0f64 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f64;
    if e == 0 {
        sign * (m / 8.0) * 2f64.powi(-6)
    } else {
        sign * (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

pub fn max_rel_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&p, &r)| ((p - r).abs() as f64) / (p.abs().max(r.abs()).max(1e-3) as f64))
        .fold(0.0, f64::max)
}

pub fn host_w_f64(packed: &[u8], scales_linear: &[u8], n: usize, k: usize, alpha: f32) -> Vec<f64> {
    let sw = nv_quant::nvfp4::swizzle_scales(scales_linear, n, k / 16);
    nv_quant::nvfp4::dequantize_packed_swizzled(packed, &sw, n, k, alpha)
        .into_iter()
        .map(|v| v as f64)
        .collect()
}

pub fn assert_rows_close(name: &str, got: &[u16], reference: &[f64], tol: f64) {
    let mut max_rel = 0f64;
    for (g, r) in got.iter().zip(reference.iter()) {
        let gv = bf16::from_bits(*g).to_f32() as f64;
        let denom = r.abs().max(0.25);
        max_rel = max_rel.max((gv - r).abs() / denom);
    }
    eprintln!("{name}: max_rel_err_vs_f64_ref={max_rel:.3e} over {}", got.len());
    assert!(
        max_rel < tol,
        "{name}: max rel err {max_rel:.3e} exceeds tolerance {tol:.1e}"
    );
}

pub fn sample_bf16(n: usize, seed: u32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let t = (i as f32 + seed as f32 * 0.5) * 0.017;
            let v = t.sin() * 2.5 + (t * 3.1).cos() * 0.75;
            bf16::from_f32(v).to_bits()
        })
        .collect()
}

pub fn cpu_rmsnorm(x: &[f32], weight: &[f32], hidden: usize, eps: f32) -> Vec<f32> {
    let batch = x.len() / hidden;
    let mut y = vec![0f32; x.len()];
    for b in 0..batch {
        let row = &x[b * hidden..(b + 1) * hidden];
        let sumsq: f32 = row.iter().map(|v| v * v).sum();
        let rms = (sumsq / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            y[b * hidden + i] = row[i] / rms * weight[i];
        }
    }
    y
}

pub fn cpu_silu_mul(x: &[f32], gate: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(gate.iter())
        .map(|(a, b)| (a / (1.0 + (-a).exp())) * b)
        .collect()
}

pub fn reference_e4m3(byte: u8) -> f32 {
    let e = ((byte >> 3) & 15) as i32;
    let m = (byte & 7) as f64;
    let mag = if e == 0 {
        m * 2f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    let v = mag as f32;
    if byte & 0x80 != 0 {
        -v
    } else {
        v
    }
}

pub fn reference_e2m1(code: u8) -> f32 {
    let e = ((code >> 1) & 3) as i32;
    let m = (code & 1) as f64;
    let mag = if e == 0 {
        m * 0.5
    } else {
        (1.0 + m * 0.5) * 2f64.powi(e - 1)
    };
    let v = mag as f32;
    if code & 8 != 0 {
        -v
    } else {
        v
    }
}

pub fn bf16_enc(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let b = x.to_bits();
    let r = 0x7fffu32 + ((b >> 16) & 1);
    (b.wrapping_add(r) >> 16) as u16
}

#[derive(Clone)]
pub struct Case {
    pub t: usize,
    pub k: usize,
    pub rank: usize,
    pub widths: Vec<usize>,
    pub max_loras: usize,
    pub mapping: Vec<i32>,
    pub slot_ranks: Vec<usize>,
    pub scale: f32,
    pub seed: u64,
    pub win: Option<(usize, usize)>,
}

impl Case {
    pub fn sum_n(&self) -> usize {
        self.widths.iter().sum()
    }
    pub fn max_n(&self) -> usize {
        *self.widths.iter().max().unwrap()
    }
    pub fn slice_starts(&self) -> Vec<usize> {
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
    pub fn window(&self) -> (usize, usize) {
        self.win.unwrap_or((0, self.sum_n()))
    }
}

pub fn build_y_base(case: &Case) -> Vec<bf16> {
    let (_, win_len) = case.window();
    (0..case.t * win_len)
        .map(|i| bf16::from_f32(frand(case.seed ^ 0x22, i) * 2.0))
        .collect()
}

pub fn build_a(case: &Case, wseed: u64) -> Vec<Vec<bf16>> {
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

pub fn build_b(case: &Case, wseed: u64) -> Vec<Vec<bf16>> {
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

pub fn lora_oracle(case: &Case, x: &[bf16], a: &[Vec<bf16>], b: &[Vec<bf16>], y_base: &[bf16]) -> Vec<bf16> {
    let starts = case.slice_starts();
    let (win_off, win_len) = case.window();
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
                let col = starts[s] + n;
                if col < win_off || col >= win_off + win_len {
                    continue;
                }
                let mut acc = 0f32;
                for r in 0..case.rank {
                    acc += tmp[r] * b[s][(slot * w + n) * case.rank + r].to_f32();
                }
                let yi = tok * win_len + (col - win_off);
                y[yi] = bf16::from_f32(y[yi].to_f32() + acc);
            }
        }
    }
    y
}

pub struct LcgShift40Top24TwoSided(pub u64);

impl LcgShift40Top24TwoSided {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 40) as u32;
        (bits as f32 / 8388608.0) - 1.0
    }
}

#[cfg(feature = "wgpu")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WmmaAccumClass {
    IeeeExact,
    TruncatingDot2,
}

#[cfg(feature = "wgpu")]
pub fn wmma_f16_dot_probe(
    ctx: &WgpuContext,
    cases: &[(Vec<f32>, Vec<f32>, f32)],
) -> Vec<f32> {
    use nv_kernels::wgpu_backend::dispatch;
    let t = 16usize;
    let n = t * t;
    let src = format!(
        "enable f16;\nenable wgpu_cooperative_matrix;\n\
         alias CA = coop_mat16x16<f16, A>;\n\
         alias CB = coop_mat16x16<f16, B>;\n\
         alias CC = coop_mat16x16<f32, C>;\n\
         const TILE: u32 = 16u;\n\
         @group(0) @binding(0) var<storage, read> ma: array<f16>;\n\
         @group(0) @binding(1) var<storage, read> mb: array<f16>;\n\
         @group(0) @binding(2) var<storage, read_write> md: array<f32>;\n\
         @compute @workgroup_size(32)\n\
         fn mma_once() {{\n\
         let a = coopLoadT<CA>(&ma[0], TILE);\n\
         let b = coopLoadT<CB>(&mb[0], TILE);\n\
         let c = coopLoadT<CC>(&md[{n}], TILE);\n\
         let d = coopMultiplyAdd(a, b, c);\n\
         coopStoreT(d, &md[0], TILE);\n\
         }}\n"
    );
    let pipe = dispatch::compute_pipeline(ctx, "wmma-accum-probe", &src, "mma_once")
        .expect("wmma probe pipeline");
    let mut out = Vec::with_capacity(cases.len());
    for (xs, ys, c) in cases {
        assert!(xs.len() <= t && ys.len() == xs.len());
        let mut a = vec![0f32; n];
        let mut b = vec![0f32; n];
        for (k, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
            a[k] = *x;
            b[k * t] = *y;
        }
        let ah: Vec<u16> = a.iter().map(|v| half::f16::from_f32(*v).to_bits()).collect();
        let bh: Vec<u16> = b.iter().map(|v| half::f16::from_f32(*v).to_bits()).collect();
        let mut dc = vec![0f32; 2 * n];
        dc[n] = *c;
        let abuf = dispatch::storage_from_slice(ctx, "wmma-probe-a", &ah);
        let bbuf = dispatch::storage_from_slice(ctx, "wmma-probe-b", &bh);
        let dbuf = dispatch::storage_from_slice(ctx, "wmma-probe-d", &dc);
        let bg = dispatch::bind_group(ctx, &pipe, &[(0, &abuf), (1, &bbuf), (2, &dbuf)]);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&pipe);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit([enc.finish()]);
        ctx.poll_blocking().unwrap();
        let got: Vec<f32> = dispatch::read_back(ctx, &dbuf, 1).unwrap();
        out.push(got[0]);
    }
    out
}

#[cfg(feature = "wgpu")]
pub fn wmma_accum_class(ctx: &WgpuContext) -> WmmaAccumClass {
    let got = wmma_f16_dot_probe(ctx, &[(vec![3.0, -1.0], vec![1.0, 1.0], 0.0)]);
    match got[0].to_bits() {
        0x4000_0000 => WmmaAccumClass::IeeeExact,
        0x3fff_ffff => WmmaAccumClass::TruncatingDot2,
        bits => panic!(
            "coopMultiplyAdd of the single dot2-lane pair (3, -1) returned {bits:#010x}: neither \
             the IEEE-exact 2.0 nor the truncating-dot2 2-2^-23 this fleet has characterized. \
             Characterize the new adapter class in wgpu_wmma_accum_model.rs before trusting any \
             coop-matrix numerics on it"
        ),
    }
}
