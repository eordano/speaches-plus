#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub const WGSL: &str = include_str!("../../../wgsl/sampler.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4 * 2 + 16;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SamplerParams {
    vocab: u32,
    batch: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    flags: u32,
    u01_bits: u32,
    inv_t: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

pub const EXACT_MAX_TOP_K: u32 = 256;

pub const EXACT_SCRATCH_BYTES: u32 = EXACT_MAX_TOP_K * 4 * 4 + WORKGROUP_SIZE * 4 * 2 + 32;

pub const EXACT_SENTINEL: u32 = 0xFFFF_FFFF;

const EXACT_FLAG_HOST_U01: u32 = 1;

const EXACT_U_MAX: f32 = 1.0 - f32::EPSILON;

#[derive(Clone, Copy, Debug)]
pub struct ExactSampling {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub u01: Option<f32>,
    pub seed: u64,
}

impl Default for ExactSampling {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            u01: None,
            seed: 0,
        }
    }
}

impl ExactSampling {
    pub fn supported(&self, vocab: usize) -> bool {
        if self.temperature <= 1e-6 {
            return true;
        }
        self.top_k >= 1 && self.top_k <= EXACT_MAX_TOP_K && (self.top_k as usize) <= vocab
    }
}

pub fn unit_from_seed(seed: u64, row: u32) -> f32 {
    let golden = 0x9e37_79b9_7f4a_7c15u64.wrapping_add(row as u64);
    let mut z = seed ^ golden;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    ((z >> 40) as f32) * (1.0f32 / 16_777_216.0)
}

fn exact_params(batch: usize, vocab: usize, p: &ExactSampling) -> SamplerParams {
    let (flags, u01_bits) = match p.u01 {
        Some(u) => (EXACT_FLAG_HOST_U01, u.clamp(0.0, EXACT_U_MAX).to_bits()),
        None => (0, 0),
    };
    SamplerParams {
        vocab: vocab as u32,
        batch: batch as u32,
        temperature: p.temperature,
        top_k: p.top_k,
        top_p: p.top_p,
        min_p: p.min_p,
        flags,
        u01_bits,
        inv_t: 1.0f32 / p.temperature.max(1e-6),
        ..Default::default()
    }
}

pub fn sampler_exact_token_buffers(
    ctx: &WgpuContext,
    logits: &wgpu::Buffer,
    token_out: &wgpu::Buffer,
    seeds: &wgpu::Buffer,
    batch: usize,
    vocab: usize,
    p: &ExactSampling,
) -> Result<()> {
    if batch == 0 || vocab == 0 {
        return Ok(());
    }
    check_device(ctx, batch * vocab)?;
    if !ctx.caps.workgroup_storage_fits(EXACT_SCRATCH_BYTES) {
        return Err(WgpuError::Unsupported(format!(
            "exact sampler scratch needs {EXACT_SCRATCH_BYTES} bytes of workgroup storage; \
             device allows {}",
            ctx.caps.max_compute_workgroup_storage_size
        )));
    }
    let params = exact_params(batch, vocab, p);
    let ub = dispatch::uniform_from(ctx, "sampler-exact-params", &params);
    let groups = dispatch::workgroup_count_1d(ctx, batch as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_sampler_exact_token",
        &compose(WGSL),
        "sampler_exact_token",
        &[(0, logits), (1, seeds), (3, token_out), (4, &ub)],
        groups,
    )
}

pub fn sampler_exact_token(
    ctx: &WgpuContext,
    logits: &[f32],
    batch: usize,
    vocab: usize,
    p: &ExactSampling,
) -> Result<Vec<u32>> {
    dispatch::check_len("sampler logits", logits.len(), batch * vocab)?;
    if batch == 0 || vocab == 0 {
        return Ok(Vec::new());
    }
    let seeds = vec![p.seed; batch];
    let lb = dispatch::storage_from_slice(ctx, "sampler-exact-logits", logits);
    let sb = dispatch::storage_from_slice(ctx, "sampler-exact-seeds", &split_seeds(&seeds));
    let tb = dispatch::storage_zeroed(ctx, "sampler-exact-token", (batch * 4) as u64);
    sampler_exact_token_buffers(ctx, &lb, &tb, &sb, batch, vocab, p)?;
    dispatch::read_back(ctx, &tb, batch)
}

fn check_device(ctx: &WgpuContext, elems: usize) -> Result<()> {
    if ctx.caps.max_compute_invocations_per_workgroup < WORKGROUP_SIZE
        || ctx.caps.max_compute_workgroup_size_x < WORKGROUP_SIZE
    {
        return Err(WgpuError::Unsupported(format!(
            "sampler needs a {WORKGROUP_SIZE}-invocation workgroup; device allows {} (x max {})",
            ctx.caps.max_compute_invocations_per_workgroup, ctx.caps.max_compute_workgroup_size_x
        )));
    }
    if !ctx.caps.workgroup_storage_fits(SCRATCH_BYTES) {
        return Err(WgpuError::Unsupported(format!(
            "sampler scratch needs {SCRATCH_BYTES} bytes of workgroup storage; device allows {}",
            ctx.caps.max_compute_workgroup_storage_size
        )));
    }
    let bytes = (elems as u64).saturating_mul(4);
    if bytes > ctx.caps.max_storage_buffer_binding_size {
        return Err(WgpuError::Unsupported(format!(
            "sampler needs a {bytes}-byte storage binding; device allows {}",
            ctx.caps.max_storage_buffer_binding_size
        )));
    }
    Ok(())
}

fn split_seeds(seeds: &[u64]) -> Vec<u32> {
    let mut out = vec![0u32; seeds.len() * 2];
    for (i, s) in seeds.iter().enumerate() {
        out[2 * i] = (*s & 0xffff_ffff) as u32;
        out[2 * i + 1] = (*s >> 32) as u32;
    }
    out
}

pub fn sampler_topk_topp(
    ctx: &WgpuContext,
    logits: &[f32],
    probs_out: &mut [f32],
    token_out: &mut [u32],
    batch: usize,
    vocab: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
) -> Result<()> {
    let seeds = vec![seed; batch];
    sampler_topk_topp_seeds(
        ctx,
        logits,
        &seeds,
        probs_out,
        token_out,
        batch,
        vocab,
        temperature,
        top_k,
        top_p,
    )
}

pub fn sampler_topk_topp_seeds(
    ctx: &WgpuContext,
    logits: &[f32],
    seeds: &[u64],
    probs_out: &mut [f32],
    token_out: &mut [u32],
    batch: usize,
    vocab: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
) -> Result<()> {
    dispatch::check_len("sampler logits", logits.len(), batch * vocab)?;
    dispatch::check_len("sampler probs_out", probs_out.len(), batch * vocab)?;
    dispatch::check_len("sampler token_out", token_out.len(), batch)?;
    dispatch::check_len("sampler seeds", seeds.len(), batch)?;
    if batch == 0 || vocab == 0 {
        return Ok(());
    }
    check_device(ctx, batch * vocab)?;

    let params = SamplerParams {
        vocab: vocab as u32,
        batch: batch as u32,
        temperature,
        top_k,
        top_p,
        ..Default::default()
    };

    let lb = dispatch::storage_from_slice(ctx, "sampler-logits", logits);
    let sb = dispatch::storage_from_slice(ctx, "sampler-seeds", &split_seeds(seeds));
    let pb = dispatch::storage_zeroed(ctx, "sampler-probs", (batch * vocab * 4) as u64);
    let tb = dispatch::storage_zeroed(ctx, "sampler-token", (batch * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "sampler-params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, batch as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_sampler_topk_topp",
        &compose(WGSL),
        "sampler_topk_topp",
        &[(0, &lb), (1, &sb), (2, &pb), (3, &tb), (4, &ub)],
        groups,
    )?;

    let probs: Vec<f32> = dispatch::read_back(ctx, &pb, batch * vocab)?;
    probs_out.copy_from_slice(&probs);
    let toks: Vec<u32> = dispatch::read_back(ctx, &tb, batch)?;
    token_out.copy_from_slice(&toks);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_split_is_little_endian_pairs() {
        let words = split_seeds(&[0x0123_4567_89ab_cdef, 1]);
        assert_eq!(words, vec![0x89ab_cdef, 0x0123_4567, 1, 0]);
    }

    #[test]
    fn shape_mismatch_is_reported() {
        let e = dispatch::check_len("sampler logits", 10, 12).unwrap_err();
        assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    }

    #[test]
    fn params_are_forty_eight_bytes() {
        assert_eq!(std::mem::size_of::<SamplerParams>(), 48);
        assert_eq!(std::mem::size_of::<SamplerParams>() % 16, 0);
    }

    #[test]
    fn exact_support_window_is_advertised_honestly() {
        let p = ExactSampling {
            temperature: 0.8,
            top_k: 40,
            ..Default::default()
        };
        assert!(p.supported(262144));
        assert!(!p.supported(8), "top_k must not exceed the vocab");
        let no_k = ExactSampling {
            temperature: 0.8,
            top_k: 0,
            ..Default::default()
        };
        assert!(
            !no_k.supported(262144),
            "top_k is mandatory off the greedy path"
        );
        let too_big = ExactSampling {
            temperature: 0.8,
            top_k: 257,
            ..Default::default()
        };
        assert!(!too_big.supported(262144));
        let greedy = ExactSampling {
            temperature: 0.0,
            top_k: 0,
            ..Default::default()
        };
        assert!(greedy.supported(262144));
    }

    #[test]
    fn host_u01_is_clamped_the_same_way_the_host_sampler_clamps() {
        let p = ExactSampling {
            u01: Some(1.5),
            ..Default::default()
        };
        let got = f32::from_bits(exact_params(1, 8, &p).u01_bits);
        assert_eq!(got, 1.0f32 - f32::EPSILON);
        let p = ExactSampling {
            u01: Some(-3.0),
            ..Default::default()
        };
        assert_eq!(f32::from_bits(exact_params(1, 8, &p).u01_bits), 0.0);
    }

    #[test]
    fn unit_from_seed_is_in_the_unit_interval() {
        for s in [0u64, 1, 0x9e37_79b9_7f4a_7c15, u64::MAX] {
            for row in 0..4u32 {
                let u = unit_from_seed(s, row);
                assert!((0.0..1.0).contains(&u), "seed={s} row={row} u={u}");
            }
        }
    }
}
