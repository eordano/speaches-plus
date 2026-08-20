#![cfg(feature = "wgpu")]

mod common;
use common::require;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::gemm_nvfp4::GemmPath;
use nv_kernels::wgpu_backend::kernels::moe_grouped_gemm as w_moe;
use nv_kernels::wgpu_backend::kernels::moe_permute as w_permute;
use nv_quant::nvfp4::{swizzle_scales, Nvfp4Tensor, BLOCK_SIZE};

fn backend(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(ctx) if ctx.qualify().qualified => {
            eprintln!("{test}: {}", ctx.summary());
            Some(ctx)
        }
        Ok(ctx) => {
            if require() {
                panic!(
                    "{test}: wgpu adapter not qualified: {:?}",
                    ctx.qualify().reason
                );
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

fn coop_available(wg: &WgpuContext, test: &str) -> bool {
    match wg.caps.coop_gemm_reason() {
        None => true,
        Some(why) => {
            if require() {
                panic!("{test}: coop_mat gemm path unavailable: {why}");
            }
            eprintln!("{test}: coop_mat unavailable: {why}");
            false
        }
    }
}

fn bf16_ord(bits: u16) -> i32 {
    if bits & 0x8000 != 0 {
        -((bits & 0x7fff) as i32)
    } else {
        bits as i32
    }
}

#[derive(Default, Clone, Copy)]
struct Stats {
    differ: usize,
    total: usize,
    max_ulp: i32,
    rel_rms: f64,
}

impl Stats {
    fn render(&self) -> String {
        format!(
            "{}/{} differ max_ulp={} rel_rms={:.3e}",
            self.differ, self.total, self.max_ulp, self.rel_rms
        )
    }
}

fn compare(got: &[u16], want: &[u16]) -> Stats {
    assert_eq!(got.len(), want.len());
    let mut s = Stats {
        total: got.len(),
        ..Stats::default()
    };
    let mut sq = 0f64;
    let mut ref_sq = 0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        if g != w {
            s.differ += 1;
            s.max_ulp = s.max_ulp.max((bf16_ord(*g) - bf16_ord(*w)).abs());
        }
        let gv = bf16::from_bits(*g).to_f32() as f64;
        let wv = bf16::from_bits(*w).to_f32() as f64;
        sq += (gv - wv) * (gv - wv);
        ref_sq += wv * wv;
    }
    s.rel_rms = (sq / ref_sq.max(1e-12)).sqrt();
    s
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Profile {
    Uniform,
    Wide,
}

struct Case {
    name: &'static str,
    e_total: usize,
    groups: Vec<(usize, usize)>,
    alphas: Vec<f32>,
    n: usize,
    k: usize,
    profile: Profile,
}

fn case(
    name: &'static str,
    e_total: usize,
    groups: Vec<(usize, usize)>,
    n: usize,
    k: usize,
) -> Case {
    let alphas = vec![1.0; groups.len()];
    Case {
        name,
        e_total,
        groups,
        alphas,
        n,
        k,
        profile: Profile::Uniform,
    }
}

fn cases() -> Vec<Case> {
    vec![
        case("one_expert", 1, vec![(0, 128)], 128, 128),
        case("one_group_of_many", 4, vec![(2, 7)], 48, 32),
        case(
            "many_uniform",
            8,
            (0..8).map(|e| (e, 16)).collect(),
            64,
            128,
        ),
        case(
            "ragged_with_empty",
            8,
            vec![(0, 5), (1, 0), (3, 33), (4, 0), (6, 64), (7, 1)],
            80,
            64,
        ),
        case("untiled_m_and_n", 4, vec![(1, 37), (2, 29)], 33, 48),
        Case {
            alphas: vec![0.375, 2.0],
            ..case("scaled", 4, vec![(0, 10), (3, 20)], 64, 64)
        },
        Case {
            profile: Profile::Wide,
            ..case("wide_profile", 4, vec![(0, 16), (1, 48)], 64, 128)
        },
        case("all_groups_empty", 4, vec![(0, 0), (1, 0), (2, 0)], 32, 32),
    ]
}

fn gen_rows(rows: usize, cols: usize, profile: Profile, freq: f32, phase: f32) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| {
                    let base = (((i * cols + j) as f32) * freq + phase).sin();
                    match profile {
                        Profile::Uniform => base,
                        Profile::Wide => {
                            let e = ((i + j / 16) % 15) as i32 - 6;
                            base * (2f32).powi(e)
                        }
                    }
                })
                .collect()
        })
        .collect()
}

struct Operands {
    a: Nvfp4Tensor,
    a_sf: Vec<u8>,
    b_packed: Vec<u8>,
    b_scales: Vec<u8>,
    b_deq: Vec<Vec<Vec<f32>>>,
    offsets: Vec<i32>,
    ids: Vec<i32>,
    total_m: usize,
}

fn operands(c: &Case) -> Operands {
    let total_m: usize = c.groups.iter().map(|(_, m)| m).sum();
    let a_rows = gen_rows(total_m.max(1), c.k, c.profile, 0.07, 0.0);
    let a = Nvfp4Tensor::quantize_rows(&a_rows[..total_m.max(1)]);
    let a_sf = swizzle_scales(&a.scales, total_m.max(1), c.k / BLOCK_SIZE);

    let mut b_packed = Vec::new();
    let mut b_scales = Vec::new();
    let mut b_deq = Vec::new();
    for e in 0..c.e_total {
        let rows = gen_rows(c.n, c.k, c.profile, 0.09, 1.7 + e as f32 * 0.31);
        let b = Nvfp4Tensor::quantize_rows(&rows);
        b_scales.extend_from_slice(&swizzle_scales(&b.scales, c.n, c.k / BLOCK_SIZE));
        b_packed.extend_from_slice(&b.data);
        b_deq.push(b.dequantize());
    }

    let mut offsets = vec![0i32];
    let mut ids = Vec::new();
    for (e, m) in &c.groups {
        ids.push(*e as i32);
        offsets.push(offsets.last().unwrap() + *m as i32);
    }
    Operands {
        a,
        a_sf,
        b_packed,
        b_scales,
        b_deq,
        offsets,
        ids,
        total_m,
    }
}

fn cpu_oracle_blocked(
    a_deq: &[Vec<f32>],
    b_deq: &[Vec<Vec<f32>>],
    offsets: &[i32],
    ids: &[i32],
    alphas: &[f32],
    n: usize,
    k: usize,
) -> Vec<u16> {
    let total_m = *offsets.last().unwrap() as usize;
    let mut out = vec![0u16; total_m * n];
    for g in 0..ids.len() {
        let lo = offsets[g] as usize;
        let hi = offsets[g + 1] as usize;
        let bd = &b_deq[ids[g] as usize];
        for row in lo..hi {
            let ar = &a_deq[row];
            for col in 0..n {
                let br = &bd[col];
                let mut acc = 0f32;
                for t in 0..k / 16 {
                    let mut block_dot = 0f32;
                    for p in t * 16..t * 16 + 16 {
                        block_dot += ar[p] * br[p];
                    }
                    acc += block_dot;
                }
                out[row * n + col] = bf16::from_f32(acc * alphas[g]).to_bits();
            }
        }
    }
    out
}

fn wgpu_grouped(wg: &WgpuContext, o: &Operands, c: &Case, path: GemmPath) -> Vec<u16> {
    let mut out = vec![0u16; o.total_m * c.n];
    let used = w_moe::moe_grouped_nvfp4_gemm_bf16(
        wg,
        &o.a.data[..o.total_m * c.k / 2],
        &o.a_sf,
        &o.b_packed,
        &o.b_scales,
        &o.offsets,
        &o.ids,
        &c.alphas,
        &mut out,
        c.n,
        c.k,
        c.e_total,
        path,
    )
    .unwrap_or_else(|e| panic!("{}: wgpu {path:?}: {e}", c.name));
    assert_eq!(used, path, "{}: requested path was not executed", c.name);
    out
}

fn run_all_cases(path: GemmPath) {
    let name = match path {
        GemmPath::Scalar => "moe_grouped_scalar",
        _ => "moe_grouped_coop",
    };
    let Some(wg) = backend(name) else { return };
    if path == GemmPath::CoopMat && !coop_available(wg, name) {
        return;
    }
    let mut total = Stats::default();
    for c in cases() {
        let o = operands(&c);
        let a_deq = o.a.dequantize();
        let want = cpu_oracle_blocked(&a_deq, &o.b_deq, &o.offsets, &o.ids, &c.alphas, c.n, c.k);
        let got = wgpu_grouped(wg, &o, &c, path);
        let s = compare(&got, &want);
        eprintln!(
            "{:20} E={} groups={:?} n={} k={}  vs k16sum: {}",
            c.name,
            c.e_total,
            o.offsets,
            c.n,
            c.k,
            s.render()
        );
        total.differ += s.differ;
        total.total += s.total;
        total.max_ulp = total.max_ulp.max(s.max_ulp);
    }
    eprintln!(
        "TOTAL {name} vs k16sum: {}/{} differ",
        total.differ, total.total
    );
    assert!(total.total > 10_000, "suite exercised too few output words");
    assert_eq!(
        total.differ, 0,
        "the wgpu grouped NVFP4 GEMM sums each 16-element block into a zeroed f32 accumulator \
         and adds blocks in order per expert group, exactly like the blocked CPU oracle, so \
         every output word must match bit for bit"
    );
}

#[test]
fn moe_grouped_scalar_matches_blocked_cpu_oracle_bit_for_bit() {
    run_all_cases(GemmPath::Scalar);
}

#[test]
fn moe_grouped_coop_matches_blocked_cpu_oracle_bit_for_bit() {
    run_all_cases(GemmPath::CoopMat);
}

#[test]
fn moe_grouped_coop_and_scalar_paths_are_bit_identical() {
    let Some(wg) = backend("moe_grouped_paths") else {
        return;
    };
    if !coop_available(wg, "moe_grouped_paths") {
        return;
    }
    let mut differ = 0usize;
    let mut total = 0usize;
    for c in cases() {
        let o = operands(&c);
        let scalar = wgpu_grouped(wg, &o, &c, GemmPath::Scalar);
        let coop = wgpu_grouped(wg, &o, &c, GemmPath::CoopMat);
        let s = compare(&coop, &scalar);
        eprintln!("{:20} coop vs scalar: {}", c.name, s.render());
        differ += s.differ;
        total += s.total;
    }
    eprintln!("TOTAL coop vs scalar: {differ}/{total} differ");
    assert_eq!(
        differ, 0,
        "the two wgpu grouped paths must agree bit for bit"
    );
}

#[test]
fn moe_grouped_after_wgpu_permute_routing_topk2_matches_oracle() {
    let Some(wg) = backend("moe_grouped_routed") else {
        return;
    };
    let (n_tokens, top_k, e_total, n, k_dim) = (25usize, 2usize, 6usize, 64usize, 64usize);
    let mut seed: u64 = 0x9e3779b97f4a7c15;
    let topk_ids: Vec<i32> = (0..n_tokens * top_k)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as u32 % e_total as u32) as i32
        })
        .collect();
    let mut offsets = vec![0i32; e_total + 1];
    let mut perm = vec![0i32; n_tokens * top_k];
    let mut inv_perm = vec![0i32; n_tokens * top_k];
    w_permute::moe_permute(
        wg,
        &topk_ids,
        &mut offsets,
        &mut perm,
        &mut inv_perm,
        n_tokens,
        top_k,
        e_total,
    )
    .expect("wgpu moe_permute");
    eprintln!("routed offsets: {offsets:?}");
    assert_eq!(*offsets.last().unwrap() as usize, n_tokens * top_k);

    let token_rows = gen_rows(n_tokens, k_dim, Profile::Uniform, 0.07, 0.0);
    let permuted: Vec<Vec<f32>> = perm
        .iter()
        .map(|&t| token_rows[t as usize].clone())
        .collect();
    let a = Nvfp4Tensor::quantize_rows(&permuted);
    let a_sf = swizzle_scales(&a.scales, permuted.len(), k_dim / BLOCK_SIZE);

    let mut b_packed = Vec::new();
    let mut b_scales = Vec::new();
    let mut b_deq = Vec::new();
    for e in 0..e_total {
        let rows = gen_rows(n, k_dim, Profile::Uniform, 0.09, 1.7 + e as f32 * 0.31);
        let b = Nvfp4Tensor::quantize_rows(&rows);
        b_scales.extend_from_slice(&swizzle_scales(&b.scales, n, k_dim / BLOCK_SIZE));
        b_packed.extend_from_slice(&b.data);
        b_deq.push(b.dequantize());
    }
    let ids: Vec<i32> = (0..e_total as i32).collect();
    let alphas = vec![1.0f32; e_total];

    let a_deq = a.dequantize();
    let want = cpu_oracle_blocked(&a_deq, &b_deq, &offsets, &ids, &alphas, n, k_dim);

    for path in [GemmPath::Scalar, GemmPath::CoopMat] {
        if path == GemmPath::CoopMat && !coop_available(wg, "moe_grouped_routed") {
            continue;
        }
        let mut got = vec![0u16; n_tokens * top_k * n];
        w_moe::moe_grouped_nvfp4_gemm_bf16(
            wg, &a.data, &a_sf, &b_packed, &b_scales, &offsets, &ids, &alphas, &mut got, n, k_dim,
            e_total, path,
        )
        .expect("grouped gemm after routing");
        let s = compare(&got, &want);
        eprintln!("routed top_k=2 {path:?} vs k16sum: {}", s.render());
        assert_eq!(
            s.differ, 0,
            "grouped GEMM fed by the wgpu moe_permute routing must match the oracle bit for bit"
        );
    }
}

#[cfg(feature = "cuda")]
mod cuda_gate {
    use super::*;
    use cudarc::driver::sys::CUdevice_attribute;
    use cudarc::driver::{CudaContext, CudaStream, DevicePtr, DevicePtrMut};
    use nv_quant::nvfp4::supports_nvfp4;
    use std::ffi::c_void;
    use std::sync::Arc;

    const CUDA_VS_ORACLE_DIFFERING: usize = 0;
    const CUDA_VS_ORACLE_MAX_ULP: i32 = 0;

    fn cuda_backend(test: &str) -> Option<Arc<CudaStream>> {
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                if require() {
                    panic!("{test}: no CUDA device 0: {e}");
                }
                eprintln!("{test}: SKIP no CUDA device 0: {e}");
                return None;
            }
        };
        let major = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0);
        let minor = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0);
        if !supports_nvfp4(major) || (major, minor) != (12, 0) {
            if require() {
                panic!("{test}: requires SM 12.0, got SM {major}.{minor}");
            }
            eprintln!("{test}: SKIP requires SM 12.0 (got SM {major}.{minor})");
            return None;
        }
        Some(ctx.default_stream())
    }

    fn run_cuda_grouped(
        stream: &Arc<CudaStream>,
        o: &Operands,
        c: &Case,
        m_per_group: usize,
    ) -> Vec<u16> {
        let a = o.ids.len();
        #[allow(deprecated)]
        let a_packed = stream.clone_htod(&o.a.data[..o.total_m * c.k / 2]).unwrap();
        #[allow(deprecated)]
        let a_scales = stream.clone_htod(&o.a_sf).unwrap();
        #[allow(deprecated)]
        let b_packed = stream.clone_htod(&o.b_packed).unwrap();
        #[allow(deprecated)]
        let b_scales = stream.clone_htod(&o.b_scales).unwrap();
        let alphas_host = vec![1.0f32; c.e_total];
        #[allow(deprecated)]
        let alphas = stream.clone_htod(&alphas_host).unwrap();
        let eo_host: Vec<i32> = o.offsets[..a].to_vec();
        #[allow(deprecated)]
        let expert_offsets = stream.clone_htod(&eo_host).unwrap();
        #[allow(deprecated)]
        let sf_offsets = stream.clone_htod(&eo_host).unwrap();
        let mut ps_host = Vec::new();
        for _ in 0..a {
            ps_host.extend_from_slice(&[m_per_group as i32, c.n as i32, c.k as i32]);
        }
        #[allow(deprecated)]
        let problem_sizes = stream.clone_htod(&ps_host).unwrap();
        #[allow(deprecated)]
        let active_ids = stream.clone_htod(&o.ids).unwrap();

        let mut d = stream.alloc_zeros::<bf16>(o.total_m * c.n).unwrap();
        let mut meta = stream.alloc_zeros::<u8>(128 * 1024).unwrap();
        let mut ws = stream.alloc_zeros::<u8>(64 * 1024 * 1024).unwrap();
        {
            let (ap, _g1) = a_packed.device_ptr(stream);
            let (asp, _g2) = a_scales.device_ptr(stream);
            let (bp, _g3) = b_packed.device_ptr(stream);
            let (bsp, _g4) = b_scales.device_ptr(stream);
            let (alp, _g5) = alphas.device_ptr(stream);
            let (dp, _g6) = d.device_ptr_mut(stream);
            let (eo, _g7) = expert_offsets.device_ptr(stream);
            let (sfo, _g8) = sf_offsets.device_ptr(stream);
            let (ps, _g9) = problem_sizes.device_ptr(stream);
            let (aei, _ga) = active_ids.device_ptr(stream);
            let (ms, _gb) = meta.device_ptr_mut(stream);
            let (wsp, _gc) = ws.device_ptr_mut(stream);
            unsafe {
                nv_kernels::cuda::cutlass_moe_grouped_fp4_gemm_sm120_bf16(
                    stream.cu_stream() as *mut c_void,
                    ap as *const c_void,
                    asp as *const c_void,
                    bp as *const c_void,
                    bsp as *const c_void,
                    alp as *const f32,
                    dp as *mut c_void,
                    eo as *const i32,
                    sfo as *const i32,
                    ps as *const i32,
                    aei as *const i32,
                    c.n as i32,
                    c.k as i32,
                    a as i32,
                    c.k as i64,
                    c.k as i64,
                    c.n as i64,
                    ms as *mut c_void,
                    128 * 1024,
                    wsp as *mut c_void,
                    64 * 1024 * 1024,
                )
                .expect("cuda grouped FP4 GEMM launch")
            };
        }
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        stream
            .memcpy_dtov(&d)
            .unwrap()
            .iter()
            .map(|x: &bf16| x.to_bits())
            .collect()
    }

    #[test]
    fn moe_grouped_wgpu_matches_the_cuda_cutlass_grouped_path() {
        let test = "moe_grouped_cuda_gate";
        let Some(stream) = cuda_backend(test) else {
            return;
        };
        let Some(wg) = backend(test) else { return };
        let coop = coop_available(wg, test);

        let m_per_group = 128usize;
        let c = case(
            "cuda_gate_remapped",
            8,
            vec![
                (3, m_per_group),
                (1, m_per_group),
                (6, m_per_group),
                (0, m_per_group),
            ],
            256,
            512,
        );
        let o = operands(&c);
        let a_deq = o.a.dequantize();
        let oracle = cpu_oracle_blocked(&a_deq, &o.b_deq, &o.offsets, &o.ids, &c.alphas, c.n, c.k);

        let cu = run_cuda_grouped(&stream, &o, &c, m_per_group);
        let nz = cu
            .iter()
            .filter(|b| bf16::from_bits(**b).to_f32() != 0.0)
            .count();
        eprintln!("cuda grouped non-zero = {nz}/{}", cu.len());
        assert!(
            nz > cu.len() / 4,
            "cuda grouped output mostly zero -- kernel did not run"
        );

        let cuda_vs_oracle = compare(&cu, &oracle);
        eprintln!("cuda   vs k16sum: {}", cuda_vs_oracle.render());

        let mut paths = vec![GemmPath::Scalar];
        if coop {
            paths.push(GemmPath::CoopMat);
        }
        for path in paths {
            let got = wgpu_grouped(wg, &o, &c, path);
            let vs_oracle = compare(&got, &oracle);
            let vs_cuda = compare(&got, &cu);
            eprintln!(
                "wgpu {path:?} vs k16sum: {} | vs cuda: {}",
                vs_oracle.render(),
                vs_cuda.render()
            );
            assert_eq!(
                vs_oracle.differ, 0,
                "wgpu {path:?} must reproduce the blocked oracle bit for bit"
            );
            assert!(
                vs_cuda.rel_rms < 1e-3,
                "wgpu {path:?} diverges from the CUDA grouped path: rel_rms={}",
                vs_cuda.rel_rms
            );
            assert_eq!(
                (vs_cuda.differ, vs_cuda.max_ulp),
                (cuda_vs_oracle.differ, cuda_vs_oracle.max_ulp),
                "wgpu is bit-exact vs the oracle, so its residual against CUDA must be exactly \
                 CUDA's own residual against the oracle"
            );
        }
        assert_eq!(
            (cuda_vs_oracle.differ, cuda_vs_oracle.max_ulp),
            (CUDA_VS_ORACLE_DIFFERING, CUDA_VS_ORACLE_MAX_ULP),
            "measured CUDA-vs-oracle residual moved; re-measure and re-pin"
        );
    }
}
