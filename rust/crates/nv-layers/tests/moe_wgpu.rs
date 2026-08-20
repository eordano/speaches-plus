#![cfg(feature = "wgpu")]

mod common;
use common::HostExperts;
use common::HostMat;
use common::sources;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::gemm_nvfp4::GemmPath;
use nv_kernels::wgpu_backend::kernels::moe_grouped_gemm as w_moe;
use nv_kernels::wgpu_backend::kernels::moe_unpermute_scatter as w_mus;
use nv_kernels::wgpu_backend::kernels::quantize_nvfp4_bf16 as w_quant;
use nv_layers::moe_wgpu::{self, MoeWgpuExpertSource, MoeWgpuWeights, MIN_TILE};
use nv_quant::nvfp4::{swizzle_scales, Nvfp4Tensor, BLOCK_SIZE};

#[path = "wgpu_common.rs"]
mod wgpu_common;

#[cfg(feature = "cuda")]
use wgpu_common::parity_require as require;
use wgpu_common::wgpu_ctx_or_skip as backend;

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn expert_mats(e_total: usize, n: usize, k: usize, seed: u64) -> HostMat {
    let mut rng = Lcg(seed);
    let rows: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..k).map(|_| rng.next_f32()).collect())
        .collect();
    let template = Nvfp4Tensor::quantize_rows(&rows);
    let row_bytes = k / 2;
    let row_scale_bytes = k / BLOCK_SIZE;
    let mut packed = Vec::with_capacity(e_total);
    let mut scales_swizzled = Vec::with_capacity(e_total);
    for e in 0..e_total {
        let shift = (e * 7) % n;
        let mut p = Vec::with_capacity(n * row_bytes);
        let mut s = Vec::with_capacity(n * row_scale_bytes);
        for r in 0..n {
            let src = (r + shift) % n;
            p.extend_from_slice(&template.data[src * row_bytes..(src + 1) * row_bytes]);
            s.extend_from_slice(
                &template.scales[src * row_scale_bytes..(src + 1) * row_scale_bytes],
            );
        }
        scales_swizzled.push(swizzle_scales(&s, n, k / BLOCK_SIZE));
        packed.push(p);
    }
    HostMat {
        packed,
        scales_swizzled,
    }
}

fn host_experts(e_total: usize, hidden: usize, inter: usize, seed: u64) -> HostExperts {
    let gate = expert_mats(e_total, inter, hidden, seed);
    let up = expert_mats(e_total, inter, hidden, seed ^ 0x5555_aaaa_1234_9876);
    let down = expert_mats(e_total, hidden, inter, seed ^ 0x0f0f_f0f0_dead_beef);
    let globals_gu: Vec<f32> = (0..e_total).map(|e| 1.5 + 0.01 * e as f32).collect();
    let globals_dn: Vec<f32> = (0..e_total).map(|e| 2.0 + 0.02 * e as f32).collect();
    HostExperts {
        gate,
        up,
        down,
        gate_alphas: globals_gu.iter().map(|g| 1.0 / g).collect(),
        up_alphas: globals_gu.iter().map(|g| 0.5 / g).collect(),
        down_alphas: globals_dn.iter().map(|g| 1.0 / g).collect(),
        globals_gu,
        globals_dn,
    }
}

fn routing(n_tokens: usize, k: usize, e_total: usize, seed: u64) -> (Vec<u32>, Vec<f32>) {
    let mut rng = Lcg(seed);
    let mut ids = Vec::with_capacity(n_tokens * k);
    let mut weights = Vec::with_capacity(n_tokens * k);
    for _ in 0..n_tokens {
        let mut chosen: Vec<u32> = Vec::with_capacity(k);
        while chosen.len() < k {
            let e = rng.next_u32() % e_total as u32;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        let raw: Vec<f32> = (0..k)
            .map(|_| 0.1 + (rng.next_u32() % 1000) as f32 / 1000.0)
            .collect();
        let z: f32 = raw.iter().sum();
        ids.extend_from_slice(&chosen);
        weights.extend(raw.iter().map(|w| w / z));
    }
    (ids, weights)
}

fn x_bf16(n_tokens: usize, hidden: usize, seed: u64) -> Vec<u16> {
    let mut rng = Lcg(seed);
    (0..n_tokens * hidden)
        .map(|_| half::bf16::from_f32(rng.next_f32()).to_bits())
        .collect()
}

struct Plan {
    active: Vec<usize>,
    m_total: usize,
    src_idx: Vec<i32>,
    inv_perm: Vec<i32>,
}

fn plan(topk_ids: &[u32], n_tokens: usize, k: usize, e_total: usize) -> Plan {
    let mut buckets: Vec<Vec<(u32, u32)>> = vec![Vec::new(); e_total];
    for n in 0..n_tokens {
        for j in 0..k {
            buckets[topk_ids[n * k + j] as usize].push((n as u32, j as u32));
        }
    }
    let active: Vec<usize> = (0..e_total).filter(|&e| !buckets[e].is_empty()).collect();
    let m_total = active.len() * MIN_TILE;
    let mut src_idx = vec![-1i32; m_total];
    let mut inv_perm = vec![-1i32; n_tokens * k];
    for (i, &e) in active.iter().enumerate() {
        for (j, &(token, slot)) in buckets[e].iter().enumerate().take(MIN_TILE) {
            src_idx[i * MIN_TILE + j] = token as i32;
            inv_perm[token as usize * k + slot as usize] = (i * MIN_TILE + j) as i32;
        }
    }
    Plan {
        active,
        m_total,
        src_idx,
        inv_perm,
    }
}

#[allow(clippy::too_many_arguments)]
fn staged_reference(
    ctx: &WgpuContext,
    h: &HostExperts,
    x: &[u16],
    topk_ids: &[u32],
    topk_weights: &[f32],
    n_tokens: usize,
    k: usize,
    e_total: usize,
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let p = plan(topk_ids, n_tokens, k, e_total);
    let a = p.active.len();
    let m_total = p.m_total;
    if a == 0 {
        return vec![0f32; n_tokens * hidden];
    }

    let mut x_sorted = vec![0u16; m_total * hidden];
    for (row, &s) in p.src_idx.iter().enumerate() {
        if s >= 0 {
            let s = s as usize;
            x_sorted[row * hidden..(row + 1) * hidden]
                .copy_from_slice(&x[s * hidden..(s + 1) * hidden]);
        }
    }

    let globals_gu: Vec<f32> = p.active.iter().map(|&e| h.globals_gu[e]).collect();
    let globals_dn: Vec<f32> = p.active.iter().map(|&e| h.globals_dn[e]).collect();

    let mut x_fp4 = vec![0u8; m_total * hidden / 2];
    let mut x_sf = vec![0u8; w_quant::swizzled_scale_bytes(m_total, hidden)];
    w_quant::quantize_nvfp4_bf16_per_expert(
        ctx,
        &x_sorted,
        &globals_gu,
        &[],
        &mut x_fp4,
        &mut x_sf,
        a,
        MIN_TILE,
        hidden,
    )
    .expect("staged quantize x");

    let mut gate_b = Vec::new();
    let mut gate_sf = Vec::new();
    let mut up_b = Vec::new();
    let mut up_sf = Vec::new();
    let mut down_b = Vec::new();
    let mut down_sf = Vec::new();
    for e in 0..e_total {
        gate_b.extend_from_slice(&h.gate.packed[e]);
        gate_sf.extend_from_slice(&h.gate.scales_swizzled[e]);
        up_b.extend_from_slice(&h.up.packed[e]);
        up_sf.extend_from_slice(&h.up.scales_swizzled[e]);
        down_b.extend_from_slice(&h.down.packed[e]);
        down_sf.extend_from_slice(&h.down.scales_swizzled[e]);
    }
    let offsets: Vec<i32> = (0..=a).map(|i| (i * MIN_TILE) as i32).collect();
    let ids: Vec<i32> = p.active.iter().map(|&e| e as i32).collect();
    let gate_alphas: Vec<f32> = p.active.iter().map(|&e| h.gate_alphas[e]).collect();
    let up_alphas: Vec<f32> = p.active.iter().map(|&e| h.up_alphas[e]).collect();
    let down_alphas: Vec<f32> = p.active.iter().map(|&e| h.down_alphas[e]).collect();

    let mut y_gate = vec![0u16; m_total * inter];
    w_moe::moe_grouped_nvfp4_gemm_bf16(
        ctx,
        &x_fp4,
        &x_sf,
        &gate_b,
        &gate_sf,
        &offsets,
        &ids,
        &gate_alphas,
        &mut y_gate,
        inter,
        hidden,
        e_total,
        GemmPath::Scalar,
    )
    .expect("staged gate gemm");
    let mut y_up = vec![0u16; m_total * inter];
    w_moe::moe_grouped_nvfp4_gemm_bf16(
        ctx,
        &x_fp4,
        &x_sf,
        &up_b,
        &up_sf,
        &offsets,
        &ids,
        &up_alphas,
        &mut y_up,
        inter,
        hidden,
        e_total,
        GemmPath::Scalar,
    )
    .expect("staged up gemm");

    let mut gate_up = Vec::with_capacity(2 * m_total * inter);
    gate_up.extend_from_slice(&y_gate);
    gate_up.extend_from_slice(&y_up);
    let mut act_fp4 = vec![0u8; m_total * inter / 2];
    let mut act_sf = vec![0u8; w_quant::swizzled_scale_bytes(m_total, inter)];
    w_quant::silu_mul_quantize_nvfp4_bf16_per_expert(
        ctx,
        &gate_up,
        &globals_dn,
        &[],
        &mut act_fp4,
        &mut act_sf,
        a,
        MIN_TILE,
        inter,
    )
    .expect("staged silu quantize");

    let mut y_down = vec![0u16; m_total * hidden];
    w_moe::moe_grouped_nvfp4_gemm_bf16(
        ctx,
        &act_fp4,
        &act_sf,
        &down_b,
        &down_sf,
        &offsets,
        &ids,
        &down_alphas,
        &mut y_down,
        hidden,
        inter,
        e_total,
        GemmPath::Scalar,
    )
    .expect("staged down gemm");

    let mut out = vec![0f32; n_tokens * hidden];
    w_mus::moe_unpermute_scatter(
        ctx,
        &y_down,
        topk_weights,
        &p.inv_perm,
        &mut out,
        n_tokens,
        k,
        hidden,
        hidden,
    )
    .expect("staged unpermute");
    out
}

fn f32_ord(x: f32) -> i64 {
    let b = x.to_bits() as i32;
    if b < 0 {
        (i32::MIN as i64) - (b as i64)
    } else {
        b as i64
    }
}

fn compare_f32(got: &[f32], want: &[f32]) -> (usize, i64) {
    assert_eq!(got.len(), want.len());
    let mut differ = 0usize;
    let mut max_ulp = 0i64;
    for (g, w) in got.iter().zip(want.iter()) {
        if g.to_bits() != w.to_bits() {
            differ += 1;
            max_ulp = max_ulp.max((f32_ord(*g) - f32_ord(*w)).abs());
        }
    }
    (differ, max_ulp)
}

#[test]
fn resident_forward_matches_staged_wrapper_pipeline_bit_for_bit() {
    let test = "moe_wgpu_resident_vs_staged";
    let Some(ctx) = backend(test) else { return };
    let (e_total, hidden, inter) = (16usize, 256usize, 192usize);
    let h = host_experts(e_total, hidden, inter, 0x1234_5678);
    let w = MoeWgpuWeights::from_expert_sources(ctx, hidden, inter, &sources(&h))
        .expect("upload weights");

    let cases: Vec<(&str, usize, usize, Vec<u32>, Vec<f32>)> = vec![
        {
            let (ids, wts) = routing(21, 4, e_total, 0xabcdef);
            ("ragged_random_topk4", 21, 4, ids, wts)
        },
        (
            "full_tile_single_expert",
            128,
            1,
            vec![3u32; 128],
            vec![1.0f32; 128],
        ),
        {
            let (raw_ids, wts) = routing(11, 3, 4, 0x51ee9);
            let ids: Vec<u32> = raw_ids.iter().map(|e| e * 5).collect();
            ("sparse_experts_with_empties", 11, 3, ids, wts)
        },
        ("one_token_one_expert", 1, 1, vec![7u32], vec![1.0f32]),
    ];

    for (name, n_tokens, k, ids, wts) in cases {
        let x = x_bf16(n_tokens, hidden, 0x9999 + n_tokens as u64);
        let got = moe_wgpu::try_forward(&w, ctx, &x, &ids, &wts, n_tokens, k)
            .expect("resident forward")
            .expect("resident forward should not decline");
        let want = staged_reference(ctx, &h, &x, &ids, &wts, n_tokens, k, e_total, hidden, inter);
        let nz_want = want.iter().filter(|v| **v != 0.0).count();
        let nz_got = got.iter().filter(|v| **v != 0.0).count();
        assert!(
            nz_want > want.len() / 4 && nz_got > got.len() / 4,
            "{name}: degenerate comparison -- staged reference {nz_want}/{} nonzero, resident \
             {nz_got}/{} nonzero; neither pipeline demonstrably ran",
            want.len(),
            got.len()
        );
        let (differ, max_ulp) = compare_f32(&got, &want);
        let p = plan(&ids, n_tokens, k, e_total);
        eprintln!(
            "{name}: tokens={n_tokens} k={k} active={}/{e_total} m_total={} -> {differ}/{} differ max_ulp={max_ulp}",
            p.active.len(),
            p.m_total,
            got.len()
        );
        assert_eq!(
            differ, 0,
            "{name}: resident chain must equal the staged wrapper pipeline bit for bit"
        );
    }
}

#[test]
fn oversubscribed_expert_declines_like_cuda() {
    let test = "moe_wgpu_declines";
    let Some(ctx) = backend(test) else { return };
    let (e_total, hidden, inter) = (4usize, 128usize, 64usize);
    let h = host_experts(e_total, hidden, inter, 0x777);
    let w = MoeWgpuWeights::from_expert_sources(ctx, hidden, inter, &sources(&h))
        .expect("upload weights");
    let n_tokens = MIN_TILE + 1;
    let ids = vec![0u32; n_tokens];
    let wts = vec![1.0f32; n_tokens];
    let x = x_bf16(n_tokens, hidden, 0x88);
    let got = moe_wgpu::try_forward(&w, ctx, &x, &ids, &wts, n_tokens, 1).expect("forward");
    assert!(
        got.is_none(),
        "counts > MIN_TILE must return None (CUDA fallback contract)"
    );
}

#[test]
fn empty_token_batch_returns_empty() {
    let test = "moe_wgpu_empty";
    let Some(ctx) = backend(test) else { return };
    let (e_total, hidden, inter) = (4usize, 128usize, 64usize);
    let h = host_experts(e_total, hidden, inter, 0x778);
    let w = MoeWgpuWeights::from_expert_sources(ctx, hidden, inter, &sources(&h))
        .expect("upload weights");
    let got = moe_wgpu::try_forward(&w, ctx, &[], &[], &[], 0, 4).expect("forward");
    assert_eq!(got, Some(Vec::new()));
}

#[cfg(feature = "cuda")]
mod cuda_parity {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use cudarc::driver::sys::CUdevice_attribute;
    use half::bf16;
    use nv_layers::moe_grouped::{self, MoeGroupedWeights};
    use nv_quant::nvfp4::{supports_nvfp4, Nvfp4GemmRunner};
    use std::sync::{Arc, Mutex};

    fn cuda_device(test: &str) -> Option<Device> {
        let ctx = match cudarc::driver::CudaContext::new(0) {
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
        match Device::new_cuda(0) {
            Ok(d) => Some(d),
            Err(e) => {
                if require() {
                    panic!("{test}: candle cuda device: {e}");
                }
                eprintln!("{test}: SKIP candle cuda device: {e}");
                None
            }
        }
    }

    fn cuda_grouped_weights_pick(
        device: &Device,
        h: &HostExperts,
        pick: &[usize],
        hidden: usize,
        inter: usize,
    ) -> MoeGroupedWeights {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = dev.cuda_stream();
        let runner = Arc::new(Mutex::new(
            Nvfp4GemmRunner::new(stream.clone()).expect("nvfp4 runner"),
        ));
        let concat = |mats: &HostMat| -> (Vec<u8>, Vec<u8>) {
            let mut p = Vec::new();
            let mut s = Vec::new();
            for &e in pick {
                p.extend_from_slice(&mats.packed[e]);
                s.extend_from_slice(&mats.scales_swizzled[e]);
            }
            (p, s)
        };
        let sub = |v: &[f32]| -> Vec<f32> { pick.iter().map(|&e| v[e]).collect() };
        let (gate_p, gate_s) = concat(&h.gate);
        let (up_p, up_s) = concat(&h.up);
        let (down_p, down_s) = concat(&h.down);
        #[allow(deprecated)]
        let htod_u8 = |v: &[u8]| stream.clone_htod(v).expect("htod");
        #[allow(deprecated)]
        let htod_f32 = |v: &[f32]| stream.clone_htod(v).expect("htod");
        MoeGroupedWeights {
            num_experts: pick.len(),
            hidden_size: hidden,
            intermediate_size: inter,
            gate_w: htod_u8(&gate_p),
            gate_w_scales: htod_u8(&gate_s),
            gate_alphas: htod_f32(&sub(&h.gate_alphas)),
            gate_a_stride_elems: hidden as i64,
            gate_b_stride_elems: hidden as i64,
            gate_c_stride_elems: inter as i64,
            up_w: htod_u8(&up_p),
            up_w_scales: htod_u8(&up_s),
            up_alphas: htod_f32(&sub(&h.up_alphas)),
            down_w: htod_u8(&down_p),
            down_w_scales: htod_u8(&down_s),
            down_alphas: htod_f32(&sub(&h.down_alphas)),
            down_a_stride_elems: inter as i64,
            down_b_stride_elems: inter as i64,
            down_c_stride_elems: hidden as i64,
            runner,
            input_globals_gate_up: htod_f32(&sub(&h.globals_gu)),
            input_globals_down: htod_f32(&sub(&h.globals_dn)),
            input_globals_gate_up_host: sub(&h.globals_gu),
            input_globals_down_host: sub(&h.globals_dn),
        }
    }

    fn cuda_grouped_weights(
        device: &Device,
        h: &HostExperts,
        e_total: usize,
        hidden: usize,
        inter: usize,
    ) -> MoeGroupedWeights {
        let all: Vec<usize> = (0..e_total).collect();
        cuda_grouped_weights_pick(device, h, &all, hidden, inter)
    }

    fn qwen36_dims() -> Option<(usize, usize, usize, usize)> {
        let home = std::env::var("HOME").ok()?;
        let snaps = format!(
            "{home}/.cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots"
        );
        let dir = std::fs::read_dir(snaps).ok()?;
        for entry in dir.flatten() {
            let cfg = entry.path().join("config.json");
            if let Ok(text) = std::fs::read_to_string(&cfg) {
                let tail = match text.find("\"text_config\"") {
                    Some(i) => &text[i..],
                    None => &text[..],
                };
                let grab = |key: &str| -> Option<usize> {
                    let i = tail.find(&format!("\"{key}\""))?;
                    let rest = &tail[i..];
                    let colon = rest.find(':')?;
                    let digits: String = rest[colon + 1..]
                        .chars()
                        .skip_while(|c| c.is_whitespace())
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    digits.parse().ok()
                };
                let hidden = grab("hidden_size")?;
                let experts = grab("num_experts")?;
                let inter = grab("moe_intermediate_size")?;
                let top_k = grab("num_experts_per_tok")?;
                eprintln!(
                    "qwen3.6 config: hidden={hidden} experts={experts} moe_inter={inter} top_k={top_k} ({})",
                    cfg.display()
                );
                return Some((experts, hidden, inter, top_k));
            }
        }
        None
    }

    fn run_case(
        name: &str,
        e_total: usize,
        hidden: usize,
        inter: usize,
        n_tokens: usize,
        k: usize,
    ) -> Option<(usize, i64, usize)> {
        let Some(device) = cuda_device(name) else {
            return None;
        };
        let Some(ctx) = backend(name) else {
            return None;
        };

        let h = host_experts(e_total, hidden, inter, 0xfeed_0000 + e_total as u64);
        let (ids, wts) = routing(n_tokens, k, e_total, 0xc0ffee + n_tokens as u64);
        let x = x_bf16(n_tokens, hidden, 0xbead + hidden as u64);

        let cuda_w = cuda_grouped_weights(&device, &h, e_total, hidden, inter);
        let x_vals: Vec<bf16> = x.iter().map(|b| bf16::from_bits(*b)).collect();
        let x_t = Tensor::from_vec(x_vals, (n_tokens, hidden), &device).expect("x tensor");
        let cuda_out_t =
            moe_grouped::forward_grouped(&cuda_w, &cuda_w, &x_t, &ids, &wts, n_tokens, k, &device)
                .expect("cuda forward_grouped");
        assert_eq!(cuda_out_t.dims(), &[n_tokens, hidden]);
        assert_eq!(cuda_out_t.dtype(), DType::F32);
        let cuda_out: Vec<f32> = cuda_out_t.flatten_all().unwrap().to_vec1().unwrap();

        let wgpu_w = MoeWgpuWeights::from_expert_sources(ctx, hidden, inter, &sources(&h))
            .expect("wgpu weights");
        let wgpu_out = moe_wgpu::try_forward(&wgpu_w, ctx, &x, &ids, &wts, n_tokens, k)
            .expect("wgpu forward")
            .expect("wgpu forward should not decline");

        let nz = cuda_out.iter().filter(|v| **v != 0.0).count();
        assert!(
            nz > cuda_out.len() / 4,
            "{name}: cuda output mostly zero ({nz}/{}) -- reference did not run",
            cuda_out.len()
        );
        let (differ, max_ulp) = compare_f32(&wgpu_out, &cuda_out);
        let p = plan(&ids, n_tokens, k, e_total);
        eprintln!(
            "{name}: E={e_total} hidden={hidden} inter={inter} tokens={n_tokens} k={k} active={} -> wgpu vs cuda: {differ}/{} differ max_ulp={max_ulp}",
            p.active.len(),
            cuda_out.len()
        );
        Some((differ, max_ulp, cuda_out.len()))
    }

    #[test]
    fn gemma4_26b_shapes_wgpu_matches_cuda_forward_grouped() {
        if let Some((differ, max_ulp, total)) =
            run_case("moe_e2e_gemma4_26b", 128, 2816, 704, 13, 8)
        {
            assert_eq!(
                (differ, max_ulp),
                (0, 0),
                "wgpu MoE forward must be bit-exact against the CUDA grouped path at \
                 Gemma4-26B-A4B shapes ({differ}/{total} differ, max_ulp={max_ulp})"
            );
        }
    }

    #[test]
    fn qwen36_shapes_wgpu_matches_cuda_forward_grouped() {
        let Some((e_total, hidden, inter, top_k)) = qwen36_dims() else {
            eprintln!("moe_e2e_qwen36: SKIP no Qwen3.6 config.json in HF cache");
            return;
        };
        if let Some((differ, max_ulp, total)) =
            run_case("moe_e2e_qwen36", e_total, hidden, inter, 9, top_k)
        {
            assert_eq!(
                (differ, max_ulp),
                (0, 0),
                "wgpu MoE forward must be bit-exact against the CUDA grouped path at \
                 Qwen3.6-35B-A3B shapes ({differ}/{total} differ, max_ulp={max_ulp})"
            );
        }
    }

    fn route_all_to(e: u32, n_tokens: usize) -> (Vec<u32>, Vec<f32>) {
        (vec![e; n_tokens], vec![1.0f32; n_tokens])
    }

    fn cuda_out(
        device: &Device,
        w: &MoeGroupedWeights,
        x: &[u16],
        n_tokens: usize,
        hidden: usize,
        ids: &[u32],
        wts: &[f32],
    ) -> Vec<f32> {
        let x_vals: Vec<bf16> = x.iter().map(|b| bf16::from_bits(*b)).collect();
        let x_t = Tensor::from_vec(x_vals, (n_tokens, hidden), device).expect("x tensor");
        let out = moe_grouped::forward_grouped(w, w, &x_t, ids, wts, n_tokens, 1, device)
            .expect("cuda forward_grouped");
        assert_eq!(out.dtype(), DType::F32);
        out.flatten_all().unwrap().to_vec1().unwrap()
    }

    #[test]
    fn cuda_grouped_per_expert_scale_stride_is_index_invariant() {
        let name = "cuda_grouped_per_expert_scale_stride_is_index_invariant";
        let Some(device) = cuda_device(name) else {
            return;
        };
        let (e_total, hidden, n_tokens) = (4usize, 2816usize, 13usize);
        let mut bad: Vec<(usize, usize, usize, usize, f64)> = Vec::new();
        for inter in [512usize, 704] {
            let h = host_experts(e_total, hidden, inter, 0x5717_0000 + inter as u64);
            let x = x_bf16(n_tokens, hidden, 0x9a31_0000 + inter as u64);
            let all: Vec<usize> = (0..e_total).collect();
            let w_all = cuda_grouped_weights_pick(&device, &h, &all, hidden, inter);
            for e in 0..e_total {
                let w_solo = cuda_grouped_weights_pick(&device, &h, &[e], hidden, inter);
                let (ids_all, wts) = route_all_to(e as u32, n_tokens);
                let (ids_solo, _) = route_all_to(0, n_tokens);
                let got = cuda_out(&device, &w_all, &x, n_tokens, hidden, &ids_all, &wts);
                let want = cuda_out(&device, &w_solo, &x, n_tokens, hidden, &ids_solo, &wts);
                let nz = want.iter().filter(|v| **v != 0.0).count();
                assert!(
                    nz > want.len() / 4,
                    "{name}: degenerate reference at inter={inter} e={e} ({nz}/{})",
                    want.len()
                );
                let mut differ = 0usize;
                let mut max_rel = 0f64;
                for (a, b) in got.iter().zip(want.iter()) {
                    if a.to_bits() != b.to_bits() {
                        differ += 1;
                        let den = (*a as f64).abs().max((*b as f64).abs()).max(1e-30);
                        max_rel = max_rel.max((*a as f64 - *b as f64).abs() / den);
                    }
                }
                eprintln!(
                    "{name}: hidden={hidden} inter={inter} (inter%128={}) expert={e}: \
                     {differ}/{} differ max_rel={max_rel:.3e}",
                    inter % 128,
                    got.len()
                );
                if differ > 0 {
                    bad.push((inter, e, differ, got.len(), max_rel));
                }
            }
        }
        assert!(
            bad.is_empty(),
            "CUDA grouped MoE output must not depend on which slot an expert occupies in the \
             concatenated weight stack; the per-expert scale-factor stride must match \
             swizzled_scale_bytes(N, K) = round_up(N,128) * round_up(K/16,4). \
             failures (inter, expert, differ, total, max_rel) = {bad:?}"
        );
    }
}
