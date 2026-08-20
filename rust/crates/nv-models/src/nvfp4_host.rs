use nv_kernels::wgpu_backend::kernels as wk;

pub const NVFP4_BLOCK: usize = 16;

#[derive(Clone, Debug)]
pub struct HostNvfp4Lin {
    pub packed: Vec<u8>,
    pub scales_swizzled: Vec<u8>,
    pub alpha: f32,
    pub input_global: f32,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Debug)]
pub struct HostNvfp4ExpertStack {
    pub packed: Vec<u8>,
    pub scales_swizzled: Vec<u8>,
    pub alphas: Vec<f32>,
    pub input_globals: Vec<f32>,
    pub e: usize,
    pub n: usize,
    pub k: usize,
}

pub fn quantize_nvfp4_host(w: &[u16], n: usize, k: usize) -> HostNvfp4Lin {
    let mut amax = 0f32;
    for v in w {
        let f = f32::from_bits((*v as u32) << 16).abs();
        if f > amax {
            amax = f;
        }
    }
    let stored_global = if amax.is_finite() && amax > 0.0 {
        (448.0f32 * 6.0) / amax
    } else {
        1.0
    };
    let alpha = if stored_global.is_finite() && stored_global != 0.0 {
        1.0 / stored_global
    } else {
        1.0
    };
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
    for r in 0..n {
        rows.push(
            w[r * k..(r + 1) * k]
                .iter()
                .map(|v| f32::from_bits((*v as u32) << 16))
                .collect(),
        );
    }
    let q = nv_quant::nvfp4::Nvfp4Tensor::quantize_rows_with_global(&rows, stored_global);
    let scales_swizzled = q.scales_swizzled();
    HostNvfp4Lin {
        packed: q.data,
        scales_swizzled,
        alpha,
        input_global: 1.0,
        n,
        k,
    }
}

pub fn dequantize_nvfp4_host(lin: &HostNvfp4Lin) -> Vec<f32> {
    let k_blocks = lin.k / NVFP4_BLOCK;
    let k_tiles = k_blocks.div_ceil(4);
    let mut out = vec![0f32; lin.n * lin.k];
    for r in 0..lin.n {
        for kb in 0..k_blocks {
            let m_tile = r / 128;
            let d2 = (r / 32) % 4;
            let d3 = r % 32;
            let k_tile = kb / 4;
            let d5 = kb % 4;
            let si = ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5;
            let sb = lin.scales_swizzled[si] as u32;
            let e = (sb >> 3) & 15;
            let m = sb & 7;
            let s = if e == 0 {
                (m as f32) * 0.001953125f32
            } else {
                f32::from_bits(((e + 120) << 23) | (m << 20))
            };
            for j in 0..NVFP4_BLOCK {
                let idx = kb * NVFP4_BLOCK + j;
                let byte = lin.packed[r * (lin.k / 2) + idx / 2];
                let nib = if idx.is_multiple_of(2) {
                    byte & 15
                } else {
                    byte >> 4
                };
                const TABLE: [f32; 16] = [
                    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0,
                    -4.0, -6.0,
                ];
                out[r * lin.k + idx] = TABLE[nib as usize] * s * lin.alpha;
            }
        }
    }
    out
}

pub fn stack_nvfp4_host(mats: &[HostNvfp4Lin]) -> HostNvfp4ExpertStack {
    let n = mats[0].n;
    let k = mats[0].k;
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut alphas = Vec::new();
    let mut globals = Vec::new();
    for m in mats {
        packed.extend_from_slice(&m.packed);
        scales.extend_from_slice(&m.scales_swizzled);
        alphas.push(m.alpha);
        globals.push(m.input_global);
    }
    HostNvfp4ExpertStack {
        packed,
        scales_swizzled: scales,
        alphas,
        input_globals: globals,
        e: mats.len(),
        n,
        k,
    }
}

pub fn expert_slice(stack: &HostNvfp4ExpertStack, e: usize) -> HostNvfp4Lin {
    let pb = stack.n * stack.k / 2;
    let sb = wk::gemv_nvfp4::swizzled_scale_len(stack.n, stack.k / NVFP4_BLOCK);
    HostNvfp4Lin {
        packed: stack.packed[e * pb..(e + 1) * pb].to_vec(),
        scales_swizzled: stack.scales_swizzled[e * sb..(e + 1) * sb].to_vec(),
        alpha: stack.alphas[e],
        input_global: stack.input_globals[e],
        n: stack.n,
        k: stack.k,
    }
}

pub fn quantize_nvfp4_stack_i8(stack: &HostNvfp4ExpertStack, group: usize) -> (Vec<u32>, Vec<f32>) {
    let (n, k) = (stack.n, stack.k);
    let gpr = k.checked_div(group).unwrap_or(1);
    let span = if group > 0 { group } else { k };
    let mut packed = vec![0u32; stack.e * n * k / 4];
    let mut scales = vec![0f32; stack.e * n * gpr];
    for e in 0..stack.e {
        let gi = stack.input_globals[e];
        let gi = if gi == 0.0 || !gi.is_finite() {
            1.0
        } else {
            gi
        };
        let wf = dequantize_nvfp4_host(&expert_slice(stack, e));
        for r in 0..n {
            let row = &wf[r * k..(r + 1) * k];
            let base = e * n * k + r * k;
            for g in 0..gpr {
                let lo = g * span;
                let hi = lo + span;
                let mut max_abs = 0f32;
                for &v in &row[lo..hi] {
                    max_abs = max_abs.max((v * gi).abs());
                }
                let sc = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
                scales[(e * n + r) * gpr + g] = sc;
                for (i, &v) in row[lo..hi].iter().enumerate() {
                    let idx = base + lo + i;
                    let q = (v * gi / sc).round().clamp(-127.0, 127.0) as i32 as u32 & 0xff;
                    packed[idx / 4] |= q << (8 * (idx % 4));
                }
            }
        }
    }
    (packed, scales)
}

pub fn w8_group_from_env(var: &str) -> usize {
    let g = std::env::var(var)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(128);
    assert!(
        g == 0 || (g >= 32 && g.is_power_of_two()),
        "{var} must be 0 (per-row) or a power of two >= 32; got {g}. \
         The GEMV indexes scales by a shift over 4-element words, so a group must be \
         a power-of-two multiple of 4 elements, and >= 32 keeps the scale buffer small."
    );
    g
}

pub struct Nvfp4Gpu {
    pub w: wgpu::Buffer,
    pub scales: wgpu::Buffer,
    pub alpha: f32,
    pub input_global: f32,
    pub n: usize,
    pub k: usize,
}

pub fn upload_nvfp4(
    b: &mut crate::wgpu_ledger::VramLedger,
    label: &str,
    l: &HostNvfp4Lin,
) -> Nvfp4Gpu {
    let words = crate::wgpu_ledger::bytes_to_words;
    Nvfp4Gpu {
        w: b.upload_u32(label, &words(&l.packed)),
        scales: b.upload_u32(&format!("{label}-sf"), &words(&l.scales_swizzled)),
        alpha: l.alpha,
        input_global: l.input_global,
        n: l.n,
        k: l.k,
    }
}

pub fn assert_w8_group_divides_k(label: &str, group: usize, k: usize, env_var: &str) {
    assert!(
        group == 0 || (k.is_multiple_of(group) && k >= group),
        "{label}: {env_var}={group} does not divide k={k}; refusing to fall back to per-row \
         int8, which the quality battery rejected"
    );
}
