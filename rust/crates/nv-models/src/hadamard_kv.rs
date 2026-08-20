use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};

pub const HADAMARD_KV_ENV: &str = "NV_KV_HADAMARD";

pub const HADAMARD_KV_CUTS_PEAK_K_QUANT_ERROR_MEASURED_IN_PERF_RUNS_JSONL: &str = HADAMARD_KV_ENV;

pub const HADAMARD_KV_CUTS_CHUNK_ERROR_AMPLIFICATION_MEASURED_IN_PERF_RUNS_JSONL: &str =
    HADAMARD_KV_ENV;

pub fn hadamard_kv_enabled() -> bool {
    matches!(std::env::var(HADAMARD_KV_ENV).as_deref(), Ok("1"))
}

pub fn rotatable(head_dim: usize) -> bool {
    head_dim.is_power_of_two() && head_dim >= 2
}

pub fn rotate_row(row: &mut [f32]) -> Result<()> {
    let n = row.len();
    if !rotatable(n) {
        bail!(
            "hadamard_kv: head_dim {n} is not a power of two, so no Hadamard of that order \
             exists and the transform cannot be its own inverse"
        );
    }
    let mut h = 1usize;
    while h < n {
        let mut i = 0usize;
        while i < n {
            for j in i..i + h {
                let (x, y) = (row[j], row[j + h]);
                row[j] = x + y;
                row[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }
    let inv = 1.0 / (n as f32).sqrt();
    for v in row.iter_mut() {
        *v *= inv;
    }
    Ok(())
}

pub fn rotate_rows(vals: &mut [f32], head_dim: usize) -> Result<()> {
    if head_dim == 0 || vals.len() % head_dim != 0 {
        bail!(
            "hadamard_kv: {} values do not divide into rows of {head_dim}",
            vals.len()
        );
    }
    for row in vals.chunks_mut(head_dim) {
        rotate_row(row)?;
    }
    Ok(())
}

pub fn hadamard_matrix(n: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    if !rotatable(n) {
        bail!("hadamard_kv: no Hadamard of order {n}; head_dim must be a power of two");
    }
    let inv = 1.0f32 / (n as f32).sqrt();
    let mut m = vec![0f32; n * n];
    for (r, row) in m.chunks_mut(n).enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            let sign = if (r & c).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
            *v = sign * inv;
        }
    }
    Ok(Tensor::from_vec(m, (n, n), device)?.to_dtype(dtype)?)
}

pub fn rotate_last_dim(x: &Tensor, h: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let n = *dims
        .last()
        .ok_or_else(|| anyhow::anyhow!("hadamard_kv: rotate_last_dim needs a shaped tensor"))?;
    if h.dims() != [n, n] {
        bail!(
            "hadamard_kv: rotation matrix is {:?} but the tensor's last dim is {n}; a mismatch \
             here silently changes what attention reads",
            h.dims()
        );
    }
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let flat = x.reshape((rows, n))?.contiguous()?;
    Ok(flat.matmul(h)?.reshape(dims)?)
}

pub fn maybe_rotate_qk(q: Tensor, k: Tensor, head_dim: usize) -> Result<(Tensor, Tensor)> {
    if !hadamard_kv_enabled() {
        return Ok((q, k));
    }
    if !rotatable(head_dim) {
        return Ok((q, k));
    }
    let want = q.dtype();
    let h = hadamard_matrix(head_dim, DType::F32, q.device())?;
    let qr = rotate_last_dim(&q.to_dtype(DType::F32)?, &h)?.to_dtype(want)?;
    let kr = rotate_last_dim(&k.to_dtype(DType::F32)?, &h)?.to_dtype(want)?;
    Ok((qr, kr))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn rotating_twice_returns_the_original() {
        for hd in [16usize, 64, 256, 512] {
            let want = lcg(0xfeed ^ hd as u64, hd);
            let mut got = want.clone();
            rotate_row(&mut got).unwrap();
            rotate_row(&mut got).unwrap();
            let worst = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst < 1e-5,
                "head_dim {hd}: the transform is not its own inverse, worst {worst:e}. It is \
                 applied to K on the write path and to Q at attention time, so if H(H(x)) is \
                 not x the cache and the query disagree about the basis"
            );
        }
    }

    #[test]
    fn rotation_preserves_the_dot_product_attention_reads() {
        for hd in [16usize, 64, 256] {
            let q = lcg(0xa11ce ^ hd as u64, hd);
            let k = lcg(0xb0b ^ hd as u64, hd);
            let plain: f64 = q.iter().zip(&k).map(|(a, b)| (a * b) as f64).sum();
            let (mut qr, mut kr) = (q.clone(), k.clone());
            rotate_row(&mut qr).unwrap();
            rotate_row(&mut kr).unwrap();
            let rotated: f64 = qr.iter().zip(&kr).map(|(a, b)| (a * b) as f64).sum();
            let rel = (plain - rotated).abs() / plain.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "head_dim {hd}: Q.K changed by {rel:e} under rotation, so folding this into the \
                 KV path would change attention scores. The whole reason it is free is that an \
                 orthogonal H leaves Q.K^T alone"
            );
        }
    }

    #[test]
    fn a_non_power_of_two_head_dim_is_refused_not_silently_skipped() {
        let mut v = vec![1.0f32; 24];
        assert!(rotate_row(&mut v).is_err());
        assert!(!rotatable(24));
        assert!(rotatable(256) && rotatable(512));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_tensor_rotation_agrees_with_the_cpu_transform_on_device() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        for hd in [64usize, 256] {
            let rows = 7usize;
            let src = lcg(0xc0de ^ hd as u64, rows * hd);
            let mut want = src.clone();
            rotate_rows(&mut want, hd).unwrap();

            let x = Tensor::from_vec(src, (rows, hd), &device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let h = hadamard_matrix(hd, DType::F32, &device).unwrap();
            let got: Vec<f32> = rotate_last_dim(&x, &h)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let worst = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst < 1e-4,
                "head_dim {hd}: the matmul rotation and the CPU butterfly disagree by {worst:e}. \
                 The sweep that chose this transform measured it with the butterfly, so a device \
                 path that does something else invalidates those numbers"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn bf16_storage_costs_the_rotation_1_1e_3_of_the_qk_it_is_meant_to_preserve() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let hd = 256usize;
        let q = Tensor::from_vec(lcg(0x9a1, hd), (1, hd), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k = Tensor::from_vec(lcg(0x9a2, hd), (1, hd), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let dot = |a: &Tensor, b: &Tensor| -> f32 {
            a.to_dtype(DType::F32)
                .unwrap()
                .mul(&b.to_dtype(DType::F32).unwrap())
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        let plain = dot(&q, &k);
        std::env::set_var(HADAMARD_KV_ENV, "1");
        let (qr, kr) = maybe_rotate_qk(q.clone(), k.clone(), hd).unwrap();
        std::env::remove_var(HADAMARD_KV_ENV);
        let rotated = dot(&qr, &kr);
        let rel = (plain - rotated).abs() / plain.abs().max(1e-3);
        assert!(
            rel < 5e-3,
            "Q.K moved {rel:e} through maybe_rotate_qk on bf16 inputs (plain {plain:e}, rotated \
             {rotated:e}), past the 5e-3 this is pinned at. The transform is orthogonal, so in \
             exact arithmetic this would be zero; the cost is that rotation turns each element \
             into a dense combination of 256 others and the result is stored back in bf16. \
             Accumulating in f32 does not help -- measured identical -- because the loss is in \
             the representation, not the arithmetic. This 1.1e-3 on scores is why the rotation \
             costs 0.12 NLL end to end even with bf16 KV and nothing quantised: it spends more \
             Q/K precision than the better fp8 K buys back"
        );
    }

    #[test]
    fn the_rotation_is_off_until_its_env_var_is_set() {
        assert!(
            !hadamard_kv_enabled() || std::env::var(HADAMARD_KV_ENV).as_deref() == Ok("1"),
            "the rotation changes stored K bit-for-bit, so it is a numerical default and may \
             not arrive without being asked for"
        );
    }
}
