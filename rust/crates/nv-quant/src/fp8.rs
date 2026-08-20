use anyhow::{bail, Result};
use half::bf16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fp8Variant {
    E4M3,
    E5M2,
}

pub const E4M3_MAX: f32 = 448.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum Fp8ScaleMode {
    PerTensor,
    #[default]
    PerOuterRow,
}

fn e4m3_scale_for(amax: f32) -> f32 {
    if amax == 0.0 || !amax.is_finite() {
        1.0
    } else {
        amax / E4M3_MAX
    }
}

fn e4m3_bytes_with_scale(values: &[bf16], scale: f32, out: &mut Vec<u8>) {
    use float8::F8E4M3;
    let inv = if scale == 0.0 || !scale.is_finite() {
        1.0
    } else {
        1.0 / scale
    };
    out.extend(
        values
            .iter()
            .map(|x| F8E4M3::from(x.to_f32() * inv).to_bits()),
    );
}

pub fn quantize_e4m3_per_tensor(values: &[bf16]) -> (Vec<u8>, f32) {
    let amax = values.iter().map(|x| x.to_f32().abs()).fold(0f32, f32::max);
    let scale = e4m3_scale_for(amax);
    let mut bytes = Vec::with_capacity(values.len());
    e4m3_bytes_with_scale(values, scale, &mut bytes);
    (bytes, scale)
}

pub fn quantize_e4m3_per_row(
    values: &[bf16],
    rows: usize,
    cols: usize,
) -> Result<(Vec<u8>, Vec<f32>)> {
    if rows.checked_mul(cols) != Some(values.len()) {
        bail!(
            "quantize_e4m3_per_row: {} values do not form a {rows}x{cols} matrix",
            values.len()
        );
    }
    let mut bytes = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &values[r * cols..(r + 1) * cols];
        let amax = row.iter().map(|x| x.to_f32().abs()).fold(0f32, f32::max);
        let scale = e4m3_scale_for(amax);
        e4m3_bytes_with_scale(row, scale, &mut bytes);
        scales.push(scale);
    }
    Ok((bytes, scales))
}

pub fn quantize_e4m3_with_row_scales(
    values: &[bf16],
    rows: usize,
    cols: usize,
    scales: &[f32],
) -> Result<Vec<u8>> {
    if rows.checked_mul(cols) != Some(values.len()) {
        bail!(
            "quantize_e4m3_with_row_scales: {} values do not form a {rows}x{cols} matrix",
            values.len()
        );
    }
    if scales.len() != rows {
        bail!(
            "quantize_e4m3_with_row_scales: {} scales for {rows} rows",
            scales.len()
        );
    }
    let mut bytes = Vec::with_capacity(values.len());
    for (r, scale) in scales.iter().enumerate() {
        e4m3_bytes_with_scale(&values[r * cols..(r + 1) * cols], *scale, &mut bytes);
    }
    Ok(bytes)
}

pub fn dequantize_e4m3_per_row(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    scales: &[f32],
) -> Result<Vec<f32>> {
    use float8::F8E4M3;
    if rows.checked_mul(cols) != Some(bytes.len()) {
        bail!(
            "dequantize_e4m3_per_row: {} bytes do not form a {rows}x{cols} matrix",
            bytes.len()
        );
    }
    if scales.len() != rows {
        bail!(
            "dequantize_e4m3_per_row: {} scales for {rows} rows",
            scales.len()
        );
    }
    let mut out = Vec::with_capacity(bytes.len());
    for r in 0..rows {
        for c in 0..cols {
            let v = f32::from(F8E4M3::from_bits(bytes[r * cols + c]));
            out.push(v * scales[r]);
        }
    }
    Ok(out)
}

pub fn cpu_e4m3_matmul_row_scaled(
    a: &[u8],
    b_weight: &[u8],
    a_scale_rows: &[f32],
    b_scale_rows: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    use float8::F8E4M3;
    if a.len() != m * k || b_weight.len() != n * k {
        bail!(
            "cpu_e4m3_matmul_row_scaled: a {} != {m}*{k}, or b {} != {n}*{k}",
            a.len(),
            b_weight.len()
        );
    }
    if a_scale_rows.len() != m || b_scale_rows.len() != n {
        bail!(
            "cpu_e4m3_matmul_row_scaled: {} act scales (want {m}), {} weight scales (want {n})",
            a_scale_rows.len(),
            b_scale_rows.len()
        );
    }
    let mut d = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                let av = f32::from(F8E4M3::from_bits(a[i * k + p]));
                let bv = f32::from(F8E4M3::from_bits(b_weight[j * k + p]));
                acc += av * bv;
            }
            d[i * n + j] = acc * a_scale_rows[i] * b_scale_rows[j];
        }
    }
    Ok(d)
}

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(feature = "cuda")]
mod cuda {
    use super::{Fp8ScaleMode, Fp8Variant};
    use anyhow::Result;
    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::cublaslt::result as cublaslt;
    use cudarc::cublaslt::sys;
    use cudarc::driver::sys::CUdeviceptr;
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use float8::F8E4M3;
    use half::bf16;
    use std::ffi::c_void;
    use std::mem;
    use std::sync::Arc;

    pub struct Fp8GemmRunner {
        handle: sys::cublasLtHandle_t,
        workspace: CudaSlice<u8>,
        workspace_bytes: usize,
        stream: Arc<CudaStream>,
    }

    unsafe impl Send for Fp8GemmRunner {}
    unsafe impl Sync for Fp8GemmRunner {}

    impl Fp8GemmRunner {
        pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
            let handle =
                cublaslt::create_handle().map_err(|e| anyhow::anyhow!("cublasLt create: {e:?}"))?;
            let workspace_bytes = 32 * 1024 * 1024;
            let workspace = stream.alloc_zeros::<u8>(workspace_bytes)?;
            Ok(Self {
                handle,
                workspace,
                workspace_bytes,
                stream,
            })
        }
    }

    impl Drop for Fp8GemmRunner {
        fn drop(&mut self) {
            unsafe {
                let _ = cublaslt::destroy_handle(self.handle);
            }
        }
    }

    const OUTER_VEC_32F: sys::cublasLtMatmulMatrixScale_t =
        sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;

    impl Fp8GemmRunner {
        pub fn matmul_e4m3_weight_row(
            &mut self,
            a: &CudaSlice<u8>,
            b_weight: &CudaSlice<u8>,
            d: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            a_scale: &CudaSlice<f32>,
            b_scale: &CudaSlice<f32>,
        ) -> Result<()> {
            unsafe {
                self.matmul_e4m3_unchecked(
                    a,
                    b_weight,
                    d,
                    m,
                    n,
                    k,
                    a_scale,
                    b_scale,
                    Fp8ScaleMode::PerTensor,
                )
            }
        }

        pub fn matmul_e4m3_row_scaled(
            &mut self,
            a: &CudaSlice<u8>,
            b_weight: &CudaSlice<u8>,
            d: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            a_scale_rows: &CudaSlice<f32>,
            b_scale_rows: &CudaSlice<f32>,
        ) -> Result<()> {
            if (a_scale_rows.len() as u64) < m {
                anyhow::bail!(
                    "fp8 row-scaled matmul: {} activation scales for m={m} rows",
                    a_scale_rows.len()
                );
            }
            if (b_scale_rows.len() as u64) < n {
                anyhow::bail!(
                    "fp8 row-scaled matmul: {} weight scales for n={n} output rows",
                    b_scale_rows.len()
                );
            }
            unsafe {
                self.matmul_e4m3_unchecked(
                    a,
                    b_weight,
                    d,
                    m,
                    n,
                    k,
                    a_scale_rows,
                    b_scale_rows,
                    Fp8ScaleMode::PerOuterRow,
                )
            }
        }

        #[allow(clippy::too_many_arguments)]
        unsafe fn matmul_e4m3_unchecked(
            &mut self,
            a: &CudaSlice<u8>,
            b_weight: &CudaSlice<u8>,
            d: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            a_scale: &CudaSlice<f32>,
            b_scale: &CudaSlice<f32>,
            scale_mode: Fp8ScaleMode,
        ) -> Result<()> {
            let handle = self.handle;
            let stream = self.stream.cu_stream();

            let dtype_fp8 = sys::cudaDataType_t::CUDA_R_8F_E4M3;
            let dtype_bf16 = sys::cudaDataType_t::CUDA_R_16BF;
            let compute = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F;
            let scale_type = sys::cudaDataType_t::CUDA_R_32F;

            let a_layout = cublaslt::create_matrix_layout(dtype_fp8, k, n, k as i64)?;
            let b_layout = cublaslt::create_matrix_layout(dtype_fp8, k, m, k as i64)?;
            let d_layout = cublaslt::create_matrix_layout(dtype_bf16, n, m, n as i64)?;

            let desc = cublaslt::create_matmul_desc(compute, scale_type)?;

            let transa = cublasOperation_t::CUBLAS_OP_T;
            let transb = cublasOperation_t::CUBLAS_OP_N;
            cublaslt::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &transa as *const _ as *const c_void,
                mem::size_of::<cublasOperation_t>(),
            )?;
            cublaslt::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &transb as *const _ as *const c_void,
                mem::size_of::<cublasOperation_t>(),
            )?;

            let (weight_scale_ptr, _g1) = b_scale.device_ptr(&self.stream);
            let (act_scale_ptr, _g2) = a_scale.device_ptr(&self.stream);
            if scale_mode == Fp8ScaleMode::PerOuterRow {
                let mode = OUTER_VEC_32F;
                cublaslt::set_matmul_desc_attribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                    &mode as *const _ as *const c_void,
                    mem::size_of::<sys::cublasLtMatmulMatrixScale_t>(),
                )?;
                cublaslt::set_matmul_desc_attribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                    &mode as *const _ as *const c_void,
                    mem::size_of::<sys::cublasLtMatmulMatrixScale_t>(),
                )?;
            }
            cublaslt::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &weight_scale_ptr as *const _ as *const c_void,
                mem::size_of::<CUdeviceptr>(),
            )?;
            cublaslt::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &act_scale_ptr as *const _ as *const c_void,
                mem::size_of::<CUdeviceptr>(),
            )?;

            let pref = cublaslt::create_matmul_pref()?;
            let ws_bytes = self.workspace_bytes;
            cublaslt::set_matmul_pref_attribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_bytes as *const _ as *const c_void,
                mem::size_of::<usize>(),
            )?;

            let heur = cublaslt::get_matmul_algo_heuristic(
                handle, desc, a_layout, b_layout, d_layout, d_layout, pref,
            )
            .map_err(|e| {
                if scale_mode == Fp8ScaleMode::PerOuterRow {
                    anyhow::anyhow!(
                        "cublasLt found no fp8 algo for {m}x{n}x{k} with per-row \
                         (OUTER_VEC_32F) scales: {e:?}. Vector fp8 scaling needs \
                         Blackwell-class cuBLASLt support; on an arch that only offers \
                         scalar fp8 scales set NV_FP8_SCALE_MODE=tensor to fall back to \
                         the coarser per-tensor granularity explicitly."
                    )
                } else {
                    anyhow::anyhow!("cublasLt fp8 heuristic for {m}x{n}x{k}: {e:?}")
                }
            })?;

            let alpha = 1.0f32;
            let beta = 0.0f32;
            let (a_ptr, _ga) = a.device_ptr(&self.stream);
            let (b_ptr, _gb) = b_weight.device_ptr(&self.stream);
            let (d_ptr, _gd) = d.device_ptr_mut(&self.stream);
            let (ws_ptr, _gw) = self.workspace.device_ptr_mut(&self.stream);

            let rc = cublaslt::matmul(
                handle,
                desc,
                &alpha as *const _ as *const c_void,
                &beta as *const _ as *const c_void,
                b_ptr as *const c_void,
                a_layout,
                a_ptr as *const c_void,
                b_layout,
                d_ptr as *const c_void,
                d_layout,
                d_ptr as *mut c_void,
                d_layout,
                &heur.algo as *const _,
                ws_ptr as *mut c_void,
                ws_bytes,
                stream as sys::cudaStream_t,
            );

            cublaslt::destroy_matmul_pref(pref)?;
            cublaslt::destroy_matmul_desc(desc)?;
            cublaslt::destroy_matrix_layout(a_layout)?;
            cublaslt::destroy_matrix_layout(b_layout)?;
            cublaslt::destroy_matrix_layout(d_layout)?;

            rc.map_err(|e| anyhow::anyhow!("cublasLtMatmul: {e:?}"))?;
            Ok(())
        }
    }

    impl Fp8GemmRunner {
        pub fn probe_per_row_scale_support(&mut self) -> Result<()> {
            let (m, n, k): (u64, u64, u64) = (16, 1024, 1024);

            let dtype_fp8 = sys::cudaDataType_t::CUDA_R_8F_E4M3;
            let dtype_bf16 = sys::cudaDataType_t::CUDA_R_16BF;
            let compute = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F;
            let scale_type = sys::cudaDataType_t::CUDA_R_32F;

            let a_layout = cublaslt::create_matrix_layout(dtype_fp8, k, n, k as i64)?;
            let b_layout = cublaslt::create_matrix_layout(dtype_fp8, k, m, k as i64)?;
            let d_layout = cublaslt::create_matrix_layout(dtype_bf16, n, m, n as i64)?;
            let desc = cublaslt::create_matmul_desc(compute, scale_type)?;
            let pref = cublaslt::create_matmul_pref()?;

            let heur = unsafe {
                let transa = cublasOperation_t::CUBLAS_OP_T;
                let transb = cublasOperation_t::CUBLAS_OP_N;
                cublaslt::set_matmul_desc_attribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    &transa as *const _ as *const c_void,
                    mem::size_of::<cublasOperation_t>(),
                )?;
                cublaslt::set_matmul_desc_attribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    &transb as *const _ as *const c_void,
                    mem::size_of::<cublasOperation_t>(),
                )?;
                let mode = OUTER_VEC_32F;
                cublaslt::set_matmul_desc_attribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                    &mode as *const _ as *const c_void,
                    mem::size_of::<sys::cublasLtMatmulMatrixScale_t>(),
                )?;
                cublaslt::set_matmul_desc_attribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                    &mode as *const _ as *const c_void,
                    mem::size_of::<sys::cublasLtMatmulMatrixScale_t>(),
                )?;
                let (ws_ptr, _g) = self.workspace.device_ptr(&self.stream);
                cublaslt::set_matmul_desc_attribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    &ws_ptr as *const _ as *const c_void,
                    mem::size_of::<CUdeviceptr>(),
                )?;
                cublaslt::set_matmul_desc_attribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    &ws_ptr as *const _ as *const c_void,
                    mem::size_of::<CUdeviceptr>(),
                )?;
                let ws_bytes = self.workspace_bytes;
                cublaslt::set_matmul_pref_attribute(
                    pref,
                    sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &ws_bytes as *const _ as *const c_void,
                    mem::size_of::<usize>(),
                )?;
                let heur = cublaslt::get_matmul_algo_heuristic(
                    self.handle,
                    desc,
                    a_layout,
                    b_layout,
                    d_layout,
                    d_layout,
                    pref,
                );
                cublaslt::destroy_matmul_pref(pref)?;
                cublaslt::destroy_matmul_desc(desc)?;
                cublaslt::destroy_matrix_layout(a_layout)?;
                cublaslt::destroy_matrix_layout(b_layout)?;
                cublaslt::destroy_matrix_layout(d_layout)?;
                heur
            };

            heur.map(|_| ()).map_err(|e| {
                anyhow::anyhow!(
                    "cublasLt served no fp8 algo for the per-row (OUTER_VEC_32F) probe shape \
                     {m}x{n}x{k}: {e:?}"
                )
            })
        }
    }

    pub fn supports_fp8(major: i32) -> bool {
        major >= 9
    }

    pub fn cpu_e4m3_matmul_weight_row(
        a: &[F8E4M3],
        b_weight: &[F8E4M3],
        a_scale: f32,
        b_scale: f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<bf16> {
        let mut d = vec![bf16::from_f32(0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    let av = f32::from(a[i * k + p]);
                    let bv = f32::from(b_weight[j * k + p]);
                    acc += av * bv;
                }
                d[i * n + j] = bf16::from_f32(acc * a_scale * b_scale);
            }
        }
        d
    }

    pub fn variant_dtype(v: Fp8Variant) -> sys::cudaDataType_t {
        match v {
            Fp8Variant::E4M3 => sys::cudaDataType_t::CUDA_R_8F_E4M3,
            Fp8Variant::E5M2 => sys::cudaDataType_t::CUDA_R_8F_E5M2,
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub fn supports_fp8(_major: i32) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
        let mut worst = 0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            let denom = w.abs().max(1e-12);
            worst = worst.max((g - w).abs() / denom);
        }
        worst
    }

    #[test]
    fn per_row_scales_survive_row_magnitude_spread_that_per_tensor_destroys() {
        let (rows, cols) = (8usize, 64usize);
        let mut vals = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let mag = 2f32.powi(-(3 * r as i32));
            for c in 0..cols {
                vals.push(bf16::from_f32(mag * ((c as f32) * 0.37).sin()));
            }
        }
        let reference: Vec<f32> = vals.iter().map(|v| v.to_f32()).collect();

        let (row_bytes, row_scales) = quantize_e4m3_per_row(&vals, rows, cols).unwrap();
        assert_eq!(row_scales.len(), rows);
        let row_back = dequantize_e4m3_per_row(&row_bytes, rows, cols, &row_scales).unwrap();

        let (tensor_bytes, tensor_scale) = quantize_e4m3_per_tensor(&vals);
        let tensor_back =
            dequantize_e4m3_per_row(&tensor_bytes, rows, cols, &vec![tensor_scale; rows]).unwrap();

        let last = (rows - 1) * cols;
        let row_err = max_rel_err(&row_back[last..], &reference[last..]);
        let tensor_err = max_rel_err(&tensor_back[last..], &reference[last..]);
        assert!(
            row_err < 0.10,
            "per-row max rel err {row_err} should stay at e4m3 resolution"
        );
        assert!(
            tensor_err > 0.5,
            "per-tensor max rel err {tensor_err} should be catastrophic on the quiet row; \
             if this ever drops the spread in the fixture stopped exercising the bug"
        );
        assert!(
            tensor_err > row_err * 5.0,
            "per-tensor {tensor_err} vs per-row {row_err}: the whole point of the fix"
        );
    }

    #[test]
    fn per_row_scales_are_positive_and_never_zero_on_an_all_zero_row() {
        let (rows, cols) = (3usize, 16usize);
        let mut vals = vec![bf16::from_f32(0.0); rows * cols];
        for c in 0..cols {
            vals[c] = bf16::from_f32(1.0);
        }
        let (_bytes, scales) = quantize_e4m3_per_row(&vals, rows, cols).unwrap();
        assert!(
            scales.iter().all(|s| *s > 0.0 && s.is_finite()),
            "a zero row must get scale 1.0, not 0.0: {scales:?}"
        );
        assert_eq!(scales[1], 1.0);
        assert_eq!(scales[2], 1.0);
    }

    #[test]
    fn per_row_quantization_saturates_the_row_amax_to_e4m3_max() {
        let (rows, cols) = (2usize, 32usize);
        let mut vals = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                vals.push(bf16::from_f32(if c == 0 {
                    if r == 0 {
                        -3.0
                    } else {
                        1e-4
                    }
                } else {
                    0.0
                }));
            }
        }
        let (_bytes, scales) = quantize_e4m3_per_row(&vals, rows, cols).unwrap();
        assert!((scales[0] - 3.0 / E4M3_MAX).abs() < 1e-9, "{scales:?}");
        assert!(
            scales[1] < scales[0],
            "quiet row must get its own smaller scale"
        );
    }

    #[test]
    fn checkpoint_supplied_row_scales_are_honoured_verbatim() {
        let (rows, cols) = (2usize, 8usize);
        let vals: Vec<bf16> = (0..rows * cols)
            .map(|i| bf16::from_f32(((i as f32) * 0.21 + 0.4).sin()))
            .collect();
        let want: Vec<f32> = vals.iter().map(|v| v.to_f32()).collect();

        let (auto_bytes, auto_scales) = quantize_e4m3_per_row(&vals, rows, cols).unwrap();

        let ckpt = vec![auto_scales[0], auto_scales[1] * 2.0];
        let bytes = quantize_e4m3_with_row_scales(&vals, rows, cols, &ckpt).unwrap();
        assert_eq!(
            &bytes[..cols],
            &auto_bytes[..cols],
            "row 0 scale is unchanged"
        );
        assert_ne!(
            &bytes[cols..],
            &auto_bytes[cols..],
            "row 1 must be encoded against the checkpoint scale, not a recomputed one"
        );

        let back = dequantize_e4m3_per_row(&bytes, rows, cols, &ckpt).unwrap();
        let err = max_rel_err(&back, &want);
        assert!(err < 0.10, "checkpoint-scaled roundtrip max rel err {err}");

        assert!(quantize_e4m3_with_row_scales(&vals, rows, cols, &ckpt[..1]).is_err());
        assert!(quantize_e4m3_per_row(&vals, rows, cols + 1).is_err());
    }

    #[test]
    fn row_scaled_cpu_matmul_applies_act_scale_by_row_and_weight_scale_by_column() {
        let (m, n, k) = (2usize, 3usize, 4usize);
        let a: Vec<u8> = (0..m * k)
            .map(|_| float8::F8E4M3::from(1.0f32).to_bits())
            .collect();
        let b: Vec<u8> = (0..n * k)
            .map(|_| float8::F8E4M3::from(1.0f32).to_bits())
            .collect();
        let a_scales = vec![2.0f32, 5.0f32];
        let b_scales = vec![1.0f32, 10.0f32, 100.0f32];
        let d = cpu_e4m3_matmul_row_scaled(&a, &b, &a_scales, &b_scales, m, n, k).unwrap();
        for i in 0..m {
            for j in 0..n {
                let want = (k as f32) * a_scales[i] * b_scales[j];
                assert_eq!(d[i * n + j], want, "d[{i}][{j}]");
            }
        }
    }

    #[test]
    fn default_scale_mode_is_per_row() {
        assert_eq!(Fp8ScaleMode::default(), Fp8ScaleMode::PerOuterRow);
    }

    const CU_PATH: &str = "../nv-kernels/cuda/quantize_nvfp4_bf16.cu";

    const UE4M3_MIN_NORMAL_DEV: f32 = 0.015625;
    const UE4M3_SUBNORMAL_STEP_DEV: f32 = 0.001953125;

    fn encode_ue4m3_dev(scale: f32) -> u8 {
        if !scale.is_finite() || scale <= 0.0 {
            return 0;
        }
        let clamped = scale.min(448.0);
        if clamped < UE4M3_MIN_NORMAL_DEV {
            let sub = (clamped / UE4M3_SUBNORMAL_STEP_DEV).round() as i32;
            if sub <= 0 {
                return 0;
            }
            if sub <= 7 {
                return sub as u8;
            }
            return 0x08;
        }
        let mut exp_v = frexp_exp(clamped) - 1;
        let mant_f = clamped * 2f32.powi(-exp_v) - 1.0;
        let mut mant = (mant_f * 8.0).round() as i32;
        if mant < 0 {
            mant = 0;
        }
        if mant > 7 {
            mant = 0;
            exp_v += 1;
        }
        let biased = (exp_v + 7).clamp(1, 15);
        let byte = ((biased as u8) << 3) | (mant as u8 & 0x07);
        if byte == 0x7F {
            0x7E
        } else {
            byte
        }
    }

    fn decode_ue4m3_dev(b: u8) -> f32 {
        let biased = ((b >> 3) & 0x0F) as i32;
        let mant = (b & 0x07) as f32;
        if biased == 0 {
            return mant * UE4M3_SUBNORMAL_STEP_DEV;
        }
        (1.0 + mant / 8.0) * 2f32.powi(biased - 7)
    }

    fn frexp_exp(x: f32) -> i32 {
        if x == 0.0 || !x.is_finite() {
            return 0;
        }
        x.abs().log2().floor() as i32 + 1
    }

    const UE4M3_NAN_BYTE: u8 = 0x7F;
    const UE4M3_MAX_FINITE_BYTE: u8 = 0x7E;
    const UE4M3_NAN_SOFTWARE_READING: f32 = 480.0;
    const UE4M3_DISTINCT_SCALES: usize = 128;

    fn e4m3fn_magnitude(byte: u8) -> Option<f64> {
        if byte & 0x7F == UE4M3_NAN_BYTE {
            return None;
        }
        let e = ((byte >> 3) & 0x0F) as i32;
        let m = (byte & 0x07) as f64;
        Some(if e == 0 {
            m * (-9f64).exp2()
        } else {
            (1.0 + m / 8.0) * ((e - 7) as f64).exp2()
        })
    }

    fn e4m3fn_nearest_code(target: f64) -> u8 {
        let mut best = 0u8;
        let mut best_d = f64::INFINITY;
        for b in 0u8..=UE4M3_MAX_FINITE_BYTE {
            let Some(v) = e4m3fn_magnitude(b) else {
                continue;
            };
            let d = (v - target).abs();
            if d <= best_d {
                best_d = d;
                best = b;
            }
        }
        best
    }

    #[test]
    fn device_decode_ue4m3_is_the_e4m3fn_value_set_on_every_byte() {
        let mut distinct = std::collections::BTreeSet::new();
        for b in 0u16..=255 {
            let b = b as u8;
            let got = decode_ue4m3_dev(b);
            match e4m3fn_magnitude(b) {
                Some(want) => assert_eq!(
                    got as f64, want,
                    "decode_ue4m3_dev({b:#04x}) = {got:e}; e4m3fn defines {want:e}"
                ),
                None => assert_eq!(
                    got, UE4M3_NAN_SOFTWARE_READING,
                    "{b:#04x} is an e4m3fn NaN pattern; ue4m3 reads it as the finite \
                     {UE4M3_NAN_SOFTWARE_READING}, got {got}"
                ),
            }
            distinct.insert(got.to_bits());
            assert_eq!(
                got.to_bits(),
                crate::nvfp4::decode_ue4m3(b).to_bits(),
                "byte {b:#04x}: device decode {got} != nv_quant::nvfp4::decode_ue4m3"
            );
        }
        assert_eq!(
            distinct.len(),
            UE4M3_DISTINCT_SCALES,
            "the 256 byte patterns must decode to {UE4M3_DISTINCT_SCALES} distinct scales; an \
             oracle that collapses codes makes agreement prove nothing"
        );

        assert_eq!(
            UE4M3_SUBNORMAL_STEP_DEV as f64,
            (-9f64).exp2(),
            "biased exponent 0 is the e4m3fn subnormal encoding, step 2^-9"
        );
        assert_eq!(UE4M3_MIN_NORMAL_DEV as f64, (-6f64).exp2());
        assert_eq!(decode_ue4m3_dev(0x00).to_bits(), 0f32.to_bits());
        assert_eq!(decode_ue4m3_dev(0x02) as f64, 2.0 * (-9f64).exp2());
        assert_ne!(
            decode_ue4m3_dev(0x02) as f64,
            (1.0 + 2.0 / 8.0) * (-7f64).exp2()
        );
    }

    #[test]
    fn device_encode_ue4m3_picks_the_nearest_e4m3fn_code() {
        let mut probes: Vec<f32> = vec![0.0, -1.0, f32::NAN, f32::INFINITY, 448.0, 500.0];
        for i in -14i32..=9 {
            let base = 2f32.powi(i);
            probes.extend([base, base * 1.03, base * 1.5, base * 1.99]);
        }
        for m in 0..=8 {
            let step = 2f32.powi(-9);
            probes.extend([
                m as f32 * step,
                (m as f32 - 0.49) * step,
                (m as f32 + 0.49) * step,
            ]);
        }
        for p in probes {
            let got = encode_ue4m3_dev(p);
            let want = if p.is_finite() && p > 0.0 {
                e4m3fn_nearest_code(p.min(448.0) as f64)
            } else {
                0
            };
            assert_eq!(
                got, want,
                "encode_ue4m3_dev({p:e}) = {got:#04x} ({:e}); the nearest e4m3fn code is \
                 {want:#04x} ({:e})",
                decode_ue4m3_dev(got),
                decode_ue4m3_dev(want)
            );
            assert_ne!(
                got, UE4M3_NAN_BYTE,
                "encode_ue4m3_dev({p:e}) produced the e4m3fn NaN byte"
            );
            assert_eq!(
                got,
                crate::nvfp4::encode_ue4m3(p),
                "encode({p:e}): device {got:#04x} != nv_quant::nvfp4::encode_ue4m3"
            );
        }
    }

    #[test]
    fn device_ue4m3_roundtrip_is_within_a_step_in_the_subnormal_range() {
        let mut s = UE4M3_SUBNORMAL_STEP_DEV * 0.51;
        while s < UE4M3_MIN_NORMAL_DEV {
            let back = decode_ue4m3_dev(encode_ue4m3_dev(s));
            assert!(
                (back - s).abs() <= UE4M3_SUBNORMAL_STEP_DEV * 0.5 + 1e-9,
                "scale {s} round-tripped to {back}"
            );
            s += UE4M3_SUBNORMAL_STEP_DEV * 0.17;
        }
    }

    #[test]
    fn cuda_source_still_carries_the_subnormal_branch() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(CU_PATH);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let after = src
            .split("__device__ __forceinline__ float decode_ue4m3_dev")
            .nth(1)
            .expect("decode_ue4m3_dev not found in the .cu");
        let body = &after[..after.find("__global__").unwrap_or(after.len())];
        assert!(
            body.contains("NV_UE4M3_SUBNORMAL_STEP"),
            "decode_ue4m3_dev lost its exp==0 subnormal branch: underflow-clamped scale \
             bytes would decode up to 8x high and disagree with the ue4m3 decode cuBLASLt \
             applies to the same byte"
        );
        assert!(
            src.contains("#define NV_UE4M3_SUBNORMAL_STEP  0.001953125f"),
            "the subnormal step must stay 2^-9 to match nv_quant::nvfp4::decode_ue4m3"
        );
    }
}
