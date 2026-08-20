pub const BLOCK_SIZE: usize = 16;
pub const MIN_TILE: usize = 128;

const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

pub fn encode_e2m1(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0b1000 } else { 0 };
    let abs = x.abs();
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, v) in E2M1_VALUES.iter().enumerate() {
        let err = (abs - v).abs();
        if err < best_err {
            best_err = err;
            best = i as u8;
        }
    }
    sign | best
}

pub fn decode_e2m1(byte: u8) -> f32 {
    let mag = E2M1_VALUES[(byte & 0b0111) as usize];
    if byte & 0b1000 != 0 {
        -mag
    } else {
        mag
    }
}

pub fn pack_e2m1_pair(lo: u8, hi: u8) -> u8 {
    (hi << 4) | (lo & 0x0F)
}

pub fn unpack_e2m1_pair(byte: u8) -> (u8, u8) {
    (byte & 0x0F, (byte >> 4) & 0x0F)
}

const UE4M3_MIN_NORMAL: f32 = 0.015625;

const UE4M3_SUBNORMAL_STEP: f32 = 0.001953125;

pub fn encode_ue4m3(scale: f32) -> u8 {
    if !(scale.is_finite()) || scale <= 0.0 {
        return 0;
    }
    let clamped = scale.min(448.0);
    if clamped < UE4M3_MIN_NORMAL {
        let m = (clamped / UE4M3_SUBNORMAL_STEP).round() as i32;
        if m <= 0 {
            return 0;
        }
        if m <= 7 {
            return m as u8;
        }

        return 0x08;
    }
    let log = clamped.log2();
    let mut exp = log.floor() as i32;
    let mantissa_f = (clamped / (2f32).powi(exp)) - 1.0;
    let mut mantissa = (mantissa_f * 8.0).round() as i32;

    if mantissa >= 8 {
        mantissa = 0;
        exp += 1;
    }
    let mantissa = mantissa.clamp(0, 7);
    let biased_exp = (exp + 7).clamp(1, 15);
    let byte = ((biased_exp as u8) << 3) | (mantissa as u8 & 0x07);
    if byte == 0x7F {
        0x7E
    } else {
        byte
    }
}

pub fn decode_ue4m3(byte: u8) -> f32 {
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mantissa = (byte & 0x07) as f32;
    if exp == 0 {
        return mantissa * UE4M3_SUBNORMAL_STEP;
    }
    let unbiased = exp - 7;
    (1.0 + mantissa / 8.0) * (2f32).powi(unbiased)
}

pub fn quantize_block(values: &[f32]) -> (Vec<u8>, u8) {
    quantize_block_with_global(values, 1.0)
}

pub fn quantize_block_with_global(values: &[f32], stored_global: f32) -> (Vec<u8>, u8) {
    assert_eq!(values.len(), BLOCK_SIZE);
    let amax = values.iter().fold(0f32, |a, b| a.max(b.abs()));
    let stored = if stored_global == 0.0 || !stored_global.is_finite() {
        1.0
    } else {
        stored_global
    };
    let local_scale = if amax == 0.0 { 1.0 } else { amax / 6.0 };
    let stored_scale = stored * local_scale;
    let scale_byte = encode_ue4m3(stored_scale);
    let scale_decoded = decode_ue4m3(scale_byte);
    let inv = if scale_decoded == 0.0 {
        1.0
    } else {
        stored / scale_decoded
    };
    let mut nibbles = Vec::with_capacity(BLOCK_SIZE);
    for v in values {
        let scaled = (v * inv).clamp(-6.0, 6.0);
        nibbles.push(encode_e2m1(scaled));
    }
    let mut packed = Vec::with_capacity(BLOCK_SIZE / 2);
    for chunk in nibbles.chunks(2) {
        packed.push(pack_e2m1_pair(chunk[0], chunk[1]));
    }
    (packed, scale_byte)
}

pub fn dequantize_block(packed: &[u8], scale_byte: u8) -> Vec<f32> {
    let scale = decode_ue4m3(scale_byte);
    let mut out = Vec::with_capacity(BLOCK_SIZE);
    for byte in packed {
        let (lo, hi) = unpack_e2m1_pair(*byte);
        out.push(decode_e2m1(lo) * scale);
        out.push(decode_e2m1(hi) * scale);
    }
    out
}

pub struct Nvfp4Tensor {
    pub data: Vec<u8>,
    pub scales: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}

impl Nvfp4Tensor {
    pub fn quantize_rows(rows: &[Vec<f32>]) -> Self {
        Self::quantize_rows_with_global(rows, 1.0)
    }

    pub fn quantize_rows_with_global(rows: &[Vec<f32>], stored_global: f32) -> Self {
        let rows_n = rows.len();
        let cols = rows[0].len();
        assert!(
            cols.is_multiple_of(BLOCK_SIZE),
            "cols must be a multiple of {BLOCK_SIZE}"
        );
        let blocks_per_row = cols / BLOCK_SIZE;
        let mut data = Vec::with_capacity(rows_n * cols / 2);
        let mut scales = Vec::with_capacity(rows_n * blocks_per_row);
        for row in rows {
            for block in row.chunks(BLOCK_SIZE) {
                let (packed, scale) = quantize_block_with_global(block, stored_global);
                data.extend_from_slice(&packed);
                scales.push(scale);
            }
        }
        Self {
            data,
            scales,
            rows: rows_n,
            cols,
        }
    }

    pub fn scales_column_major(&self) -> Vec<u8> {
        let blocks_per_row = self.cols / BLOCK_SIZE;
        let mut out = vec![0u8; self.rows * blocks_per_row];
        for r in 0..self.rows {
            for b in 0..blocks_per_row {
                let src = r * blocks_per_row + b;
                let dst = b * self.rows + r;
                out[dst] = self.scales[src];
            }
        }
        out
    }

    pub fn scales_swizzled(&self) -> Vec<u8> {
        let blocks_per_row = self.cols / BLOCK_SIZE;
        swizzle_scales(&self.scales, self.rows, blocks_per_row)
    }

    pub fn dequantize_scaled(&self, weight_mult: f32) -> Vec<Vec<f32>> {
        self.dequantize()
            .into_iter()
            .map(|row| row.into_iter().map(|v| v * weight_mult).collect())
            .collect()
    }

    pub fn dequantize(&self) -> Vec<Vec<f32>> {
        let blocks_per_row = self.cols / BLOCK_SIZE;
        let bytes_per_row = self.cols / 2;
        let mut out = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            let row_bytes = &self.data[r * bytes_per_row..(r + 1) * bytes_per_row];
            let row_scales = &self.scales[r * blocks_per_row..(r + 1) * blocks_per_row];
            let mut row = Vec::with_capacity(self.cols);
            let block_bytes = BLOCK_SIZE / 2;
            for (b, scale_byte) in row_scales.iter().enumerate() {
                let packed = &row_bytes[b * block_bytes..(b + 1) * block_bytes];
                row.extend_from_slice(&dequantize_block(packed, *scale_byte));
            }
            out.push(row);
        }
        out
    }
}

pub fn skinny_lt_from(raw: Option<&str>) -> bool {
    raw != Some("0")
}

pub fn k256_from(raw: Option<&str>) -> bool {
    raw != Some("0")
}

pub fn streamk_k256_from(raw: Option<&str>) -> bool {
    raw.is_some_and(|v| !v.is_empty() && v != "0")
}

pub fn streamk_down_splits_from(raw: Option<&str>) -> i32 {
    match raw.and_then(|v| v.trim().parse::<i32>().ok()) {
        Some(s) if s >= 2 => s,
        _ => 0,
    }
}

pub fn swizzle_scales(linear: &[u8], rows: usize, k_blocks: usize) -> Vec<u8> {
    let m_tiles = rows.div_ceil(128);
    let k_tiles = k_blocks.div_ceil(4);
    let mut out = vec![0u8; m_tiles * 128 * k_tiles * 4];
    for m in 0..rows {
        for kb in 0..k_blocks {
            let m_tile = m / 128;
            let d2 = (m / 32) % 4;
            let d3 = m % 32;
            let k_tile = kb / 4;
            let d5 = kb % 4;
            let dst = ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5;
            out[dst] = linear[m * k_blocks + kb];
        }
    }
    out
}

pub fn unswizzle_scales(sw: &[u8], rows: usize, k_blocks: usize) -> Vec<u8> {
    let k_tiles = k_blocks.div_ceil(4);
    let mut out = vec![0u8; rows * k_blocks];
    for m in 0..rows {
        for kb in 0..k_blocks {
            let m_tile = m / 128;
            let d2 = (m / 32) % 4;
            let d3 = m % 32;
            let k_tile = kb / 4;
            let d5 = kb % 4;
            let src = ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5;
            out[m * k_blocks + kb] = sw[src];
        }
    }
    out
}

pub fn dequantize_packed_linear(
    packed: &[u8],
    scales_linear: &[u8],
    rows: usize,
    cols: usize,
    weight_mult: f32,
) -> Vec<f32> {
    assert!(
        cols.is_multiple_of(BLOCK_SIZE),
        "cols must be a multiple of {BLOCK_SIZE}"
    );
    let blocks_per_row = cols / BLOCK_SIZE;
    let bytes_per_row = cols / 2;
    let block_bytes = BLOCK_SIZE / 2;
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let scale = decode_ue4m3(scales_linear[r * blocks_per_row + b]) * weight_mult;
            let base = r * bytes_per_row + b * block_bytes;
            for i in 0..block_bytes {
                let (lo, hi) = unpack_e2m1_pair(packed[base + i]);
                out[r * cols + b * BLOCK_SIZE + i * 2] = decode_e2m1(lo) * scale;
                out[r * cols + b * BLOCK_SIZE + i * 2 + 1] = decode_e2m1(hi) * scale;
            }
        }
    }
    out
}

pub fn dequantize_packed_swizzled(
    packed: &[u8],
    scales_swizzled: &[u8],
    rows: usize,
    cols: usize,
    weight_mult: f32,
) -> Vec<f32> {
    let blocks_per_row = cols / BLOCK_SIZE;
    let linear = unswizzle_scales(scales_swizzled, rows, blocks_per_row);
    dequantize_packed_linear(packed, &linear, rows, cols, weight_mult)
}

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(feature = "cuda")]
mod cuda {
    use super::{Nvfp4Tensor, BLOCK_SIZE, MIN_TILE};
    use anyhow::Result;
    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::cublaslt::result as cublaslt;
    use cudarc::cublaslt::sys;
    use cudarc::driver::sys::CUdeviceptr;
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use half::bf16;
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::mem;
    use std::sync::Arc;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Nvfp4Backend {
        CublasLt,

        Cutlass,

        Auto,
    }

    impl Nvfp4Backend {
        fn from_env() -> Self {
            match std::env::var("SPEACHES_NVFP4_BACKEND")
                .ok()
                .as_deref()
                .map(|s| s.trim().to_ascii_lowercase())
                .as_deref()
            {
                Some("cublaslt") | Some("cublas") => Self::CublasLt,
                Some("cutlass") => Self::Cutlass,
                Some("auto") | None => Self::Auto,
                Some(other) => {
                    eprintln!(
                        "SPEACHES_NVFP4_BACKEND={other:?} unrecognized; falling back to auto"
                    );
                    Self::Auto
                }
            }
        }
    }

    const CUTLASS_SM120_TILE_M: u64 = 128;
    const CUTLASS_SM120_TILE_N: u64 = 128;
    const CUTLASS_SM120_TILE_K: u64 = 128;

    pub struct Nvfp4GemmRunner {
        handle: sys::cublasLtHandle_t,
        stream: Arc<CudaStream>,
        backend: Nvfp4Backend,

        streamk_failed: bool,

        desc_cache: HashMap<DescKey, CachedDesc>,
    }

    pub const WORKSPACE_BYTES: usize = 64 * 1024 * 1024;

    type WorkspaceMap = HashMap<usize, Arc<std::sync::Mutex<CudaSlice<u8>>>>;
    static WORKSPACES: std::sync::OnceLock<std::sync::Mutex<WorkspaceMap>> =
        std::sync::OnceLock::new();

    fn workspace_map() -> &'static std::sync::Mutex<WorkspaceMap> {
        WORKSPACES.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
    }

    fn stream_is_capturing(stream: &Arc<CudaStream>) -> bool {
        use cudarc::driver::sys as drv;
        let mut st = drv::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let rc = unsafe { drv::cuStreamIsCapturing(stream.cu_stream(), &mut st) };
        rc == drv::CUresult::CUDA_SUCCESS
            && st != drv::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    }

    pub fn ensure_workspace_for_stream(stream: &Arc<CudaStream>) -> Result<()> {
        workspace_handle(stream).map(|_| ())
    }

    fn workspace_handle(stream: &Arc<CudaStream>) -> Result<Arc<std::sync::Mutex<CudaSlice<u8>>>> {
        let key = crate::stream_cache_key(stream);
        let mut map = workspace_map()
            .lock()
            .map_err(|e| anyhow::anyhow!("nvfp4 workspace map poisoned: {e}"))?;
        if let Some(ws) = map.get(&key) {
            return Ok(ws.clone());
        }
        anyhow::ensure!(
            !stream_is_capturing(stream),
            "nvfp4 workspace for stream {key:#x} must be created before graph capture \
             (call ensure_workspace_for_stream on the capture stream first)"
        );
        let ws = Arc::new(std::sync::Mutex::new(
            stream.alloc_zeros::<u8>(WORKSPACE_BYTES)?,
        ));
        map.insert(key, ws.clone());
        Ok(ws)
    }

    pub fn release_stream_workspace(cu_stream_key: usize) {
        if let Ok(mut map) = workspace_map().lock() {
            map.remove(&cu_stream_key);
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum StreamkMode {
        Never,
        Always,
        Auto,
    }

    fn streamk_mode() -> StreamkMode {
        static MODE: std::sync::OnceLock<StreamkMode> = std::sync::OnceLock::new();
        *MODE.get_or_init(|| match std::env::var("NV_NVFP4_STREAMK").ok().as_deref() {
            Some("0") => StreamkMode::Never,
            Some("1") => StreamkMode::Always,
            _ => StreamkMode::Auto,
        })
    }

    fn skinny_lt_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            super::skinny_lt_from(std::env::var("NV_NVFP4_SKINNY_LT").ok().as_deref())
        })
    }

    fn k256_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| super::k256_from(std::env::var("NV_NVFP4_K256").ok().as_deref()))
    }

    fn streamk_k256_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            super::streamk_k256_from(std::env::var("NV_NVFP4_STREAMK_K256").ok().as_deref())
        })
    }

    fn streamk_down_splits() -> i32 {
        static N: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
        *N.get_or_init(|| {
            super::streamk_down_splits_from(
                std::env::var("NV_NVFP4_STREAMK_DOWN_SPLITS")
                    .ok()
                    .as_deref(),
            )
        })
    }

    fn lt_algo_pins() -> &'static HashMap<(u64, u64, u64), usize> {
        static PINS: std::sync::OnceLock<HashMap<(u64, u64, u64), usize>> =
            std::sync::OnceLock::new();
        PINS.get_or_init(|| crate::algo_pin::pin_map_from_env("NV_NVFP4_LT_ALGO_PIN"))
    }

    fn lt_algo_log() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("NV_NVFP4_LT_ALGO_LOG").map_or(false, |v| v != "0"))
    }

    pub(crate) unsafe fn algo_config_summary(algo: &sys::cublasLtMatmulAlgo_t) -> String {
        use sys::cublasLtMatmulAlgoConfigAttributes_t as cfg;
        let mut vals = [0i32; 6];
        let attrs = [
            cfg::CUBLASLT_ALGO_CONFIG_ID,
            cfg::CUBLASLT_ALGO_CONFIG_TILE_ID,
            cfg::CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
            cfg::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
            cfg::CUBLASLT_ALGO_CONFIG_STAGES_ID,
            cfg::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
        ];
        for (v, a) in vals.iter_mut().zip(attrs) {
            let mut written = 0usize;
            let st = sys::cublasLtMatmulAlgoConfigGetAttribute(
                algo as *const _,
                a,
                v as *mut i32 as *mut c_void,
                mem::size_of::<i32>(),
                &mut written,
            );
            if st != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                *v = -1;
            }
        }
        format!(
            "id={} tile={} splitk={} red={} stages={} cluster={}",
            vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]
        )
    }

    #[derive(Hash, PartialEq, Eq, Copy, Clone)]
    pub(crate) struct DescKey {
        pub m: u64,
        pub n: u64,
        pub k: u64,
        pub a_scale_ptr: CUdeviceptr,
    }

    pub(crate) struct CachedDesc {
        pub desc: sys::cublasLtMatmulDesc_t,
        pub a_layout: sys::cublasLtMatrixLayout_t,
        pub b_layout: sys::cublasLtMatrixLayout_t,
        pub d_layout: sys::cublasLtMatrixLayout_t,
        pub algo: sys::cublasLtMatmulAlgo_t,
    }

    unsafe impl Send for Nvfp4GemmRunner {}
    unsafe impl Sync for Nvfp4GemmRunner {}

    impl Nvfp4GemmRunner {
        pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
            let handle =
                cublaslt::create_handle().map_err(|e| anyhow::anyhow!("cublasLt create: {e:?}"))?;
            ensure_workspace_for_stream(&stream)?;
            Ok(Self {
                handle,
                stream,
                backend: Nvfp4Backend::from_env(),
                streamk_failed: false,
                desc_cache: HashMap::new(),
            })
        }

        pub fn set_stream(&mut self, stream: Arc<CudaStream>) {
            self.stream = stream;
        }

        fn cutlass_supports_shape(m: u64, n: u64, k: u64) -> bool {
            (m <= CUTLASS_SM120_TILE_M || m % CUTLASS_SM120_TILE_M == 0)
                && n % CUTLASS_SM120_TILE_N == 0
                && k % CUTLASS_SM120_TILE_K == 0
        }

        pub fn supports_true_m(&self, m: u64, n: u64, k: u64) -> bool {
            self.backend != Nvfp4Backend::CublasLt && Self::cutlass_supports_shape(m, n, k)
        }

        pub fn matmul_scaled_alpha_dev(
            &mut self,
            a_data: &CudaSlice<u8>,
            a_scales_cm: &CudaSlice<u8>,
            b_data: &CudaSlice<u8>,
            b_scales_cm: &CudaSlice<u8>,
            d: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            alpha_dev: &CudaSlice<f32>,
            alpha_host: f32,
        ) -> Result<()> {
            if k % BLOCK_SIZE as u64 != 0 {
                anyhow::bail!("k={k} must be a multiple of {BLOCK_SIZE}");
            }
            let min = MIN_TILE as u64;
            if m == 0 || n < min || k < min {
                anyhow::bail!(
                    "NVFP4 GEMM requires m >= 1 and n,k >= {MIN_TILE}; got m={m} n={n} k={k}"
                );
            }
            let use_cutlass = match self.backend {
                Nvfp4Backend::CublasLt => false,
                Nvfp4Backend::Cutlass => {
                    if !Self::cutlass_supports_shape(m, n, k) {
                        anyhow::bail!(
                            "SPEACHES_NVFP4_BACKEND=cutlass but shape m={m} n={n} k={k} \
                             is not supported by the SM120 tile (128/128/128, true-m <= 128)"
                        );
                    }
                    true
                }
                Nvfp4Backend::Auto => Self::cutlass_supports_shape(m, n, k),
            };
            if !use_cutlass && m < min {
                anyhow::bail!(
                    "NVFP4 cuBLASLt path requires m >= {MIN_TILE} (got m={m}); \
                     pad the activation or use a CUTLASS-eligible shape"
                );
            }
            let lt_reroute = use_cutlass
                && self.backend == Nvfp4Backend::Auto
                && m >= min
                && skinny_lt_enabled()
                && Self::skinny_regime(m, n);
            if use_cutlass && !lt_reroute {
                unsafe {
                    self.matmul_cutlass_unchecked_alpha_dev(
                        a_data,
                        a_scales_cm,
                        b_data,
                        b_scales_cm,
                        d,
                        m,
                        n,
                        k,
                        alpha_dev,
                    )
                }
            } else {
                unsafe {
                    self.matmul_unchecked(
                        a_data,
                        a_scales_cm,
                        b_data,
                        b_scales_cm,
                        d,
                        m,
                        n,
                        k,
                        alpha_host,
                    )
                }
            }
        }

        pub const LT_NARROW_M_FLOOR_SET_BY_FP4_LT_B_DIM_GRANULARITY_SCALES_STILL_PAD_ROWS_TO_128: u64 = 16;

        pub fn matmul_scaled_lt_narrow_m(
            &mut self,
            a_data: &CudaSlice<u8>,
            a_scales_cm: &CudaSlice<u8>,
            b_data: &CudaSlice<u8>,
            b_scales_cm: &CudaSlice<u8>,
            d: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            alpha_host: f32,
        ) -> Result<()> {
            if k % BLOCK_SIZE as u64 != 0 {
                anyhow::bail!("k={k} must be a multiple of {BLOCK_SIZE}");
            }
            let floor = Self::LT_NARROW_M_FLOOR_SET_BY_FP4_LT_B_DIM_GRANULARITY_SCALES_STILL_PAD_ROWS_TO_128;
            if m < floor || m % floor != 0 || n < MIN_TILE as u64 || k < MIN_TILE as u64 {
                anyhow::bail!(
                    "NVFP4 LT narrow-m route requires m a multiple of {floor} and n,k >= \
                     {MIN_TILE}; got m={m} n={n} k={k}"
                );
            }
            unsafe {
                self.matmul_unchecked(
                    a_data,
                    a_scales_cm,
                    b_data,
                    b_scales_cm,
                    d,
                    m,
                    n,
                    k,
                    alpha_host,
                )
            }
        }

        fn skinny_tiles(m: u64, n: u64) -> bool {
            let m_tiles = m.div_ceil(CUTLASS_SM120_TILE_M);
            let n_tiles = n.div_ceil(CUTLASS_SM120_TILE_N);
            m_tiles * n_tiles <= 192
        }

        fn skinny_regime(m: u64, n: u64) -> bool {
            streamk_mode() == StreamkMode::Auto && Self::skinny_tiles(m, n)
        }

        fn streamk_wanted(&self, m: u64, n: u64) -> bool {
            if self.streamk_failed {
                return false;
            }
            match streamk_mode() {
                StreamkMode::Never => false,
                StreamkMode::Always => true,
                StreamkMode::Auto => Self::skinny_tiles(m, n),
            }
        }

        unsafe fn matmul_cutlass_unchecked_alpha_dev(
            &mut self,
            a_data: &CudaSlice<u8>,
            a_scales: &CudaSlice<u8>,
            b_data: &CudaSlice<u8>,
            b_scales: &CudaSlice<u8>,
            d: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            alpha_dev: &CudaSlice<f32>,
        ) -> Result<()> {
            let ws = workspace_handle(&self.stream)?;
            let mut ws = ws
                .lock()
                .map_err(|e| anyhow::anyhow!("nvfp4 workspace poisoned: {e}"))?;
            if self.streamk_wanted(m, n) {
                match cutlass_launch_impl(
                    &self.stream,
                    &mut ws,
                    WORKSPACE_BYTES,
                    a_data,
                    a_scales,
                    b_data,
                    b_scales,
                    d,
                    m,
                    n,
                    k,
                    alpha_dev,
                    true,
                ) {
                    Ok(()) => return Ok(()),
                    Err(e) => {
                        eprintln!(
                            "[nvfp4] Stream-K launch failed for m={m} n={n} k={k} ({e}); \
                             falling back to the DP scheduler permanently"
                        );
                        self.streamk_failed = true;
                    }
                }
            }
            cutlass_launch_impl(
                &self.stream,
                &mut ws,
                WORKSPACE_BYTES,
                a_data,
                a_scales,
                b_data,
                b_scales,
                d,
                m,
                n,
                k,
                alpha_dev,
                false,
            )
        }
    }

    unsafe fn cutlass_launch_impl(
        stream: &Arc<CudaStream>,
        workspace: &mut CudaSlice<u8>,
        workspace_bytes: usize,
        a_data: &CudaSlice<u8>,
        a_scales: &CudaSlice<u8>,
        b_data: &CudaSlice<u8>,
        b_scales: &CudaSlice<u8>,
        d: &mut CudaSlice<bf16>,
        m: u64,
        n: u64,
        k: u64,
        gsf: &CudaSlice<f32>,
        stream_k: bool,
    ) -> Result<()> {
        let stream_raw = stream.cu_stream() as *mut c_void;
        let (a_ptr, _ga) = a_data.device_ptr(stream);
        let (a_sf_ptr, _gas) = a_scales.device_ptr(stream);
        let (b_ptr, _gb) = b_data.device_ptr(stream);
        let (b_sf_ptr, _gbs) = b_scales.device_ptr(stream);
        let (gsf_ptr, _ggsf) = gsf.device_ptr(stream);
        let (d_ptr, _gd) = d.device_ptr_mut(stream);
        let (ws_ptr, _gws) = workspace.device_ptr_mut(stream);
        let splits = streamk_down_splits();
        let res = if stream_k && splits >= 2 {
            nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16_tiled(
                stream_raw,
                a_ptr as *const c_void,
                a_sf_ptr as *const c_void,
                b_ptr as *const c_void,
                b_sf_ptr as *const c_void,
                gsf_ptr as *const f32,
                d_ptr as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                0,
                splits,
                ws_ptr as *mut c_void,
                workspace_bytes,
            )
        } else if stream_k {
            if k % 256 == 0 && streamk_k256_enabled() {
                nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16_tiled(
                    stream_raw,
                    a_ptr as *const c_void,
                    a_sf_ptr as *const c_void,
                    b_ptr as *const c_void,
                    b_sf_ptr as *const c_void,
                    gsf_ptr as *const f32,
                    d_ptr as *mut c_void,
                    m as i32,
                    n as i32,
                    k as i32,
                    1,
                    1,
                    ws_ptr as *mut c_void,
                    workspace_bytes,
                )
            } else {
                nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16_streamk(
                    stream_raw,
                    a_ptr as *const c_void,
                    a_sf_ptr as *const c_void,
                    b_ptr as *const c_void,
                    b_sf_ptr as *const c_void,
                    gsf_ptr as *const f32,
                    d_ptr as *mut c_void,
                    m as i32,
                    n as i32,
                    k as i32,
                    ws_ptr as *mut c_void,
                    workspace_bytes,
                )
            }
        } else if k % 256 == 0 && k256_enabled() {
            nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16_tiled(
                stream_raw,
                a_ptr as *const c_void,
                a_sf_ptr as *const c_void,
                b_ptr as *const c_void,
                b_sf_ptr as *const c_void,
                gsf_ptr as *const f32,
                d_ptr as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                1,
                0,
                ws_ptr as *mut c_void,
                workspace_bytes,
            )
        } else {
            nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16(
                stream_raw,
                a_ptr as *const c_void,
                a_sf_ptr as *const c_void,
                b_ptr as *const c_void,
                b_sf_ptr as *const c_void,
                gsf_ptr as *const f32,
                d_ptr as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                ws_ptr as *mut c_void,
                workspace_bytes,
            )
        };
        let _needed = res.map_err(|rc| {
            anyhow::anyhow!(
                "cutlass_fp4_gemm_sm120_bf16{} returned rc={rc}",
                if stream_k { "_streamk" } else { "" }
            )
        })?;
        Ok(())
    }

    impl Nvfp4GemmRunner {
        unsafe fn matmul_unchecked(
            &mut self,
            a_data: &CudaSlice<u8>,
            a_scales: &CudaSlice<u8>,
            b_data: &CudaSlice<u8>,
            b_scales: &CudaSlice<u8>,
            d: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            alpha: f32,
        ) -> Result<()> {
            let handle = self.handle;
            let stream_raw = self.stream.cu_stream();
            let stream_arc = self.stream.clone();

            let (a_scale_ptr_val, _g_a_scale) = b_scales.device_ptr(&stream_arc);
            let (b_scale_ptr_val, _g_b_scale) = a_scales.device_ptr(&stream_arc);
            let key = DescKey {
                m,
                n,
                k,
                a_scale_ptr: a_scale_ptr_val,
            };

            if !self.desc_cache.contains_key(&key) {
                let dtype_fp4 = sys::cudaDataType_t::CUDA_R_4F_E2M1;
                let dtype_bf16 = sys::cudaDataType_t::CUDA_R_16BF;
                let compute = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F;
                let scale_type = sys::cudaDataType_t::CUDA_R_32F;

                let a_layout = cublaslt::create_matrix_layout(dtype_fp4, k, n, k as i64)?;
                let b_layout = cublaslt::create_matrix_layout(dtype_fp4, k, m, k as i64)?;
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

                let scale_mode =
                    sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
                cublaslt::set_matmul_desc_attribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                    &scale_mode as *const _ as *const c_void,
                    mem::size_of::<sys::cublasLtMatmulMatrixScale_t>(),
                )?;
                cublaslt::set_matmul_desc_attribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                    &scale_mode as *const _ as *const c_void,
                    mem::size_of::<sys::cublasLtMatmulMatrixScale_t>(),
                )?;

                cublaslt::set_matmul_desc_attribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    &a_scale_ptr_val as *const _ as *const c_void,
                    mem::size_of::<CUdeviceptr>(),
                )?;

                let pref = cublaslt::create_matmul_pref()?;
                let ws_bytes = WORKSPACE_BYTES;
                cublaslt::set_matmul_pref_attribute(
                    pref,
                    sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &ws_bytes as *const _ as *const c_void,
                    mem::size_of::<usize>(),
                )?;

                cublaslt::set_matmul_desc_attribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    &b_scale_ptr_val as *const _ as *const c_void,
                    mem::size_of::<CUdeviceptr>(),
                )?;
                let pin = lt_algo_pins().get(&(m, n, k)).copied();
                let algo = if pin.is_some() || lt_algo_log() {
                    const MAX_CANDS: usize = 24;
                    let mut results: [sys::cublasLtMatmulHeuristicResult_t; MAX_CANDS] =
                        mem::zeroed();
                    let mut returned: i32 = 0;
                    let st = sys::cublasLtMatmulAlgoGetHeuristic(
                        handle,
                        desc,
                        a_layout,
                        b_layout,
                        d_layout,
                        d_layout,
                        pref,
                        MAX_CANDS as i32,
                        results.as_mut_ptr(),
                        &mut returned,
                    );
                    anyhow::ensure!(
                        st == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS && returned > 0,
                        "cublasLtMatmulAlgoGetHeuristic nvfp4: {st:?} returned={returned}"
                    );
                    let cands: Vec<&sys::cublasLtMatmulHeuristicResult_t> = results
                        [..returned as usize]
                        .iter()
                        .filter(|r| r.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS)
                        .collect();
                    anyhow::ensure!(!cands.is_empty(), "no valid nvfp4 heuristic candidates");
                    let idx = match pin {
                        Some(i) if i < cands.len() => i,
                        Some(i) => {
                            eprintln!(
                                "[nvfp4-lt] pin idx {i} out of range for {m}x{n}x{k} \
                                 ({} cands); using 0",
                                cands.len()
                            );
                            0
                        }
                        None => 0,
                    };
                    if lt_algo_log() {
                        for (ci, cand) in cands.iter().enumerate() {
                            eprintln!(
                                "[nvfp4-lt] shape {m}x{n}x{k} cand {ci}{}: {} waves={:.2} ws={}",
                                if ci == idx { " (chosen)" } else { "" },
                                algo_config_summary(&cand.algo),
                                cand.wavesCount,
                                cand.workspaceSize,
                            );
                        }
                    }
                    cands[idx].algo
                } else {
                    let heur = cublaslt::get_matmul_algo_heuristic(
                        handle, desc, a_layout, b_layout, d_layout, d_layout, pref,
                    )?;
                    heur.algo
                };
                cublaslt::destroy_matmul_pref(pref)?;

                self.desc_cache.insert(
                    key,
                    CachedDesc {
                        desc,
                        a_layout,
                        b_layout,
                        d_layout,
                        algo,
                    },
                );
            }

            let cached = self.desc_cache.get(&key).expect("just inserted");

            cublaslt::set_matmul_desc_attribute(
                cached.desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &b_scale_ptr_val as *const _ as *const c_void,
                mem::size_of::<CUdeviceptr>(),
            )?;

            let beta = 0.0f32;
            let ws = workspace_handle(&stream_arc)?;
            let mut ws = ws
                .lock()
                .map_err(|e| anyhow::anyhow!("nvfp4 workspace poisoned: {e}"))?;
            let (a_ptr, _ga) = a_data.device_ptr(&stream_arc);
            let (b_ptr, _gb) = b_data.device_ptr(&stream_arc);
            let (d_ptr, _gd) = d.device_ptr_mut(&stream_arc);
            let (ws_ptr, _gw) = ws.device_ptr_mut(&stream_arc);

            let rc = cublaslt::matmul(
                handle,
                cached.desc,
                &alpha as *const _ as *const c_void,
                &beta as *const _ as *const c_void,
                b_ptr as *const c_void,
                cached.a_layout,
                a_ptr as *const c_void,
                cached.b_layout,
                d_ptr as *const c_void,
                cached.d_layout,
                d_ptr as *mut c_void,
                cached.d_layout,
                &cached.algo as *const _,
                ws_ptr as *mut c_void,
                WORKSPACE_BYTES,
                stream_raw as sys::cudaStream_t,
            );

            rc.map_err(|e| anyhow::anyhow!("cublasLtMatmul nvfp4: {e:?}"))?;
            Ok(())
        }
    }

    impl Drop for Nvfp4GemmRunner {
        fn drop(&mut self) {
            for cached in self.desc_cache.values() {
                unsafe {
                    let _ = cublaslt::destroy_matmul_desc(cached.desc);
                    let _ = cublaslt::destroy_matrix_layout(cached.a_layout);
                    let _ = cublaslt::destroy_matrix_layout(cached.b_layout);
                    let _ = cublaslt::destroy_matrix_layout(cached.d_layout);
                }
            }
            unsafe {
                let _ = cublaslt::destroy_handle(self.handle);
            }
        }
    }

    pub fn cpu_nvfp4_matmul_weight_row(
        a: &Nvfp4Tensor,
        b_weight: &Nvfp4Tensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<bf16> {
        let a_deq = a.dequantize();
        let b_deq = b_weight.dequantize();
        let mut d = vec![bf16::from_f32(0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    acc += a_deq[i][p] * b_deq[j][p];
                }
                d[i * n + j] = bf16::from_f32(acc);
            }
        }
        d
    }

    pub fn supports_nvfp4(major: i32) -> bool {
        major >= 10
    }

    #[cfg(test)]
    mod tactic_tests {
        use super::Nvfp4GemmRunner;

        #[test]
        fn skinny_tiles_covers_verify_shapes_not_prefill() {
            assert!(Nvfp4GemmRunner::skinny_tiles(128, 5376));
            assert!(Nvfp4GemmRunner::skinny_tiles(128, 5376 * 2));
            assert!(Nvfp4GemmRunner::skinny_tiles(128, 20480));
            assert!(!Nvfp4GemmRunner::skinny_tiles(128, 43008));
            assert!(!Nvfp4GemmRunner::skinny_tiles(2048, 5376));
            assert!(!Nvfp4GemmRunner::skinny_tiles(1024, 43008));
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub fn supports_nvfp4(_major: i32) -> bool {
    false
}

#[cfg(test)]
mod streamk_k256_gate_tests {
    use super::streamk_k256_from;

    #[test]
    fn ships_default_off() {
        assert!(!streamk_k256_from(None));
        assert!(!streamk_k256_from(Some("")));
        assert!(!streamk_k256_from(Some("0")));
    }

    #[test]
    fn env_one_enables() {
        assert!(streamk_k256_from(Some("1")));
        assert!(streamk_k256_from(Some("on")));
    }
}

#[cfg(test)]
mod b2_trio_default_pin_tests {
    use super::{k256_from, skinny_lt_from};

    #[test]
    fn skinny_lt_ships_default_on() {
        assert!(skinny_lt_from(None));
        assert!(skinny_lt_from(Some("1")));
        assert!(skinny_lt_from(Some("")));
    }

    #[test]
    fn skinny_lt_env_zero_disables() {
        assert!(!skinny_lt_from(Some("0")));
    }

    #[test]
    fn k256_ships_default_on() {
        assert!(k256_from(None));
        assert!(k256_from(Some("1")));
        assert!(k256_from(Some("")));
    }

    #[test]
    fn k256_env_zero_disables() {
        assert!(!k256_from(Some("0")));
    }
}

#[cfg(test)]
mod ue4m3_tests {
    use super::*;

    fn hw_decode(byte: u8) -> f32 {
        let e = (byte >> 3) & 0x0F;
        let m = (byte & 0x07) as f32;
        if e == 0 {
            (m / 8.0) * (2f32).powi(-6)
        } else {
            (1.0 + m / 8.0) * (2f32).powi(e as i32 - 7)
        }
    }

    #[test]
    fn decode_matches_hardware_for_every_byte() {
        for byte in 0u8..=0x7E {
            assert_eq!(
                decode_ue4m3(byte),
                hw_decode(byte),
                "byte 0x{byte:02x} decodes differently from hardware"
            );
        }
    }

    #[test]
    fn subnormal_grid_points_round_trip_exactly() {
        let step = 2f32.powi(-9);
        for m in 0u8..=7 {
            let v = m as f32 * step;
            if m > 0 {
                assert_eq!(encode_ue4m3(v), m, "encode({m} * 2^-9)");
            }
            assert_eq!(decode_ue4m3(m), v, "decode(byte {m})");
            assert_eq!(hw_decode(m), v, "hw_decode(byte {m})");
        }

        assert_eq!(encode_ue4m3(8.0 * step), 0x08);
        assert_eq!(decode_ue4m3(0x08), 2f32.powi(-6));
    }

    #[test]
    fn subnormal_boundary_cases_encode_to_nearest_hardware_value() {
        let step = 2f32.powi(-9);

        for m in 0u8..=6 {
            let below = (m as f32 + 0.49) * step;
            let above = (m as f32 + 0.51) * step;
            assert_eq!(encode_ue4m3(below), m, "({m}+0.49)*2^-9 must round down");
            assert_eq!(encode_ue4m3(above), m + 1, "({m}+0.51)*2^-9 must round up");
        }

        assert_eq!(encode_ue4m3(7.6 * step), 0x08);

        assert_eq!(encode_ue4m3(0.49 * step), 0);
        assert_eq!(encode_ue4m3(0.51 * step), 1);
    }

    #[test]
    fn old_biased_exp0_bug_cases_now_match_hardware() {
        let cases = [
            2f32.powi(-7),
            1.5 * 2f32.powi(-7),
            2.5 * 2f32.powi(-9),
            2f32.powi(-8),
            2f32.powi(-9),
            2f32.powi(-20),
        ];
        let expected_bytes = [4u8, 6, 3, 2, 1, 0];
        for (v, want) in cases.iter().zip(expected_bytes) {
            let b = encode_ue4m3(*v);
            assert_eq!(b, want, "encode({v})");
            let hw = hw_decode(b);

            assert!(
                (hw - v).abs() <= 0.5 * 2f32.powi(-9) + f32::EPSILON,
                "encode({v}) -> byte 0x{b:02x} -> hw {hw}: not nearest"
            );

            assert_eq!(decode_ue4m3(b), hw);
        }
    }

    #[test]
    fn normal_range_encoding_unchanged() {
        assert_eq!(encode_ue4m3(1.0), 0x38);
        assert_eq!(decode_ue4m3(0x38), 1.0);
        assert_eq!(encode_ue4m3(448.0), 0x7E);
        assert_eq!(encode_ue4m3(1e9), 0x7E);
        assert_eq!(encode_ue4m3(2f32.powi(-6)), 0x08);
        for byte in 0x08u8..=0x7E {
            let v = hw_decode(byte);
            assert_eq!(encode_ue4m3(v), byte, "normal round trip byte 0x{byte:02x}");
        }
    }
}

#[cfg(test)]
mod paper_validation {

    use super::*;

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]

    fn spec_e4m3_decode(byte: u8) -> f32 {
        let exp = ((byte >> 3) & 0x0F) as i32;
        let mant = (byte & 0x07) as f32;
        if exp == 0 {
            (mant / 8.0) * (2f32).powi(-6)
        } else {
            (1.0 + mant / 8.0) * (2f32).powi(exp - 7)
        }
    }

    #[test]
    fn e2m1_codebook_is_the_nvfp4_value_set_and_roundtrips() {
        let expect = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (i, &v) in expect.iter().enumerate() {
            assert_eq!(decode_e2m1(i as u8), v);
            assert_eq!(
                encode_e2m1(v) & 0x07,
                i as u8,
                "encode({v}) must return code {i}"
            );
            assert_eq!(decode_e2m1((i as u8) | 0x8), -v);
        }
        for code in 0u8..16 {
            let v = decode_e2m1(code);
            let back = encode_e2m1(v);
            assert_eq!(
                decode_e2m1(back),
                v,
                "roundtrip through decode failed for code {code}"
            );
        }
    }

    #[test]
    fn e2m1_encode_is_nearest_value() {
        let codebook = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let mut x = -8.0f32;
        while x < 8.0 {
            let enc = encode_e2m1(x);
            let dec = decode_e2m1(enc);
            let best = codebook
                .iter()
                .map(|&c| (x.abs() - c).abs())
                .fold(f32::INFINITY, f32::min);
            assert!(
                ((x.abs() - dec.abs()).abs() - best).abs() < 1e-6,
                "encode({x}) chose {dec}, not a nearest codepoint (best err {best})"
            );
            x += 0.111;
        }
    }

    #[test]
    fn deviation_e2m1_midpoint_rounding_is_toward_lower_code_not_rne() {
        assert_eq!(
            encode_e2m1(0.75) & 0x7,
            1,
            "0.75 -> 0.5 (RNE would give 1.0/code 2)"
        );
        assert_eq!(
            encode_e2m1(1.75) & 0x7,
            3,
            "1.75 -> 1.5 (RNE would give 2.0/code 4)"
        );
        assert_eq!(
            encode_e2m1(3.5) & 0x7,
            5,
            "3.5 -> 3.0 (RNE would give 4.0/code 6)"
        );

        assert_eq!(encode_e2m1(0.25) & 0x7, 0);
        assert_eq!(encode_e2m1(2.5) & 0x7, 4);
        assert_eq!(encode_e2m1(5.0) & 0x7, 6);
    }

    #[test]
    fn pack_unpack_e2m1_pairs_are_inverse() {
        for lo in 0u8..16 {
            for hi in 0u8..16 {
                assert_eq!(unpack_e2m1_pair(pack_e2m1_pair(lo, hi)), (lo, hi));
            }
        }
    }

    #[test]
    fn ue4m3_roundtrip_relative_error_bounded_in_normal_range() {
        let mut s = 0.0078125f32;
        while s <= 448.0 {
            for mul in [1.0f32, 1.03, 1.11, 1.499, 1.87] {
                let v = (s * mul).min(448.0);
                let dec = decode_ue4m3(encode_ue4m3(v));
                let rel = (dec - v).abs() / v;
                assert!(
                    rel <= 0.125 + 1e-6,
                    "ue4m3 roundtrip rel err {rel} at {v} (decoded {dec})"
                );
            }
            s *= 2.0;
        }

        assert_eq!(decode_ue4m3(encode_ue4m3(448.0)), 448.0);
        assert_eq!(decode_ue4m3(encode_ue4m3(1.0e6)), 448.0);
        assert_ne!(
            encode_ue4m3(447.9999),
            0x7F,
            "E4M3 NaN code must never be emitted"
        );
    }

    #[test]
    fn block_quantize_roundtrip_error_bound() {
        let mut rng = 0x12345678u64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as f64 / (1u64 << 31) as f64) as f32 - 0.5
        };
        for amp in [0.05f32, 1.0, 37.0, 400.0] {
            for _ in 0..50 {
                let vals: Vec<f32> = (0..BLOCK_SIZE).map(|_| next() * 2.0 * amp).collect();
                let (packed, scale_byte) = quantize_block(&vals);
                let deq = dequantize_block(&packed, scale_byte);
                let scale = decode_ue4m3(scale_byte);
                for (a, b) in vals.iter().zip(deq.iter()) {
                    assert!(
                        (a - b).abs() <= scale * 1.0 + 1e-6,
                        "block roundtrip |{a} - {b}| > scale {scale}"
                    );
                }
            }
        }
    }

    #[test]
    fn block_quantize_zero_block_is_exact() {
        let vals = vec![0.0f32; BLOCK_SIZE];
        let (packed, scale_byte) = quantize_block(&vals);
        let deq = dequantize_block(&packed, scale_byte);
        assert!(
            deq.iter().all(|&v| v == 0.0),
            "zero block must roundtrip to zeros"
        );
    }

    #[test]
    fn swizzle_scales_matches_cutlass_sm1xx_layout_and_is_injective() {
        fn reference_dst(m: usize, kb: usize, k_blocks: usize) -> usize {
            let k_tiles = k_blocks.div_ceil(4);
            let tile = (m / 128) * k_tiles + kb / 4;
            tile * 512 + (m % 32) * 16 + ((m / 32) % 4) * 4 + (kb % 4)
        }
        for (rows, k_blocks) in [(1usize, 1usize), (7, 3), (128, 4), (130, 9), (256, 8)] {
            let linear: Vec<u8> = (0..rows * k_blocks).map(|i| (i % 251) as u8 + 1).collect();
            let sw = swizzle_scales(&linear, rows, k_blocks);
            let m_tiles = rows.div_ceil(128);
            let k_tiles = k_blocks.div_ceil(4);
            assert_eq!(sw.len(), m_tiles * 128 * k_tiles * 4);
            let mut seen = vec![false; sw.len()];
            for m in 0..rows {
                for kb in 0..k_blocks {
                    let dst = reference_dst(m, kb, k_blocks);
                    assert_eq!(
                        sw[dst],
                        linear[m * k_blocks + kb],
                        "scale for (m={m}, kb={kb}) not at the CUTLASS slot"
                    );
                    assert!(!seen[dst], "two scales collided at slot {dst}");
                    seen[dst] = true;
                }
            }
        }
    }

    fn det_weight(out_f: usize, in_f: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as u32 as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        (0..out_f * in_f).map(|_| next() * 0.35).collect()
    }

    fn rel_rms(a: &[f32], b: &[f32]) -> f32 {
        let mut num = 0f64;
        let mut den = 0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            num += ((x - y) as f64).powi(2);
            den += (*y as f64).powi(2);
        }
        (num / den.max(1e-30)).sqrt() as f32
    }

    #[test]
    fn unswizzle_inverts_swizzle() {
        for (rows, k_blocks) in [(1usize, 1usize), (7, 3), (128, 4), (130, 9), (256, 8)] {
            let linear: Vec<u8> = (0..rows * k_blocks).map(|i| (i % 251) as u8 + 1).collect();
            let sw = swizzle_scales(&linear, rows, k_blocks);
            let back = unswizzle_scales(&sw, rows, k_blocks);
            assert_eq!(
                back, linear,
                "unswizzle(swizzle(x)) != x for {rows}x{k_blocks}"
            );
        }
    }

    #[test]
    fn packed_linear_dequant_equals_tensor_dequant_bit_for_bit() {
        let (out_f, in_f) = (5usize, 32usize);
        let flat = det_weight(out_f, in_f, 0xA11CE);
        let amax = flat.iter().fold(0f32, |a, &b| a.max(b.abs()));
        let stored_global = (448.0f32 * 6.0) / amax;
        let mult = 1.0 / stored_global;
        let rows: Vec<Vec<f32>> = (0..out_f)
            .map(|r| flat[r * in_f..(r + 1) * in_f].to_vec())
            .collect();
        let q = Nvfp4Tensor::quantize_rows_with_global(&rows, stored_global);
        let via_free = dequantize_packed_linear(&q.data, &q.scales, out_f, in_f, mult);
        let via_tensor: Vec<f32> = q.dequantize_scaled(mult).into_iter().flatten().collect();
        assert_eq!(
            via_free, via_tensor,
            "packed-linear dequant != tensor dequant"
        );

        let via_sw = dequantize_packed_swizzled(&q.data, &q.scales_swizzled(), out_f, in_f, mult);
        assert_eq!(via_sw, via_free, "swizzled dequant != linear dequant");
    }

    #[test]
    fn packed_dequant_roundtrip_relrms_matches_nvfp4_grid() {
        let (out_f, in_f) = (128usize, 256usize);
        let w = det_weight(out_f, in_f, 0xDECAF);
        let amax = w.iter().fold(0f32, |a, &b| a.max(b.abs()));
        let stored_global = (448.0f32 * 6.0) / amax;
        let mult = 1.0 / stored_global;
        let rows: Vec<Vec<f32>> = (0..out_f)
            .map(|r| w[r * in_f..(r + 1) * in_f].to_vec())
            .collect();
        let q = Nvfp4Tensor::quantize_rows_with_global(&rows, stored_global);
        let deq = dequantize_packed_linear(&q.data, &q.scales, out_f, in_f, mult);
        let rr = rel_rms(&deq, &w);
        println!("PACKED_DEQUANT_ROUNDTRIP_RELRMS {rr:.6e} (amax={amax:.4})");

        assert!(
            rr > 0.005,
            "rel-RMS {rr} implausibly small -- scales likely mis-decoded"
        );
        assert!(
            rr < 0.15,
            "rel-RMS {rr} exceeds the nvfp4 grid floor -- unpack unfaithful"
        );
    }

    #[test]
    fn tensor_dequantize_inverts_quantize_within_block_bound() {
        let rows: Vec<Vec<f32>> = (0..4)
            .map(|r| {
                (0..32)
                    .map(|c| ((r * 37 + c * 13) as f32 % 11.0) - 5.0)
                    .collect()
            })
            .collect();
        let t = Nvfp4Tensor::quantize_rows(&rows);
        let deq = t.dequantize();
        for (r, (orig, got)) in rows.iter().zip(deq.iter()).enumerate() {
            for (b, chunk) in orig.chunks(BLOCK_SIZE).enumerate() {
                let scale = decode_ue4m3(t.scales[r * (32 / BLOCK_SIZE) + b]);
                for (i, &x) in chunk.iter().enumerate() {
                    let y = got[b * BLOCK_SIZE + i];
                    assert!(
                        (x - y).abs() <= scale + 1e-6,
                        "row {r} block {b} elem {i}: |{x} - {y}| > {scale}"
                    );
                }
            }
        }
    }
}
