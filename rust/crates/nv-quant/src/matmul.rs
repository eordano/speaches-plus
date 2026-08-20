#[cfg(feature = "cuda")]
pub use cuda::*;

pub fn fused_qkv_bitwise_safe(m: usize, has_v: bool) -> bool {
    (2..=16).contains(&m) && (has_v || m >= 5)
}

pub fn deterministic_mode_from(raw: Option<&str>) -> bool {
    raw == Some("1")
}

pub fn splitk_timing_selection_enabled(splitk: bool, deterministic: bool) -> bool {
    splitk && !deterministic
}

#[cfg(feature = "cuda")]
mod cuda {
    use anyhow::Result;
    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::cublaslt::result as lt_result;
    use cudarc::cublaslt::sys as lt_sys;
    use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use half::bf16;
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::mem;
    use std::sync::{Arc, Mutex, OnceLock};

    static HANDLE_CACHE: OnceLock<Mutex<HashMap<usize, CudaBlasLT>>> = OnceLock::new();

    fn ensure_handle(stream: &Arc<CudaStream>) -> Result<()> {
        let cache = HANDLE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let key = crate::stream_cache_key(stream);
        let mut guard = cache
            .lock()
            .map_err(|e| anyhow::anyhow!("gemm handle cache poisoned: {e}"))?;
        if !guard.contains_key(&key) {
            guard.insert(key, CudaBlasLT::new(stream.clone())?);
        }
        Ok(())
    }

    fn with_handle<R>(
        stream: &Arc<CudaStream>,
        f: impl FnOnce(&CudaBlasLT) -> Result<R>,
    ) -> Result<R> {
        let cache = HANDLE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let key = crate::stream_cache_key(stream);
        let mut guard = cache
            .lock()
            .map_err(|e| anyhow::anyhow!("gemm handle cache poisoned: {e}"))?;
        if !guard.contains_key(&key) {
            guard.insert(key, CudaBlasLT::new(stream.clone())?);
        }
        f(guard.get(&key).unwrap())
    }

    pub struct TensorCoreGemm;

    impl TensorCoreGemm {
        pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
            ensure_handle(&stream)?;
            Ok(Self)
        }

        pub fn bf16_matmul_row_major(
            &self,
            stream: &Arc<CudaStream>,
            a: &CudaSlice<bf16>,
            b: &CudaSlice<bf16>,
            c: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            alpha: f32,
            beta: f32,
        ) -> Result<()> {
            self.bf16_matmul_row_major_offs(stream, a, 0, b, 0, c, m, n, k, alpha, beta)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn bf16_matmul_row_major_offs(
            &self,
            stream: &Arc<CudaStream>,
            a: &CudaSlice<bf16>,
            a_off: usize,
            b: &CudaSlice<bf16>,
            b_off: usize,
            c: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            alpha: f32,
            beta: f32,
        ) -> Result<()> {
            let cfg = MatmulConfig {
                transa: false,
                transb: false,
                transc: false,
                m: n,
                n: m,
                k,
                alpha,
                lda: n as i64,
                ldb: k as i64,
                beta,
                ldc: n as i64,
                stride_a: None,
                stride_b: None,
                stride_c: None,
                stride_bias: None,
                batch_size: None,
            };
            let bv = b.slice(b_off..);
            let av = a.slice(a_off..);
            with_handle(stream, |lt| {
                unsafe {
                    <CudaBlasLT as Matmul<bf16>>::matmul(lt, cfg, &bv, &av, c, None, None)?;
                }
                Ok(())
            })
        }

        pub fn bf16_matmul_row_major_bt(
            &self,
            stream: &Arc<CudaStream>,
            a: &CudaSlice<bf16>,
            w: &CudaSlice<bf16>,
            c: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            alpha: f32,
            beta: f32,
        ) -> Result<()> {
            self.bf16_matmul_row_major_bt_offs(stream, a, 0, w, 0, c, m, n, k, alpha, beta)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn bf16_matmul_row_major_bt_off(
            &self,
            stream: &Arc<CudaStream>,
            a: &CudaSlice<bf16>,
            w: &CudaSlice<bf16>,
            w_off: usize,
            c: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            alpha: f32,
            beta: f32,
        ) -> Result<()> {
            self.bf16_matmul_row_major_bt_offs(stream, a, 0, w, w_off, c, m, n, k, alpha, beta)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn bf16_matmul_row_major_bt_offs(
            &self,
            stream: &Arc<CudaStream>,
            a: &CudaSlice<bf16>,
            a_off: usize,
            w: &CudaSlice<bf16>,
            w_off: usize,
            c: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            alpha: f32,
            beta: f32,
        ) -> Result<()> {
            let cfg = MatmulConfig {
                transa: true,
                transb: false,
                transc: false,
                m: n,
                n: m,
                k,
                alpha,
                lda: k as i64,
                ldb: k as i64,
                beta,
                ldc: n as i64,
                stride_a: None,
                stride_b: None,
                stride_c: None,
                stride_bias: None,
                batch_size: None,
            };
            let wv = w.slice(w_off..);
            let av = a.slice(a_off..);
            with_handle(stream, |lt| {
                unsafe {
                    <CudaBlasLT as Matmul<bf16>>::matmul(lt, cfg, &wv, &av, c, None, None)?;
                }
                Ok(())
            })
        }

        pub fn bf16_matmul_row_major_bt_det(
            &self,
            stream: &Arc<CudaStream>,
            a: &CudaSlice<bf16>,
            w: &CudaSlice<bf16>,
            c: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            alpha: f32,
            beta: f32,
        ) -> Result<()> {
            det_bt_matmul(
                stream,
                a,
                0,
                w,
                0,
                c,
                m,
                n,
                k,
                alpha,
                beta,
                splitk_enabled(),
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn bf16_matmul_row_major_bt_det_offs(
            &self,
            stream: &Arc<CudaStream>,
            a: &CudaSlice<bf16>,
            a_off: usize,
            w: &CudaSlice<bf16>,
            w_off: usize,
            c: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            alpha: f32,
            beta: f32,
        ) -> Result<()> {
            det_bt_matmul(
                stream,
                a,
                a_off,
                w,
                w_off,
                c,
                m,
                n,
                k,
                alpha,
                beta,
                splitk_enabled(),
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn bf16_matmul_row_major_bt_det_nosplit(
            &self,
            stream: &Arc<CudaStream>,
            a: &CudaSlice<bf16>,
            w: &CudaSlice<bf16>,
            c: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            alpha: f32,
            beta: f32,
        ) -> Result<()> {
            det_bt_matmul(stream, a, 0, w, 0, c, m, n, k, alpha, beta, false)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn bf16_matmul_row_major_bt_det_splitk(
            &self,
            stream: &Arc<CudaStream>,
            a: &CudaSlice<bf16>,
            w: &CudaSlice<bf16>,
            c: &mut CudaSlice<bf16>,
            m: u64,
            n: u64,
            k: u64,
            alpha: f32,
            beta: f32,
        ) -> Result<()> {
            det_bt_matmul(stream, a, 0, w, 0, c, m, n, k, alpha, beta, true)
        }
    }

    fn bf16_algo_pins() -> &'static HashMap<(u64, u64, u64), usize> {
        static PINS: OnceLock<HashMap<(u64, u64, u64), usize>> = OnceLock::new();
        PINS.get_or_init(|| crate::algo_pin::pin_map_from_env("NV_BF16_ALGO_PIN"))
    }

    pub fn nondeterministic_reduction_was_explicitly_allowed() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("NV_BF16_ALLOW_NONDET_REDUCTION").ok().as_deref() == Some("1")
        })
    }

    pub fn splitk_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("NV_BF16_SPLITK")
                .map(|v| v != "0")
                .unwrap_or(true)
        })
    }

    pub fn deterministic_mode() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            super::deterministic_mode_from(std::env::var("NV_DETERMINISTIC").ok().as_deref())
        })
    }

    pub fn release_stream_state(cu_stream_key: usize) {
        if let Some(cache) = HANDLE_CACHE.get() {
            if let Ok(mut g) = cache.lock() {
                g.remove(&cu_stream_key);
            }
        }
        if let Some(cache) = DET_HANDLE_CACHE.get() {
            if let Ok(mut g) = cache.lock() {
                if let Some(det) = g.remove(&cu_stream_key) {
                    unsafe {
                        let _ = lt_result::destroy_handle(det.handle);
                    }
                }
            }
        }
    }

    fn stream_is_capturing(stream: &Arc<CudaStream>) -> bool {
        use cudarc::driver::sys as drv;
        let mut st = drv::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let rc = unsafe { drv::cuStreamIsCapturing(stream.cu_stream(), &mut st) };
        rc == drv::CUresult::CUDA_SUCCESS
            && st != drv::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    }

    static SPLITK_ALGO_CACHE: OnceLock<
        Mutex<HashMap<(u64, u64, u64), lt_sys::cublasLtMatmulAlgo_t>>,
    > = OnceLock::new();

    struct DetLt {
        handle: lt_sys::cublasLtHandle_t,
        workspace: CudaSlice<u8>,
        workspace_bytes: usize,
    }

    unsafe impl Send for DetLt {}

    static DET_HANDLE_CACHE: OnceLock<Mutex<HashMap<usize, DetLt>>> = OnceLock::new();

    fn ensure_det_handle(stream: &Arc<CudaStream>) -> Result<()> {
        let cache = DET_HANDLE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let key = crate::stream_cache_key(stream);
        let mut guard = cache
            .lock()
            .map_err(|e| anyhow::anyhow!("det gemm handle cache poisoned: {e}"))?;
        if !guard.contains_key(&key) {
            let handle = lt_result::create_handle()
                .map_err(|e| anyhow::anyhow!("cublasLt create: {e:?}"))?;

            let workspace_bytes = 32 * 1024 * 1024;
            let workspace = stream.alloc_zeros::<u8>(workspace_bytes)?;
            guard.insert(
                key,
                DetLt {
                    handle,
                    workspace,
                    workspace_bytes,
                },
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn det_bt_matmul(
        stream: &Arc<CudaStream>,
        a: &CudaSlice<bf16>,
        a_off: usize,
        w: &CudaSlice<bf16>,
        w_off: usize,
        c: &mut CudaSlice<bf16>,
        m: u64,
        n: u64,
        k: u64,
        alpha: f32,
        beta: f32,
        splitk: bool,
    ) -> Result<()> {
        let wv = w.slice(w_off..);
        let av = a.slice(a_off..);
        let (wp, _gw) = wv.device_ptr(stream);
        let (ap, _ga) = av.device_ptr(stream);
        let (cp, _gc) = c.device_ptr_mut(stream);
        unsafe {
            bf16_bt_matmul_det_raw(
                stream,
                ap as *const c_void,
                wp as *const c_void,
                cp as *mut c_void,
                m,
                n,
                k,
                alpha,
                beta,
                splitk,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn bf16_bt_matmul_det_raw(
        stream: &Arc<CudaStream>,
        ap: *const c_void,
        wp: *const c_void,
        cp: *mut c_void,
        m: u64,
        n: u64,
        k: u64,
        alpha: f32,
        beta: f32,
        splitk: bool,
    ) -> Result<()> {
        ensure_det_handle(stream)?;
        let cache = DET_HANDLE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let key = crate::stream_cache_key(stream);
        let mut guard = cache
            .lock()
            .map_err(|e| anyhow::anyhow!("det gemm handle cache poisoned: {e}"))?;
        let det = guard.get_mut(&key).expect("ensured above");

        {
            let bf = lt_sys::cudaDataType_t::CUDA_R_16BF;

            let a_layout = lt_result::create_matrix_layout(bf, k, n, k as i64)
                .map_err(|e| anyhow::anyhow!("layout: {e:?}"))?;
            let b_layout = lt_result::create_matrix_layout(bf, k, m, k as i64)
                .map_err(|e| anyhow::anyhow!("layout: {e:?}"))?;
            let d_layout = lt_result::create_matrix_layout(bf, n, m, n as i64)
                .map_err(|e| anyhow::anyhow!("layout: {e:?}"))?;
            let desc = lt_result::create_matmul_desc(
                lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                lt_sys::cudaDataType_t::CUDA_R_32F,
            )
            .map_err(|e| anyhow::anyhow!("desc: {e:?}"))?;
            let t = cublasOperation_t::CUBLAS_OP_T;
            let nn = cublasOperation_t::CUBLAS_OP_N;
            lt_result::set_matmul_desc_attribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &t as *const _ as *const c_void,
                mem::size_of::<cublasOperation_t>(),
            )
            .map_err(|e| anyhow::anyhow!("transa: {e:?}"))?;
            lt_result::set_matmul_desc_attribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &nn as *const _ as *const c_void,
                mem::size_of::<cublasOperation_t>(),
            )
            .map_err(|e| anyhow::anyhow!("transb: {e:?}"))?;

            let (wsp, _gws) = det.workspace.device_ptr_mut(stream);

            let lt_handle = det.handle;
            let ws_bytes = det.workspace_bytes;
            let algo = if super::splitk_timing_selection_enabled(splitk, deterministic_mode()) {
                match splitk_algo_for(
                    lt_handle,
                    ws_bytes,
                    stream,
                    desc,
                    a_layout,
                    b_layout,
                    d_layout,
                    (m, n, k),
                    wp,
                    ap,
                    cp,
                    wsp as *mut c_void,
                    alpha,
                    beta,
                ) {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("[bf16-splitk] selection failed for {m}x{n}x{k}, falling back to no-split heuristic: {e}");
                        legacy_det_algo(lt_handle, ws_bytes, desc, a_layout, b_layout, d_layout)?
                    }
                }
            } else {
                legacy_det_algo(lt_handle, ws_bytes, desc, a_layout, b_layout, d_layout)?
            };

            let rc = lt_result::matmul(
                det.handle,
                desc,
                &alpha as *const _ as *const c_void,
                &beta as *const _ as *const c_void,
                wp,
                a_layout,
                ap,
                b_layout,
                cp as *const c_void,
                d_layout,
                cp,
                d_layout,
                &algo as *const _,
                wsp as *mut c_void,
                det.workspace_bytes,
                stream.cu_stream() as lt_sys::cudaStream_t,
            );
            let desc_rc = lt_result::destroy_matmul_desc(desc);
            let a_rc = lt_result::destroy_matrix_layout(a_layout);
            let b_rc = lt_result::destroy_matrix_layout(b_layout);
            let d_rc = lt_result::destroy_matrix_layout(d_layout);
            rc.map_err(|e| anyhow::anyhow!("cublasLtMatmul det bf16: {e:?}"))?;
            desc_rc.map_err(|e| anyhow::anyhow!("desc free: {e:?}"))?;
            a_rc.map_err(|e| anyhow::anyhow!("layout free: {e:?}"))?;
            b_rc.map_err(|e| anyhow::anyhow!("layout free: {e:?}"))?;
            d_rc.map_err(|e| anyhow::anyhow!("layout free: {e:?}"))?;
        }
        Ok(())
    }

    unsafe fn legacy_det_algo(
        handle: lt_sys::cublasLtHandle_t,
        ws_bytes: usize,
        desc: lt_sys::cublasLtMatmulDesc_t,
        a_layout: lt_sys::cublasLtMatrixLayout_t,
        b_layout: lt_sys::cublasLtMatrixLayout_t,
        d_layout: lt_sys::cublasLtMatrixLayout_t,
    ) -> Result<lt_sys::cublasLtMatmulAlgo_t> {
        let pref = lt_result::create_matmul_pref().map_err(|e| anyhow::anyhow!("pref: {e:?}"))?;
        lt_result::set_matmul_pref_attribute(
            pref,
            lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws_bytes as *const _ as *const c_void,
            mem::size_of::<usize>(),
        )
        .map_err(|e| anyhow::anyhow!("pref ws: {e:?}"))?;
        let mask: u32 = 0;
        lt_result::set_matmul_pref_attribute(
            pref,
            lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
            &mask as *const _ as *const c_void,
            mem::size_of::<u32>(),
        )
        .map_err(|e| anyhow::anyhow!("pref mask: {e:?}"))?;
        let heur = lt_result::get_matmul_algo_heuristic(
            handle, desc, a_layout, b_layout, d_layout, d_layout, pref,
        )
        .map_err(|e| anyhow::anyhow!("heuristic: {e:?}"))?;
        lt_result::destroy_matmul_pref(pref).map_err(|e| anyhow::anyhow!("pref free: {e:?}"))?;
        Ok(heur.algo)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn splitk_algo_for(
        handle: lt_sys::cublasLtHandle_t,
        ws_bytes: usize,
        stream: &Arc<CudaStream>,
        desc: lt_sys::cublasLtMatmulDesc_t,
        a_layout: lt_sys::cublasLtMatrixLayout_t,
        b_layout: lt_sys::cublasLtMatrixLayout_t,
        d_layout: lt_sys::cublasLtMatrixLayout_t,
        shape: (u64, u64, u64),
        wp: *const c_void,
        ap: *const c_void,
        cp: *mut c_void,
        wsp: *mut c_void,
        alpha: f32,
        beta: f32,
    ) -> Result<lt_sys::cublasLtMatmulAlgo_t> {
        let cache = SPLITK_ALGO_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(a) = cache
            .lock()
            .map_err(|e| anyhow::anyhow!("splitk algo cache poisoned: {e}"))?
            .get(&shape)
            .copied()
        {
            return Ok(a);
        }
        let (m, n, k) = shape;

        let pref = lt_result::create_matmul_pref().map_err(|e| anyhow::anyhow!("pref: {e:?}"))?;
        lt_result::set_matmul_pref_attribute(
            pref,
            lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws_bytes as *const _ as *const c_void,
            mem::size_of::<usize>(),
        )
        .map_err(|e| anyhow::anyhow!("pref ws: {e:?}"))?;
        if !nondeterministic_reduction_was_explicitly_allowed() {
            let mask: u32 = 0x6;
            lt_result::set_matmul_pref_attribute(
                pref,
                lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                &mask as *const _ as *const c_void,
                mem::size_of::<u32>(),
            )
            .map_err(|e| anyhow::anyhow!("pref scheme mask: {e:?}"))?;
        }

        const MAX_CANDS: usize = 16;
        let mut results: [lt_sys::cublasLtMatmulHeuristicResult_t; MAX_CANDS] = mem::zeroed();
        let mut returned: i32 = 0;
        let st = lt_sys::cublasLtMatmulAlgoGetHeuristic(
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
        lt_result::destroy_matmul_pref(pref).map_err(|e| anyhow::anyhow!("pref free: {e:?}"))?;
        anyhow::ensure!(
            st == lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS && returned > 0,
            "cublasLtMatmulAlgoGetHeuristic: {st:?} returned={returned}"
        );
        let cands: Vec<&lt_sys::cublasLtMatmulHeuristicResult_t> = results[..returned as usize]
            .iter()
            .filter(|r| r.state == lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            .collect();
        anyhow::ensure!(!cands.is_empty(), "no valid heuristic candidates");

        if let Some(&pin) = bf16_algo_pins().get(&shape) {
            let idx = if pin < cands.len() {
                pin
            } else {
                eprintln!(
                    "[bf16-splitk] pin idx {pin} out of range for {m}x{n}x{k} ({} cands); using 0",
                    cands.len()
                );
                0
            };
            eprintln!(
                "[bf16-splitk] shape {m}x{n}x{k}: pinned cand {idx}/{} ({})",
                cands.len(),
                crate::nvfp4::algo_config_summary(&cands[idx].algo),
            );
            let algo = cands[idx].algo;
            cache
                .lock()
                .map_err(|e| anyhow::anyhow!("splitk algo cache poisoned: {e}"))?
                .insert(shape, algo);
            return Ok(algo);
        }

        let run_once = |algo: &lt_sys::cublasLtMatmulAlgo_t| -> Result<()> {
            lt_result::matmul(
                handle,
                desc,
                &alpha as *const _ as *const c_void,
                &beta as *const _ as *const c_void,
                wp,
                a_layout,
                ap,
                b_layout,
                cp as *const c_void,
                d_layout,
                cp,
                d_layout,
                algo as *const _,
                wsp,
                ws_bytes,
                stream.cu_stream() as lt_sys::cudaStream_t,
            )
            .map_err(|e| anyhow::anyhow!("cand matmul: {e:?}"))
        };

        let chosen_idx = if stream_is_capturing(stream) || cands.len() == 1 {
            if stream_is_capturing(stream) {
                eprintln!(
                    "[bf16-splitk] WARN: first sight of shape {m}x{n}x{k} during graph capture; using heuristic[0] untimed"
                );
            }
            0usize
        } else {
            let ctx = stream.context().clone();
            stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
            let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
            let mut times = vec![f64::INFINITY; cands.len()];
            for _pass in 0..2 {
                for (i, cand) in cands.iter().enumerate() {
                    if run_once(&cand.algo).is_err() {
                        continue;
                    }
                    if run_once(&cand.algo).is_err() {
                        continue;
                    }
                    let e0 = ctx.new_event(flags).map_err(|e| anyhow::anyhow!(e))?;
                    let e1 = ctx.new_event(flags).map_err(|e| anyhow::anyhow!(e))?;
                    e0.record(stream).map_err(|e| anyhow::anyhow!(e))?;
                    let mut bad = false;
                    for _ in 0..10 {
                        if run_once(&cand.algo).is_err() {
                            bad = true;
                            break;
                        }
                    }
                    e1.record(stream).map_err(|e| anyhow::anyhow!(e))?;
                    stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
                    if bad {
                        continue;
                    }
                    let t = e0.elapsed_ms(&e1).map_err(|e| anyhow::anyhow!(e))? as f64 / 10.0;
                    if t < times[i] {
                        times[i] = t;
                    }
                }
            }
            let best = times.iter().cloned().fold(f64::INFINITY, f64::min);
            anyhow::ensure!(best.is_finite(), "all heuristic candidates failed to run");
            let idx = times.iter().position(|&t| t <= best * 1.05).unwrap_or(0);
            let hex: String = cands[idx]
                .algo
                .data
                .iter()
                .map(|w| format!("{w:016x}"))
                .collect();
            eprintln!(
                "[bf16-splitk] shape {m}x{n}x{k}: cands={} times_ms={:?} chosen={} t={:.4}ms (heur0 {:.4}ms) algo={}",
                cands.len(),
                times.iter().map(|t| (t * 1e4).round() / 1e4).collect::<Vec<_>>(),
                idx,
                times[idx],
                times[0],
                hex
            );
            idx
        };

        let algo = cands[chosen_idx].algo;
        cache
            .lock()
            .map_err(|e| anyhow::anyhow!("splitk algo cache poisoned: {e}"))?
            .insert(shape, algo);
        Ok(algo)
    }

    pub fn cpu_bf16_matmul_row_major(
        a: &[bf16],
        b: &[bf16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<bf16> {
        let mut c = vec![bf16::from_f32(0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    acc += a[i * k + p].to_f32() * b[p * n + j].to_f32();
                }
                c[i * n + j] = bf16::from_f32(acc);
            }
        }
        c
    }
}

#[cfg(not(feature = "cuda"))]
pub struct TensorCoreGemm;

#[cfg(test)]
mod deterministic_mode_tests {
    use super::{deterministic_mode_from, splitk_timing_selection_enabled};

    #[test]
    fn only_the_literal_one_enables_deterministic_mode() {
        assert!(deterministic_mode_from(Some("1")));
        assert!(!deterministic_mode_from(None));
        assert!(!deterministic_mode_from(Some("0")));
        assert!(!deterministic_mode_from(Some("")));
        assert!(!deterministic_mode_from(Some("true")));
        assert!(!deterministic_mode_from(Some("yes")));
        assert!(!deterministic_mode_from(Some(" 1")));
        assert!(!deterministic_mode_from(Some("2")));
    }

    #[test]
    fn deterministic_mode_bypasses_timed_splitk_algo_selection() {
        assert!(splitk_timing_selection_enabled(true, false));
        assert!(!splitk_timing_selection_enabled(true, true));
        assert!(!splitk_timing_selection_enabled(false, false));
        assert!(!splitk_timing_selection_enabled(false, true));
    }
}
