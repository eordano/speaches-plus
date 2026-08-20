use std::collections::HashMap;

use candle_core::{bail, DType, Device, Result, Tensor};

pub struct LoraModuleWeights {
    pub a: Tensor,
    pub b: Tensor,
}

pub struct LoraAdapter {
    pub scaling: f64,
    pub modules: HashMap<String, LoraModuleWeights>,
}

pub struct LoraModuleSpec {
    pub name: String,
    pub in_features: usize,
    pub out_features: usize,
}

impl LoraModuleSpec {
    pub fn new(name: impl Into<String>, in_features: usize, out_features: usize) -> Self {
        Self {
            name: name.into(),
            in_features,
            out_features,
        }
    }
}

pub struct LoraSlotStack {
    a_stacked: Tensor,
    b_stacked: Tensor,
    max_loras: usize,
    max_rank: usize,
    in_features: usize,
    out_features: usize,
}

impl LoraSlotStack {
    pub fn new(
        max_loras: usize,
        max_rank: usize,
        in_features: usize,
        out_features: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if max_loras == 0 || max_rank == 0 || in_features == 0 || out_features == 0 {
            bail!("lora slot stack dims must be non-zero");
        }
        let a_stacked = Tensor::zeros((max_loras, 1, max_rank, in_features), dtype, device)?;
        let b_stacked = Tensor::zeros((max_loras, 1, out_features, max_rank), dtype, device)?;
        Ok(Self {
            a_stacked,
            b_stacked,
            max_loras,
            max_rank,
            in_features,
            out_features,
        })
    }

    pub fn max_loras(&self) -> usize {
        self.max_loras
    }

    pub fn max_rank(&self) -> usize {
        self.max_rank
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn lora_a_stacked(&self) -> &Tensor {
        &self.a_stacked
    }

    pub fn lora_b_stacked(&self) -> &Tensor {
        &self.b_stacked
    }

    fn check_index(&self, index: usize) -> Result<()> {
        if index >= self.max_loras {
            bail!(
                "lora slot index {index} out of range (max_loras={})",
                self.max_loras
            );
        }
        Ok(())
    }

    pub fn reset_lora(&self, index: usize) -> Result<()> {
        self.check_index(index)?;
        let dtype = self.a_stacked.dtype();
        let device = self.a_stacked.device();
        let za = Tensor::zeros((1, 1, self.max_rank, self.in_features), dtype, device)?;
        self.a_stacked.slice_set(&za, 0, index)?;
        let zb = Tensor::zeros((1, 1, self.out_features, self.max_rank), dtype, device)?;
        self.b_stacked.slice_set(&zb, 0, index)?;
        Ok(())
    }

    pub fn set_lora(&self, index: usize, a: &Tensor, b: &Tensor) -> Result<()> {
        self.check_index(index)?;
        let (a_rows, a_cols) = a.dims2()?;
        let (b_rows, b_cols) = b.dims2()?;
        if a_rows > self.max_rank {
            bail!("lora_a rank {a_rows} exceeds max_rank {}", self.max_rank);
        }
        if a_cols > self.in_features {
            bail!(
                "lora_a in dim {a_cols} exceeds in_features {}",
                self.in_features
            );
        }
        if b_rows > self.out_features {
            bail!(
                "lora_b out dim {b_rows} exceeds out_features {}",
                self.out_features
            );
        }
        if b_cols > self.max_rank {
            bail!("lora_b rank {b_cols} exceeds max_rank {}", self.max_rank);
        }
        if a_rows != b_cols {
            bail!("lora rank mismatch: lora_a has rank {a_rows}, lora_b has rank {b_cols}");
        }
        self.reset_lora(index)?;
        let dtype = self.a_stacked.dtype();
        let a = a.to_dtype(dtype)?;
        let b = b.to_dtype(dtype)?;
        let a_pad = corner_pad(&a, self.max_rank, self.in_features)?.reshape((
            1,
            1,
            self.max_rank,
            self.in_features,
        ))?;
        self.a_stacked.slice_set(&a_pad, 0, index)?;
        let b_pad = corner_pad(&b, self.out_features, self.max_rank)?.reshape((
            1,
            1,
            self.out_features,
            self.max_rank,
        ))?;
        self.b_stacked.slice_set(&b_pad, 0, index)?;
        Ok(())
    }

    pub fn slot_a(&self, index: usize) -> Result<Tensor> {
        self.check_index(index)?;
        self.a_stacked
            .narrow(0, index, 1)?
            .reshape((self.max_rank, self.in_features))
    }

    pub fn slot_b(&self, index: usize) -> Result<Tensor> {
        self.check_index(index)?;
        self.b_stacked
            .narrow(0, index, 1)?
            .reshape((self.out_features, self.max_rank))
    }

    pub fn delta(&self, index: usize, x: &Tensor, scaling: f64) -> Result<Tensor> {
        let a = self.slot_a(index)?;
        let b = self.slot_b(index)?;
        let h = x.matmul(&a.t()?.contiguous()?)?;
        let d = h.matmul(&b.t()?.contiguous()?)?;
        d.affine(scaling, 0.0)
    }
}

fn corner_pad(t: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
    let (r, c) = t.dims2()?;
    let t = if r < rows {
        t.pad_with_zeros(0, 0, rows - r)?
    } else {
        t.clone()
    };
    let t = if c < cols {
        t.pad_with_zeros(1, 0, cols - c)?
    } else {
        t
    };
    t.contiguous()
}

#[cfg(feature = "cuda")]
pub use cuda_runtime::{LoraDispatch, LoraHook};

#[cfg(feature = "wgpu")]
pub use wgpu_runtime::{WgpuLoraDispatch, WgpuLoraHook, WgpuLoraPath, WGPU_FUSED_MAX_M};

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub use wgpu_runtime::{WgpuLoraDispatch as LoraDispatch, WgpuLoraHook as LoraHook};

#[cfg(feature = "wgpu")]
mod wgpu_runtime {
    use super::LoraSlotStack;
    use anyhow::{anyhow, bail, Result};
    use candle_core::{DType, Device, Tensor};
    use half::bf16;
    use nv_kernels::wgpu_backend::device::WgpuContext;
    use nv_kernels::wgpu_backend::kernels::lora as wgk;
    use std::sync::{Arc, Mutex};

    pub const WGPU_FUSED_MAX_M: usize = 64;
    pub const FUSED_MAX_RANK: usize = wgk::FUSED_MAX_RANK;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum WgpuLoraPath {
        Fused,
        Grouped,
    }

    struct DispatchState {
        m: usize,
        no_lora: bool,
        armed: bool,
        meta: Option<wgk::LoraMeta>,
    }

    pub struct WgpuLoraDispatch {
        ctx: &'static WgpuContext,
        device: Device,
        max_tokens: usize,
        max_loras: usize,
        state: Mutex<DispatchState>,
    }

    impl WgpuLoraDispatch {
        pub fn new(device: &Device, max_tokens: usize, max_loras: usize) -> Result<Arc<Self>> {
            let ctx = WgpuContext::shared()
                .map_err(|e| anyhow!("LoraDispatch requires a wgpu adapter: {e}"))?;
            Self::with_context(ctx, device, max_tokens, max_loras)
        }

        pub fn with_context(
            ctx: &'static WgpuContext,
            device: &Device,
            max_tokens: usize,
            max_loras: usize,
        ) -> Result<Arc<Self>> {
            if max_tokens == 0 || max_loras == 0 {
                bail!("LoraDispatch dims must be non-zero");
            }
            Ok(Arc::new(Self {
                ctx,
                device: device.clone(),
                max_tokens,
                max_loras,
                state: Mutex::new(DispatchState {
                    m: 0,
                    no_lora: true,
                    armed: false,
                    meta: None,
                }),
            }))
        }

        pub fn context(&self) -> &'static WgpuContext {
            self.ctx
        }

        pub fn device(&self) -> &Device {
            &self.device
        }

        pub fn max_tokens(&self) -> usize {
            self.max_tokens
        }

        pub fn max_loras(&self) -> usize {
            self.max_loras
        }

        pub fn grid_loras(&self) -> usize {
            self.max_loras + 1
        }

        pub fn set_mapping(&self, mapping: &[i32]) -> Result<()> {
            if mapping.is_empty() {
                bail!("LoraDispatch.set_mapping: empty mapping");
            }
            if mapping.len() > self.max_tokens {
                bail!(
                    "LoraDispatch.set_mapping: {} tokens exceeds max_tokens {}",
                    mapping.len(),
                    self.max_tokens
                );
            }
            for &v in mapping {
                if v < -1 || v >= self.max_loras as i32 {
                    bail!("LoraDispatch.set_mapping: slot {v} out of range");
                }
            }
            let meta = wgk::LoraMeta::prepare(mapping, self.max_loras);
            let mut st = self
                .state
                .lock()
                .map_err(|e| anyhow!("LoraDispatch state poisoned: {e}"))?;
            st.m = mapping.len();
            st.no_lora = meta.no_lora;
            st.meta = Some(meta);
            st.armed = true;
            Ok(())
        }

        pub fn disarm(&self) {
            if let Ok(mut st) = self.state.lock() {
                st.armed = false;
            }
        }

        pub fn armed(&self) -> bool {
            self.state
                .lock()
                .map(|s| s.armed && !s.no_lora)
                .unwrap_or(false)
        }

        fn snapshot(&self) -> Result<Option<(usize, wgk::LoraMeta)>> {
            let st = self
                .state
                .lock()
                .map_err(|e| anyhow!("LoraDispatch state poisoned: {e}"))?;
            if !st.armed || st.no_lora {
                return Ok(None);
            }
            match &st.meta {
                Some(meta) => Ok(Some((st.m, meta.clone()))),
                None => Ok(None),
            }
        }
    }

    pub struct WgpuLoraHook {
        dispatch: Arc<WgpuLoraDispatch>,
        a_tensors: Vec<Tensor>,
        b_tensors: Vec<Tensor>,
        widths: Vec<usize>,
        rank: usize,
        in_features: usize,
        out_features: usize,
        max_n: usize,
        n_slices: usize,
    }

    fn host_bits(t: &Tensor) -> Result<Vec<u16>> {
        let v = t.flatten_all()?.to_dtype(DType::BF16)?.to_vec1::<bf16>()?;
        Ok(v.into_iter().map(|x| x.to_bits()).collect())
    }

    fn grid_fits(ctx: &WgpuContext, grid: (usize, usize, usize)) -> bool {
        let limit = ctx.caps.max_compute_workgroups_per_dimension as usize;
        [grid.0, grid.1, grid.2]
            .iter()
            .all(|&g| g >= 1 && g <= limit)
    }

    impl WgpuLoraHook {
        pub fn from_stacks(
            dispatch: Arc<WgpuLoraDispatch>,
            stacks: &[&LoraSlotStack],
        ) -> Result<Arc<Self>> {
            if stacks.is_empty() {
                bail!("LoraHook needs at least one slot stack");
            }
            let rank = stacks[0].max_rank();
            let in_features = stacks[0].in_features();
            if rank > FUSED_MAX_RANK {
                bail!("LoraHook: max_rank {rank} exceeds kernel limit {FUSED_MAX_RANK}");
            }
            let mut widths = Vec::with_capacity(stacks.len());
            let mut a_tensors = Vec::with_capacity(stacks.len());
            let mut b_tensors = Vec::with_capacity(stacks.len());
            for st in stacks {
                if st.max_rank() != rank || st.in_features() != in_features {
                    bail!("LoraHook: slot stacks must share max_rank and in_features");
                }
                if st.max_loras() != dispatch.max_loras() {
                    bail!(
                        "LoraHook: stack max_loras {} != dispatch max_loras {}",
                        st.max_loras(),
                        dispatch.max_loras()
                    );
                }
                if st.lora_a_stacked().dtype() != DType::BF16 {
                    bail!("LoraHook: slot stacks must be bf16");
                }
                a_tensors.push(st.lora_a_stacked().clone());
                b_tensors.push(st.lora_b_stacked().clone());
                widths.push(st.out_features());
            }
            let out_features: usize = widths.iter().sum();
            let max_n = *widths.iter().max().unwrap();
            let n_slices = widths.len();
            Ok(Arc::new(Self {
                dispatch,
                a_tensors,
                b_tensors,
                widths,
                rank,
                in_features,
                out_features,
                max_n,
                n_slices,
            }))
        }

        pub fn in_features(&self) -> usize {
            self.in_features
        }

        pub fn out_features(&self) -> usize {
            self.out_features
        }

        pub fn dispatch(&self) -> &Arc<WgpuLoraDispatch> {
            &self.dispatch
        }

        fn grouped_grids(&self, m: usize, grid_loras: usize) -> [(usize, usize, usize); 2] {
            let cta_m = m.div_ceil(wgk::BLOCK_M as usize);
            [
                (
                    cta_m * self.rank.div_ceil(wgk::BLOCK_N as usize),
                    self.n_slices,
                    grid_loras,
                ),
                (
                    cta_m * self.max_n.div_ceil(wgk::BLOCK_N as usize),
                    self.n_slices,
                    grid_loras,
                ),
            ]
        }

        fn fused_grid(&self, m: usize, grid_loras: usize) -> (usize, usize, usize) {
            (
                m * self.max_n.div_ceil(wgk::FUSED_N_CHUNK as usize),
                self.n_slices,
                grid_loras,
            )
        }

        pub fn plan(&self, m: usize, win: Option<(usize, usize)>) -> Result<WgpuLoraPath> {
            let (win_off, win_len) = win.unwrap_or((0, self.out_features));
            let full = win_off == 0 && win_len == self.out_features;
            let ctx = self.dispatch.context();
            let grid_loras = self.dispatch.grid_loras();
            let grouped_ok = full
                && self
                    .grouped_grids(m, grid_loras)
                    .iter()
                    .all(|g| grid_fits(ctx, *g));
            let fused_ok = grid_fits(ctx, self.fused_grid(m, grid_loras));
            if (m > WGPU_FUSED_MAX_M && grouped_ok) || (grouped_ok && !fused_ok) {
                return Ok(WgpuLoraPath::Grouped);
            }
            if fused_ok {
                return Ok(WgpuLoraPath::Fused);
            }
            bail!(
                "LoraHook.apply: m={m} with max_n={} exceeds the wgpu grid bound {} \
                 (fused grid.x={}, grouped grid.x={}); chunk the batch",
                self.max_n,
                ctx.caps.max_compute_workgroups_per_dimension,
                self.fused_grid(m, grid_loras).0,
                self.grouped_grids(m, grid_loras)[1].0
            );
        }

        pub fn apply(&self, x2: &Tensor, y2: &Tensor, win: Option<(usize, usize)>) -> Result<()> {
            let Some((m, meta)) = self.dispatch.snapshot()? else {
                return Ok(());
            };
            let (xm, xk) = x2.dims2()?;
            if xm != m {
                bail!(
                    "LoraHook.apply: batch rows {xm} != armed mapping length {m}; \
                     call LoraDispatch::set_mapping with the current token count"
                );
            }
            if xk != self.in_features {
                bail!(
                    "LoraHook.apply: x cols {xk} != in_features {}",
                    self.in_features
                );
            }
            let (win_off, win_len) = win.unwrap_or((0, self.out_features));
            if win_off + win_len > self.out_features {
                bail!("LoraHook.apply: window {win_off}+{win_len} exceeds out_features");
            }
            let (ym, yn) = y2.dims2()?;
            if ym != m || yn != win_len {
                bail!("LoraHook.apply: y dims [{ym},{yn}] != [{m},{win_len}]");
            }
            if y2.dtype() != DType::BF16 {
                bail!("LoraHook.apply: y must be bf16");
            }
            if !y2.is_contiguous() {
                bail!("LoraHook.apply: y must be contiguous");
            }

            let ctx = self.dispatch.context();
            let path = self.plan(m, win)?;

            let x_host = host_bits(&x2.to_dtype(DType::BF16)?.contiguous()?)?;
            let mut y_host = host_bits(y2)?;
            let a_owned: Vec<Vec<u16>> = self
                .a_tensors
                .iter()
                .map(host_bits)
                .collect::<Result<_>>()?;
            let b_owned: Vec<Vec<u16>> = self
                .b_tensors
                .iter()
                .map(host_bits)
                .collect::<Result<_>>()?;
            let a_refs: Vec<&[u16]> = a_owned.iter().map(|v| v.as_slice()).collect();
            let b_refs: Vec<&[u16]> = b_owned.iter().map(|v| v.as_slice()).collect();

            if path == WgpuLoraPath::Grouped {
                wgk::lora_grouped(
                    ctx,
                    &x_host,
                    &a_refs,
                    &b_refs,
                    &mut y_host,
                    &meta,
                    &self.widths,
                    m,
                    self.rank,
                    self.in_features,
                    self.out_features,
                    1.0,
                    None,
                )
                .map_err(|e| anyhow!("lora_grouped: {e}"))?;
            } else {
                wgk::lora_fused(
                    ctx,
                    &x_host,
                    &a_refs,
                    &b_refs,
                    &mut y_host,
                    &meta,
                    &self.widths,
                    m,
                    self.rank,
                    self.in_features,
                    win_off,
                    win_len,
                    win_len,
                    1.0,
                )
                .map_err(|e| anyhow!("lora_fused: {e}"))?;
            }

            let y_new: Vec<bf16> = y_host.into_iter().map(bf16::from_bits).collect();
            let out = Tensor::from_vec(y_new, (m, win_len), y2.device())?;
            y2.slice_set(&out, 0, 0)?;
            Ok(())
        }
    }

    impl crate::linear::LoraDeltaHook for WgpuLoraHook {
        fn in_features(&self) -> usize {
            self.in_features
        }

        fn out_features(&self) -> usize {
            self.out_features
        }

        fn apply(
            &self,
            x2: &Tensor,
            y2: &Tensor,
            win: Option<(usize, usize)>,
        ) -> Result<Option<Tensor>> {
            WgpuLoraHook::apply(self, x2, y2, win)?;
            Ok(None)
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda_runtime {
    use super::LoraSlotStack;
    use anyhow::{anyhow, bail, Result};
    use candle_core::{DType, Device, Tensor};
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
    use half::bf16;
    use nv_kernels::lora::LoraKernelMeta;
    use std::ffi::c_void;
    use std::sync::{Arc, Mutex};

    pub const FUSED_MAX_M: usize = 64;
    pub const FUSED_MAX_RANK: usize = 64;

    struct MetaBufs {
        map: CudaSlice<i32>,
        sorted: CudaSlice<i32>,
        counts: CudaSlice<i32>,
        start: CudaSlice<i32>,
        active: CudaSlice<i32>,
    }

    struct DispatchState {
        m: usize,
        no_lora: bool,
        armed: bool,
    }

    pub struct LoraDispatch {
        device: candle_core::CudaDevice,
        max_tokens: usize,
        max_loras: usize,
        bufs: Mutex<MetaBufs>,
        state: Mutex<DispatchState>,
        map_ptr: u64,
        sorted_ptr: u64,
        counts_ptr: u64,
        start_ptr: u64,
        active_ptr: u64,
    }

    impl LoraDispatch {
        pub fn new(device: &Device, max_tokens: usize, max_loras: usize) -> Result<Arc<Self>> {
            let dev = match device {
                Device::Cuda(d) => d.clone(),
                _ => bail!("LoraDispatch requires a CUDA device"),
            };
            if max_tokens == 0 || max_loras == 0 {
                bail!("LoraDispatch dims must be non-zero");
            }
            let stream = crate::cuda_stream::current_stream(&dev);
            let mut map = stream
                .alloc_zeros::<i32>(max_tokens)
                .map_err(|e| anyhow!(e))?;
            let mut sorted = stream
                .alloc_zeros::<i32>(max_tokens)
                .map_err(|e| anyhow!(e))?;
            let mut counts = stream
                .alloc_zeros::<i32>(max_loras + 1)
                .map_err(|e| anyhow!(e))?;
            let mut start = stream
                .alloc_zeros::<i32>(max_loras + 2)
                .map_err(|e| anyhow!(e))?;
            let mut active = stream
                .alloc_zeros::<i32>(max_loras + 1)
                .map_err(|e| anyhow!(e))?;
            fn rw(s: &mut CudaSlice<i32>, stream: &Arc<cudarc::driver::CudaStream>) -> u64 {
                let (p, g) = s.device_ptr_mut(stream);
                drop(g);
                p as u64
            }
            let map_ptr = rw(&mut map, &stream);
            let sorted_ptr = rw(&mut sorted, &stream);
            let counts_ptr = rw(&mut counts, &stream);
            let start_ptr = rw(&mut start, &stream);
            let active_ptr = rw(&mut active, &stream);
            Ok(Arc::new(Self {
                device: dev,
                max_tokens,
                max_loras,
                bufs: Mutex::new(MetaBufs {
                    map,
                    sorted,
                    counts,
                    start,
                    active,
                }),
                state: Mutex::new(DispatchState {
                    m: 0,
                    no_lora: true,
                    armed: false,
                }),
                map_ptr,
                sorted_ptr,
                counts_ptr,
                start_ptr,
                active_ptr,
            }))
        }

        pub fn max_tokens(&self) -> usize {
            self.max_tokens
        }

        pub fn max_loras(&self) -> usize {
            self.max_loras
        }

        pub fn grid_loras(&self) -> usize {
            self.max_loras + 1
        }

        pub fn set_mapping(&self, mapping: &[i32]) -> Result<()> {
            if mapping.is_empty() {
                bail!("LoraDispatch.set_mapping: empty mapping");
            }
            if mapping.len() > self.max_tokens {
                bail!(
                    "LoraDispatch.set_mapping: {} tokens exceeds max_tokens {}",
                    mapping.len(),
                    self.max_tokens
                );
            }
            for &v in mapping {
                if v < -1 || v >= self.max_loras as i32 {
                    bail!("LoraDispatch.set_mapping: slot {v} out of range");
                }
            }
            let meta = LoraKernelMeta::prepare(mapping, self.max_loras);
            let stream = crate::cuda_stream::current_stream(&self.device);
            {
                let mut bufs = self
                    .bufs
                    .lock()
                    .map_err(|e| anyhow!("LoraDispatch bufs poisoned: {e}"))?;
                stream
                    .memcpy_htod(&meta.token_lora_mapping, &mut bufs.map)
                    .map_err(|e| anyhow!(e))?;
                stream
                    .memcpy_htod(&meta.token_indices_sorted, &mut bufs.sorted)
                    .map_err(|e| anyhow!(e))?;
                stream
                    .memcpy_htod(&meta.num_tokens_per_lora, &mut bufs.counts)
                    .map_err(|e| anyhow!(e))?;
                stream
                    .memcpy_htod(&meta.lora_token_start_loc, &mut bufs.start)
                    .map_err(|e| anyhow!(e))?;
                stream
                    .memcpy_htod(&meta.active_lora_ids, &mut bufs.active)
                    .map_err(|e| anyhow!(e))?;
            }
            let mut st = self
                .state
                .lock()
                .map_err(|e| anyhow!("LoraDispatch state poisoned: {e}"))?;
            st.m = mapping.len();
            st.no_lora = meta.no_lora;
            st.armed = true;
            Ok(())
        }

        pub fn disarm(&self) {
            if let Ok(mut st) = self.state.lock() {
                st.armed = false;
            }
        }

        pub fn armed(&self) -> bool {
            self.state
                .lock()
                .map(|s| s.armed && !s.no_lora)
                .unwrap_or(false)
        }
    }

    pub struct LoraHook {
        dispatch: Arc<LoraDispatch>,
        _a_tensors: Vec<Tensor>,
        _b_tensors: Vec<Tensor>,
        _a_ptrs_d: CudaSlice<u64>,
        _b_ptrs_d: CudaSlice<u64>,
        _slice_n_d: CudaSlice<i32>,
        _slice_start_d: CudaSlice<i32>,
        _b_stride_d: CudaSlice<i64>,
        _buffer: CudaSlice<f32>,
        a_ptrs_ptr: u64,
        b_ptrs_ptr: u64,
        slice_n_ptr: u64,
        slice_start_ptr: u64,
        b_stride_ptr: u64,
        buffer_ptr: u64,
        rank: usize,
        in_features: usize,
        out_features: usize,
        max_n: usize,
        n_slices: usize,
    }

    fn cuda_bf16_ptr(t: &Tensor, stream: &Arc<cudarc::driver::CudaStream>) -> Result<u64> {
        let (storage, layout) = t.storage_and_layout();
        let s = match &*storage {
            candle_core::Storage::Cuda(s) => s,
            _ => bail!("expected cuda storage"),
        };
        let sl = s.as_cuda_slice::<bf16>()?;
        let view = sl.slice(layout.start_offset()..);
        let (p, g) = view.device_ptr(stream);
        drop(g);
        Ok(p as u64)
    }

    impl LoraHook {
        pub fn from_stacks(
            dispatch: Arc<LoraDispatch>,
            stacks: &[&LoraSlotStack],
        ) -> Result<Arc<Self>> {
            if stacks.is_empty() {
                bail!("LoraHook needs at least one slot stack");
            }
            let rank = stacks[0].max_rank();
            let in_features = stacks[0].in_features();
            if rank > FUSED_MAX_RANK {
                bail!("LoraHook: max_rank {rank} exceeds kernel limit {FUSED_MAX_RANK}");
            }
            let mut widths = Vec::with_capacity(stacks.len());
            for st in stacks {
                if st.max_rank() != rank || st.in_features() != in_features {
                    bail!("LoraHook: slot stacks must share max_rank and in_features");
                }
                if st.max_loras() != dispatch.max_loras() {
                    bail!(
                        "LoraHook: stack max_loras {} != dispatch max_loras {}",
                        st.max_loras(),
                        dispatch.max_loras()
                    );
                }
                if st.lora_a_stacked().dtype() != DType::BF16 {
                    bail!("LoraHook: slot stacks must be bf16");
                }
                widths.push(st.out_features());
            }
            let stream = crate::cuda_stream::current_stream(&dispatch.device);
            let mut a_tensors = Vec::with_capacity(stacks.len());
            let mut b_tensors = Vec::with_capacity(stacks.len());
            let mut a_addrs = Vec::with_capacity(stacks.len());
            let mut b_addrs = Vec::with_capacity(stacks.len());
            for st in stacks {
                let a = st.lora_a_stacked().clone();
                let b = st.lora_b_stacked().clone();
                a_addrs.push(cuda_bf16_ptr(&a, &stream)?);
                b_addrs.push(cuda_bf16_ptr(&b, &stream)?);
                a_tensors.push(a);
                b_tensors.push(b);
            }
            let out_features: usize = widths.iter().sum();
            let max_n = *widths.iter().max().unwrap();
            let n_slices = widths.len();
            let slice_n: Vec<i32> = widths.iter().map(|&w| w as i32).collect();
            let mut acc = 0i32;
            let slice_start: Vec<i32> = widths
                .iter()
                .map(|&w| {
                    let s = acc;
                    acc += w as i32;
                    s
                })
                .collect();
            let b_stride: Vec<i64> = widths.iter().map(|&w| (w * rank) as i64).collect();

            #[allow(deprecated)]
            let a_ptrs_d = stream.clone_htod(&a_addrs).map_err(|e| anyhow!(e))?;
            #[allow(deprecated)]
            let b_ptrs_d = stream.clone_htod(&b_addrs).map_err(|e| anyhow!(e))?;
            #[allow(deprecated)]
            let slice_n_d = stream.clone_htod(&slice_n).map_err(|e| anyhow!(e))?;
            #[allow(deprecated)]
            let slice_start_d = stream.clone_htod(&slice_start).map_err(|e| anyhow!(e))?;
            #[allow(deprecated)]
            let b_stride_d = stream.clone_htod(&b_stride).map_err(|e| anyhow!(e))?;
            let mut buffer = stream
                .alloc_zeros::<f32>(n_slices * dispatch.max_tokens() * rank)
                .map_err(|e| anyhow!(e))?;

            fn ro<T>(s: &CudaSlice<T>, stream: &Arc<cudarc::driver::CudaStream>) -> u64 {
                let (p, g) = s.device_ptr(stream);
                drop(g);
                p as u64
            }
            let a_ptrs_ptr = ro(&a_ptrs_d, &stream);
            let b_ptrs_ptr = ro(&b_ptrs_d, &stream);
            let slice_n_ptr = ro(&slice_n_d, &stream);
            let slice_start_ptr = ro(&slice_start_d, &stream);
            let b_stride_ptr = ro(&b_stride_d, &stream);
            let buffer_ptr = {
                let (p, g) = buffer.device_ptr_mut(&stream);
                drop(g);
                p as u64
            };

            Ok(Arc::new(Self {
                dispatch,
                _a_tensors: a_tensors,
                _b_tensors: b_tensors,
                _a_ptrs_d: a_ptrs_d,
                _b_ptrs_d: b_ptrs_d,
                _slice_n_d: slice_n_d,
                _slice_start_d: slice_start_d,
                _b_stride_d: b_stride_d,
                _buffer: buffer,
                a_ptrs_ptr,
                b_ptrs_ptr,
                slice_n_ptr,
                slice_start_ptr,
                b_stride_ptr,
                buffer_ptr,
                rank,
                in_features,
                out_features,
                max_n,
                n_slices,
            }))
        }

        pub fn in_features(&self) -> usize {
            self.in_features
        }

        pub fn out_features(&self) -> usize {
            self.out_features
        }

        pub fn dispatch(&self) -> &Arc<LoraDispatch> {
            &self.dispatch
        }

        pub fn apply(&self, x2: &Tensor, y2: &Tensor, win: Option<(usize, usize)>) -> Result<()> {
            let (m, no_lora) = {
                let st = self
                    .dispatch
                    .state
                    .lock()
                    .map_err(|e| anyhow!("LoraDispatch state poisoned: {e}"))?;
                if !st.armed || st.no_lora {
                    return Ok(());
                }
                (st.m, st.no_lora)
            };
            let _ = no_lora;
            let (xm, xk) = x2.dims2()?;
            if xm != m {
                bail!(
                    "LoraHook.apply: batch rows {xm} != armed mapping length {m}; \
                     call LoraDispatch::set_mapping with the current token count"
                );
            }
            if xk != self.in_features {
                bail!(
                    "LoraHook.apply: x cols {xk} != in_features {}",
                    self.in_features
                );
            }
            let (win_off, win_len) = win.unwrap_or((0, self.out_features));
            if win_off + win_len > self.out_features {
                bail!("LoraHook.apply: window {win_off}+{win_len} exceeds out_features");
            }
            let (ym, yn) = y2.dims2()?;
            if ym != m || yn != win_len {
                bail!("LoraHook.apply: y dims [{ym},{yn}] != [{m},{win_len}]");
            }
            if y2.dtype() != DType::BF16 {
                bail!("LoraHook.apply: y must be bf16");
            }
            if !matches!(x2.device(), Device::Cuda(_)) {
                bail!("LoraHook.apply: armed LoRA requires CUDA input");
            }

            let x_bf = x2.to_dtype(DType::BF16)?.contiguous()?;
            let stream = crate::cuda_stream::current_stream(&self.dispatch.device);
            let x_ptr = cuda_bf16_ptr(&x_bf, &stream)?;
            let y_ptr = cuda_bf16_ptr(y2, &stream)?;
            let d = &self.dispatch;
            let full = win_off == 0 && win_len == self.out_features;

            if full && m > FUSED_MAX_M {
                let rc = unsafe {
                    nv_kernels::lora::lora_shrink(
                        stream.cu_stream() as *mut c_void,
                        x_ptr as *const u16,
                        self.a_ptrs_ptr as *const u64,
                        self.buffer_ptr as *mut f32,
                        d.map_ptr as *const i32,
                        d.sorted_ptr as *const i32,
                        d.counts_ptr as *const i32,
                        d.start_ptr as *const i32,
                        d.active_ptr as *const i32,
                        m as i32,
                        self.rank as i32,
                        self.in_features as i32,
                        self.n_slices as i32,
                        d.grid_loras() as i32,
                        (self.rank * self.in_features) as i64,
                        1.0,
                    )
                };
                if rc != 0 {
                    bail!("lora_shrink rc={rc}");
                }
                let rc = unsafe {
                    nv_kernels::lora::lora_expand(
                        stream.cu_stream() as *mut c_void,
                        self.buffer_ptr as *const f32,
                        self.b_ptrs_ptr as *const u64,
                        y_ptr as *mut u16,
                        d.map_ptr as *const i32,
                        d.sorted_ptr as *const i32,
                        d.counts_ptr as *const i32,
                        d.start_ptr as *const i32,
                        d.active_ptr as *const i32,
                        self.slice_n_ptr as *const i32,
                        self.slice_start_ptr as *const i32,
                        m as i32,
                        self.rank as i32,
                        self.max_n as i32,
                        self.n_slices as i32,
                        d.grid_loras() as i32,
                        self.out_features as i32,
                    )
                };
                if rc != 0 {
                    bail!("lora_expand rc={rc}");
                }
            } else {
                let rc = unsafe {
                    nv_kernels::lora::lora_fused(
                        stream.cu_stream() as *mut c_void,
                        x_ptr as *const u16,
                        self.a_ptrs_ptr as *const u64,
                        self.b_ptrs_ptr as *const u64,
                        y_ptr as *mut u16,
                        d.sorted_ptr as *const i32,
                        d.counts_ptr as *const i32,
                        d.start_ptr as *const i32,
                        d.active_ptr as *const i32,
                        self.slice_n_ptr as *const i32,
                        self.slice_start_ptr as *const i32,
                        self.b_stride_ptr as *const i64,
                        m as i32,
                        self.rank as i32,
                        self.in_features as i32,
                        self.max_n as i32,
                        self.n_slices as i32,
                        d.grid_loras() as i32,
                        (self.rank * self.in_features) as i64,
                        win_off as i32,
                        win_len as i32,
                        win_len as i32,
                        1.0,
                    )
                };
                if rc != 0 {
                    bail!("lora_fused rc={rc}");
                }
            }
            Ok(())
        }
    }
}

pub struct LoraSlotManager {
    stacks: Vec<(String, LoraSlotStack)>,
    slot_ids: Vec<Option<u64>>,
    lru: Vec<u64>,
    max_loras: usize,
}

impl LoraSlotManager {
    pub fn new(
        max_loras: usize,
        max_rank: usize,
        specs: &[LoraModuleSpec],
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if specs.is_empty() {
            bail!("lora slot manager needs at least one module spec");
        }
        let mut stacks = Vec::with_capacity(specs.len());
        for spec in specs {
            let stack = LoraSlotStack::new(
                max_loras,
                max_rank,
                spec.in_features,
                spec.out_features,
                dtype,
                device,
            )?;
            stacks.push((spec.name.clone(), stack));
        }
        Ok(Self {
            stacks,
            slot_ids: vec![None; max_loras],
            lru: Vec::with_capacity(max_loras),
            max_loras,
        })
    }

    pub fn max_loras(&self) -> usize {
        self.max_loras
    }

    pub fn slot_of(&self, id: u64) -> Option<usize> {
        self.slot_ids.iter().position(|s| *s == Some(id))
    }

    pub fn slot_id(&self, slot: usize) -> Option<u64> {
        self.slot_ids.get(slot).copied().flatten()
    }

    pub fn stack(&self, name: &str) -> Option<&LoraSlotStack> {
        self.stacks.iter().find(|(n, _)| n == name).map(|(_, s)| s)
    }

    pub fn module_names(&self) -> Vec<&str> {
        self.stacks.iter().map(|(n, _)| n.as_str()).collect()
    }

    pub fn activate(&mut self, id: u64, adapter: &LoraAdapter) -> Result<usize> {
        if let Some(slot) = self.slot_of(id) {
            self.touch(id);
            return Ok(slot);
        }
        let slot = match self.slot_ids.iter().position(|s| s.is_none()) {
            Some(free) => free,
            None => {
                let Some(oldest) = self.lru.first().copied() else {
                    bail!("no free lora slots and no evictable adapter");
                };
                let Some(slot) = self.slot_of(oldest) else {
                    bail!("lru entry {oldest} has no slot");
                };
                self.slot_ids[slot] = None;
                self.lru.retain(|&x| x != oldest);
                slot
            }
        };
        for (name, stack) in &self.stacks {
            match adapter.modules.get(name) {
                Some(w) => stack.set_lora(slot, &w.a, &w.b)?,
                None => stack.reset_lora(slot)?,
            }
        }
        self.slot_ids[slot] = Some(id);
        self.lru.push(id);
        Ok(slot)
    }

    pub fn deactivate(&mut self, id: u64) -> Option<usize> {
        let slot = self.slot_of(id)?;
        self.slot_ids[slot] = None;
        self.lru.retain(|&x| x != id);
        Some(slot)
    }

    fn touch(&mut self, id: u64) {
        self.lru.retain(|&x| x != id);
        self.lru.push(id);
    }
}
