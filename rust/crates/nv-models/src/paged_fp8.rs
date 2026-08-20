use anyhow::{bail, Result};

#[cfg(feature = "cuda")]
use crate::gemma4::Gemma4Cache;
use crate::gemma4::Gemma4Config;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerKvGeometry {
    pub n_kv: usize,
    pub head_dim: usize,
}

impl LayerKvGeometry {
    pub fn kv_elems(&self) -> usize {
        self.n_kv * self.head_dim
    }
    pub fn scale_elems(&self) -> usize {
        self.n_kv
    }
}

#[derive(Clone, Debug)]
pub struct PagedPoolConfig {
    pub num_blocks: usize,
    pub block_size: usize,
    pub layers: Vec<LayerKvGeometry>,

    pub layer_blocks: Vec<usize>,

    pub layer_sliding: Vec<bool>,

    pub lanes: usize,

    pub sliding_ring_blocks: usize,
}

impl PagedPoolConfig {
    pub fn from_gemma4(config: &Gemma4Config, num_blocks: usize, block_size: usize) -> Self {
        let layers: Vec<LayerKvGeometry> = config
            .layer_types
            .iter()
            .map(|kind| LayerKvGeometry {
                n_kv: config.num_kv_heads_for(*kind),
                head_dim: config.head_dim_for(*kind),
            })
            .collect();
        let n = layers.len();
        let layer_blocks = vec![num_blocks; n];
        Self {
            num_blocks,
            block_size,
            layers,
            layer_blocks,
            layer_sliding: vec![false; n],
            lanes: 0,
            sliding_ring_blocks: 0,
        }
    }

    pub fn from_gemma4_e4b(config: &Gemma4Config, num_blocks: usize, block_size: usize) -> Self {
        let mut cfg = Self::from_gemma4(config, num_blocks, block_size);
        for (i, blocks) in cfg.layer_blocks.iter_mut().enumerate() {
            if config.kv_source_layer(i).is_some() {
                *blocks = 1;
            }
        }
        cfg
    }

    pub fn from_gemma4_hybrid(
        config: &Gemma4Config,
        full_blocks: usize,
        block_size: usize,
        lanes: usize,
    ) -> Self {
        let ring_slots = config.sliding_window + crate::gemma4::VERIFY_PREFILL_CHUNK + 128;
        let ring_blocks = ring_slots.div_ceil(block_size);
        let layers: Vec<LayerKvGeometry> = config
            .layer_types
            .iter()
            .map(|kind| LayerKvGeometry {
                n_kv: config.num_kv_heads_for(*kind),
                head_dim: config.head_dim_for(*kind),
            })
            .collect();
        let layer_sliding: Vec<bool> = config
            .layer_types
            .iter()
            .map(|kind| matches!(kind, crate::gemma4::LayerType::SlidingAttention))
            .collect();
        let layer_blocks = layer_sliding
            .iter()
            .map(|s| {
                if *s {
                    lanes.max(1) * ring_blocks
                } else {
                    full_blocks
                }
            })
            .collect();
        Self {
            num_blocks: full_blocks,
            block_size,
            layers,
            layer_blocks,
            layer_sliding,
            lanes: lanes.max(1),
            sliding_ring_blocks: ring_blocks,
        }
    }

    pub fn layer_slots(&self, layer: usize) -> usize {
        self.layer_blocks[layer] * self.block_size
    }

    pub fn max_pool_slots(&self) -> usize {
        self.layer_blocks.iter().copied().max().unwrap_or(0) * self.block_size
    }

    pub fn num_pool_slots(&self) -> usize {
        self.max_pool_slots()
    }

    fn layer_block_bytes(&self, g: &LayerKvGeometry) -> usize {
        let slots = self.block_size;
        let kv = 2 * slots * g.kv_elems();
        let sc = 2 * slots * g.scale_elems() * std::mem::size_of::<f32>();
        kv + sc
    }

    pub fn bytes_per_block(&self) -> usize {
        self.layers.iter().map(|g| self.layer_block_bytes(g)).sum()
    }

    pub fn pool_bytes(&self) -> usize {
        self.layers
            .iter()
            .enumerate()
            .map(|(i, g)| self.layer_blocks[i] * self.layer_block_bytes(g))
            .sum()
    }

    pub fn max_blocks_for_budget(&self, budget_bytes: usize) -> usize {
        budget_bytes
            .checked_div(self.bytes_per_block())
            .unwrap_or(0)
    }
}

#[inline]
pub fn physical_slot(block_table: &[u32], block_size: usize, logical: usize) -> Result<usize> {
    let blk = logical / block_size;
    let off = logical % block_size;
    let Some(&phys_block) = block_table.get(blk) else {
        bail!(
            "physical_slot: logical position {logical} -> block {blk} out of block_table len {}",
            block_table.len()
        );
    };
    Ok(phys_block as usize * block_size + off)
}

#[cfg(not(feature = "cuda"))]
pub use host::PagedKvFp8Pool;

#[cfg(feature = "cuda")]
pub use cuda::PagedKvFp8Pool;

#[cfg(feature = "cuda")]
pub use cuda::PagedGemma4Cache;

#[cfg(feature = "cuda")]
pub use cuda::{derive_v_enabled, DeriveVPlan, DERIVE_V_ENV};

#[cfg(feature = "cuda")]
pub use cuda::{paged_attn_fp8_ring_enabled, PAGED_ATTN_FP8_RING_ENV};

#[cfg(feature = "cuda")]
pub const PREFILL_MK_TILE_IS_THE_KERNELS_KMAXM_OF_8: usize = 8;

pub const PREFILL_MK_MAX_HEAD_DIM_IS_THE_KERNELS_KMAXHDMK_OF_256: usize = 256;

#[cfg(feature = "cuda")]
pub fn flash_scratch_elems_for(n_q: usize, head_dim: usize) -> usize {
    nv_kernels::cuda::flash_splitk_scratch_elems(n_q as i32, head_dim as i32)
}

#[cfg(not(feature = "cuda"))]
mod host {
    use super::*;

    const REF_QMAX: f32 = 127.0;

    pub struct PagedKvFp8Pool {
        cfg: PagedPoolConfig,
        k_fp8: Vec<Vec<u8>>,
        v_fp8: Vec<Vec<u8>>,
        k_scale: Vec<Vec<f32>>,
        v_scale: Vec<Vec<f32>>,
    }

    impl PagedKvFp8Pool {
        pub fn new(cfg: PagedPoolConfig) -> Result<Self> {
            let mut k_fp8 = Vec::with_capacity(cfg.layers.len());
            let mut v_fp8 = Vec::with_capacity(cfg.layers.len());
            let mut k_scale = Vec::with_capacity(cfg.layers.len());
            let mut v_scale = Vec::with_capacity(cfg.layers.len());
            for (i, g) in cfg.layers.iter().enumerate() {
                let slots = cfg.layer_slots(i);
                k_fp8.push(vec![0u8; slots * g.kv_elems()]);
                v_fp8.push(vec![0u8; slots * g.kv_elems()]);
                k_scale.push(vec![0f32; slots * g.scale_elems()]);
                v_scale.push(vec![0f32; slots * g.scale_elems()]);
            }
            Ok(Self {
                cfg,
                k_fp8,
                v_fp8,
                k_scale,
                v_scale,
            })
        }

        pub fn config(&self) -> &PagedPoolConfig {
            &self.cfg
        }
        pub fn pool_bytes(&self) -> usize {
            self.cfg.pool_bytes()
        }

        fn quantize_into(
            fp8: &mut [u8],
            scale: &mut [f32],
            src: &[f32],
            slot: usize,
            n_kv: usize,
            head_dim: usize,
        ) {
            for h in 0..n_kv {
                let base_src = h * head_dim;
                let amax = (0..head_dim)
                    .map(|d| src[base_src + d].abs())
                    .fold(0.0f32, f32::max);
                let s = if amax > 0.0 { amax / REF_QMAX } else { 1.0 };
                let inv = if amax > 0.0 { REF_QMAX / amax } else { 1.0 };
                scale[slot * n_kv + h] = s;
                let base_dst = (slot * n_kv + h) * head_dim;
                for d in 0..head_dim {
                    let q = (src[base_src + d] * inv).clamp(-REF_QMAX, REF_QMAX);
                    fp8[base_dst + d] = (q.round() as i32 as i8) as u8;
                }
            }
        }

        pub fn append_layer(
            &mut self,
            layer: usize,
            block_table: &[u32],
            start_logical: usize,
            k_new: &[f32],
            v_new: &[f32],
            n_tokens: usize,
        ) -> Result<()> {
            let g = self.cfg.layers[layer];
            let (n_kv, head_dim) = (g.n_kv, g.head_dim);
            let bs = self.cfg.block_size;
            for t in 0..n_tokens {
                let slot = physical_slot(block_table, bs, start_logical + t)?;
                let row = t * n_kv * head_dim;
                Self::quantize_into(
                    &mut self.k_fp8[layer],
                    &mut self.k_scale[layer],
                    &k_new[row..row + n_kv * head_dim],
                    slot,
                    n_kv,
                    head_dim,
                );
                Self::quantize_into(
                    &mut self.v_fp8[layer],
                    &mut self.v_scale[layer],
                    &v_new[row..row + n_kv * head_dim],
                    slot,
                    n_kv,
                    head_dim,
                );
            }
            Ok(())
        }

        pub fn gather_layer(
            &self,
            layer: usize,
            block_table: &[u32],
            len: usize,
        ) -> Result<(Vec<f32>, Vec<f32>)> {
            let g = self.cfg.layers[layer];
            let (n_kv, head_dim) = (g.n_kv, g.head_dim);
            let bs = self.cfg.block_size;
            let mut k_out = vec![0f32; len * n_kv * head_dim];
            let mut v_out = vec![0f32; len * n_kv * head_dim];
            for t in 0..len {
                let slot = physical_slot(block_table, bs, t)?;
                for h in 0..n_kv {
                    let ks = self.k_scale[layer][slot * n_kv + h];
                    let vs = self.v_scale[layer][slot * n_kv + h];
                    let src = (slot * n_kv + h) * head_dim;
                    let dst = (t * n_kv + h) * head_dim;
                    for d in 0..head_dim {
                        k_out[dst + d] = (self.k_fp8[layer][src + d] as i8 as f32) * ks;
                        v_out[dst + d] = (self.v_fp8[layer][src + d] as i8 as f32) * vs;
                    }
                }
            }
            Ok((k_out, v_out))
        }

        pub fn copy_block(&mut self, src_block: u32, dst_block: u32) -> Result<()> {
            if src_block == dst_block {
                return Ok(());
            }
            let bs = self.cfg.block_size;
            for layer in 0..self.cfg.layers.len() {
                let g = self.cfg.layers[layer];
                let kv = g.kv_elems();
                let sc = g.scale_elems();
                for s in 0..bs {
                    let src_slot = src_block as usize * bs + s;
                    let dst_slot = dst_block as usize * bs + s;
                    let (ks, kd) = (src_slot * kv, dst_slot * kv);
                    self.k_fp8[layer].copy_within(ks..ks + kv, kd);
                    self.v_fp8[layer].copy_within(ks..ks + kv, kd);
                    let (ss, sd) = (src_slot * sc, dst_slot * sc);
                    self.k_scale[layer].copy_within(ss..ss + sc, sd);
                    self.v_scale[layer].copy_within(ss..ss + sc, sd);
                }
            }
            Ok(())
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use anyhow::anyhow;
    use candle_core::{CudaDevice, Device, Tensor};
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use half::bf16;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    pub const DERIVE_V_ENV: &str = "NV_KV_DERIVE_V";

    const DERIVE_HEAD_DIM: usize = 512;

    const DERIVE_MAX_ROPE_ANGLES: usize = 128;

    const DERIVE_MIN_W_K: f32 = 1e-2;

    const DERIVE_ANGLE_MODE_F64: i32 = 2;

    pub fn derive_v_enabled() -> bool {
        matches!(std::env::var(DERIVE_V_ENV).as_deref(), Ok("1"))
    }

    #[derive(Clone, Debug)]
    pub struct DeriveVPlan {
        w_inv: Vec<Option<f32>>,
        inv_freq: Vec<f32>,
        rope_angles: usize,
    }

    impl DeriveVPlan {
        pub fn new(
            config: &Gemma4Config,
            pool: &PagedPoolConfig,
            w_k: &[Option<f32>],
        ) -> Result<Self> {
            let n = pool.layers.len();
            if w_k.len() != n {
                bail!("DeriveVPlan: {n} pool layers but {} w_k entries", w_k.len());
            }
            let kind = crate::gemma4::LayerType::FullAttention;
            let head_dim = config.head_dim_for(kind);
            let half = head_dim / 2;
            let base = config.rope_theta_for(kind);
            let rope_angles =
                ((config.rope_partial_factor_for(kind) * head_dim as f32 / 2.0) as usize).min(half);
            let mut inv_freq = vec![0f32; half];
            for (i, f) in inv_freq.iter_mut().enumerate().take(rope_angles) {
                *f = 1.0 / base.powf((i as f32 * 2.0) / (head_dim as f32));
            }
            let capable = config.attention_k_eq_v
                && head_dim == DERIVE_HEAD_DIM
                && (1..=DERIVE_MAX_ROPE_ANGLES).contains(&rope_angles);
            let w_inv = (0..n)
                .map(|i| {
                    if !capable
                        || pool.layer_sliding.get(i).copied().unwrap_or(false)
                        || config.layer_types.get(i) != Some(&kind)
                        || pool.layers[i].head_dim != head_dim
                    {
                        return None;
                    }
                    match w_k[i] {
                        Some(w) if w.is_finite() && w > DERIVE_MIN_W_K => Some(1.0 / w),
                        _ => None,
                    }
                })
                .collect();
            Ok(Self {
                w_inv,
                inv_freq,
                rope_angles,
            })
        }

        pub fn from_model(model: &crate::gemma4::Gemma4, pool: &PagedPoolConfig) -> Result<Self> {
            let mut w_k = vec![None; pool.layers.len()];
            for (i, slot) in w_k.iter_mut().enumerate() {
                let Some(layer) = model.layers().get(i) else {
                    continue;
                };
                if layer.self_attn.has_v {
                    continue;
                }
                *slot = scalar_norm_weight(&layer.self_attn.k_norm)?;
            }
            Self::new(model.config(), pool, &w_k)
        }

        pub fn layer_count(&self) -> usize {
            self.w_inv.iter().filter(|w| w.is_some()).count()
        }
        pub fn w_inv(&self, layer: usize) -> Option<f32> {
            self.w_inv.get(layer).copied().flatten()
        }
        pub fn rope_angles(&self) -> usize {
            self.rope_angles
        }
    }

    fn scalar_norm_weight(norm: &nv_layers::RmsNorm) -> Result<Option<f32>> {
        let w: Vec<f32> = norm
            .weight_bf16()
            .flatten_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1()?;
        let lo = w.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        Ok((lo == hi).then_some(lo))
    }

    struct LayerSlabs {
        k_fp8: CudaSlice<u8>,
        k_scale: CudaSlice<f32>,
        v_fp8: Option<CudaSlice<u8>>,
        v_scale: Option<CudaSlice<f32>>,
    }

    struct DeriveState {
        w_inv: Vec<Option<f32>>,
        inv_freq: CudaSlice<f32>,
        rope_angles: i32,
        dispatches: AtomicU64,
    }

    pub struct PagedKvFp8Pool {
        cfg: PagedPoolConfig,
        layers: Vec<LayerSlabs>,
        device: Device,
        lane_free: Vec<u32>,
        derive: Option<DeriveState>,
    }

    impl PagedKvFp8Pool {
        pub fn new(cfg: PagedPoolConfig, device: &Device) -> Result<Self> {
            Self::build(cfg, device, None)
        }

        pub fn new_derive_v(
            cfg: PagedPoolConfig,
            device: &Device,
            plan: &DeriveVPlan,
        ) -> Result<Self> {
            Self::build(cfg, device, Some(plan))
        }

        fn build(
            cfg: PagedPoolConfig,
            device: &Device,
            plan: Option<&DeriveVPlan>,
        ) -> Result<Self> {
            let dev = match device {
                Device::Cuda(d) => d.clone(),
                _ => bail!("PagedKvFp8Pool requires a CUDA device"),
            };
            let plan = plan.filter(|_| derive_v_enabled());
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let mut layers = Vec::with_capacity(cfg.layers.len());
            for (i, g) in cfg.layers.iter().enumerate() {
                let slots = cfg.layer_slots(i);
                let kv = slots * g.kv_elems();
                let sc = slots * g.scale_elems();
                let derived = plan.and_then(|p| p.w_inv(i)).is_some();
                let (v_fp8, v_scale) = if derived {
                    (None, None)
                } else {
                    (
                        Some(stream.alloc_zeros::<u8>(kv).map_err(|e| anyhow!(e))?),
                        Some(stream.alloc_zeros::<f32>(sc).map_err(|e| anyhow!(e))?),
                    )
                };
                layers.push(LayerSlabs {
                    k_fp8: stream.alloc_zeros::<u8>(kv).map_err(|e| anyhow!(e))?,
                    k_scale: stream.alloc_zeros::<f32>(sc).map_err(|e| anyhow!(e))?,
                    v_fp8,
                    v_scale,
                });
            }
            let derive = match plan {
                Some(p) if p.layer_count() > 0 => Some(DeriveState {
                    w_inv: p.w_inv.clone(),
                    #[allow(deprecated)]
                    inv_freq: stream.memcpy_stod(&p.inv_freq).map_err(|e| anyhow!(e))?,
                    rope_angles: p.rope_angles as i32,
                    dispatches: AtomicU64::new(0),
                }),
                _ => None,
            };
            let lane_free: Vec<u32> = (0..cfg.lanes as u32).rev().collect();
            Ok(Self {
                cfg,
                layers,
                device: device.clone(),
                lane_free,
                derive,
            })
        }

        fn derive_w_inv(&self, layer: usize) -> Option<f32> {
            self.derive
                .as_ref()
                .and_then(|d| d.w_inv.get(layer).copied().flatten())
        }

        pub fn derive_layers(&self) -> usize {
            self.derive
                .as_ref()
                .map_or(0, |d| d.w_inv.iter().filter(|w| w.is_some()).count())
        }

        pub fn derive_dispatches(&self) -> u64 {
            self.derive
                .as_ref()
                .map_or(0, |d| d.dispatches.load(Ordering::Relaxed))
        }

        pub fn config(&self) -> &PagedPoolConfig {
            &self.cfg
        }

        pub fn pool_bytes(&self) -> usize {
            let mut total = 0usize;
            for (i, g) in self.cfg.layers.iter().enumerate() {
                let slots = self.cfg.layer_slots(i);
                let one = slots * g.kv_elems() + slots * g.scale_elems() * 4;
                total += if self.layers[i].v_fp8.is_some() {
                    2 * one
                } else {
                    one
                };
            }
            total
        }
        pub fn device(&self) -> &Device {
            &self.device
        }

        pub fn acquire_lane(&mut self) -> Option<u32> {
            self.lane_free.pop()
        }
        pub fn release_lane(&mut self, lane: u32) {
            if (lane as usize) < self.cfg.lanes && !self.lane_free.contains(&lane) {
                self.lane_free.push(lane);
            }
        }

        fn append_one_kv(
            stream: &CudaStream,
            src: &Tensor,
            fp8: &mut CudaSlice<u8>,
            scale: &mut CudaSlice<f32>,
            start_dev: &CudaSlice<i32>,
            block_table_dev: &CudaSlice<i32>,
            block_size: usize,
            n_tokens: usize,
            n_kv: usize,
            head_dim: usize,
        ) -> Result<()> {
            let src_c = src.contiguous()?;
            let (storage, layout) = src_c.storage_and_layout();
            let cuda = match &*storage {
                candle_core::Storage::Cuda(s) => s,
                _ => bail!("paged append: source must be on CUDA"),
            };

            let slice = cuda.as_cuda_slice::<bf16>()?;
            let off = layout.start_offset();
            let view = slice.slice(off..);
            let (src_ptr, _g0) = view.device_ptr(stream);
            let (start_ptr, _g1) = start_dev.device_ptr(stream);
            let (bt_ptr, _g2) = block_table_dev.device_ptr(stream);
            let (fp8_ptr, _g3) = fp8.device_ptr_mut(stream);
            let (sc_ptr, _g4) = scale.device_ptr_mut(stream);
            let s_raw = stream.cu_stream() as *mut c_void;
            let rc = unsafe {
                nv_kernels::cuda::quantize_kv_fp8_paged(
                    s_raw,
                    src_ptr as *const u16,
                    fp8_ptr as *mut u8,
                    sc_ptr as *mut f32,
                    start_ptr as *const i32,
                    bt_ptr as *const i32,
                    block_size as i32,
                    n_tokens as i32,
                    n_kv as i32,
                    head_dim as i32,
                )
            };
            if rc != 0 {
                bail!("quantize_kv_fp8_paged rc={rc}");
            }
            Ok(())
        }

        pub fn append_layer(
            &mut self,
            layer: usize,
            k_new: &Tensor,
            v_new: &Tensor,
            n_tokens: usize,
            start_dev: &CudaSlice<i32>,
            block_table_dev: &CudaSlice<i32>,
        ) -> Result<()> {
            let g = self.cfg.layers[layer];
            let bs = self.cfg.block_size;
            let dev = match self.device.clone() {
                Device::Cuda(d) => d,
                _ => unreachable!(),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let slab = &mut self.layers[layer];
            Self::append_one_kv(
                &stream,
                k_new,
                &mut slab.k_fp8,
                &mut slab.k_scale,
                start_dev,
                block_table_dev,
                bs,
                n_tokens,
                g.n_kv,
                g.head_dim,
            )?;
            if let (Some(fp8), Some(scale)) = (slab.v_fp8.as_mut(), slab.v_scale.as_mut()) {
                Self::append_one_kv(
                    &stream,
                    v_new,
                    fp8,
                    scale,
                    start_dev,
                    block_table_dev,
                    bs,
                    n_tokens,
                    g.n_kv,
                    g.head_dim,
                )?;
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn derive_one_v(
            &self,
            layer: usize,
            stream: &Arc<CudaStream>,
            block_table_dev: &CudaSlice<i32>,
            len: usize,
            out: &mut CudaSlice<bf16>,
        ) -> Result<()> {
            let g = self.cfg.layers[layer];
            let d = self
                .derive
                .as_ref()
                .ok_or_else(|| anyhow!("derive_one_v layer {layer}: no derive state"))?;
            let w_inv = d
                .w_inv
                .get(layer)
                .copied()
                .flatten()
                .ok_or_else(|| anyhow!("derive_one_v layer {layer}: not a derive layer"))?;
            let slab = &self.layers[layer];
            let (k_ptr, _g0) = slab.k_fp8.device_ptr(stream);
            let (ks_ptr, _g1) = slab.k_scale.device_ptr(stream);
            let (inv_ptr, _g2) = d.inv_freq.device_ptr(stream);
            let (bt_ptr, _g3) = block_table_dev.device_ptr(stream);
            let (out_ptr, _g4) = out.device_ptr_mut(stream);
            let rc = unsafe {
                nv_kernels::cuda::derive_v_from_k_fp8_paged(
                    stream.cu_stream() as *mut c_void,
                    k_ptr as *const u8,
                    ks_ptr as *const f32,
                    std::ptr::null(),
                    std::ptr::null(),
                    inv_ptr as *const f32,
                    out_ptr as *mut u16,
                    bt_ptr as *const i32,
                    self.cfg.block_size as i32,
                    len as i32,
                    g.n_kv as i32,
                    g.head_dim as i32,
                    d.rope_angles,
                    DERIVE_ANGLE_MODE_F64,
                    0,
                    w_inv,
                )
            };
            if rc != 0 {
                bail!("derive_v_from_k_fp8_paged layer {layer} rc={rc}");
            }
            d.dispatches.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn gather_one_kv(
            stream: &std::sync::Arc<CudaStream>,
            fp8: &CudaSlice<u8>,
            scale: &CudaSlice<f32>,
            block_table_dev: &CudaSlice<i32>,
            block_size: usize,
            len: usize,
            n_kv: usize,
            head_dim: usize,
            dev: &CudaDevice,
        ) -> Result<Tensor> {
            let need = len * n_kv * head_dim;
            let mut out = stream.alloc_zeros::<bf16>(need).map_err(|e| anyhow!(e))?;
            {
                let (fp8_ptr, _g0) = fp8.device_ptr(stream);
                let (sc_ptr, _g1) = scale.device_ptr(stream);
                let (bt_ptr, _g2) = block_table_dev.device_ptr(stream);
                let (out_ptr, _g3) = out.device_ptr_mut(stream);
                let s_raw = stream.cu_stream() as *mut c_void;
                let rc = unsafe {
                    nv_kernels::cuda::dequantize_kv_fp8_paged(
                        s_raw,
                        fp8_ptr as *const u8,
                        sc_ptr as *const f32,
                        out_ptr as *mut u16,
                        bt_ptr as *const i32,
                        block_size as i32,
                        len as i32,
                        n_kv as i32,
                        head_dim as i32,
                    )
                };
                if rc != 0 {
                    bail!("dequantize_kv_fp8_paged rc={rc}");
                }
            }
            let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
            let t = Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, len, n_kv, head_dim),
                candle_core::op::BackpropOp::none(),
                false,
            );
            Ok(t)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn decode_attention_paged(
            &self,
            layer: usize,
            q: &Tensor,
            n_q: usize,
            block_table_dev: &CudaSlice<i32>,
            block_size: usize,
            n_total_dev: &CudaSlice<i32>,
            scratch: &mut CudaSlice<f32>,
            fan_in: &mut CudaSlice<u32>,
            sliding_window: Option<usize>,
            scaling: f32,
        ) -> Result<Tensor> {
            let g = self.cfg.layers[layer];
            let head_dim = g.head_dim;
            let dev = match self.device.clone() {
                Device::Cuda(d) => d,
                _ => unreachable!(),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);

            let expected = n_q * head_dim;
            let got: usize = q.dims().iter().product();
            if got != expected {
                bail!(
                    "decode_attention_paged layer {layer}: expected {expected} q elems, got {got}"
                );
            }
            let q_c = q.contiguous()?;
            let (q_storage, q_l) = q_c.storage_and_layout();
            let q_cuda = match &*q_storage {
                candle_core::Storage::Cuda(c) => c,
                _ => bail!("q must be on CUDA"),
            };
            let q_slice = q_cuda.as_cuda_slice::<bf16>()?;
            let mut out = stream
                .alloc_zeros::<bf16>(expected)
                .map_err(|e| anyhow!(e))?;

            self.decode_attention_paged_into(
                layer,
                q_slice,
                q_l.start_offset(),
                &mut out,
                0,
                n_q,
                block_table_dev,
                block_size,
                n_total_dev,
                scratch,
                fan_in,
                sliding_window,
                scaling,
            )?;

            let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, 1usize, n_q * head_dim),
                candle_core::op::BackpropOp::none(),
                false,
            ))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn decode_attention_paged_into(
            &self,
            layer: usize,
            q: &CudaSlice<bf16>,
            q_offset: usize,
            out: &mut CudaSlice<bf16>,
            out_offset: usize,
            n_q: usize,
            block_table_dev: &CudaSlice<i32>,
            block_size: usize,
            n_total_dev: &CudaSlice<i32>,
            scratch: &mut CudaSlice<f32>,
            fan_in: &mut CudaSlice<u32>,
            sliding_window: Option<usize>,
            scaling: f32,
        ) -> Result<()> {
            let g = self.cfg.layers[layer];
            let (n_kv, head_dim) = (g.n_kv, g.head_dim);
            let dev = match self.device.clone() {
                Device::Cuda(d) => d,
                _ => unreachable!(),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let slab = &self.layers[layer];

            let expected = n_q * head_dim;
            if q_offset + expected > q.len() {
                bail!(
                    "decode_attention_paged_into layer {layer}: q holds {} elems, need {} at {q_offset}",
                    q.len(),
                    expected
                );
            }
            if out_offset + expected > out.len() {
                bail!(
                    "decode_attention_paged_into layer {layer}: out holds {} elems, need {} at {out_offset}",
                    out.len(),
                    expected
                );
            }
            let q_slice = q.slice(q_offset..q_offset + expected);
            let out_bytes = (out_offset * std::mem::size_of::<bf16>()) as u64;

            let derive = self.derive_w_inv(layer);
            if derive.is_some() && sliding_window.is_some() {
                bail!(
                    "decode_attention_paged layer {layer}: derived V takes its RoPE angle from \
                     the LOGICAL position, which a windowed/ringed layer does not preserve"
                );
            }
            let rc = match (derive, self.derive.as_ref()) {
                (Some(w_inv), Some(d)) => {
                    let (q_ptr, _g0) = q_slice.device_ptr(&stream);
                    let (k_ptr, _g1) = slab.k_fp8.device_ptr(&stream);
                    let (ks_ptr, _g2) = slab.k_scale.device_ptr(&stream);
                    let (inv_ptr, _g3) = d.inv_freq.device_ptr(&stream);
                    let (nt_ptr, _g4) = n_total_dev.device_ptr(&stream);
                    let (bt_ptr, _g5) = block_table_dev.device_ptr(&stream);
                    let (out_ptr, _g6) = out.device_ptr_mut(&stream);
                    let (scr_ptr, _g7) = scratch.device_ptr_mut(&stream);
                    let (fan_ptr, _g8) = fan_in.device_ptr_mut(&stream);
                    let out_ptr = out_ptr + out_bytes;
                    let rc = unsafe {
                        nv_kernels::cuda::flash_decode_derivev_fp8kv_paged(
                            stream.cu_stream() as *mut c_void,
                            q_ptr as *const u16,
                            k_ptr as *const u8,
                            ks_ptr as *const f32,
                            inv_ptr as *const f32,
                            std::ptr::null(),
                            std::ptr::null(),
                            out_ptr as *mut u16,
                            nt_ptr as *const i32,
                            scr_ptr as *mut f32,
                            fan_ptr as *mut u32,
                            n_q as i32,
                            n_kv as i32,
                            head_dim as i32,
                            0,
                            0,
                            d.rope_angles,
                            w_inv,
                            scaling,
                            bt_ptr as *const i32,
                            block_size as i32,
                        )
                    };
                    if rc == 0 {
                        d.dispatches.fetch_add(1, Ordering::Relaxed);
                    }
                    rc
                }
                _ => {
                    let (Some(v_fp8), Some(v_scale)) = (slab.v_fp8.as_ref(), slab.v_scale.as_ref())
                    else {
                        bail!("decode_attention_paged layer {layer}: no V slab and no derive plan");
                    };
                    let (q_ptr, _g0) = q_slice.device_ptr(&stream);
                    let (k_ptr, _g1) = slab.k_fp8.device_ptr(&stream);
                    let (v_ptr, _g2) = v_fp8.device_ptr(&stream);
                    let (ks_ptr, _g3) = slab.k_scale.device_ptr(&stream);
                    let (vs_ptr, _g4) = v_scale.device_ptr(&stream);
                    let (nt_ptr, _g5) = n_total_dev.device_ptr(&stream);
                    let (bt_ptr, _g6) = block_table_dev.device_ptr(&stream);
                    let (out_ptr, _g7) = out.device_ptr_mut(&stream);
                    let (scr_ptr, _g8) = scratch.device_ptr_mut(&stream);
                    let (fan_ptr, _g9) = fan_in.device_ptr_mut(&stream);
                    let out_ptr = out_ptr + out_bytes;
                    let s_raw = stream.cu_stream() as *mut c_void;

                    unsafe {
                        nv_kernels::cuda::flash_decode_fused_fp8kv_paged(
                            s_raw,
                            q_ptr as *const u16,
                            k_ptr as *const u8,
                            v_ptr as *const u8,
                            ks_ptr as *const f32,
                            vs_ptr as *const f32,
                            out_ptr as *mut u16,
                            nt_ptr as *const i32,
                            scr_ptr as *mut f32,
                            fan_ptr as *mut u32,
                            n_q as i32,
                            n_kv as i32,
                            head_dim as i32,
                            sliding_window.unwrap_or(0) as i32,
                            0,
                            scaling,
                            bt_ptr as *const i32,
                            block_size as i32,
                        )
                    }
                }
            };
            if rc != 0 {
                bail!("paged decode attention layer {layer} rc={rc}");
            }
            Ok(())
        }

        pub fn prefill_attention_paged_into(
            &self,
            layer: usize,
            q: &CudaSlice<bf16>,
            q_offset: usize,
            out: &mut CudaSlice<bf16>,
            out_offset: usize,
            seq: usize,
            n_q: usize,
            block_table_dev: &CudaSlice<i32>,
            block_size: usize,
            n_total_dev: &CudaSlice<i32>,
            scratch: &mut CudaSlice<f32>,
            fan_in: &mut CudaSlice<u32>,
            sliding_window: Option<usize>,
            scaling: f32,
        ) -> Result<()> {
            let g = self.cfg.layers[layer];
            let (n_kv, head_dim) = (g.n_kv, g.head_dim);
            let dev = match self.device.clone() {
                Device::Cuda(d) => d,
                _ => unreachable!(),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let slab = &self.layers[layer];
            let (Some(v_fp8), Some(v_scale)) = (slab.v_fp8.as_ref(), slab.v_scale.as_ref()) else {
                bail!("prefill_attention_paged_into layer {layer}: no V slab");
            };
            let row = n_q * head_dim;
            if q_offset + seq * row > q.len() || out_offset + seq * row > out.len() {
                bail!(
                    "prefill_attention_paged_into layer {layer}: {seq} rows of {row} do not fit \
                     q({}) at {q_offset} or out({}) at {out_offset}",
                    q.len(),
                    out.len()
                );
            }
            let need = nv_kernels::cuda::flash_splitk_scratch_elems_mk(
                n_q as i32,
                head_dim as i32,
                PREFILL_MK_TILE_IS_THE_KERNELS_KMAXM_OF_8 as i32,
            );
            if scratch.len() < need {
                bail!(
                    "prefill_attention_paged_into layer {layer}: scratch holds {} elems, the mk \
                     kernel needs {need} for n_q {n_q} head_dim {head_dim}",
                    scratch.len()
                );
            }

            let mut at = 0usize;
            while at < seq {
                let m = PREFILL_MK_TILE_IS_THE_KERNELS_KMAXM_OF_8.min(seq - at);
                let delta = (seq - (at + m)) as i32;
                let q_view = q.slice(q_offset + at * row..q_offset + (at + m) * row);
                let (q_ptr, _g0) = q_view.device_ptr(&stream);
                let (k_ptr, _g1) = slab.k_fp8.device_ptr(&stream);
                let (v_ptr, _g2) = v_fp8.device_ptr(&stream);
                let (ks_ptr, _g3) = slab.k_scale.device_ptr(&stream);
                let (vs_ptr, _g4) = v_scale.device_ptr(&stream);
                let (nt_ptr, _g5) = n_total_dev.device_ptr(&stream);
                let (bt_ptr, _g6) = block_table_dev.device_ptr(&stream);
                let (out_ptr, _g7) = out.device_ptr_mut(&stream);
                let (scr_ptr, _g8) = scratch.device_ptr_mut(&stream);
                let (fan_ptr, _g9) = fan_in.device_ptr_mut(&stream);
                let out_row =
                    out_ptr + ((out_offset + at * row) * std::mem::size_of::<bf16>()) as u64;
                let rc = unsafe {
                    nv_kernels::cuda::flash_decode_fused_fp8kv_mk_paged(
                        stream.cu_stream() as *mut c_void,
                        q_ptr as *const u16,
                        k_ptr as *const u8,
                        v_ptr as *const u8,
                        ks_ptr as *const f32,
                        vs_ptr as *const f32,
                        out_row as *mut u16,
                        nt_ptr as *const i32,
                        delta,
                        m as i32,
                        scr_ptr as *mut f32,
                        fan_ptr as *mut u32,
                        n_q as i32,
                        n_kv as i32,
                        head_dim as i32,
                        sliding_window.unwrap_or(0) as i32,
                        0,
                        scaling,
                        bt_ptr as *const i32,
                        block_size as i32,
                    )
                };
                if rc != 0 {
                    bail!("flash_decode_fused_fp8kv_mk_paged layer {layer} tile at {at} rc={rc}");
                }
                at += m;
            }
            Ok(())
        }

        pub fn gather_layer(
            &self,
            layer: usize,
            len: usize,
            block_table_dev: &CudaSlice<i32>,
        ) -> Result<(Tensor, Tensor)> {
            let g = self.cfg.layers[layer];
            let bs = self.cfg.block_size;
            let dev = match self.device.clone() {
                Device::Cuda(d) => d,
                _ => unreachable!(),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let slab = &self.layers[layer];
            let k = Self::gather_one_kv(
                &stream,
                &slab.k_fp8,
                &slab.k_scale,
                block_table_dev,
                bs,
                len,
                g.n_kv,
                g.head_dim,
                &dev,
            )?;
            let v = match (slab.v_fp8.as_ref(), slab.v_scale.as_ref()) {
                (Some(fp8), Some(scale)) => Self::gather_one_kv(
                    &stream,
                    fp8,
                    scale,
                    block_table_dev,
                    bs,
                    len,
                    g.n_kv,
                    g.head_dim,
                    &dev,
                )?,
                _ => {
                    let mut out = stream
                        .alloc_zeros::<bf16>(len * g.kv_elems())
                        .map_err(|e| anyhow!(e))?;
                    self.derive_one_v(layer, &stream, block_table_dev, len, &mut out)?;
                    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
                    Tensor::from_storage(
                        candle_core::Storage::Cuda(storage),
                        (1usize, len, g.n_kv, g.head_dim),
                        candle_core::op::BackpropOp::none(),
                        false,
                    )
                }
            };
            Ok((k, v))
        }

        pub fn copy_block(&mut self, src_block: u32, dst_block: u32) -> Result<()> {
            if src_block == dst_block {
                return Ok(());
            }
            let bs = self.cfg.block_size;
            let dev = match self.device.clone() {
                Device::Cuda(d) => d,
                _ => unreachable!(),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let s_raw = stream.cu_stream() as *mut c_void;
            for layer in 0..self.cfg.layers.len() {
                let g = self.cfg.layers[layer];
                let slab = &mut self.layers[layer];
                let mut halves: Vec<(&mut CudaSlice<u8>, &mut CudaSlice<f32>)> =
                    vec![(&mut slab.k_fp8, &mut slab.k_scale)];
                if let (Some(fp8), Some(scale)) = (slab.v_fp8.as_mut(), slab.v_scale.as_mut()) {
                    halves.push((fp8, scale));
                }
                for (fp8, scale) in halves {
                    let (fp8_ptr, _g0) = fp8.device_ptr_mut(&stream);
                    let (sc_ptr, _g1) = scale.device_ptr_mut(&stream);
                    let rc = unsafe {
                        nv_kernels::cuda::copy_kv_block_fp8(
                            s_raw,
                            fp8_ptr as *const u8,
                            sc_ptr as *const f32,
                            fp8_ptr as *mut u8,
                            sc_ptr as *mut f32,
                            src_block as i32,
                            dst_block as i32,
                            bs as i32,
                            g.n_kv as i32,
                            g.head_dim as i32,
                        )
                    };
                    if rc != 0 {
                        bail!("copy_kv_block_fp8 rc={rc}");
                    }
                }
            }
            Ok(())
        }
    }

    pub fn paged_attn_fp8_enabled() -> bool {
        !matches!(std::env::var("NV_PAGED_ATTN_FP8").as_deref(), Ok("0"))
    }

    pub const PAGED_PREFILL_FP8_ENV: &str = "NV_PAGED_PREFILL_FP8";

    pub fn paged_prefill_fp8_enabled() -> bool {
        matches!(std::env::var(PAGED_PREFILL_FP8_ENV).as_deref(), Ok("1"))
    }

    pub const PAGED_ATTN_FP8_RING_ENV: &str = "NV_PAGED_ATTN_FP8_RING";

    pub fn paged_attn_fp8_ring_enabled() -> bool {
        !matches!(std::env::var(PAGED_ATTN_FP8_RING_ENV).as_deref(), Ok("0"))
    }

    pub struct PagedGemma4Cache {
        pool: Arc<Mutex<PagedKvFp8Pool>>,
        block_table: Vec<u32>,
        block_table_dev: CudaSlice<i32>,
        sliding_table_dev: Option<CudaSlice<i32>>,

        block_table_uploaded: usize,
        sliding_uploaded: usize,
        uploaded_stream: Option<Arc<CudaStream>>,

        htod_scratch: Vec<i32>,
        lane: Option<u32>,
        layer_sliding: Vec<bool>,
        ring_enabled: bool,
        ring_decodes: u64,
        sliding_ring_blocks: usize,
        start_dev: CudaSlice<i32>,
        n_total_dev: CudaSlice<i32>,
        flash_scratch: Option<CudaSlice<f32>>,
        flash_fan_in: Option<CudaSlice<u32>>,
        current_len: usize,
        pending_write_pos: usize,
        block_size: usize,
        device: Device,
    }

    impl PagedGemma4Cache {
        pub fn new(pool: Arc<Mutex<PagedKvFp8Pool>>, device: &Device) -> Result<Self> {
            let dev = match device {
                Device::Cuda(d) => d.clone(),
                _ => bail!("PagedGemma4Cache requires a CUDA device"),
            };
            let (block_size, max_blocks, lanes, layer_sliding, ring_blocks) = {
                let p = pool.lock().unwrap();
                let c = p.config();
                (
                    c.block_size,
                    c.num_blocks,
                    c.lanes,
                    c.layer_sliding.clone(),
                    c.sliding_ring_blocks,
                )
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let block_table_dev = stream
                .alloc_zeros::<i32>(max_blocks.max(1))
                .map_err(|e| anyhow!(e))?;
            let (lane, sliding_table_dev) = if lanes > 0 {
                let lane = pool
                    .lock()
                    .unwrap()
                    .acquire_lane()
                    .ok_or_else(|| anyhow!("PagedGemma4Cache: no free sliding-ring lane"))?;
                let t = stream
                    .alloc_zeros::<i32>(max_blocks.max(1))
                    .map_err(|e| anyhow!(e))?;
                (Some(lane), Some(t))
            } else {
                (None, None)
            };
            let start_dev = stream.alloc_zeros::<i32>(1).map_err(|e| anyhow!(e))?;
            let n_total_dev = stream.alloc_zeros::<i32>(1).map_err(|e| anyhow!(e))?;

            let (flash_scratch, flash_fan_in) = (None, None);
            Ok(Self {
                pool,
                block_table: Vec::new(),
                block_table_dev,
                sliding_table_dev,
                block_table_uploaded: 0,
                sliding_uploaded: 0,
                uploaded_stream: None,
                htod_scratch: Vec::new(),
                lane,
                layer_sliding,
                ring_enabled: paged_attn_fp8_ring_enabled(),
                ring_decodes: 0,
                sliding_ring_blocks: ring_blocks,
                start_dev,
                n_total_dev,
                flash_scratch,
                flash_fan_in,
                current_len: 0,
                pending_write_pos: 0,
                block_size,
                device: device.clone(),
            })
        }

        pub fn set_block_table(&mut self, table: &[u32]) -> Result<()> {
            let dev = match self.device.clone() {
                Device::Cuda(d) => d,
                _ => unreachable!(),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            if table.len() > self.block_table_dev.len() {
                bail!(
                    "PagedGemma4Cache: block table len {} exceeds pool capacity {}",
                    table.len(),
                    self.block_table_dev.len()
                );
            }

            let same_stream = self
                .uploaded_stream
                .as_ref()
                .is_some_and(|s| Arc::ptr_eq(s, &stream));
            let mut first_diff = 0usize;
            if same_stream && self.block_table_uploaded == self.block_table.len() {
                let common = table.len().min(self.block_table.len());
                while first_diff < common && self.block_table[first_diff] == table[first_diff] {
                    first_diff += 1;
                }
            }

            self.block_table.clear();
            self.block_table.extend_from_slice(table);

            if !same_stream {
                self.block_table_uploaded = 0;
                self.sliding_uploaded = 0;
            }

            if first_diff < table.len() {
                self.htod_scratch.clear();
                self.htod_scratch
                    .extend(table[first_diff..].iter().map(|&b| b as i32));
                let mut view = self.block_table_dev.slice_mut(first_diff..table.len());
                self.block_table_uploaded = 0;
                #[allow(deprecated)]
                stream
                    .memcpy_htod(&self.htod_scratch, &mut view)
                    .map_err(|e| anyhow!(e))?;
            }

            self.block_table_uploaded = table.len();

            if let (Some(lane), Some(sdev)) = (self.lane, self.sliding_table_dev.as_mut()) {
                let rb = self.sliding_ring_blocks.max(1);
                let from = self.sliding_uploaded;
                if from < table.len() {
                    self.htod_scratch.clear();
                    self.htod_scratch.extend(
                        (from..table.len()).map(|j| (lane as usize * rb + (j % rb)) as i32),
                    );
                    let mut view = sdev.slice_mut(from..table.len());
                    self.sliding_uploaded = 0;
                    #[allow(deprecated)]
                    stream
                        .memcpy_htod(&self.htod_scratch, &mut view)
                        .map_err(|e| anyhow!(e))?;
                    self.sliding_uploaded = table.len();
                }
            }

            self.uploaded_stream = Some(stream);
            Ok(())
        }

        fn table_for_layer(&self, layer: usize) -> &CudaSlice<i32> {
            if self.layer_sliding.get(layer).copied().unwrap_or(false) {
                self.sliding_table_dev
                    .as_ref()
                    .unwrap_or(&self.block_table_dev)
            } else {
                &self.block_table_dev
            }
        }

        pub fn apply_cow(&mut self, src_block: u32, dst_block: u32) -> Result<()> {
            self.pool.lock().unwrap().copy_block(src_block, dst_block)
        }

        pub fn ring_decodes(&self) -> u64 {
            self.ring_decodes
        }

        pub fn block_table(&self) -> &[u32] {
            &self.block_table
        }

        pub fn block_size(&self) -> usize {
            self.block_size
        }

        pub fn pending_write_pos(&self) -> usize {
            self.pending_write_pos
        }

        pub fn current_len(&self) -> usize {
            self.current_len
        }

        fn push_n_total(&mut self, n_total: usize) -> Result<()> {
            let dev = match self.device.clone() {
                Device::Cuda(d) => d,
                _ => unreachable!(),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let host = [n_total as i32];
            stream
                .memcpy_htod(&host, &mut self.n_total_dev)
                .map_err(|e| anyhow!(e))?;
            Ok(())
        }

        fn push_start(&mut self, write_pos: usize) -> Result<()> {
            let dev = match self.device.clone() {
                Device::Cuda(d) => d,
                _ => unreachable!(),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let host = [write_pos as i32];
            stream
                .memcpy_htod(&host, &mut self.start_dev)
                .map_err(|e| anyhow!(e))?;
            Ok(())
        }
    }

    impl Gemma4Cache for PagedGemma4Cache {
        fn current_len(&self) -> usize {
            self.current_len
        }
        fn advance(&mut self, n: usize) {
            self.current_len += n;
        }
        fn prepare_for_decode(&mut self, write_pos: usize, n_total: usize) -> Result<()> {
            self.pending_write_pos = write_pos;
            self.push_n_total(n_total)?;
            self.push_start(write_pos)
        }
        fn write_at(&mut self, layer: usize, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
            let dims = k_new.dims();
            if dims.len() != 4 || dims[0] != 1 {
                bail!(
                    "PagedGemma4Cache.write_at: expected [1, t, n_kv, head_dim], got {:?}",
                    dims
                );
            }
            let n_tokens = dims[1];
            let capacity = self.block_table.len() * self.block_size;
            let end = self.pending_write_pos.saturating_add(n_tokens);
            if end > capacity {
                bail!(
                    "PagedGemma4Cache.write_at: write of {n_tokens} token(s) at {} ends at {end}, \
                     past the {} block(s) mapped for this sequence ({capacity} slots)",
                    self.pending_write_pos,
                    self.block_table.len()
                );
            }
            let block_table_dev = self.table_for_layer(layer);
            let start_dev = &self.start_dev;
            self.pool.lock().unwrap().append_layer(
                layer,
                k_new,
                v_new,
                n_tokens,
                start_dev,
                block_table_dev,
            )
        }
        fn view(&mut self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
            let table = self.table_for_layer(layer);
            self.pool.lock().unwrap().gather_layer(layer, len, table)
        }

        fn try_prefill_attention_fp8(
            &mut self,
            layer: usize,
            q_rot: &Tensor,
            n_q: usize,
            seq: usize,
            sliding_window: Option<usize>,
            scaling: f32,
        ) -> Result<Option<Tensor>> {
            if !paged_attn_fp8_enabled() || !paged_prefill_fp8_enabled() {
                return Ok(None);
            }
            let sliding = self.layer_sliding.get(layer).copied().unwrap_or(false);
            if sliding && !self.ring_enabled {
                return Ok(None);
            }
            let head_dim = {
                let p = self.pool.lock().unwrap();
                p.cfg.layers[layer].head_dim
            };
            if head_dim > PREFILL_MK_MAX_HEAD_DIM_IS_THE_KERNELS_KMAXHDMK_OF_256 {
                return Ok(None);
            }
            let dev = match self.device.clone() {
                Device::Cuda(d) => d,
                _ => return Ok(None),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let need = nv_kernels::cuda::flash_splitk_scratch_elems_mk(
                n_q as i32,
                head_dim as i32,
                PREFILL_MK_TILE_IS_THE_KERNELS_KMAXM_OF_8 as i32,
            );
            if self.flash_scratch.as_ref().is_none_or(|b| b.len() < need) {
                self.flash_scratch = Some(stream.alloc_zeros::<f32>(need).map_err(|e| anyhow!(e))?);
            }
            if self
                .flash_fan_in
                .as_ref()
                .is_none_or(|b| b.len() < n_q.max(1))
            {
                self.flash_fan_in = Some(
                    stream
                        .alloc_zeros::<u32>(n_q.max(1))
                        .map_err(|e| anyhow!(e))?,
                );
            }
            let table = if sliding {
                self.sliding_table_dev
                    .as_ref()
                    .unwrap_or(&self.block_table_dev)
            } else {
                &self.block_table_dev
            };
            let q_c = q_rot.contiguous()?;
            let (q_storage, q_l) = q_c.storage_and_layout();
            let q_cuda = match &*q_storage {
                candle_core::Storage::Cuda(c) => c,
                _ => return Ok(None),
            };
            let q_slice = q_cuda.as_cuda_slice::<bf16>()?;
            let mut out = stream
                .alloc_zeros::<bf16>(seq * n_q * head_dim)
                .map_err(|e| anyhow!(e))?;
            let (Some(scratch), Some(fan_in)) =
                (self.flash_scratch.as_mut(), self.flash_fan_in.as_mut())
            else {
                return Ok(None);
            };
            self.pool.lock().unwrap().prefill_attention_paged_into(
                layer,
                q_slice,
                q_l.start_offset(),
                &mut out,
                0,
                seq,
                n_q,
                table,
                self.block_size,
                &self.n_total_dev,
                scratch,
                fan_in,
                sliding_window,
                scaling,
            )?;
            let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Some(Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, seq, n_q * head_dim),
                candle_core::op::BackpropOp::none(),
                false,
            )))
        }

        fn try_decode_attention_fp8(
            &mut self,
            layer: usize,
            q_rot: &Tensor,
            n_q: usize,
            sliding_window: Option<usize>,
            scaling: f32,
        ) -> Result<Option<Tensor>> {
            if !paged_attn_fp8_enabled() {
                return Ok(None);
            }
            let head_dim = {
                let p = self.pool.lock().unwrap();
                p.cfg.layers[layer].head_dim
            };
            let need = nv_kernels::cuda::flash_splitk_scratch_elems(n_q as i32, head_dim as i32);
            anyhow::ensure!(
                need > 0,
                "paged attn: bad scratch size n_q={n_q} hd={head_dim}"
            );
            let need = need as usize;
            let dev = match self.device.clone() {
                Device::Cuda(d) => d,
                _ => return Ok(None),
            };
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            if self.flash_scratch.as_ref().is_none_or(|b| b.len() < need) {
                self.flash_scratch = Some(stream.alloc_zeros::<f32>(need).map_err(|e| anyhow!(e))?);
            }
            if self
                .flash_fan_in
                .as_ref()
                .is_none_or(|b| b.len() < n_q.max(1))
            {
                self.flash_fan_in = Some(
                    stream
                        .alloc_zeros::<u32>(n_q.max(1))
                        .map_err(|e| anyhow!(e))?,
                );
            }
            let (Some(scratch), Some(fan_in)) =
                (self.flash_scratch.as_mut(), self.flash_fan_in.as_mut())
            else {
                return Ok(None);
            };

            let sliding = self.layer_sliding.get(layer).copied().unwrap_or(false);
            if sliding && !self.ring_enabled {
                return Ok(None);
            }
            if sliding {
                self.ring_decodes += 1;
            }
            let table = if sliding {
                self.sliding_table_dev
                    .as_ref()
                    .unwrap_or(&self.block_table_dev)
            } else {
                &self.block_table_dev
            };
            let out = self.pool.lock().unwrap().decode_attention_paged(
                layer,
                q_rot,
                n_q,
                table,
                self.block_size,
                &self.n_total_dev,
                scratch,
                fan_in,
                sliding_window,
                scaling,
            )?;
            Ok(Some(out))
        }
    }

    impl Drop for PagedGemma4Cache {
        fn drop(&mut self) {
            if let Some(lane) = self.lane.take() {
                if let Ok(mut p) = self.pool.lock() {
                    p.release_lane(lane);
                }
            }
        }
    }
}

#[cfg(test)]
#[cfg(not(feature = "cuda"))]
mod tests {
    use super::*;

    fn pool_cfg() -> PagedPoolConfig {
        PagedPoolConfig {
            num_blocks: 4,
            block_size: 4,
            layers: vec![
                LayerKvGeometry {
                    n_kv: 2,
                    head_dim: 8,
                },
                LayerKvGeometry {
                    n_kv: 2,
                    head_dim: 8,
                },
            ],
            layer_blocks: vec![4, 4],
            layer_sliding: vec![false, false],
            lanes: 0,
            sliding_ring_blocks: 0,
        }
    }

    #[test]
    fn physical_slot_routes_through_block_table() {
        let table = vec![3u32, 1u32];
        assert_eq!(physical_slot(&table, 4, 0).unwrap(), 12);
        assert_eq!(physical_slot(&table, 4, 3).unwrap(), 15);
        assert_eq!(physical_slot(&table, 4, 4).unwrap(), 4);
        assert_eq!(physical_slot(&table, 4, 7).unwrap(), 7);
        assert!(physical_slot(&table, 4, 8).is_err());
    }

    #[test]
    fn append_then_gather_round_trips_within_fp8_error() {
        let mut pool = PagedKvFp8Pool::new(pool_cfg()).unwrap();
        let table = vec![2u32, 0u32];
        let n_kv = 2usize;
        let hd = 8usize;
        let n = 6usize;
        let mut k = vec![0f32; n * n_kv * hd];
        let mut v = vec![0f32; n * n_kv * hd];
        for t in 0..n {
            for i in 0..n_kv * hd {
                k[t * n_kv * hd + i] = ((t * 31 + i) as f32 % 13.0) - 6.0;
                v[t * n_kv * hd + i] = ((t * 17 + i) as f32 % 11.0) - 5.0;
            }
        }
        for layer in 0..2 {
            pool.append_layer(layer, &table, 0, &k, &v, n).unwrap();
        }
        let (kg, vg) = pool.gather_layer(0, &table, n).unwrap();
        assert_eq!(kg.len(), n * n_kv * hd);
        for t in 0..n {
            for h in 0..n_kv {
                let kbase = (t * n_kv + h) * hd;
                let k_amax = (0..hd).map(|d| k[kbase + d].abs()).fold(0.0f32, f32::max);
                let v_amax = (0..hd).map(|d| v[kbase + d].abs()).fold(0.0f32, f32::max);
                let ktol = k_amax / 127.0 + 1e-4;
                let vtol = v_amax / 127.0 + 1e-4;
                for d in 0..hd {
                    assert!(
                        (kg[kbase + d] - k[kbase + d]).abs() <= ktol,
                        "k mismatch t={t} h={h} d={d}: {} vs {}",
                        kg[kbase + d],
                        k[kbase + d]
                    );
                    assert!(
                        (vg[kbase + d] - v[kbase + d]).abs() <= vtol,
                        "v mismatch t={t} h={h} d={d}: {} vs {}",
                        vg[kbase + d],
                        v[kbase + d]
                    );
                }
            }
        }
    }

    #[test]
    fn copy_block_duplicates_contents() {
        let mut pool = PagedKvFp8Pool::new(pool_cfg()).unwrap();
        let table = vec![1u32];
        let n_kv = 2usize;
        let hd = 8usize;
        let n = 4usize;
        let k: Vec<f32> = (0..n * n_kv * hd).map(|i| (i as f32) * 0.1 - 2.0).collect();
        let v: Vec<f32> = (0..n * n_kv * hd)
            .map(|i| (i as f32) * -0.07 + 1.0)
            .collect();
        pool.append_layer(0, &table, 0, &k, &v, n).unwrap();
        pool.append_layer(1, &table, 0, &k, &v, n).unwrap();

        pool.copy_block(1, 3).unwrap();
        let src_gather = pool.gather_layer(0, &[1u32], n).unwrap();
        let dst_gather = pool.gather_layer(0, &[3u32], n).unwrap();
        assert_eq!(src_gather.0, dst_gather.0);
        assert_eq!(src_gather.1, dst_gather.1);
    }

    #[test]
    fn pool_geometry_accounting() {
        let cfg = pool_cfg();
        assert_eq!(cfg.num_pool_slots(), 16);
        assert!(cfg.pool_bytes() > 0);
        assert_eq!(cfg.pool_bytes(), cfg.num_blocks * cfg.bytes_per_block());
        let budget = cfg.pool_bytes() * 10;
        assert!(cfg.max_blocks_for_budget(budget) >= cfg.num_blocks);
    }
}
