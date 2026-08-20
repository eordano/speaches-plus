use anyhow::{bail, Result};
use candle_core::Tensor;
use nv_layers::lora_slots::{LoraAdapter, LoraModuleSpec, LoraModuleWeights};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(any(feature = "cuda", feature = "wgpu"))]
use nv_models::gemma4::Gemma4;

pub fn qkv_module_names(layer: usize, has_v: bool) -> Vec<String> {
    let mut names = vec![format!("l{layer}.q"), format!("l{layer}.k")];
    if has_v {
        names.push(format!("l{layer}.v"));
    }
    names
}

pub fn det_rand_tensor(
    seed: u64,
    rows: usize,
    cols: usize,
    magnitude: f64,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut v = Vec::with_capacity(rows * cols);
    for i in 0..rows * cols {
        let mut z = seed
            .wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1))
            .wrapping_mul(0xBF58476D1CE4E5B9);
        z ^= z >> 29;
        z = z.wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 32;
        let u = ((z & 0xFFFF) as f32 / 65535.0) - 0.5;
        v.push(half::bf16::from_f32(u * 2.0 * magnitude as f32));
    }
    Ok(Tensor::from_vec(v, (rows, cols), device)?)
}

pub fn synth_adapter(
    specs: &[LoraModuleSpec],
    rank: usize,
    magnitude: f64,
    seed: u64,
    device: &candle_core::Device,
) -> Result<LoraAdapter> {
    let mut modules = HashMap::new();
    for (i, spec) in specs.iter().enumerate() {
        let sa = seed ^ ((i as u64) << 8) ^ spec.name.len() as u64;
        let a = det_rand_tensor(sa ^ 0xA, rank, spec.in_features, magnitude, device)?;
        let b = det_rand_tensor(sa ^ 0xB, spec.out_features, rank, magnitude, device)?;
        modules.insert(spec.name.clone(), LoraModuleWeights { a, b });
    }
    Ok(LoraAdapter {
        scaling: 1.0,
        modules,
    })
}

struct TokenMapState {
    mapping: Vec<i32>,
    no_lora: bool,
    armed: bool,
}

pub struct LoraTokenMap {
    max_tokens: usize,
    max_loras: usize,
    state: Mutex<TokenMapState>,
}

impl LoraTokenMap {
    pub fn new(max_tokens: usize, max_loras: usize) -> Result<Arc<Self>> {
        if max_tokens == 0 || max_loras == 0 {
            bail!("LoraTokenMap dims must be non-zero");
        }
        Ok(Arc::new(Self {
            max_tokens,
            max_loras,
            state: Mutex::new(TokenMapState {
                mapping: Vec::new(),
                no_lora: true,
                armed: false,
            }),
        }))
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn max_loras(&self) -> usize {
        self.max_loras
    }

    pub fn set_mapping(&self, mapping: &[i32]) -> Result<()> {
        if mapping.is_empty() {
            bail!("LoraTokenMap.set_mapping: empty mapping");
        }
        if mapping.len() > self.max_tokens {
            bail!(
                "LoraTokenMap.set_mapping: {} tokens exceeds max_tokens {}",
                mapping.len(),
                self.max_tokens
            );
        }
        for &v in mapping {
            if v < -1 || v >= self.max_loras as i32 {
                bail!("LoraTokenMap.set_mapping: slot {v} out of range");
            }
        }
        let mut st = self
            .state
            .lock()
            .map_err(|e| anyhow::anyhow!("LoraTokenMap state poisoned: {e}"))?;
        st.no_lora = mapping.iter().all(|&v| v == -1);
        st.mapping = mapping.to_vec();
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

    pub fn snapshot(&self) -> Option<Vec<i32>> {
        let st = self.state.lock().ok()?;
        if !st.armed || st.no_lora {
            return None;
        }
        Some(st.mapping.clone())
    }
}

#[cfg(feature = "wgpu")]
pub use wgpu_runtime::{WgpuLoraHook, WgpuLoraRuntime};

#[cfg(feature = "wgpu")]
mod wgpu_runtime {
    use super::LoraTokenMap;
    use anyhow::{anyhow, bail, Context, Result};
    use candle_core::{DType, Tensor};
    use half::bf16;
    use nv_kernels::wgpu_backend::kernels::lora as wgpu_lora;
    use nv_kernels::wgpu_backend::WgpuContext;
    use nv_layers::linear::{Linear, LoraDeltaHook};
    use nv_layers::lora_slots::{LoraAdapter, LoraModuleSpec, LoraSlotManager, LoraSlotStack};
    use nv_models::gemma4::Gemma4;
    use std::sync::Arc;

    pub const FUSED_MAX_M: usize = 64;

    pub struct WgpuLoraHook {
        ctx: &'static WgpuContext,
        map: Arc<LoraTokenMap>,
        a_slices: Vec<Vec<u16>>,
        b_slices: Vec<Vec<u16>>,
        widths: Vec<usize>,
        rank: usize,
        in_features: usize,
        out_features: usize,
    }

    fn host_bits(t: &Tensor) -> Result<Vec<u16>> {
        let v = t.flatten_all()?.to_dtype(DType::BF16)?.to_vec1::<bf16>()?;
        Ok(v.into_iter().map(|x| x.to_bits()).collect())
    }

    impl WgpuLoraHook {
        pub fn from_stacks(
            ctx: &'static WgpuContext,
            map: Arc<LoraTokenMap>,
            stacks: &[&LoraSlotStack],
        ) -> Result<Arc<Self>> {
            if stacks.is_empty() {
                bail!("WgpuLoraHook needs at least one slot stack");
            }
            let rank = stacks[0].max_rank();
            let in_features = stacks[0].in_features();
            if rank > wgpu_lora::FUSED_MAX_RANK {
                bail!(
                    "WgpuLoraHook: max_rank {rank} exceeds kernel limit {}",
                    wgpu_lora::FUSED_MAX_RANK
                );
            }
            let mut widths = Vec::with_capacity(stacks.len());
            let mut a_slices = Vec::with_capacity(stacks.len());
            let mut b_slices = Vec::with_capacity(stacks.len());
            for st in stacks {
                if st.max_rank() != rank || st.in_features() != in_features {
                    bail!("WgpuLoraHook: slot stacks must share max_rank and in_features");
                }
                if st.max_loras() != map.max_loras() {
                    bail!(
                        "WgpuLoraHook: stack max_loras {} != map max_loras {}",
                        st.max_loras(),
                        map.max_loras()
                    );
                }
                if st.lora_a_stacked().dtype() != DType::BF16 {
                    bail!("WgpuLoraHook: slot stacks must be bf16");
                }
                a_slices.push(host_bits(st.lora_a_stacked())?);
                b_slices.push(host_bits(st.lora_b_stacked())?);
                widths.push(st.out_features());
            }
            let out_features = widths.iter().sum();
            Ok(Arc::new(Self {
                ctx,
                map,
                a_slices,
                b_slices,
                widths,
                rank,
                in_features,
                out_features,
            }))
        }

        pub fn map(&self) -> &Arc<LoraTokenMap> {
            &self.map
        }
    }

    impl LoraDeltaHook for WgpuLoraHook {
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
            let Some(mapping) = self.map.snapshot() else {
                return Ok(None);
            };
            let m = mapping.len();
            let (xm, xk) = x2.dims2()?;
            if xm != m {
                bail!(
                    "WgpuLoraHook.apply: batch rows {xm} != armed mapping length {m}; \
                     call LoraTokenMap::set_mapping with the current token count"
                );
            }
            if xk != self.in_features {
                bail!(
                    "WgpuLoraHook.apply: x cols {xk} != in_features {}",
                    self.in_features
                );
            }
            let (win_off, win_len) = win.unwrap_or((0, self.out_features));
            if win_off + win_len > self.out_features {
                bail!("WgpuLoraHook.apply: window {win_off}+{win_len} exceeds out_features");
            }
            let (ym, yn) = y2.dims2()?;
            if ym != m || yn != win_len {
                bail!("WgpuLoraHook.apply: y dims [{ym},{yn}] != [{m},{win_len}]");
            }
            if y2.dtype() != DType::BF16 {
                bail!("WgpuLoraHook.apply: y must be bf16");
            }

            let x_host = host_bits(&x2.to_dtype(DType::BF16)?.contiguous()?)?;
            let mut y_host: Vec<u16> = y2
                .contiguous()?
                .flatten_all()?
                .to_vec1::<bf16>()?
                .into_iter()
                .map(|v| v.to_bits())
                .collect();

            let meta = wgpu_lora::LoraMeta::prepare(&mapping, self.map.max_loras());
            let a_refs: Vec<&[u16]> = self.a_slices.iter().map(|v| v.as_slice()).collect();
            let b_refs: Vec<&[u16]> = self.b_slices.iter().map(|v| v.as_slice()).collect();
            let full = win_off == 0 && win_len == self.out_features;

            if full && m > FUSED_MAX_M {
                wgpu_lora::lora_grouped(
                    self.ctx,
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
                .map_err(|e| anyhow!("wgpu lora_grouped: {e}"))?;
            } else {
                wgpu_lora::lora_fused(
                    self.ctx,
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
                .map_err(|e| anyhow!("wgpu lora_fused: {e}"))?;
            }

            let y_new: Vec<bf16> = y_host.into_iter().map(bf16::from_bits).collect();
            let out = Tensor::from_vec(y_new, (m, win_len), y2.device())?;
            Ok(Some(out))
        }
    }

    pub struct WgpuLoraRuntime {
        map: Arc<LoraTokenMap>,
        manager: LoraSlotManager,
        slot: usize,
        hooked: usize,
    }

    impl WgpuLoraRuntime {
        pub fn install(
            ctx: &'static WgpuContext,
            targets: &[(&Linear, Vec<LoraModuleSpec>)],
            adapter: &LoraAdapter,
            max_rank: usize,
            max_tokens: usize,
            device: &candle_core::Device,
        ) -> Result<Self> {
            let map = LoraTokenMap::new(max_tokens, 1)?;
            let mut specs = Vec::new();
            for (_, group) in targets {
                for s in group {
                    specs.push(LoraModuleSpec::new(
                        s.name.clone(),
                        s.in_features,
                        s.out_features,
                    ));
                }
            }
            let mut manager = LoraSlotManager::new(1, max_rank, &specs, DType::BF16, device)
                .context("create LoraSlotManager")?;
            let slot = manager.activate(1, adapter).context("activate adapter")?;
            let mut hooked = 0usize;
            for (i, (linear, group)) in targets.iter().enumerate() {
                let stacks: Vec<&LoraSlotStack> = group
                    .iter()
                    .map(|s| manager.stack(&s.name).expect("stack registered above"))
                    .collect();
                let hook = WgpuLoraHook::from_stacks(ctx, map.clone(), &stacks)
                    .with_context(|| format!("build hook for target {i}"))?;
                linear
                    .attach_lora(hook)
                    .with_context(|| format!("attach hook to target {i}"))?;
                hooked += 1;
            }
            Ok(Self {
                map,
                manager,
                slot,
                hooked,
            })
        }

        pub fn install_gemma4_qkv(
            ctx: &'static WgpuContext,
            model: &Gemma4,
            adapter: &LoraAdapter,
            max_rank: usize,
            max_tokens: usize,
        ) -> Result<Self> {
            let device = model.device();
            let mut targets: Vec<(&Linear, Vec<LoraModuleSpec>)> = Vec::new();
            for (i, layer) in model.layers().iter().enumerate() {
                let attn = &layer.self_attn;
                let k_in = attn.qkv_proj.in_features();
                let widths: Vec<usize> = if attn.has_v {
                    vec![attn.q_dim, attn.kv_dim, attn.kv_dim]
                } else {
                    vec![attn.q_dim, attn.kv_dim]
                };
                let specs: Vec<LoraModuleSpec> = super::qkv_module_names(i, attn.has_v)
                    .into_iter()
                    .zip(widths)
                    .map(|(name, w)| LoraModuleSpec::new(name, k_in, w))
                    .collect();
                targets.push((&attn.qkv_proj, specs));
            }
            Self::install(ctx, &targets, adapter, max_rank, max_tokens, device)
        }

        pub fn detach_gemma4(model: &Gemma4) {
            for layer in model.layers() {
                layer.self_attn.qkv_proj.detach_lora();
            }
        }

        pub fn arm(&self, n_tokens: usize) -> Result<()> {
            self.map.set_mapping(&vec![self.slot as i32; n_tokens])
        }

        pub fn disarm(&self) {
            self.map.disarm();
        }

        pub fn armed(&self) -> bool {
            self.map.armed()
        }

        pub fn hooked_layers(&self) -> usize {
            self.hooked
        }

        pub fn manager(&self) -> &LoraSlotManager {
            &self.manager
        }

        pub fn map(&self) -> &Arc<LoraTokenMap> {
            &self.map
        }
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub fn synth_qkv_adapter(
    model: &Gemma4,
    rank: usize,
    magnitude: f64,
    seed: u64,
) -> Result<LoraAdapter> {
    let device = model.device();
    let mut modules = HashMap::new();
    for (i, layer) in model.layers().iter().enumerate() {
        let attn = &layer.self_attn;
        let k_in = attn.qkv_proj.in_features();
        let widths: Vec<usize> = if attn.has_v {
            vec![attn.q_dim, attn.kv_dim, attn.kv_dim]
        } else {
            vec![attn.q_dim, attn.kv_dim]
        };
        for (name, w) in qkv_module_names(i, attn.has_v).into_iter().zip(widths) {
            let sa = seed ^ (i as u64) << 8 ^ name.len() as u64;
            let a = det_rand_tensor(sa ^ 0xA, rank, k_in, magnitude, device)?;
            let b = det_rand_tensor(sa ^ 0xB, w, rank, magnitude, device)?;
            modules.insert(name, LoraModuleWeights { a, b });
        }
    }
    Ok(LoraAdapter {
        scaling: 1.0,
        modules,
    })
}
