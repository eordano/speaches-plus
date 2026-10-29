use anyhow::{bail, ensure, Context, Result};
use candle_core::{Device, Tensor};
use cudarc::driver::CudaStream;
use nv_layers::pinned_host::{upload_pinned_bytes_on, PinnedBudget, PinnedHostBuf};
use nv_models::qwen4_exp::lora::expert::{self, ExpertLoraForm, ExpertLoraLayer, ExpertLoraMeta};
use nv_models::qwen4_exp::Qwen4Exp;
use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use crate::lora_spec::Qwen4ExpServingAdapter;

pub const NV_Q4EXP_LORA_HOST_PARK_GIB: &str = "NV_Q4EXP_LORA_HOST_PARK_GIB";
pub const HOST_PARK_DEFAULT_1_GIB_HOLDS_A_FEW_REPLACED_ADAPTER_VERSIONS_FOR_ROLLBACK_WITHOUT_PRESSURING_A_HOST_THAT_ALREADY_PINS_THE_PLE_TABLE: f64 = 1.0;
pub const ADAPTER_DEVICE_ESTIMATE_3X_THE_BF16_FILE_BYTES_COVERS_THE_F32_PREFILL_FACTORS_AND_THE_M1_KERNEL_COPY: usize = 3;

pub const HOT_SWAP_STAGES_IN_PINNED_HOST_RAM_VALIDATES_THERE_UPLOADS_ON_A_COPY_STREAM_THEN_SWAPS_ONE_POINTER: &str =
    "a load reads the adapter files into pinned host buffers, validates form, rank, shapes and the \
     base identity against the resident model there, checks free VRAM above the engine's floor, \
     uploads on a copy stream, and only then replaces the id's live version; a request holds the \
     Arc of the version it started with, so the old weights free when its last request ends, and the \
     replaced version's host copy stays parked within NV_Q4EXP_LORA_HOST_PARK_GIB for rollback";

struct StTensor {
    shape: Vec<usize>,
    dtype: safetensors::Dtype,
    range: (usize, usize),
}

fn safetensors_index(bytes: &[u8], what: &str) -> Result<BTreeMap<String, StTensor>> {
    let st = safetensors::SafeTensors::deserialize(bytes).with_context(|| format!("{what}: not a safetensors file"))?;
    let base = bytes.as_ptr() as usize;
    Ok(st
        .tensors()
        .into_iter()
        .map(|(name, v)| {
            let start = v.data().as_ptr() as usize - base;
            (name, StTensor { shape: v.shape().to_vec(), dtype: v.dtype(), range: (start, start + v.data().len()) })
        })
        .collect())
}

pub struct HostStagedAdapter {
    pub dir: PathBuf,
    pub version: String,
    config: serde_json::Value,
    config_json: String,
    peft: PinnedHostBuf,
    expert: Option<(ExpertLoraMeta, PinnedHostBuf)>,
}

fn read_pinned(stream: &Arc<CudaStream>, path: &Path, what: &str) -> Result<PinnedHostBuf> {
    use std::io::Read as _;
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = f.metadata()?.len() as usize;
    let mut b = PinnedHostBuf::alloc(stream, len, what)?;
    f.read_exact(b.as_mut_slice()).with_context(|| format!("read {}", path.display()))?;
    Ok(b)
}

pub struct Validated {
    pub form: ExpertLoraForm,
    pub rank: usize,
    pub modules: usize,
    pub expert_layers: Vec<usize>,
    pub tensor_bytes: usize,
}

impl HostStagedAdapter {
    pub fn read(dir: &Path, stream: &Arc<CudaStream>) -> Result<Self> {
        let config_json = std::fs::read_to_string(dir.join("adapter_config.json")).with_context(|| format!("read {}/adapter_config.json", dir.display()))?;
        let config: serde_json::Value = serde_json::from_str(&config_json).context("adapter_config.json")?;
        let peft = read_pinned(stream, &dir.join("adapter_model.safetensors"), "lora adapter staging")?;
        let expert = if dir.join(expert::EXPERT_LORA_META).exists() || dir.join(expert::EXPERT_LORA_FILE).exists() {
            let meta: ExpertLoraMeta = serde_json::from_str(&std::fs::read_to_string(dir.join(expert::EXPERT_LORA_META)).with_context(|| format!("{} without {}", expert::EXPERT_LORA_FILE, expert::EXPERT_LORA_META))?)
                .context("expert_lora.json")?;
            Some((meta, read_pinned(stream, &dir.join(expert::EXPERT_LORA_FILE), "expert lora staging")?))
        } else {
            None
        };
        let meta_json = expert.as_ref().map(|(m, _)| serde_json::to_string(m)).transpose()?.unwrap_or_default();
        let version = expert::content_hash(&[
            config_json.as_bytes(),
            peft.as_slice(),
            meta_json.as_bytes(),
            expert.as_ref().map(|(_, b)| b.as_slice()).unwrap_or(&[]),
        ]);
        Ok(Self { dir: std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf()), version, config, config_json, peft, expert })
    }

    pub fn host_bytes(&self) -> usize {
        self.peft.len() + self.expert.as_ref().map_or(0, |(_, b)| b.len())
    }

    pub fn declared_form(&self) -> Result<ExpertLoraForm> {
        ExpertLoraForm::parse(self.config.get("expert_lora").and_then(|v| v.as_str()).unwrap_or("none"))
    }

    pub fn validate(&self, model: &Qwen4Exp, base_identity: Option<&str>) -> Result<Validated> {
        let cfg = model.config();
        let peft = nv_weights::lora_adapter::PeftConfig::from_json_str(&self.config_json)?;
        if let (Some(want), Some(have)) = (self.config.get("base_identity").and_then(|v| v.as_str()), base_identity) {
            ensure!(want == have, "adapter was trained against base {want}, this engine serves {have}");
        }
        let declared = self.declared_form()?;
        let targets = model.lora_target_linears();
        let index = safetensors_index(self.peft.as_slice(), "adapter_model.safetensors")?;
        let mut pairs: BTreeMap<String, [Option<&StTensor>; 2]> = BTreeMap::new();
        for (name, t) in &index {
            let (module, is_a, is_embedding) = nv_weights::lora_adapter::parse_fine_tuned_lora_name(name)?;
            ensure!(!is_embedding, "{name}: embedding LoRA has no qwen4_exp projection");
            ensure!(t.dtype == safetensors::Dtype::BF16, "{name}: dtype {:?}, the trainer exports bf16", t.dtype);
            let module = nv_weights::lora_adapter::normalize_module_name(&module);
            pairs.entry(module).or_default()[if is_a { 0 } else { 1 }] = Some(t);
        }
        let mut tensor_bytes = 0usize;
        for (module, [a, b]) in &pairs {
            let (a, b) = (a.with_context(|| format!("{module}: no lora_A"))?, b.with_context(|| format!("{module}: no lora_B"))?);
            let (_, _, _, lin) = targets
                .iter()
                .find(|(n, ..)| n == module)
                .with_context(|| format!("{module}: not a qwen4_exp runtime-LoRA target"))?;
            ensure!(
                a.shape == [peft.r, lin.in_features()] && b.shape == [lin.out_features(), peft.r],
                "{module}: lora_A {:?} / lora_B {:?}, expected [{}, {}] / [{}, {}] at rank {}",
                a.shape,
                b.shape,
                peft.r,
                lin.in_features(),
                lin.out_features(),
                peft.r,
                peft.r
            );
            tensor_bytes += a.range.1 - a.range.0 + b.range.1 - b.range.0;
        }
        let mut form = ExpertLoraForm::None;
        let mut expert_layers = Vec::new();
        if let Some((meta, bytes)) = &self.expert {
            form = meta.validate_against(cfg, base_identity)?;
            ensure!(form == declared, "adapter_config.json declares expert_lora {:?}, expert_lora.json holds {:?}", declared.as_str(), form.as_str());
            let index = safetensors_index(bytes.as_slice(), expert::EXPERT_LORA_FILE)?;
            ensure!(index.len() == 2 * meta.layers.len(), "{}: {} tensors for {} layers", expert::EXPERT_LORA_FILE, index.len(), meta.layers.len());
            for &li in &meta.layers {
                let m = expert::expert_module(li);
                let a = index.get(&format!("{m}.lora_A")).with_context(|| format!("{m}.lora_A missing"))?;
                let b = index.get(&format!("{m}.lora_B")).with_context(|| format!("{m}.lora_B missing"))?;
                ensure!(a.dtype == safetensors::Dtype::BF16 && b.dtype == safetensors::Dtype::BF16, "{m}: expert lora tensors must be bf16");
                meta.check_tensors(form, li, &a.shape, &b.shape)?;
                tensor_bytes += a.range.1 - a.range.0 + b.range.1 - b.range.0;
            }
            expert_layers = meta.layers.clone();
        } else {
            ensure!(declared == ExpertLoraForm::None, "adapter_config.json declares expert_lora {:?} but the directory has no {}", declared.as_str(), expert::EXPERT_LORA_FILE);
        }
        ensure!(!pairs.is_empty() || !expert_layers.is_empty(), "adapter carries no module and no expert lora");
        Ok(Validated { form, rank: peft.r, modules: pairs.len(), expert_layers, tensor_bytes })
    }

    fn upload_bf16(bytes: &[u8], t: &StTensor, device: &Device, compute: &Arc<CudaStream>, copy: &Arc<CudaStream>) -> Result<(Tensor, cudarc::driver::CudaEvent)> {
        let Device::Cuda(dev) = device else { bail!("adapter upload: not cuda") };
        let (slice, ev) = upload_pinned_bytes_on::<half::bf16>(&bytes[t.range.0..t.range.1], compute, copy)?;
        let tensor = Tensor::from_storage(
            candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(slice, dev.clone())),
            t.shape.as_slice(),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok((tensor, ev))
    }

    pub fn to_device(&self, model: &Qwen4Exp, copy: &Arc<CudaStream>) -> Result<Qwen4ExpServingAdapter> {
        let device = model.device();
        let Device::Cuda(dev) = device else { bail!("adapter upload: not cuda") };
        let compute = nv_layers::cuda_stream::current_stream(dev);
        let peft = nv_weights::lora_adapter::PeftConfig::from_json_str(&self.config_json)?;
        let mut last = None;
        let mut pending: BTreeMap<String, [Option<Tensor>; 2]> = BTreeMap::new();
        for (name, t) in safetensors_index(self.peft.as_slice(), "adapter_model.safetensors")? {
            let (module, is_a, _) = nv_weights::lora_adapter::parse_fine_tuned_lora_name(&name)?;
            let (tensor, ev) = Self::upload_bf16(self.peft.as_slice(), &t, device, &compute, copy)?;
            last = Some(ev);
            pending.entry(nv_weights::lora_adapter::normalize_module_name(&module)).or_default()[if is_a { 0 } else { 1 }] = Some(tensor);
        }
        let mut experts = Vec::new();
        if let Some((meta, bytes)) = &self.expert {
            let index = safetensors_index(bytes.as_slice(), expert::EXPERT_LORA_FILE)?;
            for &li in &meta.layers {
                let m = expert::expert_module(li);
                let (a, _) = Self::upload_bf16(bytes.as_slice(), &index[&format!("{m}.lora_A")], device, &compute, copy)?;
                let (b, ev) = Self::upload_bf16(bytes.as_slice(), &index[&format!("{m}.lora_B")], device, &compute, copy)?;
                last = Some(ev);
                experts.push((li, a, b));
            }
        }
        if let Some(ev) = last {
            compute.wait(&ev)?;
        }
        let mut loras = BTreeMap::new();
        for (module, [a, b]) in pending {
            let mut w = nv_weights::lora_adapter::LoraLayerWeights {
                module_name: module.clone(),
                rank: peft.r,
                lora_a: a.with_context(|| format!("{module}: no lora_A"))?,
                lora_b: b.with_context(|| format!("{module}: no lora_B"))?,
                scaling: peft.scaling,
                is_embedding: false,
            };
            w.optimize()?;
            loras.insert(module, w);
        }
        let adapter = nv_weights::lora_adapter::LoraAdapter { config: peft, loras };
        let experts = experts
            .into_iter()
            .map(|(li, a, b)| Ok((li, Arc::new(ExpertLoraLayer::new(a, b, 1.0)?))))
            .collect::<Result<Vec<_>>>()?;
        Qwen4ExpServingAdapter::prepare_with_experts(model, &adapter, experts)
    }
}

pub struct AdapterVersion {
    pub id: String,
    pub version: String,
    pub serving: Qwen4ExpServingAdapter,
    pub form: ExpertLoraForm,
    pub rank: usize,
    pub tensor_bytes: usize,
    pub device_bytes: usize,
    host: Arc<HostStagedAdapter>,
}

impl AdapterVersion {
    pub fn tag(&self) -> String {
        format!("{}@{}", self.id, self.version)
    }

    pub fn dir(&self) -> &Path {
        &self.host.dir
    }
}

pub struct AdapterRow {
    pub id: String,
    pub version: String,
    pub form: &'static str,
    pub rank: usize,
    pub device_bytes: usize,
    pub host_bytes: usize,
    pub in_flight: usize,
}

pub struct Qwen4ExpAdapterRegistry {
    model: Arc<Qwen4Exp>,
    live: RwLock<BTreeMap<String, Arc<AdapterVersion>>>,
    parked: Mutex<VecDeque<Arc<HostStagedAdapter>>>,
    park_budget: PinnedBudget,
    min_free_vram: u64,
    copy: Arc<CudaStream>,
    load_lock: Mutex<()>,
}

fn free_vram() -> u64 {
    cudarc::driver::result::mem_get_info().map(|(f, _)| f as u64).unwrap_or(0)
}

impl Qwen4ExpAdapterRegistry {
    pub fn new(model: Arc<Qwen4Exp>, park_budget_bytes: usize, min_free_vram: u64) -> Result<Self> {
        let Device::Cuda(dev) = model.device() else { bail!("qwen4_exp adapter registry needs CUDA") };
        let copy = nv_layers::cuda_stream::current_stream(dev).fork()?;
        if park_budget_bytes > 0 {
            drop(PinnedHostBuf::alloc(&copy, 1 << 20, "adapter host park probe")?);
        }
        Ok(Self {
            model,
            live: RwLock::new(BTreeMap::new()),
            parked: Mutex::new(VecDeque::new()),
            park_budget: PinnedBudget::new(park_budget_bytes),
            min_free_vram,
            copy,
            load_lock: Mutex::new(()),
        })
    }

    pub fn park_budget_from_env() -> usize {
        PinnedBudget::from_env_gib(NV_Q4EXP_LORA_HOST_PARK_GIB, HOST_PARK_DEFAULT_1_GIB_HOLDS_A_FEW_REPLACED_ADAPTER_VERSIONS_FOR_ROLLBACK_WITHOUT_PRESSURING_A_HOST_THAT_ALREADY_PINS_THE_PLE_TABLE).cap()
    }

    pub fn model(&self) -> &Arc<Qwen4Exp> {
        &self.model
    }

    fn unpark(&self, dir: &Path) -> Option<Arc<HostStagedAdapter>> {
        let canon = std::fs::canonicalize(dir).ok()?;
        let mut parked = self.parked.lock().ok()?;
        let i = parked.iter().position(|h| h.dir == canon)?;
        let h = parked.remove(i)?;
        self.park_budget.give_back(h.host_bytes());
        Some(h)
    }

    fn park(&self, h: Arc<HostStagedAdapter>) {
        let bytes = h.host_bytes();
        let Ok(mut parked) = self.parked.lock() else { return };
        parked.retain(|p| p.dir != h.dir || p.version != h.version);
        while !self.park_budget.try_take(bytes) {
            let Some(old) = parked.pop_front() else { return };
            self.park_budget.give_back(old.host_bytes());
        }
        parked.push_back(h);
    }

    pub fn parked_bytes(&self) -> usize {
        self.park_budget.used()
    }

    pub fn parked_versions(&self) -> Vec<(PathBuf, String)> {
        self.parked.lock().map(|p| p.iter().map(|h| (h.dir.clone(), h.version.clone())).collect()).unwrap_or_default()
    }

    pub fn load(&self, id: &str, dir: &Path) -> Result<Arc<AdapterVersion>> {
        let _ = HOT_SWAP_STAGES_IN_PINNED_HOST_RAM_VALIDATES_THERE_UPLOADS_ON_A_COPY_STREAM_THEN_SWAPS_ONE_POINTER;
        let _serial = self.load_lock.lock().map_err(|e| anyhow::anyhow!("adapter load lock poisoned: {e}"))?;
        let parked = self.unpark(dir);
        let from_park = parked.is_some();
        let host = match parked {
            Some(h) => h,
            None => Arc::new(HostStagedAdapter::read(dir, &self.copy)?),
        };
        let checked = host.validate(&self.model, self.model.base_identity());
        let v = match checked {
            Ok(v) => v,
            Err(e) => {
                if from_park {
                    self.park(host);
                }
                return Err(e.context(format!("adapter {id:?} at {} refused; the live version keeps serving", dir.display())));
            }
        };
        let need = (v.tensor_bytes * ADAPTER_DEVICE_ESTIMATE_3X_THE_BF16_FILE_BYTES_COVERS_THE_F32_PREFILL_FACTORS_AND_THE_M1_KERNEL_COPY) as u64;
        let free = free_vram();
        ensure!(
            free >= need + self.min_free_vram,
            "adapter {id:?} needs ~{need} B of VRAM above the engine's {} B floor and {free} B are free; the live version keeps serving",
            self.min_free_vram
        );
        let Device::Cuda(dev) = self.model.device() else { bail!("not cuda") };
        let compute = nv_layers::cuda_stream::current_stream(dev);
        compute.synchronize()?;
        let before = free_vram();
        let serving = host.to_device(&self.model, &self.copy)?;
        compute.synchronize()?;
        let device_bytes = (before.saturating_sub(free_vram()) as usize).max(v.tensor_bytes);
        let version = Arc::new(AdapterVersion {
            id: id.to_string(),
            version: host.version.clone(),
            serving,
            form: v.form,
            rank: v.rank,
            tensor_bytes: v.tensor_bytes,
            device_bytes,
            host,
        });
        let old = self.live.write().map_err(|e| anyhow::anyhow!("adapter registry poisoned: {e}"))?.insert(id.to_string(), version.clone());
        if let Some(old) = old {
            if old.host.version != version.host.version {
                self.park(old.host.clone());
            }
        }
        Ok(version)
    }

    pub fn current(&self, id: &str) -> Option<Arc<AdapterVersion>> {
        self.live.read().ok()?.get(id).cloned()
    }

    pub fn unload(&self, id: &str) -> Option<Arc<AdapterVersion>> {
        let gone = self.live.write().ok()?.remove(id)?;
        self.park(gone.host.clone());
        Some(gone)
    }

    pub fn ids(&self) -> Vec<String> {
        self.live.read().map(|l| l.keys().cloned().collect()).unwrap_or_default()
    }

    pub fn rows(&self) -> Vec<AdapterRow> {
        self.live
            .read()
            .map(|l| {
                l.values()
                    .map(|v| AdapterRow {
                        id: v.id.clone(),
                        version: v.version.clone(),
                        form: v.form.as_str(),
                        rank: v.rank,
                        device_bytes: v.device_bytes,
                        host_bytes: v.host.host_bytes(),
                        in_flight: Arc::strong_count(v) - 1,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}
