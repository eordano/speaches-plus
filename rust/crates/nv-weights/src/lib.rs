pub mod gguf;
pub mod lora_adapter;

#[macro_export]
macro_rules! hf_json_from_file {
    ($file_fn:ident, $str_fn:ident) => {
        pub fn $file_fn(p: &::std::path::Path) -> ::anyhow::Result<Self> {
            let raw = ::anyhow::Context::with_context(::std::fs::read_to_string(p), || {
                format!("read {}", p.display())
            })?;
            Self::$str_fn(&raw)
        }
    };
    ($file_fn:ident, $str_fn:ident, as_ref) => {
        pub fn $file_fn(p: impl ::std::convert::AsRef<::std::path::Path>) -> ::anyhow::Result<Self> {
            let raw =
                ::anyhow::Context::with_context(::std::fs::read_to_string(p.as_ref()), || {
                    format!("read {}", p.as_ref().display())
                })?;
            Self::$str_fn(&raw)
        }
    };
}

pub use gguf::GgufLoader;

use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

pub use candle_core::{DType, Device, Tensor};
pub use safetensors::tensor::{Dtype as StDtype, TensorView};
pub use safetensors::SafeTensors;

struct Shard {
    path: PathBuf,
    bytes: Mmap,
    headers: HashMap<String, TensorHeader>,
}

#[derive(Clone)]
struct TensorHeader {
    dtype: StDtype,
    shape: Vec<usize>,
    offset_start: usize,
    offset_end: usize,
}

pub struct WeightLoader {
    shards: Vec<Shard>,
    name_to_shard: HashMap<String, usize>,
    device: Device,
}

pub trait TensorSource {
    fn get(&self, name: &str, dtype: DType) -> Result<Tensor>;
    fn has(&self, name: &str) -> bool;
}

impl TensorSource for WeightLoader {
    fn get(&self, name: &str, dtype: DType) -> Result<Tensor> {
        WeightLoader::get(self, name, dtype)
    }
    fn has(&self, name: &str) -> bool {
        WeightLoader::has(self, name)
    }
}

impl WeightLoader {
    pub fn open_file(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let shard = load_shard(&path)?;
        let mut name_to_shard = HashMap::with_capacity(shard.headers.len());
        for name in shard.headers.keys() {
            name_to_shard.insert(name.clone(), 0usize);
        }
        Ok(Self {
            shards: vec![shard],
            name_to_shard,
            device: device.clone(),
        })
    }

    pub fn open_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            bail!("not a directory: {}", dir.display());
        }
        let index_path = dir.join("model.safetensors.index.json");
        let (shard_paths, weight_map) = if index_path.is_file() {
            let raw = std::fs::read(&index_path)
                .with_context(|| format!("read {}", index_path.display()))?;
            let parsed: SafetensorsIndex = serde_json::from_slice(&raw)
                .with_context(|| format!("parse {}", index_path.display()))?;
            let mut unique: Vec<PathBuf> = parsed
                .weight_map
                .values()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .map(|f| dir.join(f))
                .collect();
            unique.sort();
            (unique, Some(parsed.weight_map))
        } else {
            let mut files = Vec::new();
            for entry in std::fs::read_dir(dir)? {
                let path = entry?.path();
                if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                    files.push(path);
                }
            }
            files.sort();
            if files.is_empty() {
                bail!("no .safetensors files in {}", dir.display());
            }
            (files, None)
        };

        let mut shards = Vec::with_capacity(shard_paths.len());
        let mut path_to_idx: HashMap<PathBuf, usize> = HashMap::new();
        let mut skipped_paths: HashSet<PathBuf> = HashSet::new();
        for p in shard_paths.iter() {
            if is_lfs_pointer(p)? {
                eprintln!("nv-weights: skipping LFS pointer shard {}", p.display());
                skipped_paths.insert(p.clone());
                continue;
            }
            let idx = shards.len();
            shards.push(load_shard(p)?);
            path_to_idx.insert(p.clone(), idx);
        }

        let mut name_to_shard: HashMap<String, usize> = HashMap::new();
        if let Some(map) = weight_map {
            for (name, file) in map {
                let full = dir.join(&file);
                if skipped_paths.contains(&full) {
                    continue;
                }
                let idx = *path_to_idx
                    .get(&full)
                    .ok_or_else(|| anyhow!("index references missing shard {}", full.display()))?;
                if !shards[idx].headers.contains_key(&name) {
                    bail!("shard {} missing tensor {}", full.display(), name);
                }
                name_to_shard.insert(name, idx);
            }
        } else {
            for (idx, shard) in shards.iter().enumerate() {
                for name in shard.headers.keys() {
                    if let Some(prev) = name_to_shard.insert(name.clone(), idx) {
                        bail!(
                            "duplicate tensor {} in shards {} and {}",
                            name,
                            shards[prev].path.display(),
                            shard.path.display()
                        );
                    }
                }
            }
        }

        Ok(Self {
            shards,
            name_to_shard,
            device: device.clone(),
        })
    }

    pub fn get(&self, name: &str, dtype: DType) -> Result<Tensor> {
        let shard_idx = *self
            .name_to_shard
            .get(name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
        let shard = &self.shards[shard_idx];
        let header = shard
            .headers
            .get(name)
            .ok_or_else(|| anyhow!("tensor not found in shard: {name}"))?;
        let bytes = &shard.bytes[header.offset_start..header.offset_end];

        if header.dtype == StDtype::BOOL {
            let tensor = Tensor::from_raw_buffer(bytes, DType::U8, &header.shape, &self.device)
                .map_err(|e| anyhow!("from_raw_buffer for {name}: {e}"))?;
            return if dtype == DType::U8 {
                Ok(tensor)
            } else {
                tensor
                    .to_dtype(dtype)
                    .map_err(|e| anyhow!("to_dtype({dtype:?}) for {name}: {e}"))
            };
        }

        let src_dtype = map_st_to_candle(header.dtype)?;
        let tensor = Tensor::from_raw_buffer(bytes, src_dtype, &header.shape, &self.device)
            .map_err(|e| anyhow!("from_raw_buffer for {name}: {e}"))?;
        if src_dtype == dtype {
            Ok(tensor)
        } else {
            tensor
                .to_dtype(dtype)
                .map_err(|e| anyhow!("to_dtype({dtype:?}) for {name}: {e}"))
        }
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.name_to_shard.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn has(&self, name: &str) -> bool {
        self.name_to_shard.contains_key(name)
    }

    pub fn dtype_of(&self, name: &str) -> Option<DType> {
        let idx = *self.name_to_shard.get(name)?;
        let h = self.shards[idx].headers.get(name)?;
        if h.dtype == StDtype::BOOL {
            return Some(DType::U8);
        }
        map_st_to_candle(h.dtype).ok()
    }

    pub fn shape_of(&self, name: &str) -> Option<Vec<usize>> {
        let idx = *self.name_to_shard.get(name)?;
        let h = self.shards[idx].headers.get(name)?;
        Some(h.shape.clone())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

fn load_shard(path: &Path) -> Result<Shard> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let bytes = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
    let view = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parse safetensors header in {}", path.display()))?;
    let buf_ptr = bytes.as_ptr() as usize;
    let mut headers: HashMap<String, TensorHeader> = HashMap::with_capacity(view.len());
    for (name, tv) in view.iter() {
        let data_ptr = tv.data().as_ptr() as usize;
        let offset = data_ptr - buf_ptr;
        headers.insert(
            name.to_string(),
            TensorHeader {
                dtype: tv.dtype(),
                shape: tv.shape().to_vec(),
                offset_start: offset,
                offset_end: offset + tv.data().len(),
            },
        );
    }
    Ok(Shard {
        path: path.to_path_buf(),
        bytes,
        headers,
    })
}

fn is_lfs_pointer(path: &Path) -> Result<bool> {
    let meta = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if meta.len() > 1024 {
        return Ok(false);
    }
    let mut buf = [0u8; 64];
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let n = f.read(&mut buf).unwrap_or(0);
    Ok(buf[..n].starts_with(b"version https://git-lfs.github.com/spec/"))
}

fn map_st_to_candle(d: StDtype) -> Result<DType> {
    Ok(match d {
        StDtype::BF16 => DType::BF16,
        StDtype::F16 => DType::F16,
        StDtype::F32 => DType::F32,
        StDtype::I64 => DType::I64,
        StDtype::U8 => DType::U8,

        other => bail!("unsupported source dtype {:?}", other),
    })
}

#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

#[derive(Clone, Copy, Debug)]
pub struct TpSlice {
    pub rank: usize,
    pub world: usize,
    pub axis: usize,
}

impl TpSlice {
    pub fn single() -> Self {
        Self {
            rank: 0,
            world: 1,
            axis: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantScheme {
    None,
    Fp8E4m3,
    Nvfp4,
}

#[derive(Clone, Debug)]
pub struct QuantizationConfig {
    pub scheme: QuantScheme,
    pub ignored_modules: Vec<String>,
    pub group_size: Option<usize>,
}

pub const NVFP4_GROUP_SIZE: usize = nv_quant::nvfp4::BLOCK_SIZE;

impl QuantizationConfig {
    pub fn implemented_group_size(scheme: QuantScheme) -> Option<usize> {
        match scheme {
            QuantScheme::None => None,
            QuantScheme::Nvfp4 => Some(NVFP4_GROUP_SIZE),
            QuantScheme::Fp8E4m3 => None,
        }
    }

    pub fn effective_group_size(&self) -> Option<usize> {
        self.group_size
            .or_else(|| Self::implemented_group_size(self.scheme))
    }

    pub fn validate(&self) -> Result<()> {
        let declared = match self.group_size {
            None => return Ok(()),
            Some(g) => g,
        };
        match self.scheme {
            QuantScheme::None => Ok(()),
            QuantScheme::Nvfp4 => {
                if declared == NVFP4_GROUP_SIZE {
                    Ok(())
                } else {
                    bail!(
                        "quantization_config declares group_size {declared} for 4-bit float \
                         weights, but this build implements NVFP4 with exactly \
                         {NVFP4_GROUP_SIZE} elements per ue4m3 scale (nv_quant::nvfp4::BLOCK_SIZE \
                         and cuBLASLt VEC16_UE4M3). group_size 32 with 4-bit weights is MXFP4 \
                         (per-32 ue8m0 scales), which is a different format, not a variant of \
                         this one. Loading it as NVFP4 would mis-scale every block. Refusing."
                    )
                }
            }
            QuantScheme::Fp8E4m3 => bail!(
                "quantization_config declares group_size {declared} for FP8 weights, but this \
                 build scales FP8 per output row (and per token on the activation side), not \
                 per {declared}-element block. Block-wise FP8 is not implemented; loading this \
                 checkpoint would apply the wrong granularity silently. Refusing."
            ),
        }
    }

    pub fn none() -> Self {
        Self {
            scheme: QuantScheme::None,
            ignored_modules: Vec::new(),
            group_size: None,
        }
    }

    pub fn from_hf_value(v: &serde_json::Value) -> Result<Self> {
        let parsed = Self::parse_hf_value(v)?;
        parsed.validate()?;
        Ok(parsed)
    }

    fn parse_hf_value(v: &serde_json::Value) -> Result<Self> {
        let obj = match v {
            serde_json::Value::Null => return Ok(Self::none()),
            serde_json::Value::Object(o) => o,
            _ => bail!("quantization_config must be an object"),
        };

        let ignored_modules: Vec<String> =
            match obj.get("ignore").or_else(|| obj.get("ignored_modules")) {
                Some(serde_json::Value::Array(arr)) => arr
                    .iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect(),
                _ => Vec::new(),
            };

        if let Some(method) = obj.get("quant_method").and_then(|x| x.as_str()) {
            if method == "modelopt" {
                if let Some(algo) = obj.get("quant_algo").and_then(|x| x.as_str()) {
                    let upper = algo.to_ascii_uppercase();
                    if upper.contains("NVFP4") || upper.contains("FP4") {
                        let group_size = detect_group_size(obj);
                        return Ok(Self {
                            scheme: QuantScheme::Nvfp4,
                            ignored_modules,
                            group_size,
                        });
                    }
                    if upper.contains("FP8") {
                        return Ok(Self {
                            scheme: QuantScheme::Fp8E4m3,
                            ignored_modules,
                            group_size: detect_group_size(obj),
                        });
                    }
                }
            }
        }

        let format = obj.get("format").and_then(|x| x.as_str()).unwrap_or("");
        let quant_type = obj
            .get("quantization_type")
            .and_then(|x| x.as_str())
            .unwrap_or("");
        let fmt_lower = format.to_ascii_lowercase();
        if fmt_lower.contains("nvfp4") || fmt_lower.contains("fp4") {
            let group_size = detect_group_size(obj);
            return Ok(Self {
                scheme: QuantScheme::Nvfp4,
                ignored_modules,
                group_size,
            });
        }
        if quant_type.eq_ignore_ascii_case("fp8") || fmt_lower.contains("fp8") {
            return Ok(Self {
                scheme: QuantScheme::Fp8E4m3,
                ignored_modules,
                group_size: detect_group_size(obj),
            });
        }

        if let Some(groups) = obj.get("config_groups").and_then(|x| x.as_object()) {
            for group in groups.values() {
                if let Some(weights) = group.get("weights") {
                    let num_bits = weights
                        .get("num_bits")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(0);
                    let typ = weights.get("type").and_then(|x| x.as_str()).unwrap_or("");
                    let group_size = weights
                        .get("group_size")
                        .and_then(|x| x.as_u64())
                        .map(|x| x as usize);
                    if typ.eq_ignore_ascii_case("float") && num_bits == 4 {
                        return Ok(Self {
                            scheme: QuantScheme::Nvfp4,
                            ignored_modules,
                            group_size,
                        });
                    }
                    if typ.eq_ignore_ascii_case("float") && num_bits == 8 {
                        return Ok(Self {
                            scheme: QuantScheme::Fp8E4m3,
                            ignored_modules,
                            group_size,
                        });
                    }
                }
            }
        }

        Ok(Self::none())
    }

    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s).context("parse config.json")?;
        let q = v
            .get("quantization_config")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        Self::from_hf_value(&q)
    }

    crate::hf_json_from_file!(from_hf_json_file, from_hf_json_str, as_ref);

    pub fn is_module_ignored(&self, name: &str) -> bool {
        for pat in &self.ignored_modules {
            if module_pattern_matches(pat, name) {
                return true;
            }
        }
        false
    }
}

fn detect_group_size(obj: &serde_json::Map<String, serde_json::Value>) -> Option<usize> {
    if let Some(groups) = obj.get("config_groups").and_then(|x| x.as_object()) {
        for group in groups.values() {
            if let Some(weights) = group.get("weights") {
                if let Some(g) = weights.get("group_size").and_then(|x| x.as_u64()) {
                    return Some(g as usize);
                }
            }
        }
    }
    None
}

fn module_pattern_matches(pattern: &str, name: &str) -> bool {
    if let Some(stripped) = pattern.strip_suffix('*') {
        name.starts_with(stripped)
    } else {
        name == pattern
    }
}

pub struct QuantizedWeight {
    pub scheme: QuantScheme,
    pub shape: Vec<usize>,
    pub packed_bytes: Vec<u8>,

    pub packed_dtype: StDtype,
    pub weight_scale: Option<Tensor>,
    pub weight_scale_bytes: Option<Vec<u8>>,
    pub input_scale: Option<f32>,
}

impl QuantizedWeight {
    pub fn fp8_weight_scale_rows(&self) -> Result<Option<Vec<f32>>> {
        let Some(t) = self.weight_scale.as_ref() else {
            return Ok(None);
        };
        let out_features = *self
            .shape
            .first()
            .ok_or_else(|| anyhow!("quantized weight has no shape"))?;
        let dims = t.dims().to_vec();
        let vals: Vec<f32> = t
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| anyhow!("read weight_scale as f32: {e}"))?;
        fp8_row_scales_from(&dims, &vals, out_features).map(Some)
    }
}

pub fn fp8_row_scales_from(
    scale_dims: &[usize],
    scale_values: &[f32],
    out_features: usize,
) -> Result<Vec<f32>> {
    let elems: usize = scale_dims.iter().product();
    if elems != scale_values.len() {
        bail!(
            "weight_scale shape {scale_dims:?} implies {elems} values but {} were read",
            scale_values.len()
        );
    }
    let rows = match scale_values.len() {
        1 => vec![scale_values[0]; out_features],
        n if n == out_features => scale_values.to_vec(),
        n => bail!(
            "weight_scale has {n} values with shape {scale_dims:?}, which is neither a single \
             per-tensor scale nor one scale per output row ({out_features}). A per-column or \
             block-wise fp8 scale grid is a different numeric contract than the per-row scaling \
             this build applies to cuBLASLt's A operand; refusing to reinterpret it."
        ),
    };
    if let Some(bad) = rows.iter().find(|s| !s.is_finite() || **s <= 0.0) {
        bail!("weight_scale contains a non-positive or non-finite value: {bad}");
    }
    Ok(rows)
}

impl WeightLoader {
    pub fn raw_bytes(&self, name: &str) -> Result<&[u8]> {
        let shard_idx = *self
            .name_to_shard
            .get(name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
        let shard = &self.shards[shard_idx];
        let header = shard
            .headers
            .get(name)
            .ok_or_else(|| anyhow!("tensor not found in shard: {name}"))?;
        Ok(&shard.bytes[header.offset_start..header.offset_end])
    }

    pub fn st_dtype_of(&self, name: &str) -> Option<StDtype> {
        let idx = *self.name_to_shard.get(name)?;
        let h = self.shards[idx].headers.get(name)?;
        Some(h.dtype)
    }

    pub fn load_quantized_weight(
        &self,
        base_name: &str,
        scheme: QuantScheme,
    ) -> Result<QuantizedWeight> {
        match scheme {
            QuantScheme::None => bail!("load_quantized_weight called with scheme None"),
            QuantScheme::Fp8E4m3 => {
                let weight_bytes = self.raw_bytes(base_name)?.to_vec();
                let shape = self
                    .shape_of(base_name)
                    .ok_or_else(|| anyhow!("missing shape for {base_name}"))?;
                let scale_name_candidates = [
                    format!("{base_name}_scale"),
                    format!("{}.weight_scale", strip_weight_suffix(base_name)),
                    format!("{}.scale", strip_weight_suffix(base_name)),
                ];
                let mut weight_scale = None;
                for cand in &scale_name_candidates {
                    if self.has(cand) {
                        weight_scale = Some(self.get(cand, DType::F32)?);
                        break;
                    }
                }
                let input_scale_candidates = [
                    format!("{}.input_scale", strip_weight_suffix(base_name)),
                    format!("{}.act_scale", strip_weight_suffix(base_name)),
                ];
                let mut input_scale = None;
                for cand in &input_scale_candidates {
                    if self.has(cand) {
                        let t = self.get(cand, DType::F32)?;
                        if let Ok(v) = t.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                            if let Some(x) = v.first() {
                                input_scale = Some(*x);
                                break;
                            }
                        }
                    }
                }
                let packed_dtype = self
                    .st_dtype_of(base_name)
                    .ok_or_else(|| anyhow!("missing dtype for {base_name}"))?;
                let qw = QuantizedWeight {
                    scheme,
                    shape,
                    packed_bytes: weight_bytes,
                    packed_dtype,
                    weight_scale,
                    weight_scale_bytes: None,
                    input_scale,
                };

                qw.fp8_weight_scale_rows()
                    .with_context(|| format!("{base_name}: checkpoint weight_scale"))?;
                Ok(qw)
            }
            QuantScheme::Nvfp4 => {
                let packed_bytes = self.raw_bytes(base_name)?.to_vec();
                let shape = self
                    .shape_of(base_name)
                    .ok_or_else(|| anyhow!("missing shape for {base_name}"))?;
                let scale_name_candidates = [
                    format!("{}.weight_scale", strip_weight_suffix(base_name)),
                    format!("{base_name}_scale"),
                ];
                let mut weight_scale_bytes = None;
                for cand in &scale_name_candidates {
                    if self.has(cand) {
                        weight_scale_bytes = Some(self.raw_bytes(cand)?.to_vec());
                        break;
                    }
                }
                let packed_dtype = self
                    .st_dtype_of(base_name)
                    .ok_or_else(|| anyhow!("missing dtype for {base_name}"))?;
                Ok(QuantizedWeight {
                    scheme,
                    shape,
                    packed_bytes,
                    packed_dtype,
                    weight_scale: None,
                    weight_scale_bytes,
                    input_scale: None,
                })
            }
        }
    }
}

fn strip_weight_suffix(name: &str) -> &str {
    name.strip_suffix(".weight").unwrap_or(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_safetensors_one(
        name: &str,
        dtype: StDtype,
        shape: &[usize],
        payload: &[u8],
    ) -> std::path::PathBuf {
        let dtype_str = match dtype {
            StDtype::I64 => "I64",
            StDtype::BOOL => "BOOL",
            StDtype::U8 => "U8",
            StDtype::F32 => "F32",
            StDtype::BF16 => "BF16",
            StDtype::F16 => "F16",
            _ => panic!("unsupported test dtype {:?}", dtype),
        };

        let hdr = format!(
            "{{\"{name}\":{{\"dtype\":\"{dtype_str}\",\"shape\":{shape:?},\"data_offsets\":[0,{}]}}}}",
            payload.len()
        );

        let mut hdr_bytes = hdr.into_bytes();
        while hdr_bytes.len() % 8 != 0 {
            hdr_bytes.push(b' ');
        }
        let hdr_len = (hdr_bytes.len() as u64).to_le_bytes();

        let dir = std::env::temp_dir().join(format!(
            "nv-weights-test-{}-{}",
            std::process::id(),
            payload
                .iter()
                .take(8)
                .fold(0u64, |a, b| a.wrapping_mul(131).wrapping_add(*b as u64))
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&hdr_len).unwrap();
        f.write_all(&hdr_bytes).unwrap();
        f.write_all(payload).unwrap();
        path
    }

    #[test]
    fn i64_roundtrip_through_get() {
        let values: Vec<i64> = vec![0, 1, -1, 42, -42, i64::MAX, i64::MIN];
        let mut payload = Vec::with_capacity(values.len() * 8);
        for v in &values {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let path = write_safetensors_one("d2t", StDtype::I64, &[values.len()], &payload);
        let loader = WeightLoader::open_file(&path, &Device::Cpu).unwrap();

        assert_eq!(loader.dtype_of("d2t"), Some(DType::I64));

        let t = loader.get("d2t", DType::I64).expect("get I64");
        assert_eq!(t.dtype(), DType::I64);
        assert_eq!(t.dims(), &[values.len()]);
        let got: Vec<i64> = t.to_vec1().unwrap();
        assert_eq!(got, values);
    }

    fn write_safetensors_many(
        tensors: &[(&str, StDtype, Vec<usize>, Vec<u8>)],
        tag: &str,
    ) -> std::path::PathBuf {
        let dtype_str = |d: StDtype| match d {
            StDtype::I64 => "I64",
            StDtype::BOOL => "BOOL",
            StDtype::U8 => "U8",
            StDtype::F32 => "F32",
            StDtype::BF16 => "BF16",
            StDtype::F16 => "F16",
            other => panic!("unsupported test dtype {other:?}"),
        };
        let mut entries = Vec::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = payload.len();
            payload.extend_from_slice(bytes);
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"{}\",\"shape\":{shape:?},\"data_offsets\":[{start},{}]}}",
                dtype_str(*dtype),
                payload.len()
            ));
        }
        let mut hdr_bytes = format!("{{{}}}", entries.join(",")).into_bytes();
        while hdr_bytes.len() % 8 != 0 {
            hdr_bytes.push(b' ');
        }
        let hdr_len = (hdr_bytes.len() as u64).to_le_bytes();

        let dir =
            std::env::temp_dir().join(format!("nv-weights-test-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&hdr_len).unwrap();
        f.write_all(&hdr_bytes).unwrap();
        f.write_all(&payload).unwrap();
        path
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn fp8_checkpoint_weight_and_input_scales_are_loaded_not_discarded() {
        let (out_features, in_features) = (4usize, 8usize);
        let w: Vec<u8> = (0..out_features * in_features)
            .map(|i| (i % 251) as u8)
            .collect();
        let row_scales: Vec<f32> = vec![0.25, 0.5, 1.0, 2.0];
        let path = write_safetensors_many(
            &[
                (
                    "blk.weight",
                    StDtype::U8,
                    vec![out_features, in_features],
                    w.clone(),
                ),
                (
                    "blk.weight_scale",
                    StDtype::F32,
                    vec![out_features],
                    f32_bytes(&row_scales),
                ),
                (
                    "blk.input_scale",
                    StDtype::F32,
                    vec![1],
                    f32_bytes(&[0.125]),
                ),
            ],
            "fp8-scales",
        );
        let loader = WeightLoader::open_file(&path, &Device::Cpu).unwrap();
        let qw = loader
            .load_quantized_weight("blk.weight", QuantScheme::Fp8E4m3)
            .expect("load fp8 quantized weight");

        assert_eq!(qw.shape, vec![out_features, in_features]);
        assert_eq!(qw.packed_bytes, w);
        assert_eq!(qw.packed_dtype, StDtype::U8);
        assert_eq!(qw.input_scale, Some(0.125));
        assert!(
            qw.weight_scale.is_some(),
            "checkpoint weight_scale must survive load"
        );
        assert_eq!(
            qw.fp8_weight_scale_rows().unwrap(),
            Some(row_scales),
            "per-row checkpoint scales must reach the consumer verbatim"
        );
    }

    #[test]
    fn fp8_scalar_checkpoint_scale_broadcasts_to_every_output_row() {
        let (out_features, in_features) = (3usize, 4usize);
        let path = write_safetensors_many(
            &[
                (
                    "blk.weight",
                    StDtype::U8,
                    vec![out_features, in_features],
                    vec![7u8; out_features * in_features],
                ),
                ("blk.weight_scale", StDtype::F32, vec![], f32_bytes(&[0.75])),
            ],
            "fp8-scalar-scale",
        );
        let loader = WeightLoader::open_file(&path, &Device::Cpu).unwrap();
        let qw = loader
            .load_quantized_weight("blk.weight", QuantScheme::Fp8E4m3)
            .unwrap();
        assert_eq!(
            qw.fp8_weight_scale_rows().unwrap(),
            Some(vec![0.75; out_features])
        );
    }

    #[test]
    fn fp8_row_scale_canonicalisation_covers_the_shapes_and_rejects_the_rest() {
        assert_eq!(fp8_row_scales_from(&[], &[0.5], 3).unwrap(), vec![0.5; 3]);
        assert_eq!(fp8_row_scales_from(&[1], &[0.5], 3).unwrap(), vec![0.5; 3]);
        assert_eq!(
            fp8_row_scales_from(&[1, 1], &[0.5], 3).unwrap(),
            vec![0.5; 3]
        );
        assert_eq!(
            fp8_row_scales_from(&[3], &[0.5, 1.0, 2.0], 3).unwrap(),
            vec![0.5, 1.0, 2.0]
        );
        assert_eq!(
            fp8_row_scales_from(&[3, 1], &[0.5, 1.0, 2.0], 3).unwrap(),
            vec![0.5, 1.0, 2.0]
        );

        let err = fp8_row_scales_from(&[3, 2], &[1.0; 6], 3)
            .unwrap_err()
            .to_string();
        assert!(err.contains("block-wise"), "{err}");
        assert!(fp8_row_scales_from(&[1, 8], &[1.0; 8], 3).is_err());

        assert!(fp8_row_scales_from(&[3], &[1.0, 2.0], 3).is_err());

        assert!(fp8_row_scales_from(&[3], &[1.0, 0.0, 2.0], 3).is_err());
        assert!(fp8_row_scales_from(&[3], &[1.0, -1.0, 2.0], 3).is_err());
        assert!(fp8_row_scales_from(&[3], &[1.0, f32::NAN, 2.0], 3).is_err());
    }

    #[test]
    fn nvfp4_group_size_16_is_accepted_and_is_the_effective_group_size() {
        let json = r#"{
          "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
              "group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}
            }
          }
        }"#;
        let q = QuantizationConfig::from_hf_json_str(json).unwrap();
        assert_eq!(q.scheme, QuantScheme::Nvfp4);
        assert_eq!(q.group_size, Some(16));
        assert_eq!(q.effective_group_size(), Some(NVFP4_GROUP_SIZE));
    }

    #[test]
    fn nvfp4_without_a_declared_group_size_falls_back_to_the_implemented_one() {
        let json = r#"{
          "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}
        }"#;
        let q = QuantizationConfig::from_hf_json_str(json).unwrap();
        assert_eq!(q.scheme, QuantScheme::Nvfp4);
        assert_eq!(q.group_size, None);
        assert_eq!(q.effective_group_size(), Some(NVFP4_GROUP_SIZE));
    }

    #[test]
    fn nvfp4_group_size_32_is_rejected_rather_than_silently_treated_as_16() {
        let json = r#"{
          "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
              "group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 32}}
            }
          }
        }"#;
        let err = QuantizationConfig::from_hf_json_str(json)
            .expect_err("group_size 32 must not load as NVFP4")
            .to_string();
        assert!(err.contains("32"), "{err}");
        assert!(err.contains("MXFP4"), "{err}");
        assert!(err.contains(&NVFP4_GROUP_SIZE.to_string()), "{err}");
    }

    #[test]
    fn fp8_declared_group_size_is_rejected_rather_than_dropped_on_the_floor() {
        let json = r#"{
          "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
              "group_0": {"weights": {"num_bits": 8, "type": "float", "group_size": 128}}
            }
          }
        }"#;
        let err = QuantizationConfig::from_hf_json_str(json)
            .expect_err("block-wise fp8 must not load as per-tensor fp8")
            .to_string();
        assert!(err.contains("128"), "{err}");
        assert!(err.contains("per output row"), "{err}");

        let modelopt = r#"{
          "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "FP8",
            "config_groups": {"group_0": {"weights": {"group_size": 128}}}
          }
        }"#;
        assert!(QuantizationConfig::from_hf_json_str(modelopt).is_err());
    }

    #[test]
    fn plain_fp8_without_a_group_size_still_loads() {
        let json = r#"{
          "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "float"}}},
            "ignore": ["lm_head"]
          }
        }"#;
        let q = QuantizationConfig::from_hf_json_str(json).unwrap();
        assert_eq!(q.scheme, QuantScheme::Fp8E4m3);
        assert_eq!(q.group_size, None);
        assert_eq!(q.effective_group_size(), None);
        assert!(q.is_module_ignored("lm_head"));
    }

    #[test]
    fn bool_canonicalizes_to_u8() {
        let bytes: Vec<u8> = vec![1, 0, 1, 1, 0, 0];
        let path = write_safetensors_one("t2d", StDtype::BOOL, &[bytes.len()], &bytes);
        let loader = WeightLoader::open_file(&path, &Device::Cpu).unwrap();

        assert_eq!(loader.dtype_of("t2d"), Some(DType::U8));

        let t = loader.get("t2d", DType::U8).expect("get BOOL as U8");
        assert_eq!(t.dtype(), DType::U8);
        assert_eq!(t.dims(), &[bytes.len()]);
        let got: Vec<u8> = t.to_vec1().unwrap();
        assert_eq!(got, bytes);

        assert!(got.iter().all(|b| *b == 0 || *b == 1));
    }
}
