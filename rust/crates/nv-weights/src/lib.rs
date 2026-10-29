pub mod gguf;
pub mod lora_adapter;
#[cfg(not(target_arch = "wasm32"))]
pub mod lora_merge;

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
        pub fn $file_fn(
            p: impl ::std::convert::AsRef<::std::path::Path>,
        ) -> ::anyhow::Result<Self> {
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

#[cfg(target_arch = "wasm32")]
pub type ExternReadFn = Box<dyn Fn(u64, &mut [u8]) -> Result<()>>;
#[cfg(not(target_arch = "wasm32"))]
pub type ExternReadFn = Box<dyn Fn(u64, &mut [u8]) -> Result<()> + Send + Sync>;

pub struct ExternBytes {
    pub len: u64,
    pub read: ExternReadFn,
}

enum ShardBytes {
    Mapped(Mmap),
    Owned(Vec<u8>),
    Extern(ExternBytes),
}

impl std::ops::Deref for ShardBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.resident()
            .expect("extern shard bytes are fetch-only: use range_cow, never a resident slice")
    }
}

impl ShardBytes {
    fn resident(&self) -> Result<&[u8]> {
        match self {
            ShardBytes::Mapped(m) => Ok(m),
            ShardBytes::Owned(v) => Ok(v),
            ShardBytes::Extern(_) => {
                bail!("extern shard bytes are fetch-only: use range_cow, never a resident slice")
            }
        }
    }

    fn len_bytes(&self) -> u64 {
        match self {
            ShardBytes::Mapped(m) => m.len() as u64,
            ShardBytes::Owned(v) => v.len() as u64,
            ShardBytes::Extern(e) => e.len,
        }
    }

    fn range_cow(&self, start: usize, end: usize) -> Result<std::borrow::Cow<'_, [u8]>> {
        anyhow::ensure!(
            start <= end && end as u64 <= self.len_bytes(),
            "byte range [{start}, {end}) overruns a {}-byte shard",
            self.len_bytes()
        );
        match self {
            ShardBytes::Mapped(m) => Ok(std::borrow::Cow::Borrowed(&m[start..end])),
            ShardBytes::Owned(v) => Ok(std::borrow::Cow::Borrowed(&v[start..end])),
            ShardBytes::Extern(e) => {
                let mut buf = vec![0u8; end - start];
                (e.read)(start as u64, &mut buf)?;
                Ok(std::borrow::Cow::Owned(buf))
            }
        }
    }
}

struct Shard {
    path: PathBuf,
    bytes: ShardBytes,
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

    pub fn open_from_named_bytes(
        named: impl IntoIterator<Item = (String, Vec<u8>)>,
        device: &Device,
    ) -> Result<Self> {
        let mut shards: Vec<Shard> = Vec::new();
        let mut name_to_shard: HashMap<String, usize> = HashMap::new();
        for (label, bytes) in named {
            let idx = shards.len();
            let shard = shard_from_bytes(PathBuf::from(&label), ShardBytes::Owned(bytes))?;
            for name in shard.headers.keys() {
                if let Some(prev) = name_to_shard.insert(name.clone(), idx) {
                    bail!(
                        "duplicate tensor {} in byte shards {} and {}",
                        name,
                        shards[prev].path.display(),
                        label
                    );
                }
            }
            shards.push(shard);
        }
        if shards.is_empty() {
            bail!("open_from_named_bytes: no shards given");
        }
        Ok(Self {
            shards,
            name_to_shard,
            device: device.clone(),
        })
    }

    pub fn open_extern_shards(
        named: impl IntoIterator<Item = (String, ExternBytes)>,
        device: &Device,
    ) -> Result<Self> {
        let mut shards: Vec<Shard> = Vec::new();
        let mut name_to_shard: HashMap<String, usize> = HashMap::new();
        for (label, ext) in named {
            let idx = shards.len();
            let shard = shard_from_extern(PathBuf::from(&label), ext)?;
            for name in shard.headers.keys() {
                if let Some(prev) = name_to_shard.insert(name.clone(), idx) {
                    bail!(
                        "duplicate tensor {} in extern shards {} and {}",
                        name,
                        shards[prev].path.display(),
                        label
                    );
                }
            }
            shards.push(shard);
        }
        if shards.is_empty() {
            bail!("open_extern_shards: no shards given");
        }
        Ok(Self {
            shards,
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
        let bytes = shard
            .bytes
            .range_cow(header.offset_start, header.offset_end)
            .with_context(|| format!("fetch bytes of {name}"))?;
        let bytes = &*bytes;

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
    shard_from_bytes(path.to_path_buf(), ShardBytes::Mapped(bytes))
}

pub const WASM32_PARSES_THE_SAFETENSORS_HEADER_ITSELF_BECAUSE_UPSTREAM_BIT_COUNTS_OVERFLOW_32BIT_USIZE:
    &str =
    "SafeTensors::deserialize validates every tensor as element_count * dtype.bitsize() in \
     usize, so any tensor past 512 MiB (4 Gibit -- q38's 606 MiB embed_tokens and lm_head rows) \
     fails with ValidationOverflow on wasm32; the wasm twin reads the same header fields \
     (dtype, shape, data_offsets) with u64 arithmetic and the same byte-range and size checks, \
     while the native arm keeps the upstream parse byte-identical";

#[derive(serde::Deserialize)]
struct U64HeaderEntry {
    dtype: StDtype,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

fn headers_from_u64_header_json(
    path: &Path,
    header_json: &[u8],
    base: u64,
    total: u64,
) -> Result<HashMap<String, TensorHeader>> {
    anyhow::ensure!(
        usize::try_from(total).is_ok(),
        "{}: a {total}-byte shard overruns this target's usize offsets",
        path.display()
    );
    let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(header_json)
        .with_context(|| format!("parse safetensors header json in {}", path.display()))?;
    let mut headers: HashMap<String, TensorHeader> = HashMap::with_capacity(raw.len());
    for (name, v) in raw {
        if name == "__metadata__" {
            continue;
        }
        let e: U64HeaderEntry = serde_json::from_value(v)
            .with_context(|| format!("parse header entry {name} in {}", path.display()))?;
        let (s, t) = e.data_offsets;
        let elems = e
            .shape
            .iter()
            .try_fold(1u64, |a, &d| a.checked_mul(d as u64));
        let bits = elems.and_then(|x| x.checked_mul(e.dtype.bitsize() as u64));
        anyhow::ensure!(
            t >= s && base + t <= total && bits.is_some_and(|b| b == (t - s) * 8),
            "{}: tensor {name} offsets [{s}, {t}) disagree with shape {:?} dtype {:?} in a \
             {total}-byte shard",
            path.display(),
            e.shape,
            e.dtype
        );
        headers.insert(
            name,
            TensorHeader {
                dtype: e.dtype,
                shape: e.shape,
                offset_start: (base + s) as usize,
                offset_end: (base + t) as usize,
            },
        );
    }
    Ok(headers)
}

#[cfg(target_arch = "wasm32")]
fn shard_from_bytes(path: PathBuf, bytes: ShardBytes) -> Result<Shard> {
    let _ = WASM32_PARSES_THE_SAFETENSORS_HEADER_ITSELF_BECAUSE_UPSTREAM_BIT_COUNTS_OVERFLOW_32BIT_USIZE;
    let total = bytes.len() as u64;
    if bytes.len() < 8 {
        bail!(
            "{}: shorter than the safetensors length prefix",
            path.display()
        );
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("8-byte prefix"));
    let base = n
        .checked_add(8)
        .filter(|b| *b <= total)
        .with_context(|| format!("{}: header length {n} overruns the shard", path.display()))?;
    let headers = headers_from_u64_header_json(&path, &bytes[8..base as usize], base, total)?;
    Ok(Shard {
        path,
        bytes,
        headers,
    })
}

fn shard_from_extern(path: PathBuf, ext: ExternBytes) -> Result<Shard> {
    let total = ext.len;
    if total < 8 {
        bail!(
            "{}: shorter than the safetensors length prefix",
            path.display()
        );
    }
    let mut prefix = [0u8; 8];
    (ext.read)(0, &mut prefix)
        .with_context(|| format!("read the length prefix of {}", path.display()))?;
    let n = u64::from_le_bytes(prefix);
    let base = n
        .checked_add(8)
        .filter(|b| *b <= total)
        .with_context(|| format!("{}: header length {n} overruns the shard", path.display()))?;
    let mut header_json = vec![0u8; n as usize];
    (ext.read)(8, &mut header_json)
        .with_context(|| format!("read the safetensors header of {}", path.display()))?;
    let headers = headers_from_u64_header_json(&path, &header_json, base, total)?;
    Ok(Shard {
        path,
        bytes: ShardBytes::Extern(ext),
        headers,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn shard_from_bytes(path: PathBuf, bytes: ShardBytes) -> Result<Shard> {
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
        path,
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
    MlxAffineW4,
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
            QuantScheme::MlxAffineW4 => Some(64),
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
            QuantScheme::MlxAffineW4 => Ok(()),
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

        if obj.get("mode").and_then(|x| x.as_str()) == Some("affine")
            && obj.get("bits").and_then(|x| x.as_u64()) == Some(4)
        {
            let group_size = obj
                .get("group_size")
                .and_then(|x| x.as_u64())
                .map(|g| g as usize);
            return Ok(Self {
                scheme: QuantScheme::MlxAffineW4,
                ignored_modules,
                group_size,
            });
        }

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
            .or_else(|| v.get("quantization"))
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
    pub weight_bias_bytes: Option<Vec<u8>>,
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
    fn header_of(&self, name: &str) -> Result<(&Shard, &TensorHeader)> {
        let shard_idx = *self
            .name_to_shard
            .get(name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
        let shard = &self.shards[shard_idx];
        let header = shard
            .headers
            .get(name)
            .ok_or_else(|| anyhow!("tensor not found in shard: {name}"))?;
        Ok((shard, header))
    }

    pub fn raw_bytes(&self, name: &str) -> Result<&[u8]> {
        let (shard, header) = self.header_of(name)?;
        Ok(&shard
            .bytes
            .resident()
            .with_context(|| format!("{name}: raw_bytes on an extern shard (use raw_bytes_cow)"))?
            [header.offset_start..header.offset_end])
    }

    pub fn raw_len(&self, name: &str) -> Result<usize> {
        let (_, header) = self.header_of(name)?;
        Ok(header.offset_end - header.offset_start)
    }

    pub fn raw_bytes_cow(&self, name: &str) -> Result<std::borrow::Cow<'_, [u8]>> {
        let (shard, header) = self.header_of(name)?;
        shard
            .bytes
            .range_cow(header.offset_start, header.offset_end)
            .with_context(|| format!("fetch bytes of {name}"))
    }

    pub fn raw_range_cow(
        &self,
        name: &str,
        offset: usize,
        len: usize,
    ) -> Result<std::borrow::Cow<'_, [u8]>> {
        let (shard, header) = self.header_of(name)?;
        let tensor_len = header.offset_end - header.offset_start;
        anyhow::ensure!(
            offset.checked_add(len).is_some_and(|e| e <= tensor_len),
            "{name}: range [{offset}, {offset}+{len}) overruns the {tensor_len}-byte tensor"
        );
        shard
            .bytes
            .range_cow(
                header.offset_start + offset,
                header.offset_start + offset + len,
            )
            .with_context(|| format!("fetch a range of {name}"))
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
            QuantScheme::MlxAffineW4 => {
                let packed_bytes = self.raw_bytes_cow(base_name)?.into_owned();
                let shape = self
                    .shape_of(base_name)
                    .ok_or_else(|| anyhow!("missing shape for {base_name}"))?;
                let module = strip_weight_suffix(base_name);
                let weight_scale_bytes = Some(
                    self.raw_bytes_cow(&format!("{module}.scales"))?
                        .into_owned(),
                );
                let weight_bias_bytes = Some(
                    self.raw_bytes_cow(&format!("{module}.biases"))?
                        .into_owned(),
                );
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
                    weight_bias_bytes,
                    input_scale: None,
                })
            }
            QuantScheme::Fp8E4m3 => {
                let weight_bytes = self.raw_bytes_cow(base_name)?.into_owned();
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
                    weight_bias_bytes: None,
                    input_scale,
                };

                qw.fp8_weight_scale_rows()
                    .with_context(|| format!("{base_name}: checkpoint weight_scale"))?;
                Ok(qw)
            }
            QuantScheme::Nvfp4 => {
                let packed_bytes = self.raw_bytes_cow(base_name)?.into_owned();
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
                        weight_scale_bytes = Some(self.raw_bytes_cow(cand)?.into_owned());
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
                    weight_bias_bytes: None,
                    input_scale: None,
                })
            }
        }
    }
}

fn strip_weight_suffix(name: &str) -> &str {
    name.strip_suffix(".weight").unwrap_or(name)
}

pub const RECUT_SHARD_CAP_BYTES_STREAMS_ANY_CHECKPOINT_THROUGH_WASM32_ONE_SHARD_AT_A_TIME: usize =
    1 << 30;

const RECUT_PER_TENSOR_HEADER_ALLOWANCE_BYTES: usize = 192;

impl WeightLoader {
    pub fn recut_groups(&self, cap_bytes: usize) -> Result<Vec<Vec<String>>> {
        let mut names = self.names();
        names.sort();
        let mut groups: Vec<Vec<String>> = Vec::new();
        let mut cur: Vec<String> = Vec::new();
        let mut cur_bytes = 0usize;
        for name in names {
            let cost = self.raw_len(&name)? + name.len() + RECUT_PER_TENSOR_HEADER_ALLOWANCE_BYTES;
            if cost > cap_bytes {
                bail!(
                    "tensor {name} alone costs {cost} bytes against the {cap_bytes}-byte shard \
                     cap; a single tensor cannot be recut"
                );
            }
            if !cur.is_empty() && cur_bytes + cost > cap_bytes {
                groups.push(std::mem::take(&mut cur));
                cur_bytes = 0;
            }
            cur_bytes += cost;
            cur.push(name);
        }
        if !cur.is_empty() {
            groups.push(cur);
        }
        Ok(groups)
    }

    pub fn recut_shard_bytes(&self, group: &[String]) -> Result<Vec<u8>> {
        let mut views: Vec<(&str, TensorView<'_>)> = Vec::with_capacity(group.len());
        for name in group {
            let dtype = self
                .st_dtype_of(name)
                .ok_or_else(|| anyhow!("missing dtype for {name}"))?;
            let shape = self
                .shape_of(name)
                .ok_or_else(|| anyhow!("missing shape for {name}"))?;
            let view = TensorView::new(dtype, shape, self.raw_bytes(name)?)
                .with_context(|| format!("recut view for {name}"))?;
            views.push((name.as_str(), view));
        }
        safetensors::serialize(views, None)
            .with_context(|| format!("serialize a recut shard of {} tensors", group.len()))
    }

    pub fn recut_named_shards(&self, cap_bytes: usize) -> Result<Vec<(String, Vec<u8>)>> {
        let groups = self.recut_groups(cap_bytes)?;
        let total = groups.len();
        groups
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let bytes = self.recut_shard_bytes(g)?;
                anyhow::ensure!(
                    bytes.len() <= cap_bytes,
                    "recut shard {i} serialized to {} bytes over the {cap_bytes}-byte cap",
                    bytes.len()
                );
                Ok((recut_shard_name(i, total), bytes))
            })
            .collect()
    }
}

pub fn recut_shard_name(index: usize, total: usize) -> String {
    format!("recut-{:05}-of-{:05}.safetensors", index + 1, total)
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
    fn i64_roundtrips_through_get_and_bool_canonicalizes_to_u8() {
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

        let bytes: Vec<u8> = vec![1, 0, 1, 1, 0, 0];
        let path = write_safetensors_one("t2d", StDtype::BOOL, &[bytes.len()], &bytes);
        let loader = WeightLoader::open_file(&path, &Device::Cpu).unwrap();
        assert_eq!(loader.dtype_of("t2d"), Some(DType::U8), "BOOL canonicalizes to U8");
        let t = loader.get("t2d", DType::U8).expect("get BOOL as U8");
        assert_eq!(t.dtype(), DType::U8);
        assert_eq!(t.dims(), &[bytes.len()]);
        let got: Vec<u8> = t.to_vec1().unwrap();
        assert_eq!(got, bytes);
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
    fn fp8_checkpoint_scales_are_loaded_broadcast_from_a_scalar_and_canonicalised_or_refused_by_shape() {
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
            Some(vec![0.75; out_features]),
            "a scalar checkpoint scale broadcasts to every output row"
        );

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
    fn quantization_configs_accept_nvfp4_16_and_plain_fp8_and_refuse_other_group_sizes_by_name() {
        let group = |bits: u32, group_size: &str| -> String {
            format!(
                r#"{{"quantization_config": {{"quant_method": "compressed-tensors", "config_groups": {{"group_0": {{"weights": {{"num_bits": {bits}, "type": "float"{group_size}}}}}}}, "ignore": ["lm_head"]}}}}"#
            )
        };
        let q = QuantizationConfig::from_hf_json_str(&group(4, ", \"group_size\": 16")).unwrap();
        assert_eq!(q.scheme, QuantScheme::Nvfp4);
        assert_eq!(q.group_size, Some(16));
        assert_eq!(q.effective_group_size(), Some(NVFP4_GROUP_SIZE));
        let q = QuantizationConfig::from_hf_json_str(
            r#"{"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}"#,
        )
        .unwrap();
        assert_eq!(q.scheme, QuantScheme::Nvfp4);
        assert_eq!(q.group_size, None);
        assert_eq!(q.effective_group_size(), Some(NVFP4_GROUP_SIZE), "undeclared falls back to the implemented size");
        let q = QuantizationConfig::from_hf_json_str(&group(8, "")).unwrap();
        assert_eq!(q.scheme, QuantScheme::Fp8E4m3);
        assert_eq!(q.group_size, None);
        assert_eq!(q.effective_group_size(), None);
        assert!(q.is_module_ignored("lm_head"));
        for (json, words) in [
            (group(4, ", \"group_size\": 32"), vec!["32", "MXFP4", &NVFP4_GROUP_SIZE.to_string()]),
            (group(8, ", \"group_size\": 128"), vec!["128", "per output row"]),
            (
                r#"{"quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8", "config_groups": {"group_0": {"weights": {"group_size": 128}}}}}"#.to_string(),
                vec![],
            ),
        ] {
            let err = QuantizationConfig::from_hf_json_str(&json)
                .expect_err("a group size the kernels do not implement must not load")
                .to_string();
            for w in words {
                assert!(err.contains(w), "{json}: {err}");
            }
        }
    }

    fn synth_named_shard(tensors: &[(&str, usize)]) -> (String, Vec<u8>) {
        let payloads: Vec<(String, Vec<u8>)> = tensors
            .iter()
            .map(|(n, len)| {
                let bytes: Vec<u8> = (0..*len).map(|i| (i % 251) as u8).collect();
                (n.to_string(), bytes)
            })
            .collect();
        let views: Vec<(&str, TensorView<'_>)> = payloads
            .iter()
            .map(|(n, b)| {
                (
                    n.as_str(),
                    TensorView::new(StDtype::U8, vec![b.len()], b).unwrap(),
                )
            })
            .collect();
        (
            "synth.safetensors".to_string(),
            safetensors::serialize(views, None).unwrap(),
        )
    }

    #[test]
    fn extern_shards_serve_every_tensor_bit_equal_and_recut_shards_respect_the_cap_and_round_trip() {
        let (label, bytes) =
            synth_named_shard(&[("a.weight", 700), ("b.scales", 96), ("c.biases", 40)]);
        let resident =
            WeightLoader::open_from_named_bytes([(label.clone(), bytes.clone())], &Device::Cpu)
                .unwrap();
        let backing = std::sync::Arc::new(bytes);
        let fetches = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (b2, f2) = (backing.clone(), fetches.clone());
        let read: ExternReadFn = Box::new(move |off, dst: &mut [u8]| {
            f2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let start = off as usize;
            dst.copy_from_slice(&b2[start..start + dst.len()]);
            Ok(())
        });
        let ext = WeightLoader::open_extern_shards(
            [(
                label,
                ExternBytes {
                    len: backing.len() as u64,
                    read,
                },
            )],
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(resident.names(), ext.names());
        for n in &resident.names() {
            assert_eq!(
                resident.st_dtype_of(n),
                ext.st_dtype_of(n),
                "{n} dtype drifted"
            );
            assert_eq!(resident.shape_of(n), ext.shape_of(n), "{n} shape drifted");
            assert_eq!(resident.raw_len(n).unwrap(), ext.raw_len(n).unwrap());
            assert_eq!(
                &*resident.raw_bytes_cow(n).unwrap(),
                &*ext.raw_bytes_cow(n).unwrap(),
                "{n} bytes drifted through the extern fetch"
            );
            let t_res = resident.get(n, DType::U8).unwrap().flatten_all().unwrap();
            let t_ext = ext.get(n, DType::U8).unwrap().flatten_all().unwrap();
            assert_eq!(
                t_res.to_vec1::<u8>().unwrap(),
                t_ext.to_vec1::<u8>().unwrap(),
                "{n} tensor values drifted"
            );
        }
        assert_eq!(
            &*resident.raw_range_cow("a.weight", 13, 100).unwrap(),
            &*ext.raw_range_cow("a.weight", 13, 100).unwrap(),
            "ranged fetch drifted"
        );
        assert!(
            ext.raw_range_cow("a.weight", 690, 20).is_err(),
            "a range past the tensor end must refuse"
        );
        let err = ext.raw_bytes("a.weight").unwrap_err().to_string();
        assert!(
            err.contains("raw_bytes on an extern shard"),
            "extern raw_bytes must refuse loudly, got: {err}"
        );
        assert!(
            fetches.load(std::sync::atomic::Ordering::SeqCst) > 0,
            "the extern arm never exercised its fetch closure"
        );

        let shard = synth_named_shard(&[
            ("a.weight", 700),
            ("b.weight", 700),
            ("c.scales", 700),
            ("d.biases", 40),
            ("e.weight", 700),
        ]);
        let loader = WeightLoader::open_from_named_bytes([shard], &Device::Cpu).unwrap();
        let cap = 2048;
        let recut = loader.recut_named_shards(cap).unwrap();
        assert!(
            recut.len() >= 2,
            "a 2 KiB cap over 2.8 KiB of tensors must split into multiple shards"
        );
        for (name, bytes) in &recut {
            assert!(
                bytes.len() <= cap,
                "{name} serialized to {} bytes over the {cap}-byte cap",
                bytes.len()
            );
        }
        let names: Vec<String> = recut.iter().map(|(n, _)| n.clone()).collect();
        assert_eq!(names[0], recut_shard_name(0, recut.len()));
        let re = WeightLoader::open_from_named_bytes(recut, &Device::Cpu).unwrap();
        let mut all = loader.names();
        all.sort();
        let mut re_names = re.names();
        re_names.sort();
        assert_eq!(all, re_names, "the recut lost or invented tensors");
        for n in &all {
            assert_eq!(
                loader.raw_bytes(n).unwrap(),
                re.raw_bytes(n).unwrap(),
                "{n} bytes drifted"
            );
            assert_eq!(
                loader.st_dtype_of(n),
                re.st_dtype_of(n),
                "{n} dtype drifted"
            );
            assert_eq!(loader.shape_of(n), re.shape_of(n), "{n} shape drifted");
        }
        let big = synth_named_shard(&[("big.weight", 4096)]);
        let loader = WeightLoader::open_from_named_bytes([big], &Device::Cpu).unwrap();
        let err = loader.recut_named_shards(1024).unwrap_err().to_string();
        assert!(
            err.contains("cannot be recut"),
            "oversized tensor must refuse loudly, got: {err}"
        );
    }

}
