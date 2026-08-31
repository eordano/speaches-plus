use crate::lora_adapter::LoraAdapter;
use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use half::bf16;
use std::collections::BTreeMap;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

pub const MODEL_FILE: &str = "model.safetensors";

#[derive(Clone, Debug)]
struct TensorEntry {
    dtype: String,
    shape: Vec<usize>,
    begin: u64,
    end: u64,
}

fn read_safetensors_index(path: &Path) -> Result<BTreeMap<String, TensorEntry>> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8)?;
    let hlen = u64::from_le_bytes(len8);
    let mut hdr = vec![0u8; hlen as usize];
    f.read_exact(&mut hdr)?;
    let v: serde_json::Value = serde_json::from_slice(&hdr).context("safetensors header json")?;
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow!("safetensors header is not an object"))?;
    let data_base = 8 + hlen;
    let mut out = BTreeMap::new();
    for (name, ent) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = ent["dtype"]
            .as_str()
            .ok_or_else(|| anyhow!("{name}: missing dtype"))?
            .to_string();
        let shape: Vec<usize> = ent["shape"]
            .as_array()
            .ok_or_else(|| anyhow!("{name}: missing shape"))?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as usize)
            .collect();
        let offs = ent["data_offsets"]
            .as_array()
            .ok_or_else(|| anyhow!("{name}: missing data_offsets"))?;
        let begin = data_base + offs[0].as_u64().ok_or_else(|| anyhow!("{name}: bad begin"))?;
        let end = data_base + offs[1].as_u64().ok_or_else(|| anyhow!("{name}: bad end"))?;
        out.insert(
            name.clone(),
            TensorEntry {
                dtype,
                shape,
                begin,
                end,
            },
        );
    }
    Ok(out)
}

struct BaseFile {
    map: memmap2::Mmap,
    index: BTreeMap<String, TensorEntry>,
}

impl BaseFile {
    fn open(path: &Path) -> Result<Self> {
        let index = read_safetensors_index(path)?;
        let f = std::fs::File::open(path)?;
        let map = unsafe { memmap2::Mmap::map(&f)? };
        Ok(Self { map, index })
    }

    fn entry(&self, name: &str) -> Result<&TensorEntry> {
        self.index
            .get(name)
            .ok_or_else(|| anyhow!("base checkpoint has no tensor {name}"))
    }

    fn bytes(&self, name: &str) -> Result<&[u8]> {
        let e = self.entry(name)?;
        Ok(&self.map[e.begin as usize..e.end as usize])
    }

    fn scalar_f32(&self, name: &str) -> Result<f32> {
        let e = self.entry(name)?;
        anyhow::ensure!(e.dtype == "F32", "{name}: expected F32, got {}", e.dtype);
        let b = &self.map[e.begin as usize..e.end as usize];
        anyhow::ensure!(b.len() >= 4, "{name}: short read");
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        out.extend_from_slice(&bf16::from_f32(*v).to_le_bytes());
    }
    out
}

fn lora_delta_f32(a: &Tensor, b: &Tensor, out_f: usize, in_f: usize) -> Result<Vec<f32>> {
    let a32 = a.to_dtype(DType::F32)?;
    let b32 = b.to_dtype(DType::F32)?;
    let ad = a32.dims();
    let bd = b32.dims();
    anyhow::ensure!(
        ad.len() == 2 && bd.len() == 2 && bd[1] == ad[0] && ad[1] == in_f && bd[0] == out_f,
        "lora delta shape mismatch: A {ad:?} B {bd:?} vs [{out_f}, {in_f}]"
    );
    let d = b32.matmul(&a32)?;
    Ok(d.flatten_all()?.to_vec1::<f32>()?)
}

fn par_rows<F: Fn(usize) + Sync>(rows: usize, f: F) {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(rows.max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let r = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if r >= rows {
                    break;
                }
                f(r);
            });
        }
    });
}

struct RewriteStats {
    worst_abs_err_over_amax: f32,
}

fn merge_fp8_rowscale_module(
    base: &BaseFile,
    module: &str,
    delta: &[f32],
    out_f: usize,
    in_f: usize,
    out_file: &std::fs::File,
) -> Result<RewriteStats> {
    let wname = format!("{module}.weight");
    let sname = format!("{module}.weight_scale");
    let went = base.entry(&wname)?.clone();
    let sent = base.entry(&sname)?.clone();
    anyhow::ensure!(
        went.dtype == "F8_E4M3" && went.shape == [out_f, in_f],
        "{wname}: expected F8_E4M3 [{out_f}, {in_f}], got {} {:?}",
        went.dtype,
        went.shape
    );
    anyhow::ensure!(
        sent.dtype == "BF16",
        "{sname}: expected BF16 row scales, got {}",
        sent.dtype
    );
    let wbytes = base.bytes(&wname)?;
    let scales = bf16_bytes_to_f32(base.bytes(&sname)?);
    anyhow::ensure!(
        scales.len() == out_f,
        "{sname}: {} scales for {out_f} rows",
        scales.len()
    );

    let mut new_bytes = vec![0u8; out_f * in_f];
    let mut new_scales = vec![0f32; out_f];
    let mut amax_all = 0f32;
    {
        let new_rows: Vec<std::sync::Mutex<(&mut [u8], &mut [f32])>> = {
            let mut br = new_bytes.chunks_mut(in_f);
            let mut sr = new_scales.chunks_mut(1);
            (0..out_f)
                .map(|_| std::sync::Mutex::new((br.next().unwrap(), sr.next().unwrap())))
                .collect()
        };
        let worst = std::sync::Mutex::new(0f32);
        let amax = std::sync::Mutex::new(0f32);
        par_rows(out_f, |r| {
            use float8::F8E4M3;
            let mut merged = vec![0f32; in_f];
            let mut row_amax = 0f32;
            for c in 0..in_f {
                let v = f32::from(F8E4M3::from_bits(wbytes[r * in_f + c])) * scales[r]
                    + delta[r * in_f + c];
                row_amax = row_amax.max(v.abs());
                merged[c] = v;
            }
            let scale_exact = if row_amax == 0.0 || !row_amax.is_finite() {
                1.0
            } else {
                row_amax / nv_quant::fp8::E4M3_MAX
            };
            let scale = bf16::from_f32(scale_exact).to_f32();
            let inv = if scale == 0.0 || !scale.is_finite() {
                1.0
            } else {
                1.0 / scale
            };
            let mut row_worst = 0f32;
            let mut guard = new_rows[r].lock().unwrap();
            let (rb, rs) = &mut *guard;
            for c in 0..in_f {
                let q = F8E4M3::from(merged[c] * inv);
                rb[c] = q.to_bits();
                let back = f32::from(q) * scale;
                row_worst = row_worst.max((back - merged[c]).abs());
            }
            rs[0] = scale;
            let mut w = worst.lock().unwrap();
            *w = w.max(row_worst);
            let mut a = amax.lock().unwrap();
            *a = a.max(row_amax);
        });
        amax_all = amax_all.max(*amax.lock().unwrap());
        let w = *worst.lock().unwrap();
        drop(new_rows);
        out_file.write_all_at(&new_bytes, went.begin)?;
        out_file.write_all_at(&f32_to_bf16_bytes(&new_scales), sent.begin)?;
        Ok(RewriteStats {
            worst_abs_err_over_amax: if amax_all > 0.0 { w / amax_all } else { 0.0 },
        })
    }
}

fn merge_nvfp4_packed_module(
    base: &BaseFile,
    module: &str,
    delta: &[f32],
    out_f: usize,
    in_f: usize,
    out_file: &std::fs::File,
) -> Result<RewriteStats> {
    use nv_quant::nvfp4::{quantize_block_with_global, BLOCK_SIZE};
    let pname = format!("{module}.weight_packed");
    let sname = format!("{module}.weight_scale");
    let gname = format!("{module}.weight_global_scale");
    let pent = base.entry(&pname)?.clone();
    let sent = base.entry(&sname)?.clone();
    anyhow::ensure!(
        pent.shape == [out_f, in_f / 2],
        "{pname}: expected [{out_f}, {}], got {:?}",
        in_f / 2,
        pent.shape
    );
    let blocks_per_row = in_f / BLOCK_SIZE;
    anyhow::ensure!(
        sent.shape == [out_f, blocks_per_row],
        "{sname}: expected [{out_f}, {blocks_per_row}], got {:?}",
        sent.shape
    );
    let stored_global = base.scalar_f32(&gname)?;
    let weight_mult = if stored_global == 0.0 || !stored_global.is_finite() {
        1.0
    } else {
        1.0 / stored_global
    };
    let packed = base.bytes(&pname)?;
    let scale_bytes = base.bytes(&sname)?;
    anyhow::ensure!(
        packed.len() == out_f * in_f / 2 && scale_bytes.len() == out_f * blocks_per_row,
        "{pname}: packed/scale byte lengths do not match the declared geometry"
    );

    let mut new_packed = vec![0u8; out_f * in_f / 2];
    let mut new_scales = vec![0u8; out_f * blocks_per_row];
    let worst = std::sync::Mutex::new(0f32);
    let amax = std::sync::Mutex::new(0f32);
    {
        let rows_p: Vec<std::sync::Mutex<(&mut [u8], &mut [u8])>> = {
            let mut pr = new_packed.chunks_mut(in_f / 2);
            let mut sr = new_scales.chunks_mut(blocks_per_row);
            (0..out_f)
                .map(|_| std::sync::Mutex::new((pr.next().unwrap(), sr.next().unwrap())))
                .collect()
        };
        par_rows(out_f, |r| {
            let row_packed = &packed[r * in_f / 2..(r + 1) * in_f / 2];
            let row_scales = &scale_bytes[r * blocks_per_row..(r + 1) * blocks_per_row];
            let mut merged = vec![0f32; in_f];
            let mut row_amax = 0f32;
            for b in 0..blocks_per_row {
                let deq = nv_quant::nvfp4::dequantize_block(
                    &row_packed[b * BLOCK_SIZE / 2..(b + 1) * BLOCK_SIZE / 2],
                    row_scales[b],
                );
                for (i, base_v) in deq.iter().enumerate() {
                    let c = b * BLOCK_SIZE + i;
                    let v = base_v * weight_mult + delta[r * in_f + c];
                    row_amax = row_amax.max(v.abs());
                    merged[c] = v;
                }
            }
            let mut guard = rows_p[r].lock().unwrap();
            let (pb, sb) = &mut *guard;
            let mut row_worst = 0f32;
            for b in 0..blocks_per_row {
                let block = &merged[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
                let (pbytes, sbyte) = quantize_block_with_global(block, stored_global);
                pb[b * BLOCK_SIZE / 2..(b + 1) * BLOCK_SIZE / 2].copy_from_slice(&pbytes);
                sb[b] = sbyte;
                let back = nv_quant::nvfp4::dequantize_block(&pbytes, sbyte);
                for (i, v) in back.iter().enumerate() {
                    row_worst = row_worst.max((v * weight_mult - block[i]).abs());
                }
            }
            let mut wl = worst.lock().unwrap();
            *wl = wl.max(row_worst);
            let mut a = amax.lock().unwrap();
            *a = a.max(row_amax);
        });
    }
    out_file.write_all_at(&new_packed, pent.begin)?;
    out_file.write_all_at(&new_scales, sent.begin)?;
    let a = *amax.lock().unwrap();
    let wv = *worst.lock().unwrap();
    Ok(RewriteStats {
        worst_abs_err_over_amax: if a > 0.0 { wv / a } else { 0.0 },
    })
}

fn merge_bf16_module(
    base: &BaseFile,
    module: &str,
    delta: &[f32],
    out_f: usize,
    in_f: usize,
    out_file: &std::fs::File,
) -> Result<RewriteStats> {
    let wname = format!("{module}.weight");
    let went = base.entry(&wname)?.clone();
    anyhow::ensure!(
        went.dtype == "BF16" && went.shape == [out_f, in_f],
        "{wname}: expected BF16 [{out_f}, {in_f}], got {} {:?}",
        went.dtype,
        went.shape
    );
    let wbytes = base.bytes(&wname)?;
    let mut new_bytes = vec![0u8; out_f * in_f * 2];
    let worst = std::sync::Mutex::new(0f32);
    let amax = std::sync::Mutex::new(0f32);
    {
        let rows: Vec<std::sync::Mutex<&mut [u8]>> = new_bytes
            .chunks_mut(in_f * 2)
            .map(std::sync::Mutex::new)
            .collect();
        par_rows(out_f, |r| {
            let mut guard = rows[r].lock().unwrap();
            let rb = &mut *guard;
            let mut row_worst = 0f32;
            let mut row_amax = 0f32;
            for c in 0..in_f {
                let off = (r * in_f + c) * 2;
                let v = bf16::from_le_bytes([wbytes[off], wbytes[off + 1]]).to_f32()
                    + delta[r * in_f + c];
                row_amax = row_amax.max(v.abs());
                let q = bf16::from_f32(v);
                rb[c * 2..c * 2 + 2].copy_from_slice(&q.to_le_bytes());
                row_worst = row_worst.max((q.to_f32() - v).abs());
            }
            let mut wl = worst.lock().unwrap();
            *wl = wl.max(row_worst);
            let mut a = amax.lock().unwrap();
            *a = a.max(row_amax);
        });
    }
    out_file.write_all_at(&new_bytes, went.begin)?;
    let a = *amax.lock().unwrap();
    let wv = *worst.lock().unwrap();
    Ok(RewriteStats {
        worst_abs_err_over_amax: if a > 0.0 { wv / a } else { 0.0 },
    })
}

#[derive(Debug)]
pub struct MergedModuleReport {
    pub module: String,
    pub kind: &'static str,
    pub worst_abs_err_over_amax: f32,
}

#[derive(Debug)]
pub struct MergeReport {
    pub out_model_file: PathBuf,
    pub modules: Vec<MergedModuleReport>,
}

impl MergeReport {
    pub fn count(&self, kind: &str) -> usize {
        self.modules.iter().filter(|m| m.kind == kind).count()
    }

    pub fn worst(&self) -> f32 {
        self.modules
            .iter()
            .map(|m| m.worst_abs_err_over_amax)
            .fold(0f32, f32::max)
    }
}

pub const UNTOUCHED_FILES_HARDLINK_SO_ONLY_THE_MODEL_SHARD_COSTS_DISK: &str = MODEL_FILE;

pub fn stage_snapshot_copy(base_dir: &Path, out_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("create out dir {}", out_dir.display()))?;
    for entry in std::fs::read_dir(base_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let src = base_dir.join(&name);
        let dst = out_dir.join(&name);
        if !entry.file_type()?.is_file() && !entry.file_type()?.is_symlink() {
            continue;
        }
        if dst.exists() {
            std::fs::remove_file(&dst)?;
        }
        let src_real = std::fs::canonicalize(&src)?;
        if name == std::ffi::OsStr::new(MODEL_FILE) {
            std::fs::copy(&src_real, &dst)
                .with_context(|| format!("copy {} -> {}", src_real.display(), dst.display()))?;
        } else if std::fs::hard_link(&src_real, &dst).is_err() {
            std::fs::copy(&src_real, &dst)?;
        }
    }
    Ok(out_dir.join(MODEL_FILE))
}

pub fn merge_lora_adapter_into_quantized_snapshot(
    base_dir: &Path,
    adapter_dir: &Path,
    out_dir: &Path,
) -> Result<MergeReport> {
    let adapter = LoraAdapter::load(adapter_dir, &Device::Cpu)
        .with_context(|| format!("load adapter {}", adapter_dir.display()))?;
    if adapter.loras.is_empty() {
        bail!("adapter {} carries no LoRA modules", adapter_dir.display());
    }
    let out_model = stage_snapshot_copy(base_dir, out_dir)?;
    let base = BaseFile::open(&base_dir.join(MODEL_FILE))?;
    let out_file = std::fs::OpenOptions::new()
        .write(true)
        .open(&out_model)
        .with_context(|| format!("open {} for in-place rewrite", out_model.display()))?;

    let total = adapter.loras.len();
    let mut modules = Vec::new();
    for (mi, (module, lw)) in adapter.loras.iter().enumerate() {
        if lw.is_embedding {
            bail!("{module}: embedding LoRA merge is not supported for the quantized snapshot");
        }
        anyhow::ensure!(
            lw.scaling == 1.0,
            "{module}: LoraAdapter::optimize must fold scaling into lora_b before merging"
        );
        let (kind, out_f, in_f) = classify_module(&base, module)?;
        let delta = lora_delta_f32(&lw.lora_a, &lw.lora_b, out_f, in_f)?;
        let stats = match kind {
            "nvfp4_packed" => {
                merge_nvfp4_packed_module(&base, module, &delta, out_f, in_f, &out_file)?
            }
            "fp8_rowscale" => {
                merge_fp8_rowscale_module(&base, module, &delta, out_f, in_f, &out_file)?
            }
            "bf16" => merge_bf16_module(&base, module, &delta, out_f, in_f, &out_file)?,
            other => bail!("{module}: unhandled storage kind {other}"),
        };
        eprintln!(
            "[lora-merge {}/{}] {} ({kind}) abs-err/amax {:.4}",
            mi + 1,
            total,
            module,
            stats.worst_abs_err_over_amax
        );
        modules.push(MergedModuleReport {
            module: module.clone(),
            kind,
            worst_abs_err_over_amax: stats.worst_abs_err_over_amax,
        });
    }
    out_file.sync_all()?;
    Ok(MergeReport {
        out_model_file: out_model,
        modules,
    })
}

#[derive(Debug, Clone)]
pub struct MergeableModule {
    pub module: String,
    pub kind: &'static str,
    pub out_features: usize,
    pub in_features: usize,
}

pub fn list_mergeable_modules(base_dir: &Path) -> Result<Vec<MergeableModule>> {
    let base = BaseFile::open(&base_dir.join(MODEL_FILE))?;
    let mut out = Vec::new();
    for name in base.index.keys() {
        let module = if let Some(m) = name.strip_suffix(".weight_packed") {
            m
        } else if let Some(m) = name.strip_suffix(".weight") {
            m
        } else {
            continue;
        };
        let Ok((kind, out_f, in_f)) = classify_module(&base, module) else {
            continue;
        };
        if out
            .last()
            .is_some_and(|p: &MergeableModule| p.module == module)
        {
            continue;
        }
        out.push(MergeableModule {
            module: module.to_string(),
            kind,
            out_features: out_f,
            in_features: in_f,
        });
    }
    Ok(out)
}

fn classify_module(base: &BaseFile, module: &str) -> Result<(&'static str, usize, usize)> {
    let packed = format!("{module}.weight_packed");
    if let Some(e) = base.index.get(&packed) {
        anyhow::ensure!(
            e.shape.len() == 2,
            "{packed}: expected rank-2 packed weight, got {:?}",
            e.shape
        );
        return Ok(("nvfp4_packed", e.shape[0], e.shape[1] * 2));
    }
    let wname = format!("{module}.weight");
    let e = base
        .index
        .get(&wname)
        .ok_or_else(|| anyhow!("{module}: neither {packed} nor {wname} in base checkpoint"))?;
    anyhow::ensure!(
        e.shape.len() == 2,
        "{wname}: expected rank-2 weight, got {:?}",
        e.shape
    );
    let (out_f, in_f) = (e.shape[0], e.shape[1]);
    if e.dtype == "F8_E4M3" && base.index.contains_key(&format!("{module}.weight_scale")) {
        return Ok(("fp8_rowscale", out_f, in_f));
    }
    if e.dtype == "BF16" {
        return Ok(("bf16", out_f, in_f));
    }
    bail!(
        "{wname}: unsupported storage dtype {} for LoRA merge",
        e.dtype
    )
}
