use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use zip::ZipArchive;

#[derive(Debug, Clone)]
pub struct Voice {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Voice {
    pub fn row(&self, index: usize) -> Result<Vec<f32>> {
        if self.shape.is_empty() {
            return Err(anyhow!("voice has no shape"));
        }
        let row_size: usize = self.shape[1..].iter().product();
        let off = index
            .checked_mul(row_size)
            .ok_or_else(|| anyhow!("index {index} * row_size {row_size} overflows"))?;
        if index >= self.shape[0] {
            return Err(anyhow!("index {index} >= leading dim {}", self.shape[0]));
        }
        Ok(self.data[off..off + row_size].to_vec())
    }
}

pub fn load_voices(path: &Path) -> Result<HashMap<String, Voice>> {
    let file = std::fs::File::open(path).with_context(|| format!("open {path:?}"))?;
    let mut zip = ZipArchive::new(file).context("voices.bin not a valid zip")?;
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let name = entry.name().to_string();
        let Some(stem) = name.strip_suffix(".npy") else {
            continue;
        };
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut bytes).context("read npy entry")?;
        let voice = parse_npy(&bytes).with_context(|| format!("parse {name}"))?;
        out.insert(stem.to_string(), voice);
    }
    Ok(out)
}

const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

fn parse_npy(b: &[u8]) -> Result<Voice> {
    if b.len() < 10 || &b[..6] != NPY_MAGIC {
        return Err(anyhow!("not an .npy file"));
    }
    let major = b[6];
    if !(1..=3).contains(&major) {
        return Err(anyhow!("unsupported npy version {major}"));
    }
    let (header_len, data_off) = if major == 1 {
        let len = u16::from_le_bytes([b[8], b[9]]) as usize;
        (len, 10 + len)
    } else {
        let len = u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize;
        (len, 12 + len)
    };
    if data_off > b.len() {
        return Err(anyhow!("npy header truncated"));
    }
    let header_off = data_off - header_len;
    let header = std::str::from_utf8(&b[header_off..data_off])
        .map_err(|e| anyhow!("non-utf8 npy header: {e}"))?;
    let (descr, fortran, shape) = parse_npy_header(header)?;
    if fortran {
        return Err(anyhow!("fortran-order arrays not supported"));
    }
    if descr != "<f4" {
        return Err(anyhow!("unsupported dtype {descr:?} (need <f4)"));
    }
    let total: usize = shape.iter().product();
    if data_off + total * 4 > b.len() {
        return Err(anyhow!(
            "data shorter than shape implies ({} < {})",
            b.len() - data_off,
            total * 4
        ));
    }
    let mut data = Vec::with_capacity(total);
    for chunk in b[data_off..data_off + total * 4].chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Voice { shape, data })
}

fn parse_npy_header(s: &str) -> Result<(String, bool, Vec<usize>)> {
    let descr = string_field(s, "'descr':").ok_or_else(|| anyhow!("missing descr"))?;
    let fortran = match token_after(s, "'fortran_order':").as_deref() {
        Some("True") => true,
        Some("False") => false,
        other => return Err(anyhow!("malformed fortran_order: {other:?}")),
    };
    let shape_str = tuple_after(s, "'shape':").ok_or_else(|| anyhow!("missing shape"))?;
    let mut shape = Vec::new();
    for p in shape_str.split(',') {
        let p = p.trim();
        if p.is_empty() {
            continue;
        }
        let n: usize = p
            .parse()
            .map_err(|e| anyhow!("malformed shape entry {p:?}: {e}"))?;
        shape.push(n);
    }
    if shape.is_empty() {
        return Err(anyhow!("missing shape entries: {s:?}"));
    }
    Ok((descr, fortran, shape))
}

fn string_field(s: &str, key: &str) -> Option<String> {
    let i = s.find(key)?;
    let rest = &s[i + key.len()..];
    let q = rest.find('\'')?;
    let after_quote = &rest[q + 1..];
    let end = after_quote.find('\'')?;
    Some(after_quote[..end].to_string())
}

fn token_after(s: &str, key: &str) -> Option<String> {
    let i = s.find(key)?;
    let rest = s[i + key.len()..].trim_start();
    let end = rest.find([',', ' ', '}']).unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

fn tuple_after(s: &str, key: &str) -> Option<String> {
    let i = s.find(key)?;
    let rest = &s[i + key.len()..];
    let open = rest.find('(')?;
    let close = rest.find(')')?;
    if close <= open {
        return None;
    }
    Some(rest[open + 1..close].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_kokoro_voices() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/voices.bin");
        if !path.exists() {
            eprintln!("skip: voices.bin missing");
            return;
        }
        let voices = load_voices(&path).expect("load voices");
        let v = voices.get("af_heart").expect("af_heart present");
        assert_eq!(v.shape, vec![510, 1, 256]);
        assert_eq!(v.data.len(), 510 * 256);
    }
}
