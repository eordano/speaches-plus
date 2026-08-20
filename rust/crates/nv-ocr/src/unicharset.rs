use std::collections::{HashMap, HashSet};

use crate::traineddata::Cursor;
use crate::Error;

fn err(msg: impl Into<String>) -> Error {
    Error::Traineddata(msg.into())
}

pub struct Unicharset {
    glyphs: Vec<String>,
}

impl Unicharset {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let text = String::from_utf8_lossy(data);
        let mut lines = text.lines();
        let count: usize = lines
            .next()
            .ok_or_else(|| err("empty unicharset"))?
            .trim()
            .parse()
            .map_err(|e| err(format!("bad unicharset count: {}", e)))?;
        if count == 0 || count > 1_000_000 {
            return Err(err(format!("implausible unicharset size {}", count)));
        }
        if count > data.len() {
            return Err(err(format!(
                "unicharset declares {} entries in {} bytes",
                count,
                data.len()
            )));
        }
        let mut glyphs = Vec::with_capacity(count);
        for line in lines {
            if glyphs.len() == count {
                break;
            }
            let Some(field) = line.split_whitespace().next() else {
                continue;
            };
            glyphs.push(if field == "NULL" {
                " ".to_string()
            } else {
                field.to_string()
            });
        }
        if glyphs.len() != count {
            return Err(err(format!(
                "unicharset declared {} entries, found {}",
                count,
                glyphs.len()
            )));
        }
        Ok(Self { glyphs })
    }

    pub fn len(&self) -> usize {
        self.glyphs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }

    pub fn glyph(&self, id: usize) -> Option<&str> {
        self.glyphs.get(id).map(|s| s.as_str())
    }
}

pub struct Recoder {
    encoder: Vec<Vec<i32>>,
    decoder: HashMap<Vec<i32>, usize>,
    prefixes: HashSet<Vec<i32>>,
    code_range: usize,
}

impl Recoder {
    pub fn deserialize(cur: &mut Cursor) -> Result<Self, Error> {
        let n = cur.u32()? as usize;
        if n == 0 || n > 1_000_000 {
            return Err(err(format!("implausible recoder entry count {}", n)));
        }
        if n.saturating_mul(5) > cur.remaining() {
            return Err(err(format!(
                "recoder declares {} entries in {} bytes",
                n,
                cur.remaining()
            )));
        }
        let mut encoder = Vec::with_capacity(n);
        let mut self_normalized = Vec::with_capacity(n);
        for _ in 0..n {
            let sn = cur.i8()? != 0;
            let len = cur.u32()? as usize;
            if len > 32 {
                return Err(err(format!("implausible recoder code length {}", len)));
            }
            let mut codes = Vec::with_capacity(len);
            for _ in 0..len {
                codes.push(cur.i32()?);
            }
            self_normalized.push(sn);
            encoder.push(codes);
        }
        if cur.remaining() != 0 {
            return Err(err(format!(
                "{} trailing bytes in recoder",
                cur.remaining()
            )));
        }
        let mut decoder = HashMap::new();
        let mut code_range = 0usize;
        for (id, codes) in encoder.iter().enumerate() {
            for &c in codes {
                if c < 0 {
                    return Err(err(format!("negative recoder code for unichar {}", id)));
                }
                code_range = code_range.max(c as usize + 1);
            }
            if self_normalized[id] || !decoder.contains_key(codes) {
                decoder.insert(codes.clone(), id);
            }
        }
        let mut prefixes = HashSet::new();
        for codes in &encoder {
            for l in 1..codes.len() {
                prefixes.insert(codes[..l].to_vec());
            }
        }
        Ok(Self {
            encoder,
            decoder,
            prefixes,
            code_range,
        })
    }

    pub fn len(&self) -> usize {
        self.encoder.len()
    }

    pub fn is_empty(&self) -> bool {
        self.encoder.is_empty()
    }

    pub fn code_range(&self) -> usize {
        self.code_range
    }

    pub fn encode(&self, unichar_id: usize) -> Option<&[i32]> {
        self.encoder.get(unichar_id).map(|v| v.as_slice())
    }

    pub fn decode(&self, codes: &[i32]) -> Option<usize> {
        self.decoder.get(codes).copied()
    }

    pub fn is_prefix(&self, codes: &[i32]) -> bool {
        self.prefixes.contains(codes)
    }
}
