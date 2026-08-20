use std::path::Path;
use std::sync::OnceLock;

use anyhow::{anyhow, Result};

pub const DIM: usize = 1 << 18;

const MAGIC: &[u8; 4] = b"STH1";

const TAIL_CHARS: usize = 48;

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x1_0000_0000_01B3);
    }
    h
}

fn bucket(kind: &str, payload: &str) -> u32 {
    let mut buf = Vec::with_capacity(kind.len() + 1 + payload.len());
    buf.extend_from_slice(kind.as_bytes());
    buf.push(0x1f);
    buf.extend_from_slice(payload.as_bytes());
    (fnv1a64(&buf) % DIM as u64) as u32
}

fn token_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"[\w'\-]+|[.!?,;:…]").expect("token regex"))
}

pub fn hashed_features(text: &str) -> Vec<u32> {
    let t = text.trim().to_lowercase();
    let chars: Vec<char> = t.chars().collect();
    let mut idx = Vec::with_capacity(chars.len().min(TAIL_CHARS) * 3 + 16);

    let tail: Vec<char> = chars[chars.len().saturating_sub(TAIL_CHARS)..].to_vec();
    for n in 2..=4usize {
        if tail.len() >= n {
            for win in tail.windows(n) {
                let s: String = win.iter().collect();
                idx.push(bucket(&format!("c{n}"), &s));
            }
        }
    }

    let words: Vec<&str> = token_re().find_iter(&t).map(|m| m.as_str()).collect();
    for w in words.iter().skip(words.len().saturating_sub(6)) {
        idx.push(bucket("w", w));
    }
    let lo = words.len().saturating_sub(3);
    for i in lo..words.len().saturating_sub(1) {
        idx.push(bucket("b", &format!("{}\u{1e}{}", words[i], words[i + 1])));
    }

    if let Some(last) = chars.last() {
        idx.push(bucket("lc", &last.to_string()));
        let ellipsis = t.ends_with("...") || t.ends_with('…');
        idx.push(bucket("el", if ellipsis { "1" } else { "0" }));
        let _ = last;
    }
    idx.push(bucket("nw", &(words.len() / 4).min(12).to_string()));
    idx
}

pub struct TextHead {
    bias: f32,
    weights: Vec<f32>,
}

impl TextHead {
    pub fn from_bytes(raw: &[u8]) -> Result<Self> {
        if raw.len() < 12 || &raw[..4] != MAGIC {
            return Err(anyhow!("not an STH1 text head (magic mismatch)"));
        }
        let dim = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
        let bias = f32::from_le_bytes(raw[8..12].try_into().unwrap());
        if dim != DIM {
            return Err(anyhow!("text head dim {dim} != expected {DIM}"));
        }
        if raw.len() != 12 + dim * 4 {
            return Err(anyhow!(
                "text head size {} != header-implied {}",
                raw.len(),
                12 + dim * 4
            ));
        }
        let weights = raw[12..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Self { bias, weights })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read(path.as_ref())?;
        Self::from_bytes(&raw)
    }

    pub fn prob(&self, text: &str) -> f32 {
        let z: f32 = self.bias
            + hashed_features(text)
                .iter()
                .map(|&i| self.weights[i as usize])
                .sum::<f32>();
        1.0 / (1.0 + (-z as f64).exp() as f32)
    }
}

pub fn shadow_head() -> Option<&'static TextHead> {
    static HEAD: OnceLock<Option<TextHead>> = OnceLock::new();
    HEAD.get_or_init(|| {
        let path = std::env::var(crate::defaults::env::EOU_TEXT_HEAD_PATH).ok()?;
        match TextHead::load(&path) {
            Ok(h) => {
                tracing::info!(path, "eou text head loaded (shadow mode)");
                Some(h)
            }
            Err(e) => {
                tracing::warn!(path, error = %e, "eou text head failed to load; shadow disabled");
                None
            }
        }
    })
    .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(text: &str) -> (usize, [u32; 3], [u32; 3], u64) {
        let mut b = hashed_features(text);
        b.sort_unstable();
        let first: [u32; 3] = [b[0], *b.get(1).unwrap_or(&0), *b.get(2).unwrap_or(&0)];
        let n = b.len();
        let last: [u32; 3] = [
            b[n.saturating_sub(3).min(n - 1)],
            b[n.saturating_sub(2).min(n - 1)],
            b[n - 1],
        ];
        (n, first, last, b.iter().map(|&x| x as u64).sum())
    }

    #[test]
    fn buckets_match_the_python_trainer_goldens() {
        let cases: [(&str, usize, [u32; 3], [u32; 3], u64); 8] = [
            ("So I was thinking maybe we could", 101, [2426, 2950, 6097], [255606, 257952, 260268], 12_844_529),
            ("How much did I spend at Amazon last month?", 131, [1169, 2069, 4248], [255726, 259970, 261878], 15_245_964),
            ("about 440 for the...", 65, [2804, 8564, 9033], [239173, 256981, 257952], 8_181_381),
            ("hmm.", 12, [11924, 40476, 41110], [214865, 221027, 238391], 1_635_631),
            ("", 1, [122533, 0, 0], [122533, 122533, 122533], 122_533),
            ("the quick brown fox jumps over the lazy dog.", 137, [1203, 1686, 7728], [260598, 260598, 260703], 17_515_563),
            ("Qué hora es en Madrid ahora mismo", 104, [4554, 7029, 7636], [245243, 248283, 254301], 11_776_874),
            ("I need to transfer, um,", 74, [1497, 4754, 7118], [256653, 260703, 260997], 8_750_941),
        ];
        for (text, n, first, last, sum) in cases {
            let (gn, gf, gl, gs) = digest(text);
            assert_eq!(
                (gn, gf, gl, gs),
                (n, first, last, sum),
                "feature-hash drift vs the python trainer for {text:?} -- retrained heads \
                 and this port must produce identical buckets or shadow logs are garbage"
            );
        }
    }

    #[test]
    fn sth1_roundtrip_and_score() {
        let mut raw = Vec::new();
        raw.extend_from_slice(MAGIC);
        raw.extend_from_slice(&(DIM as u32).to_le_bytes());
        raw.extend_from_slice(&0.5f32.to_le_bytes());
        raw.extend(std::iter::repeat_n(0u8, DIM * 4));
        let head = TextHead::from_bytes(&raw).expect("parse");
        let p = head.prob("hello there.");
        assert!((p - 1.0 / (1.0 + (-0.5f32 as f64).exp() as f32)).abs() < 1e-6);

        assert!(TextHead::from_bytes(&raw[..100]).is_err());
        let mut bad = raw.clone();
        bad[0] = b'X';
        assert!(TextHead::from_bytes(&bad).is_err());
    }
}
