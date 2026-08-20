use anyhow::{bail, Context, Result};

pub const NV_MEASURE_V1_PREFIX_ONE_GREPPABLE_TOKEN_SO_TOOLING_PARSES_EVERY_INSTRUMENT_THE_SAME_WAY:
    &str = "NV-MEASURE v1";

pub const REQUIRED_BASIS_KEYS_BECAUSE_COMPARISONS_FAILED_ON_SLAB_VS_SLOTSHARED_END_TO_END_VS_PER_GEMM_AND_MAX_SEQ_VS_DEPTH:
    [&str; 10] = [
    "instrument",
    "model",
    "backend",
    "device",
    "batch",
    "tokens",
    "steps",
    "warmup",
    "value",
    "unit",
];

#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    pub instrument: String,
    pub model_at_rev: String,
    pub backend: String,
    pub device: String,
    pub batch: usize,
    pub tokens: usize,
    pub steps: usize,
    pub warmup: usize,
    pub value: f64,
    pub unit: String,
    pub extras: Vec<(String, String)>,
}

fn quote_if_needed(v: &str) -> String {
    if !v.is_empty() && !v.contains([' ', '"', '\\', '\n']) {
        return v.to_string();
    }
    let mut out = String::with_capacity(v.len() + 2);
    out.push('"');
    for c in v.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn assert_bare_key(k: &str) {
    assert!(
        !k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
        "measurement key {k:?} must be a bare [A-Za-z0-9_-]+ ident so the line stays splittable"
    );
}

impl Measurement {
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let key = key.into();
        assert_bare_key(&key);
        assert!(
            !REQUIRED_BASIS_KEYS_BECAUSE_COMPARISONS_FAILED_ON_SLAB_VS_SLOTSHARED_END_TO_END_VS_PER_GEMM_AND_MAX_SEQ_VS_DEPTH
                .contains(&key.as_str()),
            "extra key {key:?} shadows a required basis field; set the struct field instead"
        );
        self.extras.push((key, value.into()));
        self
    }

    pub fn line(&self) -> String {
        for (name, v) in [
            ("instrument", &self.instrument),
            ("model", &self.model_at_rev),
            ("backend", &self.backend),
            ("unit", &self.unit),
        ] {
            assert!(!v.is_empty(), "required basis field {name} is empty; an empty basis is the mismatch bug this format exists to kill");
        }
        let mut s = format!(
            "{} instrument={} model={} backend={} device={} batch={} tokens={} steps={} warmup={} value={} unit={}",
            NV_MEASURE_V1_PREFIX_ONE_GREPPABLE_TOKEN_SO_TOOLING_PARSES_EVERY_INSTRUMENT_THE_SAME_WAY,
            quote_if_needed(&self.instrument),
            quote_if_needed(&self.model_at_rev),
            quote_if_needed(&self.backend),
            quote_if_needed(&self.device),
            self.batch,
            self.tokens,
            self.steps,
            self.warmup,
            self.value,
            quote_if_needed(&self.unit),
        );
        for (k, v) in &self.extras {
            assert_bare_key(k);
            s.push(' ');
            s.push_str(k);
            s.push('=');
            s.push_str(&quote_if_needed(v));
        }
        s
    }

    pub fn emit(&self) {
        eprintln!("{}", self.line());
    }

    pub fn parse(line: &str) -> Result<Self> {
        let rest = line
            .trim_end()
            .strip_prefix(
                NV_MEASURE_V1_PREFIX_ONE_GREPPABLE_TOKEN_SO_TOOLING_PARSES_EVERY_INSTRUMENT_THE_SAME_WAY,
            )
            .with_context(|| format!("not an NV-MEASURE v1 line: {line:?}"))?;
        let pairs = split_key_value_pairs_respecting_quotes(rest)?;
        let mut m = Measurement {
            instrument: String::new(),
            model_at_rev: String::new(),
            backend: String::new(),
            device: String::new(),
            batch: 0,
            tokens: 0,
            steps: 0,
            warmup: 0,
            value: f64::NAN,
            unit: String::new(),
            extras: Vec::new(),
        };
        let mut seen: Vec<&str> = Vec::new();
        for (k, v) in pairs {
            let required = REQUIRED_BASIS_KEYS_BECAUSE_COMPARISONS_FAILED_ON_SLAB_VS_SLOTSHARED_END_TO_END_VS_PER_GEMM_AND_MAX_SEQ_VS_DEPTH
                .iter()
                .find(|r| **r == k);
            if let Some(r) = required {
                if seen.contains(r) {
                    bail!("duplicate basis field {k}");
                }
                seen.push(r);
            }
            let count = |v: &str| v.parse::<usize>().with_context(|| format!("{k}={v:?} is not a count"));
            match k.as_str() {
                "instrument" => m.instrument = v,
                "model" => m.model_at_rev = v,
                "backend" => m.backend = v,
                "device" => m.device = v,
                "unit" => m.unit = v,
                "batch" => m.batch = count(&v)?,
                "tokens" => m.tokens = count(&v)?,
                "steps" => m.steps = count(&v)?,
                "warmup" => m.warmup = count(&v)?,
                "value" => {
                    m.value = v
                        .parse::<f64>()
                        .with_context(|| format!("value={v:?} is not a number"))?
                }
                _ => m.extras.push((k, v)),
            }
        }
        for r in REQUIRED_BASIS_KEYS_BECAUSE_COMPARISONS_FAILED_ON_SLAB_VS_SLOTSHARED_END_TO_END_VS_PER_GEMM_AND_MAX_SEQ_VS_DEPTH {
            if !seen.contains(&r) {
                bail!("missing required basis field {r} in {line:?}");
            }
        }
        Ok(m)
    }
}

fn split_key_value_pairs_respecting_quotes(s: &str) -> Result<Vec<(String, String)>> {
    let mut pairs = Vec::new();
    let mut chars = s.chars().peekable();
    loop {
        while chars.peek() == Some(&' ') {
            chars.next();
        }
        if chars.peek().is_none() {
            return Ok(pairs);
        }
        let mut key = String::new();
        loop {
            match chars.next() {
                Some('=') => break,
                Some(c) if c != ' ' => key.push(c),
                other => bail!("malformed field after key {key:?}: expected '=', got {other:?}"),
            }
        }
        let mut val = String::new();
        if chars.peek() == Some(&'"') {
            chars.next();
            loop {
                match chars.next() {
                    Some('"') => break,
                    Some('\\') => match chars.next() {
                        Some('"') => val.push('"'),
                        Some('\\') => val.push('\\'),
                        Some('n') => val.push('\n'),
                        other => bail!("bad escape \\{other:?} in value of {key}"),
                    },
                    Some(c) => val.push(c),
                    None => bail!("unterminated quoted value for {key}"),
                }
            }
        } else {
            while let Some(&c) = chars.peek() {
                if c == ' ' {
                    break;
                }
                val.push(c);
                chars.next();
            }
        }
        pairs.push((key, val));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Measurement {
        Measurement {
            instrument: "ctx-scaling".into(),
            model_at_rev: "unsloth/Qwen3.8-27B-NVFP4@e850c69".into(),
            backend: "wgpu".into(),
            device: "Example GPU (Vulkan)".into(),
            batch: 1,
            tokens: 8192,
            steps: 64,
            warmup: 200,
            value: 21.7,
            unit: "ms/tok".into(),
            extras: Vec::new(),
        }
        .extra("basis", "eager_dense_wgpu_decode_step_synthetic_state_fill")
        .extra("fill_s", "3.1")
    }

    #[test]
    fn golden_line_is_stable_because_downstream_greps_pin_it() {
        assert_eq!(
            sample().line(),
            "NV-MEASURE v1 instrument=ctx-scaling model=unsloth/Qwen3.8-27B-NVFP4@e850c69 \
             backend=wgpu device=\"Example GPU (Vulkan)\" batch=1 tokens=8192 \
             steps=64 warmup=200 value=21.7 unit=ms/tok \
             basis=eager_dense_wgpu_decode_step_synthetic_state_fill fill_s=3.1"
        );
    }

    #[test]
    fn roundtrip_preserves_every_field_including_quoted_device_and_extras_order() {
        let m = sample();
        let back = Measurement::parse(&m.line()).expect("parse own line");
        assert_eq!(back, m);
    }

    #[test]
    fn roundtrip_survives_quotes_backslashes_and_newlines_in_values() {
        let mut m = sample();
        m.device = "adapter \"X\" \\ weird\nname".into();
        m = m.extra("note", "a b\"c\\d");
        let back = Measurement::parse(&m.line()).expect("parse escaped line");
        assert_eq!(back, m);
    }

    #[test]
    fn value_roundtrips_bit_exactly_via_shortest_f64_display() {
        let mut m = sample();
        m.value = 20.900000000000002;
        let back = Measurement::parse(&m.line()).expect("parse");
        assert_eq!(back.value.to_bits(), m.value.to_bits());
    }

    #[test]
    fn parse_rejects_a_line_missing_a_required_basis_field_naming_the_field() {
        let line = sample().line().replace(" warmup=200", "");
        let err = Measurement::parse(&line).unwrap_err().to_string();
        assert!(err.contains("warmup"), "error must name the missing field: {err}");
    }

    #[test]
    fn parse_rejects_foreign_prefixes_so_legacy_lines_never_masquerade_as_canonical() {
        assert!(Measurement::parse("CTX-SCALING qwen38-wgpu depth=256").is_err());
        assert!(Measurement::parse("NV-MEASURE v2 instrument=x").is_err());
    }

    #[test]
    fn parse_rejects_duplicate_basis_fields() {
        let line = format!("{} tokens=512", sample().line());
        assert!(Measurement::parse(&line).is_err());
    }

    #[test]
    #[should_panic(expected = "shadows a required basis field")]
    fn extras_cannot_shadow_basis_fields() {
        let _ = sample().extra("tokens", "999");
    }
}
