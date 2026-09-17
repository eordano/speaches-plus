use std::collections::HashMap;

use anyhow::{Context, Result};

use super::npz::{parse_npy, Voice};

const BUILTIN: &[(&str, &[u8])] = &[];

pub fn load() -> Result<HashMap<String, Voice>> {
    let mut out = HashMap::with_capacity(BUILTIN.len());
    for (name, bytes) in BUILTIN {
        let voice = parse_npy(bytes).with_context(|| format!("builtin voice {name}"))?;
        out.insert((*name).to_string(), voice);
    }
    Ok(out)
}

pub fn names() -> impl Iterator<Item = &'static str> {
    BUILTIN.iter().map(|(name, _)| *name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_voices_parse_as_style_packs() {
        let voices = load().unwrap();
        assert_eq!(voices.len(), BUILTIN.len());
        for name in names() {
            let v = &voices[name];
            assert_eq!(v.shape.len(), 3, "{name}: shape {:?}", v.shape);
            assert_eq!(v.shape[1..], [1, 256], "{name}: shape {:?}", v.shape);
            let row = v.row(0).unwrap();
            assert_eq!(row.len(), 256);
            assert!(
                row.iter().all(|x| x.is_finite()),
                "{name}: non-finite style value"
            );
        }
    }
}
