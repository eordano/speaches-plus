use std::collections::{HashMap, HashSet};
use std::path::Path;

pub const PARAMS_ENV: &str = "NV_PUNKT_DATA";

pub const ORTHO_BEG_UC: u8 = 1 << 1;
pub const ORTHO_MID_UC: u8 = 1 << 2;
pub const ORTHO_UNK_UC: u8 = 1 << 3;
pub const ORTHO_BEG_LC: u8 = 1 << 4;
pub const ORTHO_MID_LC: u8 = 1 << 5;
pub const ORTHO_UNK_LC: u8 = 1 << 6;
pub const ORTHO_UC: u8 = ORTHO_BEG_UC | ORTHO_MID_UC | ORTHO_UNK_UC;
pub const ORTHO_LC: u8 = ORTHO_BEG_LC | ORTHO_MID_LC | ORTHO_UNK_LC;

pub const CURATED_ABBREVS: &[&str] = &[
    "adm", "al", "apr", "approx", "aug", "ave", "b.a", "blvd", "brig", "capt", "cf", "cmdr", "co",
    "col", "corp", "cpl", "dec", "dept", "dr", "e.g", "ed", "est", "et", "etc", "feb", "fig",
    "figs", "fri", "ft", "gen", "gov", "hon", "hr", "i.e", "inc", "jan", "jr", "jul", "jun", "lt",
    "ltd", "m.d", "maj", "mar", "messrs", "mfg", "mgr", "mlle", "mme", "mon", "mr", "mrs", "ms",
    "msgr", "mt", "nov", "oct", "p.m", "ph.d", "pp", "prof", "pvt", "rep", "rev", "rd", "sat",
    "sen", "sep", "sept", "sgt", "sr", "st", "sun", "thu", "thurs", "tue", "tues", "u.k", "u.n",
    "u.s", "u.s.a", "univ", "v", "vol", "vols", "vs", "wed",
];

#[derive(Debug, Clone, Default)]
pub struct PunktParameters {
    pub abbrev_types: HashSet<String>,
    pub collocations: HashSet<(String, String)>,
    pub sent_starters: HashSet<String>,
    pub ortho_context: HashMap<String, u8>,
}

impl PunktParameters {
    pub fn add_ortho_context(&mut self, typ: &str, flag: u8) {
        *self.ortho_context.entry(typ.to_string()).or_insert(0) |= flag;
    }

    pub fn ortho(&self, typ: &str) -> u8 {
        self.ortho_context.get(typ).copied().unwrap_or(0)
    }

    pub fn curated() -> Self {
        let mut p = Self::default();
        for a in CURATED_ABBREVS {
            p.abbrev_types.insert((*a).to_string());
        }
        p
    }

    pub fn load_punkt_tab(dir: &Path) -> Result<Self, String> {
        let read = |name: &str| {
            let p = dir.join(name);
            std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))
        };
        let pair = |name: &'static str, line: &str| {
            line.split_once('\t')
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .ok_or_else(|| format!("{}/{name}: bad line {line:?}", dir.display()))
        };
        let mut p = Self::default();
        for l in read("abbrev_types.txt")?.lines().filter(|l| !l.is_empty()) {
            p.abbrev_types.insert(l.to_string());
        }
        for l in read("sent_starters.txt")?.lines().filter(|l| !l.is_empty()) {
            p.sent_starters.insert(l.to_string());
        }
        for l in read("collocations.tab")?.lines().filter(|l| !l.is_empty()) {
            p.collocations.insert(pair("collocations.tab", l)?);
        }
        for l in read("ortho_context.tab")?.lines().filter(|l| !l.is_empty()) {
            let (t, f) = pair("ortho_context.tab", l)?;
            let f = f
                .parse()
                .map_err(|_| format!("{}/ortho_context.tab: bad flag {l:?}", dir.display()))?;
            p.ortho_context.insert(t, f);
        }
        Ok(p)
    }

    pub fn trained(lang: &str) -> Result<Self, String> {
        let root = std::env::var(PARAMS_ENV).map_err(|_| format!("{PARAMS_ENV} unset"))?;
        Self::load_punkt_tab(&Path::new(&root).join(lang))
    }

    pub fn english_trained() -> Result<Self, String> {
        let mut p = Self::trained("english")?;
        for a in CURATED_ABBREVS {
            p.abbrev_types.insert((*a).to_string());
        }
        Ok(p)
    }

    pub fn english() -> Self {
        Self::english_trained().unwrap_or_else(|e| {
            eprintln!("nv-punkt: {e}; falling back to curated abbreviations only");
            Self::curated()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trained_params_load() {
        let p = PunktParameters::english_trained().unwrap();
        assert!(p.ortho_context.len() >= 10_000);
        assert!(p.sent_starters.len() >= 10);
        assert!(!p.collocations.is_empty());
        assert!(p.abbrev_types.len() > CURATED_ABBREVS.len());
    }

    #[test]
    fn multilingual_langs_load() {
        for lang in ["german", "spanish", "portuguese", "french"] {
            let p = PunktParameters::trained(lang).unwrap();
            assert!(p.ortho_context.len() >= 1000, "{lang}");
            assert!(!p.abbrev_types.is_empty(), "{lang}");
        }
        let ru = PunktParameters::trained("russian").unwrap();
        assert!(ru.abbrev_types.len() >= 1000);
    }
}
