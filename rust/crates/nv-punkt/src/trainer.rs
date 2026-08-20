use std::collections::{HashMap, HashSet};

use crate::params::{
    PunktParameters, ORTHO_BEG_LC, ORTHO_BEG_UC, ORTHO_MID_LC, ORTHO_MID_UC, ORTHO_UNK_LC,
    ORTHO_UNK_UC,
};
use crate::segmenter::annotate_first_pass;
use crate::token::{tokenize, Token};

const ABBREV: f64 = 0.3;
const ABBREV_BACKOFF: u64 = 5;
const COLLOCATION: f64 = 7.88;
const SENT_STARTER: f64 = 30.0;
const MIN_COLLOC_FREQ: u64 = 1;
const INTERNAL_PUNCT: &[char] = &[',', ':', ';'];

fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x * y.ln()
    }
}

fn dunning_ll(count_a: f64, count_b: f64, count_ab: f64, n: f64) -> f64 {
    let p1 = (count_b / n).min(0.999_999);
    let p2 = 0.99f64;
    let null_hypo = xlogy(count_ab, p1) + xlogy(count_a - count_ab, 1.0 - p1);
    let alt_hypo = xlogy(count_ab, p2) + xlogy(count_a - count_ab, 1.0 - p2);
    -2.0 * (null_hypo - alt_hypo)
}

fn col_ll(count_a: f64, count_b: f64, count_ab: f64, n: f64) -> f64 {
    let p = count_b / n;
    let s1 = xlogy(count_ab, p) + xlogy(count_a - count_ab, 1.0 - p);
    let s2 = xlogy(count_b - count_ab, p) + xlogy(n - count_a - count_b + count_ab, 1.0 - p);
    let s3 = if count_a == count_ab {
        0.0
    } else {
        let p1 = count_ab / count_a;
        xlogy(count_ab, p1) + xlogy(count_a - count_ab, 1.0 - p1)
    };
    let s4 = if count_b == count_ab {
        0.0
    } else {
        let p2 = (count_b - count_ab) / (n - count_a);
        xlogy(count_b - count_ab, p2) + xlogy(n - count_a - count_b + count_ab, 1.0 - p2)
    };
    -2.0 * (s1 + s2 - s3 - s4)
}

#[derive(Default)]
pub struct PunktTrainer {
    params: PunktParameters,
    type_fdist: HashMap<String, u64>,
    total_toks: u64,
    num_period_toks: u64,
    collocation_fdist: HashMap<(String, String), u64>,
    sent_starter_fdist: HashMap<String, u64>,
    sentbreak_count: u64,
}

impl PunktTrainer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn params(&self) -> &PunktParameters {
        &self.params
    }

    pub fn type_count(&self, typ: &str) -> u64 {
        self.type_fdist.get(typ).copied().unwrap_or(0)
    }

    pub fn train(&mut self, text: &str) {
        let mut tokens = tokenize(text);
        for t in &tokens {
            *self.type_fdist.entry(t.typ.clone()).or_insert(0) += 1;
            self.total_toks += 1;
            if t.period_final {
                self.num_period_toks += 1;
            }
        }

        let unique: HashSet<String> = tokens.iter().map(|t| t.typ.clone()).collect();
        let mut add = Vec::new();
        let mut remove = Vec::new();
        for typ in &unique {
            if let Some((cand, score, is_add)) = self.reclassify(typ) {
                if score >= ABBREV {
                    if is_add {
                        add.push(cand);
                    }
                } else if !is_add {
                    remove.push(cand);
                }
            }
        }
        for a in add {
            self.params.abbrev_types.insert(a);
        }
        for r in remove {
            self.params.abbrev_types.remove(&r);
        }

        annotate_first_pass(&mut tokens, &self.params);
        self.get_orthography_data(&tokens);
        for t in &tokens {
            if t.sentbreak {
                self.sentbreak_count += 1;
            }
        }

        let mut rare = Vec::new();
        for i in 0..tokens.len().saturating_sub(1) {
            let t1 = &tokens[i];
            let t2 = &tokens[i + 1];
            if !t1.period_final {
                continue;
            }
            if self.is_rare_abbrev(t1, t2) {
                rare.push(t1.type_no_period().to_string());
            }
            if t1.sentbreak && !(t1.is_number() || t1.is_initial()) && t2.is_alpha() {
                *self.sent_starter_fdist.entry(t2.typ.clone()).or_insert(0) += 1;
            }
            if t1.sentbreak
                && (t1.is_number() || t1.is_initial())
                && t1.is_non_punct()
                && t2.is_non_punct()
            {
                *self
                    .collocation_fdist
                    .entry((
                        t1.type_no_period().to_string(),
                        t2.type_no_sentperiod().to_string(),
                    ))
                    .or_insert(0) += 1;
            }
        }
        for r in rare {
            self.params.abbrev_types.insert(r);
        }
    }

    fn reclassify(&self, typ: &str) -> Option<(String, f64, bool)> {
        if !typ.chars().any(|c| c.is_alphabetic()) || typ == "##number##" {
            return None;
        }
        let (cand, is_add) = if let Some(stripped) = typ.strip_suffix('.') {
            if self.params.abbrev_types.contains(typ) {
                return None;
            }
            (stripped.to_string(), true)
        } else {
            if !self.params.abbrev_types.contains(typ) {
                return None;
            }
            (typ.to_string(), false)
        };
        let num_periods = cand.matches('.').count() as f64 + 1.0;
        let num_nonperiods = cand.chars().count() as f64 - cand.matches('.').count() as f64 + 1.0;
        let count_with = self.type_count(&format!("{cand}.")) as f64;
        let count_without = self.type_count(&cand) as f64;
        let ll = dunning_ll(
            count_with + count_without,
            self.num_period_toks as f64,
            count_with,
            self.total_toks as f64,
        );
        let f_length = (-num_nonperiods).exp();
        let f_penalty = num_nonperiods.powf(-count_without);
        let score = ll * f_length * num_periods * f_penalty;
        Some((cand, score, is_add))
    }

    fn get_orthography_data(&mut self, tokens: &[Token]) {
        #[derive(PartialEq, Clone, Copy)]
        enum Ctx {
            Internal,
            Initial,
            Unknown,
        }
        let mut context = Ctx::Internal;
        for t in tokens {
            if t.parastart && context != Ctx::Unknown {
                context = Ctx::Initial;
            }
            if t.linestart && context == Ctx::Internal {
                context = Ctx::Unknown;
            }
            let flag = if t.first_upper() {
                match context {
                    Ctx::Initial => ORTHO_BEG_UC,
                    Ctx::Internal => ORTHO_MID_UC,
                    Ctx::Unknown => ORTHO_UNK_UC,
                }
            } else if t.first_lower() {
                match context {
                    Ctx::Initial => ORTHO_BEG_LC,
                    Ctx::Internal => ORTHO_MID_LC,
                    Ctx::Unknown => ORTHO_UNK_LC,
                }
            } else {
                0
            };
            if flag != 0 {
                let typ = t.type_no_sentperiod().to_string();
                self.params.add_ortho_context(&typ, flag);
            }
            if t.sentbreak {
                context = if t.is_number() || t.is_initial() {
                    Ctx::Unknown
                } else {
                    Ctx::Initial
                };
            } else if t.ellipsis || t.abbr {
                context = Ctx::Unknown;
            } else {
                context = Ctx::Internal;
            }
        }
    }

    fn is_rare_abbrev(&self, t1: &Token, t2: &Token) -> bool {
        if t1.abbr || !t1.sentbreak {
            return false;
        }
        let typ = t1.type_no_sentperiod();
        let shorter: String = {
            let mut s = typ.to_string();
            s.pop();
            s
        };
        let count = self.type_count(typ) + self.type_count(&shorter);
        if self.params.abbrev_types.contains(typ) || count >= ABBREV_BACKOFF {
            return false;
        }
        let first = t2.text.chars().next().unwrap_or(' ');
        if INTERNAL_PUNCT.contains(&first) {
            return true;
        }
        if t2.first_lower() {
            let ortho = self.params.ortho(t2.type_no_sentperiod());
            if (ortho & ORTHO_BEG_UC != 0) && (ortho & ORTHO_MID_UC == 0) {
                return true;
            }
        }
        false
    }

    pub fn finalize(&mut self) {
        self.params.sent_starters.clear();
        self.params.collocations.clear();

        let n = self.total_toks as f64;
        let mut starters: Vec<String> = Vec::new();
        for (typ, &count_at_break) in &self.sent_starter_fdist {
            if typ.is_empty() {
                continue;
            }
            let typ_count = self.type_count(typ) + self.type_count(&format!("{typ}."));
            if typ_count < count_at_break {
                continue;
            }
            let ll = col_ll(
                self.sentbreak_count as f64,
                typ_count as f64,
                count_at_break as f64,
                n,
            );
            if ll >= SENT_STARTER
                && n / self.sentbreak_count as f64 > typ_count as f64 / count_at_break as f64
            {
                starters.push(typ.clone());
            }
        }
        for s in starters {
            self.params.sent_starters.insert(s);
        }

        let mut collocs: Vec<(String, String)> = Vec::new();
        for ((t1, t2), &count) in &self.collocation_fdist {
            if self.params.sent_starters.contains(t2) {
                continue;
            }
            let c1 = self.type_count(t1) + self.type_count(&format!("{t1}."));
            let c2 = self.type_count(t2) + self.type_count(&format!("{t2}."));
            if c1 > 1 && c2 > 1 && MIN_COLLOC_FREQ <= count && count <= c1.min(c2) {
                let ll = col_ll(c1 as f64, c2 as f64, count as f64, n);
                if ll >= COLLOCATION && n / c1 as f64 > c2 as f64 / count as f64 {
                    collocs.push((t1.clone(), t2.clone()));
                }
            }
        }
        for c in collocs {
            self.params.collocations.insert(c);
        }
    }

    pub fn into_params(mut self) -> PunktParameters {
        self.finalize();
        self.params
    }
}
