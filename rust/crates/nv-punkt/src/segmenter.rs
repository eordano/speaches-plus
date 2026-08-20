use std::ops::Range;

use crate::params::{PunktParameters, ORTHO_BEG_LC, ORTHO_LC, ORTHO_MID_UC, ORTHO_UC};
use crate::token::{tokenize, Token};

pub(crate) fn first_pass(t: &mut Token, params: &PunktParameters) {
    let tok = t.text.as_str();
    if tok == "." || tok == "!" || tok == "?" {
        t.sentbreak = true;
    } else if t.is_ellipsis() {
        t.ellipsis = true;
    } else if t.period_final && !tok.ends_with("..") {
        let base = tok[..tok.len() - 1].to_lowercase();
        let last_dash = base.rsplit('-').next().unwrap_or("");
        if params.abbrev_types.contains(&base) || params.abbrev_types.contains(last_dash) {
            t.abbr = true;
        } else {
            t.sentbreak = true;
        }
    }
}

pub(crate) fn annotate_first_pass(tokens: &mut [Token], params: &PunktParameters) {
    for t in tokens.iter_mut() {
        first_pass(t, params);
    }
}

fn ortho_heuristic(params: &PunktParameters, t: &Token) -> Option<bool> {
    if matches!(t.text.as_str(), ";" | ":" | "," | "." | "!" | "?") {
        return Some(false);
    }
    let ortho = params.ortho(t.type_no_sentperiod());
    if t.first_upper() && (ortho & ORTHO_LC != 0) && (ortho & ORTHO_MID_UC == 0) {
        return Some(true);
    }
    if t.first_lower() && ((ortho & ORTHO_UC != 0) || (ortho & ORTHO_BEG_LC == 0)) {
        return Some(false);
    }
    None
}

fn second_pass(t1: &mut Token, t2: &Token, params: &PunktParameters) {
    if !t1.period_final {
        return;
    }
    let typ = t1.type_no_period().to_string();
    let next_typ = t2.type_no_sentperiod().to_string();
    let tok_is_initial = t1.is_initial();

    if params
        .collocations
        .contains(&(typ.clone(), next_typ.clone()))
    {
        t1.sentbreak = false;
        t1.abbr = true;
        return;
    }

    if (t1.abbr || t1.ellipsis) && !tok_is_initial {
        if ortho_heuristic(params, t2) == Some(true) {
            t1.sentbreak = true;
            return;
        }
        if t2.first_upper() && params.sent_starters.contains(&next_typ) {
            t1.sentbreak = true;
            return;
        }
    }

    if tok_is_initial || typ == "##number##" {
        match ortho_heuristic(params, t2) {
            Some(false) => {
                t1.sentbreak = false;
                t1.abbr = true;
            }
            None => {
                if tok_is_initial && t2.first_upper() && params.ortho(&next_typ) & ORTHO_LC == 0 {
                    t1.sentbreak = false;
                    t1.abbr = true;
                }
            }
            Some(true) => {}
        }
    }
}

pub(crate) fn annotate_second_pass(tokens: &mut [Token], params: &PunktParameters) {
    for i in 0..tokens.len().saturating_sub(1) {
        let (head, tail) = tokens.split_at_mut(i + 1);
        second_pass(&mut head[i], &tail[0], params);
    }
}

pub struct Segmenter {
    params: PunktParameters,
}

impl Segmenter {
    pub fn new(params: PunktParameters) -> Self {
        Self { params }
    }

    pub fn english() -> Self {
        Self::new(PunktParameters::english())
    }

    pub fn for_lang(lang: &str) -> Result<Self, String> {
        Ok(Self::new(PunktParameters::trained(lang)?))
    }

    pub fn params(&self) -> &PunktParameters {
        &self.params
    }

    pub fn annotated_tokens(&self, text: &str) -> Vec<Token> {
        let mut toks = tokenize(text);
        annotate_first_pass(&mut toks, &self.params);
        annotate_second_pass(&mut toks, &self.params);
        toks
    }

    pub fn sentences(&self, text: &str) -> Vec<Range<usize>> {
        let toks = self.annotated_tokens(text);
        let mut ranges: Vec<Range<usize>> = Vec::new();
        let mut start: Option<usize> = None;
        let mut last_end = 0usize;
        for t in &toks {
            if start.is_none() {
                start = Some(t.span.start);
            }
            last_end = t.span.end;
            if t.sentbreak {
                ranges.push(start.take().unwrap()..last_end);
            }
        }
        if let Some(s) = start {
            ranges.push(s..last_end);
        }
        realign(text, &mut ranges);
        ranges
    }

    pub fn sentence_strings<'a>(&self, text: &'a str) -> Vec<&'a str> {
        self.sentences(text).into_iter().map(|r| &text[r]).collect()
    }
}

fn is_closer(c: char) -> bool {
    matches!(c, '"' | '\'' | ')' | ']' | '}' | '\u{201d}' | '\u{2019}')
}

fn realign(text: &str, ranges: &mut Vec<Range<usize>>) {
    let mut i = 0;
    while i + 1 < ranges.len() {
        let next_start = ranges[i + 1].start;
        let next_end = ranges[i + 1].end;
        let mut p = next_start;
        while p < next_end {
            match text[p..].chars().next() {
                Some(c) if is_closer(c) => p += c.len_utf8(),
                _ => break,
            }
        }
        if p > next_start {
            let after_ok = p >= text.len()
                || text[p..].starts_with("--")
                || text[p..].chars().next().is_none_or(|c| c.is_whitespace());
            if after_ok {
                ranges[i].end = p;
                let mut q = p;
                while q < next_end {
                    match text[q..].chars().next() {
                        Some(c) if c.is_whitespace() => q += c.len_utf8(),
                        _ => break,
                    }
                }
                if q >= next_end {
                    ranges.remove(i + 1);
                    continue;
                }
                ranges[i + 1].start = q;
            }
        }
        i += 1;
    }
}
