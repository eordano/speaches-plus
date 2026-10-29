use anyhow::{bail, Context, Result};
use regex_syntax::hir::{Class, ClassUnicode, ClassUnicodeRange, Hir, HirKind, Look};

pub(crate) const STRING_CHAR: &str = r#"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})"#;

fn raw_json_chars() -> ClassUnicode {
    ClassUnicode::new([
        ClassUnicodeRange::new(' ', '!'),
        ClassUnicodeRange::new('#', '['),
        ClassUnicodeRange::new(']', char::MAX),
    ])
}

fn json_escape(c: char) -> String {
    match c {
        '"' => "\\\"".into(),
        '\\' => "\\\\".into(),
        '\n' => "\\n".into(),
        '\r' => "\\r".into(),
        '\t' => "\\t".into(),
        '\u{8}' => "\\b".into(),
        '\u{c}' => "\\f".into(),
        c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32),
        c => c.to_string(),
    }
}

fn escape_char(c: char) -> String {
    format!(r"\x{{{:x}}}", c as u32)
}

fn escaped_literal(c: char) -> String {
    json_escape(c).chars().map(escape_char).collect()
}

fn emit_class(cls: &ClassUnicode) -> String {
    let mut raw = cls.clone();
    raw.intersect(&raw_json_chars());
    let mut alts = Vec::new();
    if !raw.ranges().is_empty() {
        let body: String = raw
            .ranges()
            .iter()
            .map(|r| format!("{}-{}", escape_char(r.start()), escape_char(r.end())))
            .collect();
        alts.push(format!("[{body}]"));
    }
    for r in cls.ranges() {
        let lo = r.start() as u32;
        let hi = (r.end() as u32).min(0x5c);
        for cp in lo..=hi {
            let c = char::from_u32(cp).unwrap_or(' ');
            if cp < 0x20 || c == '"' || c == '\\' {
                alts.push(escaped_literal(c));
            }
        }
    }
    format!("(?:{})", alts.join("|"))
}

fn emit(h: &Hir, out: &mut String) -> Result<()> {
    match h.kind() {
        HirKind::Empty => {}
        HirKind::Literal(lit) => {
            let s = std::str::from_utf8(&lit.0).context("pattern literal is not UTF-8")?;
            for c in s.chars() {
                out.push_str(&escaped_literal(c));
            }
        }
        HirKind::Class(Class::Unicode(cls)) => out.push_str(&emit_class(cls)),
        HirKind::Class(Class::Bytes(_)) => {
            bail!("byte-oriented classes ((?-u)) are not supported in a string pattern")
        }
        HirKind::Look(look) => bail!(
            "the assertion {look:?} is not supported in a string pattern: only a leading ^ and a \
             trailing $ around the whole pattern are"
        ),
        HirKind::Repetition(rep) => {
            out.push_str("(?:");
            emit(&rep.sub, out)?;
            out.push(')');
            match rep.max {
                Some(max) => out.push_str(&format!("{{{},{max}}}", rep.min)),
                None => out.push_str(&format!("{{{},}}", rep.min)),
            }
        }
        HirKind::Capture(cap) => {
            out.push_str("(?:");
            emit(&cap.sub, out)?;
            out.push(')');
        }
        HirKind::Concat(items) => {
            for item in items {
                emit(item, out)?;
            }
        }
        HirKind::Alternation(alts) => {
            out.push_str("(?:");
            for (i, a) in alts.iter().enumerate() {
                if i > 0 {
                    out.push('|');
                }
                emit(a, out)?;
            }
            out.push(')');
        }
    }
    Ok(())
}

fn is_start(h: &Hir) -> bool {
    matches!(h.kind(), HirKind::Look(Look::Start))
}

fn is_end(h: &Hir) -> bool {
    matches!(h.kind(), HirKind::Look(Look::End))
}

pub(crate) fn json_string_pattern(pattern: &str) -> Result<String> {
    let hir = regex_syntax::ParserBuilder::new()
        .build()
        .parse(pattern)
        .with_context(|| format!("string pattern {pattern:?} is not a valid regular expression"))?;
    let (body, anchored_start, anchored_end): (Vec<&Hir>, bool, bool) = match hir.kind() {
        HirKind::Concat(items) => {
            let s = items.first().is_some_and(is_start);
            let e = items.len() > usize::from(s) && items.last().is_some_and(is_end);
            let lo = usize::from(s);
            let hi = items.len() - usize::from(e);
            (items[lo..hi].iter().collect(), s, e)
        }
        _ if is_start(&hir) => (Vec::new(), true, false),
        _ if is_end(&hir) => (Vec::new(), false, true),
        _ => (vec![&hir], false, false),
    };
    let mut out = String::from("\"");
    if !anchored_start {
        out.push_str(STRING_CHAR);
        out.push('*');
    }
    out.push_str("(?:");
    for h in body {
        emit(h, &mut out).with_context(|| format!("string pattern {pattern:?}"))?;
    }
    out.push(')');
    if !anchored_end {
        out.push_str(STRING_CHAR);
        out.push('*');
    }
    out.push('"');
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full(re: &str, s: &str) -> bool {
        regex_automata::meta::Regex::new(&format!(r"\A(?:{re})\z"))
            .unwrap()
            .is_match(s)
    }

    fn json(s: &str) -> String {
        serde_json::to_string(s).unwrap()
    }

    #[test]
    fn an_anchored_pattern_matches_exactly_its_strings_and_an_unanchored_one_searches() {
        let re = json_string_pattern("^[0-9]{5}$").unwrap();
        assert!(full(&re, &json("20500")));
        assert!(!full(&re, &json("2050")));
        assert!(!full(&re, &json("205001")));
        assert!(!full(&re, "20500"));
        let re = json_string_pattern("[0-9]+").unwrap();
        assert!(full(&re, &json("abc 12 def")), "an unanchored pattern is a search");
        assert!(!full(&re, &json("no digits")));
        let left = json_string_pattern("^ab").unwrap();
        assert!(full(&left, &json("abzzz")) && !full(&left, &json("zab")));
    }

    #[test]
    fn quotes_backslashes_and_controls_are_admitted_only_escaped_and_inner_assertions_are_refused() {
        let re = json_string_pattern(r#"^.*$"#).unwrap();
        for s in ["plain", "say \"hi\"", r"back\slash", "tab\there", "é 中 🌍"] {
            assert!(full(&re, &json(s)), "{s:?} as {}", json(s));
        }
        assert!(!full(&re, "\"say \"hi\"\""), "a raw quote would end the JSON string");
        let lit = json_string_pattern(r#"^a"b\\c$"#).unwrap();
        assert!(full(&lit, &json("a\"b\\c")));
        let tab = json_string_pattern(r"^\t$").unwrap();
        assert!(full(&tab, &json("\t")) && !full(&tab, "\"\t\""));
        assert!(json_string_pattern(r"^a\bb$").is_err(), "inner assertions are refused");
        assert!(json_string_pattern(r"(?m)^a$").is_err());
        assert!(json_string_pattern("[").is_err());
    }

}
