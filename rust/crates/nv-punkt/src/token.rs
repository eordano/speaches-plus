use std::ops::Range;

#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    pub span: Range<usize>,
    pub parastart: bool,
    pub linestart: bool,
    pub typ: String,
    pub period_final: bool,
    pub sentbreak: bool,
    pub abbr: bool,
    pub ellipsis: bool,
}

impl Token {
    pub fn new(text: &str, span: Range<usize>, parastart: bool, linestart: bool) -> Self {
        let lower = text.to_lowercase();
        let typ = if is_numeric(&lower) {
            "##number##".to_string()
        } else {
            lower
        };
        Token {
            period_final: text.ends_with('.'),
            text: text.to_string(),
            span,
            parastart,
            linestart,
            typ,
            sentbreak: false,
            abbr: false,
            ellipsis: false,
        }
    }

    pub fn type_no_period(&self) -> &str {
        if self.typ.len() > 1 && self.typ.ends_with('.') {
            &self.typ[..self.typ.len() - 1]
        } else {
            &self.typ
        }
    }

    pub fn type_no_sentperiod(&self) -> &str {
        if self.sentbreak {
            self.type_no_period()
        } else {
            &self.typ
        }
    }

    pub fn first_upper(&self) -> bool {
        self.text.chars().next().is_some_and(|c| c.is_uppercase())
    }

    pub fn first_lower(&self) -> bool {
        self.text.chars().next().is_some_and(|c| c.is_lowercase())
    }

    pub fn is_ellipsis(&self) -> bool {
        if self.text == "\u{2026}" {
            return true;
        }
        let dots = self.text.chars().filter(|&c| c == '.').count();
        dots >= 2 && self.text.chars().all(|c| c == '.' || c == ' ')
    }

    pub fn is_initial(&self) -> bool {
        let mut it = self.text.chars();
        matches!(
            (it.next(), it.next(), it.next()),
            (Some(a), Some('.'), None) if a.is_alphabetic()
        )
    }

    pub fn is_number(&self) -> bool {
        self.typ.starts_with("##number##")
    }

    pub fn is_alpha(&self) -> bool {
        !self.typ.is_empty() && self.typ.chars().all(|c| c.is_alphabetic())
    }

    pub fn is_non_punct(&self) -> bool {
        self.typ.chars().any(|c| c.is_alphabetic())
    }
}

fn is_numeric(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    if i < chars.len() && chars[i] == '-' {
        i += 1;
    }
    if i < chars.len() && (chars[i] == '.' || chars[i] == ',') {
        i += 1;
    }
    if i >= chars.len() || !chars[i].is_ascii_digit() {
        return false;
    }
    i += 1;
    while i < chars.len()
        && (chars[i].is_ascii_digit() || chars[i] == ',' || chars[i] == '.' || chars[i] == '-')
    {
        i += 1;
    }
    i == chars.len()
}

const NON_WORD: &[char] = &[
    '?', '!', ')', '"', ';', '}', ']', '*', ':', '@', '\'', '(', '{', '[', '\u{201d}',
    '\u{2019}', '\u{00bb}',
];
const WORD_START_EXCLUDE: &[char] = &[
    '(', '"', '`', '{', '[', ':', ';', '&', '#', '*', '@', ')', '}', ']', '-', ',',
    '\u{201c}', '\u{2018}', '\u{00ab}',
];

pub fn tokenize(text: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut parastart = false;
    let mut pos = 0usize;
    for line in text.split('\n') {
        let base = pos;
        pos += line.len() + 1;
        if line.trim().is_empty() {
            parastart = true;
            continue;
        }
        let before = out.len();
        tokenize_line(line, base, &mut out);
        if out.len() > before {
            out[before].linestart = true;
            out[before].parastart = parastart;
            parastart = false;
        }
    }
    out
}

fn tokenize_line(line: &str, base: usize, out: &mut Vec<Token>) {
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        let (_, c) = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        let start_i = i;
        if (c == '-' || c == '.') && i + 1 < n && chars[i + 1].1 == c {
            while i < n && chars[i].1 == c {
                i += 1;
            }
        } else if c == '\u{2026}' {
            i += 1;
        } else if !WORD_START_EXCLUDE.contains(&c) {
            i += 1;
            while i < n {
                let d = chars[i].1;
                if d.is_whitespace() || NON_WORD.contains(&d) || d == '\u{2026}' {
                    break;
                }
                if (d == '-' || d == '.') && i + 1 < n && chars[i + 1].1 == d {
                    break;
                }
                if d == ',' {
                    match chars.get(i + 1).map(|&(_, x)| x) {
                        None => break,
                        Some(x) if x.is_whitespace() || NON_WORD.contains(&x) => break,
                        _ => {}
                    }
                }
                i += 1;
            }
        } else {
            i += 1;
        }
        let sb = chars[start_i].0;
        let eb = if i < n { chars[i].0 } else { line.len() };
        out.push(Token::new(
            &line[sb..eb],
            base + sb..base + eb,
            false,
            false,
        ));
    }
}
