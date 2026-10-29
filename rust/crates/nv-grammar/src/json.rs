#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Lex {
    Start,
    Value,
    ArrayFirst,
    ObjectFirst,
    Key,
    Colon,
    After,
    Str { key: bool },
    Esc { key: bool },
    Hex { key: bool },
    Lit { word: u8 },
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
    Exp,
    ExpSign,
    ExpInt,
}

const LITERALS: [&[u8]; 3] = [b"true", b"false", b"null"];
const NESTING_LIMIT_IS_THE_U64_STACK_WIDTH: u8 = 64;
const WHITESPACE_RUN_LIMIT_MATCHES_THE_SCHEMA_REGEX_WS_SO_NO_RUNAWAY_BLANKS: u8 = 20;
const INT_DIGITS_LIMIT_MATCHES_THE_SCHEMA_REGEX_NUMBER: u8 = 20;
const FRAC_DIGITS_LIMIT_MATCHES_THE_SCHEMA_REGEX_NUMBER: u8 = 17;
const EXP_DIGITS_LIMIT_MATCHES_THE_SCHEMA_REGEX_NUMBER: u8 = 4;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Json {
    pub(crate) object_only: bool,
    stack: u64,
    depth: u8,
    lex: Lex,
    n: u8,
}

pub(crate) enum Step {
    Next(Json),
    Done,
    DoneBeforeThisByte,
    Reject,
}

impl Json {
    pub(crate) fn start(object_only: bool) -> Self {
        Self {
            object_only,
            stack: 0,
            depth: 0,
            lex: Lex::Start,
            n: 0,
        }
    }

    pub(crate) fn complete(&self) -> bool {
        self.depth == 0 && matches!(self.lex, Lex::Zero | Lex::Int | Lex::Frac | Lex::ExpInt)
    }

    fn to(self, lex: Lex, n: u8) -> Step {
        Step::Next(Self { lex, n, ..self })
    }

    fn top_is_object(&self) -> bool {
        self.depth > 0 && (self.stack >> (self.depth - 1)) & 1 == 1
    }

    fn value_done(self) -> Step {
        if self.depth == 0 {
            Step::Done
        } else {
            self.to(Lex::After, 0)
        }
    }

    fn open(self, object: bool) -> Step {
        if self.depth == NESTING_LIMIT_IS_THE_U64_STACK_WIDTH {
            return Step::Reject;
        }
        let bit = 1u64 << self.depth;
        Step::Next(Self {
            stack: if object {
                self.stack | bit
            } else {
                self.stack & !bit
            },
            depth: self.depth + 1,
            lex: if object {
                Lex::ObjectFirst
            } else {
                Lex::ArrayFirst
            },
            n: 0,
            ..self
        })
    }

    fn close(self, object: bool) -> Step {
        if self.depth == 0 || self.top_is_object() != object {
            return Step::Reject;
        }
        Self {
            depth: self.depth - 1,
            ..self
        }
        .value_done()
    }

    fn blank(self, b: u8) -> Option<Step> {
        matches!(b, b' ' | b'\t' | b'\n').then(|| {
            if self.n < WHITESPACE_RUN_LIMIT_MATCHES_THE_SCHEMA_REGEX_WS_SO_NO_RUNAWAY_BLANKS {
                self.to(self.lex, self.n + 1)
            } else {
                Step::Reject
            }
        })
    }

    fn value(self, b: u8) -> Step {
        match b {
            b'{' => self.open(true),
            b'[' => self.open(false),
            b'"' => self.to(Lex::Str { key: false }, 0),
            b'-' => self.to(Lex::Minus, 0),
            b'0' => self.to(Lex::Zero, 1),
            b'1'..=b'9' => self.to(Lex::Int, 1),
            b't' => self.to(Lex::Lit { word: 0 }, 1),
            b'f' => self.to(Lex::Lit { word: 1 }, 1),
            b'n' => self.to(Lex::Lit { word: 2 }, 1),
            _ => Step::Reject,
        }
    }

    fn string(self, key: bool, b: u8) -> Step {
        if self.n > 0 {
            return if b & 0xc0 == 0x80 {
                self.to(self.lex, self.n - 1)
            } else {
                Step::Reject
            };
        }
        match b {
            b'"' if key => self.to(Lex::Colon, 0),
            b'"' => self.value_done(),
            b'\\' => self.to(Lex::Esc { key }, 0),
            0x00..=0x1f | 0x80..=0xc1 | 0xf5..=0xff => Step::Reject,
            0xc2..=0xdf => self.to(self.lex, 1),
            0xe0..=0xef => self.to(self.lex, 2),
            0xf0..=0xf4 => self.to(self.lex, 3),
            _ => self.to(self.lex, 0),
        }
    }

    fn number_end(self, b: u8) -> Step {
        if self.depth == 0 {
            Step::DoneBeforeThisByte
        } else {
            Self {
                lex: Lex::After,
                n: 0,
                ..self
            }
            .step(b)
        }
    }

    pub(crate) fn step(self, b: u8) -> Step {
        let exp = matches!(b, b'e' | b'E');
        match self.lex {
            Lex::Start if self.object_only => match b {
                b'{' => self.open(true),
                _ => Step::Reject,
            },
            Lex::Start => self.value(b),
            Lex::Value => self.blank(b).unwrap_or_else(|| self.value(b)),
            Lex::ArrayFirst => self.blank(b).unwrap_or_else(|| match b {
                b']' => self.close(false),
                _ => self.value(b),
            }),
            Lex::ObjectFirst => self.blank(b).unwrap_or_else(|| match b {
                b'"' => self.to(Lex::Str { key: true }, 0),
                b'}' => self.close(true),
                _ => Step::Reject,
            }),
            Lex::Key => self.blank(b).unwrap_or_else(|| match b {
                b'"' => self.to(Lex::Str { key: true }, 0),
                _ => Step::Reject,
            }),
            Lex::Colon => self.blank(b).unwrap_or_else(|| match b {
                b':' => self.to(Lex::Value, 0),
                _ => Step::Reject,
            }),
            Lex::After => self.blank(b).unwrap_or_else(|| match b {
                b',' if self.top_is_object() => self.to(Lex::Key, 0),
                b',' => self.to(Lex::Value, 0),
                b'}' | b']' => self.close(b == b'}'),
                _ => Step::Reject,
            }),
            Lex::Str { key } => self.string(key, b),
            Lex::Esc { key } => match b {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    self.to(Lex::Str { key }, 0)
                }
                b'u' => self.to(Lex::Hex { key }, 4),
                _ => Step::Reject,
            },
            Lex::Hex { key } if b.is_ascii_hexdigit() && self.n == 1 => {
                self.to(Lex::Str { key }, 0)
            }
            Lex::Hex { .. } if b.is_ascii_hexdigit() => self.to(self.lex, self.n - 1),
            Lex::Hex { .. } => Step::Reject,
            Lex::Lit { word } => {
                let w = LITERALS[word as usize];
                match w.get(self.n as usize) {
                    Some(&c) if c == b && self.n as usize + 1 == w.len() => self.value_done(),
                    Some(&c) if c == b => self.to(self.lex, self.n + 1),
                    _ => Step::Reject,
                }
            }
            Lex::Minus => match b {
                b'0' => self.to(Lex::Zero, 1),
                b'1'..=b'9' => self.to(Lex::Int, 1),
                _ => Step::Reject,
            },
            Lex::Zero if b == b'.' => self.to(Lex::Dot, 0),
            Lex::Zero if exp => self.to(Lex::Exp, 0),
            Lex::Zero => self.number_end(b),
            Lex::Int if b == b'.' => self.to(Lex::Dot, 0),
            Lex::Int if exp => self.to(Lex::Exp, 0),
            Lex::Int
                if b.is_ascii_digit()
                    && self.n < INT_DIGITS_LIMIT_MATCHES_THE_SCHEMA_REGEX_NUMBER =>
            {
                self.to(self.lex, self.n + 1)
            }
            Lex::Int => self.number_end(b),
            Lex::Dot if b.is_ascii_digit() => self.to(Lex::Frac, 1),
            Lex::Dot => Step::Reject,
            Lex::Frac if exp => self.to(Lex::Exp, 0),
            Lex::Frac
                if b.is_ascii_digit()
                    && self.n < FRAC_DIGITS_LIMIT_MATCHES_THE_SCHEMA_REGEX_NUMBER =>
            {
                self.to(self.lex, self.n + 1)
            }
            Lex::Frac => self.number_end(b),
            Lex::Exp if matches!(b, b'+' | b'-') => self.to(Lex::ExpSign, 0),
            Lex::Exp | Lex::ExpSign if b.is_ascii_digit() => self.to(Lex::ExpInt, 1),
            Lex::Exp | Lex::ExpSign => Step::Reject,
            Lex::ExpInt
                if b.is_ascii_digit()
                    && self.n < EXP_DIGITS_LIMIT_MATCHES_THE_SCHEMA_REGEX_NUMBER =>
            {
                self.to(self.lex, self.n + 1)
            }
            Lex::ExpInt => self.number_end(b),
        }
    }
}
