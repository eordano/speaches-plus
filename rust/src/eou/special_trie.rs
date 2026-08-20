#![allow(dead_code)]

use std::collections::HashMap;

#[derive(Default)]
pub struct SpecialNode {
    children: HashMap<u8, SpecialNode>,
    terminal: bool,
    id: i64,
    content: String,
}

#[derive(Debug, Clone)]
pub struct SpecialPiece {
    pub text: String,
    pub id: i64,
    pub special: bool,
}

impl SpecialNode {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, s: &str, id: i64) {
        let mut cur = self;
        for b in s.as_bytes() {
            cur = cur.children.entry(*b).or_default();
        }
        cur.terminal = true;
        cur.id = id;
        cur.content = s.to_string();
    }

    pub fn split(&self, text: &str) -> Vec<SpecialPiece> {
        let bytes = text.as_bytes();
        let mut out: Vec<SpecialPiece> = Vec::new();
        let mut plain: Vec<u8> = Vec::new();
        let mut i = 0usize;
        let flush = |plain: &mut Vec<u8>, out: &mut Vec<SpecialPiece>| {
            if !plain.is_empty() {
                let s = String::from_utf8_lossy(plain).into_owned();
                out.push(SpecialPiece {
                    text: s,
                    id: -1,
                    special: false,
                });
                plain.clear();
            }
        };
        while i < bytes.len() {
            let (id, mlen, mtext) = self.match_at(bytes, i);
            if mlen > 0 {
                flush(&mut plain, &mut out);
                out.push(SpecialPiece {
                    text: mtext,
                    id,
                    special: true,
                });
                i += mlen;
            } else {
                plain.push(bytes[i]);
                i += 1;
            }
        }
        flush(&mut plain, &mut out);
        out
    }

    fn match_at(&self, bytes: &[u8], start: usize) -> (i64, usize, String) {
        let mut cur = self;
        let mut match_id: i64 = -1;
        let mut match_end: usize = 0;
        let mut match_text = String::new();
        let mut i = start;
        while i < bytes.len() {
            match cur.children.get(&bytes[i]) {
                Some(next) => {
                    cur = next;
                    if cur.terminal {
                        match_id = cur.id;
                        match_end = i + 1;
                        match_text = cur.content.clone();
                    }
                    i += 1;
                }
                None => break,
            }
        }
        if match_end == 0 {
            return (-1, 0, String::new());
        }
        (match_id, match_end - start, match_text)
    }
}
