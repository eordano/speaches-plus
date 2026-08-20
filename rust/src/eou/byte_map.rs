#![allow(dead_code)]

use std::sync::OnceLock;

fn build_byte_to_char() -> [char; 256] {
    let mut keep = [false; 256];
    for b in 0u32..256 {
        let k = (b >= b'!' as u32 && b <= b'~' as u32)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b);
        keep[b as usize] = k;
    }
    let mut bs: Vec<u32> = (0u32..256).filter(|b| keep[*b as usize]).collect();
    let mut cs: Vec<u32> = bs.clone();
    let mut n: u32 = 0;
    for b in 0u32..256 {
        if !keep[b as usize] {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut m = ['\0'; 256];
    for (i, b) in bs.iter().enumerate() {
        m[*b as usize] = char::from_u32(cs[i]).expect("valid scalar");
    }
    m
}

pub fn byte_to_char_table() -> &'static [char; 256] {
    static T: OnceLock<[char; 256]> = OnceLock::new();
    T.get_or_init(build_byte_to_char)
}

pub fn char_to_byte(c: char) -> Option<u8> {
    let table = byte_to_char_table();
    for (b, ch) in table.iter().enumerate() {
        if *ch == c {
            return Some(b as u8);
        }
    }
    None
}

pub fn bytes_to_bpe_chars(s: &str) -> String {
    let table = byte_to_char_table();
    let mut out = String::with_capacity(s.len());
    for b in s.as_bytes() {
        out.push(table[*b as usize]);
    }
    out
}

pub fn bpe_chars_to_bytes(s: &str) -> String {
    let mut out: Vec<u8> = Vec::with_capacity(s.len());
    for c in s.chars() {
        if let Some(b) = char_to_byte(c) {
            out.push(b);
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}
