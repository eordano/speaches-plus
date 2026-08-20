pub fn pack_u16_pairs(src: &[u16]) -> Vec<u32> {
    let mut out = vec![0u32; src.len() / 2];
    for (i, w) in out.iter_mut().enumerate() {
        *w = src[2 * i] as u32 | ((src[2 * i + 1] as u32) << 16);
    }
    out
}

pub fn unpack_u16_pairs(words: &[u32], dst: &mut [u16]) {
    for (i, w) in words.iter().enumerate() {
        dst[2 * i] = (*w & 0xffff) as u16;
        dst[2 * i + 1] = (*w >> 16) as u16;
    }
}

pub fn unpack_u16_pairs_clamped(words: &[u32], dst: &mut [u16]) {
    for (w, out) in words.iter().zip(dst.chunks_exact_mut(2)) {
        out[0] = (*w & 0xffff) as u16;
        out[1] = (*w >> 16) as u16;
    }
}

pub fn pack_u16_odd_tail_zeroed_min_one_word(src: &[u16]) -> Vec<u32> {
    let mut out = vec![0u32; src.len().div_ceil(2).max(1)];
    for (i, &v) in src.iter().enumerate() {
        out[i >> 1] |= (v as u32) << (16 * (i & 1));
    }
    out
}

pub fn unpack_u16_by_element(words: &[u32], dst: &mut [u16]) {
    for (i, slot) in dst.iter_mut().enumerate() {
        *slot = ((words[i >> 1] >> (16 * (i & 1))) & 0xffff) as u16;
    }
}

pub fn unpack_u16_first_n(words: &[u32], n: usize, out: &mut [u16]) {
    for (i, slot) in out.iter_mut().enumerate().take(n) {
        *slot = ((words[i / 2] >> (16 * (i % 2))) & 0xffff) as u16;
    }
}

pub fn pack_u16_even_min_one_word(src: &[u16]) -> Vec<u32> {
    let mut out: Vec<u32> = src
        .chunks_exact(2)
        .map(|c| (c[0] as u32) | ((c[1] as u32) << 16))
        .collect();
    if out.is_empty() {
        out.push(0);
    }
    out
}

pub fn pack_u8_words_padded_to_multiple(src: &[u8], word_multiple: usize) -> Vec<u32> {
    let words = src.len().div_ceil(4).max(1).next_multiple_of(word_multiple);
    let mut out = vec![0u32; words];
    for (i, &b) in src.iter().enumerate() {
        out[i >> 2] |= (b as u32) << (8 * (i & 3));
    }
    out
}

pub fn pack_u8_min_one_word(src: &[u8]) -> Vec<u32> {
    pack_u8_words_padded_to_multiple(src, 1)
}

pub fn unpack_u8_by_element(words: &[u32], dst: &mut [u8]) {
    for (i, b) in dst.iter_mut().enumerate() {
        *b = ((words[i >> 2] >> (8 * (i & 3))) & 0xff) as u8;
    }
}
