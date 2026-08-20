use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum F32Isa {
    Avx512,
    Avx2,
    Scalar,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum I8Isa {
    Avx512,
    Avx2,
    Scalar,
}

pub fn f32_isa() -> F32Isa {
    static ISA: OnceLock<F32Isa> = OnceLock::new();
    *ISA.get_or_init(|| {
        if std::env::var_os("NV_OCR_NO_SIMD").is_some() {
            return F32Isa::Scalar;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return F32Isa::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return F32Isa::Avx2;
            }
        }
        F32Isa::Scalar
    })
}

pub fn i8_isa() -> I8Isa {
    static ISA: OnceLock<I8Isa> = OnceLock::new();
    *ISA.get_or_init(|| {
        if std::env::var_os("NV_OCR_NO_SIMD").is_some() {
            return I8Isa::Scalar;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
            {
                return I8Isa::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return I8Isa::Avx2;
            }
        }
        I8Isa::Scalar
    })
}

pub fn matvec_f32(rows: usize, cols: usize, w: &[f32], u: &[f32], out: &mut [f32]) {
    match f32_isa() {
        #[cfg(target_arch = "x86_64")]
        F32Isa::Avx512 => unsafe { matvec_f32_avx512(rows, cols, w, u, out) },
        #[cfg(target_arch = "x86_64")]
        F32Isa::Avx2 => unsafe { matvec_f32_avx2(rows, cols, w, u, out) },
        _ => crate::lstm::matvec_f32_scalar(rows, cols, w, u, out),
    }
}

pub fn matvec_i8(rows: usize, cols: usize, w: &[i8], scales: &[f32], u: &[i8], out: &mut [f32]) {
    match i8_isa() {
        #[cfg(target_arch = "x86_64")]
        I8Isa::Avx512 => unsafe { matvec_i8_avx512(rows, cols, w, scales, u, out) },
        #[cfg(target_arch = "x86_64")]
        I8Isa::Avx2 => unsafe { matvec_i8_avx2(rows, cols, w, scales, u, out) },
        _ => crate::lstm::matvec_i8_scalar(rows, cols, w, scales, u, out),
    }
}

#[cfg(target_arch = "x86_64")]
pub fn matvec_f32_at(isa: F32Isa, rows: usize, cols: usize, w: &[f32], u: &[f32], out: &mut [f32]) {
    match isa {
        F32Isa::Avx512 => unsafe { matvec_f32_avx512(rows, cols, w, u, out) },
        F32Isa::Avx2 => unsafe { matvec_f32_avx2(rows, cols, w, u, out) },
        F32Isa::Scalar => crate::lstm::matvec_f32_scalar(rows, cols, w, u, out),
    }
}

#[cfg(target_arch = "x86_64")]
pub fn matvec_i8_at(
    isa: I8Isa,
    rows: usize,
    cols: usize,
    w: &[i8],
    scales: &[f32],
    u: &[i8],
    out: &mut [f32],
) {
    match isa {
        I8Isa::Avx512 => unsafe { matvec_i8_avx512(rows, cols, w, scales, u, out) },
        I8Isa::Avx2 => unsafe { matvec_i8_avx2(rows, cols, w, scales, u, out) },
        I8Isa::Scalar => crate::lstm::matvec_i8_scalar(rows, cols, w, scales, u, out),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn matvec_f32_avx512(rows: usize, cols: usize, w: &[f32], u: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    debug_assert!(w.len() >= rows * cols);
    debug_assert!(u.len() >= cols - 1);
    debug_assert!(out.len() >= rows);
    let n = cols - 1;
    let chunks = n / 16;
    let rem = n % 16;
    let tail: __mmask16 = if rem == 0 { 0 } else { (1u16 << rem) - 1 };
    let up = u.as_ptr();
    let wp = w.as_ptr();
    let mut r = 0usize;
    while r + 8 <= rows {
        let p0 = wp.add(r * cols);
        let p1 = p0.add(cols);
        let p2 = p1.add(cols);
        let p3 = p2.add(cols);
        let p4 = p3.add(cols);
        let p5 = p4.add(cols);
        let p6 = p5.add(cols);
        let p7 = p6.add(cols);
        let mut a0 = _mm512_setzero_ps();
        let mut a1 = _mm512_setzero_ps();
        let mut a2 = _mm512_setzero_ps();
        let mut a3 = _mm512_setzero_ps();
        let mut a4 = _mm512_setzero_ps();
        let mut a5 = _mm512_setzero_ps();
        let mut a6 = _mm512_setzero_ps();
        let mut a7 = _mm512_setzero_ps();
        for c in 0..chunks {
            let o = c * 16;
            let uv = _mm512_loadu_ps(up.add(o));
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(p0.add(o)), uv, a0);
            a1 = _mm512_fmadd_ps(_mm512_loadu_ps(p1.add(o)), uv, a1);
            a2 = _mm512_fmadd_ps(_mm512_loadu_ps(p2.add(o)), uv, a2);
            a3 = _mm512_fmadd_ps(_mm512_loadu_ps(p3.add(o)), uv, a3);
            a4 = _mm512_fmadd_ps(_mm512_loadu_ps(p4.add(o)), uv, a4);
            a5 = _mm512_fmadd_ps(_mm512_loadu_ps(p5.add(o)), uv, a5);
            a6 = _mm512_fmadd_ps(_mm512_loadu_ps(p6.add(o)), uv, a6);
            a7 = _mm512_fmadd_ps(_mm512_loadu_ps(p7.add(o)), uv, a7);
        }
        if rem != 0 {
            let o = chunks * 16;
            let uv = _mm512_maskz_loadu_ps(tail, up.add(o));
            a0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, p0.add(o)), uv, a0);
            a1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, p1.add(o)), uv, a1);
            a2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, p2.add(o)), uv, a2);
            a3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, p3.add(o)), uv, a3);
            a4 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, p4.add(o)), uv, a4);
            a5 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, p5.add(o)), uv, a5);
            a6 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, p6.add(o)), uv, a6);
            a7 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, p7.add(o)), uv, a7);
        }
        *out.get_unchecked_mut(r) = _mm512_reduce_add_ps(a0) + *p0.add(n);
        *out.get_unchecked_mut(r + 1) = _mm512_reduce_add_ps(a1) + *p1.add(n);
        *out.get_unchecked_mut(r + 2) = _mm512_reduce_add_ps(a2) + *p2.add(n);
        *out.get_unchecked_mut(r + 3) = _mm512_reduce_add_ps(a3) + *p3.add(n);
        *out.get_unchecked_mut(r + 4) = _mm512_reduce_add_ps(a4) + *p4.add(n);
        *out.get_unchecked_mut(r + 5) = _mm512_reduce_add_ps(a5) + *p5.add(n);
        *out.get_unchecked_mut(r + 6) = _mm512_reduce_add_ps(a6) + *p6.add(n);
        *out.get_unchecked_mut(r + 7) = _mm512_reduce_add_ps(a7) + *p7.add(n);
        r += 8;
    }
    while r < rows {
        let p0 = wp.add(r * cols);
        let mut a0 = _mm512_setzero_ps();
        for c in 0..chunks {
            let o = c * 16;
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(p0.add(o)), _mm512_loadu_ps(up.add(o)), a0);
        }
        if rem != 0 {
            let o = chunks * 16;
            a0 = _mm512_fmadd_ps(
                _mm512_maskz_loadu_ps(tail, p0.add(o)),
                _mm512_maskz_loadu_ps(tail, up.add(o)),
                a0,
            );
        }
        *out.get_unchecked_mut(r) = _mm512_reduce_add_ps(a0) + *p0.add(n);
        r += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_f32_avx2(rows: usize, cols: usize, w: &[f32], u: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    debug_assert!(w.len() >= rows * cols);
    debug_assert!(u.len() >= cols - 1);
    debug_assert!(out.len() >= rows);
    let n = cols - 1;
    let chunks = n / 8;
    let up = u.as_ptr();
    let wp = w.as_ptr();
    let mut r = 0usize;
    while r + 4 <= rows {
        let p0 = wp.add(r * cols);
        let p1 = p0.add(cols);
        let p2 = p1.add(cols);
        let p3 = p2.add(cols);
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        for c in 0..chunks {
            let o = c * 8;
            let uv = _mm256_loadu_ps(up.add(o));
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o)), uv, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(o)), uv, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2.add(o)), uv, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3.add(o)), uv, a3);
        }
        let mut s0 = hsum256(a0);
        let mut s1 = hsum256(a1);
        let mut s2 = hsum256(a2);
        let mut s3 = hsum256(a3);
        for j in chunks * 8..n {
            let uj = *up.add(j);
            s0 += *p0.add(j) * uj;
            s1 += *p1.add(j) * uj;
            s2 += *p2.add(j) * uj;
            s3 += *p3.add(j) * uj;
        }
        *out.get_unchecked_mut(r) = s0 + *p0.add(n);
        *out.get_unchecked_mut(r + 1) = s1 + *p1.add(n);
        *out.get_unchecked_mut(r + 2) = s2 + *p2.add(n);
        *out.get_unchecked_mut(r + 3) = s3 + *p3.add(n);
        r += 4;
    }
    while r < rows {
        let p0 = wp.add(r * cols);
        let mut a0 = _mm256_setzero_ps();
        for c in 0..chunks {
            let o = c * 8;
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o)), _mm256_loadu_ps(up.add(o)), a0);
        }
        let mut s0 = hsum256(a0);
        for j in chunks * 8..n {
            s0 += *p0.add(j) * *up.add(j);
        }
        *out.get_unchecked_mut(r) = s0 + *p0.add(n);
        r += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn matvec_i8_avx512(
    rows: usize,
    cols: usize,
    w: &[i8],
    scales: &[f32],
    u: &[i8],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    debug_assert!(w.len() >= rows * cols);
    debug_assert!(u.len() >= cols - 1);
    debug_assert!(out.len() >= rows);
    debug_assert!(scales.len() >= rows);
    let n = cols - 1;
    let chunks = n / 32;
    let rem = n % 32;
    let tail: __mmask32 = if rem == 0 { 0 } else { (1u32 << rem) - 1 };
    let up = u.as_ptr();
    let wp = w.as_ptr();
    let mut r = 0usize;
    while r + 4 <= rows {
        let p0 = wp.add(r * cols);
        let p1 = p0.add(cols);
        let p2 = p1.add(cols);
        let p3 = p2.add(cols);
        let mut a0 = _mm512_setzero_si512();
        let mut a1 = _mm512_setzero_si512();
        let mut a2 = _mm512_setzero_si512();
        let mut a3 = _mm512_setzero_si512();
        for c in 0..chunks {
            let o = c * 32;
            let uv = _mm512_cvtepi8_epi16(_mm256_loadu_si256(up.add(o) as *const _));
            let w0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(p0.add(o) as *const _));
            let w1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(p1.add(o) as *const _));
            let w2 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(p2.add(o) as *const _));
            let w3 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(p3.add(o) as *const _));
            a0 = _mm512_add_epi32(a0, _mm512_madd_epi16(uv, w0));
            a1 = _mm512_add_epi32(a1, _mm512_madd_epi16(uv, w1));
            a2 = _mm512_add_epi32(a2, _mm512_madd_epi16(uv, w2));
            a3 = _mm512_add_epi32(a3, _mm512_madd_epi16(uv, w3));
        }
        if rem != 0 {
            let o = chunks * 32;
            let uv = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(tail, up.add(o)));
            let w0 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(tail, p0.add(o)));
            let w1 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(tail, p1.add(o)));
            let w2 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(tail, p2.add(o)));
            let w3 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(tail, p3.add(o)));
            a0 = _mm512_add_epi32(a0, _mm512_madd_epi16(uv, w0));
            a1 = _mm512_add_epi32(a1, _mm512_madd_epi16(uv, w1));
            a2 = _mm512_add_epi32(a2, _mm512_madd_epi16(uv, w2));
            a3 = _mm512_add_epi32(a3, _mm512_madd_epi16(uv, w3));
        }
        let t0 = _mm512_reduce_add_epi32(a0) + (*p0.add(n) as i32) * 127;
        let t1 = _mm512_reduce_add_epi32(a1) + (*p1.add(n) as i32) * 127;
        let t2 = _mm512_reduce_add_epi32(a2) + (*p2.add(n) as i32) * 127;
        let t3 = _mm512_reduce_add_epi32(a3) + (*p3.add(n) as i32) * 127;
        *out.get_unchecked_mut(r) = t0 as f32 * *scales.get_unchecked(r);
        *out.get_unchecked_mut(r + 1) = t1 as f32 * *scales.get_unchecked(r + 1);
        *out.get_unchecked_mut(r + 2) = t2 as f32 * *scales.get_unchecked(r + 2);
        *out.get_unchecked_mut(r + 3) = t3 as f32 * *scales.get_unchecked(r + 3);
        r += 4;
    }
    while r < rows {
        let p0 = wp.add(r * cols);
        let mut a0 = _mm512_setzero_si512();
        for c in 0..chunks {
            let o = c * 32;
            let uv = _mm512_cvtepi8_epi16(_mm256_loadu_si256(up.add(o) as *const _));
            let w0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(p0.add(o) as *const _));
            a0 = _mm512_add_epi32(a0, _mm512_madd_epi16(uv, w0));
        }
        if rem != 0 {
            let o = chunks * 32;
            let uv = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(tail, up.add(o)));
            let w0 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(tail, p0.add(o)));
            a0 = _mm512_add_epi32(a0, _mm512_madd_epi16(uv, w0));
        }
        let t0 = _mm512_reduce_add_epi32(a0) + (*p0.add(n) as i32) * 127;
        *out.get_unchecked_mut(r) = t0 as f32 * *scales.get_unchecked(r);
        r += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matvec_i8_avx2(
    rows: usize,
    cols: usize,
    w: &[i8],
    scales: &[f32],
    u: &[i8],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    debug_assert!(w.len() >= rows * cols);
    debug_assert!(u.len() >= cols - 1);
    debug_assert!(out.len() >= rows);
    debug_assert!(scales.len() >= rows);
    let n = cols - 1;
    let chunks = n / 16;
    let up = u.as_ptr();
    let wp = w.as_ptr();
    let mut r = 0usize;
    while r + 2 <= rows {
        let p0 = wp.add(r * cols);
        let p1 = p0.add(cols);
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        for c in 0..chunks {
            let o = c * 16;
            let uv = _mm256_cvtepi8_epi16(_mm_loadu_si128(up.add(o) as *const _));
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(p0.add(o) as *const _));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(p1.add(o) as *const _));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(uv, w0));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(uv, w1));
        }
        let mut t0 = hsum256_epi32(a0);
        let mut t1 = hsum256_epi32(a1);
        for j in chunks * 16..n {
            let uj = *up.add(j) as i32;
            t0 += *p0.add(j) as i32 * uj;
            t1 += *p1.add(j) as i32 * uj;
        }
        t0 += (*p0.add(n) as i32) * 127;
        t1 += (*p1.add(n) as i32) * 127;
        *out.get_unchecked_mut(r) = t0 as f32 * *scales.get_unchecked(r);
        *out.get_unchecked_mut(r + 1) = t1 as f32 * *scales.get_unchecked(r + 1);
        r += 2;
    }
    while r < rows {
        let p0 = wp.add(r * cols);
        let mut a0 = _mm256_setzero_si256();
        for c in 0..chunks {
            let o = c * 16;
            let uv = _mm256_cvtepi8_epi16(_mm_loadu_si128(up.add(o) as *const _));
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(p0.add(o) as *const _));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(uv, w0));
        }
        let mut t0 = hsum256_epi32(a0);
        for j in chunks * 16..n {
            t0 += *p0.add(j) as i32 * *up.add(j) as i32;
        }
        t0 += (*p0.add(n) as i32) * 127;
        *out.get_unchecked_mut(r) = t0 as f32 * *scales.get_unchecked(r);
        r += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_epi32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b0100_1110));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b1011_0001));
    _mm_cvtsi128_si32(s)
}
