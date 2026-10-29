use anyhow::{bail, Result};

pub(crate) const MAX_INT_DIGITS: u32 = 20;
pub(crate) const ANY_FRAC: &str = r"(?:\.[0-9]{1,17})?";
const NONZERO_FRAC: &str = r"\.0{0,16}[1-9][0-9]{0,16}";
const ZERO_FRAC: &str = r"(?:\.0{1,17})?";

pub(crate) fn uint_max() -> u128 {
    10u128.pow(MAX_INT_DIGITS) - 1
}

fn same_len(a: &str, b: &str) -> String {
    if a == b {
        return a.to_string();
    }
    let (a0, ar) = (a.as_bytes()[0], &a[1..]);
    let (b0, br) = (b.as_bytes()[0], &b[1..]);
    if ar.is_empty() {
        return format!("[{}-{}]", a0 as char, b0 as char);
    }
    if a0 == b0 {
        return format!("{}{}", a0 as char, same_len(ar, br));
    }
    let k = ar.len();
    if ar.bytes().all(|c| c == b'0') && br.bytes().all(|c| c == b'9') {
        return format!("[{}-{}][0-9]{{{k}}}", a0 as char, b0 as char);
    }
    let mut parts = vec![format!("{}{}", a0 as char, same_len(ar, &"9".repeat(k)))];
    if b0 - a0 > 1 {
        parts.push(format!("[{}-{}][0-9]{{{k}}}", (a0 + 1) as char, (b0 - 1) as char));
    }
    parts.push(format!("{}{}", b0 as char, same_len(&"0".repeat(k), br)));
    format!("(?:{})", parts.join("|"))
}

pub(crate) fn uint_range(a: u128, b: u128) -> String {
    let b = b.min(uint_max());
    let mut alts = Vec::new();
    let mut lo = a;
    while lo <= b {
        let len = lo.to_string().len() as u32;
        let hi = b.min(10u128.pow(len) - 1);
        alts.push(same_len(&lo.to_string(), &hi.to_string()));
        lo = hi + 1;
    }
    format!("(?:{})", alts.join("|"))
}

pub(crate) fn int_range(lo: Option<i128>, hi: Option<i128>) -> Result<String> {
    let max = uint_max() as i128;
    let lo = lo.unwrap_or(-max).max(-max);
    let hi = hi.unwrap_or(max).min(max);
    if lo > hi {
        bail!("the integer bounds leave no value: minimum {lo} > maximum {hi}");
    }
    let mut parts = Vec::new();
    if lo < 0 {
        let mag_lo = if hi < 0 { -hi } else { 1 };
        parts.push(format!("-{}", uint_range(mag_lo as u128, (-lo) as u128)));
    }
    if hi >= 0 {
        parts.push(uint_range(lo.max(0) as u128, hi as u128));
    }
    Ok(format!("(?:{})", parts.join("|")))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Bound {
    pub(crate) at: i128,
    pub(crate) exclusive: bool,
}

fn magnitude(a: u128, a_excl: bool, b: Option<(u128, bool)>) -> Option<String> {
    let mut parts = Vec::new();
    if let Some((b, b_excl)) = b {
        if b < a {
            return None;
        }
        if b == a {
            return (!a_excl && !b_excl).then(|| format!("{a}{ZERO_FRAC}"));
        }
    }
    parts.push(if a_excl {
        format!("{a}{NONZERO_FRAC}")
    } else {
        format!("{a}{ANY_FRAC}")
    });
    let mid_hi = match b {
        Some((b, _)) => b.checked_sub(1),
        None => Some(uint_max()),
    };
    if let Some(mid_hi) = mid_hi.filter(|&h| h > a) {
        parts.push(format!("{}{ANY_FRAC}", uint_range(a + 1, mid_hi)));
    }
    if let Some((b, false)) = b {
        parts.push(format!("{b}{ZERO_FRAC}"));
    }
    Some(format!("(?:{})", parts.join("|")))
}

pub(crate) fn number_range(lo: Option<Bound>, hi: Option<Bound>) -> Result<String> {
    let max = uint_max() as i128;
    let mut parts = Vec::new();
    let nonneg_hi = hi.map(|h| (h.at, h.exclusive));
    let has_nonneg = match nonneg_hi {
        Some((u, excl)) => u > 0 || (u == 0 && !excl),
        None => true,
    };
    if has_nonneg {
        let (a, a_excl) = match lo {
            Some(l) if l.at >= 0 => (l.at.min(max) as u128, l.exclusive),
            _ => (0, false),
        };
        let b = nonneg_hi.map(|(u, e)| (u.min(max) as u128, e));
        if let Some(m) = magnitude(a, a_excl, b) {
            parts.push(m);
        }
    }
    let has_neg = lo.is_none_or(|l| l.at < 0);
    if has_neg {
        let (a, a_excl) = match hi {
            Some(h) if h.at < 0 => ((-h.at).min(max) as u128, h.exclusive),
            _ => (0, true),
        };
        let b = lo.map(|l| ((-l.at).min(max) as u128, l.exclusive));
        if let Some(m) = magnitude(a, a_excl, b) {
            parts.push(format!("-{m}"));
        }
    }
    if parts.is_empty() {
        bail!("the number bounds leave no value");
    }
    Ok(format!("(?:{})", parts.join("|")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches(re: &str, s: &str) -> bool {
        regex_automata::meta::Regex::new(&format!(r"\A(?:{re})\z"))
            .unwrap()
            .is_match(s)
    }

    #[test]
    fn integer_and_number_ranges_agree_with_arithmetic_on_a_window_and_a_decimal_grid() {
        let cases: &[(Option<i128>, Option<i128>)] = &[
            (Some(1), Some(5)),
            (Some(0), Some(100)),
            (Some(-90), Some(90)),
            (Some(7), Some(1234)),
            (Some(-1000), Some(-17)),
            (Some(0), None),
            (None, Some(-3)),
            (Some(99), Some(101)),
            (Some(10), Some(10)),
        ];
        for &(lo, hi) in cases {
            let re = int_range(lo, hi).unwrap();
            for v in -1500i128..=1500 {
                let want = lo.is_none_or(|l| v >= l) && hi.is_none_or(|h| v <= h);
                assert_eq!(matches(&re, &v.to_string()), want, "{lo:?}..={hi:?} at {v}: {re}");
            }
            assert!(!matches(&re, "01") && !matches(&re, "-0") && !matches(&re, "+1"));
        }
        assert!(int_range(Some(3), Some(2)).is_err());

        let b = |at: i128, exclusive: bool| Some(Bound { at, exclusive });
        let cases = [
            (b(0, false), b(1, false)),
            (b(0, true), b(1, true)),
            (b(-90, false), b(90, false)),
            (b(-5, true), b(-2, false)),
            (b(0, false), None),
            (None, b(0, true)),
            (b(3, false), b(3, false)),
            (b(-2, false), b(12, true)),
        ];
        for (lo, hi) in cases {
            let re = number_range(lo, hi).unwrap();
            for tenths in -1300i64..=1300 {
                let v = tenths as f64 / 10.0;
                let s = if tenths % 10 == 0 && tenths % 20 == 0 {
                    format!("{}", tenths / 10)
                } else {
                    format!("{v:.1}")
                };
                let x: f64 = s.parse().unwrap();
                let ok_lo = lo.is_none_or(|l| if l.exclusive { x > l.at as f64 } else { x >= l.at as f64 });
                let ok_hi = hi.is_none_or(|h| if h.exclusive { x < h.at as f64 } else { x <= h.at as f64 });
                assert_eq!(matches(&re, &s), ok_lo && ok_hi, "{lo:?} {hi:?} at {s}: {re}");
            }
        }
        assert!(number_range(b(1, true), b(1, false)).is_err());
    }

}
