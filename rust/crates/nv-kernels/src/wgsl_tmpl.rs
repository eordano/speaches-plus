#[macro_export]
macro_rules! wgsl {
    ($path:literal) => {{
        const SRC: &str = include_str!($path);
        const OUT: [u8; $crate::wgsl_tmpl::expanded_len(SRC)] = $crate::wgsl_tmpl::expand(SRC);
        const S: &str = match core::str::from_utf8(&OUT) {
            Ok(s) => s,
            Err(_) => panic!("wgsl!: expansion is not utf-8"),
        };
        S
    }};
}

const fn line_end(b: &[u8], mut i: usize, end: usize) -> usize {
    while i < end && b[i] != b'\n' {
        i += 1;
    }
    i
}

const fn has_prefix(b: &[u8], i: usize, j: usize, p: &[u8]) -> bool {
    if j - i < p.len() {
        return false;
    }
    let mut k = 0;
    while k < p.len() {
        if b[i + k] != p[k] {
            return false;
        }
        k += 1;
    }
    true
}

const fn directive(b: &[u8], s: usize, e: usize) -> (u8, usize, usize) {
    let mut i = s;
    while i < e && (b[i] == b' ' || b[i] == b'\t') {
        i += 1;
    }
    if i == e || b[i] != b'#' {
        return (0, 0, 0);
    }
    let mut j = e;
    while b[j - 1] == b' ' || b[j - 1] == b'\t' || b[j - 1] == b'\r' {
        j -= 1;
    }
    if j - i == 4 && has_prefix(b, i, j, b"#end") {
        (2, 0, 0)
    } else if has_prefix(b, i, j, b"#def ") {
        (1, i + 5, j)
    } else if has_prefix(b, i, j, b"#use ") {
        (3, i + 5, j)
    } else {
        panic!("wgsl!: a line starting with # must be #def NAME, #use NAME or #end")
    }
}

const MAX_DEFS: usize = 64;

const fn run(b: &[u8], out: &mut [u8], write: bool) -> usize {
    let mut name = [(0, 0); MAX_DEFS];
    let mut span = [(0, 0); MAX_DEFS];
    let mut open = [0; MAX_DEFS];
    let (mut n, mut depth, mut pos, mut i) = (0, 0, 0, 0);
    while i < b.len() {
        let le = line_end(b, i, b.len());
        let next = if le < b.len() { le + 1 } else { le };
        let (from, len) = match directive(b, i, le) {
            (0, _, _) => (usize::MAX, next - i),
            (1, s, e) => {
                assert!(n < MAX_DEFS, "wgsl!: more #def blocks than MAX_DEFS");
                name[n] = (s, e);
                span[n] = (pos, usize::MAX);
                open[depth] = n;
                depth += 1;
                n += 1;
                (0, 0)
            }
            (2, _, _) => {
                assert!(depth > 0, "wgsl!: #end without a #def");
                depth -= 1;
                span[open[depth]].1 = pos - span[open[depth]].0;
                (0, 0)
            }
            (_, s, e) => {
                let mut k = 0;
                while k < n && !same(b, name[k], (s, e)) {
                    k += 1;
                }
                assert!(
                    k < n && span[k].1 != usize::MAX,
                    "wgsl!: #use must follow its closed #def"
                );
                span[k]
            }
        };
        if write && len > 0 {
            let (done, rest) = out.split_at_mut(pos);
            let src = if from == usize::MAX {
                b.split_at(i).1
            } else {
                done.split_at(from).1
            };
            rest.split_at_mut(len)
                .0
                .copy_from_slice(src.split_at(len).0);
        }
        pos += len;
        i = next;
    }
    assert!(depth == 0, "wgsl!: #def without a matching #end");
    pos
}

const fn same(b: &[u8], (a, ae): (usize, usize), (c, ce): (usize, usize)) -> bool {
    if ae - a != ce - c {
        return false;
    }
    let mut k = 0;
    while k < ae - a && b[a + k] == b[c + k] {
        k += 1;
    }
    k == ae - a
}

pub const fn expanded_len(src: &str) -> usize {
    run(src.as_bytes(), &mut [], false)
}

pub const fn expand<const N: usize>(src: &str) -> [u8; N] {
    let mut out = [0u8; N];
    run(src.as_bytes(), &mut out, true);
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_def_stays_inline_and_every_use_repeats_it_nested_ones_included() {
        const T: &str = "a\n  #def x\n  b\n  #def y\n  c\n  #end\n  #end\n#use y\n  #use x\nd\n";
        const N: usize = super::expanded_len(T);
        assert_eq!(super::expand::<N>(T), *b"a\n  b\n  c\n  c\n  b\n  c\nd\n");
    }
}
