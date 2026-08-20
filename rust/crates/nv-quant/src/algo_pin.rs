use std::collections::HashMap;

pub fn parse_pin_spec(spec: &str) -> HashMap<(u64, u64, u64), usize> {
    let mut map = HashMap::new();
    for entry in spec.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let Some((shape, idx)) = entry.split_once('=') else {
            eprintln!("[algo-pin] ignoring malformed entry {entry:?} (expected MxNxK=IDX)");
            continue;
        };
        let dims: Vec<u64> = shape
            .trim()
            .split('x')
            .filter_map(|d| d.trim().parse().ok())
            .collect();
        let idx: Option<usize> = idx.trim().parse().ok();
        match (dims.as_slice(), idx) {
            (&[m, n, k], Some(i)) => {
                map.insert((m, n, k), i);
            }
            _ => eprintln!("[algo-pin] ignoring malformed entry {entry:?} (expected MxNxK=IDX)"),
        }
    }
    map
}

pub fn pin_map_from_env(var: &'static str) -> HashMap<(u64, u64, u64), usize> {
    std::env::var(var)
        .ok()
        .map(|s| parse_pin_spec(&s))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::parse_pin_spec;

    #[test]
    fn parses_multiple_entries_and_skips_malformed() {
        let m = parse_pin_spec("128x5376x8192=3; 128x5376x16384=1;bogus;4x5=2;;");
        assert_eq!(m.len(), 2);
        assert_eq!(m[&(128, 5376, 8192)], 3);
        assert_eq!(m[&(128, 5376, 16384)], 1);
    }

    #[test]
    fn empty_spec_is_empty() {
        assert!(parse_pin_spec("").is_empty());
        assert!(parse_pin_spec("  ").is_empty());
    }
}
