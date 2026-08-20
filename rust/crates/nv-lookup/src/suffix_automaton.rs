#[derive(Clone, Debug)]
struct SamState {
    len: u32,
    link: i32,
    firstpos: u32,
    next: Vec<(u32, u32)>,
}

impl SamState {
    fn get(&self, tok: u32) -> Option<u32> {
        match self.next.binary_search_by_key(&tok, |&(t, _)| t) {
            Ok(i) => Some(self.next[i].1),
            Err(_) => None,
        }
    }

    fn set(&mut self, tok: u32, dst: u32) {
        match self.next.binary_search_by_key(&tok, |&(t, _)| t) {
            Ok(i) => self.next[i].1 = dst,
            Err(i) => self.next.insert(i, (tok, dst)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SuffixProposal {
    pub match_len: usize,
    pub tokens: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct SuffixAutomaton {
    states: Vec<SamState>,
    last: usize,
    tokens: Vec<u32>,
}

impl Default for SuffixAutomaton {
    fn default() -> Self {
        Self::new()
    }
}

impl SuffixAutomaton {
    pub fn new() -> Self {
        Self {
            states: vec![SamState {
                len: 0,
                link: -1,
                firstpos: u32::MAX,
                next: Vec::new(),
            }],
            last: 0,
            tokens: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn extend(&mut self, tok: u32) {
        let pos = self.tokens.len() as u32;
        self.tokens.push(tok);

        let cur = self.states.len();
        self.states.push(SamState {
            len: self.states[self.last].len + 1,
            link: -1,
            firstpos: pos,
            next: Vec::new(),
        });

        let mut p = self.last as i32;
        while p >= 0 && self.states[p as usize].get(tok).is_none() {
            self.states[p as usize].set(tok, cur as u32);
            p = self.states[p as usize].link;
        }

        if p < 0 {
            self.states[cur].link = 0;
        } else {
            let q = self.states[p as usize].get(tok).unwrap() as usize;
            if self.states[p as usize].len + 1 == self.states[q].len {
                self.states[cur].link = q as i32;
            } else {
                let clone = self.states.len();
                let cloned = SamState {
                    len: self.states[p as usize].len + 1,
                    link: self.states[q].link,
                    firstpos: self.states[q].firstpos,
                    next: self.states[q].next.clone(),
                };
                self.states.push(cloned);
                while p >= 0 && self.states[p as usize].get(tok) == Some(q as u32) {
                    self.states[p as usize].set(tok, clone as u32);
                    p = self.states[p as usize].link;
                }
                self.states[q].link = clone as i32;
                self.states[cur].link = clone as i32;
            }
        }
        self.last = cur;
    }

    pub fn extend_slice(&mut self, toks: &[u32]) {
        for &t in toks {
            self.extend(t);
        }
    }

    pub fn propose(&self, max_len: usize, min_match: usize) -> Option<SuffixProposal> {
        if max_len == 0 || self.tokens.len() < 2 {
            return None;
        }
        let n = self.tokens.len();
        let link = self.states[self.last].link;
        if link <= 0 {
            return None;
        }
        let u = link as usize;
        let match_len = self.states[u].len as usize;
        if match_len < min_match.max(1) {
            return None;
        }
        let src_end = self.states[u].firstpos as usize;
        let start = src_end + 1;
        if start >= n {
            return None;
        }
        let take = (n - start).min(max_len);
        Some(SuffixProposal {
            match_len,
            tokens: self.tokens[start..start + take].to_vec(),
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AcceptEma {
    value: f64,
    alpha: f64,
}

impl AcceptEma {
    pub fn new(alpha: f64, init: f64) -> Self {
        Self {
            value: init.max(0.0),
            alpha: alpha.clamp(0.01, 1.0),
        }
    }

    pub fn observe(&mut self, accepted: usize) {
        self.value = (1.0 - self.alpha) * self.value + self.alpha * accepted as f64;
    }

    pub fn value(&self) -> f64 {
        self.value
    }
}

pub fn suffix_arm_wins(
    proposal_len: usize,
    min_match: usize,
    match_len: usize,
    drafter_ema: f64,
) -> bool {
    match_len >= min_match && proposal_len as f64 >= drafter_ema
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_propose(tokens: &[u32], max_len: usize, min_match: usize) -> Option<SuffixProposal> {
        let n = tokens.len();
        if n < 2 || max_len == 0 {
            return None;
        }
        for l in (1..n).rev() {
            let suffix = &tokens[n - l..];
            let mut first_end: Option<usize> = None;
            for end in l - 1..n - 1 {
                if &tokens[end + 1 - l..=end] == suffix {
                    first_end = Some(end);
                    break;
                }
            }
            if let Some(end) = first_end {
                if l < min_match.max(1) {
                    return None;
                }
                let start = end + 1;
                let take = (n - start).min(max_len);
                return Some(SuffixProposal {
                    match_len: l,
                    tokens: tokens[start..start + take].to_vec(),
                });
            }
        }
        None
    }

    #[test]
    fn empty_and_single_token_propose_nothing() {
        let mut sam = SuffixAutomaton::new();
        assert!(sam.propose(8, 1).is_none());
        sam.extend(5);
        assert!(sam.propose(8, 1).is_none());
    }

    #[test]
    fn no_repeat_proposes_nothing() {
        let mut sam = SuffixAutomaton::new();
        sam.extend_slice(&[1, 2, 3, 4, 5]);
        assert!(sam.propose(8, 1).is_none());
    }

    #[test]
    fn simple_repeat_continues_from_earlier_occurrence() {
        let mut sam = SuffixAutomaton::new();
        sam.extend_slice(&[10, 20, 30, 40, 10, 20]);
        let p = sam.propose(8, 1).unwrap();
        assert_eq!(p.match_len, 2);
        assert_eq!(p.tokens, vec![30, 40, 10, 20]);
    }

    #[test]
    fn max_len_caps_the_continuation() {
        let mut sam = SuffixAutomaton::new();
        sam.extend_slice(&[10, 20, 30, 40, 10, 20]);
        let p = sam.propose(2, 1).unwrap();
        assert_eq!(p.tokens, vec![30, 40]);
    }

    #[test]
    fn min_match_gates_short_matches() {
        let mut sam = SuffixAutomaton::new();
        sam.extend_slice(&[10, 20, 30, 40, 10, 20]);
        assert!(sam.propose(8, 3).is_none());
        assert!(sam.propose(8, 2).is_some());
    }

    #[test]
    fn period_one_run_proposes_more_of_the_run() {
        let mut sam = SuffixAutomaton::new();
        sam.extend_slice(&[7, 7, 7, 7]);
        let p = sam.propose(8, 1).unwrap();
        assert_eq!(p.match_len, 3);
        assert_eq!(p.tokens, vec![7]);
    }

    #[test]
    fn matches_longest_repeated_suffix_not_shorter_one() {
        let mut sam = SuffixAutomaton::new();
        sam.extend_slice(&[1, 2, 3, 9, 1, 2, 3, 5, 2, 3]);
        let p = sam.propose(8, 1).unwrap();
        assert_eq!(p.match_len, 2);
        assert_eq!(p.tokens, vec![9, 1, 2, 3, 5, 2, 3]);
    }

    #[test]
    fn agentic_tool_echo_pattern_yields_long_continuation() {
        let mut sam = SuffixAutomaton::new();
        let call: Vec<u32> = vec![100, 101, 102, 103, 104, 105, 106, 107];
        let mut stream: Vec<u32> = Vec::new();
        stream.extend_from_slice(&call);
        stream.extend_from_slice(&[1, 2, 3]);
        stream.extend_from_slice(&call[..4]);
        sam.extend_slice(&stream);
        let p = sam.propose(4, 2).unwrap();
        assert_eq!(p.match_len, 4);
        assert_eq!(p.tokens, &call[4..8]);
    }

    #[test]
    fn matches_naive_reference_on_random_streams() {
        let mut rng: u64 = 0x243f6a8885a308d3;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for trial in 0..300 {
            let alpha = 2 + (next() % 5) as u32;
            let len = 2 + (next() % 40) as usize;
            let tokens: Vec<u32> = (0..len).map(|_| (next() % alpha as u64) as u32).collect();
            let mut sam = SuffixAutomaton::new();
            sam.extend_slice(&tokens);
            for &(max_len, min_match) in &[(1usize, 1usize), (4, 1), (8, 2), (16, 4)] {
                let got = sam.propose(max_len, min_match);
                let want = naive_propose(&tokens, max_len, min_match);
                match (&got, &want) {
                    (None, None) => {}
                    (Some(g), Some(w)) => {
                        assert_eq!(
                            g.match_len, w.match_len,
                            "trial {trial} tokens {tokens:?} max {max_len} min {min_match}"
                        );
                        assert_eq!(
                            g.tokens, w.tokens,
                            "trial {trial} tokens {tokens:?} max {max_len} min {min_match}: a \
                             continuation of the right length taken from the wrong occurrence \
                             is still a wrong draft"
                        );
                    }
                    _ => panic!(
                        "trial {trial} mismatch: got {got:?} want {want:?} tokens {tokens:?} max {max_len} min {min_match}"
                    ),
                }
            }
        }
    }

    #[test]
    fn proposal_source_is_a_true_earlier_occurrence() {
        let mut rng: u64 = 0x13198a2e03707344;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..200 {
            let len = 3 + (next() % 60) as usize;
            let tokens: Vec<u32> = (0..len).map(|_| (next() % 4) as u32).collect();
            let mut sam = SuffixAutomaton::new();
            sam.extend_slice(&tokens);
            if let Some(p) = sam.propose(64, 1) {
                let n = tokens.len();
                let m = p.match_len;
                assert!(m < n);
                let suffix = &tokens[n - m..];
                let found = (m - 1..n - 1).any(|end| &tokens[end + 1 - m..=end] == suffix);
                assert!(
                    found,
                    "match_len {m} suffix {suffix:?} not found earlier in {tokens:?}"
                );
                assert!(!p.tokens.is_empty());
            }
        }
    }

    #[test]
    fn every_prefix_of_a_stream_proposes_what_brute_force_says() {
        let tokens: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 1, 4, 1, 5];
        let mut inc = SuffixAutomaton::new();
        let mut seen = 0usize;
        for (i, &t) in tokens.iter().enumerate() {
            inc.extend(t);
            let prefix = &tokens[..=i];
            let want = naive_propose(prefix, 8, 1);
            assert_eq!(inc.propose(8, 1), want, "after token {i} of {tokens:?}");
            seen += usize::from(want.is_some());
        }
        assert!(
            seen >= 5,
            "only {seen} prefixes had a proposal at all; a fixture that never proposes gates \
             nothing"
        );
        let p = inc.propose(8, 2).unwrap();
        assert_eq!(p.match_len, 5);
        assert_eq!(p.tokens, vec![9, 2, 6, 5, 3, 5, 8, 9]);
    }

    #[test]
    fn long_stream_stays_fast_and_correct() {
        let mut sam = SuffixAutomaton::new();
        let block: Vec<u32> = (0..64).map(|i| (i * 13 + 5) % 97).collect();
        for _ in 0..64 {
            sam.extend_slice(&block);
        }
        assert_eq!(sam.len(), 64 * 64);
        let p = sam.propose(16, 8).unwrap();
        assert!(p.match_len >= 8);
        assert_eq!(p.tokens.len(), 16);
        let n = sam.len();
        let m = p.match_len;
        let toks = sam.tokens();
        let suffix: Vec<u32> = toks[n - m..].to_vec();
        let mut expected: Vec<u32> = Vec::new();
        'outer: for end in m - 1..n - 1 {
            if toks[end + 1 - m..=end] == suffix[..] {
                expected = toks[end + 1..(end + 17).min(n)].to_vec();
                break 'outer;
            }
        }
        assert!(!expected.is_empty());
        assert_eq!(
            p.tokens, expected,
            "the proposal must be the continuation of the earliest earlier occurrence of the \
             {m}-token suffix"
        );
    }

    #[test]
    fn ema_moves_toward_observations() {
        let mut e = AcceptEma::new(0.5, 2.0);
        assert!((e.value() - 2.0).abs() < 1e-12);
        e.observe(4);
        assert!((e.value() - 3.0).abs() < 1e-12);
        e.observe(0);
        assert!((e.value() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn suffix_arm_decision_thresholds() {
        assert!(suffix_arm_wins(3, 2, 5, 2.5));
        assert!(!suffix_arm_wins(2, 2, 5, 2.5));
        assert!(!suffix_arm_wins(8, 4, 3, 1.0));
        assert!(suffix_arm_wins(1, 1, 1, 1.0));
    }
}
