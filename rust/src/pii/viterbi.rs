const NEG_INF: f32 = -1e30;

#[derive(Clone)]
struct Tag {
    prefix: u8,
    cls: String,
}

fn split_label(label: &str) -> Tag {
    if label == "O" {
        return Tag {
            prefix: b'O',
            cls: String::new(),
        };
    }
    let dash = label.find('-');
    match dash {
        None => Tag {
            prefix: b'?',
            cls: String::new(),
        },
        Some(d) => {
            let p = &label[..d];
            let prefix = match p {
                "B" => b'B',
                "I" => b'I',
                "E" => b'E',
                "S" => b'S',
                _ => b'?',
            };
            Tag {
                prefix,
                cls: label[d + 1..].to_string(),
            }
        }
    }
}

fn allowed(a: &Tag, b: &Tag) -> bool {
    match a.prefix {
        b'O' | b'E' | b'S' => matches!(b.prefix, b'O' | b'B' | b'S'),
        b'B' | b'I' => matches!(b.prefix, b'I' | b'E') && b.cls == a.cls,
        _ => false,
    }
}

fn build_start(tags: &[Tag]) -> Vec<f32> {
    tags.iter()
        .map(|t| {
            if t.prefix == b'I' || t.prefix == b'E' {
                NEG_INF
            } else {
                0.0
            }
        })
        .collect()
}

fn build_transitions(tags: &[Tag]) -> Vec<f32> {
    let n = tags.len();
    let mut out = vec![NEG_INF; n * n];
    for i in 0..n {
        for j in 0..n {
            if allowed(&tags[i], &tags[j]) {
                out[i * n + j] = 0.0;
            }
        }
    }
    out
}

fn log_softmax_row(row: &[f32], out: &mut [f32]) {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|&v| ((v - m) as f64).exp()).sum();
    let log_sum = (sum.ln() as f32) + m;
    for (i, &v) in row.iter().enumerate() {
        out[i] = v - log_sum;
    }
}

pub fn viterbi_decode(logits: &[f32], t: usize, l: usize, labels: &[String]) -> Vec<i32> {
    assert_eq!(l, labels.len());
    if t == 0 {
        return Vec::new();
    }

    let tags: Vec<Tag> = labels.iter().map(|s| split_label(s)).collect();
    let trans = build_transitions(&tags);
    let start = build_start(&tags);

    let mut lp = vec![0.0_f32; l];
    log_softmax_row(&logits[..l], &mut lp);

    let mut delta: Vec<f32> = start.iter().zip(lp.iter()).map(|(s, p)| s + p).collect();
    let mut bp = vec![0i32; t * l];

    for step in 1..t {
        log_softmax_row(&logits[step * l..(step + 1) * l], &mut lp);
        let mut new_delta = vec![NEG_INF; l];
        for j in 0..l {
            let mut best_val = NEG_INF;
            let mut best_i: i32 = 0;
            for i in 0..l {
                let score = delta[i] + trans[i * l + j];
                if score > best_val {
                    best_val = score;
                    best_i = i as i32;
                }
            }
            new_delta[j] = best_val + lp[j];
            bp[step * l + j] = best_i;
        }
        delta = new_delta;
    }

    let mut out = vec![0i32; t];
    let mut best_last = 0i32;
    let mut best_val = NEG_INF;
    for (j, &d) in delta.iter().enumerate().take(l) {
        if d > best_val {
            best_val = d;
            best_last = j as i32;
        }
    }
    out[t - 1] = best_last;
    for step in (1..t).rev() {
        out[step - 1] = bp[step * l + out[step] as usize];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_logits_returns_empty() {
        let labels = vec!["O".to_string(), "B-PER".to_string()];
        let result = viterbi_decode(&[], 0, 2, &labels);
        assert!(result.is_empty());
    }

    #[test]
    fn single_step_picks_best_valid() {
        let labels: Vec<String> = vec!["O", "B-PER", "I-PER", "E-PER", "S-PER"]
            .into_iter()
            .map(String::from)
            .collect();
        let mut logits = vec![0.0_f32; 5];
        logits[4] = 10.0;
        let result = viterbi_decode(&logits, 1, 5, &labels);
        assert_eq!(result, vec![4]);
    }

    #[test]
    fn ie_cannot_start() {
        let labels: Vec<String> = vec!["O", "B-PER", "I-PER", "E-PER", "S-PER"]
            .into_iter()
            .map(String::from)
            .collect();
        let mut logits = vec![0.0_f32; 5];
        logits[2] = 100.0;
        let result = viterbi_decode(&logits, 1, 5, &labels);
        assert_ne!(result[0], 2);
        assert_ne!(result[0], 3);
    }

    #[test]
    fn b_must_be_followed_by_i_or_e_of_same_class() {
        let labels: Vec<String> = vec!["O", "B-PER", "I-PER", "E-PER", "S-LOC"]
            .into_iter()
            .map(String::from)
            .collect();
        let mut logits = vec![0.0_f32; 10];
        logits[1] = 10.0;
        logits[5 + 3] = 10.0;
        let result = viterbi_decode(&logits, 2, 5, &labels);
        assert_eq!(result[0], 1);
        assert!(result[1] == 2 || result[1] == 3);
    }

    #[test]
    fn log_softmax_preserves_relative_order() {
        let row = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; 3];
        log_softmax_row(&row, &mut out);
        assert!(out[0] < out[1]);
        assert!(out[1] < out[2]);
        let sum: f64 = out.iter().map(|&v| (v as f64).exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
