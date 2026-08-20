#[derive(Debug, Clone, serde::Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct PiiSpan {
    pub start: usize,
    #[serde(rename = "endExclusive")]
    pub end_exclusive: usize,
    pub label: String,
}

pub fn assemble_spans(
    labels: &[String],
    offsets: &[(usize, usize)],
    attention_mask: &[i32],
) -> Vec<PiiSpan> {
    let mut out: Vec<PiiSpan> = Vec::new();
    let mut open_label: Option<String> = None;
    let mut open_start: usize = 0;
    let mut open_end: usize = 0;

    let close = |out: &mut Vec<PiiSpan>,
                 open_label: &mut Option<String>,
                 open_start: &mut usize,
                 open_end: &mut usize| {
        if let Some(lbl) = open_label.take() {
            if *open_start < *open_end {
                out.push(PiiSpan {
                    start: *open_start,
                    end_exclusive: *open_end,
                    label: lbl,
                });
            }
        }
        *open_start = 0;
        *open_end = 0;
    };

    for (i, tag) in labels.iter().enumerate() {
        if i >= attention_mask.len() || attention_mask[i] == 0 {
            continue;
        }
        let (s, e) = offsets[i];
        if e <= s {
            continue;
        }
        if tag == "O" {
            close(&mut out, &mut open_label, &mut open_start, &mut open_end);
            continue;
        }
        let dash = tag.find('-');
        let Some(d) = dash else {
            close(&mut out, &mut open_label, &mut open_start, &mut open_end);
            continue;
        };
        let prefix = &tag[..d];
        let cls = &tag[d + 1..];

        match prefix {
            "B" => {
                close(&mut out, &mut open_label, &mut open_start, &mut open_end);
                open_label = Some(cls.to_string());
                open_start = s;
                open_end = e;
            }
            "I" => {
                if open_label.as_deref() == Some(cls) {
                    open_end = e;
                } else {
                    close(&mut out, &mut open_label, &mut open_start, &mut open_end);
                    open_label = Some(cls.to_string());
                    open_start = s;
                    open_end = e;
                }
            }
            "E" => {
                if open_label.as_deref() == Some(cls) {
                    open_end = e;
                    close(&mut out, &mut open_label, &mut open_start, &mut open_end);
                } else {
                    close(&mut out, &mut open_label, &mut open_start, &mut open_end);
                    out.push(PiiSpan {
                        start: s,
                        end_exclusive: e,
                        label: cls.to_string(),
                    });
                }
            }
            "S" => {
                close(&mut out, &mut open_label, &mut open_start, &mut open_end);
                out.push(PiiSpan {
                    start: s,
                    end_exclusive: e,
                    label: cls.to_string(),
                });
            }
            _ => {
                close(&mut out, &mut open_label, &mut open_start, &mut open_end);
            }
        }
    }

    close(&mut out, &mut open_label, &mut open_start, &mut open_end);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        let result = assemble_spans(&[], &[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn all_o_labels() {
        let labels: Vec<String> = vec!["O", "O", "O"].into_iter().map(String::from).collect();
        let offsets = vec![(0, 3), (3, 6), (6, 9)];
        let mask = vec![1, 1, 1];
        let result = assemble_spans(&labels, &offsets, &mask);
        assert!(result.is_empty());
    }

    #[test]
    fn single_s_label() {
        let labels: Vec<String> = vec!["O", "S-email", "O"]
            .into_iter()
            .map(String::from)
            .collect();
        let offsets = vec![(0, 3), (4, 20), (21, 25)];
        let mask = vec![1, 1, 1];
        let result = assemble_spans(&labels, &offsets, &mask);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start, 4);
        assert_eq!(result[0].end_exclusive, 20);
        assert_eq!(result[0].label, "email");
    }

    #[test]
    fn bie_sequence() {
        let labels: Vec<String> = vec!["B-name", "I-name", "E-name"]
            .into_iter()
            .map(String::from)
            .collect();
        let offsets = vec![(0, 4), (5, 9), (10, 15)];
        let mask = vec![1, 1, 1];
        let result = assemble_spans(&labels, &offsets, &mask);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start, 0);
        assert_eq!(result[0].end_exclusive, 15);
        assert_eq!(result[0].label, "name");
    }

    #[test]
    fn attention_mask_skips_zeros() {
        let labels: Vec<String> = vec!["S-email", "S-phone"]
            .into_iter()
            .map(String::from)
            .collect();
        let offsets = vec![(0, 5), (6, 10)];
        let mask = vec![1, 0];
        let result = assemble_spans(&labels, &offsets, &mask);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, "email");
    }

    #[test]
    fn zero_width_offsets_skipped() {
        let labels: Vec<String> = vec!["S-email"].into_iter().map(String::from).collect();
        let offsets = vec![(5, 5)];
        let mask = vec![1];
        let result = assemble_spans(&labels, &offsets, &mask);
        assert!(result.is_empty());
    }

    #[test]
    fn mismatched_i_starts_new_span() {
        let labels: Vec<String> = vec!["B-name", "I-phone"]
            .into_iter()
            .map(String::from)
            .collect();
        let offsets = vec![(0, 4), (5, 10)];
        let mask = vec![1, 1];
        let result = assemble_spans(&labels, &offsets, &mask);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].label, "name");
        assert_eq!(result[1].label, "phone");
    }

    #[test]
    fn mismatched_e_emits_standalone() {
        let labels: Vec<String> = vec!["B-name", "E-phone"]
            .into_iter()
            .map(String::from)
            .collect();
        let offsets = vec![(0, 4), (5, 10)];
        let mask = vec![1, 1];
        let result = assemble_spans(&labels, &offsets, &mask);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].label, "name");
        assert_eq!(result[0].start, 0);
        assert_eq!(result[0].end_exclusive, 4);
        assert_eq!(result[1].label, "phone");
        assert_eq!(result[1].start, 5);
        assert_eq!(result[1].end_exclusive, 10);
    }
}
