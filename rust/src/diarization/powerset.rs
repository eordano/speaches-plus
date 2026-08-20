use super::segmentation::SegmentationLogits;

#[derive(Clone, Debug)]
pub struct Multilabel {
    pub frames: usize,
    pub speakers: usize,
    pub data: Vec<u8>,
}

impl Multilabel {
    #[inline]
    pub fn row(&self, frame: usize) -> &[u8] {
        let start = frame * self.speakers;
        &self.data[start..start + self.speakers]
    }
}

#[derive(Clone, Debug)]
pub struct PowersetDecoder {
    pub max_speakers_per_chunk: usize,
    pub max_speakers_per_frame: usize,

    mapping: Vec<Vec<usize>>,
}

impl PowersetDecoder {
    pub fn new(max_speakers_per_chunk: usize, max_speakers_per_frame: usize) -> Self {
        let mapping = build_mapping(max_speakers_per_chunk, max_speakers_per_frame);
        Self {
            max_speakers_per_chunk,
            max_speakers_per_frame,
            mapping,
        }
    }

    #[inline]
    pub fn num_classes(&self) -> usize {
        self.mapping.len()
    }

    pub fn to_multilabel_hard(&self, logits: &SegmentationLogits) -> Multilabel {
        debug_assert_eq!(
            logits.classes,
            self.num_classes(),
            "logits.classes ({}) != decoder.num_classes ({}) -- topology mismatch",
            logits.classes,
            self.num_classes(),
        );
        let speakers = self.max_speakers_per_chunk;
        let mut data = vec![0u8; logits.frames * speakers];
        for f in 0..logits.frames {
            let row = logits.row(f);
            let cls = argmax(row);
            for &spk in &self.mapping[cls] {
                data[f * speakers + spk] = 1;
            }
        }
        Multilabel {
            frames: logits.frames,
            speakers,
            data,
        }
    }
}

#[inline]
fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

fn build_mapping(num_classes: usize, max_set_size: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for size in 0..=max_set_size {
        for combo in combinations(num_classes, size) {
            out.push(combo);
        }
    }
    out
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut buf = Vec::with_capacity(k);
    pick(0, n, k, &mut buf, &mut result);
    result
}

fn pick(start: usize, n: usize, k: usize, buf: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    if buf.len() == k {
        out.push(buf.clone());
        return;
    }
    for i in start..n {
        buf.push(i);
        pick(i + 1, n, k, buf, out);
        buf.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diarizen_v2_topology() {
        let dec = PowersetDecoder::new(4, 2);

        assert_eq!(dec.num_classes(), 11);
        assert_eq!(dec.mapping[0], Vec::<usize>::new());
        assert_eq!(dec.mapping[1], vec![0]);
        assert_eq!(dec.mapping[4], vec![3]);
        assert_eq!(dec.mapping[5], vec![0, 1]);
        assert_eq!(dec.mapping[10], vec![2, 3]);
    }

    #[test]
    fn pyannote_3spk_topology() {
        let dec = PowersetDecoder::new(3, 2);
        assert_eq!(dec.num_classes(), 7);
    }

    #[test]
    fn class_indices_are_unique_and_sorted() {
        use std::collections::BTreeSet;
        let dec = PowersetDecoder::new(4, 2);
        let set: BTreeSet<Vec<usize>> = dec.mapping.iter().cloned().collect();
        assert_eq!(set.len(), dec.num_classes(), "duplicate class");
        for combo in &dec.mapping {
            let mut sorted = combo.clone();
            sorted.sort();
            assert_eq!(*combo, sorted, "class indices must be sorted: {:?}", combo);
        }
    }

    #[test]
    fn argmax_picks_silence() {
        let dec = PowersetDecoder::new(4, 2);

        let logits = SegmentationLogits {
            frames: 1,
            classes: 11,
            data: {
                let mut v = vec![-10.0f32; 11];
                v[0] = 0.0;
                v
            },
        };
        let ml = dec.to_multilabel_hard(&logits);
        assert_eq!(ml.row(0), &[0, 0, 0, 0]);
    }

    #[test]
    fn argmax_picks_overlap() {
        let dec = PowersetDecoder::new(4, 2);
        let logits = SegmentationLogits {
            frames: 1,
            classes: 11,
            data: {
                let mut v = vec![-10.0f32; 11];
                v[5] = 0.0;
                v
            },
        };
        let ml = dec.to_multilabel_hard(&logits);
        assert_eq!(ml.row(0), &[1, 1, 0, 0]);
    }
}
