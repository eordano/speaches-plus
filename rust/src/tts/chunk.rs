use std::sync::OnceLock;

use nv_punkt::Segmenter;

pub const INTRA_SENTENCE_SPLIT: usize = usize::MAX;

pub struct ChunkPlan {
    pub chunks: Vec<String>,
    pub boundaries: Vec<usize>,
    pub oversize_splits: usize,
}

fn segmenter() -> &'static Segmenter {
    static S: OnceLock<Segmenter> = OnceLock::new();
    S.get_or_init(Segmenter::english)
}

pub fn plan(text: &str, max_chars: usize) -> ChunkPlan {
    let max_chars = max_chars.max(1);
    let mut chunks: Vec<String> = Vec::new();
    let mut boundaries: Vec<usize> = Vec::new();
    let mut oversize_splits = 0usize;
    let mut cur = String::new();
    let mut cur_chars = 0usize;
    let mut cur_end = 0usize;

    let mut flush = |cur: &mut String, cur_chars: &mut usize, end: usize| {
        let trimmed = cur.trim();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
            boundaries.push(end);
        }
        cur.clear();
        *cur_chars = 0;
    };

    for range in segmenter().sentences(text) {
        let sentence = text[range.clone()].trim();
        if sentence.is_empty() {
            continue;
        }
        let sent_chars = sentence.chars().count();
        if sent_chars > max_chars {
            flush(&mut cur, &mut cur_chars, cur_end);
            oversize_splits += 1;
            for word in sentence.split_whitespace() {
                let word_chars = word.chars().count();
                let projected = cur_chars + word_chars + usize::from(!cur.is_empty());
                if projected > max_chars && !cur.is_empty() {
                    flush(&mut cur, &mut cur_chars, INTRA_SENTENCE_SPLIT);
                }
                if !cur.is_empty() {
                    cur.push(' ');
                    cur_chars += 1;
                }
                cur.push_str(word);
                cur_chars += word_chars;
            }
            cur_end = range.end;
        } else {
            let projected = cur_chars + sent_chars + usize::from(!cur.is_empty());
            if projected > max_chars && !cur.is_empty() {
                flush(&mut cur, &mut cur_chars, cur_end);
            }
            if !cur.is_empty() {
                cur.push(' ');
                cur_chars += 1;
            }
            cur.push_str(sentence);
            cur_chars += sent_chars;
            cur_end = range.end;
        }
    }
    flush(&mut cur, &mut cur_chars, cur_end);

    ChunkPlan {
        chunks,
        boundaries,
        oversize_splits,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_single_chunk() {
        let p = plan("Hello there.", 400);
        assert_eq!(p.chunks, vec!["Hello there.".to_string()]);
        assert_eq!(p.boundaries, vec![12]);
        assert_eq!(p.oversize_splits, 0);
    }

    #[test]
    fn boundaries_are_sentence_ends() {
        let text = "Dr. Smith went to Washington. He arrived at noon. The meeting ran long. \
                    Everyone left at five. It was a busy day for the whole team.";
        let p = plan(text, 60);
        assert!(p.chunks.len() > 1);
        assert_eq!(p.oversize_splits, 0);
        for &b in &p.boundaries {
            assert_ne!(b, INTRA_SENTENCE_SPLIT);
            assert_eq!(&text[b - 1..b], ".", "boundary at sentence-final period");
        }
        assert!(!p.chunks[0].ends_with("Dr."));
    }

    #[test]
    fn oversize_sentence_word_split() {
        let text = "word ".repeat(200);
        let p = plan(text.trim(), 50);
        assert!(p.chunks.len() > 1);
        assert!(p.oversize_splits >= 1);
        for c in &p.chunks {
            assert!(c.chars().count() <= 50);
        }
    }
}
