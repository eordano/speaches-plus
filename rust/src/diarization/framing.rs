use super::powerset::Multilabel;
use super::DiarSegment;

#[derive(Clone, Debug)]
pub struct Chunk {
    pub samples: Vec<f32>,

    pub t_offset_ms: u64,
}

#[derive(Clone, Debug)]
pub struct Span {
    pub sample_start: usize,

    pub sample_end: usize,
    pub t_start_ms: u64,
    pub t_end_ms: u64,
    pub local_speaker: usize,
    pub overlap: bool,
}

#[derive(Clone, Debug)]
pub struct ChunkSpans {
    pub chunk_index: usize,
    pub spans: Vec<Span>,
}

pub fn slide_chunks(
    audio: &[f32],
    sample_rate: u32,
    chunk_seconds: f32,
    hop_ratio: f32,
) -> Vec<Chunk> {
    let chunk_samples = (chunk_seconds * sample_rate as f32) as usize;
    let hop_samples = ((chunk_seconds * hop_ratio) * sample_rate as f32).max(1.0) as usize;

    if audio.len() < chunk_samples {
        let mut padded = vec![0.0f32; chunk_samples];
        let n = audio.len().min(chunk_samples);
        padded[..n].copy_from_slice(&audio[..n]);
        return vec![Chunk {
            samples: padded,
            t_offset_ms: 0,
        }];
    }

    let mut out = Vec::new();
    let mut start = 0usize;
    while start + chunk_samples <= audio.len() {
        let t_offset_ms = (start as u64 * 1000) / sample_rate as u64;
        out.push(Chunk {
            samples: audio[start..start + chunk_samples].to_vec(),
            t_offset_ms,
        });
        start += hop_samples;
    }

    out
}

pub fn median_filter_multihot(input: &Multilabel, window: usize) -> Multilabel {
    if window <= 1 {
        return input.clone();
    }
    let half = window / 2;
    let mut out = vec![0u8; input.frames * input.speakers];
    let mut buf: Vec<u8> = Vec::with_capacity(window);

    for f in 0..input.frames {
        for s in 0..input.speakers {
            buf.clear();
            let lo = f.saturating_sub(half);
            let hi = (f + half + 1).min(input.frames);
            for ff in lo..hi {
                buf.push(input.data[ff * input.speakers + s]);
            }

            let ones = buf.iter().filter(|&&v| v != 0).count();
            let v = if ones * 2 > buf.len() { 1 } else { 0 };
            out[f * input.speakers + s] = v;
        }
    }

    Multilabel {
        frames: input.frames,
        speakers: input.speakers,
        data: out,
    }
}

pub fn extract_spans(
    multihot: &Multilabel,
    frame_rate_hz: u32,
    t_offset_ms: u64,
    min_frames: usize,
) -> Vec<Span> {
    let frame_ms = 1000.0f32 / frame_rate_hz as f32;
    let samples_per_frame = (16_000.0 / frame_rate_hz as f32) as usize;

    let mut overlap = vec![false; multihot.frames];
    for (f, o) in overlap.iter_mut().enumerate() {
        let row = multihot.row(f);
        let active: usize = row.iter().filter(|&&v| v != 0).count();
        *o = active >= 2;
    }

    let mut out = Vec::new();
    for s in 0..multihot.speakers {
        let mut run_start: Option<usize> = None;
        for f in 0..multihot.frames {
            let active = multihot.data[f * multihot.speakers + s] != 0;
            match (run_start, active) {
                (None, true) => run_start = Some(f),
                (Some(start), false) => {
                    push_span(
                        &mut out,
                        start,
                        f,
                        s,
                        &overlap,
                        frame_ms,
                        t_offset_ms,
                        samples_per_frame,
                        min_frames,
                    );
                    run_start = None;
                }
                _ => {}
            }
        }
        if let Some(start) = run_start {
            push_span(
                &mut out,
                start,
                multihot.frames,
                s,
                &overlap,
                frame_ms,
                t_offset_ms,
                samples_per_frame,
                min_frames,
            );
        }
    }

    out
}

#[allow(clippy::too_many_arguments)]
fn push_span(
    out: &mut Vec<Span>,
    start: usize,
    end: usize,
    speaker: usize,
    overlap: &[bool],
    frame_ms: f32,
    t_offset_ms: u64,
    samples_per_frame: usize,
    min_frames: usize,
) {
    if end <= start {
        return;
    }
    let length = end - start;
    if length < min_frames {
        return;
    }

    let overlap_frames: usize = (start..end).filter(|&i| overlap[i]).count();
    let is_overlap = overlap_frames * 2 > length;

    let t_start_ms = t_offset_ms + (start as f32 * frame_ms) as u64;
    let t_end_ms = t_offset_ms + (end as f32 * frame_ms) as u64;
    out.push(Span {
        sample_start: t_offset_to_samples(t_offset_ms) + start * samples_per_frame,
        sample_end: t_offset_to_samples(t_offset_ms) + end * samples_per_frame,
        t_start_ms,
        t_end_ms,
        local_speaker: speaker,
        overlap: is_overlap,
    });
}

#[inline]
fn t_offset_to_samples(t_offset_ms: u64) -> usize {
    (t_offset_ms as usize * 16_000) / 1000
}

pub fn coalesce_segments(mut segments: Vec<DiarSegment>) -> Vec<DiarSegment> {
    if segments.is_empty() {
        return segments;
    }
    segments.sort_by_key(|s| s.t_start_ms);

    let mut out: Vec<DiarSegment> = Vec::with_capacity(segments.len());
    for s in segments {
        if let Some(last) = out.last_mut() {
            if last.speaker == s.speaker && s.t_start_ms <= last.t_end_ms + 250 {
                last.t_end_ms = last.t_end_ms.max(s.t_end_ms);
                last.confidence = last.confidence.max(s.confidence);
                continue;
            }
        }
        out.push(s);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slide_chunks_pads_short_utterance() {
        let audio = vec![1.0f32; 8_000];
        let chunks = slide_chunks(&audio, 16_000, 5.0, 0.1);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].samples.len(), 80_000);

        assert_eq!(chunks[0].samples[10_000], 0.0);
    }

    #[test]
    fn slide_chunks_overlapping_long_utterance() {
        let audio = vec![1.0f32; 16_000 * 11];
        let chunks = slide_chunks(&audio, 16_000, 5.0, 0.1);

        assert!(chunks.len() >= 12);
        assert_eq!(chunks[1].t_offset_ms, 500);
    }

    #[test]
    fn median_filter_smooths_singleton_blip() {
        let speakers = 1;
        let frames = 7;
        let data = vec![1, 1, 1, 0, 1, 1, 1];
        let ml = Multilabel {
            frames,
            speakers,
            data,
        };
        let smoothed = median_filter_multihot(&ml, 3);
        assert_eq!(smoothed.row(3), &[1], "singleton blip should be filtered");
    }

    #[test]
    fn coalesce_merges_adjacent_same_speaker() {
        let segs = vec![
            DiarSegment {
                speaker: 0,
                t_start_ms: 0,
                t_end_ms: 500,
                confidence: 0.9,
            },
            DiarSegment {
                speaker: 0,
                t_start_ms: 600,
                t_end_ms: 1000,
                confidence: 0.85,
            },
            DiarSegment {
                speaker: 1,
                t_start_ms: 1100,
                t_end_ms: 1500,
                confidence: 0.8,
            },
        ];
        let merged = coalesce_segments(segs);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].t_end_ms, 1000);
    }
}
