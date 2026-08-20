use anyhow::{bail, Result};
use candle_core::Tensor;

pub const GEMMA4_IMAGE_TOKEN_ID: u32 = 258880;
pub const GEMMA4_AUDIO_TOKEN_ID: u32 = 258881;
pub const GEMMA4_BOI_TOKEN_ID: u32 = 255999;
pub const GEMMA4_EOI_TOKEN_ID: u32 = 258882;
pub const GEMMA4_BOA_TOKEN_ID: u32 = 256000;
pub const GEMMA4_EOA_TOKEN_ID: u32 = 258883;
pub const VISION_SOFT_TOKENS_PER_IMAGE: usize = 280;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Image,
    Audio,
}

impl Modality {
    pub fn placeholder_token_id(self) -> u32 {
        match self {
            Modality::Image => GEMMA4_IMAGE_TOKEN_ID,
            Modality::Audio => GEMMA4_AUDIO_TOKEN_ID,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaceholderRun {
    pub modality: Modality,
    pub start: usize,
    pub len: usize,
}

#[derive(Debug)]
pub struct MmItem {
    pub modality: Modality,
    pub position: usize,
    pub embedding: Tensor,
}

pub fn expand_image_placeholder(num_soft_tokens: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(num_soft_tokens + 2);
    out.push(GEMMA4_BOI_TOKEN_ID);
    out.extend(std::iter::repeat_n(GEMMA4_IMAGE_TOKEN_ID, num_soft_tokens));
    out.push(GEMMA4_EOI_TOKEN_ID);
    out
}

pub fn expand_audio_placeholder(num_soft_tokens: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(num_soft_tokens + 2);
    out.push(GEMMA4_BOA_TOKEN_ID);
    out.extend(std::iter::repeat_n(GEMMA4_AUDIO_TOKEN_ID, num_soft_tokens));
    out.push(GEMMA4_EOA_TOKEN_ID);
    out
}

pub fn audio_num_soft_tokens(
    num_samples: usize,
    sampling_rate: usize,
    audio_seq_length: usize,
) -> usize {
    let frame_length = (sampling_rate as f64 * 20.0 / 1000.0).round() as usize;
    let hop_length = (sampling_rate as f64 * 10.0 / 1000.0).round() as usize;
    let frame_size_for_unfold = frame_length + 1;
    let pad_left = frame_length / 2;
    let padded_samples = num_samples + pad_left;
    if padded_samples < frame_size_for_unfold || hop_length == 0 {
        return 0;
    }
    let num_mel_frames = (padded_samples - frame_size_for_unfold) / hop_length + 1;
    let mut t = num_mel_frames;
    for _ in 0..2 {
        t = (t - 1) / 2 + 1;
    }
    t.min(audio_seq_length)
}

pub fn placeholder_runs(tokens: &[u32]) -> Vec<PlaceholderRun> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let modality = match tokens[i] {
            GEMMA4_IMAGE_TOKEN_ID => Modality::Image,
            GEMMA4_AUDIO_TOKEN_ID => Modality::Audio,
            _ => {
                i += 1;
                continue;
            }
        };
        let start = i;
        let id = tokens[start];
        while i < tokens.len() && tokens[i] == id {
            i += 1;
        }
        runs.push(PlaceholderRun {
            modality,
            start,
            len: i - start,
        });
    }
    runs
}

pub fn splice_mm_embeddings(
    text_embeds: &Tensor,
    tokens: &[u32],
    items: &[MmItem],
) -> Result<Tensor> {
    let dims = text_embeds.dims().to_vec();
    if dims.len() != 2 {
        bail!(
            "splice_mm_embeddings: text_embeds must be [seq, hidden], got {:?}",
            dims
        );
    }
    let (seq, hidden) = (dims[0], dims[1]);
    if seq != tokens.len() {
        bail!(
            "splice_mm_embeddings: text_embeds has {} rows but {} tokens were given",
            seq,
            tokens.len()
        );
    }
    let runs = placeholder_runs(tokens);
    if items.is_empty() {
        if !runs.is_empty() {
            bail!(
                "splice_mm_embeddings: prompt contains {} multimodal placeholder run(s) but no embeddings were provided",
                runs.len()
            );
        }
        return Ok(text_embeds.clone());
    }
    if items.len() != runs.len() {
        bail!(
            "splice_mm_embeddings: prompt contains {} placeholder run(s) but {} embedding item(s) were provided",
            runs.len(),
            items.len()
        );
    }
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by_key(|&i| items[i].position);
    let mut pieces: Vec<Tensor> = Vec::with_capacity(runs.len() * 2 + 1);
    let mut cursor = 0usize;
    for (run, &idx) in runs.iter().zip(order.iter()) {
        let item = &items[idx];
        if item.position != run.start {
            bail!(
                "splice_mm_embeddings: item at position {} does not align with the {:?} placeholder run starting at {}",
                item.position,
                run.modality,
                run.start
            );
        }
        if item.modality != run.modality {
            bail!(
                "splice_mm_embeddings: {:?} embedding given for a {:?} placeholder run at position {}",
                item.modality,
                run.modality,
                run.start
            );
        }
        let edims = item.embedding.dims().to_vec();
        if edims.len() != 2 || edims[1] != hidden {
            bail!(
                "splice_mm_embeddings: {:?} embedding at position {} must be [n, {}], got {:?}",
                item.modality,
                run.start,
                hidden,
                edims
            );
        }
        if edims[0] != run.len {
            bail!(
                "splice_mm_embeddings: {:?} placeholder run at position {} spans {} token(s) but the tower produced {} embedding(s)",
                run.modality,
                run.start,
                run.len,
                edims[0]
            );
        }
        if item.embedding.dtype() != text_embeds.dtype() {
            bail!(
                "splice_mm_embeddings: embedding dtype {:?} does not match text dtype {:?}",
                item.embedding.dtype(),
                text_embeds.dtype()
            );
        }
        if run.start > cursor {
            pieces.push(text_embeds.narrow(0, cursor, run.start - cursor)?);
        }
        pieces.push(item.embedding.clone());
        cursor = run.start + run.len;
    }
    if cursor < seq {
        pieces.push(text_embeds.narrow(0, cursor, seq - cursor)?);
    }
    Ok(Tensor::cat(&pieces, 0)?)
}
