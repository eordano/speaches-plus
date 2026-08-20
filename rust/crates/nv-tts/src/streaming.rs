use std::sync::Arc;

use anyhow::Result;
use candle_core::Tensor;

use nv_omni::vocoder::{Vocoder, NUM_CODEBOOKS};

use crate::talker::{Qwen3TtsTalker, CODEC_EOS_ID};
use crate::tokenizer::Qwen3TtsTokenizer;

pub const DEFAULT_CHUNK_FRAMES: usize = 4;

pub trait TalkerLike {
    fn step(&self, text_hidden: &Tensor, prev_speech: &[u32]) -> Result<u32>;

    fn step_with_speaker(
        &self,
        text_hidden: &Tensor,
        _speaker_prefix: Option<&Tensor>,
        prev_speech: &[u32],
    ) -> Result<u32> {
        self.step(text_hidden, prev_speech)
    }
}

impl TalkerLike for Qwen3TtsTalker {
    fn step(&self, text_hidden: &Tensor, prev_speech: &[u32]) -> Result<u32> {
        Qwen3TtsTalker::step(self, text_hidden, prev_speech)
    }

    fn step_with_speaker(
        &self,
        text_hidden: &Tensor,
        speaker_prefix: Option<&Tensor>,
        prev_speech: &[u32],
    ) -> Result<u32> {
        Qwen3TtsTalker::step_with_speaker(self, text_hidden, speaker_prefix, prev_speech)
    }
}

pub struct TtsStream<T: TalkerLike> {
    talker: Arc<T>,

    text_hidden: Tensor,

    speaker_prefix: Option<Tensor>,
    chunk_frames: usize,

    eos_id: u32,

    emitted: Vec<u32>,

    finished: bool,
}

impl<T: TalkerLike> TtsStream<T> {
    pub fn new(talker: Arc<T>, _tokenizer: Arc<Qwen3TtsTokenizer>, text_hidden: Tensor) -> Self {
        Self {
            talker,
            text_hidden,
            speaker_prefix: None,
            chunk_frames: DEFAULT_CHUNK_FRAMES,
            eos_id: CODEC_EOS_ID,
            emitted: Vec::new(),
            finished: false,
        }
    }

    pub fn with_speaker_prefix(mut self, prefix: Tensor) -> Self {
        self.speaker_prefix = Some(prefix);
        self
    }

    pub fn speaker_prefix(&self) -> Option<&Tensor> {
        self.speaker_prefix.as_ref()
    }

    pub fn with_chunk_frames(mut self, n: usize) -> Self {
        assert!(n > 0, "chunk_frames must be > 0");
        self.chunk_frames = n;
        self
    }

    pub fn with_eos_id(mut self, id: u32) -> Self {
        self.eos_id = id;
        self
    }

    pub fn chunk_frames(&self) -> usize {
        self.chunk_frames
    }

    pub fn emitted(&self) -> &[u32] {
        &self.emitted
    }
}

impl<T: TalkerLike> Iterator for TtsStream<T> {
    type Item = Result<Vec<u32>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let mut chunk: Vec<u32> = Vec::with_capacity(self.chunk_frames);
        for _ in 0..self.chunk_frames {
            let tok = match self.talker.step_with_speaker(
                &self.text_hidden,
                self.speaker_prefix.as_ref(),
                &self.emitted,
            ) {
                Ok(t) => t,
                Err(e) => return Some(Err(e)),
            };
            self.emitted.push(tok);
            chunk.push(tok);
            if tok == self.eos_id {
                self.finished = true;
                break;
            }
        }
        if chunk.is_empty() {
            None
        } else {
            Some(Ok(chunk))
        }
    }
}

impl<T: TalkerLike> TtsStream<T> {
    pub fn into_audio_stream(self, vocoder: Arc<Vocoder>) -> TtsAudioStream<T> {
        TtsAudioStream {
            inner: self,
            vocoder,
            full_frame: None,
            buffered_audio: Vec::new(),
            finished: false,
        }
    }

    pub fn tokens_stream(self) -> Self {
        self
    }
}

type FullFrameFn =
    Box<dyn Fn(u32, &[[u32; NUM_CODEBOOKS]]) -> Result<[u32; NUM_CODEBOOKS]> + Send + Sync>;

pub struct TtsAudioStream<T: TalkerLike> {
    inner: TtsStream<T>,
    vocoder: Arc<Vocoder>,

    full_frame: Option<FullFrameFn>,

    buffered_audio: Vec<f32>,
    finished: bool,
}

impl<T: TalkerLike> TtsAudioStream<T> {
    pub fn with_full_frame_fn(mut self, f: FullFrameFn) -> Self {
        self.full_frame = Some(f);
        self
    }

    pub fn samples_per_frame(&self) -> usize {
        self.vocoder.config().upsample_factor()
    }

    fn expand_frame(
        &self,
        base_tok: u32,
        history: &[[u32; NUM_CODEBOOKS]],
    ) -> Result<[u32; NUM_CODEBOOKS]> {
        if let Some(f) = &self.full_frame {
            f(base_tok, history)
        } else {
            let mut row = [0u32; NUM_CODEBOOKS];
            row[0] = base_tok;
            Ok(row)
        }
    }
}

impl<T: TalkerLike> Iterator for TtsAudioStream<T> {
    type Item = Result<Vec<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let tokens_chunk = match self.inner.next() {
            None => {
                self.finished = true;
                if self.buffered_audio.is_empty() {
                    return None;
                }
                let out = std::mem::take(&mut self.buffered_audio);
                return Some(Ok(out));
            }
            Some(Err(e)) => return Some(Err(e)),
            Some(Ok(v)) => v,
        };

        let eos_id = self.inner.eos_id;
        let mut frames: Vec<[u32; NUM_CODEBOOKS]> = Vec::with_capacity(tokens_chunk.len());
        let history_so_far: Vec<[u32; NUM_CODEBOOKS]> = Vec::new();
        for &tok in &tokens_chunk {
            if tok == eos_id {
                continue;
            }
            let frame = match self.expand_frame(tok, &history_so_far) {
                Ok(f) => f,
                Err(e) => return Some(Err(e)),
            };
            frames.push(frame);
        }

        if frames.len() < 2 {
            if self.inner.finished {
                self.finished = true;
                if self.buffered_audio.is_empty() && frames.is_empty() {
                    return None;
                }

                if frames.is_empty() {
                    let out = std::mem::take(&mut self.buffered_audio);
                    return Some(Ok(out));
                }
                let frame_count = frames.len();
                while frames.len() < 2 {
                    frames.push([0u32; NUM_CODEBOOKS]);
                }
                let pcm = match self.vocoder.decode(&frames) {
                    Ok(p) => p,
                    Err(e) => return Some(Err(e)),
                };
                let keep = frame_count * self.samples_per_frame();
                let mut out = std::mem::take(&mut self.buffered_audio);
                out.extend_from_slice(&pcm[..keep.min(pcm.len())]);
                return Some(Ok(out));
            }

            return Some(Ok(std::mem::take(&mut self.buffered_audio)));
        }

        let pcm = match self.vocoder.decode(&frames) {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };
        let mut out = std::mem::take(&mut self.buffered_audio);
        out.extend_from_slice(&pcm);
        if self.inner.finished {
            self.finished = true;
        }
        Some(Ok(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct StubTalker {
        eos_id: u32,
        eos_at: usize,
        calls: AtomicUsize,
    }

    impl TalkerLike for StubTalker {
        fn step(&self, _text_hidden: &Tensor, _prev_speech: &[u32]) -> Result<u32> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n == self.eos_at {
                Ok(self.eos_id)
            } else {
                Ok(0)
            }
        }
    }

    fn cached_tokenizer_or_skip() -> Option<Arc<Qwen3TtsTokenizer>> {
        let dir = crate::tokenizer::qwen3_tts_cache_dir()?;
        let tk = Qwen3TtsTokenizer::from_dir(&dir).ok()?;
        Some(Arc::new(tk))
    }

    fn tokenizer_for_stream() -> Arc<Qwen3TtsTokenizer> {
        if let Some(tk) = cached_tokenizer_or_skip() {
            return tk;
        }

        static FIXTURE_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let seq = FIXTURE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = std::env::temp_dir().join(format!(
            "nv_tts_streaming_test_tokenizer_{}_{seq}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&tmp);
        let vocab = tmp.join("vocab.json");
        let merges = tmp.join("merges.txt");

        let mut bs: Vec<u32> = Vec::new();
        for b in b'!'..=b'~' {
            bs.push(b as u32);
        }
        for b in 0xa1u32..=0xacu32 {
            bs.push(b);
        }
        for b in 0xaeu32..=0xffu32 {
            bs.push(b);
        }
        let mut cs: Vec<u32> = bs.clone();
        let mut n = 0u32;
        for b in 0u32..=255u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut entries: Vec<(String, u32)> = Vec::with_capacity(256);
        for (i, &c) in cs.iter().enumerate() {
            let ch = char::from_u32(c).unwrap_or('?');
            entries.push((ch.to_string(), i as u32));
        }
        let mut vocab_json = String::from("{");
        for (i, (k, v)) in entries.iter().enumerate() {
            if i > 0 {
                vocab_json.push(',');
            }
            let escaped = serde_json::to_string(k).unwrap();
            vocab_json.push_str(&format!("{escaped}:{v}"));
        }
        vocab_json.push('}');
        std::fs::write(&vocab, &vocab_json).unwrap();
        std::fs::write(&merges, "#version: 0.2\n").unwrap();
        Arc::new(Qwen3TtsTokenizer::from_files(&vocab, &merges, None).expect("synth tokenizer"))
    }

    fn dummy_text_hidden() -> Tensor {
        Tensor::ones((1usize, 1usize, 16usize), DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn yields_chunks_of_expected_size() {
        let talker = Arc::new(StubTalker {
            eos_id: 999,
            eos_at: 12,
            calls: AtomicUsize::new(0),
        });
        let tk = tokenizer_for_stream();
        let stream = TtsStream::new(talker, tk, dummy_text_hidden()).with_eos_id(999);
        let mut chunks: Vec<Vec<u32>> = Vec::new();
        for c in stream {
            chunks.push(c.expect("chunk"));
        }

        assert_eq!(chunks.len(), 4, "got chunks: {chunks:?}");
        assert_eq!(chunks[0], vec![0, 0, 0, 0]);
        assert_eq!(chunks[1], vec![0, 0, 0, 0]);
        assert_eq!(chunks[2], vec![0, 0, 0, 0]);
        assert_eq!(chunks[3], vec![999]);
    }

    #[test]
    fn stops_on_eos_within_first_chunk() {
        let talker = Arc::new(StubTalker {
            eos_id: 5,
            eos_at: 2,
            calls: AtomicUsize::new(0),
        });
        let tk = tokenizer_for_stream();
        let stream = TtsStream::new(talker, tk, dummy_text_hidden()).with_eos_id(5);
        let chunks: Vec<Vec<u32>> = stream.map(|r| r.unwrap()).collect();
        assert_eq!(chunks, vec![vec![0u32, 0, 5]]);
    }

    #[test]
    fn custom_chunk_size() {
        let talker = Arc::new(StubTalker {
            eos_id: 999,
            eos_at: 5,
            calls: AtomicUsize::new(0),
        });
        let tk = tokenizer_for_stream();
        let stream = TtsStream::new(talker, tk, dummy_text_hidden())
            .with_chunk_frames(2)
            .with_eos_id(999);
        let chunks: Vec<Vec<u32>> = stream.map(|r| r.unwrap()).collect();

        assert_eq!(chunks, vec![vec![0u32, 0], vec![0, 0], vec![0, 999]]);
    }

    struct SpkAwareStubTalker;
    impl TalkerLike for SpkAwareStubTalker {
        fn step(&self, _text_hidden: &Tensor, _prev_speech: &[u32]) -> Result<u32> {
            Ok(0u32)
        }
        fn step_with_speaker(
            &self,
            _text_hidden: &Tensor,
            speaker_prefix: Option<&Tensor>,
            _prev_speech: &[u32],
        ) -> Result<u32> {
            let v = match speaker_prefix {
                None => 0u32,
                Some(t) => {
                    let s = t
                        .to_dtype(DType::F32)?
                        .abs()?
                        .sum_all()?
                        .to_scalar::<f32>()?;
                    (s.abs() as u32).min(100)
                }
            };
            Ok(v)
        }
    }

    #[test]
    fn tts_stream_with_speaker_prefix_reaches_the_talker() {
        let talker = Arc::new(SpkAwareStubTalker);
        let tk = tokenizer_for_stream();

        let stream = TtsStream::new(talker.clone(), tk.clone(), dummy_text_hidden())
            .with_chunk_frames(2)
            .with_eos_id(999);
        let chunks: Vec<Vec<u32>> = stream.take(2).map(|r| r.unwrap()).collect();
        assert_eq!(
            chunks[0],
            vec![0u32, 0u32],
            "without a speaker prefix the stub must observe None"
        );

        let hidden_size = 16usize;
        let dev = Device::Cpu;
        let prefix: Vec<f32> = (0..hidden_size)
            .map(|i| (i as f32 * 0.3).sin() + 1.5)
            .collect();
        let prefix = Tensor::from_vec(prefix, (1usize, 1usize, hidden_size), &dev).unwrap();
        let stream = TtsStream::new(talker.clone(), tk.clone(), dummy_text_hidden())
            .with_chunk_frames(2)
            .with_eos_id(999)
            .with_speaker_prefix(prefix);
        assert!(
            stream.speaker_prefix().is_some(),
            "prefix should be materialised"
        );
        let chunks: Vec<Vec<u32>> = stream.take(1).map(|r| r.unwrap()).collect();
        assert_ne!(
            chunks[0],
            vec![0u32, 0u32],
            "with a non-zero speaker prefix the stub must observe it"
        );
    }

    #[test]
    fn audio_stream_emits_pcm_chunks_with_zero_init_vocoder() {
        use nv_omni::vocoder::{Vocoder, VocoderConfig, NUM_CODEBOOKS as NCB};

        let talker = Arc::new(StubTalker {
            eos_id: 999,
            eos_at: 6,
            calls: AtomicUsize::new(0),
        });
        let tk = tokenizer_for_stream();
        let token_stream = TtsStream::new(talker, tk, dummy_text_hidden())
            .with_chunk_frames(3)
            .with_eos_id(999);

        let cfg = VocoderConfig {
            codebook_size: 16,
            num_codebooks: NCB,
            codebook_dim: 4,
            quant_proj_dim: 8,
            latent_dim: 16,
            pre_upsample_dim: 16,
            decoder_dim: 32,
            ..VocoderConfig::default()
        };
        let voc = Arc::new(Vocoder::new(cfg.clone(), &Device::Cpu).unwrap());
        let upsample = cfg.upsample_factor();

        let audio = token_stream.into_audio_stream(voc);
        let mut total = 0usize;
        for c in audio {
            let pcm = c.expect("audio chunk");
            total += pcm.len();

            assert!(
                pcm.iter().all(|s| *s == 0.0),
                "non-zero sample from zero-init vocoder"
            );
        }

        assert_eq!(
            total,
            6 * upsample,
            "expected {} samples, got {total}",
            6 * upsample
        );
    }

    #[test]
    fn audio_stream_uses_full_frame_callback_when_set() {
        use nv_omni::vocoder::{Vocoder, VocoderConfig, NUM_CODEBOOKS as NCB};
        use std::sync::Mutex;

        let talker = Arc::new(StubTalker {
            eos_id: 999,
            eos_at: 4,
            calls: AtomicUsize::new(0),
        });
        let tk = tokenizer_for_stream();
        let token_stream = TtsStream::new(talker, tk, dummy_text_hidden())
            .with_chunk_frames(2)
            .with_eos_id(999);

        let cfg = VocoderConfig {
            codebook_size: 8,
            num_codebooks: NCB,
            codebook_dim: 4,
            quant_proj_dim: 8,
            latent_dim: 16,
            pre_upsample_dim: 16,
            decoder_dim: 32,
            ..VocoderConfig::default()
        };
        let voc = Arc::new(Vocoder::new(cfg.clone(), &Device::Cpu).unwrap());

        let invocations: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let cb_inv = invocations.clone();
        let cb: super::FullFrameFn = Box::new(move |base, _history| {
            cb_inv.lock().unwrap().push(base);
            let mut row = [0u32; NCB];
            row[0] = base;
            row[1] = 1;
            Ok(row)
        });

        let audio = token_stream.into_audio_stream(voc).with_full_frame_fn(cb);
        let chunks: Vec<Vec<f32>> = audio.map(|r| r.unwrap()).collect();
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(
            total,
            4 * cfg.upsample_factor(),
            "4 emitted base tokens -> 4 frames"
        );
        let calls = invocations.lock().unwrap();
        assert_eq!(
            calls.len(),
            4,
            "callback should fire once per non-EOS token, got {:?}",
            *calls
        );
    }

    #[test]
    fn tokens_stream_method_is_identity() {
        let talker = Arc::new(StubTalker {
            eos_id: 999,
            eos_at: 1,
            calls: AtomicUsize::new(0),
        });
        let tk = tokenizer_for_stream();
        let stream = TtsStream::new(talker, tk, dummy_text_hidden())
            .with_chunk_frames(2)
            .with_eos_id(999)
            .tokens_stream();
        let chunks: Vec<Vec<u32>> = stream.map(|r| r.unwrap()).collect();
        assert_eq!(chunks, vec![vec![0u32, 999]]);
    }

    #[test]
    fn streaming_vs_batch_decode_matches_for_zero_init_vocoder() {
        use nv_omni::vocoder::{Vocoder, VocoderConfig, NUM_CODEBOOKS as NCB};

        let total_frames = 12usize;
        let chunk_sizes = [2usize, 3, 4, 6, 12];

        let cfg = VocoderConfig {
            codebook_size: 16,
            num_codebooks: NCB,
            codebook_dim: 4,
            quant_proj_dim: 8,
            latent_dim: 16,
            pre_upsample_dim: 16,
            decoder_dim: 32,
            ..VocoderConfig::default()
        };
        let voc = Arc::new(Vocoder::new(cfg.clone(), &Device::Cpu).unwrap());

        let frames: Vec<[u32; NCB]> = (0..total_frames)
            .map(|i| {
                let mut row = [0u32; NCB];
                row[0] = (i % cfg.codebook_size) as u32;
                row
            })
            .collect();

        let batch_pcm = voc.decode(&frames).unwrap();

        for &cs in &chunk_sizes {
            let talker = Arc::new(StubTalker {
                eos_id: 999,
                eos_at: total_frames,
                calls: AtomicUsize::new(0),
            });
            let tk = tokenizer_for_stream();
            let stream = TtsStream::new(talker, tk, dummy_text_hidden())
                .with_chunk_frames(cs)
                .with_eos_id(999);
            let audio = stream.into_audio_stream(voc.clone());
            let streamed: Vec<f32> = audio.flat_map(|r| r.unwrap()).collect();

            assert_eq!(
                streamed.len(),
                batch_pcm.len(),
                "chunk_size={cs}: sample count mismatch: streamed {} vs batch {}",
                streamed.len(),
                batch_pcm.len()
            );
            for (i, (s, b)) in streamed.iter().zip(batch_pcm.iter()).enumerate() {
                assert!(
                    (s - b).abs() < 1e-5,
                    "chunk_size={cs}: sample {i} differs: streamed {s} vs batch {b}"
                );
            }
        }
    }

    #[allow(dead_code)]
    fn _typecheck_real_talker_stream(p: &Path) {
        use crate::talker::{Qwen3TtsTalker, Qwen3TtsTalkerConfig};
        let dev = Device::Cpu;
        let _ = (p, &dev);
        let cfg = Qwen3TtsTalkerConfig::default();
        let talker: Qwen3TtsTalker = Qwen3TtsTalker::new(cfg, &dev).unwrap();
        let tk = cached_tokenizer_or_skip().unwrap_or_else(tokenizer_for_stream);
        let txt = Tensor::zeros((1usize, 1usize, 2048usize), DType::BF16, &dev).unwrap();
        let _stream: TtsStream<Qwen3TtsTalker> = TtsStream::new(Arc::new(talker), tk, txt);
    }
}
