pub mod albert;
pub mod decoder;
pub mod fbank;
pub mod frontend;
pub mod generator;
pub mod layers;
pub mod melspec;
pub mod onnx_exec;
pub mod prosody;
pub mod serve;
pub mod style_fit;
pub mod train;
pub mod wespeaker;

use std::path::Path;

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarBuilder;

pub use albert::{Albert, AlbertConfig};
pub use decoder::DecoderBody;
pub use frontend::{alignment, durations, DurationPath, TextEncoder};
pub use generator::{Generator, GeneratorConfig, Rng};
pub use layers::{differentiable, set_differentiable};
pub use melspec::LogMel24k;
pub use prosody::Prosody;
pub use serve::KokoroCandle;
pub use train::{adapt, apply_adain, Adaptation, Encoders, EpochReport, Targets, TrainConfig};
pub use wespeaker::{resample_24k_to_16k, WeSpeaker};

pub const A_TRAINING_WINDOW_CROPS_THE_DECODER_INPUT_SO_THE_GENERATOR_TAPE_SCALES_WITH_THE_WINDOW_NOT_THE_SENTENCE: &str =
    "candle retains every elementwise intermediate of the generator's resblocks at the 24 kHz \
     frame rate, about 1.2 GiB per second of audio under autograd, so an adapter step that must \
     fit a wasm32 heap crops asr, f0 and n to the first window_frames decoder frames (300 \
     samples each) before the decoder and generator run; the speaker loss then sees the \
     window's audio, validation still hears the whole sentence, and None keeps the full \
     sentence the engine trains on";

pub struct KokoroFrontEnd {
    pub bert: Albert,
    pub duration: DurationPath,
    pub text_encoder: TextEncoder,
    pub prosody: Prosody,
    pub decoder: DecoderBody,
    pub generator: Generator,
    pub device: Device,
}

impl KokoroFrontEnd {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::from_var_builder(vb, device)
    }

    pub fn from_safetensors_bytes(bytes: Vec<u8>, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_buffered_safetensors(bytes, DType::F32, device)?;
        Self::from_var_builder(vb, device)
    }

    fn from_var_builder(vb: VarBuilder<'static>, device: &Device) -> Result<Self> {
        Ok(Self {
            bert: Albert::load(vb.pp("bert"), AlbertConfig::default())?,
            duration: DurationPath::load(vb.clone(), 3)?,
            text_encoder: TextEncoder::load(vb.pp("text_encoder"), 3, 5)?,
            prosody: Prosody::load(vb.pp("predictor"))?,
            decoder: DecoderBody::load(vb.pp("decoder"))?,
            generator: Generator::load(vb.pp("decoder.generator"), GeneratorConfig::default())?,
            device: device.clone(),
        })
    }

    pub fn split_style(ref_s: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((ref_s.narrow(1, 0, 128)?, ref_s.narrow(1, 128, 128)?))
    }

    pub fn synthesize_tokens(
        &self,
        ids: &[u32],
        ref_s: &Tensor,
        speed: f64,
        deterministic: bool,
        rng: &mut Rng,
    ) -> Result<(Tensor, Vec<usize>)> {
        self.synthesize_tokens_window(ids, ref_s, speed, deterministic, rng, None)
    }

    pub fn synthesize_tokens_window(
        &self,
        ids: &[u32],
        ref_s: &Tensor,
        speed: f64,
        deterministic: bool,
        rng: &mut Rng,
        window_frames: Option<usize>,
    ) -> Result<(Tensor, Vec<usize>)> {
        let _ = A_TRAINING_WINDOW_CROPS_THE_DECODER_INPUT_SO_THE_GENERATOR_TAPE_SCALES_WITH_THE_WINDOW_NOT_THE_SENTENCE;
        let dev = &self.device;
        let input = Tensor::new(ids, dev)?.unsqueeze(0)?;
        let (dec_s, pro_s) = Self::split_style(ref_s)?;
        let (pro_s, dec_s) = (pro_s.detach(), dec_s);
        let bert = self.bert.forward(&input)?;
        let (_d_en, d, logits) = self.duration.forward(&bert, &pro_s)?;
        let dur = durations(&logits, speed)?;
        let aln = alignment(&dur, dev)?.unsqueeze(0)?;
        let en = d.transpose(1, 2)?.contiguous()?.matmul(&aln)?;
        let (f0, n) = self.prosody.forward(&en, &pro_s)?;
        let asr = self.text_encoder.forward(&input)?.matmul(&aln)?;
        let (f0, n, asr) = (f0.detach(), n.detach(), asr.detach());
        let (f0, n, asr) = match window_frames {
            Some(k) if k < asr.dim(2)? => {
                let ratio = f0.dim(1)? / asr.dim(2)?;
                (
                    f0.narrow(1, 0, k * ratio)?,
                    n.narrow(1, 0, k * ratio)?,
                    asr.narrow(2, 0, k)?,
                )
            }
            _ => (f0, n, asr),
        };
        let (_enc, pre_gen) = self.decoder.forward(&asr, &f0, &n, &dec_s)?;
        let audio = self
            .generator
            .forward(&pre_gen, &dec_s, &f0, deterministic, rng)?;
        Ok((audio, dur))
    }

    pub fn make_adain_trainable(&mut self) -> Result<Vec<Var>> {
        set_differentiable(true);
        let mut vars = Vec::new();
        for a in self
            .decoder
            .adains_mut()
            .into_iter()
            .chain(self.generator.adains_mut())
        {
            vars.extend(a.make_trainable()?);
        }
        Ok(vars)
    }

    pub fn adain_names(&mut self) -> Vec<String> {
        self.decoder
            .adains_mut()
            .into_iter()
            .chain(self.generator.adains_mut())
            .map(|a| a.fc_name.clone())
            .collect()
    }
}

pub fn load_speaker_encoder(dir: &Path, device: &Device) -> Result<crate::SpeakerEncoder> {
    let raw = std::fs::read_to_string(dir.join("config.json"))?;
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    let mut cfg = crate::speaker_encoder::SpeakerEncoderConfig::default();
    let u = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
    let list = |k: &str| {
        v.get(k).and_then(|x| x.as_array()).map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|x| x as usize))
                .collect::<Vec<_>>()
        })
    };
    if let Some(x) = u("mel_dim") {
        cfg.mel_dim = x;
    }
    if let Some(x) = u("enc_dim") {
        cfg.enc_dim = x;
    }
    if let Some(x) = list("enc_channels") {
        cfg.enc_channels = x;
    }
    if let Some(x) = list("enc_kernel_sizes") {
        cfg.enc_kernel_sizes = x;
    }
    if let Some(x) = list("enc_dilations") {
        cfg.enc_dilations = x;
    }
    if let Some(x) = u("enc_attention_channels") {
        cfg.enc_attention_channels = x;
    }
    if let Some(x) = u("enc_res2net_scale") {
        cfg.enc_res2net_scale = x;
    }
    if let Some(x) = u("enc_se_channels") {
        cfg.enc_se_channels = x;
    }
    if let Some(x) = u("sample_rate") {
        cfg.sample_rate = x;
    }
    let tensors = candle_core::safetensors::load(dir.join("model.safetensors"), device)?;
    let get = |name: &str| -> Result<Tensor> {
        let t = tensors
            .get(name)
            .ok_or_else(|| anyhow!("speaker encoder: missing tensor {name}"))?;
        Ok(t.to_dtype(DType::F32)?)
    };
    crate::SpeakerEncoder::from_getter(cfg, &get, device)
}

pub fn cosine(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a = a.flatten_all()?;
    let b = b.flatten_all()?;
    let num = (&a * &b)?.sum_all()?;
    let den = (a.sqr()?.sum_all()?.sqrt()? * b.sqr()?.sum_all()?.sqrt()?)?;
    Ok((num / (den + 1e-9)?)?)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::layers::assert_close;
    use super::*;

    fn env_paths() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        let w = std::env::var_os("KOKORO_SAFETENSORS")?;
        let r = std::env::var_os("KOKORO_REF")?;
        Some((w.into(), r.into()))
    }

    #[test]
    fn stage1_front_end_matches_torch_reference() -> Result<()> {
        let Some((weights, reference)) = env_paths() else {
            eprintln!("SKIP: set KOKORO_SAFETENSORS and KOKORO_REF");
            return Ok(());
        };
        let dev = Device::Cpu;
        let m = KokoroFrontEnd::load(&weights, &dev)?;
        let r = candle_core::safetensors::load(&reference, &dev)?;
        let ids = r["input_ids"].to_dtype(DType::U32)?;
        let ref_s = &r["ref_s"];
        let (_dec_s, pro_s) = KokoroFrontEnd::split_style(ref_s)?;
        let bert = m.bert.forward(&ids)?;
        let e0 = assert_close("bert_dur", &bert, &r["bert_dur"], 2e-3)?;
        let (d_en, d, logits) = m.duration.forward(&bert, &pro_s)?;
        let e1 = assert_close("d_en", &d_en, &r["d_en"], 2e-3)?;
        let e2 = assert_close("d", &d, &r["d"], 5e-3)?;
        let e3 = assert_close("duration_logits", &logits, &r["duration_logits"], 5e-3)?;
        let dur = durations(&logits, 1.0)?;
        let ref_dur: Vec<i64> = r["pred_dur"].to_vec1()?;
        let mism = dur
            .iter()
            .zip(&ref_dur)
            .filter(|(a, b)| **a as i64 != **b)
            .count();
        let t_en = m.text_encoder.forward(&ids)?;
        let e4 = assert_close("t_en", &t_en, &r["t_en"], 5e-3)?;
        eprintln!("stage1 max abs diffs: bert {e0:.2e} d_en {e1:.2e} d {e2:.2e} logits {e3:.2e} t_en {e4:.2e}; duration mismatches {mism}/{}", dur.len());
        assert!(mism <= 1, "durations differ from torch on {mism} tokens");
        Ok(())
    }

    #[test]
    fn stage2_3_prosody_and_decoder_match_torch_reference() -> Result<()> {
        let Some((weights, reference)) = env_paths() else {
            eprintln!("SKIP: set KOKORO_SAFETENSORS and KOKORO_REF");
            return Ok(());
        };
        let dev = Device::Cpu;
        let m = KokoroFrontEnd::load(&weights, &dev)?;
        let r = candle_core::safetensors::load(&reference, &dev)?;
        let ids = r["input_ids"].to_dtype(DType::U32)?;
        let (dec_s, pro_s) = KokoroFrontEnd::split_style(&r["ref_s"])?;

        let ref_dur: Vec<i64> = r["pred_dur"].to_vec1()?;
        let dur: Vec<usize> = ref_dur.iter().map(|d| *d as usize).collect();
        let aln = alignment(&dur, &dev)?.unsqueeze(0)?;
        let en = r["d"].transpose(1, 2)?.contiguous()?.matmul(&aln)?;
        let e0 = assert_close("en", &en, &r["en"], 5e-3)?;
        let (f0, n) = m.prosody.forward(&en, &pro_s)?;
        let e1 = assert_close("F0_pred", &f0, &r["F0_pred"], 5e-3)?;
        let e2 = assert_close("N_pred", &n, &r["N_pred"], 5e-3)?;
        let t_en = m.text_encoder.forward(&ids)?;
        let asr = t_en.matmul(&aln)?;
        let e3 = assert_close("asr", &asr, &r["asr"], 5e-3)?;
        let (enc, pre_gen) = m
            .decoder
            .forward(&asr, &r["F0_pred"], &r["N_pred"], &dec_s)?;
        let e4 = assert_close("dec_encode", &enc, &r["dec_encode"], 5e-3)?;
        let e5 = assert_close("dec_pre_gen", &pre_gen, &r["dec_pre_gen"], 1e-2)?;
        eprintln!("stage2/3 max abs diffs: en {e0:.2e} F0 {e1:.2e} N {e2:.2e} asr {e3:.2e} dec_encode {e4:.2e} dec_pre_gen {e5:.2e}");
        Ok(())
    }

    #[test]
    fn conv_transpose_via_conv1d_matches_native() -> Result<()> {
        use candle_nn::VarMap;
        let dev = Device::Cpu;
        for &(cin, cout, k, stride, pad, opad, groups) in &[
            (4usize, 6usize, 20usize, 10usize, 5usize, 0usize, 1usize),
            (6, 6, 3, 2, 1, 1, 6),
            (8, 4, 12, 6, 3, 0, 1),
        ] {
            let vm = VarMap::new();
            let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
            let _ = vb.get_with_hints(
                (cin, cout / groups, k),
                "weight",
                candle_nn::Init::Randn {
                    mean: 0.0,
                    stdev: 0.5,
                },
            )?;
            let _ = vb.get_with_hints(
                cout,
                "bias",
                candle_nn::Init::Randn {
                    mean: 0.0,
                    stdev: 0.5,
                },
            )?;
            let ct = layers::ConvT1d::load(vb, cin, cout, k, stride, pad, opad, groups)?;
            let x = Tensor::randn(0f32, 1.0, (1, cin, 17), &dev)?;
            set_differentiable(false);
            let a = ct.forward(&x)?;
            let b = ct.forward_via_conv1d(&x)?;
            layers::assert_close(&format!("convT k{k} s{stride} g{groups}"), &b, &a, 1e-4)?;
        }
        Ok(())
    }

    #[test]
    fn differentiable_log_mel_matches_cpu_reference() -> Result<()> {
        let dev = Device::Cpu;
        let wav: Vec<f32> = (0..24_000 * 2)
            .map(|i| ((i as f32) * 0.02).sin() * 0.3 + ((i as f32) * 0.0031).cos() * 0.1)
            .collect();
        let (cpu, frames) = crate::spk_mel::log_mel_24k(&wav)?;
        let want = Tensor::from_vec(cpu, (1, 128, frames), &dev)?;
        let got = LogMel24k::new(&dev)?.forward(&Tensor::from_vec(wav, 24_000 * 2, &dev)?)?;
        let e = layers::assert_close("log_mel", &got, &want, 5e-3)?;
        eprintln!("log-mel max abs diff {e:.2e} over {frames} frames");
        Ok(())
    }

    #[test]
    fn wespeaker_candle_matches_onnxruntime() -> Result<()> {
        let (Some(dir), Some(reference)) = (
            std::env::var_os("WESPEAKER_CANDLE"),
            std::env::var_os("WESPEAKER_REF"),
        ) else {
            eprintln!("SKIP: set WESPEAKER_CANDLE and WESPEAKER_REF");
            return Ok(());
        };
        let dev = Device::Cpu;
        let ws = WeSpeaker::load(Path::new(&dir), &dev)?;
        let r = candle_core::safetensors::load(&reference, &dev)?;
        let feats = ws.fbank.forward(&r["wav16"])?;
        let e0 = layers::assert_close("fbank", &feats, &r["fbank"], 2e-3)?;
        let emb = ws.embed_feats(&r["fbank"])?;
        let e1 = layers::assert_close("emb", &emb, &r["emb"], 2e-3)?;
        let cos = cosine(&emb, &r["emb"])?.to_scalar::<f32>()?;
        let emb2 = ws.embed_wav16k(&r["wav16"])?;
        let cos2 = cosine(&emb2, &r["emb"])?.to_scalar::<f32>()?;
        eprintln!("wespeaker: fbank max diff {e0:.2e}, emb max diff {e1:.2e}, cos {cos:.6} (from own fbank {cos2:.6})");
        assert!(
            cos > 0.9999 && cos2 > 0.999,
            "wespeaker cosine {cos} / {cos2}"
        );
        Ok(())
    }

    #[test]
    fn serve_handle_swaps_adapters_and_restores_base() -> Result<()> {
        let Some((weights, reference)) = env_paths() else {
            eprintln!("SKIP");
            return Ok(());
        };
        let dev = Device::Cpu;
        let h = KokoroCandle::load(&weights, None, &dev)?;
        assert_eq!(h.n_adain(), 58);
        assert_eq!(h.base_adain.len(), 116);
        for k in [
            "decoder.decoder.encode.norm1.fc.weight",
            "decoder.decoder.decode.0.norm1.fc.bias",
            "decoder.decoder.generator.noise_res.0.adain1.0.fc.weight",
        ] {
            assert!(
                h.base_adain.contains_key(k),
                "{k}: adapter keys must be the ONNX initializer names adapter.npz uses"
            );
        }
        let r = candle_core::safetensors::load(&reference, &dev)?;
        let ids: Vec<u32> = r["input_ids"]
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1()?;
        let style: Vec<f32> = r["ref_s"].flatten_all()?.to_vec1()?;
        swap_and_restore(&h, &ids, &style, &dev)
    }

    fn swap_and_restore(
        h: &KokoroCandle,
        ids: &[u32],
        style: &[f32],
        adapter_dev: &Device,
    ) -> Result<()> {
        let a = h.synthesize(ids, style, 1.0, None, 5)?;
        let mut adapter = HashMap::new();
        for (k, v) in h.base_adain.iter() {
            adapter.insert(k.clone(), (v * 0.5)?.to_device(adapter_dev)?);
        }
        let b = h.synthesize(ids, style, 1.0, Some(&adapter), 5)?;
        let c = h.synthesize(ids, style, 1.0, None, 5)?;
        assert_eq!(a.len(), c.len());
        let diff_ac: f32 = a
            .iter()
            .zip(&c)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        let diff_ab: f32 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        eprintln!(
            "serve on {:?} (adapter on {:?}): base vs restored max diff {diff_ac:.2e}, base vs adapted {diff_ab:.3}",
            h.device, adapter_dev
        );
        assert!(
            diff_ac < 1e-6,
            "base weights not restored after an adapted synth"
        );
        assert!(diff_ab > 1e-3, "adapter had no effect");
        let wk = h
            .base_adain
            .keys()
            .find(|k| k.ends_with(".weight"))
            .expect("an AdaIN weight");
        let bk = format!("{}.bias", wk.strip_suffix(".weight").expect("suffix"));
        let bad = HashMap::from([
            (wk.clone(), h.base_adain[wk].flatten_all()?),
            (bk.clone(), h.base_adain[&bk].clone()),
        ]);
        let err = h.synthesize(ids, style, 1.0, Some(&bad), 5).unwrap_err();
        assert!(err.to_string().contains("shape"), "{err:#}");
        let d = h.synthesize(ids, style, 1.0, None, 5)?;
        let diff_ad: f32 = a
            .iter()
            .zip(&d)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(
            a.len() == d.len() && diff_ad < 1e-6,
            "base weights not restored after a rejected adapter"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "loads the converted Kokoro weights on the GPU"]
    fn serve_handle_applies_cpu_adapter_on_cuda() -> Result<()> {
        let weights = std::env::var_os("KOKORO_SAFETENSORS")
            .or_else(|| std::env::var_os("KOKORO_CANDLE_WEIGHTS"))
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| "models/kokoro-v1_0.safetensors".into());
        let dev = Device::cuda_if_available(0)?;
        if !dev.is_cuda() || !weights.is_file() {
            eprintln!("SKIP: need a CUDA device and {}", weights.display());
            return Ok(());
        }
        let h = KokoroCandle::load(&weights, None, &dev)?;
        let ids: Vec<u32> = vec![0, 50, 83, 54, 156, 57, 135, 16, 65, 156, 87, 158, 54, 46, 0];
        let style: Vec<f32> = (0..256).map(|i| (i as f32 * 0.37).sin() * 0.3).collect();
        swap_and_restore(&h, &ids, &style, &Device::Cpu)?;
        swap_and_restore(&h, &ids, &style, &dev)
    }

    #[test]
    fn stage5b_two_encoder_adaptation_improves_heldout() -> Result<()> {
        let (Some((weights, reference)), Some(spk), Some(wsd), Some(wsr)) = (
            env_paths(),
            std::env::var_os("QWEN_SPEAKER_ENCODER"),
            std::env::var_os("WESPEAKER_CANDLE"),
            std::env::var_os("WESPEAKER_REF"),
        ) else {
            eprintln!("SKIP: set KOKORO_SAFETENSORS, KOKORO_REF, QWEN_SPEAKER_ENCODER, WESPEAKER_CANDLE, WESPEAKER_REF");
            return Ok(());
        };
        let dev = Device::Cpu;
        let mut m = KokoroFrontEnd::load(&weights, &dev)?;
        let enc = Encoders::load(Path::new(&spk), Path::new(&wsd), &dev)?;
        let r = candle_core::safetensors::load(&reference, &dev)?;
        let wr = candle_core::safetensors::load(&wsr, &dev)?;

        let wav16 = &wr["wav16"];
        let l16 = wav16.dim(0)?;
        let idx: Vec<f32> = (0..l16 * 3 / 2).map(|i| i as f32 * 2.0 / 3.0).collect();
        let i0: Vec<u32> = idx
            .iter()
            .map(|x| (x.floor() as usize).min(l16 - 1) as u32)
            .collect();
        let i1: Vec<u32> = idx
            .iter()
            .map(|x| ((x.floor() as usize) + 1).min(l16 - 1) as u32)
            .collect();
        let t: Vec<f32> = idx.iter().map(|x| x - x.floor()).collect();
        let a = wav16.index_select(&Tensor::new(i0, &dev)?, 0)?;
        let b = wav16.index_select(&Tensor::new(i1, &dev)?, 0)?;
        let wav24 = (&a + ((b - &a)? * Tensor::new(t, &dev)?)?)?;
        let targets = enc.targets(&[wav24])?;
        let ids: Vec<u32> = r["input_ids"]
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1()?;
        let style = r["ref_s"].detach();
        let cfg = TrainConfig {
            epochs: 2,
            patience: 3,
            lr: 5e-4,
            ..Default::default()
        };
        let out = adapt(
            &mut m,
            &enc,
            &targets,
            &style,
            std::slice::from_ref(&ids),
            std::slice::from_ref(&ids),
            &cfg,
            |rep| {
                eprintln!(
                    "stage5b epoch {} train {:.4} heldout wespeaker {:.4} qwen {:.4}",
                    rep.epoch, rep.train_loss, rep.heldout_wespeaker, rep.heldout_qwen
                )
            },
        )?;
        assert_eq!(out.adain.len(), 116);
        assert!(out.reports.len() == 2);
        let first = out.reports[0].heldout_wespeaker;
        let last = out.reports[1].heldout_wespeaker;
        eprintln!(
            "stage5b heldout wespeaker {first:.4} -> {last:.4}, stop {}",
            out.stop
        );
        assert!(
            last > first - 0.01,
            "held-out score fell: {first} -> {last}"
        );

        let mut fresh = KokoroFrontEnd::load(&weights, &dev)?;
        assert_eq!(apply_adain(&mut fresh, &out.adain)?, 58);
        let mut rng = Rng(3);
        let (audio, _) = fresh.synthesize_tokens(&ids, &style, 1.0, true, &mut rng)?;
        let c = cosine(&enc.wespeaker_embed(&audio)?, &targets.wespeaker)?.to_scalar::<f32>()?;
        eprintln!("stage5b applied weights: wespeaker cosine {c:.4}");
        assert!(
            (c - out.best_heldout).abs() < 0.02,
            "applied weights do not reproduce the held-out score: {c} vs {}",
            out.best_heldout
        );
        Ok(())
    }

    #[test]
    fn stage5_adain_gradients_move_the_speaker_loss() -> Result<()> {
        let Some((weights, reference)) = env_paths() else {
            eprintln!("SKIP: set KOKORO_SAFETENSORS and KOKORO_REF");
            return Ok(());
        };
        let Some(spk) = std::env::var_os("QWEN_SPEAKER_ENCODER") else {
            eprintln!("SKIP: set QWEN_SPEAKER_ENCODER");
            return Ok(());
        };
        let dev = Device::Cpu;
        let mut m = KokoroFrontEnd::load(&weights, &dev)?;
        let enc = load_speaker_encoder(Path::new(&spk), &dev)?;
        let mel = LogMel24k::new(&dev)?;
        let r = candle_core::safetensors::load(&reference, &dev)?;
        let ids: Vec<u32> = r["input_ids"]
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1()?;

        let target = enc.forward(&mel.forward(&r["audio_det"])?)?.detach();

        let style = (r["ref_s"].clone() * 0.5)?.detach();
        let vars = m.make_adain_trainable()?;
        assert_eq!(vars.len(), 116, "58 AdaIN fc layers");
        use candle_nn::Optimizer;
        let mut opt = candle_nn::AdamW::new(
            vars.clone(),
            candle_nn::ParamsAdamW {
                lr: 1e-3,
                ..Default::default()
            },
        )?;
        let mut losses = Vec::new();
        for step in 0..3 {
            let mut rng = Rng(7);
            let (audio, _) = m.synthesize_tokens(&ids, &style, 1.0, true, &mut rng)?;
            let e = enc.forward(&mel.forward(&audio)?)?;
            let loss = (1.0 - cosine(&e, &target)?)?;
            let grads = loss.backward()?;
            let gnorm: f32 = vars
                .iter()
                .filter_map(|v| grads.get(v))
                .map(|g| {
                    g.sqr()
                        .and_then(|x| x.sum_all())
                        .and_then(|x| x.to_scalar::<f32>())
                        .unwrap_or(0.0)
                })
                .sum::<f32>()
                .sqrt();
            opt.step(&grads)?;
            let l = loss.to_scalar::<f32>()?;
            eprintln!("stage5 step {step}: loss {l:.5} grad-norm {gnorm:.3e}");
            assert!(gnorm > 0.0, "no gradient reached the AdaIN projections");
            losses.push(l);
        }
        set_differentiable(false);
        assert!(losses[2] < losses[0], "loss did not decrease: {losses:?}");
        Ok(())
    }

    #[test]
    fn stage4_generator_matches_noise_free_torch_reference() -> Result<()> {
        let Some((weights, reference)) = env_paths() else {
            eprintln!("SKIP: set KOKORO_SAFETENSORS and KOKORO_REF");
            return Ok(());
        };
        let dev = Device::Cpu;
        let m = KokoroFrontEnd::load(&weights, &dev)?;
        let r = candle_core::safetensors::load(&reference, &dev)?;
        let (dec_s, _pro_s) = KokoroFrontEnd::split_style(&r["ref_s"])?;
        let mut rng = Rng(1);

        let f0v: Vec<f32> = r["F0_pred"].flatten_all()?.to_vec1()?;
        let src = m.generator.source.forward(&f0v, &dev, true, &mut rng)?;
        if let Some(ref_src) = r.get("har_source") {
            let e = assert_close("har_source", &src, ref_src, 2e-2);
            let sv: Vec<f32> = src.flatten_all()?.to_vec1()?;
            let rv: Vec<f32> = ref_src.flatten_all()?.to_vec1()?;
            let first_bad = sv.iter().zip(&rv).position(|(a, b)| (a - b).abs() > 1e-2);
            eprintln!("har_source: len {} vs {}, first sample off by >1e-2 at {:?}; sample[0..4] ours {:?} ref {:?}; result {:?}",
                sv.len(), rv.len(), first_bad, &sv[..4], &rv[..4], e.map(|x| format!("{x:.2e}")));
        }
        let har = m
            .generator
            .harmonic_spectrogram(&r["F0_pred"], true, &mut rng)?;

        let bins = 11;
        let (mag_o, ph_o) = (har.narrow(1, 0, bins)?, har.narrow(1, bins, bins)?);
        let (mag_r, ph_r) = (
            r["har"].narrow(1, 0, bins)?,
            r["har"].narrow(1, bins, bins)?,
        );
        let e0 = assert_close("har magnitude", &mag_o, &mag_r, 1e-2)?;
        let e0p = assert_close(
            "har re",
            &mag_o.mul(&ph_o.cos()?)?,
            &mag_r.mul(&ph_r.cos()?)?,
            1e-2,
        )?
        .max(assert_close(
            "har im",
            &mag_o.mul(&ph_o.sin()?)?,
            &mag_r.mul(&ph_r.sin()?)?,
            1e-2,
        )?);
        let audio = m
            .generator
            .forward_with_har(&r["dec_pre_gen"], &dec_s, &r["har"])?;
        let rel = (&audio - &r["audio_det"])?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?
            .sqrt()
            / r["audio_det"]
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?
                .sqrt()
                .max(1e-9);
        let e1 = assert_close("audio_det", &audio, &r["audio_det"], 2e-2)?;
        eprintln!("stage4 audio_det relative L2 error {rel:.3e}");
        assert!(rel < 5e-3, "audio relative L2 error {rel}");
        let noisy =
            m.generator
                .forward(&r["dec_pre_gen"], &dec_s, &r["F0_pred"], false, &mut rng)?;
        let ref_len = r["audio"].dim(0)?;
        eprintln!("stage4 max abs diffs: har mag {e0:.2e} phase {e0p:.2e} audio_det {e1:.2e}; noisy len {} vs ref {ref_len}", noisy.dim(0)?);
        assert_eq!(noisy.dim(0)?, ref_len);
        Ok(())
    }
}

#[cfg(test)]
mod wespeaker_shape_tests {
    use super::*;
    #[test]
    fn wespeaker_accepts_odd_frame_counts() -> Result<()> {
        let Some(dir) = std::env::var_os("WESPEAKER_CANDLE") else {
            eprintln!("SKIP");
            return Ok(());
        };
        let dev = Device::Cpu;
        let ws = WeSpeaker::load(Path::new(&dir), &dev)?;
        for frames in [487usize, 486, 301] {
            let feats = Tensor::randn(0f32, 1.0, (frames, 80), &dev)?;
            let e = ws
                .embed_feats(&feats)
                .map_err(|e| anyhow!("frames {frames}: {e:#}"))?;
            assert_eq!(e.dim(0)?, 256);
        }
        Ok(())
    }
}
