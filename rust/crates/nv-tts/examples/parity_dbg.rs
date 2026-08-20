use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_tts::talker::{
    CODEC_BOS_ID, CODEC_EOS_ID, CODEC_NOTHINK_ID, CODEC_PAD_ID, CODEC_THINK_BOS_ID,
    CODEC_THINK_EOS_ID,
};
use nv_tts::tokenizer::{SPECIAL_ID_TTS_PAD, SPECIAL_ID_TTS_TEXT_BOS, SPECIAL_ID_TTS_TEXT_EOD};
use nv_tts::{
    CodecDecoderConfig, Qwen3TtsCodecDecoder, Qwen3TtsTalker, Qwen3TtsTalkerConfig,
    Qwen3TtsTokenizer,
};
use nv_weights::WeightLoader;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .map(PathBuf::from)
        .context("pass model dir as arg1")?;
    let dev_arg = args.get(2).map(|s| s.as_str()).unwrap_or("cpu");
    let n_frames: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = match dev_arg {
        "cuda" => Device::new_cuda(0)?,
        _ => Device::Cpu,
    };

    let weights = WeightLoader::open_file(&dir.join("model.safetensors"), &device)?;
    let mut tcfg = Qwen3TtsTalkerConfig::from_hf_config_file(&dir.join("config.json"))?;
    if args.get(4).map(|s| s.as_str()) == Some("f32") {
        tcfg.dtype = DType::F32;
    }
    let mut talker = Qwen3TtsTalker::new(tcfg.clone(), &device)?;
    talker.load_weights(&weights)?;
    let mut ccfg = CodecDecoderConfig::from_hf_config_file(&dir.join("config.json"))?;
    ccfg.dtype = tcfg.dtype;
    let mut cp = Qwen3TtsCodecDecoder::new(ccfg, &device)?;
    cp.load_weights(&weights)?;
    let tokenizer = Qwen3TtsTokenizer::from_dir(&dir)?;

    let text = "The quick brown fox jumps over the lazy dog near the quiet river bank.";
    let text_ids = tokenizer.encode_text(text)?;
    eprintln!("text_ids: {:?}", text_ids);

    let h = tcfg.hidden_size;
    let spk = *tcfg.spk_id.get("ryan").context("ryan spk id")?;

    let proj =
        |ids: &[u32]| -> Result<Tensor> { talker.project_text(&talker.embed_text_ids(ids)?) };
    let role = proj(&[151644, 77091, 198])?;
    let specials = proj(&[
        SPECIAL_ID_TTS_PAD,
        SPECIAL_ID_TTS_TEXT_BOS,
        SPECIAL_ID_TTS_TEXT_EOD,
    ])?;
    let tts_pad = specials.narrow(1, 0, 1)?;
    let tts_bos = specials.narrow(1, 1, 1)?;
    let tts_eos = specials.narrow(1, 2, 1)?;
    let prefix =
        talker.embed_codec_ids(&[CODEC_NOTHINK_ID, CODEC_THINK_BOS_ID, CODEC_THINK_EOS_ID])?;
    let spk_emb = talker.embed_codec_ids(&[spk])?;
    let pad_bos = talker.embed_codec_ids(&[CODEC_PAD_ID, CODEC_BOS_ID])?;
    let codec_chan = Tensor::cat(&[&prefix, &spk_emb, &pad_bos], 1)?;
    let l = codec_chan.dims()[1];
    let pads = tts_pad.expand((1usize, l - 2, h))?.contiguous()?;
    let text_over = Tensor::cat(&[&pads, &tts_bos], 1)?;
    let summed = text_over.add(&codec_chan.narrow(1, 0, l - 1)?)?;
    let first_text = proj(&text_ids[..1])?.add(&codec_chan.narrow(1, l - 1, 1)?)?;
    let prefill = Tensor::cat(&[&role, &summed, &first_text], 1)?;
    let trailing = Tensor::cat(&[&proj(&text_ids[1..])?, &tts_eos], 1)?;
    let trailing_len = trailing.dims()[1];

    let pn: Vec<f32> = prefill
        .to_dtype(DType::F32)?
        .sqr()?
        .sum(2)?
        .sqrt()?
        .flatten_all()?
        .to_vec1()?;
    eprintln!("prefill shape: {:?}", prefill.dims());
    eprintln!(
        "prefill pos-norms: {:?}",
        pn.iter()
            .map(|v| (v * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );

    if std::env::var("NV_DBG_LAYERNORMS").is_ok() {
        let (norms, logits) = talker.debug_layer_norms(&prefill)?;
        eprintln!(
            "layer norms: {:?}",
            norms
                .iter()
                .map(|v| (v * 100.0).round() / 100.0)
                .collect::<Vec<_>>()
        );
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        let top: Vec<(usize, f32)> = idx
            .iter()
            .take(10)
            .map(|&i| (i, (logits[i] * 1e4).round() / 1e4))
            .collect();
        eprintln!("uncached top10: {:?}", top);
    }
    let mut cache = talker.new_kv_cache(prefill.dims()[1] + n_frames + 2)?;
    let mut cp_cache = cp.new_kv_cache()?;
    let mut next = prefill;
    let mut bases: Vec<u32> = Vec::new();
    for t in 0..n_frames {
        let (logits, hidden) = talker.step_cached_embeds(&next, &mut cache)?;
        let host: Vec<f32> = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        if t == 0 {
            let mut idx: Vec<usize> = (0..host.len()).collect();
            idx.sort_by(|&a, &b| host[b].partial_cmp(&host[a]).unwrap());
            let top: Vec<(usize, f32)> = idx
                .iter()
                .take(10)
                .map(|&i| (i, (host[i] * 1e4).round() / 1e4))
                .collect();
            eprintln!("prefill top10: {:?}", top);
        }
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in host.iter().enumerate().take(2048) {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        if t >= 2 && host[CODEC_EOS_ID as usize] > best_v {
            best = CODEC_EOS_ID as usize;
        }
        let base = best as u32;
        bases.push(base);
        if base == CODEC_EOS_ID {
            break;
        }
        let base_emb = talker.embed_codec_ids(&[base])?;
        let hidden3 = hidden.reshape((1usize, 1usize, h))?;
        let (extras, extras_sum) = cp.predict_frame(&hidden3, &base_emb, &mut cp_cache, None)?;
        if t == 0 {
            eprintln!("frame0 extras: {:?}", extras);
        }
        let step_text = if t < trailing_len {
            trailing.narrow(1, t, 1)?
        } else {
            tts_pad.clone()
        };
        next = base_emb
            .add(&extras_sum)?
            .add(&step_text.to_dtype(base_emb.dtype())?)?;
    }
    eprintln!("greedy bases: {:?}", bases);
    Ok(())
}
