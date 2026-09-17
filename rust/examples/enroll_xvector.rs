use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Result};
use candle_core::Device;
use nv_tts::SpeakerEncoder;
use speaches_plus::audio::decode_any_to_16k_mono;
use speaches_plus::oapi::voice_profiles::extract_embedding;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        bail!("usage: enroll_xvector <base-checkpoint-dir> <reference-audio> <name> <out.json>");
    }
    let dir = PathBuf::from(&args[1]);
    let bytes = std::fs::read(&args[2])?;
    let name = args[3].clone();
    let out = PathBuf::from(&args[4]);
    let samples_16k = decode_any_to_16k_mono(&bytes, None)?;
    let enc = Arc::new(SpeakerEncoder::from_qwen3_checkpoint(&dir, &Device::Cpu)?);
    let embedding = extract_embedding(&enc, &samples_16k)?;
    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    let profile = nv_tts::VoiceProfile {
        schema_version: 2,
        name,
        embedding,
        design_params: None,
    };
    std::fs::write(&out, serde_json::to_vec(&profile)?)?;
    eprintln!(
        "wrote {} ({} dims from {:.1} s of audio, l2 norm {norm:.3})",
        out.display(),
        profile.embedding.len(),
        samples_16k.len() as f32 / 16_000.0
    );
    Ok(())
}
