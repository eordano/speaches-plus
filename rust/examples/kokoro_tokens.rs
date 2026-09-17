use anyhow::{Context, Result};

fn main() -> Result<()> {
    let espeak = std::env::var("ESPEAK_DATA_PATH").ok();
    speaches_plus::tts::phonemize::init(espeak.as_deref()).context("init phonemizer")?;
    let mut out = serde_json::Map::new();
    let extra: Vec<String> = std::env::args().skip(1).collect();
    let extra_refs: Vec<&str> = extra.iter().map(String::as_str).collect();
    let sets: [(&str, &[&str]); 3] = [
        ("train", &speaches_plus::tts::fit::TRAIN_SENTENCES),
        ("heldout", &speaches_plus::tts::fit::DEFAULT_SENTENCES),
        ("extra", &extra_refs),
    ];
    for (key, sentences) in sets {
        let mut rows = Vec::new();
        for s in sentences.iter() {
            let raw = speaches_plus::tts::phonemize::phonemize(s, "en-us")
                .with_context(|| format!("phonemize {s:?}"))?;
            let cleaned = speaches_plus::tts::vocab::clean_phonemes(&raw);
            let tokens = speaches_plus::tts::vocab::tokenize(&cleaned);
            rows.push(serde_json::json!({"text": s, "phonemes": cleaned, "tokens": tokens}));
        }
        out.insert(key.to_string(), serde_json::Value::Array(rows));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::Value::Object(out))?
    );
    Ok(())
}
