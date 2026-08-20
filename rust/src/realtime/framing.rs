use anyhow::{Context, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::Serialize;
use uuid::Uuid;

const MAX_FRAGMENT_SIZE: usize = crate::defaults::wire::DATA_CHANNEL_FRAGMENT_MAX;

#[derive(Serialize)]
#[serde(tag = "type", rename = "full_message")]
struct FullMessageEvent<'a> {
    id: &'a str,
    data: &'a str,
}

#[derive(Serialize)]
#[serde(tag = "type", rename = "partial_message")]
struct PartialMessageEvent<'a> {
    id: &'a str,
    fragment_index: usize,
    total_fragments: usize,
    data: &'a str,
}

pub fn frame_event<T: Serialize>(event: &T) -> Result<Vec<String>> {
    let payload = serde_json::to_vec(event).context("serialize event")?;
    let encoded = STANDARD.encode(&payload);

    let id = Uuid::new_v4().to_string();
    if encoded.len() <= MAX_FRAGMENT_SIZE {
        let env = FullMessageEvent {
            id: &id,
            data: &encoded,
        };
        return Ok(vec![serde_json::to_string(&env)?]);
    }

    let total = encoded.len().div_ceil(MAX_FRAGMENT_SIZE);
    let mut frames = Vec::with_capacity(total);
    for (i, chunk) in encoded.as_bytes().chunks(MAX_FRAGMENT_SIZE).enumerate() {
        let chunk = std::str::from_utf8(chunk).context("base64 always ASCII")?;
        let env = PartialMessageEvent {
            id: &id,
            fragment_index: i,
            total_fragments: total,
            data: chunk,
        };
        frames.push(serde_json::to_string(&env)?);
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn small_event_uses_full_message() -> Result<()> {
        let frames = frame_event(&json!({"type": "session.created", "id": "x"}))?;
        assert_eq!(frames.len(), 1);
        assert!(frames[0].contains("\"type\":\"full_message\""));
        assert!(frames[0].contains("\"id\":"));
        Ok(())
    }

    #[test]
    fn large_event_splits_into_partials() -> Result<()> {
        let big_text: String = "a".repeat(2000);
        let frames = frame_event(&json!({"type": "x", "blob": big_text}))?;
        assert!(frames.len() >= 2);
        for frame in &frames {
            assert!(frame.contains("\"type\":\"partial_message\""));
            assert!(frame.contains("\"total_fragments\""));
        }
        Ok(())
    }
}
