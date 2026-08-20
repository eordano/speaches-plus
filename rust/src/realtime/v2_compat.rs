use serde_json::{Map, Value};

const KNOWN_V2_NOOP_EVENTS: &[&str] = &[
    "output_audio_buffer.clear",
    "output_audio_buffer.append",
    "input_audio_buffer.dtmf.received",
    "transcription_session.update",
    "response.cancel_audio",
];

pub(super) fn normalize_session_object(session_obj: &Value) -> Value {
    let Some(obj) = session_obj.as_object() else {
        return session_obj.clone();
    };
    let mut out: Map<String, Value> = obj.clone();

    if let Some(audio) = obj.get("audio").and_then(|v| v.as_object()) {
        if let Some(ai) = audio.get("input").and_then(|v| v.as_object()) {
            if let Some(v) = ai.get("format") {
                out.entry("input_audio_format").or_insert_with(|| v.clone());
            }
            if let Some(v) = ai.get("transcription") {
                out.entry("input_audio_transcription")
                    .or_insert_with(|| v.clone());
            }
            if let Some(v) = ai.get("turn_detection") {
                out.entry("turn_detection").or_insert_with(|| v.clone());
            }
        }
        if let Some(ao) = audio.get("output").and_then(|v| v.as_object()) {
            if let Some(v) = ao.get("format") {
                out.entry("output_audio_format")
                    .or_insert_with(|| v.clone());
            }
            if let Some(v) = ao.get("voice") {
                out.entry("voice").or_insert_with(|| v.clone());
            }
        }
    }

    if obj.contains_key("output_modalities") && !obj.contains_key("modalities") {
        if let Some(v) = obj.get("output_modalities") {
            out.insert("modalities".into(), v.clone());
        }
    }

    Value::Object(out)
}

pub(super) fn enrich_session_view(view: &mut Value) {
    let Some(obj) = view.as_object_mut() else {
        return;
    };
    obj.entry("type")
        .or_insert_with(|| Value::String("realtime".into()));

    let input_audio_format = obj.get("input_audio_format").cloned();
    let input_audio_transcription = obj.get("input_audio_transcription").cloned();
    let turn_detection = obj.get("turn_detection").cloned();
    let output_audio_format = obj.get("output_audio_format").cloned();
    let voice = obj.get("voice").cloned();
    let modalities = obj.get("modalities").cloned();

    let audio_entry = obj
        .entry("audio")
        .or_insert_with(|| Value::Object(Map::new()));
    if let Some(audio_obj) = audio_entry.as_object_mut() {
        let inp = audio_obj
            .entry("input")
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(inp_obj) = inp.as_object_mut() {
            if let Some(v) = input_audio_format {
                inp_obj.entry("format").or_insert(v);
            }
            if let Some(v) = input_audio_transcription {
                inp_obj.entry("transcription").or_insert(v);
            }
            if let Some(v) = turn_detection {
                inp_obj.entry("turn_detection").or_insert(v);
            }
        }
        let outp = audio_obj
            .entry("output")
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(outp_obj) = outp.as_object_mut() {
            if let Some(v) = output_audio_format {
                outp_obj.entry("format").or_insert(v);
            }
            if let Some(v) = voice {
                outp_obj.entry("voice").or_insert(v);
            }
        }
    }

    if let Some(m) = modalities {
        obj.entry("output_modalities").or_insert(m);
    }
}

pub(super) fn is_known_v2_noop_event(event_type: &str) -> bool {
    KNOWN_V2_NOOP_EVENTS.contains(&event_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalize_nested_to_flat() {
        let nested = json!({
            "type": "realtime",
            "audio": {
                "input":  {"format": "pcm16", "turn_detection": {"type": "server_vad"}},
                "output": {"format": "g711_ulaw", "voice": "alloy"},
            },
            "output_modalities": ["audio"],
        });
        let flat = normalize_session_object(&nested);
        assert_eq!(flat["input_audio_format"], "pcm16");
        assert_eq!(flat["output_audio_format"], "g711_ulaw");
        assert_eq!(flat["voice"], "alloy");
        assert_eq!(flat["turn_detection"]["type"], "server_vad");
        assert_eq!(flat["modalities"], json!(["audio"]));
    }

    #[test]
    fn normalize_is_idempotent_on_flat() {
        let flat = json!({"input_audio_format": "pcm16"});
        let again = normalize_session_object(&flat);
        assert_eq!(again, flat);
    }

    #[test]
    fn enrich_adds_nested_shape() {
        let mut v = json!({
            "id": "sess_x",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "voice": "alloy",
            "modalities": ["audio", "text"],
        });
        enrich_session_view(&mut v);
        assert_eq!(v["type"], "realtime");
        assert_eq!(v["audio"]["input"]["format"], "pcm16");
        assert_eq!(v["audio"]["output"]["format"], "pcm16");
        assert_eq!(v["audio"]["output"]["voice"], "alloy");
        assert_eq!(v["output_modalities"], json!(["audio", "text"]));
    }
}
