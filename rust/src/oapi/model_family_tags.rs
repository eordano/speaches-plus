#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen3_5Tag {
    Moe,
    Dense,
    Neither,
}

pub fn architectures_and_model_type_tags(v: &serde_json::Value) -> Vec<String> {
    let mut tags: Vec<String> = Vec::new();
    if let Some(arr) = v.get("architectures").and_then(|x| x.as_array()) {
        tags.extend(
            arr.iter()
                .filter_map(|a| a.as_str())
                .map(|s| s.to_ascii_lowercase()),
        );
    }
    if let Some(mt) = v.get("model_type").and_then(|x| x.as_str()) {
        tags.push(mt.to_ascii_lowercase());
    }
    tags
}

pub fn any_tag_starts_with(tags: &[String], pats: &[&str]) -> bool {
    pats.iter().any(|p| tags.iter().any(|t| t.starts_with(p)))
}

pub fn is_laguna(tags: &[String]) -> bool {
    any_tag_starts_with(tags, &["laguna"])
}

pub fn is_dots(tags: &[String]) -> bool {
    any_tag_starts_with(tags, &["dots_ocr", "dotsocr"])
}

pub fn qwen3_5_tag(tags: &[String]) -> Qwen3_5Tag {
    if any_tag_starts_with(
        tags,
        &["qwen3_5moe", "qwen3_5_moe", "qwen3.5moe", "qwen3.5_moe"],
    ) {
        return Qwen3_5Tag::Moe;
    }
    if any_tag_starts_with(tags, &["qwen3_5", "qwen3.5"]) {
        return Qwen3_5Tag::Dense;
    }
    Qwen3_5Tag::Neither
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_moe_arm_wins_over_the_dense_prefix_it_also_matches() {
        for cfg in [
            r#"{"architectures":["Qwen3_5MoeForCausalLM"],"model_type":"qwen3_5_moe_text"}"#,
            r#"{"model_type":"qwen3.5_moe"}"#,
        ] {
            let v: serde_json::Value = serde_json::from_str(cfg).unwrap();
            assert_eq!(
                qwen3_5_tag(&architectures_and_model_type_tags(&v)),
                Qwen3_5Tag::Moe,
                "{cfg}: every moe tag also starts with the dense prefix, so the moe check \
                 must run first"
            );
        }
        let v: serde_json::Value = serde_json::from_str(r#"{"model_type":"qwen3_5"}"#).unwrap();
        assert_eq!(
            qwen3_5_tag(&architectures_and_model_type_tags(&v)),
            Qwen3_5Tag::Dense
        );
        let v: serde_json::Value = serde_json::from_str(r#"{"model_type":"qwen3"}"#).unwrap();
        assert_eq!(
            qwen3_5_tag(&architectures_and_model_type_tags(&v)),
            Qwen3_5Tag::Neither,
            "qwen3 (no _5) must not match the qwen3_5 prefix"
        );
    }

    #[test]
    fn laguna_matches_by_architecture_or_model_type_and_nothing_else_does() {
        for cfg in [
            r#"{"architectures":["LagunaForCausalLM"],"model_type":"laguna"}"#,
            r#"{"model_type":"laguna"}"#,
            r#"{"architectures":["LagunaForCausalLM"]}"#,
        ] {
            let v: serde_json::Value = serde_json::from_str(cfg).unwrap();
            assert!(is_laguna(&architectures_and_model_type_tags(&v)), "{cfg}");
        }
        for cfg in [
            r#"{"model_type":"qwen3_5"}"#,
            r#"{"architectures":["Gemma4ForConditionalGeneration"]}"#,
            r#"{}"#,
        ] {
            let v: serde_json::Value = serde_json::from_str(cfg).unwrap();
            assert!(!is_laguna(&architectures_and_model_type_tags(&v)), "{cfg}");
        }
    }
}
