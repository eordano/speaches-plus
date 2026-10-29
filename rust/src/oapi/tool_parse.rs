use crate::oapi::chat::{FunctionCall, ToolCall};

pub const HERMES_CALL_OPEN: &str = "<tool_call>";
pub const HERMES_CALL_CLOSE: &str = "</tool_call>";

pub const HERMES_WIRE_TOKENS: [&str; 2] = [HERMES_CALL_OPEN, HERMES_CALL_CLOSE];

const OPEN: &str = HERMES_CALL_OPEN;
const CLOSE: &str = HERMES_CALL_CLOSE;
const FN_OPEN: &str = "<function=";
const FN_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";
const ARG_KEY_OPEN: &str = "<arg_key>";
const ARG_KEY_CLOSE: &str = "</arg_key>";
const ARG_VALUE_OPEN: &str = "<arg_value>";
const ARG_VALUE_CLOSE: &str = "</arg_value>";

pub struct ParsedOutput {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

pub fn new_call_id() -> String {
    format!("call_{}", uuid::Uuid::new_v4().simple())
}

pub fn parse_tool_calls(text: &str, force_name: Option<&str>) -> ParsedOutput {
    if let Some(name) = force_name {
        let Some(args) = extract_json_object(text) else {
            let t = text.trim();
            return ParsedOutput {
                content: (!t.is_empty()).then(|| t.to_string()),
                tool_calls: Vec::new(),
            };
        };
        return ParsedOutput {
            content: None,
            tool_calls: vec![ToolCall {
                index: None,
                id: new_call_id(),
                kind: "function".into(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: args,
                },
            }],
        };
    }

    let mut calls = Vec::new();
    let mut leading = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        leading.push_str(&rest[..start]);
        let after = &rest[start + OPEN.len()..];
        let Some(end) = after.find(CLOSE) else { break };
        if let Some(tc) = parse_one_call(after[..end].trim()) {
            calls.push(tc);
        }
        rest = &after[end + CLOSE.len()..];
    }
    leading.push_str(rest);

    if calls.is_empty() && leading.contains(FN_OPEN) {
        let (stripped, xml_calls) = take_xml_calls(&leading);
        if !xml_calls.is_empty() {
            calls = xml_calls;
            leading = stripped;
        }
    }

    let t = leading.trim();
    let content = if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    };
    ParsedOutput {
        content,
        tool_calls: calls,
    }
}

fn parse_one_call(body: &str) -> Option<ToolCall> {
    match serde_json::from_str::<serde_json::Value>(body) {
        Ok(v) => json_call(&v),
        Err(_) => parse_xml_call(body)
            .map(|(tc, _)| tc)
            .or_else(|| parse_arg_tag_call(body)),
    }
}

fn parse_arg_tag_call(body: &str) -> Option<ToolCall> {
    let (name, mut rest) = match body.find(ARG_KEY_OPEN) {
        Some(i) => (body[..i].trim(), &body[i..]),
        None => (body.trim(), ""),
    };
    if name.is_empty() || name.contains(|c: char| c == '<' || c.is_whitespace()) {
        return None;
    }
    let mut args = serde_json::Map::new();
    while let Some(k) = rest.find(ARG_KEY_OPEN) {
        let after_key_open = &rest[k + ARG_KEY_OPEN.len()..];
        let key_end = after_key_open.find(ARG_KEY_CLOSE)?;
        let key = after_key_open[..key_end].trim();
        let after_key = &after_key_open[key_end + ARG_KEY_CLOSE.len()..];
        let v = after_key.find(ARG_VALUE_OPEN)?;
        let after_value_open = &after_key[v + ARG_VALUE_OPEN.len()..];
        let (raw, consumed) = match after_value_open.find(ARG_VALUE_CLOSE) {
            Some(i) => (&after_value_open[..i], i + ARG_VALUE_CLOSE.len()),
            None => (after_value_open, after_value_open.len()),
        };
        if !key.is_empty() {
            args.insert(key.to_string(), xml_param_value(raw.trim()));
        }
        rest = &after_value_open[consumed..];
    }
    let arguments = serde_json::to_string(&serde_json::Value::Object(args)).ok()?;
    Some(ToolCall {
        index: None,
        id: new_call_id(),
        kind: "function".into(),
        function: FunctionCall {
            name: name.to_string(),
            arguments,
        },
    })
}

fn json_call(v: &serde_json::Value) -> Option<ToolCall> {
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = match v.get("arguments") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).ok()?,
        None => "{}".to_string(),
    };
    Some(ToolCall {
        index: None,
        id: new_call_id(),
        kind: "function".into(),
        function: FunctionCall { name, arguments },
    })
}

fn take_xml_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut leading = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(FN_OPEN) {
        let Some((tc, used)) = parse_xml_call(&rest[start..]) else {
            break;
        };
        leading.push_str(&rest[..start]);
        calls.push(tc);
        rest = &rest[start + used..];
    }
    leading.push_str(rest);
    (leading, calls)
}

fn parse_xml_call(text: &str) -> Option<(ToolCall, usize)> {
    let start = text.find(FN_OPEN)?;
    let after = &text[start + FN_OPEN.len()..];
    let gt = after.find('>')?;
    let name = after[..gt].trim().trim_end_matches('/').trim();
    if name.is_empty() || name.contains('<') {
        return None;
    }
    let body_start = start + FN_OPEN.len() + gt + 1;
    let (body, used) = match text[body_start..].find(FN_CLOSE) {
        Some(i) => (
            &text[body_start..body_start + i],
            body_start + i + FN_CLOSE.len(),
        ),
        None => (&text[body_start..], text.len()),
    };
    let mut args = serde_json::Map::new();
    let mut rest = body;
    while let Some(p) = rest.find(PARAM_OPEN) {
        let after = &rest[p + PARAM_OPEN.len()..];
        let Some(gt) = after.find('>') else { break };
        let key = after[..gt].trim();
        let val_start = &after[gt + 1..];
        let (raw, consumed) = match val_start.find(PARAM_CLOSE) {
            Some(i) => (&val_start[..i], gt + 1 + i + PARAM_CLOSE.len()),
            None => (val_start, after.len()),
        };
        if !key.is_empty() {
            args.insert(key.to_string(), xml_param_value(raw.trim()));
        }
        rest = &after[consumed..];
    }
    let arguments = serde_json::to_string(&serde_json::Value::Object(args)).ok()?;
    Some((
        ToolCall {
            index: None,
            id: new_call_id(),
            kind: "function".into(),
            function: FunctionCall {
                name: name.to_string(),
                arguments,
            },
        },
        used,
    ))
}

#[allow(clippy::cmp_owned)]
fn xml_param_value(raw: &str) -> serde_json::Value {
    let structural = raw.starts_with('{') || raw.starts_with('[');
    let scalar = matches!(raw, "true" | "false" | "null");
    if structural || scalar {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
            return v;
        }
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
        if v.is_number() && v.to_string() == raw {
            return v;
        }
    }
    serde_json::Value::String(raw.to_string())
}

fn extract_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    let cand = &text[start..=end];
    serde_json::from_str::<serde_json::Value>(cand).ok()?;
    Some(cand.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_blocks_forced_objects_and_plain_text_parse_as_expected() {
        let p = parse_tool_calls(
            r#"sure <tool_call>{"name":"get","arguments":{"x":1}}</tool_call>"#,
            None,
        );
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].function.name, "get");
        assert_eq!(p.tool_calls[0].function.arguments, r#"{"x":1}"#);
        assert_eq!(p.content.as_deref(), Some("sure"));
        {
            let p = parse_tool_calls(r#"{"x":1}"#, Some("f"));
            assert_eq!(p.tool_calls.len(), 1);
            assert_eq!(p.tool_calls[0].function.name, "f");
            assert_eq!(p.tool_calls[0].function.arguments, r#"{"x":1}"#);
            assert!(p.content.is_none());
        }
        {
            let p = parse_tool_calls(r#"{"location": "Lis"#, Some("f"));
            assert!(
                p.tool_calls.is_empty(),
                "inventing {{}} arguments hands the caller a call its schema rejects"
            );
            assert_eq!(p.content.as_deref(), Some(r#"{"location": "Lis"#));
        }
        {
            let p = parse_tool_calls("just a normal answer", None);
            assert!(p.tool_calls.is_empty());
            assert_eq!(p.content.as_deref(), Some("just a normal answer"));
        }
    }

    #[test]
    fn qwen_xml_dialect_parses_with_and_without_the_wrapper_and_keeps_param_shapes() {
        let p = parse_tool_calls(
            "<tool_call>\n<function=get_weather>\n<parameter=city>Oslo</parameter>\n</function>\n</tool_call>",
            None,
        );
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].function.name, "get_weather");
        assert_eq!(p.tool_calls[0].function.arguments, r#"{"city":"Oslo"}"#);
        assert!(p.content.is_none());
        {
            let p = parse_tool_calls(
                "sure\n<function=get_weather>\n<parameter=city>Oslo</parameter>\n<parameter=days>3</parameter>\n<parameter=metric>true</parameter>\n</function>",
                None,
            );
            assert_eq!(p.tool_calls.len(), 1);
            assert_eq!(p.tool_calls[0].function.name, "get_weather");
            assert_eq!(
                p.tool_calls[0].function.arguments,
                r#"{"city":"Oslo","days":3,"metric":true}"#
            );
            assert_eq!(p.content.as_deref(), Some("sure"));
        }
        {
            let p = parse_tool_calls(
                "<function=a><parameter=x>1</parameter></function><function=b><parameter=y>z</parameter></function>",
                None,
            );
            assert_eq!(p.tool_calls.len(), 2);
            assert_eq!(p.tool_calls[0].function.name, "a");
            assert_eq!(p.tool_calls[1].function.name, "b");
            assert_eq!(p.tool_calls[1].function.arguments, r#"{"y":"z"}"#);
        }
        {
            assert_eq!(xml_param_value("007"), serde_json::json!("007"));
            assert_eq!(xml_param_value("3"), serde_json::json!(3));
            assert_eq!(
                xml_param_value("123 Main St"),
                serde_json::json!("123 Main St")
            );
            assert_eq!(xml_param_value(r#"{"a":1}"#), serde_json::json!({"a":1}));
            assert_eq!(xml_param_value("null"), serde_json::Value::Null);
        }
        {
            let p = parse_tool_calls("<parameter=x>1</parameter>", None);
            assert!(p.tool_calls.is_empty());
            assert_eq!(p.content.as_deref(), Some("<parameter=x>1</parameter>"));
        }
    }

    #[test]
    fn laguna_arg_tag_dialect_parses_raw_typed_bare_and_multi_calls_but_not_prose() {
        let p = parse_tool_calls(
            "<tool_call>read<arg_key>path</arg_key><arg_value>src/main.rs</arg_value></tool_call>",
            None,
        );
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].function.name, "read");
        assert_eq!(
            p.tool_calls[0].function.arguments,
            r#"{"path":"src/main.rs"}"#
        );
        assert!(p.content.is_none());
        {
            let p = parse_tool_calls(
                "<tool_call>search<arg_key>limit</arg_key><arg_value>3</arg_value><arg_key>filters</arg_key><arg_value>{\"lang\":\"rs\"}</arg_value><arg_key>fuzzy</arg_key><arg_value>true</arg_value></tool_call>",
                None,
            );
            assert_eq!(p.tool_calls.len(), 1);
            assert_eq!(p.tool_calls[0].function.name, "search");
            assert_eq!(
                p.tool_calls[0].function.arguments,
                r#"{"filters":{"lang":"rs"},"fuzzy":true,"limit":3}"#
            );
        }
        {
            let p = parse_tool_calls("<tool_call>list_files</tool_call>", None);
            assert_eq!(p.tool_calls.len(), 1);
            assert_eq!(p.tool_calls[0].function.name, "list_files");
            assert_eq!(p.tool_calls[0].function.arguments, "{}");
        }
        {
            let p = parse_tool_calls(
                "looking\n<tool_call>read<arg_key>path</arg_key><arg_value>a.rs</arg_value></tool_call><tool_call>read<arg_key>path</arg_key><arg_value>b.rs</arg_value></tool_call>",
                None,
            );
            assert_eq!(p.tool_calls.len(), 2);
            assert_eq!(p.tool_calls[0].function.arguments, r#"{"path":"a.rs"}"#);
            assert_eq!(p.tool_calls[1].function.arguments, r#"{"path":"b.rs"}"#);
            assert_eq!(p.content.as_deref(), Some("looking"));
        }
        {
            let p = parse_tool_calls("<tool_call>not a call</tool_call>", None);
            assert!(p.tool_calls.is_empty());
        }
    }

}
