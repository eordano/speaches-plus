use crate::pattern::{json_string_pattern, STRING_CHAR};
use crate::ranges::{int_range, number_range, Bound};
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{Map, Value};

pub(crate) const ANY_JSON_VALUE: &str = r"\x01";
pub(crate) const ANY_JSON_OBJECT: &str = r"\x02";
pub(crate) const WS: &str = r"[ \t\n]{0,20}";
const INTEGER: &str = r"-?(0|[1-9][0-9]{0,19})";
const NUMBER: &str = r"-?(0|[1-9][0-9]{0,19})(\.[0-9]{1,17})?([eE][-+]?[0-9]{1,4})?";

pub const MAX_REF_DEPTH_16_BOUNDS_NESTED_DEFINITIONS: usize = 16;
pub const MAX_COUNTED_REPEAT_256_KEEPS_MAXITEMS_AND_MAXLENGTH_DFAS_SMALL: u64 = 256;

const ANNOTATIONS: &[&str] = &[
    "title",
    "description",
    "default",
    "examples",
    "example",
    "$schema",
    "$id",
    "$comment",
    "$anchor",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$defs",
    "definitions",
    "contentMediaType",
    "contentEncoding",
    "discriminator",
    "xml",
    "externalDocs",
    "strict",
];

const SUPPORTED: &[&str] = &[
    "type",
    "enum",
    "const",
    "anyOf",
    "oneOf",
    "allOf",
    "$ref",
    "nullable",
    "properties",
    "required",
    "additionalProperties",
    "unevaluatedProperties",
    "items",
    "prefixItems",
    "additionalItems",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
];

const OBJECT_KEYS: &[&str] = &["properties", "required", "additionalProperties", "unevaluatedProperties"];
const ARRAY_KEYS: &[&str] = &["items", "prefixItems", "additionalItems", "minItems", "maxItems", "uniqueItems"];
const STRING_KEYS: &[&str] = &["minLength", "maxLength", "pattern", "format"];
const NUMBER_KEYS: &[&str] = &["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"];

fn regex_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        if "\\^$.|?*+()[]{}".contains(c) {
            o.push('\\');
        }
        o.push(c);
    }
    o
}

fn literal(v: &Value) -> String {
    regex_escape(&serde_json::to_string(v).unwrap_or_default())
}

fn format_regex(format: &str) -> Option<&'static str> {
    Some(match format {
        "date" => r"[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])",
        "time" => r"([01][0-9]|2[0-3]):[0-5][0-9]:([0-5][0-9]|60)(\.[0-9]{1,9})?(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])",
        "date-time" => r"[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])T([01][0-9]|2[0-3]):[0-5][0-9]:([0-5][0-9]|60)(\.[0-9]{1,9})?(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])",
        "uuid" => r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        "email" => r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(\.[A-Za-z0-9-]{1,63}){1,8}",
        "ipv4" => r"((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])",
        "hostname" => r"[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*",
        "uri" => r"[A-Za-z][A-Za-z0-9+.-]*:[^ \x22\x5c\x00-\x1f]{1,2048}",
        _ => return None,
    })
}

pub(crate) struct Compiler<'a> {
    root: &'a Value,
    refs: Vec<String>,
}

fn count(m: &Map<String, Value>, key: &str, at: &str) -> Result<Option<u64>> {
    match m.get(key) {
        None => Ok(None),
        Some(v) => v
            .as_u64()
            .or_else(|| v.as_f64().filter(|f| *f >= 0.0 && f.fract() == 0.0).map(|f| f as u64))
            .map(Some)
            .ok_or_else(|| anyhow!("{at}: {key} must be a non-negative integer, got {v}")),
    }
}

fn repeat(min: u64, max: Option<u64>, at: &str, what: &str) -> Result<String> {
    let cap = MAX_COUNTED_REPEAT_256_KEEPS_MAXITEMS_AND_MAXLENGTH_DFAS_SMALL;
    if let Some(max) = max {
        if max < min {
            bail!("{at}: min{what} {min} exceeds max{what} {max}");
        }
    }
    if min > cap || max.is_some_and(|m| m > cap) {
        bail!(
            "{at}: min{what}/max{what} above {cap} are not supported by guided decoding (the \
             counted repetition would not fit the grammar automaton); drop the bound or lower it"
        );
    }
    Ok(match max {
        Some(max) => format!("{{{min},{max}}}"),
        None => format!("{{{min},}}"),
    })
}

impl<'a> Compiler<'a> {
    pub(crate) fn new(root: &'a Value) -> Self {
        Self { root, refs: Vec::new() }
    }

    fn resolve(&self, r: &str, at: &str) -> Result<&'a Value> {
        let ptr = r
            .strip_prefix('#')
            .ok_or_else(|| anyhow!("{at}: only local $ref (\"#/...\") is supported, got {r:?}"))?;
        let mut node = self.root;
        if ptr.is_empty() {
            return Ok(node);
        }
        for raw in ptr.strip_prefix('/').unwrap_or(ptr).split('/') {
            let seg = raw.replace("~1", "/").replace("~0", "~");
            node = match node {
                Value::Object(m) => m.get(&seg),
                Value::Array(a) => seg.parse::<usize>().ok().and_then(|i| a.get(i)),
                _ => None,
            }
            .ok_or_else(|| anyhow!("{at}: $ref {r:?} does not resolve inside the schema"))?;
        }
        Ok(node)
    }

    pub(crate) fn node(&mut self, v: &'a Value, at: &str, out: &mut String) -> Result<()> {
        let m = match v {
            Value::Bool(true) => {
                out.push_str(ANY_JSON_VALUE);
                return Ok(());
            }
            Value::Bool(false) => bail!("{at}: the schema `false` admits no value"),
            Value::Object(m) => m,
            other => bail!("{at}: a schema must be an object or a boolean, got {other}"),
        };
        for k in m.keys() {
            if !SUPPORTED.contains(&k.as_str()) && !ANNOTATIONS.contains(&k.as_str()) && !k.starts_with("x-") {
                bail!(
                    "{at}: JSON Schema keyword {k:?} is not supported by guided decoding, and \
                     ignoring it would let the model emit values the schema forbids"
                );
            }
        }
        let constraints: Vec<&str> = m.keys().map(|k| k.as_str()).filter(|k| SUPPORTED.contains(k)).collect();
        let only = |allowed: &[&str]| constraints.iter().all(|k| allowed.contains(k));

        if let Some(r) = m.get("$ref") {
            let r = r.as_str().ok_or_else(|| anyhow!("{at}: $ref must be a string"))?;
            if !only(&["$ref"]) {
                bail!("{at}: $ref next to other constraint keywords is not supported; move them into the referenced schema");
            }
            if self.refs.iter().any(|x| x == r) {
                bail!("{at}: recursive $ref {r:?} cannot be compiled to a finite grammar");
            }
            if self.refs.len() >= MAX_REF_DEPTH_16_BOUNDS_NESTED_DEFINITIONS {
                bail!("{at}: $ref nesting deeper than {MAX_REF_DEPTH_16_BOUNDS_NESTED_DEFINITIONS}");
            }
            let target = self.resolve(r, at)?;
            self.refs.push(r.to_string());
            let res = self.node(target, r, out);
            self.refs.pop();
            return res;
        }
        if let Some(en) = m.get("enum") {
            let en = en.as_array().ok_or_else(|| anyhow!("{at}: enum must be an array"))?;
            if en.is_empty() {
                bail!("{at}: an empty enum admits no value");
            }
            let alts: Vec<String> = en.iter().map(literal).collect();
            out.push_str(&format!("({})", alts.join("|")));
            return Ok(());
        }
        if let Some(c) = m.get("const") {
            out.push_str(&literal(c));
            return Ok(());
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(alts) = m.get(key) {
                let alts = alts.as_array().filter(|a| !a.is_empty()).ok_or_else(|| anyhow!("{at}: {key} must be a non-empty array"))?;
                if !only(&[key, "nullable"]) {
                    bail!("{at}: {key} next to other constraint keywords is not supported; move them into each branch");
                }
                let mut parts = Vec::new();
                for (i, a) in alts.iter().enumerate() {
                    let mut s = String::new();
                    self.node(a, &format!("{at}/{key}/{i}"), &mut s)?;
                    parts.push(s);
                }
                if m.get("nullable").and_then(Value::as_bool) == Some(true) {
                    parts.push("null".into());
                }
                out.push_str(&format!("({})", parts.join("|")));
                return Ok(());
            }
        }
        if let Some(all) = m.get("allOf") {
            match all.as_array().map(|a| a.as_slice()) {
                Some([one]) if only(&["allOf"]) => return self.node(one, &format!("{at}/allOf/0"), out),
                _ => bail!("{at}: allOf is supported only as a single-element wrapper with no sibling constraints"),
            }
        }

        let mut types: Vec<String> = match m.get("type") {
            None => Vec::new(),
            Some(Value::String(t)) => vec![t.clone()],
            Some(Value::Array(ts)) => ts
                .iter()
                .map(|t| t.as_str().map(str::to_string).ok_or_else(|| anyhow!("{at}: type entries must be strings")))
                .collect::<Result<_>>()?,
            Some(t) => bail!("{at}: unsupported type value {t}"),
        };
        if types.is_empty() {
            let has = |keys: &[&str]| keys.iter().any(|k| m.contains_key(*k));
            if has(OBJECT_KEYS) {
                types.push("object".into());
            }
            if has(ARRAY_KEYS) {
                types.push("array".into());
            }
            if has(STRING_KEYS) {
                types.push("string".into());
            }
            if has(NUMBER_KEYS) {
                types.push("number".into());
            }
        }
        if m.get("nullable").and_then(Value::as_bool) == Some(true) && !types.iter().any(|t| t == "null") {
            types.push("null".into());
        }
        if types.is_empty() {
            out.push_str(ANY_JSON_VALUE);
            return Ok(());
        }
        let mut parts = Vec::new();
        for t in &types {
            let mut s = String::new();
            match t.as_str() {
                "string" => self.string(m, at, &mut s)?,
                "integer" => s.push_str(&self.integer(m, at)?),
                "number" => s.push_str(&self.number(m, at)?),
                "boolean" => s.push_str("(true|false)"),
                "null" => s.push_str("null"),
                "object" => self.object(m, at, &mut s)?,
                "array" => self.array(m, at, &mut s)?,
                other => bail!("{at}: unsupported schema type {other:?}"),
            }
            parts.push(s);
        }
        if parts.len() == 1 {
            out.push_str(&parts[0]);
        } else {
            out.push_str(&format!("({})", parts.join("|")));
        }
        Ok(())
    }

    fn string(&mut self, m: &Map<String, Value>, at: &str, out: &mut String) -> Result<()> {
        let min = count(m, "minLength", at)?;
        let max = count(m, "maxLength", at)?;
        let pattern = m.get("pattern");
        let format = m.get("format");
        let lengths = min.is_some() || max.is_some();
        if [pattern.is_some(), format.is_some(), lengths].iter().filter(|b| **b).count() > 1 {
            bail!("{at}: combining pattern, format and minLength/maxLength on one string is not supported; keep one");
        }
        if let Some(p) = pattern {
            let p = p.as_str().ok_or_else(|| anyhow!("{at}: pattern must be a string"))?;
            out.push_str(&json_string_pattern(p).with_context(|| at.to_string())?);
        } else if let Some(f) = format {
            let f = f.as_str().ok_or_else(|| anyhow!("{at}: format must be a string"))?;
            let re = format_regex(f).ok_or_else(|| {
                anyhow!("{at}: string format {f:?} is not supported by guided decoding (supported: date, time, date-time, uuid, email, ipv4, hostname, uri)")
            })?;
            out.push_str(&format!("\"({re})\""));
        } else if lengths {
            out.push('"');
            out.push_str(STRING_CHAR);
            out.push_str(&repeat(min.unwrap_or(0), max, at, "Length")?);
            out.push('"');
        } else {
            out.push_str(&format!("\"{STRING_CHAR}*\""));
        }
        Ok(())
    }

    fn bounds(m: &Map<String, Value>, at: &str, integer: bool) -> Result<(Option<Bound>, Option<Bound>)> {
        let num = |k: &str| -> Result<Option<f64>> {
            match m.get(k) {
                None | Some(Value::Bool(_)) => Ok(None),
                Some(v) => v.as_f64().map(Some).ok_or_else(|| anyhow!("{at}: {k} must be a number")),
            }
        };
        let flag = |k: &str| m.get(k).and_then(Value::as_bool) == Some(true);
        let mut lo: Option<(f64, bool)> = num("minimum")?.map(|v| (v, flag("exclusiveMinimum")));
        let mut hi: Option<(f64, bool)> = num("maximum")?.map(|v| (v, flag("exclusiveMaximum")));
        if let Some(x) = num("exclusiveMinimum")? {
            if lo.is_none_or(|(v, _)| x >= v) {
                lo = Some((x, true));
            }
        }
        if let Some(x) = num("exclusiveMaximum")? {
            if hi.is_none_or(|(v, _)| x <= v) {
                hi = Some((x, true));
            }
        }
        let to_bound = |b: Option<(f64, bool)>, lower: bool| -> Result<Option<Bound>> {
            let Some((v, exclusive)) = b else { return Ok(None) };
            if !v.is_finite() {
                bail!("{at}: numeric bound {v} is not finite");
            }
            if integer {
                let at = if lower {
                    if exclusive { v.floor() + 1.0 } else { v.ceil() }
                } else if exclusive {
                    v.ceil() - 1.0
                } else {
                    v.floor()
                };
                return Ok(Some(Bound { at: at as i128, exclusive: false }));
            }
            if v.fract() != 0.0 {
                bail!("{at}: a fractional bound ({v}) on a number is not supported by guided decoding; use an integer bound");
            }
            Ok(Some(Bound { at: v as i128, exclusive }))
        };
        Ok((to_bound(lo, true)?, to_bound(hi, false)?))
    }

    fn integer(&self, m: &Map<String, Value>, at: &str) -> Result<String> {
        let (lo, hi) = Self::bounds(m, at, true)?;
        if lo.is_none() && hi.is_none() {
            return Ok(INTEGER.into());
        }
        int_range(lo.map(|b| b.at), hi.map(|b| b.at)).with_context(|| at.to_string())
    }

    fn number(&self, m: &Map<String, Value>, at: &str) -> Result<String> {
        let (lo, hi) = Self::bounds(m, at, false)?;
        if lo.is_none() && hi.is_none() {
            return Ok(NUMBER.into());
        }
        number_range(lo, hi).with_context(|| at.to_string())
    }

    fn object(&mut self, m: &'a Map<String, Value>, at: &str, out: &mut String) -> Result<()> {
        let props = match m.get("properties") {
            None => None,
            Some(Value::Object(p)) => Some(p),
            Some(_) => bail!("{at}: properties must be an object"),
        };
        let required: Option<Vec<&str>> = match m.get("required") {
            None => None,
            Some(Value::Array(a)) => Some(
                a.iter()
                    .map(|v| v.as_str().ok_or_else(|| anyhow!("{at}: required entries must be strings")))
                    .collect::<Result<_>>()?,
            ),
            Some(_) => bail!("{at}: required must be an array"),
        };
        let addl = m.get("additionalProperties");
        let closed = addl == Some(&Value::Bool(false));
        if let Some(u) = m.get("unevaluatedProperties") {
            if u != &Value::Bool(false) {
                bail!("{at}: unevaluatedProperties other than false is not supported");
            }
        }
        let extra_schema: Option<&'a Value> = match addl {
            Some(v @ Value::Object(_)) => Some(v),
            _ => None,
        };
        let props_empty = props.is_none_or(|p| p.is_empty());
        if props_empty && required.as_ref().is_none_or(|r| r.is_empty()) {
            if closed {
                out.push_str(&format!(r"\{{{WS}\}}"));
                return Ok(());
            }
            match extra_schema {
                Some(s) => {
                    let mut v = String::new();
                    self.node(s, &format!("{at}/additionalProperties"), &mut v)?;
                    let entry = format!(r#""{STRING_CHAR}*"{WS}:{WS}{v}"#);
                    out.push_str(&format!(r"\{{{WS}({entry}({WS},{WS}{entry})*)?{WS}\}}"));
                }
                None => out.push_str(ANY_JSON_OBJECT),
            }
            return Ok(());
        }
        let mut items: Vec<(String, bool)> = Vec::new();
        let mut order: Vec<(&str, bool)> = match &required {
            Some(r) => r.iter().map(|k| (*k, true)).collect(),
            None => props.into_iter().flatten().map(|(k, _)| (k.as_str(), true)).collect(),
        };
        for (k, _) in props.into_iter().flatten() {
            if !order.iter().any(|(o, _)| *o == k.as_str()) {
                order.push((k.as_str(), false));
            }
        }
        for (key, must) in order {
            let sub = props.and_then(|p| p.get(key));
            let mut v = String::new();
            match (sub, extra_schema) {
                (Some(sub), _) => self.node(sub, &format!("{at}/properties/{key}"), &mut v)?,
                (None, _) if closed => {
                    bail!("{at}: required key {key:?} is not in properties and additionalProperties is false")
                }
                (None, Some(s)) => self.node(s, &format!("{at}/additionalProperties"), &mut v)?,
                (None, None) => v.push_str(ANY_JSON_VALUE),
            }
            items.push((format!("{}{WS}:{WS}{v}", literal(&Value::String(key.to_string()))), must));
        }
        let sep = format!("{WS},{WS}");
        let mut tail_after_first = String::new();
        let mut tail_as_first = String::new();
        for (item, must) in items.iter().rev() {
            let first = format!("{item}{tail_after_first}");
            if *must {
                tail_after_first = format!("{sep}{item}{tail_after_first}");
                tail_as_first = first;
            } else {
                tail_as_first = if tail_as_first.is_empty() {
                    format!("({first})?")
                } else {
                    format!("({first}|{tail_as_first})")
                };
                tail_after_first = format!("({sep}{item})?{tail_after_first}");
            }
        }
        out.push_str(&format!(r"\{{{WS}{tail_as_first}{WS}\}}"));
        Ok(())
    }

    fn array(&mut self, m: &'a Map<String, Value>, at: &str, out: &mut String) -> Result<()> {
        if m.get("uniqueItems").and_then(Value::as_bool) == Some(true) {
            bail!("{at}: uniqueItems: true is not supported by guided decoding");
        }
        let min = count(m, "minItems", at)?;
        let max = count(m, "maxItems", at)?;
        let sep = format!("{WS},{WS}");
        let tuple = match (m.get("prefixItems"), m.get("items")) {
            (Some(Value::Array(p)), _) => Some((p, m.get("items"))),
            (None, Some(Value::Array(p))) => Some((p, m.get("additionalItems"))),
            _ => None,
        };
        if let Some((prefix, rest)) = tuple {
            if min.is_some() || max.is_some() {
                bail!("{at}: minItems/maxItems next to a tuple (prefixItems) is not supported");
            }
            let mut parts = Vec::new();
            for (i, s) in prefix.iter().enumerate() {
                let mut v = String::new();
                self.node(s, &format!("{at}/prefixItems/{i}"), &mut v)?;
                parts.push(v);
            }
            let mut body = parts.join(&sep);
            match rest {
                Some(Value::Bool(false)) => {}
                Some(r) => {
                    let mut v = String::new();
                    self.node(r, &format!("{at}/items"), &mut v)?;
                    body = if body.is_empty() { format!("({v}({sep}{v})*)?") } else { format!("{body}({sep}{v})*") };
                }
                None => {
                    let v = ANY_JSON_VALUE;
                    body = if body.is_empty() { format!("({v}({sep}{v})*)?") } else { format!("{body}({sep}{v})*") };
                }
            }
            out.push_str(&format!(r"\[{WS}{body}{WS}\]"));
            return Ok(());
        }
        let mut item = String::new();
        match m.get("items") {
            Some(items) => self.node(items, &format!("{at}/items"), &mut item)?,
            None => item.push_str(ANY_JSON_VALUE),
        }
        let min = min.unwrap_or(0);
        let body = if max == Some(0) {
            if min > 0 {
                bail!("{at}: minItems {min} exceeds maxItems 0");
            }
            String::new()
        } else if min == 0 && max.is_none() {
            format!("(({item})({sep}({item}))*)?")
        } else {
            let tail = repeat(min.saturating_sub(1), max.map(|x| x - 1), at, "Items")?;
            let one = format!("({item})(({sep}({item})){tail})");
            if min == 0 { format!("({one})?") } else { one }
        };
        out.push_str(&format!(r"\[{WS}{body}{WS}\]"));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{schema_to_regex, JsonConstraint};
    use serde_json::json;

    fn ok(schema: &serde_json::Value, s: &str) -> bool {
        let re = schema_to_regex(schema).unwrap();
        let mut c = JsonConstraint::from_compiled_regex(&re).unwrap();
        c.advance_str(s) && c.can_terminate()
    }

    #[test]
    fn unions_nullable_and_type_arrays_take_every_branch_and_nothing_else() {
        for schema in [
            json!({"anyOf": [{"type": "integer"}, {"type": "null"}]}),
            json!({"oneOf": [{"type": "integer"}, {"type": "null"}]}),
            json!({"type": ["integer", "null"]}),
            json!({"type": "integer", "nullable": true}),
        ] {
            assert!(ok(&schema, "12") && ok(&schema, "null"), "{schema}");
            assert!(!ok(&schema, "\"12\"") && !ok(&schema, "1.5"), "{schema}");
        }
    }

    #[test]
    fn optional_properties_may_be_omitted_in_any_combination_and_follow_the_required_ones() {
        let s = json!({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}, "c": {"type": "integer"}}, "required": ["b"], "additionalProperties": false});
        for good in [r#"{"b":1}"#, r#"{"b":2,"a":1}"#, r#"{"b":2,"c":3}"#, r#"{"b":2,"a":1,"c":3}"#] {
            assert!(ok(&s, good), "{good}");
        }
        for bad in [r#"{}"#, r#"{"a":1}"#, r#"{"a":1,"b":2}"#, r#"{"b":1,"c":3,"a":1}"#, r#"{"b":1,}"#, r#"{,"b":1}"#, r#"{"b":1,"d":4}"#] {
            assert!(!ok(&s, bad), "{bad}");
        }
        let all_optional = json!({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": []});
        for good in ["{}", r#"{"a":1}"#, r#"{"b":2}"#, r#"{"a":1,"b":2}"#] {
            assert!(ok(&all_optional, good), "{good}");
        }
        assert!(!ok(&all_optional, r#"{"a":1"b":2}"#));
        let escaped_key = json!({"type": "object", "properties": {"a\"b": {"type": "integer"}}, "required": ["a\"b"]});
        assert!(ok(&escaped_key, r#"{"a\"b":1}"#), "keys needing json escapes are written escaped");
    }

    #[test]
    fn numeric_bounds_string_lengths_patterns_formats_and_item_counts_are_enforced() {
        let score = json!({"type": "integer", "minimum": 1, "maximum": 5});
        assert!(ok(&score, "1") && ok(&score, "5") && !ok(&score, "0") && !ok(&score, "6") && !ok(&score, "12"));
        let p = json!({"type": "number", "minimum": 0, "exclusiveMaximum": 1});
        assert!(ok(&p, "0") && ok(&p, "0.25") && !ok(&p, "1") && !ok(&p, "1.0") && !ok(&p, "-0.1"));
        let code = json!({"type": "string", "minLength": 3, "maxLength": 3});
        assert!(ok(&code, "\"CDG\"") && ok(&code, "\"\\u00e9ab\"") && !ok(&code, "\"-cdg\"") && !ok(&code, "\"ab\""));
        let zip = json!({"type": "string", "pattern": "^[0-9]{5}$"});
        assert!(ok(&zip, "\"20500\"") && !ok(&zip, "\"2050\""));
        let day = json!({"type": "string", "format": "date"});
        assert!(ok(&day, "\"2026-10-11\"") && !ok(&day, "\"2026-13-11\""));
        let tags = json!({"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3});
        assert!(ok(&tags, r#"["a","b"]"#) && ok(&tags, r#"["a", "b", "c"]"#));
        assert!(!ok(&tags, r#"["a"]"#) && !ok(&tags, r#"["a","b","c","d"]"#) && !ok(&tags, "[]"));
        let pair = json!({"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}], "items": false});
        assert!(ok(&pair, r#"[1,"x"]"#) && !ok(&pair, r#"[1,"x",2]"#) && !ok(&pair, r#"["x",1]"#));
    }

    #[test]
    fn local_refs_inline_and_maps_constrain_their_values() {
        let s = json!({"$defs": {"P": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}, "type": "array", "items": {"$ref": "#/$defs/P"}});
        assert!(ok(&s, r#"[{"x":1},{"x":2}]"#) && !ok(&s, r#"[{"y":1}]"#));
        let map = json!({"type": "object", "additionalProperties": {"type": "boolean"}});
        assert!(ok(&map, r#"{"a":true,"b":false}"#) && ok(&map, "{}") && !ok(&map, r#"{"a":1}"#));
    }

    #[test]
    fn unsupported_or_unsatisfiable_keywords_are_refused_with_their_name() {
        for (schema, word) in [
            (json!({"type": "integer", "multipleOf": 3}), "multipleOf"),
            (json!({"type": "array", "items": {}, "uniqueItems": true}), "uniqueItems"),
            (json!({"not": {"type": "null"}}), "not"),
            (json!({"type": "object", "patternProperties": {"^a": {}}}), "patternProperties"),
            (json!({"type": "string", "format": "iban"}), "iban"),
            (json!({"type": "number", "maximum": 0.5}), "fractional"),
            (json!({"$defs": {"N": {"type": "array", "items": {"$ref": "#/$defs/N"}}}, "$ref": "#/$defs/N"}), "recursive"),
            (json!({"$ref": "https://example.com/s.json"}), "local"),
            (json!({"allOf": [{"type": "string"}, {"maxLength": 2}]}), "allOf"),
            (json!({"type": "string", "maxLength": 100000}), "maxLength"),
        ] {
            let err = format!("{:#}", schema_to_regex(&schema).expect_err(&schema.to_string()));
            assert!(err.contains(word), "{schema}: {err}");
        }
        assert!(schema_to_regex(&json!({"type": "string", "title": "T", "description": "d", "default": "x", "x-extra": 1})).is_ok());
    }
}
