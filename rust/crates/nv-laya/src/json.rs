use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(entries) => Some(entries),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Int(n) => Some(*n as f64),
            Json::UInt(n) => Some(*n as f64),
            Json::Float(n) => Some(*n),
            _ => None,
        }
    }

    pub fn is_blank(&self) -> bool {
        matches!(self, Json::Null) || self.as_str() == Some("")
    }

    pub fn obj<K: Into<String>>(entries: impl IntoIterator<Item = (K, Json)>) -> Json {
        Json::Obj(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    pub fn set(&mut self, key: &str, value: Json) {
        if let Json::Obj(entries) = self {
            match entries.iter_mut().find(|(k, _)| k == key) {
                Some(entry) => entry.1 = value,
                None => entries.push((key.to_string(), value)),
            }
        }
    }

    pub fn remove(&mut self, key: &str) {
        if let Json::Obj(entries) = self {
            entries.retain(|(k, _)| k != key);
        }
    }

    pub fn python(&self) -> String {
        let mut out = String::new();
        self.write_python(&mut out);
        out
    }

    fn write_python(&self, out: &mut String) {
        match self {
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    item.write_python(out);
                }
                out.push(']');
            }
            Json::Obj(entries) => {
                out.push('{');
                for (i, (key, value)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&serde_json::to_string(key).expect("strings serialize"));
                    out.push_str(": ");
                    value.write_python(out);
                }
                out.push('}');
            }
            scalar => out.push_str(&serde_json::to_string(scalar).expect("scalars serialize")),
        }
    }

    pub fn state_text(&self) -> String {
        match self {
            Json::Str(s) => s.clone(),
            other => other.python(),
        }
    }
}

impl From<&str> for Json {
    fn from(s: &str) -> Self {
        Json::Str(s.to_string())
    }
}

impl From<String> for Json {
    fn from(s: String) -> Self {
        Json::Str(s)
    }
}

impl From<f64> for Json {
    fn from(n: f64) -> Self {
        Json::Float(n)
    }
}

impl From<usize> for Json {
    fn from(n: usize) -> Self {
        Json::UInt(n as u64)
    }
}

impl From<bool> for Json {
    fn from(b: bool) -> Self {
        Json::Bool(b)
    }
}

impl<T: Into<Json>> From<Option<T>> for Json {
    fn from(v: Option<T>) -> Self {
        v.map(Into::into).unwrap_or(Json::Null)
    }
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&serde_json::to_string(self).map_err(|_| fmt::Error)?)
    }
}

impl Serialize for Json {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Json::Null => s.serialize_unit(),
            Json::Bool(b) => s.serialize_bool(*b),
            Json::Int(n) => s.serialize_i64(*n),
            Json::UInt(n) => s.serialize_u64(*n),
            Json::Float(n) => s.serialize_f64(*n),
            Json::Str(v) => s.serialize_str(v),
            Json::Arr(items) => {
                let mut seq = s.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Json::Obj(entries) => {
                let mut map = s.serialize_map(Some(entries.len()))?;
                for (k, v) in entries {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Json;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_unit<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_bool<E>(self, v: bool) -> Result<Json, E> {
                Ok(Json::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Json, E> {
                Ok(Json::Int(v))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Json, E> {
                Ok(Json::UInt(v))
            }
            fn visit_f64<E>(self, v: f64) -> Result<Json, E> {
                Ok(Json::Float(v))
            }
            fn visit_str<E>(self, v: &str) -> Result<Json, E> {
                Ok(Json::Str(v.to_string()))
            }
            fn visit_string<E>(self, v: String) -> Result<Json, E> {
                Ok(Json::Str(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Json::Arr(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
                let mut entries: Vec<(String, Json)> = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, Json>()? {
                    match entries.iter_mut().find(|(k, _)| *k == key) {
                        Some(entry) => entry.1 = value,
                        None => entries.push((key, value)),
                    }
                }
                Ok(Json::Obj(entries))
            }
        }
        d.deserialize_any(V)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_key_order_renders_like_python_and_folds_repeated_keys() {
        let v = Json::parse(
            r#"{"b": 1, "a": [1, 2.5, "x", 3.0], "ok": true, "none": null, "u": "é", "neg": -4}"#,
        )
        .unwrap();
        assert_eq!(
            v.python(),
            r#"{"b": 1, "a": [1, 2.5, "x", 3.0], "ok": true, "none": null, "u": "é", "neg": -4}"#
        );
        assert_eq!(
            v.to_string(),
            r#"{"b":1,"a":[1,2.5,"x",3.0],"ok":true,"none":null,"u":"é","neg":-4}"#
        );
        let v = Json::parse(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        assert_eq!(v.python(), r#"{"a": 3, "b": 2}"#, "a repeated key keeps its first position and last value");
    }

}
