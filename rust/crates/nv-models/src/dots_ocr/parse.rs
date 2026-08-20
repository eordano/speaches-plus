use serde::{Deserialize, Serialize};

pub const CATEGORIES: [&str; 11] = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayoutElement {
    #[serde(default)]
    pub bbox: Option<[f32; 4]>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
}

impl LayoutElement {
    pub fn is_picture(&self) -> bool {
        self.category.as_deref() == Some("Picture")
    }

    pub fn category_is_known(&self) -> bool {
        self.category
            .as_deref()
            .is_some_and(|c| CATEGORIES.contains(&c))
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LayoutPage {
    pub elements: Vec<LayoutElement>,
    pub truncated: bool,
}

impl LayoutPage {
    pub fn to_plain_text(&self) -> String {
        let mut parts: Vec<&str> = Vec::new();
        for e in &self.elements {
            if e.is_picture() {
                continue;
            }
            if let Some(t) = e.text.as_deref() {
                if !t.trim().is_empty() {
                    parts.push(t);
                }
            }
        }
        parts.join("\n\n")
    }

    pub fn rescale(&mut self, sx: f32, sy: f32) {
        for e in &mut self.elements {
            if let Some(b) = &mut e.bbox {
                b[0] *= sx;
                b[1] *= sy;
                b[2] *= sx;
                b[3] *= sy;
            }
        }
    }
}

fn strip_fences(s: &str) -> &str {
    let t = s.trim();
    let Some(rest) = t.strip_prefix("```") else {
        return t;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    let rest = rest.trim_start_matches(['\r', '\n']);
    match rest.rfind("```") {
        Some(i) => &rest[..i],
        None => rest,
    }
}

fn object_spans(s: &str) -> Vec<(usize, Option<usize>)> {
    let bytes = s.as_bytes();
    let mut spans = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            b'}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    spans.push((start, Some(i + 1)));
                }
            }
            _ => {}
        }
    }
    if depth > 0 {
        spans.push((start, None));
    }
    spans
}

pub fn parse_layout_json(raw: &str) -> LayoutPage {
    let body = strip_fences(raw);
    let mut page = LayoutPage::default();
    if let Ok(v) = serde_json::from_str::<Vec<LayoutElement>>(body) {
        page.elements = v;
        return page;
    }
    for (start, end) in object_spans(body) {
        let Some(end) = end else {
            page.truncated = true;
            break;
        };
        match serde_json::from_str::<LayoutElement>(&body[start..end]) {
            Ok(e) => page.elements.push(e),
            Err(_) => page.truncated = true,
        }
    }
    if page.elements.is_empty() && !body.trim().is_empty() {
        page.truncated = true;
    }
    page
}

pub fn plain_text_fallback(raw: &str) -> LayoutPage {
    LayoutPage {
        elements: vec![LayoutElement {
            bbox: None,
            category: Some("Text".to_string()),
            text: Some(raw.to_string()),
        }],
        truncated: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_complete_array() {
        let raw = r##"[
          {"bbox": [10, 20, 30, 40], "category": "Title", "text": "# Hello"},
          {"bbox": [10, 50, 300, 90], "category": "Text", "text": "body"}
        ]"##;
        let page = parse_layout_json(raw);
        assert!(!page.truncated);
        assert_eq!(page.elements.len(), 2);
        assert_eq!(page.elements[0].bbox, Some([10.0, 20.0, 30.0, 40.0]));
        assert_eq!(page.elements[0].category.as_deref(), Some("Title"));
        assert_eq!(page.to_plain_text(), "# Hello\n\nbody");
        assert!(page.elements.iter().all(|e| e.category_is_known()));
    }

    #[test]
    fn recovers_complete_objects_from_a_truncated_array() {
        let raw = r#"[
          {"bbox": [1, 2, 3, 4], "category": "Text", "text": "one"},
          {"bbox": [5, 6, 7, 8], "category": "Text", "text": "tw"#;
        let page = parse_layout_json(raw);
        assert!(page.truncated);
        assert_eq!(page.elements.len(), 1);
        assert_eq!(page.to_plain_text(), "one");
    }

    #[test]
    fn braces_inside_strings_do_not_split_objects() {
        let raw = r#"[{"bbox":[0,0,1,1],"category":"Formula","text":"\\frac{a}{b} \" }"}]"#;
        let page = parse_layout_json(raw);
        assert!(!page.truncated);
        assert_eq!(page.elements.len(), 1);
        assert_eq!(page.elements[0].text.as_deref(), Some(r#"\frac{a}{b} " }"#));
    }

    #[test]
    fn picture_elements_have_no_text_and_are_skipped() {
        let raw = r#"[{"bbox":[0,0,10,10],"category":"Picture"},
                      {"bbox":[0,20,10,30],"category":"Text","text":"after"}]"#;
        let page = parse_layout_json(raw);
        assert_eq!(page.elements.len(), 2);
        assert!(page.elements[0].is_picture());
        assert_eq!(page.elements[0].text, None);
        assert_eq!(page.to_plain_text(), "after");
    }

    #[test]
    fn strips_markdown_code_fences() {
        let raw = "```json\n[{\"bbox\":[0,0,1,1],\"category\":\"Text\",\"text\":\"x\"}]\n```";
        let page = parse_layout_json(raw);
        assert!(!page.truncated);
        assert_eq!(page.to_plain_text(), "x");
    }

    #[test]
    fn reading_order_is_array_order_not_geometry() {
        let raw = r#"[{"bbox":[400,10,500,20],"category":"Text","text":"right column"},
                      {"bbox":[10,10,100,20],"category":"Text","text":"left column"}]"#;
        let page = parse_layout_json(raw);
        assert_eq!(page.to_plain_text(), "right column\n\nleft column");
    }

    #[test]
    fn rescale_maps_boxes_back_to_original_pixels() {
        let mut page =
            parse_layout_json(r#"[{"bbox":[10,20,30,40],"category":"Text","text":"a"}]"#);
        page.rescale(2.0, 0.5);
        assert_eq!(page.elements[0].bbox, Some([20.0, 10.0, 60.0, 20.0]));
    }

    #[test]
    fn empty_or_garbage_output_is_marked_truncated() {
        assert!(parse_layout_json("not json at all").truncated);
        assert!(!parse_layout_json("").truncated);
        assert!(parse_layout_json("").elements.is_empty());
    }

    #[test]
    fn plain_text_fallback_wraps_raw_output() {
        let page = plain_text_fallback("hello world");
        assert_eq!(page.to_plain_text(), "hello world");
        assert!(!page.truncated);
    }
}
