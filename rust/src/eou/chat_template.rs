#![allow(dead_code)]

use crate::defaults;

#[derive(Clone, Debug)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

impl Turn {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

pub fn format_qwen_chat(turns: &[Turn], partial: &str) -> String {
    let mut s = String::new();
    for t in turns {
        let role = if t.role.is_empty() {
            "user"
        } else {
            t.role.as_str()
        };
        s.push_str(defaults::eou::IM_START);
        s.push_str(role);
        s.push('\n');
        s.push_str(&t.content);
        s.push_str(defaults::eou::IM_END);
        s.push('\n');
    }
    if !partial.is_empty() {
        s.push_str(defaults::eou::IM_START);
        s.push_str("user\n");
        s.push_str(partial);
    }
    s
}

pub fn rolling_history(turns: &[Turn], max_turns: usize) -> &[Turn] {
    if max_turns == 0 || turns.len() <= max_turns {
        return turns;
    }
    &turns[turns.len() - max_turns..]
}
