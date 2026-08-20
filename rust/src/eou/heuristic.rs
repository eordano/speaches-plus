use super::EouModel;

pub struct HeuristicEouModel;

impl HeuristicEouModel {
    fn last_word(s: &str) -> &str {
        let trimmed = s.trim_end_matches(|c: char| {
            c.is_whitespace() || matches!(c, '.' | '!' | '?' | ',' | ';' | ':')
        });
        trimmed
            .rsplit(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-')
            .next()
            .unwrap_or("")
    }

    pub fn score_text(s: &str) -> f32 {
        let s = s.trim();
        if s.is_empty() {
            return 0.1;
        }
        if s.ends_with("...") || s.ends_with('…') {
            return 0.15;
        }
        let last_non_ws_char = s.chars().last().unwrap_or(' ');
        let last = Self::last_word(s).to_ascii_lowercase();
        const HESITATION: &[&str] = &["uh", "um", "uhh", "umm", "er", "erm", "hmm", "like", "so"];
        const CONTINUATIONS: &[&str] = &[
            "and", "or", "but", "with", "the", "a", "an", "to", "of", "for", "is", "was", "are",
            "were", "because", "since", "if", "when", "while", "as", "than", "that", "which",
            "who", "whom", "whose",
        ];
        match last_non_ws_char {
            '.' | '!' | '?' => return 0.95,
            ',' | ';' | ':' | '-' => return 0.25,
            _ => {}
        }
        if last.is_empty() {
            return 0.3;
        }
        if HESITATION.contains(&last.as_str()) {
            return 0.15;
        }
        if CONTINUATIONS.contains(&last.as_str()) {
            return 0.2;
        }
        0.6
    }
}

impl EouModel for HeuristicEouModel {
    fn score(&self, context: &str) -> f32 {
        Self::score_text(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_terminator_high() {
        assert!(HeuristicEouModel::score_text("are you sure?") >= 0.9);
        assert!(HeuristicEouModel::score_text("done.") >= 0.9);
        assert!(HeuristicEouModel::score_text("wow!") >= 0.9);
    }

    #[test]
    fn heuristic_continuation_low() {
        assert!(HeuristicEouModel::score_text("the cat is on the") <= 0.25);
        assert!(HeuristicEouModel::score_text("apples and") <= 0.25);
        assert!(HeuristicEouModel::score_text("she said,") <= 0.3);
    }

    #[test]
    fn heuristic_hesitation_lowest() {
        assert!(HeuristicEouModel::score_text("I think um") <= 0.2);
        assert!(HeuristicEouModel::score_text("hmm") <= 0.2);
    }

    #[test]
    fn heuristic_ellipsis_is_trailing_off_not_a_terminator() {
        assert!(
            HeuristicEouModel::score_text("about 440 for the...") <= 0.2,
            "whisper writes '...' exactly when the speaker trails off; reading the final '.' \
             as a strong terminator inverts the signal"
        );
        assert!(HeuristicEouModel::score_text("I was thinking…") <= 0.2);
        assert!(HeuristicEouModel::score_text("done.") >= 0.9);
    }

    #[test]
    fn heuristic_neutral_midrange() {
        let s = HeuristicEouModel::score_text("the train arrives at noon");
        assert!((0.4..=0.8).contains(&s), "got {s}");
    }

    #[test]
    fn heuristic_empty_low() {
        assert!(HeuristicEouModel::score_text("") <= 0.2);
        assert!(HeuristicEouModel::score_text("   ") <= 0.2);
    }
}
