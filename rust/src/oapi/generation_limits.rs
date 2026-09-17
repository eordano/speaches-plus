pub const DEFAULT_MAX_TOKENS: usize = 256;

pub const DEFAULT_MAX_TOKENS_ENV_MOVES_THE_OMITTED_MAX_TOKENS_DEFAULT_OF_256: &str =
    "NV_DEFAULT_MAX_TOKENS";

pub const MAX_MAX_TOKENS: usize = 8192;

pub const MAX_MAX_TOKENS_ENV_LIFTS_THE_8192_CAP_THE_KV_WINDOW_STAYS_THE_REAL_BOUND: &str =
    "NV_MAX_MAX_TOKENS";

pub fn default_max_tokens_from(raw: Option<&str>) -> usize {
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAX_TOKENS)
}

pub fn max_max_tokens_from(raw: Option<&str>) -> usize {
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(MAX_MAX_TOKENS)
}

pub fn resolve_max_new_tokens(asked: Option<usize>, omitted_default: usize, cap: usize) -> usize {
    asked.unwrap_or(omitted_default).clamp(1, cap.max(1))
}

pub fn resolve_request_max_new_tokens(
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    omitted_default: usize,
    cap: usize,
) -> usize {
    resolve_max_new_tokens(max_completion_tokens.or(max_tokens), omitted_default, cap)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestWindowError {
    PromptDoesNotFit {
        prompt_tokens: usize,
        capacity: usize,
    },
    NoGenerationRoom {
        prompt_tokens: usize,
        max_new: usize,
        capacity: usize,
    },
}

impl std::fmt::Display for RequestWindowError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PromptDoesNotFit {
                prompt_tokens,
                capacity,
            } => write!(
                formatter,
                "prompt of {prompt_tokens} tokens does not fit the {capacity}-token KV window"
            ),
            Self::NoGenerationRoom {
                prompt_tokens,
                max_new,
                capacity,
            } => write!(
                formatter,
                "no room to generate {max_new} tokens after a {prompt_tokens}-token prompt in the {capacity}-token KV window"
            ),
        }
    }
}

impl std::error::Error for RequestWindowError {}

pub fn request_window(
    prompt_tokens: usize,
    max_new: usize,
    configured: usize,
    model_max: usize,
) -> Result<usize, RequestWindowError> {
    let capacity = configured.min(model_max);
    if prompt_tokens >= capacity {
        return Err(RequestWindowError::PromptDoesNotFit {
            prompt_tokens,
            capacity,
        });
    }
    if max_new > capacity - prompt_tokens {
        return Err(RequestWindowError::NoGenerationRoom {
            prompt_tokens,
            max_new,
            capacity,
        });
    }
    Ok(capacity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_cap_holds_the_exact_64k_boundary() {
        assert_eq!(resolve_max_new_tokens(Some(65_536), 256, 65_536), 65_536);
        assert_eq!(resolve_max_new_tokens(Some(65_537), 256, 65_536), 65_536);
    }
    #[test]

    fn request_fields_use_completion_precedence_and_the_exact_cap() {
        assert_eq!(
            resolve_request_max_new_tokens(Some(23), Some(17), 256, 65_536),
            17
        );
        assert_eq!(
            resolve_request_max_new_tokens(Some(23), None, 256, 65_536),
            23
        );
        assert_eq!(
            resolve_request_max_new_tokens(None, Some(17), 256, 65_536),
            17
        );
        assert_eq!(resolve_request_max_new_tokens(None, None, 256, 65_536), 256);
        assert_eq!(
            resolve_request_max_new_tokens(Some(65_536), None, 256, 65_536),
            65_536
        );
        assert_eq!(
            resolve_request_max_new_tokens(Some(65_537), None, 256, 65_536),
            65_536
        );
    }

    #[test]
    fn total_window_holds_prompt_plus_generation_boundaries() {
        assert_eq!(
            request_window(196_608, 65_536, 262_144, 262_144),
            Ok(262_144)
        );
        assert_eq!(request_window(262_143, 1, 262_144, 262_144), Ok(262_144));
        assert!(request_window(262_143, 2, 262_144, 262_144).is_err());
    }

    #[test]
    fn total_window_uses_the_lower_configured_or_model_limit() {
        assert_eq!(request_window(100, 1, 200, 300), Ok(200));
        assert_eq!(request_window(100, 1, 300, 200), Ok(200));
        assert!(request_window(199, 2, 300, 200).is_err());
    }
}
