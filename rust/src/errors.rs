pub mod code {

    pub const INVALID_REQUEST_ERROR: &str = "invalid_request_error";
    pub const UNKNOWN_EVENT_TYPE: &str = "unknown_event_type";
    pub const SESSION_NOT_ACTIVE: &str = "session_not_active";
    pub const SESSION_UPDATE_INVALID: &str = "session_update_invalid";
    pub const RESPONSE_ALREADY_ACTIVE: &str = "response_already_active";
    pub const RESPONSE_CANCEL_NOT_ACTIVE: &str = "response_cancel_not_active";
    pub const INPUT_AUDIO_BUFFER_COMMIT_EMPTY: &str = "input_audio_buffer_commit_empty";
    pub const CLIENT_TOO_SLOW: &str = "client_too_slow";
    pub const EOU_KIND_UNSUPPORTED: &str = "eou_kind_unsupported";
    pub const EOU_FUSION_RULE_UNSUPPORTED: &str = "eou_fusion_rule_unsupported";

    pub const INTERNAL_STATE_ERROR: &str = "internal_state_error";
    pub const VAD_FAILED: &str = "vad_failed";
    pub const STT_FAILED: &str = "stt_failed";
    pub const MODEL_LOAD_FAILED: &str = "model_load_failed";
}

pub const KNOWN_CODES: &[&str] = &[
    code::INVALID_REQUEST_ERROR,
    code::UNKNOWN_EVENT_TYPE,
    code::SESSION_NOT_ACTIVE,
    code::SESSION_UPDATE_INVALID,
    code::RESPONSE_ALREADY_ACTIVE,
    code::RESPONSE_CANCEL_NOT_ACTIVE,
    code::INPUT_AUDIO_BUFFER_COMMIT_EMPTY,
    code::CLIENT_TOO_SLOW,
    code::EOU_KIND_UNSUPPORTED,
    code::EOU_FUSION_RULE_UNSUPPORTED,
    code::INTERNAL_STATE_ERROR,
    code::VAD_FAILED,
    code::STT_FAILED,
    code::MODEL_LOAD_FAILED,
];

pub fn is_known_code(c: &str) -> bool {
    KNOWN_CODES.contains(&c)
}

pub fn debug_assert_known_code(c: &str) {
    debug_assert!(
        is_known_code(c),
        "unknown error code {c:?}: add it to errors::code (RFC v3 §10.5)"
    );
}

pub fn error_type_for(code: &str) -> &'static str {
    match code {
        code::INVALID_REQUEST_ERROR
        | code::UNKNOWN_EVENT_TYPE
        | code::SESSION_NOT_ACTIVE
        | code::SESSION_UPDATE_INVALID
        | code::RESPONSE_ALREADY_ACTIVE
        | code::RESPONSE_CANCEL_NOT_ACTIVE
        | code::INPUT_AUDIO_BUFFER_COMMIT_EMPTY
        | code::CLIENT_TOO_SLOW
        | code::EOU_KIND_UNSUPPORTED
        | code::EOU_FUSION_RULE_UNSUPPORTED => "invalid_request_error",
        code::INTERNAL_STATE_ERROR
        | code::VAD_FAILED
        | code::STT_FAILED
        | code::MODEL_LOAD_FAILED => "server_error",
        _ => "invalid_request_error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_assert_known_code_accepts_registered() {
        debug_assert_known_code(code::INVALID_REQUEST_ERROR);
        debug_assert_known_code(code::RESPONSE_ALREADY_ACTIVE);
        debug_assert_known_code(code::INPUT_AUDIO_BUFFER_COMMIT_EMPTY);
        debug_assert_known_code(code::VAD_FAILED);
        debug_assert_known_code(code::CLIENT_TOO_SLOW);
        debug_assert_known_code(code::INTERNAL_STATE_ERROR);
        debug_assert_known_code(code::MODEL_LOAD_FAILED);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "unknown error code")]
    fn debug_assert_known_code_rejects_unknown() {
        debug_assert_known_code("totally_made_up_code");
    }

    #[test]
    fn known_codes_unique() {
        let mut sorted: Vec<&&str> = KNOWN_CODES.iter().collect();
        sorted.sort();
        let len_before = sorted.len();
        sorted.dedup();
        assert_eq!(
            len_before,
            sorted.len(),
            "duplicate entry in KNOWN_CODES: {sorted:?}"
        );
    }

    #[test]
    fn is_known_code_smoke() {
        assert!(is_known_code(code::CLIENT_TOO_SLOW));
        assert!(is_known_code(code::SESSION_NOT_ACTIVE));
        assert!(!is_known_code("not_a_real_code"));
        assert!(!is_known_code(""));
    }

    #[test]
    fn known_codes_match_rfc_v3_section_10_5() {
        let mut have: Vec<&str> = KNOWN_CODES.to_vec();
        have.sort();
        let mut want: Vec<&str> = vec![
            "invalid_request_error",
            "unknown_event_type",
            "session_not_active",
            "session_update_invalid",
            "response_already_active",
            "response_cancel_not_active",
            "input_audio_buffer_commit_empty",
            "client_too_slow",
            "eou_kind_unsupported",
            "eou_fusion_rule_unsupported",
            "internal_state_error",
            "vad_failed",
            "stt_failed",
            "model_load_failed",
        ];
        want.sort();
        assert_eq!(have, want, "registry differs from RFC v3 §10.5");
    }

    fn production_only(src: &str) -> &str {
        match src.find("\n#[cfg(test)]\n") {
            Some(at) => &src[..at],
            None => src,
        }
    }

    fn emits_code(src: &str, c: &str) -> bool {
        let ident = format!("::{}", c.to_uppercase());
        let bytes = src.as_bytes();
        let mut from = 0usize;
        while let Some(rel) = src[from..].find(&ident) {
            let end = from + rel + ident.len();
            from = end;
            if bytes
                .get(end)
                .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
            {
                continue;
            }
            let rest = src[end..].trim_start();
            if rest.starts_with("=>") || rest.starts_with('|') {
                continue;
            }
            return true;
        }
        false
    }

    #[test]
    fn every_known_code_has_an_emit_site() {
        const SOURCES: &[(&str, &str)] = &[
            ("realtime/session.rs", include_str!("realtime/session.rs")),
            ("realtime/pipeline.rs", include_str!("realtime/pipeline.rs")),
            (
                "realtime/session_update.rs",
                include_str!("realtime/session_update.rs"),
            ),
        ];
        for c in KNOWN_CODES {
            assert!(
                SOURCES
                    .iter()
                    .any(|(_, src)| emits_code(production_only(src), c)),
                "{c} is declared in KNOWN_CODES with no emit site \
                 (RFC v3 §10.5: declared-but-never-emitted is a v2 bug class); \
                 a match arm that maps the code is dispatch, not an emit -- \
                 either emit it or remove it from the registry",
            );
        }
    }

    #[test]
    fn emit_site_scan_rejects_dispatch_only_mentions() {
        let dispatch =
            "        c if c == errcode::STT_FAILED => state::TerminationReason::SttFailed,";
        assert!(
            !emits_code(dispatch, code::STT_FAILED),
            "a match arm that maps a code must not count as an emit site"
        );
        let alternation =
            "        code::STT_FAILED\n        | code::MODEL_LOAD_FAILED => \"server_error\",";
        assert!(!emits_code(alternation, code::STT_FAILED));
        let emit = "emit_error(self, errcode::STT_FAILED, &msg, None, None).await;";
        assert!(emits_code(emit, code::STT_FAILED));
        let payload = "json!({ \"code\": errcode::STT_FAILED })";
        assert!(emits_code(payload, code::STT_FAILED));
        assert!(!emits_code(emit, code::MODEL_LOAD_FAILED));
    }

    #[test]
    fn emit_site_scan_ignores_test_modules() {
        let src = "fn f() {}\n#[cfg(test)]\nmod tests {\n    assert_eq!(e.code, errcode::VAD_FAILED);\n}\n";
        assert!(!emits_code(production_only(src), code::VAD_FAILED));
    }

    #[test]
    fn eou_unsupported_codes_are_client_faults() {
        assert_eq!(
            error_type_for(code::EOU_KIND_UNSUPPORTED),
            "invalid_request_error"
        );
        assert_eq!(
            error_type_for(code::EOU_FUSION_RULE_UNSUPPORTED),
            "invalid_request_error"
        );
    }
}
