use crate::json::Json;
use crate::lang;
use anyhow::{bail, Result};

pub const MODELS: [&str; 3] = ["english", "multilingual", "typed-decisions"];

const ALIASES: &[(&str, &str)] = &[
    ("en", "english"),
    ("laya", "english"),
    ("default", "english"),
    ("multi", "multilingual"),
    ("ml", "multilingual"),
    ("laya-multilingual", "multilingual"),
    ("typed", "typed-decisions"),
    ("typed_decisions", "typed-decisions"),
    ("laya-typed-decisions", "typed-decisions"),
    ("decisions", "typed-decisions"),
];

const WORKFLOWS: &[(&str, &[&str])] = &[
    (
        "agent_trace_observability",
        &["action", "needs_review", "outcome", "risk", "urgency"],
    ),
    (
        "customer_service",
        &["action", "category", "churn_risk", "needs_human", "urgency"],
    ),
    (
        "invoice_processing",
        &[
            "discrepancy_severity",
            "disposition",
            "duplicate",
            "matches_order",
            "urgency",
        ],
    ),
    (
        "security_incidents",
        &[
            "credential_compromise",
            "disposition",
            "severity",
            "true_positive",
            "urgency",
        ],
    ),
];

pub fn directory(model: &str) -> &'static str {
    match model {
        "multilingual" => "laya-multilingual",
        "typed-decisions" => "laya-typed-decisions",
        _ => "laya",
    }
}

pub fn normalise(name: &str) -> Result<&'static str> {
    let key = name.trim().to_lowercase();
    let key = ALIASES
        .iter()
        .find(|(a, _)| *a == key)
        .map(|(_, m)| *m)
        .unwrap_or(&key)
        .to_string();
    match MODELS.iter().find(|m| **m == key) {
        Some(m) => Ok(m),
        None => bail!("unknown model {name:?}; choose one of {MODELS:?}"),
    }
}

fn workflow(questions: &[(String, Json)]) -> Option<&'static str> {
    let mut ids: Vec<&str> = questions.iter().map(|(k, _)| k.as_str()).collect();
    ids.sort_unstable();
    WORKFLOWS
        .iter()
        .find(|(_, sig)| **sig == ids[..])
        .map(|(name, _)| *name)
}

pub struct Hints<'a> {
    pub model: Option<&'a str>,
    pub task: Option<&'a str>,
    pub lang: Option<&'a str>,
}

pub struct Policy {
    pub default: &'static str,
    pub auto_task_detection: bool,
}

pub fn route(
    state: &Json,
    questions: &[(String, Json)],
    hints: &Hints,
    policy: &Policy,
) -> Result<(&'static str, Json)> {
    let decision =
        |model: &'static str, reason: String, detection: Json, workflow: Option<&str>| {
            (
                model,
                Json::obj([
                    ("model", model.into()),
                    (
                        "repo",
                        format!("convaiinnovations/{}", directory(model)).into(),
                    ),
                    ("reason", reason.into()),
                    ("detection", detection),
                    ("workflow", workflow.into()),
                ]),
            )
        };
    if let Some(model) = hints.model {
        return Ok(decision(
            normalise(model)?,
            format!("explicit model='{model}'"),
            Json::Null,
            None,
        ));
    }
    if let Some(task) = hints.task {
        let wanted = if task.to_lowercase().replace('-', "_") == "typed_decisions" {
            "typed-decisions"
        } else {
            task
        };
        return Ok(decision(
            normalise(wanted)?,
            format!("explicit task='{task}'"),
            Json::Null,
            None,
        ));
    }
    let workflow = workflow(questions);
    if let (Some(wf), true) = (workflow, policy.auto_task_detection) {
        return Ok(decision(
            "typed-decisions",
            format!("question ids match the '{wf}' typed-decisions workflow"),
            Json::Null,
            workflow,
        ));
    }
    if let Some(lang) = hints.lang {
        let english = matches!(
            lang.to_lowercase().split('-').next(),
            Some("en" | "eng" | "english")
        );
        return Ok(decision(
            if english { "english" } else { "multilingual" },
            format!("explicit lang='{lang}'"),
            Json::Null,
            workflow,
        ));
    }
    let det = lang::analyse(state);
    let (model, reason) = if det.script == "unknown" {
        (
            policy.default,
            format!(
                "no letters detected in state; using default ({})",
                policy.default
            ),
        )
    } else if det.script != "latin" {
        (
            "multilingual",
            format!(
                "non-Latin script ({}, {:.0}% of letters); the English checkpoint cannot read it",
                det.script,
                100.0 * det.non_latin_fraction
            ),
        )
    } else if !det.is_english {
        (
            "multilingual",
            format!(
                "Latin script but language looks like '{}', not English",
                det.language.unwrap_or("?")
            ),
        )
    } else {
        ("english", "English Latin text".to_string())
    };
    Ok(decision(model, reason, det.json, workflow))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pick(state: &str, hints: Hints) -> &'static str {
        route(
            &Json::from(state),
            &[],
            &hints,
            &Policy {
                default: "english",
                auto_task_detection: false,
            },
        )
        .unwrap()
        .0
    }

    #[test]
    fn precedence_and_detection_follow_the_hints_and_the_script_and_language_analysis() {
        let none = || Hints {
            model: None,
            task: None,
            lang: None,
        };
        assert_eq!(
            pick("I was charged twice for the same invoice", none()),
            "english"
        );
        assert_eq!(pick("Меня дважды списали", none()), "multilingual");
        assert_eq!(
            pick(
                "Меня дважды списали",
                Hints {
                    model: Some("en"),
                    ..none()
                }
            ),
            "english"
        );
        assert_eq!(
            pick(
                "hello",
                Hints {
                    lang: Some("de-DE"),
                    ..none()
                }
            ),
            "multilingual"
        );
        assert_eq!(
            pick(
                "hello",
                Hints {
                    task: Some("typed-decisions"),
                    ..none()
                }
            ),
            "typed-decisions"
        );
        assert!(normalise("nope").is_err());

        let of = |text: &str| crate::lang::analyse(&Json::parse(text).unwrap());
        assert!(
            of(r#"{"message": "I was charged twice, please refund the duplicate"}"#).is_english
        );
        let de = of(
            r#"{"message": "Mein Konto wurde zweimal belastet und ich bin nicht zufrieden mit der Antwort"}"#,
        );
        assert_eq!(
            (de.script.as_str(), de.language, de.is_english),
            ("latin", Some("de"), false)
        );
        assert_eq!(of(r#""तुम यहाँ से चले जाओ""#).script, "devanagari");
        assert_eq!(of(r#""请帮我退款""#).script, "han");
        assert_eq!(of(r#"{"n": 12}"#).script, "unknown");
        assert!(of(r#""hola amigo""#).is_english, "short latin text stays english");
    }
}
