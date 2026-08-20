use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

use speaches_plus::oapi::backend_select::{
    augment_capabilities, backends_report, backends_report_router, AUTO_POLICY,
    BACKENDS_REPORT_ROUTE, KNOWN_MODELS, REALTIME_CAPABILITIES_WITH_BACKENDS_ROUTE,
    WGPU_DECODERS_COMPILED_IN, WGPU_FEATURE_OFF_REASON,
};

const WHY_A_HANDLER_THAT_COMPILES_IS_NOT_A_HANDLER_THAT_IS_REACHABLE: &str =
    "handle_backends_report, realtime_capabilities_with_backends and augment_capabilities were \
     defined and unit-tested in backend_select.rs for a long while with zero references outside \
     that file: nothing mounted them, so no operator could ever read the report over HTTP. A unit \
     test on backends_report() stays green in exactly that state. This suite asserts reachability \
     instead -- it drives the Router the binary mounts, and it reads main.rs to prove the binary \
     mounts it.";

const THE_REPORT_ROUTE_IS_STATE_FREE_SO_THIS_SUITE_CAN_MOUNT_IT_ALONE: &str =
    "backends_report_router is generic over the router state and mounts no extract::State \
     handler, which is why this suite can drive the exact Router the binary merges without \
     constructing an AppState -- and therefore without loading a single model. A handler added \
     to that router that needs AppState belongs in main's own chain instead, next to \
     REALTIME_CAPABILITIES_WITH_BACKENDS_ROUTE, or this suite stops being CPU-runnable.";

const MAIN_RS_IS_THE_ONLY_PLACE_THE_BINARYS_ROUTER_IS_ASSEMBLED: &str =
    "the axum Router for the speaches-plus binary is built inline in main(); there is no \
     app_router() the suite could call, so the mounting half of reachability is asserted against \
     the source text of rust/src/main.rs. If main() is ever refactored to return its Router, \
     replace this text check with a oneshot against that Router.";

fn main_rs_source() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

async fn get_body(router: Router, method: Method, path: &str) -> (StatusCode, Vec<u8>, String) {
    let req = Request::builder()
        .method(method)
        .uri(path)
        .body(Body::empty())
        .expect("build request");
    let resp = router.oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let bytes = resp
        .into_body()
        .collect()
        .await
        .expect("collect body")
        .to_bytes()
        .to_vec();
    (status, bytes, content_type)
}

#[tokio::test]
async fn the_backends_report_is_served_as_json_on_its_route() {
    let router: Router = backends_report_router();
    let (status, bytes, content_type) =
        get_body(router, Method::GET, BACKENDS_REPORT_ROUTE).await;

    assert_eq!(
        status,
        StatusCode::OK,
        "{BACKENDS_REPORT_ROUTE} must answer 200: \
         {WHY_A_HANDLER_THAT_COMPILES_IS_NOT_A_HANDLER_THAT_IS_REACHABLE} \
         {THE_REPORT_ROUTE_IS_STATE_FREE_SO_THIS_SUITE_CAN_MOUNT_IT_ALONE}"
    );
    assert_eq!(
        content_type, "application/json",
        "an observability route an operator curls must declare JSON, not guess"
    );

    let body: Value = serde_json::from_slice(&bytes).expect("the report body must parse as JSON");

    assert_eq!(
        body["auto_policy"].as_str().unwrap_or_default(),
        AUTO_POLICY,
        "the served report must carry the auto policy verbatim; a truncated or paraphrased copy \
         is how an operator ends up believing auto is wgpu-first"
    );
    assert_eq!(
        body["wgpu_decoders_compiled_in"].as_bool(),
        Some(WGPU_DECODERS_COMPILED_IN),
        "the whole point of the route is telling an operator whether this binary has wgpu \
         decoders compiled in"
    );

    let available = body["available"]
        .as_array()
        .expect("available must be an array of backend probes");
    let names: Vec<&str> = available
        .iter()
        .filter_map(|b| b["name"].as_str())
        .collect();
    for expected in ["cuda", "wgpu", "cpu"] {
        assert!(
            names.contains(&expected),
            "backend {expected} missing from the served availability list {names:?}"
        );
    }
    for probe in available {
        let ok = probe["available"]
            .as_bool()
            .expect("every probe entry needs a boolean `available`");
        assert_eq!(
            probe["reason"].is_null(),
            ok,
            "an unavailable backend without a reason, or an available one carrying one, makes \
             the report useless for diagnosis: {probe}"
        );
    }

    assert_eq!(
        body["requested"].is_null(),
        !body["selection_error"].is_null(),
        "exactly one of requested / selection_error is set, so an operator can always tell a \
         valid NV_SERVE_BACKEND from a rejected one: {body}"
    );

    let models = body["models"]
        .as_object()
        .expect("models must be an object keyed by model id");
    assert_eq!(
        models.len(),
        KNOWN_MODELS.len(),
        "the served report must cover every model in KNOWN_MODELS"
    );
    let mut any_wgpu_servable = false;
    for (id, _) in KNOWN_MODELS {
        let entry = models
            .get(*id)
            .unwrap_or_else(|| panic!("{id} missing from the served report"));
        assert!(
            entry["class"].is_string(),
            "{id} entry must name its model class"
        );
        let wgpu_servable = entry["wgpu"]["servable"]
            .as_bool()
            .unwrap_or_else(|| panic!("{id} wgpu.servable must be a boolean"));
        assert_eq!(
            entry["wgpu"]["reason"].is_null(),
            wgpu_servable,
            "{id} must carry a refusal reason when it is not wgpu-servable and none when it is"
        );
        assert_eq!(
            entry["cuda"]["reason"].is_null(),
            entry["cuda"]["servable"].as_bool().unwrap_or(false),
            "{id} must carry a refusal reason when it is not cuda-servable and none when it is"
        );
        any_wgpu_servable |= wgpu_servable;
        if !WGPU_DECODERS_COMPILED_IN {
            assert!(
                !wgpu_servable,
                "{id} cannot be wgpu-servable in a binary built without the wgpu feature"
            );
            assert_eq!(
                entry["wgpu"]["reason"].as_str(),
                Some(WGPU_FEATURE_OFF_REASON),
                "{id} must blame the missing feature, not an incidental model gap"
            );
        }
    }
    if WGPU_DECODERS_COMPILED_IN {
        assert!(
            any_wgpu_servable,
            "a wgpu build whose report refuses every model on wgpu is reporting a lie or a \
             regression; a report that says `no` for everything is indistinguishable from a \
             stubbed one"
        );
    }
}

#[tokio::test]
async fn the_report_route_is_exact_and_get_only() {
    let router: Router = backends_report_router();
    let (status, _, _) = get_body(router, Method::GET, "/v1/backendsss").await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "the report router must not answer for neighbouring paths; a catch-all here would shadow \
         routes the binary merges after it"
    );

    let router: Router = backends_report_router();
    let (status, _, _) = get_body(router, Method::POST, BACKENDS_REPORT_ROUTE).await;
    assert_eq!(
        status,
        StatusCode::METHOD_NOT_ALLOWED,
        "the report is read-only; POST must be refused rather than silently treated as GET"
    );
}

#[test]
fn augment_capabilities_adds_the_report_without_dropping_the_base() {
    let base = json!({
        "realtime": {"turn_detection": ["server_vad"]},
        "models": {"transcription": ["whisper"]},
    });
    let out = augment_capabilities(base.clone());

    for key in ["realtime", "models"] {
        assert_eq!(
            out[key], base[key],
            "augment_capabilities must be additive: it may not rewrite or drop the capabilities \
             base that /v1/realtime/capabilities already publishes"
        );
    }
    assert_eq!(
        out["backends"]["auto_policy"].as_str(),
        Some(AUTO_POLICY),
        "the augmented capabilities must embed the same report the standalone route serves"
    );
    assert_eq!(
        out["backends"]["models"]
            .as_object()
            .map(|m| m.len())
            .unwrap_or(0),
        backends_report()["models"]
            .as_object()
            .map(|m| m.len())
            .unwrap_or(0),
        "the embedded report must be the whole report, not a subset"
    );

    let scalar = Value::String("not an object".to_string());
    assert_eq!(
        augment_capabilities(scalar.clone()),
        scalar,
        "a non-object base is returned untouched rather than replaced by a bare report"
    );
}

#[test]
fn the_binary_mounts_both_observability_routes() {
    let src = main_rs_source();
    assert!(
        src.contains("backends_report_router()"),
        "rust/src/main.rs must merge backend_select::backends_report_router() into the app \
         router. {MAIN_RS_IS_THE_ONLY_PLACE_THE_BINARYS_ROUTER_IS_ASSEMBLED} \
         {WHY_A_HANDLER_THAT_COMPILES_IS_NOT_A_HANDLER_THAT_IS_REACHABLE}"
    );
    assert!(
        src.contains("REALTIME_CAPABILITIES_WITH_BACKENDS_ROUTE")
            && src.contains("realtime_capabilities_with_backends"),
        "rust/src/main.rs must route {REALTIME_CAPABILITIES_WITH_BACKENDS_ROUTE} at \
         backend_select::realtime_capabilities_with_backends. \
         {MAIN_RS_IS_THE_ONLY_PLACE_THE_BINARYS_ROUTER_IS_ASSEMBLED}"
    );
    assert!(
        src.contains(r#".route("/v1/realtime/capabilities", get(realtime_capabilities))"#),
        "the pre-existing /v1/realtime/capabilities route must keep serving the unaugmented body \
         byte for byte; the backends view is an additional route, never a change to that one"
    );
}
