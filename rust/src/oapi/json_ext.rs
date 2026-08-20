use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request};
use axum::response::Response;
use axum::Json;
use serde::de::DeserializeOwned;

use crate::oapi::{kind, openai_error};

pub struct OaiJson<T>(pub T);

impl<T, S> FromRequest<S> for OaiJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(OaiJson(value)),
            Err(rejection) => Err(json_rejection_response(&rejection)),
        }
    }
}

pub fn json_rejection_response(rejection: &JsonRejection) -> Response {
    openai_error(
        rejection.status(),
        rejection.body_text(),
        kind::INVALID_REQUEST,
        None,
        Some(rejection_code(rejection)),
    )
}

fn rejection_code(rejection: &JsonRejection) -> &'static str {
    match rejection {
        JsonRejection::JsonSyntaxError(_) => "invalid_json",
        JsonRejection::MissingJsonContentType(_) => "unsupported_content_type",
        _ => "invalid_request_body",
    }
}
