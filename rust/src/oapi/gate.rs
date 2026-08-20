use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use axum::response::Response;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::oapi;
use crate::oapi::deadline;

#[derive(Debug)]
pub struct Busy {
    pub surface: &'static str,
    pub permits: usize,
    pub waited_ms: u64,
}

impl Busy {
    pub fn into_response(self) -> Response {
        let Busy {
            surface,
            permits,
            waited_ms,
        } = self;
        tracing::warn!(
            surface,
            permits,
            waited_ms,
            "shed a request: the surface was at capacity for the whole queue window"
        );
        oapi::openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "{surface} is at capacity ({permits} concurrent) and the request waited \
                 {waited_ms} ms without a slot. Retry shortly, or raise the surface's \
                 concurrency/queue limits."
            ),
            oapi::kind::SERVICE_UNAVAIL,
            None,
            Some("surface_busy"),
        )
    }
}

pub struct SurfaceGate {
    surface: &'static str,
    sem: Arc<Semaphore>,
    permits: usize,
    queue: Duration,
}

impl SurfaceGate {
    pub fn new(surface: &'static str, permits: usize, queue_ms: u64) -> Self {
        let permits = permits.max(1);
        Self {
            surface,
            sem: Arc::new(Semaphore::new(permits)),
            permits,
            queue: Duration::from_millis(queue_ms),
        }
    }

    pub fn from_env(
        surface: &'static str,
        permits_var: &str,
        queue_var: &str,
        default_permits: usize,
        default_queue_ms: u64,
    ) -> Self {
        let permits = std::env::var(permits_var)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(default_permits);
        let queue_ms = std::env::var(queue_var)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(default_queue_ms);
        tracing::info!(
            surface,
            permits,
            queue_ms,
            permits_var,
            queue_var,
            "surface concurrency gate armed"
        );
        Self::new(surface, permits, queue_ms)
    }

    pub fn permits(&self) -> usize {
        self.permits
    }

    pub fn queue_ms(&self) -> u64 {
        self.queue.as_millis() as u64
    }

    pub fn budget(&self, client: Option<Duration>) -> Duration {
        deadline::resolve(client, self.queue)
    }

    pub fn budget_ms(&self, client: Option<Duration>) -> u64 {
        self.budget(client).as_millis() as u64
    }

    pub async fn acquire(&self) -> Result<OwnedSemaphorePermit, Busy> {
        self.acquire_with_deadline(None).await
    }

    pub async fn acquire_with_deadline(
        &self,
        client: Option<Duration>,
    ) -> Result<OwnedSemaphorePermit, Busy> {
        let budget = self.budget(client);
        let started = Instant::now();
        match tokio::time::timeout(budget, self.sem.clone().acquire_owned()).await {
            Ok(Ok(permit)) => {
                let waited = started.elapsed();
                if waited > Duration::from_millis(50) {
                    tracing::debug!(
                        surface = self.surface,
                        waited_ms = waited.as_millis() as u64,
                        budget_ms = budget.as_millis() as u64,
                        "request queued before acquiring a slot"
                    );
                }
                Ok(permit)
            }
            Ok(Err(_)) => Err(Busy {
                surface: self.surface,
                permits: self.permits,
                waited_ms: started.elapsed().as_millis() as u64,
            }),
            Err(_) => Err(Busy {
                surface: self.surface,
                permits: self.permits,
                waited_ms: budget.as_millis() as u64,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sheds_once_permits_are_exhausted_and_the_window_expires() {
        let gate = SurfaceGate::from_env("test-surface", "NO_SUCH_PERMITS", "NO_SUCH_QUEUE", 1, 60);
        let held = gate.acquire().await.expect("first acquire");
        let t0 = Instant::now();
        let busy = gate.acquire().await.expect_err("second must shed");
        let elapsed = t0.elapsed();
        assert_eq!(busy.permits, 1);
        assert!(
            elapsed >= Duration::from_millis(50) && elapsed < Duration::from_millis(2000),
            "shed should happen at the queue deadline, took {elapsed:?}"
        );
        drop(held);
        let reused = gate
            .acquire()
            .await
            .expect("slot is reusable after release");
        drop(reused);
    }

    #[tokio::test]
    async fn a_waiter_proceeds_when_a_slot_frees_inside_the_window() {
        let gate = Arc::new(SurfaceGate::from_env(
            "test-surface-2",
            "NO_SUCH_PERMITS",
            "NO_SUCH_QUEUE",
            1,
            5_000,
        ));
        let held = gate.acquire().await.expect("first acquire");
        let g2 = gate.clone();
        let waiter = tokio::spawn(async move { g2.acquire().await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(held);
        waiter
            .await
            .expect("join")
            .expect("waiter should get the freed slot, not a shed");
    }

    #[tokio::test(start_paused = true)]
    async fn a_short_client_deadline_sheds_before_the_server_default_does() {
        let gate = Arc::new(SurfaceGate::new("test-deadline-short", 1, 5_000));
        let held = gate.acquire().await.expect("first acquire");

        let g_short = gate.clone();
        let short = tokio::spawn(async move {
            g_short
                .acquire_with_deadline(Some(Duration::from_millis(100)))
                .await
                .map(|_| ())
        });
        let g_default = gate.clone();
        let mut default = tokio::spawn(async move { g_default.acquire().await.map(|_| ()) });

        tokio::time::sleep(Duration::from_millis(200)).await;

        let busy = short
            .await
            .expect("join")
            .expect_err("the 100 ms caller must shed at its own deadline");
        assert_eq!(busy.waited_ms, 100, "shed at the client budget, not 5000");

        assert!(
            tokio::time::timeout(Duration::from_millis(500), &mut default)
                .await
                .is_err(),
            "the server-default waiter must still be queued long after the short one shed"
        );

        drop(held);
        default
            .await
            .expect("join")
            .expect("the server-default waiter still gets the freed slot");
    }

    #[tokio::test(start_paused = true)]
    async fn a_long_client_deadline_outlasts_the_server_default_but_is_clamped_to_the_max() {
        assert!(
            std::env::var(crate::oapi::deadline::MAX_VAR).is_err(),
            "this test pins the default max; unset {} to run it",
            crate::oapi::deadline::MAX_VAR
        );
        let gate = Arc::new(SurfaceGate::new("test-deadline-long", 1, 5_000));
        let held = gate.acquire().await.expect("first acquire");

        let g = gate.clone();
        let mut long = tokio::spawn(async move {
            g.acquire_with_deadline(Some(Duration::from_secs(3600)))
                .await
                .map(|_| ())
        });

        assert!(
            tokio::time::timeout(Duration::from_secs(60), &mut long)
                .await
                .is_err(),
            "a 3600 s caller must outlast the 5 s server default"
        );

        let busy = tokio::time::timeout(Duration::from_secs(120), &mut long)
            .await
            .expect("must have shed by the 120 s server maximum, not waited 3600 s")
            .expect("join")
            .expect_err("shed, not admitted");
        assert_eq!(
            busy.waited_ms,
            crate::oapi::deadline::DEFAULT_MAX_MS,
            "the reported window is the clamped max"
        );

        drop(held);
    }

    #[tokio::test(start_paused = true)]
    async fn a_garbage_header_leaves_the_server_default_in_force() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            crate::oapi::deadline::HEADER,
            axum::http::HeaderValue::from_static("soon"),
        );
        let client = crate::oapi::deadline::from_headers(&headers);
        assert_eq!(client, None);

        let gate = Arc::new(SurfaceGate::new("test-deadline-garbage", 1, 5_000));
        assert_eq!(gate.budget_ms(client), 5_000);
        let held = gate.acquire().await.expect("first acquire");

        let g = gate.clone();
        let mut waiter =
            tokio::spawn(async move { g.acquire_with_deadline(client).await.map(|_| ()) });
        assert!(
            tokio::time::timeout(Duration::from_millis(1_000), &mut waiter)
                .await
                .is_err(),
            "garbage must not collapse the window to zero"
        );

        drop(held);
        waiter
            .await
            .expect("join")
            .expect("the server-default window still admits once a slot frees");
    }

    #[test]
    fn budget_reflects_the_clamps_without_touching_the_semaphore() {
        let gate = SurfaceGate::new("test-budget", 1, 3_000);
        assert_eq!(gate.budget_ms(None), 3_000);
        assert_eq!(gate.budget_ms(Some(Duration::from_millis(250))), 250);
        assert_eq!(gate.budget_ms(Some(Duration::ZERO)), 50);
        assert_eq!(
            gate.budget_ms(Some(Duration::from_secs(86_400))),
            crate::oapi::deadline::DEFAULT_MAX_MS
        );
    }
}
