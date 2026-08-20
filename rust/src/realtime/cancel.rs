use std::future::Future;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

pub(super) const SESSION_CANCEL_QUIESCE_MS: u64 = 250;

struct LaneGuard {
    inflight: Arc<AtomicU32>,
    idle: Arc<Notify>,
}

impl LaneGuard {
    fn acquire(inflight: &Arc<AtomicU32>, idle: &Arc<Notify>) -> Self {
        inflight.fetch_add(1, Ordering::AcqRel);
        Self {
            inflight: inflight.clone(),
            idle: idle.clone(),
        }
    }
}

impl Drop for LaneGuard {
    fn drop(&mut self) {
        if self.inflight.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.idle.notify_waiters();
        }
    }
}

#[derive(Clone)]
pub(super) struct SessionCancel {
    token: CancellationToken,
    inflight: Arc<AtomicU32>,
    idle: Arc<Notify>,
}

impl Default for SessionCancel {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionCancel {
    pub(super) fn new() -> Self {
        Self {
            token: CancellationToken::new(),
            inflight: Arc::new(AtomicU32::new(0)),
            idle: Arc::new(Notify::new()),
        }
    }

    pub(super) fn token(&self) -> CancellationToken {
        self.token.clone()
    }

    pub(super) fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    pub(super) fn lanes_inflight(&self) -> u32 {
        self.inflight.load(Ordering::Acquire)
    }

    pub(super) fn wrap<F>(&self, fut: F) -> impl Future<Output = Option<F::Output>> + Send
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let guard = LaneGuard::acquire(&self.inflight, &self.idle);
        let token = self.token.clone();
        async move {
            let _guard = guard;
            token.run_until_cancelled(fut).await
        }
    }

    pub(super) fn wrap_unit<F>(&self, fut: F) -> impl Future<Output = ()> + Send
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let wrapped = self.wrap(fut);
        async move {
            let _ = wrapped.await;
        }
    }

    pub(super) async fn cancel(&self) {
        self.token.cancel();
        self.await_quiescent(Duration::from_millis(SESSION_CANCEL_QUIESCE_MS))
            .await;
    }

    pub(super) async fn await_quiescent(&self, cap: Duration) {
        let _ = tokio::time::timeout(cap, async {
            loop {
                if self.inflight.load(Ordering::Acquire) == 0 {
                    return;
                }
                let idle = self.idle.notified();
                if self.inflight.load(Ordering::Acquire) == 0 {
                    return;
                }
                idle.await;
            }
        })
        .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wrap_yields_none_once_cancelled() {
        let cancel = SessionCancel::new();
        let task = tokio::spawn(cancel.wrap(async {
            tokio::time::sleep(Duration::from_secs(30)).await;
            7u32
        }));
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(cancel.lanes_inflight(), 1);
        cancel.cancel().await;
        assert!(cancel.is_cancelled());
        assert_eq!(task.await.expect("lane joined"), None);
        assert_eq!(cancel.lanes_inflight(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_awaits_lane_quiescence_before_returning() {
        let cancel = SessionCancel::new();
        let _task = tokio::spawn(cancel.wrap(async {
            tokio::time::sleep(Duration::from_secs(30)).await;
        }));
        tokio::time::sleep(Duration::from_millis(20)).await;
        cancel.cancel().await;
        assert_eq!(cancel.lanes_inflight(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_completed_lane_leaves_nothing_inflight() {
        let cancel = SessionCancel::new();
        let out = tokio::spawn(cancel.wrap(async { 3u32 }))
            .await
            .expect("lane joined");
        assert_eq!(out, Some(3));
        assert_eq!(cancel.lanes_inflight(), 0);
        cancel.cancel().await;
    }
}
