#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;

use super::state::{
    apply_truncate_to_conversation, check_state, ResponseRuntime, SealedBuffer, SessionPhase,
    SessionState, TerminationReason, VadPhase,
};
use crate::types::{ItemId, Millis, ResponseId};

#[derive(Clone, Copy, Debug)]
pub enum FuzzOp {
    SessionActivate,
    VadSpeechStart,
    VadSpeechStop,
    StartPredicted,
    PromotePredicted,
    CreateFromNone,
    AdvanceToStreaming,
    Drain,
    RetireToNone,
    RetirePredicted,
    RetirePredictedFull,
    StoreSealedBuffer,
    DropSealedBuffer,
    TruncateConversation,
    Terminate,
}

const ALL_OPS: &[FuzzOp] = &[
    FuzzOp::SessionActivate,
    FuzzOp::VadSpeechStart,
    FuzzOp::VadSpeechStop,
    FuzzOp::StartPredicted,
    FuzzOp::PromotePredicted,
    FuzzOp::CreateFromNone,
    FuzzOp::AdvanceToStreaming,
    FuzzOp::Drain,
    FuzzOp::RetireToNone,
    FuzzOp::RetirePredicted,
    FuzzOp::RetirePredictedFull,
    FuzzOp::StoreSealedBuffer,
    FuzzOp::DropSealedBuffer,
    FuzzOp::TruncateConversation,
    FuzzOp::Terminate,
];

pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    pub fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    pub fn pick<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        let i = (self.next() as usize) % slice.len();
        &slice[i]
    }
}

fn dummy_runtime() -> ResponseRuntime {
    use std::sync::OnceLock;
    use tokio::runtime::{Handle, Runtime};
    static FALLBACK_RT: OnceLock<Runtime> = OnceLock::new();

    let h = match Handle::try_current() {
        Ok(h) => h.spawn(async {}),
        Err(_) => {
            let rt = FALLBACK_RT.get_or_init(|| {
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .enable_all()
                    .build()
                    .expect("fallback rt")
            });
            rt.spawn(async {})
        }
    };
    ResponseRuntime {
        handle: h,
        transcript_so_far: Arc::new(tokio::sync::Mutex::new(String::new())),
        wire_opened: Arc::new(AtomicBool::new(false)),
    }
}

fn apply(op: FuzzOp, state: &mut SessionState, idx: u64) {
    match op {
        FuzzOp::SessionActivate => {
            if matches!(state.session, SessionPhase::Pending) {
                state.session = SessionPhase::Active {
                    created_at_ms: Millis(idx),
                };
            }
        }
        FuzzOp::Terminate => {
            if !matches!(state.session, SessionPhase::Terminated { .. }) {
                state.session = SessionPhase::Terminated {
                    reason: TerminationReason::ClientClosed,
                };
            }
        }
        FuzzOp::VadSpeechStart => {
            if matches!(state.session, SessionPhase::Active { .. })
                && matches!(state.vad, VadPhase::Silent)
                && !state.resp.is_active()
            {
                state.vad = VadPhase::Speaking {
                    item_id: ItemId::new(format!("item_{idx}")),
                    audio_start_ms: Millis(idx * 100),
                };
            }
        }
        FuzzOp::VadSpeechStop => {
            if let VadPhase::Speaking {
                item_id,
                audio_start_ms,
            } = &state.vad
            {
                let item_id = item_id.clone();
                let audio_start_ms = *audio_start_ms;
                state.vad = VadPhase::Stopped {
                    item_id,
                    audio_start_ms,
                    audio_end_ms: Millis(audio_start_ms.raw() + 1000),
                };
            }
        }
        FuzzOp::StartPredicted => {
            if matches!(state.session, SessionPhase::Active { .. }) {
                let _ = state.resp_start_predicted(
                    ResponseId::new(format!("resp_{idx}")),
                    ItemId::new(format!("item_{idx}")),
                    0.9,
                    None,
                );
            }
        }
        FuzzOp::PromotePredicted => {
            let _ = state.resp_promote_predicted_to_created(dummy_runtime());
        }
        FuzzOp::CreateFromNone => {
            if matches!(state.session, SessionPhase::Active { .. })
                && !matches!(state.vad, VadPhase::Speaking { .. })
            {
                let _ = state.resp_create_from_none(
                    ResponseId::new(format!("resp_{idx}")),
                    ItemId::new(format!("item_{idx}")),
                    dummy_runtime(),
                );
            }
        }
        FuzzOp::AdvanceToStreaming => {
            let _ = state.resp_advance_to_streaming(Arc::new(AtomicU64::new(0)));
        }
        FuzzOp::Drain => {
            let _ = state.resp_drain(1500);
        }
        FuzzOp::RetireToNone => {
            let _ = state.resp_retire_to_none();
        }
        FuzzOp::RetirePredicted => {
            let _ = state.resp_retire_predicted();
        }
        FuzzOp::RetirePredictedFull => {
            let _ = state.resp_retire_predicted_full();
        }
        FuzzOp::StoreSealedBuffer => {
            let slot = idx % 8;
            let item_id = format!("buf_item_{slot}");
            let start = idx.wrapping_mul(50);
            state.store_sealed_buffer(SealedBuffer {
                item_id,
                audio: Vec::new(),
                audio_start_ms: start,
                audio_end_ms: start + 100,
            });
        }
        FuzzOp::DropSealedBuffer => {
            let slot = idx % 8;
            let item_id = format!("buf_item_{slot}");
            let _ = state.drop_sealed_buffer(&item_id);
        }
        FuzzOp::TruncateConversation => {
            let slot = idx % 8;
            let item_id = format!("buf_item_{slot}");
            apply_truncate_to_conversation(
                &mut state.conversation,
                &item_id,
                idx % 2_000,
                "fuzz transcript",
            );
        }
    }
}

pub fn run_random_walk(seed: u64, steps: u64) -> Result<(), (u64, FuzzOp, String)> {
    let mut state = SessionState::default();
    let mut rng = Lcg::new(seed);
    for i in 0..steps {
        let op = *rng.pick(ALL_OPS);
        apply(op, &mut state, i);
        if let Err(v) = check_state(&state) {
            return Err((i, op, format!("{v:?}")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::thread;

    #[test]
    fn fuzzer_5000_steps_seed_0() {
        run_random_walk(0, 5000).expect("invariant violation");
    }

    #[test]
    fn fuzzer_seed_diversity() {
        for seed in [1, 7, 42, 99, 2024] {
            run_random_walk(seed, 1000).unwrap_or_else(|(i, op, v)| {
                panic!("seed={seed} step={i} op={op:?}: {v}");
            });
        }
    }

    #[test]
    fn lcg_repeatable_for_same_seed() {
        let mut a = Lcg::new(42);
        let mut b = Lcg::new(42);
        for _ in 0..100 {
            assert_eq!(a.next(), b.next());
        }
    }

    #[test]
    fn concurrent_invariants_hold_8_workers_5000_ops() {
        let state = std::sync::Arc::new(StdMutex::new(SessionState::default()));
        let workers = 8_u64;
        let ops_per_worker = 5_000_u64;
        let base_seed: u64 = 0xC0FFEEBABEDEADu64;

        let mut handles = Vec::new();
        for w in 0..workers {
            let s = std::sync::Arc::clone(&state);
            let seed = base_seed ^ (w.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            handles.push(thread::spawn(move || {
                let mut rng = Lcg::new(seed);
                for i in 0..ops_per_worker {
                    let op = *rng.pick(ALL_OPS);
                    let mut g = s.lock().expect("state mutex poisoned");
                    let idx = w.wrapping_mul(1_000_000).wrapping_add(i);
                    apply(op, &mut g, idx);
                    if let Err(v) = check_state(&g) {
                        panic!(
                            "invariant violation under concurrent fuzz: \
                             worker={w} step={i} op={op:?} v={v:?}",
                        );
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }
    }

    #[test]
    fn concurrent_invariants_hold_multi_seed() {
        let mut seed_threads = Vec::new();
        for seed in [1_u64, 7, 42, 99, 2024, 31_337] {
            seed_threads.push(thread::spawn(move || {
                let state = std::sync::Arc::new(StdMutex::new(SessionState::default()));
                let mut workers = Vec::new();
                for w in 0..6_u64 {
                    let s = std::sync::Arc::clone(&state);
                    let worker_seed = seed.wrapping_mul(0x100000001b3).wrapping_add(w);
                    workers.push(thread::spawn(move || {
                        let mut rng = Lcg::new(worker_seed);
                        for i in 0..1_500_u64 {
                            let op = *rng.pick(ALL_OPS);
                            let mut g = s.lock().expect("state mutex poisoned");
                            let idx = w.wrapping_mul(1_000_000).wrapping_add(i);
                            apply(op, &mut g, idx);
                            if let Err(v) = check_state(&g) {
                                panic!(
                                    "invariant violation: seed={seed} worker={w} \
                                     step={i} op={op:?} v={v:?}",
                                );
                            }
                        }
                    }));
                }
                for w in workers {
                    w.join().expect("worker thread panicked");
                }
            }));
        }
        for t in seed_threads {
            t.join().expect("seed thread panicked");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_invariants_hold_tokio_8_tasks_5000_ops() {
        use tokio::sync::Mutex as TokioMutex;

        let state = std::sync::Arc::new(TokioMutex::new(SessionState::default()));
        let workers = 8_u64;
        let ops_per_worker = 5_000_u64;
        let base_seed: u64 = 0xFEED_FACE_CAFE_BEEFu64;

        let mut handles = Vec::new();
        for w in 0..workers {
            let s = std::sync::Arc::clone(&state);
            let seed = base_seed ^ (w.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            handles.push(tokio::spawn(async move {
                let mut rng = Lcg::new(seed);
                for i in 0..ops_per_worker {
                    let op = *rng.pick(ALL_OPS);
                    let mut g = s.lock().await;
                    let idx = w.wrapping_mul(1_000_000).wrapping_add(i);
                    apply(op, &mut g, idx);
                    if let Err(v) = check_state(&g) {
                        panic!(
                            "invariant violation under tokio fuzz: \
                             worker={w} step={i} op={op:?} v={v:?}",
                        );
                    }
                }
            }));
        }
        for h in handles {
            h.await.expect("worker task panicked or was cancelled");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn install_replace_abort_pattern_no_lost_or_double_mutation() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use tokio::sync::Mutex as TokioMutex;
        use tokio::task::JoinHandle;

        struct Slot {
            installed_gen: u64,
            handle: Option<JoinHandle<()>>,
            observed: Vec<u64>,
        }

        let slot = std::sync::Arc::new(TokioMutex::new(Slot {
            installed_gen: 0,
            handle: None,
            observed: Vec::new(),
        }));
        let next_gen = std::sync::Arc::new(AtomicU64::new(0));

        let rotators = 8_u64;
        let rounds_per_rotator = 256_u64;
        let mut rotator_handles = Vec::new();
        for _ in 0..rotators {
            let slot = slot.clone();
            let next_gen = next_gen.clone();
            rotator_handles.push(tokio::spawn(async move {
                for _ in 0..rounds_per_rotator {
                    let my_gen = next_gen.fetch_add(1, Ordering::SeqCst) + 1;
                    let slot_for_worker = slot.clone();
                    let worker = tokio::spawn(async move {
                        tokio::task::yield_now().await;
                        let mut g = slot_for_worker.lock().await;
                        g.observed.push(my_gen);
                    });
                    let mut g = slot.lock().await;
                    if let Some(prev) = g.handle.take() {
                        prev.abort();
                    }
                    g.installed_gen = my_gen;
                    g.handle = Some(worker);
                    drop(g);
                    tokio::task::yield_now().await;
                }
            }));
        }
        for h in rotator_handles {
            h.await.expect("rotator panicked");
        }

        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

        let (installed_gen, observed) = {
            let g = slot.lock().await;
            (g.installed_gen, g.observed.clone())
        };

        let total_installed = next_gen.load(Ordering::SeqCst);
        assert_eq!(
            installed_gen, total_installed,
            "final installed_gen != total fetch_adds (lock failed to serialize installs)",
        );

        let mut sorted = observed.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            observed.len(),
            "double-mutation: a worker's effects landed twice (observed={observed:?})",
        );

        for &g in &observed {
            assert!(
                g >= 1 && g <= total_installed,
                "phantom gen {g} not in installed range [1, {total_installed}]",
            );
        }

        assert!(
            !observed.is_empty(),
            "no worker ever ran -- test is not exercising the contended path",
        );

        eprintln!(
            "install-replace-abort: rounds={} observed={} final_gen={}",
            rotators * rounds_per_rotator,
            observed.len(),
            installed_gen,
        );
    }

    #[test]
    fn multi_session_no_cross_contamination() {
        let mut handles = Vec::new();
        for sess_idx in 0..16_u64 {
            handles.push(thread::spawn(move || {
                let mut state = SessionState::default();
                let mut rng = Lcg::new(0xDEAD_BEEF_u64.wrapping_add(sess_idx));
                for i in 0..1_500_u64 {
                    let op = *rng.pick(ALL_OPS);
                    let idx = sess_idx.wrapping_mul(1_000_000).wrapping_add(i);
                    apply(op, &mut state, idx);
                    if let Err(v) = check_state(&state) {
                        panic!(
                            "invariant violation in isolated session: \
                             sess={sess_idx} step={i} op={op:?} v={v:?}",
                        );
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("session thread panicked");
        }
    }
}
