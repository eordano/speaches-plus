use crate::batch_engine::{
    BatchEngine, BatchStepper, EngineEvent, GenRequest, PrefillChunk, PrefillInput, SeqInput,
    StepResult,
};
use crate::scheduler::SchedulerConfig;
use anyhow::Result;
use std::thread::JoinHandle;
use std::time::Duration;
use tokio::sync::mpsc;

impl BatchStepper for Box<dyn BatchStepper> {
    fn on_admit(&mut self, seq_id: u64, sampling: &crate::batch_engine::SamplingConfig) {
        (**self).on_admit(seq_id, sampling)
    }

    fn reuses_cached_prefix(&self) -> bool {
        (**self).reuses_cached_prefix()
    }

    fn prefill(&mut self, items: &[PrefillInput]) -> Result<Vec<StepResult>> {
        (**self).prefill(items)
    }

    fn decode(&mut self, items: &[SeqInput]) -> Result<Vec<StepResult>> {
        (**self).decode(items)
    }

    fn step_mixed(
        &mut self,
        chunks: &[PrefillChunk],
        decodes: &[SeqInput],
    ) -> Result<Vec<StepResult>> {
        (**self).step_mixed(chunks, decodes)
    }

    fn release(&mut self, seq_id: u64) {
        (**self).release(seq_id)
    }
}

enum Command {
    Submit(GenRequest),
    Abort(u64),
    Shutdown,
}

pub struct BatchEngineHandle {
    tx: mpsc::UnboundedSender<Command>,
    join: Option<JoinHandle<()>>,
}

impl BatchEngineHandle {
    pub fn spawn<S, F>(config: SchedulerConfig, idle_sleep: Duration, build: F) -> Self
    where
        S: BatchStepper + 'static,
        F: FnOnce() -> S + Send + 'static,
    {
        let (tx, mut rx) = mpsc::unbounded_channel::<Command>();
        let join = std::thread::Builder::new()
            .name("nv-batch-engine".into())
            .spawn(move || {
                let stepper = build();
                let mut engine = BatchEngine::new(config, stepper);
                run_loop(&mut engine, &mut rx, idle_sleep);
            })
            .expect("spawn nv-batch-engine thread");
        Self {
            tx,
            join: Some(join),
        }
    }

    pub fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        eos_token_ids: Vec<u32>,
        sampling: crate::batch_engine::SamplingConfig,
    ) -> mpsc::Receiver<EngineEvent> {
        let (reply, rx) = mpsc::channel(256);
        let _ = self.tx.send(Command::Submit(GenRequest {
            prompt_tokens,
            max_new_tokens,
            eos_token_ids,
            sampling,
            reply,
        }));
        rx
    }

    pub fn abort(&self, seq_id: u64) {
        let _ = self.tx.send(Command::Abort(seq_id));
    }
}

impl Drop for BatchEngineHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

fn run_loop<S: BatchStepper>(
    engine: &mut BatchEngine<S>,
    rx: &mut mpsc::UnboundedReceiver<Command>,
    idle_sleep: Duration,
) {
    loop {
        let mut shutdown = false;
        loop {
            match rx.try_recv() {
                Ok(Command::Submit(req)) => {
                    engine.admit(req);
                }
                Ok(Command::Abort(seq_id)) => {
                    engine.abort(seq_id);
                }
                Ok(Command::Shutdown) => {
                    shutdown = true;
                    break;
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    shutdown = true;
                    break;
                }
            }
        }
        if shutdown {
            break;
        }

        if engine.has_work() {
            if let Err(err) = engine.step() {
                let failed = engine.fail_all(&err.to_string());
                tracing::error!(
                    error = %err,
                    failed_sequences = failed,
                    "nv-batch-engine step failed; failed all in-flight sequences"
                );
            }
        } else {
            match rx.blocking_recv() {
                Some(Command::Submit(req)) => {
                    engine.admit(req);
                }
                Some(Command::Abort(seq_id)) => engine.abort(seq_id),
                Some(Command::Shutdown) | None => break,
            }
            if idle_sleep > Duration::ZERO {
                std::thread::sleep(idle_sleep);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_engine::SamplingConfig;
    use crate::sequence::FinishReason;

    struct CountingStepper {
        next: u32,
    }

    impl BatchStepper for CountingStepper {
        fn prefill(&mut self, items: &[PrefillInput]) -> Result<Vec<StepResult>> {
            Ok(items
                .iter()
                .map(|it| {
                    let token = self.next;
                    self.next += 1;
                    StepResult {
                        seq_id: it.seq_id,
                        token,
                    }
                })
                .collect())
        }

        fn decode(&mut self, items: &[SeqInput]) -> Result<Vec<StepResult>> {
            Ok(items
                .iter()
                .map(|it| {
                    let token = self.next;
                    self.next += 1;
                    StepResult {
                        seq_id: it.seq_id,
                        token,
                    }
                })
                .collect())
        }
    }

    struct PoisonedStepper;

    impl BatchStepper for PoisonedStepper {
        fn prefill(&mut self, _items: &[PrefillInput]) -> Result<Vec<StepResult>> {
            anyhow::bail!("null-stream sync before graph step: DriverError(CUDA_ERROR_UNKNOWN)")
        }
        fn decode(&mut self, _items: &[SeqInput]) -> Result<Vec<StepResult>> {
            anyhow::bail!("null-stream sync before graph step: DriverError(CUDA_ERROR_UNKNOWN)")
        }
    }

    #[test]
    fn a_failing_step_reports_to_the_client_instead_of_spinning() {
        let mut engine = BatchEngine::new(
            SchedulerConfig {
                max_batch_size: 4,
                max_batched_tokens: 4096,
                block_size: 4,
                num_blocks: 64,
            },
            PoisonedStepper,
        );
        let (ctx, mut crx) = mpsc::unbounded_channel();
        let (tx, mut rx) = mpsc::channel(64);
        ctx.send(Command::Submit(GenRequest {
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 8,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx,
        }))
        .unwrap();

        let h = std::thread::spawn(move || run_loop(&mut engine, &mut crx, Duration::ZERO));

        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut got_error = None;
        while std::time::Instant::now() < deadline {
            match rx.try_recv() {
                Ok(EngineEvent::Error { seq_id, message }) => {
                    got_error = Some((seq_id, message));
                    break;
                }
                Ok(_) => continue,
                Err(_) => std::thread::sleep(Duration::from_millis(10)),
            }
        }

        let (seq_id, message) = got_error.expect(
            "no Error event within 10s: the runtime is spinning on a failed step \
             instead of failing its sequences",
        );
        let _ = seq_id;
        assert!(
            message.contains("DriverError"),
            "client should see the underlying driver error, got {message:?}"
        );

        ctx.send(Command::Shutdown).unwrap();
        let joined = std::thread::spawn(move || h.join());
        let start = std::time::Instant::now();
        while !joined.is_finished() && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            joined.is_finished(),
            "run_loop never returned: it did not park in blocking_recv after the failure"
        );
    }

    #[test]
    fn boxed_stepper_runs_to_completion() {
        let boxed: Box<dyn BatchStepper> = Box::new(CountingStepper { next: 1000 });
        let mut engine = BatchEngine::new(
            SchedulerConfig {
                max_batch_size: 4,
                max_batched_tokens: 4096,
                block_size: 4,
                num_blocks: 64,
            },
            boxed,
        );
        let (tx, mut rx) = mpsc::channel(64);
        engine.admit(GenRequest {
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 3,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx,
        });
        let mut guard = 0;
        while engine.has_work() && guard < 100 {
            engine.step().unwrap();
            guard += 1;
        }
        let mut tokens = 0;
        let mut done = false;
        while let Ok(ev) = rx.try_recv() {
            match ev {
                EngineEvent::Token { .. } => tokens += 1,
                EngineEvent::Done {
                    reason,
                    completion_tokens,
                    ..
                } => {
                    assert_eq!(reason, FinishReason::MaxTokens);
                    assert_eq!(completion_tokens, 3);
                    done = true;
                }
                _ => {}
            }
        }
        assert_eq!(tokens, 3);
        assert!(done);
    }
}
