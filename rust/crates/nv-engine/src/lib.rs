pub mod batch_engine;
pub mod batch_runtime;
pub mod block_manager;
pub mod kv;
pub mod scheduler;
pub mod sequence;

pub use batch_engine::{
    BatchEngine, BatchStepper, EngineEvent as BatchEvent, GenRequest, PrefillChunk, PrefillInput,
    SamplingConfig, SeqInput, StepResult,
};
pub use batch_runtime::BatchEngineHandle;
pub use block_manager::{Allocation, Block, BlockManager, CowCopy, PoolGeometry};
pub use kv::{checked_block_offset, KvKind, PagedKv};
pub use scheduler::{
    BatchKind, ScheduledBatch, ScheduledSeqItem, Scheduler, SchedulerConfig, StepFailure,
    VerifyOutcome,
};
pub use sequence::{FinishReason, Sequence, SequenceState};

#[derive(Debug)]
pub enum EngineRequest {
    Generate {
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
        reply: tokio::sync::mpsc::Sender<EngineEvent>,
    },
    Abort {
        seq_id: u64,
    },
}

#[derive(Debug, Clone)]
pub enum EngineEvent {
    Started { seq_id: u64 },
    Token { seq_id: u64, token: u32 },
    Done { seq_id: u64, reason: FinishReason },
    Error { seq_id: u64, message: String },
}
