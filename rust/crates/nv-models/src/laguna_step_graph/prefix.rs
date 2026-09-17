use anyhow::Result;

use super::StepCache;
use crate::{laguna, laguna_fp8};

pub enum TargetSlidingSnapshot {
    Bf16(laguna::prefix::SlidingSnapshot),
    FixedOneFp8(laguna_fp8::prefix::SlidingSnapshot),
}

pub enum TargetFullSnapshot {
    Bf16(laguna::prefix::FullSnapshot),
    FixedOneFp8(laguna_fp8::prefix::FullSnapshot),
}

impl TargetSlidingSnapshot {
    pub fn pos(&self) -> usize {
        match self {
            Self::Bf16(s) => s.pos,
            Self::FixedOneFp8(s) => s.pos(),
        }
    }
    pub fn bytes(&self) -> usize {
        match self {
            Self::Bf16(s) => s.bytes,
            Self::FixedOneFp8(s) => s.bytes(),
        }
    }
}

impl TargetFullSnapshot {
    pub fn pos(&self) -> usize {
        match self {
            Self::Bf16(s) => s.pos,
            Self::FixedOneFp8(s) => s.pos(),
        }
    }
    pub fn bytes(&self) -> usize {
        match self {
            Self::Bf16(s) => s.bytes,
            Self::FixedOneFp8(s) => s.bytes(),
        }
    }
}

impl StepCache {
    pub fn prefix_supported(&self) -> bool {
        match self {
            Self::Bf16(c) => c.prefix_supported(),
            Self::Fp8(c) => c.prefix_supported(),
        }
    }
    pub fn prefix_sliding_bytes(&self) -> Result<usize> {
        match self {
            Self::Bf16(c) => c.snapshot_sliding_bytes(),
            Self::Fp8(c) => c.snapshot_sliding_bytes(),
        }
    }
    pub fn prefix_full_bytes(&self, pos: usize) -> Result<usize> {
        match self {
            Self::Bf16(c) => c.snapshot_full_bytes(pos),
            Self::Fp8(c) => c.snapshot_full_bytes(pos),
        }
    }
    pub fn prefix_snapshot_sliding(&self) -> Result<TargetSlidingSnapshot> {
        match self {
            Self::Bf16(c) => c.snapshot_sliding().map(TargetSlidingSnapshot::Bf16),
            Self::Fp8(c) => c.snapshot_sliding().map(TargetSlidingSnapshot::FixedOneFp8),
        }
    }
    pub fn prefix_snapshot_full(&self, pos: usize) -> Result<TargetFullSnapshot> {
        match self {
            Self::Bf16(c) => c.snapshot_full(pos).map(TargetFullSnapshot::Bf16),
            Self::Fp8(c) => c.snapshot_full(pos).map(TargetFullSnapshot::FixedOneFp8),
        }
    }
    pub fn prefix_holds_full_rows(&self, pos: usize) -> bool {
        match self {
            Self::Bf16(c) => c.holds_full_rows(pos),
            Self::Fp8(c) => c.holds_full_rows(pos),
        }
    }
    pub fn prefix_rewind_to(
        &mut self,
        sliding: &TargetSlidingSnapshot,
        full: Option<&TargetFullSnapshot>,
    ) -> Result<()> {
        match (self, sliding, full) {
            (Self::Bf16(c), TargetSlidingSnapshot::Bf16(s), None) => c.rewind_to(s, None),
            (Self::Bf16(c), TargetSlidingSnapshot::Bf16(s), Some(TargetFullSnapshot::Bf16(f))) => {
                c.rewind_to(s, Some(f))
            }
            (Self::Fp8(c), TargetSlidingSnapshot::FixedOneFp8(s), None) => c.rewind_to(s, None),
            (
                Self::Fp8(c),
                TargetSlidingSnapshot::FixedOneFp8(s),
                Some(TargetFullSnapshot::FixedOneFp8(f)),
            ) => c.rewind_to(s, Some(f)),
            _ => anyhow::bail!("prefix snapshot precision does not match target cache storage"),
        }
    }
}
