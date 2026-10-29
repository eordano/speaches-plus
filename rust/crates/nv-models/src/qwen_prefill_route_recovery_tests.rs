use super::*;
use anyhow::ensure;
use std::cell::Cell;

thread_local! {
    static FAIL_REPLACEMENT: Cell<bool> = const { Cell::new(false) };
}

const INJECTED: &str = "injected Qwen prefill replacement failure after graph invalidation";

pub(super) fn fail_replacement_if_armed() -> Result<()> {
    ensure!(!FAIL_REPLACEMENT.with(|flag| flag.replace(false)), INJECTED);
    Ok(())
}
