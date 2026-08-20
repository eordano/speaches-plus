#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum IntegratedVerdictAction {
    Ignored,
    None,
    StartedPredicted,
    Commit,
}
