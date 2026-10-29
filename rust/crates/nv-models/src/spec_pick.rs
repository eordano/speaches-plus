use anyhow::Result;

pub trait SpecPick {
    fn draft(&mut self, path: &[u32], row: &[f32]) -> u32;
    fn target(&mut self, path: &[u32], row: &[f32]) -> Result<u32>;
}

pub struct GreedyPick;

pub fn first_argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

impl SpecPick for GreedyPick {
    fn draft(&mut self, _: &[u32], row: &[f32]) -> u32 {
        first_argmax(row)
    }

    fn target(&mut self, _: &[u32], row: &[f32]) -> Result<u32> {
        Ok(first_argmax(row))
    }
}

pub fn accept_drafts(pick: &mut dyn SpecPick, drafts: &[u32], verify_rows: &[f32], vocab: usize) -> Result<(usize, u32)> {
    let mut accepted = 0usize;
    loop {
        let row = verify_rows.get(accepted * vocab..(accepted + 1) * vocab).ok_or_else(|| {
            anyhow::anyhow!("verify rows hold {} logits, row {accepted} of {vocab} is missing", verify_rows.len())
        })?;
        let target = pick.target(&drafts[..accepted], row)?;
        if accepted == drafts.len() || drafts[accepted] != target {
            return Ok((accepted, target));
        }
        accepted += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_acceptance_stops_at_the_first_disagreement_and_returns_the_target_there() {
        let rows = [0.0, 9.0, 0.0, 0.0, 0.0, 9.0, 9.0, 0.0, 0.0];
        assert_eq!(accept_drafts(&mut GreedyPick, &[1, 2], &rows, 3).unwrap(), (2, 0));
        assert_eq!(accept_drafts(&mut GreedyPick, &[1, 0], &rows, 3).unwrap(), (1, 2));
        assert_eq!(accept_drafts(&mut GreedyPick, &[2, 2], &rows, 3).unwrap(), (0, 1));
        let agree = [0.0, 9.0, 0.0, 0.0, 0.0, 9.0, 0.0, 9.0, 0.0];
        assert_eq!(accept_drafts(&mut GreedyPick, &[1, 2], &agree, 3).unwrap(), (2, 1));
        assert!(accept_drafts(&mut GreedyPick, &[1, 2], &agree[..6], 3).is_err());
    }
}
