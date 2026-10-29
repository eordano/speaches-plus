use super::*;
use crate::qwen3_mm_splice::{mrope_state_table_head_and_tail_start, Qwen3MropePositions};

pub const PER_STATE_ROPE_TABLE_BECAUSE_MROPE_ROWS_DIFFER_PER_PROMPT_AND_CAPTURED_GRAPHS_BAKE_ITS_ADDRESS: &str =
    "every Qwen4ExpState owns a [max_len, rotary/2] cos/sin table indexed by token index; text requests \
     hold the model's base rows (a copy, so text logits are bit-identical), a multimodal request writes \
     the interleaved t/h/w rows for its prompt and base[i + rope_delta] after it. Prefill, decode, MTP, \
     verify and the QSA indexer's block keys all read row = token index, which is exactly HF's \
     full_cos[token] semantics, and the decode/verify/draft graphs captured per state keep pointing at \
     the same table when its contents change";

pub(super) struct StateMm {
    pub(super) rope: Rope,
    standard: bool,
    rows: Vec<(usize, Tensor)>,
    key: Option<Vec<u32>>,
}

impl StateMm {
    pub(super) fn new(base: &Rope, max_len: usize) -> Result<Self> {
        let mut cfg = *base.config();
        cfg.max_seq_len = max_len;
        let half = cfg.head_dim / 2;
        let cos = Tensor::zeros((max_len, half), DType::F32, base.cos().device())?;
        let sin = Tensor::zeros((max_len, half), DType::F32, base.sin().device())?;
        cos.slice_set(&base.cos().narrow(0, 0, max_len)?, 0, 0)?;
        sin.slice_set(&base.sin().narrow(0, 0, max_len)?, 0, 0)?;
        Ok(Self {
            rope: Rope::from_precomputed_tables_one_row_per_token_so_mrope_rides_the_standard_apply(cfg, cos, sin)?,
            standard: true,
            rows: Vec::new(),
            key: None,
        })
    }

    pub(super) fn prefill_done(&mut self) {
        self.rows.clear();
        self.key = None;
    }

    fn max_len(&self) -> usize {
        self.rope.config().max_seq_len
    }
}

impl Qwen4ExpState {
    pub fn rope_table(&self) -> (&Tensor, &Tensor) {
        (self.mm.rope.cos(), self.mm.rope.sin())
    }
}

pub struct Qwen4ExpMm {
    pub positions: Qwen3MropePositions,
    pub section: [usize; 3],
    pub rows: Vec<(usize, Tensor)>,
    pub key: Vec<u32>,
}

impl Qwen4Exp {
    pub fn state_install_mm(&self, state: &mut Qwen4ExpState, mm: Option<&Qwen4ExpMm>) -> Result<()> {
        let _ = PER_STATE_ROPE_TABLE_BECAUSE_MROPE_ROWS_DIFFER_PER_PROMPT_AND_CAPTURED_GRAPHS_BAKE_ITS_ADDRESS;
        let rows = state.mm.max_len();
        let st = &mut state.mm;
        st.rows.clear();
        st.key = None;
        let Some(mm) = mm else {
            if !st.standard {
                st.rope.cos().slice_set(&self.rope.cos().narrow(0, 0, rows)?, 0, 0)?;
                st.rope.sin().slice_set(&self.rope.sin().narrow(0, 0, rows)?, 0, 0)?;
                st.standard = true;
            }
            return Ok(());
        };
        let n = mm.positions.len();
        ensure!(mm.key.len() == n, "prefix key has {} ids for a {n}-token prompt", mm.key.len());
        let h = self.cfg.hidden_size;
        for (p, r) in &mm.rows {
            let (k, w) = r.dims2()?;
            ensure!(w == h && p + k <= n, "vision rows at {p} ({k}x{w}) do not fit a {n}-token prompt of width {h}");
        }
        let (cos, sin, tail) = mrope_state_table_head_and_tail_start(&self.rope, &mm.positions, mm.section, rows)?;
        st.rope.cos().slice_set(&cos, 0, 0)?;
        st.rope.sin().slice_set(&sin, 0, 0)?;
        if rows > n {
            st.rope.cos().slice_set(&self.rope.cos().narrow(0, tail, rows - n)?, 0, n)?;
            st.rope.sin().slice_set(&self.rope.sin().narrow(0, tail, rows - n)?, 0, n)?;
        }
        st.standard = mm.positions.is_text_degenerate() && tail == n;
        st.rows = mm.rows.iter().map(|(p, r)| Ok((*p, r.to_dtype(DType::BF16)?.contiguous()?))).collect::<Result<_>>()?;
        st.key = Some(mm.key.clone());
        Ok(())
    }

    pub fn session_install_mm(&self, sess: &mut Qwen4ExpSession, mm: Option<&Qwen4ExpMm>) -> Result<()> {
        self.state_install_mm(&mut sess.state, mm)
    }

    pub(super) fn mm_prefix_key(&self, state: &Qwen4ExpState, prompt: &[u32]) -> Vec<u32> {
        match &state.mm.key {
            Some(k) if k.len() == prompt.len() => k.clone(),
            _ => prompt.to_vec(),
        }
    }

    pub(super) fn mm_splice_rows(&self, state: &Qwen4ExpState, x: Tensor, first: usize, t: usize, copies: usize) -> Result<Tensor> {
        let h = self.cfg.hidden_size;
        for (p, r) in &state.mm.rows {
            let k = r.dim(0)?;
            let (lo, hi) = ((*p).max(first), (p + k).min(first + t));
            if lo >= hi {
                continue;
            }
            let part = r.narrow(0, lo - p, hi - lo)?;
            let rep = if copies == 1 { part } else { Tensor::cat(&vec![&part; copies], 1)? };
            ensure!(rep.dim(1)? == copies * h, "vision rows width {} for {copies} streams of {h}", rep.dim(1)?);
            x.slice_set(&rep.contiguous()?, 0, lo - first)?;
        }
        Ok(x)
    }
}
