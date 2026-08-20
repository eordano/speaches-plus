#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use nv_models::gemma4_wgpu as g4w;
mod common;
use common::LcgOddSeedShift32GaussUnit as Lcg;

const ENTRY: &str = "gather2_bf16";
const MK_ENTRY: &str = "gather2_bf16_mk";

const WHY: &str = "gather2_bf16 turns the previous token into the hidden state every layer of the \
     Gemma-4 dense graph then reads, and gather2_bf16_mk does the same for a whole prefill chunk. \
     Both are named by no test in the workspace. They exist as a PAIR rather than as one kernel \
     because the 31B embedding table does not fit one storage binding, so the vocabulary is split \
     across two buffers at split_row and the gather has to decide which half a token lives in and \
     rebase the row -- a decision no other kernel in the graph makes, and one whose defect is \
     DISCRETE: the wrong side of the split is the wrong token's embedding, not a small error, and \
     no tolerance on downstream logits is the instrument for that.";

const ORACLE_IS_NOT_THE_KERNEL: &str = "The reference is the definition of the lookup, written on \
     the host: clamp the index at zero, send anything at or past the vocabulary to row 0, and \
     copy row s from the low table when s < split_row and row s - split_row from the high table \
     otherwise. There is no arithmetic between input and output, so the comparison is BIT EXACT \
     and every case asserts the whole row.";

const THE_CORPUS_STRADDLES_THE_SPLIT: &str = "A corpus that only ever gathers from the low table \
     is the same hole as a fixture that never reaches a regime: the rebase of the high half, the \
     boundary at split_row itself, and the two out-of-range clamps would all be untested while \
     every case passed. So the corpus names each of them, and a screen asserts that both halves \
     and both clamps are actually exercised rather than assuming the token ids got there. The \
     M-row arm gives every position a DIFFERENT token, half of them on each side of the split -- \
     with one token repeated the destination stride is unobservable and the M-row kernel \
     degenerates into M copies of the decode one.";

const SLACK_WORDS: usize = 64;

const POISON: u32 = 0xdead_beef;

const SLACK_DOC: &str = "SLACK_WORDS of poison past the destination row is what makes the copy \
     loop's bound observable. Without slack, a gather that wrote one word too far would write out \
     of bounds and the robust-access rule would decide what happened, which is not a property a \
     test may assert.";

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_g4w_gather2_oracle needs a real wgpu adapter; a skipped numeric gate reads as a \
         passed one, so this panics rather than returning early",
    )
}

fn source() -> String {
    let src = g4w::glue_shader_source().to_string();
    for e in [ENTRY, MK_ENTRY] {
        assert!(
            src.contains(&format!("fn {e}(")),
            "gemma4_wgpu::glue_shader_source() no longer declares {e}; this gate compiles the \
             SHIPPED text and cannot fall back to a copy"
        );
    }
    src
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Gather2Params {
    split_row: u32,
    hidden_words: u32,
    vocab: u32,
    pad0: u32,
}

struct Table {
    label: &'static str,
    split_row: usize,
    vocab: usize,
    hidden_words: usize,
    lo: Vec<u32>,
    hi: Vec<u32>,
}

impl Table {
    fn build(label: &'static str, split_row: usize, vocab: usize, hw: usize, seed: u64) -> Self {
        assert!(split_row <= vocab, "{label}: split_row past the vocabulary");
        let mut rng = Lcg::new(seed);
        let mut fill = |rows: usize| -> Vec<u32> {
            (0..rows.max(1) * hw).map(|_| rng.next_u32()).collect()
        };
        let lo = fill(split_row);
        let hi = fill(vocab - split_row);
        Self {
            label,
            split_row,
            vocab,
            hidden_words: hw,
            lo,
            hi,
        }
    }

    fn row(&self, idx: i32) -> &[u32] {
        let mut s = idx.max(0) as usize;
        if s >= self.vocab {
            s = 0;
        }
        let hw = self.hidden_words;
        if s < self.split_row {
            &self.lo[s * hw..(s + 1) * hw]
        } else {
            let b = s - self.split_row;
            &self.hi[b * hw..(b + 1) * hw]
        }
    }

    fn params(&self) -> Gather2Params {
        Gather2Params {
            split_row: self.split_row as u32,
            hidden_words: self.hidden_words as u32,
            vocab: self.vocab as u32,
            pad0: 0,
        }
    }
}

fn run(
    ctx: &WgpuContext,
    src: &str,
    entry: &str,
    t: &Table,
    idx: &[i32],
) -> anyhow::Result<(Vec<Vec<u32>>, Vec<u32>)> {
    let hw = t.hidden_words;
    let m = idx.len();
    let total = m * hw + SLACK_WORDS;
    let lo = dispatch::storage_from_slice(ctx, "g2-lo", &t.lo);
    let hi = dispatch::storage_from_slice(ctx, "g2-hi", &t.hi);
    let ib = dispatch::storage_from_slice(ctx, "g2-idx", idx);
    let out = dispatch::storage_from_slice(ctx, "g2-out", &vec![POISON; total]);
    let p = dispatch::uniform_from(ctx, "g2-p", &t.params());
    let pipe = dispatch::compute_pipeline(ctx, "g4w-gather2-probe", src, entry)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let grid = if entry == MK_ENTRY {
        (m as u32, 1, 1)
    } else {
        (1, 1, 1)
    };
    dispatch::dispatch(
        ctx,
        &pipe,
        &[(3, &lo), (4, &hi), (5, &ib), (6, &out), (7, &p)],
        grid,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let words: Vec<u32> = dispatch::read_back(ctx, &out, total).map_err(|e| anyhow::anyhow!("{e}"))?;
    let rows = (0..m).map(|i| words[i * hw..(i + 1) * hw].to_vec()).collect();
    Ok((rows, words[m * hw..].to_vec()))
}

struct Corpus {
    tables: Vec<Table>,
    tokens: Vec<Vec<i32>>,
}

fn corpus() -> Corpus {
    let tables = vec![
        Table::build("split in the middle", 96, 160, 32, 0x0000_e1b0),
        Table::build("everything in the low table", 64, 64, 16, 0x0000_105e),
        Table::build("everything in the high table", 0, 48, 16, 0x0000_1015),
    ];
    let tokens = vec![
        vec![0, 95, 96, 159, 160, -1, 7, 130],
        vec![0, 63, 64, -3, 17, 41, 2, 55],
        vec![0, 1, 47, 48, -9, 12, 33, 20],
    ];
    let c = Corpus { tables, tokens };
    c.screen();
    c
}

impl Corpus {
    fn screen(&self) {
        let mut low = 0usize;
        let mut high = 0usize;
        let mut over = 0usize;
        let mut negative = 0usize;
        let mut boundary = 0usize;
        for (t, ids) in self.tables.iter().zip(self.tokens.iter()) {
            let mut distinct = std::collections::BTreeSet::new();
            for id in ids {
                assert!(
                    distinct.insert(*id),
                    "{}: token {id} appears twice, so a collapsed destination stride in the M-row \
                     arm could go unseen. {THE_CORPUS_STRADDLES_THE_SPLIT}",
                    t.label
                );
                if *id < 0 {
                    negative += 1;
                } else if *id as usize >= t.vocab {
                    over += 1;
                } else if (*id as usize) < t.split_row {
                    low += 1;
                } else {
                    high += 1;
                }
                if *id >= 0 && *id as usize == t.split_row {
                    boundary += 1;
                }
            }
        }
        for (what, n) in [
            ("a token in the low table", low),
            ("a token in the high table", high),
            ("a token at or past the vocabulary", over),
            ("a negative token", negative),
            ("a token exactly at split_row", boundary),
        ] {
            assert!(
                n > 0,
                "the corpus never gathers {what}, so that branch of the split lookup is untested \
                 while every case passes. {THE_CORPUS_STRADDLES_THE_SPLIT}"
            );
        }
    }
}

const MUTANTS: [(&str, &str, &str); 5] = [
    (
        "split-boundary-lets-split_row-itself-read-the-low-table",
        "    if (s < g2_params.split_row) {",
        "    if (s <= g2_params.split_row) {",
    ),
    (
        "high-table-row-is-not-rebased-past-the-split",
        "        let base = (s - g2_params.split_row) * hw;",
        "        let base = s * hw;",
    ),
    (
        "out-of-range-token-lands-on-row-1-instead-of-row-0",
        "        s = 0u;\n",
        "        s = 1u;\n",
    ),
    (
        "mk-destination-token-stride-dropped",
        "    let dst = t * hw;",
        "    let dst = 0u;",
    ),
    (
        "mk-every-position-reads-the-first-token",
        "    var s = u32(max(g2_idx[t], 0));",
        "    var s = u32(max(g2_idx[0], 0));",
    ),
];

const DECODE_MUTANTS: [usize; 3] = [0, 1, 2];
const MK_MUTANTS: [usize; 5] = [0, 1, 2, 3, 4];

fn mutate(src: &str, from: &str, to: &str) -> String {
    assert!(
        src.contains(from),
        "mutation anchor not present in the shipped glue source: {from:?}. This gate is worthless \
         if its mutants no longer apply -- fix the anchors."
    );
    src.replace(from, to)
}

#[test]
fn every_mutation_anchor_still_applies_to_the_shipped_glue() {
    let src = source();
    for (name, from, to) in MUTANTS {
        assert!(
            src.contains(from),
            "anchor for mutant {name} is gone from glue_shader_source(): {from:?}. A mutant whose \
             anchor rotted is silently inert, and the GPU tests that would have caught that do \
             not run on a box with no adapter -- which is why this check is CPU-only and \
             unconditional."
        );
        assert_ne!(
            from, to,
            "mutant {name} replaces text with itself and can never turn anything red"
        );
    }
    let _ = corpus();
    eprintln!("{WHY}\n{ORACLE_IS_NOT_THE_KERNEL}\n{THE_CORPUS_STRADDLES_THE_SPLIT}\n{SLACK_DOC}");
}

#[test]
fn gather2_bf16_matches_the_split_table_lookup() {
    let ctx = ctx();
    eprintln!("[gather2-oracle] adapter: {}\n{WHY}", ctx.info.name);
    let src = source();
    let c = corpus();
    for (t, ids) in c.tables.iter().zip(c.tokens.iter()) {
        for id in ids {
            let (rows, slack) =
                run(ctx, &src, ENTRY, t, &[*id]).unwrap_or_else(|e| panic!("{}: {e}", t.label));
            assert_eq!(
                rows[0],
                t.row(*id),
                "[{}] {ENTRY} gathered the wrong row for token {id} (split_row={}, vocab={}). \
                 {ORACLE_IS_NOT_THE_KERNEL}",
                t.label,
                t.split_row,
                t.vocab
            );
            assert!(
                slack.iter().all(|w| *w == POISON),
                "[{}] {ENTRY} wrote past the hidden row for token {id}. {SLACK_DOC}",
                t.label
            );
        }
        eprintln!(
            "[{}] {ENTRY}: {} tokens gathered exactly, split_row={} vocab={}",
            t.label,
            ids.len(),
            t.split_row,
            t.vocab
        );
    }
}

#[test]
fn gather2_bf16_mk_matches_the_split_table_lookup_for_every_position() {
    let ctx = ctx();
    let src = source();
    let c = corpus();
    for (t, ids) in c.tables.iter().zip(c.tokens.iter()) {
        let (rows, slack) = run(ctx, &src, MK_ENTRY, t, ids)
            .unwrap_or_else(|e| panic!("{}: {e}", t.label));
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(
                rows[i],
                t.row(*id),
                "[{}] {MK_ENTRY} gathered the wrong row at position {i} for token {id}. \
                 {THE_CORPUS_STRADDLES_THE_SPLIT}",
                t.label
            );
        }
        assert!(
            slack.iter().all(|w| *w == POISON),
            "[{}] {MK_ENTRY} wrote past the last position's row. {SLACK_DOC}",
            t.label
        );
        eprintln!(
            "[{}] {MK_ENTRY}: {} positions gathered exactly",
            t.label,
            ids.len()
        );
    }
}

#[test]
fn every_gather2_mutant_is_caught_by_this_corpus() {
    let ctx = ctx();
    let src = source();
    let c = corpus();
    for (entry, list) in [
        (ENTRY, DECODE_MUTANTS.as_slice()),
        (MK_ENTRY, MK_MUTANTS.as_slice()),
    ] {
        for i in list {
            let (name, from, to) = MUTANTS[*i];
            let bad = mutate(&src, from, to);
            let mut caught_by: Vec<String> = Vec::new();
            for (t, ids) in c.tables.iter().zip(c.tokens.iter()) {
                if entry == ENTRY {
                    for id in ids {
                        let hit = match run(ctx, &bad, entry, t, &[*id]) {
                            Ok((rows, slack)) => {
                                rows[0] != t.row(*id) || slack.iter().any(|w| *w != POISON)
                            }
                            Err(_) => true,
                        };
                        if hit {
                            caught_by.push(format!("{}/token {id}", t.label));
                        }
                    }
                    continue;
                }
                let hit = match run(ctx, &bad, entry, t, ids) {
                    Ok((rows, slack)) => {
                        ids.iter().enumerate().any(|(i, id)| rows[i] != t.row(*id))
                            || slack.iter().any(|w| *w != POISON)
                    }
                    Err(_) => true,
                };
                if hit {
                    caught_by.push(t.label.to_string());
                }
            }
            assert!(
                !caught_by.is_empty(),
                "[{entry}] mutant {name} was NOT caught by any case. A split-table lookup that a \
                 wrong branch passes is not a gate. {WHY} {THE_CORPUS_STRADDLES_THE_SPLIT}"
            );
            eprintln!("[{entry}] MUTANT {name}: caught by {caught_by:?}");
        }
    }
}
