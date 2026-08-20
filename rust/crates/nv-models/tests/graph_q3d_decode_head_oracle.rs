#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::{dispatch, WgpuContext};

const MISC_TAG: &str = "q3d:misc";
const GATHER_ENTRY: &str = "q3w_gather_embed";
const AM1_ENTRY: &str = "q3w_argmax_stage1";
const AM2_ENTRY: &str = "q3w_argmax_stage2";
const REDUCE_HELPER: &str = "am_reduce";

const INDEPENDENCE_IS_A_BODY_LEVEL_CLAIM: &str = "A shared LINE proves nothing either way: \
     neighbouring entries in this graph spell the same idiom the same way, and \
     graph_q3d_delta_decode_oracle found `let silu = acc / (1.0 + exp(-acc));` character-identical \
     in a decode entry and its prefill twin. What is true, and what \
     the_decode_bodies_gated_here_are_no_part_of_the_prefill_source checks, is that the whole \
     FUNCTION BODY of each entry gated here is absent from shipped_prefill_source(). The same \
     check pins every mutant's anchor inside the bodies this suite has a corpus for -- q3w_argmax_\
     stage1 and q3w_argmax_stage2 do share the am_reduce helper, so it is gated with them and \
     counted with them, while a mutant whose anchor also reached q3w_silu_mul would be reporting \
     another kernel's redness as this one's.";

const WHY: &str = "The two ends of a Qwen3.5-dense decode step: q3w_gather_embed turns the \
     previous token into the hidden state every layer then reads, and q3w_argmax_stage1 plus \
     q3w_argmax_stage2 turn the logits back into the token that is emitted and fed round again. \
     All three run once per token, at q3d-gather, q3d-am1 and q3d-am2, and no test in the \
     workspace named any of them. The gather has an M-row prefill twin gated by \
     graph_q3d_elementwise_oracle, but that is a gate on different text -- the twin lives in \
     PREFILL_WGSL and this one in MISC_WGSL, and \
     the_gated_entries_are_absent_from_the_prefill_source proves the twin gate cannot see a \
     mutation of this body. The two argmax entries have no twin at all: nothing else in this graph \
     resembles them, so they were ungated outright.";

const ARGMAX_IS_DISCRETE: &str = "A wrong argmax is a WRONG TOKEN, not a small error, so no \
     relative tolerance is the instrument for it: this suite asserts the INDEX, exactly. It also \
     asserts stage1's partial VALUES bit-exactly, because they are copies of the input and no \
     arithmetic stands between. The corpus therefore has to contain the shapes where a reduction \
     can pick the wrong index while staying numerically close: an EXACT TIE across two groups and \
     an EXACT TIE inside one lane's stride, where the shipped rule is that the LOWER ELEMENT INDEX \
     wins and reversing it moves the token without moving the value at all; and a NEAR TIE of one \
     f32 ulp inside a single bf16 bucket, where any reduction that carries less than f32 through \
     the compare returns the runner-up. \
     only_the_near_tie_case_sees_a_reduction_that_compares_in_bf16 pins that last one: rounding \
     every logit to bf16 before the compare perturbs every case's partial values and changes the \
     emitted token in exactly one of them.";

const SENTINEL_VALUE: f32 = -f32::MAX;
const SENTINEL_INDEX: u32 = 0xffff_ffff;
const PARTIAL_POISON_VALUE: f32 = 3.0e38;
const PARTIAL_POISON_INDEX: u32 = 0x00c0_ffee;
const TOKEN_POISON: u32 = 0x00ba_dbad;
const GATHER_POISON: u32 = 0xdead_beef;
const GATHER_SLACK_WORDS: usize = 256;
const AM_LANES: usize = 256;

const BOUNDS_DOC: &str = "SENTINEL_VALUE is the shipped stage1 seed spelled as an f32 constant: \
     -3.4028235e38 in the WGSL parses to exactly -f32::MAX, so a group with no elements is \
     required to report that bit pattern paired with SENTINEL_INDEX, and a corpus without an empty \
     group would not notice the seed changing. PARTIAL_POISON_VALUE is deliberately LARGER than \
     any logit in the corpus and is written into every partial slot past `groups`, so a stage2 \
     that reduces the whole 256-lane workgroup instead of the live groups returns \
     PARTIAL_POISON_INDEX rather than a plausible token. GATHER_SLACK_WORDS of poison past the \
     destination row is what makes the gather's tail guard observable: without slack, dropping it \
     writes out of bounds and a robust-access clamp decides what happens, which is not a property \
     a test may assert.";

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_q3d_decode_head_oracle needs a real wgpu adapter; a skipped numeric gate reads as a \
         passed one, so this panics rather than returning early",
    )
}

fn misc_source() -> String {
    let all = nv_models::qwen3_5_dense_wgpu::nozi_audit_sources();
    let hit = all
        .into_iter()
        .find(|(tag, _)| *tag == MISC_TAG)
        .unwrap_or_else(|| {
            panic!(
                "qwen3_5_dense_wgpu::nozi_audit_sources() no longer exposes {MISC_TAG}; this gate \
                 compiles the SHIPPED text and cannot fall back to a copy"
            )
        });
    let src = hit.1;
    for e in [GATHER_ENTRY, AM1_ENTRY, AM2_ENTRY] {
        assert!(
            src.contains(&format!("fn {e}(")),
            "{MISC_TAG} no longer declares {e}; the entry moved and this gate is now testing \
             nothing"
        );
    }
    src
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 32) as u32) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn band(&mut self, n: usize, centre: f32, half: f32) -> Vec<f32> {
        (0..n).map(|_| centre + self.next() * half).collect()
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GeParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AmParams {
    n: u32,
    groups: u32,
    pad0: u32,
    pad1: u32,
}

struct GatherCase {
    label: &'static str,
    row_off: usize,
    n_rows: usize,
    vocab: usize,
    hidden_words: usize,
    token: i32,
}

fn gather_cases() -> Vec<GatherCase> {
    vec![
        GatherCase {
            label: "unsharded vocab32 hw300, token 3 (two workgroups wide)",
            row_off: 0,
            n_rows: 32,
            vocab: 32,
            hidden_words: 300,
            token: 3,
        },
        GatherCase {
            label: "shard rows 8..16 of vocab32, token 10 inside the shard",
            row_off: 8,
            n_rows: 8,
            vocab: 32,
            hidden_words: 130,
            token: 10,
        },
        GatherCase {
            label: "shard rows 8..16, token 2 below the shard",
            row_off: 8,
            n_rows: 8,
            vocab: 32,
            hidden_words: 130,
            token: 2,
        },
        GatherCase {
            label: "shard rows 8..16, token 40 out of vocabulary clamps to row 0, below the shard",
            row_off: 8,
            n_rows: 8,
            vocab: 32,
            hidden_words: 130,
            token: 40,
        },
        GatherCase {
            label: "shard rows 8..16, token -1 clamps to row 0, below the shard",
            row_off: 8,
            n_rows: 8,
            vocab: 32,
            hidden_words: 130,
            token: -1,
        },
        GatherCase {
            label: "shard rows 0..8 of vocab32, token 20 above the shard",
            row_off: 0,
            n_rows: 8,
            vocab: 32,
            hidden_words: 130,
            token: 20,
        },
        GatherCase {
            label: "shard rows 0..8, token 40 out of vocabulary clamps to row 0 inside the shard",
            row_off: 0,
            n_rows: 8,
            vocab: 32,
            hidden_words: 130,
            token: 40,
        },
        GatherCase {
            label: "shard rows 0..8, token -3 clamps to row 0 inside the shard",
            row_off: 0,
            n_rows: 8,
            vocab: 32,
            hidden_words: 130,
            token: -3,
        },
    ]
}

impl GatherCase {
    fn resolved(&self) -> usize {
        let s = if self.token > 0 {
            self.token as usize
        } else {
            0
        };
        if s >= self.vocab {
            0
        } else {
            s
        }
    }
    fn writes(&self) -> bool {
        let s = self.resolved();
        s >= self.row_off && s < self.row_off + self.n_rows
    }
    fn embed(&self) -> Vec<u32> {
        (0..self.n_rows * self.hidden_words)
            .map(|i| 0x0400_0000 + i as u32)
            .collect()
    }
    fn out_words(&self) -> usize {
        self.hidden_words + GATHER_SLACK_WORDS
    }
    fn reference(&self, emb: &[u32]) -> Vec<u32> {
        let mut want = vec![GATHER_POISON; self.out_words()];
        if self.writes() {
            let base = (self.resolved() - self.row_off) * self.hidden_words;
            want[..self.hidden_words].copy_from_slice(&emb[base..base + self.hidden_words]);
        }
        want
    }
}

fn run_gather(ctx: &WgpuContext, src: &str, c: &GatherCase) -> Result<Vec<u32>, String> {
    let emb = c.embed();
    let emb_b = dispatch::storage_from_slice(ctx, "ge-emb", &emb);
    let tok_b = dispatch::storage_from_slice(ctx, "ge-tok", &[c.token]);
    let out_b = dispatch::storage_from_slice(ctx, "ge-out", &vec![GATHER_POISON; c.out_words()]);
    let p_b = dispatch::uniform_from(
        ctx,
        "ge-p",
        &GeParams {
            row_off: c.row_off as u32,
            n_rows: c.n_rows as u32,
            hidden_words: c.hidden_words as u32,
            vocab: c.vocab as u32,
        },
    );
    let grid = dispatch::workgroup_count_1d(ctx, c.hidden_words as u64, 256);
    dispatch::run(
        ctx,
        "q3d-decode-head-oracle",
        src,
        GATHER_ENTRY,
        &[(30, &emb_b), (31, &tok_b), (32, &out_b), (33, &p_b)],
        grid,
    )
    .map_err(|e| format!("{e}"))?;
    dispatch::read_back(ctx, &out_b, c.out_words()).map_err(|e| format!("readback: {e}"))
}

#[test]
fn q3w_gather_embed_copies_only_the_row_this_shard_owns() {
    let ctx = ctx();
    eprintln!(
        "[q3d-decode-head-oracle] adapter: {}\n{WHY}\n{BOUNDS_DOC}",
        ctx.info.name
    );
    let src = misc_source();
    let mut wrote = 0usize;
    let mut skipped = 0usize;
    for c in gather_cases() {
        let emb = c.embed();
        let want = c.reference(&emb);
        let got = run_gather(ctx, &src, &c).unwrap_or_else(|e| panic!("{}: {e}", c.label));
        for w in 0..c.hidden_words {
            assert_eq!(
                got[w], want[w],
                "{}: {GATHER_ENTRY} disagrees at word {w} (0x{:08x} vs 0x{:08x}). Token {} \
                 resolves to row {} and this shard owns rows {}..{}, so the row {} be copied. \
                 Every embedding word in this corpus is distinct, so a row or a column offset off \
                 by one moves an identifiable value.",
                c.label,
                got[w],
                want[w],
                c.token,
                c.resolved(),
                c.row_off,
                c.row_off + c.n_rows,
                if c.writes() { "must" } else { "must NOT" }
            );
        }
        for w in c.hidden_words..c.out_words() {
            assert_eq!(
                got[w], GATHER_POISON,
                "{}: {GATHER_ENTRY} wrote past the destination row at word {w}. The dispatch is \
                 rounded up to whole 256-lane workgroups, so the lanes past hidden_words exist and \
                 only the tail guard stops them; on a sharded embedding those words belong to \
                 something else. {BOUNDS_DOC}",
                c.label
            );
        }
        if c.writes() {
            wrote += 1;
        } else {
            skipped += 1;
        }
    }
    assert!(
        wrote > 0 && skipped > 0,
        "the gather corpus gathered {wrote} rows and skipped {skipped}; without both, one \
         direction of the shard range check is untested"
    );
    eprintln!(
        "[q3d-decode-head-oracle] gather: {wrote} rows copied exactly, {skipped} correctly skipped"
    );
}

#[test]
fn the_gather_corpus_covers_every_token_disposition() {
    let mut below = false;
    let mut above = false;
    let mut clamped_into_the_shard = false;
    let mut nonpositive_into_the_shard = false;
    let mut inside_sharded = false;
    let mut multi_workgroup = false;
    for c in gather_cases() {
        multi_workgroup |= c.hidden_words > 256;
        let s = c.resolved();
        let out_of_vocab = c.token > 0 && c.token as usize >= c.vocab;
        clamped_into_the_shard |= out_of_vocab && c.writes();
        nonpositive_into_the_shard |= c.token <= 0 && c.writes();
        if c.row_off > 0 {
            below |= s < c.row_off;
            inside_sharded |= c.writes();
        }
        above |= s >= c.row_off + c.n_rows;
        if c.token < 0 {
            assert!(
                (c.token as u32) as usize >= c.vocab,
                "{}: this negative token cast to u32 is smaller than vocab {}. The `ge_tok[0] > 0` \
                 guard and the vocabulary clamp are redundant with each other on every negative \
                 token precisely because that cast overflows past any vocabulary, which is why no \
                 mutant of the guard appears in this suite -- deleting it changes nothing \
                 observable. That reasoning fails here and would need re-deriving",
                c.label,
                c.vocab
            );
        }
    }
    assert!(
        below && inside_sharded,
        "the sharded corpus must contain both a token this shard owns and one below its range; \
         without both, `if (s < ge_p.row_off) return;` is untested in one direction"
    );
    assert!(
        above,
        "no token lands above the shard's row range, so the upper bound check is untested"
    );
    assert!(
        clamped_into_the_shard,
        "no out-of-vocabulary token resolves to a row this shard OWNS. Only that case sees the \
         `if (s >= vocab) s = 0;` clamp: where the clamp's target row is outside the shard, \
         dropping the clamp merely trades one early return for another and nothing observable \
         changes"
    );
    assert!(
        nonpositive_into_the_shard,
        "no non-positive token resolves to a row this shard owns, so the fallback to row 0 is \
         untested in the direction that writes"
    );
    assert!(
        multi_workgroup,
        "no case has hidden_words > 256, so `wid.x * 256u + lid.x` never advances past its first \
         workgroup and the tail guard is untested"
    );
}

struct AmCase {
    label: &'static str,
    groups: usize,
    logits: Vec<f32>,
}

fn next_up(v: f32) -> f32 {
    f32::from_bits(v.to_bits() + 1)
}

const NEAR_TIE_LOW: f32 = 30.0;
const NEAR_TIE_LOW_AT: usize = 100;
const NEAR_TIE_HIGH_AT: usize = 2500;

fn am_cases() -> Vec<AmCase> {
    let mut out = Vec::new();

    let mut r = Lcg::new(0xa11a_0001);
    let mut logits = r.band(151936, 0.0, 8.0);
    logits[98765] = 40.0;
    out.push(AmCase {
        label: "vocab 151936, groups 256 (shipped shape)",
        groups: 256,
        logits,
    });

    let mut r = Lcg::new(0xa11a_0002);
    let mut logits = r.band(1000, 0.0, 8.0);
    logits[777] = 25.0;
    out.push(AmCase {
        label: "n 1000, groups 256, most groups empty",
        groups: 256,
        logits,
    });

    let mut r = Lcg::new(0xa11a_0003);
    let mut logits = r.band(4000, 0.0, 8.0);
    logits[3999] = 33.0;
    out.push(AmCase {
        label: "groups 1, max in the last element",
        groups: 1,
        logits,
    });

    let mut r = Lcg::new(0xa11a_0004);
    let mut logits = r.band(5000, 0.0, 8.0);
    logits[4321] = 21.0;
    out.push(AmCase {
        label: "groups 7, stage2 lanes past the group count idle",
        groups: 7,
        logits,
    });

    let mut r = Lcg::new(0xa11a_0005);
    let mut logits = r.band(4096, 0.0, 8.0);
    logits[300] = 17.0;
    logits[1200] = 17.0;
    out.push(AmCase {
        label: "exact tie across two groups, the lower index must win",
        groups: 8,
        logits,
    });

    let mut r = Lcg::new(0xa11a_0006);
    let mut logits = r.band(3000, 0.0, 8.0);
    logits[500] = 19.0;
    logits[1524] = 19.0;
    out.push(AmCase {
        label: "exact tie inside one lane's stride, the lower index must win",
        groups: 4,
        logits,
    });

    let mut r = Lcg::new(0xa11a_0007);
    let mut logits = r.band(4096, 0.0, 8.0);
    logits[NEAR_TIE_LOW_AT] = NEAR_TIE_LOW;
    logits[NEAR_TIE_HIGH_AT] = next_up(NEAR_TIE_LOW);
    out.push(AmCase {
        label: "near tie of one f32 ulp, the max at the higher index",
        groups: 8,
        logits,
    });

    let mut r = Lcg::new(0xa11a_0008);
    let mut logits = r.band(3000, -20.0, 10.0);
    logits[1234] = -9.5;
    out.push(AmCase {
        label: "every logit negative",
        groups: 16,
        logits,
    });

    out
}

impl AmCase {
    fn n(&self) -> usize {
        self.logits.len()
    }
    fn group_of(&self, i: usize) -> usize {
        (i / AM_LANES) % self.groups
    }
    fn stage1_reference(&self) -> (Vec<f32>, Vec<u32>) {
        let mut pv = vec![SENTINEL_VALUE; self.groups];
        let mut pi = vec![SENTINEL_INDEX; self.groups];
        for i in 0..self.n() {
            let g = self.group_of(i);
            let v = self.logits[i];
            if v > pv[g] || (v == pv[g] && (i as u32) < pi[g]) {
                pv[g] = v;
                pi[g] = i as u32;
            }
        }
        (pv, pi)
    }
    fn token_reference(&self) -> u32 {
        let (pv, pi) = self.stage1_reference();
        reduce_partials(&pv, &pi, self.groups)
    }
}

fn reduce_partials(pv: &[f32], pi: &[u32], groups: usize) -> u32 {
    let mut bv = SENTINEL_VALUE;
    let mut bi = SENTINEL_INDEX;
    for t in 0..groups.min(AM_LANES) {
        if pv[t] > bv || (pv[t] == bv && pi[t] < bi) {
            bv = pv[t];
            bi = pi[t];
        }
    }
    bi
}

struct AmRun {
    pv: Vec<f32>,
    pi: Vec<u32>,
    token: u32,
}

fn run_argmax_chain(ctx: &WgpuContext, src: &str, c: &AmCase) -> Result<AmRun, String> {
    let bits: Vec<u32> = c.logits.iter().map(|v| v.to_bits()).collect();
    let x_b = dispatch::storage_from_slice(ctx, "am-x", &bits);
    let pv_b = dispatch::storage_from_slice(ctx, "am-pv", &vec![PARTIAL_POISON_VALUE; AM_LANES]);
    let pi_b = dispatch::storage_from_slice(ctx, "am-pi", &vec![PARTIAL_POISON_INDEX; AM_LANES]);
    let out_b = dispatch::storage_from_slice(ctx, "am-out", &[TOKEN_POISON]);
    let p_b = dispatch::uniform_from(
        ctx,
        "am-p",
        &AmParams {
            n: c.n() as u32,
            groups: c.groups as u32,
            pad0: 0,
            pad1: 0,
        },
    );
    dispatch::run(
        ctx,
        "q3d-decode-head-oracle",
        src,
        AM1_ENTRY,
        &[(40, &x_b), (41, &pv_b), (42, &pi_b), (44, &p_b)],
        (c.groups as u32, 1, 1),
    )
    .map_err(|e| format!("stage1: {e}"))?;
    dispatch::run(
        ctx,
        "q3d-decode-head-oracle",
        src,
        AM2_ENTRY,
        &[(41, &pv_b), (42, &pi_b), (43, &out_b), (44, &p_b)],
        (1, 1, 1),
    )
    .map_err(|e| format!("stage2: {e}"))?;
    let pv: Vec<f32> = dispatch::read_back(ctx, &pv_b, AM_LANES).map_err(|e| format!("pv: {e}"))?;
    let pi: Vec<u32> = dispatch::read_back(ctx, &pi_b, AM_LANES).map_err(|e| format!("pi: {e}"))?;
    let token: Vec<u32> =
        dispatch::read_back(ctx, &out_b, 1).map_err(|e| format!("token: {e}"))?;
    Ok(AmRun {
        pv,
        pi,
        token: token[0],
    })
}

#[test]
fn q3w_argmax_stage1_partials_match_an_exact_host_reference() {
    let ctx = ctx();
    eprintln!("{ARGMAX_IS_DISCRETE}");
    let src = misc_source();
    let mut empty_groups = 0usize;
    for c in am_cases() {
        let (want_pv, want_pi) = c.stage1_reference();
        let got = run_argmax_chain(ctx, &src, &c).unwrap_or_else(|e| panic!("{}: {e}", c.label));
        for g in 0..c.groups {
            assert_eq!(
                got.pi[g], want_pi[g],
                "{}: {AM1_ENTRY} reported index {} for group {g}, not {}. The reference scans the \
                 whole logit vector and takes the lowest index attaining the group maximum, which \
                 is the shipped tie-break; an argmax that returns a different index returns a \
                 different TOKEN. {ARGMAX_IS_DISCRETE}",
                c.label, got.pi[g], want_pi[g]
            );
            assert_eq!(
                got.pv[g].to_bits(),
                want_pv[g].to_bits(),
                "{}: {AM1_ENTRY} reported value {} for group {g}, not {}. A partial is a COPY of a \
                 logit -- no arithmetic stands between the load and the store -- so the demand is \
                 bit-exactness and any deviation means the compare is not carrying f32. \
                 {BOUNDS_DOC}",
                c.label,
                got.pv[g],
                want_pv[g]
            );
            empty_groups += usize::from(want_pi[g] == SENTINEL_INDEX);
        }
        for g in c.groups..AM_LANES {
            assert_eq!(
                got.pv[g], PARTIAL_POISON_VALUE,
                "{}: {AM1_ENTRY} wrote partial slot {g}, past the {} groups it was dispatched \
                 with",
                c.label, c.groups
            );
        }
    }
    assert!(
        empty_groups > 0,
        "no group in this corpus is empty, so stage1's seed pair (SENTINEL_VALUE, SENTINEL_INDEX) \
         is never the answer and changing it would pass. {BOUNDS_DOC}"
    );
    eprintln!("[q3d-decode-head-oracle] stage1: {empty_groups} empty groups reported the sentinel");
}

#[test]
fn q3w_argmax_stage2_emits_the_index_the_two_stages_agree_on() {
    let ctx = ctx();
    let src = misc_source();
    for c in am_cases() {
        let want = c.token_reference();
        let got = run_argmax_chain(ctx, &src, &c).unwrap_or_else(|e| panic!("{}: {e}", c.label));
        assert_eq!(
            got.token, want,
            "{}: the shipped {AM1_ENTRY} -> {AM2_ENTRY} pair emitted token {}, not {}. The \
             reference is a host argmax over the same logits with the same lowest-index tie-break; \
             {ARGMAX_IS_DISCRETE}",
            c.label, got.token, want
        );
        let brute = (0..c.n())
            .fold((SENTINEL_VALUE, SENTINEL_INDEX), |(bv, bi), i| {
                if c.logits[i] > bv {
                    (c.logits[i], i as u32)
                } else {
                    (bv, bi)
                }
            })
            .1;
        assert_eq!(
            brute, want,
            "{}: the two-stage reference and a flat scan of the same logits disagree ({brute} vs \
             {want}); the reference is then modelling the shader's decomposition rather than the \
             operation, and it would move with the shader",
            c.label
        );
        eprintln!(
            "[q3d-decode-head-oracle] {}: n={} groups={} token={}",
            c.label,
            c.n(),
            c.groups,
            got.token
        );
    }
}

#[test]
fn the_argmax_corpus_carries_ties_near_ties_and_empty_groups() {
    let cases = am_cases();
    let mut exact_tie_across_groups = false;
    let mut exact_tie_within_a_lane = false;
    let mut near_tie = false;
    let mut all_negative = false;
    let mut empty_groups = false;
    let mut lane_visits_more_than_one_element = false;
    for c in &cases {
        let (pv, pi) = c.stage1_reference();
        empty_groups |= pi.iter().any(|i| *i == SENTINEL_INDEX);
        all_negative |= c.logits.iter().all(|v| *v < 0.0);
        lane_visits_more_than_one_element |= c.n() > c.groups * AM_LANES;
        let top = pv.iter().cloned().fold(SENTINEL_VALUE, f32::max);
        let at: Vec<usize> = (0..c.n()).filter(|i| c.logits[*i] == top).collect();
        if at.len() > 1 {
            let a = at[0];
            let b = at[1];
            if c.group_of(a) != c.group_of(b) {
                exact_tie_across_groups = true;
            }
            if c.group_of(a) == c.group_of(b) && a % AM_LANES == b % AM_LANES {
                exact_tie_within_a_lane = true;
            }
        }
        let mut sorted: Vec<f32> = c.logits.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        if sorted[0] != sorted[1] && half::bf16::from_f32(sorted[0]) == half::bf16::from_f32(sorted[1])
        {
            near_tie = true;
        }
    }
    assert!(
        exact_tie_across_groups,
        "no case has the maximum appearing at two indices in DIFFERENT groups, so stage2's \
         `am_i[o] < am_i[tid]` tie-break is untested and reversing it would pass. \
         {ARGMAX_IS_DISCRETE}"
    );
    assert!(
        exact_tie_within_a_lane,
        "no case has the maximum appearing twice inside ONE lane's stride, so stage1's `i < bi` \
         tie-break is untested and reversing it would pass. {ARGMAX_IS_DISCRETE}"
    );
    assert!(
        near_tie,
        "no case has its top two logits distinct in f32 but equal in bf16, so a reduction that \
         compares in anything less than f32 returns the same token as the shipped one and this \
         corpus cannot see it. {ARGMAX_IS_DISCRETE}"
    );
    assert!(
        all_negative,
        "no case has every logit negative, so a stage1 seeded with 0.0 instead of -f32::MAX would \
         still find the right maximum"
    );
    assert!(
        empty_groups,
        "no case leaves a group with no elements, so the sentinel path is untested"
    );
    assert!(
        lane_visits_more_than_one_element,
        "no case has n > groups * 256, so every lane loads exactly one element and stage1's strided \
         loop never iterates"
    );
    assert!(
        next_up(NEAR_TIE_LOW) > NEAR_TIE_LOW
            && half::bf16::from_f32(next_up(NEAR_TIE_LOW)) == half::bf16::from_f32(NEAR_TIE_LOW),
        "the planted near tie is no longer a near tie: {NEAR_TIE_LOW} and its f32 successor must \
         differ in f32 and coincide in bf16, or the mutant that compares in bf16 has nothing to \
         collapse"
    );
    assert!(
        NEAR_TIE_HIGH_AT > NEAR_TIE_LOW_AT,
        "the larger of the near-tie pair must sit at the HIGHER index, or a reduction that \
         collapses them still returns the right token through the lowest-index tie-break and the \
         case proves nothing"
    );
}

struct S2Case {
    label: &'static str,
    groups: usize,
    pv: Vec<f32>,
    pi: Vec<u32>,
}

fn s2_case(label: &'static str, live: &[(f32, u32)]) -> S2Case {
    let mut pv = vec![PARTIAL_POISON_VALUE; AM_LANES];
    let mut pi = vec![PARTIAL_POISON_INDEX; AM_LANES];
    for (j, (v, i)) in live.iter().enumerate() {
        pv[j] = *v;
        pi[j] = *i;
    }
    S2Case {
        label,
        groups: live.len(),
        pv,
        pi,
    }
}

fn s2_cases() -> Vec<S2Case> {
    let mut full: Vec<(f32, u32)> = (0..AM_LANES)
        .map(|j| (1.0 + (j % 37) as f32 * 0.25, (j * 593 + 11) as u32))
        .collect();
    full[200] = (99.0, 4242);
    vec![
        s2_case(
            "groups 5, one clear maximum",
            &[(1.0, 10), (3.0, 20), (2.0, 30), (7.5, 40), (0.5, 50)],
        ),
        s2_case(
            "groups 4, equal values, the lower stored index wins from a later group",
            &[(5.0, 10), (9.0, 900), (5.0, 30), (9.0, 40)],
        ),
        s2_case(
            "groups 3, two groups empty and carrying the stage1 sentinel",
            &[
                (SENTINEL_VALUE, SENTINEL_INDEX),
                (4.0, 77),
                (SENTINEL_VALUE, SENTINEL_INDEX),
            ],
        ),
        s2_case("groups 1", &[(2.5, 9)]),
        s2_case("groups 256, the full workgroup", &full),
    ]
}

fn run_stage2(ctx: &WgpuContext, src: &str, c: &S2Case) -> Result<u32, String> {
    let pv_b = dispatch::storage_from_slice(ctx, "s2-pv", &c.pv);
    let pi_b = dispatch::storage_from_slice(ctx, "s2-pi", &c.pi);
    let out_b = dispatch::storage_from_slice(ctx, "s2-out", &[TOKEN_POISON]);
    let p_b = dispatch::uniform_from(
        ctx,
        "s2-p",
        &AmParams {
            n: 0,
            groups: c.groups as u32,
            pad0: 0,
            pad1: 0,
        },
    );
    dispatch::run(
        ctx,
        "q3d-decode-head-oracle",
        src,
        AM2_ENTRY,
        &[(41, &pv_b), (42, &pi_b), (43, &out_b), (44, &p_b)],
        (1, 1, 1),
    )
    .map_err(|e| format!("{e}"))?;
    let token: Vec<u32> = dispatch::read_back(ctx, &out_b, 1).map_err(|e| format!("{e}"))?;
    Ok(token[0])
}

#[test]
fn q3w_argmax_stage2_reduces_synthetic_partials_the_way_the_definition_says() {
    let ctx = ctx();
    let src = misc_source();
    for c in s2_cases() {
        let want = reduce_partials(&c.pv, &c.pi, c.groups);
        let got = run_stage2(ctx, &src, &c).unwrap_or_else(|e| panic!("{}: {e}", c.label));
        assert_ne!(
            got, TOKEN_POISON,
            "{}: {AM2_ENTRY} left the destination untouched",
            c.label
        );
        assert_eq!(
            got, want,
            "{}: {AM2_ENTRY} emitted {got}, not {want}. Driven on synthetic partials this entry is \
             a pure reduction over `groups` (value, index) pairs keeping the largest value and, \
             among equal values, the smallest INDEX -- not the smallest lane. {ARGMAX_IS_DISCRETE}",
            c.label
        );
        eprintln!(
            "[q3d-decode-head-oracle] stage2 {}: groups={} token={got}",
            c.label, c.groups
        );
    }
}

const CHAIN: &str = "chain";
const STAGE2: &str = "stage2";
const GATHER: &str = "gather";

const BF16_COMPARE_MUTANT: (&str, &str, &str, &str) = (
    CHAIN,
    "argmax-stage1-rounds-the-logits-to-bf16-before-comparing",
    "let v = bitcast<f32>(am_x[i]);",
    "let v = bf16_decode(bf16_encode(bitcast<f32>(am_x[i])));",
);

const MUTANTS: [(&str, &str, &str, &str); 15] = [
    (
        CHAIN,
        "argmax-reduce-tie-prefers-the-higher-index",
        "am_v[o] == am_v[tid] && am_i[o] < am_i[tid]",
        "am_v[o] == am_v[tid] && am_i[o] > am_i[tid]",
    ),
    (
        CHAIN,
        "argmax-reduce-keeps-the-smaller-value",
        "if (am_v[o] > am_v[tid] ||",
        "if (am_v[o] < am_v[tid] ||",
    ),
    (
        CHAIN,
        "argmax-initial-best-is-zero-instead-of-minus-max",
        "var bv = -3.4028235e38;",
        "var bv = 0.0;",
    ),
    (
        CHAIN,
        "argmax-stage1-lane-stride-ignores-the-group-count",
        "for (var i = g * 256u + tid; i < am_p.n; i = i + am_p.groups * 256u) {",
        "for (var i = g * 256u + tid; i < am_p.n; i = i + 256u) {",
    ),
    (
        CHAIN,
        "argmax-stage1-lane-tie-prefers-the-later-index",
        "if (v > bv || (v == bv && i < bi)) {",
        "if (v > bv || (v == bv && i > bi)) {",
    ),
    BF16_COMPARE_MUTANT,
    (
        CHAIN,
        "argmax-stage1-partial-value-not-reported",
        "am_pv[g] = am_v[0];",
        "am_pv[g] = 0.0;",
    ),
    (
        STAGE2,
        "argmax-stage2-reduces-past-the-group-count",
        "if (tid < am_p.groups) {",
        "if (tid < 256u) {",
    ),
    (
        STAGE2,
        "argmax-stage2-emits-the-value-instead-of-the-index",
        "am_out[0] = am_i[0];",
        "am_out[0] = bitcast<u32>(am_v[0]);",
    ),
    (
        GATHER,
        "gather-vocabulary-clamp-never-fires",
        "if (s >= ge_p.vocab) {",
        "if (s >= ge_p.vocab + 4096u) {",
    ),
    (
        GATHER,
        "gather-row-offset-not-subtracted",
        "let base = (s - ge_p.row_off) * ge_p.hidden_words;",
        "let base = s * ge_p.hidden_words;",
    ),
    (
        GATHER,
        "gather-lower-shard-bound-never-fires",
        "if (s < ge_p.row_off) {",
        "if (s + 4096u < ge_p.row_off) {",
    ),
    (
        GATHER,
        "gather-upper-shard-bound-never-fires",
        "if (s >= ge_p.row_off + ge_p.n_rows) {",
        "if (s >= ge_p.row_off + ge_p.n_rows + 4096u) {",
    ),
    (
        GATHER,
        "gather-tail-guard-never-fires",
        "if (w >= ge_p.hidden_words) {",
        "if (w >= ge_p.hidden_words + 4096u) {",
    ),
    (
        GATHER,
        "gather-column-offset-dropped",
        "ge_out[w] = ge_emb[base + w];",
        "ge_out[w] = ge_emb[base];",
    ),
];

fn mutate(src: &str, from: &str, to: &str) -> String {
    assert!(
        src.contains(from),
        "mutation anchor not present in the shipped misc source: {from:?}. This gate is worthless \
         if its mutants no longer apply -- fix the anchors."
    );
    src.replace(from, to)
}

fn chain_disagrees(ctx: &WgpuContext, src: &str) -> Vec<&'static str> {
    let mut caught = Vec::new();
    for c in am_cases() {
        let (want_pv, want_pi) = c.stage1_reference();
        let hit = match run_argmax_chain(ctx, src, &c) {
            Err(_) => true,
            Ok(run) => {
                run.token != c.token_reference()
                    || (0..c.groups).any(|g| {
                        run.pi[g] != want_pi[g] || run.pv[g].to_bits() != want_pv[g].to_bits()
                    })
            }
        };
        if hit {
            caught.push(c.label);
        }
    }
    caught
}

fn stage2_disagrees(ctx: &WgpuContext, src: &str) -> Vec<&'static str> {
    let mut caught = Vec::new();
    for c in s2_cases() {
        let want = reduce_partials(&c.pv, &c.pi, c.groups);
        let hit = match run_stage2(ctx, src, &c) {
            Err(_) => true,
            Ok(got) => got != want,
        };
        if hit {
            caught.push(c.label);
        }
    }
    caught
}

fn gather_disagrees(ctx: &WgpuContext, src: &str) -> Vec<&'static str> {
    let mut caught = Vec::new();
    for c in gather_cases() {
        let want = c.reference(&c.embed());
        let hit = match run_gather(ctx, src, &c) {
            Err(_) => true,
            Ok(got) => got != want,
        };
        if hit {
            caught.push(c.label);
        }
    }
    caught
}

#[test]
fn every_decode_head_mutant_is_caught_by_this_corpus() {
    let ctx = ctx();
    let src = misc_source();
    for (surface, name, from, to) in MUTANTS {
        let bad = mutate(&src, from, to);
        let caught = match surface {
            CHAIN => chain_disagrees(ctx, &bad),
            STAGE2 => stage2_disagrees(ctx, &bad),
            _ => gather_disagrees(ctx, &bad),
        };
        assert!(
            !caught.is_empty(),
            "mutant {name} was NOT caught by any case on the {surface} surface. A kernel that \
             survives this mutation is not gated by this suite; either the corpus lost the case \
             that saw the defect or the mutation is inert and must be replaced by one that is not. \
             {WHY}"
        );
        eprintln!("MUTANT {name} ({surface}): caught by {caught:?}");
    }
}

#[test]
fn only_the_near_tie_case_sees_a_reduction_that_compares_in_bf16() {
    let ctx = ctx();
    let src = misc_source();
    let (_, name, from, to) = BF16_COMPARE_MUTANT;
    let bad = mutate(&src, from, to);
    let mut wrong_token: Vec<&str> = Vec::new();
    let mut right_token: Vec<&str> = Vec::new();
    let mut perturbed_partials = 0usize;
    for c in am_cases() {
        let (want_pv, _) = c.stage1_reference();
        let run = run_argmax_chain(ctx, &bad, &c).unwrap_or_else(|e| panic!("{}: {e}", c.label));
        if (0..c.groups).any(|g| run.pv[g].to_bits() != want_pv[g].to_bits()) {
            perturbed_partials += 1;
        }
        if run.token == c.token_reference() {
            right_token.push(c.label);
        } else {
            wrong_token.push(c.label);
        }
    }
    eprintln!(
        "MUTANT {name}: partials perturbed in {perturbed_partials} cases, token WRONG in \
         {wrong_token:?}, token still right in {right_token:?}"
    );
    assert_eq!(
        wrong_token.len(),
        1,
        "rounding every logit to bf16 before the compare changed the emitted token in \
         {wrong_token:?}. Exactly one case -- the planted one-ulp near tie -- may see it: that is \
         what makes the near-tie case load-bearing rather than decorative. If several cases see \
         it, the corpus has acquired an accidental near tie and the claim needs re-deriving; if \
         none do, the planted pair has stopped colliding in bf16. {ARGMAX_IS_DISCRETE}"
    );
    assert!(
        perturbed_partials > 1,
        "the bf16 compare perturbed the partial VALUES in only {perturbed_partials} case(s), so \
         the value assertions are what would catch this mutant and the near-tie claim above is \
         about a defect this corpus would have caught anyway"
    );
    assert!(
        !right_token.is_empty(),
        "every case emitted the wrong token under the bf16 compare, so nothing distinguishes the \
         near-tie case and the corpus is not measuring what this test names"
    );
}

fn body_of(src: &str, name: &str) -> String {
    let key = format!("fn {name}(");
    let at = src.find(&key).unwrap_or_else(|| {
        panic!("the shipped source no longer declares {name}; this gate cannot locate its body")
    });
    let rest = &src[at..];
    let end = rest.find("\n}\n").unwrap_or_else(|| {
        panic!(
            "no closing brace at column zero after `{key}`; the WGSL layout changed and this \
             extractor is silently returning the rest of the file"
        )
    });
    rest[..end + 3].to_string()
}

#[test]
fn the_decode_bodies_gated_here_are_no_part_of_the_prefill_source() {
    let misc = misc_source();
    let prefill = nv_models::qwen3_5_dense_wgpu::shipped_prefill_source();
    let gated = [GATHER_ENTRY, AM1_ENTRY, AM2_ENTRY, REDUCE_HELPER];
    let bodies: Vec<String> = gated.iter().map(|e| body_of(&misc, e)).collect();
    for (e, body) in gated.iter().zip(bodies.iter()) {
        assert!(
            body.len() > 120,
            "the extracted body of {e} is only {} bytes; the extractor is not finding the whole \
             function and every containment check below is vacuous",
            body.len()
        );
        assert!(
            !prefill.contains(&format!("fn {e}(")),
            "the prefill source now declares {e} as well; if the decode entry and its M-row twin \
             have become one function body, the twin's gate already covers it and this suite is \
             redundant"
        );
        assert!(
            !prefill.contains(body.as_str()),
            "the whole body of {e} now occurs verbatim in the prefill source, so \
             graph_q3d_elementwise_oracle's gather corpus does compile this text and the two gates \
             overlap. {INDEPENDENCE_IS_A_BODY_LEVEL_CLAIM}"
        );
    }
    for (surface, name, from, to) in MUTANTS {
        assert!(
            misc.contains(from),
            "anchor for mutant {name} is gone from the shipped misc source: {from:?}. A mutant \
             whose anchor rotted is silently inert, and the GPU tests that would have caught that \
             do not run on a box with no adapter -- which is why this check is CPU-only."
        );
        assert_ne!(
            from, to,
            "mutant {name} replaces text with itself and can never turn anything red"
        );
        let total = misc.matches(from).count();
        let inside: usize = bodies.iter().map(|b| b.matches(from).count()).sum();
        assert_eq!(
            total, inside,
            "mutant {name} on the {surface} surface: its anchor occurs {total} time(s) in the \
             shipped misc source but only {inside} of those are inside the bodies this suite \
             gates. {INDEPENDENCE_IS_A_BODY_LEVEL_CLAIM}"
        );
    }
    eprintln!("{WHY}\n{INDEPENDENCE_IS_A_BODY_LEVEL_CLAIM}\n{ARGMAX_IS_DISCRETE}\n{BOUNDS_DOC}");
}
