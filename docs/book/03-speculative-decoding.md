# Chapter 3 — Speculative Decoding

Instead of one forward pass per emitted token, the server runs a cheap model to
guess several tokens ahead and checks all of them in one forward pass of the
real model. Two complete implementations exist: the per-request path
(`chat_engine/gemma4_loop.rs`), which builds its speculative state inside the
request task, drives an Eagle3 or DFlash drafter, and verifies through
`nv-specdecode`'s chain API; and the persistent spec-serving engine
(`nv-models/src/laguna_serve.rs`), which owns a background thread, one KV cache
and a set of captured CUDA graphs for the process lifetime. They share the
greedy accept rule and almost nothing else — not even the same admission
conditions, because one has a host-side sampler arm and the other deliberately
does not.

## The bet

Decoding one token at batch size 1 is bandwidth-bound: the forward pass reads
every weight to produce a single row of logits. Feeding it M rows reads the
same weights once and does M times the arithmetic. **The verify pass is the
decode pass with a wider batch dimension** — `LagunaStepGraph::step` is
`cu_q = [0, 1]` and `LagunaVerifyGraph::new_inner` in the same file is
`cu_q = [0, t]` with `logits_t` shaped `[1, t, vocab]`;
`Laguna::apply_attn_w8_for(t)` is the only place the model is told the width.

A round never emits fewer than one token — `accept_prefix_argmax` always
returns `commit_len >= 1`, and `accept_block_on_host` pushes the verifier's own
token on the first mismatch. Speculation cannot lose ground, only waste a wider
forward.

`AdaptiveK::cost_ms_per_tok` (`chat_engine/spec_window.rs`) is the cost model:
with per-slot acceptance `p` as an EMA, expected tokens per round is
`tau = (1 - p^k) / (1 - p)` and cost per emitted token is
`(draft_cost * (k - 1) + verify_cost) / tau`. `AdaptiveK::choose` scans `k` over
`[EAGLE3_K_MIN, k_graph]` for the minimum, with hysteresis so it does not
oscillate. **`tau` saturates as `k` grows, the draft term does not, and the
verify term grows with batch width, so there is an interior optimum** whose
location depends on the drafter's hit rate — a property of the prompt.

`AdaptiveK` is wired into the Eagle3 graphed arm only. What makes it cheap is
the split between two `k`s:
`NV_ADAPTIVE_K_MAX` sets the graph width, the verify cache and CUDA graph
are built once for it, and `k_eff` varies underneath per round at no rebuild
cost — a short draft is padded out and the padding tokens are rejected at their
slots. `observe` keeps two draft-cost EMAs depending on whether `k_cur` is at
the graph width or below (below-width takes the eager path); lookup-arm rounds
are excluded from feedback.

## The accept rule, and why the output is unchanged

Everything that makes the greedy arm safe is one loop, written out most plainly
in `accept_block_on_host` (`nv-models/src/laguna_dflash.rs`):

```rust
for (i, &d) in drafts.iter().enumerate() {
    let vtok = vtoks[i];
    if vtok == d { emitted.push(d); accepted += 1; }
    else { emitted.push(vtok); break; }
}
if accepted == k { emitted.push(vtoks[k]); }
```

The verify forward runs over `block = [anchor] ++ drafts`, so `vtoks[i]` — the
argmax of verify row `i` — is the target's prediction for `drafts[i]`, and row
`k` is the row past the last draft. While drafted token and argmax agree,
commit the draft; at the first disagreement, discard the rest and commit the
target's own token; if every slot agreed, commit one extra token from the row
past the end — the bonus token, free because the forward already computed
it. Every committed token is therefore either the target's argmax at its
position or a drafted token proved equal to it, so the emitted stream is the
argmax stream. **The drafter's output influences the emitted text only through
an equality test, so the drafter can be arbitrarily bad, arbitrarily good, or a
lookup table, and the text does not move.**

The rule appears in three places because the accept step runs in three:
`accept_prefix_argmax` (`chain.rs`), returning
`ChainAccept { commit_len, draft_accepted, next_bonus }` — slot 0 is the
previous round's bonus and is never judged, which is why `commit_len` cannot
reach zero; `accept_block_on_host` (`laguna_dflash.rs`), the host reference for
the block form; and `dflash_accept_f32` (`nv-kernels/cuda/dflash_accept.cu`), a
two-stage row-argmax plus a single-thread chain kernel walking
`while (a < m-1 && row_argmax[a] == drafts[a]) a++`.

The device and host forms must produce identical `(accepted, emitted)` because
they are chosen at runtime — `laguna_serve.rs` accepts through
`accept_block_on_device_with_status` and falls back to `accept_block_on_host`
on a CUDA error, except when the drafts or the XQA accept status live only on
the device, where the round fails instead. That pair drives both the emitted tokens and
the KV rollback length, so a disagreement (a different tie-break on an exact-tie
row) would make the same prompt emit different text depending on which path ran
and desynchronize the target's KV cache from the emitted prefix. **Both argmax
implementations tie-break to the lowest index.**

This framing licenses most of the optimizations below. Shortening a draft is
safe by construction, because the DFlash proposer is a single causal pass over
`[anchor, MASK × k]` in which position `j` attends only to `0..=j`, so the first
m drafts of a longer proposal are bit-identical to an m-length proposal.
Narrowing which context positions the drafter attends to is safe for the same
reason: the verify pass and its argmax are untouched.

What the tests assert. `LockstepChainSpec` (`nv-specdecode/src/wgpu_spec.rs`)
runs the verifier one step at a time while simulating the accept bookkeeping and
asserts `stats.emitted == spec.greedy(...)` for an identical, silent, perturbed
and prompt-lookup drafter. No suite asserts byte identity between the drafted
Laguna path and plain greedy decoding, and none can: the accept rule
guarantees identity in exact arithmetic, but a width-`t` verify batch and a
width-1 decode take different kernel paths with different reduction orders, so a
row's argmax can flip on near ties. That is a floating-point property of the
target model, not a defect in the accept rule, and it is why
`NV_DETERMINISTIC=1` exists for exact-match gating.

## The eligibility gate

Laguna is greedy or nothing (`chat_engine/laguna_loop.rs`): eligible iff
`is_greedy()` ∧ `!has_penalties()` ∧ `guided.is_none()` ∧ `logit_bias.is_empty()`
∧ `!logprobs` ∧ `prompt + max_new + num_spec + 2 <= max_seq`
(`request_is_eligible`). The persistent engine has exactly
one accept implementation — the device argmax-equality kernel with the host
argmax as fallback — and no sampler at all, so **every clause exists because
that accept rule cannot express the request**:

| clause | why |
|---|---|
| greedy only (`temperature <= 1e-6`) | Accepting a draft against a distribution needs the speculative-sampling correction, which needs the full probability row on the host — and this engine's whole design is that only `(accepted, emitted)` crosses the bus. Admitting a sampled request would silently bias the output toward the drafter's distribution. |
| no penalties | Penalties are functions of the emitted prefix, but all `k+1` logit rows are computed before it is known how many will be committed. The device accept path applies no logit transform at all. |
| no guided decoding | A grammar advances per emitted token, and the verify argmax is over unmasked logits, so an accepted token need not be in the legal set. Nothing in `build_chain_batch` or the accept kernels consults a mask. |
| no `logit_bias` | Bias is additive before the argmax; the accept kernels argmax raw verify logits. |
| no `logprobs` | Only a handful of `u32` leaves the device per round; per-token logprobs need the full row per committed position, so `ChatEvent::Logprob` is emitted only from the non-speculative loop. |
| KV must fit | Reserves room for the speculative overhang. |

An ineligible request does not fail — control falls through to the ordinary
loop, and the Laguna path also falls back if the engine's job channel is closed
or its stream ends before the first token.

Gemma4 has a second accept arm, so a looser gate:
`spec_gate_for_request(no_spec, use_eagle3_set, greedy) = !no_spec &&
(use_eagle3_set || greedy)`, plus no guided, empty `logit_bias`, `!logprobs`,
`prompt_ids.len() < spec_ctx_disable(...)`. **Greedy is sufficient but not
necessary** (with `NV_USE_EAGLE3` set, a sampled request is admitted), and
penalties are not in the gate at all. The switch is
`gpu_accept = sampler.pure_greedy()`, threaded into
`verify_chain` as `want_logits = !gpu_accept`, where `pure_greedy` is greedy
and no penalties and no bias and no grammar. When it holds the verifier
returns `ChainJudgment::Argmax` — `k` u32s — and the integer equality walk runs;
when it does not, it returns `ChainJudgment::Logits`, the full `k × vocab` f32
block comes to the host, and `ChatSampler::accept_draft` accepts with
probability `p_target(drafted)` and on rejection draws from the target
distribution with the drafted token excluded and renormalized
(`residual_sample_checked`). `warped_dist` applies bias and penalties before
forming the distribution, so a penalized request is handled correctly rather
than refused.

The price is a `k × vocab` f32 copy per round instead of `k` u32s, and **the two
arms give different guarantees**: the GPU arm is sequence identical to greedy
decoding, the host arm is distribution preserving but consumes RNG in a
different order, so it is not token-identical to a plain sampled run with the
same seed.

What the wire says. `spec_status` is set once at load from
`eagle3_gate(spec_requested, required, drafter_loaded)` and surfaces as the
`x-spec-decode` header and `spec_decode` on each `/v1/models` chat row, mapped
to `on` / `degraded` / `off` (or `unknown`). **These report engine-level state
decided at load time, not per-request eligibility**: a request answered with
`on` may still have taken the non-speculative path through any clause above, and
no header records that. `NV_EAGLE3_REQUIRED` / `NV_DFLASH_REQUIRED` turn a
degraded load into a startup failure so a silently non-speculative server cannot
be deployed by accident.

## The drafter families

### Eagle3 speculator checkpoints

Upstream: EAGLE-3 (arXiv:2503.01840) — the `[hidden, 3 * hidden]` `fc` is its
multi-layer feature fusion over the aux hidden states named by
`eagle_aux_hidden_state_layer_ids`, and the draft-vocab `lm_head` is its shift
from feature prediction to direct token prediction; the draft-tree machinery is
EAGLE-2's dynamic draft tree (arXiv:2406.16858). `eagle3_loader.rs` loads a
one-layer model running inside the target's representation space: a full
target-vocab `embed_tokens`, the `fc` projection, one decoder layer whose
`q/k/v` take `2 * hidden` inputs, a `norm`, an `lm_head` over the draft vocab,
and two vocabulary maps. **Every load is shape-exact, so a config/checkpoint
disagreement fails at load rather than at generation time**, and
`spec_env::drafter_target_mismatch` performs the same check ahead of time,
naming which of hidden size, aux layer range or vocab size disagreed.

The `2 * hidden` input width is the architecture in one number: each row is
`cat([input_layernorm(embed[token]), hidden_norm(fc(aux))])` with `aux` the
target's hidden states at three tapped layers concatenated on the feature axis
(`chain::aux_row_extract` gathers it), **so the drafter sees both what token was
emitted and what the target was thinking when it emitted it.** For step `j >= 1`
the recurrence swaps in the drafter's own state:
`h_cond <- block_out(cat([input_layernorm(embed[tok_{j-1}]), hidden_norm(h_cond)]))`
and `token_j <- d2t_map(argmax(lm_head(norm(block_out))))`. The target's aux
states are consumed only for context rows; step 0 is where conditioning
enters (see Conditioning below).

`d2t` is an offset table, not an id table: `d2t_map(t) = t + d2t[t]`, applied
immediately after every argmax on all three paths (host, device `index_select`,
graph via `token_map_u32`), so every token downstream of the drafter head is
already in target id space — which is why the drafter's embedding table is
indexed by target ids. `t2d` is a boolean support mask, loaded and
length-validated but not consulted.

`DrafterKvCache` is append-only and never rewinds. The speculative steps' KV
rows are scratch, allocated per round, concatenated in front of the committed
context, then dropped; only committed rows are appended, so there is nothing to
roll back on rejection — the accept step only decides how many context rows to
encode next round. `NV_DRAFTER_KV_WINDOW` / `NV_DRAFTER_KV_SINK` bound the cache
with an attention-sink policy: `maybe_compact` rebuilds K and V as
`rows[0..sink] ++ rows[phys-window..phys]` once physical rows exceed
`sink + window + slack`. **Rope positions continue to use the logical length,
so eviction does not renumber positions**; `phys + evicted == len` is asserted
at every entry point mixing the two.

Tree drafting (`Eagle3Proposer`, `DraftTree`), where each node's attention sees
only its ancestors, is reachable only from the ungraphed loop: `NV_EAGLE3_TREE`
disables the multi-query verify kernels, forces a full-context re-encode each
round, and routes through `Gemma4Verifier::verify_tree` with a general mask.
`flatten_with_mask` degenerates to `lower_tri_mask` for a linear tree, asserted
in `chain.rs`.

### DFlash

Upstream: "DFlash: Block Diffusion for Flash Speculative Decoding"
(arXiv:2602.06036, ICML 2026) — the paper's parallel drafting is exactly the
`[anchor, MASK, MASK, ...]` block, the same mechanism as the DiffusionGemma
decode loop in 05-backends.md applied to drafting rather than the target. Two
implementations: `nv-specdecode/src/dflash.rs` (standalone, carries its own
`embed_tokens` and `lm_head`) and `nv-models/src/laguna_dflash.rs` (borrows the
target's).

DFlash is not autoregressive within a round. It runs one forward over the
query block (`mask_token_id` from the checkpoint config) and reads all `k` drafts
out of rows `1..=k`; row 0, the anchor's own row, is discarded. Hence
`block_size` is a hard cap: `LagunaDflash` requires `1 <= k < block_size`, and
`laguna_serve.rs` clamps `num_spec` to `block_size - 1` at construction.

The drafter never walks the token stream to build its own context. Its KV
context is manufactured from the target's tapped hidden states: `combine_aux`
applies a per-tap RmsNorm, concatenates, projects through `fc` and normalizes,
and `append_context` runs that through each drafter layer's `k_proj`/`v_proj`
with RoPE. After each round only the accepted rows are folded in —
`laguna_serve.rs` narrows the aux output to `consumed = 1 + accepted` rows first
— so the drafter's context tracks the committed prefix exactly.

`resolve_tap_layers` picks the tapped layers. The default (`TapList::Target`)
uses `dflash_config.target_layer_ids`, the list the checkpoint was trained with,
spread across the target's depth so the drafter sees early, middle and late
representations rather than only the final hidden state.
`NV_LAGUNA_TAP_LIST=eagle` selects `eagle_aux_hidden_state_layer_ids`, filtering
out-of-range ids with a warning — and warning again if the surviving count no
longer matches `fc`'s trained input width, which `combine_aux` hard-checks.

One draft-length policy changes only how many slots are offered, which the
proposer's causal structure makes flip-safe: `adapt_truncate_len`
(`NV_DFLASH_ADAPT`, `NV_DFLASH_ADAPT_THRESH`) cuts at the first slot whose
top-1-minus-top-2 margin falls below a threshold, floored at one. Every extra
draft costs a full verify column and, on the 256-expert MoE, extra verify
expert reads, so a truncated doomed tail is pure savings.

"Vegas" (`NV_SPEC_VEGAS` / `_VEGAS_K`, `vegas_topk_ctx_idx`) narrows the
drafter's attention instead: it scores every stored context position by the
anchor query's roped softmax attention mass summed across query heads — the same
quantity verify attention implicitly computes — and keeps the top
`NV_SPEC_VEGAS_K` (default 512), selected once from the anchor row at layer 0
and reused for every drafter layer. It changes only what the proposer reads, so
it is byte-identical to the window it replaces. Ships default-off.

### The suffix / lookup drafter

`nv-lookup` is the cheapest drafter in the repo: no GPU work, no weights.
`SuffixAutomaton` is an online suffix automaton over prompt-plus-emitted;
`propose(max_len, min_match)` walks the suffix link of the last state for the
longest repeated suffix, requires `min_match` length, locates the earlier
occurrence via `firstpos`, and returns what followed it. A strong bet on
quoted code, repeated tool-call scaffolding and structured output, worthless on
novel prose. Correctness needs no argument: the proposal is a plain token vector
handed to the same verifier.

The arms are mutually exclusive per round and lookup gets first refusal.
`suffix_arm_wins(proposal_len, min_match, match_len, drafter_ema)` admits the
lookup proposal only when its match is long enough and its length is at least
the model drafter's running average. **The EMA is updated only on model-drafter
rounds**, so lookup rounds never poison the baseline they are measured against.
Three call sites: `LookupState` in `laguna_dflash.rs`
(`NV_LAGUNA_LOOKUP_DRAFT`), a bare automaton plus `pad_suffix_draft` in
`gemma4_loop.rs` (`NV_SUFFIX_DRAFTER`, `NV_SUFFIX_MIN_MATCH`), and
`PromptLookupDrafter` in `wgpu_spec.rs`. `pad_suffix_draft` repeats the last
proposed token to fill out to `k - 1`, because the graphed verify path is
captured for a fixed width — padding with a token that will be rejected is
cheaper than falling off the graph.

### The Qwen MTP head

The two checkpoint families have two separate implementations. The 3.6-MoE
head is `nv-models/src/qwen3_5_mtp.rs`: `MtpHead::from_map` loads the
in-checkpoint multi-token-prediction head with its `MtpMoe` expert layout and
runs the full draft forward (fused embedding+hidden `fc`, one attention layer,
FFN, final norm through the target's `lm_head`); `MtpMoeEagerDecodeSession` and
`MtpMoeGraphedDecodeSession` drive rounds through `MtpSpecEngine`.

The 3.8-dense serving drafter is `nv-specdecode/src/qwen38_mtp.rs`
(`Qwen38DenseMtpHead::from_checkpoint`, `Qwen38MtpDecodeSession::start`), whose
shapes are pinned by `Qwen38MtpGeometry` and whose weights come from
`MTP_WEIGHTS_FILE_NAME` (`model_mtp.safetensors`) under `MTP_TENSOR_PREFIX`
(`mtp.`). It is
selected by `NV_DRAFTER=mtp` — the literal value, `mtp_drafter_selected` accepts
nothing else — with depth from `NV_MTP_K` clamped by
`MTP_CHAIN_DEPTH_MAX_SO_VERIFY_ROWS_STAY_UNDER_THE_F32_FAST_PATH_HARD_CAP` and
the checkpoint from `NV_MTP_DRAFT_DIR`. `chat_engine/build.rs` (the engine
builder, not a cargo build script) boot-fails when `NV_DRAFTER=mtp` and the head
is missing or misshapen, for the MoE and the dense head alike; the CUDA loop is
`run_qwen38_mtp_greedy_rounds`, and the wgpu side dispatches on
`Decoder::Qwen3_5Dense(m) if m.mtp_active()`.

### The Qwen3.8 two-branch verify tree (`NV_Q38_MTP_TREE=1`)

The rows-keyed verify lanes (`ensure_verify_lane` / `forward_verify_chain` in
`graph_engine.rs`) verify one linear chain per round, and `NV_Q38_MTP_TREE=1`
adds exactly one non-chain shape beside them: a fixed two-branch m=6 tree,
rows `[anchor, d1, d2, d3, c2, c2d2]` with parents `[-1, 0, 1, 2, 0, 4]` and
positions `committed + depth` where depth is `[0, 1, 2, 3, 1, 2]` (the
`VERIFY_TREE_TWO_BRANCH_M6_*` constants in `qwen3_5_moe.rs`). Branch A is the
ordinary top-1 draft chain; branch B is the drafter's top-2 token at the first
draft position plus one continuation drafted from it (one extra drafter step
per round, its step-0 hidden reused) through the device-chained draft loop,
the only draft loop. The env needs `NV_MTP_K=3` and refuses the
`NV_Q38_MTP_P_MIN` cut; suffix rounds bypass the tree, so the sufspec
composition rule is unchanged.

Three mechanisms once pinned the verify to chains; each now has a tree arm:

- GDN row forking. `gdn_conv_decode_chunk_silu_tree_bf16` and
  `gdn_decode_chunk_split_tree_bf16` take a device per-row parent index and
  load each row's conv tail and recurrent state from the parent row's
  checkpoint slot (parent `-1` reads the live pre-round state). With linear
  parents they are bitwise the chain chunk kernels, and every tree row is
  bitwise a re-chained run of its ancestor path; no suite in the tree gates
  either property at the q38 serving shape. The row-sequential GDN fallbacks
  refuse tree rounds (`lin_attn_step` errors instead of committing lossy
  logits), so a tree round rides the mrow chunk path the verify-TC scope
  always selects.
- Attention parent-path masks. Tree rounds take the same
  `verify_attention_fp8_mk` arm a chain of that depth takes (see the arm ladder
  below), and every arm carries the hole. Each row keeps its chain suffix window
  (`delta = m-1-qi`, so the window end is always self+1 under the
  branch-contiguous layout), and the two sibling rows additionally hide the
  branch-A block through a hole window passed as host deltas from `n_total`,
  masking `[n_total-5, n_total-2)`: the one-launch arms and the splitk mk
  ladder (`flash_decode_fused_fp8kv_mk_hole(.., hole_from_row=4, 5, 2)`) take
  it as launch arguments, and rows at or past `hole_from_row` skip the hole
  positions. Each one-launch arm's hole mask matches the mk hole arm and
  excludes the hole rows (`tctile_hole_mask_matches_the_mk_hole_arm_and_actually_excludes_the_hole_rows`,
  `fused_tc_hole_mask_matches_the_mk_hole_arm_and_actually_excludes_the_hole_rows`
  in `fp8_mk_ratio_oracle.rs`), and `verify_attention_fp8_mk` errors rather
  than falling to the full-KV dequant view for a tree round. Routing tree
  rounds through the chain's own arm per depth is what keeps tree-on/off
  streams bitwise under the knee gates — an arm-asymmetric tree surfaces the
  cross-arm bit difference as extra tie flips.
- Non-prefix commit. The branch-contiguous layout makes a branch-A accept
  a plain prefix commit (the existing `commit_verify_consumed` shape). A
  branch-B accept is the one memmove-class fix: `commit_verify_tree` compacts
  KV rows `[pos+4, pos+4+b)` down to `[pos+1, ..)` per layer
  (`compact_verify_tree_branch_b_rows_down`, four dtod copies per
  full-attention layer, disjoint ranges), which restores slot==position — the
  rows were RoPE'd at their tree positions when written, so the moved rows are
  already correct. Lin state rolls back by checkpoint row
  (`rollback_lin_to(bonus_row + 1)`), which is exact because the forked
  kernels write each row's checkpoint along its own ancestor path. The
  drafter KV catch-up pairs branch-B tokens with their path hiddens (rows 0
  and 4), and the reanchor hidden is the bonus row's.

The tree verify is its own graph lane and key
(`VERIFY_TREE_GRAPH_SHAPE_TOKEN_BASE...`, 0x52000-based) beside the rows-keyed
chain lanes; the parent buffer is device data written once, positions ride the
lane `pos_buf` like every chain verify, and the hole deltas are launch
constants of the fixed shape, so tree rounds replay without recapture
(`precapture_verify_tree_graph_...` captures it at session start).

Losslessness is gated by the suite covering tree-on/off emitted-stream
identity (`qwen38_tree_gate`), on the champion basis with the verify knee gates live:
at least two of the three workload classes must be bitwise over the common
prefix (through dozens of branch-B commits), and any divergence must be the
documented cross-shape tie band (one arm's argmax margin under 1 bf16 logit
unit — the same band `ADAPTIVE_TIE_MARGIN_MAX_...` pins for adaptive draft
lengths); two confident arms disagreeing fails the gate as corruption.

The wgpu twin still hosts chains only
(`VERIFY_CHAIN_FORWARDS_THE_WHOLE_BAKED_M_ROW_CHUNK_AND_COMMITS_IN_PLACE_ONLY_AT_FULL_WIDTH`
in `qwen3_5_moe_wgpu.rs`); the gemma4 `tree_verify_attn` kernels remain the
full-attention-only prior art.

The recoverable mass is measured by the suite covering drafter-alternative
recovery at rejected positions (`qwen38_pctree_headroom`), which runs the
champion basis over the three sufspec workload classes with
`NV_Q38_MTP_ROUND_DEBUG` collecting drafter top-3 per draft step: at rounds
whose first draft position the verify rejects, the verify-decided token sits
in the drafter's top-2 for ~46% and in the top-3 for ~60% of them (k=3,
`unsloth/Qwen3.8-27B-NVFP4`, 1024-token prompts, 512 generated tokens per
class; deeper positions recover 29-60%; per-class splits in the suite's
`PCTREE-HEADROOM` stderr lines). At serving, branch B lands in 5.5-14.7% of
tree rounds at 1-2 extra tokens each (the `TREE-GATE`/`CTX-TREE`
recovery_rate lines).

**The tree ships default-off because the recovery does not convert to net
tok/s.** The recovered mass buys a fraction of a token per round, but every
tree round pays one extra drafter step (including the step-0 full-row
readback that yields the top-2 sibling) and two extra verify rows — and the
verify attention scan, the lm_head rows and the logits readback are all
row-count-proportional, so the m=6 verify's premium over m=4 grows with
depth. A drafter-confidence gate that drafts the sibling only on rounds whose
step-0 top-1 is weak (recovery concentrates exactly there) does not exist.

## Conditioning

The graphed Eagle3 loop in `gemma4_loop.rs` always drafts in shift
conditioning against a persistent `DrafterKvCache`, through
`chain_draft_cached_shift_graphed_tail` or, when the chain graph declines,
the eager `chain_draft_cached_cond_tail` with `shift = true`. Context row `i`
pairs the target's aux row `i` with the embedding of token `i+1`, and the
final row is fed the bonus — the token the verifier just committed — so step
0 produces a genuine new prediction and the round drafts `k-1` tokens,
`build_chain_batch(bonus, draft, k, true)` giving `[bonus, draft[0..k-1]]`.
This works with an append-only cache because **the bonus is always
`batch[0]` and slot 0 is committed unconditionally**, so the token a row was
encoded with is exactly the token that later occupies that index.

The ungraphed loop (`NV_EAGLE3_UNGRAPHED`, or `NV_EAGLE3_TREE`, greedy only)
keeps no drafter KV: `chain_draft` / `tree_draft` re-encode the whole context
every round with unshifted pairing (row `i` = token `i` + aux row `i`), step
0 is the drafter's own argmax, and `Gemma4Verifier::verify_tree` verifies the
proposal.

## Verification

The chain form (per-request path). A round hands `verify_chain`
`build_chain_batch(bonus, draft, k, shift)` (the `k` tokens with the previous
bonus at slot 0), `chain_positions(committed, k)` = `[committed, committed+k)`
so the chain is verified as if it were a real continuation, and
`lower_tri_mask(k)`, hoisted once outside the loop since it depends only on `k`
— row `i` has ones at columns `0..=i`, so draft slot `i` sees the committed
prefix and slots `0..=i` including itself.

Rejection does not erase KV. The verify cache carries a device-side
`n_committed`; the KV append writes row `n_committed + slot` and attention reads
`[0, n_committed + k)`. The host advances `committed` only by `commit_len`, so
rows written for rejected drafts sit at or above the new `committed` and are
overwritten next round. The drafter's state is discarded the same way — Eagle3
resets `aux_proj` to `None` and `aux_base` to `context.len()`, DFlash re-appends
only accepted rows — and each round asserts the two have not desynchronized.

The block form (persistent engine). `SpecState::decode_spec` builds
`block = [anchor] ++ drafts`, calls `verify_block` (returning both logits and
the tapped aux states for the same rows), accepts on device with a host
fallback, then rolls back `block.len() - consumed` where
`consumed = 1 + accepted`. The same `consumed` prefix of the aux tensors is
appended to the drafter's context. **Because that context is derived from the
verify pass's own aux output, the two caches advance in lockstep by
construction.**

The continuous-batching form lives in `nv-engine/src/scheduler.rs`: `set_drafts`
attaches a draft vector to a running sequence, and `step()` builds a
`BatchKind::Verify` batch whenever any running sequence in `Decode` state has
drafts. **That batch must be completed through `complete_verify_step` —
`complete_step` explicitly bails on it** — because it consumes one
`VerifyOutcome { accepted_count, bonus_token, accepted_drafts }` per sequence
rather than one sampled token, appending `accepted_drafts ++ [bonus_token]` and
checking stop conditions after each append so a sequence can finish mid-round.
Blocks are allocated for `1 + draft_len()` slots up front, and a sequence that
cannot be extended has its drafts cleared and is preempted back to `Waiting`
rather than failed.

### The q38 verify attention arm ladder

`verify_attention_fp8_mk` (`qwen3_5_moe.rs`) picks one arm per call from the
host token total and the row count; the two one-launch arms serve only the
q38 geometry (group 6, head_dim 256, m ≤ 8):

- The single-pass tensor-core arm (`flash_decode_fused_tc_fp8kv_mk_paged`:
  per 32-position tile K is staged once as fp8 and scored against every
  m × group q row through m16n8k16 tiles, and PV rides the same mma) engages
  at or past `fused_tc_gate_knee_for_rows`. `NV_Q38_MK_VERIFY_FUSED_TC` unset
  or `=gate` gates at that knee, `=1` engages it always, and
  `NV_Q38_MK_VERIFY_FUSED_TC_MIN_TOTAL` overrides the knee.
- The tctile arm (`flash_decode_rowshared_tctile_fp8kv_mk_paged`, the
  llama.cpp fattn-mma shape: each warp scores one chain row's 6-row q group as
  an m16n8k16 A tile against 8-position chunks of a block-shared K stage, V
  register-prefetched through L2) takes the calls the fused-TC arm declines,
  at or past `TCTILE_GATE_KNEE_TOTAL_8192_ITS_LADDER_WINS_FROM_8192_AND_LOSES_AT_256`
  under the same grammar (`NV_Q38_MK_VERIFY_TCTILE`,
  `NV_Q38_MK_VERIFY_TCTILE_MIN_TOTAL`). The shipped fused-TC knees sit below
  it, so tctile engages only when forced below the fused-TC knee or when that
  knee is raised past its own.
- Every other call — below the knees, any other geometry, m > 8 — takes the
  splitk mk ladder (`flash_decode_fused_fp8kv_mk`, or
  `flash_decode_fused_fp8kv_mk_hole` for a tree round).

Any selector value other than unset, `gate` or `1` panics. A graphed session
picks its arm at capture depth and replays it even when the context later
crosses a knee.

Stream identity cannot gate the one-launch arms: mma tiles reassociate the
score sums, the bf16 attention output sits 1-2 ulp from the mk arm's, and the
MoE router amplifies that into confidently different greedy streams
(`STREAM_IDENTITY_IS_NOT_GATEABLE_FOR_SCORE_REORDERING_ARMS` in
`qwen38_tctile_gate.rs`). Their gates are the f64-oracle, row-0 window and
hole-mask suites in `fp8_mk_ratio_oracle.rs` and per-layer cross-arm parity
on live serving tensors (`qwen38_tctile_gate`, replaying
`NV_Q38_MK_VERIFY_DUMP` captures through the mk and tctile launchers). A
quality claim for any score-reordering arm needs matched-prompt pairs with a
sign-consistent delta across at least three prompt variants per workload
class: within-class prompt scatter dominates any arm effect, matched-prompt
deltas sign-flip across variants, and highly-tileable variants reproduce
bitwise-identical streams on both arms, so a class-aggregate comparison
across different prompts certifies nothing.

### The qwen3.8 verify row range: contiguous 2..=24, one launch shape to 8

The mk verify attention route (`mk_verify_route_available`) serves any row
count in 2..=24
(`MK_VERIFY_ROWS_MAX_24_ROWS_PAST_8_CHUNK_THROUGH_THE_8_ROW_TEMPLATES_AND_24_TIMES_32_SPLITS_FILLS_THE_SCRATCH_POOL`).
The CUDA row switches in `flash_decode.cu` instantiate the splitk mk kernel at
rows 1..8 only; a block deeper than 8 is chunked inside
`nv_kernels_flash_decode_fused_fp8kv_mk_paged` into 8-row launches whose
per-chunk `delta` reproduces each row's global attention bounds exactly — the
same algebra the head-dim-512 path uses — so every common row of two different
block depths is bitwise identical. Giant-M template instantiations are the
wrong shape here: per-thread accumulators scale with M, and the register spill
they cause at deep KV costs more than the chunk's extra KV read. Each row switch carries a
`default:` returning rc −31, so a row count with no case is a loud error, never
a silently-empty launch that times fast.

Two structures stop at 8 by design and stay there: the fused-TC and tctile
one-launch arms (their smem q block and row maps are 8-row layouts;
`one_launch_rows_fit` excludes deeper blocks, which fall to the chunked mk
ladder) and the serving draft ceiling
(`SUFFIX_DRAFT_OVERRIDE_CEILING_7_BECAUSE_DEEP_RUNGS_ARE_PRICED_BUT_THE_TCTILE_ARM_STOPS_AT_8_THE_GDN_MROW_CAP_AT_16_AND_ONLY_THE_TCTILE_CERT_MAY_FLIP_IT`).
The GDN mrow verify path serves rows to
`MROW_VERIFY_ROWS_MAX_16_THE_GDN_CHUNK_KERNEL_T_CAP`; a deeper block drops its
GDN layers to the sequential scan, which dominates the block's cost at every
depth.

What holds and what drifts across depths. Row 0 of a verify block is
bitwise m-invariant through m=8 — two gates hold it: the nv-models ladder gate
(`mk_verify_rows_are_bitwise_m_invariant_across_the_rung_ladder_2_to_24...`)
compares every common-prefix row bitwise across rungs at the kernel level, and
`qwen38_verify_m_invariance` sweeps the rung ladder eager and graphed on the
real checkpoint. Past m=8 row 0 drifts by up to a scale step because
`matmul_fp8`'s host path quantizes activations with a dynamic per-tensor amax
over the whole m-row block: deeper blocks can move the block amax, which moves
row 0's fp8 bytes. That drift is quantization-scale jitter, not a wrong path —
the gate bounds it
(`ROW0_DRIFT_BAND_8_0_PAST_THE_CONTIGUOUS_CAP...`) and requires top-1
agreement — but it means a deeper-than-8 verify is not bit-lossless against
decode until that activation scale is per-row. `NV_VERIFY_ONE_ARITH_ROWS`
(default 8; serving never exceeds 8) extends the fp8 LT span-4 plan pinning so
rows past 8 ride the same GEMM plan family as rows below it.

Past 8 rows a block loses the one-launch arms and pays the mk-ladder premium
per 8-row chunk, and past 16 it additionally pays the GDN sequential-scan
fallback. Lifting either cliff — chunking a one-launch arm past 8, or raising
the GDN mrow cap — does not by itself move the serving draft ceiling: that
flip needs the matched-prompt quality certification above, never the tiled
comparator, and no suite in the tree runs that certification.

### Why the wgpu dense graph has no verify path

`spec_route_eligible` never admits `Gemma4Dense` (its `spec_chain_slug` is
empty and no learned drafter opts it in), so on wgpu the 31B dense model
decodes one token at a time. The blocker is not the body — every
M-row kernel twin it would need exists and is bit-validated by
`gemma4_dense_chunked_prefill_is_bit_identical`. What is missing is the head:
the prefill pass list stops before the final norm, so there is no M-row
`rms -> lm_head -> softcap -> argmax`. Four constraints on anyone who builds it:

- The M-row-less arms are guarded, except one. NVFP4 projections, the
  subgroup fp8/int8 epilogue and the legacy row-scale epilogue all force
  `pf_m = 0`; an int8 lm_head must be refused too. **w4a16 has no M-row twin and
  no guard**, so landing it kills chunked prefill by falling through `gemv_mk`'s
  `Bf16`/`Fp8`-only match rather than refusing cleanly.
- The verify path exists only because `subgroup_ok` is strict
  (`min_size == max_size == 32`). The M2 Ultra reports a range, so the predicate
  is false, the tree epilogue runs, and the tree epilogue is the only one with
  M-row twins. Relaxing it to the `sg32_ok` runtime probe — the standing
  recommendation elsewhere — would **silently delete chunked prefill and any
  verify path on the 31B**. Any gate must panic rather than print one line.
- Do not touch the attention. Swapping the verify attention for a small-M
  fused kernel regressed structured spec decode severely on E4B and broke
  losslessness through a ulp-level argmax flip. The per-row split-K flash is
  bit-identical to decode by construction; the bit-safe way to remove its
  `2 * rows * layers` dispatches is to move the row index into the grid's `z`
  dimension and read per-row flash params from storage instead of a uniform.
- The rollback invariant is a capacity check, not a commit check. A verify
  round never advances `pos`, so the requirement is
  `pos + max_verify_rows <= max_seq`. Rejected and pad rows sit in unreachable
  slots only because the dense KV is not a ring (slot index == absolute
  position); turning on ring buffering for the sliding cache breaks that, so the
  verify constructor must assert it.

A verify path alone would not pay: no 31B-compatible drafter is wired
(`AssistantSpecDrafter::load` asserts `backbone_hidden_size == hidden_size`).
**Quantizing a drafter is the one place in this codebase where aggressive
quantization is unconditionally safe** — a drafter's error can only cost
acceptance, never output, because verify is exact.

## The persistent engine versus the per-request path

`build_laguna_spec_serve` runs at model load, spawning one `laguna-spec-serve`
thread running `spec_serve_loop`, which builds a `SpecState` holding one KV
cache sized to `kv_max_seq_len`, one `LagunaStepGraph`, one `DflashCtxCache`,
proposers keyed by rope theta, verify graphs keyed by block width, accept slots,
and optionally a `LookupState`. It runs `warmup` on two synthetic prompts (one
prose, one fenced Python) before signalling readiness, so the first real request
does not pay graph capture. Requests arrive as `SpecServeJob { prompt_ids,
prompt_text, max_new, eos_ids, emit }`; **the `emit` closure's `bool` return is
the backpressure/abort signal**, returning `false` when the HTTP client goes
away so `push_round` stops the generation. Detokenization, stop-string matching
and `max_new` enforcement happen on the HTTP side.

The engine serves one job at a time, and that is the point. A captured CUDA
graph owns fixed device buffers; a persistent KV cache is a single allocation
with a single write cursor; the drafter's ring context is one buffer per layer.
Sharing those across concurrent generations needs either per-request copies
(defeating the address stability the graphs depend on) or a paged design like
the batch engine's. The single-flight engine buys graph reuse and warm state;
the batch engine buys concurrency.

The per-request path is the fallback and the general case, building its own KV
cache (sized by `spec_verify_window`), verify graph and drafter KV cache inside
the request — except that the graph pool hands the verify graph and the chain
graph from one request to the next, and admission control prices them as
sticky VRAM. Because it pays setup inside the request it reorders it:
prefill does the verify forward and the aux projection but never touches the
drafter KV cache, and the drafter is armed by one `preencode_context` call
(to the last whole `DRAFTER_ENCODE_CHUNK` below the prompt length) after the
first token reaches the client.

## CUDA graph capture, and why the gates exist

| graph | file | shape key |
|---|---|---|
| `LagunaStepGraph` | `laguna_step_graph.rs` | fixed at 1 token |
| `LagunaVerifyGraph` | `laguna_step_graph.rs` | block width `t` |
| `DflashGraphProposer` | `laguna_dflash.rs` | `bs = k + 1`, keyed by the drafter's rope theta |
| `DraftChainGraph` | `eagle3_loader.rs` | `kd` (the body is fully unrolled) |

A captured graph is a frozen sequence of kernel launches with **the device
pointers and launch dimensions baked in**, replayed against whatever is in those
buffers. That single fact generates every gate below, because the failure mode
of a wrongly-replayed graph is not an error — it is a silently wrong answer
computed against a stale address.

- Shape must be part of the cache key. `CudaGraphRunner::run` takes a shape
  token, and the per-request path adds a second dimension:
  `verify_graph_reusable(cached_k, cached_capacity, k, needed)` requires the
  cached `k` to match and the cached capacity to cover the new requirement,
  with `verify_cache_capacity` rounding up to a multiple of
  `VERIFY_CACHE_GRAIN` (256) so a slightly longer generation reuses the graph.
- Addresses must be stable, so the caches are rings. `LagunaStepGraph::new`
  requires `cache.has_ring()`; `DflashGraphProposer` requires `ctx.has_ring()`
  and a matching capacity. `DflashCtxRing` preallocates `[1, cap, n_kv, hd]` per
  layer with `cap = sliding_window + block_size`, and sliding compaction shifts
  rows through a scratch buffer with device-to-device copies — rebuilding
  context with `Tensor::cat` each round produces a new allocation and cannot be
  captured. Without the ring (a non-CUDA device, or a ring allocation
  failure) the code logs and takes the concatenation path with no graph.
- **Counts that change per round travel through device buffers, not the
  capture.** The graphed proposer uploads `host_toks`, `host_pos` and
  `host_committed` into device buffers inside the captured region; the chain
  graph does the same with `n_buf[i] = phys + i` and
  `pos_steps[i] = ctx_len + i`. The graph is position-agnostic but
  capacity-bound.
- Preconditions are checked before every replay, and are exactly the
  assumptions capture froze. `DflashGraphProposer`: the ring exists, capacity
  matches, `ring.stored + bs <= ring.cap`, `ring.stored > 0`, the anchor
  position equals the cache length, and `check_rope_span(anchor_pos, bs)`
  passes.
  `DraftChainGraph`: not disabled, `kd` matches, projected physical rows plus
  `kd` fit the arena, `ctx_len + kd <= max_position_embeddings`.
  `LagunaVerifyGraph::verify`: the cache's ring metadata pointer must equal the
  captured one, and `write_pos + t <= max_seq_len`.
- **The chain arena is sized against the request's window, never against
  `kv_max_seq_len` directly.** `chain_graph_cap` uses `spec_verify_window`'s
  `max_seq`, so a 24k-token prompt on a 256k context arms a ~24k-row drafter K/V
  arena rather than a 256k one — the K/V pair costs
  `max_seq * n_kv * head_dim * 2 * 2` bytes and does not starve the deferred
  `preencode_context` into a CUDA OOM. The rebuild check uses the same cap, so
  allocation and reuse decision agree.
- Eligibility is checked before allocation. `chain_graph_eligible` requires
  contiguous BF16 dtype and embedding, an even head dimension, and every
  projection to be a contiguous bias-free BF16 dense weight with an even input
  width. The body is raw pointer arithmetic over `gemv_bf16` / `rmsnorm_bf16` /
  `rope_bf16_oop` / `kv_append_bf16` / `tree_verify_attn_bf16` / `silu_mul_bf16`
  / `argmax_bf16` / `token_map_u32`; anything needing a different kernel is
  refused before any buffer is allocated.
- Two capture hygiene details appear in every one of these files: CUDA
  context event tracking is disabled and the legacy stream synchronized before
  capture, because capture cannot proceed with tracking active; and the body is
  run once eagerly on the forked stream and synchronized first, a warm pass
  materializing any lazy allocation the body would otherwise perform during
  capture.
- A drafting failure is a fallback, never an error to the client, at two
  granularities: per-round (log `graph propose failed, eager`, take the eager
  tail, keep the proposer cached) and permanent (`eagle3_loader.rs` sets
  `g.disabled = true`; `laguna_serve.rs`'s `propose_round` stops retrying
  proposer capture after
  `PROPOSER_INIT_ATTEMPTS_2_A_PERSISTENT_CAPTURE_FAILURE_MUST_NOT_FORK_A_STREAM_EVERY_ROUND`
  failures and drafts eagerly from then on). A Laguna verify graph that fails
  to build before touching the cache falls back to an eager
  `forward_with_cache_aux_scoped`; one that fails after preparing the cache is
  removed and fails the request
  (`VERIFY_FAILED_AFTER_CACHE_PREPARE_TERMINATES_THE_REQUEST_BECAUSE_AN_EAGER_RETRY_WOULD_RESERVE_THE_BLOCK_TWICE`).
- Rope theta keys the proposer cache. The captured body dereferences the
  cos/sin pointers of the rope table live at capture, **so replaying under a
  different table would silently apply the captured one.** `LagunaDflash`
  holds exactly one table (the checkpoint's `rope_theta`, or
  `NV_DFLASH_ROPE_THETA`), so exactly one proposer is ever cached.

## Context length, rope tables and where speculation stops

Speculation extends the sequence past the committed length by up to `k` tokens
that may be discarded, and every bound has to account for that overhang.

Rope tables. Both drafters pass `max_position_embeddings` to `Rope::new`
verbatim, and `nv-layers/src/rope.rs` eagerly materializes dense
`[max_seq_len, head_dim/2]` tables. The Laguna drafter checks every span
against its table: `LagunaDflash::check_rope_span` refuses positions past it,
`propose_round` in `laguna_serve.rs` checks the `k + 1`-row span before
choosing the graph path, and the graphed proposer checks it again before every
replay. **The Eagle3 graph path has no such check**: its only positional guard
is `ctx_len + kd <= max_position_embeddings` — the eager path's
`index_select` would error out of range, but the graph path hands raw table
pointers to `rope_bf16_oop`, whose signature carries no length and which
performs no bounds check.

KV window sizing. `kv_window` clamps `max_new` so `prompt_len + max_new + 1`
fits; `spec_verify_window` additionally reserves `k + SPEC_VERIFY_HEADROOM` (16)
rows and shrinks the generation window to make room. **That is what makes the
`committed + k <= cache.max_seq()` assertion inside the verifier unfalsifiable
at runtime rather than a hope.** It returns `None` only when the prompt alone
does not fit, and the Gemma4 loop turns that into a failed request naming the
prompt length, `max_new`, `k` and the window rather than silently falling back;
the intermediate case is quieter, since a prompt that fits but leaves no room
after the reservation yields `clamped == 0` and the request completes with no
new tokens and `finish_reason: "length"`. The postconditions are proved rather
than tested — `assert_kv_window_invariants` and `assert_kv_step_in_bounds` are
driven by `#[kani::proof]` harnesses asserting no decode step can write past
`cache_len` or the cache limit for any input triple.

Context-dependent `k` and routing. `eagle3_k_default(prompt_len)` returns
`EAGLE3_K_SHORT_DEFAULT` below `EAGLE3_K_CTX_GATE` (8192) and `EAGLE3_K_DEFAULT`
above — the verify pass gets more expensive as context grows, so the default
draft length shrinks. `NV_SPEC_CTX_DISABLE` sets a prompt length above which the
request skips speculation entirely; `route_arm_for_ctx` picks DFlash below
`NV_ROUTE_CTX_GATE` (default 2048) and Eagle3 above. `resolve_drafter_arm`
resolves the loaded drafters against `NV_DRAFTER`: with both loaded, `route`
uses the context gate and everything else uses `route_drafter_arm(codeish,
dflash_ema, eagle3_ema)` — DFlash if `prompt_looks_codeish` (fences, language
keywords, line-ending punctuation) or if its EMA is at least Eagle3's. The EMAs
are process-global atomics holding f64 bit patterns, updated once at the end of
each generation with that run's drafts accepted per model round —
**suffix-arm rounds are subtracted from both numerator and denominator, so a run
carried by the lookup drafter does not flatter the model arm that was idle.** A
zero bit pattern means unseeded and reads back as the default, so the first
generation on each arm routes on the code/prose heuristic alone; the router
learns across requests within a process and forgets at restart.

## The knobs

Names only; what each selects, not what it costs. `SPEC_SNAPSHOT_KEYS` in
`spec_env.rs` is the registry, and a unit test asserts `SpecEnvSnapshot::capture`
reads exactly that list — so `NV_PROF_CHAT`'s `profile_line()` cannot silently
omit a knob that is in force. **A flag that never reached the code path it names
is the most expensive kind of misconfiguration**, because the run still produces
plausible output.

Selecting a drafter: `NV_DRAFTER` (`eagle3` | `dflash` | `mtp` | `auto` |
`route`), `NV_USE_EAGLE3`, `NV_NO_SPEC`, `NV_EAGLE3_DRAFT_DIR`,
`NV_DFLASH_DRAFT_DIR`, `NV_MTP_DRAFT_DIR`, `NV_EAGLE3_REQUIRED`,
`NV_DFLASH_REQUIRED`, `NV_ROUTE_CTX_GATE`.

Draft length: `NV_EAGLE3_K`, `NV_DFLASH_K`, `NV_DFLASH_K_PROSE`, `NV_MTP_K`,
`NV_ADAPTIVE_K`, `NV_ADAPTIVE_K_MAX`, `NV_DFLASH_ADAPT`,
`NV_DFLASH_ADAPT_THRESH`.

Drafter state: `NV_DRAFTER_KV_WINDOW`, `NV_DRAFTER_KV_SINK`,
`NV_EAGLE3_TREE`, `NV_EAGLE3_ENCODE_ATTN_SCORE_CAP`. That last one bounds
`preencode_context`, which builds the drafter K/V in `DRAFTER_ENCODE_CHUNK`
(1024) row steps whose eager `sdpa` would otherwise materialize a full
`[n_heads, chunk, prefix]` F32 score matrix — several GiB at a 24k prefix.
`causal_sdpa_qblocked` sub-blocks the query rows so the live tile stays under
`EAGLE3_ENCODE_ATTN_SCORE_F32_BYTES_CAP_BOUNDS_PREENCODE_PEAK` (512 MiB), and
because causal attention is independent per query row **sub-blocking is
numerically identical to one `sdpa`**; short prompts stay a single block.

Lookup arm: `NV_SUFFIX_DRAFTER`, `NV_SUFFIX_MIN_MATCH`,
`NV_LAGUNA_LOOKUP_DRAFT`, `NV_LOOKUP_MIN_MATCH`, `NV_LAGUNA_LOOKUP_EMA`.

Graphs: `NV_EAGLE3_GRAPH_CHAIN_EAGER`, `NV_EAGLE3_GRAPH_CHAIN_DEBUG`,
`NV_EAGLE3_UNGRAPHED`, `NV_LAGUNA_GRAPH`, `NV_LAGUNA_DFLASH_GRAPH`.

Deep verify rows (measurement only; serving stays at draft ceiling 7):
`NV_VERIFY_ONE_ARITH_ROWS` (default 8, max 24 — extends the fp8 LT span-4 plan
pinning past 8 rows so deep rungs ride the m≤8 plan family),
`NV_NVFP4_GEMV_ROWS` (m1-gemv route row ceiling, default 8).

q38 verify attention: `NV_Q38_MK_VERIFY_FUSED_TC`,
`NV_Q38_MK_VERIFY_FUSED_TC_MIN_TOTAL`, `NV_Q38_MK_VERIFY_TCTILE`,
`NV_Q38_MK_VERIFY_TCTILE_MIN_TOTAL`, `NV_Q38_MK_VERIFY_DUMP`.

Laguna serving: `NV_LAGUNA_SERVE_DRAFT`, `NV_LAGUNA_DFLASH_DIR`,
`NV_LAGUNA_SERVE_STATS`, `NV_LAGUNA_TAP_LIST`, `NV_LAGUNA_NORM_MODE`, `NV_LAGUNA_FP8_KV`,
`NV_LAGUNA_SPEC_MAX_SEQ`. The last caps the spec-serve engine's resident bf16 KV
(clamped to `NV_KV_MAX_SEQ_LEN`; unset or 0 means the full window) — on a
sliding+full hybrid like Laguna-XS that is ~40 KiB per token of full-attention
KV, so 256k resident costs ~10 GiB and a 128k cap returns ~5 GiB. **Requests
exceeding the cap are not truncated**: the eligibility gate routes them to the
per-request path, which still serves the full context at the slower non-spec
rate.

Drafter numerics: `NV_DFLASH_QUANT` (its nvfp4 arm quantizes the lm_head
with the transformer weights), `NV_DFLASH_GEMM`, `NV_DFLASH_ATTN_FP8`,
`NV_DFLASH_DRAFT_F32`, `NV_EAGLE_DRAFT_F32`, `NV_DFLASH_ROPE_THETA`,
`NV_SPEC_VEGAS`, `NV_SPEC_VEGAS_K`, `NV_Q38_DRAFTER_KV_WINDOW` (opt-in ring
over the fp8 drafter KV). The q38 MTP drafter has one numeric stack and no
knob selects another: resident e4m3 linears, the post-norm trunk hidden as
input, an fp8 drafter KV with tensor-core QK at head_dim 256, and the nvfp4
lm_head twin whenever the vocab is a multiple of 256.

Windows: `NV_SPEC_CTX_DISABLE`, `NV_SPEC_PREFILL_CHUNK`.

Spec prefill shadow: every prefill-shadow layer, the 31B's head-dim-512
full-attention layers included, takes candle flash attention
(`spec_prefill_flash_from` in `gemma4.rs`) unless `NV_DETERMINISTIC=1`, which
sends it to the chunked eager f32 path, the reference route. No suite in the
tree pins candle's sm_120 hdim512 forward against eager sdpa, so a candle bump
must re-prove that shape before it ships.

## Connections

02-model-loading.md (drafter discovery, `drafter_target_mismatch`,
`spec_status`); 03.1-mtp-drafter-notes.md (the gemma-4-E4B assistant drafter
contract); 04-kernels-and-quantization.md (`tree_verify_attn_bf16`,
`argmax_bf16`, `dflash_accept_f32`, `token_map_u32`); 05-backends.md (why the
graph and accept kernels are CUDA-only, and `wgpu_spec.rs` as the portable
reference); 06-serving-surface.md (`x-spec-decode`, streaming, admission control
pricing a cached verify graph as sticky VRAM); the basis-tuple quoting rule — every
number deliberately omitted from this chapter needs a basis tuple to be
quotable.
