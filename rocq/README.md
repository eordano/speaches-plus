# rocq/ -- mechanized efficiency & safety theorems for speaches-plus

Machine-checked (Rocq Prover 9.1.1, stdlib 9.0.0 -- `build.sh` resolves the
toolchain from `COQBIN`/`ROCQPATH`, PATH, or nixpkgs) theorems mapping the software
(kernel schedules, cache sizing, spec-decode loop) and an axiomatized
hardware model (HBM bandwidth, cache hierarchy) to validated efficiency and
safety claims. Core modules were written against an earlier main; the
util-formal lane extended the development to the MERGED stack
(Stream-K auto, ring-buffer hybrid KV + budget gate, chunked
prefills) and re-instantiated the generated modules from the merged
binary's measurements.

The `.v` files carry no comments (fleet convention). This README says what
each module proves; source anchors, faithfulness arguments and per-module
scope limits live in
[`../docs/book/08.3-rocq-proof-notes.md`](../docs/book/08.3-rocq-proof-notes.md)
-- update that file when an anchor moves, and read it before citing a
theorem.

Build + independent kernel re-check:

    ./build.sh        # coqc all modules (core + generated), then rocqchk
    ./gen/run.sh      # REGENERATE GenTraffic.v/GenRoofline.v from the live
                      # config + checkpoint + current measurements, then build

All theorems CHECKED. Assumption audit (`Print Assumptions`):

- `WindowClamp`, `KvBudget`, `AcceptSoundness`: **closed under the global
  context** -- zero axioms.
- `Roofline`: only Stdlib's classical-reals construction axioms
  (`sig_forall_dec`, `functional_extensionality_dep`). The hardware model is
  expressed as explicit hypotheses (`respects_bandwidth`,
  `weight_traffic_lb`), not global axioms, so every physics assumption is
  visible in each theorem statement.
- `RoPE`: the same classical-reals base plus `sig_not_dec` (comes in with
  `cos`/`sin`, not with anything this module states). No domain-specific
  axiom, no admitted lemma; `halfsplit_perm_interleave` is **closed under
  the global context**. `RoPE.v` is not yet in `build.sh`'s `MODS` list;
  reproduce with `coqc -R . SpeachesPlus RoPE.v` plus a one-line
  `Print Assumptions` driver, or append `RoPE` to `MODS`.

One caveat not repeated in 08.3: `KvBudget`'s hybrid function models the
window-capped policy that `Gemma4KvCache` implements for bf16 today applied
to the fp8 geometry; `Gemma4KvCacheFp8::new` is still flat at HEAD
(gemma4.rs:1357-1393) -- the theorem is the safety case for the k256 lane's
conversion, plus the proof that the flat layout can never reach 262144.

## Modules

| Module | Lane theorem | Content |
|---|---|---|
| `WindowClamp.v` | (d) | The clamped iteration range in `tree_verify_attn.cu:72-79,92,111-113` equals the sliding-window attention set `tree_window_attends` (`gemma4.rs:1484-1492`) for all positions, both for the committed prefix (`committed_range_exact`) and the tree tokens (`tree_kept_iff_attends`); full layers (`tree_layer_window` = 0) degrade to exactly causal. Plus the geometric fact that makes window-capping sound: attended keys live in the last `window` positions (`cap_retains_attended`). |
| `KvBudget.v` | (b) | Faithful Z-ports of `kv_window` (`oapi/chat_engine/spec_window.rs:40-50`), `spec_verify_cache_len` (the pre-refactor formula), the SHIPPED `spec_verify_window` (`spec_window.rs:152-168`, added 2026-08-10 -- five theorems: never sizes past the window, never grows the caller's budget, **covers every speculative row whenever the box has room**, collapses the budget to zero exactly when it does not, and is never looser than the pre-refactor formula), `verify_cache_capacity`, the per-layer cap policy (`gemma4.rs:1139-1146`) and the fp8 row geometry (`gemma4.rs:1370-1385`). Proves the full `assert_kv_window_invariants` contract (2185-2207), grain-alignment/tightness/monotonicity of the verify-cache round-up, the closed forms `flat = 457280*ctx` and `hybrid = 416000*min(ctx,1280) + 41280*ctx` **derived from the 60-layer fold**, monotonicity, `flat(262144) = 119,873,208,320 B > 76 GiB` (flat is impossible), `hybrid(262144) = 11,353,784,320 B`, and the guard: measured 37.1 GiB footprint + hybrid KV <= 76 GiB **for every ctx <= 262144** (`full_context_fits_76GiB`). Also the `from_gemma4_hybrid` block math (`paged_fp8.rs:51-75`). |
| `AcceptSoundness.v` | (c) | Greedy distribution preservation: modeling the target's greedy decoder as an oracle `next`, the shift-mode accept rule (accept iff drafted == row argmax, `chat_engine.rs:1746-1756`) emits **exactly** the target-alone greedy stream for *arbitrary* drafters (`spec_loop_emits_greedy`), any round count, including early truncation (`spec_truncated_emits_greedy`) -- by induction over rounds with the invariant *bonus = next(ctx)*. Corollary `drafter_irrelevant`: drafts can change only speed, never content. |
| `KvBudgetMerged.v` | util-formal | Z-port of the MERGED `kv_budget()` fold (`kv_budget` in `gemma4.rs`) + `enforce_gemma4_vram_budget` gate (`oapi/chat_engine/build.rs`): closed forms derived from the 60-layer fold (`verify/decode = 50*min(2176,kvm)*8320 + 10*kvm*4128`), value lemmas (`worst_total(262144) = 27,782,103,040 B = 25.87 GiB`, matching the k256-validate measured ladder), monotonicity, the startup gate, now modelled as the RULE admission.rs applies rather than the 76 GiB constant it used to hardcode (`device_budget total = 0.8*total`; `full_window_required_bytes = 67,871,391,744 B = 63.21 GiB`; `min_total_for_full_window = 84,839,239,680 B = 79.01 GiB` and tight to the byte; a 79 GiB card refuses; the 16 GiB no-device fallback refuses at every context length; `0.8 * 95 GiB = 76 GiB` exactly, so the old constant was this machine's card written down as a property of the model), ring safety (`chunk <= ring - window + 1` can never fire on the chunked paths), and the env-clamp admissibility of `spec_prefill_chunk`. |
| `StreamK.v` | util-formal | The Stream-K promotion cannot invalidate any floor: weight traffic is invariant under K-partition (`streamk_weight_traffic_invariant` -- the manifest formula counts Stream-K and DP identically), extra workspace traffic only helps satisfy the compulsory-traffic hypothesis (`streamk_extra_traffic_preserves_floor`), and the floors quantify over every execution anyway. Pins the auto routing heuristic (`ceil(m/128)*ceil(n/128) <= 192`, nvfp4.rs:424-437): `down` routes to Stream-K, `gate_up` stays DP, monotone in n. |
| `ChunkedPrefill.v` | util-formal | The chunked prefill loops (`chat_engine.rs:1852-1887` spec, `:1524-1533` non-spec) cover the prompt exactly in pieces of <= 1 chunk (`chunk_pieces_cover`, `chunk_pieces_bounded`); any affine per-forward transient therefore peaks at `f(chunk)`, independent of prompt length (`peak_transient_bounded_by_chunk`); the seed-row lm_head makes the logits transient a 512 KiB constant vs the earlier `vocab*2*prompt` (7.08 GiB at the 14.5k repro -- `logit_transient_ratio_14500`). |
| `Roofline.v` | (a) | I/O lower bound: any schedule of one verify forward moves >= W - L2 bytes over HBM, so `T_verify >= (W-L2)/B` (`verify_time_lower_bound[_div]`), and the speed-of-light ceiling `r*(W-L2) <= tpr*B` for any schedule (`speed_of_light`). Instantiated at the measured operating point (road-to-50.md §2): verify floor >= 10 ms (`verify_floor_10ms`), and at measured acceptance (2.193 tok/round, shift-K=4) `r <= 218 tok/s` (`token_rate_ceiling_218`); even an oracle K=4 drafter is capped at 497 tok/s (`token_rate_ceiling_oracle_K4`). Constants superseded by the generated instances -- see the headline correction below. |
| `RoPE.v` | rope-formal | RoPE's relative-position property over `R`, stated so that it covers the pairing our code actually uses. **Pairing invariance** (`pairing_invariance`): for ANY assembly of rotated pairs into a flat vector that preserves the inner product (`faithful`), `<asm(R_a x), asm(R_b y)> = <asm(R_{a-b} x), asm(y)>` -- with `any_pairing_faithful` proving that every *joint* coordinate permutation is such an assembly (`dot_perm`, `sumprod_perm`), and `halfsplit_perm_interleave` proving that our half-split layout (`rope.rs:123-127`, `cuda/rope.cu:33-36`, HF `rotate_half`) is a joint permutation of RoFormer's interleaved one -- hence `layout_equivalence`: the two conventions give *identical* attention scores. The side condition is load-bearing and proved so: `mixed_pairing_unsound` exhibits explicit vectors and angles (`PI/2`) where q half-split against k interleaved **breaks** the identity (0 vs -1). **Partial-rotary soundness** (`partial_relative_split`): with the zero-padded `inv_freq` tail of `gemma4.rs:2586-2591`, the padded pairs are untouched (`partial_tail_identity`), so the score splits into a relative-position term over the rotated block plus a position-*independent* term over the rest (`partial_tail_position_independent`). Instantiated at the Gemma4 full-attention config -- `rope_angles 1 4 512 = 64` of `512/2 = 256` pairs rotate, 192 are identity (`gemma4_full_partial_relative`, `gemma4_full_partial_pairing_invariance`). Plus the cheap ones: `rot_0`/`rope_0` (R_0 = id), `rot_compose`/`rope_compose` (R_a . R_b = R_{a+b}), `rot_orthogonal`/`rope_orthogonal`/`rope_norm` (norm preservation). |

## gen/ -- automated theorem generation (rocq-gen lane)

`gen/gen.py` maps software + hardware into Rocq obligations automatically, so
every future kernel/config change re-derives its efficiency bound instead of
trusting a stale analysis. Inputs:

- **the model `config.json`** (shapes, 50/10 sliding/full layer mix,
  `attention_k_eq_v`, quantization ignore list) and the drafter `config.json`
  -- paths from `NV_GEMMA4_VERIFIER_DIR` / `NV_EAGLE3_DRAFT_DIR`;
- **`gen/machine.json`** -- machine constants (HBM bandwidth, L2-residency
  credit), provisional pending the sol-roofline lane's measured section;
- **`gen/kernels.json`** -- the per-kernel byte-traffic manifest for the
  verify/draft path: every formula is a compulsory-HBM-bytes lower bound as a
  function of (ctx, K, shapes), with file:line citations into gemma4.rs,
  tree_verify_attn.cu and eagle3_loader.rs, plus checkpoint tensor witnesses;
- **`gen/measured.json`** -- pointers to the CURRENT bench artifacts; gen.py
  parses the `NV_PROF_CHAT` GRAPHED SUMMARY lines (steady-state envelope) and
  the bench JSON aggregate at generation time.

Two layers of cross-checking make the generated numbers hard to fake:

1. gen.py scans the **safetensors headers directly** (independent byte
   inventory) and emits the manifest-formula total and the checkpoint total as
   *separate* Rocq definitions whose equality is re-proved by `vm_compute`
   inside the kernel (`verify_weight_crosscheck`, `draft_step_crosscheck`);
2. **consistency tripwires**: `gen_verify_floor_consistent` /
   `gen_draft_floor_consistent` prove the derived physical floor sits *below*
   the measured wall time -- if a manifest edit ever overcounts compulsory
   bytes, generation aborts and the proofs fail. The bound cannot silently
   inflate.

### Generated theorem inventory (values re-derived at the post-Stream-K-merge main from an internal server-sol-defaults measurement log; pre-merge values in parens)

| Theorem (GenRoofline.v) | Statement | Value |
|---|---|---|
| `gen_verify_weight_floor` | no schedule finishes one verify forward faster than the weight stream | >= 17.438 ms |
| `gen_verify_floor_full_ctx` | verify floor at ctx=262144 incl. clamped-KV unique-byte reads | >= 23.904 ms |
| `gen_draft_step_floor` | drafter chain-step floor | >= 675.8 us |
| `gen_token_rate_ceiling` | speed of light at measured acceptance (tpr <= 2.21 on the merged bench sample) | <= 127 tok/s (was 134 @ tpr <= 2.33) |
| `gen_round_serial_ceiling` | ceiling for our serial draft->verify round shape | <= 110 tok/s (was 116) |
| `gen_verify_within_light_ratio` | measured verify <= r x theoretical min | r = 1.81 (was 1.96 pre-Stream-K) |
| `gen_draft_within_light_ratio` | measured K=4 draft chain <= r x theoretical min | r = 1.52 (was 1.61) |
| `gen_e2e_within_light_ratio` | SoL ceiling <= r x measured aggregate (57.27 tok/s) | r = 2.22 (was 2.38) |
| `gen_gap_decomposition` | round gap = verify-gap + draft-gap + host-gap, exact accounting | 15.80 ms total |
| `gen_gap_lives_in_verify` | >= 89% of the round gap is verify in-kernel | 14.12 ms of 15.80 |
| `gen_util_headline` | ANY schedule at measured acceptance <= r x what we already deliver | r = 2.22 |
| `bw_hypothesis_consistent` | best measured read BW <= the BW hypothesis (soundness direction) | emitted when machine.json carries a measured value |
| `verify_weight_crosscheck` (GenTraffic.v) | manifest formulas = safetensors inventory | 31,481,767,960 B |
| `kv_read_bytes_mono` | KV read floor monotone in ctx | -- |

Assumption audit: all GenTraffic lemmas are **closed under the global
context** (zero axioms); GenRoofline theorems carry only the Stdlib
classical-reals base, same as core `Roofline.v`.

**Headline correction produced by the generator**: the road-to-50 §2 traffic
estimate (18.3 GB/verify, "~30% of bandwidth, ~10.3 ms floor") undercounted
the verify stream -- it treated attention as fp4, but every `self_attn` is in
the quantization ignore list and `load_qkv_fused`/`load_attn_proj`
(gemma4.rs:3127-3176) load it bf16. Checkpoint-derived truth: **31.48 GB**
per verify forward (16.95 GB bf16 attention + 11.71 GB NVFP4 MLP + 2.82 GB
bf16 lm_head). Hence verify at 33.5-34.2 ms is already **within 1.96x of the
bandwidth roof** (~52% of peak), not 3.3x; the honest ceiling at measured
acceptance is **134 tok/s**, not 218. Core `Roofline.v` `Measured.*`
corollaries remain *sound* (a smaller assumed W only weakens the bound) but
are superseded by the generated, tighter instances. Consequence for planning:
the single biggest byte lever left in verify is the 16.95 GB bf16 attention
weight stream (54% of the pass) -- quantizing it (fp8) would cut the floor to
~12.7 ms; schedule/launch tuning alone can recover at most the 1.96x gap.
