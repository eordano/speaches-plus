
From Stdlib Require Import ZArith Lia List.
From SpeachesPlus Require Import WindowClamp.
Import ListNotations.
Open Scope Z_scope.

Definition kv_window (prompt_len max_new kv_max : Z) : option (Z * Z) :=
  if kv_max <=? prompt_len then None
  else
    let mn := Z.min max_new (kv_max - (prompt_len + 1)) in
    Some (prompt_len + mn + 1, mn).

Theorem kv_window_none_iff :
  forall p m kmax, kv_window p m kmax = None <-> kmax <= p.
Proof.
  intros p m kmax; unfold kv_window.
  destruct (Z.leb_spec kmax p); split; intros; try discriminate; auto; lia.
Qed.

Theorem kv_window_invariants :
  forall p m kmax cl mn,
    0 <= p -> 0 <= m ->
    kv_window p m kmax = Some (cl, mn) ->
    p < kmax
    /\ mn <= m
    /\ cl <= kmax
    /\ cl = p + mn + 1.
Proof.
  intros p m kmax cl mn Hp Hm H; unfold kv_window in H.
  destruct (Z.leb_spec kmax p); inversion H; subst; lia.
Qed.

Theorem kv_step_in_bounds :
  forall p m kmax cl mn step,
    0 <= p -> 0 <= m ->
    kv_window p m kmax = Some (cl, mn) ->
    0 <= step <= mn ->
    p + step < cl /\ p + step < kmax.
Proof.
  intros p m kmax cl mn step Hp Hm H Hs.
  destruct (kv_window_invariants _ _ _ _ _ Hp Hm H) as (H1 & H2 & H3 & H4); lia.
Qed.

Definition EAGLE3_K_MIN : Z := 2.
Definition EAGLE3_K_MAX : Z := 64.
Definition EAGLE3_K_DEFAULT : Z := 4.
Definition SPEC_VERIFY_HEADROOM : Z := 16.

Definition spec_verify_cache_len (prompt_len max_new k kv_max : Z) : option Z :=
  match kv_window prompt_len max_new kv_max with
  | None => None
  | Some (committed_max, _) => Some (committed_max + k + SPEC_VERIFY_HEADROOM)
  end.

Theorem spec_verify_cache_len_bound :
  forall p m k kmax n,
    0 <= p -> 0 <= m ->
    spec_verify_cache_len p m k kmax = Some n ->
    n <= kmax + k + SPEC_VERIFY_HEADROOM.
Proof.
  intros p m k kmax n Hp Hm H; unfold spec_verify_cache_len in H.
  destruct (kv_window p m kmax) as [[cl mn]|] eqn:E; inversion H; subst.
  destruct (kv_window_invariants _ _ _ _ _ Hp Hm E) as (_ & _ & H3 & _); lia.
Qed.

(* ---------------------------------------------------------------- *)
(* The PRODUCTION spec-verify sizing.                                  *)
(*                                                                     *)
(* spec_verify_cache_len above models the ORIGINAL formula, which can  *)
(* exceed kv_max by k + headroom. The shipped path is                  *)
(* spec_verify_window (oapi/chat_engine/spec_window.rs:152-168), which *)
(* pre-clamps the committed window so the reserve always fits. Rust    *)
(* uses saturating_sub on usize; sat_sub is its faithful Z model.      *)

Definition sat_sub (a b : Z) : Z := Z.max 0 (a - b).

Definition spec_verify_window (prompt_len max_new k kv_max : Z)
  : option (Z * Z) :=
  match kv_window prompt_len max_new kv_max with
  | None => None
  | Some (_, window_clamped) =>
      let reserve := k + SPEC_VERIFY_HEADROOM in
      let room := sat_sub kv_max reserve in
      let clamped := Z.min window_clamped (sat_sub room (prompt_len + 1)) in
      let committed_max := prompt_len + clamped + 1 in
      Some (Z.min (committed_max + reserve) kv_max, clamped)
  end.

(* (a) The allocation never sizes past the KV window. Mirrors the Rust
   test spec_verify_window_never_sizes_past_the_kv_window. *)
Theorem spec_verify_window_le_kv_max :
  forall p m k kmax ms cl,
    spec_verify_window p m k kmax = Some (ms, cl) ->
    ms <= kmax.
Proof.
  intros p m k kmax ms cl H.
  unfold spec_verify_window, kv_window in H.
  destruct (Z.leb_spec kmax p) as [Hle|Hlt]; inversion H; subst; lia.
Qed.

(* (b) The clamp never grows the caller's budget. *)
Theorem spec_verify_window_clamped_le_max_new :
  forall p m k kmax ms cl,
    0 <= p -> 0 <= m ->
    spec_verify_window p m k kmax = Some (ms, cl) ->
    cl <= m /\ 0 <= cl.
Proof.
  intros p m k kmax ms cl Hp Hm H.
  unfold spec_verify_window, kv_window in H.
  destruct (Z.leb_spec kmax p) as [Hle|Hlt]; inversion H; subst.
  unfold sat_sub; split; lia.
Qed.

(* (c) THE SAFETY PROPERTY: whenever there is room for the prompt plus a
   full reserve, the returned size covers the committed prefix AND every
   speculative row -- i.e. the outer Z.min never truncates. This is what
   spec_verify_window_allocation_covers_every_speculative_row asserts by
   enumeration in Rust; here it holds for ALL inputs. *)
Theorem spec_verify_window_covers_speculative_rows :
  forall p m k kmax ms cl,
    0 <= p -> 0 <= m -> 0 <= k ->
    p + 1 + (k + SPEC_VERIFY_HEADROOM) <= kmax ->
    spec_verify_window p m k kmax = Some (ms, cl) ->
    ms = (p + cl + 1) + (k + SPEC_VERIFY_HEADROOM).
Proof.
  intros p m k kmax ms cl Hp Hm Hk Hroom H.
  unfold spec_verify_window, kv_window in H.
  destruct (Z.leb_spec kmax p) as [Hle|Hlt]; inversion H; subst.
  unfold sat_sub, SPEC_VERIFY_HEADROOM in *; lia.
Qed.

(* (d) The boundary case the guard exists for: when the box is too tight
   to hold prompt + reserve, the clamp collapses to zero, so there are no
   speculative rows to under-cover. Together with (c) this partitions the
   whole input space -- there is no third case where the allocation is
   short. *)
Theorem spec_verify_window_tight_box_yields_no_budget :
  forall p m k kmax ms cl,
    0 <= p -> 0 <= m -> 0 <= k ->
    kmax < p + 1 + (k + SPEC_VERIFY_HEADROOM) ->
    spec_verify_window p m k kmax = Some (ms, cl) ->
    cl = 0.
Proof.
  intros p m k kmax ms cl Hp Hm Hk Htight H.
  unfold spec_verify_window, kv_window in H.
  destruct (Z.leb_spec kmax p) as [Hle|Hlt]; inversion H; subst.
  unfold sat_sub, SPEC_VERIFY_HEADROOM in *; lia.
Qed.

(* (e) The shipped sizing is never looser than the original formula: the
   refactor that introduced spec_verify_window only tightened it. *)
Theorem spec_verify_window_tighter_than_cache_len :
  forall p m k kmax ms cl n,
    0 <= p -> 0 <= m -> 0 <= k ->
    spec_verify_window p m k kmax = Some (ms, cl) ->
    spec_verify_cache_len p m k kmax = Some n ->
    ms <= n.
Proof.
  intros p m k kmax ms cl n Hp Hm Hk Hw Hc.
  unfold spec_verify_window, spec_verify_cache_len, kv_window in Hw, Hc.
  destruct (Z.leb_spec kmax p) as [Hle|Hlt];
    inversion Hw; inversion Hc; subst.
  unfold sat_sub, SPEC_VERIFY_HEADROOM in *; lia.
Qed.

Definition VERIFY_CACHE_GRAIN : Z := 256.

Definition verify_cache_capacity (needed : Z) : Z :=
  (Z.max needed 1 + (VERIFY_CACHE_GRAIN - 1)) / VERIFY_CACHE_GRAIN
  * VERIFY_CACHE_GRAIN.

Theorem verify_cache_capacity_ge :
  forall n, n <= verify_cache_capacity n.
Proof.
  intros n; unfold verify_cache_capacity, VERIFY_CACHE_GRAIN.
  pose proof (Z_div_mod_eq_full (Z.max n 1 + (256 - 1)) 256).
  pose proof (Z.mod_pos_bound (Z.max n 1 + (256 - 1)) 256 ltac:(lia)).
  lia.
Qed.

Theorem verify_cache_capacity_tight :
  forall n, 1 <= n -> verify_cache_capacity n < n + VERIFY_CACHE_GRAIN.
Proof.
  intros n Hn; unfold verify_cache_capacity, VERIFY_CACHE_GRAIN.
  pose proof (Z_div_mod_eq_full (Z.max n 1 + (256 - 1)) 256).
  pose proof (Z.mod_pos_bound (Z.max n 1 + (256 - 1)) 256 ltac:(lia)).
  lia.
Qed.

Theorem verify_cache_capacity_aligned :
  forall n, (verify_cache_capacity n) mod VERIFY_CACHE_GRAIN = 0.
Proof.
  intros n; unfold verify_cache_capacity; apply Z_mod_mult.
Qed.

Theorem verify_cache_capacity_mono :
  forall a b, a <= b -> verify_cache_capacity a <= verify_cache_capacity b.
Proof.
  intros a b Hab; unfold verify_cache_capacity, VERIFY_CACHE_GRAIN.
  pose proof (Z_div_mod_eq_full (Z.max a 1 + (256 - 1)) 256).
  pose proof (Z.mod_pos_bound (Z.max a 1 + (256 - 1)) 256 ltac:(lia)).
  pose proof (Z_div_mod_eq_full (Z.max b 1 + (256 - 1)) 256).
  pose proof (Z.mod_pos_bound (Z.max b 1 + (256 - 1)) 256 ltac:(lia)).
  lia.
Qed.

Definition SLIDING_WINDOW : Z := 1024.
Definition SLIDING_COMPACT_SLACK : Z := 256.

Definition layer_cap (max_seq : Z) (k : layer_kind) : Z :=
  match k with
  | Full => max_seq
  | Sliding => Z.min max_seq (SLIDING_WINDOW + SLIDING_COMPACT_SLACK)
  end.

Theorem layer_cap_le_max_seq :
  forall s k, 0 <= s -> layer_cap s k <= s.
Proof. intros s [|]; cbn; lia. Qed.

Theorem layer_cap_ge_window :
  forall s, SLIDING_WINDOW <= s -> SLIDING_WINDOW <= layer_cap s Sliding.
Proof. cbn; unfold SLIDING_WINDOW, SLIDING_COMPACT_SLACK; intros; lia. Qed.

Corollary sliding_layer_cache_sound :
  forall s qpos kpos,
    SLIDING_WINDOW <= s ->
    attends qpos kpos (tree_layer_window SLIDING_WINDOW Sliding) ->
    qpos - layer_cap s Sliding < kpos <= qpos.
Proof.
  intros s qpos kpos Hs Hatt.
  rewrite sliding_layer_window_id in Hatt.
  apply (cap_retains_attended qpos kpos SLIDING_WINDOW (layer_cap s Sliding)).
  - unfold SLIDING_WINDOW; lia.
  - apply layer_cap_ge_window; exact Hs.
  - exact Hatt.
Qed.

Definition n_kv (k : layer_kind) : Z :=
  match k with Sliding => 16 | Full => 4 end.
Definition head_dim (k : layer_kind) : Z :=
  match k with Sliding => 256 | Full => 512 end.

Definition fp8_row_bytes (k : layer_kind) : Z :=
  2 * (n_kv k * head_dim k) + 2 * (4 * n_kv k).

Lemma fp8_row_bytes_sliding : fp8_row_bytes Sliding = 8320.
Proof. reflexivity. Qed.
Lemma fp8_row_bytes_full : fp8_row_bytes Full = 4128.
Proof. reflexivity. Qed.

Definition gemma4_pattern : list layer_kind :=
  [Sliding; Sliding; Sliding; Sliding; Sliding; Full].
Definition gemma4_layers : list layer_kind :=
  concat (repeat gemma4_pattern 10).

Lemma gemma4_layers_length : length gemma4_layers = 60%nat.
Proof. reflexivity. Qed.

Definition sum_layers (f : layer_kind -> Z) (l : list layer_kind) : Z :=
  fold_right (fun k acc => f k + acc) 0 l.

Lemma sum_layers_gemma4 :
  forall f, sum_layers f gemma4_layers = 50 * f Sliding + 10 * f Full.
Proof.
  intros f; unfold gemma4_layers, gemma4_pattern;
    cbn [sum_layers fold_right concat repeat app]; ring.
Qed.

Definition kv_bytes_flat (max_seq : Z) : Z :=
  sum_layers (fun k => fp8_row_bytes k * max_seq) gemma4_layers.

Definition kv_bytes_hybrid (max_seq : Z) : Z :=
  sum_layers (fun k => fp8_row_bytes k * layer_cap max_seq k) gemma4_layers.

Lemma kv_bytes_flat_closed :
  forall s, kv_bytes_flat s = 457280 * s.
Proof.
  intros s; unfold kv_bytes_flat; rewrite sum_layers_gemma4; cbn beta.
  rewrite fp8_row_bytes_sliding, fp8_row_bytes_full; ring.
Qed.

Lemma kv_bytes_hybrid_closed :
  forall s, kv_bytes_hybrid s = 416000 * Z.min s 1280 + 41280 * s.
Proof.
  intros s; unfold kv_bytes_hybrid; rewrite sum_layers_gemma4; cbn beta.
  rewrite fp8_row_bytes_sliding, fp8_row_bytes_full; cbn [layer_cap].
  replace (SLIDING_WINDOW + SLIDING_COMPACT_SLACK) with 1280 by reflexivity.
  ring.
Qed.

Theorem kv_bytes_hybrid_mono :
  forall a b, a <= b -> kv_bytes_hybrid a <= kv_bytes_hybrid b.
Proof. intros a b H; rewrite !kv_bytes_hybrid_closed; lia. Qed.

Theorem kv_bytes_flat_mono :
  forall a b, a <= b -> kv_bytes_flat a <= kv_bytes_flat b.
Proof. intros a b H; rewrite !kv_bytes_flat_closed; lia. Qed.

Theorem hybrid_le_flat :
  forall s, 0 <= s -> kv_bytes_hybrid s <= kv_bytes_flat s.
Proof.
  intros s Hs; rewrite kv_bytes_hybrid_closed, kv_bytes_flat_closed; lia.
Qed.

Theorem hybrid_eq_flat_below_cap :
  forall s, 0 <= s <= 1280 -> kv_bytes_hybrid s = kv_bytes_flat s.
Proof.
  intros s Hs; rewrite kv_bytes_hybrid_closed, kv_bytes_flat_closed; lia.
Qed.

Definition GiB : Z := 1073741824.
Definition FULL_CONTEXT : Z := 262144.
Definition VRAM_BUDGET : Z := 76 * GiB.

Theorem flat_full_context_impossible :
  VRAM_BUDGET < kv_bytes_flat FULL_CONTEXT.
Proof.
  rewrite kv_bytes_flat_closed; apply Z.ltb_lt; vm_compute; reflexivity.
Qed.

Theorem flat_full_context_value :
  kv_bytes_flat FULL_CONTEXT = 119873208320.
Proof. rewrite kv_bytes_flat_closed; vm_compute; reflexivity. Qed.

Theorem hybrid_full_context_value :
  kv_bytes_hybrid FULL_CONTEXT = 11353784320.
Proof. rewrite kv_bytes_hybrid_closed; vm_compute; reflexivity. Qed.

Theorem kv_budget_guard :
  forall base ctx budget,
    0 <= ctx <= FULL_CONTEXT ->
    base + kv_bytes_hybrid FULL_CONTEXT <= budget ->
    base + kv_bytes_hybrid ctx <= budget.
Proof.
  intros base ctx budget Hctx Hguard.
  pose proof (kv_bytes_hybrid_mono ctx FULL_CONTEXT ltac:(lia)); lia.
Qed.

Definition MEASURED_SERVER_FOOTPRINT : Z := 39835821670.

Theorem full_context_fits_76GiB :
  forall ctx,
    0 <= ctx <= FULL_CONTEXT ->
    MEASURED_SERVER_FOOTPRINT + kv_bytes_hybrid ctx <= VRAM_BUDGET.
Proof.
  intros ctx Hctx; apply kv_budget_guard; [exact Hctx|].
  rewrite hybrid_full_context_value; apply Z.leb_le; vm_compute; reflexivity.
Qed.

Definition window_blocks (w bs : Z) : Z := (w + bs - 1) / bs + 1.

Theorem window_blocks_cover :
  forall w bs, 0 < bs -> 0 <= w -> w + 1 <= window_blocks w bs * bs.
Proof.
  intros w bs Hbs Hw; unfold window_blocks.
  pose proof (Z_div_mod_eq_full (w + bs - 1) bs).
  pose proof (Z.mod_pos_bound (w + bs - 1) bs Hbs).
  lia.
Qed.

Definition layer_blocks (w bs full_blocks : Z) (k : layer_kind) : Z :=
  match k with
  | Sliding => Z.min (window_blocks w bs) full_blocks
  | Full => full_blocks
  end.

Theorem layer_blocks_le_full :
  forall w bs fb k, 0 <= fb -> layer_blocks w bs fb k <= fb.
Proof. intros w bs fb [|]; cbn; lia. Qed.
