
From Stdlib Require Import ZArith Lia List.
Import ListNotations.
Open Scope Z_scope.

Definition window : Z := 1024.
Definition chunk : Z := 1024.
Definition headroom : Z := 128.
Definition nkv_s : Z := 16.  Definition hd_s : Z := 256.
Definition nkv_f : Z := 4.   Definition hd_f : Z := 512.
Definition drafter_row_elems : Z := 4096.
Definition kv_max_default : Z := 262144.

Inductive lkind := Sliding | Full.

Definition pattern : list lkind :=
  [Sliding; Sliding; Sliding; Sliding; Sliding; Full].
Definition layer_types : list lkind := concat (repeat pattern 10).

Lemma layer_count : Z.of_nat (length layer_types) = 60.
Proof. vm_compute. reflexivity. Qed.

Definition nkv_for (k : lkind) : Z :=
  match k with Sliding => nkv_s | Full => nkv_f end.
Definition hd_for (k : lkind) : Z :=
  match k with Sliding => hd_s | Full => hd_f end.
Definition stride_for (k : lkind) : Z := nkv_for k * hd_for k.

Definition ring_slots : Z := window + chunk + headroom.
Lemma ring_slots_val : ring_slots = 2176.
Proof. vm_compute. reflexivity. Qed.

Definition fp8_bytes_per_row (k : lkind) : Z :=
  2 * stride_for k + 2 * nkv_for k * 4.
Definition bf16_bytes_per_row (k : lkind) : Z := 2 * stride_for k * 2.

Lemma fp8_row_sliding : fp8_bytes_per_row Sliding = 8320.
Proof. vm_compute. reflexivity. Qed.
Lemma fp8_row_full : fp8_bytes_per_row Full = 4128.
Proof. vm_compute. reflexivity. Qed.

Definition vslots (fp8 rings : bool) (kvm : Z) (k : lkind) : Z :=
  match k, fp8, rings with
  | Sliding, true, true => Z.min ring_slots kvm
  | _, _, _ => kvm
  end.

Definition dslots (rings : bool) (kvm : Z) (k : lkind) : Z :=
  match k, rings with
  | Sliding, true => Z.min ring_slots kvm
  | _, _ => kvm
  end.

Definition v_bytes_per_row (fp8 : bool) (k : lkind) : Z :=
  if fp8 then fp8_bytes_per_row k else bf16_bytes_per_row k.

Definition verify_bytes (fp8 rings : bool) (kvm : Z) : Z :=
  fold_right
    (fun k acc => vslots fp8 rings kvm k * v_bytes_per_row fp8 k + acc)
    0 layer_types.
Definition decode_bytes (rings : bool) (kvm : Z) : Z :=
  fold_right
    (fun k acc => dslots rings kvm k * fp8_bytes_per_row k + acc)
    0 layer_types.

Definition scratch_rows (kvm : Z) : Z := Z.min kvm 4096.
Definition verify_scratch (fp8 : bool) (kvm : Z) : Z :=
  (if fp8
   then 2 * scratch_rows kvm * 4096 + 2 * scratch_rows kvm * 16 * 4
   else 2 * scratch_rows kvm * 4096 * 2)
  + scratch_rows kvm * 4.

Definition drafter_bytes (kvm : Z) : Z := kvm * drafter_row_elems * 2 * 2.

Definition verify_total (fp8 rings : bool) (kvm : Z) : Z :=
  verify_bytes fp8 rings kvm + verify_scratch fp8 kvm.
Definition worst_total (fp8 rings : bool) (kvm : Z) : Z :=
  verify_total fp8 rings kvm + decode_bytes rings kvm + drafter_bytes kvm.

Lemma verify_bytes_closed :
  forall kvm, verify_bytes true true kvm
    = 50 * Z.min ring_slots kvm * 8320 + 10 * kvm * 4128.
Proof.
  intros.
  cbv [verify_bytes layer_types pattern vslots v_bytes_per_row
       fp8_bytes_per_row stride_for nkv_for hd_for nkv_s hd_s nkv_f hd_f concat repeat app
       fold_right].
  set (s := Z.min ring_slots kvm).
  lia.
Qed.

Lemma decode_bytes_closed :
  forall kvm, decode_bytes true kvm
    = 50 * Z.min ring_slots kvm * 8320 + 10 * kvm * 4128.
Proof.
  intros.
  cbv [decode_bytes layer_types pattern dslots fp8_bytes_per_row
       stride_for nkv_for hd_for nkv_s hd_s nkv_f hd_f concat repeat app
       fold_right].
  set (s := Z.min ring_slots kvm).
  lia.
Qed.

Lemma verify_bytes_flat_closed :
  forall kvm, verify_bytes true false kvm = kvm * 457280.
Proof.
  intros.
  cbv [verify_bytes layer_types pattern vslots v_bytes_per_row
       fp8_bytes_per_row stride_for nkv_for hd_for nkv_s hd_s nkv_f hd_f concat repeat app
       fold_right].
  lia.
Qed.

Lemma verify_total_262144 :
  verify_total true true kv_max_default = 11760615424.
Proof. vm_compute. reflexivity. Qed.

Lemma decode_bytes_262144 : decode_bytes true kv_max_default = 11726520320.
Proof. vm_compute. reflexivity. Qed.

Lemma drafter_bytes_262144 : drafter_bytes kv_max_default = 4294967296.
Proof. vm_compute. reflexivity. Qed.

Lemma worst_total_262144 :
  worst_total true true kv_max_default = 27782103040.
Proof. vm_compute. reflexivity. Qed.

Lemma verify_bytes_mono :
  forall a b, 0 <= a -> a <= b ->
    verify_bytes true true a <= verify_bytes true true b.
Proof.
  intros a b Ha Hab. rewrite !verify_bytes_closed.
  assert (Z.min ring_slots a <= Z.min ring_slots b) by lia.
  assert (0 <= Z.min ring_slots a) by (unfold ring_slots, window, chunk, headroom; lia).
  nia.
Qed.

Lemma worst_total_mono :
  forall a b, 0 <= a -> a <= b ->
    worst_total true true a <= worst_total true true b.
Proof.
  intros a b Ha Hab.
  unfold worst_total, verify_total, drafter_bytes, verify_scratch,
    scratch_rows, drafter_row_elems.
  pose proof (verify_bytes_mono a b Ha Hab).
  rewrite !decode_bytes_closed.
  assert (Z.min ring_slots a <= Z.min ring_slots b) by lia.
  assert (0 <= Z.min ring_slots a) by (unfold ring_slots, window, chunk, headroom; lia).
  assert (Z.min a 4096 <= Z.min b 4096) by lia.
  nia.
Qed.

Definition weights_measured_bytes : Z := 38200 * 1024 * 1024.
Definition budget_bytes : Z := 76 * 1024 * 1024 * 1024.

(* The hd512 verify scratch buffer. enforce_gemma4_vram_budget adds it to
   BOTH the logged total and the refusal test (build.rs:731, 770-772), but
   this model omitted it, so the gate proved here was strictly WEAKER than
   the one production enforces -- the proof said "fits" about a predicate
   the server does not implement. Value:
   gqa512_scratch_elems(n_q=32, m=8, splits=64) * 4 B
   = 32*8*64*(kHD+2) * 4 = 32*8*64*514*4 (gqa_verify_hd512.cu:418-422,
   kHD=512; gemma4.rs:2683-2705). It is 0 unless fp8 verify KV AND the
   hd512 mk-verify path are both on; the on-value is the conservative one
   and the one the shipped default takes. *)
Definition hd512_scratch_bytes : Z := 32 * 8 * 64 * 514 * 4.

Lemma hd512_scratch_val : hd512_scratch_bytes = 33685504.
Proof. vm_compute. reflexivity. Qed.

(* Faithful port of the production predicate (build.rs:770-772):
     total = weights + worst_total + hd512_scratch  >  budget  ->  REFUSE
   so the gate ACCEPTS iff total <= budget. *)
Definition gate_accepts (weights kvm : Z) : Prop :=
  weights + worst_total true true kvm + hd512_scratch_bytes <= budget_bytes.

Theorem budget_gate_at_full_window :
  gate_accepts weights_measured_bytes kv_max_default.
Proof. unfold gate_accepts. vm_compute. intro H; discriminate H. Qed.

Theorem budget_gate_all_ctx :
  forall kvm, 0 <= kvm -> kvm <= kv_max_default ->
    gate_accepts weights_measured_bytes kvm.
Proof.
  intros kvm H0 Hub. unfold gate_accepts.
  pose proof (worst_total_mono kvm kv_max_default H0 Hub).
  pose proof budget_gate_at_full_window. unfold gate_accepts in *. lia.
Qed.

(* The scratch term is not noise at the margin: it is 32 MiB, and the proof
   above now carries it. Recording the headroom makes future drift visible
   as a number rather than as a silent re-proof. *)
Theorem budget_headroom_at_full_window :
  budget_bytes - (weights_measured_bytes + worst_total true true kv_max_default
                  + hd512_scratch_bytes) = 13732986880.
Proof. vm_compute. reflexivity. Qed.

(* ---------------------------------------------------------------------- *)
(* The unmeasured-weights hole, made formal.                               *)
(*                                                                         *)
(* build.rs:733 reads the weights term from nvidia-smi and build.rs:770    *)
(* substitutes weights_gib.unwrap_or(0.0) when that fails. The server logs *)
(* "BUDGET GUARD IS INOPERATIVE" and then PROCEEDS. So on that path the    *)
(* gate is not a weaker check -- it is vacuous at every context length     *)
(* this model covers. Proving that explicitly turns a hidden bypass into a *)
(* stated theorem: anyone re-reading the gate proofs now has to read this  *)
(* one too. *)
Definition gate_accepts_unmeasured (kvm : Z) : Prop := gate_accepts 0 kvm.

Theorem unmeasured_weights_gate_is_vacuous :
  forall kvm, 0 <= kvm -> kvm <= kv_max_default -> gate_accepts_unmeasured kvm.
Proof.
  intros kvm H0 Hub. unfold gate_accepts_unmeasured, gate_accepts.
  pose proof (worst_total_mono kvm kv_max_default H0 Hub) as Hmono.
  assert (Hfit : worst_total true true kv_max_default + hd512_scratch_bytes
                 <= budget_bytes) by (vm_compute; intro C; discriminate C).
  lia.
Qed.

(* And it is vacuous for a reason worth stating quantitatively: the KV term
   alone cannot reach the budget, so with weights counted as zero NO context
   length in range can ever trip the refusal. The bypass is total, not
   partial. *)
Theorem unmeasured_weights_cannot_refuse_any_ctx :
  forall weights kvm,
    0 <= kvm -> kvm <= kv_max_default ->
    weights = 0 ->
    weights + worst_total true true kvm + hd512_scratch_bytes < budget_bytes.
Proof.
  intros weights kvm H0 Hub Hw; subst.
  pose proof (worst_total_mono kvm kv_max_default H0 Hub) as Hmono.
  assert (Hfit : worst_total true true kv_max_default + hd512_scratch_bytes
                 < budget_bytes) by (vm_compute; reflexivity).
  lia.
Qed.

Theorem ring_admits_prefill_chunk : chunk <= ring_slots - window + 1.
Proof. vm_compute. intro H; discriminate H. Qed.

Definition spec_prefill_chunk_env (requested : Z) : Z :=
  Z.min (Z.min (Z.max requested 16) 65535) chunk.

Theorem spec_chunk_always_admissible :
  forall req, 1 <= spec_prefill_chunk_env req
    /\ spec_prefill_chunk_env req <= ring_slots - window + 1.
Proof.
  intro req. unfold spec_prefill_chunk_env, ring_slots, window, chunk, headroom.
  lia.
Qed.

(* ---------------------------------------------------------------------- *)
(* The budget stopped being a constant.                                    *)
(*                                                                         *)
(* budget_bytes above is 76 GiB because that is what admission.rs hardcoded *)
(* as DEFAULT_BUDGET_GIB. A deploy to a smaller card then died with CUDA    *)
(* out-of-memory at startup: the gate was proving "fits" about a machine    *)
(* it was not running on. admission.rs now derives the budget --            *)
(* NV_VRAM_BUDGET_GIB when set, else DEFAULT_BUDGET_FRACTION = 0.8 times    *)
(* the device total from cuMemGetInfo, else FALLBACK_BUDGET_GIB = 16 with   *)
(* no device at all. Modelling the RULE turns the machine size into a       *)
(* hypothesis instead of an assumption, so the theorems below say which     *)
(* cards can serve this model rather than asserting that every card can.    *)

Definition device_budget (total : Z) : Z := 4 * total / 5.
Definition fallback_budget_bytes : Z := 16 * 1024 * 1024 * 1024.

Definition gate_accepts_on (total weights kvm : Z) : Prop :=
  weights + worst_total true true kvm + hd512_scratch_bytes <= device_budget total.

Definition full_window_required_bytes : Z :=
  weights_measured_bytes + worst_total true true kv_max_default
  + hd512_scratch_bytes.

Theorem full_window_required_val : full_window_required_bytes = 67871391744.
Proof. vm_compute. reflexivity. Qed.

(* 0.8 * 95 GiB = 76 GiB exactly: the old constant was this machine's card,
   written down as if it were a property of the model. *)
Theorem fraction_rule_reproduces_the_old_constant :
  device_budget (95 * 1024 * 1024 * 1024) = budget_bytes.
Proof. vm_compute. reflexivity. Qed.

Definition min_total_for_full_window : Z := 84839239680.

Theorem full_window_accepts_at_min_total :
  gate_accepts_on min_total_for_full_window weights_measured_bytes kv_max_default.
Proof. unfold gate_accepts_on. vm_compute. intro H; discriminate H. Qed.

(* Tight: one byte less and the floor division drops the budget under the
   requirement, so this is the exact card size, not a round number. *)
Theorem min_total_for_full_window_is_tight :
  ~ gate_accepts_on (min_total_for_full_window - 1) weights_measured_bytes
      kv_max_default.
Proof. unfold gate_accepts_on. vm_compute. intro H; exact (H eq_refl). Qed.

Definition workstation_total_bytes : Z := 97887 * 1024 * 1024.

Theorem workstation_accepts_full_window :
  gate_accepts_on workstation_total_bytes weights_measured_bytes kv_max_default.
Proof. unfold gate_accepts_on. vm_compute. intro H; discriminate H. Qed.

(* A card just under the threshold refuses -- and refusing is the correct
   behaviour, which is the whole point of deriving the budget: the old
   constant would have accepted here and then OOMed. *)
Theorem a_79_gib_card_refuses_full_window :
  ~ gate_accepts_on (79 * 1024 * 1024 * 1024) weights_measured_bytes
      kv_max_default.
Proof. unfold gate_accepts_on. vm_compute. intro H; exact (H eq_refl). Qed.

(* The no-device fallback cannot serve this model at ANY context length:
   the weights alone are 37.3 GiB against a 16 GiB budget. So on the
   fallback path the gate is not conservative, it is a refusal. *)
Theorem fallback_budget_refuses_every_ctx :
  forall kvm, 0 <= kvm ->
    ~ (weights_measured_bytes + worst_total true true kvm + hd512_scratch_bytes
       <= fallback_budget_bytes).
Proof.
  intros kvm H0.
  assert (H00 : 0 <= 0) by lia.
  pose proof (worst_total_mono 0 kvm H00 H0) as Hmono.
  assert (Hz : 0 <= worst_total true true 0) by (vm_compute; intro C; discriminate C).
  unfold weights_measured_bytes, fallback_budget_bytes, hd512_scratch_bytes in *.
  lia.
Qed.

(* Acceptance is monotone in the card: a bigger card never refuses what a
   smaller one accepted. Stated because the floor division makes that a
   claim rather than an obvious fact. *)
Theorem bigger_card_never_refuses :
  forall t1 t2 w kvm,
    0 <= t1 -> t1 <= t2 ->
    gate_accepts_on t1 w kvm -> gate_accepts_on t2 w kvm.
Proof.
  intros t1 t2 w kvm H0 H12 H.
  unfold gate_accepts_on, device_budget in *.
  assert (4 * t1 <= 4 * t2) by lia.
  pose proof (Z.div_le_mono (4 * t1) (4 * t2) 5 ltac:(lia) ltac:(lia)).
  lia.
Qed.

(* ---------------------------------------------------------------------- *)
(* What dropping the duplicate V on the full layers would buy.             *)
(*                                                                         *)
(* config.json sets attention_k_eq_v = true, so the 10 full-attention      *)
(* layers have NO V projection: gemma4.rs feeds one raw tensor to both     *)
(* norms (v = k.clone()), k_norm carries a single scalar (measured         *)
(* constant across all 512 entries on every full layer), v_norm carries    *)
(* 1, and only k is RoPE'd. So v = unrope(k_cached) / scalar, and the      *)
(* cache is storing a value it could recompute. The per-token full-layer   *)
(* term below is fp8 K + fp8 V + both scale sets; keeping only K halves    *)
(* the data and the scales alike. Preconditions pinned by                  *)
(* nv-models/tests/gemma4_kv_share.rs. NOT YET IMPLEMENTED -- this states  *)
(* the size of the prize so the kernel work can be judged against it. *)

Definition full_layers : Z := 10.
Definition full_bytes_per_token_kv : Z := 4128.
Definition full_bytes_per_token_k_only : Z := 2064.

Theorem dropping_v_halves_the_full_layer_term :
  2 * full_bytes_per_token_k_only = full_bytes_per_token_kv.
Proof. vm_compute. reflexivity. Qed.

Definition v_savings (kvm : Z) : Z :=
  full_layers * kvm * (full_bytes_per_token_kv - full_bytes_per_token_k_only).

Theorem v_savings_at_full_window : v_savings kv_max_default = 5410652160.
Proof. vm_compute. reflexivity. Qed.

Theorem v_savings_is_a_fifth_of_the_worst_case :
  5 * v_savings kv_max_default <= worst_total true true kv_max_default.
Proof. vm_compute. intro H; discriminate H. Qed.

Definition required_without_duplicate_v : Z :=
  full_window_required_bytes - v_savings kv_max_default.

Theorem required_without_duplicate_v_val :
  required_without_duplicate_v = 62460739584.
Proof. vm_compute. reflexivity. Qed.

(* 79.01 GiB -> 72.71 GiB: the change moves which cards can serve the full
   window, not just how much headroom the ones that can have. *)
Definition min_total_without_duplicate_v : Z := 78075924480.

Theorem min_total_without_duplicate_v_suffices :
  required_without_duplicate_v <= device_budget min_total_without_duplicate_v.
Proof. unfold required_without_duplicate_v. vm_compute. intro H; discriminate H. Qed.

Theorem min_total_without_duplicate_v_is_tight :
  ~ (required_without_duplicate_v <= device_budget (min_total_without_duplicate_v - 1)).
Proof. unfold required_without_duplicate_v. vm_compute. intro H; exact (H eq_refl). Qed.

Theorem dropping_v_lowers_the_minimum_card :
  min_total_without_duplicate_v < min_total_for_full_window.
Proof. vm_compute. reflexivity. Qed.
