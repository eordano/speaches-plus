
From Stdlib Require Import Reals Lra Psatz.
Open Scope R_scope.

Record forward_exec : Type := mkForward {
  hbm_bytes : R;
  wall_time : R
}.

Definition respects_bandwidth (BW : R) (f : forward_exec) : Prop :=
  hbm_bytes f <= wall_time f * BW.

Definition weight_traffic_lb (L2 W : R) (f : forward_exec) : Prop :=
  W - L2 <= hbm_bytes f.

Theorem verify_time_lower_bound :
  forall BW L2 W f,
    respects_bandwidth BW f ->
    weight_traffic_lb L2 W f ->
    W - L2 <= wall_time f * BW.
Proof.
  unfold respects_bandwidth, weight_traffic_lb; intros; lra.
Qed.

Corollary verify_time_lower_bound_div :
  forall BW L2 W f,
    0 < BW ->
    respects_bandwidth BW f ->
    weight_traffic_lb L2 W f ->
    (W - L2) / BW <= wall_time f.
Proof.
  intros BW L2 W f HBW H1 H2.
  pose proof (verify_time_lower_bound BW L2 W f H1 H2).
  apply Rmult_le_reg_r with (r := BW); [exact HBW|].
  unfold Rdiv; rewrite Rmult_assoc, Rinv_l; lra.
Qed.

Corollary verify_time_positive :
  forall BW L2 W f,
    0 < BW -> L2 < W ->
    respects_bandwidth BW f ->
    weight_traffic_lb L2 W f ->
    0 < wall_time f.
Proof.
  intros BW L2 W f HBW HWL H1 H2.
  pose proof (verify_time_lower_bound BW L2 W f H1 H2).
  nra.
Qed.

(* ---------------------------------------------------------------------- *)
(* BATCHED speed of light.                                                  *)
(*                                                                          *)
(* speed_of_light below assumes r * T = tpr: ONE stream emitting tpr tokens *)
(* per round of wall T. That hypothesis silently fixes batch size 1, so the *)
(* ceiling it yields is a SINGLE-STREAM ceiling and says nothing about an   *)
(* engine serving B sequences in one forward -- the case the batch engine   *)
(* actually runs. Naming it a global "token rate ceiling" was the review's  *)
(* headline unsoundness.                                                    *)
(*                                                                          *)
(* The honest generalization splits traffic by how it scales with B:        *)
(* weights are read ONCE per forward regardless of B (they amortize), while *)
(* per-sequence KV is read once PER SEQUENCE (it does not). So a batched    *)
(* forward's compulsory traffic is at least (Wwt - L2) + B * Wkv, and the   *)
(* aggregate emission is B * tpr per round.                                 *)

Definition batched_traffic_lb (L2 Wwt Wkv B : R) (f : forward_exec) : Prop :=
  (Wwt - L2) + B * Wkv <= hbm_bytes f.

Theorem speed_of_light_batched :
  forall BW L2 Wwt Wkv B (f : forward_exec) (T tpr r : R),
    0 < BW -> L2 < Wwt -> 0 <= Wkv -> 1 <= B ->
    respects_bandwidth BW f ->
    batched_traffic_lb L2 Wwt Wkv B f ->
    wall_time f <= T ->
    0 <= tpr ->
    r * T = B * tpr ->
    r * ((Wwt - L2) + B * Wkv) <= B * tpr * BW.
Proof.
  intros BW L2 Wwt Wkv B f T tpr r HBW HWL HKV HB H1 H2 HT Htpr Hr.
  unfold respects_bandwidth, batched_traffic_lb in *.
  assert (Hpos : 0 < (Wwt - L2) + B * Wkv) by nra.
  assert (Htf : 0 < wall_time f) by nra.
  assert (HTpos : 0 < T) by lra.
  assert (Hrnn : 0 <= r) by nra.
  assert (Hstep1 : r * ((Wwt - L2) + B * Wkv) <= r * (wall_time f * BW)) by nra.
  assert (HrBW : 0 <= r * BW) by nra.
  assert (Hstep2 : r * (wall_time f * BW) <= r * (T * BW)) by nra.
  nra.
Qed.

(* B = 1 recovers the single-stream statement exactly, so the old ceiling is
   a corollary of the new one rather than a competing claim: nothing that
   depended on it is invalidated, it is only revealed as the B=1 instance. *)
Corollary speed_of_light_batched_at_one :
  forall BW L2 Wwt Wkv (f : forward_exec) (T tpr r : R),
    0 < BW -> L2 < Wwt -> 0 <= Wkv ->
    respects_bandwidth BW f ->
    batched_traffic_lb L2 Wwt Wkv 1 f ->
    wall_time f <= T -> 0 <= tpr -> r * T = tpr ->
    r * ((Wwt - L2) + Wkv) <= tpr * BW.
Proof.
  intros BW L2 Wwt Wkv f T tpr r HBW HWL HKV H1 H2 HT Htpr Hr.
  assert (Hr' : r * T = 1 * tpr) by lra.
  pose proof (speed_of_light_batched BW L2 Wwt Wkv 1 f T tpr r
                HBW HWL HKV (Rle_refl 1) H1 H2 HT Htpr Hr') as H.
  lra.
Qed.

(* The batched ceiling is BOUNDED: amortizing weights over more sequences
   cannot buy unbounded throughput, because per-sequence KV traffic grows
   with B. As B -> infinity the rate saturates at tpr*BW/Wkv; this states
   the bound that holds for every B. *)
Theorem batched_ceiling_saturates :
  forall BW L2 Wwt Wkv B (f : forward_exec) (T tpr r : R),
    0 < BW -> L2 < Wwt -> 0 < Wkv -> 1 <= B ->
    respects_bandwidth BW f ->
    batched_traffic_lb L2 Wwt Wkv B f ->
    wall_time f <= T -> 0 <= tpr -> r * T = B * tpr ->
    r * Wkv <= tpr * BW.
Proof.
  intros BW L2 Wwt Wkv B f T tpr r HBW HWL HKV HB H1 H2 HT Htpr Hr.
  pose proof (speed_of_light_batched BW L2 Wwt Wkv B f T tpr r
                HBW HWL (Rlt_le _ _ HKV) HB H1 H2 HT Htpr Hr) as Hsol.
  assert (Hrnn : 0 <= r).
  { unfold respects_bandwidth, batched_traffic_lb in *.
    assert (0 < wall_time f) by nra. assert (0 < T) by lra. nra. }
  assert (HB0 : 0 < B) by lra.
  assert (Hdrop : r * (B * Wkv) <= r * ((Wwt - L2) + B * Wkv)) by nra.
  nra.
Qed.

Theorem speed_of_light :
  forall BW L2 W (f : forward_exec) (T tpr r : R),
    0 < BW -> L2 < W ->
    respects_bandwidth BW f ->
    weight_traffic_lb L2 W f ->
    wall_time f <= T ->
    0 <= tpr ->
    r * T = tpr ->
    r * (W - L2) <= tpr * BW.
Proof.
  intros BW L2 W f T tpr r HBW HWL H1 H2 HT Htpr Hr.
  pose proof (verify_time_lower_bound BW L2 W f H1 H2) as Hlb.
  pose proof (verify_time_positive BW L2 W f HBW HWL H1 H2) as Htpos.
  assert (HTpos : 0 < T) by lra.
  assert (Hrnn : 0 <= r) by nra.

  assert (Hstep1 : r * (W - L2) <= r * (wall_time f * BW)) by nra.
  assert (HrBW : 0 <= r * BW) by nra.
  assert (Hstep2 : r * (wall_time f * BW) <= r * (T * BW)) by nra.
  nra.
Qed.

Module Measured.

Definition BW : R := 1790000000000.
Definition L2 : R := 268435456.
Definition W  : R := 18300000000.
Definition TPR : R := 2193 / 1000.

Lemma BW_pos : 0 < BW.
Proof. unfold BW; lra. Qed.

Lemma W_gt_L2 : L2 < W.
Proof. unfold L2, W; lra. Qed.

Theorem verify_floor_10ms :
  forall f,
    respects_bandwidth BW f ->
    weight_traffic_lb L2 W f ->
    1 / 100 <= wall_time f.
Proof.
  intros f H1 H2.
  pose proof (verify_time_lower_bound BW L2 W f H1 H2) as H.
  unfold BW, L2, W in H; lra.
Qed.

Theorem token_rate_ceiling_218 :
  forall f T tpr r,
    respects_bandwidth BW f ->
    weight_traffic_lb L2 W f ->
    wall_time f <= T ->
    0 <= tpr -> tpr <= TPR ->
    r * T = tpr ->
    r <= 218.
Proof.
  intros f T tpr r H1 H2 HT Htpr HtprUB Hr.
  pose proof (speed_of_light BW L2 W f T tpr r BW_pos W_gt_L2 H1 H2 HT Htpr Hr)
    as Hsol.
  unfold BW, L2, W, TPR in *; lra.
Qed.

Theorem token_rate_ceiling_oracle_K4 :
  forall f T tpr r,
    respects_bandwidth BW f ->
    weight_traffic_lb L2 W f ->
    wall_time f <= T ->
    0 <= tpr -> tpr <= 5 ->
    r * T = tpr ->
    r <= 497.
Proof.
  intros f T tpr r H1 H2 HT Htpr HtprUB Hr.
  pose proof (speed_of_light BW L2 W f T tpr r BW_pos W_gt_L2 H1 H2 HT Htpr Hr)
    as Hsol.
  unfold BW, L2, W in *; lra.
Qed.

End Measured.
