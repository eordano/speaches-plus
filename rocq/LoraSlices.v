
From Stdlib Require Import Reals List Lia Lra.
From SpeachesPlus Require Import TrainBackward.
Import ListNotations.
Open Scope R_scope.

Definition slice_start (ws : list nat) (s : nat) : nat := list_sum (firstn s ws).

Definition slices_total (ws : list nat) : nat := list_sum ws.

Lemma slice_start_zero : forall ws, slice_start ws 0 = 0%nat.
Proof. reflexivity. Qed.

Lemma list_sum_cons : forall (x : nat) l, list_sum (x :: l) = (x + list_sum l)%nat.
Proof. reflexivity. Qed.

Lemma slice_start_succ : forall ws s,
  (s < length ws)%nat ->
  slice_start ws (S s) = (slice_start ws s + nth s ws 0)%nat.
Proof.
  induction ws as [|w tl IH]; intros s Hs; simpl in Hs; [lia|].
  destruct s as [|s].
  - unfold slice_start. simpl. lia.
  - unfold slice_start in *. rewrite !firstn_cons, !list_sum_cons.
    replace (nth (S s) (w :: tl) 0%nat) with (nth s tl 0%nat) by reflexivity.
    rewrite IH by lia. lia.
Qed.

Lemma slice_start_mono : forall ws s s',
  (s <= s')%nat -> (s' <= length ws)%nat ->
  (slice_start ws s <= slice_start ws s')%nat.
Proof.
  intros ws s s' Hle Hlen.
  induction s' as [|s' IH].
  - replace s with 0%nat by lia. lia.
  - destruct (Nat.eq_dec s (S s')) as [->|Hne]; [lia|].
    rewrite slice_start_succ by lia.
    assert (slice_start ws s <= slice_start ws s')%nat by (apply IH; lia).
    lia.
Qed.

Lemma slice_disjoint_lt : forall ws t s,
  (t < s)%nat -> (s <= length ws)%nat ->
  (slice_start ws t + nth t ws 0 <= slice_start ws s)%nat.
Proof.
  intros ws t s Ht Hs.
  rewrite <- slice_start_succ by lia.
  apply slice_start_mono; lia.
Qed.

Definition pad_contrib (st w : nat) (d : nat -> R) (c : nat) : R :=
  if ((st <=? c)%nat && (c <? st + w)%nat)%bool then d (c - st)%nat else 0.

Definition fused_padded_sum (ws : list nat) (D : nat -> nat -> R) (c : nat) : R :=
  fsum (length ws) (fun s => pad_contrib (slice_start ws s) (nth s ws 0%nat) (D s) c).

Theorem sliced_write_equals_monolithic : forall ws D s j,
  (s < length ws)%nat -> (j < nth s ws 0)%nat ->
  fused_padded_sum ws D (slice_start ws s + j) = D s j.
Proof.
  intros ws D s j Hs Hj.
  unfold fused_padded_sum.
  rewrite (fsum_single (length ws) s).
  - unfold pad_contrib.
    replace ((slice_start ws s <=? slice_start ws s + j)%nat) with true
      by (symmetry; apply Nat.leb_le; lia).
    replace ((slice_start ws s + j <? slice_start ws s + nth s ws 0)%nat) with true
      by (symmetry; apply Nat.ltb_lt; lia).
    cbn. f_equal. lia.
  - assumption.
  - intros t Ht Hts. unfold pad_contrib.
    destruct (Nat.lt_ge_cases t s) as [Hlt|Hge].
    + pose proof (slice_disjoint_lt ws t s Hlt ltac:(lia)) as Hdis.
      replace ((slice_start ws s + j <? slice_start ws t + nth t ws 0)%nat) with false
        by (symmetry; apply Nat.ltb_ge; lia).
      now rewrite Bool.andb_false_r.
    + assert (Hgt : (s < t)%nat) by lia.
      pose proof (slice_disjoint_lt ws s t Hgt ltac:(lia)) as Hdis.
      replace ((slice_start ws t <=? slice_start ws s + j)%nat) with false
        by (symmetry; apply Nat.leb_gt; lia).
      reflexivity.
Qed.

Theorem every_output_column_lands_in_one_slice : forall ws c,
  (c < slices_total ws)%nat ->
  exists s j, (s < length ws)%nat /\ (j < nth s ws 0)%nat
    /\ c = (slice_start ws s + j)%nat.
Proof.
  induction ws as [|w tl IH]; intros c Hc;
    unfold slices_total in *; simpl in Hc; [lia|].
  destruct (Nat.lt_ge_cases c w) as [Hlt|Hge].
  - exists 0%nat, c. split; [simpl; lia|]. split; [simpl; lia|].
    unfold slice_start. reflexivity.
  - destruct (IH (c - w)%nat ltac:(lia)) as (s & j & Hs & Hj & Hcj).
    exists (S s), j. split; [simpl; lia|]. split; [simpl; exact Hj|].
    unfold slice_start in *. rewrite firstn_cons, list_sum_cons. lia.
Qed.

Theorem slice_decomposition_unique : forall ws s j s' j',
  (s < length ws)%nat -> (j < nth s ws 0)%nat ->
  (s' < length ws)%nat -> (j' < nth s' ws 0)%nat ->
  (slice_start ws s + j = slice_start ws s' + j')%nat ->
  s = s' /\ j = j'.
Proof.
  intros ws s j s' j' Hs Hj Hs' Hj' Heq.
  destruct (Nat.lt_trichotomy s s') as [Hlt|[->|Hgt]].
  - pose proof (slice_disjoint_lt ws s s' Hlt ltac:(lia)). lia.
  - split; [reflexivity| lia].
  - pose proof (slice_disjoint_lt ws s' s Hgt ltac:(lia)). lia.
Qed.

Corollary windowed_write_equals_monolithic : forall ws D s j win_off c,
  (s < length ws)%nat -> (j < nth s ws 0)%nat ->
  (slice_start ws s + j = win_off + c)%nat ->
  fused_padded_sum ws D (win_off + c) = D s j.
Proof.
  intros ws D s j win_off c Hs Hj Heq.
  rewrite <- Heq. apply sliced_write_equals_monolithic; assumption.
Qed.

Theorem qkv_slice_offsets : forall q_dim kv_dim,
  slice_start [q_dim; kv_dim; kv_dim] 0 = 0%nat
  /\ slice_start [q_dim; kv_dim; kv_dim] 1 = q_dim
  /\ slice_start [q_dim; kv_dim; kv_dim] 2 = (q_dim + kv_dim)%nat
  /\ slices_total [q_dim; kv_dim; kv_dim] = (q_dim + kv_dim + kv_dim)%nat.
Proof.
  intros. unfold slice_start, slices_total. simpl. repeat split; lia.
Qed.

Theorem gate_up_slice_offsets : forall inter,
  slice_start [inter; inter] 0 = 0%nat
  /\ slice_start [inter; inter] 1 = inter
  /\ slices_total [inter; inter] = (2 * inter)%nat.
Proof.
  intros. unfold slice_start, slices_total. simpl. repeat split; lia.
Qed.

Definition fused_uniform_stride
  (ws : list nat) (wmax : nat) (D : nat -> nat -> R) (c : nat) : R :=
  fsum (length ws) (fun s => pad_contrib (s * wmax)%nat (nth s ws 0%nat) (D s) c).

Theorem uniform_stride_offsets_are_not_the_cumsum :
  exists ws wmax (D : nat -> nat -> R) c,
    (c < slices_total ws)%nat
    /\ fused_padded_sum ws D c <> fused_uniform_stride ws wmax D c.
Proof.
  exists [1; 2]%nat, 2%nat, (fun s _ => INR (S s)), 1%nat.
  split; [cbn; lia|].
  unfold fused_padded_sum, fused_uniform_stride, pad_contrib, slice_start.
  cbn. lra.
Qed.

Definition gate_cols (nh hd : nat) : nat := (nh * (2 * hd))%nat.

Definition value_lane (hd c : nat) : Prop := (c mod (2 * hd) < hd)%nat.

Definition head_of (hd c : nat) : nat := (c / (2 * hd))%nat.

Definition value_slot (hd c : nat) : nat :=
  (head_of hd c * hd + c mod (2 * hd))%nat.

Definition gate_slot (hd c : nat) : nat :=
  (head_of hd c * hd + (c mod (2 * hd) - hd))%nat.

Theorem value_lane_split : forall nh hd h i,
  (0 < hd)%nat -> (h < nh)%nat -> (i < hd)%nat ->
  value_lane hd (idx (2 * hd) h i)
  /\ value_slot hd (idx (2 * hd) h i) = idx hd h i.
Proof.
  intros nh hd h i Hhd Hh Hi.
  unfold value_lane, value_slot, head_of.
  rewrite idx_div, idx_mod by lia.
  split; [lia| unfold idx; lia].
Qed.

Theorem gate_lane_split : forall nh hd h i,
  (0 < hd)%nat -> (h < nh)%nat -> (i < hd)%nat ->
  ~ value_lane hd (idx (2 * hd) h (hd + i))
  /\ gate_slot hd (idx (2 * hd) h (hd + i)) = idx hd h i.
Proof.
  intros nh hd h i Hhd Hh Hi.
  unfold value_lane, gate_slot, head_of.
  rewrite idx_div, idx_mod by lia.
  split; [lia| unfold idx; lia].
Qed.

Theorem every_column_is_value_or_gate : forall nh hd c,
  (0 < hd)%nat -> (c < gate_cols nh hd)%nat ->
  exists h i, (h < nh)%nat /\ (i < hd)%nat /\
    ((c = idx (2 * hd) h i /\ value_lane hd c)
     \/ (c = idx (2 * hd) h (hd + i) /\ ~ value_lane hd c)).
Proof.
  intros nh hd c Hhd Hc.
  assert (H2 : (2 * hd <> 0)%nat) by lia.
  pose proof (Nat.mod_upper_bound c (2 * hd) H2) as Hrem.
  pose proof (Nat.div_mod_eq c (2 * hd)) as Hdecomp.
  assert (Hh : (c / (2 * hd) < nh)%nat).
  { apply Nat.Div0.div_lt_upper_bound. unfold gate_cols in Hc. lia. }
  destruct (Nat.lt_ge_cases (c mod (2 * hd)) hd) as [Hv|Hg].
  - exists (c / (2 * hd))%nat, (c mod (2 * hd))%nat.
    split; [exact Hh|]. split; [exact Hv|].
    left. split; [unfold idx; lia| unfold value_lane; exact Hv].
  - exists (c / (2 * hd))%nat, (c mod (2 * hd) - hd)%nat.
    split; [exact Hh|]. split; [lia|].
    right. split; [unfold idx; lia| unfold value_lane; lia].
Qed.

Theorem value_only_adapter_leaves_gate_lanes_unadapted :
  forall (a bmat : Buf) (s : R) (nh hd k r : nat) (x : Buf) i c,
  (0 < hd)%nat -> (c < gate_cols nh hd)%nat -> ~ value_lane hd c ->
  (forall c' t, (c' < gate_cols nh hd)%nat -> ~ value_lane hd c' ->
     (t < r)%nat -> bmat (idx r c' t) = 0) ->
  lora_delta a bmat s (gate_cols nh hd) k r x (idx (gate_cols nh hd) i c) = 0.
Proof.
  intros a bmat s nh hd k r x i c Hhd Hc Hgate Hsupp.
  unfold lora_delta, scaleb.
  rewrite gemm_bt_at by lia.
  rewrite fsum_zero; [ring|].
  intros t Ht. rewrite Hsupp by assumption. ring.
Qed.

Theorem adapted_output_agrees_with_base_on_gate_lanes :
  forall (a bmat y : Buf) (s : R) (nh hd k r : nat) (x : Buf) i c,
  (0 < hd)%nat -> (c < gate_cols nh hd)%nat -> ~ value_lane hd c ->
  (forall c' t, (c' < gate_cols nh hd)%nat -> ~ value_lane hd c' ->
     (t < r)%nat -> bmat (idx r c' t) = 0) ->
  addb y (lora_delta a bmat s (gate_cols nh hd) k r x)
    (idx (gate_cols nh hd) i c)
  = y (idx (gate_cols nh hd) i c).
Proof.
  intros. unfold addb.
  rewrite value_only_adapter_leaves_gate_lanes_unadapted by assumption.
  ring.
Qed.

Theorem corner_padded_half_width_b_adapts_the_wrong_lanes :
  forall nh hd, (2 <= nh)%nat -> (0 < hd)%nat ->
  exists c c',
    (c < nh * hd)%nat /\ (c < gate_cols nh hd)%nat /\ ~ value_lane hd c
    /\ (nh * hd <= c')%nat /\ (c' < gate_cols nh hd)%nat /\ value_lane hd c'.
Proof.
  intros nh hd Hnh Hhd.
  exists hd, ((nh - 1) * (2 * hd))%nat.
  unfold gate_cols, value_lane.
  rewrite (Nat.mod_small hd (2 * hd)) by lia.
  rewrite Nat.Div0.mod_mul.
  repeat split; try lia; try nia.
Qed.
