From Stdlib Require Import QArith Lia Arith.

Fixpoint qsum (n : nat) (f : nat -> Q) : Q :=
  match n with
  | O => 0%Q
  | S m => (qsum m f + f m)%Q
  end.

Lemma qsum_ext : forall n f g,
  (forall t, (t < n)%nat -> (f t == g t)%Q) -> (qsum n f == qsum n g)%Q.
Proof.
  induction n; intros f g H; simpl.
  - reflexivity.
  - rewrite (IHn f g) by (intros; apply H; lia).
    rewrite (H n) by lia.
    reflexivity.
Qed.

Lemma qsum_plus : forall n f g,
  (qsum n (fun t => f t + g t) == qsum n f + qsum n g)%Q.
Proof.
  induction n; intros f g; simpl; [ring |].
  rewrite IHn; ring.
Qed.

Lemma qsum_scale : forall n c f,
  (qsum n (fun t => c * f t) == c * qsum n f)%Q.
Proof.
  induction n; intros c f; simpl; [ring |].
  rewrite IHn; ring.
Qed.

Lemma qsum_zero : forall n, (qsum n (fun _ => 0) == 0)%Q.
Proof.
  induction n; simpl; [reflexivity |].
  rewrite IHn; ring.
Qed.

Lemma qsum_swap : forall m n (f : nat -> nat -> Q),
  (qsum m (fun i => qsum n (fun j => f i j))
   == qsum n (fun j => qsum m (fun i => f i j)))%Q.
Proof.
  induction m; intros n f; simpl.
  - rewrite qsum_zero; reflexivity.
  - rewrite (IHm n f).
    rewrite <- qsum_plus.
    apply qsum_ext; intros t _; reflexivity.
Qed.

Definition bf16linear_fwd (Kd : nat) (x w : nat -> nat -> Q) (i j : nat) : Q :=
  qsum Kd (fun k => x i k * w j k).

Definition bf16linear_bwd_dx (Nd : nat) (dy w : nat -> nat -> Q)
    (i k : nat) : Q :=
  qsum Nd (fun j => dy i j * w j k).

Theorem bwd_dx_is_adjoint_of_fwd : forall Md Nd Kd x w dy,
  (qsum Md (fun i => qsum Nd (fun j => bf16linear_fwd Kd x w i j * dy i j))
   == qsum Md (fun i =>
        qsum Kd (fun k => x i k * bf16linear_bwd_dx Nd dy w i k)))%Q.
Proof.
  intros Md Nd Kd x w dy.
  apply qsum_ext; intros i _.
  unfold bf16linear_fwd, bf16linear_bwd_dx.
  assert (Step1 :
    (qsum Nd (fun j => qsum Kd (fun k => x i k * w j k) * dy i j)
     == qsum Nd (fun j => qsum Kd (fun k => dy i j * (x i k * w j k))))%Q).
  { apply qsum_ext; intros j _.
    rewrite Qmult_comm.
    rewrite <- qsum_scale.
    reflexivity. }
  rewrite Step1.
  etransitivity;
    [apply (qsum_swap Nd Kd (fun j k => dy i j * (x i k * w j k))) |].
  apply qsum_ext; intros k _.
  assert (Step2 :
    (qsum Nd (fun j => dy i j * (x i k * w j k))
     == qsum Nd (fun j => x i k * (dy i j * w j k)))%Q).
  { apply qsum_ext; intros j _; ring. }
  rewrite Step2.
  apply qsum_scale.
Qed.

Definition fwd_untransposed (Kd : nat) (x w : nat -> nat -> Q)
    (i j : nat) : Q :=
  qsum Kd (fun k => x i k * w k j).

Definition wit_x (i k : nat) : Q := if Nat.eqb k 0 then 1%Q else 0%Q.
Definition wit_w (a b : nat) : Q :=
  if andb (Nat.eqb a 0) (Nat.eqb b 1) then 1%Q else 0%Q.
Definition wit_dy (i j : nat) : Q := if Nat.eqb j 1 then 1%Q else 0%Q.

Theorem untransposed_fwd_breaks_the_adjoint :
  ~ (qsum 1 (fun i =>
       qsum 2 (fun j => fwd_untransposed 2 wit_x wit_w i j * wit_dy i j))
     == qsum 1 (fun i =>
          qsum 2 (fun k =>
            wit_x i k * bf16linear_bwd_dx 2 wit_dy wit_w i k)))%Q.
Proof.
  intro H; vm_compute in H; discriminate H.
Qed.

Definition masked_sum (mask : nat -> bool) (l : nat -> Q) (n : nat) : Q :=
  qsum n (fun t => if mask t then l t else 0%Q).

Definition masked_count (mask : nat -> bool) (n : nat) : Q :=
  qsum n (fun t => if mask t then 1%Q else 0%Q).

Theorem prompt_positions_cannot_affect_loss : forall mask l1 l2 n,
  (forall t, (t < n)%nat -> mask t = true -> (l1 t == l2 t)%Q) ->
  (masked_sum mask l1 n == masked_sum mask l2 n)%Q.
Proof.
  intros mask l1 l2 n H; unfold masked_sum.
  apply qsum_ext; intros t Ht.
  destruct (mask t) eqn:E; [apply H; auto | reflexivity].
Qed.

Theorem fully_masked_batch_has_zero_sum_and_count : forall mask l n,
  (forall t, (t < n)%nat -> mask t = false) ->
  (masked_sum mask l n == 0)%Q /\ (masked_count mask n == 0)%Q.
Proof.
  intros mask l n H; unfold masked_sum, masked_count; split.
  - etransitivity; [| apply qsum_zero].
    apply qsum_ext; intros t Ht; rewrite (H t Ht); reflexivity.
  - etransitivity; [| apply qsum_zero].
    apply qsum_ext; intros t Ht; rewrite (H t Ht); reflexivity.
Qed.

Definition completion_mask (p t : nat) : bool := Nat.leb p t.

Theorem completion_boundary : forall p,
  completion_mask p p = true
  /\ (forall t, (t < p)%nat -> completion_mask p t = false).
Proof.
  intros p; unfold completion_mask; split.
  - apply Nat.leb_le; lia.
  - intros t Ht; apply Nat.leb_gt; lia.
Qed.

Theorem completion_mask_sums_exactly_the_completion : forall p c l,
  (masked_sum (completion_mask p) l (p + c)
   == qsum c (fun t => l ((p + t)%nat)))%Q.
Proof.
  intros p c l; induction c.
  - rewrite Nat.add_0_r; simpl.
    apply (proj1 (fully_masked_batch_has_zero_sum_and_count
                    (completion_mask p) l p
                    (fun t Ht => proj2 (completion_boundary p) t Ht))).
  - rewrite Nat.add_succ_r; unfold masked_sum in *; simpl.
    rewrite IHc.
    replace (completion_mask p (p + c)) with true
      by (symmetry; unfold completion_mask; apply Nat.leb_le; lia).
    reflexivity.
Qed.

Theorem completion_count_is_completion_length : forall p c,
  (masked_count (completion_mask p) (p + c)
   == qsum c (fun _ => 1%Q))%Q.
Proof.
  intros p c; induction c.
  - rewrite Nat.add_0_r; simpl.
    apply (proj2 (fully_masked_batch_has_zero_sum_and_count
                    (completion_mask p) (fun _ => 0%Q) p
                    (fun t Ht => proj2 (completion_boundary p) t Ht))).
  - rewrite Nat.add_succ_r; unfold masked_count in *; simpl.
    rewrite IHc.
    replace (completion_mask p (p + c)) with true
      by (symmetry; unfold completion_mask; apply Nat.leb_le; lia).
    reflexivity.
Qed.
