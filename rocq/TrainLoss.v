From Stdlib Require Import Reals Ranalysis1 List PeanoNat Lia Lra.
Import ListNotations.
Open Scope R_scope.

Fixpoint sumexp (z : list R) : R :=
  match z with
  | [] => 0
  | a :: t => exp a + sumexp t
  end.

Definition ce (z : list R) (y : nat) : R := ln (sumexp z) - nth y z 0.

Record pos : Type := MkPos { keep : bool; logits : list R; target : nat }.

Definition pos_ce (p : pos) : R := ce (logits p) (target p).

Fixpoint masked_sum (e : list pos) : R :=
  match e with
  | [] => 0
  | p :: r => (if keep p then pos_ce p else 0) + masked_sum r
  end.

Fixpoint masked_count (e : list pos) : nat :=
  match e with
  | [] => O
  | p :: r => ((if keep p then 1 else 0) + masked_count r)%nat
  end.

Definition example_loss (e : list pos) : R := masked_sum e / INR (masked_count e).

Fixpoint batch_loss (b : list (list pos)) : R :=
  match b with
  | [] => 0
  | e :: r => example_loss e + batch_loss r
  end.

Fixpoint ce_sum (e : list pos) : R :=
  match e with
  | [] => 0
  | p :: r => pos_ce p + ce_sum r
  end.

Definition plain_example_mean (e : list pos) : R := ce_sum e / INR (length e).

Fixpoint plain_batch_loss (b : list (list pos)) : R :=
  match b with
  | [] => 0
  | e :: r => plain_example_mean e + plain_batch_loss r
  end.

Definition all_kept (e : list pos) : Prop := forall p, In p e -> keep p = true.

Lemma all_kept_sum : forall e, all_kept e -> masked_sum e = ce_sum e.
Proof.
  induction e as [|p r IH]; intros Ha; simpl; [reflexivity|].
  rewrite Ha by (left; reflexivity).
  rewrite IH; [reflexivity|].
  intros q Hq; apply Ha; right; exact Hq.
Qed.

Lemma all_kept_count : forall e, all_kept e -> masked_count e = length e.
Proof.
  induction e as [|p r IH]; intros Ha; simpl; [reflexivity|].
  rewrite Ha by (left; reflexivity).
  rewrite IH; [reflexivity|].
  intros q Hq; apply Ha; right; exact Hq.
Qed.

Theorem full_mask_metric_contract :
  forall b, (forall e, In e b -> all_kept e) -> batch_loss b = plain_batch_loss b.
Proof.
  induction b as [|e r IH]; intros Hb; simpl; [reflexivity|].
  unfold example_loss, plain_example_mean.
  rewrite all_kept_sum by (apply Hb; left; reflexivity).
  rewrite all_kept_count by (apply Hb; left; reflexivity).
  rewrite IH; [reflexivity|].
  intros e' He'; apply Hb; right; exact He'.
Qed.

Fixpoint upd {A : Type} (l : list A) (i : nat) (a : A) : list A :=
  match l, i with
  | [], _ => []
  | _ :: t, O => a :: t
  | h :: t, S j => h :: upd t j a
  end.

Definition set_logit (p : pos) (j : nat) (v : R) : pos :=
  MkPos (keep p) (upd (logits p) j v) (target p).

Fixpoint upd_example (e : list pos) (i j : nat) (v : R) : list pos :=
  match e, i with
  | [], _ => []
  | p :: r, O => set_logit p j v :: r
  | p :: r, S m => p :: upd_example r m j v
  end.

Fixpoint upd_batch (b : list (list pos)) (k i j : nat) (v : R) : list (list pos) :=
  match b, k with
  | [], _ => []
  | e :: r, O => upd_example e i j v :: r
  | e :: r, S m => e :: upd_batch r m i j v
  end.

Lemma masked_count_upd_example :
  forall e i j v, masked_count (upd_example e i j v) = masked_count e.
Proof.
  induction e as [|p r IH]; intros [|m] j v; simpl; try reflexivity.
  rewrite IH; reflexivity.
Qed.

Lemma masked_sum_upd_unkept :
  forall e i j v p0,
    nth_error e i = Some p0 -> keep p0 = false ->
    masked_sum (upd_example e i j v) = masked_sum e.
Proof.
  induction e as [|p r IH]; intros [|m] j v p0 Hnth Hk; simpl in *;
    try discriminate.
  - injection Hnth as <-.
    unfold set_logit; simpl; rewrite Hk; reflexivity.
  - rewrite (IH m j v p0 Hnth Hk); reflexivity.
Qed.

Theorem prompt_position_example_invariant :
  forall e i j v p0,
    nth_error e i = Some p0 -> keep p0 = false ->
    example_loss (upd_example e i j v) = example_loss e.
Proof.
  intros; unfold example_loss.
  rewrite (masked_sum_upd_unkept e i j v p0) by assumption.
  rewrite masked_count_upd_example; reflexivity.
Qed.

Theorem prompt_position_batch_invariant :
  forall b k i j v e0 p0,
    nth_error b k = Some e0 -> nth_error e0 i = Some p0 -> keep p0 = false ->
    batch_loss (upd_batch b k i j v) = batch_loss b.
Proof.
  induction b as [|e r IH]; intros [|m] i j v e0 p0 Hb He Hk; simpl in *;
    try discriminate.
  - injection Hb as ->.
    rewrite (prompt_position_example_invariant e0 i j v p0) by assumption.
    reflexivity.
  - rewrite (IH m i j v e0 p0 Hb He Hk); reflexivity.
Qed.

Lemma const_deriv_zero :
  forall (f : R -> R) (c x : R),
    (forall t, f t = c) -> derivable_pt_lim f x 0.
Proof.
  intros f c x Hf.
  apply derivable_pt_lim_ext with (f := fct_cte c).
  - intros z; unfold fct_cte; rewrite Hf; reflexivity.
  - apply derivable_pt_lim_const.
Qed.

Theorem prompt_logit_grad_zero :
  forall b k i j v0 e0 p0,
    nth_error b k = Some e0 -> nth_error e0 i = Some p0 -> keep p0 = false ->
    derivable_pt_lim (fun v => batch_loss (upd_batch b k i j v)) v0 0.
Proof.
  intros b k i j v0 e0 p0 Hb He Hk.
  apply const_deriv_zero with (c := batch_loss b).
  intros t; eapply prompt_position_batch_invariant; eauto.
Qed.

Definition completion_mask (prompt_len t : nat) : bool := (prompt_len <=? t)%nat.

Lemma completion_mask_prompt_positions_false :
  forall prompt_len t, (t < prompt_len)%nat -> completion_mask prompt_len t = false.
Proof. intros; unfold completion_mask; apply Nat.leb_gt; lia. Qed.

Lemma completion_mask_completion_positions_true :
  forall prompt_len t, (prompt_len <= t)%nat -> completion_mask prompt_len t = true.
Proof. intros; unfold completion_mask; apply Nat.leb_le; lia. Qed.

Definition wrong_example_loss (e : list pos) : R := masked_sum e / INR (length e).

Lemma wrong_denominator_disagrees :
  forall e,
    (0 < masked_count e)%nat ->
    (masked_count e < length e)%nat ->
    masked_sum e <> 0 ->
    example_loss e <> wrong_example_loss e.
Proof.
  intros e Hc Hl Hs Heq.
  unfold example_loss, wrong_example_loss, Rdiv in Heq.
  assert (Hc0 : 0 < INR (masked_count e)) by (apply lt_0_INR; lia).
  assert (Hl0 : 0 < INR (length e)) by (apply lt_0_INR; lia).
  assert (Hcl : INR (masked_count e) < INR (length e)) by (apply lt_INR; lia).
  assert (Hinv : / INR (length e) < / INR (masked_count e)).
  { apply Rinv_lt_contravar; [apply Rmult_lt_0_compat|]; assumption. }
  destruct (Rdichotomy _ _ Hs) as [Hneg | Hpos].
  - assert (- masked_sum e * / INR (length e) < - masked_sum e * / INR (masked_count e))
      by (apply Rmult_lt_compat_l; lra).
    lra.
  - assert (masked_sum e * / INR (length e) < masked_sum e * / INR (masked_count e))
      by (apply Rmult_lt_compat_l; lra).
    lra.
Qed.

Definition sep_row : list R := [0; 0].
Definition sep_example : list pos := [MkPos false sep_row O; MkPos true sep_row O].

Lemma sep_row_ce : ce sep_row O = ln 2.
Proof.
  unfold ce, sep_row; simpl; rewrite exp_0.
  replace (1 + (1 + 0)) with 2 by lra.
  lra.
Qed.

Lemma ln2_pos : 0 < ln 2.
Proof. rewrite <- ln_1; apply ln_increasing; lra. Qed.

Theorem masked_numerator_over_all_positions_separates :
  example_loss sep_example = ln 2 /\
  wrong_example_loss sep_example = ln 2 / 2 /\
  example_loss sep_example <> wrong_example_loss sep_example.
Proof.
  assert (H2 := ln2_pos).
  assert (Hms : masked_sum sep_example = ln 2).
  { simpl; unfold pos_ce, ce, sep_row; simpl; rewrite exp_0.
    replace (1 + (1 + 0)) with 2 by lra; lra. }
  assert (Hmc : masked_count sep_example = 1%nat) by reflexivity.
  assert (Hlen : length sep_example = 2%nat) by reflexivity.
  unfold example_loss, wrong_example_loss.
  rewrite Hms, Hmc, Hlen; simpl INR.
  split; [lra|].
  split; [lra|].
  intro Heq; lra.
Qed.

Section WindowedTraining.

Variables X Q : Type.

Fixpoint runL (ls : list (Q -> X -> X)) (ps : list Q) (x : X) : X :=
  match ls, ps with
  | f :: ls', q :: ps' => runL ls' ps' (f q x)
  | _, _ => x
  end.

Lemma runL_app :
  forall lsA psA lsB psB x,
    length lsA = length psA ->
    runL (lsA ++ lsB) (psA ++ psB) x = runL lsB psB (runL lsA psA x).
Proof.
  induction lsA as [|f lsA IH]; intros [|q psA] lsB psB x Hlen;
    simpl in *; try discriminate; try reflexivity.
  apply IH; injection Hlen; auto.
Qed.

Variable L : X -> R.

Theorem frozen_prefix_cache_sound :
  forall lsP psP lsW psW x,
    length lsP = length psP ->
    L (runL (lsP ++ lsW) (psP ++ psW) x) = L (runL lsW psW (runL lsP psP x)).
Proof. intros; rewrite runL_app; auto. Qed.

Theorem eval_rows_share_frozen_prefix :
  forall lsP psP lsW psW psW' x,
    length lsP = length psP ->
    L (runL (lsP ++ lsW) (psP ++ psW) x) = L (runL lsW psW (runL lsP psP x)) /\
    L (runL (lsP ++ lsW) (psP ++ psW') x) = L (runL lsW psW' (runL lsP psP x)).
Proof. intros; split; apply frozen_prefix_cache_sound; assumption. Qed.

Theorem window_grad_locality :
  forall lsP psP lsW x (path : R -> list Q) t0 g,
    length lsP = length psP ->
    (derivable_pt_lim (fun t => L (runL (lsP ++ lsW) (psP ++ path t) x)) t0 g
     <-> derivable_pt_lim (fun t => L (runL lsW (path t) (runL lsP psP x))) t0 g).
Proof.
  intros lsP psP lsW x path t0 g Hlen; split; apply derivable_pt_lim_ext;
    intros z.
  - apply frozen_prefix_cache_sound; assumption.
  - symmetry; apply frozen_prefix_cache_sound; assumption.
Qed.

End WindowedTraining.
