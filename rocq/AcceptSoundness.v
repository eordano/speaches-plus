
From Stdlib Require Import List Arith PeanoNat Lia.
Import ListNotations.

Section AcceptSoundness.

Variable next : list nat -> nat.

Fixpoint greedy (n : nat) (ctx : list nat) : list nat :=
  match n with
  | O => []
  | S m => next ctx :: greedy m (ctx ++ [next ctx])
  end.

Lemma greedy_app :
  forall a b ctx,
    greedy (a + b) ctx = greedy a ctx ++ greedy b (ctx ++ greedy a ctx).
Proof.
  induction a as [|a IH]; intros b ctx; cbn.
  - now rewrite app_nil_r.
  - rewrite IH, <- app_assoc; reflexivity.
Qed.

Lemma greedy_firstn :
  forall n m ctx, firstn m (greedy n ctx) = greedy (Nat.min m n) ctx.
Proof.
  induction n as [|n IH]; intros [|m] ctx; cbn; auto.
  now rewrite IH.
Qed.

Fixpoint round_scan (cur : list nat) (drafts : list nat)
  : list nat * list nat :=
  match drafts with
  | [] => ([], cur)
  | d :: rest =>
      if Nat.eqb d (next cur)
      then let '(em, cf) := round_scan (cur ++ [d]) rest in (d :: em, cf)
      else ([], cur)
  end.

Definition round (ctx : list nat) (bonus : nat) (drafts : list nat)
  : list nat * list nat * nat :=
  let '(em, cf) := round_scan (ctx ++ [bonus]) drafts in
  (bonus :: em, cf, next cf).

Lemma round_scan_greedy :
  forall drafts cur em cf,
    round_scan cur drafts = (em, cf) ->
    cf = cur ++ em /\ em = greedy (length em) cur.
Proof.
  induction drafts as [|d rest IH]; intros cur em cf H; cbn in H.
  - inversion H; subst; split; [now rewrite app_nil_r | reflexivity].
  - destruct (Nat.eqb d (next cur)) eqn:Heqb; rewrite ?Heqb in H.
    + apply Nat.eqb_eq in Heqb.
      destruct (round_scan (cur ++ [d]) rest) as [em' cf'] eqn:E.
      inversion H; subst em cf; clear H.
      destruct (IH _ _ _ E) as [Hc He].
      split.
      * rewrite Hc, <- app_assoc; reflexivity.
      * cbn [length greedy].
        rewrite <- Heqb; f_equal; exact He.
    + inversion H; subst; split; [now rewrite app_nil_r | reflexivity].
Qed.

Lemma round_greedy :
  forall ctx bonus drafts em cf b',
    bonus = next ctx ->
    round ctx bonus drafts = (em, cf, b') ->
    em = greedy (length em) ctx /\ cf = ctx ++ em /\ b' = next cf.
Proof.
  unfold round; intros ctx bonus drafts em cf b' Hb H.
  destruct (round_scan (ctx ++ [bonus]) drafts) as [em0 cf0] eqn:E.
  inversion H; subst em cf b'; clear H.
  destruct (round_scan_greedy _ _ _ _ E) as [Hc He].
  split; [|split].
  - cbn [length greedy].
    rewrite Hb at 1; f_equal.
    rewrite <- Hb; exact He.
  - rewrite Hc, <- app_assoc; reflexivity.
  - reflexivity.
Qed.

Lemma round_emits :
  forall ctx bonus drafts em cf b',
    round ctx bonus drafts = (em, cf, b') -> (1 <= length em)%nat.
Proof.
  unfold round; intros ctx bonus drafts em cf b' H.
  destruct (round_scan (ctx ++ [bonus]) drafts) as [em0 cf0].
  inversion H; subst; cbn; lia.
Qed.

Fixpoint spec_loop (rounds : nat)
    (draft_fn : list nat -> nat -> list nat)
    (ctx : list nat) (bonus : nat) : list nat :=
  match rounds with
  | O => []
  | S r =>
      let '(em, cf, b') := round ctx bonus (draft_fn ctx bonus) in
      em ++ spec_loop r draft_fn cf b'
  end.

Theorem spec_loop_emits_greedy :
  forall rounds draft_fn ctx bonus,
    bonus = next ctx ->
    spec_loop rounds draft_fn ctx bonus
      = greedy (length (spec_loop rounds draft_fn ctx bonus)) ctx.
Proof.
  induction rounds as [|r IH]; intros draft_fn ctx bonus Hb; cbn [spec_loop].
  - reflexivity.
  - destruct (round ctx bonus (draft_fn ctx bonus)) as [[em cf] b'] eqn:R.
    destruct (round_greedy _ _ _ _ _ _ Hb R) as (He & Hc & Hb').
    rewrite length_app, greedy_app.
    f_equal.
    + exact He.
    + rewrite <- He, <- Hc.
      apply IH; exact Hb'.
Qed.

Corollary drafter_irrelevant :
  forall rounds df1 df2 ctx bonus,
    bonus = next ctx ->
    length (spec_loop rounds df1 ctx bonus)
      = length (spec_loop rounds df2 ctx bonus) ->
    spec_loop rounds df1 ctx bonus = spec_loop rounds df2 ctx bonus.
Proof.
  intros rounds df1 df2 ctx bonus Hb Hlen.
  rewrite (spec_loop_emits_greedy rounds df1 ctx bonus Hb),
          (spec_loop_emits_greedy rounds df2 ctx bonus Hb), Hlen.
  reflexivity.
Qed.

Corollary spec_truncated_emits_greedy :
  forall rounds draft_fn ctx bonus m,
    bonus = next ctx ->
    firstn m (spec_loop rounds draft_fn ctx bonus)
      = greedy (Nat.min m (length (spec_loop rounds draft_fn ctx bonus))) ctx.
Proof.
  intros rounds draft_fn ctx bonus m Hb.
  rewrite (spec_loop_emits_greedy rounds draft_fn ctx bonus Hb) at 1.
  apply greedy_firstn.
Qed.

End AcceptSoundness.
