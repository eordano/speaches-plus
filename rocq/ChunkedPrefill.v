
From Stdlib Require Import ZArith Lia List.
Import ListNotations.
Open Scope Z_scope.

Definition verify_prefill_chunk : Z := 1024.
Definition vocab : Z := 262144.

Fixpoint pieces_fuel (fuel : nat) (p c : Z) : list Z :=
  match fuel with
  | O => []
  | S f => if p <=? 0 then []
           else let l := Z.min c p in l :: pieces_fuel f (p - l) c
  end.

Definition chunk_pieces (p c : Z) : list Z :=
  pieces_fuel (Z.to_nat p) p c.

Lemma pieces_fuel_sum :
  forall fuel p c, 1 <= c -> 0 <= p -> (Z.of_nat fuel >= p) ->
    fold_right Z.add 0 (pieces_fuel fuel p c) = p.
Proof.
  induction fuel as [|f IH]; intros p c Hc Hp Hfuel; simpl.
  - lia.
  - destruct (p <=? 0) eqn:E.
    + apply Z.leb_le in E. simpl. lia.
    + apply Z.leb_gt in E. simpl.
      rewrite IH; lia.
Qed.

Theorem chunk_pieces_cover :
  forall p c, 1 <= c -> 0 <= p ->
    fold_right Z.add 0 (chunk_pieces p c) = p.
Proof.
  intros. unfold chunk_pieces. apply pieces_fuel_sum; lia.
Qed.

Lemma pieces_fuel_bound :
  forall fuel p c x, 1 <= c -> In x (pieces_fuel fuel p c) ->
    1 <= x /\ x <= c.
Proof.
  induction fuel as [|f IH]; intros p c x Hc Hin; simpl in Hin.
  - contradiction.
  - destruct (p <=? 0) eqn:E; [contradiction|].
    apply Z.leb_gt in E.
    destruct Hin as [Heq | Hin].
    + subst x. lia.
    + eapply IH; eauto.
Qed.

Theorem chunk_pieces_bounded :
  forall p c x, 1 <= c -> In x (chunk_pieces p c) -> 1 <= x /\ x <= c.
Proof. intros p c x Hc Hin. eapply pieces_fuel_bound; eauto. Qed.

Definition transient (a b rows : Z) : Z := a * rows + b.

Theorem peak_transient_bounded_by_chunk :
  forall p c a b x,
    1 <= c -> 0 <= a -> In x (chunk_pieces p c) ->
    transient a b x <= transient a b c.
Proof.
  intros p c a b x Hc Ha Hin.
  destruct (chunk_pieces_bounded p c x Hc Hin) as [H1 H2].
  unfold transient. nia.
Qed.

Corollary peak_transient_production :
  forall p a b x,
    0 <= a -> In x (chunk_pieces p verify_prefill_chunk) ->
    transient a b x <= a * 1024 + b.
Proof.
  intros p a b x Ha Hin.
  pose proof (peak_transient_bounded_by_chunk p verify_prefill_chunk a b x
                ltac:(unfold verify_prefill_chunk; lia) Ha Hin) as H.
  unfold transient, verify_prefill_chunk in *. lia.
Qed.

Definition logit_rows_merged (last : bool) : Z := if last then 1 else 0.
Definition logit_transient_merged (last : bool) : Z :=
  logit_rows_merged last * vocab * 2.

Definition logit_transient_prefix (rows : Z) : Z := rows * vocab * 2.

Theorem logit_transient_merged_const :
  forall last, logit_transient_merged last <= 524288.
Proof. intro last; destruct last; vm_compute; intro H; discriminate H. Qed.

Lemma logit_transient_prefix_14500 :
  logit_transient_prefix 14500 = 7602176000.
Proof. vm_compute. reflexivity. Qed.

Lemma logit_transient_ratio_14500 :
  logit_transient_prefix 14500 = 14500 * logit_transient_merged true.
Proof. vm_compute. reflexivity. Qed.
