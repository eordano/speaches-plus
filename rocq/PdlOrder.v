From Stdlib Require Import List Arith Lia.
Import ListNotations.

Definition buffer := nat.
Definition point := nat.

Record kernel := {
  k_id : nat;
  k_prolog : point;
  k_epilog : point;
  k_writes : buffer -> list point;
  k_reads : buffer -> list point
}.

Definition epilog_after_writes (k : kernel) (b : buffer) : Prop :=
  forall p, In p (k_writes k b) -> p < k_epilog k.

Definition prolog_before_reads (k : kernel) (b : buffer) : Prop :=
  forall p, In p (k_reads k b) -> k_prolog k < p.

Definition monotone (t : kernel -> point -> nat) : Prop :=
  forall k p q, p < q -> t k p < t k q.

Definition pdl_dependent (t : kernel -> point -> nat) (A B : kernel) : Prop :=
  t A (k_epilog A) <= t B (k_prolog B).

Definition serialized (t : kernel -> point -> nat) (A B : kernel) : Prop :=
  forall p, t A p < t B (k_prolog B).

Theorem pdl_writes_precede_reads :
  forall t A B b,
    monotone t ->
    epilog_after_writes A b ->
    prolog_before_reads B b ->
    pdl_dependent t A B ->
    forall pw pr,
      In pw (k_writes A b) ->
      In pr (k_reads B b) ->
      t A pw < t B pr.
Proof.
  intros t A B b Hmono Hepi Hpro Hdep pw pr Hw Hr.
  unfold pdl_dependent in Hdep.
  assert (t A pw < t A (k_epilog A)) as H1 by (apply Hmono; apply Hepi; exact Hw).
  assert (t B (k_prolog B) < t B pr) as H2 by (apply Hmono; apply Hpro; exact Hr).
  lia.
Qed.

Theorem unwired_neighbour_is_safe :
  forall t A B b,
    monotone t ->
    prolog_before_reads B b ->
    serialized t A B ->
    forall pw pr,
      In pw (k_writes A b) ->
      In pr (k_reads B b) ->
      t A pw < t B pr.
Proof.
  intros t A B b Hmono Hpro Hser pw pr Hw Hr.
  unfold serialized in Hser.
  assert (t B (k_prolog B) < t B pr) as H2 by (apply Hmono; apply Hpro; exact Hr).
  specialize (Hser pw). lia.
Qed.

Definition sched (k : kernel) (p : point) : nat := 2 * k_id k + p.

Definition kA : kernel :=
  {| k_id := 0; k_prolog := 0; k_epilog := 1;
     k_writes := fun b => if Nat.eqb b 0 then [5] else [];
     k_reads := fun _ => [] |}.

Definition kB : kernel :=
  {| k_id := 1; k_prolog := 0; k_epilog := 9;
     k_writes := fun _ => [];
     k_reads := fun b => if Nat.eqb b 0 then [1] else [] |}.

Lemma sched_monotone : monotone sched.
Proof. intros k p q H. unfold sched. lia. Qed.

Theorem epilog_hypothesis_is_necessary :
  monotone sched
  /\ prolog_before_reads kB 0
  /\ pdl_dependent sched kA kB
  /\ ~ epilog_after_writes kA 0
  /\ In 5 (k_writes kA 0)
  /\ In 1 (k_reads kB 0)
  /\ ~ (sched kA 5 < sched kB 1).
Proof.
  repeat split.
  - exact sched_monotone.
  - intros p Hp. simpl in Hp. destruct Hp as [Hp | []]. subst. simpl. lia.
  - unfold pdl_dependent, sched. simpl. lia.
  - intros Hcontra. specialize (Hcontra 5). simpl in Hcontra.
    assert (5 < 1) by (apply Hcontra; left; reflexivity). lia.
  - simpl. left. reflexivity.
  - simpl. left. reflexivity.
  - unfold sched. simpl. lia.
Qed.

Theorem multi_output_needs_all_writes :
  forall k b1 b2,
    epilog_after_writes k b1 ->
    epilog_after_writes k b2 ->
    forall p, In p (k_writes k b1) \/ In p (k_writes k b2) -> p < k_epilog k.
Proof.
  intros k b1 b2 H1 H2 p [Hp | Hp].
  - apply H1; exact Hp.
  - apply H2; exact Hp.
Qed.
