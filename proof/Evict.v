From Stdlib Require Import Arith Bool List Lia.
Import ListNotations.

Record epolicy := EPolicy {
  idle_secs : nat;
  idle_limit : nat -> option nat -> option nat;
  idle_ok : option nat -> nat -> bool;
  needs_permit : bool;
  adapter_slot : nat;
  adapter_held : bool;
  drops_adapter : bool
}.

Definition faithful_e (p : epolicy) : Prop :=
  (forall g own, idle_limit p g own =
     if 0 <? g then Some (match own with Some o => Nat.min o g | None => g end) else None) /\
  (forall l e, idle_ok p (Some l) e = (l <=? e)) /\ (forall e, idle_ok p None e = true).

Record slot := Slot { sid : nat; put_at : nat }.

Definition limit_for p (idle : bool) global := if idle then idle_limit p global (Some (idle_secs p)) else None.

Definition held_slots p (base : list slot) (adapter : option nat) : list slot :=
  base ++ (if adapter_held p then match adapter with Some t => [Slot (adapter_slot p) t] | None => [] end else []).

Definition selected p idle global now (held : list slot) : list nat :=
  map sid (filter (fun h => idle_ok p (limit_for p idle global) (now - put_at h)) held).

Definition evict p (permit_free idle : bool) (global now : nat) (base : list slot) (adapter : option nat)
  : list nat * bool :=
  if needs_permit p && negb permit_free then ([], false) else
  match selected p idle global now (held_slots p base adapter) with
  | [] => ([], false)
  | sel => (sel, match adapter with
                 | Some _ => drops_adapter p || existsb (Nat.eqb (adapter_slot p)) sel
                 | None => false
                 end)
  end.

Theorem no_release_under_request : forall p idle global now base adapter,
  needs_permit p = true -> evict p false idle global now base adapter = ([], false).
Proof. intros. unfold evict. rewrite H. reflexivity. Qed.

Lemma in_selected : forall p idle global now held i,
  In i (selected p idle global now held) <->
  exists h, In h held /\ sid h = i /\ idle_ok p (limit_for p idle global) (now - put_at h) = true.
Proof.
  intros. unfold selected. rewrite in_map_iff.
  split; intros [h [H1 H2]]; exists h; [rewrite filter_In in H2|]; firstorder.
  apply filter_In. tauto.
Qed.

Lemma released_selected : forall p idle global now base adapter i,
  In i (fst (evict p true idle global now base adapter)) -> In i (selected p idle global now (held_slots p base adapter)).
Proof.
  intros. unfold evict in H. rewrite andb_false_r in H.
  destruct (selected p idle global now (held_slots p base adapter)); simpl in *; [contradiction|exact H].
Qed.

Lemma idle_bound : forall p global now h, faithful_e p -> 0 < global ->
  idle_ok p (limit_for p true global) (now - put_at h) = true -> Nat.min (idle_secs p) global <= now - put_at h.
Proof.
  intros p global now h [Hl [Ho _]] Hg H. unfold limit_for in H. rewrite Hl in H.
  destruct (0 <? global) eqn:E; [|apply Nat.ltb_ge in E; lia].
  rewrite Ho in H. now apply Nat.leb_le in H.
Qed.

Theorem idle_releases_only_idle_slots : forall p global now base adapter i, faithful_e p ->
  0 < global -> In i (fst (evict p true true global now base adapter)) ->
  exists h, In h (held_slots p base adapter) /\ sid h = i /\ Nat.min (idle_secs p) global <= now - put_at h.
Proof.
  intros p global now base adapter i Hf Hg Hi.
  apply released_selected, in_selected in Hi as [h [H1 [H2 H3]]].
  exists h. repeat split; auto. eapply idle_bound; eauto.
Qed.

Theorem pressure_releases_every_slot : forall p global now base adapter h, faithful_e p ->
  In h (held_slots p base adapter) -> In (sid h) (fst (evict p true false global now base adapter)).
Proof.
  intros p global now base adapter h [_ [_ Hn]] Hh.
  assert (Hs : In (sid h) (selected p false global now (held_slots p base adapter)))
    by (apply in_selected; exists h; repeat split; auto; apply Hn).
  unfold evict. rewrite andb_false_r. destruct (selected p false global now (held_slots p base adapter));
  [contradiction|exact Hs].
Qed.

Theorem adapter_released_only_when_idle : forall p global now base t, faithful_e p ->
  drops_adapter p = false -> adapter_held p = true -> 0 < global ->
  (forall x, In x base -> sid x <> adapter_slot p) ->
  snd (evict p true true global now base (Some t)) = true -> Nat.min (idle_secs p) global <= now - t.
Proof.
  intros p global now base t Hf Hd Ha Hg Hb H.
  unfold evict in H. rewrite andb_false_r, Hd in H.
  destruct (selected p true global now (held_slots p base (Some t))) as [|n l] eqn:E; [discriminate|].
  cbn [snd orb] in H. apply existsb_exists in H as [i [Hi Heq]]. apply Nat.eqb_eq in Heq. subst i.
  rewrite <- E in Hi. apply in_selected in Hi as [h [Hh [Hs Hok]]].
  unfold held_slots in Hh. rewrite Ha in Hh. apply in_app_or in Hh as [Hh|[Hh|[]]].
  - exfalso. exact (Hb h Hh Hs).
  - subst h. pose proof (idle_bound p global now (Slot (adapter_slot p) t) Hf Hg Hok). simpl in *. lia.
Qed.
