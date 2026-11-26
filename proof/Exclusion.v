From Stdlib Require Import Bool Arith Lia List.
Import ListNotations.

Inductive call := Free | WCap | WLock | WExcl | Drain.

Record policy := Policy { other_holds : option bool -> bool -> bool; unified : bool }.

Definition faithful (p : policy) : Prop :=
  forall o me, other_holds p o me = match o with Some d => negb (Bool.eqb d me) | None => false end.

Record st := St { ca : call; cb : call; capa : bool; capb : bool; ex : option (bool * nat) }.

Definition init : st := St Free Free false false None.

Definition cl s (t : bool) := if t then ca s else cb s.
Definition cap s (t : bool) := if t then capa s else capb s.
Definition set_cl s (t : bool) v := if t then St v (cb s) (capa s) (capb s) (ex s) else St (ca s) v (capa s) (capb s) (ex s).
Definition set_cap s (t : bool) v := if t then St (ca s) (cb s) v (capb s) (ex s) else St (ca s) (cb s) (capa s) v (ex s).
Definition set_ex s v := St (ca s) (cb s) (capa s) (capb s) v.

Definition excluder s := option_map fst (ex s).
Definition held s := match ex s with Some (_, n) => n | None => 0 end.
Definition excludes s (t : bool) := match ex s with Some (u, _) => Bool.eqb u t | None => false end.
Definition owner s :=
  if cap s true || excludes s true then Some true else if cap s false || excludes s false then Some false else None.
Definition counted s (t : bool) := cap s t || match cl s t with WLock => true | _ => false end.

Inductive ev := CallCap (t : bool) | GetCap (t : bool) | GetLock (t : bool) | EndCap (t : bool)
              | CallExcl (t : bool) | GetExcl (t : bool) | Drained (t : bool) | ExitExcl (t : bool).

Definition step (p : policy) (s : st) (e : ev) : option st :=
  match e with
  | CallCap t => match cl s t with Free => if cap s t then None else Some (set_cl s t WCap) | _ => None end
  | GetCap t =>
      match cl s t with
      | WCap =>
          if unified p then
            if other_holds p (owner s) t then None else Some (set_cap (set_cl s t Free) t true)
          else if other_holds p (excluder s) t then None else Some (set_cl s t WLock)
      | _ => None
      end
  | GetLock t =>
      match cl s t with
      | WLock => if unified p || cap s (negb t) then None else Some (set_cap (set_cl s t Free) t true)
      | _ => None
      end
  | EndCap t => match cl s t with Free => if cap s t then Some (set_cap s t false) else None | _ => None end
  | CallExcl t => match cl s t with Free => Some (set_cl s t WExcl) | _ => None end
  | GetExcl t =>
      match cl s t with
      | WExcl =>
          if unified p then
            if other_holds p (owner s) t then None else Some (set_ex (set_cl s t Free) (Some (t, S (held s))))
          else if other_holds p (excluder s) t then None else Some (set_ex (set_cl s t Drain) (Some (t, held s)))
      | _ => None
      end
  | Drained t =>
      match cl s t with
      | Drain => if unified p || counted s (negb t) then None
                 else Some (set_ex (set_cl s t Free) (Some (t, S (held s))))
      | _ => None
      end
  | ExitExcl t =>
      match cl s t, ex s with
      | Free, Some (u, S n) => if Bool.eqb u t then Some (set_ex s (match n with 0 => None | _ => Some (u, n) end)) else None
      | _, _ => None
      end
  end.

Definition run (p : policy) (es : list ev) : option st :=
  fold_left (fun o e => match o with Some s => step p s e | None => None end) es (Some init).

Definition completes s (t : bool) : ev :=
  match cl s t with WCap => GetCap t | WLock => GetLock t | WExcl => GetExcl t | Drain => Drained t | Free => EndCap t end.

Definition blocked p s (t : bool) : Prop := cl s t <> Free /\ step p s (completes s t) = None.

Definition holds s (t : bool) := match ex s with Some (u, S _) => Bool.eqb u t | _ => false end.

Definition one_owner (s : st) : Prop :=
  (cap s true = true -> cap s false = false) /\
  (forall t n, ex s = Some (t, n) -> cap s (negb t) = false) /\
  (forall t, cl s t <> WLock /\ cl s t <> Drain) /\
  (forall t, cl s t = WCap -> cap s t = false).

Ltac crush :=
  repeat match goal with
  | b : bool |- _ => destruct b
  | c : call |- _ => destruct c
  | o : option _ |- _ => destruct o
  | x : _ * _ |- _ => destruct x
  end; simpl in *;
  repeat match goal with
  | H : context[match ?n with 0 => _ | S _ => _ end] |- _ => is_var n; destruct n; simpl in *
  | |- context[match ?n with 0 => _ | S _ => _ end] => is_var n; destruct n; simpl in *
  end;
  try discriminate; try congruence; try lia.

Ltac solve_step :=
  unfold faithful, one_owner, blocked, completes, holds, owner, counted, excludes, excluder, held, cl, cap,
    set_cl, set_cap, set_ex in *;
  simpl in *;
  match goal with Hf : forall o me, other_holds _ o me = _ |- _ => rewrite ?Hf in * end;
  match goal with p : policy |- _ => set (u := unified p) in *; clearbody u end;
  crush;
  repeat match goal with H : Some _ = Some _ |- _ => injection H as <- end;
  simpl in *; crush;
  repeat split; intros;
  repeat match goal with H : Some _ = Some _ |- _ => injection H as <- <- end;
  crush;
  match goal with Hf : forall o me, other_holds _ o me = _ |- _ => rewrite ?Hf in * end;
  simpl in *; crush;
  repeat match goal with H : _ /\ _ |- _ => destruct H end;
  repeat match goal with H : forall (t : bool) (n : nat), @?P t n |- _ =>
    let H1 := fresh in let H2 := fresh in pose proof (H true) as H1; pose proof (H false) as H2; clear H end;
  repeat match goal with H : forall t : bool, @?P t |- _ =>
    let H1 := fresh in let H2 := fresh in pose proof (H true) as H1; pose proof (H false) as H2; clear H end;
  simpl in *;
  repeat match goal with H : _ /\ _ |- _ => destruct H end;
  repeat match goal with H : ?a = ?a -> _ |- _ => specialize (H eq_refl) end;
  repeat match goal with H : forall n : nat, Some _ = Some _ -> _ |- _ => specialize (H _ eq_refl) end;
  repeat match goal with H : _ /\ _ |- _ => destruct H end;
  try discriminate; try congruence; try lia; firstorder (try discriminate; try congruence; try lia).

Lemma step_one_owner : forall p s e s',
  faithful p -> unified p = true -> one_owner s -> step p s e = Some s' -> one_owner s'.
Proof.
  intros p [a b xa xb x] e s' Hf Hu I Hs.
  destruct e as [t|t|t|t|t|t|t|t]; unfold step in Hs; rewrite ?Hu in Hs; solve_step.
Qed.

Lemma reach {P : st -> Prop} : forall p, (forall s e s', P s -> step p s e = Some s' -> P s') -> P init ->
  forall es s, run p es = Some s -> P s.
Proof.
  intros p Hstep Hinit es. unfold run.
  assert (forall o, (forall s, o = Some s -> P s) ->
    forall s, fold_left (fun o e => match o with Some s => step p s e | None => None end) es o = Some s -> P s) as G.
  { induction es as [|e es IH]; simpl; intros o Ho s Hs; [now apply Ho|].
    refine (IH _ _ s Hs). intros s1 H1. destruct o as [s0|]; [|discriminate]. exact (Hstep s0 e s1 (Ho s0 eq_refl) H1). }
  intros s Hs. apply (G (Some init)); [|exact Hs]. intros s0 H0. now injection H0 as <-.
Qed.

Lemma one_owner_reachable : forall p, faithful p -> unified p = true ->
  forall es s, run p es = Some s -> one_owner s.
Proof.
  intros p Hf Hu. apply reach; [intros; eapply step_one_owner; eauto|].
  unfold one_owner, init, cap, cl; simpl; repeat split; intros; try destruct t; simpl in *; congruence.
Qed.

Theorem exclusive : forall p, faithful p -> unified p = true ->
  forall es s t, run p es = Some s -> holds s t = true -> cap s (negb t) = false.
Proof.
  intros p Hf Hu es s t Hr Ht. pose proof (one_owner_reachable p Hf Hu es s Hr) as [_ [I _]].
  unfold holds in Ht. destruct (ex s) as [[u [|n]]|] eqn:E; try discriminate.
  apply Bool.eqb_prop in Ht. subst. exact (I t _ eq_refl).
Qed.

Theorem never_both_blocked : forall p, faithful p -> unified p = true ->
  forall es s, run p es = Some s -> ~ (blocked p s true /\ blocked p s false).
Proof.
  intros p Hf Hu es [a b xa xb x] Hr [Ha Hb]. pose proof (one_owner_reachable p Hf Hu es _ Hr) as I.
  clear Hr. unfold blocked, completes in Ha, Hb. unfold step in Ha, Hb. rewrite ?Hu in Ha, Hb. solve_step.
Qed.

Theorem waits_on_the_owner : forall p, faithful p -> unified p = true ->
  forall es s t, run p es = Some s -> blocked p s t -> owner s = Some (negb t).
Proof.
  intros p Hf Hu es [a b xa xb x] t Hr Hb. pose proof (one_owner_reachable p Hf Hu es _ Hr) as I.
  clear Hr. unfold blocked, completes in Hb. unfold step in Hb. rewrite ?Hu in Hb. destruct t; solve_step.
Qed.
