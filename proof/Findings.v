From Stdlib Require Import Arith Bool List Lia.
Import ListNotations.
From Evict Require Import Exclusion Evict Shipped.

Lemma shipped_faithful : faithful shipped.
Proof. intros [d|] me; reflexivity. Qed.

Theorem shipped_exclusive : forall es s t,
  run shipped es = Some s -> holds s t = true -> cap s (negb t) = false.
Proof. exact (exclusive shipped shipped_faithful eq_refl). Qed.

Theorem shipped_never_both_blocked : forall es s,
  run shipped es = Some s -> ~ (blocked shipped s true /\ blocked shipped s false).
Proof. exact (never_both_blocked shipped shipped_faithful eq_refl). Qed.

Theorem shipped_waits_on_the_owner : forall es s t,
  run shipped es = Some s -> blocked shipped s t -> owner s = Some (negb t).
Proof. exact (waits_on_the_owner shipped shipped_faithful eq_refl). Qed.

Example own_capture_nests :
  run shipped [CallExcl true; GetExcl true; CallExcl true; GetExcl true; CallCap true; GetCap true;
               CallExcl true; GetExcl true; ExitExcl true; EndCap true; ExitExcl true; ExitExcl true] = Some init.
Proof. reflexivity. Qed.

Example blocking_is_not_vacuous :
  run shipped [CallExcl true; GetExcl true; CallCap false] = Some (St Free WCap false false (Some (true, 1))) /\
  blocked shipped (St Free WCap false false (Some (true, 1))) false /\
  owner (St Free WCap false false (Some (true, 1))) = Some true.
Proof. unfold blocked. repeat split; try reflexivity; discriminate. Qed.

Definition old : policy := Policy (other_holds shipped) false.

Example old_design_deadlocks :
  (exists s, run old [CallCap true; GetCap true; GetLock true; CallCap false; GetCap false;
                      CallExcl true; GetExcl true] = Some s /\ blocked old s true /\ blocked old s false) /\
  (exists s, run old [CallCap true; GetCap true; GetLock true; CallExcl false; GetExcl false;
                      CallExcl true] = Some s /\ blocked old s true /\ blocked old s false) /\
  run old [CallExcl true; GetExcl true; Drained true; ExitExcl true; CallCap false; GetCap false;
           GetLock false] = Some (St Free Free false true None).
Proof. unfold blocked. repeat split; try (eexists; repeat split); try reflexivity; discriminate. Qed.

Lemma shipped_evict_faithful : faithful_e shipped_evict.
Proof. repeat split; intros; reflexivity. Qed.

Theorem shipped_no_release_under_request : forall idle global now base adapter,
  evict shipped_evict false idle global now base adapter = ([], false).
Proof. intros. apply no_release_under_request. reflexivity. Qed.

Theorem shipped_idle_releases_only_idle_slots : forall global now base adapter i,
  0 < global -> In i (fst (evict shipped_evict true true global now base adapter)) ->
  exists h, In h (held_slots shipped_evict base adapter) /\ sid h = i /\
            Nat.min (idle_secs shipped_evict) global <= now - put_at h.
Proof. intros. exact (idle_releases_only_idle_slots _ _ _ _ _ _ shipped_evict_faithful H H0). Qed.

Theorem shipped_pressure_releases_every_slot : forall global now base adapter h,
  In h (held_slots shipped_evict base adapter) -> In (sid h) (fst (evict shipped_evict true false global now base adapter)).
Proof. intros. exact (pressure_releases_every_slot _ _ _ _ _ _ shipped_evict_faithful H). Qed.

Theorem shipped_adapter_released_only_when_idle : forall global now base t,
  0 < global -> (forall x, In x base -> sid x <> adapter_slot shipped_evict) ->
  snd (evict shipped_evict true true global now base (Some t)) = true ->
  Nat.min (idle_secs shipped_evict) global <= now - t.
Proof.
  intros. exact (adapter_released_only_when_idle _ _ _ _ _ shipped_evict_faithful eq_refl eq_refl H H0 H1).
Qed.

Example idle_edges :
  fst (evict shipped_evict true true 600 1300 [Slot 0 1000] None) = [0] /\
  fst (evict shipped_evict true true 600 1299 [Slot 0 1000] None) = [] /\
  fst (evict shipped_evict true true 60 1060 [Slot 0 1000] None) = [0] /\
  fst (evict shipped_evict true true 60 1059 [Slot 0 1000] None) = [].
Proof. repeat split. Qed.

Example adapter_slot_edges :
  evict shipped_evict true true 600 1300 [Slot 0 1000] (Some 1299) = ([0], false) /\
  evict shipped_evict true true 600 1300 [Slot 0 1000] (Some 1000) = ([0; 2], true) /\
  evict shipped_evict true false 600 1300 [] (Some 1299) = ([2], true) /\
  evict shipped_evict true true 600 1300 [Slot 0 1000] None = ([0], false).
Proof. repeat split. Qed.

Definition old_evict : epolicy :=
  EPolicy (idle_secs shipped_evict) (idle_limit shipped_evict) (idle_ok shipped_evict)
    (needs_permit shipped_evict) (adapter_slot shipped_evict) false true.

Example old_release_drops_fresh_adapter :
  evict old_evict true true 600 1300 [Slot 0 1000] (Some 1299) = ([0], true) /\
  evict old_evict true false 600 1300 [] (Some 0) = ([], false).
Proof. split; reflexivity. Qed.
