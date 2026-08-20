
From Stdlib Require Import ZArith Lia.
Open Scope Z_scope.

Definition attends (qpos kpos window : Z) : Prop :=
  kpos <= qpos /\ (window <= 0 \/ qpos - kpos < window).

Definition win_start (qpos window : Z) : Z :=
  Z.max 0 (qpos - (window - 1)).

Theorem committed_range_exact :
  forall window qpos nc p,
    0 < window ->
    nc <= qpos ->
    0 <= p < nc ->
    (win_start qpos window <= p <-> attends qpos p window).
Proof.
  unfold win_start, attends; intros; lia.
Qed.

Theorem committed_full_all :
  forall window qpos nc p,
    window <= 0 ->
    nc <= qpos ->
    0 <= p < nc ->
    attends qpos p window.
Proof.
  unfold attends; intros; lia.
Qed.

Definition kernel_tree_kept (masked : bool) (qpos kpos window : Z) : Prop :=
  masked = true /\ ~ (0 < window /\ qpos - kpos >= window).

Theorem tree_kept_iff_attends :
  forall masked qpos kpos window,
    (masked = true -> kpos <= qpos) ->
    (kernel_tree_kept masked qpos kpos window
       <-> masked = true /\ attends qpos kpos window).
Proof.
  unfold kernel_tree_kept, attends; intros masked qpos kpos window Hcausal.
  split; intros [Hm Hrest]; split; auto.
  - specialize (Hcausal Hm); lia.
  - lia.
Qed.

Theorem attended_keys_in_window :
  forall qpos kpos window,
    0 < window ->
    attends qpos kpos window ->
    qpos - window < kpos <= qpos.
Proof.
  unfold attends; intros; lia.
Qed.

Theorem cap_retains_attended :
  forall qpos kpos window cap,
    0 < window ->
    window <= cap ->
    attends qpos kpos window ->
    qpos - cap < kpos <= qpos.
Proof.
  unfold attends; intros; lia.
Qed.

Inductive layer_kind : Set := Sliding | Full.

Definition tree_layer_window (sliding_window : Z) (k : layer_kind) : Z :=
  match k with
  | Sliding => sliding_window
  | Full => 0
  end.

Theorem full_layer_attends_causal :
  forall sliding_window qpos kpos,
    attends qpos kpos (tree_layer_window sliding_window Full) <-> kpos <= qpos.
Proof.
  cbn; unfold attends; intros; lia.
Qed.

Theorem sliding_layer_window_id :
  forall w, tree_layer_window w Sliding = w.
Proof. reflexivity. Qed.
