From Stdlib Require Import Reals List Permutation Lra Lia.
Import ListNotations.
Open Scope R_scope.

Notation pairv := (R * R)%type.

Definition rot (t : R) (x : pairv) : pairv :=
  (fst x * cos t - snd x * sin t, fst x * sin t + snd x * cos t).

Definition pmul (x y : pairv) : R := fst x * fst y + snd x * snd y.

Lemma rot_0 : forall x, rot 0 x = x.
Proof.
  intros [a b]; unfold rot; simpl; rewrite cos_0, sin_0.
  f_equal; ring.
Qed.

Lemma rot_compose : forall s t x, rot s (rot t x) = rot (s + t) x.
Proof.
  intros s t [a b]; unfold rot; simpl; rewrite cos_plus, sin_plus.
  f_equal; ring.
Qed.

Lemma rot_pmul : forall s t x y,
  pmul (rot s x) (rot t y) = pmul (rot (s - t) x) y.
Proof.
  intros s t [a b] [c d]; unfold pmul, rot; simpl.
  rewrite cos_minus, sin_minus; ring.
Qed.

Lemma rot_orthogonal : forall t x y, pmul (rot t x) (rot t y) = pmul x y.
Proof.
  intros t x y; rewrite rot_pmul.
  replace (t - t) with 0 by ring; rewrite rot_0; reflexivity.
Qed.

Lemma rot_norm : forall t x, pmul (rot t x) (rot t x) = pmul x x.
Proof. intros; apply rot_orthogonal. Qed.

Fixpoint rope (fs : list R) (m : R) (X : list pairv) : list pairv :=
  match X with
  | [] => []
  | x :: xs =>
      match fs with
      | [] => x :: rope [] m xs
      | f :: fs' => rot (m * f) x :: rope fs' m xs
      end
  end.

Fixpoint pdot (X Y : list pairv) : R :=
  match X, Y with
  | x :: xs, y :: ys => pmul x y + pdot xs ys
  | _, _ => 0
  end.

Fixpoint dot (u v : list R) : R :=
  match u, v with
  | a :: us, b :: vs => a * b + dot us vs
  | _, _ => 0
  end.

Lemma rope_length : forall X fs m, length (rope fs m X) = length X.
Proof.
  induction X as [|x xs IH]; intros [|f fs] m; simpl; auto.
Qed.

Lemma rope_0 : forall X fs, rope fs 0 X = X.
Proof.
  induction X as [|x xs IH]; intros [|f fs]; simpl; try reflexivity.
  - rewrite (IH []); reflexivity.
  - rewrite Rmult_0_l, rot_0, (IH fs); reflexivity.
Qed.

Lemma rope_compose : forall X fs a b,
  rope fs a (rope fs b X) = rope fs (a + b) X.
Proof.
  induction X as [|x xs IH]; intros [|f fs] a b; simpl; try reflexivity.
  - rewrite (IH [] a b); reflexivity.
  - rewrite rot_compose.
    replace (a * f + b * f) with ((a + b) * f) by ring.
    rewrite (IH fs a b); reflexivity.
Qed.

Theorem rope_relative : forall X Y fs a b,
  pdot (rope fs a X) (rope fs b Y) = pdot (rope fs (a - b) X) Y.
Proof.
  induction X as [|x xs IH]; intros Y fs a b.
  - simpl; destruct fs; reflexivity.
  - destruct Y as [|y ys].
    + destruct fs; simpl; reflexivity.
    + destruct fs as [|f fs]; simpl.
      * rewrite (IH ys [] a b); reflexivity.
      * rewrite rot_pmul.
        replace (a * f - b * f) with ((a - b) * f) by ring.
        rewrite (IH ys fs a b); reflexivity.
Qed.

Theorem rope_orthogonal : forall X Y fs m,
  pdot (rope fs m X) (rope fs m Y) = pdot X Y.
Proof.
  intros X Y fs m; rewrite rope_relative.
  replace (m - m) with 0 by ring; rewrite rope_0; reflexivity.
Qed.

Corollary rope_norm : forall X fs m, pdot (rope fs m X) (rope fs m X) = pdot X X.
Proof. intros; apply rope_orthogonal. Qed.

Lemma rope_zeros : forall X n m, rope (repeat 0 n) m X = X.
Proof.
  induction X as [|x xs IH]; intros [|n] m; simpl; try reflexivity.
  - f_equal; apply (IH 0%nat m).
  - rewrite Rmult_0_r, rot_0; f_equal; apply (IH n m).
Qed.

Lemma rope_app : forall X1 fs1 X2 fs2 m,
  length fs1 = length X1 ->
  rope (fs1 ++ fs2) m (X1 ++ X2) = rope fs1 m X1 ++ rope fs2 m X2.
Proof.
  induction X1 as [|x xs IH]; intros [|f fs1] X2 fs2 m Hlen; simpl in *;
    try discriminate; try reflexivity.
  rewrite (IH fs1 X2 fs2 m) by lia; reflexivity.
Qed.

Lemma pdot_app : forall X1 Y1 X2 Y2,
  length X1 = length Y1 ->
  pdot (X1 ++ X2) (Y1 ++ Y2) = pdot X1 Y1 + pdot X2 Y2.
Proof.
  induction X1 as [|x xs IH]; intros [|y ys] X2 Y2 Hlen; simpl in *;
    try discriminate.
  - ring.
  - rewrite (IH ys X2 Y2) by lia; ring.
Qed.

Definition padded (active : list R) (h : nat) : list R :=
  active ++ repeat 0 (h - length active).

Theorem partial_tail_identity : forall A XA XB m h,
  length A = length XA ->
  (length XA + length XB)%nat = h ->
  rope (padded A h) m (XA ++ XB) = rope A m XA ++ XB.
Proof.
  intros A XA XB m h HA Hh; unfold padded.
  rewrite rope_app by exact HA.
  replace (h - length A)%nat with (length XB) by lia.
  rewrite rope_zeros; reflexivity.
Qed.

Theorem partial_relative_split : forall A XA XB YA YB m n h,
  length A = length XA ->
  length A = length YA ->
  (length XA + length XB)%nat = h ->
  (length YA + length YB)%nat = h ->
  pdot (rope (padded A h) m (XA ++ XB)) (rope (padded A h) n (YA ++ YB))
    = pdot (rope A (m - n) XA) YA + pdot XB YB.
Proof.
  intros A XA XB YA YB m n h HXA HYA HhX HhY.
  rewrite (partial_tail_identity A XA XB m h HXA HhX).
  rewrite (partial_tail_identity A YA YB n h HYA HhY).
  rewrite pdot_app by (rewrite !rope_length; lia).
  rewrite rope_relative; reflexivity.
Qed.

Theorem partial_tail_position_independent : forall A XA XB YA YB m n m' n' h,
  length A = length XA ->
  length A = length YA ->
  (length XA + length XB)%nat = h ->
  (length YA + length YB)%nat = h ->
  m - n = m' - n' ->
  pdot (rope (padded A h) m (XA ++ XB)) (rope (padded A h) n (YA ++ YB))
    = pdot (rope (padded A h) m' (XA ++ XB)) (rope (padded A h) n' (YA ++ YB)).
Proof.
  intros A XA XB YA YB m n m' n' h HXA HYA HhX HhY Hd.
  rewrite (partial_relative_split A XA XB YA YB m n h HXA HYA HhX HhY).
  rewrite (partial_relative_split A XA XB YA YB m' n' h HXA HYA HhX HhY).
  rewrite Hd; reflexivity.
Qed.

Definition interleave (X : list pairv) : list R :=
  flat_map (fun x => [fst x; snd x]) X.

Definition halfsplit (X : list pairv) : list R := map fst X ++ map snd X.

Definition faithful (asm : list pairv -> list R) : Prop :=
  forall X Y, length X = length Y -> dot (asm X) (asm Y) = pdot X Y.

Lemma dot_app : forall u1 v1 u2 v2,
  length u1 = length v1 ->
  dot (u1 ++ u2) (v1 ++ v2) = dot u1 v1 + dot u2 v2.
Proof.
  induction u1 as [|a us IH]; intros [|b vs] u2 v2 Hlen; simpl in *;
    try discriminate.
  - ring.
  - rewrite (IH vs u2 v2) by lia; ring.
Qed.

Lemma combine_app : forall (u1 v1 : list R) (u2 v2 : list R),
  length u1 = length v1 ->
  combine (u1 ++ u2) (v1 ++ v2) = combine u1 v1 ++ combine u2 v2.
Proof.
  induction u1 as [|a us IH]; intros [|b vs] u2 v2 Hlen; simpl in *;
    try discriminate; try reflexivity.
  rewrite (IH vs u2 v2) by lia; reflexivity.
Qed.

Definition sumprod (l : list (R * R)) : R :=
  fold_right (fun p acc => fst p * snd p + acc) 0 l.

Lemma dot_sumprod : forall u v,
  length u = length v -> dot u v = sumprod (combine u v).
Proof.
  induction u as [|a us IH]; intros [|b vs] Hlen; simpl in *;
    try discriminate; try reflexivity.
  unfold sumprod in *; simpl; rewrite (IH vs) by lia; reflexivity.
Qed.

Lemma sumprod_perm : forall l l', Permutation l l' -> sumprod l = sumprod l'.
Proof.
  intros l l' H; induction H; unfold sumprod in *; simpl in *;
    try reflexivity.
  - rewrite IHPermutation; reflexivity.
  - ring.
  - rewrite IHPermutation1, IHPermutation2; reflexivity.
Qed.

Theorem dot_perm : forall u v u' v',
  length u = length v ->
  length u' = length v' ->
  Permutation (combine u v) (combine u' v') ->
  dot u v = dot u' v'.
Proof.
  intros u v u' v' H1 H2 Hp.
  rewrite (dot_sumprod u v H1), (dot_sumprod u' v' H2).
  apply sumprod_perm; exact Hp.
Qed.

Lemma interleave_length : forall X, length (interleave X) = (2 * length X)%nat.
Proof.
  induction X as [|x xs IH]; simpl; [reflexivity | rewrite IH; lia].
Qed.

Lemma halfsplit_length : forall X, length (halfsplit X) = (2 * length X)%nat.
Proof.
  intros X; unfold halfsplit.
  rewrite length_app, !length_map; lia.
Qed.

Lemma interleave_faithful : faithful interleave.
Proof.
  unfold faithful.
  induction X as [|x xs IH]; intros [|y ys] Hlen; simpl in *;
    try discriminate; try reflexivity.
  rewrite (IH ys) by lia; unfold pmul; ring.
Qed.

Theorem halfsplit_perm_interleave : forall X Y,
  length X = length Y ->
  Permutation (combine (halfsplit X) (halfsplit Y))
              (combine (interleave X) (interleave Y)).
Proof.
  induction X as [|x xs IH]; intros [|y ys] Hlen; simpl in *;
    try discriminate.
  - apply Permutation_refl.
  - unfold halfsplit in *; simpl.
    rewrite combine_app by (rewrite !length_map; simpl; lia).
    simpl.
    apply perm_skip.
    etransitivity.
    + apply Permutation_sym, Permutation_middle.
    + apply perm_skip.
      rewrite <- combine_app by (rewrite !length_map; lia).
      apply (IH ys); lia.
Qed.

Theorem any_pairing_faithful : forall asm,
  (forall X, length (asm X) = (2 * length X)%nat) ->
  (forall X Y, length X = length Y ->
     Permutation (combine (asm X) (asm Y)) (combine (interleave X) (interleave Y))) ->
  faithful asm.
Proof.
  intros asm Hlen Hperm X Y HXY.
  rewrite (dot_perm (asm X) (asm Y) (interleave X) (interleave Y)).
  - apply interleave_faithful; exact HXY.
  - rewrite !Hlen; lia.
  - rewrite !interleave_length; lia.
  - apply Hperm; exact HXY.
Qed.

Theorem halfsplit_faithful : faithful halfsplit.
Proof.
  apply any_pairing_faithful.
  - exact halfsplit_length.
  - exact halfsplit_perm_interleave.
Qed.

Theorem pairing_invariance : forall asm,
  faithful asm ->
  forall X Y fs a b,
    length X = length Y ->
    dot (asm (rope fs a X)) (asm (rope fs b Y))
      = dot (asm (rope fs (a - b) X)) (asm Y).
Proof.
  intros asm Hf X Y fs a b Hlen.
  rewrite (Hf (rope fs a X) (rope fs b Y)) by (rewrite !rope_length; lia).
  rewrite (Hf (rope fs (a - b) X) Y) by (rewrite rope_length; lia).
  apply rope_relative.
Qed.

Corollary pairing_invariance_halfsplit : forall X Y fs a b,
  length X = length Y ->
  dot (halfsplit (rope fs a X)) (halfsplit (rope fs b Y))
    = dot (halfsplit (rope fs (a - b) X)) (halfsplit Y).
Proof. intros; apply pairing_invariance; [apply halfsplit_faithful | assumption]. Qed.

Corollary pairing_invariance_interleave : forall X Y fs a b,
  length X = length Y ->
  dot (interleave (rope fs a X)) (interleave (rope fs b Y))
    = dot (interleave (rope fs (a - b) X)) (interleave Y).
Proof. intros; apply pairing_invariance; [apply interleave_faithful | assumption]. Qed.

Theorem layout_equivalence : forall X Y,
  length X = length Y ->
  dot (halfsplit X) (halfsplit Y) = dot (interleave X) (interleave Y).
Proof.
  intros X Y H.
  rewrite (halfsplit_faithful X Y H), (interleave_faithful X Y H); reflexivity.
Qed.

Theorem halfsplit_neq_interleave :
  halfsplit [(0, 1); (0, 0)] <> interleave [(0, 1); (0, 0)].
Proof.
  unfold halfsplit, interleave; simpl; intros H.
  inversion H as [[H1 H2]].
  apply R1_neq_R0; symmetry; exact H1.
Qed.

Definition Xmix : list pairv := [(1, 0); (0, 0)].
Definition Ymix : list pairv := [(0, 0); (1, 0)].
Definition FSmix : list R := [PI / 2; PI / 2].

Theorem mixed_pairing_unsound :
  dot (halfsplit (rope FSmix 0 Xmix)) (interleave (rope FSmix 1 Ymix))
  <> dot (halfsplit (rope FSmix (0 - 1) Xmix)) (interleave Ymix).
Proof.
  assert (Ha : 0 * (PI / 2) = 0) by ring.
  assert (Hb : 1 * (PI / 2) = PI / 2) by ring.
  assert (Hc : (0 - 1) * (PI / 2) = - (PI / 2)) by ring.
  unfold Xmix, Ymix, FSmix, halfsplit, interleave, rope, rot; simpl.
  rewrite Ha, Hb, Hc.
  rewrite cos_0, sin_0, cos_neg, sin_neg, cos_PI2, sin_PI2.
  intros H; lra.
Qed.

Definition rope_angles (num den hd : nat) : nat :=
  Nat.min (num * hd / (den * 2)) (hd / 2).

Example gemma4_full_rope_angles : rope_angles 1 4 512 = 64%nat.
Proof. vm_compute; reflexivity. Qed.

Example gemma4_sliding_rope_angles : rope_angles 1 1 256 = 128%nat.
Proof. vm_compute; reflexivity. Qed.

Example gemma4_full_identity_pairs : (512 / 2 - rope_angles 1 4 512)%nat = 192%nat.
Proof. vm_compute; reflexivity. Qed.

Theorem gemma4_full_partial_relative : forall A XA XB YA YB m n,
  length A = 64%nat ->
  length XA = 64%nat -> length YA = 64%nat ->
  length XB = 192%nat -> length YB = 192%nat ->
  pdot (rope (padded A 256) m (XA ++ XB)) (rope (padded A 256) n (YA ++ YB))
    = pdot (rope A (m - n) XA) YA + pdot XB YB.
Proof.
  intros A XA XB YA YB m n HA HXA HYA HXB HYB.
  apply partial_relative_split; lia.
Qed.

Theorem gemma4_full_partial_tail_untouched : forall A XA XB m,
  length A = 64%nat -> length XA = 64%nat -> length XB = 192%nat ->
  rope (padded A 256) m (XA ++ XB) = rope A m XA ++ XB.
Proof.
  intros A XA XB m HA HXA HXB.
  apply partial_tail_identity; lia.
Qed.

Theorem gemma4_full_partial_pairing_invariance : forall A XA XB YA YB m n,
  length A = 64%nat ->
  length XA = 64%nat -> length YA = 64%nat ->
  length XB = 192%nat -> length YB = 192%nat ->
  dot (halfsplit (rope (padded A 256) m (XA ++ XB)))
      (halfsplit (rope (padded A 256) n (YA ++ YB)))
    = pdot (rope A (m - n) XA) YA + pdot XB YB.
Proof.
  intros A XA XB YA YB m n HA HXA HYA HXB HYB.
  rewrite halfsplit_faithful by (rewrite !rope_length, !length_app; lia).
  apply partial_relative_split; lia.
Qed.

Example rope_acts_nontrivially : rope [PI / 2] 1 [(1, 0)] = [(0, 1)].
Proof.
  assert (Hb : 1 * (PI / 2) = PI / 2) by ring.
  simpl; unfold rot; simpl; rewrite Hb, cos_PI2, sin_PI2.
  f_equal; f_equal; ring.
Qed.

Example gemma4_full_partial_hypotheses_satisfiable :
  exists (A : list R) (XA XB YA YB : list pairv),
    length A = 64%nat /\
    length XA = 64%nat /\ length YA = 64%nat /\
    length XB = 192%nat /\ length YB = 192%nat.
Proof.
  exists (repeat 0 64), (repeat (0, 0) 64), (repeat (0, 0) 192),
         (repeat (0, 0) 64), (repeat (0, 0) 192).
  repeat split; apply repeat_length.
Qed.

(* ---------------------------------------------------------------------- *)
(* Undoing RoPE at attention-read time (task 48).                          *)
(*                                                                         *)
(* Deriving V from the cached K means applying the inverse rotation. The   *)
(* cos/sin tables are indexed by position, so the read path cannot ask for *)
(* a negative position -- it reuses the SAME table entry with sin negated. *)
(* Modelling the tabulated entries as an arbitrary pair (c, s), rather     *)
(* than as cos t and sin t, is what makes the next three statements say    *)
(* something: they hold whatever angle the table actually encodes. *)

Definition rot_tab (c s : R) (x : pairv) : pairv :=
  (fst x * c - snd x * s, fst x * s + snd x * c).

Definition conj_tab (c s : R) (x : pairv) : pairv :=
  (fst x * c + snd x * s, snd x * c - fst x * s).

Definition scale (k : R) (x : pairv) : pairv := (k * fst x, k * snd x).
Definition sub (x y : pairv) : pairv := (fst x - fst y, snd x - snd y).

Lemma rot_tab_is_rot : forall t x, rot_tab (cos t) (sin t) x = rot t x.
Proof. intros t [a b]; unfold rot_tab, rot; simpl; f_equal; ring. Qed.

Theorem conj_tab_inverts_rot_tab_up_to_gain :
  forall c s x, conj_tab c s (rot_tab c s x) = scale (c * c + s * s) x.
Proof. intros c s [a b]; unfold conj_tab, rot_tab, scale; simpl; f_equal; ring. Qed.

(* The load-bearing one. The table may encode an angle u that is nowhere near
   the true position angle -- at position 262143 the f32 table is off by
   2.1e-3 relative, measured -- and the round trip is STILL the identity,
   because it composes the table with itself. So reconstructing V cannot
   inherit the table's angle error. *)
Theorem round_trip_exact_for_whatever_angle_the_table_encodes :
  forall u x, conj_tab (cos u) (sin u) (rot_tab (cos u) (sin u) x) = x.
Proof.
  intros u [a b].
  assert (Hg : cos u * cos u + sin u * sin u = 1)
    by (pose proof (sin2_cos2 u) as H; unfold Rsqr in H; lra).
  rewrite conj_tab_inverts_rot_tab_up_to_gain, Hg.
  unfold scale; simpl; f_equal; ring.
Qed.

(* And what error remains is exactly the gain defect, scaled by the input:
   nothing else can contribute. In f32 that defect is rounding-level, and the
   measured round trip is 2.8e-8 relative against a 2.1e-3 forward error. *)
Theorem round_trip_error_is_the_gain_defect :
  forall c s x, sub (conj_tab c s (rot_tab c s x)) x = scale (c * c + s * s - 1) x.
Proof. intros c s [a b]; unfold sub, conj_tab, rot_tab, scale; simpl; f_equal; ring. Qed.

(* V = unrope(K) / w_k, and the two steps commute, so the read path may fold
   the reciprocal into either the rotation or the epilogue. *)
Theorem the_scalar_rescale_commutes_with_the_inverse_rotation :
  forall k c s x, scale k (conj_tab c s x) = conj_tab c s (scale k x).
Proof. intros k c s [a b]; unfold scale, conj_tab; simpl; f_equal; ring. Qed.
