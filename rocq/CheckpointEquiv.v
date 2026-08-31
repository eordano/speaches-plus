From Stdlib Require Import Reals Ranalysis1 Ranalysis3 Ranalysis4 List PeanoNat Lia Lra.
From SpeachesPlus Require Import TrainLoss.
Import ListNotations.
Open Scope R_scope.

Fixpoint chain (ls : list (R -> R -> R)) (ps : list R) (x : R) : R :=
  match ls, ps with
  | f :: ls', q :: ps' => chain ls' ps' (f q x)
  | _, _ => x
  end.

Inductive chain_dx : list (R -> R -> R) -> list R -> R -> list R -> Prop :=
| chain_dx_nil : forall x, chain_dx [] [] x []
| chain_dx_cons : forall f ls q ps x d ds,
    derivable_pt_lim (fun v => f q v) x d ->
    chain_dx ls ps (f q x) ds ->
    chain_dx (f :: ls) (q :: ps) x (d :: ds).

Fixpoint prodl (l : list R) : R :=
  match l with
  | [] => 1
  | a :: t => a * prodl t
  end.

Lemma chain_dx_lengths :
  forall ls ps x ds,
    chain_dx ls ps x ds -> length ls = length ps /\ length ds = length ls.
Proof.
  induction 1; simpl; [auto|].
  destruct IHchain_dx; split; congruence.
Qed.

Lemma chain_input_deriv :
  forall ls ps x ds,
    chain_dx ls ps x ds ->
    derivable_pt_lim (fun v => chain ls ps v) x (prodl ds).
Proof.
  induction 1; simpl.
  - apply derivable_pt_lim_ext with (f := id); [reflexivity|].
    apply derivable_pt_lim_id.
  - apply derivable_pt_lim_ext
      with (f := comp (fun w => chain ls ps w) (fun v => f q v)).
    { intros z; unfold comp; reflexivity. }
    rewrite Rmult_comm.
    apply derivable_pt_lim_comp; assumption.
Qed.

Fixpoint ckpt (dxs dps : list R) (g : R) : list R * R :=
  match dxs, dps with
  | d :: dxs', q :: dps' =>
      let r := ckpt dxs' dps' g in
      (snd r * q :: fst r, snd r * d)
  | _, _ => ([], g)
  end.

Lemma ckpt_fst_length :
  forall dxs dps g,
    length dxs = length dps ->
    length (fst (ckpt dxs dps g)) = length dxs.
Proof.
  induction dxs as [|d dxs IH]; intros [|q dps] g Hl; simpl in *;
    try discriminate; [reflexivity|].
  rewrite IH; auto.
Qed.

Lemma ckpt_snd :
  forall dxs dps g,
    length dxs = length dps ->
    snd (ckpt dxs dps g) = g * prodl dxs.
Proof.
  induction dxs as [|d dxs IH]; intros [|q dps] g Hl; simpl in *;
    try discriminate.
  - ring.
  - rewrite IH by lia; ring.
Qed.

Lemma ckpt_app :
  forall dxsA dpsA dxsB dpsB g,
    length dxsA = length dpsA ->
    fst (ckpt (dxsA ++ dxsB) (dpsA ++ dpsB) g)
      = fst (ckpt dxsA dpsA (snd (ckpt dxsB dpsB g)))
          ++ fst (ckpt dxsB dpsB g)
    /\ snd (ckpt (dxsA ++ dxsB) (dpsA ++ dpsB) g)
      = snd (ckpt dxsA dpsA (snd (ckpt dxsB dpsB g))).
Proof.
  induction dxsA as [|d dxsA IH]; intros [|q dpsA] dxsB dpsB g Hl;
    simpl in *; try discriminate; [auto|].
  destruct (IH dpsA dxsB dpsB g ltac:(lia)) as [Hf Hs].
  rewrite Hf, Hs; auto.
Qed.

Lemma ckpt_middle_entry :
  forall dsA dpsA d dq dsB dpsB g,
    length dsA = length dpsA ->
    nth (length dsA) (fst (ckpt (dsA ++ d :: dsB) (dpsA ++ dq :: dpsB) g)) 0
    = snd (ckpt dsB dpsB g) * dq.
Proof.
  intros dsA dpsA d dq dsB dpsB g Hl.
  destruct (ckpt_app dsA dpsA (d :: dsB) (dq :: dpsB) g Hl) as [Hf _].
  rewrite Hf.
  rewrite <- (ckpt_fst_length dsA dpsA
                (snd (ckpt (d :: dsB) (dq :: dpsB) g)) Hl).
  simpl.
  apply nth_middle.
Qed.

Lemma surrogate_param_backward :
  forall (f : R -> R -> R) (c p h dq : R),
    derivable_pt_lim (fun q => f q h) p dq ->
    derivable_pt_lim (fun q => f q h * c) p (dq * c).
Proof. intros; apply derivable_pt_lim_scal_right; assumption. Qed.

Lemma surrogate_input_backward :
  forall (f : R -> R -> R) (c q h d : R),
    derivable_pt_lim (fun v => f q v) h d ->
    derivable_pt_lim (fun v => f q v * c) h (d * c).
Proof. intros; apply derivable_pt_lim_scal_right; assumption. Qed.

Theorem grad_next_is_dL_dboundary :
  forall ls ps h ds (L : R -> R) gN,
    chain_dx ls ps h ds ->
    derivable_pt_lim L (chain ls ps h) gN ->
    derivable_pt_lim (fun v => L (chain ls ps v)) h (snd (ckpt ds ps gN)).
Proof.
  intros ls ps h ds L gN Hdx HL.
  destruct (chain_dx_lengths _ _ _ _ Hdx) as [Hlp Hld].
  rewrite ckpt_snd by congruence.
  apply derivable_pt_lim_ext
    with (f := comp L (fun v => chain ls ps v)).
  { intros z; unfold comp; reflexivity. }
  apply derivable_pt_lim_comp; [apply chain_input_deriv; assumption | assumption].
Qed.

Theorem checkpointed_param_grad_sound :
  forall lsA psA x dsA dpsA (f : R -> R -> R) (p d dq : R)
         lsB psB dsB dpsB (L : R -> R) gN,
    chain_dx lsA psA x dsA ->
    length dsA = length dpsA ->
    derivable_pt_lim (fun q => f q (chain lsA psA x)) p dq ->
    chain_dx lsB psB (f p (chain lsA psA x)) dsB ->
    length dsB = length dpsB ->
    derivable_pt_lim L (chain lsB psB (f p (chain lsA psA x))) gN ->
    derivable_pt_lim
      (fun q => L (chain lsB psB (f q (chain lsA psA x)))) p
      (nth (length dsA)
           (fst (ckpt (dsA ++ d :: dsB) (dpsA ++ dq :: dpsB) gN)) 0).
Proof.
  intros lsA psA x dsA dpsA f p d dq lsB psB dsB dpsB L gN
         HdxA HlenA Hdq HdxB HlenB HL.
  rewrite ckpt_middle_entry by assumption.
  destruct (chain_dx_lengths _ _ _ _ HdxB) as [HlpB HldB].
  rewrite ckpt_snd by congruence.
  apply derivable_pt_lim_ext
    with (f := comp (fun v => L (chain lsB psB v))
                    (fun q => f q (chain lsA psA x))).
  { intros z; unfold comp; reflexivity. }
  apply derivable_pt_lim_comp; [assumption|].
  apply derivable_pt_lim_ext
    with (f := comp L (fun v => chain lsB psB v)).
  { intros z; unfold comp; reflexivity. }
  apply derivable_pt_lim_comp; [apply chain_input_deriv; assumption | assumption].
Qed.

Theorem retained_backward_matches_checkpointed :
  forall lsA psA x dsA dpsA (f : R -> R -> R) (p d dq : R)
         lsB psB dsB dpsB (L : R -> R) gN rg,
    chain_dx lsA psA x dsA ->
    length dsA = length dpsA ->
    derivable_pt_lim (fun q => f q (chain lsA psA x)) p dq ->
    chain_dx lsB psB (f p (chain lsA psA x)) dsB ->
    length dsB = length dpsB ->
    derivable_pt_lim L (chain lsB psB (f p (chain lsA psA x))) gN ->
    derivable_pt_lim
      (fun q => L (chain lsB psB (f q (chain lsA psA x)))) p rg ->
    rg = nth (length dsA)
             (fst (ckpt (dsA ++ d :: dsB) (dpsA ++ dq :: dpsB) gN)) 0.
Proof.
  intros.
  eapply uniqueness_limite; [eassumption|].
  eapply checkpointed_param_grad_sound; eassumption.
Qed.

Fixpoint suml (l : list R) : R :=
  match l with
  | [] => 0
  | a :: t => a + suml t
  end.

Lemma suml_app : forall l l', suml (l ++ l') = suml l + suml l'.
Proof. induction l; intros; simpl; [ring | rewrite IHl; ring]. Qed.

Definition evalc (cs : list (R -> R)) (t : R) : list R := map (fun c => c t) cs.

Inductive coords_deriv (t0 : R) : list (R -> R) -> list R -> Prop :=
| coords_deriv_nil : coords_deriv t0 [] []
| coords_deriv_cons : forall c d cs ds,
    derivable_pt_lim c t0 d ->
    coords_deriv t0 cs ds ->
    coords_deriv t0 (c :: cs) (d :: ds).

Lemma coords_deriv_length :
  forall t0 cs ds, coords_deriv t0 cs ds -> length ds = length cs.
Proof. induction 1; simpl; auto. Qed.

Lemma sum_coords_deriv :
  forall t0 cs ds,
    coords_deriv t0 cs ds ->
    derivable_pt_lim (fun t => suml (evalc cs t)) t0 (suml ds).
Proof.
  induction 1; simpl.
  - apply const_deriv_zero with (c := 0); reflexivity.
  - apply derivable_pt_lim_ext
      with (f := fun t => c t + suml (evalc cs t)); [reflexivity|].
    apply derivable_pt_lim_plus; assumption.
Qed.

Fixpoint wsum (z ds : list R) : R :=
  match z, ds with
  | a :: z', d :: ds' => exp a * d + wsum z' ds'
  | _, _ => 0
  end.

Lemma sumexp_coords_deriv :
  forall t0 cs ds,
    coords_deriv t0 cs ds ->
    derivable_pt_lim (fun t => sumexp (evalc cs t)) t0 (wsum (evalc cs t0) ds).
Proof.
  induction 1; simpl.
  - apply const_deriv_zero with (c := 0); reflexivity.
  - apply derivable_pt_lim_ext
      with (f := fun t => exp (c t) + sumexp (evalc cs t)); [reflexivity|].
    apply derivable_pt_lim_plus; [|assumption].
    apply derivable_pt_lim_ext with (f := comp exp c).
    { intros z; unfold comp; reflexivity. }
    apply derivable_pt_lim_comp; [assumption | apply derivable_pt_lim_exp].
Qed.

Lemma sumexp_nonneg : forall z, 0 <= sumexp z.
Proof.
  induction z; simpl; [lra|].
  generalize (exp_pos a); lra.
Qed.

Lemma sumexp_pos : forall z, z <> [] -> 0 < sumexp z.
Proof.
  intros [|a z] Hne; [congruence|]; simpl.
  generalize (exp_pos a) (sumexp_nonneg z); lra.
Qed.

Lemma nth_coords_deriv :
  forall t0 cs ds y,
    (y < length cs)%nat ->
    coords_deriv t0 cs ds ->
    derivable_pt_lim (fun t => nth y (evalc cs t) 0) t0 (nth y ds 0).
Proof.
  intros t0 cs ds y Hy Hcd; revert y Hy; induction Hcd; intros [|y] Hy;
    simpl in *; try lia.
  - apply derivable_pt_lim_ext with (f := c); [reflexivity | assumption].
  - apply derivable_pt_lim_ext
      with (f := fun t => nth y (evalc cs t) 0); [reflexivity|].
    apply IHHcd; lia.
Qed.

Lemma ce_coords_deriv :
  forall t0 cs ds y,
    cs <> [] ->
    (y < length cs)%nat ->
    coords_deriv t0 cs ds ->
    derivable_pt_lim (fun t => ce (evalc cs t) y) t0
      (wsum (evalc cs t0) ds / sumexp (evalc cs t0) - nth y ds 0).
Proof.
  intros t0 cs ds y Hne Hy Hcd.
  assert (Hpos : 0 < sumexp (evalc cs t0)).
  { apply sumexp_pos; destruct cs; [congruence | discriminate]. }
  apply derivable_pt_lim_ext
    with (f := fun t => ln (sumexp (evalc cs t)) - nth y (evalc cs t) 0).
  { intros z; unfold ce; reflexivity. }
  replace (wsum (evalc cs t0) ds / sumexp (evalc cs t0))
    with (/ sumexp (evalc cs t0) * wsum (evalc cs t0) ds)
    by (unfold Rdiv; ring).
  apply derivable_pt_lim_minus.
  - apply derivable_pt_lim_ext
      with (f := comp ln (fun t => sumexp (evalc cs t))).
    { intros z; unfold comp; reflexivity. }
    apply derivable_pt_lim_comp;
      [apply sumexp_coords_deriv; assumption | apply derivable_pt_lim_ln; assumption].
  - apply nth_coords_deriv; assumption.
Qed.

Lemma upd_same :
  forall (l : list R) j, (j < length l)%nat -> upd l j (nth j l 0) = l.
Proof.
  induction l; intros [|m] Hj; simpl in *; try lia; try reflexivity.
  rewrite IHl by lia; reflexivity.
Qed.

Fixpoint zeros (n : nat) : list R :=
  match n with
  | O => []
  | S m => 0 :: zeros m
  end.

Fixpoint onehot (n j : nat) : list R :=
  match n, j with
  | O, _ => []
  | S m, O => 1 :: zeros m
  | S m, S i => 0 :: onehot m i
  end.

Fixpoint coord_line (z : list R) (j : nat) : list (R -> R) :=
  match z, j with
  | [], _ => []
  | _ :: t, O => (fun v => v) :: map fct_cte t
  | a :: t, S m => fct_cte a :: coord_line t m
  end.

Lemma coord_line_length :
  forall z j, length (coord_line z j) = length z.
Proof.
  induction z; intros [|m]; simpl; auto using length_map.
Qed.

Lemma evalc_map_fct_cte : forall (l : list R) t, evalc (map fct_cte l) t = l.
Proof.
  induction l; intros; simpl; [reflexivity|].
  unfold evalc in *; simpl; unfold fct_cte at 1; rewrite IHl; reflexivity.
Qed.

Lemma evalc_coord_line :
  forall z j t, (j < length z)%nat -> evalc (coord_line z j) t = upd z j t.
Proof.
  induction z; intros [|m] t Hj; simpl in *; try lia.
  - unfold evalc; simpl; f_equal.
    apply (evalc_map_fct_cte z t).
  - unfold evalc in *; simpl; unfold fct_cte at 1; f_equal.
    apply IHz; lia.
Qed.

Lemma coords_deriv_cte_list :
  forall t0 l, coords_deriv t0 (map fct_cte l) (zeros (length l)).
Proof.
  induction l; simpl; constructor; auto using derivable_pt_lim_const.
Qed.

Lemma coord_line_deriv :
  forall z j t0,
    (j < length z)%nat ->
    coords_deriv t0 (coord_line z j) (onehot (length z) j).
Proof.
  induction z; intros [|m] t0 Hj; simpl in *; try lia; constructor.
  - apply derivable_pt_lim_ext with (f := id); [reflexivity|].
    apply derivable_pt_lim_id.
  - apply coords_deriv_cte_list.
  - apply derivable_pt_lim_const.
  - apply IHz; lia.
Qed.

Lemma wsum_zeros : forall z n, wsum z (zeros n) = 0.
Proof.
  induction z; intros [|n]; simpl; try reflexivity.
  rewrite IHz; ring.
Qed.

Lemma wsum_onehot :
  forall z j, (j < length z)%nat ->
    wsum z (onehot (length z) j) = exp (nth j z 0).
Proof.
  induction z; intros [|m] Hj; simpl in *; try lia.
  - rewrite wsum_zeros; ring.
  - rewrite IHz by lia; ring.
Qed.

Lemma nth_zeros : forall n y, nth y (zeros n) 0 = 0.
Proof.
  induction n; intros [|y]; simpl; auto.
Qed.

Lemma nth_onehot :
  forall n j y, (j < n)%nat ->
    nth y (onehot n j) 0 = if Nat.eqb j y then 1 else 0.
Proof.
  induction n; intros [|j] [|y] Hj; simpl in *; try lia; try reflexivity.
  - apply nth_zeros.
  - apply IHn; lia.
Qed.

Definition softmaxn (z : list R) (j : nat) : R := exp (nth j z 0) / sumexp z.

Theorem ce_logit_grad :
  forall z y j,
    z <> [] -> (j < length z)%nat -> (y < length z)%nat ->
    derivable_pt_lim (fun v => ce (upd z j v) y) (nth j z 0)
      (softmaxn z j - (if Nat.eqb j y then 1 else 0)).
Proof.
  intros z y j Hne Hj Hy.
  assert (Hlen := coord_line_length z j).
  assert (Hcs : coord_line z j <> []).
  { destruct z; [congruence|]; destruct j; simpl; discriminate. }
  pose proof (ce_coords_deriv (nth j z 0) (coord_line z j)
                (onehot (length z) j) y Hcs
                ltac:(rewrite Hlen; exact Hy)
                (coord_line_deriv z j (nth j z 0) Hj)) as Hd.
  rewrite evalc_coord_line in Hd by exact Hj.
  rewrite upd_same in Hd by exact Hj.
  rewrite wsum_onehot in Hd by exact Hj.
  rewrite nth_onehot in Hd by exact Hj.
  apply derivable_pt_lim_ext
    with (f := fun t => ce (evalc (coord_line z j) t) y).
  { intros t; rewrite evalc_coord_line by exact Hj; reflexivity. }
  exact Hd.
Qed.

Theorem gz_entry_sound :
  forall z y j denom,
    z <> [] -> (j < length z)%nat -> (y < length z)%nat ->
    derivable_pt_lim (fun v => ce (upd z j v) y / denom) (nth j z 0)
      ((softmaxn z j - (if Nat.eqb j y then 1 else 0)) / denom).
Proof.
  intros; apply derivable_pt_lim_div_scal, ce_logit_grad; assumption.
Qed.

Lemma cosh_pos : forall x, 0 < cosh x.
Proof.
  intros x; unfold cosh.
  pose proof (exp_pos x); pose proof (exp_pos (- x)); lra.
Qed.

Lemma cosh2_sinh2 : forall x, cosh x * cosh x - sinh x * sinh x = 1.
Proof.
  intros x.
  assert (Hab : exp x * exp (- x) = 1)
    by (rewrite <- exp_plus; replace (x + - x) with 0 by ring; apply exp_0).
  unfold cosh, sinh; rewrite <- Hab; field.
Qed.

Lemma derivable_pt_lim_tanh :
  forall x, derivable_pt_lim tanh x (1 - tanh x * tanh x).
Proof.
  intros x.
  assert (Hc := cosh_pos x).
  replace (1 - tanh x * tanh x)
    with ((cosh x * cosh x - sinh x * sinh x) / Rsqr (cosh x)).
  2:{ unfold tanh, Rsqr; field; apply Rgt_not_eq; exact Hc. }
  unfold tanh.
  apply derivable_pt_lim_div;
    [apply derivable_pt_lim_sinh
    |apply derivable_pt_lim_cosh
    |apply Rgt_not_eq; exact Hc].
Qed.

Definition softcap (c x : R) : R := c * tanh (x / c).

Lemma tanh_0 : tanh 0 = 0.
Proof. unfold tanh; rewrite sinh_0; unfold Rdiv; ring. Qed.

Lemma softcap_0 : forall c, softcap c 0 = 0.
Proof.
  intros; unfold softcap.
  replace (0 / c) with 0 by (unfold Rdiv; ring).
  rewrite tanh_0; ring.
Qed.

Lemma softcap_deriv :
  forall c x, c <> 0 ->
    derivable_pt_lim (softcap c) x
      (1 - softcap c x / c * (softcap c x / c)).
Proof.
  intros c x Hc.
  replace (1 - softcap c x / c * (softcap c x / c))
    with (c * ((1 - tanh (x / c) * tanh (x / c)) * (1 / c))).
  2:{ unfold softcap; field; exact Hc. }
  apply derivable_pt_lim_ext
    with (f := mult_real_fct c (comp tanh (fun y => y / c))).
  { intros t; unfold mult_real_fct, comp, softcap; reflexivity. }
  apply derivable_pt_lim_scal.
  apply derivable_pt_lim_comp.
  - apply derivable_pt_lim_div_scal, derivable_pt_lim_id.
  - apply derivable_pt_lim_tanh.
Qed.

Lemma map_upd :
  forall (f : R -> R) (l : list R) j v,
    map f (upd l j v) = upd (map f l) j (f v).
Proof.
  induction l; intros [|m] v; simpl; try reflexivity.
  rewrite IHl; reflexivity.
Qed.

Lemma nth_map0 :
  forall (f : R -> R) (l : list R) j,
    f 0 = 0 -> nth j (map f l) 0 = f (nth j l 0).
Proof.
  induction l; intros [|m] Hf; simpl; try (symmetry; exact Hf); auto.
Qed.

Theorem capped_gz_entry_sound :
  forall raw y j c denom,
    c <> 0 -> raw <> [] -> (j < length raw)%nat -> (y < length raw)%nat ->
    derivable_pt_lim
      (fun v => ce (map (softcap c) (upd raw j v)) y / denom)
      (nth j raw 0)
      ((softmaxn (map (softcap c) raw) j - (if Nat.eqb j y then 1 else 0))
         / denom
       * (1 - nth j (map (softcap c) raw) 0 / c
            * (nth j (map (softcap c) raw) 0 / c))).
Proof.
  intros raw y j c denom Hc Hne Hj Hy.
  set (z := map (softcap c) raw).
  assert (Hzlen : length z = length raw) by apply length_map.
  assert (Hzne : z <> []).
  { unfold z; destruct raw; [congruence | discriminate]. }
  assert (Hnth : nth j z 0 = softcap c (nth j raw 0))
    by (apply nth_map0, softcap_0).
  apply derivable_pt_lim_ext
    with (f := comp (fun w => ce (upd z j w) y / denom) (softcap c)).
  { intros t; unfold comp, z; rewrite map_upd; reflexivity. }
  apply derivable_pt_lim_comp.
  - rewrite Hnth; apply softcap_deriv; exact Hc.
  - rewrite <- Hnth.
    apply gz_entry_sound; [exact Hzne | lia | lia].
Qed.

Fixpoint dotl (u v : list R) : R :=
  match u, v with
  | a :: u', b :: v' => a * b + dotl u' v'
  | _, _ => 0
  end.

Lemma dotl_upd_affine :
  forall (u w : list R) i v,
    (i < length u)%nat -> length w = length u ->
    dotl (upd u i v) w = dotl u w + (v - nth i u 0) * nth i w 0.
Proof.
  induction u as [|a u IH]; intros [|b w] [|m] v Hi Hw; simpl in *;
    try lia; try discriminate.
  - ring.
  - rewrite (IH w m v) by lia; ring.
Qed.

Lemma affine_deriv :
  forall a b t0, derivable_pt_lim (fun v => a + v * b) t0 b.
Proof.
  intros a b t0.
  assert (H : derivable_pt_lim (plus_fct (fct_cte a) (fun v => v * b)) t0
                (0 + 1 * b)).
  { apply derivable_pt_lim_plus.
    - apply derivable_pt_lim_const.
    - apply derivable_pt_lim_scal_right, derivable_pt_lim_id. }
  replace (0 + 1 * b) with b in H by ring.
  exact H.
Qed.

Lemma dotl_upd_deriv :
  forall (u w : list R) i t0,
    (i < length u)%nat -> length w = length u ->
    derivable_pt_lim (fun v => dotl (upd u i v) w) t0 (nth i w 0).
Proof.
  intros u w i t0 Hi Hw.
  apply derivable_pt_lim_ext
    with (f := fun v => (dotl u w - nth i u 0 * nth i w 0) + v * nth i w 0).
  { intros z; rewrite dotl_upd_affine by assumption; ring. }
  apply affine_deriv.
Qed.

Definition colW (W : list (list R)) (h : nat) : list R :=
  map (fun row => nth h row 0) W.

Lemma matmul_row_coords_deriv :
  forall (hc : list R) (W : list (list R)) h0 t0,
    (h0 < length hc)%nat ->
    Forall (fun row => length row = length hc) W ->
    coords_deriv t0
      (map (fun row => fun t => dotl (upd hc h0 t) row) W)
      (colW W h0).
Proof.
  intros hc W h0 t0 Hh HF.
  induction HF as [|row W' Hrow HF IH]; simpl; constructor.
  - apply dotl_upd_deriv; assumption.
  - exact IH.
Qed.

Theorem linear_bwd_row_grad :
  forall (hc : list R) (W : list (list R)) y h0 denom,
    W <> [] -> (y < length W)%nat -> (h0 < length hc)%nat ->
    Forall (fun row => length row = length hc) W ->
    derivable_pt_lim
      (fun v => ce (map (dotl (upd hc h0 v)) W) y / denom)
      (nth h0 hc 0)
      ((wsum (map (dotl hc) W) (colW W h0) / sumexp (map (dotl hc) W)
        - nth y (colW W h0) 0) / denom).
Proof.
  intros hc W y h0 denom HW Hy Hh HF.
  set (cs := map (fun row => fun t => dotl (upd hc h0 t) row) W).
  assert (Hev : forall t,
             evalc cs t = map (fun row => dotl (upd hc h0 t) row) W).
  { intros t; unfold cs, evalc; rewrite map_map; reflexivity. }
  assert (Hcs_len : length cs = length W) by apply length_map.
  assert (Hcsne : cs <> []).
  { unfold cs; destruct W; [congruence | discriminate]. }
  assert (Hz0 : evalc cs (nth h0 hc 0) = map (dotl hc) W).
  { rewrite Hev; rewrite upd_same by exact Hh; reflexivity. }
  pose proof (ce_coords_deriv (nth h0 hc 0) cs (colW W h0) y Hcsne
                ltac:(rewrite Hcs_len; exact Hy)
                (matmul_row_coords_deriv hc W h0 (nth h0 hc 0) Hh HF)) as Hd.
  rewrite Hz0 in Hd.
  apply derivable_pt_lim_ext
    with (f := fun t => ce (evalc cs t) y / denom).
  { intros t; rewrite Hev; reflexivity. }
  apply derivable_pt_lim_div_scal; exact Hd.
Qed.

Fixpoint gz_list (S denom : R) (z : list R) (y : nat) : list R :=
  match z, y with
  | [], _ => []
  | a :: z', O =>
      ((exp a / S - 1) / denom) :: map (fun b => exp b / S / denom) z'
  | a :: z', S y' => (exp a / S / denom) :: gz_list S denom z' y'
  end.

Lemma dotl_map_wsum :
  forall S denom z ds,
    dotl (map (fun b => exp b / S / denom) z) ds = wsum z ds / S / denom.
Proof.
  induction z as [|a z IH]; intros [|d ds]; simpl; unfold Rdiv;
    try ring.
  rewrite IH; unfold Rdiv; ring.
Qed.

Lemma gz_list_dot :
  forall S denom z ds y,
    (y < length z)%nat -> length ds = length z ->
    dotl (gz_list S denom z y) ds
    = (wsum z ds / S - nth y ds 0) / denom.
Proof.
  induction z as [|a z IH]; intros [|d ds] [|y] Hy Hlen; simpl in *;
    try lia; try discriminate.
  - rewrite dotl_map_wsum; unfold Rdiv; ring.
  - rewrite IH by lia; unfold Rdiv; ring.
Qed.

Lemma gz_list_nth :
  forall S denom z y j,
    (j < length z)%nat ->
    nth j (gz_list S denom z y) 0
    = (exp (nth j z 0) / S - (if Nat.eqb j y then 1 else 0)) / denom.
Proof.
  induction z as [|a z IH]; intros [|y] [|m] Hj; simpl in *; try lia.
  - unfold Rdiv; ring.
  - rewrite (nth_indep (map (fun b => exp b / S / denom) z) 0
               (exp 0 / S / denom))
      by (rewrite length_map; lia).
    rewrite (map_nth (fun b => exp b / S / denom) z 0 m).
    unfold Rdiv; ring.
  - unfold Rdiv; ring.
  - apply IH; lia.
Qed.

Theorem chunked_ce_backward_is_dy_W :
  forall (hc : list R) (W : list (list R)) y h0 denom,
    W <> [] -> (y < length W)%nat -> (h0 < length hc)%nat ->
    Forall (fun row => length row = length hc) W ->
    derivable_pt_lim
      (fun v => ce (map (dotl (upd hc h0 v)) W) y / denom)
      (nth h0 hc 0)
      (dotl (gz_list (sumexp (map (dotl hc) W)) denom (map (dotl hc) W) y)
            (colW W h0)).
Proof.
  intros hc W y h0 denom HW Hy Hh HF.
  rewrite gz_list_dot;
    [ apply linear_bwd_row_grad; assumption
    | rewrite length_map; exact Hy
    | unfold colW; rewrite ? length_map; reflexivity ].
Qed.

Definition dcap (c x : R) : R := 1 - softcap c x / c * (softcap c x / c).

Fixpoint mull (u v : list R) : list R :=
  match u, v with
  | a :: u', b :: v' => a * b :: mull u' v'
  | _, _ => []
  end.

Lemma mull_length :
  forall u v, length (mull u v) = Nat.min (length u) (length v).
Proof.
  induction u as [|a u IH]; intros [|b v]; simpl; try reflexivity.
  rewrite IH; reflexivity.
Qed.

Lemma nth_mull :
  forall u v j, (j < length u)%nat -> (j < length v)%nat ->
    nth j (mull u v) 0 = nth j u 0 * nth j v 0.
Proof.
  induction u as [|a u IH]; intros [|b v] [|j] Hu Hv; simpl in *;
    try lia; try ring.
  apply IH; lia.
Qed.

Lemma dotl_mull_assoc :
  forall u v w, dotl (mull u v) w = dotl u (mull v w).
Proof.
  induction u as [|a u IH]; intros [|b v] [|d w]; simpl; try ring.
  rewrite IH; ring.
Qed.

Lemma gz_list_length :
  forall S denom z y, length (gz_list S denom z y) = length z.
Proof.
  induction z as [|a z IH]; intros [|y]; simpl; try reflexivity.
  - rewrite length_map; reflexivity.
  - rewrite IH; reflexivity.
Qed.

Lemma capped_matmul_row_coords_deriv :
  forall (hc : list R) (W : list (list R)) h0 c,
    c <> 0 -> (h0 < length hc)%nat ->
    Forall (fun row => length row = length hc) W ->
    coords_deriv (nth h0 hc 0)
      (map (fun row => fun t => softcap c (dotl (upd hc h0 t) row)) W)
      (mull (map (dcap c) (map (dotl hc) W)) (colW W h0)).
Proof.
  intros hc W h0 c Hc Hh HF.
  rewrite map_map; unfold colW.
  induction HF as [|row W' Hrow HF IH]; simpl; constructor; [|exact IH].
  apply derivable_pt_lim_ext
    with (f := comp (softcap c) (fun t => dotl (upd hc h0 t) row)).
  { intros z; unfold comp; reflexivity. }
  unfold dcap.
  apply derivable_pt_lim_comp.
  - apply dotl_upd_deriv; assumption.
  - rewrite upd_same by exact Hh.
    apply softcap_deriv; exact Hc.
Qed.

Theorem capped_linear_bwd_row_grad :
  forall (hc : list R) (W : list (list R)) y h0 c denom,
    c <> 0 -> W <> [] -> (y < length W)%nat -> (h0 < length hc)%nat ->
    Forall (fun row => length row = length hc) W ->
    derivable_pt_lim
      (fun v => ce (map (softcap c) (map (dotl (upd hc h0 v)) W)) y / denom)
      (nth h0 hc 0)
      ((wsum (map (softcap c) (map (dotl hc) W))
             (mull (map (dcap c) (map (dotl hc) W)) (colW W h0))
          / sumexp (map (softcap c) (map (dotl hc) W))
        - nth y (mull (map (dcap c) (map (dotl hc) W)) (colW W h0)) 0)
       / denom).
Proof.
  intros hc W y h0 c denom Hc HW Hy Hh HF.
  set (cs := map (fun row => fun t => softcap c (dotl (upd hc h0 t) row)) W).
  assert (Hev : forall t,
             evalc cs t = map (softcap c) (map (dotl (upd hc h0 t)) W)).
  { intros t; unfold cs, evalc; rewrite ! map_map; reflexivity. }
  assert (Hcs_len : length cs = length W) by apply length_map.
  assert (Hcsne : cs <> []).
  { unfold cs; destruct W; [congruence | discriminate]. }
  assert (Hz0 : evalc cs (nth h0 hc 0) = map (softcap c) (map (dotl hc) W)).
  { rewrite Hev, upd_same by exact Hh; reflexivity. }
  pose proof (ce_coords_deriv (nth h0 hc 0) cs
                (mull (map (dcap c) (map (dotl hc) W)) (colW W h0)) y
                Hcsne ltac:(rewrite Hcs_len; exact Hy)
                (capped_matmul_row_coords_deriv hc W h0 c Hc Hh HF)) as Hd.
  rewrite Hz0 in Hd.
  apply derivable_pt_lim_ext
    with (f := fun t => ce (evalc cs t) y / denom).
  { intros t; rewrite Hev; reflexivity. }
  apply derivable_pt_lim_div_scal; exact Hd.
Qed.

Theorem capped_chunked_ce_backward_is_dcap_gz_W :
  forall (hc : list R) (W : list (list R)) y h0 c denom,
    c <> 0 -> W <> [] -> (y < length W)%nat -> (h0 < length hc)%nat ->
    Forall (fun row => length row = length hc) W ->
    derivable_pt_lim
      (fun v => ce (map (softcap c) (map (dotl (upd hc h0 v)) W)) y / denom)
      (nth h0 hc 0)
      (dotl (mull (gz_list (sumexp (map (softcap c) (map (dotl hc) W))) denom
                           (map (softcap c) (map (dotl hc) W)) y)
                  (map (dcap c) (map (dotl hc) W)))
            (colW W h0)).
Proof.
  intros hc W y h0 c denom Hc HW Hy Hh HF.
  rewrite dotl_mull_assoc.
  rewrite gz_list_dot;
    [ apply capped_linear_bwd_row_grad; assumption
    | rewrite ! length_map; exact Hy
    | rewrite mull_length; unfold colW; rewrite ! length_map;
      apply Nat.min_id ].
Qed.

Fixpoint chunk_concat {A B : Type} (f : list A -> list B) (fuel c : nat)
         (l : list A) : list B :=
  match fuel with
  | O => []
  | S fu =>
      match l with
      | [] => []
      | _ :: _ =>
          f (firstn (Nat.min c (length l)) l)
            ++ chunk_concat f fu c (skipn (Nat.min c (length l)) l)
      end
  end.

Fixpoint chunk_sum {A : Type} (f : list A -> R) (fuel c : nat)
         (l : list A) : R :=
  match fuel with
  | O => 0
  | S fu =>
      match l with
      | [] => 0
      | _ :: _ =>
          f (firstn (Nat.min c (length l)) l)
            + chunk_sum f fu c (skipn (Nat.min c (length l)) l)
      end
  end.

Lemma chunk_concat_map :
  forall (A B : Type) (g : A -> B) c fuel (l : list A),
    (1 <= c)%nat -> (length l <= fuel)%nat ->
    chunk_concat (map g) fuel c l = map g l.
Proof.
  intros A B g c fuel; revert c; induction fuel as [|fu IH]; intros c l Hc Hl.
  - destruct l; simpl in *; [reflexivity | lia].
  - destruct l as [|a l']; [reflexivity|].
    cbn [chunk_concat].
    rewrite IH.
    + rewrite <- map_app, firstn_skipn; reflexivity.
    + exact Hc.
    + rewrite length_skipn; simpl length in *; lia.
Qed.

Lemma chunk_sum_suml :
  forall (A : Type) (g : A -> R) c fuel (l : list A),
    (1 <= c)%nat -> (length l <= fuel)%nat ->
    chunk_sum (fun p => suml (map g p)) fuel c l = suml (map g l).
Proof.
  intros A g c fuel; revert c; induction fuel as [|fu IH]; intros c l Hc Hl.
  - destruct l; simpl in *; [reflexivity | lia].
  - destruct l as [|a l']; [reflexivity|].
    cbn [chunk_sum].
    rewrite IH.
    + rewrite <- suml_app, <- map_app, firstn_skipn; reflexivity.
    + exact Hc.
    + rewrite length_skipn; simpl length in *; lia.
Qed.

Definition row_ce (denom : R) (p : list R * nat) : R := ce (fst p) (snd p) / denom.
Definition row_gz (denom : R) (p : list R * nat) : list R :=
  gz_list (sumexp (fst p)) denom (fst p) (snd p).

Theorem chunked_grads_equal_monolithic :
  forall denom c fuel (rows : list (list R * nat)),
    (1 <= c)%nat -> (length rows <= fuel)%nat ->
    chunk_concat (map (row_gz denom)) fuel c rows = map (row_gz denom) rows.
Proof. intros; apply chunk_concat_map; assumption. Qed.

Theorem chunked_loss_equals_monolithic :
  forall denom c fuel (rows : list (list R * nat)),
    (1 <= c)%nat -> (length rows <= fuel)%nat ->
    chunk_sum (fun p => suml (map (row_ce denom) p)) fuel c rows
    = suml (map (row_ce denom) rows).
Proof. intros; apply chunk_sum_suml; assumption. Qed.

Definition capped_row_ce (c denom : R) (p : list R * nat) : R :=
  ce (map (softcap c) (fst p)) (snd p) / denom.
Definition capped_row_gz (c denom : R) (p : list R * nat) : list R :=
  mull (gz_list (sumexp (map (softcap c) (fst p))) denom
                (map (softcap c) (fst p)) (snd p))
       (map (dcap c) (fst p)).

Theorem capped_chunked_grads_equal_monolithic :
  forall cap denom c fuel (rows : list (list R * nat)),
    (1 <= c)%nat -> (length rows <= fuel)%nat ->
    chunk_concat (map (capped_row_gz cap denom)) fuel c rows
    = map (capped_row_gz cap denom) rows.
Proof. intros; apply chunk_concat_map; assumption. Qed.

Theorem capped_chunked_loss_equals_monolithic :
  forall cap denom c fuel (rows : list (list R * nat)),
    (1 <= c)%nat -> (length rows <= fuel)%nat ->
    chunk_sum (fun p => suml (map (capped_row_ce cap denom) p)) fuel c rows
    = suml (map (capped_row_ce cap denom) rows).
Proof. intros; apply chunk_sum_suml; assumption. Qed.

Lemma combine_skipn :
  forall (A B : Type) n (l : list A) (l' : list B),
    skipn n (combine l l') = combine (skipn n l) (skipn n l').
Proof.
  induction n; intros [|a l] [|b l']; simpl;
    try (rewrite combine_nil; reflexivity); try reflexivity.
  apply IHn.
Qed.

Theorem parallel_slice_agrees :
  forall (A B : Type) n (l : list A) (l' : list B),
    firstn n (combine l l') = combine (firstn n l) (firstn n l')
    /\ skipn n (combine l l') = combine (skipn n l) (skipn n l').
Proof. intros; split; [apply combine_firstn | apply combine_skipn]. Qed.

Lemma suml_map_upd :
  forall (A : Type) (f : A -> R) (l : list A) r (a x : A),
    nth_error l r = Some x ->
    suml (map f (upd l r a)) = suml (map f l) - f x + f a.
Proof.
  induction l as [|h l IH]; intros [|m] a x Hr; simpl in *;
    try discriminate.
  - injection Hr as ->; ring.
  - rewrite (IH m a x Hr); ring.
Qed.

Theorem monolithic_grad_at_logit :
  forall (rows : list (list R * nat)) r j denom z y,
    nth_error rows r = Some (z, y) ->
    z <> [] -> (j < length z)%nat -> (y < length z)%nat ->
    derivable_pt_lim
      (fun v => suml (map (row_ce denom) (upd rows r (upd z j v, y))))
      (nth j z 0)
      (nth j (row_gz denom (z, y)) 0).
Proof.
  intros rows r j denom z y Hr Hne Hj Hy.
  unfold row_gz; simpl.
  rewrite gz_list_nth by exact Hj.
  apply derivable_pt_lim_ext
    with (f := fun v =>
                 (suml (map (row_ce denom) rows) - row_ce denom (z, y))
                 + ce (upd z j v) y / denom).
  { intros t.
    rewrite (suml_map_upd _ (row_ce denom) rows r (upd z j t, y) (z, y) Hr).
    unfold row_ce; simpl; ring. }
  replace ((exp (nth j z 0) / sumexp z - (if Nat.eqb j y then 1 else 0))
             / denom)
    with (0 + (exp (nth j z 0) / sumexp z - (if Nat.eqb j y then 1 else 0))
             / denom) by ring.
  apply derivable_pt_lim_plus.
  - apply const_deriv_zero
      with (c := suml (map (row_ce denom) rows) - row_ce denom (z, y));
      reflexivity.
  - apply gz_entry_sound; assumption.
Qed.

Lemma capped_row_gz_nth :
  forall c denom raw y j,
    (j < length raw)%nat ->
    nth j (capped_row_gz c denom (raw, y)) 0
    = (softmaxn (map (softcap c) raw) j - (if Nat.eqb j y then 1 else 0))
        / denom * dcap c (nth j raw 0).
Proof.
  intros c denom raw y j Hj.
  unfold capped_row_gz; simpl.
  rewrite nth_mull
    by (rewrite ? gz_list_length, ? length_map; exact Hj).
  rewrite gz_list_nth by (rewrite length_map; exact Hj).
  unfold softmaxn.
  rewrite (nth_indep (map (dcap c) raw) 0 (dcap c 0))
    by (rewrite length_map; exact Hj).
  rewrite (map_nth (dcap c) raw 0 j).
  reflexivity.
Qed.

Theorem capped_monolithic_grad_at_logit :
  forall (rows : list (list R * nat)) r j c denom raw y,
    c <> 0 ->
    nth_error rows r = Some (raw, y) ->
    raw <> [] -> (j < length raw)%nat -> (y < length raw)%nat ->
    derivable_pt_lim
      (fun v => suml (map (capped_row_ce c denom)
                          (upd rows r (upd raw j v, y))))
      (nth j raw 0)
      (nth j (capped_row_gz c denom (raw, y)) 0).
Proof.
  intros rows r j c denom raw y Hc Hr Hne Hj Hy.
  rewrite capped_row_gz_nth by exact Hj.
  apply derivable_pt_lim_ext
    with (f := fun v =>
                 (suml (map (capped_row_ce c denom) rows)
                    - capped_row_ce c denom (raw, y))
                 + ce (map (softcap c) (upd raw j v)) y / denom).
  { intros t.
    rewrite (suml_map_upd _ (capped_row_ce c denom) rows r
               (upd raw j t, y) (raw, y) Hr).
    unfold capped_row_ce; simpl; ring. }
  replace ((softmaxn (map (softcap c) raw) j
              - (if Nat.eqb j y then 1 else 0)) / denom
             * dcap c (nth j raw 0))
    with (0 + (softmaxn (map (softcap c) raw) j
                 - (if Nat.eqb j y then 1 else 0)) / denom
             * dcap c (nth j raw 0)) by ring.
  apply derivable_pt_lim_plus.
  - apply const_deriv_zero
      with (c := suml (map (capped_row_ce c denom) rows)
                   - capped_row_ce c denom (raw, y));
      reflexivity.
  - assert (Hnth : nth j (map (softcap c) raw) 0 = softcap c (nth j raw 0))
      by (apply nth_map0, softcap_0).
    unfold dcap; rewrite <- Hnth.
    apply capped_gz_entry_sound; assumption.
Qed.

Lemma nth_tl :
  forall (A : Type) (l : list A) t (d : A), nth t (tl l) d = nth (S t) l d.
Proof.
  destruct l; intros; simpl; [destruct t; reflexivity | reflexivity].
Qed.

Theorem next_token_shift_pairing :
  forall (A : Type) (hs : list A) (ids : list nat) t (d : A) (dn : nat),
    length hs = length (tl ids) ->
    nth t (combine hs (tl ids)) (d, dn) = (nth t hs d, nth (S t) ids dn).
Proof.
  intros; rewrite combine_nth by assumption; rewrite nth_tl; reflexivity.
Qed.

Lemma firstn_upd_high :
  forall (A : Type) n (l : list A) i (a : A),
    (n <= i)%nat -> firstn n (upd l i a) = firstn n l.
Proof.
  induction n; intros [|x l] [|m] a H; simpl; try reflexivity; try lia.
  f_equal; apply IHn; lia.
Qed.

Theorem head_last_row_zero_grad :
  forall (rows : list (list R)) (Lh : list (list R) -> R) i j v0,
    (length rows - 1 <= i)%nat ->
    derivable_pt_lim
      (fun v => Lh (firstn (length rows - 1)
                           (upd rows i (upd (nth i rows []) j v)))) v0 0.
Proof.
  intros.
  apply const_deriv_zero with (c := Lh (firstn (length rows - 1) rows)).
  intros t; rewrite firstn_upd_high by assumption; reflexivity.
Qed.

Theorem per_example_grad_accumulation :
  forall t0 (example_losses : list (R -> R)) (example_grads : list R),
    coords_deriv t0 example_losses example_grads ->
    derivable_pt_lim (fun q => suml (evalc example_losses q)) t0
      (suml example_grads).
Proof. exact sum_coords_deriv. Qed.
