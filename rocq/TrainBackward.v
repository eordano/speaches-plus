
From Stdlib Require Import Reals List Lia Lra FunctionalExtensionality.
Import ListNotations.
Open Scope R_scope.

Definition Buf := nat -> R.

Definition idx (cols i j : nat) : nat := (i * cols + j)%nat.

Fixpoint fsum (n : nat) (f : nat -> R) : R :=
  match n with
  | O => 0
  | S m => fsum m f + f m
  end.

Lemma fsum_ext_in : forall n f g,
  (forall i, (i < n)%nat -> f i = g i) -> fsum n f = fsum n g.
Proof.
  induction n as [|n IH]; intros f g H; simpl; [reflexivity|].
  rewrite (IH f g) by (intros; apply H; lia).
  rewrite H by lia. reflexivity.
Qed.

Lemma fsum_plus : forall n f g,
  fsum n (fun i => f i + g i) = fsum n f + fsum n g.
Proof.
  induction n as [|n IH]; intros; simpl; [ring|]. rewrite IH; ring.
Qed.

Lemma fsum_scal_l : forall n c f,
  fsum n (fun i => c * f i) = c * fsum n f.
Proof.
  induction n as [|n IH]; intros; simpl; [ring|]. rewrite IH; ring.
Qed.

Lemma fsum_scal_r : forall n c f,
  fsum n (fun i => f i * c) = fsum n f * c.
Proof.
  induction n as [|n IH]; intros; simpl; [ring|]. rewrite IH; ring.
Qed.

Lemma fsum_zero : forall n f,
  (forall i, (i < n)%nat -> f i = 0) -> fsum n f = 0.
Proof.
  induction n as [|n IH]; intros f H; simpl; [reflexivity|].
  rewrite IH by (intros; apply H; lia). rewrite H by lia. ring.
Qed.

Lemma fsum_swap : forall m n f,
  fsum m (fun i => fsum n (fun j => f i j))
    = fsum n (fun j => fsum m (fun i => f i j)).
Proof.
  induction m as [|m IH]; intros n f; simpl.
  - symmetry. apply fsum_zero. intros; reflexivity.
  - rewrite IH, <- fsum_plus. apply fsum_ext_in. intros; reflexivity.
Qed.

Lemma fsum_app : forall b a f,
  fsum (a + b) f = fsum a f + fsum b (fun j => f (a + j)%nat).
Proof.
  induction b as [|b IH]; intros a f.
  - replace (a + 0)%nat with a by lia. simpl. ring.
  - replace (a + S b)%nat with (S (a + b))%nat by lia. simpl.
    rewrite IH. ring.
Qed.

Lemma fsum_flatten : forall m n F,
  fsum (m * n) F = fsum m (fun i => fsum n (fun j => F (idx n i j))).
Proof.
  induction m as [|m IH]; intros n F; [reflexivity|].
  replace (S m * n)%nat with (m * n + n)%nat by lia.
  rewrite fsum_app, IH. simpl. f_equal.
Qed.

Lemma fsum_single : forall n s f,
  (s < n)%nat ->
  (forall t, (t < n)%nat -> t <> s -> f t = 0) ->
  fsum n f = f s.
Proof.
  induction n as [|n IH]; intros s f Hs H; [lia|].
  simpl. destruct (Nat.eq_dec s n) as [->|Hne].
  - rewrite fsum_zero by (intros t Ht; apply H; lia). ring.
  - rewrite (H n) by lia. rewrite (IH s f); [ring| lia |].
    intros; apply H; lia.
Qed.

Lemma idx_div : forall cols i j, (0 < cols)%nat -> (j < cols)%nat ->
  ((idx cols i j) / cols)%nat = i.
Proof.
  intros cols i j Hc Hj. unfold idx.
  symmetry. apply Nat.div_unique with (r := j); [lia| ring].
Qed.

Lemma idx_mod : forall cols i j, (0 < cols)%nat -> (j < cols)%nat ->
  ((idx cols i j) mod cols)%nat = j.
Proof.
  intros cols i j Hc Hj. unfold idx.
  symmetry. apply Nat.mod_unique with (q := i); [lia| ring].
Qed.

Definition gemm_bt (x w : Buf) (n k : nat) : Buf :=
  fun c => fsum k (fun p => x (idx k (c / n) p) * w (idx k (c mod n) p)).

Definition gemm_nn (a b : Buf) (n k : nat) : Buf :=
  fun c => fsum k (fun p => a (idx k (c / n) p) * b (idx n p (c mod n))).

Lemma gemm_bt_at : forall x w n k i j, (0 < n)%nat -> (j < n)%nat ->
  gemm_bt x w n k (idx n i j) = fsum k (fun p => x (idx k i p) * w (idx k j p)).
Proof.
  intros. unfold gemm_bt. rewrite idx_div, idx_mod by assumption. reflexivity.
Qed.

Lemma gemm_nn_at : forall a b n k i j, (0 < n)%nat -> (j < n)%nat ->
  gemm_nn a b n k (idx n i j) = fsum k (fun p => a (idx k i p) * b (idx n p j)).
Proof.
  intros. unfold gemm_nn. rewrite idx_div, idx_mod by assumption. reflexivity.
Qed.

Record CmCfg := mkCm {
  cm_transa : bool;
  cm_m : nat;
  cm_n : nat;
  cm_k : nat;
  cm_alpha : R;
  cm_lda : nat;
  cm_ldb : nat;
  cm_beta : R;
  cm_ldc : nat
}.

Definition cm_op_a (cfg : CmCfg) (A : Buf) (r p : nat) : R :=
  if cm_transa cfg then A (p + r * cm_lda cfg)%nat else A (r + p * cm_lda cfg)%nat.

Definition cublas_cm (cfg : CmCfg) (A B C : Buf) : Buf :=
  fun f =>
    if andb (Nat.ltb (f mod cm_ldc cfg) (cm_m cfg))
            (Nat.ltb (f / cm_ldc cfg) (cm_n cfg))
    then cm_alpha cfg
         * fsum (cm_k cfg) (fun p =>
             cm_op_a cfg A (f mod cm_ldc cfg) p
             * B (p + (f / cm_ldc cfg) * cm_ldb cfg)%nat)
         + cm_beta cfg * C f
    else C f.

Lemma cublas_cm_writes_only_the_m_by_n_block :
  forall cfg A B C f,
  (cm_m cfg <= f mod cm_ldc cfg)%nat \/ (cm_n cfg <= f / cm_ldc cfg)%nat ->
  cublas_cm cfg A B C f = C f.
Proof.
  intros cfg A B C f [H|H]; unfold cublas_cm.
  - rewrite (proj2 (Nat.ltb_ge _ _) H). reflexivity.
  - rewrite (proj2 (Nat.ltb_ge _ _) H), Bool.andb_false_r. reflexivity.
Qed.

Definition bt_offs_cfg (m n k : nat) (al be : R) : CmCfg :=
  mkCm true n m k al k k be n.

Definition nn_offs_cfg (m n k : nat) (al be : R) : CmCfg :=
  mkCm false n m k al n k be n.

Theorem bt_offs_config_realizes_alpha_x_wt_plus_beta_c :
  forall x w c0 al be m n k i j,
  (0 < n)%nat -> (i < m)%nat -> (j < n)%nat ->
  cublas_cm (bt_offs_cfg m n k al be) w x c0 (idx n i j)
    = al * gemm_bt x w n k (idx n i j) + be * c0 (idx n i j).
Proof.
  intros x w c0 al be m n k i j Hn Hi Hj.
  unfold cublas_cm, bt_offs_cfg, cm_op_a; simpl.
  rewrite idx_div, idx_mod by assumption.
  destruct (Nat.ltb_spec j n) as [_|]; [|lia].
  destruct (Nat.ltb_spec i m) as [_|]; [|lia].
  simpl.
  rewrite gemm_bt_at by assumption.
  f_equal. f_equal.
  apply fsum_ext_in. intros p _.
  replace (p + j * k)%nat with (idx k j p) by (unfold idx; lia).
  replace (p + i * k)%nat with (idx k i p) by (unfold idx; lia).
  ring.
Qed.

Theorem bt_offs_config_realizes_x_times_w_transposed :
  forall x w c0 m n k i j, (0 < n)%nat -> (i < m)%nat -> (j < n)%nat ->
  cublas_cm (bt_offs_cfg m n k 1 0) w x c0 (idx n i j)
    = gemm_bt x w n k (idx n i j).
Proof.
  intros.
  rewrite bt_offs_config_realizes_alpha_x_wt_plus_beta_c by assumption. ring.
Qed.

Theorem nn_offs_config_realizes_alpha_x_b_plus_beta_c :
  forall a b c0 al be m n k i j,
  (0 < n)%nat -> (i < m)%nat -> (j < n)%nat ->
  cublas_cm (nn_offs_cfg m n k al be) b a c0 (idx n i j)
    = al * gemm_nn a b n k (idx n i j) + be * c0 (idx n i j).
Proof.
  intros a b c0 al be m n k i j Hn Hi Hj.
  unfold cublas_cm, nn_offs_cfg, cm_op_a; simpl.
  rewrite idx_div, idx_mod by assumption.
  destruct (Nat.ltb_spec j n) as [_|]; [|lia].
  destruct (Nat.ltb_spec i m) as [_|]; [|lia].
  simpl.
  rewrite gemm_nn_at by assumption.
  f_equal. f_equal.
  apply fsum_ext_in. intros p _.
  replace (j + p * n)%nat with (idx n p j) by (unfold idx; lia).
  replace (p + i * k)%nat with (idx k i p) by (unfold idx; lia).
  ring.
Qed.

Theorem nn_offs_config_realizes_x_times_b :
  forall a b c0 m n k i j, (0 < n)%nat -> (i < m)%nat -> (j < n)%nat ->
  cublas_cm (nn_offs_cfg m n k 1 0) b a c0 (idx n i j)
    = gemm_nn a b n k (idx n i j).
Proof.
  intros.
  rewrite nn_offs_config_realizes_alpha_x_b_plus_beta_c by assumption. ring.
Qed.

Definition axpy (t : R) (d x : Buf) : Buf := fun c => x c + t * d c.

Definition addb (u v : Buf) : Buf := fun c => u c + v c.

Definition scaleb (s : R) (u : Buf) : Buf := fun c => s * u c.

Definition inner (sz : nat) (u v : Buf) : R := fsum sz (fun t => u t * v t).

Definition linearb (f : Buf -> Buf) : Prop :=
  forall t d x, f (axpy t d x) = axpy t (f d) (f x).

Definition adjointb (szi szo : nat) (f g : Buf -> Buf) : Prop :=
  forall u v, inner szi (g u) v = inner szo u (f v).

Definition has_grad (sz : nat) (L : Buf -> R) (x G : Buf) : Prop :=
  forall d, derivable_pt_lim (fun t => L (axpy t d x)) 0 (inner sz G d).

Lemma inner_flatten : forall m n u v,
  inner (m * n) u v
    = fsum m (fun i => fsum n (fun j => u (idx n i j) * v (idx n i j))).
Proof.
  intros. unfold inner. apply (fsum_flatten m n (fun t => u t * v t)).
Qed.

Lemma inner_scaleb_l : forall sz s u v,
  inner sz (scaleb s u) v = s * inner sz u v.
Proof.
  intros. unfold inner, scaleb. rewrite <- fsum_scal_l.
  apply fsum_ext_in; intros; ring.
Qed.

Lemma inner_scaleb_r : forall sz s u v,
  inner sz u (scaleb s v) = s * inner sz u v.
Proof.
  intros. unfold inner, scaleb. rewrite <- fsum_scal_l.
  apply fsum_ext_in; intros; ring.
Qed.

Lemma inner_addb_l : forall sz u u' v,
  inner sz (addb u u') v = inner sz u v + inner sz u' v.
Proof.
  intros. unfold inner, addb. rewrite <- fsum_plus.
  apply fsum_ext_in; intros; ring.
Qed.

Lemma inner_addb_r : forall sz u v v',
  inner sz u (addb v v') = inner sz u v + inner sz u v'.
Proof.
  intros. unfold inner, addb. rewrite <- fsum_plus.
  apply fsum_ext_in; intros; ring.
Qed.

Theorem gemm_bt_backward_is_adjoint : forall (w : Buf) m n k,
  (0 < n)%nat -> (0 < k)%nat ->
  adjointb (m * k) (m * n)
    (fun x => gemm_bt x w n k)
    (fun dy => gemm_nn dy w k n).
Proof.
  intros w m n k Hn Hk u v.
  rewrite !inner_flatten.
  transitivity
    (fsum m (fun i => fsum n (fun j =>
       u (idx n i j) * fsum k (fun p => v (idx k i p) * w (idx k j p))))).
  2:{ apply fsum_ext_in; intros i _. apply fsum_ext_in; intros j Hj.
      rewrite gemm_bt_at by assumption. reflexivity. }
  apply fsum_ext_in; intros i _.
  transitivity
    (fsum k (fun p =>
       fsum n (fun j => u (idx n i j) * w (idx k j p)) * v (idx k i p))).
  { apply fsum_ext_in; intros p Hp.
    rewrite gemm_nn_at by assumption. reflexivity. }
  transitivity
    (fsum k (fun p =>
       fsum n (fun j => u (idx n i j) * w (idx k j p) * v (idx k i p)))).
  { apply fsum_ext_in; intros p _. rewrite <- fsum_scal_r. reflexivity. }
  rewrite fsum_swap.
  apply fsum_ext_in; intros j _.
  rewrite <- fsum_scal_l.
  apply fsum_ext_in; intros p _. ring.
Qed.

Definition deltab (c : nat) : Buf := fun t => if Nat.eqb t c then 1 else 0.

Lemma inner_deltab : forall sz u c, (c < sz)%nat ->
  inner sz u (deltab c) = u c.
Proof.
  intros sz u c Hc. unfold inner.
  rewrite (fsum_single sz c).
  - unfold deltab. rewrite Nat.eqb_refl. cbn. ring.
  - assumption.
  - intros t Ht Htc. unfold deltab.
    destruct (Nat.eqb_spec t c) as [->|_]; [congruence| cbn; ring].
Qed.

Theorem backward_contract_is_the_unique_adjoint : forall (w : Buf) m n k,
  (0 < n)%nat -> (0 < k)%nat ->
  forall g, adjointb (m * k) (m * n) (fun x => gemm_bt x w n k) g ->
  forall u c, (c < m * k)%nat -> g u c = gemm_nn u w k n c.
Proof.
  intros w m n k Hn Hk g Hg u c Hc.
  rewrite <- (inner_deltab (m * k) (g u) c Hc).
  rewrite <- (inner_deltab (m * k) (gemm_nn u w k n) c Hc).
  rewrite Hg. rewrite (gemm_bt_backward_is_adjoint w m n k Hn Hk).
  reflexivity.
Qed.

Lemma gemm_bt_linear : forall w n k, linearb (fun x => gemm_bt x w n k).
Proof.
  intros w n k t d x. cbv beta.
  apply functional_extensionality; intro c.
  unfold gemm_bt, axpy. cbv beta.
  transitivity
    (fsum k (fun p => x (idx k (c / n) p) * w (idx k (c mod n) p)
                    + t * (d (idx k (c / n) p) * w (idx k (c mod n) p)))).
  { apply fsum_ext_in; intros; ring. }
  rewrite fsum_plus, fsum_scal_l. reflexivity.
Qed.

Lemma linearb_comp : forall f g,
  linearb f -> linearb g -> linearb (fun x => g (f x)).
Proof.
  intros f g Hf Hg t d x. cbv beta. rewrite Hf, Hg. reflexivity.
Qed.

Lemma linearb_scale : forall s f,
  linearb f -> linearb (fun x => scaleb s (f x)).
Proof.
  intros s f Hf t d x. cbv beta. rewrite Hf.
  apply functional_extensionality; intro c. unfold scaleb, axpy. ring.
Qed.

Lemma linearb_add : forall f g,
  linearb f -> linearb g -> linearb (fun x => addb (f x) (g x)).
Proof.
  intros f g Hf Hg t d x. cbv beta. rewrite Hf, Hg.
  apply functional_extensionality; intro c. unfold addb, axpy. ring.
Qed.

Lemma adjointb_comp : forall s1 s2 s3 f f' g g',
  adjointb s1 s2 f g -> adjointb s2 s3 f' g' ->
  adjointb s1 s3 (fun x => f' (f x)) (fun u => g (g' u)).
Proof.
  intros s1 s2 s3 f f' g g' H H' u v. cbv beta. rewrite H. apply H'.
Qed.

Lemma adjointb_scale : forall si so s f g,
  adjointb si so f g ->
  adjointb si so (fun x => scaleb s (f x)) (fun u => g (scaleb s u)).
Proof.
  intros si so s f g H u v. cbv beta.
  rewrite H, inner_scaleb_l, inner_scaleb_r. reflexivity.
Qed.

Lemma adjointb_add : forall si so f f' g g',
  adjointb si so f g -> adjointb si so f' g' ->
  adjointb si so (fun x => addb (f x) (f' x)) (fun u => addb (g u) (g' u)).
Proof.
  intros si so f f' g g' H H' u v. cbv beta.
  rewrite inner_addb_l, inner_addb_r, H, H'. reflexivity.
Qed.

Lemma grad_transport : forall (f g : Buf -> Buf) szi szo (L : Buf -> R) x G,
  linearb f ->
  adjointb szi szo f g ->
  has_grad szo L (f x) G ->
  has_grad szi (fun u => L (f u)) x (g G).
Proof.
  intros f g szi szo L x G Hlin Hadj HL d. cbv beta.
  replace (inner szi (g G) d) with (inner szo G (f d))
    by (symmetry; apply Hadj).
  assert (Heq : (fun t => L (f (axpy t d x)))
                  = (fun t => L (axpy t (f d) (f x)))).
  { apply functional_extensionality; intro t. rewrite Hlin. reflexivity. }
  rewrite Heq. apply HL.
Qed.

Theorem bf16linearop_backward_computes_the_gradient :
  forall (w : Buf) m n k (L : Buf -> R) x G,
  (0 < n)%nat -> (0 < k)%nat ->
  has_grad (m * n) L (gemm_bt x w n k) G ->
  has_grad (m * k) (fun u => L (gemm_bt u w n k)) x (gemm_nn G w k n).
Proof.
  intros w m n k L x G Hn Hk HL.
  apply (grad_transport (fun x => gemm_bt x w n k)
                        (fun dy => gemm_nn dy w k n)
                        (m * k) (m * n) L x G).
  - apply gemm_bt_linear.
  - apply gemm_bt_backward_is_adjoint; assumption.
  - exact HL.
Qed.

Definition wT : Buf := fun c => if Nat.eqb c 1 then 1 else 0.

Definition dyT : Buf := fun c => if Nat.eqb c 0 then 1 else 0.

Definition bwd_transposed_copy (dy : Buf) : Buf :=
  fun c => fsum 2 (fun j => dy (idx 2 (c / 2) j) * wT (idx 2 (c mod 2) j)).

Theorem transposed_copy_backward_differs :
  gemm_nn dyT wT 2 2 1%nat = 1 /\ bwd_transposed_copy dyT 1%nat = 0.
Proof.
  unfold gemm_nn, bwd_transposed_copy, dyT, wT, idx. cbn. split; lra.
Qed.

Theorem transposed_copy_backward_is_not_the_adjoint :
  ~ adjointb (1 * 2) (1 * 2) (fun x => gemm_bt x wT 2 2) bwd_transposed_copy.
Proof.
  intro Hadj.
  pose proof (backward_contract_is_the_unique_adjoint
                wT 1 2 2 ltac:(lia) ltac:(lia)
                bwd_transposed_copy Hadj dyT 1 ltac:(lia)) as Hu.
  destruct transposed_copy_backward_differs as [H1 H2].
  rewrite H1, H2 in Hu. lra.
Qed.

Definition lora_delta (a b : Buf) (s : R) (n k r : nat) (x : Buf) : Buf :=
  scaleb s (gemm_bt (gemm_bt x a r k) b n r).

Definition lora_bwd (a b : Buf) (s : R) (n k r : nat) (dy : Buf) : Buf :=
  gemm_nn (gemm_nn (scaleb s dy) b r n) a k r.

Lemma gemm_bt_scale_b : forall x b s n r c,
  gemm_bt x (scaleb s b) n r c = s * gemm_bt x b n r c.
Proof.
  intros. unfold gemm_bt, scaleb. rewrite <- fsum_scal_l.
  apply fsum_ext_in; intros; ring.
Qed.

Lemma gemm_bt_scale_x : forall x b s n r c,
  gemm_bt (scaleb s x) b n r c = s * gemm_bt x b n r c.
Proof.
  intros. unfold gemm_bt, scaleb. rewrite <- fsum_scal_l.
  apply fsum_ext_in; intros; ring.
Qed.

Theorem optimize_scaling_fold_preserves_delta :
  forall a b s n k r x c,
  lora_delta a b s n k r x c = lora_delta a (scaleb s b) 1 n k r x c.
Proof.
  intros. unfold lora_delta. unfold scaleb at 1 2.
  rewrite gemm_bt_scale_b. ring.
Qed.

Theorem scale_in_shrink_equals_scale_on_delta :
  forall a b s n k r x c,
  gemm_bt (scaleb s (gemm_bt x a r k)) b n r c = lora_delta a b s n k r x c.
Proof.
  intros. unfold lora_delta. unfold scaleb at 2.
  rewrite gemm_bt_scale_x. reflexivity.
Qed.

Theorem lora_backward_is_adjoint :
  forall a b s m n k r,
  (0 < n)%nat -> (0 < k)%nat -> (0 < r)%nat ->
  adjointb (m * k) (m * n)
    (fun x => lora_delta a b s n k r x)
    (fun dy => lora_bwd a b s n k r dy).
Proof.
  intros a b s m n k r Hn Hk Hr.
  unfold lora_delta, lora_bwd.
  apply (adjointb_scale (m * k) (m * n) s
           (fun x => gemm_bt (gemm_bt x a r k) b n r)
           (fun u => gemm_nn (gemm_nn u b r n) a k r)).
  apply (adjointb_comp (m * k) (m * r) (m * n)
           (fun x => gemm_bt x a r k)
           (fun h => gemm_bt h b n r)
           (fun u => gemm_nn u a k r)
           (fun u => gemm_nn u b r n)).
  - apply gemm_bt_backward_is_adjoint; assumption.
  - apply gemm_bt_backward_is_adjoint; assumption.
Qed.

Lemma lora_delta_linear : forall a b s n k r,
  linearb (fun x => lora_delta a b s n k r x).
Proof.
  intros. unfold lora_delta.
  apply (linearb_scale s (fun x => gemm_bt (gemm_bt x a r k) b n r)).
  apply (linearb_comp (fun x => gemm_bt x a r k)
                      (fun h => gemm_bt h b n r));
    apply gemm_bt_linear.
Qed.

Theorem lora_backward_computes_the_gradient :
  forall a b s m n k r (L : Buf -> R) x G,
  (0 < n)%nat -> (0 < k)%nat -> (0 < r)%nat ->
  has_grad (m * n) L (lora_delta a b s n k r x) G ->
  has_grad (m * k) (fun u => L (lora_delta a b s n k r u)) x
           (lora_bwd a b s n k r G).
Proof.
  intros a b s m n k r L x G Hn Hk Hr HL.
  apply (grad_transport (fun x => lora_delta a b s n k r x)
                        (fun dy => lora_bwd a b s n k r dy)
                        (m * k) (m * n) L x G).
  - apply lora_delta_linear.
  - apply lora_backward_is_adjoint; assumption.
  - exact HL.
Qed.

Definition linear_with_hook (w a b : Buf) (s : R) (n k r : nat) (x : Buf) : Buf :=
  addb (gemm_bt x w n k) (lora_delta a b s n k r x).

Definition linear_with_hook_bwd (w a b : Buf) (s : R) (n k r : nat) (dy : Buf) : Buf :=
  addb (gemm_nn dy w k n) (lora_bwd a b s n k r dy).

Theorem hooked_linear_backward_is_adjoint :
  forall w a b s m n k r,
  (0 < n)%nat -> (0 < k)%nat -> (0 < r)%nat ->
  adjointb (m * k) (m * n)
    (fun x => linear_with_hook w a b s n k r x)
    (fun dy => linear_with_hook_bwd w a b s n k r dy).
Proof.
  intros w a b s m n k r Hn Hk Hr.
  unfold linear_with_hook, linear_with_hook_bwd.
  apply (adjointb_add (m * k) (m * n)
           (fun x => gemm_bt x w n k)
           (fun x => lora_delta a b s n k r x)
           (fun dy => gemm_nn dy w k n)
           (fun dy => lora_bwd a b s n k r dy)).
  - apply gemm_bt_backward_is_adjoint; assumption.
  - apply lora_backward_is_adjoint; assumption.
Qed.

Theorem hooked_linear_backward_computes_the_gradient :
  forall w a b s m n k r (L : Buf -> R) x G,
  (0 < n)%nat -> (0 < k)%nat -> (0 < r)%nat ->
  has_grad (m * n) L (linear_with_hook w a b s n k r x) G ->
  has_grad (m * k) (fun u => L (linear_with_hook w a b s n k r u)) x
           (linear_with_hook_bwd w a b s n k r G).
Proof.
  intros w a b s m n k r L x G Hn Hk Hr HL.
  apply (grad_transport (fun x => linear_with_hook w a b s n k r x)
                        (fun dy => linear_with_hook_bwd w a b s n k r dy)
                        (m * k) (m * n) L x G).
  - unfold linear_with_hook.
    apply (linearb_add (fun x => gemm_bt x w n k)
                       (fun x => lora_delta a b s n k r x)).
    + apply gemm_bt_linear.
    + apply lora_delta_linear.
  - apply hooked_linear_backward_is_adjoint; assumption.
  - exact HL.
Qed.

Definition shiftb (o : nat) (u : Buf) : Buf := fun t => u (o + t)%nat.

Theorem window_narrow_is_column_restriction :
  forall b (xr : Buf) n r off rows i j,
  (0 < n)%nat -> (0 < rows)%nat ->
  (j < rows)%nat -> (off + rows <= n)%nat ->
  gemm_bt xr (shiftb (off * r) b) rows r (idx rows i j)
    = gemm_bt xr b n r (idx n i (off + j)).
Proof.
  intros b xr n r off rows i j Hn Hrows Hj Hwin.
  rewrite gemm_bt_at by assumption.
  rewrite gemm_bt_at by lia.
  apply fsum_ext_in; intros p _.
  unfold shiftb; f_equal; f_equal; unfold idx; nia.
Qed.
