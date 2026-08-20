
From Stdlib Require Import ZArith Lia List Reals Lra.
From SpeachesPlus Require Import Roofline.
Import ListNotations.
Open Scope Z_scope.

Definition nvfp4_weight_bytes_x16 (n k : Z) : Z := 9 * n * k.

Definition slice_traffic_x16 (n : Z) (slices : list Z) : Z :=
  fold_right (fun ki acc => nvfp4_weight_bytes_x16 n ki + acc) 0 slices.

Definition is_partition_of (k : Z) (slices : list Z) : Prop :=
  fold_right Z.add 0 slices = k.

Theorem streamk_weight_traffic_invariant :
  forall n k slices,
    is_partition_of k slices ->
    slice_traffic_x16 n slices = nvfp4_weight_bytes_x16 n k.
Proof.
  intros n k slices Hp.
  unfold is_partition_of in Hp.
  revert k Hp.
  induction slices as [|ki tl IH]; intros k Hp; simpl in *.
  - subst k. unfold nvfp4_weight_bytes_x16. ring.
  - rewrite (IH (fold_right Z.add 0 tl)) by reflexivity.
    unfold nvfp4_weight_bytes_x16. rewrite <- Hp. ring.
Qed.

Open Scope R_scope.

Theorem streamk_extra_traffic_preserves_floor :
  forall (BW L2 W : R) (f_dp f_sk : forward_exec),
    respects_bandwidth BW f_sk ->
    weight_traffic_lb L2 W f_dp ->
    (hbm_bytes f_dp <= hbm_bytes f_sk)%R ->
    (W - L2 <= wall_time f_sk * BW)%R.
Proof.
  intros BW L2 W f_dp f_sk Hbw Hlb Hextra.
  unfold respects_bandwidth, weight_traffic_lb in *. lra.
Qed.

Corollary streamk_obeys_verify_floor :
  forall (BW L2 W : R) (f_sk : forward_exec),
    respects_bandwidth BW f_sk ->
    weight_traffic_lb L2 W f_sk ->
    (W - L2 <= wall_time f_sk * BW)%R.
Proof. intros; apply verify_time_lower_bound; assumption. Qed.

Close Scope R_scope.

Definition tile : Z := 128.
Definition auto_threshold : Z := 192.

Definition tiles (x : Z) : Z := (x + tile - 1) / tile.

Definition streamk_auto (m n : Z) : bool :=
  (tiles m * tiles n <=? auto_threshold).

Lemma auto_routes_down     : streamk_auto 128 5376  = true.
Proof. vm_compute. reflexivity. Qed.
Lemma auto_keeps_gate_up_dp : streamk_auto 128 43008 = false.
Proof. vm_compute. reflexivity. Qed.

Lemma auto_monotone_n :
  forall n n', 0 < n -> n <= n' ->
    streamk_auto 128 n = false -> streamk_auto 128 n' = false.
Proof.
  intros n n' Hn Hle H.
  unfold streamk_auto, tiles, tile, auto_threshold in *.
  apply Z.leb_gt in H. apply Z.leb_gt.
  assert (Hd : (n + 128 - 1) / 128 <= (n' + 128 - 1) / 128)
    by (apply Z.div_le_mono; lia).
  vm_compute ((128 + 128 - 1) / 128) in *. lia.
Qed.
