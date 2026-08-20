From Stdlib Require Import ZArith Lia.
Open Scope Z_scope.

Definition max_grid_x : Z := 2147483647.
Definition max_grid_y : Z := 65535.
Definition max_grid_z : Z := 65535.

Record grid := { gx : Z; gy : Z; gz : Z }.

Definition launchable (g : grid) : Prop :=
  1 <= gx g <= max_grid_x /\
  1 <= gy g <= max_grid_y /\
  1 <= gz g <= max_grid_z.

Definition max_position_embeddings : Z := 262144.
Definition prefill_chunk : Z := 1024.

Definition dequant_fixed (len n_kv : Z) : grid :=
  {| gx := len; gy := n_kv; gz := 1 |}.

Definition dequant_old (len n_kv : Z) : grid :=
  {| gx := n_kv; gy := len; gz := 1 |}.

Theorem dequant_fixed_launchable :
  forall len n_kv,
    1 <= len <= max_position_embeddings ->
    1 <= n_kv <= max_grid_y ->
    launchable (dequant_fixed len n_kv).
Proof.
  intros len n_kv Hl Hk.
  unfold launchable, dequant_fixed, max_position_embeddings,
         max_grid_x, max_grid_y, max_grid_z in *.
  simpl. lia.
Qed.

Theorem dequant_old_unlaunchable_at_65536 :
  ~ launchable (dequant_old 65536 4).
Proof.
  unfold launchable, dequant_old, max_grid_x, max_grid_y, max_grid_z.
  simpl. lia.
Qed.

Theorem dequant_old_fails_inside_the_declared_context :
  exists len n_kv,
    1 <= len <= max_position_embeddings /\
    1 <= n_kv <= max_grid_y /\
    ~ launchable (dequant_old len n_kv).
Proof.
  exists 65536, 4.
  unfold launchable, dequant_old, max_position_embeddings,
         max_grid_x, max_grid_y, max_grid_z.
  simpl. repeat split; lia.
Qed.

Theorem chunk_boundary_witness :
  63 * prefill_chunk = 64512
  /\ 64 * prefill_chunk = 65536
  /\ 64512 <= max_grid_y
  /\ 65536 > max_grid_y.
Proof.
  unfold prefill_chunk, max_grid_y. repeat split; lia.
Qed.

Definition depthwise_grid (B C ztiles : Z) : grid :=
  {| gx := B; gy := C; gz := ztiles |}.

Theorem depthwise_launchable :
  forall B C ztiles,
    1 <= B <= max_grid_x ->
    1 <= C <= max_grid_y ->
    1 <= ztiles <= max_grid_z ->
    launchable (depthwise_grid B C ztiles).
Proof.
  intros B C z HB HC Hz.
  unfold launchable, depthwise_grid. simpl. lia.
Qed.

Theorem depthwise_z_safe_when_tiled :
  forall T tile,
    0 < tile ->
    T + tile - 1 <= max_grid_z * tile ->
    (T + tile - 1) / tile <= max_grid_z.
Proof.
  intros T tile Ht HT. unfold max_grid_z in *.
  apply Z.div_le_upper_bound; lia.
Qed.

Corollary depthwise_z_safe_at_tile_256 :
  forall T, 1 <= T <= 16776705 -> (T + 256 - 1) / 256 <= max_grid_z.
Proof.
  intros T HT. apply depthwise_z_safe_when_tiled; unfold max_grid_z; lia.
Qed.
