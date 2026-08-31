From Stdlib Require Import ZArith QArith Lia.
Open Scope Z_scope.

Definition block_size : Z := 16.

Definition qguard_inv (x : Q) : Q := if Qeq_dec x 0 then 1%Q else (/ x)%Q.

Lemma qguard_inv_zero : forall x : Q, (x == 0)%Q -> qguard_inv x = 1%Q.
Proof.
  intros x Hx; unfold qguard_inv.
  destruct (Qeq_dec x 0); [reflexivity | contradiction].
Qed.

Lemma qguard_inv_nonzero : forall x : Q, ~ (x == 0)%Q -> qguard_inv x = (/ x)%Q.
Proof.
  intros x Hx; unfold qguard_inv.
  destruct (Qeq_dec x 0); [contradiction | reflexivity].
Qed.

Definition pack_byte (n : Z -> Z) (i : Z) : Z := 16 * n (2 * i + 1) + n (2 * i).

Definition nib_lo (b : Z) : Z := b mod 16.
Definition nib_hi (b : Z) : Z := (b / 16) mod 16.

Definition nib_at (bytes : Z -> Z) (j : Z) : Z :=
  if Z.even j then nib_lo (bytes (j / 2)) else nib_hi (bytes (j / 2)).

Lemma pack_byte_lo : forall n i,
  (forall t, 0 <= n t < 16) -> nib_lo (pack_byte n i) = n (2 * i).
Proof.
  intros n i H; unfold nib_lo, pack_byte.
  replace (16 * n (2 * i + 1) + n (2 * i)) with (n (2 * i) + n (2 * i + 1) * 16)
    by ring.
  rewrite Z.mod_add by lia.
  apply Z.mod_small; auto.
Qed.

Lemma pack_byte_hi : forall n i,
  (forall t, 0 <= n t < 16) -> nib_hi (pack_byte n i) = n (2 * i + 1).
Proof.
  intros n i H; unfold nib_hi, pack_byte.
  replace (16 * n (2 * i + 1) + n (2 * i)) with (n (2 * i + 1) * 16 + n (2 * i))
    by ring.
  rewrite Z.div_add_l by lia.
  rewrite (Z.div_small (n (2 * i))) by auto.
  rewrite Z.add_0_r; apply Z.mod_small; auto.
Qed.

Theorem nibble_unpack_pack : forall n j, 0 <= j ->
  (forall t, 0 <= n t < 16) -> nib_at (pack_byte n) j = n j.
Proof.
  intros n j Hj H; unfold nib_at.
  destruct (Z.even j) eqn:E.
  - rewrite pack_byte_lo by auto.
    apply Z.even_spec in E; destruct E as [k Hk].
    assert (Hd : j / 2 = k)
      by (subst; rewrite (Z.mul_comm 2 k); rewrite Z.div_mul by lia;
          reflexivity).
    rewrite Hd; f_equal; lia.
  - rewrite pack_byte_hi by auto.
    assert (O : Z.odd j = true) by (rewrite <- Z.negb_even, E; reflexivity).
    apply Z.odd_spec in O; destruct O as [k Hk].
    assert (Hd : j / 2 = k)
      by (subst; replace (2 * k + 1) with (k * 2 + 1) by ring;
          rewrite Z.div_add_l by lia; change (1 / 2) with 0; lia).
    rewrite Hd; f_equal; lia.
Qed.

Theorem byte_pack_unpack : forall b, 0 <= b < 256 ->
  16 * nib_hi b + nib_lo b = b.
Proof.
  intros b Hb; unfold nib_hi, nib_lo.
  rewrite (Z.mod_small (b / 16))
    by (split; [apply Z.div_pos | apply Z.div_lt_upper_bound]; lia).
  pose proof (Z.div_mod b 16 ltac:(lia)); lia.
Qed.

Theorem nibble_range : forall B p, 0 <= B < 256 -> 0 <= p < 2 ->
  0 <= (if (p =? 0)%Z then nib_lo B else nib_hi B) < 16.
Proof.
  intros B p HB Hp; destruct (p =? 0)%Z; unfold nib_lo, nib_hi;
    apply Z.mod_pos_bound; lia.
Qed.

Section ScaleSwizzle.

Variable rows kblocks : Z.
Hypothesis Hrows : 0 < rows.
Hypothesis Hkb : 0 < kblocks.

Definition m_tiles : Z := (rows + 127) / 128.
Definition k_tiles : Z := (kblocks + 3) / 4.

Definition sw_dst (m k : Z) : Z :=
  ((m / 128 * k_tiles + k / 4) * 32 + m mod 32) * 16 + (m / 32) mod 4 * 4
  + k mod 4.

Definition sw_m (dst : Z) : Z :=
  dst / 512 / k_tiles * 128 + (dst mod 16) / 4 * 32 + (dst / 16) mod 32.

Definition sw_k (dst : Z) : Z := (dst / 512) mod k_tiles * 4 + dst mod 4.

Lemma tile_bound : forall a n d, 0 < d -> 0 <= a < n -> a / d < (n + d - 1) / d.
Proof.
  intros a n d Hd Ha.
  assert (E : (n + d - 1) / d = (n - 1) / d + 1).
  { replace (n + d - 1) with ((n - 1) + 1 * d) by ring.
    rewrite Z.div_add by lia; lia. }
  assert (a / d <= (n - 1) / d) by (apply Z.div_le_mono; lia).
  lia.
Qed.

Lemma m_tile_bound : forall m, 0 <= m < rows -> m / 128 < m_tiles.
Proof.
  intros m Hm; unfold m_tiles.
  replace (rows + 127) with (rows + 128 - 1) by ring.
  apply tile_bound; lia.
Qed.

Lemma k_tile_bound : forall k, 0 <= k < kblocks -> k / 4 < k_tiles.
Proof.
  intros k Hk; unfold k_tiles.
  replace (kblocks + 3) with (kblocks + 4 - 1) by ring.
  apply tile_bound; lia.
Qed.

Lemma k_tiles_pos : 0 < k_tiles.
Proof.
  unfold k_tiles.
  assert (4 / 4 <= (kblocks + 3) / 4) by (apply Z.div_le_mono; lia).
  change (4 / 4) with 1 in H; lia.
Qed.

Lemma m_tiles_pos : 0 < m_tiles.
Proof.
  unfold m_tiles.
  assert (128 / 128 <= (rows + 127) / 128) by (apply Z.div_le_mono; lia).
  change (128 / 128) with 1 in H; lia.
Qed.

Lemma m_decomp : forall m, 0 <= m ->
  m = 128 * (m / 128) + 32 * ((m / 32) mod 4) + m mod 32.
Proof.
  intros m Hm.
  pose proof (Z.div_mod m 32 ltac:(lia)).
  pose proof (Z.div_mod (m / 32) 4 ltac:(lia)).
  rewrite (Z.div_div m 32 4) in * by lia.
  change (32 * 4) with 128 in *; lia.
Qed.

Lemma sw_dst_radix : forall m k,
  sw_dst m k = 512 * (m / 128 * k_tiles + k / 4)
               + 16 * (m mod 32) + 4 * ((m / 32) mod 4) + k mod 4.
Proof. intros; unfold sw_dst; ring. Qed.

Lemma radix_digits : forall t c3 c2 c1,
  0 <= t -> 0 <= c3 < 32 -> 0 <= c2 < 4 -> 0 <= c1 < 4 ->
  (512 * t + 16 * c3 + 4 * c2 + c1) / 512 = t
  /\ ((512 * t + 16 * c3 + 4 * c2 + c1) / 16) mod 32 = c3
  /\ ((512 * t + 16 * c3 + 4 * c2 + c1) mod 16) / 4 = c2
  /\ (512 * t + 16 * c3 + 4 * c2 + c1) mod 4 = c1.
Proof.
  intros t c3 c2 c1 Ht H3 H2 H1.
  split.
  { replace (512 * t + 16 * c3 + 4 * c2 + c1)
      with ((16 * c3 + 4 * c2 + c1) + t * 512) by ring.
    rewrite Z.div_add by lia; rewrite Z.div_small by lia; lia. }
  split.
  { replace (512 * t + 16 * c3 + 4 * c2 + c1)
      with ((32 * t + c3) * 16 + (4 * c2 + c1)) by ring.
    rewrite Z.div_add_l by lia.
    rewrite (Z.div_small (4 * c2 + c1)) by lia.
    rewrite Z.add_0_r.
    replace (32 * t + c3) with (c3 + t * 32) by ring.
    rewrite Z.mod_add by lia; apply Z.mod_small; lia. }
  split.
  { replace (512 * t + 16 * c3 + 4 * c2 + c1)
      with ((4 * c2 + c1) + (32 * t + c3) * 16) by ring.
    rewrite Z.mod_add by lia.
    rewrite (Z.mod_small (4 * c2 + c1)) by lia.
    replace (4 * c2 + c1) with (c2 * 4 + c1) by ring.
    rewrite Z.div_add_l by lia; rewrite Z.div_small by lia; lia. }
  { replace (512 * t + 16 * c3 + 4 * c2 + c1)
      with (c1 + (128 * t + 4 * c3 + c2) * 4) by ring.
    rewrite Z.mod_add by lia; apply Z.mod_small; lia. }
Qed.

Theorem sw_roundtrip : forall m k, 0 <= m < rows -> 0 <= k < kblocks ->
  sw_m (sw_dst m k) = m /\ sw_k (sw_dst m k) = k.
Proof.
  intros m k Hm Hk.
  pose proof k_tiles_pos as Hktp.
  pose proof (m_decomp m ltac:(lia)) as Em.
  pose proof (Z.div_mod k 4 ltac:(lia)) as Ek.
  pose proof (Z.mod_pos_bound m 32 ltac:(lia)).
  pose proof (Z.mod_pos_bound (m / 32) 4 ltac:(lia)).
  pose proof (Z.mod_pos_bound k 4 ltac:(lia)).
  assert (Hkt : k / 4 < k_tiles) by (apply k_tile_bound; lia).
  assert (Hk4 : 0 <= k / 4) by (apply Z.div_pos; lia).
  assert (Hm128 : 0 <= m / 128) by (apply Z.div_pos; lia).
  assert (Ht : 0 <= m / 128 * k_tiles + k / 4) by nia.
  rewrite sw_dst_radix.
  destruct (radix_digits (m / 128 * k_tiles + k / 4) (m mod 32)
              ((m / 32) mod 4) (k mod 4) Ht ltac:(lia) ltac:(lia) ltac:(lia))
    as (R512 & R3 & R2 & R1).
  unfold sw_m, sw_k.
  rewrite R512, R3, R2, R1.
  assert (Dq : (m / 128 * k_tiles + k / 4) / k_tiles = m / 128)
    by (rewrite Z.div_add_l by lia;
        rewrite (Z.div_small (k / 4) k_tiles) by lia; lia).
  assert (Dm : (m / 128 * k_tiles + k / 4) mod k_tiles = k / 4)
    by (replace (m / 128 * k_tiles + k / 4) with (k / 4 + m / 128 * k_tiles)
          by ring;
        rewrite Z.mod_add by lia; apply Z.mod_small; lia).
  rewrite Dq, Dm; lia.
Qed.

Theorem sw_dst_range : forall m k, 0 <= m < rows -> 0 <= k < kblocks ->
  0 <= sw_dst m k < m_tiles * 128 * (k_tiles * 4).
Proof.
  intros m k Hm Hk.
  pose proof k_tiles_pos; pose proof m_tiles_pos.
  assert (Hmt : m / 128 < m_tiles) by (apply m_tile_bound; lia).
  assert (Hkt : k / 4 < k_tiles) by (apply k_tile_bound; lia).
  assert (0 <= m / 128) by (apply Z.div_pos; lia).
  assert (0 <= k / 4) by (apply Z.div_pos; lia).
  pose proof (Z.mod_pos_bound m 32 ltac:(lia)).
  pose proof (Z.mod_pos_bound (m / 32) 4 ltac:(lia)).
  pose proof (Z.mod_pos_bound k 4 ltac:(lia)).
  rewrite sw_dst_radix.
  assert (m / 128 * k_tiles + k / 4 <= m_tiles * k_tiles - 1) by nia.
  nia.
Qed.

Theorem sw_dst_injective : forall m k m' k',
  0 <= m < rows -> 0 <= k < kblocks -> 0 <= m' < rows -> 0 <= k' < kblocks ->
  sw_dst m k = sw_dst m' k' -> m = m' /\ k = k'.
Proof.
  intros m k m' k' Hm Hk Hm' Hk' E.
  destruct (sw_roundtrip m k Hm Hk) as (A & B).
  destruct (sw_roundtrip m' k' Hm' Hk') as (A' & B').
  rewrite E in A, B; split; congruence.
Qed.

Theorem sw_dst_complete : forall dst,
  0 <= dst < m_tiles * 128 * (k_tiles * 4) ->
  sw_dst (sw_m dst) (sw_k dst) = dst.
Proof.
  intros dst Hdst.
  pose proof k_tiles_pos; pose proof m_tiles_pos.
  assert (Ht : 0 <= dst / 512 < m_tiles * k_tiles)
    by (split; [apply Z.div_pos; lia | apply Z.div_lt_upper_bound; nia]).
  assert (Hc3 : 0 <= (dst / 16) mod 32 < 32) by (apply Z.mod_pos_bound; lia).
  assert (Hc2 : 0 <= (dst mod 16) / 4 < 4)
    by (pose proof (Z.mod_pos_bound dst 16 ltac:(lia));
        split; [apply Z.div_pos; lia | apply Z.div_lt_upper_bound; lia]).
  assert (Hc1 : 0 <= dst mod 4 < 4) by (apply Z.mod_pos_bound; lia).
  assert (Hrec : dst = 512 * (dst / 512) + 16 * ((dst / 16) mod 32)
                       + 4 * ((dst mod 16) / 4) + dst mod 4).
  { pose proof (Z.div_mod dst 16 ltac:(lia)).
    pose proof (Z.div_mod (dst / 16) 32 ltac:(lia)).
    rewrite (Z.div_div dst 16 32) in * by lia.
    change (16 * 32) with 512 in *.
    pose proof (Z.div_mod (dst mod 16) 4 ltac:(lia)).
    assert ((dst mod 16) mod 4 = dst mod 4)
      by (apply Z.mod_mod_divide; exists 4; reflexivity).
    lia. }
  assert (HAB : dst / 512 = dst / 512 / k_tiles * k_tiles
                            + (dst / 512) mod k_tiles)
    by (pose proof (Z.div_mod (dst / 512) k_tiles ltac:(lia)); lia).
  assert (HB : 0 <= (dst / 512) mod k_tiles < k_tiles)
    by (apply Z.mod_pos_bound; lia).
  assert (HA : 0 <= dst / 512 / k_tiles)
    by (apply Z.div_pos; [apply Z.div_pos |]; lia).
  unfold sw_m, sw_k.
  set (A := dst / 512 / k_tiles) in *.
  set (B := (dst / 512) mod k_tiles) in *.
  set (c3 := (dst / 16) mod 32) in *.
  set (c2 := (dst mod 16) / 4) in *.
  set (c1 := dst mod 4) in *.
  assert (Mdiv : (A * 128 + c2 * 32 + c3) / 128 = A).
  { replace (A * 128 + c2 * 32 + c3) with ((c2 * 32 + c3) + A * 128) by ring.
    rewrite Z.div_add by lia; rewrite Z.div_small by lia; lia. }
  assert (Mmod32 : (A * 128 + c2 * 32 + c3) mod 32 = c3).
  { replace (A * 128 + c2 * 32 + c3) with (c3 + (A * 4 + c2) * 32) by ring.
    rewrite Z.mod_add by lia; apply Z.mod_small; lia. }
  assert (Mdiv32 : ((A * 128 + c2 * 32 + c3) / 32) mod 4 = c2).
  { replace (A * 128 + c2 * 32 + c3) with ((A * 4 + c2) * 32 + c3) by ring.
    rewrite Z.div_add_l by lia.
    rewrite (Z.div_small c3) by lia.
    rewrite Z.add_0_r.
    replace (A * 4 + c2) with (c2 + A * 4) by ring.
    rewrite Z.mod_add by lia; apply Z.mod_small; lia. }
  assert (Kdiv : (B * 4 + c1) / 4 = B)
    by (rewrite Z.div_add_l by lia; rewrite (Z.div_small c1 4) by lia; lia).
  assert (Kmod : (B * 4 + c1) mod 4 = c1)
    by (replace (B * 4 + c1) with (c1 + B * 4) by ring;
        rewrite Z.mod_add by lia; apply Z.mod_small; lia).
  unfold sw_dst.
  rewrite Mdiv, Mmod32, Mdiv32, Kdiv, Kmod.
  rewrite <- HAB in *; lia.
Qed.

Definition swizzled (lin : Z -> Z) (dst : Z) : Z :=
  if (sw_m dst <? rows) && (sw_k dst <? kblocks)
  then lin (sw_m dst * kblocks + sw_k dst)
  else 0.

Definition unswizzled (s : Z -> Z) (m k : Z) : Z := s (sw_dst m k).

Theorem unswizzle_swizzle_id : forall lin m k,
  0 <= m < rows -> 0 <= k < kblocks ->
  unswizzled (swizzled lin) m k = lin (m * kblocks + k).
Proof.
  intros lin m k Hm Hk; unfold unswizzled, swizzled.
  destruct (sw_roundtrip m k Hm Hk) as (Em & Ek).
  rewrite Em, Ek.
  replace (m <? rows) with true by (symmetry; apply Z.ltb_lt; lia).
  replace (k <? kblocks) with true by (symmetry; apply Z.ltb_lt; lia).
  reflexivity.
Qed.

Theorem unswizzle_swizzle_flat_id : forall lin idx,
  0 <= idx < rows * kblocks ->
  unswizzled (swizzled lin) (idx / kblocks) (idx mod kblocks) = lin idx.
Proof.
  intros lin idx Hidx.
  assert (0 <= idx / kblocks < rows)
    by (split; [apply Z.div_pos; lia | apply Z.div_lt_upper_bound; lia]).
  pose proof (Z.mod_pos_bound idx kblocks ltac:(lia)).
  rewrite unswizzle_swizzle_id by lia.
  f_equal; pose proof (Z.div_mod idx kblocks ltac:(lia)); lia.
Qed.

Theorem swizzle_pad_reads_zero : forall lin dst,
  rows <= sw_m dst \/ kblocks <= sw_k dst -> swizzled lin dst = 0.
Proof.
  intros lin dst [H | H]; unfold swizzled.
  - replace (sw_m dst <? rows) with false
      by (symmetry; apply Z.ltb_ge; lia).
    reflexivity.
  - replace (sw_k dst <? kblocks) with false
      by (symmetry; apply Z.ltb_ge; lia).
    rewrite Bool.andb_false_r; reflexivity.
Qed.

End ScaleSwizzle.

Section RowMajorLayout.

Variable q : Z.
Hypothesis Hq : 0 < q.

Definition cols : Z := 16 * q.
Definition bytes_per_row : Z := cols / 2.
Definition blocks_per_row : Z := cols / 16.

Lemma bytes_per_row_val : bytes_per_row = 8 * q.
Proof.
  unfold bytes_per_row, cols.
  replace (16 * q) with (8 * q * 2) by ring; apply Z.div_mul; lia.
Qed.

Lemma blocks_per_row_val : blocks_per_row = q.
Proof.
  unfold blocks_per_row, cols; rewrite Z.mul_comm; apply Z.div_mul; lia.
Qed.

Lemma loop_writer_unique : forall j b i p,
  0 <= b -> 0 <= i < 8 -> 0 <= p < 2 ->
  b * 16 + i * 2 + p = j ->
  b = j / 16 /\ i = (j mod 16) / 2 /\ p = j mod 2.
Proof.
  intros j b i p Hb Hi Hp Hj.
  assert (Hb' : b = j / 16)
    by (apply (Z.div_unique_pos j 16 b (i * 2 + p)); lia).
  assert (Hm : i * 2 + p = j mod 16)
    by (apply (Z.mod_unique_pos j 16 b (i * 2 + p)); lia).
  split; [exact Hb' |].
  split.
  - apply (Z.div_unique_pos (j mod 16) 2 i p); lia.
  - apply (Z.mod_unique_pos j 2 (8 * b + i) p); lia.
Qed.

Lemma loop_covers : forall j, 0 <= j < cols ->
  0 <= j / 16 < blocks_per_row /\ 0 <= (j mod 16) / 2 < 8 /\
  0 <= j mod 2 < 2 /\
  (j / 16) * 16 + ((j mod 16) / 2) * 2 + j mod 2 = j.
Proof.
  intros j Hj.
  rewrite blocks_per_row_val.
  pose proof (Z.div_mod j 16 ltac:(lia)).
  pose proof (Z.mod_pos_bound j 16 ltac:(lia)).
  pose proof (Z.div_mod (j mod 16) 2 ltac:(lia)).
  pose proof (Z.mod_pos_bound (j mod 16) 2 ltac:(lia)).
  assert ((j mod 16) mod 2 = j mod 2)
    by (apply Z.mod_mod_divide; exists 8; reflexivity).
  assert (j / 16 < q)
    by (apply Z.div_lt_upper_bound; unfold cols in Hj; lia).
  assert (0 <= j / 16) by (apply Z.div_pos; lia).
  assert (0 <= (j mod 16) / 2 < 8)
    by (split; [apply Z.div_pos; lia | apply Z.div_lt_upper_bound; lia]).
  lia.
Qed.

Lemma sub_block_byte_flat : forall j, 0 <= j < cols ->
  j / 16 * 8 + (j mod 16) / 2 = j / 2.
Proof.
  intros j Hj.
  pose proof (Z.div_mod j 16 ltac:(lia)).
  pose proof (Z.mod_pos_bound j 16 ltac:(lia)).
  apply (Z.div_unique_pos j 2 (j / 16 * 8 + (j mod 16) / 2) (j mod 2)).
  - pose proof (Z.mod_pos_bound j 2 ltac:(lia)); lia.
  - pose proof (Z.div_mod (j mod 16) 2 ltac:(lia)).
    assert ((j mod 16) mod 2 = j mod 2)
      by (apply Z.mod_mod_divide; exists 8; reflexivity).
    lia.
Qed.

Lemma data_byte_flat : forall r j, 0 <= r -> 0 <= j < cols ->
  r * bytes_per_row + j / 16 * 8 + (j mod 16) / 2 = (r * cols + j) / 2.
Proof.
  intros r j Hr Hj.
  rewrite bytes_per_row_val.
  rewrite <- Z.add_assoc, sub_block_byte_flat by auto.
  unfold cols.
  replace (r * (16 * q) + j) with (j + r * (8 * q) * 2) by ring.
  rewrite Z.div_add by lia; ring.
Qed.

Lemma scale_index_flat : forall r j, 0 <= r -> 0 <= j < cols ->
  r * blocks_per_row + j / 16 = (r * cols + j) / 16.
Proof.
  intros r j Hr Hj.
  rewrite blocks_per_row_val; unfold cols.
  replace (r * (16 * q) + j) with (j + r * q * 16) by ring.
  rewrite Z.div_add by lia; ring.
Qed.

Lemma flat_row_split : forall r j, 0 <= r -> 0 <= j < cols ->
  (r * cols + j) / cols = r /\ (r * cols + j) mod cols = j.
Proof.
  intros r j Hr Hj.
  assert (0 < cols) by (unfold cols; lia).
  split.
  - symmetry; apply (Z.div_unique_pos (r * cols + j) cols r j); lia.
  - symmetry; apply (Z.mod_unique_pos (r * cols + j) cols r j); lia.
Qed.

Definition elem_to_bytepos (r j : Z) : Z * Z :=
  ((r * cols + j) / 2, (r * cols + j) mod 2).

Definition bytepos_to_elem (B p : Z) : Z * Z :=
  ((2 * B + p) / cols, (2 * B + p) mod cols).

Theorem elem_bytepos_inverse : forall r j, 0 <= r -> 0 <= j < cols ->
  bytepos_to_elem (fst (elem_to_bytepos r j)) (snd (elem_to_bytepos r j))
  = (r, j).
Proof.
  intros r j Hr Hj; unfold bytepos_to_elem, elem_to_bytepos; cbn [fst snd].
  assert (Ho : 2 * ((r * cols + j) / 2) + (r * cols + j) mod 2 = r * cols + j)
    by (pose proof (Z.div_mod (r * cols + j) 2 ltac:(lia)); lia).
  rewrite Ho.
  destruct (flat_row_split r j Hr Hj) as (E1 & E2).
  rewrite E1, E2; reflexivity.
Qed.

Theorem bytepos_elem_inverse : forall B p, 0 <= B -> 0 <= p < 2 ->
  elem_to_bytepos (fst (bytepos_to_elem B p)) (snd (bytepos_to_elem B p))
  = (B, p).
Proof.
  intros B p HB Hp; unfold bytepos_to_elem, elem_to_bytepos; cbn [fst snd].
  assert (Hc : 0 < cols) by (unfold cols; lia).
  assert (Ho : (2 * B + p) / cols * cols + (2 * B + p) mod cols = 2 * B + p)
    by (pose proof (Z.div_mod (2 * B + p) cols ltac:(lia)); lia).
  rewrite Ho.
  assert (D : (2 * B + p) / 2 = B)
    by (replace (2 * B + p) with (p + B * 2) by ring;
        rewrite Z.div_add by lia; rewrite Z.div_small by lia; lia).
  assert (M : (2 * B + p) mod 2 = p)
    by (replace (2 * B + p) with (p + B * 2) by ring;
        rewrite Z.mod_add by lia; apply Z.mod_small; lia).
  rewrite D, M; reflexivity.
Qed.

Section ServeDequant.

Variables dE dS : Z -> Q.
Variables packed scales : Z -> Z.
Variable wm : Q.

Definition serve_loop_val (r b i p : Z) : Q :=
  ((if (p =? 0)%Z
    then dE (nib_lo (packed (r * bytes_per_row + b * 8 + i)))
    else dE (nib_hi (packed (r * bytes_per_row + b * 8 + i))))
   * (dS (scales (r * blocks_per_row + b)) * wm))%Q.

Definition serve_at (r j : Z) : Q :=
  serve_loop_val r (j / 16) ((j mod 16) / 2) (j mod 2).

Lemma nib_at_parity : forall bytes j, 0 <= j ->
  nib_at bytes j
  = (if (j mod 2 =? 0)%Z then nib_lo (bytes (j / 2))
     else nib_hi (bytes (j / 2))).
Proof.
  intros bytes j Hj; unfold nib_at.
  destruct (Z.even j) eqn:E.
  - replace (j mod 2 =? 0)%Z with true; [reflexivity |].
    symmetry; apply Z.eqb_eq.
    apply Z.even_spec in E; destruct E as [k Hk]; subst.
    replace (2 * k) with (0 + k * 2) by ring.
    rewrite Z.mod_add by lia; reflexivity.
  - replace (j mod 2 =? 0)%Z with false; [reflexivity |].
    symmetry; apply Z.eqb_neq.
    assert (O : Z.odd j = true) by (rewrite <- Z.negb_even, E; reflexivity).
    apply Z.odd_spec in O; destruct O as [k Hk]; subst.
    replace (2 * k + 1) with (1 + k * 2) by ring.
    rewrite Z.mod_add by lia; discriminate.
Qed.

Theorem serve_own_block_scale : forall r j, 0 <= r -> 0 <= j < cols ->
  (serve_at r j ==
     dE (nib_at (fun t => packed (r * bytes_per_row + t)) j)
     * (dS (scales ((r * cols + j) / 16)) * wm))%Q.
Proof.
  intros r j Hr Hj; unfold serve_at, serve_loop_val.
  rewrite nib_at_parity by lia.
  cbn beta.
  rewrite scale_index_flat by auto.
  replace (r * bytes_per_row + j / 16 * 8 + (j mod 16) / 2)
    with (r * bytes_per_row + (j / 16 * 8 + (j mod 16) / 2)) by ring.
  rewrite sub_block_byte_flat by auto.
  destruct (j mod 2 =? 0)%Z; reflexivity.
Qed.

End ServeDequant.

Section SmallMatrixServe.

Variables dE3 dS3 : Z -> Q.
Variables packed3 scales3 : Z -> Z.
Variable g3 : Q.

Definition small_serve_at (r j : Z) : Q :=
  ((if (j mod 2 =? 0)%Z
    then dE3 (nib_lo (packed3
      (r * bytes_per_row + j / 16 * 8 + (j mod 16) / 2)))
    else dE3 (nib_hi (packed3
      (r * bytes_per_row + j / 16 * 8 + (j mod 16) / 2))))
   * (dS3 (scales3 (r * blocks_per_row + j / 16)) * qguard_inv g3))%Q.

Theorem small_serve_agrees_serve : forall r j,
  (small_serve_at r j
   == serve_at dE3 dS3 packed3 scales3 (qguard_inv g3) r j)%Q.
Proof.
  intros r j; unfold small_serve_at, serve_at, serve_loop_val.
  destruct (j mod 2 =? 0)%Z; reflexivity.
Qed.

End SmallMatrixServe.

Section TensorScaleGrouping.

Variables dE4 dS4 : Z -> Q.
Variables packed4 scales4 : Z -> Z.
Variable alpha : Q.

Definition block_then_tensor_at (r j : Z) : Q :=
  (((if (j mod 2 =? 0)%Z
     then dE4 (nib_lo (packed4
       (r * bytes_per_row + j / 16 * 8 + (j mod 16) / 2)))
     else dE4 (nib_hi (packed4
       (r * bytes_per_row + j / 16 * 8 + (j mod 16) / 2))))
    * dS4 (scales4 (r * blocks_per_row + j / 16))) * alpha)%Q.

Theorem grouping_immaterial_exact : forall r j,
  (block_then_tensor_at r j
   == serve_at dE4 dS4 packed4 scales4 alpha r j)%Q.
Proof.
  intros r j; unfold block_then_tensor_at, serve_at, serve_loop_val.
  destruct (j mod 2 =? 0)%Z; ring.
Qed.

End TensorScaleGrouping.

Section TrainHostDequant.

Variables dE1 dS1 : Z -> Q.
Variables packed1 scales1 : Z -> Z.
Variable rows blk : Z.
Hypothesis Hblk : 0 < blk.
Variable g : Q.

Lemma par_row_blocks_guarded_block_positive : forall rs ts,
  0 < rs -> 0 < ts -> 0 < (rs + ts - 1) / ts.
Proof.
  intros rs ts Hrs Hts; apply Z.div_str_pos; lia.
Qed.

Definition train_chunk_start (bi : Z) : Z := bi * blk.
Definition train_chunk_count (bi : Z) : Z :=
  Z.min blk (rows - train_chunk_start bi).

Definition train_host_at (r j : Z) : Q :=
  ((if (j mod 2 =? 0)%Z
    then dE1 (nib_lo (packed1
      (train_chunk_start (r / blk) * cols / 2
       + ((r - train_chunk_start (r / blk)) * (cols / 2)
          + j / 16 * 8 + (j mod 16) / 2))))
    else dE1 (nib_hi (packed1
      (train_chunk_start (r / blk) * cols / 2
       + ((r - train_chunk_start (r / blk)) * (cols / 2)
          + j / 16 * 8 + (j mod 16) / 2)))))
   * (dS1 (scales1
        (train_chunk_start (r / blk) * (cols / 16)
         + ((r - train_chunk_start (r / blk)) * (cols / 16) + j / 16)))
      * qguard_inv g))%Q.

Theorem chunk_covers : forall r, 0 <= r < rows ->
  train_chunk_start (r / blk) <= r
  < train_chunk_start (r / blk) + train_chunk_count (r / blk).
Proof.
  intros r Hr; unfold train_chunk_count, train_chunk_start.
  pose proof (Z.div_mod r blk ltac:(lia)).
  pose proof (Z.mod_pos_bound r blk ltac:(lia)).
  destruct (Z.min_spec blk (rows - r / blk * blk)) as [(A & E) | (A & E)];
    rewrite E; nia.
Qed.

Theorem chunk_exists : forall r, 0 <= r < rows ->
  r / blk * (blk * cols) < rows * cols.
Proof.
  intros r Hr.
  assert (0 < cols) by (unfold cols; lia).
  assert (0 <= r / blk) by (apply Z.div_pos; lia).
  pose proof (Z.div_mod r blk ltac:(lia)).
  pose proof (Z.mod_pos_bound r blk ltac:(lia)).
  nia.
Qed.

Theorem chunk_len_matches : forall bi,
  Z.min (blk * cols) (rows * cols - train_chunk_start bi * cols)
  = train_chunk_count bi * cols.
Proof.
  intros bi; unfold train_chunk_count, train_chunk_start.
  assert (0 <= cols) by (unfold cols; lia).
  replace (rows * cols - bi * blk * cols) with ((rows - bi * blk) * cols)
    by ring.
  rewrite Z.mul_min_distr_nonneg_r by lia; reflexivity.
Qed.

Theorem train_host_agrees_serve : forall r j, 0 <= r < rows -> 0 <= j < cols ->
  (train_host_at r j
   == serve_at dE1 dS1 packed1 scales1 (qguard_inv g) r j)%Q.
Proof.
  intros r j Hr Hj; unfold train_host_at, serve_at, serve_loop_val.
  set (start := train_chunk_start (r / blk)).
  assert (Hcols2 : cols / 2 = bytes_per_row) by reflexivity.
  assert (Hcols16 : cols / 16 = blocks_per_row) by reflexivity.
  assert (Hhalf : start * cols / 2 = start * (cols / 2)).
  { rewrite Hcols2, bytes_per_row_val.
    unfold cols.
    replace (start * (16 * q)) with (start * (8 * q) * 2) by ring.
    rewrite Z.div_mul by lia; ring. }
  assert (Ebyte : start * cols / 2
                  + ((r - start) * (cols / 2) + j / 16 * 8 + (j mod 16) / 2)
                  = r * bytes_per_row + j / 16 * 8 + (j mod 16) / 2).
  { rewrite Hhalf, Hcols2; ring. }
  assert (Escale : start * (cols / 16) + ((r - start) * (cols / 16) + j / 16)
                   = r * blocks_per_row + j / 16).
  { rewrite Hcols16; ring. }
  rewrite Ebyte, Escale.
  destruct (j mod 2 =? 0)%Z; reflexivity.
Qed.

End TrainHostDequant.

Section TrainRunnerDequant.

Variables dE2 dS2 : Z -> Q.
Variables packed2 scales2 : Z -> Z.
Variable s2 : Q.

Definition train_runner_at (mult : Q) (r j : Z) : Q :=
  ((if (j mod 2 =? 0)%Z
    then dE2 (nib_lo (packed2
      (r * (cols / 2) + j / 16 * 8 + (j mod 16) / 2)))
    else dE2 (nib_hi (packed2
      (r * (cols / 2) + j / 16 * 8 + (j mod 16) / 2))))
   * (dS2 (scales2 (r * (cols / 16) + j / 16)) * mult))%Q.

Definition modelopt_serve_alpha : Q := qguard_inv (qguard_inv s2).

Definition train_runner_guarded_mult : Q :=
  if Qeq_dec s2 0 then 1%Q else s2.

Theorem modelopt_alpha_faithful :
  ~ (s2 == 0)%Q -> (modelopt_serve_alpha == s2)%Q.
Proof.
  intros Hnz; unfold modelopt_serve_alpha.
  rewrite (qguard_inv_nonzero s2 Hnz).
  assert (Hinz : ~ (/ s2 == 0)%Q).
  { intro E.
    apply Hnz.
    rewrite <- (Qinv_involutive s2), E; reflexivity. }
  rewrite (qguard_inv_nonzero (/ s2) Hinz).
  apply Qinv_involutive.
Qed.

Theorem train_runner_agrees_serve : ~ (s2 == 0)%Q ->
  forall r j,
  (train_runner_at s2 r j
   == serve_at dE2 dS2 packed2 scales2 modelopt_serve_alpha r j)%Q.
Proof.
  intros Hnz r j; unfold train_runner_at, serve_at, serve_loop_val.
  rewrite (modelopt_alpha_faithful Hnz).
  destruct (j mod 2 =? 0)%Z; reflexivity.
Qed.

Theorem train_runner_zero_scale2_collapses : (s2 == 0)%Q ->
  forall r j, (train_runner_at s2 r j == 0)%Q.
Proof.
  intros Hz r j; unfold train_runner_at.
  destruct (j mod 2 =? 0)%Z; rewrite Hz; ring.
Qed.

Theorem serve_zero_scale2_is_unscaled : (s2 == 0)%Q ->
  forall r j,
  (serve_at dE2 dS2 packed2 scales2 modelopt_serve_alpha r j
   == (if (j mod 2 =? 0)%Z
       then dE2 (nib_lo (packed2
         (r * bytes_per_row + j / 16 * 8 + (j mod 16) / 2)))
       else dE2 (nib_hi (packed2
         (r * bytes_per_row + j / 16 * 8 + (j mod 16) / 2))))
      * dS2 (scales2 (r * blocks_per_row + j / 16)))%Q.
Proof.
  intros Hz r j; unfold serve_at, serve_loop_val, modelopt_serve_alpha.
  rewrite (qguard_inv_zero s2 Hz).
  assert (H1 : qguard_inv 1 = (/ 1)%Q)
    by (apply qguard_inv_nonzero; intro E; discriminate E).
  rewrite H1.
  assert (Hinv1 : (/ 1)%Q = 1%Q) by reflexivity.
  rewrite Hinv1.
  destruct (j mod 2 =? 0)%Z; ring.
Qed.

Definition train_runner_fixed_mult : Q := qguard_inv (qguard_inv s2).

Theorem train_runner_fixed_mult_is_serve_alpha :
  train_runner_fixed_mult = modelopt_serve_alpha.
Proof. reflexivity. Qed.

Theorem train_runner_fixed_agrees_serve : forall r j,
  (train_runner_at train_runner_fixed_mult r j
   == serve_at dE2 dS2 packed2 scales2 modelopt_serve_alpha r j)%Q.
Proof.
  intros r j; unfold train_runner_fixed_mult, modelopt_serve_alpha,
    train_runner_at, serve_at, serve_loop_val.
  destruct (j mod 2 =? 0)%Z; reflexivity.
Qed.

Theorem train_runner_guarded_agrees_serve : forall r j,
  (train_runner_at train_runner_guarded_mult r j
   == serve_at dE2 dS2 packed2 scales2 modelopt_serve_alpha r j)%Q.
Proof.
  intros r j; unfold train_runner_guarded_mult.
  destruct (Qeq_dec s2 0) as [Hz | Hnz].
  - unfold train_runner_at, serve_at, serve_loop_val, modelopt_serve_alpha.
    rewrite (qguard_inv_zero s2 Hz).
    assert (H1 : qguard_inv 1 = (/ 1)%Q)
      by (apply qguard_inv_nonzero; intro E; discriminate E).
    rewrite H1.
    assert (Hinv1 : (/ 1)%Q = 1%Q) by reflexivity.
    rewrite Hinv1.
    destruct (j mod 2 =? 0)%Z; reflexivity.
  - apply train_runner_agrees_serve; auto.
Qed.

End TrainRunnerDequant.

End RowMajorLayout.

Theorem train_runner_raw_zero_scale2_diverges_from_serving :
  ~ (train_runner_at 1 (fun _ => 1%Q) (fun _ => 1%Q)
       (fun _ => 0%Z) (fun _ => 0%Z) 0%Q 0%Z 0%Z
     == serve_at 1 (fun _ => 1%Q) (fun _ => 1%Q)
          (fun _ => 0%Z) (fun _ => 0%Z) (modelopt_serve_alpha 0%Q) 0%Z 0%Z)%Q.
Proof.
  intro H; vm_compute in H; discriminate H.
Qed.
