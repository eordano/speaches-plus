From Stdlib Require Import List Arith Lia.
Import ListNotations.
From SpeachesPlus Require Import PdlOrder GenPdl.

Theorem pair_residual_rmsnorm :
  forall t, monotone t -> pdl_dependent t kResidual kRmsnorm ->
    forall pw pr, In pw (k_writes kResidual b_resid) -> In pr (k_reads kRmsnorm b_resid) ->
      t kResidual pw < t kRmsnorm pr.
Proof.
  intros t Hm Hd. apply pdl_writes_precede_reads with (b := b_resid);
  [exact Hm | exact kResidual_writes_b_resid_before_epilog
   | exact kRmsnorm_reads_b_resid_after_prolog | exact Hd].
Qed.

Theorem pair_rmsnorm_gemv :
  forall t, monotone t -> pdl_dependent t kRmsnorm kGemv ->
    forall pw pr, In pw (k_writes kRmsnorm b_norm) -> In pr (k_reads kGemv b_norm) ->
      t kRmsnorm pw < t kGemv pr.
Proof.
  intros t Hm Hd. apply pdl_writes_precede_reads with (b := b_norm);
  [exact Hm | exact kRmsnorm_writes_b_norm_before_epilog
   | exact kGemv_reads_b_norm_after_prolog | exact Hd].
Qed.

Theorem pair_gemv_rope :
  forall t, monotone t -> pdl_dependent t kGemv kRope ->
    forall pw pr, In pw (k_writes kGemv b_gemv) -> In pr (k_reads kRope b_gemv) ->
      t kGemv pw < t kRope pr.
Proof.
  intros t Hm Hd. apply pdl_writes_precede_reads with (b := b_gemv);
  [exact Hm | exact kGemv_writes_b_gemv_before_epilog
   | exact kRope_reads_b_gemv_after_prolog | exact Hd].
Qed.

Theorem pair_rope_kvfp8 :
  forall t, monotone t -> pdl_dependent t kRope kKvFp8 ->
    forall pw pr, In pw (k_writes kRope b_k) -> In pr (k_reads kKvFp8 b_k) ->
      t kRope pw < t kKvFp8 pr.
Proof.
  intros t Hm Hd. apply pdl_writes_precede_reads with (b := b_k);
  [exact Hm | exact kRope_writes_b_k_before_epilog
   | exact kKvFp8_reads_b_k_after_prolog | exact Hd].
Qed.

Theorem pair_kvfp8_flash_scales :
  forall t, monotone t -> pdl_dependent t kKvFp8 kFlash ->
    forall pw pr, In pw (k_writes kKvFp8 b_scales) -> In pr (k_reads kFlash b_scales) ->
      t kKvFp8 pw < t kFlash pr.
Proof.
  intros t Hm Hd. apply pdl_writes_precede_reads with (b := b_scales);
  [exact Hm | exact kKvFp8_writes_b_scales_before_epilog
   | exact kFlash_reads_b_scales_after_prolog | exact Hd].
Qed.

Theorem pair_kvfp8_flash_fp8 :
  forall t, monotone t -> pdl_dependent t kKvFp8 kFlash ->
    forall pw pr, In pw (k_writes kKvFp8 b_fp8) -> In pr (k_reads kFlash b_fp8) ->
      t kKvFp8 pw < t kFlash pr.
Proof.
  intros t Hm Hd. apply pdl_writes_precede_reads with (b := b_fp8);
  [exact Hm | exact kKvFp8_writes_b_fp8_before_epilog
   | exact kFlash_reads_b_fp8_after_prolog | exact Hd].
Qed.

Theorem pair_rope_flash_q :
  forall t, monotone t -> pdl_dependent t kRope kFlash ->
    forall pw pr, In pw (k_writes kRope b_q) -> In pr (k_reads kFlash b_q) ->
      t kRope pw < t kFlash pr.
Proof.
  intros t Hm Hd. apply pdl_writes_precede_reads with (b := b_q);
  [exact Hm | exact kRope_writes_b_q_before_epilog
   | exact kFlash_reads_b_q_after_prolog | exact Hd].
Qed.

Theorem kvfp8_both_outputs_covered :
  forall p, In p (k_writes kKvFp8 b_scales) \/ In p (k_writes kKvFp8 b_fp8) ->
    p < k_epilog kKvFp8.
Proof.
  apply multi_output_needs_all_writes;
  [exact kKvFp8_writes_b_scales_before_epilog | exact kKvFp8_writes_b_fp8_before_epilog].
Qed.

Theorem rope_both_outputs_covered :
  forall p, In p (k_writes kRope b_q) \/ In p (k_writes kRope b_k) ->
    p < k_epilog kRope.
Proof.
  apply multi_output_needs_all_writes;
  [exact kRope_writes_b_q_before_epilog | exact kRope_writes_b_k_before_epilog].
Qed.

Theorem derivev_has_no_explicit_trigger_and_no_programmatic_launch :
  flash_splitk_fused_fp8_derivev_kernel_launched_with_programmatic_serialization = false
  /\ flash_splitk_fused_fp8_kernel_launched_with_programmatic_serialization = true.
Proof. split; reflexivity. Qed.

Theorem pair_kvfp8_flash_derivev_scales_holds_because_it_is_unwired :
  forall t, monotone t -> serialized t kKvFp8 kFlashDeriveV ->
    forall pw pr, In pw (k_writes kKvFp8 b_scales) -> In pr (k_reads kFlashDeriveV b_scales) ->
      t kKvFp8 pw < t kFlashDeriveV pr.
Proof.
  intros t Hm Hs. apply unwired_neighbour_is_safe with (b := b_scales);
  [exact Hm | exact kFlashDeriveV_reads_b_scales_after_prolog | exact Hs].
Qed.

Theorem pair_kvfp8_flash_derivev_fp8_holds_because_it_is_unwired :
  forall t, monotone t -> serialized t kKvFp8 kFlashDeriveV ->
    forall pw pr, In pw (k_writes kKvFp8 b_fp8) -> In pr (k_reads kFlashDeriveV b_fp8) ->
      t kKvFp8 pw < t kFlashDeriveV pr.
Proof.
  intros t Hm Hs. apply unwired_neighbour_is_safe with (b := b_fp8);
  [exact Hm | exact kFlashDeriveV_reads_b_fp8_after_prolog | exact Hs].
Qed.

Theorem pair_rope_flash_derivev_q_holds_because_it_is_unwired :
  forall t, monotone t -> serialized t kRope kFlashDeriveV ->
    forall pw pr, In pw (k_writes kRope b_q) -> In pr (k_reads kFlashDeriveV b_q) ->
      t kRope pw < t kFlashDeriveV pr.
Proof.
  intros t Hm Hs. apply unwired_neighbour_is_safe with (b := b_q);
  [exact Hm | exact kFlashDeriveV_reads_b_q_after_prolog | exact Hs].
Qed.
