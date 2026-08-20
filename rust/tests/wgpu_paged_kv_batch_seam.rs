#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_paged_kv_batch_seam_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_paged_kv_batch_seam compiled OUT (no `wgpu` feature). This is a SKIP, not a pass: a \
         cfg-out prints 0 passed AND 0 ignored. Re-run with \
         NVK_PKG=speaches-plus NVK_FEATURES=cuda,wgpu."
    );
}

#[cfg(feature = "wgpu")]
mod gated {
    use nv_kernels::wgpu_backend::device::WgpuContext;
    use nv_kernels::wgpu_backend::kernels::kv_fp8_paged as paged;
    use speaches_plus::oapi::chat_engine_wgpu::batch;

    pub const KV_FP8_PAGED_ENTRIES_ARE_HOST_ARRAY_SHAPED_SO_A_POOL_CANNOT_STAY_RESIDENT: &str =
        "every published kv_fp8_paged entry takes the WHOLE pool as a host slice and reads it \
         back after the dispatch, so a serving consumer built on it would download and upload \
         the entire KV pool per call; wiring the pool into decode needs resident-buffer entries \
         in nv-kernels (GpuTensor in, GpuTensor out) and a decoder whose attention indexes KV \
         through a block table";

    const GEMMA4_SHAPED_CONFIG: &str = r#"{"num_hidden_layers":60,"num_key_value_heads":16,
        "head_dim":256,"num_global_key_value_heads":4,"global_head_dim":512,
        "sliding_window":1024,"hidden_size":5376,"num_attention_heads":32}"#;

    const STREAMS: usize = 4;
    const BLOCK_SIZE: usize = 16;
    const BLOCKS_PER_STREAM: usize = 2;
    const N_KV: usize = 4;
    const HEAD_DIM: usize = 8;

    fn num_blocks() -> usize {
        STREAMS * BLOCKS_PER_STREAM
    }

    fn pool_slots() -> usize {
        num_blocks() * BLOCK_SIZE
    }

    fn stream_table(stream: usize) -> Vec<i32> {
        (0..BLOCKS_PER_STREAM)
            .map(|b| (stream * BLOCKS_PER_STREAM + b) as i32)
            .collect()
    }

    fn bf16_bits(x: f32) -> u16 {
        let b = x.to_bits();
        ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
    }

    fn rows(n_tokens: usize, seed: u32) -> Vec<u16> {
        (0..n_tokens * N_KV * HEAD_DIM)
            .map(|i| {
                let t = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) & 0xff;
                bf16_bits((t as f32 - 128.0) / 64.0)
            })
            .collect()
    }

    fn ctx() -> &'static WgpuContext {
        WgpuContext::shared().unwrap_or_else(|e| {
            panic!(
                "no wgpu adapter: {e}. This suite must not pass by skipping -- run it under \
                 scripts/nvk.sh, which sets the Vulkan loader."
            )
        })
    }

    #[test]
    fn one_pool_gives_every_stream_position_a_physical_slot_no_other_stream_can_reach() {
        let mut seen = std::collections::HashMap::new();
        for stream in 0..STREAMS {
            let table = stream_table(stream);
            for pos in 0..BLOCKS_PER_STREAM * BLOCK_SIZE {
                let slot = paged::paged_slot(&table, BLOCK_SIZE, pos);
                assert!(
                    slot < pool_slots(),
                    "stream {stream} position {pos} mapped outside the pool"
                );
                if let Some((other, other_pos)) = seen.insert(slot, (stream, pos)) {
                    panic!(
                        "physical slot {slot} is claimed by stream {stream} position {pos} and \
                         by stream {other} position {other_pos}: a batched decode over this pool \
                         would serve one stream's KV to another"
                    );
                }
            }
        }
        assert_eq!(seen.len(), STREAMS * BLOCKS_PER_STREAM * BLOCK_SIZE);
    }

    #[test]
    fn block_granular_capacity_charges_the_context_a_stream_uses_not_the_window_it_could_use() {
        let kv = batch::kv_geometry_from_config(GEMMA4_SHAPED_CONFIG, batch::KV_ELEM_BYTES)
            .expect("the gemma4-shaped config parses into a KV geometry");
        let max_seq = 32_768u64;
        let used_ctx = 512u64;
        let block = 256u64;
        let slot_static = kv.bytes_at(max_seq);
        let bytes_per_pos = kv.bytes_at(2) - kv.bytes_at(1);
        let paged_bytes = used_ctx.div_ceil(block) * block * bytes_per_pos;
        assert!(
            paged_bytes * 4 < slot_static,
            "block-granular accounting charged {paged_bytes} bytes for a {used_ctx}-token stream \
             against {slot_static} for a static {max_seq}-token slot: if paging does not shrink \
             the per-stream charge by a wide margin there is no admission to win back"
        );
        let admission = batch::Admission {
            budget: batch::StepBudget {
                weight_bytes: 32_760_000_000,
                kv,
            },
            max_seq,
            knobs: batch::BatchKnobs {
                max_batch: 64,
                ..batch::BatchKnobs::default()
            },
        };
        let static_slots = admission.memory_slots(64.0);
        let paged_slots = ((64.0 - admission.knobs.headroom_gib)
            / (paged_bytes as f64 / (1u64 << 30) as f64))
            .floor() as usize;
        eprintln!(
            "at {used_ctx} live tokens and a {block}-position block: static slots {static_slots}, \
             block-granular slots {paged_slots}"
        );
        assert!(
            paged_slots > static_slots,
            "paging must widen admission at short context or it buys nothing: {paged_slots} vs \
             {static_slots}"
        );
    }

    #[test]
    fn the_paged_kernel_writes_a_streams_rows_only_into_that_streams_blocks() {
        let ctx = ctx();
        let n_tokens = BLOCK_SIZE + 3;
        let x = rows(n_tokens, 0x51ee);
        let table = stream_table(1);

        let mut out = vec![0u8; pool_slots() * N_KV * HEAD_DIM];
        let mut scales = vec![0f32; pool_slots() * N_KV];
        paged::quantize_kv_fp8_paged(
            ctx,
            &x,
            &mut out,
            &mut scales,
            &table,
            &[0],
            n_tokens,
            N_KV,
            HEAD_DIM,
            BLOCK_SIZE,
        )
        .expect("gpu paged quantize");

        let mut want = vec![0u8; out.len()];
        let mut want_scales = vec![0f32; scales.len()];
        paged::cpu_quantize_kv_fp8_paged(
            &x,
            &mut want,
            &mut want_scales,
            &table,
            0,
            n_tokens,
            N_KV,
            HEAD_DIM,
            BLOCK_SIZE,
        );
        assert_eq!(
            out, want,
            "the gpu paged quantize disagrees with the host reference the seam's block math is \
             written against"
        );
        assert_eq!(scales, want_scales, "paged scales disagree");

        for stream in 0..STREAMS {
            if stream == 1 {
                continue;
            }
            for b in stream_table(stream) {
                let base = b as usize * BLOCK_SIZE * N_KV * HEAD_DIM;
                let end = base + BLOCK_SIZE * N_KV * HEAD_DIM;
                assert!(
                    out[base..end].iter().all(|v| *v == 0),
                    "writing stream 1 touched stream {stream} block {b}"
                );
            }
        }
    }

    #[test]
    fn a_stream_reads_back_the_rows_it_wrote_through_its_own_block_table() {
        let ctx = ctx();
        let n_tokens = BLOCK_SIZE + 3;
        let x = rows(n_tokens, 0x1234);
        let table = stream_table(2);
        let mut out = vec![0u8; pool_slots() * N_KV * HEAD_DIM];
        let mut scales = vec![0f32; pool_slots() * N_KV];
        paged::quantize_kv_fp8_paged(
            ctx, &x, &mut out, &mut scales, &table, &[0], n_tokens, N_KV, HEAD_DIM, BLOCK_SIZE,
        )
        .expect("gpu paged quantize");

        let mut back = vec![0u16; n_tokens * N_KV * HEAD_DIM];
        paged::dequantize_kv_fp8_paged(
            ctx, &out, &scales, &table, &mut back, n_tokens, N_KV, HEAD_DIM, BLOCK_SIZE,
        )
        .expect("gpu paged dequantize");

        let mut want = vec![0u16; back.len()];
        paged::cpu_dequantize_kv_fp8_paged(
            &out, &scales, &table, &mut want, n_tokens, N_KV, HEAD_DIM, BLOCK_SIZE,
        );
        assert_eq!(
            back, want,
            "gpu and host paged dequantize disagree on the same pool and block table"
        );

        let mut worst = 0f32;
        let mut amax = 0f32;
        for (got, src) in back.iter().zip(&x) {
            let g = f32::from_bits((*got as u32) << 16);
            let s = f32::from_bits((*src as u32) << 16);
            amax = amax.max(s.abs());
            worst = worst.max((g - s).abs());
        }
        assert!(
            amax > 0.0 && worst < 0.08 * amax,
            "paged fp8 round trip lost {worst:.4} against a {amax:.4} full scale: e4m3 with one \
             scale per (slot, head) holds three mantissa bits, so an error past a few percent of \
             full scale is a block-table fault, not quantization"
        );
    }

    #[test]
    fn a_shared_prefix_block_copies_from_one_streams_table_into_anothers() {
        let ctx = ctx();
        let n_tokens = BLOCK_SIZE;
        let x = rows(n_tokens, 0xbeef);
        let donor = stream_table(0);
        let mut k = vec![0u8; pool_slots() * N_KV * HEAD_DIM];
        let mut ks = vec![0f32; pool_slots() * N_KV];
        paged::quantize_kv_fp8_paged(
            ctx, &x, &mut k, &mut ks, &donor, &[0], n_tokens, N_KV, HEAD_DIM, BLOCK_SIZE,
        )
        .expect("gpu paged quantize");
        let mut v = k.clone();
        let mut vs = ks.clone();

        let mut want = k.clone();
        let mut want_scales = ks.clone();
        let (src_block, dst_block) = (donor[0] as usize, stream_table(3)[0] as usize);
        paged::cpu_copy_kv_block_fp8(
            &mut want,
            &mut want_scales,
            src_block,
            dst_block,
            BLOCK_SIZE,
            N_KV,
            HEAD_DIM,
        );
        paged::copy_kv_block_fp8(
            ctx,
            &mut k,
            &mut v,
            &mut ks,
            &mut vs,
            src_block,
            dst_block,
            BLOCK_SIZE,
            N_KV,
            HEAD_DIM,
        )
        .expect("gpu paged block copy");
        assert_eq!(
            k, want,
            "the gpu block copy disagrees with the host reference: a shared-prefix batch would \
             hand the second stream the wrong KV"
        );
        assert_eq!(ks, want_scales, "block copy dropped the per-slot scales");
        assert_eq!(v, k, "the k and v halves of one copy diverged");

        let borrower = stream_table(3);
        let mut got = vec![0u16; n_tokens * N_KV * HEAD_DIM];
        paged::dequantize_kv_fp8_paged(
            ctx, &k, &ks, &borrower, &mut got, n_tokens, N_KV, HEAD_DIM, BLOCK_SIZE,
        )
        .expect("gpu paged dequantize");
        let mut donor_rows = vec![0u16; got.len()];
        paged::dequantize_kv_fp8_paged(
            ctx,
            &k,
            &ks,
            &donor,
            &mut donor_rows,
            n_tokens,
            N_KV,
            HEAD_DIM,
            BLOCK_SIZE,
        )
        .expect("gpu paged dequantize");
        assert_eq!(
            got, donor_rows,
            "after the copy the borrowing stream must read the donor's prefix through its own \
             block table"
        );
    }

    #[test]
    fn the_published_entries_address_the_whole_pool_so_a_window_is_not_addressable() {
        let ctx = ctx();
        let n_tokens = 1;
        let x = rows(n_tokens, 0x77);
        let stream = 3usize;
        let table = stream_table(stream);

        let mut whole = vec![0u8; pool_slots() * N_KV * HEAD_DIM];
        let mut whole_scales = vec![0f32; pool_slots() * N_KV];
        paged::quantize_kv_fp8_paged(
            ctx,
            &x,
            &mut whole,
            &mut whole_scales,
            &table,
            &[0],
            n_tokens,
            N_KV,
            HEAD_DIM,
            BLOCK_SIZE,
        )
        .expect("gpu paged quantize into the whole pool");

        let mut window = vec![0u8; BLOCKS_PER_STREAM * BLOCK_SIZE * N_KV * HEAD_DIM];
        let mut window_scales = vec![0f32; BLOCKS_PER_STREAM * BLOCK_SIZE * N_KV];
        let rebased: Vec<i32> = (0..BLOCKS_PER_STREAM as i32).collect();
        paged::quantize_kv_fp8_paged(
            ctx,
            &x,
            &mut window,
            &mut window_scales,
            &rebased,
            &[0],
            n_tokens,
            N_KV,
            HEAD_DIM,
            BLOCK_SIZE,
        )
        .expect("gpu paged quantize into a per-stream window");

        let whole_off = stream * BLOCKS_PER_STREAM * BLOCK_SIZE * N_KV * HEAD_DIM;
        let one_row = N_KV * HEAD_DIM;
        assert_eq!(
            &whole[whole_off..whole_off + one_row],
            &window[..one_row],
            "the same rows must land at the same offset within a block whichever pool is passed"
        );
        assert!(
            whole.len() > window.len(),
            "{KV_FP8_PAGED_ENTRIES_ARE_HOST_ARRAY_SHAPED_SO_A_POOL_CANNOT_STAY_RESIDENT}: \
             addressing one stream's window means passing a different pool and rebasing its \
             block table, which is exactly the copy a resident pool would not need"
        );
        eprintln!(
            "{KV_FP8_PAGED_ENTRIES_ARE_HOST_ARRAY_SHAPED_SO_A_POOL_CANNOT_STAY_RESIDENT}. One \
             write of {n_tokens} row(s) moved {} pool bytes across the bus in the whole-pool \
             form and {} in the rebased form.",
            whole.len() * 2,
            window.len() * 2
        );
    }
}
