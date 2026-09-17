pub struct LagunaStudyCaptureState {
    cache: LagunaKvCache,
    owner: usize,
    layer: usize,
    max_context: usize,
}

impl Laguna {
    pub fn new_study_capture_state(
        &self,
        layer: usize,
        max_context: usize,
    ) -> Result<LagunaStudyCaptureState> {
        anyhow::ensure!(layer < self.layers.len(), "capture layer out of range");
        anyhow::ensure!(
            max_context > 0 && max_context <= self.config.max_position_embeddings,
            "invalid capture context capacity"
        );
        Ok(LagunaStudyCaptureState {
            cache: LagunaKvCache::allocate_with_prefill_chunk(
                &self.config,
                max_context,
                &self.device,
                self.dtype,
                self.prefill_chunk,
            )?,
            owner: self as *const Self as usize,
            layer,
            max_context,
        })
    }

    pub fn study_capture_prefix(
        &self,
        input: &[u32],
        state: &mut LagunaStudyCaptureState,
    ) -> Result<(Tensor, usize, usize)> {
        anyhow::ensure!(
            state.owner == self as *const Self as usize,
            "capture state belongs to a different model"
        );
        anyhow::ensure!(
            !input.is_empty() && input.len() <= state.max_context,
            "complete capture context required; truncation forbidden"
        );
        anyhow::ensure!(
            input.iter().all(|&t| (t as usize) < self.config.vocab_size),
            "invalid capture token"
        );
        state.cache.reset();
        let mut position = 0;
        let mut layer_calls = 0;
        let mut chunks = 0;
        let mut captured = None;
        while position < input.len() {
            let seq = self.next_prefill_chunk_len(input.len() - position);
            let end = position + seq;
            let tokens = Tensor::from_vec(input[position..end].to_vec(), (1, seq), &self.device)?;
            let positions = Tensor::from_vec(
                (position as u32..end as u32).collect::<Vec<_>>(),
                seq,
                &self.device,
            )?;
            self.apply_attn_w8_for(seq);
            let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
            let mut hidden = self
                .embed_tokens(&tokens_flat)?
                .reshape((1usize, seq, self.config.hidden_size))?
                .to_dtype(self.dtype)?;
            let write_start = state.cache.current_len();
            let new_total = write_start + seq;
            state.cache.prepare_for_decode(write_start, new_total)?;
            for layer in 0..=state.layer {
                hidden = self.layer_forward(
                    layer,
                    &hidden,
                    &positions,
                    &mut state.cache,
                    seq,
                    new_total,
                )?;
                layer_calls += 1;
            }
            state.cache.advance(seq);
            chunks += 1;
            if end == input.len() {
                captured = Some(
                    hidden
                        .narrow(1, seq - 1, 1)?
                        .flatten_all()?
                        .to_dtype(DType::F32)?,
                );
            }
            position = end;
        }
        anyhow::ensure!(
            layer_calls == chunks * (state.layer + 1),
            "wrong capture depth"
        );
        Ok((captured.context("missing capture")?, layer_calls, chunks))
    }
}

impl Laguna {
    pub fn study_capture_record_tokens(
        &self,
        input: &[u32],
        state: &mut LagunaStudyCaptureState,
    ) -> Result<(Tensor, usize, usize)> {
        anyhow::ensure!(
            state.owner == self as *const Self as usize,
            "capture state belongs to a different model"
        );
        anyhow::ensure!(
            !input.is_empty() && input.len() <= state.max_context,
            "complete capture context required; truncation forbidden"
        );
        anyhow::ensure!(
            input.iter().all(|&t| (t as usize) < self.config.vocab_size),
            "invalid capture token"
        );
        state.cache.reset();
        let mut position = 0;
        let mut layer_calls = 0;
        let mut chunks = 0;
        let mut captured = Vec::new();
        while position < input.len() {
            let seq = self.next_prefill_chunk_len(input.len() - position);
            let end = position + seq;
            let tokens = Tensor::from_vec(input[position..end].to_vec(), (1, seq), &self.device)?;
            let positions = Tensor::from_vec(
                (position as u32..end as u32).collect::<Vec<_>>(),
                seq,
                &self.device,
            )?;
            self.apply_attn_w8_for(seq);
            let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
            let mut hidden = self
                .embed_tokens(&tokens_flat)?
                .reshape((1usize, seq, self.config.hidden_size))?
                .to_dtype(self.dtype)?;
            let write_start = state.cache.current_len();
            let new_total = write_start + seq;
            state.cache.prepare_for_decode(write_start, new_total)?;
            for layer in 0..=state.layer {
                hidden = self.layer_forward(
                    layer,
                    &hidden,
                    &positions,
                    &mut state.cache,
                    seq,
                    new_total,
                )?;
                layer_calls += 1;
            }
            state.cache.advance(seq);
            chunks += 1;
            captured.push(hidden.reshape((seq, self.config.hidden_size))?.to_dtype(DType::F32)?);
            position = end;
        }
        anyhow::ensure!(
            layer_calls == chunks * (state.layer + 1),
            "wrong capture depth"
        );
        Ok((Tensor::cat(&captured, 0)?, layer_calls, chunks))
    }
}
