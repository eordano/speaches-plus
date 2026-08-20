macro_rules! impl_wgpu_state_snapshot {
    ($t:ty, $cap:ident) => {
        impl $t {
            pub fn state_blob_count(&self) -> usize {
                self.state_buffers.len()
            }

            pub fn state_blob_words(&self, i: usize) -> Option<usize> {
                self.state_buffers
                    .get(i)
                    .map(|(_, bytes)| *bytes as usize / 4)
            }

            pub fn state_blob_download(&self, i: usize) -> anyhow::Result<Option<Vec<u32>>> {
                let Some((buf, bytes)) = self.state_buffers.get(i) else {
                    return Ok(None);
                };
                anyhow::ensure!(
                    bytes.is_multiple_of(4),
                    "state blob {i} of {bytes} bytes is not word-aligned"
                );
                nv_kernels::wgpu_backend::dispatch::read_back::<u32>(
                    self.ctx,
                    buf,
                    *bytes as usize / 4,
                )
                .map(Some)
                .map_err(|e| anyhow::anyhow!("state blob {i} download: {e}"))
            }

            pub fn state_blob_restore(&mut self, i: usize, words: &[u32]) -> anyhow::Result<bool> {
                let Some((buf, bytes)) = self.state_buffers.get(i) else {
                    return Ok(false);
                };
                anyhow::ensure!(
                    words.len() == *bytes as usize / 4,
                    "state blob {i}: got {} words, buffer holds {}",
                    words.len(),
                    *bytes as usize / 4
                );
                if !words.is_empty() {
                    self.ctx
                        .queue
                        .write_buffer(buf, 0, bytemuck::cast_slice(words));
                }
                Ok(true)
            }

            pub fn restore_pos(&mut self, pos: usize) -> anyhow::Result<()> {
                anyhow::ensure!(
                    pos <= self.$cap,
                    "restore_pos {pos} past capacity {}",
                    self.$cap
                );
                self.pos = pos;
                Ok(())
            }
        }
    };
}

pub(crate) use impl_wgpu_state_snapshot;
