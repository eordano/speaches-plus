#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvKind {
    K,
    V,
}

impl KvKind {
    pub fn index(self) -> usize {
        match self {
            KvKind::K => 0,
            KvKind::V => 1,
        }
    }
}

pub fn checked_block_offset(
    layer: usize,
    num_layers: usize,
    kv: KvKind,
    block_idx: u32,
    num_blocks: usize,
    block_elements: usize,
) -> anyhow::Result<usize> {
    if layer >= num_layers {
        anyhow::bail!("PagedKv.block_offset: layer {layer} out of range ({num_layers} layers)");
    }
    if block_idx as usize >= num_blocks {
        anyhow::bail!("PagedKv.block_offset: block {block_idx} out of range ({num_blocks} blocks)");
    }
    let kv_stride = num_blocks
        .checked_mul(block_elements)
        .ok_or_else(|| anyhow::anyhow!("PagedKv.block_offset: kv stride overflows usize"))?;
    let layer_stride = kv_stride
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("PagedKv.block_offset: layer stride overflows usize"))?;
    layer
        .checked_mul(layer_stride)
        .and_then(|v| v.checked_add(kv.index() * kv_stride))
        .and_then(|v| v.checked_add((block_idx as usize) * block_elements))
        .ok_or_else(|| anyhow::anyhow!("PagedKv.block_offset: offset overflows usize"))
}

#[cfg(feature = "cuda")]
pub use cuda::PagedKv;

#[cfg(not(feature = "cuda"))]
pub use stub::PagedKv;

#[cfg(feature = "cuda")]
mod cuda {
    use super::KvKind;
    use anyhow::Result;
    use cudarc::driver::{CudaSlice, CudaStream};
    use half::bf16;
    use std::sync::Arc;

    pub struct PagedKv {
        pub storage: CudaSlice<bf16>,
        pub num_layers: usize,
        pub num_blocks: usize,
        pub num_kv_heads: usize,
        pub block_size: usize,
        pub head_dim: usize,
        pub stream: Arc<CudaStream>,
    }

    impl PagedKv {
        pub fn new(
            stream: Arc<CudaStream>,
            num_layers: usize,
            num_blocks: usize,
            num_kv_heads: usize,
            block_size: usize,
            head_dim: usize,
        ) -> Result<Self> {
            let total = num_layers
                .checked_mul(2)
                .and_then(|v| v.checked_mul(num_blocks))
                .and_then(|v| v.checked_mul(num_kv_heads))
                .and_then(|v| v.checked_mul(block_size))
                .and_then(|v| v.checked_mul(head_dim))
                .ok_or_else(|| anyhow::anyhow!("paged kv shape overflows usize"))?;
            let storage = stream.alloc_zeros::<bf16>(total)?;
            Ok(Self {
                storage,
                num_layers,
                num_blocks,
                num_kv_heads,
                block_size,
                head_dim,
                stream,
            })
        }

        pub fn block_elements(&self) -> usize {
            self.num_kv_heads * self.block_size * self.head_dim
        }

        pub fn layer_stride(&self) -> usize {
            2 * self.num_blocks * self.block_elements()
        }

        pub fn kv_stride(&self) -> usize {
            self.num_blocks * self.block_elements()
        }

        pub fn block_offset(&self, layer: usize, kv: KvKind, block_idx: u32) -> Result<usize> {
            super::checked_block_offset(
                layer,
                self.num_layers,
                kv,
                block_idx,
                self.num_blocks,
                self.block_elements(),
            )
        }

        pub fn total_elements(&self) -> usize {
            self.num_layers
                * 2
                * self.num_blocks
                * self.num_kv_heads
                * self.block_size
                * self.head_dim
        }

        pub fn slice_mut(&mut self) -> &mut CudaSlice<bf16> {
            &mut self.storage
        }

        pub fn storage(&self) -> &CudaSlice<bf16> {
            &self.storage
        }

        pub fn stream(&self) -> &Arc<CudaStream> {
            &self.stream
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod stub {
    use super::KvKind;
    use anyhow::{bail, Result};

    pub struct PagedKv {
        pub num_layers: usize,
        pub num_blocks: usize,
        pub num_kv_heads: usize,
        pub block_size: usize,
        pub head_dim: usize,
    }

    impl PagedKv {
        pub fn new(
            _num_layers: usize,
            _num_blocks: usize,
            _num_kv_heads: usize,
            _block_size: usize,
            _head_dim: usize,
        ) -> Result<Self> {
            bail!("PagedKv requires the `cuda` feature");
        }

        pub fn block_offset(&self, layer: usize, kv: KvKind, block_idx: u32) -> Result<usize> {
            super::checked_block_offset(
                layer,
                self.num_layers,
                kv,
                block_idx,
                self.num_blocks,
                self.num_kv_heads * self.block_size * self.head_dim,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{checked_block_offset, KvKind};

    #[test]
    fn block_offset_matches_manual_layout() {
        let (num_layers, num_blocks, block_elems) = (2usize, 4usize, 4 * 16 * 64usize);
        let off = checked_block_offset(1, num_layers, KvKind::V, 2, num_blocks, block_elems)
            .expect("in range");
        let manual = 2 * num_blocks * block_elems + num_blocks * block_elems + 2 * block_elems;
        assert_eq!(off, manual);
        assert_eq!(
            checked_block_offset(0, num_layers, KvKind::K, 0, num_blocks, block_elems).unwrap(),
            0
        );
    }

    #[test]
    fn block_offset_rejects_out_of_range() {
        let (num_layers, num_blocks, block_elems) = (2usize, 4usize, 8usize);
        assert!(
            checked_block_offset(2, num_layers, KvKind::K, 0, num_blocks, block_elems).is_err()
        );
        assert!(
            checked_block_offset(0, num_layers, KvKind::K, 4, num_blocks, block_elems).is_err()
        );
        assert!(
            checked_block_offset(0, num_layers, KvKind::K, u32::MAX, num_blocks, block_elems)
                .is_err()
        );
    }

    #[test]
    fn block_offset_rejects_overflowing_geometry() {
        assert!(checked_block_offset(0, 1, KvKind::K, 0, usize::MAX, 2).is_err());
    }
}
