#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;
use half::bf16;
use nv_engine::{KvKind, PagedKv};

#[test]
fn paged_kv_block_offset_round_trip() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let num_layers = 2usize;
    let num_blocks = 4usize;
    let num_kv_heads = 4usize;
    let block_size = 16usize;
    let head_dim = 64usize;

    let mut kv = PagedKv::new(
        stream.clone(),
        num_layers,
        num_blocks,
        num_kv_heads,
        block_size,
        head_dim,
    )
    .unwrap();

    let block_elems = num_kv_heads * block_size * head_dim;
    assert_eq!(kv.block_elements(), block_elems);
    assert_eq!(
        kv.total_elements(),
        num_layers * 2 * num_blocks * block_elems
    );

    let target_layer = 1usize;
    let target_kind = KvKind::V;
    let target_block = 2u32;
    let offset = kv
        .block_offset(target_layer, target_kind, target_block)
        .unwrap();
    assert!(kv.block_offset(num_layers, target_kind, 0).is_err());
    assert!(kv.block_offset(0, target_kind, num_blocks as u32).is_err());

    let manual = target_layer * 2 * num_blocks * block_elems
        + target_kind.index() * num_blocks * block_elems
        + (target_block as usize) * block_elems;
    assert_eq!(offset, manual);

    let pattern: Vec<bf16> = (0..block_elems)
        .map(|i| bf16::from_f32((i as f32) * 0.001 - 0.5))
        .collect();
    {
        let mut view = kv.slice_mut().slice_mut(offset..offset + block_elems);
        #[allow(deprecated)]
        stream.memcpy_htod(&pattern, &mut view).unwrap();
    }
    stream.synchronize().unwrap();

    let host = stream.clone_dtoh(kv.storage()).unwrap();
    for i in 0..block_elems {
        assert_eq!(
            host[offset + i].to_f32(),
            pattern[i].to_f32(),
            "mismatch at i={}",
            i
        );
    }
    for &kind in &[KvKind::K, KvKind::V] {
        for layer in 0..num_layers {
            for b in 0..num_blocks as u32 {
                if layer == target_layer && kind == target_kind && b == target_block {
                    continue;
                }
                let off = kv.block_offset(layer, kind, b).unwrap();
                for i in 0..block_elems {
                    assert_eq!(host[off + i].to_f32(), 0.0, "leak at off={} i={}", off, i);
                }
            }
        }
    }
}
