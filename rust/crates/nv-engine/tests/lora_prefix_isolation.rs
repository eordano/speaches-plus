use nv_engine::{BlockManager, Sequence, SequenceState};
use xxhash_rust::xxh3::xxh3_64_with_seed;

fn seq(id: u64, prompt: Vec<u32>) -> Sequence {
    Sequence::new(id, prompt, 16)
}

fn chain_hashes(tokens: &[u32], block_size: usize, adapter: Option<&str>) -> Vec<u64> {
    let mut prefix = 0u64;
    let mut out = Vec::new();
    for chunk in tokens.chunks(block_size) {
        if chunk.len() != block_size {
            break;
        }
        let h = BlockManager::compute_block_hash_tagged(chunk, prefix, adapter);
        out.push(h);
        prefix = h;
    }
    out
}

#[test]
fn no_adapter_path_matches_legacy_hash_exactly() {
    let cases: Vec<(Vec<u32>, u64)> = vec![
        (vec![1, 2, 3, 4], 0),
        (vec![1, 2, 3, 4], 0xdead_beef),
        (vec![0], 42),
        (vec![u32::MAX, 0, 7, 7, 7], 0x9e37_79b9_7f4a_7c15),
        ((0..64).collect(), 123456789),
    ];
    for (tokens, prefix) in cases {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                tokens.as_ptr() as *const u8,
                std::mem::size_of_val(tokens.as_slice()),
            )
        };
        let legacy = xxh3_64_with_seed(bytes, prefix);
        assert_eq!(BlockManager::compute_block_hash(&tokens, prefix), legacy);
        assert_eq!(
            BlockManager::compute_block_hash_tagged(&tokens, prefix, None),
            legacy
        );
    }
    assert_eq!(
        BlockManager::compute_block_hash(&[10, 11, 12, 13], 0),
        xxh3_64_with_seed(
            &[10u32, 11, 12, 13]
                .iter()
                .flat_map(|t| t.to_ne_bytes())
                .collect::<Vec<u8>>(),
            0
        )
    );
}

#[test]
fn different_adapters_produce_different_hashes() {
    let tokens: Vec<u32> = (0..8).collect();
    let none = chain_hashes(&tokens, 4, None);
    let a = chain_hashes(&tokens, 4, Some("adapter-a"));
    let b = chain_hashes(&tokens, 4, Some("adapter-b"));
    let empty = chain_hashes(&tokens, 4, Some(""));
    assert_eq!(none.len(), 2);
    for i in 0..2 {
        assert_ne!(none[i], a[i]);
        assert_ne!(none[i], b[i]);
        assert_ne!(a[i], b[i]);
        assert_ne!(none[i], empty[i]);
        assert_ne!(a[i], empty[i]);
    }
}

#[test]
fn same_adapter_produces_identical_hashes() {
    let tokens: Vec<u32> = (100..116).collect();
    let h1 = chain_hashes(&tokens, 4, Some("adapter-a"));
    let h2 = chain_hashes(&tokens, 4, Some("adapter-a"));
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 4);
}

#[test]
fn chaining_property_holds_with_adapters() {
    let tokens: Vec<u32> = (0..12).collect();
    let base = chain_hashes(&tokens, 4, Some("adapter-a"));

    let mut mutated = tokens.clone();
    mutated[1] = 999;
    let diverged = chain_hashes(&mutated, 4, Some("adapter-a"));
    assert_ne!(base[0], diverged[0]);
    assert_ne!(base[1], diverged[1]);
    assert_ne!(base[2], diverged[2]);

    let mut tail_mutated = tokens.clone();
    tail_mutated[9] = 999;
    let tail = chain_hashes(&tail_mutated, 4, Some("adapter-a"));
    assert_eq!(base[0], tail[0]);
    assert_eq!(base[1], tail[1]);
    assert_ne!(base[2], tail[2]);

    let manual_prefix =
        BlockManager::compute_block_hash_tagged(&tokens[0..4], 0, Some("adapter-a"));
    let manual_second =
        BlockManager::compute_block_hash_tagged(&tokens[4..8], manual_prefix, Some("adapter-a"));
    assert_eq!(manual_second, base[1]);
}

#[test]
fn different_adapters_do_not_share_blocks() {
    let mut bm = BlockManager::new(16, 4);
    let s1 = seq(1, (0..8).collect());
    let t1 = bm.allocate_for_tagged(&s1, Some("adapter-a")).unwrap();
    let s2 = seq(2, (0..8).collect());
    let t2 = bm.allocate_for_tagged(&s2, Some("adapter-b")).unwrap();
    let s3 = seq(3, (0..8).collect());
    let t3 = bm.allocate_for_tagged(&s3, None).unwrap();

    for i in 0..2 {
        assert_ne!(t1[i], t2[i]);
        assert_ne!(t1[i], t3[i]);
        assert_ne!(t2[i], t3[i]);
        assert_eq!(bm.ref_count(t1[i]), 1);
        assert_eq!(bm.ref_count(t2[i]), 1);
        assert_eq!(bm.ref_count(t3[i]), 1);
    }
    assert_eq!(bm.num_free(), 16 - 6);
}

#[test]
fn same_adapter_still_shares_blocks() {
    let mut bm = BlockManager::new(16, 4);
    let s1 = seq(1, vec![1, 2, 3, 4, 5, 6]);
    let t1 = bm.allocate_for_tagged(&s1, Some("adapter-a")).unwrap();
    let s2 = seq(2, vec![1, 2, 3, 4, 9, 9]);
    let t2 = bm.allocate_for_tagged(&s2, Some("adapter-a")).unwrap();

    assert_eq!(t1[0], t2[0]);
    assert_ne!(t1[1], t2[1]);
    assert_eq!(bm.ref_count(t1[0]), 2);
    assert_eq!(bm.ref_count(t1[1]), 1);
    assert_eq!(bm.ref_count(t2[1]), 1);
}

#[test]
fn untagged_allocation_path_is_unchanged() {
    let mut legacy = BlockManager::new(8, 4);
    let mut tagged = BlockManager::new(8, 4);
    let s1 = seq(1, vec![1, 2, 3, 4, 5, 6]);
    let s2 = seq(2, vec![1, 2, 3, 4, 9, 9]);

    let l1 = legacy.allocate_for(&s1).unwrap();
    let l2 = legacy.allocate_for(&s2).unwrap();
    let t1 = tagged.allocate_for_tagged(&s1, None).unwrap();
    let t2 = tagged.allocate_for_tagged(&s2, None).unwrap();

    assert_eq!(l1, t1);
    assert_eq!(l2, t2);
    assert_eq!(l1[0], l2[0]);

    let h_legacy = legacy.blocks[l1[0] as usize].hash.unwrap();
    let h_tagged = tagged.blocks[t1[0] as usize].hash.unwrap();
    assert_eq!(h_legacy, h_tagged);
    assert_eq!(h_legacy, BlockManager::compute_block_hash(&[1, 2, 3, 4], 0));
}

#[test]
fn commit_filled_block_tagged_shares_only_within_adapter() {
    let mut bm = BlockManager::new(16, 4);
    let mut s1 = seq(1, vec![1, 2, 3]);
    s1.state = SequenceState::Decode;
    s1.block_table = bm.allocate_for_tagged(&s1, Some("adapter-a")).unwrap();
    s1.output.push(4);
    bm.commit_filled_block_tagged(&s1, 0, Some("adapter-a"));
    let id0 = s1.block_table[0];
    assert!(bm.blocks[id0 as usize].hash.is_some());

    let s2 = seq(2, vec![1, 2, 3, 4, 9]);
    let t2 = bm.allocate_for_tagged(&s2, Some("adapter-a")).unwrap();
    assert_eq!(t2[0], id0);
    assert_eq!(bm.ref_count(id0), 2);

    let s3 = seq(3, vec![1, 2, 3, 4, 9]);
    let t3 = bm.allocate_for_tagged(&s3, Some("adapter-b")).unwrap();
    assert_ne!(t3[0], id0);

    let s4 = seq(4, vec![1, 2, 3, 4, 9]);
    let t4 = bm.allocate_for(&s4).unwrap();
    assert_ne!(t4[0], id0);
    assert_ne!(t4[0], t3[0]);
}

#[test]
fn multi_block_chain_isolated_per_adapter_end_to_end() {
    let block_size = 4;
    let mut bm = BlockManager::new(32, block_size);
    let prompt: Vec<u32> = (0..16).collect();

    let sa = seq(1, prompt.clone());
    let ta = bm.allocate_for_tagged(&sa, Some("adapter-a")).unwrap();
    let sb = seq(2, prompt.clone());
    let tb = bm.allocate_for_tagged(&sb, Some("adapter-b")).unwrap();

    assert_eq!(ta.len(), 4);
    assert_eq!(tb.len(), 4);
    for i in 0..4 {
        assert_ne!(ta[i], tb[i]);
        let ha = bm.blocks[ta[i] as usize].hash.unwrap();
        let hb = bm.blocks[tb[i] as usize].hash.unwrap();
        assert_ne!(ha, hb);
    }

    let expect_a = chain_hashes(&prompt, block_size, Some("adapter-a"));
    let expect_b = chain_hashes(&prompt, block_size, Some("adapter-b"));
    for i in 0..4 {
        assert_eq!(bm.blocks[ta[i] as usize].hash.unwrap(), expect_a[i]);
        assert_eq!(bm.blocks[tb[i] as usize].hash.unwrap(), expect_b[i]);
    }

    let sa2 = seq(3, prompt.clone());
    let ta2 = bm.allocate_for_tagged(&sa2, Some("adapter-a")).unwrap();
    assert_eq!(ta, ta2);
    for &id in &ta {
        assert_eq!(bm.ref_count(id), 2);
    }
}
