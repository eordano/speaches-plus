use nv_engine::block_manager::BlockManager;
use nv_engine::sequence::Sequence;

fn seq(id: u64, prompt: Vec<u32>, max_new: usize) -> Sequence {
    Sequence::new(id, prompt, max_new)
}

#[test]
fn failed_allocate_after_prefix_share_loses_blocks() {
    let block_size = 16;
    let num_blocks = 16;
    let mut bm = BlockManager::new(num_blocks, block_size);

    let shared: Vec<u32> = (0..32u32).collect();
    let mut a = seq(1, shared.clone(), 8);
    a.block_table = bm.allocate_for(&a).unwrap();
    a.transition_to_prefill().unwrap();
    a.record_computed(shared.len());
    a.transition_to_decode().unwrap();
    bm.commit_filled_block(&a, 0);
    bm.commit_filled_block(&a, 1);
    bm.deallocate(&a);
    let free_after_a = bm.num_free();

    let long: Vec<u32> = shared.iter().copied().chain(100..100 + 240u32).collect();
    let b = seq(2, long, 8);
    let err = bm.allocate_for(&b).unwrap_err();
    let free_after_failure = bm.num_free();

    println!(
        "block_size={block_size} num_blocks={num_blocks} free_after_a={free_after_a} \
         free_after_failed_allocate={free_after_failure} leaked={} err={}",
        free_after_a as i64 - free_after_failure as i64,
        err
    );

    let c = seq(3, (0..32u32).collect(), 8);
    let retry = bm.allocate_for(&c);
    println!(
        "retry_small_seq_ok={} free_now={}",
        retry.is_ok(),
        bm.num_free()
    );
    assert_eq!(
        free_after_failure,
        free_after_a,
        "a failed allocation leaked {} block(s)",
        free_after_a - free_after_failure
    );
}

#[test]
fn preempted_sequence_needs_more_blocks_than_it_held() {
    let block_size = 16;
    let mut s = seq(1, (0..24u32).collect(), 128);
    let before = s.num_blocks_needed(block_size);
    s.transition_to_prefill().unwrap();
    s.record_computed(24);
    s.transition_to_decode().unwrap();
    for t in 0..100u32 {
        s.append_token(t).unwrap();
        s.record_computed(1);
    }
    let held = s.num_blocks_needed(block_size);
    s.reset_for_recompute();
    let after_preempt = s.num_blocks_needed(block_size);
    println!(
        "blocks_needed at_admit={before} after_100_tokens={held} after_reset_for_recompute={after_preempt}"
    );
    assert_eq!(
        after_preempt, held,
        "reset_for_recompute changed the recompute footprint: held {held} blocks before \
         preemption, needs {after_preempt} after. The starvation this file guards depends on \
         those being equal; if that changed, re-derive the livelock before relaxing this."
    );
    assert!(
        after_preempt > before,
        "a preempted sequence needs {after_preempt} blocks but was admitted needing {before}; \
         if these were equal there would be no re-admission cliff and no livelock to guard"
    );
}
