use nv_specdecode::lora_spec::LoraTokenMap;

const MAX_TOKENS: usize = 8;
const MAX_LORAS: usize = 3;

fn map() -> std::sync::Arc<LoraTokenMap> {
    LoraTokenMap::new(MAX_TOKENS, MAX_LORAS).unwrap()
}

#[test]
fn a_mapping_of_all_minus_one_reads_as_inactive_through_every_accessor() {

    let m = map();
    m.set_mapping(&[-1, -1, -1]).unwrap();
    assert!(!m.armed(), "an all -1 mapping must not read as armed");
    assert!(m.snapshot().is_none(), "an all -1 mapping must not hand out a mapping");

    m.set_mapping(&[-1, 1, -1]).unwrap();
    assert!(m.armed(), "one real slot makes the batch active");
    assert_eq!(
        m.snapshot().unwrap(),
        vec![-1, 1, -1],
        "the sentinels must survive into the snapshot or per-token routing is lost"
    );
}

#[test]
fn disarm_hides_a_live_mapping_without_destroying_it() {
    let m = map();
    m.set_mapping(&[0, 1]).unwrap();
    assert!(m.armed() && m.snapshot().is_some(), "precondition: a live mapping");

    m.disarm();
    assert!(!m.armed(), "disarm must take the map out of service");
    assert!(m.snapshot().is_none(), "a disarmed map must hand out nothing");

    m.set_mapping(&[0, 1]).unwrap();
    assert!(m.armed(), "set_mapping is what re-arms");
}

#[test]
fn the_slot_range_accepts_the_sentinel_and_the_last_slot_and_rejects_either_neighbour() {
    let m = map();
    m.set_mapping(&[-1]).expect("-1 is the sentinel, not an out-of-range slot");
    m.set_mapping(&[MAX_LORAS as i32 - 1]).expect("the last slot is in range");

    for bad in [-2, MAX_LORAS as i32] {
        assert!(
            m.set_mapping(&[0, bad]).is_err(),
            "slot {bad} is outside [-1, {}] and must be refused; an accepted \
             out-of-range slot indexes the adapter stack out of bounds",
            MAX_LORAS - 1
        );
    }
}

#[test]
fn a_rejected_mapping_leaves_the_previous_one_untouched() {

    let m = map();
    m.set_mapping(&[0, 1]).unwrap();
    assert!(m.set_mapping(&[0, MAX_LORAS as i32]).is_err());
    assert_eq!(
        m.snapshot().unwrap(),
        vec![0, 1],
        "a rejected mapping must not disturb the one already in service"
    );
}

#[test]
fn the_dimension_limits_are_enforced_at_construction_and_at_every_set() {
    assert!(LoraTokenMap::new(0, MAX_LORAS).is_err(), "zero max_tokens is not a usable map");
    assert!(LoraTokenMap::new(MAX_TOKENS, 0).is_err(), "zero max_loras is not a usable map");

    let m = map();
    assert_eq!((m.max_tokens(), m.max_loras()), (MAX_TOKENS, MAX_LORAS));
    assert!(m.set_mapping(&[]).is_err(), "an empty mapping routes nothing and is a bug upstream");
    assert!(
        m.set_mapping(&vec![0i32; MAX_TOKENS]).is_ok(),
        "exactly max_tokens must fit, or the last slot of every full batch is unusable"
    );
    assert!(
        m.set_mapping(&vec![0i32; MAX_TOKENS + 1]).is_err(),
        "one over max_tokens must be refused; it would overrun the device-side map"
    );
}

#[test]
fn a_fresh_map_is_inactive_before_anything_is_set() {

    let m = map();
    assert!(!m.armed(), "a map with no mapping set must not be armed");
    assert!(m.snapshot().is_none(), "a map with no mapping set must hand out nothing");
}
