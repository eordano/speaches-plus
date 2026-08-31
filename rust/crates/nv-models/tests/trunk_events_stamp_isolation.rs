#![cfg(feature = "cuda")]

use candle_core::Device;

#[test]
fn a_marked_slot_holds_a_nonzero_globaltimer_stamp_after_sync() {
    let Ok(device) = Device::new_cuda_with_stream(0) else {
        eprintln!("[skip] no cuda device");
        return;
    };
    std::env::set_var("NV_Q36_TRUNK_EVENTS", "1");
    std::env::set_var("NV_Q36_TRUNK_EVENTS_M1", "1");
    use nv_models::qwen3_5_moe::trunk_events;
    trunk_events::prebuild_if_armed_because_the_ladder_cannot_allocate_mid_capture(&device, 4);
    let tm = trunk_events::begin_if_armed_verify_rows(1, &device, 4);
    tm.head(0);
    tm.head(1);
    tm.tail(0);
    tm.tail(1);
    let stamps = trunk_events::read_stamps_for_the_isolation_gate(&device)
        .expect("the ladder must exist after an armed begin");
    let nonzero = stamps.iter().filter(|&&s| s != 0).count();
    assert!(
        nonzero >= 4,
        "marked 4 slots but only {nonzero} stamps are nonzero of {}: the stamp write path \
         loses launches (device_ptr_mut guard semantics are the prime suspect)",
        stamps.len()
    );
}

#[test]
fn marks_inside_a_captured_graph_stamp_at_replay() {
    let Ok(device) = Device::new_cuda_with_stream(0) else {
        eprintln!("[skip] no cuda device");
        return;
    };
    std::env::set_var("NV_Q36_TRUNK_EVENTS", "1");
    std::env::set_var("NV_Q36_TRUNK_EVENTS_M1", "1");
    use nv_models::qwen3_5_moe::trunk_events;
    trunk_events::prebuild_if_armed_because_the_ladder_cannot_allocate_mid_capture(&device, 4);
    let candle_core::Device::Cuda(d) = &device else {
        unreachable!()
    };
    let stream = d.cuda_stream();
    let mut runner = nv_kernels::graph::CudaGraphRunner::new(stream.clone());
    runner
        .run(0x7e57, |s| {
            nv_layers::cuda_stream::with_stream(s.clone(), || {
                let tm = trunk_events::begin_if_armed_verify_rows(1, &device, 4);
                tm.head(0);
                tm.head(1);
                tm.tail(0);
                tm.tail(1);
                Ok(())
            })
        })
        .expect("capture with marks");
    runner.run(0x7e57, |_s| Ok(())).expect("replay");
    let stamps = trunk_events::read_stamps_for_the_isolation_gate(&device)
        .expect("ladder exists");
    let nonzero = stamps.iter().filter(|&&s| s != 0).count();
    assert!(
        nonzero >= 4,
        "marked 4 slots inside a captured graph but only {nonzero} stamps are nonzero: \
         the timestamp launches are lost by capture (the m=1 ladder failure reproduced \
         WITHOUT the engine)"
    );
}
