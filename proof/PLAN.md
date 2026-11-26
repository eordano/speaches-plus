# Evict: what is proven and what to fix

Whether the cold tier can free a qwen4_exp session that something on the CPU side still uses, and whether graph captures and device-synchronizing calls can deadlock each other: the engine permit and idle-age selection in evict, the held slots (two base sessions and the kept adapter session) that release_sessions_to_disk may drop, and the capture-ownership protocol in capture_exclusion.rs, for two threads. GPU-side ordering of frees against queued kernels, the CUDA driver, cudarc and the spill format are left out.

11 claims proven, 0 findings open, 2 repaired, 0 accepted.

## To fix (0 open)

## Proven

| Claim | Says | Theorem | Code |
| --- | --- | --- | --- |
| C01 | While a thread holds a NoCaptureInFlight, no other thread holds a capture. | `Findings.shipped_exclusive` | `rust/crates/nv-kernels/src/capture_exclusion.rs:74`, `rust/crates/nv-kernels/src/capture_exclusion.rs:88`, `rust/crates/nv-kernels/src/capture_exclusion.rs:93`, `rust/crates/nv-kernels/src/capture_exclusion.rs:47`, `rust/crates/nv-kernels/src/capture_exclusion.rs:51`, `rust/crates/nv-kernels/src/capture_exclusion.rs:61`, `rust/crates/nv-kernels/src/capture_exclusion.rs:79` |
| C02 | Two threads are never both blocked in capture_in_flight or no_capture_in_flight. | `Findings.shipped_never_both_blocked` | `rust/crates/nv-kernels/src/capture_exclusion.rs:74`, `rust/crates/nv-kernels/src/capture_exclusion.rs:88`, `rust/crates/nv-kernels/src/capture_exclusion.rs:47`, `rust/crates/nv-kernels/src/graph.rs:336`, `rust/crates/nv-kernels/src/graph.rs:309`, `rust/crates/nv-kernels/src/capture_exclusion.rs:51` |
| C03 | One thread can nest exclusions and a capture without waiting on itself, and releasing them all frees the ownership. | `Findings.own_capture_nests` | `rust/crates/nv-kernels/src/capture_exclusion.rs:74`, `rust/crates/nv-kernels/src/capture_exclusion.rs:88`, `rust/crates/nv-kernels/src/capture_exclusion.rs:93`, `rust/crates/nv-kernels/src/capture_exclusion.rs:51`, `rust/crates/nv-kernels/src/capture_exclusion.rs:61`, `rust/crates/nv-kernels/src/capture_exclusion.rs:79` |
| C04 | A thread blocked in either call waits on the other thread, which holds a capture or the exclusion; by C02 that thread is not blocked, so it can go on and release. | `Findings.shipped_waits_on_the_owner` | `rust/crates/nv-kernels/src/capture_exclusion.rs:47`, `rust/crates/nv-kernels/src/capture_exclusion.rs:51` |
| C05 | A thread does block while another holds the exclusion, so C02 and C04 constrain states that occur. | `Findings.blocking_is_not_vacuous` | `rust/crates/nv-kernels/src/capture_exclusion.rs:47`, `rust/crates/nv-kernels/src/capture_exclusion.rs:51` |
| C06 | An eviction that cannot take the engine permit releases nothing; a request holds that permit from take_session to put_session, so eviction never frees a session a request is using. | `Findings.shipped_no_release_under_request` | `rust/src/oapi/chat_engine/qwen4_exp_loop.rs:69`, `rust/src/oapi/chat_engine/qwen4_exp_loop.rs:268` |
| C07 | An idle eviction releases only held slots whose last put is at least the shorter of 300 s and the global idle limit old. | `Findings.shipped_idle_releases_only_idle_slots` | `rust/src/oapi/chat_engine/qwen4_exp_loop.rs:69`, `rust/src/oapi/chat_engine/qwen4_exp_loop.rs:26`, `rust/src/oapi/cold_tier.rs:326`, `rust/crates/nv-models/src/qwen4_exp_residency.rs:511`, `rust/src/oapi/chat_engine/qwen4_exp_loop.rs:39`, `rust/crates/nv-models/src/qwen4_exp_residency.rs:503` |
| C08 | A pressure eviction releases every held slot. | `Findings.shipped_pressure_releases_every_slot` | `rust/src/oapi/chat_engine/qwen4_exp_loop.rs:69`, `rust/src/oapi/chat_engine/qwen4_exp_loop.rs:39` |
| C09 | The idle comparison includes its edge: a slot idle exactly the limit is released, one second less is kept, and a shorter global limit wins. | `Findings.idle_edges` | `rust/src/oapi/chat_engine/qwen4_exp_loop.rs:69`, `rust/src/oapi/chat_engine/qwen4_exp_loop.rs:26`, `rust/src/oapi/cold_tier.rs:326` |
| C10 | An idle eviction releases the kept adapter session only when its last put is at least the idle limit old. | `Findings.shipped_adapter_released_only_when_idle` | `rust/crates/nv-models/src/qwen4_exp_residency.rs:516`, `rust/src/oapi/chat_engine/qwen4_exp_loop.rs:39`, `rust/crates/nv-models/src/qwen4_exp_residency.rs:503`, `rust/crates/nv-models/src/qwen4_exp.rs:3153`, `rust/crates/nv-models/src/qwen4_exp.rs:3319` |
| C11 | A fresh adapter session survives an idle eviction of base slot 0, an idle one is released with it, and pressure eviction releases it when it is the only session held. | `Findings.adapter_slot_edges` | `rust/crates/nv-models/src/qwen4_exp_residency.rs:516`, `rust/src/oapi/chat_engine/qwen4_exp_loop.rs:39`, `rust/crates/nv-models/src/qwen4_exp_residency.rs:503`, `rust/crates/nv-models/src/qwen4_exp.rs:3153` |

## Repaired

| Finding | What was wrong | Test | Claim |
| --- | --- | --- | --- |
| F01 | A thread that takes the exclusion inside its own capture could deadlock the capture machinery | `nv-kernels capture_exclusion tests an_exclusion_inside_a_capture_does_not_wait_on_a_capture_queued_behind_it and an_exclusion_inside_a_capture_does_not_wait_on_an_excluder_waiting_for_that_capture, which timed out on the old code` | C02 |
| F02 | Any eviction dropped the kept adapter session whatever its age, and the cold tier never released it on its own | `nv-models qwen4_exp_session_residency releasing_a_base_slot_keeps_a_fresh_adapter_session_which_the_cold_tier_sees_as_its_own_slot` | C10 |

## Not covered

- GPU-side ordering: whether a freed session buffer can still be read by kernels or graph replays queued on the model stream or the MoE auxiliary stream (cudarc frees with cuMemFreeAsync on the slice's stream; event tracking is off).
- That only cuCtxSynchronize-class calls invalidate a capture in flight is taken from the probe recorded in A_DEVICE_WIDE_SYNCHRONIZE_ON_ANY_THREAD_INVALIDATES_EVERY_CAPTURE_IN_FLIGHT_IN_THE_PROCESS, not modeled.
- More than two threads: the model has two. The argument (the owner never waits on these guards, every waiter waits on the owner) does not depend on the count, but it is proven for two.
- Fairness: nothing bounds how long a waiting thread waits while other threads keep taking turns as owner.
- A CaptureInFlight or NoCaptureInFlight dropped on another thread than took it: both are Send, and the ownership is keyed by ThreadId at acquisition only.
- gemma4_e4b captures under its own private capture_lock, outside this protocol.
- Parked-session contents: the spill format, the residency LRU and its budgets, install_parked.
- Whether parked_bytes at max_len matches the adapter session's real device footprint.
- Mutex poisoning, which the guard recovers from with into_inner.
