
use candle_core::{DType, Device, Tensor};
use nv_layers::lora_slots::{
    LoraAdapter, LoraModuleSpec, LoraModuleWeights, LoraSlotManager, LoraSlotStack,
};
use std::collections::HashMap;

const LADDER: [f32; 4] = [-1.0, -0.5, 0.5, 1.0];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn ladder(&mut self) -> f32 {
        LADDER[(self.next() >> 33) as usize % LADDER.len()]
    }
}

fn ladder_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut r = Rng(seed);
    (0..n).map(|_| r.ladder()).collect()
}

fn tensor(v: &[f32], rows: usize, cols: usize, dev: &Device) -> Tensor {
    assert_eq!(v.len(), rows * cols, "fixture shape");
    Tensor::from_vec(v.to_vec(), (rows, cols), dev).unwrap()
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn host_delta_f64(
    x: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    rank: usize,
    n: usize,
    scaling: f64,
) -> Vec<f64> {
    let mut h = vec![0f64; m * rank];
    for i in 0..m {
        for r in 0..rank {
            let mut acc = 0f64;
            for j in 0..k {
                acc += x[i * k + j] as f64 * a[r * k + j] as f64;
            }
            h[i * rank + r] = acc;
        }
    }
    let mut d = vec![0f64; m * n];
    for i in 0..m {
        for o in 0..n {
            let mut acc = 0f64;
            for r in 0..rank {
                acc += h[i * rank + r] * b[o * rank + r] as f64;
            }
            d[i * n + o] = acc * scaling;
        }
    }
    d
}

fn assert_non_degenerate(v: &[f64], what: &str) {
    let nz = v.iter().filter(|x| **x != 0.0).count();
    assert!(
        nz * 4 > v.len(),
        "{what}: only {nz}/{} reference entries are nonzero; this fixture cannot distinguish a \
         correct delta from a dead one",
        v.len()
    );
    let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        hi - lo > 0.0,
        "{what}: reference is constant at {lo}; a broken delta that returns a constant would pass"
    );
}

fn assert_bit_equal(got: &[f32], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut bad = 0usize;
    let mut first = None;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if (*g as f64) != *w {
            bad += 1;
            if first.is_none() {
                first = Some((i, *g, *w));
            }
        }
    }
    if let Some((i, g, w)) = first {
        panic!(
            "{what}: {bad}/{} entries differ; first at {i}: got {g} want {w}",
            got.len()
        );
    }
}

fn stack(max_loras: usize, max_rank: usize, k: usize, n: usize, dev: &Device) -> LoraSlotStack {
    LoraSlotStack::new(max_loras, max_rank, k, n, DType::F32, dev).unwrap()
}

#[test]
fn delta_is_scaled_by_the_scaling_factor_against_an_f64_host_reference() {
    let dev = Device::Cpu;
    let (m, k, rank, n) = (5usize, 16usize, 4usize, 12usize);
    let xv = ladder_vec(0x11, m * k);
    let av = ladder_vec(0x12, rank * k);
    let bv = ladder_vec(0x13, n * rank);

    let st = stack(2, rank, k, n, &dev);
    st.set_lora(0, &tensor(&av, rank, k, &dev), &tensor(&bv, n, rank, &dev))
        .unwrap();
    let x = tensor(&xv, m, k, &dev);

    let unit = host_delta_f64(&xv, &av, &bv, m, k, rank, n, 1.0);
    assert_non_degenerate(&unit, "unit-scaling reference");

    for &scaling in &[1.0f64, 2.0, 0.25, -1.0, 8.0] {
        let want = host_delta_f64(&xv, &av, &bv, m, k, rank, n, scaling);
        assert_non_degenerate(&want, &format!("scaling={scaling} reference"));
        let got = flat(&st.delta(0, &x, scaling).unwrap());
        assert_bit_equal(&got, &want, &format!("delta at scaling={scaling}"));

        if scaling != 1.0 {
            let moved = got
                .iter()
                .zip(unit.iter())
                .filter(|(g, u)| (**g as f64) != **u)
                .count();
            assert!(
                moved * 4 > got.len(),
                "scaling={scaling} moved only {moved}/{} entries away from scaling=1.0; `delta` \
                 may be ignoring its scaling argument",
                got.len()
            );
        }
    }
    eprintln!(
        "delta scaling contract: m={m} k={k} rank={rank} n={n}, 5 scaling factors, bit-exact vs \
         f64 host reference"
    );
}

#[test]
fn low_rank_adapter_in_a_wide_slot_pads_with_zeros_and_matches_the_tight_slot() {
    let dev = Device::Cpu;
    let (m, k, rank, n) = (4usize, 16usize, 4usize, 8usize);
    let wide_rank = 16usize;
    let xv = ladder_vec(0x21, m * k);
    let av = ladder_vec(0x22, rank * k);
    let bv = ladder_vec(0x23, n * rank);
    let a = tensor(&av, rank, k, &dev);
    let b = tensor(&bv, n, rank, &dev);
    let x = tensor(&xv, m, k, &dev);

    let tight = stack(1, rank, k, n, &dev);
    tight.set_lora(0, &a, &b).unwrap();
    let wide = stack(1, wide_rank, k, n, &dev);
    wide.set_lora(0, &a, &b).unwrap();

    let scaling = 2.0f64;
    let want = host_delta_f64(&xv, &av, &bv, m, k, rank, n, scaling);
    assert_non_degenerate(&want, "padded-slot reference");

    let d_tight = flat(&tight.delta(0, &x, scaling).unwrap());
    let d_wide = flat(&wide.delta(0, &x, scaling).unwrap());
    assert_bit_equal(&d_tight, &want, "rank-4 slot");
    assert_bit_equal(&d_wide, &want, "rank-4 adapter in a rank-16 slot");

    let padded_a = flat(&wide.slot_a(0).unwrap());
    assert_eq!(padded_a.len(), wide_rank * k);
    let live = &padded_a[..rank * k];
    assert_eq!(
        live,
        av.as_slice(),
        "the live rows of A must survive padding"
    );
    assert!(
        padded_a[rank * k..].iter().all(|v| *v == 0.0),
        "padding rows of A must be exact zero, not stale slot contents"
    );
}

#[test]
fn a_module_absent_from_the_adapter_contributes_exact_zero() {
    let dev = Device::Cpu;
    let (m, k, rank, n) = (3usize, 16usize, 4usize, 8usize);
    let specs = vec![
        LoraModuleSpec::new("q_proj", k, n),
        LoraModuleSpec::new("v_proj", k, n),
    ];
    let mut mgr = LoraSlotManager::new(2, rank, &specs, DType::F32, &dev).unwrap();

    let av = ladder_vec(0x31, rank * k);
    let bv = ladder_vec(0x32, n * rank);
    let mut modules = HashMap::new();
    modules.insert(
        "v_proj".to_string(),
        LoraModuleWeights {
            a: tensor(&av, rank, k, &dev),
            b: tensor(&bv, n, rank, &dev),
        },
    );
    let adapter = LoraAdapter {
        scaling: 1.5,
        modules,
    };
    let slot = mgr.activate(7, &adapter).unwrap();

    let xv = ladder_vec(0x33, m * k);
    let x = tensor(&xv, m, k, &dev);

    let want_v = host_delta_f64(&xv, &av, &bv, m, k, rank, n, adapter.scaling);
    assert_non_degenerate(&want_v, "v_proj reference");
    let got_v = flat(
        &mgr.stack("v_proj")
            .unwrap()
            .delta(slot, &x, adapter.scaling)
            .unwrap(),
    );
    assert_bit_equal(&got_v, &want_v, "v_proj delta");

    let got_q = flat(
        &mgr.stack("q_proj")
            .unwrap()
            .delta(slot, &x, adapter.scaling)
            .unwrap(),
    );
    assert!(
        got_q.iter().all(|v| *v == 0.0),
        "an untrained module must contribute exact zero, got {} nonzero entries",
        got_q.iter().filter(|v| **v != 0.0).count()
    );
}

#[test]
fn reactivation_over_a_dirty_slot_leaves_no_residue() {
    let dev = Device::Cpu;
    let (m, k, n) = (4usize, 16usize, 8usize);
    let max_rank = 8usize;
    let specs = vec![LoraModuleSpec::new("q_proj", k, n)];
    let mut mgr = LoraSlotManager::new(1, max_rank, &specs, DType::F32, &dev).unwrap();

    let big_a = ladder_vec(0x41, max_rank * k);
    let big_b = ladder_vec(0x42, n * max_rank);
    let mut m1 = HashMap::new();
    m1.insert(
        "q_proj".to_string(),
        LoraModuleWeights {
            a: tensor(&big_a, max_rank, k, &dev),
            b: tensor(&big_b, n, max_rank, &dev),
        },
    );
    assert_eq!(
        mgr.activate(
            1,
            &LoraAdapter {
                scaling: 1.0,
                modules: m1
            }
        )
        .unwrap(),
        0
    );

    let small_rank = 2usize;
    let sa = ladder_vec(0x43, small_rank * k);
    let sb = ladder_vec(0x44, n * small_rank);
    let mut m2 = HashMap::new();
    m2.insert(
        "q_proj".to_string(),
        LoraModuleWeights {
            a: tensor(&sa, small_rank, k, &dev),
            b: tensor(&sb, n, small_rank, &dev),
        },
    );
    assert_eq!(mgr.deactivate(1), Some(0));
    assert_eq!(
        mgr.activate(
            2,
            &LoraAdapter {
                scaling: 1.0,
                modules: m2
            }
        )
        .unwrap(),
        0
    );

    let xv = ladder_vec(0x45, m * k);
    let x = tensor(&xv, m, k, &dev);
    let want = host_delta_f64(&xv, &sa, &sb, m, k, small_rank, n, 1.0);
    assert_non_degenerate(&want, "post-reactivation reference");
    let got = flat(&mgr.stack("q_proj").unwrap().delta(0, &x, 1.0).unwrap());
    assert_bit_equal(&got, &want, "reactivated slot");

    let stale = host_delta_f64(&xv, &big_a, &big_b, m, k, max_rank, n, 1.0);
    let differ = want
        .iter()
        .zip(stale.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differ * 4 > want.len(),
        "the two adapters are too similar ({differ}/{} entries differ) for residue to be visible",
        want.len()
    );
}

#[test]
fn lru_evicts_the_oldest_adapter_and_touch_refreshes_it() {
    let dev = Device::Cpu;
    let (k, n, rank) = (8usize, 8usize, 2usize);
    let specs = vec![LoraModuleSpec::new("q_proj", k, n)];
    let mut mgr = LoraSlotManager::new(2, rank, &specs, DType::F32, &dev).unwrap();

    let mk = |seed: u64| -> LoraAdapter {
        let mut modules = HashMap::new();
        modules.insert(
            "q_proj".to_string(),
            LoraModuleWeights {
                a: tensor(&ladder_vec(seed, rank * k), rank, k, &dev),
                b: tensor(&ladder_vec(seed ^ 0xff, n * rank), n, rank, &dev),
            },
        );
        LoraAdapter {
            scaling: 1.0,
            modules,
        }
    };

    assert_eq!(mgr.activate(10, &mk(0x51)).unwrap(), 0);
    assert_eq!(mgr.activate(11, &mk(0x52)).unwrap(), 1);
    assert_eq!(mgr.activate(10, &mk(0x51)).unwrap(), 0);
    let slot = mgr.activate(12, &mk(0x53)).unwrap();
    assert_eq!(slot, 1, "the touched adapter must not be the one evicted");
    assert_eq!(mgr.slot_of(10), Some(0), "10 was touched and must survive");
    assert_eq!(mgr.slot_of(11), None, "11 was the least recently used");
    assert_eq!(mgr.slot_id(1), Some(12));
}

#[test]
fn set_lora_rejects_overflow_and_rank_mismatch_rather_than_truncating() {
    let dev = Device::Cpu;
    let (k, n, max_rank) = (16usize, 8usize, 4usize);
    let st = stack(1, max_rank, k, n, &dev);
    let ok_a = tensor(&ladder_vec(0x61, max_rank * k), max_rank, k, &dev);
    let ok_b = tensor(&ladder_vec(0x62, n * max_rank), n, max_rank, &dev);
    st.set_lora(0, &ok_a, &ok_b).unwrap();

    let wide_a = tensor(&ladder_vec(0x63, (max_rank + 1) * k), max_rank + 1, k, &dev);
    let wide_b = tensor(&ladder_vec(0x64, n * (max_rank + 1)), n, max_rank + 1, &dev);
    assert!(
        st.set_lora(0, &wide_a, &wide_b).is_err(),
        "rank {} exceeds max_rank {max_rank} and must be rejected, not silently truncated",
        max_rank + 1
    );

    let long_a = tensor(&ladder_vec(0x65, max_rank * (k + 8)), max_rank, k + 8, &dev);
    assert!(
        st.set_lora(0, &long_a, &ok_b).is_err(),
        "in_features overflow must be rejected"
    );

    let mismatched_b = tensor(&ladder_vec(0x66, n * 2), n, 2, &dev);
    assert!(
        st.set_lora(0, &ok_a, &mismatched_b).is_err(),
        "A rank {max_rank} against B rank 2 must be rejected"
    );

    assert!(
        st.set_lora(1, &ok_a, &ok_b).is_err(),
        "slot index 1 is out of range for max_loras=1"
    );
    assert!(st
        .delta(1, &tensor(&ladder_vec(0x67, k), 1, k, &dev), 1.0)
        .is_err());
}
