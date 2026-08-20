#[allow(dead_code)]
const SET_VAR_IS_ONLY_SOUND_HERE_BECAUSE_THIS_BINARY_HOLDS_EXACTLY_ONE_TEST_DO_NOT_MERGE_INTO_ROPE_TABLE_PRECISION: () = ();

use candle_core::{DType, Device};
use nv_layers::rope::{
    build_rope_tables_f32, build_rope_tables_f64, rope_table_precision_from_env, Rope, RopeConfig,
    RopeKind, RopeTablePrecision,
};

fn inv_freq_f64(dim: usize, base: f64) -> Vec<f64> {
    (0..dim / 2)
        .map(|i| 1.0 / base.powf((i as f64 * 2.0) / dim as f64))
        .collect()
}

fn cfg(head_dim: usize, rows: usize, base: f32) -> RopeConfig {
    RopeConfig {
        head_dim,
        max_seq_len: rows,
        base,
        kind: RopeKind::Standard,
    }
}

fn table_of(r: &Rope) -> (Vec<f32>, Vec<f32>) {
    (
        r.cos()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        r.sin()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
    )
}

#[test]
fn env_knob_default_off_and_wired() {
    let dev = Device::Cpu;
    let dim = 128usize;
    let rows = 4096usize;
    let base = 10000.0f32;
    let inv64 = inv_freq_f64(dim, base as f64);
    let inv32: Vec<f32> = inv64.iter().map(|v| *v as f32).collect();

    assert_eq!(
        std::env::var("NV_ROPE_TABLE").ok(),
        None,
        "this test owns NV_ROPE_TABLE; it must start unset"
    );
    assert_eq!(rope_table_precision_from_env(), RopeTablePrecision::F32);

    let legacy = build_rope_tables_f32(&inv32, rows);
    let default_rope = Rope::from_inv_freq(cfg(dim, rows, base), &inv32, &dev).unwrap();
    let (dc, ds) = table_of(&default_rope);
    assert_eq!(
        dc, legacy.0,
        "default path must be bit-identical to the shipped f32 build"
    );
    assert_eq!(ds, legacy.1);

    let default_new = Rope::new(cfg(dim, rows, base), &dev).unwrap();
    let (nc, _) = table_of(&default_new);
    let legacy_new_inv: Vec<f32> = (0..dim / 2)
        .map(|i| 1.0 / base.powf((i as f32 * 2.0) / (dim as f32)))
        .collect();
    assert_eq!(
        nc,
        build_rope_tables_f32(&legacy_new_inv, rows).0,
        "Rope::new default must keep the f32 powf inv_freq and f32 product"
    );

    std::env::set_var("NV_ROPE_TABLE", "f64");
    assert_eq!(rope_table_precision_from_env(), RopeTablePrecision::F64);
    let wide = Rope::from_inv_freq(cfg(dim, rows, base), &inv32, &dev).unwrap();
    let (wc, ws) = table_of(&wide);

    let widened: Vec<f64> = inv32.iter().map(|v| *v as f64).collect();
    let expect = build_rope_tables_f64(&widened, rows);
    assert_eq!(wc, expect.0, "f64 arm must round the f64 build once to f32");
    assert_eq!(ws, expect.1);
    let changed = wc.iter().zip(dc.iter()).filter(|(a, b)| a != b).count();
    println!(
        "NV_ROPE_TABLE=f64 changed {changed}/{} cos entries over rows=0..{rows}",
        wc.len()
    );
    assert!(changed > 0, "the knob must actually change the table");

    let wide_new = Rope::new(cfg(dim, rows, base), &dev).unwrap();
    let (wnc, _) = table_of(&wide_new);
    assert_eq!(
        wnc,
        build_rope_tables_f64(&inv64, rows).0,
        "Rope::new f64 arm must build inv_freq in f64 too"
    );

    let b2 = Rope::from_inv_freq_f64(cfg(dim, rows, base), &inv64, &dev).unwrap();
    assert_eq!(
        table_of(&b2).0,
        build_rope_tables_f64(&inv64, rows).0,
        "from_inv_freq_f64 is arm B2: exact f64 inv_freq all the way through"
    );

    std::env::set_var("NV_ROPE_TABLE", "f32");
    assert_eq!(rope_table_precision_from_env(), RopeTablePrecision::F32);
    let back = Rope::from_inv_freq(cfg(dim, rows, base), &inv32, &dev).unwrap();
    assert_eq!(table_of(&back).0, dc, "explicit f32 must equal the default");
    std::env::remove_var("NV_ROPE_TABLE");
}
