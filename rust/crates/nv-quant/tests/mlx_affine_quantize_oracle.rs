use nv_quant::mlx_affine::{dequantize_mlx_affine_linear, quantize_mlx_affine_linear};

fn vecf(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn vecu(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

#[test]
fn quantize_matches_mx_quantize_bit_for_bit_on_every_oracle_case() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("mlx_affine_quantize_oracle.json")).expect("oracle json");
    let cases = cases.as_object().unwrap();
    assert_eq!(cases.len(), 6, "all six oracle cases must be present");
    for (name, case) in cases {
        let arr = case.as_array().unwrap();
        let (w, want_q, want_s, want_b) =
            (vecf(&arr[0]), vecu(&arr[1]), vecf(&arr[2]), vecf(&arr[3]));
        assert_eq!(
            w.len(),
            want_s.len() * 64,
            "case {name}: oracle shapes disagree"
        );
        let flat_k_equals_row_major_because_groups_never_straddle_rows = w.len();
        let (got_q, got_s, got_b) = quantize_mlx_affine_linear(
            &w,
            1,
            flat_k_equals_row_major_because_groups_never_straddle_rows,
            64,
        );
        assert_eq!(
            got_q, want_q,
            "case {name}: packed codes differ from mx.quantize"
        );
        assert_eq!(
            got_s, want_s,
            "case {name}: stored scales differ from mx.quantize"
        );
        assert_eq!(
            got_b, want_b,
            "case {name}: stored biases differ from mx.quantize"
        );
        let deq = dequantize_mlx_affine_linear(&got_q, &got_s, &got_b, 1, w.len(), 64);
        for (i, (d, orig)) in deq.iter().zip(w.iter()).enumerate() {
            let half_range = got_s[i / 64].abs() * 8.0 + 1e-6;
            assert!(
                (d - orig).abs() <= half_range,
                "case {name} elem {i}: round-trip {d} vs {orig} beyond the code grid's half-range"
            );
        }
    }
}
