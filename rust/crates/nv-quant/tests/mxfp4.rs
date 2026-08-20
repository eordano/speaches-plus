use nv_quant::mxfp4::{
    block_scale_byte, cpu_mxfp4_matmul_weight_row, decode_e2m1, decode_e8m0, dequantize_block,
    encode_e8m0, exp2_i32, ilog2_f32, quantize_block, Mxfp4Tensor, BLOCK_BYTES, BLOCK_SIZE,
    E8M0_BIAS, E8M0_NAN,
};

fn is_exact_power_of_two(v: f32) -> bool {
    if !v.is_finite() || v <= 0.0 {
        return false;
    }
    v.to_bits().count_ones() == 1 || (v.to_bits() & 0x007F_FFFF) == 0
}

#[test]
fn e8m0_every_byte_is_exact_power_of_two_and_reencodes() {
    for byte in 0u8..=254 {
        let v = decode_e8m0(byte);
        assert!(v.is_finite() && v > 0.0, "byte {byte} decoded to {v}");
        assert!(
            is_exact_power_of_two(v),
            "byte {byte} decoded to {v} which is not a power of two"
        );
        assert_eq!(v, exp2_i32(byte as i32 - E8M0_BIAS));
        assert_eq!(
            encode_e8m0(v),
            byte,
            "re-encode of 2^{}",
            byte as i32 - E8M0_BIAS
        );
        assert_eq!(ilog2_f32(v), byte as i32 - E8M0_BIAS);
    }
    assert!(decode_e8m0(E8M0_NAN).is_nan());
    assert_eq!(encode_e8m0(f32::NAN), E8M0_NAN);
    assert_eq!(encode_e8m0(0.0), E8M0_NAN);
    assert_eq!(encode_e8m0(-1.0), E8M0_NAN);
    assert_eq!(decode_e8m0(E8M0_BIAS as u8), 1.0);
    assert_eq!(decode_e8m0(0), 2f32.powi(-127));
}

#[test]
fn e8m0_encode_is_floor_not_nearest_and_has_no_tie_hazard() {
    for byte in 0u8..=253 {
        let v = decode_e8m0(byte);
        assert_eq!(encode_e8m0(v * 1.5), byte, "1.5 * 2^e must floor to e");
        if byte >= 1 {
            assert_eq!(encode_e8m0(v * 1.9999999), byte);
        }
        assert_eq!(encode_e8m0(v * 2.0), byte + 1);
    }
}

fn grid_block(e: i32) -> Vec<f32> {
    let scale = exp2_i32(e);
    let mut vals = Vec::with_capacity(BLOCK_SIZE);
    for code in 0u8..16 {
        vals.push(decode_e2m1(code) * scale);
    }
    for code in (0u8..16).rev() {
        vals.push(decode_e2m1(code) * scale);
    }
    vals
}

#[test]
fn e2m1_grid_values_survive_roundtrip_bit_exactly() {
    for e in (-100..=100).step_by(7).chain([-127i32, -126, 100]) {
        let vals = grid_block(e);
        let (packed, scale_byte) = quantize_block(&vals);
        assert_eq!(scale_byte as i32 - E8M0_BIAS, e, "scale exponent for e={e}");
        let deq = dequantize_block(&packed, scale_byte);
        for (i, (a, b)) in vals.iter().zip(deq.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "e={e} elem {i}: {a} != {b} after roundtrip"
            );
        }
    }
}

#[test]
fn block_scales_are_always_exact_powers_of_two() {
    let mut rng = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32 / (1u64 << 23) as f32) - 0.5
    };
    for amp_exp in [-9i32, -3, 0, 5, 30, -80] {
        let amp = exp2_i32(amp_exp);
        for _ in 0..100 {
            let vals: Vec<f32> = (0..BLOCK_SIZE).map(|_| next() * 2.0 * amp).collect();
            let (_, scale_byte) = quantize_block(&vals);
            assert_ne!(scale_byte, E8M0_NAN);
            let scale = decode_e8m0(scale_byte);
            assert!(
                is_exact_power_of_two(scale),
                "scale {scale} (byte {scale_byte}) is not a power of two"
            );
            let amax = vals.iter().fold(0f32, |a, b| a.max(b.abs()));
            assert_eq!(scale_byte, block_scale_byte(amax));
        }
    }
}

#[test]
fn tie_heavy_block_quantizes_deterministically_toward_lower_code() {
    for e in [-6i32, 0, 9] {
        let s = exp2_i32(e);
        let mut vals = vec![
            0.75 * s,
            1.75 * s,
            3.5 * s,
            -0.75 * s,
            -1.75 * s,
            -3.5 * s,
            0.25 * s,
            2.5 * s,
            5.0 * s,
            -0.25 * s,
            -2.5 * s,
            -5.0 * s,
            6.0 * s,
            -6.0 * s,
        ];
        while vals.len() < BLOCK_SIZE {
            vals.push(0.75 * s);
        }
        let (packed_a, scale_a) = quantize_block(&vals);
        let (packed_b, scale_b) = quantize_block(&vals);
        assert_eq!(packed_a, packed_b);
        assert_eq!(scale_a, scale_b);
        assert_eq!(scale_a as i32 - E8M0_BIAS, e);
        let deq = dequantize_block(&packed_a, scale_a);
        let expect = [
            0.5 * s,
            1.5 * s,
            3.0 * s,
            -0.5 * s,
            -1.5 * s,
            -3.0 * s,
            0.0,
            2.0 * s,
            4.0 * s,
            -0.0,
            -2.0 * s,
            -4.0 * s,
            6.0 * s,
            -6.0 * s,
        ];
        for (i, want) in expect.iter().enumerate() {
            assert_eq!(
                deq[i].to_bits(),
                want.to_bits(),
                "e={e} elem {i}: tie broke to {} not {want}",
                deq[i]
            );
        }
        for v in &deq[expect.len()..] {
            assert_eq!(v.to_bits(), (0.5 * s).to_bits());
        }
    }
}

#[test]
fn zero_block_is_exact_and_scale_is_one() {
    let vals = vec![0.0f32; BLOCK_SIZE];
    let (packed, scale_byte) = quantize_block(&vals);
    assert_eq!(scale_byte, E8M0_BIAS as u8);
    assert_eq!(packed, [0u8; BLOCK_BYTES]);
    let deq = dequantize_block(&packed, scale_byte);
    assert!(deq.iter().all(|&v| v == 0.0));
}

#[test]
fn storage_is_4_25_bits_per_value_with_no_global_scale() {
    assert_eq!(BLOCK_SIZE, 32);
    assert_eq!(BLOCK_BYTES, 16);
    let bits = (BLOCK_BYTES + 1) * 8;
    assert_eq!(bits as f32 / BLOCK_SIZE as f32, 4.25);
    let rows: Vec<Vec<f32>> = (0..3).map(|_| vec![1.0f32; 64]).collect();
    let t = Mxfp4Tensor::quantize_rows(&rows);
    assert_eq!(t.data.len(), 3 * 64 / 2);
    assert_eq!(t.scales.len(), 3 * 2);
}

#[test]
fn roundtrip_error_is_bounded_by_grid_and_clamp() {
    let mut rng = 0x12345678u64;
    let mut next = move || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 33) as f64 / (1u64 << 31) as f64) as f32 - 0.5
    };
    for amp in [0.001f32, 0.05, 1.0, 37.0, 4000.0] {
        for _ in 0..50 {
            let vals: Vec<f32> = (0..BLOCK_SIZE).map(|_| next() * 2.0 * amp).collect();
            let (packed, scale_byte) = quantize_block(&vals);
            let deq = dequantize_block(&packed, scale_byte);
            let scale = decode_e8m0(scale_byte);
            for (a, b) in vals.iter().zip(deq.iter()) {
                let err = (a - b).abs();
                assert!(
                    err <= scale * 2.0,
                    "|{a} - {b}| = {err} > clamp bound {}",
                    scale * 2.0
                );
                let tier = if a.abs() <= scale * 2.0 {
                    0.25
                } else if a.abs() <= scale * 4.0 {
                    0.5
                } else {
                    2.0
                };
                assert!(
                    err <= scale * tier + 1e-12,
                    "|{a} - {b}| = {err} > tier bound {}",
                    scale * tier
                );
            }
        }
    }
}

#[test]
fn tensor_roundtrip_and_gpt_oss_layout_are_identical() {
    let rows: Vec<Vec<f32>> = (0..4)
        .map(|r| {
            (0..96)
                .map(|c| (((r * 37 + c * 13) % 23) as f32 - 11.0) * 0.03)
                .collect()
        })
        .collect();
    let t = Mxfp4Tensor::quantize_rows(&rows);
    let t2 = Mxfp4Tensor::from_gpt_oss_row_major(&t.data, &t.scales, 4, 96);
    let d1 = t.dequantize();
    let d2 = t2.dequantize();
    for (a, b) in d1.iter().flatten().zip(d2.iter().flatten()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let grid: Vec<Vec<f32>> = vec![grid_block(-4), grid_block(2)];
    let tg = Mxfp4Tensor::quantize_rows(&grid);
    let dg = tg.dequantize();
    for (row, orig) in dg.iter().zip(grid.iter()) {
        for (a, b) in row.iter().zip(orig.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

#[test]
fn cpu_reference_matmul_matches_dense_on_grid_inputs() {
    let w_rows: Vec<Vec<f32>> = vec![grid_block(0), grid_block(-2), grid_block(3)];
    let w = Mxfp4Tensor::quantize_rows(&w_rows);
    let x: Vec<f32> = (0..2 * 32).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
    let got = cpu_mxfp4_matmul_weight_row(&x, &w, 2);
    for i in 0..2 {
        for j in 0..3 {
            let mut acc = 0f32;
            for p in 0..32 {
                acc += x[i * 32 + p] * w_rows[j][p];
            }
            assert_eq!(got[i * 3 + j], acc, "({i},{j})");
        }
    }
}

fn gpt_oss_snapshot() -> Option<std::path::PathBuf> {
    let mut roots: Vec<std::path::PathBuf> = Vec::new();
    if let Some(h) = std::env::var_os("HF_HUB_CACHE") {
        roots.push(std::path::PathBuf::from(h));
    }
    if let Some(h) = std::env::var_os("HF_HOME") {
        roots.push(std::path::PathBuf::from(h).join("hub"));
    }
    if let Some(h) = std::env::var_os("HOME") {
        roots.push(std::path::PathBuf::from(h).join(".cache/huggingface/hub"));
    }
    for root in roots {
        let snaps = root.join("models--openai--gpt-oss-20b/snapshots");
        let mut cands: Vec<std::path::PathBuf> = std::fs::read_dir(&snaps)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.join("config.json").exists()
                    && p.join("model-00000-of-00002.safetensors").exists()
            })
            .collect();
        cands.sort();
        if let Some(p) = cands.pop() {
            return Some(p);
        }
    }
    None
}

#[test]
#[ignore]
fn gpt_oss_20b_checkpoint_layout_is_mxfp4_row_major() {
    use std::io::{Read, Seek, SeekFrom};
    let Some(snap) = gpt_oss_snapshot() else {
        if std::env::var("NV_GPT_OSS_ALLOW_SKIP").as_deref() == Ok("1") {
            eprintln!(
                "SKIP (NV_GPT_OSS_ALLOW_SKIP=1): no openai/gpt-oss-20b snapshot with \
                 model-00000-of-00002.safetensors. This is a SKIP, not a pass -- the MXFP4 \
                 checkpoint layout claim was not checked."
            );
            return;
        }
        panic!(
            "gpt_oss_20b_checkpoint_layout_is_mxfp4_row_major: no openai/gpt-oss-20b snapshot \
             carrying config.json + model-00000-of-00002.safetensors under HF_HUB_CACHE, HF_HOME \
             or $HOME/.cache/huggingface/hub. This test reads safetensors HEADERS only -- no GPU, \
             no dequant -- so a present checkpoint is the whole precondition. The old \
             NV_GPT_OSS_TEST=1 gate returned quietly and printed a pass; this one refuses to. Set \
             NV_GPT_OSS_ALLOW_SKIP=1 to skip on purpose."
        );
    };
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(snap.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg["quantization_config"]["quant_method"], "mxfp4");
    let mut f = std::fs::File::open(snap.join("model-00000-of-00002.safetensors")).unwrap();
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8).unwrap();
    let hlen = u64::from_le_bytes(len8);
    let mut hbuf = vec![0u8; hlen as usize];
    f.read_exact(&mut hbuf).unwrap();
    let hdr: serde_json::Value = serde_json::from_slice(&hbuf).unwrap();
    let base = 8 + hlen;
    let blocks = &hdr["model.layers.0.mlp.experts.gate_up_proj_blocks"];
    let scales = &hdr["model.layers.0.mlp.experts.gate_up_proj_scales"];
    assert_eq!(blocks["dtype"], "U8");
    assert_eq!(scales["dtype"], "U8");
    assert_eq!(
        blocks["shape"].as_array().unwrap(),
        &vec![
            serde_json::json!(32),
            serde_json::json!(5760),
            serde_json::json!(90),
            serde_json::json!(16)
        ]
    );
    assert_eq!(
        scales["shape"].as_array().unwrap(),
        &vec![
            serde_json::json!(32),
            serde_json::json!(5760),
            serde_json::json!(90)
        ]
    );
    let rows = 5760usize;
    let cols = 90 * BLOCK_SIZE;
    let bstart = blocks["data_offsets"][0].as_u64().unwrap();
    let sstart = scales["data_offsets"][0].as_u64().unwrap();
    let mut bdata = vec![0u8; rows * (cols / 2)];
    f.seek(SeekFrom::Start(base + bstart)).unwrap();
    f.read_exact(&mut bdata).unwrap();
    let mut sdata = vec![0u8; rows * (cols / BLOCK_SIZE)];
    f.seek(SeekFrom::Start(base + sstart)).unwrap();
    f.read_exact(&mut sdata).unwrap();
    let mut smin = 255u8;
    let mut smax = 0u8;
    for &s in &sdata {
        assert_ne!(s, E8M0_NAN, "checkpoint contains an E8M0 NaN scale");
        smin = smin.min(s);
        smax = smax.max(s);
    }
    let t = Mxfp4Tensor::from_gpt_oss_row_major(&bdata, &sdata, rows, cols);
    let deq = t.dequantize();
    let mut amax = 0f32;
    let mut nonzero = 0usize;
    for row in &deq {
        for &v in row {
            assert!(v.is_finite());
            amax = amax.max(v.abs());
            if v != 0.0 {
                nonzero += 1;
            }
        }
    }
    println!(
        "expert0 gate_up: rows={rows} cols={cols} scale_exp=[{}..{}] amax={amax} nonzero={nonzero}/{}",
        smin as i32 - E8M0_BIAS,
        smax as i32 - E8M0_BIAS,
        rows * cols
    );
    assert!(nonzero > rows * cols / 2);
    assert!(amax > 0.0 && amax < 32.0);
}
