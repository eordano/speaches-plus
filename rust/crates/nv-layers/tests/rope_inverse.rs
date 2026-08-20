use candle_core::{DType, Device, Tensor};
use nv_layers::rope::{Rope, RopeConfig, RopeKind};

const HEAD_DIM: usize = 512;
const BASE: f32 = 1_000_000.0;
const PARTIAL: f32 = 0.25;
const MAX_SEQ: usize = 262144;

const FORWARD_TABLE_DRIFT_CEILING: f32 = 2e-2;

fn full_layer_rope(device: &Device) -> Rope {
    let half = HEAD_DIM / 2;
    let angles = ((PARTIAL * HEAD_DIM as f32 / 2.0) as usize).min(half);
    let mut inv_freq = vec![0f32; half];
    for (i, f) in inv_freq[..angles].iter_mut().enumerate() {
        *f = 1.0 / BASE.powf((i as f32 * 2.0) / (HEAD_DIM as f32));
    }
    Rope::from_inv_freq(
        RopeConfig {
            head_dim: HEAD_DIM,
            max_seq_len: MAX_SEQ,
            base: BASE,
            kind: RopeKind::Standard,
        },
        &inv_freq,
        device,
    )
    .expect("rope")
}

fn conjugate(rope: &Rope, x: &Tensor, positions: &Tensor) -> Tensor {
    let dims = x.dims().to_vec();
    let n_heads = dims[dims.len() - 2];
    let tokens: usize = dims[..dims.len() - 2].iter().product();
    let half = HEAD_DIM / 2;

    let xf = x
        .to_dtype(DType::F32)
        .unwrap()
        .reshape((tokens, n_heads, HEAD_DIM))
        .unwrap();
    let idx = positions.to_dtype(DType::U32).unwrap();
    let cos = rope
        .cos()
        .index_select(&idx, 0)
        .unwrap()
        .unsqueeze(1)
        .unwrap();
    let sin = rope
        .sin()
        .index_select(&idx, 0)
        .unwrap()
        .unsqueeze(1)
        .unwrap();

    let lo = xf.narrow(2, 0, half).unwrap();
    let hi = xf.narrow(2, half, half).unwrap();
    let out_lo = lo
        .broadcast_mul(&cos)
        .unwrap()
        .add(&hi.broadcast_mul(&sin).unwrap())
        .unwrap();
    let out_hi = hi
        .broadcast_mul(&cos)
        .unwrap()
        .sub(&lo.broadcast_mul(&sin).unwrap())
        .unwrap();
    Tensor::cat(&[&out_lo, &out_hi], 2)
        .unwrap()
        .reshape(dims)
        .unwrap()
}

fn rel_rms(a: &Tensor, b: &Tensor) -> f32 {
    let e: Vec<f32> = a.sub(b).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let r: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
    let num: f64 = e.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    let den: f64 = r.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    (num.sqrt() / den.sqrt()) as f32
}

#[test]
fn the_conjugate_rotation_inverts_rope_at_every_context_depth() {
    let device = Device::Cpu;
    let rope = full_layer_rope(&device);

    let pos: Vec<u32> = vec![0, 1, 1024, 65535, 131072, 262143];
    let tokens = pos.len();
    let heads = 4usize;
    let positions = Tensor::from_vec(pos.clone(), tokens, &device).unwrap();

    let k = Tensor::rand(-2f32, 2f32, (tokens, heads, HEAD_DIM), &device).unwrap();
    let (_, k_rot) = rope.apply(&k, &k, &positions).unwrap();
    let back = conjugate(&rope, &k_rot, &positions);

    let rel = rel_rms(&back, &k);
    eprintln!("[rope-inverse] round-trip rel-rms over 6 depths: {rel:e}");
    assert!(
        rel < 1e-5,
        "conjugate rotation did not invert RoPE: rel-rms {rel:e}. Deriving V from \
         the cached K depends on this being a rounding-level quantity."
    );

    for (i, p) in pos.iter().enumerate() {
        let a = back.narrow(0, i, 1).unwrap();
        let b = k.narrow(0, i, 1).unwrap();
        let r = rel_rms(&a, &b);
        assert!(r < 1e-5, "position {p}: round-trip rel-rms {r:e}");
    }
}

#[test]
fn the_round_trip_does_not_inherit_the_table_angle_error() {
    let device = Device::Cpu;
    let rope = full_layer_rope(&device);
    let p = 262143u32;
    let positions = Tensor::from_vec(vec![p], 1, &device).unwrap();
    let k = Tensor::rand(-2f32, 2f32, (1usize, 1usize, HEAD_DIM), &device).unwrap();

    let (_, k_rot) = rope.apply(&k, &k, &positions).unwrap();
    let back = conjugate(&rope, &k_rot, &positions);
    let round_trip = rel_rms(&back, &k);

    let kv: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
    let half = HEAD_DIM / 2;
    let angles = ((PARTIAL * HEAD_DIM as f32 / 2.0) as usize).min(half);
    let mut exact = vec![0f32; HEAD_DIM];
    let mut conjugated = vec![0f32; HEAD_DIM];
    for i in 0..half {
        let (c, s) = if i < angles {
            let inv = 1.0f64 / (BASE as f64).powf((i as f64 * 2.0) / (HEAD_DIM as f64));
            let th = p as f64 * inv;
            (th.cos(), th.sin())
        } else {
            (1.0, 0.0)
        };
        let (lo, hi) = (kv[i] as f64, kv[half + i] as f64);
        exact[i] = (lo * c - hi * s) as f32;
        exact[half + i] = (lo * s + hi * c) as f32;
        conjugated[i] = (lo * c + hi * s) as f32;
        conjugated[half + i] = (hi * c - lo * s) as f32;
    }
    let exact = Tensor::from_vec(exact, (1usize, 1usize, HEAD_DIM), &device).unwrap();
    let conjugated = Tensor::from_vec(conjugated, (1usize, 1usize, HEAD_DIM), &device).unwrap();
    let forward = rel_rms(&k_rot, &exact);
    let sign_flip = rel_rms(&conjugated, &exact);

    eprintln!(
        "[rope-inverse] at position {p}: forward-vs-f64 {forward:e}, round-trip {round_trip:e}, \
         conjugated-table probe {sign_flip:e}"
    );
    assert!(
        forward < FORWARD_TABLE_DRIFT_CEILING,
        "forward-vs-f64 is {forward:e}, above the {FORWARD_TABLE_DRIFT_CEILING:e} table-drift \
         ceiling, which the shipped f32 table clears by 8x at 2.35e-3. The two assertions after \
         this one cannot catch a wrong table -- the conjugate cancels any angle error, and \
         dividing by {forward:e} only gets more permissive as the forward path degrades -- so \
         this bound is the file's ONLY gate on the shipped cos/sin entries actually encoding \
         the RoPE angles."
    );
    assert!(
        sign_flip > FORWARD_TABLE_DRIFT_CEILING * 10.0,
        "the ceiling above cannot fail: rotating by the CONJUGATE of the true angle reads \
         {sign_flip:e}, which does not clear {:e}, so this fixture cannot separate a correct \
         table from a sign-flipped one",
        FORWARD_TABLE_DRIFT_CEILING * 10.0
    );
    assert!(
        round_trip < forward / 10.0 || forward < 1e-6,
        "round-trip {round_trip:e} is not decisively smaller than the forward \
         table error {forward:e}; the cancellation argument no longer holds"
    );
    assert!(round_trip < 1e-5, "round-trip rel-rms {round_trip:e}");
}
