use half::bf16;
use nv_weights::{DType, Device, WeightLoader};
use safetensors::tensor::{Dtype as StDtype, TensorView};

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

fn lcg_f32(rng: &mut Lcg, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let r = (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
        v.push(r);
    }
    v
}

fn to_bf16(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|x| bf16::from_f32(*x)).collect()
}

fn bf16_bytes(v: &[bf16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[test]
fn roundtrip_bf16_and_f32_with_dtype_casts() {
    let mut rng = Lcg::new(0xC0FFEE);

    let weight_f32 = lcg_f32(&mut rng, 16 * 32);
    let weight_bf16 = to_bf16(&weight_f32);
    let bias_f32 = lcg_f32(&mut rng, 8);

    let weight_bytes = bf16_bytes(&weight_bf16);
    let bias_bytes = f32_bytes(&bias_f32);

    let tv_w = TensorView::new(StDtype::BF16, vec![16, 32], &weight_bytes).unwrap();
    let tv_b = TensorView::new(StDtype::F32, vec![8], &bias_bytes).unwrap();

    let dir = std::env::temp_dir().join(format!("nv-weights-roundtrip-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file_path = dir.join("model.safetensors");

    safetensors::serialize_to_file(
        vec![("weight".to_string(), tv_w), ("bias".to_string(), tv_b)],
        None,
        &file_path,
    )
    .unwrap();

    let device = Device::Cpu;
    let loader = WeightLoader::open_file(&file_path, &device).unwrap();

    let mut names = loader.names();
    names.sort();
    assert_eq!(names, vec!["bias".to_string(), "weight".to_string()]);
    assert!(loader.has("weight"));
    assert!(loader.has("bias"));
    assert!(!loader.has("missing"));
    assert_eq!(loader.shape_of("weight"), Some(vec![16, 32]));
    assert_eq!(loader.shape_of("bias"), Some(vec![8]));
    assert_eq!(loader.dtype_of("weight"), Some(DType::BF16));
    assert_eq!(loader.dtype_of("bias"), Some(DType::F32));

    let w_bf16 = loader.get("weight", DType::BF16).unwrap();
    assert_eq!(w_bf16.dims(), &[16, 32]);
    assert_eq!(w_bf16.dtype(), DType::BF16);
    let w_back: Vec<Vec<bf16>> = w_bf16.to_vec2().unwrap();
    for r in 0..16 {
        for c in 0..32 {
            assert_eq!(w_back[r][c], weight_bf16[r * 32 + c]);
        }
    }

    let w_f32 = loader.get("weight", DType::F32).unwrap();
    assert_eq!(w_f32.dims(), &[16, 32]);
    assert_eq!(w_f32.dtype(), DType::F32);
    let w_f32_back: Vec<Vec<f32>> = w_f32.to_vec2().unwrap();
    for r in 0..16 {
        for c in 0..32 {
            let expected = weight_bf16[r * 32 + c].to_f32();
            assert!((w_f32_back[r][c] - expected).abs() <= 1e-6);
        }
    }

    let b_f32 = loader.get("bias", DType::F32).unwrap();
    assert_eq!(b_f32.dims(), &[8]);
    assert_eq!(b_f32.dtype(), DType::F32);
    let b_back: Vec<f32> = b_f32.to_vec1().unwrap();
    for i in 0..8 {
        assert!((b_back[i] - bias_f32[i]).abs() <= 1e-6);
    }

    std::fs::remove_dir_all(&dir).ok();
}
