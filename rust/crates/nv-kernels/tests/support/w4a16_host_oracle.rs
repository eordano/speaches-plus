const NIBBLES_PER_WORD: usize = 8;
const QUANT_OFFSET: i32 = 8;

const CROSS_DECOMPOSITION_REL_TOL_BOUNDS_F32_VS_F64_ACCUMULATION: f64 = 3e-5;

#[allow(dead_code)]
const KERNEL_VS_HOST_ORACLE_REL_TOL_PINNED_BY_GEMV_W4A16_SUITES: f32 = 1e-2;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / u32::MAX as f64) as f32 * 2.0 - 1.0
    }
}

fn gen_inputs(n: usize, k: usize, gs: usize, seed: u64) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
    let mut rng = Lcg::new(seed);
    let packed: Vec<u32> = (0..n * k / NIBBLES_PER_WORD).map(|_| rng.next_u32()).collect();
    let scales: Vec<u16> = (0..n * (k / gs))
        .map(|_| bf16::from_f32(0.005 + 0.01 * rng.next_f32().abs()).to_bits())
        .collect();
    let x: Vec<u16> = (0..k)
        .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
        .collect();
    (packed, scales, x)
}

fn ref_row_major(
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
    gs: usize,
) -> Vec<f32> {
    let xf: Vec<f32> = x.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
    let sf: Vec<f32> = scales.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
    let kw = k / NIBBLES_PER_WORD;
    let kg = k / gs;
    let mut y = vec![0f32; n];
    for row in 0..n {
        let mut acc = 0f64;
        for kk in 0..k {
            let word = packed[row * kw + kk / NIBBLES_PER_WORD];
            let q = ((word >> (4 * (kk % NIBBLES_PER_WORD))) & 0xF) as i32 - QUANT_OFFSET;
            acc += (q as f32 * sf[row * kg + kk / gs] * xf[kk]) as f64;
        }
        y[row] = acc as f32;
    }
    y
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlantedBug {
    None,
    ScaleIndexOffByOneGroup,
    NibbleOrderReversed,
    MissingSignOffset,
}

fn ref_group_major_f64(
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
    gs: usize,
    bug: PlantedBug,
) -> Vec<f32> {
    let kg = k / gs;
    let kw = k / NIBBLES_PER_WORD;
    let mut y = vec![0f32; n];
    for row in 0..n {
        let mut row_acc = 0f64;
        for g in 0..kg {
            let scale_group = match bug {
                PlantedBug::ScaleIndexOffByOneGroup => (g + 1) % kg,
                _ => g,
            };
            let s = bf16::from_bits(scales[row * kg + scale_group]).to_f64();
            let mut group_acc = 0f64;
            for j in 0..gs {
                let kk = g * gs + j;
                let word = packed[row * kw + kk / NIBBLES_PER_WORD];
                let lane = match bug {
                    PlantedBug::NibbleOrderReversed => {
                        NIBBLES_PER_WORD - 1 - (kk % NIBBLES_PER_WORD)
                    }
                    _ => kk % NIBBLES_PER_WORD,
                };
                let raw = ((word >> (4 * lane)) & 0xF) as i32;
                let q = match bug {
                    PlantedBug::MissingSignOffset => raw,
                    _ => raw - QUANT_OFFSET,
                };
                group_acc += q as f64 * bf16::from_bits(x[kk]).to_f64();
            }
            row_acc += s * group_acc;
        }
        y[row] = row_acc as f32;
    }
    y
}

fn max_rel_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&p, &q)| {
            let denom = p.abs().max(q.abs()).max(1e-3) as f64;
            ((p - q).abs() as f64) / denom
        })
        .fold(0.0, f64::max)
}

#[allow(dead_code)]
fn max_rel_bf16_output(got: &[u16], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "compared buffers differ in length");
    assert!(!got.is_empty(), "nothing was compared");
    assert!(
        want.iter().any(|v| v.abs() > 1e-3),
        "the reference is all zeros; the comparison would be vacuous"
    );
    let mut worst = 0f32;
    for (i, (&g, &e)) in got.iter().zip(want.iter()).enumerate() {
        let gf = bf16::from_bits(g).to_f32();
        assert!(
            gf.is_finite(),
            "row {i}: kernel output is {gf} -- the row was never written, or the kernel \
             produced a non-finite value"
        );
        let rel = (gf - e).abs() / e.abs().max(0.5);
        if rel > worst {
            worst = rel;
        }
    }
    worst
}
