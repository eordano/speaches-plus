pub const DETRNG_AND_A_INIT_REPLAY_NV_TRAIN_EXACTLY_SO_TRAJECTORIES_ARE_COMPARABLE_ACROSS_ARMS:
    &str =
    "LoRA A draws from the same splitmix64 Box-Muller stream nv-train's DetRng produces, scaled \
     by 1/sqrt(in_features), and B starts at zero, with one seed increment per attached module \
     in window order; libm ln/cos/sqrt on the same host make the init bit-identical to the \
     candle arm, so trajectory differences are attributable to the kernels, never to the start \
     point";

pub const ADAMW_REPLAYS_CANDLE_NN_SEMANTICS_WITH_ZERO_WEIGHT_DECAY: &str =
    "the update is candle-nn 0.11 AdamW with weight_decay 0: m = b1*m + (1-b1)*g, \
     v = b2*v + (1-b2)*g^2, bias-corrected by 1/(1-b^t) as f64 scalars cast to f32, \
     theta -= lr * m_hat / (sqrt(v_hat) + eps); b_lr_ratio scales the lr of every odd (B) var, \
     which is exactly nv-train's split WindowOptimizer because both halves step every time";

pub struct DetRng(u64);

impl DetRng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as f32;
        (bits / (1u64 << 24) as f32).max(f32::MIN_POSITIVE)
    }

    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32();
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

pub struct LoraVar {
    pub module: String,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub r: usize,
    pub n: usize,
    pub k: usize,
    pub scaling: f32,
}

impl LoraVar {
    pub fn init(module: String, n: usize, k: usize, r: usize, alpha: f64, seed: u64) -> Self {
        let _ =
            DETRNG_AND_A_INIT_REPLAY_NV_TRAIN_EXACTLY_SO_TRAJECTORIES_ARE_COMPARABLE_ACROSS_ARMS;
        let mut rng = DetRng::new(seed);
        let std = 1.0f32 / (k as f32).sqrt();
        let a = (0..r * k).map(|_| rng.next_normal() * std).collect();
        Self {
            module,
            a,
            b: vec![0f32; n * r],
            r,
            n,
            k,
            scaling: (alpha / r as f64) as f32,
        }
    }

    pub fn a_t(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.k * self.r];
        for row in 0..self.r {
            for col in 0..self.k {
                out[col * self.r + row] = self.a[row * self.k + col];
            }
        }
        out
    }

    pub fn b_t(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.r * self.n];
        for row in 0..self.n {
            for col in 0..self.r {
                out[col * self.n + row] = self.b[row * self.r + col];
            }
        }
        out
    }
}

pub struct AdamW {
    lr: f64,
    b_lr_ratio: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    t: i32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamW {
    pub fn new(sizes: &[usize], lr: f64, b_lr_ratio: f64) -> Self {
        let _ = ADAMW_REPLAYS_CANDLE_NN_SEMANTICS_WITH_ZERO_WEIGHT_DECAY;
        Self {
            lr,
            b_lr_ratio,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: sizes.iter().map(|&n| vec![0f32; n]).collect(),
            v: sizes.iter().map(|&n| vec![0f32; n]).collect(),
        }
    }

    pub fn step(&mut self, loras: &mut [LoraVar], grads: &[Vec<f32>]) {
        assert_eq!(
            grads.len(),
            loras.len() * 2,
            "AdamW expects one grad per A and B factor in attach order"
        );
        self.t += 1;
        let scale_m = (1f64 / (1f64 - self.beta1.powi(self.t))) as f32;
        let scale_v = (1f64 / (1f64 - self.beta2.powi(self.t))) as f32;
        let b1 = self.beta1 as f32;
        let b2 = self.beta2 as f32;
        let eps = self.eps as f32;
        for (li, lv) in loras.iter_mut().enumerate() {
            for half in 0..2 {
                let vi = 2 * li + half;
                let lr = if half == 1 {
                    (self.lr * self.b_lr_ratio) as f32
                } else {
                    self.lr as f32
                };
                let p = if half == 0 { &mut lv.a } else { &mut lv.b };
                let g = &grads[vi];
                assert_eq!(p.len(), g.len(), "grad {vi} shape mismatch");
                let ms = &mut self.m[vi];
                let vs = &mut self.v[vi];
                for j in 0..p.len() {
                    let gj = g[j];
                    let mj = b1 * ms[j] + (1.0 - b1) * gj;
                    let vj = b2 * vs[j] + (1.0 - b2) * gj * gj;
                    let m_hat = mj * scale_m;
                    let v_hat = vj * scale_v;
                    p[j] -= lr * (m_hat / (v_hat.sqrt() + eps));
                    ms[j] = mj;
                    vs[j] = vj;
                }
            }
        }
    }
}
