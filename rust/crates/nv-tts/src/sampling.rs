use anyhow::Result;

#[derive(Clone, Debug)]
pub struct SamplerConfig {
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            do_sample: true,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 0x5EED_1234_ABCD_9876,
        }
    }
}

pub struct Sampler {
    cfg: SamplerConfig,
    state: u64,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        let state = cfg.seed | 1;
        Self { cfg, state }
    }

    pub fn config(&self) -> &SamplerConfig {
        &self.cfg
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    pub fn sample(
        &mut self,
        logits: &[f32],
        penalize: &[u32],
        allowed: impl Fn(usize) -> bool,
    ) -> Result<u32> {
        if logits.is_empty() {
            anyhow::bail!("Sampler.sample: empty logits");
        }
        let mut work: Vec<f32> = logits.to_vec();
        if (self.cfg.repetition_penalty - 1.0).abs() > f32::EPSILON {
            let p = self.cfg.repetition_penalty;
            for &id in penalize {
                let i = id as usize;
                if i < work.len() {
                    let v = work[i];
                    work[i] = if v > 0.0 { v / p } else { v * p };
                }
            }
        }
        for (i, v) in work.iter_mut().enumerate() {
            if !allowed(i) {
                *v = f32::NEG_INFINITY;
            }
        }
        if !self.cfg.do_sample {
            return Ok(argmax(&work));
        }
        let t = self.cfg.temperature.max(1e-5);
        for v in work.iter_mut() {
            *v /= t;
        }
        let mut idx: Vec<u32> = (0..work.len() as u32).collect();
        idx.sort_unstable_by(|&a, &b| {
            work[b as usize]
                .partial_cmp(&work[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let k = if self.cfg.top_k == 0 {
            idx.len()
        } else {
            self.cfg.top_k.min(idx.len())
        };
        idx.truncate(k);
        let max_l = work[idx[0] as usize];
        if max_l == f32::NEG_INFINITY {
            anyhow::bail!("Sampler.sample: all candidate logits suppressed");
        }
        let mut probs: Vec<f32> = idx
            .iter()
            .map(|&i| (work[i as usize] - max_l).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        if self.cfg.top_p < 1.0 {
            let mut cum = 0.0f32;
            let mut cut = probs.len();
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if cum >= self.cfg.top_p {
                    cut = i + 1;
                    break;
                }
            }
            probs.truncate(cut);
            idx.truncate(cut);
            let s: f32 = probs.iter().sum();
            for p in probs.iter_mut() {
                *p /= s;
            }
        }
        let r = self.next_f32();
        let mut cum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r < cum {
                return Ok(idx[i]);
            }
        }
        Ok(*idx.last().unwrap())
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_ignores_rng_and_respects_mask() {
        let mut s = Sampler::new(SamplerConfig {
            do_sample: false,
            ..Default::default()
        });
        let logits = vec![0.1, 5.0, 3.0, 4.0];
        let tok = s.sample(&logits, &[], |i| i != 1).unwrap();
        assert_eq!(tok, 3);
    }

    #[test]
    fn repetition_penalty_can_flip_argmax() {
        let mut s = Sampler::new(SamplerConfig {
            do_sample: false,
            repetition_penalty: 2.0,
            ..Default::default()
        });
        let logits = vec![1.0, 1.5];
        let tok = s.sample(&logits, &[1], |_| true).unwrap();
        assert_eq!(tok, 0);
    }

    #[test]
    fn sampling_stays_in_allowed_set() {
        let mut s = Sampler::new(SamplerConfig::default());
        let logits: Vec<f32> = (0..100).map(|i| (i % 7) as f32 * 0.3).collect();
        for _ in 0..200 {
            let tok = s.sample(&logits, &[], |i| i < 50).unwrap();
            assert!(tok < 50);
        }
    }

    #[test]
    fn all_suppressed_is_an_error() {
        let mut s = Sampler::new(SamplerConfig::default());
        let logits = vec![1.0, 2.0];
        assert!(s.sample(&logits, &[], |_| false).is_err());
    }
}
