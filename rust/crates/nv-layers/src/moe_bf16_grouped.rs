use anyhow::{Context, Result};
use candle_core::{DType, Tensor};

pub struct Bf16GroupedExperts {
    num_experts: usize,
    hidden_size: usize,
    intermediate_size: usize,

    gate_up: Tensor,

    down: Tensor,
}

impl Bf16GroupedExperts {
    pub fn new(gate_up: &Tensor, down: &Tensor) -> Result<Self> {
        let gu = gate_up
            .dims3()
            .context("gate_up must be [E, 2*inter, hidden]")?;
        let dn = down.dims3().context("down must be [E, hidden, inter]")?;
        let (e, two_inter, hidden) = gu;
        let (e2, hidden2, inter) = dn;
        anyhow::ensure!(e == e2, "gate_up experts {e} != down experts {e2}");
        anyhow::ensure!(
            hidden == hidden2,
            "gate_up hidden {hidden} != down hidden {hidden2}"
        );
        anyhow::ensure!(
            two_inter == 2 * inter,
            "gate_up out {two_inter} != 2*inter ({})",
            2 * inter
        );
        let gate_up = gate_up.to_dtype(DType::F32)?.contiguous()?;
        let down = down.to_dtype(DType::F32)?.contiguous()?;
        Ok(Self {
            num_experts: e,
            hidden_size: hidden,
            intermediate_size: inter,
            gate_up,
            down,
        })
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    pub fn forward(
        &self,
        x_flat: &Tensor,
        topk_ids: &[u32],
        topk_weights: &[f32],
        k: usize,
    ) -> Result<Tensor> {
        let (n_tokens, hidden) = x_flat
            .dims2()
            .context("x_flat must be [n_tokens, hidden]")?;
        anyhow::ensure!(
            hidden == self.hidden_size,
            "x hidden {hidden} != {}",
            self.hidden_size
        );
        anyhow::ensure!(k > 0, "k must be > 0");
        anyhow::ensure!(
            topk_ids.len() == n_tokens * k && topk_weights.len() == n_tokens * k,
            "topk_ids/weights len {}/{} != n_tokens*k {}",
            topk_ids.len(),
            topk_weights.len(),
            n_tokens * k
        );
        let device = x_flat.device().clone();
        let x_flat = x_flat.to_dtype(DType::F32)?.contiguous()?;

        let mut rows_per_expert: Vec<Vec<u32>> = vec![Vec::new(); self.num_experts];
        let mut w_per_expert: Vec<Vec<f32>> = vec![Vec::new(); self.num_experts];
        for n in 0..n_tokens {
            for j in 0..k {
                let e = topk_ids[n * k + j] as usize;
                anyhow::ensure!(
                    e < self.num_experts,
                    "routed expert id {e} out of range ({} experts)",
                    self.num_experts
                );
                rows_per_expert[e].push(n as u32);
                w_per_expert[e].push(topk_weights[n * k + j]);
            }
        }

        let inter = self.intermediate_size;
        let mut acc = Tensor::zeros((n_tokens, hidden), DType::F32, &device)?;
        for e in 0..self.num_experts {
            let rows = &rows_per_expert[e];
            if rows.is_empty() {
                continue;
            }
            let m = rows.len();
            let idx = Tensor::from_vec(rows.clone(), m, &device)?;
            let xe = x_flat.index_select(&idx, 0)?.contiguous()?;

            let w_gu = self.gate_up.narrow(0, e, 1)?.squeeze(0)?.contiguous()?;
            let gu = xe.matmul(&w_gu.t()?.contiguous()?)?;
            let gate = gu.narrow(1, 0, inter)?.contiguous()?;
            let up = gu.narrow(1, inter, inter)?.contiguous()?;
            let act = silu(&gate)?.mul(&up)?;

            let w_dn = self.down.narrow(0, e, 1)?.squeeze(0)?.contiguous()?;
            let ye = act.matmul(&w_dn.t()?.contiguous()?)?;

            let w_t = Tensor::from_vec(w_per_expert[e].clone(), (m, 1), &device)?;
            let weighted = ye.broadcast_mul(&w_t)?;
            acc = acc.index_add(&idx, &weighted, 0)?;
        }
        Ok(acc)
    }
}

fn silu(x: &Tensor) -> Result<Tensor> {
    let s = candle_nn::ops::sigmoid(x)?;
    Ok(x.mul(&s)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn det(vals: usize, seed: f32) -> Vec<f32> {
        (0..vals)
            .map(|i| ((i as f32 + seed) * 0.3137).sin() * 0.8)
            .collect()
    }

    fn silu_s(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }

    #[allow(clippy::too_many_arguments)]
    fn reference(
        x: &[f32],
        gate_up: &[f32],
        down: &[f32],
        topk_ids: &[u32],
        topk_w: &[f32],
        n_tokens: usize,
        k: usize,
        hidden: usize,
        inter: usize,
    ) -> Vec<f32> {
        let two_inter = 2 * inter;
        let mut out = vec![0f32; n_tokens * hidden];
        for n in 0..n_tokens {
            let xr = &x[n * hidden..(n + 1) * hidden];
            for j in 0..k {
                let e = topk_ids[n * k + j] as usize;
                let w = topk_w[n * k + j];
                let gu_base = e * two_inter * hidden;
                let dn_base = e * hidden * inter;

                let mut act = vec![0f32; inter];
                for r in 0..inter {
                    let mut g = 0f32;
                    let mut u = 0f32;
                    for c in 0..hidden {
                        g += gate_up[gu_base + r * hidden + c] * xr[c];
                        u += gate_up[gu_base + (inter + r) * hidden + c] * xr[c];
                    }
                    act[r] = silu_s(g) * u;
                }
                for h in 0..hidden {
                    let mut acc = 0f32;
                    for r in 0..inter {
                        acc += down[dn_base + h * inter + r] * act[r];
                    }
                    out[n * hidden + h] += w * acc;
                }
            }
        }
        out
    }

    #[test]
    fn bf16_grouped_matches_reference_dense_loop_exact() {
        let device = Device::Cpu;
        let e = 5usize;
        let hidden = 8usize;
        let inter = 6usize;
        let n_tokens = 7usize;
        let k = 3usize;

        let gate_up_host = det(e * 2 * inter * hidden, 1.0);
        let down_host = det(e * hidden * inter, 50.0);
        let x_host = det(n_tokens * hidden, 200.0);

        let gate_up =
            Tensor::from_vec(gate_up_host.clone(), (e, 2 * inter, hidden), &device).unwrap();
        let down = Tensor::from_vec(down_host.clone(), (e, hidden, inter), &device).unwrap();
        let x = Tensor::from_vec(x_host.clone(), (n_tokens, hidden), &device).unwrap();

        let mut topk_ids = Vec::with_capacity(n_tokens * k);
        let mut topk_w = Vec::with_capacity(n_tokens * k);
        for n in 0..n_tokens {
            let mut ws = Vec::with_capacity(k);
            for j in 0..k {
                topk_ids.push(((n * 3 + j * 2 + 1) % e) as u32);
                ws.push(((n + j + 1) as f32).sqrt());
            }
            let z: f32 = ws.iter().sum();
            for w in ws {
                topk_w.push(w / z);
            }
        }

        let experts = Bf16GroupedExperts::new(&gate_up, &down).unwrap();
        let got = experts.forward(&x, &topk_ids, &topk_w, k).unwrap();
        let got_host: Vec<f32> = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let want = reference(
            &x_host,
            &gate_up_host,
            &down_host,
            &topk_ids,
            &topk_w,
            n_tokens,
            k,
            hidden,
            inter,
        );

        let mut max_abs = 0f32;
        for (a, b) in got_host.iter().zip(want.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 1e-5,
            "grouped vs reference max abs diff {max_abs} too large"
        );
    }
}
