use anyhow::{anyhow, Result};

use super::generator::Rng;

pub const STYLE_FIT_IS_A_ONE_PLUS_ONE_EVOLUTION_OVER_STOCK_BLEND_WEIGHTS: &str =
    "the style manifold is the mean of the stock voice rows plus a weighted sum of their \
     deviations; the search seeds the weights on the three best-scoring stock voices, samples a \
     population of gaussian perturbations per generation, keeps the best candidate when it \
     beats the incumbent (sigma x1.15, else x0.85) and early-stops on the held-out score, so \
     the same loop serves the engine's /v1/voices/fit and the browser trainer";

#[derive(Clone, Debug)]
pub struct StyleFitConfig {
    pub budget: usize,
    pub population: usize,
    pub lambda: f32,
    pub patience: usize,
    pub min_delta: f32,
    pub seed: u64,
}

impl Default for StyleFitConfig {
    fn default() -> Self {
        Self {
            budget: 60,
            population: 12,
            lambda: 0.02,
            patience: 8,
            min_delta: 0.005,
            seed: 1,
        }
    }
}

pub struct StyleFitEvent {
    pub generation: usize,
    pub validation: f32,
    pub best: f32,
}

pub struct StyleFit {
    pub style: Vec<f32>,
    pub weights: Vec<f32>,
    pub best_stock: (String, f32),
    pub history: Vec<f32>,
    pub generations: usize,
    pub best_step: usize,
    pub stop: &'static str,
    pub validation: f32,
}

pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f32,
    pub budget: usize,
    pub best: f32,
    pub best_step: usize,
    pub steps: usize,
    since: usize,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f32, budget: usize) -> Self {
        Self {
            patience,
            min_delta,
            budget,
            best: f32::NEG_INFINITY,
            best_step: 0,
            steps: 0,
            since: 0,
        }
    }

    pub fn update(&mut self, score: f32) -> bool {
        self.steps += 1;
        if score > self.best + self.min_delta {
            self.best = score;
            self.best_step = self.steps - 1;
            self.since = 0;
        } else {
            self.since += 1;
        }
        self.since >= self.patience || self.steps >= self.budget
    }

    pub fn improved(&self) -> bool {
        self.since == 0
    }
}

pub fn evolve(
    stock: &[(String, Vec<f32>)],
    cfg: &StyleFitConfig,
    mut score: impl FnMut(&[f32]) -> Result<f32>,
    mut heldout_score: impl FnMut(&[f32]) -> Result<f32>,
    mut on_start: impl FnMut(&(String, f32)),
    mut on_generation: impl FnMut(&StyleFitEvent),
) -> Result<StyleFit> {
    let _ = STYLE_FIT_IS_A_ONE_PLUS_ONE_EVOLUTION_OVER_STOCK_BLEND_WEIGHTS;
    if stock.len() < 3 {
        return Err(anyhow!(
            "need >= 3 stock voices to define the style manifold"
        ));
    }
    let dim = stock[0].1.len();
    let mu: Vec<f32> = (0..dim)
        .map(|i| stock.iter().map(|(_, v)| v[i]).sum::<f32>() / stock.len() as f32)
        .collect();
    let basis: Vec<Vec<f32>> = stock
        .iter()
        .map(|(_, v)| v.iter().zip(&mu).map(|(a, m)| a - m).collect())
        .collect();
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(stock.len());
    for (i, (_, v)) in stock.iter().enumerate() {
        scored.push((i, score(v)?));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let best_stock = (
        stock[scored[0].0].0.clone(),
        heldout_score(&stock[scored[0].0].1)?,
    );
    on_start(&best_stock);
    let style_of = |w: &[f32]| -> Vec<f32> {
        let mut s = mu.clone();
        for (wi, b) in w.iter().zip(&basis) {
            if *wi != 0.0 {
                for (sj, bj) in s.iter_mut().zip(b) {
                    *sj += wi * bj;
                }
            }
        }
        s
    };
    let mut w = vec![0.0f32; stock.len()];
    for (i, _) in scored.iter().take(3) {
        w[*i] = 1.0 / 3.0;
    }
    let mut cur = score(&style_of(&w))?;
    let mut best_w = w.clone();
    let mut es = EarlyStopping::new(cfg.patience, cfg.min_delta, cfg.budget);
    let mut rng = Rng(cfg.seed.max(1) ^ 0x9E37_79B9_7F4A_7C15);
    let mut sigma = 0.35f32;
    let mut history: Vec<f32> = Vec::new();
    let mut stop = "budget";
    for gen in 0..cfg.budget {
        let mut gen_best: Option<(f32, Vec<f32>)> = None;
        for _ in 0..cfg.population.max(2) {
            let cand: Vec<f32> = w.iter().map(|x| x + sigma * rng.normal()).collect();
            let penalty = cfg.lambda * cand.iter().map(|x| x * x).sum::<f32>() / cand.len() as f32;
            let s = score(&style_of(&cand))? - penalty;
            if gen_best.as_ref().map(|(b, _)| s > *b).unwrap_or(true) {
                gen_best = Some((s, cand));
            }
        }
        let (s, cand) = gen_best.expect("population >= 2");
        if s > cur {
            cur = s;
            w = cand;
            sigma *= 1.15;
        } else {
            sigma *= 0.85;
        }
        let validation = heldout_score(&style_of(&w))?;
        history.push(validation);
        let done = es.update(validation);
        if es.improved() {
            best_w = w.clone();
        }
        on_generation(&StyleFitEvent {
            generation: gen,
            validation,
            best: es.best,
        });
        if done {
            stop = if es.steps < cfg.budget {
                "patience"
            } else {
                "budget"
            };
            break;
        }
    }
    let style = style_of(&best_w);
    let validation = heldout_score(&style)?;
    Ok(StyleFit {
        style,
        weights: best_w,
        best_stock,
        history,
        generations: es.steps,
        best_step: es.best_step,
        stop,
        validation,
    })
}
