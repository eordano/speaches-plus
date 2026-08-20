use super::embedding::cosine_sim;

pub type ClusterId = u32;

#[derive(Clone, Debug)]
struct Centroid {
    id: ClusterId,
    vec: Vec<f32>,

    count: u64,
}

#[derive(Clone, Debug)]
pub struct OnlineClusterer {
    centroids: Vec<Centroid>,
    next_id: ClusterId,
    threshold: f32,
    max_speakers: usize,

    ema_smoothing: f32,
}

impl OnlineClusterer {
    pub fn new(threshold: f32, max_speakers: usize) -> Self {
        Self {
            centroids: Vec::with_capacity(max_speakers),
            next_id: 0,
            threshold,
            max_speakers,
            ema_smoothing: 0.9,
        }
    }

    pub fn with_ema(mut self, ema_smoothing: f32) -> Self {
        self.ema_smoothing = ema_smoothing.clamp(0.0, 0.999);
        self
    }

    pub fn reset(&mut self) {
        self.centroids.clear();
        self.next_id = 0;
    }

    pub fn num_clusters(&self) -> usize {
        self.centroids.len()
    }

    pub fn assign(&mut self, emb: &[f32]) -> (ClusterId, f32) {
        let best = self.best_match(emb);
        match best {
            Some((idx, sim)) if sim >= self.threshold => {
                self.update_centroid(idx, emb);
                (self.centroids[idx].id, sim)
            }
            _ if self.centroids.len() < self.max_speakers => {
                let id = self.next_id;
                self.next_id += 1;
                self.centroids.push(Centroid {
                    id,
                    vec: emb.to_vec(),
                    count: 1,
                });
                (id, best.map(|(_, s)| s).unwrap_or(1.0))
            }
            _ => {
                let (idx, sim) = best.expect("non-empty since max_speakers > 0");
                self.update_centroid(idx, emb);
                (self.centroids[idx].id, sim)
            }
        }
    }

    pub fn lookup(&self, emb: &[f32]) -> Option<(ClusterId, f32)> {
        let (idx, sim) = self.best_match(emb)?;
        if sim >= self.threshold {
            Some((self.centroids[idx].id, sim))
        } else {
            None
        }
    }

    fn best_match(&self, emb: &[f32]) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.centroids.iter().enumerate() {
            if c.vec.len() != emb.len() {
                continue;
            }
            let sim = cosine_sim(&c.vec, emb);
            match best {
                None => best = Some((i, sim)),
                Some((_, s)) if sim > s => best = Some((i, sim)),
                _ => {}
            }
        }
        best
    }

    fn update_centroid(&mut self, idx: usize, emb: &[f32]) {
        let c = &mut self.centroids[idx];
        let alpha = self.ema_smoothing;
        for (cv, &ev) in c.vec.iter_mut().zip(emb.iter()) {
            *cv = alpha * *cv + (1.0 - alpha) * ev;
        }

        let norm: f32 = c.vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for cv in c.vec.iter_mut() {
            *cv /= norm;
        }
        c.count = c.count.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / n).collect()
    }

    #[test]
    fn first_embedding_creates_cluster_zero() {
        let mut c = OnlineClusterer::new(0.5, 4);
        let v = unit(vec![1.0, 0.0, 0.0]);
        let (id, _) = c.assign(&v);
        assert_eq!(id, 0);
        assert_eq!(c.num_clusters(), 1);
    }

    #[test]
    fn similar_embedding_joins_cluster() {
        let mut c = OnlineClusterer::new(0.5, 4);
        let v1 = unit(vec![1.0, 0.0, 0.0]);
        let v2 = unit(vec![0.99, 0.01, 0.0]);
        let (id1, _) = c.assign(&v1);
        let (id2, sim) = c.assign(&v2);
        assert_eq!(id1, id2);
        assert!(sim > 0.9);
        assert_eq!(c.num_clusters(), 1);
    }

    #[test]
    fn dissimilar_embedding_creates_new_cluster() {
        let mut c = OnlineClusterer::new(0.5, 4);
        let v1 = unit(vec![1.0, 0.0, 0.0]);
        let v2 = unit(vec![0.0, 1.0, 0.0]);
        let (id1, _) = c.assign(&v1);
        let (id2, _) = c.assign(&v2);
        assert_ne!(id1, id2);
        assert_eq!(c.num_clusters(), 2);
    }

    #[test]
    fn max_speakers_caps_cluster_creation() {
        let mut c = OnlineClusterer::new(0.99, 2);
        let v1 = unit(vec![1.0, 0.0, 0.0]);
        let v2 = unit(vec![0.0, 1.0, 0.0]);
        let v3 = unit(vec![0.0, 0.0, 1.0]);
        c.assign(&v1);
        c.assign(&v2);
        let (_, _) = c.assign(&v3);

        assert_eq!(c.num_clusters(), 2);
    }

    #[test]
    fn lookup_does_not_create_clusters() {
        let mut c = OnlineClusterer::new(0.5, 4);
        let v = unit(vec![1.0, 0.0, 0.0]);
        c.assign(&v);
        let probe = unit(vec![0.0, 1.0, 0.0]);
        assert!(c.lookup(&probe).is_none());
        assert_eq!(c.num_clusters(), 1);
    }

    #[test]
    fn reset_clears_state() {
        let mut c = OnlineClusterer::new(0.5, 4);
        c.assign(&unit(vec![1.0, 0.0, 0.0]));
        c.reset();
        assert_eq!(c.num_clusters(), 0);
        let (id, _) = c.assign(&unit(vec![1.0, 0.0, 0.0]));
        assert_eq!(id, 0, "next_id should restart at 0 after reset");
    }
}
