use anyhow::{bail, Result};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MixerKind {
    FullAttention,
    LinearAttention,
}

impl MixerKind {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "full_attention" => Ok(Self::FullAttention),
            "linear_attention" => Ok(Self::LinearAttention),
            other => bail!("unknown layer type {other:?}"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullAttention => "full_attention",
            Self::LinearAttention => "linear_attention",
        }
    }
}

#[derive(Debug)]
pub enum Mixed<F, L> {
    Full(F),
    Linear(L),
}

impl<F, L> Mixed<F, L> {
    pub fn kind(&self) -> MixerKind {
        match self {
            Self::Full(_) => MixerKind::FullAttention,
            Self::Linear(_) => MixerKind::LinearAttention,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerMixPlan {
    kinds: Vec<MixerKind>,
    full_slot_for_layer: Vec<Option<usize>>,
    linear_slot_for_layer: Vec<Option<usize>>,
    n_full: usize,
    n_linear: usize,
}

impl LayerMixPlan {
    pub fn from_kinds(kinds: Vec<MixerKind>) -> Self {
        let mut full_slot_for_layer = Vec::with_capacity(kinds.len());
        let mut linear_slot_for_layer = Vec::with_capacity(kinds.len());
        let mut n_full = 0usize;
        let mut n_linear = 0usize;
        for k in &kinds {
            match k {
                MixerKind::FullAttention => {
                    full_slot_for_layer.push(Some(n_full));
                    linear_slot_for_layer.push(None);
                    n_full += 1;
                }
                MixerKind::LinearAttention => {
                    full_slot_for_layer.push(None);
                    linear_slot_for_layer.push(Some(n_linear));
                    n_linear += 1;
                }
            }
        }
        Self {
            kinds,
            full_slot_for_layer,
            linear_slot_for_layer,
            n_full,
            n_linear,
        }
    }

    pub fn from_layer_type_strs<'a, I>(types: I) -> Result<Self>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let kinds = types
            .into_iter()
            .map(MixerKind::parse)
            .collect::<Result<Vec<_>>>()?;
        if kinds.is_empty() {
            bail!("empty layer_types");
        }
        Ok(Self::from_kinds(kinds))
    }

    pub fn from_full_attention_interval(num_layers: usize, interval: usize) -> Result<Self> {
        if interval == 0 {
            bail!("full_attention_interval must be >= 1");
        }
        if num_layers == 0 {
            bail!("num_layers must be >= 1");
        }
        let kinds = (0..num_layers)
            .map(|i| {
                if (i + 1) % interval == 0 {
                    MixerKind::FullAttention
                } else {
                    MixerKind::LinearAttention
                }
            })
            .collect();
        Ok(Self::from_kinds(kinds))
    }

    pub fn uniform(num_layers: usize, kind: MixerKind) -> Self {
        Self::from_kinds(vec![kind; num_layers])
    }

    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    pub fn kind(&self, layer: usize) -> Option<MixerKind> {
        self.kinds.get(layer).copied()
    }

    pub fn kinds(&self) -> &[MixerKind] {
        &self.kinds
    }

    pub fn n_full_layers(&self) -> usize {
        self.n_full
    }

    pub fn n_linear_layers(&self) -> usize {
        self.n_linear
    }

    pub fn full_slot_for_layer(&self, layer: usize) -> Option<usize> {
        self.full_slot_for_layer.get(layer).copied().flatten()
    }

    pub fn linear_slot_for_layer(&self, layer: usize) -> Option<usize> {
        self.linear_slot_for_layer.get(layer).copied().flatten()
    }

    pub fn full_slot_map(&self) -> &[Option<usize>] {
        &self.full_slot_for_layer
    }

    pub fn linear_slot_map(&self) -> &[Option<usize>] {
        &self.linear_slot_for_layer
    }

    pub fn full_layer_indices(&self) -> Vec<usize> {
        self.kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == MixerKind::FullAttention)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn linear_layer_indices(&self) -> Vec<usize> {
        self.kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == MixerKind::LinearAttention)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn matches_full_attention_interval(&self, interval: usize) -> bool {
        if interval == 0 {
            return false;
        }
        self.kinds.iter().enumerate().all(|(i, k)| {
            let expect_full = (i + 1) % interval == 0;
            (*k == MixerKind::FullAttention) == expect_full
        })
    }

    pub fn build<F, L>(
        &self,
        mut make_full: impl FnMut(usize) -> Result<F>,
        mut make_linear: impl FnMut(usize) -> Result<L>,
    ) -> Result<Vec<Mixed<F, L>>> {
        self.kinds
            .iter()
            .enumerate()
            .map(|(i, k)| match k {
                MixerKind::FullAttention => Ok(Mixed::Full(make_full(i)?)),
                MixerKind::LinearAttention => Ok(Mixed::Linear(make_linear(i)?)),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds_of(plan: &LayerMixPlan) -> String {
        plan.kinds()
            .iter()
            .map(|k| if *k == MixerKind::FullAttention { 'F' } else { 'L' })
            .collect()
    }

    #[test]
    fn a_slot_is_dense_within_its_own_kind_and_a_layer_has_exactly_one() {
        let plan = LayerMixPlan::from_full_attention_interval(8, 3).unwrap();
        assert_eq!(kinds_of(&plan), "LLFLLFLL");

        let mut next_full = 0usize;
        let mut next_linear = 0usize;
        for layer in 0..plan.len() {
            match plan.kind(layer).unwrap() {
                MixerKind::FullAttention => {
                    assert_eq!(plan.full_slot_for_layer(layer), Some(next_full));
                    assert_eq!(plan.linear_slot_for_layer(layer), None, "layer {layer} is full");
                    next_full += 1;
                }
                MixerKind::LinearAttention => {
                    assert_eq!(plan.linear_slot_for_layer(layer), Some(next_linear));
                    assert_eq!(plan.full_slot_for_layer(layer), None, "layer {layer} is linear");
                    next_linear += 1;
                }
            }
        }
        assert_eq!((next_full, next_linear), (plan.n_full_layers(), plan.n_linear_layers()));
        assert_eq!(next_full + next_linear, plan.len(), "every layer got exactly one slot");
        assert_eq!(plan.full_layer_indices(), vec![2, 5]);
        assert_eq!(plan.linear_layer_indices(), vec![0, 1, 3, 4, 6, 7]);
    }

    #[test]
    fn the_interval_counts_from_one_so_the_first_layer_is_never_full() {
        assert_eq!(kinds_of(&LayerMixPlan::from_full_attention_interval(6, 2).unwrap()), "LFLFLF");
        assert_eq!(kinds_of(&LayerMixPlan::from_full_attention_interval(6, 3).unwrap()), "LLFLLF");
        assert_eq!(kinds_of(&LayerMixPlan::from_full_attention_interval(4, 1).unwrap()), "FFFF");

        assert_eq!(kinds_of(&LayerMixPlan::from_full_attention_interval(3, 9).unwrap()), "LLL");
    }

    #[test]
    fn recognising_an_interval_is_the_exact_inverse_of_building_from_one() {
        for num_layers in 1..=12usize {
            for interval in 1..=6usize {
                let plan = LayerMixPlan::from_full_attention_interval(num_layers, interval).unwrap();
                assert!(
                    plan.matches_full_attention_interval(interval),
                    "{num_layers}/{interval}: built from an interval it does not recognise"
                );
                for other in 1..=6usize {
                    if other == interval {
                        continue;
                    }
                    let same = LayerMixPlan::from_full_attention_interval(num_layers, other)
                        .unwrap()
                        .kinds()
                        == plan.kinds();
                    assert_eq!(
                        plan.matches_full_attention_interval(other),
                        same,
                        "{num_layers}: interval {other} vs {interval} disagree with the layout"
                    );
                }
            }
        }
        assert!(
            !LayerMixPlan::uniform(4, MixerKind::FullAttention).matches_full_attention_interval(0),
            "interval 0 is not a layout, it is the error case"
        );
    }

    #[test]
    fn build_walks_layers_in_order_and_hands_each_maker_the_layer_index_not_the_slot() {
        let plan = LayerMixPlan::from_full_attention_interval(6, 3).unwrap();
        let mut full_seen = Vec::new();
        let mut linear_seen = Vec::new();
        let built = plan
            .build(
                |i| {
                    full_seen.push(i);
                    Ok(i)
                },
                |i| {
                    linear_seen.push(i);
                    Ok(i)
                },
            )
            .unwrap();
        assert_eq!(full_seen, vec![2, 5]);
        assert_eq!(linear_seen, vec![0, 1, 3, 4]);
        assert_eq!(built.len(), 6);
        for (layer, m) in built.iter().enumerate() {
            assert_eq!(m.kind(), plan.kind(layer).unwrap());
        }

        let e: Result<Vec<Mixed<usize, usize>>> =
            plan.build(|_| Ok(0usize), |_| bail!("no linear weights"));
        assert!(e.is_err());
    }

    #[test]
    fn the_error_cases_are_refused_rather_than_defaulted() {
        assert!(LayerMixPlan::from_full_attention_interval(4, 0).is_err(), "interval 0");
        assert!(LayerMixPlan::from_full_attention_interval(0, 4).is_err(), "no layers");
        assert!(LayerMixPlan::from_layer_type_strs(Vec::<&str>::new()).is_err(), "empty types");
        assert!(LayerMixPlan::from_layer_type_strs(["full_attention", "nope"]).is_err());
        assert!(MixerKind::parse("FullAttention").is_err(), "the wire name is snake_case");
    }

    #[test]
    fn a_layer_type_list_round_trips_through_its_own_wire_names() {
        for k in [MixerKind::FullAttention, MixerKind::LinearAttention] {
            assert_eq!(MixerKind::parse(k.as_str()).unwrap(), k);
        }
        let plan = LayerMixPlan::from_layer_type_strs([
            "linear_attention",
            "linear_attention",
            "full_attention",
        ])
        .unwrap();
        assert_eq!(kinds_of(&plan), "LLF");
        assert_eq!(plan.n_full_layers(), 1);
        assert_eq!(plan.kind(3), None, "past the end is None, not a default kind");
        assert_eq!(plan.full_slot_for_layer(99), None);
    }
}
