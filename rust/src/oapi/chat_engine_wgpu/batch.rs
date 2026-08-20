#[derive(Clone, Copy, Debug)]
pub struct KvGeometry {
    pub sliding_layers: u64,
    pub sliding_window: u64,
    pub sliding_bytes_per_pos: u64,
    pub global_layers: u64,
    pub global_bytes_per_pos: u64,
}

impl KvGeometry {
    pub fn bytes_at(&self, ctx_len: u64) -> u64 {
        self.sliding_layers * ctx_len.min(self.sliding_window) * self.sliding_bytes_per_pos
            + self.global_layers * ctx_len * self.global_bytes_per_pos
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StepBudget {
    pub weight_bytes: u64,
    pub kv: KvGeometry,
}

impl StepBudget {
    pub fn bytes_per_token(&self, batch: u64, ctx_len: u64) -> f64 {
        let b = batch.max(1) as f64;
        self.weight_bytes as f64 / b + self.kv.bytes_at(ctx_len) as f64
    }

    pub fn knee(&self, ctx_len: u64) -> f64 {
        self.weight_bytes as f64 / self.kv.bytes_at(ctx_len).max(1) as f64
    }
}

pub fn gemma4_31b_budget(kv_elem_bytes: u64) -> StepBudget {
    let sliding_kv_heads = 16u64;
    let sliding_head_dim = 256u64;
    let global_kv_heads = 4u64;
    let global_head_dim = 512u64;
    StepBudget {
        weight_bytes: 32_760_000_000,
        kv: KvGeometry {
            sliding_layers: 50,
            sliding_window: 1024,
            sliding_bytes_per_pos: 2 * sliding_kv_heads * sliding_head_dim * kv_elem_bytes,
            global_layers: 10,
            global_bytes_per_pos: 2 * global_kv_heads * global_head_dim * kv_elem_bytes,
        },
    }
}

pub const KV_ELEM_BYTES: u64 = 2;

pub fn kv_geometry_from_config(raw: &str, kv_elem_bytes: u64) -> Option<KvGeometry> {
    let root: serde_json::Value = serde_json::from_str(raw).ok()?;
    let cfg = root.get("text_config").unwrap_or(&root);
    let num = |k: &str| -> Option<u64> {
        cfg.get(k)
            .or_else(|| root.get(k))
            .and_then(|v| v.as_u64())
            .filter(|v| *v > 0)
    };
    let layers = num("num_hidden_layers")?;
    let kv_heads = num("num_key_value_heads")?;
    let head_dim = num("head_dim").or_else(|| {
        let h = num("hidden_size")?;
        let a = num("num_attention_heads")?;
        (h % a == 0).then_some(h / a)
    })?;
    let global_kv_heads = num("num_global_key_value_heads").unwrap_or(kv_heads);
    let global_head_dim = num("global_head_dim").unwrap_or(head_dim);
    let window = num("sliding_window").unwrap_or(0);
    let types = cfg
        .get("layer_types")
        .or_else(|| root.get("layer_types"))
        .and_then(|v| v.as_array());
    let sliding_layers = match (types, window) {
        (Some(t), w) if w > 0 => t
            .iter()
            .filter(|v| v.as_str().is_some_and(|s| s.contains("sliding")))
            .count() as u64,
        _ => 0,
    };
    Some(KvGeometry {
        sliding_layers,
        sliding_window: window,
        sliding_bytes_per_pos: 2 * kv_heads * head_dim * kv_elem_bytes,
        global_layers: layers.saturating_sub(sliding_layers),
        global_bytes_per_pos: 2 * global_kv_heads * global_head_dim * kv_elem_bytes,
    })
}

pub const BATCH_ENV: &str = "NV_WGPU_BATCH";
pub const BATCH_WINDOW_MS_ENV: &str = "NV_WGPU_BATCH_WINDOW_MS";
pub const BATCH_HEADROOM_ENV: &str = "NV_WGPU_BATCH_HEADROOM_GIB";

pub const FREE_ROWS: usize = 8;

pub const DEFAULT_WINDOW_MS: u64 = 10;

pub const DEFAULT_HEADROOM_GIB: f64 = 6.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchKnobs {
    pub max_batch: usize,
    pub window: std::time::Duration,
    pub headroom_gib: f64,
}

impl Default for BatchKnobs {
    fn default() -> Self {
        Self {
            max_batch: 1,
            window: std::time::Duration::from_millis(DEFAULT_WINDOW_MS),
            headroom_gib: DEFAULT_HEADROOM_GIB,
        }
    }
}

impl BatchKnobs {
    pub fn from_env() -> Self {
        fn num<T: std::str::FromStr>(k: &str) -> Option<T> {
            std::env::var(k).ok().and_then(|v| v.trim().parse().ok())
        }
        let d = Self::default();
        Self {
            max_batch: num::<usize>(BATCH_ENV)
                .filter(|v| *v > 0)
                .unwrap_or(d.max_batch),
            window: num::<u64>(BATCH_WINDOW_MS_ENV)
                .map(std::time::Duration::from_millis)
                .unwrap_or(d.window),
            headroom_gib: num::<f64>(BATCH_HEADROOM_ENV)
                .filter(|v| v.is_finite() && *v >= 0.0)
                .unwrap_or(d.headroom_gib),
        }
    }

    pub fn enabled(&self) -> bool {
        self.max_batch > 1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    NeedsHostLogits,

    MultiRowRoute,

    MmSplice,

    KindHasNoBatchGraph,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::NeedsHostLogits => "request needs host logits",
            Self::MultiRowRoute => "request takes a multi-row decode route",
            Self::MmSplice => "request carries image or audio splice media",
            Self::KindHasNoBatchGraph => "this model kind has no batched decode graph",
        };
        f.write_str(s)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchGap {
    pub slotted_kv: bool,

    pub m_row_decode_step: bool,

    pub per_slot_recurrent_state: bool,
}

impl BatchGap {
    pub const ATTENTION_KV_IN_ONE_REGION: Self = Self {
        slotted_kv: true,
        m_row_decode_step: true,
        per_slot_recurrent_state: false,
    };

    pub const RECURRENT_STATE_IN_ONE_REGION: Self = Self {
        slotted_kv: true,
        m_row_decode_step: true,
        per_slot_recurrent_state: true,
    };

    pub fn missing(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.slotted_kv {
            v.push(
                "slotted KV (reset_slot(slot) / prefill_slot(slot, tokens) / select_slot(slot) \
                 over per-slot base offsets, the nv-models::gemma4_wgpu MK_MAX pattern)",
            );
        }
        if self.m_row_decode_step {
            v.push(
                "an M-row decode step (decode_step_batch): the M-row prefill and verify_chain \
                 graphs this kind does have pack rows of ONE sequence against ONE KV region, and \
                 mask by a single total, so they cannot carry one token from each of N streams",
            );
        }
        if self.per_slot_recurrent_state {
            v.push(
                "per-slot recurrent state: the DeltaNet/GDN state buffers are singletons, so two \
                 streams sharing the graph would overwrite each other's state every step",
            );
        }
        v
    }
}

impl std::fmt::Display for BatchGap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("batched serving needs ")?;
        let missing = self.missing();
        for (i, m) in missing.iter().enumerate() {
            if i > 0 {
                f.write_str(if i + 1 == missing.len() {
                    ", and "
                } else {
                    ", "
                })?;
            }
            f.write_str(m)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Admission {
    pub budget: StepBudget,
    pub max_seq: u64,
    pub knobs: BatchKnobs,
}

impl Admission {
    pub fn slot_gib(&self) -> f64 {
        self.budget.kv.bytes_at(self.max_seq) as f64 / (1u64 << 30) as f64
    }

    pub fn memory_slots(&self, available_gib: f64) -> usize {
        let usable = available_gib - self.knobs.headroom_gib;
        let per = self.slot_gib();
        if per <= 0.0 || usable <= 0.0 {
            return 1;
        }
        ((usable / per).floor() as usize).max(1)
    }

    pub fn knee_slots(&self) -> usize {
        let k = self.budget.knee(self.max_seq);
        if !k.is_finite() {
            return self.knobs.max_batch;
        }
        (k.round().max(0.0) as usize).max(FREE_ROWS)
    }

    pub fn slots(&self, graph_capacity: usize, available_gib: Option<f64>) -> usize {
        let mut n = self
            .knobs
            .max_batch
            .min(graph_capacity)
            .min(self.knee_slots());
        if let Some(a) = available_gib {
            n = n.min(self.memory_slots(a));
        }
        n.max(1)
    }
}

pub trait BatchStepper {
    fn batch_capacity(&self) -> usize {
        1
    }

    fn reset_batch(&mut self, slots: usize) -> anyhow::Result<()>;

    fn prefill_slot(&mut self, slot: usize, tokens: &[u32]) -> anyhow::Result<u32>;

    fn decode_step_batch(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<u32>>;

    fn end_batch(&mut self) {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotStep {
    Feed(u32),
    Done,
}

pub trait SlotSink {
    fn accept(&mut self, sampled: u32) -> anyhow::Result<SlotStep>;

    fn finish(&mut self);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SlotStats {
    pub emitted: u64,
    pub live_s: f64,
}

impl SlotStats {
    pub fn ms_per_token(&self) -> f64 {
        if self.emitted == 0 {
            return 0.0;
        }
        self.live_s * 1000.0 / self.emitted as f64
    }
}

#[derive(Clone, Debug)]
pub struct BatchStats {
    pub batch: usize,
    pub steps: u64,
    pub wall_s: f64,
    pub slots: Vec<SlotStats>,
}

impl BatchStats {
    pub fn emitted(&self) -> u64 {
        self.slots.iter().map(|s| s.emitted).sum()
    }

    pub fn aggregate_tok_s(&self) -> f64 {
        if self.wall_s <= 0.0 {
            return 0.0;
        }
        self.emitted() as f64 / self.wall_s
    }

    pub fn worst_stream_ms_per_token(&self) -> f64 {
        self.slots
            .iter()
            .filter(|s| s.emitted > 0)
            .map(|s| s.ms_per_token())
            .fold(0.0, f64::max)
    }

    pub fn best_stream_ms_per_token(&self) -> f64 {
        self.slots
            .iter()
            .filter(|s| s.emitted > 0)
            .map(|s| s.ms_per_token())
            .fold(f64::INFINITY, f64::min)
    }

    pub fn summary(&self) -> String {
        format!(
            "B={} steps={} emitted={} | aggregate {:.2} tok/s | per-stream {:.2}..{:.2} ms/token",
            self.batch,
            self.steps,
            self.emitted(),
            self.aggregate_tok_s(),
            self.best_stream_ms_per_token(),
            self.worst_stream_ms_per_token(),
        )
    }
}

struct SlotRun<S> {
    sink: S,
    feed: u32,
    live: bool,
    stats: SlotStats,
}

pub struct Batch<S: SlotSink> {
    slots: Vec<SlotRun<S>>,
    live: usize,
    steps: u64,
    started: std::time::Instant,
}

impl<S: SlotSink> Batch<S> {
    pub fn new(seeded: Vec<(S, u32)>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            seeded.len() >= 2,
            "a batch of {} must not be scheduled: B=1 is the single-stream path, byte for byte",
            seeded.len()
        );
        let started = std::time::Instant::now();
        let mut slots = Vec::with_capacity(seeded.len());
        let mut live = 0usize;
        for (sink, first) in seeded {
            let mut run = SlotRun {
                sink,
                feed: first,
                live: true,
                stats: SlotStats::default(),
            };
            match run.sink.accept(first)? {
                SlotStep::Feed(t) => {
                    run.feed = t;
                    run.stats.emitted += 1;
                    live += 1;
                }
                SlotStep::Done => {
                    run.live = false;
                    run.stats.live_s = started.elapsed().as_secs_f64();
                    run.sink.finish();
                }
            }
            slots.push(run);
        }
        Ok(Self {
            slots,
            live,
            steps: 0,
            started,
        })
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn live(&self) -> usize {
        self.live
    }

    pub fn step(&mut self, model: &mut dyn BatchStepper) -> anyhow::Result<()> {
        let tokens: Vec<u32> = self.slots.iter().map(|s| s.feed).collect();
        let out = model.decode_step_batch(&tokens)?;
        anyhow::ensure!(
            out.len() == tokens.len(),
            "decode_step_batch returned {} ids for {} slots: the graph and the scheduler \
             disagree about the batch, and a short return would silently serve one stream's \
             token to another",
            out.len(),
            tokens.len()
        );
        self.steps += 1;
        for (slot, sampled) in self.slots.iter_mut().zip(out) {
            if !slot.live {
                continue;
            }
            match slot.sink.accept(sampled)? {
                SlotStep::Feed(t) => {
                    slot.feed = t;
                    slot.stats.emitted += 1;
                }
                SlotStep::Done => {
                    slot.live = false;
                    slot.stats.live_s = self.started.elapsed().as_secs_f64();
                    slot.sink.finish();
                    self.live -= 1;
                }
            }
        }
        Ok(())
    }

    pub fn drain(
        &mut self,
        model: &mut dyn BatchStepper,
        max_steps: u64,
    ) -> anyhow::Result<BatchStats> {
        while self.live > 0 && self.steps < max_steps {
            self.step(model)?;
        }
        let wall_s = self.started.elapsed().as_secs_f64();
        for slot in self.slots.iter_mut() {
            if slot.live {
                slot.live = false;
                slot.stats.live_s = wall_s;
                slot.sink.finish();
                self.live -= 1;
            }
        }
        Ok(BatchStats {
            batch: self.slots.len(),
            steps: self.steps,
            wall_s,
            slots: self.slots.iter().map(|s| s.stats).collect(),
        })
    }
}

const BATCH_FIXED_MS: f64 = 86.1;
const BATCH_PER_SLOT_MS: f64 = 29.45;
const SINGLE_STEP_MS: f64 = 57.02;

pub fn batch_pays(live: usize) -> bool {
    if live < 2 {
        return false;
    }
    let batched = BATCH_FIXED_MS + BATCH_PER_SLOT_MS * live as f64;
    let serial = live as f64 * SINGLE_STEP_MS;
    batched < serial
}

pub fn batch_break_even(cap: usize) -> Option<usize> {
    (2..=cap).find(|&n| batch_pays(n))
}

#[cfg(test)]
mod batch_pays_tests {
    use super::*;

    #[test]
    fn the_model_reproduces_the_measured_pair() {
        let ratio = (2.0 * SINGLE_STEP_MS) / (BATCH_FIXED_MS + BATCH_PER_SLOT_MS * 2.0);
        assert!(
            (ratio - 0.787).abs() < 0.005,
            "model says {ratio:.3}x, measurement said 0.787x"
        );
        assert!(!batch_pays(2));
    }

    #[test]
    fn four_is_where_the_fixed_term_amortizes() {
        assert_eq!(batch_break_even(16), Some(4));
        assert!(!batch_pays(3));
        assert!(batch_pays(4));
    }

    #[test]
    fn a_lone_request_never_batches() {
        assert!(!batch_pays(1));
        assert!(!batch_pays(0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kv_term_stops_growing_once_the_sliding_window_is_full() {
        let b = gemma4_31b_budget(2);
        let at_1k = b.kv.bytes_at(1024);
        let at_2k = b.kv.bytes_at(2048);
        let sliding = 50 * 1024 * 2 * 16 * 256 * 2;
        let global_per_pos = 10 * 2 * 4 * 512 * 2;
        assert_eq!(at_1k, sliding + 1024 * global_per_pos);
        assert_eq!(at_2k - at_1k, 1024 * global_per_pos);
    }

    #[test]
    fn the_knee_is_where_doubling_the_batch_stops_paying() {
        let b = gemma4_31b_budget(2);
        for l in [1024u64, 8192, 32768, 131072] {
            let k = b.knee(l);
            let at_knee = b.bytes_per_token(k.round() as u64, l);
            let at_double = b.bytes_per_token((k.round() * 2.0) as u64, l);
            let gain = at_knee / at_double;
            assert!(
                (1.25..1.4).contains(&gain),
                "ctx {l}: doubling B past the knee ({k:.1}) gained {gain:.3}x, not the ~1.33x \
                 that a half-weight-half-KV split predicts"
            );
        }
    }

    #[test]
    fn bytes_per_token_floors_at_the_kv_term_however_large_the_batch() {
        let b = gemma4_31b_budget(2);
        let floor = b.kv.bytes_at(32768) as f64;
        assert!(b.bytes_per_token(1_000_000, 32768) / floor < 1.001);
    }
}

#[cfg(test)]
mod sched_tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    const TOY_EOS: u32 = 7;

    fn toy_advance(state: &mut u64, token: u32) -> u32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(u64::from(token) | 1);
        ((*state >> 33) as u32 % 97) + 3
    }

    struct ToyLm {
        state: Vec<u64>,
        capacity: usize,
        short_by: usize,
    }

    impl ToyLm {
        fn new(capacity: usize) -> Self {
            Self {
                state: Vec::new(),
                capacity,
                short_by: 0,
            }
        }
    }

    impl BatchStepper for ToyLm {
        fn batch_capacity(&self) -> usize {
            self.capacity
        }

        fn reset_batch(&mut self, slots: usize) -> anyhow::Result<()> {
            self.state = vec![0u64; slots];
            Ok(())
        }

        fn prefill_slot(&mut self, slot: usize, tokens: &[u32]) -> anyhow::Result<u32> {
            let s = self
                .state
                .get_mut(slot)
                .ok_or_else(|| anyhow::anyhow!("slot {slot} out of range"))?;
            let mut out = 0;
            for t in tokens {
                out = toy_advance(s, *t);
            }
            Ok(out)
        }

        fn decode_step_batch(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<u32>> {
            anyhow::ensure!(tokens.len() == self.state.len());
            let mut out: Vec<u32> = tokens
                .iter()
                .zip(self.state.iter_mut())
                .map(|(t, s)| toy_advance(s, *t))
                .collect();
            out.truncate(out.len() - self.short_by);
            Ok(out)
        }
    }

    fn single_stream_reference(prompt: &[u32], max_new: usize) -> Vec<u32> {
        let mut s = 0u64;
        let mut out = Vec::new();
        let mut next = 0;
        for t in prompt {
            next = toy_advance(&mut s, *t);
        }
        for _ in 0..max_new {
            if next == TOY_EOS {
                break;
            }
            out.push(next);
            next = toy_advance(&mut s, next);
        }
        out
    }

    #[derive(Default, Debug)]
    struct Collected {
        tokens: Vec<u32>,
        finished: usize,
    }

    struct ToySink {
        max_new: usize,
        out: Rc<RefCell<Collected>>,
    }

    impl SlotSink for ToySink {
        fn accept(&mut self, sampled: u32) -> anyhow::Result<SlotStep> {
            let mut c = self.out.borrow_mut();
            if sampled == TOY_EOS || c.tokens.len() >= self.max_new {
                return Ok(SlotStep::Done);
            }
            c.tokens.push(sampled);
            Ok(SlotStep::Feed(sampled))
        }

        fn finish(&mut self) {
            self.out.borrow_mut().finished += 1;
        }
    }

    fn run_batch(
        prompts: &[&[u32]],
        max_new: usize,
        short_by: usize,
    ) -> anyhow::Result<(Vec<Rc<RefCell<Collected>>>, BatchStats)> {
        let mut lm = ToyLm::new(prompts.len());
        lm.short_by = short_by;
        lm.reset_batch(prompts.len())?;
        let mut sinks = Vec::new();
        let mut seeded = Vec::new();
        for (i, p) in prompts.iter().enumerate() {
            let first = lm.prefill_slot(i, p)?;
            let out = Rc::new(RefCell::new(Collected::default()));
            sinks.push(out.clone());
            seeded.push((ToySink { max_new, out }, first));
        }
        let mut batch = Batch::new(seeded)?;
        let stats = batch.drain(&mut lm, max_new as u64 + 4)?;
        Ok((sinks, stats))
    }

    #[test]
    fn a_batch_of_one_is_refused_rather_than_scheduled() {
        let out = Rc::new(RefCell::new(Collected::default()));
        let made = Batch::new(vec![(
            ToySink {
                max_new: 4,
                out: out.clone(),
            },
            11,
        )]);
        let err = match made {
            Ok(_) => panic!("B=1 constructed a batch"),
            Err(e) => e,
        };
        assert!(format!("{err}").contains("single-stream"));
    }

    #[test]
    fn each_slot_gets_its_own_single_stream_tokens() {
        let prompts: [&[u32]; 4] = [&[1, 2, 3], &[9], &[4, 4, 4, 4], &[31, 17]];
        let max_new = 60;
        let (sinks, stats) = run_batch(&prompts, max_new, 0).unwrap();
        let mut early = 0usize;
        for (i, p) in prompts.iter().enumerate() {
            let want = single_stream_reference(p, max_new);
            let got = sinks[i].borrow().tokens.clone();
            assert_eq!(got, want, "slot {i} diverged from its single-stream tokens");
            assert_eq!(sinks[i].borrow().finished, 1, "slot {i} finish() count");
            if want.len() < max_new {
                early += 1;
            }
        }
        assert_eq!(
            early, 2,
            "the fixture must retire slots at several different steps, or it does not exercise \
             the claim it is named for"
        );
        assert_eq!(stats.batch, 4);
        assert_eq!(
            stats.emitted(),
            sinks
                .iter()
                .map(|s| s.borrow().tokens.len() as u64)
                .sum::<u64>()
        );
    }

    #[test]
    fn a_finished_slot_holds_its_index_and_emits_nothing_further() {
        let prompts: [&[u32]; 3] = [&[1, 2, 3], &[9], &[4, 4, 4, 4]];
        let mut lm = ToyLm::new(3);
        lm.reset_batch(3).unwrap();
        let mut sinks = Vec::new();
        let mut seeded = Vec::new();
        for (i, p) in prompts.iter().enumerate() {
            let first = lm.prefill_slot(i, p).unwrap();
            let out = Rc::new(RefCell::new(Collected::default()));
            sinks.push(out.clone());

            let max_new = if i == 1 { 2 } else { 20 };
            seeded.push((ToySink { max_new, out }, first));
        }
        let mut batch = Batch::new(seeded).unwrap();
        batch.step(&mut lm).unwrap();
        batch.step(&mut lm).unwrap();
        let after_two = sinks[1].borrow().tokens.clone();
        assert_eq!(batch.live(), 2, "slot 1 should have retired");
        let stats = batch.drain(&mut lm, 40).unwrap();
        assert_eq!(
            sinks[1].borrow().tokens,
            after_two,
            "a retired slot emitted more tokens"
        );
        assert_eq!(sinks[1].borrow().finished, 1);
        assert_eq!(
            stats.slots.len(),
            3,
            "the batch compacted a retired slot away"
        );
        for (i, p) in prompts.iter().enumerate() {
            if i == 1 {
                continue;
            }
            let want = single_stream_reference(p, 20);
            assert_eq!(
                sinks[i].borrow().tokens,
                want,
                "slot {i} was corrupted by slot 1 retiring"
            );
        }
    }

    #[test]
    fn a_short_return_from_the_graph_is_refused_not_realigned() {
        let err = run_batch(&[&[1, 2], &[3, 4], &[5, 6]], 8, 1).expect_err("short return accepted");
        assert!(format!("{err}").contains("decode_step_batch returned"));
    }

    #[test]
    fn stats_separate_aggregate_from_per_stream() {
        let (_, stats) = run_batch(&[&[1, 2], &[3, 4], &[5, 6], &[7, 8]], 12, 0).unwrap();
        assert!(stats.aggregate_tok_s() > 0.0);
        assert!(stats.worst_stream_ms_per_token() >= stats.best_stream_ms_per_token());
        let s = stats.summary();
        assert!(s.contains("aggregate") && s.contains("per-stream"));
    }

    #[test]
    fn batching_is_off_unless_the_knob_is_set() {
        let d = BatchKnobs::default();
        assert_eq!(d.max_batch, 1);
        assert!(!d.enabled());
    }

    fn adm(max_batch: usize, headroom: f64, ctx: u64) -> Admission {
        Admission {
            budget: gemma4_31b_budget(KV_ELEM_BYTES),
            max_seq: ctx,
            knobs: BatchKnobs {
                max_batch,
                headroom_gib: headroom,
                ..BatchKnobs::default()
            },
        }
    }

    #[test]
    fn admission_refuses_slots_that_would_eat_the_reserve() {
        let a = adm(32, 6.0, 32768);
        let per = a.slot_gib();
        assert!(per > 0.0);
        let exactly_four = 6.0 + 4.0 * per;
        assert_eq!(a.memory_slots(exactly_four), 4);
        assert_eq!(a.memory_slots(exactly_four - per), 3);
        assert_eq!(a.memory_slots(1.0), 1);
    }

    #[test]
    fn the_knee_closes_the_batch_before_the_knob_does_at_long_context() {
        let short = adm(64, 0.0, 1024);
        let long = adm(64, 0.0, 131_072);
        assert!(
            long.knee_slots() < short.knee_slots(),
            "knee at 128k ({}) should bind harder than at 1k ({})",
            long.knee_slots(),
            short.knee_slots()
        );
        assert!(long.slots(64, None) < 64);
        assert!(long.knee_slots() >= FREE_ROWS);
    }

    #[test]
    fn the_free_fragment_is_a_floor_the_knee_cannot_cross() {
        let a = Admission {
            budget: StepBudget {
                weight_bytes: 1,
                kv: gemma4_31b_budget(KV_ELEM_BYTES).kv,
            },
            max_seq: 131_072,
            knobs: BatchKnobs {
                max_batch: 32,
                ..BatchKnobs::default()
            },
        };
        assert_eq!(a.knee_slots(), FREE_ROWS);
    }

    #[test]
    fn a_graph_without_a_batch_path_gets_one_slot() {
        assert_eq!(adm(32, 0.0, 1024).slots(1, Some(1024.0)), 1);
    }

    #[test]
    fn kv_geometry_reads_a_gemma4_style_mixed_layer_stack() {
        let raw = r#"{"num_hidden_layers":6,"num_key_value_heads":4,"head_dim":256,
            "num_global_key_value_heads":2,"global_head_dim":512,"sliding_window":1024,
            "layer_types":["sliding_attention","sliding_attention","sliding_attention",
            "sliding_attention","sliding_attention","full_attention"]}"#;
        let g = kv_geometry_from_config(raw, 2).expect("gemma4-shaped config");
        assert_eq!(g.sliding_layers, 5);
        assert_eq!(g.global_layers, 1);
        assert_eq!(g.sliding_bytes_per_pos, 2 * 4 * 256 * 2);
        assert_eq!(g.global_bytes_per_pos, 2 * 2 * 512 * 2);
        assert_eq!(g.bytes_at(2048), 5 * 1024 * 4096 + 2048 * 4096);
    }

    #[test]
    fn the_parser_reproduces_the_hand_written_31b_budget() {
        let mut layers = vec!["\"sliding_attention\"".to_string(); 50];
        layers.extend(std::iter::repeat_n("\"full_attention\"".to_string(), 10));
        let raw = format!(
            r#"{{"num_hidden_layers":60,"num_key_value_heads":16,"head_dim":256,
               "num_global_key_value_heads":4,"global_head_dim":512,"sliding_window":1024,
               "hidden_size":5376,"num_attention_heads":32,"layer_types":[{}]}}"#,
            layers.join(",")
        );
        let parsed = kv_geometry_from_config(&raw, KV_ELEM_BYTES).expect("31B config");
        let hand = gemma4_31b_budget(KV_ELEM_BYTES).kv;
        for ctx in [1024u64, 8192, 32768, 131_072] {
            assert_eq!(
                parsed.bytes_at(ctx),
                hand.bytes_at(ctx),
                "ctx {ctx}: parsed geometry disagrees with the 31B budget"
            );
        }
    }

    #[test]
    fn kv_geometry_charges_an_unwindowed_stack_entirely_as_global() {
        let raw = r#"{"num_hidden_layers":48,"num_key_value_heads":8,"hidden_size":4096,
            "num_attention_heads":32}"#;
        let g = kv_geometry_from_config(raw, 2).expect("dense config");
        assert_eq!(g.sliding_layers, 0);
        assert_eq!(g.global_layers, 48);
        assert_eq!(g.global_bytes_per_pos, 2 * 8 * 128 * 2);
    }

    #[test]
    fn kv_geometry_is_none_when_the_stack_cannot_be_read() {
        assert!(kv_geometry_from_config("{}", 2).is_none());
        assert!(kv_geometry_from_config("not json", 2).is_none());
        assert!(kv_geometry_from_config(r#"{"num_hidden_layers":32}"#, 2).is_none());
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod probe {
    use std::time::Instant;

    use nv_kernels::wgpu_backend::device::WgpuContext;
    use nv_kernels::wgpu_backend::kernels::gemm_coop_f16 as coop;
    use nv_kernels::wgpu_backend::{compose, dispatch};

    const SLC_DEFEAT_BYTES: u64 = 1_800_000_000;

    const COOP_LADDER_SPEEDUPS_ON_RECORD_ARE_FROM_AN_APPLE_SILICON_ADAPTER: &str =
        "the only coop-vs-m-row speedups on record for this ladder were measured on an Apple \
         silicon Metal adapter and say nothing about this box (the coop route outruns the m-row \
         arm at batch); the ladder above prints this adapter's own numbers when the coop arm runs";

    fn mrow_source(m: u32) -> String {
        use std::fmt::Write as _;
        let mut b = String::new();
        b.push_str("struct MrowParams {\n    n_rows: u32,\n    k_elems: u32,\n    row_words: u32,\n    groups_x: u32,\n    m: u32,\n    y_stride: u32,\n    pad0: u32,\n    pad1: u32,\n};\n\n");
        b.push_str("@group(0) @binding(0) var<storage, read> mr_w: array<u32>;\n");
        b.push_str("@group(0) @binding(1) var<storage, read> mr_x: array<u32>;\n");
        b.push_str("@group(0) @binding(2) var<storage, read_write> mr_y: array<f32>;\n");
        b.push_str("@group(0) @binding(3) var<uniform> mr_p: MrowParams;\n\n");
        b.push_str("const MR_LANES: u32 = 32u;\nconst MR_ROWS: u32 = 8u;\n\n");
        b.push_str("var<workgroup> mr_partial: array<f32, 256>;\n\n");
        b.push_str("fn mr_reduce(tid: u32, lane: u32, acc: f32) -> f32 {\n    workgroupBarrier();\n    mr_partial[tid] = acc;\n    workgroupBarrier();\n    for (var stride = MR_LANES >> 1u; stride > 0u; stride = stride >> 1u) {\n        if (lane < stride) {\n            mr_partial[tid] = mr_partial[tid] + mr_partial[tid + stride];\n        }\n        workgroupBarrier();\n    }\n    return mr_partial[tid - lane];\n}\n\n");
        b.push_str("@compute @workgroup_size(256)\n");
        writeln!(b, "fn mrow_bf16_m{m}(").unwrap();
        b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>\n) {\n");
        b.push_str("    let tid = lid.x;\n    let lane = tid & (MR_LANES - 1u);\n    let warp = tid / MR_LANES;\n");
        b.push_str("    let row = (wid.x + wid.y * mr_p.groups_x) * MR_ROWS + warp;\n");
        b.push_str("    let live = row < mr_p.n_rows;\n");
        b.push_str("    let kv = select(0u, mr_p.k_elems >> 3u, live);\n");
        b.push_str("    let w_base = select(0u, row * mr_p.row_words, live);\n");
        for t in 0..m {
            writeln!(b, "    var acc{t} = 0.0;").unwrap();
        }
        b.push_str("    for (var v = lane; v < kv; v = v + MR_LANES) {\n");
        b.push_str("        let wo = w_base + (v << 2u);\n        let xo = v << 2u;\n");
        b.push_str("        for (var j = 0u; j < 4u; j = j + 1u) {\n");
        b.push_str("            let ww = mr_w[wo + j];\n            let wl = bf16_lo(ww);\n            let wh = bf16_hi(ww);\n");
        for t in 0..m {
            writeln!(
                b,
                "            let xw{t} = mr_x[{t}u * mr_p.row_words + xo + j];\n            acc{t} = acc{t} + (wl * bf16_lo(xw{t}) + wh * bf16_hi(xw{t}));"
            )
            .unwrap();
        }
        b.push_str("        }\n    }\n");
        for t in 0..m {
            writeln!(b, "    {{\n        let total{t} = mr_reduce(tid, lane, acc{t});\n        if (lane == 0u && live) {{ mr_y[{t}u * mr_p.y_stride + row] = total{t}; }}\n    }}").unwrap();
        }
        b.push_str("}\n");
        compose(&b)
    }

    struct Lcg(u64);
    impl Lcg {
        fn next_unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((self.0 >> 32) as u32 & 0xff_ffff) as f32 / 16777216.0) * 2.0 - 1.0
        }
    }

    fn bf16_bits(x: f32) -> u16 {
        let b = x.to_bits();
        ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
    }

    fn shared_values(n: usize, seed: u64) -> Vec<f32> {
        let mut r = Lcg(seed);
        (0..n)
            .map(|_| {
                let u = r.next_unit();
                let v = u.signum() * (0.25 + 0.25 * u.abs());
                f32::from_bits((bf16_bits(v) as u32) << 16)
            })
            .collect()
    }

    fn pack_bf16(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 2);
        for x in v {
            out.extend_from_slice(&bf16_bits(*x).to_le_bytes());
        }
        out
    }

    fn to_f16(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 2);
        for x in v {
            let b = x.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
            assert!(
                (1..=30).contains(&exp),
                "operand {x} is outside f16's normal range; the ladder's rates would \
                 include denormal handling in one arm only"
            );
            let mant = ((b & 0x7f_ffff) >> 13) as u16;
            out.extend_from_slice(&(sign | ((exp as u16) << 10) | mant).to_le_bytes());
        }
        out
    }

    struct Replicas {
        buf: dispatch::GpuTensor<u8>,
        stride: u64,
        count: u64,
    }

    fn replicas(ctx: &WgpuContext, label: &str, one: &[u8], target: u64) -> Replicas {
        let stride = (one.len() as u64).div_ceil(256) * 256;
        let max = ctx
            .caps
            .max_buffer_size
            .min(ctx.caps.max_storage_buffer_binding_size);
        let count = target
            .div_ceil(stride)
            .clamp(1, 64)
            .min((max / stride).max(1));
        let mut flat = vec![0u8; (stride * count) as usize];
        for i in 0..count {
            let o = (i * stride) as usize;
            flat[o..o + one.len()].copy_from_slice(one);
        }
        Replicas {
            buf: dispatch::GpuTensor::upload(ctx, label, &flat),
            stride,
            count,
        }
    }

    macro_rules! timed {
        ($ctx:expr, $pipeline:expr, $binds:expr, $groups:expr, $passes:expr) => {{
            let submit = |n: usize| {
                let mut enc = $ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&$pipeline);
                    for i in 0..n {
                        pass.set_bind_group(0, &$binds[i % $binds.len()], &[]);
                        pass.dispatch_workgroups($groups.0, $groups.1, $groups.2);
                    }
                }
                $ctx.queue.submit([enc.finish()]);
                $ctx.poll_blocking().unwrap();
            };
            submit(2);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t0 = Instant::now();
                submit($passes);
                best = best.min(t0.elapsed().as_secs_f64() / $passes as f64);
            }
            best
        }};
    }

    fn bench_mrow(ctx: &WgpuContext, w: &Replicas, m: u32, n: u32, k: u32, passes: usize) -> f64 {
        let x = shared_values((m * k) as usize, 0x3333_4444);
        let pipeline =
            dispatch::compute_pipeline(ctx, "mrow", &mrow_source(m), &format!("mrow_bf16_m{m}"))
                .unwrap_or_else(|e| panic!("mrow pipeline m={m}: {e}"));
        let xbuf = dispatch::storage_from_slice(ctx, "mrow-x", &pack_bf16(&x));
        let y = dispatch::storage_zeroed(ctx, "mrow-y", (m as u64) * (n as u64) * 4);
        let groups = dispatch::workgroup_count_1d(ctx, n.div_ceil(8) as u64, 1);
        let pbuf =
            dispatch::uniform_from(ctx, "mrow-p", &[n, k, k / 2, groups.0, m, n, 0u32, 0u32]);
        let binds: Vec<_> = (0..w.count)
            .map(|i| {
                dispatch::bind_group_offsets(
                    ctx,
                    &pipeline,
                    &[
                        (0, w.buf.raw(), i * w.stride),
                        (1, &xbuf, 0),
                        (2, &y, 0),
                        (3, &pbuf, 0),
                    ],
                )
            })
            .collect();
        timed!(ctx, pipeline, binds, groups, passes)
    }

    fn assert_n_and_k_are_whole_fragments(g: coop::CoopGemm, n: u32, k: u32) {
        assert!(
            n.is_multiple_of(g.tile) && k.is_multiple_of(g.tile),
            "N={n} K={k} are not whole {t}x{t} fragments: the epilogue stores whole fragments, \
             so a ragged N writes into the next row and a ragged K drops the tail of the \
             reduction -- either way the coop column below would be a fast wrong number",
            t = g.tile
        );
    }

    fn bench_coop(
        ctx: &WgpuContext,
        g: coop::CoopGemm,
        w: &Replicas,
        m: u32,
        n: u32,
        k: u32,
        cfg: (u32, u32, u32, u32),
        passes: usize,
    ) -> f64 {
        assert_n_and_k_are_whole_fragments(g, n, k);
        let (tm, tn, sg, ku) = cfg;
        let (bm, bn) = g.grid(m, n, tm, tn, sg);

        let rows = bm * g.rows_per_block(tm);
        let x = shared_values((rows * k) as usize, 0x3333_4444);
        let src = g.source(tm, tn, sg, ku);
        let entry = g.entry(tm, tn, sg, ku);
        let pipeline = dispatch::compute_pipeline(ctx, "coop", &src, &entry)
            .unwrap_or_else(|e| panic!("coop pipeline {} {cfg:?}: {e}", g.request().label()));
        let xbuf = dispatch::storage_from_slice(ctx, "coop-x", &to_f16(&x));
        let y = dispatch::storage_zeroed(ctx, "coop-y", (rows as u64) * (n as u64) * 4);
        let zero = dispatch::storage_from_slice(ctx, "coop-zero", &vec![0f32; g.zero_elems()]);
        let groups = dispatch::workgroup_count_1d(ctx, (bm as u64) * (bn as u64), 1);
        let pbuf = dispatch::uniform_from(
            ctx,
            "coop-p",
            &coop::CoopGemmParams {
                n_rows: n,
                k_elems: k,
                m_rows: m,
                blocks_n: bn,
                y_stride: n,
                groups_x: groups.0,
                pad0: 0,
                pad1: 0,
            },
        );
        let binds: Vec<_> = (0..w.count)
            .map(|i| {
                dispatch::bind_group_offsets(
                    ctx,
                    &pipeline,
                    &[
                        (0, w.buf.raw(), i * w.stride),
                        (1, &xbuf, 0),
                        (2, &y, 0),
                        (3, &pbuf, 0),
                        (4, &zero, 0),
                    ],
                )
            })
            .collect();
        timed!(ctx, pipeline, binds, groups, passes)
    }

    fn best_coop(
        ctx: &WgpuContext,
        g: coop::CoopGemm,
        w: &Replicas,
        m: u32,
        n: u32,
        k: u32,
        passes: usize,
    ) -> (f64, (u32, u32, u32, u32)) {
        let mut best = (f64::INFINITY, (0, 0, 0, 0));
        let mut benched = 0usize;
        for acc in [8u32, 16, 32] {
            let (tm, tn) = g.tiles(m, acc);
            if tm * tn > 32 || !g.acc_fits_a_register_file(tm, tn) {
                continue;
            }
            for sg in [2u32, 4] {
                for ku in [1u32, 2] {
                    if !k.is_multiple_of(g.tile * ku) {
                        continue;
                    }
                    let cfg = (tm, tn, sg, ku);
                    if cfg == best.1 {
                        continue;
                    }
                    benched += 1;
                    let t = bench_coop(ctx, g, w, m, n, k, cfg, passes);
                    if t < best.0 {
                        best = (t, cfg);
                    }
                }
            }
        }
        assert!(
            benched > 0 && best.0.is_finite(),
            "B={m} N={n} K={k}: no {} config survived the tm*tn<=32 fragment cap, the \
             {}-dword-per-lane accumulator budget and K being whole {}-element fragments, so the \
             coop column had nothing to time and would have printed inf as a measurement",
            g.request().label(),
            coop::ACC_LANE_DWORD_BUDGET,
            g.tile
        );
        best
    }

    fn assert_coop_is_computing_at_small_batch(ctx: &WgpuContext, g: coop::CoopGemm, m: u32) {
        let (n, k) = (128u32, 256u32);
        assert_n_and_k_are_whole_fragments(g, n, k);
        let (tm, tn) = g.tiles(m, coop::ACC_FRAGS);
        assert!(
            g.acc_fits_a_register_file(tm, tn),
            "B={m}: the oracle check would dispatch {tm}x{tn} accumulator fragments of {t}x{t}, \
             {} dwords per lane against a {} budget; a config over that budget wedged this \
             adapter for over fifteen minutes on a single dispatch",
            g.acc_lane_dwords(tm, tn),
            coop::ACC_LANE_DWORD_BUDGET,
            t = g.tile
        );
        let (sg, ku) = (2u32, 1u32);
        let (bm, bn) = g.grid(m, n, tm, tn, sg);
        let rows = bm * g.rows_per_block(tm);
        let w = shared_values((n * k) as usize, 0x5eed_0001);
        let x = shared_values((rows * k) as usize, 0x5eed_0002);
        let src = g.source(tm, tn, sg, ku);
        let entry = g.entry(tm, tn, sg, ku);
        let pipeline = dispatch::compute_pipeline(ctx, "coop-check", &src, &entry)
            .unwrap_or_else(|e| panic!("coop pipeline {}: {e}", g.request().label()));
        let wbuf = dispatch::storage_from_slice(ctx, "chk-w", &to_f16(&w));
        let xbuf = dispatch::storage_from_slice(ctx, "chk-x", &to_f16(&x));
        let y = dispatch::storage_zeroed(ctx, "chk-y", (rows as u64) * (n as u64) * 4);
        let zero = dispatch::storage_from_slice(ctx, "chk-zero", &vec![0f32; g.zero_elems()]);
        let groups = dispatch::workgroup_count_1d(ctx, (bm as u64) * (bn as u64), 1);
        let pbuf = dispatch::uniform_from(
            ctx,
            "chk-p",
            &coop::CoopGemmParams {
                n_rows: n,
                k_elems: k,
                m_rows: m,
                blocks_n: bn,
                y_stride: n,
                groups_x: groups.0,
                pad0: 0,
                pad1: 0,
            },
        );
        let bind = dispatch::bind_group_offsets(
            ctx,
            &pipeline,
            &[
                (0, &wbuf, 0),
                (1, &xbuf, 0),
                (2, &y, 0),
                (3, &pbuf, 0),
                (4, &zero, 0),
            ],
        );
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        ctx.queue.submit([enc.finish()]);
        ctx.poll_blocking().unwrap();
        let got: Vec<f32> = dispatch::read_back(ctx, &y, (rows * n) as usize).unwrap();

        let mut num = 0f64;
        let mut den = 0f64;
        for mi in 0..m as usize {
            for ni in 0..n as usize {
                let mut s = 0f64;
                for kk in 0..k as usize {
                    s += (x[mi * k as usize + kk] as f64) * (w[ni * k as usize + kk] as f64);
                }
                let d = got[mi * n as usize + ni] as f64 - s;
                num += d * d;
                den += s * s;
            }
        }
        let rel = (num / den.max(1e-30)).sqrt();
        assert!(den > 0.0, "B={m}: the f64 oracle is identically zero");
        assert!(
            rel < 3e-3,
            "B={m}: coop output is {rel:.3e} off a f64 oracle -- the arm is not computing \
             the batch it is being timed for"
        );
    }

    #[test]
    #[ignore = "measurement probe, 40+ min in debug: run via nvk.sh probe (release + \
                exclusive GPU lock), never in the correctness gate path"]
    fn decode_batch_ladder() {
        let ctx = match WgpuContext::shared() {
            Ok(c) => c,
            Err(e) => panic!("no wgpu adapter: {e}"),
        };
        eprintln!("adapter: {}", ctx.summary());
        let selected = coop::select(ctx, coop::Operand::F16);
        match &selected {
            Ok(g) => {
                eprintln!(
                    "coop fragment selected from the adapter's advertised list: {}; the ladder \
                     emits coop_mat{t}x{t} entry points",
                    g.request().label(),
                    t = g.tile
                );
                for b in [1u32, 2, 4, 8] {
                    assert_coop_is_computing_at_small_batch(ctx, *g, b);
                }
                eprintln!("coop output verified against a f64 oracle at B = 1, 2, 4, 8");
            }
            Err(why) => eprintln!(
                "\n!!! COOP ARM NOT MEASURED: {why}\n\
                 !!! every coop column below reads `--`, and THIS TEST WILL FAIL at the end.\n\
                 !!! The m-row columns are real measurements; the coop ones do not exist."
            ),
        }

        let shapes: [(&str, u32, u32); 4] = [
            ("31B gate_up 43008x5376", 43008, 5376),
            ("31B down     5376x21504", 5376, 21504),
            ("31B q_proj   8192x5376 ", 8192, 5376),
            ("31B KV win  16384x256  ", 16384, 256),
        ];
        let batches = [1u32, 2, 4, 8, 16, 32, 64];
        let mut rate_b1: Vec<(&str, f64)> = Vec::new();

        for (label, n, k) in shapes {
            let w = shared_values((n as usize) * (k as usize), 0x1111_2222);
            let wbf16 = replicas(ctx, "ladder-w-bf16", &pack_bf16(&w), SLC_DEFEAT_BYTES);
            let wf16 = selected
                .is_ok()
                .then(|| replicas(ctx, "ladder-w-f16", &to_f16(&w), SLC_DEFEAT_BYTES));
            drop(w);

            let wbytes = (n as f64) * (k as f64) * 2.0;
            let passes = ((3.0e9 / wbytes).ceil() as usize).clamp(8, 512);
            eprintln!(
                "\n{label}: {} replicas x {:.1} MiB cycled; {passes} dispatches/timing, best of 3",
                wbf16.count,
                wbf16.stride as f64 / 1048576.0
            );

            let null_a = bench_mrow(ctx, &wbf16, 1, n, k, passes);
            let null_b = bench_mrow(ctx, &wbf16, 1, n, k, passes);
            eprintln!(
                "  null control  m-row B=1 twice: {:.4} / {:.4} ms  spread {:+.2}%",
                null_a * 1e3,
                null_b * 1e3,
                (null_b / null_a - 1.0) * 100.0
            );

            eprintln!(
                "  B      m-row ms  ms/row    GB/s   aggr    lat |   coop ms  ms/row    GB/s   aggr    lat | coop/m-row"
            );
            let mut t_mrow_1 = 0.0;
            let mut t_coop_1 = 0.0;
            for b in batches {
                let tm_ = bench_mrow(ctx, &wbf16, b, n, k, passes);
                let coop = match (&selected, wf16.as_ref()) {
                    (Ok(g), Some(wf16)) => {
                        let (tc, cfg) = best_coop(ctx, *g, wf16, b, n, k, passes);
                        Some((tc, cfg, g.tile))
                    }
                    _ => None,
                };
                if b == 1 {
                    t_mrow_1 = tm_;
                    if let Some((tc, _, _)) = coop {
                        t_coop_1 = tc;
                    }
                    rate_b1.push((label, wbytes / tm_ / 1e9));
                }
                let bytes = wbytes + (b as f64) * (k as f64) * 2.0 + (b as f64) * (n as f64) * 4.0;
                let mrow = format!(
                    "  {b:<4} {:>9.4} {:>7.4} {:>7.1} {:>5.2}x {:>5.2}x",
                    tm_ * 1e3,
                    tm_ * 1e3 / b as f64,
                    bytes / tm_ / 1e9,
                    t_mrow_1 * b as f64 / tm_,
                    tm_ / t_mrow_1,
                );
                match coop {
                    Some((tc, cfg, tile)) => eprintln!(
                        "{mrow} | {:>9.4} {:>7.4} {:>7.1} {:>5.2}x {:>5.2}x | {:>6.2}x  coop{tile} tm{} tn{} sg{} ku{}",
                        tc * 1e3,
                        tc * 1e3 / b as f64,
                        bytes / tc / 1e9,
                        t_coop_1 * b as f64 / tc,
                        tc / t_coop_1,
                        tm_ / tc,
                        cfg.0,
                        cfg.1,
                        cfg.2,
                        cfg.3
                    ),
                    None => eprintln!(
                        "{mrow} | {:>9} {:>7} {:>7} {:>6} {:>6} | {:>6}  NOT MEASURED",
                        "--", "--", "--", "--", "--", "--"
                    ),
                }
            }
            let drift = bench_mrow(ctx, &wbf16, 1, n, k, passes);
            eprintln!(
                "  drift check   m-row B=1 after the ladder: {:.4} ms  {:+.2}% vs the null pair",
                drift * 1e3,
                (drift / null_a.min(null_b) - 1.0) * 100.0
            );
        }

        let weight_rate = rate_b1
            .iter()
            .find(|(l, _)| l.contains("gate_up"))
            .map(|(_, r)| *r)
            .expect("gate_up arm did not run");
        let kv_rate = rate_b1
            .iter()
            .find(|(l, _)| l.contains("KV win"))
            .map(|(_, r)| *r)
            .expect("KV arm did not run");
        eprintln!(
            "\nknee for the 31B, weight stream priced at {weight_rate:.1} GB/s and KV at \
             {kv_rate:.1} GB/s. The KV element width is the checkpoint's, not a constant: \
             an fp8 cache doubles every knee."
        );
        eprintln!("  kv_elem  ctx      KV GB/step/stream   byte knee B*   time knee B*");
        for elem in [2u64, 1] {
            let budget = super::gemma4_31b_budget(elem);
            for l in [1024u64, 4096, 8192, 32768, 131072] {
                let kv = budget.kv.bytes_at(l) as f64;
                let byte_knee = budget.weight_bytes as f64 / kv;
                eprintln!(
                    "  {elem}B       {l:<8} {:>15.3}   {byte_knee:>12.1}   {:>12.1}",
                    kv / 1e9,
                    byte_knee * kv_rate / weight_rate
                );
            }
        }

        if let Err(why) = selected {
            panic!(
                "the m-row ladder above ran and its numbers are real; the COOP LADDER WAS NOT \
                 MEASURED and this run proves nothing about it. \
                 gemm_coop_f16::select(ctx, Operand::F16) -- which checks the device features, \
                 then the subgroup width, then every fragment shape in {:?} against the \
                 adapter's advertised list -- returned no usable fragment: {why}. That is a \
                 statement about THIS ADAPTER, not about the tree: gemm_coop_f16 emits \
                 coop_mat8x8 AND coop_mat16x16, and this ladder threads the selected CoopGemm \
                 through source/entry/grid/tiles/rows_per_block/zero_elems, so an adapter \
                 advertising 16x16x16 f16xf16->f32 at subgroup width 32 is measured here with no \
                 further porting. The adapter reports {} cooperative-matrix config(s), \
                 coop_gemm_tile()={:?}, coop_gemm_reason()={:?}. \
                 {COOP_LADDER_SPEEDUPS_ON_RECORD_ARE_FROM_AN_APPLE_SILICON_ADAPTER}. Adapter: {}",
                coop::TILES,
                ctx.caps.coop_configs.len(),
                ctx.caps.coop_gemm_tile(),
                ctx.caps.coop_gemm_reason(),
                ctx.summary()
            );
        }
    }
}
