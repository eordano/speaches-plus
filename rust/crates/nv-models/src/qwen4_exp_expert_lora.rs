use super::*;

pub const EXPERT_LORA_ADAPTS_THE_ROUTED_DOWN_PROJECTION_ON_THE_BF16_ACTIVATION_SO_THE_NVFP4_GROUPED_GEMMS_STAY_BIT_IDENTICAL: &str =
    "the delta B_e A_e act_e(x) is added to each routed expert's down-projection output and summed \
     with the router weights; act_e is silu(gate)*up from the served W4A4 grouped gate/up GEMM before \
     its NVFP4 requantization, and the delta is accumulated in f32 beside the untouched grouped \
     output, so a zero adapter reproduces the served forward bitwise";

pub const EXPERT_LORA_SORTED_TILE_16_ROWS_KEEPS_PADDING_BELOW_ONE_TILE_PER_ACTIVE_EXPERT: usize = 16;

pub const EXPERT_LORA_FILE: &str = "expert_lora.safetensors";
pub const EXPERT_LORA_META: &str = "expert_lora.json";
pub const EXPERT_LORA_FORMAT: &str = "nv-qwen4exp-expert-lora/1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertLoraForm {
    None,
    PerExpert,
    ExpertA,
    Hydra,
    Shared,
}

impl ExpertLoraForm {
    pub const ALL: [ExpertLoraForm; 5] = [Self::None, Self::PerExpert, Self::ExpertA, Self::Hydra, Self::Shared];

    pub fn parse(s: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|f| f.as_str() == s)
            .ok_or_else(|| anyhow::anyhow!("expert-lora form {s:?} is not one of none, per-expert, expert-a, hydra, shared"))
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PerExpert => "per-expert",
            Self::ExpertA => "expert-a",
            Self::Hydra => "hydra",
            Self::Shared => "shared",
        }
    }

    pub fn stacks(&self, experts: usize) -> (usize, usize) {
        match self {
            Self::None => (0, 0),
            Self::PerExpert => (experts, experts),
            Self::ExpertA => (experts, 1),
            Self::Hydra => (1, experts),
            Self::Shared => (1, 1),
        }
    }

    pub fn of_stacks(ea: usize, eb: usize, experts: usize) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|f| *f != Self::None && f.stacks(experts) == (ea, eb))
            .ok_or_else(|| anyhow::anyhow!("expert-lora stacks A {ea} / B {eb} match no form for {experts} experts"))
    }

    pub fn params_per_layer(&self, experts: usize, hidden: usize, inter: usize, rank: usize) -> usize {
        let (ea, eb) = self.stacks(experts);
        rank * (ea * inter + eb * hidden)
    }
}

pub fn expert_module(li: usize) -> String {
    format!("{}.mlp.experts", layer_prefix(li))
}

pub struct ExpertLoraLayer {
    a: Tensor,
    b: Tensor,
    scale: f64,
}

struct TilePlan {
    sorted_slot: Tensor,
    tile_experts: Tensor,
    slot_pos: Tensor,
    rows: usize,
}

impl TilePlan {
    fn new(slot_experts: &[u32], experts: usize, device: &Device) -> Result<Self> {
        let tile = EXPERT_LORA_SORTED_TILE_16_ROWS_KEEPS_PADDING_BELOW_ONE_TILE_PER_ACTIVE_EXPERT;
        let s = slot_experts.len();
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); experts];
        for (slot, &e) in slot_experts.iter().enumerate() {
            ensure!((e as usize) < experts, "expert lora: routed expert {e} outside {experts}");
            buckets[e as usize].push(slot as u32);
        }
        let (mut sorted, mut tiles, mut pos) = (Vec::new(), Vec::new(), vec![0u32; s]);
        for (e, b) in buckets.iter().enumerate() {
            for chunk in b.chunks(tile) {
                tiles.push(e as u32);
                for j in 0..tile {
                    match chunk.get(j) {
                        Some(&slot) => {
                            pos[slot as usize] = sorted.len() as u32;
                            sorted.push(slot);
                        }
                        None => sorted.push(s as u32),
                    }
                }
            }
        }
        let rows = sorted.len();
        Ok(Self {
            sorted_slot: Tensor::from_vec(sorted, (rows,), device)?,
            tile_experts: Tensor::from_vec(tiles.clone(), (tiles.len(),), device)?,
            slot_pos: Tensor::from_vec(pos, (s,), device)?,
            rows,
        })
    }

    fn tiles(&self) -> usize {
        self.rows / EXPERT_LORA_SORTED_TILE_16_ROWS_KEEPS_PADDING_BELOW_ONE_TILE_PER_ACTIVE_EXPERT
    }

    fn sort(&self, slot_rows: &Tensor) -> Result<Tensor> {
        let cols = slot_rows.dim(1)?;
        let padded = Tensor::cat(&[slot_rows, &Tensor::zeros((1, cols), slot_rows.dtype(), slot_rows.device())?], 0)?;
        Ok(padded.index_select(&self.sorted_slot, 0)?)
    }
}

impl ExpertLoraLayer {
    pub fn new(a: Tensor, b: Tensor, scale: f64) -> Result<Self> {
        let (ea, r, _) = a.dims3()?;
        let (eb, _, rb) = b.dims3()?;
        ensure!(r == rb && ea >= 1 && eb >= 1, "expert lora: A {:?} and B {:?} disagree on the rank", a.dims(), b.dims());
        Ok(Self { a, b, scale })
    }

    pub fn rank(&self) -> usize {
        self.a.dims()[1]
    }

    pub fn inter(&self) -> usize {
        self.a.dims()[2]
    }

    pub fn hidden(&self) -> usize {
        self.b.dims()[1]
    }

    pub fn stacks(&self) -> (usize, usize) {
        (self.a.dims()[0], self.b.dims()[0])
    }

    pub fn tensors(&self) -> (&Tensor, &Tensor) {
        (&self.a, &self.b)
    }

    pub fn device_bytes(&self) -> usize {
        self.a.elem_count() * self.a.dtype().size_in_bytes() + self.b.elem_count() * self.b.dtype().size_in_bytes()
    }

    pub fn delta(&self, act_slots: &Tensor, slot_experts: &[u32], weights: &Tensor, n: usize, k: usize, track: bool) -> Result<Tensor> {
        let (a, b) = if track { (self.a.clone(), self.b.clone()) } else { (self.a.detach(), self.b.detach()) };
        let (a, b) = (a.to_dtype(DType::F32)?, b.to_dtype(DType::F32)?);
        let (ea, r, inter) = a.dims3()?;
        let (eb, h, _) = b.dims3()?;
        let s = n * k;
        ensure!(act_slots.dims() == [s, inter] && slot_experts.len() == s, "expert lora: {s} slots vs act {:?}", act_slots.dims());
        let act = act_slots.to_dtype(DType::F32)?;
        let w = weights.to_dtype(DType::F32)?.reshape((s, 1))?;
        let tile = EXPERT_LORA_SORTED_TILE_16_ROWS_KEEPS_PADDING_BELOW_ONE_TILE_PER_ACTIVE_EXPERT;
        let plan = if ea > 1 || eb > 1 { Some(TilePlan::new(slot_experts, ea.max(eb), act.device())?) } else { None };
        let u = match &plan {
            Some(p) if ea > 1 => {
                let at = a.index_select(&p.tile_experts, 0)?;
                let u = p.sort(&act)?.reshape((p.tiles(), tile, inter))?.matmul(&at.transpose(1, 2)?)?;
                u.reshape((p.rows, r))?.index_select(&p.slot_pos, 0)?
            }
            _ => act.matmul(&a.get(0)?.t()?)?,
        };
        let wu = u.broadcast_mul(&w)?;
        let out = match &plan {
            Some(p) if eb > 1 => {
                let bt = b.index_select(&p.tile_experts, 0)?;
                let d = p.sort(&wu)?.reshape((p.tiles(), tile, r))?.matmul(&bt.transpose(1, 2)?)?;
                d.reshape((p.rows, h))?.index_select(&p.slot_pos, 0)?.reshape((n, k, h))?.sum(1)?
            }
            _ => wu.reshape((n, k, r))?.sum(1)?.matmul(&b.get(0)?.t()?)?,
        };
        Ok(if self.scale == 1.0 { out } else { out.affine(self.scale, 0.0)? })
    }
}

impl MoeLayer {
    pub fn attach_expert_lora(&self, l: Arc<ExpertLoraLayer>, cfg: &Qwen4ExpConfig) -> Result<()> {
        let (ea, eb) = l.stacks();
        ExpertLoraForm::of_stacks(ea, eb, cfg.num_experts)?;
        ensure!(
            l.inter() == cfg.moe_intermediate_size && l.hidden() == cfg.hidden_size,
            "expert lora inter {} / hidden {} vs the model's {} / {}",
            l.inter(),
            l.hidden(),
            cfg.moe_intermediate_size,
            cfg.hidden_size
        );
        *self.expert_lora.write().map_err(|e| anyhow::anyhow!("expert lora lock poisoned: {e}"))? = Some(l);
        Ok(())
    }

    pub fn detach_expert_lora(&self) {
        if let Ok(mut s) = self.expert_lora.write() {
            *s = None;
        }
    }

    pub fn expert_lora(&self) -> Option<Arc<ExpertLoraLayer>> {
        self.expert_lora.read().ok().and_then(|s| s.clone())
    }

    pub(crate) fn served_expert_lora_delta(&self, l: &ExpertLoraLayer, x: &Tensor, logits: &Tensor, cfg: &Qwen4ExpConfig) -> Result<Tensor> {
        let (t, _) = x.dims2()?;
        let k = cfg.num_experts_per_tok;
        let rows = nv_layers::moe_grouped::routed_expert_act_rows_through_the_served_w4a4_gate_up(&self.grouped, x, logits, t, k, x.device())?;
        l.delta(&rows.act_slots, &rows.slot_experts, &rows.slot_weights, t, k, false)
    }
}

pub fn init_expert_lora(form: ExpertLoraForm, experts: usize, hidden: usize, inter: usize, rank: usize, seed: u64, device: &Device) -> Result<(Tensor, Tensor)> {
    let (ea, eb) = form.stacks(experts);
    ensure!(ea > 0 && rank > 0, "expert lora form {} at rank {rank} has nothing to train", form.as_str());
    let mut rng = seed | 1;
    let bound = 1.0 / (inter as f32).sqrt();
    let a: Vec<f32> = (0..ea * rank * inter)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * bound
        })
        .collect();
    Ok((Tensor::from_vec(a, (ea, rank, inter), device)?, Tensor::zeros((eb, hidden, rank), DType::F32, device)?))
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ExpertLoraMeta {
    pub format: String,
    pub form: String,
    pub rank: usize,
    pub layers: Vec<usize>,
    pub num_experts: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub base_identity: Option<String>,
}

impl ExpertLoraMeta {
    pub fn validate_against(&self, cfg: &Qwen4ExpConfig, base_identity: Option<&str>) -> Result<ExpertLoraForm> {
        ensure!(self.format == EXPERT_LORA_FORMAT, "expert lora format {:?}, this loader reads {EXPERT_LORA_FORMAT:?}", self.format);
        let form = ExpertLoraForm::parse(&self.form)?;
        ensure!(form != ExpertLoraForm::None, "an expert_lora.json with form none must not exist");
        ensure!(
            self.num_experts == cfg.num_experts && self.hidden_size == cfg.hidden_size && self.moe_intermediate_size == cfg.moe_intermediate_size,
            "expert lora was trained for {} experts, hidden {}, inter {}; this model has {}, {}, {}",
            self.num_experts,
            self.hidden_size,
            self.moe_intermediate_size,
            cfg.num_experts,
            cfg.hidden_size,
            cfg.moe_intermediate_size
        );
        ensure!(self.rank >= 1 && !self.layers.is_empty(), "expert lora rank {} over layers {:?}", self.rank, self.layers);
        for &li in &self.layers {
            ensure!(li < cfg.num_hidden_layers, "expert lora layer {li} outside the model's {} layers", cfg.num_hidden_layers);
        }
        if let (Some(want), Some(have)) = (self.base_identity.as_deref(), base_identity) {
            ensure!(want == have, "expert lora was trained against base {want}, this engine serves {have}");
        }
        Ok(form)
    }

    pub fn check_tensors(&self, form: ExpertLoraForm, li: usize, a: &[usize], b: &[usize]) -> Result<()> {
        let (ea, eb) = form.stacks(self.num_experts);
        ensure!(
            a == [ea, self.rank, self.moe_intermediate_size] && b == [eb, self.hidden_size, self.rank],
            "layer {li}: expert lora A {a:?} / B {b:?}, form {} at rank {} expects [{ea}, {}, {}] / [{eb}, {}, {}]",
            form.as_str(),
            self.rank,
            self.rank,
            self.moe_intermediate_size,
            self.hidden_size,
            self.rank
        );
        Ok(())
    }
}

struct Fnv128(u128);

impl Fnv128 {
    fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u128;
            self.0 = self.0.wrapping_mul(0x0000000001000000000000000000013B);
        }
    }
}

pub fn content_hash(parts: &[&[u8]]) -> String {
    let mut h = Fnv128(0x6c62272e07bb014262b821756295c58d);
    for p in parts {
        h.update(&(p.len() as u64).to_le_bytes());
        h.update(p);
    }
    format!("{:032x}", h.0)
}

pub fn base_identity_for_dir(dir: &Path) -> Result<String> {
    use std::io::Read as _;
    let mut h = Fnv128(0x6c62272e07bb014262b821756295c58d);
    h.update(&std::fs::read(dir.join("config.json")).with_context(|| format!("read {}/config.json", dir.display()))?);
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    for f in files {
        h.update(f.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default().as_bytes());
        let mut file = std::fs::File::open(&f)?;
        h.update(&file.metadata()?.len().to_le_bytes());
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let mut head = vec![0u8; u64::from_le_bytes(len).min(1 << 24) as usize];
        file.read_exact(&mut head)?;
        h.update(&head);
    }
    Ok(format!("fnv128:{:032x}", h.0))
}

pub fn write_expert_lora(dir: &Path, meta: &ExpertLoraMeta, layers: &[(usize, Tensor, Tensor)]) -> Result<()> {
    let mut tensors = HashMap::new();
    for (li, a, b) in layers {
        tensors.insert(format!("{}.lora_A", expert_module(*li)), a.to_dtype(DType::BF16)?.to_device(&Device::Cpu)?);
        tensors.insert(format!("{}.lora_B", expert_module(*li)), b.to_dtype(DType::BF16)?.to_device(&Device::Cpu)?);
    }
    std::fs::create_dir_all(dir)?;
    candle_core::safetensors::save(&tensors, dir.join(EXPERT_LORA_FILE))?;
    std::fs::write(dir.join(EXPERT_LORA_META), serde_json::to_string_pretty(meta)? + "\n")?;
    Ok(())
}
