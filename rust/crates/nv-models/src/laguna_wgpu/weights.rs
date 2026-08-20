use anyhow::{Context, Result};

use super::config::{GateKind, LagunaShapes, LayerShape, LayerType, NVFP4_BLOCK};

pub mod names {
    pub const EMBED: &str = "model.embed_tokens.weight";
    pub const FINAL_NORM: &str = "model.norm.weight";
    pub const LM_HEAD: &str = "lm_head.weight";

    pub fn layer_prefix(idx: usize) -> String {
        format!("model.layers.{idx}")
    }

    pub fn input_layernorm(idx: usize) -> String {
        format!("model.layers.{idx}.input_layernorm.weight")
    }

    pub fn post_attention_layernorm(idx: usize) -> String {
        format!("model.layers.{idx}.post_attention_layernorm.weight")
    }

    pub fn q_proj(idx: usize) -> String {
        format!("model.layers.{idx}.self_attn.q_proj")
    }

    pub fn k_proj(idx: usize) -> String {
        format!("model.layers.{idx}.self_attn.k_proj")
    }

    pub fn v_proj(idx: usize) -> String {
        format!("model.layers.{idx}.self_attn.v_proj")
    }

    pub fn o_proj(idx: usize) -> String {
        format!("model.layers.{idx}.self_attn.o_proj")
    }

    pub fn g_proj(idx: usize) -> String {
        format!("model.layers.{idx}.self_attn.g_proj")
    }

    pub fn q_norm(idx: usize) -> String {
        format!("model.layers.{idx}.self_attn.q_norm.weight")
    }

    pub fn k_norm(idx: usize) -> String {
        format!("model.layers.{idx}.self_attn.k_norm.weight")
    }

    pub fn dense_gate_proj(idx: usize) -> String {
        format!("model.layers.{idx}.mlp.gate_proj")
    }

    pub fn dense_up_proj(idx: usize) -> String {
        format!("model.layers.{idx}.mlp.up_proj")
    }

    pub fn dense_down_proj(idx: usize) -> String {
        format!("model.layers.{idx}.mlp.down_proj")
    }

    pub fn router(idx: usize) -> String {
        format!("model.layers.{idx}.mlp.gate.weight")
    }

    pub fn selection_bias(idx: usize) -> String {
        format!("model.layers.{idx}.mlp.experts.e_score_correction_bias")
    }

    pub fn expert_gate_proj(idx: usize, e: usize) -> String {
        format!("model.layers.{idx}.mlp.experts.{e}.gate_proj")
    }

    pub fn expert_up_proj(idx: usize, e: usize) -> String {
        format!("model.layers.{idx}.mlp.experts.{e}.up_proj")
    }

    pub fn expert_down_proj(idx: usize, e: usize) -> String {
        format!("model.layers.{idx}.mlp.experts.{e}.down_proj")
    }

    pub fn shared_gate_proj(idx: usize) -> String {
        format!("model.layers.{idx}.mlp.shared_expert.gate_proj")
    }

    pub fn shared_up_proj(idx: usize) -> String {
        format!("model.layers.{idx}.mlp.shared_expert.up_proj")
    }

    pub fn shared_down_proj(idx: usize) -> String {
        format!("model.layers.{idx}.mlp.shared_expert.down_proj")
    }

    pub fn bf16_weight(module: &str) -> String {
        format!("{module}.weight")
    }

    pub fn nvfp4_packed(module: &str) -> String {
        format!("{module}.weight_packed")
    }

    pub fn nvfp4_weight_scale(module: &str) -> String {
        format!("{module}.weight_scale")
    }

    pub fn nvfp4_weight_global_scale(module: &str) -> String {
        format!("{module}.weight_global_scale")
    }

    pub fn nvfp4_input_global_scale(module: &str) -> String {
        format!("{module}.input_global_scale")
    }
}

#[derive(Clone, Debug)]
pub struct HostBf16Lin {
    pub w: Vec<u16>,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Debug)]
pub struct HostBf16ExpertStack {
    pub w: Vec<u16>,
    pub e: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Debug)]
pub enum HostLin {
    Bf16(HostBf16Lin),
    Nvfp4(HostNvfp4Lin),
}

impl HostLin {
    pub fn n(&self) -> usize {
        match self {
            HostLin::Bf16(l) => l.n,
            HostLin::Nvfp4(l) => l.n,
        }
    }

    pub fn k(&self) -> usize {
        match self {
            HostLin::Bf16(l) => l.k,
            HostLin::Nvfp4(l) => l.k,
        }
    }

    pub fn is_nvfp4(&self) -> bool {
        matches!(self, HostLin::Nvfp4(_))
    }
}

#[derive(Clone, Debug)]
pub enum HostExperts {
    Bf16(HostBf16ExpertStack),
    Nvfp4(HostNvfp4ExpertStack),
}

impl HostExperts {
    pub fn num_experts(&self) -> usize {
        match self {
            HostExperts::Bf16(s) => s.e,
            HostExperts::Nvfp4(s) => s.e,
        }
    }

    pub fn n(&self) -> usize {
        match self {
            HostExperts::Bf16(s) => s.n,
            HostExperts::Nvfp4(s) => s.n,
        }
    }

    pub fn k(&self) -> usize {
        match self {
            HostExperts::Bf16(s) => s.k,
            HostExperts::Nvfp4(s) => s.k,
        }
    }

    pub fn input_globals(&self) -> Vec<f32> {
        match self {
            HostExperts::Bf16(s) => vec![1.0; s.e],
            HostExperts::Nvfp4(s) => s.input_globals.clone(),
        }
    }

    pub fn expert(&self, e: usize) -> HostLin {
        match self {
            HostExperts::Bf16(s) => HostLin::Bf16(HostBf16Lin {
                w: s.w[e * s.n * s.k..(e + 1) * s.n * s.k].to_vec(),
                n: s.n,
                k: s.k,
            }),
            HostExperts::Nvfp4(s) => HostLin::Nvfp4(expert_slice(s, e)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct HostAttention {
    pub q: HostLin,
    pub k: HostLin,
    pub v: HostLin,
    pub o: HostLin,
    pub g: Option<HostLin>,
    pub q_norm: Vec<u16>,
    pub k_norm: Vec<u16>,
}

#[derive(Clone, Debug)]
pub struct HostDenseMlp {
    pub gate: HostLin,
    pub up: HostLin,
    pub down: HostLin,
}

#[derive(Clone, Debug)]
pub struct HostMoe {
    pub router: HostBf16Lin,
    pub selection_bias: Vec<f32>,
    pub experts_gate: HostExperts,
    pub experts_up: HostExperts,
    pub experts_down: HostExperts,
    pub shared_gate: HostLin,
    pub shared_up: HostLin,
    pub shared_down: HostLin,
}

#[derive(Clone, Debug)]
pub enum HostFfn {
    Dense(Box<HostDenseMlp>),
    Moe(Box<HostMoe>),
}

#[derive(Clone, Debug)]
pub struct HostLayer {
    pub kind: LayerType,
    pub input_ln: Vec<u16>,
    pub post_attn_ln: Vec<u16>,
    pub attn: HostAttention,
    pub ffn: HostFfn,
}

#[derive(Clone, Debug)]
pub struct HostWeights {
    pub embed: Vec<u16>,
    pub final_norm: Vec<u16>,
    pub lm_head: Vec<u16>,
    pub layers: Vec<HostLayer>,
}

pub enum WeightSource<'a> {
    Host(&'a HostWeights),
    Loader(&'a nv_weights::WeightLoader),
}

impl WeightSource<'_> {
    pub fn embed(&self, shapes: &LagunaShapes) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.embed.clone()),
            Self::Loader(w) => {
                load_bf16_named(w, &[names::EMBED], &[shapes.vocab_size, shapes.hidden_size])
            }
        }
    }

    pub fn final_norm(&self, shapes: &LagunaShapes) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.final_norm.clone()),
            Self::Loader(w) => load_bf16_named(w, &[names::FINAL_NORM], &[shapes.hidden_size]),
        }
    }

    pub fn lm_head(&self, shapes: &LagunaShapes) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => {
                if h.lm_head.is_empty() {
                    Ok(h.embed.clone())
                } else {
                    Ok(h.lm_head.clone())
                }
            }
            Self::Loader(w) => {
                if shapes.tie_word_embeddings || !w.has(names::LM_HEAD) {
                    return self.embed(shapes);
                }
                load_bf16_named(
                    w,
                    &[names::LM_HEAD],
                    &[shapes.vocab_size, shapes.hidden_size],
                )
            }
        }
    }

    pub fn layer_input_ln(&self, shapes: &LagunaShapes, idx: usize) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.layers[idx].input_ln.clone()),
            Self::Loader(w) => {
                load_bf16_named(w, &[&names::input_layernorm(idx)], &[shapes.hidden_size])
            }
        }
    }

    pub fn layer(&self, shapes: &LagunaShapes, idx: usize) -> Result<HostLayer> {
        match self {
            Self::Host(h) => {
                anyhow::ensure!(
                    idx < h.layers.len(),
                    "host weights have {} layers, want {idx}",
                    h.layers.len()
                );
                Ok(h.layers[idx].clone())
            }
            Self::Loader(w) => {
                let layer = shapes.layer(idx);
                let ffn = if layer.is_moe() {
                    HostFfn::Moe(Box::new(load_moe(w, shapes, layer)?))
                } else {
                    HostFfn::Dense(Box::new(load_dense_mlp(w, shapes, layer)?))
                };
                Ok(HostLayer {
                    kind: layer.attn_kind,
                    input_ln: load_bf16_named(
                        w,
                        &[&names::input_layernorm(idx)],
                        &[shapes.hidden_size],
                    )?,
                    post_attn_ln: load_bf16_named(
                        w,
                        &[&names::post_attention_layernorm(idx)],
                        &[shapes.hidden_size],
                    )?,
                    attn: load_attention(w, shapes, layer)?,
                    ffn,
                })
            }
        }
    }
}

pub fn load_bf16_named(
    w: &nv_weights::WeightLoader,
    names: &[&str],
    shape: &[usize],
) -> Result<Vec<u16>> {
    for n in names {
        if w.has(n) {
            let t = w
                .get(n, candle_core::DType::BF16)
                .with_context(|| format!("load {n}"))?;
            anyhow::ensure!(t.dims() == shape, "{n}: shape {:?} != {shape:?}", t.dims());
            let v: Vec<half::bf16> = t.flatten_all()?.to_vec1()?;
            return Ok(v.into_iter().map(|x| x.to_bits()).collect());
        }
    }
    anyhow::bail!("none of {names:?} found")
}

fn scalar_f32(w: &nv_weights::WeightLoader, name: &str) -> Result<f32> {
    let t = w
        .get(name, candle_core::DType::F32)
        .with_context(|| format!("load {name}"))?;
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(*v.first().unwrap_or(&1.0))
}

fn load_nvfp4(
    w: &nv_weights::WeightLoader,
    module: &str,
    n: usize,
    k: usize,
) -> Result<HostNvfp4Lin> {
    anyhow::ensure!(
        k.is_multiple_of(NVFP4_BLOCK),
        "{module}: k {k} not a multiple of {NVFP4_BLOCK}"
    );
    let packed_name = names::nvfp4_packed(module);
    let shape = w
        .shape_of(&packed_name)
        .ok_or_else(|| anyhow::anyhow!("missing {packed_name}"))?;
    anyhow::ensure!(
        shape.len() == 2 && shape[0] == n && shape[1] == k / 2,
        "{module}: weight_packed shape {shape:?}, want [{n}, {}]",
        k / 2
    );
    let packed = w.raw_bytes(&packed_name)?.to_vec();
    let scale_raw = w.raw_bytes(&names::nvfp4_weight_scale(module))?.to_vec();
    anyhow::ensure!(
        scale_raw.len() == n * k / NVFP4_BLOCK,
        "{module}: weight_scale {} bytes, want {}",
        scale_raw.len(),
        n * k / NVFP4_BLOCK
    );
    let scales = nv_quant::nvfp4::swizzle_scales(&scale_raw, n, k / NVFP4_BLOCK);
    let gw = scalar_f32(w, &names::nvfp4_weight_global_scale(module))?;
    let gi_name = names::nvfp4_input_global_scale(module);
    let gi = if w.has(&gi_name) {
        scalar_f32(w, &gi_name)?
    } else {
        1.0
    };
    let recip = |x: f32| {
        if x == 0.0 || !x.is_finite() {
            1.0
        } else {
            1.0 / x
        }
    };
    Ok(HostNvfp4Lin {
        packed,
        scales_swizzled: scales,
        alpha: recip(gw) * recip(gi),
        input_global: gi,
        n,
        k,
    })
}

pub fn load_lin(
    w: &nv_weights::WeightLoader,
    module: &str,
    n_out_rows: usize,
    k_in_cols: usize,
) -> Result<HostLin> {
    if w.has(&names::nvfp4_packed(module)) {
        return Ok(HostLin::Nvfp4(load_nvfp4(
            w, module, n_out_rows, k_in_cols,
        )?));
    }
    Ok(HostLin::Bf16(HostBf16Lin {
        w: load_bf16_named(w, &[&names::bf16_weight(module)], &[n_out_rows, k_in_cols])?,
        n: n_out_rows,
        k: k_in_cols,
    }))
}

pub fn load_expert_stack(
    w: &nv_weights::WeightLoader,
    modules: &[String],
    n_out_rows: usize,
    k_in_cols: usize,
) -> Result<HostExperts> {
    anyhow::ensure!(
        !modules.is_empty(),
        "expert stack needs at least one module"
    );
    let mut bf16 = Vec::new();
    let mut nvfp4 = Vec::new();
    for m in modules {
        match load_lin(w, m, n_out_rows, k_in_cols)? {
            HostLin::Bf16(l) => bf16.push(l),
            HostLin::Nvfp4(l) => nvfp4.push(l),
        }
    }
    if !bf16.is_empty() && !nvfp4.is_empty() {
        anyhow::bail!(
            "expert stack {}..: mixed bf16/nvfp4 experts ({} bf16, {} nvfp4)",
            modules[0],
            bf16.len(),
            nvfp4.len()
        );
    }
    if nvfp4.is_empty() {
        Ok(HostExperts::Bf16(stack_bf16_host(&bf16)))
    } else {
        Ok(HostExperts::Nvfp4(stack_nvfp4_host(&nvfp4)))
    }
}

pub fn load_attention(
    w: &nv_weights::WeightLoader,
    shapes: &LagunaShapes,
    layer: &LayerShape,
) -> Result<HostAttention> {
    let hidden = shapes.hidden_size;
    let idx = layer.idx;
    let g = match shapes.gate_kind {
        GateKind::None => None,
        _ => Some(load_lin(w, &names::g_proj(idx), layer.gate_rows, hidden)?),
    };
    Ok(HostAttention {
        q: load_lin(w, &names::q_proj(idx), layer.q_rows, hidden)?,
        k: load_lin(w, &names::k_proj(idx), layer.kv_rows, hidden)?,
        v: load_lin(w, &names::v_proj(idx), layer.kv_rows, hidden)?,
        o: load_lin(w, &names::o_proj(idx), hidden, layer.q_rows)?,
        g,
        q_norm: load_bf16_named(w, &[&names::q_norm(idx)], &[layer.head_dim])?,
        k_norm: load_bf16_named(w, &[&names::k_norm(idx)], &[layer.head_dim])?,
    })
}

pub fn load_dense_mlp(
    w: &nv_weights::WeightLoader,
    shapes: &LagunaShapes,
    layer: &LayerShape,
) -> Result<HostDenseMlp> {
    let hidden = shapes.hidden_size;
    let inter = shapes.dense_intermediate_size;
    let idx = layer.idx;
    Ok(HostDenseMlp {
        gate: load_lin(w, &names::dense_gate_proj(idx), inter, hidden)?,
        up: load_lin(w, &names::dense_up_proj(idx), inter, hidden)?,
        down: load_lin(w, &names::dense_down_proj(idx), hidden, inter)?,
    })
}

pub fn load_moe(
    w: &nv_weights::WeightLoader,
    shapes: &LagunaShapes,
    layer: &LayerShape,
) -> Result<HostMoe> {
    let hidden = shapes.hidden_size;
    let idx = layer.idx;
    let n_e = shapes.num_experts;
    anyhow::ensure!(n_e > 0, "layer {idx} is MoE but num_experts is 0");
    let inter = shapes.moe_intermediate_size;
    let sinter = shapes.shared_expert_intermediate_size;

    let router = HostBf16Lin {
        w: load_bf16_named(w, &[&names::router(idx)], &[n_e, hidden])?,
        n: n_e,
        k: hidden,
    };
    let bias_name = names::selection_bias(idx);
    let selection_bias = if w.has(&bias_name) {
        let t = w
            .get(&bias_name, candle_core::DType::F32)
            .with_context(|| format!("load {bias_name}"))?;
        let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
        anyhow::ensure!(
            v.len() == n_e,
            "{bias_name}: {} values, want {n_e}",
            v.len()
        );
        v
    } else {
        vec![0f32; n_e]
    };

    let gate_mods: Vec<String> = (0..n_e).map(|e| names::expert_gate_proj(idx, e)).collect();
    let up_mods: Vec<String> = (0..n_e).map(|e| names::expert_up_proj(idx, e)).collect();
    let down_mods: Vec<String> = (0..n_e).map(|e| names::expert_down_proj(idx, e)).collect();

    Ok(HostMoe {
        router,
        selection_bias,
        experts_gate: load_expert_stack(w, &gate_mods, inter, hidden)?,
        experts_up: load_expert_stack(w, &up_mods, inter, hidden)?,
        experts_down: load_expert_stack(w, &down_mods, hidden, inter)?,
        shared_gate: load_lin(w, &names::shared_gate_proj(idx), sinter, hidden)?,
        shared_up: load_lin(w, &names::shared_up_proj(idx), sinter, hidden)?,
        shared_down: load_lin(w, &names::shared_down_proj(idx), hidden, sinter)?,
    })
}

pub fn bf16_bits(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

pub fn bf16_val(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

pub use crate::gemma4_wgpu_shared::pack_pairs;
pub use crate::wgpu_ledger::bytes_to_words;

pub use crate::nvfp4_host::{
    dequantize_nvfp4_host, expert_slice, quantize_nvfp4_host, stack_nvfp4_host,
    HostNvfp4ExpertStack, HostNvfp4Lin,
};

pub fn stack_bf16_host(mats: &[HostBf16Lin]) -> HostBf16ExpertStack {
    let n = mats[0].n;
    let k = mats[0].k;
    let mut w = Vec::with_capacity(mats.len() * n * k);
    for m in mats {
        w.extend_from_slice(&m.w);
    }
    HostBf16ExpertStack {
        w,
        e: mats.len(),
        n,
        k,
    }
}

pub struct Lcg(pub u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }

    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    pub fn fill_bf16(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n).map(|_| bf16_bits(self.next_f32() * scale)).collect()
    }
}

fn rand_bf16_lin(rng: &mut Lcg, n: usize, k: usize) -> HostBf16Lin {
    HostBf16Lin {
        w: rng.fill_bf16(n * k, (k as f32).powf(-0.5)),
        n,
        k,
    }
}

fn rand_lin(rng: &mut Lcg, n: usize, k: usize) -> HostLin {
    HostLin::Bf16(rand_bf16_lin(rng, n, k))
}

fn rand_norm(rng: &mut Lcg, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| bf16_bits(1.0 + 0.25 * rng.next_f32()))
        .collect()
}

fn rand_expert_stack(rng: &mut Lcg, e: usize, n: usize, k: usize) -> HostExperts {
    let mats: Vec<HostBf16Lin> = (0..e).map(|_| rand_bf16_lin(rng, n, k)).collect();
    HostExperts::Bf16(stack_bf16_host(&mats))
}

pub fn random_host_weights(shapes: &LagunaShapes, seed: u64) -> HostWeights {
    let mut rng = Lcg::new(seed);
    let hidden = shapes.hidden_size;
    let embed = rng.fill_bf16(shapes.vocab_size * hidden, 0.5);

    let mut layers = Vec::with_capacity(shapes.num_layers);
    for li in 0..shapes.num_layers {
        let ls = *shapes.layer(li);
        let input_ln = rand_norm(&mut rng, hidden);
        let post_attn_ln = rand_norm(&mut rng, hidden);
        let attn = HostAttention {
            q: rand_lin(&mut rng, ls.q_rows, hidden),
            k: rand_lin(&mut rng, ls.kv_rows, hidden),
            v: rand_lin(&mut rng, ls.kv_rows, hidden),
            o: rand_lin(&mut rng, hidden, ls.q_rows),
            g: match shapes.gate_kind {
                GateKind::None => None,
                _ => Some(rand_lin(&mut rng, ls.gate_rows, hidden)),
            },
            q_norm: rand_norm(&mut rng, ls.head_dim),
            k_norm: rand_norm(&mut rng, ls.head_dim),
        };
        let ffn = if ls.is_moe() {
            let n_e = shapes.num_experts;
            let inter = shapes.moe_intermediate_size;
            let sinter = shapes.shared_expert_intermediate_size;
            HostFfn::Moe(Box::new(HostMoe {
                router: rand_bf16_lin(&mut rng, n_e, hidden),
                selection_bias: (0..n_e).map(|_| 0.1 * rng.next_f32()).collect(),
                experts_gate: rand_expert_stack(&mut rng, n_e, inter, hidden),
                experts_up: rand_expert_stack(&mut rng, n_e, inter, hidden),
                experts_down: rand_expert_stack(&mut rng, n_e, hidden, inter),
                shared_gate: rand_lin(&mut rng, sinter, hidden),
                shared_up: rand_lin(&mut rng, sinter, hidden),
                shared_down: rand_lin(&mut rng, hidden, sinter),
            }))
        } else {
            let inter = ls.ffn_intermediate;
            HostFfn::Dense(Box::new(HostDenseMlp {
                gate: rand_lin(&mut rng, inter, hidden),
                up: rand_lin(&mut rng, inter, hidden),
                down: rand_lin(&mut rng, hidden, inter),
            }))
        };
        layers.push(HostLayer {
            kind: ls.attn_kind,
            input_ln,
            post_attn_ln,
            attn,
            ffn,
        });
    }

    let final_norm = rand_norm(&mut rng, hidden);
    let lm_head = if shapes.tie_word_embeddings {
        embed.clone()
    } else {
        rng.fill_bf16(shapes.vocab_size * hidden, (hidden as f32).powf(-0.5))
    };

    HostWeights {
        embed,
        final_norm,
        lm_head,
        layers,
    }
}

pub fn gate_rows_for(gate_kind: GateKind, num_q_heads: usize, head_dim: usize) -> usize {
    gate_kind.rows_for(num_q_heads, head_dim)
}
