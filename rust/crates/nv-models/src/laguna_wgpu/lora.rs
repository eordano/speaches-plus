use anyhow::{Context, Result};

use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::WgpuContext;

use crate::wgpu_lora::{
    build_lora_site, emit_lora_site_pk, upload_lora_site, LoraPipelines, LoraSite, LoraSiteGpu,
};

use super::config::{FfnKind, LagunaConfig, LagunaShapes};
use super::gpu::{Builder, Pass};

pub const LAGUNA_LORA_PREFIX_MATCHES_NORMALIZE_MODULE_NAME_NOT_THE_CHECKPOINT: &str =
    "model.language_model.layers";

pub const LAGUNA_LORA_TARGETS: [&str; 8] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.g_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

#[derive(Clone, Debug, Default)]
pub(crate) struct LagunaLoraLayer {
    pub(crate) qkv: Option<LoraSite>,
    pub(crate) g: Option<LoraSite>,
    pub(crate) o: Option<LoraSite>,
    pub(crate) gate_up: Option<LoraSite>,
    pub(crate) down: Option<LoraSite>,
}

impl LagunaLoraLayer {
    fn sites(&self) -> impl Iterator<Item = &LoraSite> {
        [&self.qkv, &self.g, &self.o, &self.gate_up, &self.down]
            .into_iter()
            .flatten()
    }
}

pub struct LagunaLora {
    rank: usize,
    matched: usize,
    skipped: Vec<String>,
    pub(crate) layers: Vec<LagunaLoraLayer>,
}

impl LagunaLora {
    pub fn from_peft_dir(dir: impl AsRef<std::path::Path>, config: &LagunaConfig) -> Result<Self> {
        let adapter =
            nv_weights::lora_adapter::LoraAdapter::load(dir.as_ref(), &candle_core::Device::Cpu)
                .with_context(|| format!("load lora adapter from {}", dir.as_ref().display()))?;
        Self::from_adapter(&adapter, config)
    }

    pub fn from_adapter(
        adapter: &nv_weights::lora_adapter::LoraAdapter,
        config: &LagunaConfig,
    ) -> Result<Self> {
        let rank = adapter.config.r;
        anyhow::ensure!(
            rank > 0 && rank <= wk::lora::FUSED_MAX_RANK,
            "lora rank {rank} outside 1..={}",
            wk::lora::FUSED_MAX_RANK
        );
        let shapes = LagunaShapes::derive(config, 1)?;
        let hidden = shapes.hidden_size;
        let prefix = LAGUNA_LORA_PREFIX_MATCHES_NORMALIZE_MODULE_NAME_NOT_THE_CHECKPOINT;
        let mut matched = 0usize;
        let mut wanted: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut layers = Vec::with_capacity(shapes.num_layers);
        for li in 0..shapes.num_layers {
            let ls = *shapes.layer(li);
            let p = format!("{prefix}.{li}");
            let mut layer = LagunaLoraLayer::default();

            let q = format!("{p}.self_attn.q_proj");
            let kp = format!("{p}.self_attn.k_proj");
            let vp = format!("{p}.self_attn.v_proj");
            let qkv_slices: Vec<(&str, usize)> = vec![
                (q.as_str(), ls.q_rows),
                (kp.as_str(), ls.kv_rows),
                (vp.as_str(), ls.kv_rows),
            ];
            let qkv_segs: Vec<usize> = qkv_slices.iter().map(|(_, w)| *w).collect();
            let (qkv, n) = build_lora_site(adapter, rank, hidden, &qkv_slices, &qkv_segs)?;
            matched += n;
            layer.qkv = qkv;
            for (name, _) in &qkv_slices {
                wanted.insert((*name).to_string());
            }

            let gp = format!("{p}.self_attn.g_proj");
            if ls.gate_rows > 0 && ls.gate_rows.is_multiple_of(2) {
                let (g, n) = build_lora_site(
                    adapter,
                    rank,
                    hidden,
                    &[(gp.as_str(), ls.gate_rows)],
                    &[ls.gate_rows],
                )?;
                matched += n;
                layer.g = g;
                wanted.insert(gp);
            }

            let op = format!("{p}.self_attn.o_proj");
            let (o, n) = build_lora_site(
                adapter,
                rank,
                ls.q_rows,
                &[(op.as_str(), hidden)],
                &[hidden],
            )?;
            matched += n;
            layer.o = o;
            wanted.insert(op);

            if ls.ffn_kind == FfnKind::Dense {
                let inter = ls.ffn_intermediate;
                let gp = format!("{p}.mlp.gate_proj");
                let up = format!("{p}.mlp.up_proj");
                let (gate_up, n) = build_lora_site(
                    adapter,
                    rank,
                    hidden,
                    &[(gp.as_str(), inter), (up.as_str(), inter)],
                    &[inter, inter],
                )?;
                matched += n;
                layer.gate_up = gate_up;
                let dp = format!("{p}.mlp.down_proj");
                let (down, n) =
                    build_lora_site(adapter, rank, inter, &[(dp.as_str(), hidden)], &[hidden])?;
                matched += n;
                layer.down = down;
                wanted.insert(gp);
                wanted.insert(up);
                wanted.insert(dp);
            }
            layers.push(layer);
        }
        let layer_ns = format!("{prefix}.");
        let skipped: Vec<String> = adapter
            .loras
            .keys()
            .filter(|n| !wanted.contains(*n))
            .cloned()
            .collect();
        if !skipped.is_empty() {
            let moe_mlp = skipped
                .iter()
                .filter(|n| n.starts_with(&layer_ns) && n.contains(".mlp."))
                .count();
            eprintln!(
                "[laguna_wgpu] lora: {} adapter modules have no site in this graph \
                 ({moe_mlp} are mlp projections routed through moe expert stacks; the rest \
                 are g_proj without an even-row gate or names outside {layer_ns}*); first is {}",
                skipped.len(),
                skipped[0]
            );
        }
        anyhow::ensure!(
            matched > 0,
            "lora adapter matched no laguna projection; it carries {} modules, the first of \
             which is {:?}. Expected names like {prefix}.0.self_attn.q_proj (the checkpoint's \
             model.layers.* rewritten by nv_weights::lora_adapter::normalize_module_name)",
            adapter.loras.len(),
            adapter.loras.keys().next()
        );
        Ok(Self {
            rank,
            matched,
            skipped,
            layers,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn matched_modules(&self) -> usize {
        self.matched
    }

    pub fn skipped_modules(&self) -> &[String] {
        &self.skipped
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_pass_count(&self, layer: usize) -> usize {
        2 * self.layers[layer].sites().count()
    }

    pub fn total_pass_count(&self) -> usize {
        (0..self.layers.len())
            .map(|i| self.layer_pass_count(i))
            .sum()
    }
}

#[derive(Default)]
pub struct LagunaLayerLoraGpu {
    pub(crate) qkv: Option<LoraSiteGpu>,
    pub(crate) g: Option<LoraSiteGpu>,
    pub(crate) o: Option<LoraSiteGpu>,
    pub(crate) gate_up: Option<LoraSiteGpu>,
    pub(crate) down: Option<LoraSiteGpu>,
}

pub(crate) fn upload_laguna_lora_layer(
    ctx: &WgpuContext,
    l: &LagunaLoraLayer,
) -> Result<LagunaLayerLoraGpu> {
    let up = |site: &Option<LoraSite>, label: &str| {
        site.as_ref()
            .map(|s| upload_lora_site(ctx, label, s))
            .transpose()
    };
    Ok(LagunaLayerLoraGpu {
        qkv: up(&l.qkv, "lgw-lora-qkv")?,
        g: up(&l.g, "lgw-lora-g")?,
        o: up(&l.o, "lgw-lora-o")?,
        gate_up: up(&l.gate_up, "lgw-lora-gate-up")?,
        down: up(&l.down, "lgw-lora-down")?,
    })
}

pub struct LoraSiteCtx<'a> {
    pub(crate) pl: &'a LoraPipelines,
    pub(crate) keep: &'a std::cell::RefCell<Vec<Box<dyn std::any::Any + Send>>>,
    pub(crate) layer: LagunaLayerLoraGpu,
}

impl LoraSiteCtx<'_> {
    pub(crate) fn push_site(
        &self,
        b: &mut Builder,
        site: &LoraSiteGpu,
        x: &wgpu::Buffer,
        segs: &[&wgpu::Buffer],
    ) {
        let ctx = b.ctx;
        let mut emitted: Vec<crate::wgpu_lora::Pass> = Vec::new();
        let mut keep = self.keep.borrow_mut();
        emit_lora_site_pk(ctx, &mut emitted, &mut keep, self.pl, 1, site, x, segs);
        for p in emitted {
            b.passes.push(Pass {
                pipeline: p.pipeline,
                bind: p.bind,
                grid: p.grid,
                entry: "lgw_lora_pk".to_string(),
            });
        }
    }
}
