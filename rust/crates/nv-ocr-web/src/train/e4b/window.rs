use std::collections::BTreeMap;

use anyhow::{anyhow, ensure, Context, Result};
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch::{drive_blocking, ReadbackArm};
use nv_weights::WeightLoader;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;

use super::schedule::{
    build_examples, kv_sources, layer_forward, rope_tables_for, rt_of, train_window,
    window_start_checked, AuxLayout, E4bTrainSample, KvPair, WindowRun,
};
use super::weights::{E4bDims, E4bLayer, E4bTower, E4bWindow};
use crate::train::schedule::{
    pack_buckets_aux, Bucket, Ex, StepFactors, TrainOptions, TrainReport,
};

pub const WINDOW_DUMP_IS_THE_E4B_TWIN_OF_THE_DOTS_VISION_DUMP: &str =
    "the frozen prefix (embed, per-layer rows, layers below the window) is position- and \
     step-invariant, so it runs natively once per sample at batch one and the dump keeps only \
     what the window consumes: the hidden state at window_start, the window layers' per-layer \
     input rows, and the rope'd k / normed v of every kv source a window layer reads; a tab \
     packs those into buckets exactly as the native schedule packs its own, so the window \
     trains the same trajectory from a few hundred kilobytes per sample";

pub const WINDOW_SAMPLE_KV_IS_GROUPED_AT_BATCH_ONE_AND_ZERO_PADDED_INTO_THE_BUCKET: &str =
    "a sample's shared k/v is stored in the attention kernel's grouped layout [kv_heads, t, \
     head_dim]; the bucket layout is [rows * kv_heads, tp, head_dim] with zeros past t, which \
     is what the native prefix computes for padded rows (zero hidden rows norm, project and \
     rope to zero) and which causal masking never reads";

#[derive(Clone, Debug)]
pub struct E4bWindowSample {
    pub ids: Vec<u32>,
    pub completion_start: usize,
    pub window_start: usize,
    pub h: Vec<f32>,
    pub aux: Vec<f32>,
    pub kv: BTreeMap<usize, KvPair>,
}

fn f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn u32_le(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl E4bWindowSample {
    pub fn t(&self) -> usize {
        self.ids.len() - 1
    }

    pub fn to_safetensors_bytes(&self, d: &E4bDims) -> Result<Vec<u8>> {
        let t = self.t();
        let grad_layers = d.layers - self.window_start;
        let mut blobs: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = vec![
            (
                "ids".into(),
                Dtype::U32,
                vec![self.ids.len()],
                self.ids.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
            (
                "completion_start".into(),
                Dtype::U32,
                vec![1],
                (self.completion_start as u32).to_le_bytes().to_vec(),
            ),
            (
                "window_start".into(),
                Dtype::U32,
                vec![1],
                (self.window_start as u32).to_le_bytes().to_vec(),
            ),
            (
                "h".into(),
                Dtype::F32,
                vec![t, d.hidden],
                self.h.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
            (
                "aux".into(),
                Dtype::F32,
                vec![t, grad_layers * d.hpl],
                self.aux.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
        ];
        for (src, (k, v)) in &self.kv {
            let kvh = d.kv_heads(*src);
            let hd = d.head_dim(*src);
            for (leaf, data) in [("k", k), ("v", v)] {
                blobs.push((
                    format!("kv.{src}.{leaf}"),
                    Dtype::F32,
                    vec![kvh, t, hd],
                    data.iter().flat_map(|x| x.to_le_bytes()).collect(),
                ));
            }
        }
        let views = blobs
            .iter()
            .map(|(name, dt, shape, bytes)| {
                TensorView::new(*dt, shape.clone(), bytes).map(|v| (name.as_str(), v))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(safetensors::serialize(views, None)?)
    }

    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes).context("parse window sample safetensors")?;
        let get = |name: &str| st.tensor(name).map_err(|e| anyhow!("{name}: {e}"));
        let scalar = |name: &str| -> Result<usize> {
            let v = u32_le(get(name)?.data());
            ensure!(v.len() == 1, "{name}: expected one value");
            Ok(v[0] as usize)
        };
        let ids = u32_le(get("ids")?.data());
        ensure!(ids.len() >= 2, "ids must hold a prompt and a completion");
        let t = ids.len() - 1;
        let h_t = get("h")?;
        ensure!(
            h_t.shape().first() == Some(&t),
            "h rows {:?} != shifted length {t}",
            h_t.shape()
        );
        let aux_t = get("aux")?;
        ensure!(
            aux_t.shape().first() == Some(&t),
            "aux rows {:?} != shifted length {t}",
            aux_t.shape()
        );
        let mut kv: BTreeMap<usize, KvPair> = BTreeMap::new();
        for (name, view) in st.tensors() {
            let Some(rest) = name.strip_prefix("kv.") else {
                continue;
            };
            let Some((src, leaf)) = rest.split_once('.') else {
                continue;
            };
            let src: usize = src
                .parse()
                .with_context(|| format!("{name}: source index"))?;
            ensure!(
                view.shape().len() == 3 && view.shape()[1] == t,
                "{name}: shape {:?} is not [kv_heads, {t}, head_dim]",
                view.shape()
            );
            let slot = kv.entry(src).or_default();
            match leaf {
                "k" => slot.0 = f32_le(view.data()),
                "v" => slot.1 = f32_le(view.data()),
                other => anyhow::bail!("{name}: unknown kv leaf {other}"),
            }
        }
        for (src, (k, v)) in &kv {
            ensure!(
                !k.is_empty() && k.len() == v.len(),
                "kv source {src}: k and v must both be present with equal length"
            );
        }
        Ok(Self {
            ids,
            completion_start: scalar("completion_start")?,
            window_start: scalar("window_start")?,
            h: f32_le(h_t.data()),
            aux: f32_le(aux_t.data()),
            kv,
        })
    }

    fn check(&self, d: &E4bDims, window_start: usize, i: usize) -> Result<()> {
        let t = self.t();
        ensure!(
            self.window_start == window_start,
            "sample {i}: dumped at window_start {} but grad_layers asks for {window_start}",
            self.window_start
        );
        ensure!(
            self.completion_start >= 1 && self.completion_start < self.ids.len(),
            "sample {i}: completion_start {} outside 1..{}",
            self.completion_start,
            self.ids.len()
        );
        ensure!(
            self.h.len() == t * d.hidden,
            "sample {i}: h length {} != {t} x {}",
            self.h.len(),
            d.hidden
        );
        let width = (d.layers - window_start) * d.hpl;
        ensure!(
            self.aux.len() == t * width,
            "sample {i}: aux length {} != {t} x {width}",
            self.aux.len()
        );
        for src in kv_sources(d, window_start..d.layers) {
            let (k, v) = self.kv.get(&src).ok_or_else(|| {
                anyhow!("sample {i}: no shared k/v dumped for source layer {src}")
            })?;
            let want = d.kv_heads(src) * t * d.head_dim(src);
            ensure!(
                k.len() == want && v.len() == want,
                "sample {i}: kv source {src} length {} != {want}",
                k.len()
            );
        }
        Ok(())
    }
}

pub fn dump_window_samples(
    ctx: &WgpuContext,
    loader: &WeightLoader,
    config_json: &str,
    samples: &[E4bTrainSample],
    opts: &TrainOptions,
) -> Result<Vec<E4bWindowSample>> {
    drive_blocking(dump_window_samples_arm(
        ctx,
        loader,
        config_json,
        samples,
        opts,
        ReadbackArm::Blocking,
    ))
}

pub async fn dump_window_samples_arm(
    ctx: &WgpuContext,
    loader: &WeightLoader,
    config_json: &str,
    samples: &[E4bTrainSample],
    opts: &TrainOptions,
    arm: ReadbackArm,
) -> Result<Vec<E4bWindowSample>> {
    let _ = WINDOW_DUMP_IS_THE_E4B_TWIN_OF_THE_DOTS_VISION_DUMP;
    let d = E4bDims::from_config_json(config_json)?;
    let window_start = window_start_checked(&d, opts)?;
    ensure!(!samples.is_empty(), "no samples to dump");
    let sources = kv_sources(&d, 0..d.layers);
    let window_sources = kv_sources(&d, window_start..d.layers);
    let tower = E4bTower::load(ctx, loader, &d).context("load E4B tower")?;
    let exs = build_examples(ctx, loader, &d, &tower, samples, arm).await?;
    drop(tower);
    let max_t = exs.iter().map(|e| e.t).max().unwrap_or(2).max(2);
    let rts = rope_tables_for(&d, max_t);
    let empty_facs = StepFactors {
        at: Vec::new(),
        bt: Vec::new(),
    };
    let lay = AuxLayout::full(&d);
    let mut hs: Vec<Vec<f32>> = exs.iter().map(|e| e.h.clone()).collect();
    let mut kv_stash: BTreeMap<usize, Vec<KvPair>> = BTreeMap::new();
    for li in 0..window_start {
        let layer = E4bLayer::load(ctx, loader, &d, li)
            .with_context(|| format!("load frozen prefix layer {li}"))?;
        let mut kvs = Vec::with_capacity(exs.len());
        for (i, ex) in exs.iter().enumerate() {
            let positions: Vec<u32> = (0..ex.t as u32).collect();
            let shared = layer.kv_source.map(|s| &kv_stash[&s][i]);
            let f = layer_forward(
                ctx,
                &layer,
                &d,
                rt_of(&rts, layer.kind),
                &[],
                None,
                &empty_facs,
                &hs[i],
                1,
                ex.t,
                &positions,
                &ex.aux,
                lay,
                shared,
                false,
                arm,
            )
            .await?;
            hs[i] = f.out;
            if sources.contains(&li) {
                kvs.push(f.kv.expect("a kv source layer computes its own k and v"));
            }
        }
        if sources.contains(&li) {
            kv_stash.insert(li, kvs);
        }
    }
    let width = (d.layers - window_start) * d.hpl;
    let mut out = Vec::with_capacity(exs.len());
    for (i, (ex, h)) in exs.iter().zip(hs).enumerate() {
        let mut aux = Vec::with_capacity(ex.t * width);
        for r in 0..ex.t {
            let base = r * d.ple_row + window_start * d.hpl;
            aux.extend_from_slice(&ex.aux[base..base + width]);
        }
        let kv = window_sources
            .iter()
            .map(|&s| (s, kv_stash[&s][i].clone()))
            .collect();
        out.push(E4bWindowSample {
            ids: samples[i].ids.clone(),
            completion_start: samples[i].completion_start,
            window_start,
            h,
            aux,
            kv,
        });
    }
    Ok(out)
}

fn bucket_kv(d: &E4bDims, src: usize, bkt: &Bucket, samples: &[E4bWindowSample]) -> KvPair {
    let _ = WINDOW_SAMPLE_KV_IS_GROUPED_AT_BATCH_ONE_AND_ZERO_PADDED_INTO_THE_BUCKET;
    let kvh = d.kv_heads(src);
    let hd = d.head_dim(src);
    let mut k = vec![0f32; bkt.rows * kvh * bkt.tp * hd];
    let mut v = k.clone();
    for (row, &i) in bkt.members.iter().enumerate() {
        let s = &samples[i];
        let t = s.t();
        let (sk, sv) = &s.kv[&src];
        for g in 0..kvh {
            let dst = ((row * kvh + g) * bkt.tp) * hd;
            let srcb = g * t * hd;
            k[dst..dst + t * hd].copy_from_slice(&sk[srcb..srcb + t * hd]);
            v[dst..dst + t * hd].copy_from_slice(&sv[srcb..srcb + t * hd]);
        }
    }
    (k, v)
}

pub fn run_e4b_window_qlora(
    ctx: &WgpuContext,
    loader: &WeightLoader,
    config_json: &str,
    samples: &[E4bWindowSample],
    opts: &TrainOptions,
) -> Result<TrainReport> {
    drive_blocking(run_e4b_window_qlora_arm(
        ctx,
        loader,
        config_json,
        samples,
        opts,
        ReadbackArm::Blocking,
        &mut |_, _| {},
    ))
}

pub async fn run_e4b_window_qlora_pumped(
    ctx: &WgpuContext,
    loader: &WeightLoader,
    config_json: &str,
    samples: &[E4bWindowSample],
    opts: &TrainOptions,
    on_step: &mut dyn FnMut(usize, f32),
) -> Result<TrainReport> {
    run_e4b_window_qlora_arm(
        ctx,
        loader,
        config_json,
        samples,
        opts,
        ReadbackArm::Pumped,
        on_step,
    )
    .await
}

async fn run_e4b_window_qlora_arm(
    ctx: &WgpuContext,
    loader: &WeightLoader,
    config_json: &str,
    samples: &[E4bWindowSample],
    opts: &TrainOptions,
    arm: ReadbackArm,
    on_step: &mut dyn FnMut(usize, f32),
) -> Result<TrainReport> {
    let d = E4bDims::from_config_json(config_json)?;
    let window_start = window_start_checked(&d, opts)?;
    ensure!(!samples.is_empty(), "no window samples staged");
    for (i, s) in samples.iter().enumerate() {
        s.check(&d, window_start, i)?;
    }
    let lay = AuxLayout::window(&d, window_start);
    let exs: Vec<Ex> = samples
        .iter()
        .map(|s| Ex {
            h: s.h.clone(),
            t: s.t(),
            tgt: s.ids[1..].to_vec(),
            mask: (1..s.ids.len())
                .map(|p| if p >= s.completion_start { 1.0 } else { 0.0 })
                .collect(),
            aux: s.aux.clone(),
        })
        .collect();
    let buckets = pack_buckets_aux(&exs, d.hidden, lay.width)?;
    drop(exs);
    let hs: Vec<Vec<f32>> = buckets.iter().map(|b| b.h0.clone()).collect();
    let kv_stash: BTreeMap<usize, Vec<KvPair>> = kv_sources(&d, window_start..d.layers)
        .into_iter()
        .map(|src| {
            (
                src,
                buckets
                    .iter()
                    .map(|b| bucket_kv(&d, src, b, samples))
                    .collect(),
            )
        })
        .collect();
    let window = E4bWindow::load(ctx, loader, &d, window_start).context("load E4B window")?;
    train_window(
        ctx,
        WindowRun {
            d: &d,
            window: &window.layers,
            final_norm: &window.final_norm,
            head: &window.head,
            buckets: &buckets,
            hs: &hs,
            kv_stash: &kv_stash,
            lay,
            window_start,
            n_samples: samples.len(),
        },
        opts,
        arm,
        on_step,
    )
    .await
}
