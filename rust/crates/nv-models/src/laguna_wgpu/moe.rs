use anyhow::Result;

use super::config::{LagunaShapes, LayerShape, NVFP4_BLOCK};
use super::gpu::{
    alloc_lin_scratch, push_gemv_bf16, push_gemv_bf16_experts, push_gemv_i8_experts,
    push_gemv_nvfp4, push_lin_gemv, push_quant_rows, push_silu_mul, upload_bf16, upload_experts,
    upload_lin, Builder, ExpertsGpu, Sources, W8Scope,
};
use super::weights::{HostExperts, HostMoe};
use super::{rbf, ref_gemv_bf16, ref_gemv_lin, sigmoid, silu};

pub const MOE_WGSL: &str = include_str!("../../../nv-kernels/wgsl/laguna_moe.wgsl");

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RouterParams {
    pub n_experts: u32,
    pub top_k: u32,
    pub norm_topk: u32,
    pub softcap: f32,
    pub routed_scaling: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CombineParams {
    pub hidden_words: u32,
    pub top_k: u32,
    pub slot_stride_words: u32,
    pub routed_scaling: f32,
}

pub const ROUTER_MAX_EXPERTS: usize = 256;

fn selection_bias(shapes: &LagunaShapes, w: &HostMoe) -> Result<Vec<f32>> {
    if w.selection_bias.is_empty() {
        return Ok(vec![0f32; shapes.num_experts]);
    }
    anyhow::ensure!(
        w.selection_bias.len() == shapes.num_experts,
        "selection_bias has {} entries for {} experts",
        w.selection_bias.len(),
        shapes.num_experts
    );
    Ok(w.selection_bias.clone())
}

fn check_moe(shapes: &LagunaShapes, layer: &LayerShape, w: &HostMoe) -> Result<()> {
    let hidden = shapes.hidden_size;
    let inter = layer.ffn_intermediate;
    anyhow::ensure!(
        shapes.num_experts > 0 && shapes.top_k > 0,
        "moe layer {} needs num_experts>0 and num_experts_per_tok>0",
        layer.idx
    );
    anyhow::ensure!(
        shapes.top_k <= shapes.num_experts,
        "num_experts_per_tok {} exceeds num_experts {}",
        shapes.top_k,
        shapes.num_experts
    );
    anyhow::ensure!(
        shapes.num_experts <= ROUTER_MAX_EXPERTS,
        "router top-k kernel caps num_experts at {ROUTER_MAX_EXPERTS}, got {}",
        shapes.num_experts
    );
    anyhow::ensure!(
        w.router.n == shapes.num_experts && w.router.k == hidden,
        "router is [{}, {}], want [{}, {hidden}]",
        w.router.n,
        w.router.k,
        shapes.num_experts
    );
    for (name, e, n, k) in [
        ("gate", &w.experts_gate, inter, hidden),
        ("up", &w.experts_up, inter, hidden),
        ("down", &w.experts_down, hidden, inter),
    ] {
        anyhow::ensure!(
            e.num_experts() == shapes.num_experts && e.n() == n && e.k() == k,
            "expert {name} stack is [{}, {}, {}], want [{}, {n}, {k}]",
            e.num_experts(),
            e.n(),
            e.k(),
            shapes.num_experts
        );
    }
    let sinter = shapes.shared_expert_intermediate_size;
    anyhow::ensure!(
        w.shared_gate.n() == sinter
            && w.shared_gate.k() == hidden
            && w.shared_up.n() == sinter
            && w.shared_up.k() == hidden
            && w.shared_down.n() == hidden
            && w.shared_down.k() == sinter,
        "shared expert shapes disagree with shared_expert_intermediate_size {sinter}"
    );
    let gate_q = matches!(&w.experts_gate, HostExperts::Nvfp4(_));
    anyhow::ensure!(
        gate_q == matches!(&w.experts_up, HostExperts::Nvfp4(_)),
        "expert gate/up dtypes differ; the wgpu path quantizes the token once per slot"
    );
    if gate_q {
        anyhow::ensure!(
            w.experts_gate.input_globals() == w.experts_up.input_globals(),
            "expert gate/up input_global_scale differ; the wgpu path quantizes the token once per slot"
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prepare_expert_input(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    stack: &ExpertsGpu,
    x_packed: &wgpu::Buffer,
    ids: &wgpu::Buffer,
    k_elems: usize,
    slots: usize,
    x_per_slot: bool,
) -> Result<Option<(wgpu::Buffer, wgpu::Buffer)>> {
    match stack {
        ExpertsGpu::Bf16(_) | ExpertsGpu::Int8(_) => Ok(None),
        ExpertsGpu::Nvfp4(st) => {
            let k_blocks = k_elems / NVFP4_BLOCK;
            let xq = b.zeros(&format!("{label}-xq"), (slots * k_elems / 2) as u64);
            let xs = b.zeros(&format!("{label}-xs"), (slots * k_blocks) as u64);
            push_quant_rows(
                b,
                s,
                &format!("{label}-quant"),
                x_packed,
                &xq,
                &xs,
                ids,
                &st.globals,
                k_elems,
                slots,
                true,
                x_per_slot,
            )?;
            Ok(Some((xq, xs)))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_expert_gemv(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    stack: &ExpertsGpu,
    x_packed: &wgpu::Buffer,
    quant: Option<(&wgpu::Buffer, &wgpu::Buffer)>,
    y: &wgpu::Buffer,
    ids: &wgpu::Buffer,
    slots: usize,
    x_per_slot: bool,
) -> Result<()> {
    match stack {
        ExpertsGpu::Bf16(st) => {
            push_gemv_bf16_experts(b, s, label, st, x_packed, y, ids, slots, x_per_slot)
        }
        ExpertsGpu::Int8(st) => {
            push_gemv_i8_experts(b, s, label, st, x_packed, y, ids, slots, true, x_per_slot)
        }
        ExpertsGpu::Nvfp4(st) => {
            let (xq, xs) = quant.ok_or_else(|| {
                anyhow::anyhow!("{label}: nvfp4 expert stack needs a quantized x")
            })?;
            push_gemv_nvfp4(
                b, s, label, &st.w, &st.scales, xq, xs, y, ids, &st.alphas, st.n, st.k, slots,
                true, 1.0,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_moe_layer(
    b: &mut Builder,
    s: &Sources,
    shapes: &LagunaShapes,
    layer: &LayerShape,
    w: &HostMoe,
    x_normed_packed: &wgpu::Buffer,
    out_packed: &wgpu::Buffer,
) -> Result<()> {
    check_moe(shapes, layer, w)?;

    let hidden = shapes.hidden_size;
    let hidden_words = shapes.hidden_words();
    let inter = layer.ffn_intermediate;
    let sinter = shapes.shared_expert_intermediate_size;
    let n_e = shapes.num_experts;
    let k_top = shapes.top_k;
    let li = layer.idx;

    let router = upload_bf16(b, &format!("lgw-moe{li}-router"), &w.router);
    let rlogits = b.zeros(&format!("lgw-moe{li}-rlogits"), (n_e * 4) as u64);
    push_gemv_bf16(
        b,
        s,
        &format!("lgw-moe{li}-router"),
        &router,
        x_normed_packed,
        &rlogits,
        true,
        0,
    )?;

    let bias = b.upload_f32(&format!("lgw-moe{li}-bias"), &selection_bias(shapes, w)?);
    let ids = b.zeros(&format!("lgw-moe{li}-ids"), (k_top * 4) as u64);
    let wts = b.zeros(&format!("lgw-moe{li}-wts"), (k_top * 4) as u64);
    let rp = b.uni(
        &format!("lgw-moe{li}-router-p"),
        RouterParams {
            n_experts: n_e as u32,
            top_k: k_top as u32,
            norm_topk: u32::from(shapes.norm_topk_prob),
            softcap: shapes.router_softcap,
            routed_scaling: shapes.routed_scaling,
            ..Default::default()
        },
    );
    b.push(
        &format!("lgw-moe{li}-topk"),
        &s.moe,
        "lgw_moe_router_topk",
        &[(0, &rlogits), (1, &bias), (2, &ids), (3, &wts), (4, &rp)],
        (1, 1, 1),
    )?;

    let eg = upload_experts(b, &format!("lgw-moe{li}-eg"), &w.experts_gate);
    let eu = upload_experts(b, &format!("lgw-moe{li}-eu"), &w.experts_up);
    let ed = upload_experts(b, &format!("lgw-moe{li}-ed"), &w.experts_down);

    let xq = prepare_expert_input(
        b,
        s,
        &format!("lgw-moe{li}-x"),
        &eg,
        x_normed_packed,
        &ids,
        hidden,
        k_top,
        false,
    )?;
    let xq_ref = xq.as_ref().map(|(q, sc)| (q, sc));

    let y_gate = b.zeros(&format!("lgw-moe{li}-ygate"), (k_top * inter * 2) as u64);
    let y_up = b.zeros(&format!("lgw-moe{li}-yup"), (k_top * inter * 2) as u64);
    push_expert_gemv(
        b,
        s,
        &format!("lgw-moe{li}-gate"),
        &eg,
        x_normed_packed,
        xq_ref,
        &y_gate,
        &ids,
        k_top,
        false,
    )?;
    push_expert_gemv(
        b,
        s,
        &format!("lgw-moe{li}-up"),
        &eu,
        x_normed_packed,
        xq_ref,
        &y_up,
        &ids,
        k_top,
        false,
    )?;

    let act = b.zeros(&format!("lgw-moe{li}-act"), (k_top * inter * 2) as u64);
    push_silu_mul(
        b,
        s,
        &format!("lgw-moe{li}-silu"),
        &y_gate,
        &y_up,
        &act,
        k_top * inter,
    )?;

    let aq = prepare_expert_input(
        b,
        s,
        &format!("lgw-moe{li}-a"),
        &ed,
        &act,
        &ids,
        inter,
        k_top,
        true,
    )?;
    let y_down = b.zeros(&format!("lgw-moe{li}-ydown"), (k_top * hidden * 2) as u64);
    push_expert_gemv(
        b,
        s,
        &format!("lgw-moe{li}-down"),
        &ed,
        &act,
        aq.as_ref().map(|(q, sc)| (q, sc)),
        &y_down,
        &ids,
        k_top,
        true,
    )?;

    let sg = upload_lin(b, &format!("lgw-moe{li}-sg"), &w.shared_gate, W8Scope::Ffn);
    let su = upload_lin(b, &format!("lgw-moe{li}-su"), &w.shared_up, W8Scope::Ffn);
    let sd = upload_lin(b, &format!("lgw-moe{li}-sd"), &w.shared_down, W8Scope::Ffn);
    let sg_scratch = alloc_lin_scratch(b, &format!("lgw-moe{li}-sg"), &sg);
    let su_scratch = alloc_lin_scratch(b, &format!("lgw-moe{li}-su"), &su);
    let sd_scratch = alloc_lin_scratch(b, &format!("lgw-moe{li}-sd"), &sd);

    let sy_gate = b.zeros(&format!("lgw-moe{li}-syg"), (sinter * 2) as u64);
    let sy_up = b.zeros(&format!("lgw-moe{li}-syu"), (sinter * 2) as u64);
    push_lin_gemv(
        b,
        s,
        &format!("lgw-moe{li}-sgate"),
        &sg,
        &sg_scratch,
        x_normed_packed,
        &sy_gate,
    )?;
    push_lin_gemv(
        b,
        s,
        &format!("lgw-moe{li}-sup"),
        &su,
        &su_scratch,
        x_normed_packed,
        &sy_up,
    )?;
    let sact = b.zeros(&format!("lgw-moe{li}-sact"), (sinter * 2) as u64);
    push_silu_mul(
        b,
        s,
        &format!("lgw-moe{li}-ssilu"),
        &sy_gate,
        &sy_up,
        &sact,
        sinter,
    )?;
    let shared_out = b.zeros(&format!("lgw-moe{li}-sout"), (hidden * 2) as u64);
    push_lin_gemv(
        b,
        s,
        &format!("lgw-moe{li}-sdown"),
        &sd,
        &sd_scratch,
        &sact,
        &shared_out,
    )?;

    let cp = b.uni(
        &format!("lgw-moe{li}-comb-p"),
        CombineParams {
            hidden_words: hidden_words as u32,
            top_k: k_top as u32,
            slot_stride_words: hidden_words as u32,
            routed_scaling: shapes.routed_scaling,
        },
    );
    let grid = b.grid1(hidden_words as u64, 64);
    b.push(
        &format!("lgw-moe{li}-combine"),
        &s.moe,
        "lgw_moe_combine",
        &[
            (20, &y_down),
            (21, &wts),
            (22, &shared_out),
            (23, out_packed),
            (24, &cp),
        ],
        grid,
    )
}

pub fn ref_router_topk(
    shapes: &LagunaShapes,
    w: &HostMoe,
    x_normed: &[f32],
) -> Result<(Vec<u32>, Vec<f32>)> {
    anyhow::ensure!(
        shapes.top_k > 0 && shapes.top_k <= shapes.num_experts,
        "num_experts_per_tok {} invalid for {} experts",
        shapes.top_k,
        shapes.num_experts
    );
    anyhow::ensure!(
        w.router.n == shapes.num_experts && w.router.k == x_normed.len(),
        "router is [{}, {}], want [{}, {}]",
        w.router.n,
        w.router.k,
        shapes.num_experts,
        x_normed.len()
    );
    let bias = selection_bias(shapes, w)?;
    let softcap = shapes.router_softcap;
    let logits = ref_gemv_bf16(&w.router, x_normed);

    let scores: Vec<f32> = logits
        .iter()
        .map(|l| {
            let capped = if softcap > 0.0 {
                softcap * (*l / softcap).tanh()
            } else {
                *l
            };
            sigmoid(capped)
        })
        .collect();
    let selection: Vec<f32> = (0..shapes.num_experts)
        .map(|i| scores[i] + bias[i])
        .collect();

    let mut order: Vec<usize> = (0..shapes.num_experts).collect();
    order.sort_by(|a, b| {
        selection[*b]
            .partial_cmp(&selection[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    let ids: Vec<u32> = order[..shapes.top_k].iter().map(|e| *e as u32).collect();
    let mut wts: Vec<f32> = ids.iter().map(|e| scores[*e as usize]).collect();
    if shapes.norm_topk_prob {
        let mut sum = 0f32;
        for v in &wts {
            sum += *v;
        }
        for v in wts.iter_mut() {
            *v /= sum;
        }
    }
    Ok((ids, wts))
}

pub fn ref_shared_expert(shapes: &LagunaShapes, w: &HostMoe, x_normed: &[f32]) -> Result<Vec<f32>> {
    let sinter = shapes.shared_expert_intermediate_size;
    anyhow::ensure!(
        w.shared_gate.n() == sinter && w.shared_up.n() == sinter && w.shared_down.k() == sinter,
        "shared expert width disagrees with shared_expert_intermediate_size {sinter}"
    );
    let g = ref_gemv_lin(&w.shared_gate, x_normed);
    let u = ref_gemv_lin(&w.shared_up, x_normed);
    let act: Vec<f32> = (0..sinter)
        .map(|i| rbf(silu(rbf(g[i])) * rbf(u[i])))
        .collect();
    Ok(ref_gemv_lin(&w.shared_down, &act)
        .into_iter()
        .map(rbf)
        .collect())
}

pub fn ref_moe(shapes: &LagunaShapes, w: &HostMoe, x_normed: &[f32]) -> Result<Vec<f32>> {
    let hidden = shapes.hidden_size;
    let inter = shapes.moe_intermediate_size;
    anyhow::ensure!(
        x_normed.len() == hidden,
        "moe input has {} elements, want {hidden}",
        x_normed.len()
    );
    let (ids, wts) = ref_router_topk(shapes, w, x_normed)?;

    let mut acc = vec![0f32; hidden];
    for (j, e) in ids.iter().enumerate() {
        let e = *e as usize;
        let gate = w.experts_gate.expert(e);
        let up = w.experts_up.expert(e);
        let down = w.experts_down.expert(e);
        let yg = ref_gemv_lin(&gate, x_normed);
        let yu = ref_gemv_lin(&up, x_normed);
        let act: Vec<f32> = (0..inter)
            .map(|i| rbf(silu(rbf(yg[i])) * rbf(yu[i])))
            .collect();
        let yd = ref_gemv_lin(&down, &act);
        for i in 0..hidden {
            acc[i] = rbf(yd[i]).mul_add(wts[j], acc[i]);
        }
    }

    let shared = ref_shared_expert(shapes, w, x_normed)?;
    Ok((0..hidden)
        .map(|i| rbf(acc[i] * shapes.routed_scaling + shared[i]))
        .collect())
}
