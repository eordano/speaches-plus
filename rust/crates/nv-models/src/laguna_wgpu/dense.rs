use anyhow::Result;

use super::config::{LagunaShapes, LayerShape};
use super::gpu::{
    alloc_lin_scratch, alloc_lin_scratch_rows, alloc_pair_scratch, alloc_pair_scratch_rows,
    push_lin_gemv, push_lin_gemv_pair, push_lin_gemv_pair_rows, push_lin_gemv_rows, push_silu_mul,
    upload_lin, Builder, LinGpu, MRows, Sources, W8Scope,
};
use super::lora::LoraSiteCtx;
use super::weights::HostDenseMlp;
use super::{rbf, ref_gemv_lin, silu};

pub const DENSE_WGSL: &str = "";

pub struct DenseGpu {
    pub gate: LinGpu,
    pub up: LinGpu,
    pub down: LinGpu,
}

fn check_shapes(shapes: &LagunaShapes, layer: &LayerShape, w: &HostDenseMlp) -> Result<usize> {
    let hidden = shapes.hidden_size;
    let inter = layer.ffn_intermediate;
    anyhow::ensure!(
        inter > 0 && inter.is_multiple_of(2),
        "layer {} dense intermediate {inter} must be positive and even",
        layer.idx
    );
    anyhow::ensure!(
        w.gate.n() == inter && w.gate.k() == hidden,
        "layer {} gate_proj [{}, {}] != [{inter}, {hidden}]",
        layer.idx,
        w.gate.n(),
        w.gate.k()
    );
    anyhow::ensure!(
        w.up.n() == inter && w.up.k() == hidden,
        "layer {} up_proj [{}, {}] != [{inter}, {hidden}]",
        layer.idx,
        w.up.n(),
        w.up.k()
    );
    anyhow::ensure!(
        w.down.n() == hidden && w.down.k() == inter,
        "layer {} down_proj [{}, {}] != [{hidden}, {inter}]",
        layer.idx,
        w.down.n(),
        w.down.k()
    );
    Ok(inter)
}

#[allow(clippy::too_many_arguments)]
pub fn build_dense_mlp(
    b: &mut Builder,
    s: &Sources,
    shapes: &LagunaShapes,
    layer: &LayerShape,
    w: &HostDenseMlp,
    x_normed_packed: &wgpu::Buffer,
    out_packed: &wgpu::Buffer,
    lora: Option<&LoraSiteCtx>,
) -> Result<DenseGpu> {
    let inter = check_shapes(shapes, layer, w)?;
    let label = format!("lgw-dense{}", layer.idx);

    let gate = upload_lin(b, &format!("{label}-gate"), &w.gate, W8Scope::Ffn);
    let up = upload_lin(b, &format!("{label}-up"), &w.up, W8Scope::Ffn);
    let down = upload_lin(b, &format!("{label}-down"), &w.down, W8Scope::Ffn);

    let gu_scratch = alloc_pair_scratch(
        b,
        &format!("{label}-gate"),
        &gate,
        &format!("{label}-up"),
        &up,
    );
    let down_sc = alloc_lin_scratch(b, &format!("{label}-down"), &down);

    let y_gate = b.zeros(&format!("{label}-yg"), (inter * 2) as u64);
    let y_up = b.zeros(&format!("{label}-yu"), (inter * 2) as u64);
    let act = b.zeros(&format!("{label}-act"), (inter * 2) as u64);

    push_lin_gemv_pair(
        b,
        s,
        &format!("{label}-gate"),
        &gate,
        &format!("{label}-up"),
        &up,
        &gu_scratch,
        x_normed_packed,
        &y_gate,
        &y_up,
    )?;
    if let Some(lc) = lora {
        if let Some(site) = &lc.layer.gate_up {
            lc.push_site(b, site, x_normed_packed, &[&y_gate, &y_up]);
        }
    }
    push_silu_mul(b, s, &format!("{label}-silu"), &y_gate, &y_up, &act, inter)?;
    push_lin_gemv(
        b,
        s,
        &format!("{label}-down"),
        &down,
        &down_sc,
        &act,
        out_packed,
    )?;
    if let Some(lc) = lora {
        if let Some(site) = &lc.layer.down {
            lc.push_site(b, site, &act, &[out_packed]);
        }
    }
    Ok(DenseGpu { gate, up, down })
}

pub fn build_dense_mlp_rows(
    b: &mut Builder,
    s: &Sources,
    layer: &LayerShape,
    g: &DenseGpu,
    x_rows_packed: &wgpu::Buffer,
    out_rows_packed: &wgpu::Buffer,
    mp: &MRows,
) -> Result<()> {
    let inter = layer.ffn_intermediate;
    let m = mp.m;
    let label = format!("lgw-mdense{}", layer.idx);

    let gu_scratch = alloc_pair_scratch_rows(
        b,
        &format!("{label}-gate"),
        &g.gate,
        &format!("{label}-up"),
        &g.up,
        m,
    );
    let down_sc = alloc_lin_scratch_rows(b, &format!("{label}-down"), &g.down, m);

    let y_gate = b.zeros(&format!("{label}-yg"), (m * inter * 2) as u64);
    let y_up = b.zeros(&format!("{label}-yu"), (m * inter * 2) as u64);
    let act = b.zeros(&format!("{label}-act"), (m * inter * 2) as u64);

    push_lin_gemv_pair_rows(
        b,
        s,
        &format!("{label}-gate"),
        &g.gate,
        &format!("{label}-up"),
        &g.up,
        &gu_scratch,
        x_rows_packed,
        &y_gate,
        &y_up,
        mp,
    )?;
    push_silu_mul(
        b,
        s,
        &format!("{label}-silu"),
        &y_gate,
        &y_up,
        &act,
        m * inter,
    )?;
    push_lin_gemv_rows(
        b,
        s,
        &format!("{label}-down"),
        &g.down,
        &down_sc,
        &act,
        out_rows_packed,
        mp,
    )
}

pub fn ref_dense_mlp(
    shapes: &LagunaShapes,
    layer: &LayerShape,
    w: &HostDenseMlp,
    x_normed: &[f32],
) -> Result<Vec<f32>> {
    let inter = check_shapes(shapes, layer, w)?;
    anyhow::ensure!(
        x_normed.len() == shapes.hidden_size,
        "dense mlp input {} != hidden {}",
        x_normed.len(),
        shapes.hidden_size
    );
    let g = ref_gemv_lin(&w.gate, x_normed);
    let u = ref_gemv_lin(&w.up, x_normed);
    let act: Vec<f32> = (0..inter)
        .map(|i| rbf(silu(rbf(g[i])) * rbf(u[i])))
        .collect();
    let y = ref_gemv_lin(&w.down, &act);
    Ok(y.into_iter().map(rbf).collect())
}
