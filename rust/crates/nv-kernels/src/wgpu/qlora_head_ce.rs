use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{dispatch, Result, WgpuError};

pub const W4HEAD_WGSL_PORTS_THE_FIVE_MSL_KERNELS_RIDING_THE_CHECKPOINTS_OWN_W4_BYTES: &str =
    "the WGSL twin of nv_train::dots_w4head: a dequant-in-kernel logits GEMM over the mlx affine \
     u4 gs64 lm_head words (value = fma(nibble, scale, bias), nibble e of a u32 word at bits \
     4e, exactly nv_quant::mlx_affine), one row-stats kernel producing lse and the weighted CE \
     rows, an in-place rewrite of the stashed logits into dlogits = w*g*(softmax - onehot) so \
     the backward never allocates a second vocab-wide buffer, a quantized-read dX GEMM over \
     dlogits split ns ways along vocab, and a fixed-order ns-way reduce. Rows chunk at \
     HEAD_CE_CHUNK_ROWS so only one vocab-wide logits slice is ever live. Both GEMM chains are \
     pure sequential fma so a fixed-order host replica reproduces them bit-exactly; the CE \
     reductions use exp/log whose GPU rounding differs from the host libm, so loss and lse gate \
     on tolerance. Reduction orders are fixed, so reruns of the same binary are deterministic.";

pub const HEAD_CE_CHUNK_ROWS: usize = 256;

pub const HEAD_DX_SPLITS_MIRROR_THE_MSL_NSPLIT_CONSTANT_SO_THE_REDUCE_ORDER_IS_SHARED: usize = 8;

pub const GROUP_SIZE: usize = 64;

pub const WORKGROUP_SIZE: u32 = 256;

pub const WGSL: &str = include_str!("../../wgsl/qlora_head_ce.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    m: u32,
    n: u32,
    k: u32,
    ns: u32,
    span: u32,
    cap: f32,
    pad1: u32,
    pad2: u32,
}

pub const SOFTCAP_ZERO_IS_OFF_AND_A_POSITIVE_CAP_TANH_SQUASHES_EVERY_LOGIT_BEFORE_THE_CE: &str =
    "W4Head.softcap is the Gemma final_logit_softcapping: the raw logits stay in the stash, \
     w4h_ce_stats reduces over cap*tanh(l/cap), and w4h_dlogits_inplace multiplies the \
     softmax residual by 1-tanh^2 so dX flows through the cap; zero leaves the kernel on the \
     uncapped dots path bit for bit";

pub struct W4Head {
    wq: wgpu::Buffer,
    sc: wgpu::Buffer,
    bi: wgpu::Buffer,
    pub n: usize,
    pub k: usize,
    pub softcap: f32,
}

pub fn upload_head(
    ctx: &WgpuContext,
    words: &[u32],
    scales: &[f32],
    biases: &[f32],
    n: usize,
    k: usize,
) -> Result<W4Head> {
    let _ = W4HEAD_WGSL_PORTS_THE_FIVE_MSL_KERNELS_RIDING_THE_CHECKPOINTS_OWN_W4_BYTES;
    if !k.is_multiple_of(GROUP_SIZE) || !k.is_multiple_of(8) || k == 0 || n == 0 {
        return Err(WgpuError::Shape(format!(
            "w4 head needs k % {GROUP_SIZE} == 0, got n {n} k {k}"
        )));
    }
    dispatch::check_len("w4 head words", words.len(), n * k / 8)?;
    dispatch::check_len("w4 head scales", scales.len(), n * (k / GROUP_SIZE))?;
    dispatch::check_len("w4 head biases", biases.len(), n * (k / GROUP_SIZE))?;
    let wq_bytes = (words.len() * 4) as u64;
    if wq_bytes > ctx.caps.max_storage_buffer_binding_size {
        return Err(WgpuError::Unsupported(format!(
            "w4 head words need a {wq_bytes}-byte binding, device allows {}",
            ctx.caps.max_storage_buffer_binding_size
        )));
    }
    Ok(W4Head {
        wq: dispatch::storage_from_slice(ctx, "w4h-words", words),
        sc: dispatch::storage_from_slice(ctx, "w4h-scales", scales),
        bi: dispatch::storage_from_slice(ctx, "w4h-biases", biases),
        n,
        k,
        softcap: 0.0,
    })
}

fn params_for(head: &W4Head, m: usize, ns: usize) -> Result<Params> {
    if m == 0 || m > HEAD_CE_CHUNK_ROWS {
        return Err(WgpuError::Shape(format!(
            "w4 head chunk rows must be 1..={HEAD_CE_CHUNK_ROWS}, got {m}"
        )));
    }
    Ok(Params {
        m: m as u32,
        n: head.n as u32,
        k: head.k as u32,
        ns: ns as u32,
        span: head.n.div_ceil(ns) as u32,
        cap: if head.softcap.is_finite() && head.softcap > 0.0 {
            let _ = SOFTCAP_ZERO_IS_OFF_AND_A_POSITIVE_CAP_TANH_SQUASHES_EVERY_LOGIT_BEFORE_THE_CE;
            head.softcap
        } else {
            0.0
        },
        pad1: 0,
        pad2: 0,
    })
}

fn elementwise_groups(ctx: &WgpuContext, what: &str, total: usize) -> Result<(u32, u32, u32)> {
    let wgs = (total as u64).div_ceil(WORKGROUP_SIZE as u64).max(1);
    if wgs > ctx.caps.max_compute_workgroups_per_dimension as u64 {
        return Err(WgpuError::Shape(format!(
            "{what}: {total} elements need {wgs} workgroups, device dimension limit is {}",
            ctx.caps.max_compute_workgroups_per_dimension
        )));
    }
    Ok((wgs as u32, 1, 1))
}

fn check_logits_binding(ctx: &WgpuContext, m: usize, n: usize) -> Result<()> {
    let bytes = (m * n * 4) as u64;
    if bytes > ctx.caps.max_storage_buffer_binding_size {
        return Err(WgpuError::Unsupported(format!(
            "the {m}x{n} logits slice needs a {bytes}-byte binding, device allows {}",
            ctx.caps.max_storage_buffer_binding_size
        )));
    }
    Ok(())
}

pub struct HeadCeFwd {
    params: wgpu::Buffer,
    logits: wgpu::Buffer,
    lse: wgpu::Buffer,
    pub loss_rows: Vec<f32>,
    pub m: usize,
}

pub fn head_logits(ctx: &WgpuContext, head: &W4Head, x: &[f32], m: usize) -> Result<Vec<f32>> {
    dispatch::drive_blocking(head_logits_arm(
        ctx,
        head,
        x,
        m,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn head_logits_arm(
    ctx: &WgpuContext,
    head: &W4Head,
    x: &[f32],
    m: usize,
    arm: dispatch::ReadbackArm,
) -> Result<Vec<f32>> {
    dispatch::check_len("w4h x", x.len(), m * head.k)?;
    check_logits_binding(ctx, m, head.n)?;
    let params = params_for(
        head,
        m,
        HEAD_DX_SPLITS_MIRROR_THE_MSL_NSPLIT_CONSTANT_SO_THE_REDUCE_ORDER_IS_SHARED,
    )?;
    let pb = dispatch::uniform_from(ctx, "w4h-params", &params);
    let xbuf = dispatch::storage_from_slice(ctx, "w4h-x", x);
    let lg = dispatch::storage_zeroed(ctx, "w4h-logits", (m * head.n * 4) as u64);
    dispatch::run(
        ctx,
        "w4h-logits",
        WGSL,
        "w4h_logits",
        &[
            (0, &pb),
            (1, &xbuf),
            (2, &head.wq),
            (3, &head.sc),
            (4, &head.bi),
            (5, &lg),
        ],
        elementwise_groups(ctx, "w4h_logits", m * head.n)?,
    )?;
    dispatch::read_back_arm(ctx, &lg, m * head.n, arm).await
}

pub fn head_ce_fwd(
    ctx: &WgpuContext,
    head: &W4Head,
    x: &[f32],
    tgt: &[u32],
    wrow: &[f32],
    m: usize,
) -> Result<HeadCeFwd> {
    dispatch::drive_blocking(head_ce_fwd_arm(
        ctx,
        head,
        x,
        tgt,
        wrow,
        m,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn head_ce_fwd_arm(
    ctx: &WgpuContext,
    head: &W4Head,
    x: &[f32],
    tgt: &[u32],
    wrow: &[f32],
    m: usize,
    arm: dispatch::ReadbackArm,
) -> Result<HeadCeFwd> {
    dispatch::check_len("w4h x", x.len(), m * head.k)?;
    dispatch::check_len("w4h tgt", tgt.len(), m)?;
    dispatch::check_len("w4h wrow", wrow.len(), m)?;
    dispatch::require_workgroup_and_scratch(ctx, "w4h ce", WORKGROUP_SIZE, 256 * 4)?;
    check_logits_binding(ctx, m, head.n)?;
    for (i, &t) in tgt.iter().enumerate() {
        if t as usize >= head.n {
            return Err(WgpuError::Shape(format!(
                "target {t} at row {i} out of vocab {}",
                head.n
            )));
        }
    }
    let params = params_for(
        head,
        m,
        HEAD_DX_SPLITS_MIRROR_THE_MSL_NSPLIT_CONSTANT_SO_THE_REDUCE_ORDER_IS_SHARED,
    )?;
    let pb = dispatch::uniform_from(ctx, "w4h-params", &params);
    let xbuf = dispatch::storage_from_slice(ctx, "w4h-x", x);
    let lg = dispatch::storage_zeroed(ctx, "w4h-logits", (m * head.n * 4) as u64);
    let tb = dispatch::storage_from_slice(ctx, "w4h-tgt", tgt);
    let wb = dispatch::storage_from_slice(ctx, "w4h-wrow", wrow);
    let lseb = dispatch::storage_zeroed(ctx, "w4h-lse", (m * 4) as u64);
    let lossb = dispatch::storage_zeroed(ctx, "w4h-loss", (m * 4) as u64);
    dispatch::run(
        ctx,
        "w4h-logits",
        WGSL,
        "w4h_logits",
        &[
            (0, &pb),
            (1, &xbuf),
            (2, &head.wq),
            (3, &head.sc),
            (4, &head.bi),
            (5, &lg),
        ],
        elementwise_groups(ctx, "w4h_logits", m * head.n)?,
    )?;
    dispatch::run(
        ctx,
        "w4h-ce-stats",
        WGSL,
        "w4h_ce_stats",
        &[
            (0, &pb),
            (5, &lg),
            (6, &tb),
            (7, &wb),
            (8, &lseb),
            (9, &lossb),
        ],
        (m as u32, 1, 1),
    )?;
    let loss_rows: Vec<f32> = dispatch::read_back_arm(ctx, &lossb, m, arm).await?;
    Ok(HeadCeFwd {
        params: pb,
        logits: lg,
        lse: lseb,
        loss_rows,
        m,
    })
}

pub fn head_ce_bwd(
    ctx: &WgpuContext,
    head: &W4Head,
    fwd: &HeadCeFwd,
    tgt: &[u32],
    wrow: &[f32],
    gup: &[f32],
) -> Result<Vec<f32>> {
    dispatch::drive_blocking(head_ce_bwd_arm(
        ctx,
        head,
        fwd,
        tgt,
        wrow,
        gup,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn head_ce_bwd_arm(
    ctx: &WgpuContext,
    head: &W4Head,
    fwd: &HeadCeFwd,
    tgt: &[u32],
    wrow: &[f32],
    gup: &[f32],
    arm: dispatch::ReadbackArm,
) -> Result<Vec<f32>> {
    let m = fwd.m;
    dispatch::check_len("w4h tgt", tgt.len(), m)?;
    dispatch::check_len("w4h wrow", wrow.len(), m)?;
    dispatch::check_len("w4h gup", gup.len(), m)?;
    let ns = HEAD_DX_SPLITS_MIRROR_THE_MSL_NSPLIT_CONSTANT_SO_THE_REDUCE_ORDER_IS_SHARED;
    let tb = dispatch::storage_from_slice(ctx, "w4h-tgt", tgt);
    let wb = dispatch::storage_from_slice(ctx, "w4h-wrow", wrow);
    let gb = dispatch::storage_from_slice(ctx, "w4h-gup", gup);
    let dxpb = dispatch::storage_zeroed(ctx, "w4h-dxp", (ns * m * head.k * 4) as u64);
    let dxb = dispatch::storage_zeroed(ctx, "w4h-dx", (m * head.k * 4) as u64);
    let pb = &fwd.params;
    dispatch::run(
        ctx,
        "w4h-dlogits",
        WGSL,
        "w4h_dlogits_inplace",
        &[
            (0, pb),
            (5, &fwd.logits),
            (6, &tb),
            (7, &wb),
            (8, &fwd.lse),
            (10, &gb),
        ],
        elementwise_groups(ctx, "w4h_dlogits_inplace", m * head.n)?,
    )?;
    dispatch::run(
        ctx,
        "w4h-dx-partial",
        WGSL,
        "w4h_dx_partial",
        &[
            (0, pb),
            (2, &head.wq),
            (3, &head.sc),
            (4, &head.bi),
            (5, &fwd.logits),
            (11, &dxpb),
        ],
        elementwise_groups(ctx, "w4h_dx_partial", ns * m * head.k)?,
    )?;
    dispatch::run(
        ctx,
        "w4h-dx-reduce",
        WGSL,
        "w4h_dx_reduce",
        &[(0, pb), (11, &dxpb), (12, &dxb)],
        elementwise_groups(ctx, "w4h_dx_reduce", m * head.k)?,
    )?;
    dispatch::read_back_arm(ctx, &dxb, m * head.k, arm).await
}

pub fn read_stash_logits(ctx: &WgpuContext, head: &W4Head, fwd: &HeadCeFwd) -> Result<Vec<f32>> {
    dispatch::read_back(ctx, &fwd.logits, fwd.m * head.n)
}

pub struct HeadCeChunked {
    pub loss_rows: Vec<f32>,
    pub dx: Vec<f32>,
}

pub fn head_ce_chunked(
    ctx: &WgpuContext,
    head: &W4Head,
    x: &[f32],
    tgt: &[u32],
    wrow: &[f32],
    gup: &[f32],
) -> Result<HeadCeChunked> {
    let m = tgt.len();
    dispatch::check_len("w4h x", x.len(), m * head.k)?;
    dispatch::check_len("w4h wrow", wrow.len(), m)?;
    dispatch::check_len("w4h gup", gup.len(), m)?;
    let mut loss_rows = Vec::with_capacity(m);
    let mut dx = Vec::with_capacity(m * head.k);
    for c0 in (0..m).step_by(HEAD_CE_CHUNK_ROWS) {
        let c1 = (c0 + HEAD_CE_CHUNK_ROWS).min(m);
        let fwd = head_ce_fwd(
            ctx,
            head,
            &x[c0 * head.k..c1 * head.k],
            &tgt[c0..c1],
            &wrow[c0..c1],
            c1 - c0,
        )?;
        loss_rows.extend_from_slice(&fwd.loss_rows);
        dx.extend(head_ce_bwd(
            ctx,
            head,
            &fwd,
            &tgt[c0..c1],
            &wrow[c0..c1],
            &gup[c0..c1],
        )?);
    }
    Ok(HeadCeChunked { loss_rows, dx })
}
