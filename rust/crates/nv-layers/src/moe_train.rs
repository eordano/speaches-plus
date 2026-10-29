use anyhow::{Context, Result};
use candle_core::{CustomOp1, CustomOp2, DType, Layout, Shape, Storage, Tensor};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use crate::mlp::Mlp;
use crate::moe_grouped::{marlin_moe_stack_from_linear, MarlinMoeStack};

pub const GROUPED_EXPERT_TRAINING_RUNS_FOUR_MARLIN_MOE_LAUNCHES_PER_LAYER_PER_DIRECTION_INSTEAD_OF_THREE_PER_TOUCHED_EXPERT: &str =
    "the per-expert path launched three QuantLinearOps plus their gathers and cats for every touched \
     expert (22k launches per 512-token row per direction); the grouped op runs gate_up, silu*mul and \
     down as Marlin w4a16 MoE launches over the stacked experts and its backward runs the two \
     transposed stacks the same way, so the expert path is launch-count-flat in the token count";

pub struct PackedNvfp4Rows<'a> {
    pub packed: &'a [u8],
    pub scales_linear: &'a [u8],
    pub global: f32,
}

pub struct MarlinMoeTrainStacks {
    gate_up: MarlinMoeStack,
    down: MarlinMoeStack,
    gate_transposed: MarlinMoeStack,
    up_transposed: MarlinMoeStack,
    down_transposed: MarlinMoeStack,
    pub experts: usize,
    pub hidden: usize,
    pub inter: usize,
    c_tmp_floats: i64,
    workspace_ints: i32,
}

fn unswizzled_linear_scales(
    stream: &Arc<CudaStream>,
    swizzled: &CudaSlice<u8>,
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>> {
    let h = stream
        .clone_dtoh(swizzled)
        .map_err(|e| anyhow::anyhow!("dtoh scales: {e:?}"))?;
    Ok(nv_quant::nvfp4::unswizzle_scales(&h, rows, cols / 16))
}

pub const TRANSPOSED_STACKS_ARE_BUILT_ON_THE_DEVICE_BY_THE_DX_TWIN_REQUANT_KERNEL_PER_EXPERT_BECAUSE_THE_HOST_REQUANT_COST_10_S_PER_LAYER: usize = 1;

fn transposed_stack_on_the_device(
    stream: &Arc<CudaStream>,
    parts: &[(&CudaSlice<u8>, &CudaSlice<u8>, f32)],
    n_out: usize,
    k_in: usize,
) -> Result<MarlinMoeStack> {
    let experts = parts.len();
    let (n_t, k_t) = (k_in, n_out);
    let words = n_t * k_t / 8;
    let scale_bytes = n_t * k_t / 16;
    let mut b_q: CudaSlice<u32> = stream
        .alloc_zeros::<u32>(experts * words)
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut scales: CudaSlice<u8> = stream
        .alloc_zeros::<u8>(experts * scale_bytes)
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut globals: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(experts)
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut kmajor: CudaSlice<u32> =
        unsafe { stream.alloc::<u32>(words) }.map_err(|e| anyhow::anyhow!(e))?;
    let mut scales_linear: CudaSlice<u8> =
        unsafe { stream.alloc::<u8>(scale_bytes) }.map_err(|e| anyhow::anyhow!(e))?;
    let mut vmin: CudaSlice<u32> = stream
        .alloc_zeros::<u32>(1)
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut flags: CudaSlice<i32> = stream
        .alloc_zeros::<i32>(1)
        .map_err(|e| anyhow::anyhow!(e))?;
    for (e, (w_packed, w_scales, alpha)) in parts.iter().enumerate() {
        let rc = unsafe {
            let (wp, _g0) = w_packed.device_ptr(stream);
            let (sp, _g1) = w_scales.device_ptr(stream);
            let (kp, _g2) = kmajor.device_ptr_mut(stream);
            let (lp, _g3) = scales_linear.device_ptr_mut(stream);
            nv_kernels::cuda::dx_twin_nvfp4_requant_transposed_kmajor(
                stream.cu_stream() as *mut c_void,
                wp as *const u8,
                sp as *const u8,
                *alpha,
                n_out as i32,
                k_in as i32,
                kp as *mut u32,
                lp as *mut u8,
            )
        };
        anyhow::ensure!(
            rc == 0,
            "moe train transposed stack: requant expert {e} rc={rc}"
        );
        let rc = unsafe {
            let (pi, _g1) = kmajor.device_ptr(stream);
            let (po, _g2) = b_q.device_ptr_mut(stream);
            nv_kernels::cuda::marlin_repack_w4a16(
                stream.cu_stream() as *mut c_void,
                pi as *const c_void,
                (po + (e * words * 4) as u64) as *mut c_void,
                n_out as i32,
                k_in as i32,
                4,
            )
        };
        anyhow::ensure!(
            rc == 0,
            "moe train transposed stack: repack expert {e} rc={rc}"
        );
        let rc = unsafe {
            let (lp, _g0) = scales_linear.device_ptr(stream);
            let (ep, _g1) = scales.device_ptr_mut(stream);
            let (gp, _g2) = globals.device_ptr_mut(stream);
            let (vp, _g3) = vmin.device_ptr_mut(stream);
            let (fp, _g4) = flags.device_ptr_mut(stream);
            nv_kernels::cuda::dx_twin_nvfp4_scales_to_marlin(
                stream.cu_stream() as *mut c_void,
                lp as *const u8,
                n_t as i32,
                (k_t / 16) as i32,
                1.0,
                (ep + (e * scale_bytes) as u64) as *mut u8,
                (gp + (e * 4) as u64) as *mut f32,
                vp as *mut u32,
                fp as *mut i32,
            )
        };
        anyhow::ensure!(
            rc == 0,
            "moe train transposed stack: scales expert {e} rc={rc}"
        );
    }
    Ok(MarlinMoeStack {
        b_q,
        scales,
        globals,
    })
}

impl MarlinMoeTrainStacks {
    pub fn build_from_experts(
        experts: &[Mlp],
        hidden: usize,
        inter: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        let e_total = experts.len();
        anyhow::ensure!(
            e_total > 0 && hidden % 128 == 0 && inter % 64 == 0 && inter % 16 == 0,
            "moe train stacks: shape hidden {hidden} inter {inter} does not tile marlin"
        );
        let mut gu_packed = Vec::with_capacity(e_total * 2 * inter * hidden / 2);
        let mut gu_scales: Vec<Vec<u8>> = Vec::with_capacity(e_total);
        let mut gu_globals = Vec::with_capacity(e_total);
        let mut dn_packed = Vec::with_capacity(e_total * hidden * inter / 2);
        let mut dn_scales: Vec<Vec<u8>> = Vec::with_capacity(e_total);
        let mut dn_globals = Vec::with_capacity(e_total);
        let mut gate_parts: Vec<(&CudaSlice<u8>, &CudaSlice<u8>, f32)> =
            Vec::with_capacity(e_total);
        let mut up_parts: Vec<(&CudaSlice<u8>, &CudaSlice<u8>, f32)> = Vec::with_capacity(e_total);
        let mut down_parts: Vec<(&CudaSlice<u8>, &CudaSlice<u8>, f32)> =
            Vec::with_capacity(e_total);
        for (e, mlp) in experts.iter().enumerate() {
            let (gw, gs, ga, gi) = mlp
                .gate_proj()
                .nvfp4_parts()
                .with_context(|| format!("expert {e} gate_proj is not NVFP4"))?;
            let (uw, us, ua, ui) = mlp
                .up_proj()
                .nvfp4_parts()
                .with_context(|| format!("expert {e} up_proj is not NVFP4"))?;
            let (dw, ds, da, di) = mlp
                .down_proj()
                .nvfp4_parts()
                .with_context(|| format!("expert {e} down_proj is not NVFP4"))?;
            anyhow::ensure!(
                (ga * gi - ua * ui).abs() <= 1e-6 * ga.abs().max(1e-30),
                "expert {e}: gate and up globals differ ({} vs {}), the concat stack needs one",
                ga * gi,
                ua * ui
            );
            let gp = stream
                .clone_dtoh(gw)
                .map_err(|e| anyhow::anyhow!("dtoh gate: {e:?}"))?;
            let up = stream
                .clone_dtoh(uw)
                .map_err(|e| anyhow::anyhow!("dtoh up: {e:?}"))?;
            let dp = stream
                .clone_dtoh(dw)
                .map_err(|e| anyhow::anyhow!("dtoh down: {e:?}"))?;
            let gsl = unswizzled_linear_scales(stream, gs, inter, hidden)?;
            let usl = unswizzled_linear_scales(stream, us, inter, hidden)?;
            let dsl = unswizzled_linear_scales(stream, ds, hidden, inter)?;
            gu_packed.extend_from_slice(&gp);
            gu_packed.extend_from_slice(&up);
            let mut s = gsl.clone();
            s.extend_from_slice(&usl);
            gu_scales.push(s);
            gu_globals.push(ga * gi);
            dn_packed.extend_from_slice(&dp);
            dn_scales.push(dsl.clone());
            dn_globals.push(da * di);
            gate_parts.push((gw, gs, ga * gi));
            up_parts.push((uw, us, ua * ui));
            down_parts.push((dw, ds, da * di));
        }
        let gate_up = marlin_moe_stack_from_linear(
            stream,
            &gu_packed,
            &gu_scales,
            &gu_globals,
            e_total,
            2 * inter,
            hidden,
        )?;
        let down = marlin_moe_stack_from_linear(
            stream,
            &dn_packed,
            &dn_scales,
            &dn_globals,
            e_total,
            hidden,
            inter,
        )?;
        let _ = TRANSPOSED_STACKS_ARE_BUILT_ON_THE_DEVICE_BY_THE_DX_TWIN_REQUANT_KERNEL_PER_EXPERT_BECAUSE_THE_HOST_REQUANT_COST_10_S_PER_LAYER;
        let gate_transposed = transposed_stack_on_the_device(stream, &gate_parts, inter, hidden)?;
        let up_transposed = transposed_stack_on_the_device(stream, &up_parts, inter, hidden)?;
        let down_transposed = transposed_stack_on_the_device(stream, &down_parts, hidden, inter)?;
        Self::assemble(
            [gate_up, down, gate_transposed, up_transposed, down_transposed],
            e_total,
            hidden,
            inter,
        )
    }

    pub fn build_from_checkpoint_rows(
        experts: &[[PackedNvfp4Rows<'_>; 3]],
        hidden: usize,
        inter: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        let e_total = experts.len();
        anyhow::ensure!(
            e_total > 0 && hidden % 128 == 0 && inter % 64 == 0,
            "moe train stacks: shape hidden {hidden} inter {inter} does not tile marlin"
        );
        let shapes = [(inter, hidden), (inter, hidden), (hidden, inter)];
        for (e, projs) in experts.iter().enumerate() {
            for (p, (rows, cols)) in projs.iter().zip(shapes) {
                anyhow::ensure!(
                    p.packed.len() == rows * cols / 2 && p.scales_linear.len() == rows * cols / 16,
                    "expert {e}: packed {} / scale {} bytes do not match [{rows}, {cols}]",
                    p.packed.len(),
                    p.scales_linear.len()
                );
            }
            anyhow::ensure!(
                projs[0].global == projs[1].global,
                "expert {e}: gate and up weight_scale_2 differ ({} vs {}), the concat gate_up stack needs one",
                projs[0].global,
                projs[1].global
            );
        }
        let mut gu_packed = Vec::with_capacity(e_total * inter * hidden);
        let mut gu_scales = Vec::with_capacity(e_total);
        let mut dn_packed = Vec::with_capacity(e_total * hidden * inter / 2);
        let mut dn_scales = Vec::with_capacity(e_total);
        for [g, u, d] in experts {
            gu_packed.extend_from_slice(g.packed);
            gu_packed.extend_from_slice(u.packed);
            gu_scales.push([g.scales_linear, u.scales_linear].concat());
            dn_packed.extend_from_slice(d.packed);
            dn_scales.push(d.scales_linear.to_vec());
        }
        let globals = |i: usize| experts.iter().map(|p| p[i].global).collect::<Vec<f32>>();
        let gate_up = marlin_moe_stack_from_linear(stream, &gu_packed, &gu_scales, &globals(0), e_total, 2 * inter, hidden)?;
        let down = marlin_moe_stack_from_linear(stream, &dn_packed, &dn_scales, &globals(2), e_total, hidden, inter)?;
        drop((gu_packed, gu_scales, dn_packed, dn_scales));
        let transposed = |i: usize, rows: usize, cols: usize| -> Result<MarlinMoeStack> {
            let mut dev = Vec::with_capacity(e_total);
            for p in experts {
                let p = &p[i];
                let w = stream.clone_htod(p.packed).map_err(|e| anyhow::anyhow!(e))?;
                let s = stream
                    .clone_htod(&nv_quant::nvfp4::swizzle_scales(p.scales_linear, rows, cols / 16))
                    .map_err(|e| anyhow::anyhow!(e))?;
                dev.push((w, s, p.global));
            }
            let parts: Vec<(&CudaSlice<u8>, &CudaSlice<u8>, f32)> = dev.iter().map(|(w, s, g)| (w, s, *g)).collect();
            transposed_stack_on_the_device(stream, &parts, rows, cols)
        };
        let gate_transposed = transposed(0, inter, hidden)?;
        let up_transposed = transposed(1, inter, hidden)?;
        let down_transposed = transposed(2, hidden, inter)?;
        Self::assemble(
            [gate_up, down, gate_transposed, up_transposed, down_transposed],
            e_total,
            hidden,
            inter,
        )
    }

    pub fn device_bytes(&self) -> usize {
        let s = |m: &MarlinMoeStack| m.b_q.len() * 4 + m.scales.len() + m.globals.len() * 4;
        s(&self.gate_up) + s(&self.down) + s(&self.gate_transposed) + s(&self.up_transposed) + s(&self.down_transposed)
    }

    fn assemble(stacks: [MarlinMoeStack; 5], experts: usize, hidden: usize, inter: usize) -> Result<Self> {
        let [gate_up, down, gate_transposed, up_transposed, down_transposed] = stacks;
        let e_total = experts;
        let mut workspace_ints = 0i32;
        let rc =
            unsafe { nv_kernels::cuda::marlin_moe_workspace_ints(&mut workspace_ints as *mut i32) };
        anyhow::ensure!(
            rc == 0 && workspace_ints > 0,
            "marlin moe workspace ints rc={rc}"
        );
        let mut c_tmp_floats = 0i64;
        let rc = unsafe {
            nv_kernels::cuda::marlin_moe_nvfp4_c_tmp_floats(&mut c_tmp_floats as *mut i64)
        };
        anyhow::ensure!(
            rc == 0 && c_tmp_floats > 0,
            "marlin moe c_tmp floats rc={rc}"
        );
        Ok(Self {
            gate_up,
            down,
            gate_transposed,
            up_transposed,
            down_transposed,
            experts: e_total,
            hidden,
            inter,
            c_tmp_floats,
            workspace_ints,
        })
    }
}

pub struct HostMarlinMoeTrainStacks {
    parts: Vec<[crate::pinned_host::PinnedHostBuf; 3]>,
    experts: usize,
    hidden: usize,
    inter: usize,
}

impl MarlinMoeTrainStacks {
    fn stacks(&self) -> [&MarlinMoeStack; 5] {
        [&self.gate_up, &self.down, &self.gate_transposed, &self.up_transposed, &self.down_transposed]
    }

    pub fn park_on_host(&self, stream: &Arc<CudaStream>) -> Result<HostMarlinMoeTrainStacks> {
        use crate::pinned_host::PinnedHostBuf;
        let parts = self
            .stacks()
            .into_iter()
            .map(|m| {
                Ok([
                    PinnedHostBuf::download(stream, &m.b_q, "marlin train stack weights")?,
                    PinnedHostBuf::download(stream, &m.scales, "marlin train stack scales")?,
                    PinnedHostBuf::download(stream, &m.globals, "marlin train stack globals")?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(HostMarlinMoeTrainStacks { parts, experts: self.experts, hidden: self.hidden, inter: self.inter })
    }
}

impl HostMarlinMoeTrainStacks {
    pub fn bytes(&self) -> usize {
        self.parts.iter().flat_map(|p| p.iter()).map(|b| b.len()).sum()
    }

    pub fn upload(&self, compute: &Arc<CudaStream>, copy: &Arc<CudaStream>) -> Result<(MarlinMoeTrainStacks, cudarc::driver::CudaEvent)> {
        let mut stacks = Vec::with_capacity(5);
        let mut last = None;
        for [w, s, g] in &self.parts {
            let (b_q, _) = w.upload_on::<u32>(compute, copy)?;
            let (scales, _) = s.upload_on::<u8>(compute, copy)?;
            let (globals, ev) = g.upload_on::<f32>(compute, copy)?;
            last = Some(ev);
            stacks.push(MarlinMoeStack { b_q, scales, globals });
        }
        let stacks: [MarlinMoeStack; 5] = stacks.try_into().map_err(|_| anyhow::anyhow!("host marlin stacks hold other than five stacks"))?;
        let built = MarlinMoeTrainStacks::assemble(stacks, self.experts, self.hidden, self.inter)?;
        Ok((built, last.context("no stacks uploaded")?))
    }
}

pub const BLOCK8_ALIGNMENT_ON_THE_DEVICE_WHEN_THE_IDS_ARE_THERE_SO_THE_ROUTER_NEVER_DRAINS_THE_STREAM: usize = 8;

struct Align {
    sorted: CudaSlice<i32>,
    expert_ids: CudaSlice<i32>,
    padded: CudaSlice<i32>,
    unit_weights: CudaSlice<f32>,
}

struct Scratch {
    c_tmp: CudaSlice<f32>,
    workspace: CudaSlice<i32>,
}

struct Saved {
    gate_up_rows: CudaSlice<bf16>,
    align: Align,
    down_rows_for_the_slot_weight_grad: Option<CudaSlice<bf16>>,
}

pub struct MoeExpertsMarlinTrainOp {
    stacks: Arc<MarlinMoeTrainStacks>,
    ids_i32: Vec<i32>,
    ids_dev_u32_tokens_by_top_k: Option<Tensor>,
    tokens: usize,
    top_k: usize,
    saved: Mutex<Option<Saved>>,
}

impl MoeExpertsMarlinTrainOp {
    pub fn new(
        stacks: Arc<MarlinMoeTrainStacks>,
        top_ids: &[u32],
        tokens: usize,
        top_k: usize,
    ) -> Self {
        Self {
            stacks,
            ids_i32: top_ids.iter().map(|&i| i as i32).collect(),
            ids_dev_u32_tokens_by_top_k: None,
            tokens,
            top_k,
            saved: Mutex::new(None),
        }
    }

    pub fn new_with_device_ids(
        stacks: Arc<MarlinMoeTrainStacks>,
        ids_u32: Tensor,
        tokens: usize,
        top_k: usize,
    ) -> Self {
        Self {
            stacks,
            ids_i32: Vec::new(),
            ids_dev_u32_tokens_by_top_k: Some(ids_u32),
            tokens,
            top_k,
            saved: Mutex::new(None),
        }
    }

    fn rows(&self) -> usize {
        self.tokens * self.top_k
    }

    fn scratch(&self, stream: &Arc<CudaStream>) -> Result<Scratch> {
        Ok(Scratch {
            c_tmp: stream
                .alloc_zeros::<f32>(self.stacks.c_tmp_floats as usize)
                .map_err(|e| anyhow::anyhow!(e))?,
            workspace: stream
                .alloc_zeros::<i32>(self.stacks.workspace_ints as usize)
                .map_err(|e| anyhow::anyhow!(e))?,
        })
    }

    fn align(&self, stream: &Arc<CudaStream>) -> Result<Align> {
        let rows = self.rows();
        let experts = self.stacks.experts;
        if let Some(ids) = &self.ids_dev_u32_tokens_by_top_k {
            let _ = BLOCK8_ALIGNMENT_ON_THE_DEVICE_WHEN_THE_IDS_ARE_THERE_SO_THE_ROUTER_NEVER_DRAINS_THE_STREAM;
            anyhow::ensure!(
                ids.dtype() == DType::U32 && ids.elem_count() == rows,
                "moe train op: device ids must be {rows} u32 values, got {:?} {:?}",
                ids.dtype(),
                ids.dims()
            );
            let blocks_max = rows.div_ceil(8) + experts;
            let mut sorted: CudaSlice<i32> = stream
                .alloc_zeros::<i32>(blocks_max * 8)
                .map_err(|e| anyhow::anyhow!(e))?;
            let mut expert_ids: CudaSlice<i32> = stream
                .alloc_zeros::<i32>(blocks_max)
                .map_err(|e| anyhow::anyhow!(e))?;
            let mut padded: CudaSlice<i32> = stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| anyhow::anyhow!(e))?;
            let rc = {
                let (st, l) = ids.storage_and_layout();
                let candle_core::Storage::Cuda(c) = &*st else {
                    anyhow::bail!("moe train op: device ids not cuda");
                };
                let (ip, _gi) = c.as_cuda_slice::<u32>()?.device_ptr(stream);
                let ip = ip + (l.start_offset() * 4) as u64;
                let (sp, _gs) = sorted.device_ptr_mut(stream);
                let (ep, _ge) = expert_ids.device_ptr_mut(stream);
                let (pp, _gp) = padded.device_ptr_mut(stream);
                unsafe {
                    nv_kernels::cuda::marlin_moe_align_block8_small(
                        stream.cu_stream() as *mut c_void,
                        ip as *const i32,
                        rows as i32,
                        experts as i32,
                        sp as *mut i32,
                        ep as *mut i32,
                        pp as *mut i32,
                    )
                }
            };
            anyhow::ensure!(rc == 0, "moe train op: device block8 alignment rc={rc}");
            let unit_weights: CudaSlice<f32> = stream
                .clone_htod(&vec![1f32; rows])
                .map_err(|e| anyhow::anyhow!(e))?;
            return Ok(Align {
                sorted,
                expert_ids,
                padded,
                unit_weights,
            });
        }
        let mut count = vec![0usize; experts];
        for &e in &self.ids_i32 {
            anyhow::ensure!(
                (e as usize) < experts,
                "moe train op: expert id {e} out of {experts}"
            );
            count[e as usize] += 1;
        }
        let mut block_start = vec![0usize; experts];
        let mut blocks = 0usize;
        for e in 0..experts {
            block_start[e] = blocks;
            blocks += count[e].div_ceil(8);
        }
        let mut sorted = vec![rows as i32; blocks * 8];
        let mut expert_ids = vec![0i32; blocks];
        let mut cursor = vec![0usize; experts];
        for e in 0..experts {
            for b in 0..count[e].div_ceil(8) {
                expert_ids[block_start[e] + b] = e as i32;
            }
        }
        for (i, &e) in self.ids_i32.iter().enumerate() {
            let e = e as usize;
            sorted[block_start[e] * 8 + cursor[e]] = i as i32;
            cursor[e] += 1;
        }
        let sorted: CudaSlice<i32> = stream.clone_htod(&sorted).map_err(|e| anyhow::anyhow!(e))?;
        let expert_ids: CudaSlice<i32> = stream
            .clone_htod(&expert_ids)
            .map_err(|e| anyhow::anyhow!(e))?;
        let padded: CudaSlice<i32> = stream
            .clone_htod(&[(blocks * 8) as i32])
            .map_err(|e| anyhow::anyhow!(e))?;
        let unit_weights: CudaSlice<f32> = stream
            .clone_htod(&vec![1f32; rows])
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(Align {
            sorted,
            expert_ids,
            padded,
            unit_weights,
        })
    }

    fn gemm(
        &self,
        stream: &Arc<CudaStream>,
        stack: &MarlinMoeStack,
        align: &Align,
        scratch: &mut Scratch,
        a_ptr: u64,
        c: &mut CudaSlice<bf16>,
        size_m: usize,
        top_k: usize,
        n: usize,
        k: usize,
        what: &str,
    ) -> Result<()> {
        let _span = crate::prof_events::span(
            match what {
                "gate_up" | "down" => "fwd_moe_grouped_marlin_gemm",
                _ => "bwd_moe_grouped_marlin_gemm_transposed",
            },
            stream,
        );
        let (pb, _h1) = stack.b_q.device_ptr(stream);
        let (ps, _h2) = stack.scales.device_ptr(stream);
        let (pg, _h3) = stack.globals.device_ptr(stream);
        let (cp, _h4) = c.device_ptr_mut(stream);
        let (ct, _h5) = scratch.c_tmp.device_ptr_mut(stream);
        let (ws, _h6) = scratch.workspace.device_ptr_mut(stream);
        let (so, _h7) = align.sorted.device_ptr(stream);
        let (ex, _h8) = align.expert_ids.device_ptr(stream);
        let (pd, _h9) = align.padded.device_ptr(stream);
        let (uw, _h10) = align.unit_weights.device_ptr(stream);
        let rc = unsafe {
            nv_kernels::cuda::marlin_moe_nvfp4_w4a16(
                stream.cu_stream() as *mut c_void,
                a_ptr as *const c_void,
                pb as *const c_void,
                ps as *const c_void,
                pg as *const f32,
                cp as *mut c_void,
                ct as *mut c_void,
                self.stacks.c_tmp_floats,
                ws as *mut c_void,
                so as *const i32,
                ex as *const i32,
                pd as *const i32,
                uw as *const f32,
                self.stacks.experts as i32,
                top_k as i32,
                0,
                size_m as i32,
                n as i32,
                k as i32,
            )
        };
        anyhow::ensure!(
            rc == 0,
            "marlin moe train {what} rc={rc} m={size_m} n={n} k={k}"
        );
        Ok(())
    }

    fn act_rows(&self, stream: &Arc<CudaStream>, x_ptr: u64) -> Result<(CudaSlice<bf16>, CudaSlice<bf16>, Align, Scratch)> {
        let (rows, hidden, inter) = (self.rows(), self.stacks.hidden, self.stacks.inter);
        let align = self.align(stream)?;
        let mut scratch = self.scratch(stream)?;
        let mut gate_up_rows: CudaSlice<bf16> =
            unsafe { stream.alloc::<bf16>(rows * 2 * inter) }.map_err(|e| anyhow::anyhow!(e))?;
        let mut act_rows: CudaSlice<bf16> =
            unsafe { stream.alloc::<bf16>(rows * inter) }.map_err(|e| anyhow::anyhow!(e))?;
        self.gemm(
            stream,
            &self.stacks.gate_up,
            &align,
            &mut scratch,
            x_ptr,
            &mut gate_up_rows,
            self.tokens,
            self.top_k,
            2 * inter,
            hidden,
            "gate_up",
        )?;
        let rc = unsafe {
            let _span = crate::prof_events::span("fwd_moe_silu_mul_kernel", stream);
            let (gp, _g0) = gate_up_rows.device_ptr(stream);
            let (ap, _g1) = act_rows.device_ptr_mut(stream);
            nv_kernels::cuda::marlin_moe_silu_mul_halves(
                stream.cu_stream() as *mut c_void,
                gp as *const u16,
                ap as *mut u16,
                rows as i32,
                inter as i32,
            )
        };
        anyhow::ensure!(rc == 0, "marlin moe train silu_mul rc={rc}");
        Ok((act_rows, gate_up_rows, align, scratch))
    }

    fn forward_rows(&self, stream: &Arc<CudaStream>, x_ptr: u64) -> Result<CudaSlice<bf16>> {
        let (rows, hidden, inter) = (self.rows(), self.stacks.hidden, self.stacks.inter);
        let (act_rows, gate_up_rows, align, mut scratch) = self.act_rows(stream, x_ptr)?;
        let mut down_rows: CudaSlice<bf16> =
            unsafe { stream.alloc::<bf16>(rows * hidden) }.map_err(|e| anyhow::anyhow!(e))?;
        let (ap, _ga) = act_rows.device_ptr(stream);
        self.gemm(
            stream,
            &self.stacks.down,
            &align,
            &mut scratch,
            ap,
            &mut down_rows,
            rows,
            1,
            hidden,
            inter,
            "down",
        )?;
        *self.saved.lock().unwrap() = Some(Saved {
            gate_up_rows,
            align,
            down_rows_for_the_slot_weight_grad: None,
        });
        Ok(down_rows)
    }

    fn backward_rows(
        &self,
        stream: &Arc<CudaStream>,
        dev: &candle_core::CudaDevice,
        d_down_rows: &Tensor,
    ) -> Result<Tensor> {
        let (rows, hidden, inter) = (self.rows(), self.stacks.hidden, self.stacks.inter);
        let saved = self
            .saved
            .lock()
            .unwrap()
            .take()
            .context("moe train op: backward without a saved forward")?;
        let mut scratch = self.scratch(stream)?;
        let d_down = d_down_rows.to_dtype(DType::BF16)?.contiguous()?;
        let mut d_act: CudaSlice<bf16> =
            unsafe { stream.alloc::<bf16>(rows * inter) }.map_err(|e| anyhow::anyhow!(e))?;
        {
            let (st, l) = d_down.storage_and_layout();
            let sl = match &*st {
                Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
                _ => anyhow::bail!("moe train op: d_down not cuda"),
            };
            let view = sl.slice(l.start_offset()..);
            let (dp, _g) = view.device_ptr(stream);
            self.gemm(
                stream,
                &self.stacks.down_transposed,
                &saved.align,
                &mut scratch,
                dp,
                &mut d_act,
                rows,
                1,
                inter,
                hidden,
                "down_t",
            )?;
        }
        let d_act_t = Tensor::from_storage(
            Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
                d_act,
                dev.clone(),
            )),
            (rows, inter),
            candle_core::op::BackpropOp::none(),
            false,
        );
        self.dx_from_d_act(stream, dev, saved, &d_act_t)
    }

    fn dx_from_d_act(
        &self,
        stream: &Arc<CudaStream>,
        dev: &candle_core::CudaDevice,
        saved: Saved,
        d_act: &Tensor,
    ) -> Result<Tensor> {
        let (rows, hidden, inter) = (self.rows(), self.stacks.hidden, self.stacks.inter);
        let mut scratch = self.scratch(stream)?;
        let silu_bwd_span = crate::prof_events::span("bwd_moe_silu_mul_candle_f32", stream);
        let gate_up = Tensor::from_storage(
            Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
                saved.gate_up_rows,
                dev.clone(),
            )),
            (rows, 2 * inter),
            candle_core::op::BackpropOp::none(),
            false,
        )
        .to_dtype(DType::F32)?;
        let d_act_t = d_act.to_dtype(DType::F32)?;
        let g = gate_up.narrow(1, 0, inter)?;
        let u = gate_up.narrow(1, inter, inter)?;
        let sig = candle_nn::ops::sigmoid(&g)?;
        let silu = (&g * &sig)?;
        let d_g = (&d_act_t * &u)?.mul(
            &(&sig * (Tensor::ones_like(&g)? + (&g * (Tensor::ones_like(&sig)? - &sig)?)?)?)?,
        )?;
        let d_u = (&d_act_t * &silu)?;
        drop(silu_bwd_span);
        let mut dx_rows: CudaSlice<bf16> =
            unsafe { stream.alloc::<bf16>(rows * hidden) }.map_err(|e| anyhow::anyhow!(e))?;
        let mut dx_rows_up: CudaSlice<bf16> =
            unsafe { stream.alloc::<bf16>(rows * hidden) }.map_err(|e| anyhow::anyhow!(e))?;
        for (grad, stack, out, what) in [
            (&d_g, &self.stacks.gate_transposed, &mut dx_rows, "gate_t"),
            (&d_u, &self.stacks.up_transposed, &mut dx_rows_up, "up_t"),
        ] {
            let grad = grad.to_dtype(DType::BF16)?.contiguous()?;
            let (st, l) = grad.storage_and_layout();
            let sl = match &*st {
                Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
                _ => anyhow::bail!("moe train op: {what} grad not cuda"),
            };
            let view = sl.slice(l.start_offset()..);
            let (dp, _g) = view.device_ptr(stream);
            self.gemm(
                stream,
                stack,
                &saved.align,
                &mut scratch,
                dp,
                out,
                rows,
                1,
                hidden,
                inter,
                what,
            )?;
        }
        let mut dx: CudaSlice<f32> =
            unsafe { stream.alloc::<f32>(self.tokens * hidden) }.map_err(|e| anyhow::anyhow!(e))?;
        let rc = unsafe {
            let (ap, _g1) = dx_rows.device_ptr(stream);
            let (bp, _g2) = dx_rows_up.device_ptr(stream);
            let (op, _g3) = dx.device_ptr_mut(stream);
            nv_kernels::cuda::moe_slot_sum_pairs_bf16_f32(
                stream.cu_stream() as *mut c_void,
                ap as *const u16,
                bp as *const u16,
                op as *mut f32,
                self.tokens as i32,
                self.top_k as i32,
                hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "moe_slot_sum_pairs rc={rc}");
        Ok(Tensor::from_storage(
            Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(dx, dev.clone())),
            (self.tokens, hidden),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }
}

pub struct MoeExpertsMarlinWeightedTrainOp {
    inner: MoeExpertsMarlinTrainOp,
}

impl MoeExpertsMarlinWeightedTrainOp {
    pub fn new(
        stacks: Arc<MarlinMoeTrainStacks>,
        top_ids: &[u32],
        tokens: usize,
        top_k: usize,
    ) -> Self {
        Self {
            inner: MoeExpertsMarlinTrainOp::new(stacks, top_ids, tokens, top_k),
        }
    }

    pub fn new_with_device_ids(
        stacks: Arc<MarlinMoeTrainStacks>,
        ids_u32: Tensor,
        tokens: usize,
        top_k: usize,
    ) -> Self {
        Self {
            inner: MoeExpertsMarlinTrainOp::new_with_device_ids(stacks, ids_u32, tokens, top_k),
        }
    }
}

fn f32_slice_ptr(
    s: &candle_core::CudaStorage,
    l: &Layout,
    stream: &Arc<CudaStream>,
    what: &str,
) -> candle_core::Result<u64> {
    let Some((start, _end)) = l.contiguous_offsets() else {
        return Err(candle_core::Error::Msg(format!(
            "moe train op: {what} must be contiguous"
        )));
    };
    let x = s.as_cuda_slice::<f32>()?;
    let view = x.slice(start..);
    let (p, _g) = view.device_ptr(stream);
    Ok(p)
}

impl CustomOp2 for MoeExpertsMarlinWeightedTrainOp {
    fn name(&self) -> &'static str {
        "moe_experts_marlin_weighted_train"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle_core::CpuStorage,
        _l1: &Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "moe_experts_marlin_weighted_train is CUDA-only; the per-expert path serves CPU".into(),
        ))
    }

    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        let op = &self.inner;
        let dev = s1.device().clone();
        let stream = crate::cuda_stream::current_stream(&dev);
        let Some((start, _end)) = l1.contiguous_offsets() else {
            return Err(candle_core::Error::Msg(
                "moe train op: input must be contiguous".into(),
            ));
        };
        if l1.shape().dims() != [op.tokens, op.stacks.hidden] {
            return Err(candle_core::Error::Msg(format!(
                "moe train op: input {:?} vs tokens {} hidden {}",
                l1.shape(),
                op.tokens,
                op.stacks.hidden
            )));
        }
        if l2.shape().dims() != [op.tokens, op.top_k] {
            return Err(candle_core::Error::Msg(format!(
                "moe train op: slot weights {:?} vs tokens {} top_k {}",
                l2.shape(),
                op.tokens,
                op.top_k
            )));
        }
        let x = s1.as_cuda_slice::<bf16>()?;
        let x_view = x.slice(start..);
        let (xp, _g) = x_view.device_ptr(&stream);
        let wp = f32_slice_ptr(s2, l2, &stream, "slot weights")?;
        let down_rows = op
            .forward_rows(&stream, xp)
            .map_err(|e| candle_core::Error::Msg(format!("moe train op forward: {e:#}")))?;
        let hidden = op.stacks.hidden;
        let mut out: CudaSlice<f32> = unsafe { stream.alloc::<f32>(op.tokens * hidden) }
            .map_err(|e| candle_core::Error::Msg(format!("moe train op alloc: {e:?}")))?;
        let rc = unsafe {
            let (rp, _g1) = down_rows.device_ptr(&stream);
            let (opp, _g2) = out.device_ptr_mut(&stream);
            nv_kernels::cuda::moe_slot_weighted_sum_bf16_f32(
                stream.cu_stream() as *mut c_void,
                rp as *const u16,
                wp as *const f32,
                opp as *mut f32,
                op.tokens as i32,
                op.top_k as i32,
                hidden as i32,
            )
        };
        if rc != 0 {
            return Err(candle_core::Error::Msg(format!(
                "moe_slot_weighted_sum rc={rc}"
            )));
        }
        if let Some(saved) = op.saved.lock().unwrap().as_mut() {
            saved.down_rows_for_the_slot_weight_grad = Some(down_rows);
        }
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, dev),
            Shape::from((op.tokens, hidden)),
        ))
    }

    fn bwd(
        &self,
        arg1: &Tensor,
        arg2: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        let op = &self.inner;
        let candle_core::Device::Cuda(dev) = arg1.device().clone() else {
            return Err(candle_core::Error::Msg("moe train op bwd: not cuda".into()));
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let hidden = op.stacks.hidden;
        let rows = op.rows();
        let d_out = grad_res.detach().to_dtype(DType::F32)?.contiguous()?;
        let w = arg2.detach().to_dtype(DType::F32)?.contiguous()?;
        let down_rows = op
            .saved
            .lock()
            .unwrap()
            .as_mut()
            .and_then(|s| s.down_rows_for_the_slot_weight_grad.take())
            .ok_or_else(|| {
                candle_core::Error::Msg("moe train op bwd: no saved down rows".into())
            })?;
        let mut dw: CudaSlice<f32> = unsafe { stream.alloc::<f32>(rows) }
            .map_err(|e| candle_core::Error::Msg(format!("moe train op alloc: {e:?}")))?;
        let mut d_rows: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(rows * hidden) }
            .map_err(|e| candle_core::Error::Msg(format!("moe train op alloc: {e:?}")))?;
        {
            let (ds, dl) = d_out.storage_and_layout();
            let (ws, wl) = w.storage_and_layout();
            let (Storage::Cuda(dc), Storage::Cuda(wc)) = (&*ds, &*ws) else {
                return Err(candle_core::Error::Msg(
                    "moe train op bwd: operands not cuda".into(),
                ));
            };
            let dp = f32_slice_ptr(dc, dl, &stream, "d_out")?;
            let wp = f32_slice_ptr(wc, wl, &stream, "slot weights")?;
            let rc = unsafe {
                let (rp, _g1) = down_rows.device_ptr(&stream);
                let (dwp, _g2) = dw.device_ptr_mut(&stream);
                nv_kernels::cuda::moe_slot_weight_grad_bf16_f32(
                    stream.cu_stream() as *mut c_void,
                    rp as *const u16,
                    dp as *const f32,
                    dwp as *mut f32,
                    op.tokens as i32,
                    op.top_k as i32,
                    hidden as i32,
                )
            };
            if rc != 0 {
                return Err(candle_core::Error::Msg(format!(
                    "moe_slot_weight_grad rc={rc}"
                )));
            }
            let rc = unsafe {
                let (drp, _g3) = d_rows.device_ptr_mut(&stream);
                nv_kernels::cuda::moe_slot_scatter_grad_f32_bf16(
                    stream.cu_stream() as *mut c_void,
                    dp as *const f32,
                    wp as *const f32,
                    drp as *mut u16,
                    op.tokens as i32,
                    op.top_k as i32,
                    hidden as i32,
                )
            };
            if rc != 0 {
                return Err(candle_core::Error::Msg(format!(
                    "moe_slot_scatter_grad rc={rc}"
                )));
            }
        }
        let d_rows_t = Tensor::from_storage(
            Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
                d_rows,
                dev.clone(),
            )),
            (rows, hidden),
            candle_core::op::BackpropOp::none(),
            false,
        );
        let dx = op
            .backward_rows(&stream, &dev, &d_rows_t)
            .map_err(|e| candle_core::Error::Msg(format!("moe train op backward: {e:#}")))?;
        let dw_t = Tensor::from_storage(
            Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(dw, dev.clone())),
            (op.tokens, op.top_k),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok((
            Some(dx.to_dtype(arg1.dtype())?),
            Some(dw_t.to_dtype(arg2.dtype())?),
        ))
    }
}

impl CustomOp1 for MoeExpertsMarlinTrainOp {
    fn name(&self) -> &'static str {
        "moe_experts_marlin_train"
    }

    fn cpu_fwd(
        &self,
        _s: &candle_core::CpuStorage,
        _l: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "moe_experts_marlin_train is CUDA-only; the per-expert path serves CPU".into(),
        ))
    }

    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        let dev = s.device().clone();
        let stream = crate::cuda_stream::current_stream(&dev);
        let Some((start, _end)) = l.contiguous_offsets() else {
            return Err(candle_core::Error::Msg(
                "moe train op: input must be contiguous".into(),
            ));
        };
        if l.shape().dims() != [self.tokens, self.stacks.hidden] {
            return Err(candle_core::Error::Msg(format!(
                "moe train op: input {:?} vs tokens {} hidden {}",
                l.shape(),
                self.tokens,
                self.stacks.hidden
            )));
        }
        let x = s.as_cuda_slice::<bf16>()?;
        let x_view = x.slice(start..);
        let (xp, _g) = x_view.device_ptr(&stream);
        let rows_out = self
            .forward_rows(&stream, xp)
            .map_err(|e| candle_core::Error::Msg(format!("moe train op forward: {e:#}")))?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(rows_out, dev),
            Shape::from((self.rows(), self.stacks.hidden)),
        ))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let candle_core::Device::Cuda(dev) = arg.device().clone() else {
            return Err(candle_core::Error::Msg("moe train op bwd: not cuda".into()));
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let dx = self
            .backward_rows(&stream, &dev, &grad_res.detach())
            .map_err(|e| candle_core::Error::Msg(format!("moe train op backward: {e:#}")))?;
        Ok(Some(dx.to_dtype(arg.dtype())?))
    }
}

pub struct MoeExpertsMarlinActRowsOp {
    inner: MoeExpertsMarlinTrainOp,
}

impl MoeExpertsMarlinActRowsOp {
    pub fn new_with_device_ids(
        stacks: Arc<MarlinMoeTrainStacks>,
        ids_u32: Tensor,
        tokens: usize,
        top_k: usize,
    ) -> Self {
        Self {
            inner: MoeExpertsMarlinTrainOp::new_with_device_ids(stacks, ids_u32, tokens, top_k),
        }
    }
}

impl CustomOp1 for MoeExpertsMarlinActRowsOp {
    fn name(&self) -> &'static str {
        "moe_experts_marlin_act_rows"
    }

    fn cpu_fwd(
        &self,
        _s: &candle_core::CpuStorage,
        _l: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        Err(candle_core::Error::Msg("moe_experts_marlin_act_rows is CUDA-only".into()))
    }

    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        let op = &self.inner;
        let dev = s.device().clone();
        let stream = crate::cuda_stream::current_stream(&dev);
        let Some((start, _end)) = l.contiguous_offsets() else {
            return Err(candle_core::Error::Msg("moe act rows op: input must be contiguous".into()));
        };
        if l.shape().dims() != [op.tokens, op.stacks.hidden] {
            return Err(candle_core::Error::Msg(format!(
                "moe act rows op: input {:?} vs tokens {} hidden {}",
                l.shape(),
                op.tokens,
                op.stacks.hidden
            )));
        }
        let x = s.as_cuda_slice::<bf16>()?;
        let x_view = x.slice(start..);
        let (xp, _g) = x_view.device_ptr(&stream);
        let (act, gate_up_rows, align, _scratch) = op
            .act_rows(&stream, xp)
            .map_err(|e| candle_core::Error::Msg(format!("moe act rows op forward: {e:#}")))?;
        *op.saved.lock().unwrap() = Some(Saved {
            gate_up_rows,
            align,
            down_rows_for_the_slot_weight_grad: None,
        });
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(act, dev),
            Shape::from((op.rows(), op.stacks.inter)),
        ))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let op = &self.inner;
        let candle_core::Device::Cuda(dev) = arg.device().clone() else {
            return Err(candle_core::Error::Msg("moe act rows op bwd: not cuda".into()));
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let saved = op
            .saved
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| candle_core::Error::Msg("moe act rows op: backward without a saved forward".into()))?;
        let dx = op
            .dx_from_d_act(&stream, &dev, saved, &grad_res.detach())
            .map_err(|e| candle_core::Error::Msg(format!("moe act rows op backward: {e:#}")))?;
        Ok(Some(dx.to_dtype(arg.dtype())?))
    }
}
