#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use crate::mlp::Mlp;

const MIN_TILE: usize = nv_quant::nvfp4::MIN_TILE;
const BLOCK_SIZE: usize = nv_quant::nvfp4::BLOCK_SIZE;

pub fn moe_arm_on_unless_the_key_is_explicitly_off_which_is_how_the_pre_round_82_path_is_reached(
    key: &str,
) -> bool {
    !matches!(
        std::env::var(key).ok().as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}

pub fn folded_shared_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        moe_arm_on_unless_the_key_is_explicitly_off_which_is_how_the_pre_round_82_path_is_reached(
            "NV_MOE_SHARED_FOLD",
        )
    })
}

pub fn route_fuse_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        moe_arm_on_unless_the_key_is_explicitly_off_which_is_how_the_pre_round_82_path_is_reached(
            "NV_MOE_ROUTE_FUSE",
        )
    })
}

pub fn chain_slim_launches_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        moe_arm_on_unless_the_key_is_explicitly_off_which_is_how_the_pre_round_82_path_is_reached(
            "NV_MOE_CHAIN_SLIM",
        )
    })
}

const ROUTE_QUANT_SPLIT_CTAS_UNDER_CHAIN_SLIM: i32 = 8;

const ROUTE_FUSE_SINGLE_CTA_TOPK_HOLDS_UP_TO_256_EXPERTS: usize = 256;

pub const ROUTE_FUSE_ONE_CTA_PER_TOKEN_HOLDS_TO_8_VERIFY_ROWS: usize = 8;

pub struct MoeGroupedWeights {
    pub num_experts: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,

    pub folded_shared: bool,

    pub gate_w: CudaSlice<u8>,
    pub gate_w_scales: CudaSlice<u8>,
    pub gate_alphas: CudaSlice<f32>,
    pub gate_a_stride_elems: i64,
    pub gate_b_stride_elems: i64,
    pub gate_c_stride_elems: i64,

    pub up_w: CudaSlice<u8>,
    pub up_w_scales: CudaSlice<u8>,
    pub up_alphas: CudaSlice<f32>,

    pub down_w: CudaSlice<u8>,
    pub down_w_scales: CudaSlice<u8>,
    pub down_alphas: CudaSlice<f32>,
    pub down_a_stride_elems: i64,
    pub down_b_stride_elems: i64,
    pub down_c_stride_elems: i64,

    pub runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,

    pub input_globals_gate_up: CudaSlice<f32>,

    pub input_globals_down: CudaSlice<f32>,

    pub input_globals_gate_up_host: Vec<f32>,

    pub input_globals_down_host: Vec<f32>,
    pub marlin_moe_twin_built_once_from_the_resident_grouped_buffers:
        std::sync::OnceLock<Option<Arc<MarlinMoeTwin>>>,
}

pub struct MarlinMoeStack {
    pub b_q: CudaSlice<u32>,
    pub scales: CudaSlice<u8>,
    pub globals: CudaSlice<f32>,
}

pub struct MarlinMoeTwin {
    pub gate: MarlinMoeStack,
    pub up: MarlinMoeStack,
    pub gate_up_concat_when_every_expert_shares_one_global_scale: Option<MarlinMoeStack>,
    pub down: MarlinMoeStack,
    pub experts_total: usize,
    pub hidden: usize,
    pub inter: usize,
}

fn marlin_moe_stack(
    stream: &Arc<CudaStream>,
    packed: &CudaSlice<u8>,
    swizzled: &CudaSlice<u8>,
    alphas: &CudaSlice<f32>,
    input_globals_host: &[f32],
    experts: usize,
    n: usize,
    k: usize,
) -> Result<MarlinMoeStack> {
    let packed_h = stream.clone_dtoh(packed)?;
    let swz_h = stream.clone_dtoh(swizzled)?;
    let alphas_h = stream.clone_dtoh(alphas)?;
    let swz_per = swizzled_scale_bytes(n, k);
    let mut linear_per_expert = Vec::with_capacity(experts);
    for e in 0..experts {
        linear_per_expert.push(nv_quant::nvfp4::unswizzle_scales(
            &swz_h[e * swz_per..(e + 1) * swz_per],
            n,
            k / 16,
        ));
    }
    let globals: Vec<f32> = (0..experts)
        .map(|e| alphas_h[e] * input_globals_host[e])
        .collect();
    marlin_moe_stack_from_linear(
        stream,
        &packed_h,
        &linear_per_expert,
        &globals,
        experts,
        n,
        k,
    )
}

pub(crate) fn marlin_moe_stack_from_linear(
    stream: &Arc<CudaStream>,
    packed_h: &[u8],
    scales_linear_per_expert: &[Vec<u8>],
    globals_in: &[f32],
    experts: usize,
    n: usize,
    k: usize,
) -> Result<MarlinMoeStack> {
    anyhow::ensure!(
        packed_h.len() == experts * n * k / 2,
        "marlin moe stack: packed bytes {} != E*n*k/2",
        packed_h.len()
    );
    anyhow::ensure!(
        scales_linear_per_expert.len() == experts && globals_in.len() == experts,
        "marlin moe stack: per-expert inputs short"
    );
    let words = k * n / 8;
    let mut b_q: CudaSlice<u32> = stream.alloc_zeros::<u32>(experts * words)?;
    let mut scales_all = Vec::with_capacity(experts * k * n / 16);
    let mut globals = Vec::with_capacity(experts);
    for e in 0..experts {
        let pk = &packed_h[e * n * k / 2..(e + 1) * n * k / 2];
        let kmajor = nv_quant::marlin_nvfp4::kmajor_int32_from_packed_rows(pk, n, k);
        let sc = nv_quant::marlin_nvfp4::marlin_scales_from_linear(
            &scales_linear_per_expert[e],
            n,
            k,
            globals_in[e],
        );
        #[allow(deprecated)]
        let d_kmajor: CudaSlice<u32> = stream.clone_htod(&kmajor)?;
        let rc = unsafe {
            let (pi, _g1) = d_kmajor.device_ptr(stream);
            let (po, _g2) = b_q.device_ptr_mut(stream);
            nv_kernels::cuda::marlin_repack_w4a16(
                stream.cu_stream() as *mut c_void,
                pi as *const c_void,
                (po + (e * words * 4) as u64) as *mut c_void,
                k as i32,
                n as i32,
                4,
            )
        };
        anyhow::ensure!(rc == 0, "marlin moe repack expert {e} rc={rc}");
        stream.synchronize()?;
        scales_all.extend_from_slice(&sc.bytes_e5m3);
        globals.push(sc.global_scale_processed);
    }
    #[allow(deprecated)]
    let scales = stream.clone_htod(&scales_all)?;
    #[allow(deprecated)]
    let globals = stream.clone_htod(&globals)?;
    Ok(MarlinMoeStack {
        b_q,
        scales,
        globals,
    })
}

impl MoeGroupedWeights {
    pub fn marlin_moe_twin(&self, stream: &Arc<CudaStream>) -> Option<Arc<MarlinMoeTwin>> {
        self.marlin_moe_twin_built_once_from_the_resident_grouped_buffers
            .get_or_init(|| {
                let hidden = self.hidden_size;
                let inter = self.intermediate_size;
                let experts_total = self.gate_w.len() * 2 / (inter * hidden);
                let build = || -> Result<MarlinMoeTwin> {
                    let gate = marlin_moe_stack(stream, &self.gate_w, &self.gate_w_scales, &self.gate_alphas, &self.input_globals_gate_up_host, experts_total, inter, hidden)?;
                    let up = marlin_moe_stack(stream, &self.up_w, &self.up_w_scales, &self.up_alphas, &self.input_globals_gate_up_host, experts_total, inter, hidden)?;
                    let down = marlin_moe_stack(stream, &self.down_w, &self.down_w_scales, &self.down_alphas, &self.input_globals_down_host, experts_total, hidden, inter)?;
                    let ga = stream.clone_dtoh(&self.gate_alphas)?;
                    let ua = stream.clone_dtoh(&self.up_alphas)?;
                    let concat = if ga[..experts_total] == ua[..experts_total] {
                        let gp = stream.clone_dtoh(&self.gate_w)?;
                        let upk = stream.clone_dtoh(&self.up_w)?;
                        let gs = stream.clone_dtoh(&self.gate_w_scales)?;
                        let us = stream.clone_dtoh(&self.up_w_scales)?;
                        let per_w = inter * hidden / 2;
                        let swz_per = swizzled_scale_bytes(inter, hidden);
                        let mut packed = Vec::with_capacity(experts_total * 2 * per_w);
                        let mut lin = Vec::with_capacity(experts_total);
                        for e in 0..experts_total {
                            packed.extend_from_slice(&gp[e * per_w..(e + 1) * per_w]);
                            packed.extend_from_slice(&upk[e * per_w..(e + 1) * per_w]);
                            let mut l = nv_quant::nvfp4::unswizzle_scales(&gs[e * swz_per..(e + 1) * swz_per], inter, hidden / 16);
                            l.extend_from_slice(&nv_quant::nvfp4::unswizzle_scales(&us[e * swz_per..(e + 1) * swz_per], inter, hidden / 16));
                            lin.push(l);
                        }
                        let globals: Vec<f32> = (0..experts_total).map(|e| ga[e] * self.input_globals_gate_up_host[e]).collect();
                        Some(marlin_moe_stack_from_linear(stream, &packed, &lin, &globals, experts_total, 2 * inter, hidden)?)
                    } else {
                        None
                    };
                    Ok(MarlinMoeTwin { gate, up, gate_up_concat_when_every_expert_shares_one_global_scale: concat, down, experts_total, hidden, inter })
                };
                match build() {
                    Ok(t) => {
                        eprintln!("[marlin_moe_twin] built experts={} hidden={hidden} inter={inter} gate_up_concat={}", t.experts_total, t.gate_up_concat_when_every_expert_shares_one_global_scale.is_some());
                        Some(Arc::new(t))
                    }
                    Err(e) => {
                        eprintln!("[marlin_moe_twin] unavailable: {e:#}");
                        None
                    }
                }
            })
            .clone()
    }
}

impl MoeGroupedWeights {
    pub fn build_from_experts(
        experts: &[Mlp],
        hidden_size: usize,
        intermediate_size: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        let list: Vec<&Mlp> = experts.iter().collect();
        Self::build_from_expert_list(&list, experts.len(), hidden_size, intermediate_size, stream)
    }

    pub fn build_from_experts_folding_shared_as_a_fixed_extra_tile(
        experts: &[Mlp],
        shared: &Mlp,
        hidden_size: usize,
        intermediate_size: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        let mut list: Vec<&Mlp> = experts.iter().collect();
        list.push(shared);
        Self::build_from_expert_list(&list, experts.len(), hidden_size, intermediate_size, stream)
    }

    fn build_from_expert_list(
        experts: &[&Mlp],
        routed_experts: usize,
        hidden_size: usize,
        intermediate_size: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        let num_experts = experts.len();
        anyhow::ensure!(num_experts > 0, "MoeGroupedWeights: no experts");
        anyhow::ensure!(
            routed_experts == num_experts || routed_experts + 1 == num_experts,
            "MoeGroupedWeights: expected the appended list to be routed or routed+shared, got {num_experts} entries for {routed_experts} routed"
        );

        let runner = experts[0]
            .gate_proj()
            .nvfp4_runner()
            .context("MoeGroupedWeights: gate_proj is not NVFP4 -- cannot build grouped storage")?;

        let gate_w_bytes_per_expert = intermediate_size * hidden_size / 2;
        let gate_w_scale_bytes_per_expert = swizzled_scale_bytes(intermediate_size, hidden_size);

        let mut gate_w_concat = vec![0u8; num_experts * gate_w_bytes_per_expert];
        let mut gate_w_scales_concat = vec![0u8; num_experts * gate_w_scale_bytes_per_expert];
        let mut gate_alphas = vec![0f32; num_experts];

        let mut up_w_concat = vec![0u8; num_experts * gate_w_bytes_per_expert];
        let mut up_w_scales_concat = vec![0u8; num_experts * gate_w_scale_bytes_per_expert];
        let mut up_alphas = vec![0f32; num_experts];

        let down_w_bytes_per_expert = hidden_size * intermediate_size / 2;
        let down_w_scale_bytes_per_expert = swizzled_scale_bytes(hidden_size, intermediate_size);

        let mut down_w_concat = vec![0u8; num_experts * down_w_bytes_per_expert];
        let mut down_w_scales_concat = vec![0u8; num_experts * down_w_scale_bytes_per_expert];
        let mut down_alphas = vec![0f32; num_experts];

        let mut globals_gate_up_host = Vec::with_capacity(num_experts);
        let mut globals_down_host = Vec::with_capacity(num_experts);
        for (e, mlp) in experts.iter().enumerate() {
            let g_gate = mlp
                .gate_proj()
                .nvfp4_parts()
                .with_context(|| format!("expert {e} gate_proj missing nvfp4 parts"))?
                .3;
            let g_up = mlp
                .up_proj()
                .nvfp4_parts()
                .with_context(|| format!("expert {e} up_proj missing nvfp4 parts"))?
                .3;
            let g_down = mlp
                .down_proj()
                .nvfp4_parts()
                .with_context(|| format!("expert {e} down_proj missing nvfp4 parts"))?
                .3;
            if (g_gate - g_up).abs() > 1e-6 {
                anyhow::bail!(
                    "expert {e}: gate and up have different input_global_scale ({} vs {})",
                    g_gate,
                    g_up
                );
            }
            globals_gate_up_host.push(g_gate);
            globals_down_host.push(g_down);
        }

        for (e, mlp) in experts.iter().enumerate() {
            let (gw, gs, ga, _) = mlp
                .gate_proj()
                .nvfp4_parts()
                .with_context(|| format!("MoeGroupedWeights: expert {e} gate_proj not NVFP4"))?;
            let (uw, us, ua, _) = mlp
                .up_proj()
                .nvfp4_parts()
                .with_context(|| format!("MoeGroupedWeights: expert {e} up_proj not NVFP4"))?;
            let (dw, ds, da, _) = mlp
                .down_proj()
                .nvfp4_parts()
                .with_context(|| format!("MoeGroupedWeights: expert {e} down_proj not NVFP4"))?;

            anyhow::ensure!(
                gw.len() == gate_w_bytes_per_expert,
                "expert {e} gate weight size {} != expected {}",
                gw.len(),
                gate_w_bytes_per_expert
            );
            anyhow::ensure!(
                uw.len() == gate_w_bytes_per_expert,
                "expert {e} up weight size mismatch"
            );
            anyhow::ensure!(
                dw.len() == down_w_bytes_per_expert,
                "expert {e} down weight size mismatch"
            );

            #[allow(deprecated)]
            let gw_host = stream.memcpy_dtov(gw)?;
            #[allow(deprecated)]
            let gs_host = stream.memcpy_dtov(gs)?;
            #[allow(deprecated)]
            let uw_host = stream.memcpy_dtov(uw)?;
            #[allow(deprecated)]
            let us_host = stream.memcpy_dtov(us)?;
            #[allow(deprecated)]
            let dw_host = stream.memcpy_dtov(dw)?;
            #[allow(deprecated)]
            let ds_host = stream.memcpy_dtov(ds)?;

            gate_w_concat[e * gate_w_bytes_per_expert..(e + 1) * gate_w_bytes_per_expert]
                .copy_from_slice(&gw_host);
            up_w_concat[e * gate_w_bytes_per_expert..(e + 1) * gate_w_bytes_per_expert]
                .copy_from_slice(&uw_host);
            down_w_concat[e * down_w_bytes_per_expert..(e + 1) * down_w_bytes_per_expert]
                .copy_from_slice(&dw_host);

            let gs_slice = &gs_host[..gate_w_scale_bytes_per_expert.min(gs_host.len())];
            gate_w_scales_concat[e * gate_w_scale_bytes_per_expert
                ..e * gate_w_scale_bytes_per_expert + gs_slice.len()]
                .copy_from_slice(gs_slice);
            let us_slice = &us_host[..gate_w_scale_bytes_per_expert.min(us_host.len())];
            up_w_scales_concat[e * gate_w_scale_bytes_per_expert
                ..e * gate_w_scale_bytes_per_expert + us_slice.len()]
                .copy_from_slice(us_slice);
            let ds_slice = &ds_host[..down_w_scale_bytes_per_expert.min(ds_host.len())];
            down_w_scales_concat[e * down_w_scale_bytes_per_expert
                ..e * down_w_scale_bytes_per_expert + ds_slice.len()]
                .copy_from_slice(ds_slice);

            gate_alphas[e] = ga;
            up_alphas[e] = ua;
            down_alphas[e] = da;
        }
        #[allow(deprecated)]
        let input_globals_gate_up = stream.memcpy_stod(&globals_gate_up_host)?;
        #[allow(deprecated)]
        let input_globals_down = stream.memcpy_stod(&globals_down_host)?;

        #[allow(deprecated)]
        let gate_w = stream.memcpy_stod(&gate_w_concat)?;
        #[allow(deprecated)]
        let gate_w_scales = stream.memcpy_stod(&gate_w_scales_concat)?;
        #[allow(deprecated)]
        let gate_alphas_dev = stream.memcpy_stod(&gate_alphas)?;
        #[allow(deprecated)]
        let up_w = stream.memcpy_stod(&up_w_concat)?;
        #[allow(deprecated)]
        let up_w_scales = stream.memcpy_stod(&up_w_scales_concat)?;
        #[allow(deprecated)]
        let up_alphas_dev = stream.memcpy_stod(&up_alphas)?;
        #[allow(deprecated)]
        let down_w = stream.memcpy_stod(&down_w_concat)?;
        #[allow(deprecated)]
        let down_w_scales = stream.memcpy_stod(&down_w_scales_concat)?;
        #[allow(deprecated)]
        let down_alphas_dev = stream.memcpy_stod(&down_alphas)?;

        let built = Self {
            num_experts: routed_experts,
            hidden_size,
            intermediate_size,
            folded_shared: num_experts != routed_experts,
            gate_w,
            gate_w_scales,
            gate_alphas: gate_alphas_dev,
            gate_a_stride_elems: hidden_size as i64,
            gate_b_stride_elems: hidden_size as i64,
            gate_c_stride_elems: intermediate_size as i64,
            up_w,
            up_w_scales,
            up_alphas: up_alphas_dev,
            down_w,
            down_w_scales,
            down_alphas: down_alphas_dev,
            down_a_stride_elems: intermediate_size as i64,
            down_b_stride_elems: intermediate_size as i64,
            down_c_stride_elems: hidden_size as i64,
            runner,
            input_globals_gate_up,
            input_globals_down,
            input_globals_gate_up_host: globals_gate_up_host,
            input_globals_down_host: globals_down_host,
            marlin_moe_twin_built_once_from_the_resident_grouped_buffers: std::sync::OnceLock::new(
            ),
        };
        if marlin_moe_route_opt_in() {
            let _ = built.marlin_moe_twin(stream);
        }
        Ok(built)
    }

    pub fn append_shared_expert_as_tile_e_costs_one_extra_expert_copy_on_top_of_the_cached_base(
        base: &Self,
        shared: &Mlp,
        shared_expert_intermediate_size: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        let e = base.num_experts;
        let hidden = base.hidden_size;
        let inter = base.intermediate_size;
        anyhow::ensure!(
            !base.folded_shared,
            "base grouped weights already fold the shared expert"
        );
        anyhow::ensure!(
            shared_expert_intermediate_size == inter,
            "shared fold needs shared_expert_intermediate_size {shared_expert_intermediate_size} \
             == moe_intermediate_size {inter}"
        );
        let (sg_w, sg_s, sg_a, sg_g) = shared
            .gate_proj()
            .nvfp4_parts()
            .context("shared fold: shared gate_proj is not NVFP4")?;
        let (su_w, su_s, su_a, su_g) = shared
            .up_proj()
            .nvfp4_parts()
            .context("shared fold: shared up_proj is not NVFP4")?;
        let (sd_w, sd_s, sd_a, sd_g) = shared
            .down_proj()
            .nvfp4_parts()
            .context("shared fold: shared down_proj is not NVFP4")?;
        anyhow::ensure!(
            (sg_g - su_g).abs() <= 1e-6,
            "shared fold: gate/up input_global_scale differ ({sg_g} vs {su_g})"
        );
        let gate_bytes = inter * hidden / 2;
        let gate_scale_bytes = swizzled_scale_bytes(inter, hidden);
        let down_bytes = hidden * inter / 2;
        let down_scale_bytes = swizzled_scale_bytes(hidden, inter);

        let append_bytes = |old: &CudaSlice<u8>,
                            tail: &CudaSlice<u8>,
                            per: usize,
                            exact: bool|
         -> Result<CudaSlice<u8>> {
            anyhow::ensure!(
                old.len() == e * per,
                "shared fold: concat len {} != experts {e} x {per}",
                old.len()
            );
            anyhow::ensure!(
                !exact || tail.len() == per,
                "shared fold: shared tensor len {} != per-expert {per}",
                tail.len()
            );
            let mut grown = stream.alloc_zeros::<u8>((e + 1) * per)?;
            let n = per.min(tail.len());
            #[allow(deprecated)]
            {
                stream.memcpy_dtod(old, &mut grown.slice_mut(0..e * per))?;
                stream.memcpy_dtod(
                    &tail.slice(0..n),
                    &mut grown.slice_mut(e * per..e * per + n),
                )?;
            }
            Ok(grown)
        };
        let append_alpha = |old: &CudaSlice<f32>, alpha: f32| -> Result<CudaSlice<f32>> {
            anyhow::ensure!(
                old.len() == e,
                "shared fold: alpha len {} != {e}",
                old.len()
            );
            let mut grown = stream.alloc_zeros::<f32>(e + 1)?;
            #[allow(deprecated)]
            {
                stream.memcpy_dtod(old, &mut grown.slice_mut(0..e))?;
                stream.memcpy_htod(&[alpha], &mut grown.slice_mut(e..e + 1))?;
            }
            Ok(grown)
        };

        let gate_w = append_bytes(&base.gate_w, sg_w, gate_bytes, true)?;
        let gate_w_scales = append_bytes(&base.gate_w_scales, sg_s, gate_scale_bytes, false)?;
        let gate_alphas = append_alpha(&base.gate_alphas, sg_a)?;
        let up_w = append_bytes(&base.up_w, su_w, gate_bytes, true)?;
        let up_w_scales = append_bytes(&base.up_w_scales, su_s, gate_scale_bytes, false)?;
        let up_alphas = append_alpha(&base.up_alphas, su_a)?;
        let down_w = append_bytes(&base.down_w, sd_w, down_bytes, true)?;
        let down_w_scales = append_bytes(&base.down_w_scales, sd_s, down_scale_bytes, false)?;
        let down_alphas = append_alpha(&base.down_alphas, sd_a)?;

        let mut globals_gate_up_host = base.input_globals_gate_up_host.clone();
        globals_gate_up_host.push(sg_g);
        let mut globals_down_host = base.input_globals_down_host.clone();
        globals_down_host.push(sd_g);
        anyhow::ensure!(
            globals_gate_up_host.len() == e + 1 && globals_down_host.len() == e + 1,
            "shared fold: globals host vec sized {} / {}, want {}",
            globals_gate_up_host.len(),
            globals_down_host.len(),
            e + 1
        );
        #[allow(deprecated)]
        let input_globals_gate_up = stream.memcpy_stod(&globals_gate_up_host)?;
        #[allow(deprecated)]
        let input_globals_down = stream.memcpy_stod(&globals_down_host)?;

        Ok(Self {
            num_experts: e,
            hidden_size: hidden,
            intermediate_size: inter,
            folded_shared: true,
            gate_w,
            gate_w_scales,
            gate_alphas,
            gate_a_stride_elems: base.gate_a_stride_elems,
            gate_b_stride_elems: base.gate_b_stride_elems,
            gate_c_stride_elems: base.gate_c_stride_elems,
            up_w,
            up_w_scales,
            up_alphas,
            down_w,
            down_w_scales,
            down_alphas,
            down_a_stride_elems: base.down_a_stride_elems,
            down_b_stride_elems: base.down_b_stride_elems,
            down_c_stride_elems: base.down_c_stride_elems,
            runner: base.runner.clone(),
            input_globals_gate_up,
            input_globals_down,
            input_globals_gate_up_host: globals_gate_up_host,
            input_globals_down_host: globals_down_host,
            marlin_moe_twin_built_once_from_the_resident_grouped_buffers: std::sync::OnceLock::new(
            ),
        })
    }
}

pub fn per_expert_bytes(hidden_size: usize, intermediate_size: usize) -> usize {
    let gate_w = intermediate_size * hidden_size / 2;
    let down_w = hidden_size * intermediate_size / 2;
    2 * (gate_w + swizzled_scale_bytes(intermediate_size, hidden_size))
        + (down_w + swizzled_scale_bytes(hidden_size, intermediate_size))
}

pub fn swizzled_scale_bytes(rows: usize, cols: usize) -> usize {
    let k_blocks = cols / BLOCK_SIZE;
    let m_tiles = rows.div_ceil(128);
    let k_tiles = k_blocks.div_ceil(4);
    m_tiles * 128 * k_tiles * 4
}

struct RoutingPlan {
    active_experts: Vec<usize>,

    real_m_per_active: Vec<usize>,

    m_total_padded: usize,

    src_idx: Vec<i32>,

    inv_perm: Vec<i32>,

    overflow: bool,
}

fn plan_routing(topk_ids: &[u32], n_tokens: usize, k: usize, num_experts: usize) -> RoutingPlan {
    let mut buckets: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_experts];
    for n in 0..n_tokens {
        for j in 0..k {
            let e = topk_ids[n * k + j] as usize;
            buckets[e].push((n as u32, j as u32));
        }
    }
    let mut active_experts = Vec::new();
    let mut real_m_per_active = Vec::new();
    for (e, b) in buckets.iter().enumerate() {
        if !b.is_empty() {
            active_experts.push(e);
            real_m_per_active.push(b.len());
        }
    }
    let m_total_padded = active_experts.len() * MIN_TILE;
    let mut src_idx = vec![-1i32; m_total_padded];
    let mut inv_perm = vec![-1i32; n_tokens * k];
    let mut overflow = false;
    for (i, &e) in active_experts.iter().enumerate() {
        let base = i * MIN_TILE;
        for (j, &(token, slot)) in buckets[e].iter().enumerate() {
            if j >= MIN_TILE {
                overflow = true;
                break;
            }
            src_idx[base + j] = token as i32;
            inv_perm[(token as usize) * k + slot as usize] = (base + j) as i32;
        }
    }
    RoutingPlan {
        active_experts,
        real_m_per_active,
        m_total_padded,
        src_idx,
        inv_perm,
        overflow,
    }
}

struct DeviceTopk {
    stream: Arc<CudaStream>,
    ids: CudaSlice<i32>,
    weights: CudaSlice<f32>,
}

pub struct PrefillDeviceRouting {
    stream: Arc<CudaStream>,
    _topk_ids: CudaSlice<i32>,
    topk_weights: CudaSlice<f32>,
    src_idx: CudaSlice<i32>,
    inv_perm: CudaSlice<i32>,
    active_experts: Vec<usize>,
    m_total_padded: usize,
}

#[allow(clippy::too_many_arguments)]
fn route_topk_device(
    router_logits: &Tensor,
    selection_bias: Option<&Tensor>,
    n_tokens: usize,
    num_experts: usize,
    k: usize,
    mode: i32,
    softcap: f32,
    norm_topk: bool,
    routed_scaling: f32,
    device: &Device,
) -> Result<DeviceTopk> {
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("device prefill routing requires CUDA"),
    };
    anyhow::ensure!(
        n_tokens > 0,
        "device prefill routing requires at least one token"
    );
    anyhow::ensure!(
        k > 0 && k <= num_experts,
        "invalid prefill top-k {k}/{num_experts}"
    );
    let stream = crate::cuda_stream::current_stream(&dev);
    let logits = router_logits.to_dtype(DType::F32)?.contiguous()?;
    anyhow::ensure!(
        logits.elem_count() == n_tokens * num_experts,
        "prefill router logits have {} elements, expected {}x{}",
        logits.elem_count(),
        n_tokens,
        num_experts
    );
    let bias = match selection_bias {
        Some(b) => Some(b.to_dtype(DType::F32)?.contiguous()?),
        None => None,
    };
    if let Some(b) = &bias {
        anyhow::ensure!(
            b.elem_count() == num_experts,
            "prefill selection bias has {} elements, expected {num_experts}",
            b.elem_count()
        );
    }

    let routes = n_tokens
        .checked_mul(k)
        .context("prefill routing route count overflow")?;
    let mut ids = stream.alloc_zeros::<i32>(routes)?;
    let mut weights = stream.alloc_zeros::<f32>(routes)?;
    {
        let (ls, ll) = logits.storage_and_layout();
        let l_cuda = match &*ls {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("prefill router logits must be CUDA"),
        };
        let l_slice = l_cuda.as_cuda_slice::<f32>()?;
        let (lp, _gl) = l_slice.device_ptr(&stream);
        let lp = lp + (ll.start_offset() * std::mem::size_of::<f32>()) as u64;
        let bias_ptr = match &bias {
            Some(b) => {
                let (bs, bl) = b.storage_and_layout();
                let b_cuda = match &*bs {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("prefill selection bias must be CUDA"),
                };
                let b_slice = b_cuda.as_cuda_slice::<f32>()?;
                let (bp, _gb) = b_slice.device_ptr(&stream);
                bp + (bl.start_offset() * std::mem::size_of::<f32>()) as u64
            }
            None => 0,
        };
        let (ip, _gi) = ids.device_ptr_mut(&stream);
        let (wp, _gw) = weights.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::moe_route_topk(
                stream.cu_stream() as *mut c_void,
                lp as *const f32,
                bias_ptr as *const f32,
                ip as *mut i32,
                wp as *mut f32,
                n_tokens as i32,
                num_experts as i32,
                k as i32,
                mode,
                softcap,
                norm_topk as i32,
                routed_scaling,
            )
        };
        anyhow::ensure!(rc == 0, "prefill moe_route_topk rc={rc}");
    }
    Ok(DeviceTopk {
        stream,
        ids,
        weights,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn route_topk_ids_on_the_device_as_u32_tokens_by_k(
    router_logits: &Tensor,
    selection_bias: Option<&Tensor>,
    n_tokens: usize,
    num_experts: usize,
    k: usize,
    mode: i32,
    softcap: f32,
    device: &Device,
) -> Result<Tensor> {
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("device routing requires CUDA"),
    };
    anyhow::ensure!(n_tokens > 0 && k > 0 && k <= num_experts, "invalid device routing shape");
    let stream = crate::cuda_stream::current_stream(&dev);
    let logits = router_logits.to_dtype(DType::F32)?.contiguous()?;
    anyhow::ensure!(
        logits.elem_count() == n_tokens * num_experts,
        "router logits have {} elements, expected {n_tokens}x{num_experts}",
        logits.elem_count()
    );
    let bias = match selection_bias {
        Some(b) => Some(b.to_dtype(DType::F32)?.contiguous()?),
        None => None,
    };
    let routes = n_tokens * k;
    let mut ids = stream.alloc_zeros::<u32>(routes)?;
    let mut weights = stream.alloc_zeros::<f32>(routes)?;
    let f32_ptr = |t: &Tensor| -> Result<u64> {
        let (st, l) = t.storage_and_layout();
        let candle_core::Storage::Cuda(c) = &*st else {
            anyhow::bail!("device routing operand must be CUDA");
        };
        let (p, _g) = c.as_cuda_slice::<f32>()?.device_ptr(&stream);
        Ok(p + (l.start_offset() * 4) as u64)
    };
    let lp = f32_ptr(&logits)?;
    let bp = match &bias {
        Some(b) => f32_ptr(b)?,
        None => 0,
    };
    let rc = unsafe {
        let (ip, _gi) = ids.device_ptr_mut(&stream);
        let (wp, _gw) = weights.device_ptr_mut(&stream);
        nv_kernels::cuda::moe_route_topk(
            stream.cu_stream() as *mut std::ffi::c_void,
            lp as *const f32,
            bp as *const f32,
            ip as *mut i32,
            wp as *mut f32,
            n_tokens as i32,
            num_experts as i32,
            k as i32,
            mode,
            softcap,
            0,
            1.0,
        )
    };
    anyhow::ensure!(rc == 0, "device moe_route_topk rc={rc}");
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(ids, dev)),
        (n_tokens, k),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn route_topk_device_then_copy_compact_for_prefill_dispatch(
    router_logits: &Tensor,
    selection_bias: Option<&Tensor>,
    n_tokens: usize,
    num_experts: usize,
    k: usize,
    mode: i32,
    softcap: f32,
    norm_topk: bool,
    routed_scaling: f32,
    device: &Device,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let routed = route_topk_device(
        router_logits,
        selection_bias,
        n_tokens,
        num_experts,
        k,
        mode,
        softcap,
        norm_topk,
        routed_scaling,
        device,
    )?;

    #[allow(deprecated)]
    let ids_host: Vec<i32> = routed.stream.memcpy_dtov(&routed.ids)?;
    #[allow(deprecated)]
    let weights_host: Vec<f32> = routed.stream.memcpy_dtov(&routed.weights)?;
    anyhow::ensure!(
        ids_host
            .iter()
            .all(|&e| e >= 0 && (e as usize) < num_experts),
        "device prefill router returned an invalid expert ID"
    );
    anyhow::ensure!(
        weights_host.iter().all(|w| w.is_finite()),
        "device prefill router returned a non-finite weight"
    );
    Ok((
        ids_host.into_iter().map(|e| e as u32).collect(),
        weights_host,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn route_topk_and_plan_padded_on_device_for_prefill_dispatch(
    router_logits: &Tensor,
    selection_bias: Option<&Tensor>,
    n_tokens: usize,
    num_experts: usize,
    k: usize,
    mode: i32,
    softcap: f32,
    norm_topk: bool,
    routed_scaling: f32,
    device: &Device,
) -> Result<PrefillDeviceRouting> {
    let routed = route_topk_device(
        router_logits,
        selection_bias,
        n_tokens,
        num_experts,
        k,
        mode,
        softcap,
        norm_topk,
        routed_scaling,
        device,
    )?;
    let total = n_tokens
        .checked_mul(k)
        .context("prefill dispatch route count overflow")?;
    let padding = num_experts
        .checked_mul(MIN_TILE - 1)
        .context("prefill dispatch padding overflow")?;
    let padded_capacity = total
        .checked_add(padding)
        .context("prefill dispatch capacity overflow")?;
    anyhow::ensure!(
        padded_capacity <= i32::MAX as usize,
        "prefill dispatch capacity {padded_capacity} exceeds i32"
    );

    let mut src_idx = routed.stream.alloc_zeros::<i32>(padded_capacity)?;
    let mut expert_offsets = routed.stream.alloc_zeros::<i32>(num_experts + 2)?;
    let mut inv_perm = routed.stream.alloc_zeros::<i32>(total)?;
    let mut scratch_counts = routed.stream.alloc_zeros::<i32>(num_experts)?;
    {
        let (ids_ptr, _g1) = routed.ids.device_ptr(&routed.stream);
        let (weights_ptr, _g2) = routed.weights.device_ptr(&routed.stream);
        let (src_ptr, _g3) = src_idx.device_ptr_mut(&routed.stream);
        let (off_ptr, _g4) = expert_offsets.device_ptr_mut(&routed.stream);
        let (inv_ptr, _g5) = inv_perm.device_ptr_mut(&routed.stream);
        let (count_ptr, _g6) = scratch_counts.device_ptr_mut(&routed.stream);
        let rc = unsafe {
            nv_kernels::cuda::moe_permute_padded(
                routed.stream.cu_stream() as *mut c_void,
                ids_ptr as *const i32,
                weights_ptr as *const f32,
                src_ptr as *mut i32,
                off_ptr as *mut i32,
                inv_ptr as *mut i32,
                count_ptr as *mut i32,
                n_tokens as i32,
                k as i32,
                num_experts as i32,
                MIN_TILE as i32,
                padded_capacity as i32,
            )
        };
        anyhow::ensure!(rc == 0, "prefill moe_permute_padded rc={rc}");
    }

    #[allow(deprecated)]
    let offsets_host: Vec<i32> = routed.stream.memcpy_dtov(&expert_offsets)?;
    anyhow::ensure!(
        offsets_host.len() == num_experts + 2,
        "prefill padded dispatch returned truncated validation metadata"
    );
    let status = offsets_host[num_experts + 1];
    if status != 0 {
        let mut issues = Vec::new();
        if status & 1 != 0 {
            issues.push("invalid expert ID");
        }
        if status & 2 != 0 {
            issues.push("non-finite route weight");
        }
        if status & 4 != 0 {
            issues.push("insufficient distinct route winners");
        }
        if status & 8 != 0 {
            issues.push("padded dispatch capacity overflow");
        }
        anyhow::bail!(
            "device prefill router rejected malformed routes (status={status}: {})",
            issues.join(", ")
        );
    }
    anyhow::ensure!(
        offsets_host[0] == 0,
        "prefill padded dispatch returned malformed expert offsets"
    );
    let mut active_experts = Vec::new();
    for expert in 0..num_experts {
        let start = offsets_host[expert];
        let end = offsets_host[expert + 1];
        anyhow::ensure!(
            start >= 0 && end >= start && ((end - start) as usize).is_multiple_of(MIN_TILE),
            "prefill padded dispatch returned invalid offsets for expert {expert}: {start}..{end}"
        );
        active_experts.extend(std::iter::repeat_n(
            expert,
            (end - start) as usize / MIN_TILE,
        ));
    }
    let m_total_padded = offsets_host[num_experts] as usize;
    anyhow::ensure!(
        m_total_padded <= padded_capacity,
        "prefill padded dispatch used {m_total_padded} rows beyond capacity {padded_capacity}"
    );
    anyhow::ensure!(
        active_experts.len() * MIN_TILE == m_total_padded,
        "prefill padded dispatch tile accounting mismatch"
    );

    Ok(PrefillDeviceRouting {
        stream: routed.stream,
        _topk_ids: routed.ids,
        topk_weights: routed.weights,
        src_idx,
        inv_perm,
        active_experts,
        m_total_padded,
    })
}

pub fn forward_grouped(
    w: &MoeGroupedWeights,
    experts_concat: &MoeGroupedWeights,
    x_flat: &Tensor,
    topk_ids: &[u32],
    topk_weights: &[f32],
    n_tokens: usize,
    k: usize,
    device: &Device,
) -> Result<Tensor> {
    let _ = experts_concat;
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("grouped MoE requires CUDA device"),
    };
    let stream = dev.cuda_stream();

    let plan = plan_routing(topk_ids, n_tokens, k, w.num_experts);
    anyhow::ensure!(
        !plan.overflow,
        "forward_grouped: an expert received more than {MIN_TILE} rows -- caller must split the token range"
    );
    if plan.active_experts.is_empty() {
        return Ok(Tensor::zeros(
            (n_tokens, w.hidden_size),
            DType::F32,
            device,
        )?);
    }

    #[allow(deprecated)]
    let src_idx_dev = stream.memcpy_stod(&plan.src_idx)?;
    #[allow(deprecated)]
    let inv_perm_dev = stream.memcpy_stod(&plan.inv_perm)?;
    #[allow(deprecated)]
    let topk_weights_dev = stream.memcpy_stod(topk_weights)?;

    let out = forward_grouped_from_device_buffers(
        w,
        x_flat,
        n_tokens,
        k,
        device,
        &stream,
        &plan.active_experts,
        plan.m_total_padded,
        &src_idx_dev,
        &inv_perm_dev,
        &topk_weights_dev,
        MAX_A_PER_GROUPED_CALL,
    )?;
    let _ = plan.real_m_per_active;
    Ok(out)
}

pub fn forward_grouped_device_routing(
    w: &MoeGroupedWeights,
    experts_concat: &MoeGroupedWeights,
    x_flat: &Tensor,
    routing: &PrefillDeviceRouting,
    n_tokens: usize,
    k: usize,
    device: &Device,
) -> Result<Tensor> {
    let _ = experts_concat;
    anyhow::ensure!(
        routing.topk_weights.len() == n_tokens * k && routing.inv_perm.len() == n_tokens * k,
        "device prefill routing shape mismatch"
    );
    if routing.active_experts.is_empty() {
        return Ok(Tensor::zeros(
            (n_tokens, w.hidden_size),
            DType::F32,
            device,
        )?);
    }
    forward_grouped_from_device_buffers(
        w,
        x_flat,
        n_tokens,
        k,
        device,
        &routing.stream,
        &routing.active_experts,
        routing.m_total_padded,
        &routing.src_idx,
        &routing.inv_perm,
        &routing.topk_weights,
        prefill_tiles_per_grouped_call_default_256_user_signoff_20260912_bitwise_unset_is_256_and_8_restores_the_old_launch_width()?,
    )
}

#[allow(clippy::too_many_arguments)]
fn forward_grouped_from_device_buffers(
    w: &MoeGroupedWeights,
    x_flat: &Tensor,
    n_tokens: usize,
    k: usize,
    device: &Device,
    stream: &Arc<CudaStream>,
    active_experts: &[usize],
    m_total: usize,
    src_idx_dev: &CudaSlice<i32>,
    inv_perm_dev: &CudaSlice<i32>,
    topk_weights_dev: &CudaSlice<f32>,
    max_a_per_grouped_call: usize,
) -> Result<Tensor> {
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("grouped MoE requires CUDA device"),
    };
    let hidden = w.hidden_size;
    let inter = w.intermediate_size;
    let a = active_experts.len();
    anyhow::ensure!(
        a > 0 && m_total == a * MIN_TILE,
        "grouped MoE device plan has {a} tiles and {m_total} padded rows"
    );

    let active_ids_i32: Vec<i32> = active_experts.iter().map(|&e| e as i32).collect();
    let mut input_gu_mini = vec![0f32; a];
    let mut input_dn_mini = vec![0f32; a];
    for (i, &e) in active_experts.iter().enumerate() {
        anyhow::ensure!(e < w.num_experts, "device dispatch expert {e} out of range");
        input_gu_mini[i] = w.input_globals_gate_up_host[e];
        input_dn_mini[i] = w.input_globals_down_host[e];
    }
    #[allow(deprecated)]
    let input_gu_dev = stream.memcpy_stod(&input_gu_mini)?;
    #[allow(deprecated)]
    let input_dn_dev = stream.memcpy_stod(&input_dn_mini)?;

    let x_flat_bf = x_flat.to_dtype(DType::BF16)?.contiguous()?;
    let mut x_sorted: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(m_total * hidden)? };
    {
        let (xs, xl) = x_flat_bf.storage_and_layout();
        let x_cuda = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("x_flat must be CUDA"),
        };
        let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
        let (xp, _g1) = x_slice.device_ptr(stream);
        let xp = xp + (xl.start_offset() * std::mem::size_of::<bf16>()) as u64;
        let (sp, _g2) = src_idx_dev.device_ptr(stream);
        let (op, _g3) = x_sorted.device_ptr_mut(stream);
        let rc = unsafe {
            nv_kernels::cuda::gather_rows_bf16(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                sp as *const i32,
                op as *mut u16,
                m_total as i32,
                hidden as i32,
                n_tokens as i32,
            )
        };
        anyhow::ensure!(rc == 0, "gather_rows_bf16 rc={rc}");
    }

    let x_fp4_bytes = m_total * hidden / 2;
    let x_sf_bytes = swizzled_scale_bytes(m_total, hidden);
    let mut x_fp4: CudaSlice<u8> = unsafe { stream.alloc::<u8>(x_fp4_bytes)? };
    let mut x_sf: CudaSlice<u8> = unsafe { stream.alloc::<u8>(x_sf_bytes)? };
    {
        let (xp, _g1) = x_sorted.device_ptr(stream);
        let (fp, _g2) = x_fp4.device_ptr_mut(stream);
        let (sp, _g3) = x_sf.device_ptr_mut(stream);
        let (gp, _g4) = input_gu_dev.device_ptr(stream);
        let rc = unsafe {
            nv_kernels::cuda::quantize_nvfp4_bf16_per_expert(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                fp as *mut u8,
                sp as *mut u8,
                gp as *const f32,
                MIN_TILE as i32,
                m_total as i32,
                hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "quantize_nvfp4_bf16_per_expert (x) rc={rc}");
    }

    let mut meta_scratch: CudaSlice<u8> = unsafe { stream.alloc::<u8>(128 * 1024)? };
    let mut gemm_ws: CudaSlice<u8> = unsafe { stream.alloc::<u8>(16 * 1024 * 1024)? };

    let mut y_gate: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(m_total * inter)? };
    grouped_gemm_chunked(
        stream,
        &x_fp4,
        &x_sf,
        &w.gate_w,
        &w.gate_w_scales,
        &w.gate_alphas,
        &mut y_gate,
        &active_ids_i32,
        inter as i32,
        hidden as i32,
        w.gate_a_stride_elems,
        w.gate_b_stride_elems,
        w.gate_c_stride_elems,
        &mut meta_scratch,
        &mut gemm_ws,
        max_a_per_grouped_call,
    )?;

    let mut y_up: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(m_total * inter)? };
    grouped_gemm_chunked(
        stream,
        &x_fp4,
        &x_sf,
        &w.up_w,
        &w.up_w_scales,
        &w.up_alphas,
        &mut y_up,
        &active_ids_i32,
        inter as i32,
        hidden as i32,
        w.gate_a_stride_elems,
        w.gate_b_stride_elems,
        w.gate_c_stride_elems,
        &mut meta_scratch,
        &mut gemm_ws,
        max_a_per_grouped_call,
    )?;

    let y_act_fp4_bytes = m_total * inter / 2;
    let y_act_sf_bytes = swizzled_scale_bytes(m_total, inter);
    let mut y_act_fp4: CudaSlice<u8> = unsafe { stream.alloc::<u8>(y_act_fp4_bytes)? };
    let mut y_act_sf: CudaSlice<u8> = unsafe { stream.alloc::<u8>(y_act_sf_bytes)? };
    {
        let (gp_ptr, _g1) = y_gate.device_ptr(stream);
        let (up_ptr, _g2) = y_up.device_ptr(stream);
        let (fp, _g3) = y_act_fp4.device_ptr_mut(stream);
        let (sp, _g4) = y_act_sf.device_ptr_mut(stream);
        let (gl, _g5) = input_dn_dev.device_ptr(stream);
        let rc = unsafe {
            nv_kernels::cuda::silu_mul_quantize_nvfp4_bf16_per_expert(
                stream.cu_stream() as *mut c_void,
                gp_ptr as *const u16,
                up_ptr as *const u16,
                fp as *mut u8,
                sp as *mut u8,
                gl as *const f32,
                MIN_TILE as i32,
                m_total as i32,
                inter as i32,
            )
        };
        anyhow::ensure!(rc == 0, "silu_mul_quantize_nvfp4_bf16_per_expert rc={rc}");
    }

    let mut y_down: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(m_total * hidden)? };
    grouped_gemm_chunked(
        stream,
        &y_act_fp4,
        &y_act_sf,
        &w.down_w,
        &w.down_w_scales,
        &w.down_alphas,
        &mut y_down,
        &active_ids_i32,
        hidden as i32,
        inter as i32,
        w.down_a_stride_elems,
        w.down_b_stride_elems,
        w.down_c_stride_elems,
        &mut meta_scratch,
        &mut gemm_ws,
        max_a_per_grouped_call,
    )?;

    let mut y_acc: CudaSlice<f32> = unsafe { stream.alloc::<f32>(n_tokens * hidden)? };
    {
        let (yp, _g1) = y_down.device_ptr(stream);
        let (wp, _g2) = topk_weights_dev.device_ptr(stream);
        let (ip, _g3) = inv_perm_dev.device_ptr(stream);
        let (ap, _g4) = y_acc.device_ptr_mut(stream);
        let rc = unsafe {
            nv_kernels::cuda::moe_unpermute_scatter(
                stream.cu_stream() as *mut c_void,
                yp as *const u16,
                wp as *const f32,
                ip as *const i32,
                ap as *mut f32,
                n_tokens as i32,
                k as i32,
                hidden as i32,
                hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "moe_unpermute_scatter rc={rc}");
    }

    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_acc, dev);
    let out = candle_core::Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        (n_tokens, hidden),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok(out)
}

pub struct MarlinMoeScratch {
    sorted_ids: CudaSlice<i32>,
    expert_ids: CudaSlice<i32>,
    padded: CudaSlice<i32>,
    gate_rows: CudaSlice<bf16>,
    up_rows: CudaSlice<bf16>,
    act_rows: CudaSlice<bf16>,
    down_rows: CudaSlice<bf16>,
    c_tmp: CudaSlice<f32>,
    workspace: CudaSlice<i32>,
    rows: usize,
}

const MARLIN_MOE_ROUTE_MIN_ROWS_3_BELOW_WHICH_THE_FUSED_GEMV_DECODE_PATH_IS_FASTER: usize = 3;

fn marlin_moe_route_opt_in_again_since_20260913_peta31_measured_dflash_acceptance_halved_at_168k_prose_because_verify_rows_diverge_from_decode_rows(
) -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_MOE_MARLIN")
}

fn marlin_moe_route_opt_in() -> bool {
    marlin_moe_route_opt_in_again_since_20260913_peta31_measured_dflash_acceptance_halved_at_168k_prose_because_verify_rows_diverge_from_decode_rows()
}

pub struct GroupedDecodeContext {
    marlin_scratch_built_once_per_context: Option<MarlinMoeScratch>,
    k: usize,
    tiles_per_token: usize,
    n_tokens: usize,
    hidden: usize,
    inter: usize,
    num_experts: usize,
    src_idx: CudaSlice<i32>,
    pub inv_perm: CudaSlice<i32>,
    pub topk_ids: CudaSlice<i32>,
    pub topk_weights: CudaSlice<f32>,
    input_gu_mini: CudaSlice<f32>,
    input_dn_mini: CudaSlice<f32>,
    chunk_meta: Vec<DecodeChunkMeta>,
    x_sorted: CudaSlice<bf16>,
    x_fp4: CudaSlice<u8>,
    x_sf: CudaSlice<u8>,
    y_gate: CudaSlice<bf16>,
    y_up: CudaSlice<bf16>,
    y_act_fp4: CudaSlice<u8>,
    y_act_sf: CudaSlice<u8>,
    pub y_down: CudaSlice<bf16>,
    meta_scratch: CudaSlice<u8>,
    gemm_ws: CudaSlice<u8>,
    meta_scratch2: CudaSlice<u8>,
    gemm_ws2: CudaSlice<u8>,
    pub x_in: CudaSlice<bf16>,
    pub logits_f32: CudaSlice<f32>,
    sg_logits_bf16_written_by_the_router_gemv_arm: CudaSlice<bf16>,
    pub y_acc: CudaSlice<f32>,
    pub shared_f32: CudaSlice<f32>,
    pub out_f32: CudaSlice<f32>,
    pub resid_in: CudaSlice<bf16>,
    pub ffn_bf16: CudaSlice<bf16>,
    pub out_bf16: CudaSlice<bf16>,
    last_route_fused: bool,
    last_chain_slim: bool,
    last_chain_marlin: bool,
}

struct DecodeChunkMeta {
    expert_offsets: CudaSlice<i32>,
    sf_offsets: CudaSlice<i32>,
    problem_sizes_gate_up: CudaSlice<i32>,
    problem_sizes_down: CudaSlice<i32>,
    chunk_size: usize,
}

impl GroupedDecodeContext {
    pub fn new(
        hidden: usize,
        inter: usize,
        k: usize,
        num_experts: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        Self::new_multi(hidden, inter, k, num_experts, 1, stream)
    }

    pub fn new_folded_shared(
        hidden: usize,
        inter: usize,
        k: usize,
        num_experts: usize,
        n_tokens: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        Self::new_impl(hidden, inter, k, num_experts, n_tokens, true, stream)
    }

    pub fn new_multi(
        hidden: usize,
        inter: usize,
        k: usize,
        num_experts: usize,
        n_tokens: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        Self::new_impl(hidden, inter, k, num_experts, n_tokens, false, stream)
    }

    fn new_impl(
        hidden: usize,
        inter: usize,
        k: usize,
        num_experts: usize,
        n_tokens: usize,
        folded_shared: bool,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        anyhow::ensure!(k > 0, "GroupedDecodeContext: k must be > 0");
        anyhow::ensure!(n_tokens > 0, "GroupedDecodeContext: n_tokens must be > 0");
        anyhow::ensure!(
            num_experts > 0,
            "GroupedDecodeContext: num_experts must be > 0"
        );
        let tiles_per_token = k + folded_shared as usize;
        let tiles = n_tokens * tiles_per_token;
        anyhow::ensure!(
            tiles <= MAX_TILES_PER_GROUPED_CALL,
            "GroupedDecodeContext: {tiles} (token,expert) tiles exceeds {MAX_TILES_PER_GROUPED_CALL}"
        );
        let m_total = tiles * MIN_TILE;
        let mut src_idx_host = vec![-1i32; m_total];
        let mut inv_perm_host = vec![0i32; tiles];
        for t in 0..tiles {
            src_idx_host[t * MIN_TILE] = (t / tiles_per_token) as i32;
            inv_perm_host[t] = (t * MIN_TILE) as i32;
        }
        #[allow(deprecated)]
        let src_idx = stream.memcpy_stod(&src_idx_host)?;
        #[allow(deprecated)]
        let inv_perm = stream.memcpy_stod(&inv_perm_host)?;
        let topk_ids = stream.alloc_zeros::<i32>(tiles)?;
        let topk_weights = stream.alloc_zeros::<f32>(tiles)?;
        let input_gu_mini = stream.alloc_zeros::<f32>(tiles)?;
        let input_dn_mini = stream.alloc_zeros::<f32>(tiles)?;

        let per_call = if n_tokens == 1 {
            MAX_A_PER_GROUPED_CALL.max(tiles_per_token)
        } else {
            MAX_TILES_PER_GROUPED_CALL
        };
        let mut chunk_meta = Vec::new();
        let mut off = 0usize;
        while off < tiles {
            let chunk_size = (tiles - off).min(per_call);
            let mut expert_offsets: Vec<i32> = Vec::with_capacity(chunk_size);
            let mut sf_offsets: Vec<i32> = Vec::with_capacity(chunk_size);
            let mut ps_gu: Vec<i32> = Vec::with_capacity(chunk_size * 3);
            let mut ps_dn: Vec<i32> = Vec::with_capacity(chunk_size * 3);
            for i in 0..chunk_size {
                expert_offsets.push((i * MIN_TILE) as i32);
                sf_offsets.push((i * MIN_TILE) as i32);
                ps_gu.extend_from_slice(&[1, inter as i32, hidden as i32]);
                ps_dn.extend_from_slice(&[1, hidden as i32, inter as i32]);
            }
            #[allow(deprecated)]
            let eo = stream.memcpy_stod(&expert_offsets)?;
            #[allow(deprecated)]
            let sfo = stream.memcpy_stod(&sf_offsets)?;
            #[allow(deprecated)]
            let ps_gate_up = stream.memcpy_stod(&ps_gu)?;
            #[allow(deprecated)]
            let ps_down = stream.memcpy_stod(&ps_dn)?;
            chunk_meta.push(DecodeChunkMeta {
                expert_offsets: eo,
                sf_offsets: sfo,
                problem_sizes_gate_up: ps_gate_up,
                problem_sizes_down: ps_down,
                chunk_size,
            });
            off += chunk_size;
        }

        let x_sorted = unsafe { stream.alloc::<bf16>(m_total * hidden)? };
        let x_fp4 = unsafe { stream.alloc::<u8>(m_total * hidden / 2)? };
        let x_sf = unsafe { stream.alloc::<u8>(swizzled_scale_bytes(m_total, hidden))? };
        let y_gate = unsafe { stream.alloc::<bf16>(m_total * inter)? };
        let y_up = unsafe { stream.alloc::<bf16>(m_total * inter)? };
        let y_act_fp4 = unsafe { stream.alloc::<u8>(m_total * inter / 2)? };
        let y_act_sf = unsafe { stream.alloc::<u8>(swizzled_scale_bytes(m_total, inter))? };
        let y_down = unsafe { stream.alloc::<bf16>(m_total * hidden)? };
        let meta_scratch = unsafe { stream.alloc::<u8>(128 * 1024)? };
        let gemm_ws = unsafe { stream.alloc::<u8>(16 * 1024 * 1024)? };
        let meta_scratch2 = unsafe { stream.alloc::<u8>(128 * 1024)? };
        let gemm_ws2 = unsafe { stream.alloc::<u8>(16 * 1024 * 1024)? };
        let x_in = stream.alloc_zeros::<bf16>(n_tokens * hidden)?;
        let logits_f32 = stream.alloc_zeros::<f32>(n_tokens * num_experts)?;
        let sg_logits_bf16_written_by_the_router_gemv_arm = stream.alloc_zeros::<bf16>(n_tokens)?;
        let y_acc = stream.alloc_zeros::<f32>(n_tokens * hidden)?;
        let shared_f32 = stream.alloc_zeros::<f32>(n_tokens * hidden)?;
        let out_f32 = stream.alloc_zeros::<f32>(n_tokens * hidden)?;
        let resid_in = stream.alloc_zeros::<bf16>(n_tokens * hidden)?;
        let ffn_bf16 = stream.alloc_zeros::<bf16>(n_tokens * hidden)?;
        let out_bf16 = stream.alloc_zeros::<bf16>(n_tokens * hidden)?;

        Ok(Self {
            k,
            tiles_per_token,
            n_tokens,
            hidden,
            inter,
            num_experts,
            src_idx,
            inv_perm,
            topk_ids,
            topk_weights,
            input_gu_mini,
            input_dn_mini,
            chunk_meta,
            x_sorted,
            x_fp4,
            x_sf,
            y_gate,
            y_up,
            y_act_fp4,
            y_act_sf,
            y_down,
            meta_scratch,
            gemm_ws,
            meta_scratch2,
            gemm_ws2,
            x_in,
            logits_f32,
            sg_logits_bf16_written_by_the_router_gemv_arm,
            y_acc,
            shared_f32,
            out_f32,
            marlin_scratch_built_once_per_context: None,
            resid_in,
            ffn_bf16,
            out_bf16,
            last_route_fused: false,
            last_chain_slim: false,
            last_chain_marlin: false,
        })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn n_tokens(&self) -> usize {
        self.n_tokens
    }

    pub fn tiles_per_token(&self) -> usize {
        self.tiles_per_token
    }

    pub fn folds_shared(&self) -> bool {
        self.tiles_per_token > self.k
    }

    fn marlin_scratch(&mut self, stream: &Arc<CudaStream>) -> Result<&mut MarlinMoeScratch> {
        if self.marlin_scratch_built_once_per_context.is_none() {
            let rows = self.n_tokens * self.tiles_per_token;
            let mut ws_ints = 0i32;
            let rc =
                unsafe { nv_kernels::cuda::marlin_moe_workspace_ints(&mut ws_ints as *mut i32) };
            anyhow::ensure!(rc == 0 && ws_ints > 0, "marlin moe workspace ints rc={rc}");
            let mut c_tmp_floats = 0i64;
            let rc = unsafe {
                nv_kernels::cuda::marlin_moe_nvfp4_c_tmp_floats(&mut c_tmp_floats as *mut i64)
            };
            anyhow::ensure!(
                rc == 0 && c_tmp_floats > 0,
                "marlin moe c_tmp floats rc={rc}"
            );
            self.marlin_scratch_built_once_per_context = Some(MarlinMoeScratch {
                sorted_ids: stream.alloc_zeros::<i32>(rows * 8)?,
                expert_ids: stream.alloc_zeros::<i32>(rows)?,
                padded: stream.alloc_zeros::<i32>(1)?,
                gate_rows: stream.alloc_zeros::<bf16>(rows * 2 * self.inter)?,
                up_rows: stream.alloc_zeros::<bf16>(rows * self.inter)?,
                act_rows: stream.alloc_zeros::<bf16>(rows * self.inter)?,
                down_rows: stream.alloc_zeros::<bf16>(rows * self.hidden)?,
                c_tmp: stream.alloc_zeros::<f32>(c_tmp_floats as usize)?,
                workspace: stream.alloc_zeros::<i32>(ws_ints as usize)?,
                rows,
            });
        }
        Ok(self.marlin_scratch_built_once_per_context.as_mut().unwrap())
    }

    pub fn last_route_was_marlin(&self) -> bool {
        self.last_chain_marlin
    }

    pub fn top_k(&self) -> usize {
        self.k
    }

    pub fn last_call_used_fused_small_m_route_and_chain(&self) -> bool {
        self.last_route_fused && self.last_chain_slim
    }

    pub fn copy_scattered_f32_into_owned_tensor(
        &self,
        device: &Device,
        stream: &Arc<CudaStream>,
    ) -> Result<Tensor> {
        let dev = match device {
            Device::Cuda(dev) => dev.clone(),
            _ => anyhow::bail!("grouped routed output copy requires CUDA device"),
        };
        let elems = self.n_tokens * self.hidden;
        let mut owned: CudaSlice<f32> = unsafe { stream.alloc(elems)? };
        {
            let (src, _src_guard) = self.y_acc.device_ptr(stream);
            let (dst, _dst_guard) = owned.device_ptr_mut(stream);
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    dst,
                    src,
                    elems * std::mem::size_of::<f32>(),
                    stream.cu_stream(),
                )
            }
            .map_err(|e| anyhow::anyhow!("grouped routed output copy dtod: {e:?}"))?;
        }
        let storage = candle_core::CudaStorage::wrap_cuda_slice(owned, dev);
        Ok(Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (self.n_tokens, self.hidden),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    pub fn cast_router_logits_bf16_into_logits_f32_softcap_applied_at_route_not_here(
        &mut self,
        stream: &Arc<CudaStream>,
        router_logits_bf16_dev_ptr: u64,
    ) -> Result<()> {
        let n = self.n_tokens * self.num_experts;
        let (dp, _g) = self.logits_f32.device_ptr_mut(stream);
        let rc = unsafe {
            nv_kernels::cuda::tanh_softcap_bf16_to_f32(
                stream.cu_stream() as *mut c_void,
                router_logits_bf16_dev_ptr as *const u16,
                dp as *mut f32,
                0.0,
                n,
            )
        };
        anyhow::ensure!(rc == 0, "grouped chain router logits cast rc={rc}");
        Ok(())
    }

    pub fn stage_x_bf16_rows_dtod(
        &mut self,
        stream: &Arc<CudaStream>,
        x_bf16_dev_ptr: u64,
    ) -> Result<()> {
        let bytes = self.n_tokens * self.hidden * std::mem::size_of::<bf16>();
        let (dst, _g) = self.x_in.device_ptr_mut(stream);
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                dst,
                x_bf16_dev_ptr,
                bytes,
                stream.cu_stream(),
            )
        }
        .map_err(|e| anyhow::anyhow!("grouped chain x_in stage dtod: {e:?}"))?;
        Ok(())
    }

    pub fn fused_router_gemv_sg_and_x_stage_replacing_gate_gemm_cast_and_dtod(
        &mut self,
        stream: &Arc<CudaStream>,
        x_bf16_dev_ptr: u64,
        router_w_bf16_dev_ptr: u64,
        shared_gate_w_bf16_dev_ptr: Option<u64>,
    ) -> Result<()> {
        let m = self.n_tokens;
        anyhow::ensure!(
            m <= ROUTE_FUSE_ONE_CTA_PER_TOKEN_HOLDS_TO_8_VERIFY_ROWS,
            "router gemv arm holds to {} rows, got {m}",
            ROUTE_FUSE_ONE_CTA_PER_TOKEN_HOLDS_TO_8_VERIFY_ROWS
        );
        let (lp, _g1) = self.logits_f32.device_ptr_mut(stream);
        let (xp, _g2) = self.x_in.device_ptr_mut(stream);
        let (sp, _g3) = self
            .sg_logits_bf16_written_by_the_router_gemv_arm
            .device_ptr_mut(stream);
        let (wsg, sg_out) = match shared_gate_w_bf16_dev_ptr {
            Some(w) => (w as *const u16, sp as *mut u16),
            None => (std::ptr::null(), std::ptr::null_mut()),
        };
        let rc = unsafe {
            nv_kernels::cuda::moe_router_gemv_sg_xstage_small_m(
                stream.cu_stream() as *mut c_void,
                x_bf16_dev_ptr as *const u16,
                router_w_bf16_dev_ptr as *const u16,
                wsg,
                lp as *mut f32,
                sg_out,
                xp as *mut u16,
                m as i32,
                self.num_experts as i32,
                self.hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "moe_router_gemv_sg_xstage_small_m rc={rc}");
        Ok(())
    }

    pub fn unpermute_scatter_tail_reading_the_sg_slot_the_router_gemv_arm_wrote(
        &self,
        stream: &Arc<CudaStream>,
        out_bf16_dev_ptr: u64,
    ) -> Result<()> {
        let (sp, _g) = self
            .sg_logits_bf16_written_by_the_router_gemv_arm
            .device_ptr(stream);
        self.unpermute_scatter_tail_weighting_the_folded_shared_tile_by_sigmoid_of_its_gate_logit(
            stream,
            sp,
            out_bf16_dev_ptr,
        )
    }

    pub fn unpermute_scatter_tail_weighting_the_folded_shared_tile_by_sigmoid_of_its_gate_logit(
        &self,
        stream: &Arc<CudaStream>,
        shared_gate_logit_bf16_dev_ptr: u64,
        out_bf16_dev_ptr: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.folds_shared(),
            "sigmoid-gate scatter tail needs a folded-shared GroupedDecodeContext"
        );
        let (yp, _g1) = self.y_down.device_ptr(stream);
        let (wp, _g2) = self.topk_weights.device_ptr(stream);
        let (ip, _g3) = self.inv_perm.device_ptr(stream);
        let rc = unsafe {
            nv_kernels::cuda::moe_unpermute_scatter_shared_sig_tail(
                stream.cu_stream() as *mut c_void,
                yp as *const u16,
                wp as *const f32,
                ip as *const i32,
                shared_gate_logit_bf16_dev_ptr as *const u16,
                out_bf16_dev_ptr as *mut u16,
                self.n_tokens as i32,
                self.tiles_per_token as i32,
                self.hidden as i32,
                self.hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "grouped chain shared sig scatter tail rc={rc}");
        Ok(())
    }

    pub fn unpermute_scatter_tail_adding_shared_f32_then_resid_bf16(
        &self,
        stream: &Arc<CudaStream>,
        shared_f32_dev_ptr: u64,
        resid_bf16_dev_ptr: u64,
        out_bf16_dev_ptr: u64,
    ) -> Result<()> {
        let (yp, _g1) = self.y_down.device_ptr(stream);
        let (wp, _g2) = self.topk_weights.device_ptr(stream);
        let (ip, _g3) = self.inv_perm.device_ptr(stream);
        let rc = unsafe {
            nv_kernels::cuda::moe_unpermute_scatter_tail(
                stream.cu_stream() as *mut c_void,
                yp as *const u16,
                wp as *const f32,
                ip as *const i32,
                shared_f32_dev_ptr as *const f32,
                resid_bf16_dev_ptr as *const u16,
                out_bf16_dev_ptr as *mut u16,
                self.n_tokens as i32,
                self.tiles_per_token as i32,
                self.hidden as i32,
                self.hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "grouped chain scatter tail rc={rc}");
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn forward_grouped_decode(
    w: &MoeGroupedWeights,
    ctx: &mut GroupedDecodeContext,
    x_flat: &Tensor,
    router_logits: &Tensor,
    selection_bias: Option<&Tensor>,
    mode: i32,
    softcap: f32,
    norm_topk: bool,
    routed_scaling: f32,
    device: &Device,
) -> Result<Tensor> {
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("forward_grouped_decode requires CUDA device"),
    };
    let stream = crate::cuda_stream::current_stream(&dev);

    let prof = nv_config::env_arm_present_at_all_reread_every_call("NV_MOE_DECODE_PROF");
    let mut prof_times: Vec<(&'static str, f64)> = Vec::new();
    let mut prof_last = std::time::Instant::now();
    let prof_mark = |label: &'static str,
                     times: &mut Vec<(&'static str, f64)>,
                     last: &mut std::time::Instant| {
        if prof {
            let _ = stream.synchronize();
            times.push((label, last.elapsed().as_secs_f64() * 1e3));
            *last = std::time::Instant::now();
        }
    };
    let hidden = w.hidden_size;
    let inter = w.intermediate_size;
    anyhow::ensure!(
        ctx.hidden == hidden && ctx.inter == inter,
        "GroupedDecodeContext shape mismatch"
    );
    let k = ctx.k;
    let n_tokens = ctx.n_tokens;
    let tiles = n_tokens * ctx.tiles_per_token;
    let e = w.num_experts;
    let m_total = tiles * MIN_TILE;
    anyhow::ensure!(
        !ctx.folds_shared() || w.folded_shared,
        "GroupedDecodeContext folds the shared expert but the grouped weights were built without it"
    );

    let logits_c = router_logits.to_dtype(DType::F32)?.contiguous()?;
    anyhow::ensure!(
        logits_c.elem_count() == n_tokens * e,
        "router_logits has {} elems, expected {}",
        logits_c.elem_count(),
        n_tokens * e
    );
    let bias_c = match selection_bias {
        Some(b) => Some(b.to_dtype(DType::F32)?.contiguous()?),
        None => None,
    };
    prof_mark("prep", &mut prof_times, &mut prof_last);

    {
        let (ls, ll) = logits_c.storage_and_layout();
        let l_cuda = match &*ls {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("router_logits must be CUDA"),
        };
        let l_slice = l_cuda.as_cuda_slice::<f32>()?;
        let (lp, _g1) = l_slice.device_ptr(&stream);
        let lp = lp + (ll.start_offset() * std::mem::size_of::<f32>()) as u64;
        let bias_ptr: u64 = match &bias_c {
            Some(b) => {
                let (bs, bl) = b.storage_and_layout();
                let b_cuda = match &*bs {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("selection_bias must be CUDA"),
                };
                let b_slice = b_cuda.as_cuda_slice::<f32>()?;
                let (bp, _g) = b_slice.device_ptr(&stream);
                bp + (bl.start_offset() * std::mem::size_of::<f32>()) as u64
            }
            None => 0,
        };
        let folds = ctx.folds_shared();
        let (ip, _g2) = ctx.topk_ids.device_ptr_mut(&stream);
        let (wp, _g3) = ctx.topk_weights.device_ptr_mut(&stream);
        let rc = if folds {
            unsafe {
                nv_kernels::cuda::moe_route_topk_shared_tail(
                    stream.cu_stream() as *mut c_void,
                    lp as *const f32,
                    bias_ptr as *const f32,
                    ip as *mut i32,
                    wp as *mut f32,
                    n_tokens as i32,
                    e as i32,
                    k as i32,
                    mode,
                    softcap,
                    norm_topk as i32,
                    routed_scaling,
                    e as i32,
                )
            }
        } else {
            unsafe {
                nv_kernels::cuda::moe_route_topk(
                    stream.cu_stream() as *mut c_void,
                    lp as *const f32,
                    bias_ptr as *const f32,
                    ip as *mut i32,
                    wp as *mut f32,
                    n_tokens as i32,
                    e as i32,
                    k as i32,
                    mode,
                    softcap,
                    norm_topk as i32,
                    routed_scaling,
                )
            }
        };
        anyhow::ensure!(rc == 0, "moe_route_topk rc={rc}");
    }
    prof_mark("route", &mut prof_times, &mut prof_last);

    for (globals, mini) in [
        (&w.input_globals_gate_up, &mut ctx.input_gu_mini),
        (&w.input_globals_down, &mut ctx.input_dn_mini),
    ] {
        let (sp, _g1) = globals.device_ptr(&stream);
        let (ip, _g2) = ctx.topk_ids.device_ptr(&stream);
        let (dp, _g3) = mini.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::gather_f32_by_ids(
                stream.cu_stream() as *mut c_void,
                sp as *const f32,
                ip as *const i32,
                dp as *mut f32,
                tiles as i32,
            )
        };
        anyhow::ensure!(rc == 0, "gather_f32_by_ids rc={rc}");
    }
    prof_mark("gather_scales", &mut prof_times, &mut prof_last);

    let x_flat_bf = x_flat.to_dtype(DType::BF16)?.contiguous()?;
    {
        let (xs, xl) = x_flat_bf.storage_and_layout();
        let x_cuda = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("x_flat must be CUDA"),
        };
        let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
        let (xp, _g1) = x_slice.device_ptr(&stream);
        let xp = xp + (xl.start_offset() * std::mem::size_of::<bf16>()) as u64;
        let (sp, _g2) = ctx.src_idx.device_ptr(&stream);
        let (op, _g3) = ctx.x_sorted.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::gather_rows_bf16_strided(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                sp as *const i32,
                op as *mut u16,
                m_total as i32,
                hidden as i32,
                n_tokens as i32,
                MIN_TILE as i32,
            )
        };
        anyhow::ensure!(rc == 0, "gather_rows_bf16_strided rc={rc}");
    }
    prof_mark("gather_rows", &mut prof_times, &mut prof_last);

    {
        let (xp, _g1) = ctx.x_sorted.device_ptr(&stream);
        let (fp, _g2) = ctx.x_fp4.device_ptr_mut(&stream);
        let (sp, _g3) = ctx.x_sf.device_ptr_mut(&stream);
        let (gp, _g4) = ctx.input_gu_mini.device_ptr(&stream);
        let rc = unsafe {
            nv_kernels::cuda::quantize_nvfp4_bf16_per_expert_strided(
                stream.cu_stream() as *mut c_void,
                xp as *const u16,
                fp as *mut u8,
                sp as *mut u8,
                gp as *const f32,
                MIN_TILE as i32,
                m_total as i32,
                hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "quantize_nvfp4_bf16_per_expert (x) rc={rc}");
    }
    prof_mark("quant_x", &mut prof_times, &mut prof_last);

    grouped_gemm_decode(
        &stream,
        &ctx.x_fp4,
        &ctx.x_sf,
        &w.gate_w,
        &w.gate_w_scales,
        &w.gate_alphas,
        &mut ctx.y_gate,
        &ctx.chunk_meta,
        GemmShape::GateUp,
        &ctx.topk_ids,
        inter as i32,
        hidden as i32,
        w.gate_a_stride_elems,
        w.gate_b_stride_elems,
        w.gate_c_stride_elems,
        &mut ctx.meta_scratch,
        &mut ctx.gemm_ws,
    )?;
    prof_mark("gemm_gate", &mut prof_times, &mut prof_last);
    grouped_gemm_decode(
        &stream,
        &ctx.x_fp4,
        &ctx.x_sf,
        &w.up_w,
        &w.up_w_scales,
        &w.up_alphas,
        &mut ctx.y_up,
        &ctx.chunk_meta,
        GemmShape::GateUp,
        &ctx.topk_ids,
        inter as i32,
        hidden as i32,
        w.gate_a_stride_elems,
        w.gate_b_stride_elems,
        w.gate_c_stride_elems,
        &mut ctx.meta_scratch,
        &mut ctx.gemm_ws,
    )?;
    prof_mark("gemm_up", &mut prof_times, &mut prof_last);

    {
        let (gp_ptr, _g1) = ctx.y_gate.device_ptr(&stream);
        let (up_ptr, _g2) = ctx.y_up.device_ptr(&stream);
        let (fp, _g3) = ctx.y_act_fp4.device_ptr_mut(&stream);
        let (sp, _g4) = ctx.y_act_sf.device_ptr_mut(&stream);
        let (gl, _g5) = ctx.input_dn_mini.device_ptr(&stream);
        let rc = unsafe {
            nv_kernels::cuda::silu_mul_quantize_nvfp4_bf16_per_expert_strided(
                stream.cu_stream() as *mut c_void,
                gp_ptr as *const u16,
                up_ptr as *const u16,
                fp as *mut u8,
                sp as *mut u8,
                gl as *const f32,
                MIN_TILE as i32,
                m_total as i32,
                inter as i32,
            )
        };
        anyhow::ensure!(rc == 0, "silu_mul_quantize_nvfp4_bf16_per_expert rc={rc}");
    }
    prof_mark("silu_quant", &mut prof_times, &mut prof_last);

    grouped_gemm_decode(
        &stream,
        &ctx.y_act_fp4,
        &ctx.y_act_sf,
        &w.down_w,
        &w.down_w_scales,
        &w.down_alphas,
        &mut ctx.y_down,
        &ctx.chunk_meta,
        GemmShape::Down,
        &ctx.topk_ids,
        hidden as i32,
        inter as i32,
        w.down_a_stride_elems,
        w.down_b_stride_elems,
        w.down_c_stride_elems,
        &mut ctx.meta_scratch,
        &mut ctx.gemm_ws,
    )?;
    prof_mark("gemm_down", &mut prof_times, &mut prof_last);

    let mut y_acc: CudaSlice<f32> = unsafe { stream.alloc::<f32>(n_tokens * hidden)? };
    {
        let (yp, _g1) = ctx.y_down.device_ptr(&stream);
        let (wp, _g2) = ctx.topk_weights.device_ptr(&stream);
        let (ip, _g3) = ctx.inv_perm.device_ptr(&stream);
        let (ap, _g4) = y_acc.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::moe_unpermute_scatter(
                stream.cu_stream() as *mut c_void,
                yp as *const u16,
                wp as *const f32,
                ip as *const i32,
                ap as *mut f32,
                n_tokens as i32,
                ctx.tiles_per_token as i32,
                hidden as i32,
                hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "moe_unpermute_scatter rc={rc}");
    }
    prof_mark("scatter", &mut prof_times, &mut prof_last);
    if prof {
        let line: Vec<String> = prof_times
            .iter()
            .map(|(l, ms)| format!("{l}={ms:.3}ms"))
            .collect();
        eprintln!("[moe_decode_prof] {}", line.join(" "));
    }

    if n_tokens > 1 && nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_MOE_MT_STALL") {
        stream.synchronize().map_err(|e| {
            anyhow::anyhow!(
                "forward_grouped_decode nv_moe_mt_stall_1_diagnostic_stall_the_m_gt_1_chain_is_bit_identical_and_capture_safe_without_it_since_the_device_side_route_gather_consolidation: {e:?}"
            )
        })?;
    }

    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_acc, dev);
    let out = candle_core::Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        (n_tokens, hidden),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn forward_grouped_decode_into(
    w: &MoeGroupedWeights,
    ctx: &mut GroupedDecodeContext,
    bias_ptr: u64,
    mode: i32,
    softcap: f32,
    norm_topk: bool,
    routed_scaling: f32,
    run_scatter: bool,
    stream: &Arc<CudaStream>,
    aux_stream: Option<&Arc<CudaStream>>,
    prof_base: Option<u64>,
) -> Result<()> {
    let prof_mark = |slot: usize, s: &Arc<CudaStream>| -> Result<()> {
        if let Some(base) = prof_base {
            let rc = unsafe {
                nv_kernels::cuda::prof_timestamp(
                    s.cu_stream() as *mut c_void,
                    (base + (slot * 8) as u64) as *mut u64,
                )
            };
            anyhow::ensure!(rc == 0, "grouped prof timestamp rc={rc}");
        }
        Ok(())
    };
    let hidden = w.hidden_size;
    let inter = w.intermediate_size;
    anyhow::ensure!(
        ctx.hidden == hidden && ctx.inter == inter && ctx.num_experts == w.num_experts,
        "GroupedDecodeContext shape mismatch"
    );
    let k = ctx.k;
    let e = w.num_experts;
    let n_tokens = ctx.n_tokens;
    let tiles = n_tokens * ctx.tiles_per_token;
    let m_total = tiles * MIN_TILE;
    anyhow::ensure!(
        !ctx.folds_shared() || w.folded_shared,
        "GroupedDecodeContext folds the shared expert but the grouped weights were built without it"
    );
    let stream = stream.clone();

    let fuse_route = n_tokens <= ROUTE_FUSE_ONE_CTA_PER_TOKEN_HOLDS_TO_8_VERIFY_ROWS
        && route_fuse_enabled()
        && e <= ROUTE_FUSE_SINGLE_CTA_TOPK_HOLDS_UP_TO_256_EXPERTS;
    ctx.last_route_fused = fuse_route;
    if fuse_route {
        let shared_tail_id: i32 = if ctx.folds_shared() { e as i32 } else { -1 };
        let (lp, _g1) = ctx.logits_f32.device_ptr(&stream);
        let (xp, _g2) = ctx.x_in.device_ptr(&stream);
        let (ggu, _g3) = w.input_globals_gate_up.device_ptr(&stream);
        let (gdn, _g4) = w.input_globals_down.device_ptr(&stream);
        let (ip, _g5) = ctx.topk_ids.device_ptr_mut(&stream);
        let (wp, _g6) = ctx.topk_weights.device_ptr_mut(&stream);
        let (gm, _g7) = ctx.input_gu_mini.device_ptr_mut(&stream);
        let (dm, _g8) = ctx.input_dn_mini.device_ptr_mut(&stream);
        let (fp, _g9) = ctx.x_fp4.device_ptr_mut(&stream);
        let (sp, _g10) = ctx.x_sf.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::moe_route_gather_quant_small_m(
                stream.cu_stream() as *mut c_void,
                lp as *const f32,
                bias_ptr as *const f32,
                xp as *const u16,
                ggu as *const f32,
                gdn as *const f32,
                ip as *mut i32,
                wp as *mut f32,
                gm as *mut f32,
                dm as *mut f32,
                fp as *mut u8,
                sp as *mut u8,
                n_tokens as i32,
                e as i32,
                k as i32,
                mode,
                softcap,
                norm_topk as i32,
                routed_scaling,
                shared_tail_id,
                hidden as i32,
                MIN_TILE as i32,
                if chain_slim_launches_enabled() {
                    ROUTE_QUANT_SPLIT_CTAS_UNDER_CHAIN_SLIM
                } else {
                    1
                },
            )
        };
        anyhow::ensure!(rc == 0, "moe_route_gather_quant_small_m rc={rc}");
        prof_mark(0, &stream)?;
        prof_mark(1, &stream)?;
    } else {
        {
            let (lp, _g1) = ctx.logits_f32.device_ptr(&stream);
            let folds = ctx.folds_shared();
            let (ip, _g2) = ctx.topk_ids.device_ptr_mut(&stream);
            let (wp, _g3) = ctx.topk_weights.device_ptr_mut(&stream);
            let rc = if folds {
                unsafe {
                    nv_kernels::cuda::moe_route_topk_shared_tail(
                        stream.cu_stream() as *mut c_void,
                        lp as *const f32,
                        bias_ptr as *const f32,
                        ip as *mut i32,
                        wp as *mut f32,
                        n_tokens as i32,
                        e as i32,
                        k as i32,
                        mode,
                        softcap,
                        norm_topk as i32,
                        routed_scaling,
                        e as i32,
                    )
                }
            } else {
                unsafe {
                    nv_kernels::cuda::moe_route_topk(
                        stream.cu_stream() as *mut c_void,
                        lp as *const f32,
                        bias_ptr as *const f32,
                        ip as *mut i32,
                        wp as *mut f32,
                        n_tokens as i32,
                        e as i32,
                        k as i32,
                        mode,
                        softcap,
                        norm_topk as i32,
                        routed_scaling,
                    )
                }
            };
            anyhow::ensure!(rc == 0, "moe_route_topk rc={rc}");
        }
        prof_mark(0, &stream)?;

        for (globals, mini) in [
            (&w.input_globals_gate_up, &mut ctx.input_gu_mini),
            (&w.input_globals_down, &mut ctx.input_dn_mini),
        ] {
            let (sp, _g1) = globals.device_ptr(&stream);
            let (ip, _g2) = ctx.topk_ids.device_ptr(&stream);
            let (dp, _g3) = mini.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::gather_f32_by_ids(
                    stream.cu_stream() as *mut c_void,
                    sp as *const f32,
                    ip as *const i32,
                    dp as *mut f32,
                    tiles as i32,
                )
            };
            anyhow::ensure!(rc == 0, "gather_f32_by_ids rc={rc}");
        }

        {
            let (xp, _g1) = ctx.x_in.device_ptr(&stream);
            let (sp, _g2) = ctx.src_idx.device_ptr(&stream);
            let (op, _g3) = ctx.x_sorted.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::gather_rows_bf16_strided(
                    stream.cu_stream() as *mut c_void,
                    xp as *const u16,
                    sp as *const i32,
                    op as *mut u16,
                    m_total as i32,
                    hidden as i32,
                    n_tokens as i32,
                    MIN_TILE as i32,
                )
            };
            anyhow::ensure!(rc == 0, "gather_rows_bf16_strided rc={rc}");
        }

        {
            let (xp, _g1) = ctx.x_sorted.device_ptr(&stream);
            let (fp, _g2) = ctx.x_fp4.device_ptr_mut(&stream);
            let (sp, _g3) = ctx.x_sf.device_ptr_mut(&stream);
            let (gp, _g4) = ctx.input_gu_mini.device_ptr(&stream);
            let rc = unsafe {
                nv_kernels::cuda::quantize_nvfp4_bf16_per_expert_strided(
                    stream.cu_stream() as *mut c_void,
                    xp as *const u16,
                    fp as *mut u8,
                    sp as *mut u8,
                    gp as *const f32,
                    MIN_TILE as i32,
                    m_total as i32,
                    hidden as i32,
                )
            };
            anyhow::ensure!(
                rc == 0,
                "quantize_nvfp4_bf16_per_expert_strided (x) rc={rc}"
            );
        }
        prof_mark(1, &stream)?;
    }

    ctx.last_chain_marlin = false;
    if n_tokens >= MARLIN_MOE_ROUTE_MIN_ROWS_3_BELOW_WHICH_THE_FUSED_GEMV_DECODE_PATH_IS_FASTER
        && marlin_moe_route_opt_in()
    {
        if let Some(tw) = w.marlin_moe_twin(&stream) {
            forward_grouped_decode_marlin_moe(
                w,
                ctx,
                &tw,
                hidden,
                inter,
                n_tokens,
                run_scatter,
                &stream,
                &prof_mark,
            )?;
            ctx.last_chain_marlin = true;
            return Ok(());
        }
    }
    let slim = use_decode_gemv()
        && chain_slim_launches_enabled()
        && n_tokens <= ROUTE_FUSE_ONE_CTA_PER_TOKEN_HOLDS_TO_8_VERIFY_ROWS
        && ctx.chunk_meta.len() == 1;
    ctx.last_chain_slim = slim;
    if slim {
        let experts_total_gu = w.gate_w.len() * 2 / (inter * hidden);
        anyhow::ensure!(
            w.gate_a_stride_elems == hidden as i64 && w.gate_c_stride_elems == inter as i64,
            "slim fused gate+up+silu gemv requires contiguous rows"
        );
        let (xf, _g1) = ctx.x_fp4.device_ptr(&stream);
        let (xs, _g2) = ctx.x_sf.device_ptr(&stream);
        let (gw, _g3) = w.gate_w.device_ptr(&stream);
        let (gs, _g4) = w.gate_w_scales.device_ptr(&stream);
        let (ga, _g5) = w.gate_alphas.device_ptr(&stream);
        let (uw, _g6) = w.up_w.device_ptr(&stream);
        let (us, _g7) = w.up_w_scales.device_ptr(&stream);
        let (ua, _g8) = w.up_alphas.device_ptr(&stream);
        let (dm, _g9) = ctx.input_dn_mini.device_ptr(&stream);
        let (ip, _g10) = ctx.topk_ids.device_ptr(&stream);
        let (yf, _g11) = ctx.y_act_fp4.device_ptr_mut(&stream);
        let (ys, _g12) = ctx.y_act_sf.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::moe_grouped_fp4_gemv_gate_up_silu_quant_small_m(
                stream.cu_stream() as *mut c_void,
                xf as *const u8,
                xs as *const u8,
                gw as *const u8,
                gs as *const u8,
                ga as *const f32,
                uw as *const u8,
                us as *const u8,
                ua as *const f32,
                dm as *const f32,
                ip as *const i32,
                yf as *mut u8,
                ys as *mut u8,
                tiles as i32,
                experts_total_gu as i32,
                inter as i32,
                hidden as i32,
                MIN_TILE as i32,
            )
        };
        anyhow::ensure!(
            rc == 0,
            "moe_grouped_fp4_gemv_gate_up_silu_quant_small_m rc={rc}"
        );
        prof_mark(2, &stream)?;
        prof_mark(3, &stream)?;
    } else if let Some(s2) = aux_stream {
        let ev = stream
            .record_event(None)
            .map_err(|e| anyhow::anyhow!("fork event: {e:?}"))?;
        s2.wait(&ev)
            .map_err(|e| anyhow::anyhow!("aux wait: {e:?}"))?;
        grouped_gemm_decode(
            s2,
            &ctx.x_fp4,
            &ctx.x_sf,
            &w.up_w,
            &w.up_w_scales,
            &w.up_alphas,
            &mut ctx.y_up,
            &ctx.chunk_meta,
            GemmShape::GateUp,
            &ctx.topk_ids,
            inter as i32,
            hidden as i32,
            w.gate_a_stride_elems,
            w.gate_b_stride_elems,
            w.gate_c_stride_elems,
            &mut ctx.meta_scratch2,
            &mut ctx.gemm_ws2,
        )?;
        grouped_gemm_decode(
            &stream,
            &ctx.x_fp4,
            &ctx.x_sf,
            &w.gate_w,
            &w.gate_w_scales,
            &w.gate_alphas,
            &mut ctx.y_gate,
            &ctx.chunk_meta,
            GemmShape::GateUp,
            &ctx.topk_ids,
            inter as i32,
            hidden as i32,
            w.gate_a_stride_elems,
            w.gate_b_stride_elems,
            w.gate_c_stride_elems,
            &mut ctx.meta_scratch,
            &mut ctx.gemm_ws,
        )?;
        let ev2 = s2
            .record_event(None)
            .map_err(|e| anyhow::anyhow!("join event: {e:?}"))?;
        stream
            .wait(&ev2)
            .map_err(|e| anyhow::anyhow!("join wait: {e:?}"))?;
    } else {
        grouped_gemm_decode(
            &stream,
            &ctx.x_fp4,
            &ctx.x_sf,
            &w.gate_w,
            &w.gate_w_scales,
            &w.gate_alphas,
            &mut ctx.y_gate,
            &ctx.chunk_meta,
            GemmShape::GateUp,
            &ctx.topk_ids,
            inter as i32,
            hidden as i32,
            w.gate_a_stride_elems,
            w.gate_b_stride_elems,
            w.gate_c_stride_elems,
            &mut ctx.meta_scratch,
            &mut ctx.gemm_ws,
        )?;
        grouped_gemm_decode(
            &stream,
            &ctx.x_fp4,
            &ctx.x_sf,
            &w.up_w,
            &w.up_w_scales,
            &w.up_alphas,
            &mut ctx.y_up,
            &ctx.chunk_meta,
            GemmShape::GateUp,
            &ctx.topk_ids,
            inter as i32,
            hidden as i32,
            w.gate_a_stride_elems,
            w.gate_b_stride_elems,
            w.gate_c_stride_elems,
            &mut ctx.meta_scratch,
            &mut ctx.gemm_ws,
        )?;
    }
    if !slim {
        prof_mark(2, &stream)?;

        {
            let (gp_ptr, _g1) = ctx.y_gate.device_ptr(&stream);
            let (up_ptr, _g2) = ctx.y_up.device_ptr(&stream);
            let (fp, _g3) = ctx.y_act_fp4.device_ptr_mut(&stream);
            let (sp, _g4) = ctx.y_act_sf.device_ptr_mut(&stream);
            let (gl, _g5) = ctx.input_dn_mini.device_ptr(&stream);
            let rc = unsafe {
                nv_kernels::cuda::silu_mul_quantize_nvfp4_bf16_per_expert_strided(
                    stream.cu_stream() as *mut c_void,
                    gp_ptr as *const u16,
                    up_ptr as *const u16,
                    fp as *mut u8,
                    sp as *mut u8,
                    gl as *const f32,
                    MIN_TILE as i32,
                    m_total as i32,
                    inter as i32,
                )
            };
            anyhow::ensure!(rc == 0, "silu_mul_quantize_nvfp4_bf16_per_expert rc={rc}");
        }
        prof_mark(3, &stream)?;
    }

    grouped_gemm_decode(
        &stream,
        &ctx.y_act_fp4,
        &ctx.y_act_sf,
        &w.down_w,
        &w.down_w_scales,
        &w.down_alphas,
        &mut ctx.y_down,
        &ctx.chunk_meta,
        GemmShape::Down,
        &ctx.topk_ids,
        hidden as i32,
        inter as i32,
        w.down_a_stride_elems,
        w.down_b_stride_elems,
        w.down_c_stride_elems,
        &mut ctx.meta_scratch,
        &mut ctx.gemm_ws,
    )?;
    prof_mark(4, &stream)?;

    if run_scatter {
        let (yp, _g1) = ctx.y_down.device_ptr(&stream);
        let (wp, _g2) = ctx.topk_weights.device_ptr(&stream);
        let (ip, _g3) = ctx.inv_perm.device_ptr(&stream);
        let (ap, _g4) = ctx.y_acc.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::moe_unpermute_scatter(
                stream.cu_stream() as *mut c_void,
                yp as *const u16,
                wp as *const f32,
                ip as *const i32,
                ap as *mut f32,
                n_tokens as i32,
                ctx.tiles_per_token as i32,
                hidden as i32,
                hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "moe_unpermute_scatter rc={rc}");
    }
    prof_mark(5, &stream)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forward_grouped_decode_marlin_moe(
    w: &MoeGroupedWeights,
    ctx: &mut GroupedDecodeContext,
    tw: &MarlinMoeTwin,
    hidden: usize,
    inter: usize,
    n_tokens: usize,
    run_scatter: bool,
    stream: &Arc<CudaStream>,
    prof_mark: &dyn Fn(usize, &Arc<CudaStream>) -> Result<()>,
) -> Result<()> {
    let slots = ctx.tiles_per_token;
    let rows = n_tokens * slots;
    anyhow::ensure!(
        tw.hidden == hidden && tw.inter == inter,
        "marlin moe twin shape mismatch"
    );
    let experts_total = tw.experts_total;
    anyhow::ensure!(
        experts_total >= w.num_experts + usize::from(ctx.folds_shared()),
        "marlin moe twin has {experts_total} experts, route needs at least {}",
        w.num_experts + usize::from(ctx.folds_shared())
    );
    ctx.marlin_scratch(stream)?;
    let mut sc = ctx
        .marlin_scratch_built_once_per_context
        .take()
        .expect("marlin scratch built above");
    let result = forward_grouped_decode_marlin_moe_launches(
        ctx,
        tw,
        &mut sc,
        hidden,
        inter,
        n_tokens,
        rows,
        experts_total,
        run_scatter,
        stream,
        prof_mark,
    );
    ctx.marlin_scratch_built_once_per_context = Some(sc);
    result
}

#[allow(clippy::too_many_arguments)]
fn forward_grouped_decode_marlin_moe_launches(
    ctx: &mut GroupedDecodeContext,
    tw: &MarlinMoeTwin,
    sc: &mut MarlinMoeScratch,
    hidden: usize,
    inter: usize,
    n_tokens: usize,
    rows: usize,
    experts_total: usize,
    run_scatter: bool,
    stream: &Arc<CudaStream>,
    prof_mark: &dyn Fn(usize, &Arc<CudaStream>) -> Result<()>,
) -> Result<()> {
    let slots = ctx.tiles_per_token;
    anyhow::ensure!(sc.rows == rows, "marlin scratch rows {} != {rows}", sc.rows);
    let c_tmp_floats = sc.c_tmp.len() as i64;
    let (xin, _gx) = ctx.x_in.device_ptr(stream);
    let (ids, _gi) = ctx.topk_ids.device_ptr(stream);
    let (tkw, _gw) = ctx.topk_weights.device_ptr(stream);
    let (ivp, _gp) = ctx.inv_perm.device_ptr(stream);
    let (yacc, _ga) = ctx.y_acc.device_ptr_mut(stream);
    let (ydown, _gd) = ctx.y_down.device_ptr_mut(stream);
    let (so, _g1) = sc.sorted_ids.device_ptr_mut(stream);
    let (ex, _g2) = sc.expert_ids.device_ptr_mut(stream);
    let (pd, _g3) = sc.padded.device_ptr_mut(stream);
    let rc = unsafe {
        nv_kernels::cuda::marlin_moe_align_block8_small(
            stream.cu_stream() as *mut c_void,
            ids as *const i32,
            rows as i32,
            experts_total as i32,
            so as *mut i32,
            ex as *mut i32,
            pd as *mut i32,
        )
    };
    anyhow::ensure!(rc == 0, "marlin moe align rc={rc}");
    prof_mark(1, stream)?;
    let (ct, _g4) = sc.c_tmp.device_ptr_mut(stream);
    let (ws, _g5) = sc.workspace.device_ptr_mut(stream);
    let gemm = |st: &MarlinMoeStack,
                a: u64,
                c: u64,
                size_m: usize,
                top_k: usize,
                mul: i32,
                n: usize,
                k: usize,
                what: &str|
     -> Result<()> {
        let (pb, _h1) = st.b_q.device_ptr(stream);
        let (ps, _h2) = st.scales.device_ptr(stream);
        let (pg, _h3) = st.globals.device_ptr(stream);
        let rc = unsafe {
            nv_kernels::cuda::marlin_moe_nvfp4_w4a16(
                stream.cu_stream() as *mut c_void,
                a as *const c_void,
                pb as *const c_void,
                ps as *const c_void,
                pg as *const f32,
                c as *mut c_void,
                ct as *mut c_void,
                c_tmp_floats,
                ws as *mut c_void,
                so as *const i32,
                ex as *const i32,
                pd as *const i32,
                tkw as *const f32,
                experts_total as i32,
                top_k as i32,
                mul,
                size_m as i32,
                n as i32,
                k as i32,
            )
        };
        anyhow::ensure!(rc == 0, "marlin moe {what} rc={rc}");
        Ok(())
    };
    let (gr, _g6) = sc.gate_rows.device_ptr_mut(stream);
    let (ur, _g7) = sc.up_rows.device_ptr_mut(stream);
    let (ar, _g8) = sc.act_rows.device_ptr_mut(stream);
    let (dr, _g9) = sc.down_rows.device_ptr_mut(stream);
    let rc = if let Some(cat) = tw
        .gate_up_concat_when_every_expert_shares_one_global_scale
        .as_ref()
    {
        gemm(
            cat,
            xin,
            gr,
            n_tokens,
            slots,
            0,
            2 * inter,
            hidden,
            "gate_up",
        )?;
        prof_mark(2, stream)?;
        unsafe {
            nv_kernels::cuda::marlin_moe_silu_mul_halves(
                stream.cu_stream() as *mut c_void,
                gr as *const u16,
                ar as *mut u16,
                rows as i32,
                inter as i32,
            )
        }
    } else {
        gemm(&tw.gate, xin, gr, n_tokens, slots, 0, inter, hidden, "gate")?;
        gemm(&tw.up, xin, ur, n_tokens, slots, 0, inter, hidden, "up")?;
        prof_mark(2, stream)?;
        unsafe {
            nv_kernels::cuda::silu_mul_bf16_candle(
                stream.cu_stream() as *mut c_void,
                gr as *const u16,
                ur as *const u16,
                ar as *mut u16,
                rows * inter,
            )
        }
    };
    anyhow::ensure!(rc == 0, "marlin moe silu_mul rc={rc}");
    prof_mark(3, stream)?;
    gemm(
        &tw.down,
        ar,
        dr,
        rows,
        1,
        i32::from(run_scatter),
        hidden,
        inter,
        "down",
    )?;
    prof_mark(4, stream)?;
    let rc = if run_scatter {
        unsafe {
            nv_kernels::cuda::marlin_moe_sum_slots_into_f32(
                stream.cu_stream() as *mut c_void,
                dr as *const u16,
                yacc as *mut f32,
                n_tokens as i32,
                slots as i32,
                hidden as i32,
            )
        }
    } else {
        unsafe {
            nv_kernels::cuda::marlin_moe_place_rows_by_inv_perm(
                stream.cu_stream() as *mut c_void,
                dr as *const u16,
                ivp as *const i32,
                ydown as *mut u16,
                rows as i32,
                hidden as i32,
            )
        }
    };
    anyhow::ensure!(
        rc == 0,
        "marlin moe tail placement rc={rc} run_scatter={run_scatter}"
    );
    prof_mark(5, stream)?;
    Ok(())
}

#[derive(Clone, Copy)]
enum GemmShape {
    GateUp,
    Down,
}

const DECODE_TILE_CTA_THRESHOLD: usize = 96;

fn use_decode_tile(n: i32, num_experts: usize) -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NV_MOE_FP4_DECODE_TILE").ok().as_deref(),
            Some("1") | Some("on") | Some("true") | Some("yes")
        )
    });
    if !enabled {
        return false;
    }
    let ctas = num_experts.saturating_mul((n.max(0) as usize).div_ceil(128));
    ctas < DECODE_TILE_CTA_THRESHOLD
}

fn use_decode_gemv() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        moe_arm_on_unless_the_key_is_explicitly_off_which_is_how_the_pre_round_82_path_is_reached(
            "NV_MOE_FP4_DECODE_GEMV",
        )
    })
}

fn grouped_gemm_decode(
    stream: &Arc<CudaStream>,
    a_packed: &CudaSlice<u8>,
    a_scales: &CudaSlice<u8>,
    b_packed: &CudaSlice<u8>,
    b_scales: &CudaSlice<u8>,
    alphas: &CudaSlice<f32>,
    d_bf16: &mut CudaSlice<bf16>,
    chunk_meta: &[DecodeChunkMeta],
    shape: GemmShape,
    topk_ids: &CudaSlice<i32>,
    n: i32,
    k_dim: i32,
    a_row_stride_elems: i64,
    b_row_stride_elems: i64,
    c_row_stride_elems: i64,
    meta_scratch: &mut CudaSlice<u8>,
    gemm_ws: &mut CudaSlice<u8>,
) -> Result<()> {
    let n_us = n as usize;
    let k_us = k_dim as usize;
    let half_k = k_us / 2;
    let group_k = k_us / BLOCK_SIZE;

    let ms_bytes = meta_scratch.len();
    let ws_bytes = gemm_ws.len();
    let (a_base, _ga) = a_packed.device_ptr(stream);
    let (a_sf_base, _gas) = a_scales.device_ptr(stream);
    let (b_base, _gb) = b_packed.device_ptr(stream);
    let (b_sf_base, _gbs) = b_scales.device_ptr(stream);
    let (al_base, _gal) = alphas.device_ptr(stream);
    let (ids_base, _gi) = topk_ids.device_ptr(stream);
    let (d_base, _gd) = d_bf16.device_ptr_mut(stream);
    let (ms_ptr, _gm) = meta_scratch.device_ptr_mut(stream);
    let (ws_ptr, _gw) = gemm_ws.device_ptr_mut(stream);

    let mut chunk_start = 0usize;
    for meta in chunk_meta {
        let chunk_start_rows = chunk_start * MIN_TILE;
        let a_chunk_ptr = a_base + (chunk_start_rows * half_k) as u64;
        let a_sf_chunk_ptr = a_sf_base + (chunk_start_rows * group_k) as u64;
        let d_chunk_ptr = d_base + (chunk_start_rows * n_us * 2) as u64;
        let aid_ptr = ids_base + (chunk_start * std::mem::size_of::<i32>()) as u64;

        if use_decode_gemv() {
            anyhow::ensure!(
                group_k.is_multiple_of(4),
                "decode GEMV requires K/16 divisible by 4 (K={k_dim})"
            );
            anyhow::ensure!(
                a_row_stride_elems == k_dim as i64
                    && b_row_stride_elems == k_dim as i64
                    && c_row_stride_elems == n as i64,
                "decode GEMV requires contiguous rows (a={a_row_stride_elems} b={b_row_stride_elems} c={c_row_stride_elems} n={n} k={k_dim})"
            );
            let num_experts_total = b_packed.len() * 2 / (n_us * k_us);
            let rc = unsafe {
                nv_kernels::cuda::moe_grouped_fp4_gemv_m1_bf16(
                    stream.cu_stream() as *mut c_void,
                    a_chunk_ptr as *const u8,
                    a_sf_chunk_ptr as *const u8,
                    b_base as *const u8,
                    b_sf_base as *const u8,
                    al_base as *const f32,
                    d_chunk_ptr as *mut u16,
                    aid_ptr as *const i32,
                    meta.chunk_size as i32,
                    num_experts_total as i32,
                    n,
                    k_dim,
                    MIN_TILE as i32,
                    (MIN_TILE * n_us) as i64,
                )
            };
            anyhow::ensure!(rc == 0, "moe_grouped_fp4_gemv_m1_bf16 rc={rc}");
            chunk_start += meta.chunk_size;
            continue;
        }

        let ps_dev = match shape {
            GemmShape::GateUp => &meta.problem_sizes_gate_up,
            GemmShape::Down => &meta.problem_sizes_down,
        };
        let (eo, _g1) = meta.expert_offsets.device_ptr(stream);
        let (sfo, _g2) = meta.sf_offsets.device_ptr(stream);
        let (ps, _g3) = ps_dev.device_ptr(stream);

        let gemm_fn = if use_decode_tile(n, meta.chunk_size) {
            nv_kernels::cuda::cutlass_moe_grouped_fp4_gemm_sm120_bf16_decode
        } else {
            nv_kernels::cuda::cutlass_moe_grouped_fp4_gemm_sm120_bf16
        };

        unsafe {
            gemm_fn(
                stream.cu_stream() as *mut c_void,
                a_chunk_ptr as *const c_void,
                a_sf_chunk_ptr as *const c_void,
                b_base as *const c_void,
                b_sf_base as *const c_void,
                al_base as *const f32,
                d_chunk_ptr as *mut c_void,
                eo as *const i32,
                sfo as *const i32,
                ps as *const i32,
                aid_ptr as *const i32,
                n,
                k_dim,
                meta.chunk_size as i32,
                a_row_stride_elems,
                b_row_stride_elems,
                c_row_stride_elems,
                ms_ptr as *mut c_void,
                ms_bytes,
                ws_ptr as *mut c_void,
                ws_bytes,
            )
            .map_err(|rc| anyhow::anyhow!("decode grouped FP4 GEMM rc={rc}"))?
        };
        chunk_start += meta.chunk_size;
    }
    Ok(())
}

const MAX_A_PER_GROUPED_CALL: usize = 8;
const PREFILL_TILES_PER_GROUPED_CALL_MAX_1024_IS_THE_STARTS_KERNEL_BLOCK_DIM: usize = 1024;
const PREFILL_TILES_PER_GROUPED_CALL_DEFAULT_256_ONE_LAUNCH_PER_GEMM_STAGE: usize = 256;

fn prefill_tiles_per_grouped_call_default_256_user_signoff_20260912_bitwise_unset_is_256_and_8_restores_the_old_launch_width(
) -> Result<usize> {
    match std::env::var("NV_LAGUNA_PREFILL_GROUP") {
        Ok(v) => {
            let n: usize = v.trim().parse().map_err(|_| {
                anyhow::anyhow!("NV_LAGUNA_PREFILL_GROUP must be an integer tile count, got {v:?}")
            })?;
            anyhow::ensure!(
                (1..=PREFILL_TILES_PER_GROUPED_CALL_MAX_1024_IS_THE_STARTS_KERNEL_BLOCK_DIM)
                    .contains(&n),
                "NV_LAGUNA_PREFILL_GROUP must be 1..=1024, got {n}"
            );
            Ok(n)
        }
        Err(std::env::VarError::NotPresent) => Ok(
            if nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_PREFILL_GROUP32") {
                32
            } else {
                PREFILL_TILES_PER_GROUPED_CALL_DEFAULT_256_ONE_LAUNCH_PER_GEMM_STAGE
            },
        ),
        Err(e) => Err(e).context("NV_LAGUNA_PREFILL_GROUP must be valid UTF-8"),
    }
}

const MAX_TILES_PER_GROUPED_CALL: usize = 256;

fn grouped_gemm_chunked(
    stream: &Arc<CudaStream>,
    a_packed: &CudaSlice<u8>,
    a_scales: &CudaSlice<u8>,
    b_packed: &CudaSlice<u8>,
    b_scales: &CudaSlice<u8>,
    alphas: &CudaSlice<f32>,
    d_bf16: &mut CudaSlice<bf16>,
    active_ids_host: &[i32],
    n: i32,
    k: i32,
    a_row_stride_elems: i64,
    b_row_stride_elems: i64,
    c_row_stride_elems: i64,
    meta_scratch: &mut CudaSlice<u8>,
    gemm_ws: &mut CudaSlice<u8>,
    max_a_per_grouped_call: usize,
) -> Result<()> {
    let total_a = active_ids_host.len();
    if total_a == 0 {
        return Ok(());
    }
    let min_tile = MIN_TILE;
    let n_us = n as usize;
    let k_us = k as usize;
    let group_size = BLOCK_SIZE;
    let half_k = k_us / 2;
    let group_k = k_us / group_size;
    let group_k_pad = group_k.div_ceil(4) * 4;

    let ms_bytes = meta_scratch.len();
    let ws_bytes = gemm_ws.len();
    let (a_base, _ga) = a_packed.device_ptr(stream);
    let (a_sf_base, _gas) = a_scales.device_ptr(stream);
    let (b_base, _gb) = b_packed.device_ptr(stream);
    let (b_sf_base, _gbs) = b_scales.device_ptr(stream);
    let (al_base, _gal) = alphas.device_ptr(stream);
    let (d_base, _gd) = d_bf16.device_ptr_mut(stream);
    let (ms_ptr, _gm) = meta_scratch.device_ptr_mut(stream);
    let (ws_ptr, _gw) = gemm_ws.device_ptr_mut(stream);

    for (chunk_idx, chunk) in active_ids_host.chunks(max_a_per_grouped_call).enumerate() {
        let chunk_size = chunk.len();
        let chunk_start_rows = chunk_idx * max_a_per_grouped_call * min_tile;

        let a_chunk_ptr = a_base + (chunk_start_rows * half_k) as u64;

        let a_sf_chunk_ptr = a_sf_base + (chunk_start_rows * group_k_pad) as u64;
        let d_chunk_ptr = d_base + (chunk_start_rows * n_us * 2) as u64;

        let mut expert_offsets: Vec<i32> = Vec::with_capacity(chunk_size);
        let mut sf_offsets: Vec<i32> = Vec::with_capacity(chunk_size);
        let mut problem_sizes: Vec<i32> = Vec::with_capacity(chunk_size * 3);
        for i in 0..chunk_size {
            expert_offsets.push((i * min_tile) as i32);
            sf_offsets.push((i * min_tile) as i32);
            problem_sizes.extend_from_slice(&[min_tile as i32, n, k]);
        }
        #[allow(deprecated)]
        let eo_dev = stream.memcpy_stod(&expert_offsets)?;
        #[allow(deprecated)]
        let sfo_dev = stream.memcpy_stod(&sf_offsets)?;
        #[allow(deprecated)]
        let ps_dev = stream.memcpy_stod(&problem_sizes)?;
        #[allow(deprecated)]
        let aid_dev = stream.memcpy_stod(chunk)?;

        let (eo, _g1) = eo_dev.device_ptr(stream);
        let (sfo, _g2) = sfo_dev.device_ptr(stream);
        let (ps, _g3) = ps_dev.device_ptr(stream);
        let (aid, _g4) = aid_dev.device_ptr(stream);

        let _ = unsafe {
            nv_kernels::cuda::cutlass_moe_grouped_fp4_gemm_sm120_bf16(
                stream.cu_stream() as *mut c_void,
                a_chunk_ptr as *const c_void,
                a_sf_chunk_ptr as *const c_void,
                b_base as *const c_void,
                b_sf_base as *const c_void,
                al_base as *const f32,
                d_chunk_ptr as *mut c_void,
                eo as *const i32,
                sfo as *const i32,
                ps as *const i32,
                aid as *const i32,
                n,
                k,
                chunk_size as i32,
                a_row_stride_elems,
                b_row_stride_elems,
                c_row_stride_elems,
                ms_ptr as *mut c_void,
                ms_bytes,
                ws_ptr as *mut c_void,
                ws_bytes,
            )
            .map_err(|rc| {
                anyhow::anyhow!(
                    "chunked grouped FP4 GEMM rc={rc} (chunk {chunk_idx} of {})",
                    total_a.div_ceil(max_a_per_grouped_call)
                )
            })?
        };
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod prefill_device_routing_tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA; directly proves copied routed F32 output remains owned after persistent scratch reuse"]
    fn copied_scattered_f32_output_survives_context_scratch_overwrite() {
        let device = Device::new_cuda(0).expect("CUDA");
        let dev = match &device {
            Device::Cuda(dev) => dev.clone(),
            _ => unreachable!(),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let mut ctx = GroupedDecodeContext::new_multi(128, 128, 1, 1, 2, &stream).unwrap();
        let first: Vec<f32> = (0..256).map(|i| i as f32 * 0.25 - 7.0).collect();
        stream.memcpy_htod(&first, &mut ctx.y_acc).unwrap();
        let owned = ctx
            .copy_scattered_f32_into_owned_tensor(&device, &stream)
            .unwrap();
        let second: Vec<f32> = (0..256).map(|i| 1000.0 - i as f32).collect();
        stream.memcpy_htod(&second, &mut ctx.y_acc).unwrap();
        stream.synchronize().unwrap();
        let retained = owned.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            retained.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            first.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        eprintln!(
            "LAGUNA-B2-OWNED-OUTPUT-GATE copied_y_acc=bitwise retained_after_scratch_overwrite=bitwise"
        );
    }

    #[test]
    fn malformed_gpu_router_rows_fail_safely_and_recover_same_context() {
        let device = Device::new_cuda(0).expect("CUDA");
        let num_experts = 8usize;
        let k = 4usize;

        let route = |values: Vec<f32>, routed_scaling: f32| {
            let logits = Tensor::from_vec(values, (1, num_experts), &device).unwrap();
            route_topk_and_plan_padded_on_device_for_prefill_dispatch(
                &logits,
                None,
                1,
                num_experts,
                k,
                1,
                0.0,
                true,
                routed_scaling,
                &device,
            )
        };
        let valid_logits = || (0..num_experts).map(|i| i as f32 - 4.0).collect();

        let invalid_cases = [
            ("all_nonfinite_winners", vec![f32::NAN; num_experts], 1.0),
            (
                "insufficient_finite_winners",
                vec![
                    0.0,
                    f32::NAN,
                    f32::NAN,
                    f32::NAN,
                    f32::NAN,
                    f32::NAN,
                    f32::NAN,
                    f32::NAN,
                ],
                1.0,
            ),
            ("nonfinite_route_weights", valid_logits(), f32::INFINITY),
        ];

        for (name, logits, scaling) in invalid_cases {
            let err = route(logits, scaling)
                .err()
                .unwrap_or_else(|| panic!("{name} unexpectedly produced a dispatch plan"));
            let message = format!("{err:#}");
            assert!(
                message.contains("device prefill router rejected malformed routes"),
                "{name} returned the wrong error: {message}"
            );

            let recovery = route(valid_logits(), 1.0)
                .unwrap_or_else(|e| panic!("valid route after {name} did not recover: {e:#}"));
            assert_eq!(recovery.topk_weights.len(), k);
            assert_eq!(recovery.inv_perm.len(), k);
            assert_eq!(recovery.m_total_padded, k * MIN_TILE);
        }
    }
}
