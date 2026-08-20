#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CoopScalar {
    F32,
    F16,
    I32,
    U32,
}

impl CoopScalar {
    pub fn wgsl(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::I32 => "i32",
            Self::U32 => "u32",
        }
    }

    pub fn from_wgsl(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "i32" => Some(Self::I32),
            "u32" => Some(Self::U32),
            _ => None,
        }
    }

    pub fn from_wgpu(t: wgpu::CooperativeScalarType) -> Self {
        match t {
            wgpu::CooperativeScalarType::F32 => Self::F32,
            wgpu::CooperativeScalarType::F16 => Self::F16,
            wgpu::CooperativeScalarType::I32 => Self::I32,
            wgpu::CooperativeScalarType::U32 => Self::U32,
        }
    }
}

impl std::fmt::Display for CoopScalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.wgsl())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoopConfig {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub ab: CoopScalar,
    pub cr: CoopScalar,
}

impl CoopConfig {
    pub fn new(m: u32, n: u32, k: u32, ab: CoopScalar, cr: CoopScalar) -> Self {
        Self { m, n, k, ab, cr }
    }

    pub fn ab_f16(&self) -> bool {
        self.ab == CoopScalar::F16
    }

    pub fn cr_f32(&self) -> bool {
        self.cr == CoopScalar::F32
    }

    pub fn label(&self) -> String {
        format!(
            "{}x{}x{} {}x{}->{}",
            self.m, self.n, self.k, self.ab, self.ab, self.cr
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoopRequest {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub ab: CoopScalar,
    pub cr: CoopScalar,
}

impl CoopRequest {
    pub fn new(m: u32, n: u32, k: u32, ab: CoopScalar, cr: CoopScalar) -> Self {
        Self { m, n, k, ab, cr }
    }

    pub fn square(tile: u32, ab: CoopScalar, cr: CoopScalar) -> Self {
        Self::new(tile, tile, tile, ab, cr)
    }

    pub fn label(&self) -> String {
        format!(
            "{}x{}x{} {}x{}->{}",
            self.m, self.n, self.k, self.ab, self.ab, self.cr
        )
    }

    pub fn matches(&self, cfg: &CoopConfig) -> bool {
        self.m == cfg.m
            && self.n == cfg.n
            && self.k == cfg.k
            && self.ab == cfg.ab
            && self.cr == cfg.cr
    }

    pub fn is_advertised(&self, advertised: &[CoopConfig]) -> bool {
        advertised.iter().any(|c| self.matches(c))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoopTile {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

pub const COOP_SUBGROUP_SIZE: u32 = 32;

pub const COOP_UNSAFE_SWEEP_ENV: &str = "NV_KERNELS_WGPU_COOP_UNSAFE_SWEEP";

pub fn coop_unsafe_sweep_enabled() -> bool {
    matches!(
        std::env::var(COOP_UNSAFE_SWEEP_ENV)
            .ok()
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoopDecision {
    Compile,

    Skip(String),

    CompileUnadvertised(String),
}

impl CoopDecision {
    pub fn should_compile(&self) -> bool {
        !matches!(self, Self::Skip(_))
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Compile => None,
            Self::Skip(r) | Self::CompileUnadvertised(r) => Some(r.as_str()),
        }
    }
}

pub fn coop_skip_reason(req: &CoopRequest, advertised: &[CoopConfig]) -> Option<String> {
    if req.is_advertised(advertised) {
        return None;
    }
    if advertised.is_empty() {
        return Some(format!(
            "{} is unadvertised: the adapter reports no cooperative-matrix configurations at all",
            req.label()
        ));
    }
    let have: Vec<String> = advertised.iter().map(|c| c.label()).collect();
    Some(format!(
        "{} is not among the {} configuration(s) the adapter advertises [{}]",
        req.label(),
        advertised.len(),
        have.join(", ")
    ))
}

pub fn coop_decide(
    req: &CoopRequest,
    advertised: &[CoopConfig],
    allow_unadvertised: bool,
) -> CoopDecision {
    match coop_skip_reason(req, advertised) {
        None => CoopDecision::Compile,
        Some(why) if allow_unadvertised => CoopDecision::CompileUnadvertised(why),
        Some(why) => CoopDecision::Skip(why),
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoopPartition {
    pub compile: Vec<CoopRequest>,
    pub skipped: Vec<(CoopRequest, String)>,
    pub forced: Vec<(CoopRequest, String)>,
}

pub fn coop_partition(
    requests: &[CoopRequest],
    advertised: &[CoopConfig],
    allow_unadvertised: bool,
) -> CoopPartition {
    let mut out = CoopPartition::default();
    for req in requests {
        match coop_decide(req, advertised, allow_unadvertised) {
            CoopDecision::Compile => out.compile.push(*req),
            CoopDecision::CompileUnadvertised(why) => {
                out.compile.push(*req);
                out.forced.push((*req, why));
            }
            CoopDecision::Skip(why) => out.skipped.push((*req, why)),
        }
    }
    out
}

pub fn coop_square_sweep(tiles: &[u32], combos: &[(CoopScalar, CoopScalar)]) -> Vec<CoopRequest> {
    let mut out = Vec::with_capacity(tiles.len() * combos.len());
    for tile in tiles {
        for (ab, cr) in combos {
            out.push(CoopRequest::square(*tile, *ab, *cr));
        }
    }
    out
}

pub fn coop_configs(props: &[wgpu::CooperativeMatrixProperties]) -> Vec<CoopConfig> {
    props
        .iter()
        .map(|p| CoopConfig {
            m: p.m_size,
            n: p.n_size,
            k: p.k_size,
            ab: CoopScalar::from_wgpu(p.ab_type),
            cr: CoopScalar::from_wgpu(p.cr_type),
        })
        .collect()
}

fn coop_type_uses(src: &str) -> Vec<(u32, u32, CoopScalar, char)> {
    const NEEDLE: &str = "coop_mat";
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(rel) = src[from..].find(NEEDLE) {
        let start = from + rel;
        from = start + NEEDLE.len();
        let mut i = from;
        let take_num = |i: &mut usize| -> Option<u32> {
            let s = *i;
            while *i < bytes.len() && bytes[*i].is_ascii_digit() {
                *i += 1;
            }
            if *i == s {
                return None;
            }
            src[s..*i].parse::<u32>().ok()
        };
        let skip_ws = |i: &mut usize| {
            while *i < bytes.len() && (bytes[*i] as char).is_whitespace() {
                *i += 1;
            }
        };
        let Some(rows) = take_num(&mut i) else {
            continue;
        };
        if i >= bytes.len() || bytes[i] != b'x' {
            continue;
        }
        i += 1;
        let Some(cols) = take_num(&mut i) else {
            continue;
        };
        skip_ws(&mut i);
        if i >= bytes.len() || bytes[i] != b'<' {
            continue;
        }
        i += 1;
        skip_ws(&mut i);
        let ts = i;
        while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
            i += 1;
        }
        let Some(scalar) = CoopScalar::from_wgsl(&src[ts..i]) else {
            continue;
        };
        skip_ws(&mut i);
        if i >= bytes.len() || bytes[i] != b',' {
            continue;
        }
        i += 1;
        skip_ws(&mut i);
        if i >= bytes.len() {
            continue;
        }
        let usage = bytes[i] as char;
        if !matches!(usage, 'A' | 'B' | 'C') {
            continue;
        }
        i += 1;
        skip_ws(&mut i);
        if i >= bytes.len() || bytes[i] != b'>' {
            continue;
        }
        out.push((rows, cols, scalar, usage));
    }
    out
}

pub fn coop_requests_in_wgsl(src: &str) -> Vec<CoopRequest> {
    let uses = coop_type_uses(src);
    let mut out: Vec<CoopRequest> = Vec::new();
    for (am, ak, at, au) in uses.iter().copied().filter(|u| u.3 == 'A') {
        for (bk, bn, bt, _) in uses.iter().copied().filter(|u| u.3 == 'B') {
            if ak != bk || at != bt {
                continue;
            }
            for (cm, cn, ct, _) in uses.iter().copied().filter(|u| u.3 == 'C') {
                if cm != am || cn != bn {
                    continue;
                }
                let _ = au;
                let req = CoopRequest::new(am, bn, ak, at, ct);
                if !out.contains(&req) {
                    out.push(req);
                }
            }
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    pub adapter_name: String,
    pub backend: String,
    pub device_type: String,
    pub driver: String,
    pub shader_f16: bool,
    pub f16_in_f32: bool,
    pub subgroup: bool,
    pub timestamp_query: bool,
    pub subgroup_min_size: u32,
    pub subgroup_max_size: u32,
    pub subgroup_runtime_width: Option<u32>,
    pub cooperative_matrix: bool,
    pub coop_configs: Vec<CoopConfig>,
    pub coop_note: Option<String>,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub max_storage_buffers_per_shader_stage: u32,
}

impl Capabilities {
    pub fn probe(
        info: &wgpu::AdapterInfo,
        features: wgpu::Features,
        limits: &wgpu::Limits,
        downlevel: &wgpu::DownlevelCapabilities,
    ) -> Self {
        Self {
            adapter_name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            driver: format!("{} {}", info.driver, info.driver_info),
            shader_f16: features.contains(wgpu::Features::SHADER_F16),
            f16_in_f32: downlevel
                .flags
                .contains(wgpu::DownlevelFlags::SHADER_F16_IN_F32),
            subgroup: features.contains(wgpu::Features::SUBGROUP),
            timestamp_query: features.contains(wgpu::Features::TIMESTAMP_QUERY),
            subgroup_min_size: info.subgroup_min_size,
            subgroup_max_size: info.subgroup_max_size,
            subgroup_runtime_width: None,
            cooperative_matrix: features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX),
            coop_configs: Vec::new(),
            coop_note: None,
            max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
        }
    }

    pub fn reduction_strategy(&self) -> ReductionStrategy {
        if self.subgroup && self.subgroup_min_size >= 4 {
            ReductionStrategy::Subgroup
        } else {
            ReductionStrategy::WorkgroupTree
        }
    }

    pub fn gemm_strategy(&self) -> GemmStrategy {
        if self.cooperative_matrix {
            GemmStrategy::CoopMat
        } else {
            GemmStrategy::Scalar
        }
    }

    pub fn subgroup_width_known(&self) -> Option<u32> {
        if !self.subgroup {
            return None;
        }
        self.subgroup_runtime_width
            .or(if self.subgroup_min_size == self.subgroup_max_size {
                Some(self.subgroup_min_size)
            } else {
                None
            })
    }

    pub fn subgroup32_reason(&self) -> Option<String> {
        if self.subgroup_width_known() == Some(COOP_SUBGROUP_SIZE) {
            return None;
        }
        if self.subgroup
            && self.subgroup_min_size <= COOP_SUBGROUP_SIZE
            && COOP_SUBGROUP_SIZE <= self.subgroup_max_size
        {
            return Some(format!(
                "adapter advertises {}..{} but compute ran at {:?}, need {COOP_SUBGROUP_SIZE}",
                self.subgroup_min_size, self.subgroup_max_size, self.subgroup_runtime_width
            ));
        }
        Some(format!(
            "subgroup size {}..{} is not exactly {COOP_SUBGROUP_SIZE}",
            self.subgroup_min_size, self.subgroup_max_size
        ))
    }

    pub fn coop_gemm_tile(&self) -> Option<CoopTile> {
        if !self.cooperative_matrix || !self.shader_f16 {
            return None;
        }
        if self.subgroup_width_known() != Some(COOP_SUBGROUP_SIZE) {
            return None;
        }
        self.coop_configs
            .iter()
            .find(|c| c.m == 16 && c.n == 16 && c.k == 16 && c.ab_f16() && c.cr_f32())
            .map(|c| CoopTile {
                m: c.m,
                n: c.n,
                k: c.k,
            })
    }

    pub fn coop_advertises(&self, req: &CoopRequest) -> bool {
        req.is_advertised(&self.coop_configs)
    }

    pub fn coop_skip_reason(&self, req: &CoopRequest) -> Option<String> {
        coop_skip_reason(req, &self.coop_configs)
    }

    pub fn coop_decision(&self, req: &CoopRequest) -> CoopDecision {
        coop_decide(req, &self.coop_configs, coop_unsafe_sweep_enabled())
    }

    pub fn coop_gemm_reason(&self) -> Option<String> {
        if !self.cooperative_matrix {
            return Some(match &self.coop_note {
                Some(note) => {
                    format!("device does not expose EXPERIMENTAL_COOPERATIVE_MATRIX: {note}")
                }
                None => "device does not expose EXPERIMENTAL_COOPERATIVE_MATRIX".to_string(),
            });
        }
        if !self.shader_f16 {
            return Some("device does not expose SHADER_F16".to_string());
        }
        if let Some(why) = self.subgroup32_reason() {
            return Some(why);
        }
        if self.coop_gemm_tile().is_none() {
            return Some(format!(
                "no 16x16x16 f16xf16->f32 config among {} reported",
                self.coop_configs.len()
            ));
        }
        None
    }

    pub fn accum_dtype(&self) -> AccumDtype {
        if self.shader_f16 {
            AccumDtype::F16InputF32Accum
        } else {
            AccumDtype::F32
        }
    }

    pub fn workgroup_storage_fits(&self, bytes: u32) -> bool {
        bytes <= self.max_compute_workgroup_storage_size
    }

    pub fn summary(&self) -> String {
        format!(
            "{} [{} {}] driver={} shader_f16={} f16_in_f32={} subgroup={}({}..{} runtime={:?}) coop_mat={}({} cfg, gemm_tile={:?}) wg_storage={} wg_invocations={} wg_per_dim={}",
            self.adapter_name,
            self.backend,
            self.device_type,
            self.driver,
            self.shader_f16,
            self.f16_in_f32,
            self.subgroup,
            self.subgroup_min_size,
            self.subgroup_max_size,
            self.subgroup_runtime_width,
            self.cooperative_matrix,
            self.coop_configs.len(),
            self.coop_gemm_tile(),
            self.max_compute_workgroup_storage_size,
            self.max_compute_invocations_per_workgroup,
            self.max_compute_workgroups_per_dimension,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionStrategy {
    Subgroup,
    WorkgroupTree,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmStrategy {
    CoopMat,
    Scalar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumDtype {
    F16InputF32Accum,
    F32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualStatus {
    pub backend: String,
    pub qualified: bool,
    pub reason: Option<String>,
}

pub const MIN_STORAGE_BUFFERS: u32 = 6;
pub const MIN_WORKGROUP_INVOCATIONS: u32 = 256;
pub const MIN_WORKGROUP_STORAGE: u32 = 16384;

pub fn qualify(caps: &Capabilities) -> QualStatus {
    let mut reasons: Vec<String> = Vec::new();
    if caps.max_compute_invocations_per_workgroup < MIN_WORKGROUP_INVOCATIONS {
        reasons.push(format!(
            "max_compute_invocations_per_workgroup {} < {MIN_WORKGROUP_INVOCATIONS}",
            caps.max_compute_invocations_per_workgroup
        ));
    }
    if caps.max_compute_workgroup_size_x < MIN_WORKGROUP_INVOCATIONS {
        reasons.push(format!(
            "max_compute_workgroup_size_x {} < {MIN_WORKGROUP_INVOCATIONS}",
            caps.max_compute_workgroup_size_x
        ));
    }
    if caps.max_compute_workgroup_storage_size < MIN_WORKGROUP_STORAGE {
        reasons.push(format!(
            "max_compute_workgroup_storage_size {} < {MIN_WORKGROUP_STORAGE}",
            caps.max_compute_workgroup_storage_size
        ));
    }
    if caps.max_storage_buffers_per_shader_stage < MIN_STORAGE_BUFFERS {
        reasons.push(format!(
            "max_storage_buffers_per_shader_stage {} < {MIN_STORAGE_BUFFERS}",
            caps.max_storage_buffers_per_shader_stage
        ));
    }
    QualStatus {
        backend: caps.backend.clone(),
        qualified: reasons.is_empty(),
        reason: if reasons.is_empty() {
            None
        } else {
            Some(reasons.join("; "))
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline() -> Capabilities {
        Capabilities {
            adapter_name: String::from("fake"),
            backend: String::from("Vulkan"),
            device_type: String::from("DiscreteGpu"),
            driver: String::from("test"),
            shader_f16: false,
            f16_in_f32: false,
            subgroup: false,
            timestamp_query: false,
            subgroup_min_size: 0,
            subgroup_max_size: 0,
            subgroup_runtime_width: None,
            cooperative_matrix: false,
            coop_configs: Vec::new(),
            coop_note: None,
            max_compute_workgroup_storage_size: 16384,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroups_per_dimension: 65535,
            max_storage_buffer_binding_size: 1 << 28,
            max_buffer_size: 1 << 30,
            max_storage_buffers_per_shader_stage: 8,
        }
    }

    #[test]
    fn baseline_device_qualifies() {
        let st = qualify(&baseline());
        assert!(st.qualified, "{:?}", st.reason);
        assert!(st.reason.is_none());
    }

    #[test]
    fn small_workgroup_storage_disqualifies() {
        let mut caps = baseline();
        caps.max_compute_workgroup_storage_size = 8192;
        let st = qualify(&caps);
        assert!(!st.qualified);
        assert!(st
            .reason
            .expect("reason")
            .contains("max_compute_workgroup_storage_size 8192"));
    }

    #[test]
    fn coop_mat_bit_selects_the_coop_schedule() {
        let mut caps = baseline();
        assert_eq!(caps.gemm_strategy(), GemmStrategy::Scalar);
        caps.cooperative_matrix = true;
        assert_eq!(caps.gemm_strategy(), GemmStrategy::CoopMat);
    }

    fn coop_ready() -> Capabilities {
        let mut caps = baseline();
        caps.cooperative_matrix = true;
        caps.shader_f16 = true;
        caps.subgroup = true;
        caps.subgroup_min_size = 32;
        caps.subgroup_max_size = 32;
        caps.coop_configs = vec![
            CoopConfig::new(16, 8, 16, CoopScalar::F16, CoopScalar::F32),
            CoopConfig::new(16, 16, 16, CoopScalar::F16, CoopScalar::F16),
            CoopConfig::new(16, 16, 16, CoopScalar::F16, CoopScalar::F32),
        ];
        caps
    }

    fn strix_halo_advertised() -> Vec<CoopConfig> {
        vec![
            CoopConfig::new(16, 16, 16, CoopScalar::F16, CoopScalar::F16),
            CoopConfig::new(16, 16, 16, CoopScalar::F16, CoopScalar::F32),
        ]
    }

    #[test]
    fn coop_gemm_tile_needs_a_16x16x16_f16_f32_config() {
        let caps = coop_ready();
        assert_eq!(
            caps.coop_gemm_tile(),
            Some(CoopTile {
                m: 16,
                n: 16,
                k: 16
            })
        );
        assert_eq!(caps.coop_gemm_reason(), None);
    }

    #[test]
    fn coop_gemm_tile_is_refused_without_the_device_feature() {
        let mut caps = coop_ready();
        caps.cooperative_matrix = false;
        assert_eq!(caps.coop_gemm_tile(), None);
        assert!(caps
            .coop_gemm_reason()
            .expect("reason")
            .contains("EXPERIMENTAL_COOPERATIVE_MATRIX"));
    }

    #[test]
    fn coop_gemm_tile_is_refused_on_a_non_32_wide_subgroup() {
        let mut caps = coop_ready();
        caps.subgroup_min_size = 64;
        caps.subgroup_max_size = 64;
        assert_eq!(caps.coop_gemm_tile(), None);
        assert!(caps.coop_gemm_reason().expect("reason").contains("64..64"));
    }

    #[test]
    fn coop_gemm_tile_is_refused_when_only_f16_accumulators_exist() {
        let mut caps = coop_ready();
        caps.coop_configs.retain(|c| !c.cr_f32());
        assert_eq!(caps.coop_gemm_tile(), None);
        assert!(caps
            .coop_gemm_reason()
            .expect("reason")
            .contains("16x16x16 f16xf16->f32"));
    }

    #[test]
    fn subgroup_bit_selects_the_subgroup_reduction() {
        let mut caps = baseline();
        assert_eq!(caps.reduction_strategy(), ReductionStrategy::WorkgroupTree);
        caps.subgroup = true;
        caps.subgroup_min_size = 32;
        caps.subgroup_max_size = 32;
        assert_eq!(caps.reduction_strategy(), ReductionStrategy::Subgroup);
    }

    #[test]
    fn shader_f16_selects_mixed_precision_accumulation() {
        let mut caps = baseline();
        assert_eq!(caps.accum_dtype(), AccumDtype::F32);
        caps.shader_f16 = true;
        assert_eq!(caps.accum_dtype(), AccumDtype::F16InputF32Accum);
    }

    #[test]
    fn workgroup_storage_budget_rejects_the_flash_decode_sacc_tile() {
        let caps = baseline();
        assert!(caps.workgroup_storage_fits(2048 + 64));
        assert!(!caps.workgroup_storage_fits(18496));
    }

    const SWEEP_COMBOS: [(CoopScalar, CoopScalar); 3] = [
        (CoopScalar::F16, CoopScalar::F16),
        (CoopScalar::F16, CoopScalar::F32),
        (CoopScalar::F32, CoopScalar::F32),
    ];

    #[test]
    fn the_config_that_crashes_aco_is_the_one_the_filter_removes() {
        let advertised = strix_halo_advertised();
        let crasher = CoopRequest::square(16, CoopScalar::F32, CoopScalar::F32);
        assert!(!crasher.is_advertised(&advertised));
        let why = coop_skip_reason(&crasher, &advertised).expect("must be skipped");
        assert!(why.contains("16x16x16 f32xf32->f32"), "{why}");
        assert!(why.contains("f16xf16->f32"), "{why}");
        assert_eq!(
            coop_decide(&crasher, &advertised, false),
            CoopDecision::Skip(why)
        );
    }

    #[test]
    fn both_advertised_configs_survive_the_filter() {
        let advertised = strix_halo_advertised();
        for cfg in &advertised {
            let req = CoopRequest::new(cfg.m, cfg.n, cfg.k, cfg.ab, cfg.cr);
            assert_eq!(coop_skip_reason(&req, &advertised), None, "{}", req.label());
            assert_eq!(
                coop_decide(&req, &advertised, false),
                CoopDecision::Compile,
                "{}",
                req.label()
            );
        }
    }

    #[test]
    fn the_probe_sweep_shrinks_to_the_advertised_pair_on_strix_halo() {
        let advertised = strix_halo_advertised();
        let sweep = coop_square_sweep(&[8, 16], &SWEEP_COMBOS);
        assert_eq!(sweep.len(), 6);
        let part = coop_partition(&sweep, &advertised, false);
        assert_eq!(
            part.compile,
            vec![
                CoopRequest::square(16, CoopScalar::F16, CoopScalar::F16),
                CoopRequest::square(16, CoopScalar::F16, CoopScalar::F32),
            ]
        );
        assert_eq!(part.skipped.len(), 4);
        assert!(part.forced.is_empty());
        let skipped: Vec<String> = part.skipped.iter().map(|(r, _)| r.label()).collect();
        assert_eq!(
            skipped,
            vec![
                "8x8x8 f16xf16->f16",
                "8x8x8 f16xf16->f32",
                "8x8x8 f32xf32->f32",
                "16x16x16 f32xf32->f32",
            ]
        );
    }

    #[test]
    fn an_adapter_that_advertises_nothing_gets_nothing_compiled() {
        let sweep = coop_square_sweep(&[8, 16], &SWEEP_COMBOS);
        let part = coop_partition(&sweep, &[], false);
        assert!(part.compile.is_empty());
        assert_eq!(part.skipped.len(), 6);
        assert!(part.skipped[0]
            .1
            .contains("no cooperative-matrix configurations"));
    }

    #[test]
    fn the_unsafe_sweep_flag_restores_every_combination_and_records_why() {
        let advertised = strix_halo_advertised();
        let sweep = coop_square_sweep(&[8, 16], &SWEEP_COMBOS);
        let part = coop_partition(&sweep, &advertised, true);
        assert_eq!(part.compile, sweep);
        assert!(part.skipped.is_empty());
        assert_eq!(part.forced.len(), 4);
        assert!(matches!(
            coop_decide(
                &CoopRequest::square(16, CoopScalar::F32, CoopScalar::F32),
                &advertised,
                true,
            ),
            CoopDecision::CompileUnadvertised(_)
        ));
    }

    #[test]
    fn the_filter_separates_scalar_types_that_a_boolean_would_merge() {
        let advertised = vec![CoopConfig::new(
            16,
            16,
            16,
            CoopScalar::I32,
            CoopScalar::I32,
        )];
        let f32_req = CoopRequest::square(16, CoopScalar::F32, CoopScalar::F32);
        assert!(!f32_req.is_advertised(&advertised));
        let i32_req = CoopRequest::square(16, CoopScalar::I32, CoopScalar::I32);
        assert!(i32_req.is_advertised(&advertised));
    }

    #[test]
    fn non_square_and_mismatched_extents_are_not_waved_through() {
        let advertised = vec![CoopConfig::new(16, 8, 16, CoopScalar::F16, CoopScalar::F32)];
        assert!(
            CoopRequest::new(16, 8, 16, CoopScalar::F16, CoopScalar::F32)
                .is_advertised(&advertised)
        );
        for bad in [
            CoopRequest::new(8, 16, 16, CoopScalar::F16, CoopScalar::F32),
            CoopRequest::new(16, 8, 8, CoopScalar::F16, CoopScalar::F32),
            CoopRequest::new(16, 8, 16, CoopScalar::F16, CoopScalar::F16),
        ] {
            assert!(!bad.is_advertised(&advertised), "{}", bad.label());
        }
    }

    #[test]
    fn wgsl_scan_finds_the_combination_a_shader_will_ask_for() {
        let src = "enable f16;\nenable wgpu_cooperative_matrix;\n\
                   alias CA = coop_mat16x16<f16, A>;\n\
                   alias CB = coop_mat16x16<f16, B>;\n\
                   alias CC = coop_mat16x16<f32, C>;\n";
        assert_eq!(
            coop_requests_in_wgsl(src),
            vec![CoopRequest::square(16, CoopScalar::F16, CoopScalar::F32)]
        );
        let f32_src = src.replace("<f16,", "<f32,");
        assert_eq!(
            coop_requests_in_wgsl(&f32_src),
            vec![CoopRequest::square(16, CoopScalar::F32, CoopScalar::F32)]
        );
        assert!(!coop_requests_in_wgsl(&f32_src)[0].is_advertised(&strix_halo_advertised()));
    }

    #[test]
    fn wgsl_scan_reads_non_square_declarations_and_ignores_noise() {
        let src = "alias A0 = coop_mat16x8 < f16 , A > ;\n\
                   alias B0 = coop_mat8x32<f16, B>;\n\
                   alias C0 = coop_mat16x32<f32, C>;\n\
                   // coop_mat99x99<f64, Z> and coop_matNxM<f16, A> are not types\n";
        assert_eq!(
            coop_requests_in_wgsl(src),
            vec![CoopRequest::new(
                16,
                32,
                8,
                CoopScalar::F16,
                CoopScalar::F32
            )]
        );
        assert!(coop_requests_in_wgsl("no coop matrices here").is_empty());
        assert!(coop_requests_in_wgsl("alias X = coop_mat8x8<f16, A>;").is_empty());
    }
}
