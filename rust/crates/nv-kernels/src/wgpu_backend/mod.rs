pub mod dequant;
pub mod device;
pub mod dispatch;
pub mod kernels;
pub mod na;
pub mod pack;
pub mod na_attn;
pub mod na_bf16;
pub mod qualify;

pub use dequant::DEQUANT_WGSL;
pub use device::WgpuContext;
pub use qualify::{Capabilities, QualStatus};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgpuError {
    NoAdapter(String),
    DeviceRequest(String),
    Unsupported(String),
    ShaderCompile(String),
    Readback(String),
    Shape(String),
    Unimplemented(&'static str),
}

impl std::fmt::Display for WgpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter(m) => write!(f, "no wgpu adapter: {m}"),
            Self::DeviceRequest(m) => write!(f, "wgpu device request failed: {m}"),
            Self::Unsupported(m) => write!(f, "unsupported on this wgpu device: {m}"),
            Self::ShaderCompile(m) => write!(f, "wgsl compilation failed: {m}"),
            Self::Readback(m) => write!(f, "buffer readback failed: {m}"),
            Self::Shape(m) => write!(f, "shape mismatch: {m}"),
            Self::Unimplemented(k) => write!(f, "wgpu kernel not implemented: {k}"),
        }
    }
}

impl std::error::Error for WgpuError {}

pub type Result<T> = std::result::Result<T, WgpuError>;

pub fn compose(body: &str) -> String {
    format!("{DEQUANT_WGSL}\n{body}\n")
}

pub fn compose_enabled(enables: &[&str], body: &str) -> String {
    let mut out = String::new();
    for e in enables {
        out.push_str("enable ");
        out.push_str(e);
        out.push_str(";\n");
    }
    out.push_str(&compose(body));
    out
}
