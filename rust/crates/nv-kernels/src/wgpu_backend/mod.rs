pub mod dequant;
pub mod device;
pub mod dispatch;
pub mod kernels;
pub mod na;
pub mod na_attn;
pub mod na_bf16;
pub mod pack;
pub mod qualify;

pub use dequant::DEQUANT_WGSL;
pub use device::{
    WgpuContext, SG_WIDTH_UNPRIMED_ENV_OPT_IN_MIRRORS_THE_BROWSERS_UNPROBED_RUNTIME_WIDTH,
    SUBGROUPS_UNGRANTED_ENV_OPT_IN_RUNS_THE_STRIPPED_FALLBACK_TWIN_ON_A_GRANTING_DEVICE,
};
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

pub const STRICT_VALIDATORS_REJECT_SUBGROUP_FNS_AT_MODULE_LEVEL_SO_UNGRANTED_DEVICES_NEED_THEM_STRIPPED_NOT_MERELY_UNSELECTED:
    &str = "Tint validates every function in a module, so a module carrying one subgroup call \
     fails createShaderModule wholesale on a device without the subgroups feature even when the \
     pipeline entry point is subgroup-free; the split keeps subgroup entry points out of the \
     module entirely, and the caller must also de-select the arms that dispatched them";

pub fn strip_subgroup_fns(src: &str) -> String {
    let _ = STRICT_VALIDATORS_REJECT_SUBGROUP_FNS_AT_MODULE_LEVEL_SO_UNGRANTED_DEVICES_NEED_THEM_STRIPPED_NOT_MERELY_UNSELECTED;
    let mut out = String::new();
    let mut item = String::new();
    let mut depth = 0i64;
    let mut braced = false;
    for line in src.lines() {
        item.push_str(line);
        item.push('\n');
        for c in line.chars() {
            match c {
                '{' => {
                    depth += 1;
                    braced = true;
                }
                '}' => depth -= 1,
                _ => {}
            }
        }
        assert!(depth >= 0, "unbalanced braces while splitting wgsl items");
        let closed = depth == 0 && (braced || line.trim_end().ends_with(';'));
        if closed {
            if !(item.contains("fn ") && item.contains("subgroup")) {
                out.push_str(&item);
            }
            item.clear();
            braced = false;
        }
    }
    assert!(
        depth == 0 && item.trim().is_empty(),
        "wgsl source ended mid-item while stripping subgroup fns"
    );
    assert!(
        !out.contains("subgroup"),
        "a subgroup token survived outside a strippable fn item"
    );
    out
}

#[cfg(test)]
mod strip_subgroup_tests {
    use super::strip_subgroup_fns;

    const MIXED: &str = "struct P { n: u32, };\n\
@group(0) @binding(0) var<storage, read> w: array<u32>;\n\
fn helper(a: f32) -> f32 {\n    return a + 1.0;\n}\n\
@compute @workgroup_size(256)\n\
fn sg_entry(@builtin(subgroup_invocation_id) lane: u32) {\n\
    var acc = 0.0;\n\
    acc = acc + subgroupShuffleXor(acc, 1u);\n\
}\n\
@compute @workgroup_size(64)\n\
fn plain_entry(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let x = helper(1.0);\n\
}\n";

    #[test]
    fn subgroup_fns_drop_with_their_attributes_and_the_rest_survives() {
        let out = strip_subgroup_fns(MIXED);
        assert!(!out.contains("subgroup"));
        assert!(!out.contains("sg_entry"));
        assert!(out.contains("struct P"));
        assert!(out.contains("var<storage, read> w"));
        assert!(out.contains("fn helper"));
        assert!(out.contains("fn plain_entry"));
        assert!(out.contains("@compute @workgroup_size(64)"));
    }

    #[test]
    fn a_subgroup_free_source_passes_through_with_every_item_intact() {
        let clean = "const A: u32 = 1u;\nfn f() -> u32 {\n    return A;\n}\n";
        assert_eq!(strip_subgroup_fns(clean), clean);
    }
}
