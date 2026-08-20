use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for watched in ["include", "cuda", "cuda_sm120", "hip", "cpp", "third_party"] {
        if Path::new(watched).exists() {
            println!("cargo:rerun-if-changed={watched}");
        }
    }
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_ARCH_LIST");
    println!("cargo:rerun-if-env-changed=ROCM_DEVICE_LIB_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH_LIST");
    println!("cargo:rerun-if-env-changed=CUTLASS_DIR");
    println!("cargo:rerun-if-env-changed=FLASHINFER_DIR");
    println!("cargo:rerun-if-env-changed=NCCL_ROOT");
    println!("cargo:rerun-if-env-changed=CUDNN_ROOT");
    println!("cargo:rerun-if-env-changed=CUDNN_FRONTEND_DIR");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    println!("cargo:rerun-if-env-changed=NVK_NVCC_WRAPPER");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));

    emit_bindings(&manifest_dir, &out_dir);

    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        build_cuda(&manifest_dir, &out_dir);
    }
}

fn emit_bindings(manifest_dir: &Path, out_dir: &Path) {
    let header = manifest_dir.join("include").join("nv_kernels.h");
    let builder = bindgen::Builder::default()
        .header(header.to_string_lossy().to_string())
        .clang_arg(format!("-I{}", manifest_dir.join("include").display()))
        .allowlist_function("nv_kernels_.*")
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    let bindings = builder.generate().expect("bindgen failed");
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("write bindings.rs failed");
}

fn resolve_launcher(name: &str) -> Option<PathBuf> {
    if name.contains('/') {
        let p = PathBuf::from(name);
        return p.is_file().then_some(p);
    }
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|d| d.join(name))
        .find(|c| c.is_file())
}

fn nvcc_launcher(nvcc: &Path, out_dir: &Path) -> PathBuf {
    if !cfg!(unix) {
        return nvcc.to_path_buf();
    }
    let Ok(raw) = env::var("NVK_NVCC_WRAPPER") else {
        return nvcc.to_path_buf();
    };
    let raw = raw.trim();
    if raw.is_empty() || raw == "0" {
        return nvcc.to_path_buf();
    }
    let Some(launcher) = resolve_launcher(raw) else {
        println!("cargo:warning=NVK_NVCC_WRAPPER={raw} not found on PATH; using nvcc directly");
        return nvcc.to_path_buf();
    };

    let dir = out_dir.join("nvcc-launcher");
    if std::fs::create_dir_all(&dir).is_err() {
        return nvcc.to_path_buf();
    }
    let shim = dir.join("nvcc");
    let body = format!(
        "#!/bin/sh\nexec '{}' '{}' \"$@\"\n",
        launcher.display(),
        nvcc.display()
    );
    let stale = std::fs::read_to_string(&shim)
        .map(|c| c != body)
        .unwrap_or(true);
    if stale && std::fs::write(&shim, &body).is_err() {
        return nvcc.to_path_buf();
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755)).is_err() {
            return nvcc.to_path_buf();
        }
    }
    shim
}

fn build_cuda(manifest_dir: &Path, out_dir: &Path) {
    let target = env::var("TARGET").unwrap_or_default();
    let target_us = target.replace('-', "_");
    for var in [
        "CXXFLAGS".to_string(),
        "CFLAGS".to_string(),
        "HOST_CXXFLAGS".to_string(),
        "HOST_CFLAGS".to_string(),
        format!("CXXFLAGS_{}", target),
        format!("CXXFLAGS_{}", target_us),
        format!("CFLAGS_{}", target),
        format!("CFLAGS_{}", target_us),
    ] {
        unsafe {
            env::remove_var(&var);
        }
    }

    let cuda_path = env::var("CUDA_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(default_cuda_path)
        .expect("CUDA_PATH not set and no default toolkit found");

    let nvcc = cuda_path.join("bin").join("nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {}", nvcc.display());
    }
    let nvcc = nvcc_launcher(&nvcc, out_dir);

    let arches: Vec<String> = env::var("CUDA_ARCH_LIST")
        .unwrap_or_else(|_| "8.9;12.0".into())
        .split([';', ','])
        .filter(|s| !s.is_empty())
        .map(|s| s.replace('.', ""))
        .collect();

    let include_dirs = collect_includes(manifest_dir, &cuda_path);

    let mut cu_files: Vec<PathBuf> = Vec::new();
    collect_sources(&manifest_dir.join("cuda"), "cu", &mut cu_files);

    let third_party_fa = manifest_dir.join("third_party").join("flash-attention");
    if third_party_fa.exists() {
        collect_sources(&third_party_fa.join("csrc"), "cu", &mut cu_files);
    }
    let third_party_vllm = manifest_dir.join("third_party").join("vllm-csrc");
    if third_party_vllm.exists() {
        collect_sources(&third_party_vllm, "cu", &mut cu_files);
    }

    let mut cpp_files: Vec<PathBuf> = Vec::new();
    collect_sources(&manifest_dir.join("cpp"), "cpp", &mut cpp_files);

    let profile = env::var("PROFILE").unwrap_or_default();
    let is_release = profile == "release";

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .cudart("static")
        .cpp(true)
        .std("c++17")
        .compiler(nvcc.clone())
        .no_default_flags(true)
        .warnings(false)
        .debug(false)
        .opt_level(if is_release { 3 } else { 0 });

    for inc in &include_dirs {
        build.include(inc);
    }

    build.include(manifest_dir.join("cuda").join("marlin"));
    build.include(manifest_dir.join("cuda").join("marlin").join("generated"));
    for arch in &arches {
        build.flag(format!("-gencode=arch=compute_{a},code=sm_{a}", a = arch));
    }
    build
        .flag("-Xcompiler=-fPIC")
        .flag("-Xcompiler=-fno-strict-aliasing")
        .flag("--expt-relaxed-constexpr")
        .flag("--expt-extended-lambda");

    for src in cu_files.iter().chain(cpp_files.iter()) {
        build.file(src);
    }
    build.compile("nv_kernels");

    let sm120_dir = manifest_dir.join("cuda_sm120");
    let mut sm120_files: Vec<PathBuf> = Vec::new();
    collect_sources(&sm120_dir, "cu", &mut sm120_files);
    let has_sm120 = arches.iter().any(|a| a == "120");
    if !sm120_files.is_empty() && has_sm120 {
        let mut sm120 = cc::Build::new();
        sm120
            .cuda(true)
            .cudart("static")
            .cpp(true)
            .std("c++17")
            .compiler(nvcc.clone())
            .no_default_flags(true)
            .warnings(false)
            .debug(false)
            .opt_level(if is_release { 3 } else { 0 });
        for inc in &include_dirs {
            sm120.include(inc);
        }

        sm120
            .flag("-gencode=arch=compute_120a,code=sm_120a")
            .flag("-gencode=arch=compute_120f,code=sm_120f")
            .flag("-Xcompiler=-fPIC")
            .flag("-Xcompiler=-fno-strict-aliasing")
            .flag("--expt-relaxed-constexpr")
            .flag("--expt-extended-lambda")
            .flag("-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1");
        for src in &sm120_files {
            sm120.file(src);
        }
        sm120.compile("nv_kernels_sm120");
    }

    println!(
        "cargo:rustc-link-search=native={}",
        cuda_path.join("lib64").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_path.join("lib").display()
    );
    if let Ok(nccl) = env::var("NCCL_ROOT") {
        println!("cargo:rustc-link-search=native={}/lib", nccl);
    }
    if let Ok(cudnn) = env::var("CUDNN_ROOT") {
        println!("cargo:rustc-link-search=native={}/lib", cudnn);
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=cudnn");
    println!("cargo:rustc-link-lib=dylib=nvrtc");
    println!("cargo:rustc-link-lib=dylib=nccl");
}

fn default_cuda_path() -> Option<PathBuf> {
    for p in ["/usr/local/cuda", "/opt/cuda"] {
        if Path::new(p).join("bin").join("nvcc").exists() {
            return Some(PathBuf::from(p));
        }
    }
    None
}

fn collect_includes(manifest_dir: &Path, cuda_path: &Path) -> Vec<PathBuf> {
    let mut v = vec![manifest_dir.join("include"), cuda_path.join("include")];
    if let Ok(cutlass) = env::var("CUTLASS_DIR") {
        let root = PathBuf::from(cutlass);
        v.push(root.join("include"));
        v.push(root.join("tools").join("util").join("include"));
    }
    if let Ok(cudnn_fe) = env::var("CUDNN_FRONTEND_DIR") {
        v.push(PathBuf::from(cudnn_fe).join("include"));
    }
    if let Ok(fi) = env::var("FLASHINFER_DIR") {
        let fi_root = PathBuf::from(fi);
        v.push(fi_root.join("include"));

        let bundled_util = fi_root
            .join("3rdparty")
            .join("cutlass")
            .join("tools")
            .join("util")
            .join("include");
        if bundled_util.is_dir() {
            v.push(bundled_util);
        }
    }
    if let Ok(cudnn) = env::var("CUDNN_ROOT") {
        v.push(PathBuf::from(cudnn).join("include"));
    }
    if let Ok(nccl) = env::var("NCCL_ROOT") {
        v.push(PathBuf::from(nccl).join("include"));
    }
    let fa = manifest_dir.join("third_party").join("flash-attention");
    if fa.exists() {
        v.push(fa.join("csrc"));
        v.push(fa.join("hopper"));
    }
    v
}

fn collect_sources(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    if !dir.is_dir() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_sources(&path, ext, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some(ext) {
            out.push(path);
        }
    }
}
