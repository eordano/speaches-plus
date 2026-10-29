fn main() {
    println!("cargo:rerun-if-changed=src/tts/phonemize_glue.c");
    emit_build_git_sha();

    if cfg!(target_os = "macos") && std::env::var_os("CC").is_none() {
        let xcode_clang = "/usr/bin/clang";
        if std::path::Path::new(xcode_clang).exists() {
            unsafe { std::env::set_var("CC", xcode_clang) };
        }
    }

    let mut build = cc::Build::new();
    build.file("src/tts/phonemize_glue.c");
    build.opt_level(2);

    let mut candidates: Vec<String> = Vec::new();
    if let Ok(prefix) = std::env::var("ESPEAK_PREFIX") {
        candidates.push(format!("{prefix}/include"));
    }
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(format!("{home}/.nix-profile/include"));
    }
    for inc in &candidates {
        if std::path::Path::new(inc)
            .join("espeak-ng/speak_lib.h")
            .exists()
        {
            build.include(inc);
            break;
        }
    }
    build.compile("speaches_plus_espeak_glue");
    println!("cargo:rustc-link-lib=dylib=espeak-ng");
    if let Ok(prefix) = std::env::var("ESPEAK_PREFIX") {
        println!("cargo:rustc-link-search=native={prefix}/lib");
    }

    if !cfg!(target_os = "macos") {
        return;
    }

    if let Some(framework_parent) = python3_framework_parent() {
        println!("cargo:rustc-link-arg-tests=-Wl,-rpath,{}", framework_parent);
    }

    let nix_build_where_walked_libs_are_undeclared_and_gc_deletes_them =
        std::env::var_os("NIX_BUILD_TOP").is_some();
    if nix_build_where_walked_libs_are_undeclared_and_gc_deletes_them {
        return;
    }
    let Ok(entries) = std::fs::read_dir("/nix/store") else {
        return;
    };
    let mut want = vec![
        ("libiconv-1.", "libiconv.dylib"),
        ("libopus-1.", "libopus.dylib"),
        ("libespeak-ng-1.", "libespeak-ng.dylib"),
        ("espeak-ng-1.", "libespeak-ng.dylib"),
    ];
    for entry in entries.flatten() {
        if want.is_empty() {
            break;
        }
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(idx) = want.iter().position(|(prefix, _)| name.contains(prefix)) else {
            continue;
        };
        let lib = path.join("lib");
        if lib.join(want[idx].1).exists() {
            println!("cargo:rustc-link-search=native={}", lib.display());
            want.swap_remove(idx);
        }
    }
}

fn emit_build_git_sha() {
    println!("cargo:rerun-if-env-changed=NV_BUILD_GIT_SHA");
    let sha = std::env::var("NV_BUILD_GIT_SHA")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(git_head_sha)
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=NV_BUILD_GIT_SHA={sha}");
}

fn git_head_sha() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if sha.is_empty() {
        return None;
    }
    if let Some(git_dir) = git_dir_path() {
        println!("cargo:rerun-if-changed={}/HEAD", git_dir);
    }
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .is_some_and(|o| o.status.success() && !o.stdout.is_empty());
    Some(if dirty { format!("{sha}-dirty") } else { sha })
}

fn git_dir_path() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--git-dir"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8(out.stdout).ok()?.trim().to_string())
}

fn python3_framework_parent() -> Option<String> {
    let python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = std::process::Command::new(&python)
        .args(["-c", "import sys; print(sys.prefix)"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let prefix = String::from_utf8(output.stdout).ok()?.trim().to_string();
    let mut p = std::path::PathBuf::from(prefix);
    while let Some(parent) = p.parent() {
        if parent.file_name().and_then(|n| n.to_str()) == Some("Python3.framework") {
            return parent.parent().map(|pp| pp.display().to_string());
        }
        p = parent.to_path_buf();
    }
    None
}
