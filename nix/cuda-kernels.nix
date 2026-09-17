{
  pkgs,
  craneLib,
  commonArgs,
}:
let
  nativeSrc = pkgs.lib.fileset.toSource {
    root = ../rust/crates/nv-kernels;
    fileset = pkgs.lib.fileset.unions [
      ../rust/crates/nv-kernels/build.rs
      ../rust/crates/nv-kernels/include
      ../rust/crates/nv-kernels/cuda
      ../rust/crates/nv-kernels/cuda_sm120
      (pkgs.lib.fileset.maybeMissing ../rust/crates/nv-kernels/cpp)
      (pkgs.lib.fileset.maybeMissing ../rust/crates/nv-kernels/third_party)
    ];
  };
  kernelSrc = craneLib.mkDummySrc {
    src = commonArgs.src;
    extraDummyScript = commonArgs.extraDummyScript + ''
      cp -r --no-preserve=mode,ownership ${nativeSrc}/. $out/crates/nv-kernels/
      sed -i 's|^build = .*|build = "build.rs"|' "$out/crates/nv-kernels/Cargo.toml"
    '';
  };
in
craneLib.buildPackage (
  commonArgs
  // {
    pname = "nv-cuda-kernels";
    src = kernelSrc;
    cargoArtifacts = null;
    cargoExtraArgs = "--locked -p nv-kernels --features cuda --lib";
    installPhaseCommand = ''
      mkdir -p "$out/lib"
      found=0
      for built in target/release/build/nv-kernels-*/out; do
        if [ -f "$built/libnv_kernels.a" ]; then
          test "$found" = 0
          found=1
          cp "$built"/*.a "$out/lib/"
          cp "$built/bindings.rs" "$out/bindings.rs"
          sed -n '/^cargo:rustc-link-/p' "$built/../output" \
            | sed "s|$(realpath "$built")|$out/lib|g" > "$out/link-directives"
        fi
      done
      test "$found" = 1
      test -s "$out/link-directives"
      if grep -q '/build/' "$out/link-directives"; then
        echo 'kernel linker metadata retains a build directory' >&2
        exit 1
      fi
    '';
    meta = {
      description = "Reusable CUDA kernel archives and bindings for speaches-plus";
      license = pkgs.lib.licenses.mit;
      platforms = pkgs.lib.platforms.linux;
    };
  }
)
