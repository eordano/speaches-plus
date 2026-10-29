{
  pkgs,
  craneLib,
  commonArgs,
}:
let
  inherit (pkgs) lib;
  toml = pkgs.formats.toml { };
  workspace = lib.importTOML ../rust/Cargo.toml;
  manifest = lib.importTOML ../rust/crates/nv-kernels/Cargo.toml;
  lock = lib.importTOML ../rust/Cargo.lock;

  buildDeps = lib.attrNames manifest.build-dependencies;
  buildDepSpec =
    name:
    let
      spec = manifest.build-dependencies.${name};
      shared = workspace.workspace.dependencies.${name};
      base = if builtins.isString shared then { version = shared; } else shared;
    in
    if spec.workspace or false then
      base
      // removeAttrs spec [ "workspace" ]
      // lib.optionalAttrs (spec ? features || base ? features) {
        features = lib.unique ((base.features or [ ]) ++ (spec.features or [ ]));
      }
    else
      spec;

  byName = lib.groupBy (p: p.name) lock.package;
  lockEntry =
    dep:
    let
      parts = lib.splitString " " dep;
      candidates = byName.${lib.head parts};
    in
    if lib.length parts == 1 then
      lib.head candidates
    else
      lib.findFirst (p: p.version == lib.elemAt parts 1) (throw "Cargo.lock has no ${dep}") candidates;
  kernelsEntry = lib.head byName.nv-kernels;
  buildDepRefs = lib.filter (
    d: lib.elem (lib.head (lib.splitString " " d)) buildDeps
  ) kernelsEntry.dependencies;
  closure = lib.genericClosure {
    startSet = map (d: rec {
      pkg = lockEntry d;
      key = "${pkg.name} ${pkg.version}";
    }) buildDepRefs;
    operator =
      item:
      map (d: rec {
        pkg = lockEntry d;
        key = "${pkg.name} ${pkg.version}";
      }) (item.pkg.dependencies or [ ]);
  };
  prunedLock = {
    inherit (lock) version;
    package = [
      {
        name = "nv-kernels";
        inherit (manifest.package) version;
        dependencies = buildDepRefs;
      }
    ]
    ++ map (i: i.pkg) closure;
  };

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
  kernelSrc = pkgs.runCommand "nv-kernels-source" { } ''
    mkdir -p $out/crates/nv-kernels/src
    cp ${
      toml.generate "Cargo.toml" {
        workspace = {
          members = [ "crates/nv-kernels" ];
          resolver = workspace.workspace.resolver or "2";
        };
        profile.release = workspace.profile.release or { };
      }
    } $out/Cargo.toml
    cp ${toml.generate "Cargo.lock" prunedLock} $out/Cargo.lock
    cp -r --no-preserve=mode,ownership ${nativeSrc}/. $out/crates/nv-kernels/
    cp ${
      toml.generate "Cargo.toml" {
        package = {
          inherit (manifest.package) name version links;
          inherit (workspace.workspace.package) edition;
          build = "build.rs";
          publish = false;
        };
        lib.path = "src/lib.rs";
        build-dependencies = lib.genAttrs buildDeps buildDepSpec;
        features.cuda = [ ];
      }
    } $out/crates/nv-kernels/Cargo.toml
    touch $out/crates/nv-kernels/src/lib.rs
  '';
in
craneLib.buildPackage (
  removeAttrs commonArgs [
    "postUnpack"
    "extraDummyScript"
  ]
  // {
    pname = "nv-cuda-kernels";
    src = kernelSrc;
    cargoVendorDir = craneLib.vendorCargoDeps { cargoLockParsed = prunedLock; };
    cargoArtifacts = null;
    cargoExtraArgs = "--offline -j ${commonArgs.env.CUDAFORGE_THREADS} -p nv-kernels --features cuda --lib";
    preBuild = (commonArgs.preBuild or "") + ''
      if [ -d /var/cache/ccache ] && [ -w /var/cache/ccache ]; then
        export NVK_NVCC_WRAPPER=${pkgs.ccache}/bin/ccache
        export CCACHE_DIR=/var/cache/ccache CCACHE_BASEDIR="$PWD" CCACHE_UMASK=007 CCACHE_MAXSIZE=30G
      fi
    '';
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
