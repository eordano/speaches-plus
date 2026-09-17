{
  pkgs,
  craneLib,
  src,
  thirdParty,
  cargoVendorDir ? craneLib.vendorCargoDeps { inherit src; },
}:
let
  wasm-bindgen-cli = pkgs.buildWasmBindgenCli rec {
    version = "0.2.127";
    src = pkgs.fetchzip {
      urls = [
        "https://git.decent.foundation/api/packages/mirrors/generic/wasm-bindgen-cli/${version}/wasm-bindgen-cli-${version}.crate"
        "https://static.crates.io/crates/wasm-bindgen-cli/wasm-bindgen-cli-${version}.crate"
      ];
      extension = "tar.gz";
      hash = "sha256-di+qBAdd7pENLiIB9CoZoab+W5xeDoByMREcCGTSzWo=";
    };
    cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
      inherit src;
      hash = "sha256-FTv2GZIAQs0ePdIZXIXil7JbZ6kIT05VG6vqC1qNFxQ=";
    };
  };

  commonArgs = {
    pname = "nv-ocr-web";
    version = "0.0.1";
    inherit src cargoVendorDir;

    postUnpack = ''
      rm -rf $sourceRoot/third-party
      cp -r --no-preserve=mode,ownership ${thirdParty} $sourceRoot/third-party
    '';

    extraDummyScript = ''
      rm -rf $out/third-party
      cp -r --no-preserve=mode,ownership ${thirdParty} $out/third-party
    '';

    cargoExtraArgs = "-p nv-ocr-web --no-default-features --features browser --locked";
    CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
    RUSTFLAGS = ''--cfg getrandom_backend="wasm_js"'';

    LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";

    nativeBuildInputs = [ pkgs.lld ];
    doCheck = false;
    strictDeps = true;
  };
in
craneLib.buildPackage (
  commonArgs
  // {
    cargoArtifacts = craneLib.buildDepsOnly commonArgs;
    nativeBuildInputs = commonArgs.nativeBuildInputs ++ [ wasm-bindgen-cli ];

    installPhaseCommand = ''
      mkdir -p $out
      wasm-bindgen --target web --no-typescript --out-dir $out \
        target/wasm32-unknown-unknown/release/nv_ocr_web.wasm
      grep -q 'export function engine_init' $out/nv_ocr_web.js
    '';

    passthru = { inherit wasm-bindgen-cli; };

    meta = {
      description = "nv-ocr-web browser build (wasm-bindgen glue + wasm)";
      license = pkgs.lib.licenses.mit;
      platforms = pkgs.lib.platforms.unix;
    };
  }
)
