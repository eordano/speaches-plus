{ pkgs }:
let
  runtimeLibraries = [
    pkgs.stdenv.cc.cc.lib
    pkgs.glibc
  ];
in
pkgs.runCommand "laguna-xqa-fp8-runtime-g83-f8c830fc" {
  src = ./runtime;
  dontStrip = true;
  dontPatchELF = true;
  dontFixup = true;
  passthru.runtimeLibraryPath = pkgs.lib.makeLibraryPath runtimeLibraries;
} ''
  set -eu
  mkdir -p "$out"
  cp "$src"/liblaguna_xqa_fp8_adapter.so "$out/"
  cp "$src"/laguna-xqa-fp8-m1.cubin "$out/"
  cp "$src"/laguna-xqa-fp8-m2.cubin "$out/"
  cp "$src"/laguna-xqa-fp8-m3.cubin "$out/"
  cp "$src"/laguna-xqa-fp8-m5.cubin "$out/"
  cp "$src"/sha256.txt "$out/"
  cd "$out"
  sha256sum --check --strict sha256.txt
''
