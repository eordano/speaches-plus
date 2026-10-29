{
  description = "speaches-plus -- Rust/Go realtime servers + Python nano-vLLM (chat / ASR / TTS / image+video)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.nix-hug.url = "github:eordano/nix-hug";
  inputs.nix-hug.inputs.nixpkgs.follows = "nixpkgs";
  inputs.nixpkgs-pyannote3.url = "github:NixOS/nixpkgs/nixos-24.05";
  inputs.crane.url = "github:ipetkov/crane";

  outputs =
    {
      self,
      nixpkgs,
      nix-hug,
      nixpkgs-pyannote3,
      crane,
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems f;

      rustSrc =
        let
          fs = nixpkgs.lib.fileset;
          crateDirs = nixpkgs.lib.attrNames (
            nixpkgs.lib.filterAttrs (_: type: type == "directory") (builtins.readDir ./rust/crates)
          );
          declaresExampleTargets =
            crate:
            nixpkgs.lib.hasInfix "[[example]]" (builtins.readFile (./rust/crates + "/${crate}/Cargo.toml"));
          crateNonBuildTrees = fs.unions (
            nixpkgs.lib.concatMap (
              crate:
              map (dir: fs.maybeMissing (./rust/crates + "/${crate}/${dir}")) (
                [
                  "tests"
                  "benches"
                  "hip"
                ]
                ++ nixpkgs.lib.optional (!declaresExampleTargets crate) "examples"
              )
            ) crateDirs
          );
          crateNotes = fs.fileFilter (
            file: file.hasExt "md" || file.hasExt "py" || file.hasExt "html" || file.hasExt "sh"
          ) ./rust/crates;
          buildTimeFixtures = ./rust/crates/nv-kernels/tests/fixtures;
        in
        fs.toSource {
          root = ./rust;
          fileset =
            fs.difference
              (fs.unions [
                ./rust/Cargo.toml
                ./rust/Cargo.lock
                ./rust/build.rs
                ./rust/.cargo
                ./rust/src
                ./rust/crates
                (fs.fileFilter (file: !file.hasExt "npy") ./rust/voices)
                ./rust/static
              ])
              (
                fs.difference (fs.unions [
                  crateNonBuildTrees
                  crateNotes
                ]) buildTimeFixtures
              );
        };

      cudaCapabilities = [
        "8.9"
        "12.0"
      ];

      skipPy =
        pkg:
        pkg.overridePythonAttrs (_: {
          doCheck = false;
          doInstallCheck = false;
        });
      disablePy =
        tests: pkg:
        pkg.overridePythonAttrs (old: {
          disabledTests = (old.disabledTests or [ ]) ++ tests;
        });

      unbreakDlinfoOverlay = _final: prev: {
        python314Packages = prev.python314Packages.override {
          overrides =
            _pyFinal: pyPrev:
            {
              dlinfo = (skipPy pyPrev.dlinfo).overridePythonAttrs (_: {
                meta.broken = false;
              });
            }
            // nixpkgs.lib.optionalAttrs prev.stdenv.hostPlatform.isDarwin {
              requests-futures = skipPy pyPrev.requests-futures;
              aiohttp = skipPy pyPrev.aiohttp;
              geoip2 = skipPy pyPrev.geoip2;
              pyarrow = disablePy [ "test_timezone_absent" ] pyPrev.pyarrow;
              httplib2 = disablePy [ "test_socks5_auth" ] pyPrev.httplib2;
            };
        };

        pythonPackagesExtensions =
          (prev.pythonPackagesExtensions or [ ])
          ++ nixpkgs.lib.optionals prev.stdenv.hostPlatform.isDarwin [
            (_pyFinal: pyPrev: {
              httpcore2 = skipPy pyPrev.httpcore2;
              httpx2 = skipPy pyPrev.httpx2;
              django = skipPy pyPrev.django;
              nltk = skipPy pyPrev.nltk;
              google-api-core = disablePy [ "test_apply_passthrough" ] pyPrev.google-api-core;
              moto = disablePy [ "test_state_machine_with_parallel_states" ] pyPrev.moto;
              openai = disablePy [
                "test_mtls_example_presents_full_client_chain"
                "test_default_chain_uses_container_credentials"
                "test_default_chain_uses_ec2_instance_metadata"
                "test_existing_aiohttp_adapter_is_unchanged_when_installed"
              ] pyPrev.openai;
            })
          ];
      };

      ffmpegResignOverlay =
        _final: prev:
        nixpkgs.lib.optionalAttrs prev.stdenv.hostPlatform.isDarwin {
          ffmpeg-headless = prev.ffmpeg-headless.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ prev.darwin.sigtool ];
            postFixup = (old.postFixup or "") + ''
              for out in $(getAllOutputNames); do
                eval "outPath=\$$out"
                [ -d "$outPath/lib" ] || continue
                for f in "$outPath"/lib/*.dylib; do
                  [ -L "$f" ] && continue
                  [ -f "$f" ] || continue
                  codesign --remove-signature "$f" 2>/dev/null || true
                  codesign --sign - --force "$f"
                done
              done
            '';
          });
        };

      llamaCppNoCheckDarwinOverlay =
        _final: prev:
        nixpkgs.lib.optionalAttrs prev.stdenv.hostPlatform.isDarwin {
          llama-cpp = prev.llama-cpp.overrideAttrs { doCheck = false; };
        };

      nodejsSlimDarwinOverlay =
        _final: prev:
        nixpkgs.lib.optionalAttrs prev.stdenv.hostPlatform.isDarwin {
          nodejs-slim_26 = (prev.nodejs-slim_26.overrideAttrs { doCheck = false; }).overrideAttrs (old: {
            outputChecks = builtins.mapAttrs (_: v: v // { disallowedReferences = [ ]; }) (
              old.outputChecks or { }
            );
          });
        };

      arrowCppDarwinOverlay =
        _final: prev:
        nixpkgs.lib.optionalAttrs prev.stdenv.hostPlatform.isDarwin {
          arrow-cpp = prev.arrow-cpp.overrideAttrs (_: {
            installCheckPhase = ''
              runHook preInstallCheck
              ctest -L unittest --exclude-regex '^(arrow-flight-test|arrow-gcsfs-test|arrow-flight-integration-test|arrow-orc-adapter-test|arrow-flight-internals-test|arrow-flight-sql-test|arrow-azurefs-test|parquet-encryption-test)$'
              runHook postInstallCheck
            '';
          });
        };

      cudaSetupHookNvccFixOverlay =
        _final: prev:
        let
          fixCudaScope =
            scope:
            scope.overrideScope (
              _cudaFinal: cudaPrev: {
                cudnn-frontend = cudaPrev.cudnn-frontend.overrideAttrs (old: {
                  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ _cudaFinal.cuda_nvcc ];

                  cmakeFlags =
                    builtins.filter (
                      f: !(builtins.isString f && (builtins.match "-DCUDNN_FRONTEND_BUILD_(SAMPLES|TESTS).*" f) != null)
                    ) (old.cmakeFlags or [ ])
                    ++ [
                      "-DCUDNN_FRONTEND_BUILD_SAMPLES:BOOL=OFF"
                      "-DCUDNN_FRONTEND_BUILD_TESTS:BOOL=OFF"
                    ];

                  postInstall = (old.postInstall or "") + ''
                    mkdir -p "$legacy_samples" "$samples" "$tests"
                  '';
                });

                cutlass = cudaPrev.cutlass.overrideAttrs (old: {
                  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ _cudaFinal.cuda_nvcc ];
                });

                setupCudaHook = cudaPrev.setupCudaHook.overrideAttrs (old: {
                  buildCommand = (old.buildCommand or "") + ''
                    cat >> "$out/nix-support/setup-hook" <<'HOOKFIX'

                    setupCUDAToolkit_ROOT() {
                      (("''${NIX_DEBUG:-0}" >= 1)) && echo "setupCUDAToolkit_ROOT: cudaHostPathsSeen=''${!cudaHostPathsSeen[*]}" >&2

                      for path in "''${!cudaHostPathsSeen[@]}"; do
                        addToSearchPathWithCustomDelimiter ";" CUDAToolkit_ROOT "$path"
                        if [[ -d "$path/include" ]]; then
                          addToSearchPathWithCustomDelimiter ";" CUDAToolkit_INCLUDE_DIR "$path/include"
                        fi
                      done

                      local nvccExe
                      if nvccExe="$(type -P nvcc)"; then
                        addToSearchPathWithCustomDelimiter ";" CUDAToolkit_ROOT "''${nvccExe%/bin/nvcc}"
                      fi

                      if [[ -n ''${CUDAToolkit_INCLUDE_DIR-} ]]; then
                        cmakeFlagsArray+=("-DCUDAToolkit_INCLUDE_DIR=''${CUDAToolkit_INCLUDE_DIR}")
                      fi
                      if [[ -n ''${CUDAToolkit_ROOT-} ]]; then
                        cmakeFlagsArray+=("-DCUDAToolkit_ROOT=''${CUDAToolkit_ROOT}")
                      fi
                    }
                    HOOKFIX
                  '';
                });
              }
            );
        in
        {
          cudaPackages = fixCudaScope prev.cudaPackages;
          cudaPackages_12 = fixCudaScope prev.cudaPackages_12;
        };

      onnxruntimeNoNcclOverlay = _final: prev: {
        onnxruntime = prev.onnxruntime.override { ncclSupport = false; };
      };

      mkPkgsWith =
        cuda: system:
        import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            nvidia.acceptLicense = true;
            inherit cudaCapabilities;
            cudaForwardCompat = true;
          }
          // nixpkgs.lib.optionalAttrs cuda { cudaSupport = true; };
          overlays = [
            unbreakDlinfoOverlay
            ffmpegResignOverlay
            llamaCppNoCheckDarwinOverlay
            nodejsSlimDarwinOverlay
            arrowCppDarwinOverlay
            cudaSetupHookNvccFixOverlay
          ]
          ++ nixpkgs.lib.optionals cuda [ onnxruntimeNoNcclOverlay ];
        };
      mkPkgs = mkPkgsWith false;
      mkCudaPkgs = mkPkgsWith true;

      ctranslate2For =
        pkgs: withCUDA: cudaPackages:
        if withCUDA then
          pkgs.ctranslate2.override {
            inherit cudaPackages;
            withCUDA = true;
            withCuDNN = true;
          }
        else
          pkgs.ctranslate2;

      whisperCppFor =
        pkgs: withCUDA: cudaPackages:
        if withCUDA then
          pkgs.whisper-cpp.override {
            inherit cudaPackages;
            cudaSupport = true;
          }
        else
          pkgs.whisper-cpp;

      ortLibFor =
        pkgs:
        "${pkgs.onnxruntime}/lib/libonnxruntime${
          if pkgs.stdenv.hostPlatform.isLinux then ".so" else ".dylib"
        }";

      cutlass44For =
        pkgs:
        pkgs.fetchFromGitHub {
          owner = "NVIDIA";
          repo = "cutlass";
          tag = "v4.4.2";
          hash = "sha256-0q9Ad0Z6E/rO2PdM4uQc8H0E0qs9uKc3reHepiHhjEc=";
        };

      flashinfer064For =
        pkgs:
        pkgs.fetchFromGitHub {
          owner = "flashinfer-ai";
          repo = "flashinfer";
          tag = "v0.6.4";
          fetchSubmodules = true;
          hash = "sha256-Hq3oTeEJHRvXwThI8vc06E3Ot/FnPP0sZUfze3ISa2o=";
        };

      enumBaseFlags = "-Wno-elaborated-enum-base -Wno-error=elaborated-enum-base";

      profileModels = p: profiles.${p} or (throw "unknown profile: ${p}");

      mkModels =
        system:
        let
          nix-hug-lib = nix-hug.lib.${system};
          gitRepoHashes = builtins.fromJSON (builtins.readFile ./nix-model-githashes.json);
          snapshotDir = url: rev: "models--${builtins.replaceStrings [ "/" ] [ "--" ] url}/snapshots/${rev}";
          mkModel =
            {
              url,
              rev,
              fileTreeHash,
              gitRepoHash,
              envVars,
              filters ? null,
            }:
            {
              inherit
                url
                rev
                fileTreeHash
                envVars
                ;
              snapshot = snapshotDir url rev;
              drv = nix-hug-lib.fetchModel (
                {
                  inherit
                    url
                    rev
                    fileTreeHash
                    gitRepoHash
                    ;
                }
                // (if filters != null then { inherit filters; } else { })
              );
            };
        in
        builtins.mapAttrs (name: def: mkModel (def // { gitRepoHash = gitRepoHashes.${name}; })) {
          qwen36-text = {
            url = "RedHatAI/Qwen3.6-35B-A3B-NVFP4";
            rev = "e94a930c8e9b8a7d3763e449289c261c9c7d3e5e";
            fileTreeHash = "sha256-cscbYPYWKYpv20WP6hybJ38V/CdicDAOGhgMWLuuRD8=";
            filters.exclude = [
              "model_visual\\.safetensors"
              "model_mtp\\.safetensors"
            ];
            envVars = [ "NV_CHAT_MODEL_DIR" ];
          };

          qwen36-mm = {
            url = "RedHatAI/Qwen3.6-35B-A3B-NVFP4";
            rev = "e94a930c8e9b8a7d3763e449289c261c9c7d3e5e";
            fileTreeHash = "sha256-cscbYPYWKYpv20WP6hybJ38V/CdicDAOGhgMWLuuRD8=";
            filters.include = [
              ".*\\.safetensors"
              ".*\\.json"
              ".*\\.jinja"
              ".*\\.yaml"
              ".*\\.md"
            ];
            envVars = [ "NV_CHAT_MODEL_DIR" ];
          };

          qwen35-dense = {
            url = "Qwen/Qwen3.5-9B";
            rev = "c202236235762e1c871ad0ccb60c8ee5ba337b9a";
            fileTreeHash = "sha256-WetPYBatNqQzqYvzVSYif9FZx3BtuTbSi3kR+eaxSZ8=";
            envVars = [
              "NV_QWEN35_DENSE_DIR"
              "NV_CHAT_MODEL_DIR"
            ];
          };

          qwen35-dense-nvfp4 = {
            url = "ig1/Qwen3.5-9B-NVFP4";
            rev = "3b9e07b0328357c2a7d16e0b3160956a1aaae057";
            fileTreeHash = "sha256-072rUEAQOUf/nxC9uyh1jAFHFbJSurlopH/DUX6Phzo=";
            envVars = [
              "NV_QWEN35_DENSE_NVFP4_DIR"
              "NV_CHAT_MODEL_DIR"
            ];
          };

          qwen38-dense-nvfp4 = {
            url = "unsloth/Qwen3.8-27B-NVFP4";
            rev = "f0b7c9e722f5565102fff8481c99e4d86ae099c7";
            fileTreeHash = "sha256-SNqYuhKrwjFtAcpiVx9VyvOtL/Mpf/6CvOB3uSTTs0s=";
            envVars = [
              "NV_QWEN38_DIR"
              "NV_CHAT_MODEL_DIR"
            ];
          };

          parakeet-tdt = {
            url = "istupakov/parakeet-tdt-0.6b-v2-onnx";
            rev = "0bbb45a3365852604aef28b538a8f066f4ccaa85";
            fileTreeHash = "sha256-5aifgY0uAwcmavrGsFnttD76IaB9justp+yecnOHQ00=";
            filters.exclude = [ ".*int8.*" ];
            envVars = [ "STT_PARAKEET_DIR" ];
          };

          eagle3-draft = {
            url = "RedHatAI/gemma-4-31B-it-speculator.eagle3";
            rev = "28a1c8b4bb64dbaee883ba35341841138bdf1fe3";
            fileTreeHash = "sha256-gyfiEZZJ4s/GmOK4D6RMe6azc7G9Y3JfXEU2D75qm9g=";
            envVars = [ "NV_EAGLE3_DRAFT_DIR" ];
          };

          gemma4-verifier = {
            url = "nvidia/Gemma-4-31B-IT-NVFP4";
            rev = "4135a98a9b728a548947683219633b25682223ac";
            fileTreeHash = "sha256-K/1k0U5OWRyT0j/4KKzZ8oSw30N+sFsJRJGps0X7tM0=";
            envVars = [
              "NV_GEMMA4_VERIFIER_DIR"
              "NV_CHAT_MODEL_DIR"
            ];
          };

          gemma4-e4b = {
            url = "google/gemma-4-E4B-it";
            rev = "ee0ef6023621cff504d758262d4e04895a5af4a2";
            fileTreeHash = "sha256-UnskmK8DBZy/VmRvd702z3tRk3DJClRG2CjcDouCWh4=";
            envVars = [ "GEMMA4_E4B_DIR" ];
          };

          gemma4-e4b-qat = {
            url = "google/gemma-4-E4B-it-qat-w4a16-ct";
            rev = "6cd26aaa2357fb2bad8c51699a7558a4d1a965bb";
            fileTreeHash = "sha256-JWmGAOuoGEwMgjiYgBD5PkXviUuUidUK5IIDTetUi8c=";
            envVars = [ "GEMMA4_E4B_QAT_DIR" ];
          };

          qwen3-tts = {
            url = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";
            rev = "5d83992436eae1d760afd27aff78a71d676296fc";
            fileTreeHash = "sha256-GKWzpwo3PApyZXPkqzI9lgjV3g+i+G21wFBxcFSvljc=";
            envVars = [
              "NV_TTS_TALKER_DIR"
              "NV_TTS_BASE_TALKER_DIR"
            ];
          };

          qwen3-embed = {
            url = "Qwen/Qwen3-Embedding-0.6B";
            rev = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3";
            fileTreeHash = "sha256-7Wg4Vahxbg8wVlzWRcI/X408XB+c4rDo/j3YrP1C7lE=";
            envVars = [ "NV_EMBEDDING_MODEL_DIR" ];
          };

          whisper-ct2 = {
            url = "deepdml/faster-whisper-large-v3-turbo-ct2";
            rev = "4df90f75321148c3a29a9e2351b7ddf8f5b115a8";
            fileTreeHash = "sha256-4VYZIYulFY9PlgKDlIyuLT+TJObMeXk9eHlHuEvQqtY=";
            envVars = [ "CT2_MODEL" ];
          };

          qwen-omni-template = {
            url = "Qwen/Qwen3-Omni-30B-A3B-Instruct";
            rev = "26291f793822fb6be9555850f06dfe95f2d7e695";
            fileTreeHash = "sha256-A4Q5OeXk7tfSzBhmy2ZrdrwJoCweW1pa7/cqe1K7xFc=";
            envVars = [ ];
            filters = {
              files = [
                "chat_template.json"
                "config.json"
                "generation_config.json"
                "tokenizer_config.json"
              ];
            };
          };

          qwen-aligner-template = {
            url = "Qwen/Qwen3-ForcedAligner-0.6B";
            rev = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7";
            fileTreeHash = "sha256-FAVTYzlkTvjkXotsQljzpHYx9fJhAKrDP3FN4ligkL0=";
            envVars = [ ];
            filters = {
              files = [
                "chat_template.json"
                "config.json"
                "generation_config.json"
                "tokenizer_config.json"
              ];
            };
          };

          wespeaker = {
            url = "pyannote/wespeaker-voxceleb-resnet34-LM";
            rev = "837717ddb9ff5507820346191109dc79c958d614";
            fileTreeHash = "sha256-X6meYLcrkjfV2X8rLebIzgY8BTC99R7qL8Bqsn7gEzg=";
            envVars = [ "NV_SPEAKER_MODEL_DIR" ];
          };
        };

      profiles = {
        full = [
          "gemma4-verifier"
          "eagle3-draft"
          "whisper-ct2"
          "qwen3-tts"
          "qwen3-embed"
          "wespeaker"
          "qwen-omni-template"
          "qwen-aligner-template"
        ];
        minimal = [
          "gemma4-verifier"
          "whisper-ct2"
        ];
        chat-only = [
          "gemma4-verifier"
          "eagle3-draft"
        ];
        qwen36 = [
          "qwen36-text"
          "qwen3-tts"
          "whisper-ct2"
          "qwen3-embed"
          "wespeaker"
        ];
        qwen35-dense = [ "qwen35-dense" ];
        qwen35-dense-nvfp4 = [ "qwen35-dense-nvfp4" ];
        parakeet = [ "parakeet-tdt" ];
        tts-only = [ "qwen3-tts" ];
        tts-clone = [
          "qwen3-tts"
          "whisper-ct2"
        ];
        audio-only = [
          "whisper-ct2"
          "qwen3-tts"
          "wespeaker"
        ];
        audio-light = [
          "whisper-ct2"
          "qwen3-embed"
        ];
        mm = [
          "qwen36-mm"
          "qwen3-tts"
          "whisper-ct2"
          "qwen3-embed"
          "wespeaker"
        ];
      };

      mkSpeachesModelsHub =
        system: profileName:
        let
          nix-hug-lib = nix-hug.lib.${system};
          models = mkModels system;
          names = profileModels profileName;
        in
        nix-hug-lib.buildCache {
          models = map (n: models.${n}.drv) names;
        };

      mkProfileEnv =
        system: profileName: hub:
        let
          models = mkModels system;
          names = profileModels profileName;
          oneModel =
            n:
            let
              m = models.${n};
              dir = "${hub}/${m.snapshot}";
            in
            nixpkgs.lib.listToAttrs (map (v: nixpkgs.lib.nameValuePair v dir) m.envVars);
        in
        nixpkgs.lib.foldl' (acc: n: acc // (oneModel n)) { } names;

      localHubHook =
        system: profileName:
        let
          models = mkModels system;
          names = profileModels profileName;
          lines = nixpkgs.lib.concatMapStringsSep "\n" (
            n:
            let
              m = models.${n};
              vars = nixpkgs.lib.concatStringsSep " " m.envVars;
            in
            ''
              _sp_dir="$HF_HUB_CACHE/${m.snapshot}"
              if [ -d "$_sp_dir" ]; then
                for _v in ${vars}; do export "$_v=$_sp_dir"; done
                echo "  ${n}: PRESENT" >&2
              else
                echo "  ${n}: MISSING at $_sp_dir -- the pinned revision is not in this cache; \
              the graph will not find it, which is the intended failure. Fetch it, or repin." >&2
              fi
            ''
          ) names;
        in
        ''
          export HF_HUB_CACHE="''${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
          export SPEACHES_PROFILE_NAME="${profileName}"
          echo "speaches-plus local-hub: profile=${profileName} HF_HUB_CACHE=$HF_HUB_CACHE (nothing realized into /nix/store)" >&2
          ${lines}
          unset _sp_dir _v
        '';

      mkShell =
        system:
        {
          withCUDA ? false,
          profile ? "full",
          localHub ? false,
        }:
        let
          pkgs = mkPkgs system;
          isLinux = pkgs.stdenv.hostPlatform.isLinux;
          cudaPackages = pkgs.cudaPackages_12;
          speachesModelsHub =
            if profile != null && !localHub then mkSpeachesModelsHub system profile else null;
          profileEnv =
            if profile != null && !localHub then mkProfileEnv system profile speachesModelsHub else { };

          punkt-tab = pkgs.fetchzip {
            url = "https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers/punkt_tab.zip";
            hash = "sha256-RwvF6O91YFg2DDMnykMOWQZCdmXAwfucHzkzwNHi3YY=";
          };

          ctranslate2 = ctranslate2For pkgs withCUDA cudaPackages;
          whisper-cpp = whisperCppFor pkgs withCUDA cudaPackages;
          ortLib = ortLibFor pkgs;

          runtimeLibs = [
            ctranslate2
            whisper-cpp
            pkgs.libopus
            pkgs.espeak-ng
            pkgs.onnxruntime
            pkgs.ffmpeg.dev
          ];

          cudaLibs = pkgs.lib.optionals withCUDA (
            with cudaPackages;
            [
              cuda_cudart
              cccl
              cuda_nvrtc
              libcublas
              cudnn
              cudnn-frontend
              cutlass
              nccl
            ]
          );

          shellBuilder =
            if withCUDA then pkgs.mkShell.override { stdenv = cudaPackages.backendStdenv; } else pkgs.mkShell;
        in
        shellBuilder {
          packages =
            with pkgs;
            [
              go_latest
              cargo
              rustc
              rustfmt
              rust-analyzer
              cmake
              gnumake
              pkg-config
              python3
              python3Packages.requests
              uv
              git
              libclang
            ]
            ++ pkgs.lib.optional isLinux pkgs.ccache
            ++ runtimeLibs
            ++ pkgs.lib.optionals isLinux [
              pkgs.stdenv.cc.cc.lib
              pkgs.libcxx
            ]
            ++ pkgs.lib.optionals withCUDA (
              cudaLibs
              ++ [
                cudaPackages.cuda_nvcc
                cudaPackages.nsight_compute
              ]
            );

          env = {
            LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
            ORT_DYLIB_PATH = ortLib;
            ONNXRUNTIME_LIB = ortLib;
            ESPEAK_DATA_PATH = "${pkgs.espeak-ng}/share/espeak-ng-data";
            CT2_DEVICE = if withCUDA then "cuda" else "cpu";
            CUDA_PATH = pkgs.lib.optionalString withCUDA "${cudaPackages.cudatoolkit}";
            CUDA_ARCH_LIST = pkgs.lib.optionalString withCUDA (pkgs.lib.concatStringsSep ";" cudaCapabilities);
            CUDA_COMPUTE_CAP = pkgs.lib.optionalString withCUDA (
              builtins.replaceStrings [ "." ] [ "" ] (pkgs.lib.last cudaCapabilities)
            );
            CUTLASS_DIR = pkgs.lib.optionalString withCUDA "${cutlass44For pkgs}";
            NCCL_ROOT = pkgs.lib.optionalString withCUDA "${cudaPackages.nccl}";
            CUDNN_ROOT = pkgs.lib.optionalString withCUDA "${cudaPackages.cudnn}";
            CUDNN_FRONTEND_DIR = pkgs.lib.optionalString withCUDA "${cudaPackages.cudnn-frontend}";

            FLASHINFER_DIR = pkgs.lib.optionalString withCUDA "${flashinfer064For pkgs}";

            CMAKE_CUDA_ARCHITECTURES = pkgs.lib.optionalString withCUDA (
              pkgs.lib.concatStringsSep ";" (map (c: builtins.replaceStrings [ "." ] [ "" ] c) cudaCapabilities)
            );
            CMAKE_POLICY_VERSION_MINIMUM = "3.5";
            CFLAGS = pkgs.lib.optionalString (!isLinux) enumBaseFlags;
            CXXFLAGS = pkgs.lib.optionalString (!isLinux) enumBaseFlags;
            BINDGEN_EXTRA_CLANG_ARGS = pkgs.lib.optionalString (!isLinux) "-Wno-elaborated-enum-base";
            SPEACHES_PROFILE =
              if withCUDA then
                "linux-cuda"
              else if isLinux then
                "linux-cpu"
              else
                "macos";
            SPEACHES_CARGO_FEATURE =
              if withCUDA then
                "cuda"
              else if !isLinux then
                "metal,wgpu"
              else
                "";
            HF_HUB_CACHE = if speachesModelsHub != null then "${speachesModelsHub}" else "";
            TRANSFORMERS_OFFLINE = if speachesModelsHub != null then "1" else "0";
            SPEACHES_PROFILE_NAME = if profile != null then profile else "";
            NV_PUNKT_DATA = "${punkt-tab}";
          }
          // profileEnv;

          shellHook = ''
            echo "speaches-plus dev shell: $SPEACHES_PROFILE${
              pkgs.lib.optionalString (profile != null) " · profile=${profile}"
            }" >&2
            if [ -n "$SPEACHES_CARGO_FEATURE" ]; then
              echo "  cargo: --features $SPEACHES_CARGO_FEATURE" >&2
            fi
            export REAL_CMAKE="${pkgs.cmake}/bin/cmake"
            SPEACHES_CMAKE_SHIM_DIR="$(mktemp -d -t speaches-cmake-shim.XXXXXX)"
            install -m 0755 "${./rust/scripts/cmake-install-shim.sh}" "$SPEACHES_CMAKE_SHIM_DIR/cmake"
            export PATH="$SPEACHES_CMAKE_SHIM_DIR:$PATH"
          ''
          + pkgs.lib.optionalString isLinux ''
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.vulkan-loader}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            if [ -z "''${VK_ICD_FILENAMES:-}" ] && [ -e /run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json ]; then
              export VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json
            fi
          ''
          + pkgs.lib.optionalString (!isLinux) ''
            export CC="${pkgs.stdenv.cc}/bin/cc"
            export CXX="${pkgs.stdenv.cc}/bin/c++"
          ''
          + pkgs.lib.optionalString withCUDA ''
            export LIBRARY_PATH="${cudaPackages.cuda_cudart}/lib/stubs''${LIBRARY_PATH:+:$LIBRARY_PATH}"
            if [ -d "$HOME/.cargo/registry/src" ]; then
              bash "${./rust/scripts/patch-ct2rs-cuda.sh}" >/dev/null 2>&1 || true
              bash "${./rust/scripts/patch-ct2rs-cmake-min.sh}" >/dev/null 2>&1 || true
            fi
          ''
          + pkgs.lib.optionalString (localHub && profile != null) (localHubHook system profile)
          + ''
            if [ -z "''${SPEACHES_PLUS_MODELS:-}" ] && [ -f rust/models/silero_vad.onnx ]; then
              export SPEACHES_PLUS_MODELS="$PWD/rust/models"
            fi
          '';
        };
      mkStripCommentsApp =
        system:
        let
          pkgs = mkPkgs system;
          script = pkgs.writeShellApplication {
            name = "strip-comments";
            runtimeInputs = [ pkgs.python3 ];
            text = ''
              exec python3 ${./scripts/strip-comments.py} "$@"
            '';
          };
        in
        {
          type = "app";
          program = "${script}/bin/strip-comments";
        };

      mkSpeachesPlusGo =
        {
          pkgs,
          withCUDA ? false,
          cudaPackages ? pkgs.cudaPackages_12,
        }:
        let
          ctranslate2 = ctranslate2For pkgs withCUDA cudaPackages;
          whisper-cpp = whisperCppFor pkgs withCUDA cudaPackages;
          ortLib = ortLibFor pkgs;
        in
        pkgs.buildGoLatestModule {
          pname = "speaches-plus-go";
          version = "0.0.1";
          src = pkgs.lib.fileset.toSource {
            root = ./go;
            fileset = pkgs.lib.fileset.unions [
              ./go/go.mod
              ./go/go.sum
              ./go/cmd
              ./go/internal
            ];
          };
          vendorHash = "sha256-XZyrKttJSdvszkWcGeOxN+wDwgGE7FRyW4W7Rk2C1Fk=";
          subPackages = [ "cmd/server" ];

          nativeBuildInputs = [ pkgs.pkg-config ];

          buildInputs = [
            pkgs.libopus
            pkgs.espeak-ng
            ctranslate2
            whisper-cpp
            pkgs.onnxruntime
            pkgs.ffmpeg.dev
          ];

          env = {
            ESPEAK_DATA_PATH = "${pkgs.espeak-ng}/share/espeak-ng-data";
            ORT_DYLIB_PATH = ortLib;
            ONNXRUNTIME_LIB = ortLib;
          };

          doCheck = false;

          postInstall = ''
            mv $out/bin/server $out/bin/speaches-plus-go
          '';

          meta = with pkgs.lib; {
            description = "speaches-plus realtime audio server (Go)";
            mainProgram = "speaches-plus-go";
            license = licenses.mit;
            platforms = platforms.unix;
          };
        };

      mkSpeachesPlus =
        {
          pkgs,
          system ? pkgs.stdenv.hostPlatform.system,
          withCUDA ? false,
          withVulkan ? false,
          cudaPackages ? pkgs.cudaPackages_12,
          profile ? "full",
          crate ? null,
          cudaComputeCap ? null,
        }:
        let
          inherit (pkgs) lib;
          isLinux = pkgs.stdenv.hostPlatform.isLinux;
          modelsHub = if profile != null then mkSpeachesModelsHub system profile else null;
          profileEnv = if profile != null then mkProfileEnv system profile modelsHub else { };

          cudaforgeThreadsBoundSoParallelNvccFitsInRam = "16";

          cutlassForCudaforge = pkgs.fetchgit {
            url = "https://github.com/NVIDIA/cutlass.git";
            rev = "7d49e6c7e2f8896c47f586706e67e1fb215529dc";
            hash = "sha256-cSWVzyuDC8EidTAZzHbVz0jUNK4zx5AAwfUV6lUXTXs=";
            leaveDotGit = true;
            deepClone = false;
          };

          ctranslate2 = ctranslate2For pkgs withCUDA cudaPackages;
          whisper-cpp = whisperCppFor pkgs withCUDA cudaPackages;

          features =
            if withCUDA then
              [ "cuda" ]
            else if pkgs.stdenv.hostPlatform.isDarwin then
              [
                "metal"
                "wgpu"
              ]
            else if withVulkan then
              [ "wgpu" ]
            else
              [ ];

          ortLib = ortLibFor pkgs;

          craneLib =
            let
              base = crane.mkLib pkgs;
            in
            if withCUDA then
              base.overrideScope (_final: _prev: { stdenvSelector = _: cudaPackages.backendStdenv; })
            else
              base;

          thirdParty = mkThirdParty pkgs;

          cargoVendorDir = craneLib.vendorCargoDeps {
            src = rustSrc;
            overrideVendorCargoPackage =
              p: drv:
              if withCUDA && p.name == "ct2rs" then
                drv.overrideAttrs (old: {
                  postInstall = (old.postInstall or "") + ''
                    CT2_HELPERS_OVERRIDE="$out/CTranslate2/src/cuda/helpers.h" \
                      bash ${./rust/scripts/patch-ct2rs-cuda.sh}
                  '';
                })
              else
                drv;
          };

          commonArgs = {
            pname = if crate == null then "speaches-plus" else crate;
            version = "0.0.1";
            src = rustSrc;

            postUnpack = ''
              rm -rf $sourceRoot/third-party
              cp -r --no-preserve=mode,ownership ${thirdParty} $sourceRoot/third-party
            '';

            extraDummyScript = ''
              rm -rf $out/third-party
              cp -r --no-preserve=mode,ownership ${thirdParty} $out/third-party
            '';

            inherit cargoVendorDir;

            cargoExtraArgs =
              "--locked"
              + lib.optionalString (crate != null) " -p ${crate}"
              + lib.optionalString (features != [ ]) (" --features " + lib.concatStringsSep "," features);

            nativeBuildInputs =
              with pkgs;
              [
                pkg-config
                cmake
                pkgs.rustPlatform.bindgenHook
              ]
              ++ lib.optionals withCUDA (with cudaPackages; [ cuda_nvcc ])
              ++ lib.optionals withCUDA [
                pkgs.git
              ];

            buildInputs =
              with pkgs;
              [
                libopus
                espeak-ng
                ctranslate2
                whisper-cpp
                onnxruntime
              ]
              ++ lib.optionals (!isLinux) [ pkgs.libiconv ]
              ++ lib.optionals withVulkan [ pkgs.vulkan-loader ]
              ++ lib.optionals withCUDA (
                with cudaPackages;
                [
                  cuda_cudart
                  cccl
                  cuda_nvrtc
                  libcublas
                  cudnn
                  cudnn-frontend
                  cutlass
                  nccl
                ]
              );

            env = {
              ESPEAK_PREFIX = "${pkgs.espeak-ng}";
              ORT_DYLIB_PATH = ortLib;
              ONNXRUNTIME_LIB = ortLib;
              ESPEAK_DATA_PATH = "${pkgs.espeak-ng}/share/espeak-ng-data";
              CMAKE_POLICY_VERSION_MINIMUM = "3.5";
            }
            // lib.optionalAttrs (!isLinux) {
              CFLAGS = enumBaseFlags;
              CXXFLAGS = enumBaseFlags;
              BINDGEN_EXTRA_CLANG_ARGS = "-Wno-elaborated-enum-base";
            }
            // lib.optionalAttrs withCUDA {
              CARGO_PROFILE_RELEASE_CODEGEN_UNITS = "16";
              CUDAFORGE_THREADS = cudaforgeThreadsBoundSoParallelNvccFitsInRam;
              CUDA_PATH = "${cudaPackages.cudatoolkit}";
              CUDA_ARCH_LIST = lib.concatStringsSep ";" cudaCapabilities;
              CUDA_COMPUTE_CAP =
                if cudaComputeCap != null then
                  cudaComputeCap
                else
                  builtins.replaceStrings [ "." ] [ "" ] (lib.last cudaCapabilities);
              CUTLASS_DIR = "${cutlass44For pkgs}";
              FLASHINFER_DIR = "${flashinfer064For pkgs}";
              NCCL_ROOT = "${cudaPackages.nccl}";
              CUDNN_ROOT = "${cudaPackages.cudnn}";
              CUDNN_FRONTEND_DIR = "${cudaPackages.cudnn-frontend}";

              CMAKE_CUDA_ARCHITECTURES = lib.concatStringsSep ";" (
                map (c: builtins.replaceStrings [ "." ] [ "" ] c) cudaCapabilities
              );

              CC = "${cudaPackages.backendStdenv.cc}/bin/cc";
              CXX = "${cudaPackages.backendStdenv.cc}/bin/c++";
              "CC_x86_64-unknown-linux-gnu" = "${cudaPackages.backendStdenv.cc}/bin/cc";
              "CXX_x86_64-unknown-linux-gnu" = "${cudaPackages.backendStdenv.cc}/bin/c++";
              HOST_CC = "${cudaPackages.backendStdenv.cc}/bin/cc";
              HOST_CXX = "${cudaPackages.backendStdenv.cc}/bin/c++";
            };

            preConfigure = ''
              # candle-flash-attn -> cudaforge caches git checkouts. HOME is the
              # read-only /homeless-shelter, so redirect both under $TMPDIR,
              # which nix makes per-build on every platform. Not /tmp: darwin
              # builds unsandboxed, so a fixed /tmp path collides with whatever
              # an earlier build left there under a different _nixbld user.
              # Set in preConfigure so phases before preBuild are covered too.
              export HOME="$TMPDIR/build-home"
              export CUDAFORGE_HOME="$TMPDIR/cudaforge-home"
              mkdir -p "$HOME" "$CUDAFORGE_HOME"
            '';

            preBuild = ''
              # Idempotent re-export: preBuild must not depend on preConfigure
              # having run, or an unset CUDAFORGE_HOME roots the seed at /.
              export HOME="$TMPDIR/build-home"
              export CUDAFORGE_HOME="$TMPDIR/cudaforge-home"
              mkdir -p "$HOME" "$CUDAFORGE_HOME"
              # Pre-seed the cutlass checkout that candle-flash-attn requires
              # at the exact commit it pins. The leaveDotGit=true fetchgit
              # preserves .git/, so cudaforge's `git rev-parse HEAD` matches
              # and the cache early-return path skips the (network-needing)
              # clone. dir name = "cutlass-<first 16 chars of commit>".
              CF_CUTLASS_DIR="$CUDAFORGE_HOME/git/checkouts/cutlass-7d49e6c7e2f8896c"
              mkdir -p "$CUDAFORGE_HOME/git/checkouts"
              cp -r --no-preserve=mode,ownership "${cutlassForCudaforge}" "$CF_CUTLASS_DIR"
              chmod -R u+w "$CF_CUTLASS_DIR"
            ''
            + lib.optionalString withCUDA ''
              export LIBRARY_PATH="${cudaPackages.cuda_cudart}/lib/stubs''${LIBRARY_PATH:+:$LIBRARY_PATH}"
              export CC="${cudaPackages.backendStdenv.cc}/bin/cc"
              export CXX="${cudaPackages.backendStdenv.cc}/bin/c++"
              export CC_x86_64_unknown_linux_gnu="${cudaPackages.backendStdenv.cc}/bin/cc"
              export CXX_x86_64_unknown_linux_gnu="${cudaPackages.backendStdenv.cc}/bin/c++"
              export HOST_CC="${cudaPackages.backendStdenv.cc}/bin/cc"
              export HOST_CXX="${cudaPackages.backendStdenv.cc}/bin/c++"
            '';

            doCheck = false;

            meta = with pkgs.lib; {
              description =
                if crate == null then
                  "speaches-plus realtime audio server (Rust)"
                else
                  "${crate} from the speaches-plus workspace";
              mainProgram = if crate == null then "speaches-plus" else crate;
              license = licenses.mit;
              platforms = platforms.unix;
            };
          };

          cudaKernels = import ./nix/cuda-kernels.nix {
            inherit pkgs craneLib commonArgs;
          };
          applicationArgs = commonArgs // {
            env =
              commonArgs.env
              // lib.optionalAttrs withCUDA {
                NV_KERNELS_PREBUILT = "${cudaKernels}";
              };
          };
          cargoArtifacts = craneLib.buildDepsOnly (
            applicationArgs
            // {
              pname = "speaches-plus-workspace";
              cargoExtraArgs =
                "--locked --workspace"
                + lib.optionalString (features != [ ]) (" --features " + lib.concatStringsSep "," features);
              meta = { };
            }
          );
          passthru = {
            inherit
              profile
              modelsHub
              profileEnv
              cargoArtifacts
              ;
          }
          // lib.optionalAttrs withCUDA { inherit cudaKernels; };
        in
        craneLib.buildPackage (
          applicationArgs
          // {
            inherit passthru cargoArtifacts;
            preBuild = applicationArgs.preBuild + ''
              find "''${CARGO_TARGET_DIR:-target}" -path '*/out/build/CMakeCache.txt' -delete
            '';
          }
        );

      mkPythonPackages = system: pkgs: rec {
        nix-hug-lib = nix-hug.lib.${system};

        models = import ./python/nix/models.nix { inherit nix-hug-lib pkgs; };

        hf-cache = nix-hug-lib.buildCache {
          models = with models; [
            kokoroOnnxCommunity
            qwenTts17Cv
            qwenTts06Cv
            qwenTts17Vd
            qwenTts06B
            qwenAligner
            qwenOmni30B
            gemma4E4B
            gemma431B
            whisperGgml
            whisperCt2
            kokoroSpeaches
            smartTurnV3
            diarizen
            wespeaker
          ];
        };

        hf-cache-slim = nix-hug-lib.buildCache {
          models = with models; [
            kokoroOnnxCommunity
            qwenTts17Cv
            qwenTts06Cv
            qwenTts17Vd
            qwenTts06B
            qwenAligner
            gemma4E4B
            gemma4E4Bw4a16
            whisperGgml
            whisperCt2
            kokoroSpeaches
            smartTurnV3
            diarizen
            wespeaker
          ];
        };

        speaches-plus-python-pkg = pkgs.python314Packages.buildPythonPackage {
          pname = "speaches-plus-python";
          version = "0.2.0";
          format = "pyproject";
          src = pkgs.lib.fileset.toSource {
            root = ./python;
            fileset = pkgs.lib.fileset.unions [
              ./python/aligner
              ./python/audio
              ./python/conversation
              ./python/diarization
              ./python/eou
              ./python/inspect_api
              ./python/nano_vllm
              ./python/oapi
              ./python/omni
              ./python/realtime
              ./python/stt
              ./python/tts
              ./python/vad
              ./python/env.py
              ./python/errors.py
              ./python/ids.py
              ./python/otel.py
              ./python/server.py
              ./python/trace.py
              ./python/pyproject.toml
              ./python/LICENSE
              ./python/NOTICE.md
            ];
          };
          nativeBuildInputs = with pkgs.python314Packages; [
            setuptools
            wheel
          ];
          propagatedBuildInputs = with pkgs.python314Packages; [
            fastapi
            uvicorn
            python-multipart
            pydantic
            soundfile
            librosa
            torch
            numpy
            transformers
            accelerate
            huggingface-hub
            pillow
            torchvision
            onnxruntime
            phonemizer
            xxhash
            imageio
            av
            xgrammar
            opuslib
            aiortc
          ];
          dontCheckRuntimeDeps = true;
          doCheck = false;
        };

        serverEnv = pkgs.python314.withPackages (_: [ speaches-plus-python-pkg ]);

        speaches-plus-python = pkgs.writeShellScriptBin "speaches-plus-python" ''
          export PHONEMIZER_ESPEAK_LIBRARY="''${PHONEMIZER_ESPEAK_LIBRARY:-${pkgs.espeak-ng}/lib/libespeak-ng${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}}"
          export ESPEAK_DATA_PATH="''${ESPEAK_DATA_PATH:-${pkgs.espeak-ng}/share/espeak-ng-data}"
          export PATH="${pkgs.espeak-ng}/bin:$PATH"
          export HF_HUB_CACHE="''${HF_HUB_CACHE:-${hf-cache-slim}}"
          export HF_HUB_OFFLINE="''${HF_HUB_OFFLINE:-1}"
          export TRANSFORMERS_OFFLINE="''${TRANSFORMERS_OFFLINE:-1}"
          export KOKORO_VOICES_DIR="''${KOKORO_VOICES_DIR:-${models.kokoroOnnxCommunity}/voices}"
          exec ${serverEnv}/bin/speaches-plus-python "$@"
        '';
      };

      cpuPythonPackages = forAllSystems (system: mkPythonPackages system (mkPkgs system));
      cudaPythonPackages = forAllSystems (system: mkPythonPackages system (mkCudaPkgs system));

      mkDiarizenExporter =
        system:
        let
          pkgs3 = import nixpkgs-pyannote3 { inherit system; };
          diarizenCkpt = cpuPythonPackages.${system}.models.diarizen;
          diarizenSrc = pkgs3.fetchFromGitHub {
            owner = "BUTSpeechFIT";
            repo = "DiariZen";
            rev = "a60b18151dbbe246e4199d8ef5cd2ece3872ea94";
            hash = "sha256-i6IsyJz63vshkXmOqWZjZ+nPHe6i9YKLu5nzE+NtLgA=";
          };
          pyannoteFork = pkgs3.python3Packages.pyannote-audio.overridePythonAttrs (_: {
            version = "3.1.1-diarizen";
            src = diarizenSrc;
            sourceRoot = "${diarizenSrc.name}/pyannote-audio";
            doCheck = false;
            dontCheckRuntimeDeps = true;
            pythonImportsCheck = [ ];
          });
          diarizen = pkgs3.python3Packages.buildPythonPackage {
            pname = "diarizen";
            version = "0.0.1-unstable-2026-06-17";
            format = "pyproject";
            src = diarizenSrc;
            nativeBuildInputs = with pkgs3.python3Packages; [ flit-core ];
            propagatedBuildInputs = with pkgs3.python3Packages; [
              torch
              torchaudio
              einops
              accelerate
              onnx
              pyannoteFork
            ];
            pythonImportsCheck = [ ];
            dontCheckRuntimeDeps = true;
            doCheck = false;
          };
          env = pkgs3.python3.withPackages (_: [
            diarizen
            pkgs3.python3Packages.toml
          ]);
        in
        {
          inherit env;
          onnx =
            pkgs3.runCommand "diarizen-segmentation.onnx"
              {
                nativeBuildInputs = [ env ];
              }
              ''
                export HOME="$TMPDIR" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
                # --chunk-seconds 16 must match DiarConfig.chunk_seconds in the
                # rust/go backends (the ONNX sample axis is static).
                python3 ${./rust/scripts/export-diarizen-onnx.py} --chunk-seconds 16 ${diarizenCkpt} "$out"
              '';
        };

      mkThirdParty =
        pkgs:
        let
          inherit (pkgs) lib;
          thirdParty =
            let
              forge = "https://git.decent.foundation";
              crate =
                {
                  name,
                  version,
                  hash,
                }:
                let
                  tarball = pkgs.fetchurl {
                    urls = [
                      "${forge}/api/packages/mirrors/generic/${name}/${version}/${name}-${version}.crate"
                      "https://static.crates.io/crates/${name}/${name}-${version}.crate"
                    ];
                    inherit hash;
                  };
                  patch = ./rust/third-party/patches + "/${name}-${version}.patch";
                in
                ''
                  tar -xzf ${tarball} -C $out
                  patch -d $out/${name}-${version} -p1 < ${patch}
                '';
              mlx = pkgs.fetchzip {
                urls = [
                  "${forge}/mirrors/mlx/archive/1f8e74e3f12f31365464a6867c6579f0e9b29d85.tar.gz"
                  "https://github.com/ml-explore/mlx/archive/1f8e74e3f12f31365464a6867c6579f0e9b29d85.tar.gz"
                ];
                hash = "sha256-pAQZ1RJq9Mb66qXMdZUt3znNKY+bhpxYckY708i0CmY=";
              };
              mlxDir = ./rust/third-party/mlx-steel-attn;
              mlxFiles = map (l: lib.last (lib.splitString "  " l)) (
                lib.filter (l: l != "") (lib.splitString "\n" (lib.fileContents (mlxDir + "/FILES")))
              );
            in
            pkgs.runCommand "speaches-plus-third-party" { } ''
              mkdir -p $out/mlx-steel-attn
              ${crate {
                name = "wgpu";
                version = "30.0.0";
                hash = "sha256-bY9L1E2S2lJw8DQJ26n5UtqyTxKOBdalVJJhAdG/kRQ=";
              }}
              ${crate {
                name = "wgpu-hal";
                version = "30.0.0";
                hash = "sha256-z3ZRMtjV9Q4ZLniARkiQwT9OdFeq/o5UZugXRYbp8QE=";
              }}
              ${crate {
                name = "candle-core";
                version = "0.11.0";
                hash = "sha256-XsskUJOw95G4nTQgw9+cbUnGCrY7pU24lr+KO69IZwY=";
              }}
              ${crate {
                name = "candle-metal-kernels";
                version = "0.11.0";
                hash = "sha256-JC6Dxqz2ObsnPJKdc8Z6iCu03QihQPEhCW4Zui8hPT4=";
              }}
              for f in ATTRIBUTION LICENSE FILES; do
                cp ${mlxDir}/$f $out/mlx-steel-attn/$f
              done
              ${lib.concatMapStrings (f: ''
                install -Dm644 ${mlx}/${f} $out/mlx-steel-attn/${f}
              '') mlxFiles}
            '';

        in
        thirdParty;

      mkOcrWeb =
        pkgs:
        import ./nix/ocr-web.nix {
          inherit pkgs;
          craneLib = crane.mkLib pkgs;
          src = rustSrc;
          thirdParty = mkThirdParty pkgs;
        };
    in
    {
      devShells = forAllSystems (
        system:
        let
          isLinux = (mkPkgs system).stdenv.hostPlatform.isLinux;
          profileNames = builtins.attrNames profiles;
          mkShells =
            withCUDA: prefix:
            nixpkgs.lib.listToAttrs (
              map (
                p:
                nixpkgs.lib.nameValuePair "${prefix}${p}" (
                  mkShell system {
                    inherit withCUDA;
                    profile = p;
                  }
                )
              ) profileNames
            );

          pyPkgs = mkPkgs system;
          pyAssets = cpuPythonPackages.${system};
          pyModels = pyAssets.models;
          pyHfCache = pyAssets.hf-cache-slim;
          pythonDevShell = pyPkgs.mkShell {
            packages = [
              pyAssets.serverEnv
              pyPkgs.ffmpeg
              pyPkgs.sox
              pyPkgs.curl
              pyPkgs.file
              pyPkgs.espeak-ng
              pyPkgs.ruff
              pyPkgs.ty
              pyPkgs.python314Packages.pyinstrument
              pyPkgs.python314Packages.pybind11
              pyPkgs.python314Packages.setuptools
              pyPkgs.python314Packages.wheel
              pyPkgs.pkg-config
              pyPkgs.cmake
            ];
            buildInputs = [
              pyPkgs.ctranslate2
              pyPkgs.whisper-cpp
              pyPkgs.python314Packages.pybind11
              pyPkgs.stdenv.cc.cc
            ];
            PHONEMIZER_ESPEAK_LIBRARY = "${pyPkgs.espeak-ng}/lib/libespeak-ng${pyPkgs.stdenv.hostPlatform.extensions.sharedLibrary}";
            ESPEAK_DATA_PATH = "${pyPkgs.espeak-ng}/share/espeak-ng-data";
            HF_HUB_CACHE = "${pyHfCache}";
            HF_HUB_OFFLINE = "1";
            TRANSFORMERS_OFFLINE = "1";
            KOKORO_VOICES_DIR = "${pyModels.kokoroOnnxCommunity}/voices";
            VAD_MODEL_FILE = "${pyModels.sileroVad}";
            CT2_INCLUDE_DIR = "${pyPkgs.ctranslate2}/include";
            CT2_LIBRARY_DIR = "${pyPkgs.ctranslate2}/lib";
            WHISPER_INCLUDE_DIR = "${pyPkgs.whisper-cpp}/include";
            WHISPER_LIBRARY_DIR = "${pyPkgs.whisper-cpp}/lib";

            shellHook = ''
              _ln() { ln -sfn "$@" 2>/dev/null || true; }
              for tree in rust/models go/models; do
                [ -e "$tree" ] || continue
                mkdir -p "$tree" "$tree/whisper-ct2" 2>/dev/null || true
                _ln ${pyModels.sileroVad}                                "$tree/silero_vad.onnx"
                _ln ${pyModels.whisperGgml}/ggml-large-v3-turbo.bin       "$tree/ggml-large-v3-turbo.bin"
                for f in config.json model.bin tokenizer.json vocabulary.json preprocessor_config.json; do
                  _ln ${pyModels.whisperCt2}/$f                          "$tree/whisper-ct2/$f"
                done
                _ln ${pyModels.kokoroSpeaches}/model.onnx                 "$tree/kokoro-v1.0.onnx"
                _ln ${pyModels.kokoroSpeaches}/voices.bin                 "$tree/voices.bin"
                _ln ${pyModels.smartTurnV3}/smart-turn-v3.2-gpu.onnx      "$tree/smart-turn-v3.onnx"
                _ln ${pyModels.diarizen}                                  "$tree/diarizen-large-s80-v2"
                _ln ${pyModels.wespeaker}/voxceleb_resnet293_LM.onnx      "$tree/wespeaker-resnet293-LM.onnx"
                ${nixpkgs.lib.optionalString pyPkgs.stdenv.hostPlatform.isLinux ''_ln ${(mkDiarizenExporter system).onnx} "$tree/diarizen-segmentation.onnx"''}
              done
              ${
                if pyPkgs.stdenv.hostPlatform.isLinux then
                  ''
                    export LD_LIBRARY_PATH="${pyPkgs.ctranslate2}/lib:${pyPkgs.whisper-cpp}/lib:$LD_LIBRARY_PATH"
                  ''
                else
                  ""
              }
            '';
          };
        in
        {
          default = mkShell system {
            withCUDA = false;
            profile = "full";
          };
          no-models = mkShell system {
            withCUDA = false;
            profile = null;
          };
          local-hub = mkShell system {
            withCUDA = false;
            profile = "full";
            localHub = true;
          };
          python = pythonDevShell;
        }
        // mkShells false ""
        // nixpkgs.lib.optionalAttrs isLinux (
          {
            cuda = mkShell system {
              withCUDA = true;
              profile = "full";
            };
            cuda-no-models = mkShell system {
              withCUDA = true;
              profile = null;
            };
          }
          // mkShells true "cuda-"
        )
      );

      apps = forAllSystems (
        system:
        let
          pkgs = mkPkgs system;
          fetchModelsApp =
            let
              script = pkgs.writeShellApplication {
                name = "fetch-models";
                runtimeInputs = [
                  pkgs.bash
                  pkgs.curl
                  pkgs.coreutils
                ];
                text = ''
                  exec bash ${./rust/scripts/fetch-models.sh} "$@"
                '';
              };
            in
            {
              type = "app";
              program = "${script}/bin/fetch-models";
            };
        in
        {
          strip-comments = mkStripCommentsApp system;
          fetch-models = fetchModelsApp;
          default = mkStripCommentsApp system;
        }
      );

      packages = forAllSystems (
        system:
        let
          pkgs = mkPkgs system;
          isLinux = pkgs.stdenv.hostPlatform.isLinux;
          profileNames = builtins.attrNames profiles;
          mkPkg =
            withCUDA: profile:
            mkSpeachesPlus {
              inherit
                pkgs
                system
                withCUDA
                profile
                ;
            };
          perProfile = nixpkgs.lib.listToAttrs (
            map (p: nixpkgs.lib.nameValuePair "speaches-plus-${p}" (mkPkg false p)) profileNames
          );
          perProfileCuda = nixpkgs.lib.optionalAttrs isLinux (
            nixpkgs.lib.listToAttrs (
              map (p: nixpkgs.lib.nameValuePair "speaches-plus-cuda-${p}" (mkPkg true p)) profileNames
            )
          );
          mkPkgVulkan =
            profile:
            mkSpeachesPlus {
              inherit pkgs system profile;
              withVulkan = true;
            };
          perProfileVulkan = nixpkgs.lib.optionalAttrs isLinux (
            nixpkgs.lib.listToAttrs (
              map (p: nixpkgs.lib.nameValuePair "speaches-plus-vulkan-${p}" (mkPkgVulkan p)) profileNames
            )
          );
          hubs = nixpkgs.lib.listToAttrs (
            map (
              p: nixpkgs.lib.nameValuePair "speaches-models-hub-${p}" (mkSpeachesModelsHub system p)
            ) profileNames
          );
          py = cpuPythonPackages.${system};
          pyCuda = cudaPythonPackages.${system};

          diarizenExporter = mkDiarizenExporter system;
        in
        {
          speaches-plus = mkPkg false "full";
          nv-sae = mkSpeachesPlus {
            inherit pkgs system;
            withCUDA = false;
            profile = null;
            crate = "nv-sae";
          };
          nv-sae-cuda = mkSpeachesPlus {
            inherit pkgs system;
            withCUDA = true;
            profile = null;
            crate = "nv-sae";
          };
          ocr-web = mkOcrWeb pkgs;
          speaches-plus-go = mkSpeachesPlusGo {
            inherit pkgs;
            withCUDA = false;
          };
          default = mkPkg false "full";

          inherit (py) speaches-plus-python;
          inherit (py) speaches-plus-python-pkg;
          inherit (py) hf-cache;
        }
        // perProfile
        // hubs
        // nixpkgs.lib.optionalAttrs isLinux (
          {
            speaches-plus-cuda = mkPkg true "full";
            nv-cuda-kernels = (mkPkg true null).cudaKernels;

            diarizen-onnx-exporter = diarizenExporter.env;
            diarizen-segmentation-onnx = diarizenExporter.onnx;
            speaches-plus-go-cuda = mkSpeachesPlusGo {
              inherit pkgs;
              withCUDA = true;
            };
            speaches-plus-python-cuda = pyCuda.speaches-plus-python;
            speaches-plus-python-pkg-cuda = pyCuda.speaches-plus-python-pkg;

            speaches-plus-vulkan = mkPkgVulkan "full";
          }
          // perProfileCuda
          // perProfileVulkan
        )
      );

      lib = forAllSystems (system: {
        inherit mkSpeachesPlus mkSpeachesPlusGo;
        mkSpeachesHubCache = mkSpeachesModelsHub system;
        inherit profiles;
        models = mkModels system;
      });
    };
}
