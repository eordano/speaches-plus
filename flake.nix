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
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);

      cudaCapabilities = [
        "8.9"
        "12.0"
      ];

      unbreakDlinfoOverlay = final: prev: {
        python3Packages = prev.python3Packages.override {
          overrides =
            pyFinal: pyPrev:
            {
              dlinfo = pyPrev.dlinfo.overridePythonAttrs (_: {
                meta.broken = false;
                doCheck = false;
                doInstallCheck = false;
              });
            }
            // nixpkgs.lib.optionalAttrs prev.stdenv.isDarwin {
              requests-futures = pyPrev.requests-futures.overridePythonAttrs (_: {
                doCheck = false;
                doInstallCheck = false;
              });
              aiohttp = pyPrev.aiohttp.overridePythonAttrs (_: {
                doCheck = false;
                doInstallCheck = false;
              });
              geoip2 = pyPrev.geoip2.overridePythonAttrs (_: {
                doCheck = false;
                doInstallCheck = false;
              });
              pyarrow = pyPrev.pyarrow.overridePythonAttrs (old: {
                disabledTests = (old.disabledTests or [ ]) ++ [ "test_timezone_absent" ];
              });
              httplib2 = pyPrev.httplib2.overridePythonAttrs (old: {
                disabledTests = (old.disabledTests or [ ]) ++ [ "test_socks5_auth" ];
              });
            };
        };

        pythonPackagesExtensions =
          (prev.pythonPackagesExtensions or [ ])
          ++ nixpkgs.lib.optionals prev.stdenv.isDarwin [
            (_pyFinal: pyPrev: {

              httpcore2 = pyPrev.httpcore2.overridePythonAttrs (_: {
                doCheck = false;
                doInstallCheck = false;
              });

              google-api-core = pyPrev.google-api-core.overridePythonAttrs (old: {
                disabledTests = (old.disabledTests or [ ]) ++ [ "test_apply_passthrough" ];
              });

              nltk = pyPrev.nltk.overridePythonAttrs (_: {
                doCheck = false;
                doInstallCheck = false;
              });
            })
          ];
      };

      ffmpegResignOverlay =
        final: prev:
        nixpkgs.lib.optionalAttrs prev.stdenv.isDarwin {
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

      arrowCppDarwinOverlay =
        final: prev:
        nixpkgs.lib.optionalAttrs prev.stdenv.isDarwin {
          arrow-cpp = prev.arrow-cpp.overrideAttrs (_: {
            installCheckPhase = ''
              runHook preInstallCheck
              ctest -L unittest --exclude-regex '^(arrow-flight-test|arrow-gcsfs-test|arrow-flight-integration-test|arrow-orc-adapter-test|arrow-flight-internals-test|arrow-flight-sql-test|arrow-azurefs-test|parquet-encryption-test)$'
              runHook postInstallCheck
            '';
          });
        };

      cudaSetupHookNvccFixOverlay =
        final: prev:
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

      mkPkgs =
        system:
        import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            nvidia.acceptLicense = true;
            cudaCapabilities = cudaCapabilities;
            cudaForwardCompat = true;
          };
          overlays = [
            unbreakDlinfoOverlay
            ffmpegResignOverlay
            arrowCppDarwinOverlay
            cudaSetupHookNvccFixOverlay
          ];
        };

      mkCudaPkgs =
        system:
        import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            nvidia.acceptLicense = true;
            cudaSupport = true;
            cudaCapabilities = cudaCapabilities;
            cudaForwardCompat = true;
          };
          overlays = [
            unbreakDlinfoOverlay
            ffmpegResignOverlay
            arrowCppDarwinOverlay
            cudaSetupHookNvccFixOverlay
            onnxruntimeNoNcclOverlay
          ];
        };

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
            rev = "e850c696e6d75f965367e816c16bc7dacd955ffa";
            fileTreeHash = "sha256-3boLVG/JJWfX/f7jyd+3AHf718vNElLqRgl+XRa5KQk=";
            filters.exclude = [
              "model_visual\\.safetensors"
              "model_mtp\\.safetensors"
            ];
            envVars = [ "NV_CHAT_MODEL_DIR" ];
          };

          qwen36-mm = {
            url = "RedHatAI/Qwen3.6-35B-A3B-NVFP4";
            rev = "e850c696e6d75f965367e816c16bc7dacd955ffa";
            fileTreeHash = "sha256-3boLVG/JJWfX/f7jyd+3AHf718vNElLqRgl+XRa5KQk=";
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
            rev = "16b6615af3548b88e2d8e382457bc705b00479cf";
            fileTreeHash = "sha256-YJUUMm5pKLqdRgU8crDO+/Y5OYG0cauil4hiDl6UfuY=";
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
            rev = "e5ef03afa233c35cb000323ff098d4291e1dd07c";
            fileTreeHash = "sha256-oPR/XnakSFTZENuL2yxiWrKukAhxUvMsS8O5rx+zo4U=";
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
            url = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice";
            rev = "85e237c12c027371202489a0ec509ded67b5e4b5";
            fileTreeHash = "sha256-KKIURDkYeRgDUViVRBJwo2N02zA7T3lpRSWTEXcKFdk=";
            envVars = [ "NV_TTS_TALKER_DIR" ];
          };

          qwen3-tts-base = {
            url = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";
            rev = "5d83992436eae1d760afd27aff78a71d676296fc";
            fileTreeHash = "sha256-GKWzpwo3PApyZXPkqzI9lgjV3g+i+G21wFBxcFSvljc=";
            envVars = [ "NV_TTS_BASE_TALKER_DIR" ];
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
        tts-only = [
          "qwen3-tts"
          "qwen3-tts-base"
        ];
        tts-clone = [
          "qwen3-tts-base"
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
          names = profiles.${profileName} or (throw "unknown profile: ${profileName}");
        in
        nix-hug-lib.buildCache {
          models = map (n: models.${n}.drv) names;
        };

      mkProfileEnv =
        system: profileName: hub:
        let
          models = mkModels system;
          names = profiles.${profileName} or (throw "unknown profile: ${profileName}");
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
          names = profiles.${profileName} or (throw "unknown profile: ${profileName}");
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

          cutlass-44 = pkgs.fetchFromGitHub {
            owner = "NVIDIA";
            repo = "cutlass";
            tag = "v4.4.2";
            hash = "sha256-0q9Ad0Z6E/rO2PdM4uQc8H0E0qs9uKc3reHepiHhjEc=";
          };

          flashinfer-064 = pkgs.fetchFromGitHub {
            owner = "flashinfer-ai";
            repo = "flashinfer";
            tag = "v0.6.4";
            fetchSubmodules = true;
            hash = "sha256-Hq3oTeEJHRvXwThI8vc06E3Ot/FnPP0sZUfze3ISa2o=";
          };

          cutlassForCudaforge = pkgs.fetchgit {
            url = "https://github.com/NVIDIA/cutlass.git";
            rev = "7d49e6c7e2f8896c47f586706e67e1fb215529dc";
            hash = "sha256-cSWVzyuDC8EidTAZzHbVz0jUNK4zx5AAwfUV6lUXTXs=";
            leaveDotGit = true;
            deepClone = false;
          };

          ctranslate2 =
            if withCUDA then
              pkgs.ctranslate2.override {
                inherit cudaPackages;
                withCUDA = true;
                withCuDNN = true;
              }
            else
              pkgs.ctranslate2;

          whisper-cpp =
            if withCUDA then
              pkgs.whisper-cpp.override {
                inherit cudaPackages;
                cudaSupport = true;
              }
            else
              pkgs.whisper-cpp;

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
              cuda_cccl
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
              go
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
              ccache
            ]
            ++ runtimeLibs
            ++ pkgs.lib.optionals isLinux [
              pkgs.stdenv.cc.cc.lib
              pkgs.libcxx
            ]
            ++ pkgs.lib.optionals withCUDA (cudaLibs ++ [ cudaPackages.cuda_nvcc ]);

          env = {
            LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
            ORT_DYLIB_PATH = "${pkgs.onnxruntime}/lib/libonnxruntime${if isLinux then ".so" else ".dylib"}";
            ONNXRUNTIME_LIB = "${pkgs.onnxruntime}/lib/libonnxruntime${if isLinux then ".so" else ".dylib"}";
            ESPEAK_DATA_PATH = "${pkgs.espeak-ng}/share/espeak-ng-data";
            CT2_DEVICE = if withCUDA then "cuda" else "cpu";
            CUDA_PATH = pkgs.lib.optionalString withCUDA "${cudaPackages.cudatoolkit}";
            CUDA_ARCH_LIST = pkgs.lib.optionalString withCUDA (pkgs.lib.concatStringsSep ";" cudaCapabilities);
            CUDA_COMPUTE_CAP = pkgs.lib.optionalString withCUDA (
              builtins.replaceStrings [ "." ] [ "" ] (pkgs.lib.last cudaCapabilities)
            );
            CUTLASS_DIR = pkgs.lib.optionalString withCUDA "${cutlass-44}";
            NCCL_ROOT = pkgs.lib.optionalString withCUDA "${cudaPackages.nccl}";
            CUDNN_ROOT = pkgs.lib.optionalString withCUDA "${cudaPackages.cudnn}";
            CUDNN_FRONTEND_DIR = pkgs.lib.optionalString withCUDA "${cudaPackages.cudnn-frontend}";

            FLASHINFER_DIR = pkgs.lib.optionalString withCUDA "${flashinfer-064}";

            CMAKE_CUDA_ARCHITECTURES = pkgs.lib.optionalString withCUDA (
              pkgs.lib.concatStringsSep ";" (map (c: builtins.replaceStrings [ "." ] [ "" ] c) cudaCapabilities)
            );
            CMAKE_POLICY_VERSION_MINIMUM = "3.5";
            CFLAGS = pkgs.lib.optionalString (
              !isLinux
            ) "-Wno-elaborated-enum-base -Wno-error=elaborated-enum-base";
            CXXFLAGS = pkgs.lib.optionalString (
              !isLinux
            ) "-Wno-elaborated-enum-base -Wno-error=elaborated-enum-base";
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
            install -m 0755 "${self}/rust/scripts/cmake-install-shim.sh" "$SPEACHES_CMAKE_SHIM_DIR/cmake"
            export PATH="$SPEACHES_CMAKE_SHIM_DIR:$PATH"
          ''
          + pkgs.lib.optionalString isLinux ''
            export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          ''
          + pkgs.lib.optionalString (!isLinux) ''
            export CC="${pkgs.stdenv.cc}/bin/cc"
            export CXX="${pkgs.stdenv.cc}/bin/c++"
          ''
          + pkgs.lib.optionalString withCUDA ''
            export LIBRARY_PATH="${cudaPackages.cuda_cudart}/lib/stubs''${LIBRARY_PATH:+:$LIBRARY_PATH}"
            if [ -d "$HOME/.cargo/registry/src" ]; then
              bash "${self}/rust/scripts/patch-ct2rs-cuda.sh" >/dev/null 2>&1 || true
              bash "${self}/rust/scripts/patch-ct2rs-cmake-min.sh" >/dev/null 2>&1 || true
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
              exec python3 ${self}/scripts/strip-comments.py "$@"
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
          inherit (pkgs) lib;
          isLinux = pkgs.stdenv.hostPlatform.isLinux;

          ctranslate2 =
            if withCUDA then
              pkgs.ctranslate2.override {
                inherit cudaPackages;
                withCUDA = true;
                withCuDNN = true;
              }
            else
              pkgs.ctranslate2;

          whisper-cpp =
            if withCUDA then
              pkgs.whisper-cpp.override {
                inherit cudaPackages;
                cudaSupport = true;
              }
            else
              pkgs.whisper-cpp;

          ortLib = "${pkgs.onnxruntime}/lib/libonnxruntime${if isLinux then ".so" else ".dylib"}";
        in
        pkgs.buildGoModule {
          pname = "speaches-plus-go";
          version = "0.0.1";
          src = ./.;
          modRoot = "go";
          vendorHash = "sha256-bqooFaAFeabBSL0Hgt+fPlAQPWdugA1Zy4iKVj+WX28=";
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
        }:
        let
          inherit (pkgs) lib;
          isLinux = pkgs.stdenv.hostPlatform.isLinux;
          modelsHub = if profile != null then mkSpeachesModelsHub system profile else null;
          profileEnv = if profile != null then mkProfileEnv system profile modelsHub else { };

          cutlass-44 = pkgs.fetchFromGitHub {
            owner = "NVIDIA";
            repo = "cutlass";
            tag = "v4.4.2";
            hash = "sha256-0q9Ad0Z6E/rO2PdM4uQc8H0E0qs9uKc3reHepiHhjEc=";
          };

          flashinfer-064 = pkgs.fetchFromGitHub {
            owner = "flashinfer-ai";
            repo = "flashinfer";
            tag = "v0.6.4";
            fetchSubmodules = true;
            hash = "sha256-Hq3oTeEJHRvXwThI8vc06E3Ot/FnPP0sZUfze3ISa2o=";
          };

          cutlassForCudaforge = pkgs.fetchgit {
            url = "https://github.com/NVIDIA/cutlass.git";
            rev = "7d49e6c7e2f8896c47f586706e67e1fb215529dc";
            hash = "sha256-cSWVzyuDC8EidTAZzHbVz0jUNK4zx5AAwfUV6lUXTXs=";
            leaveDotGit = true;
            deepClone = false;
          };

          ctranslate2 =
            if withCUDA then
              pkgs.ctranslate2.override {
                inherit cudaPackages;
                withCUDA = true;
                withCuDNN = true;
              }
            else
              pkgs.ctranslate2;

          whisper-cpp =
            if withCUDA then
              pkgs.whisper-cpp.override {
                inherit cudaPackages;
                cudaSupport = true;
              }
            else
              pkgs.whisper-cpp;

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

          ortLib = "${pkgs.onnxruntime}/lib/libonnxruntime${if isLinux then ".so" else ".dylib"}";

          craneLib =
            let
              base = crane.mkLib pkgs;
            in
            if withCUDA then
              base.overrideScope (_final: _prev: { stdenvSelector = _: cudaPackages.backendStdenv; })
            else
              base;

          cargoVendorDir = craneLib.vendorCargoDeps {
            src = ./rust;
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
            pname = "speaches-plus";
            version = "0.0.1";
            src = ./rust;

            # mkDummySrc stubs every path-dependency crate, but
            # third-party/wgpu-hal-30.0.0 is a [patch.crates-io] override that
            # wgpu-core compiles against during buildDepsOnly: a stubbed lib.rs
            # there fails the whole deps build on unresolved hal imports, so the
            # real sources are restored into the dummy tree.
            extraDummyScript = ''
              rm -rf $out/third-party
              cp -r --no-preserve=mode,ownership ${./rust/third-party} $out/third-party
            '';

            inherit cargoVendorDir;

            cargoExtraArgs =
              "--locked"
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
              ++ lib.optionals withVulkan [ pkgs.vulkan-loader ]
              ++ lib.optionals withCUDA (
                with cudaPackages;
                [
                  cuda_cudart
                  cuda_cccl
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
              CFLAGS = "-Wno-elaborated-enum-base -Wno-error=elaborated-enum-base";
              CXXFLAGS = "-Wno-elaborated-enum-base -Wno-error=elaborated-enum-base";
              BINDGEN_EXTRA_CLANG_ARGS = "-Wno-elaborated-enum-base";
            }
            // lib.optionalAttrs withCUDA {
              CUDA_PATH = "${cudaPackages.cudatoolkit}";
              CUDA_ARCH_LIST = lib.concatStringsSep ";" cudaCapabilities;
              CUDA_COMPUTE_CAP = builtins.replaceStrings [ "." ] [ "" ] (lib.last cudaCapabilities);
              CUTLASS_DIR = "${cutlass-44}";
              FLASHINFER_DIR = "${flashinfer-064}";
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
              description = "speaches-plus realtime audio server (Rust)";
              mainProgram = "speaches-plus";
              license = licenses.mit;
              platforms = platforms.unix;
            };
          };

          passthru = {
            inherit profile modelsHub profileEnv;
          };
        in
        craneLib.buildPackage (
          commonArgs
          // {
            inherit passthru;
            cargoArtifacts = craneLib.buildDepsOnly commonArgs;
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

        speaches-plus-python-pkg = pkgs.python3Packages.buildPythonPackage {
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
          nativeBuildInputs = with pkgs.python3Packages; [
            setuptools
            wheel
          ];
          propagatedBuildInputs = with pkgs.python3Packages; [
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

        serverEnv = pkgs.python3.withPackages (_: [ speaches-plus-python-pkg ]);

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
              pyPkgs.python3Packages.pyinstrument
              pyPkgs.python3Packages.pybind11
              pyPkgs.python3Packages.setuptools
              pyPkgs.python3Packages.wheel
              pyPkgs.pkg-config
              pyPkgs.cmake
            ];
            buildInputs = [
              pyPkgs.ctranslate2
              pyPkgs.whisper-cpp
              pyPkgs.python3Packages.pybind11
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
              # Symlinks below may already exist with read-only nix-store
              # targets; ln -sfn errors with EPERM rather than overwriting.
              # Suppress noise -- if the symlink target is already pointing at
              # the right /nix/store path, the test runners will find the file.
              _ln() { ln -sfn "$@" 2>/dev/null || true; }
              for tree in rust/models go/models; do
                [ -e "$tree" ] || continue  # skip if speaches-plus checkout missing
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
              done
              # Linux-only: ct2_bindings/_ct2.so + whisper_bindings/_whisper.so
              # don't have rpath baked in (setuptools doesn't auto-emit -Wl,-rpath
              # for nix store paths). Without LD_LIBRARY_PATH the loader fails
              # at import time with "libctranslate2.so.4: cannot open shared
              # object file". On macOS this isn't needed -- dylibs encode their
              # absolute install_name and the loader finds them by id.
              ${
                if pyPkgs.stdenv.isLinux then
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
                  exec bash ${self}/rust/scripts/fetch-models.sh "$@"
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

          diarizenExporter =
            let
              pkgs3 = import nixpkgs-pyannote3 { inherit system; };
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
                    python3 ${./rust/scripts/export-diarizen-onnx.py} --chunk-seconds 16 ${py.models.diarizen} "$out"
                  '';
            };
        in
        {
          speaches-plus = mkPkg false "full";
          speaches-plus-go = mkSpeachesPlusGo {
            inherit pkgs;
            withCUDA = false;
          };
          default = mkPkg false "full";

          speaches-plus-python = py.speaches-plus-python;
          speaches-plus-python-pkg = py.speaches-plus-python-pkg;
          hf-cache = py.hf-cache;
        }
        // perProfile
        // hubs
        // nixpkgs.lib.optionalAttrs isLinux (
          {
            speaches-plus-cuda = mkPkg true "full";

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
