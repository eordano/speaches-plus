{ nix-hug-lib, pkgs }:
let
  gitRepoHashes = builtins.fromJSON (builtins.readFile ./model-githashes.json);

  modelDefs = {
    kokoroOnnxCommunity = {
      url = "onnx-community/Kokoro-82M-v1.0-ONNX";
      rev = "1939ad2a8e416c0acfeecc08a694d14ef25f2231";
      fileTreeHash = "sha256-vfJB5oWDslMm0789t4I7TEPO0xFGaLo8HNGdCNJZ4JE=";
    };

    qwenTts17Cv = {
      url = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice";
      rev = "0c0e3051f131929182e2c023b9537f8b1c68adfe";
      fileTreeHash = "sha256-tyQxzQ6kNL87xXHgINLKo6fhpk9iuR1dRTUDPztJlJc=";
    };

    qwenTts06Cv = {
      url = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice";
      rev = "85e237c12c027371202489a0ec509ded67b5e4b5";
      fileTreeHash = "sha256-KKIURDkYeRgDUViVRBJwo2N02zA7T3lpRSWTEXcKFdk=";
    };

    qwenTts17Vd = {
      url = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign";
      rev = "5ecdb67327fd37bb2e042aab12ff7391903235d3";
      fileTreeHash = "sha256-Dp46by0MQ9/oRpBM2PeO70bSoHyxKYcWRgNdYDB/pL8=";
    };

    qwenTts06B = {
      url = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";
      rev = "5d83992436eae1d760afd27aff78a71d676296fc";
      fileTreeHash = "sha256-GKWzpwo3PApyZXPkqzI9lgjV3g+i+G21wFBxcFSvljc=";
    };

    qwenAligner = {
      url = "Qwen/Qwen3-ForcedAligner-0.6B";
      rev = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7";
      fileTreeHash = "sha256-FAVTYzlkTvjkXotsQljzpHYx9fJhAKrDP3FN4ligkL0=";
    };

    qwenOmni30B = {
      url = "Qwen/Qwen3-Omni-30B-A3B-Instruct";
      rev = "26291f793822fb6be9555850f06dfe95f2d7e695";
      fileTreeHash = "sha256-A4Q5OeXk7tfSzBhmy2ZrdrwJoCweW1pa7/cqe1K7xFc=";
    };

    gemma4E4Bw4a16 = {
      url = "google/gemma-4-E4B-it-qat-w4a16-ct";
      rev = "6cd26aaa2357fb2bad8c51699a7558a4d1a965bb";
      fileTreeHash = "sha256-JWmGAOuoGEwMgjiYgBD5PkXviUuUidUK5IIDTetUi8c=";
    };

    gemma4E4B = {
      url = "google/gemma-4-E4B-it";
      rev = "3555bddc93a623db8887dd2e52123facc45ade77";
      fileTreeHash = "sha256-4uPmNOiA3VGtvLR8LygSXe2AmelMKOuhDGvSylPc9M0=";
    };

    gemma431B = {
      url = "google/gemma-4-31B-it";
      rev = "ba74f5b6c647c0911554e50278d6f6f4477f9010";
      fileTreeHash = "sha256-nRsFyVTqn/di0QBrpCkaC5I5z51AgfAzRyfg5iZa+II=";
    };

    whisperGgml = {
      url = "ggerganov/whisper.cpp";
      rev = "5359861c739e955e79d9a303bcbc70fb988958b1";
      fileTreeHash = "sha256-YKu/+tF/eWjkFjdD5AspIry8qtlakZ+lmmInscG1dcs=";
      filters = {
        files = [ "ggml-large-v3-turbo.bin" ];
      };
    };

    whisperCt2 = {
      url = "deepdml/faster-whisper-large-v3-turbo-ct2";
      rev = "4df90f75321148c3a29a9e2351b7ddf8f5b115a8";
      fileTreeHash = "sha256-4VYZIYulFY9PlgKDlIyuLT+TJObMeXk9eHlHuEvQqtY=";
    };

    kokoroSpeaches = {
      url = "speaches-ai/Kokoro-82M-v1.0-ONNX";
      rev = "dc196c76d64fed9203906231372bcb98135815df";
      fileTreeHash = "sha256-+Aea1c28vvS+pfOs2alshOajGzW6I7ujDVIIAQ5KlgI=";
    };

    smartTurnV3 = {
      url = "pipecat-ai/smart-turn-v3";
      rev = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93";
      fileTreeHash = "sha256-oh3MdbkVbYGSQUUvuA1c7DMMnu1naPKoDBA+aXjNDyo=";
      filters = {
        files = [
          "smart-turn-v3.2-cpu.onnx"
          "smart-turn-v3.2-gpu.onnx"
        ];
      };
    };

    diarizen = {
      url = "BUT-FIT/diarizen-wavlm-large-s80-md-v2";
      rev = "f27b9ffbedcf422856d104ecee9b94be37ea578e";
      fileTreeHash = "sha256-q8q1I6jFLQpKEbqMTd6d/LUjLcfJcop11rILS/aLwdk=";
    };

    wespeaker = {
      url = "Wespeaker/wespeaker-voxceleb-resnet293-LM";
      rev = "6e6bffe5bf3d772a1f143dc6dbfea58a0799ea83";
      fileTreeHash = "sha256-eTl4/kPqPiVhq28Q4v4J+8g1g9muvFftp94zJO0DXFM=";
    };
  };

  models = builtins.mapAttrs (
    name: def: nix-hug-lib.fetchModel (def // { gitRepoHash = gitRepoHashes.${name}; })
  ) modelDefs;
in
models
// {
  sileroVad = pkgs.fetchurl {
    url = "https://github.com/snakers4/silero-vad/raw/v6.2/src/silero_vad/data/silero_vad.onnx";
    sha256 = "1qw8hyfjfrac2xz2ns4895dv5pp8hndnyzg6jhm2k7jhyhi3l58s";
  };
}
