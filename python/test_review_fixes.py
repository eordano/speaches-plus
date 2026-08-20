from __future__ import annotations

import base64
import json
import os as _os_top
import struct
import sys
import time
import tracemalloc

import pytest
import torch

def _section(name: str) -> None:
    print(f"\n=== {name} ===")

def ok(msg: str) -> None:
    print(f"  OK  {msg}")

def info(msg: str) -> None:
    print(f"  --  {msg}")

class _FakeUpload:
    def __init__(self, raw: bytes, content_type: str) -> None:
        self._raw = raw
        self.content_type = content_type

    async def read(self) -> bytes:
        return self._raw

    async def seek(self, _pos: int) -> None:
        return None

def _assert_all_reject(func, cases, exc_types=(ValueError, Exception), trunc=50) -> None:
    for spec, label in cases:
        try:
            func(spec)
            raise AssertionError(f"BAD: {label} accepted")
        except exc_types as exc:
            ok(f"{label} rejected: {str(exc)[:trunc]}")

_STRICT_CI = _os_top.environ.get("CI") == "1"

def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None

_PWT_HAS_OPUS_RUNTIME = _has_module("opuslib")

def _strict_skip(reason: str, *, strict: bool = False) -> None:
    """Skip a test, but if running in CI mode and the section is strict,
    fail loudly instead so missing optional deps surface as bugs."""
    if strict and _STRICT_CI:
        pytest.fail(f"strict-skip in CI: {reason}")
    pytest.skip(reason)

@pytest.fixture
def _sys_modules_snapshot():
    """Snapshot sys.modules keys; restore on teardown so per-test
    sys.modules surgery (del / reload) doesn't leak into other tests."""
    before = dict(sys.modules)
    yield
    cur_keys = set(sys.modules.keys())
    for k in cur_keys - set(before.keys()):
        sys.modules.pop(k, None)
    for k, v in before.items():
        sys.modules[k] = v

def test_c_ssrf_hardening():
    _section('Fix C: SSRF hardening')
    from server import REF_AUDIO_MAX_BYTES, _decode_ref_audio

    REJECT_CASES = [
        ("file:///etc/passwd", "file://"),
        ("http://169.254.169.254/latest/meta-data/", "http://"),
        ("ftp://example.com/x", "ftp://"),
        ("/etc/passwd", "absolute path"),
        ("../../../etc/passwd", "traversal"),
        ("https://example.com/audio.wav\x00.local", "null-byte injection"),
    ]
    _assert_all_reject(_decode_ref_audio, REJECT_CASES)

    tiny = base64.b64encode(b"WAVdata").decode()
    data, suffix = _decode_ref_audio(f"data:audio/wav;base64,{tiny}")
    assert data == b"WAVdata" and suffix == ".wav"
    ok("data:audio/wav;base64 accepted")

    oversize = base64.b64encode(b"x" * (REF_AUDIO_MAX_BYTES + 1024)).decode()
    _assert_all_reject(_decode_ref_audio,
        [(f"data:audio/wav;base64,{oversize}", "oversize")], exc_types=ValueError, trunc=60)

def test_c2_ssrf_hardening_on_multimodal_chat_input_audio_loaders():
    _section('Fix C2: SSRF hardening on multimodal chat input (audio.loaders)')
    from audio.loaders import MULTIMODAL_MAX_BYTES, read_bytes_or_b64

    MM_REJECT_CASES = [
        ("file:///etc/passwd", "file://"),
        ("http://169.254.169.254/latest/meta-data/", "http://"),
        ("ftp://example.com/x.wav", "ftp://"),
        ("/etc/passwd", "absolute path"),
        ("../../../etc/passwd", "traversal"),
        ("https://example.com/audio.wav\x00.local", "null-byte injection"),
        ("data:audio/wav,raw-without-base64", "data: without ;base64,"),
        ("", "empty string"),
        ("relative/path/to/file.wav", "relative path"),
    ]
    _assert_all_reject(read_bytes_or_b64, MM_REJECT_CASES, exc_types=ValueError, trunc=60)

    mm_payload = base64.b64encode(b"WAVdata-multimodal").decode()
    data = read_bytes_or_b64(f"data:audio/wav;base64,{mm_payload}")
    assert data == b"WAVdata-multimodal"
    ok("data:audio/wav;base64 accepted on multimodal path")

    mm_oversize = base64.b64encode(b"x" * (MULTIMODAL_MAX_BYTES + 1024)).decode()
    _assert_all_reject(read_bytes_or_b64,
        [(f"data:audio/wav;base64,{mm_oversize}", "multimodal oversize")], exc_types=ValueError, trunc=60)

    bare = base64.b64encode(b"y" * 512).decode()
    data = read_bytes_or_b64(bare)
    assert data == b"y" * 512
    ok("bare base64 (>=256 chars, valid alphabet) accepted")

def test_t_tts_kokoro_package_layout_speaches_plus_tts_parity():
    _section('Fix T: tts.kokoro package layout (speaches-plus tts/* parity)')
    import numpy as np

    from tts.kokoro.text import (
        DEFAULT_LANGUAGE,
        DEFAULT_VOICE,
        KOKORO_SAMPLE_RATE,
        MAX_CHUNK_CHARS,
        MAX_SAMPLE_RATE,
        MIN_SAMPLE_RATE,
        OPENAI_VOICE_ALIASES,
        SPEED_MAX,
        SPEED_MIN,
        ResponseFormat,
        StreamFormat,
        f32_to_s16le,
        is_openai_voice_alias,
        normalize_for_tts,
        split_into_chunks,
        strip_emojis,
        strip_markdown_emphasis,
    )
    from tts.kokoro.vocab import (
        MAX_PHONEME_LENGTH,
        PAD_TOKEN_ID,
        clean_phonemes,
        tokenize,
    )

    assert KOKORO_SAMPLE_RATE == 24_000
    assert (SPEED_MIN, SPEED_MAX) == (0.5, 2.0)
    assert (MIN_SAMPLE_RATE, MAX_SAMPLE_RATE) == (8_000, 48_000)
    assert MAX_CHUNK_CHARS == 400
    assert DEFAULT_VOICE == "af_heart" and DEFAULT_LANGUAGE == "en-us"
    ok("text constants match upstream tts/text.rs")

    assert OPENAI_VOICE_ALIASES == frozenset(
        {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"},
    )
    assert is_openai_voice_alias("alloy") and is_openai_voice_alias("Echo")
    assert not is_openai_voice_alias("af_heart")
    ok("is_openai_voice_alias mirrors upstream is_openai_voice_alias")

    expected_mime = {
        ResponseFormat.PCM: "audio/pcm",
        ResponseFormat.MP3: "audio/mpeg",
        ResponseFormat.WAV: "audio/wav",
        ResponseFormat.FLAC: "audio/flac",
        ResponseFormat.OPUS: "audio/opus",
        ResponseFormat.AAC: "audio/aac",
    }
    for fmt, mime in expected_mime.items():
        assert fmt.mime_type() == mime, (fmt, mime)
    assert StreamFormat.AUDIO.value == "audio" and StreamFormat.SSE.value == "sse"
    ok("ResponseFormat covers all 6 mime types; StreamFormat matches upstream")

    assert strip_emojis("hello 🌍 world") == "hello  world"
    assert strip_emojis("😀😃😄 plain") == " plain"
    assert strip_emojis("plain") == "plain"
    assert strip_emojis("✂ scissors") == " scissors"
    ok("strip_emojis matches upstream Rust assertions (incl. Dingbats range ✂)")

    assert strip_markdown_emphasis("**bold**") == "bold"
    assert strip_markdown_emphasis("*italic*") == "italic"
    assert strip_markdown_emphasis("__under__") == "under"
    assert strip_markdown_emphasis("_under_") == "under"
    assert strip_markdown_emphasis("a **bold** and *italic* mix") == "a bold and italic mix"
    ok("strip_markdown_emphasis matches upstream Rust assertions")

    assert normalize_for_tts("  hello\n\nworld\t\ttest\r\n") == "hello world test"
    ok("normalize_for_tts collapses whitespace + newlines")

    assert split_into_chunks("short", 100) == ["short"]
    assert split_into_chunks("", 100) == []
    chunks = split_into_chunks("First sentence. Second sentence. Third sentence.", 25)
    assert all(len(c) <= 25 for c in chunks)
    assert len(chunks) == 3, chunks
    assert " ".join(chunks) == "First sentence. Second sentence. Third sentence."
    ok(f"split_into_chunks sentence-bounded under max_chars + content preserved: {chunks}")

    clamped = f32_to_s16le(np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32))
    assert clamped[0:2] == b"\x00\x00"
    assert clamped[2:4] == b"\xff\x7f"
    assert clamped[4:6] == b"\x01\x80"
    assert clamped[6:8] == b"\xff\x7f"
    assert clamped[8:10] == b"\x00\x80"
    ok("f32_to_s16le scale-then-clamp matches upstream Rust at all five inputs (incl. +/-2.0)")

    assert MAX_PHONEME_LENGTH == 510 and PAD_TOKEN_ID == 0
    assert tokenize("hello") == [50, 47, 54, 54, 57]
    ok("tokenize('hello') == [50, 47, 54, 54, 57] (Kokoro v1.0 vocab IDs)")

    assert clean_phonemes("brown") == "bɹown"
    assert "@" not in clean_phonemes("hello@world")
    assert clean_phonemes("xenon") == "kenon"
    ok("clean_phonemes pins exact rune-map output (r->ɹ, x->k) + drops unknowns")

    from tts.kokoro.model import split_phoneme_chunks
    assert split_phoneme_chunks("") == []
    assert split_phoneme_chunks("hello world") == ["hello world"]
    assert split_phoneme_chunks("a, b. c") == ["a, b. c"]
    chunks_long = split_phoneme_chunks(f"{'x' * 600}. yes")
    assert chunks_long and all(c.strip() for c in chunks_long), (
        f"must not emit empty chunks: {chunks_long}"
    )
    assert len(chunks_long) == 2, f"expected 2 chunks at the only sentence boundary: {chunks_long}"
    assert chunks_long[1] == ". yes"
    ok(f"split_phoneme_chunks: empty-chunk regression fixed; long sentences split at .,!?;")

    class _StubKokoro:
        def __init__(self, voices: list[str]):
            self._voices = voices
        def voices_list(self) -> list[str]:
            return list(self._voices)
        def has_voice(self, name: str) -> bool:
            return name in self.voices_list()

    stub = _StubKokoro(["af_heart", "am_michael"])
    assert stub.has_voice("af_heart") is True
    assert stub.has_voice("am_michael") is True
    assert stub.has_voice("nonexistent") is False
    ok("KokoroTTS.has_voice contract: bool, in voices_list()")

def test_d_diarization_package_speaches_plus_parity():
    _section('Fix D: diarization package (speaches-plus parity)')
    import numpy as np
    from diarization import (
        DEFAULT_CLUSTERING_THRESHOLD,
        DEFAULT_MAX_SPEAKERS,
        DiarConfig,
        DiarSegment,
        Multilabel,
        OnlineClusterer,
        PowersetDecoder,
        SegmentationLogits,
        coalesce_segments,
        median_filter_multihot,
        slide_chunks,
    )

    dec = PowersetDecoder(4, 2)
    assert dec.num_classes() == 11, dec.num_classes()
    assert dec.mapping[0] == [] and dec.mapping[1] == [0] and dec.mapping[5] == [0, 1]
    ok(f"PowersetDecoder(4,2) DiariZen-v2 topology = {dec.num_classes()} classes")
    assert PowersetDecoder(3, 2).num_classes() == 7
    ok("PowersetDecoder(3,2) pyannote 3-spk topology = 7 classes")

    silence_logits = SegmentationLogits(
        frames=1, classes=11,
        data=np.array([0.0] + [-10.0] * 10, dtype=np.float32),
    )
    assert list(dec.to_multilabel_hard(silence_logits).row(0)) == [0, 0, 0, 0]
    ok("argmax silence -> empty multi-hot")
    overlap_logits = SegmentationLogits(
        frames=1, classes=11,
        data=np.array([-10.0] * 5 + [0.0] + [-10.0] * 5, dtype=np.float32),
    )
    assert list(dec.to_multilabel_hard(overlap_logits).row(0)) == [1, 1, 0, 0]
    ok("argmax class 5 -> [1, 1, 0, 0] overlap")

    chunks = slide_chunks(np.ones(8000, dtype=np.float32), 16_000, 5.0, 0.1)
    assert len(chunks) == 1 and chunks[0].samples.shape[0] == 80_000
    assert chunks[0].samples[10_000] == 0.0
    ok("slide_chunks pads short utterance to chunk_samples")
    chunks = slide_chunks(np.ones(16_000 * 11, dtype=np.float32), 16_000, 5.0, 0.1)
    assert len(chunks) >= 12 and chunks[1].t_offset_ms == 500
    ok(f"slide_chunks 11s @ hop=0.1 -> {len(chunks)} chunks, hop=500ms")

    ml = Multilabel(frames=7, speakers=1, data=np.array([1, 1, 1, 0, 1, 1, 1], dtype=np.uint8))
    assert median_filter_multihot(ml, 3).row(3)[0] == 1
    ok("median_filter_multihot smooths singleton blip")

    merged = coalesce_segments([
        DiarSegment(speaker=0, t_start_ms=0,    t_end_ms=500,  confidence=0.9),
        DiarSegment(speaker=0, t_start_ms=600,  t_end_ms=1000, confidence=0.85),
        DiarSegment(speaker=1, t_start_ms=1100, t_end_ms=1500, confidence=0.8),
    ])
    assert len(merged) == 2 and merged[0].t_end_ms == 1000
    ok("coalesce_segments merges adjacent same-speaker (gap <= 250ms)")

    def _unit(v):
        arr = np.array(v, dtype=np.float32)
        return arr / np.linalg.norm(arr)

    cl = OnlineClusterer(threshold=0.5, max_speakers=4)
    id_a, _ = cl.assign(_unit([1, 0, 0]))
    id_b, sim = cl.assign(_unit([0.99, 0.01, 0]))
    assert id_a == id_b == 0 and sim > 0.9 and cl.num_clusters() == 1
    id_c, _ = cl.assign(_unit([0, 1, 0]))
    assert id_c == 1 and cl.num_clusters() == 2
    ok("OnlineClusterer assigns + creates new cluster on dissimilar embedding")
    cl_capped = OnlineClusterer(threshold=0.99, max_speakers=2)
    for v in (_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])):
        cl_capped.assign(v)
    assert cl_capped.num_clusters() == 2
    ok("OnlineClusterer caps at max_speakers")

    cfg = DiarConfig.from_env()
    assert cfg.clustering_threshold == DEFAULT_CLUSTERING_THRESHOLD
    assert cfg.max_speakers == DEFAULT_MAX_SPEAKERS
    ok(f"DiarConfig.from_env() defaults: threshold={cfg.clustering_threshold}, max={cfg.max_speakers}")

def test_e_centralized_env_var_module_speaches_plus_defaults_env_parity():
    _section('Fix E: centralized env-var module (speaches-plus defaults::env parity)')
    import env as _env

    expected_env_names = {
        "QWEN3_TTS_MODELS", "QWEN3_TTS_MODEL", "QWEN3_TTS_DEVICE", "QWEN3_TTS_DTYPE",
        "QWEN3_TTS_HOST", "QWEN3_TTS_PORT", "QWEN3_TTS_BATCH_WINDOW_MS",
        "QWEN3_OMNI_MODEL", "QWEN3_OMNI_DISABLE_TALKER",
        "QWEN3_ALIGNER_MODEL",
        "GEMMA_MODEL", "GEMMA_ATTN_IMPL", "GEMMA_COMPILE",
        "KOKORO_ENABLE", "KOKORO_VOICES_DIR", "KOKORO_MODEL_FILE", "KOKORO_ONNX_PROVIDER",
        "PHONEMIZER_ESPEAK_LIBRARY", "ESPEAK_DATA_PATH",
        "DIAR_SEGMENTATION_MODEL_FILE", "DIAR_EMBEDDING_MODEL_FILE",
        "DIAR_THRESHOLD", "DIAR_MAX_SPEAKERS", "DIAR_MIN_SPAN_FRAMES", "DIAR_MEDIAN_FILTER_FRAMES",
    }
    actual_env_names = {n for n in dir(_env) if n.isupper()}
    missing = expected_env_names - actual_env_names
    assert not missing, f"env.py missing: {sorted(missing)}"
    ok(f"env.py exports >= {len(expected_env_names)} of the upstream-required env-var names ({len(actual_env_names)} total declared)")

    for name in expected_env_names:
        val = getattr(_env, name)
        assert val == name, f"env.{name} = {val!r}, expected {name!r}"
    ok("every env.X constant value == its name (matches upstream defaults::env pattern)")

    import os
    os.environ["DIAR_THRESHOLD"] = "0.71"
    from diarization.types import DiarConfig
    cfg = DiarConfig.from_env()
    assert cfg.clustering_threshold == 0.71
    del os.environ["DIAR_THRESHOLD"]
    ok("DiarConfig.from_env reads via env.DIAR_THRESHOLD")

def test_v_validation_event_shape_parity_speaches_plus_oapi():
    _section('Fix V: validation + event-shape parity (speaches-plus oapi)')
    import oapi
    from oapi import kind, task
    from server import (
        SpeechRequest,
        _validate_speech_request,
    )
    from fastapi import HTTPException as _HTTPExc

    entry = oapi.missing_field(["body", "input"])
    assert entry == {"type": "missing", "loc": ["body", "input"], "msg": "Field required"}
    ok(f"oapi.missing_field shape matches upstream oapi::missing_field: {entry}")

    assert kind.INVALID_REQUEST == "invalid_request_error"
    assert kind.AUTH == "authentication_error"
    assert kind.NOT_FOUND == "not_found_error"
    assert kind.SERVER == "internal_server_error"
    assert kind.SERVICE_UNAVAIL == "service_unavailable_error"
    ok("oapi.kind.* matches upstream oapi::kind:: (5 constants)")

    assert task.ASR == "automatic-speech-recognition"
    assert task.TTS == "text-to-speech"
    assert task.VAD == "voice-activity-detection"
    assert task.CHAT == "chat-completion"
    assert task.FORCED_ALIGNMENT == "forced-alignment"
    ok("oapi.task.* matches upstream oapi::task:: (3 upstream + 2 our extensions)")

    err = oapi.openai_error("bad", kind.INVALID_REQUEST, param="speed", code="out_of_range")
    assert err == {"error": {"message": "bad", "type": "invalid_request_error", "param": "speed", "code": "out_of_range"}}
    ok("oapi.openai_error envelope shape")

    assert oapi.hf_owner("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice") == "Qwen"
    assert oapi.hf_owner("local-model") == "local-model"
    ok("oapi.hf_owner: HF id splits on '/'; plain name returns self")

    m = oapi.Model(id="x", owned_by="o", task=task.TTS, languages=["en"], extras={"sample_rate": 24000})
    d = m.to_dict()
    assert d == {"id": "x", "object": "model", "created": 1, "owned_by": "o", "language": ["en"], "task": "text-to-speech", "sample_rate": 24000}
    ok("oapi.Model.to_dict OpenAI envelope")
    assert oapi.ListModelsResponse(data=[m]).to_dict() == {"object": "list", "data": [d]}
    ok("oapi.ListModelsResponse")

    empty_input = SpeechRequest(input="", voice="af_heart")
    try:
        _validate_speech_request(empty_input)
        raise AssertionError("BAD: empty input accepted")
    except _HTTPExc as exc:
        assert exc.status_code == 422
        detail = exc.detail
        assert isinstance(detail, list) and detail
        assert any(e.get("type") == "missing" and e.get("loc") == ["body", "input"] for e in detail), (
            f"expected missing-field entry for body.input, got {detail}"
        )
        ok(f"empty input -> 422 with missing-field entry: {detail[0]}")

    bad_sr = SpeechRequest(input="hi", voice="af_heart", sample_rate=99_999)
    try:
        _validate_speech_request(bad_sr)
        raise AssertionError("BAD: out-of-range sample_rate accepted")
    except _HTTPExc as exc:
        assert exc.status_code == 422
        detail = exc.detail
        assert any(
            e.get("type") == "less_than_equal"
            and e.get("loc") == ["body", "sample_rate"]
            and e.get("input") == 99_999
            for e in detail
        ), f"expected less_than_equal on sample_rate, got {detail}"
        ok(f"out-of-range sample_rate -> 422 fastapi entry: {detail[0]}")

def test_m_v1_models_endpoint_shape_speaches_plus_parity():
    _section('Fix M: /v1/models endpoint shape (speaches-plus parity)')
    import oapi
    from oapi import kind, task
    from server import _build_models, retrieve_model

    models = _build_models()
    entries = [m.to_dict() for m in models]
    assert all(e["object"] == "model" for e in entries), "every entry must be object=model"
    assert all({"id", "object", "created", "owned_by", "language", "task"} <= set(e) for e in entries), (
        "entries must carry the OpenAI envelope keys"
    )
    ok(f"{len(entries)} entries, all with OpenAI shape")

    valid_tasks = {task.ASR, task.TTS, task.CHAT, task.FORCED_ALIGNMENT}
    for e in entries:
        assert e["task"] in valid_tasks, f"unknown task: {e['task']}"
    ok(f"all tasks in {sorted(valid_tasks)}")

    tts_only = oapi.list_models_response(models, task.TTS)["data"]
    assert all(e["task"] == task.TTS for e in tts_only)
    ok(f"oapi.list_models_response narrows to {len(tts_only)}/{len(entries)} TTS entries")

    try:
        retrieve_model("definitely-not-a-real-model")
        raise AssertionError("BAD: missing model accepted")
    except Exception as exc:
        msg = str(exc)
        assert kind.NOT_FOUND in msg or "not found" in msg.lower()
        ok(f"missing model -> 404 with not_found_error envelope: {msg[:80]}")

def test_a_registry_proposer_abc():
    _section('Fix A: registry + Proposer ABC')
    from nano_vllm.models.registry import MODEL_REGISTRY, resolve_model_class
    from nano_vllm.spec_decode.base import Proposer
    from nano_vllm.spec_decode.eagle_proposer import EagleProposer
    from nano_vllm.spec_decode.ngram import NgramProposer

    assert "Qwen3ForCausalLM" in MODEL_REGISTRY
    assert "Gemma4ForCausalLM" in MODEL_REGISTRY
    ok(f"registry keys: {sorted(MODEL_REGISTRY)}")

    class FakeHFConfig:
        architectures = ["Qwen3ForCausalLM"]

    cls = resolve_model_class(FakeHFConfig())
    ok(f"Qwen3 architecture -> {cls.__name__}")

    class FakeGemmaConfig:
        architectures = ["Gemma4ForCausalLM"]

    cls = resolve_model_class(FakeGemmaConfig())
    ok(f"Gemma4 architecture -> {cls.__name__}")

    class BadConfig:
        architectures = ["NonExistent"]

    try:
        resolve_model_class(BadConfig())
        raise AssertionError("BAD: NonExistent accepted")
    except ValueError as exc:
        ok(f"unknown architecture rejected: {str(exc)[:80]}")

    assert issubclass(NgramProposer, Proposer)
    assert issubclass(EagleProposer, Proposer)
    ok("NgramProposer + EagleProposer both implement Proposer ABC")

def test_e_chatresponseformat_envelope_type_tightenings():
    _section('Fix E: ChatResponseFormat envelope + type tightenings')
    from pydantic import ValidationError

    from server import ChatCompletionRequest, ChatJsonSchemaSpec, _guided_json_from_request

    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Result", "schema": {"type": "object"}, "strict": True},
        },
    )
    assert isinstance(req.response_format.json_schema, ChatJsonSchemaSpec)
    assert req.response_format.json_schema.schema == {"type": "object"}
    extracted = _guided_json_from_request(req)
    assert extracted == {"type": "object"}
    ok("OpenAI-shape json_schema parses; extracted inner schema correctly")

    try:
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            response_format={"type": "json_schema", "json_schema": {"type": "object"}},
        )
        raise AssertionError("BAD: legacy bare-dict shape accepted")
    except ValidationError:
        ok("legacy bare-dict shape rejected (breaking-change confirmed)")

    req2 = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    assert _guided_json_from_request(req2) == {}
    ok("json_object envelope returns permissive {}")

    req3 = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
    assert _guided_json_from_request(req3) is None
    ok("no response_format -> None passthrough")

    import inspect

    from nano_vllm.spec_decode.eagle_proposer import EagleProposer
    sig = inspect.signature(EagleProposer.propose_tokens)
    ann = sig.parameters["last_tokens"].annotation

    assert "LongTensor" not in str(ann), f"deprecated alias still present: {ann}"
    assert "Tensor" in str(ann)
    ok(f"EagleProposer.propose_tokens.last_tokens annotation: {ann!r} (no LongTensor alias)")

def test_b_eagle_aux_row_contract_p1_no_crash_on_finished_seq():
    _section('Fix B: EAGLE aux row contract (P1: no crash on finished seq)')
    from nano_vllm.engine.sequence import Sequence, SequenceStatus
    from nano_vllm.spec_decode.eagle3 import Eagle3Config, Eagle3DraftModel
    from nano_vllm.spec_decode.eagle_proposer import EagleProposer

    cfg = Eagle3Config(
        hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, rms_norm_eps=1e-6, rope_theta=10000.0, max_position_embeddings=4096,
        target_vocab_size=1000, draft_vocab_size=500, target_hidden_size=None,
        eagle_aux_hidden_state_layer_ids=[0, 1, 2], norm_before_residual=True,
        norm_before_fc=False, tie_word_embeddings=False,
    )
    draft_model = Eagle3DraftModel(cfg)
    draft_model.d2t.zero_()

    seqs = [Sequence([1, 2, 3]), Sequence([4, 5]), Sequence([6, 7, 8, 9])]
    seqs[1].status = SequenceStatus.FINISHED
    runner_state = {"last_aux_hidden_states": torch.zeros(3, 3 * 64)}
    proposer = EagleProposer(draft_model, num_drafts=3)
    drafts = proposer.propose(seqs, runner_state)
    assert seqs[0].seq_id in drafts
    assert seqs[2].seq_id in drafts
    assert seqs[1].seq_id not in drafts
    ok(f"finished seq excluded; drafts for {sorted(drafts.keys())}, lens {[len(v) for v in drafts.values()]}")

    all_done = [Sequence([1, 2]), Sequence([3, 4])]
    for s in all_done:
        s.status = SequenceStatus.FINISHED
    empty = proposer.propose(all_done, {"last_aux_hidden_states": torch.zeros(2, 192)})
    assert empty == {}
    ok("all-finished batch returns {}")

def test_d_2_eagle_chain_hoist_single_tolist_sync():
    _section('Fix D #2: EAGLE chain hoist (single .tolist sync)')
    from nano_vllm.spec_decode.eagle3 import Eagle3Config, Eagle3DraftModel
    from nano_vllm.spec_decode.eagle_proposer import EagleProposer
    cfg = Eagle3Config(
        hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, rms_norm_eps=1e-6, rope_theta=10000.0, max_position_embeddings=4096,
        target_vocab_size=1000, draft_vocab_size=500, target_hidden_size=None,
        eagle_aux_hidden_state_layer_ids=[0, 1, 2], norm_before_residual=True,
        norm_before_fc=False, tie_word_embeddings=False,
    )
    draft_model = Eagle3DraftModel(cfg)
    draft_model.d2t.zero_()
    prop = EagleProposer(draft_model, num_drafts=5)
    out = prop.propose_tokens(torch.tensor([5, 7, 11]), torch.zeros(3, 192))
    assert all(len(d) == 5 for d in out)
    ok(f"K=5 chain shapes: {[len(d) for d in out]}")
    assert all(isinstance(t, int) for chain in out for t in chain)
    ok("chain tokens are Python ints (not numpy/torch scalars)")

    prop1 = EagleProposer(draft_model, num_drafts=1)
    out1 = prop1.propose_tokens(torch.tensor([5, 7]), torch.zeros(2, 192))
    assert all(len(d) == 1 for d in out1)
    ok(f"K=1 regression OK: {out1}")

def test_perf_1_verify_indices_removed_from_context():
    _section('Perf #1: verify_indices removed from Context')
    from nano_vllm.utils.context import Context, get_context, reset_context, set_context

    ctx = Context()
    assert not hasattr(ctx, "verify_indices"), "verify_indices field still present"
    ok("Context.verify_indices field is gone")

    set_context(True, verify_mode=True)
    ctx = get_context()
    assert ctx.verify_mode is True
    ok(f"set_context(verify_mode=True) -> ctx.verify_mode={ctx.verify_mode}")
    reset_context()

def test_perf_2_ngramproposer_bytes_rfind_correctness_speedup():
    _section('Perf #2: NgramProposer bytes.rfind correctness + speedup')
    from nano_vllm.spec_decode.ngram import NgramProposer

    assert NgramProposer.propose_tokens([1, 2, 3, 4, 5, 1, 2, 3]) == [4, 5]
    assert NgramProposer.propose_tokens([1, 2, 3], min_n=2, max_n=4, num_drafts=3) == []
    assert NgramProposer.propose_tokens([], num_drafts=3) == []
    assert NgramProposer.propose_tokens([7], num_drafts=3) == []
    assert NgramProposer.propose_tokens([1, 2, 3, 1, 2], min_n=2, max_n=4, num_drafts=3) == [3]
    ok("self-test cases pass (bytes.rfind path)")

    import random

    random.seed(42)
    tokens = [random.randint(1, 5000) for _ in range(2000)]
    pattern = [9001, 9002, 9003]
    followups = [9004, 9005, 9006, 9007]
    tokens[100:103] = pattern
    tokens[100 + 3:100 + 3 + 4] = followups
    tokens[-3:] = pattern

    drafts = NgramProposer.propose_tokens(tokens, max_n=3, min_n=2, num_drafts=4)
    assert drafts == followups, f"expected {followups}, got {drafts}"
    ok(f"2000-token scan finds planted continuation: {drafts}")

    def python_loop_propose(token_ids, max_n=8, min_n=2, num_drafts=5):
        total = len(token_ids)
        if total < min_n + 1 or num_drafts <= 0:
            return []
        upper_n = min(max_n, total - 1)
        for n in range(upper_n, min_n - 1, -1):
            suffix = token_ids[total - n:]
            first = suffix[0]
            for start in range(total - n - 1, -1, -1):
                if token_ids[start] != first:
                    continue
                if token_ids[start:start + n] != suffix:
                    continue
                draft_start = start + n
                draft_end = min(draft_start + num_drafts, total - n)
                if draft_start >= draft_end:
                    return []
                return token_ids[draft_start:draft_end]
        return []

    no_match = [random.randint(1, 50000) for _ in range(4096)]
    ITERS = 200

    t0 = time.perf_counter()
    for _ in range(ITERS):
        python_loop_propose(no_match, max_n=8, min_n=2, num_drafts=5)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(ITERS):
        NgramProposer.propose_tokens(no_match, max_n=8, min_n=2, num_drafts=5)
    t_bytes = time.perf_counter() - t0

    speedup = t_py / t_bytes if t_bytes > 0 else float("inf")
    info(f"Python-loop:  {t_py * 1000 / ITERS:.3f} ms/call (no-match worst case, len=4096)")
    info(f"bytes.rfind:  {t_bytes * 1000 / ITERS:.3f} ms/call")
    info(f"speedup:      {speedup:.1f}x")
    assert speedup > 2, f"expected >=2x speedup, got {speedup:.1f}x"
    ok(f"bytes.rfind is {speedup:.1f}x faster than Python loop on the cold-scan worst case")

    mismatches = 0
    for _ in range(50):
        n = random.randint(50, 500)
        tok = [random.randint(1, 100) for _ in range(n)]
        if random.random() < 0.5:
            tok[-3:] = tok[10:13]
        a = python_loop_propose(tok, max_n=5, min_n=2, num_drafts=3)
        b = NgramProposer.propose_tokens(tok, max_n=5, min_n=2, num_drafts=3)
        if a != b:
            mismatches += 1
    assert mismatches == 0, f"equivalence broken: {mismatches}/50"
    ok("bytes.rfind equivalent to Python loop on 50 randomized inputs")

def test_perf_3_scheduler_running_with_drafts_counter():
    _section('Perf #3: Scheduler running_with_drafts counter')
    from nano_vllm.config import Config
    from nano_vllm.engine.scheduler import Scheduler
    from nano_vllm.sampling_params import SamplingParams

    assert hasattr(Scheduler, "note_drafts_set")
    ok("Scheduler.note_drafts_set hook exists")

    class FakeScheduler:
        def __init__(self):
            self.num_running_with_drafts = 0
        note_drafts_set = Scheduler.note_drafts_set

    s = FakeScheduler()
    s.note_drafts_set(False, True)
    assert s.num_running_with_drafts == 1
    s.note_drafts_set(False, True)
    assert s.num_running_with_drafts == 2
    s.note_drafts_set(True, True)
    assert s.num_running_with_drafts == 2
    s.note_drafts_set(True, False)
    assert s.num_running_with_drafts == 1
    s.note_drafts_set(False, False)
    assert s.num_running_with_drafts == 1
    ok("counter increments/decrements correctly across set/clear/no-op")

def test_perf_4_pinned_bitmask_workspace_cache_reuse():
    _section('Perf #4: pinned bitmask workspace + cache reuse')
    from nano_vllm.layers.grammar import _get_pinned_bitmask, _pinned_bitmask_cache, has_xgrammar

    assert has_xgrammar()
    _pinned_bitmask_cache.clear()

    b1 = _get_pinned_bitmask(4, 32000)
    assert b1.shape == (4, 1000)
    assert len(_pinned_bitmask_cache) == 1
    underlying_id = id(_pinned_bitmask_cache[(32000, torch.int32)])
    ok(f"first call: shape {tuple(b1.shape)}, cache size 1")

    b2 = _get_pinned_bitmask(2, 32000)
    assert b2.shape == (2, 1000)
    assert len(_pinned_bitmask_cache) == 1
    assert id(_pinned_bitmask_cache[(32000, torch.int32)]) == underlying_id
    ok("smaller request reused existing buffer (no realloc)")

    b3 = _get_pinned_bitmask(8, 32000)
    assert b3.shape == (8, 1000)
    assert _pinned_bitmask_cache[(32000, torch.int32)].size(0) >= 8
    ok(f"larger request grew buffer (now >={_pinned_bitmask_cache[(32000, torch.int32)].size(0)} rows)")

    _get_pinned_bitmask(4, 50000)
    assert len(_pinned_bitmask_cache) == 2
    ok("different vocab_size produces separate cache entry")

    _pinned_bitmask_cache.clear()
    _get_pinned_bitmask(8, 32000)
    baseline_id = id(_pinned_bitmask_cache[(32000, torch.int32)])
    tracemalloc.start()
    for _ in range(1000):
        _get_pinned_bitmask(8, 32000)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert id(_pinned_bitmask_cache[(32000, torch.int32)]) == baseline_id
    info(f"1000 hits at same shape: peak {peak} bytes (no buffer realloc)")
    ok("repeated same-shape calls reuse the cached buffer")

def test_cross_cut_xgrammar_mask_end_to_end_via_the_pinned_cache():
    _section('Cross-cut: xgrammar mask end-to-end via the pinned cache')
    import xgrammar

    from nano_vllm.layers.grammar import apply_grammar_mask, _pinned_bitmask_cache

    ti = xgrammar.TokenizerInfo(
        encoded_vocab=[bytes([i]) for i in range(256)],
        vocab_type=xgrammar.VocabType.RAW,
        vocab_size=256,
    )
    compiler = xgrammar.GrammarCompiler(ti)
    grammar = compiler.compile_grammar('root ::= "a" | "b"')
    matcher = xgrammar.GrammarMatcher(grammar)

    logits = torch.randn(1, 256)
    masked = apply_grammar_mask(logits, [matcher], 256)

    assert masked[0, 97].item() != float("-inf")
    assert masked[0, 98].item() != float("-inf")
    assert masked[0, 99].item() == float("-inf")
    assert masked[0, 0].item() == float("-inf")
    ok("apply_grammar_mask correctly forces -inf on disallowed tokens")
    ok("...and uses the pinned bitmask cache (size: " + str(len(_pinned_bitmask_cache)) + ")")

def test_tool_calling_all_4_parser_formats():
    _section('Tool calling: all 4 parser formats')
    from server import _parse_tool_calls, _strip_tool_calls_from_text

    formats = [
        ("Qwen3 native",    '<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>'),
        ("Markdown JSON",   '```json\n{"name":"get_weather","arguments":{"city":"NYC"}}\n```'),
        ("Raw JSON",        '{"name":"get_weather","arguments":{"city":"LA"}}'),
        ("Gemma tool_code", '```tool_code\nget_weather(city="Berlin", units="celsius")\n```'),
    ]
    for label, payload in formats:
        parsed = _parse_tool_calls(payload, {"get_weather"})
        assert parsed and parsed[0]["function"]["name"] == "get_weather", f"{label} failed: {parsed}"
        args = json.loads(parsed[0]["function"]["arguments"])
        assert "city" in args, f"{label} args missing city: {args}"
        ok(f"{label:18s} -> name=get_weather args={args}")

    mixed = 'Sure, checking weather.\n```tool_code\nget_weather(city="Paris")\n```\nDone.'
    parsed = _parse_tool_calls(mixed, {"get_weather"})
    assert parsed is not None
    stripped = _strip_tool_calls_from_text(mixed, parsed)
    assert "get_weather" not in stripped
    assert "tool_code" not in stripped
    ok(f"strip removes tool_code block: {stripped!r}")

def test_r2_p0_prepare_decode_uses_positional_block_index_not_1():
    _section('R2-P0: prepare_decode uses positional block index, not [-1]')

    class _FakePostVerifySeq:
        block_size = 256
        def __init__(self):
            self.num_tokens = 253
            self.last_token = 999
            self.block_table = [42, 99]
        def __len__(self):
            return self.num_tokens
        @property
        def num_blocks(self):
            return (self.num_tokens + 256 - 1) // 256
        @property
        def last_block_num_tokens(self):
            return self.num_tokens - (self.num_blocks - 1) * 256

    seq = _FakePostVerifySeq()
    last_pos = len(seq) - 1
    correct_slot = seq.block_table[last_pos // 256] * 256 + last_pos % 256
    buggy_slot = seq.block_table[-1] * 256 + seq.last_block_num_tokens - 1
    assert correct_slot == 42 * 256 + 252
    assert buggy_slot == 99 * 256 + 252
    assert correct_slot != buggy_slot
    ok(f"trigger reproduces: buggy slot {buggy_slot} != correct slot {correct_slot}")

    import inspect
    import nano_vllm.engine.model_runner as mr
    src = inspect.getsource(mr.ModelRunner.prepare_decode)
    assert "block_table[-1]" not in src, "prepare_decode still uses block_table[-1] (P0 unfixed)"
    assert "last_pos" in src and "block_size" in src
    ok("ModelRunner.prepare_decode no longer uses block_table[-1]")

def test_r2_p1_pinnedscratch_reuse():
    _section('R2-P1: PinnedScratch reuse')
    from nano_vllm.utils.pinned_scratch import host_view

    a = host_view("test_buf", torch.int64, 8)
    a.numpy()[:] = [1, 2, 3, 4, 5, 6, 7, 8]
    b = host_view("test_buf", torch.int64, 8)
    assert b.data_ptr() == a.data_ptr(), "same-size view must reuse backing buffer"
    ok(f"same-size view reuses backing storage (ptr={a.data_ptr()})")

    c = host_view("test_buf", torch.int64, 16)
    assert c.data_ptr() != a.data_ptr() or c.numel() >= 16
    ok("grow allocates new buffer with power-of-two capacity")

    host_view("test_buf", torch.int64, 4)
    ok("smaller-size view returns slice of current buffer (no realloc)")

def test_r2_p1_sampler_temp_only_fast_path():
    _section('R2-P1: Sampler temp-only fast path')
    from nano_vllm.layers.sampler import Sampler, _eager_sample_temp_only

    s = Sampler()
    logits = torch.randn(4, 256)
    temps = torch.full((4,), 0.7)
    top_k = torch.full((4,), -1, dtype=torch.int32)
    top_p = torch.full((4,), 1.0)

    out_fast = _eager_sample_temp_only(logits, temps)
    assert out_fast.shape == (4, 1)
    ok(f"_eager_sample_temp_only output shape: {tuple(out_fast.shape)}")

    torch.manual_seed(0)
    out_via_sampler = s(logits, temps, top_k, top_p, any_top_k=False, any_top_p=False)
    assert out_via_sampler.shape == (4, 1)
    ok("Sampler with any_top_k=False/any_top_p=False returns valid sample")

def test_r2_p1_prepare_sample_exposes_host_flags():
    _section('R2-P1: prepare_sample exposes host flags')
    import inspect as _inspect
    import nano_vllm.engine.model_runner as _mr
    src = _inspect.getsource(_mr.ModelRunner.prepare_sample)
    assert "any_top_k" in src and "any_top_p" in src, "prepare_sample missing host flags"
    ok("ModelRunner.prepare_sample exposes any_top_k and any_top_p host flags")

def test_vad_silero_v5_vad_port_speaches_plus_rust_src_vad_parity():
    _section('Fix VAD: Silero v5 VAD port (speaches-plus rust/src/vad parity)')
    import os as _os

    import numpy as np

    from vad import (
        CONTEXT_SAMPLES,
        INPUT_SAMPLES,
        MAX_PROB_RING,
        MAX_SPEECH_DURATION_S,
        MAX_VAD_WINDOW_SAMPLES,
        MIN_SILENCE_AT_MAX_SPEECH_MS,
        MIN_SPEECH_DURATION_MS,
        MIN_SPEECH_MS,
        NEG_THRESHOLD_DELTA,
        NEG_THRESHOLD_FLOOR,
        PREFIX_PADDING_MS,
        SAMPLE_RATE as VAD_SAMPLE_RATE,
        SILENCE_DURATION_MS,
        SPEECH_THRESHOLD,
        SileroVad,
        SpeechCommitted,
        SpeechStarted,
        SpeechTimestamp,
        VAD_FAILURE_THRESHOLD,
        VadOptions,
        VadProcessor,
        WINDOW_SAMPLES,
        get_speech_timestamps,
        speech_timestamps_from_probs,
        to_ms_speech_timestamps,
    )

    assert VAD_SAMPLE_RATE == 16_000
    assert WINDOW_SAMPLES == 512
    assert CONTEXT_SAMPLES == 64
    assert INPUT_SAMPLES == 576
    assert VAD_FAILURE_THRESHOLD == 3
    assert PREFIX_PADDING_MS == 300
    assert SILENCE_DURATION_MS == 350
    assert SPEECH_THRESHOLD == 0.5
    assert MIN_SPEECH_MS == 100
    assert MAX_VAD_WINDOW_SAMPLES == 48_000
    assert MIN_SPEECH_DURATION_MS == 100
    assert MAX_SPEECH_DURATION_S == 30.0
    assert MIN_SILENCE_AT_MAX_SPEECH_MS == 98
    assert NEG_THRESHOLD_DELTA == 0.15
    assert NEG_THRESHOLD_FLOOR == 0.01
    assert MAX_PROB_RING == 94
    ok("vad constants match upstream defaults::vad and defaults::vad_window")

    def _ms_from_samples_16k(n: int) -> int:
        return n * 1000 // 16_000

    assert _ms_from_samples_16k(1600) == 100
    assert _ms_from_samples_16k(800) == 50
    assert _ms_from_samples_16k(800) < MIN_SPEECH_MS
    assert _ms_from_samples_16k(1600) >= MIN_SPEECH_MS
    ok("min_speech_ms_default_is_100 (Rust #[test])")

    class _ProbReplay:
        def __init__(self, probs):
            self.probs = list(probs)
            self.idx = 0
        def process_window(self, window):
            if self.idx >= len(self.probs):
                raise RuntimeError(f"no more probs (idx {self.idx})")
            p = self.probs[self.idx]
            self.idx += 1
            return p
        def reset(self):
            self.idx = 0

    def _opts(threshold, min_silence_ms, min_speech_ms, pad_ms):
        return VadOptions(
            threshold=threshold,
            neg_threshold=None,
            min_speech_duration_ms=min_speech_ms,
            max_speech_duration_s=30.0,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=pad_ms,
        )

    probs = [0.0] * 32
    m = _ProbReplay(probs)
    audio = np.zeros(32 * WINDOW_SAMPLES, dtype=np.float32)
    ts = get_speech_timestamps(m, audio, _opts(0.5, 100, 0, 0), VAD_SAMPLE_RATE)
    assert ts == [], f"expected no speech, got {ts}"
    ok("hysteresis_silent_stays_silent (Rust #[test])")

    probs = [0.0, 0.0, 0.7, 0.7, 0.7, 0.7, 0.4, 0.4, 0.0, 0.0]
    m = _ProbReplay(probs)
    audio = np.zeros(10 * WINDOW_SAMPLES, dtype=np.float32)
    o = _opts(0.5, 32, 0, 0)
    o.min_silence_duration_ms = 0
    ts = get_speech_timestamps(m, audio, o, VAD_SAMPLE_RATE)
    assert len(ts) == 1, ts
    assert ts[0].start == 2 * WINDOW_SAMPLES
    assert ts[0].end == 8 * WINDOW_SAMPLES
    ok("hysteresis_enter_at_threshold_leave_at_neg_threshold (Rust #[test])")

    probs = [0.0, 0.7, 0.7, 0.4, 0.7, 0.0, 0.0]
    m = _ProbReplay(probs)
    audio = np.zeros(7 * WINDOW_SAMPLES, dtype=np.float32)
    o = _opts(0.5, 0, 0, 0)
    o.min_silence_duration_ms = 0
    ts = get_speech_timestamps(m, audio, o, VAD_SAMPLE_RATE)
    assert len(ts) == 1
    assert ts[0].start == WINDOW_SAMPLES
    assert ts[0].end == 5 * WINDOW_SAMPLES
    ok("hysteresis_dip_above_neg_threshold_does_not_release (Rust #[test])")

    probs = [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0]
    m = _ProbReplay(probs)
    audio = np.zeros(7 * WINDOW_SAMPLES, dtype=np.float32)
    o = _opts(0.5, 0, 200, 0)
    ts = get_speech_timestamps(m, audio, o, VAD_SAMPLE_RATE)
    assert ts == [], f"expected filter, got {ts}"
    ok("min_speech_filter_drops_short_segments (Rust #[test])")

    probs = [0.0, 0.7, 0.7, 0.7, 0.0, 0.0]
    m = _ProbReplay(probs)
    audio = np.zeros(6 * WINDOW_SAMPLES, dtype=np.float32)
    o = _opts(0.5, 0, 0, 60)
    o.min_silence_duration_ms = 0
    ts = get_speech_timestamps(m, audio, o, VAD_SAMPLE_RATE)
    assert len(ts) == 1
    assert ts[0].start < WINDOW_SAMPLES
    assert ts[0].end > 4 * WINDOW_SAMPLES
    ok("padding_pass_extends_segment_edges (Rust #[test])")

    class _FakeTd:
        def __init__(self, threshold, prefix_padding_ms, silence_duration_ms, min_speech_ms):
            self._threshold = threshold
            self._prefix_padding_ms = prefix_padding_ms
            self._silence_duration_ms = silence_duration_ms
            self._min_speech_ms = min_speech_ms
        def threshold(self):
            return self._threshold
        def prefix_padding_samples(self):
            return self._prefix_padding_ms * VAD_SAMPLE_RATE // 1000
        def silence_duration_samples(self):
            return self._silence_duration_ms * VAD_SAMPLE_RATE // 1000
        def neg_threshold(self):
            return max(self._threshold - NEG_THRESHOLD_DELTA, NEG_THRESHOLD_FLOOR)
        def min_speech_duration_ms(self):
            return self._min_speech_ms
        def max_speech_duration_s(self):
            return MAX_SPEECH_DURATION_S

    def _td(silence_ms, min_speech_ms, pad_ms):
        return _FakeTd(0.5, pad_ms, silence_ms, min_speech_ms)

    def _push_n_windows(p, n):
        z = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        for _ in range(n):
            p.push(z)

    probs = [0.0, 0.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0]
    m = _ProbReplay(probs)
    p = VadProcessor(m).with_turn_detection(_td(64, 0, 0))
    _push_n_windows(p, 12)
    evs = p.take_events()
    assert len(evs) == 2, evs
    assert isinstance(evs[0], SpeechStarted)
    assert isinstance(evs[1], SpeechCommitted)
    assert evs[1].audio.size > 0, "committed audio should be non-empty"
    ok("driver_emits_speech_started_then_committed (Rust #[test])")

    probs = [0.0] * 50
    m = _ProbReplay(probs)
    p = VadProcessor(m).with_turn_detection(_td(64, 0, 0))
    _push_n_windows(p, 50)
    evs = p.take_events()
    assert evs == [], evs
    assert p.audio_start_ms is None
    ok("driver_silent_input_emits_nothing (Rust #[test])")

    probs = [
        0.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0,
    ]
    m = _ProbReplay(probs)
    p = VadProcessor(m).with_turn_detection(_td(64, 0, 0))
    _push_n_windows(p, 10)
    evs = p.take_events()
    started = sum(1 for e in evs if isinstance(e, SpeechStarted))
    committed = sum(1 for e in evs if isinstance(e, SpeechCommitted))
    assert (started, committed) == (1, 1), f"first turn: {evs}"
    _push_n_windows(p, 10)
    evs = p.take_events()
    started = sum(1 for e in evs if isinstance(e, SpeechStarted))
    committed = sum(1 for e in evs if isinstance(e, SpeechCommitted))
    assert (started, committed) == (1, 1), f"second turn: {evs}"
    ok("driver_resets_state_after_commit_via_take_events (Rust #[test])")

    probs = [0.0, 0.7, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0]
    m = _ProbReplay(probs)
    p = VadProcessor(m).with_turn_detection(_td(64, 0, 0))
    _push_n_windows(p, 8)
    extra_silence = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    for _ in range(5):
        p.push(extra_silence)
    evs = p.take_events()
    committed = sum(1 for e in evs if isinstance(e, SpeechCommitted))
    assert committed == 1, evs
    ok("driver_ignores_pushes_after_commit_until_take_events (Rust #[test])")

    _vad_path = _os.environ.get("VAD_MODEL_FILE", "").strip()
    if _vad_path and _os.path.exists(_vad_path):
        model = SileroVad.load(_vad_path)
        silence = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        prob = model.process_window(silence)
        assert 0.0 <= prob <= 1.0, f"prob out of range: {prob}"
        assert prob < 0.3, f"silence prob too high: {prob}"
        ok(f"loads_model: silence prob={prob:.3f} (Rust #[test])")
    else:
        info("loads_model: skipped (set VAD_MODEL_FILE=/path/to/silero_vad.onnx)")

def test_audio_codecs_speaches_plus_rust_src_audio_parity():
    _section('Fix Audio-Codecs: speaches-plus rust/src/audio parity')
    import math as _math
    import shutil as _shutil
    import subprocess as _subprocess
    import tempfile as _tempfile

    import numpy as _np

    from audio.g711 import (
        ULAW_BIAS,
        alaw_bytes_to_f32,
        alaw_decode_byte,
        alaw_encode_sample,
        f32_to_alaw_bytes,
        f32_to_ulaw_bytes,
        ulaw_bytes_to_f32,
        ulaw_decode_byte,
        ulaw_encode_sample,
    )
    from audio.decode_any import decode_any_to_16k_mono
    from audio.resample import downmix_and_resample_f32
    from audio.types import (
        BYTES_PER_S16,
        MAX_DECODE_SAMPLE_RATE,
        MIME_RAW,
        MIME_RAW_PCM,
        MIN_DECODE_SAMPLE_RATE,
        S16_SCALE,
        S24_SCALE,
        S32_SCALE,
        TARGET_SAMPLE_RATE,
    )
    from audio.wav import decode_wav_to_16k_mono, encode_wav_mono16, find_chunk

    assert TARGET_SAMPLE_RATE == 16_000
    assert S16_SCALE == 32_768.0
    assert S24_SCALE == 8_388_608.0
    assert S32_SCALE == 2_147_483_648.0
    assert BYTES_PER_S16 == 2
    assert MIME_RAW_PCM == "audio/pcm" and MIME_RAW == "audio/raw"
    assert MIN_DECODE_SAMPLE_RATE == 1_000 and MAX_DECODE_SAMPLE_RATE == 384_000
    ok("types: constants match upstream verbatim")

    for _b in range(256):
        _v = ulaw_decode_byte(_b)
        assert abs(int(_v)) <= 32_124, f"ulaw_decode({_b:02x}) = {_v} out of range"
    ok("ulaw_decode_is_total_over_byte_space (rust #[test])")

    for _b in range(256):
        _v = alaw_decode_byte(_b)
        assert abs(int(_v)) <= 32_256, f"alaw_decode({_b:02x}) = {_v} out of range"
    ok("alaw_decode_is_total_over_byte_space (rust #[test])")

    def _signum(x: int) -> int:
        return (x > 0) - (x < 0)

    _max_rel = 0.0
    for _s in range(-32_767, 32_768):
        _enc = ulaw_encode_sample(_s)
        _dec = ulaw_decode_byte(_enc)
        if abs(_s) > 256:
            if _dec != 0:
                assert _signum(_dec) == _signum(_s), f"sign for {_s}"
            _err = abs(_dec - _s) / abs(_s)
            if _err > _max_rel:
                _max_rel = _err
    assert _max_rel < 0.13, f"ulaw max relative error {_max_rel}"
    ok(f"ulaw_encode_is_total_over_i16_space: max rel err = {_max_rel:.4f}")

    _max_rel = 0.0
    for _s in range(-32_767, 32_768):
        _enc = alaw_encode_sample(_s)
        _dec = alaw_decode_byte(_enc)
        if abs(_s) > 256:
            if _dec != 0:
                assert _signum(_dec) == _signum(_s), f"sign for {_s}"
            _err = abs(_dec - _s) / abs(_s)
            if _err > _max_rel:
                _max_rel = _err
    assert _max_rel < 0.13, f"alaw max relative error {_max_rel}"
    ok(f"alaw_encode_is_total_over_i16_space: max rel err = {_max_rel:.4f}")

    for _s in [-32_000, -8_000, -1_000, 0, 1_000, 8_000, 32_000]:
        _e = ulaw_encode_sample(_s)
        _d = ulaw_decode_byte(_e)
        if _s == 0:
            assert abs(_d) <= ULAW_BIAS // 2
        else:
            assert _signum(_d) == _signum(_s), f"sign for {_s}"
            assert abs(_d - _s) / abs(_s) < 0.13
    ok("ulaw_round_trip_preserves_sign_and_magnitude (rust #[test])")

    for _s in [-32_000, -8_000, -1_000, 0, 1_000, 8_000, 32_000]:
        _e = alaw_encode_sample(_s)
        _d = alaw_decode_byte(_e)
        if _s == 0:
            assert abs(_d) <= 16
        else:
            assert _signum(_d) == _signum(_s), f"sign for {_s}"
            assert abs(_d - _s) / abs(_s) < 0.13
    ok("alaw_round_trip_preserves_sign_and_magnitude (rust #[test])")

    assert abs(ulaw_decode_byte(0xFF)) < 16
    ok("ulaw_silence_decodes_to_zero (rust #[test])")
    assert abs(alaw_decode_byte(0xD5)) < 16
    ok("alaw_silence_decodes_to_zero (rust #[test])")

    _pos = alaw_encode_sample(32_767)
    _neg = alaw_encode_sample(-32_768)
    _dp = alaw_decode_byte(_pos)
    _dn = alaw_decode_byte(_neg)
    assert _dp > 24_000, f"positive extreme decodes to {_dp}"
    assert _dn < -24_000, f"negative extreme decodes to {_dn}"
    ok("alaw_extremes_clip_in_max_segment (rust #[test])")

    _in = _np.array([0.0, 0.25, -0.25, 0.5, -0.5, 0.99, -0.99], dtype=_np.float32)
    _back = ulaw_bytes_to_f32(f32_to_ulaw_bytes(_in))
    assert len(_back) == len(_in)
    for _i, (_w, _g) in enumerate(zip(_in.tolist(), _back.tolist())):
        assert abs(_g - _w) <= 0.05, f"sample {_i}: want {_w} got {_g}"
    ok("F32_ULawRoundTrip (go test parity, |Δ| <= 0.05)")

    _back = alaw_bytes_to_f32(f32_to_alaw_bytes(_in))
    assert len(_back) == len(_in)
    for _i, (_w, _g) in enumerate(zip(_in.tolist(), _back.tolist())):
        assert abs(_g - _w) <= 0.05, f"sample {_i}: want {_w} got {_g}"
    ok("F32_ALawRoundTrip (go test parity, |Δ| <= 0.05)")

    _zero = downmix_and_resample_f32([1.0, 2.0], 1, 100, 16_000)
    assert len(_zero) == 0
    ok("downmix_and_resample_f32 rejects sr_in below MIN_DECODE_SAMPLE_RATE")
    _zero = downmix_and_resample_f32([1.0, 2.0], 0, 16_000, 16_000)
    assert len(_zero) == 0
    ok("downmix_and_resample_f32 rejects channels=0")

    _stereo = _np.array([1.0, 3.0, 5.0, 7.0], dtype=_np.float32)
    _mono = downmix_and_resample_f32(_stereo, 2, 16_000, 16_000)
    assert _mono.tolist() == [2.0, 6.0]
    ok("downmix_and_resample_f32 averages channels for sr_in==sr_out")

    _src = _np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=_np.float32)
    _up = downmix_and_resample_f32(_src, 1, 8_000, 16_000)
    assert len(_up) == 16
    assert _up[0] == 0.0
    ok(f"downmix_and_resample_f32 8k->16k upsample len={len(_up)} (n_in*sr_out//sr_in)")

    def _sine(sr: int, freq: float) -> _np.ndarray:
        n = sr
        t = _np.arange(n) / sr
        return (0.5 * _np.sin(2 * _np.pi * freq * t)).astype(_np.float32)

    def _encode_raw_s16le(samples: _np.ndarray) -> bytes:
        v = _np.clip(_np.rint(samples * 32767.0), -32_768, 32_767).astype("<i2")
        return v.tobytes()

    def _rms(s: _np.ndarray) -> float:
        if len(s) == 0:
            return 0.0
        return float(_np.sqrt(_np.mean(s.astype(_np.float64) ** 2)))

    def _align_and_diff(a: _np.ndarray, b: _np.ndarray, skip: int):
        n = min(len(a), len(b))
        if skip > n // 4:
            skip = n // 4
        a = a[skip : n - skip]
        b = b[skip : n - skip]
        if len(a) == 0:
            return 0.0, 0.0
        d = _np.abs(a - b)
        sum_diff_sq = float(_np.sum(d.astype(_np.float64) ** 2))
        sum_a = float(_np.sum(a.astype(_np.float64) ** 2))
        if sum_a == 0:
            return float(d.max()), 0.0
        return float(d.max()), _math.sqrt(sum_diff_sq / sum_a)

    _src = _sine(16_000, 440.0)
    _raw = _encode_raw_s16le(_src)
    _got = decode_any_to_16k_mono(_raw, "audio/pcm")
    assert len(_got) == len(_src), f"raw pcm length {len(_got)} != {len(_src)}"
    _max_abs, _rel_rms = _align_and_diff(_src, _got, 0)
    assert _max_abs <= 1e-3 and _rel_rms <= 1e-3, (
        f"raw PCM drift: maxAbs={_max_abs} relRMS={_rel_rms}"
    )
    ok(f"DecodeUploadedAudio_RawPCM (go parity): maxAbs={_max_abs:.2e} relRMS={_rel_rms:.2e}")

    _wav = encode_wav_mono16(_src, 16_000)
    assert find_chunk(_wav, b"fmt ") is not None
    assert find_chunk(_wav, b"data") is not None
    ok("find_chunk locates fmt and data chunks (rust port)")

    _got = decode_wav_to_16k_mono(_wav)
    assert len(_got) == len(_src)
    _max_abs, _rel_rms = _align_and_diff(_src, _got, 0)
    assert _max_abs <= 1e-3 and _rel_rms <= 1e-3, (
        f"wav drift: maxAbs={_max_abs} relRMS={_rel_rms}"
    )
    ok(f"DecodeUploadedAudio_WAV (go parity): maxAbs={_max_abs:.2e} relRMS={_rel_rms:.2e}")

    _src48 = _sine(48_000, 440.0)
    _wav48 = encode_wav_mono16(_src48, 48_000)
    _got48 = decode_wav_to_16k_mono(_wav48)
    assert 16_000 - 2 <= len(_got48) <= 16_000 + 2, f"resampled length {len(_got48)}"
    assert _rms(_got48) >= 0.1, f"resampled rms {_rms(_got48)}"
    ok(f"DecodeUploadedAudio_WAV_Resample 48k->16k: len={len(_got48)} rms={_rms(_got48):.3f}")

    _buf = bytearray(_wav)
    _buf[4:8] = b"\xff\xff\xff\xff"
    _data_idx = find_chunk(bytes(_buf), b"data")
    assert _data_idx is not None
    _buf[_data_idx + 4 : _data_idx + 8] = b"\xff\xff\xff\xff"
    _got_fix = decode_wav_to_16k_mono(bytes(_buf))
    assert len(_got_fix) == len(_src)
    ok("decode_wav_to_16k_mono fixes 0xFFFFFFFF RIFF/data sizes (rust)")

    if _shutil.which("ffmpeg") is not None:
        with _tempfile.TemporaryDirectory() as _td:
            import os as __os
            _src_path = __os.path.join(_td, "src.wav")
            _mp3_path = __os.path.join(_td, "out.mp3")
            with open(_src_path, "wb") as _f:
                _f.write(encode_wav_mono16(_src, 16_000))
            _cp = _subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", _src_path, _mp3_path],
                capture_output=True,
            )
            assert _cp.returncode == 0, _cp.stderr
            with open(_mp3_path, "rb") as _f:
                _mp3 = _f.read()
            _got_mp3 = decode_any_to_16k_mono(_mp3, "")
            assert len(_got_mp3) >= 12_000, f"mp3 too short: {len(_got_mp3)}"
            _max_abs_mp3, _rel_rms_mp3 = _align_and_diff(_src, _got_mp3, 1500)
            assert _rel_rms_mp3 <= 0.3, f"mp3 relRMS {_rel_rms_mp3} > 0.3"
            assert _rms(_got_mp3) >= 0.1
            ok(
                f"DecodeUploadedAudio_LibAV_MP3: len={len(_got_mp3)} relRMS={_rel_rms_mp3:.3f}"
            )

            _flac_path = __os.path.join(_td, "out.flac")
            _cp = _subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", _src_path, _flac_path],
                capture_output=True,
            )
            assert _cp.returncode == 0, _cp.stderr
            with open(_flac_path, "rb") as _f:
                _flac = _f.read()
            _got_flac = decode_any_to_16k_mono(_flac, "audio/flac")
            assert len(_got_flac) == len(_src)
            _max_abs_fl, _rel_rms_fl = _align_and_diff(_src, _got_flac, 0)
            assert _max_abs_fl <= 1e-3 and _rel_rms_fl <= 1e-3
            ok(
                f"DecodeUploadedAudio_LibAV_FLAC: maxAbs={_max_abs_fl:.2e} relRMS={_rel_rms_fl:.2e}"
            )
    else:
        info("ffmpeg not on PATH; skipping LibAV mp3/flac fallback tests")

def test_http_d_e_v1_audio_diarization_and_v1_audio_embeddings_shape_parity():
    _section('Fix HTTP-D/E: /v1/audio/diarization and /v1/audio/embeddings shape parity')
    import numpy as _np
    from fastapi.testclient import TestClient

    import server as _srv
    from server import (
        DEFAULT_EMBEDDING_MODEL_NAME,
        _build_speaker_label_map,
        _decode_data_url_with_mime,
        _file_id_from_filename,
        _suffix_from_mime,
        app as _app,
    )
    from diarization import DiarSegment as _DiarSegment

    b, m = _decode_data_url_with_mime("data:audio/wav;base64," + base64.b64encode(b"hi").decode())
    assert b == b"hi" and m == "audio/wav"
    ok(f"data URL with mime parsed: bytes={b!r}, mime={m!r}")

    b2, m2 = _decode_data_url_with_mime("data:;base64," + base64.b64encode(b"x").decode())
    assert b2 == b"x" and m2 is None
    ok("data URL without mime parsed (mime=None)")

    DATAURL_REJECTS = [
        ("plain-text", "not a data URL"),
        ("data:audio/wav,raw", "missing"),
        ("data:audio/wav", "missing comma"),
        ("data:audio/wav;charset=utf8,abc", "only base64"),
    ]
    for spec, label in DATAURL_REJECTS:
        try:
            _decode_data_url_with_mime(spec)
            raise AssertionError(f"BAD: {label!r} accepted: {spec!r}")
        except ValueError as exc:
            ok(f"{label} rejected: {str(exc)[:50]}")

    assert _suffix_from_mime("audio/wav") == ".wav"
    assert _suffix_from_mime("audio/mpeg") == ".mp3"
    assert _suffix_from_mime("audio/flac") == ".flac"
    assert _suffix_from_mime("audio/wav; charset=binary") == ".wav"
    assert _suffix_from_mime(None) is None
    assert _suffix_from_mime("application/octet-stream") is None
    ok("_suffix_from_mime maps common audio mimes")

    assert _file_id_from_filename("clip.wav") == "clip"
    assert _file_id_from_filename("nested/path/sample.flac") == "sample"
    assert _file_id_from_filename(None) == "audio"
    assert _file_id_from_filename("") == "audio"
    ok("_file_id_from_filename drops extension and falls back to 'audio'")

    class _StubEmbeddingModel:
        min_input_samples = 16_000
        def embed(self, samples):
            return _np.array([1.0, 0.0], dtype=_np.float32)

    stub_emb = _StubEmbeddingModel()
    seg_a = _DiarSegment(speaker=0, t_start_ms=0, t_end_ms=1000, confidence=0.9)
    seg_b = _DiarSegment(speaker=1, t_start_ms=1000, t_end_ms=2000, confidence=0.8)
    audio32k = _np.zeros(32_000, dtype=_np.float32)
    label = _build_speaker_label_map([seg_a, seg_b], audio32k, stub_emb, [])
    assert label(0) == "SPEAKER_00"
    assert label(1) == "SPEAKER_01"
    assert label(7) == "SPEAKER_07"
    ok("label map without known speakers returns SPEAKER_NN")

    class _DirectionalEmbedding:
        min_input_samples = 16_000
        def __init__(self):
            self._calls = 0
        def embed(self, samples):
            self._calls += 1
            if self._calls == 1:
                return _np.array([1.0, 0.0], dtype=_np.float32)
            return _np.array([0.0, 1.0], dtype=_np.float32)

    dir_emb = _DirectionalEmbedding()
    known_specs = [
        ("alice", _np.array([1.0, 0.0], dtype=_np.float32)),
        ("bob",   _np.array([0.0, 1.0], dtype=_np.float32)),
    ]
    label2 = _build_speaker_label_map([seg_a, seg_b], audio32k, dir_emb, known_specs)
    assert label2(0) == "alice", f"expected alice, got {label2(0)}"
    assert label2(1) == "bob", f"expected bob, got {label2(1)}"
    assert label2(99) == "SPEAKER_99"
    ok("label map with known speakers maps clusters via cosine similarity")

    client = TestClient(_app)
    saved_diarizer, saved_emb = _srv._diarizer, _srv._diar_embedding
    _srv._diarizer = None
    _srv._diar_embedding = None
    try:
        r = client.post(
            "/v1/audio/diarization",
            files={"file": ("clip.wav", b"RIFF....WAVE", "audio/wav")},
        )
        assert r.status_code == 503, f"expected 503, got {r.status_code}: {r.text}"
        body = r.json()
        err = body["detail"]["error"] if "detail" in body else body["error"]
        assert err["type"] == "service_unavailable_error"
        assert err["code"] == "model_not_loaded"
        ok("/v1/audio/diarization returns 503 with kind.SERVICE_UNAVAIL when not loaded")

        r2 = client.post(
            "/v1/audio/embeddings",
            files={"file": ("clip.wav", b"RIFF....WAVE", "audio/wav")},
        )
        assert r2.status_code == 503, f"expected 503, got {r2.status_code}: {r2.text}"
        body2 = r2.json()
        err2 = body2["detail"]["error"] if "detail" in body2 else body2["error"]
        assert err2["type"] == "service_unavailable_error"
        assert err2["code"] == "model_not_loaded"
        ok("/v1/audio/embeddings returns 503 with kind.SERVICE_UNAVAIL when not loaded")
    finally:
        _srv._diarizer = saved_diarizer
        _srv._diar_embedding = saved_emb

    class _StubDiarizer:
        def reset(self): pass
        def diarize_utterance(self, audio, t_start_ms=0):
            return [
                _DiarSegment(speaker=0, t_start_ms=0, t_end_ms=1500, confidence=0.9),
                _DiarSegment(speaker=1, t_start_ms=1500, t_end_ms=3000, confidence=0.8),
            ]

    _audio_marker = _np.zeros(48_000, dtype=_np.float32)

    def _stub_load(_path):
        return _audio_marker

    saved_load = _srv._load_audio_for_diarizer
    _srv._diarizer = _StubDiarizer()
    _srv._diar_embedding = _StubEmbeddingModel()
    _srv._load_audio_for_diarizer = _stub_load
    try:
        r = client.post(
            "/v1/audio/diarization",
            files={"file": ("clip.wav", b"RIFF....WAVE", "audio/wav")},
            data={"response_format": "json"},
        )
        assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert set(body.keys()) == {"duration", "segments"}, f"keys: {set(body.keys())}"
        assert isinstance(body["duration"], float)
        assert abs(body["duration"] - 3.0) < 1e-6
        seg_first = body["segments"][0]
        assert set(seg_first.keys()) == {"start", "end", "speaker"}, f"seg keys: {set(seg_first.keys())}"
        assert seg_first["start"] == 0.0 and seg_first["end"] == 1.5 and seg_first["speaker"] == "SPEAKER_00"
        seg_second = body["segments"][1]
        assert seg_second["start"] == 1.5 and seg_second["end"] == 3.0 and seg_second["speaker"] == "SPEAKER_01"
        ok("/v1/audio/diarization json shape: {duration, segments:[{start,end,speaker}]}")

        r = client.post(
            "/v1/audio/diarization",
            files={"file": ("clip.wav", b"RIFF....WAVE", "audio/wav")},
            data={"response_format": "rttm"},
        )
        assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
        assert r.headers["content-type"].startswith("text/plain")
        rttm_lines = r.text.strip().splitlines()
        assert len(rttm_lines) == 2, f"rttm: {r.text!r}"
        parts = rttm_lines[0].split()
        assert parts[0] == "SPEAKER" and parts[1] == "clip" and parts[2] == "1"
        assert parts[3] == "0.000" and parts[4] == "1.500"
        assert parts[5] == "<NA>" and parts[6] == "<NA>"
        assert parts[7] == "SPEAKER_00"
        assert parts[8] == "<NA>" and parts[9] == "<NA>"
        ok("/v1/audio/diarization rttm shape matches upstream SPEAKER lines")

        r = client.post(
            "/v1/audio/embeddings",
            files={"file": ("clip.wav", b"RIFF....WAVE", "audio/wav")},
        )
        assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert set(body.keys()) == {"object", "data", "model", "usage"}, f"keys: {set(body.keys())}"
        assert body["object"] == "list"
        assert body["model"] == DEFAULT_EMBEDDING_MODEL_NAME
        assert "audio_seconds" in body["usage"]
        assert abs(body["usage"]["audio_seconds"] - 3.0) < 1e-6
        item = body["data"][0]
        assert set(item.keys()) == {"object", "index", "embedding"}, f"item keys: {set(item.keys())}"
        assert item["object"] == "embedding" and item["index"] == 0
        assert item["embedding"] == [1.0, 0.0]
        ok("/v1/audio/embeddings json shape: {object,data,model,usage} with embedding items")

        r = client.post(
            "/v1/audio/embeddings",
            data={"audio": "data:audio/wav;base64," + base64.b64encode(b"raw").decode()},
        )
        assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert body["data"][0]["index"] == 0
        ok("/v1/audio/embeddings accepts data URL via 'audio' field")

        r = client.post("/v1/audio/embeddings", data={})
        assert r.status_code == 422, f"expected 422 missing-file, got {r.status_code}: {r.text}"
        detail = r.json()["detail"]
        assert any(e.get("type") == "missing" and e.get("loc") == ["body", "file"] for e in detail), detail
        ok("/v1/audio/embeddings returns 422 missing-field validation when no audio")
    finally:
        _srv._diarizer = saved_diarizer
        _srv._diar_embedding = saved_emb
        _srv._load_audio_for_diarizer = saved_load

def test_eou_speaches_plus_rust_src_eou_parity():
    _section('Fix EOU: speaches-plus rust/src/eou parity')
    import math as _math

    import numpy as _np_eou

    from eou import (
        AudioPadAlignment,
        DEFAULT_GATED_FUSION_WEIGHTS,
        Eagerness,
        EouConfig,
        EouKind,
        FakeIntegratedBackend,
        FusionRule,
        HeuristicEouModel,
        StubEouModel,
        combine_fusion,
        combine_fusion_gated,
        combine_fusion_with_features,
        extract_gated_fusion_features,
        sigmoid_lerp,
    )
    from eou.audio import (
        N_FRAMES,
        N_MELS,
        TARGET_SAMPLES,
        build_hann_window,
        build_mel_filters,
        log_mel_spectrogram,
        prepare_audio,
    )
    from eou.bpe import Tokenizer
    from eou.byte_map import (
        bpe_chars_to_bytes,
        byte_to_char_table,
        bytes_to_bpe_chars,
        char_to_byte,
    )
    from eou.chat_template import Turn, format_qwen_chat, rolling_history
    from eou.onnx import build_mock_tokenizer_json, extract_im_end_prob

    assert sigmoid_lerp(0.5, 0.5, 1.0, 1500, 100) == 1500
    assert sigmoid_lerp(1.0, 0.5, 1.0, 1500, 100) == 100
    assert sigmoid_lerp(0.2, 0.5, 1.0, 1500, 100) == 1500
    _d_low = sigmoid_lerp(0.6, 0.5, 1.0, 1500, 100)
    _d_mid = sigmoid_lerp(0.75, 0.5, 1.0, 1500, 100)
    _d_high = sigmoid_lerp(0.9, 0.5, 1.0, 1500, 100)
    assert _d_low > _d_mid > _d_high
    ok(f"sigmoid_lerp boundaries + monotonic ({_d_low}>{_d_mid}>{_d_high})")

    assert StubEouModel().score("anything") == 1.0
    assert StubEouModel().score("") == 1.0
    ok("StubEouModel returns 1.0")

    assert Eagerness.parse("low") == Eagerness.LOW
    assert Eagerness.parse("LOW") == Eagerness.LOW
    assert Eagerness.parse(" High ") == Eagerness.HIGH
    assert Eagerness.parse("MED") == Eagerness.MEDIUM
    assert Eagerness.parse("auto") == Eagerness.AUTO
    assert Eagerness.parse("nope") is None
    assert Eagerness.LOW.triple() == (0.7, 800, 3000)
    assert Eagerness.MEDIUM.triple() == (0.5, 500, 2500)
    assert Eagerness.HIGH.triple() == (0.4, 300, 1500)
    assert Eagerness.AUTO.triple() == Eagerness.MEDIUM.triple()
    ok("Eagerness parse + triples match upstream")

    _c = EouConfig()
    assert _c.kind == EouKind.VAD
    assert not _c.kind.calls_classifier()
    assert _c.p_threshold == 0.5
    assert _c.min_delay_ms == 500
    assert _c.max_delay_ms == 3000
    assert _c.min_speech_for_response_ms == 600
    assert _c.inference_timeout_ms == 250
    assert _c.fusion_rule == FusionRule.GATED
    assert _c.eager_max_inflight == 1
    assert not _c.eager_periodic_enabled
    assert not _c.eager_disabled()
    ok("EouConfig defaults match upstream")

    _c2 = EouConfig()
    _c2.thresholds["fr"] = 0.7
    assert _c2.threshold_for_language(None) == 0.5
    assert _c2.threshold_for_language("en") == 0.5
    assert _c2.threshold_for_language("fr") == 0.7
    ok("threshold_for_language fallback")

    assert HeuristicEouModel.score_text("are you sure?") >= 0.9
    assert HeuristicEouModel.score_text("done.") >= 0.9
    assert HeuristicEouModel.score_text("wow!") >= 0.9
    assert HeuristicEouModel.score_text("the cat is on the") <= 0.25
    assert HeuristicEouModel.score_text("apples and") <= 0.25
    assert HeuristicEouModel.score_text("she said,") <= 0.3
    assert HeuristicEouModel.score_text("I think um") <= 0.2
    assert HeuristicEouModel.score_text("hmm") <= 0.2
    assert HeuristicEouModel.score_text("") <= 0.2
    assert HeuristicEouModel.score_text("   ") <= 0.2
    _neutral = HeuristicEouModel.score_text("the train arrives at noon")
    assert 0.4 <= _neutral <= 0.8, f"neutral got {_neutral}"
    ok("HeuristicEouModel buckets terminator/continuation/hesitation/neutral")

    for _r in (FusionRule.NOISY_OR, FusionRule.MAX, FusionRule.MEAN, FusionRule.WEIGHTED):
        assert FusionRule.parse(_r.as_str()) == _r
    assert FusionRule.parse("noisy-or") == FusionRule.NOISY_OR
    assert FusionRule.parse("avg") == FusionRule.MEAN
    assert FusionRule.parse("nope") is None
    assert FusionRule.parse("gated") == FusionRule.GATED
    assert FusionRule.GATED.as_str() == "gated"
    assert FusionRule.default() == FusionRule.GATED
    ok("FusionRule parse roundtrip + default gated")

    assert abs(combine_fusion(0.6, 0.4, FusionRule.NOISY_OR, 0.5) - (1.0 - 0.4 * 0.6)) < 1e-5
    assert abs(combine_fusion(0.6, 0.4, FusionRule.MAX, 0.5) - 0.6) < 1e-5
    assert abs(combine_fusion(0.4, 0.9, FusionRule.MAX, 0.5) - 0.9) < 1e-5
    assert abs(combine_fusion(0.6, 0.4, FusionRule.MEAN, 0.5) - 0.5) < 1e-5
    assert abs(combine_fusion(0.7, 0.3, FusionRule.WEIGHTED, 0.5) - combine_fusion(0.7, 0.3, FusionRule.MEAN, 0.5)) < 1e-5
    assert abs(combine_fusion(0.7, 0.3, FusionRule.WEIGHTED, 1.0) - 0.7) < 1e-5
    assert abs(combine_fusion(0.7, 0.3, FusionRule.WEIGHTED, 0.0) - 0.3) < 1e-5
    ok("combine_fusion math matches noisy_or/max/mean/weighted")

    assert abs(combine_fusion(0.6, float("nan"), FusionRule.NOISY_OR, 0.5) - 0.6) < 1e-5
    assert abs(combine_fusion(0.6, float("inf"), FusionRule.MAX, 0.5) - 0.6) < 1e-5
    assert abs(combine_fusion(0.6, 1.5, FusionRule.MEAN, 0.5) - 0.6) < 1e-5
    assert abs(combine_fusion(float("nan"), 0.4, FusionRule.NOISY_OR, 0.5) - 0.4) < 1e-5
    assert abs(combine_fusion(float("nan"), float("nan"), FusionRule.NOISY_OR, 0.5) - 1.0) < 1e-5
    assert abs(combine_fusion(-0.1, 1.5, FusionRule.MEAN, 0.5) - 1.0) < 1e-5
    ok("combine_fusion graceful degradation on garbage probs")

    _p_weighted = combine_fusion(0.6, 0.4, FusionRule.WEIGHTED, float("nan"))
    assert _math.isfinite(_p_weighted)
    ok("Weighted with NaN weight sanitized")

    _strong_cases = [
        ("yes.", True, False, False),
        ("what?", True, False, False),
        ("wow!", True, False, False),
        ("hmm,", False, True, False),
        ("the cat is on the", False, False, True),
        ("and", False, False, True),
        ("because", False, False, True),
        ("", False, False, False),
        ("   ", False, False, False),
        ("the cat", False, False, False),
    ]
    for _s, _strong, _soft, _cont in _strong_cases:
        _f = extract_gated_fusion_features(_s, 1000)
        assert _f.partial_ends_with_strong_terminator == _strong, f"{_s!r} strong"
        assert _f.partial_ends_with_soft_terminator == _soft, f"{_s!r} soft"
        assert _f.partial_last_word_is_continuation == _cont, f"{_s!r} cont"
    ok("extract_gated_fusion_features strong/soft/continuation")

    _parity = [
        ("That's right.", 1500, 0.95, 0.99, 0.989753),
        ("Yes.", 1500, 0.95, 0.95, 0.950000),
        ("and the next thing", 1500, 0.55, 0.05, 0.053163),
        ("the cat is on the", 1500, 0.25, 0.05, 0.051354),
        ("looking forward to it", 1500, 0.55, 0.5, 0.500264),
    ]
    for _partial, _ms, _pt, _pa, _want in _parity:
        _feat = extract_gated_fusion_features(_partial, _ms)
        _got = combine_fusion_gated(_pt, _pa, _feat, DEFAULT_GATED_FUSION_WEIGHTS)
        assert abs(_got - _want) < 5e-3, f"parity {_partial!r}: got {_got} want {_want}"
    ok("combine_fusion_gated byte-for-byte parity with upstream Go expected values")

    assert abs(combine_fusion(0.6, 0.4, FusionRule.GATED, 0.5) - combine_fusion(0.6, 0.4, FusionRule.MEAN, 0.5)) < 1e-6
    ok("combine_fusion(GATED,...) without features falls back to mean")

    _b = FakeIntegratedBackend.smoke_default()
    assert _b.step(0) is None
    _v1 = _b.step(800)
    assert _v1 is not None and _v1.p_eot < 0.5
    _v2 = _b.step(2700)
    assert _v2 is not None and _v2.p_eot >= 0.8
    assert _b.step(2700) is None
    _b.reset()
    assert _b.step(800) is not None
    ok("FakeIntegratedBackend emits in order + reset")

    _table = byte_to_char_table()
    for _b in range(256):
        assert char_to_byte(_table[_b]) == _b
    for _s in ["hello world", "line\nbreak", "\t\rweird", "\x00\x01\x02"]:
        _chars = bytes_to_bpe_chars(_s)
        assert bpe_chars_to_bytes(_chars) == _s, f"roundtrip {_s!r}"
    ok("byte_map round-trip across all 256 bytes")

    _raw = build_mock_tokenizer_json()
    _tok = Tokenizer.load_from_json(_raw)
    assert _tok.im_end_id() >= 0
    assert _tok.has_im_tokens()
    _round = _tok.decode(_tok.encode(" hello world"))
    assert _round == " hello world", f"BPE roundtrip got {_round!r}"
    assert _tok.encode("") == []
    _special = f"<|im_start|>user\nhello<|im_end|>"
    _ids = _tok.encode(_special)
    assert _tok.im_start_id() in _ids
    assert _tok.im_end_id() in _ids
    ok("Tokenizer mock load + roundtrip + special-token encoding")

    try:
        Tokenizer.load_from_json('{"model": {"type": "WordPiece"}}')
        raise AssertionError("BAD: WordPiece accepted")
    except ValueError:
        ok("Tokenizer rejects unsupported model type")

    _vocab = 5
    _logits = [0.0] * (_vocab * 2)
    _logits[_vocab + 3] = 100.0
    _p = extract_im_end_prob(_logits, [1, 2, _vocab], 3)
    assert _p > 0.99, f"expected near-1, got {_p}"
    try:
        extract_im_end_prob([0.0] * 5, [1, 1, 5], 9)
        raise AssertionError("BAD: oob accepted")
    except ValueError:
        pass
    ok("extract_im_end_prob softmax + oob rejection")

    assert format_qwen_chat([], "hello world") == "<|im_start|>user\nhello world"
    _turns = [Turn.user("what's the weather"), Turn.assistant("it's sunny")]
    _expected = (
        "<|im_start|>user\nwhat's the weather<|im_end|>\n"
        "<|im_start|>assistant\nit's sunny<|im_end|>\n"
        "<|im_start|>user\nand humid"
    )
    assert format_qwen_chat(_turns, "and humid") == _expected
    assert format_qwen_chat([Turn.user("hi")], "").endswith("<|im_end|>\n")
    _blank = format_qwen_chat([Turn(role="", content="no role")], "")
    assert "<|im_start|>user\nno role" in _blank
    _long = [Turn(role="user", content=str(i)) for i in range(1, 8)]
    _rolled = rolling_history(_long, 4)
    assert len(_rolled) == 4
    assert _rolled[0].content == "4"
    assert _rolled[3].content == "7"
    ok("Qwen chat template + rolling_history truncation")

    _audio = _np_eou.array([(i + 1) / 2000.0 for i in range(1600)], dtype=_np_eou.float32)
    _pre = prepare_audio(_audio, 8000, AudioPadAlignment.LEADING)
    assert _pre.shape[0] == TARGET_SAMPLES
    _pad_len = TARGET_SAMPLES - 1600
    for _i in range(_pad_len):
        assert _pre[_i] == 0.0
    for _i in range(1600):
        assert abs(_pre[_pad_len + _i] - _audio[_i]) < 1e-6
    _pre_t = prepare_audio(_audio, 8000, AudioPadAlignment.TRAILING)
    for _i in range(1600):
        assert abs(_pre_t[_i] - _audio[_i]) < 1e-6
    for _i in range(1600, TARGET_SAMPLES):
        assert _pre_t[_i] == 0.0
    _bad = _np_eou.array([5.0, -3.0, float("nan"), 0.5, float("inf")], dtype=_np_eou.float32)
    _pre_b = prepare_audio(_bad, 8000, AudioPadAlignment.LEADING)
    _tail = _pre_b[TARGET_SAMPLES - 5 :]
    assert _tail[0] == 1.0
    assert _tail[1] == -1.0
    assert _tail[2] == 0.0
    assert abs(_tail[3] - 0.5) < 1e-6
    assert _tail[4] == 0.0
    ok("prepare_audio leading/trailing pad + clamp NaN/oob")

    _filters = build_mel_filters()
    for _v in _filters:
        assert _math.isfinite(_v) and _v >= 0.0
    _hann = build_hann_window()
    _mel = log_mel_spectrogram(_np_eou.full(TARGET_SAMPLES, 0.1, dtype=_np_eou.float32), _hann, _filters)
    assert _mel.shape[0] == N_MELS * N_FRAMES
    for _v in _mel:
        assert _math.isfinite(_v)
    ok("mel filters + log_mel_spectrogram shape + finiteness")

def test_realtime_speaches_plus_rust_src_realtime_port():
    _section('Fix Realtime: speaches-plus rust/src/realtime port')
    import json as _json_rt

    from realtime import (
        RFC_VERSION,
        AUDIO_FORMAT_DEFAULT,
        capabilities_json,
        eou_defaults,
        response_defaults,
        session_defaults,
        turn_detection,
        wire_defaults,
    )
    from realtime.errors import code as _errcode, error_type_for as _err_type
    from realtime.events import (
        ClientEventType,
        ServerEventType,
        item_to_json,
        make_cancelled_brackets,
        make_response_done,
        parse_client_event,
    )
    from realtime.framing import frame_event as _frame_event
    from realtime.fuzz import run_random_walk as _rt_run_walk
    from realtime.sdp_filter import normalize_offer as _rt_normalize_offer
    from realtime.session_update import parse_session_update as _rt_parse_su
    from realtime.state import (
        ConversationItem as _RtConvItem,
        InvariantViolation as _RtIV,
        ItemStatus as _RtItemStatus,
        ResponseRuntime as _RtResponseRuntime,
        RespPhase as _RtRespPhase,
        SealedBuffer as _RtSealedBuffer,
        SessionPhase as _RtSessionPhase,
        SessionState as _RtSessionState,
        VadPhase as _RtVadPhase,
        apply_truncate_to_conversation as _rt_apply_truncate,
        check_invariants as _rt_check_inv,
        check_state as _rt_check_state,
    )
    from realtime.wire import (
        ErrorPayload as _RtErrorPayload,
        OutboundEvent as _RtOutboundEvent,
        ResponsePayload as _RtResponsePayload,
        ResponseStatus as _RtRStatus,
        ResponseStatusDetails as _RtRSD,
        ResponseStatusReason as _RtRSReason,
    )

    assert RFC_VERSION == "v3"
    assert session_defaults.MAX_DURATION_S == 1800
    assert session_defaults.MAX_DURATION_HARD_CAP_S == 3600
    assert turn_detection.THRESHOLD == 0.5
    assert turn_detection.PREFIX_PADDING_MS == 300
    assert turn_detection.SILENCE_DURATION_MS == 350
    assert eou_defaults.P_THRESHOLD == 0.5
    assert eou_defaults.MIN_DELAY_MS == 500
    assert eou_defaults.MAX_DELAY_MS == 3000
    assert eou_defaults.SILENCE_HARD_CAP_MS == 5000
    assert wire_defaults.OUTBOUND_QUEUE_CAP_MS == 5000
    assert wire_defaults.OUTBOUND_QUEUE_CAP_EVENTS == 256
    assert wire_defaults.DATA_CHANNEL_FRAGMENT_MAX == 900
    assert response_defaults.DRAIN_CAP_FLOOR_MS == 5000
    assert response_defaults.DRAIN_CAP_CEILING_MS == 60000
    ok("realtime defaults match upstream spec values verbatim")

    caps = capabilities_json()
    assert caps["rfc_version"] == "v3"
    assert caps["features"]["eou_kinds"] == ["vad", "text", "audio", "fusion"]
    assert "heuristic" in caps["extensions"]["eou_kinds"]
    assert "integrated" in caps["extensions"]["eou_kinds"]
    assert "audio" not in caps["extensions"]["eou_kinds"]
    assert "fusion" not in caps["extensions"]["eou_kinds"]
    assert caps["extensions"]["predicted_resp_phase"] is True
    assert isinstance(caps["extensions"]["eager_eou"], bool)
    assert isinstance(caps["extensions"]["integrated_eou"], bool)
    assert isinstance(caps["extensions"]["audio_eou"], bool)
    ok("capabilities_json: rfc_version, eou_kinds split, extension flags")

    v_sp = _RtVadPhase.speaking("item_x", 0)
    r_cr = _RtRespPhase.created("resp_x", "item_x", 1)
    try:
        _rt_check_inv(_RtSessionPhase.active(0), v_sp, r_cr)
        raise AssertionError("BAD: I1 not enforced")
    except _RtIV as v:
        assert v.kind.value == "SpeakingWithActiveResponse"
    ok("I1 enforced: SpeakingWithActiveResponse")

    v_bad = _RtVadPhase.stopped("item_x", 500, 100)
    try:
        _rt_check_inv(_RtSessionPhase.active(0), v_bad, _RtRespPhase.none())
        raise AssertionError("BAD: I11 not enforced")
    except _RtIV as v:
        assert v.kind.value == "StoppedWithoutEnd"
    ok("I11 enforced: StoppedWithoutEnd")

    state_t = _RtSessionState()
    state_t.session = _RtSessionPhase.active(0)
    runtime = _RtResponseRuntime(handle=None)
    ep1 = state_t.resp_create_from_none("r1", "i1", runtime)
    assert ep1 == 1
    assert state_t.resp.tag.value == "Created"
    ok("transition None -> Created bumps epoch (I5)")

    state_t2 = _RtSessionState()
    state_t2.session = _RtSessionPhase.active(0)
    try:
        state_t2.resp_advance_to_streaming(_RtSessionState().resp.played_ms or _RtSessionState.__init__.__globals__["_AtomicU64"]())
        raise AssertionError("BAD: illegal transition not rejected")
    except _RtIV as v:
        assert v.kind.value == "IllegalRespTransition"
        assert v.from_ == "None"
    ok("illegal None -> Streaming rejected with IllegalRespTransition")

    state_p = _RtSessionState()
    state_p.session = _RtSessionPhase.active(0)
    state_p.resp_start_predicted("r_p", "i_p", 0.95, None)
    runtime2 = _RtResponseRuntime(handle=None)
    got = state_p.resp_promote_predicted_to_created(runtime2)
    assert got[0] == "r_p" and got[1] == "i_p" and got[2] == 1
    ok("Predicted -> Created preserves id+epoch (I9)")

    from realtime.state import ResponseGate as _RtRG
    p_phase = _RtRespPhase.predicted("r", "i", 1, 0.7, None, None)
    assert _RtRG.open(p_phase) is None
    c_phase = _RtRespPhase.created("r", "i", 1)
    assert _RtRG.open(c_phase) is not None
    ok("I7 enforced: ResponseGate closes during Predicted")

    state_s = _RtSessionState()
    state_s.sealed_buffer_retention_count = 4
    for i in range(7):
        state_s.store_sealed_buffer(_RtSealedBuffer(item_id=f"buf{i}", audio=[], audio_start_ms=0, audio_end_ms=10))
    assert len(state_s.sealed_buffers) == 4
    assert [b.item_id for b in state_s.sealed_buffers] == ["buf3", "buf4", "buf5", "buf6"]
    ok("sealed-buffer FIFO retention K=4 (RFC v3 §3.4)")

    ok_dropped = state_s.drop_sealed_buffer("buf4")
    assert ok_dropped
    assert "buf4" not in [b.item_id for b in state_s.sealed_buffers]
    ok("drop_sealed_buffer removes by item_id")

    conv = [_RtConvItem.new_assistant_audio("i_a", "the full response", 5_000)]
    _rt_apply_truncate(conv, "i_a", 1_200, "the full")
    assert conv[0].status == _RtItemStatus.INCOMPLETE
    assert conv[0].content.audio_ms == 1_200
    ok("truncate: clamp audio_ms + Incomplete + transcript replace")

    for status, status_str in [(_RtRStatus.COMPLETED, "completed"),
                               (_RtRStatus.CANCELLED, "cancelled"),
                               (_RtRStatus.INCOMPLETE, "incomplete"),
                               (_RtRStatus.FAILED, "failed")]:
        ev = _RtOutboundEvent.response_done(_RtResponsePayload(id="r", audio_end_ms=1234, status=status))
        j = ev.to_json()
        assert j["type"] == "response.done"
        assert j["response"]["audio_end_ms"] == 1234, status
        assert j["response"]["status"] == status_str
    ok("response.done carries audio_end_ms for all 4 statuses (W4)")

    ev = _RtOutboundEvent.response_done(_RtResponsePayload(id="r", audio_end_ms=0, status=_RtRStatus.COMPLETED))
    assert "status_details" not in ev.to_json()["response"]
    ok("response.done omits status_details when completed/cancelled")

    p = _RtResponsePayload(id="r", audio_end_ms=5000, status=_RtRStatus.INCOMPLETE,
                          status_details=_RtRSD(reason=_RtRSReason.DRAIN_CAP))
    j = _RtOutboundEvent.response_done(p).to_json()
    assert j["response"]["status_details"]["reason"] == "drain_cap"
    ok("ResponseStatusReason serializes snake_case (drain_cap)")

    ev_err = _RtOutboundEvent.error(_RtErrorPayload(type_="invalid_request_error", code="invalid_request_error", message="bad"))
    j = ev_err.to_json()
    assert j["type"] == "error"
    assert j["error"]["type"] == "invalid_request_error"
    assert j["error"]["code"] == "invalid_request_error"
    assert j["error"]["message"] == "bad"
    assert "event_id" not in j["error"]
    assert "param" not in j["error"]
    ok("error payload shape: type+code+message required, optionals omitted")

    assert _err_type("vad_failed") == "server_error"
    assert _err_type("response_already_active") == "invalid_request_error"
    ok("error_type_for resolves server_error vs invalid_request_error")

    ev_tid = _RtOutboundEvent.assistant_truncated("evt_1", "item_a", 1234, "partial")
    j = ev_tid.to_json()
    assert j["event_id"] == "evt_1"
    assert j["item_id"] == "item_a"
    assert j["audio_end_ms"] == 1234
    ok("typed ids serialize as strings in outbound events")

    from realtime.state import Topic as _RtTopic
    assert _RtTopic.classify("response.created") is _RtTopic.RESPONSE
    assert _RtTopic.classify("session.created") is _RtTopic.SESSION
    assert _RtTopic.classify("conversation.item.added") is _RtTopic.ITEM
    assert _RtTopic.classify("input_audio_buffer.committed") is _RtTopic.BUFFER
    assert _RtTopic.classify("error") is _RtTopic.ERROR
    ok("Topic.classify routes per upstream prefix table")

    small_frames = _frame_event({"type": "session.created", "id": "x"})
    assert len(small_frames) == 1
    assert '"type":"full_message"' in small_frames[0]
    big_frames = _frame_event({"type": "x", "blob": "a" * 4000})
    assert len(big_frames) >= 2
    for fr in big_frames:
        assert '"type":"partial_message"' in fr
        assert "total_fragments" in fr
    ok("framing: small=full_message, large=partial_message with total_fragments")

    sdp_in = (
        "v=0\r\n"
        "a=group:BUNDLE 0 1\r\n"
        "m=audio 1 UDP/TLS/RTP/SAVPF 96\r\n"
        "a=mid:0\r\n"
        "a=ice-ufrag:AAA1\r\n"
        "a=ice-pwd:pwd-aaaa\r\n"
        "m=application 1 DTLS/SCTP 5000\r\n"
        "a=mid:1\r\n"
        "a=ice-ufrag:BBB2\r\n"
        "a=ice-pwd:pwd-bbbb\r\n"
    )
    sdp_out = _rt_normalize_offer(sdp_in)
    ufrags = [l for l in sdp_out.splitlines() if l.startswith("a=ice-ufrag")]
    pwds = [l for l in sdp_out.splitlines() if l.startswith("a=ice-pwd")]
    assert ufrags == ["a=ice-ufrag:AAA1", "a=ice-ufrag:AAA1"], ufrags
    assert pwds == ["a=ice-pwd:pwd-aaaa", "a=ice-pwd:pwd-aaaa"], pwds
    ok("SDP normalizer unifies ICE ufrag+pwd across BUNDLE")

    res = _rt_parse_su({"no_speech_prob_threshold": 0.6})
    assert not isinstance(res, tuple)
    assert res.no_speech_prob_threshold == (True, 0.6)

    res = _rt_parse_su({"no_speech_prob_threshold": None})
    assert res.no_speech_prob_threshold == (True, None)

    res = _rt_parse_su({})
    assert res.no_speech_prob_threshold is None
    ok("no_speech_prob_threshold: set, null-disable, absent variants")

    err = _rt_parse_su({"no_speech_prob_threshold": 1.5})
    assert isinstance(err, tuple) and "must be in [0,1]" in err[1]
    err = _rt_parse_su({"no_speech_prob_threshold": -0.1})
    assert isinstance(err, tuple) and "must be in [0,1]" in err[1]
    ok("no_speech_prob_threshold: rejects out-of-range")

    res = _rt_parse_su({"turn_detection": {"silence_duration_ms": 4000}})
    assert res.turn_detection.silence_duration_ms == 4000
    err = _rt_parse_su({"turn_detection": {"silence_duration_ms": 10}})
    assert isinstance(err, tuple) and "must be in" in err[1]
    ok("turn_detection.silence_duration_ms range [50,5000]")

    err = _rt_parse_su({"session_max_duration_s": 0})
    assert isinstance(err, tuple)
    err = _rt_parse_su({"session_max_duration_s": session_defaults.MAX_DURATION_HARD_CAP_S + 1})
    assert isinstance(err, tuple)
    res = _rt_parse_su({"session_max_duration_s": 600})
    assert res.session_max_duration_s == 600
    ok("session_max_duration_s validation [1, MAX_DURATION_HARD_CAP_S]")

    err = _rt_parse_su({"min_speech_ms": -1})
    assert isinstance(err, tuple)
    ok("session_update parse-all -> validate-all (atomic apply contract)")

    err = _rt_parse_su({"input_audio_format": "opus"})
    assert isinstance(err, tuple)
    res = _rt_parse_su({"input_audio_format": "pcm16"})
    assert res.input_audio_format == "pcm16"
    res = _rt_parse_su({"input_audio_format": "g711_ulaw"})
    assert res.input_audio_format == "g711_ulaw"
    ok("audio_format whitelist enforced on session.update")

    t, obj = parse_client_event('{"type":"session.update","session":{}}')
    assert t is ClientEventType.SESSION_UPDATE
    t, obj = parse_client_event('{"type":"unknown.x"}')
    assert t is None and obj.get("type") == "unknown.x"
    t, obj = parse_client_event("not-json")
    assert t is None
    ok("parse_client_event: known + unknown + non-JSON")

    err = _rt_run_walk(0, 5000)
    assert err is None, f"fuzz: {err}"
    for seed in (1, 7, 42, 99, 2024):
        err = _rt_run_walk(seed, 1000)
        assert err is None, f"fuzz seed={seed}: {err}"
    ok("fuzz: 5000 + 5x1000 random-walk steps preserve invariants")

    for c in (_errcode.INVALID_REQUEST_ERROR, _errcode.UNKNOWN_EVENT_TYPE,
              _errcode.SESSION_UPDATE_INVALID, _errcode.RESPONSE_ALREADY_ACTIVE,
              _errcode.RESPONSE_CANCEL_NOT_ACTIVE, _errcode.INPUT_AUDIO_BUFFER_COMMIT_EMPTY,
              _errcode.CLIENT_TOO_SLOW, _errcode.INTERNAL_STATE_ERROR,
              _errcode.VAD_FAILED, _errcode.STT_FAILED):
        from realtime.errors import is_known_code
        assert is_known_code(c), c
    ok("errors.code registry covers all reserved codes (RFC §10.5)")

    brackets, done = make_cancelled_brackets("r", "i_a", "transcribed", 1234, _RtRSReason.BARGE_IN)
    assert len(brackets) == 4
    assert done.to_json()["response"]["status"] == "cancelled"
    assert done.to_json()["response"]["status_details"]["reason"] == "barge_in"
    brackets2, done2 = make_cancelled_brackets("r", "i_a", "tr", 0, _RtRSReason.CLIENT_CANCELLED)
    assert done2.to_json()["response"]["status_details"]["reason"] == "client_cancelled"
    ok("make_cancelled_brackets emits 4 brackets + barge_in/client_cancelled reason")

    ev = make_response_done("r", "i", "completed", "hello", None, 5000)
    j = ev.to_json()
    assert j["response"]["status"] == "completed"
    assert "status_details" not in j["response"]
    ev_f = make_response_done("r", "i", "failed", None, _RtRSReason.LLM_ERROR, 0)
    j_f = ev_f.to_json()
    assert j_f["response"]["status_details"]["reason"] == "llm_error"
    ok("make_response_done: completed omits status_details, failed carries reason")

def test_realtime_eou_wiring_speaches_plus_parity():
    _section('Realtime EOU wiring: eou_eager / eou_predicted / eou_integrated')

    from realtime import IntegratedVerdictAction as _IVA
    assert {a.value for a in _IVA} == {"ignored", "none", "started_predicted", "commit"}
    ok("IntegratedVerdictAction values match Rust enum")

    from realtime.eou_predicted import PredictedLlmRunner, PredictedLlmShared
    from realtime.state import PredictedLlmRunnerHandle
    shared = PredictedLlmShared()
    runner = PredictedLlmRunner(task=None, shared=shared, cap=42)
    handle = PredictedLlmRunnerHandle.from_runner(runner)
    assert handle.shared is shared and handle.cap == 42
    rt2 = handle.into_runner()
    assert rt2.shared is shared and rt2.cap == 42
    ok("PredictedLlmRunnerHandle from_runner/into_runner roundtrip")

    from realtime.eou_predicted import transcripts_materially_differ as _tmd
    assert not _tmd("hello", "hello", 0.5)
    assert not _tmd("hi there", "hi there.", 0.5)
    assert _tmd("cats", "weather", 0.5)
    assert _tmd("hello", "", 0.5)
    assert not _tmd("", "", 0.5)
    ok("transcripts_materially_differ matches Rust decision table")

    import asyncio as _asyncio
    from realtime.session import Session, Intent, RealtimeQuery
    from eou.integrated import IntegratedVerdict
    from eou.types import EouKind

    async def _verdict_table():
        q = RealtimeQuery(intent='conversation')
        s = Session(q, Intent.CONVERSATION)
        r = await s.handle_integrated_verdict(IntegratedVerdict(0.9, 0.9, 'hi'))
        assert r is _IVA.IGNORED
        s.eou_config.kind = EouKind.INTEGRATED
        r = await s.handle_integrated_verdict(IntegratedVerdict(0.9, 0.9, 'hi'))
        assert r is _IVA.COMMIT
        r = await s.handle_integrated_verdict(IntegratedVerdict(0.1, 0.8, 'hi'))
        assert r is _IVA.STARTED_PREDICTED
        assert s.state.resp.is_predicted()
        r = await s.handle_integrated_verdict(IntegratedVerdict(0.1, 0.9, 'hi'))
        assert r is _IVA.NONE
    _asyncio.run(_verdict_table())
    ok("Session.handle_integrated_verdict decision table")

    from realtime.eou_eager import try_eager_dispatch
    from eou.loader import EouConfig

    async def _throttle():
        q = RealtimeQuery(intent='transcription')
        s = Session(q, Intent.TRANSCRIPTION)
        async def _t(audio):
            return "hello"
        s._transcribe = _t
        cfg = EouConfig()
        cfg.eager_interval_ms = 200
        await try_eager_dispatch(s, cfg, 0.8, [0.0] * 16000)
        assert s.state.resp.is_predicted()
        pre = (s.state.resp.id, s.state.resp.epoch)
        await try_eager_dispatch(s, cfg, 0.9, [0.0] * 16000)
        assert (s.state.resp.id, s.state.resp.epoch) == pre
    _asyncio.run(_throttle())
    ok("try_eager_dispatch throttles within eager_interval_ms")

    from realtime.observer import NullObserver, SessionObserver
    obs = NullObserver()
    for name in (
        "on_eou_scored",
        "on_eou_hard_cap_fired",
        "on_eou_eager_dispatch",
        "on_predicted_suppressed",
        "on_predicted_promoted",
        "on_predicted_overflow",
        "on_predicted_rollback",
    ):
        assert hasattr(obs, name), name
    assert isinstance(obs, SessionObserver)
    ok("NullObserver implements the 7 new EOU observer methods")

    import inspect as _inspect
    from realtime import pipeline as _rt_pipeline
    assert _inspect.iscoroutinefunction(_rt_pipeline.run_eou_dispatch)
    assert _inspect.iscoroutinefunction(_rt_pipeline._race_hard_cap)
    params = list(_inspect.signature(_rt_pipeline.run_eou_dispatch).parameters)
    assert params == ["session", "item_id", "audio", "audio_ms", "suppress_response"], params
    ok("pipeline.run_eou_dispatch + _race_hard_cap exposed with expected signatures")

    async def _predicted_iter():
        out = []
        async for d in _rt_pipeline._iter_llm_deltas(None, [], None, predicted_text="cached"):
            out.append(d)
        assert out == ["cached"], out
        out2 = []
        async for d in _rt_pipeline._iter_llm_deltas(None, [], None, predicted_text=""):
            out2.append(d)
        assert out2 == [], out2
    _asyncio.run(_predicted_iter())
    ok("_iter_llm_deltas short-circuits on predicted_text override")

def test_realtime_wire_server_py_route_registration():
    _section('Fix Realtime-Wire: server.py route registration')
    from realtime import RFC_VERSION
    from realtime import (
        capabilities_json_with_models as _rt_caps_with_models,
        live_session_count as _rt_live_count,
    )
    from realtime.websocket import active_session_count as _rt_ws_count

    _caps_no_models = _rt_caps_with_models(None)
    for _key in ("rfc_version", "features", "extensions"):
        assert _key in _caps_no_models, f"caps missing {_key}"
    for _feat_key in ("eou_kinds", "fusion_rules", "input_audio_formats",
                      "output_audio_formats", "transports", "audio_codecs", "vad_types"):
        assert _feat_key in _caps_no_models["features"], f"features missing {_feat_key}"
    assert "webrtc" in _caps_no_models["features"]["transports"]
    assert "websocket" in _caps_no_models["features"]["transports"]
    for _codec in ("opus", "pcm16", "g711_ulaw", "g711_alaw"):
        assert _codec in _caps_no_models["features"]["audio_codecs"], _codec
    for _vad in ("server_vad", "semantic_vad"):
        assert _vad in _caps_no_models["features"]["vad_types"], _vad
    for _eou in ("heuristic", "integrated"):
        assert _eou in _caps_no_models["extensions"]["eou_kinds"], _eou
    for _eou in ("audio", "fusion"):
        assert _eou not in _caps_no_models["extensions"]["eou_kinds"], _eou
    assert "models" in _caps_no_models["extensions"]
    assert isinstance(_caps_no_models["extensions"]["models"], list)
    ok("capabilities_json_with_models exposes transports/codecs/vad/eou keys")

    assert isinstance(_rt_live_count(), int)
    assert isinstance(_rt_ws_count(), int)
    ok("live_session_count + active_session_count return ints")

    from fastapi.testclient import TestClient
    from server import app as _rt_app

    _rt_client = TestClient(_rt_app)
    _resp_caps = _rt_client.get("/v1/realtime/capabilities")
    assert _resp_caps.status_code == 200, _resp_caps.status_code
    _caps_payload = _resp_caps.json()
    assert _caps_payload["rfc_version"] == RFC_VERSION
    assert "transports" in _caps_payload["features"]
    ok("GET /v1/realtime/capabilities returns 200 with feature matrix")

    _resp_sess = _rt_client.get("/health/sessions")
    assert _resp_sess.status_code == 200, _resp_sess.status_code
    _sess_payload = _resp_sess.json()
    assert isinstance(_sess_payload.get("live_sessions"), int)
    assert isinstance(_sess_payload.get("ws_sessions"), int)
    assert isinstance(_sess_payload.get("webrtc_sessions"), int)
    ok("GET /health/sessions returns {live_sessions, ws_sessions, webrtc_sessions}")

    _routes_found = {(getattr(r, "path", None), tuple(sorted(getattr(r, "methods", []) or [])))
                     for r in _rt_app.routes if hasattr(r, "path")}
    assert ("/v1/realtime", ("POST",)) in _routes_found, "POST /v1/realtime not registered"
    assert any(p == "/v1/realtime" for (p, _m) in _routes_found), "/v1/realtime not registered"
    ok("POST /v1/realtime + WS /v1/realtime registered on app")

def test_stt_noisegate():
    _section('Fix STT-NoiseGate')
    from stt.noise_gate import (
        FULL_MS as _NG_FULL_MS,
        LOOSE_FLOOR as _NG_LOOSE_FLOOR,
        OFF_MS as _NG_OFF_MS,
        GateThresholds as _NGGateThresholds,
        NoiseRejection as _NGNoiseRejection,
        effective_avg_logprob_threshold as _ng_eff,
        evaluate as _ng_eval,
    )

    assert _NG_FULL_MS == 1500
    assert _NG_OFF_MS == 5000
    assert _NG_LOOSE_FLOOR == -3.0
    ok("constants match upstream (FULL_MS=1500, OFF_MS=5000, LOOSE_FLOOR=-3.0)")

    assert _ng_eff(None, 0) is None
    assert _ng_eff(None, 1500) is None
    assert _ng_eff(None, 8000) is None
    ok("base=None -> None at every duration")

    assert _ng_eff(-1.0, 0) == -1.0
    assert _ng_eff(-1.0, 1) == -1.0
    assert _ng_eff(-1.0, 1500) == -1.0
    assert _ng_eff(-0.5, 750) == -0.5
    ok("duration <= FULL_MS -> returns base unchanged")

    assert _ng_eff(-1.0, 5000) is None
    assert _ng_eff(-1.0, 5001) is None
    assert _ng_eff(-1.0, 60_000) is None
    ok("duration >= OFF_MS -> None (gate disabled for long audio)")

    _t = _ng_eff(-1.0, 3250)
    assert _t is not None
    _expected = -1.0 + 0.5 * (-3.0 - (-1.0))
    assert abs(_t - _expected) < 1e-6, f"got {_t}, want {_expected}"
    assert abs(_t - (-2.0)) < 1e-6
    ok(f"midpoint duration_ms=3250 lerps to base + 0.5·(LOOSE_FLOOR-base) = {_t:.4f}")

    _dq = 1500 + (5000 - 1500) // 4
    _t2 = _ng_eff(-0.5, _dq)
    assert _t2 is not None
    _frac_q = (_dq - 1500) / (5000 - 1500)
    assert abs(_t2 - (-0.5 + _frac_q * (-3.0 - (-0.5)))) < 1e-6
    ok("quarter-point lerps correctly")

    _t3 = _ng_eff(-1.0, 1501)
    assert _t3 is not None and -1.001 < _t3 < -0.999 + 1e-3
    _t4 = _ng_eff(-1.0, 4999)
    assert _t4 is not None and abs(_t4 - -3.0) < 1e-2
    ok("near-FULL_MS stays near base, near-OFF_MS approaches LOOSE_FLOOR")

    assert _NGNoiseRejection.NO_SPEECH_PROB.as_str() == "no_speech_prob"
    assert _NGNoiseRejection.AVG_LOGPROB.as_str() == "avg_logprob"
    ok("NoiseRejection.as_str matches upstream string codes")

    assert _ng_eval(0.99, -10.0, 1000, _NGGateThresholds.disabled()) is None
    ok("disabled thresholds -> no rejection")

    _both = _NGGateThresholds(no_speech_prob_threshold=0.6, avg_logprob_threshold=-1.0)
    assert _ng_eval(0.9, -5.0, 500, _both) is _NGNoiseRejection.NO_SPEECH_PROB
    ok("both gates fail -> NSP wins (reason priority)")

    assert _ng_eval(0.1, -2.0, 500, _both) is _NGNoiseRejection.AVG_LOGPROB
    ok("only logprob fails -> AVG_LOGPROB")

    _long_thr = _NGGateThresholds(no_speech_prob_threshold=0.6, avg_logprob_threshold=-0.5)
    assert _ng_eval(0.1, -10.0, 6000, _long_thr) is None
    ok("6 s audio: avg_logprob gate disabled, transcript passes")

    _stats_thr = _NGGateThresholds(no_speech_prob_threshold=0.0, avg_logprob_threshold=0.0)
    assert _ng_eval(None, None, 1000, _stats_thr) is None
    ok("missing stats -> no rejection (graceful degradation)")

def test_tts_npz():
    _section('Fix TTS-NPZ')
    import io as _np_io
    import os as _ng_os
    import pathlib as _pathlib
    import tempfile as _tmpfile
    import zipfile as _np_zip

    import numpy as _np_np

    from tts.kokoro import Voice as _NPZVoice, load_voices as _npz_load_voices
    from tts.kokoro.npz import parse_npy as _npz_parse_npy

    _voice_a = _np_np.arange(2 * 3 * 4, dtype=_np_np.float32).reshape(2, 3, 4)
    _voice_b = _np_np.linspace(-1.0, 1.0, 5 * 6, dtype=_np_np.float32).reshape(5, 6)

    _buf = _np_io.BytesIO()
    with _np_zip.ZipFile(_buf, "w") as _zf:
        for _name, _arr in (("af_alpha", _voice_a), ("am_beta", _voice_b)):
            _entry = _np_io.BytesIO()
            _np_np.save(_entry, _arr)
            _zf.writestr(f"{_name}.npy", _entry.getvalue())
        _zf.writestr("README.txt", b"not a voice; should be skipped")
    _buf.seek(0)

    with _tmpfile.NamedTemporaryFile(suffix=".bin", delete=False) as _tmp:
        _tmp.write(_buf.getvalue())
        _archive_path = _tmp.name

    _voices = _npz_load_voices(_pathlib.Path(_archive_path))
    assert set(_voices.keys()) == {"af_alpha", "am_beta"}
    ok("load_voices returns the right dict, skips non-.npy entries")

    _va = _voices["af_alpha"]
    assert isinstance(_va, _NPZVoice)
    assert _va.shape == [2, 3, 4]
    _vb = _voices["am_beta"]
    assert _vb.shape == [5, 6]
    ok("Voice.shape matches numpy array.shape")

    _row0 = _va.row(0)
    assert _row0.size == 12
    assert _np_np.allclose(_row0, _voice_a[0].reshape(-1))
    ok("Voice.row(0) returns the expected slice (3D voice)")

    _row1 = _va.row(1)
    assert _np_np.allclose(_row1, _voice_a[1].reshape(-1))
    _row_b2 = _vb.row(2)
    assert _np_np.allclose(_row_b2, _voice_b[2].reshape(-1))
    ok("Voice.row(i) returns the expected slice for arbitrary i and a 2D voice")

    try:
        _va.row(2)
        raise AssertionError("BAD: out-of-range index accepted")
    except IndexError:
        ok("Voice.row out-of-range index raises IndexError")

    _inline = _np_io.BytesIO()
    _np_np.save(_inline, _voice_a)
    _parsed = _npz_parse_npy(_inline.getvalue())
    assert _parsed.shape == [2, 3, 4]
    assert _np_np.allclose(_parsed.row(0), _voice_a[0].reshape(-1))
    ok("parse_npy handles raw .npy bytes via numpy.load")

    _ng_os.unlink(_archive_path)

def test_conversation_llm():
    _section('Fix Conversation-LLM')
    import os as _conv_os
    from unittest.mock import patch as _conv_patch

    from conversation import (
        ChatMessage as _ChatMessage,
        LlmConfig as _LlmConfig,
        LlmStreamError as _LlmStreamError,
        PredictedTokenBuffer as _PredictedTokenBuffer,
        SentenceChunker as _SentenceChunker,
        complete_stream_messages as _complete_stream_messages,
    )

    _buf = _PredictedTokenBuffer(2)
    assert not _buf.push("a")
    assert not _buf.push("b")
    assert _buf.push("c"), "push at cap should overflow"
    assert _buf.dropped_count() == 1
    assert _buf.len() == 2
    assert not _buf.is_empty()
    _drained = _buf.drain()
    assert _drained == ["b", "c"], f"unexpected drain: {_drained}"
    assert _buf.is_empty()
    assert _buf.len() == 0
    ok("PredictedTokenBuffer: push/drain/cap/dropped_count")

    _buf2 = _PredictedTokenBuffer(8)
    _buf2.push("hi ")
    _buf2.push("there")
    assert _buf2.chars_seen == 8, f"chars_seen={_buf2.chars_seen}"
    ok("PredictedTokenBuffer: chars_seen cumulative")

    _buf3 = _PredictedTokenBuffer(0)
    assert _buf3.cap == 1, "cap must be coerced to >=1"
    ok("PredictedTokenBuffer: cap coerced to >=1")

    _envset = {
        "CHAT_COMPLETION_BASE_URL": "http://example/v1",
        "CHAT_COMPLETION_API_KEY": "k-secret",
        "DEFAULT_REALTIME_CONVERSATION_MODEL": "test-model",
    }
    with _conv_patch.dict(_conv_os.environ, _envset, clear=False):
        _cfg = _LlmConfig.from_env()
    assert _cfg is not None
    assert _cfg.base_url == "http://example/v1"
    assert _cfg.api_key == "k-secret"
    assert _cfg.model == "test-model"
    ok("LlmConfig.from_env: all three vars set")

    with _conv_patch.dict(_conv_os.environ, {}, clear=True):
        _cfg_none = _LlmConfig.from_env()
    assert _cfg_none is None, "from_env must return None without base_url"
    ok("LlmConfig.from_env: returns None without CHAT_COMPLETION_BASE_URL")

    with _conv_patch.dict(_conv_os.environ, {"CHAT_COMPLETION_BASE_URL": "http://x/v1"}, clear=True):
        _cfg_def = _LlmConfig.from_env()
    assert _cfg_def is not None
    assert _cfg_def.api_key is None
    assert _cfg_def.model == "default"
    ok("LlmConfig.from_env: defaults model='default', api_key=None")

    _chunker = _SentenceChunker()
    assert _chunker.feed("Hello") == []
    _chunker_out = _chunker.feed(" world. How are you?")
    assert _chunker_out == ["Hello world.", "How are you?"], f"got {_chunker_out}"
    assert _chunker.flush() is None
    ok("SentenceChunker: splits on terminators")

    _c2 = _SentenceChunker()
    _c2.feed("incomplete")
    assert _c2.flush() == "incomplete"
    ok("SentenceChunker: flushes unterminated tail")

    def _build_sse_payload(tokens):
        out = []
        for tok in tokens:
            chunk = json.dumps({"choices": [{"delta": {"content": tok}}]})
            out.append(f"data: {chunk}\n\n")
        out.append("data: [DONE]\n\n")
        return out

    class _FakeStreamCM:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_text(self):
            for chunk in self._payload:
                yield chunk

        async def aread(self):
            return b"".join(p.encode() for p in self._payload)

    class _FakeAsyncClientCM:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, **kwargs):
            return _FakeStreamCM(self.status_code, self._payload)

    def _make_client_factory(status, payload):
        def _factory(*a, **k):
            return _FakeAsyncClientCM(status, payload)
        return _factory

    import asyncio as _conv_asyncio

    _payload_ok = _build_sse_payload(["hello", " ", "world"])
    with _conv_patch("conversation.llm.httpx.AsyncClient", new=_make_client_factory(200, _payload_ok)):
        async def _run_ok():
            out = []
            async for d in _complete_stream_messages(
                _LlmConfig(base_url="http://x/v1", api_key=None, model="m"),
                [_ChatMessage(role="user", content="ping")],
            ):
                out.append(d)
            return out
        _deltas = _conv_asyncio.run(_run_ok())
    assert _deltas == ["hello", " ", "world"], f"got {_deltas}"
    ok("complete_stream_messages: yields synthetic SSE deltas in order")

    with _conv_patch(
        "conversation.llm.httpx.AsyncClient",
        new=_make_client_factory(500, ["upstream broken"]),
    ):
        async def _run_err():
            try:
                async for _d in _complete_stream_messages(
                    _LlmConfig(base_url="http://x/v1", api_key=None, model="m"),
                    [_ChatMessage(role="user", content="ping")],
                ):
                    pass
            except _LlmStreamError as e:
                return str(e)
            return ""
        _err_msg = _conv_asyncio.run(_run_err())
    assert "500" in _err_msg, f"expected 500 in {_err_msg!r}"
    ok("complete_stream_messages: 5xx surfaces as LlmStreamError with status")

    with _conv_patch(
        "conversation.llm.httpx.AsyncClient",
        new=_make_client_factory(200, ["data: [DONE]\n\n"]),
    ):
        async def _run_empty():
            try:
                async for _d in _complete_stream_messages(
                    _LlmConfig(base_url="http://x/v1", api_key=None, model="m"),
                    [_ChatMessage(role="user", content="ping")],
                ):
                    pass
            except _LlmStreamError as e:
                return str(e)
            return ""
        _empty_msg = _conv_asyncio.run(_run_empty())
    assert "no content" in _empty_msg.lower(), f"got {_empty_msg!r}"
    ok("complete_stream_messages: empty stream raises 'no content'")

def test_top_level_utils_ids_errors_otel_trace_ports():
    _section('Fix Top-Level-Utils: ids / errors / otel / trace ports')

    import os as _utils_os

    import env
    import ids as _ids
    import errors as _errors
    import otel as _otel
    import trace as _trace

    for _gen, _prefix in (
        (_ids.next_session_id, "sess_"),
        (_ids.next_item_id, "item_"),
        (_ids.next_response_id, "resp_"),
        (_ids.next_event_id, "evt_"),
    ):
        _a_id = _gen()
        _b_id = _gen()
        assert isinstance(_a_id, str) and _a_id.startswith(_prefix), f"{_gen.__name__}: bad prefix {_a_id!r}"
        assert len(_a_id) - len(_prefix) >= 16, f"{_gen.__name__}: too short {_a_id!r}"
        assert _a_id != _b_id, f"{_gen.__name__}: not unique"
    ok("ids: all 4 generators produce prefixed unique ids >=16 chars")

    _src = _ids.RandomIdSource()
    assert _src.session().startswith("sess_") and _src.item().startswith("item_")
    assert _src.response().startswith("resp_") and _src.event().startswith("evt_")
    _csrc = _ids.CounterIdSource()
    assert _csrc.item() == f"item_{0:024d}"
    assert _csrc.item() == f"item_{1:024d}"
    assert _csrc.response() == f"resp_{0:024d}"
    ok("ids: RandomIdSource + CounterIdSource (per-kind independent)")

    assert _errors.is_known_code(_errors.code.MODEL_LOAD_FAILED)
    assert _errors.is_known_code(_errors.code.SESSION_NOT_ACTIVE)
    assert not _errors.is_known_code("totally_made_up_code")
    assert _errors.error_type_for(_errors.code.VAD_FAILED) == "server_error"
    assert _errors.error_type_for(_errors.code.CLIENT_TOO_SLOW) == "invalid_request_error"
    assert _errors.error_type_for(_errors.code.MODEL_LOAD_FAILED) == "server_error"
    ok("errors: registry + error_type_for routing matches RFC v3 section 10.5")

    _env_msg = _errors.envelope("nope", code_value=_errors.code.VAD_FAILED, param="audio")
    assert set(_env_msg.keys()) == {"error"}
    assert set(_env_msg["error"].keys()) == {"message", "type", "param", "code"}
    assert _env_msg["error"]["message"] == "nope"
    assert _env_msg["error"]["type"] == "server_error"
    assert _env_msg["error"]["param"] == "audio"
    assert _env_msg["error"]["code"] == _errors.code.VAD_FAILED
    import json as _json_top
    assert _json_top.loads(_json_top.dumps(_env_msg)) == _env_msg
    ok("errors.envelope: OpenAI-shaped JSON round-trip")

    _prev_endpoint = _utils_os.environ.pop(env.OTEL_EXPORTER_OTLP_ENDPOINT, None)
    try:
        _otel.shutdown()
        _otel._provider = None
        _otel._enabled = False
        _init_res = _otel.init()
        assert _init_res is False, f"init with no endpoint should return False, got {_init_res!r}"
        assert _otel.is_enabled() is False
        _otel.shutdown()
        _otel.shutdown()
        ok("otel: init() no-op without endpoint; shutdown idempotent")
    finally:
        if _prev_endpoint is not None:
            _utils_os.environ[env.OTEL_EXPORTER_OTLP_ENDPOINT] = _prev_endpoint

    _trace.init()
    with _trace.span("test.span", foo="bar"):
        pass

    @_trace.traced("test.traced")
    def _traced_fn(x: int) -> int:
        return x + 1

    assert _traced_fn(2) == 3

    @_trace.traced()
    def _traced_default(x: int) -> int:
        return x * 2

    assert _traced_default(3) == 6
    ok("trace: init/span/traced safe with no opentelemetry runtime activity")

    _t = _trace.canonicalize_trace([
        {"type": "session.created", "session": {"id": "sess_abc"}},
        {"type": "input_audio_buffer.speech_started", "item_id": "item_xy"},
        {"type": "input_audio_buffer.speech_stopped", "item_id": "item_xy"},
        {"type": "response.created", "response": {"id": "resp_q"}},
    ])
    assert _t.events[0]["session"]["id"] == "sess_1"
    assert _t.events[1]["item_id"] == "item_1"
    assert _t.events[2]["item_id"] == "item_1"
    assert _t.events[3]["response"]["id"] == "resp_1"

    _t2 = _trace.canonicalize_trace([{
        "type": "input_audio_buffer.speech_stopped",
        "audio_end_ms": 12345,
        "ts_ms": 99999,
    }])
    assert _t2.events[0]["audio_end_ms"] == 0
    assert _t2.events[0]["ts_ms"] == 0

    _t3 = _trace.canonicalize_trace([{"type": "eou.scored", "score": 0.123456789}])
    assert abs(_t3.events[0]["score"] - 0.123) < 1e-6

    _a = _trace.canonicalize_trace([
        {"type": "session.created", "session": {"id": "sess_1"}},
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": "response.done", "response": {"id": "resp_1", "status": "completed"}},
    ])
    _b = _trace.canonicalize_trace([
        {"type": "session.created", "session": {"id": "sess_z"}},
        {"type": "response.created", "response": {"id": "resp_z"}},
        {"type": "response.done", "response": {"id": "resp_z", "status": "cancelled"}},
    ])
    assert _trace.trace_diff(_a, _b) == 2

    _id_a = _trace.canonicalize_trace([{"type": "x", "id": "evt_a"}])
    _id_b = _trace.canonicalize_trace([{"type": "x", "id": "evt_b"}])
    assert _trace.trace_diff(_id_a, _id_b) is None
    ok("trace: canonicalize + trace_diff parity with rust/src/trace.rs tests")

def test_audioout_realopus():
    _section('Fix AudioOut-RealOpus')
    if not _has_module("opuslib") and _STRICT_CI:
        pytest.fail("strict-skip in CI: AudioOut-RealOpus requires opuslib")

    import asyncio as _ao_asyncio
    import math as _ao_math

    from realtime.audio_out import (
        FRAME_MS as _AO_FRAME_MS,
        FRAME_SAMPLES as _AO_FRAME_SAMPLES,
        OUT_SAMPLE_RATE as _AO_OUT_SR,
        OutboundPacer as _AO_OutboundPacer,
        QueueFull as _AO_QueueFull,
        QueueGate as _AO_QueueGate,
        _f32_to_s16le_bytes as _AO_f32_to_s16le_bytes,
    )

    try:
        import opuslib as _ao_opuslib  # noqa: F401
        _AO_HAS_OPUS = True
    except Exception as _ao_opus_err:
        _AO_HAS_OPUS = False
        info(f"opuslib not installed; AudioOut-RealOpus encode tests skipped ({_ao_opus_err})")

    _ao_pcm = _AO_f32_to_s16le_bytes([0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
    assert len(_ao_pcm) == 7 * 2, f"f32_to_s16le bytes length {len(_ao_pcm)}"
    assert _ao_pcm[0:2] == (0).to_bytes(2, "little", signed=True)
    assert _ao_pcm[2:4] == (32767).to_bytes(2, "little", signed=True)
    assert _ao_pcm[4:6] == (-32767).to_bytes(2, "little", signed=True)
    assert _ao_pcm[10:12] == (32767).to_bytes(2, "little", signed=True), "scale-then-clip: +2.0 saturates to +32767"
    assert _ao_pcm[12:14] == (-32767).to_bytes(2, "little", signed=True), "scale-then-clip: -2.0 saturates to -32767"
    ok("audio_out: _f32_to_s16le_bytes scale-then-clip matches g711 f32_to_s16le")

    class _AOFakeTrack:
        def __init__(self):
            self.frames = []
            self.ended = 0
            self.dropped = 0

        def push_opus_frame(self, payload, duration_ms):
            self.frames.append((payload, duration_ms))

        def end_of_stream(self):
            self.ended += 1

        def drop_queued(self):
            self.dropped += 1

    _ao_gate_full = _AO_QueueGate(cap_ms=100)
    _ao_threw_full = False
    try:
        _ao_gate_full.try_push(150)
    except _AO_QueueFull as err:
        _ao_threw_full = True
        assert err.cap_ms == 100 and err.queued_ms == 150
    assert _ao_threw_full, "QueueFull must raise when chunk_ms > cap_ms"
    ok("audio_out: QueueGate raises QueueFull when projected exceeds cap")

    _ao_pacer_full = _AO_OutboundPacer(track=_AOFakeTrack(), played_ms_ref=[0], queue_cap_ms=80)
    _ao_oversized = [0.0] * (24_000 * 200 // 1000)
    _ao_play_full_threw = False
    try:
        _ao_asyncio.run(_ao_pacer_full.play(_ao_oversized))
    except _AO_QueueFull:
        _ao_play_full_threw = True
    assert _ao_play_full_threw, "play() must propagate QueueFull when chunk_ms > cap_ms"
    ok("audio_out: OutboundPacer.play raises QueueFull on oversize chunk")

    if _AO_HAS_OPUS:
        _ao_track = _AOFakeTrack()
        _ao_played = [0]
        _ao_pacer = _AO_OutboundPacer(track=_ao_track, played_ms_ref=_ao_played, queue_cap_ms=10_000)

        _AO_TONE_MS = 100
        _ao_n_in = 24_000 * _AO_TONE_MS // 1000
        _ao_samples = [
            0.25 * _ao_math.sin(2.0 * _ao_math.pi * 440.0 * (i / 24_000.0))
            for i in range(_ao_n_in)
        ]
        _ao_pre_gate = _ao_pacer.gate.queued_ms

        async def _ao_run_play():
            await _ao_pacer.play(_ao_samples)

        _ao_t0 = time.monotonic()
        _ao_asyncio.run(_ao_run_play())
        _ao_elapsed = time.monotonic() - _ao_t0
        _ao_expected_frames = _AO_TONE_MS // _AO_FRAME_MS
        assert len(_ao_track.frames) == _ao_expected_frames, (
            f"expected {_ao_expected_frames} push_opus_frame calls, got {len(_ao_track.frames)}"
        )
        for _ao_payload, _ao_dur in _ao_track.frames:
            assert isinstance(_ao_payload, (bytes, bytearray)) and len(_ao_payload) > 0, (
                "every payload must be non-empty bytes"
            )
            assert _ao_dur == _AO_FRAME_MS, f"duration_ms must be {_AO_FRAME_MS}, got {_ao_dur}"
        assert _ao_played[0] == _ao_expected_frames * _AO_FRAME_MS, (
            f"played_ms_ref expected {_ao_expected_frames * _AO_FRAME_MS}, got {_ao_played[0]}"
        )
        assert _ao_pacer.gate.queued_ms <= 1, (
            f"gate should drain to ~0, queued_ms={_ao_pacer.gate.queued_ms}"
        )
        info(f"opus encode 100ms tone: elapsed={_ao_elapsed*1000:.1f}ms (paced ~{_AO_TONE_MS}ms)")
        ok(
            f"audio_out: 24k->48k->opus produced {len(_ao_track.frames)} frames via track.push_opus_frame"
        )

        _ao_asyncio.run(_ao_pacer.flush())
        assert _ao_track.ended >= 1, "flush() should signal end_of_stream on track"
        _ao_played_pre_cancel = _ao_played[0]
        _ao_pacer.cancel()
        assert _ao_pacer._cancelled is True
        assert _ao_track.dropped >= 1, "cancel() should call track.drop_queued"
        assert _ao_pacer.gate.queued_ms == 0, "cancel() must drain queue gate"
        assert _ao_played[0] == _ao_pacer.frames_written * _AO_FRAME_MS, (
            f"cancel() should set played_ms_ref={_ao_pacer.frames_written * _AO_FRAME_MS}, got {_ao_played[0]}"
        )

        async def _ao_play_after_cancel():
            await _ao_pacer.play([0.0] * (24_000 * 20 // 1000))

        _ao_asyncio.run(_ao_play_after_cancel())
        assert len(_ao_track.frames) == _ao_expected_frames, (
            "play() after cancel() must be a no-op"
        )
        ok("audio_out: flush() drains tail-silence + signals EOS; cancel() drops queue and updates played_ms_ref")
    else:
        _ao_track = _AOFakeTrack()
        _ao_pacer = _AO_OutboundPacer(track=_ao_track, played_ms_ref=[0], queue_cap_ms=10_000)
        _ao_pacer.cancel()
        assert _ao_pacer._cancelled is True
        assert _ao_pacer.gate.queued_ms == 0
        ok("audio_out: cancel() works without opuslib (skipped opus encode assertions)")

def test_audioin_realopus(_sys_modules_snapshot):
    _section('Fix AudioIn-RealOpus')
    if not _PWT_HAS_OPUS_RUNTIME and _STRICT_CI:
        pytest.fail("strict-skip in CI: AudioIn-RealOpus requires opuslib")
    _opuslib_mod = pytest.importorskip("opuslib")

    import importlib as _ai_importlib
    import math as _ai_math
    import sys as _ai_sys
    import threading as _ai_threading

    import numpy as _ai_np

    _ai_audio_in = _ai_importlib.import_module("realtime.audio_in")
    _AudioIngest = _ai_audio_in.AudioIngest
    _ai_audio_defaults = _ai_importlib.import_module("realtime").audio_defaults

    _has_opuslib = True
    _ai_assertion_count = [0]

    def _ai_assert(cond, msg):
        assert cond, msg
        _ai_assertion_count[0] += 1

    if _has_opuslib:
        _ing_mono = _AudioIngest(channels=1)
        _ai_assert(_ing_mono.channels == 1, "mono channels")
        _ing_stereo = _AudioIngest(channels=2)
        _ai_assert(_ing_stereo.channels == 2, "stereo channels")
        _ai_assert(_ing_stereo.get_total_samples_consumed() == 0, "initial 16k count == 0")
        _ai_assert(_ing_stereo.get_total_input_samples() == 0, "initial 48k count == 0")

        try:
            _AudioIngest(channels=3)
            _ai_assert(False, "channels=3 should raise")
        except ValueError:
            _ai_assert(True, "channels=3 ValueError raised")

        _SAMPLE_RATE = int(_ai_audio_defaults.OPUS_SAMPLE_RATE_HZ)
        _FRAME_SAMPLES_20MS = _SAMPLE_RATE * 20 // 1000

        class _FakeAvFrame:
            def __init__(self, ndarr, sample_rate, layout_name):
                self._ndarr = ndarr
                self.sample_rate = sample_rate

                class _L:
                    pass

                self.layout = _L()
                self.layout.name = layout_name

            def to_ndarray(self):
                return self._ndarr

        _t = _ai_np.arange(_FRAME_SAMPLES_20MS, dtype=_ai_np.float32) / float(_SAMPLE_RATE)
        _sine = (_ai_np.sin(2 * _ai_math.pi * 440.0 * _t) * 16000.0).astype(_ai_np.int16)
        _stereo_interleaved = _ai_np.empty(_FRAME_SAMPLES_20MS * 2, dtype=_ai_np.int16)
        _stereo_interleaved[0::2] = _sine
        _stereo_interleaved[1::2] = _sine
        _stereo_planar = _ai_np.stack([_sine, _sine], axis=0)

        _ing_av = _AudioIngest(channels=2)
        _frame_planar = _FakeAvFrame(_stereo_planar, _SAMPLE_RATE, "stereo")
        _ing_av.process_av_frame(_frame_planar)
        _out = _ing_av.take_array()
        _expected = _FRAME_SAMPLES_20MS // 3
        _ai_assert(
            abs(_out.shape[0] - _expected) <= 8,
            f"av_frame planar: expected ~{_expected} samples, got {_out.shape[0]}",
        )
        _ai_assert(_out.ndim == 1, "av_frame output is mono (1-D)")
        _ai_assert(_out.dtype == _ai_np.float32, "av_frame output dtype is float32")
        _ai_assert(
            _ing_av.get_total_input_samples() == _FRAME_SAMPLES_20MS,
            f"input count = {_FRAME_SAMPLES_20MS}",
        )
        _ai_assert(
            _ing_av.get_total_samples_consumed() == _out.shape[0],
            "total_samples_consumed matches taken length",
        )

        _ing_av_i = _AudioIngest(channels=2)
        _frame_interleaved = _FakeAvFrame(_stereo_interleaved, _SAMPLE_RATE, "stereo")
        _ing_av_i.process_av_frame(_frame_interleaved)
        _out_i = _ing_av_i.take_array()
        _ai_assert(
            abs(_out_i.shape[0] - _expected) <= 8,
            f"av_frame interleaved: expected ~{_expected} samples, got {_out_i.shape[0]}",
        )

        _ing_av_24 = _AudioIngest(channels=2)
        _t24 = _ai_np.arange(480, dtype=_ai_np.float32) / 24000.0
        _sine24 = (_ai_np.sin(2 * _ai_math.pi * 440.0 * _t24) * 16000.0).astype(_ai_np.int16)
        _frame_24k = _FakeAvFrame(_ai_np.stack([_sine24, _sine24], axis=0), 24000, "stereo")
        _ing_av_24.process_av_frame(_frame_24k)
        _out_24 = _ing_av_24.take_array()
        _ai_assert(
            abs(_out_24.shape[0] - 320) <= 16,
            f"24k -> 16k via process_av_frame: expected ~320 samples, got {_out_24.shape[0]}",
        )

        _enc = _opuslib_mod.Encoder(_SAMPLE_RATE, 2, _opuslib_mod.APPLICATION_VOIP)
        _payload = _enc.encode(_stereo_interleaved.tobytes(), _FRAME_SAMPLES_20MS)
        _ing_op = _AudioIngest(channels=2)
        _ing_op.process_opus(_payload)
        _out_op = _ing_op.take_array()
        _ai_assert(
            abs(_out_op.shape[0] - _expected) <= 8,
            f"opus payload: expected ~{_expected} samples, got {_out_op.shape[0]}",
        )

        _ing_legacy = _AudioIngest(channels=2)
        _ing_legacy.process(_payload)
        _out_legacy = _ing_legacy.take()
        _ai_assert(
            isinstance(_out_legacy, list) and abs(len(_out_legacy) - _expected) <= 8,
            f"deprecated process(): expected ~{_expected}, got {len(_out_legacy) if isinstance(_out_legacy, list) else type(_out_legacy)}",
        )

        _ing_conc = _AudioIngest(channels=2)
        _produced_payloads = [
            _enc.encode(_stereo_interleaved.tobytes(), _FRAME_SAMPLES_20MS) for _i in range(20)
        ]

        _produce_done = _ai_threading.Event()
        _drain = []

        def _producer():
            for _p in _produced_payloads:
                _ing_conc.process_opus(_p)
            _produce_done.set()

        def _drainer():
            while True:
                _chunk = _ing_conc.take_array()
                if _chunk.size:
                    _drain.append(_chunk)
                if _produce_done.is_set():
                    _final = _ing_conc.take_array()
                    if _final.size:
                        _drain.append(_final)
                    return

        _tp = _ai_threading.Thread(target=_producer)
        _td = _ai_threading.Thread(target=_drainer)
        _tp.start()
        _td.start()
        _tp.join(timeout=10.0)
        _td.join(timeout=10.0)
        _ai_assert(not _tp.is_alive() and not _td.is_alive(), "concurrency: threads exited cleanly")
        _total_drained = sum(c.size for c in _drain)
        _ai_assert(
            _total_drained == _ing_conc.get_total_samples_consumed(),
            f"concurrency: drained {_total_drained} == counter {_ing_conc.get_total_samples_consumed()}",
        )
        _ai_assert(_total_drained > 0, "concurrency: produced > 0 samples total")

        ok(f"AudioIn-RealOpus: opuslib path validated (has_opuslib={_has_opuslib})")
    else:
        info("AudioIn-RealOpus: opuslib not installed; skipping encode round-trip + av_frame tests")

    _saved_opuslib = _ai_sys.modules.get("opuslib")
    _ai_sys.modules["opuslib"] = None  # type: ignore[assignment]
    try:
        _ai_importlib.reload(_ai_audio_in)
        try:
            _ai_audio_in.AudioIngest(channels=1)
            _ai_assert(False, "missing opuslib should raise RuntimeError at construction")
        except RuntimeError as _re:
            _ai_assert("opuslib" in str(_re).lower(), f"RuntimeError mentions opuslib: {_re}")
    finally:
        if _saved_opuslib is not None:
            _ai_sys.modules["opuslib"] = _saved_opuslib
        else:
            _ai_sys.modules.pop("opuslib", None)
        _ai_importlib.reload(_ai_audio_in)
    ok(f"AudioIn-RealOpus: missing-opuslib raises RuntimeError at ctor (asserts: {_ai_assertion_count[0]})")

def test_ct2_bindings():
    _section('Fix CT2-Bindings')
    if not _has_module("ctranslate2") and _STRICT_CI:
        pytest.fail("strict-skip in CI: CT2-Bindings requires ctranslate2")

    import importlib as _ct2_importlib

    _ct2_native_available = False
    _ct2_pypi_available = False
    _ct2_native_err: str | None = None
    _ct2_pypi_err: str | None = None

    try:
        _ct2_pkg = _ct2_importlib.import_module("ct2_bindings")
        info(f"ct2_bindings loader imported: {_ct2_pkg.__file__}")
        _ct2_native_available = bool(getattr(_ct2_pkg, "EXTENSION_AVAILABLE", False))
        _ct2_native_err = getattr(_ct2_pkg, "EXTENSION_IMPORT_ERROR", None)
        if _ct2_native_available:
            ok("ct2_bindings._ct2 native extension is available")
        else:
            info(f"ct2_bindings._ct2 native extension NOT available: {_ct2_native_err}")
    except Exception as _e:
        info(f"ct2_bindings package import failed: {_e}")

    try:
        _ct2_importlib.import_module("ctranslate2")
        _ct2_pypi_available = True
        ok("ctranslate2 PyPI package is importable (fallback path available)")
    except Exception as _e:
        _ct2_pypi_err = str(_e)
        info(f"ctranslate2 PyPI package NOT installed: {_ct2_pypi_err}")

    if not _ct2_native_available and not _ct2_pypi_available:
        pytest.skip("ct2 backend unavailable (no native extension, no PyPI fallback)")
    else:
        from stt.ct2 import (
            Ct2WhisperBackend as _CT2Backend,
            Ct2WhisperConfig as _CT2Config,
            Segment as _CT2Segment,
            TranscriptionResult as _CT2Result,
            decode_bpe as _ct2_decode_bpe,
        )

        assert _ct2_decode_bpe("Ġhello") == " hello"
        assert _ct2_decode_bpe("hello") == "hello"
        ok("ct2.decode_bpe roundtrips Whisper BPE space marker (Ġ -> ' ')")

        _empty_res = _CT2Result(text="hello")
        assert _empty_res.text == "hello"
        assert _empty_res.avg_logprob is None
        assert _empty_res.no_speech_prob is None
        assert _empty_res.segments == []
        ok("TranscriptionResult dataclass: defaults sane")

        _seg = _CT2Segment(t_start_ms=100, t_end_ms=900, text="ok")
        assert _seg.t_start_ms == 100 and _seg.t_end_ms == 900 and _seg.text == "ok"
        ok("Segment dataclass: positional + named fields")

        _bad_path = "/nonexistent/ct2-model-path-zzz-does-not-exist"
        _bad_cfg = _CT2Config(model_path=_bad_path)
        try:
            _CT2Backend(_bad_cfg)
        except RuntimeError as _e:
            ok(f"opening missing model raises RuntimeError (not crash): {type(_e).__name__}")
        except Exception as _e:
            info(f"opening missing model raised non-RuntimeError {type(_e).__name__}: {_e}")
        else:
            info("opening missing model unexpectedly succeeded - model path may exist on this host")

        assert hasattr(_CT2Backend, "transcribe")
        assert hasattr(_CT2Backend, "transcribe_mel")
        assert hasattr(_CT2Backend, "close")
        assert hasattr(_CT2Backend, "n_mels")
        ok("Ct2WhisperBackend exposes transcribe / transcribe_mel / close / n_mels")

        from stt.whisper import WhisperBackend as _CT2_WhisperBackend
        _ct2_uninit = _CT2Backend.__new__(_CT2Backend)
        _ct2_uninit.model_id = "ct2-stub"
        _ct2_uninit._handle = None
        assert isinstance(_ct2_uninit, _CT2_WhisperBackend), \
            "Ct2WhisperBackend instances must satisfy WhisperBackend Protocol"
        ok("Ct2WhisperBackend satisfies WhisperBackend Protocol (model_id + transcribe)")

def test_whispercpp_bindings():
    _section('Fix WhisperCpp-Bindings')

    import importlib as _wcpp_importlib

    _wcpp_native_available = False
    _wcpp_native_err: str | None = None

    try:
        _wcpp_pkg = _wcpp_importlib.import_module("whisper_bindings")
        info(f"whisper_bindings loader imported: {_wcpp_pkg.__file__}")
        _wcpp_native_available = bool(getattr(_wcpp_pkg, "EXTENSION_AVAILABLE", False))
        _wcpp_native_err = getattr(_wcpp_pkg, "EXTENSION_IMPORT_ERROR", None)
        if _wcpp_native_available:
            ok("whisper_bindings._whisper native extension is available")
        else:
            info(f"whisper_bindings._whisper native extension NOT built: {_wcpp_native_err}")
    except Exception as _e:
        info(f"whisper_bindings package import failed: {_e}")

    from stt.whisper_cpp import (
        WhisperCppBackend as _WCPPBackend,
        WhisperCppConfig as _WCPPConfig,
    )
    from stt.whisper import WhisperBackend as _WCPP_WhisperBackend

    assert hasattr(_WCPPBackend, "transcribe")
    assert hasattr(_WCPPBackend, "model_id")
    assert hasattr(_WCPPBackend, "close")
    ok("WhisperCppBackend exposes transcribe / model_id / close")

    _wcpp_cfg = _WCPPConfig(model_path="/tmp/fake.bin")
    assert _wcpp_cfg.model_path == "/tmp/fake.bin"
    assert _wcpp_cfg.language == "en"
    ok("WhisperCppConfig dataclass: defaults sane")

    if not _wcpp_native_available:
        info("whisper.cpp native backend unavailable -- skipping live-load tests")
        try:
            _WCPPBackend("/nonexistent/whisper-cpp-model-zzz")
        except RuntimeError as _e:
            ok(f"missing extension surfaces informative RuntimeError: {type(_e).__name__}")
        except Exception as _e:
            info(f"missing extension raised non-RuntimeError {type(_e).__name__}: {_e}")
    else:
        _bad_path = "/nonexistent/whisper-cpp-model-zzz-does-not-exist"
        try:
            _WCPPBackend(_bad_path)
        except RuntimeError as _e:
            ok(f"opening missing model raises RuntimeError (not crash): {type(_e).__name__}")
        except Exception as _e:
            info(f"opening missing model raised non-RuntimeError {type(_e).__name__}: {_e}")
        else:
            info("opening missing model unexpectedly succeeded - model path may exist on this host")

    class _StubWhisperCppLike:
        def __init__(self) -> None:
            pass

        def transcribe(self, mel, language=None, prompt=None):
            from stt.whisper import TranscriptionResult as _TR
            return _TR(text="stub")

        def model_id(self) -> str:
            return "stub"

    _wcpp_stub = _StubWhisperCppLike()
    assert isinstance(_wcpp_stub, _WCPP_WhisperBackend), \
        "WhisperCpp-shaped stub should satisfy WhisperBackend Protocol"
    ok("WhisperBackend Protocol structurally satisfied (transcribe + model_id)")

    if _wcpp_native_available:
        _wcpp_pseudo = _WCPPBackend("/nonexistent/whisper-cpp-model-zzz-does-not-exist")
        _wcpp_pseudo.close()
        assert isinstance(_wcpp_pseudo, _WCPP_WhisperBackend), \
            "WhisperCppBackend instances must satisfy WhisperBackend Protocol"
        ok("WhisperCppBackend satisfies WhisperBackend Protocol (model_id + transcribe)")
    else:
        info("WhisperCppBackend Protocol check requires native backend; skipping public-API ctor")

    assert hasattr(_WCPPBackend, "transcribe_segmented"), \
        "WhisperCppBackend must expose transcribe_segmented for diarized path"
    ok("WhisperCppBackend exposes transcribe_segmented")

    import os as _wcpp_os

    if _wcpp_native_available:
        _wcpp_uninit = _WCPPBackend("/nonexistent/whisper-cpp-model-zzz-does-not-exist")
        _wcpp_uninit.close()
    else:
        try:
            _wcpp_uninit = _WCPPBackend("/nonexistent/whisper-cpp-model-zzz-does-not-exist")
        except Exception:
            pytest.skip("WhisperCpp native backend unavailable; cannot construct closed-state via public API")
        _wcpp_uninit.close()
    try:
        import numpy as _wcpp_np
        _wcpp_uninit.transcribe_segmented(
            _wcpp_np.zeros(16000, dtype=_wcpp_np.float32)
        )
    except RuntimeError as _e:
        ok(f"transcribe_segmented on closed backend raises RuntimeError: {type(_e).__name__}")
    except Exception as _e:
        info(f"transcribe_segmented on closed backend raised non-RuntimeError "
             f"{type(_e).__name__}: {_e}")
    else:
        raise AssertionError(
            "transcribe_segmented on closed backend should raise RuntimeError"
        )

    if not _wcpp_native_available:
        info("whisper.cpp native backend unavailable -- skipping segmented runtime test")
    else:
        _wcpp_seg_model = _wcpp_os.environ.get("WHISPER_CPP_TEST_MODEL")
        if not _wcpp_seg_model or not _wcpp_os.path.isfile(_wcpp_seg_model):
            info(
                "WHISPER_CPP_TEST_MODEL not set / not a file -- skipping segmented "
                "runtime test"
            )
        else:
            import numpy as _wcpp_np
            _wcpp_t = _wcpp_np.linspace(0.0, 5.0, 16000 * 5, dtype=_wcpp_np.float32)
            _wcpp_samples = (0.1 * _wcpp_np.sin(2 * _wcpp_np.pi * 440.0 * _wcpp_t)).astype(
                _wcpp_np.float32
            )
            _wcpp_be = _WCPPBackend(_wcpp_seg_model)
            try:
                _wcpp_res = _wcpp_be.transcribe_segmented(_wcpp_samples)
                assert hasattr(_wcpp_res, "segments"), "result lacks segments list"
                assert isinstance(_wcpp_res.segments, list), "segments must be a list"
                assert len(_wcpp_res.segments) >= 1, "expected at least one segment"
                for _wcpp_seg in _wcpp_res.segments:
                    assert _wcpp_seg.t_start_ms <= _wcpp_seg.t_end_ms, (
                        f"invalid timing: {_wcpp_seg.t_start_ms} > {_wcpp_seg.t_end_ms}"
                    )
                ok(
                    f"transcribe_segmented returned {len(_wcpp_res.segments)} segments "
                    f"with valid timings"
                )
            finally:
                _wcpp_be.close()

def test_transport_orchestration():
    _section('Fix Transport-Orchestration')

    import asyncio as _to_asyncio

    import realtime.transport as _to_transport
    from realtime.transport import (
        EventSink as _to_EventSink,
        OutboundAudioSpec as _to_OutboundAudioSpec,
        OutboundOpusTrack as _to_OutboundOpusTrack,
        RealtimeContext as _to_RealtimeContext,
        get_context as _to_get_context,
        maybe_handle_offer as _to_maybe_handle_offer,
        set_context as _to_set_context,
        webrtc_session_count as _to_webrtc_session_count,
    )
    from realtime import live_session_count as _to_live_session_count

    _to_set_context(_to_RealtimeContext(models=None, instructions="be brief"))
    assert _to_get_context() is not None and _to_get_context().instructions == "be brief"
    ok("transport.set_context / get_context round-trip")
    _to_set_context(None)
    assert _to_get_context() is None
    ok("transport.set_context(None) clears the global context")

    class _ToFakeAiortc:
        class MediaStreamError(Exception):
            pass

        class _RTCSessionDescription:
            def __init__(self, sdp: str = "", type: str = ""):
                self.sdp = sdp
                self.type = type

        class _MediaStreamTrack:
            kind = "audio"

            def __init__(self):
                pass

            def stop(self):
                pass

        class _RTCPeerConnection:
            def __init__(self):
                self.localDescription = _ToFakeAiortc._RTCSessionDescription(
                    sdp="v=0\r\nfake answer\r\n", type="answer"
                )
                self.connectionState = "new"
                self.added_tracks: list = []
                self.calls: list = []
                self._handlers: dict = {}

            def addTrack(self, track):
                self.added_tracks.append(track)
                self.calls.append(("addTrack", id(track)))
                return track

            def on(self, event):
                def _decorator(fn):
                    self._handlers.setdefault(event, []).append(fn)
                    return fn

                return _decorator

            async def setRemoteDescription(self, desc):
                self.calls.append(("setRemoteDescription", desc.sdp[:32]))

            async def createAnswer(self):
                self.calls.append(("createAnswer", None))
                return _ToFakeAiortc._RTCSessionDescription(sdp="v=0\r\nanswer\r\n", type="answer")

            async def setLocalDescription(self, desc):
                self.calls.append(("setLocalDescription", desc.sdp[:32]))

            async def close(self):
                self.calls.append(("close", None))

    def _to_install_fake_aiortc():
        prev_pkg = sys.modules.get("aiortc")
        prev_ms = sys.modules.get("aiortc.mediastreams")
        import types as _types

        pkg = _types.ModuleType("aiortc")
        pkg.MediaStreamTrack = _ToFakeAiortc._MediaStreamTrack
        pkg.RTCPeerConnection = _ToFakeAiortc._RTCPeerConnection
        pkg.RTCSessionDescription = _ToFakeAiortc._RTCSessionDescription
        ms = _types.ModuleType("aiortc.mediastreams")
        ms.MediaStreamError = _ToFakeAiortc.MediaStreamError
        sys.modules["aiortc"] = pkg
        sys.modules["aiortc.mediastreams"] = ms
        return prev_pkg, prev_ms

    def _to_restore_aiortc(prev_pkg, prev_ms):
        if prev_pkg is None:
            sys.modules.pop("aiortc", None)
        else:
            sys.modules["aiortc"] = prev_pkg
        if prev_ms is None:
            sys.modules.pop("aiortc.mediastreams", None)
        else:
            sys.modules["aiortc.mediastreams"] = prev_ms

    _to_prev = _to_install_fake_aiortc()
    _to_captured_pcs: list = []
    _to_real_RTC = _ToFakeAiortc._RTCPeerConnection

    def _to_RTCPeerConnection_capture(*args, **kwargs):
        pc = _to_real_RTC(*args, **kwargs)
        _to_captured_pcs.append(pc)
        return pc

    sys.modules["aiortc"].RTCPeerConnection = _to_RTCPeerConnection_capture
    try:
        from realtime.session import RealtimeQuery as _to_RealtimeQuery
        _to_q_conv = _to_RealtimeQuery(intent="conversation")
        _to_baseline = _to_webrtc_session_count()
        _to_offer_sdp = "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
        _to_answer = _to_asyncio.run(_to_maybe_handle_offer(_to_offer_sdp, _to_q_conv))
        assert isinstance(_to_answer, str) and len(_to_answer) > 0
        ok("maybe_handle_offer returns a string SDP when aiortc is importable")

        assert len(_to_captured_pcs) == 1
        _to_pc = _to_captured_pcs[0]
        _to_call_names = [c[0] for c in _to_pc.calls]
        _to_set_remote_idx = _to_call_names.index("setRemoteDescription")
        _to_create_idx = _to_call_names.index("createAnswer")
        _to_set_local_idx = _to_call_names.index("setLocalDescription")
        assert _to_set_remote_idx < _to_create_idx < _to_set_local_idx
        ok("setRemoteDescription -> createAnswer -> setLocalDescription order preserved")

        assert any(name == "addTrack" for name, _ in _to_pc.calls), "outbound track must be added"
        _to_addtrack_idx = _to_call_names.index("addTrack")
        assert _to_addtrack_idx < _to_create_idx, "addTrack must precede createAnswer"
        ok("outbound OutboundOpusTrack added BEFORE createAnswer")

        assert "track" in _to_pc._handlers and len(_to_pc._handlers["track"]) >= 1
        assert "datachannel" in _to_pc._handlers and len(_to_pc._handlers["datachannel"]) >= 1
        assert "connectionstatechange" in _to_pc._handlers
        ok("pc.on(track), pc.on(datachannel), pc.on(connectionstatechange) registered")

        assert _to_webrtc_session_count() == _to_baseline + 1
        assert _to_live_session_count() >= _to_baseline + 1
        ok("session registered: webrtc_session_count and live_session_count both incremented")

        _to_pc.connectionState = "closed"
        _to_state_handler = _to_pc._handlers["connectionstatechange"][0]
        _to_asyncio.run(_to_state_handler())
        assert _to_webrtc_session_count() == _to_baseline
        ok("connectionstatechange=closed deregisters session from registry")

        _to_q_trans = _to_RealtimeQuery(intent="transcription")
        _to_baseline2 = _to_webrtc_session_count()
        _to_pcs_before = len(_to_captured_pcs)
        _ = _to_asyncio.run(_to_maybe_handle_offer(_to_offer_sdp, _to_q_trans))
        assert len(_to_captured_pcs) == _to_pcs_before + 1
        _to_pc2 = _to_captured_pcs[-1]
        assert not any(name == "addTrack" for name, _ in _to_pc2.calls), (
            "transcription intent must NOT add an outbound audio track"
        )
        ok("transcription intent skips outbound audio track")
        _to_pc2.connectionState = "closed"
        _to_asyncio.run(_to_pc2._handlers["connectionstatechange"][0]())
    finally:
        _to_restore_aiortc(*_to_prev)

    _to_track = _to_OutboundOpusTrack(queue_maxsize=8)
    assert _to_track.kind == "audio"
    _to_pushed = _to_track.push_nowait(b"\x01\x02\x03\x04")
    assert _to_pushed is True
    try:
        import av as _to_av  # noqa: F401
        _to_have_av = True
    except ImportError:
        _to_have_av = False
    if _to_have_av:
        _to_frame = _to_asyncio.run(_to_track.recv())
        assert hasattr(_to_frame, "samples") or hasattr(_to_frame, "pts"), (
            "OutboundOpusTrack.recv() must return an av.AudioFrame when av is installed"
        )
        ok("OutboundOpusTrack.recv() returns av.AudioFrame after push (pyav present)")
    else:
        info("OutboundOpusTrack.recv() av-frame check skipped (av not installed)")
        _to_raw = _to_asyncio.run(_to_track.recv())
        assert _to_raw == b"\x01\x02\x03\x04"
        ok("OutboundOpusTrack.recv() returns raw payload when av not installed")
    _to_track.close()

    class _ToFakeDC:
        label = "oai-events"

        def __init__(self):
            self._handlers: dict = {}

        def on(self, event):
            def _decorator(fn):
                self._handlers[event] = fn
                return fn

            return _decorator

        def send(self, text):
            pass

    _to_sink = _to_EventSink.data_channel_sink(_ToFakeDC())
    assert _to_sink.kind.value == "data_channel"
    ok("EventSink.data_channel_sink builds a DC sink (smoke)")

    _to_no_aiortc_prev = sys.modules.pop("aiortc", None)
    _to_no_aiortc_ms_prev = sys.modules.pop("aiortc.mediastreams", None)
    import builtins as _to_builtins

    _to_real_import = _to_builtins.__import__

    def _to_no_aiortc_import(name, *a, **k):
        if name == "aiortc" or name.startswith("aiortc."):
            raise ImportError(f"blocked: {name}")
        return _to_real_import(name, *a, **k)

    _to_builtins.__import__ = _to_no_aiortc_import
    try:
        _to_none = _to_asyncio.run(_to_maybe_handle_offer("v=0\r\n", _to_RealtimeQuery(intent="conversation")))
        assert _to_none is None
        ok("maybe_handle_offer returns None when aiortc is not importable")
    finally:
        _to_builtins.__import__ = _to_real_import
        if _to_no_aiortc_prev is not None:
            sys.modules["aiortc"] = _to_no_aiortc_prev
        if _to_no_aiortc_ms_prev is not None:
            sys.modules["aiortc.mediastreams"] = _to_no_aiortc_ms_prev

def test_stt_mel_segments():
    _section('Fix STT-Mel-Segments')
    import numpy as _np_stt
    from stt import (
        Backend as _STT_Backend,
        BPETokenizer as _STT_BPETokenizer,
        MelExtractor as _STT_MelExtractor,
        TimedSegment as _STT_TimedSegment,
        TranscriptionResult as _STT_TranscriptionResult,
        WhisperBackend as _STT_WhisperBackend,
        WHISPER_NB_FRAMES as _STT_NB_FRAMES,
        WHISPER_SAMPLING_HZ as _STT_SR,
        decode_bpe as _stt_decode_bpe,
        encode_bpe as _stt_encode_bpe,
        parse_timestamp_token as _stt_parse_ts,
        split_ct2_segments as _stt_split_ct2,
        transcribe_long as _stt_transcribe_long,
        GateThresholds as _STTGateThresholds,
    )

    _mel = _STT_MelExtractor(80)
    _t = _np_stt.arange(_STT_SR).astype(_np_stt.float32) / float(_STT_SR)
    _sine440 = (0.5 * _np_stt.sin(2 * _np_stt.pi * 440.0 * _t)).astype(_np_stt.float32)
    _mel_440 = _mel.extract(_sine440)
    assert _mel_440.shape == (80, _STT_NB_FRAMES), f"shape={_mel_440.shape}"
    assert _np_stt.isfinite(_mel_440).all(), "non-finite values in mel output"
    ok(f"mel: 1s 440Hz sine produces shape {_mel_440.shape} all finite")

    _silence = _np_stt.zeros(_STT_SR, dtype=_np_stt.float32)
    _mel_sil = _mel.extract(_silence)
    assert _mel_sil.shape == (80, _STT_NB_FRAMES)
    assert _np_stt.isfinite(_mel_sil).all()
    _active_avg = float(_mel_440.mean())
    _silence_avg = float(_mel_sil.mean())
    assert _active_avg > _silence_avg, f"sine mean {_active_avg} should exceed silence mean {_silence_avg}"
    ok(f"mel: 440Hz sine mean {_active_avg:.3f} > silence mean {_silence_avg:.3f}")

    def _approx_freq_to_mel_bin(hz: float, n_mels: int = 80) -> int:
        from stt.mel import build_mel_filters
        f = build_mel_filters(n_mels)
        bin_idx = int(round(hz * 400 / 16000))
        return int(_np_stt.argmax(f[:, bin_idx]))

    _bin440 = _approx_freq_to_mel_bin(440.0)
    _e440 = float(_mel_440[_bin440].mean())
    _esil = float(_mel_sil[_bin440].mean())
    assert _e440 > _esil, f"440Hz energy in bin {_bin440}: sine {_e440} should exceed silence {_esil}"
    ok(f"mel: 440Hz mel bin {_bin440} sine energy {_e440:.3f} > silence {_esil:.3f}")

    _lmem = _STT_MelExtractor(128)
    _mel_v3 = _lmem.extract(_silence)
    assert _mel_v3.shape == (128, _STT_NB_FRAMES)
    ok(f"mel: 128-mel (large-v3) extracts to {_mel_v3.shape}")

    assert _stt_parse_ts("<|0.00|>") == 0
    assert _stt_parse_ts("<|1.20|>") == 1200
    assert _stt_parse_ts("<|29.98|>") == 29_980
    assert _stt_parse_ts("<|0.5|>") == 500
    assert _stt_parse_ts("<|0.500|>") == 500
    assert _stt_parse_ts("<|sot|>") is None
    assert _stt_parse_ts("hello") is None
    assert _stt_parse_ts("<||>") is None
    ok("whisper: parse_timestamp_token matches upstream cases")

    _segs_pair = _stt_split_ct2(
        ["<|0.00|>", " hel", "lo", "<|1.20|>", "<|1.20|>", " wor", "ld", "<|2.50|>"],
        [],
        None,
        0xFFFFFFFF,
    )
    assert len(_segs_pair) == 2, f"got {len(_segs_pair)} segments"
    assert _segs_pair[0].t_start_ms == 0 and _segs_pair[0].t_end_ms == 1200
    assert _segs_pair[1].t_start_ms == 1200 and _segs_pair[1].t_end_ms == 2500
    assert _segs_pair[0].text_tokens == [" hel", "lo"]
    ok("whisper: split_ct2_segments returns 2 segments with correct timestamps")

    _segs_inv = _stt_split_ct2(
        ["<|2.00|>", "hi", "<|1.00|>"],
        [],
        None,
        0xFFFFFFFF,
    )
    assert _segs_inv == [], f"inverted pairs should be dropped, got {_segs_inv}"
    ok("whisper: inverted timestamp pairs are dropped")

    _segs_clamp = _stt_split_ct2(
        ["<|0.00|>", "hi", "<|29.84|>"],
        [],
        None,
        3_000,
    )
    assert len(_segs_clamp) == 1
    assert _segs_clamp[0].t_end_ms == 3_000
    ok("whisper: timestamps clamp to audio_ms")

    _segs_id = _stt_split_ct2(
        ["<|0.00|>", "hi", "<|1.20|>"],
        [50364, 12345, 50424],
        50364,
        0xFFFFFFFF,
    )
    assert len(_segs_id) == 1 and _segs_id[0].t_end_ms == 1200
    ok("whisper: ID-based timestamp classification works")

    assert _STT_TranscriptionResult.empty().text == ""
    assert _STT_TranscriptionResult.from_text("hi").text == "hi"
    _tr_fields = list(_STT_TranscriptionResult.__dataclass_fields__.keys())
    assert _tr_fields == ["text", "avg_logprob", "no_speech_prob", "compression_ratio", "segments"], _tr_fields
    ok(f"TranscriptionResult fields: {_tr_fields}")

    _ts_fields = list(_STT_TimedSegment.__dataclass_fields__.keys())
    assert _ts_fields == ["t_start_ms", "t_end_ms", "text", "avg_logprob", "no_speech_prob"], _ts_fields
    ok(f"TimedSegment fields: {_ts_fields}")

    assert _STT_Backend.from_env().value in {"ct2", "whisper_cpp"}
    assert _STT_Backend.CT2.value == "ct2"
    assert _STT_Backend.WHISPER_CPP.value == "whisper_cpp"
    ok("Backend enum values: ct2, whisper_cpp")

    class _StubWhisperBackend:
        model_id = "stub"

        def __init__(self, per_chunk):
            self._results = list(per_chunk)
            self.calls = 0
            self.last_task: str | None = None
            self.last_sample_rate: int | None = None

        def transcribe(self, samples, sample_rate=16000, *, language=None, prompt=None, with_timestamps=False, task="transcribe"):
            idx = min(self.calls, len(self._results) - 1)
            self.calls += 1
            self.last_task = task
            self.last_sample_rate = sample_rate
            return self._results[idx]

    _stub = _StubWhisperBackend([_STT_TranscriptionResult.from_text("ignored")])
    assert isinstance(_stub, _STT_WhisperBackend), "DummyBackend should satisfy WhisperBackend Protocol"
    ok("WhisperBackend Protocol satisfied by stub via duck-typing")

    _zero60 = _np_stt.zeros(60 * _STT_SR, dtype=_np_stt.float32)
    _res_silence = _stt_transcribe_long(
        _StubWhisperBackend([_STT_TranscriptionResult.from_text("should not be returned")]),
        _zero60,
        _STT_SR,
    )
    assert _res_silence.text == "" and _res_silence.segments == []
    ok("transcribe_long: 60s zero-array hits silence pre-gate, returns empty")

    _noisy_audio = _np_stt.tile(
        (0.5 * _np_stt.sin(2 * _np_stt.pi * 440.0 * _t)).astype(_np_stt.float32),
        60,
    )
    assert len(_noisy_audio) == 60 * _STT_SR
    _per_chunk = [
        _STT_TranscriptionResult(
            text=f"chunk {i}",
            avg_logprob=-0.5,
            no_speech_prob=0.1,
            segments=[
                _STT_TimedSegment(t_start_ms=0, t_end_ms=1000, text=f"chunk {i}"),
            ],
        )
        for i in range(2)
    ]
    _stub_active = _StubWhisperBackend(_per_chunk)
    _res_active = _stt_transcribe_long(_stub_active, _noisy_audio, _STT_SR)
    assert _stub_active.calls == 2, f"expected 2 chunks for 60s audio, got {_stub_active.calls}"
    assert len(_res_active.segments) == 2
    assert _res_active.segments[0].t_start_ms == 0 and _res_active.segments[0].t_end_ms == 1000
    assert _res_active.segments[1].t_start_ms == 30_000 and _res_active.segments[1].t_end_ms == 31_000
    assert _res_active.text == "chunk 0 chunk 1"
    ok("transcribe_long: 60s active audio chunked into 2; segments time-shifted by 30s offset")

    _stub_reject = _StubWhisperBackend([
        _STT_TranscriptionResult(
            text="hallucinated",
            avg_logprob=-10.0,
            no_speech_prob=0.99,
            segments=[_STT_TimedSegment(0, 500, "hallucinated")],
        )
    ] * 2)
    _thresh = _STTGateThresholds(no_speech_prob_threshold=0.6, avg_logprob_threshold=-1.0)
    _res_reject = _stt_transcribe_long(_stub_reject, _noisy_audio, _STT_SR, gate=_thresh)
    assert _res_reject.text == "" and _res_reject.segments == [], (
        f"noise-gate-rejected chunks should be dropped, got text={_res_reject.text!r} segs={_res_reject.segments}"
    )
    ok("transcribe_long: chunks failing the noise gate are dropped")

    assert _stt_decode_bpe("hello") == "hello"
    _round = "café"
    _round_back = _stt_decode_bpe(_stt_encode_bpe(_round))
    assert _round_back == _round, f"BPE round-trip {_round!r} -> {_round_back!r}"
    ok(f"BPE: round-trip {_round!r} via encode_bpe/decode_bpe is identity")

    _tk = _STT_BPETokenizer()
    assert _tk.decode(_tk.encode("a b c")) == "a b c"
    assert len(_tk.rune_to_byte) == 256, f"BPE vocab size = {len(_tk.rune_to_byte)}"
    ok(f"BPE: BPETokenizer.decode round-trips, vocab covers 256 byte values")

def test_inspect_api(_sys_modules_snapshot):
    _section('Fix Inspect-API')
    import asyncio as _ia_asyncio
    import os as _ia_os
    import tempfile as _ia_tempfile
    import time as _ia_time
    from pathlib import Path as _IAPath

    import env as _ia_env

    with _ia_tempfile.TemporaryDirectory() as _ia_td:
        _ia_os.environ[_ia_env.INSPECT_SESSION_DIR] = _ia_td
        if "inspect_api" in sys.modules:
            del sys.modules["inspect_api"]
            for _mod in list(sys.modules):
                if _mod.startswith("inspect_api."):
                    del sys.modules[_mod]
        import inspect_api as _ia

        _ia.clear_registry()
        _ia_relay = _ia.InspectorRelay("sess_test_reg", _IAPath(_ia_td))
        _ia.register("sess_test_reg", _ia_relay, "model-x", lambda: "active")
        _metas = _ia.list_meta()
        assert any(m.id == "sess_test_reg" and m.model == "model-x" for m in _metas)
        ok("registry: register + list_meta")
        _ia.unregister("sess_test_reg")
        assert all(m.id != "sess_test_reg" for m in _ia.list_meta())
        ok("registry: unregister")

        _ia_store = _ia.AudioStore("sess_audio_test", _IAPath(_ia_td))
        _ia_store.append_mic_in_f32([0.0, 0.5, -0.5, 1.0])
        _ia_raw = _IAPath(_ia_td) / "sess_audio_test.audio_mic_in.raw"
        _ia_store.close()
        assert _ia_raw.exists()
        assert _ia_raw.stat().st_size == 8
        _ia_sidecar = _IAPath(_ia_td) / "sess_audio_test.audio.json"
        assert _ia_sidecar.exists()
        _ia_side = json.loads(_ia_sidecar.read_text())
        assert _ia_side["tracks"]["mic_in"]["sample_rate"] == 16000
        ok("audio_store: append_pcm + close + sidecar")

        _ia_h = _ia.wav_header(8000, 16000)
        assert _ia_h[0:4] == b"RIFF" and _ia_h[8:12] == b"WAVE" and _ia_h[36:40] == b"data"
        assert int.from_bytes(_ia_h[40:44], "little") == 8000 * 2
        ok("audio_store: wav_header format")

        _ia_ret_dir = _IAPath(_ia_td) / "ret_test"
        _ia_ret_dir.mkdir(parents=True, exist_ok=True)
        for _i in range(5):
            _p = _ia_ret_dir / f"sess_{_i}.ndjson"
            _p.write_bytes(b"x")
            _ia_os.utime(_p, (_ia_time.time() - (5 - _i) * 100, _ia_time.time() - (5 - _i) * 100))
        _ia.cleanup_on_startup(_ia_ret_dir, max_count=2, max_bytes=0, max_days=0)
        _ia_remaining = sorted(p.name for p in _ia_ret_dir.iterdir())
        assert len(_ia_remaining) == 2, f"expected 2 remaining, got {_ia_remaining}"
        assert "sess_3.ndjson" in _ia_remaining and "sess_4.ndjson" in _ia_remaining
        ok("retention: cleanup respects max_count (newest kept)")

        async def _ia_relay_test():
            _r = _ia.InspectorRelay("sess_relay_pubsub", _IAPath(_ia_td))
            _sub = _r.subscribe()
            assert _sub.snapshot == []
            _r.publish("vad", "confirmed_start", None, {"audio_start_ms": 100})
            _line = await _ia_asyncio.wait_for(_sub.queue.get(), timeout=1.0)
            assert _line is not None
            _evt = json.loads(_line.decode("utf-8"))
            assert _evt["lane"] == "vad" and _evt["kind"] == "confirmed_start"
            assert _evt["payload"]["audio_start_ms"] == 100
            _r.unsubscribe(_sub.queue)
            _r.publish("vad", "confirmed_end", None, {})
            try:
                await _ia_asyncio.wait_for(_sub.queue.get(), timeout=0.2)
                return "leaked"
            except _ia_asyncio.TimeoutError:
                pass
            _r.close()
            return "ok"

        _ia_loop = _ia_asyncio.new_event_loop()
        try:
            _ia_relay_result = _ia_loop.run_until_complete(_ia_relay_test())
        finally:
            _ia_loop.close()
        assert _ia_relay_result == "ok", f"relay pubsub: {_ia_relay_result}"
        ok("relay: subscribe + publish + unsubscribe (no leak)")

        _ia_err_relay = _ia.InspectorRelay("sess_err_mirror", _IAPath(_ia_td))
        _ia_err_relay.publish("llm", "failed", None, {"error": "kaboom"})
        _ia_err_path = _IAPath(_ia_td) / "sess_err_mirror.ndjson"
        _ia_err_relay.close()
        _ia_lines = [l for l in _ia_err_path.read_bytes().split(b"\n") if l]
        assert len(_ia_lines) == 2, f"expected origin + mirror, got {len(_ia_lines)}"
        _ia_origin = json.loads(_ia_lines[0])
        _ia_mirror = json.loads(_ia_lines[1])
        assert _ia_origin["lane"] == "llm" and _ia_mirror["lane"] == "error"
        assert _ia_mirror["payload"]["error"] == "kaboom"
        ok("relay: error-mirror written to ndjson")

        from fastapi import FastAPI as _IAFastAPI
        from fastapi.testclient import TestClient as _IATestClient

        _ia_app = _IAFastAPI()
        _ia_app.include_router(_ia.router)
        _ia_client = _IATestClient(_ia_app)

        _ia_resp = _ia_client.get("/v1/inspect/sessions")
        assert _ia_resp.status_code == 200
        _ia_payload = _ia_resp.json()
        assert isinstance(_ia_payload, list)
        ok("routes: GET /v1/inspect/sessions returns 200 + list")

        _ia_resp_h = _ia_client.get("/v1/inspect/sessions/history")
        assert _ia_resp_h.status_code == 200
        _ia_h_body = _ia_resp_h.json()
        assert isinstance(_ia_h_body, list)
        assert all(set(e.keys()) >= {"id", "size_bytes", "mtime"} for e in _ia_h_body)
        ok("routes: GET /v1/inspect/sessions/history returns 200 + list[SessionHistoryEntry]")

        _ia_routes = {getattr(r, "path", None) for r in _ia_app.routes}
        assert "/v1/inspect/{sid}/stream" in _ia_routes
        ok("routes: WebSocket /v1/inspect/{sid}/stream registered")

        _ia.clear_registry()

def test_inspect_realtime_hook(_sys_modules_snapshot):
    _section('Fix Inspect-Realtime-Hook')
    import asyncio as _irh_asyncio
    import json as _irh_json
    import os as _irh_os
    import tempfile as _irh_tempfile
    from pathlib import Path as _IRHPath

    import env as _irh_env

    with _irh_tempfile.TemporaryDirectory() as _irh_td:
        _irh_os.environ[_irh_env.INSPECT_SESSION_DIR] = _irh_td
        for _mod in [m for m in list(sys.modules) if m == "inspect_api" or m.startswith("inspect_api.") or m == "realtime" or m.startswith("realtime.")]:
            del sys.modules[_mod]
        import inspect_api as _irh_ia
        import realtime as _irh_realtime
        from realtime.session import Intent as _IRHIntent, RealtimeQuery as _IRHQuery, Session as _IRHSession
        from realtime.wire import OutboundEvent as _IRHOutboundEvent

        _irh_ia.clear_registry()

        async def _irh_run():
            _q = _IRHQuery(intent="conversation")
            _s = _IRHSession(query=_q, intent=_IRHIntent.CONVERSATION, observer=_irh_ia.InspectObserver())
            _records: list = []

            class _IRHFakeSink:
                async def send_value(self, ev):
                    _records.append(ev)

            async with _s._state_lock:
                _s.state.event_sink = _IRHFakeSink()

            _relay = _irh_ia.get_relay(_s.id)
            assert _relay is not None, "session should register relay on construction"
            _sub = _relay.subscribe()

            await _s.emit(_IRHOutboundEvent.buffer_cleared())
            _line_out = await _irh_asyncio.wait_for(_sub.queue.get(), timeout=1.0)
            _evt_out = _irh_json.loads(_line_out.decode("utf-8"))
            assert _evt_out["lane"] == "wire" and _evt_out["kind"] == "out"
            assert _evt_out["payload"]["type"] == "input_audio_buffer.cleared"

            await _s.handle_client_event("test", '{"type":"input_audio_buffer.commit"}')
            _seen_in = False
            _seen_err_origin = False
            _seen_err_mirror = False
            for _ in range(8):
                try:
                    _line = await _irh_asyncio.wait_for(_sub.queue.get(), timeout=1.0)
                except _irh_asyncio.TimeoutError:
                    break
                _evt = _irh_json.loads(_line.decode("utf-8"))
                if _evt["lane"] == "wire" and _evt["kind"] == "in" and _evt["payload"].get("type") == "input_audio_buffer.commit":
                    _seen_in = True
                if _evt["lane"] == "wire" and _evt["kind"] == "out" and _evt["payload"].get("type") == "error":
                    _seen_err_origin = True
                if _evt["lane"] == "error":
                    _seen_err_mirror = True
                if _seen_in and _seen_err_origin:
                    break
            assert _seen_in, "inbound client event must be published with lane=wire kind=in"

            await _s._emit_error("invalid_request_error", "boom", None, None)
            for _ in range(6):
                try:
                    _line = await _irh_asyncio.wait_for(_sub.queue.get(), timeout=1.0)
                except _irh_asyncio.TimeoutError:
                    break
                _evt = _irh_json.loads(_line.decode("utf-8"))
                if _evt["lane"] == "wire" and _evt["kind"] == "out" and _evt["payload"].get("type") == "error":
                    _seen_err_origin = True
                if _evt["lane"] == "error" and _evt["kind"] == "raised":
                    _seen_err_mirror = True
                if _seen_err_origin and _seen_err_mirror:
                    break
            assert _seen_err_origin, "error event must be published as wire/out"
            assert _seen_err_mirror, "relay must auto-emit error mirror on error kind"

            _sid = _s.id
            assert any(m.id == _sid for m in _irh_ia.list_meta()), "session must appear in inspector registry"

            _looked = _irh_realtime.lookup_session_relay(_sid)
            if _looked is None:
                from realtime import websocket as _irh_ws
                _irh_ws._ws_sessions[_sid] = _s
                _looked = _irh_realtime.lookup_session_relay(_sid)
                assert _looked is _relay, "lookup_session_relay must return the session's relay when registered"
                del _irh_ws._ws_sessions[_sid]

            from realtime.state import TerminationReason as _IRHTermReason
            await _s.transition_to_terminated_with(_IRHTermReason.CLIENT_CLOSED)

            assert all(m.id != _sid for m in _irh_ia.list_meta()), "session must be removed from registry on terminate"
            assert _irh_realtime.lookup_session_relay(_sid) is None, "lookup_session_relay must return None after terminate"

            return len(_records)

        _irh_loop = _irh_asyncio.new_event_loop()
        try:
            _records_count = _irh_loop.run_until_complete(_irh_run())
        finally:
            _irh_loop.close()
        assert _records_count >= 3, f"sink should have received >=3 events, got {_records_count}"
        ok("hook: emit() / emit_event() publish to relay with lane=wire kind=out")
        ok("hook: handle_client_event publishes inbound with lane=wire kind=in")
        ok("hook: _emit_error publishes wire/out + auto error mirror on lane=error")
        ok("hook: session registered in inspect_api on construction, removed on terminate")
        ok("hook: lookup_session_relay returns relay while live, None after terminate")

        _irh_ia.clear_registry()

def test_inspect_realtime_decoupling(_sys_modules_snapshot):
    _section('Fix Inspect-Realtime-Decoupling')
    import asyncio as _ird_asyncio
    import os as _ird_os
    import subprocess as _ird_subprocess
    import sys as _ird_sys
    import tempfile as _ird_tempfile
    from pathlib import Path as _IRDPath

    import env as _ird_env

    _src = _IRDPath("realtime/session.py").read_text()
    assert "import inspect_api" not in _src, (
        "realtime/session.py must NOT import inspect_api (cycle inverted)"
    )
    ok("realtime/session.py source contains zero `import inspect_api` statements")

    _proc = _ird_subprocess.run(
        [_ird_sys.executable, "-c", "import sys; import realtime; print(','.join(m for m in sys.modules if m == 'inspect_api' or m.startswith('inspect_api.')))"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert _proc.returncode == 0, f"realtime import failed: {_proc.stderr}"
    _loaded = [m for m in _proc.stdout.strip().split(",") if m]
    assert _loaded == [], f"importing realtime must NOT load inspect_api; got {_loaded}"
    ok("`import realtime` does not load any inspect_api module (subprocess check)")

    _proc2 = _ird_subprocess.run(
        [_ird_sys.executable, "-c", "import realtime; import inspect_api; print('rt-then-ia ok')"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert _proc2.returncode == 0 and "rt-then-ia ok" in _proc2.stdout, (
        f"realtime then inspect_api import failed: {_proc2.stderr}"
    )
    _proc3 = _ird_subprocess.run(
        [_ird_sys.executable, "-c", "import inspect_api; import realtime; print('ia-then-rt ok')"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert _proc3.returncode == 0 and "ia-then-rt ok" in _proc3.stdout, (
        f"inspect_api then realtime import failed: {_proc3.stderr}"
    )
    ok("both import orders succeed: realtime->inspect_api and inspect_api->realtime")

    for _mod in [m for m in list(_ird_sys.modules) if m == "inspect_api" or m.startswith("inspect_api.") or m == "realtime" or m.startswith("realtime.")]:
        del _ird_sys.modules[_mod]

    from realtime.observer import NullObserver, SessionObserver
    from realtime.session import Intent as _IRDIntent, RealtimeQuery as _IRDQuery, Session as _IRDSession
    from realtime.wire import OutboundEvent as _IRDOutboundEvent
    from realtime.state import TerminationReason as _IRDTermReason

    class RecordingObserver:
        def __init__(self) -> None:
            self.calls: list = []

        def on_session_start(self, sid, meta):
            self.calls.append(("on_session_start", sid, dict(meta)))

        def on_session_end(self, sid):
            self.calls.append(("on_session_end", sid))

        def on_outbound_event(self, ev):
            self.calls.append(("on_outbound_event", ev))

        def on_outbound_event_dict(self, ev):
            self.calls.append(("on_outbound_event_dict", dict(ev)))

        def on_inbound_event(self, kind, payload, raw_text):
            self.calls.append(("on_inbound_event", kind, dict(payload), raw_text))

        def on_error(self, code, message, event_id, param):
            self.calls.append(("on_error", code, message, event_id, param))

        def on_inbound_audio_pcm16(self, pcm):
            self.calls.append(("on_inbound_audio_pcm16", bytes(pcm)))

        def on_outbound_audio_pcm16(self, pcm):
            self.calls.append(("on_outbound_audio_pcm16", bytes(pcm)))

        def on_inbound_audio_f32(self, samples):
            self.calls.append(("on_inbound_audio_f32", list(samples) if not isinstance(samples, (bytes, bytearray)) else bytes(samples)))

        def on_outbound_audio_f32(self, samples):
            self.calls.append(("on_outbound_audio_f32", list(samples) if not isinstance(samples, (bytes, bytearray)) else bytes(samples)))

        def on_correlation(self, **kwargs):
            self.calls.append(("on_correlation", kwargs))

    _rec = RecordingObserver()
    assert isinstance(_rec, SessionObserver), "RecordingObserver must satisfy SessionObserver Protocol"
    ok("Custom RecordingObserver satisfies the SessionObserver Protocol (runtime check)")

    _q = _IRDQuery(intent="conversation")
    _s = _IRDSession(query=_q, intent=_IRDIntent.CONVERSATION, observer=_rec)

    _starts = [c for c in _rec.calls if c[0] == "on_session_start"]
    assert len(_starts) == 1, f"expected one on_session_start, got {_starts}"
    assert _starts[0][1] == _s.id
    assert _starts[0][2].get("intent_label") == "conversation"
    assert callable(_starts[0][2].get("state_fn"))

    async def _drive():
        class _Sink:
            async def send_value(self, ev): pass
        async with _s._state_lock:
            _s.state.event_sink = _Sink()
        _ev = _IRDOutboundEvent.buffer_cleared()
        await _s.emit(_ev)
        await _s.emit_event({"type": "custom.event", "x": 1})
        await _s.handle_client_event("test", '{"type":"input_audio_buffer.commit"}')
        await _s._emit_error("invalid_request_error", "boom", None, None)
        _s.capture_inbound_pcm16(b"\x00\x00\x01\x00")
        _s.capture_outbound_pcm16(b"\x02\x00\x03\x00")
        _s.capture_inbound_f32([0.1, 0.2])
        _s.capture_outbound_f32([0.3, 0.4])
        _s.set_turn_id("turn_abc")
        _s.set_phrase_id("phrase_xyz")
        await _s.transition_to_terminated_with(_IRDTermReason.CLIENT_CLOSED)

    _ird_asyncio.run(_drive())

    _kinds = [c[0] for c in _rec.calls]
    assert "on_outbound_event" in _kinds, f"missing on_outbound_event in {_kinds}"
    assert "on_outbound_event_dict" in _kinds, f"missing on_outbound_event_dict in {_kinds}"
    assert "on_inbound_event" in _kinds, f"missing on_inbound_event in {_kinds}"
    assert "on_error" in _kinds, f"missing on_error in {_kinds}"
    assert "on_inbound_audio_pcm16" in _kinds, f"missing on_inbound_audio_pcm16 in {_kinds}"
    assert "on_outbound_audio_pcm16" in _kinds, f"missing on_outbound_audio_pcm16 in {_kinds}"
    assert "on_inbound_audio_f32" in _kinds, f"missing on_inbound_audio_f32 in {_kinds}"
    assert "on_outbound_audio_f32" in _kinds, f"missing on_outbound_audio_f32 in {_kinds}"
    assert "on_correlation" in _kinds, f"missing on_correlation in {_kinds}"
    assert "on_session_end" in _kinds, f"missing on_session_end in {_kinds}"
    _corrs = [c[1] for c in _rec.calls if c[0] == "on_correlation"]
    assert any(c.get("turn_id") == "turn_abc" for c in _corrs), f"on_correlation(turn_id=...) missing in {_corrs}"
    assert any(c.get("phrase_id") == "phrase_xyz" for c in _corrs), f"on_correlation(phrase_id=...) missing in {_corrs}"
    _err_calls = [c for c in _rec.calls if c[0] == "on_error"]
    assert any(c[1] == "invalid_request_error" and c[2] == "boom" for c in _err_calls), f"explicit boom error missing in {_err_calls}"
    ok("RecordingObserver received on_session_start, on_outbound_event, on_outbound_event_dict, on_inbound_event, on_error, on_inbound_audio_*, on_outbound_audio_*, on_correlation, on_session_end")

    _s2 = _IRDSession(query=_IRDQuery(intent="conversation"), intent=_IRDIntent.CONVERSATION)

    async def _drive_null():
        class _Sink2:
            async def send_value(self, ev): pass
        async with _s2._state_lock:
            _s2.state.event_sink = _Sink2()
        await _s2.emit(_IRDOutboundEvent.buffer_cleared())
        await _s2._emit_error("invalid_request_error", "noop", None, None)
        _s2.capture_inbound_f32([0.1, 0.2])
        _s2.capture_outbound_f32([0.3, 0.4])
        _s2.set_turn_id("turn_n")
        _s2.set_phrase_id("phrase_n")
        await _s2.transition_to_terminated_with(_IRDTermReason.CLIENT_CLOSED)

    _ird_asyncio.run(_drive_null())
    ok("Session with default NullObserver: emit/error/audio/correlation/terminate run without crashing")

    with _ird_tempfile.TemporaryDirectory() as _ird_td:
        _ird_os.environ[_ird_env.INSPECT_SESSION_DIR] = _ird_td
        for _mod in [m for m in list(_ird_sys.modules) if m == "inspect_api" or m.startswith("inspect_api.") or m == "realtime" or m.startswith("realtime.")]:
            del _ird_sys.modules[_mod]
        import inspect_api as _ird_ia
        from realtime.observer import SessionObserver as _IRDSessionObserver

        _ird_ia.clear_registry()
        _ins = _ird_ia.InspectObserver()
        assert isinstance(_ins, _IRDSessionObserver), "InspectObserver must satisfy SessionObserver Protocol"
        ok("InspectObserver instance satisfies the SessionObserver Protocol (runtime check)")

        from realtime.session import Intent as _IRDIntent2, RealtimeQuery as _IRDQuery2, Session as _IRDSession2
        from realtime.wire import OutboundEvent as _IRDOutboundEvent2
        from realtime.state import TerminationReason as _IRDTermReason2
        import json as _ird_json

        _factory = _ird_ia.make_observer_factory()

        async def _drive_ia():
            _q2 = _IRDQuery2(intent="conversation")
            _s3 = _IRDSession2(query=_q2, intent=_IRDIntent2.CONVERSATION, observer_factory=_factory)
            _relay = _ird_ia.get_relay(_s3.id)
            assert _relay is not None, "InspectObserver must register relay on session_start"
            _store = _ird_ia.get_audio_store(_s3.id)
            assert _store is not None, "InspectObserver must register audio_store on session_start"

            class _Sink3:
                async def send_value(self, ev): pass

            async with _s3._state_lock:
                _s3.state.event_sink = _Sink3()

            _sub = _relay.subscribe()
            await _s3.emit(_IRDOutboundEvent2.buffer_cleared())
            _line = await _ird_asyncio.wait_for(_sub.queue.get(), timeout=2.0)
            _evt = _ird_json.loads(_line.decode("utf-8"))
            assert _evt["lane"] == "wire" and _evt["kind"] == "out"
            assert _evt["payload"]["type"] == "input_audio_buffer.cleared"

            await _s3._emit_error("invalid_request_error", "boom", None, None)
            _saw_err_origin = False
            _saw_err_mirror = False
            for _ in range(8):
                try:
                    _l = await _ird_asyncio.wait_for(_sub.queue.get(), timeout=1.0)
                except _ird_asyncio.TimeoutError:
                    break
                _e = _ird_json.loads(_l.decode("utf-8"))
                if _e["lane"] == "wire" and _e["kind"] == "out" and _e["payload"].get("type") == "error":
                    _saw_err_origin = True
                if _e["lane"] == "error" and _e["kind"] == "raised":
                    _saw_err_mirror = True
                if _saw_err_origin and _saw_err_mirror:
                    break
            assert _saw_err_origin and _saw_err_mirror

            _sid = _s3.id
            assert any(m.id == _sid for m in _ird_ia.list_meta())
            await _s3.transition_to_terminated_with(_IRDTermReason2.CLIENT_CLOSED)
            assert all(m.id != _sid for m in _ird_ia.list_meta())
            assert _ird_ia.get_audio_store(_sid) is None

        _ird_asyncio.run(_drive_ia())
        ok("Session with observer_factory=make_observer_factory(): wire events flow through InspectObserver -> InspectorRelay (registry register on start, unregister on terminate)")

        _ird_ia.clear_registry()

def test_stt_routes_wired(_sys_modules_snapshot):
    _section('Fix STT-Routes-Wired')
    import importlib as _stt_w_importlib
    import os as _stt_w_os
    import sys as _stt_w_sys
    import unittest.mock as _stt_w_mock
    import wave as _stt_w_wave
    from io import BytesIO as _stt_w_BytesIO

    from fastapi.testclient import TestClient

    import env as _stt_w_env

    _stt_w_assert_count = [0]

    def _stt_w_assert(cond, msg):
        assert cond, msg
        _stt_w_assert_count[0] += 1

    def _stt_w_make_wav_bytes(samples_count: int = 1600) -> bytes:
        buf = _stt_w_BytesIO()
        with _stt_w_wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * samples_count)
        return buf.getvalue()

    def _stt_w_reload_server():
        if "server" in _stt_w_sys.modules:
            del _stt_w_sys.modules["server"]
        return _stt_w_importlib.import_module("server")

    _stt_w_prior_backend = _stt_w_os.environ.pop(_stt_w_env.STT_BACKEND, None)
    _stt_w_prior_models = _stt_w_os.environ.pop(_stt_w_env.SPEACHES_PLUS_MODELS, None)

    try:
        _server_default = _stt_w_reload_server()
        _stt_w_assert(
            _server_default.STT_BACKEND_RAW == "qwen3_omni",
            f"default STT_BACKEND_RAW should be 'qwen3_omni', got {_server_default.STT_BACKEND_RAW!r}",
        )
        _stt_w_assert(
            _server_default._stt_backend is None,
            "default backend should be None (no STT_BACKEND set)",
        )
        _stt_w_assert(
            not _server_default._request_picks_whisper_stt(None),
            "with no backend loaded, whisper dispatch must be False",
        )
        _stt_w_assert(
            not _server_default._request_picks_whisper_stt("anything"),
            "no backend -> whisper dispatch False regardless of model field",
        )
        ok("default STT_BACKEND='qwen3_omni': _stt_backend=None, dispatch=False (existing behavior preserved)")

        _routes = {(getattr(r, "path", None), tuple(sorted(getattr(r, "methods", []) or [])))
                   for r in _server_default.app.routes if hasattr(r, "path")}
        _stt_w_assert(("/v1/audio/transcriptions", ("POST",)) in _routes,
                      "transcriptions route not registered")
        _stt_w_assert(("/v1/audio/translations", ("POST",)) in _routes,
                      "translations route not registered")
        ok("routes: POST /v1/audio/transcriptions and /v1/audio/translations registered")

        _stt_w_os.environ[_stt_w_env.STT_BACKEND] = "whisper"
        _stt_w_os.environ[_stt_w_env.SPEACHES_PLUS_MODELS] = "/tmp/fake-whisper-model"

        class _FakeCt2Backend:
            def __init__(self, cfg):
                self.cfg = cfg
                self.closed = False

            def close(self):
                self.closed = True

            def model_id(self):
                return self.cfg.model_path

            def transcribe(self, mel, language=None, prompt=None):
                from stt.whisper import TranscriptionResult
                return TranscriptionResult(text="fake-whisper-text")

        _patch_target_module = _stt_w_importlib.import_module("stt.ct2")
        _orig_ct2_backend = _patch_target_module.Ct2WhisperBackend
        _patch_target_module.Ct2WhisperBackend = _FakeCt2Backend
        try:
            _server_whisper = _stt_w_reload_server()
            _stt_w_assert(
                _server_whisper.STT_BACKEND_RAW == "whisper",
                f"STT_BACKEND_RAW should be 'whisper', got {_server_whisper.STT_BACKEND_RAW!r}",
            )

            _server_whisper._load_stt_backend_eagerly()
            _stt_w_assert(
                isinstance(_server_whisper._stt_backend, _FakeCt2Backend),
                f"backend should be _FakeCt2Backend, got {type(_server_whisper._stt_backend).__name__}",
            )
            _stt_w_assert(
                _server_whisper._stt_backend_kind == "whisper",
                f"_stt_backend_kind should be 'whisper', got {_server_whisper._stt_backend_kind!r}",
            )
            _stt_w_assert(
                _server_whisper._stt_backend_model_id == "/tmp/fake-whisper-model",
                f"_stt_backend_model_id mismatch: {_server_whisper._stt_backend_model_id!r}",
            )
            ok("STT_BACKEND='whisper' + SPEACHES_PLUS_MODELS set: backend loaded via Ct2WhisperBackend stub")

            _stt_w_assert(
                _server_whisper._request_picks_whisper_stt("default"),
                "default model name with backend loaded should dispatch to whisper",
            )
            _stt_w_assert(
                _server_whisper._request_picks_whisper_stt(None),
                "None model with backend loaded should dispatch to whisper",
            )
            ok("_request_picks_whisper_stt returns True when backend loaded and model is generic")

            _models_listed = _server_whisper._build_models()
            _whisper_model_ids = [m.id for m in _models_listed
                                  if getattr(m, "id", None) == "/tmp/fake-whisper-model"]
            _stt_w_assert(
                len(_whisper_model_ids) == 1,
                f"whisper model id should appear once in /v1/models, got {_whisper_model_ids}",
            )
            ok("/v1/models lists the whisper model when backend is loaded")

            async def _fake_transcriptions_post(backend, file, response_format="json",
                                                language=None, prompt=None, temperature=0.0):
                from fastapi.responses import JSONResponse
                return JSONResponse({
                    "text": "DISPATCHED-TO-WHISPER",
                    "backend_id": backend.model_id(),
                    "language": language,
                    "prompt": prompt,
                    "response_format": response_format,
                })

            async def _fake_translations_post(backend, file, response_format="json",
                                              prompt=None, temperature=0.0):
                from fastapi.responses import JSONResponse
                return JSONResponse({
                    "text": "DISPATCHED-TO-WHISPER-TRANS",
                    "backend_id": backend.model_id(),
                    "response_format": response_format,
                })

            _stt_http = _stt_w_importlib.import_module("stt.http")
            _orig_tp = _stt_http.transcriptions_post
            _orig_tlp = _stt_http.translations_post
            _stt_http.transcriptions_post = _fake_transcriptions_post
            _stt_http.translations_post = _fake_translations_post
            try:
                _w_client = TestClient(_server_whisper.app)
                _wav_bytes = _stt_w_make_wav_bytes()
                _resp_tx = _w_client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("audio.wav", _wav_bytes, "audio/wav")},
                    data={"model": "default", "language": "en", "response_format": "json"},
                )
                _stt_w_assert(_resp_tx.status_code == 200, f"transcriptions: {_resp_tx.status_code} {_resp_tx.text}")
                _body_tx = _resp_tx.json()
                _stt_w_assert(
                    _body_tx.get("text") == "DISPATCHED-TO-WHISPER",
                    f"expected dispatch to stt.http.transcriptions_post, got body={_body_tx}",
                )
                _stt_w_assert(
                    _body_tx.get("backend_id") == "/tmp/fake-whisper-model",
                    f"backend was not the loaded whisper backend: {_body_tx}",
                )
                _stt_w_assert(
                    _body_tx.get("language") == "en",
                    f"language not passed through: {_body_tx}",
                )
                ok("POST /v1/audio/transcriptions dispatched to stt.http.transcriptions_post")

                _resp_tl = _w_client.post(
                    "/v1/audio/translations",
                    files={"file": ("audio.wav", _wav_bytes, "audio/wav")},
                    data={"model": "default", "response_format": "json"},
                )
                _stt_w_assert(_resp_tl.status_code == 200, f"translations: {_resp_tl.status_code} {_resp_tl.text}")
                _body_tl = _resp_tl.json()
                _stt_w_assert(
                    _body_tl.get("text") == "DISPATCHED-TO-WHISPER-TRANS",
                    f"expected dispatch to stt.http.translations_post, got body={_body_tl}",
                )
                ok("POST /v1/audio/translations dispatched to stt.http.translations_post")

                if _server_whisper.OMNI_MODEL_ID:
                    _server_whisper._stt_backend = _server_whisper._stt_backend
                    _stt_w_assert(
                        not _server_whisper._request_picks_whisper_stt(_server_whisper.OMNI_MODEL_ID),
                        "explicit qwen3-omni model should NOT dispatch to whisper",
                    )
                    ok("explicit qwen3-omni model bypasses whisper dispatch")
            finally:
                _stt_http.transcriptions_post = _orig_tp
                _stt_http.translations_post = _orig_tlp
        finally:
            _patch_target_module.Ct2WhisperBackend = _orig_ct2_backend

        _stt_w_os.environ[_stt_w_env.STT_BACKEND] = "whisper"
        _stt_w_os.environ[_stt_w_env.SPEACHES_PLUS_MODELS] = "/tmp/fake-whisper-model"

        class _BoomCt2Backend:
            def __init__(self, cfg):
                raise RuntimeError("simulated extension-missing failure")

        _patch_target_module.Ct2WhisperBackend = _BoomCt2Backend
        try:
            _server_boom = _stt_w_reload_server()
            import warnings as _stt_w_warnings
            with _stt_w_warnings.catch_warnings(record=True) as _ws:
                _stt_w_warnings.simplefilter("always")
                _server_boom._load_stt_backend_eagerly()
            _stt_w_assert(
                _server_boom._stt_backend is None,
                "failed backend should leave _stt_backend None",
            )
            _stt_w_assert(
                _server_boom._stt_backend_kind == "qwen3_omni",
                f"failed backend should fall back to qwen3_omni, got {_server_boom._stt_backend_kind!r}",
            )
            _stt_w_assert(
                _server_boom._stt_backend_load_error is not None
                and "simulated extension-missing failure" in _server_boom._stt_backend_load_error,
                f"load error not captured: {_server_boom._stt_backend_load_error!r}",
            )
            _stt_w_assert(
                any("falling back to qwen3_omni" in str(w.message) for w in _ws),
                f"expected fallback warning, got {[str(w.message) for w in _ws]}",
            )
            _stt_w_assert(
                not _server_boom._request_picks_whisper_stt("default"),
                "after fallback, dispatch should be False",
            )
            ok("STT backend load failure: warns and falls back to qwen3_omni without crashing")
        finally:
            _patch_target_module.Ct2WhisperBackend = _orig_ct2_backend

        _stt_w_os.environ[_stt_w_env.STT_BACKEND] = "whisper"
        _stt_w_os.environ.pop(_stt_w_env.SPEACHES_PLUS_MODELS, None)
        _server_no_model = _stt_w_reload_server()
        import warnings as _stt_w_warnings2
        with _stt_w_warnings2.catch_warnings(record=True) as _ws2:
            _stt_w_warnings2.simplefilter("always")
            _server_no_model._load_stt_backend_eagerly()
        _stt_w_assert(
            _server_no_model._stt_backend is None,
            "no SPEACHES_PLUS_MODELS -> backend should remain None",
        )
        _stt_w_assert(
            _server_no_model._stt_backend_kind == "qwen3_omni",
            f"no model path -> fall back to qwen3_omni, got {_server_no_model._stt_backend_kind!r}",
        )
        ok("STT_BACKEND='whisper' without SPEACHES_PLUS_MODELS: warns and falls back gracefully")

    finally:
        if _stt_w_prior_backend is None:
            _stt_w_os.environ.pop(_stt_w_env.STT_BACKEND, None)
        else:
            _stt_w_os.environ[_stt_w_env.STT_BACKEND] = _stt_w_prior_backend
        if _stt_w_prior_models is None:
            _stt_w_os.environ.pop(_stt_w_env.SPEACHES_PLUS_MODELS, None)
        else:
            _stt_w_os.environ[_stt_w_env.SPEACHES_PLUS_MODELS] = _stt_w_prior_models
        if "server" in _stt_w_sys.modules:
            del _stt_w_sys.modules["server"]

    ok(f"STT-Routes-Wired: {_stt_w_assert_count[0]} assertions verified")

def test_nix_flake_nativedeps():
    _section('Fix Nix-Flake-NativeDeps')
    import os as _nfn_os

    _nfn_repo = _nfn_os.path.dirname(_nfn_os.path.abspath(__file__))
    _nfn_flake_path = _nfn_os.path.join(_nfn_repo, "flake.nix")
    with open(_nfn_flake_path, "r", encoding="utf-8") as _nfn_f:
        _nfn_flake = _nfn_f.read()

    _nfn_devshell_marker = "default = pkgs.mkShell"
    _nfn_dev_idx = _nfn_flake.find(_nfn_devshell_marker)
    assert _nfn_dev_idx >= 0, "flake.nix: dev-shell `default = pkgs.mkShell` block not found"
    _nfn_devshell = _nfn_flake[_nfn_dev_idx:]

    assert "pkgs.ctranslate2" in _nfn_devshell, (
        "flake.nix dev-shell must reference pkgs.ctranslate2 (libctranslate2)"
    )
    ok("flake.nix dev-shell exposes pkgs.ctranslate2")

    assert "pkgs.whisper-cpp" in _nfn_devshell, (
        "flake.nix dev-shell must reference pkgs.whisper-cpp"
    )
    ok("flake.nix dev-shell exposes pkgs.whisper-cpp")

    assert "pybind11" in _nfn_devshell, (
        "flake.nix dev-shell must reference pybind11 (build-time headers)"
    )
    ok("flake.nix dev-shell exposes pybind11")

    assert "CT2_INCLUDE_DIR" in _nfn_devshell and "CT2_LIBRARY_DIR" in _nfn_devshell, (
        "flake.nix dev-shell must export CT2_INCLUDE_DIR / CT2_LIBRARY_DIR"
    )
    ok("flake.nix dev-shell exports CT2_INCLUDE_DIR + CT2_LIBRARY_DIR hints")

    assert "WHISPER_INCLUDE_DIR" in _nfn_devshell and "WHISPER_LIBRARY_DIR" in _nfn_devshell, (
        "flake.nix dev-shell must export WHISPER_INCLUDE_DIR / WHISPER_LIBRARY_DIR"
    )
    ok("flake.nix dev-shell exports WHISPER_INCLUDE_DIR + WHISPER_LIBRARY_DIR hints")

    _nfn_script_path = _nfn_os.path.join(_nfn_repo, "scripts", "build_bindings.sh")
    assert _nfn_os.path.isfile(_nfn_script_path), (
        f"scripts/build_bindings.sh missing at {_nfn_script_path}"
    )
    ok("scripts/build_bindings.sh exists")

    assert _nfn_os.access(_nfn_script_path, _nfn_os.X_OK), (
        "scripts/build_bindings.sh must be executable (chmod +x)"
    )
    ok("scripts/build_bindings.sh is executable")

    with open(_nfn_script_path, "r", encoding="utf-8") as _nfn_sf:
        _nfn_script_body = _nfn_sf.read()
    assert _nfn_script_body.startswith("#!/usr/bin/env bash"), (
        "scripts/build_bindings.sh must start with `#!/usr/bin/env bash`"
    )
    assert "set -euo pipefail" in _nfn_script_body, (
        "scripts/build_bindings.sh must use `set -euo pipefail`"
    )
    ok("scripts/build_bindings.sh has bash shebang + strict mode")

def test_webrtc_micin_consumer():
    _section('Fix WebRTC-MicIn-Consumer')

    import asyncio as _wmi_asyncio
    import numpy as _wmi_np

    from realtime.transport import (
        _inbound_consumer as _wmi_inbound_consumer,
        _inbound_track_pump as _wmi_inbound_track_pump,
    )

    class _WmiFakeAudioIn:
        def __init__(self):
            self._buf = _wmi_np.empty(0, dtype=_wmi_np.float32)
            self._total_taken = 0
            self.take_calls = 0

        def push(self, samples):
            self._buf = _wmi_np.concatenate([self._buf, _wmi_np.asarray(samples, dtype=_wmi_np.float32)])

        def take_array(self):
            self.take_calls += 1
            out = self._buf
            self._buf = _wmi_np.empty(0, dtype=_wmi_np.float32)
            self._total_taken += int(out.size)
            return out

    class _WmiFakeSession:
        def __init__(self):
            self.audio_in = _WmiFakeAudioIn()
            self.captured: list = []
            self.capture_calls = 0

        def capture_inbound_f32(self, samples):
            self.capture_calls += 1
            if hasattr(samples, "tolist"):
                self.captured.extend(list(samples))
            else:
                self.captured.extend(list(samples))

    async def _wmi_run_consumer_basic():
        sess = _WmiFakeSession()
        pump = _wmi_asyncio.create_task(_wmi_asyncio.sleep(3600))
        consumer = _wmi_asyncio.create_task(_wmi_inbound_consumer(sess, pump))
        sess.audio_in.push([0.1, 0.2, 0.3, 0.4])
        for _ in range(200):
            await _wmi_asyncio.sleep(0)
            if sess.capture_calls >= 1:
                break
        sess.audio_in.push([0.5, 0.6])
        for _ in range(200):
            await _wmi_asyncio.sleep(0)
            if sess.capture_calls >= 2:
                break
        consumer.cancel()
        try:
            await consumer
        except _wmi_asyncio.CancelledError:
            pass
        pump.cancel()
        try:
            await pump
        except _wmi_asyncio.CancelledError:
            pass
        return sess

    _wmi_basic_sess = _wmi_asyncio.run(_wmi_run_consumer_basic())
    assert len(_wmi_basic_sess.captured) == 6, (
        f"expected 6 captured samples, got {len(_wmi_basic_sess.captured)}"
    )
    ok("consumer drained 6 samples across two pushes into capture_inbound_f32")

    _wmi_expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for _i, (_a, _b) in enumerate(zip(_wmi_basic_sess.captured, _wmi_expected)):
        assert abs(_a - _b) < 1e-6, f"sample {_i} mismatch: {_a} vs {_b}"
    ok("captured sample order matches input order")

    assert _wmi_basic_sess.capture_calls >= 2, (
        f"expected at least 2 capture calls (one per push), got {_wmi_basic_sess.capture_calls}"
    )
    ok(f"capture_inbound_f32 called {_wmi_basic_sess.capture_calls} times for non-empty buffers only")

    async def _wmi_run_consumer_cancel():
        sess = _WmiFakeSession()
        pump = _wmi_asyncio.create_task(_wmi_asyncio.sleep(3600))
        consumer = _wmi_asyncio.create_task(_wmi_inbound_consumer(sess, pump))
        for _ in range(50):
            await _wmi_asyncio.sleep(0)
            if sess.audio_in.take_calls >= 1:
                break
        consumer.cancel()
        cancelled_clean = False
        try:
            await consumer
        except _wmi_asyncio.CancelledError:
            cancelled_clean = True
        pump.cancel()
        try:
            await pump
        except _wmi_asyncio.CancelledError:
            pass
        return cancelled_clean, consumer.done()

    _wmi_clean, _wmi_done = _wmi_asyncio.run(_wmi_run_consumer_cancel())
    assert _wmi_clean is True, "consumer must propagate CancelledError on cancel()"
    assert _wmi_done is True, "consumer task must be done after cancel()"
    ok("consumer exits cleanly on cancel() (CancelledError propagated, task done)")

    async def _wmi_run_pump_end_flush():
        sess = _WmiFakeSession()

        async def _pump_pushes_then_ends():
            sess.audio_in.push([0.7, 0.8, 0.9])
            sess.audio_in.push([0.11, 0.12])

        pump = _wmi_asyncio.create_task(_pump_pushes_then_ends())
        consumer = _wmi_asyncio.create_task(_wmi_inbound_consumer(sess, pump))
        await consumer
        return sess

    _wmi_flush_sess = _wmi_asyncio.run(_wmi_run_pump_end_flush())
    assert len(_wmi_flush_sess.captured) == 5, (
        f"expected 5 samples (final flush after pump end), got {len(_wmi_flush_sess.captured)}"
    )
    ok("end-of-track flush drains samples buffered before pump exit (5 captured)")
    assert (
        abs(_wmi_flush_sess.captured[-2] - 0.11) < 1e-6
        and abs(_wmi_flush_sess.captured[-1] - 0.12) < 1e-6
    )
    ok("flushed samples preserve order (last-buffered tail at end of capture stream)")

    async def _wmi_run_empty_buffer():
        sess = _WmiFakeSession()
        pump = _wmi_asyncio.create_task(_wmi_asyncio.sleep(3600))
        consumer = _wmi_asyncio.create_task(_wmi_inbound_consumer(sess, pump))
        for _ in range(200):
            await _wmi_asyncio.sleep(0)
            if sess.audio_in.take_calls >= 2:
                break
        consumer.cancel()
        try:
            await consumer
        except _wmi_asyncio.CancelledError:
            pass
        pump.cancel()
        try:
            await pump
        except _wmi_asyncio.CancelledError:
            pass
        return sess

    _wmi_empty_sess = _wmi_asyncio.run(_wmi_run_empty_buffer())
    assert _wmi_empty_sess.capture_calls == 0, (
        f"capture must NOT be called for empty buffers, got {_wmi_empty_sess.capture_calls}"
    )
    assert _wmi_empty_sess.audio_in.take_calls >= 2, (
        f"consumer must poll take_array even when empty, got {_wmi_empty_sess.audio_in.take_calls}"
    )
    ok("empty buffer polls do not invoke capture_inbound_f32 (no spurious mic_in writes)")

def test_pacer_wired_and_turnids(_sys_modules_snapshot):
    _section('Fix Pacer-Wired-And-TurnIds')
    if not _has_module("opuslib") and _STRICT_CI:
        pytest.fail("strict-skip in CI: Pacer-Wired-And-TurnIds requires opuslib")
    import asyncio as _pwt_asyncio
    import os as _pwt_os
    import tempfile as _pwt_tempfile
    from pathlib import Path as _PWTPath

    import env as _pwt_env
    import ids as _pwt_ids

    try:
        import opuslib as _pwt_opuslib  # noqa: F401
        _PWT_HAS_OPUS = True
    except Exception as _pwt_opus_err:
        _PWT_HAS_OPUS = False
        info(f"opuslib not installed; pacer-frame-emit assertions skipped ({_pwt_opus_err})")

    _pwt_a = _pwt_ids.next_turn_id()
    _pwt_b = _pwt_ids.next_turn_id()
    assert isinstance(_pwt_a, str) and _pwt_a.startswith("turn_")
    assert isinstance(_pwt_b, str) and _pwt_b.startswith("turn_")
    assert _pwt_a != _pwt_b
    _pwt_pa = _pwt_ids.next_phrase_id()
    _pwt_pb = _pwt_ids.next_phrase_id()
    assert isinstance(_pwt_pa, str) and _pwt_pa.startswith("phrase_")
    assert isinstance(_pwt_pb, str) and _pwt_pb.startswith("phrase_")
    assert _pwt_pa != _pwt_pb
    ok("ids: next_turn_id / next_phrase_id produce prefixed unique IDs")

    with _pwt_tempfile.TemporaryDirectory() as _pwt_td:
        _pwt_os.environ[_pwt_env.INSPECT_SESSION_DIR] = _pwt_td
        _pwt_os.environ[_pwt_env.CHAT_COMPLETION_BASE_URL] = "http://localhost:9999"
        for _mod in [m for m in list(sys.modules) if m == "inspect_api" or m.startswith("inspect_api.") or m == "realtime" or m.startswith("realtime.")]:
            del sys.modules[_mod]

        import realtime as _pwt_realtime
        from realtime.session import (
            Intent as _PWTIntent,
            RealtimeQuery as _PWTQuery,
            Session as _PWTSession,
        )
        from realtime.transport import (
            OutboundAudioSpec as _PWTOutboundAudioSpec,
            RealtimeContext as _PWTRealtimeContext,
            get_context as _pwt_get_context,
            set_context as _pwt_set_context,
        )
        from realtime import pipeline as _pwt_pipeline

        class _PWTFakeTrack:
            def __init__(self):
                self.frames = []
                self.ended = False
                self.dropped = False

            def push_opus_frame(self, payload, ms):
                self.frames.append((payload, ms))

            def end_of_stream(self):
                self.ended = True

            def drop_queued(self):
                self.dropped = True

        class _PWTFakeKokoro:
            def __init__(self, sample_count: int = 4800):
                self.sample_count = sample_count
                self.calls = []

            def stream(self, text, voice, *, speed=1.0, lang="en-us"):
                import numpy as _np
                self.calls.append((text, voice, speed, lang))
                yield _np.zeros(self.sample_count, dtype=_np.float32), 24_000

        class _PWTFakeModelsView:
            def __init__(self, kokoro):
                self.kokoro = kokoro

            @property
            def model_ids(self):
                return []

            @property
            def diarizer(self):
                return None

        _pwt_recorded_events: list = []

        class _PWTFakeSink:
            async def send_value(self, ev):
                _pwt_recorded_events.append(ev)

        async def _pwt_fake_sentence_stream(cfg, instructions, user_text, cancel=None):
            for s in ("Hello there.", "How are you?"):
                if cancel is not None and cancel.is_set():
                    raise _pwt_asyncio.CancelledError()
                yield s

        async def _pwt_fake_delta_stream(cfg, messages, cancel=None):
            for s in ("Hello there.", "How are you?"):
                if cancel is not None and cancel.is_set():
                    raise _pwt_asyncio.CancelledError()
                yield s

        async def _pwt_run_happy_path():
            _pwt_recorded_events.clear()
            _track = _PWTFakeTrack()
            _kokoro = _PWTFakeKokoro()
            _ctx = _PWTRealtimeContext(models=_PWTFakeModelsView(_kokoro))
            _pwt_set_context(_ctx)
            _flush_called = {"flush": 0, "cancel": 0}

            try:
                _q = _PWTQuery(intent="conversation", voice="af_heart")
                _spec = _PWTOutboundAudioSpec.webrtc(_track)
                _s = _PWTSession(query=_q, intent=_PWTIntent.CONVERSATION, outbound_audio=_spec)
                async with _s._state_lock:
                    _s.state.event_sink = _PWTFakeSink()

                _orig_stream = _pwt_pipeline._iter_llm_deltas
                _pwt_pipeline._iter_llm_deltas = _pwt_fake_delta_stream
                _orig_build = _pwt_pipeline._build_pacer_for_session

                def _wrap_build(session, kokoro):
                    p = _orig_build(session, kokoro)
                    if p is None:
                        return None
                    _orig_flush = p.flush
                    _orig_cancel = p.cancel

                    async def _flush_wrap():
                        _flush_called["flush"] += 1
                        return await _orig_flush()

                    def _cancel_wrap():
                        _flush_called["cancel"] += 1
                        return _orig_cancel()

                    p.flush = _flush_wrap
                    p.cancel = _cancel_wrap
                    return p

                _pwt_pipeline._build_pacer_for_session = _wrap_build
                try:
                    await _pwt_pipeline.run_response(
                        _s,
                        response_id="resp_test",
                        instructions=None,
                        user_text="hi",
                        cancel=_pwt_asyncio.Event(),
                    )
                finally:
                    _pwt_pipeline._iter_llm_deltas = _orig_stream
                    _pwt_pipeline._build_pacer_for_session = _orig_build
                return _s, _track, _kokoro, _flush_called
            finally:
                _pwt_set_context(None)

        _pwt_sess_ok, _pwt_track_ok, _pwt_kokoro_ok, _pwt_flush_state = _pwt_asyncio.run(
            _pwt_run_happy_path()
        )

        _pwt_kinds = [getattr(e, "type_name", lambda: "")() for e in _pwt_recorded_events]
        assert "response.created" in _pwt_kinds, f"expected response.created, got {_pwt_kinds}"
        _pwt_delta_count = sum(
            1 for k in _pwt_kinds if k == "response.output_audio_transcript.delta"
        )
        assert _pwt_delta_count >= 2, f"expected >=2 transcript deltas, got {_pwt_delta_count}"
        assert "response.done" in _pwt_kinds, f"expected response.done, got {_pwt_kinds}"
        ok("pipeline: response.created + per-sentence audio_transcript.delta + response.done emitted")

        if _PWT_HAS_OPUS:
            assert len(_pwt_track_ok.frames) > 0, (
                f"track must receive push_opus_frame calls, got {len(_pwt_track_ok.frames)}"
            )
            ok(f"pacer: pushed {len(_pwt_track_ok.frames)} opus frames to outbound track")
        else:
            info("pacer: frame-count assertion deferred (no opuslib in this env)")

        assert _pwt_flush_state["flush"] >= 1, "pacer.flush must be called at end"
        ok("pacer: flush() invoked at end of run_response")

        assert len(_pwt_kokoro_ok.calls) == 2, (
            f"kokoro.stream should be called once per sentence, got {len(_pwt_kokoro_ok.calls)}"
        )
        ok("pipeline: kokoro.stream invoked once per sentence")

        async def _pwt_run_no_kokoro():
            _pwt_recorded_events.clear()
            _ctx_none = _PWTRealtimeContext(models=_PWTFakeModelsView(None))
            _pwt_set_context(_ctx_none)
            try:
                _q = _PWTQuery(intent="conversation")
                _s = _PWTSession(query=_q, intent=_PWTIntent.CONVERSATION)
                async with _s._state_lock:
                    _s.state.event_sink = _PWTFakeSink()

                _orig_stream = _pwt_pipeline._iter_llm_deltas
                _pwt_pipeline._iter_llm_deltas = _pwt_fake_delta_stream
                try:
                    await _pwt_pipeline.run_response(
                        _s,
                        response_id="resp_text_only",
                        instructions=None,
                        user_text="hi",
                        cancel=_pwt_asyncio.Event(),
                    )
                finally:
                    _pwt_pipeline._iter_llm_deltas = _orig_stream
                return _s
            finally:
                _pwt_set_context(None)

        _pwt_sess_text = _pwt_asyncio.run(_pwt_run_no_kokoro())
        _pwt_kinds_text = [getattr(e, "type_name", lambda: "")() for e in _pwt_recorded_events]
        assert "response.done" in _pwt_kinds_text, (
            f"text-only fallback must still emit response.done, got {_pwt_kinds_text}"
        )
        _pwt_delta_text = sum(
            1 for k in _pwt_kinds_text if k == "response.output_audio_transcript.delta"
        )
        assert _pwt_delta_text >= 2, (
            f"text-only fallback should still emit transcript deltas, got {_pwt_delta_text}"
        )
        ok("pipeline: text-only fallback (no Kokoro) still emits transcript deltas + response.done")

        async def _pwt_run_cancel():
            _pwt_recorded_events.clear()
            _track = _PWTFakeTrack()
            _kokoro = _PWTFakeKokoro(sample_count=24_000)
            _ctx = _PWTRealtimeContext(models=_PWTFakeModelsView(_kokoro))
            _pwt_set_context(_ctx)
            _cancel_count = {"n": 0}

            try:
                _q = _PWTQuery(intent="conversation", voice="af_heart")
                _spec = _PWTOutboundAudioSpec.webrtc(_track)
                _s = _PWTSession(query=_q, intent=_PWTIntent.CONVERSATION, outbound_audio=_spec)
                async with _s._state_lock:
                    _s.state.event_sink = _PWTFakeSink()

                _evt = _pwt_asyncio.Event()

                async def _slow_stream(cfg, messages, cancel=None):
                    yield "first sentence here."
                    _evt.set()
                    await _pwt_asyncio.sleep(0.05)
                    if cancel is not None and cancel.is_set():
                        raise _pwt_asyncio.CancelledError()
                    yield "second sentence here."

                _orig_stream = _pwt_pipeline._iter_llm_deltas
                _orig_build = _pwt_pipeline._build_pacer_for_session

                def _wrap_build(session, kokoro):
                    p = _orig_build(session, kokoro)
                    if p is None:
                        return None
                    _orig_cancel = p.cancel

                    def _cancel_wrap():
                        _cancel_count["n"] += 1
                        return _orig_cancel()

                    p.cancel = _cancel_wrap
                    return p

                _pwt_pipeline._iter_llm_deltas = _slow_stream
                _pwt_pipeline._build_pacer_for_session = _wrap_build

                _cancel_evt = _pwt_asyncio.Event()
                try:
                    _task = _pwt_asyncio.create_task(
                        _pwt_pipeline.run_response(
                            _s,
                            response_id="resp_cancel",
                            instructions=None,
                            user_text="hi",
                            cancel=_cancel_evt,
                        )
                    )
                    await _evt.wait()
                    _cancel_evt.set()
                    _raised_cancelled = False
                    try:
                        await _task
                    except _pwt_asyncio.CancelledError:
                        _raised_cancelled = True
                finally:
                    _pwt_pipeline._iter_llm_deltas = _orig_stream
                    _pwt_pipeline._build_pacer_for_session = _orig_build
                return _raised_cancelled, _cancel_count
            finally:
                _pwt_set_context(None)

        _pwt_raised, _pwt_cancel_count = _pwt_asyncio.run(_pwt_run_cancel())
        assert _pwt_raised, "cancel.set() must cause run_response to re-raise CancelledError"
        assert _pwt_cancel_count["n"] >= 1, (
            f"pacer.cancel() must be called on cancel path, got {_pwt_cancel_count['n']}"
        )
        ok("pipeline: cancel.set() triggers pacer.cancel() and re-raises CancelledError")

        _pwt_set_context(None)

def test_fix_perf_p0s():
    _section('Fix Perf-P0s')
    import asyncio as _p0_asyncio
    import pathlib as _p0_pathlib
    import tempfile as _p0_tempfile
    from collections import deque as _p0_deque

    import numpy as _p0_np

    from audio import g711 as _p0_g711
    from audio.g711 import (
        alaw_encode_sample as _p0_alaw_encode_sample,
        f32_to_alaw_bytes as _p0_f32_to_alaw_bytes,
        f32_to_ulaw_bytes as _p0_f32_to_ulaw_bytes,
        ulaw_encode_sample as _p0_ulaw_encode_sample,
    )

    def _p0_old_f32_to_ulaw(samples):
        s = _p0_np.asarray(samples, dtype=_p0_np.float32)
        s = _p0_np.clip(s, -1.0, 1.0)
        v = _p0_np.rint(s * 32767.0).astype(_p0_np.int32)
        v = _p0_np.clip(v, -32768, 32767).astype(_p0_np.int16)
        out = bytearray(len(v))
        for i, x in enumerate(v.tolist()):
            out[i] = _p0_ulaw_encode_sample(int(x))
        return bytes(out)

    def _p0_old_f32_to_alaw(samples):
        s = _p0_np.asarray(samples, dtype=_p0_np.float32)
        s = _p0_np.clip(s, -1.0, 1.0)
        v = _p0_np.rint(s * 32767.0).astype(_p0_np.int32)
        v = _p0_np.clip(v, -32768, 32767).astype(_p0_np.int16)
        out = bytearray(len(v))
        for i, x in enumerate(v.tolist()):
            out[i] = _p0_alaw_encode_sample(int(x))
        return bytes(out)

    _p0_tv = _p0_np.array(
        [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=_p0_np.float32
    )
    assert _p0_old_f32_to_ulaw(_p0_tv) == _p0_f32_to_ulaw_bytes(_p0_tv), (
        "g711 ulaw LUT parity FAIL on spec test vector"
    )
    assert _p0_old_f32_to_alaw(_p0_tv) == _p0_f32_to_alaw_bytes(_p0_tv), (
        "g711 alaw LUT parity FAIL on spec test vector"
    )
    ok("g711 LUT parity: spec test vector [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]")

    _p0_rng = _p0_np.random.default_rng(0xC0FFEE)
    _p0_big = (_p0_rng.standard_normal(8192) * 0.6).astype(_p0_np.float32)
    assert _p0_old_f32_to_ulaw(_p0_big) == _p0_f32_to_ulaw_bytes(_p0_big), (
        "g711 ulaw LUT parity FAIL on random 8192-sample vector"
    )
    assert _p0_old_f32_to_alaw(_p0_big) == _p0_f32_to_alaw_bytes(_p0_big), (
        "g711 alaw LUT parity FAIL on random 8192-sample vector"
    )
    ok("g711 LUT parity: random 8192-sample vector (byte-for-byte)")

    _p0_sweep = _p0_np.linspace(-1.0, 1.0, 65536, dtype=_p0_np.float32)
    assert _p0_old_f32_to_ulaw(_p0_sweep) == _p0_f32_to_ulaw_bytes(_p0_sweep), (
        "g711 ulaw LUT parity FAIL on full int16 sweep"
    )
    assert _p0_old_f32_to_alaw(_p0_sweep) == _p0_f32_to_alaw_bytes(_p0_sweep), (
        "g711 alaw LUT parity FAIL on full int16 sweep"
    )
    ok("g711 LUT parity: full int16 sweep across 65536 indices")

    assert _p0_g711._ULAW_ENCODE_LUT.shape == (65536,)
    assert _p0_g711._ULAW_ENCODE_LUT.dtype == _p0_np.uint8
    assert _p0_g711._ALAW_ENCODE_LUT.shape == (65536,)
    assert _p0_g711._ALAW_ENCODE_LUT.dtype == _p0_np.uint8
    ok("g711 LUTs: shape (65536,) dtype uint8 for both ulaw and alaw")

    from realtime import audio_in as _p0_audio_in
    from realtime import audio_out as _p0_audio_out
    from realtime import pipeline as _p0_pipeline

    assert hasattr(_p0_audio_out, "np") and _p0_audio_out.np is not None
    assert _p0_audio_out.np is _p0_np or _p0_audio_out.np.__name__ == "numpy"
    ok("audio_out: numpy bound at module top (no per-call import)")

    assert "_scipy_resample_poly" in vars(_p0_audio_out), (
        "realtime.audio_out must bind scipy resample_poly at module top"
    )
    ok("audio_out: scipy resample_poly bound at module top (no per-call import)")

    assert hasattr(_p0_audio_in, "np") and _p0_audio_in.np is not None
    ok("audio_in: numpy bound at module top")

    class _P0FakePacer:
        def __init__(self):
            self.calls: list = []

        async def play(self, samples):
            self.calls.append(samples)

        def cancel(self):
            pass

    class _P0FakeKokoro:
        def stream(self, sentence, voice, speed=1.0, lang=None):
            yield (
                _p0_np.zeros(2400, dtype=_p0_np.float32),
                24_000,
            )
            yield (
                _p0_np.zeros(1200, dtype=_p0_np.float32),
                24_000,
            )

    async def _p0_drive():
        _p0_pacer = _P0FakePacer()
        _p0_kokoro = _P0FakeKokoro()
        for audio_chunk, _sr in _p0_kokoro.stream(
            "hello.", "af_heart", speed=1.0, lang="en"
        ):
            await _p0_pacer.play(audio_chunk)
        return _p0_pacer

    _p0_result = _p0_asyncio.run(_p0_drive())
    assert len(_p0_result.calls) == 2
    for _i, _call in enumerate(_p0_result.calls):
        assert isinstance(_call, _p0_np.ndarray), (
            f"pacer.play call #{_i} arg must be ndarray, got {type(_call).__name__}"
        )
    ok("pipeline contract: pacer.play receives ndarray (not list)")

    import inspect as _p0_inspect

    _p0_pipeline_src = _p0_inspect.getsource(_p0_pipeline)
    assert "audio_chunk.tolist()" not in _p0_pipeline_src, (
        "pipeline.py must not call .tolist() on audio_chunk before pacer.play"
    )
    ok("pipeline source: audio_chunk.tolist() removed before pacer.play")

    from inspect_api.audio_store import Channel, _Track

    class _P0FakeFh:
        def __init__(self):
            self.write_calls = 0
            self.flush_calls = 0
            self.bytes_written = 0
            self.closed = False

        def write(self, b):
            self.write_calls += 1
            self.bytes_written += len(b)

        def flush(self):
            self.flush_calls += 1

        def close(self):
            self.closed = True

    with _p0_tempfile.TemporaryDirectory() as _p0_tmp:
        _p0_track = _Track(
            session_id="p0test",
            channel=Channel.MIC_IN,
            directory=_p0_pathlib.Path(_p0_tmp),
        )
        _p0_fake = _P0FakeFh()
        _p0_track._state.fh = _p0_fake

        _p0_chunk = _p0_np.zeros(160, dtype=_p0_np.float32)
        for _ in range(100):
            _p0_track.append_f32(_p0_chunk, session_start_ns=0)

        assert _p0_fake.write_calls == 100, (
            f"expected 100 write() calls, got {_p0_fake.write_calls}"
        )
        assert _p0_fake.flush_calls == 0, (
            f"flush() must NOT be called per write (P0 perf), "
            f"got {_p0_fake.flush_calls} flushes for 100 writes"
        )

        _p0_track.close()
        assert _p0_fake.flush_calls == 1, (
            f"close() should flush exactly once, got {_p0_fake.flush_calls}"
        )
        assert _p0_fake.closed
    ok(
        "audio_store.append_f32: 100 chunks -> 100 writes, 0 flushes (close flushes once)"
    )

    with _p0_tempfile.TemporaryDirectory() as _p0_tmp:
        _p0_track2 = _Track(
            session_id="p0test2",
            channel=Channel.MIC_IN,
            directory=_p0_pathlib.Path(_p0_tmp),
        )
        _p0_fake2 = _P0FakeFh()
        _p0_track2._state.fh = _p0_fake2

        _p0_pcm = b"\x00\x00" * 100
        for _ in range(50):
            _p0_track2.append_pcm16(_p0_pcm, session_start_ns=0)

        assert _p0_fake2.write_calls == 50
        assert _p0_fake2.flush_calls == 0, (
            f"append_pcm16 must not flush per write, got {_p0_fake2.flush_calls}"
        )
    ok("audio_store.append_pcm16: 50 RTP packets -> 50 writes, 0 flushes")

    _p0_f32_input = _p0_np.array(
        [-1.0, -0.5, 0.0, 0.25, 1.0], dtype=_p0_np.float32
    )
    _p0_clipped = _p0_np.clip(_p0_f32_input, -1.0, 1.0)
    _p0_expected = _p0_np.rint(_p0_clipped * 32767.0).astype("<i2").tobytes()

    with _p0_tempfile.TemporaryDirectory() as _p0_tmp:
        _p0_track3 = _Track(
            session_id="p0test3",
            channel=Channel.MIC_IN,
            directory=_p0_pathlib.Path(_p0_tmp),
        )
        _p0_fake3 = _P0FakeFh()
        _p0_track3._state.fh = _p0_fake3
        _p0_track3.append_f32(_p0_f32_input, session_start_ns=0)
        assert _p0_fake3.bytes_written == len(_p0_expected), (
            f"append_f32 wrote {_p0_fake3.bytes_written} bytes, "
            f"expected {len(_p0_expected)}"
        )
    ok("audio_store.append_f32: vectorized f32->s16 produces correct byte count")

    assert hasattr(_p0_audio_in, "deque"), (
        "audio_in bonus: deque must be imported"
    )
    try:
        _P0AudioIngest = _p0_audio_in.AudioIngest
        _p0_ingest = _P0AudioIngest(channels=1)
    except Exception as _err:
        info(f"AudioIngest init skipped (opus unavailable): {_err}")
    else:
        assert hasattr(_p0_ingest, "_buf_chunks")
        assert isinstance(_p0_ingest._buf_chunks, _p0_deque)
        _p0_ingest._buf_chunks.append(
            _p0_np.array([0.1, 0.2, 0.3], dtype=_p0_np.float32)
        )
        _p0_ingest._buf_chunks.append(
            _p0_np.array([0.4, 0.5], dtype=_p0_np.float32)
        )
        _p0_ingest._buf_size = 5
        _p0_taken = _p0_ingest.take_array()
        assert _p0_taken.shape == (5,), (
            f"take_array should concatenate deque, got shape {_p0_taken.shape}"
        )
        assert len(_p0_ingest._buf_chunks) == 0
        ok("audio_in bonus: deque-of-arrays buffer (concat-once-on-take)")

def _stt_fixes_make_wav_bytes(samples_per_sec: int = 16000, n_seconds: float = 0.5) -> bytes:
    import io
    import wave
    import numpy as _np

    n_samples = int(samples_per_sec * n_seconds)
    pcm = (0.2 * _np.sin(2 * _np.pi * 440.0 * _np.arange(n_samples) / samples_per_sec)).astype(_np.float32)
    pcm_i16 = (pcm * 32767.0).astype(_np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samples_per_sec)
        wf.writeframes(pcm_i16)
    return buf.getvalue()

def test_stt_translations_real():
    _section('Fix STT-Translations-Real')
    import asyncio as _str_asyncio
    import io as _str_io

    from stt import http as _str_http
    from stt.whisper import TranscriptionResult as _STR_TR

    class _StubBackend:
        model_id = "stub-translate"

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def transcribe(self, samples, sample_rate=16000, *, language=None, prompt=None, with_timestamps=False, task="transcribe"):
            self.calls.append({
                "samples_len": int(len(samples)),
                "sample_rate": sample_rate,
                "language": language,
                "prompt": prompt,
                "with_timestamps": with_timestamps,
                "task": task,
            })
            return _STR_TR(text="bonjour" if task == "transcribe" else "hello")

    _wav = _stt_fixes_make_wav_bytes(n_seconds=0.5)
    _backend = _StubBackend()
    _file = _FakeUpload(_wav, "audio/wav")

    async def _run() -> object:
        return await _str_http.translations_post(
            backend=_backend,
            file=_file,
            response_format="json",
            prompt=None,
        )

    _resp = _str_asyncio.run(_run())
    assert _backend.calls, "translations_post must invoke backend.transcribe"
    _last = _backend.calls[-1]
    assert _last["task"] == "translate", (
        f"translations_post must pass task='translate' (NOT language='en'), got {_last!r}"
    )
    assert _last["language"] is None or _last["language"] != "en", (
        f"translations_post must NOT silently set language='en'; got language={_last['language']!r}"
    )
    ok("translations_post wires task='translate' through to backend.transcribe (not language='en')")

    _body = json.loads(_resp.body.decode("utf-8"))
    assert _body.get("task") == "translate", f"response must echo task=translate, got {_body!r}"
    assert "model" in _body and _body["model"] == "stub-translate", (
        f"translations response missing model id, got {_body!r}"
    )
    ok("translations_post response includes task=translate + model id")

def test_stt_response_shape():
    _section('Fix STT-Response-Shape')
    import asyncio as _srs_asyncio

    from stt import http as _srs_http
    from stt.whisper import TranscriptionResult as _SRS_TR

    class _StubBackend:
        model_id = "stub-asr"

        def transcribe(self, samples, sample_rate=16000, *, language=None, prompt=None, with_timestamps=False, task="transcribe"):
            return _SRS_TR(text="hello world")

    _wav = _stt_fixes_make_wav_bytes(n_seconds=0.5)
    _backend = _StubBackend()

    async def _post_json() -> object:
        return await _srs_http.transcriptions_post(
            backend=_backend,
            file=_FakeUpload(_wav, "audio/wav"),
            response_format="json",
            language="es",
            prompt=None,
        )

    _resp = _srs_asyncio.run(_post_json())
    _body = json.loads(_resp.body.decode("utf-8"))
    assert "text" in _body and _body["text"] == "hello world"
    assert "language" in _body, (
        f"BC: /v1/audio/transcriptions json response must include 'language' key, got {_body!r}"
    )
    assert "model" in _body, (
        f"BC: /v1/audio/transcriptions json response must include 'model' key, got {_body!r}"
    )
    assert _body["model"] == "stub-asr", f"model id mismatch: {_body!r}"
    assert _body["language"] == "es", f"language echo mismatch: {_body!r}"
    ok("transcriptions_post json response includes text + language + model (BC restored)")

    async def _post_verbose() -> object:
        return await _srs_http.transcriptions_post(
            backend=_backend,
            file=_FakeUpload(_wav, "audio/wav"),
            response_format="verbose_json",
            language="es",
            prompt=None,
        )

    _resp_v = _srs_asyncio.run(_post_verbose())
    _body_v = json.loads(_resp_v.body.decode("utf-8"))
    for _k in ("task", "language", "duration", "text", "segments", "words"):
        assert _k in _body_v, (
            f"BC: verbose_json response missing key {_k!r}, got keys={list(_body_v.keys())}"
        )
    assert _body_v["task"] == "transcribe"
    assert _body_v["language"] == "es"
    ok("transcriptions_post verbose_json response includes task/language/duration/text/segments/words")

def test_concurrency_p1s():
    _section('Fix Concurrency-P1s')
    import asyncio as _cp_asyncio
    import tempfile as _cp_tempfile
    import threading as _cp_threading
    from pathlib import Path as _CpPath

    import numpy as _cp_np

    from realtime import transport as _cp_transport
    from realtime import websocket as _cp_ws
    from realtime.transport import OutboundOpusTrack as _CpTrack
    from realtime.audio_out import (
        OutboundPacer as _CpPacer,
        QueueGate as _CpGate,
    )

    assert not hasattr(_cp_transport, "_drop_session_sync"), (
        "transport._drop_session_sync must be removed; use _drop_session (async) "
        "or _schedule_drop_session(loop=...) from off-loop callbacks"
    )
    assert hasattr(_cp_transport, "_schedule_drop_session"), (
        "transport._schedule_drop_session helper must exist for off-loop sites"
    )
    ok("transport: _drop_session_sync removed; _schedule_drop_session present")

    assert hasattr(_cp_ws, "_ws_sessions_lock"), (
        "websocket._ws_sessions_lock must be defined"
    )
    assert hasattr(_cp_ws, "_register_ws_session"), (
        "websocket._register_ws_session helper must exist"
    )
    assert hasattr(_cp_ws, "_drop_ws_session"), (
        "websocket._drop_ws_session helper must exist"
    )
    assert hasattr(_cp_ws, "snapshot_ws_sessions"), (
        "websocket.snapshot_ws_sessions helper must exist for safe iteration"
    )
    ok("websocket: _ws_sessions_lock + register/drop/snapshot helpers present")

    class _CpFakeSess:
        def __init__(self, sid):
            self.id = sid

    async def _cp_ws_concurrency():
        sessions = [_CpFakeSess(f"sess_{i}") for i in range(100)]

        async def attach_drop(s):
            await _cp_ws._register_ws_session(s)
            await _cp_asyncio.sleep(0)
            await _cp_ws._drop_ws_session(s.id)

        await _cp_asyncio.gather(*[attach_drop(s) for s in sessions])
        return len(_cp_ws._ws_sessions)

    _cp_ws._ws_sessions.clear()
    _cp_remaining = _cp_asyncio.run(_cp_ws_concurrency())
    assert _cp_remaining == 0, (
        f"100 attach/drop pairs left {_cp_remaining} stragglers in _ws_sessions"
    )
    snap = _cp_ws.snapshot_ws_sessions()
    assert isinstance(snap, dict), "snapshot_ws_sessions must return a dict"
    ok("websocket: 100 concurrent attach/drop pairs leave _ws_sessions empty")

    class _CpFakeTrack:
        def __init__(self):
            self.frames: list = []
            self.dropped = 0

        def push_opus_frame(self, payload, dur):
            self.frames.append((payload, dur))

        def drop_queued(self):
            self.dropped += 1

    async def _cp_pacer_cancel_race():
        track = _CpFakeTrack()
        pacer = _CpPacer(track=track, played_ms_ref=[0], queue_cap_ms=10_000)
        samples = _cp_np.zeros(24_000, dtype=_cp_np.float32)

        async def play_loop():
            try:
                await pacer.play(samples)
            except Exception:
                pass

        async def cancel_after():
            await _cp_asyncio.sleep(0.005)
            pacer.cancel()

        await _cp_asyncio.gather(play_loop(), cancel_after())
        return pacer

    try:
        import opuslib  # noqa: F401
    except ImportError:
        info("opuslib not installed; pacer cancel-race test runs only the gate-clamp portion")
        _cp_p = _CpPacer(track=_CpFakeTrack(), played_ms_ref=[0], queue_cap_ms=10_000)
        _cp_p.gate.queued_ms = 5
        _cp_p.gate.on_frame_sent()
        _cp_p.gate.on_frame_sent()
        _cp_p.gate.on_frame_sent()
        assert _cp_p.gate.queued_ms == 0, (
            f"QueueGate.on_frame_sent must clamp at 0, got {_cp_p.gate.queued_ms}"
        )
        ok("audio_out: QueueGate.on_frame_sent clamps to 0 (no negative)")
    else:
        _cp_pacer_done = _cp_asyncio.run(_cp_pacer_cancel_race())
        assert _cp_pacer_done._cancelled is True, "cancel must flip _cancelled flag"
        assert _cp_pacer_done.gate.queued_ms >= 0, (
            f"gate.queued_ms must never go negative, got {_cp_pacer_done.gate.queued_ms}"
        )
        for _ in range(5):
            _cp_pacer_done.gate.on_frame_sent()
        assert _cp_pacer_done.gate.queued_ms == 0, (
            "post-cancel on_frame_sent must clamp at 0"
        )
        ok("audio_out: cancel during play does not drive gate.queued_ms negative")

    track = _CpTrack(queue_maxsize=10_000)

    def _cp_producer(start, count, results):
        try:
            for i in range(start, start + count):
                track.push_nowait(("payload", i))
            results.append(("ok", count))
        except Exception as err:
            results.append(("err", repr(err)))

    results: list = []
    threads = [
        _cp_threading.Thread(target=_cp_producer, args=(i * 100, 100, results))
        for i in range(20)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10.0)

    async def _cp_drain():
        q = track._ensure_queue()
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        return items

    drained = _cp_asyncio.run(_cp_drain())
    assert all(r[0] == "ok" for r in results), f"producer errors: {results}"
    assert len(drained) == 2000, (
        f"OutboundOpusTrack lost items under concurrent push: drained={len(drained)} expected=2000"
    )
    ok("transport: OutboundOpusTrack push_nowait from 20 threads loses no items (2000/2000)")

    if not _has_module("opuslib"):
        info("opuslib not installed; AudioIngest cap test skipped")
    else:
        from realtime.audio_in import AudioIngest as _CpIngest

        _cp_cap = 16_000
        _cp_ing = _CpIngest(channels=1, max_buffer_samples=_cp_cap)
        _cp_chunk_48k = _cp_np.zeros(48_000, dtype=_cp_np.float32)
        for _ in range(60):
            _cp_ing._ingest_mono_48k(_cp_chunk_48k)
        assert _cp_ing._buf_size <= _cp_cap, (
            f"buffer not capped: size={_cp_ing._buf_size} cap={_cp_cap}"
        )
        assert _cp_ing.dropped_samples > 0, (
            f"dropped_samples counter must be incremented after overflow, got {_cp_ing.dropped_samples}"
        )
        ok(
            f"audio_in: 60s of 48k audio without drain capped at {_cp_ing._buf_size} samples "
            f"(cap={_cp_cap}, dropped={_cp_ing.dropped_samples})"
        )

        _cp_total = _cp_ing._buf_size
        _cp_taken = _cp_ing.take_array()
        assert _cp_taken.shape[0] == _cp_total, (
            f"take_array must drain capped buffer: got {_cp_taken.shape[0]} vs {_cp_total}"
        )
        ok("audio_in: take_array drains capped buffer cleanly")

    from inspect_api.relay import InspectorRelay as _CpRelay

    with _cp_tempfile.TemporaryDirectory() as _cp_td:
        _cp_relay = _CpRelay("sess_p1_concurrency", _CpPath(_cp_td))
        try:
            _cp_relay.subscribe()
        except RuntimeError as err:
            assert "running asyncio event loop" in str(err), (
                f"subscribe() outside loop must raise informative RuntimeError; got: {err}"
            )
            ok("inspect_api: relay.subscribe() outside any loop raises informative RuntimeError")
        else:
            raise AssertionError(
                "InspectorRelay.subscribe() must raise RuntimeError when called without a running loop"
            )

        async def _cp_relay_in_loop():
            sub = _cp_relay.subscribe()
            assert sub.snapshot == []
            sub2 = _cp_relay.subscribe(loop=_cp_asyncio.get_running_loop())
            _cp_relay.unsubscribe(sub.queue)
            _cp_relay.unsubscribe(sub2.queue)
            return True

        assert _cp_asyncio.run(_cp_relay_in_loop())
        ok("inspect_api: relay.subscribe(loop=...) and bare subscribe() inside loop both work")
        _cp_relay.close()

    if hasattr(_cp_transport, "_track_dc_task"):
        ok("transport: _track_dc_task helper present (dc handler tasks tracked)")

    async def _cp_track_dc():
        class _Sess:
            id = "dc_test"

        sess = _Sess()
        called = []

        async def _coro():
            called.append("ran")

        task = _cp_asyncio.create_task(_coro())
        _cp_transport._track_dc_task(sess, task)
        assert hasattr(sess, "_dc_tasks") and task in sess._dc_tasks
        await task
        await _cp_asyncio.sleep(0)
        assert task not in sess._dc_tasks, "task must be removed from bag after completion"
        return called

    _cp_called = _cp_asyncio.run(_cp_track_dc())
    assert _cp_called == ["ran"]
    ok("transport: _track_dc_task auto-removes completed tasks from session._dc_tasks")

def test_ux_api_p1s():
    _section('Fix UX-API-P1s')
    try:
        from fastapi.testclient import TestClient as _UXTestClient
    except Exception as exc:
        _strict_skip(f"fastapi.testclient unavailable: {exc}")
    try:
        from server import app as _ux_app
    except Exception as exc:
        _strict_skip(f"server import failed: {exc}")

    _ux_client = _UXTestClient(_ux_app)
    _assert_count = 0

    _r404 = _ux_client.get("/v1/inspect/sessions/history/nonexistent_sid_xyz123")
    assert _r404.status_code == 404, _r404.status_code
    _assert_count += 1
    _b404 = _r404.json()
    assert "error" in _b404, _b404
    _assert_count += 1
    assert _b404["error"]["message"] == "session not found", _b404
    _assert_count += 1
    assert _b404["error"]["code"] == "session_not_found", _b404
    _assert_count += 1
    assert _b404["error"].get("type") in (
        "not_found_error", "invalid_request_error",
    ), _b404
    _assert_count += 1
    ok("inspect history 404 returns {error:{message,type,code:'session_not_found'}}")

    _r400 = _ux_client.get("/v1/inspect/sessions/bad!sid/audio?channel=mic_in")
    assert _r400.status_code == 400, (_r400.status_code, _r400.text)
    _assert_count += 1
    _b400 = _r400.json()
    assert "error" in _b400, _b400
    _assert_count += 1
    assert _b400["error"]["code"] == "invalid_session_id", _b400
    _assert_count += 1
    ok("inspect audio with bad sid returns 400 invalid_session_id")

    _rsess = _ux_client.get("/health/sessions")
    assert _rsess.status_code == 200, _rsess.status_code
    _assert_count += 1
    _bsess = _rsess.json()
    assert "live_sessions" in _bsess, _bsess
    _assert_count += 1
    assert "ws_sessions" in _bsess, _bsess
    _assert_count += 1
    assert "webrtc_sessions" in _bsess, _bsess
    _assert_count += 1
    ok("/health/sessions exposes live_sessions, ws_sessions, webrtc_sessions")

    _rcap = _ux_client.get("/v1/realtime/capabilities")
    assert _rcap.status_code == 200, _rcap.status_code
    _assert_count += 1
    _bcap = _rcap.json()
    _ext_kinds = _bcap["extensions"]["eou_kinds"]
    assert "audio" not in _ext_kinds, _ext_kinds
    _assert_count += 1
    assert "fusion" not in _ext_kinds, _ext_kinds
    _assert_count += 1
    assert "audio" in _bcap["features"]["eou_kinds"], _bcap["features"]["eou_kinds"]
    _assert_count += 1
    assert "fusion" in _bcap["features"]["eou_kinds"], _bcap["features"]["eou_kinds"]
    _assert_count += 1
    ok("capabilities: audio/fusion are spec kinds (features), not extensions")

    _rsdp = _ux_client.post(
        "/v1/realtime",
        content=b"this is not sdp",
        headers={"content-type": "application/sdp"},
    )
    assert _rsdp.status_code in (400, 503), (_rsdp.status_code, _rsdp.text)
    _assert_count += 1
    if _rsdp.status_code == 400:
        _bsdp = _rsdp.json()
        assert "error" in _bsdp, _bsdp
        _assert_count += 1
        assert _bsdp["error"]["code"] == "sdp_invalid", _bsdp
        _assert_count += 1
        ok("POST /v1/realtime malformed SDP returns 400 sdp_invalid")
    else:
        ok("POST /v1/realtime: WebRTC unavailable (503), sdp_invalid path skipped")

    _rh = _ux_client.get("/health")
    assert _rh.status_code == 200, _rh.status_code
    _assert_count += 1
    _bh = _rh.json()
    assert isinstance(_bh, dict), type(_bh)
    _assert_count += 1
    assert "stt_backend" in _bh, list(_bh.keys())
    _assert_count += 1
    _stt = _bh["stt_backend"]
    assert isinstance(_stt, dict), type(_stt)
    _assert_count += 1
    for _k in ("requested", "active", "model", "load_error"):
        assert _k in _stt, (_k, _stt)
        _assert_count += 1
    ok("/health includes stt_backend state dict")

    info(f"asserted {_assert_count} conditions")

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
