from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diarization import Diarizer

    from .session import Session

log = logging.getLogger("realtime.diarization")

async def run_diarization(
    session: "Session",
    diarizer: "Diarizer | None",
    item_id: str,
    audio: list[float],
    audio_end_ms: int,
) -> None:
    if diarizer is None:
        return

    sr = 16_000
    utt_start_ms = max(0, audio_end_ms - (len(audio) * 1000) // sr)
    t0 = time.monotonic()
    try:
        try:
            import numpy as np

            arr = np.asarray(audio, dtype=np.float32)
        except ImportError:
            arr = audio
        segments = diarizer.diarize_utterance(arr, utt_start_ms)
    except Exception as err:
        log.warning("diarization failed: %s (item_id=%s)", err, item_id)
        return
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if not segments:
        return

    sink = await session.event_sink()
    if sink is None:
        return
    payload: dict[str, Any] = {
        "type": "conversation.item.diarization",
        "item_id": item_id,
        "audio_end_ms": int(audio_end_ms),
        "elapsed_ms": elapsed_ms,
        "segments": [
            {
                "speaker": f"SPEAKER_{int(s.speaker):02d}",
                "start": s.t_start_ms / 1000.0,
                "end": s.t_end_ms / 1000.0,
                "confidence": float(s.confidence),
            }
            for s in segments
        ],
    }
    await sink.send_value(payload)
