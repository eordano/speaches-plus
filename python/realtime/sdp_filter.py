from __future__ import annotations

def normalize_offer(sdp: str) -> str:
    canonical_ufrag: str | None = None
    canonical_pwd: str | None = None
    out: list[str] = []

    lines = sdp.splitlines(keepends=True)
    for line in lines:
        trimmed = line.rstrip("\r\n")
        if trimmed.startswith("a=ice-ufrag:"):
            rest = trimmed[len("a=ice-ufrag:") :]
            if canonical_ufrag is None:
                canonical_ufrag = rest
                out.append(line)
            else:
                out.append("a=ice-ufrag:" + canonical_ufrag + _line_terminator(line))
        elif trimmed.startswith("a=ice-pwd:"):
            rest = trimmed[len("a=ice-pwd:") :]
            if canonical_pwd is None:
                canonical_pwd = rest
                out.append(line)
            else:
                out.append("a=ice-pwd:" + canonical_pwd + _line_terminator(line))
        else:
            out.append(line)

    return _ensure_rtcp_mux("".join(out))

def _ensure_rtcp_mux(sdp: str) -> str:
    """Force-add `a=rtcp-mux` to every audio/video media section that lacks it.

    aiortc's RTCPeerConnection raises `ValueError("RTCP mux is not enabled")`
    when answering an offer whose media section omits `a=rtcp-mux`. Reference
    impls (rust/go) tolerate the omission by muxing implicitly. We bring nano
    in line by inserting the attribute right after the matching `m=` line so
    aiortc accepts the offer.
    """
    lines = sdp.splitlines(keepends=True)
    if not lines:
        return sdp

    sections: list[list[str]] = []
    current: list[str] = []
    media_kinds: list[str | None] = []
    current_kind: str | None = None
    for line in lines:
        trimmed = line.rstrip("\r\n")
        if trimmed.startswith("m="):
            if current:
                sections.append(current)
                media_kinds.append(current_kind)
            current = [line]
            parts = trimmed[2:].split(" ", 1)
            current_kind = parts[0] if parts else None
        else:
            current.append(line)
    if current:
        sections.append(current)
        media_kinds.append(current_kind)

    out: list[str] = []
    for sec, kind in zip(sections, media_kinds):
        if kind not in ("audio", "video"):
            out.extend(sec)
            continue
        has_mux = any(s.rstrip("\r\n") == "a=rtcp-mux" for s in sec)
        if has_mux:
            out.extend(sec)
            continue
        m_line = sec[0]
        term = _line_terminator(m_line) or "\r\n"
        out.append(m_line)
        out.append("a=rtcp-mux" + term)
        out.extend(sec[1:])
    return "".join(out)

def _line_terminator(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    return ""
