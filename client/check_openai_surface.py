#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx>=0.28",
# ]
# ///
"""
Probe a target server against the speaches OpenAPI surface and report
which endpoints are implemented, missing, or respond unexpectedly.

The "source" is the canonical surface (speaches itself). The "target" is
the implementation under test (our Go or Rust server). For each path /
method declared by the source's OpenAPI document we probe the target and
classify the response.

This is a structural check -- we do not validate response *bodies* against
the source schema, only that the route is wired up and the target's
status code is consistent with "the route exists." /v1/realtime has its
own conformance harness elsewhere; it is skipped by default.

Usage
-----
    ./client/check_openai_surface.py \\
        --source https://speaches.example.com \\
        --target http://localhost:8000

    ./client/check_openai_surface.py \\
        --source http://localhost:1327 \\
        --target http://localhost:8765 \\
        --output json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Any
from urllib.parse import urljoin

import httpx

DEFAULT_SOURCE = "http://localhost:1327"
DEFAULT_TARGET = "http://localhost:8000"
DEFAULT_OPENAPI_PATH = "/openapi.json"

PLACEHOLDER = "__compliance_probe__"

PRESENT_STATUSES = {
    200, 201, 202, 204, 206,
    400, 401, 403, 409, 413, 415,
    422,
    426,
    429,
    500, 502, 503, 504,
}
MISSING_STATUSES = {404}
WRONG_METHOD_STATUSES = {405}

@dataclass
class ProbeResult:
    path: str
    method: str
    target_url: str
    status: int | None
    verdict: str
    detail: str = ""

def build_target_url(target_base: str, path_template: str) -> str:
    concrete = path_template
    while "{" in concrete and "}" in concrete:
        start = concrete.index("{")
        end = concrete.index("}", start)
        concrete = concrete[:start] + PLACEHOLDER + concrete[end + 1 :]
    return target_base.rstrip("/") + concrete

def classify(status: int, documented_responses: set[int] | None = None) -> str:
    if status in PRESENT_STATUSES:
        return "present"
    if status in MISSING_STATUSES:
        if documented_responses and 404 in documented_responses:
            return "present"
        return "missing"
    if status in WRONG_METHOD_STATUSES:
        return "wrong-method"
    return "unexpected"

def documented_status_codes(spec: dict[str, Any], path: str, method: str) -> set[int]:
    op = (((spec.get("paths") or {}).get(path) or {}).get(method.lower()) or {})
    out: set[int] = set()
    for code in (op.get("responses") or {}):
        try:
            out.add(int(code))
        except (TypeError, ValueError):
            pass
    return out

def probe(
    client: httpx.Client,
    target_base: str,
    path: str,
    method: str,
    documented: set[int],
) -> ProbeResult:
    url = build_target_url(target_base, path)
    method_up = method.upper()

    if method_up in ("DELETE", "PUT", "PATCH"):
        try:
            resp = client.options(url, timeout=5.0)
        except httpx.RequestError as exc:
            return ProbeResult(path, method_up, url, None, "error", str(exc))
        if resp.status_code in (200, 204):
            verdict = "present"
        elif resp.status_code == 405:
            verdict = "present"
        elif resp.status_code == 404:
            verdict = "inconclusive"
        else:
            verdict = classify(resp.status_code)
        detail = f"OPTIONS returned {resp.status_code}; not exercising {method_up} (destructive)"
        return ProbeResult(path, method_up, url, resp.status_code, verdict, detail)

    try:
        if method_up == "GET":
            resp = client.get(url, timeout=10.0)
        elif method_up == "POST":
            resp = client.post(url, timeout=10.0)
        elif method_up == "HEAD":
            resp = client.head(url, timeout=5.0)
        elif method_up == "OPTIONS":
            resp = client.options(url, timeout=5.0)
        else:
            return ProbeResult(path, method_up, url, None, "skipped", "unsupported method")
    except httpx.RequestError as exc:
        return ProbeResult(path, method_up, url, None, "error", str(exc))

    detail = ""
    if resp.status_code == 404 and 404 in documented:
        detail = "404 documented as a response -- likely entity-not-found, not route-missing"
    return ProbeResult(path, method_up, url, resp.status_code, classify(resp.status_code, documented), detail)

def fetch_openapi(client: httpx.Client, source_base: str, openapi_path: str) -> dict[str, Any]:
    url = urljoin(source_base.rstrip("/") + "/", openapi_path.lstrip("/"))
    resp = client.get(url, timeout=15.0)
    resp.raise_for_status()
    return resp.json()

def collect_endpoints(spec: dict[str, Any], skip_paths: set[str]) -> list[tuple[str, str]]:
    endpoints: list[tuple[str, str]] = []
    for path, ops in (spec.get("paths") or {}).items():
        if any(path == s or path.startswith(s + "/") for s in skip_paths):
            continue
        for method in ops:
            if method.lower() in ("get", "post", "put", "patch", "delete", "head", "options"):
                endpoints.append((path, method.lower()))
    endpoints.sort()
    return endpoints

def render_text(
    source_base: str, target_base: str, results: list[ProbeResult], skipped_paths: list[str]
) -> str:
    out: list[str] = []
    out.append(f"source : {source_base}")
    out.append(f"target : {target_base}")
    out.append("")

    counts: dict[str, int] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1

    width_method = max((len(r.method) for r in results), default=6)
    width_path = max((len(r.path) for r in results), default=6)

    header = f"  {'METHOD':<{width_method}}  {'PATH':<{width_path}}  STATUS  VERDICT"
    out.append(header)
    out.append("  " + "-" * (len(header) - 2))
    for r in sorted(results, key=lambda r: (r.path, r.method)):
        status = "" if r.status is None else str(r.status)
        line = f"  {r.method:<{width_method}}  {r.path:<{width_path}}  {status:>6}  {r.verdict}"
        if r.detail:
            line += f"   ({r.detail})"
        out.append(line)

    out.append("")
    out.append("summary:")
    for verdict in ("present", "missing", "wrong-method", "unexpected", "inconclusive", "error", "skipped"):
        if counts.get(verdict, 0):
            out.append(f"  {verdict:<13} {counts[verdict]}")
    out.append(f"  total         {len(results)}")

    if skipped_paths:
        out.append("")
        out.append(f"skipped paths (use --include-realtime to include): {', '.join(skipped_paths)}")

    return "\n".join(out)

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help=f"OpenAPI source server (default: {DEFAULT_SOURCE})")
    p.add_argument("--target", default=DEFAULT_TARGET,
                   help=f"target server under test (default: {DEFAULT_TARGET})")
    p.add_argument("--openapi-path", default=DEFAULT_OPENAPI_PATH,
                   help="path on --source serving OpenAPI JSON (default: /openapi.json)")
    p.add_argument("--include-realtime", action="store_true",
                   help="include /v1/realtime* in the probe (default: skipped -- covered by client/test_e2e_*)")
    p.add_argument("--include-inspect", action="store_true",
                   help="include /v1/inspect* in the probe (default: skipped -- these are diagnostic, not OpenAI-compat)")
    p.add_argument("--output", choices=("text", "json"), default="text")
    p.add_argument("--fail-on", choices=("missing", "any", "never"), default="missing",
                   help="exit non-zero when results contain this class (default: missing)")
    args = p.parse_args()

    skip_paths: set[str] = set()
    if not args.include_realtime:
        skip_paths.add("/v1/realtime")
    if not args.include_inspect:
        skip_paths.add("/v1/inspect")

    with httpx.Client(follow_redirects=True) as client:
        try:
            spec = fetch_openapi(client, args.source, args.openapi_path)
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            print(f"error: could not fetch OpenAPI from {args.source}{args.openapi_path}: {exc}",
                  file=sys.stderr)
            return 2

        endpoints = collect_endpoints(spec, skip_paths)
        results = [
            probe(client, args.target, path, method, documented_status_codes(spec, path, method))
            for path, method in endpoints
        ]

    if args.output == "json":
        payload = {
            "source": args.source,
            "target": args.target,
            "skipped_paths": sorted(skip_paths),
            "results": [asdict(r) for r in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(args.source, args.target, results, sorted(skip_paths)))

    if args.fail_on == "never":
        return 0
    if args.fail_on == "any":
        bad = {"missing", "wrong-method", "unexpected", "error"}
    else:
        bad = {"missing"}
    return 1 if any(r.verdict in bad for r in results) else 0

if __name__ == "__main__":
    sys.exit(main())
