import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from client.eou_lib.heuristic import heuristic_score

FIXTURE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "conformance"
    / "fixtures"
    / "060-eou-heuristic-parity"
    / "fixture.json"
)

def test_heuristic_matches_shared_parity_corpus():
    assert FIXTURE.is_file(), f"shared corpus is required, not optional: {FIXTURE}"
    fx = json.loads(FIXTURE.read_text())
    cases = fx["cases"]
    assert cases, "fixture has no cases"

    bad = []
    for c in cases:
        got = heuristic_score(c["text"])
        if abs(got - c["score"]) > 1e-6:
            bad.append(f"{c['text']!r} ({c['branch']}): py={got} want={c['score']}")
    assert not bad, "python drifted from the canonical Rust scores:\n  " + "\n  ".join(bad)

if __name__ == "__main__":
    test_heuristic_matches_shared_parity_corpus()
    print(f"ok: {len(json.loads(FIXTURE.read_text())['cases'])} cases matched")
