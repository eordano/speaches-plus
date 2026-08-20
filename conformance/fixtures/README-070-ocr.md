# 070-ocr — classical OCR fixture family

Single-artifact classical OCR cases: a clean line, a paragraph, skewed pages,
noise/contrast variants, and two real 1892 scan crops. Rendered-then-committed
PNGs with exact ground truth by construction; the real-scan crops are
transcribed by hand.

Fixtures live directly under `conformance/fixtures/` as
`070-ocr-<name>/` like every other family — there is no `ocr/` subtree. The
companion full-page band is `071-ocr-layout`; see
`README-071-ocr-layout.md`.

Each fixture dir holds `fixture.json` (declarative gates, runner-agnostic),
`input.png`, and a `README.md`. PNGs are committed so tests are hermetic.

## Manifest shape

Same top-level head as every other family: `name`, `family`, `description`,
`comparison_strategy` (`cer+layout_gates`), `skip_when_no_model`.
Family-specific payload keys: `input`, `expected_text`, `render` (or `source`
for the real scans), `gates`, `oracle`.

`skip_when_no_model: true` here means the tesseract `eng.traineddata` cache.
(The old `skip_when_no_traineddata` key is gone; nothing read it — the
requirement is stated in each `description`.)

## Cases

| fixture | gates |
|---|---|
| 070-ocr-clean-line | CER = 0 (tessdata_best), 1 line, 4 words |
| 070-ocr-paragraph | CER <= 0.5%, 6 lines, reading order exact |
| 070-ocr-skew-2deg | deskew estimate within 0.2 deg, CER <= 1% |
| 070-ocr-skew-12deg | deskew estimate within 0.5 deg, 6 lines, CER <= 2% |
| 070-ocr-lowcontrast-gradient | CER <= 2%, Otsu strictly worse than Sauvola |
| 070-ocr-noise-gauss | CER <= 2% (Gaussian sigma ~12) |
| 070-ocr-small-font | CER <= 3% (~16 px cap height) |
| 070-ocr-multiword-boxes | word rect IoU >= 0.6 vs measured render extents |
| 070-ocr-photo-noise-surround | >= 4 of 6 lines, per-line CER <= 0.1 |
| 070-ocr-realscan-1892 | real scan crop (not rendered): 4 lines, CER <= 2% |
| 070-ocr-sparse-illustration-1892 | both caption lines, per-line CER <= 0.1 |

CER = Levenshtein / truth length after NFC + whitespace-squash normalization.

## Consumers

- `rust/crates/nv-ocr/tests/layout.rs` — layout-level gates (line/word counts,
  deskew angle, word-box IoU). Runs without model assets.
- `rust/crates/nv-ocr/tests/e2e_fixtures.rs` — CER gates; requires the
  eng.traineddata cache (`NV_OCR_TESSDATA`), skips loudly otherwise.
- `rust/crates/nv-ocr/tests/oracle_parity.rs` — tesseract CLI comparison,
  env-gated on `NV_OCR_ORACLE=1` + binary + cache.
- `rust/tests/ocr_endpoint.rs`, `rust/tests/ocr_wgpu_grace.rs` — server-side
  smoke paths.

## Regenerating

`conformance/tools/generate-070-ocr-fixtures.sh` rerenders the PNGs and
rewrites each `fixture.json` + `README.md`. It writes into
`conformance/fixtures/` by default (`OUT` overrides) and uses the nix-store
imagemagick and DejaVuSans.ttf pinned in the script (`MAGICK` / `FONT`
overrides). Neither real-scan fixture is produced by it. It is not a test-time
dependency.
