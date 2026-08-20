# 071-ocr-layout — full-page OCR layout fixture family

Five rendered full pages (invoice, newspaper, report, letter, lab notes) that
exercise page-level layout: multi-column reading order, ruled tables, header
blocks. The single-artifact classical band is `070-ocr`; see
`README-070-ocr.md`.

Fixtures live directly under `conformance/fixtures/` as
`071-ocr-layout-<name>/` like every other family. Each dir holds
`fixture.json`, `input.png` and a `README.md`; PNGs are committed so tests are
hermetic.

## Manifest shape

Same top-level head as every other family: `name`, `family`, `description`,
`comparison_strategy` (`cer_markup_neutral+layout_gates`),
`skip_when_no_model`. Family-specific payload keys: `input`, `expected_text`,
`render`, `gates`, `oracle`, `notes`, `scoring`.

`scoring` names the two numbers: `raw` is character edit distance against
`expected_text` verbatim; `neutral` is the same distance after
`nv-ocr tests/markup_neutral.rs::markup_neutral` (markup stripped, whitespace
collapsed). **The neutral number is the gating one.**

`skip_when_no_model: true` covers both backends: classical gates need the
tesseract `eng.traineddata` cache (`NV_OCR_TESSDATA`), model gates need
DeepSeek-OCR weights.

## Cases

| fixture | layout | neutral CER gate (model / classical) |
|---|---|---|
| 071-ocr-layout-invoice | header blocks, ruled item table, totals column | 0.02 / 0.18 |
| 071-ocr-layout-newspaper | two columns, masthead, headlines | 0.02 / 0.65 |
| 071-ocr-layout-report | single-column numbered report, bullet metrics | 0.02 / 0.01 |
| 071-ocr-layout-letter | address block, salutation, body | 0.02 / 0.01 |
| 071-ocr-layout-labnotes | heading, data table, observations | 0.02 / 0.02 |

Raw (non-neutral) gates are recorded in each `fixture.json::gates` as
`cer_max_model` / `cer_max_classical`; they are looser by design because a
model that emits markdown scores badly on raw CER while being correct.

## Consumers

- `rust/crates/nv-ocr/tests/e2e_fixtures.rs::e2e_layout_fixtures_classical_gates`
  — classical gates.
- `rust/crates/nv-ocr/tests/markup_neutral.rs` — the neutralizer, plus an
  `#[ignore]`d dump helper for differential runs.
- `rust/crates/nv-models/tests/deepseek_ocr_graph*.rs`,
  `deepseek_ocr_pipeline.rs` — model gates.

## Regenerating

`conformance/tools/generate-071-ocr-layout-fixtures.py` renders the five pages
plus their ground-truth text into `OUT` (default: a `rendered/` scratch dir
beside the script); the fixture directories are assembled from that output.
Not a test-time dependency.
