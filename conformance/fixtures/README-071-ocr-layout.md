# 071-ocr-layout — full-page OCR layout fixture family

Five rendered full pages (invoice, newspaper, report, letter, lab notes) that
exercise page-level layout: multi-column reading order, ruled tables, header
blocks. The single-artifact classical band is `070-ocr`; see
`README-070-ocr.md`.

Fixtures live directly under `conformance/fixtures/` as
`071-ocr-layout-<name>/` like every other family. Each dir holds
`fixture.json`, `input.png` and a `README.md`; PNGs are gitignored and rebuilt
by `conformance/tools/regenerate-fixtures.sh`.

## Manifest shape

Same top-level head as every other family: `name`, `family`, `description`,
`comparison_strategy` (`cer_markup_neutral+layout_gates`),
`skip_when_no_model`. Family-specific payload keys: `input`, `expected_text`,
`render`, `gates`, `oracle`, `notes`, `scoring`.

`scoring` names the two numbers: `raw` is character edit distance against
`expected_text` verbatim; `neutral` is the same distance after
`nv-ocr tests/e2e_fixtures.rs::markup_neutral` (markup stripped, whitespace
collapsed). **The neutral number is the gating one.**

`skip_when_no_model: true` covers both backends: classical gates need the
tesseract `eng.traineddata` cache (`NV_OCR_TESSDATA`), model gates need
DeepSeek-OCR weights.

## Cases

| fixture | layout | neutral CER gate (model / classical) |
|---|---|---|
| 071-ocr-layout-invoice | header blocks, ruled item table, totals column | 0.02 / 0.18 |
| 071-ocr-layout-newspaper | two columns, masthead, headlines | 0.02 / 0.03 |
| 071-ocr-layout-report | single-column numbered report, bullet metrics | 0.02 / 0.01 |
| 071-ocr-layout-letter | address block, salutation, body | 0.02 / 0.01 |
| 071-ocr-layout-labnotes | heading, data table, observations | 0.02 / 0.02 |

Raw (non-neutral) gates are recorded in each `fixture.json::gates` as
`cer_max_model` / `cer_max_classical`; they are looser by design because a
model that emits markdown scores badly on raw CER while being correct.

## Consumers

- `rust/crates/nv-ocr/tests/e2e_fixtures.rs` — the classical gates
  (`e2e_layout_fixtures_classical_gates`), the neutralizer (`markup_neutral`),
  the gate-shape check that runs with no assets at all
  (`every_071_fixture_declares_all_four_cer_gates`), and an `#[ignore]`d
  re-measure table (`report_layout_fixture_cers`).
- `rust/crates/nv-models/tests/deepseek_ocr_graph_batch.rs`,
  `dots_ocr_bench.rs`, `got_ocr_real.rs` — these walk the same five-fixture list
  but assert graphed==eager token identity or measure speed.
- `scripts/ocr_fixture_cer.py` — the `cer_max_model` /
  `cer_max_model_neutral` runner: scores any generative backend of a live
  `/v1/ocr` against `expected_text` and fails on a neutral gate.

## Regenerating

`conformance/tools/generate-071-ocr-layout-fixtures.py` renders the five pages
plus their ground-truth text into `OUT` (default: a gitignored `rendered/`
scratch dir beside the script; `FONTS` names the DejaVu directory, found via
`fc-list` when unset). `conformance/tools/regenerate-fixtures.sh` copies each
`render.case` PNG into its fixture directory. Not a test-time dependency.
