# 071-ocr-layout-report

Single-column numbered report with bullet metrics. The easiest layout in the
band for classical OCR (cer_max_classical 0.01); it exists as the control
against which the harder pages are read.

Rendered by `conformance/tools/generate-071-ocr-layout-fixtures.py` (PIL,
DejaVu fonts, 1400x1900); the PNG is committed so tests are hermetic.

Gates live in `fixture.json`: `cer_max_classical*` for the tesseract path,
`cer_max_model*` for the DeepSeek-OCR path. The `*_neutral` numbers are the
gating ones — they compare after
`nv-ocr tests/e2e_fixtures.rs::markup_neutral` (markup stripped, whitespace
collapsed).

Consumed by `rust/crates/nv-ocr/tests/e2e_fixtures.rs` (classical gates).
`rust/crates/nv-models/tests/deepseek_ocr_graph_batch.rs` walks the same list but
asserts graphed==eager token identity, so `cer_max_model*` has no runner.
