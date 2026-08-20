# 071-ocr-layout-invoice

Invoice page: header blocks, a ruled item table and a totals column. The
ruled table is what separates markup-neutral scoring from raw scoring — a
model that emits a markdown table scores badly on raw CER and well after
neutralization.

Rendered by `conformance/tools/generate-071-ocr-layout-fixtures.py` (PIL,
DejaVu fonts, 1400x1900); the PNG is committed so tests are hermetic.

Gates live in `fixture.json`: `cer_max_classical*` for the tesseract path,
`cer_max_model*` for the DeepSeek-OCR path. The `*_neutral` numbers are the
gating ones — they compare after `nv-ocr tests/markup_neutral.rs`
neutralization (markup stripped, whitespace collapsed).

Consumed by `rust/crates/nv-ocr/tests/e2e_fixtures.rs` (classical gates) and
`rust/crates/nv-models/tests/deepseek_ocr_graph*.rs` (model gates).
