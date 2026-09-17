# 071-ocr-layout-newspaper

Two-column newspaper page with a masthead and headlines. The two-column
reading order is the point: a backend that reads the two columns as one wide
line scores catastrophically here while passing every single-column page in the
family, so this fixture is where column detection is actually gated
(`cer_max_classical` 0.04 raw, 0.03 neutral).

Rendered by `conformance/tools/generate-071-ocr-layout-fixtures.py` (PIL,
DejaVu fonts, 1400x1900); the PNG is gitignored and rebuilt
by `conformance/tools/regenerate-fixtures.sh`.

Gates live in `fixture.json`: `cer_max_classical*` for the tesseract path,
`cer_max_model*` for the DeepSeek-OCR path. The `*_neutral` numbers are the
gating ones — they compare after
`nv-ocr tests/e2e_fixtures.rs::markup_neutral` (markup stripped, whitespace
collapsed).

Consumed by `rust/crates/nv-ocr/tests/e2e_fixtures.rs` (classical gates).
`rust/crates/nv-models/tests/deepseek_ocr_graph_batch.rs` walks the same list but
asserts graphed==eager token identity, so `cer_max_model*` has no runner.
