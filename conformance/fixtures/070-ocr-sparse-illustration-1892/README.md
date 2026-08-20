# 070-ocr-sparse-illustration-1892

A sparse real-scan page: one large halftone engraving, wide blank margins, and a
two-line caption that is the only text on the page. The 070/071 corpus was all
text-dense before this fixture, which is why the task #78 speckle filter shipped
green while destroying this page class.

The engraving binarizes into one enormous connected component holding ~72% of
the page ink plus ~2600 stipple fragments. `speckle_filter`'s area-weighted
median height is therefore the height of the engraving (921 px), not a glyph
height, and the 0.35x cut lands at 322 px — above every caption glyph. The
surviving population is a single component, the page collapses to one band, and
the LSTM emits one junk token. The guard is `SPECKLE_MIN_KEPT`: a filter that
leaves fewer components than a text line's worth has not found a text
population, so the page takes the untouched path.

Gate is per-line, matching 070-ocr-photo-noise-surround: both caption lines
recovered with per-line CER <= 0.1. Pre-guard this fixture recovers 0 of 2.
