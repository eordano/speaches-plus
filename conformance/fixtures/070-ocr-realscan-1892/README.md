# 070-ocr-realscan-1892

Crop of a real 1892 book scan (public domain), not a rendered PNG: grey paper
background, JPEG compression artifacts, and tight leading — descenders of one
line nearly touch the ascenders of the next, so a full-bbox horizontal
projection finds no blank rows between lines. Regression fixture for the
center-band line segmentation in `layout.rs::line_bands`; before that fix the
whole page collapsed into a handful of multi-line bands and the LSTM emitted
garbage. Gates: 4 lines, CER <= 2% with tessdata_best.
