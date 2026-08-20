The 070-ocr-paragraph page at 150% scale, composited onto a grainy mid-grey
surround under a vertical shadow gradient - the Real5-OmniDocBench capture mode
where a document is photographed on a textured surface.

The surround binarizes into a dense speckle field whose components outnumber the
text 100:1. The count-median component height collapses to ~3 px, the whole-width
cover profile in line_bands is non-zero on every row, and the page yields ONE
band. Recovery needs both the grey-level page-region suppression and the
speckle filter; with neither, this page produces a single line and one token.

The gate is deliberately per-line, not page CER: a 64 px rim of surround survives
the cell-granular page mask and still emits junk lines interleaved with the real
ones. Gate: at least 4 of the 6 expected lines recovered with per-line CER <= 0.1.
