The 070-ocr-paragraph page rotated +12.0 degrees, outside the legacy +/-5 degree
deskew sweep. Before the coarse-to-fine sweep landed, estimate_skew_scored could
not see past +/-5 deg, the confidence gate never fired, and horizontal cover
bands sliced every rotated line into fragments.

Gates: deskew estimate within 0.5 deg of 12.0 in magnitude, 6 lines, CER <= 2%.
The 12 deg rotation also requires the deskew canvas to be expanded (rotating in
place at that angle clips the corners of the page).
