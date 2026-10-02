# Visual review of the standalone PDFs

All three final PDFs were rendered and inspected individually. Each retains its own legend and axis labels. Bottom panel titles, panel letters, top titles, and descriptive footers are absent.

- Gauge: all five legend entries, both nonzero curves/IQRs, and numerical-zero controls remain visible.
- Excess error: both method curves and the RBLA IQR are visible, including the SP+ numerical-zero curve. The added black dashed **Optimum (zero)** reference is drawn above SP+ and remains clearly visible, with its own third legend entry. The updated PDF was rendered and visually checked.
- Relative error: the black optimal-reference dashes remain visible over SP+, with both measured method curves and client-rank markers retained.
- Text, legends, and ticks are legible and do not overlap or extend beyond the page.
- All three horizontal axes show `r/R` and exactly the six ticks `0, 0.2, 0.4, 0.6, 0.8, 1`. The measured gauge curves start at `1/16`; the synthetic curves start at `1/32`. Zero is only a tick (and part of the analytic zero-reference line), not an added sample.
- The gauge and synthetic excess plots both use **Normalized excess error**, with separate vertical ranges. Their common definition is `(E_method - E_opt) / ||X||_F`. Converting the archived gauge denominator changes individual excess values by at most `2.220446049250313e-16`; the source CSV remains unchanged.
- Every legend uses exactly **SP+**. The relative-error plot retains its appropriate **Relative Frobenius error** label.
- Temporary PNG previews were removed after review; PDF is the only delivered figure format.

All recorded hashes in the original combined-figure folder match their pre-task values, including the existing A-C-B PDF. The combined figure was neither overwritten nor deleted.
