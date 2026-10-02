# Visual review

Reviewed the final main figure at full chart scale and the contact sheet rendered from all seven final PDFs. The review was completed after the automated export checks passed.

- Main and supplementary figures use the same Times New Roman / STIX typography and red/blue/black palette.
- Both panel titles, axis labels, tick labels, the shared legend, and the setting/statistics footer are readable and do not overlap or extend beyond the page.
- The dashed optimal reference remains visible over the blue SP+ curve; hollow blue markers identify actual client ranks. The full-rank zero endpoint is visible.
- Median and IQR shading are visible without obscuring the curves. IQRs are narrow in some settings, reflecting the recorded data.
- Each setting has its own data-driven y range; no values are capped at one. No logarithmic scale hides zero or tiny signed raw residuals.
- All PDFs are one-page vector figures with embedded Times New Roman and STIX fonts. All seven SVGs parse correctly, and all seven PNGs are 600 dpi, 6837 x 2882 pixels.
- Poppler was not available on PATH; PDF previews were rendered with the installed PyMuPDF 1.28.2. The PDF-rendered contact sheet and the main PNG were both inspected.

Scientific data verification also passed: all 210 saved factor fingerprints match the raw paired-target identity; seven prespecified first-seed reproductions match saved global factors bitwise and all checked raw errors exactly; each reproduction uses exactly one R x R SVD. Every client-weighted summary was independently recomputed from the actual rank configuration.

`output_verification.json` contains automated check details. Preview PNGs are QA artifacts; final paper exports are in `../figures/`.
