# Final visual review

The final layout is one horizontal row of three panels, spanning both IEEE text columns. Left-to-right order: (a) gauge, (c) excess reconstruction error, (b) relative reconstruction error. Each panel has a separate legend. The PDF master is 12 x 5.25 inches and follows the typography and panel proportions of `fedgra/semantic_coupling_comparison.pdf`; use `figure*` and `width=\textwidth` for placement.

- Top titles are removed; full panel labels appear below the axes. The descriptive footer is removed.
- Enlarged text remains readable without overlap: 22 pt axis labels, 20 pt ticks and panel labels, 17 pt per-panel legends. Temporary rendered previews were inspected after the final spacing adjustment.
- The three panels preserve their respective source axis semantics and error scales.
- The black optimal reference is visible over the blue SP+ curve in panel (b).
- Gauge controls at numerical zero remain in panel (a); source median/IQR statistics are retained.
- All fonts are embedded in the PDF. PDF is the only delivered image format; old SVG/PNG exports and temporary QA rasters were removed.
- Only the source data snapshots and new derived files in this folder were written. The original gauge and synthetic figures were not edited.

The horizontal subplot spacing is reduced from 0.43 to 0.34, with room retained for enlarged vertical labels. The PDF checks confirm that all text lies within the page. Full captions and a figure* LaTeX snippet are provided alongside the figure.

The latest A-C-B PDF was rendered and inspected. Each legend contains only the methods shown in that panel; the gauge, excess and relative panels contain 5, 2 and 3 legend entries, respectively. No text overlap or clipping was observed.
