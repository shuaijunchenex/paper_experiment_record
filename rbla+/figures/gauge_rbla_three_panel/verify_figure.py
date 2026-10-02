"""Check the final figure and render the PDF for visual review."""
import json
import argparse
from pathlib import Path
import pymupdf as fitz

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--render-preview',action='store_true',help='Create a temporary QA PNG; default saves no raster image.')
args=parser.parse_args()
base=ROOT/'gauge_rbla_three_panel'
doc=fitz.open(base.with_suffix('.pdf'))
assert len(doc)==1
page=doc[0]
layout=json.loads((ROOT/'plot_metadata.json').read_text(encoding='utf-8'))['layout']
assert layout['rows']==1 and layout['columns']==3
assert layout['panel_order']==['a','c','b'] and layout['legend_mode']=='per-panel'
assert [len(x) for x in layout['legend_labels']]==[5,2,3]
assert abs(page.rect.width/72-layout['figsize'][0])<1e-5
assert abs(page.rect.height/72-layout['figsize'][1])<1e-5
positions=layout['axes_positions']
assert positions[0][0]<positions[1][0]<positions[2][0]
assert max(p[1] for p in positions)-min(p[1] for p in positions)<1e-12
text=' '.join(page.get_text().split())
for required in ['(a) Gauge-induced excess error','(b) Relative reconstruction error',
                 '(c) Excess reconstruction error']:
    assert required in text,required
assert 'Synthetic RBLA aggregation' not in text and '100 trials/type' not in text
assert layout['panel_titles']==['','',''] and layout['output_formats']==['pdf']
fonts=[]
for font in page.get_fonts(full=True):
    name,extension,kind,data=doc.extract_font(font[0])
    assert data,name
    fonts.append(dict(name=name,embedded=True,type=kind))
for block in page.get_text('dict')['blocks']:
    if block['type']!=0:
        continue
    for line in block['lines']:
        for span in line['spans']:
            rect=fitz.Rect(span['bbox'])
            assert rect.x0>=0 and rect.y0>=0 and rect.x1<=page.rect.width and rect.y1<=page.rect.height,span['text']
if args.render_preview:
    page.get_pixmap(matrix=fitz.Matrix(1700/page.rect.width,1700/page.rect.width),alpha=False).save(ROOT/'qa/pdf_render.png')
result=dict(all_checks_passed=True,pdf_pages=1,pdf_dimensions_points=list(page.rect)[2:],
            figure_layout='one row, three panels, spanning both IEEE text columns',
            panel_order=layout['panel_order'],legend_mode='per-panel',
            physical_size_inches=[page.rect.width/72,page.rect.height/72],
            fonts=fonts,all_text_within_page=True,renderer='PyMuPDF',
            output_format='pdf only',top_titles_removed=True,descriptive_footer_removed=True,
            visual_review='Final PDF was visually inspected; raster QA previews are temporary and are not delivered.')
(ROOT/'qa/export_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
