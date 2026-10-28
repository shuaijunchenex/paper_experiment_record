"""Verify saved data, prespecified reproductions, and exported figure files."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import pymupdf as fitz
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from core import aggregate_rbla,analyze,client_factors

ROOT=Path(__file__).resolve().parent


def verify(output):
    config=json.loads((output/'config_frozen.json').read_text(encoding='utf-8'))
    manifest=json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))
    raw=pd.read_csv(output/'raw_all.csv',float_precision='round_trip')
    clients=pd.read_csv(output/'client_weighted_by_seed.csv',float_precision='round_trip')
    assert manifest['raw_sha256']==hashlib.sha256((output/'raw_all.csv').read_bytes()).hexdigest()
    # run.py hashes canonical UTF-8 text before Windows CRLF translation.
    assert manifest['config_sha256']==hashlib.sha256((output/'config_frozen.json').read_text(encoding='utf-8').encode()).hexdigest()
    assert len(raw)==sum(s['R']*len(config['seeds']) for s in config['settings'])
    assert not raw.duplicated(['setting','seed','r']).any()
    assert raw.checks_passed.all() and raw.sp_decompositions.eq(1).all()
    qa=output/'qa'
    qa.mkdir(exist_ok=True)
    factor_checks, reproductions, images, pdf_checks = [],[],[],[]
    with threadpool_limits(limits=1):
        for setting in config['settings']:
            frame=raw[raw.setting.eq(setting['id'])]
            assert set(frame.seed)==set(config['seeds'])
            archive=np.load(output/'factors'/f"{setting['id']}.npz") if config['save_global_factors'] else None
            for seed, group in frame.groupby('seed'):
                assert sorted(group.r.tolist())==list(range(1,setting['R']+1))
                if archive is not None:
                    b,a=archive[f'{seed}_B_global'],archive[f'{seed}_A_global']
                    fingerprint=hashlib.sha256(b.tobytes()+a.tobytes()).hexdigest()
                    assert group.global_factor_sha256.eq(fingerprint).all()
                    factor_checks.append(dict(setting=setting['id'],seed=int(seed),passed=True))
                chosen=group.set_index('r').loc[setting['ranks']]
                client=clients[clients.setting.eq(setting['id']) & clients.seed.eq(seed)].iloc[0]
                for metric in ['relative_pre','relative_sp','excess_pre','excess_sp','gain']:
                    expected=np.array(setting['weights'])@chosen[metric].to_numpy()
                    assert np.isclose(expected,client[metric],rtol=1e-13,atol=1e-15,equal_nan=True)
            # Same first seed chosen for every setting before looking at results.
            seed=config['seeds'][0]
            factors,_,_=client_factors(setting,seed)
            b,a,_=aggregate_rbla(factors,setting['weights'])
            if archive is not None:
                assert np.array_equal(b,archive[f'{seed}_B_global'])
                assert np.array_equal(a,archive[f'{seed}_A_global'])
            calls=[]
            original_svd=np.linalg.svd
            def monitored_svd(matrix,*args,**kwargs):
                calls.append(list(matrix.shape))
                return original_svd(matrix,*args,**kwargs)
            try:
                np.linalg.svd=monitored_svd
                rows,_,_,_,_=analyze(b,a)
            finally:
                np.linalg.svd=original_svd
            assert calls==[[setting['R'],setting['R']]]
            saved=frame[frame.seed.eq(seed)].sort_values('r')
            rerun=pd.DataFrame(rows).sort_values('r')
            columns=['error_pre','error_sp','error_opt','relative_pre','relative_sp','gain']
            gap=float(np.max(np.abs(saved[columns].to_numpy()-rerun[columns].to_numpy())))
            assert gap<1e-13
            reproductions.append(dict(setting=setting['id'],seed=seed,bitwise_global_factors=True,
                                      max_raw_metric_gap=gap,svd_shapes=calls,passed=True))

    poppler=shutil.which('pdftoppm')
    for pdf in sorted((output/'figures').glob('*.pdf')):
        doc=fitz.open(pdf)
        assert len(doc)==1
        page=doc[0]
        txt=page.get_text()
        for required in ['Relative reconstruction error','Excess reconstruction error','RBLA prefix','SP+']:
            assert required in txt, (pdf.name,required)
        font_rows=[]
        for font in page.get_fonts(full=True):
            basename,extension,kind,content=doc.extract_font(font[0])
            assert content, f'Unembedded font: {font}'
            font_rows.append(dict(name=basename,type=kind,embedded_bytes=len(content)))
        inside=True
        for block in page.get_text('dict')['blocks']:
            if block['type']!=0:
                continue
            for line in block['lines']:
                for span in line['spans']:
                    rect=fitz.Rect(span['bbox'])
                    inside &= rect.x0>=0 and rect.y0>=0 and rect.x1<=page.rect.width and rect.y1<=page.rect.height
        assert inside, f'Out-of-page text: {pdf}'
        preview=qa/f'{pdf.stem}_pdf_render.png'
        if poppler:
            subprocess.run([poppler,'-scale-to','1700','-singlefile','-png',str(pdf),str(preview.with_suffix(''))],check=True,capture_output=True)
        else:
            page.get_pixmap(matrix=fitz.Matrix(1700/page.rect.width,1700/page.rect.width),alpha=False).save(preview)
        png=Image.open(pdf.with_suffix('.png'))
        assert min(png.info.get('dpi',(0,0)))>599
        assert png.width>6000 and png.height>2000
        ET.parse(pdf.with_suffix('.svg'))
        images.append((pdf.stem,preview))
        pdf_checks.append(dict(file=pdf.name,pages=1,fonts=font_rows,text_inside_page=bool(inside),
                               png_pixels=list(png.size),png_dpi=list(png.info['dpi']),
                               svg_xml_valid=True,renderer='Poppler' if poppler else 'PyMuPDF'))
    width, tile_w, tile_h = 1800,900,420
    sheet=Image.new('RGB',(width,tile_h*((len(images)+1)//2)), 'white')
    draw=ImageDraw.Draw(sheet)
    for i,(name,path) in enumerate(images):
        im=Image.open(path).convert('RGB')
        im.thumbnail((tile_w-20,tile_h-35))
        x,y=(i%2)*tile_w,(i//2)*tile_h
        draw.text((x+10,y+5),name,fill='black')
        sheet.paste(im,(x+10,y+30))
    sheet.save(qa/'all_figures_contact_sheet.png')
    result=dict(all_checks_passed=True,unique_rank_records=len(raw),
                saved_factor_identities_checked=len(factor_checks),
                raw_sha256_verified=True,config_sha256_verified=True,
                actual_client_weighting_verified=True,reproductions=reproductions,
                figures=pdf_checks,visual_review='Render files are ready; human/agent visual inspection is a separate step.')
    (qa/'output_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['figures','reproductions']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    verify(parser.parse_args().results)
