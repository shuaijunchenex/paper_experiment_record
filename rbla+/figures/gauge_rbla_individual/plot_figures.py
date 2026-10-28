"""Three standalone PDFs, without panel titles; preserves the combined figure."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pymupdf as fitz
from gauge_analysis import prepare_excess_error_data
from plot_helpers import draw_synthetic

ROOT=Path(__file__).resolve().parent
RED,ORANGE,GRAY,BLUE='#C44E52','#DD8452','#7F7F7F','#4C72B0'


def verify_original(provenance):
    source=Path(provenance['source_folder'])
    for name,digest in provenance['protected_original_files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest, name


def load_gauge_with_common_normalization():
    """Convert archived E/(||X||_F + eps) to E/||X||_F; keep source CSV intact."""
    frame=pd.read_csv(ROOT/'sources/gauge_raw.csv',float_precision='round_trip')
    assert frame.R.nunique()==1 and frame.spectrum.eq('exp(-4*i/(R-1))').all()
    rank=int(frame.R.iloc[0])
    # The recorded target has orthonormal left/right factors and this spectrum.
    # Its norm is invariant under all gauges in the archived experiment.
    singular_values=np.exp(-4*np.arange(rank,dtype=np.float64)/(rank-1))
    norm_x=float(np.linalg.norm(singular_values))
    eps=np.finfo(np.float64).eps
    correction=(norm_x+eps)/norm_x
    archived_excess=frame.broadcast_error-frame.dense_optimal_error
    analytic_opt=np.array([np.linalg.norm(singular_values[int(r):])/(norm_x+eps)
                           for r in frame.r])
    reference_gap=float(np.max(np.abs(frame.dense_optimal_error-analytic_opt)))
    assert reference_gap<1e-12
    for column in ['broadcast_error','dense_optimal_error','full_update_error',
                   'sp_plus_dense_error_gap']:
        frame[column]*=correction
    frame['excess_error']=frame.broadcast_error-frame.dense_optimal_error
    audit=dict(common_excess_definition='(E_method - E_opt) / ||X||_F; unsquared Frobenius norms',
               archived_gauge_denominator='||X||_F + float64 machine epsilon',
               gauge_norm_from_recorded_spectrum=norm_x,machine_epsilon=float(eps),
               gauge_conversion_multiplier=correction,
               maximum_archived_optimum_vs_spectrum_gap=reference_gap,
               maximum_gauge_excess_change=float(np.max(np.abs(frame.excess_error-archived_excess))),
               raw_source_csv_unchanged=True)
    return frame,audit


def main(preview=False,figure=None):
    provenance=json.loads((ROOT/'sources/provenance.json').read_text(encoding='utf-8'))
    verify_original(provenance)
    for name,digest in provenance['input_sha256'].items():
        assert hashlib.sha256((ROOT/'sources'/name).read_bytes()).hexdigest()==digest
    gauge_frame,normalization_audit=load_gauge_with_common_normalization()
    _,gstats,spstats,checks,gmetrics=prepare_excess_error_data(gauge_frame)
    assert gmetrics['all_checks_passed'],checks
    config=json.loads((ROOT/'sources/synthetic_config.json').read_text(encoding='utf-8'))
    setting=next(s for s in config['settings'] if s['id']==config['main_setting'])
    frame=pd.read_csv(ROOT/'sources/synthetic_main_raw.csv',float_precision='round_trip')
    assert frame.checks_passed.all()
    assert len(frame)==len(config['seeds'])*setting['R']
    definition_gaps={method:float(np.max(np.abs(
        frame[f'excess_{method}']-(frame[f'error_{method}']-frame.error_opt)/frame.norm_x)))
        for method in ['pre','sp']}
    assert all(gap<=2*np.finfo(np.float64).eps for gap in definition_gaps.values())
    normalization_audit['synthetic_definition_maximum_gaps']=definition_gaps
    (ROOT/'qa/normalization_audit.json').write_text(json.dumps(normalization_audit,indent=2),encoding='utf-8')
    for metric in ['excess_pre','excess_sp']:
        tol=frame.absolute_tolerance/frame.norm_x
        assert not (frame[metric]<-tol).any()
        frame.loc[frame[metric].between(-tol,0),metric]=0.
    fonts={f.name for f in font_manager.fontManager.ttflist}
    family='Times New Roman' if 'Times New Roman' in fonts else 'DejaVu Serif'
    rc={'font.family':family,'font.size':20,'axes.labelsize':22,
        'xtick.labelsize':20,'ytick.labelsize':20,'axes.linewidth':1,
        'axes.spines.top':False,'axes.spines.right':False,'axes.labelpad':5,
        'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'stix'}
    method_handles=[Line2D([],[],color=RED,lw=2.1,marker='o',ms=4.8,label='RBLA prefix'),
                    Line2D([],[],color=BLUE,lw=2.5,marker='s',ms=4.8,mfc='white',label='SP+'),
                    Line2D([],[],color='black',lw=1.5,ls=(0,(4,3)),label='Optimal rank-r reference')]
    names=[('a','gauge_excess_error'),('c','rbla_excess_error'),('b','rbla_relative_error')]
    if figure is not None:
        names=[item for item in names if item[1]==figure]
        assert names,figure
    untouched={path:hashlib.sha256(path.read_bytes()).hexdigest()
               for path in ROOT.glob('*.pdf') if path.stem not in [name for _,name in names]}
    results=[]
    for original_panel,name in names:
        with plt.rc_context(rc):
            fig,ax=plt.subplots(figsize=(5.2,4.9))
            fig.subplots_adjust(left=.20,right=.96,bottom=.20,top=.79)
            if original_panel=='a':
                zero_reference=ax.axhline(0,color='black',ls='--',lw=1.5,zorder=1)
                styles=[('permutation',RED,'-','o','RBLA-perm.'),
                        ('orthogonal',ORANGE,'--','s','RBLA-orth.'),
                        ('diagonal',GRAY,':','^','RBLA-diag.')]
                handles=[]
                for gauge,color,linestyle,marker,label in styles:
                    group=gstats[gstats.gauge_type.eq(gauge)].sort_values('r')
                    x=group.r/gmetrics['maximum_rank']
                    if gauge!='diagonal':
                        ax.fill_between(x,group.q25,group.q75,color=color,alpha=.15,lw=0,zorder=2)
                    ax.plot(x,group['median'],color=color,ls=linestyle,marker=marker,
                            mfc='white' if gauge=='diagonal' else color,ms=6 if gauge=='diagonal' else 4.8,
                            lw=2.1,zorder=3)
                    handles.append(Line2D([],[],color=color,lw=2.1,ls=linestyle,marker=marker,
                                          ms=4.8,mfc='white' if gauge=='diagonal' else color,label=label))
                ax.plot(spstats.r/gmetrics['maximum_rank'],spstats['median'],color=BLUE,lw=2.5,marker='s',ms=4.8,
                        mfc='white',mec=BLUE,mew=.9,zorder=5)
                handles.extend([Line2D([],[],color=BLUE,lw=2.5,marker='s',ms=4.8,mfc='white',label='SP+'),
                                Line2D([],[],color='black',lw=1.5,ls='--',label='Optimum (zero)')])
                ax.set_ylabel('Normalized excess error')
                ax.set_ylim(-.025*gmetrics['maximum_excess'],1.08*gmetrics['maximum_excess'])
                sample_ranks=np.asarray(gmetrics['ranks'])/gmetrics['maximum_rank']
                data_lines=[line for line in ax.lines if line is not zero_reference]
            else:
                is_excess=original_panel=='c'
                metrics=['excess_pre','excess_sp'] if is_excess else ['relative_pre','relative_sp','relative_opt']
                draw_synthetic(ax,frame,metrics,setting['ranks'],setting['R'])
                ax.set_ylabel('Normalized excess error' if is_excess else 'Relative Frobenius error')
                ax.set_yticks([0,.05,.10,.15] if is_excess else [0,.2,.4,.6,.8,1.])
                sample_ranks=np.sort(frame.r.unique())/setting['R']
                if is_excess:
                    # Draw above SP+ so the coincident optimal reference remains visible.
                    ax.plot(sample_ranks,np.zeros_like(sample_ranks),color='black',
                            lw=1.5,ls=(0,(4,3)),zorder=5)
                    handles=method_handles[:2]+[Line2D([],[],color='black',lw=1.5,
                                                     ls=(0,(4,3)),label='Optimum (zero)')]
                else:
                    handles=method_handles
                data_lines=list(ax.lines)
            ax.set_xlabel(r'Normalized client rank $r/R$')
            ax.set_xlim(0,1.025)
            ax.set_xticks([0,.2,.4,.6,.8,1.],labels=['0','0.2','0.4','0.6','0.8','1'])
            # Zero is an axis tick, never an added experimental sample.
            assert (sample_ranks>0).all()
            for line in data_lines:
                assert np.array_equal(line.get_xdata(),sample_ranks)
            ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.30),
                      ncol=2 if original_panel=='a' else 1,frameon=False,fontsize=17,
                      handlelength=1.5 if original_panel=='a' else 1.8,columnspacing=.6,
                      handletextpad=.35,borderaxespad=0,labelspacing=.3)
            ax.grid(axis='y',ls=':',alpha=.35,lw=.5,zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(direction='out',width=1,length=4,pad=3)
            # No top title, panel letter, bottom title or descriptive footer.
            assert not ax.get_title() and not ax.texts and not fig.texts
            target=ROOT/f'{name}.pdf'
            fig.savefig(target,bbox_inches=None)
            plt.close(fig)
        doc=fitz.open(target)
        assert len(doc)==1
        page=doc[0]
        text=page.get_text()
        assert not any(label in text for label in ['(a)','(b)','(c)','Gauge-induced','reconstruction error'])
        assert 'SP+' in text
        assert 'compact QR' not in text
        if original_panel=='c':
            assert 'Optimum (zero)' in text
        embedded=[]
        for font in page.get_fonts(full=True):
            basename,extension,kind,data=doc.extract_font(font[0])
            assert data,basename
            embedded.append(basename)
        for block in page.get_text('dict')['blocks']:
            if block['type']!=0:
                continue
            for line in block['lines']:
                for span in line['spans']:
                    rect=fitz.Rect(span['bbox'])
                    assert rect.x0>=0 and rect.y0>=0 and rect.x1<=page.rect.width and rect.y1<=page.rect.height,span['text']
        if preview:
            page.get_pixmap(matrix=fitz.Matrix(1300/page.rect.width,1300/page.rect.width),alpha=False).save(ROOT/'qa'/f'{name}_preview.png')
        results.append(dict(file=target.name,original_panel=original_panel,pdf_pages=1,
                            fonts_embedded=embedded,panel_title_removed=True,
                            independent_legend=[h.get_label() for h in handles],
                            x_axis='r/R',x_ticks=[0,.2,.4,.6,.8,1.],
                            normalized_sample_ranks=sample_ranks.tolist(),no_added_zero_sample=True,
                            y_axis=ax.get_ylabel(),
                            optimal_reference_present=True,
                            all_text_inside_page=True,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        print('Created:',target.name,flush=True)
    verify_original(provenance)
    for path,digest in untouched.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,path.name
    if figure is not None and (ROOT/'qa/verification.json').exists():
        previous=json.loads((ROOT/'qa/verification.json').read_text(encoding='utf-8'))
        updated={item['file']:item for item in results}
        results=[updated.get(item['file'],item) for item in previous['figures']]
    (ROOT/'qa/verification.json').write_text(json.dumps(dict(all_checks_passed=True,
                    original_combined_folder_unchanged=True,
                    protected_original_files=len(provenance['protected_original_files']),
                    new_experiments_run=False,figure_format='pdf only',figures=results),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true',help='Write temporary QA previews; default is PDF only.')
    parser.add_argument('--figure',choices=['gauge_excess_error','rbla_excess_error','rbla_relative_error'],
                        help='Regenerate only the selected PDF.')
    args=parser.parse_args()
    main(args.preview,args.figure)
