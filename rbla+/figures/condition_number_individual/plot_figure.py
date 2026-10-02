"""Standalone condition-number excess-error PDF, styled like the three rank plots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, FixedLocator, FixedFormatter
import numpy as np
import pandas as pd
import pymupdf as fitz

ROOT=Path(__file__).resolve().parent
RED,BLUE='#C44E52','#4C72B0'


def verify_sources(provenance):
    for name,digest in provenance['protected_original_files'].items():
        assert hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()==digest,name
    for name,digest in provenance['input_sha256'].items():
        assert hashlib.sha256((ROOT/'sources'/name).read_bytes()).hexdigest()==digest,name


def prepare_data(config):
    frame=pd.read_csv(ROOT/'sources/condition_raw.csv',float_precision='round_trip')
    if config.get('additional_data'):
        additional=ROOT/config['additional_data']
        assert hashlib.sha256(additional.read_bytes()).hexdigest()==config['additional_sha256']
        frame=pd.concat([frame,pd.read_csv(additional,float_precision='round_trip')],ignore_index=True)
    assert frame.numerical_checks_passed.all()
    assert frame.R.eq(config['full_rank']).all() and frame.r.eq(config['client_rank']).all()
    assert frame.trial.nunique()==config['trials']
    assert not frame.duplicated(['trial','target_condition_number','method']).any()
    assert len(frame)==config['trials']*len(config['kappas'])*2
    assert np.array_equal(np.sort(frame.target_condition_number.unique()),config['kappas'])

    # Reconstruct only the one fixed target, using the original generator.
    # No gauges, client trials, or full experiments are rerun.
    generator=runpy.run_path(str(ROOT/'sources/source_generator.py'))
    b,a,_=generator['make_structured_factors'](
        config['d_out'],config['d_in'],config['full_rank'],config['target_seed'],config['alpha'])
    target=b@a
    norm_x=float(np.linalg.norm(target,ord='fro'))
    singular_values=np.linalg.svd(target,compute_uv=False)
    norm_s=float(np.linalg.norm(singular_values))
    eps=np.finfo(np.float64).eps
    rank=config['client_rank']
    old_opt=float(np.linalg.norm(singular_values[rank:])/(norm_s+eps))
    reference_gap=float(np.max(np.abs(frame.dense_optimal_error-old_opt)))
    assert reference_gap<1e-12
    # The archived method and optimal errors used slightly different norm
    # evaluations plus eps. Undo each denominator before normalizing by ||X||_F.
    method_multiplier=(norm_x+eps)/norm_x
    optimum_multiplier=(norm_s+eps)/norm_x
    corrected=frame.broadcast_error*method_multiplier-frame.dense_optimal_error*optimum_multiplier
    assert np.isfinite(corrected).all()
    assert corrected.min()>=-config['tolerance']
    frame['normalized_excess_error']=corrected
    # Retain unmodified corrected statistics, including signed numerical noise.
    summary=(frame.groupby(['target_condition_number','method']).normalized_excess_error
             .agg(median='median',q25=lambda s:s.quantile(.25),q75=lambda s:s.quantile(.75),
                  minimum='min',maximum='max',count='count').reset_index())
    assert summary['count'].eq(config['trials']).all()
    summary.to_csv(ROOT/'plot_summary.csv',index=False,float_format='%.17g')
    audit=dict(excess_definition='(E_method - E_opt) / ||X||_F; unsquared Frobenius norms',
               target_frobenius_norm=norm_x,dense_singular_value_norm=norm_s,
               archived_method_denominator='||X||_F + float64 machine epsilon',
               archived_optimum_denominator='||svd(X)||_2 + float64 machine epsilon',
               method_multiplier=method_multiplier,optimum_multiplier=optimum_multiplier,
               archived_optimum_reconstruction_gap=reference_gap,
               maximum_excess_change=float(np.max(np.abs(corrected-frame.excess_error))),
               maximum_sp_excess=float(corrected[frame.method.eq('SP+ (compact QR)')].abs().max()),
               raw_source_csv_unchanged=True,new_experiments_run=config.get('new_experiments_run',False),
               fixed_target_reconstructed_for_normalization=True)
    (ROOT/'qa/normalization_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return frame,summary


def main(preview=False):
    provenance=json.loads((ROOT/'sources/provenance.json').read_text(encoding='utf-8'))
    config=json.loads((ROOT/'extended_config.json').read_text(encoding='utf-8'))
    verify_sources(provenance)
    frame,summary=prepare_data(config)
    fonts={f.name for f in font_manager.fontManager.ttflist}
    family='Times New Roman' if 'Times New Roman' in fonts else 'DejaVu Serif'
    rc={'font.family':family,'font.size':20,'axes.labelsize':22,
        'xtick.labelsize':20,'ytick.labelsize':20,'axes.linewidth':1,
        'axes.spines.top':False,'axes.spines.right':False,'axes.labelpad':5,
        'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'stix'}
    with plt.rc_context(rc):
        fig,ax=plt.subplots(figsize=(5.2,4.9))
        fig.subplots_adjust(left=.20,right=.96,bottom=.20,top=.79)
        ax.axhline(0,color='black',ls='--',lw=1.5,zorder=1)
        for method,color,marker in [('RBLA prefix',RED,'o'),('SP+ (compact QR)',BLUE,'s')]:
            group=summary[summary.method.eq(method)].sort_values('target_condition_number')
            x=group.target_condition_number.to_numpy()
            assert np.array_equal(x,config['kappas'])
            # Only tolerance-sized negatives are displayed as zero; CSV stays raw.
            y=group[['q25','median','q75']].to_numpy(copy=True)
            assert not (y<-config['tolerance']).any()
            y[(y<0)&(y>=-config['tolerance'])]=0.
            ax.fill_between(x,y[:,0],y[:,2],color=color,alpha=.17,lw=0,zorder=2)
            ax.plot(x,y[:,1],color=color,lw=2.1 if method=='RBLA prefix' else 2.5,
                    marker=marker,ms=4.8,mfc=color if method=='RBLA prefix' else 'white',
                    mew=.9,zorder=3 if method=='RBLA prefix' else 5)
        handles=[Line2D([],[],color=RED,lw=2.1,marker='o',ms=4.8,label='RBLA prefix'),
                 Line2D([],[],color=BLUE,lw=2.5,marker='s',ms=4.8,mfc='white',label='SP+'),
                 Line2D([],[],color='black',lw=1.5,ls='--',label='Optimum (zero)')]
        ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.30),
                  ncol=1,frameon=False,fontsize=17,handlelength=1.8,
                  handletextpad=.35,borderaxespad=0,labelspacing=.3)
        ax.set_xscale('log')
        decade_ticks=[10**i for i in range(6)]
        ax.set_xticks(decade_ticks,labels=[rf'$10^{{{i}}}$' for i in range(6)])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(10**(-.10),10**5.10)
        ax.set_xlabel(r'Gauge condition number $\kappa(G)$')
        ax.set_ylabel('Normalized excess error')
        maximum=float(frame.loc[frame.method.eq('RBLA prefix'),'normalized_excess_error'].max())
        ax.set_yscale('symlog',linthresh=1,linscale=.7)
        ax.set_ylim(-.15,1.08*maximum)
        ax.yaxis.set_major_locator(FixedLocator([0,1,10,100,1000]))
        ax.yaxis.set_major_formatter(FixedFormatter(['0',r'$10^0$',r'$10^1$',r'$10^2$',r'$10^3$']))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(axis='y',ls=':',alpha=.35,lw=.5,zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(direction='out',width=1,length=4,pad=3)
        assert not ax.get_title() and not ax.texts and not fig.texts
        target=ROOT/'condition_number_excess_error.pdf'
        fig.savefig(target,bbox_inches=None)
        plt.close(fig)

    with fitz.open(target) as doc:
        assert len(doc)==1
        page=doc[0]
        text=page.get_text()
        assert 'Normalized excess error' in text and 'SP+' in text
        assert not any(s in text for s in ['(a)','(b)','compact QR','Broadcast excess','Numerical stability'])
        embedded=[]
        for font in page.get_fonts(full=True):
            basename,_,_,data=doc.extract_font(font[0])
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
            page.get_pixmap(matrix=fitz.Matrix(1300/page.rect.width,1300/page.rect.width),alpha=False).save(ROOT/'qa/preview.png')
    verify_sources(provenance)
    verification=dict(all_checks_passed=True,existing_figures_unchanged=True,
                      protected_original_files=len(provenance['protected_original_files']),
                      file=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                      page_size_inches=[5.2,4.9],font=family,embedded_fonts=embedded,
                      axis_label_size_pt=22,tick_label_size_pt=20,legend_size_pt=17,
                      legend=[h.get_label() for h in handles],no_title_or_footer=True,
                      x_axis='Gauge condition number kappa(G)',x_scale='log',
                      x_ticks=decade_ticks,sampled_kappas=config['kappas'],no_added_samples=False,
                      y_scale='symlog',y_linthresh=1,y_linscale=.7,
                      new_experiments_run=config.get('new_experiments_run',False),
                      fixed_rank_ratio=config['client_rank']/config['full_rank'],trials=config['trials'],
                      statistic='median and interquartile interval',figure_format='pdf only',
                      dependencies={'python':platform.python_version(),'numpy':np.__version__,
                                    'pandas':pd.__version__,'matplotlib':matplotlib.__version__,
                                    'pymupdf':fitz.VersionBind})
    (ROOT/'qa/verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print('Created:',target,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true',help='Create a temporary PNG for visual review.')
    main(parser.parse_args().preview)
