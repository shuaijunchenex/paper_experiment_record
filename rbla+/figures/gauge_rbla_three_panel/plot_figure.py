"""Compact 1x3 PDF for IEEE figure*, styled after semantic_coupling_comparison.

Reads frozen source data only. It does not run new experiments or alter inputs.
"""
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
from gauge_analysis import prepare_excess_error_data

ROOT=Path(__file__).resolve().parent
RED, ORANGE, GRAY, BLUE = '#C44E52', '#DD8452', '#7F7F7F', '#4C72B0'

CAPTION = r"""Gauge sensitivity and reconstruction error in synthetic RBLA aggregation.
Left-to-right panel order is (a), (c), (b), with a separate legend for each panel.
(a) Gauge-induced excess reconstruction error for a fixed structured 128 x 96
rank-16 update with singular values exp(-4j/15), j=0,...,15. The existing control
experiment applies 100 random transformations per gauge type (permutation,
orthogonal, or positive diagonal scaling) to the same factorization while
preserving the full update. Curves show the pointwise median; shaded regions
show the 25th-75th percentiles for permutation and orthogonal gauges. The SP+
curve pools the existing 300 gauge trials, as in the original gauge figure;
SP+, the diagonal control, and the dense optimum coincide at numerical zero.
The horizontal axis in (a) is the absolute target rank r, with R=16.
(b,c) Synthetic RBLA aggregation experiment using N=10 equally weighted clients,
m=n=512, maximum adapter rank R=32, and client ranks
[8,8,12,12,16,16,20,24,28,32]. Client LoRA factors are shared iid Gaussian
reference blocks plus independent perturbations with eta=0.3; the noise standard
deviation equals the RMS of the corresponding reference block. Effective LoRA
scaling is one. Each of 30 fixed independent seeds produces one original RBLA
participation-normalized factor aggregation. RBLA prefix and SP+ are paired on
exactly the same aggregated B_global and A_global, with X=B_global A_global.
Unlike the gauge control in (a), no additional gauge transformation, rotation,
permutation, or reordering is applied after aggregation in (b,c).
Panel (b) shows relative reconstruction errors and the optimal rank-r reference;
panel (c) shows excess reconstruction errors above that reference, normalized
by ||X||_F. SP+ errors are measured from its actual compact QR-SVD output factors.
The black dashed optimum in (b) overlaps SP+. Curves and shading in (b,c) are
pointwise medians and interquartile ranges across seeds, not confidence intervals.
Their horizontal axes show r/R; markers identify actual client ranks.
All errors use unsquared Frobenius norms. The archived gauge measurements use
||X||_F + machine epsilon as their normalization denominator; the synthetic RBLA
measurements use ||X||_F with an explicit reliability check. This existing
convention is preserved; the gauge target is nonzero and well-scaled.
Only tolerance-level negative excess errors in (b,c) are displayed as zero;
the raw data are preserved. These synthetic matrix experiments do not establish
improvements in task accuracy, convergence, fairness, or aggregation bias.
"""


def draw_synthetic(ax,frame,metrics,client_ranks,R):
    values=[]
    summaries=[]
    for metric in metrics:
        q=frame.groupby('r')[metric].quantile([.25,.5,.75]).unstack()
        x=q.index.to_numpy()/R
        marker_indices=[i for i,r in enumerate(q.index) if r in client_ranks]
        pre,opt=metric.endswith('pre'),metric.endswith('opt')
        color=RED if pre else ('black' if opt else BLUE)
        if not opt:
            ax.fill_between(x,q[.25],q[.75],color=color,alpha=.17,lw=0,zorder=1)
        ax.plot(x,q[.5],color=color,lw=1.5 if opt else (2.1 if pre else 2.5),
                ls=(0,(4,3)) if opt else '-',marker=None if opt else ('o' if pre else 's'),
                ms=4.8,markevery=marker_indices,mfc=color if pre else 'white',
                mew=.9,zorder=5 if opt else 3)
        values.extend(q.to_numpy().ravel())
        for rank,row in q.iterrows():
            summaries.append(dict(metric=metric,r=int(rank),normalized_rank=rank/R,
                                  q25=row[.25],median=row[.5],q75=row[.75]))
    low,high=min(0,float(np.nanmin(values))),float(np.nanmax(values))
    span=max(high-low,.01)
    ax.set_ylim(low-.035*span,high+.09*span)
    ax.set_xlim(0,1.025)
    ax.set_xticks([0,.5,1.])
    ax.set_xlabel(r'Normalized client rank $r/R$')
    return summaries


def plot_three_panel(*, figsize=(12.0,5.25), dpi=600, font_family=None,
                     font_size=20, axes_label_size=22, title_size=20,
                     legend_fontsize=17, wspace=.34, output_base=None):
    output_base=Path(output_base) if output_base else ROOT/'gauge_rbla_three_panel'
    provenance=json.loads((ROOT/'sources/provenance.json').read_text(encoding='utf-8'))
    for name in ['gauge_raw.csv','synthetic_main_raw.csv','synthetic_config.json']:
        assert hashlib.sha256((ROOT/'sources'/name).read_bytes()).hexdigest()==provenance[name]['sha256']
    gauge,gstats,spstats,checks,gmetrics=prepare_excess_error_data(ROOT/'sources/gauge_raw.csv')
    assert gmetrics['all_checks_passed'], checks
    config=json.loads((ROOT/'sources/synthetic_config.json').read_text(encoding='utf-8'))
    setting=next(s for s in config['settings'] if s['id']==config['main_setting'])
    synthetic=pd.read_csv(ROOT/'sources/synthetic_main_raw.csv',float_precision='round_trip')
    assert synthetic.setting.eq(config['main_setting']).all() and synthetic.checks_passed.all()
    assert len(synthetic)==len(config['seeds'])*setting['R']
    assert set(synthetic.seed)==set(config['seeds'])
    displayed=synthetic.copy()
    for metric in ['excess_pre','excess_sp']:
        tol=displayed.absolute_tolerance/displayed.norm_x
        assert not (displayed[metric]<-tol).any()
        displayed.loc[displayed[metric].between(-tol,0),metric]=0.
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font_family=font_family or ('Times New Roman' if 'Times New Roman' in fonts else 'DejaVu Serif')
    rc={'font.family':font_family,'font.size':font_size,'axes.labelsize':axes_label_size,
        'axes.titlesize':title_size,'xtick.labelsize':font_size,'ytick.labelsize':font_size,
        'axes.linewidth':1.0,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelpad':5,'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'stix'}
    with plt.rc_context(rc):
        fig,axes=plt.subplots(1,3,figsize=figsize)
        fig.subplots_adjust(left=.085,right=.985,bottom=.275,top=.80,wspace=wspace)
        ax=axes[0]
        ax.axhline(0,color='black',ls='--',lw=1.5,zorder=1)
        styles=[('permutation',RED,'-','o','RBLA-perm.'),
                ('orthogonal',ORANGE,'--','s','RBLA-orth.'),
                ('diagonal',GRAY,':','^','RBLA-diag.')]
        for name,color,linestyle,marker,label in styles:
            group=gstats[gstats.gauge_type.eq(name)].sort_values('r')
            if name!='diagonal':
                ax.fill_between(group.r,group.q25,group.q75,color=color,alpha=.15,lw=0,zorder=2)
            ax.plot(group.r,group['median'],color=color,ls=linestyle,marker=marker,
                    mfc='white' if name=='diagonal' else color,ms=6.0 if name=='diagonal' else 4.8,
                    lw=2.1,label=label,zorder=3)
        ax.plot(spstats.r,spstats['median'],color=BLUE,lw=2.5,marker='s',ms=4.8,
                mfc='white',mec=BLUE,mew=.9,label='SP+',zorder=5)
        ax.set_xlabel(r'Client rank $r$')
        ax.set_ylabel('Excess error over optimum')
        # Keep every measured rank; reduce labels to avoid crowding at IEEE width.
        ax.set_xticks([1,4,8,12,16])
        ax.set_xlim(min(gmetrics['ranks'])-.45,max(gmetrics['ranks'])+.45)
        ax.set_ylim(-.025*gmetrics['maximum_excess'],1.08*gmetrics['maximum_excess'])
        rows=draw_synthetic(axes[1],displayed,['excess_pre','excess_sp'],setting['ranks'],setting['R'])
        rows+=draw_synthetic(axes[2],displayed,['relative_pre','relative_sp','relative_opt'],setting['ranks'],setting['R'])
        axes[1].set_ylabel('Normalized excess error')
        axes[1].set_yticks([0,.05,.10,.15])
        axes[2].set_ylabel('Relative Frobenius error')
        axes[2].set_yticks([0,.2,.4,.6,.8,1.])
        # Each panel has its own legend; retain panel identities in A-C-B order.
        gauge_handles=[Line2D([],[],color=c,lw=2.1,ls=ls,marker=mk,ms=4.8,
                              mfc='white' if name=='diagonal' else c,label=label)
                       for name,c,ls,mk,label in styles]
        method_handles=[Line2D([],[],color=RED,lw=2.1,marker='o',ms=4.8,label='RBLA prefix'),
                        Line2D([],[],color=BLUE,lw=2.5,marker='s',ms=4.8,mfc='white',label='SP+ (compact QR)'),
                        Line2D([],[],color='black',lw=1.5,ls=(0,(4,3)),label='Optimal rank-r reference')]
        gauge_handles.extend([
            Line2D([],[],color=BLUE,lw=2.5,marker='s',ms=4.8,mfc='white',label='SP+'),
            Line2D([],[],color='black',lw=1.5,ls='--',label='Optimum (zero)')])
        panel_legend_handles=[gauge_handles,method_handles[:2],method_handles]
        for index,(ax,handles) in enumerate(zip(axes,panel_legend_handles)):
            ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.335),
                      ncol=2 if index==0 else 1,frameon=False,fontsize=legend_fontsize,
                      handlelength=1.5 if index==0 else 1.8,columnspacing=.6,
                      handletextpad=.35,borderaxespad=0,labelspacing=.3)
        panel_labels=['(a) Gauge-induced\nexcess error',
                      '(c) Excess\nreconstruction error',
                      '(b) Relative\nreconstruction error']
        for ax,label in zip(axes,panel_labels):
            ax.text(.5,-.29,label,transform=ax.transAxes,ha='center',va='top',
                    fontsize=title_size,linespacing=1.05)
        for ax in axes:
            ax.grid(axis='y',ls=':',alpha=.35,lw=.5,zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(direction='out',width=1.0,length=4,pad=3)
        fig.align_xlabels(axes)
        # PDF is the only figure export. Captions contain experiment details;
        # the graphic itself has no descriptive footer or top titles.
        fig.savefig(output_base.with_suffix('.pdf'),dpi=dpi,bbox_inches=None)
        layout=dict(figsize=list(figsize),font_family=font_family,dpi=dpi,rows=1,columns=3,
                    panel_order=['a','c','b'],legend_mode='per-panel',
                    legend_labels=[[h.get_label() for h in handles] for handles in panel_legend_handles],
                    output_formats=['pdf'],ieee_two_column_placement='figure* width=\\textwidth',
                    reference_figure='fedgra/semantic_coupling_comparison.pdf',
                    axes_label_size=axes_label_size,tick_label_size=font_size,
                    panel_label_size=title_size,legend_fontsize=legend_fontsize,wspace=wspace,
                    font_sizes_at_7_16_inches={name:points*7.16/figsize[0] for name,points in
                                             [('axes',axes_label_size),('ticks',font_size),('panels',title_size),('legend',legend_fontsize)]},
                    axes_positions=[list(a.get_position().bounds) for a in axes],
                    panel_titles=[a.get_title() for a in axes],panel_labels=panel_labels,
                    xlabels=[a.get_xlabel() for a in axes],ylimits=[list(a.get_ylim()) for a in axes])
        plt.close(fig)
    gstats.to_csv(ROOT/'gauge_plot_summary.csv',index=False,float_format='%.17g')
    spstats.to_csv(ROOT/'gauge_sp_plot_summary.csv',index=False,float_format='%.17g')
    pd.DataFrame(rows).to_csv(ROOT/'synthetic_plot_summary.csv',index=False,float_format='%.17g')
    (ROOT/'caption.txt').write_text(CAPTION,encoding='utf-8')
    (ROOT/'plot_metadata.json').write_text(json.dumps(dict(layout=layout,gauge_checks=checks,
                        source_rows=dict(gauge=len(gauge),synthetic=len(synthetic)),
                        gauge_trials_per_type=gauge.groupby('gauge_type').trial.nunique().to_dict(),
                        synthetic_seeds=len(config['seeds']),new_experiments_run=False),indent=2),encoding='utf-8')
    print('Created three-panel vector figure:',output_base)
    return output_base


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width',type=float,default=12.0)
    parser.add_argument('--height',type=float,default=5.25)
    parser.add_argument('--dpi',type=int,default=600)
    args=parser.parse_args()
    plot_three_panel(figsize=(args.width,args.height),dpi=args.dpi)
