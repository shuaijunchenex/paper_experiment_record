"""Paper figures from saved raw measurements; no experiment rerun needed."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RED, BLUE = '#C44E52', '#4C72B0'


def caption(config, setting):
    generation = (f"Shared reference factors plus independent client perturbations "
                  f"(eta={setting['eta']:g}); reference entries are independent Gaussian "
                  "variables, and noise standard deviation equals the RMS of the corresponding "
                  "reference block" if setting['generator']=='shared_reference' else
                  "Independent Gaussian client factors, with B and A entry standard deviations "
                  "1/sqrt(m) and 1/sqrt(n), respectively")
    return (f"Synthetic RBLA aggregation experiment. {generation}. "
            f"N={len(setting['ranks'])}, m=n={setting['m']}, R={setting['R']}, "
            f"client ranks={setting['ranks']}, and uniform client weights p_i=1/10. "
            f"Curves show pointwise medians over {len(config['seeds'])} independent random seeds; "
            "shaded bands show the 25th-75th percentiles (IQR), not confidence intervals. "
            "Both redistribution methods are paired on the same single RBLA factor aggregation "
            "per seed, with effective LoRA scaling=1 and no additional post-aggregation gauge "
            "transformation, rotation, permutation, or reordering. All errors use the unsquared "
            "Frobenius norm and the target X=B_global A_global. Panel (a) plots relative "
            "reconstruction errors and the optimal rank-r reference; the dashed black reference "
            "overlaps the measured blue SP+ curve. Panel (b) plots excess errors above that "
            "reference, normalized by ||X||_F. Markers identify ranks actually assigned to clients; "
            "the full r=1,...,R scan is diagnostic. Only tolerance-level negative excess errors "
            "are displayed as zero; raw values are retained.")


def plot_all(output: Path):
    config = json.loads((output/'config_frozen.json').read_text(encoding='utf-8'))
    raw = pd.read_csv(output/'raw_all.csv')
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = 'Times New Roman' if 'Times New Roman' in fonts else 'DejaVu Serif'
    rc = {'font.family':font,'font.size':14,'axes.labelsize':18,'axes.titlesize':17,
          'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':13,
          'axes.linewidth':1.1,'axes.spines.top':False,'axes.spines.right':False,
          'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stix'}
    caption_blocks, qa = [], []
    for setting in config['settings']:
        frame = raw[raw.setting.eq(setting['id'])].copy()
        for metric in ['excess_pre','excess_sp']:
            tol = frame.absolute_tolerance/frame.norm_x
            if (frame[metric] < -tol).any():
                raise AssertionError('Significant negative excess; refuse to clip or plot')
            frame.loc[frame[metric].between(-tol,0), metric] = 0.
        with plt.rc_context(rc):
            fig, axes = plt.subplots(1,2,figsize=(11.6,4.8))
            fig.subplots_adjust(left=.075,right=.985,bottom=.20,top=.79,wspace=.28)
            handles = [Line2D([],[],color=RED,lw=2.1,marker='o',ms=5,label='RBLA prefix'),
                       Line2D([],[],color=BLUE,lw=2.5,marker='s',ms=5,mfc='white',label='SP+ (compact QR)'),
                       Line2D([],[],color='black',lw=1.4,ls=(0,(4,3)),label='Optimal rank-r reference')]
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,1.005),
                       ncol=3,frameon=False,handlelength=2.6,columnspacing=2.0)
            for index, (ax, metrics, title, ylabel) in enumerate(zip(axes,
                    [['relative_pre','relative_sp','relative_opt'],['excess_pre','excess_sp']],
                    ['(a) Relative reconstruction error','(b) Excess reconstruction error'],
                    ['Relative Frobenius error','Normalized excess error'])):
                shown = []
                for metric in metrics:
                    q = frame.groupby('r')[metric].quantile([.25,.5,.75]).unstack()
                    x = q.index.to_numpy()/setting['R']
                    mark = [i for i,r in enumerate(q.index) if r in setting['ranks']]
                    pre, opt = metric.endswith('pre'), metric.endswith('opt')
                    color = RED if pre else ('black' if opt else BLUE)
                    if not opt:
                        ax.fill_between(x,q[.25],q[.75],color=color,alpha=.17,lw=0,zorder=1)
                    ax.plot(x,q[.5],color=color,lw=1.4 if opt else (2.1 if pre else 2.5),
                            ls=(0,(4,3)) if opt else '-',marker=None if opt else ('o' if pre else 's'),
                            ms=4.8,markevery=mark,mfc='white' if not pre else color,
                            mew=1.0,zorder=5 if opt else 3)
                    shown.extend(q.to_numpy().ravel())
                low, high = min(0,float(np.nanmin(shown))), float(np.nanmax(shown))
                span = max(high-low,.01)
                ax.set_ylim(low-.035*span,high+.09*span)
                ax.set_xlim(0,1.025)
                ax.set_xticks([0,.25,.5,.75,1.])
                ax.set_xlabel(r'Normalized client rank $r/R$')
                ax.set_ylabel(ylabel)
                ax.set_title(title,pad=12)
                ax.grid(axis='y',ls=':',alpha=.35,lw=.6,zorder=0)
                ax.set_axisbelow(True)
                if index == 1:
                    ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            setting_text = (rf"Shared reference + perturbations ($\eta={setting['eta']:g}$)"
                            if setting['generator']=='shared_reference' else 'Independent random client factors')
            fig.text(.53,.035,setting_text+rf"  |  $R={setting['R']}$  |  {len(config['seeds'])} seeds: median and IQR  |  markers: client ranks",
                     ha='center',va='center',fontsize=12)
            name = 'main_R32_eta0.3' if setting['id']==config['main_setting'] else setting['id']
            for ext in ['pdf','svg','png']:
                fig.savefig(figures/f'{name}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.08)
            fig.canvas.draw()
            # Store text and axes geometry for reproducible layout QA.
            qa.append(dict(setting=setting['id'],filename=name,font_family=font,png_dpi=600,
                           ylimits=[list(ax.get_ylim()) for ax in axes],
                           titles=[ax.get_title() for ax in axes]))
            plt.close(fig)
        caption_blocks.append(f"## {name}\n\n{caption(config,setting)}\n")
        print(f'Exported {name}: PDF, SVG, 600 dpi PNG',flush=True)
    (figures/'captions.md').write_text('\n'.join(caption_blocks),encoding='utf-8')
    main = next(s for s in config['settings'] if s['id']==config['main_setting'])
    (figures/'main_caption.txt').write_text(caption(config,main)+'\n',encoding='utf-8')
    (figures/'plot_metadata.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    plot_all(parser.parse_args().results)
