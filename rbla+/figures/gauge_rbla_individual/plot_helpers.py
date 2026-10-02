"""Frozen drawing helper from the existing three-panel figure."""
import numpy as np
RED, BLUE = "#C44E52", "#4C72B0"

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
