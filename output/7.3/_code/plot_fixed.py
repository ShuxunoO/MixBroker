# coding=utf-8
"""Stage-E : fixed-version comparison figures (V0 leaky / V1 mig / V2 calibrated),
imbalance sweep to 1:100. English labels (no CJK font on host). AML python."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE='/Shuxun/RelatedWork/MixBroker/output/7.3'; FIG=f'{BASE}/figures'
OKA=dict(orange='#E69F00',sky='#56B4E9',green='#009E73',blue='#0072B2',verm='#D55E00',purple='#CC79A7',grey='#7F7F7F')
plt.rcParams.update({'figure.dpi':130,'savefig.dpi':130,'font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.color':'#E6E6E6','grid.linewidth':0.8,
    'axes.axisbelow':True,'axes.edgecolor':'#B0B0B0'})

df=pd.read_csv(f'{BASE}/aggregate/fixed_sweep.csv')
def bym(tag): return df[df.tag==tag].groupby('ratio').mean(numeric_only=True)
lk,mg=bym('v0_leaky'),bym('mig')
R=sorted(df.ratio.unique()); x=np.array(R)

def xax(ax):
    ax.set_xscale('log'); ax.set_xticks(R); ax.set_xticklabels([f'1:{r}' for r in R],fontsize=9)
    ax.set_xlabel('Negative:Positive ratio (log)'); ax.minorticks_off()

# Fig7 : AUC-PR — leakage attribution (V0 vs V1) + imbalance collapse to 1:100
fig,ax=plt.subplots(figsize=(8.4,5.2))
ax.plot(x,lk.test_auc_pr_mean.reindex(R),'-o',color=OKA['verm'],lw=2,ms=5,label='V0 Leaky (edge_index==edge_label_index)')
ax.plot(x,mg.test_auc_pr_mean.reindex(R),'-s',color=OKA['blue'],lw=2,ms=5,label='V1 MIG (leakage removed)')
ax.plot(x,mg.test_prevalence_mean.reindex(R),'--',color=OKA['grey'],lw=1.5,label='prevalence (AUC-PR random floor)')
ax.fill_between(x,mg.test_auc_pr_mean.reindex(R),lk.test_auc_pr_mean.reindex(R),color=OKA['orange'],alpha=0.15,label='leakage inflation (small on native)')
ax.set_title('Fixed-version: AUC-PR vs imbalance\nleakage removal = small gap; realistic imbalance = collapse to the floor',fontsize=11.5,fontweight='bold',pad=8)
ax.set_ylabel('AUC-PR'); xax(ax); ax.set_ylim(0,0.9); ax.legend(loc='upper right',fontsize=8.5)
fig.tight_layout(); fig.savefig(f'{FIG}/fig7_fixed_aucpr_leakage.png'); plt.close(fig); print('fig7')

# Fig8 : F1 — threshold calibration recovers F1 (V0@.75 / V1@.75 / V2@cal)
fig,ax=plt.subplots(figsize=(8.4,5.2))
ax.plot(x,lk.f75_f1_mean.reindex(R),'-o',color=OKA['verm'],lw=2,ms=5,label='V0 Leaky · F1 @0.75')
ax.plot(x,mg.f75_f1_mean.reindex(R),'-s',color=OKA['orange'],lw=2,ms=5,label='V1 MIG · F1 @0.75')
ax.plot(x,mg.cal_f1_mean.reindex(R),'-^',color=OKA['green'],lw=2,ms=6,label='V2 MIG · F1 @calibrated')
ax.set_title('Fixed-version: threshold calibration recovers F1\n(calibrated threshold beats fixed 0.75 at every ratio)',fontsize=11.5,fontweight='bold',pad=8)
ax.set_ylabel('F1'); xax(ax); ax.set_ylim(0,0.8); ax.legend(loc='upper right',fontsize=8.5)
fig.tight_layout(); fig.savefig(f'{FIG}/fig8_fixed_f1_calibration.png'); plt.close(fig); print('fig8')

# Fig9 : honest headline — MIG ROC (flat) vs AUC-PR (collapse) to 1:100
fig,ax=plt.subplots(figsize=(8.4,5.2))
ax.plot(x,mg.test_auc_mean.reindex(R),'-o',color=OKA['verm'],lw=2,ms=5,label='ROC-AUC (looks passable)')
ax.plot(x,mg.test_auc_pr_mean.reindex(R),'-s',color=OKA['blue'],lw=2,ms=5,label='AUC-PR (honest)')
ax.plot(x,mg.cal_recall_mean.reindex(R),'-^',color=OKA['purple'],lw=2,ms=5,label='Recall @calibrated')
ax.plot(x,mg.test_prevalence_mean.reindex(R),'--',color=OKA['grey'],lw=1.5,label='prevalence floor')
ax.set_title('Fixed & honest (MIG): ROC-AUC stays 0.64-0.77 while AUC-PR\ncollapses toward the prevalence floor at realistic 1:100',fontsize=11.5,fontweight='bold',pad=8)
ax.set_ylabel('Score'); xax(ax); ax.set_ylim(0,0.85); ax.legend(loc='upper right',fontsize=8.5)
fig.tight_layout(); fig.savefig(f'{FIG}/fig9_fixed_honest_headline.png'); plt.close(fig); print('fig9')

# Fig10 : calibrated threshold t* rises with imbalance (why fixed 0.75 fails)
fig,ax=plt.subplots(figsize=(7.6,4.6))
ax.plot(x,mg.cal_threshold_mean.reindex(R),'-o',color=OKA['green'],lw=2,ms=6,label='calibrated threshold t*')
ax.axhline(0.75,ls='--',color=OKA['verm'],lw=1.5,label='fixed 0.75 (MixBroker)')
ax.set_title('Why fixed 0.75 fails: the optimal threshold RISES with imbalance',fontsize=11.5,fontweight='bold',pad=8)
ax.set_ylabel('threshold'); xax(ax); ax.set_ylim(0.4,0.95); ax.legend(loc='lower right',fontsize=9)
fig.tight_layout(); fig.savefig(f'{FIG}/fig10_fixed_threshold.png'); plt.close(fig); print('fig10')
print('DONE')
