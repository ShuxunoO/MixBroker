# coding=utf-8
"""Stage-D : build figures from the aggregate sweep CSVs.
Line charts (ratio on x, 0-1 score on y, single axis). Series = metric identity,
fixed Okabe-Ito colorblind-safe order, legend + selective direct labels, recessive
grid, std-over-5-seeds shaded bands. Saves PNGs to output/7.3/figures/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/Shuxun/RelatedWork/MixBroker/output/7.3'
FIG = f'{BASE}/figures'
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito (CVD-safe by construction)
OKA = dict(black='#000000', orange='#E69F00', sky='#56B4E9', green='#009E73',
           yellow='#F0E442', blue='#0072B2', verm='#D55E00', purple='#CC79A7', grey='#7F7F7F')

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#E6E6E6', 'grid.linewidth': 0.8,
    'axes.axisbelow': True, 'axes.edgecolor': '#B0B0B0',
})

# metric -> (label, color, linestyle)
SERIES = [
    ('test_auc',        'ROC-AUC (threshold-free, ranking)',   OKA['verm'],  '-'),
    ('test_auc_pr',     'AUC-PR (threshold-free, imbalance)',  OKA['blue'],  '-'),
    ('test_f_1',        'F1 @0.75',                            OKA['green'], '-'),
    ('test_precision',  'Precision @0.75',                     OKA['orange'],'-'),
    ('test_recall',     'Recall @0.75',                        OKA['purple'],'-'),
]


def agg_by_ratio(csv, tag):
    df = pd.read_csv(csv)
    df = df[df.tag == tag]
    out = {}
    for key, *_ in SERIES:
        g = df.groupby('ratio')[key + '_mean']
        out[key] = (g.mean(), g.std().fillna(0))
    prev = df.groupby('ratio')['test_prevalence_mean'].mean()
    ratios = sorted(df.ratio.unique())
    return ratios, out, prev


def panel(ax, csv, tag, title):
    ratios, out, prev = agg_by_ratio(csv, tag)
    x = np.array(ratios)
    for key, label, color, ls in SERIES:
        m, s = out[key]
        m = m.reindex(ratios).values
        s = s.reindex(ratios).values
        ax.plot(x, m, ls, color=color, lw=2, marker='o', ms=5, label=label, zorder=3)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.12, lw=0, zorder=1)
    # prevalence baseline (AUC-PR random floor)
    ax.plot(x, prev.reindex(ratios).values, '--', color=OKA['grey'], lw=1.5,
            label='prevalence (AUC-PR random floor)', zorder=2)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('Negative:Positive ratio  (1:1 ... 1:10)')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels([f'1:{r}' for r in ratios], fontsize=9)
    ax.set_ylim(0, 1.02)
    return ratios, out, prev


def single(csv, tag, title, fname):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    panel(ax, csv, tag, title)
    ax.legend(loc='lower left', fontsize=8.5, framealpha=0.9, ncol=1)
    fig.tight_layout()
    fig.savefig(f'{FIG}/{fname}')
    plt.close(fig)
    print('wrote', fname)


# --- individual panels ---
single(f'{BASE}/aggregate/exp1_sweep.csv', 'exp1_native_42d',
        'EXP1 - MixBroker native data (42d, N=68,419)\nnegatives = semantically valid (deposit->withdraw)', 'fig1_exp1_native.png')
single(f'{BASE}/aggregate/exp2_sweep.csv', 'exp2_tc_42d',
        'EXP2 - Real TC data (42d, N=155,204)', 'fig2_exp2_tc_42d.png')
single(f'{BASE}/aggregate/exp2_sweep.csv', 'exp2_tc_82d',
        'EXP2 - Real TC data (82d, N=155,204)', 'fig3_exp2_tc_82d.png')

# --- headline contrast: ROC-AUC (flat) vs AUC-PR (declines) across all three ---
fig, ax = plt.subplots(figsize=(8.6, 5.4))
sets = [('exp1_sweep.csv', 'exp1_native_42d', 'Native 42d', OKA['verm']),
        ('exp2_sweep.csv', 'exp2_tc_42d', 'TC 42d', OKA['blue']),
        ('exp2_sweep.csv', 'exp2_tc_82d', 'TC 82d', OKA['green'])]
for f, tag, name, col in sets:
    ratios, out, prev = agg_by_ratio(f'{BASE}/aggregate/{f}', tag)
    x = np.array(ratios)
    roc = out['test_auc'][0].reindex(ratios).values
    ap = out['test_auc_pr'][0].reindex(ratios).values
    ax.plot(x, roc, '-', color=col, lw=2, marker='o', ms=4, label=f'{name} · ROC-AUC')
    ax.plot(x, ap, '--', color=col, lw=2, marker='s', ms=4, label=f'{name} · AUC-PR')
ax.set_title('Key contrast: ROC-AUC is nearly flat (solid) while AUC-PR declines (dashed)', fontsize=12, fontweight='bold', pad=8)
ax.set_xlabel('Negative:Positive ratio')
ax.set_ylabel('Score')
ax.set_xticks(x); ax.set_xticklabels([f'1:{r}' for r in ratios], fontsize=9)
ax.set_ylim(0.3, 1.02)
ax.legend(loc='lower left', fontsize=8, ncol=3)
fig.tight_layout(); fig.savefig(f'{FIG}/fig4_roc_vs_aucpr_contrast.png'); plt.close(fig)
print('wrote fig4')

# --- 42d vs 82d generality (F1 & AUC-PR) ---
fig, ax = plt.subplots(figsize=(8.2, 5.2))
for tag, name, col in [('exp2_tc_42d', '42d', OKA['orange']), ('exp2_tc_82d', '82d', OKA['blue'])]:
    ratios, out, prev = agg_by_ratio(f'{BASE}/aggregate/exp2_sweep.csv', tag)
    x = np.array(ratios)
    ax.plot(x, out['test_f_1'][0].reindex(ratios).values, '-', color=col, lw=2, marker='o', ms=5, label=f'{name} · F1@0.75')
    ax.plot(x, out['test_auc_pr'][0].reindex(ratios).values, '--', color=col, lw=2, marker='s', ms=5, label=f'{name} · AUC-PR')
ax.set_title('Generality: the flaw holds under both 42d and 82d\n(82d slightly better, same trend)', fontsize=12, fontweight='bold', pad=8)
ax.set_xlabel('Negative:Positive ratio'); ax.set_ylabel('Score')
ax.set_xticks(x); ax.set_xticklabels([f'1:{r}' for r in ratios], fontsize=9)
ax.set_ylim(0.5, 1.0); ax.legend(loc='lower left', fontsize=9)
fig.tight_layout(); fig.savefig(f'{FIG}/fig5_42d_vs_82d.png'); plt.close(fig)
print('wrote fig5')

# --- anchor: negative-construction effect (native 1:1) ---
anc = pd.read_csv(f'{BASE}/aggregate/anchor.csv').iloc[0]
con_r, con_out, _ = agg_by_ratio(f'{BASE}/aggregate/exp1_sweep.csv', 'exp1_native_42d')
con1 = {k: con_out[k][0].loc[1] for k in ['test_f_1', 'test_auc', 'test_auc_pr', 'test_precision']}
labels = ['F1', 'ROC-AUC', 'AUC-PR', 'Precision@.75']
keys = ['test_f_1', 'test_auc', 'test_auc_pr', 'test_precision']
orig = [anc[k + '_mean'] for k in keys]
cons = [con1[k] for k in keys]
fig, ax = plt.subplots(figsize=(7.6, 4.8))
xb = np.arange(len(labels)); w = 0.38
ax.bar(xb - w/2, orig, w, color=OKA['grey'], label='original unconstrained random neg 1:1 (reproduces paper)')
ax.bar(xb + w/2, cons, w, color=OKA['blue'], label='semantically valid (deposit->withdraw) neg 1:1')
for i, (o, c) in enumerate(zip(orig, cons)):
    ax.text(i - w/2, o + 0.01, f'{o:.2f}', ha='center', fontsize=8.5)
    ax.text(i + w/2, c + 0.01, f'{c:.2f}', ha='center', fontsize=8.5)
ax.set_title('Negative CONSTRUCTION alone also inflates metrics\n(native data, both at 1:1)', fontsize=12, fontweight='bold', pad=8)
ax.set_xticks(xb); ax.set_xticklabels(labels); ax.set_ylabel('Score'); ax.set_ylim(0, 1.0)
ax.legend(loc='upper right', fontsize=8.5)
fig.tight_layout(); fig.savefig(f'{FIG}/fig6_anchor_negative_construction.png'); plt.close(fig)
print('wrote fig6')
print('ALL FIGURES DONE ->', FIG)
