# coding=utf-8
"""Stage-A tool 2/3 : ratio-sweep trainer.

FAITHFUL to run_tc/ablation_run.py (seed 1029, KFold-10 rs=42, hidden 32->16,
Adam lr0.01, BCEWithLogits, 100 epochs, min_epochs 10, threshold 0.75,
best-by-F1, edge_index==edge_label_index). TWO deliberate changes only:

  1. PERF: the feature table is parsed ONCE (id->index map + x tensor built once)
     and reused for every (ratio, seed) config -> no per-fold CSV reloads / iterrows.
     Numerically identical to the original loader.

  2. METRICS: at the SAME best-F1 epoch we additionally record threshold-FREE and
     imbalance-aware scores so the sweep can expose the 1:1 / 0.75 / ROC-AUC illusion:
        auc_pr        = average_precision_score(labels, prob)   (a.k.a. AUPRC)
        prevalence    = #pos / (#pos + #neg)  in the test fold  (AUPRC random baseline)
        auc_pr_lift   = auc_pr / prevalence
        p_at_k, r_at_k= precision/recall @K, K = #pos in fold (R-precision)

The 7 original metrics (accuracy/precision/recall/f1/FPR/FNR/roc_auc) are kept
byte-for-byte in definition. Everything is reported at the best-F1 epoch.
"""
import argparse, json, os, random, time
import numpy as np
import pandas as pd
import torch
import sklearn.metrics as sm
from sklearn.model_selection import KFold
from torch_geometric.data import Data
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# reuse the exact ablation backbone (gcn/gat/sage, 2-layer in->32->16, dot decoder)
sys.path.insert(0, '/Shuxun/RelatedWork/MixBroker/run_tc')
from model_ablation import GNN_NET

THRESH = 0.75


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed = seed          # NOTE: kept verbatim from repo (assignment, not call)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mytest(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        model.train()
    y = data.edge_label.cpu().numpy()
    prob = out.cpu().numpy()
    pred = np.where(prob > THRESH, 1, 0)
    TP = int(((y == 1) & (pred == 1)).sum())
    FP = int(((y == 0) & (pred == 1)).sum())
    FN = int(((y == 1) & (pred == 0)).sum())
    TN = int(((y == 0) & (pred == 0)).sum())
    FPR = FP / (FP + TN) if (FP + TN) else 0.0
    FNR = FN / (TP + FN) if (TP + FN) else 0.0
    acc = sm.accuracy_score(y, pred)
    prec = sm.precision_score(y, pred, zero_division=0)
    rec = sm.recall_score(y, pred, zero_division=0)
    f1 = sm.f1_score(y, pred, zero_division=0)
    roc = sm.roc_auc_score(y, prob) if len(np.unique(y)) > 1 else 0.5
    # ---- added, threshold-free / imbalance-aware ----
    ap = sm.average_precision_score(y, prob) if len(np.unique(y)) > 1 else float(y.mean())
    n_pos = int((y == 1).sum())
    prev = n_pos / len(y) if len(y) else 0.0
    lift = ap / prev if prev > 0 else float('nan')
    # Precision@K / Recall@K with K = #pos in fold
    order = np.argsort(-prob)
    topk = order[:n_pos] if n_pos > 0 else order[:0]
    hit = int(y[topk].sum()) if n_pos > 0 else 0
    p_at_k = hit / n_pos if n_pos > 0 else 0.0
    r_at_k = hit / n_pos if n_pos > 0 else 0.0
    return dict(test_accuracy=acc, test_precision=prec, test_recall=rec, test_f_1=f1,
                test_FPR=FPR, test_FNR=FNR, test_auc=roc,
                test_auc_pr=ap, test_prevalence=prev, test_auc_pr_lift=lift,
                test_p_at_k=p_at_k, test_r_at_k=r_at_k)


METRIC_COLS = ['test_accuracy', 'test_precision', 'test_recall', 'test_f_1',
               'test_FPR', 'test_FNR', 'test_auc', 'test_auc_pr', 'test_prevalence',
               'test_auc_pr_lift', 'test_p_at_k', 'test_r_at_k']


class Sweeper:
    """Loads a feature table once; runs many (pos,neg) 10-fold configs against it."""

    def __init__(self, feat, drop, backbone, device='cpu'):
        self.backbone = backbone
        self.device = torch.device(device)
        t = pd.read_csv(feat)
        t = t.drop(columns=drop)
        ids = t.iloc[:, 0].astype(int).to_numpy()
        self.id2idx = {int(v): i for i, v in enumerate(ids)}
        x = t.iloc[:, 1:].to_numpy().astype(np.float32)
        x[x == np.inf] = 1.0
        x[np.isnan(x)] = 0.0
        self.x = torch.tensor(x, dtype=torch.float32).to(self.device)
        self.n_features = x.shape[1]
        print(f'[sweeper] feat={os.path.basename(feat)} N={x.shape[0]} F={self.n_features}')

    def _make_data(self, edge_df):
        m = self.id2idx
        idx1 = edge_df['nodeid1'].astype(int).map(m)
        idx2 = edge_df['nodeid2'].astype(int).map(m)
        keep = idx1.notna() & idx2.notna()
        e = np.stack([idx1[keep].to_numpy(), idx2[keep].to_numpy()]).astype(np.int64)
        lab = edge_df['label'].astype(np.float32).to_numpy()[keep.to_numpy()]
        ei = torch.tensor(e, dtype=torch.long).to(self.device)
        y = torch.tensor(lab, dtype=torch.float32).to(self.device)
        return Data(x=self.x, edge_index=ei, edge_label=y, edge_label_index=ei)

    def run(self, pos_file, neg_file):
        pos = pd.read_csv(pos_file)
        neg = pd.read_csv(neg_file)
        for c in ('Unnamed: 0',):
            if c in pos.columns: pos = pos.drop(columns=[c])
            if c in neg.columns: neg = neg.drop(columns=[c])
        skf = KFold(n_splits=10, shuffle=True, random_state=42)
        neg_tr, neg_te = [], []
        for tr, te in skf.split(neg):
            neg_tr.append(tr); neg_te.append(te)
        rows = []
        for fold, (ptr, pte) in enumerate(skf.split(pos)):
            df_tr = pd.concat([pos.iloc[ptr], neg.iloc[neg_tr[fold]]]).reset_index(drop=True)
            df_te = pd.concat([pos.iloc[pte], neg.iloc[neg_te[fold]]]).reset_index(drop=True)
            train_data = self._make_data(df_tr)
            test_data = self._make_data(df_te)
            model = GNN_NET(self.n_features, 32, 16, backbone=self.backbone).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            crit = torch.nn.BCEWithLogitsLoss()
            best = None
            model.train()
            for epoch in range(1, 101):
                opt.zero_grad()
                out = model(train_data.x, train_data.edge_index, train_data.edge_label_index).view(-1)
                loss = crit(out, train_data.edge_label)
                loss.backward(); opt.step()
                m = mytest(model, test_data)
                if epoch > 10 and (best is None or m['test_f_1'] > best['test_f_1']):
                    best = m
            if best is None:                     # safety (never triggers with 100>10 epochs)
                best = m
            rows.append(best)
        df = pd.DataFrame(rows)[METRIC_COLS]
        df.insert(0, 'fold', range(1, len(df) + 1))
        summary = {c: {'mean': float(df[c].mean()), 'std': float(df[c].std())} for c in METRIC_COLS}
        return df, summary


def parse_ratios(s):
    if '-' in s:
        a, b = s.split('-'); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


def main():
    P = argparse.ArgumentParser()
    P.add_argument('--backbone', default='sage', choices=['gcn', 'gat', 'sage'])
    P.add_argument('--feat', required=True)
    P.add_argument('--drop', required=True, help='json list of columns to drop')
    P.add_argument('--pos', required=True)
    P.add_argument('--neg-dir', required=True)
    P.add_argument('--ratios', default='1-10')
    P.add_argument('--seeds', type=int, default=5)
    P.add_argument('--out-dir', required=True)
    P.add_argument('--tag', required=True, help='e.g. exp1_native_42d')
    P.add_argument('--agg', required=True, help='aggregate long-CSV to append to')
    args = P.parse_args()

    seed_torch(1029)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    drop = json.loads(args.drop)
    os.makedirs(args.out_dir, exist_ok=True)
    sw = Sweeper(args.feat, drop, args.backbone)
    ratios = parse_ratios(args.ratios)
    agg_rows = []
    t0 = time.perf_counter()
    for r in ratios:
        for s in range(args.seeds):
            neg_file = os.path.join(args.neg_dir, f'neg_r{r}_s{s}.csv')
            tic = time.perf_counter()
            df, summary = sw.run(args.pos, neg_file)
            pref = os.path.join(args.out_dir, f'r{r}_s{s}')
            df.to_csv(pref + '_per_fold.csv', index=False)
            summary['_meta'] = {'tag': args.tag, 'backbone': args.backbone, 'ratio': r, 'seed': s,
                                'feat': os.path.basename(args.feat), 'n_features': sw.n_features,
                                'sec': round(time.perf_counter() - tic, 1)}
            json.dump(summary, open(pref + '_summary.json', 'w'), indent=2)
            row = {'tag': args.tag, 'backbone': args.backbone, 'ratio': r, 'seed': s}
            for c in METRIC_COLS:
                row[c + '_mean'] = summary[c]['mean']
                row[c + '_std'] = summary[c]['std']
            agg_rows.append(row)
            print(f'  r{r} s{s}  f1={summary["test_f_1"]["mean"]:.3f} '
                  f'aucpr={summary["test_auc_pr"]["mean"]:.3f} '
                  f'prec@.75={summary["test_precision"]["mean"]:.3f} '
                  f'roc={summary["test_auc"]["mean"]:.3f} ({summary["_meta"]["sec"]}s)')
    out = pd.DataFrame(agg_rows)
    # append (create header if new)
    header = not os.path.exists(args.agg)
    out.to_csv(args.agg, mode='a', header=header, index=False)
    print(f'[{args.tag}] done {len(agg_rows)} configs in {round(time.perf_counter()-t0,1)}s '
          f'-> appended to {args.agg}')


if __name__ == '__main__':
    main()
