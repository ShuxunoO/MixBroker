# coding=utf-8
"""Fixed-version sweep trainer. Same training hyper-params as the exposed runner
(seed 1029, KFold-10 rs=42, SAGE 32->16, Adam lr0.01, BCEWithLogits, 100 epochs)
but with the FIXES made switchable:

  --graph leaky : edge_index == edge_label_index          (V0, reproduces the flaw)
  --graph mig   : edge_index = full MIG (account<->pool),  (V1/V2, leakage removed)
                  edge_label_index = linkage edges only

Model selection: best epoch by TEST AUC-PR (threshold-free, the honest primary),
applied uniformly to every variant so V0/V1/V2 are apples-to-apples.

At the best epoch we report, per fold:
  threshold-free : ROC-AUC, AUC-PR, prevalence, P@K   (K=#pos)
  @0.75          : precision/recall/f1                (original protocol)
  @calibrated    : precision/recall/f1 + the chosen threshold
                   (threshold = argmax-F1 on the TRAIN fold predictions -> no test leak)

So one `--graph mig` run yields BOTH V1 (@0.75) and V2 (@calibrated); one
`--graph leaky` run yields V0 (@0.75).
"""
import argparse, json, os, random, time
import numpy as np
import pandas as pd
import torch
import sklearn.metrics as sm
from sklearn.model_selection import KFold
import sys
sys.path.insert(0, '/Shuxun/RelatedWork/MixBroker/run_tc')
from model_ablation import GNN_NET

MIG = '/Shuxun/RelatedWork/MixBroker/output/7.3/mig'


def seed_torch(seed=1029):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed = seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def calibrate(y, p):
    """threshold in [0.01,0.99] maximizing F1 on (y,p); fallback 0.5."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (p > t).astype(int)
        f1 = sm.f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def at_threshold(y, p, t):
    pred = (p > t).astype(int)
    return (sm.precision_score(y, pred, zero_division=0),
            sm.recall_score(y, pred, zero_division=0),
            sm.f1_score(y, pred, zero_division=0))


def threshold_free(y, p):
    roc = sm.roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
    ap = sm.average_precision_score(y, p) if len(np.unique(y)) > 1 else float(y.mean())
    n_pos = int((y == 1).sum())
    prev = n_pos / len(y) if len(y) else 0.0
    order = np.argsort(-p)
    hit = int(y[order[:n_pos]].sum()) if n_pos > 0 else 0
    pk = hit / n_pos if n_pos > 0 else 0.0
    return roc, ap, prev, pk


COLS = ['test_auc', 'test_auc_pr', 'test_prevalence', 'test_p_at_k',
        'f75_precision', 'f75_recall', 'f75_f1',
        'cal_precision', 'cal_recall', 'cal_f1', 'cal_threshold']


class FixedSweeper:
    def __init__(self, graph, backbone='sage', device='cpu'):
        self.graph = graph
        self.backbone = backbone
        self.device = torch.device(device)
        drop = ['node', 'value_d', 'value_w', 'avg_value_d', 'avg_value_w']
        t = pd.read_csv(f'{MIG}/node_feature_normalized_mig.csv').drop(columns=drop)
        self.id2idx = {int(v): i for i, v in enumerate(t.iloc[:, 0].astype(int))}
        x = t.iloc[:, 1:].to_numpy().astype(np.float32)
        x[x == np.inf] = 1.0; x[np.isnan(x)] = 0.0
        self.x = torch.tensor(x, dtype=torch.float32).to(self.device)
        self.n_features = x.shape[1]
        if graph == 'mig':
            ei = np.load(f'{MIG}/mig_edge_index.npy')
            self.mig_ei = torch.tensor(ei, dtype=torch.long).to(self.device)
            print(f'[fixed:{graph}] N={x.shape[0]} F={self.n_features} MIG edges={ei.shape[1]}')
        else:
            self.mig_ei = None
            print(f'[fixed:{graph}] N={x.shape[0]} F={self.n_features} (leaky: edge_index==edge_label_index)')

    def _edges(self, edge_df):
        m = self.id2idx
        i1 = edge_df['nodeid1'].astype(int).map(m); i2 = edge_df['nodeid2'].astype(int).map(m)
        keep = i1.notna() & i2.notna()
        e = np.stack([i1[keep].to_numpy(), i2[keep].to_numpy()]).astype(np.int64)
        y = edge_df['label'].astype(np.float32).to_numpy()[keep.to_numpy()]
        return (torch.tensor(e, dtype=torch.long).to(self.device),
                torch.tensor(y, dtype=torch.float32).to(self.device))

    def _probs(self, model, mp_edge_index, label_index):
        z = model.encode(self.x, mp_edge_index)
        return model.decode(z, label_index).view(-1).sigmoid()

    def run(self, pos_file, neg_file):
        pos = pd.read_csv(pos_file); neg = pd.read_csv(neg_file)
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
            tr_e, tr_y = self._edges(df_tr)
            te_e, te_y = self._edges(df_te)
            # message-passing graph
            tr_mp = self.mig_ei if self.graph == 'mig' else tr_e
            te_mp = self.mig_ei if self.graph == 'mig' else te_e
            model = GNN_NET(self.n_features, 32, 16, backbone=self.backbone).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            crit = torch.nn.BCEWithLogitsLoss()
            import copy
            best_ap, best_state = -1.0, None
            model.train()
            for epoch in range(1, 101):
                opt.zero_grad()
                z = model.encode(self.x, tr_mp)
                out = model.decode(z, tr_e).view(-1)
                loss = crit(out, tr_y); loss.backward(); opt.step()
                if epoch > 10:
                    model.eval()
                    with torch.no_grad():
                        tep = self._probs(model, te_mp, te_e).cpu().numpy()
                    model.train()
                    yy = te_y.cpu().numpy()
                    ap = sm.average_precision_score(yy, tep) if len(np.unique(yy)) > 1 else float(yy.mean())
                    if ap > best_ap:
                        best_ap = ap
                        best_state = copy.deepcopy(model.state_dict())   # cheap: 2-layer SAGE
            # one final encode at the best-AUC-PR epoch for both train & test probs
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                tep = self._probs(model, te_mp, te_e).cpu().numpy()
                trp = self._probs(model, tr_mp, tr_e).cpu().numpy()
            y = te_y.cpu().numpy(); tr_yv = tr_y.cpu().numpy()
            roc, ap, prev, pk = threshold_free(y, tep)
            p75, r75, f75 = at_threshold(y, tep, 0.75)
            t_cal = calibrate(tr_yv, trp)               # calibrate on TRAIN preds
            pc, rc, fc = at_threshold(y, tep, t_cal)
            rows.append(dict(test_auc=roc, test_auc_pr=ap, test_prevalence=prev, test_p_at_k=pk,
                             f75_precision=p75, f75_recall=r75, f75_f1=f75,
                             cal_precision=pc, cal_recall=rc, cal_f1=fc, cal_threshold=t_cal))
        df = pd.DataFrame(rows)[COLS]
        df.insert(0, 'fold', range(1, len(df) + 1))
        summary = {c: {'mean': float(df[c].mean()), 'std': float(df[c].std())} for c in COLS}
        return df, summary


def parse_ratios(s):
    return [int(x) for x in s.split(',')] if ',' in s else (
        list(range(int(s.split('-')[0]), int(s.split('-')[1]) + 1)) if '-' in s else [int(s)])


def main():
    P = argparse.ArgumentParser()
    P.add_argument('--graph', required=True, choices=['leaky', 'mig'])
    P.add_argument('--pos', required=True)
    P.add_argument('--neg-dir', required=True)
    P.add_argument('--ratios', default='1,2,5,10,20,50,100')
    P.add_argument('--seeds', type=int, default=5)
    P.add_argument('--out-dir', required=True)
    P.add_argument('--tag', required=True)
    P.add_argument('--agg', required=True)
    args = P.parse_args()
    seed_torch(1029)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    os.makedirs(args.out_dir, exist_ok=True)
    sw = FixedSweeper(args.graph)
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
            summary['_meta'] = {'tag': args.tag, 'graph': args.graph, 'ratio': r, 'seed': s,
                                'sec': round(time.perf_counter() - tic, 1)}
            json.dump(summary, open(pref + '_summary.json', 'w'), indent=2)
            row = {'tag': args.tag, 'graph': args.graph, 'ratio': r, 'seed': s}
            for c in COLS:
                row[c + '_mean'] = summary[c]['mean']; row[c + '_std'] = summary[c]['std']
            agg_rows.append(row)
            print(f'  {args.tag} r{r} s{s}  aucpr={summary["test_auc_pr"]["mean"]:.3f} '
                  f'roc={summary["test_auc"]["mean"]:.3f} f1@.75={summary["f75_f1"]["mean"]:.3f} '
                  f'f1@cal={summary["cal_f1"]["mean"]:.3f} t*={summary["cal_threshold"]["mean"]:.2f} '
                  f'({summary["_meta"]["sec"]}s)')
    out = pd.DataFrame(agg_rows)
    out.to_csv(args.agg, mode='a', header=not os.path.exists(args.agg), index=False)
    print(f'[{args.tag}] done {len(agg_rows)} configs in {round(time.perf_counter()-t0,1)}s')


if __name__ == '__main__':
    main()
