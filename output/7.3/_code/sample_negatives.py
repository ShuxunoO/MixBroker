# coding=utf-8
"""Stage-A tool 1/3 : negative-edge sampler with (deposit -> withdraw) constraint.

Draws negatives as (a, b) where a is a DEPOSIT-capable node and b is a
WITHDRAW-capable node, a != b, and the *unordered* pair {a,b} is NOT a positive.
No (deposit,deposit) / (withdraw,withdraw) pairs are ever produced.

Per seed we draw the LARGEST ratio's worth once (max_ratio * n_pos) and then the
smaller-ratio files are prefixes of that draw -> ratios are NESTED per seed, so
the ONLY thing changing across the sweep is "more negatives added", nothing else.

Output files match the MixBroker Dataset/Graph schema exactly:
    ,nodeid1,nodeid2,label      (leading unnamed index col; label==0)
so run_tc/ablation_run.py's `.drop(columns=['Unnamed: 0'])` works unchanged.

Role source:
    --role-mode flag   -> is_deposit_seed>0 / is_withdraw_seed>0   (82d files)
    --role-mode count  -> num_d_all>0       / num_w_all>0          (raw native file)
"""
import argparse, json, os
import numpy as np
import pandas as pd


def parse_ratios(s):
    if '-' in s:
        a, b = s.split('-'); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


def load_roles(role_file, role_mode):
    if role_mode == 'flag':
        cols = ['nodeid', 'is_deposit_seed', 'is_withdraw_seed']
        df = pd.read_csv(role_file, usecols=cols)
        dep = df.loc[df['is_deposit_seed'].astype(float) > 0, 'nodeid'].astype(int).to_numpy()
        wd = df.loc[df['is_withdraw_seed'].astype(float) > 0, 'nodeid'].astype(int).to_numpy()
    elif role_mode == 'count':
        cols = ['nodeid', 'num_d_all', 'num_w_all']
        df = pd.read_csv(role_file, usecols=cols)
        dep = df.loc[df['num_d_all'].astype(float) > 0, 'nodeid'].astype(int).to_numpy()
        wd = df.loc[df['num_w_all'].astype(float) > 0, 'nodeid'].astype(int).to_numpy()
    else:
        raise ValueError(role_mode)
    return dep, wd


def load_pos_pairs(pos_file):
    p = pd.read_csv(pos_file)
    a = p['nodeid1'].astype(int).to_numpy()
    b = p['nodeid2'].astype(int).to_numpy()
    n_pos = len(p)
    pos_set = set()
    for x, y in zip(a, b):
        pos_set.add((x, y) if x <= y else (y, x))
    return n_pos, pos_set


def sample_one_seed(dep, wd, pos_set, n_target, seed):
    """Return list of (d, w) accepted in draw order, length == n_target."""
    rng = np.random.RandomState(seed)
    seen = set()
    out = []
    batch = max(4096, n_target)
    while len(out) < n_target:
        da = dep[rng.randint(0, len(dep), size=batch)]
        wa = wd[rng.randint(0, len(wd), size=batch)]
        for d, w in zip(da, wa):
            d = int(d); w = int(w)
            if d == w:
                continue
            key = (d, w) if d <= w else (w, d)
            if key in pos_set or key in seen:
                continue
            seen.add(key)
            out.append((d, w))
            if len(out) >= n_target:
                break
    return out


def main():
    P = argparse.ArgumentParser()
    P.add_argument('--pos', required=True)
    P.add_argument('--role-file', required=True)
    P.add_argument('--role-mode', required=True, choices=['flag', 'count'])
    P.add_argument('--ratios', default='1-10')
    P.add_argument('--seeds', type=int, default=5, help='number of seeds (0..N-1)')
    P.add_argument('--out-dir', required=True)
    args = P.parse_args()

    ratios = parse_ratios(args.ratios)
    max_ratio = max(ratios)
    dep, wd = load_roles(args.role_file, args.role_mode)
    n_pos, pos_set = load_pos_pairs(args.pos)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'[sampler] pos={n_pos}  deposit-pool={len(dep)}  withdraw-pool={len(wd)}  '
          f'ratios={ratios}  seeds={args.seeds}')
    manifest = {'n_pos': n_pos, 'deposit_pool': int(len(dep)), 'withdraw_pool': int(len(wd)),
                'ratios': ratios, 'seeds': args.seeds, 'role_mode': args.role_mode,
                'candidate_space': int(len(dep)) * int(len(wd)), 'files': []}

    for s in range(args.seeds):
        n_max = max_ratio * n_pos
        draw = sample_one_seed(dep, wd, pos_set, n_max, seed=1000 + s)
        for r in ratios:
            k = r * n_pos
            sub = draw[:k]                      # nested prefix
            df = pd.DataFrame(sub, columns=['nodeid1', 'nodeid2'])
            df['label'] = 0
            fn = os.path.join(args.out_dir, f'neg_r{r}_s{s}.csv')
            df.to_csv(fn)                        # leading unnamed index col -> 'Unnamed: 0'
            manifest['files'].append({'ratio': r, 'seed': s, 'n_neg': k, 'file': os.path.basename(fn)})
        print(f'  seed {s}: drew {n_max} unique (dep,wd) negatives -> sliced into {len(ratios)} ratio files')

    json.dump(manifest, open(os.path.join(args.out_dir, 'manifest.json'), 'w'), indent=2)
    print(f'[sampler] wrote {len(manifest["files"])} neg files + manifest.json to {args.out_dir}')


if __name__ == '__main__':
    main()
