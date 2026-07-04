# coding=utf-8
"""Fixed-version prep: build the native Mixing-Interaction-Graph (MIG) edge_index
and an extended normalized feature table that includes the 4 pool nodes.

MIG = account <-> pool bipartite from ETH.csv (account 'From' -> pool 'To').
Two accounts that used the same pool become 2-hop neighbours through the pool
hub -> this is the message-passing graph the model SHOULD use, decoupled from
the linkage edges being predicted (removes edge_index==edge_label_index leakage).

Outputs (to output/7.3/mig/):
  node_feature_normalized_mig.csv : original 68,419 rows + 4 pool rows (features=0
                                    i.e. the mean in z-scored space)
  mig_edge_index.npy              : int64 [2, E] bidirectional account<->pool edges
"""
import os
import numpy as np
import pandas as pd

BASE = '/Shuxun/RelatedWork/MixBroker'
OUT = f'{BASE}/output/7.3/mig'
os.makedirs(OUT, exist_ok=True)

# --- address -> nodeid map (raw table has nodeid, node=address) ---
nf = pd.read_csv(f'{BASE}/Dataset/Graph/node_feature.csv', usecols=['nodeid', 'node'])
addr2id = dict(zip(nf['node'], nf['nodeid'].astype(int)))
n_acc = int(nf['nodeid'].max()) + 1
print(f'accounts: {n_acc}')

# --- ETH.csv: account(From) -> pool(To) ---
eth = pd.read_csv(f'{BASE}/Dataset/Graph/ETH.csv', usecols=['From', 'To'])
pools = sorted(eth['To'].unique())
print(f'pools ({len(pools)}):', [p[:10] + '..' for p in pools])
pool2id = {p: n_acc + i for i, p in enumerate(pools)}   # ids 68419..68422

src = eth['From'].map(addr2id)
dst = eth['To'].map(pool2id)
keep = src.notna() & dst.notna()
src = src[keep].astype(np.int64).to_numpy()
dst = dst[keep].astype(np.int64).to_numpy()
# unique account<->pool pairs, then make bidirectional
pairs = np.unique(np.stack([src, dst], axis=1), axis=0)
a, p = pairs[:, 0], pairs[:, 1]
ei = np.concatenate([np.stack([a, p]), np.stack([p, a])], axis=1).astype(np.int64)
np.save(f'{OUT}/mig_edge_index.npy', ei)
print(f'unique account-pool pairs: {len(pairs)}  ->  bidirectional edges: {ei.shape[1]}')

# --- extended normalized feature table: append 4 pool rows (features = 0) ---
norm = pd.read_csv(f'{BASE}/Dataset/Graph/node_feature_normalized.csv')
feat_cols = [c for c in norm.columns if c not in ('nodeid', 'node')]
pool_rows = pd.DataFrame({'nodeid': list(pool2id.values()), 'node': list(pool2id.keys())})
for c in feat_cols:
    pool_rows[c] = 0.0
ext = pd.concat([norm, pool_rows[norm.columns]], ignore_index=True)
ext.to_csv(f'{OUT}/node_feature_normalized_mig.csv', index=False)
print(f'extended feature table: {norm.shape[0]} + {len(pool_rows)} = {ext.shape[0]} rows, {len(feat_cols)} feature cols')
print('DONE ->', OUT)
