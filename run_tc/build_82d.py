# coding=utf-8
"""Build 82d node feature file for the seed nodes, MixBroker-style:
take the PyG export's 82 modelling features (4 identity + 42 behavioural + 36
structural), StandardScaler them (same normalisation as the 42d experiment),
write nodeid, node, <82 cols>. Edge files are reused from the 42d build."""
import pandas as pd, numpy as np
from sklearn import preprocessing

PYG = "/Shuxun/AML_for_Blockchain/TODO/nebula_graph_artifacts/pyg"
OUT = "/Shuxun/RelatedWork/MixBroker/run_tc/Dataset/Graph"

IDENT = ['is_seed','is_deposit_seed','is_withdraw_seed','is_tornado_fixed']
BEHAV = ['pool_0_1_num','pool_1_num','pool_10_num','pool_100_num','num_all','pool_0_1_num_d','pool_0_1_num_w',
         'pool_1_num_d','pool_1_num_w','pool_10_num_d','pool_10_num_w','pool_100_num_d','pool_100_num_w','num_d_all','num_w_all',
         'early_time','late_time','total_time_gap','min_time_gap','max_time_gap','avg_time_gap',
         'early_time_d','late_time_d','total_time_gap_d','min_time_gap_d','max_time_gap_d','avg_time_gap_d',
         'early_time_w','late_time_w','total_time_gap_w','min_time_gap_w','max_time_gap_w','avg_time_gap_w',
         'min_gasprice_all','max_gasprice_all','avg_gasprice_all','min_gasprice_d','max_gasprice_d',
         'avg_gasprice_d','min_gasprice_w','max_gasprice_w','avg_gasprice_w']
STRUCT = ['deg_out','deg_in','deg_total','tx_count','out_in_ratio','distinct_seeds','distinct_deposit_seeds',
          'distinct_withdraw_seeds','is_bridge','bridge_balance','val_out_sum','val_out_mean','val_out_max',
          'val_in_sum','val_in_mean','val_in_max','val_total_sum','val_in_out_ratio','asset_eth','asset_usdt',
          'asset_weth','asset_usdc','asset_dai','asset_wbtc','asset_torn','cat_external','cat_internal','cat_erc20',
          'pool_0_1','pool_1','pool_10','pool_100','first_block','last_block','active_span_blocks','active_days']
FEATS = IDENT + BEHAV + STRUCT
assert len(FEATS) == 82, len(FEATS)

nf = pd.read_parquet(f"{PYG}/nodes_features.parquet", columns=['node_id','vid','is_seed'] + [c for c in FEATS if c != 'is_seed'])
seeds = nf[nf['is_seed'] == 1].copy()
print("seed rows:", len(seeds))

out = pd.DataFrame()
out['nodeid'] = seeds['node_id'].astype('int64')
out['node'] = seeds['vid'].str.replace('addr:', '', regex=False)
feat = seeds[FEATS].astype('float64').copy()
feat.replace([np.inf, -np.inf], np.nan, inplace=True)
scaled = preprocessing.StandardScaler().fit_transform(feat)
norm = pd.DataFrame(scaled, columns=FEATS)
norm.insert(0, 'node', out['node'].values)
norm.insert(0, 'nodeid', out['nodeid'].values)
norm.to_csv(f"{OUT}/node_feature_82d_normalized.csv", index=False)
print("wrote node_feature_82d_normalized.csv", norm.shape)
