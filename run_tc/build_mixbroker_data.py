# coding=utf-8
"""Convert PyG export artifacts -> MixBroker Dataset/Graph CSV format.

Produces, in the MixBroker repo's exact schema:
  - node_feature_normalized.csv : nodeid, node, <46 features> (StandardScaler z-scored, as feature_extract.normalize_handle)
  - train_pos_edge_10fold.csv   : ,nodeid1,nodeid2,label  (y==1)
  - train_neg_edge_10fold.csv   : ,nodeid1,nodeid2,label  (y==0)

42d used by MixBroker (after it drops node,value_d,value_w,avg_value_d,avg_value_w)
maps 1:1 onto the PyG export's 42 behavioural features.
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing

PYG = "/Shuxun/AML_for_Blockchain/TODO/nebula_graph_artifacts/pyg"
OUT = "/Shuxun/RelatedWork/MixBroker/run_tc/Dataset/Graph"

# --- MixBroker node_feature.csv column order (46 features) ---
COUNT = ['0.1_num','1_num','10_num','100_num','num_all','0.1_num_d','0.1_num_w',
         '1_num_d','1_num_w','10_num_d','10_num_w','100_num_d','100_num_w','num_d_all','num_w_all']
VALUE = ['value_d','value_w','avg_value_d','avg_value_w']
GAS   = ['min_gasprice_all','max_gasprice_all','avg_gasprice_all','min_gasprice_d','max_gasprice_d',
         'avg_gasprice_d','min_gasprice_w','max_gasprice_w','avg_gasprice_w']
TIME  = ['early_time','late_time','total_time_gap','min_time_gap','max_time_gap','avg_time_gap',
         'early_time_d','late_time_d','total_time_gap_d','min_time_gap_d','max_time_gap_d','avg_time_gap_d',
         'early_time_w','late_time_w','total_time_gap_w','min_time_gap_w','max_time_gap_w','avg_time_gap_w']
FEATS = COUNT + VALUE + GAS + TIME  # 46

# PyG name -> MixBroker name (only the count block differs; gas/time names already match)
PYG_COUNT = {'0.1_num':'pool_0_1_num','1_num':'pool_1_num','10_num':'pool_10_num','100_num':'pool_100_num',
             'num_all':'num_all','0.1_num_d':'pool_0_1_num_d','0.1_num_w':'pool_0_1_num_w',
             '1_num_d':'pool_1_num_d','1_num_w':'pool_1_num_w','10_num_d':'pool_10_num_d','10_num_w':'pool_10_num_w',
             '100_num_d':'pool_100_num_d','100_num_w':'pool_100_num_w','num_d_all':'num_d_all','num_w_all':'num_w_all'}

print("Loading nodes_features.parquet ...")
need = ['node_id','vid','is_seed'] + list(PYG_COUNT.values()) + GAS + TIME
nf = pd.read_parquet(f"{PYG}/nodes_features.parquet", columns=need)
print("  total rows:", len(nf))
seeds = nf[nf['is_seed'] == 1].copy()
print("  seed rows :", len(seeds))

out = pd.DataFrame()
out['nodeid'] = seeds['node_id'].astype('int64')
out['node'] = seeds['vid'].str.replace('addr:', '', regex=False)

# count features (renamed)
for mb, pg in PYG_COUNT.items():
    out[mb] = seeds[pg].astype('float64')
# value features computed exactly like feature_extract.feature_extract_value (dropped before training)
out['value_d'] = 0.1*out['0.1_num_d'] + 1*out['1_num_d'] + 10*out['10_num_d'] + 100*out['100_num_d']
out['value_w'] = 0.1*out['0.1_num_w'] + 1*out['1_num_w'] + 10*out['10_num_w'] + 100*out['100_num_w']
out['avg_value_d'] = np.where(out['num_d_all']>0, out['value_d']/out['num_d_all'].replace(0,np.nan), 0.0)
out['avg_value_w'] = np.where(out['num_w_all']>0, out['value_w']/out['num_w_all'].replace(0,np.nan), 0.0)
# gas + time (names already match)
for c in GAS + TIME:
    out[c] = seeds[c].astype('float64')

# --- normalize exactly as feature_extract.normalize_handle: StandardScaler over all feature cols ---
feat = out[FEATS].copy()
feat.replace([np.inf, -np.inf], np.nan, inplace=True)
scaled = preprocessing.StandardScaler().fit_transform(feat)
norm = pd.DataFrame(scaled, columns=FEATS)
norm.insert(0, 'node', out['node'].values)
norm.insert(0, 'nodeid', out['nodeid'].values)
norm.to_csv(f"{OUT}/node_feature_normalized.csv", index=False)
print("  wrote node_feature_normalized.csv:", norm.shape)
seed_ids = set(out['nodeid'].tolist())

# --- labels -> pos/neg edge files ---
lab = pd.read_parquet(f"{PYG}/labels.parquet", columns=['node_id_a','node_id_b','y'])
def miss(df):
    return int((~df['node_id_a'].isin(seed_ids) | ~df['node_id_b'].isin(seed_ids)).sum())
pos = lab[lab['y']==1][['node_id_a','node_id_b','y']].reset_index(drop=True)
neg = lab[lab['y']==0][['node_id_a','node_id_b','y']].reset_index(drop=True)
print(f"  pos edges: {len(pos)} (endpoints missing from node set: {miss(pos)})")
print(f"  neg edges: {len(neg)} (endpoints missing from node set: {miss(neg)})")
pos.columns = ['nodeid1','nodeid2','label']
neg.columns = ['nodeid1','nodeid2','label']
pos.to_csv(f"{OUT}/train_pos_edge_10fold.csv")   # default index -> 'Unnamed: 0'
neg.to_csv(f"{OUT}/train_neg_edge_10fold.csv")
print("  wrote train_pos_edge_10fold.csv / train_neg_edge_10fold.csv")
print("DONE")
