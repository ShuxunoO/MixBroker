#!/bin/bash
set -e
PY=/Shuxun/APP/Miniconda3/envs/mixbroker/bin/python
F42=./Dataset/Graph/node_feature_normalized.csv
F82=./Dataset/Graph/node_feature_82d_normalized.csv
D42='["node","value_d","value_w","avg_value_d","avg_value_w"]'
D82='["node"]'
for bb in gcn gat; do
  echo "=== $bb 42d ==="; $PY ablation_run.py --backbone $bb --feat $F42 --drop "$D42" --out results_${bb}_42d > logs_${bb}_42d.log 2>&1
  grep MEANS logs_${bb}_42d.log
done
for bb in sage gcn gat; do
  echo "=== $bb 82d ==="; $PY ablation_run.py --backbone $bb --feat $F82 --drop "$D82" --out results_${bb}_82d > logs_${bb}_82d.log 2>&1
  grep MEANS logs_${bb}_82d.log
done
echo ALL_DONE
