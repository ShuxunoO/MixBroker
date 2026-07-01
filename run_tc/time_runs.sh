#!/bin/bash
PY=/Shuxun/APP/Miniconda3/envs/mixbroker/bin/python
echo "### SAGE 42d (subset, what we ran) ###"
/usr/bin/time -v $PY ablation_run.py --backbone sage --feat ./Dataset/Graph/node_feature_42d_sub.csv \
  --drop '["node","value_d","value_w","avg_value_d","avg_value_w"]' --out /tmp/t_sage42 2>&1 | grep -E "MEANS|time |Maximum resident|Elapsed|Percent of CPU"
echo "### SAGE 82d (subset) ###"
/usr/bin/time -v $PY ablation_run.py --backbone sage --feat ./Dataset/Graph/node_feature_82d_sub.csv \
  --drop '["node"]' --out /tmp/t_sage82 2>&1 | grep -E "MEANS|time |Maximum resident|Elapsed|Percent of CPU"
echo "### SAGE 42d FULL 155k nodes (for memory contrast) ###"
/usr/bin/time -v $PY ablation_run.py --backbone sage --feat ./Dataset/Graph/node_feature_normalized.csv \
  --drop '["node","value_d","value_w","avg_value_d","avg_value_w"]' --out /tmp/t_sage42full 2>&1 | grep -E "MEANS|time |Maximum resident|Elapsed|Percent of CPU"
echo TIMING_DONE
