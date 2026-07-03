#!/bin/bash
# coding=utf-8
# Stage-A tool 3/3 : full experiment driver.
#   1. sample (deposit->withdraw) negatives for exp1 (native) & exp2 (TC)
#   2. exp1: native 42d ratio sweep 1:1..1:10 x5 seeds
#   3. exp2: TC 42d + 82d ratio sweep 1:1..1:10 x5 seeds (shared negatives)
#   4. anchor: native ORIGINAL unconstrained 1:1 (published-result reference)
# Faithful MixBroker semantics; only negative ratio varies. CPU only.

PY=/Shuxun/APP/Miniconda3/envs/mixbroker/bin/python
BASE=/Shuxun/RelatedWork/MixBroker
OUT=$BASE/output/7.3
CODE=$OUT/_code
export OMP_NUM_THREADS=8

D42='["node","value_d","value_w","avg_value_d","avg_value_w"]'
D82='["node"]'
RATIOS=1-10
SEEDS=5

say(){ echo; echo "======== $(date '+%H:%M:%S') $* ========"; }

say "STEP 1a  sample negatives : exp1 native (count-mode roles, pool=68,419)"
$PY $CODE/sample_negatives.py \
  --pos $BASE/Dataset/Graph/train_pos_edge_10fold.csv \
  --role-file $BASE/Dataset/Graph/node_feature.csv --role-mode count \
  --ratios $RATIOS --seeds $SEEDS --out-dir $OUT/negatives/exp1_native

say "STEP 1b  sample negatives : exp2 TC (flag-mode roles, pool=155,204)"
$PY $CODE/sample_negatives.py \
  --pos $BASE/run_tc/Dataset/Graph/train_pos_edge_10fold.csv \
  --role-file $BASE/run_tc/Dataset/Graph/node_feature_82d_normalized.csv --role-mode flag \
  --ratios $RATIOS --seeds $SEEDS --out-dir $OUT/negatives/exp2_tc

say "STEP 2   EXP1  native 42d  (N=68,419)  50 ten-fold runs"
$PY $CODE/sweep_runner.py --backbone sage \
  --feat $BASE/Dataset/Graph/node_feature_normalized.csv --drop "$D42" \
  --pos $BASE/Dataset/Graph/train_pos_edge_10fold.csv \
  --neg-dir $OUT/negatives/exp1_native --ratios $RATIOS --seeds $SEEDS \
  --out-dir $OUT/exp1_native_ratio_sweep --tag exp1_native_42d \
  --agg $OUT/aggregate/exp1_sweep.csv

say "STEP 3a  EXP2  TC 42d  (N=155,204)  50 ten-fold runs"
$PY $CODE/sweep_runner.py --backbone sage \
  --feat $BASE/run_tc/Dataset/Graph/node_feature_normalized.csv --drop "$D42" \
  --pos $BASE/run_tc/Dataset/Graph/train_pos_edge_10fold.csv \
  --neg-dir $OUT/negatives/exp2_tc --ratios $RATIOS --seeds $SEEDS \
  --out-dir $OUT/exp2_tc_generality/sage_42d --tag exp2_tc_42d \
  --agg $OUT/aggregate/exp2_sweep.csv

say "STEP 3b  EXP2  TC 82d  (N=155,204)  50 ten-fold runs"
$PY $CODE/sweep_runner.py --backbone sage \
  --feat $BASE/run_tc/Dataset/Graph/node_feature_82d_normalized.csv --drop "$D82" \
  --pos $BASE/run_tc/Dataset/Graph/train_pos_edge_10fold.csv \
  --neg-dir $OUT/negatives/exp2_tc --ratios $RATIOS --seeds $SEEDS \
  --out-dir $OUT/exp2_tc_generality/sage_82d --tag exp2_tc_82d \
  --agg $OUT/aggregate/exp2_sweep.csv

say "STEP 4   ANCHOR  native ORIGINAL unconstrained 1:1 (published reference)"
mkdir -p $OUT/negatives/anchor_native $OUT/exp1_native_ratio_sweep/anchor
cp $BASE/Dataset/Graph/train_neg_edge_10fold.csv $OUT/negatives/anchor_native/neg_r1_s0.csv
$PY $CODE/sweep_runner.py --backbone sage \
  --feat $BASE/Dataset/Graph/node_feature_normalized.csv --drop "$D42" \
  --pos $BASE/Dataset/Graph/train_pos_edge_10fold.csv \
  --neg-dir $OUT/negatives/anchor_native --ratios 1 --seeds 1 \
  --out-dir $OUT/exp1_native_ratio_sweep/anchor --tag native_original_1to1 \
  --agg $OUT/aggregate/anchor.csv

say "ALL DONE"
