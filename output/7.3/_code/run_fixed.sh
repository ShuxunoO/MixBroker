#!/bin/bash
# coding=utf-8
# Fixed-version driver (native data). Ablation ladder V0/V1/V2 across a realistic
# imbalance sweep, to quantify how much each flaw inflated the exposed-version scores.
#   1. sample (deposit->withdraw) negatives up to 1:100
#   2. V0 = leaky  (edge_index==edge_label_index, threshold 0.75)
#   3. V1/V2 = mig (full MIG message passing; reports @0.75 AND @calibrated)
# Faithful hyper-params; model selection by test AUC-PR (uniform across variants).

PY=/Shuxun/APP/Miniconda3/envs/mixbroker/bin/python
BASE=/Shuxun/RelatedWork/MixBroker
OUT=$BASE/output/7.3
CODE=$OUT/_code
export OMP_NUM_THREADS=8
RATIOS=1,2,5,10,20,50,100
SEEDS=5

say(){ echo; echo "======== $(date '+%H:%M:%S') $* ========"; }

say "STEP 0  build MIG (idempotent)"
$PY $CODE/build_mig.py

say "STEP 1  sample negatives up to 1:100 (deposit->withdraw), pool=68,419"
$PY $CODE/sample_negatives.py \
  --pos $BASE/Dataset/Graph/train_pos_edge_10fold.csv \
  --role-file $BASE/Dataset/Graph/node_feature.csv --role-mode count \
  --ratios $RATIOS --seeds $SEEDS --out-dir $OUT/negatives/exp3_fixed_native

say "STEP 2  V0  LEAKY  (edge_index==edge_label_index, thr 0.75)  35 configs"
$PY $CODE/sweep_runner_fixed.py --graph leaky \
  --pos $BASE/Dataset/Graph/train_pos_edge_10fold.csv \
  --neg-dir $OUT/negatives/exp3_fixed_native --ratios $RATIOS --seeds $SEEDS \
  --out-dir $OUT/exp3_fixed_native/leaky --tag v0_leaky \
  --agg $OUT/aggregate/fixed_sweep.csv

say "STEP 3  V1/V2  MIG  (full MIG message passing; @0.75 and @calibrated)  35 configs"
$PY $CODE/sweep_runner_fixed.py --graph mig \
  --pos $BASE/Dataset/Graph/train_pos_edge_10fold.csv \
  --neg-dir $OUT/negatives/exp3_fixed_native --ratios $RATIOS --seeds $SEEDS \
  --out-dir $OUT/exp3_fixed_native/mig --tag mig \
  --agg $OUT/aggregate/fixed_sweep.csv

say "ALL DONE (fixed)"
