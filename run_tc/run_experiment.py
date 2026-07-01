# coding=utf-8
"""Thin runner: invokes the UNCHANGED repo training (main_cpu.train_10_fold)
with the repo's exact settings (seed 1029, KFold-10 rs=42, SAGE 32->16,
Adam lr0.01, 100 epochs, thr 0.75, best-by-F1). Only adds result persistence."""
import json
import main_cpu

main_cpu.seed_torch(1029)
df = main_cpu.train_10_fold()

df = df.reset_index(drop=True)
df.insert(0, 'fold', range(1, len(df) + 1))
df.to_csv('result_per_fold.csv', index=False)

cols = ['test_accuracy','test_precision','test_recall','test_f_1','test_FPR','test_FNR','test_auc']
summary = {c: {'mean': float(df[c].mean()), 'std': float(df[c].std())} for c in cols}
with open('result_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("SAVED result_per_fold.csv + result_summary.json")
