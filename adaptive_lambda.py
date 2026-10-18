"""自适应正则强度对照：固定 λ vs λ0·ln(1+p)/√p。

原论文提出了自适应方案（式 4），但第一版代码里没有实现。这里补上并做对照，
证明它不是摆设——至少在部分数据集上有差别。

    python adaptive_lambda.py
"""
import csv
import math
import os
import time

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score

from raddt import RADDTClassifier

SEED = 42
LAMBDA0 = 1e-3
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

DATASETS = {
    "Iris": load_iris(),
    "Wine": load_wine(),
    "BreastCancer": load_breast_cancer(),
    "Digits": load_digits(),
}

rows = []
for name, ds in DATASETS.items():
    X, y = ds.data, ds.target
    depth = max(3, math.ceil(math.log2(len(np.unique(y)))))
    row = {"dataset": name, "n_features": X.shape[1], "depth": depth}

    for tag, adaptive in [("fixed", False), ("adaptive", True)]:
        t0 = time.time()
        model = RADDTClassifier(depth=depth, lr=0.01, epochs=150,
                                alpha_min=2.0, alpha_max=200.0,
                                n_alpha_stages=4, n_restarts=2,
                                l1_lambda=LAMBDA0, adaptive_l1=adaptive, seed=SEED)
        s = cross_val_score(model, X, y, cv=CV, scoring="accuracy")
        model.fit(X, y)
        row[f"{tag}_acc"] = s.mean()
        row[f"{tag}_sparsity"] = model.sparsity()
        row[f"{tag}_sec"] = time.time() - t0

    rows.append(row)
    print(f"{name:14s} fixed {row['fixed_acc']:.4f} ({row['fixed_sparsity']:.1%}) | "
          f"adaptive {row['adaptive_acc']:.4f} ({row['adaptive_sparsity']:.1%})", flush=True)

os.makedirs("results", exist_ok=True)
with open("results/adaptive_lambda.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
print("saved results/adaptive_lambda.csv")
