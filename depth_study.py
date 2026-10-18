"""深度对照实验：固定其余因素，只改树深。

这是对"修正树深带来提升"这个说法的受控检验：
同一个数据集（Digits，10 类）、同一个 l1_lambda（0）、同一个随机种子，
只让 depth 从 2 变到 4。

    python depth_study.py
"""
import csv
import os
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score

from raddt import RADDTClassifier

X, y = load_digits(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows = []
for depth in (2, 3, 4):
    t0 = time.time()
    model = RADDTClassifier(depth=depth, lr=0.01, epochs=150,
                            alpha_min=2.0, alpha_max=200.0,
                            n_alpha_stages=4, n_restarts=2,
                            l1_lambda=0.0, seed=42)
    s = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    rows.append({"depth": depth, "leaves": 2 ** depth,
                 "accuracy": s.mean(), "std": s.std(), "sec": time.time() - t0})
    print(f"depth={depth} leaves={2**depth} acc={s.mean():.4f}±{s.std():.4f}", flush=True)

os.makedirs("results", exist_ok=True)
with open("results/depth_study.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
print("saved results/depth_study.csv")
