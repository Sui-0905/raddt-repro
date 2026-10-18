"""正则强度扫描：λ 取多大时，稀疏性上去了而精度没塌。

Digits 上 λ=0.01 会把准确率从 0.88 拉到 0.82，说明 0.01 太强。
这里扫三个数量级，找有没有兼顾两者的取值。

    python lambda_sweep.py
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
for lam in (0.0, 1e-4, 1e-3):
    t0 = time.time()
    model = RADDTClassifier(depth=4, lr=0.01, epochs=150,
                            alpha_min=2.0, alpha_max=200.0,
                            n_alpha_stages=4, n_restarts=2,
                            l1_lambda=lam, seed=42)
    s = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    model.fit(X, y)
    rows.append({"l1_lambda": lam, "accuracy": s.mean(), "std": s.std(),
                 "sparsity": model.sparsity(), "sec": time.time() - t0})
    print(f"lambda={lam:g} acc={s.mean():.4f}±{s.std():.4f} "
          f"sparsity={model.sparsity():.1%}", flush=True)

os.makedirs("results", exist_ok=True)
with open("results/lambda_sweep.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
print("saved results/lambda_sweep.csv")
