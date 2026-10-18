"""主实验：5 折交叉验证，比较 CART / Random Forest / RADDT / Sparse-RADDT。

关键设计（相对第一版的修正）：
- 树深按 leaves >= classes 设定，而不是所有数据集统一用一个深度；
- RADDT 与 Sparse-RADDT 除 l1_lambda 外所有超参完全一致（同深、同种子）；
- 显式传参，保证 sklearn clone() 不会丢参数；
- 同时记录真实稀疏度（|A| < 1e-6 的比例），不靠可视化阈值。

    python run_experiment.py
"""
import csv
import math
import os
import sys
import time

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from raddt import RADDTClassifier

SEED = 42
N_FOLDS = 5
SPARSE_LAMBDA = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
DATASETS = {
    "Iris": load_iris(),
    "Wine": load_wine(),
    "BreastCancer": load_breast_cancer(),
    "Digits": load_digits(),
}


def depth_for(n_classes):
    return max(3, math.ceil(math.log2(n_classes)))


def main():
    os.makedirs("results", exist_ok=True)
    rows = []

    for name, ds in DATASETS.items():
        X, y = ds.data, ds.target
        n_classes = len(np.unique(y))
        depth = depth_for(n_classes)
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        print(f"\n{name}: n={len(y)} dim={X.shape[1]} classes={n_classes} depth={depth}", flush=True)
        row = {"dataset": name, "n_samples": len(y), "n_features": X.shape[1],
               "n_classes": n_classes, "depth": depth}

        t0 = time.time()
        cart = DecisionTreeClassifier(random_state=SEED)
        s = cross_val_score(cart, X, y, cv=cv, scoring="accuracy")
        row.update(cart_acc=s.mean(), cart_std=s.std(), cart_sec=time.time() - t0)

        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
        s = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
        row.update(rf_acc=s.mean(), rf_std=s.std(), rf_sec=time.time() - t0)

        for tag, lam in [("raddt", 0.0), ("sparse", SPARSE_LAMBDA)]:
            t0 = time.time()
            model = RADDTClassifier(depth=depth, lr=0.01, epochs=150,
                                    alpha_min=2.0, alpha_max=200.0,
                                    n_alpha_stages=4, n_restarts=2,
                                    l1_lambda=lam, seed=SEED)
            # clone 之后参数必须还在（这就是第一版踩的坑）
            cloned = model.get_params()
            assert cloned["n_alpha_stages"] == 4 and cloned["n_restarts"] == 2, "clone 丢参数"

            s = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            row.update({f"{tag}_acc": s.mean(), f"{tag}_std": s.std(), f"{tag}_sec": time.time() - t0})

            model.fit(X, y)
            row[f"{tag}_sparsity"] = model.sparsity()

        rows.append(row)
        print(f"  CART {row['cart_acc']:.4f} | RF {row['rf_acc']:.4f} | "
              f"RADDT {row['raddt_acc']:.4f} | Sparse {row['sparse_acc']:.4f} "
              f"| sparsity {row['sparse_sparsity']:.1%}", flush=True)

    with open("results/main.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("\nsaved results/main.csv")


if __name__ == "__main__":
    main()
