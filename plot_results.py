"""三张图：主表对比、树深对照、正则强度扫描。

    python plot_results.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------- 图 1：四个数据集上的对比 ----------
rows = read("results/main.csv")
names = [r["dataset"] for r in rows]
series = [
    ("CART", [float(r["cart_acc"]) for r in rows], "#9aa5b1"),
    ("Random Forest", [float(r["rf_acc"]) for r in rows], "#7fb3d5"),
    ("RADDT", [float(r["raddt_acc"]) for r in rows], "#4c72b0"),
    ("Sparse-RADDT", [float(r["sparse_acc"]) for r in rows], "#c44e52"),
]
x = np.arange(len(names))
w = 0.2
plt.figure(figsize=(8, 4.2))
for i, (label, vals, color) in enumerate(series):
    plt.bar(x + (i - 1.5) * w, vals, w, label=label, color=color)
plt.xticks(x, [f"{r['dataset']}\n({r['n_classes']} classes, depth {r['depth']})" for r in rows],
           fontsize=8)
plt.ylabel("5-fold CV accuracy")
plt.ylim(0.6, 1.02)
plt.title("RADDT vs baselines (leaves >= classes)")
plt.grid(axis="y", alpha=0.25)
plt.legend(fontsize=8, ncol=4, loc="lower center")
plt.tight_layout()
plt.savefig("figures/accuracy.png", dpi=160)
plt.close()

# ---------- 图 2：树深对照 ----------
rows = read("results/depth_study.csv")
depths = [int(r["depth"]) for r in rows]
accs = [float(r["accuracy"]) for r in rows]
stds = [float(r["std"]) for r in rows]

plt.figure(figsize=(5, 3.8))
plt.errorbar(depths, accs, yerr=stds, marker="o", lw=2, capsize=4, color="#4c72b0")
for d, a in zip(depths, accs):
    plt.annotate(f"{a:.3f}", (d, a), textcoords="offset points", xytext=(0, 10),
                 ha="center", fontsize=8)
plt.axhline(0.979, ls=":", lw=1, color="gray")
plt.text(2.02, 0.983, "Random Forest (0.979)", fontsize=7, color="gray")
plt.xlabel("tree depth (leaves = 2^depth)")
plt.ylabel("accuracy on Digits (10 classes)")
plt.xticks(depths, [f"{d}\n{2**d} leaves" for d in depths])
plt.ylim(0.3, 1.02)
plt.title("Leaves must be >= classes")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("figures/depth_study.png", dpi=160)
plt.close()

# ---------- 图 3：正则强度 ----------
rows = read("results/lambda_sweep.csv")
lams = [float(r["l1_lambda"]) for r in rows]
accs = [float(r["accuracy"]) for r in rows]
spars = [float(r["sparsity"]) for r in rows]

fig, ax1 = plt.subplots(figsize=(5.4, 3.8))
ax1.plot(lams, accs, marker="o", lw=2, color="#4c72b0", label="accuracy")
ax1.set_xlabel("L1 strength λ")
ax1.set_ylabel("accuracy (Digits)", color="#4c72b0")
ax1.set_ylim(0.78, 0.94)
ax2 = ax1.twinx()
ax2.plot(lams, spars, marker="s", lw=1.6, ls="--", color="#c44e52", label="sparsity")
ax2.set_ylabel("weight sparsity", color="#c44e52")
ax2.set_ylim(0, 0.12)
plt.title("L1 strength: too strong hurts, 1e-3 is sweet")
plt.tight_layout()
plt.savefig("figures/lambda_sweep.png", dpi=160)
plt.close()

print("saved figures/accuracy.png, figures/depth_study.png, figures/lambda_sweep.png")
