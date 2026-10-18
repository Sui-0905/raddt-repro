"""可解释性与收敛性可视化（新接口版）。

生成三张图：
    figures/feature_importance.png   Wine 上 RADDT vs Sparse-RADDT 的特征重要性
    figures/decision_boundary.png    Iris（前两个特征）上的决策边界对比
    figures/convergence.png          λ 对训练收敛的影响

    python visualizations.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, load_iris, load_wine

from raddt import RADDTClassifier

os.makedirs("figures", exist_ok=True)
SEED = 42


def base(**kw):
    return RADDTClassifier(depth=3, lr=0.01, epochs=150, alpha_min=2.0, alpha_max=200.0,
                           n_alpha_stages=4, n_restarts=2, seed=SEED, **kw)


# ---------------- 1. 特征重要性 ----------------
X, y = load_wine(return_X_y=True)
plain = base(l1_lambda=0.0).fit(X, y)
sparse = base(l1_lambda=1e-3).fit(X, y)
imp_p = plain.get_feature_importance()
imp_s = sparse.get_feature_importance()
order = np.argsort(-imp_p)[:10]

fig, ax = plt.subplots(figsize=(7, 3.6))
idx = np.arange(len(order))
ax.bar(idx - 0.2, imp_p[order], 0.4, label="RADDT", color="#4c72b0")
ax.bar(idx + 0.2, imp_s[order], 0.4, label="Sparse-RADDT", color="#c44e52")
ax.set_xticks(idx)
ax.set_xticklabels([f"f{i}" for i in order], fontsize=8)
ax.set_ylabel("normalized importance (|a| summed over nodes)")
ax.set_title("Wine: L1 concentrates and removes feature weight")
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=160)
plt.close()
print("saved figures/feature_importance.png", "sparsity", f"{sparse.sparsity():.1%}")

# ---------------- 2. 决策边界 ----------------
X, y = load_iris(return_X_y=True)
X2 = X[:, :2]
plain2 = base(l1_lambda=0.0).fit(X2, y)
sparse2 = base(l1_lambda=1e-3).fit(X2, y)

pad = 0.5
xx, yy = np.meshgrid(
    np.linspace(X2[:, 0].min() - pad, X2[:, 0].max() + pad, 200),
    np.linspace(X2[:, 1].min() - pad, X2[:, 1].max() + pad, 200),
)
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
for ax, model, title in [(axes[0], plain2, "RADDT"), (axes[1], sparse2, "Sparse-RADDT")]:
    z = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, z, alpha=0.25, cmap="coolwarm")
    ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap="coolwarm", s=12, edgecolor="k", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("sepal length")
axes[0].set_ylabel("sepal width")
plt.suptitle("Decision boundary on Iris (first two features)", fontsize=11)
plt.tight_layout()
plt.savefig("figures/decision_boundary.png", dpi=160)
plt.close()
print("saved figures/decision_boundary.png")

# ---------------- 3. 收敛曲线 ----------------
X, y = load_digits(return_X_y=True)
plt.figure(figsize=(6.4, 3.6))
for lam, color in [(0.0, "#4c72b0"), (1e-3, "#c44e52")]:
    model = RADDTClassifier(depth=4, lr=0.01, epochs=150, alpha_min=2.0, alpha_max=200.0,
                            n_alpha_stages=4, n_restarts=2, l1_lambda=lam, seed=SEED)
    model.fit(X, y)
    hist = np.array(model.history_)
    smooth = np.convolve(hist, np.ones(15) / 15, mode="valid")
    plt.plot(smooth, lw=1.6, color=color, label=f"λ={lam:g}")
plt.xlabel("training epoch (all annealing stages concatenated)")
plt.ylabel("training loss (moving average)")
plt.title("Digits: L1 slows early convergence but reaches lower loss")
plt.legend(fontsize=8)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("figures/convergence.png", dpi=160)
plt.close()
print("saved figures/convergence.png")
