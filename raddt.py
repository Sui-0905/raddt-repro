"""RADDT: ReLU + Argmin 重构的可微分决策树，附 L1 稀疏变体。

实现要点（相对常见复现坑的修正）：
1. 所有超参数都写成 __init__ 的显式参数，不用 **kwargs——
   否则 sklearn 的 clone() 会静默丢掉它们，交叉验证时参数根本不会生效。
2. 每次 restart 前固定 torch 随机种子，结果可复现。
3. predict 区分分类/回归，回归返回连续值而不是 argmax。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler


class RADDTBase(BaseEstimator):
    def __init__(self, depth=3, lr=0.01, epochs=150, alpha_min=2.0, alpha_max=200.0,
                 n_alpha_stages=4, n_restarts=2, l1_lambda=0.0, adaptive_l1=False,
                 seed=0, device="cpu"):
        self.depth = depth
        self.lr = lr
        self.epochs = epochs
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.n_alpha_stages = n_alpha_stages
        self.n_restarts = n_restarts
        self.l1_lambda = l1_lambda
        self.adaptive_l1 = adaptive_l1
        self.seed = seed
        self.device = device
        self._build_tree_structure()

    # ---------------- 树结构 ----------------
    def _build_tree_structure(self):
        self.n_leaves = 2 ** self.depth
        self.n_internal = self.n_leaves - 1
        self.leaf_ancestors, self.leaf_directions = [], []
        leaf_start = self.n_leaves - 1
        for leaf in range(self.n_leaves):
            node, anc, dire = leaf_start + leaf, [], []
            while node > 0:
                parent = (node - 1) // 2
                anc.append(parent)
                dire.append("left" if node % 2 == 1 else "right")
                node = parent
            self.leaf_ancestors.append(anc)
            self.leaf_directions.append(dire)

        left = torch.zeros(self.n_leaves, self.n_internal)
        right = torch.zeros(self.n_leaves, self.n_internal)
        for t in range(self.n_leaves):
            for a, d in zip(self.leaf_ancestors[t], self.leaf_directions[t]):
                (left if d == "left" else right)[t, a] = 1.0
        self.left_mask, self.right_mask = left, right

    # ---------------- 前向 ----------------
    def _violations(self, X, A, b):
        scores = X @ A.T - b.unsqueeze(0)
        left = F.relu(scores)
        right = F.relu(-scores)
        U = (left.unsqueeze(1) * self.left_mask.to(X.device).unsqueeze(0)
             + right.unsqueeze(1) * self.right_mask.to(X.device).unsqueeze(0))
        return U.sum(dim=2)                     # (B, n_leaves)

    def _loss(self, U, y, leaf_preds, A, task, alpha):
        routing = F.softmax(-alpha * U, dim=1)  # scaling softmin
        if task == "classification":
            pred = routing @ leaf_preds
            base = F.cross_entropy(pred, y)
        else:
            pred = (routing * leaf_preds.squeeze(-1)).sum(dim=1)
            base = F.mse_loss(pred, y.float())
        if self.l1_lambda > 0:
            lam = self.l1_lambda
            if self.adaptive_l1:
                # λ = λ0 · ln(1+p) / sqrt(p)：维度越高，正则越强，但被 ln 拉住不过度
                p = A.shape[1]
                lam = self.l1_lambda * math.log(1 + p) / math.sqrt(p)
            base = base + lam * A.abs().sum()
        return base

    # ---------------- 训练 ----------------
    def fit(self, X, y, task="classification"):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        n_features = X.shape[1]
        self.task_ = task

        if task == "classification":
            n_out = int(np.max(y)) + 1
            y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)
        else:
            n_out = 1
            y = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=self.device)

        alphas = np.logspace(math.log10(self.alpha_min), math.log10(self.alpha_max),
                             self.n_alpha_stages)
        best_loss, best = float("inf"), None
        self.history_ = []

        for restart in range(self.n_restarts):
            torch.manual_seed(self.seed * 1000 + restart)
            A = nn.Parameter(torch.randn(self.n_internal, n_features, device=self.device) * 0.1)
            b = nn.Parameter(torch.randn(self.n_internal, device=self.device) * 0.1)
            leaf = nn.Parameter(torch.randn(self.n_leaves, n_out, device=self.device) * 0.1)

            for alpha in alphas:
                opt = torch.optim.Adam([A, b, leaf], lr=self.lr)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
                for _ in range(self.epochs):
                    opt.zero_grad()
                    U = self._violations(X, A, b)
                    loss = self._loss(U, y, leaf, A, task, alpha)
                    loss.backward()
                    opt.step()
                    sched.step()
                    self.history_.append(loss.item())

            with torch.no_grad():
                U = self._violations(X, A, b)
                hard = U.argmin(dim=1)
                exact = self._exact_loss(X, y, hard, leaf, task)
            if exact < best_loss:
                best_loss = exact
                best = (A.detach().clone(), b.detach().clone(), leaf.detach().clone())

        self.A, self.b, self.leaf_preds = [nn.Parameter(t) for t in best]
        return self

    def _exact_loss(self, X, y, hard, leaf, task):
        total, n = 0.0, len(y)
        for t in range(self.n_leaves):
            mask = hard == t
            if mask.sum() == 0:
                continue
            if task == "classification":
                total += F.cross_entropy(
                    leaf[t].unsqueeze(0).expand(int(mask.sum()), -1), y[mask]
                ).item() * int(mask.sum())
            else:
                total += F.mse_loss(
                    leaf[t].squeeze().expand(int(mask.sum())), y[mask]
                ).item() * int(mask.sum())
        return total / n

    # ---------------- 推理 ----------------
    def _forward(self, X):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            U = self._violations(X, self.A, self.b)
            return U.argmin(dim=1).cpu().numpy()

    def predict(self, X):
        leaf_idx = self._forward(X)
        if getattr(self, "task_", "classification") == "regression":
            return self.leaf_preds.squeeze(-1).detach().cpu().numpy()[leaf_idx]
        classes = self.leaf_preds.argmax(dim=1).cpu().numpy()
        return classes[leaf_idx]

    def predict_proba(self, X):
        leaf_idx = self._forward(X)
        proba = F.softmax(self.leaf_preds, dim=1).cpu().numpy()
        return proba[leaf_idx]

    # ---------------- 诊断 ----------------
    def get_feature_importance(self):
        imp = self.A.abs().sum(dim=0).detach().cpu().numpy()
        return imp / imp.sum()

    def sparsity(self, tol=1e-6):
        """分裂权重中绝对值小于 tol 的比例（真实测出来的，不是阈值拍脑袋）。"""
        return float((self.A.abs() < tol).float().mean().item())


class RADDTClassifier(ClassifierMixin, RADDTBase):
    def __init__(self, depth=3, lr=0.01, epochs=150, alpha_min=2.0, alpha_max=200.0,
                 n_alpha_stages=4, n_restarts=2, l1_lambda=0.0, adaptive_l1=False,
                 seed=0, device="cpu"):
        super().__init__(depth, lr, epochs, alpha_min, alpha_max,
                         n_alpha_stages, n_restarts, l1_lambda, adaptive_l1, seed, device)

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(np.asarray(y))
        self.scaler_ = StandardScaler().fit(X)
        super().fit(self.scaler_.transform(X), self.encoder_.transform(y), task="classification")
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(super().predict(self.scaler_.transform(X)))


class RADDTRegressor(RegressorMixin, RADDTBase):
    def __init__(self, depth=3, lr=0.01, epochs=150, alpha_min=2.0, alpha_max=200.0,
                 n_alpha_stages=4, n_restarts=2, l1_lambda=0.0, adaptive_l1=False,
                 seed=0, device="cpu"):
        super().__init__(depth, lr, epochs, alpha_min, alpha_max,
                         n_alpha_stages, n_restarts, l1_lambda, adaptive_l1, seed, device)

    def fit(self, X, y):
        self.scaler_X_ = StandardScaler().fit(X)
        y = np.asarray(y, dtype=float).ravel()
        self.scaler_y_ = StandardScaler().fit(y.reshape(-1, 1))
        super().fit(self.scaler_X_.transform(X),
                    self.scaler_y_.transform(y.reshape(-1, 1)).ravel(), task="regression")
        return self

    def predict(self, X):
        y = super().predict(self.scaler_X_.transform(X))
        return self.scaler_y_.inverse_transform(np.asarray(y).reshape(-1, 1)).ravel()
