"""最小自检：不跑完整实验，只验证实现没坏。

    python test_raddt.py
"""
import numpy as np
import torch
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from raddt import RADDTClassifier, RADDTRegressor

CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def test_params_survive_clone():
    """曾经踩过的坑：__init__ 用 **kwargs 收参数，sklearn.clone 会静默丢掉它们。"""
    m = RADDTClassifier(n_alpha_stages=4, n_restarts=2, l1_lambda=0.01)
    p = m.get_params()
    assert p["n_alpha_stages"] == 4, p
    assert p["n_restarts"] == 2, p
    assert p["l1_lambda"] == 0.01, p


def test_sklearn_protocol():
    """回归测试：mixin 顺序写反会让 is_classifier() 返回 False，
    于是 cross_val_score(cv=3) 退化成普通 KFold；iris 按类别排序，每折整折同类，直接得 0 分。"""
    from sklearn.base import is_classifier, is_regressor

    assert is_classifier(RADDTClassifier()), "应被识别为分类器"
    assert is_regressor(RADDTRegressor()), "应被识别为回归器"

    X, y = load_iris(return_X_y=True)
    m = RADDTClassifier(depth=3, epochs=40, n_alpha_stages=3, n_restarts=1, seed=0)
    scores = cross_val_score(m, X, y, cv=3)          # 注意：整数 cv
    assert scores.mean() > 0.8, f"整数 cv 应走分层抽样，实际 {scores}"


def test_fits_iris():
    X, y = load_iris(return_X_y=True)
    m = RADDTClassifier(depth=3, epochs=40, n_alpha_stages=3, n_restarts=1, seed=0)
    scores = cross_val_score(m, X, y, cv=CV)
    assert scores.mean() > 0.85, scores


def test_regression_returns_continuous():
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=0)
    m = RADDTRegressor(depth=3, epochs=40, n_alpha_stages=3, n_restarts=1, seed=0)
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == y.shape, (pred.shape, y.shape)
    assert len(np.unique(pred)) > 2, "回归应该返回连续值，而不是几个离散类别"
    assert np.corrcoef(pred, y)[0, 1] > 0.5


def test_deterministic():
    X, y = load_iris(return_X_y=True)
    a = RADDTClassifier(depth=3, epochs=30, n_alpha_stages=3, n_restarts=1, seed=7).fit(X, y)
    b = RADDTClassifier(depth=3, epochs=30, n_alpha_stages=3, n_restarts=1, seed=7).fit(X, y)
    assert torch.allclose(a.A, b.A), "同一个种子应该得到同一棵树"


if __name__ == "__main__":
    for fn in [test_params_survive_clone, test_sklearn_protocol, test_fits_iris,
               test_regression_returns_continuous, test_deterministic]:
        fn()
        print(f"PASS {fn.__name__}")
