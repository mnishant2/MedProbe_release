"""Train probes: linear logistic regression (default) or a small MLP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedProbe:
    scaler: StandardScaler
    clf: Any

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]


def train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    max_iter: int = 2000,
    solver: str = "lbfgs",
) -> FittedProbe:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(C=C, max_iter=max_iter, solver=solver)
    clf.fit(Xs, y)
    return FittedProbe(scaler=scaler, clf=clf)


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_sizes: tuple[int, ...] = (128,),
    max_iter: int = 200,
    alpha: float = 1e-3,
    seed: int = 42,
) -> FittedProbe:
    """Small MLP probe. Default is a single 128-unit hidden layer."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        max_iter=max_iter,
        alpha=alpha,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
    )
    clf.fit(Xs, y)
    return FittedProbe(scaler=scaler, clf=clf)
