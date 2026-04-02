"""Sklearn linear probes on cached backbone activations (per-layer flattened C*H*W)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from chess_experiments.activations import layer_key_to_probe_id

from scripts.constants import SEED

if TYPE_CHECKING:
    pass


def _build_logistic_regression(num_classes: int) -> "object":
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        random_state=SEED,
        multi_class="multinomial" if num_classes > 2 else "auto",
    )


def run_per_layer_sklearn_probes(
    layer_keys: list[str],
    train_act: torch.Tensor,
    test_act: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    num_classes: int,
    compute_metrics: Callable[..., dict],
) -> dict[str, dict]:
    """Fit one sklearn LogisticRegression per layer on flattened [C*H*W] features."""

    L, Bt, _, H, W = train_act.shape
    _, Bv, _, _, _ = test_act.shape

    ytr = train_labels.detach().cpu().numpy()
    yte = test_labels.detach().cpu().numpy()
    if ytr.ndim == 1 and ytr.dtype != np.float32 and ytr.dtype != np.float64:
        ytr = ytr.astype(np.int64, copy=False)
        yte = yte.astype(np.int64, copy=False)

    from sklearn.base import clone

    predict_metrics: dict[str, dict] = {}
    clf_template = _build_logistic_regression(num_classes)

    for ell, lk in enumerate(layer_keys):
        Xtr = train_act[ell].reshape(Bt, -1).detach().cpu().numpy()
        Xte = test_act[ell].reshape(Bv, -1).detach().cpu().numpy()
        pid = layer_key_to_probe_id(lk)

        if ytr.ndim == 1 and np.unique(ytr).size < 2:
            predict_metrics[pid] = {"acc": float("nan"), "f1": float("nan")}
            continue

        clf = clone(clf_template)
        try:
            clf.fit(Xtr, ytr)
            preds = clf.predict(Xte)
        except ValueError:
            predict_metrics[pid] = {"acc": float("nan"), "f1": float("nan")}
            continue

        predict_metrics[pid] = compute_metrics(preds, yte)

    return predict_metrics


def run_per_layer_probes_for_squares(
    layer_keys: list[str],
    train_act: torch.Tensor,
    test_act: torch.Tensor,
    train_labels_64: torch.Tensor,
    test_labels_64: torch.Tensor,
    *,
    compute_metrics: Callable[..., dict],
) -> list[dict[str, dict]]:
    """64 binary probes per layer: one activation cache, 64 * L sklearn fits."""
    per_square: list[dict[str, dict]] = []
    for sq in range(64):
        m = run_per_layer_sklearn_probes(
            layer_keys,
            train_act,
            test_act,
            train_labels_64[:, sq].long(),
            test_labels_64[:, sq].long(),
            num_classes=2,
            compute_metrics=compute_metrics,
        )
        per_square.append(m)
    return per_square
