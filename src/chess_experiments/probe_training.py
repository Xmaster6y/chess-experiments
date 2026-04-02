"""Sklearn linear probes on cached backbone activations (per-layer flattened C*H*W)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from chess_experiments.activations import layer_key_to_probe_id

from scripts.constants import SEED


def _build_logistic_regression(num_classes: int) -> "object":
    from sklearn.linear_model import LogisticRegression

    _ = num_classes
    return LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        random_state=SEED,
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


def run_per_layer_linear_probes(
    layer_keys: list[str],
    train_act: torch.Tensor,
    test_act: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    num_classes: int,
    compute_metrics: Callable[..., dict],
    probe_epochs: int,
    probe_batch_size: int,
) -> dict[str, dict]:
    """Fit one torch linear classifier per layer on flattened [C*H*W] features."""

    L, Bt, _, H, W = train_act.shape
    _, Bv, _, _, _ = test_act.shape
    del H, W

    ytr = train_labels.detach().cpu().long()
    yte = test_labels.detach().cpu().long()

    predict_metrics: dict[str, dict] = {}
    epochs = max(0, probe_epochs)
    batch_size = max(1, probe_batch_size)

    for ell, lk in enumerate(layer_keys):
        Xtr = train_act[ell].reshape(Bt, -1).detach().cpu().float()
        Xte = test_act[ell].reshape(Bv, -1).detach().cpu().float()
        pid = layer_key_to_probe_id(lk)

        if ytr.ndim != 1:
            raise ValueError("torch linear per-layer probes require 1D labels")
        if torch.unique(ytr).numel() < 2:
            predict_metrics[pid] = {"acc": float("nan"), "f1": float("nan")}
            continue

        torch.manual_seed(SEED)
        clf = torch.nn.Linear(Xtr.shape[-1], num_classes)
        optim = torch.optim.Adam(clf.parameters(), lr=1e-3)

        for _ in range(epochs):
            perm = torch.randperm(Bt)
            for start in range(0, Bt, batch_size):
                idx = perm[start : start + batch_size]
                logits = clf(Xtr[idx])
                loss = F.cross_entropy(logits, ytr[idx])
                optim.zero_grad()
                loss.backward()
                optim.step()

        with torch.no_grad():
            preds = clf(Xte).argmax(dim=-1).cpu().numpy()
        predict_metrics[pid] = compute_metrics(preds, yte.cpu().numpy())

    return predict_metrics


def run_per_layer_probes_for_squares(
    layer_keys: list[str],
    train_act: torch.Tensor,
    test_act: torch.Tensor,
    train_labels_64: torch.Tensor,
    test_labels_64: torch.Tensor,
    *,
    estimator: str,
    probe_epochs: int,
    probe_batch_size: int,
    compute_metrics: Callable[..., dict],
) -> list[dict[str, dict]]:
    """64 binary probes per layer with selectable estimator."""
    per_square: list[dict[str, dict]] = []
    for sq in range(64):
        match estimator:
            case "sklearn":
                m = run_per_layer_sklearn_probes(
                    layer_keys,
                    train_act,
                    test_act,
                    train_labels_64[:, sq].long(),
                    test_labels_64[:, sq].long(),
                    num_classes=2,
                    compute_metrics=compute_metrics,
                )
            case "linear":
                m = run_per_layer_linear_probes(
                    layer_keys,
                    train_act,
                    test_act,
                    train_labels_64[:, sq].long(),
                    test_labels_64[:, sq].long(),
                    num_classes=2,
                    compute_metrics=compute_metrics,
                    probe_epochs=probe_epochs,
                    probe_batch_size=probe_batch_size,
                )
            case _:
                raise ValueError(f"estimator must be 'linear' or 'sklearn'; got {estimator!r}")
        per_square.append(m)
    return per_square
