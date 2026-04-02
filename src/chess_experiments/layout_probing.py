"""Activation layout probing: individual / per-layer / per-square layouts, sharing, and prediction modes."""

from __future__ import annotations

import re
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any

import chess
import numpy as np
import torch

from chess_experiments.activations import collect_backbone_activations

from scripts.constants import SEED

if TYPE_CHECKING:
    from lczerolens import LczeroModel


class ProbeLayout(str, Enum):
    INDIVIDUAL = "individual"
    PER_LAYER = "per_layer"
    PER_SQUARE = "per_square"


class PredictionMode(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    SCALAR = "scalar"
    MULTI_OUTPUT = "multi_output"


def latent_hw_to_chess_square(h: int, w: int) -> int:
    """Map backbone 8×8 index (h, w) to python-chess square 0..63 (lczerolens input grid)."""
    return chess.square(w, h)


def chess_square_to_hw(sq: int) -> tuple[int, int]:
    """Inverse of `latent_hw_to_chess_square`: return (h, w)."""
    return chess.square_rank(sq), chess.square_file(sq)


def validate_layout_sharing(
    layout_mode: ProbeLayout,
    share_across_squares: bool,
    share_across_layers: bool,
) -> None:
    if layout_mode == ProbeLayout.PER_LAYER:
        if share_across_squares or share_across_layers:
            warnings.warn(
                "share_across_squares / share_across_layers are ignored for per_layer probing",
                UserWarning,
                stacklevel=2,
            )
        return
    if layout_mode == ProbeLayout.PER_SQUARE and share_across_squares:
        warnings.warn(
            "share_across_squares is N/A for per_square (one head per square); ignoring",
            UserWarning,
            stacklevel=2,
        )


def normalize_labels(
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    layout_mode: ProbeLayout,
    prediction_mode: PredictionMode,
    num_classes: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure shapes (B,) or (B, K); warn on odd combinations."""
    if train_labels.dim() not in (1, 2):
        raise ValueError("train_labels must be 1D or 2D")
    if test_labels.dim() != train_labels.dim():
        raise ValueError("train_labels and test_labels must have the same number of dimensions")
    if train_labels.dim() == 2 and train_labels.shape[-1] != 64:
        if layout_mode != ProbeLayout.PER_LAYER:
            warnings.warn(
                f"train_labels last dim is {train_labels.shape[-1]} (expected 64 for board squares)",
                UserWarning,
                stacklevel=2,
            )
    if prediction_mode == PredictionMode.MULTI_OUTPUT:
        if train_labels.dim() != 2:
            raise ValueError("multi_output requires labels of shape [B, K]")
    if prediction_mode == PredictionMode.BINARY and num_classes is not None and num_classes != 2:
        raise ValueError("prediction_mode binary requires num_classes=2")
    return train_labels, test_labels


def _build_sklearn_estimator(
    prediction_mode: PredictionMode,
    num_classes: int,
):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.multioutput import MultiOutputClassifier

    if prediction_mode == PredictionMode.SCALAR:
        return Ridge(random_state=SEED, alpha=1.0)
    if prediction_mode == PredictionMode.MULTI_OUTPUT:
        base = LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            random_state=SEED,
            multi_class="multinomial" if num_classes > 2 else "auto",
        )
        return MultiOutputClassifier(base)
    if prediction_mode == PredictionMode.BINARY:
        return LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED)
    return LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        random_state=SEED,
        multi_class="multinomial" if num_classes > 2 else "auto",
    )


def _metrics_classify(preds: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    preds = preds.astype(np.int64, copy=False).ravel()
    y = y.astype(np.int64, copy=False).ravel()
    acc = float(accuracy_score(y, preds))
    if np.unique(y).size < 2:
        f1 = float("nan")
    else:
        f1 = float(f1_score(y, preds, average="binary", pos_label=1, zero_division=0))
    return {"acc": acc, "f1": f1}


def _metrics_multiclass(preds: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    acc = float(accuracy_score(y, preds))
    f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    return {"acc": acc, "f1": f1}


def _metrics_scalar(preds: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, r2_score

    return {
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }


def _metrics_multioutput(preds: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Per-output binary accuracy averaged."""
    if preds.ndim == 1:
        preds = preds.reshape(len(preds), -1)
    if y.ndim == 1:
        y = y.reshape(len(y), -1)
    accs = []
    for j in range(y.shape[1]):
        accs.append(float((preds[:, j] == y[:, j]).mean()))
    return {"acc": float(np.mean(accs)) if accs else 0.0, "f1": float("nan")}


def _fit_predict_metrics(
    clf: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    prediction_mode: PredictionMode,
) -> dict[str, float]:
    if prediction_mode not in (PredictionMode.SCALAR, PredictionMode.MULTI_OUTPUT):
        if np.unique(y_train).size < 2:
            return {"acc": float("nan"), "f1": float("nan")}
    try:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
    except ValueError:
        return {"acc": float("nan"), "f1": float("nan")}
    if prediction_mode == PredictionMode.SCALAR:
        return _metrics_scalar(np.ravel(preds), np.ravel(y_test))
    if prediction_mode == PredictionMode.MULTI_OUTPUT:
        return _metrics_multioutput(preds, y_test)
    if prediction_mode == PredictionMode.BINARY:
        return _metrics_classify(preds, y_test)
    return _metrics_multiclass(preds, y_test)


def _run_sklearn_probes(
    probe_specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    prediction_mode: PredictionMode,
    num_classes: int,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for probe_id, Xtr, ytr, Xte, yte in probe_specs:
        if len(Xtr) == 0:
            continue
        clf = _build_sklearn_estimator(prediction_mode, num_classes)
        out[probe_id] = _fit_predict_metrics(clf, Xtr, ytr, Xte, yte, prediction_mode)
    return out


def run_probing_layout(
    model: "LczeroModel",
    train_boards: list,
    test_boards: list,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    layout_mode: ProbeLayout | str,
    prediction_mode: PredictionMode | str = PredictionMode.BINARY,
    num_classes: int = 2,
    share_across_squares: bool = False,
    share_across_layers: bool = False,
    backbone_pattern: str | None = None,
    activation_batch_size: int = 64,
    probe_epochs: int = 20,
    probe_batch_size: int = 64,
    estimator: str = "linear",
) -> dict[str, Any]:
    """Probe with a chosen activation layout (cached activations + sklearn).

    Labels: ``(B,)`` global or ``(B, 64)`` per chess square.
    """
    _ = probe_epochs, probe_batch_size, estimator

    if isinstance(layout_mode, str):
        layout_mode = ProbeLayout(layout_mode)
    if isinstance(prediction_mode, str):
        prediction_mode = PredictionMode(prediction_mode)

    validate_layout_sharing(layout_mode, share_across_squares, share_across_layers)
    train_labels, test_labels = normalize_labels(
        train_labels,
        test_labels,
        layout_mode=layout_mode,
        prediction_mode=prediction_mode,
        num_classes=num_classes,
    )

    layer_keys, train_act = collect_backbone_activations(
        model,
        train_boards,
        backbone_pattern=backbone_pattern,
        batch_size=activation_batch_size,
    )
    _, test_act = collect_backbone_activations(
        model,
        test_boards,
        backbone_pattern=backbone_pattern,
        batch_size=activation_batch_size,
    )

    L, Bt, C, H, W = train_act.shape
    _, Bv, _, _, _ = test_act.shape
    assert H == 8 and W == 8

    specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    if layout_mode == ProbeLayout.PER_LAYER:
        for ell, lk in enumerate(layer_keys):
            Xtr = train_act[ell].reshape(Bt, -1).cpu().numpy()
            Xte = test_act[ell].reshape(Bv, -1).cpu().numpy()
            if prediction_mode == PredictionMode.MULTI_OUTPUT:
                if train_labels.dim() != 2:
                    raise ValueError("multi_output requires 2D labels [B, K]")
                ytr = train_labels.cpu().numpy().astype(np.int64, copy=False)
                yte = test_labels.cpu().numpy().astype(np.int64, copy=False)
            elif train_labels.dim() == 1:
                ytr = train_labels.cpu().numpy()
                yte = test_labels.cpu().numpy()
                if prediction_mode != PredictionMode.SCALAR:
                    ytr = ytr.astype(np.int64, copy=False)
                    yte = yte.astype(np.int64, copy=False)
            else:
                raise ValueError(
                    "per_layer with 2D labels requires prediction_mode=multi_output (or use per_square / individual)"
                )
            block_m = re.search(r"block(\d+)", lk)
            pid = f"block{block_m.group(1)}" if block_m else lk
            specs.append((pid, Xtr, ytr, Xte, yte))

    elif layout_mode == ProbeLayout.PER_SQUARE:
        for sq in range(64):
            h, w = chess_square_to_hw(sq)
            if not share_across_layers:
                parts_tr = [train_act[ell, :, :, h, w].reshape(Bt, C) for ell in range(L)]
                parts_te = [test_act[ell, :, :, h, w].reshape(Bv, C) for ell in range(L)]
                Xtr = torch.cat(parts_tr, dim=-1).cpu().numpy()
                Xte = torch.cat(parts_te, dim=-1).cpu().numpy()
            else:
                Xtr = torch.cat([train_act[ell, :, :, h, w].reshape(Bt, C) for ell in range(L)], dim=0).cpu().numpy()
                Xte = torch.cat([test_act[ell, :, :, h, w].reshape(Bv, C) for ell in range(L)], dim=0).cpu().numpy()
                if train_labels.dim() == 1:
                    ytr = np.tile(train_labels.cpu().numpy(), L)
                    yte = np.tile(test_labels.cpu().numpy(), L)
                else:
                    ytr = np.tile(train_labels[:, sq].cpu().numpy(), L)
                    yte = np.tile(test_labels[:, sq].cpu().numpy(), L)
                if prediction_mode != PredictionMode.SCALAR:
                    ytr = np.asarray(ytr, dtype=np.int64)
                    yte = np.asarray(yte, dtype=np.int64)
                specs.append((f"sq{sq}", Xtr, ytr, Xte, yte))
                continue

            if train_labels.dim() == 1:
                ytr = train_labels.cpu().numpy()
                yte = test_labels.cpu().numpy()
            else:
                ytr = train_labels[:, sq].cpu().numpy()
                yte = test_labels[:, sq].cpu().numpy()
            if prediction_mode != PredictionMode.SCALAR:
                ytr = np.asarray(ytr, dtype=np.int64)
                yte = np.asarray(yte, dtype=np.int64)
            specs.append((f"sq{sq}", Xtr, ytr, Xte, yte))

    elif layout_mode == ProbeLayout.INDIVIDUAL:
        if not share_across_squares and not share_across_layers:
            for ell, lk in enumerate(layer_keys):
                block_m = re.search(r"block(\d+)", lk)
                bid = block_m.group(1) if block_m else str(ell)
                for h in range(H):
                    for w in range(W):
                        sq = latent_hw_to_chess_square(h, w)
                        Xtr = train_act[ell, :, :, h, w].cpu().numpy()
                        Xte = test_act[ell, :, :, h, w].cpu().numpy()
                        if train_labels.dim() == 1:
                            ytr = train_labels.cpu().numpy()
                            yte = test_labels.cpu().numpy()
                        else:
                            ytr = train_labels[:, sq].cpu().numpy()
                            yte = test_labels[:, sq].cpu().numpy()
                        if prediction_mode != PredictionMode.SCALAR:
                            ytr = np.asarray(ytr, dtype=np.int64)
                            yte = np.asarray(yte, dtype=np.int64)
                        specs.append((f"block{bid}_sq{sq}", Xtr, ytr, Xte, yte))
        elif share_across_squares and not share_across_layers:
            for ell, lk in enumerate(layer_keys):
                block_m = re.search(r"block(\d+)", lk)
                bid = block_m.group(1) if block_m else str(ell)
                xs_tr = []
                xs_te = []
                ys_tr = []
                ys_te = []
                for h in range(H):
                    for w in range(W):
                        sq = latent_hw_to_chess_square(h, w)
                        xs_tr.append(train_act[ell, :, :, h, w])
                        xs_te.append(test_act[ell, :, :, h, w])
                        if train_labels.dim() == 1:
                            ys_tr.append(train_labels.float())
                            ys_te.append(test_labels.float())
                        else:
                            ys_tr.append(train_labels[:, sq].float())
                            ys_te.append(test_labels[:, sq].float())
                Xtr = torch.cat(xs_tr, dim=0).cpu().numpy()
                Xte = torch.cat(xs_te, dim=0).cpu().numpy()
                ytr = torch.cat(ys_tr, dim=0).cpu().numpy()
                yte = torch.cat(ys_te, dim=0).cpu().numpy()
                if prediction_mode != PredictionMode.SCALAR:
                    ytr = ytr.astype(np.int64, copy=False)
                    yte = yte.astype(np.int64, copy=False)
                specs.append((f"block{bid}", Xtr, ytr, Xte, yte))
        elif not share_across_squares and share_across_layers:
            for h in range(H):
                for w in range(W):
                    sq = latent_hw_to_chess_square(h, w)
                    xs_tr = [train_act[ell, :, :, h, w] for ell in range(L)]
                    xs_te = [test_act[ell, :, :, h, w] for ell in range(L)]
                    Xtr = torch.cat(xs_tr, dim=0).cpu().numpy()
                    Xte = torch.cat(xs_te, dim=0).cpu().numpy()
                    if train_labels.dim() == 1:
                        ytr = np.tile(train_labels.cpu().numpy(), L)
                        yte = np.tile(test_labels.cpu().numpy(), L)
                    else:
                        ytr = np.tile(train_labels[:, sq].cpu().numpy(), L)
                        yte = np.tile(test_labels[:, sq].cpu().numpy(), L)
                    specs.append((f"sq{sq}", Xtr, ytr, Xte, yte))
        else:
            xs_tr = []
            xs_te = []
            ys_tr = []
            ys_te = []
            for ell in range(L):
                for h in range(H):
                    for w in range(W):
                        sq = latent_hw_to_chess_square(h, w)
                        xs_tr.append(train_act[ell, :, :, h, w])
                        xs_te.append(test_act[ell, :, :, h, w])
                        if train_labels.dim() == 1:
                            ys_tr.append(train_labels)
                            ys_te.append(test_labels)
                        else:
                            ys_tr.append(train_labels[:, sq])
                            ys_te.append(test_labels[:, sq])
            Xtr = torch.cat(xs_tr, dim=0).cpu().numpy()
            Xte = torch.cat(xs_te, dim=0).cpu().numpy()
            ytr = torch.cat(ys_tr, dim=0).cpu().numpy()
            yte = torch.cat(ys_te, dim=0).cpu().numpy()
            if prediction_mode != PredictionMode.SCALAR:
                ytr = ytr.astype(np.int64, copy=False)
                yte = yte.astype(np.int64, copy=False)
            specs.append(("global_shared", Xtr, ytr, Xte, yte))

    per_probe = _run_sklearn_probes(specs, prediction_mode, num_classes)

    return {
        "layout_mode": layout_mode.value,
        "prediction_mode": prediction_mode.value,
        "share_across_squares": share_across_squares,
        "share_across_layers": share_across_layers,
        "layer_keys": layer_keys,
        "per_probe": per_probe,
    }


def macro_mean_metric(per_probe: dict[str, dict[str, float]], key: str = "acc") -> float:
    vals = [m[key] for m in per_probe.values() if key in m and not (isinstance(m[key], float) and np.isnan(m[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def label_expansion_factor(
    layout_mode: ProbeLayout,
    *,
    share_across_squares: bool,
    share_across_layers: bool,
    L: int,
    H: int = 8,
    W: int = 8,
) -> int:
    """How many training rows per original board position for a single probe (or probe group)."""
    if layout_mode == ProbeLayout.PER_LAYER:
        return 1
    if layout_mode == ProbeLayout.PER_SQUARE:
        return L if share_across_layers else 1
    if share_across_squares and share_across_layers:
        return L * H * W
    if share_across_squares:
        return H * W
    if share_across_layers:
        return L
    return 1


def parse_probe_sharing(raw: Any) -> tuple[bool, bool]:
    if raw is None:
        raise ValueError("probing.probe_sharing is required (use [] for independent probes per site)")
    if isinstance(raw, str):
        raise TypeError("probing.probe_sharing must be a list, not a string")
    try:
        seq = list(raw)
    except TypeError:
        raise TypeError(f"probing.probe_sharing must be a list; got {type(raw)}") from None
    share_sq = False
    share_la = False
    for item in seq:
        s = str(item).lower().strip()
        if s == "across_squares":
            share_sq = True
        elif s == "across_layers":
            share_la = True
        elif s == "both":
            share_sq = True
            share_la = True
        else:
            raise ValueError(f"Unknown probe_sharing token {item!r}; use across_squares, across_layers, or both")
    return share_sq, share_la


def resolve_probing_settings(probing_cfg: Any | None) -> tuple[list[str], tuple[bool, bool]]:
    """Parse ``probing:`` (modes + probe_sharing only)."""
    if probing_cfg is None:
        raise ValueError("Missing probing: config (add defaults: - /probing/no_layout or another probing preset)")
    get = probing_cfg.get if hasattr(probing_cfg, "get") else lambda k, d=None: d
    if get("modes", None) is None:
        raise ValueError("probing.modes is required (use [] to disable layout probing runs)")
    modes = get("modes")
    if isinstance(modes, str):
        modes = [modes]
    raw_sharing = get("probe_sharing", None)
    if raw_sharing is None:
        raise ValueError("probing.probe_sharing is required (use [] for no weight tying)")
    share_sq, share_la = parse_probe_sharing(raw_sharing)
    return list(modes), (share_sq, share_la)


def resolve_probing_block(task_cfg: Any | None) -> tuple[list[str], tuple[bool, bool]]:
    """Read ``probing`` from a task config node."""
    if task_cfg is None:
        raise ValueError("Task config is missing")
    get = task_cfg.get if hasattr(task_cfg, "get") else lambda k, d=None: d
    if get("probing") is not None:
        return resolve_probing_settings(get("probing"))
    raise ValueError("Missing probing: block (use defaults: - /probing/no_layout or a named probing preset)")
