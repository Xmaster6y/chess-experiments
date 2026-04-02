"""Activation layout probing: individual / per-layer / per-square layouts, sharing, and prediction modes."""

from __future__ import annotations

import json
import re
import warnings
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chess
import numpy as np
import torch
from loguru import logger

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
        if share_across_squares:
            raise ValueError("layout_probe.mode=per_layer does not allow sharing.across_squares=true")
        return
    if layout_mode == ProbeLayout.PER_SQUARE and share_across_layers:
        raise ValueError("layout_probe.mode=per_square does not allow sharing.across_layers=true")


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
        )
        return MultiOutputClassifier(base)
    if prediction_mode == PredictionMode.BINARY:
        return LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED)
    return LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        random_state=SEED,
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


def _score_from_predictions(
    preds: np.ndarray,
    y_test: np.ndarray,
    prediction_mode: PredictionMode,
) -> dict[str, float]:
    if prediction_mode == PredictionMode.SCALAR:
        return _metrics_scalar(np.ravel(preds), np.ravel(y_test))
    if prediction_mode == PredictionMode.MULTI_OUTPUT:
        return _metrics_multioutput(preds, y_test)
    if prediction_mode == PredictionMode.BINARY:
        return _metrics_classify(preds, y_test)
    return _metrics_multiclass(preds, y_test)


def _fit_predict_metrics(
    clf: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    prediction_mode: PredictionMode,
) -> tuple[dict[str, float], np.ndarray | None]:
    if prediction_mode not in (PredictionMode.SCALAR, PredictionMode.MULTI_OUTPUT):
        if np.unique(y_train).size < 2:
            return {"acc": float("nan"), "f1": float("nan")}, None
    try:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
    except ValueError:
        return {"acc": float("nan"), "f1": float("nan")}, None
    return _score_from_predictions(np.asarray(preds), y_test, prediction_mode), np.asarray(preds)


def _run_sklearn_probes(
    probe_specs: list[dict[str, Any]],
    prediction_mode: PredictionMode,
    num_classes: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, float]] = {}
    probe_outputs: dict[str, dict[str, Any]] = {}
    for spec in probe_specs:
        probe_id = spec["probe_id"]
        Xtr = spec["Xtr"]
        ytr = spec["ytr"]
        Xte = spec["Xte"]
        yte = spec["yte"]
        if len(Xtr) == 0:
            continue
        clf = _build_sklearn_estimator(prediction_mode, num_classes)
        metrics, preds = _fit_predict_metrics(clf, Xtr, ytr, Xte, yte, prediction_mode)
        out[probe_id] = metrics
        if preds is not None:
            probe_outputs[probe_id] = {
                "preds": np.asarray(preds),
                "labels": np.asarray(yte),
                "layer_idx": spec.get("layer_idx"),
                "square_idx": spec.get("square_idx"),
            }
    return out, probe_outputs


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
    input_encoding=None,
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
        input_encoding=input_encoding,
    )
    _, test_act = collect_backbone_activations(
        model,
        test_boards,
        backbone_pattern=backbone_pattern,
        batch_size=activation_batch_size,
        input_encoding=input_encoding,
    )

    L, Bt, C, H, W = train_act.shape
    _, Bv, _, _, _ = test_act.shape
    assert H == 8 and W == 8
    logger.info(
        "Collected activations: train(B,L,C,H,W)=({},{},{},{},{}), test(B,L,C,H,W)=({},{},{},{},{})",
        Bt,
        L,
        C,
        H,
        W,
        Bv,
        test_act.shape[0],
        test_act.shape[2],
        test_act.shape[3],
        test_act.shape[4],
    )

    specs: list[dict[str, Any]] = []

    def _add_spec(
        probe_id: str,
        Xtr: np.ndarray,
        ytr: np.ndarray,
        Xte: np.ndarray,
        yte: np.ndarray,
        *,
        layer_idx: np.ndarray | None = None,
        square_idx: np.ndarray | None = None,
    ) -> None:
        specs.append(
            {
                "probe_id": probe_id,
                "Xtr": Xtr,
                "ytr": ytr,
                "Xte": Xte,
                "yte": yte,
                "layer_idx": layer_idx,
                "square_idx": square_idx,
            }
        )

    if layout_mode == ProbeLayout.PER_LAYER:
        if not share_across_layers:
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
                _add_spec(pid, Xtr, ytr, Xte, yte)
        else:
            Xtr = torch.cat([train_act[ell].reshape(Bt, -1) for ell in range(L)], dim=0).cpu().numpy()
            Xte = torch.cat([test_act[ell].reshape(Bv, -1) for ell in range(L)], dim=0).cpu().numpy()
            if prediction_mode == PredictionMode.MULTI_OUTPUT:
                if train_labels.dim() != 2:
                    raise ValueError("multi_output requires 2D labels [B, K]")
                ytr = np.tile(train_labels.cpu().numpy(), (L, 1)).astype(np.int64, copy=False)
                yte = np.tile(test_labels.cpu().numpy(), (L, 1)).astype(np.int64, copy=False)
            elif train_labels.dim() == 1:
                ytr = np.tile(train_labels.cpu().numpy(), L)
                yte = np.tile(test_labels.cpu().numpy(), L)
                if prediction_mode != PredictionMode.SCALAR:
                    ytr = ytr.astype(np.int64, copy=False)
                    yte = yte.astype(np.int64, copy=False)
            else:
                raise ValueError(
                    "per_layer with 2D labels requires prediction_mode=multi_output (or use per_square / individual)"
                )
            layer_idx = np.repeat(np.arange(L, dtype=np.int64), Bv)
            _add_spec("shared_across_layers", Xtr, ytr, Xte, yte, layer_idx=layer_idx)

    elif layout_mode == ProbeLayout.PER_SQUARE:
        if share_across_squares:
            xs_tr = []
            xs_te = []
            ys_tr = []
            ys_te = []
            for sq in range(64):
                h, w = chess_square_to_hw(sq)
                parts_tr = [train_act[ell, :, :, h, w].reshape(Bt, C) for ell in range(L)]
                parts_te = [test_act[ell, :, :, h, w].reshape(Bv, C) for ell in range(L)]
                xs_tr.append(torch.cat(parts_tr, dim=-1))
                xs_te.append(torch.cat(parts_te, dim=-1))
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
            square_idx = np.repeat(np.arange(64, dtype=np.int64), Bv)
            _add_spec("shared_across_squares", Xtr, ytr, Xte, yte, square_idx=square_idx)
        else:
            for sq in range(64):
                h, w = chess_square_to_hw(sq)
                parts_tr = [train_act[ell, :, :, h, w].reshape(Bt, C) for ell in range(L)]
                parts_te = [test_act[ell, :, :, h, w].reshape(Bv, C) for ell in range(L)]
                Xtr = torch.cat(parts_tr, dim=-1).cpu().numpy()
                Xte = torch.cat(parts_te, dim=-1).cpu().numpy()

                if train_labels.dim() == 1:
                    ytr = train_labels.cpu().numpy()
                    yte = test_labels.cpu().numpy()
                else:
                    ytr = train_labels[:, sq].cpu().numpy()
                    yte = test_labels[:, sq].cpu().numpy()
                if prediction_mode != PredictionMode.SCALAR:
                    ytr = np.asarray(ytr, dtype=np.int64)
                    yte = np.asarray(yte, dtype=np.int64)
                _add_spec(f"sq{sq}", Xtr, ytr, Xte, yte)

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
                        _add_spec(f"block{bid}_sq{sq}", Xtr, ytr, Xte, yte)
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
                square_idx = np.repeat(np.arange(64, dtype=np.int64), Bv)
                _add_spec(f"block{bid}", Xtr, ytr, Xte, yte, square_idx=square_idx)
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
                    layer_idx = np.repeat(np.arange(L, dtype=np.int64), Bv)
                    _add_spec(f"sq{sq}", Xtr, ytr, Xte, yte, layer_idx=layer_idx)
        else:
            xs_tr = []
            xs_te = []
            ys_tr = []
            ys_te = []
            layer_te_idx = []
            square_te_idx = []
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
                        layer_te_idx.append(np.full(Bv, ell, dtype=np.int64))
                        square_te_idx.append(np.full(Bv, sq, dtype=np.int64))
            Xtr = torch.cat(xs_tr, dim=0).cpu().numpy()
            Xte = torch.cat(xs_te, dim=0).cpu().numpy()
            ytr = torch.cat(ys_tr, dim=0).cpu().numpy()
            yte = torch.cat(ys_te, dim=0).cpu().numpy()
            if prediction_mode != PredictionMode.SCALAR:
                ytr = ytr.astype(np.int64, copy=False)
                yte = yte.astype(np.int64, copy=False)
            _add_spec(
                "global_shared",
                Xtr,
                ytr,
                Xte,
                yte,
                layer_idx=np.concatenate(layer_te_idx, axis=0),
                square_idx=np.concatenate(square_te_idx, axis=0),
            )

    def _range_desc(values: list[int]) -> str:
        if not values:
            return "n/a"
        lo = min(values)
        hi = max(values)
        return str(lo) if lo == hi else f"{lo}-{hi}"

    train_rows_per_probe = [spec["Xtr"].shape[0] for spec in specs]
    test_rows_per_probe = [spec["Xte"].shape[0] for spec in specs]
    input_dim_per_probe = [spec["Xtr"].shape[1] if spec["Xtr"].ndim > 1 else 1 for spec in specs]
    logger.info(
        "Probe plan: layout={}, prediction={}, sharing(squares={}, layers={}), probes={}, probe_inputs(train)={}, probe_inputs(test)={}, probe_input_dim={}",
        layout_mode.value,
        prediction_mode.value,
        share_across_squares,
        share_across_layers,
        len(specs),
        _range_desc(train_rows_per_probe),
        _range_desc(test_rows_per_probe),
        _range_desc(input_dim_per_probe),
    )

    per_probe, probe_outputs = _run_sklearn_probes(specs, prediction_mode, num_classes)

    return {
        "layout_mode": layout_mode.value,
        "prediction_mode": prediction_mode.value,
        "share_across_squares": share_across_squares,
        "share_across_layers": share_across_layers,
        "layer_keys": layer_keys,
        "per_probe": per_probe,
        "probe_outputs": probe_outputs,
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
        return L if share_across_layers else 1
    if layout_mode == ProbeLayout.PER_SQUARE:
        if share_across_squares:
            return H * W
        return L if share_across_layers else 1
    if share_across_squares and share_across_layers:
        return L * H * W
    if share_across_squares:
        return H * W
    if share_across_layers:
        return L
    return 1


def resolve_layout_probe_settings(layout_cfg: Any | None) -> tuple[str, tuple[bool, bool]]:
    """Parse ``layout_probe:`` (mode, sharing)."""
    if layout_cfg is None:
        raise ValueError("Missing layout_probe config block")
    get = layout_cfg.get if hasattr(layout_cfg, "get") else lambda k, d=None: d
    mode = get("mode", None)
    if mode is None:
        raise ValueError("layout_probe.mode is required")
    mode_str = str(mode)
    try:
        mode_enum = ProbeLayout(mode_str)
    except ValueError as exc:
        raise ValueError(f"Invalid layout_probe.mode: {mode_str!r}") from exc
    sharing = get("sharing", None)
    if sharing is None:
        raise ValueError("layout_probe.sharing is required")
    share_sq = bool(getattr(sharing, "get", lambda *_: False)("across_squares", False))
    share_la = bool(getattr(sharing, "get", lambda *_: False)("across_layers", False))
    validate_layout_sharing(mode_enum, share_sq, share_la)
    return mode_str, (share_sq, share_la)


def resolve_layout_probe_block(task_cfg: Any | None) -> tuple[str, tuple[bool, bool]]:
    """Read ``layout_probe`` from a task config node."""
    if task_cfg is None:
        raise ValueError("Task config is missing")
    get = task_cfg.get if hasattr(task_cfg, "get") else lambda k, d=None: d
    block = get("layout_probe")
    if block is None:
        raise ValueError("Missing layout_probe block")
    return resolve_layout_probe_settings(block)


def _parse_probe_id_indices(probe_id: str) -> tuple[int | None, int | None]:
    m_ind = re.fullmatch(r"block(\d+)_sq(\d+)", probe_id)
    if m_ind:
        return int(m_ind.group(1)), int(m_ind.group(2))
    m_layer = re.fullmatch(r"block(\d+)", probe_id)
    if m_layer:
        return int(m_layer.group(1)), None
    m_square = re.fullmatch(r"sq(\d+)", probe_id)
    if m_square:
        return None, int(m_square.group(1))
    return None, None


def _selected_metric(metrics: dict[str, float], metric_key: str) -> float:
    v = metrics.get(metric_key, float("nan"))
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return float("nan")


def _metric_from_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
    prediction_mode: PredictionMode,
    metric_key: str,
) -> float:
    metrics = _score_from_predictions(preds, labels, prediction_mode)
    return _selected_metric(metrics, metric_key)


def _sort_group_scores(group_scores: dict[str, float], prefix: str) -> dict[str, float]:
    return dict(sorted(group_scores.items(), key=lambda kv: int(kv[0].replace(prefix, ""))))


def _probe_sort_key(probe_id: str) -> tuple[int, int, int]:
    layer_idx, square_idx = _parse_probe_id_indices(probe_id)
    if layer_idx is not None and square_idx is not None:
        return (0, layer_idx, square_idx)
    if layer_idx is not None:
        return (1, layer_idx, -1)
    if square_idx is not None:
        return (2, square_idx, -1)
    return (3, 0, 0)


def _group_scores_from_probe_outputs(
    probe_outputs: dict[str, dict[str, Any]],
    *,
    prediction_mode: PredictionMode,
    metric_key: str,
    group: str,
) -> dict[str, float]:
    grouped: dict[str, dict[str, list[np.ndarray]]] = {}

    for probe_id, out in probe_outputs.items():
        preds = out["preds"]
        labels = out["labels"]
        n_rows = labels.shape[0]
        layer_const, square_const = _parse_probe_id_indices(probe_id)

        layer_idx = out.get("layer_idx")
        square_idx = out.get("square_idx")
        if layer_idx is None and layer_const is not None:
            layer_idx = np.full(n_rows, layer_const, dtype=np.int64)
        if square_idx is None and square_const is not None:
            square_idx = np.full(n_rows, square_const, dtype=np.int64)

        if group == "layer":
            group_idx = layer_idx
            key_prefix = "block"
        elif group == "square":
            group_idx = square_idx
            key_prefix = "sq"
        else:
            if layer_idx is None or square_idx is None:
                continue
            group_idx = np.stack([layer_idx, square_idx], axis=1)
            key_prefix = ""

        if group == "individual":
            for pair in np.unique(group_idx, axis=0):
                li = int(pair[0])
                si = int(pair[1])
                mask = np.logical_and(layer_idx == li, square_idx == si)
                key = f"block{li}_sq{si}"
                grouped.setdefault(key, {"preds": [], "labels": []})
                grouped[key]["preds"].append(preds[mask])
                grouped[key]["labels"].append(labels[mask])
        else:
            if group_idx is None:
                continue
            for g in np.unique(group_idx):
                gi = int(g)
                mask = group_idx == gi
                key = f"{key_prefix}{gi}"
                grouped.setdefault(key, {"preds": [], "labels": []})
                grouped[key]["preds"].append(preds[mask])
                grouped[key]["labels"].append(labels[mask])

    scores: dict[str, float] = {}
    for key, vals in grouped.items():
        cat_preds = np.concatenate(vals["preds"], axis=0)
        cat_labels = np.concatenate(vals["labels"], axis=0)
        scores[key] = _metric_from_predictions(cat_preds, cat_labels, prediction_mode, metric_key)

    if group == "layer":
        return _sort_group_scores(scores, "block")
    if group == "square":
        return _sort_group_scores(scores, "sq")
    return dict(
        sorted(
            scores.items(),
            key=lambda kv: tuple(int(x) for x in re.fullmatch(r"block(\d+)_sq(\d+)", kv[0]).groups()),
        )
    )


def build_probe_score_report(
    results: dict[str, Any],
    *,
    metric_key: str,
) -> dict[str, Any]:
    per_probe = results["per_probe"]
    probe_outputs = results.get("probe_outputs", {})
    prediction_mode = PredictionMode(results["prediction_mode"])

    per_probe_scores = {
        probe_id: _selected_metric(metrics, metric_key)
        for probe_id, metrics in sorted(per_probe.items(), key=lambda kv: _probe_sort_key(kv[0]))
    }

    global_score = float("nan")
    if probe_outputs:
        all_preds = [v["preds"] for v in probe_outputs.values()]
        all_labels = [v["labels"] for v in probe_outputs.values()]
        if all_preds and all_labels:
            global_score = _metric_from_predictions(
                np.concatenate(all_preds, axis=0),
                np.concatenate(all_labels, axis=0),
                prediction_mode,
                metric_key,
            )

    report: dict[str, Any] = {
        "report_metric": metric_key,
        "layout_mode": results["layout_mode"],
        "share_across_squares": bool(results.get("share_across_squares", False)),
        "share_across_layers": bool(results.get("share_across_layers", False)),
        "global_score": global_score,
        "per_probe": per_probe_scores,
    }

    if probe_outputs:
        per_layer = _group_scores_from_probe_outputs(
            probe_outputs,
            prediction_mode=prediction_mode,
            metric_key=metric_key,
            group="layer",
        )
        per_square = _group_scores_from_probe_outputs(
            probe_outputs,
            prediction_mode=prediction_mode,
            metric_key=metric_key,
            group="square",
        )
        individual = _group_scores_from_probe_outputs(
            probe_outputs,
            prediction_mode=prediction_mode,
            metric_key=metric_key,
            group="individual",
        )
        if per_layer:
            report["per_layer"] = per_layer
        if per_square:
            report["per_square"] = per_square
        if individual:
            report["individual"] = individual

    return report


def save_probe_metric_breakdowns(
    save_dir: Path,
    *,
    stem: str,
    results: dict[str, Any],
    report_metric: str,
) -> None:
    """Persist selected-metric scores in a single summary file."""
    report = build_probe_score_report(results, metric_key=report_metric)

    summary_path = save_dir / f"{stem}_metrics.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for stale_name in ("per_probe", "per_layer", "per_square", "individual", "full", "other"):
        stale_path = save_dir / f"{stem}_metrics_{stale_name}.json"
        if stale_path.exists():
            stale_path.unlink()
