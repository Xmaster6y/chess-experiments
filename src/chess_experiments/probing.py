"""Probing infrastructure for linear probes on model activations."""

import re
from typing import TYPE_CHECKING, Callable

import chess
import numpy as np
import torch
from tensordict import TensorDict

from chess_experiments.activations import collect_backbone_activations
from chess_experiments.probe_training import run_per_layer_probes_for_squares, run_per_layer_sklearn_probes

from scripts.constants import BACKBONE_PATTERN, D_LATENT

if TYPE_CHECKING:
    from lczerolens import LczeroBoard, LczeroModel


def get_absolute_best_labels(policy: torch.Tensor) -> torch.Tensor:
    """Argmax over full policy (all moves). Returns tensor of move indices."""
    return policy.argmax(dim=-1)


def get_best_legal_labels(boards: list["LczeroBoard"], policy: torch.Tensor) -> list[int]:
    """Argmax restricted to legal moves only. Returns list of move indices."""
    labels = []
    for i, board in enumerate(boards):
        legal_idx = board.get_legal_indices()
        if legal_idx.numel() == 0:
            labels.append(int(policy[i].argmax().item()))
            continue
        logits_legal = policy[i, legal_idx]
        best_j = logits_legal.argmax().item()
        labels.append(int(legal_idx[best_j].item()))
    return labels


def move_idx_to_squares(
    move_idx: int,
    board: "LczeroBoard | None" = None,
    rotate: bool = True,
) -> tuple[int, int]:
    """Convert policy index to (from_square, to_square), each 0–63.

    rotate=True: side-to-move perspective (parse POLICY_INDEX).
    rotate=False: absolute board coords (requires board for decode).
    """
    if rotate:
        from lczerolens.constants import POLICY_INDEX

        uci = POLICY_INDEX[move_idx]
        from_sq = chess.SQUARE_NAMES.index(uci[:2])
        to_sq = chess.SQUARE_NAMES.index(uci[2:4])
        return from_sq, to_sq
    if board is None:
        raise ValueError("board must be provided when rotate=False")
    move = board.decode_move(move_idx)
    return move.from_square, move.to_square


def run_probing_64_binary_squares(
    model: "LczeroModel",
    train_boards: list,
    test_boards: list,
    train_labels_64: torch.Tensor,
    test_labels_64: torch.Tensor,
    backbone_pattern: str | None = None,
    d_latent: int = D_LATENT,
    probe_epochs: int = 20,
    probe_batch_size: int = 64,
    estimator: str = "linear",
) -> dict:
    """64 binary probes (one per square); single activation cache, sklearn per layer."""
    assert train_labels_64.shape[-1] == 64 and test_labels_64.shape[-1] == 64
    del d_latent
    _ = probe_epochs
    if estimator not in ("linear", "sklearn"):
        raise ValueError(f"estimator must be 'linear' or 'sklearn' (sklearn LogisticRegression); got {estimator!r}")

    pattern = backbone_pattern or BACKBONE_PATTERN
    layer_keys, train_act = collect_backbone_activations(
        model, train_boards, backbone_pattern=pattern, batch_size=probe_batch_size
    )
    _, test_act = collect_backbone_activations(
        model, test_boards, backbone_pattern=pattern, batch_size=probe_batch_size
    )

    per_square_metrics = run_per_layer_probes_for_squares(
        layer_keys,
        train_act,
        test_act,
        train_labels_64,
        test_labels_64,
        compute_metrics=_compute_binary_acc_f1,
    )

    layer_names = list(extract_layer_metric(per_square_metrics[0], "acc").keys())
    macro_mean_acc_by_layer = _macro_mean_metric_by_layer(per_square_metrics, layer_names, "acc")
    macro_mean_f1_by_layer = _macro_mean_metric_by_layer(per_square_metrics, layer_names, "f1")

    return {
        "per_square": per_square_metrics,
        "macro_mean_acc_by_layer": macro_mean_acc_by_layer,
        "macro_mean_f1_by_layer": macro_mean_f1_by_layer,
    }


def extract_layer_metric(metrics: dict, key: str) -> dict[str, float]:
    """Extract one metric per layer from probe metrics (e.g. acc or f1)."""
    out = {}
    for k, val in metrics.items():
        if isinstance(val, dict) and key in val:
            m = re.search(r"block(\d+)", k)
            if m:
                v = val[key]
                if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                    out[f"block{m.group(1)}"] = float(v)
    return dict(sorted(out.items(), key=lambda x: int(x[0].replace("block", ""))))


def extract_layer_accuracies(metrics: dict, key: str = "acc") -> dict[str, float]:
    """Extract accuracy per layer from probe metrics. Returns dict layer_name -> acc."""
    return extract_layer_metric(metrics, key)


def _macro_mean_metric_by_layer(
    per_square_metrics: list[dict],
    layer_names: list[str],
    metric_key: str,
) -> dict[str, float]:
    """Mean over squares per layer; ignores NaN (e.g. undefined F1 when test labels are one class)."""
    macro: dict[str, float] = {}
    for layer in layer_names:
        vals: list[float] = []
        for sqm in per_square_metrics:
            by_block = extract_layer_metric(sqm, metric_key)
            if layer not in by_block:
                continue
            v = by_block[layer]
            if not np.isnan(v):
                vals.append(v)
        if vals:
            macro[layer] = float(np.mean(vals))
    return macro


def _compute_accuracy(predictions: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray) -> dict:
    preds = predictions.cpu().numpy() if hasattr(predictions, "cpu") else np.asarray(predictions)
    labs = labels.cpu().numpy() if hasattr(labels, "cpu") else np.asarray(labels)
    acc = (preds == labs).mean()
    return {"acc": float(acc)}


def _compute_binary_acc_f1(predictions: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray) -> dict:
    """Accuracy and binary F1 (positive class = 1, touched); F1 is NaN if test labels are single-class."""
    from sklearn.metrics import f1_score

    preds = predictions.cpu().numpy() if hasattr(predictions, "cpu") else np.asarray(predictions)
    labs = labels.cpu().numpy() if hasattr(labels, "cpu") else np.asarray(labels)
    preds = preds.astype(np.int64, copy=False).ravel()
    labs = labs.astype(np.int64, copy=False).ravel()
    acc = float((preds == labs).mean())
    if np.unique(labs).size < 2:
        f1 = float("nan")
    else:
        f1 = float(f1_score(labs, preds, average="binary", pos_label=1, zero_division=0))
    return {"acc": acc, "f1": f1}


def run_probing(
    model: "LczeroModel",
    train_boards: list,
    test_boards: list,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    backbone_pattern: str | None = None,
    d_latent: int = D_LATENT,
    probe_epochs: int = 20,
    probe_batch_size: int = 64,
    estimator: str = "linear",
    compute_metrics: Callable[..., dict] | None = None,
    activation_batch_size: int | None = None,
) -> dict:
    """Per-layer probes on cached backbone activations (sklearn on flattened [C*H*W]).

    Boards: objects with ``.board`` or raw boards. ``probe_epochs`` is unused (sklearn).
    ``compute_metrics``: optional ``(preds, labels) -> dict``; default is accuracy only.
    """
    del d_latent
    _ = probe_epochs
    if estimator not in ("linear", "sklearn"):
        raise ValueError(f"estimator must be 'linear' or 'sklearn'; got {estimator!r}")

    pattern = backbone_pattern or BACKBONE_PATTERN
    act_bs = activation_batch_size if activation_batch_size is not None else probe_batch_size

    layer_keys, train_act = collect_backbone_activations(
        model, train_boards, backbone_pattern=pattern, batch_size=act_bs
    )
    _, test_act = collect_backbone_activations(model, test_boards, backbone_pattern=pattern, batch_size=act_bs)

    if compute_metrics is None:
        compute_metrics = _compute_accuracy

    return run_per_layer_sklearn_probes(
        layer_keys,
        train_act,
        test_act,
        train_labels,
        test_labels,
        num_classes=num_classes,
        compute_metrics=compute_metrics,
    )


def run_cross_model_probing(
    source_model: "LczeroModel",
    target_model: "LczeroModel",
    train_boards: list["LczeroBoard"],
    test_boards: list["LczeroBoard"],
    label_type: str,
    num_classes: int,
    backbone_pattern: str | None = None,
    d_latent: int = D_LATENT,
    probe_epochs: int = 20,
    probe_batch_size: int = 64,
    activation_batch_size: int | None = None,
) -> dict:
    """Source activations, target policy labels; metrics per layer. ``probe_epochs`` unused.

    ``label_type``: ``absolute`` or ``legal``.
    """
    del d_latent
    _ = probe_epochs

    pattern = backbone_pattern or BACKBONE_PATTERN
    act_bs = activation_batch_size if activation_batch_size is not None else probe_batch_size

    board_tensor_train = target_model.prepare_boards(*train_boards)
    board_tensor_test = target_model.prepare_boards(*test_boards)

    with torch.no_grad():
        target_out_train = target_model(TensorDict({"board": board_tensor_train}, batch_size=[len(train_boards)]))
        target_out_test = target_model(TensorDict({"board": board_tensor_test}, batch_size=[len(test_boards)]))

    policy_train = target_out_train["policy"]
    policy_test = target_out_test["policy"]

    if label_type == "absolute":
        labels_train = get_absolute_best_labels(policy_train)
        labels_test = get_absolute_best_labels(policy_test)
    else:
        labels_train = torch.tensor(
            get_best_legal_labels(train_boards, policy_train),
            dtype=torch.long,
            device=policy_train.device,
        )
        labels_test = torch.tensor(
            get_best_legal_labels(test_boards, policy_test),
            dtype=torch.long,
            device=policy_test.device,
        )

    layer_keys, train_act = collect_backbone_activations(
        source_model, train_boards, backbone_pattern=pattern, batch_size=act_bs
    )
    _, test_act = collect_backbone_activations(source_model, test_boards, backbone_pattern=pattern, batch_size=act_bs)

    return run_per_layer_sklearn_probes(
        layer_keys,
        train_act,
        test_act,
        labels_train,
        labels_test,
        num_classes=num_classes,
        compute_metrics=_compute_accuracy,
    )
