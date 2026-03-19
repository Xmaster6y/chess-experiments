"""Probing infrastructure for linear probes on model activations."""

import re
from typing import TYPE_CHECKING

import chess
import numpy as np
import torch
from tensordict import TensorDict
from tdhook.latent import Probing
from tdhook.latent.probing import LinearEstimator, ProbeManager

from scripts.constants import D_LATENT, SEED

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


def extract_layer_accuracies(metrics: dict, key: str = "acc") -> dict[str, float]:
    """Extract accuracy per layer from probe metrics. Returns dict layer_name -> acc."""
    out = {}
    for k, val in metrics.items():
        if isinstance(val, dict) and key in val:
            m = re.search(r"block(\d+)", k)
            if m:
                out[f"block{m.group(1)}"] = val[key]
    return dict(sorted(out.items(), key=lambda x: int(x[0].replace("block", ""))))


def _compute_accuracy(predictions: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray) -> dict:
    preds = predictions.cpu().numpy() if hasattr(predictions, "cpu") else np.asarray(predictions)
    labs = labels.cpu().numpy() if hasattr(labels, "cpu") else np.asarray(labels)
    acc = (preds == labs).mean()
    return {"acc": float(acc)}


class SklearnLinearProbe:
    """sklearn LogisticRegression wrapper for tdhook Probing (handles tensor -> numpy)."""

    def __init__(self, d_latent: int, num_classes: int, **kwargs):
        from sklearn.linear_model import LogisticRegression

        self.clf = LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            random_state=SEED,
            **kwargs,
        )

    def fit(self, X, y=None):
        X_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
        y_np = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)
        self.clf.fit(X_np, y_np)
        return self

    def predict(self, X):
        X_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
        return self.clf.predict(X_np)


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
) -> dict:
    """Run probing on backbone layers. Returns metrics per layer.

    train_boards/test_boards: list of objects with .board attr, or list of LczeroBoard.
    estimator: 'linear' (tdhook LinearEstimator) or 'sklearn' (LogisticRegression).
    """
    from scripts.constants import BACKBONE_PATTERN

    pattern = backbone_pattern or BACKBONE_PATTERN

    if estimator == "sklearn":
        estimator_class = SklearnLinearProbe
        estimator_kwargs = {"d_latent": d_latent, "num_classes": num_classes}
    else:
        estimator_class = LinearEstimator
        estimator_kwargs = {
            "d_latent": d_latent,
            "num_classes": num_classes,
            "epochs": probe_epochs,
            "batch_size": probe_batch_size,
            "verbose": False,
        }

    probe_manager = ProbeManager(
        estimator_class=estimator_class,
        estimator_kwargs=estimator_kwargs,
        compute_metrics=_compute_accuracy,
        allow_overwrite=True,
    )

    def _get_board(obj):
        return obj.board if hasattr(obj, "board") else obj

    board_train = model.prepare_boards(*[_get_board(b) for b in train_boards])
    board_test = model.prepare_boards(*[_get_board(b) for b in test_boards])

    with Probing(
        pattern,
        probe_manager.probe_factory,
        additional_keys=["labels", "step_type"],
    ).prepare(model) as hooked_model:
        train_td = TensorDict(
            {
                "board": board_train,
                "labels": train_labels,
                "step_type": "fit",
            },
            batch_size=len(train_boards),
        )
        hooked_model(train_td)

        with torch.no_grad():
            test_td = TensorDict(
                {
                    "board": board_test,
                    "labels": test_labels,
                    "step_type": "predict",
                },
                batch_size=len(test_boards),
            )
            hooked_model(test_td)

    return probe_manager.predict_metrics


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
) -> dict:
    """Run probing: source latents -> target move labels. Returns metrics per layer.

    label_type: 'absolute' (argmax full policy) or 'legal' (argmax over legal moves).
    """
    from scripts.constants import BACKBONE_PATTERN

    pattern = backbone_pattern or BACKBONE_PATTERN

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

    probe_manager = ProbeManager(
        estimator_class=LinearEstimator,
        estimator_kwargs={
            "d_latent": d_latent,
            "num_classes": num_classes,
            "epochs": probe_epochs,
            "batch_size": probe_batch_size,
            "verbose": False,
        },
        compute_metrics=_compute_accuracy,
        allow_overwrite=True,
    )

    board_tensor_train_src = source_model.prepare_boards(*train_boards)
    board_tensor_test_src = source_model.prepare_boards(*test_boards)

    with Probing(
        pattern,
        probe_manager.probe_factory,
        additional_keys=["labels", "step_type"],
    ).prepare(source_model) as hooked_model:
        train_td = TensorDict(
            {
                "board": board_tensor_train_src,
                "labels": labels_train,
                "step_type": "fit",
            },
            batch_size=len(train_boards),
        )
        hooked_model(train_td)

        with torch.no_grad():
            test_td = TensorDict(
                {
                    "board": board_tensor_test_src,
                    "labels": labels_test,
                    "step_type": "predict",
                },
                batch_size=len(test_boards),
            )
            hooked_model(test_td)

    return probe_manager.predict_metrics
