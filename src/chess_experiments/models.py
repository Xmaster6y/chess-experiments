"""Model helpers for loading and evaluating chess models."""

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from lczerolens import LczeroBoard, LczeroModel


def load_model(model_id: str) -> "LczeroModel":
    """Load LczeroModel from HuggingFace."""
    from lczerolens import LczeroModel

    model = LczeroModel.from_hf(model_id)
    model.eval()
    return model


def get_policy_size(model: "LczeroModel", sample_board: "LczeroBoard | None" = None) -> int:
    """Get policy output size (num_classes) from model."""
    if sample_board is None:
        from lczerolens import LczeroBoard

        sample_board = LczeroBoard()
    td = TensorDict({"board": model.prepare_boards(sample_board)}, batch_size=1)
    return model(td)["policy"].shape[-1]


def get_best_legal_idx(board: "LczeroBoard", policy: torch.Tensor, idx: int = 0) -> int:
    """Get best legal move index from policy logits for a single board."""
    legal_idx = board.get_legal_indices()
    if legal_idx.numel() == 0:
        return int(policy[idx].argmax().item())
    logits = policy[idx, legal_idx]
    best_j = logits.argmax().item()
    return int(legal_idx[best_j].item())


def compute_solve_rate(
    model: "LczeroModel",
    puzzles: list[tuple["LczeroBoard", int]],
    batch_size: int = 64,
) -> tuple[int, int, float]:
    """Compute how many puzzles the model solves. Returns (n_solved, n_total, rate)."""
    n_solved = 0
    for start in range(0, len(puzzles), batch_size):
        batch = puzzles[start : start + batch_size]
        boards = [p[0] for p in batch]
        gt_indices = [p[1] for p in batch]
        td = TensorDict({"board": model.prepare_boards(*boards)}, batch_size=[len(boards)])
        with torch.no_grad():
            policy = model(td)["policy"]
        for i, (board, gt_idx) in enumerate(zip(boards, gt_indices)):
            pred_idx = get_best_legal_idx(board, policy, i)
            if pred_idx == gt_idx:
                n_solved += 1
    n_total = len(puzzles)
    rate = n_solved / n_total if n_total > 0 else 0.0
    return n_solved, n_total, rate
