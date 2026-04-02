"""Policy-based trajectory expansion, legal perplexity, and 64-bit touched-square masks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from lczerolens import LczeroBoard, LczeroModel

SamplingMode = Literal["perplexity_pruned", "random", "tempered"]
BranchingMode = Literal["perplexity", "fixed_k"]


def compute_legal_entropy_perplexity_k(
    logits: torch.Tensor,
    legal_idx: torch.Tensor,
) -> tuple[float, float, int]:
    """Entropy and perplexity over softmax restricted to legal moves; k = ceil(P), capped by |legal|.

    Returns (entropy, perplexity, k).
    """
    if legal_idx.numel() == 0:
        return 0.0, 1.0, 1
    _l = logits[legal_idx].float()
    return compute_entropy_perplexity_k_from_legal_logits(_l)


def compute_entropy_perplexity_k_from_legal_logits(legal_logits: torch.Tensor) -> tuple[float, float, int]:
    """Entropy / perplexity / k from a 1D tensor of logits over legal moves only."""
    if legal_logits.numel() == 0:
        return 0.0, 1.0, 1
    p = torch.softmax(legal_logits.float(), dim=-1)
    log_p = torch.log(p + 1e-12)
    h = float(-(p * log_p).sum().item())
    ppl = math.exp(h)
    k = int(math.ceil(ppl))
    k = max(1, min(k, legal_logits.numel()))
    return h, ppl, k


def _temperature_at_depth(
    depth: int,
    height: int,
    temp_base: float,
    temp_end: float,
) -> float:
    if height <= 0:
        return temp_base
    t = depth / float(height)
    return float(temp_base + t * (temp_end - temp_base))


def select_expansion_moves(
    board: "LczeroBoard",
    logits_full: torch.Tensor,
    *,
    sampling_mode: SamplingMode,
    branching_mode: BranchingMode,
    fixed_k: int | None,
    rng: np.random.Generator,
    depth: int,
    height: int,
    temp_base: float = 1.0,
    temp_end: float = 2.0,
) -> list[int]:
    """Choose move indices (policy indices) to expand at this node."""
    legal_idx = board.get_legal_indices()
    if legal_idx.numel() == 0:
        return []

    n_legal = int(legal_idx.numel())
    legal_logits = logits_full[legal_idx].float()

    if sampling_mode == "tempered":
        t = _temperature_at_depth(depth, height, temp_base, temp_end)
        t = max(t, 1e-6)
        l_for_moves = legal_logits / t
        if branching_mode == "perplexity":
            _, _, k = compute_entropy_perplexity_k_from_legal_logits(l_for_moves)
        else:
            k = max(1, min(int(fixed_k or 1), n_legal))
    else:
        l_for_moves = legal_logits
        if branching_mode == "perplexity":
            _, _, k = compute_legal_entropy_perplexity_k(logits_full, legal_idx)
        else:
            k = max(1, min(int(fixed_k or 1), n_legal))

    k = max(1, min(k, n_legal))
    probs = torch.softmax(l_for_moves, dim=-1)

    if sampling_mode == "random":
        perm = rng.permutation(n_legal)
        chosen = perm[:k]
        return [int(legal_idx[int(j)].item()) for j in chosen]

    # perplexity_pruned or tempered: top-k by probability under l_for_moves
    top_j = torch.topk(probs, k=k, largest=True).indices
    return [int(legal_idx[int(j)].item()) for j in top_j.tolist()]


def path_to_touched_mask(root: "LczeroBoard", path_move_indices: list[int]) -> np.ndarray:
    """Binary (64,) float32: 1 iff square is from or to of any move along the path."""
    b = root.copy()
    mask = np.zeros(64, dtype=np.float32)
    for mid in path_move_indices:
        m = b.decode_move(mid)
        mask[m.from_square] = 1.0
        mask[m.to_square] = 1.0
        b.push(m)
    return mask


def enumerate_paths_dfs(
    root: "LczeroBoard",
    model: "LczeroModel",
    *,
    height: int,
    max_paths_per_root: int,
    nodes_remaining: list[int],
    sampling_mode: SamplingMode,
    branching_mode: BranchingMode,
    fixed_k: int | None,
    rng: np.random.Generator,
    temp_base: float = 1.0,
    temp_end: float = 2.0,
) -> list[list[int]]:
    """Enumerate leaf paths (move index lists) from root via DFS; caps paths and global node budget."""
    paths: list[list[int]] = []

    def dfs(board: "LczeroBoard", depth: int, path: list[int]) -> None:
        if len(paths) >= max_paths_per_root or nodes_remaining[0] <= 0:
            return
        if depth >= height or board.is_game_over():
            paths.append(list(path))
            return

        nodes_remaining[0] -= 1
        with torch.no_grad():
            td = TensorDict({"board": model.prepare_boards(board)}, batch_size=1)
            logits = model(td)["policy"][0]

        move_indices = select_expansion_moves(
            board,
            logits,
            sampling_mode=sampling_mode,
            branching_mode=branching_mode,
            fixed_k=fixed_k,
            rng=rng,
            depth=depth,
            height=height,
            temp_base=temp_base,
            temp_end=temp_end,
        )
        for mid in move_indices:
            if len(paths) >= max_paths_per_root or nodes_remaining[0] <= 0:
                return
            b2 = board.copy()
            b2.push(b2.decode_move(mid))
            dfs(b2, depth + 1, path + [mid])

    dfs(root, 0, [])
    return paths


def pick_one_path_per_root(
    paths: list[list[int]],
    rng: np.random.Generator,
) -> list[int] | None:
    """If multiple paths, choose one uniformly; if none, None."""
    if not paths:
        return None
    i = int(rng.integers(0, len(paths)))
    return paths[i]


def collect_samples_for_roots(
    roots: list["LczeroBoard"],
    model: "LczeroModel",
    *,
    height: int,
    max_paths_per_root: int,
    max_nodes_total: int,
    sampling_mode: SamplingMode,
    branching_mode: BranchingMode,
    fixed_k: int | None,
    rng: np.random.Generator,
    temp_base: float = 1.0,
    temp_end: float = 2.0,
) -> tuple[list["LczeroBoard"], np.ndarray]:
    """One trajectory per root: enumerate paths (capped), pick one path, return roots and (N, 64) labels."""
    boards: list = []
    rows: list[np.ndarray] = []
    nodes_remaining = [max_nodes_total]

    for root in roots:
        if nodes_remaining[0] <= 0:
            break
        paths = enumerate_paths_dfs(
            root,
            model,
            height=height,
            max_paths_per_root=max_paths_per_root,
            nodes_remaining=nodes_remaining,
            sampling_mode=sampling_mode,
            branching_mode=branching_mode,
            fixed_k=fixed_k,
            rng=rng,
            temp_base=temp_base,
            temp_end=temp_end,
        )
        path = pick_one_path_per_root(paths, rng)
        if path is None:
            continue
        boards.append(root)
        rows.append(path_to_touched_mask(root, path))

    if not rows:
        return boards, np.zeros((0, 64), dtype=np.float32)
    return boards, np.stack(rows, axis=0)
