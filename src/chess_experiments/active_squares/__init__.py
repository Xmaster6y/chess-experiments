"""Active squares: trajectory sampling and touched-square labels."""

from chess_experiments.active_squares.trajectories import (
    collect_samples_for_roots,
    compute_entropy_perplexity_k_from_legal_logits,
    compute_legal_entropy_perplexity_k,
    enumerate_paths_dfs,
    path_to_touched_mask,
    pick_one_path_per_root,
    select_expansion_moves,
)

__all__ = [
    "collect_samples_for_roots",
    "compute_entropy_perplexity_k_from_legal_logits",
    "compute_legal_entropy_perplexity_k",
    "enumerate_paths_dfs",
    "path_to_touched_mask",
    "pick_one_path_per_root",
    "select_expansion_moves",
]
