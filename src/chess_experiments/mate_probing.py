"""Shared helpers for mate probing scripts."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from chess_experiments import datasets, models, probing


def seed_everything(seed: int) -> np.random.Generator:
    """Seed torch and return a numpy RNG."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    return rng


def load_mate_in_1_samples(
    *,
    dataset_id: str,
    n_samples: int,
    seed: int,
    filter_failed_only: bool,
    model_id: str,
    rotate_moves: bool,
    input_encoding: Any,
) -> list[datasets.MateInOneSample]:
    """Load mate-in-1 samples, optionally filtering to model failures."""
    n_load = n_samples * 10 if filter_failed_only else n_samples
    mate_samples = datasets.load_mate_in_one_puzzles(dataset_id, n_load, seed)
    if not filter_failed_only or not mate_samples:
        return mate_samples

    model_temp = models.load_model(model_id)
    failed_samples: list[datasets.MateInOneSample] = []
    for start in range(0, len(mate_samples), 64):
        batch = mate_samples[start : start + 64]
        boards = [s.board for s in batch]
        board_kwargs = {"input_encoding": input_encoding} if input_encoding is not None else {}
        td = TensorDict({"board": model_temp.prepare_boards(*boards, **board_kwargs)}, batch_size=[len(boards)])
        with torch.no_grad():
            policy = model_temp(td)["policy"]
        best_legal = probing.get_best_legal_labels(boards, policy)
        failed_samples.extend(s for i, s in enumerate(batch) if best_legal[i] != s.move_idx)

    del model_temp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return failed_samples


def split_binary_samples(
    *,
    mate_samples: list[datasets.MateInOneSample],
    non_mate_samples: list[datasets.MateInOneSample],
    n_mate_train: int,
    n_mate_test: int,
    n_non_mate_train: int,
    n_non_mate_test: int,
    rng: np.random.Generator,
) -> tuple[list[datasets.MateInOneSample], list[datasets.MateInOneSample]]:
    """Split and shuffle train/test samples for binary mate classification."""
    n_mate_avail = len(mate_samples)
    train_mate = mate_samples[: min(n_mate_train, n_mate_avail)]
    test_mate = mate_samples[n_mate_train : min(n_mate_train + n_mate_test, n_mate_avail)]
    train_non_mate = non_mate_samples[:n_non_mate_train]
    test_non_mate = non_mate_samples[n_non_mate_train : n_non_mate_train + n_non_mate_test]
    train_samples = train_mate + train_non_mate
    test_samples = test_mate + test_non_mate
    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


def split_move_samples(samples: list[Any], n_train: int, n_test: int) -> tuple[list[Any], list[Any]]:
    """Split move samples with graceful fallback when fewer items are available."""
    n_avail = len(samples)
    n_total = n_train + n_test
    if n_avail >= n_total:
        return samples[:n_train], samples[n_train : n_train + n_test]

    ratio = n_train / n_total
    adjusted_n_train = max(1, int(n_avail * ratio))
    adjusted_n_test = n_avail - adjusted_n_train
    if adjusted_n_test == 0 and n_avail >= 2:
        adjusted_n_train -= 1
        adjusted_n_test = 1
    return samples[:adjusted_n_train], samples[adjusted_n_train : adjusted_n_train + adjusted_n_test]


def mate_in_1_active_squares_labels_64(samples: list[Any], rotate_moves: bool) -> torch.Tensor:
    """Build 64 binary labels for from/to squares of the sample move."""
    y = torch.zeros(len(samples), 64, dtype=torch.long)
    for i, s in enumerate(samples):
        if s.move_idx is None:
            continue
        from_sq, to_sq = probing.move_idx_to_squares(
            s.move_idx,
            None if rotate_moves else s.board,
            rotate_moves,
        )
        y[i, from_sq] = 1
        y[i, to_sq] = 1
    return y
