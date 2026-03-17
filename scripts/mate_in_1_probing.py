"""Mate-in-1 probing: binary (is it mate in one?) and exact move probes."""

import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from chess_experiments import datasets, models, probing


def main(cfg: DictConfig, *, save_dir: str | None = None):
    c = cfg.mate_in_1_probing
    seed = c.get("seed", 42)
    model_id = c.model
    dataset_mate = c.dataset_mate
    dataset_non_mate = c.dataset_non_mate
    n_mate_binary = c.n_mate_binary
    n_non_mate = c.n_non_mate
    n_test_mate_binary = c.get("n_test_mate_binary", n_mate_binary // 4)
    n_test_non_mate = c.get("n_test_non_mate", n_non_mate // 4)
    n_mate_move_train = c.n_mate_move_train
    n_mate_move_test = c.n_mate_move_test
    probe_epochs = c.get("probe_epochs", 20)
    probe_batch_size = c.get("probe_batch_size", 64)
    estimator = c.get("estimator", "linear")
    filter_failed_only = c.get("filter_failed_only", False)
    move_probe_mode = c.get("move_probe_mode", "full")
    rotate_moves = c.get("rotate_moves", True)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    n_mate_total = max(
        n_mate_binary + n_test_mate_binary,
        n_mate_move_train + n_mate_move_test,
    )
    mate_samples = datasets.load_mate_in_one_puzzles(dataset_mate, n_mate_total, seed)

    if filter_failed_only and mate_samples:
        model_temp = models.load_model(model_id)
        failed_samples = []
        for start in range(0, len(mate_samples), 64):
            batch = mate_samples[start : start + 64]
            boards = [s.board for s in batch]
            from tensordict import TensorDict

            td = TensorDict({"board": model_temp.prepare_boards(*boards)}, batch_size=[len(boards)])
            with torch.no_grad():
                policy = model_temp(td)["policy"]
            best_legal = probing.get_best_legal_labels(boards, policy)
            failed_samples.extend(s for i, s in enumerate(batch) if best_legal[i] != s.move_idx)
        mate_samples = failed_samples
        del model_temp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    non_mate_boards = datasets.load_tcec_boards(dataset_non_mate, n_non_mate + n_test_non_mate, seed)
    non_mate_samples = [
        datasets.MateInOneSample(board=b, is_mate_in_one=False, move_idx=None) for b in non_mate_boards
    ]

    train_mate_binary = mate_samples[:n_mate_binary]
    test_mate_binary = mate_samples[n_mate_binary : n_mate_binary + n_test_mate_binary]
    train_non_mate = non_mate_samples[:n_non_mate]
    test_non_mate = non_mate_samples[n_non_mate : n_non_mate + n_test_non_mate]
    train_samples_binary = train_mate_binary + train_non_mate
    test_samples_binary = test_mate_binary + test_non_mate
    rng.shuffle(train_samples_binary)
    rng.shuffle(test_samples_binary)

    train_mate_move = mate_samples[:n_mate_move_train]
    test_mate_move = mate_samples[n_mate_move_train : n_mate_move_train + n_mate_move_test]

    logger.info(f"Binary probe: train={len(train_samples_binary)}, test={len(test_samples_binary)}")
    logger.info(f"Move probe: train={len(train_mate_move)}, test={len(test_mate_move)}")

    model = models.load_model(model_id)
    num_classes_move = models.get_policy_size(model, train_mate_move[0].board if train_mate_move else None)

    train_labels_binary = torch.tensor(
        [1 if s.is_mate_in_one else 0 for s in train_samples_binary],
        dtype=torch.long,
    )
    test_labels_binary = torch.tensor(
        [1 if s.is_mate_in_one else 0 for s in test_samples_binary],
        dtype=torch.long,
    )
    results_binary = probing.run_probing(
        model=model,
        train_boards=train_samples_binary,
        test_boards=test_samples_binary,
        train_labels=train_labels_binary,
        test_labels=test_labels_binary,
        num_classes=2,
        probe_epochs=probe_epochs,
        probe_batch_size=probe_batch_size,
        estimator=estimator,
    )
    logger.info("Binary probe complete.")

    save_dir = Path(save_dir or "results/mate_in_1_probing/default")
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "binary_metrics.json").write_text(
        json.dumps(probing.extract_layer_accuracies(results_binary), indent=2),
        encoding="utf-8",
    )

    if move_probe_mode == "full":
        train_labels_move = torch.tensor([s.move_idx for s in train_mate_move], dtype=torch.long)
        test_labels_move = torch.tensor([s.move_idx for s in test_mate_move], dtype=torch.long)
        results_move = probing.run_probing(
            model=model,
            train_boards=train_mate_move,
            test_boards=test_mate_move,
            train_labels=train_labels_move,
            test_labels=test_labels_move,
            num_classes=num_classes_move,
            probe_epochs=probe_epochs,
            probe_batch_size=probe_batch_size,
            estimator=estimator,
        )
        logger.info("Move probe complete.")
        (save_dir / "move_metrics.json").write_text(
            json.dumps(probing.extract_layer_accuracies(results_move), indent=2),
            encoding="utf-8",
        )
    else:
        assert move_probe_mode == "double"
        train_labels_from = torch.tensor(
            [
                probing.move_idx_to_squares(s.move_idx, s.board if not rotate_moves else None, rotate_moves)[0]
                for s in train_mate_move
            ],
            dtype=torch.long,
        )
        train_labels_to = torch.tensor(
            [
                probing.move_idx_to_squares(s.move_idx, s.board if not rotate_moves else None, rotate_moves)[1]
                for s in train_mate_move
            ],
            dtype=torch.long,
        )
        test_labels_from = torch.tensor(
            [
                probing.move_idx_to_squares(s.move_idx, s.board if not rotate_moves else None, rotate_moves)[0]
                for s in test_mate_move
            ],
            dtype=torch.long,
        )
        test_labels_to = torch.tensor(
            [
                probing.move_idx_to_squares(s.move_idx, s.board if not rotate_moves else None, rotate_moves)[1]
                for s in test_mate_move
            ],
            dtype=torch.long,
        )
        results_from = probing.run_probing(
            model=model,
            train_boards=train_mate_move,
            test_boards=test_mate_move,
            train_labels=train_labels_from,
            test_labels=test_labels_from,
            num_classes=64,
            probe_epochs=probe_epochs,
            probe_batch_size=probe_batch_size,
            estimator=estimator,
        )
        logger.info("From-square probe complete.")
        results_to = probing.run_probing(
            model=model,
            train_boards=train_mate_move,
            test_boards=test_mate_move,
            train_labels=train_labels_to,
            test_labels=test_labels_to,
            num_classes=64,
            probe_epochs=probe_epochs,
            probe_batch_size=probe_batch_size,
            estimator=estimator,
        )
        logger.info("To-square probe complete.")
        (save_dir / "from_square_metrics.json").write_text(
            json.dumps(probing.extract_layer_accuracies(results_from), indent=2),
            encoding="utf-8",
        )
        (save_dir / "to_square_metrics.json").write_text(
            json.dumps(probing.extract_layer_accuracies(results_to), indent=2),
            encoding="utf-8",
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Saved metrics to {save_dir}")
