"""Mate-in-3 probing: binary (is it mate in 3?) and exact first-move probes."""

from pathlib import Path

import torch
from loguru import logger
from omegaconf import DictConfig
from tensordict import TensorDict

from chess_experiments import datasets, models, probing
from chess_experiments.layout_probing import (
    PredictionMode,
    ProbeLayout,
    resolve_layout_probe_block,
    run_probing_layout,
    save_probe_metric_breakdowns,
)
from chess_experiments.mate_probing import seed_everything
from chess_experiments.probing_validate import probe_train_field, validate_mate_in_3_task


def main(cfg: DictConfig, *, save_dir: str | None = None):
    c = cfg.mate_in_3_probing
    validate_mate_in_3_task(c)
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
    probe_epochs = c.probe_train.get("epochs", 0)
    probe_batch_size = probe_train_field(c, "batch_size")
    estimator = probe_train_field(c, "estimator")
    report_metric_binary = str(c.get("report_metric_binary", "f1"))
    report_metric_move = str(c.get("report_metric_move", "f1"))
    filter_failed_only = c.get("filter_failed_only", False)
    move_probe_mode = c.get("move_probe_mode", "full")
    rotate_moves = c.get("rotate_moves", True)
    layout_mode_str, (share_sq, share_la) = resolve_layout_probe_block(c)
    layout_mode = ProbeLayout(layout_mode_str)

    rng = seed_everything(seed)

    n_mate_total = max(
        n_mate_binary + n_test_mate_binary,
        n_mate_move_train + n_mate_move_test,
    )
    mate_puzzles = datasets.load_mate_in_3_puzzles(dataset_mate, n_mate_total, seed)
    mate_samples = [_PuzzleSample(board=b, is_mate=True, move_idx=idx) for b, idx in mate_puzzles]

    if filter_failed_only and mate_samples:
        model_temp = models.load_model(model_id)
        failed_samples = []
        for start in range(0, len(mate_samples), 64):
            batch = mate_samples[start : start + 64]
            boards = [s.board for s in batch]
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
    non_mate_samples = [_PuzzleSample(board=b, is_mate=False, move_idx=None) for b in non_mate_boards]

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
        [1 if s.is_mate else 0 for s in train_samples_binary],
        dtype=torch.long,
    )
    test_labels_binary = torch.tensor(
        [1 if s.is_mate else 0 for s in test_samples_binary],
        dtype=torch.long,
    )
    results_binary = run_probing_layout(
        model=model,
        train_boards=train_samples_binary,
        test_boards=test_samples_binary,
        train_labels=train_labels_binary,
        test_labels=test_labels_binary,
        layout_mode=layout_mode,
        prediction_mode=PredictionMode.BINARY,
        num_classes=2,
        share_across_squares=share_sq,
        share_across_layers=share_la,
        probe_epochs=probe_epochs,
        probe_batch_size=probe_batch_size,
        estimator=estimator,
        activation_batch_size=probe_batch_size,
    )
    logger.info("Binary probe complete.")

    save_dir = Path(save_dir or "results/mate_in_3_probing/default")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_probe_metric_breakdowns(
        save_dir,
        stem="binary",
        results=results_binary,
        report_metric=report_metric_binary,
    )

    if move_probe_mode == "full":
        train_labels_move = torch.tensor([s.move_idx for s in train_mate_move], dtype=torch.long)
        test_labels_move = torch.tensor([s.move_idx for s in test_mate_move], dtype=torch.long)
        results_move = run_probing_layout(
            model=model,
            train_boards=train_mate_move,
            test_boards=test_mate_move,
            train_labels=train_labels_move,
            test_labels=test_labels_move,
            layout_mode=layout_mode,
            prediction_mode=PredictionMode.MULTICLASS,
            num_classes=num_classes_move,
            share_across_squares=share_sq,
            share_across_layers=share_la,
            probe_epochs=probe_epochs,
            probe_batch_size=probe_batch_size,
            estimator=estimator,
            activation_batch_size=probe_batch_size,
        )
        logger.info("Move probe complete.")
        save_probe_metric_breakdowns(
            save_dir,
            stem="move",
            results=results_move,
            report_metric=report_metric_move,
        )
    else:
        assert move_probe_mode == "double"
        train_labels_from = torch.tensor(
            [
                probing.move_idx_to_squares(s.move_idx, None if rotate_moves else s.board, rotate_moves)[0]
                for s in train_mate_move
            ],
            dtype=torch.long,
        )
        train_labels_to = torch.tensor(
            [
                probing.move_idx_to_squares(s.move_idx, None if rotate_moves else s.board, rotate_moves)[1]
                for s in train_mate_move
            ],
            dtype=torch.long,
        )
        test_labels_from = torch.tensor(
            [
                probing.move_idx_to_squares(s.move_idx, None if rotate_moves else s.board, rotate_moves)[0]
                for s in test_mate_move
            ],
            dtype=torch.long,
        )
        test_labels_to = torch.tensor(
            [
                probing.move_idx_to_squares(s.move_idx, None if rotate_moves else s.board, rotate_moves)[1]
                for s in test_mate_move
            ],
            dtype=torch.long,
        )
        results_from = run_probing_layout(
            model=model,
            train_boards=train_mate_move,
            test_boards=test_mate_move,
            train_labels=train_labels_from,
            test_labels=test_labels_from,
            layout_mode=layout_mode,
            prediction_mode=PredictionMode.MULTICLASS,
            num_classes=64,
            share_across_squares=share_sq,
            share_across_layers=share_la,
            probe_epochs=probe_epochs,
            probe_batch_size=probe_batch_size,
            estimator=estimator,
            activation_batch_size=probe_batch_size,
        )
        logger.info("From-square probe complete.")
        results_to = run_probing_layout(
            model=model,
            train_boards=train_mate_move,
            test_boards=test_mate_move,
            train_labels=train_labels_to,
            test_labels=test_labels_to,
            layout_mode=layout_mode,
            prediction_mode=PredictionMode.MULTICLASS,
            num_classes=64,
            share_across_squares=share_sq,
            share_across_layers=share_la,
            probe_epochs=probe_epochs,
            probe_batch_size=probe_batch_size,
            estimator=estimator,
            activation_batch_size=probe_batch_size,
        )
        logger.info("To-square probe complete.")
        save_probe_metric_breakdowns(
            save_dir,
            stem="from_square",
            results=results_from,
            report_metric=report_metric_move,
        )
        save_probe_metric_breakdowns(
            save_dir,
            stem="to_square",
            results=results_to,
            report_metric=report_metric_move,
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Saved metrics to {save_dir}")


class _PuzzleSample:
    """Internal sample for mate-in-3: board + is_mate + move_idx."""

    def __init__(self, board, is_mate: bool, move_idx: int | None):
        self.board = board
        self.is_mate = is_mate
        self.move_idx = move_idx
