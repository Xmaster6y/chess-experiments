"""Mate-in-1 best-move multiclass probing."""

from pathlib import Path

import torch
from loguru import logger
from omegaconf import DictConfig

from chess_experiments import models
from chess_experiments.layout_probing import (
    PredictionMode,
    ProbeLayout,
    resolve_layout_probe_block,
    run_probing_layout,
    save_probe_metric_breakdowns,
)
from chess_experiments.mate_probing import load_mate_in_1_samples, seed_everything, split_move_samples
from chess_experiments.probing_validate import probe_train_field, validate_mate_in_1_move_task


def main(cfg: DictConfig, *, save_dir: str | None = None):
    c = cfg.mate_in_1_move_probing
    validate_mate_in_1_move_task(c)

    seed = c.get("seed", 42)
    model_id = c.model
    dataset_mate = c.dataset_mate
    n_mate_train = c.n_mate_train
    n_mate_test = c.n_mate_test
    probe_epochs = c.probe_train.get("epochs", 0)
    probe_batch_size = probe_train_field(c, "batch_size")
    estimator = probe_train_field(c, "estimator")
    report_metric = str(c.get("report_metric", "f1"))
    filter_failed_only = c.get("filter_failed_only", False)
    input_encoding = models.resolve_input_encoding(c.get("input_encoding"))
    layout_mode_str, (share_sq, share_la) = resolve_layout_probe_block(c)
    layout_mode = ProbeLayout(layout_mode_str)

    seed_everything(seed)
    mate_samples = load_mate_in_1_samples(
        dataset_id=dataset_mate,
        n_samples=n_mate_train + n_mate_test,
        seed=seed,
        filter_failed_only=filter_failed_only,
        model_id=model_id,
        rotate_moves=True,
        input_encoding=input_encoding,
    )
    train_samples, test_samples = split_move_samples(mate_samples, n_mate_train, n_mate_test)
    logger.info(f"Move task: train={len(train_samples)}, test={len(test_samples)}")
    if not train_samples or not test_samples:
        raise RuntimeError("Not enough mate-in-1 samples for move probing after filtering")

    model = models.load_model(model_id)
    num_classes = models.get_policy_size(model, train_samples[0].board, input_encoding=input_encoding)
    train_labels = torch.tensor([s.move_idx for s in train_samples], dtype=torch.long)
    test_labels = torch.tensor([s.move_idx for s in test_samples], dtype=torch.long)

    results = run_probing_layout(
        model=model,
        train_boards=train_samples,
        test_boards=test_samples,
        train_labels=train_labels,
        test_labels=test_labels,
        layout_mode=layout_mode,
        prediction_mode=PredictionMode.MULTICLASS,
        num_classes=num_classes,
        share_across_squares=share_sq,
        share_across_layers=share_la,
        probe_epochs=probe_epochs,
        probe_batch_size=probe_batch_size,
        estimator=estimator,
        activation_batch_size=probe_batch_size,
        input_encoding=input_encoding,
    )
    logger.info("Best-move probe complete.")

    save_dir = Path(save_dir or "results/mate_in_1_move_probing/default")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_probe_metric_breakdowns(
        save_dir,
        stem="move",
        results=results,
        report_metric=report_metric,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Saved metrics to {save_dir}")
