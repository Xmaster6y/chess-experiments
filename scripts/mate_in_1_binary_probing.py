"""Binary mate-in-1 probing (mate vs non-mate)."""

from pathlib import Path

import torch
from loguru import logger
from omegaconf import DictConfig

from chess_experiments import datasets, models
from chess_experiments.layout_probing import (
    PredictionMode,
    ProbeLayout,
    resolve_layout_probe_block,
    run_probing_layout,
    save_probe_metric_breakdowns,
)
from chess_experiments.mate_probing import (
    load_mate_in_1_samples,
    seed_everything,
    split_binary_samples,
)
from chess_experiments.probing_validate import probe_train_field, validate_mate_in_1_binary_task


def main(cfg: DictConfig, *, save_dir: str | None = None):
    c = cfg.mate_in_1_binary_probing
    validate_mate_in_1_binary_task(c)

    seed = c.get("seed", 42)
    model_id = c.model
    dataset_mate = c.dataset_mate
    dataset_non_mate = c.dataset_non_mate
    n_mate_train = c.n_mate_train
    n_non_mate_train = c.n_non_mate_train
    n_mate_test = c.get("n_mate_test", n_mate_train // 4)
    n_non_mate_test = c.get("n_non_mate_test", n_non_mate_train // 4)
    probe_epochs = c.probe_train.get("epochs", 0)
    probe_batch_size = probe_train_field(c, "batch_size")
    estimator = probe_train_field(c, "estimator")
    report_metric = str(c.get("report_metric", "f1"))
    input_encoding = models.resolve_input_encoding(c.get("input_encoding"))
    layout_mode_str, (share_sq, share_la) = resolve_layout_probe_block(c)
    layout_mode = ProbeLayout(layout_mode_str)

    rng = seed_everything(seed)
    mate_samples = load_mate_in_1_samples(
        dataset_id=dataset_mate,
        n_samples=n_mate_train + n_mate_test,
        seed=seed,
        filter_failed_only=False,
        model_id=model_id,
        rotate_moves=True,
        input_encoding=input_encoding,
    )

    non_mate_boards = datasets.load_tcec_boards(dataset_non_mate, n_non_mate_train + n_non_mate_test, seed)
    non_mate_samples = [
        datasets.MateInOneSample(board=b, is_mate_in_one=False, move_idx=None) for b in non_mate_boards
    ]
    train_samples, test_samples = split_binary_samples(
        mate_samples=mate_samples,
        non_mate_samples=non_mate_samples,
        n_mate_train=n_mate_train,
        n_mate_test=n_mate_test,
        n_non_mate_train=n_non_mate_train,
        n_non_mate_test=n_non_mate_test,
        rng=rng,
    )
    logger.info(f"Binary task: train={len(train_samples)}, test={len(test_samples)}")

    model = models.load_model(model_id)
    train_labels = torch.tensor([1 if s.is_mate_in_one else 0 for s in train_samples], dtype=torch.long)
    test_labels = torch.tensor([1 if s.is_mate_in_one else 0 for s in test_samples], dtype=torch.long)
    results = run_probing_layout(
        model=model,
        train_boards=train_samples,
        test_boards=test_samples,
        train_labels=train_labels,
        test_labels=test_labels,
        layout_mode=layout_mode,
        prediction_mode=PredictionMode.BINARY,
        num_classes=2,
        share_across_squares=share_sq,
        share_across_layers=share_la,
        probe_epochs=probe_epochs,
        probe_batch_size=probe_batch_size,
        estimator=estimator,
        activation_batch_size=probe_batch_size,
        input_encoding=input_encoding,
    )
    logger.info("Binary probe complete.")

    save_dir = Path(save_dir or "results/mate_in_1_binary_probing/default")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_probe_metric_breakdowns(
        save_dir,
        stem="binary",
        results=results,
        report_metric=report_metric,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Saved metrics to {save_dir}")
