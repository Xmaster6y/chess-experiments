"""Mate-in-1 active-squares probing from mating move (from/to squares)."""

import json
from pathlib import Path

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from chess_experiments import models
from chess_experiments.layout_probing import (
    PredictionMode,
    ProbeLayout,
    macro_mean_metric,
    resolve_layout_probe_block,
    run_probing_layout,
    save_probe_metric_breakdowns,
)
from chess_experiments.mate_probing import (
    mate_in_1_active_squares_labels_64,
    load_mate_in_1_samples,
    seed_everything,
    split_move_samples,
)
from chess_experiments.probing_validate import probe_train_field, validate_mate_in_1_active_squares_task


def main(cfg: DictConfig, *, save_dir: str | None = None):
    c = cfg.mate_in_1_active_squares_probing
    validate_mate_in_1_active_squares_task(c)

    seed = c.get("seed", 42)
    model_id = c.model
    dataset_mate = c.dataset_mate
    n_mate_train = c.n_mate_train
    n_mate_test = c.n_mate_test
    probe_epochs = c.probe_train.get("epochs", 0)
    probe_batch_size = probe_train_field(c, "batch_size")
    estimator = probe_train_field(c, "estimator")
    filter_failed_only = c.get("filter_failed_only", False)
    rotate_moves = c.get("rotate_moves", True)
    input_encoding = models.resolve_input_encoding(c.get("input_encoding"))
    layout_mode_str, (share_sq, share_la) = resolve_layout_probe_block(c)
    layout_mode = ProbeLayout(layout_mode_str)
    pred_mode = PredictionMode.MULTI_OUTPUT

    seed_everything(seed)
    mate_samples = load_mate_in_1_samples(
        dataset_id=dataset_mate,
        n_samples=n_mate_train + n_mate_test,
        seed=seed,
        filter_failed_only=filter_failed_only,
        model_id=model_id,
        rotate_moves=rotate_moves,
        input_encoding=input_encoding,
    )
    train_samples, test_samples = split_move_samples(mate_samples, n_mate_train, n_mate_test)
    logger.info(f"Active-squares task: train={len(train_samples)}, test={len(test_samples)}")
    if not train_samples or not test_samples:
        raise RuntimeError("Not enough mate-in-1 samples for active-squares probing after filtering")

    train_y = mate_in_1_active_squares_labels_64(train_samples, rotate_moves=rotate_moves)
    test_y = mate_in_1_active_squares_labels_64(test_samples, rotate_moves=rotate_moves)

    model = models.load_model(model_id)
    save_dir = Path(save_dir or "results/mate_in_1_active_squares_probing/default")
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Mate-move active-squares layout probe: layout={layout_mode.value} "
        f"sharing={OmegaConf.select(c, 'layout_probe.sharing')}"
    )
    res = run_probing_layout(
        model=model,
        train_boards=train_samples,
        test_boards=test_samples,
        train_labels=train_y,
        test_labels=test_y,
        layout_mode=layout_mode,
        prediction_mode=pred_mode,
        num_classes=2,
        share_across_squares=share_sq,
        share_across_layers=share_la,
        activation_batch_size=probe_batch_size,
        probe_epochs=probe_epochs,
        probe_batch_size=probe_batch_size,
        estimator=estimator,
        input_encoding=input_encoding,
    )
    save_probe_metric_breakdowns(
        save_dir,
        stem="active_squares",
        results=res,
        report_metric="f1",
    )
    per_probe = res["per_probe"]
    summary = {
        "task": "mate_in_1_active_squares_probing",
        "layout_mode": res["layout_mode"],
        "prediction_mode": res["prediction_mode"],
        "share_across_squares": res.get("share_across_squares"),
        "share_across_layers": res.get("share_across_layers"),
        "n_probes": len(per_probe),
        "macro_mean_acc": macro_mean_metric(per_probe, "acc"),
        "macro_mean_f1": macro_mean_metric(per_probe, "f1"),
    }
    if len(per_probe) <= 128:
        summary["per_probe"] = per_probe
    layout_summaries = [summary]

    (save_dir / "active_squares_layout_probe.json").write_text(
        json.dumps(layout_summaries, indent=2),
        encoding="utf-8",
    )
    logger.info("Mate-move active-squares layout probing complete.")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Saved metrics to {save_dir}")
