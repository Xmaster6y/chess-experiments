from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from chess_experiments.layout_probing import PredictionMode, ProbeLayout, resolve_probing_block


def probe_field(task_cfg: Any, key: str) -> Any:
    """Read required ``probe.<key>``."""
    v = OmegaConf.select(task_cfg, f"probe.{key}")
    if v is None:
        raise ValueError(f"probe.{key} must be set in the task config")
    return v


def validate_probe_estimator(estimator: str) -> None:
    if estimator not in ("linear", "sklearn"):
        raise ValueError(
            f"probe.estimator must be 'linear' or 'sklearn' (sklearn LogisticRegression); got {estimator!r}"
        )


def validate_mate_probe_task(task_cfg: Any) -> None:
    probe_field(task_cfg, "model")
    validate_probe_estimator(probe_field(task_cfg, "estimator"))
    probe_field(task_cfg, "prediction_mode")
    probe_field(task_cfg, "activation_batch_size")
    probe_field(task_cfg, "probe_epochs")
    probe_field(task_cfg, "probe_batch_size")


def validate_active_squares_task(task_cfg: Any) -> None:
    validate_mate_probe_task(task_cfg)


def validate_probing_block_for_square_labels(task_cfg: Any) -> None:
    modes, _ = resolve_probing_block(task_cfg)
    if not modes:
        return
    pred = PredictionMode(str(probe_field(task_cfg, "prediction_mode")))
    if pred != PredictionMode.BINARY:
        raise ValueError(
            "Active-squares layout probing with per-square binary labels requires "
            f"probe.prediction_mode=binary; got {pred.value!r}"
        )
    for m in modes:
        ProbeLayout(m)
