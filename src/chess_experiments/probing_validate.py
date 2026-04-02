from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from chess_experiments.layout_probing import ProbeLayout, resolve_layout_probe_block


def probe_train_field(task_cfg: Any, key: str) -> Any:
    """Read required ``probe_train.<key>``."""
    v = OmegaConf.select(task_cfg, f"probe_train.{key}")
    if v is None:
        raise ValueError(f"probe_train.{key} must be set in the task config")
    return v


def validate_probe_estimator(estimator: str) -> None:
    if estimator not in ("linear", "sklearn"):
        raise ValueError(
            f"probe_train.estimator must be 'linear' or 'sklearn' (sklearn LogisticRegression); got {estimator!r}"
        )


def validate_probe_train_block(task_cfg: Any) -> None:
    model_id = task_cfg.get("model")
    if model_id is None:
        raise ValueError("model must be set in the task config")
    estimator = str(probe_train_field(task_cfg, "estimator"))
    validate_probe_estimator(estimator)
    probe_train_field(task_cfg, "batch_size")
    if estimator != "sklearn":
        probe_train_field(task_cfg, "epochs")


def validate_report_metric(task_cfg: Any, *, field: str = "report_metric") -> None:
    metric = str(task_cfg.get(field, "acc"))
    if metric not in {"acc", "f1"}:
        raise ValueError(f"{field} must be one of ['acc', 'f1']; got {metric!r}")


def validate_mate_in_1_binary_task(task_cfg: Any) -> None:
    validate_probe_train_block(task_cfg)
    validate_report_metric(task_cfg)


def validate_mate_in_1_move_task(task_cfg: Any) -> None:
    validate_probe_train_block(task_cfg)
    validate_report_metric(task_cfg)


def validate_mate_in_1_active_squares_task(task_cfg: Any) -> None:
    validate_probe_train_block(task_cfg)
    mode, _ = resolve_layout_probe_block(task_cfg)
    ProbeLayout(mode)


def validate_mate_in_3_task(task_cfg: Any) -> None:
    validate_probe_train_block(task_cfg)
    validate_report_metric(task_cfg, field="report_metric_binary")
    validate_report_metric(task_cfg, field="report_metric_move")
    move_mode = str(task_cfg.get("move_probe_mode", "full"))
    if move_mode not in {"full", "double"}:
        raise ValueError(f"mate_in_3_probing.move_probe_mode must be one of ['full', 'double']; got {move_mode!r}")


def validate_active_squares_task(task_cfg: Any) -> None:
    validate_probe_train_block(task_cfg)


def validate_layout_probe_for_square_labels(task_cfg: Any) -> None:
    mode, _ = resolve_layout_probe_block(task_cfg)
    ProbeLayout(mode)
