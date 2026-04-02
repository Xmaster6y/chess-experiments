"""Active squares probing: 64 binary probes for squares touched along sampled trajectories."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from chess_experiments import active_squares, datasets, models, probing
from chess_experiments.layout_probing import (
    PredictionMode,
    ProbeLayout,
    macro_mean_metric,
    resolve_probing_block,
    run_probing_layout,
)
from chess_experiments.probing_validate import (
    probe_field,
    validate_active_squares_task,
    validate_probing_block_for_square_labels,
)


def _sanitize_for_json(obj):
    """Replace NaN/Inf with None for strict JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(x) for x in obj]
    return obj


def main(cfg: DictConfig, *, save_dir: str | None = None):
    c = cfg.active_squares_probing
    validate_active_squares_task(c)
    validate_probing_block_for_square_labels(c)
    seed = c.get("seed", 42)
    model_id = probe_field(c, "model")
    dataset_id = c.dataset
    n_roots_train = c.n_roots_train
    n_roots_test = c.n_roots_test
    height = c.height
    max_paths_per_root = c.max_paths_per_root
    max_nodes_total = c.max_nodes_total
    branching_mode = c.branching
    fixed_k = c.get("fixed_k")
    if branching_mode == "fixed_k" and fixed_k is None:
        raise ValueError("active_squares_probing.branching=fixed_k requires fixed_k")
    sampling_mode = OmegaConf.select(c, "sampling.sampling_mode", default=c.get("sampling_mode"))
    if sampling_mode is None:
        raise ValueError("Set sampling_mode via active_squares_probing/sampling/*.yaml defaults")
    temp_base = c.get("temp_base", 1.0)
    temp_end = c.get("temp_end", 2.0)
    probe_epochs = probe_field(c, "probe_epochs")
    probe_batch_size = probe_field(c, "probe_batch_size")
    estimator = probe_field(c, "estimator")
    pred_mode = PredictionMode(str(probe_field(c, "prediction_mode")))
    activation_batch_size = int(probe_field(c, "activation_batch_size"))
    fen = c.get("fen")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if fen:
        from lczerolens import LczeroBoard

        b = LczeroBoard(fen)
        n_train = min(max(n_roots_train, 1), 1)
        n_test = min(n_roots_test, 1)
        train_roots = [b.copy() for _ in range(n_train)]
        test_roots = [b.copy() for _ in range(n_test)] if n_test else []
        logger.info("Using single FEN root (copies for train/test; trajectories differ via RNG)")
    else:
        n_load = n_roots_train + n_roots_test
        roots_all = datasets.load_tcec_boards(dataset_id, n_load, seed)
        if len(roots_all) < n_load:
            logger.warning(f"Only got {len(roots_all)} boards, requested {n_load}")

        n_avail = len(roots_all)
        n_train = min(n_roots_train, n_avail)
        n_test = min(n_roots_test, max(0, n_avail - n_train))
        train_roots = roots_all[:n_train]
        test_roots = roots_all[n_train : n_train + n_test]

    logger.info(
        f"Sampling={sampling_mode} branching={branching_mode} height={height} "
        f"train_roots={len(train_roots)} test_roots={len(test_roots)}"
    )

    model = models.load_model(model_id)

    train_boards, train_y = active_squares.collect_samples_for_roots(
        train_roots,
        model,
        height=height,
        max_paths_per_root=max_paths_per_root,
        max_nodes_total=max_nodes_total,
        sampling_mode=sampling_mode,
        branching_mode=branching_mode,
        fixed_k=fixed_k,
        rng=rng,
        temp_base=temp_base,
        temp_end=temp_end,
    )
    test_boards, test_y = active_squares.collect_samples_for_roots(
        test_roots,
        model,
        height=height,
        max_paths_per_root=max_paths_per_root,
        max_nodes_total=max_nodes_total,
        sampling_mode=sampling_mode,
        branching_mode=branching_mode,
        fixed_k=fixed_k,
        rng=rng,
        temp_base=temp_base,
        temp_end=temp_end,
    )

    if len(train_boards) == 0 or len(test_boards) == 0:
        raise RuntimeError("No trajectory samples; relax caps or add roots")

    train_labels_64 = torch.tensor(train_y, dtype=torch.float32)
    test_labels_64 = torch.tensor(test_y, dtype=torch.float32)

    layout_modes, (share_sq, share_la) = resolve_probing_block(c)

    save_dir = Path(save_dir or "results/active_squares_probing/default")
    save_dir.mkdir(parents=True, exist_ok=True)

    if layout_modes:
        train_l = train_labels_64.long()
        test_l = test_labels_64.long()
        layout_runs = []
        for mode_str in layout_modes:
            mode = ProbeLayout(mode_str)
            logger.info(
                f"Active-squares layout probe: layout={mode.value} "
                f"probe_sharing={OmegaConf.select(c, 'probing.probe_sharing')}"
            )
            res = run_probing_layout(
                model=model,
                train_boards=train_boards,
                test_boards=test_boards,
                train_labels=train_l,
                test_labels=test_l,
                layout_mode=mode,
                prediction_mode=pred_mode,
                num_classes=2,
                share_across_squares=share_sq,
                share_across_layers=share_la,
                activation_batch_size=activation_batch_size,
                probe_epochs=probe_epochs,
                probe_batch_size=probe_batch_size,
                estimator=estimator,
            )
            per_probe = res["per_probe"]
            layout_runs.append(
                {
                    "task": "active_squares_touched",
                    "layout_mode": res["layout_mode"],
                    "prediction_mode": res["prediction_mode"],
                    "n_probes": len(per_probe),
                    "macro_mean_acc": macro_mean_metric(per_probe, "acc"),
                    "macro_mean_f1": macro_mean_metric(per_probe, "f1"),
                    **({"per_probe": per_probe} if len(per_probe) <= 128 else {}),
                }
            )
        out = {
            "layout_probe_runs": layout_runs,
            "n_train": len(train_boards),
            "n_test": len(test_boards),
            "config": OmegaConf.to_container(c, resolve=True),
        }
    else:
        results = probing.run_probing_64_binary_squares(
            model=model,
            train_boards=train_boards,
            test_boards=test_boards,
            train_labels_64=train_labels_64,
            test_labels_64=test_labels_64,
            probe_epochs=probe_epochs,
            probe_batch_size=probe_batch_size,
            estimator=estimator,
        )

        per_square_summary = []
        for sq, sqm in enumerate(results["per_square"]):
            layers_acc = probing.extract_layer_metric(sqm, "acc")
            layers_f1 = probing.extract_layer_metric(sqm, "f1")
            per_square_summary.append(
                {
                    "square": sq,
                    "layers_acc": layers_acc,
                    "layers_f1": layers_f1,
                }
            )

        out = {
            "macro_mean_acc_by_layer": results["macro_mean_acc_by_layer"],
            "macro_mean_f1_by_layer": results["macro_mean_f1_by_layer"],
            "per_square": per_square_summary,
            "n_train": len(train_boards),
            "n_test": len(test_boards),
            "config": OmegaConf.to_container(c, resolve=True),
        }
    (save_dir / "metrics.json").write_text(
        json.dumps(_sanitize_for_json(out), indent=2),
        encoding="utf-8",
    )
    if layout_modes:
        logger.info(f"Layout probe runs saved ({len(layout_modes)} layout(s))")
    else:
        logger.info(f"Macro mean acc by layer: {out['macro_mean_acc_by_layer']}")
        logger.info(f"Macro mean F1 by layer: {out['macro_mean_f1_by_layer']}")
    logger.info(f"Saved to {save_dir}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
