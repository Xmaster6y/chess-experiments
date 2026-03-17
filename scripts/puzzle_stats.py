"""Puzzle stats: model success rate on puzzles with optional visualizations."""

import json
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig

from chess_experiments import datasets, models, visualization


def main(cfg: DictConfig, *, save_dir: str | None = None):
    puzzle_cfg = cfg.puzzle_stats
    seed = puzzle_cfg.get("seed", 42)
    num_samples = puzzle_cfg.num_samples
    dataset_id = puzzle_cfg.dataset
    save_viz = puzzle_cfg.get("save_viz", False)
    viz_limit = puzzle_cfg.get("viz_limit", 10)

    models_list = puzzle_cfg.get("models") or [puzzle_cfg.get("model")]
    if models_list is None or (isinstance(models_list, list) and None in models_list):
        raise ValueError("Config must specify either 'model' or 'models'")
    if not isinstance(models_list, list):
        models_list = [models_list]

    if "mate-in-1" in dataset_id:
        samples = datasets.load_mate_in_one_puzzles(dataset_id, num_samples, seed)
        puzzles = [(s.board, s.move_idx) for s in samples if s.move_idx is not None]
    elif "mate-in-3" in dataset_id:
        puzzles = datasets.load_mate_in_3_puzzles(dataset_id, num_samples, seed)
    else:
        raise ValueError(f"Unknown dataset type for {dataset_id}; use mate-in-1 or mate-in-3")

    if len(puzzles) < 4:
        raise RuntimeError(f"Not enough valid puzzles ({len(puzzles)}); increase num_samples")

    logger.info(f"Loaded {len(puzzles)} puzzles from {dataset_id}")

    stats = {}
    for model_id in models_list:
        model = models.load_model(model_id)
        n_solved, n_total, rate = models.compute_solve_rate(model, puzzles)
        name = model_id.split("/")[-1]
        stats[name] = {"n_solved": n_solved, "n_total": n_total, "rate": rate}
        logger.info(f"{name}: {n_solved}/{n_total} ({100 * rate:.1f}%)")
        del model
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_dir = Path(save_dir or "results/puzzle_stats/default")
    save_dir.mkdir(parents=True, exist_ok=True)
    stats_path = save_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"Saved stats to {stats_path}")

    if save_viz and puzzles:
        viz_dir = save_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        limit = min(viz_limit, len(puzzles))
        for i in range(limit):
            board, gt_idx = puzzles[i]
            gt_move = board.decode_move(gt_idx)
            gt_uci = gt_move.uci()
            model_predictions = {}
            for model_id in models_list:
                model = models.load_model(model_id)
                from tensordict import TensorDict

                td = TensorDict({"board": model.prepare_boards(board)}, batch_size=1)
                with __import__("torch").no_grad():
                    policy = model(td)["policy"]
                pred_idx = models.get_best_legal_idx(board, policy, 0)
                pred_move = board.decode_move(pred_idx)
                name = model_id.split("/")[-1]
                model_predictions[name] = (pred_move.uci(), pred_idx == gt_idx)
                del model
            visualization.save_puzzle_viz(viz_dir / f"puzzle_{i}", board, gt_uci, model_predictions, i)
        logger.info(f"Saved {limit} visualizations to {viz_dir}")
