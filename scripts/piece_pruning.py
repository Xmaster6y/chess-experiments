"""Piece pruning evaluation: counterfactual analysis of MCTS and model policy/value."""

import json
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig

from chess_experiments import datasets, models, piece_pruning
from scripts.constants import SEED


def _serialize(obj):
    """Convert to JSON-serializable form."""
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, dict) and not isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if hasattr(obj, "uci"):
        return obj.uci()
    if hasattr(obj, "fen"):
        return obj.fen()
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def main(cfg: DictConfig, *, save_dir: str | None = None):
    c = cfg.piece_pruning
    model_id = c.model
    dataset = c.get("dataset")
    fen = c.get("fen")
    num_samples = c.get("num_samples", 8)
    seed = c.get("seed", SEED)
    mcts_iterations = c.get("mcts_iterations", 20)
    evaluators = list(c.get("evaluators", ["global_sensitivity", "square_impact"]))
    save_heatmaps = c.get("save_heatmaps", True)
    heatmap_limit = c.get("heatmap_limit", 4)

    model = models.load_model(model_id)
    from lczerolens.search import ModelHeuristic

    heuristic = ModelHeuristic(model)

    if fen:
        from lczerolens import LczeroBoard

        boards = [LczeroBoard(fen)]
        logger.info(f"Using single FEN: {fen[:50]}...")
    elif dataset:
        boards = datasets.load_tcec_boards(dataset, num_samples, seed)
        if len(boards) < 1:
            raise RuntimeError(f"No valid boards from {dataset}; increase num_samples")
        logger.info(f"Loaded {len(boards)} boards from {dataset}")
    else:
        raise ValueError("Config must specify either 'fen' or 'dataset'")

    save_dir = Path(save_dir or "results/piece_pruning/default")
    save_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = save_dir / "viz" if save_heatmaps else None
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {"boards": [], "summary": {}}
    heatmap_count = 0

    for board_idx, board in enumerate(boards):
        board_fen = board.fen()
        board_results: dict = {"fen": board_fen, "evaluators": {}}

        if "global_sensitivity" in evaluators:
            impacts = piece_pruning.evaluate_counterfactual_impacts(board, heuristic, iterations=mcts_iterations)
            mean_impact = sum(impacts) / len(impacts) if impacts else 0.0
            max_impact = max(impacts) if impacts else 0.0
            board_results["evaluators"]["global_sensitivity"] = {
                "mean_impact": mean_impact,
                "max_impact": max_impact,
                "n_counterfactuals": len(impacts),
            }
            logger.info(f"Board {board_idx + 1}: mean_impact={mean_impact:.4f}, max={max_impact:.4f}")

        if "square_impact" in evaluators:
            square_impacts = piece_pruning.evaluate_square_impact(board, heuristic, iterations=mcts_iterations)
            board_results["evaluators"]["square_impact"] = {str(sq): float(v) for sq, v in square_impacts.items()}
            if save_heatmaps and heatmap_limit and heatmap_count < heatmap_limit:
                out_path = viz_dir / f"square_impact_{board_idx}.svg"
                piece_pruning.save_square_heatmap(
                    out_path, board, square_impacts, title=f"Square impact (board {board_idx})"
                )
                heatmap_count += 1

        if "bestmove_impact" in evaluators:
            best_move, p_orig, results = piece_pruning.evaluate_counterfactual_bestmove_impacts_store(
                board, heuristic, iterations=mcts_iterations
            )
            filtered = piece_pruning.filter_bestmove_changes(results, top_k=15)
            board_results["evaluators"]["bestmove_impact"] = {
                "best_move": best_move.uci() if best_move else None,
                "p_orig": p_orig,
                "top_changes": [
                    {
                        "cf_fen": r["cf_board"].fen() if "cf_board" in r and r["cf_board"] else None,
                        "p_cf": r.get("p_cf"),
                        "delta_abs": r.get("delta_abs"),
                    }
                    for r in filtered[:5]
                ],
            }
            if save_heatmaps and heatmap_limit and heatmap_count < heatmap_limit:
                square_best = piece_pruning.evaluate_square_bestmove_impact(
                    board, heuristic, iterations=mcts_iterations
                )
                out_path = viz_dir / f"square_bestmove_{board_idx}.svg"
                piece_pruning.save_square_heatmap(
                    out_path, board, square_best, title=f"square bestmove impact (board {board_idx})"
                )
                heatmap_count += 1

        if "model_impact" in evaluators:
            best_move, p_orig, v_orig, results = piece_pruning.evaluate_counterfactual_model_impacts_store(
                board, heuristic
            )
            filtered = piece_pruning.filter_model_changes(results, top_k=10)
            board_results["evaluators"]["model_impact"] = {
                "best_move": best_move.uci() if best_move else None,
                "p_orig": p_orig,
                "v_orig": v_orig,
                "top_changes": [
                    {
                        "delta_p_abs": r.get("delta_p_abs"),
                        "delta_v_abs": r.get("delta_v_abs"),
                    }
                    for r in filtered[:5]
                ],
            }

        all_results["boards"].append(board_results)

    results_path = save_dir / "results.json"
    results_path.write_text(json.dumps(_serialize(all_results), indent=2), encoding="utf-8")
    logger.info(f"Saved results to {results_path}")
    if viz_dir and heatmap_count > 0:
        logger.info(f"Saved {heatmap_count} heatmaps to {viz_dir}")
