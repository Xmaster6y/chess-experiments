"""Puzzle comparison visualization (SVG board + metadata for paper captions)."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from chess_experiments.visualization.board import build_arrows, render_board_svg

if TYPE_CHECKING:
    from lczerolens import LczeroBoard


def render_puzzle_board(
    board: "LczeroBoard",
    gt_uci: str,
    model_predictions: dict[str, tuple[str, bool]],
    size: int = 390,
) -> str:
    """Render puzzle board with arrows as SVG (no HTML).

    Arrows: green (ground truth), red (wrong predictions).
    model_predictions: {model_name: (pred_uci, is_correct)}
    """
    arrows = build_arrows(gt_uci, model_predictions)
    return render_board_svg(board, arrows=arrows, size=size)


def save_puzzle_viz(
    base_path: Path,
    board: "LczeroBoard",
    gt_uci: str,
    model_predictions: dict[str, tuple[str, bool]],
    puzzle_idx: int = 0,
) -> None:
    """Save puzzle visualization as SVG board + JSON metadata (paper-friendly).

    Writes {base_path}_board.svg and {base_path}_meta.json.
    Use meta.json for figure captions.
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    svg_content = render_puzzle_board(board, gt_uci, model_predictions)
    board_path = base_path.parent / f"{base_path.name}_board.svg"
    board_path.write_text(svg_content, encoding="utf-8")

    meta = {
        "puzzle_idx": puzzle_idx,
        "gt_uci": gt_uci,
        "model_predictions": {
            name: {"pred_uci": pred_uci, "is_correct": is_correct}
            for name, (pred_uci, is_correct) in model_predictions.items()
        },
    }
    meta_path = base_path.parent / f"{base_path.name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
