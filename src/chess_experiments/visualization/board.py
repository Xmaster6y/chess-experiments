"""Board SVG rendering and arrow helpers."""

from typing import TYPE_CHECKING

import chess
import chess.svg

if TYPE_CHECKING:
    from lczerolens import LczeroBoard

GT_ARROW_COLOR = "#22c55e"
WRONG_ARROW_COLOR = "#ef4444"


def build_arrows(
    gt_uci: str,
    model_predictions: dict[str, tuple[str, bool]],
) -> list[chess.svg.Arrow]:
    """Build arrows: ground truth (green) + wrong predictions (red)."""
    arrows: list[chess.svg.Arrow] = []
    try:
        move = chess.Move.from_uci(gt_uci)
        arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=GT_ARROW_COLOR))
    except ValueError:
        pass
    for pred_uci, is_correct in model_predictions.values():
        if is_correct or pred_uci == gt_uci:
            continue
        try:
            move = chess.Move.from_uci(pred_uci)
            arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=WRONG_ARROW_COLOR))
        except ValueError:
            pass
    return arrows


def render_board_svg(
    board: "LczeroBoard",
    arrows: list[chess.svg.Arrow] | None = None,
    size: int = 400,
    relative_view: bool = True,
    fill: dict[int, str] | None = None,
) -> str:
    """Render board as SVG using chess.svg.board."""
    orientation = board.turn if relative_view else chess.WHITE
    return chess.svg.board(
        board,
        orientation=orientation,
        arrows=arrows or [],
        fill=fill or {},
        size=size,
    )
