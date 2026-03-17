"""Piece pruning evaluation: counterfactual analysis of MCTS and model policy/value.

Measures how move probabilities change when pieces are added or removed from the board.
"""

from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Literal

import chess
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from lczerolens import LczeroBoard
    from lczerolens.search import ModelHeuristic, Node

PIECE_VALUES: dict[int, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}


# --- MCTS & impact metrics ---


def run_mcts(
    board: "LczeroBoard",
    heuristic: "ModelHeuristic",
    iterations: int = 100,
) -> "Node":
    """Run MCTS from board and return the root Node."""
    from lczerolens.search import MCTS, Node

    root = Node(board=board, parent=None)
    mcts = MCTS()
    mcts.search_(root, heuristic, iterations=iterations)
    return root


def node_to_move_probs(root: "Node") -> dict[chess.Move, float]:
    """Convert root Node visit counts into move probability distribution."""
    total = root.visits.sum().item() if hasattr(root.visits, "sum") else float(root.visits.sum())
    if total <= 0:
        return {}
    return {move: (root.visits[i].item() / total) for i, move in enumerate(root.legal_moves)}


def impact_l1(p: dict[chess.Move, float], q: dict[chess.Move, float]) -> float:
    """L1 distance between two discrete distributions on possibly different supports."""
    all_moves = set(p.keys()) | set(q.keys())
    return float(sum(abs(p.get(m, 0.0) - q.get(m, 0.0)) for m in all_moves))


def compute_root_impact(
    root_a: "Node",
    root_b: "Node",
    metric: Literal["l1"] = "l1",
) -> float:
    """Compare two roots by extracting move distributions and applying a metric."""
    pa = node_to_move_probs(root_a)
    pb = node_to_move_probs(root_b)
    if metric == "l1":
        return impact_l1(pa, pb)
    raise ValueError(f"Unknown metric: {metric}")


def get_best_move_and_prob(root: "Node") -> tuple[chess.Move | None, float]:
    """Best move according to MCTS visits at root. Returns (best_move, prob_best_move)."""
    total = root.visits.sum().item() if hasattr(root.visits, "sum") else float(root.visits.sum())
    if total <= 0 or len(root.legal_moves) == 0:
        return None, 0.0
    best_idx = int(torch.argmax(root.visits).item()) if torch.is_tensor(root.visits) else int(root.visits.argmax())
    best_move = root.legal_moves[best_idx]
    best_prob = float(root.visits[best_idx].item() / total)
    return best_move, best_prob


def prob_of_move(root: "Node", move: chess.Move) -> float:
    """Probability of a specific move under root visit distribution."""
    total = root.visits.sum().item() if hasattr(root.visits, "sum") else float(root.visits.sum())
    if total <= 0:
        return 0.0
    for i, m in enumerate(root.legal_moves):
        if m == move:
            return float(root.visits[i].item() / total)
    return 0.0


def compute_bestmove_impact(
    original_root: "Node",
    cf_root: "Node",
    mode: Literal["signed", "abs"] = "abs",
) -> dict:
    """Impact focusing only on the original best move probability."""
    best_move, p_orig = get_best_move_and_prob(original_root)
    if best_move is None:
        return {"best_move": None, "p_orig": 0.0, "p_cf": 0.0, "delta_signed": 0.0, "delta_abs": 0.0}
    p_cf = prob_of_move(cf_root, best_move)
    delta_signed = float(p_cf - p_orig)
    delta_abs = float(abs(delta_signed))
    return {
        "best_move": best_move,
        "p_orig": float(p_orig),
        "p_cf": float(p_cf),
        "delta_signed": delta_signed,
        "delta_abs": delta_abs,
        "impact": delta_abs if mode == "abs" else delta_signed,
    }


def filter_bestmove_changes(
    results: list[dict],
    threshold: float = 1e-6,
    top_k: int = 20,
    sort_by: Literal["impact", "delta_abs"] = "impact",
) -> list[dict]:
    """Keep counterfactuals where best-move probability changed by at least threshold."""
    filtered = [r for r in results if float(r.get("delta_abs", 0.0)) >= threshold]
    filtered.sort(key=lambda r: float(r.get(sort_by, 0.0)), reverse=True)
    return filtered[:top_k]


# --- Model-only (no MCTS) ---


def model_evaluate_policy_value(
    board: "LczeroBoard",
    heuristic: "ModelHeuristic",
) -> tuple[tuple[chess.Move, ...], torch.Tensor, float]:
    """Returns (moves, probs, value) from model policy and value head."""
    td = heuristic.evaluate(board)
    policy_logits = td["policy"]
    probs = F.softmax(policy_logits, dim=0)
    moves = tuple(board.legal_moves)
    value = float(td["value"].item())
    return moves, probs, value


def get_model_best_move_prob_value(
    board: "LczeroBoard",
    heuristic: "ModelHeuristic",
) -> tuple[chess.Move | None, float, float]:
    """Best move from model policy only. Returns (best_move, p_best, value)."""
    moves, probs, value = model_evaluate_policy_value(board, heuristic)
    if len(moves) == 0:
        return None, 0.0, value
    best_idx = int(torch.argmax(probs).item())
    return moves[best_idx], float(probs[best_idx].item()), float(value)


def model_prob_of_move(
    board: "LczeroBoard",
    heuristic: "ModelHeuristic",
    move: chess.Move,
) -> float:
    """Probability of a move under model policy."""
    moves, probs, _ = model_evaluate_policy_value(board, heuristic)
    for i, m in enumerate(moves):
        if m == move:
            return float(probs[i].item())
    return 0.0


def compute_model_bestmove_value_impact(
    original_board: "LczeroBoard",
    cf_board: "LczeroBoard",
    heuristic: "ModelHeuristic",
    mode: Literal["signed", "abs"] = "abs",
) -> dict:
    """Delta p(best_move_orig) and delta value under model policy/value."""
    best_move, p_orig, v_orig = get_model_best_move_prob_value(original_board, heuristic)
    if best_move is None:
        _, _, v_cf = get_model_best_move_prob_value(cf_board, heuristic)
        return {
            "best_move": None,
            "p_orig": 0.0,
            "p_cf": 0.0,
            "delta_p_signed": 0.0,
            "delta_p_abs": 0.0,
            "v_orig": float(v_orig),
            "v_cf": float(v_cf),
            "delta_v_signed": float(v_cf - v_orig),
            "delta_v_abs": float(abs(v_cf - v_orig)),
        }
    p_cf = model_prob_of_move(cf_board, heuristic, best_move)
    _, _, v_cf = get_model_best_move_prob_value(cf_board, heuristic)
    dp_signed = float(p_cf - p_orig)
    dv_signed = float(v_cf - v_orig)
    return {
        "best_move": best_move,
        "p_orig": float(p_orig),
        "p_cf": float(p_cf),
        "delta_p_signed": dp_signed,
        "delta_p_abs": float(abs(dp_signed)),
        "v_orig": float(v_orig),
        "v_cf": float(v_cf),
        "delta_v_signed": dv_signed,
        "delta_v_abs": float(abs(dv_signed)),
        "impact_p": float(abs(dp_signed)) if mode == "abs" else dp_signed,
        "impact_v": float(abs(dv_signed)) if mode == "abs" else dv_signed,
    }


def filter_model_changes(
    results: list[dict],
    p_threshold: float = 1e-4,
    v_threshold: float = 1e-3,
    top_k: int = 15,
    sort_by: Literal["delta_p_abs", "delta_v_abs"] = "delta_p_abs",
) -> list[dict]:
    """Keep counterfactuals that change |Δp| or |Δv| above threshold."""
    kept = [
        r
        for r in results
        if float(r.get("delta_p_abs", 0.0)) >= p_threshold or float(r.get("delta_v_abs", 0.0)) >= v_threshold
    ]
    kept.sort(key=lambda r: float(r.get(sort_by, 0.0)), reverse=True)
    return kept[:top_k]


# --- Counterfactual generators ---


def generate_counterfactuals(board: "LczeroBoard") -> list["LczeroBoard"]:
    """Generate counterfactual boards: remove each non-king piece, add pieces on empty squares."""
    boards: list["LczeroBoard"] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            new_board = deepcopy(board)
            new_board.remove_piece_at(square)
            boards.append(new_board)
    add_piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    for square in chess.SQUARES:
        if board.piece_at(square) is None:
            for piece_type in add_piece_types:
                for color in [chess.WHITE, chess.BLACK]:
                    new_board = deepcopy(board)
                    new_board.set_piece_at(square, chess.Piece(piece_type, color))
                    boards.append(new_board)
    return boards


def generate_counterfactuals_with_metadata(board: "LczeroBoard") -> list[tuple[int, float, "LczeroBoard"]]:
    """Generate counterfactuals with (square, signed_piece_value, new_board)."""
    counterfactuals: list[tuple[int, float, "LczeroBoard"]] = []
    player = board.turn
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            new_board = deepcopy(board)
            new_board.remove_piece_at(square)
            base_value = PIECE_VALUES.get(piece.piece_type, 0.0)
            signed = (+base_value) if piece.color != player else (-base_value)
            counterfactuals.append((square, signed, new_board))
    for square in chess.SQUARES:
        if board.piece_at(square) is None:
            for piece_type, base_value in PIECE_VALUES.items():
                for color in [chess.WHITE, chess.BLACK]:
                    new_board = deepcopy(board)
                    new_board.set_piece_at(square, chess.Piece(piece_type, color))
                    signed = (+base_value) if color == player else (-base_value)
                    counterfactuals.append((square, signed, new_board))
    return counterfactuals


def generate_legal_2step_counterfactuals(board: "LczeroBoard") -> list["LczeroBoard"]:
    """Generate counterfactuals from all legal 2-move sequences."""
    legal_2step_boards = []
    for move1 in board.legal_moves:
        board.push(move1)
        secondary_moves = list(board.legal_moves)
        if not secondary_moves:
            legal_2step_boards.append(deepcopy(board))
        else:
            for move2 in secondary_moves:
                board.push(move2)
                legal_2step_boards.append(deepcopy(board))
                board.pop()
        board.pop()
    return legal_2step_boards


# --- Evaluators ---


def evaluate_counterfactual_impacts(
    original_board: "LczeroBoard",
    heuristic: "ModelHeuristic",
    iterations: int = 20,
    metric: Literal["l1"] = "l1",
    generator: Callable[["LczeroBoard"], list["LczeroBoard"]] | None = None,
) -> list[float]:
    """Run MCTS on original and all counterfactuals, return impact list."""
    root = run_mcts(original_board, heuristic, iterations=iterations)
    gen = generator or generate_counterfactuals
    impacts: list[float] = []
    for cf_board in gen(original_board):
        cf_root = run_mcts(cf_board, heuristic, iterations=iterations)
        impacts.append(compute_root_impact(root, cf_root, metric=metric))
    return impacts


def evaluate_square_impact(
    original_board: "LczeroBoard",
    heuristic: "ModelHeuristic",
    iterations: int = 20,
    metric: Literal["l1"] = "l1",
    weighting: Literal["signed", "abs", "none"] = "signed",
) -> dict[int, float]:
    """Compute mean impact score per square using counterfactuals with metadata."""
    root = run_mcts(original_board, heuristic, iterations=iterations)
    counterfactuals = generate_counterfactuals_with_metadata(original_board)
    square_impacts: dict[int, list[float]] = defaultdict(list)
    for square, signed_value, cf_board in counterfactuals:
        cf_root = run_mcts(cf_board, heuristic, iterations=iterations)
        impact = compute_root_impact(root, cf_root, metric=metric)
        if weighting == "signed":
            weighted = impact * signed_value
        elif weighting == "abs":
            weighted = impact * abs(signed_value)
        elif weighting == "none":
            weighted = impact
        else:
            raise ValueError(f"Unknown weighting: {weighting}")
        square_impacts[square].append(float(weighted))
    return {sq: (sum(vals) / len(vals)) for sq, vals in square_impacts.items() if len(vals) > 0}


def evaluate_pair_impact(
    original_board: "LczeroBoard",
    new_board: "LczeroBoard",
    heuristic: "ModelHeuristic",
    iterations: int = 20,
    metric: Literal["l1"] = "l1",
) -> float:
    """Compare two boards by MCTS root distribution shift."""
    root_a = run_mcts(original_board, heuristic, iterations=iterations)
    root_b = run_mcts(new_board, heuristic, iterations=iterations)
    return compute_root_impact(root_a, root_b, metric=metric)


def evaluate_counterfactual_bestmove_impacts(
    original_board: "LczeroBoard",
    heuristic: "ModelHeuristic",
    iterations: int = 20,
    mode: Literal["signed", "abs"] = "abs",
    generator: Callable[["LczeroBoard"], list["LczeroBoard"]] | None = None,
) -> tuple[chess.Move | None, float, list[dict]]:
    """MCTS best move impact per counterfactual. Returns (best_move, p_orig, results)."""
    root = run_mcts(original_board, heuristic, iterations=iterations)
    best_move, p_orig = get_best_move_and_prob(root)
    gen = generator or generate_counterfactuals
    results: list[dict] = []
    for cf_board in gen(original_board):
        cf_root = run_mcts(cf_board, heuristic, iterations=iterations)
        results.append(compute_bestmove_impact(root, cf_root, mode=mode))
    return best_move, float(p_orig), results


def evaluate_counterfactual_bestmove_impacts_store(
    original_board: "LczeroBoard",
    heuristic: "ModelHeuristic",
    iterations: int = 20,
    mode: Literal["signed", "abs"] = "abs",
    generator: Callable[["LczeroBoard"], list["LczeroBoard"]] | None = None,
) -> tuple[chess.Move | None, float, list[dict]]:
    """Same as evaluate_counterfactual_bestmove_impacts but stores cf_board in each result."""
    root = run_mcts(original_board, heuristic, iterations=iterations)
    best_move, p_orig = get_best_move_and_prob(root)
    gen = generator or generate_counterfactuals
    results: list[dict] = []
    for cf_board in gen(original_board):
        cf_root = run_mcts(cf_board, heuristic, iterations=iterations)
        info = compute_bestmove_impact(root, cf_root, mode=mode)
        info["cf_board"] = cf_board
        results.append(info)
    return best_move, float(p_orig), results


def evaluate_square_bestmove_impact(
    original_board: "LczeroBoard",
    heuristic: "ModelHeuristic",
    iterations: int = 20,
    mode: Literal["signed", "abs"] = "abs",
    weighting: Literal["signed", "abs", "none"] = "signed",
) -> dict[int, float]:
    """Square attribution using only original best move probability shift."""
    root = run_mcts(original_board, heuristic, iterations=iterations)
    best_move, p_orig = get_best_move_and_prob(root)
    counterfactuals = generate_counterfactuals_with_metadata(original_board)
    square_impacts: dict[int, list[float]] = defaultdict(list)
    for square, signed_value, cf_board in counterfactuals:
        cf_root = run_mcts(cf_board, heuristic, iterations=iterations)
        p_cf = prob_of_move(cf_root, best_move) if best_move is not None else 0.0
        delta_signed = float(p_cf - p_orig)
        delta = float(abs(delta_signed)) if mode == "abs" else delta_signed
        if weighting == "signed":
            weighted = delta * signed_value
        elif weighting == "abs":
            weighted = delta * abs(signed_value)
        elif weighting == "none":
            weighted = delta
        else:
            raise ValueError(f"Unknown weighting: {weighting}")
        square_impacts[square].append(float(weighted))
    return {sq: (sum(vals) / len(vals)) for sq, vals in square_impacts.items() if len(vals) > 0}


def evaluate_counterfactual_model_impacts_store(
    original_board: "LczeroBoard",
    heuristic: "ModelHeuristic",
    mode: Literal["signed", "abs"] = "abs",
    generator: Callable[["LczeroBoard"], list["LczeroBoard"]] | None = None,
) -> tuple[chess.Move | None, float, float, list[dict]]:
    """Model-only impacts (policy + value). Returns (best_move, p_orig, v_orig, results)."""
    best_move, p_orig, v_orig = get_model_best_move_prob_value(original_board, heuristic)
    gen = generator or generate_counterfactuals
    results: list[dict] = []
    for cf_board in gen(original_board):
        info = compute_model_bestmove_value_impact(original_board, cf_board, heuristic, mode=mode)
        info["cf_board"] = cf_board
        results.append(info)
    return best_move, float(p_orig), float(v_orig), results
