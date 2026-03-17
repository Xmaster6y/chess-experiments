"""Dataset loaders for chess puzzles and boards."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
from datasets import Dataset, load_dataset

from scripts.constants import SEED

if TYPE_CHECKING:
    from lczerolens import LczeroBoard


@dataclass
class MateInOneSample:
    """Mate-in-1 puzzle: board after opponent move, with mating move index."""

    board: "LczeroBoard"
    is_mate_in_one: bool
    move_idx: int | None = None


def row_to_board(row: dict) -> "LczeroBoard | None":
    """Convert a dataset row to LczeroBoard. Handles FEN/fen keys."""
    from lczerolens import LczeroBoard

    fen = row.get("FEN") or row.get("fen") if isinstance(row, dict) else None
    if not fen:
        return None
    try:
        return LczeroBoard(fen)
    except Exception:
        return None


def row_to_mate_sample(row: dict) -> MateInOneSample | None:
    """Convert mate-in-1 row to MateInOneSample. Applies opponent move, returns board + move_idx."""
    from lczerolens import LczeroBoard

    fen = row.get("FEN") or row.get("fen")
    moves_str = row.get("Moves")
    if not fen or not moves_str:
        return None
    try:
        board = LczeroBoard(fen)
    except Exception:
        return None
    parts = moves_str.split()
    if len(parts) < 2:
        return None
    try:
        opp_move = chess.Move.from_uci(parts[0])
        mating_move = chess.Move.from_uci(parts[1])
    except ValueError:
        return None
    if opp_move not in board.legal_moves:
        return None
    board.push(opp_move)
    if mating_move not in board.legal_moves:
        return None
    move_idx = board.encode_move(mating_move, us=board.turn)
    return MateInOneSample(board=board, is_mate_in_one=True, move_idx=int(move_idx))


def row_to_mate3_puzzle(row: dict) -> tuple["LczeroBoard", int] | None:
    """Convert mate-in-3 row to (board, move_idx). Board after opp move, our first move."""
    from lczerolens import LczeroBoard

    fen = row.get("FEN") or row.get("fen")
    moves_str = row.get("Moves")
    if not fen or not moves_str:
        return None
    try:
        board = LczeroBoard(fen)
    except Exception:
        return None
    parts = moves_str.split()
    if len(parts) < 6:
        return None
    try:
        opp1 = chess.Move.from_uci(parts[0])
        our1 = chess.Move.from_uci(parts[1])
    except ValueError:
        return None
    if opp1 not in board.legal_moves:
        return None
    board.push(opp1)
    if our1 not in board.legal_moves:
        return None
    move_idx = board.encode_move(our1, us=board.turn)
    return board, int(move_idx)


def load_mate_in_one_puzzles(
    dataset_id: str,
    n: int,
    seed: int = SEED,
) -> list[MateInOneSample]:
    """Load mate-in-1 puzzles. Returns list of MateInOneSample."""
    ds = load_dataset(dataset_id, split="train")
    ds = ds.shuffle(seed=seed)
    samples: list[MateInOneSample] = []
    for row in ds:
        if len(samples) >= n:
            break
        sample = row_to_mate_sample(dict(row))
        if sample is not None:
            samples.append(sample)
    return samples


def load_mate_in_3_puzzles(
    dataset_id: str,
    n: int,
    seed: int = SEED,
) -> list[tuple["LczeroBoard", int]]:
    """Load mate-in-3 puzzles. Returns list of (board, move_idx)."""
    from lczerolens import LczeroBoard

    ds = load_dataset(dataset_id, split="train")
    ds = ds.shuffle(seed=seed)
    puzzles: list[tuple[LczeroBoard, int]] = []
    for row in ds:
        if len(puzzles) >= n:
            break
        p = row_to_mate3_puzzle(dict(row))
        if p is not None:
            puzzles.append(p)
    return puzzles


def load_tcec_boards(
    dataset_id: str,
    n: int,
    seed: int = SEED,
    streaming: bool = True,
) -> list["LczeroBoard"]:
    """Load TCEC boards. Returns list of LczeroBoard."""
    from lczerolens import LczeroBoard

    if streaming:
        raw_stream = load_dataset(dataset_id, split="train", streaming=True)
        stream_subset = raw_stream.shuffle(seed=seed).take(n * 5)
        boards: list[LczeroBoard] = []
        for row in stream_subset:
            if len(boards) >= n:
                break
            b = row_to_board(dict(row))
            if b is not None:
                boards.append(b)
        return boards
    else:
        ds = load_dataset(dataset_id, split="train")
        ds = ds.shuffle(seed=seed)
        boards = []
        for row in ds:
            if len(boards) >= n:
                break
            b = row_to_board(dict(row))
            if b is not None:
                boards.append(b)
        return boards


def to_eager_subset(stream_ds, n: int, seed: int = SEED) -> Dataset:
    """Convert streaming dataset to eager subset of n samples."""
    stream_ds = stream_ds.shuffle(seed=seed).take(n)
    return Dataset.from_generator(lambda: (yield from stream_ds), features=stream_ds.features)
