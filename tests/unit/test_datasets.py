"""Unit tests for chess_experiments.datasets."""

import pytest

from chess_experiments.datasets import row_to_board, row_to_mate_sample, row_to_mate3_puzzle


@pytest.mark.unit
def test_row_to_board_valid(global_fixture):
    row = {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}
    board = row_to_board(row)
    assert board is not None
    assert global_fixture == "global_fixture"


@pytest.mark.unit
def test_row_to_board_fen_key():
    row = {"FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}
    board = row_to_board(row)
    assert board is not None


@pytest.mark.unit
def test_row_to_board_invalid():
    assert row_to_board({}) is None
    assert row_to_board({"fen": ""}) is None


@pytest.mark.unit
def test_row_to_mate_sample_valid():
    row = {
        "FEN": "6k1/5ppp/8/8/8/8/8/R7 b - - 0 1",
        "Moves": "f7f6 a1a8",
    }
    sample = row_to_mate_sample(row)
    assert sample is not None
    assert sample.is_mate_in_one
    assert sample.move_idx is not None


@pytest.mark.unit
def test_row_to_mate_sample_invalid():
    assert row_to_mate_sample({}) is None
    assert row_to_mate_sample({"FEN": "8/8/8/8/8/8/8/8 w - - 0 1", "Moves": ""}) is None


@pytest.mark.unit
def test_row_to_mate3_puzzle_invalid():
    assert row_to_mate3_puzzle({}) is None
    assert row_to_mate3_puzzle({"FEN": "8/8/8/8/8/8/8/8 w - - 0 1", "Moves": "e2e4"}) is None
