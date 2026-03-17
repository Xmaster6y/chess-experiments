"""Heatmap visualization using LczeroBoard.render_heatmap (SVG output)."""

from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lczerolens import LczeroBoard


def square_dict_to_heatmap_tensor(
    square_values: dict[int, float],
    device: str = "cpu",
) -> torch.Tensor:
    """Convert {square: value} to torch.Tensor(64) for LczeroBoard.render_heatmap."""
    heatmap = torch.zeros(64, device=device)
    for square, value in square_values.items():
        heatmap[square] = float(value)
    return heatmap


def save_square_heatmap(
    path: Path,
    board: "LczeroBoard",
    square_impacts: dict[int, float],
    title: str = "Square impact heatmap",
    cmap: str = "inferno",
    alpha: float = 0.85,
) -> None:
    """Save square impact heatmap as SVG (paper-friendly).

    path must end with .svg. Writes {base}_board.svg and {base}_colorbar.svg
    via LczeroBoard.render_heatmap.
    """
    path = Path(path)
    if path.suffix.lower() != ".svg":
        raise ValueError("path must end with .svg; only SVG output is supported")
    path.parent.mkdir(parents=True, exist_ok=True)

    heatmap = square_dict_to_heatmap_tensor(square_impacts)
    board.render_heatmap(
        heatmap=heatmap,
        normalise="abs",
        cmap_name=cmap,
        alpha=alpha,
        heatmap_mode="absolute",
        save_to=str(path),
    )
