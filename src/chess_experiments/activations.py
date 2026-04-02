"""Collect backbone activations via tdhook ActivationCaching."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict
from tdhook.latent.activation_caching import ActivationCaching

from scripts.constants import BACKBONE_PATTERN

if TYPE_CHECKING:
    from lczerolens import LczeroModel


def sort_layer_keys(keys: list[str]) -> list[str]:
    """Stable order by block index in key string."""

    def sort_key(k: str) -> tuple[int, str]:
        m = re.search(r"block(\d+)", k)
        return (int(m.group(1)) if m else 9999, k)

    return sorted(keys, key=sort_key)


def collect_backbone_activations(
    model: "LczeroModel",
    boards: list,
    *,
    backbone_pattern: str | None = None,
    batch_size: int = 64,
    in_keys: list[str] | None = None,
    out_keys: list[str] | None = None,
    input_encoding=None,
) -> tuple[list[str], torch.Tensor]:
    """Run forwards with ActivationCaching; return (layer_keys, tensor [L, B, C, H, W]).

    boards: LczeroBoard or objects with `.board`.
    """
    pattern = backbone_pattern or BACKBONE_PATTERN
    in_keys = in_keys or ["board"]
    out_keys = out_keys or ["policy", "wdl"]

    def _get_board(obj):
        return obj.board if hasattr(obj, "board") else obj

    layer_keys: list[str] = []
    chunks: list[torch.Tensor] = []

    device = next(model.parameters()).device

    with ActivationCaching(pattern).prepare(model, in_keys=in_keys, out_keys=out_keys) as hooked:
        for start in range(0, len(boards), batch_size):
            batch = boards[start : start + batch_size]
            board_kwargs = {"input_encoding": input_encoding} if input_encoding is not None else {}
            board_t = model.prepare_boards(*[_get_board(b) for b in batch], **board_kwargs)
            td = TensorDict({"board": board_t}, batch_size=[len(batch)])
            with torch.no_grad():
                hooked(td)
            cache = hooked.hooking_context.cache
            keys = sort_layer_keys([k for k in cache.keys() if isinstance(k, str)])
            if not keys:
                raise RuntimeError("ActivationCaching produced no string keys; check backbone_pattern")
            if not layer_keys:
                layer_keys = keys
            elif layer_keys != keys:
                raise RuntimeError(f"Layer key mismatch: {layer_keys} vs {keys}")

            stack = torch.stack([cache[k].detach() for k in keys], dim=0)
            if stack.dim() != 5:
                raise RuntimeError(f"Expected [L,B,C,H,W] stack, got {stack.shape}")
            chunks.append(stack)

    full = torch.cat(chunks, dim=1).to(device)
    return layer_keys, full


def layer_key_to_probe_id(layer_key: str) -> str:
    """Short probe id from a layer cache key (e.g. block index) for metrics dicts."""
    m = re.search(r"block(\d+)", layer_key)
    return f"block{m.group(1)}" if m else layer_key
