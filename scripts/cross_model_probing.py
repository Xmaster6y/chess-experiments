"""Cross-model probing: source latents -> target move labels.

Trains linear probes on a source model's backbone to predict moves chosen by a target model.
Compares absolute best (argmax full policy) vs best legal (argmax over legal moves).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from chess_experiments import datasets, models, probing
from scripts.constants import DEFAULT_MODELS, SEED


def main(cfg: DictConfig, *, save_dir: str | None = None):
    c = cfg.cross_model_probing
    seed = c.get("seed", SEED)
    dataset = c.dataset
    model_ids = list(c.get("model_ids", DEFAULT_MODELS))
    n_samples = c.n_samples
    n_test = c.n_test
    probe_epochs = c.get("probe_epochs", 50)
    probe_batch_size = c.get("probe_batch_size", 64)

    np.random.seed(seed)
    torch.manual_seed(seed)

    total = n_samples + n_test
    boards = datasets.load_tcec_boards(dataset, total, seed)
    if len(boards) < 16:
        raise RuntimeError(f"Not enough valid boards after filtering (got {len(boards)}); increase n_samples.")
    train_boards = boards[:n_samples]
    test_boards = boards[n_samples:total]

    logger.info(f"Loaded boards: {len(boards)} | train: {len(train_boards)} | test: {len(test_boards)}")

    # Get policy size from first model
    model_temp = models.load_model(model_ids[0])
    num_classes = models.get_policy_size(model_temp, train_boards[0] if train_boards else None)
    del model_temp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {}
    for src_id in model_ids:
        for tgt_id in model_ids:
            pair_key = f"{src_id.split('/')[-1]} -> {tgt_id.split('/')[-1]}"
            results[pair_key] = {}

            source_model = models.load_model(src_id)
            target_model = models.load_model(tgt_id)

            for label_type, label_name in [("absolute", "absolute_best"), ("legal", "best_legal")]:
                metrics = probing.run_cross_model_probing(
                    source_model=source_model,
                    target_model=target_model,
                    train_boards=train_boards,
                    test_boards=test_boards,
                    label_type=label_type,
                    num_classes=num_classes,
                    probe_epochs=probe_epochs,
                    probe_batch_size=probe_batch_size,
                )
                results[pair_key][label_name] = probing.extract_layer_accuracies(metrics)

            del source_model, target_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("Probing complete.")

    save_dir = Path(save_dir or "results/cross_model_probing/default")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save results (extracted layer accuracies)
    (save_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Saved metrics to {save_dir}")

    # Build and save heatmaps
    layer_names = sorted(
        set(k for pair in results.values() for label_data in pair.values() for k in label_data.keys()),
        key=lambda x: int(x.replace("block", "")),
    )

    def build_heatmap_data(label_name: str) -> tuple[list, list, np.ndarray]:
        pairs = list(results.keys())
        data = np.zeros((len(pairs), len(layer_names)))
        for i, pair in enumerate(pairs):
            layer_accs = results[pair][label_name]
            for j, layer in enumerate(layer_names):
                data[i, j] = layer_accs.get(layer, np.nan)
        return pairs, layer_names, data

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, label_name, title in zip(
        axes,
        ["absolute_best", "best_legal"],
        ["Absolute Best (argmax over full policy)", "Best Legal (argmax over legal moves only)"],
    ):
        pairs, layers, data = build_heatmap_data(label_name)
        im = ax.imshow(data, aspect="auto", vmin=0, vmax=0.5, cmap="viridis")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha="right")
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pairs, fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Backbone layer")
        plt.colorbar(im, ax=ax, label="Top-1 accuracy")

    plt.tight_layout()
    heatmap_path = save_dir / "heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved heatmap to {heatmap_path}")
