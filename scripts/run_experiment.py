"""
Orchestrator for experiment scripts.

Run a script with its config:

```bash
uv run -m scripts.run_experiment puzzle_stats=base
uv run -m scripts.run_experiment mate_in_1_binary_probing=base
uv run -m scripts.run_experiment mate_in_1_move_probing=base
uv run -m scripts.run_experiment mate_in_1_active_squares_probing=base
uv run -m scripts.run_experiment mate_in_3_probing=base
uv run -m scripts.run_experiment piece_pruning=base
uv run -m scripts.run_experiment active_squares_probing=quick
```

By default, no script runs. Sweep with:

```bash
uv run -m scripts.run_experiment -m puzzle_stats=??? \
    hydra/sweeper=groups_optuna \
    hydra/launcher=local
```
"""

import logging
from loguru import logger
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from scripts.active_squares_probing import main as run_active_squares_probing
from scripts.cross_model_probing import main as run_cross_model_probing
from scripts.mate_in_1_active_squares_probing import main as run_mate_in_1_active_squares_probing
from scripts.mate_in_1_binary_probing import main as run_mate_in_1_binary_probing
from scripts.mate_in_1_move_probing import main as run_mate_in_1_move_probing
from scripts.mate_in_3_probing import main as run_mate_in_3_probing
from scripts.piece_pruning import main as run_piece_pruning
from scripts.puzzle_stats import main as run_puzzle_stats

SCRIPTS = {
    "puzzle_stats": run_puzzle_stats,
    "active_squares_probing": run_active_squares_probing,
    "mate_in_1_binary_probing": run_mate_in_1_binary_probing,
    "mate_in_1_move_probing": run_mate_in_1_move_probing,
    "mate_in_1_active_squares_probing": run_mate_in_1_active_squares_probing,
    "mate_in_3_probing": run_mate_in_3_probing,
    "cross_model_probing": run_cross_model_probing,
    "piece_pruning": run_piece_pruning,
}

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


@hydra.main(config_path="../configs", config_name="run_experiment.yaml", version_base=None)
def main(cfg: DictConfig):
    selected = [name for name in SCRIPTS if OmegaConf.select(cfg, name, default=None) is not None]

    if not selected:
        logger.info("No script specified; nothing to run")
        return
    if len(selected) > 1:
        logger.error(f"Only one script per run; got {selected}")
        raise SystemExit(1)

    name = selected[0]
    try:
        choice = HydraConfig.get().runtime.choices.get(name)
    except Exception:
        choice = None
    save_leaf = str(choice) if choice else "default"
    save_dir = f"results/{name}/{save_leaf}"
    SCRIPTS[name](cfg, save_dir=save_dir)


if __name__ == "__main__":
    main()
