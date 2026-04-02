# Project Status

## Phase

ideation

## Last updated

2026-04-02

## Key artifacts

- Paper: `latex/`, `latex/main.tex`, `latex/ieee_cog.tex`
- Code: `src/chess_experiments/` (incl. `probing.py`, `active_squares/`), `scripts/`, `configs/`
- Active-squares probing: `configs/active_squares_probing/`, `scripts/active_squares_probing.py`, `tests/test_active_squares.py`
- Configs: `configs/`, `configs/hydra/`
- Results: `results/` (git-ignored)

## Next steps

1. **Operationalize “planning” vs “pattern recognition”** in 2–3 measurable behaviors (e.g., success on deep non-forcing test positions vs matched tactical sets; policy–value agreement under counterfactuals; probe for multi-ply outcomes). Write one short paragraph each: definition, prediction, what would **falsify** your hypothesis.
2. **Pick a minimal experiment battery** (see Open decisions): start with one behavioral split (forcing vs quiet / shallow vs deep) plus one internal check (probe or intervention) so results can triangulate.
3. Run `/design-experiments` or implement the smallest scripted evaluation (curated PGN/SVG + metric script) and log configs under `configs/` so runs are reproducible.
4. Optionally run `/literature-review` on NN chess planning, probing, and “tactical vs positional” evaluation so claims are scoped to what prior work already shows.

## Open decisions / blockers

- **What would convince you?** Pre-specify primary outcomes (e.g., accuracy gap thresholds, or “internal measure X tracks behavior Y only in condition Z”). Without that, any result can be reinterpreted.
- **Which comparison is fair?** “Forcing vs non-forcing” must control for difficulty, material, and engine-eval similarity so pattern-matching difficulty is not confounded.
- **Search vs policy:** If using LC0-style models, decide whether the question is about the **network** alone (fixed nodes) vs **MCTS**—planning may live in search, not weights.
