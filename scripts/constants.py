"""Constants shared across experiment scripts."""

SEED = 42
BACKBONE_PATTERN = r".*block\d/conv2/relu.*"
D_LATENT = 64 * 8 * 8
DEFAULT_DATASETS = {
    "mate_in_1": "lczerolens/lichess-puzzles-mate-in-1",
    "mate_in_3": "lczerolens/lichess-puzzles-mate-in-3",
    "tcec": "lczerolens/tcec-boards",
}
DEFAULT_MODELS = ["lczerolens/maia-1100", "lczerolens/maia-1500", "lczerolens/maia-1900"]
