"""MetaboNet glucose-prediction experiments.

Pipeline:  build_dataset -> splits -> models -> train -> run_all
See DATA_PIPELINE.md for every data decision and RUN.md for the commands.
"""
__all__ = ["config", "clean", "segments", "features", "build_dataset", "splits",
           "metrics", "models", "train", "run_all"]
