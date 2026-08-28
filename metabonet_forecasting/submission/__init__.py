"""Competition submissions: a trained run -> a template-shaped parquet.

Kept apart from the benchmark on purpose. Everything under ``evaluation`` scores
the project's own frozen splits; this package exists solely to produce the file
the organisers accept, and it reuses the training pipeline end to end so that
inference features cannot drift from training features.
"""

from .predict import predict_template, load_run
from .recentre import CENTRE, fit_mu, multiplier, fit_from_predictions

__all__ = ["predict_template", "load_run",
           "CENTRE", "fit_mu", "multiplier", "fit_from_predictions"]
