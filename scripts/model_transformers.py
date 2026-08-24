"""Small reusable model-only transformers.

This module does not load data or train any model.  It exists so saved sklearn
pipelines can reload the transformer from Streamlit without depending on the
training script's ``__main__`` module.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureWeightTransformer(TransformerMixin, BaseEstimator):
    """Multiply each normalized input feature by a fixed model-specific weight."""

    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0)):
        self.weights = weights

    def fit(self, X, y=None):
        if X.shape[1] != len(self.weights):
            raise ValueError(
                "FeatureWeightTransformer received "
                f"{X.shape[1]} features but {len(self.weights)} weights."
            )
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * np.asarray(
            self.weights,
            dtype=float,
        )
