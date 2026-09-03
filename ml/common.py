"""Shared preprocessing code for the E-commerce Customer Behavior application.

Imported both by the training notebook (notebook/customer_behavior.ipynb) and by the
deployment API (api/app.py), so training-time and inference-time preprocessing can never
drift apart (see Assignment 02, Part XI / "Important: Data Leakage").
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC = ["Age", "Rating", "Positive Feedback Count"]
CATEGORICAL = ["Division Name", "Department Name", "Class Name"]
TEXT_COL = "Review Text"
TABULAR_FEATURES = NUMERIC + CATEGORICAL
FEATURE_ORDER = NUMERIC + CATEGORICAL + [TEXT_COL]
TARGET = "Recommended IND"


class TextCleaner(BaseEstimator, TransformerMixin):
    """Minimal, deterministic text-cleaning rule applied identically everywhere:
    missing comments -> empty string, collapse whitespace, strip. No statistics are
    learned from data, so this is leakage-safe. Accepts a 1D array/Series (as passed by
    ColumnTransformer for a single text column) and returns a 1D array of strings, ready
    for TfidfVectorizer (Text -> Tokens -> Token IDs -> Vector, Part IV of the PDF).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for v in np.asarray(X).ravel():
            s = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
            s = " ".join(s.split())  # collapse repeated/irregular whitespace
            out.append(s)
        return np.array(out)
