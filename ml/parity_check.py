"""Prove the browser port reproduces scikit-learn exactly.

Runs the original .joblib pipeline over every row of data/ecommerce_raw.csv, then hands
the same rows to ../src/model/infer.js through Node and compares the two probabilities.
Exits non-zero if any predicted label differs or the numeric gap exceeds 1e-9.

Usage:  python ml/parity_check.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ML_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(ML_DIR)
sys.path.insert(0, ML_DIR)

import joblib  # noqa: E402
import common  # noqa: E402,F401  (defines TextCleaner, needed to unpickle)
from common import FEATURE_ORDER  # noqa: E402


def dump_cases(path):
    df = pd.read_csv(os.path.join(ML_DIR, "data", "ecommerce_raw.csv"))[FEATURE_ORDER]
    pipe = joblib.load(os.path.join(ML_DIR, "model", "model_pipeline.joblib"))
    proba = pipe.predict_proba(df)[:, 1]

    rows = []
    for rec, p in zip(df.to_dict(orient="records"), proba):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                clean[k] = None          # missing review text -> null, like the web form
            elif isinstance(v, np.integer):
                clean[k] = int(v)
            elif isinstance(v, np.floating):
                clean[k] = float(v)
            else:
                clean[k] = v
        rows.append({"input": clean, "proba": float(p)})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    return len(rows)


def main():
    tmp = tempfile.mkdtemp(prefix="parity-")
    cases_path = os.path.join(tmp, "cases.json")
    n = dump_cases(cases_path)
    print(f"scikit-learn scored {n} rows")

    result = subprocess.run(
        [
            "node",
            os.path.join(ML_DIR, "parity.mjs"),
            os.path.join(REPO_DIR, "public", "model.json"),
            cases_path,
            os.path.join(REPO_DIR, "src", "model", "infer.js"),
        ],
        check=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
