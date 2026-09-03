"""Export the fitted scikit-learn pipeline to a plain-JSON bundle the browser can run.

The trained pipeline (notebook/customer_behavior.ipynb -> model/model_pipeline.joblib)
is a Logistic Regression on three concatenated blocks:

    [ 3 numeric  ]  median-impute -> standard-scale
    [ 29 one-hot ]  most-frequent-impute -> OneHotEncoder(handle_unknown='ignore')
    [ 2000 tfidf ]  TextCleaner -> TfidfVectorizer(1-2 grams, english stop words, l2)

Every inference-time operation over those blocks is a lookup, a multiply and a dot
product, so the deployed site needs no Python at all: this script writes the constants
to ../public/model.json and src/model/infer.js redoes the same arithmetic in the browser.
Run parity_check.py afterwards to prove the two agree.

Usage:  python ml/export_model.py            # writes ../public/model.json
        python ml/export_model.py OUT.json
"""
import json
import os
import sys

import numpy as np

ML_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(ML_DIR)
sys.path.insert(0, ML_DIR)

import joblib  # noqa: E402
import common  # noqa: E402,F401  (defines TextCleaner, needed to unpickle the pipeline)
from common import NUMERIC, CATEGORICAL, TEXT_COL  # noqa: E402

MODEL_PATH = os.path.join(ML_DIR, "model", "model_pipeline.joblib")
DEFAULT_OUT = os.path.join(REPO_DIR, "public", "model.json")


def round_floats(values, ndigits=12):
    """Trim trailing float noise so the JSON stays small. 12 digits keeps the browser's
    probability within ~1e-11 of scikit-learn's (verified by parity_check.py) while
    shaving roughly a third off the file compared with full double precision."""
    return [float(round(float(v), ndigits)) for v in values]


def build_bundle(pipe):
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]

    num_pipe = prep.named_transformers_["num"]
    cat_pipe = prep.named_transformers_["cat"]
    txt_pipe = prep.named_transformers_["text"]

    num_imputer = num_pipe.named_steps["imputer"]
    scaler = num_pipe.named_steps["scaler"]
    cat_imputer = cat_pipe.named_steps["imputer"]
    ohe = cat_pipe.named_steps["onehot"]
    tfidf = txt_pipe.named_steps["tfidf"]

    # The JS port reimplements exactly this configuration; refuse to export anything else.
    assert tfidf.analyzer == "word", tfidf.analyzer
    assert tfidf.norm == "l2" and not tfidf.sublinear_tf and not tfidf.binary
    assert tfidf.strip_accents is None

    n_num = len(NUMERIC)
    n_cat = sum(len(c) for c in ohe.categories_)
    n_txt = len(tfidf.vocabulary_)

    coef = np.asarray(clf.coef_).ravel()
    assert coef.shape[0] == n_num + n_cat + n_txt, (
        f"coef width {coef.shape[0]} != {n_num}+{n_cat}+{n_txt}"
    )

    # vocabulary_ maps term -> column index; invert it so idf and coef can be stored as
    # arrays parallel to the term list.
    terms = [None] * n_txt
    for term, idx in tfidf.vocabulary_.items():
        terms[idx] = term
    assert all(t is not None for t in terms)

    return {
        "app": "customer_behavior",
        "task": "classification",
        "model": "LogisticRegression",
        "classes": [int(c) for c in clf.classes_],
        "intercept": float(clf.intercept_[0]),
        "numeric": {
            "names": list(NUMERIC),
            "fill": round_floats(num_imputer.statistics_),   # median learned on TRAIN
            "mean": round_floats(scaler.mean_),
            "scale": round_floats(scaler.scale_),
            "coef": round_floats(coef[:n_num]),
        },
        "categorical": {
            "names": list(CATEGORICAL),
            # most-frequent category on TRAIN, used when a field is left blank
            "fill": [str(v) for v in cat_imputer.statistics_],
            # an unseen category encodes as an all-zero block (handle_unknown='ignore')
            "categories": [[str(v) for v in cats] for cats in ohe.categories_],
            "coef": round_floats(coef[n_num:n_num + n_cat]),
        },
        "text": {
            "column": TEXT_COL,
            "lowercase": bool(tfidf.lowercase),
            "ngram_range": list(tfidf.ngram_range),
            "norm": tfidf.norm,
            "sublinear_tf": bool(tfidf.sublinear_tf),
            "binary": bool(tfidf.binary),
            "token_pattern": tfidf.token_pattern,
            "stop_words": sorted(tfidf.get_stop_words()),
            "vocab": terms,
            "idf": round_floats(tfidf.idf_),
            "coef": round_floats(coef[n_num + n_cat:]),
        },
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    bundle = build_bundle(joblib.load(MODEL_PATH))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))

    n_num = len(bundle["numeric"]["names"])
    n_cat = len(bundle["categorical"]["coef"])
    n_txt = len(bundle["text"]["vocab"])
    # relative path: absolute ones can carry characters a cp1252 console cannot print
    shown = os.path.relpath(out_path, REPO_DIR)
    print(f"wrote {shown}  ({os.path.getsize(out_path) / 1024:.1f} KB)")
    print(f"  numeric={n_num} categorical={n_cat} tfidf={n_txt} total={n_num + n_cat + n_txt}")
    print(f"  stop_words={len(bundle['text']['stop_words'])} intercept={bundle['intercept']:.6f}")


if __name__ == "__main__":
    main()
