"""
CONSEQUENCE OF SHORTCUTS: cross-dataset transfer, and benchmark contamination.

Part 1  overlap  : exact-text overlap between FineFake / WELFake / ISOT.
                   WELFake was assembled by merging four corpora (one of them
                   the Kaggle real-vs-fake set == ISOT), so "cross-dataset"
                   evaluation between them may be leakage, not transfer.
Part 2  transfer : train on A, test on B, with any text appearing in A removed
                   from B first, so the number is honest out-of-distribution
                   generalisation. Run for the RAW model and for the
                   length-de-biased model.

The claim under test: shortcut-heavy in-distribution scores do not survive a
distribution shift, and de-biasing trades in-distribution accuracy for
out-of-distribution robustness.

    python scripts/shortcut_transfer.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer   # noqa: E402
from sklearn.linear_model import LogisticRegression           # noqa: E402
from sklearn.metrics import f1_score                          # noqa: E402

from src import data as D                                     # noqa: E402
from scripts.shortcut_audit import (                          # noqa: E402
    ISOT_DIR, SEED, length_balance, load_welfake, strat_split)

OUT = REPO / "results"


def norm_key(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.lower().str.replace(r"\W+", " ", regex=True)
            .str.strip())


def mf1(y, p):
    return float(f1_score(y, p, average="macro"))


def train_model(tr):
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                          max_features=100_000, sublinear_tf=True,
                          strip_accents="unicode")
    X = vec.fit_transform(tr["text"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=SEED).fit(X, tr["label"])
    return vec, clf


def main() -> None:
    # ---------------- load, all on the repo convention fake=0 real=1 --------
    fine = D.load_finefake()
    wel = load_welfake()
    # WELFake ships 0=real, 1=fake (verified: 21,266 "(Reuters)" datelines sit
    # in label 0 versus 16 in label 1). Flip onto the repo convention.
    wel = wel.copy()
    wel["label"] = 1 - wel["label"]
    isot = D.load_isot(ISOT_DIR)
    isot = isot[~isot["text"].str.strip().str.lower().duplicated(keep="first")]
    isot = isot.reset_index(drop=True)
    sets = {"FineFake": fine, "WELFake": wel, "ISOT": isot}
    keys = {k: set(norm_key(v["text"])) for k, v in sets.items()}

    # ---------------- Part 1: contamination ---------------------------------
    print("=========== EXACT-TEXT OVERLAP BETWEEN BENCHMARKS ===========")
    ov_rows = []
    for a in sets:
        for b in sets:
            if a >= b:
                continue
            inter = len(keys[a] & keys[b])
            ov_rows.append({"dataset_a": a, "dataset_b": b,
                            "n_a": len(sets[a]), "n_b": len(sets[b]),
                            "shared_texts": inter,
                            "pct_of_a": round(100 * inter / len(sets[a]), 2),
                            "pct_of_b": round(100 * inter / len(sets[b]), 2)})
    ov = pd.DataFrame(ov_rows)
    print(ov.to_string(index=False), flush=True)
    ov.to_csv(OUT / "benchmark_overlap.csv", index=False)

    # ---------------- Part 2: honest transfer -------------------------------
    print("\n=========== CROSS-DATASET TRANSFER (contamination removed) ===========",
          flush=True)
    rows = []
    for tr_name, tr_df in sets.items():
        for cond in ("raw", "length-de-biased"):
            src = tr_df if cond == "raw" else length_balance(tr_df)
            tr, te_in = strat_split(src)
            vec, clf = train_model(tr)
            in_dist = mf1(te_in["label"], clf.predict(vec.transform(te_in["text"])))
            rows.append({"train_on": tr_name, "condition": cond,
                         "test_on": tr_name + " (in-dist)",
                         "n_test": len(te_in), "macro_f1": round(in_dist, 4)})
            print(f"  {tr_name:9s} [{cond:16s}] in-dist            "
                  f"{in_dist:.4f}", flush=True)

            seen = set(norm_key(tr["text"]))
            for te_name, te_df in sets.items():
                if te_name == tr_name:
                    continue
                k = norm_key(te_df["text"])
                clean = te_df[~k.isin(seen)]
                if len(clean) < 200:
                    continue
                score = mf1(clean["label"],
                            clf.predict(vec.transform(clean["text"])))
                rows.append({"train_on": tr_name, "condition": cond,
                             "test_on": te_name, "n_test": len(clean),
                             "macro_f1": round(score, 4),
                             "dropped_as_contaminated": len(te_df) - len(clean)})
                print(f"  {tr_name:9s} [{cond:16s}] -> {te_name:9s} "
                      f"{score:.4f}   (n={len(clean):,}, "
                      f"removed {len(te_df)-len(clean):,} overlapping)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "transfer_matrix.csv", index=False)
    print("\n=========== TRANSFER MATRIX ===========")
    print(res.to_string(index=False))
    print(f"\nwrote {OUT/'benchmark_overlap.csv'} and {OUT/'transfer_matrix.csv'}")


if __name__ == "__main__":
    main()
