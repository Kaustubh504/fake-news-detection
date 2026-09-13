"""
Cross-corpus transfer test: FineFake-trained systems evaluated on ISOT.

Neither the baseline nor the Approach-2 cluster/router is retrained on ISOT.
Every ISOT article is pushed through the SAME FineFake-fitted TF-IDF/SVD
pipeline, routed to its nearest FineFake cluster, and classified by that
cluster's specialist. This measures how the two systems generalise to a corpus
they have never seen (the report's "cross-corpus macro-F1" robustness metric).

Run from the repo root:  python scripts/eval_isot_transfer.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, splits, features as F, metrics as M
from src.approaches import approach_a as A

SEED = 42


def route_and_classify(X_sparse, cluster_assignment, cluster_models, fallback_clf, fallback_clusters):
    n = X_sparse.shape[0]
    pred = np.empty(n, dtype=int)
    prob = np.empty(n, dtype=float)
    for c in np.unique(cluster_assignment):
        mask = cluster_assignment == c
        clf = fallback_clf if c in fallback_clusters else cluster_models.get(c, fallback_clf)
        Xc = X_sparse[mask]
        fake_col = list(clf.classes_).index(0)
        pred[mask] = clf.predict(Xc)
        prob[mask] = clf.predict_proba(Xc)[:, fake_col]
    return pred, prob


def main():
    # ---- FineFake-fitted artifacts (produced by the Approach-2 run) ----
    vec = F.load_pipeline("approach_b_tfidf")
    svd = F.load_pipeline("approach_b_svd")
    kmeans = F.load_pipeline("approach_b_kmeans")
    bundle = F.load_pipeline("approach_b_cluster_models")
    cluster_models = bundle["models"]
    fallback_clf = bundle["fallback"]          # == the global baseline (all FineFake train rows)
    fallback_clusters = bundle["fallback_clusters"]

    # ---- ISOT, entirely held out ----
    isot = data.load_isot()
    y = isot["label"].to_numpy()
    print(f"ISOT: {len(isot):,} rows   fake={int((y==0).sum())} real={int((y==1).sum())}")

    X_sparse = F.transform(vec, isot["text"])
    X_dense = svd.transform(X_sparse)

    # ---- baseline (single global FineFake classifier) on ISOT ----
    fake_col = list(fallback_clf.classes_).index(0)
    base_pred = fallback_clf.predict(X_sparse)
    base_prob = fallback_clf.predict_proba(X_sparse)[:, fake_col]
    base = M.evaluate(y, base_pred, base_prob)

    # ---- Approach 2 (route to nearest FineFake cluster, then specialist) ----
    cluster = kmeans.predict(X_dense)
    b_pred, b_prob = route_and_classify(X_sparse, cluster, cluster_models, fallback_clf, fallback_clusters)
    routed = M.evaluate(y, b_pred, b_prob)

    # ---- report ----
    def fmt(d):
        return (f"macro_f1={d['macro_f1']:.4f}  acc={d['accuracy']:.4f}  "
                f"prec(fake)={d['precision']:.4f}  rec(fake)={d['recall']:.4f}  "
                f"fpr={d['fpr']:.4f}  roc_auc={d['roc_auc']:.4f}  mcc={d['mcc']:.4f}")

    print("\n=== FineFake -> ISOT transfer (nothing retrained on ISOT) ===")
    print("baseline (global)          :", fmt(base))
    print("Approach 2 (cluster+route) :", fmt(routed))

    print("\nISOT cluster routing distribution (which FineFake clusters ISOT lands in):")
    print(pd.Series(cluster).value_counts().sort_index().to_string())

    # ---- persist a small transfer results file ----
    # n_train is the number of FineFake TRAINING ROWS these models saw.
    # (Previously passed fallback_clf.n_features_in_, which is the feature
    # count - 100,000 - not a row count.)
    n_train = len(splits.load_splits()["train"])
    rows = [
        M.result_row("D", "baseline_transfer_isot", "isot", SEED, n_train, len(isot), base),
        M.result_row("D", "clustered_routed_transfer_isot", "isot", SEED, n_train, len(isot), routed),
    ]
    out = Path(__file__).resolve().parents[1] / "results" / "isot_transfer.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
