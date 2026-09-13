"""
Explainability metrics for the Approach B retrieval layer.

    python scripts/explainability_metrics.py

Computes the five numbers the Phase-B report quotes, which the Approach B
notebook only partly produces:

  cluster purity              size-weighted majority-topic share of each cluster
  NMI / homogeneity           agreement between discovered clusters and true
                              topics. REPORTED ONLY - never used to choose k
  neighbour topic match       share of the 5 retrieved neighbours that share
                              the query's true topic, computed both
                              cluster-scoped and globally. The gap is what
                              cluster-scoping actually buys
  neighbour label agreement   share of retrieved neighbours sharing the
                              query's fake/real label - what a user sees as
                              "4 of 5 similar articles are also fake"
  mean retrieved similarity   mean cosine similarity of the 5 neighbours

Retrieval uses the dense SVD vector, the same representation that routes the
article, so the evidence shown is similar in exactly the sense that decided
the routing.

Requires the Approach B artifacts (run notebooks/02_approach_b.ipynb first).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

from sklearn.metrics import (  # noqa: E402
    homogeneity_score,
    normalized_mutual_info_score,
)

from src import data, features as F, splits  # noqa: E402

TOP_K = 5


def purity(clusters: np.ndarray, topics: np.ndarray) -> float:
    tbl = pd.crosstab(pd.Series(clusters), pd.Series(topics))
    return float(tbl.max(axis=1).sum() / tbl.to_numpy().sum())


def main() -> None:
    df = data.load_finefake()
    sp = splits.load_splits()
    splits.verify_splits(df, sp)
    tr, _, te = splits.apply_splits(df, sp)

    vec = F.load_pipeline("approach_b_tfidf")
    svd = F.load_pipeline("approach_b_svd")
    kmeans = F.load_pipeline("approach_b_kmeans")

    # Dense vectors are L2-normalised by the SVD pipeline, so a dot product
    # IS the cosine similarity.
    D_tr = svd.transform(F.transform(vec, tr["text"]))
    D_te = svd.transform(F.transform(vec, te["text"]))

    c_tr = kmeans.predict(D_tr)
    c_te = kmeans.predict(D_te)

    topic_tr = tr["topic"].to_numpy()
    topic_te = te["topic"].to_numpy()
    y_tr = tr["label"].to_numpy()
    y_te = te["label"].to_numpy()

    print(f"k = {kmeans.n_clusters}   train {len(tr):,}   test {len(te):,}\n")

    print("--- cluster quality vs true topics (reported only) ---")
    pur = purity(c_tr, topic_tr)
    nmi = normalized_mutual_info_score(topic_tr, c_tr)
    hom = homogeneity_score(topic_tr, c_tr)
    print(f"cluster purity (train)      : {pur:.4f}")
    print(f"NMI (clusters vs topics)    : {nmi:.4f}")
    print(f"homogeneity                 : {hom:.4f}")

    # ---- retrieval, cluster-scoped vs global ----
    sim_all = D_te @ D_tr.T          # (n_test, n_train) cosine
    rows = []

    for scope in ("cluster", "global"):
        topic_hits, label_hits, sims = [], [], []
        for i in range(len(te)):
            s = sim_all[i]
            if scope == "cluster":
                allowed = np.flatnonzero(c_tr == c_te[i])
                if len(allowed) < TOP_K:
                    continue
                order = allowed[np.argsort(s[allowed])[::-1][:TOP_K]]
            else:
                order = np.argsort(s)[::-1][:TOP_K]
            topic_hits.append((topic_tr[order] == topic_te[i]).mean())
            label_hits.append((y_tr[order] == y_te[i]).mean())
            sims.append(s[order].mean())
        rows.append({
            "scope": scope,
            "n_queries": len(topic_hits),
            "neighbour_topic_match": round(float(np.mean(topic_hits)), 4),
            "neighbour_label_agreement": round(float(np.mean(label_hits)), 4),
            "mean_similarity": round(float(np.mean(sims)), 4),
        })

    res = pd.DataFrame(rows)
    print("\n--- retrieval (top-5 neighbours) ---")
    print(res.to_string(index=False))

    cs = res[res.scope == "cluster"].iloc[0]
    gl = res[res.scope == "global"].iloc[0]
    print(f"\ntopic match, cluster-scoped vs global: "
          f"{cs.neighbour_topic_match:.4f} / {gl.neighbour_topic_match:.4f}"
          f"   (delta {cs.neighbour_topic_match - gl.neighbour_topic_match:+.4f})")
    print("A delta near zero means cluster-scoping adds little topic focus over")
    print("simply retrieving the nearest neighbours from the whole corpus.")

    # ---- per-cluster majority topic, for the report's table ----
    tbl = pd.crosstab(pd.Series(c_tr, name="cluster"), pd.Series(topic_tr))
    majority = tbl.idxmax(axis=1)
    sizes = pd.Series(c_tr).value_counts().sort_index()
    print("\n--- per-cluster majority topic ---")
    print(pd.DataFrame({"n_train": sizes, "majority_topic": majority}).to_string())

    out = REPO / "results" / "explainability_metrics.csv"
    summary = res.assign(cluster_purity=pur, nmi=round(nmi, 4),
                         homogeneity=round(hom, 4), k=kmeans.n_clusters)
    summary.to_csv(out, index=False)
    print(f"\nwritten: {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
