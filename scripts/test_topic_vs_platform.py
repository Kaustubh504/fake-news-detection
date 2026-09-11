"""
Is the specialisation gain about TOPIC, or about PLATFORM?

    python scripts/test_topic_vs_platform.py

On the full corpus, training a classifier on its own topic beats training on
a random sample of the same size by about +0.033 macro-F1. But topic and
platform are tangled: Health skews to cdc_gov (1.1% fake), Entertainment to
snope and reddit (67-76% fake). So a "topic specialist" may really be a
platform specialist that has learned each outlet's base rate.

This script holds platform CONSTANT and re-runs the same matched-size test
inside each one. Within a single platform there is no outlet to learn and
far less length variation, so any remaining gain is genuinely about topic.

    specialist : trained on (platform, topic) rows
    matched    : trained on the same NUMBER of rows drawn at random from
                 that platform, all topics

    gain = specialist - matched

Reads the frozen split so results stay comparable with Approach A.
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

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from src import data, features as F, splits  # noqa: E402

pd.set_option("display.width", 200)

MIN_TRAIN = 150      # per (platform, topic) cell
MIN_TEST = 30
N_REPEATS = 5


def fit_f1(X_tr, y_tr, X_te, y_te) -> float:
    if len(np.unique(y_tr)) < 2:
        return np.nan
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_tr, y_tr)
    return f1_score(y_te, clf.predict(X_te), average="macro")


def main() -> None:
    df = data.load_finefake()
    sp = splits.load_splits()
    splits.verify_splits(df, sp)
    tr, _, te = splits.apply_splits(df, sp)

    vec = F.fit_tfidf(tr["text"], use_char=False)
    X_tr_all = F.transform(vec, tr["text"])
    y_tr_all = tr["label"].to_numpy()

    print("Platform composition (train split):")
    comp = pd.DataFrame({
        "n": tr["platform"].value_counts(),
        "pct_fake": (tr.groupby("platform")["label"]
                     .apply(lambda s: 100 * (s == 0).mean())).round(1),
        "median_words": tr.groupby("platform")["n_words"].median().round(0),
    }).sort_values("n", ascending=False)
    print(comp.to_string())
    print()

    rows = []
    for plat in comp.index:
        p_tr = tr[tr.platform == plat]
        p_te = te[te.platform == plat]
        if len(p_tr) < 2 * MIN_TRAIN or p_tr["label"].nunique() < 2:
            continue

        # positional indices of this platform's rows inside the train matrix
        p_pos = np.flatnonzero((tr["platform"] == plat).to_numpy())

        for topic in sorted(p_tr["topic"].unique()):
            a = p_tr[p_tr.topic == topic]
            b = p_te[p_te.topic == topic]
            if len(a) < MIN_TRAIN or len(b) < MIN_TEST or a["label"].nunique() < 2:
                continue

            X_te_t = F.transform(vec, b["text"])
            y_te_t = b["label"].to_numpy()

            spec = fit_f1(F.transform(vec, a["text"]), a["label"].to_numpy(),
                          X_te_t, y_te_t)

            ctrl = []
            for r in range(N_REPEATS):
                rng = np.random.default_rng(42 + r)
                idx = rng.choice(p_pos, len(a), replace=False)
                ctrl.append(fit_f1(X_tr_all[idx], y_tr_all[idx], X_te_t, y_te_t))
            ctrl = float(np.nanmean(ctrl))

            rows.append({
                "platform": plat, "topic": topic,
                "n_train": len(a), "n_test": len(b),
                "pct_fake": round(100 * (a["label"] == 0).mean(), 1),
                "specialist": round(spec, 4),
                "matched": round(ctrl, 4),
                "gain": round(spec - ctrl, 4),
            })

    if not rows:
        print("No (platform, topic) cell met the size thresholds.")
        return

    res = pd.DataFrame(rows)
    print("WITHIN-PLATFORM topic specialisation")
    print("(platform held constant, so any gain is about topic alone)\n")
    print(res.to_string(index=False))

    print("\nby platform:")
    by_p = res.groupby("platform").agg(
        cells=("gain", "size"), mean_gain=("gain", "mean"),
        wins=("gain", lambda s: int((s > 0).sum())))
    print(by_p.round(4).to_string())

    overall = res["gain"].mean()
    wins = int((res["gain"] > 0).sum())
    print(f"\nOVERALL mean within-platform topic gain = {overall:+.4f}")
    print(f"positive in {wins}/{len(res)} cells")
    print("\ncompare: ACROSS platforms (Approach A) the same test gave +0.0329")
    print()
    if overall > 0.01 and wins > len(res) / 2:
        print("READ: topic specialisation survives with platform held constant.")
        print("      The effect is genuinely about topic. Premise confirmed.")
    elif overall <= 0.005:
        print("READ: the gain largely DISAPPEARS once platform is held constant.")
        print("      What looked like topic specialisation was mostly the model")
        print("      learning each outlet's base rate. Report this - it is a")
        print("      finding about the corpus, not a failure of the method.")
    else:
        print("READ: partial. Some topic effect remains but it is much smaller")
        print("      than the across-platform number suggested.")

    out = REPO / "results" / "topic_vs_platform.csv"
    res.to_csv(out, index=False)
    print(f"\nwritten: {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
