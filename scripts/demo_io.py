"""
Worked input/output example for the stylistic feature-union model (Appendix C).

Trains the final architecture on the de-duplicated WELFake training split, then
scores a handful of held-out test articles and prints the full input/output record
for each: snippet, true label, P(fake), verdict, and the surface statistics that
the stylistic blocks respond to.

    python scripts/demo_io.py
"""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED, FAKE, REAL, CHAR_CAP = 42, 0, 1, 1200
WELFAKE_CSV = Path("/mnt/user-data/uploads/Downloads/WELFake_Dataset.csv")
_STOP = frozenset(ENGLISH_STOP_WORDS)


def surface_stats(t: pd.Series) -> np.ndarray:
    t = t.astype(str)
    n_ch = t.str.len().clip(lower=1)
    toks = t.str.split()
    n_tk = toks.str.len().clip(lower=1)
    sr = toks.apply(lambda ws: sum(w.lower() in _STOP for w in ws) / len(ws) if ws else 0.0)
    return np.column_stack([
        np.log1p(n_tk), np.log1p(n_ch),
        t.str.count(r"[A-Z]") / n_ch, t.str.count(r"[A-Z]{2,}") / n_tk,
        t.str.count(r"!") / n_tk, t.str.count(r"\?") / n_tk,
        t.str.count(r'["“”]') / n_tk, t.str.count(r"\.\.\.") / n_tk,
        t.str.count(r"[0-9]") / n_ch, t.str.count(r"\s") / n_ch,
        t.str.count(r"[,;:]") / n_tk, sr]).astype(np.float64)


def main():
    df = pd.read_csv(WELFAKE_CSV)
    df["title"] = df["title"].fillna(""); df["text"] = df["text"].fillna("")
    df["doc"] = (df["title"] + ". " + df["text"]).str.strip()
    df = df[df["doc"].str.len() > 0].dropna(subset=["label"]).drop_duplicates(subset=["doc"])
    df["label"] = 1 - df["label"].astype(int)
    df = df.reset_index(drop=True)[["doc", "label"]]

    tr, tmp = train_test_split(df, test_size=.30, stratify=df["label"], random_state=SEED)
    va, te = train_test_split(tmp, test_size=.50, stratify=tmp["label"], random_state=SEED)
    tr, va, te = (x.reset_index(drop=True) for x in (tr, va, te))
    y_tr, y_va, y_te = (x["label"].to_numpy() for x in (tr, va, te))

    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                           max_features=100_000, sublinear_tf=True,
                           strip_accents="unicode", dtype=np.float32)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=5,
                           max_features=80_000, sublinear_tf=True,
                           lowercase=False, dtype=np.float32)
    sc = StandardScaler()

    Wtr = word.fit_transform(tr["doc"]); Ctr = char.fit_transform(tr["doc"].str.slice(0, CHAR_CAP))
    Str = sc.fit_transform(surface_stats(tr["doc"]))
    Xtr = sparse.hstack([Wtr, Ctr, sparse.csr_matrix(Str, dtype=np.float32)],
                        format="csr", dtype=np.float32)

    def featurise(s: pd.Series):
        return sparse.hstack([word.transform(s), char.transform(s.str.slice(0, CHAR_CAP)),
                              sparse.csr_matrix(sc.transform(surface_stats(s)), dtype=np.float32)],
                             format="csr", dtype=np.float32)

    clf = LogisticRegression(max_iter=1500, C=4.0, class_weight="balanced",
                             random_state=SEED).fit(Xtr, y_tr)
    col = list(clf.classes_).index(FAKE)

    pv = clf.predict_proba(featurise(va["doc"]))[:, col]
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        f = f1_score(y_va, np.where(pv >= t, FAKE, REAL), average="macro")
        if f > best_f:
            best_f, best_t = f, t
    print(f"tuned threshold tau = {best_t:.3f} (validation macro-F1 {best_f:.4f})\n")

    # pick 2 true-fake and 2 true-real test articles, deterministically
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(y_te == FAKE)[0], 2, replace=False),
                          rng.choice(np.where(y_te == REAL)[0], 2, replace=False)])
    sub = te.iloc[idx].reset_index(drop=True)
    p = clf.predict_proba(featurise(sub["doc"]))[:, col]
    S = surface_stats(sub["doc"])
    names = ["log_words", "log_chars", "caps_ratio", "allcaps_runs", "excl", "quest",
             "quotes", "ellipsis", "digits", "whitespace", "commas", "stopword_ratio"]

    out = []
    for i, row in sub.iterrows():
        rec = {
            "input_snippet": row["doc"][:220].replace("\n", " ") + " …",
            "true_label": "FAKE" if row["label"] == FAKE else "REAL",
            "p_fake": round(float(p[i]), 4),
            "threshold": round(float(best_t), 3),
            "verdict": "FAKE" if p[i] >= best_t else "REAL",
            "correct": bool((p[i] >= best_t) == (row["label"] == FAKE)),
            "surface_stats": {n: round(float(v), 4) for n, v in zip(names, S[i])},
        }
        out.append(rec)
        print(json.dumps(rec, indent=2)[:1400]); print("-" * 70)

    (REPO / "results").mkdir(exist_ok=True)
    with open(REPO / "results" / "demo_io.json", "w") as fh:
        json.dump({"threshold": round(float(best_t), 3), "examples": out}, fh, indent=2)
    print("wrote results/demo_io.json")


if __name__ == "__main__":
    main()
