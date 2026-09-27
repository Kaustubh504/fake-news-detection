"""
STYLISTIC ARCHITECTURE — can we pass 98% on WELFake?

Motivation comes straight from our own shortcut audit: on WELFake a punctuation-only
classifier reaches macro-F1 0.739 and all content-free cues combined reach 0.804.
That signal is STYLISTIC - casing, punctuation, spacing, register. Word-level TF-IDF
deliberately destroys it (it lower-cases and strips punctuation), so our 97.65% linear
system was scoring while blindfolded to the corpus's strongest cue.

This script restores it with a three-block feature union:

  1. WORD TF-IDF     1-2 grams, 100k   - topical / lexical content
  2. CHAR_WB TF-IDF  3-5 grams, 150k   - casing, punctuation, affixes, boilerplate
  3. SURFACE STATS   12 scaled features - length, caps ratio, punctuation rates,
                                          stopword ratio, digit ratio, whitespace

on top of which we fit calibrated LinearSVC / LogReg / SGD and a stacked combiner,
tune the decision threshold on validation, and evaluate once on test.

Both evaluation protocols are run, because published numbers use the raw file:
  A  de-duplicated (ours, honest)   B  raw with duplicates (published convention)

    python scripts/stylistic_welfake.py
"""
from __future__ import annotations

import re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42
FAKE, REAL = 0, 1
WELFAKE_CSV = Path("/mnt/user-data/uploads/Downloads/WELFake_Dataset.csv")
OUT = REPO / "results"; OUT.mkdir(exist_ok=True)
CHAR_CAP = 1200          # chars fed to the char vectoriser, for tractability
_STOP = frozenset(ENGLISH_STOP_WORDS)


# ------------------------------------------------------------------ metrics
def tune_threshold(y, p):
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        f = f1_score(y, np.where(p >= t, FAKE, REAL), average="macro")
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


def full_metrics(y, p, t):
    pred = np.where(p >= t, FAKE, REAL)
    cm = confusion_matrix(y, pred, labels=[FAKE, REAL])
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    return {"macro_f1": f1_score(y, pred, average="macro"),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, pos_label=FAKE, zero_division=0),
            "recall": recall_score(y, pred, pos_label=FAKE, zero_division=0),
            "roc_auc": roc_auc_score((y == FAKE).astype(int), p),
            "mcc": matthews_corrcoef(y, pred),
            "fake_FNR": fn / max(fn + tp, 1), "fake_FPR": fp / max(fp + tn, 1)}


# ------------------------------------------------------------------ features
def surface_stats(texts: pd.Series) -> np.ndarray:
    """The cues the audit showed carry most of WELFake's signal, made explicit."""
    t = texts.astype(str)
    n_ch = t.str.len().clip(lower=1)
    toks = t.str.split()
    n_tk = toks.str.len().clip(lower=1)

    def stop_ratio(ws):
        return sum(w.lower() in _STOP for w in ws) / len(ws) if ws else 0.0

    return np.column_stack([
        np.log1p(n_tk), np.log1p(n_ch),
        t.str.count(r"[A-Z]") / n_ch,                  # caps ratio
        t.str.count(r"[A-Z]{2,}") / n_tk,              # ALL-CAPS runs
        t.str.count(r"!") / n_tk,
        t.str.count(r"\?") / n_tk,
        t.str.count(r'["“”]') / n_tk,
        t.str.count(r"\.\.\.") / n_tk,
        t.str.count(r"[0-9]") / n_ch,                  # digit ratio
        t.str.count(r"\s") / n_ch,                     # whitespace ratio
        t.str.count(r"[,;:]") / n_tk,
        toks.apply(stop_ratio),
    ]).astype(np.float64)


def build_features(tr, va, te):
    t0 = time.time()
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                           max_features=100_000, sublinear_tf=True,
                           strip_accents="unicode", dtype=np.float32)
    Wtr = word.fit_transform(tr); Wva = word.transform(va); Wte = word.transform(te)
    print(f"    word block {Wtr.shape}  ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    # char_wb keeps casing and punctuation - exactly what the word block discards
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=5,
                           max_features=80_000, sublinear_tf=True, lowercase=False, dtype=np.float32)
    Ctr = char.fit_transform(tr.str.slice(0, CHAR_CAP))
    Cva = char.transform(va.str.slice(0, CHAR_CAP))
    Cte = char.transform(te.str.slice(0, CHAR_CAP))
    print(f"    char block {Ctr.shape}  ({time.time()-t0:.0f}s)", flush=True)

    sc = StandardScaler()
    Str = sc.fit_transform(surface_stats(tr))
    Sva = sc.transform(surface_stats(va)); Ste = sc.transform(surface_stats(te))
    print(f"    surface block {Str.shape}", flush=True)

    f32 = lambda M: sparse.csr_matrix(M, dtype=np.float32)
    return (sparse.hstack([Wtr, Ctr, f32(Str)], format="csr", dtype=np.float32),
            sparse.hstack([Wva, Cva, f32(Sva)], format="csr", dtype=np.float32),
            sparse.hstack([Wte, Cte, f32(Ste)], format="csr", dtype=np.float32),
            (Wtr, Wva, Wte))


# ------------------------------------------------------------------ data
def load_welfake(deduplicate: bool) -> pd.DataFrame:
    df = pd.read_csv(WELFAKE_CSV)
    df["title"] = df["title"].fillna(""); df["text"] = df["text"].fillna("")
    df["doc"] = (df["title"] + ". " + df["text"]).str.strip()
    df = df[df["doc"].str.len() > 0].dropna(subset=["label"])
    if deduplicate:
        df = df.drop_duplicates(subset=["doc"])
    df["label"] = 1 - df["label"].astype(int)     # ship 0=real -> repo fake=0/real=1
    return df.reset_index(drop=True)[["doc", "label"]]


class PlattSVC:
    """LinearSVC + Platt scaling fitted on the VALIDATION split.

    CalibratedClassifierCV(cv=3) refits the model three times and tripled peak
    memory on a 250k-feature matrix (it was OOM-killed). Since we already hold out
    a validation split, one fit plus a 1-D logistic on its decision scores gives the
    same calibrated probabilities far more cheaply.
    """

    def __init__(self, seed=SEED):
        self.svc = LinearSVC(class_weight="balanced", random_state=seed)
        self.platt = LogisticRegression(max_iter=1000)
        self.classes_ = np.array([FAKE, REAL])

    def fit(self, X, y):
        self.svc.fit(X, y)
        return self

    def calibrate(self, Xva, yva):
        d = self.svc.decision_function(Xva).reshape(-1, 1)
        self.platt.fit(d, (yva == FAKE).astype(int))
        return self

    def proba_fake(self, X):
        d = self.svc.decision_function(X).reshape(-1, 1)
        return self.platt.predict_proba(d)[:, 1]


def models(seed=SEED):
    # liblinear (LinearSVC) copies the matrix to float64 and was OOM-killed on the
    # 220k-feature union, so the union uses memory-light solvers only.
    return {
        "SGD (modified Huber)": SGDClassifier(loss="modified_huber", alpha=1e-6,
                                              class_weight="balanced", max_iter=3000,
                                              tol=1e-4, random_state=seed),
        "SGD (log loss)": SGDClassifier(loss="log_loss", alpha=1e-6,
                                        class_weight="balanced", max_iter=3000,
                                        tol=1e-4, random_state=seed),
        "LogisticRegression (lbfgs)": LogisticRegression(max_iter=1500, C=4.0,
                                                 class_weight="balanced", random_state=seed),
    }


def p_fake(clf, X):
    if isinstance(clf, PlattSVC):
        return clf.proba_fake(X)
    return clf.predict_proba(X)[:, list(clf.classes_).index(FAKE)]


def main():
    rows = []
    for dedup in (True, False):        # raw first: it is the one to compare with
        tag = "de-duplicated (ours)" if dedup else "RAW (published protocol)"
        df = load_welfake(dedup)
        tr, tmp = train_test_split(df, test_size=.30, stratify=df["label"], random_state=SEED)
        va, te = train_test_split(tmp, test_size=.50, stratify=tmp["label"], random_state=SEED)
        tr, va, te = (x.reset_index(drop=True) for x in (tr, va, te))
        y_tr, y_va, y_te = (x["label"].to_numpy() for x in (tr, va, te))
        print(f"\n===== WELFake — {tag} =====")
        print(f"  rows {len(df):,}  train {len(tr):,}  val {len(va):,}  test {len(te):,}", flush=True)

        Xtr, Xva, Xte, (Wtr, Wva, Wte) = build_features(tr["doc"], va["doc"], te["doc"])

        # reference: word-only block, our previous best architecture
        ref = PlattSVC().fit(Wtr, y_tr).calibrate(Wva, y_va)
        t, _ = tune_threshold(y_va, p_fake(ref, Wva))
        m = full_metrics(y_te, p_fake(ref, Wte), t)
        rows.append({"protocol": tag, "features": "word TF-IDF only (previous best)",
                     "model": "calibrated LinearSVC", **m})
        print(f"  [reference] word-only        acc {m['accuracy']:.4f}", flush=True)

        # full union
        P_va, P_te = {}, {}
        for name, clf in models().items():
            t0 = time.time()
            clf.fit(Xtr, y_tr)
            if isinstance(clf, PlattSVC):
                clf.calibrate(Xva, y_va)
            pv, pt = p_fake(clf, Xva), p_fake(clf, Xte)
            P_va[name], P_te[name] = pv, pt
            t, _ = tune_threshold(y_va, pv)
            m = full_metrics(y_te, pt, t)
            rows.append({"protocol": tag, "features": "word + char_wb + surface",
                         "model": name, **m})
            print(f"  {name:22s} acc {m['accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

        # stacked combiner over the three model probabilities
        Mva = np.column_stack(list(P_va.values())); Mte = np.column_stack(list(P_te.values()))
        stack = LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=SEED).fit(Mva, y_va)
        sv, st = p_fake(stack, Mva), p_fake(stack, Mte)
        t, _ = tune_threshold(y_va, sv)
        m = full_metrics(y_te, st, t)
        rows.append({"protocol": tag, "features": "word + char_wb + surface",
                     "model": "STACKED (3 models)", **m})
        print(f"  {'STACKED':22s} acc {m['accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}", flush=True)

    res = pd.DataFrame(rows).round(4)
    print("\n================ RESULTS ================")
    print(res.to_string(index=False))
    res.to_csv(OUT / "stylistic_welfake.csv", index=False)
    print(f"\nwrote {OUT/'stylistic_welfake.csv'}")


if __name__ == "__main__":
    main()
