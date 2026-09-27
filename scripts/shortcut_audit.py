"""
SHORTCUT AUDIT — "how much of the headline score is real?"

Stage 1  probe    : deliberately crippled classifiers that see ONE surface cue
                    (length, source/platform, caps ratio, punctuation,
                    stopword ratio) plus a combined "cheap cues" model.
Stage 2  attribute: share of the full model's above-chance gap that each cue
                    alone explains.
Stage 3  de-bias  : neutralise the dominant cues (exact-length truncation,
                    length-stratified rebalancing, source/boilerplate masking)
                    retrain the FULL text model and re-measure. The drop is
                    the cue's real contribution.

Run across FineFake, WELFake and ISOT. ISOT is the positive control: it is
known to be separable by the "(Reuters)" dateline, so a correct instrument
must flag it hardest.

    python scripts/shortcut_audit.py
"""
from __future__ import annotations

import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

from sklearn.dummy import DummyClassifier                      # noqa: E402
from sklearn.feature_extraction.text import (                  # noqa: E402
    ENGLISH_STOP_WORDS, TfidfVectorizer)
from sklearn.linear_model import LogisticRegression            # noqa: E402
from sklearn.metrics import f1_score                           # noqa: E402
from sklearn.model_selection import train_test_split           # noqa: E402
from sklearn.preprocessing import StandardScaler               # noqa: E402

from src import data as D                                      # noqa: E402

SEED = 42
OUT = REPO / "results"
OUT.mkdir(exist_ok=True)
WELFAKE_CSV = Path("/mnt/user-data/uploads/Downloads/WELFake_Dataset.csv")
ISOT_DIR = Path("/mnt/user-data/uploads/Downloads/News-_dataset (1)")

# ----------------------------------------------------------------- utilities

def mf1(y, p) -> float:
    return float(f1_score(y, p, average="macro"))


def fit_lr(X, y):
    return LogisticRegression(max_iter=2000, class_weight="balanced",
                              random_state=SEED).fit(X, y)


def full_text_model(tr, te) -> float:
    """The reference: sparse word TF-IDF + logistic regression."""
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                          max_features=100_000, sublinear_tf=True,
                          strip_accents="unicode")
    Xtr = vec.fit_transform(tr["text"]); Xte = vec.transform(te["text"])
    return mf1(te["label"], fit_lr(Xtr, tr["label"]).predict(Xte))


def floor_score(tr, te) -> float:
    """Best no-information predictor: max(majority, uniform-random)."""
    best = 0.0
    for strat in ("most_frequent", "uniform"):
        d = DummyClassifier(strategy=strat, random_state=SEED)
        d.fit(np.zeros((len(tr), 1)), tr["label"])
        best = max(best, mf1(te["label"], d.predict(np.zeros((len(te), 1)))))
    return best


# ------------------------------------------------------------- cue extractors

_STOP = frozenset(ENGLISH_STOP_WORDS)


def cue_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Surface cues computed from the raw string — no vocabulary, no content."""
    t = df["text"].astype(str)
    n_chars = t.str.len().clip(lower=1)
    toks = t.str.split()
    n_toks = toks.str.len().clip(lower=1)

    def stop_ratio(ws):
        if not ws:
            return 0.0
        return sum(w.lower() in _STOP for w in ws) / len(ws)

    return pd.DataFrame({
        "log_words":      np.log1p(df["n_words"].to_numpy()),
        "caps_ratio":     (t.str.count(r"[A-Z]") / n_chars).to_numpy(),
        "exclaim":        (t.str.count(r"!") / n_toks).to_numpy(),
        "question":       (t.str.count(r"\?") / n_toks).to_numpy(),
        "quote":          (t.str.count(r'["“”]') / n_toks).to_numpy(),
        "ellipsis":       (t.str.count(r"\.\.\.") / n_toks).to_numpy(),
        "stopword_ratio": toks.apply(stop_ratio).to_numpy(),
    }, index=df.index)


PROBE_GROUPS = {
    "length only (log word count)": ["log_words"],
    "caps ratio only":              ["caps_ratio"],
    "punctuation only":             ["exclaim", "question", "quote", "ellipsis"],
    "stopword ratio only":          ["stopword_ratio"],
    "ALL cheap cues combined":      ["log_words", "caps_ratio", "exclaim",
                                     "question", "quote", "ellipsis",
                                     "stopword_ratio"],
}


def run_probes(tr, te) -> dict[str, float]:
    Ctr, Cte = cue_frame(tr), cue_frame(te)
    out = {}
    for name, cols in PROBE_GROUPS.items():
        sc = StandardScaler().fit(Ctr[cols])
        out[name] = mf1(te["label"],
                        fit_lr(sc.transform(Ctr[cols]), tr["label"])
                        .predict(sc.transform(Cte[cols])))
    # platform identity, where the corpus has more than one platform
    if tr["platform"].nunique() > 1:
        oh_tr = pd.get_dummies(tr["platform"])
        oh_te = pd.get_dummies(te["platform"]).reindex(columns=oh_tr.columns,
                                                       fill_value=0)
        out["source/platform identity only"] = mf1(
            te["label"], fit_lr(oh_tr, tr["label"]).predict(oh_te))
    return out


# ------------------------------------------------------------- de-biasing ops

# Outlet names, datelines and boilerplate that identify the SOURCE rather than
# the truth of the claim. ISOT's real half is entirely Reuters wire copy.
_OUTLETS = [
    "reuters", "cnn", "associated press", "ap news", "apnews",
    "washington post", "washingtonpost", "new york times", "nytimes",
    "snopes", "reddit", "twitter", "cdc", "fox news", "breitbart",
    "bbc", "npr", "guardian", "huffington post", "politico", "bloomberg",
    "getty images", "featured image", "21st century wire", "infowars",
]
_OUTLET_RE = re.compile("|".join(re.escape(o) for o in _OUTLETS), re.I)
# "WASHINGTON (Reuters) - " style dateline that opens every ISOT real article
_DATELINE_RE = re.compile(r"^\s*[A-Z][A-Za-z .,/'-]{0,40}\s*\([^)]{0,40}\)\s*[-–—]\s*")
_URL_RE = re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+")
_HANDLE_RE = re.compile(r"@\w+")


def mask_sources(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(_DATELINE_RE, "", regex=True)
    s = s.str.replace(_URL_RE, " ", regex=True)
    s = s.str.replace(_HANDLE_RE, " ", regex=True)
    s = s.str.replace(_OUTLET_RE, " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def apply_source_mask(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text"] = mask_sources(out["text"])
    out = out[out["text"].str.strip() != ""]
    out["n_words"] = out["text"].str.split().str.len().astype(int)
    return out


def exact_length(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Every surviving document becomes EXACTLY n words: length carries 0 bits."""
    out = df[df["n_words"] >= n].copy()
    out["text"] = out["text"].str.split().str[:n].str.join(" ")
    out["n_words"] = n
    return out


def length_balance(df: pd.DataFrame, n_buckets: int = 10, seed: int = SEED) -> pd.DataFrame:
    """Within each length decile, downsample so the labels are 50/50.

    Length then predicts nothing, but no text is truncated and short
    documents are not thrown away wholesale.
    """
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["_b"] = pd.qcut(d["n_words"].rank(method="first"), n_buckets,
                      labels=False, duplicates="drop")
    keep = []
    for _, g in d.groupby("_b"):
        counts = g["label"].value_counts()
        if len(counts) < 2:
            continue
        m = int(counts.min())
        for lab in counts.index:
            idx = g.index[g["label"] == lab].to_numpy()
            keep.extend(rng.choice(idx, size=m, replace=False))
    return d.loc[sorted(keep)].drop(columns="_b")


# ------------------------------------------------------------------- datasets

def strat_split(df, seed=SEED):
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df["label"],
                               random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["label"],
                              random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def load_welfake() -> pd.DataFrame:
    df = pd.read_csv(WELFAKE_CSV)
    df["title"] = df["title"].fillna(""); df["text"] = df["text"].fillna("")
    df["text"] = (df["title"] + ". " + df["text"]).str.strip()
    df = df[df["text"].str.len() > 0].dropna(subset=["label"])
    df = df.drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(int)
    df["topic"] = "unknown"; df["platform"] = "welfake"
    df["n_words"] = df["text"].str.split().str.len().astype(int)
    return df.reset_index(drop=True)[D.SCHEMA]


def load_all() -> dict[str, pd.DataFrame]:
    sets = {}
    sets["FineFake"] = D.load_finefake()
    sets["WELFake"] = load_welfake()
    isot = D.load_isot(ISOT_DIR)
    isot = isot[~isot["text"].str.strip().str.lower().duplicated(keep="first")]
    sets["ISOT"] = isot.reset_index(drop=True)
    return sets


# ----------------------------------------------------------------------- main

def main() -> None:
    sets = load_all()
    probe_rows, debias_rows = [], []

    for name, df in sets.items():
        t0 = time.time()
        tr, te = strat_split(df)
        full = full_text_model(tr, te)
        flo = floor_score(tr, te)
        print(f"\n===== {name} =====  n={len(df):,}  "
              f"train={len(tr):,} test={len(te):,}", flush=True)
        print(f"  full text model macro-F1 = {full:.4f}   floor = {flo:.4f}", flush=True)

        for probe, score in run_probes(tr, te).items():
            share = (score - flo) / (full - flo) if full > flo else np.nan
            probe_rows.append({"dataset": name, "probe": probe,
                               "macro_f1": round(score, 4),
                               "share_of_gap": round(float(share), 4)})
            print(f"    {probe:34s} {score:.4f}   "
                  f"explains {share*100:5.1f}% of the gap", flush=True)

        probe_rows.append({"dataset": name, "probe": "FULL text model (reference)",
                           "macro_f1": round(full, 4), "share_of_gap": 1.0})
        probe_rows.append({"dataset": name, "probe": "no-information floor",
                           "macro_f1": round(flo, 4), "share_of_gap": 0.0})

        # ---- Stage 3: de-biasing, retrain the FULL text model each time ----
        n_trunc = int(max(20, np.percentile(df["n_words"], 40)))
        variants = {
            "raw (no intervention)": df,
            f"length-controlled (exactly {n_trunc} words)": exact_length(df, n_trunc),
            "length-stratified rebalance": length_balance(df),
            "source/boilerplate masked": apply_source_mask(df),
            "masked + length-stratified": length_balance(apply_source_mask(df)),
        }
        for vname, vdf in variants.items():
            if len(vdf) < 500 or vdf["label"].nunique() < 2:
                continue
            vtr, vte = strat_split(vdf)
            score = full_text_model(vtr, vte)
            lp = run_probes(vtr, vte)["length only (log word count)"]
            debias_rows.append({
                "dataset": name, "variant": vname, "n_rows": len(vdf),
                "full_macro_f1": round(score, 4),
                "drop_vs_raw": round(score - full, 4),
                "length_probe_f1": round(lp, 4)})
            print(f"    [debias] {vname:42s} n={len(vdf):6,}  "
                  f"F1={score:.4f}  ({score-full:+.4f})  lenprobe={lp:.4f}",
                  flush=True)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    pr = pd.DataFrame(probe_rows); dr = pd.DataFrame(debias_rows)
    pr.to_csv(OUT / "shortcut_probes.csv", index=False)
    dr.to_csv(OUT / "shortcut_debias.csv", index=False)
    print("\n\n================ PROBE / ATTRIBUTION ================")
    print(pr.to_string(index=False))
    print("\n================ DE-BIASING ================")
    print(dr.to_string(index=False))
    print(f"\nwrote {OUT/'shortcut_probes.csv'} and {OUT/'shortcut_debias.csv'}")


if __name__ == "__main__":
    main()
