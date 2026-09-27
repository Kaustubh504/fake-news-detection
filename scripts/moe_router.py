"""
APPROACH 3 — soft-gated mixture of heterogeneous experts with a global fallback.

Approach 2 routed each article to exactly ONE logistic-regression specialist chosen
by nearest centroid. That design pays the full starvation cost: a specialist sees
only its cluster's rows, and a mis-routed article is scored by a model that never
saw anything like it. This script fixes all three weaknesses at once:

  1. HETEROGENEOUS EXPERTS  each cluster picks its own classifier family
     (LogReg / ComplementNB / SGD-modified-huber / calibrated LinearSVC) by
     validation macro-F1 on that cluster, instead of logistic regression everywhere.
  2. SOFT GATING            membership is a softmax over negative squared distance
     to every centroid, temperature T tuned on validation - so an article near a
     boundary is scored by BOTH neighbours in proportion, not by a coin flip.
  3. GLOBAL FALLBACK BLEND  the final probability always mixes in the global model,
     p = a*p_global + (1-a)*sum_c w_c*p_c, with a tuned on validation. This is the
     direct antidote to starvation: thin or unreliable clusters simply get pulled
     back towards the model trained on everything.

Every system - the baseline included - also gets its decision threshold tuned on
validation for macro-F1, so no system wins merely by being tuned when others are not.
All tuning uses the validation split only; the test split is touched once.

    python scripts/moe_router.py --dataset finefake
    python scripts/moe_router.py --dataset welfake
"""
from __future__ import annotations

import argparse, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from src import data as D

SEED = 42
OUT = REPO / "results"; OUT.mkdir(exist_ok=True)
WELFAKE_CSV = Path("/mnt/user-data/uploads/Downloads/WELFake_Dataset.csv")
FAKE, REAL = 0, 1                      # repo convention


# --------------------------------------------------------------- expert zoo
def expert_zoo(seed=SEED):
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=seed),
        "cnb":    ComplementNB(),
        "sgd":    SGDClassifier(loss="modified_huber", class_weight="balanced",
                                max_iter=2000, random_state=seed),
        "svc":    CalibratedClassifierCV(
                      LinearSVC(class_weight="balanced", random_state=seed),
                      cv=3, method="sigmoid"),
    }


def fit_expert(name, X, y, seed=SEED):
    clf = expert_zoo(seed)[name]
    clf.fit(X, y)
    return clf


def p_fake(clf, X):
    """P(label == FAKE) from any fitted expert."""
    cols = list(clf.classes_)
    return clf.predict_proba(X)[:, cols.index(FAKE)]


# --------------------------------------------------------------- evaluation
def tune_threshold(y, p):
    """Threshold maximising macro-F1 on the validation split."""
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, np.where(p >= t, FAKE, REAL), average="macro")
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


def full_metrics(y, p, t):
    pred = np.where(p >= t, FAKE, REAL)
    cm = confusion_matrix(y, pred, labels=[FAKE, REAL])
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    return {
        "macro_f1":  f1_score(y, pred, average="macro"),
        "accuracy":  accuracy_score(y, pred),
        "precision": precision_score(y, pred, pos_label=FAKE, zero_division=0),
        "recall":    recall_score(y, pred, pos_label=FAKE, zero_division=0),
        "roc_auc":   roc_auc_score((y == FAKE).astype(int), p),
        "mcc":       matthews_corrcoef(y, pred),
        "fake_FNR":  fn / max(fn + tp, 1),
        "fake_FPR":  fp / max(fp + tn, 1),
    }


# --------------------------------------------------------------- the router
def soft_weights(Dx, centroids, T):
    """Softmax over negative squared distance to each centroid."""
    d2 = ((Dx[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    z = -d2 / max(T, 1e-6)
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def build_experts(X_tr, y_tr, c_tr, X_va, y_va, c_va, k, global_clf, min_rows=150):
    """One expert per cluster; family chosen by validation macro-F1 on that cluster."""
    experts, chosen = {}, {}
    for c in range(k):
        m_tr, m_va = c_tr == c, c_va == c
        if m_tr.sum() < min_rows or len(np.unique(y_tr[m_tr])) < 2 or m_va.sum() < 30:
            experts[c], chosen[c] = global_clf, "global(fallback)"
            continue
        best, best_f = None, -1.0
        for name in expert_zoo():
            try:
                clf = fit_expert(name, X_tr[m_tr], y_tr[m_tr])
                f = f1_score(y_va[m_va],
                             np.where(p_fake(clf, X_va[m_va]) >= .5, FAKE, REAL),
                             average="macro")
            except Exception:
                continue
            if f > best_f:
                best_f, best, bname = f, clf, name
        if best is None:
            experts[c], chosen[c] = global_clf, "global(fallback)"
        else:
            experts[c], chosen[c] = best, bname
    return experts, chosen


def expert_matrix(experts, X, k):
    """P_fake from every expert for every row -> (n, k)."""
    return np.column_stack([p_fake(experts[c], X) for c in range(k)])


# --------------------------------------------------------------- data loading
def load_welfake():
    df = pd.read_csv(WELFAKE_CSV)
    df["title"] = df["title"].fillna(""); df["text"] = df["text"].fillna("")
    df["text"] = (df["title"] + ". " + df["text"]).str.strip()
    df = df[df["text"].str.len() > 0].dropna(subset=["label"])
    df = df.drop_duplicates(subset=["text"])
    # WELFake ships 0=real, 1=fake -> flip onto the repo convention (fake=0, real=1)
    df["label"] = 1 - df["label"].astype(int)
    df["topic"] = "unknown"; df["platform"] = "welfake"
    df["n_words"] = df["text"].str.split().str.len().astype(int)
    return df.reset_index(drop=True)[D.SCHEMA]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="finefake", choices=["finefake", "welfake"])
    args = ap.parse_args()

    df = D.load_finefake() if args.dataset == "finefake" else load_welfake()
    tr, tmp = train_test_split(df, test_size=.30, stratify=df["label"], random_state=SEED)
    va, te = train_test_split(tmp, test_size=.50, stratify=tmp["label"], random_state=SEED)
    tr, va, te = (x.reset_index(drop=True) for x in (tr, va, te))
    y_tr, y_va, y_te = (x["label"].to_numpy() for x in (tr, va, te))
    print(f"[{args.dataset}] train {len(tr):,}  val {len(va):,}  test {len(te):,}", flush=True)

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                          max_features=100_000, sublinear_tf=True,
                          strip_accents="unicode")
    X_tr = vec.fit_transform(tr["text"]); X_va = vec.transform(va["text"])
    X_te = vec.transform(te["text"])

    svd = make_pipeline(TruncatedSVD(256, random_state=SEED), Normalizer(copy=False))
    Dtr = svd.fit_transform(X_tr); Dva = svd.transform(X_va); Dte = svd.transform(X_te)
    print("features ready", flush=True)

    # ---- S0: global baseline, best family chosen on validation ----
    best_g, best_gf, gname = None, -1.0, None
    for name in expert_zoo():
        clf = fit_expert(name, X_tr, y_tr)
        f = f1_score(y_va, np.where(p_fake(clf, X_va) >= .5, FAKE, REAL), average="macro")
        print(f"   global candidate {name:7s} val {f:.4f}", flush=True)
        if f > best_gf:
            best_gf, best_g, gname = f, clf, name
    g_va, g_te = p_fake(best_g, X_va), p_fake(best_g, X_te)
    t_g, _ = tune_threshold(y_va, g_va)
    rows = [dict(system=f"S0 global baseline (best={gname})", **full_metrics(y_te, g_te, t_g))]
    print(f"   -> global = {gname}, threshold {t_g:.2f}", flush=True)

    # ---- grid over k, gate temperature and blend weight, all on validation ----
    best = None
    for k in (4, 6, 8, 10):
        km = KMeans(k, random_state=SEED, n_init=10).fit(Dtr)
        cen = km.cluster_centers_
        c_tr, c_va, c_te = km.labels_, km.predict(Dva), km.predict(Dte)
        experts, chosen = build_experts(X_tr, y_tr, c_tr, X_va, y_va, c_va, k, best_g)
        Eva, Ete_ = expert_matrix(experts, X_va, k), expert_matrix(experts, X_te, k)

        # hard routing (Approach-2 style) for reference, with these experts
        hard_va = Eva[np.arange(len(Eva)), c_va]
        hard_te = Ete_[np.arange(len(Ete_)), c_te]

        for T in (0.02, 0.05, 0.1, 0.25, 0.5):
            Wva = soft_weights(Dva, cen, T); Wte = soft_weights(Dte, cen, T)
            soft_va = (Wva * Eva).sum(1); soft_te = (Wte * Ete_).sum(1)
            for a in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                pv = a * g_va + (1 - a) * soft_va
                t, f = tune_threshold(y_va, pv)
                if best is None or f > best["val_f1"]:
                    best = dict(val_f1=f, k=k, T=T, alpha=a, thr=t, chosen=chosen,
                                soft_te=soft_te, hard_te=hard_te, hard_va=hard_va,
                                c_te=c_te, experts=experts)
        print(f"   k={k:2d} swept   best val so far {best['val_f1']:.4f} "
              f"(k={best['k']}, T={best['T']}, alpha={best['alpha']})", flush=True)

    k, T, a, thr = best["k"], best["T"], best["alpha"], best["thr"]

    # ---- S1/S2: hard routing (single-family vs best-family experts) ----
    km = KMeans(k, random_state=SEED, n_init=10).fit(Dtr)
    c_tr, c_va, c_te = km.labels_, km.predict(Dva), km.predict(Dte)
    lr_experts = {}
    for c in range(k):
        m = c_tr == c
        lr_experts[c] = (fit_expert("logreg", X_tr[m], y_tr[m])
                         if m.sum() >= 150 and len(np.unique(y_tr[m])) > 1 else best_g)
    L_va = expert_matrix(lr_experts, X_va, k); L_te = expert_matrix(lr_experts, X_te, k)
    hv = L_va[np.arange(len(L_va)), c_va]; ht = L_te[np.arange(len(L_te)), c_te]
    t1, _ = tune_threshold(y_va, hv)
    rows.append(dict(system="S1 hard routing, LogReg experts (Approach 2)",
                     **full_metrics(y_te, ht, t1)))

    t2, _ = tune_threshold(y_va, best["hard_va"])
    rows.append(dict(system="S2 hard routing, best-family experts",
                     **full_metrics(y_te, best["hard_te"], t2)))

    # ---- S3: soft gating, no global blend ----
    experts = best["experts"]
    Eva2 = expert_matrix(experts, X_va, k); Ete2 = expert_matrix(experts, X_te, k)
    Wva = soft_weights(Dva, km.cluster_centers_, T); Wte = soft_weights(Dte, km.cluster_centers_, T)
    sv, st = (Wva * Eva2).sum(1), (Wte * Ete2).sum(1)
    t3, _ = tune_threshold(y_va, sv)
    rows.append(dict(system="S3 soft-gated MoE (no global blend)",
                     **full_metrics(y_te, st, t3)))

    # ---- S4: soft gating + global fallback blend  (the full system) ----
    pv = a * g_va + (1 - a) * sv; pt = a * g_te + (1 - a) * st
    t4, _ = tune_threshold(y_va, pv)
    rows.append(dict(system=f"S4 soft MoE + global blend (k={k}, T={T}, a={a})",
                     **full_metrics(y_te, pt, t4)))

    # ---- S5: stacked meta-learner over [global, experts, gate] ----
    Mva = np.column_stack([g_va, Eva2, Wva]); Mte = np.column_stack([g_te, Ete2, Wte])
    meta = LogisticRegression(max_iter=2000, class_weight="balanced",
                              random_state=SEED).fit(Mva, y_va)
    mv, mt = p_fake(meta, Mva), p_fake(meta, Mte)
    t5, _ = tune_threshold(y_va, mv)
    rows.append(dict(system="S5 stacked meta-learner", **full_metrics(y_te, mt, t5)))

    # ---- S6 CONTROL: plain global ensemble, no clustering, no routing ----
    # If this matches S4, the MoE's gain is ensembling rather than routing.
    gl = {n: fit_expert(n, X_tr, y_tr) for n in expert_zoo()}
    ev = np.column_stack([p_fake(c, X_va) for c in gl.values()]).mean(1)
    et = np.column_stack([p_fake(c, X_te) for c in gl.values()]).mean(1)
    t6, _ = tune_threshold(y_va, ev)
    rows.append(dict(system="S6 CONTROL global ensemble (no clustering)",
                     **full_metrics(y_te, et, t6)))

    res = pd.DataFrame(rows).round(4)
    print(f"\n================ {args.dataset.upper()} ================")
    print(f"selected: k={k}  gate T={T}  blend alpha={a}  threshold={thr:.2f}")
    print("experts per cluster:", best["chosen"])
    print(res.to_string(index=False))
    res.insert(0, "dataset", args.dataset)
    res.to_csv(OUT / f"moe_{args.dataset}.csv", index=False)
    pd.DataFrame([{"dataset": args.dataset, "k": k, "gate_T": T, "blend_alpha": a,
                   "threshold": round(thr, 3), "global_family": gname,
                   "experts": str(best["chosen"])}]).to_csv(
        OUT / f"moe_{args.dataset}_config.csv", index=False)
    print(f"\nwrote {OUT/f'moe_{args.dataset}.csv'}")


if __name__ == "__main__":
    main()
