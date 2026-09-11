"""
Approach A - Baseline, Ceiling, and the Matched-Size Control.

Establishes the reference points every later approach is measured against.

WHY THERE ARE THREE SYSTEMS, NOT TWO
------------------------------------
A naive ceiling - one classifier per topic, trained only on that topic's
rows - conflates two opposing effects:

    + specialisation gain   training on your own topic helps
    - starvation cost       each specialist sees ~1/7 of the data

On this corpus those two roughly cancel, so the naive ceiling scores the
same as the global baseline and looks like "specialising does not work".
That conclusion would be wrong. Measured at equal training size, topic
composition helps in every topic (mean +0.03 macro-F1).

So we report three systems and decompose the difference:

    baseline   one global classifier, ALL training rows
    ceiling    one classifier per topic, that topic's rows only
    matched    one classifier per topic-sized RANDOM sample of all rows
               (same volume as the specialist, different composition)

    specialisation effect = ceiling  - matched     <- the real quantity
    starvation cost       = matched  - baseline    <- the price of splitting
    net partition effect  = ceiling  - baseline    <- what a hard split buys

Done when: the specialisation effect is positive. If it is not, topic
composition carries no advantage on this data and we report that.

The multi-domain literature (MDFEND, M3FEND) avoids the starvation cost
entirely by SHARING parameters across domains rather than partitioning -
which is exactly what Approach C does, and why it matters here.

DESIGN NOTE - shared feature space
----------------------------------
The vectoriser is fitted ONCE on all training rows and shared by every
system, so the comparison isolates training data alone rather than
confounding it with vocabulary differences.

Owner: Kaustubh Das
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .. import features as F
from .. import metrics as M
from .. import splits as S
from ..data import LABEL_FAKE

APPROACH = "A"
MIN_TOPIC_TRAIN = 120   # below this a specialist is not meaningfully trainable


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, sp: dict, use_char: bool = False):
    """Fit the shared sparse vectoriser on TRAINING ROWS ONLY."""
    return F.fit_tfidf(df.loc[sp["train"], "text"], use_char=use_char)


def _fit_predict(X_tr, y_tr, X_te, seed: int):
    """Fit and predict. Returns (clf, y_pred, prob_of_fake).

    If the training slice holds only one class, returns that constant rather
    than raising, so one thin topic cannot abort a whole run.
    """
    y_tr = np.asarray(y_tr)
    if len(np.unique(y_tr)) < 2:
        only = int(y_tr[0])
        n = X_te.shape[0]
        return None, np.full(n, only), np.full(n, 1.0 if only == LABEL_FAKE else 0.0)

    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed
    ).fit(X_tr, y_tr)
    fake_col = list(clf.classes_).index(LABEL_FAKE)
    return clf, clf.predict(X_te), clf.predict_proba(X_te)[:, fake_col]


def _trainable_topics(tr: pd.DataFrame, te: pd.DataFrame) -> list[str]:
    """Topics with enough training rows and both classes present."""
    keep = []
    for t in sorted(tr["topic"].unique()):
        a, b = tr[tr.topic == t], te[te.topic == t]
        if len(b) and len(a) >= MIN_TOPIC_TRAIN and a["label"].nunique() > 1:
            keep.append(t)
    return keep


# --------------------------------------------------------------------------
# systems
# --------------------------------------------------------------------------

def run_baseline(df: pd.DataFrame, sp: dict, vec, seed: int = 42):
    """One classifier on ALL training rows. The floor."""
    tr, te = df.loc[sp["train"]], df.loc[sp["test"]]
    X_tr, X_te = F.transform(vec, tr["text"]), F.transform(vec, te["text"])
    clf, pred, prob = _fit_predict(X_tr, tr["label"], X_te, seed)

    rows = [M.result_row(APPROACH, "baseline", "overall", seed,
                         len(tr), len(te), M.evaluate(te["label"], pred, prob))]
    for topic in sorted(te["topic"].unique()):
        mask = (te["topic"] == topic).to_numpy()
        if mask.sum() == 0:
            continue
        rows.append(M.result_row(
            APPROACH, "baseline", str(topic), seed, len(tr), int(mask.sum()),
            M.evaluate(te["label"].to_numpy()[mask], pred[mask], prob[mask]),
        ))
    return rows, clf


def run_ceiling(df: pd.DataFrame, sp: dict, vec, seed: int = 42) -> list[dict]:
    """One classifier per topic, on that topic's rows only.

    Routed at test time by the TRUE topic label - the oracle that makes this
    an upper bound on hard partitioning rather than a deployable system.
    """
    tr, te = df.loc[sp["train"]], df.loc[sp["test"]]
    rows, pooled = [], []

    for topic in _trainable_topics(tr, te):
        a, b = tr[tr.topic == topic], te[te.topic == topic]
        _, pred, prob = _fit_predict(
            F.transform(vec, a["text"]), a["label"], F.transform(vec, b["text"]), seed)
        rows.append(M.result_row(
            APPROACH, "ceiling", str(topic), seed, len(a), len(b),
            M.evaluate(b["label"], pred, prob)))
        pooled.append((b["label"].to_numpy(), pred, prob))

    if pooled:
        y = np.concatenate([p[0] for p in pooled])
        rows.insert(0, M.result_row(
            APPROACH, "ceiling", "overall", seed, len(tr), len(y),
            M.evaluate(y, np.concatenate([p[1] for p in pooled]),
                       np.concatenate([p[2] for p in pooled]))))
    return rows


def run_matched(df: pd.DataFrame, sp: dict, vec, seed: int = 42,
                n_repeats: int = 5) -> list[dict]:
    """Control: same training VOLUME as each specialist, random composition.

    For a topic whose specialist trained on n rows, train instead on n rows
    drawn at random from the whole training set and evaluate on that topic's
    test rows. The gap between this and the specialist is the specialisation
    effect with data volume held constant - the number the naive ceiling
    hides.

    Repeated n_repeats times with different draws; each repeat is its own
    row so the variance stays visible rather than collapsing into a mean.

    Caveat: a random draw carries the global label balance rather than the
    topic's. class_weight="balanced" absorbs most of that, but it is not a
    perfect control for label composition.
    """
    tr, te = df.loc[sp["train"]], df.loc[sp["test"]]
    X_all, y_all = F.transform(vec, tr["text"]), tr["label"].to_numpy()
    rows = []

    for topic in _trainable_topics(tr, te):
        n = int((tr["topic"] == topic).sum())
        b = te[te.topic == topic]
        X_te = F.transform(vec, b["text"])
        for r in range(n_repeats):
            rng = np.random.default_rng(seed + r)
            idx = rng.choice(len(tr), n, replace=False)
            _, pred, prob = _fit_predict(X_all[idx], y_all[idx], X_te, seed + r)
            rows.append(M.result_row(
                APPROACH, "matched", str(topic), seed + r, n, len(b),
                M.evaluate(b["label"], pred, prob)))
    return rows


# --------------------------------------------------------------------------
# orchestrator + reporting
# --------------------------------------------------------------------------

def run(df: pd.DataFrame, sp: dict, seed: int = 42, use_char: bool = False,
        write: bool = True, splits_path=None, n_repeats: int = 5) -> pd.DataFrame:
    """Run all three systems and return the result rows as a frame.

    Verifies the splits before doing anything - a mismatch means the numbers
    would not be comparable with any other approach.

    splits_path
        Which splits file to verify against. Defaults to the main
        artifacts/splits.json; pass a variant path when running a filtered
        corpus, which has its own split.
    """
    S.verify_splits(df, sp, path=splits_path or S.SPLITS_PATH)
    vec = prepare_features(df, sp, use_char=use_char)

    base_rows, base_clf = run_baseline(df, sp, vec, seed)
    rows = (base_rows
            + run_ceiling(df, sp, vec, seed)
            + run_matched(df, sp, vec, seed, n_repeats))

    if write:
        M.append_results(rows, APPROACH)

    frame = pd.DataFrame(rows)
    frame.attrs["vectoriser"] = vec
    frame.attrs["baseline_clf"] = base_clf
    return frame


def decompose(frame: pd.DataFrame) -> pd.DataFrame:
    """Split the ceiling-vs-baseline difference into its two causes.

    Columns
    -------
    specialisation : ceiling - matched   what topic composition buys
    starvation     : matched - baseline  the price of training on less data
    net            : ceiling - baseline  what a hard partition actually gives

    `net` is what the naive two-system comparison reports, and it is simply
    the sum of the other two - which is why on its own it misleads.
    """
    per = frame[frame["scope"] != "overall"]
    piv = per.pivot_table(index="scope", columns="system",
                          values="macro_f1", aggfunc="mean")
    n_tr = per[per.system == "ceiling"].set_index("scope")["n_train"].astype(int)

    out = pd.DataFrame(index=piv.index)
    out["n_train"] = n_tr
    for col in ("baseline", "matched", "ceiling"):
        out[col] = piv.get(col)
    out["specialisation"] = out["ceiling"] - out["matched"]
    out["starvation"] = out["matched"] - out["baseline"]
    out["net"] = out["ceiling"] - out["baseline"]
    return (out.dropna(subset=["ceiling"])
               .sort_values("n_train", ascending=False)
               .round(4))


def gap(frame: pd.DataFrame) -> float:
    """Overall ceiling minus baseline - the naive, confounded number."""
    o = frame[frame["scope"] == "overall"].set_index("system")["macro_f1"]
    return float(o["ceiling"] - o["baseline"])


def specialisation_effect(frame: pd.DataFrame) -> float:
    """Mean per-topic specialisation effect - the honest headline number."""
    return float(decompose(frame)["specialisation"].mean())


def inspect_terms(frame: pd.DataFrame, k: int = 15):
    """Largest-magnitude terms behind the baseline classifier.

    If the top terms are stopwords or platform slang rather than deception
    cues, the score is driven by something other than the task - see the
    length/platform confound documented in src/data.py.
    """
    vec, clf = frame.attrs.get("vectoriser"), frame.attrs.get("baseline_clf")
    if vec is None or clf is None:
        raise ValueError("Frame carries no fitted model; call run() first.")
    return F.top_terms(vec, clf.coef_, k)
