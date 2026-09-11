"""
Confound probes.

Run these alongside every experiment. A detector can score well on this
corpus without reading the text at all, so a headline number means nothing
until you know how much of it survives these checks.

Background: FineFake pools Snopes one-liners (median 18 words, 67% fake)
with full news articles (CNN 266 words 10% fake, APNews 891 words 9% fake).
Short therefore means fake, and a model can learn that instead of learning
deception. On the raw corpus, log(word count) ALONE reaches macro-F1 0.72
against a full TF-IDF baseline of 0.77.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from .data import LABEL_FAKE
from .metrics import evaluate


def _fit_eval(X_tr, y_tr, X_te, y_te) -> dict[str, float]:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_tr, y_tr)
    fake_col = list(clf.classes_).index(LABEL_FAKE)
    return evaluate(y_te, clf.predict(X_te), clf.predict_proba(X_te)[:, fake_col])


def shortcut_baselines(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    """How far do you get WITHOUT reading the text?

    Returns macro-F1 for three text-free predictors. Compare each against
    your real baseline: the gap is how much your text features are actually
    contributing.
    """
    rows = []

    d = DummyClassifier(strategy="most_frequent").fit(tr[["n_words"]], tr["label"])
    rows.append(("majority class", evaluate(te["label"], d.predict(te[["n_words"]]))))

    rows.append((
        "log(word count) alone",
        _fit_eval(np.log1p(tr[["n_words"]]), tr["label"],
                  np.log1p(te[["n_words"]]), te["label"]),
    ))

    oh_tr = pd.get_dummies(tr["platform"])
    oh_te = pd.get_dummies(te["platform"]).reindex(columns=oh_tr.columns, fill_value=0)
    rows.append(("platform identity alone", _fit_eval(oh_tr, tr["label"], oh_te, te["label"])))

    return pd.DataFrame(
        [{"probe": name, "macro_f1": s["macro_f1"], "roc_auc": s["roc_auc"]}
         for name, s in rows]
    )


def length_signal_by_topic(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    """Is the length shortcut present INSIDE each topic?

    This is the decisive check for Approach A. If length predicts the label
    equally well within every topic, then specialising cannot help: each
    specialist just relearns the same non-topic-specific rule the global
    model already had, and the ceiling cannot beat the floor.

    A correlation near zero and a low length-only F1 mean the shortcut has
    been controlled and the comparison is measuring something real.
    """
    rows = []
    for topic in sorted(tr["topic"].unique()):
        a, b = tr[tr.topic == topic], te[te.topic == topic]
        if len(b) == 0 or a["label"].nunique() < 2:
            continue
        s = _fit_eval(np.log1p(a[["n_words"]]), a["label"],
                      np.log1p(b[["n_words"]]), b["label"])
        corr = float(np.corrcoef(np.log1p(a["n_words"]), a["label"])[0, 1])
        rows.append({
            "topic": topic,
            "n_train": len(a),
            "length_only_f1": round(s["macro_f1"], 4),
            "corr_len_real": round(corr, 4),
        })
    return pd.DataFrame(rows)


def report(tr: pd.DataFrame, te: pd.DataFrame, baseline_f1: float | None = None) -> None:
    """Print both probes, with an interpretation line if a baseline is given."""
    print("--- text-free shortcut baselines ---")
    sb = shortcut_baselines(tr, te)
    print(sb.round(4).to_string(index=False))

    if baseline_f1 is not None:
        best = sb["macro_f1"].max()
        print(f"\n  real baseline macro_f1 = {baseline_f1:.4f}")
        print(f"  best text-free probe   = {best:.4f}")
        print(f"  contribution of text   = {baseline_f1 - best:+.4f}")

    print("\n--- length signal within each topic ---")
    print(length_signal_by_topic(tr, te).to_string(index=False))
