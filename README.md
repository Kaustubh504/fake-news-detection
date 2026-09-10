# Fake News Detection Without Domain Labels

Cluster-scoped routing and retrieval — a detector that discovers what kind of
news article it is looking at, routes it to a specialist trained on that kind,
and shows similar already-labelled articles as evidence for its verdict.

Applied Data Mining course project.

---

## Quick start

```bash
git clone <repo-url>
cd fake-news-detection

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Then get the data — see [`data/README.md`](data/README.md). Nothing in `data/`
is committed; the dataset is 186 MB and GitHub rejects files over 100 MB.

Finally, run `notebooks/00_profile.ipynb` once. It caches a slimmed copy of the
dataset so every later load is fast.

---

## Repository layout

```
fake-news-detection/
├── data/                  gitignored — see data/README.md
├── src/
│   ├── data.py            load + normalise corpora to a fixed schema
│   ├── splits.py          the frozen train/val/test split
│   ├── features.py        TF-IDF (sparse) and SVD (dense)
│   ├── metrics.py         scoring + result recording
│   └── approaches/
│       ├── approach_a.py  baseline and ceiling
│       ├── approach_b.py  geometric partition (clustering)
│       ├── approach_c.py  learned partition (mixture-of-experts)
│       └── approach_d.py  stress testing and serving
├── notebooks/             one per approach — the readable narrative
├── artifacts/             splits.json (committed) + fitted models (ignored)
├── results/               one CSV per approach, committed
└── requirements.txt
```

**The rule:** anything reused across approaches lives in `src/`. Anything
specific to one approach lives in that approach's module and notebook.

---

## The one rule that matters most

`artifacts/splits.json` is created **once** and committed. Every approach loads
it. **Nobody regenerates it.**

If two approaches run on different splits, the comparison between them is
meaningless — and nothing in the output will tell you. Call
`splits.verify_splits()` at the top of every experiment; it is cheap and it
catches the single most damaging silent error in this project.

The same applies to `features.py`: fit on the **training rows only**. Fitting
the vectoriser on the full corpus leaks test vocabulary into training and
quietly inflates every number that follows.

---

## What each `src/` module gives you

These signatures are a **contract**. They were agreed up front so that several
people can work in parallel. Changing one means telling the team.

### `data.py`
```python
load_finefake(path=None) -> DataFrame   # columns: text, label, topic, platform
load_isot(path=None)     -> DataFrame   # same schema; Approach D only
cache_slim(df, path)     -> None        # fast-reload parquet
```
`label` is fixed: **0 = fake, 1 = real**. The returned index is a clean
`RangeIndex` — split indices refer to it, so it must be deterministic.

### `splits.py`
```python
make_splits(df, seed=42, ratios=(.70,.15,.15)) -> dict
save_splits(splits, path)                      -> None
load_splits(path)                              -> dict
verify_splits(df, splits)                      -> None   # raises on mismatch
```
Stratified jointly on `(label, topic)`, so every split preserves both the
fake/real balance and the topic proportions.

### `features.py`
```python
fit_tfidf(train_texts)        -> Pipeline   # sparse — for classifiers
fit_svd(train_tfidf)          -> Pipeline   # dense  — for clustering/routing
transform(pipeline, texts)    -> ndarray
save_pipeline / load_pipeline
```
Two representations from the same text: **sparse TF-IDF** for the classifier so
per-word coefficients stay readable, **dense SVD** for clustering, routing and
retrieval.

### `metrics.py`
```python
evaluate(y_true, y_pred, y_prob=None) -> dict
result_row(...)                       -> dict
append_result(row, approach)          -> Path
load_all_results()                    -> DataFrame
```
**macro-F1 is the primary metric.** Accuracy is recorded but not relied on —
the corpus is roughly 1.6 : 1 real to fake, so always predicting "real" scores
deceptively well.

---

## The four approaches

Run in sequence. Each must meet its success criterion before the next begins.

| | Approach | What it does | Status |
|---|---|---|---|
| **A** | Baseline and Ceiling | One global classifier (floor) vs. one classifier per topic routed by the true label (ceiling) | In progress |
| **B** | Geometric partition | Clustering discovers the topics; nearest-centroid routes; same-cluster neighbours give evidence | Not started |
| **C** | Learned partition | Sparse mixture-of-experts learns the routing end to end, no clustering step | Not started |
| **D** | Stress and Serve | Unseen-topic and cross-corpus tests, router attack, web interface | Not started |

**Approach A defines the reference points.** Every later approach reports where
it lands between that floor and that ceiling — that gap is the headline result
of the whole project.

---

## Working in parallel

**One owner per file.** You edit your own approach module and your own
notebook. Nobody edits someone else's. This alone removes most conflict risk.

**One results file per approach.** `results/approach_a.csv`,
`results/approach_b.csv`, and so on. A single shared CSV would conflict on
every push once more than one person is running experiments.

**Branch per approach**, merged by pull request:

```
main
 ├── approach-a
 ├── approach-b
 └── ...
```

**`src/` is shared** — treat changes to it as team decisions, not personal
ones. It should stabilise during Approach A and stay fixed afterwards.

**Notebooks store their outputs inside the file**, so re-running one produces a
large diff. Since each person owns their own notebook this rarely bites, but if
it becomes painful, `nbstripout` strips outputs automatically on commit.

---

## Design constraints

Three constraints govern every decision. Each is stated as something the
project **forbids**, including choices that would improve reported numbers.

**Deployability** — raw article text in, verdict out. No domain label, no
publisher identity, no social signals, no external service call at inference.

**Explainability** — every verdict carries a reason at two levels: which
features drove it (local), and what the routed cluster is about (global). No
black-box final layer.

**Robustness** — performance must not collapse on an unseen topic or under
deliberate evasion. Clean accuracy is never reported alone.

---

## Scope

This is **fabrication detection, not fact verification**. The system reads how
an article is written; it never checks a claim against the world. A true report
of a landslide killing 100 people and a false one reporting 50 differ by a
single token and receive the same verdict.

Output is a probability for triage, not a verdict.
