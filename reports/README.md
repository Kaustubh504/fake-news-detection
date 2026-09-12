# Approach 2 — Cluster-Scoped Routing (documentation & report)

This folder documents **Approach 2** of the fake-news-detection project:
*Cluster-Scoped Routing and Retrieval for Fake News Detection Without Domain
Labels* (Advanced Data Mining, Phase A, Group E).

Approach 2 detects fake news **without any domain/topic label**. It discovers
clusters in the training data on its own, trains one specialist classifier per
cluster, and routes an unseen article to the cluster it most resembles — the
*"closest cluster decides"* rule.

---

## Contents of this folder

| File | What it is |
|------|------------|
| `Approach2_Architecture_Report.pdf` | Full illustrated report — architecture diagrams, method, and every result table (FineFake + ISOT transfer). |
| `README.md` | This file. |

## Related files elsewhere in the repo

| Path | What it is |
|------|------------|
| `notebooks/02_approach_b.ipynb` | The complete, runnable Approach-2 notebook — dataset-download steps, clustering, per-cluster specialists, routing, evaluation, and the ISOT transfer test. Executed with real outputs. |
| `results/approach_b.csv` | FineFake results: overall + per-cluster rows for the routed system. |
| `results/isot_transfer.csv` | Cross-corpus transfer results (baseline vs Approach 2 on ISOT). |
| `scripts/eval_isot_transfer.py` | Standalone script for the ISOT transfer test (same logic as notebook Section 8). |

> Nothing in the original project (`src/`, the frozen `artifacts/splits.json`,
> Approach A's notebook or `results/approach_a.csv`) was modified. Approach 2 only
> **imports** the shared `src/` contract and adds new files.

---

## Datasets used and how to download them

Neither dataset is committed to git (FineFake alone is ~186 MB). Place both in the
repo's `data/` folder.

### FineFake — primary corpus (training + testing)
- Repo: <https://github.com/Accuser907/FineFake>
- Google Drive: <https://drive.google.com/file/d/16D9ix7ZOisa4VVBznBTBcv1N7TA-jodH/view>
- Save the file as `data/FineFake.pkl`.
- Command line / Colab: `pip install gdown && gdown --id 16D9ix7ZOisa4VVBznBTBcv1N7TA-jodH -O data/FineFake.pkl`
- 16,909 English articles · 6 topics · 8 platforms · binary label **0 = fake, 1 = real**.

### ISOT — transfer test only (never trained on)
- Source: <https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/>
- Download the zip and extract `True.csv` and `Fake.csv` into `data/`.
- 44,898 articles (real = Reuters, fake = flagged sites); median ~375 words.

---

## How to run

```bash
# from the repo root, after placing the datasets in data/
pip install -r requirements.txt
jupyter notebook notebooks/02_approach_b.ipynb   # Run all
```

The notebook loads FineFake, verifies the committed split, builds features,
clusters (k chosen by silhouette), trains per-cluster specialists, routes and
evaluates, then runs the ISOT transfer test. It writes `results/approach_b.csv`
and `results/isot_transfer.csv`, and saves the fitted models under
`artifacts/approach_b_*` (gitignored, regenerable).

---

## Results at a glance (real data)

### In-corpus — FineFake test set (2,537 articles), macro-F1 primary

| System | macro-F1 | accuracy | ROC-AUC |
|--------|:-------:|:-------:|:-------:|
| Baseline — one global model (floor) | **0.7732** | 0.7781 | 0.851 |
| Approach 2 — clustered + routed (no labels) | 0.7494 | 0.7572 | 0.838 |
| Ceiling — oracle routing by true topic | 0.7715 | 0.7770 | 0.847 |

Clusters chosen: **k = 10** (silhouette). Cluster purity vs true topics: **0.393**.

### Cross-corpus transfer — FineFake → ISOT (nothing retrained)

| System | in-corpus F1 | ISOT F1 | ISOT accuracy | fake recall | ROC-AUC |
|--------|:-----------:|:-------:|:-------------:|:-----------:|:-------:|
| Baseline — global | 0.7732 | 0.3955 | 0.5067 | 0.074 | 0.582 |
| Approach 2 — routed | 0.7494 | **0.4185** | 0.5037 | 0.116 | 0.579 |

---

## Findings

- **In-corpus, routing does not beat a single model** (−0.024 macro-F1). Cluster
  purity is low (0.393), so the discovered groups are not clean topics and the
  specialists are trained on mixed data — the cost of splitting isn't recovered.
- **On transfer, Approach 2 is marginally more robust** (+0.023 macro-F1, and it
  catches more fakes: recall 0.116 vs 0.074).
- **Neither model generalises to ISOT** (both ≈0.40, ROC-AUC ≈0.58) — the project's
  central point made concrete.
- **The router collapses under distribution shift**: ~90.6% of ISOT articles route
  to a single FineFake cluster, so routing carries little information off-domain.

These motivate **Approach 3** — a *learned* partition (gated mixture of experts)
where routing is trained from the classification loss rather than raw geometry.

*Scope: this is fabrication detection, not fact verification. Output is a triage
probability, not a verdict on truth.*
