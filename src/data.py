"""
Loading and normalising the corpora.

CONTRACT — agreed across all four approaches. Do not change these signatures
without telling the team; every approach depends on them.

IMPORTANT PROPERTY OF FINEFAKE
------------------------------
Text length is strongly confounded with both platform and label:

    platform         n      median words   % fake
    snope          7556          18         67.3
    reddit         4048           7         76.3
    twitter         952          22         62.1
    cnn            2310         266         10.3
    cdc_gov         268         244          1.1
    apnews          734         891          9.3
    washingtonpost 1041        1021         32.3

Short text is mostly fake; long text is mostly real. A model can therefore
score well by learning "short = fake", which is document length rather than
deception. Keep `n_words` in the frame so this stays visible, and treat any
result that looks too good as suspect until length is ruled out.

Only ~24% of rows reach 100 words. `min_words` is available but defaults to
None so that nothing is filtered silently.

DUPLICATES
----------
The raw corpus has 2,362 exactly duplicated rows, over half of all Reddit
content. Loading without deduplication leaks 12.5% of the test split into
training and inflates every score. `deduplicate=True` is therefore the
default - see the parameter docs below.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

# Canonical columns every approach expects, whatever the source corpus.
SCHEMA = ["text", "label", "topic", "platform", "n_words"]

# Label semantics, fixed. Verified against FineFake's fine-grained label:
# fine-grained 0 == real, 1..5 == fake types, and these map exactly onto
# label 1 and label 0 respectively.
LABEL_FAKE = 0
LABEL_REAL = 1

_REPO = Path(__file__).resolve().parents[1]


def find_data_dir() -> Path:
    """Locate the directory holding FineFake.pkl.

    Checks, in order:
      1. <repo>/data/      a fresh clone that followed data/README.md
      2. <repo>/../data/   the shared folder alongside the repo

    Raises
    ------
    FileNotFoundError
        If neither location contains the dataset.
    """
    for candidate in (_REPO / "data", _REPO.parent / "data"):
        if (candidate / "FineFake.pkl").exists() or (candidate / "finefake.parquet").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find FineFake. Looked in:\n"
        f"  {_REPO / 'data'}\n"
        f"  {_REPO.parent / 'data'}\n"
        "See data/README.md for how to obtain it."
    )


def load_finefake(
    path: Path | None = None,
    min_words: int | None = None,
    truncate_words: int | None = None,
    drop_uncategorized: bool = False,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Load FineFake and normalise it to SCHEMA.

    Prefers a cached `finefake.parquet` if present (fast); otherwise reads
    the ~186 MB pickle and is slow. Call `cache_slim()` once to create it.

    Parameters
    ----------
    path
        Path to FineFake.pkl or finefake.parquet. If None, auto-resolved.
    min_words
        If given, drop rows with fewer than this many whitespace-separated
        tokens. Default None — nothing is filtered. See the module docstring
        before choosing a value; filtering to 100 keeps only ~24% of rows.
    truncate_words
        If given, cut every document to its first N whitespace tokens.
        Combined with min_words=N this makes every document EXACTLY N words,
        which removes document length as a predictive feature by
        construction. See the length confound in the module docstring: on the
        raw corpus, log(word count) alone reaches macro-F1 0.72, so any
        result obtained without controlling for it is largely measuring
        length rather than deception.
    drop_uncategorized
        FineFake has a 7th "Uncategorized" topic with 113 rows, too few to
        train a specialist on. Set True to exclude it.
    deduplicate
        Default True, and it matters. FineFake contains 2,362 exactly
        duplicated rows (52.9% of all Reddit rows). Left in, 12.5% of the
        test split has text that also appears in training, and the model
        scores macro-F1 0.93 on those memorised rows against 0.75 on unseen
        ones - inflating every headline number.

        Rows whose text carries BOTH labels (22 groups) are dropped
        entirely, since neither label can be trusted; remaining repeats keep
        their first occurrence. Set False only to reproduce the inflated
        pre-deduplication numbers.

    Returns
    -------
    pd.DataFrame
        Columns exactly SCHEMA, in that order:
          text     : str  article body (often short — see module docstring)
          label    : int  0 = fake, 1 = real
          topic    : str  Politics, Society, Entertainment, Conflict,
                          Business, Health, Uncategorized
          platform : str  snope, reddit, cnn, washingtonpost, twitter,
                          apnews, cdc_gov
          n_words  : int  token count, retained so the length confound
                          stays inspectable

        Rows with empty text are dropped. The index is reset to a clean
        RangeIndex — split indices refer to THIS index, so it must be
        deterministic. Any filtering therefore changes the index, which is
        why splits.json records a fingerprint of the frame it was built on.

    Notes
    -----
    The raw pickle also carries knowledge embeddings, entity relations,
    author, date, comments and image paths. None are used: this project is
    text-only by design (deployability vertical). They are dropped here.
    """
    if path is None:
        d = find_data_dir()
        parquet, pickle_ = d / "finefake.parquet", d / "FineFake.pkl"
        path = parquet if parquet.exists() else pickle_
    path = Path(path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_pickle(path)

    if "n_words" not in df.columns:
        df = df.copy()
        df["text"] = df["text"].fillna("").astype(str)
        df["n_words"] = df["text"].str.split().str.len().fillna(0).astype(int)

    df = df[[c for c in SCHEMA if c in df.columns]].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]

    if deduplicate:
        key = df["text"].str.strip().str.lower()
        # Same text with BOTH labels means neither can be trusted - drop all.
        conflicted = df.groupby(key)["label"].transform("nunique") > 1
        df = df[~conflicted]
        # Then keep one copy of each remaining repeated text.
        df = df[~df["text"].str.strip().str.lower().duplicated(keep="first")]

    if drop_uncategorized:
        df = df[df["topic"] != "Uncategorized"]
    if min_words is not None:
        df = df[df["n_words"] >= min_words]

    if truncate_words is not None:
        df = df.copy()
        df["text"] = df["text"].str.split().str[:truncate_words].str.join(" ")
        df["n_words"] = df["text"].str.split().str.len().astype(int)

    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)[SCHEMA]


def load_isot(path: Path | None = None) -> pd.DataFrame:
    """Load ISOT and normalise it to SCHEMA.

    Used only in Approach D as a held-out transfer test. Never trained on.

    `topic` is taken from ISOT's `subject` column; `platform` is set to
    "isot" because ISOT has no comparable grouping.

    Warning
    -------
    ISOT's labels are source-level: an article is real because it came from
    Reuters and fake because its host site was flagged — not because that
    individual claim was checked. Weaker ground truth than FineFake, which
    is exactly why it is transfer-only.
    """
    d = Path(path) if path else find_data_dir()
    true_csv, fake_csv = d / "True.csv", d / "Fake.csv"
    if not true_csv.exists() or not fake_csv.exists():
        raise FileNotFoundError(f"Expected True.csv and Fake.csv in {d}")

    real = pd.read_csv(true_csv)
    fake = pd.read_csv(fake_csv)
    real["label"], fake["label"] = LABEL_REAL, LABEL_FAKE

    df = pd.concat([real, fake], ignore_index=True)
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    df["topic"] = df["subject"].fillna("unknown")
    df["platform"] = "isot"
    df["n_words"] = df["text"].str.split().str.len().fillna(0).astype(int)

    df = df[df["text"] != ""]
    return df.reset_index(drop=True)[SCHEMA]


def cache_slim(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the SCHEMA columns to parquet for fast reloads.

    The raw pickle is ~186 MB and takes tens of seconds. Call this once
    after the first load; afterwards `load_finefake()` picks up the parquet
    automatically.
    """
    if path is None:
        path = find_data_dir() / "finefake.parquet"
    path = Path(path)
    df[SCHEMA].to_parquet(path, index=False)
    return path


def fingerprint(df: pd.DataFrame) -> str:
    """Stable short hash of a frame's identity.

    Recorded in splits.json so a later run can detect that the underlying
    data changed — which would silently invalidate the saved split indices.
    """
    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    h.update("|".join(df.columns).encode())
    for pos in (0, len(df) // 2, len(df) - 1):
        if 0 <= pos < len(df):
            h.update(str(df.iloc[pos]["text"])[:200].encode("utf-8", "ignore"))
            h.update(str(df.iloc[pos]["label"]).encode())
    return h.hexdigest()[:16]
