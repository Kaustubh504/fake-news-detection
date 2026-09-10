"""
The train / validation / test split.

CONTRACT — agreed across all four approaches.

CRITICAL: the split is created ONCE and committed to artifacts/splits.json.
Every approach loads that file. Nobody regenerates it. If two approaches run
on different splits, the comparison between them is meaningless and nothing
in the results will reveal it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import fingerprint

SPLITS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "splits.json"

SEED = 42
RATIOS = (0.70, 0.15, 0.15)  # train, val, test


def _strata(df: pd.DataFrame) -> pd.Series:
    """Joint (label, topic) stratification key.

    Stratifying on label alone would let a whole topic land mostly in one
    split, quietly breaking every per-topic experiment. Both must be held.
    """
    return df["label"].astype(str) + "|" + df["topic"].astype(str)


def make_splits(
    df: pd.DataFrame,
    seed: int = SEED,
    ratios: tuple[float, float, float] = RATIOS,
) -> dict[str, list[int]]:
    """Create a stratified train/val/test split.

    Parameters
    ----------
    df
        Frame from data.load_finefake(), with its clean RangeIndex.
    seed
        Fixed. Do not vary this to "see if results improve".
    ratios
        Must sum to 1.0.

    Returns
    -------
    dict
        {"train": [...], "val": [...], "test": [...]} — positional indices
        into `df`, disjoint, together covering every row.

    Raises
    ------
    ValueError
        If ratios do not sum to 1, or if any (label, topic) stratum is too
        small to split. The message names the offending stratum, since the
        fix is a data decision (drop the topic, or merge it) rather than
        something to silence.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} summing to {sum(ratios)}")

    strata = _strata(df)
    too_small = strata.value_counts()
    too_small = too_small[too_small < 3]
    if len(too_small):
        raise ValueError(
            "These (label|topic) strata have fewer than 3 rows and cannot be "
            f"split three ways:\n{too_small.to_string()}\n"
            "Drop or merge the topic before splitting."
        )

    idx = df.index.to_numpy()
    train_frac, val_frac, test_frac = ratios

    train_idx, rest_idx = train_test_split(
        idx, train_size=train_frac, random_state=seed, stratify=strata.to_numpy()
    )
    # val vs test, proportioned within the remainder
    rest_strata = strata.loc[rest_idx].to_numpy()
    val_share = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=val_share, random_state=seed, stratify=rest_strata
    )

    return {
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
        "test": sorted(int(i) for i in test_idx),
    }


def save_splits(
    splits: dict[str, list[int]],
    df: pd.DataFrame,
    path: Path = SPLITS_PATH,
    seed: int = SEED,
    ratios: tuple[float, float, float] = RATIOS,
) -> Path:
    """Write splits plus the metadata needed to verify them later.

    The stored fingerprint lets a later run detect that the underlying data
    changed — a filtering decision, say — which would invalidate these
    indices without any visible error.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "ratios": list(ratios),
        "n_rows": int(len(df)),
        "fingerprint": fingerprint(df),
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_splits(path: Path = SPLITS_PATH) -> dict[str, list[int]]:
    """Load the committed splits.

    Raises if missing rather than silently regenerating — an absent splits
    file is a setup error, not something to paper over.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It is committed to the repo on purpose. "
            "If this is a fresh setup, run notebooks/00_profile.ipynb to "
            "create it — but do NOT regenerate it if it already existed, or "
            "results stop being comparable across approaches."
        )
    return json.loads(path.read_text(encoding="utf-8"))["splits"]


def verify_splits(
    df: pd.DataFrame,
    splits: dict[str, list[int]] | None = None,
    path: Path = SPLITS_PATH,
) -> None:
    """Assert the splits still match the data.

    Call this at the top of every experiment. It is cheap, and it catches
    the single most damaging silent error in this project.

    Raises
    ------
    ValueError
        Naming exactly what mismatched.
    """
    path = Path(path)
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if splits is None:
        splits = meta["splits"]

    sets = {k: set(v) for k, v in splits.items()}
    all_idx = set().union(*sets.values())

    if len(all_idx) != sum(len(v) for v in sets.values()):
        raise ValueError("Split indices overlap — train/val/test are not disjoint.")
    if len(df) != len(all_idx):
        raise ValueError(
            f"Split covers {len(all_idx)} rows but the frame has {len(df)}. "
            "The data was filtered differently from when the split was made."
        )
    if meta.get("fingerprint") and meta["fingerprint"] != fingerprint(df):
        raise ValueError(
            "Data fingerprint does not match splits.json. The underlying "
            "frame changed (different filtering, or a new dataset version), "
            "so these split indices no longer mean what they meant. "
            "Do not proceed — results would not be comparable."
        )


def apply_splits(
    df: pd.DataFrame, splits: dict[str, list[int]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience: return (train_df, val_df, test_df)."""
    return (
        df.loc[splits["train"]],
        df.loc[splits["val"]],
        df.loc[splits["test"]],
    )
