"""
Run Approach A on both corpus variants and compare.

    python scripts/run_approach_a.py

Two configurations:

  raw          all 16,909 rows, text as-is.
               Document length is a strong predictor here, so both the
               floor and the ceiling are partly measuring word count.

  trunc100     rows with >= 100 words, every document cut to exactly 100.
               Length is constant by construction, so it cannot carry any
               signal. Fewer rows, but the comparison is real.

Each variant gets its own frozen split, because filtering changes the row
count and therefore the index that split indices refer to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import data, diagnostics, splits  # noqa: E402
from src.approaches import approach_a as A  # noqa: E402

pd.set_option("display.width", 200)

VARIANTS = {
    "raw": dict(min_words=None, truncate_words=None),
    "trunc100": dict(min_words=100, truncate_words=100),
}


def splits_path_for(variant: str) -> Path:
    return REPO / "artifacts" / (
        "splits.json" if variant == "raw" else f"splits_{variant}.json"
    )


def prepare(variant: str) -> tuple[pd.DataFrame, dict]:
    """Load a variant and load-or-create its split."""
    df = data.load_finefake(**VARIANTS[variant])
    path = splits_path_for(variant)
    if path.exists():
        sp = splits.load_splits(path)
        try:
            splits.verify_splits(df, sp, path)
        except ValueError:
            sp = splits.make_splits(df)
            splits.save_splits(sp, df, path)
    else:
        sp = splits.make_splits(df)
        splits.save_splits(sp, df, path)
    return df, sp


def run_variant(variant: str) -> pd.DataFrame:
    df, sp = prepare(variant)
    tr, _, te = splits.apply_splits(df, sp)

    print("=" * 78)
    print(f"VARIANT: {variant}   rows={len(df):,}   "
          f"train={len(tr):,}  test={len(te):,}   "
          f"median words={int(df['n_words'].median())}")
    print("=" * 78)

    res = A.run(df, sp, seed=42, write=False,
                splits_path=splits_path_for(variant))

    overall = res[res["scope"] == "overall"]
    print("\noverall")
    print(overall[["system", "macro_f1", "accuracy", "roc_auc", "mcc"]]
          .round(4).to_string(index=False))
    print(f"\npooled ceiling - baseline = {A.gap(res):+.4f}   (the naive, confounded number)")

    dec = A.decompose(res)
    print("\nper topic - decomposed")
    print(dec.to_string())
    print(f"\n  mean specialisation = {dec['specialisation'].mean():+.4f}"
          f"   positive in {(dec['specialisation'] > 0).sum()}/{len(dec)} topics")
    print(f"  mean starvation     = {dec['starvation'].mean():+.4f}")
    print(f"  mean net            = {dec['net'].mean():+.4f}"
          f"   (= specialisation + starvation)")

    base_f1 = float(overall.set_index("system").loc["baseline", "macro_f1"])
    print()
    diagnostics.report(tr, te, baseline_f1=base_f1)
    print()

    res = res.assign(variant=variant)
    return res


if __name__ == "__main__":
    frames = [run_variant(v) for v in VARIANTS]
    allres = pd.concat(frames, ignore_index=True)

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    rows = []
    for v in VARIANTS:
        sub = allres[allres.variant == v]
        ov = sub[sub["scope"] == "overall"].set_index("system")["macro_f1"]
        d = A.decompose(sub)
        rows.append({
            "variant": v,
            "baseline": ov.get("baseline"),
            "ceiling": ov.get("ceiling"),
            "pooled gap": ov.get("ceiling") - ov.get("baseline"),
            "specialisation": d["specialisation"].mean(),
            "starvation": d["starvation"].mean(),
            "net (per-topic)": d["net"].mean(),
        })
    print(pd.DataFrame(rows).set_index("variant").round(4).to_string())
    print("\n  net (per-topic) = specialisation + starvation.")
    print("  'pooled gap' is computed on all test rows at once, so it is a")
    print("  different aggregation and will not match the per-topic mean.")
    print("\n  A near-zero net does NOT mean specialising fails - it means the")
    print("  gain is cancelled by the cost of splitting the training data.")

    out = REPO / "results" / "approach_a.csv"
    allres.to_csv(out, index=False)
    print(f"\nwritten: {out.relative_to(REPO)}")
