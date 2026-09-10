"""
Text -> numbers.

CONTRACT — agreed across all four approaches.

Two representations from the same text, serving different jobs:

  sparse TF-IDF   -> the classifier. Each dimension is a word, so learned
                     coefficients stay readable (explainability vertical).
  dense SVD       -> clustering, routing and retrieval. Compact, and
                     distances capture topical similarity.

Approach A needs only the sparse side; B, C and D need both.

A NOTE ON THIS CORPUS
---------------------
FineFake's median record is ~18 words. Character n-grams help more than
usual on text that short, but they also inflate the feature space, so
MAX_FEATURES caps each block. If runs feel slow, set use_char=False first —
it is the expensive half.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
import joblib

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# Defaults. Changing these invalidates comparison with earlier runs, so note
# it in the results if you do.
WORD_NGRAMS = (1, 2)
CHAR_NGRAMS = (3, 5)
MIN_DF = 3
MAX_FEATURES = 100_000
SVD_COMPONENTS = 256


def build_tfidf(
    word_ngrams: tuple[int, int] = WORD_NGRAMS,
    char_ngrams: tuple[int, int] = CHAR_NGRAMS,
    min_df: int = MIN_DF,
    max_features: int = MAX_FEATURES,
    use_char: bool = True,
) -> FeatureUnion | TfidfVectorizer:
    """Construct (but do not fit) the sparse vectoriser."""
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngrams,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    if not use_char:
        return word
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngrams,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        lowercase=True,
    )
    return FeatureUnion([("word", word), ("char", char)])


def fit_tfidf(train_texts, **kwargs):
    """Fit the sparse vectoriser on TRAINING TEXT ONLY.

    Fitting on the full corpus leaks test-set vocabulary statistics into
    training and inflates every downstream number. Pass train rows only.

    Returns
    -------
    Fitted vectoriser. Use `transform()` for val/test — never fit again.
    """
    vec = build_tfidf(**kwargs)
    vec.fit(list(train_texts))
    return vec


def build_svd(n_components: int = SVD_COMPONENTS, seed: int = 42) -> Pipeline:
    """TruncatedSVD followed by L2 normalisation.

    The normalisation matters: k-means minimises Euclidean distance, but for
    text the meaningful signal is direction (which words are emphasised),
    not magnitude (how long the document is). Unit-length vectors make
    Euclidean distance behave like cosine similarity — which also blunts the
    length confound described in data.py.
    """
    return Pipeline([
        ("svd", TruncatedSVD(n_components=n_components, random_state=seed)),
        ("norm", Normalizer(copy=False)),
    ])


def fit_svd(train_matrix: sparse.spmatrix, n_components: int = SVD_COMPONENTS,
            seed: int = 42) -> Pipeline:
    """Fit SVD + normalisation on the TRAINING TF-IDF matrix only."""
    n_components = min(n_components, min(train_matrix.shape) - 1)
    pipe = build_svd(n_components=n_components, seed=seed)
    pipe.fit(train_matrix)
    return pipe


def transform(fitted, data):
    """Apply an already-fitted vectoriser or pipeline. Never fits."""
    return fitted.transform(data if not isinstance(data, (list, tuple)) else list(data))


def save_pipeline(obj, name: str) -> Path:
    """Persist a fitted object to artifacts/<name>.joblib.

    Gitignored — regenerable and large. Saved so a kernel restart does not
    mean refitting.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{name}.joblib"
    joblib.dump(obj, path)
    return path


def load_pipeline(name: str):
    """Load an object saved by save_pipeline()."""
    path = ARTIFACTS / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — fit and save it first.")
    return joblib.load(path)


def top_terms(vectoriser, coefs: np.ndarray, k: int = 15) -> list[tuple[str, float]]:
    """Largest-magnitude features for a linear model.

    This is what makes the explainability vertical concrete: each returned
    pair is a literal word or character n-gram and its learned weight.
    """
    names = np.asarray(vectoriser.get_feature_names_out())
    coefs = np.asarray(coefs).ravel()
    order = np.argsort(np.abs(coefs))[::-1][:k]
    return [(str(names[i]), float(coefs[i])) for i in order]
