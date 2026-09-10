"""
Approach A - Baseline and Ceiling.

Establishes the two reference points every later approach is measured against.

    baseline : one classifier trained on all topics mixed together.
               The floor - what specialising must beat to be worth doing.

    ceiling  : one classifier per topic, six of them. At test time each
               article is routed to its own topic's classifier USING THE
               TRUE TOPIC LABEL. That oracle routing is what makes this an
               upper bound rather than a working system.

Done when: the ceiling beats the baseline. If it does not, specialising
cannot help on this corpus, and that is reported rather than worked around.

Owner: <name>
"""


def run_baseline(df, splits, features, seed):
    """Train one classifier on all training rows; evaluate on the test split.

    Returns
    -------
    list[dict]
        Result rows: one "overall", plus one per topic so per-topic
        performance can be compared against the ceiling's specialists.
    """
    raise NotImplementedError


def run_ceiling(df, splits, features, seed):
    """Train one classifier per topic; route test rows by their TRUE topic.

    Returns
    -------
    list[dict]
        One row per topic, plus an "overall" row aggregating all test rows.

    Notes
    -----
    Each specialist sees roughly a sixth of the data the baseline sees, so a
    weak ceiling may reflect data scarcity rather than specialisation
    failing. Record n_train per topic so this is visible in the results.
    """
    raise NotImplementedError
