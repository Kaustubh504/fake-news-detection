"""
Approach C - Learned partition (sparse mixture-of-experts).

No clustering step. A gate and N experts train jointly, end to end, with
routing learned from the classification objective alone.

Owner: <name>
"""


def run(df, splits, features, seed, n_experts=None, top_k=1, alpha=0.01):
    """Train the gated mixture and evaluate it.

    Returns
    -------
    list[dict]
        Result rows in the shape defined by src.metrics.RESULT_FIELDS.
    """
    raise NotImplementedError
