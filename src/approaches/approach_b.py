"""
Approach B - Geometric partition.

Discovers topics with clustering instead of being told them, then routes to
per-cluster specialists and retrieves same-cluster neighbours as evidence.

Depends on src/ being stable. Uses the SAME splits and features as Approach
A - load them, do not recreate them.

Owner: <name>
"""


def run(df, splits, features, seed, k=None):
    """Cluster, train per-cluster specialists, route by nearest centroid.

    Returns
    -------
    list[dict]
        Result rows in the shape defined by src.metrics.RESULT_FIELDS.
    """
    raise NotImplementedError
