"""
Approach D - Stress and Serve.

Takes whichever of B or C won and tries to break it: an unseen topic, a
different corpus, and a deliberately misrouted article. Then serves it.

Owner: <name>
"""


def leave_one_domain_out(df, splits, features, seed, held_out_topic):
    """Train on five topics, test entirely on the sixth."""
    raise NotImplementedError


def transfer_to_isot(model, isot_df, features):
    """Score a FineFake-trained model on all of ISOT. Never trains."""
    raise NotImplementedError


def router_attack(model, df, splits, features):
    """Prepend an off-topic intro and measure how often routing flips."""
    raise NotImplementedError
