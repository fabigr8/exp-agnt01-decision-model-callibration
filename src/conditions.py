"""Option orders (protocol 4.5) and mapping outputs back to the original option identity.

An order is a list where order[position] = index of the original option shown at that position.
"""

import numpy as np


def _round_half_up(x):
    return int(x + 0.5)


def option_orders(k, seed):
    """Return {order_id: order} for k options. k = 2: O0, O1. k > 2: O0..O5."""
    identity = list(range(k))
    if k == 2:
        return {"O0": identity, "O1": [1, 0]}

    def shift(s):
        return [(i + s) % k for i in range(k)]

    return {
        "O0": identity,
        "O1": identity[::-1],
        "O2": shift(_round_half_up(k / 4)),
        "O3": shift(_round_half_up(k / 2)),
        "O4": shift(_round_half_up(3 * k / 4)),
        "O5": [int(i) for i in np.random.default_rng(seed).permutation(k)],
    }


def to_original(probs_by_position, order):
    """Map per-position probabilities back to the original option indices."""
    out = [0.0] * len(order)
    for pos, original in enumerate(order):
        out[original] = probs_by_position[pos]
    return out


def top_and_margin(probs):
    """Top option index and margin (top minus runner-up; equals |2p - 1| for two options)."""
    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    return ranked[0], probs[ranked[0]] - probs[ranked[1]]
