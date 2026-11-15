import math

import pytest

from backends import label_probs
from common import derive_seed, master_seed
from conditions import option_orders, to_original, top_and_margin
from prep_data import ragtruth_splits


def test_master_seed_and_derivation_are_fixed():
    assert master_seed() == 20261113
    assert derive_seed("B", "split") == derive_seed("B", "split")
    assert derive_seed("B", "split") != derive_seed("A", "split")
    assert derive_seed("B", "split") != derive_seed("B", "order_O5")


def test_binary_orders():
    assert option_orders(2, seed=1) == {"O0": [0, 1], "O1": [1, 0]}


@pytest.mark.parametrize("k, shifts", [(18, (5, 9, 14)), (16, (4, 8, 12))])
def test_multiclass_orders(k, shifts):
    orders = option_orders(k, seed=123)
    assert sorted(orders) == ["O0", "O1", "O2", "O3", "O4", "O5"]
    for order in orders.values():
        assert sorted(order) == list(range(k))
    assert orders["O1"] == list(range(k))[::-1]
    for oid, s in zip(("O2", "O3", "O4"), shifts):
        assert orders[oid][0] == s
    assert option_orders(k, seed=123)["O5"] == orders["O5"]
    assert len({tuple(o) for o in orders.values()}) == 6


def test_map_back_to_original_identity():
    order = [2, 0, 1]                      # position 0 shows option 2, ...
    assert to_original([0.7, 0.2, 0.1], order) == [0.2, 0.1, 0.7]


def test_margin():
    top, margin = top_and_margin([0.2, 0.8])
    assert top == 1 and margin == pytest.approx(abs(2 * 0.8 - 1))


def test_label_probs_all_present_renormalised():
    pairs = [("A", math.log(0.6)), (" A", math.log(0.1)), ("B", math.log(0.2)), ("x", math.log(0.1))]
    probs, missing, flagged = label_probs(pairs, 2)
    assert probs == pytest.approx([0.7 / 0.9, 0.2 / 0.9])
    assert missing == pytest.approx(0.1) and not flagged


def test_label_probs_missing_label_gets_remaining_mass():
    pairs = [("A", math.log(0.5)), ("B", math.log(0.3))]
    probs, missing, flagged = label_probs(pairs, 4)
    assert probs == pytest.approx([0.5, 0.3, 0.1, 0.1])
    assert flagged and sum(probs) == pytest.approx(1.0)


def _synthetic_ragtruth(n_train_sources, n_test_sources, per_source=6):
    items = []
    for split, n in (("train", n_train_sources), ("test", n_test_sources)):
        for s in range(n):
            for r in range(per_source):
                items.append({"item_id": f"{split}-{s}-{r}", "group_id": f"{split}-{s}",
                              "official_split": split, "label": "supported"})
    return items


CFG = {"dev_share": 0.10, "min_official_test": 1500, "calib_size": 1000, "calib_size_fallback": 500, "min_test": 1000}


def _check_disjoint(splits):
    groups = [{it["group_id"] for it in v} for v in splits.values()]
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i + 1:])


def test_ragtruth_uses_official_test_when_large_enough():
    splits, info = ragtruth_splits(_synthetic_ragtruth(500, 450), CFG, seed=7)
    assert info["split_rule"] == "official_test"
    assert {it["official_split"] for it in splits["calib"] + splits["test"]} == {"test"}
    assert len(splits["calib"]) <= 1000 and len(splits["test"]) >= 1000
    _check_disjoint(splits)


def test_ragtruth_resplits_and_falls_back_when_small():
    splits, info = ragtruth_splits(_synthetic_ragtruth(300, 50), CFG, seed=7)
    assert info["split_rule"] == "resplit"
    assert info["calib_size_cap"] == 500 and len(splits["calib"]) <= 500
    _check_disjoint(splits)


def test_ragtruth_split_is_deterministic():
    a, _ = ragtruth_splits(_synthetic_ragtruth(300, 300), CFG, seed=7)
    b, _ = ragtruth_splits(_synthetic_ragtruth(300, 300), CFG, seed=7)
    assert a == b
