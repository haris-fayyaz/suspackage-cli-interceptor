"""Grouped splits: holdout, cross-validation folds and the cross-dataset split."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

log = logging.getLogger(__name__)


def limit_rows_per_group(idx: np.ndarray, groups: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    """Keep at most `cap` rows per group (random, seeded). Used on training rows only."""
    idx = np.asarray(idx)
    if not cap:
        return np.sort(idx)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(idx))
    shuffled, g = idx[order], pd.Series(groups[idx][order])
    rank = g.groupby(g).cumcount().to_numpy()
    return np.sort(shuffled[rank < cap])


def make_holdout_split(y: np.ndarray, groups: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Group holdout: returns (dev_idx, holdout_idx). The holdout is never tuned on."""
    k = max(2, round(1 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    dev, hold = next(iter(sgkf.split(np.zeros(len(y)), y, groups)))
    return dev, hold


def make_grouped_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int,
             cap: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped stratified folds. Training side is capped per group, validation side is not."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        folds.append((limit_rows_per_group(tr, groups, cap, seed + k), va))
    return folds


@dataclass
class CrossDatasetSplit:
    """Cross-dataset split: train on pypi_malregistry + benign, test on DataDog + held-out benign."""
    train: np.ndarray
    test: np.ndarray
    stats: dict


def make_cross_dataset_split(source: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int,
               benign_test_frac: float = 0.2) -> CrossDatasetSplit:
    """Cross-dataset split. Any group that touches the test set is removed from training."""
    source, y, groups = np.asarray(source), np.asarray(y), np.asarray(groups)
    dd = source == "datadog"
    mr = source == "pypi_malregistry"
    benign = y == 0
    rng = np.random.default_rng(seed)
    ben_groups = np.unique(groups[benign])
    n_test = max(1, int(round(len(ben_groups) * benign_test_frac)))
    ben_test_groups = rng.choice(ben_groups, size=n_test, replace=False)

    test_mask = dd | (benign & np.isin(groups, ben_test_groups))
    test_groups = np.unique(groups[test_mask])
    train_mask = (mr | benign) & ~np.isin(groups, test_groups)
    shared = int(len(np.intersect1d(groups[mr], groups[dd])))
    stats = {
        "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        "train_malicious": int((train_mask & (y == 1)).sum()),
        "test_malicious": int((test_mask & (y == 1)).sum()),
        "malregistry_rows_dropped_overlap": int((mr & np.isin(groups, test_groups)).sum()),
        "groups_shared_by_both_malware_sets": shared,
    }
    log.info("cross-dataset split: %s", stats)
    return CrossDatasetSplit(np.flatnonzero(train_mask), np.flatnonzero(test_mask), stats)
