"""
Train / validation / test splits with 5-fold cross-validation.

Two splitting schemes are produced so their difference can be MEASURED:

  "random"  : ordinary stratified split, ignores near-duplicates (the naive way)
  "grouped" : stratified split by group_id -> near-duplicates of a test image
              can never be in train/validation (leakage-free, used for all main results)

For fold k: test = fold k (about 20%), and the remaining 80% is divided into
train (about 2/3 of the whole dataset) and validation (about 13%).
Every image is in the test set exactly once across the 5 folds.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import config


def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    y, groups = df.label.values, df.group_id.values
    idx = np.arange(len(df))
    out = pd.DataFrame({"index": idx, "label": y, "group_id": groups})

    for scheme in ["random", "grouped"]:
        # outer: test fold
        if scheme == "grouped":
            outer = StratifiedGroupKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED)
            outer_iter = outer.split(idx, y, groups)
        else:
            outer = StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED)
            outer_iter = outer.split(idx, y)
        for k, (rest, test) in enumerate(outer_iter):
            role = np.empty(len(df), dtype=object)
            role[test] = "test"
            # inner: take one part of the remaining data as validation
            if scheme == "grouped":
                inner = StratifiedGroupKFold(config.VAL_FRACTION_FOLDS, shuffle=True,
                                             random_state=config.SEED + k)
                tr, va = next(inner.split(rest, y[rest], groups[rest]))
            else:
                inner = StratifiedKFold(config.VAL_FRACTION_FOLDS, shuffle=True,
                                        random_state=config.SEED + k)
                tr, va = next(inner.split(rest, y[rest]))
            role[rest[tr]] = "train"
            role[rest[va]] = "val"
            out[f"{scheme}_fold{k}"] = role
    return out


def get_fold(splits: pd.DataFrame, scheme: str, k: int):
    col = splits[f"{scheme}_fold{k}"].values
    return (np.where(col == "train")[0], np.where(col == "val")[0], np.where(col == "test")[0])


def leakage_report(splits: pd.DataFrame) -> pd.DataFrame:
    """For every scheme/fold: how many test images have a near-duplicate
    (same group_id) inside train or validation?"""
    rows = []
    for scheme in ["random", "grouped"]:
        for k in range(config.N_FOLDS):
            tr, va, te = get_fold(splits, scheme, k)
            seen = set(splits.group_id.values[np.concatenate([tr, va])])
            leaked = sum(g in seen for g in splits.group_id.values[te])
            rows.append(dict(scheme=scheme, fold=k, n_train=len(tr), n_val=len(va),
                             n_test=len(te), test_counterfeit=int(splits.label.values[te].sum()),
                             test_images_with_duplicate_in_train=int(leaked)))
    rep = pd.DataFrame(rows)
    assert rep.loc[rep.scheme == "grouped", "test_images_with_duplicate_in_train"].sum() == 0, \
        "grouped split leaks!"
    return rep
