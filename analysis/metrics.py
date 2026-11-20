"""
Evaluation metrics. Counterfeit (label 1) is the POSITIVE class.

  TP = counterfeit predicted counterfeit      FN = counterfeit predicted genuine  (the costly error)
  TN = genuine predicted genuine              FP = genuine predicted counterfeit

  accuracy  = (TP + TN) / all
  precision = TP / (TP + FP)      "of the packs we flagged, how many were really counterfeit?"
  recall    = TP / (TP + FN)      "of the real counterfeits, how many did we catch?"
  F1        = 2PR / (P + R)
  ROC-AUC   = probability that a random counterfeit gets a higher score than a random genuine
              image (threshold-independent; 0.5 = chance)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def compute_metrics(y_true, score, threshold=0.5) -> dict:
    """score = model score for the counterfeit class (probability-like for CNN,
    decision value mapped with threshold for SVM)."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    y_pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, score)
    except ValueError:
        auc = float("nan")
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=auc,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


METRIC_COLS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]


def summarise_folds(fold_rows: list) -> dict:
    """mean and std of each metric across folds + summed confusion matrix."""
    df = pd.DataFrame(fold_rows)
    out = {}
    for m in METRIC_COLS:
        out[f"{m}_mean"] = float(df[m].mean())
        out[f"{m}_std"] = float(df[m].std(ddof=1)) if len(df) > 1 else 0.0
    for c in ["tn", "fp", "fn", "tp"]:
        out[c] = int(df[c].sum())
    out["n_folds"] = len(df)
    return out


def fmt(mean, std):
    return f"{100 * mean:.1f} ± {100 * std:.1f}"
