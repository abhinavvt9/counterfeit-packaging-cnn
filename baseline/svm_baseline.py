"""
SVM baseline: handcrafted features -> standardisation -> SVM (RBF kernel).

For each of the 5 folds:
  * features are standardised using statistics of the TRAINING images only
  * C and gamma are chosen by the best F1 on the VALIDATION images
  * the chosen model is evaluated once on the TEST images
"""
import itertools
import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from analysis.metrics import compute_metrics
from preprocessing.make_splits import get_fold

C_GRID = [0.1, 1, 10, 100]
GAMMA_GRID = ["scale", 1e-3, 1e-2]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def run_svm(F, y, splits, protocol, scheme, run_dir, tag="svm"):
    run_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    for k in range(config.N_FOLDS):
        out_file = run_dir / f"{tag}__{protocol}-{scheme}__fold{k}.json"
        tr, va, te = get_fold(splits, scheme, k)
        best = None
        for C, gamma in itertools.product(C_GRID, GAMMA_GRID):
            model = make_pipeline(StandardScaler(),
                                  SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced"))
            model.fit(F[tr], y[tr])
            f1 = f1_score(y[va], model.predict(F[va]), zero_division=0)
            if best is None or f1 > best[0]:
                best = (f1, C, gamma, model)
        _, C, gamma, model = best
        # decision value 0 == SVM boundary; sigmoid maps it to a 0-1 "score" with 0.5 as boundary
        val_score = _sigmoid(model.decision_function(F[va]))
        test_score = _sigmoid(model.decision_function(F[te]))
        res = dict(model=tag, experiment="svm", protocol=protocol, scheme=scheme, fold=k,
                   params=dict(C=C, gamma=gamma),
                   val=compute_metrics(y[va], val_score),
                   test=compute_metrics(y[te], test_score),
                   test_index=te.tolist(), test_score=test_score.tolist())
        out_file.write_text(json.dumps(res))
        fold_results.append(res)
        print(f"  [{tag} {protocol}-{scheme} fold{k}] C={C} gamma={gamma} "
              f"test acc={res['test']['accuracy']:.3f} f1={res['test']['f1']:.3f} "
              f"auc={res['test']['roc_auc']:.3f}")
    return fold_results
