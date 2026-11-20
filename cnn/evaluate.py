"""
Runs a CNN configuration over all 5 folds and stores the results.

For each fold:
  1. train on the fold's TRAIN images (validation images used for early stopping)
  2. score VALIDATION images  -> used only to compare configurations
  3. score TEST images        -> reported results
Results go to results/runs/<name>__<protocol>-<scheme>__fold<k>.json and the
trained weights to cache/fold_models/ (used later for Grad-CAM on test images).
Already-finished folds are skipped, so an interrupted run can simply be restarted.
"""
import json
import time

import torch

import config
from analysis.metrics import compute_metrics
from cnn.model import count_parameters
from cnn.train import get_device, predict_scores, train_cnn
from preprocessing.make_splits import get_fold


def run_cnn_folds(name, cfg, X, y, splits, protocol, scheme, run_dir, verbose=False):
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = config.CACHE_DIR / "fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    results = []
    for k in range(config.N_FOLDS):
        stem = f"{name}__{protocol}-{scheme}__fold{k}"
        out_file = run_dir / f"{stem}.json"
        if out_file.exists():
            results.append(json.loads(out_file.read_text()))
            print(f"  [{stem}] already done - skipped")
            continue
        tr, va, te = get_fold(splits, scheme, k)
        t0 = time.time()
        model, history, best_epoch = train_cnn(cfg, X, y, tr, va, seed=config.SEED + k,
                                               device=device, verbose=verbose)
        val_score = predict_scores(model, X[va], device)
        test_score = predict_scores(model, X[te], device)
        res = dict(model="cnn", experiment=name, protocol=protocol, scheme=scheme, fold=k,
                   params=cfg, n_parameters=count_parameters(model), best_epoch=best_epoch,
                   epochs_run=len(history), train_seconds=round(time.time() - t0, 1),
                   history=history,
                   val=compute_metrics(y[va], val_score),
                   test=compute_metrics(y[te], test_score),
                   test_index=te.tolist(), test_score=test_score.tolist())
        torch.save(model.state_dict(), model_dir / f"{stem}.pt")
        out_file.write_text(json.dumps(res))
        results.append(res)
        print(f"  [{stem}] best epoch {best_epoch}/{len(history)} | "
              f"val auc {res['val']['roc_auc']:.3f} | test acc {res['test']['accuracy']:.3f} "
              f"f1 {res['test']['f1']:.3f} auc {res['test']['roc_auc']:.3f} | {res['train_seconds']}s")
    return results
