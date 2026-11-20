"""
Error analysis on out-of-fold TEST predictions (each image is predicted by the
model of the fold in which it was a test image, i.e. a model that never saw it).

Outputs (results/error_analysis/):
  misclassified_<model>.csv        every wrong prediction with its score and file info
  errors_<model>.png               picture grid of the wrong predictions
  gradcam_examples.png             Grad-CAM heatmaps (CNN only, needs PyTorch)
  gradcam_border_share.json        how much of the CNN's attention falls on the image border
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from analysis.report_tables import MAIN, load_runs, select_final_cnn
from preprocessing.preprocessing import load_cache


def oof_predictions(runs, model, experiment, n):
    score = np.full(n, np.nan)
    fold = np.full(n, -1)
    for r in runs:
        if (r["model"], r["experiment"], r["protocol"], r["scheme"]) == (model, experiment, *MAIN):
            score[r["test_index"]] = r["test_score"]
            fold[r["test_index"]] = r["fold"]
    return score, fold


def picture_grid(X, idx, titles, out, ncol=8, title=""):
    if len(idx) == 0:
        return
    nrow = int(np.ceil(len(idx) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.6, nrow * 1.85))
    axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.axis("off")
    for a, i, t in zip(axes.ravel(), idx, titles):
        a.imshow(X[i])
        a.set_title(t, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run_error_analysis(df, y):
    out = config.RESULTS_DIR / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    X = load_cache("controlled")
    final = select_final_cnn(runs)
    targets = [("svm", "svm")] + ([("cnn", final)] if final else [])
    summary = {}

    for model, exp in targets:
        score, fold = oof_predictions(runs, model, exp, len(y))
        pred = (score >= 0.5).astype(int)
        wrong = np.where((pred != y) & ~np.isnan(score))[0]
        d = df.iloc[wrong][["rel_path", "label", "file_format", "width", "height",
                            "file_size_kb", "mean_brightness"]].copy()
        d["true_class"] = d.label.map(config.CLASS_NAMES)
        d["model_score"] = score[wrong]
        d["error_type"] = np.where(y[wrong] == 1, "FN (missed counterfeit)", "FP (genuine flagged)")
        d["fold"] = fold[wrong]
        d = d.sort_values("model_score")
        d.to_csv(out / f"misclassified_{model}.csv", index=False)

        # confident mistakes are the most informative ones -> show them first
        conf = np.abs(score[wrong] - 0.5)
        order = wrong[np.argsort(-conf)][:32]
        titles = [f"{'FN' if y[i] else 'FP'}  score {score[i]:.2f}" for i in order]
        picture_grid(X, order, titles, out / f"errors_{model}.png",
                     title=f"{model.upper()} mistakes (most confident first). FN = missed counterfeit, "
                           f"FP = genuine flagged. Score = model score for 'counterfeit'.")
        summary[model] = dict(n_errors=int(len(wrong)),
                              false_negatives=int((y[wrong] == 1).sum()),
                              false_positives=int((y[wrong] == 0).sum()),
                              confident_errors=int((conf > 0.4).sum()))
        print(f"[errors] {model}: {summary[model]}")

    if final:
        try:
            summary["gradcam"] = gradcam_analysis(runs, final, X, y, out)
        except ImportError:
            print("[errors] PyTorch not available - Grad-CAM skipped")
    (out / "error_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def gradcam_analysis(runs, final, X, y, out):
    import torch
    from cnn.gradcam import border_energy_share, gradcam, overlay
    from cnn.model import build_model
    from cnn.train import get_device
    from experiments.configs import EXPERIMENTS

    device = get_device()
    cfg = EXPERIMENTS[final]
    rng = np.random.default_rng(config.SEED)
    shares = {0: [], 1: []}
    examples = []
    for r in sorted([r for r in runs if r["model"] == "cnn" and r["experiment"] == final
                     and (r["protocol"], r["scheme"]) == MAIN], key=lambda r: r["fold"]):
        stem = f"{final}__{MAIN[0]}-{MAIN[1]}__fold{r['fold']}"
        model = build_model(cfg, img_size=X.shape[1]).to(device)
        model.load_state_dict(torch.load(config.CACHE_DIR / "fold_models" / f"{stem}.pt",
                                         map_location=device))
        te = np.array(r["test_index"])
        for i in te:
            cam, s = gradcam(model, X[i], device)
            shares[int(y[i])].append(border_energy_share(cam))
        # 2 examples per class per fold for the figure
        for c in [0, 1]:
            for i in rng.choice(te[y[te] == c], size=min(2, (y[te] == c).sum()), replace=False):
                cam, s = gradcam(model, X[i], device)
                examples.append((i, overlay(X[i], cam), s))

    fig, axes = plt.subplots(4, 5, figsize=(10, 8.8))
    ex_sorted = sorted(examples, key=lambda e: y[e[0]])[:10] + sorted(examples, key=lambda e: -y[e[0]])[:10]
    for a in axes.ravel():
        a.axis("off")
    for a, (i, img, s) in zip(axes.ravel(), ex_sorted):
        a.imshow(img)
        a.set_title(f"true: {config.CLASS_NAMES[int(y[i])]}\nscore {s:.2f}", fontsize=8)
    fig.suptitle("Grad-CAM of the selected CNN on TEST images (red = evidence for 'counterfeit')",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "gradcam_examples.png", dpi=150)
    plt.close(fig)

    res = dict(border_share_genuine=float(np.mean(shares[0])),
               border_share_counterfeit=float(np.mean(shares[1])),
               border_area_share=float(1 - 0.7 ** 2),
               note="share of Grad-CAM mass in the outer 15% frame; the frame is 51% of the area")
    (out / "gradcam_border_share.json").write_text(json.dumps(res, indent=2))
    print("[errors] Grad-CAM border share:", res)
    return res
