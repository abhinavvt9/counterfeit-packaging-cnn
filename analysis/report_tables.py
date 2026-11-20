"""
Collects every run in results/runs/ and produces the tables and figures used in
the report and slides. Needs only pandas/matplotlib (no PyTorch).
"""
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import config
from analysis.metrics import METRIC_COLS, fmt, summarise_folds
from experiments.configs import DESCRIPTIONS

# palette (validated categorical slots 1-3)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white",
})

MAIN = ("controlled", "grouped")
CONDITION_LABELS = {
    ("raw", "random"): "Naive\n(raw images, random split)",
    ("raw", "grouped"): "Leakage removed\n(raw images, grouped split)",
    ("controlled", "grouped"): "Controlled (ours)\n(normalised, grouped split)",
}


def load_runs(run_dir=None):
    run_dir = run_dir or (config.RESULTS_DIR / "runs")
    return [json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))]


def summarise(runs, **filt):
    sel = [r for r in runs if all(r.get(k) == v for k, v in filt.items())]
    sel = sorted(sel, key=lambda r: r["fold"])
    if not sel:
        return None
    s = {f"test_{k}": v for k, v in summarise_folds([r["test"] for r in sel]).items()}
    s.update({f"val_{k}": v for k, v in summarise_folds([r["val"] for r in sel]).items()})
    return s


def select_final_cnn(runs):
    rows = []
    for name in DESCRIPTIONS:
        s = summarise(runs, model="cnn", experiment=name, protocol=MAIN[0], scheme=MAIN[1])
        if s and s["test_n_folds"] == config.N_FOLDS:
            rows.append((s["val_roc_auc_mean"], s["val_f1_mean"], name))
    if not rows:
        return None
    return max(rows)[2]


def ablation_table(runs):
    rows = []
    for name, desc in DESCRIPTIONS.items():
        sel = [r for r in runs if r["model"] == "cnn" and r["experiment"] == name
               and (r["protocol"], r["scheme"]) == MAIN]
        s = summarise(runs, model="cnn", experiment=name, protocol=MAIN[0], scheme=MAIN[1])
        if not s:
            continue
        rows.append({
            "Experiment": name.split("_")[0], "Change": desc,
            "Parameters": sel[0]["n_parameters"],
            "Mean best epoch": round(np.mean([r["best_epoch"] for r in sel]), 1),
            "Val AUC": fmt(s["val_roc_auc_mean"], s["val_roc_auc_std"]),
            "Test Acc": fmt(s["test_accuracy_mean"], s["test_accuracy_std"]),
            "Test Recall": fmt(s["test_recall_mean"], s["test_recall_std"]),
            "Test F1": fmt(s["test_f1_mean"], s["test_f1_std"]),
            "Test AUC": fmt(s["test_roc_auc_mean"], s["test_roc_auc_std"]),
            "_val_auc": s["val_roc_auc_mean"], "_val_auc_std": s["val_roc_auc_std"],
            "_test_auc": s["test_roc_auc_mean"], "_test_auc_std": s["test_roc_auc_std"],
            "_name": name,
        })
    return pd.DataFrame(rows)


def comparison_table(runs, final):
    rows = []
    for label, filt in [("Handcrafted features + SVM", dict(model="svm", experiment="svm")),
                        (f"Custom CNN ({final.split('_')[0]})", dict(model="cnn", experiment=final))]:
        s = summarise(runs, protocol=MAIN[0], scheme=MAIN[1], **filt)
        if not s:
            continue
        row = {"Model": label}
        for m in METRIC_COLS:
            row[m.replace("_", " ").title()] = fmt(s[f"test_{m}_mean"], s[f"test_{m}_std"])
        row.update(TN=s["test_tn"], FP=s["test_fp"], FN=s["test_fn"], TP=s["test_tp"])
        rows.append(row)
    return pd.DataFrame(rows)


def shortcut_table(runs, final, audit):
    rows = []
    for (protocol, scheme), cond in CONDITION_LABELS.items():
        for label, filt in [("SVM", dict(model="svm", experiment="svm")),
                            ("CNN", dict(model="cnn", experiment=final))]:
            s = summarise(runs, protocol=protocol, scheme=scheme, **filt)
            if not s:
                continue
            rows.append(dict(condition=cond.replace("\n", " "), protocol=protocol, scheme=scheme,
                             model=label, accuracy=s["test_accuracy_mean"],
                             accuracy_std=s["test_accuracy_std"],
                             roc_auc=s["test_roc_auc_mean"], roc_auc_std=s["test_roc_auc_std"],
                             f1=s["test_f1_mean"], f1_std=s["test_f1_std"]))
    s = summarise(runs, model="svm_border_only", protocol=MAIN[0], scheme=MAIN[1])
    if s:
        rows.append(dict(condition="Background probe: centre hidden, border only (controlled, grouped)",
                         protocol=MAIN[0], scheme=MAIN[1], model="SVM (border only)",
                         accuracy=s["test_accuracy_mean"], accuracy_std=s["test_accuracy_std"],
                         roc_auc=s["test_roc_auc_mean"], roc_auc_std=s["test_roc_auc_std"],
                         f1=s["test_f1_mean"], f1_std=s["test_f1_std"]))
    df = pd.DataFrame(rows)
    if audit:
        df.attrs["metadata_only_accuracy"] = audit["metadata_only_classifier_accuracy"]
    return df


# ------------------------------------------------------------------ figures
def fig_confusion(runs, final, out):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, (title, filt) in zip(axes, [("SVM baseline", dict(model="svm", experiment="svm")),
                                        ("Custom CNN", dict(model="cnn", experiment=final))]):
        s = summarise(runs, protocol=MAIN[0], scheme=MAIN[1], **filt)
        if not s:
            ax.axis("off")
            continue
        cm = np.array([[s["test_tn"], s["test_fp"]], [s["test_fn"], s["test_tp"]]])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.25)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.0%} of row)", ha="center",
                        va="center", fontsize=10, color=INK)
        ax.set_xticks([0, 1], ["Genuine", "Counterfeit"])
        ax.set_yticks([0, 1], ["Genuine", "Counterfeit"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{title}\n(all 5 test folds combined)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_ablation(abl, final, out):
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    x = np.arange(len(abl))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, abl["_val_auc"], w, yerr=abl["_val_auc_std"], color=BLUE,
           label="Validation AUC (used to choose)", capsize=3, error_kw=dict(ecolor=MUTED, lw=1))
    ax.bar(x + w / 2 + 0.01, abl["_test_auc"], w, yerr=abl["_test_auc_std"], color=ORANGE,
           label="Test AUC (reported)", capsize=3, error_kw=dict(ecolor=MUTED, lw=1))
    labels = [f"{e}\n" + "\n".join(textwrap.wrap(c, 13)) + ("\n(selected)" if n == final else "")
              for e, c, n in zip(abl.Experiment, abl.Change, abl["_name"])]
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(-0.45, 0.505, "chance", color=MUTED, fontsize=8, ha="left", va="bottom")
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("ROC-AUC (mean ± std, 5 folds)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("CNN experiments (controlled protocol, grouped split)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_shortcut(sc, out):
    main_sc = sc[sc.model.isin(["SVM", "CNN"])]
    conds = list(dict.fromkeys(main_sc.condition))
    probe = sc[sc.model == "SVM (border only)"]
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    if len(probe):
        xp = len(conds)
        p = probe.iloc[0]
        b = ax.bar(xp, p.accuracy, 0.36, yerr=p.accuracy_std, color=AQUA, capsize=3,
                   error_kw=dict(ecolor=MUTED, lw=1), label="SVM, image border only")
        ax.text(xp, 0.02, f"{p.accuracy:.0%}", ha="center", va="bottom", color="white",
                fontsize=8.5, fontweight="bold")
        CONDITION_LABELS_REV[p.condition] = "Background probe\n(centre hidden,\nborder only)"
        conds_all = conds + [p.condition]
    else:
        conds_all = conds
    x = np.arange(len(conds))
    w = 0.36
    for off, model, col in [(-w / 2 - 0.01, "SVM", BLUE), (w / 2 + 0.01, "CNN", ORANGE)]:
        d = main_sc[main_sc.model == model].set_index("condition").reindex(conds)
        bars = ax.bar(x + off, d.accuracy, w, yerr=d.accuracy_std, color=col, label=model,
                      capsize=3, error_kw=dict(ecolor=MUTED, lw=1))
        for b, v in zip(bars, d.accuracy):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, 0.02, f"{v:.0%}", ha="center",
                        va="bottom", color="white", fontsize=8.5, fontweight="bold")
    meta = sc.attrs.get("metadata_only_accuracy")
    if meta is not None:
        ax.axhline(meta, color=INK, lw=1.2, ls=":")
        ax.text(-0.5, meta + 0.01, f"file-metadata-only classifier: {meta:.0%}",
                color=INK, fontsize=8, ha="left", va="bottom")
    ax.set_xticks(np.arange(len(conds_all)), [CONDITION_LABELS_REV.get(c, c) for c in conds_all],
                  fontsize=8)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Test accuracy (mean ± std, 5 folds)")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
    ax.set_title("How much of the accuracy comes from shortcuts and leakage?", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


CONDITION_LABELS_REV = {v.replace("\n", " "): v for v in CONDITION_LABELS.values()}


def fig_curves(runs, final, out):
    pairs = [("E1_proposal_cnn", "E1: proposal CNN (Flatten)"), (final, f"{final.split('_')[0]}: selected CNN")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (name, title) in zip(axes, pairs):
        r = [r for r in runs if r["model"] == "cnn" and r["experiment"] == name
             and (r["protocol"], r["scheme"]) == MAIN and r["fold"] == 0]
        if not r:
            ax.axis("off")
            continue
        h = pd.DataFrame(r[0]["history"])
        ax.plot(h.epoch, h.train_loss, color=BLUE, lw=2, label="training loss")
        ax.plot(h.epoch, h.val_loss, color=ORANGE, lw=2, label="validation loss")
        ax.axvline(r[0]["best_epoch"], color=MUTED, ls="--", lw=1)
        ax.text(r[0]["best_epoch"], ax.get_ylim()[1] * 0.95, " best epoch", color=MUTED, fontsize=8)
        ax.set_title(title + " - fold 0", fontsize=10)
        ax.set_xlabel("epoch")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("loss (binary cross-entropy)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_roc(runs, final, y, out):
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    for label, filt, col in [("SVM", dict(model="svm", experiment="svm"), BLUE),
                             ("CNN", dict(model="cnn", experiment=final), ORANGE)]:
        sel = [r for r in runs if all(r.get(k) == v for k, v in filt.items())
               and (r["protocol"], r["scheme"]) == MAIN]
        if not sel:
            continue
        idx = np.concatenate([r["test_index"] for r in sel])
        sc = np.concatenate([r["test_score"] for r in sel])
        fpr, tpr, _ = roc_curve(y[idx], sc)
        ax.plot(fpr, tpr, color=col, lw=2, label=label)
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("False positive rate (genuine flagged)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC, pooled out-of-fold test predictions", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def build_all(y, audit=None):
    runs = load_runs()
    out = config.RESULTS_DIR
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    final = select_final_cnn(runs)
    summary = dict(final_cnn=final)

    if final:
        abl = ablation_table(runs)
        abl.drop(columns=[c for c in abl if c.startswith("_")]).to_csv(
            out / "tables/table_cnn_experiments.csv", index=False)
        fig_ablation(abl, final, out / "figures/cnn_experiments.png")
        fig_curves(runs, final, out / "figures/training_curves.png")

    comp = comparison_table(runs, final or "none")
    comp.to_csv(out / "tables/table_svm_vs_cnn.csv", index=False)
    fig_confusion(runs, final, out / "figures/confusion_matrices.png")
    fig_roc(runs, final, y, out / "figures/roc_curves.png")

    sc = shortcut_table(runs, final or "none", audit)
    if len(sc):
        sc.to_csv(out / "tables/table_shortcut_study.csv", index=False)
        fig_shortcut(sc, out / "figures/shortcut_study.png")

    # machine-readable summary of every model/condition
    all_rows = []
    keys = sorted({(r["model"], r["experiment"], r["protocol"], r["scheme"]) for r in runs})
    for m, e, p, s in keys:
        su = summarise(runs, model=m, experiment=e, protocol=p, scheme=s)
        all_rows.append(dict(model=m, experiment=e, protocol=p, scheme=s, **su))
    pd.DataFrame(all_rows).to_csv(out / "metrics.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[report] final CNN configuration: {final}")
    print(comp.to_string(index=False))
    return summary
