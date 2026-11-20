"""
Provenance / shortcut audit.

Question: can the label be predicted WITHOUT looking at the packaging at all,
only from how the file was produced (format, resolution, file size, brightness)?

If a classifier that sees only these 5 numbers is highly accurate, then the two
classes were collected differently, and any image model may learn that
difference ("shortcut learning") instead of counterfeit-packaging features.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config


def metadata_features(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        (df.file_format == "PNG").astype(float),
        np.log(df.width), np.log(df.height),
        np.log(df.file_size_kb + 1e-3), df.mean_brightness,
    ])


def run_audit(df: pd.DataFrame, out_dir=None) -> dict:
    out_dir = out_dir or (config.RESULTS_DIR / "audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    d["class"] = d.label.map(config.CLASS_NAMES)
    d["min_side"] = np.minimum(d.width, d.height)

    # ---- 1. descriptive table per class
    fmt = pd.crosstab(d["class"], d.file_format)
    summary = d.groupby("class").agg(
        n_images=("label", "size"),
        median_width=("width", "median"), median_height=("height", "median"),
        median_file_kb=("file_size_kb", "median"), mean_file_kb=("file_size_kb", "mean"),
        mean_brightness=("mean_brightness", "mean"),
        share_smaller_than_input=("min_side", lambda s: float((s < config.IMG_SIZE).mean())),
    )
    summary.to_csv(out_dir / "audit_summary_by_class.csv")
    fmt.to_csv(out_dir / "audit_file_format_by_class.csv")

    # ---- 2. metadata-only classifier (5 numbers, no pixels of packaging)
    X, y, g = metadata_features(d), d.label.values, d.group_id.values
    cv = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pred = cross_val_predict(clf, X, y, groups=g, cv=cv)
    meta_acc = float((pred == y).mean())

    # single rules
    fmt_rule_acc = float(max(((d.file_format == "PNG") == (y == 1)).mean(),
                             ((d.file_format == "PNG") == (y == 0)).mean()))
    thr = np.median(d.mean_brightness)
    bright_rule_acc = float(max(((d.mean_brightness > thr) == (y == 1)).mean(),
                                ((d.mean_brightness > thr) == (y == 0)).mean()))

    result = dict(
        n_images=int(len(d)),
        metadata_only_classifier_accuracy=meta_acc,
        file_format_rule_accuracy=fmt_rule_acc,
        brightness_median_rule_accuracy=bright_rule_acc,
        majority_class_accuracy=float(max(y.mean(), 1 - y.mean())),
    )
    (out_dir / "audit_result.json").write_text(json.dumps(result, indent=2))

    # ---- 3. figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    colors = {"genuine": "#2a78d6", "counterfeit": "#eb6834"}
    for c in ["genuine", "counterfeit"]:
        s = d[d["class"] == c]
        axes[0].hist(s.mean_brightness, bins=25, alpha=.6, color=colors[c], label=c)
        axes[1].hist(np.log10(s.file_size_kb), bins=25, alpha=.6, color=colors[c], label=c)
        axes[2].scatter(s.width, s.height, s=8, alpha=.5, color=colors[c], label=c)
    axes[0].set_xlabel("mean brightness (0-1)")
    axes[1].set_xlabel("log10 file size (KB)")
    axes[2].set_xlabel("width (px)"); axes[2].set_ylabel("height (px)")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    for a in axes:
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Provenance audit: a classifier using only file metadata reaches "
                 f"{meta_acc:.1%} accuracy", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "audit_plots.png", dpi=160)
    plt.close(fig)

    print("[audit] per-class summary:\n", summary.round(3).to_string())
    print("[audit] file formats:\n", fmt.to_string())
    print("[audit]", json.dumps(result, indent=2))
    return result


def sample_sheets(df: pd.DataFrame, n=40, seed=7, out_dir=None):
    """Random sample of each class for a manual visual review (results/audit/review_<class>.png)."""
    from preprocessing.preprocessing import preprocess_file
    out_dir = out_dir or (config.RESULTS_DIR / "audit")
    rng = np.random.default_rng(seed)
    for lab, name in config.CLASS_NAMES.items():
        idx = rng.choice(np.where(df.label.values == lab)[0], n, replace=False)
        fig, axes = plt.subplots(5, 8, figsize=(16, 10.5))
        for k, (a, i) in enumerate(zip(axes.ravel(), idx)):
            a.imshow(preprocess_file(df.path.iloc[i], "raw"))
            a.axis("off")
            a.set_title(str(k), fontsize=9)
        fig.suptitle(f"Random sample of {n} '{name}' images (seed {seed})", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"review_{name}.png", dpi=80)
        plt.close(fig)
