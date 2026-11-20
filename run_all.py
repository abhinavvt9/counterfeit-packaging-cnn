"""
One command runs the whole project, stage by stage:

    python run_all.py --data dataset                 # everything
    python run_all.py --data dataset --stages cnn    # only some stages

Stages (in order):
  index    scan images, remove exact duplicates, group near-duplicates   -> results/dataset_index.csv
  audit    provenance / shortcut audit                                   -> results/audit/
  splits   5-fold grouped and random splits + leakage check              -> results/splits.csv
  cache    preprocess every image under both protocols                   -> cache/*.npy
  svm      SVM baseline under 3 conditions                               -> results/runs/
  cnn      CNN experiments E1-E7 + shortcut conditions for final CNN     -> results/runs/   (needs PyTorch)
  report   tables and figures                                            -> results/tables, results/figures
  errors   error analysis + Grad-CAM                                     -> results/error_analysis/ (Grad-CAM needs PyTorch)
  final    train the demo model on all data                              -> models/final_cnn.pt  (needs PyTorch)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ALL_STAGES = ["index", "audit", "splits", "cache", "svm", "cnn", "report", "errors", "final"]
CONDITIONS = [("raw", "random"), ("raw", "grouped"), ("controlled", "grouped")]


def load_index():
    return pd.read_csv(config.RESULTS_DIR / "dataset_index.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--stages", nargs="+", default=ALL_STAGES)
    ap.add_argument("--experiments", nargs="+", default=None,
                    help="subset of CNN experiments to run, e.g. E1_proposal_cnn E2_bn_gap")
    ap.add_argument("--verbose", action="store_true", help="print every training epoch")
    args = ap.parse_args()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = config.RESULTS_DIR / "runs"

    # ------------------------------------------------------------ index
    if "index" in args.stages:
        from preprocessing.preprocessing import compute_groups, drop_conflicting_groups, scan_dataset
        df = scan_dataset(args.data)
        if len(df) == 0:
            sys.exit("No labelled images found - check the --data folder.")
        cleaning = dict(df.attrs["cleaning"])
        df = drop_conflicting_groups(compute_groups(df))
        cleaning.update(final_images=len(df),
                        removed_conflicting_label_images=cleaning["unique_images"] - len(df))
        (config.RESULTS_DIR / "audit").mkdir(parents=True, exist_ok=True)
        (config.RESULTS_DIR / "audit/cleaning.json").write_text(json.dumps(cleaning, indent=2))
        df.to_csv(config.RESULTS_DIR / "dataset_index.csv", index=False)

    df = load_index()
    y = df.label.values.astype(int)

    # ------------------------------------------------------------ audit
    audit = None
    if "audit" in args.stages:
        from preprocessing.audit import run_audit, sample_sheets
        audit = run_audit(df)
        sample_sheets(df)
    elif (config.RESULTS_DIR / "audit/audit_result.json").exists():
        audit = json.loads((config.RESULTS_DIR / "audit/audit_result.json").read_text())

    # ------------------------------------------------------------ splits
    from preprocessing.make_splits import leakage_report, make_splits
    if "splits" in args.stages:
        splits = make_splits(df)
        splits.to_csv(config.RESULTS_DIR / "splits.csv", index=False)
        rep = leakage_report(splits)
        rep.to_csv(config.RESULTS_DIR / "leakage_report.csv", index=False)
        print("[splits] leakage check:\n", rep.to_string(index=False))
    splits = pd.read_csv(config.RESULTS_DIR / "splits.csv")

    # ------------------------------------------------------------ cache
    from preprocessing.preprocessing import build_cache, load_cache
    if "cache" in args.stages:
        for protocol in ["raw", "controlled"]:
            build_cache(df, protocol)

    # ------------------------------------------------------------ SVM
    if "svm" in args.stages:
        from baseline.features import extract_all
        from baseline.svm_baseline import run_svm
        for protocol in ["raw", "controlled"]:
            F = extract_all(load_cache(protocol))
            print(f"[svm] {protocol}: feature matrix {F.shape}")
            for p, scheme in CONDITIONS:
                if p == protocol:
                    run_svm(F, y, splits, protocol, scheme, run_dir)
        # Background probe: hide the centre of every image (where the packaging
        # usually is) and let the SVM see ONLY the outer border. If this still
        # works well, the classes differ in photographic style/background.
        from preprocessing.preprocessing import mask_centre
        F_border = extract_all(mask_centre(load_cache("controlled")))
        run_svm(F_border, y, splits, "controlled", "grouped", run_dir, tag="svm_border_only")

    # ------------------------------------------------------------ CNN
    if "cnn" in args.stages:
        from cnn.evaluate import run_cnn_folds
        from cnn.train import get_device
        from experiments.configs import EXPERIMENTS
        print(f"[cnn] device: {get_device()}")
        X = load_cache("controlled")
        names = args.experiments or list(EXPERIMENTS)
        for name in names:
            print(f"[cnn] experiment {name}: {EXPERIMENTS[name]}")
            run_cnn_folds(name, EXPERIMENTS[name], X, y, splits, "controlled", "grouped",
                          run_dir, verbose=args.verbose)
        # shortcut study: final configuration under the two "naive" conditions
        if args.experiments is None:
            from analysis.report_tables import load_runs, select_final_cnn
            final = select_final_cnn(load_runs())
            print(f"[cnn] selected configuration (best validation AUC): {final}")
            Xraw = load_cache("raw")
            for protocol, scheme in CONDITIONS[:2]:
                run_cnn_folds(final, EXPERIMENTS[final], Xraw, y, splits, protocol, scheme,
                              run_dir, verbose=args.verbose)

    # ------------------------------------------------------------ report
    if "report" in args.stages:
        from analysis.report_tables import build_all
        build_all(y, audit)

    # ------------------------------------------------------------ errors
    if "errors" in args.stages:
        from analysis.error_analysis import run_error_analysis
        run_error_analysis(df, y)

    # ------------------------------------------------------------ final model
    if "final" in args.stages:
        from cnn.final_model import train_final_model
        train_final_model(df, y)


if __name__ == "__main__":
    main()
