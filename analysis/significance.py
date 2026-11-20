"""
Paired significance test: is the SVM really better than the CNN?

Both models predict every one of the 636 images exactly once (out-of-fold, controlled
protocol, grouped split). The exact McNemar test looks only at the images on which the
two models disagree: b = wrong only by the SVM, c = wrong only by the CNN. If the models
were equally good, b and c would follow Binomial(b + c, 0.5).

Run:   python analysis/significance.py        -> results/tables/mcnemar.json
"""
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from analysis.error_analysis import oof_predictions  # noqa: E402
from analysis.report_tables import load_runs, select_final_cnn  # noqa: E402


def mcnemar():
    runs = load_runs()
    y = pd.read_csv(config.RESULTS_DIR / "dataset_index.csv").label.values
    final = select_final_cnn(runs)
    svm_scores, _ = oof_predictions(runs, "svm", "svm", len(y))
    cnn_scores, _ = oof_predictions(runs, "cnn", final, len(y))
    svm_wrong = (svm_scores >= 0.5).astype(int) != y
    cnn_wrong = (cnn_scores >= 0.5).astype(int) != y
    b = int((svm_wrong & ~cnn_wrong).sum())
    c = int((cnn_wrong & ~svm_wrong).sum())
    both = int((svm_wrong & cnn_wrong).sum())
    p = binomtest(b, b + c, 0.5).pvalue
    out = dict(cnn_configuration=final, wrong_only_svm=b, wrong_only_cnn=c,
               wrong_both=both, p_value=round(float(p), 4))
    (config.RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "tables/mcnemar.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = mcnemar()
    print(json.dumps(r, indent=2))
    print("significant at 5%" if r["p_value"] < 0.05 else "NOT significant at 5%")
