"""
Quick check (about 1 minute) that the PyTorch parts work before the long run:
builds every CNN configuration, trains one for 2 epochs, runs Grad-CAM.
Needs the "index", "splits" and "cache" stages to have been run first.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from cnn.gradcam import border_energy_share, gradcam, overlay  # noqa: E402
from cnn.model import build_model, count_parameters  # noqa: E402
from cnn.train import get_device, predict_scores, train_cnn  # noqa: E402
from experiments.configs import EXPERIMENTS  # noqa: E402
from preprocessing.preprocessing import load_cache  # noqa: E402

import pandas as pd  # noqa: E402
from preprocessing.make_splits import get_fold  # noqa: E402

device = get_device()
print("PyTorch", torch.__version__, "| device:", device)
X = load_cache("controlled")
y = pd.read_csv(config.RESULTS_DIR / "dataset_index.csv").label.values.astype(int)
splits = pd.read_csv(config.RESULTS_DIR / "splits.csv")

for name, cfg in EXPERIMENTS.items():
    m = build_model(cfg, img_size=X.shape[1])
    out = m(torch.zeros(2, 3, X.shape[1], X.shape[1]))
    assert out.shape == (2,), out.shape
    print(f"  {name:18s} parameters: {count_parameters(m):>10,}  output shape OK")

config.MAX_EPOCHS = 2
tr, va, te = get_fold(splits, "grouped", 0)
for name in ["E1_proposal_cnn", "E4_augmentation"]:
    model, hist, best = train_cnn(EXPERIMENTS[name], X, y, tr, va, device=device, verbose=True)
    s = predict_scores(model, X[te[:10]], device)
    assert s.shape == (10,) and np.all((s >= 0) & (s <= 1))
cam, score = gradcam(model, X[te[0]], device)
assert cam.shape == X.shape[1:3]
overlay(X[te[0]], cam)
print(f"  Grad-CAM OK (score {score:.2f}, border share {border_energy_share(cam):.2f})")
print("SMOKE TEST PASSED")
