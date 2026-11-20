"""
Train the model used by the demo.

The 5-fold results already tell us how well the chosen configuration works.
For the demo we train ONE more model on (almost) all images with that
configuration; 15% of the groups are held back only for early stopping.
Its performance is NOT reported separately. The cross-validation numbers are the results.
"""
import json

import torch
from sklearn.model_selection import StratifiedGroupKFold

import config
from analysis.report_tables import load_runs, select_final_cnn
from cnn.model import count_parameters
from cnn.train import train_cnn
from experiments.configs import EXPERIMENTS
from preprocessing.preprocessing import load_cache


def train_final_model(df, y):
    final = select_final_cnn(load_runs())
    if final is None:
        raise RuntimeError("Run the CNN experiments first.")
    cfg = EXPERIMENTS[final]
    X = load_cache("controlled")
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=config.SEED)
    tr, va = next(splitter.split(X, y, df.group_id.values))
    model, history, best_epoch = train_cnn(cfg, X, y, tr, va, seed=config.SEED, verbose=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.MODELS_DIR / "final_cnn.pt")
    meta = dict(experiment=final, config=cfg, img_size=int(X.shape[1]), protocol="controlled",
                n_parameters=count_parameters(model), best_epoch=best_epoch,
                n_train=int(len(tr)), n_val=int(len(va)), class_names=config.CLASS_NAMES)
    (config.MODELS_DIR / "final_config.json").write_text(json.dumps(meta, indent=2))
    print(f"[final] saved models/final_cnn.pt ({final}, best epoch {best_epoch})")
