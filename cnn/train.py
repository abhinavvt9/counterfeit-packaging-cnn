"""
Training loop for the custom CNN.

  loss      : binary cross-entropy on the logit (BCEWithLogitsLoss); the
              counterfeit class is weighted by (#genuine / #counterfeit) in the
              training set so a class imbalance cannot be "won" by predicting
              the majority class
  optimiser : Adam
  batches   : 32 images
  stopping  : after every epoch the validation loss is measured; training stops
              when it has not improved for 10 epochs, and the weights from the
              best epoch are restored ("early stopping")
"""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import config
from cnn.model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Realistic, mild augmentation. NO horizontal/vertical flips: mirrored text
# never occurs on real packaging.
def make_augmentation(img_size):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
    ])


def to_tensor(X_uint8: np.ndarray) -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N,3,H,W) float in [0,1]."""
    return torch.from_numpy(X_uint8).permute(0, 3, 1, 2).float() / 255.0


def normalise(x: torch.Tensor) -> torch.Tensor:
    """[0,1] -> [-1,1] (zero-centred inputs train faster)."""
    return (x - 0.5) / 0.5


class ImageDataset(Dataset):
    def __init__(self, X_uint8, y, augment=False):
        self.x = to_tensor(X_uint8)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.aug = make_augmentation(X_uint8.shape[1]) if augment else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.aug is not None:
            x = self.aug(x).clamp(0, 1)
        return normalise(x), self.y[i]


@torch.no_grad()
def predict_scores(model, X_uint8, device, batch_size=128) -> np.ndarray:
    """Model score = sigmoid(logit), a number in 0-1 for the counterfeit class."""
    model.eval()
    out = []
    x_all = normalise(to_tensor(X_uint8))
    for i in range(0, len(x_all), batch_size):
        out.append(torch.sigmoid(model(x_all[i:i + batch_size].to(device))).cpu())
    return torch.cat(out).numpy()


def train_cnn(cfg, X, y, tr, va, seed=config.SEED, device=None, verbose=False):
    device = device or get_device()
    set_seed(seed)
    model = build_model(cfg, img_size=X.shape[1]).to(device)

    n_pos = float(y[tr].sum())
    n_neg = float(len(tr) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg.get("weight_decay", 0.0))

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(ImageDataset(X[tr], y[tr], augment=cfg["augment"]),
                              batch_size=config.BATCH_SIZE, shuffle=True, generator=gen,
                              drop_last=len(tr) > config.BATCH_SIZE)
    val_loader = DataLoader(ImageDataset(X[va], y[va], augment=False),
                            batch_size=config.BATCH_SIZE * 2, shuffle=False)

    history, best_loss, best_state, best_epoch, wait = [], float("inf"), None, 0, 0
    for epoch in range(1, config.MAX_EPOCHS + 1):
        # ---- train one epoch
        model.train()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimiser.step()
            tot_loss += loss.item() * len(yb)
            tot_correct += ((logits > 0).float() == yb).sum().item()
            tot_n += len(yb)
        # ---- validate
        model.eval()
        v_loss, v_correct, v_n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss += loss_fn(logits, yb).item() * len(yb)
                v_correct += ((logits > 0).float() == yb).sum().item()
                v_n += len(yb)
        rec = dict(epoch=epoch, train_loss=tot_loss / tot_n, train_acc=tot_correct / tot_n,
                   val_loss=v_loss / v_n, val_acc=v_correct / v_n)
        history.append(rec)
        if verbose:
            print("    epoch {epoch:3d} | train loss {train_loss:.3f} acc {train_acc:.3f} | "
                  "val loss {val_loss:.3f} acc {val_acc:.3f}".format(**rec))
        # ---- early stopping
        if rec["val_loss"] < best_loss - 1e-4:
            best_loss, best_epoch, wait = rec["val_loss"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= config.EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, history, best_epoch
