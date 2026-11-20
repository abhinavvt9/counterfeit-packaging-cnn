"""
Grad-CAM: which image regions pushed the CNN towards its decision?

  1. run the image forward, keep the feature maps A_k of the LAST conv layer
  2. backpropagate the output logit to get dLogit/dA_k
  3. weight of each map  w_k = average of its gradient over all positions
  4. heatmap = ReLU( sum_k w_k * A_k ), upsampled to the image size

Red regions = evidence FOR "counterfeit". If the heatmap lands on the
background or image borders instead of the packaging, the model is likely
using a shortcut.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

from cnn.train import normalise, to_tensor


def _last_conv(model):
    convs = [m for m in model.features if isinstance(m, nn.Conv2d)]
    return convs[-1]


def gradcam(model, img_uint8: np.ndarray, device):
    """img_uint8: HxWx3. Returns (heatmap HxW in 0-1, score for counterfeit)."""
    model.eval()
    for m in model.modules():                 # in-place ReLU would overwrite the stored maps
        if isinstance(m, nn.ReLU):
            m.inplace = False
    store = {}

    def fwd_hook(module, inp, out):
        store["act"] = out.detach()
        out.register_hook(lambda g: store.__setitem__("grad", g.detach()))

    handle = _last_conv(model).register_forward_hook(fwd_hook)
    try:
        x = normalise(to_tensor(img_uint8[None])).to(device)
        logit = model(x)
        model.zero_grad()
        logit.sum().backward()
        act, grad = store["act"][0], store["grad"][0]          # (C,h,w)
        weights = grad.mean(dim=(1, 2))                          # (C,)
        cam = torch.relu((weights[:, None, None] * act).sum(0)).cpu().numpy()
    finally:
        handle.remove()
    cam = cv2.resize(cam, (img_uint8.shape[1], img_uint8.shape[0]))
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam, float(torch.sigmoid(logit).item())


def overlay(img_uint8, cam, alpha=0.45):
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * img_uint8 + alpha * heat)


def border_energy_share(cam, border_frac=0.15):
    """Share of heatmap mass in the outer 15% frame of the image
    (the frame covers ~51% of the area, so ~0.51 means no preference)."""
    h, w = cam.shape
    bh, bw = int(h * border_frac), int(w * border_frac)
    inner = cam[bh:h - bh, bw:w - bw].sum()
    total = cam.sum() + 1e-8
    return float(1 - inner / total)
