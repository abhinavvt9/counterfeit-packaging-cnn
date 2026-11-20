"""
Handcrafted features for the SVM baseline.

A human designs these features; the SVM only learns how to weigh them.
(The CNN, in contrast, learns its own features from pixels.)

Per image (128x128 RGB) we compute:
  1. Colour histogram in HSV space   16 bins x 3 channels  = 48   (colour scheme / ink colours)
  2. Colour moments (mean, std RGB)                         =  6   (overall colour and contrast)
  3. LBP texture histograms (2 scales)                      = 28   (print texture, fine patterns)
  4. HOG on a 64x64 grey image                              = 324  (edges, text strokes, layout/shape)
  5. Edge density (Canny)                                   =  1   (how "busy" the print is)
                                                     total  = 407 numbers per image
"""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


def colour_histogram(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    feats = []
    for ch, rng in zip(range(3), [180, 256, 256]):
        h = cv2.calcHist([hsv], [ch], None, [16], [0, rng]).flatten()
        feats.append(h / (h.sum() + 1e-8))
    return np.concatenate(feats)


def colour_moments(rgb):
    x = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    return np.concatenate([x.mean(0), x.std(0)])


def lbp_histograms(gray):
    feats = []
    for P, R in [(8, 1), (16, 2)]:
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        h, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2))
        feats.append(h / (h.sum() + 1e-8))
    return np.concatenate(feats)


def hog_features(gray):
    g = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    return hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True)


def edge_density(gray):
    return np.array([(cv2.Canny(gray, 100, 200) > 0).mean()])


def extract_features(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.concatenate([colour_histogram(rgb), colour_moments(rgb),
                           lbp_histograms(gray), hog_features(gray),
                           edge_density(gray)]).astype(np.float32)


def extract_all(X_uint8: np.ndarray) -> np.ndarray:
    return np.stack([extract_features(x) for x in X_uint8])
