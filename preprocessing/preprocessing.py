"""
Dataset indexing, duplicate grouping and image preprocessing.

Pipeline for one image
----------------------
  file on disk
    -> decode, fix EXIF rotation, convert to RGB
    -> protocol "raw":        resize (squash) to 128x128            (the naive pipeline)
    -> protocol "controlled": resize short side + centre crop to 128x128
                              -> re-encode as JPEG (quality 75)     (removes PNG-vs-JPG difference)
                              -> normalise brightness & contrast    (removes brightness difference)
    -> uint8 array 128x128x3, stored in a cache file

Normalisation to the 0-1 range (divide by 255) happens later, when the arrays
are handed to the SVM feature extractor or to the CNN.
"""
import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import config


# ============================================================ 1. indexing
def _label_from_path(path: Path):
    """Return 0/1 from the nearest parent folder whose name is a class alias."""
    for part in reversed(path.parts[:-1]):
        key = part.strip().lower()
        if key in config.LABEL_ALIASES:
            return config.LABEL_ALIASES[key]
    return None


def _open_rgb(path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)          # respect phone-camera rotation flags
    return img.convert("RGB")


def scan_dataset(data_dir) -> pd.DataFrame:
    """
    Walk the dataset folder and build one row per UNIQUE image file.

    Exact duplicates (byte-identical files, e.g. the same image in a pre-made
    train/ and test/ folder) are removed here, keeping the first copy.
    Records the file metadata used by the provenance audit.
    """
    data_dir = Path(data_dir)
    rows, seen_md5 = [], {}
    n_dupes, n_unlabelled, n_broken, n_files = 0, 0, 0, 0
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in config.IMAGE_EXTENSIONS:
            continue
        n_files += 1
        label = _label_from_path(path.relative_to(data_dir))
        if label is None:
            n_unlabelled += 1
            continue
        raw = path.read_bytes()
        md5 = hashlib.md5(raw).hexdigest()
        if md5 in seen_md5:
            n_dupes += 1
            continue
        try:
            pil = Image.open(io.BytesIO(raw))
            fmt = (pil.format or path.suffix[1:]).upper()
            rgb = ImageOps.exif_transpose(pil).convert("RGB")
        except Exception:
            n_broken += 1
            continue
        seen_md5[md5] = True
        w, h = rgb.size
        gray = np.asarray(rgb.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
        rows.append(dict(
            path=str(path), rel_path=str(path.relative_to(data_dir)),
            label=label, file_format=fmt, width=w, height=h,
            file_size_kb=len(raw) / 1024.0, mean_brightness=float(gray.mean()),
            md5=md5,
        ))
    df = pd.DataFrame(rows)
    df.attrs["cleaning"] = dict(image_files_found=n_files, unique_images=len(df), exact_duplicates_removed=n_dupes,
                                unlabelled_skipped=n_unlabelled, unreadable_skipped=n_broken)
    print(f"[scan] {len(df)} unique images | exact duplicates removed: {n_dupes} | "
          f"unlabelled skipped: {n_unlabelled} | unreadable skipped: {n_broken}")
    if len(df):
        print("[scan] class counts:", df.label.map(config.CLASS_NAMES).value_counts().to_dict())
    return df


# ============================================================ 2. near-duplicate groups
def _phash_bits(gray32: np.ndarray) -> np.ndarray:
    """64-bit perceptual hash: low-frequency DCT coefficients vs their median."""
    dct = cv2.dct(gray32.astype(np.float32))
    low = dct[:8, :8].flatten()
    return (low > np.median(low[1:])).astype(np.uint8)


def compute_groups(df: pd.DataFrame, threshold=config.PHASH_THRESHOLD) -> pd.DataFrame:
    """
    Assign a group_id so that near-duplicate images (same photo re-saved,
    re-sized or rotated by 90/180/270 degrees) share a group.

    Splitting BY GROUP later guarantees that no near-copy of a test image is
    ever in the training set (this is how data leakage is prevented).
    """
    n = len(df)
    hashes = np.zeros((4, n, 64), dtype=np.uint8)
    for i, p in enumerate(df.path):
        g = np.asarray(_open_rgb(p).convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float32)
        for r in range(4):
            hashes[r, i] = _phash_bits(np.rot90(g, r))

    # distance(i, j) = min over rotations of j of hamming(hash_i, rot_hash_j)
    base = hashes[0].astype(np.int16)
    dist = np.full((n, n), 64, dtype=np.int16)
    for r in range(4):
        other = hashes[r].astype(np.int16)
        d = (base[:, None, :] != other[None, :, :]).sum(-1).astype(np.int16)
        dist = np.minimum(dist, d)

    # union-find over all pairs below the threshold
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = np.where(np.triu(dist <= threshold, k=1))
    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    df = df.copy()
    df["group_id"] = [remap[r] for r in roots]
    df["phash"] = ["".join(map(str, h)) for h in hashes[0]]

    sizes = df.group_id.value_counts()
    print(f"[groups] {df.group_id.nunique()} groups for {n} images | "
          f"images in multi-image groups: {int(sizes[sizes > 1].sum())}")
    return df


def drop_conflicting_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Remove groups that contain BOTH labels.

    On this dataset these turned out not to be true duplicates: pHash grouping
    is transitive (A~B and B~C puts A and C together), so a few visually
    different images were chained into one group (see
    results/audit/conflicting_label_groups.png). A group must sit on one side of
    every split and have one label, so these 10 images are dropped as a
    conservative choice and reported."""
    n_labels = df.groupby("group_id").label.nunique()
    bad = n_labels[n_labels > 1].index
    if len(bad):
        print(f"[groups] removing {int(df.group_id.isin(bad).sum())} images in "
              f"{len(bad)} groups that carry BOTH labels")
    return df[~df.group_id.isin(bad)].reset_index(drop=True)


# ============================================================ 3. image preprocessing
def _resize_squash(rgb: np.ndarray, size: int) -> np.ndarray:
    interp = cv2.INTER_AREA if min(rgb.shape[:2]) >= size else cv2.INTER_LINEAR
    return cv2.resize(rgb, (size, size), interpolation=interp)


def _resize_center_crop(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = size / min(h, w)
    nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    rgb = cv2.resize(rgb, (nw, nh), interpolation=interp)
    top, left = (nh - size) // 2, (nw - size) // 2
    return rgb[top:top + size, left:left + size]


def _jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _normalise_brightness(rgb: np.ndarray) -> np.ndarray:
    """Give every image the same global brightness and contrast.
    Works on the L (lightness) channel of LAB colour space so colours are kept."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    std = L.std()
    L = (L - L.mean()) / (std if std > 1e-3 else 1.0)
    lab[..., 0] = np.clip(L * config.TARGET_L_STD + config.TARGET_L_MEAN, 0, 100)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def preprocess_array(rgb: np.ndarray, protocol: str, size=config.IMG_SIZE) -> np.ndarray:
    """rgb: HxWx3 uint8 -> size x size x 3 uint8."""
    if protocol == "raw":
        return _resize_squash(rgb, size)
    if protocol == "controlled":
        x = _resize_center_crop(rgb, size)
        x = _jpeg_roundtrip(x, config.JPEG_QUALITY)
        x = _normalise_brightness(x)
        return x
    raise ValueError(f"unknown protocol {protocol!r}")


def preprocess_file(path, protocol: str, size=config.IMG_SIZE) -> np.ndarray:
    return preprocess_array(np.asarray(_open_rgb(path)), protocol, size)


def build_cache(df: pd.DataFrame, protocol: str, size=config.IMG_SIZE) -> np.ndarray:
    """Preprocess every image once and store as cache/<protocol>_<size>.npy (N,H,W,3 uint8)."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.CACHE_DIR / f"{protocol}_{size}.npy"
    X = np.stack([preprocess_file(p, protocol, size) for p in df.path])
    np.save(out, X)
    print(f"[cache] {protocol}: {X.shape} -> {out}")
    return X


def mask_centre(X: np.ndarray, keep_border=0.15) -> np.ndarray:
    """Replace the central region with flat grey, keeping only an outer frame
    (15% of each side, about half of the image area). Used by the background probe."""
    X = X.copy()
    n, h, w, _ = X.shape
    bh, bw = int(h * keep_border), int(w * keep_border)
    X[:, bh:h - bh, bw:w - bw, :] = 128
    return X


def load_cache(protocol: str, size=config.IMG_SIZE) -> np.ndarray:
    return np.load(config.CACHE_DIR / f"{protocol}_{size}.npy")
