#!/usr/bin/env python3


"""
VERY SIMPLE IMPLEMENTATION OF A MASK, JUST COVERS 
BOTTOM OF FACE AND THEN REVEALS THE REST OF THE FACE
AND TESTS, SO VERY LOW top-1 accuracy
"""
import os
import random
import argparse
from typing import List, Tuple, Optional

import cv2
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


def list_people(dataset_dir: str) -> List[str]:
    people = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    people.sort()
    return people


def list_images(person_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    imgs = [f for f in os.listdir(person_dir) if f.lower().endswith(exts)]
    imgs.sort()
    return imgs


def load_gray_resize(path: str, image_size: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def apply_synthetic_mask(img_gray: np.ndarray,
                         color: float = 0.0,
                         coverage: float = 0.45,
                         soften: int = 7) -> np.ndarray:
    """
    Apply a simple mask covering lower part of the face.
    - img_gray: HxW float32 [0..1]
    - color: gray value for mask region (e.g., 0.0 black, 0.3 dark gray)
    - coverage: fraction of image height to cover from bottom upward
    - soften: Gaussian blur kernel size to soften mask edges (odd int); 0 disables
    """
    h, w = img_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Start covering from y0 towards bottom.
    cover_h = int(h * coverage)
    y0 = h - cover_h

    # Base rectangle
    cv2.rectangle(mask, (0, y0), (w - 1, h - 1), color=255, thickness=-1)

    # Add a curved top edge using an ellipse to mimic mask curvature
    center = (w // 2, y0)
    axes = (w // 2 + 2, max(8, cover_h // 2))
    # Draw filled half-ellipse upwards from the starting line
    cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=180, color=255, thickness=-1)

    # Optional feathering
    if soften and soften % 2 == 1 and soften > 1:
        mask = cv2.GaussianBlur(mask, (soften, soften), 0)

    # Blend: where mask>0, set to color, else keep original
    masked = img_gray.copy()
    m = (mask.astype(np.float32) / 255.0)[..., None]  # HxWx1
    masked = (1 - m.squeeze()) * masked + m.squeeze() * float(color)
    return masked.astype(np.float32)


def standardize_per_image(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = float(x.mean())
    sd = float(x.std())
    if sd < eps: sd = eps
    return (x - mu) / sd


def make_finite(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0 for numerical safety."""
    x = np.asarray(x)
    x = np.where(np.isfinite(x), x, 0.0)
    return x


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def build_gallery_and_probe(
    dataset_dir: str,
    image_size: Tuple[int, int],
    max_people: Optional[int],
    seed: int,
    mask_color: float,
    mask_coverage: float,
    mask_soften: int,
    per_image_standardize: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      Xg: [Ng, D] gallery (unmasked)
      yg: [Ng] person ids
      gids: list of person names aligned with yg
      Xp: [Np, D] probe (masked)
      yp: [Np] person ids
      pids: list of person names aligned with yp
    """
    rng = random.Random(seed)
    people = list_people(dataset_dir)
    if max_people is not None:
        rng.shuffle(people)
        people = people[:max_people]
        people.sort()

    Xg, yg, gids = [], [], []
    Xp, yp, pids = [], [], []

    for person in tqdm(people, desc="Preparing data"):
        pdir = os.path.join(dataset_dir, person)
        imgs = list_images(pdir)
        if len(imgs) < 2:
            continue
        # Choose two distinct images (deterministic shuffle per person)
        rng.shuffle(imgs)
        gallery_img = imgs[0]
        probe_img = imgs[1]

        g = load_gray_resize(os.path.join(pdir, gallery_img), image_size)
        p = load_gray_resize(os.path.join(pdir, probe_img), image_size)
        p_masked = apply_synthetic_mask(p, color=mask_color, coverage=mask_coverage, soften=mask_soften)

        if per_image_standardize:
            g = standardize_per_image(g)
            p_masked = standardize_per_image(p_masked)

        Xg.append(g.flatten())
        yg.append(person)
        gids.append(person)

        Xp.append(p_masked.flatten())
        yp.append(person)
        pids.append(person)

    if not Xg or not Xp:
        raise RuntimeError("No valid (gallery, probe) pairs found. Check dataset_dir structure and contents.")

    return (np.stack(Xg, axis=0), np.array(yg), gids,
            np.stack(Xp, axis=0), np.array(yp), pids)


def fit_pca(Xg: np.ndarray, n_components: int) -> Tuple[PCA, np.ndarray]:
    pca = PCA(n_components=n_components, svd_solver="auto", whiten=False, random_state=0)
    Xg_pca = pca.fit_transform(Xg)
    return pca, Xg_pca


def euclidean_nn_predict(Xg: np.ndarray, yg: np.ndarray, Xp: np.ndarray) -> np.ndarray:
    """Nearest neighbor by Euclidean distance with float64 stability."""
    Xg64 = Xg.astype(np.float64, copy=False)
    Xp64 = Xp.astype(np.float64, copy=False)
    g2 = np.sum(Xg64 * Xg64, axis=1, keepdims=True)            # [Ng, 1]
    p2 = np.sum(Xp64 * Xp64, axis=1, keepdims=True).T          # [1, Np]
    cross = Xg64 @ Xp64.T                                      # [Ng, Np]
    d2 = g2 + p2 - 2.0 * cross                                 # [Ng, Np]
    # Numerical cleanup
    d2 = np.maximum(d2, 0.0)
    d2 = make_finite(d2)
    nn_idx = np.argmin(d2, axis=0)                             # [Np]
    return yg[nn_idx]


def cosine_nn_predict(Xg: np.ndarray, yg: np.ndarray, Xp: np.ndarray) -> np.ndarray:
    """Nearest neighbor by cosine similarity (assumes inputs are roughly L2-normalized)."""
    Xg64 = Xg.astype(np.float64, copy=False)
    Xp64 = Xp.astype(np.float64, copy=False)
    sim = Xg64 @ Xp64.T                                        # in [-1, 1] if truly normalized
    sim = make_finite(sim)
    nn_idx = np.argmax(sim, axis=0)
    return yg[nn_idx]


def evaluate_top1(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred == target).mean())


def main():
    parser = argparse.ArgumentParser(description="Face recognition with synthetic mask noise + Euclidean NN")
    parser.add_argument("--dataset_dir", type=str, default="FaceRecognitionDset/lfw_funneled")
    parser.add_argument("--image_size", type=int, nargs=2, default=[100, 100])
    parser.add_argument("--max_people", type=int, default=300, help="Limit number of identities (None for all)")
    parser.add_argument("--seed", type=int, default=42)

    # Mask parameters
    parser.add_argument("--mask_color", type=float, default=0.0, help="Gray value [0..1] for mask")
    parser.add_argument("--mask_coverage", type=float, default=0.45, help="Fraction of face height covered from bottom")
    parser.add_argument("--mask_soften", type=int, default=7, help="Odd kernel size for Gaussian blur; 0 disables")

    # Preprocessing
    parser.add_argument("--per_image_standardize", action="store_true", help="Standardize each image (z-score)")
    parser.add_argument("--l2_normalize", action="store_true", help="L2-normalize feature vectors before NN")

    # PCA (0 disables)
    parser.add_argument("--pca_components", type=int, default=0, help="0 disables PCA (default)")

    args = parser.parse_args()

    img_size = (int(args.image_size[0]), int(args.image_size[1]))

    Xg, yg, gids, Xp, yp, pids = build_gallery_and_probe(
        dataset_dir=args.dataset_dir,
        image_size=img_size,
        max_people=(None if args.max_people in (None, 0, -1) else args.max_people),
        seed=args.seed,
        mask_color=float(args.mask_color),
        mask_coverage=float(args.mask_coverage),
        mask_soften=int(args.mask_soften),
        per_image_standardize=bool(args.per_image_standardize),
    )

    # Global standardization (fit on gallery, apply to both)
    g_mean = Xg.mean(axis=0, keepdims=True)
    g_std = Xg.std(axis=0, keepdims=True)
    g_std = np.maximum(g_std, 1e-6)
    Xg_std = (Xg - g_mean) / g_std
    Xp_std = (Xp - g_mean) / g_std

    # Ensure finite values
    Xg_std = make_finite(Xg_std)
    Xp_std = make_finite(Xp_std)

    if args.pca_components and args.pca_components > 0:
        ncomp = min(args.pca_components, min(Xg_std.shape[0], Xg_std.shape[1]))
        pca, Xg_tr = fit_pca(Xg_std, n_components=ncomp)
        Xp_tr = pca.transform(Xp_std)
    else:
        Xg_tr, Xp_tr = Xg_std, Xp_std

    # Optional L2 normalization
    if args.l2_normalize:
        Xg_tr = l2_normalize_rows(Xg_tr)
        Xp_tr = l2_normalize_rows(Xp_tr)

    # Choose metric based on normalization
    if args.l2_normalize:
        pred = cosine_nn_predict(Xg_tr, yg, Xp_tr)
    else:
        pred = euclidean_nn_predict(Xg_tr, yg, Xp_tr)
    acc = evaluate_top1(pred, yp)

    if args.pca_components and args.pca_components > 0:
        print("First 10 EVR:", pca.explained_variance_ratio_[:10])
        print("Cumulative 60:", pca.explained_variance_ratio_[:60].sum())
        print(f"rows = falary samples, columns = flattened pixels after proprecessing: {Xg.shape}")
        print("Cumulative 80:", pca.explained_variance_ratio_[:80].sum())

    print("Evaluation (1-NN, Euclidean) with synthetic mask on probe")
    print(f"- Identities (used): {len(np.unique(yg))}")
    print(f"- Gallery size:      {Xg.shape[0]}")
    print(f"- Probe size:        {Xp.shape[0]}")
    if args.pca_components and args.pca_components > 0:
        print(f"- PCA components:    {Xg_tr.shape[1]}")
    else:
        print("- PCA:               disabled")
    print(f"- Per-image zscore:  {args.per_image_standardize}")
    print(f"- L2 normalize:      {args.l2_normalize}")
    print(f"- Mask: color={args.mask_color:.2f}, coverage={args.mask_coverage:.2f}, soften={args.mask_soften}")
    print(f"- Top-1 Accuracy:    {acc:.4f}")


if __name__ == "__main__":
    main()