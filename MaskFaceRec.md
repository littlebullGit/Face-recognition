# MaskFaceRec: Euclidean Baseline with Synthetic Mask Noise

This document explains the full pipeline implemented in `MaskFaceRec.py`: how the data is prepared, how the synthetic mask is applied, what preprocessing is performed, and how nearest-neighbor matching is evaluated. The goal is to measure how much a lower-face mask degrades a simple face recognition baseline.

## Overview

- Build a gallery of unmasked reference images and a probe set of masked images.
- Preprocess images (grayscale, resize, optional per-image z-score, global standardization).
- Optionally reduce dimensionality with PCA.
- Optionally L2-normalize features.
- Perform 1-NN matching with Euclidean distance or cosine similarity.
- Report Top-1 accuracy across identities.

## Dataset structure

Expected layout (LFW-style):

```
FaceRecognitionDset/lfw_funneled/
  person_1/
    img_0001.jpg
    img_0002.jpg
    ...
  person_2/
    img_0001.jpg
    img_0002.jpg
    ...
  ...
```

Only identities with at least 2 images are used (1 for gallery, 1 for probe).

## Key functions and components

### `list_people(dataset_dir)` and `list_images(person_dir)`
- Enumerate people folders and image files.
- Images allowed: `.jpg`, `.jpeg`, `.png`, `.bmp`.

### `load_gray_resize(path, image_size)`
- Reads BGR with OpenCV, converts to grayscale, resizes to `image_size` (e.g. `100x100`), and scales to `[0, 1]` float32.

### `apply_synthetic_mask(img_gray, color=0.0, coverage=0.45, soften=7)`
- Creates a synthetic “face mask” covering the lower portion of the image to simulate occlusion.
- Steps:
  - Compute a binary mask that covers the bottom `coverage` fraction of the image height.
  - Draw a rectangle and a top half-ellipse to get a curved upper boundary.
  - Optionally blur the mask edges with Gaussian blur (`soften` must be odd; `0` disables).
  - Blend by replacing masked pixels with a uniform gray `color` (default `0.0` = black).
- Returns a masked grayscale image in `[0,1]` float32.

### `standardize_per_image(x)`
- Optional per-image z-score standardization: `(x - mean) / std` with epsilon guard.
- Makes each image locally standardized; can help with illumination variation.

### `build_gallery_and_probe(...)`
- For each person with ≥2 images:
  - Select one image for the gallery (unmasked) and another for the probe (masked).
  - Load, optionally per-image standardize, then flatten to 1D features.
- Returns:
  - `Xg` `[Ng, D]`: gallery features.
  - `yg` `[Ng]`: gallery labels (person names).
  - `Xp` `[Np, D]`: probe features.
  - `yp` `[Np]`: probe labels.

### Global standardization (in `main()`)
- Compute mean and std from the gallery features only.
- Apply `(X - mean) / std` to both gallery and probe features.
- Replace any non-finite values with `0` (safety guard).

### PCA (optional)
- If `--pca_components > 0`, fit PCA on standardized gallery and transform both gallery and probe.
- Useful to reduce dimensionality; for this masked baseline, improvements are modest.

### L2 normalization (optional)
- If `--l2_normalize` is set, L2-normalize each feature vector.
- When L2 normalization is enabled, cosine similarity is used for matching. Otherwise, Euclidean distance is used.

### Nearest neighbor matching
- Euclidean path (`euclidean_nn_predict`):
  - Compute squared distances via `||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b` in float64.
  - Clamp small negative values to 0 and remove NaN/Inf.
- Cosine path (`cosine_nn_predict`):
  - Compute `sim = Xg @ Xp^T` in float64 after L2 normalization.
  - Select nearest neighbor by maximum similarity.

### Metrics
- `evaluate_top1(pred, target)`: Top-1 accuracy across probes, i.e., fraction of probes where the nearest gallery identity matches the true identity.

## CLI usage

The script provides a CLI to control the entire experiment. Examples:

- Baseline without PCA, with z-score and L2 (recommended for stability):

```
python MaskFaceRec.py --dataset_dir FaceRecognitionDset/lfw_funneled \
  --image_size 100 100 \
  --max_people 300 \
  --per_image_standardize \
  --l2_normalize \
  --pca_components 0
```

- With PCA (e.g., 60 components):

```
python MaskFaceRec.py --dataset_dir FaceRecognitionDset/lfw_funneled \
  --image_size 100 100 \
  --max_people 300 \
  --per_image_standardize \
  --l2_normalize \
  --pca_components 60
```

- Adjust mask strength:

```
python MaskFaceRec.py --dataset_dir FaceRecognitionDset/lfw_funneled \
  --mask_color 0.0 --mask_coverage 0.45 --mask_soften 7
```

### Arguments reference
- `--dataset_dir`: Path to LFW-like dataset root.
- `--image_size W H`: Resize target.
- `--max_people`: Limit number of identities (e.g., 300). Use all if omitted or set to negative/zero.
- `--seed`: Random seed for repeatable sampling of images per identity.
- `--mask_color`: Gray value in `[0,1]` for the synthetic mask fill.
- `--mask_coverage`: Fraction of height to cover from the bottom.
- `--mask_soften`: Gaussian blur kernel (odd int; `0` disables).
- `--per_image_standardize`: Use per-image z-score before flattening.
- `--pca_components`: Components for PCA; `0` disables (default).
- `--l2_normalize`: L2-normalize features and use cosine similarity.

## Expected results and interpretation
- Heavy lower-face occlusion substantially degrades simple appearance-based features. Top-1 accuracy may be low (e.g., ~3–5%) depending on `coverage`, preprocessing, and number of people.
- `--per_image_standardize` + `--l2_normalize` improves stability and modestly improves accuracy but remains a tough setting.
- PCA can help or hurt slightly; try 30–200 components after enabling standardization and L2.

## Troubleshooting and numerical stability
- The script uses float64 and sanitizes NaN/Inf when computing distances/similarities.
- If you still see warnings:
  - Keep `--pca_components 0` to avoid PCA’s internal randomized SVD creating large intermediate values.
  - Ensure `--per_image_standardize` and L2 normalization are enabled.
  - Reduce `--max_people` or the mask severity (`--mask_coverage`).

## Possible extensions
- Use more than one gallery image per identity and average features.
- Crop or emphasize upper-face regions for both gallery and probe.
- Replace raw pixels with hand-crafted features (e.g., HOG) or learned embeddings (e.g., `CNNFaceRec.py`).

## File references
- Script: `MaskFaceRec.py`
  - Core functions: `apply_synthetic_mask()`, `build_gallery_and_probe()`, `euclidean_nn_predict()`, `cosine_nn_predict()`.
  - Entry point: `main()`.
