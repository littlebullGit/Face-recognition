# Face Recognition with PCA (Dimensionality Reduction)

This document describes a PCA-based face verification system on the LFW (Labeled Faces in the Wild) dataset, implemented in [DimensionRed.ipynb](cci:7://file:///Users/derek/work/Classes/MM/FaceRecognition/DimensionRed.ipynb:0:0-0:0).

## Overview

The [DimensionRed.ipynb](cci:7://file:///Users/derek/work/Classes/MM/FaceRecognition/DimensionRed.ipynb:0:0-0:0) notebook demonstrates a baseline that:
- Uses raw pixel features with a light preprocessing pipeline
- Reduces dimensionality via PCA (retaining 95% variance)
- Matches faces using Euclidean distance in PCA space
- Optimizes a decision threshold on development data
- Evaluates on standardized LFW pair protocols

## Dataset

We use the aligned and funneled version of LFW:
- 5,749 individuals with in-the-wild images
- Standard same/different evaluation pairs
- Pre-aligned faces: `lfw_funneled`

## Implementation Details

### Core Components

1. PCABasedFaceRecognition class
   - Loads and preprocesses images
   - Extracts flattened grayscale features
   - Fits PCA on a subset of images referenced by pair files
   - Transforms features into PCA space
   - Verifies pairs via Euclidean distance and thresholding

2. Image Preprocessing Pipeline
   - Resize to 100×100
   - Convert to grayscale
   - Gaussian blur (3×3) to reduce noise
   - Normalize to [0,1]
   - Optionally per-image z-score (disabled by default to promote stability)

3. PCA Fitting
   - Collects unique images referenced in dev-train pairs (up to 800)
   - Fits PCA with n_components=0.95 (retain 95% variance)
   - No whitening (to avoid amplifying noise)
   - Randomized SVD for stability
   - Typical outcome: 800 samples, 10,000 dims → ~159 components

4. Threshold Optimization
   - Computes distances for dev-train (or dev-test) pairs
   - Grid-searches a dynamic range to maximize accuracy (or F1)

5. Evaluation Framework
   - Parses LFW pair files (same/different formats)
   - Computes Accuracy, Precision, Recall, F1
   - Reports TP/FP/TN/FN counts

## Results

### PCA Fit Summary
- Samples used to fit PCA: 800
- Original dimension: 10,000 (100×100)
- Reduced dimension: 159 (at 95% variance)

### Performance on LFW (example run)
- Accuracy: 57.0%
- Precision: 57.1%
- Recall: 56.5%
- F1-Score: 56.8%
- Notes:
  - PCA often improves recall vs raw-pixel Euclidean baseline by reducing noise and concentrating informative variance.
  - Exact metrics depend on preprocessing choices and threshold tuning.

## Usage

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt