# Face Recognition using Euclidean Distance

This repository contains a simple face recognition system implemented using raw pixel values and Euclidean distance metrics on the LFW (Labeled Faces in the Wild) dataset.

## Overview

The `FaceRecEuclidian.ipynb` notebook demonstrates a baseline approach to face recognition that:
- Uses raw pixel intensities as features
- Applies Euclidean distance as a similarity metric
- Optimizes decision thresholds using validation data
- Evaluates performance on standardized test pairs

## Dataset

The implementation uses the **LFW (Labeled Faces in the Wild)** dataset:
- 5,749 different individuals
- Face images in natural, uncontrolled conditions
- Standardized evaluation protocol with same/different person pairs
- Images are preprocessed and aligned (lfw_funneled version)

## Implementation Details

### Core Components

1. **EuclideanFaceRecognition Class**
   - Handles image loading and preprocessing
   - Extracts normalized pixel features
   - Computes Euclidean distances between feature vectors
   - Performs pair verification based on threshold

2. **Image Preprocessing Pipeline**
   - Resize to 100×100 pixels (reduces noise and computation)
   - Convert to grayscale
   - Apply Gaussian blur (σ=3) to reduce noise
   - Normalize to [0,1] range
   - Apply z-score normalization (mean=0, std=1)

3. **Threshold Optimization**
   - Collects distance distributions for same/different person pairs
   - Performs grid search over dynamically determined range
   - Maximizes classification accuracy on validation data

4. **Evaluation Framework**
   - Processes LFW pair files (handles multiple formats)
   - Computes standard classification metrics
   - Provides detailed performance analysis

## Results

### Performance on LFW Test Set
- **Accuracy**: 61.0%
- **Precision**: 72.9% 
- **Recall**: 35.0%
- **F1-Score**: 47.3%

### Performance Characteristics
- **Conservative classifier**: High precision, lower recall
- **Good specificity**: 87% true negative rate (correctly identifies different persons)
- **Challenge with variations**: Struggles with lighting, pose, and expression changes

## Usage

### Prerequisites
Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Notebook
1. Extract the LFW dataset (`lfw-funneled.tgz`)
2. Run cells sequentially in `FaceRecEuclidian.ipynb`
3. The notebook will automatically:
   - Initialize the face recognition system
   - Find optimal threshold using development data
   - Evaluate on test sets
   - Display performance metrics and examples

### Key Functions

```python
# Initialize system
recognizer = EuclideanFaceRecognition()

# Find optimal threshold
threshold = recognizer.find_optimal_threshold("pairsDevTrain.txt")

# Evaluate performance  
accuracy, precision, recall, f1 = evaluate_face_recognition(
    recognizer, "pairs.txt", threshold
)

# Verify individual pairs
is_same = recognizer.verify_pair("Person1", 1, "Person2", 1, threshold)
```

## Technical Insights

### Why Euclidean Distance Works (Partially)
- Captures overall pixel-level similarity
- Simple and interpretable baseline
- No training required

### Limitations of Raw Pixel Approach
- **Sensitive to variations**: Lighting, pose, expression changes create large pixel differences
- **No invariance**: Cannot handle transformations that preserve identity
- **High dimensionality**: 10,000 features (100×100) with limited data
- **Distance paradox**: Same person pairs often have higher distances than different person pairs

### Example Distance Analysis
```
Same person (George_W_Bush): Distance = 122.061 → Classified as Different ❌
Different persons (Bush vs Blair): Distance = 110.591 → Classified as Different ✅
```

This illustrates the fundamental challenge: intra-person variation can exceed inter-person variation in pixel space.

## Educational Value

This implementation serves as an important baseline demonstrating:
1. **Feature extraction** from raw image data
2. **Distance-based classification** principles  
3. **Threshold optimization** techniques
4. **Evaluation methodology** for biometric systems
5. **Limitations of naive approaches** in computer vision

The modest performance (61% accuracy) highlights why modern face recognition systems use learned representations (deep learning embeddings) that are invariant to pose, lighting, and expression changes.

## File Structure

```
FaceRecognitionDset/
├── lfw-funneled.tgz          # LFW dataset archive
├── pairs.txt                 # Main evaluation pairs  
├── pairsDevTest.txt          # Development test pairs
├── pairsDevTrain.txt         # Development training pairs
└── lfw_funneled/             # Extracted face images
    ├── Person_Name/
    │   ├── Person_Name_0001.jpg
    │   └── ...
    └── ...

FaceRecEuclidian.ipynb        # Main implementation notebook
requirements.txt              # Python dependencies
README.md                     # This documentation
```

## Future Improvements

To enhance performance, consider:
- **Feature engineering**: HOG, LBP, or other descriptors
- **Dimensionality reduction**: PCA or other techniques
- **Distance metrics**: Cosine similarity, Mahalanobis distance
- **Deep learning**: CNN-based embeddings (FaceNet, ArcFace)
- **Data augmentation**: Increase training data diversity

This baseline implementation provides a foundation for understanding face recognition principles before advancing to more sophisticated approaches.

