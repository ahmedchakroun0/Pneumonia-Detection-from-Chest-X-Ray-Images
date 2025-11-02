# 🫁 Pneumonia Detection from Chest X-Ray Images

A deep learning project for automated pneumonia detection using transfer learning with ResNet34. This project addresses class imbalance through custom balanced sampling and creates a proper train/validation split from the original inadequate dataset structure.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset Challenges](#dataset-challenges)
- [Key Technical Components](#key-technical-components)
  - [Custom Dataset Class](#1-custom-dataset-class)
  - [Balanced Sampling Method](#2-balanced-sampling-method)
- [Model Architecture](#model-architecture)
- [Results](#results)

---

## 🎯 Overview

This project implements a CNN for binary classification of chest X-ray images (NORMAL vs PNEUMONIA). Key features:

- ✅ Custom 70/30 train/validation split (fixing original 16-image validation set)
- ✅ Weighted random sampling for balanced class representation
- ✅ Transfer learning with ResNet34 pre-trained on ImageNet
- ✅ Comprehensive data augmentation and regularization
- ✅ Achieved 89.4% test accuracy with balanced per-class performance

**Medical Context:**
Pneumonia is a serious respiratory infection requiring timely diagnosis. This model assists radiologists by providing automated preliminary analysis of chest X-rays, potentially reducing interpretation time and catching cases that might be overlooked.

---

## 📊 Dataset Challenges

### Original Kaggle Dataset Structure

**Training Set:**
- NORMAL: 1,341 images (25.7%)
- PNEUMONIA: 3,875 images (74.3%)
- **Problem 1:** Severe class imbalance (3:1 ratio)

**Validation Set:**
- NORMAL: 8 images
- PNEUMONIA: 8 images
- **Problem 2:** Only 16 total validation images!

**Test Set:**
- NORMAL: 234 images
- PNEUMONIA: 390 images
- Kept separate for final evaluation

### Why These Are Critical Problems

#### Problem 1: Class Imbalance

Without proper handling, a model trained on this imbalanced data will:
- Develop strong bias toward predicting PNEUMONIA
- Achieve ~75% accuracy by always predicting the majority class
- Have poor performance on NORMAL cases (the minority class)
- Make unreliable clinical predictions

**Real Impact:** A model that always predicts pneumonia would miss zero pneumonia cases but incorrectly diagnose all healthy patients—completely useless clinically.

#### Problem 2: Tiny Validation Set

With only 16 validation images:
- A single misclassified image = 12.5% accuracy swing
- Metrics fluctuate wildly between epochs
- Impossible to distinguish real improvements from random noise
- Cannot reliably select best model during training
- Unrepresentative of real-world data diversity

**Example:** Model A gets 13/16 correct (81.25%), Model B gets 14/16 correct (87.5%). Is Model B truly better or just lucky on those specific 16 images?

### Our Solution

**Step 1:** Combine original train + val = 5,232 images total

**Step 2:** Create new 70/30 split using fixed random seed
- Training: 3,662 images (70%)
- Validation: 1,570 images (30%)
- Test: 624 images (unchanged)

**Benefits:**
- Validation set is now 98× larger (1,570 vs 16 images)
- Stable, reliable validation metrics
- Better representation of data diversity
- Reproducible splits for fair model comparisons

---

## 🔧 Key Technical Components

### 1. Custom Dataset Class

#### Why We Need It

PyTorch's standard ImageFolder loader cannot:
- Create custom splits from combined directories
- Apply different transforms to training vs validation
- Handle our reproducible splitting strategy with fixed seeds
- Give us fine-grained control over which images go where

We need a custom solution that:
- Loads images from a unified list of paths
- Uses index lists to define train/val membership
- Applies appropriate transforms per split
- Maintains reproducibility

#### How It Works

**Architecture:**

The custom dataset class has three core components:

**1. Initialization**
- Receives: Complete list of (image_path, label) tuples
- Receives: List of indices defining which samples belong to this split
- Receives: Transform pipeline to apply when loading images
- Stores these for later use

**2. Length Method**
- Returns: Number of samples in this split
- Used by PyTorch to know when to stop iterating

**3. Get Item Method**
- Input: Index from 0 to (length - 1)
- Process: Maps to actual sample index → loads image from disk → applies transforms
- Output: Preprocessed image tensor and label

#### Step-by-Step Process

**Phase 1: Discovery**

Use ImageFolder to scan directories and build lists:
- Train folder: 5,216 samples as (path, label) tuples
- Val folder: 16 samples as (path, label) tuples
- Combined list: 5,232 total samples

**Phase 2: Shuffling**

Generate random permutation with fixed seed:
- Create random ordering of indices 0 to 5,231
- Fixed seed (42) ensures reproducibility
- Example: [3421, 1456, 892, 4231, 0, 2890, ...]

**Phase 3: Splitting**

Divide shuffled indices:
- First 70% (indices 0-3,661) → training indices
- Last 30% (indices 3,662-5,231) → validation indices

**Phase 4: Dataset Creation**

Create two dataset instances:
- Training dataset: Uses training indices + heavy augmentation transforms
- Validation dataset: Uses validation indices + minimal preprocessing only

Both reference the same underlying image list but access different subsets with different preprocessing.

#### Example Workflow

**When training loop requests an image:**

1. Calls training_dataset[0]
2. Dataset maps 0 → training_indices[0] → e.g., 3421
3. Looks up all_samples[3421] → ("/path/to/image.jpg", 1)
4. Loads image from disk
5. Applies augmentation transforms (flip, rotate, color jitter, etc.)
6. Returns (augmented_tensor, 1)

**When validation loop requests an image:**

1. Calls validation_dataset[0]
2. Dataset maps 0 → validation_indices[0] → e.g., 892
3. Looks up all_samples[892] → ("/path/to/different_image.jpg", 0)
4. Loads image from disk
5. Applies minimal transforms (resize, normalize only)
6. Returns (clean_tensor, 0)

#### Key Benefits

✅ **Flexibility:** Easy to change split ratios or implement k-fold cross-validation
✅ **Efficiency:** No data duplication—images stay in original locations
✅ **Reproducibility:** Fixed seed guarantees identical splits across runs
✅ **Control:** Different transforms for training (augmentation) vs validation (clean)
✅ **Simplicity:** Clean separation between data organization and data loading

---

### 2. Balanced Sampling Method

#### The Core Problem

**Natural Distribution:**
- NORMAL: 950 training images (26%)
- PNEUMONIA: 2,712 training images (74%)

**Standard Random Sampling:**

With uniform random sampling, each image has equal selection probability. A typical batch of 32 images contains:
- ~8 NORMAL images (25%)
- ~24 PNEUMONIA images (75%)

**The Bias This Creates:**

Over an entire epoch, the model:
- Sees PNEUMONIA examples 3× more often
- Updates weights based primarily on pneumonia patterns
- Learns to default to predicting pneumonia when uncertain
- Develops poor recognition of normal chest X-ray features

**Clinical Impact:**

Without balanced sampling, typical model performance:
- NORMAL recall: 45-60% (misses many healthy patients)
- PNEUMONIA recall: 90-95% (catches most sick patients)
- Overall accuracy: ~78% (misleadingly decent)
- **Problem:** Unreliable for ruling out pneumonia

#### The Weighted Sampling Solution

**Core Principle:**

Assign each image a sampling weight **inversely proportional** to its class frequency. Rare classes get high weights, common classes get low weights.

**Mathematical Foundation:**

For each class, calculate inverse frequency weight:

- NORMAL class has 950 samples → weight = 1/950 = 0.00105
- PNEUMONIA class has 2,712 samples → weight = 1/2,712 = 0.00037

**Key Observation:** NORMAL weight is 2.85× higher than PNEUMONIA weight

**Sample-Level Weights:**

Each image inherits its class weight:
- Every NORMAL image gets sampling weight: 0.00105
- Every PNEUMONIA image gets sampling weight: 0.00037

**Selection Probability:**

When creating a batch, probability of selecting a specific image:

P(selecting image) = (image weight) / (sum of all weights)

After normalization:
- P(selecting any NORMAL image) ≈ 50%
- P(selecting any PNEUMONIA image) ≈ 50%

**Remarkable Result:** Equal representation despite 3:1 imbalance!

#### How Sampling Works in Practice

**Creating the Sampler:**

**Step 1: Extract Labels**
For each training index, look up its label (0=NORMAL, 1=PNEUMONIA)
Result: [1, 0, 1, 1, 0, 1, 0, 1, ...] for 3,662 samples

**Step 2: Count Class Frequencies**
Count occurrences: [950 NORMAL, 2,712 PNEUMONIA]

**Step 3: Calculate Inverse Weights**
Compute 1/count for each class: [0.00105, 0.00037]

**Step 4: Assign Sample Weights**
Map each training sample to its class weight:
- Sample 0 (PNEUMONIA) → 0.00037
- Sample 1 (NORMAL) → 0.00105
- Sample 2 (PNEUMONIA) → 0.00037
- ... and so on for all 3,662 samples

**Step 5: Configure Sampler**
Create PyTorch WeightedRandomSampler with:
- Sample weights list (3,662 weights)
- Number of samples per epoch: 3,662 (same as dataset size)
- Replacement: True (allow resampling same image)

#### Why Replacement is Essential

**With Replacement:**
- Can sample the same image multiple times in one epoch
- Allows oversampling minority class to match majority class
- NORMAL images appear ~2.85× per epoch on average
- PNEUMONIA images appear ~1.0× per epoch on average

**Without Replacement:**
- Could only sample each image once
- Would run out of NORMAL images before creating balanced batches
- Cannot achieve equal representation

**Analogy:** Like drawing colored balls from a bag. With replacement, after drawing a rare color, you put it back so it can be drawn again. This lets rare colors appear as often as common ones.

#### Practical Training Impact

**Batch Composition Comparison:**

**Without Balanced Sampling:**
- Batch 1: [P, P, P, N, P, P, P, P, P, P, P, P, P, N, P, P, ...]
- Batch 2: [P, N, P, P, P, P, P, P, P, P, P, N, P, P, P, P, ...]
- Pattern: ~75% pneumonia, ~25% normal

**With Balanced Sampling:**
- Batch 1: [N, P, N, P, P, N, N, P, N, P, P, N, P, N, P, P, ...]
- Batch 2: [P, N, P, N, N, P, P, N, N, P, N, P, N, P, P, N, ...]
- Pattern: ~50% pneumonia, ~50% normal

**Training Dynamics:**

Every batch provides balanced learning signal:
- Gradient updates consider both classes equally
- Model learns robust features for both classes
- Prevents bias toward majority class
- Leads to balanced predictions

#### Important Usage Notes

**Only for Training:**
Balanced sampling is ONLY used during training. Validation and test sets use normal sequential sampling to evaluate on realistic class distributions.

**Complements Class Weights:**
Balanced sampling (equal data presentation) + weighted loss (equal loss contribution) = comprehensive imbalance handling

**Doesn't Create New Data:**
We're not generating synthetic images. We're just changing sampling frequency of existing images.

#### Performance Comparison

| Metric | Without Balancing | With Balancing |
|--------|-------------------|----------------|
| NORMAL Recall | 45-60% | 80-90% |
| PNEUMONIA Recall | 90-95% | 85-92% |
| Overall Accuracy | 78-82% | 85-90% |
| Clinical Utility | Poor | Good |

**Key Insight:** Small decrease in pneumonia recall is acceptable trade-off for large increase in normal recall, resulting in a balanced, clinically useful model.

---

## 🏗️ Model Architecture

```
┌─────────────────────────────────────────────────────┐
│              INPUT: Chest X-Ray (224×224×3)         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                   RESNET34 BACKBONE                  │
│                  (Pre-trained on ImageNet)           │
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │  Conv1 + BatchNorm + ReLU + MaxPool        │    │
│  │  Output: 64 channels, 56×56               │    │
│  │  Status: FROZEN ❄️                         │    │
│  └────────────────────────────────────────────┘    │
│                       │                              │
│                       ▼                              │
│  ┌────────────────────────────────────────────┐    │
│  │  Layer 1: 3 Residual Blocks                │    │
│  │  Features: Edges, textures, basic shapes   │    │
│  │  Output: 64 channels, 56×56               │    │
│  │  Status: FROZEN ❄️                         │    │
│  └────────────────────────────────────────────┘    │
│                       │                              │
│                       ▼                              │
│  ┌────────────────────────────────────────────┐    │
│  │  Layer 2: 4 Residual Blocks                │    │
│  │  Features: Tissue boundaries, patterns     │    │
│  │  Output: 128 channels, 28×28              │    │
│  │  Status: FROZEN ❄️                         │    │
│  └────────────────────────────────────────────┘    │
│                       │                              │
│                       ▼                              │
│  ┌────────────────────────────────────────────┐    │
│  │  Layer 3: 6 Residual Blocks                │    │
│  │  Features: Lung structures, rib patterns   │    │
│  │  Output: 256 channels, 14×14              │    │
│  │  Status: FROZEN ❄️                         │    │
│  └────────────────────────────────────────────┘    │
│                       │                              │
│                       ▼                              │
│  ┌────────────────────────────────────────────┐    │
│  │  Layer 4: 3 Residual Blocks                │    │
│  │  Features: Disease-specific patterns       │    │
│  │  Output: 512 channels, 7×7                │    │
│  │  Status: FINE-TUNED 🔥                     │    │
│  └────────────────────────────────────────────┘    │
│                       │                              │
│                       ▼                              │
│  ┌────────────────────────────────────────────┐    │
│  │  Global Average Pooling                     │    │
│  │  Output: 512-dimensional feature vector    │    │
│  └────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│            CUSTOM CLASSIFICATION HEAD                │
│                (Fully Trainable 🔥)                  │
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │  Dropout (p=0.5)                            │    │
│  │  Drop 50% of features randomly             │    │
│  └────────────────────────────────────────────┘    │
│                       │                              │
│                       ▼                              │
│  ┌────────────────────────────────────────────┐    │
│  │  Linear Layer (512 → 256)                  │    │
│  │  First dense layer                         │    │
│  └────────────────────────────────────────────┘    │
│                       │                              │
│                       ▼                              │
│  ┌────────────────────────────────────────────┐    │
│  │  ReLU Activation                            │    │
│  │  Non-linearity                              │    │
│  └────────────────────────────────────────────┘    │
│                       │                              │
│                       ▼                              │
│  ┌────────────────────────────────────────────┐    │
│  │  Dropout (p=0.25)                           │    │
│  │  Drop 25% of features randomly             │    │
│  └────────────────────────────────────────────┘    │
│                       │                              │
│                       ▼                              │
│  ┌────────────────────────────────────────────┐    │
│  │  Linear Layer (256 → 2)                    │    │
│  │  Output layer: [NORMAL, PNEUMONIA]        │    │
│  └────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│    OUTPUT: Class Probabilities + Prediction         │
│    • NORMAL probability                              │
│    • PNEUMONIA probability                           │
│    • Predicted class (max probability)              │
│    • Confidence score                                │
└─────────────────────────────────────────────────────┘
```

### Key Architecture Decisions

**ResNet34 Choice:**
- 21M parameters total
- Good balance between capacity and efficiency
- Less overfitting risk than ResNet50
- Better performance than ResNet18

**Transfer Learning Strategy:**
- Frozen layers (1-3): 88.7% of parameters - generic features
- Fine-tuned layer4: 11.3% of parameters - X-ray specific features
- Custom classifier: Task-specific decision making

**Regularization:**
- Heavy dropout (0.5, 0.25) prevents overfitting
- Weight decay (L2 regularization) encourages simpler models
- Label smoothing prevents overconfident predictions

---

## 📊 Results

### Test Set Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **89.4%** |
| **F1 Score** | **0.89** |
| Precision | 0.89 |
| Recall | 0.89 |

### Per-Class Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NORMAL | 0.87 | 0.82 | 0.84 | 234 |
| PNEUMONIA | 0.91 | 0.93 | 0.92 | 390 |

### Confusion Matrix

```
                    Predicted
                NORMAL  PNEUMONIA
           ┌─────────┬──────────┐
     NORMAL│   192   │    42    │  82% correct
           │ (True-) │ (False+) │  18% false positives
    Actual ├─────────┼──────────┤
 PNEUMONIA │    27   │   363    │  93% correct
           │ (False-)│ (True+)  │  7% false negatives
           └─────────┴──────────┘
```

### Clinical Interpretation

✅ **High pneumonia recall (93%)**: Catches most cases - critical for patient safety
✅ **Good normal recall (82%)**: Identifies most healthy patients
⚠️ **18% false positive rate**: Some healthy patients flagged - acceptable for screening
✅ **Balanced performance**: Not biased toward either class

### Training Details

- **Training time:** ~30 minutes on GPU
- **Best epoch:** 18 out of 50
- **Early stopping:** Triggered at epoch 30 (patience=12)
- **Final validation loss:** 0.2876
- **Model saved:** best_pneumonia_model.pth

---

## 🚀 Quick Start

### Installation

Requires: Python 3.7+, PyTorch, torchvision, numpy, matplotlib, seaborn, scikit-learn

### Dataset Setup

Download from Kaggle: "Chest X-Ray Images (Pneumonia)"
Extract to maintain directory structure with train/val/test folders

### Training

Run the provided notebook cells sequentially. The code automatically:
1. Creates 70/30 train/val split
2. Applies balanced sampling
3. Trains with early stopping
4. Saves best model
5. Generates evaluation metrics and visualizations

### Inference

Run all the notebook cells sequentially to train the model.

---

## 📚 Key Takeaways

1. **Class imbalance** requires both balanced sampling AND weighted loss
2. **Small validation sets** make training unreliable - always check dataset quality
3. **Transfer learning** is essential for medical imaging with limited data
4. **Aggressive regularization** (dropout, augmentation, weight decay) prevents overfitting
5. **Multiple metrics** (precision, recall, F1) are crucial for medical applications

---

## 🔮 Future Work

- Ensemble multiple models for improved reliability
- Add attention mechanisms to visualize decision regions
- Expand to multi-class (bacterial vs viral pneumonia)
- Deploy as web application for clinical testing
- Validate on external datasets from different hospitals

---

## 📖 References

- **Dataset:** Kermany et al. (2018) - Medical image diagnosis by deep learning
- **Architecture:** He et al. (2016) - Deep Residual Learning for Image Recognition  
- **Methods:** Transfer learning and class imbalance handling techniques from medical AI literature