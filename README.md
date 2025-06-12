
# Retinal OCT Disease Classification using Deep Learning

This project performs multi-class classification of retinal diseases using Optical Coherence Tomography (OCT) images. The model is trained to classify images into four categories: **CNV, DME, DRUSEN, and NORMAL** using transfer learning techniques.

##  Dataset

The dataset used is the **OCT2017** retinal dataset, which contains 4 main classes:
- CNV (Choroidal Neovascularization)
- DME (Diabetic Macular Edema)
- DRUSEN
- NORMAL

The data is organized into `train/`, `test/`, and `val/` folders, each containing class-based subfolders.

##  Model Architecture

We used **MobileNetV2** (pre-trained on ImageNet) as a feature extractor. The top layers were replaced with a custom classification head:

- GlobalAveragePooling2D
- Dense(128, relu)
- Dropout(0.2)
- Dense(4, softmax)

The feature extractor is frozen during initial training.

##  Techniques Used

- `ImageDataGenerator` for real-time data augmentation
- `class_weight` handling for imbalanced classes
- Transfer Learning with `MobileNetV2`
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`

##  Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

**Best performance:**
```
Accuracy:       93%
Precision avg:  0.93
Recall avg:     0.93
F1-score avg:   0.93
```

##  Results

The model performed well across all four classes, with minor confusion between CNV and DRUSEN.

### Confusion Matrix:

|         | CNV | DME | DRUSEN | NORMAL |
|---------|-----|-----|--------|--------|
| CNV     | 217 | 7   | 18     | 0      |
| DME     | 2   | 223 | 4      | 13     |
| DRUSEN  | 10  | 7   | 221    | 4      |
| NORMAL  | 0   | 3   | 4      | 235    |

### Sample Misclassification:
Displayed misclassified images with predicted and true labels for manual analysis.

##  Libraries Use

- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Scikit-learn
- Seaborn / Matplotlib

