# Melanoma Skin Cancer Detection Using Deep Learning

This project builds a **Convolutional Neural Network (CNN)** to classify **9 different skin cancer types**, with a focus on **melanoma detection**, using dermoscopic images.  
Melanoma accounts for a large proportion of skin cancer–related deaths and early detection is critical.

The pipeline addresses **overfitting, class imbalance, and limited data** using augmentation, regularization, and careful evaluation.

---

## Problem Statement

Melanoma is a highly aggressive form of skin cancer that can be fatal if not detected early.  
Manual diagnosis using dermoscopic images is time-consuming and requires expert dermatologists.

This project aims to:
- Automatically classify skin lesion images
- Detect melanoma among multiple skin cancer classes
- Reduce diagnostic effort using deep learning

---

## Dataset

- Source: **ISIC Skin Cancer Dataset**
- Total classes: **9**
  - Actinic keratosis  
  - Basal cell carcinoma  
  - Dermatofibroma  
  - Melanoma  
  - Nevus  
  - Pigmented benign keratosis  
  - Seborrheic keratosis  
  - Squamous cell carcinoma  
  - Vascular lesion  

Images are resized to **180 × 180 × 3** and loaded using `tf.keras.preprocessing.image_dataset_from_directory`.

---

## Initial Model Architecture

A baseline CNN was built using:

- Rescaling (pixel normalization)
- 3 × Conv2D + MaxPooling blocks
- Dropout for regularization
- Fully connected dense layers
- Softmax output for multi-class classification

**Model size:** ~26.3M parameters

---

## Training Observations

- Trained for **20 epochs**
- Initial model showed **overfitting**
  - Training and validation loss diverged after ~17 epochs
- Class imbalance was observed:
  - Minority classes: *seborrheic keratosis*
  - Dominant classes: *melanoma*, *pigmented benign keratosis*

---

## Improvements Applied

### 1. Data Augmentation
To reduce overfitting:
- Random horizontal & vertical flips
- Random rotations
- Applied during training using Keras preprocessing layers

### 2. Class Imbalance Handling
Used **Augmentor** to:
- Generate **500 synthetic images per class**
- Balance the dataset across all 9 categories

### 3. Enhanced CNN Architecture
- Added **Batch Normalization**
- Increased robustness to class imbalance
- Retrained model on augmented dataset

---

## Final Results

- Improved convergence between training and validation curves
- Reduced overfitting
- More stable validation accuracy
- Better generalization across all skin cancer classes

This demonstrates the effectiveness of **data augmentation + balanced training** in medical imaging tasks.

---

## How to Run

1. Open `melanoma_skin_cancer_detection.ipynb` in **Google Colab**
2. Mount Google Drive (dataset path used in notebook)
3. Run cells top-to-bottom:
   - Data loading & visualization
   - Baseline CNN training
   - Overfitting analysis
   - Augmentation & class balancing
   - Final model training & evaluation

> ⚠️ Dataset paths may need adjustment based on your Drive structure.

---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- Augmentor
- Google Colab

---

## Notes

- This is a **research / academic project**, not a clinical diagnostic tool
- Results depend on dataset quality and augmentation strategy
- Further improvements could include:
  - Transfer learning (ResNet, EfficientNet)
  - Class-weighted loss
  - ROC-AUC and sensitivity metrics for melanoma

---

## References

- ISIC Skin Cancer Dataset
- TensorFlow Keras Documentation
- Augmentor Library

