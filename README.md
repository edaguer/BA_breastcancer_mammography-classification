# RSNA-mammography-classification

Deep learning-based classification of breast cancer in mammography images using the RSNA Screening Mammography Dataset. Comparing a custom CNN trained from scratch against fine-tuned ResNet18 with metadata analysis.

**Bachelor's Thesis | Edanur Gür | MCI Innsbruck | Medizin-, Gesundheits- und Sporttechnologie**

---

## Overview

This project implements and evaluates deep learning models for the automatic binary classification of breast cancer (malignant vs. benign) from mammography screening images. The work follows a step-by-step approach from a simple baseline CNN to transfer learning with ResNet18, extended by a multimodal model that incorporates patient age as an additional input.

The dataset used is the [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/) dataset (Kaggle), consisting of DICOM mammography images with binary cancer labels.

---

## Models

| Model | Architecture | Description |
|-------|-------------|-------------|
| **Model A** | Custom CNN | 3 convolutional layers trained from scratch, baseline |
| **Model B** | ResNet18 | Pretrained ImageNet weights, fine-tuned on mammography data |
| **Model C** | ResNet18 + Age | Multimodal: image features (512) concatenated with normalized patient age |

---

## Project Structure

```
rsna-mammography-classification/
│
├── classification_v3.ipynb   # Main notebook – full pipeline
├── requirements.txt          # Python dependencies
├── README.md
└── .gitignore
```

> **Note:** The dataset (`train_images/`, `train.csv`) is not included in this repository due to size constraints (~300GB). Download it from [Kaggle](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data).

---

## Pipeline

```
1. Imports & Configuration
2. Preprocessing (DICOM loading, normalization, cropping, resize)
3. Data loading, file check & class balancing (50/50)
4. Dataset & DataLoader (train/val split 80/20)
5. Model A – Custom CNN from scratch
6. Model B – ResNet18 Transfer Learning
7. Training & Validation
8. Evaluation (ROC curve, AUC, Confusion Matrix, F1)
9. Metadata Analysis A – Performance by age group
10. Metadata Analysis B – Age as additional model input (multimodal)
11. PCA Visualization of learned features
12. Architecture comparison
```

---

## Preprocessing

Each DICOM image goes through the following steps before being fed into the model:

1. **Inversion** – if `PhotometricInterpretation == MONOCHROME1` (white background), pixel values are inverted
2. **Normalization** – pixel values scaled to [0, 255]
3. **Cropping** – black borders removed by masking pixels > 15
4. **Resize** – scaled to 256×256 pixels
5. **Augmentation** (training only) – random horizontal flip, random rotation ±10°

---

## Class Balancing

The RSNA dataset is heavily imbalanced (~98% benign, ~2% malignant). To prevent the model from simply predicting "benign" for everything, the training data is balanced to a **50/50 split** by randomly undersampling the benign class.

---

## Evaluation Metrics

- **AUC** (Area Under the ROC Curve) – primary metric
- **ROC Curve** – visualizes tradeoff between sensitivity and specificity
- **Confusion Matrix** – true/false positives and negatives
- **Precision, Recall, F1-Score** – per class

---

## Metadata Analysis

Beyond image classification, the project investigates whether patient metadata influences model performance:

- **Age distribution** – histograms and boxplots by diagnosis
- **Cancer prevalence by age group** – `<40`, `40–50`, `50–60`, `60–70`, `>70`
- **AUC per age group** – does the model perform equally well across all ages?
- **Multimodal model** – patient age concatenated with image features before final classification

---

## Requirements

See `requirements.txt`. Main dependencies:

- Python 3.10
- PyTorch
- torchvision
- pydicom
- opencv-python
- scikit-learn
- pandas, numpy, matplotlib, seaborn

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data) and place it in the project folder:
```
train_images/
train.csv
```

2. Open `classification_v3.ipynb` in Jupyter Lab

3. Run all cells from top to bottom

---

## Supervisor

Martin Nocker, MSc — MCI Innsbruck, Teaching & Research Assistant
