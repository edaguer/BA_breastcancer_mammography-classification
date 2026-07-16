# Evaluation of Deep Learning Models for Breast Cancer Classification in Screening Mammography

Bachelor thesis by Edanur Gür, MCI Innsbruck.
Supervisor: Martin Nocker, MSc, MCI Innsbruck.

## About the project

This project studies the automated classification of breast cancer in mammography
images using deep learning. Based on the
[RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
dataset (Kaggle), three model architectures are compared:

- Model A: a convolutional neural network trained from scratch (baseline)
- Model B: ResNet18 with transfer learning and fine tuning
- Model C: a multimodal model that adds the patient age to the image

On top of that, a series of systematic experiments (E0 to E7) changes one factor
at a time (training data size, image resolution, fine tuning depth, learning rate)
to see what actually drives the classification performance. The experiments include
a statistical check across several random seeds and a Grad-CAM analysis for
interpretability.

The whole pipeline lives in the notebook `classification_v3.ipynb`.

## Research question

> Which factors determine the performance of deep learning models for breast cancer
> classification in mammography images, and how can this performance be improved
> through targeted methodological choices?

## Method

**Preprocessing.** Each DICOM is read with pydicom, converted with the VOI-LUT
(the same windowing a radiologist would see), inverted if needed, cropped to the
breast area, padded to a square so the aspect ratio is preserved, and resized to
384x384. Processed images are cached to disk so later notebook runs are fast.

**Split.** The data is split 80/20 by patient, not by image, so that the multiple
views of one patient (left/right, CC/MLO) never end up in both training and
validation. Experiment E6 shows what happens without this: an image-level split
inflates the AUC. For the final honest evaluation, a separate 70/15/15 split by
patient adds a test set that is never used for training or for picking the best
epoch (see "Held-out test set" below).

**Models.** Three architectures are compared: a small CNN trained from scratch
(Model A), a ResNet18 with transfer learning and fine tuning (Model B), and a
multimodal variant that adds patient age to the ResNet18 features (Model C).

**Ablations (E0 to E7).** Starting from a working baseline, one factor is changed
at a time: fine tuning depth, learning rate, image resolution, class balancing
strategy, and training set size. An overfit test (E0) confirms the pipeline can
learn at all, and repeated training across multiple seeds (E5) gives a 95 percent
confidence interval instead of relying on a single run.

**Evaluation and interpretability.** AUC is used as the main metric since it does
not depend on a classification threshold, which matters given the class imbalance.
The final model is also evaluated as a concrete yes/no classification (confusion
matrix, precision, recall). Grad-CAM highlights which image regions drove each
prediction, but since the dataset only has image-level labels and no ground-truth
bounding boxes, this is an interpretability tool, not a validated tumor
localization.

**Held-out test set.** In the main experiments the validation set does two jobs at
once: it picks the best epoch and it reports the score. Taking the maximum across
epochs on the same data makes that score slightly optimistic. To quantify this, the
best setup is retrained from scratch on a 70/15/15 patient-level split, where the
test set is touched exactly once at the very end. The code asserts that the three
patient sets do not overlap.

## Repository structure

```
classification_v3.ipynb   main notebook (preprocessing, models, experiments, Grad-CAM)
scripts/                  data preparation script
data/                     derived image selection list
figures/                  result figures from the experiments
results/                  metric tables and training curves from the latest run
requirements.txt          Python dependencies
```

Large files (the raw dataset, preprocessing caches and trained model weights) are
not part of the repository. They are either covered by the Kaggle terms or can be
reproduced by running the notebook.

## Setup and usage

### 1. Clone the repository

```bash
git clone https://github.com/edaguer/BA_breastcancer_mammography-classification.git
cd BA_breastcancer_mammography-classification
```

### 2. Create a conda environment and install the dependencies

This project was built and run with [Anaconda](https://www.anaconda.com/download)
(Python 3.10). From the Anaconda Prompt:

```bash
conda create -n ba_breastcancer python=3.10
conda activate ba_breastcancer
pip install -r requirements.txt
```

Note on GPU/CUDA: the command above installs the CPU version of PyTorch. For much
faster training on an NVIDIA GPU, install the matching CUDA build instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

(choose the CUDA version that fits your graphics driver)

### 3. Get the dataset

The dataset is part of the RSNA Kaggle competition and is not included here for
licensing reasons:

1. Download it from [Kaggle](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data)
2. Place `train.csv` and the folder `train_images/` in the repository root

### 4. Run the notebook

```bash
jupyter notebook classification_v3.ipynb
```

Run the cells from top to bottom. The first run preprocesses and caches all DICOM
images once (this can take a few minutes), so every later run loads the cache in
seconds. A GPU is strongly recommended for the experiments.

## Results in short

The best setup (ResNet18 with layer3 and layer4 fine tuned, trained on all
available data with class weighting) reaches a validation AUC of about 0.66 to 0.69
(95 percent confidence interval across several seeds). The result is clearly above
chance, but given the small number of cancer images it is not yet clinically
usable. The overfit test (E0) confirms the pipeline can learn, so the modest score
reflects the difficulty and the data size rather than a bug, and the learning curve
(E7) suggests that more data would help most.

On the held-out test set (126 patients, 340 images, never used for training or for
picking the best epoch) the same setup reaches an **AUC of 0.669** (95 percent
bootstrap confidence interval 0.609 to 0.727). This is essentially the same as the
0.658 validation AUC of the final model, which means the reported validation scores
were not noticeably inflated by selecting the best epoch. The test set also shows
the asymmetry of the model: cancer recall is around 0.50 while healthy recall is
around 0.77, so a relevant share of cancer cases is still missed.

One caveat remains: the fine tuning depth (layer3 and layer4) and the learning rate
were chosen in E3 and E4, which ran on all patients. The weights never see the test
set, but the choice of configuration was informed by data that later ended up in it.
A fully clean setup would require holding out the test set before any experiment.

Detailed metrics for every run are in `results/`, and more detail is in the notebook
and in the thesis.

## License

The code in this repository is released under the [MIT License](LICENSE). The
dataset has its own Kaggle terms and is not covered by this license or included in
this repository.
