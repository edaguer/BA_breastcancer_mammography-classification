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

### 2. Create a Python environment

Python 3.11 is recommended. A virtual environment is optional but useful:

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

Note on GPU/CUDA: the command above installs the CPU version of PyTorch. For much
faster training on an NVIDIA GPU, install the matching CUDA build instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

(choose the CUDA version that fits your graphics driver)

### 4. Get the dataset

The dataset is part of the RSNA Kaggle competition and is not included here for
licensing reasons:

1. Download it from [Kaggle](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data)
2. Place `train.csv` and the folder `train_images/` in the repository root

### 5. Run the notebook

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

All reported AUC values come from the validation set. No separate test set was held
out, because with so few cancer cases a single test split would be dominated by
sampling noise. Robustness is instead estimated through repeated training with
different seeds and a confidence interval (E5). More detail is in the notebook and
in the thesis.

## License

The code in this repository is released under the [MIT License](LICENSE). The
dataset has its own Kaggle terms and is not covered by this license or included in
this repository.
