# Classical CV Recapture Detection Baseline

## Assignment Overview

This repository implements a classical Computer Vision baseline for binary recapture detection:

- `0` = real photo
- `1` = photo of a screen (recapture)

The objective is to provide a lightweight, explainable, CPU-friendly pipeline with the required prediction interface:

`python predict.py image.jpg`

## Folder Structure

- `ASSIGNMENT.pdf` - PDF document containing the assignment specifications and requirements
- `dataset_loader.py` - deterministic dataset indexing and labels (`real=0`, `screen=1`)
- `preprocessing.py` - deterministic image loading, RGB/gray conversion, resize+center-crop, normalization
- `features.py` - full handcrafted feature extraction (FFT + LBP + Laplacian + Gradient + Noise residual)
- `train.py` - feature extraction, scaling, cross-validation, model fitting, artifact export
- `evaluate.py` - evaluation metrics (Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix utilities)
- `predict.py` - assignment inference entrypoint (`python predict.py image.jpg`)
- `utils.py` - logging, timing, seed, and image validation helpers
- `requirements.txt` - Python dependencies

## Current Implementation Summary

The baseline is fully classical (no deep learning) and includes:

- deterministic preprocessing
- handcrafted feature extraction
- Logistic Regression classifier
- StandardScaler feature normalization
- artifact serialization for inference consistency

## Classical Computer Vision Pipeline

1. Input image
2. Preprocessing
3. Feature extraction
   - Local FFT + radial spectrum + FFT peak consistency
   - LBP histogram features
   - Laplacian sharpness statistics
   - Gradient magnitude/orientation statistics
   - Noise residual statistics
4. Feature scaling
5. Logistic Regression probability output in `[0, 1]`

## Installation Instructions

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Basic Usage

Train baseline model and save artifacts:

```bash
python train.py
```

Evaluate saved artifacts:

```bash
python evaluate.py
```

Run assignment prediction interface:

```bash
python predict.py image.jpg
```

The prediction command prints one probability score in `[0, 1]`.
