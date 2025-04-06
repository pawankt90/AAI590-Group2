# Pneumonia Detection with CNNs and GAN-Based Data Augmentation

**Capstone Project – MS-AAI-590**  
**University of San Diego – Shiley-Marcos School of Engineering**  
**Contributors:** Ned Kost, Pawan Tahiliani, Kim Vierczhalek  
**Instructor:** Prof. Marbut  
**Date:** March 3, 2025

---

## Project Overview

This project explores the use of **Convolutional Neural Networks (CNNs)** and **Generative Adversarial Networks (GANs)** to improve the diagnostic accuracy of pneumonia detection from chest X-ray images. Motivated by the limited availability and imbalance of real-world medical data, we leverage synthetic image generation to enhance training datasets and demonstrate significant gains in model performance.

The project includes:

- Data exploration and preprocessing
- A custom-built CNN classifier
- Transfer learning using a VGG-16 model
- GAN image generation using ACGAN (from scratch)
- Transfer learning with StyleGAN3
- Extensive performance evaluation and comparison

---

## 📁 Repository Structure

```bash
.
├── 01_Imports_And_EDA.ipynb              # Exploratory Data Analysis and image inspection
├── 02_Custom_ACGAN.ipynb                 # ACGAN (Auxiliary Classifier GAN) image generator
├── 03_StyleGAN Fine Tuning.ipynb         # StyleGAN3 fine-tuning for synthetic image generation
├── 04_CNN_Hyperparam_Search.ipynb        # CNN hyperparameter tuning with Keras Tuner
├── 05_CNN_Classification_128.ipynb       # Final CNN classifier with real/synthetic data
├── 06_VGG_Model.ipynb                    # Transfer Learning using VGG-16 for classification
├── data_utils.py                         # Preprocessing utilities for data loading & augmentation
├── data/
│   └── chest_xray/                       # Contains NORMAL and PNEUMONIA class images
├── Sandbox/                              # Archived/old versions of code used during development
└── README.md                             # You are here
```
---

## Dataset
**Source**: Kermany, Zhang, & Goldbaum (2018)
**Title**: *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*
**License**: CC BY 4.0
*Link*: Mendeley Dataset

- Total: 5,856 grayscale X-ray images

- Classes: NORMAL, PNEUMONIA

- Structure: pre-sorted folders into train, test, val

**Note: Due to class imbalance (~73% Pneumonia), augmentation and synthetic generation techniques were applied to balance the dataset.**

---
## Key Technologies

- Python 3.8+
- TensorFlow / Keras
- PyTorch
- Google Colab (StyleGAN3)
- OpenCV / Pandas / NumPy
- Keras Tuner for hyperparameter optimization

---

## License
This project uses publicly available datasets licensed under **Creative Commons Attribution 4.0 International** (CC BY 4.0).
