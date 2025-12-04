🧠 3D Attention UNet for Brain Tumor Segmentation and Survival Prediction
This repository contains the official implementation of the paper: "Brain Tumor Segmentation and Survival Prediction using 3D Attention UNet" (preprint).
The project provides a deep learning framework for 3D brain tumor segmentation from MRI scans, leveraging attention mechanisms to improve model focus and segmentation accuracy.
The baseline UNet3D implementation is adopted from PyTorch-3DUNet.

Features

3D Attention UNet Architecture: Enhances standard UNet with attention gates to focus on relevant tumor regions.
Multi-Modal MRI Support: Works with FLAIR, T1, T1CE, and T2 MRI scans from BraTS 2019 dataset.
Segmentation and Survival Prediction: Provides modules for both tumor segmentation and optional survival prediction tasks.
Advanced Preprocessing & Augmentation: Normalization, patch extraction, flipping, rotation, and intensity augmentation.
Flexible PyTorch Implementation: Modular design for easy experimentation with hyperparameters, architectures, and datasets.
Evaluation Metrics: Dice Score, Hausdorff Distance, Sensitivity, Specificity, and more.

Repository Structure
BRaTs-Attention-Tumor-Segmentation-UNet/
├── Segmentation/
│   ├── 3d_attention_unet.py      # Main UNet 3D model with attention
│   ├── BuildingBlocks.py          # Building blocks for the UNet architecture
│   └── sca_3d.py                  # 3D spatial channel attention module
│
├── Survival_Prediction/
│   ├── Matlab/
│   │   ├── Brats_valid/           # Validation and feature files for survival prediction
│   │   │   ├── Best_mode_withRFE_XGB.ipynb
│   │   │   ├── Normalizing.ipynb
│   │   │   ├── XGB_withRFE_crossvalidation.ipynb
│   │   │   ├── XGBregressor.ipynb
│   │   │   ├── radiomics_normalized.csv
│   │   │   ├── radiomic_normalized_SS.csv
│   │   │   ├── radiomic_normalized_new.csv
│   │   │   ├── radiomic_test_normalized.csv
│   │   │   ├── radiomic_valid_normalized.csv
│   │   │   ├── radiomic_valid_normalized.SS.csv
│   │   │   ├── submission.csv
│   │   │   ├── submission_best_14.csv
│   │   │   ├── submission_best_14_csv
│   │   │   └── ...other CSV and Jupyter files
│   │   │
│   │   └── Feature_Extraction/    # Feature extraction notebooks
│   │       ├── Bland_Altman_plot.ipynb
│   │       ├── Filename_into_textfile.ipynb
│   │       ├── keplen_mier.ipynb
│   │       ├── npy_fromcsv.ipynb
│   │       └── spearmanr.ipynb
│   │
│   └── Python/
│       ├── Classification/         # Classification models and notebooks
│       └── Regression/             # Regression models and notebooks
│
├── train.py                       # Training pipeline for segmentation
├── evaluate.py                    # Evaluation pipeline for segmentation metrics
├── inference.py                   # Inference scripts for new MRI volumes
├── utils/                         # Helper functions (data loaders, augmentation, metrics)
├── configs/                       # YAML/JSON configuration files for experiments
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation

🧪 Dataset
This project uses the BraTS 2019 dataset:

Multi-modal MRI scans: FLAIR, T1, T1CE, T2
Tumor annotations: Enhancing tumor, tumor core, whole tumor
Preprocessing:

Normalization (zero-mean, unit variance per modality)
Patch extraction (3D volumes for memory efficiency)
Data augmentation: rotations, flips, intensity shifts
Download instructions: Register on the official BraTS 2019 website to access the dataset.


Acknowledgements

Baseline UNet3D implementation: PyTorch-3DUNet
BraTS 2019 dataset organizers
