
# 3D Attention UNet for Brain Tumor Segmentation and Survival Prediction

This repository contains the official implementation of the paper: **“Brain Tumor Segmentation and Survival Prediction using 3D Attention UNet”** ([preprint](https://arxiv.org/pdf/2104.00985.pdf)).  
The project provides a deep learning framework for **3D brain tumor segmentation** from MRI scans, leveraging attention mechanisms to improve model focus and segmentation accuracy.  

The baseline UNet3D implementation is adopted from [PyTorch-3DUNet](https://github.com/wolny/pytorch-3dunet).

---

Files Exist

---

## Features

- **3D Attention UNet Architecture**: Enhances standard UNet with attention gates to focus on relevant tumor regions.  
- **Multi-Modal MRI Support**: Works with FLAIR, T1, T1CE, and T2 MRI scans from BraTS 2019 dataset.  
- **Segmentation and Survival Prediction**: Provides modules for both tumor segmentation and optional survival prediction tasks.  
- **Advanced Preprocessing & Augmentation**: Normalization, patch extraction, flipping, rotation, and intensity augmentation.  
- **Flexible PyTorch Implementation**: Modular design for easy experimentation with hyperparameters, architectures, and datasets.  
- **Evaluation Metrics**: Dice Score, Hausdorff Distance, Sensitivity, Specificity, and more.

---

## Repository Structure
/BRaTs-attention-Tumor-Segmentatiin-UNet-
├── Segmentation
    |-3d_attention_unet.py
    |-BuildingBlocks.py
    |-sca_3d.py
    
├── survivak_prediction
    |-matlab
        |-Brats_valid#All the Model feature of UNET and BraTS 
            |- Best_mode_withRFE_XGB.ipynb
            |-Normalizing.ipynb
            |-XGB_withRFE_crossvalidation
            |-XGBregressor.ipynb
            |-radiomics_normalized.csv
            |-radiomic_normalized_SS.csv
            |-radiomic_normalized_new.csv
            |-radiomic_test_normalized.csv
            |-radiomic_valid_normalized.csv
            |-radiomic_valid_normalized.SS.csv
            |-Another tow of radiomic files
            |-submission.csv
            |-submission_best_14.csv
            |-submission_best_14_csv
            |-rest of all the csv and jupyter files 
        |-Feature Extraction
            |-Bland Altman plot.ipynb
            |-Filename_into_textfile.ipynb
            |-keplen mier.ipynb
            |-npy_fromcsv.ipynb
            |-spearmanr.ipynb
    -Python
        |-Classification
            |- All the folders and Model availables with jupyter notebook
        |-Regression 
            |- All folders and Model availabkle wirh notebook

├── train.py # Training pipeline
├── evaluate.py # Evaluation pipeline for segmentation metrics
├── inference.py # Inference scripts for new MRI volumes
├── utils/ # Helper functions (data loaders, augmentation, metrics)
├── configs/ # YAML/JSON configuration files for experiments
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## Dataset

This project uses the **[BraTS 2019 dataset](https://www.med.upenn.edu/cbica/brats2019/data.html)**:  

- **Multi-modal MRI scans**: FLAIR, T1, T1CE, T2  
- **Tumor annotations**: Enhancing tumor, tumor core, whole tumor  
- **Preprocessing**:  
  - Normalization (zero-mean, unit variance per modality)  
  - Patch extraction (3D volumes for memory efficiency)  
  - Data augmentation: rotations, flips, intensity shifts  
- **Download instructions**: Register on the official BraTS 2019 website to access the dataset.

---

Acknowledgements

Baseline UNet3D implementation: PyTorch-3DUNet

BraTS 2019 dataset organizers

All contributors and reviewers of the original paper

