🧠 3D Attention UNet for Brain Tumor Segmentation and Survival PredictionThis repository contains the official implementation of the paper: “Brain Tumor Segmentation and Survival Prediction using 3D Attention UNet” (preprint).The project delivers a comprehensive deep learning framework for 3D brain tumor segmentation from MRI scans, utilizing attention mechanisms to significantly improve model focus and segmentation accuracy.The baseline UNet3D implementation is adopted from PyTorch-3DUNet.✨ Key Features3D Attention UNet Architecture: Enhances the standard UNet architecture with attention gates to prioritize relevant tumor regions within 3D volumes.Multi-Modal MRI Support: Designed to process FLAIR, T1, T1CE, and T2 MRI scans from the BraTS 2019 dataset.Segmentation & Survival Prediction: Provides distinct modules for both core tumor segmentation and an optional survival prediction task.Robust Preprocessing & Augmentation: Includes normalization, 3D patch extraction (for memory efficiency), rotations, flips, and intensity augmentation.Flexible PyTorch Implementation: Features a modular design for easy experimentation with hyperparameters, custom architectures, and diverse datasets.Comprehensive Evaluation Metrics: Supports quantitative evaluation using Dice Score, Hausdorff Distance, Sensitivity, and Specificity.📂 Repository StructureThe project is logically divided into Segmentation and Survival Prediction components:BRaTs-Attention-Tumor-Segmentation-UNet/
├── Segmentation/
│   ├── 3d_attention_unet.py       # Main UNet 3D model with attention mechanism
│   ├── BuildingBlocks.py          # Core building blocks for the UNet architecture
│   └── sca_3d.py                  # 3D spatial channel attention module
│
├── Survival_Prediction/
│   ├── Matlab/
│   │   ├── Brats_valid/           # Validation and feature files (radiomics, XGBoost notebooks)
│   │   │   └── ...CSV and Jupyter files (e.g., radiomics_normalized.csv, submission.csv)
│   │   │
│   │   └── Feature_Extraction/    # Notebooks for radiomics and statistical feature analysis
│   │       └── ...Jupyter files (e.g., Bland_Altman_plot.ipynb, keplen_mier.ipynb)
│   │
│   └── Python/
│       ├── Classification/        # Classification models and notebooks
│       └── Regression/            # Regression models and notebooks
│
├── train.py                       # Training pipeline for segmentation models
├── evaluate.py                    # Evaluation pipeline for segmentation metrics
├── inference.py                   # Inference scripts for new MRI volumes
├── utils/                          # Helper functions (data loaders, augmentation, metrics)
├── configs/                        # YAML/JSON configuration files for experiments
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
🧪 DatasetThis work is based on the Multimodal Brain Tumor Segmentation Challenge 2019 (BraTS 2019) dataset.MRI ModalitiesAnnotationsKey Preprocessing StepsFLAIR, T1, T1CE, T2Enhancing tumor, Tumor core, Whole tumorNormalization (zero-mean, unit variance), 3D Patch ExtractionDownload Instructions: Registration is required on the official BraTS 2019 website to access the dataset.🙏 AcknowledgementsWe acknowledge the critical contributions of the following:Baseline UNet3D Implementation: PyTorch-3DUNetDataset Organizers: The BraTS 2019 dataset organizers.Contributors: All contributors and reviewers of the original research paper.
