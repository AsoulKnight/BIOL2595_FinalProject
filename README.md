# BIOL2595_FinalProject

Multimodal Deep Learning for Alzheimer's Disease Dementia Assessment
This repository contains an end-to-end deep learning pipeline for classifying Alzheimer's Disease stages (Normal, Mild Cognitive Impairment [MCI], and Alzheimer's Disease [AD]) using a multimodal approach. The model leverages 3D structural MRI scans alongside clinical and demographic tabular data from the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset.

Key Features
Multimodal Architecture: Combines a 3D Convolutional Neural Network (ResNet-18) for volumetric MRI processing with a Gradient Boosting model (CatBoost) for clinical tabular features.
Robust Data Pipeline: Automatically aligns clinical scores (MMSE, ADAS, CDR, etc.) and demographics with corresponding extracted .nii MRI volumes.
Memory-Efficient 3D Training: Implements Automatic Mixed Precision (AMP) and Gradient Accumulation to train large 3D medical models on standard GPUs without Out-Of-Memory (OOM) errors.
Imbalance Handling: Utilizes a WeightedRandomSampler and a custom Combined Loss function (Cross Entropy + Focal Loss) to combat severe class imbalances.
Micro-Batch Normalization: Dynamically replaces standard Batch Normalization with InstanceNorm3d to prevent statistic collapse when training with a batch size of 1.
Interpretability: Includes evaluation pipelines with Confusion Matrices and Saliency Maps (heatmaps) to visualize which brain regions drive the model's predictions.

Dependencies
torch, torchvision
monai (Medical Open Network for AI)
catboost
nibabel (for .nii MRI file parsing)
pandas, numpy, scikit-learn
matplotlib, seaborn

Pipeline Overview
1. Data Preparation & Table 1 Generation
Loads raw clinical CSVs, extracts baseline visits, recodes diagnoses, normalizes variables, and generates a standard "Table 1" statistical summary of patient demographics (Age, Gender, Education, MMSE, ADAS, etc.).

2. MRI Preprocessing & Augmentation (MONAI)
MRI scans are passed through an advanced preprocessing pipeline:

CropForegroundd: Automatically strips the empty background space to force the network to focus on brain tissue.
Resized: Standardizes all volumes to a uniform 128x128x128 spatial dimension.
Augmentations: Applies random 3D rotations, flips, and affine transformations during training to prevent overfitting.
3. Model Training (MRI Branch)
A 3D ResNet-18 model acts as the visual feature extractor. Training is stabilized using:

Combined Loss: Cross Entropy with Label Smoothing + Focal Loss.
Gradient Accumulation: Simulates a batch size of 8 while physically passing 1 volume at a time to prevent OOM errors.
Checkpointing: Automatically tracks validation accuracy and saves the best model state.
4. Evaluation & Interpretability
The validation phase tracks exact classification metrics. To ensure clinical transparency, Saliency Maps project the model's attention back onto the original Sagittal, Coronal, and Axial MRI slices.

Table1_data_gen.py is originally used to extract information for table 1 of clinical data
