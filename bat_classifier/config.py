# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Onciul Alexandra and Klinovsky Sebastian
# This code was written with the assistance of an AI (e.g. ChatGPT).

import torch
from torchvision.models import EfficientNet_B0_Weights
from albumentations.pytorch import ToTensorV2
import albumentations as A

# Device setup: use GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels for classification
CLASS_NAMES = ["background", "bats"]

# Input image size for models
IMAGE_SIZE = 224

# Number of output classes
NUM_CLASSES = 2

# Ensemble weights for Vision Transformer and EfficientNet
ENSEMBLE_VIT_WEIGHT = 0.85
ENSEMBLE_CNN_WEIGHT = 0.15

# Confidence thresholds for classification
HIGH_CONF_THRESHOLD = 0.85

# Normalization parameters (mean and std) for pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# EfficientNet pretrained weights and transform pipeline
_EFF_WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1
EFFICIENTNET_TRANSFORM = _EFF_WEIGHTS.transforms()

# Vision Transformer preprocessing pipeline using Albumentations
VIT_TRANSFORM = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
