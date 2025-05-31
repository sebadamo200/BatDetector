# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Onciul Alexandra and Klinovsky Sebastian

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from transformers import ViTForImageClassification

def build_efficientnet(num_classes=2):
    # create EfficientNet-B0 with new output layer
    model = efficientnet_b0(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_feat, num_classes)
    return model


def build_vit(weights_path: str, device: torch.device):
    """
    Instantiate a ViT‐base (224×224, patch 16) with a 2‐class head, compile it 
    (so that its state_dict keys are prefixed with '_orig_mod.'), and then load 
    local weights from `weights_path`.
    """
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        local_files_only=True
    )

    embed_dim = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(embed_dim, 2)
    )

    model = model.to(device)
    try:
        model = torch.compile(model)
    except Exception:
        pass

    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def load_model(model_path, device):
    # load TorchScript model (MobileNetV3) to remove most probable background
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model