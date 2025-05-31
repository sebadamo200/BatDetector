# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Onciul Alexandra and Klinovsky Sebastian

# ensemble inference
import torch
import numpy as np
from PIL import Image
from config import (
    DEVICE,
    HIGH_CONF_THRESHOLD,
    ENSEMBLE_CNN_WEIGHT,
    ENSEMBLE_VIT_WEIGHT,
    EFFICIENTNET_TRANSFORM,
    VIT_TRANSFORM,
)

@torch.no_grad()
def predict_ensemble_from_array(crop_efn: np.ndarray,
                                crop_vit: np.ndarray,
                                effi_model,
                                vit_model=None):
    if crop_efn.shape[2] == 3:
        img = Image.fromarray(crop_efn[..., ::-1])
    else:
        img = Image.fromarray(crop_efn)

    t_effi = EFFICIENTNET_TRANSFORM(img).unsqueeze(0).to(DEVICE)  # [1,C,H,W]
    cnn_logits = effi_model(t_effi)
    cnn_prob = torch.sigmoid(cnn_logits)[0, 1].item()             # CNN prob

    if vit_model is not None:
        if crop_vit.shape[2] == 3:
            img = Image.fromarray(crop_vit[..., ::-1])
        else:
            img = Image.fromarray(crop_vit)
        vit_inputs = VIT_TRANSFORM(image=np.array(img))["image"]  # still [C,H,W]
        vit_inputs = vit_inputs.unsqueeze(0).to(DEVICE)           # [1,C,H,W]
        vit_logits = vit_model(pixel_values=vit_inputs).logits
        vit_prob = vit_logits.softmax(dim=1)[0, 1].item()         # VIT prob
        # ensemble
        prob = ENSEMBLE_CNN_WEIGHT * cnn_prob + ENSEMBLE_VIT_WEIGHT * vit_prob
        print(f"effn: {cnn_prob:.4f} vit: {vit_prob:.4f}")
    else:
        prob = cnn_prob
        vit_prob = None
        # print(f"effn: {cnn_prob:.4f}")

    label = "bat" if prob >= HIGH_CONF_THRESHOLD else "background" # decision bat/background
    return label, prob