# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Onciul Alexandra and Klinovsky Sebastian
# This code was written with the assistance of an AI (e.g. ChatGPT).

import cv2
import numpy as np
from PIL import Image

def load_image_as_tensor(img_path, transform, device):
     # Load image with OpenCV, convert BGR to RGB, apply transform, send to device
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {img_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return transform(image=image_rgb)["image"].to(device)

def gamma_correction(img, gamma=1.2):
    # Apply gamma correction using lookup table (faster)
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(img, table)

def histogram_equalization_color(img):
    # Equalize histogram on Y channel in YCrCb color space
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    return cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)
