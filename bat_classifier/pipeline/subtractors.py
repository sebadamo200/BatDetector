# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Onciul Alexandra and Klinovsky Sebastian
# This code was written with the assistance of an AI (e.g. ChatGPT).

from __future__ import annotations

import cv2
import math
import os
from pathlib import Path
from typing import Iterable, Tuple, Sequence

import numpy as np
import pybgs as bgs

# Factory to create background subtractors by name
def get_subtractor(name: str):
    n = name.upper()
    if n == "MOG2":
        # OpenCV MOG2 subtractor with preset parameters
        return cv2.createBackgroundSubtractorMOG2(history=500, 
                                                  varThreshold=16, 
                                                  detectShadows=True)
    if n == "KNN":
        # OpenCV KNN subtractor with preset parameters
        return cv2.createBackgroundSubtractorKNN(history=500, 
                                                 dist2Threshold=500.0, 
                                                 detectShadows=True)
    if n == "PBAS":
        # Pixel-Based Adaptive Segmenter from pybgs
        return bgs.PixelBasedAdaptiveSegmenter()
    raise ValueError(f"Unknown background subtractor: {name}")


def _is_opencv_subtractor(sub) -> bool:
    # Check if subtractor is one of OpenCV's built-in types
    from cv2 import BackgroundSubtractorMOG2, BackgroundSubtractorKNN
    
    return isinstance(sub, (BackgroundSubtractorMOG2, BackgroundSubtractorKNN))

# Train background subtractor with a list of images inside ROI
def train_subtractor(
    sub,
    image_paths: Sequence[str | Path],
    roi: Tuple[int, int, int, int],
    lr: float = 0.005,
):
    """
    Update subtractor model on images cropped to region of interest.

    Parameters
    ----------
    sub : background subtractor instance (from get_subtractor)
    image_paths : list of file paths to training images
    roi : tuple (x, y, width, height) defining crop area for training
    lr : learning rate (OpenCV subtractors only)
    """
    x, y, w, h = roi
    is_opencv = _is_opencv_subtractor(sub)

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        roi_img = img[y : y + h, x : x + w]

        if is_opencv:
            sub.apply(roi_img, learningRate=lr)
        else:  # PBAS
            sub.apply(roi_img)