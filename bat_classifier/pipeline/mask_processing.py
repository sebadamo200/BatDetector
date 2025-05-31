# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Onciul Alexandra and Klinovsky Sebastian
# This code was written with the assistance of an AI (e.g. ChatGPT).

from __future__ import annotations

import cv2
import numpy as np
from typing import Literal

CombineMethod = Literal["average", "and", "or"]

def combine_masks(mask1, mask2, method: CombineMethod = "average") -> np.ndarray:
    """
    Combine two binary masks into one using specified method.

    Args:
        mask1, mask2: uint8 masks (0 or 255)
        method: how to combine ("average", "and", or "or")

    Returns:
        Combined uint8 mask (0 or 255)
        
    Raises:
        ValueError: if method is unknown
    """
    if method == "average":
        # Blend masks and threshold to binary
        blended = cv2.addWeighted(mask1.astype("float32"), 0.5, mask2.astype("float32"), 0.5, 0)
        _, out = cv2.threshold(blended, 127, 255, cv2.THRESH_BINARY)
        return out.astype("uint8")
    if method == "and":
        # Intersection (bitwise AND)
        return cv2.bitwise_and(mask1, mask2)
    if method == "or":
         # Union (bitwise OR)
        return cv2.bitwise_or(mask1, mask2)
    raise ValueError(f"Unknown combine method: {method}")