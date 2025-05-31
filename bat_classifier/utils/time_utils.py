# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Onciul Alexandra and Klinovsky Sebastian
# This code was written with the assistance of an AI (e.g. ChatGPT).

import numpy as np
from PIL import Image, ExifTags

def time_to_2d(seconds: float) -> np.ndarray:
    """
    Convert seconds since midnight to a 2D point on a circle,
    representing time cyclically (midnight near 23:59).
    Returns (sin(angle), cos(angle)) where angle ∈ [0, 2π].
    """
    theta = 2.0 * np.pi * (seconds / 86400.0) # 86400 seconds = 24h
    return np.array([np.sin(theta), np.cos(theta)])

def get_exif_time_in_seconds(image_path: str, tag: str = "DateTime") -> int | None:
    """
    Get the time of day from image EXIF DateTime tag as seconds since midnight.
    Returns None if missing or unreadable.
    """
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
        if not exif:
            return None
        for tag_id, val in exif.items():
            if ExifTags.TAGS.get(tag_id) == tag:
                _, time = val.split(" ") # split date/time
                h, m, s = map(int, time.split(":"))
                return h * 3600 + m * 60 + s
    except:
        pass
    return None