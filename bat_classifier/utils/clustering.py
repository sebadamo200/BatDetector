# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Onciul Alexandra and Klinovsky Sebastian
# This code was written with the assistance of an AI (e.g. ChatGPT).

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Sequence, Dict, List

from sklearn.cluster import KMeans

from .time_utils import get_exif_time_in_seconds, time_to_2d


def collect_image_paths(dir_path: str | os.PathLike) -> list[str]:
    # Return list of file paths (files only) in given directory
    files = os.listdir(dir_path)
    return [
        os.path.join(dir_path, f)
        for f in files
        if os.path.isfile(os.path.join(dir_path, f))
    ]


# --------------------------------------------------------------------------- #
# Cluster images by capture time                                              #
# --------------------------------------------------------------------------- #
def cluster_by_time(
    image_paths: Sequence[str],
    k: int,
    random_state: int = 42,
) -> Dict[str | int, List[str]]:
    
    # If k <= 0, put all images in one cluster named "all"
    if k <= 0:
        return {"all": list(image_paths)}

    time_vecs, valid_imgs, fallback = [], [], []

    for p in image_paths:
        sec = get_exif_time_in_seconds(p)       # get capture time in seconds
        if sec is None:
            fallback.append(p)                  # no valid EXIF time, save for fallback
        else:
            valid_imgs.append(p)
            time_vecs.append(time_to_2d(sec))   # convert time to 2D point on a circle

    
    # If no images with valid EXIF, return fallback group only
    if not valid_imgs:
        return {"fallback": fallback}

    # K-means clustering on 24h circle points
    km = KMeans(n_clusters=k, random_state=random_state)
    labels = km.fit_predict(np.asarray(time_vecs))

    clusters: Dict[str | int, List[str]] = {}
    for lbl, path in zip(labels, valid_imgs):
        clusters.setdefault(lbl, []).append(path)
    if fallback:
        clusters["fallback"] = fallback

    return clusters
