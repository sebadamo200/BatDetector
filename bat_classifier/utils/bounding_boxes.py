# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Onciul Alexandra and Klinovsky Sebastian
# This code was written with the assistance of an AI (e.g. ChatGPT).

import numpy as np


def boxes_overlap(a, b):
    # Check if two boxes (x, y, w, h) overlap
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay)


def merge_boxes(a, b):
    # Return smallest box covering both input boxes
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (x1, y1, x2 - x1, y2 - y1)


def boxes_close_or_overlap(a, b, dist_threshold=1):
    # Return True if boxes overlap or are closer than dist_threshold
    if boxes_overlap(a, b):
        return True
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0, max(bx - (ax + aw), ax - (bx + bw)))
    dy = max(0, max(by - (ay + ah), ay - (by + bh)))
    return np.hypot(dx, dy) < dist_threshold