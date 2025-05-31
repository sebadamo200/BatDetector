# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Onciul Alexandra and Klinovsky Sebastian
# This code was written with the assistance of an AI (e.g. ChatGPT).

import os
import json
from datetime import datetime
from pathlib import Path
import cv2

def build_coco_inference(
    image_dir: str,
    detections: dict,
    output_json: str,
    licenses: list = None,
    info: dict = None,
    categories: list = None,
):
    """
    Build a COCO-format JSON with your inference results.

    Args:
        image_dir: folder containing the source images.
        detections: a dict mapping filename → list of detections, where each detection is
            {
                'bbox': [x, y, w, h],
                'category_id': int,
                'score': float,
                'iscrowd': 0,
                # optional extra attrs:
                'attributes': {...}
            }
        output_json: path to write the COCO JSON file.
        licenses: list of license dicts (id, name, url).
        info: dict with keys (year, version, description, url, date_created, contributor).
        categories: list of category dicts (id, name, supercategory).
    """
    # defaults
    licenses = licenses or [{"id": 0, "name": "", "url": ""}]
    info = info or {
        "year": datetime.now().year,
        "version": "1.0",
        "description": "Inference results",
        "contributor": "",
        "url": "",
        "date_created": datetime.now().isoformat()
    }
    categories = categories or [
        {"id": 1, "name": "bat", "supercategory": ""},
        {"id": 2, "name": "groupbats", "supercategory": ""}
    ]

    # image entries
    images = []
    filename_to_id = {}
    for idx, fname in enumerate(sorted(os.listdir(image_dir)), start=1):
        path = os.path.join(image_dir, fname)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path)
        h, w = img.shape[:2]
        images.append({
            "id": idx,
            "width": w,
            "height": h,
            "file_name": fname,
            "license": licenses[0]["id"],
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        })
        filename_to_id[fname] = idx

    # annotation entries
    annotations = []
    ann_id = 1
    for fname, dets in detections.items():
        image_id = filename_to_id.get(fname)
        if image_id is None:
            continue
        for det in dets:
            x, y, w, h = det["bbox"]
            area = w * h
            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": det["category_id"],
                "bbox": [x, y, w, h],
                "area": area,
                "segmentation": det.get("segmentation", []),
                "iscrowd": det.get("iscrowd", 0),
                "attributes": det.get("attributes", {
                    "SingleOrMultiple": "single",
                    "occluded": False,
                    "rotation": 0.0
                })
            }
            # if you want to include detection confidence:
            if "score" in det:
                ann["score"] = det["score"]
            annotations.append(ann)
            ann_id += 1

    # assemble and write
    coco = {
        "licenses": licenses,
        "info": info,
        "categories": categories,
        "images": images,
        "annotations": annotations
    }
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"COCO JSON saved to {output_json}")