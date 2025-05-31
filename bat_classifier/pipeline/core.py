# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Onciul Alexandra and Klinovsky Sebastian
# This code was written with the assistance of an AI (e.g. ChatGPT).

# --------------------------------------------------------------------------- #
# Imports                                                                     #
# --------------------------------------------------------------------------- #
from __future__ import annotations

# Std-lib
import argparse
import math
import os
import shutil
from pathlib import Path
from typing import Tuple

# Third-party
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from PIL import Image

# Project modules
from config import DEVICE, MEAN, STD, HIGH_CONF_THRESHOLD
from utils.bounding_boxes import boxes_close_or_overlap, merge_boxes
from utils.image_utils import gamma_correction, histogram_equalization_color
from utils.clustering import collect_image_paths, cluster_by_time
from models.models import build_efficientnet, build_vit, load_model
from models.inference import predict_ensemble_from_array
from pipeline.subtractors import get_subtractor, train_subtractor
from pipeline.mask_processing import combine_masks

# Torch / Albumentations
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# COCO + CSV
from utils.coco_utils import build_coco_inference
import csv



# --------------------------------------------------------------- #
# Pre-processing transforms                                       #
# --------------------------------------------------------------- #
classifier_tf = transforms.Compose(
    [
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

vit_tf = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
# --------------------------------------------------------------- #
# Default Models Loader                                           #
# --------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent   #  .../bat_classifier
MODELS_DIR = REPO_ROOT / "models_classification"

DEFAULT_SMALL_CLS   = MODELS_DIR / "MobileNetV3"      / "mobileNetV3.pt"
DEFAULT_EFFICIENT   = MODELS_DIR / "EfficientNetB0"   / "efficientNetB0.pth"
DEFAULT_VIT_CKPT    = MODELS_DIR / "VIT"            / "ViT.pth"

# efficientNetB0
#second_best_model_effi
# ---------------------------------------------------------------
# Helper: warm-up background subtractors with sample images
# ---------------------------------------------------------------

def _warmup_subtractors(
    sub_a,
    sub_b,
    bg_path: str | None,
    roi_rect: Tuple[int, int, int, int],
    min_count: int = 100,
):
    if bg_path is None:
        return

    imgs = [
        os.path.join(bg_path, f)
        for f in os.listdir(bg_path)
        if os.path.isfile(os.path.join(bg_path, f))
    ]
    if not imgs:
        return

    # Duplicate list until we have at least min_count = 100 frames
    if 0 < len(imgs) < min_count:
        reps = math.ceil(min_count / len(imgs))
        imgs = (imgs * reps)[:min_count]

    for sub in (sub_a, sub_b):
        train_subtractor(sub, imgs, roi_rect)


# --------------------------------------------------------------- #
# Helper: lightweight CNN to quickly reject obvious background    #
# --------------------------------------------------------------- #
def _classifier_filter(frame_crop, model, confidence: float = 0.90) -> bool:
    """Return True if crop is *possibly* a bat (p_bat > threshold)."""
    crop_rgb = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    inp = classifier_tf(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)
        p_bg, p_bat = probs[0]
    return p_bg.item() < confidence


# --------------------------------------------------------------- #
# Main processing loop for one folder of images                   #
# --------------------------------------------------------------- #      
def process_images_dual(
    path_img_with_bats: str,                                        # Path to images containing bats
    small_cls_model: torch.nn.Module,                               # Small classification PyTorch model
    efficientnet_model: torch.nn.Module,                            # EfficientNetB0 PyTorch model
    vit_model=None,                                                 # Vision Transformer PyTorch model (optional)
    bg_path: str | None = None,                                     # Path to background images for warmup
    roi_rect: Tuple[int, int, int, int] = (0, 0, 2560, 1376),       # Region of Interest rectangle (x, y, width, height)
    output_folder: str = "subtraction_crops",                       # Folder to save cropped images
    mask_output_folder: str = "masks_dual",                         # Folder to save masks
    area_threshold: int = 180,                                      # Min area to keep detected objects
    dist_threshold: int = 1,                                        # Distance threshold for merging boxes   
    bgs_primary: str = "PBAS",                                      # Primary background subtractor
    bgs_secondary: str = "MOG2",                                    # Secondary background subtractor
    combine_method: str = "average",                                # Method to combine masks ("average", "and", "or")
    visualization: bool = False,                                    # Show debug images during processing
    gamma_value: float = 1.0,                                       # Gamma correction value for image enhancement

):
    """
    Main loop: double background subtraction → mask fusion →
    tiny-CNN filter → saving the crops, 
    → ensemble prediction → COCO and CSV export.
    """   

    
    detections: dict[str, list[dict]] = {}  # COCO export structure

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(mask_output_folder, exist_ok=True)

    parent_dir = Path(output_folder).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    csv_path = parent_dir / "predictions.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["crop_name", "pred_label", "conf_bat", "conf_bg", "x", "y", "w", "h"])

    summary_csv_path = parent_dir / "summary.csv"
    summary_file = open(summary_csv_path, "w", newline="")
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(["photo_name", "threshold", "bat_present"])
    

    # Create subtractors and warm with background images if provided
    sub_a = get_subtractor(bgs_primary)
    sub_b = get_subtractor(bgs_secondary)
    _warmup_subtractors(sub_a, sub_b, bg_path, roi_rect)
    
    # Sorted list of frames to process based on their end number in the filename
    img_paths = sorted(
        (
            os.path.join(path_img_with_bats, f)
            for f in os.listdir(path_img_with_bats)
            if os.path.isfile(os.path.join(path_img_with_bats, f))
        ),
        key=lambda p: int(__import__("re").search(r"_IM_(\d+)", Path(p).name).group(1))
        if __import__("re").search(r"_IM_(\d+)", Path(p).name)
        else 0,
    )

    
    x_roi, y_roi, w_roi, h_roi = roi_rect
    bat_total = 0           # global bat count across all images
    processed = 0           # how many images done
    n_imgs = len(img_paths) # total images

    # Iterate over frames
    for img_path in tqdm(img_paths, desc="Processing images"):
        img_name = Path(img_path).name

        processed += 1
        frame_full = cv2.imread(img_path)
        if frame_full is None:
            continue
        frame = frame_full[y_roi : y_roi + h_roi, x_roi : x_roi + w_roi]

        # ----------------------------------------------------------------------------- #
        # HERE CAN APPLY GAMMA CORRECTION AND/OR HISTOGRAM EQUALIZATION ONTO THE FRAME  #
        # ----------------------------------------------------------------------------- #
        frame = gamma_correction(frame, gamma_value)  # Uncomment to apply gamma correction
        #frame = histogram_equalization_color(frame)   # Uncomment to apply histogram equalization (VERY SLOW)

        # Background subtraction 
        mask_a = sub_a.apply(frame)
        mask_b = sub_b.apply(frame)

        # if visualization: # Uncomment to visualize masks
        #     # show the masks
        #     disp_a = cv2.resize(mask_a, None, fx=0.5, fy=0.5)
        #     disp_b = cv2.resize(mask_b, None, fx=0.5, fy=0.5)
        #     cv2.imshow("Mask A", disp_a)
        #     cv2.imshow("Mask B", disp_b)
        #     # wait key to press
        #     if cv2.waitKey(0) & 0xFF == ord("q"):
        #         break

        # Remove salt-and-pepper noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_a = cv2.morphologyEx(mask_a, cv2.MORPH_OPEN, kernel_open)
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel_open)

        # Fuse masks (average/AND/OR)
        fg_mask = combine_masks(mask_a, 
                                mask_b, 
                                combine_method)
        

        # if visualization: # Uncomment to visualize the foreground mask
        #     disp_mask = cv2.resize(fg_mask, None, fx=0.5, fy=0.5)
        #     cv2.imshow("Foreground Mask", disp_mask)
        #     if cv2.waitKey(0) & 0xFF == ord("q"):
        #         break
        
        # Morphological cleaning
        mask = cv2.medianBlur(fg_mask, 5)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        closed_small = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
        kernel_open2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(closed_small, cv2.MORPH_OPEN, kernel_open2)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        kernel_close2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(closed, kernel_close2, iterations=3)

        # if visualization: # Uncomment to visualize the dilated mask
        #     disp_mask = cv2.resize(dilated, None, fx=0.5, fy=0.5)
        #     cv2.imshow("Dilated Mask", disp_mask)
        #     if cv2.waitKey(0) & 0xFF == ord("q"):
        #         break

        # Bounding boxes from contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        max_area = (w_roi * h_roi) / 3 # Bat cannot be larger than 1/3 of the frame
        for c in contours:
            area = cv2.contourArea(c)
            if area_threshold <= area <= max_area:
                rects.append(cv2.boundingRect(c))
        
        
        
        # Reject obvious background with small CNN
        rects_filtered = [r for r in rects if _classifier_filter(frame[r[1]:r[1]+r[3], r[0]:r[0]+r[2]], small_cls_model)]

        # if visualization: # Uncomment to visualize the filtered rectangles
        #     debug_frame = frame.copy()
        #     for (x, y, w, h) in rects_filtered:
        #         cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     debug_frame = cv2.resize(debug_frame, None, fx=0.5, fy=0.5)  # RESIZE HERE
        #     cv2.imshow("After Filtering", debug_frame)
        #     if cv2.waitKey(0) & 0xFF == ord("q"):
        #         break

        # Merge close / overlapping boxes
        merged =rects_filtered[:]
        while True:
            any_merge = False
            tmp = []
            while merged:
                cur = merged.pop()
                merged_flag = False
                for i in range(len(tmp)):
                    if boxes_close_or_overlap(cur, tmp[i], dist_threshold):
                        tmp[i] = merge_boxes(cur, tmp[i])
                        merged_flag = True
                        any_merge = True
                        break
                if not merged_flag:
                    tmp.append(cur)
            merged = tmp
            if not any_merge:
                break

        # if visualization: # Uncomment to visualize the merged rectangles
        #     debug_frame = frame.copy()
        #     for (x, y, w, h) in merged:
        #         cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     debug_frame = cv2.resize(debug_frame, None, fx=0.5, fy=0.5)  # RESIZE HERE
        #     cv2.imshow("After Merging", debug_frame)
        #     if cv2.waitKey(0) & 0xFF == ord("q"):
        #         break


        #flag bat present 
        bat_found_this_photo = False

        # Final crops → ensemble prediction
        w_max, h_max = 125 * 1.5, 161 * 1.5    # size bounds
        w_min, h_min = 33 * 0.5, 39 * 0.5
        min_a, max_a = w_min * h_min, w_max * h_max * 3

        for idx, (x, y, w, h) in enumerate(merged):
            area = w * h
            if not (min_a <= area <= max_a):
                continue

            # Slightly enlarge crop for ViT branch
            t = 1 - (area - min_a) / (max_a - min_a)
            scale = 1 + 0.5 * (t ** 2)
            new_w, new_h = int(w * scale), int(h * scale)
            dx, dy = (new_w - w) // 2, (new_h - h) // 2
            x2, y2 = max(0, x - dx), max(0, y - dy)

            crop_efn = frame[y : y + h, x : x + w]
            crop_vit = frame[y2 : y2 + new_h, x2 : x2 + new_w]

            # Ensemble prediction
            label_str, conf = predict_ensemble_from_array(
                crop_efn=crop_efn,
                crop_vit=crop_vit,
                effi_model=efficientnet_model,
                vit_model=vit_model,
            )

            crop_name = f"{Path(img_path).stem}_{label_str}_{idx}.png"
            cv2.imwrite(os.path.join(output_folder, crop_name), crop_efn)
            cv2.imwrite(os.path.join(mask_output_folder, crop_name), fg_mask[y : y + h, x : x + w])

            csv_writer.writerow(
                [crop_name, label_str, f"{conf:.4f}", f"{1-conf:.4f}", x, y, w, h]
            )
            if label_str.lower() == "bat":
                bat_found_this_photo = True

            # Save positive detections for COCO
            if label_str == "bat":
                detections.setdefault(Path(img_path).name, []).append(
                    {
                        "bbox": [x, y, w, h],
                        "category_id": 1,
                        "score": float(conf),
                        "attributes": {"SingleOrMultiple": "single"},
                    }
                )

            # Optional debug overlay
            if visualization:
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h),
                    (0, 255, 0) if label_str == "bat" else (0, 0, 255),
                    1,
                )
        
        yes_no = "yes" if bat_found_this_photo else "no"
        summary_writer.writerow([img_name, HIGH_CONF_THRESHOLD ,yes_no])

        # Counter for total bats in this image
        bats_in_img = sum(1 for d in detections.get(Path(img_path).name, []))
        bat_total += bats_in_img

        if visualization:
            font_scale = 1.0
            thickness = 2
            color = (255, 255, 255)

            # top-left: per-image count
            text_tl = f"Image {processed}: {bats_in_img} bats"
            pos_tl = (10, 30)  

            cv2.putText(
                frame,
                text_tl,
                pos_tl,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

            # top-right: global count
            txt_tr = f"Bat counter: {bat_total}"
            (tw, th), _ = cv2.getTextSize(txt_tr, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            pos_tr = (frame.shape[1] - tw - 10, 30)

            cv2.putText(
                frame,
                txt_tr,
                pos_tr,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

            prog = f"Processed {processed}/{n_imgs}"
            (pw, ph), _ = cv2.getTextSize(prog, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            pos_bc = ((frame.shape[1] - pw) // 2, frame.shape[0] - 15)

            cv2.putText(
                frame,
                prog,
                pos_bc,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

            cv2.imshow("Debug", cv2.resize(frame, None, fx=0.5, fy=0.5))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    csv_file.close()
    summary_file.close()
    timing_csv = Path(output_folder).parent / "timing.csv"
    

    print(f"✅  Timing metrics saved to {timing_csv}")
    return detections

# --------------------------------------------------------------------------- #
# Temporal Clustering + Complete Pipeline                                     #
# --------------------------------------------------------------------------- #
def cluster_and_process(args):

    # Create output directories
    base_out = Path(args.output_dir)
    crops_dir = base_out / "crops"
    masks_dir = base_out / "mask"
    coco_file = base_out / (base_out.name + ".coco.json")

    # Ensure directories exist
    base_out.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    # Override individual paths
    args.output_folder = str(crops_dir)
    args.mask_output_folder = str(masks_dir)
    args.coco_output = str(coco_file)

    # load all models, then notify
    print("Loading models…")

    # MobileNetV3
    small_cls = load_model(args.model_path, DEVICE)
    
    # EfficientNet
    effi = build_efficientnet().to(DEVICE)
    effi = torch.compile(effi, backend="eager")        # works on CPU or GPU
    effi.load_state_dict(torch.load(args.efficientnet,
                                     map_location=DEVICE,
                                     weights_only=False))
    effi.eval()


    # Vision Transformer
    vit = build_vit(
    weights_path=args.vit_model,
    device=DEVICE
    )
    print("Models loaded successfully.")

    # Collect all image paths
    all_imgs = collect_image_paths(args.path_img_bats)

    all_detections: dict[str, list[dict]] = {}
    
    # No time clustering
    if args.num_splits <= 0:
        all_detections = process_images_dual(
            path_img_with_bats=args.path_img_bats,
            small_cls_model=small_cls,
            efficientnet_model=effi,
            vit_model=vit,
            bg_path=args.bg_path,
            roi_rect=tuple(map(int, args.roi_rect.split(","))),
            output_folder=args.output_folder,
            mask_output_folder=args.mask_output_folder,
            area_threshold=args.area_threshold,
            dist_threshold=args.dist_threshold,
            bgs_primary=args.bgs_primary,
            bgs_secondary=args.bgs_secondary,
            combine_method=args.combine_method,
            visualization=args.visualization,
            gamma_value=args.gamma_value,
        )
        build_coco_inference(
            image_dir=args.path_img_bats,
            detections=all_detections,
            output_json=args.coco_output,
        )
        print(f"✅  COCO JSON written to {args.coco_output}")
        return


    # Cluster by time and process each cluster separately
    clusters = cluster_by_time(all_imgs, k=args.num_splits)
    for idx, (lbl, paths) in enumerate(clusters.items(), 1):
        print(f"[{idx}/{len(clusters)}] Cluster {lbl!s} – {len(paths)} images")

        tmp_dir = Path(args.output_folder) / f"tmp_cluster_{lbl}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for p in paths:
            shutil.copy2(p, tmp_dir / Path(p).name)

        dets = process_images_dual(
            path_img_with_bats=str(tmp_dir),
            small_cls_model=small_cls,
            efficientnet_model=effi,
            vit_model=vit,
            bg_path=args.bg_path,
            roi_rect=tuple(map(int, args.roi_rect.split(","))),
            output_folder=args.output_folder,
            mask_output_folder=args.mask_output_folder,
            area_threshold=args.area_threshold,
            dist_threshold=args.dist_threshold,
            bgs_primary=args.bgs_primary,
            bgs_secondary=args.bgs_secondary,
            combine_method=args.combine_method,
            visualization=args.visualization,
            gamma_value=args.gamma_value,
        )
        for fn, lst in dets.items():
                all_detections.setdefault(fn, []).extend(lst)
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Couldn't delete temporary folder '{tmp_dir}': {e}")


    build_coco_inference(args.path_img_bats, all_detections, args.coco_output)
    print(f"✅  COCO JSON written to {args.coco_output}")


# --------------------------------------------------------------- #
# CLI builder                                                     #   
# --------------------------------------------------------------- #
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dual-background-subtraction bat detector")

    # Required only for the raw images and output location
    p.add_argument("--path_img_bats", required=True,
                   help="Folder with raw bat images")

    # Model paths
    p.add_argument("--model_path", default=str(DEFAULT_SMALL_CLS),
                   help=f"Small-classifier .pth file "
                        f"(default: {DEFAULT_SMALL_CLS.relative_to(REPO_ROOT)})")

    p.add_argument("--efficientnet", default=str(DEFAULT_EFFICIENT),
                   help=f"EfficientNet weights (.pth) "
                        f"(default: {DEFAULT_EFFICIENT.relative_to(REPO_ROOT)})")

    p.add_argument("--vit_model", default=str(DEFAULT_VIT_CKPT), help=(
                    "Path to the fine-tuned ViT .pth checkpoint "
                    f"(default: {DEFAULT_VIT_CKPT.relative_to(REPO_ROOT)})"
                )
    )
    # Subtractor parameters
    p.add_argument("--bgs_primary",   default="PBAS")
    p.add_argument("--bgs_secondary", default="MOG2")
    p.add_argument("--combine_method", default="average",
                   choices=["average", "and", "or"])
    p.add_argument("--bg_path", default=None,
                   help="Warm-up frame folder for subtractors")

    # ROI & post-processing
    p.add_argument("--roi_rect",       default="0,0,2560,1376")
    p.add_argument("--gamma_value",    type=float, default=1.0)
    p.add_argument("--area_threshold", type=int,   default=180)
    p.add_argument("--dist_threshold", type=int,   default=1)

    # Temporal clustering
    p.add_argument("--num_splits", type=int, default=0)

    # Output & visualisation
    p.add_argument("--output_dir", required=True,
                   help="Base dir for crops/, mask/, coco.json, etc.")
    p.add_argument("--visualization", action="store_true")

    return p