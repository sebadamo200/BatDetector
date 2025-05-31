# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Onciul Alexandra and Klinovsky Sebastian
# This code was written with the assistance of an AI (e.g. ChatGPT).


import json
import os
import pandas as pd
import argparse


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Each box is [x, y, width, height].

    here x is the min x, y is the min y, w is width, h is height
    """
    xA = max(box1[0],box2[0]) # left (min X) 
    yA = max(box1[1],box2[1]) # top (min Y) 

    xB = min(box1[0]+box1[2],box2[0]+box2[2]) # right (max X)
    yB = min(box1[1]+box1[3],box2[1]+box2[3]) # bottom (max Y)

    interArea = max(0, xB - xA) * max(0, yB - yA +1)
    box1Area = box1[2] * box1[3]
    box2Area = box2[2] * box2[3]

    union_area = box1Area + box2Area - interArea
    if union_area == 0:
        return 0
    iou = interArea / float(union_area) #A∩B/A∪B
    return iou

def compute_inclusion_ratio(gt_box, pred_box, eps: float = 1e-9):
    """
    How much of the GT box is covered by the prediction.
    Returns a value in [0,1].
    """
    xA = max(gt_box[0], pred_box[0])
    yA = max(gt_box[1], pred_box[1])
    xB = min(gt_box[0] + gt_box[2], pred_box[0] + pred_box[2])
    yB = min(gt_box[1] + gt_box[3], pred_box[1] + pred_box[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    gt_area = gt_box[2] * gt_box[3]
    return interArea / (gt_area + eps)


def count_total_bat_boxes(coco_file):
    """Return the total number of bat / groupbats bounding-boxes in a COCO file."""
    return sum(len(b) for b in get_bbox_per_image(coco_file).values())


def compute_global_inclusion_metrics(gt_file, pred_file, inclusion_thr=0.3):
    """
    Computes dataset‑level detection metrics based on GT‑inclusion instead of IoU.

    A GT–pred pair counts as a match (TP) when
        inclusion_ratio(GT, pred) ≥ inclusion_thr.

    The matching is greedy and one‑to‑one:
    each GT can match at most one prediction and vice‑versa.

    Differences w.r.t the previous implementation
    ----------------------------------------------
    * Replaces the `while gt and pr` loop with a single
      pass over the ground‑truth boxes (`for g in gt`).
    * Predictions are removed from `pr` as soon as they are
      assigned, preventing double assignment.
    """
    gt_boxes = get_bbox_per_image(gt_file)
    pr_boxes = get_bbox_per_image(pred_file)

    common_imgs = set(gt_boxes).intersection(pr_boxes)
    tp = fp = fn = 0

    for img in common_imgs:
        gt = gt_boxes[img][:]
        pr = pr_boxes[img][:]

        # iterate once over each GT box
        for g in gt:
            # find the prediction that maximises inclusion with this GT
            best_inc = 0.0
            best_j = None
            for j, p in enumerate(pr):
                r = compute_inclusion_ratio(g, p)
                if r > best_inc:
                    best_inc, best_j = r, j

            # assign if above threshold
            if best_inc >= inclusion_thr and best_j is not None:
                tp += 1
                #pr.pop(best_j)      # remove the matched prediction
            else:
                fn += 1             # this GT remained unmatched

        # any remaining predictions were unmatched
        fp += len(pr)

    return {"matches": tp, "unmatched_gt": fn, "unmatched_pred": fp}


def get_bbox_per_image(coco_file_path):
    """
    A dictionary mapping:
        image_name_without_ext -> [list of bounding boxes]
    where each bounding box is [x_min, y_min, width, height].
    """
    with open(coco_file_path, 'r') as f:
        data = json.load(f)
    category_map = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    image_id_to_file = {}
    for image in data.get("images", []):
        file_name_no_ext = os.path.splitext(image["file_name"])[0]
        image_id_to_file[image["id"]] = file_name_no_ext

    bboxes_per_image = {file_name:[] for file_name in image_id_to_file.values()}
    for annotation in data.get("annotations", []):
        image_id = annotation.get("image_id")
        file_name = image_id_to_file.get(image_id)
        if file_name:
            cat_id = annotation.get("category_id")
            cat_name = category_map.get(cat_id, "")
            if cat_name in ("bat", "groupbats"):
                bbox = annotation.get("bbox")
                if bbox:
                    bboxes_per_image[file_name].append(bbox)
    #print(bboxes_per_image)
    return bboxes_per_image 

def match_and_compute_iou(gt_bboxes, pred_bboxes, iou_threshold=0.5):
    """
    Match each GT to exactly one predicted box (one with the highest IoU)
    as long as is that IoU above some threshold 

    • TP for each matched pair (IoU ≥ threshold).

    • FP for each predicted box that doesn’t match a GT box (or whose best match is below threshold).
       OR where predicted something but in fact it was not a bat or below the threshold.

    • FN for each GT box not matched to any prediction.
       OR There is in fact a bat but the model failed to detect one.
    """
    # to be mutable
    gt_boxes = gt_bboxes[:]
    pr_boxes = pred_bboxes[:]

    matches = 0
    iou_sum = 0.0

    # While we still have boxes in both sets
    while gt_boxes and pr_boxes:
        best_iou = 0.0
        best_pair = (None, None)  # (gt_index, pr_index)

        # Find the highest IoU pair
        for i, gt_box in enumerate(gt_boxes):
            for j, pr_box in enumerate(pr_boxes):
                iou = compute_iou(gt_box, pr_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pair = (i, j)

        # If the best IoU is above the threshold, consider it a match
        if best_iou >= iou_threshold:
            matches += 1
            iou_sum += best_iou
            # Remove those boxes so they can't be matched again
            gt_index, pr_index = best_pair
            # Remove in descending index order so you don't mess up the smaller index
            for index in sorted([gt_index, pr_index], reverse=True):
                if index < len(gt_boxes) and index == gt_index:
                    gt_boxes.pop(index)
                elif index < len(pr_boxes) and index == pr_index:
                    pr_boxes.pop(index)
        else:
            break

    unmatched_gt = len(gt_boxes)
    unmatched_pred = len(pr_boxes)
    
    avg_iou_matched = (iou_sum / matches) if matches > 0 else 0.0

    return {
        "matches": matches,
        "avg_iou_for_matched": avg_iou_matched,
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": unmatched_pred
    }

def compute_global_iou_metrics(gt_file, pred_file, iou_threshold=0.5):
    """
    Parse bounding boxes from ground-truth and prediction,
    match them image-by-image, and produce global IoU metrics.
    """
    gt_bboxes = get_bbox_per_image(gt_file)
    pred_file = get_bbox_per_image(pred_file)

    common_images = set(gt_bboxes.keys()).intersection(pred_file.keys())
    
    total_matches = 0
    total_iou_sum = 0.0
    total_unmatched_gt = 0
    total_unmatched_pred = 0

    for img in common_images:
        match_info = match_and_compute_iou(gt_bboxes[img], pred_file[img], iou_threshold=iou_threshold)
        
        total_matches += match_info["matches"]
        total_iou_sum += match_info["avg_iou_for_matched"] * match_info["matches"]
        total_unmatched_gt += match_info["unmatched_gt"]
        total_unmatched_pred += match_info["unmatched_pred"]
    global_avg_iou = total_iou_sum/total_matches if total_matches > 0.0 else 0.0
    return {
        "iou_threshold": iou_threshold,
        "total_matched_pairs": total_matches,
        "global_avg_iou": global_avg_iou,
        "unmatched_gt_boxes": total_unmatched_gt,
        "unmatched_pred_boxes": total_unmatched_pred
    }

def count_labels_per_image(coco_file_path):
    """
    Count the number of annotations with category "bat" or "groupbats"
    for each image in a COCO JSON file. The returned dictionary maps each
    image file name (without the .png extension) to the count of matching
    bounding boxes.
    """
    with open(coco_file_path, 'r') as f:
        data = json.load(f)
    
    # Map category id to lower-case category name for case-insensitive matching
    category_map = {cat["id"]: cat["name"].lower() for cat in data.get("categories", [])}
    
    # Map image id to file name (without extension)
    image_id_to_file = {}
    for image in data.get("images", []):
        file_name_no_ext = os.path.splitext(image["file_name"])[0]
        image_id_to_file[image["id"]] = file_name_no_ext
    
    # Initialize dictionary: each image starts with 0 annotations
    labels_count = {file_name: 0 for file_name in image_id_to_file.values()}
    
    # Count annotations only for the desired categories
    for annotation in data.get("annotations", []):
        image_id = annotation.get("image_id")
        file_name = image_id_to_file.get(image_id)
        if file_name:
            category_id = annotation.get("category_id")
            cat_name = category_map.get(category_id, "")
            # Using lower-case comparison for robustness
            if cat_name in ("bat", "groupbats"):
                labels_count[file_name] += 1

    return labels_count

def compare_counts_exact(gt_dict, pred_dict):
    """
    Compare two dictionaries with an exact match requirement.
    
    For each image (common key):
      - True Positive (TP): ground truth count > 0 and predicted count exactly equals the ground truth count.
      - True Negative (TN): both ground truth and prediction are 0.
      - False Positive (FP): ground truth is 0 but prediction > 0.
      - False Negative (FN): ground truth > 0 but predicted count does not match.
    
    Returns:
      TP, TN, FP, FN, and accuracy calculated over all common keys.
    """
    common_keys = set(gt_dict.keys()).intersection(pred_dict.keys())
    if not common_keys:
        print("No common keys found between the two dictionaries.")
        return 0, 0, 0, 0, 0.0

    TP = sum(1 for key in common_keys if gt_dict[key] > 0 and pred_dict[key] == gt_dict[key])
    TN = sum(1 for key in common_keys if gt_dict[key] == 0 and pred_dict[key] == 0)
    FP = sum(1 for key in common_keys if gt_dict[key] == 0 and pred_dict[key] > 0)
    FN = sum(1 for key in common_keys if gt_dict[key] > 0 and pred_dict[key] != gt_dict[key])
    
    accuracy = (TP + TN) / len(common_keys)
    return TP, TN, FP, FN, accuracy

def compare_counts_threshold(gt_dict, pred_dict, threshold=0.7):
    """
    Compare two dictionaries by treating an image as positive if its predicted count is
    > threshold * the count in the ground truth. Calculates the true positives, true negatives,
    false positives, false negatives, and accuracy.

    • True Positive (TP):
      Both ground truth and prediction have bats and the prediction is in the range of at least threshold * gt_dict[key] but not more than the number of the ground truth.
    • True Negative (TN):
      In both ground truth and prediction, there are no bats.
    • False Positive (FP):
      In the ground truth, there are no bats, but the prediction has bats.
    • False Negative (FN):
      In the ground truth, there are bats, but the prediction is either below the treshold or above the ground truth count.
    """
    # threshold = [0, 1]
    if threshold >= 0 or threshold <= 1:
        common_keys = set(gt_dict.keys()).intersection(pred_dict.keys())
        if not common_keys:
            print("No common keys found between the two dictionaries.")
            return 0, 0, 0, 0, 0.0

        # here we mean that the prediction is in the range of the ground truth [0, threshold * gt_dict[key]] but not more than the number of the ground truth
        TP = sum(1 for key in common_keys if pred_dict[key] >= threshold * gt_dict[key] and pred_dict[key]<=gt_dict[key] and gt_dict[key] > 0)
        TN = sum(1 for key in common_keys if gt_dict[key] == 0 and pred_dict[key] == 0)
        FP = sum(1 for key in common_keys if gt_dict[key] == 0 and pred_dict[key] > 0)
        # bats are present but not detected, which is particularly critical if missing them is a concern.
        FN = sum(1 for key in common_keys if gt_dict[key] > 0 and (pred_dict[key] < threshold * gt_dict[key] or pred_dict[key] > gt_dict[key]))
        accuracy = (TP + TN) / len(common_keys) if len(common_keys) else 0.0

        return TP, TN, FP, FN, accuracy
    else:
        print("Threshold should be between 0 and 1")
        return 0, 0, 0, 0, 0.0


def compare_counts_positive(dict1, dict2):
    """
    Compare two dictionaries by treating an image as positive if its count is > 0.
    Calculates the true positives, true negatives, false positives, false negatives, and accuracy.
    
    • True Positive (TP):
    An image in which the ground truth has one or more bats (or bat groups) and your prediction also detects at least one bat. In other words, your model correctly identified that bats are present.

    • True Negative (TN):
    An image in which the ground truth shows no bats and your prediction also does not detect any bats. This means your model correctly recognized the absence of bats.

    • False Positive (FP):
    An image where the ground truth does not have any bats, but your model incorrectly predicts that there are bats. This is a case of a false alarm where the model “sees” bats that aren’t really there.

    • False Negative (FN):
    An image in which the ground truth has one or more bats, but your model fails to detect any. This is the scenario where bats are present but not detected, which is particularly critical if missing them is a concern.
    or classify as not bats when in fact they are.
    """
    common_keys = set(dict1.keys()).intersection(dict2.keys())
    if not common_keys:
        print("No common keys found between the two dictionaries.")
        return 0, 0, 0, 0, 0.0

    TP = sum(1 for key in common_keys if dict1[key] > 0 and dict2[key] > 0)
    TN = sum(1 for key in common_keys if dict1[key] == 0 and dict2[key] == 0)
    FP = sum(1 for key in common_keys if dict1[key] == 0 and dict2[key] > 0)
    FN = sum(1 for key in common_keys if dict1[key] > 0 and (dict2[key] == 0 or key not in dict2))
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return TP, TN, FP, FN, accuracy

def evaluate_annotations(ground_truth_file, predicted_file):
    """
    Evaluate by comparing ground truth and predicted COCO annotations and return metrics.
    """
    ground_truth = count_labels_per_image(ground_truth_file)
    predicted = count_labels_per_image(predicted_file)
    common_keys = set(ground_truth.keys()).intersection(predicted.keys())
    if not common_keys:
        print("No common keys found between the two dictionaries.")
        return None
    
    total_common = len(common_keys)
    gt_annotation_count = sum(1 for key in ground_truth if ground_truth[key] > 0)
    missing_annotations = [key for key in ground_truth if ground_truth[key] > 0 and (key not in predicted or predicted[key] == 0)]
    
    tp, tn, fp, fn, acc = compare_counts_positive(ground_truth, predicted)
    #apply count threshold
    tp_thres, tn_thres, fp_thres, fn_thres, acc_thres = compare_counts_threshold(ground_truth, predicted, threshold=0.7)

    #apply exact match
    tp_exact, tn_exact, fp_exact, fn_exact, acc_exact = compare_counts_exact(ground_truth, predicted)

    # IoU metric
    iou_results = compute_global_iou_metrics(ground_truth_file, predicted_file, iou_threshold = 0.5)
    # print(iou_results)

    # --- new global counts ----------------------------------------------------
    gt_total_bats  = count_total_bat_boxes(ground_truth_file)
    pred_total_bats = count_total_bat_boxes(predicted_file)

    # --- IoU 0.30 metrics -----------------------------------------------------
    iou03 = compute_global_iou_metrics(ground_truth_file, predicted_file,
                                       iou_threshold=0.1)
    TP_iou_0_3 = iou03["total_matched_pairs"]
    FN_iou_0_3 = iou03["unmatched_gt_boxes"]
    FP_iou_0_3 = iou03["unmatched_pred_boxes"]

    precision_iou_0_3 = TP_iou_0_3 / (TP_iou_0_3 + FP_iou_0_3) if TP_iou_0_3 + FP_iou_0_3 else 0
    recall_iou_0_3    = TP_iou_0_3 / (TP_iou_0_3 + FN_iou_0_3) if TP_iou_0_3 + FN_iou_0_3 else 0
    f1_iou_0_3        = 2*TP_iou_0_3 / (2*TP_iou_0_3 + FP_iou_0_3 + FN_iou_0_3) if 2*TP_iou_0_3 + FP_iou_0_3 + FN_iou_0_3 else 0

    # --- Inclusion 0.30 metrics ----------------------------------------------
    inc03 = compute_global_inclusion_metrics(ground_truth_file, predicted_file,
                                             inclusion_thr=0.3)
    TP_incl_0_3 = inc03["matches"]
    FN_incl_0_3 = inc03["unmatched_gt"]
    FP_incl_0_3 = inc03["unmatched_pred"]

    precision_incl_0_3 = TP_incl_0_3 / (TP_incl_0_3 + FP_incl_0_3) if TP_incl_0_3 + FP_incl_0_3 else 0
    recall_incl_0_3    = TP_incl_0_3 / (TP_incl_0_3 + FN_incl_0_3) if TP_incl_0_3 + FN_incl_0_3 else 0
    f1_incl_0_3        = 2*TP_incl_0_3 / (2*TP_incl_0_3 + FP_incl_0_3 + FN_incl_0_3) if 2*TP_incl_0_3 + FP_incl_0_3 + FN_incl_0_3 else 0


    

    return {
            "total_common": total_common,
            "gt_annotation_count": gt_annotation_count,
            # Number of images where bats are present in GT but not in prediction
            "missing_annotations": len(missing_annotations),

            "accuracy": acc,
            "precision" : tp/(tp+fp) if (tp+fp) > 0 else 0,
            "recall" : tp/(tp+fn) if (tp+fn) > 0 else 0,
            "f1_score" : 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0,


            
            "accuracy_threshold": acc_thres,
            "precision_threshold" : tp_thres/(tp_thres+fp_thres) if (tp_thres+fp_thres) > 0 else 0,
            "recall_threshold" : tp_thres/(tp_thres+fn_thres) if (tp_thres+fn_thres) > 0 else 0,
            "f1_score_threshold" : 2*tp_thres/(2*tp_thres+fp_thres+fn_thres) if (2*tp_thres+fp_thres+fn_thres) > 0 else 0,


           
            "accuracy_exact": acc_exact,
            "precision_exact" : tp_exact/(tp_exact+fp_exact) if (tp_exact+fp_exact) > 0 else 0,
            "recall_exact" : tp_exact/(tp_exact+fn_exact) if (tp_exact+fn_exact) > 0 else 0,
            "f1_score_exact" : 2*tp_exact/(2*tp_exact+fp_exact +fn_exact) if (2*tp_exact+fp_exact +fn_exact) > 0 else 0,

            # ---- new global counts ------------------------------------------
            "gt_total_bats": gt_total_bats,
            "pred_total_bats": pred_total_bats,

            # ---- IoU ≥ 0.30 metrics -----------------------------------------
            
            "precision_iou_0_3": precision_iou_0_3,
            "recall_iou_0_3": recall_iou_0_3,
            "f1_iou_0_3": f1_iou_0_3,

            # ---- Inclusion ≥ 0.30 metrics -----------------------------------
            "precision_incl_0_3": precision_incl_0_3,
            "recall_incl_0_3": recall_incl_0_3,
            "f1_incl_0_3": f1_incl_0_3,


            "iou_threshold_used": iou_results["iou_threshold"],
            "total_iou_matched_pairs": iou_results["total_matched_pairs"],
            "global_avg_iou": iou_results["global_avg_iou"],
            "unmatched_gt_boxes": iou_results["unmatched_gt_boxes"],
            "unmatched_pred_boxes": iou_results["unmatched_pred_boxes"]
    }

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate COCO annotations based on ground truth and predicted files."
    )
    parser.add_argument("-true","--ground_truth_file", type=str, required=True, help="Ground truth COCO annotations file")
    parser.add_argument("-pred","--predicted_file", type=str, required=True, help="Predicted COCO annotations file")
    parser.add_argument("-out","--output_file", type=str, default="evaluation_metrics.csv", help="Output file for evaluation metrics")
    args = parser.parse_args()

    metrics = evaluate_annotations(args.ground_truth_file, args.predicted_file)
    if metrics:
        #save to csv
        df = pd.DataFrame([metrics])
        df.to_csv(args.output_file, index=False)
        print(f"Evaluation metrics saved to evaluation_metrics.csv")
    else:
        print("No common keys found between the two dictionaries.")
    
if __name__ == "__main__":
    main()