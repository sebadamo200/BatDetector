import json
import os
import glob
import pandas as pd

def compute_iou(boxA, boxB):
    """
    Compute Intersection-over-Union (IoU) between two bounding boxes.
    boxA, boxB are [x, y, w, h], top-left coordinates + width, height.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    unionArea = float(boxA_area + boxB_area - interArea)
    if unionArea == 0:
        return 0.0
    return interArea / unionArea


def compute_inclusion_ratio(gt_box, pred_box):

    xA = max(gt_box[0], pred_box[0])
    yA = max(gt_box[1], pred_box[1])
    xB = min(gt_box[0] + gt_box[2], pred_box[0] + pred_box[2])
    yB = min(gt_box[1] + gt_box[3], pred_box[1] + pred_box[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    gt_area = gt_box[2] * gt_box[3]
    if gt_area == 0:
        return 0.0
    return interArea / float(gt_area)



def evaluate_matched_vs_all(crops_folder, coco_json_path, iou_threshold=0.5):
    """
    1. Reads ground-truth boxes from a COCO JSON (coco_json_path).
    2. Reads predicted boxes from `crops_folder` (.txt files).
       - Assumes each .txt has a single bounding box [x y w h], or adapt for multiple lines if needed.
    3. For each ground-truth box, checks if ANY predicted box matches (IoU >= threshold).
       - If yes =>  += 1
       - If no  =>  += 1  (In user’s custom sense: “unmatched ground truth”)
    4. Accuracy = TP / (TP + FP)
    5. Prints results.
    Ignores any predicted boxes that do not match a ground-truth box.
    """
    # --------------------------
    # 1) Load COCO ground truths
    # --------------------------
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    ground_truths = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        bbox = ann["bbox"]  # [x, y, w, h]
        if img_id not in ground_truths:
            ground_truths[img_id] = []
        ground_truths[img_id].append(bbox)

    image_id_for_filename = {}
    for img_info in coco_data["images"]:
        file_name = img_info["file_name"]
        image_id_for_filename[file_name] = img_info["id"]

    # --------------------------
    # 2) Load predicted boxes
    # --------------------------
    predictions = {}  
    txt_files = glob.glob(os.path.join(crops_folder, "*.txt"))
    for txt_path in txt_files:
        base_name = os.path.basename(txt_path)
        
        split_name = base_name.split("_crop_")[0]
        original_image_name = split_name + ".JPG"  

        if original_image_name not in image_id_for_filename:
            # Not in COCO => skip
            continue

        pred_img_id = image_id_for_filename[original_image_name]

        with open(txt_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            continue  

        try:
            x_pred, y_pred, w_pred, h_pred = map(float, lines[0].split())
        except:
            continue  

        if pred_img_id not in predictions:
            predictions[pred_img_id] = []
        predictions[pred_img_id].append([x_pred, y_pred, w_pred, h_pred])

    # --------------------------
    # 3) Check how many GT boxes get matched
    # --------------------------
    TP = 0  # matched ground truths
    FP = 0  # unmatched ground truths, in user’s custom definition

    for img_id, gt_boxes in ground_truths.items():
        # predicted boxes for this image (if any)
        pred_boxes = predictions.get(img_id, [])

        for gt_box in gt_boxes:
            matched = False
            for pred_box in pred_boxes:
                iou = compute_iou(gt_box, pred_box)
                if iou >= iou_threshold:
                    matched = True
                    break

            if matched:
                TP += 1
            else:
                FP += 1

    total_gt = TP + FP  # all GT boxes
    if total_gt > 0:
        accuracy = TP / total_gt
    else:
        accuracy = 0

    return TP, FP, total_gt, accuracy




def evaluate_inclusion_ratio(
        crops_folder: str,
        coco_json_path: str,
        inclusion_threshold: float = 0.9,
    ):
    """
    1. Reads ground-truth boxes from a COCO JSON (coco_json_path).
    2. Reads predicted boxes from `crops_folder` (.txt files).
       - Assumes each .txt has a single bounding box [x y w h], or adapt for multiple lines if needed.
    3. For each ground-truth box, checks if ANY predicted box matches (IoU >= threshold).
       - If yes =>  += 1
       - If no  =>  += 1  (In user’s custom sense: “unmatched ground truth”)
    4. Accuracy = TP / (TP + FP)
    5. Prints results.
    Ignores any predicted boxes that do not match a ground-truth box.
    """
    # --------------------------
    # 1) Load COCO ground truths
    # --------------------------
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    
    ground_truths = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        bbox = ann["bbox"]  # [x, y, w, h]
        if img_id not in ground_truths:
            ground_truths[img_id] = []
        ground_truths[img_id].append(bbox)

    image_id_for_filename = {}
    for img_info in coco_data["images"]:
        file_name = img_info["file_name"]
        image_id_for_filename[file_name] = img_info["id"]

    # --------------------------
    # 2) Load predicted boxes
    # --------------------------
    predictions = {}  # image_id -> list of predicted [x, y, w, h]
    txt_files = glob.glob(os.path.join(crops_folder, "*.txt"))
    for txt_path in txt_files:
        base_name = os.path.basename(txt_path)
        split_name = base_name.split("_crop_")[0]
        original_image_name = split_name + ".JPG"  # Adjust if needed

        if original_image_name not in image_id_for_filename:
            # Not in COCO => skip
            continue

        pred_img_id = image_id_for_filename[original_image_name]

        with open(txt_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            continue  # empty file => skip

        # If only one box per file:
        try:
            x_pred, y_pred, w_pred, h_pred = map(float, lines[0].split())
        except:
            continue  # can't parse => skip

        if pred_img_id not in predictions:
            predictions[pred_img_id] = []
        predictions[pred_img_id].append([x_pred, y_pred, w_pred, h_pred])

    # --------------------------
    # 3) Check how many GT boxes get matched
    # --------------------------
    TP = 0  # matched ground truths
    FP = 0  # unmatched ground truths, in user’s custom definition

    for img_id, gt_boxes in ground_truths.items():
        # predicted boxes for this image (if any)
        pred_boxes = predictions.get(img_id, [])

        for gt_box in gt_boxes:
            matched = False
            for pred_box in pred_boxes:
                iou = compute_inclusion_ratio(gt_box, pred_box)
                if iou >= inclusion_threshold:
                    matched = True
                    break

            if matched:
                TP += 1
            else:
                FP += 1

    total_gt = TP + FP  # all GT boxes
    if total_gt > 0:
        accuracy = TP / total_gt
    else:
        accuracy = 0
        
    return TP, FP, total_gt, accuracy




if __name__ == "__main__":
    # Example usage
    coco_jsons = [
        "COCO/Chateau_Xhos.json",
        "COCO/Bornival_cam3_modified.json",
        "COCO/Bornival_cam4_modified.json",
        "COCO/Chaumont_cam1_modified.json",
        "COCO/Chaumont_cam2_modified.json",
        "COCO/jenneret_modified.json",
        "COCO/Modave_camera3_toiture_modified.json",
        "COCO/modave_plancher_modified.json",
        "COCO/Pont_Bousval_2022_NO_HIT_modified.json",
        "COCO/Pont_Bousval_2023_2022CAM12_modified.json",
        "COCO/Pont_Bousval_2023_2023CAM05_modified.json",
        "COCO/Pont_Bousval_2023_2023CAM06_modified.json",
        "COCO/Bousval_WK_modified.json",
    ]

    crops_folders = [
        "results_first_stage/Anthisnes_Chateau_de_Xhos_Camera_1_HIT/crops",
        "results_first_stage/Bornival_PHOTO_2023CAM03/crops",
        "results_first_stage/Bornival_PHOTO_2023CAM04/crops",
        "results_first_stage/Chaumont_Gistoux_Camera_1/crops",
        "results_first_stage/Chaumont_Gistoux_Camera_2/crops",
        "results_first_stage/Jenneret_Camera_1_PHOTO/crops",
        "results_first_stage/Modave_Camera_3_toiture_PHOTO/crops",
        "results_first_stage/Modave_Camera_plancher_PHOTO/crops",
        "results_first_stage/Pont_de_Bousval_Photos_2022_PHOTO/crops",
        "results_first_stage/Pont_de_Bousval_Photos_2023_PHOTO_2022CAM12/crops",
        "results_first_stage/Pont_de_Bousval_Photos_2023_PHOTO_2023CAM05/crops",
        "results_first_stage/Pont_de_Bousval_Photos_2023_PHOTO_2023CAM06/crops",
        "results_first_stage/Pont_de_Bousval_Photos_2023_PHOTO_WK6HDBOUSVAL/crops",
    ]

    names = [
        "Anthisnes_Chateau_de_Xhos_Camera_1_HIT",
        "Bornival_PHOTO_2023CAM03",
        "Bornival_PHOTO_2023CAM04",
        "Chaumont_Gistoux_Camera_1",
        "Chaumont_Gistoux_Camera_2",
        "Jenneret_Camera_1_PHOTO",
        "Modave_Camera_3_toiture_PHOTO",
        "Modave_Camera_plancher_PHOTO",
        "Pont_de_Bousval_Photos_2022_PHOTO",
        "Pont_de_Bousval_Photos_2023_PHOTO_2022CAM12",
        "Pont_de_Bousval_Photos_2023_PHOTO_2023CAM05",
        "Pont_de_Bousval_Photos_2023_PHOTO_2023CAM06",
        "Pont_de_Bousval_Photos_2023_PHOTO_WK6HDBOUSVAL",
    ]

   
    # Evaluate each pair and collect results
    results = []
    iou_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]
    inclusion_ratios=[0.1, 0.3, 0.5, 0.7, 0.9]
    for i in range(len(crops_folders)):
        crop_folder   = crops_folders[i]
        print(names[i])
        num_crops = len(glob.glob(os.path.join(crop_folder, "*.txt")))
        for j in range(len(iou_thresholds)):
            TP, FP, total_gt, accuracy = evaluate_matched_vs_all(
                crops_folder=crops_folders[i],
                coco_json_path=coco_jsons[i],
                iou_threshold=iou_thresholds[j]
            )
            TP2, FP2, total_gt2, accuracy2 = evaluate_inclusion_ratio(
                crops_folder=crops_folders[i],
                coco_json_path=coco_jsons[i],
                inclusion_threshold=inclusion_ratios[j]
            )
            results.append({
                "name": names[i],
                "crops_folder": crops_folders[i],
                "coco_json": coco_jsons[i],
                "IoU_threshold": iou_thresholds[j],
                "Matched GT": TP,
                "Unmached GT": FP,
                "Total_GT": total_gt,
                "Accuracy": accuracy,

                "Inclusion_threshold2": inclusion_ratios[j],
                "Matched GT2": TP2,
                "Unmached GT2": FP2,
                "Total_GT2": total_gt2,
                "Accuracy2": accuracy2,

                "Num_crops": num_crops,
            })

    # Save the results to a CSV file
    df = pd.DataFrame(results)
    mean_accuracy = df["Accuracy"].mean()
    print(f"Mean accuracy across all evaluations: {mean_accuracy:.3f}")
    df.to_csv("results_first_stage/results_stage_1.csv", index=False, sep=";")

