# BatDetector: Open Source End-to-End Pipeline for Bat Monitoring

## Overview

This project provides a robust and scalable bat detection pipeline tailored for wildlife imagery. Leveraging dual background subtraction, morphological post-processing, and an ensemble of lightweight CNN and Vision Transformer (ViT) classifiers, it efficiently identifies and localizes bats in complex scenes.

This pipeline was designed and implemented by **Onciul Alexandra** and **Klinovsky Sebastian**.

## Features

* Dual background subtraction using configurable algorithms (e.g., PBAS, MOG2)
* Morphological operations for noise removal and contour extraction
* ROI (Region of interest) cropping and enhancement via gamma correction / histogram equalization
* Fast rejection of background using a tiny CNN (MobileNetV3)
* Accurate classification using an ensemble of EfficientNet-B0 and ViT
* COCO format export for detected bats and summary CSV reporting
* Optional temporal clustering for improved batch processing
* Fully configurable via CLI

---

## Project Structure

```
bat_classifier/
│
├── models/
│   ├── inference.py
│   └── models.py
│
├── pipeline/
│   ├── core.py
│   ├── mask_processing.py
│   ├── subtractors.py
│   └── models_classification/  # Weights folder
│
├── utils/
│   ├── bounding_boxes.py
│   ├── clustering.py
│   ├── coco_utils.py
│   ├── image_utils.py
│   └── time_utils.py
│
├── config.py
├── main.py
└── README.md
```

---

## Setup and Usage

### 1. **Create and activate a virtual environment**

In your terminal, make sure you’re in the project root directory, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. **Install all requirements**

```bash
pip install -r requirements.txt
```

### 3. **Troubleshooting (if `pip install -r requirements.txt` fails)**

If you encounter errors while installing dependencies, first upgrade your packaging tools and install any missing core libraries:

```bash
pip install --upgrade pip setuptools wheel
pip install numpy
pip install pybgs
.
.
.
```

Then retry:

```bash
pip install -r requirements.txt
```

### 4. **Run an example inference**

Once all dependencies are installed, you can run the bat classifier on a sample folder of images. For example:

```bash
python3 bat_classifier/main.py \
    --path_img_bats Cam1/ \
    --output_dir outputs/Cam1 \
    --visualization
```

* `--path_img_bats Cam1/` points to the directory containing your bat images (e.g., `Cam1/`).
* `--output_dir outputs/Cam1` specifies where to save the results.
* `--visualization` enables per-image visual outputs (e.g., bounding boxes or heatmaps).

## Output
After the command finishes, you’ll find the results inside the `outputs/`


* `output_dir/crops/` — Cropped images around detected bats
* `output_dir/mask/` — Binary masks from the dual subtraction
* `output_dir/*.coco.json` — COCO-format annotation file
* `output_dir/predictions.csv` — Detailed predictions with confidences
* `output_dir/summary.csv` — Per-image summary of bat presence

You can use the generated **COCO JSON** annotations along with tools **CVAT** to manually refine bounding boxes or retrain models on reannotated data. This is useful for boosting accuracy on edge cases or tailoring the model to specific bat species.

---

## Configuration Options

| Argument                            | Description                                            |
| ----------------------------------- | ------------------------------------------------------ |
| `--path_img_bats`                   | Directory with input images                            |
| `--output_dir`                      | Root folder for outputs                                |
| `--model_path`                      | Path to small CNN (MobileNetV3) weights                |
| `--efficientnet`                    | Path to EfficientNet-B0 weights                        |
| `--vit_model`                       | Path to ViT model weights                              |
| `--bg_path`                         | Directory of background frames (for warm-up)           |
| `--roi_rect`                        | ROI cropping rectangle (x,y,w,h)                       |
| `--gamma_value`                     | Gamma correction (default: 1.0)                        |
| `--combine_method`                  | Mask fusion strategy: `average`, `and`, or `or`        |
| `--bgs_primary` / `--bgs_secondary` | Background subtractor names                            |
| `--num_splits`                      | Clusters for temporal segmentation (0 = no clustering) |
| `--visualization`                   | Show intermediate outputs with OpenCV GUI              |

---

## Acknowledgements

This system was developed in collaboration with the **Natagora Microclimate Project**, a five-year initiative studying bat roost conditions in buildings across Wallonia, Belgium.

Due to the decline of traditional roosts from renovations and insulation, Natagora aims to identify environmental factors like temperature, humidity, and materials that support bat presence.

Our contribution focuses on the **automated detection of bats in camera footage**, reducing manual effort and helping quickly identify relevant images for conservation analysis.

---

## License

```
SPDX-License-Identifier: Apache-2.0
© 2025 Onciul Alexandra and Klinovsky Sebastian
```
