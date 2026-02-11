# Model Information (PC Test Python)

## Overview

This document describes the PPE object-detection models used by the `pc_test_python` pipeline, including:

- Model sources (where available)
- File locations and formats (`.pt` and `.onnx`)
- Verified class lists and class-id mapping
- How the pipeline loads the model and what “YOLO format” outputs look like

This folder is the Windows/PC implementation of the data collector that runs Ultralytics YOLO inference without Hailo hardware.

## Features

- Supports Ultralytics YOLO model loading from `.pt` and `.onnx`
- Provides class-name mapping through `model.names` (preferred) and/or `config.yaml` (fallback)
- Produces detections (bounding boxes + confidence + class id)
- Saves dataset annotations in YOLO text format under `dataset_pc_test/`

## Model Inventory

### Model A: `ppe_best` (10 classes)

**Files (in this folder)**

| Artifact        | Path                                  | Size (bytes) | SHA256                                                             |
| --------------- | ------------------------------------- | -----------: | ------------------------------------------------------------------ |
| PyTorch weights | `pc_test_python/models/ppe_best.pt`   |    6,251,955 | `4D07BBD92CA30D5C12DD67CCF52B2F54F533C9CCFEF534284124682EF9F56129` |
| ONNX export     | `pc_test_python/models/ppe_best.onnx` |   12,272,850 | `35FEC9FEBD478E2BFCC77F3F8003B242935E5D861265DE5C8E2F45449D242777` |

**Source (downloaded weights)**

- The repository includes a download script that fetches upstream `best.pt` and stores it locally as `ppe_best.pt`.
- Upstream project:
  - https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection
- Direct weight download (used by the script):
  - https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection/raw/main/models/best.pt
  - Fallback: https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection/raw/master/models/best.pt

Implementation reference:

- [download_ppe_model.py](file:///c:/Users/youss/OneDrive/Desktop/project/datacollector/pc_test_python/scripts/download_ppe_model.py#L19-L40)

**Verified classes (Ultralytics `model.names`)**

`ppe_best` contains 10 classes:

| Class ID | Class name     |
| -------: | -------------- |
|        0 | Hardhat        |
|        1 | Mask           |
|        2 | NO-Hardhat     |
|        3 | NO-Mask        |
|        4 | NO-Safety Vest |
|        5 | Person         |
|        6 | Safety Cone    |
|        7 | Safety Vest    |
|        8 | machinery      |
|        9 | vehicle        |

**Training metadata embedded in the `.pt` (Ultralytics checkpoint)**

This is what is embedded in `ppe_best.pt` and can help provenance checks:

- Ultralytics version: `8.0.43`
- Training data path (as stored by trainer): `/kaggle/working/ppe_data.yaml`
- Base model: `yolov8n.pt`
- Training params: `imgsz=640`, `epochs=100`, `batch=16`

Note: The embedded metadata does not include a Git remote URL.

the data if the model in case needed :
 `https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow`

### Model B: `p4_ppe` (17 classes)

**Files (currently located one level above `pc_test_python`)**

| Artifact        | Path                        | Size (bytes) | SHA256                                                             |
| --------------- | --------------------------- | -----------: | ------------------------------------------------------------------ |
| PyTorch weights | `datacollector/p4_ppe.pt`   |    6,227,939 | `8F281C89C938D5AE92144150E8B2A22F0E9AC9B2D719D5A39DB8DBB6B0CDF732` |
| ONNX export     | `datacollector/p4_ppe.onnx` |   12,278,400 | `33344783B446E2DA1960DEC64289E3782290AC19858F547203180D168336881E` |

**Source**

- This repository does not include a download URL/script for `p4_ppe.pt`.
- Only the class list and usage are referenced in configuration.
- If you need the GitHub source, you must identify who provided the model or where it was trained/exported.

**Verified classes (Ultralytics `model.names`)**

`p4_ppe` contains 17 classes:

| Class ID | Class name   |
| -------: | ------------ |
|        0 | person       |
|        1 | ear          |
|        2 | ear-mufs     |
|        3 | face         |
|        4 | face-guard   |
|        5 | face-mask    |
|        6 | foot         |
|        7 | tool         |
|        8 | glasses      |
|        9 | gloves       |
|       10 | helmet       |
|       11 | hands        |
|       12 | head         |
|       13 | medical-suit |
|       14 | shoes        |
|       15 | safety-suit  |
|       16 | safety-vest  |

**Training metadata embedded in the `.pt` (Ultralytics checkpoint)**

- Ultralytics version: `8.3.162`
- License field embedded by Ultralytics: `AGPL-3.0 (https://ultralytics.com/license)`
- Training date stored in checkpoint: `2025-07-26T17:45:31.929696`
- Training args (as stored by trainer):
  - `data`: `p4_datasets/sh17/dataset_config.yaml`
  - `model`: `yolov8n.pt`
  - `project`: `p4_fine_tuning_results`
  - `name`: `yolov8n_2`
  - `imgsz=640`, `epochs=5`, `batch=16`, `device=0`

This metadata still does not include a Git remote URL.

## Usage

### 1) Download `ppe_best.pt` (if missing)

From `pc_test_python/`:

```powershell
python scripts\download_ppe_model.py
```

### 2) Export `.pt` to `.onnx`

The repository provides an export script for ONNX:

- [export_onnx.py](file:///c:/Users/youss/OneDrive/Desktop/project/datacollector/pc_test_python/scripts/export_onnx.py)

Example:

```powershell
python scripts\export_onnx.py --model models\ppe_best.pt --output models\ppe_best.onnx
```

### 3) List classes for a model file

From `pc_test_python/`:

```powershell
python ..\check_model_classes.py "models\ppe_best.pt"
python ..\check_model_classes.py "models\ppe_best.onnx"
python ..\check_model_classes.py "..\p4_ppe.pt"
python ..\check_model_classes.py "..\p4_ppe.onnx"
```

### 4) Run inference + save visualization

Use the inference helper script:

- [yolo_inference.py](file:///c:/Users/youss/OneDrive/Desktop/project/datacollector/pc_test_python/yolo_inference.py)

Examples:

```powershell
python yolo_inference.py --model models\ppe_best.onnx --image dataset_pc_test\images\train\<some_image>.jpg --conf-threshold 0.3
python yolo_inference.py --model ..\p4_ppe.onnx --image dataset_pc_test\images\train\<some_image>.jpg --conf-threshold 0.3
```

## Pipeline Integration

### Configuration (`config/config.yaml`)

- The model path is configured under `inference.model_path`.
- In this repository, `pc_test_python/config/config.yaml` currently points to `models/p4_ppe.onnx`.
- The file `models/p4_ppe.onnx` does not exist in `pc_test_python/models/` in the current workspace; the `p4_ppe.onnx` file is located at `datacollector/p4_ppe.onnx`.

Reference:

- [config.yaml](file:///c:/Users/youss/OneDrive/Desktop/project/datacollector/pc_test_python/config/config.yaml#L17-L30)

### Inference engine (`src/inference_engine.py`)

The PC inference engine loads Ultralytics if:

- `ultralytics` is installed, and
- `model_path` ends with `.pt` or `.onnx`

Reference:

- [InferenceEngine](file:///c:/Users/youss/OneDrive/Desktop/project/datacollector/pc_test_python/src/inference_engine.py#L1-L55)

## YOLO Output Format (Labels)

When saving labels as YOLO text files (`dataset_pc_test/labels/.../*.txt`), each line typically follows the YOLO detection format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Where:

- Coordinates are normalized to `[0, 1]` relative to the image width/height.
- `class_id` is the integer index matching the class tables in this document.

## ONNX Runtime Notes

- When loading ONNX models via Ultralytics, the runtime commonly logs:
  - “Using ONNX Runtime CPUExecutionProvider”
- If you see the warning “Unable to automatically guess model task”, the model is still treated as `detect` by default.
- If you want to avoid the warning in custom scripts, Ultralytics supports explicitly setting the task:

```python
from ultralytics import YOLO
model = YOLO("models/ppe_best.onnx", task="detect")
```

## Examples

### Python: run one image and print detections

```python
from ultralytics import YOLO
import cv2

model = YOLO(r"models\ppe_best.onnx", task="detect")
image = cv2.imread(r"dataset_pc_test\images\train\your_image.jpg")
results = model(image, conf=0.3)

for box in results[0].boxes:
    cls_id = int(box.cls[0].item())
    conf = float(box.conf[0].item())
    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
    name = results[0].names[cls_id]
    print(cls_id, name, conf, (x1, y1, x2, y2))
```

## Notes

- Class names and IDs are verified by loading each model using Ultralytics and reading `model.names`.
- The `p4_ppe` model provenance (GitHub source) cannot be determined from this repository alone; the checkpoint embeds training arguments but not the training repo URL.
- If you need strict reproducibility, keep the SHA256 hashes above alongside the artifacts.
