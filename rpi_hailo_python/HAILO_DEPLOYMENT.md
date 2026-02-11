# Deployment Guide: Data Collector on Raspberry Pi 5

**Project:** Smart PPE Detection System (Edge Data Collector)
**Target Hardware:** Raspberry Pi 5 + Hailo-8L AI Kit

---

## 1. System Overview

This document provides step-by-step instructions for deploying the Data Collector module. This device is designed to operate autonomously on the edge, capturing and filtering video data for the **EYE-D** ecosystem.

## 2. Prerequisites

- **Hardware**:
  - Raspberry Pi 5 (8GB RAM recommended).
  - Hailo-8L AI Kit (M.2 HAT + Neural Processing Unit).
  - USB Webcam or IP Camera.
- **Software**:
  - Raspberry Pi OS (64-bit).
  - HailoRT (Runtime) and Hailo TAPPAS installed.

---

## 3. Installation & Setup

### Step 1: Transfer Files to the Pi
Copy the application folder to the user directory on the Raspberry Pi.

```bash
# From your development PC
scp -r rpi_hailo_python pi@<PI_IP_ADDRESS>:~/datacollector
```

### Step 2: Install Dependencies
Connect to the Raspberry Pi and install the required Python libraries.

```bash
ssh pi@<PI_IP_ADDRESS>
cd ~/datacollector
pip install -r requirements.txt
```
*Note: Ensure the `hailo-platform` wheel is installed (typically included with the Hailo software suite).*

---

## 4. Running the Application

### Standard Execution
To start the data collector with the default configuration:

```bash
python -m src.main --config config/config.yaml
```

### Verifying Operation
- The system will initialize the Hailo NPU.
- Logs will indicate "Inference Engine: HailoAsyncInference initialized".
- Images detected with PPE (Person, Helmet, Vest) will be saved to `dataset_ppe/images`.
- Metadata is stored in `dataset_ppe/datacollector_ppe.db`.

---

## 5. Advanced: Model Compilation (Optional)

If you need to update the AI model (e.g., after fine-tuning on new data), you must compile the model from PyTorch (`.pt`) or ONNX (`.onnx`) to Hailo Executable Format (`.hef`).

**This step requires a PC with the Hailo Dataflow Compiler (DFC) installed.**

1.  **Export to ONNX**:
    ```bash
    python scripts/export_onnx.py --model models/ppe_best.pt --output models/ppe_best.onnx
    ```

2.  **Compile with DFC**:
    Use the Hailo DFC Python API to quantize and compile the model for the `hailo8l` architecture.
    *Refer to the Hailo DFC User Guide for detailed calibration and compilation scripts.*

3.  **Deploy New Model**:
    Copy the generated `.hef` file to `models/` on the Pi and update `config/config.yaml`:
    ```yaml
    inference:
      model_path: "models/new_model.hef"
    ```
