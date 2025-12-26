# NVIDIA Jetson Integration Guide

This guide explains how to adapt and run the Data Collector system on NVIDIA Jetson platforms (Nano, NX, AGX Orin) using the `ultralytics` engine with CUDA acceleration.

## 1. Environment Setup

### 1.1 JetPack Installation
Ensure your Jetson is flashed with **JetPack 5.x or 6.x** (Ubuntu 20.04 or 22.04).
- Verify CUDA is installed:
  ```bash
  nvcc --version
  ```

### 1.2 Python Dependencies
Jetson requires specific versions of PyTorch and Torchvision compatible with the JetPack version. **Do not install via standard pip.**

1.  **Install System Dependencies**:
    ```bash
    sudo apt update
    sudo apt install python3-pip libopenblas-base libopenmpi-dev libomp-dev
    ```

2.  **Install PyTorch & Torchvision**:
    Follow the [NVIDIA Forum Guide](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) to install the correct wheels.
    *Example for JetPack 5.1 (Python 3.8):*
    ```bash
    wget https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-1.14.0a0+44dac51c.nv23.01-cp38-cp38-linux_aarch64.whl
    pip3 install torch-1.14.0a0+44dac51c.nv23.01-cp38-cp38-linux_aarch64.whl
    
    # Install torchvision (build from source matching pytorch version)
    git clone --branch v0.14.1 https://github.com/pytorch/vision torchvision
    cd torchvision
    python3 setup.py install --user
    ```

3.  **Install Ultralytics**:
    ```bash
    pip3 install ultralytics
    ```

4.  **Install Other Requirements**:
    ```bash
    pip3 install -r requirements_jetson.txt
    ```

## 2. Configuration for Jetson

Edit `config/config.yaml`:
1.  **Model Path**: Change `model_path` to point to a `.pt` file (PyTorch) or `.engine` file (TensorRT).
    ```yaml
    inference:
      model_path: "models/yolov8s-seg.pt"  # Or .engine for max speed
      input_shape: [640, 640]
    ```

## 3. Running the System

Use the Jetson-specific entry point:

```bash
python3 -m src.main_jetson --config config/config.yaml
```

## 4. Optimization (TensorRT)

For maximum FPS on Jetson, export your YOLO model to TensorRT engine format.

1.  **Export**:
    ```bash
    yolo export model=models/yolov8s-seg.pt format=engine device=0 half=True
    ```
    *`half=True` enables FP16 precision, critical for Jetson performance.*

2.  **Update Config**:
    Set `model_path: "models/yolov8s-seg.engine"` in `config.yaml`.

## 5. Troubleshooting

-   **OOM (Out of Memory)**: If using Jetson Nano (4GB), use a smaller model (`yolov8n-seg.pt`) or reduce input size to `[320, 320]`.
-   **Slow Inference**: Ensure you are running with `device='cuda'` (the code handles this auto-detection). Verify with `jtop` (install `jetson-stats`) that GPU load increases during execution.
