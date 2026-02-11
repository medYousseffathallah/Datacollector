# NVIDIA Jetson Nano Integration Guide

This guide explains how to adapt and run the Data Collector system on **NVIDIA Jetson Nano** using the `ultralytics` engine with CUDA acceleration.

## 1. Environment Setup (JetPack 4.6)

**Important:** Jetson Nano typically runs JetPack 4.6.1 (Ubuntu 18.04), which includes Python 3.6 by default. However, **Ultralytics YOLOv8 requires Python 3.8 or newer**.

### 1.1 Prerequisites

Ensure your Jetson Nano is running JetPack 4.6.1.

```bash
cat /etc/nv_tegra_release
```

### 1.2 Install Python 3.8

Since Ubuntu 18.04 comes with Python 3.6, you must install Python 3.8 manually:

```bash
sudo apt update
sudo apt install -y python3.8-dev python3.8-distutils
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.8 get-pip.py
rm get-pip.py
```

### 1.3 Install PyTorch & Torchvision for Python 3.8

You must use NVIDIA's pre-built wheels for JetPack 4.6 (aarch64). **Do not use standard pip install torch.**

1.  **Install System Libraries**:

    ```bash
    sudo apt install -y libopenblas-base libopenmpi-dev libomp-dev libjpeg-dev zlib1g-dev
    ```

2.  **Install PyTorch 1.13.0** (Compatible with JetPack 4.6 / Python 3.8):

    ```bash
    wget https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl
    python3.8 -m pip install torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl
    ```

3.  **Install Torchvision 0.14.0** (Matches PyTorch 1.13):
    ```bash
    git clone --branch v0.14.0 https://github.com/pytorch/vision torchvision
    cd torchvision
    python3.8 setup.py install --user
    cd ..
    ```

### 1.4 Install Project Dependencies

```bash
python3.8 -m pip install -r requirements.txt
```

**Note**: Ensure `~/.local/bin` is in your PATH to use the `yolo` command.
```bash
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
source ~/.bashrc
```

## 2. Configuration for Jetson Nano

Edit `config/config.yaml` to ensure it fits Jetson Nano capabilities:

1.  **Model Selection**:
    The default configuration uses the standard **YOLOv8n-seg** model (`models/yolov8n-seg.pt`) for initial testing and validation. This model is lightweight and works well on Jetson Nano.

    To switch to the custom **PPE Detection Model** (`models/ppe_best.pt`):
    -   Edit `config/config.yaml`.
    -   Uncomment `model_path: "models/ppe_best.pt"`.
    -   (Optional) Uncomment `class_names` if you need to enforce specific class mappings.

    ```yaml
    inference:
      # model_path: "models/yolov8n-seg.pt"
      model_path: "models/ppe_best.pt"
      input_shape: [640, 640]
    ```

2.  **TensorRT Optimization (Highly Recommended)**:
    Running `.pt` files directly is slow (~5 FPS). Exporting to TensorRT `.engine` can boost this to ~15-20 FPS.

    **Export Command (for standard model)**:
    ```bash
    yolo export model=models/yolov8n-seg.pt format=engine half=True device=0
    ```

    **Export Command (for PPE model)**:
    ```bash
    yolo export model=models/ppe_best.pt format=engine half=True device=0
    ```
    *Note: This process takes 5-10 minutes on Jetson Nano.*

    **Update Config**:
    After exporting, update `config/config.yaml` to point to the `.engine` file:

    ```yaml
    inference:
      model_path: "models/yolov8n-seg.engine" # or models/ppe_best.engine
    ```

## 3. Running the System

Use Python 3.8 to run the application:

```bash
python3.8 -m src.main --config config/config.yaml
```

## 4. Troubleshooting

- **Memory Issues (OOM)**:
  - Jetson Nano has 4GB shared RAM. Close other apps (Chromium, VS Code) when running.
  - Reduce `input_shape` to `[320, 320]` or `[416, 416]` in `config.yaml`.
  - Create a swap file (4GB recommended) if not already present.
- **"Illegal Instruction"**:
  - This usually means PyTorch was installed via `pip install torch` (x86_64 or wrong arch) instead of the Jetson wheel. Uninstall it and use the NVIDIA wheel instructions above.
