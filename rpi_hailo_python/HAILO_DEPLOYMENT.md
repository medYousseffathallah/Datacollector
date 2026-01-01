# Manual Steps for Raspberry Pi Deployment

This document outlines the manual steps required to compile the model and deploy the system to the Raspberry Pi 5 with Hailo AI Kit.

## 1. Compile Model to HEF (Requires Hailo DFC)

The Hailo Dataflow Compiler (DFC) is required to convert the standard ONNX model into the Hailo Executable Format (.hef). This software typically runs on a Linux PC or Windows with WSL.

### Step 1.1: Export to ONNX

Run the helper script on your PC:

```bash
python scripts/export_onnx.py --model models/ppe_best.pt --output models/ppe_best.onnx
```

### Step 1.2: Compile (in DFC Environment)

Assuming you have the Hailo DFC installed (v3.27 or later), run the following Python commands in the DFC environment:

```python
from hailo_sdk_client import ClientRunner

model_name = "ppe_best"
onnx_path = "ppe_best.onnx"
chosen_hw_arch = "hailo8l" # Use 'hailo8' for Pi 5 AI Kit (it's actually Hailo-8L usually, check your kit)

runner = ClientRunner(hw_arch=chosen_hw_arch)
runner.translate_onnx_model(onnx_path, model_name)

# Quantization (Required)
# You need a calibration dataset. For simplicity, we can use random data or a few real images.
# This is a simplified example.
runner.optimize(calib_dataset)

# Compile
hef = runner.compile()
with open(f"{model_name}.hef", "wb") as f:
    f.write(hef)
```

**Note:** If you don't have DFC, you can use the **Hailo Model Zoo** pre-compiled models or ask someone with DFC access to compile `ppe_best.onnx` for `hailo8l`.

## 2. Deploy to Raspberry Pi

### Step 2.1: Transfer Files

Copy the entire `rpi_hailo_python` folder and the compiled `ppe_best.hef` to the Pi.

```bash
scp -r rpi_hailo_python pi@<PI_IP_ADDRESS>:~/datacollector
scp models/ppe_best.hef pi@<PI_IP_ADDRESS>:~/datacollector/rpi_hailo_python/models/
```

### Step 2.2: Install Dependencies on Pi

SSH into the Pi and install requirements:

```bash
ssh pi@<PI_IP_ADDRESS>
cd ~/datacollector/rpi_hailo_python
pip install -r requirements.txt
# Ensure hailo-platform is installed (usually comes with the AI Kit software suite)
```

### Step 2.3: Update Configuration

Edit `config/config.yaml` on the Pi to point to the HEF file if needed (default is `models/ppe_best.hef`):

```yaml
inference:
  model_path: "models/ppe_best.hef"
```

### Step 2.4: Run

```bash
python -m src.main --config config/config.yaml
```
