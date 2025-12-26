# Edge AI Data Collector for YOLO Fine-tuning

This system captures, labels, and stores annotated segmentation frames from multiple CCTV cameras, optimized for Raspberry Pi 5 with Hailo AI Accelerator.

## Features

- **Multi-Camera Support**: Handles multiple RTSP streams with auto-reconnect.
- **Hailo Inference**: Real-time YOLO segmentation using HailoRT.
- **YOLO Format**: Saves data directly in YOLO segmentation format (images + .txt labels).
- **Edge Optimized**: Efficient threading and resource management.
- **Configurable**: YAML-based configuration for streams, models, and capture rules.

## Requirements

- Raspberry Pi 5 (8GB recommended)
- Hailo AI HAT (Hailo-8 / Hailo-8L)
- HailoRT installed (v4.17+)
- Python 3.9+

## Installation

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   _Note: `hailo_platform` must be installed via the HailoRT PCIe driver / Python package instructions._

2. **Configuration**
   Edit `config/config.yaml` to set your camera URLs and model path.

## Usage

Run the collector:

```bash
python -m src.main --config config/config.yaml
```

## Operations & Deployment

For hardware integration, troubleshooting, and production deployment (systemd, auto-start), please refer to **[OPERATIONS.md](OPERATIONS.md)**.

## Project Structure

- `config/`: Configuration files.
- `src/`: Source code.
  - `camera_manager.py`: RTSP stream handling.
  - `inference_engine.py`: HailoRT integration.
  - `dataset_writer.py`: File saving and DB logging.
- `dataset/`: Output directory for images/labels.

## Deliverables

- **Architecture**: Modular design with separate threads for capture, inference, and writing.
- **Code**: `src/inference_engine.py` contains the HailoRT integration logic.
- **Format**: Output follows standard YOLO segmentation format (normalized polygons).
