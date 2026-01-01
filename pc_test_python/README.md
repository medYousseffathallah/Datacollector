# PC Webcam Testing - PPE Detection

This folder contains the PC testing version of the PPE Data Collector system.
It uses Ultralytics YOLOv8 to run inference on a webcam feed without requiring Hailo hardware.

## Prerequisites

1. Python 3.8+
2. A webcam connected to the PC.
3. Internet connection (for first-time model download if needed).

## Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This will install `ultralytics`, `opencv-python`, and other required packages.*

## Usage

Run the main script with the provided configuration:

```bash
python -m src.main --config config/config.yaml
```

## Configuration

The configuration file is located at `config/config.yaml`.
- **Model**: Default is `models/ppe_best.pt`.
- **Camera**: Default is webcam (ID `0`).
- **Classes**: Configured to detect ["Person", "Hardhat", "Safety Vest", etc.].

## Output

- **Logs**: Printed to console.
- **Dataset**: Saved to `dataset_pc_test/` (images and labels).
- **Database**: Metadata saved to `dataset_pc_test/datacollector_test.db`.

## Troubleshooting

- If the webcam is not detected, change the `url` in `config/config.yaml` to `1` or another index.
- If the model is not found, ensure `models/ppe_best.pt` exists or run `python scripts/download_ppe_model.py`.
