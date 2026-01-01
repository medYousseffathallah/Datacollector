# Edge AI Data Collector System

This project is a modular data collection system for edge AI devices. It captures video streams from CCTV cameras, performs real-time YOLO segmentation inference, and saves the data in YOLO format for fine-tuning.

## Supported Platforms

The system is architected into three independent implementations to support different hardware and performance requirements:

### 1. [Raspberry Pi Python Version](./rpi_hailo_python/)

- **Path**: `rpi_hailo_python/`
- **Hardware**: Raspberry Pi 5 + Hailo-8/8L AI HAT.
- **Language**: Python.
- **Use Case**: Standard deployment, easy to modify, rapid prototyping. Uses `hailo_platform` for inference.
- **New Features**:
  - **Motion Detection**: Reduces NPU load by skipping inference on static scenes.
  - **PC Testing**: Can run on Windows/Linux PC with webcam using Ultralytics fallback.
  - **PPE Support**: Configured for Person/Helmet/Vest detection.
  - **Data Visualization**: Integrated Streamlit dashboard for browsing dataset and metadata.

### 2. [Raspberry Pi C++ Version](./rpi_hailo_cpp/)

- **Path**: `rpi_hailo_cpp/`
- **Hardware**: Raspberry Pi 5 + Hailo-8/8L AI HAT.
- **Language**: C++.
- **Use Case**: High-performance, low-latency deployment. Ideal for many cameras or high frame rates where Python's GIL is a bottleneck.

### 3. [NVIDIA Jetson Version](./jetson_cuda_python/)

- **Path**: `jetson_cuda_python/`
- **Hardware**: NVIDIA Jetson Nano, NX, Orin.
- **Language**: Python.
- **Use Case**: Deployment on NVIDIA edge devices using CUDA and TensorRT (via Ultralytics).
- **New Features**:
  - **Motion Detection**: Integrated to optimize GPU usage.
  - **Webcam Support**: Enhanced compatibility with USB cameras (numeric indexing).

---

## Recent Updates

### Motion Detection & Resource Optimization

Both the Raspberry Pi (Python) and Jetson implementations now include a **Motion Detection** pre-processing step.

- **Logic**: Uses background subtraction to detect movement in the frame.
- **Benefit**: Inference is skipped for static frames, significantly reducing power consumption and thermal load on edge devices.
- **Configuration**: Adjustable threshold and minimum area in `config.yaml`.

### PC Testing & Cross-Platform Support

The Python implementations have been updated to support **Mock/PC Mode**.

- **Hardware Independence**: If the specific NPU/GPU hardware (Hailo/Jetson) is not found, the system falls back to:
  - **Ultralytics YOLO**: Runs on standard CPU/GPU if available.
  - **Mock Mode**: Generates dummy data if no inference engine is present.
- **Webcam Integration**: Improved support for USB webcams on PC (Windows/Linux) for easy testing and development before deploying to edge hardware.

### PPE Detection Integration

Support for Personal Protective Equipment (PPE) detection has been added.

- **Classes**: Detects `Person`, `Hardhat`, `Safety Vest` (and "NO-" variants).
- **Model**: Integrated download scripts for specialized PPE YOLO models.
- **Mapping**: Configuration files updated to map specific PPE class names for data collection.

### Data Visualization Dashboard

A new Streamlit-based visualization tool is available to inspect the collected dataset.

- **Features**:
  - **Dashboard**: View collection statistics (total frames, class distribution).
  - **Gallery**: Browse captured images with YOLO segmentation masks overlaid.
  - **Database View**: Execute SQL queries on the metadata database.
- **Usage**:
  ```bash
  cd rpi_hailo_python
  streamlit run scripts/visualize_data.py
  ```

## Directory Structure

```
project/
├── .vscode/                # VS Code Configuration
│   ├── mock_includes/      # Dummy headers (SQLite, OpenCV, Hailo) for Windows IntelliSense
│   └── c_cpp_properties.json # C++ IDE config
│
├── rpi_hailo_python/       # RPi 5 + Hailo (Python)
│   ├── src/                # Source code
│   ├── config/             # Configuration files
│   ├── deploy/             # Systemd service files
│   └── README.md           # Implementation details
│
├── rpi_hailo_cpp/          # RPi 5 + Hailo (C++)
│   ├── src/                # C++ Source code
│   ├── config/             # Configuration files
│   ├── build/              # Compile artifacts (after building)
│   └── README.md           # Implementation details
│
├── jetson_cuda_python/     # NVIDIA Jetson (Python)
│   ├── src/                # Source code (CUDA-optimized)
│   ├── config/             # Configuration files
│   └── README.md           # Implementation details
```

## Development vs. Deployment

### Windows Development Environment

This project includes **Mock Headers** in `.vscode/mock_includes/`. These allow you to view and edit C++ code on Windows without seeing "red squiggles" for missing Linux libraries (OpenCV, SQLite, HailoRT).

- **Do not remove** these files; they help VS Code understand the code.
- They are **ignored** during the actual build on the Raspberry Pi.

### Getting Started

Navigate to the folder matching your hardware and follow the `README.md` inside.
