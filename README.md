# Edge AI Data Collector System

This project is a modular data collection system for edge AI devices. It captures video streams from CCTV cameras, performs real-time YOLO segmentation inference, and saves the data in YOLO format for fine-tuning.

## Supported Platforms

The system is architected into three independent implementations to support different hardware and performance requirements:

### 1. [Raspberry Pi Python Version](./rpi_hailo_python/)

- **Path**: `rpi_hailo_python/`
- **Hardware**: Raspberry Pi 5 + Hailo-8/8L AI HAT.
- **Language**: Python.
- **Use Case**: Standard deployment, easy to modify, rapid prototyping. Uses `hailo_platform` for inference.

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

---

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
