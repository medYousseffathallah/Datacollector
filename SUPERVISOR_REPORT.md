# Project Progress Report: Edge AI Data Collector for PPE Detection

**Date:** 2026-01-21
**Student:** [Your Name]
**Project:** Smart PPE Detection System using Raspberry Pi 5 & Hailo-8L

---

## 1. Overview

This report outlines the development of the **Data Collector Module**, a critical component of the Smart PPE Detection project. The Data Collector is an intelligent edge application designed to autonomously capture, filter, and store training data from video feeds (Webcam/RTSP) to create a custom dataset for high-precision model training.

## 2. How the Data Collector Was Created

The system was built using a modular Python architecture to ensure scalability and ease of maintenance. The development was divided into distinct components:

### A. Core Architecture

- **Language**: Python 3.11
- **Frameworks**: OpenCV (Vision), Ultralytics YOLOv8 (Inference), HailoRT (NPU Acceleration), SQLite (Metadata).

### B. Key Modules

1.  **Camera Manager (`camera_manager.py`)**:
    - Implemented multi-threaded video capture to handle high-FPS streams without blocking the main processing loop.
    - Supports both USB Webcams (for testing) and IP Cameras (RTSP) for site deployment.

2.  **Inference Engine (`inference_engine.py`)**:
    - **Dual-Mode Design**: Created a flexible engine that runs on:
      - **PC (Development)**: Uses `ultralytics` YOLOv8 for rapid prototyping and debugging.
      - **Raspberry Pi (Production)**: Uses `hailo_platform` to leverage the Hailo-8L NPU for real-time performance.
    - Solved compatibility issues between float32 masks and OpenCV's 8-bit requirements.

3.  **Intelligent Data Writer (`dataset_writer.py`)**:
    - Instead of saving every frame, the system uses "Smart Capture":
      - **Motion Detection**: Checks for pixel changes to ignore static scenes.
      - **Confidence Filtering**: Only saves images containing target classes (Person, Helmet, Vest) above a specific confidence threshold (e.g., >60%).
      - **Auto-Annotation**: Automatically saves bounding box labels in YOLO format alongside the images, significantly reducing manual labeling effort.

4.  **Visualization Dashboard (`visualize_data.py`)**:
    - Built a **Streamlit** web app to review collected data, visualize detection overlays, and monitor dataset statistics in real-time.

## 3. Integration into the Full Project: The EYE-D Ecosystem

The Data Collector is a distinct, specialized device designed to serve the larger **EYE-D** system.

### The Role of EYE-D
**EYE-D** is the project's main orchestrator and primary camera detector. It is responsible for real-time site monitoring, safety compliance enforcement, and high-level decision making.

### The Active Learning Pipeline
The Data Collector acts as the "training ground" for EYE-D, ensuring the main system evolves and adapts to new environments.

1.  **Data Collection (Edge Device)**: The Data Collector is deployed independently to capture site-specific imagery (e.g., specific uniforms, lighting conditions).
2.  **Annotation & Verification**: Collected data is offloaded and passes through a rigorous human-in-the-loop verification process to ensure label quality.
3.  **Fine-Tuning EYE-D**: The verified dataset is used to retrain the core models of the EYE-D orchestrator.
4.  **Deployment**: The updated, higher-accuracy models are pushed back to the EYE-D main units.

This loop solves the "domain shift" problem, ensuring EYE-D remains robust across different construction sites.

## 4. Model Reference

We are utilizing a YOLOv8-based model trained for Construction Site Safety.

- **Base Model Architecture**: YOLOv8 (Nano/Small)
- **Source Repository**: [Construction Site Safety PPE Detection](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection)
- **Classes Detected**: Person, Hardhat, Safety Vest, Machinery, Vehicle.

## 5. Project Timetable

| Phase      | Task                        | Status         | Description                                                                                                        |
| :--------- | :-------------------------- | :------------- | :----------------------------------------------------------------------------------------------------------------- |
| **Week 1** | **Architecture Design**     | ✅ Completed   | Defined system modules, data flow, and hardware requirements (RPi 5 + Hailo).                                      |
| **Week 2** | **Core Development**        | ✅ Completed   | Implemented Camera Manager, Inference Engine, and SQLite storage.                                                  |
| **Week 3** | **PC Testing & Validation** | ✅ Completed   | Tested with Webcam. Fixed OpenCV runtime errors. Split project into `PC` and `Hailo` folders for clean deployment. |
| **Week 4** | **Deployment to Edge**      | ⏳ In Progress | Deploying code to Raspberry Pi 5. Setting up systemd services for auto-start.                                      |
| **Week 5** | **Data Collection**         | 📅 Planned     | Running system on-site to collect ~500 diverse images.                                                             |
| **Week 6** | **Model Fine-Tuning**       | 📅 Planned     | Retraining the model on the new dataset and compiling to HEF (Hailo Executable Format).                            |

## 6. Next Steps

To move from the current prototype to the final production system, the following steps are required:

1.  **Hardware Deployment**:
    - Transfer the `rpi_hailo_python` folder to the Raspberry Pi 5.
    - Install Hailo drivers and dependencies.

2.  **Data Collection Campaign**:
    - Run the collector for 24-48 hours in the target environment.
    - Goal: Capture 500+ high-quality images of people with/without PPE.

3.  **Data Curation**:
    - Review the auto-generated annotations using the Streamlit dashboard or LabelImg.
    - Correct any mislabeled instances (Quality Assurance).

4.  **Training & Compilation**:
    - Train the YOLOv8 model on a GPU-enabled PC using the new dataset.
    - Use the **Hailo Dataflow Compiler (DFC)** to convert the trained `.pt` model into `.hef` format for the NPU.

5.  **Final Integration**:
    - Update the Raspberry Pi config to use the new custom-trained `.hef` model.
    - Enable the alerting system (GPIO/Email) for real-time safety violations.

---

_If you require further technical details regarding the code structure or the specific hyperparameters used for the thresholding logic, please let me know._
