# C++ High-Performance Implementation

If you encounter latency issues with the Python version, especially when scaling to many cameras or high frame rates, you can switch to this C++ implementation.

## Prerequisites

Ensure the following libraries are installed on your Raspberry Pi 5:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libopencv-dev libyaml-cpp-dev libsqlite3-dev
# HailoRT C++ headers and libs should already be installed via hailo-all
```

## Build Instructions

1.  **Navigate to the C++ directory**:
    ```bash
    cd rpi_hailo_cpp
    ```

2.  **Create a build directory**:
    ```bash
    mkdir build
    cd build
    ```

3.  **Configure and Compile**:
    ```bash
    cmake ..
    make -j4
    ```

## Usage

Run the compiled executable from the project root (so it can find `config/`):

```bash
# Go back to project root
cd ../../
./rpi_hailo_cpp/build/datacollector config/config.yaml
```

## Code Structure

-   `CameraManager`: Uses OpenCV C++ `VideoCapture` with `std::thread` for non-blocking capture.
-   `InferenceEngine`: Framework for HailoRT C++ API (currently set to Mock Mode for testing). To enable real inference, you need to uncomment the HailoRT specific code in `InferenceEngine.cpp` and link against `libhailort`.
-   `DatasetWriter`: Efficient filesystem writing and SQLite logging.
-   `Utils`: Optimized `cv::findContours` and `approxPolyDP` for mask-to-polygon conversion.

## Performance Notes

-   **Memory**: C++ uses significantly less RAM overhead than Python.
-   **Threading**: `std::thread` provides true parallelism, unlike Python's GIL-constrained threading.
-   **Latency**: Lower overhead in the main loop allows for tighter polling of camera buffers.
